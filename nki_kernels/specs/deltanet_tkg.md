# DeltaNet TKG — Conv + Recurrence Implementation Notes

Design rationale for `deltanet/components/conv.py` and `deltanet/components/recurrence.py`, plus the
input contracts. The kernels themselves carry only the layout and math; everything here is the
"why", extracted so the source stays readable.

Status: **IMPLEMENTED & on-device validated.** Value-head sharded, LNC=2.

---

## 1. Input contracts

### Recurrence (`gated_delta_rule_tkg`), all f32, d = 128 = P_MAX

| Tensor | Shape | Notes |
|---|---|---|
| `q`, `k` | (Hk, T, 128) | raw `silu(conv)`; l2-normed over d in-kernel, q also scaled by 1/sqrt(d) |
| `v` | (Hv, T, 128) | raw `silu(conv)` |
| `a`, `b` | (T, Hv) | raw `in_proj_a` / `in_proj_b`, token-major, head on free |
| `A_log` | (Hv,) | per-head decay param |
| `dt_bias` | (Hv,) | per-head decay bias |
| `init_state` | (Hv, 128, 128) | carried recurrent state |

### Conv (`deltanet_conv`)

| Tensor | Shape | Notes |
|---|---|---|
| `qkv` | (T, conv_dim) | raw `in_proj_qkv` output, token-major, `cat(q,k,v)` on the free axis |
| `conv_state` | (conv_dim, K-1) | carried window, from `conv_state_buffer` |
| `conv_weight` | (conv_dim, K) | per-channel taps, no bias |
| `key_dim` | python int | q/k segment width; `Hk = key_dim//128`, `value_dim = conv_dim - 2*key_dim` |

`conv_dim % 128 == 0`; `K-1 == conv_state` width; output dtype follows `qkv.dtype`. Channels are
laid out `[ q:0..key_dim | k:key_dim..2*key_dim | v:2*key_dim.. ]`.

---

## 2. Per-token recurrence math

Identical to `NeuronGatedDeltaNet._recurrent_step`:

```
decay   Sp = src * exp(g)              per-head scalar, free-broadcast across j
read    kv = sum_i Sp[i,:] * k_h[i]    partition-reduce
delta   d  = (v - kv) * beta
update  Sp[i,:] += k_h[i] * d
output  o  = sum_i Sp[i,:] * q_h[i]    partition-reduce
```

Folded-in input glue, so the caller does no transposes/reshapes/replication:

- l2norm of q/k over d, then `q *= 1/sqrt(d)`
- `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)`, then `exp(g)`
- GQA head replication as an access-pattern broadcast (no data copy): value-head `h` reads k-head
  `h//rep`. This equals the global mapping because `Hv = rep*Hk` holds per core.

---

## 3. Recurrence implementation rationale

- **State double-buffering (S0/S1 ping-pong).** Token `t`'s output reads its working tile while
  token `t+1`'s out-of-place decay writes the other, which breaks the inter-token
  write-after-read on the state.
- **Hoisted gating.** The decay scalar is partition-broadcast once for the whole block (width
  `T*Hv`) and free-broadcast across `j`; the per-token operand is just an AP at offset `t*Hv`.
  `exp(g)` for every (head, token) is computed once before the loop — one activation-table load.
- **beta folded into the key rows.** `k*beta` is built for all tokens up front, so the per-token
  delta step collapses to a single subtract.
- **Rank-1 matmul update.** One matmul per value-head (key row stationary, delta segment moving),
  so the outer product needs no partition broadcast of delta and no Vector product tile.
- **Paired read, one state pass.** The GQA group's key/query column pair is the stationary operand
  and its state columns the moving tile, so the elementwise multiply happens inside the PE array
  (no `[128, W]` Vector product) and both reads come out of one pass, landing on PSUM partitions 0
  (key) and 1 (query). A stream shuffle moves the query read to partition 0 on the output branch —
  compute accesses must start on a quadrant boundary — which keeps it off the state chain.
- **Closed-form output.** The output takes its read on the PRE-update state and adds the update's
  rank-1 term in closed form, `(q·kbeta)*delta`, so it never makes a second pass over the state.
  That term collapses to the per-head scalar `a_{t,h}`.
- **PSUM residency.** Partition broadcast/reduce results stay in PSUM and are consumed directly.
- **q/k load path.** Loaded once up front via a free-axis contiguous bulk DMA, l2-normed over d,
  then a single on-chip `nc_transpose` each, landing dim on the 128 partitions the reduce
  contraction needs. Only `Hk` copies live in SBUF.

### Constraints

| Constraint | Reason |
|---|---|
| `Hk*T <= 128` | `nc_transpose` output partitions. Tile the token axis for larger blocks. |
| `rep*dim <= 512` | The paired read is one matmul per GQA group, so a group's state columns must fit one PSUM bank (512 f32). |
| `Hv % n == 0`, `Hk % n == 0` | Whole heads per core. |
| `Hv % rep == 0` | Whole GQA groups per core. |

### v-bridge gather (SBUF path)

`v_row` is gathered per head, not as one strided DMA: a multi-dim SBUF AP can only stride whole
partitions at free-dim granularity, and the conv tile's head partitions are interleaved by token
(stride `T`), so the gather isn't expressible as a single multi-partition AP.

The fp32 working buffer in `_load_normed_qk` is required even on the SBUF path, because gen3
`nc_transpose` needs `dst dtype == input dtype` and the recurrence matmuls consume fp32 q/k (the
conv tile is bf16). The conv tile itself is left unmutated.

---

## 4. Conv implementation rationale

- **Channel-on-partition compute layout.** Per-channel taps, the carried state window, and the
  candidate windows move through HBM as bulk contiguous DMAs onto NT partitions, then get
  transposed into the compute layout with TensorE. `qkv` uses a strided DMA.
- **The MAC.** One windowed `tensor_tensor(multiply)` (sliding-window image AP, filter broadcast
  over the T outputs) feeding a `tensor_reduce(add)` over the K-tap axis — all T output columns at
  once. Accumulates in fp32; SiLU runs in fp32; outputs cast to the I/O dtype.
- **SiLU fused with the head_dim transpose.** `nc_transpose` moves head_dim off the partition axis
  onto the contiguous free axis (head/token on partition), and the SiLU activation doubles as the
  PSUM→SBUF cast. This lets the store write q/k/v as a contiguous bulk DMA instead of a per-element
  transposed store.
- **Bit-exact state.** The candidate columns are exact copies of the raw projection output, so the
  carried state window is bit-exact.
- **Per-segment buffer sizing.** Buffers are allocated at each segment's exact tile count, because
  `nc_transpose` requires the data-AP partition stride to equal the tensor free dim — they can't be
  over-sized and shared.
- **Separate q/k/v output tiles.** Keeping them separate (rather than one combined `out_sbuf`)
  leaves every tile partition-0-based, which is exactly the layout the recurrence's
  `_load_normed_qk` consumes (q/k) and the SBUF→SBUF v bridge gathers (v). The Activation engine
  cannot target a partition offset, so a combined buffer would need an extra placement DMA.
- **Deferred conv-state stores.** `conv_qkv_sbuf` returns the scatter as a pending list. The stores
  read only the conv windows, so draining them after the recurrence keeps their transposes off the
  conv → recurrence hand-off. They MUST be drained via `conv_state_store_pending` or the conv state
  is never written.

---

## 5. Value-head sharding

SPMD broadcasts the full HBM tensors to every core. Core `c` of `n` computes only its
`Hv = Hv_full//n` value-heads and writes a disjoint slice of every full-shape output. The per-token
math body is fully parametric in the LOCAL head counts; the only per-core differences are the AP
offsets threaded into each HBM load/store. `nl.num_programs`/`nl.program_id` fold to trace-time
ints, so all the shard math runs at trace time and drives the AP offsets directly.

`n=1` reduces to a single core owning all heads at offset 0.

For the conv, each core owns a contiguous slice of the q segment, the k segment, and the v segment
(3 channel sub-ranges), so the q/k/v tiles a core's downstream recurrence heads consume are all
produced locally. Channels are independent (depthwise), so cores never interact:
`qkv_out`/`conv_cand` are bit-identical to a single-core run, only the producing core differs.
