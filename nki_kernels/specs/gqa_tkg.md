# GQA TKG (head_dim=256) — Tensor Layouts & Implementation Notes

Layout contracts and design rationale for `nki_kernels/gqa/`. The kernels carry the math and the
essential contract; the full tensor layouts and the "why" live here.

Status: **IMPLEMENTED & on-device validated.** TP=4, LNC=2, `head_dim=256`, `q_heads=4`,
`kv_heads=1` per rank.

---

## 1. Why head_dim is tiled on the partition axis

nkilib's TKG attention caps `head_dim` at 128, because `D` sits on the SBUF/PSUM partition axis and
the PE array limits contraction-K, stationary-free-M, and partition all to <= 128.

`head_dim=256` is handled by tiling `D` into `D_TILES = ceil(D/128) = 2` partition tiles, applied at
exactly two matmul sites:

- **QK^T** (contraction over `D`) — split-K: the `D_TILES` partial products PSUM-accumulate into one
  scores tile.
- **P.V** (`D` is the stationary-free, so it becomes the output partition) — `D_TILES` stationary
  V-slices produce `D_TILES` PSUM halves `[128, Tq]`, one per head_dim tile.

Softmax reduces over the KV-length axis, which is on the free axis, so it is untouched by the
tiling.

The live path (`components/attention.py`) is a shim over the **vendored** `attention_tkg`: the
decode QK^T / online-softmax / P.V compute path is byte-for-byte AWS code, and only the
head_dim-on-partition layout sites are tiled. The banner in `vendored/attention_tkg.py` marks the
patch sites.

`components/attention_fresh_ref.py` is the earlier from-scratch core, kept for cross-checking. It
is **not** on the perf-critical path — the perf-critical attention math is AWS-authored.

---

## 2. Attention tensor layouts

`D_TILES = ceil(d_head/128)`.

| Tensor | Shape | Buffer | Indexing |
|---|---|---|---|
| `q_sb` | `[128, D_TILES, B*H*s_active]` | SBUF io_type | `q_sb[d_in, dt, b*H*s_active + h*s_active + s] = Q[dt*128+d_in, b, h, s]` |
| `k_active_sb` | `[128, D_TILES, B*s_active]` | SBUF io_type | `[d_in, dt, b*s_active + s]` |
| `k_prior` | `[B, 1, d_head, L]` | HBM io_type | already transposed; flat-KV, `tp_k_prior=False` |
| `v_prior` | `[B, 1, L, d_head]` | HBM io_type | |
| `v_active` | `[B, 1, s_active, d_head]` | HBM io_type | or SBUF `[s_active, d_head]` when `v_in_sb` and `bs == 1` |
| `mask` | `[L, B, H, s_active]` | HBM uint8 | 1=keep, s_prior-major (linear) |
| `out_sb` | `[128, D_TILES, B*H*s_active]` | SBUF | written in place (`out_in_sb=True`) |

`k_prior`'s last `s_active` s_prior slots are overwritten by `k_active` inside the kernel.

### Input contract

- **Q arrives pre-scaled** by `1/sqrt(d_head)` and already RoPE'd / qk-normed. The vendored kernel
  applies the `1/sqrt(d)` scale internally *only* when `fuse_rope=True`; here `fuse_rope=False`, so
  the caller owns scaling, RoPE and norm. K is RoPE'd / normed but **not** scaled. The attention
  composable does not touch Q/K/V values.
- The caller supplies the full attention **mask**; `use_pos_id=False`, so the kernel generates no
  causality itself.
- `curr_sprior` (== full KV length `L`, prior + active) **must be a multiple of 128**, and a
  multiple of **256** when `s_prior` is sharded across 2 cores (so `L >= 256` under LNC2).
- The active tokens occupy the **last** `s_active` slots of the `L`-length KV.

Precision: bf16 IO, fp32 matmul/softmax accumulate.

### Scores layout choice (fresh-ref core)

Scores are produced as `[Tq, L]` — token*head on partition, KV-length on free — so softmax is a
plain free-axis reduce: no partition-axis reduction, no transpose-for-max, no cross-core
online-softmax combine. The cost is one PE transpose of the probabilities `[Tq, L] -> [L, Tq]`
before P.V, which is cheap because `Tq <= 8`. This is the minimal correct shape for a tiny decode
width with a contiguous KV of length `L`.

---

## 3. qk_norm — why it is a free-axis reduce

The QKV tile is `[T, N, D]` in SBUF: `T = B*S` tokens on the **partition** axis; on the **free**
axis, `N` heads head-major (order `[q0..q_{Q-1} | k0..k_{K-1} | v0..v_{K-1}]`) then `head_dim D`
contiguous within each head.

Because `D` sits entirely on the free axis and `D=256 <= 512` (the free limit), RMSNorm over `D` is
a single free-axis reduction per token — no partition tiling, no splitting of `head_dim`. This is
precisely why the norm is applied in this layout rather than as a head_dim-on-partition RMSNorm,
which would cap `d` at 128.

Math per normed head (`gamma` is the layernorm weight, **not** `1+weight`):

```
y[t, :] = x[t, :] * rsqrt(mean_D(x[t, :]^2) + eps) * gamma
```

bf16 (or fp32) IO; the square, reduction, rsqrt and scale all run in fp32.

SBUF-in / SBUF-out: the projection's `[B*S, I]` SBUF result is the *same buffer* viewed as
`[T, N, D]` (`I = N*D`, head-major), so a caller passes it straight through with no copy.

---

## 4. o_proj sub-head ordering

`output_projection_tkg` folds `N` sub-heads, each at most 128 wide, and PSUM-accumulates them. A
`head_dim=256` q-head is therefore presented as **two** 128-wide sub-heads, so `q_heads=4` gives
`4*2 = 8` sub-heads of 128 — structurally identical to the DeltaNet o_proj (8 value-heads of 128).
The GQA composable *wraps* `output_projection_tkg`; it does not patch it.

Because the attention core's output already has `head_dim` on the partition axis (split into
`D_TILES` tiles of 128), **no per-head PE transpose is needed** — unlike the DeltaNet o_proj. Each
`(q-head, d-tile)` pair is one 128-wide sub-head already sitting on the partition axis. The
composable only reorders the free axes into `attention [d=128, B=1, N, T]`, gathers the other LNC
core's sub-heads via `sendrecv`, then runs the H-sharded matmul.

**Sub-head ordering is q-head major, d-tile minor:** global sub-head
`n = h_global * D_TILES + d_tile` maps to value_dim block `[n*128, (n+1)*128)`. The weight
`out_w [value_dim, hidden]` is row-indexed by value_dim, so `out_w[n*128:(n+1)*128]` is exactly
sub-head `n`'s weight — it already matches `output_projection_tkg`'s `weight [N*D, H]` indexing
`[n*D + d, h]`, and is passed through unreshaped.

The TP all-reduce of the per-rank o_proj partial is deferred; the composable returns the per-rank
partial `[T, hidden]`.

---

## 5. pre_attn_norm sharding

`num_H_shards` defaults to `lnc` (2 at LNC=2, from the launch grid). This lays the `H1=16` output
columns out as `[shard0_H2(8) | shard1_H2(8)]` — the byte-for-byte column order `qkv_tkg`'s
`NO_NORM` path slices per shard.

At `T = B*S <= SHARDING_THRESHOLD` (18) both cores compute the full replicated norm — no BxS shard,
no `sendrecv` — matching the unsharded runtime `input_layernorm`.

`gamma` is `input_layernorm.weight` in **standard** form: the `(1+w)` conversion is applied once at
checkpoint load, so `gamma` is fed directly with no `+1` in-kernel.

---

## 6. Fused layer — KV cache write modes

The fused layer always returns the post-norm/RoPE **active K/V** (`active_k` BHDS, `active_v` BHSD),
which is what NxDI's `update_kv_by_layer_id` scatters into the caches when
`k_cache_transposed=True`. This is the default mode and needs no `kv_write_idx`.

When `kv_write_idx` is given, the kernel *additionally* scatters the active K/V into the caches
**in place** at `[idx : idx+T]`, and returns the mutated `(k_cache, v_cache)` handles so callers can
observe or alias the write. The first three returns are unchanged, so the default mode keeps working.

The in-place scatter reuses the nkilib indirect-DMA primitive (`nisa.dma_copy` with `scalar_offset`
on the runtime write-start slot and `indirect_dim` on the cache's L axis) from
`attention_block_tkg._update_flat_cache`:

- **K** is BHDS `[B,1,D,L]` — head_dim on partition, so the write is **tiled over `D_TILES`** with
  L-axis stride 1.
- **V** is BHSD `[B,1,L,D]` — token on partition, L-axis stride `HEAD_DIM`, so a single DMA.

It must run **after** the attention read, since attention takes prior context from the caches.

Under LNC2 both cores hold replicated active K/V, so **V is written on prg 0 and K on prg 1** — that
gating is what makes the `[2]` launch write each cache exactly once instead of twice.
