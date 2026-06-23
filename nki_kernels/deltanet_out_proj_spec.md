# DeltaNet Output Projection (+ in-kernel TP all-reduce) — TKG Kernel Spec & Plan

Status: **PROPOSED — not yet implemented.**
Scope: the last fusion seam of the DeltaNet attention layer — everything in `attn_output()`
(`modeling_qwen36_a3b.py:466-480`) *after* the gated per-head RMSNorm, i.e. **`out_proj` + the
RowParallelLinear TP all-reduce**. Folding this into the existing fused conv→recurrence→gated-norm
kernel (`nki_deltanet_fused_tkg.py`) makes the whole DeltaNet attention block one SBUF-resident NKI
launch that returns the *final* hidden-shaped output (post-projection, post-reduce). After this, the
only host op left in the attention layer is the residual add.

Reuse: **`nkilib.core.output_projection.output_projection_tkg` does the matmul**; **`nki.collectives.all_reduce`
does the TP reduce** (the `transformer_tkg` idiom, `transformer_tkg.py:62-90,379,425`). This is the plan
already sketched in the megakernel scoping doc (`megakernel_scoping.html:306,314`: "out_proj … D=128 fits
… add TP all-reduce" + "TP=4 all-reduce … `nki.collectives.all_reduce` … proven in `transformer_tkg`").

Prerequisite: the gated norm must switch to **Layout B** (head_dim on the partition axis) — see
`deltanet_gated_rmsnorm_spec.md` §3, which explicitly defers Layout B to "once `out_proj` is also
in-kernel." That time is now.

---

## 1. What this kernel computes

After the gated per-head RMSNorm emits `gated [T, value_dim]` (head-major), the model does
(`modeling_qwen36_a3b.py:480`, `722`):

```python
output = self.out_proj(gated)          # RowParallelLinear: matmul + cross-rank all-reduce
```

`out_proj` is `RowParallelLinear(global_value_dim, hidden_size, input_is_parallel=True)`
(`modeling_qwen36_a3b.py:373`). So two ops:

1. **Matmul** `o_partial[t, h] = Σ_v gated[t, v] · W[v, h]`, contracting over the rank's local
   `value_dim` channels. `W` (per rank) is the input-sharded `[hidden_size, value_dim]` nn.Linear
   weight; the kernel wants its **transpose** `[value_dim, hidden_size]` (same convention as the
   in_proj `proj_w`, see `nki_deltanet_in_proj.py:8`).
2. **TP all-reduce** (sum) of `o_partial` across the `tp_degree` ranks — because each rank holds only
   `1/tp_degree` of the global value heads. This is the only collective in the DeltaNet layer.

Output is the final attention-block output `o_out [T, hidden_size]` (full hidden, fully reduced).

### Concrete dims (A3B, TP=4, LNC=2)
Ground-truth from `config.json` `text_config` (per `deltanet_gated_rmsnorm_spec.md` §2; the modeling-file
comment numbers `1536`/`48-head` are stale): `hidden_size H = 2048`, `linear_num_value_heads = 32`,
`linear_value_head_dim d = 128`.

| Quantity | Global | Per rank (TP=4) | Per core (LNC=2) |
|---|---|---|---|
| value heads `Hv` | 32 | **8** | **4** (`Hv_core`) |
| `value_dim = Hv·d` | 4096 | **1024** | 512 (`W_core`) |
| hidden `H` | 2048 | 2048 | **1024** (`H_shard = H/LNC`) |
| `out_proj` weight (transposed `[value_dim, H]`) | — | **[1024, 2048]** | weight H-sharded → [1024, 1024] |
| tokens `T` | — | 1 (decode) / 2 (verify) | same |

Write the kernel **parameterized** (`Hv`, `value_dim`, `H`, `d`); the numbers above are illustration.
Constraints check (all pass): `d = 128 ≤ P_MAX` ✓; `B·S = T ≤ 128` ✓; `H % LNC == 0` ✓ (2048 % 2).

---

## 2. The central design decision — sharding reconciliation

The recurrence + gated norm are **value-head sharded** across the 2 LNC cores: core `c` owns heads
`[c·Hv_core, (c+1)·Hv_core)` and emits the gated output for *only its 4 heads*
(`nki_deltanet_tkg.py:299` `vh_off`, `col_off`). The `out_proj` matmul **contracts over all value
heads**. So the o_proj cannot read straight off the per-core norm output — the contraction needs heads
the core doesn't own. Two ways to bridge:

### Design A — H-shard the o_proj (RECOMMENDED, reuses `output_projection_tkg`)
`output_projection_tkg` with LNC>1 **shards the output `H` across cores** and requires the **full
contraction (all `Hv` heads) on each core** (`output_projection_tkg.py:34-37,284-286`). So:
1. **Gather** each core's 4 gated heads → all 8 heads on both cores (a *tiny* cross-LNC `sendrecv`,
   §5). This moves the cross-core traffic to the small *attention* tensor, pre-matmul.
2. `output_projection_tkg` → each core produces a **disjoint `H_shard = H/2 = 1024`** rank-partial.
3. `nki.collectives.all_reduce` over the TP group, per core, on its `[T, H_shard]` partial.
4. Each core stores its reduced `[T, H_shard]` to a disjoint HBM slice → complete reduced `[T, H]`.

Cross-core exchange = the gather of `[d, B, Hv_core, T]` bf16 ≈ **1 KB** at T=1. The all-reduce runs on
the **half-size** `[T, 1024]` shard and needs **no LNC reduce** (shards are disjoint in `H`). This is
exactly `transformer_tkg`'s o_proj/down_proj pattern.

### Design B — contraction-shard the o_proj (rejected)
Each core matmuls its *own* 4 heads against its weight rows → a **full-`H` partial** `[T, 2048]`;
then **sum the two cores' partials** (LNC reduce of the full hidden via `sendrecv`+add) → rank-partial;
then TP all-reduce. No pre-gather, but: (a) it **cannot reuse `output_projection_tkg`'s LNC path** (that
path shards on H, not the contraction — it reads `program_id` and would mis-shard), so it's a custom
matmul; (b) the cross-core reduce is on the *full* `[T, 2048]` output (2× the data of Design A's gather)
and the all-reduce is on full `[T, 2048]` not the half-shard. Same matmul FLOPs, strictly more traffic,
no library reuse. **Reject.**

**Decision: Design A.** Same `program_id`=core keys both shardings consistently within the `[2]` launch:
core 0 = recurrence heads 0-3 → o_proj H-shard `[0:1024]`; core 1 = heads 4-7 → H-shard `[1024:2048]`;
the gather bridges the two for the contraction.

---

## 3. nkilib reuse — `output_projection_tkg` arg mapping

`output_projection_tkg(attention, weight, bias=None, quantization_type=NONE, …, OUT_IN_SB, sbm)`
(`output_projection_tkg.py:69`). For DeltaNet (no bias, no quant, float path):

| kernel arg | DeltaNet value | notes |
|---|---|---|
| `attention` | `[D=d=128, B=1, N=Hv=8, S=T]` in **SBUF** | head_dim on partition; all heads (post-gather). `_load_attn_to_sbuf` passes SBUF through unchanged (`:1006`). |
| `weight` | `[N·D = value_dim = 1024, H = 2048]` in HBM | transpose of `out_proj.weight` (`[H, value_dim]`). Sharded on `H` by `program_id` internally (`:284-286`). |
| `bias` | `None` | DeltaNet `out_proj` has `bias=False`. |
| `quantization_type` | `QuantizationType.NONE` | float path; `weight.dtype` = io dtype. |
| `TRANSPOSE_OUT` | `False` | want `[B·S, H]` = `[T, H]`. |
| `OUT_IN_SB` | `True` | keep the rank-partial in SBUF for the in-place all-reduce (no HBM round-trip). |
| `sbm` | the megakernel's `BufferManager` | compose under one allocator scope. |

With `OUT_IN_SB=True, TRANSPOSE_OUT=False` it returns `out_sb [min(T,P_MAX), H_shard] = [T, 1024]`
(`:297-303`) — exactly what the all-reduce consumes. Head packing is a no-op here: `d=128=P_MAX` →
`group_size=1`, so it issues `Hv=8` accumulating `nc_matmul`s of 128-deep contraction (`:727-731`).

**Validate this composes as a sub-call.** `output_projection_tkg` is `@nki.jit`; `in_proj_compose`
already nests `@nki.jit qkv_tkg` inside the fused kernel and works (`nki_deltanet_in_proj.py:23`), so the
pattern is proven — but `transformer_tkg.py:91` warns about "double-jit stack overflow." If nesting
fails, call the inner `_output_projection_tkg_impl` / shuffle helpers directly, or strip the decorator at
the call site. Resolve at implementation time (§9).

---

## 4. Prerequisite: switch the gated norm to Layout B (head_dim on partition)

`output_projection_tkg` wants `attention` as `[D, B, N, S]` — **head_dim on the partition axis**. The
current gated norm emits **Layout A** (`head_dim on the free axis`, `[T, W]` head-major,
`nki_deltanet_norm_gate.py:27-60`). `deltanet_gated_rmsnorm_spec.md` §3 anticipated this exact moment:

> "Layout B pays off only once `out_proj` is also in-kernel … head_dim-on-partition feeds the matmul
> transpose-free. Until then it is pure added transpose cost."

So add a **Layout B variant** of `norm_gate_row` that produces `[d=128, Hv_core·T]` (head_dim on
partition, `(head, t)` on free):
- sum-of-squares = **partition-axis reduce via a ones-`nc_matmul`** (`stationary=ones[128,1]`,
  `moving=x²[128, Hv_core·T]` → `[1, Hv_core·T]` PSUM) — the technique the recurrence already has
  (`nki_deltanet_tkg.py` `partition_broadcast_psum`/`ones_row`/`red_p`) and `rmsnorm_tkg.py:267` uses.
- `rsqrt(mean+eps)` broadcast back over the 128 partitions; `gamma [128,1]` per-partition multiply;
  `silu(z)` gate (z also transposed to head_dim-on-partition).
- Output `[d, Hv_core·T]` → reshape to `[d, B, Hv_core, T]` = the o_proj `attention` slice for this
  core's heads, **transpose-free** into the gather+matmul.

This replaces the Layout-A write at the recurrence output seam (`nki_deltanet_tkg.py:491-507`) **only on
the o_proj-fused path**; keep Layout A for the standalone gated-norm and the non-o_proj fused entrypoints
(they still emit `attn_out [T, W]` head-major). Gate it behind the same plumbing flag that turns o_proj on.

---

## 5. The LNC head-gather bridge

After the Layout-B norm, core `c` holds `attn_loc [d=128, B, Hv_core=4, T]` (its heads). Each core needs
`attn_full [d=128, B, Hv=8, T]` for the contraction. Two-step, all SBUF:
1. Copy local heads into the core's slot: `attn_full[:, :, c·Hv_core : (c+1)·Hv_core, :] = attn_loc`.
2. `nisa.sendrecv(src=attn_loc, dst=attn_full[:, :, other·Hv_core : …, :], send_to_rank=other,
   recv_from_rank=other, pipe_id=…)` with `other = 1 - c` — the cross-LNC exchange, identical in form to
   `transformer_tkg.py:76-83`. Optionally `nisa.core_barrier(cores=(0,1))` to order it.

Size: `[128, 1, 4, T]` bf16 ≈ 1 KB (T=1). Negligible vs the matmul.

---

## 6. The TP all-reduce

Per core, on its rank-partial `out_sb [T, H_shard]`:

```python
import nki.collectives as nccl
rg = nccl.ReplicaGroup(replica_groups)              # TP group, built host-side (§7)
out_reduced = nl.ndarray(out_sb.shape, dtype=..., buffer=nl.sbuf)
nccl.all_reduce(dsts=[out_reduced], srcs=[out_sb], op=nl.add, replica_group=rg)
```

`all_reduce(srcs, dsts, replica_group, op, priority)` (verified signature). SBUF tensors are supported
(doc: "Tensors can reside on either HBM or SBUF") — keep it in SBUF (smaller, no round-trip), matching
`transformer_tkg.py:67`. Both cores call `all_reduce` with the **same `rg`** on their **own** disjoint
H-shard; the runtime handles per-core. Then DMA each reduced shard to its disjoint HBM slice of
`o_out [T, H]`. **No LNC gather of the result** — the shards are disjoint in `H`, so the HBM tensor is
complete and downstream (`residual + attn_out`) reads it normally.

---

## 7. Kernel API & entrypoints

Add a thin composable + new entrypoints (mirroring the in_proj precedent: helper in a dedicated file,
composition in the fused file).

```python
# nki_deltanet_out_proj.py — thin reuse wrapper (+ standalone test harness)
def out_proj_compose(attn_full_sb, out_w, replica_groups, sbm):
    """attn_full_sb [d,B,Hv,T] SBUF (all heads, head_dim on partition), out_w [value_dim,H] HBM
    -> reduced o_out shard [T, H_shard] SBUF. Calls output_projection_tkg(OUT_IN_SB) then nccl.all_reduce."""

@nki.jit
def deltanet_out_proj_fwd(attn_full, out_w, replica_groups):
    """Standalone HBM harness for unit testing: attn_full [d,B,Hv,T] -> o_out [T,H]. Launch [2]."""
```

Then extend the fused composes/entrypoints in `nki_deltanet_fused_tkg.py`. The full attention megakernel
(in_proj → conv → recurrence → gated-norm(Layout B) → gather → o_proj → all-reduce):

```python
@nki.jit
def deltanet_in_proj_out_fused_tkg_fwd(hidden, proj_w, gamma, eps, conv_state, conv_weight, key_dim,
                                       A_log, dt_bias, init_state, z_gamma, out_w, replica_groups, z_eps=1e-6):
    """Decode/commit (T=1): whole DeltaNet attention block in one launch.
    Returns (o_out [T, H], final_state [Hv,128,128], new_conv_state [conv_dim,K-1]). Launch [2]."""

@nki.jit
def deltanet_in_proj_out_fused_tkg_fwd_state(...):
    """Verify (T>=2): returns (o_out [T,H], candidate_states [T,Hv,128,128], conv_cand [T,conv_dim,K-1])."""
```

**Output-contract change:** the entrypoints now return the *projected, reduced* `o_out [T, H]` instead of
the raw/gated `attn_out [T, value_dim]`. The candidate_states / conv_cand outputs (verify) are **unchanged**
— o_proj only touches the attention *output*, not the recurrence state. Keep the existing
non-o_proj entrypoints (`deltanet_in_proj_fused_tkg_fwd`, …) for incremental bring-up and A/B.

---

## 8. Model integration (`modeling_qwen36_a3b.py`)

1. **Weight:** pass `self.out_proj.weight.t()` (`[value_dim, H]`) into the kernel, same transpose
   convention as `proj_w` (`nki_deltanet_in_proj.py:8`). Consider pre-transposing in
   `convert_qwen36_a3b_hf_to_neuron_state_dict()` to avoid a per-call `.t()`.
2. **Replica group:** build `replica_groups: List[List[int]]` from
   `parallel_state.get_tensor_model_parallel_group()` (the TP ranks), passed to the kernel as a
   trace-time arg — the form `transformer_tkg` takes (`transformer_tkg.py:110,195`).
3. **Bypass RowParallelLinear's reduce:** the kernel now does the all-reduce, so `out_proj` must NOT
   reduce again. Either set `out_proj.reduce_output=False` / `input_is_parallel` accordingly, or wrap the
   call to no-op `reduce_from_tensor_model_parallel_region` exactly like `_defer_moe_allreduce()`
   (`modeling_qwen36_a3b.py:1971-1982`). The shared-expert (`down_proj` `reduce_output=False`,
   `:1994,2034`) is the precedent.
4. **Collapse `attn_output()`** (`:466-480`): on the kernel path it becomes a pass-through — the kernel
   returns `o_out` directly. `_forward_decode_tkg` (`:547`) and `verify_block` (`:652`) drop the
   `self.attn_output(...)` call; `fused_attn_nki` (`:433`) returns `o_out` in place of `attn_raw`.
5. Keep the PyTorch norm+gate+out_proj path behind `use_tkg_attention_kernel` for one A/B cycle; delete
   after parity holds.

---

## 9. Validation plan

- **Standalone first** (`deltanet_out_proj_fwd`, single rank, no all-reduce): feed random
  `attn_full [d,B,Hv,T]`, compare `attn_full @ out_w` to torch `gated @ out_proj.weight.T` (no reduce).
  Validates the matmul + layout + head-gather in isolation. `T ∈ {1,2}`, `cores=1` then `[2]`.
- **Multi-rank all-reduce:** run TP=4, compare the kernel `o_out` to the full `RowParallelLinear`
  output (matmul + `reduce_from_tensor_model_parallel_region`). This needs a multi-rank harness — reuse
  the `transformer_tkg` collective test scaffold if present, else a 4-rank `torchrun` driver.
- **Fused end-to-end:** `deltanet_in_proj_out_fused_tkg_fwd` vs the current
  `deltanet_in_proj_fused_tkg_fwd` + torch `attn_output` (norm+gate+out_proj+reduce). Decode (T=1) and
  verify (T=2); assert candidate_states/conv_cand bit-match the existing kernel (untouched path).
- **`cores=1` vs `cores=2`** bit-identical cross-check (the existing `test_stage4_shard_bit_identical`
  pattern, `test_deltanet_fused_tkg_kernel.py:181`) — but note the all-reduce makes cores=1 vs cores=2
  differ in *sharding* not *result*; compare the assembled full `[T,H]`, not per-core shards.
- **dtype/parity:** bf16 io, f32 matmul accumulate; the norm's bf16-round-first parity (gated-norm spec
  §1) is unchanged. Watch all-reduce summation order vs torch (float-add non-associativity) — expect
  tight, not bit-exact, agreement across ranks; validate with atol/rtol + cosine-sim.
- New test `tests/test_deltanet_out_proj_kernel.py` (standalone) + extend
  `tests/test_deltanet_in_proj_fused_kernel.py` for the full megakernel; profile with a sibling of
  `tests/profile_in_proj_fused_kernel.py`.

---

## 10. Optimizations (preserved / chosen)

- **Reuse `output_projection_tkg`** — gets the tuned `[D,B,N,S]` shuffle, head packing, weight blocking/
  budgeting (`output_projection_tkg.py:1120`), and PSUM pipelining for free. Don't re-derive the matmul.
- **Layout B → transpose-free matmul:** head_dim-on-partition norm output feeds the o_proj with no
  transpose (the whole point of doing it in-kernel; gated-norm spec §3).
- **Cross-core traffic on the small tensor:** gather the `~1 KB` attention pre-matmul (Design A), not the
  `[T, H]` output post-matmul (Design B).
- **SBUF-resident all-reduce** on the half-size `[T, H_shard]` shard; single DMA store to HBM after.
- **One collective for the whole layer:** the DeltaNet block now has exactly one TP all-reduce (vs the
  MoE block's deferred-fused single reduce, `:1971`); no extra LNC reduce (disjoint H-shards).

---

## 11. Open questions / risks

1. **Nesting `@nki.jit output_projection_tkg`** inside the fused kernel — proven for `qkv_tkg`, but
   confirm no double-jit overflow (`transformer_tkg.py:91`). Fallback: call its `_impl` helpers directly.
2. **Replica-group plumbing into an `@nki.jit` kernel** — confirm how NxDI exposes the TP group as the
   `List[List[int]]` the kernel traces against, and that LNC=2 + TP=4 collectives compose (each core
   reducing its own H-shard over the same group). `transformer_tkg` is the reference; verify it runs in
   *this* model's launch config, not just simulation.
3. **`all_reduce` on SBUF with LNC=2** — both cores issuing `all_reduce` on disjoint SBUF shards over the
   same `rg`; confirm correctness/ordering (may need `core_barrier`). The scoping doc flags the reduce as
   ⚠ ("the 6 nkilib `collectives/` files are gather/transpose, not reduce" — only `all_reduce` itself is
   the reduce, and it's used in `transformer_tkg`).
4. **Bypassing RowParallelLinear's reduce cleanly** — `_defer_moe_allreduce`-style monkeypatch vs a
   constructor flag; pick the one that doesn't disturb the non-kernel fallback path.
5. **Float-add non-associativity** across the in-kernel all-reduce vs torch's reduce — set the accuracy
   gate to atol/rtol, not bit-exact, for the reduced output.
6. **Decode vs verify weight reuse:** `out_w` is identical for both entrypoints; ensure it's loaded once
   per launch, not per-token (T-loop), and that the verify T=2 path doesn't duplicate the matmul.
