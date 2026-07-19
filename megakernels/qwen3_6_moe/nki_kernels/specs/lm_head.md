# LM Head — SBUF-Resident TKG Building Block (Spec & Plan)

Status: **IMPLEMENTED — device-validated, not yet wired into the model.**
Scope: the shared tail of both Qwen3.6-A3B heads — final RMSNorm → vocab-parallel `lm_head`
matmul → per-rank greedy argmax. Today this runs in PyTorch at two call sites that differ only in
`T`: the verify pass (`modeling_qwen36_a3b.py:4279-4285`, `_verify_epilogue`, `T=2`) and the MTP
draft step (`modeling_qwen36_a3b.py:2477`, `draft_step`, `T=1`). Goal: one composable that runs
norm → matmul → argmax **SBUF-resident with zero HBM round-trip between stages**, standalone
isolation-testable now, and later a drop-in for the megakernel epilogue.

Reuse: `nkilib.core.subkernels.rmsnorm_tkg` (via `nki_kernels/common.py:26` `rmsnorm_to_sbuf`) does
100% of the norm math; `nkilib.core.output_projection.output_projection_tkg` does 100% of the
matmul, blocking and weight streaming. Hand-written: one layout permutation (§3) and the staged
argmax (§4). **No cross-rank collective** — the composable stops at the per-rank `(max, index)`
pair, per the repo rule at `moe/components/moe_layer.py:16-19`.

**Out of scope (deliberately):** the cross-rank argmax, which belongs next to
`all_reduce_gather_h` / `all_reduce_gather_tokens` in
`megakernel/qwen36_verify_megakernel.py`; megakernel wiring; model wiring.

---

## 0. Ground-truth facts verified

- **Dims.** `H=2048`, `vocab_global=248320`, `TP=4` → `V_rank=62080`, `LNC=2` → `V_core=31040`,
  `T ∈ {1,2}`, `H0=128`, `H1=16`, `eps=1e-6`. `248320 / 4 = 62080` exactly ⇒ `lm_head` pad_size is
  **0**, so `vocab_valid_len` masking is a no-op here (§1).
- **Weights are BF16 `[2048, 62080]`.** Both `target_model.lm_head.weight` and
  `draft_model.mtp_head.mtp_lm_head.weight`, verified in
  `/home/ubuntu/models/qwen36_a3b_fused_selective/weights/tp0_sharded_checkpoint.safetensors`.
- **Weight layout is already contraction-first.** `TransposedColumnParallelLinear`
  (`modeling_qwen36_a3b.py:209-238`) stores `[input_size, output_size_per_partition]` = `[H, V_rank]`,
  which is exactly `output_projection_tkg`'s documented `weight [N*D, H_out]` shape
  (`output_projection_tkg.py:102-104`). **No repack, no transpose** — consumed verbatim.
- **Precision is bf16 end-to-end, and that is correct.** The model runs
  `torch_dtype=torch.bfloat16` and `TransposedColumnParallelLinear.forward` is a bare
  `torch.matmul` with no cast; the `.float()` calls happen *after* the matmul and recover nothing.
  `output_projection_tkg` sets `io_dtype = attention.dtype`
  (`output_projection_tkg.py:498`, `:780`), so a bf16 `normed` yields bf16 logits — matching the
  model exactly. **Do not upcast to fp32**: that would make the kernel *more* accurate than the
  oracle it is gated against and would produce false failures on legitimate bf16 ties.
- **Tie-break.** `torch.argmax` returns the **lowest** index. `cascaded_max._grouped_reduce_max`
  (`cascaded_max.py:329`) reduces masked indices with `nl.maximum` ⇒ **highest** index. Our
  reduction must therefore be built to return the lowest (§4).

---

## 1. Contract

```python
def lm_head_compose(hidden, gamma, lm_head_w, eps=1e-6, vocab_valid_len=None,
                    return_logits=False, name_prefix=""):
    """-> (rank_max [T,1] SBUF, rank_idx [T,1] int32 SBUF)   -- index is within V_rank."""
```

| name | shape | dtype | buffer / notes |
|---|---|---|---|
| `hidden` | `[H0=128, T, H1]` **or** `[B, S, H]` | bf16 / fp32 | SBUF (megakernel tp2013 residual) **or** HBM. Auto-detected via `hidden.buffer == nl.sbuf`, the `common.py:54` idiom — **no flag**. Left untouched (it is the residual). |
| `gamma` | `[1, H]` | bf16 / fp32 | HBM. `model.norm.weight` in **standard form**; the `(1+w)` fold happens once at checkpoint load (`modeling_qwen36_a3b.py:3571-3574`) so there is **no `+1` in-kernel**. |
| `lm_head_w` | `[H, V_rank]` | bf16 | HBM, consumed **verbatim**. |
| `vocab_valid_len` | scalar / `None` | — | Replaces `mask_padded_logits`. Expected no-op (pad_size 0); asserted, not implemented (§7.4). |
| **`rank_max`** | `[T, 1]` | = `hidden.dtype` | SBUF. Per-rank max logit. |
| **`rank_idx`** | `[T, 1]` | int32 | SBUF. Index **within `V_rank`**; the caller adds `rank_id * V_rank`. |

`return_logits=True` additionally returns `logits_sb [T, V_core]` SBUF (this core's vocab shard) so
the isolation test can gate P1. The two-tuple is the production contract.

House-convention twin:

```python
@nki.jit
def lm_head_fwd(hidden, gamma, lm_head_w, eps=1e-6):
    """-> (logits [T, V_rank], rank_max [n_prgs, T], rank_idx [n_prgs, T]) all nl.shared_hbm."""
```

Pattern: `moe/components/moe_layer.py:174` (compose) vs `:251-285` (fwd). Results are emitted
per-core (`[n_prgs, T]`) rather than as a single `[T,1]` so the two cores never write the same HBM
address; the test asserts the rows agree, which is a free check that the LNC combine (§5) worked.

---

## 2. Stage dataflow

```
hidden [B,S,H] HBM  |  [H0,T,H1] SBUF (tp2013 residual)
   │  rmsnorm_to_sbuf(gamma, eps, single_core_forced=False)      ← NORM ONCE
   ▼
normed_sb [H0=128, T, H1=16] SBUF, tp2013 column order
   │  head_major_from_tp2013()                                   ← §3, the one custom layout step
   ▼
attn_sb [D=128, B=1, N=H1, S=T] SBUF   (weight row n*D+d ≡ hidden column)
   │  output_projection_tkg(lm_head_w, TRANSPOSE_OUT=False, OUT_IN_SB=True, NONE)
   ▼                                    LNC-shards the OUTPUT (vocab) dim: V_core = V_rank/n_prgs
logits_sb [T, V_core] SBUF, bf16
   │  staged chunked max  →  core_max [T,1]                      ← §4
   │  reversed-index masked max  →  core_idx [T,1] (lowest index wins)
   ▼
   │  LNC combine over the 2 core-local pairs (sendrecv, §5)
   ▼
rank_max [T,1], rank_idx [T,1] int32      ← STOP. Cross-rank argmax is the megakernel's job.
```

`output_projection_tkg` shards `H_out` (= the vocab) across logical cores by design
(`output_projection_tkg.py:36-38, 286, 734`), which gives **free vocab-parallelism with no
collective** — `V_core = 31040` per core. It requires `V_rank % n_prgs == 0`
(`output_projection_tkg.py:686-689`); `62080 % 2 = 0` ✓.

---

## 3. ⚠ The one real design finding: tp2013 and `output_projection_tkg` disagree on column order

**This contradicts the brief's "you need a free-axis reorder to `[D=H0, B=1, N=H1, S=T]`
(partition axis unchanged)". A free-axis reorder is provably insufficient.** Evidence:

- `rmsnorm_tkg` with `hidden_dim_tp=False` (what `rmsnorm_to_sbuf` uses) loads
  `(BxS, H) → reshape_dim(1, [num_H_shards, H0, H2]) → permute([2,0,1,3])`
  (`norm_tkg_utils.py:388,395`), i.e. output `[h0, t, s*H2+h2]` ↔ **hidden column
  `s·H0·H2 + h0·H2 + h2`**. The kernel docstring pseudocode says the same
  (`rmsnorm_tkg.py:95-101`), and the repo's own `sigma_gate_compose` weight AP
  `pattern=[[H2, H0], [1, H2]], offset=s*H0*H2` (`moe/components/moe_layer.py:79`) confirms it:
  **the partition axis `h0` has hidden-stride `H2`, not 1.**
- `output_projection_tkg` indexes its weight as `[n*D + d, h]` (`output_projection_tkg.py:104`),
  i.e. the partition axis `d` has **hidden-stride 1**.

`H2 = H1/n_s ∈ {8, 16}` and never equals `D = 128`, so no permutation of the *free* axes can
reconcile them. Passing `normed_sb` through with only a free-axis reorder silently computes
`logits = W_permuted^T · x` — a wrong answer that still has the right shape.

Three resolutions were considered:

| option | verdict |
|---|---|
| `D = H2, N = n_s·H0` (exact match with only an `h0↔h2` transpose) | ✗ `D ∈ {8,16}` ⇒ `d_original % 32 != 0` ⇒ no head packing (`output_projection_tkg.py:727-731`), `n_size` = 128/256, matmul contraction `K = 8` (6% PE use) and `_shuffle_attn` emits `static_range(n_size)` copies (`:1083`). ~16× the matmul instruction count. |
| `rmsnorm_tkg(hidden_dim_tp=True)` — emits column `h1·128 + h0`, an exact zero-cost match (`norm_tkg_utils.py:365-374`) | ✗ as the *only* path: it forces `num_H_shards == 1` (`rmsnorm_tkg.py:663`) and, decisively, it cannot help the SBUF-in case at all — with SBUF input `rmsnorm_tkg` passes the caller's column order straight through (`rmsnorm_tkg.py:672-673`), so a tp2013 megakernel residual stays tp2013 regardless. |
| **`head_major_from_tp2013()` — an explicit permutation (chosen)** | ✓ Keeps the settled contract (`rmsnorm_to_sbuf`, `single_core_forced=False`, weight verbatim, `D=128`/`N=16` efficient matmul) and works **identically for the HBM and SBUF-in paths**. |

### The permutation

With `H2 = H1/n_s` and `G = H0/H2`, write `h0 = a·G + b` (`a ∈ [0,H2)`, `b ∈ [0,G)`). Then

```
s·H0·H2 + h0·H2 + h2  =  s·H0·H2 + a·H0 + (b·H2 + h2)
```

so the target `[d, n]` decomposition `column = n·128 + d` is **exactly**

```
attn[b·H2 + h2, 0, s·H2 + a, t]  =  normed_sb[a·G + b, t, s·H2 + h2]
```

`h2` moves free→partition and `a` moves partition→free. **No single primitive expresses this**, and
three plausible ones were each rejected by the compiler in turn:

| attempt | rejected by |
|---|---|
| `normed_sb.ap(pattern=[[G*T*H1, H2], [1, H2]], ...)` — strided partition read | `ap() pattern has invalid partition stride. Partition step 512 must equal tensor free dimension size 32` — compute access patterns are lane-locked to a partition stride of one |
| one bulk DMA with the partition dim split via `TensorView.reshape_dim(0, ...)` | `partition dim cannot be reshaped` |
| `TensorView(...).permute([1,0])` to transpose in the view | `Partition dimension stay the outermost dimension` |

What works is two moves:
1. one `nisa.nc_transpose` per `(s, t)` of the **contiguous** `[H0, H2]` block → `st[h2, h0]`,
   putting `h2` on the partition axis (dtype-matched PSUM tile, the repo's own idiom —
   `gqa/decode/fused_layer.py:86`, `deltanet/components/out_proj.py:74-76`);
2. one DMA per `(b, s, t)` of `st[0:H2, b::G]` into partitions `[b*H2, (b+1)*H2)` — a free-axis
   stride plus a partition **base offset**, which the DMA engine handles.

`n_s·T` transposes plus `G·n_s·T ≤ 64` small DMAs — negligible against a ~127 MB/core weight stream.

---

## 4. Staged argmax (lowest-index tie-break)

`nisa` reduce chunks are capped at `2^14` elements per DVE instruction
(`rotational_topk_utils.py:549`), so a `V_core = 31040` reduce must be staged:
`n_chunks = div_ceil(V_core, 16384)`, `C = div_ceil(V_core, n_chunks)` → 2 chunks of 15520 at
LNC=2, 4 chunks of 15520 at LNC=1.

**Pass 1 (value).** Per chunk `tensor_reduce(op=nl.maximum)` → `chunk_max [T, n_chunks]`; one more
`tensor_reduce` → `core_max [T,1]`.

**Pass 2 (index), reversed-index trick.** `nisa.nc_find_index8` reports the **first** positions
matching a given value, so *within* a chunk the lowest index is free. *Across* chunks, define
`r = V_core − g` for global index `g`, so `r ∈ [1, V_core]` and **maximising `r` minimises `g`** —
a plain `nl.maximum` reduce then reproduces `torch.argmax`'s lowest-index tie-break, and "no match
in this chunk" falls out as `r = 0`. Per chunk `c`:

```
ind    = nc_find_index8(logits_chunk, vals=core_max broadcast to 8)   # [T,8]; ind[:,0] = first hit
hit    = (chunk_max[:,c] == core_max)        # ind is meaningless unless this chunk holds the max
ind_f  = float(ind[:,0])                     # MUST widen before float arithmetic (see §7.7)
best_r = max(best_r, hit * (V_core − c·C − ind_f))
```

Finally `core_idx = V_core − best_r`, plus this core's vocab-shard base `prg_id · V_core`
(`output_projection_tkg.py:286` shards `[p·V_core, (p+1)·V_core)`). No `[T, C]` scratch is needed
anywhere, which matters: at fp32 the logits tile alone is already near the SBUF per-partition limit
(§7.6).

Because the equality test is against a max taken over the *same* tile, it is exact in bf16 — no
tolerance is involved anywhere in the reduction.

**`cascaded_max` is deliberately not used as an entrypoint** (§6): only its reduction *shape* is
borrowed. Our logits already live in SBUF, so the whole `predicated_folded_load` /
fold-into-partitions machinery is unnecessary — the staged reduce above operates on
`logits_sb [T, V_core]` in place and needs no runtime-indexed `TensorView.select`.

---

## 5. LNC combine

Each core owns a disjoint vocab shard, so a core-local pair is not yet the per-rank answer. The
`(max, idx)` pair is packed into **one contiguous fp32 `[T, 2]` tile** and exchanged with a single
`nisa.sendrecv`, then both cores run the same reduction:

```
peak      = max(self_max, peer_max)
r_self    = (self_max == peak) * (V_rank - self_idx)         # reversed index again
r_peer    = (peer_max == peak) * (V_rank - peer_idx)
rank_idx  = V_rank - max(r_self, r_peer)
```

Reusing `r = V_rank − idx` makes the choice **symmetric** — no dependence on which core is which —
and resolves a cross-shard tie to the lower vocab index, exactly as `torch.argmax` does. Packing the
max as fp32 keeps the equality test exact, since both cores widen the same bf16 value identically.

**The destination must be a contiguous whole tile.** Exchanging into a strided `[T,1]` column of a
`[T,2]` tile compiled cleanly and silently delivered nothing (both cores then reported their own
shard-local max). Every `sendrecv` in the repo exchanges a contiguous tile
(`gqa/components/out_proj.py:123`, `moe/components/shared_expert.py:172`,
`megakernel/qwen36_verify_megakernel.py:84`) — that is the reason.

`sendrecv` is an **intra-core-pair LNC primitive, not a TP collective** — the "no `nki.collectives`"
rule (`moe/components/moe_layer.py:16-19`) is respected. At `n_prgs == 1` the whole block is
skipped.

---

## 6. nkilib reuse table

| `file:line` | what | verdict | caller adds |
|---|---|---|---|
| `nki_kernels/common.py:26` `rmsnorm_to_sbuf` (→ `core/subkernels/rmsnorm_tkg.py:45`) | final RMSNorm, fp32 internal | **✓ drop-in** | `single_core_forced=False` (tp2013), matching `moe/components/routed_experts.py:48`. Needs a new **optional** `sbm=` passthrough (§7.1). |
| `core/output_projection/output_projection_tkg.py:69` | the `[T,H]·[H,V]` matmul, H-blocking, weight streaming, LNC vocab shard | **⚠ adapt** | `TRANSPOSE_OUT=False`, `OUT_IN_SB=True`, `QuantizationType.NONE`, **a manual `BufferManager`** (§7.1) and the §3 permutation. |
| `core/max/cascaded_max.py:288` `_grouped_reduce_max` | reduction *shape* (mask → masked-index → reduce) | **⚠ adapt** | reduce with the reversed index so the **lowest** index wins (`:329` uses `nl.maximum` on raw indices ⇒ highest). Reimplemented on SBUF tiles; ~10 lines. |
| `core/max/cascaded_max.py:31` `cascaded_max` (as an entrypoint) | — | **✗ do not use** | its loader `predicated_folded_load` is **HBM-only** (`cascaded_max_utils.py:58`) and it shards over `BxS`, forcing LNC1 whenever `BxS ≤ 2` (`cascaded_max.py:147-149`) — exactly our regime, which would throw away the vocab parallelism §2 relies on. Our logits are already in SBUF. |
| `experimental/output_projection/output_projection_tkg_primitives.py` | — | **✗ do not use** | `all_weights_preloaded` is referenced **undefined** at `:219`, which fires precisely on the multi-block large-vocab path we are on; and `OUT_IN_SB` is silently ignored at `:143`. |
| `output_projection_tkg(TRANSPOSE_OUT=True)` | — | **✗ do not use** | enforces `MAX_VALIDATED_N_TIMES_H_SIZE = 163840` on `N*H` (`output_projection_tkg.py:64, 722-725`); ours is `16 × 62080 ≈ 10⁶`. It also preloads all weights at once. |
| `nisa.nc_find_index8` | first positions matching a value | **✓ drop-in** | one call per chunk on `logits_sb`, with `core_max` broadcast over the 8 candidate slots via `core_max.ap([[1,T],[0,8]])` (the `cascaded_max.py:258` idiom). Its "first hit" semantics give the lowest within-chunk index for free. Its `uint32` output **must** be widened to fp32 before any float arithmetic (§7.7). |
| `nisa.max8` | 8-wide local max | **✗ not needed** | it solves the *fold-into-partitions* shape `cascaded_max` creates. On `[T, V_core]` a staged `tensor_reduce` is simpler and fewer instructions. |
| `nki.collectives` | — | **✗ banned here** | repo rule, `moe/components/moe_layer.py:16-19`. The cross-rank argmax lives in the megakernel. |

---

## 7. Open questions / risks (honest)

1. **Auto-allocation cannot handle vocab scale — a manual `BufferManager` is mandatory.**
   `_budget_weight_blocks` short-circuits under auto-alloc (`output_projection_tkg.py:1158-1161`)
   and returns `num_w_h_blocks = len(sizes)`, making `all_weights_preloaded` True
   (`:1330`). At `h_sharded = 31040` that is 16 blocks × 16 heads × `[128, 2048]` bf16 ≈ **134 MB**
   of SBUF against a ~24 MB budget. Only the manual path
   (`BufferManager(..., use_auto_alloc=False)`) runs the real free-space budget and the circular
   weight buffer. Consequence: **every** SBUF tile in this composable must be allocated from that
   same manager, or a compiler-placed `nl.ndarray` may overlap a manually-placed one. That is why
   `rmsnorm_to_sbuf` needs an optional `sbm=` passthrough (default `None` ⇒ unchanged behaviour for
   every existing caller). **This is the highest-risk item in the design.**
2. **The §3 permutation is the contract deviation.** It is additive (nothing in the settled
   contract is weakened; the weight stays verbatim, the norm stays tp2013) but it *is* a step the
   brief did not anticipate — the brief's "free-axis reorder, partition axis unchanged" is provably
   insufficient. Settled by measurement: with the permutation in place P1 passes at every
   dtype/T/cores point.
3. **bf16 tie legitimacy.** At `V_rank = 62080` bf16 logits, exact ties in the top-2 are not rare.
   P2b therefore treats a token mismatch against the PyTorch path as a bug **only** when the top-2
   logits differ by more than the measured bf16 floor. P2a (kernel tokens vs argmax of the
   *kernel's own* logits) is the gate that actually tests the reduction and is exact.
4. **`vocab_valid_len` is asserted, not implemented.** `pad_size = 0` for this checkpoint (§0), so
   the argument exists purely to keep the signature stable for a future padded vocab. If a padded
   vocab ever appears, the fix is to clamp the last chunk's `mask` — the §4 formula already handles
   partial chunks.
5. **`T` is capped at `P_MAX`.** `output_projection_tkg` asserts `B*S ≤ 128` when
   `OUT_IN_SB=True` (`output_projection_tkg.py:692-696`). Fine at `T ≤ 2`.
6. **`OUT_IN_SB=True` caps the vocab that fits in SBUF — the binding constraint.** The manual
   budget allocates `out_sb` (`output_projection_tkg.py:297`) and *then* requires
   `bytes_per_h_block + out_sb_bytes <= free_space` (`:1187`), so `out_sb` is charged **twice**:
   the composable needs roughly `2·V_core·sizeof(io_dtype) + min_block_bytes` in the managed
   region. Block sizes are also only tried by repeated halving from `h_sharded` down to `F_MAX`
   (`:1170-1179`), so the smallest candidate is often far above 512 and the fit can miss by a
   little. Measured consequence at `V_rank = 62080`: **bf16 on 2 cores fits and is the only
   full-vocab configuration that does** — which is exactly the production configuration (bf16,
   LNC=2), but it means the fp32 gate has to run at a reduced vocab (§8). Raising
   `SBM_SIZE_BYTES` helps only marginally; claiming all of SBUF fails outright because the backend
   needs its own scratch.
7. **`nc_find_index8` returns `uint32`; float arithmetic on it must be widened first.** Doing
   `tensor_scalar(op0=multiply, operand0=-1.0)` straight on the integer tile evaluates in the
   integer dtype and wraps. `cascaded_max.py:276` widens with an explicit `tensor_copy` first, and
   so does this kernel. Cost of getting it wrong: a silently wrong token, only in chunks ≥ 1.
8. **No cosine gate anywhere.** `specs/out_proj.md:257` recommends one — that is **stale** and
   violates the repo-wide ban; it is not copied here. `tests/test_gqa_out_proj_kernel.py` predates
   the ban likewise.

---

## 8. Isolation test & gates

`tests/test_lm_head_kernel.py`, template `tests/test_gqa_pre_attn_rmsnorm_kernel.py`. Constants
`ATOL=1e-5`, `RTOL=1e-2`, `BF16_HEADROOM=3.0`. CPU oracle in fp32 on inputs **first rounded to the
IO dtype**. Real device run via `torch_xla` — no simulation, no `nki.baremetal`.

| gate | compares |
|---|---|
| **P1 fp32** | kernel logits vs fp32 CPU oracle, `allclose(atol=1e-5, rtol=1e-2)` — **HARD** |
| **P1 bf16** | kernel logits vs the measured bf16 quantization floor × 3.0 (floor measured on the SAME oracle tensor) |
| **P2a** | kernel tokens vs `argmax` of the **kernel's own** logits — exact int32 match. This is what actually tests the reduction. |
| **P2b** | kernel tokens vs the PyTorch `lm_head(norm(h))` + `argmax` path — exact, **except** a mismatch is only a bug if the top-2 logits differ by more than the bf16 floor (otherwise it is a legitimate tie) |

A fifth check, free from the `[n_prgs, T]` output shape: **both cores must report the same
`(max, index)`**, which is what catches a silently-dropped `sendrecv` (§5).

Because of §7.6 the matrix runs at two vocab widths. `V_rank = 62080` covers the production
configuration (bf16, LNC=2, the only one that fits SBUF at full width); `SMALL_VOCAB = 20480`
covers the **complete** dtype × `T ∈ {1,2}` × cores ∈ {1,2} matrix including the fp32 hard gate, and
still yields a multi-chunk reduce at LNC=1 (`V_core = 20480 > 2^14`). Only the vocab width changes;
no tolerance is relaxed and the kernel is parameterized on `(H, V_rank, T)`.

### Measured (trn3pre, LNC=2)

All 10 configurations pass all four gates.

| case | P1 | P2a | P2b |
|---|---|---|---|
| full/bf16/T1/c2 | max_abs 1.330e-2 ≤ 3.850e-2 (floor 1.283e-2) | exact `[38668]` | exact |
| full/bf16/T2/c2 | max_abs 1.642e-2 ≤ 4.450e-2 (floor 1.483e-2) | exact `[7063, 17316]` | exact |
| small/fp32/T1/c1 | max_abs 1.431e-5, max_rel 4.023e-3 | exact `[5564]` | exact |
| small/fp32/T2/c1 | max_abs 1.812e-5, max_rel 4.515e-3 | exact `[15035, 10399]` | exact |
| small/fp32/T1/c2 | max_abs 5.221e-5, max_rel 4.038e-3 | exact `[16256]` | exact |
| small/fp32/T2/c2 | max_abs 1.636e-4, max_rel 3.465e-3 | exact `[20456, 4057]` | exact |
| small/bf16/T1/c1 | max_abs 1.128e-2 ≤ 2.336e-2 (floor 7.788e-3) | exact `[11166]` | exact |
| small/bf16/T2/c1 | max_abs 1.276e-2 ≤ 3.829e-2 (floor 1.276e-2) | exact `[7374, 18313]` | exact |
| small/bf16/T1/c2 | max_abs 1.192e-2 ≤ 3.576e-2 (floor 1.192e-2) | exact `[11238]` | **tie**: torch picks 14983, top-2 margin 4.3e-3 < floor 1.2e-2 |
| small/bf16/T2/c2 | max_abs 1.356e-2 ≤ 4.069e-2 (floor 1.356e-2) | exact `[4492, 306]` | exact |

The fp32 P1 numbers clear `allclose` on the `rtol` term (`max_rel ≈ 4e-3 < 1e-2`); `max_abs` alone
exceeds `atol=1e-5` at the larger logit magnitudes, which is expected for a 2048-deep fp32
contraction reassociated against torch's. The one P2b tie is the designed-for case and shows the
tie-classification path is genuinely exercised, not vacuous.

Run:

```
NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
python -m models.qwen3_6_moe.tests.test_lm_head_kernel
```

(single-core cases use `NEURON_RT_VISIBLE_CORES=2`; the box has 8 physical / 4 logical cores at
LNC=2, so pin per job.)
