# Trn3 Optimization Matrix — Phase 1

**Date**: 2026-04-21
**Phase 0 data anchor**: `kernels/moe_fused_tkg/quantized/OPTIMIZATION_LOG_TRN3.md` §"Trn3 Ground Truth — 2026-04-21" and §"Phase 0 — Harness blockers"
**Scoring formula** (per user):
```
Δ_score = 2 × (current_speedup / current_time_ms) × Δμs_per_token × NKI_FLOP_Ratio
        = 2 × (1.00 / 8.88 ms) × Δμs × 0.80
        = 0.180 × Δμs_per_token
```
All constants (`current_speedup=1.00`, `current_time_ms=8.88 ms`, `NKI_FLOP_Ratio=0.80`, `gap_μs=25`, `AR_μs=12`, `post_transformer_μs=300`) are **PLACEHOLDERS** until `[Z.1]` lands or `main.py --mode generate_accuracy_baselines --platform-target trn3` is run. Every Δscore is a **relative ranking**, not an achievable absolute score delta.

---

## Prerequisite — NOT in the ranked matrix

### `[Z.1]` Make qwen_complete compile e2e on trn3 (measurement infrastructure)

**Out-of-band infrastructure.** Handled in Phase 2 as the implicit Round 1 candidate #1 regardless of ranking.

- **Why**: no `qwen_*` module end-to-end compiles on trn3 today (Phase 0 Block 4: `kernel_v19b` fails NKI BIR emission). Without this, no e2e measurement exists, so 10 of 16 matrix rows below cannot be validated.
- **Concrete scope**: swap `kernel_v19b.qwen3_moe_fused_tkg` → `kernels.moe_fused_tkg.quantized.v14a.qwen3_moe_fused_tkg` inside `qwen_complete.py:641`; reconcile signatures / weight layouts; preserve the Phase 0 harness fixes (`block_size=8192`, router `platform_target=` strip, `kernel_v19b` import path).
- **Deliverable**: `python3 main.py --mode evaluate_single --platform-target trn3 --qwen qwen_complete --prompt <P0>` produces real TTFT, tok/s, `NKI_FLOP_Ratio`, `score`.
- **Implementation cost**: ~1 day of integration + validation.
- **Accuracy risk**: medium — v14a has different weight layout / quantization scales than v19b; state-dict conversion must be verified against the reference implementation. Run `run_accuracy_check` with tight tolerance on 1-2 prompts before benching.
- **Does not count in the ranking**; it unblocks the ranking.

---

## Ranked matrix (descending by Δscore)

| # | Strategy | Target kernel | Mechanism | Δμs/tok | Δscore | Risk | Accuracy risk | Blocker to check | Requires [Z.1]? | Tier | Impl cost |
|---:|:---|:---|:---|---:|---:|:---|:---|:---|:---:|:---|:---|
| 1 | **[G]** Cross-layer expert prefetch | `v14a` MoE (per-layer) | Speculative top-K prediction (Fate, arXiv 2502.12224, 97% top-8 accuracy) for layer N+1 from layer N's gate input; double-buffered SBUF ring; DMA during attention of next layer | **2438** | **439** | high | low | ring-buffer SBUF budget (~2× weight tile overhead); predictor mispredict cost ≤ 3% of layer time at 97% accuracy | **Y** | Load-bearing | 3-5 days |
| 2 | **[H]** Multi-layer fused TKG megakernel | All 48 decoder layers | Single NKI invocation; SBUF-resident residual; in-place KV cache write; move `input_layernorm` into kernel. Removes 48× Python/NxDI round-trip per token | **1175** | **211** | high | medium | SBUF budget across 48 layers (KV cache + residual); compiler scheduler fragility (Phase 0 trn2 history: v6c/v12g/v12h/v12k all plausibly-theory-positive regressions); CLAUDE.md `docs/multilayer_fused_tkg_plan.md` | **Y** | Load-bearing | 5-10 days |
| 3 | **[Z.2]** Integrated v14a + v13bc NKI TKG baseline | `qwen_complete` TKG path | Shipped NKI TKG (v10e attn → v13bc, v19b MoE → v14a). Captures the full kernel-upgrade gain over the reference integration | **~600** (est.) | **~108** | medium | medium | signature / weight-layout conversion for v13bc (tile-transposed) and v14a (fp8 pre-quant) | **Y** | Load-bearing | 2-3 days |
| 4 | **[F2]** ⭐ Direct DGE for expert weight loads | `v14a` MoE (per-layer) | Precompute per-expert base offsets once into SBUF; switch `nisa.dma_copy(..., dge_mode=0, scalar_offset=expert_id, indirect_dim=0)` → `dge_mode=1` (static stride). **This is the only top-5 lever grounded in the measured Phase 0 bottleneck** (GpSimdE stalls = 13.4 μs of 88 μs = 15% of v14a time) | **480** | **86** | medium | none | whether trn3 NKI compiler accepts `dge_mode=1` when scalar_offset is SBUF-resident (trn2 had compiler block per memory note); tile size regressions from precompute overhead | **N** | Load-bearing | 1-2 days |
| 5 | **[K]** Symmetric `_sb2sb_all_reduce_gather` on post-attn AR | `transformer_qwen.py:145` | Replace bare `nccl.all_reduce` (trn2 ENC_ALG_MESH workaround) with symmetric sb2sb collective. Retest on trn3 | **288** | **52** | low | none | `nccl` compatibility on trn3 — the workaround may no longer be needed | **Y** | Cleanup | 0.5 day |
| 6 | **[X1]** `nisa.exponential` in attention softmax | `v13bc` attention | Replace `nisa.activation(op=nl.exp)` → `nisa.exponential(..., max_value=row_max)`. 4× VectorE throughput on trn3. Phase 0 NTFF shows VectorE is 50.3% of v13bc (highest engine) | **288** | **52** | low | low | `nisa.exponential` numerical drift (rtol=1e-3 should hold, verify) | **N** | Cleanup | 2-4 hours |
| 7 | **[F3]** Pack `gate_up` + scales into single DMA per expert | `v14a` MoE (per-layer) | Co-locate `gate_up_w[e, :, :]` with `gate_up_scales[e, :]` in HBM so one DMA per (expert, tile) replaces two. Directly attacks GpSimdE descriptor count on the same root cause as [F2] | **288** | **52** | high | low | HBM weight re-layout requires touching the weight-loader path (`qwen_complete.py:252` MoE conversion). Integration risk — ask before touching unauthorized files | **N** (kernel-only verification) / **Y** (full integration) | Load-bearing | 2-3 days |
| 8 | **[F4]** Scale preloading retry | `v14a` MoE (prologue) | Load all 128 experts' `gate_up_scales` (~192 KB) + `down_scales` (~96 KB) into SBUF once per forward pass; per-expert load becomes SBUF copy, eliminating 2 of 5 indirect DMAs. Trn2 blocked by `NCC_INLA001` (SBUF dynamic indexing). Recheck on trn3 | **240** | **43** | low | none | `NCC_INLA001` compiler error on trn3 (may be lifted); SBUF +288 KB permanently resident across 48 layers = 13.8 MB total, within the 32 MiB budget | **N** | Cleanup | 0.5 day (5-min recheck first, then half-day impl if unblocked) |
| 9 | **[E]** LM head as NKI kernel (fused + on-device argmax) | New kernel, post-transformer | Replace stock LM head matmul [H=2048 × V=151936] + torch argmax with a fused NKI kernel. **Double win**: saves ~150 μs/token AND bumps NKI_FLOP_Ratio (linear score multiplier — Δscore cell **understates** the real impact) | **150** (+NKI_ratio bump) | **27+** | medium | low | SBUF for 151K-token vocab projection (tile across V); argmax semantics (handle ties deterministically to match stock sampler) | **Y** | Load-bearing | 2-3 days |
| 10 | **[X3]** `nisa.activation2` fused scale+bias+reduce | RMSNorm + softmax | Replace separate `multiply + add + reduce + activation` with single `nisa.activation2`. ScalarE flexible bias + reduction in one pass | **96** | **17** | low | low | `nisa.activation2` support for the specific bias/reduction pattern used in RMSNorm | **N** | Cleanup | 4-6 hours |
| 11 | **[X2]** Background TensorE transpose in attention QKV | `v13bc` attention | Issue matmul first, then transpose — trn3 TensorE can run the transpose in parallel with the next matmul (NeuronCore-v4 feature) | **72** | **13** | low | none | compiler scheduling — verify overlap materializes in NTFF post-change (not all transpose patterns are parallelizable) | **N** | Cleanup | 4-8 hours |
| 12 | **[Y1]** FP8 KV cache with fused dequant | `v13bc` attention | Store K/V cache as FP8 (halve bytes), dequant fused into attention matmul. KV cache = 320 KB/layer at S=640, halving → 160 KB/layer | **72** | **13** | medium | medium | FP8 range saturation on KV values (clip/normalize); dequant scale storage (+ small HBM overhead) | **N** | Cleanup | 2 days |
| 13 | **[D]** Attention W8A8 via `quantize_mx` | `v13bc` attention | Use `quantize_mx` for per-block activation scales; keep `nc_matmul` (not `_mx`) with FP8 moving + FP8 stationary. Only worth bundling with [Y1] | **24** | **4** | low | low | `quantize_mx` VectorE cost on T=1 moving tile (Phase 0 T=1 reframing: activation quantize-on-device has marginal benefit at T=1) | **N** | Cleanup | 1 day |
| — | **[J]** CC-pipeline tiling factor = 2/4 | Compiler flag under [H] | Currently 1 for TKG; under 48-layer AR ([H]), retry `--cc-pipeline-tiling-factor=2/4` | **UNMEASURED** | **?** | low | none | Only meaningful after [H] lands (no 48-layer AR without megakernel) | **Y** (gated on [H]) | Cleanup | 1 hour probe |
| — | **[I]** `@nki.compiler.skip_middle_end_transformations` | `v14a` + `v13bc` | Toggle compiler pass on isolated kernels first; if regression on isolated, never try on full model | **UNMEASURED** (±5 μs per kernel plausible) | **?** | low | low | Compile-time semantics; some passes are required for correctness | **N** (isolated verification), **Y** (real Δscore) | Cleanup | 30 min probe |
| — | **[F]** Audit HLO for non-NKI dots; convert biggest offenders | CTE router, CTE attention, embedding dots | Raises `NKI_FLOP_Ratio` **linearly** in score. Biggest unknown — could be 0.80 → 0.95 if CTE router/attention dominate | **UNMEASURED** (main gain is NKI_ratio, not Δμs) | **UNMEASURED, potentially huge** | medium | low | HLO dump requires e2e compile (so [Z.1]); converting CTE kernels is a days-per-kernel effort | **Y** | Load-bearing | 0.5 day audit + 2-3 days per kernel |

---

## Detailed hypotheses per row

### [G] Cross-layer expert prefetch
- **Hypothesis**: `dma_active_time_percent` (layer N+1's weight-load portion) drops from ~60% to ~5%; `tensor_engine_active_time_percent` on attention stays ≥ 40%. Per-layer time approaches `max(attention, MoE)` not `attention + MoE`.
- **Blocker to check**: ring-buffer SBUF for 2× `[128P × 4 × gate_up_tile]`; predictor implementation (Fate-style top-k gating on hidden_N).
- **Accuracy risk**: **low** — on predictor miss (3%), reload the correct expert (adds `MoE_DMA` back for that layer only). Expected avg slowdown 1.5% vs perfect prefetch; still net win.

### [H] Multi-layer fused TKG megakernel
- **Hypothesis**: `neuron-profile show-session --show-trace` CC-core-8 `TPB_TRIGGER` gap-2..gap-97 collapses from ~180 μs each to ~140 μs each (save ≈ `gap_μs` per layer boundary).
- **Blocker to check**: SBUF budget — 48× `residual` + `kv_cache` state + per-layer working set. Phase 0 v14a is 0 spill at single-layer; megakernel needs explicit allocator (nkilib SBUF manager).
- **Accuracy risk**: **medium** — in-place KV write at every layer; if slot indexing is off by one anywhere, all subsequent tokens corrupt. CLAUDE.md flags this plan as ongoing with known landmines.

### [Z.2] v14a + v13bc integrated baseline
- **Hypothesis**: per-layer time drops from v10e+v19b baseline (estimated ~60 + ~100 = 160 μs) to v13bc+v14a (measured 53.71 + 88.06 = 141.77 μs). Δμs ≈ 18 μs/layer × 48 ≈ 864. Being conservative with 600.
- **Blocker to check**: v13bc expects tile-transposed QKV weights (CLAUDE.md "tile-transpose required for v10e"); v14a expects fp8 pre-quant + fp32 per-neuron scales. State-dict converter must produce both correctly.
- **Accuracy risk**: **medium** — weight-layout drift is the usual mode of failure. Run `run_accuracy_check --num-tokens-to-check 5` with tight tolerance before benching.

### [F2] Direct DGE for expert weight loads ⭐
- **Hypothesis**: NTFF per-instruction analysis shows GpSimdE `evt_wait_time` on lines `v14a.py:482, 512, 524, 550` drops from 13.4 μs total → <2 μs total; `dma_active_time_percent` essentially unchanged; `device_time_us` 88 → 78 μs.
- **Blocker to check**: can trn3 NKI compiler accept `dge_mode=1` with an SBUF-resident address table? Memory note from trn2: "dge_mode tuning regressed" (but the reason was different — trn2 compiler didn't support the specific pattern). Quick Phase 2 probe: 30-line isolated kernel that precomputes offsets then does a direct-DGE load.
- **Accuracy risk**: **none** — identical data movement, only descriptor-gen path changes.
- **Why the ⭐**: this is the only top-5 lever grounded in measured Phase 0 data. Every other top-5 is a theoretical DMA or gap win from literature/estimation. [F2] has a 13.4 μs stall receipt.

### [K] Symmetric sb2sb AR
- **Hypothesis**: `cc_op_active_time` drops ~50% on the post-attention AR; `dma_active_time_percent` for the AR-DMA handoff drops since symmetric pattern reuses SBUF buffers.
- **Blocker to check**: trn3 `nccl` vs the bare AR path; verify the original ENC_ALG_MESH workaround is no longer needed (`transformer_qwen.py:145`).
- **Accuracy risk**: **none** — AR is bitwise-identical regardless of protocol.

### [X1] `nisa.exponential` in attention softmax
- **Hypothesis**: `vector_engine_active_time_percent` in v13bc drops from 50.3% → ~35%; `device_time_us` from 53.71 → ~47 μs (−6 μs = Δμs/layer).
- **Blocker to check**: `nisa.exponential` API (takes `max_value` arg — must pre-reduce); edge cases where the pre-max is very negative.
- **Accuracy risk**: **low** — different exp kernel; should be within FP16 tolerance. Rerun v13bc correctness harness with `rtol=1e-3, atol=1e-3`.

### [F3] Pack gate_up + scales into single DMA per expert
- **Hypothesis**: GpSimdE stall lines 444/512 (gate_up_scales loads, 2.8 μs total) absorbed into the gate_up weight DMA; v14a stall budget 13.4 → 10.6 μs.
- **Blocker to check**: HBM weight conversion in `qwen_complete.py:252` — requires interleaving scales with weights at specific stride boundaries. **This is integration work that touches files outside the kernel** — flag for user approval.
- **Accuracy risk**: **low** — bytes moved are identical, just re-ordered in HBM.
- **Substitutes with [F4]**: if [F4] preloads all scales to SBUF, the per-expert scale DMA disappears entirely. Pick one, not both.

### [F4] Scale preloading retry
- **Hypothesis**: 5-min investigation first: write a minimal NKI test that loads `scales[e, :]` where `e` is an SBUF-resident expert index. If `NCC_INLA001` does NOT fire on trn3, proceed with implementation. If it fires, kill this lever.
- **Blocker to check**: `NCC_INLA001: ...SBUF indirect indexing...` — trn2 raised this when the index wasn't HBM-resident. Trn3 compiler may have lifted the restriction.
- **Accuracy risk**: **none** — static load-once pattern.

### [E] LM head as NKI kernel (+ on-device argmax)
- **Hypothesis**: LM head matmul (estimated ~300 μs stock) drops ~50%; NKI_FLOP_Ratio rises from placeholder 0.80 → ~0.88 (LM head is a large contributor to total MACs). Linear in score.
- **Blocker to check**: SBUF tile for [H=2048] × [V=151936]; partition the V dim across iterations; on-device argmax over large V.
- **Accuracy risk**: **low** — argmax is deterministic; just need to match stock sampler's tie-breaking.

### [X3] `nisa.activation2`
- **Hypothesis**: RMSNorm ScalarE time drops ~30%; attention softmax saves additional 1-2 μs/layer.
- **Blocker to check**: `nisa.activation2` operand-shape constraints vs our exact RMSNorm pattern.
- **Accuracy risk**: **low** — fused op; numerical path unchanged.

### [X2] Background TensorE transpose
- **Hypothesis**: v13bc tensor_engine_active_time_percent rises slightly (more work in flight); `device_time_us` drops 1-2 μs/layer as transpose overlaps with next matmul.
- **Blocker to check**: compiler scheduler — not all transpose patterns can be interleaved. Verify in NTFF post-change.
- **Accuracy risk**: **none** — transpose itself is exact.

### [Y1] FP8 KV cache with fused dequant
- **Hypothesis**: attention `hbm_read_KiB` drops from 9541 → ~9381 (half of 320 KB KV cache saved); `device_time_us` drops 1-2 μs/layer.
- **Blocker to check**: FP8 E4M3 saturation on KV values; per-head scale storage.
- **Accuracy risk**: **medium** — KV quantization compounds across the context. Validate with `rtol=5e-2, atol=5e-2` at full S_prior=640.

### [D] W8A8 attention via `quantize_mx`
- **Hypothesis**: minor VectorE reduction on attention matmul path; small overall win.
- **Blocker to check**: `quantize_mx` T=1 overhead — Phase 0 skill warns "Quantize-on-device gives no benefit here because the moving tile is too small to amortize." Bundle with [Y1] or skip.
- **Accuracy risk**: **low** — already paired FP8 path; tolerances relaxed.

### [J] CC-pipeline tiling factor
- **Hypothesis**: under [H], `cc_op_active_time` overlaps more with compute.
- **Blocker to check**: only meaningful with megakernel in place.
- **Accuracy risk**: **none** — scheduling change.

### [I] `@nki.compiler.skip_middle_end_transformations`
- **Hypothesis**: compile-time pass removal exposes opportunities the middle-end clobbered.
- **Blocker to check**: some passes are correctness-critical; regressions likely.
- **Accuracy risk**: **low-medium** — depends on which passes get skipped.

### [F] Audit HLO for non-NKI dots
- **Hypothesis**: NKI_FLOP_Ratio placeholder 0.80 → 0.90-0.95 possible. Biggest NKI_ratio lever in the matrix.
- **Blocker to check**: requires e2e HLO dump (so [Z.1]); CTE conversions are days-per-kernel.
- **Accuracy risk**: **low** — kernel conversions, not algorithmic changes.

---

## Round 1 recommendation

User constraint: "almost certainly include [Z.1] as an out-of-band prerequisite plus 2-3 in-matrix candidates that don't require [Z.1] to start on."

Candidates with **Requires [Z.1]? = N** that can start immediately in Phase 2, ranked by Δscore among that subset:

| Rank (within N-set) | Lever | Δscore | Rationale |
|:---|:---|---:|:---|
| 1 | **[F2]** Direct DGE | 86 | Evidence-backed — the only lever attacking the Phase 0 measured root cause (GpSimdE 13.4 μs stall). One isolated kernel, no integration risk, `v14a.py` only. |
| 2 | **[X1]** `nisa.exponential` | 52 | Cheapest (2-4 hrs), direct API swap in v13bc; verifiable in the existing `bench_v13bc_trn3.py` harness with measured VectorE drop. |
| 3 | **[F4]** Scale preloading retry | 43 | 5-min compiler-constraint probe first; if `NCC_INLA001` is lifted on trn3, this is a cheap follow-on to [F2] on the same root cause. Kill fast if the compiler still blocks. |

### Recommended Round 1 (top-3 in-matrix + prerequisite)

1. **[Z.1]** — out-of-band; unblocks e2e measurement for every other lever.
2. **[F2]** — primary attack on the Phase 0 measured bottleneck.
3. **[X1]** — cheapest VectorE win; can complete inside the [F2] dev loop.
4. **[F4]** — probe-first, implement-if-unblocked; cheap to de-risk.

All three in-matrix picks are verifiable against the existing `bench_v14a_trn3.py` / `bench_v13bc_trn3.py` harnesses and don't touch any qwen module or CTE path.

---

## Notes and caveats

- **All Δscore values are relative rankings.** Replacing the placeholder constants with real numbers after [Z.1] may reshuffle 2-3 ranks. The `[G] > [H] > [Z.2] > [F2]` ordering is robust only to order-of-magnitude assumptions.
- **NKI_FLOP_Ratio of 0.80** is a placeholder used for ranking. The real value is unmeasured, and levers that raise NKI_ratio ([E], [F]) have systematically understated Δscore in this table.
- **Phase 0 isolated numbers** (v14a = 88.06 μs, v13bc = 53.71 μs, GpSimdE stall = 13.4 μs) are the only measured evidence in this matrix. Every other Δμs is derived from literature, trn2 history, or estimate.
- **`[G]` and `[H]` are the largest Δscores but also the highest-risk.** `[H]` has a Phase 0 track record of plausible-theory-positive regressions (v6c, v12g, v12h, v12k). Round-1 readiness of either is unlikely without [Z.1] in place.
- **Hard non-goals still apply** per Phase 0 master brief: no `nc_matmul_mx` on gate/up/down TKG projections, no `nc_stream_shuffle` tree reductions, no in-kernel AllReduce, no CTE-path edits until TKG wins land. All matrix rows respect these.
