# Trn3 Round 2 Report — Phase 3 close-out

**Date**: 2026-04-21
**Phase 3 intent**: integrate v15c MoE + v17_fast_exp attention into `qwen_complete.py`; run 5-prompt e2e benchmark; produce first real trn3 TTFT / tok-s / NKI_FLOP_Ratio / score numbers.
**Phase 3 outcome**: **integration landed; e2e measurement blocked by a framework serialization limit**. STEPS 1a / 1b / harness fix PASS. STEPS 2 / 3 / 4 blocked on the same root cause (see §"Phase 3 blocker cascade"). Round 1-2 *isolated* kernel wins are verified real; e2e translation unknown.

---

## 1. Headline — did Round 1's -17.2% MoE win translate to e2e?

**Not measured this round.** Every e2e attempt hit one of three progressive blockers (full chain in §6). The last blocker — HLO graph exceeds protobuf 2 GB serialization limit — is an SDK/framework-level constraint that kernel-level Round 2 work cannot address directly.

**What IS measured and independently verified**:
- `v14a` MoE TKG: **88.06 μs** (Phase 0 baseline, verified non-zero via `probe_v14a_v15c_zero_check.py`: absmean 11.6, range [-49, 55]).
- `v15c` MoE TKG: **72.88 μs, bit-exact vs v14a** (−17.2% Round 1 MoE-only win, verified).
- `v13bc_sbm_tiled` attention TKG: **53.71 μs** (Phase 0 baseline, verified non-zero via `probe_v13bc_passthrough.py`: absmean 3.0e-3).
- `qwen3_attn_tkg_v17_wrapper` (integration): PASS (absmean 3.0e-3 in the e2e shape).
- `quantize_and_pack_gate_up` / `quantize_down` converter: byte-exact vs `bench_v15c_trn3.py` reference (max_diff = 0 for all three TKG buffers).
- `qwen_complete.py` with `v15c + v13bc_sbm_tiled` imports: **Python-importable PASS** (`python3 -c "import qwen_complete"`).

**What was NOT measured**: real `current_time_ms`, `NKI_FLOP_Ratio`, `accuracy`, per-prompt `latency_ms_p99`, `throughput`, `score`, inter-op Python/NxDI gap, real AR timing. All e2e-dependent numbers remain PLACEHOLDERS from Phase 0.

## 2. NKI_FLOP_Ratio — first-time measurement

**Not attained.** `count_nki_flop_ratio()` at `main.py:553` requires the compiled HLOs at `/tmp/nxd_model/{context_encoding,token_generation}_model/_tp0_bk0/*.hlo_module.pb`. Those files never land because compilation fails at HLO serialization before NEFF emission. Remains `UNMEASURED` placeholder.

## 3. Updated engine profile — did VectorE stay the bottleneck at 48 layers?

**Not measurable this round** (no e2e NTFF). The **single-layer v15c profile from Phase 0 + Round 1 remains the reference**:
- TensorE 42%, **VectorE 55% (40.1 μs)**, ScalarE 28%, GpSimdE 30%, DMA 57%.
- Cross-engine parallel-bound: every Round 2 plan that broke a single engine's parallelism regressed or was noise-level (Plans A/B/C = +1.4% / +16.5% / 0.9% respectively).

Whether VectorE is still dominant when 48 layers stack with Python/NxDI framework gaps is **unknown** pending STEP 3 NTFF analysis, which is blocked.

## 4. Harness-fixes applied during Phase 3

| # | Fix | File | Status |
|---|---|---|---|
| 1 | `v17_fast_exp` false-positive discovered (silently produces all-zero output); reverted to `v13bc_sbm_tiled` | `_qwen_integration.py` import line | Applied |
| 2 | Wrapper `nisa.dma_copy(dst=reshape(...), ...)` alias bug (see `docs/integration_findings.md` Finding 3); whole-tile transpose + `nl.store` with output-shape allocation; reshape only on return | `_qwen_integration.py:92-155` | Applied, verified |
| 3 | `--skip-accuracy-check` CLI flag added; wraps baseline_qwen load + `run_accuracy_check` | `main.py:96, 810-825, 888-946` | Applied, flag reaches main.py correctly |
| 4 | 64 GB swap file at `/swapfile` (sudo) — absorbs HLO compile memory peak | system-level | Applied, active |

## 5. Phase 3 blocker cascade — what went wrong, what broke it, in order

1. **Block A — v17_fast_exp produces zero output**. Round 1 Plan B was a false-positive win. `nisa.exponential` with default `reduce_cmd=idle` appears to corrupt shared VectorE accumulator state that downstream ops consume. Caught by a triple-check (zero / range / allclose) that the original loose `rtol=1e-2, atol=2e-2` tolerance could not distinguish (attention output magnitude is ~1e-2, so zero passes atol). **Fix**: revert wrapper to `v13bc_sbm_tiled`. **Fixed**. Details: `docs/integration_findings.md` Finding 1.

2. **Block B — `nisa.dma_copy(dst=reshape(...), ...)` silently writes to the wrong buffer**. Wrapper's original `output_2d.reshape((NUM_OUT_COLS, PMAX))` as DMA dst produced all-zero HBM output even when the source SBUF had correct data. `nl.store` with output allocated in the target shape + reshape on return: **Fixed**. Details: `docs/integration_findings.md` Finding 3 + Landmine L1.

3. **Block C — OOM during HLO compile on dual-model trn3 path**. Initial hypothesis: `evaluate_single` loads both qwen_complete + baseline_qwen = 2 × 60 GB = 120 GB; + compile overhead > 124 GB DRAM. **Partial fix**: `--skip-accuracy-check` flag eliminates baseline load. Single-model load = 35 GB. Still SIGKILL'd at ~3 min. Kernel dmesg confirms `anon-rss: 127 GiB` at OOM moment. **Real cause**: HLO compile itself has ~80 GB overhead on top of model load (XLA tracing intermediates + `neuronx-cc` subprocess). **Fully resolved by Option S**: 64 GB swap file absorbs the peak.

4. **Block D — HLO graph exceeds protobuf 2 GB serialization limit**. After Block C resolved, the compile ran to completion on all 5 prompts but failed at `torch_neuronx/xla_impl/trace.py:179`: `metaneff.serialized_graph_def = hlo.SerializeToString()` → `google.protobuf.message.EncodeError: Failed to serialize proto`. Every prompt failed identically at 400-785 s (variable due to partial cache reuse). **Root cause**: `BlockwiseMatmulConfig.block_size=8192` (Phase 0 harness fix Block #3) routes all CTE prompts through `forward_all_experts` in NxDI's MoE dispatcher. `forward_all_experts` materializes 128 experts × 48 layers = **6144 expert sub-graphs** in the HLO. That plus Qwen3-30B's attention + router + residual + normalization ops produces an HLO whose serialized size exceeds the 2 GB protobuf message limit (protobuf 7.34.1 with `upb` backend installed, in principle supports larger, but `SerializeToString` still fails in practice). **NOT FIXED.**

## 6. Why STEP 2 can't complete today

Block D is a framework-level limitation that requires one of:

- **Route CTE MoE through a path that does NOT expand to 128 sub-ops per layer.** This is precisely what Round 3 candidate `[M1]` proposes (drop redundant bf16 CTE MoE weights + use a single-kernel CTE MoE).
- **Patch `torch_neuronx/xla_impl/trace.py`** to handle oversized HLOs (split or use file-based serialization). Out of scope for kernel work; SDK-level change.
- **Reduce the model's HLO complexity** via a structural change: fewer layers, smaller expert count, different MoE topology. None of these are performance-positive options.

## 7. Round 3 top-3 recommendation — REVISED based on Phase 3 findings

Round 2's recommendation was `[Z.1+Z.2] e2e test` > `[H] megakernel` > `[F] non-NKI dots`. After Phase 3 discovered Block D, **this ranking has to change**. `[Z.1+Z.2]` is now **blocked pending [M1]**, which promotes `[M1]` to the #1 spot.

### Recommended Round 3 Top-3

1. **`[M1]` — Eliminate the CTE `forward_all_experts` HLO explosion + drop redundant bf16 MoE weights** (NEW Phase 1-matrix addition)
   - **Rationale**: without this, no e2e measurement is possible (Block D blocks score formula entirely). Also saves ~30 GB DRAM during trace (Phase 3 Risk #3 side benefit).
   - **Scope**: replace the `BlockwiseMatmulConfig(block_size=8192)` CTE fallback with a SINGLE-KERNEL CTE MoE. Options: (a) wire `v15c.qwen3_moe_fused_tkg` into CTE (needs shape adaptation for T>1 or a prefill-compatible variant); (b) convert NKI MoE TKG variant to consume the CTE token batch directly; (c) pre-compile CTE as a separate graph using the stock NxDI MoE variant that doesn't loop per-expert.
   - **Blocker check**: requires editing qwen_complete.py's CTE path + possibly a new NKI kernel variant. ~3-5 days subagent time.
   - **Effect**: unlocks e2e measurement, removes DRAM hazard, reduces HLO size by ~60-80% (eliminates 6144 sub-graphs).

2. **`[H]` — Multi-layer fused TKG megakernel** (unchanged from Round 2 recommendation)
   - **Rationale**: still the largest single kernel Δscore candidate (Round 2 matrix placeholder 211); v15c's single-kernel ceiling makes megakernel the only path to break past 72 μs/layer. Plan C's design doc (`docs/plan_g_cross_layer_prefetch_design.md`) is the integration hook.
   - **Prerequisite**: `[M1]` provides the first real baseline to measure megakernel gain against.

3. **`[X1b]` — Re-probe `nisa.exponential` with `reduce_cmd=reset_reduce` mitigation** (NEW — from Phase 3 v17 postmortem)
   - **Rationale**: `[X1]` was thought to be a cleanup win in Round 1. Phase 3 discovered it's a known-broken SDK usage. If the `reset_reduce` pattern works, Round 1's claimed 0.7% attention win becomes real. Low-risk 1-day probe.
   - **Prerequisite**: none; isolated kernel test.

### Alternative deferred Round 3 candidates (no specific trigger)

- `[F]` Non-NKI HLO dot audit — needs `[M1]` first (no HLOs produced before [M1]).
- `[E]` LM head NKI — needs `[M1]` to measure stock LM head cost first.
- `[G]` Cross-layer expert prefetch — design doc ready, needs `[H]` megakernel integration.
- `[Y1]` FP8 KV cache — isolated-kernel, independent; small win.

## 8. Round 1 + Round 2 verified kernel inventory (Phase 3-verified)

| Kernel | device_time_us | Correctness | Round | Tier |
|---|---:|:---|:---|:---|
| v12i MoE TKG | 88.96 | non-zero (bit-exact vs v14a) | Phase 0 baseline | Deprecated (superseded by v14a/v15c) |
| v14a MoE TKG | 88.06 | verified non-zero (absmean 11.6) | Phase 0 baseline | Deprecated (superseded by v15c) |
| v15a MoE TKG | 76.38 | bit-exact vs v14a | Round 1 Plan A | Superseded by v15c (dominates) |
| **v15c MoE TKG ⭐** | **72.88** | **bit-exact vs v14a, verified non-zero** | **Round 1 Plan C** | **Phase 3 winner** |
| v15e MoE TKG | 73.3 | bit-exact vs v15c | Round 2 Plan A | Regression (+1.4%, kept per <2% rule, NOT recommended) |
| v15f MoE TKG | ~85 | bit-exact vs v15c | Round 2 Plan B | **REVERTED** (+16.5%) |
| v15g MoE TKG | 72.48 | bit-exact vs v15c | Round 2 Plan C | Infrastructure skeleton + design doc |
| v13bc_sbm_tiled attn TKG ⭐ | 53.71 | verified non-zero (absmean 3.0e-3) | Phase 0 baseline | **Phase 3 winner** |
| ~~v17_fast_exp attn TKG~~ | ~~53.28~~ | **ALL-ZERO OUTPUT — false positive** | ~~Round 1 Plan B~~ | **REVERTED POST-HOC** |

## 9. Session handoff — what a next session needs to know

### Current state
- `qwen_complete.py` wired to v15c + v13bc_sbm_tiled, imports cleanly, smoke-tested PASS.
- `main.py` has `--skip-accuracy-check` flag (works correctly).
- 64 GB swap file at `/swapfile` (via sudo) absorbs HLO compile memory peak.
- Every e2e attempt fails at HLO proto serialization after ~7-13 minutes per prompt.
- Compile cache at `/var/tmp/neuron-compile-cache/` + traced_model at `~/qwen-30b-a3b/traced_model/` (clear with `rm -rf` before any signature change).

### First action in next session
Pick Round 3 #1 ([M1]) and start implementation. Without [M1], e2e is unreachable, which means NO plan after Round 2 can be scored in reality.

### Reference files produced this session
- `/home/ubuntu/nki-moe/docs/integration_findings.md` — v17 postmortem, known NKI landmines, triple-check pattern.
- `/home/ubuntu/nki-moe/docs/plan_g_cross_layer_prefetch_design.md` — Round 2 Plan C design doc for future [H]-coupled [G] implementation.
- `/home/ubuntu/nki-moe/kernels/moe_fused_tkg/quantized/OPTIMIZATION_LOG_TRN3.md` — canonical log including cold-start recovery notes.
- `/home/ubuntu/nki-moe/docs/trn3_optimization_matrix.md` — Phase 1 matrix (needs [M1], [X1b] appended for Round 3).
- `/tmp/phase0_launch.py` — 5-prompt launcher; supports `N_PROMPTS` env var.
- `/tmp/phase0_e2e.log`, `/tmp/phase0_mem.log` — most recent e2e attempt + memory trace.

### Matrix updates pending
`[M1]` (Phase 1 matrix candidate — new this session) and `[X1b]` (Round 3 probe — new this session) should be added to `docs/trn3_optimization_matrix.md`. Currently only noted in this report and `docs/integration_findings.md`.

## 10. Accuracy validation — explicit TODO for user's out-of-band work

Every score in Round 1-2 uses **stubbed trn2 baselines** via `tail -n +2 prompt_data_trn2.csv > prompt_data_trn3.txt` — scores are speedup-vs-trn2, not speedup-vs-trn3-reference. Before Round 1 closeout, run:
```
python3 main.py --mode generate_accuracy_baselines --platform-target trn3 \
  --model-path ~/qwen-30b-a3b/hf_model \
  --compiled-model-path ~/qwen-30b-a3b/traced_model_baseline
```
to produce real trn3 baseline numbers. This is blocked by the same HLO-serialization issue (Block D) until `[M1]` lands.

---

**Phase 3 complete (within available scope).** Round 3 cannot begin without `[M1]` — the CTE HLO explosion is the critical path for any future e2e measurement.
