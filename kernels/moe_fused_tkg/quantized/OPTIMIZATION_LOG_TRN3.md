# FP8 MoE Fused TKG Kernel — Optimization Log (trn3)

Kernel: `qwen3_moe_fused_tkg` — fused RMSNorm + Router + TopK(8) + Expert MLP for Qwen3-30B-A3B (TP=4, LNC=2, T=1 TKG).
Hardware: trn3 (NeuronCore-v4 / trn3.3xlarge). Starting point: v12i (best trn2 kernel at ~90 μs).
Platform target: `NEURON_PLATFORM_TARGET_OVERRIDE=trn3`.

**All trn2 benchmarks in `OPTIMIZATION_LOG.md` are for reference only — trends not absolute numbers.**

---

## Trn3 Baseline: v12i on trn3

Benchmarked v12i (unchanged, just NEURON_PLATFORM_TARGET_OVERRIDE=trn3) on trn3.3xlarge.

```
v12i trn3 baseline profile:
  device_time_us       = 89.80
  tensor_engine_pct    = 36.2%
  scalar_engine_pct    = 23.7%
  vector_engine_pct    = 47.7%
  gpsimd_engine_pct    = 45.3%
  dma_active_pct       = 57.4%
  spill_bytes          = 0
  hbm_read_KiB         = 16,488.0
  hbm_write_KiB        = 4.0
  mfu_estimated_pct    = 0.003%
  mm_arithmetic_intensity = 2.546
```

**Bottleneck**: DMA at 57.4% (~51.5 μs). HBM reads fixed at 16,488 KiB (8 expert FP8 weight matrices).

**vs trn2 v12i (90.54 μs)**: −0.74 μs (−0.8%) — essentially identical. DMA bottleneck is architecture-agnostic.

**Key trn3 differences vs trn2 v12i profile:**
- scalar_engine_pct: 44.6% → 23.7% (trn3 handles scale ops faster — smaller bottleneck)
- tensor_engine_pct: 44.1% → 36.2% (slightly lower)
- gpsimd_engine_pct: 45.3% (new metric — GpSimdE at similar level as DMA)

**Critical trn3 ISA incompatibility found:**
- `tensor_partition_reduce` with `op=nl.maximum` is NOT supported on trn3
- This blocks v13a (W8A8 FP8 with per-token absmax) from compiling on trn3
- All trn3 kernels must avoid `tensor_partition_reduce(op=nl.maximum)`
- `tensor_partition_reduce(op=nl.add)` DOES work (used in RMSNorm norm computation)

**Available on trn3 (not on trn2):**
- `nisa.nc_matmul_mx` — MXFP quantized matmul with integrated dequant
- `nisa.quantize_mx` — hardware quantization to MXFP8 format

---

## Round 1

### Plan A — trn3 Port of v12i (nc_matmul_mx Probe)

**Idea**: Simple trn3 port of v12i using `@nki.jit(platform_target="trn3")`. Probe whether nc_matmul_mx can be used with existing FP8 weights. Establishes clean trn3 baseline.

**Result**: v14a — 88.68 μs (−1.3% vs baseline)

**Key finding**: `nc_matmul_mx` CANNOT load MX dtypes from HBM (NCC_IBIR285). Weights must be produced in-kernel via `quantize_mx` or stored as raw bytes + separate MXFP scale tensors. The kernel interface (FP8 int8 weights + fp32 per-neuron scales) is incompatible with nc_matmul_mx without format conversion.

---

### Plan B — W8A8 via nc_stream_shuffle Tree Reduction

**Idea**: Port v13a's W8A8 per-token FP8 quantization to trn3 by replacing the unsupported `tensor_partition_reduce(op=nl.maximum)` with a 7-step nc_stream_shuffle binary tree reduction for cross-partition absmax.

**Result**: v14b — 137.52 μs (+53.1% regression)

**Key finding**: The nc_stream_shuffle tree reduction is extremely expensive — 7 shuffle rounds plus elementwise ops dwarfs the cost of the original single `tensor_partition_reduce`. Not viable.

---

### Plan C — Single 8-Expert Wave (Merged Loop Structure)

**Idea**: Merge the 2-wave structure (2×4 experts) into a single `affine_range(8)` loop to eliminate wave boundary overhead and give the compiler better scheduling freedom.

**Result**: v14c — 89.13 μs (compiler produced identical NEFF as v14a — same hash)

**Key finding**: The compiler's HLO canonicalization collapsed the single-wave restructuring into the same code as v14a. Python-level loop restructuring provides no benefit here.

---

## Round 1 Synthesis

```
| Plan | device_time_us | tensor_eng% | dma% | spill | vs baseline |
|------|---------------|-------------|------|-------|-------------|
| A (v14a) | 88.68  | ~36%        | ~57% | 0     | −1.3%       |
| B (v14b) | 137.52 | N/A         | N/A  | 0     | +53.1%      |
| C (v14c) | 89.13  | identical NEFF to v14a | — | 0 | −0.7% |
```

**Best from Round 1**: v14a (88.68 μs) — marginal improvement from clean trn3 port.

**Analysis**: The DMA bottleneck (57.4%) persists. 16,488 KiB of FP8 weight data must be loaded per call. The main path to improvement on trn3 is:
1. **nc_matmul_mx with proper weight format**: requires in-kernel weight SBUF transposition + fp32→uint8 MXFP scale format conversion. The nkilib patterns (quadrant-based sparse scale layout, quantize_mx for activations) are now fully understood from `docs/nkilib_mxfp_kernel_patterns.md`.
2. **MXFP4 weights**: halves weight HBM footprint (8,244 KiB instead of 16,488 KiB), directly addresses DMA bottleneck. Requires offline pre-quantized FP4 weights + separate MXFP scale tensors.
3. **quantize_mx activation path**: Even without nc_matmul_mx, using quantize_mx avoids the per-token absmax (which was blocked on trn3). This enables W8A8 if we pass the scale separately from the matmul.

**Key constraint**: The current interface `gate_up_w [E=128, H=2048, 2*I=384] int8` stores weights with partition=output_neurons (I). nc_matmul_mx requires partition=H_contraction. SBUF transposition is required.

**Next round focus**: Implement proper nc_matmul_mx with MXFP8 activation quantization using the correct weight + scale layout from nkilib patterns.

---

## Round 2

### Plan D — nc_matmul_mx with In-Kernel Weight Transposition + MXFP8 Activations

**Idea**: Full nc_matmul_mx implementation.
1. Load FP8 weight from HBM as raw bytes `[E, H=2048, GU=384]` → transpose in SBUF to `[128_H, H/512=4_tiles, I]` layout (partition=H_contraction)
2. Convert fp32 per-neuron scales to uint8 MXFP8 scales in sparse SBUF quadrant layout (4 rows per 32-partition quadrant)
3. Use `quantize_mx` on bf16 RMSNorm output → float8_e4m3fn_x4 + uint8 sparse scale
4. Call `nc_matmul_mx` with stationary=weight (FP8), moving=activation (FP8)
5. Expected: higher arithmetic intensity per DMA byte, reduced tensor_engine_pct

**Result**: TBD

---

### Plan E — quantize_mx Activations + nc_matmul (No Weight Transpose)

**Idea**: Use `quantize_mx` only for activation quantization (bypasses tree reduction issue), but keep `nc_matmul` (not nc_matmul_mx) for the actual matmul. The weight stays in current format; the quantized activations are dequantized-multiply with the fp32 scale inside nc_matmul's existing path.
- Actually: use quantize_mx to get FP8 activations, then call nc_matmul with FP8 moving tensor (matches current stationary=fp8_weight approach). The per-token scale from quantize_mx is combined with the per-neuron weight scale in a post-matmul correction.
- This avoids SBUF transposition but still tests if the trn3 quantize_mx instruction provides any speedup.

**Result**: TBD

---

### Plan F — GpSimdE Reduction via Direct DGE (dge_mode=1)

**Idea**: The trn3 baseline shows GpSimdE at 45.3%, indicating high indirect DMA (dge_mode=0) overhead for per-expert weight loading. Switch from indirect DGE (dge_mode=0, compute address at runtime) to direct DGE (dge_mode=1, fixed address stride) for the weight tensor_copy operations. This requires the weight to be in a strided layout in HBM.
- Currently: expert weights are accessed via `expert_id` indexing → indirect DGE
- Alternative: precompute addresses using nl.add on the expert_id, then use static-stride DMA

**Result**: TBD

---

## Summary Table (trn3)

| Version | device_time_us | vs v12i trn3 | Bottleneck | Key change |
|---------|---------------|-------------|------------|------------|
| v12i (trn3 baseline) | 89.80 | — | DMA 57.4% | v12i on trn3 hardware (unchanged) |
| v14a | 88.68 | −1.3% | DMA ~57% | Clean trn3 port, nc_matmul_mx probe |
| v14b | 137.52 | +53.1% | Shuffle tree | W8A8 tree-reduction (too expensive) |
| v14c | 89.13 | −0.7% | DMA ~57% | Single 8-expert wave (compiler same NEFF) |

---

## Trn3 Ground Truth — 2026-04-21

Phase 0 re-measurement with the trn3-skill harness (adds `vector_engine_pct`
to `BenchmarkResult`; writes NTFFs that the source-attribution recipes can
parse). All numbers are from the hardware NTFF captured in one
`wrap_benchmark(warmup=5, iters=50)` pass per kernel. NeuronCores are
exclusive; kernels ran sequentially.

### Harness / integration fixes (Phase 0)

1. **trn3 benchmark harness NTFF regex** — `.claude/skills/nki-kernel-optimizer-trn3/scripts/benchmark.py` had a filename-pattern bug that prevented NTFF discovery on trn3 (NEFFs land as `neff_<hash>_vnc_<N>.neff`, NTFFs as `<hash>_vnc_*.ntff`; the old stripping logic searched for `<hash>_vnc_<N>_vnc_*.ntff`). Fixed the skill source (split on `_vnc_`). All Phase 0 isolated benches use the corrected version.

2. **`@nki.jit(platform_target="trn2")` kwarg deprecated by SDK** — stripped from `kernels/router_topk/qwen3_router_topk_plan_a.py:83` (`@nki.jit(platform_target="trn2")` → `@nki.jit`). The env var `NEURON_PLATFORM_TARGET_OVERRIDE=trn3` is now the single source of truth for platform selection. Scope: only this one file, because it is the sole `platform_target=` hit on the `qwen_complete` import graph. 22 other occurrences in the repo were left untouched per CLAUDE.md.

3. **`qwen_complete.py:86` import path updated** — `kernel_v19b` moved to `kernels/moe_fused_tkg/versions/` in commit `e934cb0` ("3 kernel integration") but the import in `qwen_complete.py` was never updated:
   ```
   -from kernels.moe_fused_tkg import kernel_v19b as custom_moe_fused_kernel
   +from kernels.moe_fused_tkg.versions import kernel_v19b as custom_moe_fused_kernel
   ```
   `qwen_complete.py` was non-importable on the current checkout until this fix. Verified with `python3 -c "import qwen_complete; print('import OK')"` → **PASS**.

4. **FINDING: "current production" baseline (`qwen_complete`) was broken end-to-end for an unknown duration.** Since commit `e934cb0` on 2026-04-01, the authoritative submission candidate module has failed to import on any fresh checkout. The two sibling modules with identical MoE wiring (`qwen_baseline_quant.py:107`, `qwen_round_one.py:95`) have the same `kernel_v19b` import commented out, so they fall back to stock MoE for TKG (and do not exercise the NKI MoE kernel). **Implication for Phase 1 matrix candidate `[Z]`**: expand from "wire v14a + v13bc into qwen_complete" to "wire v14a + v13bc into a functional production module (qwen_complete import fix + kernel upgrade)." The import fix is Phase 0 plumbing; the kernel upgrade is the [Z] lever itself.

5. **NxDI `forward_blockwise` hits a missing private kernel in this SDK** — compile raised `NotImplementedError: _call_shard_hidden_kernel is not available` at `blockwise.py:269` because `neuronxcc.nki._private.blockwise_mm` is absent (the `.so` file exists at `_private_kernels/blockwise_mm.cpython-312-x86_64-linux-gnu.so` but `nki_import.py`'s search for the specific function `blockwise_mm_baseline_shard_hidden` fails across all 3 candidate module paths). **Workaround**: bumped `BlockwiseMatmulConfig.block_size` from **128 → 8192** in `qwen_complete.py:107`. Dispatcher at `expert_mlps_v2.py:1495-1498` chooses `forward_all_experts` (plain PyTorch matmul) when `total_tokens × top_k < block_size`. Our longest prompt is `ctx=402 × top_k=8 = 3216 < 8192` ✓, so all 5 Phase 0 prompts route through the blockwise-free path. TKG (`seq_len=1`) is unaffected — it was never on the blockwise path. **Cost**: CTE becomes slower (loads all 128 experts per prompt instead of only-blocks-touched), but CTE is not the optimization target; TKG numbers are unchanged. **Follow-up**: drop `block_size` back toward 128 once the private kernel is restored in the SDK.



### Isolated kernel benchmarks (trn3.3xlarge, trn3 target, LNC=2 where applicable)

| Kernel                | device_time_us | TensorE% | VectorE% | ScalarE% | GpSimdE% | DMA%  | spill | HBM_read_KiB | mfu%   | mbu%   | mm_AI |
|-----------------------|---------------:|---------:|---------:|---------:|---------:|------:|------:|-------------:|-------:|-------:|------:|
| v12i MoE (declared 89.80) | **88.96**  | 35.6     | 45.9     | 22.9     | 44.0     | 58.1  | 0     | 16,488       | 0.0031 | 0.159  | 2.546 |
| v14a MoE (declared 88.68) | **88.06**  | 35.3     | 46.6     | 22.9     | 45.6     | 59.5  | 0     | 16,488       | 0.0031 | 0.159  | 2.546 |
| v13bc attention           | **53.71**  | 45.5     | 50.3     | 14.8     |  5.8     | 46.6  | 0     |  9,541       | 0.0047 | 0.155  | 1.425 |

(Percentages are `<engine>_active_time_percent` from the NTFF summary; they
do not sum to 100% because engines run in parallel.)

Router TKG is **fused inside v14a** (`router_w` matmul + topk(8) + softmax,
v14a.py:99–573). The standalone kernels in `kernels/router_topk/` are CTE
(context-encoding / prefill) only; there is no separate router-TKG kernel
to bench, and the router's cost is already accounted for inside v14a's
88.06 μs. This matches the non-goal on CTE-path changes.

### Bottleneck characterization (per skill Step 2 taxonomy)

| Kernel | Classification | Supporting metrics |
|--------|---------------|--------------------|
| v14a MoE | **Memory-bound (DMA) + GpSimd-stall hybrid** | DMA active 59.5% (52.4 μs of 88.1), HBM read 16.9 MB dominated by 8 experts × (gate_up FP8 + down FP8 + scales). TensorE only 35.3%. MFU 0.003% — the matmuls are tiny (T=1), compute is nowhere near saturation. |
| v12i MoE | Same as v14a (v14a saves ~0.9 μs on same NEFF class) | DMA 58.1%, HBM read 16.9 MB, MFU 0.003%. |
| v13bc attention | **Memory-bound mildly, more balanced** | DMA 46.6% (25.1 μs of 53.7), TensorE 45.5%, VectorE 50.3%. HBM read 9.5 MB (QKV + Wo + K_cache + V_cache). Much higher balance between compute and DMA than MoE. |

The MoE is the single biggest per-token cost. **No spill on any kernel**
(SBUF 32 MiB is ample).

### v14a NTFF per-instruction analysis (neuron-profile JSON export, trn3)

Top opcode counts for the 3,834-instruction trace:

| Opcode              | Count | Notes                                   |
|---------------------|------:|-----------------------------------------|
| LDWEIGHTS           | 1,332 | stationary-weight loads into TensorE    |
| MATMUL              | 1,332 | paired 1:1 with LDWEIGHTS               |
| PSEUDO_DMA_DIRECT2D |   160 | explicit DMA tiles                      |
| ALU_OP              |   144 | cross-engine addr / scalar ops          |
| EVENT_SEMAPHORE     |   137 | synchronization overhead                |
| ACTIVATE            |   130 | SiLU/Swish                              |

- **Stall budget dominated by GpSimdE**: 13.4 μs of `evt_wait_time` on GpSimd
  vs 2.2/1.6/1.5/1.4 μs on Tensor/Vector/Scalar/Sync. GpSimdE is ~15% of
  total device time *just waiting on semaphores* — caused by indirect DGE
  (`dge_mode=0, scalar_offset=expert_id`) address computation serializing
  against the DMA engines.
- **Top stall lines (all GpSimd, all indirect-DGE expert weight loads)**:
  v14a.py:**512** (gate_up_scales W1 load, 1.6 μs×56), **524** (down_w tile 0
  W1 load, 1.4 μs×52), **482** (down_scales W0 load, 1.2 μs×46),
  **550** (down_scales W1 load, 1.2 μs×44), **421** (expert-id scalar
  scratch at offset 0, 1.1 μs×32). These are exactly the per-expert indirect
  weight loads that `[G] cross-layer expert prefetch` targets.
- **Skinny DMA count is low**: only 16 transfers < 1 KiB (all 4-byte
  expert-id fetches on line 421). 34 transfers in the 256 B–4 KiB band.
  The rest of the weight-load traffic is fat (16 transfers each in the
  16–64 KiB, 64–256 KiB, and 256 KiB–1 MiB bands). No tile-size pathology.
- **No engine idle gaps > 0.1 μs**: scheduling is tightly interlocked;
  there is no compiler-scheduler hole to close.
- **cc_op_count = 0** on all isolated NTFFs — as expected for a single-kernel
  MoE/attention bench (no AllReduce in these traces; that lever lives at
  the transformer-layer level which requires e2e to measure).

### E2E benchmark — ran with stubbed trn3 baselines

Weights landed at `~/qwen-30b-a3b/hf_model` (16 safetensors shards).
`prompts.txt` holds 24 prompts; `prompt_data_trn2.csv` holds 5 real
baselines (prompts 0–4). No `prompt_data_trn3.txt` existed.

**Scope chosen**: run `evaluate_single` on the 5 prompts that have real
trn2 baselines (0, 1, 2, 3, 4) — matches the actual competition evaluation
scope per README FAQ and `docs/winning_strategy.md`. Stubbed
`prompt_data_trn3.txt = tail -n +2 prompt_data_trn2.csv`.

> ⚠ **Baselines = trn2 reference numbers (stubbed). Scores are
> speedup-vs-trn2, NOT speedup-vs-trn3-reference. TODO: regenerate with
> `main.py --mode generate_accuracy_baselines --platform-target trn3`
> before Round-1 closeout.**

Each of the 5 runs invokes `main.py --mode evaluate_single --platform-target trn3
--enable-nki --prompt <N> --base-latency <row[3]> --base-throughput <row[4]>`
so `calculate_score(...)` at `main.py:836` uses the per-prompt baseline
from the stub file.



### Marginal-value table (partial — e2e constant unknown)

The scoring coefficient in `Δ_score = 2 × (current_speedup / current_time_ms) × Δμs × NKI_ratio`
needs e2e numbers we do not have yet. What we can populate:

| Lever | Bottleneck targeted | Δμs estimate (per token-gen step) | Derivation / caveat |
|------:|:--------------------|----------------------------------:|:---------------------|
| **[H]** Multi-layer fused TKG megakernel | Inter-op Python/NxDI round-trip × 48 | **UNKNOWN** pending e2e | Needs CC-core 8 gap analysis to measure per-layer boundary μs. Hypothesis from v12i prior logs: ~10–40 μs/layer → 0.5–2.0 ms total. Load-bearing lever iff gap is observable. |
| **[G]** Cross-layer expert prefetch | v14a MoE DMA=52.4 μs, all indirect-DGE with GpSimd stall budget 13.4 μs | Per-layer MoE DMA × 0.97 ≈ **~50.8 μs/layer** if prefetch overlaps with preceding attention | Feasible because v14a weight loads are all fat (16× 256KiB–1MiB) — low-latency prefetch. Needs e2e to confirm attention is long enough to cover the DMA. |
| **[K]** sb2sb AR on post-attention | cc_op_active_time (unknown without e2e) | **UNKNOWN** | Needs AR duration from CC-core gaps. Per skill's `layer_latency.md`, trn2 QwenMoE had ~75 μs AR per layer in steady state; trn3 expected similar to lower. |
| **[X1]** `nisa.exponential` in attention softmax | v13bc VectorE active 27.0 μs includes softmax exp | Softmax exp ≲ 30% of VectorE × 0.75 speedup ≈ **~6 μs/layer on attention** | Conservative bound; `nisa.exponential` is 4× faster than `nisa.activation(op=nl.exp)`. Verify with per-opcode breakdown before and after. |
| **[L]** On-device sampling | Inter-step CPU gap (unknown without e2e) | **UNKNOWN** | Measured as inter-token latency minus device time. Dominates low-layer-count configs; less impactful at 48 layers. |
| **[E]** LM head as NKI | LM head matmul time (unknown without e2e) | **UNKNOWN** | LM head is a single [H]×[H,V] matmul. With V=152K tokenizer, this is large. Needs e2e per-op split to quantify. NKI_ratio boost is the other half of the win. |
| **[D]** Attention W8A8 via `quantize_mx` | Attention HBM read 9.5 MB = (Wq+Wk+Wv+Wo BF16 + KV_cache BF16) | KV_cache bytes × 0.5 ≈ **~0.5 μs/layer** at 322 GB/s effective | KV cache at S_prior=640 is only ~320 KB/layer — halving yields ~160 KB, tiny save. Weight quantization to FP8 (Wq/Wk/Wv/Wo) would be larger but runs into trn3's `tensor_partition_reduce(maximum)` restriction per earlier finding. |

**Ordering by known Δμs (ignoring the unknowns)**: `[G]` > `[X1]` > `[D]`.
The unknowns `[H]`, `[K]`, `[L]`, `[E]` are plausibly the largest but
cannot be ranked without e2e measurement.

### Surprises vs v14a declared 88.68 μs / 57% DMA baseline

1. **GpSimdE is the real stall king, not DMA**. DMA active 59.5% is
   *utilization*, not *stall*. When you look at `evt_wait_time` (who is
   waiting on whom), GpSimd carries 13.4 μs — more than all other engines
   combined (6.7 μs total). The indirect-DGE pattern for
   `scalar_offset=expert_id` serializes GpSimd against every per-expert
   load. That's the highest-leverage per-kernel target inside v14a.
2. **VectorE is 46.6%, higher than TensorE's 35.3%**. The trn3 harness was
   written to flag this as "quantize-bound," but v14a does no on-device
   quantization — the VectorE load is RMSNorm + SiLU + topk softmax +
   gate×up + the final expert combine. `[X1]` (`nisa.exponential`) is a
   direct swap on part of this path; cheap opt worth taking.
3. **v14a saves 0.9 μs vs v12i on the same NEFF class** — smaller than
   the declared 1.3%. Within hardware noise; real improvement is marginal.
4. **No skinny DMAs, no SBUF spill, no engine idle gaps**. The v12i/v14a
   scheduler is extremely clean. Opportunities are structural (cross-layer,
   engine-mix) not micro (tile size, skinny DMA).
5. **The matmul engine count is 1,332 pairs of (LDWEIGHTS, MATMUL)**. This
   is *enormous* for a ~88 μs kernel. MFU at 0.003% confirms each is a
   T=1 skinny matmul. No amount of per-matmul FLOP optimization changes
   anything — only bytes-moved matters, exactly as the T=1 reframing
   predicted.

---

## Phase 0 — Harness blockers (close-out)

Blocker chain encountered attempting to run `main.py --mode evaluate_single --platform-target trn3 --qwen qwen_complete` on prompts 0-4 with stubbed trn2 baselines.

| # | Blocker | Status | Fix / note |
|---|--------|--------|-----------|
| 1 | `@nki.jit(platform_target="trn2")` kwarg rejected by installed nki SDK (`ValueError: platform_target parameter is deprecated`). Fires at router import (`kernels/router_topk/qwen3_router_topk_plan_a.py:83`). | **RESOLVED** | Stripped the kwarg. `NEURON_PLATFORM_TARGET_OVERRIDE=trn3` is the single source of truth now. |
| 2 | `qwen_complete.py:86` imported `kernel_v19b` from `kernels.moe_fused_tkg` but file is at `kernels/moe_fused_tkg/versions/`. Non-importable since commit `e934cb0` (2026-04-01). | **RESOLVED** | Fixed import path → `.versions.kernel_v19b`. `python3 -c "import qwen_complete"` PASS. |
| 3 | `NotImplementedError: _call_shard_hidden_kernel is not available` at `NxDI blockwise.py:269` during CTE trace. `neuronxcc.nki._private.blockwise_mm` is absent in this SDK venv (the `.so` exists at `_private_kernels/` but the specific function `blockwise_mm_baseline_shard_hidden` is not found across all 3 candidate import paths). | **RESOLVED (workaround)** | Bumped `BlockwiseMatmulConfig.block_size` 128→8192 in `qwen_complete.py:107`. Dispatcher `expert_mlps_v2.py:1495-1498` now routes CTE through `forward_all_experts` (plain PyTorch matmul, no private kernel) because `total_tokens × top_k = 402 × 8 = 3216 < 8192`. CTE slower but compiles. |
| 4 | `RuntimeError: BIR emission failed` when NKI compiler lowers `kernel_v19b.qwen3_moe_fused_tkg` at `qwen_complete.py:641` during TKG trace. Failed identically for all 4 attempted prompts (~45-48 s each). `kernel_v19b` is a trn2-authored NKI kernel that does not lower on trn3 with the current SDK. | **UNRESOLVED** | No Phase 0 fix. E2E numbers deferred. |

**FINDING**: No `qwen_*` module in this repo currently end-to-end compiles on trn3. A "current production baseline" does not exist on trn3 as of 2026-04-21. The baseline Phase 0 was meant to anchor against is a phantom.

Consequence: Phase 0 metrics that require e2e (TTFT, tok/s, per-prompt `NKI_FLOP_Ratio`, `score`, per-layer AllReduce timing from CC-core trace, inter-op Python/NxDI round-trip μs) are **UNMEASURED** and cannot be produced with the current SDK without Phase 1 work.

---

## Phase 1 matrix candidate `[Z]` — expanded

**OLD**: `[Z] Wire v14a + v13bc into qwen_complete (no megakernel)` — lower-risk intermediate between baseline and `[H]` multilayer megakernel.

**NEW** (scope grew based on the Phase 0 harness chain):

> **`[Z] Build a shippable trn3 baseline module from scratch`**:
> a functional `qwen_complete.py` that compiles and runs e2e on trn3 today.
> - TKG MoE: `v14a` (confirmed trn3-compiling in Phase 0 isolated bench).
> - TKG attention: `v13bc_sbm_tiled` (confirmed trn3-compiling in Phase 0 isolated bench).
> - CTE MoE: `forward_all_experts` via `BlockwiseMatmulConfig(block_size=8192)` (confirmed working with Phase 0 harness fix).
> - CTE attention: stock (unblocked path in Phase 0 work).
> - No megakernel integration. No multi-layer fusion.
> **Deliverable**: *first* trn3-e2e-compiling NKI-enabled qwen module.
> **Prerequisite for**: every other lever with a real `Δscore`. Without `[Z]`, all e2e numbers remain UNMEASURED and no lever can be benched against a real baseline.

---

## Marginal-value table — Phase 0 close-out (estimated constants)

### Inputs used

| Constant | Value | Source | Label |
|---|---|---|---|
| `v14a_μs` (MoE TKG per-layer, isolated) | **88.06** | Phase 0 Step 1, this log | **MEASURED** |
| `v13bc_μs` (attn TKG per-layer, isolated) | **53.71** | Phase 0 Step 1, this log | **MEASURED** |
| `AR_μs` (per-layer AllReduce) | **~12** | v12i trn2 log (`OPTIMIZATION_LOG.md`) | **ESTIMATE, from trn2** |
| `gap_μs` (per-layer Python/NxDI round-trip) | **~25** | Not captured — would require e2e NTFF CC-core trace | **PLACEHOLDER — TO MEASURE** |
| `post_transformer_μs` (final RMSNorm + LM head + sampling) | **~300** | Estimate for Qwen3-30B-A3B LM head (stock, no NKI) | **PLACEHOLDER** |
| `N_layers` | **48** | Qwen3-30B-A3B config | **MEASURED** |
| `NKI_FLOP_Ratio` | **0.80** | User-supplied placeholder for ranking only | **PLACEHOLDER** |
| `current_speedup` | **1.00** | Placeholder (no e2e number) | **PLACEHOLDER** |

### Implied e2e per-token inference time

```
implied_e2e_μs = (v14a + v13bc + AR + gap) × 48 + post_transformer
               = (88.06 + 53.71 + 12 + 25) × 48 + 300
               = 178.77 × 48 + 300
               = 8581 + 300
               = 8881 μs ≈ 8.88 ms per token
```

**Bottleneck decomposition per layer** (178.77 μs total):
- MoE TKG: **88.06 μs** (49.3%)
- Attention TKG: **53.71 μs** (30.0%)
- AllReduce: **12 μs** (6.7%) — estimated
- Inter-op gap: **25 μs** (14.0%) — placeholder, **TO MEASURE**

### Δscore formula

Per user:
```
Δ_score = 2 × (current_speedup / current_time_ms) × Δμs × NKI_ratio
        = 2 × (1.00 / 8.88) × Δμs × 0.80
        = 0.180 × Δμs_per_token
```

All Δscore values below are **relative ranking placeholders** — they become real only after `[Z]` lands or you run `generate_accuracy_baselines --platform-target trn3` to replace the estimated constants.

### Table

All Δμs are **per-token** (per-layer × 48 where applicable). Δscore calculated via formula above.

| Rank | # | Lever | Δμs/layer | Δμs/token | Δscore | Tier | Notes |
|---:|:---|:---|---:|---:|---:|:---|:---|
| 1 | **[Z]** | Build shippable trn3 baseline module | — | — | **∞ (gate)** | **Prerequisite** | Without [Z], accuracy=0 → score=0. Every other Δscore is meaningful only after [Z]. |
| 2 | **[G]** | Cross-layer expert prefetch (Fate) | 50.8 (≈MoE DMA × 0.97) | 2438 | **439** | Load-bearing | Overlap v14a weight loads with preceding attention. Needs ring-buffer + predictor. |
| 3 | **[H]** | Multi-layer fused TKG megakernel | 25 (=gap) | 1175 (×47 gaps) | **211** | Load-bearing | Removes inter-layer Python/NxDI round-trip. gap_μs=25 is a PLACEHOLDER — could be 10 or 50 in reality. |
| 4 | **[L]** | On-device sampling (`--on-device-sampling`) | — (outside per-layer) | ~1000 (estimate) | **180** | Load-bearing | Saves inter-step CPU→device round-trip. Δμs is a generous estimate for Qwen3-30B; real value depends on prompt generation pattern. |
| 5 | **[F2]** | Direct DGE for weight loads (dge_mode=0→1) | 10 (GpSimd stall) | 480 | **86** | Load-bearing | Directly attacks the 13.4 μs GpSimd stall budget in v14a (the Phase 0 bottleneck finding). Trn2 had a compiler block; retest on trn3. |
| 6 | [K] | Symmetric `_sb2sb_all_reduce_gather` on post-attn AR | 6 (½ current AR) | 288 | **52** | Cleanup | AR_μs is estimated from trn2. Retest on trn3. |
| 7 | [X1] | `nisa.exponential` in attention softmax | 6 (VectorE) | 288 | **52** | Cleanup | 4× VectorE throughput for exp on trn3. Direct API swap in attention kernel. |
| 8 | [F3] | Weight+scale DMA packing | 6 | 288 | **52** | Load-bearing | Cuts trigger count per expert from 5 to 2-3. Requires HBM weight-layout change (integration risk). Substitutes with [F4]. |
| 9 | [F4] | Scale preloading via dge_mode=3 | 5 | 240 | **43** | Cleanup | Preload all-expert scales (~288 KB) once per forward pass. Trn2 had compiler block; retest on trn3. Substitutes with [F3]. |
| 10 | [E] | LM head as NKI kernel (+on-device argmax) | — (outside per-layer) | 150 (estimate: halve current) | **27** | Load-bearing | Also lifts `NKI_FLOP_Ratio` — linear score boost independent of Δμs. True Δscore is higher than 27. |
| 11 | [Y1] | FP8 KV cache with fused dequant in attention | 1.5 | 72 | **13** | Cleanup | KV cache small (~320 KB/layer at S=640). |
| 12 | [D] | Attention W8A8 via `quantize_mx` | 0.5 | 24 | **4** | Cleanup | Smallest single-lever save. Only worth doing if bundled with [Y1]. |
| — | [J] | CC-pipeline tiling factor | ? | ? | **UNMEASURED** | — | Gated on [H] being landed. Retry `--cc-pipeline-tiling-factor=2/4` under 48-layer AR. |
| — | [I] | `@nki.compiler.skip_middle_end_transformations` | ? | ? | **UNMEASURED** | — | Toggle probe on MoE + attn. Low-cost to try. |
| — | [X2] | Background TensorE transpose | ? | ? | **UNMEASURED** | Cleanup | Small (1-2 μs/layer). Applies to attention QKV reshape paths. |
| — | [X3] | `nisa.activation2` fused scale+bias+reduce | ? | ? | **UNMEASURED** | Cleanup | RMSNorm / softmax path. |
| — | [F] | Audit HLO for non-NKI dots; convert biggest | ? | ? | **UNMEASURED (NKI_ratio)** | Load-bearing | Raises `NKI_FLOP_Ratio` linearly. Biggest unknown in score. |
| — | [M] | `causal_lm_async_execution` | — | ? | **UNMEASURED** | Cleanup | Only if [L] is on. |
| — | [A] | MXFP4 expert weights via `nc_matmul_mx` | ? | ? | **STRETCH** | Stretch | Gated on 10-line microbench showing ≥1.5× speedup for `[128P, I_tile] × [128P, T=1]`. Phase 0 skill note + T=1 reframing predict **this will fail the gate** — do it anyway. |
| — | [N] | EAGLE / fused speculation | ? | ? | **STRETCH** | Stretch | 2-2.5× ceiling, high integration cost. |

**Sanity check**: the Δscore values above are relative only. `current_speedup=1.00` and `current_time_ms=8.88` ms are both fabrications (we never observed either). When [Z] lands and produces real numbers, recompute every Δscore with the actual `score / time` scaling factor.

### Top-5 levers (ranked, Phase 0 close-out view)

1. **[Z]** — Build shippable trn3 baseline module (**prerequisite**; score gate).
2. **[G]** — Cross-layer expert prefetch (**Δscore ≈ 439**; attacks v14a's 52 μs DMA per layer).
3. **[H]** — Multi-layer fused TKG megakernel (**Δscore ≈ 211**; depends on gap_μs measurement).
4. **[L]** — On-device sampling (**Δscore ≈ 180**; eliminates inter-step CPU gap).
5. **[F2]** — Direct DGE for weight loads (**Δscore ≈ 86**; direct attack on Phase 0's GpSimd stall finding).

All five become real numbers only after `[Z]` lands or after `main.py --mode generate_accuracy_baselines --platform-target trn3` is run out-of-band to produce the trn3 baseline constants (`current_time_ms`, real `NKI_FLOP_Ratio`, real per-prompt `base_latency`/`base_throughput`).

---

## Round 1 — Side probe [F4] result (2026-04-21)

**Probe**: `probe_f4_sbuf_dyn_idx.py` — 40-line trn3 NKI test; SBUF→SBUF `nisa.tensor_copy` with `scalar_offset=<SBUF-resident int32>`. Trn2 raised `NCC_INLA001` on this pattern (`OPTIMIZATION_LOG.md` Round 7 Plan M).

**Trn3 result**: **FAIL with `[NCC_IBIR010] Requested Argument index 1 out of bounds (1)`** at `nisa.tensor_copy(...)` — different error code than trn2, reached a deeper codegen stage before rejecting.

**Interpretation**: inconclusive. Trn3 did not raise `NCC_INLA001` — the "dynamic access" constraint may be lifted, but my 40-line probe's `scalar_offset` on a `tensor_copy`-`.ap()` combination failed a downstream bounds check. Resolving this would take a dedicated investigation session beyond the 5-min budget.

**Action taken**: treat `[F4]` as blocked for Round 1; revisit if `[F2]` + `[F3]` leave residual GpSimdE stall budget that a scale-preload-only path could clear. Not promoted to Round 2.

---

## Round 1 — Plan A [F2] Direct DGE for weight loads — **v15a.py**

**Implementation**: copy of `v14a.py` with the 10 indirect-DGE expert DMAs reclassified:
- 4 **scale** DMAs (gate_up_scales × 2 waves + down_scales × 2 waves, lines 444/482/512/550) → **`nisa.dge_mode.hwdge`** (HWDGE = hardware-DGE, off the GpSimd path; fp32→fp32, no dtype cast required).
- 6 **weight** DMAs (gate_up_w + down_w tiles 0 & 1, × 2 waves, lines 431/456/469/499/524/537) → **explicit SWDGE** (`dge_mode=1`). HWDGE rejected these with `"HWDGE requires same src/dst element type, got src='i8' vs dst='f8E4M3'. Use SWDGE or NoDGE for type casting."`

**Deviation from Plan A literal spec**: the subagent did NOT precompute `expert_id × per_expert_stride` into an SBUF offset table. Reason: `dge_mode=0` vs `dge_mode=1` are equivalent for `scalar_offset`-bearing loads (both route through SWDGE/GpSimd on this SDK). The real lever was HWDGE on the scale DMAs. Precomputing an offset still requires a GpSimd multiply producing a value GpSimd reads — net-zero for the stall path. Documented in v15a.py.

**Correctness**: **PASS** max_diff=0.00e+00 (bit-exact vs v14a at rtol=1e-3, atol=1e-3, seed=42).

**Benchmark results (v15a, trn3, one wrap_benchmark warmup=5 iters=50)**:

| Metric | v14a (fresh) | v15a | Δ |
|---|---:|---:|---:|
| device_time_us | 88.36 | **76.38** | **−11.98 (−13.6%)** |
| tensor_engine_pct | 35.35 | 39.94 | +4.59 pp (same abs time, smaller denom) |
| vector_engine_pct | 45.20 | 50.78 | +5.58 pp (same) |
| scalar_engine_pct | 22.71 | 24.86 | +2.15 pp (same) |
| **gpsimd_engine_pct** | **44.60** | **35.05** | **−9.55 pp** |
| dma_active_pct | 57.73 | 53.33 | −4.40 pp |
| spill_bytes | 0 | 0 | 0 |
| hbm_read_KiB | 16488.0 | 16488.0 | 0 |
| mfu_estimated_pct | 0.003 | 0.0036 | +0.0006 |

**Hypothesis validation**: Phase 0 predicted GpSimdE stall budget 13.4 μs → <2 μs (save ~10 μs). **Observed**: GpSimd active time went 44.60 × 88.36 = 39.4 μs → 35.05 × 76.38 = 26.8 μs, a **−12.6 μs reduction in GpSimd active time**. Total device time dropped **−11.98 μs**. The two deltas nearly match — GpSimd was on the critical path, as predicted. ✓

**Remaining bottleneck**: still GpSimd + DMA (35% + 53%). The 6 weight DMAs stay on SWDGE because of the i8→f8 dtype-cast restriction. **Follow-up [F2b] for Round 2**: load weights as `int8` into SBUF, view-reinterpret as `float8_e4m3` at matmul consumption — would unblock HWDGE on all 10 expert DMAs, potentially another double-digit % win.

**Files shipped**:
- `kernels/moe_fused_tkg/quantized/v15a.py`
- `kernels/moe_fused_tkg/quantized/test_v15a.py`
- `kernels/moe_fused_tkg/quantized/bench_v15a_trn3.py`

**Tier**: **Load-bearing, kept.** >2% improvement, zero accuracy cost, zero integration risk (kernel-only). Plan A is the new MoE TKG best at 76.38 μs.

---

## Round 1 — Plan B [X1] `nisa.exponential` in v13bc softmax — **v17_fast_exp.py**

**Implementation**: `kernels/attn_tkg/agents/v17_fast_exp.py` — copy of `v13bc_sbm_tiled.py`. Two `nisa.activation(op=nl.exp)` call sites replaced with `nisa.exponential`:
- Pass 2 per-tile softmax exp
- Active-position exp

Kept the existing `scores + neg_max` subtract (required because `nisa.exponential`'s `max_value=` expects `(P, 1)` partition-broadcast, but this kernel's per-head max varies along the GQA free dim — the shape doesn't fit). Fallback to `nisa.exponential(dst, data=score_shifted)` with default `max_value=0.0`.

**Correctness**: **PASS** max_abs_err=1.65e-02 (rtol=1e-2, atol=2e-2 vs PyTorch float32 ref; algebraically identical to v13bc).

**Benchmark results (mean of 3 runs)**:

| Metric | v13bc (mean 3 runs) | v17_fast_exp (mean 3 runs) | Δ |
|---|---:|---:|---:|
| device_time_us | 53.65 | **53.28** | **−0.37 (−0.7%)** |
| tensor_engine_pct | ~45.5 | 46.6 | +1.1 pp |
| vector_engine_pct | 50.4 | 50.7 | +0.3 pp (~flat) |
| **scalar_engine_pct** | **14.8** | **12.0** | **−2.8 pp** (confirms exp moved off ScalarE) |
| gpsimd_engine_pct | ~6 | 8.3 | +2.3 pp |
| dma_active_pct | ~46.6 | 46.2 | −0.4 pp |
| spill_bytes | 0 | 0 | 0 |
| hbm_read_KiB | 9541 | 9542 | +1 |

**Hypothesis validation**: Phase 0 predicted ~6 μs drop from 4× VectorE throughput. Observed: **−0.37 μs only**. VectorE% did NOT drop — the 4× throughput gain on VectorE was offset by exp work *shifting onto VectorE* (trn3's stock `nisa.activation(op=nl.exp)` apparently ran partially on ScalarE). Net exp cost: slightly cheaper, but the savings are small because exp wasn't the bottleneck. The real bottleneck is multi-engine-balanced (TensorE, VectorE, DMA all near 45-50%). ✓ partial — hypothesis direction correct, magnitude overestimated.

**Remaining bottleneck**: balanced across engines — no single dominator. Further per-engine wins diminishing. Structural changes (HBM traffic reduction on Wq/Wo weight reloads, transpose-count reduction in the scores path) are needed to break below ~50 μs.

**Files shipped**:
- `kernels/attn_tkg/agents/v17_fast_exp.py` (exported symbol: `qwen3_attn_tkg_fused_oproj_v13bc`, preserved for bench compat)
- `kernels/attn_tkg/agents/bench_v17_fast_exp.py`

**Tier**: **Cleanup, kept.** Sub-2% gain, zero accuracy cost, zero integration risk. Not a headline win but not a regression. Compose cleanly with Plan A in any future integrated bench.

> ⚠️ **POST-HOC CORRECTION (Phase 3 STEP 1a, 2026-04-21)**: `v17_fast_exp` produces
> all-zero output. `nisa.exponential` swap silently corrupts the kernel. The
> 53.28 μs "PASS" was a false-positive from loose `rtol=1e-2, atol=2e-2`
> masking zero-vs-small-ref. See `docs/integration_findings.md` Finding 1.
>
> **Round 1 Plan B was not actually a -0.7% win** — it was a broken kernel
> producing zeros. The real trn3 attention TKG baseline remains `v13bc_sbm_tiled`
> at 53.71 μs (verified non-zero by `probe_v13bc_passthrough.py`).
>
> **Required going forward**: every correctness test MUST use the triple-check
> pattern (zero-check + range-check + allclose) — `rtol/atol` alone cannot
> detect zero output when the reference magnitude is smaller than `atol`.

---

## Round 1 — Plan C [F3] Weight+scale DMA packing + int8-reinterpret — **v15c.py** ⭐

**Discovery this round**: the int8-reinterpret HWDGE trick works on trn3. Subagent packed `gate_up_w [i8]` + `gate_up_scales [fp32]` into a single HBM blob, loaded as int8 via `dge_mode=nisa.dge_mode.hwdge` (no type cast → HWDGE accepts), then used `NkiTensor.view(nl.float8_e4m3)` for the weight slice and `NkiTensor.view(nl.float32)` for the scale slice at consumption. Compiler accepted end-to-end. This SUBSUMES Plan A's HWDGE-on-scales win because the merged DMA is already on HWDGE.

**Implementation**: `kernels/moe_fused_tkg/quantized/v15c.py` — copy of v14a.py with:
- Packed HBM layout: `gate_up_packed_w[E=128, H_free+1=17, PMAX=128, GU_FLAT=384] int8`. Planes 0..15 = weight bytes. Plane 16 = fp32 scales as raw bytes (first 3 fp32 slots, zero-padded). Per-expert HBM: 786432 → 835584 bytes (+6.25%).
- Merged gate_up DMA (Wave 0 + Wave 1) on HWDGE with int8-reinterpret.
- `down_scales` DMAs kept on HWDGE (same as v15a pattern).
- `down_w` tiles stay on SWDGE (still have i8→f8 cast; not packed in v15c — natural Round 2 follow-up).
- All v14a opts preserved: SBUF hoisting, affine_range tile-1 prefetch, ring buffers, pre-read of 8 expert IDs, combined-scale precompute.

**Scope constraint honored**: no state-dict converter / qwen_complete changes. Bench generates fake weights in the packed layout directly.

**Correctness**: **PASS** max_diff=0.00e+00 (bit-exact vs v14a, rtol=1e-3, atol=1e-3, seed=42).

**Benchmark results (v15c, trn3)**:

| Metric | v14a (Phase 0) | v15c | Δ vs v14a |
|---|---:|---:|---:|
| device_time_us | 88.06 | **72.88** | **−15.18 (−17.2%)** |
| tensor_engine_pct | 35.3 | 41.75 | +6.45 pp (shrinking denom) |
| vector_engine_pct | 46.6 | 55.08 | +8.48 pp (same) |
| scalar_engine_pct | 22.9 | 28.13 | +5.23 pp (same) |
| **gpsimd_engine_pct** | **45.6** | **30.47** | **−15.13 pp** (absolute GpSimd time 40.2 μs → 22.2 μs = −18 μs) |
| dma_active_pct | 59.5 | 56.58 | −2.92 pp |
| spill_bytes | 0 | 0 | 0 |
| hbm_read_KiB | 16488 | 17232 | +744 (scale-plane padding overhead, expected) |
| hbm_write_KiB | 4 | 4 | 0 |

**Hypothesis validation**: Phase 0 matrix predicted Δμs ≈ 6/layer (288/token), Δscore ≈ 52. **Actual: Δ−15.18 μs/kernel, ~2.5× the prediction.** The int8-reinterpret trick delivered both DMA-count reduction AND HWDGE offload simultaneously. ✓ and then some.

**Remaining bottleneck**: VectorE (55%, ~40 μs absolute). DMA still 57% but with GpSimd freed up, descriptor-pipeline wins have diminishing returns. Next levers target:
- VectorE reduction on post-matmul activation chain (gate_silu + up_f32_scaled + tensor_tensor multiply per expert)
- Apply int8-reinterpret HWDGE pattern to the 4 remaining `down_w` SWDGE DMAs → **natural Round 2 candidate [F3b]**

**Files shipped**:
- `kernels/moe_fused_tkg/quantized/v15c.py`
- `kernels/moe_fused_tkg/quantized/test_v15c.py`
- `kernels/moe_fused_tkg/quantized/bench_v15c_trn3.py`

**Tier**: **Load-bearing, Round 1 winner.** Largest measured improvement; bit-exact correctness; dominates Plan A on the same baseline. Discovered a reusable technique (int8-reinterpret HWDGE) that applies to other weight-load sites.

---

## Round 1 Synthesis

### Baseline
- v14a (Phase 0): **88.06 μs** — MoE TKG best going into Round 1.
- v13bc (Phase 0): **53.71 μs** — attention TKG best going into Round 1.
- Bottleneck: DMA 59.5% + GpSimdE 45.6% (v14a); VectorE 50.3% (v13bc).

### Results table

| Plan | Kernel | Target | device_time_us | tensor% | vector% | scalar% | gpsimd% | dma% | spill | mfu% | vs baseline | Tier | Notes |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| — | v14a | MoE baseline | 88.06 | 35.3 | 46.6 | 22.9 | 45.6 | 59.5 | 0 | 0.003 | — | — | Phase 0 baseline |
| A | v15a | MoE | 76.38 | 39.9 | 50.8 | 24.9 | 35.1 | 53.3 | 0 | 0.004 | **−13.3%** | Load-bearing kept | HWDGE on 4 scale DMAs; 6 weight DMAs stuck on SWDGE |
| — | v13bc | Attn baseline | 53.71 | 45.5 | 50.3 | 14.8 | 5.8 | 46.6 | 0 | 0.005 | — | — | Phase 0 baseline |
| ~~B~~ | ~~v17_fast_exp~~ | ~~Attn~~ | ~~53.28~~ | ~~46.6~~ | ~~50.7~~ | ~~12.0~~ | ~~8.3~~ | ~~46.2~~ | ~~0~~ | ~~—~~ | ~~−0.7%~~ | **FALSE POSITIVE** | **Plan B was a false positive — v17_fast_exp silently produces all-zero output (see `docs/integration_findings.md` Finding 1). The 53.28 μs is the speed of computing nothing. Attention TKG baseline unchanged at `v13bc_sbm_tiled` 53.71 μs. Real Round 1 net win = MoE TKG only (−17.2% via Plan C).** |
| C ⭐ | v15c | MoE | **72.88** | 41.8 | 55.1 | 28.1 | **30.5** | 56.6 | 0 | — | **−17.2%** | **Load-bearing winner** | int8-reinterpret HWDGE on packed gate_up DMA; dominates A on same baseline |

### Analysis

1. **Plan A + Plan C are alternatives, not complements.** Plan A moves the 4 scale DMAs to HWDGE; Plan C packs gate_up weight+scales so those 2 gate_up_scales DMAs disappear entirely (merged into one HWDGE-eligible int8-load). Plan C's DMA count goes from 5 to 4 per expert; Plan A's DMA count stays 5. Both improve over v14a; Plan C improves more. **Do not stack — pick Plan C.**
2. **Plan B is small.** Hypothesis direction correct (exp moved off ScalarE, confirmed by −2.8pp ScalarE%). Magnitude overestimated because exp wasn't the actual VectorE bottleneck — the post-matmul activation chain is. Plan B is cleanup-tier but still kept (no regression).
3. **Biggest surprise**: the int8-reinterpret HWDGE trick unlocked the weight DMA path. The subagent discovered this while probing Plan C and validated it: int8 load → `.view(fp8)` at matmul consumption is compiler-accepted. This reusable technique applies to every weight-load site currently stuck on SWDGE due to dtype cast. The remaining 4 `down_w` SWDGE DMAs in v15c are the obvious next target.
4. **Phase 0 predictions vs observed**:
   - Plan A [F2]: predicted Δ−9.6 μs (save most of 13.4 μs stall). Observed: −11.98 μs. ✓ close
   - Plan B [X1]: predicted Δ−6 μs. Observed: −0.37 μs. ✗ overestimated 16×
   - Plan C [F3]: predicted Δ−6 μs (packing only). Observed: −15.18 μs. ✓ but for a different reason than predicted (HWDGE via int8-reinterpret, not just descriptor-count reduction)
5. **GpSimdE stall finding held up**: Plan A dropped GpSimd by 9.55pp, Plan C by 15.13pp. The Phase 0 evidence-grounded claim was the right target.

### Single winner for Phase 3 integration

**v15c** (Plan C). 72.88 μs, bit-exact, lowest risk given the simple kernel-level change + bench-only input packing. The state-dict converter change (packed gate_up layout) is a ~1-day extension to `[Z.1+Z.2]` integration work already dispatched out-of-band.

### Recommended top-3 for Round 2

All three are isolated-kernel candidates (independent of `[Z.1]`), ordered by expected Δscore:

1. **[F3b]** Apply int8-reinterpret HWDGE to the 4 remaining `down_w` DMAs in v15c.
   - **Rationale**: the trick is now proven. Four more SWDGE→HWDGE conversions should drop GpSimdE further (from 30.5% → ~20%?) and take another 3–5 μs off v15c. Cheapest, highest-confidence Round 2 candidate. Ship as `v15d.py`.

2. **[X3]** VectorE attack — fuse the post-matmul activation chain (gate_silu + up_f32_scaled + tensor_tensor multiply) via `nisa.activation2` where possible.
   - **Rationale**: v15c's new bottleneck is VectorE at 55% (40 μs absolute) from the activation chain. This is the #1 remaining engine cost. Phase 1 matrix had [X3] as Cleanup-tier; Plan C has promoted it to the next Load-bearing target on v15c.

3. **[G]** Cross-layer expert prefetch prototype on v15c.
   - **Rationale**: the Phase 1 matrix #1 Δscore candidate (439). Can be prototyped in isolation against v15c — the MoE DMA time that [G] hides is v15c's ~41 μs (estimate, from `dma_active_pct × device_time`). If the prefetch covers most of that during preceding attention, this is the single largest remaining kernel-level win before the megakernel. Higher risk than [F3b] and [X3].

**Deferred to Round 3+** (need [Z.1] to land first): [H] megakernel, [Z.2] integrated shipping, [E] LM head NKI, [F] non-NKI dot audit.

**Stopping per brief.** Awaiting your Round 2 top-3 pick.

---

# Round 2

## Round 2 — Side probe [F4] v2

Re-ran `probe_f4_v2_sbuf_dyn_idx.py` (20-line variant). **Result: same as Round 1 — `NCC_IBIR010` ("Requested Argument index 1 out of bounds"), not `NCC_INLA001`.** Trn3 compiler reaches a deeper codegen stage than trn2 before rejecting, but the SBUF→SBUF `nisa.tensor_copy(..., scalar_offset=<sbuf>)` pattern does not compile cleanly. Per user instruction: ambiguous — deferred to Round 3 as a dedicated investigation. Not promoted to Round 3 as a candidate.

## Round 2 — Plan A [X3] `nisa.activation2` fusion — **v15e.py** (regressed, kept)

**Implementation**: `kernels/moe_fused_tkg/quantized/v15e.py` — copy of v15c with two post-matmul fusion sites:
1. Per-tile up + GLU multiply: replaced `activation(up_psum, scale=up_scale)` + `tensor_tensor(gate_silu, up_scaled, multiply)` with `scalar_tensor_tensor(data=up_psum, op0=multiply, operand0=up_scale, op1=multiply, operand1=gate_silu)`.
2. Per-expert down combined-scale + multiply: replaced `tensor_tensor(down_scale, aff, multiply)` + `tensor_tensor(down_h_raw, combined, multiply)` with single `scalar_tensor_tensor`.

**API note**: `nisa.activation2` does not exist in this NKI release. Subagent used `nisa.scalar_tensor_tensor` (computes `(data op0 operand0) op1 operand1` in one VectorE instruction).

**Correctness**: **PASS** max_diff=3.81e-06 (rtol=1e-3, atol=1e-3 vs v15c).

**Benchmark results (5-run means)**:

| Metric | v15c (Round 1 winner) | v15e | Δ |
|---|---:|---:|---:|
| device_time_us (5-run mean) | 72.2 | **73.3** | **+1.1 (+1.4%) — REGRESSION** |
| vector_engine_pct (5-run mean) | 54.4 | 52.4 | **−2.0 pp** (hypothesis direction correct) |
| scalar_engine_pct | 28.1 | 26.5 | −1.6 pp |
| gpsimd_engine_pct | 30.5 | 25.3 | −5.2 pp |
| dma_active_pct | 56.6 | 57.6 | +1.0 pp |
| spill_bytes | 0 | 0 | 0 |
| hbm_read_KiB | 17232 | 17232 | 0 |

**Hypothesis validation**: VectorE% dropped as predicted (−2.0pp, ~1.5 μs absolute), but total device time increased slightly. The compiler had already been running the per-expert ScalarE `activation(copy)` PSUM drain in parallel with VectorE `tensor_tensor` multiplies across iterations. Fusing them onto VectorE via `scalar_tensor_tensor` reduced cross-engine parallelism. Net: raw instruction count ↓, effective parallelism ↓, wall-clock roughly flat (slight regression).

**Remaining bottleneck**: VectorE ~52% + DMA ~58% co-dominant. Further VectorE work reduction is not cheap because the compiler is already pipelining it well.

**Files shipped**:
- `kernels/moe_fused_tkg/quantized/v15e.py`
- `kernels/moe_fused_tkg/quantized/test_v15e.py`
- `kernels/moe_fused_tkg/quantized/bench_v15e_trn3.py`

**Tier**: **Regression, kept per <2% rule.** Within the no-revert threshold. NOT recommended for Phase 3. Useful lesson: on v15c, compiler pipeline is so tight that reducing op-count at the cost of cross-engine parallelism is net-neutral-to-negative. Future VectorE attacks need to REDUCE total VectorE work, not collapse it onto fewer instructions at the same total cost.

---

## Round 2 — Out-of-band [Z.1+Z.2] — integrated into qwen_complete (ready for e2e test)

**Scope delivered** (background subagent):
- `qwen_complete.py` imports swapped: `v19b → v15c`, `v10e → v17_fast_exp` (via `qwen3_attn_tkg_v17_wrapper` to preserve the `[B, 1, H_wo]` output shape).
- New utility module `kernels/moe_fused_tkg/quantized/_qwen_integration.py` containing:
  - `quantize_and_pack_gate_up(bf16)` — per-expert per-output-neuron FP8 abs-max quant → calls `v15c.pack_gate_up`.
  - `quantize_down(bf16)` — per-expert per-H FP8 abs-max quant → `(int8 bytes, fp32 scales)`. No packing yet.
  - `qwen3_attn_tkg_v17_wrapper` — `@nki.jit` wrapper that calls the v17 sub-function and per-column transposes output to match v10e's `[B, 1, H_wo]` shape.
- State-dict converter (`convert_qwen3_moe_hf_to_neuron_state_dict`) emits 3 new TKG-only buffers per layer: `gate_up_packed_w_tkg` (int8, v15c-packed), `down_w_tkg` (int8), `down_scales_tkg` (fp32). Existing bf16 `gate_up_proj`/`down_proj` retained for CTE path.
- New `_register_tkg_quant_buffers` registers the three TKG buffers on `mlp.moe_fused_tkg` with dtype-protection hooks against `model.to(bfloat16)` coercion.
- Two TODO comments at `qwen_complete.py:325-328` and `749-750` flagging the `down_w_tkg` repack required when `[F3b]` lands (user-requested).

**Import smoke test**: **PASS** (`python3 -c "import qwen_complete; print('OK')"`).

**Ready for e2e**: **yes, pending kernel queue**. Command to run once Round 2 Plans A/B/C finish:
```
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
NEURON_PLATFORM_TARGET_OVERRIDE=trn3 python3 main.py --mode evaluate_single \
  --platform-target trn3 --qwen qwen_complete \
  --prompt "<prompt>" --base-latency <X> --base-throughput <Y> \
  --model-path ~/qwen-30b-a3b/hf_model \
  --compiled-model-path ~/qwen-30b-a3b/traced_model
```

**Risks flagged by the integration agent** (5):
1. **v17 output-order restoration untested on device.** Wrapper does 16 `nc_transpose + tensor_copy + dma_copy` iterations to emit `[B, 1, H_wo]`. Arithmetic checked vs bench's `out.flatten(order="F")`, but on-device correctness unverified.
2. **Buffer dtype preservation**: the `_apply` hook restores int8/fp32 after `model.to(bfloat16)`; if NxDI traces through a different code path, these may coerce at compile time.
3. **Memory cost**: TKG buffers held in addition to bf16 params (CTE path still needs them). Roughly doubles per-layer MoE weight storage.
4. **Preshard behavior**: new buffers registered directly on `moe_fused_tkg`; if NxDI's trace/shard filters non-parameter buffers, loading may fail.
5. **`config.moe_intermediate_size`**: interpretation is post-pad; buffer sizes infer from padded tensor shape, consistent with state-dict pad logic.

---

## Round 2 — Plan B [F3b] int8-reinterpret HWDGE on down_w — **v15f.py** (REVERTED)

**Implementation**: `kernels/moe_fused_tkg/quantized/v15f.py` — copy of v15c with all 4 down_w DMA sites converted:
- SBUF allocation: `nl.float8_e4m3` → `nl.int8`
- DMA: dropped `dtype=nl.float8_e4m3` kwarg; `dge_mode=swdge` → `dge_mode=nisa.dge_mode.hwdge`
- Matmul consumption: `.view(nl.float8_e4m3)` reinterpret on int8 tile
- `nisa.memset(value=0.0)` retargeted to int8 buffer with `value=0` (int type-check)

Compiler **accepted HWDGE on all 4** — no fallback required.

**Correctness**: **PASS** max_diff=0.00e+00 (bit-exact vs v15c).

**Benchmark results (6-run process-level mean, dropping outlier)**:

| Metric | v15c (Round 1 winner) | v15f | Δ |
|---|---:|---:|---:|
| device_time_us (6-run mean) | 73.1 | **~85** | **+12.0 μs (+16.5%) — MAJOR REGRESSION** |
| tensor_engine_pct | 42.3 | 35.1 | −7.2 pp |
| vector_engine_pct | 54.7 | 48.7 | −6.0 pp |
| **gpsimd_engine_pct** | 26.8 | **8.6** | **−18.2 pp** (mechanism works as predicted) |
| dma_active_pct | 56.4 | 59.3 | +2.9 pp |
| spill_bytes | 0 | 0 | 0 |
| hbm_read_KiB | 17232 | 17232 | 0 |

**Hypothesis validation**: predicted Δ−3 to −5 μs. **Observed: +12 μs (opposite direction).**
- GpSimdE ABSOLUTE time dropped 19.5 → 7.3 μs (**−12 μs** — matches "cheap GpSimd" theory)
- DMA ABSOLUTE time grew 41.2 → 50.4 μs (**+9 μs** — hardware-queue serialization)
- Net: +12 μs regression

**Root cause (subagent analysis)**: v15c's mixed SWDGE+HWDGE schedule had SWDGE descriptor-gen on GpSimd running *in parallel with* HWDGE DMA transfer. Collapsing all DMAs onto HWDGE forced all 10 expert-load descriptors through a single hardware queue, serializing what was previously pipelined. The mechanism change was correct; the scheduling side-effect wasn't.

**Remaining bottleneck**: DMA-pipeline / data-transfer time now ~50 μs (60% of v15f). Observed engine times: TensorE, VectorE, ScalarE essentially unchanged between v15c and v15f — the regression is purely DMA-pipeline serialization cost.

**Files shipped** (kept as documented regression, NOT used in Phase 3):
- `kernels/moe_fused_tkg/quantized/v15f.py`
- `kernels/moe_fused_tkg/quantized/test_v15f.py`
- `kernels/moe_fused_tkg/quantized/bench_v15f_trn3.py`

**Tier**: **REVERTED per >2% rule.** Crossed the threshold at +16.5%. Important learning: not all DMAs benefit from HWDGE uniformly — cross-engine parallelism between SWDGE-on-GpSimd and HWDGE-on-DMA can be MORE valuable than either mechanism alone. v15c's mixed schedule is load-bearing. Future HWDGE work on down_w (if attempted) must balance pipeline parallelism, e.g., convert only 2 of the 4 down_w DMAs.

---

## Round 2 — Plan C [G] cross-layer expert prefetch prototype — **v15g.py** (skeleton, no-op)

**Implementation**: `kernels/moe_fused_tkg/quantized/v15g.py` — structurally identical to v15c, preserving ALL Round 1 optimizations. Three within-kernel reorder variants were probed (all regressed):

| Variant | Change | device_time_us | Δ vs v15c |
|---|---|---:|---:|
| 1a | Defer Wave 1 scale extract / up-scale assembly | 78.60 | **+8.7%** |
| 1b | Group DMAs by type (gate_up / down0 / down1 / scales) | 74.92 | **+3.6%** |
| 1c | Hoist only Wave 1 gate_up ahead of Wave 0 down_w | 75.20 | **+4.0%** |

Each variant reduced gpsimd% AND dma%, but tensor_engine_pct dropped correspondingly → wall time increased. v15c's compiler schedule already hits the within-kernel overlap ceiling.

**Correctness**: **PASS** max_diff=0.00e+00 (bit-exact vs v15c; skeleton has identical DMA ordering).

**Benchmark results (v15g, 5-run mean)**:

| Metric | v15c (Round 1 winner) | v15g | Δ |
|---|---:|---:|---:|
| device_time_us (5-run mean) | 73.11 | 72.48 | **−0.63 μs (−0.9%) — within noise** |
| tensor_engine_pct | 41.30 | 40.93 | −0.37 pp |
| vector_engine_pct | 55.19 | 54.19 | −1.00 pp |
| scalar_engine_pct | 28.52 | 27.67 | −0.85 pp |
| gpsimd_engine_pct | 30.43 | 30.04 | −0.39 pp |
| dma_active_pct | 57.42 | 56.67 | −0.75 pp |
| spill_bytes | 0 | 0 | 0 |
| hbm_read_KiB | 17232 | 17232 | 0 |

**Design doc produced**: `docs/plan_g_cross_layer_prefetch_design.md` — predictor interface (`pred_w: [H, E] bf16` per layer, feeds off rmsnorm output), 3-slot SBUF ring buffer (~9.6 MiB in 32 MiB budget), Phase-2b-tail splicing of speculative DMAs into the next layer's attention. Estimated effort: ~5.5 days contingent on [H] megakernel availability.

**Key finding**: **real [G] gain requires cross-layer overlap, which requires [H] megakernel**. Plan [G] cannot ship standalone in single-kernel land. The [G] design doc is a Round 3+ blueprint.

**Files shipped**:
- `kernels/moe_fused_tkg/quantized/v15g.py` (structural copy of v15c; NOT a new winner)
- `kernels/moe_fused_tkg/quantized/test_v15g.py`
- `kernels/moe_fused_tkg/quantized/bench_v15g_trn3.py`
- `docs/plan_g_cross_layer_prefetch_design.md` (Round 3+ integration spec)

**Tier**: **Infrastructure, kept.** No measurable kernel-level change. Design doc is the actual artifact; it unblocks post-megakernel [G] work. Not a Phase 3 winner — v15c remains best.

---

## Round 2 Synthesis

### Baseline
- v15c (Round 1 winner): **72.88 μs** (stated) / **73.1 μs** (this-session fresh mean) — MoE TKG best entering Round 2.
- v17_fast_exp: **53.28 μs** (Round 1) — attention TKG best.

### Results

| Plan | Kernel | device_time_us | vs v15c | Tier |
|:---|:---|---:|---:|:---|
| — | v15c (baseline) | 73.1 (mean) | — | — |
| A | v15e ([X3] `scalar_tensor_tensor` fusion) | 73.3 | **+1.4%** | Regression, kept per <2% rule, NOT recommended |
| B | v15f ([F3b] all-4-down_w HWDGE) | ~85 | **+16.5%** | **REVERTED** (>2% threshold) |
| C | v15g ([G] within-kernel reorder + design doc) | 72.48 | −0.9% (within noise) | Infrastructure skeleton; design doc shipped |

**Zero Round 2 plans produced a measurable improvement over v15c.** One outright regression reverted; two within-noise no-ops kept.

### Analysis

1. **Plan A ([X3])**: VectorE% dropped as hypothesized (−2pp, ~1.5 μs absolute), but total time unchanged. v15c's compiler already pipelines ScalarE `activation(copy)` PSUM drain in parallel with VectorE `tensor_tensor` multiplies across expert iterations. Collapsing onto `scalar_tensor_tensor` reduced cross-engine parallelism. Lesson: **v15c is cross-engine-parallel-bound, not op-count-bound**.

2. **Plan B ([F3b])**: mechanism worked as predicted (bit-exact, GpSimdE −18pp, HWDGE accepted all 4 sites), but DMA ABSOLUTE time grew +9 μs — hardware queue serialization. Lesson: **v15c's mixed SWDGE+HWDGE DMA schedule is load-bearing**. Collapsing all DMAs onto HWDGE forces single-queue serialization; the "waste" of SWDGE-on-GpSimd is actually valuable parallelism.

3. **Plan C ([G])**: three within-kernel reorderings all regressed (+3.6% to +8.7%). Lesson: **v15c hits the within-kernel overlap ceiling**. Real [G] gain requires cross-layer overlap → needs megakernel.

4. **Common thread across all 3 plans**: v15c is in a very tight local optimum. Every plausible within-kernel lever has been probed. Further isolated-kernel gains will be fractional. **The next meaningful Δ requires structural change**: megakernel ([H]), full-model HLO audit ([F]), or a full e2e measurement to redirect effort.

### Updated engine-utilization profile (v15c = Phase 3 winner)

| Engine | v15c % | v15c absolute (μs) | Status |
|---|---:|---:|:---|
| TensorE | 42 | 30.6 | Active; matmul-bound only if we add MORE matmul |
| **VectorE** | **55** | **40.1** | **Primary compute bottleneck**; post-matmul activation chain, but compiler-pipelined — reducing op count hurts (Plan A evidence) |
| ScalarE | 28 | 20.5 | Active; pipelined with VectorE |
| GpSimdE | 30 | 21.9 | DMA descriptor-gen (still ~15 μs of indirect-DGE on down_w); [F3b]-style removal causes pipeline serialization |
| DMA | 57 | 41.6 | Data-transfer time; roughly balanced with compute-active time (41 vs ~41 μs) |

**Interpretation**: v15c is **cross-engine-parallel-bound**. TensorE, VectorE, and DMA are all simultaneously active ~42-57% of the time. Reducing any single engine's load tends to shift time to another — the compiler schedule finds local optima. Breaking past 72-73 μs in isolation requires either reducing total work (fewer matmuls, fewer bytes) or larger structural changes (cross-layer overlap via megakernel).

### Single winner for Phase 3 integration

**v15c** (unchanged from Round 1). Still 72.88 μs, bit-exact, lowest integration risk. [Z.1+Z.2] integration into `qwen_complete.py` is **complete and importable** as of Round 2; ready for e2e test once the kernel queue clears.

### Recommended top-3 for Round 3

The Round 2 negative-result pattern tells us isolated-kernel work has diminishing returns on v15c. Shift emphasis to structural and measurement levers.

1. **[Z.1+Z.2] e2e test** (highest priority — MOVE FROM "out-of-band" TO "Round 3 #1").
   - **Rationale**: unblocks every Δscore placeholder. Without real `current_time_ms`, `NKI_FLOP_Ratio`, and per-prompt speedup, every marginal-value calculation remains a ranking-only fabrication. This is now the single highest-value item on the board. Run `main.py --mode evaluate_single --qwen qwen_complete` on prompts 0-4; verify the 5 risks flagged by the integration agent (most notably v17 output-order restoration).
   - **Expected outcome**: first trn3 per-prompt TTFT/tok/s/NKI_FLOP_Ratio/score numbers. Converts Round 1-2 kernel wins into measured end-to-end impact.

2. **[H] multi-layer fused TKG megakernel kickoff**.
   - **Rationale**: the Phase 1 matrix #2 candidate (Δscore placeholder 211). Plans B and C both confirmed v15c has hit the single-kernel ceiling; [H] is the only path to break past 72 μs/layer. Plan C's design doc (`docs/plan_g_cross_layer_prefetch_design.md`) is the [G] hook that [H] enables.
   - **Prerequisite**: can start with [Z.1+Z.2]'s compiled model in hand.

3. **[F] non-NKI HLO dot audit**.
   - **Rationale**: the ONLY lever that directly raises `NKI_FLOP_Ratio` (linear score multiplier). Phase 1 matrix flagged it as "potentially huge" but UNMEASURED. Now that [Z.1+Z.2] produces a real HLO dump, can be quantified.
   - **Prerequisite**: [Z.1+Z.2] e2e successful compile (to get HLOs).

### Alternative isolated-kernel Round 3 candidates (lower priority)

- **Partial [F3b]**: revert 2 of 4 down_w HWDGE changes to find the pipeline sweet spot (insight from Plan B regression). Small ROI — maybe 1-2 μs.
- **[X2]** background TensorE transpose — sub-2-μs cleanup.
- **[Y1]** FP8 KV cache — sub-2-μs cleanup.

These are all sub-2-μs or noise-level wins on v15c; prioritize #1/#2/#3 above first.

### Round 3+ deferred

- **[F4]** SBUF dynamic indexing — probe v2 still inconclusive (`NCC_IBIR010`). Requires a dedicated investigation session if it becomes load-bearing for Round 4+ work.
- **[E]** LM head NKI — new kernel; only useful after [Z.1+Z.2] e2e runs and measures current LM-head cost.

**Stopping per brief.** Awaiting your Round 3 top-3 pick.

---

## Phase 3 STEP 2 — BLOCKED on host memory (2026-04-21)

**Root cause**: `main.py --mode evaluate_single --platform-target trn3` (lines 804-844) loads TWO Qwen3-30B-A3B models via `prepare_inference()` — the NKI candidate (`qwen_complete`) and `baseline_qwen` (stock NxDI Qwen3MoE — `neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe`). Both go through `model.compile(...)` + `model.load(...)` — both are **fully traced, HLO-compiled, and Neuron-loaded**, not one-is-eager. Peak DRAM during HLO→NEFF compile exceeds 124 GB host memory (no swap). Prompts 0 and 1 were SIGKILL'd (returncode=-9) at ~200 s, both at `generating HLO: token_generation_model`.

**Evidence**: `MemTotal=124GiB`, `SwapTotal=0`, disk / has 123 GB free (sufficient for a swap file if needed). 2× 30B BF16 model ≈ 120 GB for weights alone; trace+compile buffers push peak over 124 GB.

**Status**: STEP 1 (risk verification — wrapper + converter) complete. STEP 2 (e2e compile + bench) paused. STEP 3/4 blocked until memory unblocked. Investigation pending user decision on P/Q/S/... mitigation.

---

## Phase 3 — Harness (`--skip-accuracy-check` workaround, 2026-04-21)

**OOM root cause**: `main.py --mode {evaluate_single, evaluate_all} --platform-target trn3` loads 2× Qwen3-30B (the NKI candidate `qwen_complete` AND `baseline_qwen` from NxDI) via `prepare_inference()`. Both models are **fully traced, HLO-compiled, and Neuron-loaded** — baseline_qwen is not a Python-only eager reference; `check_accuracy_logits` runs both models on-device. Peak DRAM during HLO→NEFF compile ≈ 120 GB weights + trace buffers > 124 GB host DRAM (no swap) → `returncode=-9` SIGKILL for every prompt.

**Workaround applied** (Option P per user authorization): added `--skip-accuracy-check` flag to `main.py`:
- `main.py` argparse line ~96: `parser.add_argument("--skip-accuracy-check", action="store_true", help="Skip baseline model load + accuracy comparison. Produces latency/throughput/NKI_FLOP_Ratio only. Use when DRAM < 2x model size.")`
- `main.py:810-825` (evaluate_single trn3 branch): wrap baseline load + `run_accuracy_check` in `elif args.skip_accuracy_check: accuracy = 1.0; accuracy_is_placeholder = True`.
- `main.py:888-939` (evaluate_all trn3 branch): same wrap pattern; per-prompt placeholder handling.
- Score calculation: `calculate_score(..., accuracy=1.0, ...)` — placeholder numeric value satisfies formula.
- Output: when flag is set, accuracy prints as `1.0 (PLACEHOLDER — baseline skipped)` in per-prompt output.

**Consequence — Phase 3 numbers are latency/throughput/NKI_FLOP_Ratio only.** Real accuracy validation pending before any Round 1 final submission. User's out-of-band follow-up: add swap (Option S) + run 1-2 prompts WITHOUT the flag to confirm correctness.

**Memory observed at t=30s smoke test (N_PROMPTS=1)**:
- baseline (idle): `used=3.8 GiB, avail=124 GiB`
- single-model-trace steady: `used=35 GiB, avail=93 GiB`
- Expected peak during HLO compile: ~60-70 GB (well under the 80 GB budget).

**Smoke test result (2026-04-21 16:49 UTC)**: `--skip-accuracy-check` confirmed working (flag value `'skip_accuracy_check': True` reached main.py; only `qwen_complete` loaded, no baseline_qwen load messages). However **prompt 0 SIGKILL'd again at 192 s** at the same HLO-compile stage. Memory trace shows peak **127 GiB anon-rss** at 16:52:59 (kernel oom-killer log confirms). The single-model path uses ~35 GB; compile phase then allocates +80 GB of XLA tracing intermediates + `neuronx-cc` subprocess pressure, hitting the 124 GB ceiling. Dual-model wasn't the bottleneck — HLO compile alone needs ~115-120 GB peak for this model size.

**Mitigation applied (Option S — 2026-04-21 17:07 UTC)**: added **64 GB swap file** at `/swapfile`.
- `sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`
- Swap priority -2 (low-urgency, won't be touched unless DRAM exhausted).
- Current state: `Mem 124 GiB / Swap 63 GiB`. Compile peak expected ~127 GB total → ~3-4 GB spills to swap, rest stays in DRAM.
- Expected e2e compile time increase: +15-30% over DRAM-only case.

**Relaunched the full 5-prompt bench** at 17:08 UTC with `--skip-accuracy-check`. ETA 60-120 min.

---

## COLD-START RECOVERY NOTES (for a new session picking up from here)

If this session terminates before STEP 2 completes, the following artifacts reconstruct full context:

| Artifact | What it has |
|---|---|
| `/tmp/phase0_e2e.log` | Full launcher + main.py stdout/stderr. `grep "^\[launcher\]"` for per-prompt status. `grep "Final Score"` for completed prompts. |
| `/tmp/phase0_mem.log` | Memory + swap usage every 15 s. Confirms whether Option S absorbed peaks. |
| `/home/ubuntu/nki-moe/_phase0_e2e_ntff/` | NTFFs captured during compile/run. Pick the `token_generation_model` one for STEP 3. |
| `/home/ubuntu/qwen-30b-a3b/traced_model/` | Cached NEFFs. Future runs with `--skip-compile` reuse these (~minutes instead of ~hours). |
| `/tmp/phase0_launch.py` | Launcher; env `N_PROMPTS=N` caps prompt count for smoke runs. |

**State as of this recovery checkpoint**:
- **Round 1 winner**: v15c MoE at 72.88 μs (bit-exact vs v14a 88.06 μs = −17.2%). Attention TKG baseline: v13bc_sbm_tiled at 53.71 μs (Round 1 Plan B `v17_fast_exp` was a FALSE POSITIVE, see `docs/integration_findings.md` Finding 1).
- **Round 2**: zero measurable wins. v15e/v15f/v15g all noise/regression. v15c remains Phase 3 integration target.
- **Integration done**: `qwen_complete.py` wired to v15c + v13bc_sbm_tiled via `_qwen_integration.py` wrapper. Smoke-tested PASS (STEP 1a: absmean 3.0e-3, triple-check; STEP 1b: byte-exact packed-weight converter).
- **Harness fix**: `main.py --skip-accuracy-check` flag added (only disables baseline load; HLO compile memory overhead needed Option S swap).
- **Phase 3 task states**: STEP 1 complete; STEP 2 running (5 prompts with swap); STEPS 3-4 queued.

**Next actions for a cold-start session**:
1. Check `/tmp/phase0_e2e.log` tail + `grep "TOTAL elapsed"` — if present, STEP 2 done.
2. Per-prompt results: `grep -E "Final Score|Latency:|Throughput:|NKI FLOPs"` — 5 blocks expected (one per prompt 0-4).
3. STEP 3: run `neuron-profile view --output-format summary-json` on one token-gen NTFF in `_phase0_e2e_ntff/`. Use the skill's `references/layer_latency.md` (CC-core 8 gap analysis) for per-layer inter-op gap; use `references/kernel_source_attribution.md` for per-source-line stall hotspots.
4. STEP 4: write `docs/trn3_round_2_report.md`. Sections: (a) did −17.2% MoE win translate to e2e? what fraction? (b) real NKI_FLOP_Ratio — first time measured; (c) engine profile from 48-layer NTFF vs v15c's isolated single-layer profile; (d) Round 3 top-3 proposal including [M1], [X1b], and re-ranked [H]/[G]/[F]/[E].
5. Round 3 candidate additions (not yet written to matrix):
   - **[M1]**: drop redundant bf16 CTE MoE weights from qwen_complete (~30 GB DRAM save). Not an NKI kernel lever — production memory-hazard fix.
   - **[X1b]**: re-probe fast-exp with `reduce_cmd=reset_reduce` mitigation per `docs/integration_findings.md` Finding 1.
6. DO NOT touch `main.py` (already touched once, authorized), `qwen.py` (forbidden), or any file outside `kernels/` without explicit user approval.

---

## Phase 3 STEP 2 — FINAL STATUS: blocked on Block D (HLO protobuf 2 GB limit)

**Timeline**:
- 17:08 UTC: launched full 5-prompt bench with `--skip-accuracy-check` + 64 GB swap.
- 17:08 → 17:31 UTC: prompt 0 compiled (35 GB model + ~60 GB HLO trace peak).
- 17:31 UTC: prompt 0 failed with `google.protobuf.message.EncodeError: Failed to serialize proto` at `torch_neuronx/xla_impl/trace.py:179` (HLO too large for `SerializeToString`).
- 17:31 → 17:52 UTC: prompts 1-4 failed identically (cache reuse made each ~7-13 min instead of 13 min).
- 17:52 UTC: TOTAL elapsed 2651.5s. **0/5 prompts completed to Final Score.**

**Root cause of Block D**: Phase 0 harness fix `BlockwiseMatmulConfig.block_size=8192` (added because `neuronxcc.nki._private.blockwise_mm` is missing in this SDK) routes all CTE prompts through `forward_all_experts`, which materializes 128 experts × 48 layers = **6144 expert sub-graphs in the HLO**. Qwen3-30B's HLO exceeds protobuf's 2 GB `SerializeToString` limit (protobuf 7.34.1 with `upb` backend; limit still hit in practice). Every prompt fails at the same serialization step.

**Memory behavior with swap** (proves Option S worked for memory but not for HLO):
- Peak DRAM used: ~110 GB
- Peak swap used: ~1 GB (only a small spillover thanks to swap absorbing the edge)
- DRAM no longer the bottleneck — HLO size is.

**Phase 3 close-out artifacts**:
- `/home/ubuntu/nki-moe/docs/trn3_round_2_report.md` — full Phase 3 writeup with Round 1-2 inventory, Block-cascade narrative, revised Round 3 top-3, cold-start handoff.
- `docs/integration_findings.md` — v17_fast_exp postmortem + Known NKI landmines (L1-L4) + triple-check pattern.

**Phase 3 final verdict**:
- **Real Round 1 MoE win (−17.2% v15c vs v14a) is verified and durable.**
- **e2e translation is UNMEASURED until [M1] (Round 3 #1 candidate) eliminates the CTE HLO explosion.**
- Attention TKG baseline is `v13bc_sbm_tiled` at 53.71 μs (Round 1 Plan B was a false positive).
- All other Round 2 plans (A/B/C) produced no measurable wins on v15c.

---

## Block D update — (b1) config lands for CTE, TKG is a separate blocker (2026-04-21)

The (b1) config (`block_size=256 + use_shard_on_block_dynamic_while=True + block_sharding_strategy="PING_PONG"`, mirrored from `qwen_fused_transformer.py:78-86`) **fixes CTE HLO for both qwen_fused_transformer AND qwen_complete** — all 3 CTE buckets (ctx=128/256/640) compile cleanly. The Phase 0 narrative "missing private blockwise_mm kernel blocks CTE" was a red herring: the installed beta2 `bwmm_shard_on_block` kernel handles CTE once the config flag routes to it (see `docs/m1_investigation.md` for the nki_import probe results).

However, **qwen_complete's TKG HLO ([1,1]) still exceeds the protobuf 2 GB serialization limit** because of a per-layer op-count difference:
- `qwen_complete` TKG emits **3 NKI custom-calls per layer** (v13bc attn + `qwen3_router_topk_plan_a` + v15c MoE) × 48 layers = **144 NKI ops** + surrounding NxDI glue (residuals, 2× RMSNorm, 2× AllReduce per layer).
- `qwen_fused_transformer` TKG emits **1 fused NKI call per layer** (`transformer_qwen3_moe_tkg_v2_jit`) × 48 layers = **48 NKI ops**. Succeeds at HLO serialization.

The 3× NKI-op-count delta is sufficient to push qwen_complete's TKG HLO past 2 GB.

**Status**:
- `[M1a]` (CTE HLO) = **complete** for qwen_complete via (b1).
- `[M1a']` (TKG HLO) = **open**. New blocker surfaced in STEP C smoke test.
- Next investigation: C4 HLO size probe (90-min timebox) to quantify per-op byte contribution before picking a fix path.

---

## Round 3 [M1a'] update — (v.1) validated, disk-full blocker surfaced (2026-04-22)

**(v.1) fix** (`register_buffer` → `nn.Parameter(requires_grad=False)` for the three TKG-only quantized MoE buffers: `gate_up_packed_w_tkg`, `down_w_tkg`, `down_scales_tkg`) **resolves the TKG HLO protobuf 2 GB serialization issue**. All 6 NEFFs (3 CTE + 3 TKG) compile cleanly on trn3.

**Root cause confirmed**: `neuronx_distributed/trace/model_builder.py:806` iterates `model.named_parameters()` only — not `named_buffers()`. Tensors registered via `register_buffer()` fall outside the trace-input set, get captured as Python closures, and materialize as `constant` HLO ops (often with 4× dtype upcast int8 → fp32). Qwen3-30B's three TKG buffers × 48 layers produced ~30 GB of `constant` HLO ops totaling > 2 GB serialized, blowing past `hlo.SerializeToString()`. See `docs/integration_findings.md` landmine **L5**.

**HLO probe results** (before (v.1), `qwen_complete` with (b1)):
- CTE HLO: 12.85–12.99 MB per bucket, 32,846 ops, `constant` bytes 1.79 MB — clean.
- TKG HLO: `ByteSize()` FAILS, 8,047 ops, `constant` bytes **30.25 GB** (530 ops, avg 57 MB). Top-10 individual `constant` ops all **427.82 MB** = exactly `int8 [128, 17, 128, 384] × 4 bytes/fp32` → `gate_up_packed_w_tkg` upcast + inlined, one per layer.

### Secondary blocker surfaced post-fix

After (v.1), HLO compile succeeded end-to-end. Process died at **t=2321s** in NxDI's `shard_weights_with_cache` → `save_file(...)` → `safetensors_rust.SafetensorError: I/O error: No space left on device (os error 28)`. **Not OOM, not a kernel bug — disk-full during shard write.**

Numbers at death:
- DRAM used: 113 GB (was 127 GB peak ~30 s earlier, trending DOWN when disk filled)
- Swap used: 63 GB / 63 GB (100% of /swapfile)
- Disk /: 0 B free (247 GB used of 247 GB)
- Offender on disk: /swapfile (65 GB), `/home/ubuntu/qwen-30b-a3b/hf_model` (57 GB), partial `tp0_sharded_checkpoint.safetensors`.

**Memory was transient** (shard-intermediate allocations; DRAM was releasing). **Disk was the hard wall.** Qwen3-30B sharded-checkpoint write needs ~80 GB for 4 TP shards + existing hf_model (57 GB) — doesn't fit with a 64 GB /swapfile on a 247 GB root FS.

### Retry plan in progress

1. **/swapfile removed** → 91 GB disk free (retry fits shard write).
2. **Hook fix (H1)** applied: `_apply_protect_quant` now does in-place `p.data = p.data.to(dtype)` instead of creating a fresh `nn.Parameter(...)` on dtype mismatch. Preserves Parameter identity in `tkg._parameters[...]`, avoids memory accumulation during NxDI's cast ping-pong, and removes the minor leak on top of the (v.1) solution.
3. **Retry with `--skip-compile`**: NEFFs are cached (85 MB `traced_model` + 305 MB `/var/tmp/neuron-compile-cache`) — this should skip HLO gen + NEFF compile entirely, dropping the DRAM peak (no XLA tracing intermediates to hold).
4. **No swap on retry initially.** If DRAM peak still exceeds 124 GB, add a small swap (~16-32 GB) with disk headroom.

### Preserved artifacts

- NEFF cache: `~/qwen-30b-a3b/traced_model/` (85 MB) + `/var/tmp/neuron-compile-cache` (305 MB).
- (b1) config + (v.1) Parameter registration + (H1) hook fix in `qwen_complete.py`.
- HLO probe evidence in `/tmp/hlo_probe.log` (read-only reference going forward).

---

## Phase 3 — CLOSEOUT (2026-04-22)

**Decision**: Phase 3 end-to-end measurement is closed. No further retry attempts. The remaining blocker is an **infrastructure constraint** (NxDI load-time DRAM peak on a 124 GB-DRAM trn3.3xlarge), not a kernel problem. Continued iteration has diminishing returns.

### What IS validated (shippable)

| Artifact | Evidence | Status |
|---|---|---|
| **v15c MoE kernel** — 72.88 μs, −17.2% vs v14a (88.06 μs) | Bit-exact vs v14a (`probe_v14a_v15c_zero_check.py`, absmean ~11.6, range [−49, 55]); Round 1 synthesis table | **Locked** |
| **int8-reinterpret HWDGE trick** (novel trn3 primitive) | v15c source + Round 1 win narrative | **Locked** |
| **qwen_complete trn3 compile path** | 6/6 NEFFs (3 CTE + 3 TKG) compile cleanly under (b1)+(v.1) | **Locked** |
| **(b1) CTE shard-on-block config** | qwen_complete CTE HLO drops to 12.85–12.99 MB; all 3 buckets compile | **Locked** |
| **(v.1) Parameter promotion** (`register_buffer` → `nn.Parameter(requires_grad=False)`) | TKG HLO `constant` bytes collapse 30.25 GB → <2 GB; `ByteSize()` succeeds | **Locked** |
| **(H1) in-place dtype hook** | `_apply_protect_quant` mutates `p.data` rather than allocating new `nn.Parameter`; preserves identity in `_parameters[...]` | **Locked** |
| **L6 NxDI streaming shard_checkpoint monkey-patch** | qwen_complete.py:111-232, `NKI_STREAMING_SHARD_PATCH=1`; logged `[L6] NxDI streaming shard_checkpoint patch ACTIVE` at runtime | **Installed, correct — never exercised (superseded by L7)** |
| **L1-L5 NKI landmines catalog** | `docs/integration_findings.md` | **Locked** |

### What is NOT captured

Real on-device e2e measurements — specifically:
- **TTFT** (time-to-first-token)
- **tok/s** (throughput)
- **NKI_FLOP_Ratio** per prompt
- **Final Score** vs baseline

### Exact reason

The `/tmp/phase0_e2e.log` run (1593 lines, 2026-04-22 05:41 UTC) SIGKILL'd at **t=1167.8s on prompt 0** (`returncode=-9`). Last logged action was the dtype-cast pass:

```
Neuron: casting layers.47.mlp.moe_fused_tkg.down_scales_tkg from torch.bfloat16 to torch.float32
[launcher] prompt 0 finished in 1167.8s  returncode=-9
```

The L6 streaming-shard patch was active (`[L6] NxDI streaming shard_checkpoint patch ACTIVE` at log line 22) and correctly installed — but the process died **one stage upstream** of where the patch intercepts. `~/qwen-30b-a3b/traced_model/weights/` is empty (no `tp{0..3}_sharded_checkpoint.safetensors`); `shard_checkpoint` was never reached.

**Root cause — L7**: NxDI's load-time dtype conversion pass at
`neuronx_distributed_inference/models/application_base.py:655`
(`Found torch.float32 weights in checkpoint: ... Will convert to torch.bfloat16`)
allocates a fresh tensor at the new dtype while the original tensor is still resident in the checkpoint dict. For Qwen3-30B-A3B with 48 layers × `down_scales_tkg` (fp32 scales) + the full bf16 checkpoint and the original fp32 copies, this doubles precision-varying tensors in RAM **before `shard_weights` ever runs**. Peak crosses the 124 GB DRAM ceiling with only the 32 GB `/swapfile` available.

**Memory headroom options we considered and ruled out** for further retry:
- **Bigger swap** — `/` has only 58 GB free after the 57 GB `hf_model` and current compile artifacts; can't grow swap to 96 GB+ without disk cleanup that risks the preserved NEFF cache.
- **fp32-scales-in-checkpoint** — would require invasive changes to the HF checkpoint format and the preshard hook contract; high risk, out of scope.
- **Larger-DRAM instance** — simplest resolution but requires instance-class change beyond trn3.3xlarge (not available this session).

### L7 added to landmines catalog

- **L6**: NxDI `shard_checkpoint` accumulates all TP-rank sharded dicts in RAM before returning (~4×15 GB for Qwen3-30B TP=4). Fix: streaming monkey-patch in qwen_complete.py. **Installed, correct — superseded by L7 as dominant peak.**
- **L7** (new): NxDI load-time dtype cast pass (`application_base.py:655`) doubles precision-varying tensors in RAM without streaming. Peak occurs **before** any disk write. 48×`down_scales_tkg` bf16↔fp32 round-trip + full checkpoint residency exceeds 124 GB DRAM on trn3.3xlarge. No streaming hook available in NxDI; fix requires either upstream NxDI change, offline fp32-scales checkpoint preparation, or a larger-DRAM host.

See `docs/integration_findings.md` for postmortems and cross-references.

---

## Phase 3 — ESTIMATED e2e (placeholder, computed from isolated benches)

**These numbers are estimates, not measured.** They are assembled from single-kernel isolated benchmarks and coarse per-layer glue costs. Real e2e numbers are blocked on L7 (see CLOSEOUT above).

### Per-layer composition

```
implied_per_layer_us = v15c(72.88) + v13bc_sbm_tiled(53.71) + AR_est(12) + gap_est(25)
                     = 72.88 + 53.71 + 12 + 25
                     = 163.59 μs
```

Components and sources:

| Component | Value (μs) | Source | Confidence |
|---|---|---|---|
| v15c MoE (MXFP4+HWDGE, 1 layer, TP=4 shard, TKG) | 72.88 | Round 1 synthesis, `OPTIMIZATION_LOG_TRN3.md` § "Round 1 results" | **Measured, locked** |
| v13bc_sbm_tiled attention (1 layer, TP=4 shard, TKG, seq=0 active) | 53.71 | `bench_v13bc_sbm_tiled.py` Round 1 | **Measured, locked** |
| `AR_est` — 2× AllReduce (qkv+ffn) per layer | 12 | Placeholder: typical trn3 AR at H=2048 TP=4; NOT measured | **Estimate** |
| `gap_est` — residuals + RMSNorm + router + inter-op scheduling gaps | 25 | Placeholder: budget from prior transformer TKG timings; NOT measured | **Estimate** |

### Model-level composition

```
implied_e2e_ms = 48 × implied_per_layer_us + post_transformer
               = 48 × 163.59 μs + post_transformer_est
               = 7852 μs + post_transformer_est
               ≈ 7.85 ms + post_transformer_est
```

Where `post_transformer_est` covers final RMSNorm + LM head + sampler + host-side overhead (sub-ms on a well-compiled graph, order 500 μs–2 ms range).

**Implied TKG latency**: ~**8–10 ms per token** on trn3 TP=4 LNC=2, assuming the estimates above hold. Baseline for comparison is `base_latency=12498.0 μs` → 12.5 ms per token (from `main.py` config), giving an implied **~20-35% TKG latency improvement** attributable to v15c+v13bc+(b1)+(v.1)+(H1).

**Explicit disclaimers**:
- `AR_est` and `gap_est` are load-bearing placeholders. Neither is measured on this model at this TP config. They could be materially off (±50%) once measured.
- Isolated single-kernel benches do not capture inter-kernel scheduling gaps, NeuronCore stalls around AllReduce, or compiler decisions at the 48-layer scope. NTFF-based per-layer timing would replace these with ground truth.
- The v15c −17.2% win is **verified and durable** at the kernel level. Its translation to e2e is estimated, not measured.
- Do not cite these numbers as submission results. They exist to set expectations during handoff only.






