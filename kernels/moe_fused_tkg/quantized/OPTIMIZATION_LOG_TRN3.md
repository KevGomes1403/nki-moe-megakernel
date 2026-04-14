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
