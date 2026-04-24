# Trainium2 (NeuronCore-v3) Architecture Reference

---

## Device Generations

| Instance | NeuronCore | Notes |
|---|---|---|
| `trn1`, `trn1n`, `inf2` | NeuronCore-v2 | Baseline |
| `trn2` | NeuronCore-v3 | Default for this environment |
| `trn3` | NeuronCore-v4 | Latest generation |

---

## Device Overview (trn2)

| Resource | Spec |
|---|---|
| NeuronCores per device | 8 (NeuronCore-v3) |
| HBM capacity | 96 GiB (4 stacks) |
| HBM bandwidth | 3 TB/s |
| DMA engines per device | 128 (16 per NeuronCore) |
| CC-Cores | 20 |
| NeuronLink-v3 | 4 (device-to-device) |

---

## Memory per NeuronCore

| Memory | Capacity | Notes |
|---|---|---|
| SBUF | 28 MiB | 128 partitions × 224 KiB (↑ from 24 MiB on trn1) |
| PSUM | 2 MiB | Unchanged from NeuronCore-v2 |
| HBM | Shared device pool | Accessed via DMA |

**Partition dimension**: always 128 on trn2. Free dimension is all remaining dimensions.

---

## Compute Engines

### Tensor Engine (TensorE)

Peak throughput: **158 FP8 / 79 BF16/FP16/TF32 / 20 FP32 dense TFLOPS**; 316 TFLOPS sparse.
Data-path: 4×128 FP8, 2×128 BF16/FP16, 5×128 sparse input; 1×128 output. Frequency: 2.4 GHz.

**Double FP8 matmul** (NeuronCore-v3 only)
- FP8_E4/FP8_E5 inputs run at **2× BF16 throughput** by doubling the contraction dim from 128 → 256.
- Stationary shape: `[128, 2, K//2]`, moving shape: `[128, 2, N]`, output: `[128, N]`.
- Enable via `nisa.nc_matmul(..., perf_mode=nisa.matmul_perf_mode.double_row)`.
- **Cannot** be combined with: column tiling mode, sparse matmul, or transpose mode.

**Built-in transpose mode** (NeuronCore-v3 only)
- Correctly transposes tensors containing NaN/Inf (prior matmul-with-identity approach was not bit-accurate).
- 2× speedup for FP32 transpose; FP16/BF16 transpose produces FP16/BF16 PSUM output (faster eviction).
- Enable via `nisa.nc_matmul(..., is_transpose=True)` or `nisa.nc_transpose(..., engine=nisa.constants.engine.tensor)`.

### Vector Engine (VectorE)

Peak: **1.0 TFLOPS FP32**. Data-path: 512 BF16/FP16, 256 other types. Frequency: 0.96 GHz.

**Performance mode** (automatic, no explicit opt-in needed):
- **4× throughput** for `nisa.tensor_copy` / `nisa.tensor_scalar` when: both tensors in SBUF, both BF16/FP16, and inner-most free dimension is physically contiguous.
- **2× throughput** for `nisa.tensor_copy` / `nisa.tensor_scalar`: same dtype conditions but one tensor in PSUM, or non-contiguous inner-most dim. Also for `nisa.tensor_tensor` when both inputs are SBUF BF16/FP16.

**New in NeuronCore-v3**: VectorE and GpSimdE can access SBUF in parallel (disallowed in v2). VectorE and ScalarE can access PSUM in parallel (disallowed in v2).

### Scalar Engine (ScalarE)

Optimized for element-wise scalar ops and non-linear functions (Gelu, Sqrt, etc.). 128 input/output per cycle at 1.2 GHz. Adds bit-accurate tensor copy without FP32 intermediate cast (new in v3).

### GpSimd Engine (GpSimdE)

8 fully programmable processors for arbitrary ML operators. Frequency: 1.2 GHz.

**New in NeuronCore-v3**: each processor has an integrated DMA engine running in parallel to computation and to the main DMA engines. Total integrated DMA bandwidth: **307 GB/s** (153 GB/s per read/write direction). Can reach any SBUF/HBM on-chip or off-chip in the same trn2 instance.

---

## Data Movement

### DMA Transpose (`nisa.dma_transpose`)

Transposes while moving data — swaps the most-minor HBM/SBUF dimension into the SBUF partition dimension.

**HBM → SBUF transpose**: use when HBM layout is `[free, par]` but compute needs `[par, free]`.
```python
# hbm_src: [512, 128] → sbuf_dst: [128, 512]
sbuf_dst = nisa.dma_transpose(src=hbm_src)
```
- Up to **90% DMA throughput** with hardware-friendly shapes.
- Best throughput: output SBUF partition dimension is multiple of 128, inner-most free dim is multiple of 16.

**SBUF → SBUF transpose**: alternative to TensorE transpose; reads/writes SBUF directly.
```python
# sbuf_src: [128, 128] → sbuf_dst: [128, 128]
sbuf_dst = nisa.dma_transpose(src=sbuf_src)
```
- Up to **50% DMA throughput**. Preferred when ScalarE/VectorE are the bottleneck (e.g., self-attention).

### Descriptor Generation Engine (DGE)

New hardware block in NeuronCore-v3. Two DGE instances per NeuronCore. Generates DMA descriptors on-demand, replacing static (HBM-stored) or software (GpSimdE-based) descriptor generation.

- Frees GpSimdE for computation and eliminates HBM descriptor storage overhead.
- Triggered from ScalarE or SyncE (compiler decides).
- Enable via `dge_mode=nisa.dge_mode.hw_dge` in `nisa.dma_copy` / `nisa.dma_transpose`.
- Execution time: ~600 ns per DGE-based DMA instruction; can overlap with ScalarE compute pipeline.
- **Limitation**: does not support indirect DMA (gather/scatter) — use software DGE (GpSimdE) for those.

---

## Key Design Rules for trn2 Kernels

1. **SBUF partition dim is always 128** — tile shapes must respect this.
2. **Prefer FP8 for matmul-bound kernels** — double throughput vs BF16, no extra API changes beyond `perf_mode`.
3. **VectorE performance mode is automatic** — ensure BF16/FP16 SBUF tensors have contiguous inner-most free dims to hit 4× mode.
4. **DMA transpose instead of TensorE transpose** when VectorE/ScalarE is the bottleneck.
5. **Use hardware DGE** (`dge_mode=nisa.dge_mode.hw_dge`) for all DMA operations that don't need gather/scatter.
6. **Bucket sizes must be 128-aligned** for MoE kernels on trn2.
7. **DGE `scalar_offset` pattern**: always use `offset=0` (see nki-syntax-quickref.md for details).
