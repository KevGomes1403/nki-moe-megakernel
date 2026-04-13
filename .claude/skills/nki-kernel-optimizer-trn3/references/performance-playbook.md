# NKI Performance Playbook — trn3

A well-optimized NKI kernel should be either **compute-bound** (TensorE active ≥ 90%) or
**memory-bound** (HBM bandwidth utilization ≥ 60%). The trn3-specific MXFP quantization path
is the most impactful single optimization for matmul-heavy kernels.

---

## trn3-First Optimizations (do these before general opts)

### Opt-Q1: Migrate matmul to MXFP8 (`nc_matmul_mx`)

**When**: kernel is compute-bound on BF16 matmul OR memory-bound due to weight loading.
**Effect**: 4× TensorE throughput for matmul; also halves weight HBM bytes (BF16→MXFP8).
**How**:
1. Decide quantization strategy (see `mxfp-quantization.md` strategy table).
2. For static weights: quantize offline at model-load time with `quantize_mx`.
3. For dynamic activations: call `quantize_mx` inside the kernel, one tile ahead of `nc_matmul_mx`.
4. Replace `nisa.nc_matmul` with `nisa.nc_matmul_mx`.
5. Verify correctness with relaxed tolerances (rtol=5e-2, atol=5e-2).

**Correctness risk**: MXFP8 quantization error is ~1%. FP4 error is ~5%. Use appropriate tolerances.

---

### Opt-Q2: Pipeline on-device quantization with matmul

**When**: `quantize_mx` is on the critical path (VectorE active, TensorE idle).
**Effect**: TensorE and VectorE run simultaneously — hides quantization latency.
**How**: Structure the tile loop so VectorE quantizes tile `k+1` while TensorE computes matmul on tile `k`.

```python
# Good pipeline pattern:
for k in nl.affine_range(K // TILE_K):
    # Load tile k
    act_tile = nl.load(act_hbm[m*M:(m+1)*M, k*TILE_K:(k+1)*TILE_K])
    # Quantize on VectorE — compiler schedules this to overlap with prior matmul
    nisa.quantize_mx(dst=act_q[k % 2], src=act_tile, dst_scale=act_sc[k % 2])
    # Matmul on TensorE — overlaps with next quantize
    nisa.nc_matmul_mx(dst=psum, stationary=act_q[k % 2], moving=w_q[k],
                      stationary_scale=act_sc[k % 2], moving_scale=w_sc[k],
                      accumulate=(k > 0))
```

Use double-buffering (`k % 2`) to avoid read-after-write hazards between quantize and matmul.

---

### Opt-Q3: Use fast exponential (`nisa.exponential`)

**When**: kernel has softmax, sigmoid, or GELU activation — any `exp()` call.
**Effect**: 4× VectorE throughput for exp vs `nisa.activation(..., op=nl.exp)`.
**How**: Replace `nisa.activation(dst, data, op=nl.exp)` with `nisa.exponential(dst, data, max_value=max_val)`.

```python
# Before (trn2 style):
nisa.activation(dst=exp_out, data=x_shifted, op=nl.exp)

# After (trn3 optimized):
nisa.exponential(dst=exp_out, data=x, max_value=row_max)  # 4x faster
```

---

### Opt-Q4: Use background TensorE transpose

**When**: kernel needs a matrix transpose immediately after or before a matmul.
**Effect**: Transpose runs in parallel with the next matmul — zero added latency.
**How**: Issue the matmul first, then issue the transpose; TensorE schedules them in parallel.

```python
# Background transpose pattern:
nisa.nc_matmul_mx(dst=out, stationary=a, moving=b, ...)   # start matmul
# Issue transpose of a different tensor — TensorE runs it in parallel
nisa.nc_transpose(dst=bt_sbuf, data=b_hbm_tile)           # overlaps with matmul above
```

---

## General Optimizations (apply after trn3-specific opts)

These are adapted from the trn2 playbook and remain valid on trn3.

### Opt #1: Hoist loads to minimize HBM reloads

**Problem**: Static weight tile loaded once per activation tile instead of once per weight tile.
**Solution**: Hoist weight loads to the outermost loop that covers all consumers.

```python
# Good: weight tile loaded once, reused across all M tiles
for k in nl.affine_range(K // TILE_K):
    w_q = nl.load(w_q_hbm[k*TILE_K//4:(k+1)*TILE_K//4, :])   # load once
    for m in nl.affine_range(M // TILE_M):
        act_q = ...  # quantize activation tile
        nisa.nc_matmul_mx(...)
```

---

### Opt #2: Fuse operations to eliminate intermediate HBM traffic

Fuse `matmul → layernorm`, `matmul → activation`, `quantize → matmul` in a single kernel.
No intermediate HBM stores between fused ops.

---

### Opt #3: Overlap DMA loading with computation

Use `nl.affine_range` on load loops so the compiler overlaps DMA for tile `n+1` with TensorE for tile `n`.

---

### Opt #4: Increase tile sizes to improve instruction efficiency

Aim for at least 128 elements per partition per instruction. Too-small free dimensions cause
high per-instruction overhead. On trn3, SBUF is 32 MiB — take advantage of the extra capacity
to hold larger tiles.

---

### Opt #5: Use `affine_range` for independent loads, `sequential_range` for accumulation

```python
# Independent DMA loads → affine_range (enables prefetch + overlap)
for i in nl.affine_range(num_tiles):
    tile = nl.load(...)

# Loop with carried dependency (accumulation, scan) → sequential_range
for k in nl.sequential_range(K // TILE_K):
    psum[...] += nl.load(...)
```

---

### Opt #6: Combine instructions with `nisa` multi-op primitives

```python
# Instead of 3 separate ops:
scaled   = nl.multiply(data, scale)
shifted  = nl.add(scaled, bias)
exp_out  = nisa.exponential(shifted, max_value=max_v)   # trn3: 4x faster than activation

# Consider combining scale+shift with ScalarE activation2 instruction (trn3):
# nisa.activation2 supports flexible bias + reduction in one pass
```

---

### Opt #7: Prefer short tensors as moving in `nc_matmul_mx`

For non-square matmuls, map the shorter dimension to the `moving` position to leverage
fast LoadStationary behavior (same principle as trn2).

---

### Opt #8: Reduce transposes — use background TensorE transpose

On trn3, the TensorE can transpose in parallel with another matmul (see Opt-Q4).
For cases where TensorE background transpose doesn't apply, use DMA transpose as on trn2.

---

### Opt #9: Perform sufficiently large DMA transfers

Aim for ≥ 32 KiB per transfer. On trn3 with 4.7 TB/s bandwidth, efficient transfers require
sufficiently large tiles. With MXFP8 weights (halved size), you may need to increase K tile
to maintain transfer size.

---

## SBUF Budget Guide (trn3)

SBUF = 32 MiB, 128 partitions. Per-partition: 256 KiB.

Typical MXFP8 GEMM tile budget:
| Tensor | Shape | Dtype | Bytes |
|--------|-------|-------|-------|
| act BF16 tile | [128, 128] | bf16 | 32 KiB |
| act MXFP8 tile (x2 for double-buffer) | [128, 32] × 2 | float8_x4 | 8 KiB |
| act scale (x2) | [16, 32] × 2 | uint8 | 1 KiB |
| weight MXFP8 tile | [32, 128] | float8_x4 | 4 KiB |
| weight scale | [4, 32] | uint8 | 128 B |
| output PSUM | [128, 128] | fp32 | 64 KiB |
| output SBUF | [128, 128] | bf16 | 32 KiB |
| **Total** | | | **~141 KiB / 256 KiB** |

Fits comfortably — room to increase N tile or hoist more weight tiles.

---

## Quick Reference

| Goal | Key Opts |
|---|---|
| Switch to MXFP8 matmul | Q1 (nc_matmul_mx), Q2 (pipeline quantize+matmul) |
| Reduce HBM weight traffic | Q1 (MXFP8 halves weight bytes) |
| Speed up softmax/attention exp | Q3 (fast exp, 4x) |
| Hide transpose latency | Q4 (background TensorE transpose) |
| Reduce HBM reads | #1 (hoist loads), #2 (fuse ops) |
| Keep engines busy | #3 (overlap DMA), #4 (tile sizes) |
| Avoid SBUF spill | #4 (tile sizes), declare buffers inside inner loops |
| Fix sequential dependencies | Use sequential_range not affine_range |
