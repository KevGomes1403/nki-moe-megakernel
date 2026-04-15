# NKI Syntax Quick Reference — trn3

Extends the base NKI syntax. trn3-exclusive APIs are marked **[trn3 only]**.

---

## Imports

```python
import nki
import nki.language as nl
import nki.isa as nisa
```

---

## Kernel Declaration

```python
@nki.jit(platform_target="trn3")   # ← trn3, not trn2
def my_kernel(input_hbm: nl.ndarray, output_hbm: nl.ndarray):
    ...

# Trace mode (for dispatch wrappers)
status = nki.jit(wrapper_fn, mode="trace")(arg1=..., arg2=...)
```

**Set env var before any import:**
```bash
export NEURON_PLATFORM_TARGET_OVERRIDE=trn3
```

---

## Memory Allocation

```python
# SBUF — 32 MiB on trn3 (↑ from 28 MiB on trn2), 128 partitions
buf = nl.ndarray((nl.par_dim(128), 512), dtype=nl.float32, buffer=nl.sbuf)

# PSUM
psum = nl.ndarray((nl.par_dim(128), 512), dtype=nl.float32, buffer=nl.psum)

# HBM
out = nl.ndarray(shape, dtype=nl.float16, buffer=nl.shared_hbm)

# MXFP8 packed buffer (4 elements per value)
mxfp8_buf = nl.ndarray((nl.par_dim(128), 128), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
# ↑ stores 128×4 = 512 logical FP8 elements

# Scale buffer (1 uint8 per 32-element group: 8P × 4F)
scale_buf = nl.ndarray((nl.par_dim(16), 128), dtype=nl.uint8, buffer=nl.sbuf)
```

**Partition dimension**: always 128. Free dimension: remaining dimensions.

---

## MXFP Quantization [trn3 only]

```python
# Quantize BF16/FP16 → MXFP8 packed + uint8 scales
# src: [128P, F] bf16  →  dst: [128P, F//4] float8_e4m3fn_x4
#                          dst_scale: [16P, F//4] uint8
nisa.quantize_mx(dst=mxfp8_buf, src=bf16_buf, dst_scale=scale_buf)

# Supported dst dtypes: float8_e4m3fn_x4, float8_e5m2_x4
# Constraints: P must be multiple of 32; F must be multiple of 4; all tensors in SBUF
```

---

## MXFP Matrix Multiplication [trn3 only]

```python
# MXFP8/MXFP4 matmul with integrated dequantization — 4× BF16 throughput
# stationary: [128P, 128F] float8_x4 (represents 512 logical cols)
# moving:     [128P, 512F] float8_x4 (represents 2048 logical cols)
# stationary_scale: [16P, 128F] uint8
# moving_scale:     [16P, 512F] uint8
# dst: [128P, F_moving_eff] fp32 or bf16 in PSUM
nisa.nc_matmul_mx(
    dst=psum_out,
    stationary=w_mxfp8,
    moving=act_mxfp8,
    stationary_scale=w_scale,
    moving_scale=act_scale,
    accumulate=True,    # add to dst vs overwrite
)

# Tile size constraints:
#   P dim (both inputs): multiple of 32, max 128
#   F dim stationary: even, max 128
#   F dim moving: max 512 (fp32 output) or 1024 (bf16 output)
```

---

## Standard Matrix Multiply

```python
# BF16 matmul (same as trn2)
# stationary: [par_dim, K], moving: [K, free_dim], dst: [par_dim, free_dim] PSUM
nisa.nc_matmul(dst=psum_out, stationary=weight_sbuf, moving=input_sbuf)

# Do NOT use double_row perf_mode on trn3 — use nc_matmul_mx instead
```

---

## Fast Exponential [trn3 only]

```python
# 4× faster than nisa.activation(nl.exp) on trn3
nisa.exponential(dst=exp_out, data=x, max_value=max_val)
# max_value: shift applied before exp (numerically equivalent to exp(x - max_value))
```

---

## Background Transpose [trn3 only]

```python
# TensorE runs this in parallel with an in-flight matmul
# Issue matmul first, then transpose — hardware overlaps them
nisa.nc_matmul_mx(dst=out, ...)           # start matmul
nisa.nc_transpose(dst=bt_sbuf, data=b)    # runs in background while matmul executes
```

---

## VectorE — Other Ops (trn3 enhanced)

```python
# PRNG — uses XORWOW on trn3 (higher quality than trn2 LFSR)
nisa.rng(dst=out)
nisa.rand2(dst=out)
nisa.rand_set_state(state=s)   # reproducible runs via state save/restore

# Tensor-tensor elementwise
nisa.tensor_tensor(dst=out, data1=a, data2=b, op=nl.add)
nisa.tensor_tensor(dst=out, data1=a, data2=b, op=nl.multiply)

# Tensor-scalar
nisa.tensor_scalar(dst=out, data=a, scalar0=scale, op0=nl.multiply)

# Reduction
nisa.tensor_reduce(dst=out, data=a, op=nl.add, axis=[1])
```

---

## ScalarE — New in trn3

```python
# ScalarE can now execute tensor_scalar and tensor_copy (previously VectorE-only)
# This enables VectorE + ScalarE to run simultaneously on different ops

# Activation2 — flexible bias + multiple reduction types (trn3)
# nisa.activation2(dst=out, data=a, ...)

# Standard activation (still works on trn3)
nisa.activation(dst=out, data=a, op=nl.exp)
nisa.activation(dst=out, data=a, op=nl.rsqrt)
```

---

## DMA — New in trn3

```python
# Standard DMA (same as trn2)
nisa.dma_copy(dst=dst_tensor, src=src_tensor)
nisa.dma_transpose(dst=out, src=a)

# SBUF Read-Add-Write [trn3 only] — on-the-fly accumulation near memory
# Use for expert accumulation in MoE to avoid explicit load-add-store sequences
# (compiler may emit this automatically; check docs for explicit API)
```

---

## Loop Types

```python
# Sequential — safe default; use for carry-dependent loops (accumulation, scan)
for i in nl.sequential_range(N):
    ...

# Affine — independent iterations; enables DMA-compute overlap and prefetch
for i in nl.affine_range(N):
    ...

# Static — compile-time unroll hint
for i in nl.static_range(N):
    ...
```

---

## Data Movement

```python
# HBM → SBUF
tile = nl.load(hbm_tensor[i, ...])
tile = nl.load(hbm_tensor[i, ...], dtype=nl.bfloat16)

# SBUF → HBM
nl.store(hbm_tensor[i, ...], tile)

# SBUF copy
nisa.tensor_copy(dst=sbuf_out, src=psum_in)  # also used for PSUM → SBUF
```

---

## Common Data Types (trn3)

| NKI dtype | Description | Notes |
|-----------|-------------|-------|
| `nl.float32` | 32-bit float | Full precision |
| `nl.bfloat16` | BFloat16 | Default compute dtype |
| `nl.float16` | Float16 | Also supported for quantize_mx input |
| `nl.float8_e4m3fn_x4` | MXFP8 E4M3, 4 packed | trn3 only; use for `nc_matmul_mx` |
| `nl.float8_e5m2_x4` | MXFP8 E5M2, 4 packed | trn3 only; wider dynamic range |
| `nl.float4_e2m1fn_x4` | MXFP4 E2M1, 4 packed | trn3 only; highest throughput |
| `nl.uint8` | 8-bit unsigned | Scale tensors for MXFP |
| `nl.int32` | 32-bit integer | Indices, expert IDs |

**Important**: POST_SCALE MoE mode requires `float32` affinities — `bfloat16` causes compiler error (same as trn2).

---

## trn3 vs trn2 API Differences

| Feature | trn2 | trn3 |
|---------|------|------|
| High-throughput matmul | `nc_matmul(double_row)` — 2× BF16 | `nc_matmul_mx` — 4× BF16 |
| Quantization | Manual / software | `quantize_mx` (VectorE hardware) |
| Fast exp | `nisa.activation(nl.exp)` | `nisa.exponential` (4× faster) |
| Transpose | Sequential with matmul | Background (parallel with matmul) |
| SBUF | 28 MiB | 32 MiB |
| VectorE peak | 1.0 TFLOPS | 1.2 TFLOPS |
| ScalarE tensor_scalar | Not available | Available |
| PRNG quality | LFSR | XORWOW |
| DMA scatter/gather | GpSimdE software | Hardware indirect addressing |
