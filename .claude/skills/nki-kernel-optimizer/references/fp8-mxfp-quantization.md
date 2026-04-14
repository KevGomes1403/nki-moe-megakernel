# FP8 and MXFP Quantization — NKI Patterns

Source: `nkilib/core/mlp/mlp_cte/mlp_cte_quantization.py`, `core/rmsnorm/rmsnorm_quant.py`, `core/moe/moe_tkg/all_expert_mx_impl.py`

---

## Quantization Types

```python
class QuantizationType(Enum):
    NONE   = 0  # BF16/FP16 passthrough
    STATIC = 1  # Per-tensor: single scale for entire activation
    ROW    = 2  # Per-row (per-token): one scale per output row
    MX     = 3  # Per-block: MXFP8/MXFP4, shared exponent per 8×4 block
```

**FP8 constants:**
- Dtype: `nl.float8_e4m3` (E4M3 format)
- Max representable value: `240.0`
- Min scale floor: `1e-6`

---

## Row-Wise (Per-Token) FP8 Quantization

Five-step workflow. Each step maps to one NKI call.

### Step 1 — Absmax per row

```python
nisa.tensor_scalar_reduce(
    dst=quant_abs_sbuf[0:T, 0:I],
    data=src_sbuf[0:T, 0:I],
    op0=nl.abs,
    operand0=0.0,
    reduce_op=nl.maximum,
    reduce_res=row_dequant_scales_sbuf[0:T, 0:1],  # output: [T, 1]
)
```

### Step 2 — Scale computation

```python
nisa.tensor_scalar(
    dst=row_dequant_scales_sbuf[0:T, 0:1],
    data=row_dequant_scales_sbuf[0:T, 0:1],
    op0=nl.multiply,
    operand0=1.0 / 240.0,   # scale = absmax / FP8_MAX
    op1=nl.maximum,
    operand1=1e-6,           # floor: prevents 1/0
)
```

### Step 3 — Invert scale for quantization

```python
nisa.reciprocal(
    dst=quant_scales_sbuf[0:T, 0:1],
    data=row_dequant_scales_sbuf[0:T, 0:1],
)
# row_dequant_scales_sbuf → stored with tensor for dequant later
# quant_scales_sbuf       → used immediately during step 4
```

### Step 4 — Apply quant scale

```python
nisa.activation(
    dst=quantized_output_sbuf[0:T, 0:I],
    data=src_sbuf[0:T, 0:I],
    op=nl.copy,
    scale=quant_scales_sbuf[0:T, 0:1],  # [T, 1] broadcasts over I
    bias=bias_vector[0:T, 0:1],
)
```

### Step 5 — Clamp to FP8 range

```python
nisa.tensor_scalar(
    dst=quantized_output_sbuf[0:T, 0:I],
    data=quantized_output_sbuf[0:T, 0:I],
    op0=nl.minimum,
    operand0=240.0,
    op1=nl.maximum,
    operand1=-240.0,
)
```

### Scale storage format

Row dequant scales are appended to the quantized output:
- Quantized tensor: `[T, H]` in `float8_e4m3`
- Row scales: `[T, 4]` — one `float32` packed as 4 FP8 bytes
- Combined on-device layout: `[T, H+4]`

---

## Static (Per-Tensor) FP8 Quantization

Scale is precomputed offline. Load it once into SBUF **before all loops** — never inside the tile loop.

```python
# Outside all loops:
nisa.dma_copy(src=in_scale_hbm[0:nl.tile_size.pmax, 0:1],
              dst=in_scale_sbuf[0:nl.tile_size.pmax, 0:1])
nisa.reciprocal(data=in_scale_sbuf[0:nl.tile_size.pmax, 0:1],
                dst=in_scale_sbuf[0:nl.tile_size.pmax, 0:1])

# Inner loop — reuse SBUF scale, no HBM traffic:
nisa.activation(dst=src_sbuf[0:T, 0:I], op=nl.copy, data=src_sbuf[0:T, 0:I],
                scale=static_input_quant_scale_sbuf[0:T, 0:1],
                bias=bias_vector[0:T, 0:1])
nisa.tensor_scalar(dst=..., data=..., op0=nl.minimum, operand0=240.0,
                   op1=nl.maximum, operand1=-240.0)
```

For output projection, weight and input scales are combined into one before the loop:
```python
# weight_scale_sbuf = weight_scale * input_scale — single dequant call covers both
input_scale_sbuf  = load_static_quant_input_scales(input_scale_hbm)
weight_scale_sbuf = load_static_quant_weight_scales(weight_scale_hbm, input_scale_sbuf)
invert_static_quant_scales(input_scale_sbuf)  # in-place reciprocal
```

---

## Dequantization at PSUM Drain

Fuse dequantization into the PSUM→SBUF copy. This eliminates a separate multiply pass.

```python
nisa.activation(
    dst=result_sb[0:T, 0:H],
    op=nl.copy,
    data=res_psum[0:T, 0:H],             # FP32 from hardware accumulator
    scale=dequant_scale_sbuf[0:T, 0:1],  # [T, 1] broadcasts over H
    bias=zero_bias_sbuf[0:T, 0:1],
)
```

One instruction: reads FP32 PSUM → multiplies by dequant scale → writes BF16 to SBUF. Works identically for row-wise (scale varies per token) and static (same scalar broadcast everywhere).

---

## FP8 Matmul: `double_row` Mode

Use `perf_mode=nisa.matmul_perf_mode.double_row` when both inputs are FP8. Doubles effective throughput by processing two output rows per cycle.

```python
nisa.nc_matmul(
    stationary=weight_tile[0:H0, 0:I, 0:H1],
    moving=input_tile[0:H0, 0:T],
    result=res_psum[0:T, 0:H1],
    perf_mode=nisa.matmul_perf_mode.double_row,  # only valid when both inputs are FP8
)
```

---

## RMSNorm Fused with FP8 Quantization

The canonical pattern for producing quantized activations for downstream MoE/MLP kernels.

```python
# Step 1: sum-of-squares in one instruction
nisa.activation_reduce(
    dst=squared_sbuf[0:P, 0:H],
    op=nl.square,
    data=in_tile_sbuf[0:P, 0:H],
    reduce_op=nl.add,
    reduce_res=inverse_rms_scale_sbuf[0:P, 0:1],
    bias=zero_bias_sbuf[0:P, 0:1],
    scale=1.0,
)

# Step 2: rsqrt with fused epsilon add (no separate instruction)
nisa.activation(
    dst=inverse_rms_scale_sbuf[0:P, 0:1],
    op=nl.rsqrt,
    data=inverse_rms_scale_sbuf[0:P, 0:1],
    bias=eps_bias_sbuf[0:P, 0:1],   # adds epsilon before rsqrt
    scale=1.0 / H,                   # divide by sequence length
)

# Step 3: compose RMS scale with FP8 quant scale — single activation call
nisa.tensor_scalar(
    dst=combined_scale_sbuf[0:P, 0:1],
    data=inverse_rms_scale_sbuf[0:P, 0:1],
    op0=nl.multiply,
    operand0=1.0 / 240.0,
)
nisa.activation(
    dst=output_sbuf[0:P, 0:H],
    op=nl.copy,
    data=in_tile_sbuf[0:P, 0:H],
    scale=combined_scale_sbuf[0:P, 0:1],
    bias=zero_bias_sbuf[0:P, 0:1],
)
```

This produces quantized output in one activation call rather than three separate passes (normalize → scale → quantize).

---

## MXFP8 / MXFP4 Quantization

### Hardware block shape

```
8 partitions × 4 free-dim elements = 32 unpacked values → 1 uint8 scale
```

One H512 contraction tile = 128 partitions × 4 packed = **512 unpacked contraction elements**.

### Critical gotcha: sparse scale layout in SBUF

Scales exist at a **sparse** layout in SBUF. Hardware places 4 scale rows at the top of each 32-partition quadrant; remaining 28 rows are zero.

```
HBM scale shape:  [P // 8 = 16,  n_tiles, F]   — dense
SBUF scale shape: [P     = 128,  n_tiles, F]   — sparse, with holes

SBUF partitions:  0  1  2  3 | 4 .. 31 | 32 33 34 35 | 36 .. 63 | 64 65 66 67 | ...
Scale present:    ✓  ✓  ✓  ✓ | zeros   |  ✓  ✓  ✓  ✓ | zeros    |  ✓  ✓  ✓  ✓ | ...
```

**Required DMA load pattern — load 4 rows from each HBM quadrant into the matching SBUF quadrant offset:**

```python
n_quadrants = H0 // SBUF_QUADRANT_SIZE  # 128 / 32 = 4
for i_quad in range(n_quadrants):
    nisa.dma_copy(
        src=weight_scale_hbm[i_quad*4 : (i_quad+1)*4, ...],  # 4 dense rows
        dst=weight_scale_sb[i_quad*32 : i_quad*32 + 4, ...], # sparse quadrant slot
    )
```

### Input layout required for `nisa.quantize_mx`

Input must be swizzled to `[H0=128, H/512, T, 4]` before calling `quantize_mx`:

```python
# Step 1: DMA tiles into SBUF
for t32_tile_idx in nl.affine_range(n_T32_tiles):
    nisa.dma_copy(src=input[t32_tile_idx, :, :, :],
                  dst=input_sb[:, t32_tile_idx, :, :])

# Step 2: Transpose each block into swizzled layout
for h512_tile_idx in nl.affine_range(n_H512_tiles):
    input_transposed_psum = nl.ndarray((TILE_H, T32_H4), dtype=..., buffer=nl.psum)
    nisa.nc_transpose(data=input_sb[:, t32_tile_idx, h512_tile_idx, :],
                      dst=input_transposed_psum)
    nisa.tensor_copy(src=input_transposed_psum,
                     dst=input_swizzled_sb[:, h512_tile_idx, t32_tile_idx, :])
```

### `nisa.quantize_mx` call

```python
nisa.quantize_mx(
    src=input_swizzled_sb,    # [128, H/512*T, 4] swizzled
    dst=output_quant_sb,      # [128, H/512*T] in float8_e4m3fn_x4 (packed ×4)
    dst_scale=output_scale_sb, # [128, H/512*T] in uint8
)
```

Hardware-accelerated: determines per-block max and encodes shared exponent in one pass.

### `nisa.nc_matmul_mx` call

```python
nisa.nc_matmul_mx(
    dst=out_psum[cur_I_sz, i_I_tile, cur_T_sz],
    stationary=weight_qtz_sb[:, i_H512_tile, ...],    # float8/float4 x4
    moving=hidden_qtz_sb[:, i_H512_tile, ...],         # float8_e4m3fn_x4
    stationary_scale=weight_scale_sb[:, i_H512_tile, ...],  # uint8
    moving_scale=hidden_scale_sb[:, i_H512_tile, ...],      # uint8
)
```

### Pre-quantized input fast path

When hidden states arrive pre-quantized (e.g. from a fused RMSNorm+quant kernel), skip the layout+quantize step entirely:

```python
if params.hidden_input_scale is None:
    input_quant_sb, input_scale_sb = _layout_adapter_qmx_sb(params.input, ...)
else:
    input_quant_sb = params.input          # already quantized
    input_scale_sb = params.hidden_input_scale  # already in MX layout
```

Pre-quantized input shape is `[H0, H/512, T]` (MX layout), **not** `[H0, T, H1]`. The `T` index is at `shape[2]`, not `shape[1]`.

### MXFP8 vs MXFP4

| Aspect | MXFP8 | MXFP4 |
|--------|-------|-------|
| Weight dtype | `float8_e4m3fn_x4` | `float4_e2m1fn_x4` |
| Activation dtype | `float8_e4m3fn_x4` | `float8_e4m3fn_x4` (same) |
| Online weight quantization | Supported | **Not supported** — offline only |
| HBM weight footprint | 1 byte/elem | 0.5 bytes/elem |
| Scale layout | Identical | Identical |
| `nc_matmul_mx` call | Identical | Identical |

FP4 activations do not exist. Activations are always FP8. Only weights can be FP4, and they must be pre-quantized offline.

---

## Shape Constraints Checklist

**For `nisa.quantize_mx`:**
- P dim must be in `{32, 64, 96, 128}` — use `pad_to_valid_qmx_partitions()` to round up
- T (tokens) must be divisible by 4 — pad before calling
- H must be divisible by 512 (one full contraction tile per loop pass)
- Input dtype: `bfloat16` or `float16` only; output is always `float8_e4m3fn_x4`

**For `nisa.nc_matmul_mx`:**
- P dim must be in `{32, 64, 128}`, minimum 8
- Zero-pad last I-tile if `I % 512 != 0`:
```python
if p_I != _pmax:
    nisa.memset(dst=weight_qtz[:, last_tile, :], value=0.0)
    nisa.memset(dst=weight_qtz_scale[:, last_tile, :], value=0)
```
- Scalar DGE P count must be exactly 1 or a multiple of 16 when loading partial scale rows

---

## NKI API Quick Reference

| Operation | NKI Call |
|-----------|----------|
| Absmax per row | `nisa.tensor_scalar_reduce(op0=nl.abs, reduce_op=nl.maximum)` → `[T, 1]` |
| Scale × floor | `nisa.tensor_scalar(op0=nl.multiply, op1=nl.maximum)` |
| Invert scale | `nisa.reciprocal(dst, data)` — in-place supported |
| Apply quant scale | `nisa.activation(op=nl.copy, scale=[T,1])` — broadcasts over free dim |
| Clamp to FP8 | `nisa.tensor_scalar(op0=nl.minimum, op1=nl.maximum)` with `±240.0` |
| Fused PSUM dequant | `nisa.activation(op=nl.copy, data=psum, scale=dequant_scale)` |
| RMS sum-of-squares | `nisa.activation_reduce(op=nl.square, reduce_op=nl.add)` |
| RMS inverse + eps | `nisa.activation(op=nl.rsqrt, bias=eps_bias)` |
| MX quantize | `nisa.quantize_mx(src, dst, dst_scale)` |
| MX matmul | `nisa.nc_matmul_mx(dst, stationary, moving, stationary_scale, moving_scale)` |
| FP8 matmul 2× | `nisa.nc_matmul(..., perf_mode=matmul_perf_mode.double_row)` |

---

## Common Pitfalls

**Scale shape must be `[T, 1]` not `[T]`**: NKI broadcasting requires an explicit trailing `1` on the free axis.

**Missing clamp before cast**: Casting to `float8_e4m3` without clamping to `[-240, 240]` first produces saturation artifacts. The `nisa.tensor_scalar` clamp must come before the dtype cast.

**Loading scales inside loops**: Any `nisa.dma_copy` for scales inside the inner tile loop causes per-tile HBM traffic. Hoist all scale loads to before all loops.

**`hidden_input_scale` wrong shape**: Pre-quantized input to MX kernels must be `[H0, H/512, T]`. Passing `[H0, T, H1]` silently produces wrong results because `T` is read from `shape[2]` vs `shape[1]`.

**MX T alignment**: T must be divisible by 4 before calling `quantize_mx`. Verify before invoking the all-expert MX path.

**SBUF scale holes**: SBUF scale tensor allocates `[128, ...]` but only rows `[0-3, 32-35, 64-67, 96-99]` contain valid data. Never read or write other rows as if they contain scales.
