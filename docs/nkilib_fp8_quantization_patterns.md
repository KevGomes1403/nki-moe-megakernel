# nkilib FP8 Quantization — NKI Patterns and Performance Reference

This document covers the concrete NKI API patterns used in nkilib for FP8 quantization and dequantization, with emphasis on the performance techniques relevant to MoE and MLP kernels.

---

## 1. Quantization Types and Dispatch

nkilib supports three FP8 quantization granularities, selected at kernel dispatch time in `moe_tkg.py`:

```python
class QuantizationType(Enum):
    NONE   = 0  # BF16/FP16 passthrough
    STATIC = 1  # Per-tensor: single scale for entire activation
    ROW    = 2  # Per-row: one scale per token (output row)
    MX     = 3  # Per-block: MXFP4/8, exponent packed per 8×4 block
```

**Dispatch logic** (weight dtype and scale presence determine type):

```python
# moe_tkg.py
if expert_gate_up_weights.dtype in _SUPPORTED_MX_DTYPES:  # float8_e4m3fn_x4, float4_e2m1fn_x4
    quant_type = QuantizationType.MX
elif gate_up_input_scale is not None and down_input_scale is not None:
    quant_type = QuantizationType.STATIC
elif expert_gate_up_weights_scale is not None and expert_down_weights_scale is not None:
    quant_type = QuantizationType.ROW
```

**FP8 constants:**
- Dtype: `nl.float8_e4m3` (E4M3 format)
- Max representable value: `240.0`
- Min scale floor: `1e-6` (prevents divide-by-zero)

---

## 2. Row-Wise (Per-Token) Quantization

**Source:** `core/mlp/mlp_cte/mlp_cte_quantization.py`

This is the most commonly used path for intermediate activations. The workflow is: find absmax per row → compute scale → invert scale → apply via `nisa.activation`.

### Step 1: Absmax reduction

```python
# tensor_scalar_reduce: parallel reduce along I dimension, output shape [T, 1]
nisa.tensor_scalar_reduce(
    dst=quant_abs_sbuf[0:T, 0:I],        # scratch: abs values
    data=src_sbuf[0:T, 0:I],
    op0=nl.abs,
    operand0=0.0,
    reduce_op=nl.maximum,
    reduce_res=row_dequant_scales_sbuf[0:T, 0:1],  # [T, 1] output
)
```

`tensor_scalar_reduce` with `op0=nl.abs` + `reduce_op=nl.maximum` is the canonical pattern for finding per-row absmax in a single instruction.

### Step 2: Scale computation (absmax → dequant scale)

```python
# Multiply by 1/240.0, clamp to minimum
nisa.tensor_scalar(
    dst=row_dequant_scales_sbuf[0:T, 0:1],
    data=row_dequant_scales_sbuf[0:T, 0:1],
    op0=nl.multiply,
    operand0=1.0 / 240.0,   # scale = absmax / FP8_MAX
    op1=nl.maximum,
    operand1=1e-6,           # floor: avoids 1/0
)
```

### Step 3: Invert scale (dequant → quant scale)

```python
# Compute 1/dequant_scale for use during quantization
nisa.reciprocal(
    dst=quant_scales_sbuf[0:T, 0:1],
    data=row_dequant_scales_sbuf[0:T, 0:1],
)
```

The two scale tensors serve different purposes:
- `row_dequant_scales_sbuf`: stored alongside the quantized tensor for later dequantization
- `quant_scales_sbuf`: used immediately to scale values during quantization

### Step 4: Quantize via activation

```python
# activation with scale applies: output = input * quant_scale
# Scale shape [T, 1] broadcasts across the I (partition) dimension
nisa.activation(
    dst=quantized_output_sbuf[0:T, 0:I],
    data=src_sbuf[0:T, 0:I],
    op=nl.copy,
    scale=quant_scales_sbuf[0:T, 0:1],   # [T, 1] broadcast over I
    bias=bias_vector[0:T, 0:1],
)
```

### Step 5: Clamp to FP8 range

```python
# Two-sided clamp to [-240, 240] before casting to float8_e4m3
nisa.tensor_scalar(
    dst=quantized_output_sbuf[0:T, 0:I],
    data=quantized_output_sbuf[0:T, 0:I],
    op0=nl.minimum,
    operand0=240.0,
    op1=nl.maximum,
    operand1=-240.0,
)
```

### Scale storage format (row-wise)

Row dequant scales are stored **appended to the quantized output**:
- Quantized tensor: `[T, H]` in `float8_e4m3`
- Row scales: `[T, 4]` — one `float32` scale packed as 4 FP8 bytes
- Combined: `[T, H+4]`

When loading for dequantization, the scale is extracted from the last 4 columns and reinterpreted as `float32`.

---

## 3. Static (Per-Tensor) Quantization

**Source:** `core/mlp/mlp_cte/mlp_cte_quantization.py`, `core/output_projection/output_projection_cte/output_projection_cte_quantization.py`

Scale is precomputed offline (not on-device). It is loaded once into SBUF before the kernel loop.

### Load and invert scale

```python
# rmsnorm_quant.py — load dequant scale, invert to get quant scale
nisa.dma_copy(
    src=in_scale_hbm[0:nl.tile_size.pmax, 0:1],
    dst=in_scale_sbuf[0:nl.tile_size.pmax, 0:1],
)
nisa.reciprocal(
    data=in_scale_sbuf[0:nl.tile_size.pmax, 0:1],
    dst=in_scale_sbuf[0:nl.tile_size.pmax, 0:1],
)
```

### Apply static scale

```python
# activation with static scale: same scale broadcast over all T and I
nisa.activation(
    dst=src_sbuf[0:T, 0:I],
    op=nl.copy,
    data=src_sbuf[0:T, 0:I],
    scale=static_input_quant_scale_sbuf[0:T, 0:1],  # [T, 1], single value broadcast
    bias=bias_vector[0:T, 0:1],
)
# Then clamp:
nisa.tensor_scalar(dst=..., data=..., op0=nl.minimum, operand0=240.0, op1=nl.maximum, operand1=-240.0)
```

For output projection, weight and input scales are combined:

```python
# output_projection_cte_quantization.py
input_scale_sbuf  = load_static_quant_input_scales(input_scale_hbm)
weight_scale_sbuf = load_static_quant_weight_scales(weight_scale_hbm, input_scale_sbuf)
invert_static_quant_scales(input_scale_sbuf)  # in-place reciprocal
```

The combined `weight_scale_sbuf = weight_scale * input_scale` is used in a single `nisa.activation` dequantization call after matmul.

---

## 4. Dequantization at PSUM Drain

**Source:** `core/mlp/mlp_tkg/mlp_tkg_gate_up_projection.py`, `core/output_projection/output_projection_cte/output_projection_cte_quantization.py`

Dequantization is fused into the PSUM→SBUF copy via `nisa.activation`. This eliminates a separate multiply pass.

```python
# After nc_matmul produces result in PSUM:
nisa.activation(
    dst=result_sb[0:T, 0:H],
    op=nl.copy,
    data=res_psum[0:T, 0:H],          # FP32 from PSUM
    scale=dequant_scale_sbuf[0:T, 0:1],  # dequant scale, [T,1] broadcast over H
    bias=zero_bias_sbuf[0:T, 0:1],
)
```

This pattern:
- Reads FP32 from PSUM (hardware accumulator)
- Multiplies by dequant scale during the copy
- Writes BF16/FP16 to SBUF in one instruction
- `scale` shape `[T, 1]` broadcasts across the free (H) dimension

For row-wise, the scale varies per token. For static, the same scalar broadcasts over all positions. Both use identical NKI call structure — only the scale tensor content differs.

---

## 5. MX (Microscaling) Quantization

**Source:** `core/moe/moe_tkg/all_expert_mx_impl.py`

MX quantization operates at block granularity (8 partitions × 4 free-dim elements). It uses `nisa.quantize_mx` and a transposed layout.

### Required input layout

MX requires the input to be swizzled to `[H0, H/512, T, 4]` before quantization. This is done via DMA + `nisa.nc_transpose`:

```python
# Step 1: DMA into SBUF with tiling
for t32_tile_idx in nl.affine_range(n_T32_tiles):
    nisa.dma_copy(
        src=input[t32_tile_idx, :, :, :],
        dst=input_sb[:, t32_tile_idx, :, :],
    )

# Step 2: Transpose each block
for h512_tile_idx in nl.affine_range(n_H512_tiles):
    input_transposed_psum = nl.ndarray((TILE_H, T32_H4), dtype=..., buffer=nl.psum)
    nisa.nc_transpose(
        data=input_sb[:, t32_tile_idx, h512_tile_idx, :],
        dst=input_transposed_psum,
    )
    nisa.tensor_copy(
        src=input_transposed_psum,
        dst=input_swizzled_sb[:, h512_tile_idx, t32_tile_idx, :],
    )
```

### MX quantization call

```python
# quantize_mx: produces packed FP8 tensor + uint8 scale tensor
nisa.quantize_mx(
    src=input_swizzled_sb,    # [16H*8H, H/512*T*4H] - swizzled layout
    dst=output_quant_sb,      # [16H*8H, H/512*T] in float8_e4m3fn_x4 (packed ×4)
    dst_scale=output_scale_sb, # [16H*8H, H/512*T] in uint8
)
```

`nisa.quantize_mx` is hardware-accelerated — it determines per-block max and encodes the shared exponent into `uint8` without separate reduction passes.

### Pre-quantized input fast path (`hidden_input_scale`)

When hidden states are pre-quantized upstream (e.g., by a fused RMSNorm+quant kernel), the entire layout+quantize step is skipped:

```python
# all_expert_mx_impl.py
if params.hidden_input_scale is None:
    # Full path: layout adapt → quantize
    input_quant_sb, input_scale_sb = _layout_adapter_qmx_sb(
        params.input, T32_H4, tile_H, n_T32_tiles, n_H512_tiles
    )
else:
    # Fast path: input already quantized and swizzled
    input_quant_sb  = params.input
    input_scale_sb  = params.hidden_input_scale
```

The pre-quantized input shape differs from the normal input:
- Normal: `[H0, T, H1]`
- Pre-quantized: `[H0, H/512, T]` — already in MX layout

Shape disambiguation in `MLPParameters`:
```python
if hidden_input_scale is not None:
    T = hidden_tensor.shape[2]   # MX layout: last dim is T
else:
    T = hidden_tensor.shape[1]   # Normal layout: middle dim is T
```

**Constraint:** `hidden_input_scale` is only valid when `is_mx_kernel AND is_all_expert`. Passing it to selective-expert or non-MX paths raises a `kernel_assert`.

---

## 6. RMSNorm Fused with FP8 Quantization

**Source:** `core/rmsnorm/rmsnorm_quant.py`

The RMSNorm+quant kernel is the canonical way to produce quantized activations for downstream MoE/MLP kernels, enabling the `hidden_input_scale` fast path.

### RMS computation

```python
# activation_reduce: compute sum-of-squares in one instruction
nisa.activation_reduce(
    dst=squared_sbuf[0:P, 0:H],
    op=nl.square,
    data=in_tile_sbuf[0:P, 0:H],
    reduce_op=nl.add,                         # sum across H
    reduce_res=inverse_rms_scale_sbuf[0:P, 0:1],
    bias=zero_bias_sbuf[0:P, 0:1],
    scale=1.0,
)

# rsqrt to get 1/RMS
nisa.activation(
    dst=inverse_rms_scale_sbuf[0:P, 0:1],
    op=nl.rsqrt,
    data=inverse_rms_scale_sbuf[0:P, 0:1],
    bias=eps_bias_sbuf[0:P, 0:1],          # adds epsilon before rsqrt
    scale=1.0 / H,                          # divide by sequence length
)
```

`activation_reduce` with `op=nl.square` + `reduce_op=nl.add` computes sum-of-squares in one pass. The `rsqrt` with a bias operand handles the `+eps` in the denominator without extra instructions.

### Combined RMS scale + quantization scale

For ROW quantization, the RMS normalization scale and the FP8 quantization scale can be composed:

```python
# After computing inverse_rms_scale [P, 1]:
# Multiply by gamma (LN weight) and FP8 quant scale to get combined scale
nisa.tensor_scalar(
    dst=combined_scale_sbuf[0:P, 0:1],
    data=inverse_rms_scale_sbuf[0:P, 0:1],
    op0=nl.multiply,
    operand0=1.0 / 240.0,    # FP8 quant scale baked in
)
# Then apply combined scale to input during quantization
nisa.activation(
    dst=output_sbuf[0:P, 0:H],
    op=nl.copy,
    data=in_tile_sbuf[0:P, 0:H],
    scale=combined_scale_sbuf[0:P, 0:1],
    bias=zero_bias_sbuf[0:P, 0:1],
)
```

---

## 7. SBUF Hoisting Pattern

Scale tensors are loaded from HBM into SBUF **once, outside all loops**. This is consistent across all quantization paths.

```python
# Typical structure:
# --- Outside loop: hoist scales to SBUF ---
input_scale_sbuf  = load_static_quant_input_scales(input_scale_hbm)   # HBM → SBUF
weight_scale_sbuf = load_static_quant_weight_scales(weight_scale_hbm, input_scale_sbuf)
invert_static_quant_scales(input_scale_sbuf)   # in-place reciprocal in SBUF

# --- Inner loop: reuse SBUF scales ---
for h_block in nl.affine_range(n_h_blocks):
    for s_block in nl.affine_range(n_s_blocks):
        nisa.activation(
            dst=result_sb[...],
            data=res_psum[...],
            scale=weight_scale_sbuf[...],   # reused, no reload
            bias=zero_bias_sbuf[...],
        )
```

Violating this pattern (loading scales inside the inner loop) causes repeated HBM traffic and significantly degrades throughput.

---

## 8. Partition-Dimension Broadcasting

All scale tensors are shaped `[T, 1]` or `[S, 1]` — one value per partition-dimension position, broadcast across the free dimension. This matches NKI's SBUF layout where the partition dimension (first axis) is the parallel axis.

```
Data:   [T, I]   — T tokens × I features
Scale:  [T, 1]   — one scale per token, shape compatible via broadcasting
Result: [T, I]   — each row scaled independently
```

This means a single `nisa.activation` call handles per-row scaling for the entire tile without explicit loops over the free dimension.

For **static** quantization, the scale is still `[T, 1]` but all values are identical — NKI broadcasts the same scalar to all T×I elements.

---

## 9. DMA Transpose for Scale Loading

Row-wise scales stored in weight tensors (shape `[H, I]`) need to be transposed for use in activation (which expects `[T, 1]` or per-row shape). nkilib uses `nisa.dma_transpose` for this:

```python
# mlp_tkg_gate_up_projection.py
# Reshape dequant_scale [T, I] → [num_128_I_tiles, I0] then transpose-DMA into SBUF
dequant_scale_view = dequant_scale.slice(...).reshape_dim(dim=0, shape=(num_128_I_tiles, I0))
nisa.dma_transpose(
    src=dequant_scale_view.slice(dim=0, start=0, end=num_128_I_tiles).get_view(),
    dst=dequant_tile.ap(pattern=[
        [dequant_tile.shape[1], I0],
        [1, 1],
        [1, 1],
        [1, num_128_I_tiles],
    ]),
)
```

`dma_transpose` is preferred over `nc_transpose` when the source is in HBM, as it avoids an intermediate SBUF allocation.

---

## 10. Quantized Matmul (`double_row` mode)

**Source:** `core/mlp/mlp_tkg/mlp_tkg_gate_up_projection.py`

FP8 matmuls use `perf_mode=nisa.matmul_perf_mode.double_row` to improve throughput on Trainium:

```python
nisa.nc_matmul(
    stationary=weight_tile[0:H0, 0:I, 0:H1],
    moving=input_tile[0:H0, 0:T],
    result=res_psum[0:T, 0:H1],
    perf_mode=nisa.matmul_perf_mode.double_row,  # FP8 throughput optimization
)
```

`double_row` doubles the effective matmul throughput for FP8 inputs by processing two output rows per cycle. This is only valid when both stationary and moving tensors are in FP8.

For MX quantized matmuls, the equivalent is `nisa.nc_matmul_mx` or `nisa.matmul_mx` (hardware-accelerated MXFP).

---

## 11. Weight Ring Buffer (Double-Buffering Weights)

**Source:** `core/mlp/mlp_tkg/mlp_tkg_gate_up_projection.py`

To hide DMA latency, nkilib allocates multiple weight tile slots and cycles through them:

```python
# Allocate N slots
weight_tiles = [
    sbm.alloc_stack((H0, I), name=f"gate_w_{i}", dtype=fp8_dtype)
    for i in range(tiles.num_allocated_w_tile)
]

# Use as ring buffer: load ahead while computing with current
for i_tile in nl.affine_range(n_I_tiles):
    slot = i_tile % tiles.num_allocated_w_tile
    nisa.dma_copy(dst=weight_tiles[slot][...], src=weight_hbm[i_tile, ...])
    # compute uses weight_tiles[(slot - 1) % N]  ← previous slot
```

This pattern overlaps DMA with compute, amortizing HBM latency across multiple matmul tiles.

---

## 12. NKI API Reference — Quantization Operations

| Operation | NKI Call | Notes |
|-----------|----------|-------|
| Absmax per row | `nisa.tensor_scalar_reduce(op0=nl.abs, reduce_op=nl.maximum)` | Output `[T, 1]` |
| Scale computation | `nisa.tensor_scalar(op0=nl.multiply, op1=nl.maximum)` | `absmax/240`, floor at `1e-6` |
| Scale inversion | `nisa.reciprocal(dst=..., data=...)` | In-place supported |
| Apply quant scale | `nisa.activation(op=nl.copy, scale=quant_scale)` | Scale `[T,1]` broadcasts over `I` |
| Clamp to FP8 | `nisa.tensor_scalar(op0=nl.minimum, op1=nl.maximum)` | `[-240, 240]` |
| Fused dequant | `nisa.activation(op=nl.copy, scale=dequant_scale, data=psum)` | PSUM drain + scale in one call |
| RMS square-reduce | `nisa.activation_reduce(op=nl.square, reduce_op=nl.add)` | `[T, H] → [T, 1]` |
| RMS inverse | `nisa.activation(op=nl.rsqrt, bias=eps_bias)` | Fused `eps` add |
| MX quantization | `nisa.quantize_mx(src, dst, dst_scale)` | Produces packed FP8 + uint8 scale |
| DMA transpose | `nisa.dma_transpose(src, dst, pattern)` | Transpose during HBM load |
| Swizzle transpose | `nisa.nc_transpose(data, dst)` | SBUF→PSUM transpose |
| FP8 matmul | `nisa.nc_matmul(..., perf_mode=double_row)` | Requires both inputs FP8 |

---

## 13. Common Pitfalls

**Scale shape mismatch:** Scales must be `[T, 1]` not `[T]`. NKI broadcasting requires explicit trailing `1` dimension on the free axis.

**Missing clamp before cast:** Casting to `float8_e4m3` without clamping to `[-240, 240]` first produces saturation artifacts. The clamp (`nisa.tensor_scalar`) must precede the dtype cast.

**Loading scales inside loops:** Any scale load (`nisa.dma_copy` for scales) inside the inner tile loop causes per-tile HBM traffic. Hoist to outside all loops.

**`hidden_input_scale` shape:** When passing pre-quantized input to `moe_tkg`, the tensor shape must be `[H0, H/512, T]` (MX swizzled layout), not the standard `[H0, T, H1]`. Passing the wrong shape silently produces incorrect results because `T` is read from `shape[2]` vs `shape[1]`.

**MX T alignment:** T must be divisible by 4 for `nisa.quantize_mx` and `nc_matmul_mx`. Verify before calling all-expert MX path.
