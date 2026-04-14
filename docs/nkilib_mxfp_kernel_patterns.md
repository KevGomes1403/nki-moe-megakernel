# NKI Library: MXFP8 / MXFP4 Kernel Programming Guide

## Key Files

| Purpose | File |
|---|---|
| Constants & shape rules | `core/mlp/mlp_tkg/projection_mx_constants.py` |
| Gate/up projection (TKG, H-shard) | `core/mlp/mlp_tkg/gate_up_projection_mx_shard_H.py` |
| Down projection (TKG, H-shard) | `core/mlp/mlp_tkg/down_projection_mx_shard_H.py` |
| End-to-end MLP pipeline | `core/mlp/mlp_tkg/mlp_tkg_mx.py` |
| Swizzled load utilities | `experimental/mxfp_subkernels/mxfp_load_utils.py` |
| Golden/reference implementation | `core/utils/mx_torch_common.py` |
| MoE TKG kernels | `core/moe/moe_tkg/gate_up_projection_mx.py`, `down_projection_mx.py` |

---

## Hardware Block Shape

Everything in MX is built around one fundamental block:

```
8 partitions × 4 free-dim elements = 32 unpacked values → 1 uint8 scale
```

Constants:
```python
_q_height = 8    # partitions per quantization block
_q_width  = 4    # free-dim elements per block (packed as x4)
_pmax     = 128  # max partitions per SBUF tile
```

One contraction "H512 tile" = 128 partitions × 4 packed = **512 unpacked contraction elements**.

---

## The Most Critical Gotcha: Scale Layout in SBUF

Scales exist at a **sparse layout** in SBUF. The hardware places 4 scale rows at the top of each 32-partition quadrant, with the remaining 28 rows zeroed.

```
HBM scale shape:  [P // 8 = 16,  n_tiles, F]   (dense)
SBUF scale shape: [P     = 128,  n_tiles, F]   (sparse, with holes)

SBUF partition:   0  1  2  3 | 4 .. 31 | 32 33 34 35 | 36 .. 63 | 64 65 66 67 | ...
Scale present:    ✓  ✓  ✓  ✓ | zeros   | ✓  ✓  ✓  ✓  | zeros    | ✓  ✓  ✓  ✓  | ...
```

**The DMA load pattern** (from `gate_up_projection_mx_shard_H.py:120–131`):
```python
n_quadrants = H0 // SBUF_QUADRANT_SIZE  # 128 / 32 = 4
for i_quad in range(n_quadrants):
    nisa.dma_copy(
        src=weight_scale_hbm[i_quad*4 : (i_quad+1)*4, ...],  # 4 rows from HBM
        dst=weight_scale_sb[i_quad*32 : i_quad*32 + 4, ...], # into quadrant offset
    )
```

> **Scalar DGE constraint:** when loading exactly 1 partition of scale, use AP with `[[scale_f, 1], [1, scale_f]]`. Scalar DGE requires P count of exactly 1 or a multiple of 16.

---

## Shape Constraints

### For `nisa.quantize_mx`
- P dim must be in **{32, 64, 96, 128}** — use `pad_to_valid_qmx_partitions()` to round up
- F dim must be divisible by 4 (packed x4)
- Token count T must be divisible by 4
- Input dtype: `bfloat16` or `float16` only
- Output dtype: **always `float8_e4m3fn_x4`** — FP4 output is NOT supported

### For `nisa.nc_matmul_mx`
- P dim must be in **{32, 64, 128}** and ≥ 8
- When contraction dim I is not a multiple of 512, **zero-pad the last weight and weight-scale tile**:
```python
if p_I != _pmax:
    nisa.memset(dst=weight_qtz[:, last_tile, :], value=0.0)
    nisa.memset(dst=weight_qtz_scale[:, last_tile, :], value=0)
```

### General dimension requirements
- H must be divisible by 512 (one full contraction tile per loop pass)
- For MoE H-shard: `H % 128 == 0`, `H1_sharded % 4 == 0`
- For LNC=2 sharding: `BxS % 4 == 0`

---

## Activation Quantization Pipeline

Given activations `[BxS, H]` in bf16, the steps are:

1. **Pad T** to multiple of 4
2. **Layout-adapt** to `[128, n_H512_tiles, T_padded, 4]` via transpose + swizzle (`_layout_adapter_sb`)
3. **`nisa.quantize_mx`** → outputs:
   - `inp_qtz[128, n_H512_tiles * T_padded]` in `float8_e4m3fn_x4`
   - `inp_scale[128, n_H512_tiles * T_padded]` in `uint8` (already in sparse quadrant layout)
4. **Reshape** back to `[128, n_H512_tiles, T_padded]` for matmul

The "experimental swizzled load" path in `mxfp_load_utils.py` uses a stride-2 `nc_transpose` trick for performance when M=128 and K % 512 == 0.

---

## Scale Semantics

Scale is a **uint8 exponent bias**: `value = mantissa * 2^(scale - 127)`

From `mx_torch_common.py:196–221`:
```python
# Per-dtype max exponents and clamp values:
# float8_e5m2_x4:      max_exp=15, max_val=57344.0
# float8_e4m3fn_x4:    max_exp=8,  max_val=448.0
# float4_e2m1fn_x4:    max_exp=2,  max_val=6.0

exp_field     = (data_f32.view(uint32) >> 23) & 0xFF
block_max_exp = exp_field.reshape(P//8, 8, F//4, 4).max(axis=(1,3))
scale_uint8   = clip(block_max_exp - max_exp, 0, 255).astype(uint8)
```

---

## `nc_matmul_mx` Call Pattern

```python
nisa.nc_matmul_mx(
    dst             = out_psum[cur_I_sz, i_I_tile, cur_T_sz],
    stationary      = weight_qtz_sb[:, i_H512_tile, ...],   # float8/float4 x4
    moving          = hidden_qtz_sb[:, i_H512_tile, ...],   # float8_e4m3fn_x4
    stationary_scale= weight_scale_sb[:, i_H512_tile, ...], # uint8
    moving_scale    = hidden_scale_sb[:, i_H512_tile, ...], # uint8
)
```

Both `stationary=weight` and `stationary=activation` are valid — pick based on which is reused across loop iterations. The hardware is symmetric.

**PSUM output tip:** PSUM is in bf16. Max tokens per tile = `512 * 2 // 4 = 256`.

---

## PSUM Eviction with Output Swizzle

After accumulation, the strided AP evicts PSUM into a layout ready for the next `quantize_mx` call (`gate_up_projection_mx_shard_H.py:203-205`):

```python
# Target SBUF layout: [P, tiles, BxS, _q_width=4] for quantize_mx compatibility
nisa.activation(
    dst  = out_sb[...],
    op   = nl.copy,
    data = psum,
    # AP: [[_q_width * BxS, P], [1, BxS], [BxS, _q_width]]
)
```

This avoids a separate relayout step before the intermediate quantization.

---

## MXFP8 vs MXFP4: What Differs

| Aspect | MXFP8 | MXFP4 |
|---|---|---|
| Weight dtype | `float8_e4m3fn_x4` | `float4_e2m1fn_x4` |
| Activation dtype | `float8_e4m3fn_x4` | `float8_e4m3fn_x4` (same) |
| Online weight quantization | Supported by `nisa.quantize_mx` | **Not supported** — offline only |
| HBM weight footprint | 1 byte/elem | 0.5 bytes/elem |
| Scale layout | Identical | Identical |
| `nc_matmul_mx` call | Identical | Identical (dtype in tensor) |
| Fused gate+up stacking | Separate | Stacked: `[P, 2, n_H512, I]` |

> **FP4 activations do not exist.** Activations are always quantized to FP8 online. Only weights can be FP4, and they must be pre-quantized offline.

---

## End-to-End MLP Sequence (TKG)

```
1. RMSNorm (optional)          → bf16 [128, T, H]
2. _layout_adapter_sb          → bf16 [128, n_H512, T_pad, 4]
3. nisa.quantize_mx            → fp8_x4 + uint8 scales [128, n_H512*T_pad]
4. DMA load weights (sparse scale copy per quadrant)
5. Memset last I-tile zeros (if I % 512 != 0)
6. Gate projection loop (H512 tiles) → PSUM
7. Strided AP eviction to out_sb [128, tiles, BxS, 4]
8. nisa.activation (silu)
9. Up projection (same as gate)
10. Elementwise gate ⊙ up
11. nisa.quantize_mx on intermediate
12. Down projection (nc_matmul_mx per H-tile, nisa.activation to fold bias)
13. LNC reduce (sendrecv + add) if n_prgs > 1
14. DMA spill to HBM
```

---

## Complete Gotcha Checklist

1. `quantize_mx` **cannot** emit FP4 — prepare FP4 weights offline
2. `quantize_mx` P dim must be in {32, 64, 96, 128} — round up with `pad_to_valid_qmx_partitions()`
3. `nc_matmul_mx` P dim must be {32, 64, 128}, minimum 8 — zero-pad last I-tile for weights+scales
4. Scale SBUF is sparse: rows at `[0-3, 32-35, 64-67, 96-99]` only; rest must be zero
5. Scale HBM shape is `[P//8=16, ...]` dense; SBUF allocates `[P=128, ...]` with holes
6. T (tokens) must be padded to multiple of 4 before `quantize_mx`
7. H must be divisible by 512 (one contraction tile = 128×4=512 unpacked elements)
8. Scalar DGE P count must be exactly 1 or a multiple of 16 when loading partial scale rows
9. PSUM tile size bounded by `_psum_fmax*2 // _q_width = 256` tokens
10. Intermediate output must be laid out `[P, tiles, BxS, 4]` before the next `quantize_mx` — use strided AP on PSUM eviction to get this for free
11. `out_p_offset` (if used) must be 32-aligned and `out_p_offset + BxS ≤ 128`
12. For QKV fast path: H dim requires pre-shuffled `[..., 4_H, H//512, 128_H]` ordering — done offline at weight prep time
