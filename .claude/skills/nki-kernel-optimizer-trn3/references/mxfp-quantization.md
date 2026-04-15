# MXFP Quantization Reference (trn3 / NeuronCore-v4)

**These APIs are trn3-exclusive.** Do not use `nc_matmul_mx` or `quantize_mx` on trn2.

---

## Overview

MXFP (Microscaling FP) is an OCP-standard quantization format. Groups of 32 elements along the
matmul contraction dimension share one uint8 scale value. The TensorE performs dequantization
and matmul in a single instruction, achieving **4× BF16 throughput**.

Supported packed types:
- `float8_e4m3fn_x4` — FP8 E4M3, 4 elements packed per value (recommended for inference)
- `float8_e5m2_x4` — FP8 E5M2, 4 elements packed per value (wider dynamic range)
- `float4_e2m1fn_x4` — FP4 E2M1, 4 elements packed per value (maximum throughput)

---

## Quantization Strategies

### Strategy 1: Offline (Static Weights)

Pre-quantize weights at model-load time. Store MXFP8+scales in HBM. Load once per forward pass.

```python
# ---- At model load time (runs once, not in kernel) ----
@nki.jit(platform_target="trn3")
def quantize_weight_kernel(weight_hbm, quant_weight_hbm, scale_hbm):
    # weight_hbm: [K, N] BF16
    # quant_weight_hbm: [K//4, N] float8_e4m3fn_x4
    # scale_hbm: [K//8//4, N//4] uint8  (one scale per 32-element group, packed)
    for n in nl.affine_range(N // TILE_N):
        w_tile = nl.load(weight_hbm[0:K, n*TILE_N:(n+1)*TILE_N])
        q_data = nl.ndarray((nl.par_dim(128), TILE_N // 4), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        q_scale = nl.ndarray((nl.par_dim(128 // 8), TILE_N // 4), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.quantize_mx(dst=q_data, src=w_tile, dst_scale=q_scale)
        nl.store(quant_weight_hbm[0:K//4, n*TILE_N:(n+1)*TILE_N], q_data)
        nl.store(scale_hbm[0:K//32, n*TILE_N//4:(n+1)*TILE_N//4], q_scale)
```

### Strategy 2: On-Device (Dynamic Activations)

Quantize each tile of activations inside the kernel immediately before `nc_matmul_mx`.

```python
# Inside the kernel's inner loop:
act_tile_bf16 = nl.load(activation_hbm[m*TILE_M:(m+1)*TILE_M, k*TILE_K:(k+1)*TILE_K])

# Quantize activation tile on VectorE
act_q = nl.ndarray((nl.par_dim(128), TILE_K // 4), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
act_scale = nl.ndarray((nl.par_dim(16), TILE_K // 4), dtype=nl.uint8, buffer=nl.sbuf)
nisa.quantize_mx(dst=act_q, src=act_tile_bf16, dst_scale=act_scale)

# Load pre-quantized static weight
w_q = nl.load(quant_weight_hbm[k*TILE_K//4:(k+1)*TILE_K//4, n*TILE_N:(n+1)*TILE_N])
w_scale = nl.load(scale_hbm[k*TILE_K//32:, n*TILE_N//4:])

# MXFP matmul
nisa.nc_matmul_mx(dst=psum_tile, stationary=act_q, moving=w_q,
                  stationary_scale=act_scale, moving_scale=w_scale)
```

---

## `nisa.quantize_mx` API

```python
nisa.quantize_mx(dst, src, dst_scale, name=None)
```

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `dst` | SBUF tile | Output quantized data. dtype: `float8_e4m3fn_x4` or `float8_e5m2_x4` |
| `src` | SBUF tile | Input data. dtype: `bfloat16` or `float16` |
| `dst_scale` | SBUF tile | Output scales. dtype: `uint8` |

**Shape relationships (MXFP8 example):**

If `src` has shape `[128P, F]`:
- `dst` shape: `[128P, F//4]` — 4 FP8 elements packed into one value
- `dst_scale` shape: `[128//8, F//4]` = `[16, F//4]` — one scale per 8-partition × 4-free-element group (= 32 elements)

**Constraints:**
- All tensors must be in **SBUF**.
- Partition dim of `src` and `dst`: multiple of 32, max 128.
- Free dim of `src`: multiple of 4.
- Runs on **VectorE**.

**Behavior:**
- Divides `src` into groups of 32 elements (8P × 4F).
- Computes the max-abs scale for each group.
- Quantizes each element: `q = round(x / scale)`, clamped to MXFP8 range.
- Packs 4 consecutive FP8 values into one `_x4` value (interleaved layout).

**Important — output layout is interleaved:**
The packed output places elements from positions 128 apart together:
`dst[p, f] = pack(src[p, 4f], src[p, 4f+1], src[p, 4f+2], src[p, 4f+3])`
This interleaved layout is exactly what `nc_matmul_mx` expects — do not reorder.

---

## `nisa.nc_matmul_mx` API

```python
nisa.nc_matmul_mx(dst, stationary, moving, stationary_scale, moving_scale,
                  tile_position=None, tile_size=None, accumulate=None, name=None)
```

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `dst` | PSUM tile | Output. dtype: `float32` or `bfloat16` |
| `stationary` | SBUF tile | Quantized stationary (weight) matrix |
| `moving` | SBUF tile | Quantized moving (activation) matrix |
| `stationary_scale` | SBUF tile | Dequantization scales for stationary. dtype: `uint8` |
| `moving_scale` | SBUF tile | Dequantization scales for moving. dtype: `uint8` |
| `tile_position` | tuple(int,int) | `(start_row, start_col)` for sub-tile positioning |
| `tile_size` | tuple(int,int) | `(rows, cols)` for sub-tile sizing |
| `accumulate` | bool | True = add to `dst`; False = overwrite `dst` |

**Supported input dtypes:** `float8_e5m2_x4`, `float8_e4m3fn_x4`, `float4_e2m1fn_x4`
**Output dtypes:** `float32`, `bfloat16`

**Tile size constraints:**

| Dimension | Constraint |
|-----------|------------|
| Partition dim (both inputs) | Multiple of 32, max 128 |
| Free dim of stationary | Even, max 128 |
| Free dim of moving | Max 512 (FP32 output) or 1024 (BF16 output) |

**Scale tensor shapes:**
- `stationary_scale`: `[P_stat//8, F_stat//4]` — must match stationary tile's P and F.
- `moving_scale`: `[P_mov//8, F_mov//4]` — must match moving tile's P and F.

**Behavior:**
Performs: `dst = dequant(stationary, stationary_scale)^T × dequant(moving, moving_scale)`
- Dequantization and matmul are fused into a single TensorE instruction.
- Output shape: `[P_stat, F_mov_effective]` where `F_mov_effective = F_mov * 4` (unpacking).
- Available on NeuronCore-v4 only.

---

## Memory Layout Diagram

```
BF16 weight tensor [K, N]:
  K=128 rows (partition dim), N=512 cols (free dim)

After quantize_mx → MXFP8:
  quant_data: [128P, 128F]  (float8_e4m3fn_x4, packed 4:1 so represents 512 logical cols)
  scale:      [16P, 128F]   (uint8, one scale per 32 elements = 8P × 4F)

Scaling group structure:
  ┌──────────────────────────────────────────┐
  │  8 partitions × 4 free-dim values        │
  │  = 32 elements share 1 scale (1 byte)    │
  └──────────────────────────────────────────┘
  This repeats: K/8 groups along partition dim × N/4 groups along free dim

nc_matmul_mx call:
  stationary: quant_data [128P, 128F]
  stationary_scale: scale [16P, 128F]
  moving: act_quant [128P, 128F]          ← activation (per tile)
  moving_scale: act_scale [16P, 128F]
  dst: output_psum [128P, F_moving_eff]   ← FP32 or BF16
```

---

## Complete MXFP8 GEMM Kernel Template

```python
import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"

import nki
import nki.language as nl
import nki.isa as nisa
import numpy as np

# Tile constants for MXFP8 path
TILE_K = 128   # contraction tile (packed: 128/4 = 32 actual MXFP_x4 values, but 128 logical)
TILE_N = 128   # output tile (free dim of moving)
TILE_M = 128   # partition dim tile (always 128 on trn3)

@nki.jit(platform_target="trn3")
def mxfp8_gemm(
    act_hbm,       # [M, K] BF16 — activations (quantized on-device)
    w_q_hbm,       # [K//4, N] float8_e4m3fn_x4 — pre-quantized weights
    w_scale_hbm,   # [K//32, N//4] uint8 — weight scales (one per 32-element group)
    out_hbm,       # [M, N] BF16 — output
):
    M, K = act_hbm.shape[0], act_hbm.shape[1]
    N = out_hbm.shape[1]

    for m in nl.affine_range(M // TILE_M):
        # Allocate output accumulator in PSUM
        out_psum = nl.zeros((nl.par_dim(TILE_M), TILE_N), dtype=nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // TILE_K):
            # --- Load BF16 activation tile from HBM ---
            act_tile = nl.load(act_hbm[m*TILE_M:(m+1)*TILE_M, k*TILE_K:(k+1)*TILE_K])

            # --- Quantize activation on VectorE (on-device) ---
            # After packing: [128P, TILE_K//4 F] float8_e4m3fn_x4
            act_q  = nl.ndarray((nl.par_dim(TILE_M), TILE_K // 4),
                                  dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
            # Scale shape: [TILE_M//8, TILE_K//4] uint8
            act_sc = nl.ndarray((nl.par_dim(TILE_M // 8), TILE_K // 4),
                                  dtype=nl.uint8, buffer=nl.sbuf)
            nisa.quantize_mx(dst=act_q, src=act_tile, dst_scale=act_sc)

            # --- Load pre-quantized weight tile and its scales ---
            w_q = nl.load(w_q_hbm[k*TILE_K//4:(k+1)*TILE_K//4, 0:TILE_N])
            w_sc = nl.load(w_scale_hbm[k*TILE_K//32:(k+1)*TILE_K//32, 0:TILE_N//4])

            # --- MXFP8 matmul: 4× BF16 throughput ---
            nisa.nc_matmul_mx(
                dst=out_psum,
                stationary=act_q,
                moving=w_q,
                stationary_scale=act_sc,
                moving_scale=w_sc,
                accumulate=(k > 0),  # accumulate across K tiles
            )

        # --- Copy PSUM → SBUF → HBM ---
        out_sbuf = nl.ndarray((nl.par_dim(TILE_M), TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_sbuf, src=out_psum)
        nl.store(out_hbm[m*TILE_M:(m+1)*TILE_M, 0:TILE_N], out_sbuf)
```

---

## Correctness Notes for MXFP Kernels

1. **Relaxed tolerances**: MXFP8 introduces ~1–2% relative error. Use `rtol=5e-2, atol=5e-2` in `assert_allclose`.
2. **FP4 error is larger**: use `rtol=1e-1, atol=1e-1` for `float4_e2m1fn_x4`.
3. **Reference baseline**: compare against BF16 matmul, not FP32, to avoid double-counting precision differences.
4. **Scale tensor misalignment** is the #1 correctness bug — verify scale shapes match the formulas above exactly.
5. **Input range matters**: MXFP quantization saturates for values far outside `[-448, 448]` (FP8 E4M3). Clip or scale inputs if needed.
6. **Packing order**: the interleaved `_x4` layout is produced by `quantize_mx` and consumed by `nc_matmul_mx` — do not manually re-order or copy packed tensors.

---

## Performance Tips

| Scenario | Recommendation |
|----------|---------------|
| Static weights (inference) | Offline quantize at load time — zero VectorE cost per forward pass |
| Dynamic activations | Pipeline `quantize_mx` on VectorE with TensorE matmul on prior tile |
| Weight size dominates HBM bandwidth | MXFP8 halves weight bytes; MXFP4 quarters them |
| VectorE is bottleneck | Use offline quantization, or increase K tile to amortize quantize cost |
| Mixed precision output | Use BF16 output from `nc_matmul_mx` directly — avoids FP32→BF16 cast step |

---

## MoE TKG Quantization Patterns (nkilib reference)

Source: `nkilib/core/moe/moe_tkg/` — the authoritative production implementation.

### Quantization Type Detection

The MoE TKG kernel auto-detects quantization mode from the weight dtype and presence of scale tensors:

| `QuantizationType` | Condition | Notes |
|--------------------|-----------|-------|
| `NONE` | weights dtype is bf16/fp16, no scales | standard float path |
| `ROW` | `expert_gate_up_weights_scale` and `expert_down_weights_scale` provided | FP8 row quantization |
| `MX` | weights dtype is `float4_e2m1fn_x4` or `float8_e4m3fn_x4` | MXFP path, requires scales |

Static quantization (`gate_up_input_scale`, `down_input_scale`) is **not supported** in the MoE TKG kernel.

### HBM Tensor Shapes (MX path)

All weights are stored **pre-quantized** in HBM. The packing layout encodes the contraction dimension:

| Tensor | HBM Shape | Notes |
|--------|-----------|-------|
| `expert_gate_up_weights` | `[E_L, 128_H, 2, H/512, I]` | `x4` packed dtype; dim-2 fuses gate (idx 0) and up (idx 1) |
| `expert_gate_up_weights_scale` | `[E_L, 16_H, 2, H/512, I]` | uint8; 16P = 128//8 (one scale per 32-element group) |
| `expert_down_weights` | `[E_L, 128_I, I/512, H]` | `x4` packed dtype |
| `expert_down_weights_scale` | `[E_L, 16_I, I/512, H]` | uint8; 16P = 128//8 |
| `hidden_input_scale` (optional) | `[H0, H/512, T]` | uint8; only in all-expert mode when input already quantized |

**Key constants** (from `projection_mx_constants.py`):
- `MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM = 512` — max H contraction tiles per matmul pass
- `_q_width = 4` — packing factor (`_x4` dtypes)
- `_q_height = 8` — scale group height (8 partitions share one scale → 8×4=32 elements per scale)

### All-Expert MX Flow

```
Input [T, H] (bf16, HBM)
  │
  ▼ layout_adapter: swizzle to [128P, H/512, T] in SBUF
  ▼ quantize_mx  →  input_quant [128P, H/512, T] (MXFP8_x4)
                 →  input_scale  [16P,  H/512, T] (uint8)
  │
  │  (Skip the above if hidden_input_scale is provided — input already quantized upstream)
  │
  └─ for each expert in E_L (sequential):
       │
       ├─ load_gate_up_weight_scale_bias(expert_idx, gate_or_up_idx=GATE)
       │     → gate_weight_sb  [128P, H/512, I_local]
       │     → gate_scale_sb   [128P, H/512, I_local]
       │
       ├─ load_gate_up_weight_scale_bias(expert_idx, gate_or_up_idx=UP)
       │     → up_weight_sb / up_scale_sb  (same shape)
       │
       ├─ gate_up_projection_mx_shard_I():
       │     nc_matmul_mx(gate_weight, input_quant) → gate_psum
       │     bias_add → clamp → activation(SiLU/Swish) → gate_sb
       │     nc_matmul_mx(up_weight, input_quant) → up_psum
       │     bias_add → clamp → up_sb
       │     gate_sb × up_sb → intermediate_sb
       │     quantize_mx(intermediate_sb) → inter_quant [128P, I/512, T] (MXFP8_x4)
       │                                 → inter_scale  [16P,  I/512, T] (uint8)
       │
       ├─ load_broadcast_down_weight_scale_bias(expert_idx)
       │     → down_weight_sb  [128P, I/512_local, H]
       │     → down_scale_sb   [128P, I/512_local, H]
       │
       └─ down_projection_mx():
             nc_matmul_mx(down_weight, inter_quant) → output_psum
             optional POST_SCALE: output *= expert_affinity
             accumulate into output_sb (zero-init on first expert)
```

### Selective-Expert (Top-K) MX Flow

Selective mode processes one token × one top-k expert at a time. Key differences from all-expert:

- Input is quantized **once** before the loop: `quantize_mx(input) → inp_qtz [128P, H/512, T_padded]`
- LNC=2 shards on **K** (top-k): each NC handles `K//2` experts, then `nisa.sendrecv` + accumulate
- Weight loading uses **DGE (dynamic gather engine)** with precomputed `p_idx_vector` indices to avoid per-token HBM round-trips
- Gate/up uses `process_fused_gate_up_projection_mxfp4()` (H-sharded variant)
- Down uses `down_projection_mx_tp_shard_H()` (H-sharded variant, output shape `[H0, H1_shard, 4]`)
- Affinity is applied per-expert before accumulate; result is stored in SBUF `output_temp [H0, T, H1_shard]`
- Final transpose + DMA store to HBM `output [T, H]`

### LNC=2 Sharding Strategy

| Mode | Shard dimension | Gate/Up | Down | Reduce |
|------|----------------|---------|------|--------|
| All-expert | I (intermediate) | Each NC loads I/2 tiles | Each NC loads matching I/2 tiles | `stream_shuffle_broadcast` on H |
| Selective | K (top-k) | NC0: experts 0..K/2-1, NC1: K/2..K-1 | Same K split | `nisa.sendrecv` between NCs, then add |

For all-expert, `shard_on_I = (total_I512_tiles >= n_prgs)`. When true, NC0 takes first `n_I512_tiles_local` tiles, NC1 takes the rest.

### Pre-Quantized Input Optimization

In all-expert mode, if the upstream kernel (e.g., RMSNorm) already produces MX-quantized output in SBUF, pass it as `hidden_input` (in SBUF) + `hidden_input_scale`. The kernel skips the `layout_adapter` + `quantize_mx` step entirely.

Shape requirement: `hidden_input` must be `[H0, H/512, T]` in SBUF (same layout the adapter produces), `hidden_input_scale` is `[H0, H/512, T]` (uint8).

### Bias Layout (MX path)

| Projection | HBM bias shape |
|------------|---------------|
| Gate/Up (fused) | `[E_L, I_p, 2, ceil(I/512), 4]` where `I_p = I//4 if I≤512 else 128` |
| Down | `[E_L, H]` |

In selective mode, down projection bias is applied **after** down matmul via a weighted sum: `weighted_bias[T, H] = expert_affinities[T, E] @ down_bias[E, H]`.
