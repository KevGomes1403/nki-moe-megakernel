# FP8 Quantization on Trn2: NKI and PyTorch Options

## Hardware Constraints: Trn2 (NeuronCore-v3) vs Trn3

Trn2 maps to **NeuronCore-v3** (`nc_version.gen3`). The key difference from Trn3:

| Feature | Trn2 (NC-v3) | Trn3 (NC-v4) |
|---|---|---|
| `float8_e4m3`, `float8_e5m2` dtypes | Yes | Yes |
| `nisa.nc_matmul` with FP8 inputs | Yes | Yes |
| `double_row` perf mode (2x matmul throughput) | Yes | Yes |
| `nisa.quantize_mx()` (microscaling) | **No** | Yes |
| `nisa.nc_matmul_mx()` (MXFP matmul) | **No** | Yes |

The available quantization path on Trn2 is **cFP8** (conventional FP8): a single scalar or per-row scale applied to the whole tensor or each row. No block-level microscaling.

All final logits **must** be `bfloat16`. FP8 is used only internally — for weights, activations, or both — to increase matmul throughput via `double_row` mode or reduce memory bandwidth. BF16 accumulation happens inside `nc_matmul` and results are spilled to SBUF/HBM in BF16.

---

## NKI ISA Primitives for cFP8 Quantization

These are the building blocks for any in-kernel quantization on Trn2.

### 1. Scale and cast: `nisa.activation`

The Scalar Engine performs math in float32 and casts the result to `dst.dtype` at no extra cost.

```python
# Cast bf16 → fp8_e4m3 with a per-tensor or per-row scale
quant_out = nl.ndarray(src.shape, dtype=nl.float8_e4m3, buffer=nl.sbuf)
nisa.activation(
    dst=quant_out,      # written as float8_e4m3
    op=nl.copy,
    data=src,           # bf16 input tile
    scale=inv_scale,    # 1/scale, broadcast over free dim
    bias=zero_bias,     # nl.zeros tile, required for activation API
)
```

### 2. Clamp to FP8 range: `nisa.tensor_scalar`

`float8_e4m3` max positive value is **240.0**. Clamp before casting to avoid saturation artifacts:

```python
clamped = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.sbuf)
nisa.tensor_scalar(
    dst=clamped,
    data=src,
    op0=nl.minimum, operand0=240.0,
    op1=nl.maximum, operand1=-240.0,
)
```

### 3. Per-row absmax for row-wise quantization: `nisa.tensor_scalar_reduce`

```python
row_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
nisa.tensor_scalar_reduce(
    dst=tmp,        # intermediate, same shape as src
    data=src,
    op0=nl.abs, operand0=0.0,
    reduce_op=nl.maximum,
    reduce_res=row_max,   # shape [P, 1], one max per partition row
)
```

### 4. Inverse scale: `nisa.reciprocal`

```python
inv_scale = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
nisa.reciprocal(dst=inv_scale, data=row_max)
```

### 5. FP8 matmul with `double_row` mode

`double_row` requires **both** stationary and moving tiles to be FP8, and the contraction dimension must be split: P=128 partitions × first-free-dim=2 elements = 256 per tile.

```python
# Tile layout for double_row:
#   stationary: [128_P, 2, out_free]  (FP8)
#   moving:     [128_P, 2, in_free]   (FP8)
nisa.nc_matmul(
    dst_psum,
    stationary=weight_fp8_tile,  # [128, 2, H_out]
    moving=act_fp8_tile,          # [128, 2, T]
    perf_mode=nisa.matmul_perf_mode.double_row,
)
```

### 6. Dequantization (output side)

`nc_matmul` accumulates in float32 in PSUM. To dequantize before spilling to BF16 HBM:

```python
# Multiply by weight_scale * activation_scale
nisa.tensor_scalar(
    dst=result_bf16,
    data=result_psum,       # fp32 accumulation
    op0=nl.multiply, operand0=combined_scale,  # scalar or broadcast tensor
)
```

---

## Quantization Strategies

### Strategy A: Static (Tensor-wise) Quantization

One scale per tensor, computed offline or calibrated. Fastest but least accurate.

```
scale = max(|W|) / 240.0
W_fp8 = clamp(W / scale, -240, 240).to(float8_e4m3)

# At runtime:
out_bf16 = matmul(A_fp8, W_fp8) * (scale_A * scale_W)
```

- Scale computed offline (calibration) or as `amax(|W|) / 240`.
- Stored as a single float32 scalar per weight tensor.
- In-kernel: multiply the PSUM accumulation by the combined scale before casting to BF16.

### Strategy B: Row-wise (Per-token / Per-channel) Quantization

One scale per row of the activation, or per output-channel of the weight.

```
# For activations (per-token):
scale[t] = max(|A[t, :]|) / 240.0
A_fp8[t, :] = clamp(A[t, :] / scale[t], -240, 240).to(float8_e4m3)

# Dequant: multiply each output row by scale[t] * weight_scale
```

- More accurate than static for activations with high token-to-token variance.
- Weight scales can still be static (per-expert or per-matrix).
- In-kernel: requires a scale vector `[T, 1]` loaded alongside the activation tile.

### Strategy C: Online Weight Quantization (In-Kernel)

Weights loaded from HBM as BF16, quantized tile-by-tile in SBUF before each matmul. No pre-quantized weight storage needed.

```
for each weight tile W_tile (bf16, from HBM):
    scale = amax(|W_tile|) / 240.0    # per-tile scalar
    W_tile_fp8 = clamp(W_tile / scale, -240, 240).to(float8_e4m3)
    result += matmul(A_fp8, W_tile_fp8) * (act_scale * tile_scale)
```

- Extra latency per tile (one `tensor_scalar_reduce` + `reciprocal` + `activation` before each matmul).
- Allows the model to keep weights in BF16 on HBM; no offline quantization step.
- Weight quantization accuracy is as good as per-tile row-wise.

---

## Entry Points in `qwen_fused_moe_tkg.py`

The model has four natural places where quantization/dequantization can be inserted. They are listed from outermost (PyTorch) to innermost (NKI kernel).

### Entry Point 1: Weight Preparation at Load Time (PyTorch)

**Where:** `convert_qwen3_moe_hf_to_neuron_state_dict` (line 195)

**What happens here:** BF16 expert weights are read from the HF checkpoint, transposed, and assembled into the fused `gate_up_proj` tensor (`[E, H, 2, I]`) and `down_proj` tensor (`[E, I, H]`). This is the earliest and cheapest moment to quantize weights — it runs once on CPU before compilation.

```python
# After assembling gate_up_proj:
gate_up_proj_fp8, gate_up_scale = quantize_to_fp8_static(gate_up_proj)
neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj_fp8
neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.scale"] = gate_up_scale
```

**Tradeoffs:**
- No runtime overhead; quantization is a one-time CPU cost.
- Requires the NKI kernel to accept FP8 weight inputs and a scale argument.
- Does not quantize activations (hidden states and intermediate activations remain BF16 unless also handled).

**Suitable for:** Static weight-only quantization (W8A16-style, but FP8 instead of INT8).

---

### Entry Point 2: TKG Forward Pass, Before Kernel Call (PyTorch)

**Where:** `NeuronQwen3MoeDecoderLayerFusedTKG.forward`, line 702, just before `kernel_v9b.qwen3_moe_fused_tkg[2](...)`.

```python
# Current call:
moe_out = kernel_v9b.qwen3_moe_fused_tkg[2](
    hidden_states.data,
    self.post_attention_layernorm.weight.unsqueeze(0).data,
    tkg.router.weight_T.data,
    gate_up_w,
    down_w,
)
```

**What can be added here:** PyTorch-level quantization of `gate_up_w` and `down_w` (and optionally `hidden_states`) before they are passed to the kernel. This runs on-device inside the XLA trace and will be compiled into the Neuron graph.

```python
# Example: static quantization of weights in PyTorch before the kernel call
gate_up_w_fp8 = gate_up_w.to(torch.float8_e4m3fnuz)   # cast (scale must be pre-applied)
down_w_fp8    = down_w.to(torch.float8_e4m3fnuz)

moe_out = kernel_v9b.qwen3_moe_fused_tkg[2](
    hidden_states.data,
    self.post_attention_layernorm.weight.unsqueeze(0).data,
    tkg.router.weight_T.data,
    gate_up_w_fp8,
    down_w_fp8,
    gate_up_scale,  # new arg
    down_scale,     # new arg
)
```

**Tradeoffs:**
- Quantization executes as part of the compiled Neuron graph — incurs runtime HBM bandwidth for reading BF16 and writing FP8.
- Simpler than in-kernel quantization (no NKI code changes for the quantize step).
- Weight quantization at this level re-quantizes every decode step. Cache the FP8 tensors as pre-computed parameters instead (Entry Point 1 is better for weights).
- Best used for **activation quantization** (hidden states are computed at runtime, so must be quantized here or inside the kernel).

**Suitable for:** Activation-only quantization (A8-only), or as a bridge while refactoring Entry Point 4.

---

### Entry Point 3: Inside the Kernel — Gate/Up Projection (NKI)

**Where:** `kernel_v9b.qwen3_moe_fused_tkg`, Stage 3 (gate/up matmul loop).

The kernel loads `gate_up_w` tiles of shape `[H_free=16, _PMAX=128, 2, I0=128]` (BF16) from HBM, transposes them in SBUF, and calls `nisa.nc_matmul`. The `2` free dimension is the fused gate/up channel dimension, not the double_row contraction split.

**Quantization insertion point:**

```python
# After loading gate_up tile into fused_tile_3d [128_P, H_free, 2, I0]:
# 1. Compute per-tile scale (or use pre-loaded static scale)
tile_scale = nisa.tensor_scalar_reduce(
    data=fused_tile_3d, op0=nl.abs, reduce_op=nl.maximum, ...
)  # → [128, 1]
inv_tile_scale = nisa.reciprocal(tile_scale)  # → [128, 1]

# 2. Quantize tile to FP8
fused_tile_fp8 = nl.ndarray(fused_tile_3d.shape, dtype=nl.float8_e4m3, buffer=nl.sbuf)
nisa.activation(dst=fused_tile_fp8, op=nl.copy, data=fused_tile_3d, scale=inv_tile_scale, ...)

# 3. Quantize activation (rmsnorm_out) tile similarly
act_fp8 = ...

# 4. Matmul in FP8 — standard mode (not double_row, because the 2 dim is gate/up, not contraction)
nisa.nc_matmul(psum_out, stationary=fused_tile_fp8, moving=act_fp8)

# 5. Dequantize psum before SwiGLU
nisa.tensor_scalar(dst=inter_bf16, data=psum_out, op0=nl.multiply, operand0=combined_scale)
```

**Note on `double_row` for gate/up:** The `[128_P, H_free, 2, I0]` layout means the free dim `2` is the gate/up channel split, not the contraction dim half. `double_row` requires the contraction dim to be partitioned across P=128 and first-free=2. For the gate/up matmul (contracting over H=2048), you would need the tile to be `[128_P, 2, I0]` with contraction stride = 256. This requires restructuring the weight tile layout and the H_free outer loop — non-trivial.

**Suitable for:** Online weight quantization (BF16 weights stay in HBM, quantized per-tile in SBUF). Adds ~3 extra ISA calls per tile iteration but enables FP8 matmul without any weight preprocessing.

---

### Entry Point 4: Inside the Kernel — Down Projection (NKI)

**Where:** `kernel_v9b.qwen3_moe_fused_tkg`, Stage 5 (down matmul loop).

The down projection matmul contracts over `I=256` split as `I_tiles=2` tiles of `I0=128`. The tile layout is naturally `[128_P, I_tiles=2, H_shard_free=8]` for the weight and `[128_P, I_tiles=2, T]` for the activation. This is exactly the layout `double_row` requires: contraction dim = P × first_free = 128 × 2 = 256 = I.

**`double_row` is directly applicable to the down projection:**

```python
# Weight tile: [128_P, I_tiles=2, H_free_shard=8] → quantize to FP8
down_tile_fp8 = nl.ndarray(down_tile_3d.shape, dtype=nl.float8_e4m3, buffer=nl.sbuf)
# ... (activation + clamp as above)

# Activation (intermediate after SwiGLU): [128_P, I_tiles=2, T] → quantize to FP8
inter_fp8 = nl.ndarray(inter_buf.shape, dtype=nl.float8_e4m3, buffer=nl.sbuf)
# ... (activation + clamp)

# double_row matmul — 2x throughput on Trn2
nisa.nc_matmul(
    down_buf,                     # PSUM accumulator [128, H_free_shard, T]
    stationary=down_tile_fp8,     # [128, 2, H_free_shard] FP8
    moving=inter_fp8,             # [128, 2, T] FP8
    perf_mode=nisa.matmul_perf_mode.double_row,
)

# Dequantize PSUM before spilling to HBM (output must be BF16)
nisa.tensor_scalar(
    dst=out_bf16,
    data=down_buf,
    op0=nl.multiply, operand0=down_scale * inter_scale,
)
```

**Why this is the highest-value entry point:** The down projection is the largest matmul in the kernel (`[T, I] @ [I, H_shard]`). Using `double_row` on Trn2 doubles the tensor engine throughput for this operation. The tile layout already has `I_tiles=2` as the first free dim, so **no weight layout changes are required**.

**Constraint:** Both stationary and moving must be FP8 for `double_row`. If only weights are quantized (W8A16 style), `double_row` is not available.

---

## Recommended Quantization Approach

Given the constraints (BF16 weights from HF, FP8 allowed during inference, BF16 final logits):

### Option 1: Weight-only FP8 (simplest)

1. Quantize `gate_up_proj` and `down_proj` weights **at load time** in `convert_qwen3_moe_hf_to_neuron_state_dict` (Entry Point 1) using static per-expert or per-matrix scales.
2. Store FP8 weights + float32 scales in the state dict.
3. Modify `kernel_v9b` to accept FP8 weight tensors and scale arguments, and multiply PSUM by the scale before BF16 spill.
4. `double_row` is not available (activations remain BF16).

**Benefit:** ~2x weight HBM bandwidth reduction; reduces DMA time for weight tiles.

### Option 2: W8A8 FP8 with `double_row` on Down Projection (highest throughput)

1. Quantize weights at load time (Entry Point 1) — store as FP8.
2. Inside `kernel_v9b`, after the SwiGLU activation, quantize the intermediate buffer online (Entry Point 4) using `nisa.activation` + `nisa.tensor_scalar`.
3. Use `double_row` for the down matmul.
4. Dequantize PSUM (multiply by `weight_scale * act_scale`) before BF16 spill to HBM.
5. For the gate/up matmul, use standard FP8 matmul without `double_row` (activation quantized similarly at Entry Point 3).

**Benefit:** 2x tensor engine throughput for the down projection; reduced HBM bandwidth for all weight loads.

### What NOT to do

- Do not use `nisa.quantize_mx()` — not available on Trn2.
- Do not use `nisa.nc_matmul_mx()` — not available on Trn2.
- Do not apply quantization after the final `lm_head` projection — logits must remain BF16.
- Do not quantize `router_w` (float32, tiny matrix, correctness-critical for routing decisions).
- Do not quantize the RMSNorm path — adds complexity with no matmul benefit.

---

## Data Flow Summary

```
HBM (BF16 weights)
    │
    │ Entry Point 1: quantize at load time (PyTorch, offline)
    ▼
HBM (FP8 weights + float32 scales)
    │
    │ DMA into SBUF per tile
    ▼
SBUF: weight_tile (FP8)          SBUF: act_tile (BF16 from RMSNorm)
    │                                 │
    │                                 │ Entry Point 3/4: online act quant
    │                                 │ (nisa.activation + tensor_scalar)
    │                                 ▼
    │                            SBUF: act_tile (FP8)
    │                                 │
    └──────────────────┬──────────────┘
                       │
                       ▼
            nisa.nc_matmul [double_row for down proj]
                       │
                       ▼ (float32 PSUM accumulation)
            nisa.tensor_scalar (× weight_scale × act_scale)
                       │
                       ▼ (BF16)
                   HBM output
                       │
                       ▼ (after TP all-reduce + residual add)
               lm_head → logits (BF16)  ✓
```
