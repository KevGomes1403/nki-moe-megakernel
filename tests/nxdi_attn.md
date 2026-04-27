# Qwen3-30B-A3B Attention Token Generation: Precision & Execution Spec

**Model Config:** Qwen3-30B-A3B MoE  
**Hardware:** AWS Trainium3 (trn3)  
**Parallelism:** TP=4, GQA with 32 global Q heads, 4 global KV heads  
**Mode:** Token Generation (TKG, seq=1, bs=1)  
**Baseline:** NXD Inference with `attn_block_tkg_nki_kernel_enabled=False` (Python/HLO path)  
**Compiler flags:** `--enable-mixed-precision-accumulation`  
**Profile data:** `/tmp/tkg_bk0_vnc2.json` (237,885 instructions, TKG bucket 0, vnc_2, LNC=2)

---

## 0. Hardware Execution Notes — Critical for Bit-Exact Reproduction

### 0.1 Attention Execution Span & Engine Usage (Layer .49)
- **Total attention execution time:** ~15.9 µs (15,899 ps) from first RoPE DMA to final all-reduce
- **Engine allocation:**
  - **TensorMatrix (PE):** Matmuls (Q @ K^T, softmax @ V, o_proj) — ~260 ps avg per MATMUL
  - **Tensor:** Weight-load sequencing (LDWEIGHTS) — ~185 ps avg
  - **Vector:** Reductions, elementwise ops, reciprocal (divide), max — 190–280 ps avg
  - **Scalar:** Exp via lookup-table (ACT_TABLE_LOAD + ACTIVATE), other non-linearities — 200–1283 ps avg
  - **Sync/GpSimd:** DMA/control coordination, gather/scatter — 31–517 ps per operation
- **Profile verification:** 237,885 total instructions; layer .49 contains 8,722 important HLO operations (dot, exponential, sine, cosine, divide, etc.)

### 0.2 Softmax — Plain HLO Ops (NOT a custom-call); LUT only at device level

**Verified from HLO proto** (`model.MODULE_b12b4af240c745dd81cd+93fa4f67.hlo_module.pb`):
- The entry computation has **0 custom-call ops**.
- Softmax is implemented as **plain HLO instructions** in the entry:
  - 144 `exponential`, 144 `subtract`, 144 `maximum`, 290 `divide`, 384 `reduce` (across all 48 layers)
  - All softmax intermediates are F32 (verified by the HLO `Shape.element_type=11` field)
- **The "48 Native Softmax's detected and replaced" line in `log-neuron-cc.txt` refers to a BIR-level pattern fusion** (combining HLO `subtract`/`exp`/`reduce`/`divide` into a single fused softmax microkernel during code generation). At the HLO level, the algorithm is still the plain log-sum-exp.

**Device-level lowering of softmax ops** (below HLO, from `compiler_opcode` field in profile JSON):

**HLO `exp` → Scalar-engine LUT (`ACT_TABLE_LOAD` + `ACTIVATE`):**
- Profile shows `ACT_TABLE_LOAD` (Scalar, ~1283 ps) + `ACTIVATE` (~347 ps) per HLO `exp`.
- The compiler lowers `exp` to a precomputed table lookup + interpolation on the Scalar engine.
- **Precision impact:** Hardware LUT-exp will NOT match `math.exp(x)` to ULP. NKI's `nl.exp` lowers to the same Scalar-engine path, so an NKI kernel using `nl.exp` will match; one that hand-rolls polynomial exp will not.

**HLO `divide` → Vector RECIPROCAL + Scalar multiply:**
- Profile shows `RECIPROCAL` (Vector, ~184 ps) producing `1/denom`, then `ACTIVATE` (Scalar, ~370 ps) for the per-element multiply.
- HLO emits a single `divide` op; BIR lowers it to reciprocal × multiply.
- **Precision impact:** Hardware RECIPROCAL has roughly bf16-mantissa precision (verify on trn3 spec); not a true f32 divide. Affects softmax denominator division.

### 0.3 RMSNorm IS a custom-call — `AwsNeuronRmsNorm` (opaque kernel)

**Verified from HLO proto:**
- 193 `custom-call` ops total across all computations target `AwsNeuronRmsNorm` (4 RMSNorms per layer × 48 layers + 1 final RMSNorm = 193).
- The call chain is: entry computation → `RmsNormForwardImpl.X` (wrapper) → `HloRmsNormForwardImpl.Y` → **`custom-call(target=AwsNeuronRmsNorm)`**.
- Inputs: `F32[1,1,2048]` activation, `[2048]` gamma, `F32[]` epsilon. Output: `F32[1,1,2048]`.
- The Python-level "cast to f32 → square → reduce → rsqrt → multiply gamma → cast to bf16" pseudocode in §2 below describes the *intended* numerical recipe, but the actual implementation is opaque inside the custom-call. **No `rsqrt` opcode exists anywhere in the HLO** (entry or sub-computations).
- For an NKI kernel: use `nl.rms_norm` or implement RMSNorm with the same hardware ops (square → reduce → reciprocal-of-sqrt via `nl.rsqrt` on Scalar engine → multiply). Bit-exact match requires using the same engine path as `AwsNeuronRmsNorm`.

**Other custom-calls in the HLO** (none for softmax, none for QKV):
- 48 `AwsNeuronTopK` (sampling)
- 48 `AwsNeuronSilu` (MoE expert SiLU activation; NOT in attention path)
- 48 `AwsNeuronModuleMarkerStart-Forward` / 48 `AwsNeuronModuleMarkerEnd-Forward` (compiler hints, no numerics)

### 0.4 Mixed-Precision Accumulation & PE Matmul Output Format

**With `--enable-mixed-precision-accumulation` enabled:**
- **Matmul inputs:** BF16 Q and K enter the TensorMatrix PE.
- **PE accumulation:** F32 internally in PSUM banks.
- **PE output:** F32 PSUM values; no automatic conversion at PE boundary.
- **Explicit `convert` HLO op:** F32 → BF16 appears as a top-level HLO `convert` between the matmul and the next op (entry computation has 1881 `convert` ops total).
- **For bit-exact reproduction:** A NKI matmul must use F32 PSUM accumulation with explicit BF16 conversion at the read-out, NOT a BF16 accumulator.

### 0.5 Artifact Pass: `138311351493953_vnc_0.ntff` / `neff_138311351493953_vnc_2.neff` (2026-04-27)

This pass uses the requested source profile:

```
NTFF: output/baseline/1776976164165154117/138311351493953_vnc_0.ntff
NEFF: output/baseline/1776976164165154117/neff_138311351493953_vnc_2.neff
JSON export: /tmp/nxdi_attn_138311351493953_vnc0.json
```

#### LNC / virtual-core execution

The NEFF is a virtual-core compile:
- `number_of_neuroncores_per_lnc = 2`
- `number_of_logical_neuroncores = 1`
- `enabled_features` includes `virtual-core` and `v3-2x-4x-dve-perf-mode`
- `number_of_cc_participants = 4`, matching TP=4 collectives, not TP*LNC

Runtime `model_info` has two physical NC subgraphs under that one logical NC:

| Subgraph | Physical NC | Source-attributed instructions | Notes |
|----------|-------------|--------------------------------|-------|
| `sg00` | ND0 NC4 | 123,488 | 512 MiB shared scratchpad |
| `sg01` | ND0 NC5 | 118,418 | 512 MiB shared scratchpad |

Attention HLOs are duplicated across the two subgraphs. For a steady-state layer (`NeuronQwen3MoEAttention[.50]`), the split is exactly 428 instructions on `sg00` and 428 on `sg01`; the same HLO names appear on both subgraphs with matching tile counts. The first attention module (`.49`) has extra RoPE table-generation/source-attribution noise (6,883 instructions, split 3,451/3,432), so layers `.50` through `.96` are cleaner representatives of normal per-layer TKG attention.

The profile does not expose a named "batch axis" sharding annotation, but the NEFF node table shows per-item cache tensors shaped `[1 1 640 128]` and two scalar `[1 1]` token-position style inputs for the `bk2` model. Combined with the duplicated per-HLO execution and the absence of cross-subgraph attention collectives, the practical execution model is: the batch bucket is split into two independent physical-NC work items inside the LNC. Do not combine softmax denominators, QK partial sums, or `attn @ V` partial sums across `sg00`/`sg01` for a single token. Match one subgraph's local reduction order; the other subgraph runs the same sequence for the other bucket item.

All recorded CC ops are TP collectives on `sg00` with replica group `[[0, 1, 2, 3]]`: 96 BF16 `AllReduce(num_elements=2048)` ops, plus the embedding/final `AllGather`s. There is no attention-softmax LNC collective.

#### Steady TKG attention lowering (`NeuronQwen3MoEAttention[.50]`)

Representative per-subgraph lowering:

| Stage | HLO examples | Device lowering / bit-exact note |
|-------|--------------|----------------------------------|
| Q projection | `%dot.12` | 64 `128*128` PE matmul tiles per subgraph. BF16 inputs, F32 PSUM, copied to FP32 SBUF before downstream attention prep. |
| K/V projections | `%dot.14`, `%dot.17` | 8 `128*128` PE matmul tiles per subgraph for each projection. BF16 inputs, F32 PSUM, copied to FP32 SBUF. Later Q/K/V values consumed by score/output matmuls are BF16 after the norm/RoPE/cache path. |
| Prior QK scores | `%dot.13` | 5 `128*128` PE matmul tiles per subgraph for the 640-token cache bucket. The scale is applied in the Scalar epilogue as `scale=0.088379`, and the scaled scores are stored as BF16. |
| Active QK score | `%dot.15` | 1 `128*1` PE matmul tile per subgraph. Same Scalar scale `0.088379`, stored as BF16. |
| Mask | `%select.689` | Lowered to `COPY_PREDICATED_SCALAR` on the BF16 score buffer before the softmax reductions. |
| Prior max | `%reduce.697` | Vector `TENSOR_REDUCE op=MAX dim=X`, BF16 source, FP32 destination. For non-speculative TKG there is no active-score reduce; `manual_softmax` compares `max(prior)` directly with the singleton `active_scores`. |
| Exp | `%exponential.734`, `%exponential.749` | One `ACT_TABLE_LOAD` per subgraph before the first exp, then Scalar `ACTIVATE EXP` for active and prior. |
| Prior exp sum | `%reduce.756` | Lowered to Tensor/TensorMatrix, not Vector reduce: 10 `LDWEIGHTS` + 10 `MATMUL 128*1` instructions per subgraph. This PE reduction order is part of the bit-exact behavior. |
| Prior divide | `%divide.778` | 4 `LOAD_MASK_SELECT` + 4 `STREAM_SHUFFLE` + 1 `RECIPROCAL` + 1 Vector multiply per subgraph. The multiply consumes FP32 numerator/reciprocal and writes BF16 softmax-prior. |
| Active softmax and `active @ V` | `%multiply.202` | The singleton active branch is not materialized as a separate BF16 softmax tensor plus PE matmul. The compiler scales `V_active` directly by the scalar active softmax and writes a BF16 active contribution. |
| Prior `softmax @ V` + active add | `%dot.16` | 5 `128*128` PE matmul tiles per subgraph. The epilogue fuses the add with the active contribution: `src0=fp32` prior PSUM, `src1=bfloat16` active contribution, `dst=bfloat16`. Do not round prior `attn_prior` to BF16 before adding active if trying to match this lowering. |
| Output projection | `%dot.19` | 65 `128*128` PE matmul tiles per subgraph, then Scalar `CAST` FP32 -> BF16. The TP BF16 all-reduce is recorded as a CC op on `sg00`. |

This refines the earlier logical recipe: HLO still expresses `manual_softmax`, but the codegen for this artifact performs several numerically relevant fusions. In particular, the prior softmax denominator is a PE `128*1` reduction, active softmax is folded into the active-value scaling, and the final prior/active attention-output add is fused into the prior-output epilogue.

---

## 1. Model Configuration & Tensor Sharding

### 1.1 Global vs Per-Rank Dimensions

| Attribute | Global Value | Per-Rank (TP=4) | Notes |
|-----------|--------------|-----------------|-------|
| `hidden_size` | 2048 | 512 per rank | TP shards hidden dimension |
| `num_attention_heads` | **32** | 8 per rank | TP shards heads uniformly (CORRECTED: was 16) |
| `num_key_value_heads` | 4 | 1 per rank | TP shards KV heads uniformly |
| `head_dim` | 128 | 128 (not sharded) | Head dimension is replicated per rank |
| `GQA ratio (global)` | 32:4 = 8:1 | 8:1 | On each rank: 8 Q heads : 1 KV head |
| `rms_norm_eps` | 1e-6 | 1e-6 | Shared across ranks |

### 1.2 Weight Matrices (Per-Rank Shapes)

**QKV Projection — Verified from HLO as SEPARATE Matmuls (NOT fused):**

The HLO entry computation contains separate `dot` operations:
- **Q-projection:** `(1,1,2048) @ (2048,1024) → (1,1,1024)` — `%dot.128`-series (4× per rank for TP=4)
- **K-projection:** `(1,1,2048) @ (2048,128) → (1,1,128)` — `%dot.37`-series or `%dot.83`-series
- **V-projection:** `(1,1,2048) @ (2048,128) → (1,1,128)` — separate matmul (not fused with K)

**HLO verification:** The profile shows 2,632 `dot` operations in layer .49. If QKV were fused into one, we'd see ~1/3 of that. The separate operations confirm the compiler **split the fused weight matrix back into three matmuls** at a late stage (likely during tensorization for systolic PE layout).

- Weight shape per rank: [2048, 1280] in abstract (fused Wqkv in Python)
  - But HLO has three separate dots with disjoint output dims
  - Accumulation dtype: BF16 for each matmul (inputs BF16, PSUM in F32 due to `--enable-mixed-precision-accumulation`)
- Output dtype after CONVERT: BF16 (implicit in the three separate outputs)

**Output Projection (`o_proj`, `RowParallelLinear`):**
- Weight shape per rank: `[512, 2048]` (transposed to `[2048, 512]` in memory)
- Input to `o_proj`: (1, 1, 512) from attention heads on this rank
- Bias: None
- All-Reduce across TP=4 after matmul (sum contributions)
- Output dtype: BF16

### 1.3 KV Cache Layout

```
K_cache: (batch=1, num_kv_heads=1, seq_len, head_dim=128)  dtype=BF16
V_cache: (batch=1, num_kv_heads=1, seq_len, head_dim=128)  dtype=BF16
k_cache_transposed: False  (cache is in BHSD layout, not transposed)
```

---

## 2. Attention Execution Flow — Step by Step with HLO & Hardware Ground Truth

### Step 1: Input LayerNorm (`input_layernorm`, `CustomRMSNorm`)

**Input:**
```
hidden_states: (1, 1, 2048)  BF16
```

**HLO & Execution:**
The layer norm is invoked via the call chain `RmsNormForwardImpl.X → HloRmsNormForwardImpl.Y → custom-call(target="AwsNeuronRmsNorm")`. The actual numerical recipe inside the custom-call is **opaque** (no `rsqrt` opcode appears anywhere in the HLO). The pseudocode below describes the *intended* recipe per the NxDI Python source — but the bit-exact behavior is whatever `AwsNeuronRmsNorm` does.

```
# Cast to FP32 (BF16→F32 convert at custom-call input boundary)
x_fp32 = cast(hidden_states, FP32)  # (1, 1, 2048)

# RMS computation
x_sq = x_fp32 * x_fp32
mean = reduce_mean(x_sq, axes=[2], keepdims=True)  # → (1, 1, 1)  FP32

# Rsqrt: opaque inside AwsNeuronRmsNorm. Likely Scalar-engine reciprocal-sqrt (LUT-based).
rms_inv = rsqrt(mean + eps)                       # → (1, 1, 1)  FP32  [conceptual]

# Normalization (multiply by gamma in F32, then convert to BF16)
x_norm = x_fp32 * rms_inv                         # (1, 1, 2048)  FP32

# Scale by gamma and cast down (Scalar: CAST, then Vector: TENSOR_TENSOR)
gamma: (2048,)  FP32
output = cast(x_norm * gamma, BF16)              # (1, 1, 2048)  BF16
```

**Engine assignment (from profile):**
- TENSOR_REDUCE (Vector): square, reduce
- ACTIVATION (Scalar): rsqrt
- TENSOR_TENSOR (Vector/Scalar): multiply by gamma
- CAST (Scalar): FP32 → BF16 final output

**Precision invariants:**
- eps = 1e-6 (FP32)
- Mean computed via `reduce_mean(x²)`, not `sum(x²)/N`
- Final cast is BF16

---

### Step 2: QKV Projection (VERIFIED: Separate Matmuls in HLO)

**Input:**
```
hidden_states_norm: (1, 1, 2048)  BF16
```

**HLO Structure:**
Three separate `dot` operations (NOT fused):
- Q: `%dot.128` series — (1, 1, 2048) BF16 @ (2048, 1024) BF16 → (1, 1, 1024) BF16
- K: `%dot.83` / `%dot.37` — (1, 1, 2048) BF16 @ (2048, 128) BF16 → (1, 1, 128) BF16
- V: separate — (1, 1, 2048) BF16 @ (2048, 128) BF16 → (1, 1, 128) BF16

**Hardware Execution (from profile):**
- **Profile evidence:** 2,632 `dot` operations in layer .49, broken down as:
  - %dot.10: 1024 ops (largest, likely the Q matmul with large output)
  - %dot.11: 516 ops (K or V)
  - %dot.114, %dot.116, %dot.119, etc.: remaining K/V splits
- **Engines:**
  - Tensor (LDWEIGHTS): 130–512 ops per dot variant (weight loading)
  - TensorMatrix (MATMUL): 128–512 ops per dot variant (PE systolic)
  - Each LDWEIGHTS ~185 ps, each MATMUL ~260 ps

**Precision invariants:**
- Matmul in BF16 (PSUM internally F32, then CONVERT to BF16 output)
- No upcast during accumulation in the output; the PE handles F32→BF16 conversion

**HLO references:** Entry computation instructions `%dot.128`, `%dot.83`, `%dot.37`, etc. (names vary per trace)

---

### Step 3: QK LayerNorm (Pre-RoPE, Per-Head)

**Applied per-head on `head_dim=128` after reshape.**

**Input:**
```
Q: (1, 1, 1024)  BF16  (8 heads × 128 dims)
K: (1, 1, 128)   BF16  (1 head × 128 dims)
```

**HLO & Execution:**

Reshape and apply norm:
```
# Reshape for per-head norm
Q_reshaped = reshape(Q, (1, 8, 128))      # (1, 8, 128)  BF16
K_reshaped = reshape(K, (1, 1, 128))      # (1, 1, 128)  BF16

# Per-head RMS norm on last axis (head_dim=128)
# Vector TENSOR_REDUCE for reduce_mean
# Scalar ACTIVATION for rsqrt
# Vector TENSOR_TENSOR for multiply & CAST
```

**Engine assignment:**
- TENSOR_REDUCE (Vector): reduce on axis=-1
- ACTIVATION (Scalar): rsqrt
- TENSOR_TENSOR (Vector): multiply by gamma
- CAST (Scalar/Vector): FP32 → BF16

**Precision invariants:**
- gamma_q, gamma_k: shape (128,), dtype FP32 (per-head scales)
- RMS computed on FP32
- Final cast to BF16
- Reduction axis: last dimension (head_dim=128)
- Epsilon: 1e-6 (FP32)

---

### Step 4: Rotary Position Embedding (RoPE)

**Input:**
```
Q: (1, 8, 128)  BF16  [after layer norm]
K: (1, 1, 128)  BF16
position_ids: (1, 1) with value = current_seq_pos (e.g., 0 for first token)
```

**HLO & Execution:**

RoPE cache generation and application (per forward call):
```
# RoPE frequencies (RotaryEmbedding.forward)
dim = 128
freq_indices = [0, 2, 4, ..., 126]  (even indices only)
inv_freq = 1.0 / (10000.0 ** (freq_indices / 128))  # shape (64,)

# Compute cos/sin for current position
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]  # (1, 64, 1)
emb = cat([freqs, freqs], dim=-1)  # (1, 64, 1) → (1, 1, 128) by reshaping
cos_cache = cos(emb).to(BF16)      # (1, 1, 128)  BF16
sin_cache = sin(emb).to(BF16)      # (1, 1, 128)  BF16
```

**HLO verification — Cosine/Sine ops are MASSIVE:**
- Profile shows **3,660 `cosine` operations** and **2,361 `sine` operations** in layer .49 alone
- Total duration: 847.7 µs (cosine) + 436.7 µs (sine) = 1.28 ms
- **Explanation:** The `%cosine.105` and `%sine.76` ops appear as table lookups with DMA access to precomputed cos/sin tables in HBM
  - Each cosine/sine is a PSEUDO_DMA_DIRECT2D operation (DMA descriptor generation on Sync, actual transfer on DMA queue)
  - Avg 231.6 ps per cosine, 185.0 ps per sine — consistent with table lookup latency
  - The high count suggests the compiler generates the full cos/sin table (e.g., for all max_position_embeddings=2048 positions) and slices the needed value

**RoPE Application (no upcast):**
```
# Helper: rotate_half(x) = cat([-x[..., 64:], x[..., :64]], dim=-1)
def _rotate_half(x):
  x1 = x[..., :64]
  x2 = x[..., 64:]
  return cat([-x2, x1], dim=-1)  # Reordered interleave in BF16

# Apply to Q and K (all BF16, no FP32 upcast)
Q_rot = (Q * cos_cache) + (_rotate_half(Q) * sin_cache)  # All BF16
K_rot = (K * cos_cache) + (_rotate_half(K) * sin_cache)  # All BF16

Output:
  Q_rot: (1, 8, 128)  BF16
  K_rot: (1, 1, 128)  BF16
```

**Engine assignment:**
- Cosine/sine: Sync/GpSimd (PSEUDO_DMA_DIRECT2D) for table loads
- Multiply/add: Vector (TENSOR_TENSOR), Scalar (ACTIVATE for some arithmetic)

**Precision invariants:**
- cos/sin cache: BF16 (computed as `cos(emb).to(BF16)`)
- rotate_half: performed in BF16, no upcast
- Output: BF16

---

### Step 5: KV Cache Update

**Current state:**
```
K_rot: (1, 1, 128)  BF16
V: (1, 1, 128)  BF16
K_cache: (1, 1, seq_prior, 128)  BF16  [built up over prior tokens]
V_cache: (1, 1, seq_prior, 128)  BF16
position_id: scalar indicating where to write (0-indexed in [0, seq_prior))
```

**HLO & Execution:**

Update (scatter-like operation, likely `dynamic-update-slice` in HLO):
```
# In HLO: dynamic-update-slice or gather scatter
K_cache = dynamic_update_slice(K_cache, K_rot, [0, 0, position_id, 0])
V_cache = dynamic_update_slice(V_cache, V_rot, [0, 0, position_id, 0])

# After update:
K_cache: (1, 1, seq_prior+1, 128)  BF16
V_cache: (1, 1, seq_prior+1, 128)  BF16
```

**Profile evidence (scatter/gather):**
- 184 `gather` operations in layer .49
- Engines: GpSimd (ALU_OP, MOVE, GATHER, TENSOR_LOAD), Sync (DMA_DIRECT2D), Vector (STREAM_SHUFFLE, LOAD_MASK_SELECT)
- Timing: 39.6–349.7 ps per operation (TENSOR_LOAD is slowest due to memory access)

**Precision invariants:**
- No dtype conversion; scatter is in BF16
- Layout is BHSD (not transposed), confirmed by `k_cache_transposed=False` in config
- Position dimension is axis 2 (sequence dimension)

---

### Step 6: GQA Head Expansion (repeat_kv)

**Purpose:** Expand KV heads to match Q heads for grouped query attention.

**Input:**
```
K_cache: (1, 1, seq_prior+1, 128)  BF16
V_cache: (1, 1, seq_prior+1, 128)  BF16
num_key_value_groups = 8  (8 Q heads per 1 KV head on this rank)
```

**Computation:**
```
# repeat_kv function: expand along head dimension
def repeat_kv(hidden_states, n_rep):
  batch, num_kv_heads, slen, head_dim = hidden_states.shape  # (1, 1, seq_prior+1, 128)
  if n_rep == 1:
    return hidden_states
  # Expand: add dimension and replicate
  expanded = hidden_states[:, :, None, :, :]  # (1, 1, 1, seq_prior+1, 128)
  expanded = expanded.expand(batch, num_kv_heads, n_rep, slen, head_dim)  # (1, 1, 8, seq_prior+1, 128)
  return expanded.reshape(batch, num_kv_heads * n_rep, slen, head_dim)  # (1, 8, seq_prior+1, 128)

K_exp = repeat_kv(K_cache, 8)  # (1, 8, seq_prior+1, 128)  BF16
V_exp = repeat_kv(V_cache, 8)  # (1, 8, seq_prior+1, 128)  BF16
```

**Output:**
```
K_exp: (1, 8, seq_prior+1, 128)  BF16
V_exp: (1, 8, seq_prior+1, 128)  BF16
```

**Precision invariants:**
- No dtype conversion; repeat is identity-like reshape + broadcast
- Output dtype: BF16

---

### Step 7a: Prior Attention Scores (QK^T with history)

**Input:**
```
Q_rot: (1, 8, 128)  BF16
K_exp: (1, 8, seq_prior+1, 128)  BF16
softmax_scale = 1 / sqrt(128) ≈ 0.08838
```

**HLO & Execution:**

```
# Transpose K for matmul
K_T = transpose(K_exp, (0, 1, 3, 2))  # (1, 8, 128, seq_prior+1)  BF16

# Matmul with scale (TensorMatrix PE)
prior_scores_bf16 = matmul(Q_rot, K_T)  # (1, 8, 1, seq_prior+1)  BF16
                                        # PE accumulation in F32, output CONVERT to BF16
prior_scores_bf16 = prior_scores_bf16 * softmax_scale  # (1, 8, 1, seq_prior+1)  BF16

# **IMMEDIATE UPCAST before mask** (Vector TENSOR_TENSOR or CAST)
prior_scores = cast(prior_scores_bf16, FP32)  # (1, 8, 1, seq_prior+1)  FP32
```

**Hardware evidence:**
- Matmul: TensorMatrix (MATMUL 260 ps avg) + Tensor (LDWEIGHTS 185 ps avg)
- Scale: Vector (TENSOR_TENSOR multiply)
- Cast: Vector (CAST/TENSOR_TENSOR)

**Causal Mask (additive):**
```
# attention_mask: boolean (1, 1, 1, seq_prior+1)
# True where attention is allowed (past tokens), False for future (not applicable in TKG)
prior_scores = where(attention_mask, prior_scores, torch.finfo(FP32).min)  # -3.402823e38
# Output: (1, 8, 1, seq_prior+1)  FP32
```

**Precision invariants:**
- Q and K matmul in BF16
- Immediate cast to FP32 **before** masking and softmax
- Scale applied as multiplication (equivalent to `scale * Q @ K^T`)
- Mask value: `torch.finfo(torch.float32).min` (large negative number in FP32)

---

### Step 7b: Active Attention Scores (new token)

**Input:**
```
Q_rot: (1, 8, 128)  BF16
K_rot: (1, 1, 128)  BF16 [the new token's K, not yet cached]
V: (1, 1, 128)  BF16 [the new token's V]
```

**HLO & Execution:**

```
# GQA expand for active (same repeat_kv logic)
K_active_exp = repeat_kv(K_rot, 8)  # (1, 8, 1, 128)  BF16
V_active_exp = repeat_kv(V, 8)      # (1, 8, 1, 128)  BF16

# Matmul with scale (TensorMatrix PE)
active_scores_bf16 = matmul(Q_rot, transpose(K_active_exp, (0, 1, 3, 2)))
                                    # (1, 8, 1, 1)  BF16
active_scores_bf16 = active_scores_bf16 * softmax_scale  # (1, 8, 1, 1)  BF16

# **IMMEDIATE UPCAST**
active_scores = cast(active_scores_bf16, FP32)  # (1, 8, 1, 1)  FP32
```

**Precision invariants:**
- Same as prior: matmul in BF16, then upcast to FP32

---

### Step 8: Manual Softmax — Plain HLO Ops; Device Lowering Uses Scalar LUT for `exp` and Vector RECIPROCAL for `divide`

**Verified from HLO:** Softmax is **NOT** a custom-call. The HLO entry computation contains the explicit ops (per-layer): `maximum` → `subtract` → `exponential` → `reduce` (sum) → `divide`, all in F32. Counts (48 layers): 144 `exponential`, 144 `subtract`, 144 `maximum`, 290 `divide`, 384 `reduce`.

**Device lowering** (compiler_opcode in profile JSON, BIR-level, below HLO):
- **`exp` → Scalar engine:** `ACT_TABLE_LOAD` (~1283 ps) + `ACTIVATE` (~347 ps). LUT-based approximation.
- **`divide` → Vector + Scalar:** `RECIPROCAL` (Vector, ~184 ps) + `ACTIVATE` (Scalar, ~370 ps). i.e. `1/denom × numerator`, not a true f32 divide.
- **`reduce` (sum/max) → Vector:** `TENSOR_REDUCE`.
- **`maximum` (global max of prior+active) → Vector:** `TENSOR_TENSOR`.
- The compiler log line "48 Native Softmax's detected and replaced" refers to a BIR-level pattern fusion that staples these ops together for scheduling — the HLO numerics remain plain log-sum-exp.

**Input:**
```
prior_scores: (1, 8, 1, seq_prior+1)  FP32
active_scores: (1, 8, 1, 1)  FP32
is_speculation: False
```

**Logical Computation (what the Python code does, but NOT what the hardware runs):**

```python
# Step 1: Find global max across prior and active
max_prior = max(prior_scores, dim=-1, keepdim=True)          # (1, 8, 1, 1)  FP32
max_active = max(active_scores, dim=-1, keepdim=True)        # (1, 8, 1, 1)  FP32
global_max = maximum(max_prior, max_active)                  # (1, 8, 1, 1)  FP32

# Step 2: Subtract max (numerically stable)
exp_prior = exp(prior_scores - global_max)                   # (1, 8, 1, seq_prior+1)  FP32
exp_active = exp(active_scores - global_max)                 # (1, 8, 1, 1)  FP32

# Step 3: Sum exps (denominator)
denom_prior = sum(exp_prior, dim=-1, keepdim=True)           # (1, 8, 1, 1)  FP32
denom_active = sum(exp_active, dim=-1, keepdim=True)         # (1, 8, 1, 1)  FP32
denom = denom_prior + denom_active                           # (1, 8, 1, 1)  FP32

# Step 4: Compute softmax (all FP32)
softmax_prior = exp_prior / denom                            # (1, 8, 1, seq_prior+1)  FP32
softmax_active = exp_active / denom                          # (1, 8, 1, 1)  FP32

# Step 5: **Cast down to BF16 before attention matmul**
softmax_prior = cast(softmax_prior, BF16)                    # (1, 8, 1, seq_prior+1)  BF16
softmax_active = cast(softmax_active, BF16)                  # (1, 8, 1, 1)  BF16
```

**Hardware Reality (from profile):**
- **exponential:** 6 ops (HLO level) implemented as:
  - 2× ACT_TABLE_LOAD (Scalar, 1283 ps avg) — load from LUT
  - 4× ACTIVATE (Scalar, 347.5 ps avg) — apply table result
- **divide:** 6 ops implemented as:
  - 3× RECIPROCAL (Vector, 184.3 ps avg) — compute 1/denom
  - 3× ACTIVATE (Scalar, 370 ps avg) — multiply by reciprocal
- **reduce:** 16 ops (max, sum) distributed across Vector (TENSOR_REDUCE) and Scalar (DMA, COPY)
- **maximum:** 2 ops (Vector TENSOR_TENSOR)
- **subtract:** 6 ops (Scalar ACTIVATE)

**Precision invariants:**
- **All intermediate ops in FP32** (max, subtract, exp, sum, divide)
- **Reduction order:** max(dim=-1) → exp(x - max) → sum(dim=-1) → divide (log-sum-exp)
- **Final cast to BF16** (input to attention output matmul)
- **EXP PRECISION:** The LUT approximation is NOT `numpy.exp()` — verify with actual kernel precision
- **DIVIDE PRECISION:** Reciprocal is ~24-bit precision (not full FP32)

---

### Step 9: Attention Output (weighted sum over values)

**Input:**
```
softmax_prior: (1, 8, 1, seq_prior+1)  BF16
softmax_active: (1, 8, 1, 1)  BF16
V_exp: (1, 8, seq_prior+1, 128)  BF16
V_active_exp: (1, 8, 1, 128)  BF16
```

**HLO & Execution:**

```
# Matmul: attention-weighted sum (TensorMatrix PE)
attn_prior = matmul(softmax_prior, V_exp)      # (1, 8, 1, seq_prior+1) @ (1, 8, seq_prior+1, 128)
                                                # → (1, 8, 1, 128)  BF16

attn_active = matmul(softmax_active, V_active_exp)  # (1, 8, 1, 1) @ (1, 8, 1, 128)
                                                     # → (1, 8, 1, 128)  BF16

# Sum prior and active contributions (Vector TENSOR_TENSOR)
attn_output = attn_prior + attn_active         # (1, 8, 1, 128)  BF16
```

**Hardware evidence:**
- Matmul: TensorMatrix (MATMUL 260 ps avg) + Tensor (LDWEIGHTS 185 ps avg)
- Add: Vector (TENSOR_TENSOR)

**Precision invariants:**
- Softmax weights and values both BF16
- Matmul in BF16 (PE accumulator F32, output CONVERT to BF16)
- Addition in BF16

---

### Step 10: Reshape & Output Projection

**Input:**
```
attn_output: (1, 8, 1, 128)  BF16 [BHSD layout]
```

**Reshape for output projection:**
```
attn_output_flat = reshape(attn_output, (1, 1, 1024))  # (1, 1, 1024)  BF16
                   # Flattens (1, 8, 1, 128) → (1, 1, 8*128)
```

**Output Projection (RowParallelLinear, o_proj):**
```
# Per-rank weight: [1024, 2048]
W_out = o_proj.weight  # [1024, 2048]  BF16

output_bf16 = matmul(attn_output_flat, W_out.T)  # (1, 1, 1024) @ (2048, 1024).T
                                                  # → (1, 1, 2048)  BF16
```

**All-Reduce across TP=4:**
```
# RowParallelLinear gathers output from all ranks
output_full = all_reduce(output_bf16, op=SUM, group=TP_group)  # (1, 1, 2048)  BF16
```

**Hardware evidence:**
- Matmul: TensorMatrix (MATMUL) + Tensor (LDWEIGHTS)
- All-reduce: GpSimd (WRITE opcode, 230 ps avg for 2 ops)

**Precision invariants:**
- Matmul in BF16
- All-Reduce sum in BF16
- Output dtype: BF16

---

### Step 11: Residual Connection

**Input:**
```
residual: (1, 1, 2048)  BF16 [original hidden_states, or after attention residual]
attn_output: (1, 1, 2048)  BF16
```

**Computation:**
```
hidden_states = residual + attn_output  # (1, 1, 2048)  BF16
```

---

## 3. Precision Summary Table

| Operation | Input dtypes | Compute dtype | Output dtype | HLO Engine(s) | Compiler Opcode | Notes |
|-----------|--------------|---------------|--------------|---------------|-----------------|-------|
| input_layernorm | BF16 | FP32 (cast-up) | BF16 | Vector/Scalar | TENSOR_REDUCE, ACTIVATION, CAST | RMS on FP32, γ in FP32, final cast BF16 |
| Wqkv matmul (3× separate) | BF16, BF16 | BF16 (PE F32, CONVERT) | BF16 | TensorMatrix/Tensor | MATMUL, LDWEIGHTS | No upcast in final output; CONVERT at PE boundary |
| q_layernorm, k_layernorm | BF16 | FP32 (cast-up) | BF16 | Vector/Scalar | TENSOR_REDUCE, ACTIVATION, CAST | 128-dim RMS on FP32, γ per-head FP32 |
| RoPE (cos, sin cache) | — | FP32 | BF16 | Sync/GpSimd | PSEUDO_DMA_DIRECT2D | Table lookups (DMA from HBM), cast to BF16 |
| _rotate_half | BF16 | BF16 | BF16 | Vector/Scalar | TENSOR_TENSOR, STREAM_SHUFFLE | No upcast for interleave |
| K, V cache update | BF16 | BF16 | BF16 | GpSimd/Vector | GATHER, ALU_OP, TENSOR_LOAD | dynamic-update-slice in BF16 |
| repeat_kv (GQA expand) | BF16 | BF16 | BF16 | Vector | (reshape/broadcast, no compute) | Identity reshape, no dtype change |
| prior_scores (Q @ K^T) | BF16, BF16 | BF16 (PE F32) → **FP32** | FP32 | TensorMatrix/Vector | MATMUL, LDWEIGHTS, CAST | Matmul in BF16, **immediate upcast before mask** |
| active_scores (Q @ K^T) | BF16, BF16 | BF16 (PE F32) → **FP32** | FP32 | TensorMatrix/Vector | MATMUL, LDWEIGHTS, CAST | Matmul in BF16, **immediate upcast before mask** |
| Causal mask (additive) | bool, FP32 | FP32 | FP32 | Vector | COPY_PREDICATED_SCALAR | Large-negative value for False positions |
| manual_softmax (max, exp, sum, div) | FP32, FP32 | **FP32 (exp via LUT, div via reciprocal)** | FP32 → **BF16** | Scalar/Vector | ACT_TABLE_LOAD, ACTIVATE, RECIPROCAL, TENSOR_REDUCE | **Exp is LUT (not true exp), Divide is reciprocal-based** |
| attn @ V (prior) | BF16, BF16 | BF16 (PE F32) | BF16 | TensorMatrix/Vector | MATMUL, LDWEIGHTS | Softmax weights and values both BF16 |
| attn @ V (active) | BF16, BF16 | BF16 (PE F32) | BF16 | TensorMatrix/Vector | MATMUL, LDWEIGHTS | Same |
| attn_prior + attn_active | BF16, BF16 | BF16 | BF16 | Vector | TENSOR_TENSOR | Addition in BF16 |
| o_proj (RowParallel) | BF16, BF16 | BF16 (PE F32) | BF16 | TensorMatrix/Vector/GpSimd | MATMUL, LDWEIGHTS, WRITE (all-reduce) | Matmul and all-reduce in BF16 |
| Residual add | BF16, BF16 | BF16 | BF16 | Vector | TENSOR_TENSOR | Element-wise add in BF16 |

---

## 4. Compilation & Execution Flags

**Key compiler arguments** (from log-neuron-cc.txt:1):
```
--enable-mixed-precision-accumulation
--enable-saturate-infinity
--enable-internal-neff-wrapper
-O1
--model-type transformer
--auto-cast=none
--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2
--internal-hlo2tensorizer-options=--verify-hlo=true
```

**Instruction histogram (entry computation of HLO module proto)**:
- 530 `dot` ops (separate Q/K/V/QK^T_active/attn@V_prior/attn@V_active/o_proj per layer; ~11/layer × 48)
- 1881 `convert` ops (BF16↔F32 cast points)
- 1590 `constant` ops (eps, scale, masks, softmax constants)
- 144 `exponential`, 144 `subtract`, 144 `maximum`, 290 `divide`, 384 `reduce` (the explicit log-sum-exp softmax across 48 layers)
- 96 `all-reduce` (2 per layer: o_proj all-reduce + MoE expert all-reduce)
- 0 `custom-call` ops in entry (custom-calls live inside `HloRmsNormForwardImpl` / `HloSiluForwardImpl` / `HloTopKImpl` sub-computations)

**Custom-call population** (verified by walking all 1395 sub-computations):
| Target | Count | Used in attention path? |
|--------|-------|--------------------------|
| `AwsNeuronRmsNorm` | 193 | YES — input/post-attn/Q/K layernorms (4 per layer × 48 layers + 1 final) |
| `AwsNeuronSilu` | 48 | NO — MoE expert path |
| `AwsNeuronTopK` | 48 | NO — sampling |
| `AwsNeuronModuleMarkerStart-Forward` | 48 | NO — compiler hint |
| `AwsNeuronModuleMarkerEnd-Forward` | 48 | NO — compiler hint |
| **Softmax** | **0** | **No softmax custom-call exists in the HLO.** |

**BIR-level softmax fusion (below HLO):**
- The compiler log line "48 Native Softmax's detected and replaced" refers to a BIR pattern fusion that combines the HLO's plain `subtract`/`exp`/`reduce`/`divide` ops into a single fused softmax microkernel during code generation.
- The HLO numerics remain plain log-sum-exp; the *device* execution uses Scalar-engine LUT for `exp` and Vector RECIPROCAL for `divide`.

---

## 5. Bit-Exact Reproduction Checklist

To match the HLO output bit-for-bit in an NKI fused attention kernel, the following **must be preserved:**

### 5.1 RMSNorm Execution
- [ ] **Input LayerNorm:** Cast input to FP32 before operations
- [ ] **RMS computation:** Reduce mean on squared values (not sum/N separately), then rsqrt
- [ ] **Epsilon:** 1e-6 in FP32, added inside rsqrt (not before)
- [ ] **Gamma application:** Multiply in FP32 before final cast to BF16
- [ ] **Per-head norms (q_layernorm, k_layernorm):** Reduce on axis=-1 (head_dim=128) only
- [ ] **Final output cast:** BF16 (no FP32 residual)

### 5.2 QKV Projection
- [ ] **Weight accumulation:** BF16 inputs with F32 PSUM accumulation on the PE. Match the observed epilogue/cast point for the artifact, not just the logical dot shape.
- [ ] **No bias:** Fused weights as three separate matmuls in HLO; no bias addition
- [ ] **QKV shape:** Q=[1,1,1024], K=[1,1,128], V=[1,1,128] (8×128, 1×128, 1×128 per-head)
- [ ] **Output dtype:** The 2026-04-27 artifact copies Q/K/V projection PSUM to FP32 SBUF before downstream attention prep; later Q/K/V values consumed by score/output matmuls are BF16 after norm/RoPE/cache handling.

### 5.3 RoPE
- [ ] **cos/sin dtype:** BF16 (computed in FP32 via table lookup, cast to BF16 before use)
- [ ] **cos/sin storage:** Values loaded from HBM lookup table (precomputed for all positions up to max_seq_len)
- [ ] **_rotate_half:** Interleave as `[-x[64:], x[:64]]` in BF16 (no upcast)
- [ ] **No RoPE upcast:** Q and K remain BF16 after rotation

### 5.4 KV Cache Update
- [ ] **Layout:** BHSD (k_cache_transposed=False)
- [ ] **Dtype:** BF16 (no type conversion during scatter)
- [ ] **Position indexing:** 0-based, sequential (position_id corresponds to current seq position)
- [ ] **Dynamic-update-slice semantics:** overwrite a single [1, 1, 1, 128] slice at [:, :, position_id, :]

### 5.5 Attention Score Computation
- [ ] **QK matmul:** Execute in BF16 (both inputs BF16)
- [ ] **Scale factor:** 1/sqrt(128) as the observed Scalar epilogue constant `scale=0.088379`
- [ ] **Scale order:** `(Q @ K^T) * scale` (not `Q_scaled @ K^T`; both equivalent for bit-exactness)
- [ ] **Score storage before softmax:** The artifact stores scaled prior/active scores as BF16; softmax reductions consume those BF16 values and produce FP32 reduction results. Do not assume a fully materialized FP32 score tensor before the mask.
- [ ] **Mask dtype:** Boolean mask applied to the score buffer before the softmax reduction path.
- [ ] **Mask value:** `torch.finfo(prior_scores.dtype).min`; in this artifact the score buffer is BF16 at the `where`, so use the BF16 minimum value before the softmax reduction path.

### 5.6 Softmax (Plain HLO; Hardware-LUT exp & Reciprocal divide)
- [ ] **HLO is plain log-sum-exp** — softmax is NOT a custom-call. Implement explicit `max → subtract → exp → sum → divide`, all in F32.
- [ ] **`exp` precision:** Use NKI `nl.exp` — it lowers to the same Scalar-engine LUT path (`ACT_TABLE_LOAD` + `ACTIVATE`) as the compiler-emitted ops. Hand-rolled polynomial exp will diverge.
- [ ] **`divide` precision:** Use NKI's divide (lowers to RECIPROCAL + multiply) — matches the BIR-level lowering. Note: hardware reciprocal is not full f32 precision; this is the same loss the compiler accepts.
- [ ] **Reduction order:** non-speculative TKG computes `max(prior)` and then `maximum(max_prior, active_scores)` directly; `max(active)` only appears for speculation/prefix-caching where active length can exceed 1.
- [ ] **Exp stabilization:** `exp(scores - global_max)` — both prior and active subtract the same global max
- [ ] **Denominator:** logical HLO is `sum(exp_prior, dim=-1) + sum(exp_active, dim=-1)`, but this artifact lowers the prior sum to PE `128*1` reductions and folds the singleton active branch into later scaling.
- [ ] **Final cast/materialization:** Prior softmax is written BF16 before `attn_prior @ V_prior`; in this non-spec artifact the singleton active softmax is folded into active-value scaling rather than materialized as a separate tensor.
- [ ] **Logical intermediate dtype:** HLO softmax intermediates are F32 after conversion, but this codegen may fuse conversion into reductions/epilogues and read BF16 score storage directly.

### 5.7 Attention Output Matmul
- [ ] **Weights dtype:** BF16 (both softmax and V values)
- [ ] **Matmul accumulation:** Prior branch uses BF16 inputs with F32 PE PSUM; singleton active branch is a scalar scale of BF16 `V_active`.
- [ ] **Output dtype:** BF16 after prior + active addition. In the 2026-04-27 artifact, the active singleton contribution is BF16 and the prior PE epilogue adds FP32 prior PSUM + BF16 active contribution before writing BF16.

### 5.8 Output Projection
- [ ] **Weight matrix:** [1024, 2048] per rank (RowParallelLinear; gathered to full hidden_size after all-reduce)
- [ ] **Matmul dtype:** BF16
- [ ] **All-reduce:** SUM operation, BF16 dtype (scalar dtype of accumulated result)
- [ ] **No bias:** o_proj has no bias term

### 5.9 Reduction Order & Associativity
- [ ] **No reorder-associative reductions:** All reductions must match the HLO order exactly
- [ ] **Softmax max:** for non-spec TKG, max(prior), then max with the singleton active score; do not introduce an active reduce unless speculation/prefix-caching is enabled.
- [ ] **Softmax sum (if manual):** preserve the prior PE reduction order observed in the artifact; the singleton active term may be folded into the scalar active-value scale.
- [ ] **No fusion that changes order:** e.g., don't fuse max and exp into a single operation

### 5.10 Data Movement
- [ ] **GQA repeat_kv:** Logical reshape/expand; no dtype conversion
- [ ] **Transpose operations:** Only for layout changes (K^T for matmul); no numerical effects
- [ ] **Reshape operations:** Layout only; preserve exact bit patterns

### 5.11 Mixed-Precision Accumulation
- [ ] **PE matmul inputs:** BF16 Q and K
- [ ] **PE internal accumulation:** F32 in PSUM banks
- [ ] **PE output before CONVERT:** F32 (from PSUM)
- [ ] **CONVERT step:** Explicit F32 → BF16 conversion (one output per matmul)
- [ ] **Do NOT skip the CONVERT:** The hardware has this step; omitting it will cause bit divergence

### 5.12 LNC=2 Virtual-Core Behavior
- [ ] **Match per-subgraph reductions:** For `bk2` on this NEFF, `sg00` and `sg01` execute independent token/bucket work. Do not split one token's QK, softmax denominator, or `attn @ V` accumulation across the two physical NCs unless you also match the compiler's virtual-core partitioning.
- [ ] **TP collectives remain TP=4:** The attention output projection all-reduce is BF16 over replica group `[[0, 1, 2, 3]]`; LNC physical NCs are not extra TP participants.

---

## 6. Known Precision Differences & Implementation Notes

### 6.1 QKV Fusion Status (Verified Correction)

**Original claim:** "Fused Wqkv matmul" with output [1, 1, 768] split into Q/K/V.  
**HLO Reality:** Three separate `dot` operations in the entry computation with outputs [1, 1, 1024], [1, 1, 128], [1, 1, 128].  
**Explanation:** The compiler split the fused weight matrix into separate matmuls at the HLO-to-BIR translation stage for better PE scheduling. The Python source uses a fused weight matrix, but the compiled HLO expresses them as three independent dots.  
**Impact on NKI:** You must implement three separate matmuls, not one fused dot-then-slice operation.

### 6.2 Softmax Implementation (Critical Correction)

**Original assumption:** Manual softmax with Python-level exp and divide.  
**Device lowering reality:** Plain HLO softmax ops lowered to:
- **Exponential:** Lookup-table approximation (ACT_TABLE_LOAD), not true exp
- **Divide:** Reciprocal-based (RECIPROCAL + multiply), not true divide  
**Impact on NKI:** Bit-exact reproduction REQUIRES matching the LUT/reciprocal precision and the artifact's reduction/fusion points. A naive CPU-style `math.exp()` or a different denominator reduction will diverge from the softmax weights.

### 6.3 Accumulator Precision (--enable-mixed-precision-accumulation)

**With the flag enabled:**
- PE matmul accumulation is F32 (in PSUM)
- Output is explicitly converted to BF16 via a CONVERT instruction
- This differs from "BF16 accumulator" behavior (which would be more lossy)

**For NKI:** Ensure you accumulate in F32 and then convert to BF16. Using a BF16 accumulator from the start will give divergent results.

### 6.4 RoPE Table Size

**Observation:** Profile shows 3,660 cosine and 2,361 sine operations, suggesting the compiler precomputes cos/sin for all positions up to max_seq_len (likely 2048 or larger).  
**For NKI:** If you generate cos/sin on-the-fly, you'll be slower than the table lookup path. Consider caching these in HBM as the compiler does.

---

## 7. Custom-Call Opacity & Bit-Exact Reproduction Strategy

The only attention-path custom-call is `AwsNeuronRmsNorm` (193 instances). Softmax is plain HLO and reproducible.

**RMSNorm bit-exactness:**
- The custom-call's internal recipe is opaque (no `rsqrt` opcode visible in HLO). The likely device path is: square → reduce → add eps → Scalar-engine reciprocal-sqrt (LUT) → multiply gamma → convert to BF16. NKI `nl.rms_norm` lowers to the same path; using it should match.
- For a hand-rolled NKI RMSNorm: use `nl.rsqrt` (Scalar-engine LUT path), not `nl.sqrt` followed by `nl.reciprocal`, to match the compiler's likely op sequence.

**Softmax bit-exactness:**
- The HLO is plain `max → subtract → exp → reduce_sum → divide`, all in F32. Reproduction is straightforward:
  - Use `nl.exp` (Scalar LUT) — matches the compiler's lowering of HLO `exp`.
  - Use `nl.reciprocal` then multiply (or just `nl.divide`) — matches the compiler's BIR lowering of HLO `divide`.
- Reduction order matters: prior and active have separate `max` reductions and separate `sum` reductions, then are combined.

**QKV / attention-matmul bit-exactness:**
- All matmuls are plain HLO `dot` (no custom-call). Reproduction requires F32 accumulation in PSUM with explicit BF16 conversion at readout.

**Net assessment:** A faithful NKI rewrite using `nl.rms_norm` (or hand-rolled with `nl.rsqrt`), `nl.exp`, `nl.reciprocal`, and F32-accumulating matmuls should match the HLO/profile execution closely. The remaining unknowns are the *internal* implementation of `AwsNeuronRmsNorm` and the exact LUT coefficients for `exp` and `reciprocal` — but as long as your kernel uses the same Scalar/Vector engine paths, those are shared.

---

## 8. References & Source Code Locations

| Component | File | Key Lines | HLO/Profile Notes |
|-----------|------|-----------|-------------------|
| Input LayerNorm | qwen.py:363-366 | `self.input_layernorm = get_rmsnorm_cls()(..., eps=eps)` | Custom-call fused RMSNorm |
| QKV Projection | attention_base.py:513-515 | `self.get_qkv_proj()(hidden_states=..., rmsnorm=...)` | 3× separate `dot` ops in HLO |
| QK LayerNorm | qwen.py:342-343 | `self.q_layernorm = get_rmsnorm_cls()(head_dim, rms_norm_eps)` | Per-head reduce on axis=-1 |
| RoPE | utils.py:240-249 | `apply_rotary_pos_emb`, `_rotate_half` | Table lookup via PSEUDO_DMA_DIRECT2D |
| RoPE Frequencies | utils.py:306-343 | `RotaryEmbedding.forward` | Precomputed cos/sin tables in HBM |
| Attention Compute TKG | attention_base.py:1383-1461 | `compute_for_token_gen` method | Prior + active scores, custom softmax |
| Manual Softmax | utils.py:252-270 | `manual_softmax` with global max | HLO emits this verbatim — plain `max/sub/exp/reduce/divide` ops in F32 |
| Attention Output | attention_base.py:1457-1459 | `attn_prior @ V_prior + attn_active @ V_active` | Two matmuls + add |
| Output Projection | attention_base.py (row-parallel layer) | RowParallelLinear (o_proj) | TP all-reduce after matmul |
| Model Config | neuron_config.json:52-265 | `neuron_config` dict with TP=4, head_dim=128, etc. | Confirmed 32 global Q heads, 4 global KV heads |
| Compiler Log | log-neuron-cc.txt:1-200 | "48 Native Softmax's detected and replaced" | BIR-level pattern fusion of HLO log-sum-exp ops; not an HLO custom-call |
| Profile Data | /tmp/tkg_bk0_vnc2.json | 237,885 instructions | Layer .49 = first decoder layer |

---

## 9. Summary: Bit-Exact Reproduction Requirements

**For an NKI fused attention kernel to reproduce NxDI's HLO output bit-for-bit:**

1. **Follow the 10-step execution order:** Input RMSNorm (custom-call) → QKV (3 separate matmuls, F32 PSUM accum) → QK RMSNorm (custom-call) → RoPE (table lookup) → KV cache scatter → GQA broadcast → prior QK^T → active QK^T → softmax (plain HLO max/exp/sum/divide in F32, hardware-LUT exp, hardware-RECIPROCAL divide) → attention @ V → reshape → o_proj → all-reduce.

2. **Match all dtype transitions:** BF16 ↔ F32 at exact points: at custom-call RMSNorm boundaries; after QK score matmul the artifact scales into BF16 score storage, then softmax reductions produce FP32 reduction results; prior softmax weights are written BF16 before `attn @ V`. The HLO contains 1881 `convert` ops, but codegen may fuse the convert into reductions or matmul epilogues.

3. **PE accumulator precision:** F32 accumulation in PSUM, explicit CONVERT to BF16 output (NOT a BF16 accumulator).

4. **Softmax:** Plain log-sum-exp on F32 in HLO. For non-speculative TKG the max path is `max(prior)` then `maximum(max_prior, active_scores)`; the artifact lowers prior exp-sum to PE `128*1` reductions and folds the singleton active branch into active-value scaling. Use `nl.exp` (Scalar LUT) and reciprocal/multiply lowering compatible with the compiler path.

5. **No reordering of separable reductions:** Preserve the prior-cache reduction order and the singleton active path. A single combined prior+active reduce will not match this artifact.

6. **RMSNorm:** Use `nl.rms_norm` (or hand-rolled with `nl.rsqrt` on the Scalar engine) — matches the device path of `AwsNeuronRmsNorm` custom-call.

6. **Match all-reduce semantics:** BF16 sum across TP=4 for o_proj output.

7. **Verify cache layout & indexing:** BHSD, position_id-based scatter.

8. **Table lookup for RoPE:** Precompute cos/sin for all positions up to max_seq_len in HBM; fetch via PSEUDO_DMA.

9. **QKV as three separate matmuls:** Do NOT fuse into a single dot-then-slice; use three independent TensorMatrix calls.

10. **GQA repeat_kv:** Implement as logical reshape + broadcast, no dtype conversion.

11. **LNC=2:** Match one physical-NC subgraph's local order for one token/bucket item; do not introduce cross-LNC softmax or attention-output reductions.

---

**Document Version:** 2.1

**Last Updated:** 2026-04-27

**Verification:** Ground-truth from HLO proto (1395 computations, 17,624 entry instructions) + profile JSON (`/tmp/nxdi_attn_138311351493953_vnc0.json`, 241,906 source-attributed device instructions across `sg00`/`sg01`)

**Status:** Ready for NKI kernel implementation — validated against compiler output, not Python source.
