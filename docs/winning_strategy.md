# Winning Strategy: MLSys 2026 NKI-MoE Competition

Updated 2026-04-07.

---

## The Scoring Formula Is Quadratic

The actual scoring formula from `main.py` (lines 637-652):

```
SCORE = Accuracy * Reduced_Latency * Increased_Throughput * NKI_FLOP_Ratio
```

At **batch=1**, `Reduced_Latency = base_time / your_time` and `Increased_Throughput = your_throughput / base_throughput = base_time / your_time`. They are the **same number**. So:

**SCORE = Accuracy * (Speedup)^2 * NKI_FLOP_Ratio**

This is **quadratic in speedup**. Going from 1.9x to 2.5x doesn't gain 32% -- it gains `(2.5/1.9)^2 = 1.73x` more score. Every incremental speedup compounds dramatically.

| Speedup | Score (NKI=0.85) | vs Competitor (1.9x) |
|---|---|---|
| 1.37x (current) | 1.60 | 0.52x their score |
| 1.9x (competitor) | 3.07 | 1.0x |
| 2.2x | 4.11 | **1.34x** |
| 2.5x | 5.31 | **1.73x** |
| 3.0x | 7.65 | **2.49x** |

The game is not to match 1.9x. It is to blow past it, because the quadratic scoring rewards the gap exponentially.

---

## Current Per-Step Breakdown (48 layers)

Profiler data (single decode step):

| Component | Time | Source |
|---|---|---|
| Attn kernel x 48 | 2,640 us | 55 us x 48 |
| Attn AllReduce x 48 | ~576 us | ~12 us x 48 |
| MoE kernel x 48 | 4,944 us | 103 us x 48 |
| MoE AllReduce x 48 | ~576 us | ~12 us x 48 |
| Post-transformer (LM head + AllGather + 250 us dead) | 550 us | Profiler |
| **Accounted in NEFF** | **9,286 us** | |
| **NEFF total** | **9,340 us** | Profiler |
| **In-NEFF gap** | **~54 us** | Embedding + residuals |
| **Framework/harness** | **1,260 us** | 10,600 - 9,340 |
| **Benchmark total** | **10,600 us** | benchmark_report_best.json |

Key: MoE is 53% of NEFF time. It is purely memory-bandwidth bound at 0.5 FLOPS/byte (vs 232 FLOPS/byte ridgeline). No amount of instruction scheduling changes the physics. **Must reduce the bytes.**

---

## The Three Leverage Points That Stack

### Leverage 1: MXFP4 Expert Weights -- Native Hardware Support Exists

The nkilib already has a working MXFP4 MoE kernel implementation:

```
/opt/.../neuronxcc/nki/_pre_prod_kernels/mxfp4_mlp_tkg_proj.py
/opt/.../neuronxcc/nki/_pre_prod_kernels/mlp_tkg/expert_mlp_tkg_all_token_mxfp4_impl.py
```

With native `nisa.nc_matmul_mx()` ISA instruction.

**The math:**
- Current: 18.6 MB expert weights per layer (BF16) -> 103 us (DMA-bound)
- FP8: 9.3 MB -> our tests showed 120 us (WORSE, because of DMA call overhead from per-row scales)
- **MXFP4: 4.65 MB -> theoretical DMA floor ~6 us at 800 GB/s**

MXFP4 solves the DMA call overhead problem that killed FP8. MX format uses **block-wise shared scales** (1 scale per 32 elements), not per-row scales. The scale data is tiny (~2% overhead) and packed inline with weights. Fewer DMA calls, much less descriptor overhead.

At 4.65 MB per layer: even with descriptor overhead inflating DMA to 15-20 us, the compute becomes the bottleneck (SiLU + matmuls at ~15-20 us). Total MoE: **25-35 us** vs current 103 us.

**Savings: (103-30) x 48 layers x 640 tokens = 2.83 seconds.** From 7.17s -> ~4.3s = **2.35x speedup**.

**Accuracy**: MXFP4 with calibration has been shown to work for MoE inference (MxMoE, ICML 2025). The competition uses tolerance-based validation (`rtol=0.05, atol=1e-5`), NOT bit-exact matching. The `divergence_difference_tol=0.001` means the argmax token just needs to be within 0.001 of the reference top logit. MXFP4 expert weights with BF16 activations and BF16 accumulation should pass this.

### Leverage 2: Cross-Layer Expert Prefetching -- Hide DMA Behind Compute

Recent research (Fate, 2025; arxiv 2502.12224) shows that **using layer N's gate input to predict layer N+1's experts achieves 97% accuracy**. Gate inputs across adjacent MoE layers are highly correlated.

On Trainium2, DMA and TensorE run **concurrently**. While layer N's expert matmuls execute on TensorE (~15-20 us with MXFP4), the DMA engine can prefetch layer N+1's expert weights into a second SBUF buffer set. SBUF has 24 MB per core -- more than enough for double-buffering 4.65 MB of MXFP4 weights.

If prediction is 97% accurate: 97% of the time, DMA is completely hidden. 3% of the time, reload (one cache miss per ~33 layers on average). Effective MoE latency drops to the **compute floor**: ~15-20 us.

**This turns the bottleneck from memory-bandwidth to compute.** The problem flips from 0.5 FLOPS/byte (hopelessly memory-bound) to TensorE-limited, which can then be attacked with `perf_mode=double_row` for 2x TensorE throughput.

### Leverage 3: NKI_FLOP_Ratio Is a Direct Score Multiplier

The scoring formula uses `NKI_FLOP_Ratio = NKI_MACs / Total_MACs`. Every dot-product running through XLA instead of NKI **dilutes the score**.

Current non-NKI operations (XLA-compiled):
- **LM head**: [2048, 37984] per TP rank = 77.8M MACs per step -- significant fraction of total MACs
- **Embedding lookup**: small but non-zero
- **CTE attention**: if not in NKI, the entire context-encoding MACs are in the denominator but not numerator

Converting the LM head alone to NKI could shift NKI_FLOP_Ratio from ~0.80 to ~0.90+. At quadratic scoring: **12.5% score increase** for free.

---

## The Combined Play

| Optimization | MoE per layer | e2e estimate | Speedup | Score (NKI=0.90) |
|---|---|---|---|---|
| Current (BF16) | 103 us | 7165 ms | 1.37x | 1.69 |
| MXFP4 weights | ~30 us | 4355 ms | 2.33x | 4.88 |
| + Cross-layer prefetch | ~18 us | 3775 ms | 2.68x | 6.46 |
| + double_row TensorE | ~12 us | 3487 ms | 2.90x | 7.57 |
| + Framework overhead fix | ~12 us | 2680 ms | 3.78x | 12.85 |

Competitor at 1.9x, NKI=0.85: **Score = 3.07**

At 2.7x, NKI=0.90: **Score = 6.56 -- more than double their score.**

---

## Implementation Plan (Priority Order)

### Phase 1: MXFP4 MoE Kernel (highest leverage)

- Start from `expert_mlp_tkg_all_token_mxfp4_impl.py` in nkilib
- Quantize expert weights offline using `quantize_mx()` from the private API
- Integrate into fused MoE TKG kernel (add MXFP4 weight loading + `nc_matmul_mx()`)
- Validate accuracy against the competition's tolerance thresholds
- This alone could reach 2.3x

### Phase 2: Cross-Layer Expert Prefetching

- In the MoE kernel, after computing the router output for layer N, predict layer N+1's top-8 experts
- Issue DMA for predicted experts into secondary SBUF buffer set while computing current experts
- On cache miss (wrong prediction), fall back to standard load
- Implementation is kernel restructuring -- add double-buffering + prediction logic
- Reference: Fate paper (arxiv 2502.12224), 97% prediction accuracy

### Phase 3: Maximize NKI_FLOP_Ratio

- Write NKI LM head kernel (straightforward matmul + on-device sampling fusion)
- Convert CTE attention to NKI if not already (captures those MACs in the numerator)
- Every non-NKI dot product hurts the score

### Phase 4: Multi-Sequence-Length Optimization

- Competition tests 5 prompts at seq_len 64, 128, 256, 640
- Current kernels tuned for 640 -- verify performance at short lengths
- CTE optimization matters more at short lengths (higher fraction of total time)

---

## Scoring Details

### Accuracy Validation (from main.py lines 418-500)

- NOT bit-exact -- uses floating-point tolerance maps
- `tol_map = {None: (1e-5, 0.05), 1000: (1e-5, 0.03), 50: (1e-5, 0.03), 5: (1e-5, 0.03)}`
- `divergence_difference_tol = 0.001`
- Greedy: argmax of logits must match or be within 0.001 of the expected top token
- If accuracy = 0, **entire submission scores 0** (hard failure)

### NKI FLOP Ratio (from main.py lines 552-634)

- Parses HLO modules from context encoding + token generation
- NKI MACs: from `AwsNeuronCustomNativeKernel` custom-calls' `mac_count` backend config
- Total MACs: all NKI custom-calls + all dot operations
- Ratio = NKI_MACs / Total_MACs (0 to 1)

### Test Configurations

- Batch size: always 1
- 5 test prompts with varying lengths (64, 128, 256, 640 tokens)
- Uses P99 latency (tail latency matters)
- Aggregate score across all prompts

---

## Model Architecture Reference

| Parameter | Value |
|---|---|
| hidden_size (H) | 2048 |
| num_attention_heads (Hq) | 32 |
| num_key_value_heads (Hkv) | 4 (GQA, replicated across TP) |
| head_dim (d) | 128 |
| moe_intermediate_size (I) | 768 (192 per TP rank, padded to 256) |
| num_experts (E) | 128 |
| num_experts_per_tok (K) | 8 |
| num_hidden_layers (L) | 48 |
| vocab_size (V) | 151936 |

### Per Decode Step Memory Bandwidth

| Component | Per Layer | x48 Layers | Notes |
|---|---|---|---|
| Expert weights (K=8, BF16) | 18.6 MB | 893 MB | Dominant cost |
| Expert weights (K=8, MXFP4) | 4.65 MB | 223 MB | 4x reduction |
| KV cache reads (seq=640) | 1.3 MB | 62 MB | Small |
| Arithmetic intensity (BF16) | 0.5 FLOPS/byte | -- | Severely bandwidth-bound |
| Trn2 ridgeline | 232 FLOPS/byte | -- | 460x gap |

---

## Why FP8 Failed and MXFP4 Won't

FP8 kernel (v7) at 120 us vs BF16 (v28f) at 98 us -- FP8 was SLOWER because:

- FP8 uses per-row scales: 5 DMA calls per expert (weights + scales separately)
- 40 total DMA calls with high descriptor overhead (GpSimdE)
- FP8 bandwidth utilization: 242 GB/s (30% of peak) vs BF16: 510 GB/s (64% of peak)
- The DMA descriptor overhead is FIXED per call -- halving data size doesn't halve DMA time

MXFP4 avoids this because:
- Block-wise scales (1 per 32 elements) packed inline with weights
- Fewer, larger DMA transfers -- better amortization of descriptor overhead
- Native `nc_matmul_mx()` ISA -- no explicit dequantization step
- 4x data reduction (vs 2x for FP8) overwhelms any remaining overhead

---

## Research References

- **Fate** (2025): Cross-layer expert prediction, 97% accuracy. arxiv 2502.12224
- **MoBE** (2025): Shared basis expert decomposition, 24-30% parameter reduction. arxiv 2508.05257
- **MxMoE** (ICML 2025): Mixed-precision per-block MoE quantization, 3.4x speedup. arxiv 2505.05799
- **AdapMoE** (ICCAD 2024): Adaptive expert count per layer. arxiv 2408.10284
- **Read-ME** (NeurIPS 2024): Router-decoupled MoE, up to 6.1x latency improvement. arxiv 2410.19123
- **TD-MoE**: Tensor decomposition for cross-expert redundancy. openreview D9cnZNZfxX
- **Awesome MoE Inference**: github.com/MoE-Inf/awesome-moe-inference/

---

## Hardware Capabilities (Trainium2)

### Available but Unexploited

- **`nisa.nc_matmul_mx()`**: Native MXFP4/MXFP8 matmul (2-4x throughput vs BF16)
- **`sendrecv()` / `sendrecv_cce()`**: Inter-core communication with compute overlap (private API)
- **`perf_mode=double_row`**: 2x TensorE throughput (requires both operands in low precision)
- **`nc_matmul_sparse()`**: Sparse matmul (exists in private API, minimal documentation)
- **LNC=2 asymmetric work**: Two cores can do DIFFERENT work (one prefetch, one compute)
- **`@nki.compiler.skip_middle_end_transformations`**: Bypass compiler middle-end (used in nkilib reference)

### Key File Locations

- MXFP4 MoE kernel: `/opt/.../nki/_pre_prod_kernels/mxfp4_mlp_tkg_proj.py`
- MXFP4 expert impl: `/opt/.../nki/_pre_prod_kernels/mlp_tkg/expert_mlp_tkg_all_token_mxfp4_impl.py`
- MX quantization: `/opt/.../nki/_pre_prod_nkl/experimental/mxfp8/`
- Private API: `/opt/.../nki/_private/private_api.py`
- Collective kernels: `/opt/.../nki/_pre_prod_kernels/experimental/misc/collective_kernels.py`

---

## Constraints

- Cannot use in-kernel allreduce (documented in ncc_illc059_bug_report.md)
- Async mode not useful with on-device sampling
- Bucket sizes must be 128-aligned
- Expert/tensor dimensions intentionally hardcoded; pad inputs in forward pass, not kernel
- TP=4 shape assumptions must be verified before editing

---

## MXFP4 MoE Fused Kernel: Detailed Implementation Plan

### Overview

Convert `kernel_v28f.py` (BF16 expert weights, 103 μs) to use MXFP4 expert weights with MXFP8 activations and native `nc_matmul_mx()`. Based on analysis of nkilib reference implementations:

- `expert_mlp_tkg_all_token_mxfp4_impl.py` — full MXFP4 MoE kernel
- `mxfp4_mlp_tkg_proj.py` — gate/up and down projection MXFP4 kernels
- `mxfp8.py` — quantization utilities (`quantize_mx_x4_wrapper`, `interleave_load_wrapper`, `matmul_mx_x4_wrapper`)

### Part 1: Offline Weight Conversion (Python, outside kernel)

#### Current BF16 Weight Shapes (v28f)

```
gate_up_w:  [E=128, H=2048, 2*I=384]  bf16  → 18.6 MB per expert × 8 = 149 MB/layer
down_w:     [E=128, I=192,  H=2048]    bf16
```

#### Target MXFP4 Weight Shapes

MXFP4 uses x4 packing: 4 partition elements packed into 1 free element. The MX format groups scales as 1 uint8 scale per block of `_q_height=8` partition rows × `_q_width=4` free columns.

**Gate/Up weights:**
```
# Step 1: Reshape to expose 512-element tiles along H
# H=2048 → n_H512_tiles=4 tiles of 512 (128 partition × 4 free)
gate_up_w_bf16:  [E, 128, 2, 4, I=192]           bf16   (reshaped from [E, H, 2I])
                  ^P    ^gate/up ^H512  ^I_free

# Step 2: Quantize each [128, 4*I] slice to MXFP4
gate_up_w_mx4:   [E, 128, 2, 4, I=192]           float4_e2m1fn_x4
gate_up_w_scale: [E,  16, 2, 4, I=192]           uint8
                  ^P//8          ^same free dims
```

The x4 packing means each `float4_e2m1fn_x4` element holds 4 FP4 values from adjacent partition rows. So the physical storage is `[E, 32, 2, 4, I]` in x4 containers, but logical shape is `[E, 128, 2, 4, I]`.

**Down weights:**
```
# Current: [E, I=192, H=2048]
# Reshape: I → partition dim (128 max), H → free dim

# Tile 0 (I0=128): [E, 128, H=2048]
# Tile 1 (I1=64, zero-padded to 128): [E, 128, H=2048]

# With x4 packing (I partition → I//4 physical):
# BUT: I=192 < 512, so n_I512_tiles=1, with I_p = I//4 = 48

down_w_mx4:      [E, 128, 1, H=2048]             float4_e2m1fn_x4
down_w_scale:    [E,  16, 1, H=2048]             uint8
```

Wait — let's be more precise. The nkilib down projection expects:
```
down_weights:       mxfp4_x4[E, I_p, ceil(I/512), H]
down_weights_scale: uint8[E, I_p // _q_height, ceil(I/512), H]
```
where `I_p = I // 4 = 48` when `I=192 < 512`, and `ceil(I/512) = 1`.

So: `down_w_mx4: [E, 48, 1, H=2048]` in float4_e2m1fn_x4, `down_w_scale: [E, 6, 1, H=2048]` in uint8.

**But our kernel uses per-TP-shard H**: H_shard=1024 per program. So DMA loads `[*, *, H_shard]` slices.

#### Quantization Procedure

```python
from neuronxcc.nki._pre_prod_nkl.experimental.mxfp8.quantize_mx import quantize_mx

def convert_gate_up_weights(gate_up_w_bf16):
    """
    gate_up_w_bf16: [E=128, H=2048, 2*I=384] bf16
    Returns: (gate_up_mx4, gate_up_scale)
    """
    E, H, GU = gate_up_w_bf16.shape  # 128, 2048, 384
    I = GU // 2  # 192
    _pmax = 128
    n_H512_tiles = H // (_pmax * 4)  # 2048 // 512 = 4

    # Reshape: [E, H, 2I] → [E, 128, n_H512, 4, 2, I]
    w = gate_up_w_bf16.reshape(E, _pmax, n_H512_tiles, 4, 2, I)
    # Permute to: [E, 128, 2, n_H512, 4*I] for interleaved x4 layout
    w = w.permute(0, 1, 4, 2, 3, 5).reshape(E, _pmax, 2, n_H512_tiles, 4 * I)

    # Interleave along partition dim: [128, 4*I] → x4 packed
    # For each [128, F] slice, the x4 interleave:
    #   rows 0:32 from section 0, rows 32:64 from section 1, etc.
    #   packed into [128, F] float4_e2m1fn_x4 (physical [32, F])

    # quantize_mx handles the interleaving + quantization
    # Output: data [128, F//4] as x4, scale [16, F//4] as uint8
    gate_up_mx4 = torch.empty(E, _pmax, 2, n_H512_tiles, I, dtype=torch.uint8)
    gate_up_scale = torch.empty(E, _pmax // 8, 2, n_H512_tiles, I, dtype=torch.uint8)

    for e in range(E):
        for g in range(2):
            for h_tile in range(n_H512_tiles):
                data, scale = quantize_mx(w[e, :, g, h_tile, :])
                gate_up_mx4[e, :, g, h_tile, :] = data
                gate_up_scale[e, :, g, h_tile, :] = scale

    return gate_up_mx4, gate_up_scale
```

Similar procedure for down weights, but partition is along I and free is along H.

#### Data Size Comparison

| Weight | BF16 Size | MXFP4 Size | Scales Size | Total MXFP4 | Reduction |
|--------|-----------|------------|-------------|-------------|-----------|
| gate_up (per expert) | 2048 × 384 × 2B = 1.5 MB | 128 × 2 × 4 × 192 × 0.5B = 96 KB | 16 × 2 × 4 × 192 × 1B = 24 KB | 120 KB | **12.5x** |
| down (per expert) | 192 × 2048 × 2B = 768 KB | 48 × 1 × 2048 × 0.5B = 48 KB | 6 × 1 × 2048 × 1B = 12 KB | 60 KB | **12.8x** |
| **8 experts total** | **18.1 MB** | **1.44 MB** | **0.29 MB** | **1.73 MB** | **10.5x** |

Note: The x4 packing gives an effective 4:1 ratio on the partition dimension PLUS 4-bit gives 4:1 on element size = not quite 16x because scales add overhead. The actual DMA data transferred per layer drops from ~18.6 MB to ~1.73 MB.

### Part 2: Kernel Signature Changes

```python
@nki.compiler.skip_middle_end_transformations
@nki.jit(platform_target='trn3')
def qwen3_moe_fused_tkg_mxfp4(
    inp,              # [B, 1, H=2048]  bf16  — unchanged
    gamma,            # [1, H=2048]     bf16  — unchanged
    router_w,         # [H=2048, E=128] bf16  — unchanged (router stays BF16)
    gate_up_w,        # [E=128, 128, 2, 4, I=192] float4_e2m1fn_x4  ← NEW
    gate_up_w_scale,  # [E=128,  16, 2, 4, I=192] uint8              ← NEW
    down_w,           # [E=128,  48, 1, H=2048]   float4_e2m1fn_x4  ← NEW
    down_w_scale,     # [E=128,   6, 1, H=2048]   uint8              ← NEW
):
```

**New constants:**
```python
_Q_WIDTH = 4       # x4 interleave factor
_Q_HEIGHT = 8      # scale block height (8 partition rows per scale)
_N_H512 = _H // (_PMAX * _Q_WIDTH)  # = 4 tiles of 512 along H
_N_H512_SHARD = _N_H512 // _N_PRGS  # = 2 per LNC program
```

### Part 3: Stage-by-Stage Implementation

#### Stage 1: RMSNorm — UNCHANGED

The RMSNorm produces `rmsnorm_normed_bf16[128, H_free * T]` in SBUF. This is the activation input for gate/up matmul.

#### Stage 2: Router + TopK — UNCHANGED

Router uses BF16 `router_w`. No quantization needed — the router is a tiny fraction of compute.

#### Stage 3: Activation Quantization (NEW STAGE)

Before expert matmuls, quantize the RMSNorm output from BF16 to MXFP8_x4 format:

```python
# rmsnorm_normed_bf16: [128, H_free * T]  bf16 in SBUF
# For T=1: shape is [128, 16]

# Step 1: Reshape to expose x4 interleave structure
# For gate/up: partition=128 (H0), free varies by H512 tile
# We need [128, n_H512_shard, T_padded] in MXFP8_x4

# For T=1: the "free" dim per H512 tile is just T=1
# After x4 interleave: [128, 1] → [32, 4] in physical layout
# After quantize_mx: [32, 1] mxfp8_x4 + [4, 1] uint8 scale

# But we need the full H dimension interleaved.
# rmsnorm output is [128, 16] (128 partition, 16 = H_free tiles × T tokens)
# Per H512 tile (4 H_free tiles): [128, 4*T] = [128, 4]
# After interleave: [32, 16] → quantize → [32, 4] mxfp8_x4 + [32, 4] uint8

# Per LNC shard: n_H512_shard=2 tiles
inp_for_gate_up = nl.ndarray((_PMAX, _N_H512_SHARD, T), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
inp_scale_for_gate_up = nl.ndarray((_PMAX, _N_H512_SHARD, T), dtype=nl.uint8, buffer=nl.sbuf)

for h512_tile in nl.static_range(_N_H512_SHARD):
    # Extract [128, 4*T] slice from rmsnorm output
    h_start = (prg_id * _N_H512_SHARD + h512_tile) * _Q_WIDTH * T
    src_slice = rmsnorm_normed_bf16[0:_PMAX, nl.ds(h_start, _Q_WIDTH * T)]
    # src_slice is [128, 4] for T=1

    # Interleave: [128, 4*T] → [32, 16*T] physical in SBUF
    interleaved = nl.ndarray((_PMAX // 4, _Q_WIDTH * _Q_WIDTH * T),
                             dtype=rmsnorm_normed_bf16.dtype, buffer=nl.sbuf)
    interleave_load_wrapper(dst_sbuf=interleaved, src_tensor=src_slice)

    # Quantize: [32, 16*T] → [32, 4*T] mxfp8_x4 + [4, 4*T] uint8 scale
    # But for P=32 ≤ 32: simple scale shape (P//8, F//4) = (4, T)
    quantize_mx_x4_wrapper(
        dst_data=inp_for_gate_up[0:_PMAX, h512_tile, 0:T],
        dst_scale=inp_scale_for_gate_up[0:_PMAX, h512_tile, 0:T],
        src_sbuf=interleaved,
    )
```

**Critical detail for T=1 decode:**
- RMSNorm output per H512 tile: `[128, 4]` (128 partition × 4 free = 512 BF16 elements)
- After x4 interleave: `[32, 16]` (32 physical rows, 16 = 4 sections × 4 free)
- After quantize_mx: `[32, 4]` mxfp8_x4 + `[4, 4]` uint8 scale (P=32 ≤ 32, simple path)
- Logical shape: `[128, T=1]` in mxfp8_x4

**SBUF cost**: Negligible — the quantized activation is tiny (128 × 2 × 1 = 256 bytes for data + 256 bytes for scales per LNC shard).

#### Stage 4: Expert Weight Loading (MODIFIED)

##### Current v28f DMA pattern (BF16, per expert):
```
3 DMA calls per expert:
  gate_up_bufs[k]       ← gate_up_w[expert_id, :, :]       [128, 16, 384] bf16 = 1.5 MB
  down_full0_bufs[k]    ← down_w[expert_id, 0:128, H_shard] [128, 1024]   bf16 = 256 KB
  down_full1_bufs[k][0:64,:] ← down_w[expert_id, 128:192, H_shard] [64, 1024] bf16 = 128 KB
Total: 3 DMAs × 4 experts = 12 DMAs per wave, ~9.4 MB per wave
```

##### New MXFP4 DMA pattern (per expert):
```
6 DMA calls per expert:
  gate_up_bufs[k]       ← gate_up_w[expert_id, :, :, shard, :]     [128, 2, 2, 192] mx4_x4 = 48 KB
  gate_up_scale_bufs[k] ← gate_up_w_scale[expert_id, :, :, shard, :] [16, 2, 2, 192] uint8 = 12 KB
  down_bufs[k]          ← down_w[expert_id, :, :, H_shard]          [48, 1, 1024]   mx4_x4 = 24 KB
  down_scale_bufs[k]    ← down_w_scale[expert_id, :, :, H_shard]    [6, 1, 1024]    uint8 = 6 KB
Total: 4 DMAs × 4 experts = 16 DMAs per wave, BUT only ~0.72 MB per wave
```

Wait — we can likely pack gate_up weights and scales into fewer DMAs. The nkilib reference loads weights + scales separately because of shape differences, but the total data per expert is:
- gate_up: 48 KB data + 12 KB scales = 60 KB (vs 1.5 MB BF16 — **25x smaller**)
- down: 24 KB data + 6 KB scales = 30 KB (vs 384 KB BF16 — **12.8x smaller**)
- **Total per expert: 90 KB vs 1.88 MB — 20.9x reduction**

**The DMA call count increases (6 vs 3) but total bytes drop by 20x.** Since the FP8 failure was caused by descriptor overhead dominating at only 2x data reduction, MXFP4's 20x reduction should overwhelmingly compensate for extra descriptors.

At 800 GB/s HBM bandwidth: `90 KB / 800 GB/s = 0.11 μs` theoretical per expert. Even at 30% utilization (like FP8): `0.11 / 0.30 = 0.37 μs` per expert. **8 experts × 0.37 = 3 μs total DMA** — essentially free.

The bottleneck shifts entirely to compute (TensorE).

##### Indirect DMA with MXFP4

v28f uses `TensorView.ap()` with `scalar_offset` + `indirect_dim=0` for expert-index-based indirect DMA. This must work with the new MXFP4 layout:

```python
# gate_up_w shape: [E=128, 128, 2, n_H512_shard, I=192]
# Element 0 stride: 128 * 2 * n_H512_shard * I (in x4 elements)
# The TensorView pattern must account for the x4 element type

expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

nisa.dma_copy(
    dst=gate_up_bufs[k],  # [128, 2, n_H512_shard, I] mx4_x4 in SBUF
    src=gate_up_w.ap(
        pattern=<must compute strides for [128, 2, n_H512_shard, I] within [E, 128, 2, n_H512, I]>,
        offset=prg_id * _N_H512_SHARD * I,  # shard offset into H512 dim
        scalar_offset=expert_id,
        indirect_dim=0,
    ),
    dge_mode=0,  # software DGE (required for indirect)
)
```

**Pattern computation**: The `.ap()` pattern maps SBUF coordinates to HBM addresses. For `gate_up_w[E, 128, 2, 4, 192]`:
- Expert dim stride = `128 * 2 * 4 * 192` = 196,608 (x4 elements)
- The DMA fetches a `[128, 2, n_H512_shard=2, 192]` slice per expert

This needs careful TensorView pattern encoding. The safest approach: reshape the HBM tensor to 2D `[E, 128 * 2 * n_H512 * I]` and compute a flat pattern, similar to how v28f handles `gate_up_w[E, H, 2I]`.

##### SBUF Buffer Allocation (MODIFIED)

```python
# v28f BF16 buffers:
#   4 × gate_up_bufs [128, 16, 384] bf16 = 4 × 12288 × 2B = 96 KiB
#   4 × down_full0   [128, 1024]    bf16 = 4 × 131072 × 2B = 1024 KiB  ← HUGE
#   4 × down_full1   [128, 1024]    bf16 = 4 × 131072 × 2B = 1024 KiB
#   Total weight SBUF: ~2144 KiB

# MXFP4 buffers:
#   4 × gate_up_bufs       [128, 2, 2, 192] mx4_x4 = 4 × 768 × 0.5B = 1.5 KiB
#   4 × gate_up_scale_bufs [128, 2, 2, 192] uint8  = 4 × ~3 KiB      = 12 KiB
#   4 × down_bufs          [48, 1, 1024]    mx4_x4 = 4 × 49152 × 0.5B = 96 KiB
#   4 × down_scale_bufs    [6, 1, 1024]     uint8  = 4 × 6144 × 1B    = 24 KiB
#   Total weight SBUF: ~134 KiB (vs 2144 KiB — 16x reduction)
```

This massive SBUF savings opens up double-buffering or larger tile sizes.

Actually wait — the weight shapes above need to account for x4 packing correctly. In `float4_e2m1fn_x4`, each element is a 4-bit float packed 4-to-a-byte (x4). The physical storage:
- `gate_up_bufs[128, 2, 2, 192]` in float4_e2m1fn_x4: physical partition = 128/4 = 32 rows, free = 2×2×192 = 768. So 32 × 768 × 1B = 24 KB per buffer. 4 buffers = 96 KB.
- `down_bufs[128, 1, 1024]` in float4_e2m1fn_x4: physical partition = 128/4 = 32 rows, free = 1024. So 32 × 1024 × 1B = 32 KB per buffer. 4 buffers = 128 KB.

Revised SBUF budget: ~224 KB for weights + ~36 KB for scales = ~260 KB (vs ~2144 KB in BF16). Still a huge improvement.

#### Stage 5: Gate/Up Matmul (MODIFIED)

##### Current v28f Gate/Up pattern:
```python
for h1 in nl.affine_range(H_free):        # 16 iterations
    for i_tile in nl.static_range(I_tiles):  # 2 iterations (I0=128, I1=64)
        nisa.nc_matmul(
            dst=gate_up_psum[0:128, col],
            stationary=gate_up_bufs[k][0:128, h1, 0:128],   # [128, 128] bf16
            moving=rmsnorm_normed_bf16[0:128, h1*T : h1*T+T],  # [128, T] bf16
        )
```
Total: 16 × 2 × 2 = 64 `nc_matmul` calls per expert (gate + up).

##### New MXFP4 Gate/Up pattern:

The key change: `nc_matmul` → `nc_matmul_mx` with quantized operands.

```python
# Stationary: gate_up weight [128, I] in float4_e2m1fn_x4
# Moving: activation [128, T] in float8_e4m3fn_x4
# Both with uint8 scale tensors

for h512_tile in nl.affine_range(_N_H512_SHARD):  # 2 iterations (was 16 via H_free)
    # Each H512 tile covers 512 elements = 128P × 4F (after x4 packing)
    # The nc_matmul_mx handles the full 512-element contraction in one call per I tile

    for i_tile in nl.static_range(I_tiles):  # still 2 iterations
        # Stationary: weight [128, I_tile_size] in mx4_x4
        # Moving: activation [128, T] in mx8_x4

        # Scale indexing (P=128 > 32, need split-based indexing)
        ss_p, ss_f = nl.mgrid[0:_PMAX, 0:I_tile_size]
        ms_p, ms_f = nl.mgrid[0:_PMAX, 0:T]
        ss_p_0, _, ss_p_1 = ss_p.split(_PMAX // 32, 8, 4)  # split(4, 8, 4)
        ms_p_0, _, ms_p_1 = ms_p.split(_PMAX // 32, 8, 4)
        i_ss_p = ss_p_0 * 32 + ss_p_1
        i_ms_p = ms_p_0 * 32 + ms_p_1

        gate_up_psum[0:_PMAX, col] += nisa.nc_matmul_mx(
            stationary=gate_up_bufs[k][0:_PMAX, 0, h512_tile, i_tile_slice],  # mx4_x4
            moving=inp_for_gate_up[0:_PMAX, h512_tile, 0:T],                  # mx8_x4
            stationary_scale=gate_up_scale_bufs[k][i_ss_p, 0, h512_tile, i_tile_slice],
            moving_scale=inp_scale_for_gate_up[i_ms_p, h512_tile, ms_f],
        )
```

**Key differences:**
1. **Loop count drops from 16 to 2** on H dimension: each H512 tile handles 512 H elements (128P × 4 via x4 packing), so 2048/512 = 4 total, /2 for LNC shard = 2 iterations. v28f needed 16 iterations (2048/128 = 16) because BF16 only uses 128 elements per matmul contraction.
2. **Matmul instruction**: `nc_matmul` → `nc_matmul_mx` — native MX ISA, no explicit dequantization
3. **Scale arguments**: Two extra uint8 tensors with P>32 split-based partition indexing
4. **Accumulation**: `nc_matmul_mx` accumulates into PSUM in FP32 (same as before)

Total nc_matmul_mx calls per expert for gate/up: `2 H512 × 2 I_tiles × 2 (gate+up) = 8` (vs 64 nc_matmul in v28f). **8x fewer matmul instructions.**

#### Stage 6: SiLU + ElementMul — UNCHANGED

The SiLU and elementwise multiply operate on PSUM/SBUF data in FP32. No change needed.

#### Stage 7: Intermediate Quantization (NEW)

Before the down projection, we need to quantize the SiLU*Up result from BF16 to MXFP8_x4:

```python
# inter_bf16: [128, I_tiles=2] bf16 in SBUF
# = [128, 2] logically, [128, 192] fully expanded

# For down matmul: partition is along I (192), free is along H
# I=192 < 512, so n_I512_tiles=1
# Reshape to [128, 192] → interleave → quantize

# Actually for T=1: inter_bf16 is [128, 2] where each "2" represents
# an I_tile (128 and 64 elements respectively)
# The PSUM result for I0 tile is [128, 1] and I1 tile is [128, 1]
# After expanding: [192, 1] (192 I elements, 1 token)

# For nc_matmul_mx down: stationary=inter [I_p, T], moving=down_w [I_p, H_tile]
# I_p in x4 = 192/4 = 48 partition rows
# T in x4 = needs padding to at least 4 for x4 interleave

# Actually simpler: intermediate is [128, 2] in SBUF representing [I, T_tiles]
# We need to reshape this to [I_padded, T] with x4 packing for the matmul

# For T=1: [192, 1] → interleave [48, 4] → quantize → [48, 1] mx8_x4 + [6, 1] uint8
inter_qtz = nl.ndarray((_PMAX, T), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
inter_scale = nl.ndarray((_PMAX, T), dtype=nl.uint8, buffer=nl.sbuf)

# Reshape inter_bf16 from [128_H, I_tiles] to [I, T] = [192, 1]
# Then zero-pad I to 192 (already exact), interleave, quantize
inter_for_down = nl.ndarray((I, _Q_WIDTH * T), dtype=inter_bf16.dtype, buffer=nl.sbuf)
# ... fill from SiLU*Up result ...

interleave_load_wrapper(dst_sbuf=inter_interleaved, src_tensor=inter_for_down)
quantize_mx_x4_wrapper(dst_data=inter_qtz, dst_scale=inter_scale, src_sbuf=inter_interleaved)
```

**Note**: This is the trickiest part of the conversion. The intermediate result lives in SBUF in a transposed layout `[128_H, I_tiles]` from the gate/up PSUM drain. We need to reorganize it to `[I, T]` orientation for the down matmul. In v28f, this reorganization is implicit (the down matmul just reads different SBUF addresses). With MXFP4, we need explicit reshape + interleave + quantize.

For T=1 with I=192: the quantize path uses P=48 (I//4) which is > 32, so the P>32 scale indexing path is used. Actually, 48 > 32, so we use the split-based path: `s_p.split(48//32=1, 8, 4)` — this works but the split factors need to divide evenly. Since 48/32 = 1.5, this doesn't divide cleanly. We may need to zero-pad I to 256 (next multiple of 128) for x4 packing = 64 partition rows.

**Alternative**: Skip intermediate quantization. Keep intermediate in BF16 and use `nc_matmul` (not `_mx`) for the down projection with MXFP4 stationary weights only. The nkilib reference `down_proj_mxfp4` does quantize the intermediate, but if the compiler accepts mixed precision (MXFP4 stationary + BF16 moving), we can avoid this complexity. Need to test.

#### Stage 8: Down Matmul (MODIFIED)

##### Current v28f down pattern:
```python
for h1_out in nl.affine_range(H_free_shard):  # 8 iterations
    nisa.nc_matmul(
        dst=down_psum[0:128, d_base + h1_out],
        stationary=down_full0_bufs[k][0:128, h1_out*128 : (h1_out+1)*128],  # [128,128] bf16
        moving=inter_bf16[0:128, 0:1],  # [128, T=1] bf16
    )
    nisa.nc_matmul(  # I1 tile (64 valid + 64 zero)
        dst=down_psum[0:128, d_base + h1_out],
        stationary=down_full1_bufs[k][0:128, h1_out*128 : (h1_out+1)*128],
        moving=inter_bf16[0:128, 1:2],
    )
```
Total: 8 × 2 = 16 nc_matmul per expert.

##### New MXFP4 down pattern:

```python
# Stationary: down_w [I_p=48, H_tile] in mx4_x4
# Moving: inter [I_p=48, T] in mx8_x4
# n_I512_tiles = 1 (I=192 < 512)

for h1_out in nl.affine_range(H_free_shard):  # 8 iterations (H_shard/128)
    # Scale indexing for P=128 (weight is [128, 1, H_shard] but
    # matmul tile is [128, 128] per h1_out)

    down_psum[0:_PMAX, d_base + h1_out] += nisa.nc_matmul_mx(
        stationary=down_bufs[k][0:_PMAX, 0, h1_out*_PMAX : (h1_out+1)*_PMAX],  # mx4_x4
        moving=inter_qtz[0:_PMAX, 0:T],  # mx8_x4
        stationary_scale=down_scale_bufs[k][i_ss_p, 0, h1_out*_PMAX + ss_f],
        moving_scale=inter_scale[i_ms_p, ms_f],
    )
```

**Key change**: Two nc_matmul per h1_out (I0 + I1 tiles) collapse into ONE nc_matmul_mx. With MXFP4, the full I=192 is represented in a single [128, H_tile] weight tile (since 192 elements pack into 48 x4 rows < 128 partition max). No more two-tile I splitting.

Total nc_matmul_mx for down: 8 per expert (vs 16 nc_matmul in v28f). **2x fewer.**

#### Stage 9: Affinity Scaling + Accumulation — UNCHANGED

The down matmul output in PSUM is drained to SBUF, scaled by expert affinity, and accumulated. No change.

#### Stage 10: Output Store — UNCHANGED

### Part 4: Integration Changes (qwen_with_nki.py)

```python
# Current integration (line ~2219):
qwen3_moe_fused_tkg[2](hidden_states.data, layernorm_weight, router_w,
                        gate_up_proj.weight, down_proj.weight)

# New integration:
qwen3_moe_fused_tkg_mxfp4[2](hidden_states.data, layernorm_weight, router_w,
                              gate_up_proj.weight_mx4, gate_up_proj.scale,
                              down_proj.weight_mx4, down_proj.scale)
```

The weight conversion must happen at model load time (once), not per-step. Options:
1. **Offline conversion script**: Convert checkpoint, save as new format
2. **On-load conversion**: In `QwenForCausalLM.__init__`, quantize BF16 weights to MXFP4 and replace parameters
3. **Custom `QuantizedLinear` wrapper**: Store both formats, use MXFP4 in forward

Option 2 is simplest for the competition.

### Part 5: Expected Performance

#### Compute Estimate (nc_matmul_mx)

Per expert:
- Gate/up: 8 nc_matmul_mx calls (vs 64 nc_matmul)
- Down: 8 nc_matmul_mx calls (vs 16 nc_matmul)
- Total: 16 nc_matmul_mx per expert (vs 80 nc_matmul — **5x fewer instructions**)

`nc_matmul_mx` throughput on trn3 with MXFP4 stationary + MXFP8 moving: approximately 2-4x the FLOPs/cycle of BF16 nc_matmul (native MX ISA uses lower-precision tensor cores). With `perf_mode=double_row`: another 2x (requires both operands < BF16, which MXFP4+MXFP8 satisfies).

#### DMA Estimate

Per wave (4 experts): ~360 KB total data (vs ~9.4 MB BF16). At 30% HBM utilization (conservative): `360 KB / (800 × 0.30 GB/s) = 1.5 μs`. The DMA engine is essentially idle — **all bottleneck moves to TensorE compute**.

#### Total MoE Kernel Estimate

| Component | v28f (BF16) | MXFP4 (estimate) |
|-----------|-------------|-------------------|
| RMSNorm | ~3 μs | ~3 μs (unchanged) |
| Router + TopK | ~8 μs | ~8 μs (unchanged) |
| Activation quant | — | ~2 μs (new) |
| Expert DMA (8 experts) | ~50 μs | ~3 μs |
| Gate/Up matmul | ~25 μs | ~8 μs (fewer calls + faster MX ISA) |
| SiLU + mul | ~5 μs | ~5 μs (unchanged) |
| Inter quant | — | ~2 μs (new) |
| Down matmul | ~10 μs | ~5 μs (fewer calls + faster MX ISA) |
| Affinity + accum | ~2 μs | ~2 μs (unchanged) |
| **Total** | **~103 μs** | **~38 μs** |

Conservative estimate: **38 μs** (2.7x faster than v28f).
Optimistic with `perf_mode=double_row`: **~25 μs** (4.1x faster).

#### End-to-End Impact

At 38 μs MoE per layer:
```
MoE: 38 × 48 × 640 = 1,167 ms (was 3,178 ms)
Attn: 55 × 48 × 640 = 1,690 ms (unchanged)
AllReduce: 24 × 48 × 640 = 737 ms
Post-transformer: 550 × 640 = 352 ms
Framework: 1,260 × 640 = 806 ms
Total: ~4,752 ms → speedup = 9800 / 4752 = **2.06x**
```

At 25 μs MoE (with double_row):
```
MoE: 25 × 48 × 640 = 768 ms
Total: ~4,353 ms → speedup = 9800 / 4353 = **2.25x**
```

With cross-layer prefetching hiding remaining DMA: could push MoE below 20 μs → **2.5x+ speedup**.

### Part 6: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Accuracy failure (MXFP4 logit divergence) | Medium | Fatal (score=0) | Test with competition tolerance thresholds early; MXFP4 with BF16 accumulation shown to work for MoE (MxMoE, ICML 2025) |
| nc_matmul_mx not available on trn3 | Low | Fatal | Already confirmed in nkilib with `platform_target='trn3'` |
| Indirect DMA incompatible with x4 element types | Medium | Must restructure DMA | Fall back to direct DMA with manual expert-index lookup; small perf cost |
| Scale indexing bugs (P>32 split path) | High | Correctness | Use nkilib's `matmul_mx_x4_wrapper` directly initially; optimize later |
| Inter-quantization layout complexity | Medium | Delay | Start with BF16 intermediate (skip inter-quant); quantize down-proj moving operand later |
| Compiler scheduling regression with new instructions | Medium | 10-20% perf loss | Apply `@nki.compiler.skip_middle_end_transformations` |

### Part 7: Implementation Sequence

1. **Weight conversion script** — quantize BF16 weights to MXFP4 offline, validate shapes
2. **Minimal kernel v29**: replace gate/up nc_matmul with nc_matmul_mx, keep down in BF16, no inter-quantization. Validates DMA + gate/up matmul path.
3. **Kernel v30**: add intermediate quantization + down nc_matmul_mx. Full MXFP4 pipeline.
4. **Accuracy validation**: run against competition tolerance on all 5 test prompts
5. **Profile + tune**: identify new bottleneck (expect TensorE-bound), try `perf_mode=double_row`
6. **Integration**: wire into `qwen_with_nki.py`, run e2e benchmark
