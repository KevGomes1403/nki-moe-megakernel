# FP8 Expert Quantization: Feasibility Report

## Executive Summary

FP8 quantization of expert weights is **feasible** but non-trivial. The hard parts are different for the two code paths. The TKG path requires in-kernel dequantization with careful SBUF budgeting and new DMA patterns. The CTE path is simpler: dequantize on host at load time (already partially implemented). A full integration has ~4 distinct work items of varying complexity.

---

## Current State

There is already a `qwen_baseline_quant.py` with 128-block FP8 quantization logic and scale computation. The state dict keys, block shapes, and per-expert scale tensors are defined. **No runtime dequantization kernel exists yet** — the existing code only quantizes for storage.

**Quantized weight shapes (from existing baseline):**

| Weight | FP8 Shape | Scale Shape |
|---|---|---|
| `gate_up_proj` | `[E=128, H=2048, 2*I=384]` | `[E=128, 2*I=384, H/128=16]` |
| `down_proj` | `[E=128, I=192, H=2048]` | `[E=128, H=2048, I/128≈2]` |

---

## Path 1: CTE Path

**Mechanism:** Standard `self.mlp` dispatch — weights are plain Python tensors before kernel launch. Dequantization can happen entirely on the host before the forward pass.

**Approach:** At model load time, run `(fp8_weight * scale).to(bf16)` for each expert and store the result. The CTE path then uses those bf16 weights as normal — zero changes to any kernel code.

**Complexity: Low.** The dequantization math is already in `qwen_baseline_quant.py` lines 253–268. Just wire it into model loading.

**Caveat:** This negates memory savings since you're storing bf16 at runtime. If HBM compression is the goal, you'd need a dedicated dequant NKI kernel that fuses with the CTE blockwise MLP call. That is harder — the library blockwise kernel doesn't expose an fp8 interface.

---

## Path 2: TKG Fused Kernel

This is where the real complexity lies. The TKG kernel (`kernel_v19b`) performs:
1. Per-expert indirect DMA loads of weights into SBUF
2. `nc_matmul` (gate, up, down) with bf16 stationary tiles
3. All fused in a single NKI kernel

To support FP8 here, you must load both the quantized weights AND their per-block scales, then dequantize in SBUF before the matmul.

**Per-expert matmul tiles and their scale requirements:**

| Matmul | Weight Tile in SBUF | Scale Needed |
|---|---|---|
| Gate (tile-0, K=128) | `[128, 128]` fp8 | 1 scalar per block (already 128-aligned) |
| Gate (tile-1, K=128) | `[128, 128]` fp8 (padded) | 1 scalar per block |
| Up (tile-0, K=128) | `[128, 128]` fp8 | 1 scalar per block |
| Up (tile-1, K=128) | `[128, 128]` fp8 (padded) | 1 scalar per block |
| Down (tile-0, per H-subtile) | `[128, 64]` fp8 | 1 scalar per 128-col block |
| Down (tile-1, per H-subtile) | `[128, 64]` fp8 (padded) | 1 scalar per 128-col block |

Note: Gate and Up tiles are each exactly 128 columns wide — they align perfectly to the 128x128 block scheme. Each tile's dequant scale is a single scalar per PMAX row.

**SBUF impact:**
- FP8 weight tiles are **half the size** of bf16 (positive)
- Scale tensors add ~2–40 KB depending on loading strategy
- Net SBUF change is likely slightly favorable, but needs verification against the ~174 KB peak budget

---

## Key Technical Challenges

### 1. Indirect DMA for Scales

Current DMA:
```python
nki.dma_copy(dst=gate_up_sbuf, src=gate_up_w, scalar_offset=expert_id)
```
Scales have a different tensor layout (`[E, 2*I, H/128]`) — you need a second DMA per weight per expert for scales. Indirect DMA with `scalar_offset=expert_id` should work the same way but must be verified for the scale tensor stride.

### 2. Dequantization Before nc_matmul

NKI's `nc_matmul` accepts bf16 stationary tensors. FP8 cannot be passed directly. After loading the fp8 tile + its scale, you must:
```python
weight_bf16 = nki.language.multiply(fp8_tile.cast(bf16), scale_tile)
```
This requires a free SBUF region to write the dequantized tile — effectively doubling the SBUF footprint for the weight at that moment (fp8 in + bf16 out). With the current 174 KB peak, this may require staggering loads more carefully or reducing the expert pipelining depth.

### 3. Down Weight Scale Alignment

Down weight scales are `[E=128, H=2048, I/128≈2]`. The matmul loads `[128, 64]` H-subtiles. Since `I=192 = 128 + 64`, the two I-tiles map to 2 scale values per H-subtile — this is clean. No scatter-gather needed.

### 4. TP Sharding + Scales

Each TP rank only loads the H-shard columns it owns (`H_shard = 512` per rank). Scales must be pre-sliced the same way at model load time (slice `gate_up_scales` along the H-blocks dimension) so each rank only holds `[E, 2*I, H_shard/128=4]` scale entries. This is a model sharding step, not a kernel change.

---

## Implementation Plan

### Step 1: Quantize + Shard at Model Load (host, Python)
- Run block-FP8 quantization on each expert's `gate_up_proj` and `down_proj`
- Shard both the fp8 weights and the scales across TP ranks along the H dimension
- Store as new buffers alongside existing weights (or replace them)
- **Files:** `qwen_with_nki.py` (model loading section), `qwen_baseline_quant.py`

### Step 2: CTE Path Dequantization (host, simple)
- At load time: `dequant_weight = (fp8_w * scale).to(bf16)` per expert
- Replace the CTE path's expert weights with dequanted bf16 tensors
- Zero kernel changes
- **Files:** `qwen_with_nki.py` (CTE model weight init)

### Step 3: TKG Kernel — Scale Loading
- Add new input tensors `gate_up_scales` and `down_scales` to the kernel signature
- Add DMA loads for scale tiles alongside each weight DMA (using same `scalar_offset=expert_id` pattern)
- **Files:** New kernel version (e.g. `kernel_v20_fp8.py`)

### Step 4: TKG Kernel — In-SBUF Dequantization
- After each weight tile DMA completes, broadcast the scale and multiply: `fp8_tile.cast(bf16) * scale`
- Store result in a separate SBUF region, use that region as the `nc_matmul` stationary input
- Audit SBUF budget — may need to reduce in-flight expert pipeline depth if memory is tight
- **Files:** New kernel version

### Step 5: Integration + Correctness Testing
- Compare fp8-dequant output vs bf16 baseline on a single forward pass
- Check accuracy degradation (expected: < 1% perplexity increase for block-128 fp8)
- Benchmark latency: fp8 reduces HBM bandwidth per weight, but adds dequant ops — net effect TBD

---

## Risk Summary

| Risk | Severity | Mitigation |
|---|---|---|
| SBUF overflow during dequant (fp8 in + bf16 out coexist) | Medium | Stagger expert loading, reduce pipeline depth |
| Scale DMA stride alignment for indirect addressing | Low–Medium | Verify contiguous layout before shard; pre-transpose if needed |
| CTE library kernel incompatibility with fp8 input | Low | Use host dequant; no kernel changes needed for CTE |
| TP shard boundary misalignment with 128-col blocks | Low | `H=2048`, `H/TP=512`, `512/128=4` — perfectly aligned |
| `nc_matmul` not accepting fp8 directly | Confirmed | Must cast to bf16 in SBUF before matmul |

---

## Bottom Line

The quantization math and weight sharding are already partially built. The CTE path can be done quickly (host dequant). The TKG kernel path is the real work — roughly a new kernel version with modified DMA patterns and an in-SBUF dequant step per matmul, plus SBUF budget verification. The tensor shapes align well with 128-wide blocks and there are no fundamental blockers.
