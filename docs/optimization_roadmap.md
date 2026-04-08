# Qwen3-30B-A3B NKI Optimization Roadmap

For the MLSys 2026 competition. Updated 2026-04-03.

**Current state**: 7232ms e2e latency (1.35x vs 9800ms baseline).  
**Competition target**: 1.75–1.8x = 5444–5600ms. Need ~1635–1791ms more reduction.

All three base kernels are implemented:
- Attention TKG (v13_BC): 59.47 μs — fused QKV + per-head RMSNorm + RoPE + flash decode + output proj
- MoE TKG (v27d): 103.48 μs — fused RMSNorm + router + top-8 + expert MLPs
- Router CTE: fused linear + softmax + top-k for context encoding

---

## Bottleneck Analysis

At 7232ms with 640 decode tokens:
- CTE (context encoding): ~184ms (2.6%)
- TKG (token generation): ~6848ms (94.7%)
  - MoE TKG: 103.48 μs × 640 = **66.2ms per layer × 40 layers ≈ dominant cost**
  - Attn TKG: 59.47 μs × 640 = 38.1ms per layer

To hit 1.75x (5600ms), need ~23% reduction from TKG. MoE is the primary target.

---

## Priority 1: MoE TKG Kernel Micro-Optimizations

Current v27d: 103.48 μs. Goal: push toward 85–90 μs (15–18% reduction).  
Profiling shows: DMA 63% active, TensorE 41% active — dual bottleneck.

### 1a. Remove redundant memsets and intermediate copies

- `nisa.memset(aff_bcast)` before `tensor_copy + nc_stream_shuffle` is a no-op write — the subsequent ops overwrite it fully. Eliminate.
- Look for any `sum_reduced_sb` intermediates that are written then immediately read once. Fold those into the consumer op.
- Each removed instruction reduces SBUF pressure, which improves compiler scheduling quality.

**Risk**: Low. Test one removal at a time; revert immediately on regression.

### 1b. Replace separate activation+reduce patterns with `nisa.activation_reduce()`

The fused ISA op computes `activation(x) → reduce_add` in a single pass over the data. Two candidate sites:
- RMSNorm: `square(x) → reduce_add` → replace with `activation_reduce(x, op=SQUARE, reduce=SUM)`
- Softmax numerator: `exp(x) → reduce_add` → replace with `activation_reduce(x, op=EXP, reduce=SUM)`

Each eliminates one SBUF read + one TensorE instruction. Estimated 1–3 μs each if successful.

### 1c. DMA double-buffering for expert weight loads

Current kernel loads expert gate_up weights for 8 experts sequentially. Pattern to try:
```python
# While computing expert i, DMA-prefetch expert i+1 weights
# Requires two SBUF buffer sets for weights
```
This is only beneficial if TensorE is the gating constraint during those windows. Profile before and after to verify overlap actually occurs. The compiler may already do this — check profiler DMA idle% vs TensorE idle%.

### 1d. Instruction ordering sensitivity

The v27d scheduler is fragile. Any change that reorders instructions can cause +5–15% regression. When testing 1a–1c:
- Change exactly one thing per test.
- If latency worsens by >2%, revert and try a different variant.
- Do not "clean up" unrelated code around a change.

---

## Priority 2: Attention TKG Kernel Micro-Optimizations

Current v13_BC: 59.47 μs. Goal: push toward 50–54 μs (9–16% reduction).  
Profiling shows: DMA 54% active, TensorE 46% — DMA-bound.

### 2a. K/V cache load overlap

For each decode step, the kernel loads the full KV cache for the current sequence. At seq=640, this is:
- K cache: [640, 8, 128] × 40 layers × BF16 = ~83MB total across layers
- DMA fetches are sequential within the kernel

Investigate whether the compiler's `multi_buffer(2)` directive (from `neuronxcc.nki.compiler.backends.neuron.LexicalScopeDirective`) can be applied to the K/V load loop to pipeline loads with QK compute. This is the same double-buffering technique used in the nkilib attention reference.

### 2b. Causal mask precomputation

The attention mask [1, num_heads, 1, 640] is recomputed or re-fetched every token. For decode:
- The mask changes by exactly 1 position each step (one more position becomes valid)
- The incremental update `mask[:, :, 0, step_idx] = 1` is O(1)
- If the current kernel recomputes the full mask, replace with in-place update

Check whether `gen_cache_mask_for_attention_tkg_kernel` from neuronx_distributed_inference is being used, or if a fresh mask is generated each call.

### 2c. Verify hardware DGE coverage

DGE (Direct Generate Enable) was applied in v13_BC. Confirm it's active on all DMA ops, not just the ones that had it enabled explicitly. Use profiler `dma_descriptor_overhead%` to check.

---

## Priority 3: Fused Layer-Level Pipelining

This is the highest-ceiling optimization and the hardest to implement correctly.

### Concept

A single decoder layer currently runs:
1. Attn TKG kernel (59 μs)
2. Residual add
3. MoE TKG kernel (103 μs)
4. Residual add

The two kernels and residuals are separate XLA ops, which means: kernel 1 finishes → result written to HBM → kernel 2 reads from HBM → compute. The HBM round-trip between kernels costs ~5–10 μs per layer.

If attention output can be kept in SBUF and passed directly to MoE input, eliminating the HBM write + read, saves ~5–10 μs × 40 layers = 200–400ms e2e.

### Feasibility

The `output_in_sbuf=True` parameter in `moe_tkg()` suggests the infrastructure for this exists. The challenge:
- Attention kernel must also expose `output_in_sbuf=True`
- Both kernels must be in the same `@nki.jit` call or the SBUF content is lost between calls
- SBUF is 128KB per core; need to verify both intermediate tensors fit simultaneously

### Path

1. Check if attention v13_BC supports `output_in_sbuf` — look at how it allocates its output tensor
2. If not, add an `output_in_sbuf` flag that conditionally allocates output as `nl.sbuf` vs HBM
3. Write a wrapper kernel that calls attention then MoE in the same jit scope, passing SBUF pointer
4. Profile: does it actually reduce HBM traffic? Verify with `dma_active%` going down.

**Risk**: High. SBUF size constraints are a real limit; compiler may reject or degrade this. Attempt only after Priority 1 and 2 wins are exhausted.

---

## Priority 4: Compiler and Runtime Flags

Low effort, potentially meaningful.

### 4a. `@nki.compiler.skip_middle_end_transformations`

Found in neuronx_distributed_inference MoE kernel reference:
```python
@nki.compiler.skip_middle_end_transformations
@nki.jit(platform_target='trn2', ...)
def moe_token_gen_kernel(...):
```
This bypasses certain compiler middle-end passes that may pessimize hand-optimized NKI code. Try adding to both kernels and measure. If the compiler is already at a local optimum, this is a no-op. If the middle-end is inserting suboptimal ops, could be 2–5% win.

### 4b. `NEURON_RT_EXEC_TIMEOUT` and async execution

Look at `causal_lm_async_execution()` in `neuronx_distributed_inference/modules/async_execution.py`. If the model is currently running synchronously (CPU waits for each token's compute before scheduling the next), enabling async decode would pipeline CPU scheduling overhead with device execution. This is a pure Python-level change, not a kernel change.

Measure: what fraction of e2e latency is device execution vs CPU overhead? Use `torch.profiler` or `neuron_profile` to get host-side timeline.

---

## Priority 5: MoE Weight Quantization (FP8)

nkilib's `moe_tkg` and `tokengen_moe_megakernel_forward_all_experts` both accept optional weight scale tensors for FP8 quantization:
```python
W_expert_gate_up_scale: Optional[nl.ndarray]  # Shape [E_L, 2, I]
W_expert_down_scale: Optional[nl.ndarray]      # Shape [E_L, H]
```

FP8 weights are half the size of BF16, directly halving DMA bandwidth for weight loads. Since MoE TKG is DMA-bound (63% DMA active), this could give 20–30% latency reduction on MoE — the single largest potential win on the list.

**Constraints**:
- Logits must match BF16 within tolerance. FP8 quantization introduces error. Must validate.
- Quantization calibration: need to run forward passes to compute per-tensor or per-row scales.
- Use `neuron_config.quantized_mlp_kernel_enabled = True` as the config knob.
- Test with `expert_affinities_scaling_mode='post_scale'` to compensate for quantization noise.

**Risk**: Medium-high. The logit tolerance requirement is a hard gate. If FP8 expert weights degrade output quality beyond tolerance, this can't be used.

**Recommended path**: Implement, run logit comparison against CPU BF16 baseline on 100+ tokens across diverse prompts. If mean absolute error is within 1e-2 in BF16 scale, it's likely acceptable.

---

## Priority 6: MxFP4 Quantization (Aggressive)

nkilib supports MxFP4 (microscaling FP4) for expert weights:
- Weight shape changes from `[E_L, H, 2, I]` to `[E_L, 128, 2, ceil(H/512), I]` 
- 8× size reduction vs BF16, 4× vs FP8

Potential: 35–50% MoE kernel latency reduction. But:
- Larger quantization error — logit tolerance is very likely violated
- More complex to implement (shape changes, scale tensor management)
- Less supported/tested path

Treat as a stretch goal. Attempt only if FP8 is validated and more headroom is needed.

---

## What Not to Try Again

Based on the optimization log:
- **LNC=2 Wo sharding**: +14.6% regression
- **dma_transpose for K-cache**: +13.5% regression  
- **Stride-0 DMA broadcast**: +29% regression
- **4-wave MoE with 2 experts (v27e)**: +3–13% regression depending on variant
- **Speculative decoding**: not applicable for batch=1 with no draft model available

---

## Realistic Projection

| Phase | Approach | Expected Gain | Risk | Target Latency |
|-------|----------|--------------|------|---------------|
| Current | — | — | — | 7232ms (1.35x) |
| P1 | MoE micro-opts (memset, activation_reduce) | 5–10% MoE | Low | ~6900ms (1.42x) |
| P2 | Attn micro-opts (mask precompute, double-buffer) | 5–10% Attn | Medium | ~6600ms (1.48x) |
| P3 | Compiler flags + async execution | 2–5% e2e | Low | ~6400ms (1.53x) |
| P4 | FP8 MoE weights | 15–25% MoE | Medium-High | ~5700ms (1.72x) |
| P5 | SBUF-fused attn+MoE layer | 3–6% e2e | High | ~5400ms (1.81x) |

The path to 1.75–1.8x likely requires FP8 quantization (P4) succeeding. Everything before it closes the gap but doesn't reach the target alone.

---

## Implementation Protocol

1. Profile before and after every change: `device_time_us`, `tensor_engine%`, `dma_active%`
2. One change at a time — the compiler scheduler is extremely sensitive to instruction ordering
3. Revert immediately if regression > 2% — don't try to fix regressions with more changes
4. Validate BF16 logit tolerance after any change that affects computation (not just layout)
5. Run e2e benchmark (not just kernel benchmark) to confirm wall-clock improvement
