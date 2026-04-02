# Decode Performance Optimization Roadmap for Qwen3-30B-A3B

## Current State Analysis

**Model**: Qwen3-30B-A3B (48 layers, H=2048, 128 experts, top-8, moe_intermediate=768)
**Hardware**: TRN2, TP=4, LNC=2, batch=1, seq_len=640
**Current decode**: **10.25ms/token** (TKG p50), 97.7 tok/s TKG throughput, 63.1 tok/s E2E

The NKI kernels (v10e attention, v19b MoE, Plan A router) are already deeply optimized at the individual kernel level. Most low-hanging microarchitectural fruit is exhausted (DMA coalescing, SBUF hoisting, activation fusion, transpose elimination). The remaining gains must come from **architectural/framework-level** changes, not kernel micro-optimization.

### Where the 10.25ms Goes (per decode step)

Per step, 48 layers each execute:
- Attention TKG (fused QKV + RMSNorm + RoPE + flash-decode + O-proj)
- MoE TKG (fused RMSNorm + Router + TopK + 8 expert MLPs)
- Residual adds + TP all-reduce (2x per layer: one after attention, one after MoE)
- LM head + sampling (final step)

The **TP all-reduce** after MoE happens 48 times per step and is a significant overhead that can't be eliminated with NKI kernels alone. The Python/framework overhead between steps (inter-step gap) also adds up.

---

## Tier 1: High Impact, Realistic (Target: 1.3-2.5x improvement)

### 1. Speculative Decoding (~2-3x effective throughput)

**Single highest-leverage optimization.** The library has full support.

**How it works**: A small draft model generates K candidate tokens cheaply, then the target model verifies all K tokens in a single forward pass. If acceptance rate is ~70%, you get ~3 tokens per target step instead of 1.

**Options available in the library**:

- **Fused speculation** (`--enable-fused-speculation --speculation-length 5`): Uses prompt-lookup (n-gram matching) -- no separate draft model needed. Simpler but lower acceptance rate (~40-60% on typical text). Still gives ~1.5-2x.

- **EAGLE speculation** (`--enable-eagle-speculation`): Uses a lightweight draft head attached to the target model's hidden states. The library has full EAGLE support (`neuronx_distributed_inference/modules/eagle/`) with rolling buffers and token trees. Higher acceptance rate (~70-80%). **Most promising path.** An EAGLE draft head for Qwen3-MoE would be small (a few transformer layers on top of the hidden states) and run very fast compared to the full MoE forward pass.

- **Unfused speculation** with a small dense draft model (e.g., Qwen3-0.6B): The draft model generates candidates, target verifies. Works but requires compiling/loading a second model.

**Realistic impact**:
- Fused speculation: ~1.5x (effective ~6.8ms/token)
- EAGLE: ~2-2.5x (effective ~4-5ms/token)

**Configuration**:
```
--enable-fused-speculation --speculation-length 5
```
or
```
--enable-eagle-speculation --speculation-length 5
```

**Constraint**: Incompatible with batch bucketing (but batch=1 with fixed seq_len, so not an issue).

### 2. On-Device Sampling (~0.5-1.5ms savings per step)

The inter-step gap instrumentation in `main_new.py:425-446` measures CPU round-trip for sampling + Python overhead between steps. On-device sampling eliminates the device->CPU->device roundtrip for token selection.

**Already supported**: `qwen_with_nki.py:2186` checks for `on_device_sampling_config`, and `lm_head` uses `gather_output=False` when enabled.

**Enable with**:
```
--on-device-sampling --top-k 1
```

This moves argmax/sampling entirely onto the NeuronCore. With `top_k=1` (greedy), the on-device argmax kernel is very efficient.

**Realistic impact**: ~5-15% reduction in per-token latency (from ~10.25ms to ~9.0-9.7ms) by eliminating the inter-step gap overhead.

### 3. Upgrade to Latest NKI Kernels (v21b MoE, v13a attention)

The integration currently uses **v10e attention** and **v19b MoE**. Available upgrades:
- **v21b MoE**: Fused SiLU into PSUM flush, eliminated memsets (-8 ops per 8 experts per layer)
- **v13a attention TKG**: LNC O-projection sharding -- each core handles H_half=1024 Wo columns, halving Wo DMA and PSUM per core

**Realistic impact**: ~5-10% cumulative across 48 layers. Small per-layer savings compound significantly.

---

## Tier 2: Moderate Impact (Target: additional 1.1-1.3x)

### 4. Expert Parallelism with Hybrid Sharding

Currently MoE uses `moe_tp_degree=1, moe_ep_degree=1`, meaning each TP rank holds all 128 experts (with weights sharded across TP for the intermediate dimension).

The library's **MoE v2** module supports `HybridShardingConfig` for **different EP/TP degrees for CTE vs TKG**:

```python
hybrid_sharding_config=HybridShardingConfig(
    moe_cte_tp_degree=4, moe_cte_ep_degree=1,   # prefill: full TP
    moe_tkg_tp_degree=1, moe_tkg_ep_degree=4,   # decode: EP across 4 devices
)
```

With EP=4 for TKG, each device only holds 32 experts. Since top-8 picks from 128, each device processes ~2 active experts on average instead of 8. This reduces per-device MoE compute by ~4x at the cost of an all-to-all communication for expert dispatch.

**The tradeoff**: EP adds collective communication overhead, but for decode (B=1, single token), the expert MLP compute is small and the communication is also small (one token's worth of activations). Whether this wins depends on the all-to-all latency on TRN2.

**Important caveat**: The custom NKI MoE kernel would need to be adapted for EP (it currently assumes all 128 experts are local). May need to fall back to the library's blockwise matmul kernel for the EP path, or write an EP-aware NKI kernel.

**Realistic impact**: Uncertain -- potentially 1.2-1.5x for MoE portion if all-to-all is fast, but requires significant integration work.

### 5. KV Cache Tiling + Cascaded Attention

The library supports `kv_cache_tiling=True` which enables **cascaded reduction** optimizations for the attention computation. The current flash-decode iterates over `NUM_S_TILES` K-cache tiles sequentially.

With cascaded attention (`attn_block_tkg_nki_kernel_cascaded_attention`), the KV cache access pattern is reorganized for better hardware utilization.

**Enable with**:
```
--attn-block-tkg-nki-kernel-enabled --attn-block-tkg-nki-kernel-cascaded-attention
```

**Caveat**: This uses the library's built-in TKG attention kernel, not the custom v10e. Need to evaluate whether the library's optimized kernel is faster, or incorporate cascaded attention into the NKI kernel.

**Realistic impact**: ~5-10% attention speedup if sequence lengths grow beyond 640.

### 6. Compiler Flag Tuning

Current TKG compiler args (`qwen_with_nki.py:2267-2277`) are already good. Consider:

- **`--cc-pipeline-tiling-factor=1`**: Already set. For pure decode (single token), tiling factor 1 is correct.
- **`layer_boundary_markers=True`**: `ModuleMarkerStartWrapper`/`ModuleMarkerEndWrapper` are used. Ensure `--layer-boundary-markers` is passed at runtime.
- **`--enable-spill-reload-dge`**: If any kernels are spilling to HBM, DGE-based spill/reload can speed up the spill path. Worth testing.

---

## Tier 3: Marginal / Experimental

### 7. Attention Data Parallelism (attention_dp_degree)

With TP=4, could set `attention_dp_degree=2` to split batch across attention compute. But with batch=1, this doesn't help -- ADP requires `batch_size % attention_dp_degree == 0`.

**Only useful if batch size is increased.**

### 8. Flash Decoding (Library Feature)

`--flash-decoding-enabled` optimizes KV cache updates with position-specific masking. At seq_len=640 this has limited impact, but if generating up to the full 640 tokens, the KV cache grows and flash decoding helps with the later steps.

**Realistic impact**: ~2-5% for later decode steps.

### 9. Reduce TP All-Reduce Overhead

Each decoder layer does **2 all-reduces** (one after attention O-proj, one after MoE). With 48 layers, that's 96 all-reduces per step. On TP=4 TRN2, each all-reduce over NeuronLink takes ~2-5us, totaling ~200-500us per step (~2-5% of 10.25ms).

Payloads are already tiny ([1, 1, 2048] = 4KB). Already near-optimal. No action needed.

---

## Realistic Performance Projection

| Optimization | Estimated Savings | New TKG/token | Difficulty |
|---|---|---|---|
| **Baseline** | -- | 10.25ms | -- |
| On-device sampling | 0.5-1.0ms | ~9.5ms | Easy (flag flip) |
| Kernel upgrade (v21b + v13a) | 0.5-1.0ms | ~9.0ms | Medium (integration) |
| Fused speculation (5 tokens) | ~1.5x effective | ~6.0ms effective | Medium |
| EAGLE speculation | ~2-2.5x effective | ~4-5ms effective | Hard (need draft head) |
| EP hybrid sharding | uncertain | -- | Hard |

## Recommended Priority Order

1. **Enable on-device sampling** -- Immediate, minimal code change, ~5-15% gain
2. **Upgrade to v21b MoE + v13a attention TKG** -- Kernels already exist, just need integration
3. **Enable fused speculation** -- Built into the library, test with `--enable-fused-speculation --speculation-length 5`. This alone could push from 1.05x to ~1.5x
4. **Investigate EAGLE speculation** -- Highest ceiling (~2.5x) but requires finding/training a draft head for Qwen3-MoE

## What Won't Help

- **Further NKI kernel micro-optimization**: At 1-3% MFU, kernels are memory-bound. Making compute faster doesn't help when waiting on DMA. Remaining kernel headroom is <5%.
- **Quantization**: Off the table per constraint.
- **Larger batch sizes**: Only helps throughput, not per-token latency (competition scores latency).
- **Context parallelism**: Only relevant for long prefills, not decode.

## The Path to 1.5-2x

The realistic path to significant improvement: **on-device sampling + kernel upgrades + speculative decoding**. Speculation is the game-changer -- it's the only technique that fundamentally changes the tokens-per-forward-pass ratio rather than shaving microseconds off each step.
