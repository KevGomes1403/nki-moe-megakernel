# Qwen3-30B-A3B NKI Optimization Roadmap

For the MLSys 2026 competition. MoE TKG kernel is already implemented; this covers what's next.

---

## Priority 1: Attention TKG with Fused QK-Norm

Qwen3 applies **per-head RMSNorm on Q and K** after projection (`q_layernorm`, `k_layernorm`), which the standard `attention_isa_kernel` does not fuse. This currently costs two extra HBM round-trips per layer per token.

A custom NKI attention kernel should fuse:
1. QKV projection matmul
2. Per-head Q/K RMSNorm
3. Flash decode (KV-cache read + softmax + output accumulation)

**Why it's high impact**: 40 decoder layers × 2 extra norm passes per layer = 80 extra memory round-trips per token. Fusing eliminates all of them. The per-head QK-norm is Qwen3-specific — no reference NKI kernel does this, making it novel for the innovation score.

**Base**: nkilib `attention_tkg` kernel + add fused per-head norm before the QK dot-product.

**Model dims** (Qwen3-30B-A3B, TP=4):
- 64 Q heads, 8 KV heads (8:1 GQA ratio), head_dim=128, hidden_size=2048

---

## Priority 2: MoE CTE Kernel

The TKG kernel doesn't help **time-to-first-token** — CTE processes the full prompt (T tokens) dispatched to 8-of-128 active experts. The current path uses `initialize_moe_module` with standard nkilib ops.

Key characteristics to optimize for:
- H=2048, I_per_TP=192 (768/4), K=8, E=128
- Wide hidden states, relatively small intermediate → DMA-bound on weight loads
- Coalesced gate+up weight DMA across the active 8 experts per token batch

**Base**: nkilib `moe_cte` as reference; apply the same coalesced-DMA and widened-matmul techniques from the v6b TKG kernel.

---

## Priority 3: Fused Expert Router

The router computes `[T, H] × [H, 128] → softmax → top-8`. Trivial for TKG (T=1) but meaningful for CTE at long prompt lengths. A fused NKI kernel combining linear + softmax + topk:
- Reduces memory traffic by keeping the [T, 128] logit tensor on-chip
- Counts toward the "3 NKI parts" verification requirement
- Low complexity relative to the attention and MoE kernels

---

## Summary

| Kernel | Primary Metric | Novelty | Priority |
|--------|---------------|---------|----------|
| Attention TKG + fused QK-norm | Per-token latency | High (Qwen3-specific) | **#1** |
| MoE CTE | Time-to-first-token | Medium | **#2** |
| MoE TKG (v6b) | Per-token latency | Done | ✅ |
| Fused router | Minor / compliance | Low | **#3** |

The attention kernel with fused QK-norm is the strongest differentiator: real latency impact, model-specific, and not covered by any reference implementation.
