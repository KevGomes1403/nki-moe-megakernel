"""
Fused MoE token generation kernel for Qwen3-30B-A3B (TP=4, LNC=2).

Fuses: pre-layer RMSNorm + Router + Top-K + Expert MLPs

Shapes (per TP rank, token generation):
  inp:       [B, S, H]    = [1, 1, 2048] bf16
  gamma:     [1, H]       = [1, 2048]    bf16
  router_w:  [H, E]       = [2048, 128]  bf16
  gate_up_w: [E, H, 2, I] = [128, 2048, 2, 256] bf16  (I=256 padded from 192)
  down_w:    [E, I, H]    = [128, 256, 2048] bf16     (I=256 padded from 192)
  output:    [T, H]       = [1, 2048]    bf16

Uses moe_block_tkg from nkilib with LNC=2, which internally:
  1. Runs RMSNorm on the input
  2. Computes router logits and applies softmax + top-K (K=8)
  3. Normalizes top-K weights (norm_topk_prob=True)
  4. Runs selective-expert MLP for each token's K experts
  5. Accumulates results weighted by normalized expert affinities (POST_SCALE)
"""

import nki.language as nl
from nkilib.core.moe_block.moe_block_tkg import moe_block_tkg
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode, RouterActFnType

# Qwen3-30B-A3B MoE configuration
_QWEN3_CONFIG = dict(
    eps=1e-6,
    top_k=8,
    router_act_fn=RouterActFnType.SOFTMAX,
    router_pre_norm=True,
    norm_topk_prob=True,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    hidden_act_fn=ActFnType.SiLU,
    router_mm_dtype=nl.bfloat16,
    skip_router_logits=True,
    is_all_expert=False,
)

# The kernel is moe_block_tkg from nkilib, configured for Qwen3.
# Calling convention: qwen3_moe_fused_tkg[2](inp, gamma, ...) uses LNC=2.
qwen3_moe_fused_tkg = moe_block_tkg


def run(inp, gamma, router_w, gate_up_w, down_w):
    """
    Run the fused Qwen3 MoE TKG kernel with LNC=2.

    Returns the expert MLP output after routing and accumulation.
    The caller must add the residual connection (not included here).
    """
    outputs = qwen3_moe_fused_tkg[2](
        inp=inp,
        gamma=gamma,
        router_weights=router_w,
        expert_gate_up_weights=gate_up_w,
        expert_down_weights=down_w,
        **_QWEN3_CONFIG,
    )
    # outputs is a tuple: (out,) when skip_router_logits=True
    return outputs[0]
