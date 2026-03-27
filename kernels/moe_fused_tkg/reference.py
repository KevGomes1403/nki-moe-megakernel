"""
PyTorch CPU reference implementation of the fused Qwen3 MoE TKG block.

Implements exactly: RMSNorm -> Router(softmax+topK+normalize) -> Expert MLPs -> weighted sum.
"""

import torch
import torch.nn.functional as F


def qwen3_moe_fused_tkg_reference(
    inp: torch.Tensor,        # [B, S, H] or [T, H], bf16
    gamma: torch.Tensor,      # [1, H], bf16
    router_w: torch.Tensor,   # [H, E], bf16
    gate_up_w: torch.Tensor,  # [E, H, 2, I], bf16   (gate=[:,:,0,:], up=[:,:,1,:])
    down_w: torch.Tensor,     # [E, I, H], bf16
    eps: float = 1e-6,
    top_k: int = 8,
) -> torch.Tensor:
    """
    Returns: [T, H] bf16
    """
    # Flatten to [T, H]
    T_shape = inp.shape
    x = inp.reshape(-1, inp.shape[-1])  # [T, H]
    T, H = x.shape

    # RMSNorm in float32 for precision
    x_f32 = x.float()
    rms = x_f32.pow(2).mean(-1, keepdim=True)        # [T, 1]
    x_norm = x_f32 * torch.rsqrt(rms + eps)           # [T, H]
    x_norm = x_norm * gamma.float().reshape(1, H)     # [T, H] * [1, H]
    x_norm_bf16 = x_norm.to(torch.bfloat16)

    # Router: x_norm @ router_w using bf16 matmul (matches kernel's router_mm_dtype=bfloat16)
    logits = x_norm_bf16.float() @ router_w.float()   # [T, E]

    # Softmax (router_pre_norm=True: apply before top-K)
    affinities = torch.softmax(logits, dim=-1)        # [T, E], float32

    # Top-K selection
    topk_weights, topk_idx = torch.topk(affinities, k=top_k, dim=-1)  # [T, K]

    # Normalize top-K weights (norm_topk_prob=True)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # [T, K]

    # Expert MLPs with POST_SCALE accumulation
    # Use bf16 matmul to match the kernel's arithmetic precision
    output = torch.zeros(T, H, dtype=torch.float32)
    for t in range(T):
        for k in range(top_k):
            e = topk_idx[t, k].item()
            w = topk_weights[t, k].item()

            # Select expert weights (keep in bf16 to match kernel)
            gate_w = gate_up_w[e, :, 0, :]  # [H, I] bf16
            up_w   = gate_up_w[e, :, 1, :]  # [H, I] bf16
            d_w    = down_w[e, :, :]         # [I, H] bf16

            xt = x_norm_bf16[t].unsqueeze(0)  # [1, H] bf16

            gate = (xt @ gate_w).float()       # [1, I] bf16 -> fp32
            up   = (xt @ up_w).float()         # [1, I] bf16 -> fp32

            # SiLU(gate) * up  (SiLU = x * sigmoid(x))
            inter = F.silu(gate) * up                           # [1, I] fp32
            inter_bf16 = inter.to(torch.bfloat16)               # [1, I] bf16
            out_e = (inter_bf16 @ d_w).float()                  # [1, H] fp32

            # POST_SCALE: multiply by routing weight then accumulate
            output[t] += w * out_e.squeeze(0)

    return output.to(torch.bfloat16)
