"""
BF16 PyTorch reference implementation for Qwen3 MoE TKG kernel.
Implements: RMSNorm + Router + TopK(8) + Expert MLP in BF16,
dequantizing FP8 weights using the provided scales.
"""
import torch
import torch.nn.functional as F


def pytorch_bf16_reference(inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales):
    """
    BF16 PyTorch reference implementation.
    inp: [B, 1, H=2048] bf16
    gamma: [1, H=2048] bf16
    router_w: [H=2048, E=128] bf16
    gate_up_w: [E=128, H=2048, GU=384] int8 (fp8_e4m3fn encoded)
    gate_up_scales: [E=128, GU=384] float32 per-output-neuron
    down_w: [E=128, I=192, H=2048] int8 (fp8_e4m3fn encoded)
    down_scales: [E=128, H=2048] float32 per-output-neuron
    Returns: [T, H=2048] bf16
    """
    H, E, I = 2048, 128, 192
    GU = 2 * I  # 384
    T = inp.shape[0]

    # Move everything to CPU for the reference
    inp_cpu = inp.reshape(T, H).float().cpu()
    gamma_f = gamma.reshape(H).float().cpu()
    router_w_f = router_w.float().cpu()
    gate_up_w_cpu = gate_up_w.cpu()
    gate_up_scales_cpu = gate_up_scales.float().cpu()
    down_w_cpu = down_w.cpu()
    down_scales_cpu = down_scales.float().cpu()

    # RMSNorm
    rms = inp_cpu.pow(2).mean(-1, keepdim=True)
    normed = inp_cpu * torch.rsqrt(rms + 1e-6) * gamma_f.unsqueeze(0)  # [T, H] fp32
    normed_bf16 = normed.to(torch.bfloat16)

    # Router
    logits = normed_bf16.float() @ router_w_f  # [T, E]
    # Stable softmax
    logits_max = logits.max(-1, keepdim=True).values
    exp_l = (logits - logits_max).exp()
    probs = exp_l / exp_l.sum(-1, keepdim=True)  # [T, E]

    # TopK(8) with normalization
    top8_vals, top8_idx = probs.topk(8, dim=-1)  # [T, 8]
    norm_weights = top8_vals / top8_vals.sum(-1, keepdim=True)  # [T, 8]

    output = torch.zeros(T, H, dtype=torch.float32)

    for t_idx in range(T):
        for k in range(8):
            eid = top8_idx[t_idx, k].item()
            w = norm_weights[t_idx, k].item()

            # Dequantize gate_up weights: [H, GU] × scale [GU] per output neuron
            gu_fp8 = gate_up_w_cpu[eid].view(torch.float8_e4m3fn).to(torch.float32)  # [H, GU]
            gu_scales = gate_up_scales_cpu[eid]  # [GU] fp32
            gu_bf16 = (gu_fp8 * gu_scales.unsqueeze(0)).to(torch.bfloat16)  # [H, GU]

            # Gate + Up projection: normed [1, H] @ gu_bf16 [H, GU] → [1, GU]
            h = normed_bf16[t_idx:t_idx + 1].float() @ gu_bf16.float()  # [1, GU]

            gate = h[:, :I]   # [1, I]
            up = h[:, I:]     # [1, I]
            inter = F.silu(gate) * up  # [1, I] — SiLU(gate) × up
            inter_bf16 = inter.to(torch.bfloat16)

            # Dequantize down weights: [I, H] × scale [H] per output neuron
            d_fp8 = down_w_cpu[eid].view(torch.float8_e4m3fn).to(torch.float32)  # [I, H]
            d_scales = down_scales_cpu[eid]  # [H] fp32
            d_bf16 = (d_fp8 * d_scales.unsqueeze(0)).to(torch.bfloat16)  # [I, H]

            # Down projection: inter [1, I] @ d_bf16 [I, H] → [1, H]
            out_contrib = inter_bf16.float() @ d_bf16.float()  # [1, H]

            output[t_idx] += w * out_contrib[0]

    return output.to(torch.bfloat16)
