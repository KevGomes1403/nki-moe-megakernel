"""
Correctness test for v14b (trn3-compatible W8A8 FP8, global activation quantization
via nc_stream_shuffle tree reduction) vs:
  1. PyTorch W8A8 global-max reference (same quantization strategy as kernel)
  2. PyTorch bf16 reference (full precision, loose tolerance)
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from kernels.moe_fused_tkg.quantized.v14b import run as run_v14b

device = xm.xla_device()

# Shapes matching kernel constants
B = 1
T = B
H = 2048
E = 128
I = 192
K = 8
GU = 384
FP8_MAX = 240.0

torch.manual_seed(42)

# -----------------------------------------------------------------------
# Generate inputs
# -----------------------------------------------------------------------
inp    = (torch.randn(B, 1, H) * 0.1).to(torch.bfloat16)
gamma  = (torch.ones(1, H) + torch.randn(1, H) * 0.1).to(torch.bfloat16)
router_w = (torch.randn(H, E) * 0.01).to(torch.bfloat16)

# Gate+Up weights: proper fp8 quantization — per-output-neuron scale
gate_up_w_fp32 = torch.randn(E, H, GU) * 0.1
gate_up_scales = (gate_up_w_fp32.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-6)  # [E, GU]
gate_up_w_q = (gate_up_w_fp32 / gate_up_scales.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.int8)

# Down weights: proper fp8 quantization — per-output-neuron scale
down_w_fp32 = torch.randn(E, I, H) * 0.1
down_scales_full = (down_w_fp32.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-6)  # [E, H]
down_w_q = (down_w_fp32 / down_scales_full.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.int8)

# -----------------------------------------------------------------------
# PyTorch bf16 reference (full precision, for accuracy check)
# -----------------------------------------------------------------------
def bf16_reference(inp_f, gamma_f, router_w_f, gate_up_w_fp32_f, gate_up_scales_f,
                   down_w_fp32_f, down_scales_f):
    """Full-precision MoE forward in bf16."""
    # RMSNorm
    inp2d = inp_f.reshape(T, H).float()
    rms = (inp2d.pow(2).mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
    normed = (inp2d * rms * gamma_f.float())  # [T, H]

    # Router: [T, H] @ [H, E] -> logits [T, E]
    logits = normed @ router_w_f.float()  # [T, E]

    # Softmax + TopK
    probs = torch.softmax(logits, dim=-1)  # [T, E]
    topk_vals, topk_idx = torch.topk(probs, K, dim=-1)  # [T, K]

    # Normalize affinities
    norm_w = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # [T, K]

    # Dequantize weights for bf16 forward
    gu_dequant = gate_up_w_fp32_f  # already fp32
    down_dequant = down_w_fp32_f   # already fp32

    out = torch.zeros(T, H, dtype=torch.float32)
    for t_idx in range(T):
        for ki in range(K):
            e = topk_idx[t_idx, ki].item()
            w = norm_w[t_idx, ki].item()

            x = normed[t_idx]  # [H] fp32

            # gate_up: [H, GU]
            gu_w = gu_dequant[e]  # [H, GU]
            gate_proj = x @ gu_w[:, :I]   # [I]
            up_proj   = x @ gu_w[:, I:]   # [I]
            hidden = torch.nn.functional.silu(gate_proj) * up_proj  # [I]

            # down: [I, H]
            down_w_e = down_dequant[e]  # [I, H]
            out[t_idx] += w * (hidden @ down_w_e)  # [H]

    return out.bfloat16()


# -----------------------------------------------------------------------
# PyTorch W8A8 reference (global max — matches v14b's tree-reduction strategy)
# -----------------------------------------------------------------------
def w8a8_reference(inp_f, gamma_f, router_w_f, gate_up_w_q_f, gate_up_scales_f,
                   down_w_q_f, down_scales_f):
    """W8A8 reference: quantize activations with global absmax (same as kernel)."""
    # RMSNorm
    inp2d = inp_f.reshape(T, H).float()
    rms = (inp2d.pow(2).mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
    normed = (inp2d * rms * gamma_f.float())  # [T, H] fp32

    # Quantize RMSNorm output — per-token global absmax (same as v13a/v14b)
    act_absmax_gu = normed.abs().max()
    act_scale_gu = act_absmax_gu / FP8_MAX
    act_scale_gu = act_scale_gu.clamp(min=1e-10)
    normed_q = (normed / act_scale_gu).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    normed_dq = normed_q.float() * act_scale_gu  # dequantize for matmul

    # Router: use bf16 (not quantized per spec)
    logits = normed @ router_w_f.float()
    probs = torch.softmax(logits, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, K, dim=-1)
    norm_w = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

    # Dequantize gate_up and down weights
    gate_up_w_dq = gate_up_w_q_f.view(torch.float8_e4m3fn).float() * gate_up_scales_f.unsqueeze(1)  # [E, H, GU]
    down_w_dq    = down_w_q_f.view(torch.float8_e4m3fn).float() * down_scales_f.unsqueeze(1)        # [E, I, H]

    out = torch.zeros(T, H, dtype=torch.float32)
    for t_idx in range(T):
        for ki in range(K):
            e = topk_idx[t_idx, ki].item()
            w = norm_w[t_idx, ki].item()

            x = normed_dq[t_idx]  # [H] — dequantized activation

            # gate_up
            gate_proj = x @ gate_up_w_dq[e, :, :I]   # [I]
            up_proj   = x @ gate_up_w_dq[e, :, I:]   # [I]

            # Quantize intermediate (SiLU*Up output) — global absmax same as kernel
            inter = torch.nn.functional.silu(gate_proj) * up_proj  # [I]
            act_absmax_down = inter.abs().max()
            act_scale_down = (act_absmax_down / FP8_MAX).clamp(min=1e-10)
            inter_q = (inter / act_scale_down).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            inter_dq = inter_q.float() * act_scale_down

            # down
            out[t_idx] += w * (inter_dq @ down_w_dq[e])  # [H]

    return out.bfloat16()


# -----------------------------------------------------------------------
# Run kernel
# -----------------------------------------------------------------------
inp_dev    = inp.to(device)
gamma_dev  = gamma.to(device)
router_dev = router_w.to(device)
gu_w_dev   = gate_up_w_q.to(device)
gu_sc_dev  = gate_up_scales.to(device)
dw_dev     = down_w_q.to(device)
ds_dev     = down_scales_full.to(device)
xm.mark_step()

print("Running v14b kernel (trn3)...")
v14b_result = run_v14b(inp_dev, gamma_dev, router_dev, gu_w_dev, gu_sc_dev, dw_dev, ds_dev)
xm.mark_step()
v14b_np = v14b_result.cpu().float().numpy()

# -----------------------------------------------------------------------
# Run PyTorch references (CPU)
# -----------------------------------------------------------------------
print("Running PyTorch W8A8 reference (global max)...")
ref_w8a8_pp = w8a8_reference(inp, gamma, router_w, gate_up_w_q, gate_up_scales, down_w_q, down_scales_full)
ref_w8a8_pp_np = ref_w8a8_pp.float().numpy()

print("Running PyTorch bf16 reference...")
ref_bf16 = bf16_reference(inp, gamma, router_w, gate_up_w_fp32, gate_up_scales, down_w_fp32, down_scales_full)
ref_bf16_np = ref_bf16.float().numpy()

# -----------------------------------------------------------------------
# Check 1: vs W8A8 per-partition reference (tight — same quantization)
# -----------------------------------------------------------------------
max_diff_w8a8 = np.abs(v14b_np - ref_w8a8_pp_np).max()
print(f"\n--- Check 1: v14b vs W8A8 global-max reference ---")
print(f"max_diff = {max_diff_w8a8:.4e}")
try:
    np.testing.assert_allclose(v14b_np, ref_w8a8_pp_np, rtol=0.05, atol=0.6)
    print(f"PASS  max_diff={max_diff_w8a8:.2e}  (rtol=0.05, atol=0.6)")
except AssertionError as e:
    print(f"FAIL  max_diff={max_diff_w8a8:.2e}")
    print(str(e))
    sys.exit(1)

# -----------------------------------------------------------------------
# Check 2: vs bf16 reference (loose — fp8 quantization degrades precision)
# -----------------------------------------------------------------------
max_diff_bf16 = np.abs(v14b_np - ref_bf16_np).max()
print(f"\n--- Check 2: v14b vs bf16 reference (accuracy check) ---")
print(f"max_diff = {max_diff_bf16:.4e}")
try:
    np.testing.assert_allclose(v14b_np, ref_bf16_np, rtol=0.15, atol=2.0)
    print(f"PASS  max_diff={max_diff_bf16:.2e}  (rtol=0.15, atol=2.0)")
except AssertionError as e:
    print(f"FAIL  max_diff={max_diff_bf16:.2e}")
    print(str(e))
    # Don't exit — this is an accuracy check, not a correctness check
