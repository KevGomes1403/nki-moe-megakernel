"""
Correctness test for v14a (trn3 port of v12i — W8A16 BF16 activations) vs:
  1. PyTorch bf16 reference (full precision, tight tolerance since no activation quantization)
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from kernels.moe_fused_tkg.quantized.v14a import run as run_v14a

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
# PyTorch bf16 reference (full precision, W8A16 — weights dequantized)
# -----------------------------------------------------------------------
def bf16_reference(inp_f, gamma_f, router_w_f, gate_up_w_q_f, gate_up_scales_f,
                   down_w_q_f, down_scales_f):
    """W8A16 reference: weights dequantized to fp32, activations in bf16."""
    # RMSNorm
    inp2d = inp_f.reshape(T, H).float()
    rms = (inp2d.pow(2).mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
    normed = (inp2d * rms * gamma_f.float())  # [T, H] fp32

    # Router: [T, H] @ [H, E] → logits [T, E]
    logits = normed @ router_w_f.float()  # [T, E]

    # Softmax + TopK
    probs = torch.softmax(logits, dim=-1)  # [T, E]
    topk_vals, topk_idx = torch.topk(probs, K, dim=-1)  # [T, K]

    # Normalize affinities
    norm_w = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # [T, K]

    # Dequantize gate_up and down weights
    gate_up_w_dq = gate_up_w_q_f.view(torch.float8_e4m3fn).float() * gate_up_scales_f.unsqueeze(1)  # [E, H, GU]
    down_w_dq    = down_w_q_f.view(torch.float8_e4m3fn).float() * down_scales_f.unsqueeze(1)        # [E, I, H]

    out = torch.zeros(T, H, dtype=torch.float32)
    for t_idx in range(T):
        for ki in range(K):
            e = topk_idx[t_idx, ki].item()
            w = norm_w[t_idx, ki].item()

            x = normed[t_idx]  # [H] fp32

            # gate_up: [H, GU]
            gate_proj = x @ gate_up_w_dq[e, :, :I]   # [I]
            up_proj   = x @ gate_up_w_dq[e, :, I:]   # [I]
            hidden = torch.nn.functional.silu(gate_proj) * up_proj  # [I]

            # down: [I, H]
            out[t_idx] += w * (hidden @ down_w_dq[e])  # [H]

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

print("Running v14a kernel...")
v14a_result = run_v14a(inp_dev, gamma_dev, router_dev, gu_w_dev, gu_sc_dev, dw_dev, ds_dev)
xm.mark_step()
v14a_np = v14a_result.cpu().float().numpy()

# -----------------------------------------------------------------------
# Run PyTorch bf16 reference (CPU)
# -----------------------------------------------------------------------
print("Running PyTorch bf16/W8A16 reference...")
ref_bf16 = bf16_reference(inp, gamma, router_w, gate_up_w_q, gate_up_scales, down_w_q, down_scales_full)
ref_bf16_np = ref_bf16.float().numpy()

# -----------------------------------------------------------------------
# Check 1: vs bf16/W8A16 reference (tight — v14a uses bf16 activations, fp8 weights)
# -----------------------------------------------------------------------
max_diff_bf16 = np.abs(v14a_np - ref_bf16_np).max()
print(f"\n--- Check 1: v14a vs bf16/W8A16 reference ---")
print(f"max_diff = {max_diff_bf16:.4e}")
try:
    np.testing.assert_allclose(v14a_np, ref_bf16_np, rtol=0.05, atol=0.2)
    print(f"PASS  max_diff={max_diff_bf16:.2e}  (rtol=0.05, atol=0.2)")
except AssertionError as e:
    print(f"FAIL  max_diff={max_diff_bf16:.2e}")
    print(str(e))
    sys.exit(1)

# -----------------------------------------------------------------------
# Check 2: broader check for overall accuracy
# -----------------------------------------------------------------------
max_diff_2 = np.abs(v14a_np - ref_bf16_np).max()
print(f"\n--- Check 2: v14a vs bf16 reference (accuracy check) ---")
print(f"max_diff = {max_diff_2:.4e}")
try:
    np.testing.assert_allclose(v14a_np, ref_bf16_np, rtol=0.15, atol=2.0)
    print(f"PASS  max_diff={max_diff_2:.2e}  (rtol=0.15, atol=2.0)")
except AssertionError as e:
    print(f"FAIL  max_diff={max_diff_2:.2e}")
    print(str(e))

print("\n=== All checks done ===")
