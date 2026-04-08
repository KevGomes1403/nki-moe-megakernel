"""
Correctness test for kernel_v29_fp8.

Tests:
1. vs PyTorch FP8 dequant reference (functional correctness)
2. vs v28f bf16 kernel (accuracy impact of quantization)
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v28f = load_module("kernel_v28f", "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v28f.py")
v29  = load_module("kernel_v29_fp8", "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v29_fp8.py")

device = xm.xla_device()
torch.manual_seed(42)

B, H, E, K, I = 1, 2048, 128, 8, 192
GU_FLAT = 2 * I  # 384
GU_J_BLOCKS = GU_FLAT // 128  # 3  (blocks along GU_FLAT cols)
H_BLOCKS = H // 128            # 16 (blocks along H)
scale = 0.1

# Inputs
inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16).to(device)
gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16).to(device)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16).to(device)
xm.mark_step()

# Generate bf16 weights, then quantize to fp8
gate_up_w_bf16_cpu = torch.zeros(E, H, GU_FLAT, dtype=torch.float32)
gate_up_w_bf16_cpu[:, :, 0:I]      = torch.randn(E, H, I) * scale
gate_up_w_bf16_cpu[:, :, I:2*I]    = torch.randn(E, H, I) * scale
gate_up_w_bf16_cpu = gate_up_w_bf16_cpu.to(torch.bfloat16)

down_w_bf16_cpu = (torch.randn(E, I, H) * scale).to(torch.bfloat16)

# -----------------------------------------------------------------------
# Quantize gate_up_w: per-block-of-128 along GU_FLAT (column) dim
# gate_up_scales[e, h, j_blk] = max(|gate_up_w[e, h, j_blk*128:(j_blk+1)*128]|) / 448
# Scale constant across 128 GU columns, varies per H row (partition dim).
# -----------------------------------------------------------------------
gate_up_w_f32 = gate_up_w_bf16_cpu.float()  # [E, H, GU_FLAT]
# Reshape to [E, H, GU_J_BLOCKS=3, 128]
gate_up_w_rsh = gate_up_w_f32.reshape(E, H, GU_J_BLOCKS, 128)
gate_up_scales_cpu = gate_up_w_rsh.abs().amax(dim=3) / 240.0  # [E, H, GU_J_BLOCKS]
gate_up_scales_cpu = gate_up_scales_cpu.clamp(min=1e-12)

# Quantize: gate_up_w[e, h, j] / gate_up_scales[e, h, j//128]
# Clamp to ±240 (max safe fp8_e4m3fn value on trn2 hardware; values ≥256 have exponent=15 which maps to inf/NaN)
gate_up_scales_bcast = gate_up_scales_cpu.repeat_interleave(128, dim=2)  # [E, H, GU_FLAT]
gate_up_w_fp8_cpu = (gate_up_w_f32 / gate_up_scales_bcast).clamp(-240, 240).to(torch.float8_e4m3fn)

# -----------------------------------------------------------------------
# Quantize down_w: per-block-of-128 along H (column) dim
# down_scales[e, i, h_blk] = max(|down_w[e, i, h_blk*128:(h_blk+1)*128]|) / 448
# Scale constant across 128 H columns, varies per I row (partition dim).
# -----------------------------------------------------------------------
down_w_f32 = down_w_bf16_cpu.float()  # [E, I, H]
# Reshape to [E, I, H_BLOCKS=16, 128]
down_w_rsh = down_w_f32.reshape(E, I, H_BLOCKS, 128)
down_scales_cpu = down_w_rsh.abs().amax(dim=3) / 240.0  # [E, I, H_BLOCKS]
down_scales_cpu = down_scales_cpu.clamp(min=1e-12)

# Quantize: down_w[e, i, h] / down_scales[e, i, h//128]
# Clamp to ±240 (max safe fp8_e4m3fn value on trn2 hardware)
down_scales_bcast = down_scales_cpu.repeat_interleave(128, dim=2)  # [E, I, H]
down_w_fp8_cpu = (down_w_f32 / down_scales_bcast).clamp(-240, 240).to(torch.float8_e4m3fn)

print("Scales shapes:")
print(f"  gate_up_scales: {gate_up_scales_cpu.shape}")  # [E, H, GU_J_BLOCKS=3]
print(f"  down_scales: {down_scales_cpu.shape}")         # [E, I, H_BLOCKS=16]

# Move to device — pass fp8 as int8 to avoid XLA fp8 HBM limitation
gate_up_w_int8 = gate_up_w_fp8_cpu.view(torch.int8).to(device)
gate_up_scales_dev = gate_up_scales_cpu.to(device)
down_w_int8 = down_w_fp8_cpu.view(torch.int8).to(device)
down_scales_dev = down_scales_cpu.to(device)
xm.mark_step()

# -----------------------------------------------------------------------
# Reference 1: PyTorch FP8 dequant reference
# -----------------------------------------------------------------------
def pytorch_moe_ref(inp_cpu, gamma_cpu, router_w_cpu, gate_up_w_fp8_cpu, gate_up_scales_cpu,
                    down_w_fp8_cpu, down_scales_cpu, topk=8):
    """Pure PyTorch MoE reference with FP8 dequant.

    gate_up_scales: [E, H, GU_J_BLOCKS=3]  (per-block-of-128 along GU cols)
    down_scales:    [E, I, H_BLOCKS=16]     (per-block-of-128 along H cols)
    """
    B, _, Hd = inp_cpu.shape
    T = B
    Ec = gate_up_w_fp8_cpu.shape[0]
    GU = gate_up_w_fp8_cpu.shape[2]
    Ic = GU // 2

    # Dequantize gate_up: gate_up_w_fp8[e, h, j] * gate_up_scales[e, h, j//128]
    gate_up_w_dq = torch.zeros(Ec, Hd, GU, dtype=torch.float32)
    for e in range(Ec):
        w_fp8 = gate_up_w_fp8_cpu[e].float()  # [H, GU]
        scales = gate_up_scales_cpu[e]  # [H, GU_J_BLOCKS]
        # Broadcast: scales_bcast[h, j] = scales[h, j//128]
        scales_bcast = scales.repeat_interleave(128, dim=1)  # [H, GU]
        gate_up_w_dq[e] = w_fp8 * scales_bcast
    gate_up_w_dq = gate_up_w_dq.to(torch.bfloat16)

    # Dequantize down_w: down_w_fp8[e, i, h] * down_scales[e, i, h//128]
    Ireal = down_w_fp8_cpu.shape[1]
    down_w_dq = torch.zeros(Ec, Ireal, Hd, dtype=torch.float32)
    for e in range(Ec):
        w_fp8 = down_w_fp8_cpu[e].float()  # [I, H]
        scales = down_scales_cpu[e]  # [I, H_BLOCKS]
        # Broadcast: scales_bcast[i, h] = scales[i, h//128]
        scales_bcast = scales.repeat_interleave(128, dim=1)  # [I, H]
        down_w_dq[e] = w_fp8 * scales_bcast
    down_w_dq = down_w_dq.to(torch.bfloat16)

    inp_2d = inp_cpu.reshape(T, Hd).float()

    # RMSNorm
    norm_w = gamma_cpu.reshape(Hd).float()
    rms = (inp_2d.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
    normed = (inp_2d / rms) * norm_w

    # Router
    logits = normed @ router_w_cpu.float()  # [T, E]
    probs = torch.softmax(logits, dim=-1)
    top_vals, top_idx = probs.topk(topk, dim=-1)
    norm_weights = top_vals / top_vals.sum(dim=-1, keepdim=True)

    output = torch.zeros(T, Hd, dtype=torch.float32)
    for t in range(T):
        for k in range(topk):
            e = top_idx[t, k].item()
            w = norm_weights[t, k].item()
            x = normed[t]  # [H]
            gate_proj = x @ gate_up_w_dq[e, :, :Ic].float()  # [I]
            up_proj   = x @ gate_up_w_dq[e, :, Ic:].float()   # [I]
            inter = torch.nn.functional.silu(gate_proj) * up_proj  # [I]
            out_e = inter @ down_w_dq[e].float()  # [H]
            output[t] += w * out_e
    return output.to(torch.bfloat16)


inp_cpu = inp.cpu().to(torch.bfloat16)
gamma_cpu = gamma.cpu().to(torch.bfloat16)
router_w_cpu = router_w.cpu().to(torch.bfloat16)

print("\nRunning PyTorch reference...")
ref_out = pytorch_moe_ref(inp_cpu, gamma_cpu, router_w_cpu,
                           gate_up_w_fp8_cpu, gate_up_scales_cpu,
                           down_w_fp8_cpu, down_scales_cpu)
print(f"Ref output shape: {ref_out.shape}, dtype: {ref_out.dtype}")

print("\nRunning kernel_v29_fp8...")
kernel_out = v29.run(inp, gamma, router_w, gate_up_w_int8, gate_up_scales_dev, down_w_int8, down_scales_dev)
xm.mark_step()
kernel_out_cpu = kernel_out.cpu().to(torch.float32)
ref_out_f32 = ref_out.float()

max_diff_ref = (kernel_out_cpu - ref_out_f32).abs().max().item()
mean_diff_ref = (kernel_out_cpu - ref_out_f32).abs().mean().item()
print(f"\n=== vs PyTorch FP8 ref ===")
print(f"max_diff  = {max_diff_ref:.4e}")
print(f"mean_diff = {mean_diff_ref:.4e}")
# Tolerance for fp8+bf16 dequant compute: expect small relative error vs identical-logic ref
# Use absolute tol = 0.5 since outputs can be O(1) in magnitude with scale=0.1 weights
tol = 0.5
if max_diff_ref < tol:
    print(f"PASS (tol={tol})")
else:
    print(f"FAIL (tol={tol})")

# -----------------------------------------------------------------------
# Reference 2: v28f bf16 kernel with dequantized weights
# -----------------------------------------------------------------------
print("\nRunning v28f with dequantized bf16 weights...")
# Reconstruct dequantized bf16 weights (same as ref_out used)
gate_up_dq_bf16 = torch.zeros(E, H, GU_FLAT, dtype=torch.bfloat16)
for e in range(E):
    w_fp8 = gate_up_w_fp8_cpu[e].float()
    scales = gate_up_scales_cpu[e]  # [H, GU_J_BLOCKS]
    scales_bcast = scales.repeat_interleave(128, dim=1)  # [H, GU_FLAT]
    gate_up_dq_bf16[e] = (w_fp8 * scales_bcast).to(torch.bfloat16)

down_dq_bf16 = torch.zeros(E, I, H, dtype=torch.bfloat16)
for e in range(E):
    w_fp8 = down_w_fp8_cpu[e].float()  # [I, H]
    scales = down_scales_cpu[e]  # [I, H_BLOCKS]
    scales_bcast = scales.repeat_interleave(128, dim=1)  # [I, H]
    down_dq_bf16[e] = (w_fp8 * scales_bcast).to(torch.bfloat16)

gate_up_dq_dev = gate_up_dq_bf16.to(device)
down_dq_dev = down_dq_bf16.to(device)
xm.mark_step()

v28f_out = v28f.run(inp, gamma, router_w, gate_up_dq_dev, down_dq_dev)
xm.mark_step()
v28f_out_cpu = v28f_out.cpu().to(torch.float32)

max_diff_v28f = (kernel_out_cpu - v28f_out_cpu).abs().max().item()
mean_diff_v28f = (kernel_out_cpu - v28f_out_cpu).abs().mean().item()
print(f"\n=== vs v28f with same dequantized weights ===")
print(f"max_diff  = {max_diff_v28f:.4e}")
print(f"mean_diff = {mean_diff_v28f:.4e}")
print("(no strict threshold — this measures kernel implementation error only)")
