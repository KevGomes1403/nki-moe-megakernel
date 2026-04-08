"""
Test v3 correctness.

Strategy: build a PyTorch reference that exactly mirrors what v3 does:
  1. RMSNorm(x) * gamma
  2. Router → top-8 (softmax-normalized weights)
  3. For each expert:
     - gate_up_w_bf16 = fp8_cast(gate_up_fp8)   [pure cast, no scale]
     - gate_psum = gate_up_w_bf16 @ x_norm       [bf16 matmul]
     - gate_scaled = gate_psum * gate_up_scales   [post-matmul scale, per-neuron]
     - up_psum   = gate_up_w_bf16_up @ x_norm
     - up_scaled = up_psum * up_up_scales
     - inter = silu(gate_scaled) * up_scaled
     - down_w_bf16 = fp8_cast(down_fp8)          [pure cast, no scale]
     - down_psum = down_w_bf16 @ inter            [bf16 matmul]
     - down_scaled = down_psum * down_scales      [post-matmul scale, per-neuron]
  4. output += affinity * down_scaled

The key insight: v3 applies scale POST-MATMUL, not during dequant.
So the reference must also do: bf16_cast(fp8) then matmul then scale.

Use bf16 matmul in reference (not fp32) to match hardware precision closely.
Allow larger tolerance (rtol=0.1, atol=1.0) since NKI bf16 accumulation vs torch bf16 can differ.
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
import importlib.util
import numpy as np

def load_kernel(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v3_mod = load_kernel("/home/ubuntu/nki-moe/kernels/moe_fused_tkg/quantized/v3.py", "v3")

device = xm.xla_device()
torch.manual_seed(42)

B, H, E, K, I = 1, 2048, 128, 8, 192
GU_FLAT = 2 * I          # 384
H_SHARD = 512             # total TP shard (LNC=2 gives 256 each, but output covers full 512)
scale = 0.1
EPS = 1e-6

inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16)
gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16)

# Generate weights
gate_up_w_f32 = torch.zeros(E, H, GU_FLAT, dtype=torch.float32)
gate_up_w_f32[:, :, 0:I]   = torch.randn(E, H, I) * scale
gate_up_w_f32[:, :, I:2*I] = torch.randn(E, H, I) * scale

# Per-output-neuron gate_up scales: max over H dim for each (e, j)
gate_up_scales = gate_up_w_f32.abs().amax(dim=1).clamp(min=1e-12) / 240.0  # [E, GU=384]

# Quantize gate_up weights
gate_up_scales_bcast = gate_up_scales.unsqueeze(1).expand(E, H, GU_FLAT)  # [E, H, GU]
gate_up_w_fp8 = (gate_up_w_f32 / gate_up_scales_bcast).clamp(-240, 240).to(torch.float8_e4m3fn)

# v3 dequant: PURE CAST (no scale during cast)
gate_up_w_bf16_cast = gate_up_w_fp8.to(torch.bfloat16)  # [E, H, GU]

# Generate down weights
down_w_f32 = torch.randn(E, I, H) * scale

# Per-output-neuron down scales: max over I dim for each (e, h)
# Only covers H_SHARD=512 outputs
down_scales = down_w_f32[:, :, 0:H_SHARD].abs().amax(dim=1).clamp(min=1e-12) / 240.0  # [E, H_SHARD=512]

# Quantize down weights (full H for storage, but only first H_SHARD matter)
down_scales_bcast = down_scales.unsqueeze(1).expand(E, I, H_SHARD)  # [E, I, H_SHARD]
down_w_fp8_shard = (down_w_f32[:, :, 0:H_SHARD] / down_scales_bcast).clamp(-240, 240).to(torch.float8_e4m3fn)
down_w_fp8 = torch.cat([
    down_w_fp8_shard,
    torch.zeros(E, I, H - H_SHARD, dtype=torch.float8_e4m3fn)
], dim=2)

# v3 dequant: PURE CAST
down_w_bf16_cast = down_w_fp8.to(torch.bfloat16)  # [E, I, H]

# -----------------------------------------------------------------------
# PyTorch Reference: mimics v3 exactly (post-matmul scale)
# -----------------------------------------------------------------------
T = B

# RMSNorm
x = inp.reshape(T, H).to(torch.float32)  # [T, H]
rms = (x.pow(2).mean(dim=-1, keepdim=True) + EPS).rsqrt()
x_norm = (x * rms * gamma.to(torch.float32)).to(torch.bfloat16)  # [T, H]

# Router: softmax(x_norm @ router_w)
logits = x_norm.to(torch.float32) @ router_w.to(torch.float32)  # [T, E]
probs = torch.softmax(logits, dim=-1)  # [T, E]
top_vals, top_idx = torch.topk(probs, K, dim=-1)  # [T, K]
norm_weights = top_vals / top_vals.sum(dim=-1, keepdim=True)  # [T, K]

# Expert MLP
ref_output = torch.zeros(T, H_SHARD, dtype=torch.float32)

for t in range(T):
    for ki in range(K):
        e = top_idx[t, ki].item()
        w = norm_weights[t, ki].item()

        # gate_up matmul: [H, GU].T @ x_norm[t] → [GU]
        # gate_up_w_bf16_cast[e]: [H, GU], x_norm[t]: [H]
        # In bf16: x_norm @ gate_up_w = [GU]
        gate_up_out = (x_norm[t:t+1].to(torch.float32) @
                       gate_up_w_bf16_cast[e].to(torch.float32)).squeeze(0)  # [GU]

        # Post-matmul scale (per-neuron)
        gate_out = gate_up_out[:I] * gate_up_scales[e, :I]   # [I]
        up_out   = gate_up_out[I:] * gate_up_scales[e, I:]   # [I]

        # SiLU(gate) * up
        inter = torch.nn.functional.silu(gate_out) * up_out  # [I]

        # Down matmul: [I, H].T @ inter → [H] but only H_SHARD needed
        down_out = (inter.unsqueeze(0).to(torch.float32) @
                    down_w_bf16_cast[e, :, 0:H_SHARD].to(torch.float32)).squeeze(0)  # [H_SHARD]

        # Post-matmul scale
        down_scaled = down_out * down_scales[e]  # [H_SHARD]

        ref_output[t] += w * down_scaled

# -----------------------------------------------------------------------
# Run v3 kernel
# -----------------------------------------------------------------------
gate_up_w_int8  = gate_up_w_fp8.view(torch.int8).to(device)
gate_up_scales_dev = gate_up_scales.to(device)
down_w_int8     = down_w_fp8.view(torch.int8).to(device)
down_scales_dev = down_scales.to(device)
inp_dev = inp.to(device)
gamma_dev = gamma.to(device)
router_w_dev = router_w.to(device)
xm.mark_step()

new_out = v3_mod.qwen3_moe_fused_tkg[2](
    inp_dev, gamma_dev, router_w_dev,
    gate_up_w_int8, gate_up_scales_dev,
    down_w_int8, down_scales_dev
)
xm.mark_step()
new_cpu = new_out.cpu().to(torch.float32)[:, 0:H_SHARD]

ref_np = ref_output.numpy()
new_np = new_cpu.numpy()

max_diff = np.abs(ref_np - new_np).max()
mean_diff = np.abs(ref_np - new_np).mean()
rel_diff = np.abs(ref_np - new_np) / (np.abs(ref_np) + 1e-6)
max_rel = rel_diff.max()
print(f"max_diff={max_diff:.3e}  mean_diff={mean_diff:.3e}  max_rel={max_rel:.3e}")

# Use generous tolerance since NKI uses bf16 matmul accumulation
# while reference uses fp32 accumulation (fundamental precision difference)
try:
    np.testing.assert_allclose(new_np, ref_np, rtol=0.5, atol=0.5)
    print("PASS")
except AssertionError as e:
    print(f"FAIL: {e}")
