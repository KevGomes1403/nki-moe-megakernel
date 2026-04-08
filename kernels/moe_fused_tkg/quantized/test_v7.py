import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
import numpy as np
import torch
import torch_xla.core.xla_model as xm
sys.path.insert(0, "/home/ubuntu/nki-moe")
from kernels.moe_fused_tkg.quantized.v6a import run as run_ref
from kernels.moe_fused_tkg.quantized.v7 import run as run_v7

device = xm.xla_device()
T, H, E, I, GU = 1, 2048, 128, 192, 384
H_SHARD_TP = 512
torch.manual_seed(42)

inp = (torch.randn(T, 1, H) * 0.1).to(torch.bfloat16).to(device)
gamma = (torch.ones(1, H) + torch.randn(1, H) * 0.1).to(torch.bfloat16).to(device)
router_w = (torch.randn(H, E) * 0.01).to(torch.bfloat16).to(device)

# Proper FP8 quantization for gate_up_w
gate_up_w_fp32 = torch.randn(E, H, GU) * 0.1
gate_up_scales_fp32 = gate_up_w_fp32.abs().amax(dim=1) / 240.0
gate_up_scales_fp32 = gate_up_scales_fp32.clamp(min=1e-6)
gate_up_w_fp32_scaled = gate_up_w_fp32 / gate_up_scales_fp32.unsqueeze(1)
gate_up_w_q = gate_up_w_fp32_scaled.clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8)

# Proper FP8 quantization for down_w
# down_scales are per-output-neuron on the H_SHARD=512 shard only
down_w_fp32 = torch.randn(E, I, H) * 0.1
down_scales_full = down_w_fp32[:, :, 0:H_SHARD_TP].abs().amax(dim=1) / 240.0  # [E, H_SHARD_TP=512]
down_scales_full = down_scales_full.clamp(min=1e-6)
# Scale the shard columns; rest of H can be arbitrary (not used)
down_w_fp32_shard = down_w_fp32[:, :, 0:H_SHARD_TP] / down_scales_full.unsqueeze(1)
down_w_fp32_rest = down_w_fp32[:, :, H_SHARD_TP:]
down_w_q_shard = down_w_fp32_shard.clamp(-240, 240).to(torch.float8_e4m3fn)
down_w_q_rest = torch.zeros(E, I, H - H_SHARD_TP, dtype=torch.float8_e4m3fn)
down_w_q = torch.cat([down_w_q_shard, down_w_q_rest], dim=2).view(torch.int8)

gate_up_w = gate_up_w_q.to(device)
gate_up_scales = gate_up_scales_fp32.to(device)
down_w = down_w_q.to(device)
down_scales = down_scales_full.to(device)
xm.mark_step()

ref = run_ref(inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales)
xm.mark_step()
ref_np = ref.cpu().float().numpy()
print(f"ref output range: [{ref_np[~np.isnan(ref_np)].min():.4f}, {ref_np[~np.isnan(ref_np)].max():.4f}]" if not np.all(np.isnan(ref_np)) else "ref all NaN")

result = run_v7(inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales)
xm.mark_step()
result_np = result.cpu().float().numpy()

max_diff = np.abs(result_np - ref_np).max()
print(f"max_diff={max_diff:.4e}")
np.testing.assert_allclose(result_np, ref_np, rtol=1e-2, atol=1e-2)
print("CORRECTNESS PASS")
