import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
import numpy as np
import torch
import torch_xla.core.xla_model as xm
sys.path.insert(0, "/home/ubuntu/nki-moe")
from kernels.moe_fused_tkg.quantized.v8 import run as run_v8

device = xm.xla_device()
T, H, E, I, GU = 1, 2048, 128, 192, 384
torch.manual_seed(42)

inp   = (torch.randn(T, 1, H) * 0.1).to(torch.bfloat16).to(device)
gamma = (torch.ones(1, H) + torch.randn(1, H) * 0.1).to(torch.bfloat16).to(device)
router_w = (torch.randn(H, E) * 0.01).to(torch.bfloat16).to(device)

gate_up_w_fp32 = torch.randn(E, H, GU) * 0.1
gate_up_scales = (gate_up_w_fp32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
gate_up_w_q = (gate_up_w_fp32 / gate_up_scales.unsqueeze(1)).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8)

down_w_fp32 = torch.randn(E, I, H) * 0.1
# down_scales_full: [E, H] — full H=2048, used for v8
down_scales_full = (down_w_fp32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)  # [E, H]
down_w_q = (down_w_fp32 / down_scales_full.unsqueeze(1)).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8)

# v8 uses full [E, H=2048]
gate_up_w_dev      = gate_up_w_q.to(device)
gate_up_scales_dev = gate_up_scales.to(device)
down_w_dev         = down_w_q.to(device)
down_scales_v8     = down_scales_full.to(device)              # v8 shape [E, 2048]
xm.mark_step()

result = run_v8(inp, gamma, router_w, gate_up_w_dev, gate_up_scales_dev, down_w_dev, down_scales_v8)
xm.mark_step()
result_np = result.cpu().float().numpy()

assert not np.all(np.isnan(result_np)), "v8 output is all NaN"
nan_frac = np.isnan(result_np).mean()
print(f"NaN fraction: {nan_frac:.2%}")
print(f"Output range: [{np.nanmin(result_np):.4f}, {np.nanmax(result_np):.4f}]")
print("BASIC SANITY PASS")
