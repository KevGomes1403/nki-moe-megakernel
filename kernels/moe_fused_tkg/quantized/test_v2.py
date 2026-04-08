import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
import importlib.util

def load_kernel(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ref_mod = load_kernel("/home/ubuntu/nki-moe/kernels/moe_fused_tkg/quantized/v0.py", "v0")
new_mod = load_kernel("/home/ubuntu/nki-moe/kernels/moe_fused_tkg/quantized/v2.py", "v2")

device = xm.xla_device()
torch.manual_seed(42)
B, H, E, K, I = 1, 2048, 128, 8, 192
GU_FLAT = 2 * I; GU_J_BLOCKS = GU_FLAT // 128; H_BLOCKS = H // 128; scale = 0.1

inp   = (torch.randn(B, 1, H) * scale).to(torch.bfloat16).to(device)
gamma = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16).to(device)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16).to(device)

gate_up_w_f32 = torch.zeros(E, H, GU_FLAT)
gate_up_w_f32[:, :, 0:I]   = torch.randn(E, H, I) * scale
gate_up_w_f32[:, :, I:2*I] = torch.randn(E, H, I) * scale
gate_up_w_rsh = gate_up_w_f32.reshape(E, H, GU_J_BLOCKS, 128)
gate_up_scales_cpu = gate_up_w_rsh.abs().amax(dim=3).clamp(min=1e-12) / 240.0
gate_up_scales_bcast = gate_up_scales_cpu.repeat_interleave(128, dim=2)
gate_up_w_fp8 = (gate_up_w_f32 / gate_up_scales_bcast).clamp(-240, 240).to(torch.float8_e4m3fn)

down_w_f32 = torch.randn(E, I, H) * scale
down_w_rsh = down_w_f32.reshape(E, I, H_BLOCKS, 128)
down_scales_cpu = down_w_rsh.abs().amax(dim=3).clamp(min=1e-12) / 240.0
down_scales_bcast = down_scales_cpu.repeat_interleave(128, dim=2)
down_w_fp8 = (down_w_f32 / down_scales_bcast).clamp(-240, 240).to(torch.float8_e4m3fn)

gate_up_w_int8 = gate_up_w_fp8.view(torch.int8).to(device)
gate_up_scales_dev = gate_up_scales_cpu.to(device)
down_w_int8 = down_w_fp8.view(torch.int8).to(device)
down_scales_dev = down_scales_cpu.to(device)
xm.mark_step()

ref_out = ref_mod.qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w_int8, gate_up_scales_dev, down_w_int8, down_scales_dev)
xm.mark_step()
ref_cpu = ref_out.cpu().to(torch.float32)

new_out = new_mod.qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w_int8, gate_up_scales_dev, down_w_int8, down_scales_dev)
xm.mark_step()
new_cpu = new_out.cpu().to(torch.float32)

import numpy as np
max_diff = np.abs(ref_cpu.numpy() - new_cpu.numpy()).max()
mean_diff = np.abs(ref_cpu.numpy() - new_cpu.numpy()).mean()
print(f"max_diff={max_diff:.3e}  mean_diff={mean_diff:.3e}")
try:
    np.testing.assert_allclose(new_cpu.numpy(), ref_cpu.numpy(), rtol=5e-2, atol=5e-2)
    print("PASS")
except AssertionError as e:
    print(f"FAIL: {e}")
