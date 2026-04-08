"""Correctness test: kernel_v28c vs kernel_v27d."""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import importlib.util

def load_kernel(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ref_mod = load_kernel("/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v27d.py", "v27d")
new_mod = load_kernel("/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v28c.py", "v28c")

device = xm.xla_device()
torch.manual_seed(42)
B, H, E, K, I = 1, 2048, 128, 8, 192
scale = 0.1
inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16).to(device)
gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16).to(device)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16).to(device)
gate_up_w = torch.zeros(E, H, 2*I, dtype=torch.bfloat16)
gate_up_w[:, :, 0:I]   = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
gate_up_w[:, :, I:2*I] = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
gate_up_w = gate_up_w.to(device)
xm.mark_step()
down_w = (torch.randn(E, I, H) * scale).to(torch.bfloat16).to(device)
xm.mark_step()

ref_out = ref_mod.qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
xm.mark_step()
ref_np = ref_out.cpu().to(torch.float32).numpy()

new_out = new_mod.qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
xm.mark_step()
new_np = new_out.cpu().to(torch.float32).numpy()

max_diff = np.abs(ref_np - new_np).max()
print(f"max_diff = {max_diff:.2e}")
np.testing.assert_allclose(new_np, ref_np, rtol=1e-2, atol=1e-2)
print("CORRECTNESS: PASS")
