"""Correctness check: kernel_v17a vs kernel_v16c reference."""

import os
import sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
import importlib.util

# Load v16c (reference)
spec16 = importlib.util.spec_from_file_location(
    "_kernel16c", "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v16c.py"
)
mod16 = importlib.util.module_from_spec(spec16)
spec16.loader.exec_module(mod16)
kernel_v16c = mod16.qwen3_moe_fused_tkg

# Load v17a (candidate)
spec17 = importlib.util.spec_from_file_location(
    "_kernel17a", "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v17a.py"
)
mod17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(mod17)
kernel_v17a = mod17.qwen3_moe_fused_tkg

# Create inputs
device = xm.xla_device()
torch.manual_seed(42)

B, H, E, K = 1, 2048, 128, 8
I = 192
scale = 0.1

inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16).to(device)
gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16).to(device)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16).to(device)

gate_up_w = torch.zeros(E, H, 2 * I, dtype=torch.bfloat16)
gate_up_w[:, :, 0:I]   = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
gate_up_w[:, :, I:2*I] = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
gate_up_w = gate_up_w.to(device)
xm.mark_step()

down_w = (torch.randn(E, I, H) * scale).to(torch.bfloat16).to(device)
xm.mark_step()

args = (inp, gamma, router_w, gate_up_w, down_w)

# Run reference
print("Running v16c reference...")
ref_out = kernel_v16c[2](*args)
xm.mark_step()
ref_cpu = ref_out.cpu()

# Run candidate
print("Running v17a candidate...")
cand_out = kernel_v17a[2](*args)
xm.mark_step()
cand_cpu = cand_out.cpu()

# Compare
max_diff = (ref_cpu - cand_cpu).abs().max().item()
print(f"\nmax_diff = {max_diff:.4e}")

try:
    torch.testing.assert_close(cand_cpu, ref_cpu, rtol=1e-2, atol=1e-2)
    print("CORRECTNESS: PASS")
except AssertionError as e:
    print(f"CORRECTNESS: FAIL\n{e}")
    sys.exit(1)
