"""Correctness check: kernel_v27h vs kernel_v19b."""
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
    return mod.qwen3_moe_fused_tkg

kernel_ref  = load_kernel("/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v19b.py", "_k19b")
kernel_cand = load_kernel("/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v27h.py", "_k27h")

device = xm.xla_device()
torch.manual_seed(42)
B, H, E, K, I = 1, 2048, 128, 8, 192
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

print("Running v19b reference...")
ref_out  = kernel_ref[2](*args); xm.mark_step(); ref_cpu  = ref_out.cpu()
print("Running v27h candidate...")
cand_out = kernel_cand[2](*args); xm.mark_step(); cand_cpu = cand_out.cpu()

max_diff = (ref_cpu - cand_cpu).abs().max().item()
print(f"\nmax_diff = {max_diff:.4e}")
try:
    torch.testing.assert_close(cand_cpu, ref_cpu, rtol=1e-2, atol=1e-2)
    print("CORRECTNESS: PASS")
except AssertionError as e:
    print(f"CORRECTNESS: FAIL\n{e}")
    sys.exit(1)
