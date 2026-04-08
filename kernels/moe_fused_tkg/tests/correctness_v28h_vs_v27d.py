"""Correctness check: kernel_v28h vs kernel_v27d (reference)."""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
import importlib.util

B, H, E, K, I = 1, 2048, 128, 8, 192
scale = 0.1
SEEDS = [0, 1, 2, 3, 4, 5, 6]

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

kernel_new = load_module("/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v28h.py", "_k28h")
kernel_ref = load_module("/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v27d.py", "_k27d")

device = xm.xla_device()
passed = 0
failed = 0

for seed in SEEDS:
    torch.manual_seed(seed)

    inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16)
    gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16)
    router_w = (torch.randn(H, E) * scale).to(torch.bfloat16)
    gate_up_flat = torch.zeros(E, H, 2 * I, dtype=torch.bfloat16)
    gate_up_flat[:, :, 0:I]   = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
    gate_up_flat[:, :, I:2*I] = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
    down_w   = (torch.randn(E, I, H) * scale).to(torch.bfloat16)

    inp_d      = inp.to(device)
    gamma_d    = gamma.to(device)
    router_w_d = router_w.to(device)
    gate_up_d  = gate_up_flat.to(device)
    down_w_d   = down_w.to(device)
    xm.mark_step()

    ref_out  = kernel_ref.run(inp_d, gamma_d, router_w_d, gate_up_d, down_w_d)
    xm.mark_step()
    new_out  = kernel_new.run(inp_d, gamma_d, router_w_d, gate_up_d, down_w_d)
    xm.mark_step()

    ref_cpu = ref_out.cpu().reshape(-1, H).float()
    new_cpu = new_out.cpu().reshape(-1, H).float()

    max_diff = (ref_cpu - new_cpu).abs().max().item()
    ok = torch.allclose(new_cpu, ref_cpu, rtol=1e-2, atol=1e-2)
    status = "PASS" if ok else "FAIL"
    print(f"seed={seed}  max_diff={max_diff:.4e}  {status}")
    if ok:
        passed += 1
    else:
        failed += 1

print(f"\n{'='*50}")
print(f"SUMMARY: {passed}/{len(SEEDS)} seeds PASS  (allclose rtol=1e-2 atol=1e-2)")
if failed > 0:
    print("OVERALL: FAIL")
    sys.exit(1)
else:
    print("OVERALL: PASS")
