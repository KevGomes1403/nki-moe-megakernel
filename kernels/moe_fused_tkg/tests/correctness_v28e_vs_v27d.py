"""
Correctness check: kernel_v28e vs kernel_v27d.
Tests 7 seeds. Reports max_diff per seed and allclose(rtol=0.05, atol=0.05).
"""
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

v27d_mod = load_module(
    "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v27d.py", "_k27d"
)
v28e_mod = load_module(
    "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v28e.py", "_k28e"
)

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
    down_w = (torch.randn(E, I, H) * scale).to(torch.bfloat16)

    inp_d      = inp.to(device)
    gamma_d    = gamma.to(device)
    router_w_d = router_w.to(device)
    gate_up_d  = gate_up_flat.to(device)
    down_w_d   = down_w.to(device)
    xm.mark_step()

    ref_out = v27d_mod.run(inp_d, gamma_d, router_w_d, gate_up_d, down_w_d)
    xm.mark_step()
    ref_cpu = ref_out.cpu()

    # Re-move tensors since mark_step may have invalidated them
    inp_d      = inp.to(device)
    gamma_d    = gamma.to(device)
    router_w_d = router_w.to(device)
    gate_up_d  = gate_up_flat.to(device)
    down_w_d   = down_w.to(device)
    xm.mark_step()

    kern_out = v28e_mod.run(inp_d, gamma_d, router_w_d, gate_up_d, down_w_d)
    xm.mark_step()
    kern_cpu = kern_out.cpu()

    ref_flat  = ref_cpu.reshape(-1, H).float()
    kern_flat = kern_cpu.reshape(-1, H).float()

    max_diff = (ref_flat - kern_flat).abs().max().item()
    ok = torch.allclose(kern_flat, ref_flat, rtol=0.05, atol=0.05)
    status = "PASS" if ok else "FAIL"
    print(f"seed={seed}  max_diff={max_diff:.4e}  {status}")
    if ok:
        passed += 1
    else:
        failed += 1
        fails = torch.where((ref_flat - kern_flat).abs() > 0.05 * ref_flat.abs() + 0.05)
        for i in fails[1][:5].tolist():
            print(f"  fail pos={i}: kern={kern_flat[0,i]:.4f} ref={ref_flat[0,i]:.4f}")

print(f"\n{'='*50}")
print(f"SUMMARY: {passed}/{len(SEEDS)} seeds PASS  (allclose rtol=0.05 atol=0.05)")
if failed > 0:
    print("OVERALL: FAIL")
    sys.exit(1)
else:
    print("OVERALL: PASS")
