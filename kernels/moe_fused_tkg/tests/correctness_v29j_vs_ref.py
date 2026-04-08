"""
Correctness check: kernel_v29j vs kernel_v27d as reference.

Tests 5+ seeds. Reports max_diff per seed and allclose(rtol=1e-2, atol=1e-2).
Weight layout: gate_up_w as [E, H, 2*I=384] flat bf16 passed to kernel.
Reference receives [E, H, 2, I=192] (gate=[:,:,0,:], up=[:,:,1,:]).
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
import numpy as np
import importlib.util

B, H, E, K, I = 1, 2048, 128, 8, 192
scale = 0.1
SEEDS = [0, 1, 2, 3, 4, 5, 6]

# ── load kernel modules ────────────────────────────────────────────────────────
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v29j_mod = load_module(
    "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v29j.py", "_k29j"
)
v27d_mod = load_module(
    "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v27d.py", "_k27d"
)

device = xm.xla_device()

passed = 0
failed = 0
results = []

for seed in SEEDS:
    torch.manual_seed(seed)

    inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16)
    gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16)
    router_w = (torch.randn(H, E) * scale).to(torch.bfloat16)

    # gate_up_w as [E, H, 2*I] flat (kernel native layout)
    gate_up_flat = torch.zeros(E, H, 2 * I, dtype=torch.bfloat16)
    gate_up_flat[:, :, 0:I]   = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
    gate_up_flat[:, :, I:2*I] = (torch.randn(E, H, I) * scale).to(torch.bfloat16)

    down_w = (torch.randn(E, I, H) * scale).to(torch.bfloat16)

    # ── v27d reference ─────────────────────────────────────────────────────────
    inp_d      = inp.to(device)
    gamma_d    = gamma.to(device)
    router_w_d = router_w.to(device)
    gate_up_d  = gate_up_flat.to(device)
    down_w_d   = down_w.to(device)
    xm.mark_step()

    ref_out = v27d_mod.run(inp_d, gamma_d, router_w_d, gate_up_d, down_w_d)
    xm.mark_step()
    ref_cpu = ref_out.cpu().reshape(-1, H)

    # ── v29j kernel ────────────────────────────────────────────────────────────
    inp_d2      = inp.to(device)
    gamma_d2    = gamma.to(device)
    router_w_d2 = router_w.to(device)
    gate_up_d2  = gate_up_flat.to(device)
    down_w_d2   = down_w.to(device)
    xm.mark_step()

    kern_out = v29j_mod.run(inp_d2, gamma_d2, router_w_d2, gate_up_d2, down_w_d2)
    xm.mark_step()
    kern_cpu = kern_out.cpu().reshape(-1, H)

    result_np = kern_cpu.float().numpy()
    ref_np    = ref_cpu.float().numpy()

    max_diff = float(np.abs(result_np - ref_np).max())
    try:
        np.testing.assert_allclose(result_np, ref_np, rtol=1e-2, atol=1e-2)
        ok = True
    except AssertionError:
        ok = False

    status = "PASS" if ok else "FAIL"
    results.append((seed, max_diff, ok))
    print(f"seed={seed}  max_diff={max_diff:.4e}  {status}")
    if ok:
        passed += 1
    else:
        failed += 1

print(f"\n{'='*50}")
print(f"SUMMARY: {passed}/{len(SEEDS)} seeds PASS  (assert_allclose rtol=1e-2 atol=1e-2)")
if failed > 0:
    print("OVERALL: FAIL")
    sys.exit(1)
else:
    print("OVERALL: PASS")
