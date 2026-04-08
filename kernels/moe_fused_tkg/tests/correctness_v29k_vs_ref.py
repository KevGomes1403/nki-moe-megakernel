"""
Correctness check: kernel_v29k vs kernel_v27d reference.

Tests 7 seeds. Reports max_diff per seed and allclose(rtol=1e-2, atol=1e-2).
Weight layout: gate_up_w as [E, H, 2*I=384] flat bf16 passed to kernel.
Reference (v27d) receives same flat layout.
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import numpy as np
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


kernel_mod = load_module(
    "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v29k.py", "_k29k"
)
ref_mod = load_module(
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

    # ── Reference kernel (v27d) ───────────────────────────────────────────────
    inp_d      = inp.to(device)
    gamma_d    = gamma.to(device)
    router_w_d = router_w.to(device)
    gate_up_d  = gate_up_flat.to(device)
    down_w_d   = down_w.to(device)
    xm.mark_step()

    ref_out = ref_mod.run(inp_d, gamma_d, router_w_d, gate_up_d, down_w_d)
    xm.mark_step()
    ref_cpu = ref_out.cpu()  # [T, H]

    # ── NKI kernel (v29k) ─────────────────────────────────────────────────────
    inp_d2      = inp.to(device)
    gamma_d2    = gamma.to(device)
    router_w_d2 = router_w.to(device)
    gate_up_d2  = gate_up_flat.to(device)
    down_w_d2   = down_w.to(device)
    xm.mark_step()

    kern_out = kernel_mod.run(inp_d2, gamma_d2, router_w_d2, gate_up_d2, down_w_d2)
    xm.mark_step()
    kern_cpu = kern_out.cpu()  # [T, H]

    ref_flat  = ref_cpu.reshape(-1, H).float().numpy()
    kern_flat = kern_cpu.reshape(-1, H).float().numpy()

    max_diff = float(np.abs(ref_flat - kern_flat).max())
    ok = True
    try:
        np.testing.assert_allclose(kern_flat, ref_flat, rtol=1e-2, atol=1e-2)
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
print(f"SUMMARY: {passed}/{len(SEEDS)} seeds PASS  (allclose rtol=1e-2 atol=1e-2)")
if failed > 0:
    print("OVERALL: FAIL")
    sys.exit(1)
else:
    print("OVERALL: PASS")
