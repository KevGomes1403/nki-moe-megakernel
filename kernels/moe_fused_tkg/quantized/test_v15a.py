"""
Correctness test for v15a vs v14a as reference.

v15a is a mechanical rewrite of v14a that only changes the DMA descriptor
generation path for the 10 indirect expert-weight / scale loads (dge_mode=0
=> nisa.dge_mode.hwdge). Functional output must match v14a bit-for-bit up
to floating-point noise (rtol=1e-3, atol=1e-3), since both kernels use the
same seed, same weights, same reduction order, and same activation path.

Exits non-zero on failure.
"""
import os, sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from kernels.moe_fused_tkg.quantized.v14a import run as run_v14a
from kernels.moe_fused_tkg.quantized.v15a import run as run_v15a

device = xm.xla_device()

# Shapes matching kernel constants
B = 1
T = B
H = 2048
E = 128
I = 192
K = 8
GU = 384
FP8_MAX = 240.0

torch.manual_seed(42)

# -----------------------------------------------------------------------
# Generate inputs — same recipe as test_v14a.py so the numerics lineage
# is identical to what the reference kernel was validated against.
# -----------------------------------------------------------------------
inp    = (torch.randn(B, 1, H) * 0.1).to(torch.bfloat16)
gamma  = (torch.ones(1, H) + torch.randn(1, H) * 0.1).to(torch.bfloat16)
router_w = (torch.randn(H, E) * 0.01).to(torch.bfloat16)

gate_up_w_fp32 = torch.randn(E, H, GU) * 0.1
gate_up_scales = (gate_up_w_fp32.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-6)
gate_up_w_q = (gate_up_w_fp32 / gate_up_scales.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.int8)

down_w_fp32 = torch.randn(E, I, H) * 0.1
down_scales_full = (down_w_fp32.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-6)
down_w_q = (down_w_fp32 / down_scales_full.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.int8)

# -----------------------------------------------------------------------
# Move to device
# -----------------------------------------------------------------------
inp_dev    = inp.to(device)
gamma_dev  = gamma.to(device)
router_dev = router_w.to(device)
gu_w_dev   = gate_up_w_q.to(device)
gu_sc_dev  = gate_up_scales.to(device)
dw_dev     = down_w_q.to(device)
ds_dev     = down_scales_full.to(device)
xm.mark_step()

# -----------------------------------------------------------------------
# Run reference (v14a)
# -----------------------------------------------------------------------
print("Running v14a reference kernel...")
v14a_result = run_v14a(inp_dev, gamma_dev, router_dev, gu_w_dev, gu_sc_dev, dw_dev, ds_dev)
xm.mark_step()
v14a_np = v14a_result.cpu().float().numpy()

# -----------------------------------------------------------------------
# Run v15a
# -----------------------------------------------------------------------
print("Running v15a kernel...")
v15a_result = run_v15a(inp_dev, gamma_dev, router_dev, gu_w_dev, gu_sc_dev, dw_dev, ds_dev)
xm.mark_step()
v15a_np = v15a_result.cpu().float().numpy()

# -----------------------------------------------------------------------
# Check: v15a vs v14a (tight, since only DMA path changed)
# -----------------------------------------------------------------------
max_diff = float(np.abs(v15a_np - v14a_np).max())
print(f"\nmax_diff = {max_diff:.4e}")
try:
    np.testing.assert_allclose(v15a_np, v14a_np, rtol=1e-3, atol=1e-3)
    print(f"PASS  max_diff={max_diff:.2e}  (rtol=1e-3, atol=1e-3, ref=v14a)")
except AssertionError as e:
    print(f"FAIL  max_diff={max_diff:.2e}")
    print(str(e))
    sys.exit(1)
