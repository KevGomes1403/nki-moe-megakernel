"""
Correctness test for v15g vs v15c.

v15g [G] within-kernel Wave 0->Wave 1 prefetch: identical arithmetic to v15c;
only op ordering within the stage-4 block differs (Wave 1 scale extraction
and up-scale rearrange are deferred to right before Wave 1 compute, so the
Wave 1 DMA can overlap with Wave 0 compute without a scale-setup serialization
barrier). Output must match v15c exactly (bit-exact expected; allow tiny
rtol/atol for any compiler reordering of non-associative fp ops).

Exits non-zero on failure.
"""
import os, sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from kernels.moe_fused_tkg.quantized.v15c import run as run_v15c
from kernels.moe_fused_tkg.quantized.v15c import pack_gate_up
from kernels.moe_fused_tkg.quantized.v15g import run as run_v15g

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
# Generate inputs — same recipe as test_v15c.py
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
# Pack + move to device
# -----------------------------------------------------------------------
print("Packing gate_up for v15c/v15g...")
gate_up_packed = pack_gate_up(gate_up_w_q, gate_up_scales)  # [E, 17, 128, 384] int8

inp_dev    = inp.to(device)
gamma_dev  = gamma.to(device)
router_dev = router_w.to(device)
gu_packed_dev = gate_up_packed.to(device)
dw_dev     = down_w_q.to(device)
ds_dev     = down_scales_full.to(device)
xm.mark_step()

# -----------------------------------------------------------------------
# Run reference (v15c) and candidate (v15g)
# -----------------------------------------------------------------------
print("Running v15c reference kernel...")
v15c_result = run_v15c(inp_dev, gamma_dev, router_dev, gu_packed_dev, dw_dev, ds_dev)
xm.mark_step()
v15c_np = v15c_result.cpu().float().numpy()

print("Running v15g kernel...")
v15g_result = run_v15g(inp_dev, gamma_dev, router_dev, gu_packed_dev, dw_dev, ds_dev)
xm.mark_step()
v15g_np = v15g_result.cpu().float().numpy()

# -----------------------------------------------------------------------
# Check: v15g vs v15c
# -----------------------------------------------------------------------
max_diff = float(np.abs(v15g_np - v15c_np).max())
print(f"\nmax_diff = {max_diff:.4e}")
try:
    np.testing.assert_allclose(v15g_np, v15c_np, rtol=1e-3, atol=1e-3)
    print(f"PASS  max_diff={max_diff:.2e}  (rtol=1e-3, atol=1e-3, ref=v15c)")
except AssertionError as e:
    print(f"FAIL  max_diff={max_diff:.2e}")
    print(str(e))
    sys.exit(1)
