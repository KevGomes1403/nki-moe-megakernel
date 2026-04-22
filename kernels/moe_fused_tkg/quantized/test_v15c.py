"""
Correctness test for v15c vs v14a.

v15c [F3]: packs gate_up_w + gate_up_scales into a single merged HBM tensor
and loads them via a single (HWDGE-compatible) int8 DMA per (expert, wave).
The SBUF tile is then re-viewed as fp8 (for matmul) and fp32 (for scales).

Aside from the DMA packing, every arithmetic step is identical to v14a, so
the output must match v14a up to floating-point noise.

Exits non-zero on failure.
"""
import os, sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from kernels.moe_fused_tkg.quantized.v14a import run as run_v14a
from kernels.moe_fused_tkg.quantized.v15c import run as run_v15c
from kernels.moe_fused_tkg.quantized.v15c import pack_gate_up

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
# Generate inputs — same recipe as test_v15a.py
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
# Move to device for v14a
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
# Pack gate_up for v15c and run
# -----------------------------------------------------------------------
print("Packing gate_up (weight + scales) for v15c...")
gate_up_packed = pack_gate_up(gate_up_w_q, gate_up_scales)  # [E, 17, 128, 384] int8
gate_up_packed_dev = gate_up_packed.to(device)
xm.mark_step()

print("Running v15c kernel...")
v15c_result = run_v15c(inp_dev, gamma_dev, router_dev, gate_up_packed_dev, dw_dev, ds_dev)
xm.mark_step()
v15c_np = v15c_result.cpu().float().numpy()

# -----------------------------------------------------------------------
# Check: v15c vs v14a
# -----------------------------------------------------------------------
max_diff = float(np.abs(v15c_np - v14a_np).max())
print(f"\nmax_diff = {max_diff:.4e}")
try:
    np.testing.assert_allclose(v15c_np, v14a_np, rtol=1e-3, atol=1e-3)
    print(f"PASS  max_diff={max_diff:.2e}  (rtol=1e-3, atol=1e-3, ref=v14a)")
except AssertionError as e:
    print(f"FAIL  max_diff={max_diff:.2e}")
    print(str(e))
    sys.exit(1)
