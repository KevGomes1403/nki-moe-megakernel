"""
Correctness test for v15e vs v15c.

v15e [X3]: fuses the post-matmul activation/scaling chain using
`nisa.scalar_tensor_tensor`. All tensors and DMA paths are identical to
v15c; only the per-i_tile up scale + GLU multiply and the per-expert
combined_down_scale + down multiply are collapsed into single VectorE
scalar_tensor_tensor instructions. The math is algebraically identical
(multiplications commute) and the engine runs fp32 internally, so
the result must match v15c up to BF16 / fp32 rounding noise.

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
from kernels.moe_fused_tkg.quantized.v15e import run as run_v15e

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
# Generate inputs — same recipe as test_v15c.py / bench_v15c_trn3.py
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

# Pack gate_up (weight + scales) for v15c and v15e — same packed HBM tensor
gate_up_packed = pack_gate_up(gate_up_w_q, gate_up_scales)

# -----------------------------------------------------------------------
# Move to device
# -----------------------------------------------------------------------
inp_dev    = inp.to(device)
gamma_dev  = gamma.to(device)
router_dev = router_w.to(device)
gup_dev    = gate_up_packed.to(device)
dw_dev     = down_w_q.to(device)
ds_dev     = down_scales_full.to(device)
xm.mark_step()

# -----------------------------------------------------------------------
# Run reference (v15c)
# -----------------------------------------------------------------------
print("Running v15c reference kernel...")
v15c_result = run_v15c(inp_dev, gamma_dev, router_dev, gup_dev, dw_dev, ds_dev)
xm.mark_step()
v15c_np = v15c_result.cpu().float().numpy()

# -----------------------------------------------------------------------
# Run candidate (v15e)
# -----------------------------------------------------------------------
print("Running v15e kernel...")
v15e_result = run_v15e(inp_dev, gamma_dev, router_dev, gup_dev, dw_dev, ds_dev)
xm.mark_step()
v15e_np = v15e_result.cpu().float().numpy()

# -----------------------------------------------------------------------
# Check: v15e vs v15c
# -----------------------------------------------------------------------
max_diff = float(np.abs(v15e_np - v15c_np).max())
print(f"\nmax_diff = {max_diff:.4e}")
try:
    np.testing.assert_allclose(v15e_np, v15c_np, rtol=1e-3, atol=1e-3)
    print(f"PASS  max_diff={max_diff:.2e}  (rtol=1e-3, atol=1e-3, ref=v15c)")
except AssertionError as e:
    print(f"FAIL  max_diff={max_diff:.2e}")
    print(str(e))
    sys.exit(1)
