"""Zero-output diagnostic for v14a and v15c. If both produce zeros bit-exactly,
the Round 1 MoE wins are the same false-positive as v17_fast_exp."""
import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"

import sys
sys.path.insert(0, "/home/ubuntu/nki-moe")
import numpy as np
import torch
import torch_xla.core.xla_model as xm
from kernels.moe_fused_tkg.quantized import v14a, v15c

E, H, I, GU = 128, 2048, 192, 384
torch.manual_seed(42)
dev = xm.xla_device()

# v14a inputs
inp = torch.randn(1, 1, H, dtype=torch.bfloat16).to(dev)
gamma = torch.randn(1, H, dtype=torch.bfloat16).to(dev)
router_w = torch.randn(H, E, dtype=torch.bfloat16).to(dev)
g = torch.randn(E, H, GU) * 0.1
gu_s = (g.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
gu_w = (g / gu_s.unsqueeze(1)).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8).to(dev)
gu_s = gu_s.to(dev)
d = torch.randn(E, I, H) * 0.1
dn_s = (d.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
dn_w = (d / dn_s.unsqueeze(1)).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8).to(dev)
dn_s = dn_s.to(dev)

print("=== v14a ===")
out = v14a.qwen3_moe_fused_tkg[2](inp, gamma, router_w, gu_w, gu_s, dn_w, dn_s)
r = out.cpu().float().numpy()
print(f"  shape={r.shape} range=[{r.min():.4f}, {r.max():.4f}] absmean={abs(r).mean():.4e}")
if abs(r).mean() < 1e-4:
    print("  [HARD FAIL] v14a output is ZERO")
else:
    print("  v14a OK (non-zero)")

print("=== v15c ===")
# v15c takes packed gate_up instead of separate gate_up_w + gate_up_scales
from kernels.moe_fused_tkg.quantized.v15c import pack_gate_up
# Recreate int8 and scales from the same seed, then pack
torch.manual_seed(42)
_ = torch.randn(1, 1, H, dtype=torch.bfloat16)
_ = torch.randn(1, H, dtype=torch.bfloat16)
_ = torch.randn(H, E, dtype=torch.bfloat16)
g2 = torch.randn(E, H, GU) * 0.1
gu_s2 = (g2.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
gu_w2 = (g2 / gu_s2.unsqueeze(1)).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8)
packed = pack_gate_up(gu_w2, gu_s2).to(dev)
# down_w / down_scales unchanged — reuse dn_w/dn_s from v14a block

out2 = v15c.qwen3_moe_fused_tkg[2](inp, gamma, router_w, packed, dn_w, dn_s)
r2 = out2.cpu().float().numpy()
print(f"  shape={r2.shape} range=[{r2.min():.4f}, {r2.max():.4f}] absmean={abs(r2).mean():.4e}")
if abs(r2).mean() < 1e-4:
    print("  [HARD FAIL] v15c output is ZERO")
else:
    print("  v15c OK (non-zero)")

# Cross-check: v14a and v15c should match bit-exactly if both work
print(f"=== cross-check ===")
print(f"  v14a vs v15c max_abs_err = {abs(r - r2).max():.6e}")
