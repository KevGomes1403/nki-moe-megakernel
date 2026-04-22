"""
Phase 3 STEP 1a — verify qwen3_attn_tkg_v17_wrapper's output-order restoration.

Flow:
  - Run v17_fast_exp VIA THE WRAPPER: output shape [B, 1, H_wo] (v10e-compatible).
  - Compute PyTorch fp32 reference (identical recipe used in bench_v17_fast_exp).
  - assert_allclose(rtol=1e-2, atol=2e-2) — same tolerance the existing v17 bench uses.

If this passes, the 16× nc_transpose + dma_copy chain inside the wrapper produces
output in the SAME linear element order v10e produces.
"""
import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import math
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import sys
sys.path.insert(0, "/home/ubuntu/nki-moe")
from kernels.moe_fused_tkg.quantized._qwen_integration import qwen3_attn_tkg_v17_wrapper

B, H, d, Hq_tp, S_prior, H_wo, PMAX = 1, 2048, 128, 8, 640, 2048, 128

torch.manual_seed(42)
dev = xm.xla_device()
hs = torch.randn(B, 1, H, dtype=torch.bfloat16)
Wq = torch.randn(Hq_tp * d, H, dtype=torch.bfloat16) * 0.02
Wk = torch.randn(d, H, dtype=torch.bfloat16) * 0.02
Wv = torch.randn(d, H, dtype=torch.bfloat16) * 0.02
Wo = torch.randn(Hq_tp * d, H_wo, dtype=torch.bfloat16) * 0.02
q_n = torch.ones(d, dtype=torch.bfloat16)
k_n = torch.ones(d, dtype=torch.bfloat16)
Kc = torch.randn(B, 1, S_prior, d, dtype=torch.bfloat16) * 0.1
Vc = torch.randn(B, 1, S_prior, d, dtype=torch.bfloat16) * 0.1
cos = torch.cos(torch.linspace(0, 1, d)).reshape(B, d).to(torch.bfloat16)
sin = torch.sin(torch.linspace(0, 1, d)).reshape(B, d).to(torch.bfloat16)
pid = torch.tensor([[S_prior // 2]], dtype=torch.int32)

out, _, _ = qwen3_attn_tkg_v17_wrapper(
    hs.to(dev), Wq.to(dev), Wk.to(dev), Wv.to(dev), Wo.to(dev),
    q_n.to(dev), k_n.to(dev), Kc.to(dev), Vc.to(dev), cos.to(dev), sin.to(dev), pid.to(dev),
)
got = out.cpu().float().numpy()  # [B, 1, H_wo]

# PyTorch fp32 reference (same body as bench_v17_fast_exp.reference_attn).
h = hs.reshape(H, B).float()
q = (Wq.float() @ h).reshape(Hq_tp, d, B)
k = (Wk.float() @ h)
v = (Wv.float() @ h)
k_ms = (k**2).mean(dim=0, keepdim=True) + 1e-6
k = k * k_ms.rsqrt() * k_n.reshape(d, 1).float()
half = d // 2
cos_v = cos.reshape(d, B).float(); sin_v = sin.reshape(d, B).float()
rot_k = torch.cat([-k[half:], k[:half]], dim=0)
k_rope = k * cos_v + rot_k * sin_v
q_ms = (q**2).mean(dim=1, keepdim=True) + 1e-6
q = q * q_ms.rsqrt() * q_n.reshape(1, d, 1).float()
cos_h = cos_v.unsqueeze(0).expand(Hq_tp, -1, -1); sin_h = sin_v.unsqueeze(0).expand(Hq_tp, -1, -1)
rot_q = torch.cat([-q[:, half:], q[:, :half]], dim=1)
q = (q * cos_h + rot_q * sin_h) / math.sqrt(d)
pos = int(pid[0, 0])
K_all = torch.cat([Kc.reshape(S_prior, d).float().T, k_rope], dim=1)
V_all = torch.cat([Vc.reshape(S_prior, d).float().T, v], dim=1)
mask = torch.zeros(S_prior + 1, 1); mask[pos:S_prior] = -1e9
outs = []
for h_idx in range(Hq_tp):
    scores = (K_all.T @ q[h_idx]) + mask
    outs.append(V_all @ torch.softmax(scores, dim=0))
flat = torch.stack(outs, dim=1).reshape(d, Hq_tp).T.reshape(1, Hq_tp * d)
ref = (flat @ Wo.float()).reshape(B, 1, H_wo).numpy()

max_abs = np.abs(got - ref).max()
got_absmean = float(np.abs(got).mean())
ref_absmean = float(np.abs(ref).mean())
got_max, ref_max = float(np.abs(got).max()), float(np.abs(ref).max())

print(f"got  shape={got.shape} range=[{got.min():.6f}, {got.max():.6f}]  absmean={got_absmean:.6e}")
print(f"ref  shape={ref.shape} range=[{ref.min():.6f}, {ref.max():.6f}]  absmean={ref_absmean:.6e}")
print(f"max_abs_err = {max_abs:.6e}")

# TRIPLE-CHECK PATTERN (zero / range / allclose) — see docs/integration_findings.md.
# 1. zero-check
assert got_absmean > 1e-4, f"[HARD FAIL] wrapper output is zero (absmean={got_absmean:.2e})"
# 2. range-check (magnitude not off by >2x)
assert got_max > 0.5 * ref_max, f"[HARD FAIL] got max {got_max:.4f} << ref max {ref_max:.4f}"
assert got_max < 2.0 * ref_max, f"[HARD FAIL] got max {got_max:.4f} >> ref max {ref_max:.4f}"
# 3. tight allclose (tighter than the default v13bc bench tolerance — attention is approximate but not THAT approximate)
np.testing.assert_allclose(got, ref, rtol=1e-2, atol=2e-2)
print("[WRAPPER OUTPUT-ORDER] PASS — triple-check (zero / range / allclose) cleared")
