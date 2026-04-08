"""
Corrected correctness test for v6a vs v5.

Root cause of NaN in original test_v6a_e4m3.py:
  Random int8 weights were generated with rng.integers(-127, 127), which includes
  -1 (0xFF in two's complement). When reinterpreted as fp8_e4m3fn, 0xFF = -NaN.
  This causes ~400K NaN values in the weight tensors, which propagate to all outputs.

Fix: generate fp32 weights and quantize them properly to fp8_e4m3fn with derived
     per-output-neuron scales, then reinterpret as int8 for the kernel interface.
     This ensures no NaN-valued fp8 entries exist in the weight tensors.
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from kernels.moe_fused_tkg.quantized.v5 import run as run_v5
from kernels.moe_fused_tkg.quantized.v6a import run as run_v6a

torch.manual_seed(42)
device = xm.xla_device()

B, H, E, K, I = 1, 2048, 128, 8, 192
GU_FLAT = 2 * I    # 384
H_SHARD = 512      # TP shard
EPS = 1e-6
scale = 0.1        # small but non-zero activations

# ── Activations ─────────────────────────────────────────────────────────────
inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16)
gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16)

# ── PyTorch reference: verify routing is NaN-free ────────────────────────────
x = inp.reshape(B, H).to(torch.float32)
rms = (x.pow(2).mean(dim=-1, keepdim=True) + EPS).rsqrt()
x_norm = (x * rms * gamma.to(torch.float32)).to(torch.bfloat16)
logits = x_norm.to(torch.float32) @ router_w.to(torch.float32)
probs  = torch.softmax(logits, dim=-1)
top_vals, top_idx = torch.topk(probs, K, dim=-1)
norm_weights = top_vals / top_vals.sum(dim=-1, keepdim=True)
assert not x_norm.isnan().any(),     "RMSNorm produced NaN"
assert not probs.isnan().any(),      "Router softmax produced NaN"
assert not norm_weights.isnan().any(), "Norm weights NaN"
print(f"Router OK: top_idx={top_idx[0].tolist()}")
print(f"           norm_weights={[f'{v:.4f}' for v in norm_weights[0].tolist()]}")

# ── Generate FP8 weights with proper derived scales ──────────────────────────
# gate_up_w: [E, H, GU_FLAT=384], scale per output-neuron (dim 1 = H = contracting)
gate_up_w_f32 = torch.randn(E, H, GU_FLAT) * scale
gate_up_scales = gate_up_w_f32.abs().amax(dim=1).clamp(min=1e-12) / 240.0  # [E, GU]
gate_up_scales_bcast = gate_up_scales.unsqueeze(1).expand(E, H, GU_FLAT)
gate_up_w_fp8 = (gate_up_w_f32 / gate_up_scales_bcast).clamp(-240, 240).to(torch.float8_e4m3fn)

# down_w: [E, I=192, H=2048], scale per output-neuron (only H_SHARD=512 needed)
down_w_f32 = torch.randn(E, I, H) * scale
down_scales = down_w_f32[:, :, 0:H_SHARD].abs().amax(dim=1).clamp(min=1e-12) / 240.0  # [E, H_SHARD]
down_scales_bcast = down_scales.unsqueeze(1).expand(E, I, H_SHARD)
down_w_fp8_shard = (down_w_f32[:, :, 0:H_SHARD] / down_scales_bcast).clamp(-240, 240).to(torch.float8_e4m3fn)
down_w_fp8 = torch.cat([
    down_w_fp8_shard,
    torch.zeros(E, I, H - H_SHARD, dtype=torch.float8_e4m3fn)
], dim=2)

# Verify no NaN in fp8 weights
gate_up_w_bf16 = gate_up_w_fp8.to(torch.bfloat16)
down_w_bf16    = down_w_fp8.to(torch.bfloat16)
assert not gate_up_w_bf16.isnan().any(), "gate_up_w has NaN after proper quantization"
assert not down_w_bf16.isnan().any(),    "down_w has NaN after proper quantization"
print(f"Weight quantization OK: gate_up [{gate_up_w_bf16.float().min():.3f}, {gate_up_w_bf16.float().max():.3f}]")
print(f"                        down    [{down_w_bf16.float().min():.3f}, {down_w_bf16.float().max():.3f}]")

# ── PyTorch reference (mirrors v5 exactly: fp8-stationary = bf16-cast + matmul) ─
ref_output = torch.zeros(B, H_SHARD, dtype=torch.float32)
T = B
for t in range(T):
    for ki in range(K):
        e = top_idx[t, ki].item()
        w = norm_weights[t, ki].item()
        gu_out = (x_norm[t:t+1].to(torch.float32) @ gate_up_w_bf16[e].to(torch.float32)).squeeze(0)
        gate_out = gu_out[:I] * gate_up_scales[e, :I]
        up_out   = gu_out[I:] * gate_up_scales[e, I:]
        inter = torch.nn.functional.silu(gate_out) * up_out
        d_out = (inter.unsqueeze(0) @ down_w_bf16[e, :, 0:H_SHARD].to(torch.float32)).squeeze(0)
        ref_output[t] += w * (d_out * down_scales[e])

assert not ref_output.isnan().any(), "PyTorch reference produced NaN — inputs still bad"
print(f"PyTorch reference OK: output [{ref_output.min():.6f}, {ref_output.max():.6f}]")

# ── Move to device ────────────────────────────────────────────────────────────
gate_up_w_int8   = gate_up_w_fp8.view(torch.int8).to(device)
gate_up_scales_d = gate_up_scales.to(device)
down_w_int8      = down_w_fp8.view(torch.int8).to(device)
down_scales_d    = down_scales.to(device)
inp_d            = inp.to(device)
gamma_d          = gamma.to(device)
router_w_d       = router_w.to(device)
xm.mark_step()

# ── Run v5 (reference kernel) ─────────────────────────────────────────────────
v5_out = run_v5(inp_d, gamma_d, router_w_d, gate_up_w_int8, gate_up_scales_d, down_w_int8, down_scales_d)
xm.mark_step()
v5_np = v5_out.cpu().float().numpy()[:, 0:H_SHARD]

assert not np.isnan(v5_np).any(), "v5 kernel produced NaN — still broken"
print(f"v5 output OK: [{v5_np.min():.6f}, {v5_np.max():.6f}]")

# ── Run v6a ────────────────────────────────────────────────────────────────────
v6a_out = run_v6a(inp_d, gamma_d, router_w_d, gate_up_w_int8, gate_up_scales_d, down_w_int8, down_scales_d)
xm.mark_step()
v6a_np = v6a_out.cpu().float().numpy()[:, 0:H_SHARD]

nan_in_v6a = np.isnan(v6a_np).any()
print(f"v6a output: nan={nan_in_v6a}  [{np.nanmin(v6a_np):.6f}, {np.nanmax(v6a_np):.6f}]")

# ── Compare v6a vs v5 ─────────────────────────────────────────────────────────
max_diff  = np.abs(v6a_np - v5_np).max()
mean_diff = np.abs(v6a_np - v5_np).mean()
print(f"\nv6a vs v5:  max_diff={max_diff:.4e}  mean_diff={mean_diff:.4e}")

# FP8 quantization tolerance: ~0.5 relative, but absolute can be larger for
# larger activation values. Use the same tolerance as test_v5.py.
ATOL, RTOL = 0.5, 0.5
try:
    np.testing.assert_allclose(v6a_np, v5_np, rtol=RTOL, atol=ATOL)
    print("CORRECTNESS PASS")
except AssertionError as err:
    print(f"CORRECTNESS FAIL: {err}")
    # Additional diagnostics
    diff = np.abs(v6a_np - v5_np)
    bad = diff > ATOL
    print(f"  Elements exceeding atol={ATOL}: {bad.sum()} / {bad.size}")
    if bad.any():
        worst_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  Worst element: idx={worst_idx}  v5={v5_np[worst_idx]:.6f}  v6a={v6a_np[worst_idx]:.6f}  diff={diff[worst_idx]:.6f}")
