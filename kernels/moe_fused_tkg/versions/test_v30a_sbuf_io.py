"""
Test harness for kernel_v30a_sbuf_io.py
Compares qwen3_moe_fused_tkg_sbuf_io (v30a) against qwen3_moe_fused_tkg (v28f reference).
"""
import sys
import os

# Must be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import numpy as np
import torch
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_v28f import qwen3_moe_fused_tkg, run as run_v28f
from kernel_v30a_sbuf_io import qwen3_moe_fused_tkg_sbuf_io, run as run_v30a

# Fixed dims matching kernel contracts
_H = 2048
_E = 128
_I = 192
_K = 8

rng = np.random.default_rng(42)
B = 1

device = xm.xla_device()

def make_tensor(shape, dtype=torch.bfloat16, scale=0.1):
    arr = rng.standard_normal(shape).astype(np.float32) * scale
    return torch.tensor(arr, dtype=dtype).to(device)

# Generate inputs matching run() contract
inp       = make_tensor((B, 1, _H))                     # [B, 1, H] bf16
gamma     = make_tensor((1, _H))                        # [1, H] bf16
router_w  = make_tensor((_H, _E))                       # [H, E] bf16
gate_up_w = make_tensor((_E, _H, 2 * _I))               # [E, H, 2*I] bf16
down_w    = make_tensor((_E, _I, _H))                   # [E, I, H] bf16

print("Running v28f reference kernel...")
out_v28 = run_v28f(inp, gamma, router_w, gate_up_w, down_w)
xm.mark_step()

print("Running v30a sbuf_io kernel...")
out_v30a = run_v30a(inp, gamma, router_w, gate_up_w, down_w)
xm.mark_step()

# Transfer to CPU for comparison
out_v28_cpu = out_v28.cpu().float().numpy()
out_v30a_cpu = out_v30a.cpu().float().numpy()

max_diff = np.max(np.abs(out_v28_cpu - out_v30a_cpu))
mean_diff = np.mean(np.abs(out_v28_cpu - out_v30a_cpu))

print(f"\n--- Comparison Results ---")
print(f"v28f output shape : {out_v28_cpu.shape}")
print(f"v30a output shape : {out_v30a_cpu.shape}")
print(f"max_diff          : {max_diff:.6f}")
print(f"mean_diff         : {mean_diff:.6f}")

try:
    np.testing.assert_allclose(out_v28_cpu, out_v30a_cpu, rtol=1e-2, atol=1e-2)
    print("\nPASS — outputs match within rtol=1e-2, atol=1e-2")
except AssertionError as e:
    print(f"\nFAIL — {e}")
    sys.exit(1)
