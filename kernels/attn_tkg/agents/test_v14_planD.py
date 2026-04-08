"""
Correctness test for v14_planD (LNC=2 Wo sharding) vs v12e (LNC=1 baseline).
"""
import os
import sys

# MUST be before any neuron/torch/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import torch
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v12e import qwen3_attn_tkg_fused_oproj_v12e
from v14_planD import qwen3_attn_tkg_fused_oproj_v14d

device = xm.xla_device()

# --- Shape parameters (same as all other tests) ---
B = 1
H = 2048
d = 128
Hq_tp = 8
Hkv_tp = 1
S_prior = 640
Hq_out = 1024
pos = 320

torch.manual_seed(42)

def make_inputs():
    hidden_states = torch.randn(B, 1, H, dtype=torch.bfloat16).to(device)
    Wq = torch.randn(Hq_tp * d, H, dtype=torch.bfloat16).to(device)
    Wk = torch.randn(Hkv_tp * d, H, dtype=torch.bfloat16).to(device)
    Wv = torch.randn(Hkv_tp * d, H, dtype=torch.bfloat16).to(device)
    Wo = torch.randn(Hq_tp * d, H, dtype=torch.bfloat16).to(device)
    q_norm_weight = torch.randn(d, dtype=torch.bfloat16).to(device)
    k_norm_weight = torch.randn(d, dtype=torch.bfloat16).to(device)
    K_cache = torch.randn(B, 1, S_prior, d, dtype=torch.bfloat16).to(device)
    V_cache = torch.randn(B, 1, S_prior, d, dtype=torch.bfloat16).to(device)
    cos = torch.randn(B, d, dtype=torch.bfloat16).to(device)
    sin = torch.randn(B, d, dtype=torch.bfloat16).to(device)
    position_ids = torch.tensor([[pos]], dtype=torch.int32).to(device)
    return (hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
            K_cache, V_cache, cos, sin, position_ids)

inputs = make_inputs()

print("Running v12e[1] reference...")
out12e, k12e, v12e_out = qwen3_attn_tkg_fused_oproj_v12e[1](*inputs)
xm.mark_step()

print("Running v14d[2] under test...")
try:
    out14d, k14d, v14d_out = qwen3_attn_tkg_fused_oproj_v14d[2](*inputs)
    xm.mark_step()
except Exception as e:
    print(f"COMPILATION/RUNTIME FAILURE: {e}")
    sys.exit(1)

# Move to CPU for comparison
out12e_cpu = out12e.cpu().float()
out14d_cpu = out14d.cpu().float()
k12e_cpu = k12e.cpu().float()
k14d_cpu = k14d.cpu().float()
v12e_cpu = v12e_out.cpu().float()
v14d_cpu = v14d_out.cpu().float()

print(f"\nout shape: v12e={out12e_cpu.shape}, v14d={out14d_cpu.shape}")

# Compute max diffs
out_maxdiff = (out12e_cpu - out14d_cpu).abs().max().item()
k_maxdiff = (k12e_cpu - k14d_cpu).abs().max().item()
v_maxdiff = (v12e_cpu - v14d_cpu).abs().max().item()

print(f"\nMax abs diff — output: {out_maxdiff:.3e}, k_rope_out: {k_maxdiff:.3e}, v_out: {v_maxdiff:.3e}")

rtol, atol = 1e-3, 1e-3
pass_out = torch.allclose(out12e_cpu, out14d_cpu, rtol=rtol, atol=atol)
pass_k   = torch.allclose(k12e_cpu,   k14d_cpu,   rtol=rtol, atol=atol)
pass_v   = torch.allclose(v12e_cpu,   v14d_cpu,   rtol=rtol, atol=atol)

print(f"\noutput allclose (rtol={rtol}, atol={atol}): {pass_out}")
print(f"k_rope allclose: {pass_k}")
print(f"v_out  allclose: {pass_v}")

overall_max = max(out_maxdiff, k_maxdiff, v_maxdiff)
if pass_out and pass_k and pass_v:
    print(f"\nCORRECTNESS: PASS  max_diff={overall_max:.3e}")
else:
    print(f"\nCORRECTNESS: FAIL  max_diff={overall_max:.3e}")
    if not pass_out:
        H_wo = 2048
        first_half_max = (out12e_cpu[..., :H_wo//2] - out14d_cpu[..., :H_wo//2]).abs().max().item()
        second_half_max = (out12e_cpu[..., H_wo//2:] - out14d_cpu[..., H_wo//2:]).abs().max().item()
        print(f"  output first-half max diff:  {first_half_max:.3e}")
        print(f"  output second-half max diff: {second_half_max:.3e}")
    sys.exit(1)
