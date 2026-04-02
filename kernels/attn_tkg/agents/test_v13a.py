"""
Correctness test for v13a (LNC=2 O-proj sharding) vs v12d (LNC=1 baseline).
"""
import os
import sys

# MUST be before any neuron/torch/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import torch
import torch_xla.core.xla_model as xm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v12d import qwen3_attn_tkg_fused_oproj_v12d
from v13a import qwen3_attn_tkg_fused_oproj_v13a

device = xm.xla_device()

# --- Shape parameters ---
B = 1
H = 2048
d = 128
Hq_tp = 8
Hkv_tp = 1
S_prior = 640

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
    position_ids = torch.tensor([[100]], dtype=torch.int32).to(device)
    return (hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
            K_cache, V_cache, cos, sin, position_ids)

inputs = make_inputs()

print("Running v12d[1] reference...")
out12d, k12d, v12d = qwen3_attn_tkg_fused_oproj_v12d[1](*inputs)
xm.mark_step()

print("Running v13a[2] under test...")
out13a, k13a, v13a = qwen3_attn_tkg_fused_oproj_v13a[2](*inputs)
xm.mark_step()

# Move to CPU for comparison
out12d_cpu = out12d.cpu().float()
out13a_cpu = out13a.cpu().float()
k12d_cpu = k12d.cpu().float()
k13a_cpu = k13a.cpu().float()
v12d_cpu = v12d.cpu().float()
v13a_cpu = v13a.cpu().float()

print(f"\nout shape: v12d={out12d_cpu.shape}, v13a={out13a_cpu.shape}")
print(f"k_rope shape: v12d={k12d_cpu.shape}, v13a={k13a_cpu.shape}")
print(f"v_out shape:  v12d={v12d_cpu.shape}, v13a={v13a_cpu.shape}")

# Compute max diffs
out_maxdiff = (out12d_cpu - out13a_cpu).abs().max().item()
k_maxdiff = (k12d_cpu - k13a_cpu).abs().max().item()
v_maxdiff = (v12d_cpu - v13a_cpu).abs().max().item()

print(f"\nMax abs diff — output: {out_maxdiff:.3e}, k_rope_out: {k_maxdiff:.3e}, v_out: {v_maxdiff:.3e}")

rtol, atol = 1e-2, 1e-2
pass_out = torch.allclose(out12d_cpu, out13a_cpu, rtol=rtol, atol=atol)
pass_k   = torch.allclose(k12d_cpu,   k13a_cpu,   rtol=rtol, atol=atol)
pass_v   = torch.allclose(v12d_cpu,   v13a_cpu,   rtol=rtol, atol=atol)

print(f"\noutput allclose: {pass_out}")
print(f"k_rope allclose: {pass_k}")
print(f"v_out  allclose: {pass_v}")

if pass_out and pass_k and pass_v:
    print("\nCORRECTNESS: PASS")
else:
    print("\nCORRECTNESS: FAIL")
    if not pass_out:
        # Print per-position breakdown to diagnose shard boundary issues
        out12d_flat = out12d_cpu.reshape(-1)
        out13a_flat = out13a_cpu.reshape(-1)
        diff_flat = (out12d_flat - out13a_flat).abs()
        top10 = diff_flat.topk(10)
        print(f"  Top-10 output diffs at indices: {top10.indices.tolist()}")
        print(f"  Values: {top10.values.tolist()}")
        # Check first vs second half
        H_wo = 2048
        first_half_max = (out12d_cpu[..., :H_wo//2] - out13a_cpu[..., :H_wo//2]).abs().max().item()
        second_half_max = (out12d_cpu[..., H_wo//2:] - out13a_cpu[..., H_wo//2:]).abs().max().item()
        print(f"  output first-half max diff: {first_half_max:.3e}")
        print(f"  output second-half max diff: {second_half_max:.3e}")
    sys.exit(1)
