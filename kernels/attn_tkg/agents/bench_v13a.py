"""
Correctness check + benchmark: v13a (LNC=2) vs v12d (LNC=1).

Per task: verify ap() with dynamic core_id offset compiles on HBM dst,
check numerical correctness, then benchmark both kernels.
"""
import os
import sys

# MUST be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output_v13a_fix"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import wrap_benchmark

import torch
import torch_xla.core.xla_model as xm

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

inputs = (hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
          K_cache, V_cache, cos, sin, position_ids)

# =========================================================================
# CORRECTNESS CHECK
# =========================================================================
print("=" * 60)
print("CORRECTNESS CHECK: v13a[2] vs v12d[1]")
print("=" * 60)

print("Running v12d[1]...")
out12d = qwen3_attn_tkg_fused_oproj_v12d[1](*inputs)
xm.mark_step()

print("Running v13a[2]...")
try:
    out13a = qwen3_attn_tkg_fused_oproj_v13a[2](*inputs)
    xm.mark_step()
except Exception as e:
    print(f"COMPILATION/RUNTIME FAILURE: {e}")
    sys.exit(1)

output_12d, k_rope_12d, v_out_12d = out12d
output_13a, k_rope_13a, v_out_13a = out13a

output_12d_cpu = output_12d.cpu().float()
output_13a_cpu = output_13a.cpu().float()
k_rope_12d_cpu = k_rope_12d.cpu().float()
k_rope_13a_cpu = k_rope_13a.cpu().float()
v_out_12d_cpu = v_out_12d.cpu().float()
v_out_13a_cpu = v_out_13a.cpu().float()

rtol, atol = 1e-2, 1e-2

out_ok = torch.allclose(output_12d_cpu, output_13a_cpu, rtol=rtol, atol=atol)
k_ok = torch.allclose(k_rope_12d_cpu, k_rope_13a_cpu, rtol=rtol, atol=atol)
v_ok = torch.allclose(v_out_12d_cpu, v_out_13a_cpu, rtol=rtol, atol=atol)

out_max_diff = (output_12d_cpu - output_13a_cpu).abs().max().item()
k_max_diff = (k_rope_12d_cpu - k_rope_13a_cpu).abs().max().item()
v_max_diff = (v_out_12d_cpu - v_out_13a_cpu).abs().max().item()

print(f"  output  allclose={out_ok}   max_diff={out_max_diff:.2e}")
print(f"  k_rope  allclose={k_ok}   max_diff={k_max_diff:.2e}")
print(f"  v_out   allclose={v_ok}   max_diff={v_max_diff:.2e}")

overall_max = max(out_max_diff, k_max_diff, v_max_diff)
if out_ok and k_ok and v_ok:
    print(f"CORRECTNESS: PASS  max_diff={overall_max:.2e}")
else:
    print(f"CORRECTNESS: FAIL  max_diff={overall_max:.2e}")
    if not out_ok:
        print(f"  output FAILED: max_diff={out_max_diff:.2e}")
    if not k_ok:
        print(f"  k_rope FAILED: max_diff={k_max_diff:.2e}")
    if not v_ok:
        print(f"  v_out  FAILED: max_diff={v_max_diff:.2e}")
    sys.exit(1)

# =========================================================================
# BENCHMARKING — run per-kernel scripts separately to avoid NTFF artifact
# ambiguity. Per the benchmarking reference, each wrap_benchmark call tracks
# its own NEFF via snapshot diffing. Run each kernel sequentially here.
# =========================================================================
print("\n" + "=" * 60)
print("BENCHMARKING v12d[1] ...")
print("=" * 60)

v12d_wrapped = wrap_benchmark(qwen3_attn_tkg_fused_oproj_v12d[1], warmup=5, iters=50)
v12d_wrapped(*inputs)
xm.mark_step()

r12d = v12d_wrapped.last_result
if r12d:
    print(f"v12d (LNC=1):")
    print(f"  device_time_us       = {r12d.device_time_us:.2f}")
    print(f"  tensor_engine_pct    = {r12d.tensor_engine_pct:.1f}%")
    print(f"  dma_active_pct       = {r12d.dma_active_pct:.1f}%")
    print(f"  spill_bytes          = {r12d.spill_bytes}")
    print(f"  hbm_read_KiB         = {r12d.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"  hbm_write_KiB        = {r12d.prof.get('hbm_write_bytes', 0)/1024:.1f}")
else:
    print("v12d benchmark result not available (NTFF not captured)")

print("\n" + "=" * 60)
print("BENCHMARKING v13a[2] ...")
print("=" * 60)

v13a_wrapped = wrap_benchmark(qwen3_attn_tkg_fused_oproj_v13a[2], warmup=5, iters=50)
v13a_wrapped(*inputs)
xm.mark_step()

r13a = v13a_wrapped.last_result
if r13a:
    print(f"v13a (LNC=2):")
    print(f"  device_time_us       = {r13a.device_time_us:.2f}")
    print(f"  tensor_engine_pct    = {r13a.tensor_engine_pct:.1f}%")
    print(f"  dma_active_pct       = {r13a.dma_active_pct:.1f}%")
    print(f"  spill_bytes          = {r13a.spill_bytes}")
    print(f"  hbm_read_KiB         = {r13a.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"  hbm_write_KiB        = {r13a.prof.get('hbm_write_bytes', 0)/1024:.1f}")
else:
    print("v13a benchmark result not available (NTFF not captured)")

# --- Speedup ---
if r12d and r13a:
    speedup = r12d.device_time_us / r13a.device_time_us
    print(f"\nSpeedup v13a vs v12d: {speedup:.2f}x")
