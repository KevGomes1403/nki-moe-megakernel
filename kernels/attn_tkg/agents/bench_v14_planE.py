"""
Benchmark for v14_planE (fused mask computation, Plan E).
Output dir: _bench_out_v14e
"""
import os
import sys

# MUST be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v14e"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import wrap_benchmark

import torch
import torch_xla.core.xla_model as xm

from v14_planE import qwen3_attn_tkg_fused_oproj_v14e

device = xm.xla_device()

# --- Shape parameters ---
B = 1
H = 2048
d = 128
Hq_tp = 8
Hkv_tp = 1
S_prior = 640
pos = 320

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
position_ids = torch.tensor([[pos]], dtype=torch.int32).to(device)

inputs = (hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
          K_cache, V_cache, cos, sin, position_ids)

print("=" * 60)
print("BENCHMARKING v14_planE[2] (Fused Mask Computation)")
print("=" * 60)

v14e_wrapped = wrap_benchmark(qwen3_attn_tkg_fused_oproj_v14e[2], warmup=5, iters=50)
v14e_wrapped(*inputs)
xm.mark_step()

r = v14e_wrapped.last_result
if r:
    print(f"\nv14_planE (Plan E — Fused Mask):")
    print(f"  device_time_us       = {r.device_time_us:.2f}")
    print(f"  tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
    print(f"  dma_active_pct       = {r.dma_active_pct:.1f}%")
    print(f"  spill_bytes          = {r.spill_bytes}")
    hbm_read_kib  = r.prof.get('hbm_read_bytes', 0) / 1024
    hbm_write_kib = r.prof.get('hbm_write_bytes', 0) / 1024
    print(f"  hbm_read_KiB         = {hbm_read_kib:.1f}")
    print(f"  hbm_write_KiB        = {hbm_write_kib:.1f}")

    v13bc_us = 59.47
    v12e_us  = 65.33
    pct_vs_v13bc = (r.device_time_us - v13bc_us) / v13bc_us * 100
    pct_vs_v12e  = (r.device_time_us - v12e_us)  / v12e_us  * 100
    print(f"\n  vs v13_BC baseline ({v13bc_us} μs): {r.device_time_us:.2f} μs  ({pct_vs_v13bc:+.1f}%)")
    print(f"  vs v12e baseline  ({v12e_us} μs): {r.device_time_us:.2f} μs  ({pct_vs_v12e:+.1f}%)")
else:
    print("v14_planE benchmark result not available (NTFF not captured)")
