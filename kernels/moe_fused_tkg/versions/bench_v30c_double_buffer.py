"""
Benchmark script for kernel_v30c_double_buffer.py vs kernel_v30b_gu_shard.py
Compares device_time_us, tensor_engine_pct, dma_active_pct, spill_bytes, mfu, hbm traffic.

Usage:
    python bench_v30c_double_buffer.py v30b   # benchmark v30b only
    python bench_v30c_double_buffer.py v30c   # benchmark v30c only
    python bench_v30c_double_buffer.py both   # benchmark both (default)
"""
import os
import sys

# MUST be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v30c"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import wrap_benchmark

import torch
import torch_xla.core.xla_model as xm
import numpy as np

from kernel_v30b_gu_shard import qwen3_moe_fused_tkg_sbuf_io as kernel_v30b
from kernel_v30c_double_buffer import qwen3_moe_fused_tkg_sbuf_io as kernel_v30c

# Fixed dims
_H = 2048
_E = 128
_I = 192

device = xm.xla_device()
rng = np.random.default_rng(0)

def make_tensor(shape, dtype=torch.bfloat16, scale=0.1):
    arr = rng.standard_normal(shape).astype(np.float32) * scale
    return torch.tensor(arr, dtype=dtype).to(device)

B = 1
inp       = make_tensor((B, 1, _H))
gamma     = make_tensor((1, _H))
router_w  = make_tensor((_H, _E))
gate_up_w = make_tensor((_E, _H, 2 * _I))
down_w    = make_tensor((_E, _I, _H))

def fmt(val, fmt_str=".2f"):
    if val is None:
        return "N/A"
    return format(val, fmt_str)

def print_result(label, r):
    if r is None:
        print(f"{label}: no profiling data captured")
        return
    print(f"\n--- {label} ---")
    print(f"  device_time_us    = {fmt(r.device_time_us)}")
    print(f"  tensor_engine_pct = {fmt(r.tensor_engine_pct)}%")
    print(f"  dma_active_pct    = {fmt(r.dma_active_pct)}%")
    print(f"  spill_bytes       = {fmt(r.spill_bytes, 'd')}")
    print(f"  mfu_estimated     = {fmt(r.prof.get('mfu_estimated_percent', 0))}%")
    print(f"  hbm_read_KiB      = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"  hbm_write_KiB     = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")

mode = sys.argv[1] if len(sys.argv) > 1 else "both"

r30b = r30c = None

if mode in ("v30b", "both"):
    print("Benchmarking v30b (H-sharded gate/up reference)...")
    v30b_wrapped = wrap_benchmark(kernel_v30b[2], warmup=5, iters=50)
    v30b_wrapped(inp, gamma, router_w, gate_up_w, down_w)
    r30b = v30b_wrapped.last_result
    print_result("v30b_gu_shard", r30b)

if mode in ("v30c", "both"):
    print("\nBenchmarking v30c (double-buffered)...")
    v30c_wrapped = wrap_benchmark(kernel_v30c[2], warmup=5, iters=50)
    v30c_wrapped(inp, gamma, router_w, gate_up_w, down_w)
    r30c = v30c_wrapped.last_result
    print_result("v30c_double_buffer", r30c)

if r30b and r30c:
    print("\n" + "=" * 65)
    print(f"{'Metric':<28} {'v30b':>15} {'v30c':>15}")
    print("-" * 65)
    print(f"{'device_time_us':<28} {fmt(r30b.device_time_us):>15} {fmt(r30c.device_time_us):>15}")
    print(f"{'tensor_engine_pct':<28} {fmt(r30b.tensor_engine_pct):>14}% {fmt(r30c.tensor_engine_pct):>14}%")
    print(f"{'dma_active_pct':<28} {fmt(r30b.dma_active_pct):>14}% {fmt(r30c.dma_active_pct):>14}%")
    print(f"{'spill_bytes':<28} {fmt(r30b.spill_bytes, 'd'):>15} {fmt(r30c.spill_bytes, 'd'):>15}")
    print(f"{'mfu_estimated_pct':<28} {fmt(r30b.prof.get('mfu_estimated_percent',0)):>14}% {fmt(r30c.prof.get('mfu_estimated_percent',0)):>14}%")
    print(f"{'hbm_read_KiB':<28} {r30b.prof.get('hbm_read_bytes',0)/1024:>14.1f}  {r30c.prof.get('hbm_read_bytes',0)/1024:>14.1f}")
    print(f"{'hbm_write_KiB':<28} {r30b.prof.get('hbm_write_bytes',0)/1024:>14.1f}  {r30c.prof.get('hbm_write_bytes',0)/1024:>14.1f}")
    print("=" * 65)
