"""
Benchmark script for kernel_v32a.py vs kernel_v30b_gu_shard.py
Compares device_time_us, tensor_engine_pct, dma_active_pct, spill_bytes, mfu, hbm traffic.

Usage:
    python bench_v32a.py v30b   # benchmark v30b only
    python bench_v32a.py v32a   # benchmark v32a only
    python bench_v32a.py both   # benchmark both (may have artifact ambiguity)
"""
import os
import sys

# MUST be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v32a"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import wrap_benchmark

import torch
import torch_xla.core.xla_model as xm
import numpy as np

from kernel_v30b_gu_shard import qwen3_moe_fused_tkg_sbuf_io as kernel_v30b
from kernel_v32a import qwen3_moe_fused_tkg_sbuf_io as kernel_v32a

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
    print(f"  tensor_engine_pct = {fmt(r.prof.get('tensor_engine_active_time_percent', 0))}%")
    print(f"  dma_active_pct    = {fmt(r.prof.get('dma_active_time_percent', 0))}%")
    print(f"  spill_bytes       = {fmt(r.spill_bytes, 'd')}")
    print(f"  mfu_estimated     = {fmt(r.prof.get('mfu_estimated_percent', 0))}%")
    print(f"  hbm_read_KiB      = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"  hbm_write_KiB     = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")

mode = sys.argv[1] if len(sys.argv) > 1 else "both"

r30b = r32a = None

if mode in ("v30b", "both"):
    print("Benchmarking v30b (baseline, H-sharded gate/up)...")
    v30b_wrapped = wrap_benchmark(kernel_v30b[2], warmup=5, iters=50)
    v30b_wrapped(inp, gamma, router_w, gate_up_w, down_w)
    r30b = v30b_wrapped.last_result
    print_result("v30b_gu_shard (baseline)", r30b)

if mode in ("v32a", "both"):
    print("\nBenchmarking v32a (Plan A: section rotation pipelined K=8)...")
    v32a_wrapped = wrap_benchmark(kernel_v32a[2], warmup=5, iters=50)
    v32a_wrapped(inp, gamma, router_w, gate_up_w, down_w)
    r32a = v32a_wrapped.last_result
    print_result("v32a_section_rotation", r32a)

if r30b and r32a:
    te30b = r30b.prof.get('tensor_engine_active_time_percent', 0)
    te32a = r32a.prof.get('tensor_engine_active_time_percent', 0)
    da30b = r30b.prof.get('dma_active_time_percent', 0)
    da32a = r32a.prof.get('dma_active_time_percent', 0)
    mfu30b = r30b.prof.get('mfu_estimated_percent', 0)
    mfu32a = r32a.prof.get('mfu_estimated_percent', 0)

    def delta(a, b):
        if a is None or b is None:
            return "N/A"
        return f"{b - a:+.2f}"

    print("\n" + "=" * 72)
    print(f"{'Metric':<28} {'v30b (baseline)':>18} {'v32a':>15} {'delta':>8}")
    print("-" * 72)
    print(f"{'device_time_us':<28} {fmt(r30b.device_time_us):>18} {fmt(r32a.device_time_us):>15} {delta(r30b.device_time_us, r32a.device_time_us):>8}")
    print(f"{'tensor_engine_pct':<28} {fmt(te30b):>17}% {fmt(te32a):>14}% {delta(te30b, te32a):>7}%")
    print(f"{'dma_active_pct':<28} {fmt(da30b):>17}% {fmt(da32a):>14}% {delta(da30b, da32a):>7}%")
    print(f"{'spill_bytes':<28} {fmt(r30b.spill_bytes, 'd'):>18} {fmt(r32a.spill_bytes, 'd'):>15} {r32a.spill_bytes - r30b.spill_bytes:>+8}")
    print(f"{'mfu_estimated_pct':<28} {fmt(mfu30b):>17}% {fmt(mfu32a):>14}% {delta(mfu30b, mfu32a):>7}%")
    print(f"{'hbm_read_KiB':<28} {r30b.prof.get('hbm_read_bytes',0)/1024:>17.1f}  {r32a.prof.get('hbm_read_bytes',0)/1024:>14.1f}  {(r32a.prof.get('hbm_read_bytes',0)-r30b.prof.get('hbm_read_bytes',0))/1024:>+7.1f}")
    print("=" * 72)
