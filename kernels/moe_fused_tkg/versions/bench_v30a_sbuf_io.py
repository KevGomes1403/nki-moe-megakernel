"""
Benchmark script for kernel_v30a_sbuf_io.py vs kernel_v28f.py
Compares device_time_us, tensor_engine_pct, dma_active_pct, spill_bytes.

Usage:
    python bench_v30a_sbuf_io.py v28    # benchmark v28f only
    python bench_v30a_sbuf_io.py v30a   # benchmark v30a only
    python bench_v30a_sbuf_io.py both   # benchmark both (may have artifact ambiguity)

Per benchmarking-api.md, running two wrap_benchmark calls in one script can
cause NEFF artifact ambiguity. Use per-variant mode when in doubt.
"""
import os
import sys

# MUST be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v30a"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import wrap_benchmark

import torch
import torch_xla.core.xla_model as xm
import numpy as np

from kernel_v28f import qwen3_moe_fused_tkg
from kernel_v30a_sbuf_io import qwen3_moe_fused_tkg_sbuf_io

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

r28 = r30a = None

if mode in ("v28", "both"):
    print("Benchmarking v28f (reference)...")
    v28_kernel = wrap_benchmark(qwen3_moe_fused_tkg, warmup=5, iters=50)
    v28_kernel(inp, gamma, router_w, gate_up_w, down_w)
    r28 = v28_kernel.last_result
    print_result("v28f", r28)

if mode in ("v30a", "both"):
    print("\nBenchmarking v30a (sbuf_io column-major output)...")
    v30a_kernel = wrap_benchmark(qwen3_moe_fused_tkg_sbuf_io, warmup=5, iters=50)
    v30a_kernel(inp, gamma, router_w, gate_up_w, down_w)
    r30a = v30a_kernel.last_result
    print_result("v30a_sbuf_io", r30a)

if r28 and r30a:
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'v28f':>12} {'v30a':>12}")
    print("-" * 60)
    print(f"{'device_time_us':<25} {fmt(r28.device_time_us):>12} {fmt(r30a.device_time_us):>12}")
    print(f"{'tensor_engine_pct':<25} {fmt(r28.tensor_engine_pct):>11}% {fmt(r30a.tensor_engine_pct):>11}%")
    print(f"{'dma_active_pct':<25} {fmt(r28.dma_active_pct):>11}% {fmt(r30a.dma_active_pct):>11}%")
    print(f"{'spill_bytes':<25} {fmt(r28.spill_bytes, 'd'):>12} {fmt(r30a.spill_bytes, 'd'):>12}")
    print(f"{'mfu_estimated_pct':<25} {fmt(r28.prof.get('mfu_estimated_percent',0)):>11}% {fmt(r30a.prof.get('mfu_estimated_percent',0)):>11}%")
    print("=" * 60)
