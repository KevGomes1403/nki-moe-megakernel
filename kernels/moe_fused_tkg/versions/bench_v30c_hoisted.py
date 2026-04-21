"""
Benchmark script for kernel_v30c_hoisted.py vs kernel_v30b_gu_shard.py.

Two modes for v30c:
  Mode 1 — back-compat: both gamma_sb_ready and router_w_wide_sb = None (identical to v30b)
  Mode 2 — hoisted:     both kwargs supplied (gamma and router_w pre-loaded in SBUF)

NOTE on Mode 2 single-call harness:
  The prefetch of gamma / router_w happens inside the same JIT function as the MoE
  computation. The compiler sees the full HBM traffic in one graph, so overall HBM
  reads may look unchanged compared to back-compat mode. The real savings materialize
  in the multi-layer megakernel where these loads are hoisted outside the layer loop
  and amortised across N layers. The benchmark here measures kernel-internal device
  time only (no multi-layer loop overhead).

Usage:
    python bench_v30c_hoisted.py v30b        # benchmark v30b only
    python bench_v30c_hoisted.py backcompat  # benchmark v30c back-compat only
    python bench_v30c_hoisted.py hoisted     # benchmark v30c hoisted only
    python bench_v30c_hoisted.py all         # benchmark all three (default)
"""
import os
import sys

# MUST be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v30c_hoisted"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import wrap_benchmark

import torch
import torch_xla.core.xla_model as xm
import numpy as np

from kernel_v30b_gu_shard import qwen3_moe_fused_tkg_sbuf_io as kernel_v30b
from kernel_v30c_hoisted import qwen3_moe_fused_tkg_sbuf_io as kernel_v30c_backcompat
from kernel_v30c_hoisted import qwen3_moe_fused_tkg_sbuf_io_hoisted as kernel_v30c_hoisted_fn

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

mode = sys.argv[1] if len(sys.argv) > 1 else "all"

r30b = r30c_bc = r30c_h = None

if mode in ("v30b", "all"):
    print("Benchmarking v30b (gu_shard reference)...")
    v30b_wrapped = wrap_benchmark(kernel_v30b[2], warmup=5, iters=50)
    v30b_wrapped(inp, gamma, router_w, gate_up_w, down_w)
    r30b = v30b_wrapped.last_result
    print_result("v30b_gu_shard", r30b)

if mode in ("backcompat", "all"):
    print("\nBenchmarking v30c back-compat (both kwargs=None — should match v30b)...")
    v30c_bc_wrapped = wrap_benchmark(kernel_v30c_backcompat[2], warmup=5, iters=50)
    v30c_bc_wrapped(inp, gamma, router_w, gate_up_w, down_w)
    r30c_bc = v30c_bc_wrapped.last_result
    print_result("v30c_backcompat", r30c_bc)

if mode in ("hoisted", "all"):
    print("\nBenchmarking v30c hoisted (both gamma + router_w pre-loaded)...")
    # hoisted_gamma=True, hoisted_router_w=True
    v30c_h_wrapped = wrap_benchmark(
        lambda i, g, rw, gu, d: kernel_v30c_hoisted_fn[2](i, g, rw, gu, d, True, True),
        warmup=5, iters=50
    )
    v30c_h_wrapped(inp, gamma, router_w, gate_up_w, down_w)
    r30c_h = v30c_h_wrapped.last_result
    print_result("v30c_hoisted", r30c_h)

# Summary table
print("\n" + "=" * 72)
print(f"{'Metric':<28} {'v30b':>13} {'v30c_bc':>13} {'v30c_hoisted':>13}")
print("-" * 72)
rows = [
    ("device_time_us",        "device_time_us"),
    ("tensor_engine_pct",     "tensor_engine_pct"),
    ("dma_active_pct",        "dma_active_pct"),
    ("spill_bytes",           "spill_bytes"),
]

def _get(r, key):
    if r is None:
        return "N/A"
    if key == "device_time_us":
        return fmt(r.device_time_us)
    if key == "tensor_engine_pct":
        return f"{fmt(r.tensor_engine_pct)}%"
    if key == "dma_active_pct":
        return f"{fmt(r.dma_active_pct)}%"
    if key == "spill_bytes":
        return fmt(r.spill_bytes, 'd')
    return "N/A"

for label, key in rows:
    print(f"{label:<28} {_get(r30b, key):>13} {_get(r30c_bc, key):>13} {_get(r30c_h, key):>13}")

def _hbm(r, k):
    if r is None:
        return "N/A"
    return f"{r.prof.get(k, 0)/1024:.1f}"

print(f"{'hbm_read_KiB':<28} {_hbm(r30b,'hbm_read_bytes'):>13} {_hbm(r30c_bc,'hbm_read_bytes'):>13} {_hbm(r30c_h,'hbm_read_bytes'):>13}")
print(f"{'hbm_write_KiB':<28} {_hbm(r30b,'hbm_write_bytes'):>13} {_hbm(r30c_bc,'hbm_write_bytes'):>13} {_hbm(r30c_h,'hbm_write_bytes'):>13}")
print("=" * 72)

# Regression check: v30c back-compat vs v30b
if r30b and r30c_bc and r30b.device_time_us and r30c_bc.device_time_us:
    delta_pct = (r30c_bc.device_time_us - r30b.device_time_us) / r30b.device_time_us * 100
    if abs(delta_pct) <= 5.0:
        print(f"\nREGRESSION CHECK: PASS — v30c back-compat is within noise of v30b ({delta_pct:+.1f}%)")
    else:
        print(f"\nREGRESSION CHECK: FAIL — v30c back-compat deviates {delta_pct:+.1f}% from v30b")

# Delta: hoisted vs back-compat
if r30c_bc and r30c_h and r30c_bc.device_time_us and r30c_h.device_time_us:
    delta_h_pct = (r30c_h.device_time_us - r30c_bc.device_time_us) / r30c_bc.device_time_us * 100
    print(f"HOISTED vs BACK-COMPAT delta: {delta_h_pct:+.1f}%")
    print(f"  NOTE: In a single-call harness, the prefetch is inside the same JIT function,")
    print(f"  so overall HBM traffic may appear unchanged. Multi-layer savings are realized")
    print(f"  when gamma/router_w loads are hoisted outside the layer loop.")
