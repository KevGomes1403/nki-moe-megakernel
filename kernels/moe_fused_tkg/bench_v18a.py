"""
Benchmark for kernel_v18a — Plan C tensor_partition_reduce for RMSNorm cross-partition sum.

Replaces the 128×128 TensorE ones-matmul with:
  tensor_partition_reduce (GpSimdE) + nc_stream_shuffle broadcast.
Eliminates the 64 KB matmul_const SBUF allocation.
"""

import os
import sys

# Must be set before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output_v18a"
)

sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm

# benchmark.py must be in the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import wrap_benchmark

import importlib.util

spec = importlib.util.spec_from_file_location(
    "_kernel", "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v18a.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
_kernel_orig = mod.qwen3_moe_fused_tkg

kernel_fn = wrap_benchmark(lambda *args: _kernel_orig[2](*args), warmup=10, iters=100)

# ---------------------------------------------------------------------------
# Create inputs — native layouts, no repack (match bench_v16c.py)
# ---------------------------------------------------------------------------
device = xm.xla_device()
torch.manual_seed(42)

B, H, E, K = 1, 2048, 128, 8
I = 192
scale = 0.1

inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16).to(device)
gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16).to(device)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16).to(device)

# gate_up_w: native [E, H, 2*I=384] — gate cols 0:I, up cols I:2I
gate_up_w = torch.zeros(E, H, 2 * I, dtype=torch.bfloat16)
gate_up_w[:, :, 0:I]   = (torch.randn(E, H, I) * scale).to(torch.bfloat16)  # gate
gate_up_w[:, :, I:2*I] = (torch.randn(E, H, I) * scale).to(torch.bfloat16)  # up
gate_up_w = gate_up_w.to(device)
xm.mark_step()

# down_w: native [E, I=192, H=2048]
down_w = (torch.randn(E, I, H) * scale).to(torch.bfloat16).to(device)
xm.mark_step()

# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------
output = kernel_fn(inp, gamma, router_w, gate_up_w, down_w)

r = kernel_fn.last_result
if r:
    print(f"\n{'='*60}")
    print("  kernel_v18a — Plan C tensor_partition_reduce")
    print(f"{'='*60}")
    # Note: last_result stores engine pct as fractions (0.0–1.0), multiply by 100 for display
    te_pct = r.tensor_engine_pct * 100 if r.tensor_engine_pct <= 1.0 else r.tensor_engine_pct
    dma_pct = r.dma_active_pct * 100 if r.dma_active_pct <= 1.0 else r.dma_active_pct
    ve_raw = r.prof.get('vector_engine_active_time_percent', 0)
    se_raw = r.prof.get('scalar_engine_active_time_percent', 0)
    ve_pct = ve_raw * 100 if ve_raw <= 1.0 else ve_raw
    se_pct = se_raw * 100 if se_raw <= 1.0 else se_raw
    mfu_raw = r.prof.get('mfu_estimated_percent', 0)
    mfu_pct = mfu_raw * 100 if mfu_raw <= 1.0 else mfu_raw
    print(f"  device_time_us       = {r.device_time_us:.2f}")
    print(f"  tensor_engine_pct    = {te_pct:.1f}%")
    print(f"  dma_active_pct       = {dma_pct:.1f}%")
    print(f"  spill_bytes          = {r.spill_bytes}")
    print(f"  hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"  hbm_write_KiB        = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")
    print(f"  vector_engine_pct    = {ve_pct:.1f}%")
    print(f"  scalar_engine_pct    = {se_pct:.1f}%")
    print(f"  mfu_estimated        = {mfu_pct:.2f}%")
    sw_pkts = r.prof.get(
        'software_dynamic_dma_packet_count',
        r.prof.get('sw_dynamic_dma_packet_count', 'N/A')
    )
    print(f"  sw_dma_packet_count  = {sw_pkts}")
    print()
    print("  Prior kernel baselines:")
    print("    v17a (Plan A base): device_time_us=124.23, hbm_read_KiB=31760")
else:
    print("ERROR: no benchmark result")
