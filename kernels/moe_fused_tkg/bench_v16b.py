"""
Benchmark for kernel_v16b — Plan B: per-h-tile 2D DMA in h1 loop.

kernel_v16b improvements vs v15:
  gate_up: H_free=16 individual 2D indirect DMAs per expert (one per h-tile, inside h1 loop)
           vs one 3D indirect DMA per expert in v15.
           2D pattern [[_GU_FLAT, _PMAX], [1, _GU_FLAT]] is simpler than
           v15's 3D [[_GU_FLAT, _PMAX], [_PMAX*_GU_FLAT, H_free], [1, _GU_FLAT]].
           DMA for h1+1 overlaps with matmul for h1 via nl.affine_range(H_free).
  SBUF:    gate_t1/up_t1 are [_PMAX, I0] per h-tile (not [_PMAX, H_free, I0]).
  down:    unchanged from v15 — ONE coalesced full-H DMA per I-tile per expert.
"""

import os
import sys

# Must be set before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output_v16b"
)

sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
from kernels.benchmarking_workspace.benchmark import wrap_benchmark

import importlib.util

spec = importlib.util.spec_from_file_location(
    "_kernel", "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v16b.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
_kernel_orig = mod.qwen3_moe_fused_tkg

kernel_fn = wrap_benchmark(lambda *args: _kernel_orig[2](*args), warmup=10, iters=100)

# ---------------------------------------------------------------------------
# Create inputs — native layouts, no repack
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
    print("  kernel_v16b — Plan B: per-h-tile 2D DMA in h1 loop")
    print(f"{'='*60}")
    print(f"  device_time_us       = {r.device_time_us:.2f}")
    print(f"  tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
    print(f"  dma_active_pct       = {r.dma_active_pct:.1f}%")
    print(f"  spill_bytes          = {r.spill_bytes}")
    print(f"  hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"  mfu_estimated        = {r.prof.get('mfu_estimated_percent', 0):.2f}%")
    sw_pkts = r.prof.get(
        'software_dynamic_dma_packet_count',
        r.prof.get('sw_dynamic_dma_packet_count', 'N/A')
    )
    print(f"  sw_dma_packet_count  = {sw_pkts}")
    print()
    print("  Baseline (v15):")
    print("    device_time_us:       145.37")
    print("    sw_dma_packet_count:  9856")
    print("    dma_active:           67%")
else:
    print("ERROR: no benchmark result")
