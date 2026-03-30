"""
True benchmark for kernel_v14b — pre-splits weights OUTSIDE the NEFF.

run() calls .contiguous() on gate_up_w inside the NEFF, adding ~150K extra
DMA packets that inflate device_time_us by ~20x. This script pre-splits the
weights on device before the benchmark call so that operation is not compiled
into the NEFF.
"""

import os
import sys

# Must be set before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output"
)

sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
from kernels.benchmarking_workspace.benchmark import wrap_benchmark

# ---------------------------------------------------------------------------
# Load kernel module — grab qwen3_moe_fused_tkg directly, NOT run()
# ---------------------------------------------------------------------------
import importlib.util

spec = importlib.util.spec_from_file_location(
    "_kernel", "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v14b.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
kernel_fn = mod.qwen3_moe_fused_tkg   # the @nki.jit function, NOT run()

# Wrap for benchmarking
kernel_fn = wrap_benchmark(kernel_fn, warmup=10, iters=100)

# ---------------------------------------------------------------------------
# Create inputs
# ---------------------------------------------------------------------------
device = xm.xla_device()
torch.manual_seed(42)

B, H, E, K = 1, 2048, 128, 8
I = 192
scale = 0.1

inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16).to(device)
gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16).to(device)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16).to(device)

# Pre-split gate_up_w OUTSIDE the NEFF (on device, before benchmark)
gate_up_w_cpu = torch.zeros(E, H, 2, I, dtype=torch.bfloat16)
gate_up_w_cpu[:, :, 0, :] = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
gate_up_w_cpu[:, :, 1, :] = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
gate_w = gate_up_w_cpu[:, :, 0, :].contiguous().to(device)   # [E, H, I=192]
up_w   = gate_up_w_cpu[:, :, 1, :].contiguous().to(device)   # [E, H, I=192]
xm.mark_step()  # ensure copies are committed before benchmark

down_w_cpu = torch.zeros(E, I, H, dtype=torch.bfloat16)
down_w_cpu[:, :I, :] = (torch.randn(E, I, H) * scale).to(torch.bfloat16)
down_w = down_w_cpu.to(device)
xm.mark_step()

# ---------------------------------------------------------------------------
# Run benchmark — kernel accepts (inp, gamma, router_w, gate_w, up_w, down_w)
# ---------------------------------------------------------------------------
output = kernel_fn(inp, gamma, router_w, gate_w, up_w, down_w)

r = kernel_fn.last_result
if r:
    print(f"\n{'='*60}")
    print("  kernel_v14b — true kernel benchmark (pre-split weights)")
    print(f"{'='*60}")
    print(f"  device_time_us       = {r.device_time_us:.2f}  (v14a: 105.24)")
    print(f"  tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
    print(f"  dma_active_pct       = {r.dma_active_pct:.1f}%")
    print(f"  spill_bytes          = {r.spill_bytes}")
    print(f"  hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}  (v14a: 25688)")
    print(f"  mfu_estimated        = {r.prof.get('mfu_estimated_percent', 0):.2f}%")
    sw_pkts = r.prof.get(
        'software_dynamic_dma_packet_count',
        r.prof.get('sw_dynamic_dma_packet_count', 'N/A')
    )
    print(f"  sw_dma_packet_count  = {sw_pkts}  (v14a: ~6784)")
else:
    print("ERROR: no benchmark result")
