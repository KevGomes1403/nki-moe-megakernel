"""
Parametric MoE kernel benchmark runner.

Usage (must be run as a subprocess — env vars set before any neuron import):
    python bench_runner.py <kernel_version>
    e.g.: python bench_runner.py v1a

Calls run(inp, gamma, router_w, gate_up_w, down_w) uniformly across all versions.
Uses padded gate_up_w [E, H, 2, I_pad=256] so run() handles slicing internally.
"""
import sys
import os

if len(sys.argv) < 2:
    print("Usage: bench_runner.py <version>")
    sys.exit(1)

version = sys.argv[1]  # e.g. "v1a"

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.join(os.path.dirname(BENCH_DIR), "versions")

# Must be set before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    BENCH_DIR, f"output_{version}"
)

sys.path.insert(0, "/home/ubuntu/nki-moe")

import importlib.util
import torch
import torch_xla.core.xla_model as xm
from kernels.benchmarking_workspace.benchmark import wrap_benchmark

# ---------------------------------------------------------------------------
# Load kernel from versions/
# ---------------------------------------------------------------------------
kernel_file = os.path.join(VERSIONS_DIR, f"kernel_{version}.py")
if not os.path.exists(kernel_file):
    print(f"ERROR: kernel file not found: {kernel_file}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location(f"_kernel_{version}", kernel_file)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if not hasattr(mod, "run"):
    print(f"ERROR: no run() function in {kernel_file}")
    sys.exit(1)

run_fn = mod.run

# ---------------------------------------------------------------------------
# Inputs — padded format accepted by all run() versions
# ---------------------------------------------------------------------------
device = xm.xla_device()
torch.manual_seed(42)
B, H, E, K = 1, 2048, 128, 8
I_pad = 256   # padded intermediate dim; run() slices to 192 for v14+

scale = 0.1
inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16).to(device)
gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16).to(device)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16).to(device)
gate_up_w = (torch.randn(E, H, 2, I_pad) * scale).to(torch.bfloat16).to(device)
down_w   = (torch.randn(E, I_pad, H) * scale).to(torch.bfloat16).to(device)
xm.mark_step()

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
kernel_bench = wrap_benchmark(run_fn, warmup=10, iters=100)
kernel_bench(inp, gamma, router_w, gate_up_w, down_w)

r = kernel_bench.last_result
if r:
    print(f"device_time_us       = {r.device_time_us:.2f}")
    print(f"tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
    print(f"dma_active_pct       = {r.dma_active_pct:.1f}%")
    print(f"spill_bytes          = {r.spill_bytes}")
    print(f"mfu_estimated        = {r.prof.get('mfu_estimated_percent', 0):.2f}%")
    print(f"hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"hbm_write_KiB        = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")
else:
    print("ERROR: last_result is None — NTFF not captured")
