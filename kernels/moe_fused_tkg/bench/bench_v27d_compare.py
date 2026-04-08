"""Benchmark comparison: kernel_v27d vs kernel_v27d_fixed."""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "/tmp/bench_v27d_compare"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
from kernels.benchmarking_workspace.benchmark import wrap_benchmark
import importlib.util


def load_kernel(path, name):
    spec = importlib.util.spec_from_file_location("_kernel_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.qwen3_moe_fused_tkg


kernel_v27d_orig = load_kernel(
    "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v27d.py", "v27d"
)
kernel_v27d_fixed_orig = load_kernel(
    "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v27d_fixed.py", "v27d_fixed"
)

kernel_v27d_fn = wrap_benchmark(
    lambda *args: kernel_v27d_orig[2](*args), warmup=10, iters=100
)
kernel_v27d_fixed_fn = wrap_benchmark(
    lambda *args: kernel_v27d_fixed_orig[2](*args), warmup=10, iters=100
)

device = xm.xla_device()
torch.manual_seed(42)
B, H, E, K, I = 1, 2048, 128, 8, 192
scale = 0.1
inp       = (torch.randn(B, 1, H) * scale).to(torch.bfloat16).to(device)
gamma     = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16).to(device)
router_w  = (torch.randn(H, E) * scale).to(torch.bfloat16).to(device)
gate_up_w = torch.zeros(E, H, 2 * I, dtype=torch.bfloat16)
gate_up_w[:, :, 0:I]   = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
gate_up_w[:, :, I:2*I] = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
gate_up_w = gate_up_w.to(device)
xm.mark_step()
down_w = (torch.randn(E, I, H) * scale).to(torch.bfloat16).to(device)
xm.mark_step()


def print_result(r, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if not r:
        print("  ERROR: no benchmark result")
        return
    print(f"  device_time_us       = {r.device_time_us:.2f}")
    print(f"  tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
    print(f"  dma_active_pct       = {r.dma_active_pct:.1f}%")
    print(f"  spill_bytes          = {r.spill_bytes}")
    print(f"  hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"  hbm_write_KiB        = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")
    print(f"  vector_engine_pct    = {r.prof.get('vector_engine_active_time_percent', 0):.1f}%")
    print(f"  mfu_estimated_pct    = {r.prof.get('mfu_estimated_percent', 0):.2f}%")


# --- Benchmark kernel_v27d ---
print("\n[*] Running kernel_v27d ...")
kernel_v27d_fn(inp, gamma, router_w, gate_up_w, down_w)
r_v27d = kernel_v27d_fn.last_result
print_result(r_v27d, "kernel_v27d")

# --- Benchmark kernel_v27d_fixed ---
print("\n[*] Running kernel_v27d_fixed ...")
kernel_v27d_fixed_fn(inp, gamma, router_w, gate_up_w, down_w)
r_fixed = kernel_v27d_fixed_fn.last_result
print_result(r_fixed, "kernel_v27d_fixed")

# --- Delta report ---
print(f"\n{'='*60}")
print("  Delta: v27d_fixed vs v27d  (positive = fixed is better)")
print(f"{'='*60}")
if r_v27d and r_fixed:
    def delta(a, b, label, fmt=".2f", higher_is_better=False):
        d = b - a
        arrow = "^" if (d > 0) == higher_is_better else "v"
        print(f"  {label:<28} {a:{fmt}} -> {b:{fmt}}  ({arrow} {abs(d):{fmt}})")

    delta(r_v27d.device_time_us,   r_fixed.device_time_us,   "device_time_us",       higher_is_better=False)
    delta(r_v27d.tensor_engine_pct, r_fixed.tensor_engine_pct, "tensor_engine_pct",   fmt=".1f", higher_is_better=True)
    delta(r_v27d.dma_active_pct,   r_fixed.dma_active_pct,   "dma_active_pct",       fmt=".1f", higher_is_better=True)
    delta(r_v27d.spill_bytes,      r_fixed.spill_bytes,       "spill_bytes",          fmt="d",   higher_is_better=False)
    delta(r_v27d.prof.get('hbm_read_bytes', 0)/1024,
          r_fixed.prof.get('hbm_read_bytes', 0)/1024,         "hbm_read_KiB",         fmt=".1f", higher_is_better=False)
    delta(r_v27d.prof.get('hbm_write_bytes', 0)/1024,
          r_fixed.prof.get('hbm_write_bytes', 0)/1024,        "hbm_write_KiB",        fmt=".1f", higher_is_better=False)
    delta(r_v27d.prof.get('vector_engine_active_time_percent', 0),
          r_fixed.prof.get('vector_engine_active_time_percent', 0), "vector_engine_pct", fmt=".1f", higher_is_better=True)
    delta(r_v27d.prof.get('mfu_estimated_percent', 0),
          r_fixed.prof.get('mfu_estimated_percent', 0),        "mfu_estimated_pct",    fmt=".2f", higher_is_better=True)
else:
    print("  Cannot compute delta — one or both results missing.")
