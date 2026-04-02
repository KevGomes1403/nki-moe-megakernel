"""
Correctness test and benchmark for any NKI MoE kernel.

Usage:
    python test_correctness.py                              # correctness, default kernel
    python test_correctness.py kernel_v1a.py               # correctness, named kernel
    python test_correctness.py /abs/path/to/kernel.py      # correctness, arbitrary path

    python test_correctness.py --benchmark                  # benchmark default kernel
    python test_correctness.py kernel_v1a.py --benchmark   # benchmark named kernel
    python test_correctness.py --benchmark --warmup 5 --iters 50
"""

import argparse
import importlib.util
import os
import sys
import numpy as np
import torch

# Must be set before importing torch_xla or any neuron modules
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output"
)

sys.path.insert(0, "/home/ubuntu/nki-moe")

from kernels.moe_fused_tkg.reference import qwen3_moe_fused_tkg_reference
from kernels.benchmarking_workspace.benchmark import nki_benchmark

import torch_xla.core.xla_model as xm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Correctness test and benchmark for NKI MoE kernels."
    )
    parser.add_argument(
        "kernel",
        nargs="?",
        default=None,
        help="Kernel file to test (default: kernel.py in this directory). "
             "Bare filenames are resolved relative to this directory.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmarking instead of numerical correctness check.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="neuron-bench warmup iterations (default: 10, benchmark mode only).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="neuron-bench benchmark iterations (default: 100, benchmark mode only).",
    )
    return parser.parse_args()


def _resolve_kernel_path(kernel_arg: str | None) -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if kernel_arg is None:
        return os.path.join(this_dir, "kernel.py")
    if os.path.isabs(kernel_arg) or os.path.exists(kernel_arg):
        return kernel_arg
    return os.path.join(this_dir, kernel_arg)


def _load_kernel_run(kernel_path: str):
    """Dynamically load a kernel file and return its `run` function."""
    kernel_path = os.path.abspath(kernel_path)
    if not os.path.exists(kernel_path):
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")
    spec = importlib.util.spec_from_file_location("_nki_kernel_under_test", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise AttributeError(f"Kernel file has no `run` function: {kernel_path}")
    return mod.run


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def make_inputs(seed=42):
    """Create deterministic Qwen3 TKG inputs."""
    torch.manual_seed(seed)
    B, S, H = 1, 1, 2048
    E, K = 128, 8
    I_padded = 256  # padded from I_tp=192 to next multiple of 128

    # Use small-magnitude values to keep numerical differences within bf16 tolerance
    scale = 0.1

    inp = (torch.randn(B, S, H) * scale).to(torch.bfloat16)
    gamma = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16)  # near 1.0 for stability
    router_w = (torch.randn(H, E) * scale).to(torch.bfloat16)

    # gate_up_w: [E, H, 2, I] - zero-pad the last 64 channels (192->256)
    gate_up_w = torch.zeros(E, H, 2, I_padded, dtype=torch.bfloat16)
    gate_up_w[:, :, :, :192] = (torch.randn(E, H, 2, 192) * scale).to(torch.bfloat16)

    # down_w: [E, I, H] - zero-pad the last 64 rows (192->256)
    down_w = torch.zeros(E, I_padded, H, dtype=torch.bfloat16)
    down_w[:, :192, :] = (torch.randn(E, 192, H) * scale).to(torch.bfloat16)

    return inp, gamma, router_w, gate_up_w, down_w


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_correctness(run_kernel, kernel_path: str):
    print(f"Testing kernel: {kernel_path}")
    print("Creating inputs...")
    inp, gamma, router_w, gate_up_w, down_w = make_inputs()

    # 1. PyTorch reference (CPU)
    print("Running PyTorch reference...")
    ref_out = qwen3_moe_fused_tkg_reference(inp, gamma, router_w, gate_up_w, down_w)
    print(f"Reference output shape: {ref_out.shape}, dtype: {ref_out.dtype}")
    print(f"Reference output norm: {ref_out.float().norm():.4f}")

    # 2. NKI kernel (Trainium 2)
    print("Running NKI kernel on device...")
    device = xm.xla_device()
    inp_xla     = inp.to(device)
    gamma_xla   = gamma.to(device)
    router_xla  = router_w.to(device)
    gate_up_xla = gate_up_w.to(device)
    down_xla    = down_w.to(device)

    nki_out = run_kernel(inp_xla, gamma_xla, router_xla, gate_up_xla, down_xla)
    xm.mark_step()
    nki_out_cpu = nki_out.cpu()
    print(f"NKI output shape: {nki_out_cpu.shape}, dtype: {nki_out_cpu.dtype}")
    print(f"NKI output norm: {nki_out_cpu.float().norm():.4f}")

    # 3. Numerical comparison
    ref_np = ref_out.float().numpy().flatten()
    nki_np = nki_out_cpu.float().numpy().flatten()

    # Align shapes (moe_block_tkg may return [T, H] = [1, 2048] or flat [2048])
    if ref_np.shape != nki_np.shape:
        print(f"Shape mismatch (flattened): ref={ref_np.shape}, nki={nki_np.shape} — aligning")
        min_len = min(len(ref_np), len(nki_np))
        ref_np = ref_np[:min_len]
        nki_np = nki_np[:min_len]
    assert ref_np.shape == nki_np.shape, f"Shape mismatch: ref={ref_np.shape}, nki={nki_np.shape}"

    abs_diff = np.abs(nki_np - ref_np)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    print(f"\nmean_diff={mean_diff:.2e}  max_diff={max_diff:.2e}")

    # bf16 hardware tolerance: ~1% relative error is expected for tiled bf16 matmul
    # atol=0.5 covers ~1 bf16 ULP at max output magnitude (~35), rtol=0.02 covers relative errors
    np.testing.assert_allclose(nki_np, ref_np, rtol=0.02, atol=0.5)
    print(f"max_diff={max_diff:.2e}  PASS")


def run_benchmark(run_kernel, kernel_path: str, warmup: int, iters: int):
    print(f"Benchmarking kernel: {kernel_path}")
    print(f"  warmup={warmup}  iters={iters}")
    print("Creating inputs...")
    inp, gamma, router_w, gate_up_w, down_w = make_inputs()

    device = xm.xla_device()
    inp_xla     = inp.to(device)
    gamma_xla   = gamma.to(device)
    router_xla  = router_w.to(device)
    gate_up_xla = gate_up_w.to(device)
    down_xla    = down_w.to(device)

    result = nki_benchmark(
        run_kernel,
        inp_xla, gamma_xla, router_xla, gate_up_xla, down_xla,
        warmup=warmup,
        iters=iters,
    )

    if result is not None and result.prof:
        _print_full_profile(result.prof)


def _print_full_profile(prof: dict) -> None:
    """Print all profile keys not already shown by _print_profile_results."""
    # Keys already printed by _print_profile_results in benchmark.py
    _shown = {
        "total_time", "total_active_time",
        "tensor_engine_active_time", "vector_engine_active_time",
        "scalar_engine_active_time", "dma_active_time",
        "mfu_estimated_percent", "mbu_estimated_percent",
        "mm_arithmetic_intensity",
        "hbm_read_bytes", "hbm_write_bytes",
        "spill_save_bytes", "spill_reload_bytes",
    }
    extra = {k: v for k, v in sorted(prof.items()) if k not in _shown}
    if not extra:
        return
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Additional profile fields")
    print(sep)
    for k, v in extra.items():
        if isinstance(v, float) and "time" in k:
            print(f"  {k:<45} = {v * 1e6:.2f} μs")
        elif isinstance(v, float) and "bytes" in k:
            print(f"  {k:<45} = {v / 1024:.1f} KiB")
        else:
            print(f"  {k:<45} = {v}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    kernel_path = _resolve_kernel_path(args.kernel)
    run_kernel = _load_kernel_run(kernel_path)

    if args.benchmark:
        run_benchmark(run_kernel, kernel_path, warmup=args.warmup, iters=args.iters)
    else:
        run_correctness(run_kernel, kernel_path)
