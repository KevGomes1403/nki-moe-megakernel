"""
Benchmark for qwen3_moe_2wave_half kernel.

Usage:
    python kernels/moe_fused_tkg/bench_2wave_half.py
"""

import os
import sys

# Must be set BEFORE any neuron/torch_xla imports
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_2wave_half"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ubuntu/nki-moe")

from benchmark import wrap_benchmark

import torch
import torch_xla.core.xla_model as xm

from kernels.moe_fused_tkg.kernel_2wave_half import qwen3_moe_2wave_half


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def make_inputs(seed=42):
    torch.manual_seed(seed)

    B, H = 1, 2048
    E, GU_FLAT = 128, 384

    scale = 0.1

    inp      = (torch.randn(B, H) * scale).to(torch.bfloat16)
    gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16)
    router_w = (torch.randn(H, E) * scale).to(torch.bfloat16)
    gate_up_w = (torch.randn(E, H, GU_FLAT) * scale).to(torch.bfloat16)
    down_w   = (torch.randn(E, 192, H) * scale).to(torch.bfloat16)

    return inp, gamma, router_w, gate_up_w, down_w


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  qwen3_moe_2wave_half — Benchmark (warmup=5, iters=50)")
    print("=" * 60)

    # Ensure output dir exists
    out_dir = os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    print("\nCreating inputs...")
    inp, gamma, router_w, gate_up_w, down_w = make_inputs()

    device = xm.xla_device()
    inp_xla     = inp.to(device)
    gamma_xla   = gamma.to(device)
    router_xla  = router_w.to(device)
    gate_up_xla = gate_up_w.to(device)
    down_xla    = down_w.to(device)

    # Wrap the grid-[2] kernel call for benchmarking
    def kernel_call(*args):
        return qwen3_moe_2wave_half[2](*args)

    kernel_call.__name__ = "qwen3_moe_2wave_half"

    benchmarked = wrap_benchmark(kernel_call, warmup=5, iters=50)

    print("\nRunning benchmark (this compiles + profiles the kernel)...")
    benchmarked(inp_xla, gamma_xla, router_xla, gate_up_xla, down_xla)

    result = benchmarked.last_result
    if result is not None and result.prof:
        prof = result.prof
        total_s = prof.get("total_time", 0)

        def _pct(key):
            return prof.get(key, 0) / total_s * 100 if total_s > 0 else 0.0

        print("\n### qwen3_moe_2wave_half — Summary")
        print(f"  device_time_us:    {result.device_time_us:.2f}")
        print(f"  tensor_engine_pct: {_pct('tensor_engine_active_time'):.1f}%")
        print(f"  dma_active_pct:    {_pct('dma_active_time'):.1f}%")
        print(f"  spill_bytes:       {result.spill_bytes}")
        print(f"  mfu_estimated_pct: {prof.get('mfu_estimated_percent', 0):.2f}%")
        print(f"  hbm_read_KiB:      {prof.get('hbm_read_bytes', 0) / 1024:.1f}")
        print(f"  hbm_write_KiB:     {prof.get('hbm_write_bytes', 0) / 1024:.1f}")
    else:
        print("\n[No profile data — check NEURON_RT_INSPECT_ENABLE and NEURON_RT_INSPECT_DEVICE_PROFILE]")


if __name__ == "__main__":
    main()
