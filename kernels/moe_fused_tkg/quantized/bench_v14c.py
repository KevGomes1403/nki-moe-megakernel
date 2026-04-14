"""
Benchmark for v14c (Plan C: single 8-expert wave).
"""
import os
import sys

# MUST be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v14c"
)

import shutil
shutil.copy(
    "/home/ubuntu/nki-moe/.claude/skills/nki-kernel-optimizer/scripts/benchmark.py",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py"),
)

sys.path.insert(0, "/home/ubuntu/nki-moe")

from benchmark import wrap_benchmark

import torch
import torch_xla.core.xla_model as xm

from kernels.moe_fused_tkg.quantized.v14c import qwen3_moe_fused_tkg

_E = 128
_H = 2048
_I = 192
_GU_FLAT = 2 * _I  # 384


def make_inputs(seed=42):
    torch.manual_seed(seed)
    device = xm.xla_device()

    B = 1
    inp = torch.randn(B, 1, _H, dtype=torch.bfloat16).to(device)
    gamma = torch.randn(1, _H, dtype=torch.bfloat16).to(device)
    router_w = torch.randn(_H, _E, dtype=torch.bfloat16).to(device)

    # Use properly quantized weights (not random int8 — avoids NaN from fp8 NaN bit pattern)
    gate_up_w_fp32 = torch.randn(_E, _H, _GU_FLAT) * 0.1
    gate_up_scales = (gate_up_w_fp32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
    gate_up_w = (gate_up_w_fp32 / gate_up_scales.unsqueeze(1)).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8).to(device)
    gate_up_scales = gate_up_scales.to(device)

    down_w_fp32 = torch.randn(_E, _I, _H) * 0.1
    down_scales = (down_w_fp32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
    down_w = (down_w_fp32 / down_scales.unsqueeze(1)).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8).to(device)
    down_scales = down_scales.to(device)

    return inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales


kernel = wrap_benchmark(
    lambda *args: qwen3_moe_fused_tkg[2](*args),
    warmup=5,
    iters=50,
)

args = make_inputs()
xm.mark_step()

print("Running benchmark for v14c (Plan C: single 8-expert wave)...")
kernel(*args)

r = kernel.last_result
if r and r.prof:
    # NOTE: neuron-profile summary-json returns *_percent fields as fractions [0,1],
    # not percentages [0,100]. Multiply by 100 to get actual percentages.
    def pct(key):
        return r.prof.get(key, 0) * 100

    tensor_pct = pct('tensor_engine_active_time_percent')
    dma_pct    = pct('dma_active_time_percent')
    scalar_pct = pct('scalar_engine_active_time_percent')
    vector_pct = pct('vector_engine_active_time_percent')
    gpsimd_pct = pct('gpsimd_engine_active_time_percent')
    mfu        = r.prof.get('mfu_estimated_percent', 0) * 100
    hbm_read_kib  = r.prof.get('hbm_read_bytes', 0) / 1024
    hbm_write_kib = r.prof.get('hbm_write_bytes', 0) / 1024
    spill = r.spill_bytes

    print(f"\n=== v14c Benchmark Results (trn3) ===")
    print(f"device_time_us       = {r.device_time_us:.2f}")
    print(f"tensor_engine_pct    = {tensor_pct:.1f}%")
    print(f"dma_active_pct       = {dma_pct:.1f}%")
    print(f"spill_bytes          = {spill}")
    print(f"scalar_engine_pct    = {scalar_pct:.1f}%")
    print(f"vector_engine_pct    = {vector_pct:.1f}%")
    print(f"gpsimd_engine_pct    = {gpsimd_pct:.1f}%")
    print(f"mfu_estimated_pct    = {mfu:.2f}%")
    print(f"hbm_read_KiB         = {hbm_read_kib:.1f}")
    print(f"hbm_write_KiB        = {hbm_write_kib:.1f}")
    print(f"\n--- vs v12i baseline (89.80 μs on trn3) ---")
    delta = r.device_time_us - 89.80
    pct_diff = delta / 89.80 * 100
    sign = "+" if delta >= 0 else ""
    print(f"delta = {sign}{delta:.2f} μs  ({sign}{pct_diff:.1f}%)")
    print(f"\n--- vs v14a (Plan A: 88.68 μs) ---")
    delta_a = r.device_time_us - 88.68
    pct_diff_a = delta_a / 88.68 * 100
    sign_a = "+" if delta_a >= 0 else ""
    print(f"delta = {sign_a}{delta_a:.2f} μs  ({sign_a}{pct_diff_a:.1f}%)")
elif r:
    print("Benchmark ran but no NTFF profile data was collected.")
    print("To get metrics, delete the NEFF cache and re-run.")
else:
    print("No benchmark result collected (NTFF artifact not found)")
