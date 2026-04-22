"""
Benchmark for v15f (Plan B [F3b]: int8-reinterpret HWDGE on down_w) using the
trn3 skill's benchmark harness.

Same input recipe and reporting fields as bench_v15c_trn3.py for direct
comparison.
"""
import os
import sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v15f_trn3"
)

import shutil
_HERE = os.path.dirname(os.path.abspath(__file__))
shutil.copy(
    "/home/ubuntu/nki-moe/.claude/skills/nki-kernel-optimizer-trn3/scripts/benchmark.py",
    os.path.join(_HERE, "benchmark.py"),
)

sys.path.insert(0, "/home/ubuntu/nki-moe")
sys.path.insert(0, _HERE)
from benchmark import wrap_benchmark

import torch
import torch_xla.core.xla_model as xm

from kernels.moe_fused_tkg.quantized.v15f import qwen3_moe_fused_tkg
from kernels.moe_fused_tkg.quantized.v15c import pack_gate_up

_E = 128
_H = 2048
_I = 192
_GU_FLAT = 2 * _I
_PMAX = 128
_H_FREE = _H // _PMAX  # 16
_GU_PACKED_PLANES = _H_FREE + 1  # 17


def make_inputs(seed=42):
    torch.manual_seed(seed)
    device = xm.xla_device()
    inp = torch.randn(1, 1, _H, dtype=torch.bfloat16).to(device)
    gamma = torch.randn(1, _H, dtype=torch.bfloat16).to(device)
    router_w = torch.randn(_H, _E, dtype=torch.bfloat16).to(device)

    # Build gate_up weight + scales using the same recipe as bench_v15c_trn3.py
    gate_up_w_fp32 = torch.randn(_E, _H, _GU_FLAT) * 0.1
    gate_up_scales = (gate_up_w_fp32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
    gate_up_w_i8 = (
        gate_up_w_fp32 / gate_up_scales.unsqueeze(1)
    ).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8)

    # v15f: packed gate_up (identical layout to v15c) + int8 down_w.
    gate_up_packed = pack_gate_up(gate_up_w_i8, gate_up_scales).to(device)

    down_w_fp32 = torch.randn(_E, _I, _H) * 0.1
    down_scales = (down_w_fp32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
    down_w = (
        down_w_fp32 / down_scales.unsqueeze(1)
    ).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8).to(device)
    down_scales = down_scales.to(device)

    return inp, gamma, router_w, gate_up_packed, down_w, down_scales


kernel = wrap_benchmark(
    lambda *args: qwen3_moe_fused_tkg[2](*args),
    warmup=5,
    iters=50,
)

args = make_inputs()
xm.mark_step()

print("Running v15f bench (trn3 harness) — multi-run noise estimate...")

_NUM_RUNS = 5
results = []
for run_idx in range(_NUM_RUNS):
    kernel(*args)
    r = kernel.last_result
    if r and r.prof:
        results.append(r)
        print(f"  run {run_idx+1}: device_time_us = {r.device_time_us:.3f}")
    else:
        print(f"  run {run_idx+1}: no profile!")
        sys.exit(1)

times = [r.device_time_us for r in results]
mean_us = sum(times) / len(times)

# Use first run for the detailed profile dump (all runs share the same compiled graph,
# so engine percentages will be near-identical)
r = results[0]
prof = r.prof
def pct(key): return prof.get(key, 0) * 100

print("\n=== v15f Benchmark (trn3 harness) ===")
print(f"device_time_us_runs  = {[f'{t:.3f}' for t in times]}")
print(f"device_time_us_mean  = {mean_us:.3f}  (N={_NUM_RUNS})")
print(f"device_time_us_min   = {min(times):.3f}")
print(f"device_time_us_max   = {max(times):.3f}")
print(f"tensor_engine_pct    = {pct('tensor_engine_active_time_percent'):.2f}%")
print(f"vector_engine_pct    = {pct('vector_engine_active_time_percent'):.2f}%")
print(f"scalar_engine_pct    = {pct('scalar_engine_active_time_percent'):.2f}%")
print(f"gpsimd_engine_pct    = {pct('gpsimd_engine_active_time_percent'):.2f}%")
print(f"dma_active_pct       = {pct('dma_active_time_percent'):.2f}%")
print(f"spill_bytes          = {r.spill_bytes}")
print(f"mfu_estimated_pct    = {prof.get('mfu_estimated_percent', 0):.4f}%")
print(f"mbu_estimated_pct    = {prof.get('mbu_estimated_percent', 0):.4f}%")
print(f"mm_arithmetic_intensity = {prof.get('mm_arithmetic_intensity', 0):.3f}")
print(f"hbm_read_KiB         = {prof.get('hbm_read_bytes', 0)/1024:.1f}")
print(f"hbm_write_KiB        = {prof.get('hbm_write_bytes', 0)/1024:.1f}")
print(f"cc_op_count          = {prof.get('cc_op_count', 0)}")
print(f"cc_op_active_time_us = {prof.get('cc_op_active_time', 0)*1e6:.2f}")
print(f"\nv15c baseline: 72.88 μs  delta_mean = {mean_us - 72.88:+.2f} μs ({(mean_us - 72.88)/72.88*100:+.2f}%)")
