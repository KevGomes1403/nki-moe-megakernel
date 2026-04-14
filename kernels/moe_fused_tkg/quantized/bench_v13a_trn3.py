import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"  # trn3 target
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v13a_trn3"
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
from kernels.moe_fused_tkg.quantized.v13a import qwen3_moe_fused_tkg

_E = 128; _H = 2048; _I = 192; _GU_FLAT = 384

def make_inputs(seed=42):
    torch.manual_seed(seed)
    device = xm.xla_device()
    inp = torch.randn(1, 1, _H, dtype=torch.bfloat16).to(device)
    gamma = torch.randn(1, _H, dtype=torch.bfloat16).to(device)
    router_w = torch.randn(_H, _E, dtype=torch.bfloat16).to(device)
    gate_up_w_fp32 = torch.randn(_E, _H, _GU_FLAT) * 0.1
    gate_up_scales = (gate_up_w_fp32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
    gate_up_w = (gate_up_w_fp32 / gate_up_scales.unsqueeze(1)).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8).to(device)
    gate_up_scales = gate_up_scales.to(device)
    down_w_fp32 = torch.randn(_E, _I, _H) * 0.1
    down_scales = (down_w_fp32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
    down_w = (down_w_fp32 / down_scales.unsqueeze(1)).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8).to(device)
    down_scales = down_scales.to(device)
    return inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales

kernel = wrap_benchmark(lambda *args: qwen3_moe_fused_tkg[2](*args), warmup=5, iters=50)
args = make_inputs()
xm.mark_step()
print("Benchmarking v13a on trn3...")
kernel(*args)
r = kernel.last_result
if r:
    scalar_pct = r.prof.get('scalar_engine_active_time_percent', 0) * 100
    vector_pct = r.prof.get('vector_engine_active_time_percent', 0) * 100
    hbm_read_kib = r.prof.get('hbm_read_bytes', 0) / 1024
    hbm_write_kib = r.prof.get('hbm_write_bytes', 0) / 1024
    mfu = r.prof.get('mfu_estimated_percent', 0)
    print(f"\n=== v13a trn3 Benchmark ===")
    print(f"device_time_us       = {r.device_time_us:.2f}")
    print(f"tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
    print(f"scalar_engine_pct    = {scalar_pct:.1f}%")
    print(f"vector_engine_pct    = {vector_pct:.1f}%")
    print(f"dma_active_pct       = {r.dma_active_pct:.1f}%")
    print(f"spill_bytes          = {r.spill_bytes}")
    print(f"hbm_read_KiB         = {hbm_read_kib:.1f}")
    print(f"hbm_write_KiB        = {hbm_write_kib:.1f}")
    print(f"mfu_estimated        = {mfu:.2f}%")
    print(f"\n--- vs trn2 v12i baseline (90.54 μs) ---")
    delta = r.device_time_us - 90.54
    sign = '+' if delta >= 0 else ''
    print(f"delta = {sign}{delta:.2f} μs  ({sign}{delta/90.54*100:.1f}%)")
else:
    print("No benchmark result — check env var setup")
