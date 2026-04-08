import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v10c"
)
sys.path.insert(0, "/home/ubuntu/nki-moe")
import shutil
shutil.copy(
    "/home/ubuntu/nki-moe/.claude/skills/nki-kernel-optimizer/scripts/benchmark.py",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")
)
from benchmark import wrap_benchmark
import torch, numpy as np
import torch_xla.core.xla_model as xm
from kernels.moe_fused_tkg.quantized.v10c import qwen3_moe_fused_tkg

device = xm.xla_device()
T, H, E, I, GU = 1, 2048, 128, 192, 384
torch.manual_seed(42)

inp   = (torch.randn(T, 1, H) * 0.1).to(torch.bfloat16).to(device)
gamma = (torch.ones(1, H) + torch.randn(1, H) * 0.1).to(torch.bfloat16).to(device)
router_w = (torch.randn(H, E) * 0.01).to(torch.bfloat16).to(device)

gate_up_w_fp32 = torch.randn(E, H, GU) * 0.1
gate_up_scales = (gate_up_w_fp32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
gate_up_w_q = (gate_up_w_fp32 / gate_up_scales.unsqueeze(1)).clamp(-240,240).to(torch.float8_e4m3fn).view(torch.int8)

down_w_fp32 = torch.randn(E, I, H) * 0.1
down_scales_full = (down_w_fp32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
down_w_q = (down_w_fp32 / down_scales_full.unsqueeze(1)).clamp(-240,240).to(torch.float8_e4m3fn).view(torch.int8)

gate_up_w    = gate_up_w_q.to(device)
gate_up_scales_dev = gate_up_scales.to(device)
down_w       = down_w_q.to(device)
down_scales  = down_scales_full.to(device)   # [E, H=2048]
xm.mark_step()

kernel = wrap_benchmark(lambda *args: qwen3_moe_fused_tkg[2](*args), warmup=5, iters=50)
kernel(inp, gamma, router_w, gate_up_w, gate_up_scales_dev, down_w, down_scales)

r = kernel.last_result
if r:
    print(f"device_time_us       = {r.device_time_us:.2f}")
    print(f"tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
    print(f"dma_active_pct       = {r.dma_active_pct:.1f}%")
    print(f"spill_bytes          = {r.spill_bytes}")
    print(f"scalar_engine_pct    = {r.prof.get('scalar_engine_active_time_percent', 0):.1f}%")
    print(f"mfu_estimated        = {r.prof.get('mfu_estimated_percent', 0):.2f}%")
    print(f"hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"hbm_write_KiB        = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")
else:
    print("ERROR: last_result is None")
