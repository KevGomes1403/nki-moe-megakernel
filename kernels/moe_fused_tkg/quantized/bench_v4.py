import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output_v4"
)
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
import importlib.util

from kernels.moe_fused_tkg.quantized.benchmark import wrap_benchmark

spec = importlib.util.spec_from_file_location(
    "_kernel_v4", "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/quantized/v4.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
_kernel_orig = mod.qwen3_moe_fused_tkg

kernel_fn = wrap_benchmark(lambda *args: _kernel_orig[2](*args), warmup=5, iters=50)

device = xm.xla_device()
torch.manual_seed(42)
B, H, E, K, I = 1, 2048, 128, 8, 192
GU_FLAT = 2 * I
H_SHARD_TEST = 512
scale = 0.1

inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16).to(device)
gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16).to(device)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16).to(device)
xm.mark_step()

gate_up_w_f32 = torch.zeros(E, H, GU_FLAT, dtype=torch.float32)
gate_up_w_f32[:, :, 0:I]   = torch.randn(E, H, I) * scale
gate_up_w_f32[:, :, I:2*I] = torch.randn(E, H, I) * scale

# Per-column scales: [E, GU=384]
gate_up_scales_cpu = gate_up_w_f32.abs().amax(dim=1).clamp(min=1e-12) / 240.0
gate_up_scales_bcast = gate_up_scales_cpu.unsqueeze(1).expand(E, H, GU_FLAT)
gate_up_w_fp8_cpu = (gate_up_w_f32 / gate_up_scales_bcast).clamp(-240, 240).to(torch.float8_e4m3fn)

down_w_f32 = torch.randn(E, I, H) * scale
# Per-column scales: [E, H_shard=512]
down_scales_cpu = down_w_f32[:, :, 0:H_SHARD_TEST].abs().amax(dim=1).clamp(min=1e-12) / 240.0
down_scales_bcast = down_scales_cpu.unsqueeze(1).expand(E, I, H_SHARD_TEST)
down_w_fp8_shard = (down_w_f32[:, :, 0:H_SHARD_TEST] / down_scales_bcast).clamp(-240, 240).to(torch.float8_e4m3fn)
down_w_fp8_cpu = torch.cat([
    down_w_fp8_shard,
    torch.zeros(E, I, H - H_SHARD_TEST, dtype=torch.float8_e4m3fn)
], dim=2)

gate_up_w_int8  = gate_up_w_fp8_cpu.view(torch.int8).to(device)
gate_up_scales_dev = gate_up_scales_cpu.to(device)
down_w_int8     = down_w_fp8_cpu.view(torch.int8).to(device)
down_scales_dev = down_scales_cpu.to(device)
xm.mark_step()

output = kernel_fn(inp, gamma, router_w, gate_up_w_int8, gate_up_scales_dev, down_w_int8, down_scales_dev)

r = kernel_fn.last_result
if r:
    total_s = r.prof.get("total_time", 0)

    def _pct(key):
        v = r.prof.get(key, 0)
        return (v / total_s * 100) if total_s > 0 else 0.0

    print(f"device_time_us      = {r.device_time_us:.2f}")
    print(f"tensor_engine_pct   = {_pct('tensor_engine_active_time'):.1f}%")
    print(f"vector_engine_pct   = {_pct('vector_engine_active_time'):.1f}%")
    print(f"scalar_engine_pct   = {_pct('scalar_engine_active_time'):.1f}%")
    print(f"dma_active_pct      = {_pct('dma_active_time'):.1f}%")
    print(f"spill_bytes         = {r.spill_bytes}")
    hbm_read = r.prof.get('hbm_read_bytes', 0) / 1024
    print(f"hbm_read_KiB        = {hbm_read:.1f}")
    hbm_write = r.prof.get('hbm_write_bytes', 0) / 1024
    print(f"hbm_write_KiB       = {hbm_write:.1f}")
    mfu = r.prof.get('mfu_estimated_percent', 0)
    print(f"mfu_estimated_pct   = {mfu:.2f}%")
else:
    print("No benchmark result captured.")
