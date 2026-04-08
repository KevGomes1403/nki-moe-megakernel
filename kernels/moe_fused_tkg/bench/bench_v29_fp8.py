import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output_v29_fp8"
)
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
from kernels.benchmarking_workspace.benchmark import wrap_benchmark
import importlib.util

spec = importlib.util.spec_from_file_location(
    "_kernel", "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v29_fp8.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
_kernel_orig = mod.qwen3_moe_fused_tkg

kernel_fn = wrap_benchmark(lambda *args: _kernel_orig[2](*args), warmup=10, iters=100)

device = xm.xla_device()
torch.manual_seed(42)
B, H, E, K, I = 1, 2048, 128, 8, 192
GU_FLAT = 2 * I  # 384
GU_J_BLOCKS = GU_FLAT // 128  # 3
H_BLOCKS = H // 128            # 16
scale = 0.1

inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16).to(device)
gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16).to(device)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16).to(device)
xm.mark_step()

# Build fp8 weights with ±240 clamp (safe range for trn2 hardware fp8 activation pipeline)
gate_up_w_f32 = torch.zeros(E, H, GU_FLAT, dtype=torch.float32)
gate_up_w_f32[:, :, 0:I]   = torch.randn(E, H, I) * scale
gate_up_w_f32[:, :, I:2*I] = torch.randn(E, H, I) * scale
gate_up_w_rsh = gate_up_w_f32.reshape(E, H, GU_J_BLOCKS, 128)
gate_up_scales_cpu = gate_up_w_rsh.abs().amax(dim=3).clamp(min=1e-12) / 240.0
gate_up_scales_bcast = gate_up_scales_cpu.repeat_interleave(128, dim=2)
gate_up_w_fp8_cpu = (gate_up_w_f32 / gate_up_scales_bcast).clamp(-240, 240).to(torch.float8_e4m3fn)

down_w_f32 = (torch.randn(E, I, H) * scale)
down_w_rsh = down_w_f32.reshape(E, I, H_BLOCKS, 128)
down_scales_cpu = down_w_rsh.abs().amax(dim=3).clamp(min=1e-12) / 240.0
down_scales_bcast = down_scales_cpu.repeat_interleave(128, dim=2)
down_w_fp8_cpu = (down_w_f32 / down_scales_bcast).clamp(-240, 240).to(torch.float8_e4m3fn)

gate_up_w_int8 = gate_up_w_fp8_cpu.view(torch.int8).to(device)
gate_up_scales_dev = gate_up_scales_cpu.to(device)
down_w_int8 = down_w_fp8_cpu.view(torch.int8).to(device)
down_scales_dev = down_scales_cpu.to(device)
xm.mark_step()

output = kernel_fn(inp, gamma, router_w, gate_up_w_int8, gate_up_scales_dev, down_w_int8, down_scales_dev)

r = kernel_fn.last_result
if r:
    print(f"device_time_us = {r.device_time_us:.2f}")
    print(f"tensor_engine_pct = {r.tensor_engine_pct:.1f}%")
    print(f"dma_active_pct = {r.dma_active_pct:.1f}%")
    print(f"spill_bytes = {r.spill_bytes}")
    sw_pkts = r.prof.get('software_dynamic_dma_packet_count', r.prof.get('sw_dynamic_dma_packet_count', 'N/A'))
    print(f"sw_dma_packet_count = {sw_pkts}")
    hbm_read = r.prof.get('hbm_read_KiB', r.prof.get('hbm_read', 'N/A'))
    print(f"hbm_read_KiB = {hbm_read}")
    print(f"  baseline v28f=98.15us; fp8 target: ~70-80us (halved weight bytes)")
