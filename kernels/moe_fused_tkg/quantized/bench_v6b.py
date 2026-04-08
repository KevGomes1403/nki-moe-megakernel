import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v6b"
)

import shutil
shutil.copy(
    "/home/ubuntu/nki-moe/.claude/skills/nki-kernel-optimizer/scripts/benchmark.py",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")
)
from benchmark import wrap_benchmark

import numpy as np
import torch
import torch_xla.core.xla_model as xm

sys.path.insert(0, "/home/ubuntu/nki-moe")
from kernels.moe_fused_tkg.quantized.v6b import qwen3_moe_fused_tkg

rng = np.random.default_rng(42)
device = xm.xla_device()
T, H, E, I, GU = 1, 2048, 128, 192, 384
H_SHARD_TP = 512

inp = torch.tensor(rng.random((T, 1, H)).astype(np.float16) * 0.1, dtype=torch.bfloat16).to(device)
gamma = torch.tensor(rng.random((1, H)).astype(np.float16) + 0.5, dtype=torch.bfloat16).to(device)
router_w = torch.tensor(rng.random((H, E)).astype(np.float16) * 0.1, dtype=torch.bfloat16).to(device)
gate_up_w = torch.tensor((rng.integers(-127, 127, (E, H, GU))).astype(np.int8)).to(device)
gate_up_scales = torch.tensor(rng.random((E, GU)).astype(np.float32) * 0.01 + 0.001).to(device)
down_w = torch.tensor((rng.integers(-127, 127, (E, I, H))).astype(np.int8)).to(device)
down_scales = torch.tensor(rng.random((E, H_SHARD_TP)).astype(np.float32) * 0.01 + 0.001).to(device)
xm.mark_step()

kernel = wrap_benchmark(lambda *args: qwen3_moe_fused_tkg[2](*args), warmup=5, iters=50)
kernel(inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales)

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
