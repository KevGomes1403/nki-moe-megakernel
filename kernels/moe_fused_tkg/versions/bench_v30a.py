"""
Standalone benchmark script for kernel_v30a_sbuf_io.py on trn3.

Run from the versions directory:
    cd /home/ubuntu/nki-moe/kernels/moe_fused_tkg/versions
    python bench_v30a.py
"""
import os
import sys

# MUST be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v30a"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import wrap_benchmark

import torch
import torch_xla.core.xla_model as xm
import numpy as np

from kernel_v30a_sbuf_io import qwen3_moe_fused_tkg_sbuf_io

# Fixed dims matching kernel contracts
_H = 2048
_E = 128
_I = 192

device = xm.xla_device()
rng = np.random.default_rng(42)

def make_tensor(shape, dtype=torch.bfloat16, scale=0.1):
    arr = rng.standard_normal(shape).astype(np.float32) * scale
    return torch.tensor(arr, dtype=dtype).to(device)

B = 1
inp       = make_tensor((B, 1, _H))           # [B, 1, H] bf16
gamma     = make_tensor((1, _H))              # [1, H] bf16
router_w  = make_tensor((_H, _E))             # [H, E] bf16
gate_up_w = make_tensor((_E, _H, 2 * _I))     # [E, H, 2*I] bf16
down_w    = make_tensor((_E, _I, _H))         # [E, I, H] bf16

print("Benchmarking v30a (sbuf_io) on trn3...")
# Wrap the LNC=2 subscripted kernel directly so wrap_benchmark can call it.
# qwen3_moe_fused_tkg_sbuf_io[2] selects the LNC=2 specialisation.
bench_kernel = wrap_benchmark(qwen3_moe_fused_tkg_sbuf_io[2], warmup=5, iters=50)
bench_kernel(inp, gamma, router_w, gate_up_w, down_w)


r = bench_kernel.last_result

print(f"\n--- v30a Benchmark Metrics ---")
if r is not None:
    print(f"device_time_us        : {r.device_time_us:.2f}")
    print(f"tensor_engine_pct     : {r.tensor_engine_pct:.2f}")
    print(f"dma_active_pct        : {r.dma_active_pct:.2f}")
    print(f"spill_bytes           : {r.spill_bytes}")
    print(f"mfu_estimated_percent : {r.prof.get('mfu_estimated_percent', 'N/A')}")
    print(f"hbm_read_KiB          : {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"hbm_write_KiB         : {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")
else:
    print("No profiling data captured.")
