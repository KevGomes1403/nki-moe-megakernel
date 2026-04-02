"""Benchmark v12a: dma_transpose K-cache optimization."""
import os
import sys

# MUST be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v12a"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import wrap_benchmark

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from v12a import qwen3_attn_tkg_fused_oproj_v12a

v12a_bench = wrap_benchmark(qwen3_attn_tkg_fused_oproj_v12a, warmup=5, iters=50)

device = xm.xla_device()
rng = np.random.default_rng(42)

B, H, d, S_prior = 1, 2048, 128, 640
Hq_out = 1024

hidden = torch.tensor(rng.random((B, 1, H)).astype(np.float16), dtype=torch.bfloat16).to(device)
Wq = torch.tensor(rng.random((Hq_out, H)).astype(np.float16), dtype=torch.bfloat16).to(device)
Wk = torch.tensor(rng.random((d, H)).astype(np.float16), dtype=torch.bfloat16).to(device)
Wv = torch.tensor(rng.random((d, H)).astype(np.float16), dtype=torch.bfloat16).to(device)
Wo = torch.tensor(rng.random((Hq_out, H)).astype(np.float16), dtype=torch.bfloat16).to(device)
q_nw = torch.tensor(rng.random((d,)).astype(np.float16), dtype=torch.bfloat16).to(device)
k_nw = torch.tensor(rng.random((d,)).astype(np.float16), dtype=torch.bfloat16).to(device)
K_cache = torch.tensor(rng.random((B, 1, S_prior, d)).astype(np.float16), dtype=torch.bfloat16).to(device)
V_cache = torch.tensor(rng.random((B, 1, S_prior, d)).astype(np.float16), dtype=torch.bfloat16).to(device)
cos = torch.tensor(rng.random((B, d)).astype(np.float16), dtype=torch.bfloat16).to(device)
sin = torch.tensor(rng.random((B, d)).astype(np.float16), dtype=torch.bfloat16).to(device)
pos = torch.tensor([[320]], dtype=torch.int32).to(device)

v12a_bench(hidden, Wq, Wk, Wv, Wo, q_nw, k_nw, K_cache, V_cache, cos, sin, pos)

r = v12a_bench.last_result
if r:
    print(f"=== v12a Benchmark Results ===")
    print(f"device_time_us       = {r.device_time_us:.2f}")
    print(f"tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
    print(f"dma_active_pct       = {r.dma_active_pct:.1f}%")
    print(f"spill_bytes          = {r.spill_bytes}")
    print(f"mfu_estimated        = {r.prof.get('mfu_estimated_percent', 0):.2f}%")
    print(f"hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"hbm_write_KiB        = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")
else:
    print("Benchmark result not available (NTFF not captured)")
