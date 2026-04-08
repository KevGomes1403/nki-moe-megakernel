"""Benchmark v5: free-dim packing. No o_proj; Hkv_tp=4."""
import os
import sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v5"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import wrap_benchmark

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from v5_freedim_packing import qwen3_attn_tkg_fused

device = xm.xla_device()
rng = np.random.default_rng(42)
# v0-v5: Hkv_tp=4, no Wo; different from v6+ (Hkv_tp=1, with Wo)
B, H, d, S_prior = 1, 2048, 128, 640
Hq_out = 1024   # Hq_tp=8, d=128
Hkv_out = 512   # Hkv_tp=4, d=128

hidden = torch.tensor(rng.random((B, 1, H)).astype(np.float16), dtype=torch.bfloat16).to(device)
Wq = torch.tensor(rng.random((Hq_out, H)).astype(np.float16), dtype=torch.bfloat16).to(device)
Wk = torch.tensor(rng.random((Hkv_out, H)).astype(np.float16), dtype=torch.bfloat16).to(device)
Wv = torch.tensor(rng.random((Hkv_out, H)).astype(np.float16), dtype=torch.bfloat16).to(device)
q_nw = torch.tensor(rng.random((d,)).astype(np.float16), dtype=torch.bfloat16).to(device)
k_nw = torch.tensor(rng.random((d,)).astype(np.float16), dtype=torch.bfloat16).to(device)
K_cache = torch.tensor(rng.random((B, 4, S_prior, d)).astype(np.float16), dtype=torch.bfloat16).to(device)
V_cache = torch.tensor(rng.random((B, 4, S_prior, d)).astype(np.float16), dtype=torch.bfloat16).to(device)
cos = torch.tensor(rng.random((B, d)).astype(np.float16), dtype=torch.bfloat16).to(device)
sin = torch.tensor(rng.random((B, d)).astype(np.float16), dtype=torch.bfloat16).to(device)
pos = torch.tensor([[320]], dtype=torch.int32).to(device)

kernel = wrap_benchmark(qwen3_attn_tkg_fused, warmup=5, iters=50)
kernel(hidden, Wq, Wk, Wv, q_nw, k_nw, K_cache, V_cache, cos, sin, pos)

r = kernel.last_result
if r:
    print(f"device_time_us       = {r.device_time_us:.2f}")
    print(f"tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
    print(f"dma_active_pct       = {r.dma_active_pct:.1f}%")
    print(f"spill_bytes          = {r.spill_bytes}")
    print(f"mfu_estimated        = {r.prof.get('mfu_estimated_percent', 0):.2f}%")
    print(f"hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"hbm_write_KiB        = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")
else:
    print("ERROR: last_result is None — NTFF not captured")
