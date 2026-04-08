import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import numpy as np
import torch
import torch_xla.core.xla_model as xm

sys.path.insert(0, "/home/ubuntu/nki-moe")
from kernels.moe_fused_tkg.quantized.v5 import run as run_v5
from kernels.moe_fused_tkg.quantized.v6a import run as run_v6a

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

ref = run_v5(inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales)
xm.mark_step()
ref_np = ref.cpu().float().numpy()

result = run_v6a(inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales)
xm.mark_step()
result_np = result.cpu().float().numpy()

max_diff = np.abs(result_np - ref_np).max()
mean_diff = np.abs(result_np - ref_np).mean()
print(f"max_diff={max_diff:.4e}  mean_diff={mean_diff:.4e}")
np.testing.assert_allclose(result_np, ref_np, rtol=1e-1, atol=1e-1)
print("CORRECTNESS PASS")
