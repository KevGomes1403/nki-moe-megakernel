"""Test the original baseline kernel to verify it produces correct results."""
import numpy as np
import torch
import torch_xla.core.xla_model as xm
from qwen3_router_topk_cte_original import qwen3_router_topk_cte_original

import os 

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_CC_FLAGS"] = " "
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"]= "1"
os.environ["XLA_HLO_DEBUG"]= "1"
os.environ["NEURON_RT_INSPECT_ENABLE"]= "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"]= "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]= "./output"

H = 2048; E = 128; K = 8
T = 640
rng = np.random.default_rng(42)
x_np = rng.standard_normal((T, H)).astype(np.float16)
w_np = rng.standard_normal((H, E)).astype(np.float16)

ref = x_np.astype(np.float32) @ w_np.astype(np.float32)
print("Ref[0,:5]:", ref[0, :5])

device = xm.xla_device()
rl = torch.zeros((T, E), dtype=torch.float32, device=device)
ea = torch.zeros((T, E), dtype=torch.float32, device=device)
ei = torch.zeros((T, K), dtype=torch.int32, device=device)
x_th = torch.tensor(x_np, dtype=torch.bfloat16, device=device)
w_t = torch.tensor(w_np, dtype=torch.bfloat16, device=device)

print("Running original baseline kernel (x: [T, H])...")
qwen3_router_topk_cte_original[2](x_th, w_t, rl, ea, ei)
xm.mark_step()
print("rl[0,:5]:", rl.cpu()[0, :5].tolist())
print("rl max abs diff:", abs(rl.cpu().numpy() - ref).max())
print("ea sum[0]:", ea.cpu()[0].sum().item())
print("ei[0]:", ei.cpu()[0].tolist())
