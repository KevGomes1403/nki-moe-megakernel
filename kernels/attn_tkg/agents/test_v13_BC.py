"""
v13_BC correctness test: compare v13_BC vs v12e.
rtol=1e-3, atol=1e-3.
"""
import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "./output/"
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from v12e import qwen3_attn_tkg_fused_oproj_v12e
from v13_BC import qwen3_attn_tkg_fused_oproj_v13bc

device = xm.xla_device()
rng = np.random.default_rng(42)
B, H, d, S_prior, Hq_out = 1, 2048, 128, 640, 1024

pos_val = 320
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
pos = torch.tensor([[pos_val]], dtype=torch.int32).to(device)

inputs = (hidden, Wq, Wk, Wv, Wo, q_nw, k_nw, K_cache, V_cache, cos, sin, pos)

ref_out, ref_k, ref_v = qwen3_attn_tkg_fused_oproj_v12e[2](*inputs)
new_out, new_k, new_v = qwen3_attn_tkg_fused_oproj_v13bc[2](*inputs)
xm.mark_step()

ro = ref_out.cpu().float().numpy()
rk = ref_k.cpu().float().numpy()
rv = ref_v.cpu().float().numpy()
no = new_out.cpu().float().numpy()
nk = new_k.cpu().float().numpy()
nv = new_v.cpu().float().numpy()

print("v13_BC vs v12e correctness:")
try:
    np.testing.assert_allclose(no, ro, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(nk, rk, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(nv, rv, rtol=1e-3, atol=1e-3)
    max_diff_out = np.abs(no - ro).max()
    max_diff_k = np.abs(nk - rk).max()
    max_diff_v = np.abs(nv - rv).max()
    print(f"  output max_diff={max_diff_out:.2e}  PASS")
    print(f"  k_rope max_diff={max_diff_k:.2e}  PASS")
    print(f"  v_out  max_diff={max_diff_v:.2e}  PASS")
    print(f"\nCORRECTNESS: PASS  max_diff={max_diff_out:.2e}")
except AssertionError as e:
    print(f"  FAIL: {e}")
    raise
