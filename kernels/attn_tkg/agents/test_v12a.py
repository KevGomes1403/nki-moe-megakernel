import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import numpy as np
import torch
import torch_xla.core.xla_model as xm

# Import both kernels
from v10e import qwen3_attn_tkg_fused_oproj_v10e
from v12a import qwen3_attn_tkg_fused_oproj_v12a

device = xm.xla_device()
rng = np.random.default_rng(42)

B, H, d, S_prior = 1, 2048, 128, 640
Hq_out = 1024

def make_inputs():
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
    return hidden, Wq, Wk, Wv, Wo, q_nw, k_nw, K_cache, V_cache, cos, sin, pos

inputs = make_inputs()
ref_out, ref_k, ref_v = qwen3_attn_tkg_fused_oproj_v10e(*inputs)
xm.mark_step()

new_out, new_k, new_v = qwen3_attn_tkg_fused_oproj_v12a(*inputs)
xm.mark_step()

ref_out_np = ref_out.float().cpu().numpy()
new_out_np = new_out.float().cpu().numpy()
np.testing.assert_allclose(new_out_np, ref_out_np, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(new_k.float().cpu().numpy(),
                            ref_k.float().cpu().numpy(), rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(new_v.float().cpu().numpy(),
                            ref_v.float().cpu().numpy(), rtol=1e-3, atol=1e-3)
print(f"max_diff_out={np.abs(new_out_np - ref_out_np).max():.2e}  PASS")
