"""
Diagnostic: is the v17 sub-function populating out_sb at all when called
from a separate wrapper? Replicate bench_v17_fast_exp.attn_kernel_wrapper
pattern exactly — NO transpose, NO reshape — and check that output has data.
"""
import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import math
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import sys
sys.path.insert(0, "/home/ubuntu/nki-moe")

import nki
import nki.language as nl
import nki.isa as nisa
from nkilib.core.utils.allocator import SbufManager
from kernels.attn_tkg.agents.v17_fast_exp import qwen3_attn_tkg_fused_oproj_v13bc


@nki.jit
def passthrough_wrapper(hidden_states, Wq, Wk, Wv, Wo,
                         q_norm_weight, k_norm_weight,
                         K_cache, V_cache, cos, sin, position_ids):
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("wrapper_outer")
    out_sb, k_rope_out, v_out = qwen3_attn_tkg_fused_oproj_v13bc(
        hidden_states, Wq, Wk, Wv, Wo,
        q_norm_weight, k_norm_weight,
        K_cache, V_cache, cos, sin, position_ids,
        out_sb=None, sbm=sbm,
    )
    out_hbm = nl.ndarray((128, 16), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out_hbm, src=out_sb, dge_mode=nisa.dge_mode.hwdge)
    sbm.close_scope()
    sbm.close_scope()
    return out_hbm, k_rope_out, v_out


B, H, d, Hq_tp, S_prior, H_wo, PMAX = 1, 2048, 128, 8, 640, 2048, 128
torch.manual_seed(42)
dev = xm.xla_device()
hs = torch.randn(B, 1, H, dtype=torch.bfloat16).to(dev)
Wq = (torch.randn(Hq_tp * d, H, dtype=torch.bfloat16) * 0.02).to(dev)
Wk = (torch.randn(d, H, dtype=torch.bfloat16) * 0.02).to(dev)
Wv = (torch.randn(d, H, dtype=torch.bfloat16) * 0.02).to(dev)
Wo = (torch.randn(Hq_tp * d, H_wo, dtype=torch.bfloat16) * 0.02).to(dev)
q_n = torch.ones(d, dtype=torch.bfloat16).to(dev)
k_n = torch.ones(d, dtype=torch.bfloat16).to(dev)
Kc = (torch.randn(B, 1, S_prior, d, dtype=torch.bfloat16) * 0.1).to(dev)
Vc = (torch.randn(B, 1, S_prior, d, dtype=torch.bfloat16) * 0.1).to(dev)
cos = torch.cos(torch.linspace(0, 1, d)).reshape(B, d).to(torch.bfloat16).to(dev)
sin = torch.sin(torch.linspace(0, 1, d)).reshape(B, d).to(torch.bfloat16).to(dev)
pid = torch.tensor([[S_prior // 2]], dtype=torch.int32).to(dev)

out, _, _ = passthrough_wrapper(hs, Wq, Wk, Wv, Wo, q_n, k_n, Kc, Vc, cos, sin, pid)
got = out.cpu().float().numpy()

print(f"out_sb (as-is) shape={got.shape} range=[{got.min():.6f}, {got.max():.6f}]  absmean={np.abs(got).mean():.4e}")
assert np.abs(got).mean() > 1e-4, f"[HARD FAIL] out_sb itself is zero (absmean={np.abs(got).mean():.2e}) — v17 sub-function is not populating it"
print("[PASSTHROUGH PROBE] PASS — out_sb has valid data. Transpose/store chain is the bug.")
