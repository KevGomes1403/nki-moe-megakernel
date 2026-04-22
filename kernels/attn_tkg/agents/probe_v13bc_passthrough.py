"""Zero-output diagnostic for v13bc_sbm_tiled — same pattern as v17_fast_exp."""
import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
import sys
sys.path.insert(0, "/home/ubuntu/nki-moe")
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import nki
import nki.language as nl
import nki.isa as nisa
from nkilib.core.utils.allocator import SbufManager
from kernels.attn_tkg.agents.v13bc_sbm_tiled import qwen3_attn_tkg_fused_oproj_v13bc


@nki.jit
def v13bc_passthrough(hs, Wq, Wk, Wv, Wo, qn, kn, Kc, Vc, cos, sin, pid):
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("wrapper_outer")
    o, _, _ = qwen3_attn_tkg_fused_oproj_v13bc(hs, Wq, Wk, Wv, Wo, qn, kn, Kc, Vc, cos, sin, pid,
                                                out_sb=None, sbm=sbm)
    h = nl.ndarray((128, 16), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=h, src=o, dge_mode=nisa.dge_mode.hwdge)
    sbm.close_scope()
    sbm.close_scope()
    return h


torch.manual_seed(42)
dev = xm.xla_device()
hs = torch.randn(1, 1, 2048, dtype=torch.bfloat16).to(dev)
Wq = (torch.randn(1024, 2048, dtype=torch.bfloat16) * 0.02).to(dev)
Wk = (torch.randn(128, 2048, dtype=torch.bfloat16) * 0.02).to(dev)
Wv = (torch.randn(128, 2048, dtype=torch.bfloat16) * 0.02).to(dev)
Wo = (torch.randn(1024, 2048, dtype=torch.bfloat16) * 0.02).to(dev)
qn = torch.ones(128, dtype=torch.bfloat16).to(dev)
kn = torch.ones(128, dtype=torch.bfloat16).to(dev)
Kc = (torch.randn(1, 1, 640, 128, dtype=torch.bfloat16) * 0.1).to(dev)
Vc = (torch.randn(1, 1, 640, 128, dtype=torch.bfloat16) * 0.1).to(dev)
co = torch.cos(torch.linspace(0, 1, 128)).reshape(1, 128).to(torch.bfloat16).to(dev)
si = torch.sin(torch.linspace(0, 1, 128)).reshape(1, 128).to(torch.bfloat16).to(dev)
pi = torch.tensor([[320]], dtype=torch.int32).to(dev)

o = v13bc_passthrough(hs, Wq, Wk, Wv, Wo, qn, kn, Kc, Vc, co, si, pi)
r = o.cpu().float().numpy()
print(f"v13bc out shape={r.shape} range=[{r.min():.6f}, {r.max():.6f}] absmean={abs(r).mean():.4e}")
assert abs(r).mean() > 1e-4, "v13bc_sbm_tiled ALSO produces zeros — attention baseline is fiction"
print("v13bc_sbm_tiled output is non-zero")
