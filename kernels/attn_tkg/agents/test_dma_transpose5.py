import os
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'
os.environ['NEURON_LOGICAL_NC_CONFIG'] = '2'

import nki
import nki.language as nl
import nki.isa as nisa
import numpy as np
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
PMAX = 128

K_np = np.arange(640*128, dtype=np.float32).reshape(1, 1, 640, 128)
K = torch.from_numpy(K_np).to(torch.bfloat16).to(device)

@nki.jit
def probe_layout(K_cache):
    K_4d = K_cache.reshape((16, 1, 40, 128))
    k_ct_4d = nl.ndarray((PMAX, 1, 8, 16), dtype=nl.bfloat16, buffer=nl.sbuf, name='k_ct_4d')
    nisa.dma_transpose(dst=k_ct_4d, src=K_4d[0:16, 0:1, 0:8, :])
    k_ct = k_ct_4d.reshape((PMAX, PMAX))
    out = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.shared_hbm, name='out')
    nisa.dma_copy(dst=out, src=k_ct)
    return out

result = probe_layout(K)
xm.mark_step()
r_np = result.cpu().float().numpy()

print("Row mapping for d=0 (r_np[0, j] / 128 = row):")
for j in range(40):
    row = r_np[0, j] / 128
    print(f"  j={j:3d} -> row={int(row):4d}")
