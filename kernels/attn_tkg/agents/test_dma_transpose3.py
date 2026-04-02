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

@nki.jit
def test_dma_t_hwdge(K_cache):
    # K_cache: [1, 1, 640, 128]
    K_4d = K_cache.reshape((16, 1, 40, 128))
    k_ct_4d = nl.ndarray((PMAX, 1, 8, 16), dtype=nl.bfloat16, buffer=nl.sbuf, name='k_ct_4d')
    nisa.dma_transpose(dst=k_ct_4d, src=K_4d[0:16, 0:1, 0:8, :])
    k_ct = k_ct_4d.reshape((PMAX, PMAX))
    out = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.shared_hbm, name='out')
    nisa.dma_copy(dst=out, src=k_ct)
    return out

# Use simple sequential values to see the mapping clearly
K_np = np.arange(640*128, dtype=np.float32).reshape(1, 1, 640, 128)
K = torch.from_numpy(K_np).to(torch.bfloat16).to(device)
result = test_dma_t_hwdge(K)
xm.mark_step()
r_np = result.cpu().float().numpy()

print("Input K_np[0,0, 0:4, 0:4] (first 4 rows, first 4 cols):")
print(K_np[0,0,0:4,0:4])
print()
print("Expected transpose K_np[0,0, 0:128, :].T [0:4, 0:4] (dst[d,s]=K[0,0,s,d]):")
expected = K_np[0,0,0:128,:].T
print(expected[0:4, 0:4])
print()
print("Actual result r_np[0:4, 0:4]:")
print(r_np[0:4, 0:4])
print()
print(f"max_diff_vs_expected: {np.abs(r_np - expected).max():.2f}")
# What does r_np[0,:] look like?
print("r_np[0, 0:8]:", r_np[0, 0:8])
print("r_np[1, 0:8]:", r_np[1, 0:8])
# Check if it's raw (untransposed)
raw = K_np[0,0,0:128,0:128]
print(f"max_diff_vs_raw: {np.abs(r_np - raw).max():.2f}")
