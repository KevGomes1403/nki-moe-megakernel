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

# Try hwdge path: src.shape[0] == 16, src.shape[-1] % 128 == 0
# K_cache has shape [1, 1, 640, 128]; rearrange so dim0=16
# Reshape [1, 1, 640, 128] -> [16, 1, 40, 128] ?
# No - we need [16, ?, ?, ?] for hwdge

# Alternative: reshape K_cache to [16, S_prior//16, 128] ... no that's 3D

# Try 4D: [16, 1, 8, 128] -> hwdge permutation [3,1,2,0] -> dst [128, 1, 8, 16]
# That reads 16*8=128 rows of d=128 columns — exactly one tile!

@nki.jit
def test_dma_t_hwdge(K_cache):
    # K_cache: [1, 1, 640, 128]
    # Reshape to get [16, 1, 8, 128] for one tile of 128 rows
    # [1,1,640,128] -> need to view as [16, 1, 8, 128] for first tile
    # This is tricky — let's try with a reshape
    K_4d = K_cache.reshape((16, 1, 40, 128))  # 16 * 40 = 640 rows total
    # Slice first 8 "chunks" of 16: K_4d[0:16, 0:1, 0:8, :] -> [16, 1, 8, 128]
    # hwdge permutation [3,1,2,0]: dst[d, 1, chunk, row16] = src[row16, 1, chunk, d]
    # dst shape: [128, 1, 8, 16]
    k_ct_4d = nl.ndarray((PMAX, 1, 8, 16), dtype=nl.bfloat16, buffer=nl.sbuf, name='k_ct_4d')
    nisa.dma_transpose(dst=k_ct_4d, src=K_4d[0:16, 0:1, 0:8, :])
    # After reshape [128, 1, 8, 16] -> [128, 128]
    k_ct = k_ct_4d.reshape((PMAX, PMAX))
    out = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.shared_hbm, name='out')
    nisa.dma_copy(dst=out, src=k_ct)
    return out

rng = np.random.default_rng(42)
K_np = rng.random((1, 1, 640, 128)).astype(np.float32)
K = torch.from_numpy(K_np).to(torch.bfloat16).to(device)
try:
    result = test_dma_t_hwdge(K)
    xm.mark_step()
    r_np = result.cpu().float().numpy()
    # Expected: K_np[0,0,0:128,:].T (first 128 rows transposed)
    expected = K_np[0,0,0:128,:].T
    print('hwdge 4D OK, shape:', result.shape)
    expected = K_np[0,0,0:128,:].T  # shape [128, 128]: expected[d,s] = K_orig[0,0,s,d]
    max_diff = np.abs(r_np - expected).max()
    if max_diff < 0.01:
        print('Transpose is CORRECT')
    else:
        print(f'Transpose WRONG, max_diff={max_diff:.4f}')
        print('r_np[0,0:8]:', r_np[0, 0:8])
        print('expected[0,0:8]:', expected[0, 0:8])
        # Print what it actually IS
        K_2d = K_np[0,0,:,:]  # [640, 128]
        # Find where r_np[0,0:8] matches K_2d
        for s_try in range(8):
            if abs(r_np[0,0] - K_2d[s_try, 0]) < 0.01:
                print(f'r_np[0,0] matches K_2d[{s_try}, 0]')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'hwdge 4D ERROR: {str(e)[:800]}')
