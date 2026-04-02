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

# K_cache [1, 1, 640, 128] original shape
# Test 3D reshape: [640, 1, 128] - take slice [s*128:(s+1)*128, 0:1, :] -> shape [128, 1, 128]
# dma_transpose with 3D: permutation [2, 1, 0]
# dst: [128, 1, 128] -> permuted: dst[i,j,k] = src[k,j,i]
# dst[d_head, 0, s_pos] = src[s_pos, 0, d_head]
# That's the transpose we want! -> dst shape [128, 1, 128]
# Then reshape to [128, 128]

@nki.jit
def test_dma_t_3d_reshape(K_cache):
    # K_cache: [1, 1, 640, 128]
    K_3d = K_cache.reshape((640, 1, 128))  # [S_prior, 1, d]
    # dst: [d=128, 1, PMAX=128] -> after [2,1,0] permute from [PMAX=128, 1, d=128]
    k_ct_3d = nl.ndarray((PMAX, 1, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name='k_ct_3d')
    nisa.dma_transpose(dst=k_ct_3d, src=K_3d[0:PMAX, 0:1, :])
    k_ct = k_ct_3d.reshape((PMAX, PMAX))
    out = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.shared_hbm, name='out')
    nisa.dma_copy(dst=out, src=k_ct)
    return out

rng = np.random.default_rng(42)
K_np = rng.random((1, 1, 640, 128)).astype(np.float16)
K = torch.tensor(K_np, dtype=torch.bfloat16).to(device)
try:
    result = test_dma_t_3d_reshape(K)
    xm.mark_step()
    r_np = result.cpu().numpy()
    # K_np[0,0,0:128,d] should equal r_np[d, s] -> r_np[d, s] = K_np[0,0,s,d]
    expected = K_np[0,0,0:128,:].T  # [128, 128] transposed
    print('3D reshape OK, shape:', result.shape)
    if np.allclose(r_np.astype(np.float32), expected.astype(np.float32), atol=0.01):
        print('Transpose is CORRECT')
    else:
        max_diff = np.abs(r_np.astype(np.float32) - expected.astype(np.float32)).max()
        print(f'Transpose WRONG, max_diff={max_diff:.4f}')
        print('r_np[0,:5]:', r_np[0,:5])
        print('expected[0,:5]:', expected[0,:5])
except Exception as e:
    print(f'3D reshape ERROR: {str(e)[:600]}')
