"""Try various dma_transpose shapes to find what works for K-cache transpose."""
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

# Use sequential values to check layout
K_np = np.arange(640*128, dtype=np.float32).reshape(1, 1, 640, 128)
K = torch.from_numpy(K_np).to(torch.bfloat16).to(device)

# Expected: out[d, s] = K[0,0,s,d]
expected = K_np[0,0,0:128,:].T.astype(np.float32)

# Approach: src [16, 1, 128//16, 128] = [16, 1, 8, 128] with hwdge
# Wait - hwdge: src[0]=16 rows, src[-1]=128 columns
# Permutation [3,1,2,0]: dst[d, 0, chunk, r] = src[r, 0, chunk, d]
# src[r=row_in_16, 0, chunk, d] -- but this is reading K in which order?
# We need K[0,0, s, d] where s = r + chunk*16 (but due to reshape, it goes s = chunk + r*40 ???)
#
# The issue: reshape [1,1,640,128] -> [16,1,40,128] lays out rows in fast-row-first order
# Row 0 -> K_4d[0,0,0,:]
# Row 1 -> K_4d[1,0,0,:]
# ...
# Row 15 -> K_4d[15,0,0,:]
# Row 16 -> K_4d[0,0,1,:]   ← chunk 1 starts here
# Row 17 -> K_4d[1,0,1,:]
# So K_4d[r, 0, chunk, d] = K_orig[0,0, r + chunk*16, d]  ✓

# But the output: dst[d, 0, chunk, r] has this layout when reshaped [128,1,8,16] -> [128,128]:
# element at flat [d*128 + chunk*16 + r] corresponds to [d, chunk*16+r]
# So r_np[d, chunk*16+r] = K_orig[0,0, r+chunk*16, d] = K_orig[0, 0, (chunk*16+r), d]
# Let s = chunk*16 + r, then r_np[d, s] = K_orig[0, 0, s, d] ← CORRECT!
# But we see r_np[0, 1] = K_orig[0, 0, 40, 0] which means j=1 → s=40 not s=1
# This means the actual reshape is chunk-outer, not r-outer in memory

# The dst ndarray [128,1,8,16] in NKI SBUF may have dimension ordering different
# from numpy C-order. In NKI, the SBUF layout is [partition, free] = [128, 128]
# The shape [128,1,8,16] → [128,128] reshape: the "partition" dim is 128 (first),
# "free" dims are 1*8*16=128. In NKI's memory model, the free dimension is contiguous.
# When we say [128,1,8,16], the contiguous ordering in free space is: 16 (fastest), then 8, then 1.
# So flat free index = chunk*16 + r_in_chunk → that IS chunk*16+r = s.
# BUT: the transpose writes dst[d,0,chunk,r] and dst has SBUF layout [par=128, free=1*8*16=128]
# In NKI SBUF, [par, f1, f2, f3] means:
#   - partition = d
#   - free = f1*8*16 + f2*16 + f3 for shape [d, 1, 8, 16]
#   Actually free dimensions are contiguous: [1, 8, 16] - the total free is 128
#   free_index = 0*(8*16) + chunk*16 + r = chunk*16 + r

# If the hardware writes dst[d, 0, chunk, r] and the free ordering is [f1*8*16 + f2*16 + f3]:
# Then the resulting 2D view [par=d, free=chunk*16+r] should give r_np[d, chunk*16+r] = K[0,0,r+chunk*16,d]
# r_np[0, 1] should be K[0,0,1,0] = 128 but we get 5120 = K[0,0,40,0]

# 40 = 5120/128 = row 40; and 40 = 1*40 = chunk=1, r=0 in the 40-chunk (8-wide) version
# Wait: 40 in original K = K_4d[0, 0, 2, 0] (chunk=2? r=0+40mod16=8, chunk=40//16=2?)
# No: 40 mod 16 = 8, 40 // 16 = 2. So K_orig[0,0,40,:] = K_4d[8,0,2,:]
# But r_np[0,1] = K_orig[0,0,40,0] = K_4d[8,0,2,0]
# dst[0, 0, chunk=2, r=8] = K_4d[8, 0, 2, 0] → free = 0*(8*16) + 2*16 + 8 = 40
# So r_np[0, 40] = K_4d[8,0,2,0] = K_orig[0,0,40,0] -- but we see it at r_np[0,1]!

# This means the actual free index of dst[d=0, 0, chunk=2, r=8] is 1, not 40.
# That implies the free ordering is [r, chunk] not [chunk, r] -- i.e., r is SLOWER varying
# and chunk is FASTER varying. But that contradicts normal C-order...

# Maybe NKI stores the free dimensions differently. Let's test with all-ones except one element.
print("Testing layout with single-element probe...")

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

# r_np[d, j] = K_orig[0, 0, s, d] for what value of s?
# Find s: r_np[0, j] / 128 = s (since K_orig[0,0,s,0] = s*128)
# r_np[0, j] = row_s * 128, so row_s = r_np[0, j] / 128
print("r_np[0, 0:10] / 128 (row mapping for d=0):")
print(r_np[0, 0:10] / 128)
print("Expected row mapping: [0,1,2,...,9]")
print()
print("r_np[1, 0:5] - 1 (to get row for d=1):")
# K_orig[0,0,s,1] = s*128 + 1
print((r_np[1, 0:5] - 1) / 128)
print()
# The stride in j for same d:
# If r_np[0, j] = j*40*128, then stride = 40
stride = (r_np[0, 1] - r_np[0, 0]) / 128
print(f"Stride in j (rows apart): {stride}")
