"""
Detailed DMA pattern analysis for down_w loading in kernel_v27d.

The down_w tensor shape is [E=128, I=192, H=2048].
The kernel loads it with pattern=[[H, I0], [1, H_shard]].

Let's verify what elements actually get loaded.
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import importlib.util

B, H, E, K, I = 1, 2048, 128, 8, 192
I0, I1 = 128, 64
H_SHARD = 1024  # H // 2 (for prg_id 0)
H_FREE = H // 128  # = 16
scale = 0.1
SEED = 0

torch.manual_seed(SEED)

inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16)
gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16)
gate_up_flat = torch.zeros(E, H, 2 * I, dtype=torch.bfloat16)
gate_up_flat[:, :, 0:I]   = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
gate_up_flat[:, :, I:2*I] = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
down_w = (torch.randn(E, I, H) * scale).to(torch.bfloat16)
gate_up_4d = torch.stack([gate_up_flat[:, :, 0:I], gate_up_flat[:, :, I:2*I]], dim=2)

# DMA pattern analysis for down_full0:
# src = down_w.ap(
#     pattern=[[H, I0], [1, H_shard]],
#     offset=prg_id * H_shard,
#     scalar_offset=expert_id,
#     indirect_dim=0,
# )
# down_w shape [E, I, H] in memory: row-major, so element [e, i, h] is at
#   offset = e * I * H + i * H + h
# For expert e0, prg_id=0:
#   offset = e0 * I * H + 0 = base of expert e0's block
#   The ap() pattern [[H, I0], [1, H_shard]] with offset=0:
#     This describes a 2D view: I0 rows with stride H, H_shard cols with stride 1
#     Starting at element offset=prg_id*H_shard = 0
#   So element [row, col] = offset + row * H + col * 1
#     = e0*I*H + row * H + col  where row in [0, I0), col in [0, H_shard)
#     = down_w[e0, row, col]  -- CORRECT! rows 0:I0 = intermediate neurons 0..127

# down_full0_bufs[k] shape [P=128, H_shard=1024]
# DMA loads into [P=128, H_shard=1024]:
#   down_full0[p, h] corresponds to what element?
# DMA from pattern [[H, I0], [1, H_shard]]:
#   2D source with I0 "rows" (stride=H) and H_shard "cols" (stride=1)
# Destination [P, H_shard] with P=I0=128:
#   dest[p, h] = down_w[e0, p, h]  for p in [0, I0), h in [0, H_shard)
print("down_full0[p, h] = down_w[e, p, 0:H_shard] for p in 0..I0-1  ✓")
print("down_full1[p, h] = down_w[e, I0+p, 0:H_shard] for p in 0..I1-1  ✓")

# Now: nc_matmul for down:
# dst=down_psum[P, d_base+h1_out:d_base+h1_out+1] -- [P, 1]
# stationary=down_full0[P, h1_out*P : h1_out*P+P] -- [P, P]
# moving=inter_bf16[P, 0:1] -- [P, 1]
# nc_matmul(dst=[P,1], stationary=[P,P], moving=[P,1]):
#   dst[i, 0] += sum_p( stationary[p, i] * moving[p, 0] )
#              = sum_p( down_w[e, p, h1_out*128+i] * inter_bf16[p, 0] )
#              = sum_{j=0}^{I0-1} inter[j] * down_w[e, j, h1_out*128+i]
# This is the partial output sum (over first I0 intermediate dims) for output neuron h1_out*128+i.

# Full output:
# out[h] = sum_{j=0}^{I-1} inter[j] * down_w[e, j, h]
#        = sum_{j=0}^{I0-1} inter[j] * down_w[e, j, h]
#        + sum_{j=I0}^{I-1} inter[j] * down_w[e, j, h]
# The two nc_matmuls should give this split. Looks correct.

# So what IS wrong? Let me check the gate/up matmul stationary shape more carefully.
# gate_up_bufs[k] shape: [P=128, H_free=16, _GU_FLAT=384]
# For i_tile=0:
#   g_stat = gate_up_bufs[k][0:P, h1, 0:I0] -- shape [P, I0]  -- gate cols 0..127
#   u_stat = gate_up_bufs[k][0:P, h1, I:I+I0] -- shape [P, I0]  -- up cols 192..319
# nc_matmul(dst=[P, 1], stationary=[P, I0], moving=[P, T=1]):
#   dst[i, 0] += sum_p( stationary[p, i] * moving[p, 0] )
# BUT gate_up_bufs[k][p, h1, 0:I0] should be gate_w[h1*128+p, 0:I0].
# Let me verify the DMA pattern for gate_up_w:
# src = gate_up_w.ap(
#     pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
#     offset=0, scalar_offset=expert_id, indirect_dim=0
# )
# gate_up_w shape [E, H, GU_FLAT=384]:
#   element [e, h, c] at offset e*H*GU_FLAT + h*GU_FLAT + c
# 3D pattern [[GU_FLAT, P], [P*GU_FLAT, H_free], [1, GU_FLAT]]:
#   element [r0, r1, r2] at offset r0*GU_FLAT + r1*P*GU_FLAT + r2
#   r0 in [0, P), r1 in [0, H_free), r2 in [0, GU_FLAT)
# With offset=0 (for expert e0, scalar_offset handles the E dimension):
#   offset = e0 * H * GU_FLAT
#   element [r0, r1, r2] at: e0*H*GU_FLAT + r0*GU_FLAT + r1*P*GU_FLAT + r2
#   = gate_up_w[e0, r1*P + r0, r2]  (since H = H_free * P = 16 * 128)
# Destination [P, H_free, GU_FLAT]:
#   dst[p, h1, c] = gate_up_w[e0, h1*P + p, c]  = gate_up_w[e0, h1*128+p, c]  ✓

print("gate_up_bufs[k][p, h1, c] = gate_up_w[e, h1*128+p, c]  ✓")
print()

# So gate_up_bufs[k][p, h1, 0:I0] = gate_up_w[e, h1*128+p, 0:I0] = gate_w[h1*128+p, 0:I0]
# nc_matmul: gate_psum[i, 0] += sum_p( gate_w[h1*128+p, i] * x_norm[h1*128+p] )
# After all h1: gate_psum[i, 0] = sum_h( gate_w[h, i] * x_norm[h] ) = gate_out[i]  ✓

# Everything looks correct on paper. Let me check the actual numerical results by
# running a simplified single-expert kernel.

print("=== Checking RMSNorm ===")
x = inp.reshape(-1, H).float()
gamma_f = gamma.float().reshape(H)
rms = x.pow(2).mean(-1, keepdim=True)
x_norm = x * torch.rsqrt(rms + 1e-6) * gamma_f.reshape(1, H)
x_norm_bf16 = x_norm.to(torch.bfloat16)
print(f"  x[:8] = {inp.reshape(H)[:8].tolist()}")
print(f"  x_norm_bf16[:8] = {x_norm_bf16.reshape(H)[:8].tolist()}")

# Now compute gate output for expert 0 manually
e0 = 0
gate_w = gate_up_4d[e0, :, 0, :]   # [H, I]
up_w   = gate_up_4d[e0, :, 1, :]   # [H, I]
d_w    = down_w[e0]                 # [I, H]
xt     = x_norm_bf16[0].unsqueeze(0)  # [1, H]

gate = (xt @ gate_w).float()   # [1, I]
up   = (xt @ up_w).float()     # [1, I]
inter = F.silu(gate) * up
inter_bf16 = inter.to(torch.bfloat16)
out_e0 = (inter_bf16 @ d_w).float()[0]  # [H]

print(f"\n  Expert 0 output[:8] = {out_e0[:8].tolist()}")
print(f"  Expert 0 output norm = {out_e0.norm():.4f}")

# Now load module and inspect what the kernel actually stores
# We'll use a minimal kernel that only runs expert 0 with weight 1.0
# to isolate where the computation differs.
print("\n  (Kernel output is what was printed in diagnose1)")
print("  Compare expert 0 with weight 1: PyTorch should ~match kernel / K / norm_weight")
