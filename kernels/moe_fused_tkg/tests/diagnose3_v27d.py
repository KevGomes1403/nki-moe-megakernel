"""
Minimal diagnostic: simulate kernel v27d math in PyTorch step by step,
comparing against the reference to isolate which stage introduces error.

Focus: reproduce exactly what the kernel computes for a single expert.
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
P = 128  # partition dim
H_FREE = H // P  # = 16
H_SHARD = H // 2  # = 1024 (prg_id=0)
H_FREE_SHARD = H_SHARD // P  # = 8
GU_FLAT = 2 * I  # = 384
scale = 0.1
SEED = 0
prg_id = 0  # simulate first program

torch.manual_seed(SEED)
inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16)
gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16)
gate_up_flat = torch.zeros(E, H, 2 * I, dtype=torch.bfloat16)
gate_up_flat[:, :, 0:I]   = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
gate_up_flat[:, :, I:2*I] = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
down_w = (torch.randn(E, I, H) * scale).to(torch.bfloat16)
gate_up_4d = torch.stack([gate_up_flat[:, :, 0:I], gate_up_flat[:, :, I:2*I]], dim=2)

# ─── Reference internal computation ─────────────────────────────────────────
x = inp.reshape(H).float()
g = gamma.reshape(H).float()
x_norm_f32 = x * torch.rsqrt(x.pow(2).mean() + 1e-6) * g
x_norm_bf16 = x_norm_f32.to(torch.bfloat16)

logits = x_norm_bf16.float() @ router_w.float()
affinities = torch.softmax(logits, dim=-1)
topk_w, topk_idx = torch.topk(affinities, k=K, dim=-1)
topk_w = topk_w / topk_w.sum()

print("=== Reference routing ===")
print(f"  top8 experts: {topk_idx.tolist()}")
print(f"  top8 weights: {[f'{w:.4f}' for w in topk_w.tolist()]}")

# ─── Simulate kernel's gate/up matmul layout ─────────────────────────────────
# gate_up_bufs[k][p, h1, c] = gate_up_flat[e, h1*128+p, c]
# nc_matmul result (accumulated over h1):
#   gate_psum[i, 0] = sum_{h1} sum_p gate_up_flat[e, h1*128+p, i] * x_norm_bf16[h1*128+p]
#   = sum_{h=0}^{H-1} gate_up_flat[e, h, i] * x_norm_bf16[h]  for i in [0, I0)
# Tile 1: same but i in [I0, I) mapped to gate_t1_128[p, h1, i-I0] for i in [0, I1)

e0 = topk_idx[0].item()
print(f"\n=== Verifying gate/up matmul for expert {e0} ===")

gate_w = gate_up_4d[e0, :, 0, :]   # [H, I]
up_w   = gate_up_4d[e0, :, 1, :]   # [H, I]

# Full reference gate/up output
gate_full = (x_norm_bf16.unsqueeze(0) @ gate_w).squeeze(0).float()  # [I]
up_full   = (x_norm_bf16.unsqueeze(0) @ up_w).squeeze(0).float()    # [I]

# Kernel tile 0 gate: cols 0:I0
# nc_matmul: sum_p(gate_up_flat[e, h1*128+p, i] * x_norm_bf16[h1*128+p]) for i in [0,I0)
# = x_norm_bf16 @ gate_w[:, 0:I0]
gate_tile0 = (x_norm_bf16.unsqueeze(0) @ gate_w[:, 0:I0]).squeeze(0).float()

# Kernel tile 1 gate: cols I0:I (=192-128=64 valid)
# gate_t1_128[p, h1, 0:I1] = gate_up_flat[e, h1*128+p, I0:I0+I1]
# nc_matmul: psum[i, 0] = sum_p(gate_t1_128[p, h1, i] * x_norm[p, h1]) for i in [0, I0)
#            = gate_full[I0+i] for i in [0, I1), 0 for i in [I1, I0)
gate_tile1 = (x_norm_bf16.unsqueeze(0) @ gate_w[:, I0:I]).squeeze(0).float()  # [I1=64]
gate_tile1_padded = torch.zeros(I0)
gate_tile1_padded[:I1] = gate_tile1

print(f"  gate_full[:4]    = {gate_full[:4].tolist()}")
print(f"  gate_tile0[:4]   = {gate_tile0[:4].tolist()}")
print(f"  gate_tile0[I0:I0+4] would be I1: {gate_full[I0:I0+4].tolist()}")
print(f"  gate_tile1[:4]   = {gate_tile1[:4].tolist()}")
print(f"  gate_tile1 vs gate_full[I0:]: max_diff = {(gate_tile1 - gate_full[I0:]).abs().max():.4e}")

# ─── Simulate SiLU + inter ─────────────────────────────────────────────────
# inter_f32 = silu(gate_psum) * up_psum -- shape [P=128, I_tiles=2]
# col 0 = silu(gate_tile0[0:P]) * up_tile0[0:P]
#       = silu(gate_full[0:128]) * up_full[0:128]
# col 1 = silu(gate_tile1_padded[0:P]) * up_tile1_padded[0:P]
#       = silu(gate_full[128:192] padded to 128) * up_full[128:192] padded to 128

# The PSUM layout: gate_up_psum[P=128, K_WAVE*2*I_tiles]
# For expert k=0, gu_base=0:
#   col 0: gate tile0 → psum[0:128, 0] = gate_full[0:128]
#   col 1: gate tile1 → psum[0:128, 1] = gate_full[128:192] (first 64 valid, 64 zeros)
#   col 2: up tile0  → psum[0:128, 2] = up_full[0:128]
#   col 3: up tile1  → psum[0:128, 3] = up_full[128:192] (first 64 valid, 64 zeros)

gate_psum_col0 = gate_full[0:P]             # [128]
gate_psum_col1 = torch.zeros(P)
gate_psum_col1[:I1] = gate_full[I0:I]       # [64 valid, 64 zero]

up_psum_col2 = up_full[0:P]                 # [128]
up_psum_col3 = torch.zeros(P)
up_psum_col3[:I1] = up_full[I0:I]          # [64 valid, 64 zero]

# SiLU + multiply
silu_col0 = F.silu(gate_psum_col0)
silu_col1 = F.silu(gate_psum_col1)
inter_col0 = silu_col0 * up_psum_col2       # [128]
inter_col1 = silu_col1 * up_psum_col3       # [128, but 64:128 = silu(0)*0 = 0]

# Cast to bf16
inter_bf16_col0 = inter_col0.to(torch.bfloat16).float()
inter_bf16_col1 = inter_col1.to(torch.bfloat16).float()

# Reference inter:
inter_ref = (F.silu(gate_full) * up_full).to(torch.bfloat16).float()  # [I=192]

print(f"\n=== Verifying inter (SiLU * up) for expert {e0} ===")
print(f"  inter_ref[0:4]    = {inter_ref[0:4].tolist()}")
print(f"  inter_col0[0:4]   = {inter_bf16_col0[0:4].tolist()}")
print(f"  inter_ref[128:132]= {inter_ref[I0:I0+4].tolist()}")
print(f"  inter_col1[0:4]   = {inter_bf16_col1[0:4].tolist()}")

# Check: do col0 and col1 reconstruct inter_ref?
inter_kernel = torch.cat([inter_bf16_col0, inter_bf16_col1[:I1]])  # [192]
print(f"  inter_kernel vs inter_ref max_diff = {(inter_kernel - inter_ref[:I]).abs().max():.4e}")

# ─── Simulate down matmul ────────────────────────────────────────────────────
# down_full0[p, h] = down_w[e, p, prg_id*H_SHARD + h]  for p in [0, I0), h in [0, H_SHARD)
# down_full1[p, h] = down_w[e, I0+p, prg_id*H_SHARD + h] for p in [0, I1), h in [0, H_SHARD)
# (rows I1:I0 zero-padded)

d_w = down_w[e0]  # [I=192, H=2048]
d_full0 = d_w[0:I0, prg_id*H_SHARD:(prg_id+1)*H_SHARD]   # [128, 1024]
d_full1 = torch.zeros(I0, H_SHARD)
d_full1[0:I1, :] = d_w[I0:I0+I1, prg_id*H_SHARD:(prg_id+1)*H_SHARD]  # [64 valid, 64 zero rows]

# nc_matmul: for each h1_out tile (P=128 output neurons):
#   stationary = d_full0[0:P, h1_out*P:(h1_out+1)*P]  -- [128, 128]
#   moving = inter_bf16[0:P, 0:1]                       -- [128, 1]
#   dst[i, 0] = sum_p(d_full0[p, h1_out*128+i] * inter_bf16_col0[p])
#             = sum_{j=0}^{127} down_w[e, j, h1_out*128+i] * inter_bf16_col0[j]
# Second matmul:
#   dst[i, 0] += sum_p(d_full1[p, h1_out*128+i] * inter_bf16_col1[p])
#              = sum_{j=0}^{63} down_w[e, I0+j, h1_out*128+i] * inter_bf16_col1[j]

# Compute simulated kernel down output for prg_id=0 (first 1024 outputs):
down_kernel_sim = torch.zeros(H_SHARD)
for h1_out in range(H_FREE_SHARD):
    for i in range(P):
        h_out = h1_out * P + i
        # First tile
        val0 = sum(d_full0[p, h1_out*P + i].item() * inter_bf16_col0[p].item() for p in range(P))
        # Second tile
        val1 = sum(d_full1[p, h1_out*P + i].item() * inter_bf16_col1[p].item() for p in range(P))
        down_kernel_sim[h_out] = val0 + val1

# Reference down output:
d_w_ref = down_w[e0]  # [I, H]
out_ref = (inter_ref[:I].to(torch.bfloat16).unsqueeze(0) @ d_w_ref).squeeze(0).float()  # [H]
out_ref_shard = out_ref[prg_id*H_SHARD:(prg_id+1)*H_SHARD]  # [H_SHARD=1024]

print(f"\n=== Verifying down matmul for expert {e0} (prg_id=0, first 1024 outputs) ===")
print(f"  out_ref_shard[:8]        = {out_ref_shard[:8].tolist()}")
print(f"  down_kernel_sim[:8]      = {down_kernel_sim[:8].tolist()}")
down_diff = (out_ref_shard.float() - down_kernel_sim.float()).abs()
print(f"  max_diff = {down_diff.max():.4e}")
print(f"  mean_diff = {down_diff.mean():.4e}")

# ─── Full pipeline sim with affinity scaling ─────────────────────────────────
print(f"\n=== Full pipeline simulation (single expert {e0}, weight 1.0) ===")
# If the down matmul is correct, scale by affinity and check against reference
aff0 = topk_w[0].item()
out_scaled = down_kernel_sim * aff0
# Reference with single expert weight 1.0:
out_single_ref = out_ref_shard * aff0
diff_single = (out_single_ref - out_scaled).abs()
print(f"  max_diff (affinty-scaled, shard 0) = {diff_single.max():.4e}")
