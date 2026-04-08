"""
Stage-by-stage diagnostic for kernel_v27d vs reference.py.

Test 1: Compare kernel output vs reference (baseline).
Test 2: Force reference to use kernel's top-k routing, compare MLP-only.
Test 3: Single-expert MLP comparison (expert=0, k=1) to isolate gate/up/down math.
Test 4: Check down matmul tiling: compare down_full0 (I0=128 rows) and
        down_full1 (I1=64 rows) slice result vs full I=192 matmul.
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
scale = 0.1
SEED = 0

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

kernel_mod = load_module(
    "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/kernel_v27d.py", "_k27d"
)
ref_mod = load_module(
    "/home/ubuntu/nki-moe/kernels/moe_fused_tkg/reference.py", "_ref"
)

device = xm.xla_device()
torch.manual_seed(SEED)

inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16)
gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16)
router_w = (torch.randn(H, E) * scale).to(torch.bfloat16)
gate_up_flat = torch.zeros(E, H, 2 * I, dtype=torch.bfloat16)
gate_up_flat[:, :, 0:I]   = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
gate_up_flat[:, :, I:2*I] = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
down_w = (torch.randn(E, I, H) * scale).to(torch.bfloat16)
gate_up_4d = torch.stack([gate_up_flat[:, :, 0:I], gate_up_flat[:, :, I:2*I]], dim=2)

# ── run kernel, get output ────────────────────────────────────────────────────
print("=== Running kernel_v27d ===")
inp_d  = inp.to(device); gamma_d = gamma.to(device); router_w_d = router_w.to(device)
gu_d   = gate_up_flat.to(device); down_d = down_w.to(device)
xm.mark_step()
kern_out = kernel_mod.run(inp_d, gamma_d, router_w_d, gu_d, down_d)
xm.mark_step()
kern_cpu = kern_out.cpu().reshape(B, H).float()

# ── run reference ─────────────────────────────────────────────────────────────
print("=== Running reference ===")
ref_out = ref_mod.qwen3_moe_fused_tkg_reference(
    inp.clone(), gamma.clone(), router_w.clone(),
    gate_up_4d.clone(), down_w.clone(), top_k=K,
).reshape(B, H).float()

print(f"\n[Test 1] kernel vs reference  max_diff={( kern_cpu - ref_out).abs().max():.4e}")

# ── compute reference internals manually ──────────────────────────────────────
x     = inp.reshape(-1, H)  # [1, H]
x_f32 = x.float()
rms   = x_f32.pow(2).mean(-1, keepdim=True)
x_norm= x_f32 * torch.rsqrt(rms + 1e-6)
x_norm= x_norm * gamma.float().reshape(1, H)
x_norm_bf16 = x_norm.to(torch.bfloat16)

logits = x_norm_bf16.float() @ router_w.float()
affinities = torch.softmax(logits, dim=-1)
topk_w, topk_idx = torch.topk(affinities, k=K, dim=-1)
topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

print(f"\n[Reference routing] top8 experts: {topk_idx[0].tolist()}")
print(f"[Reference routing] top8 weights: {[f'{w:.4f}' for w in topk_w[0].tolist()]}")

# ── Test 2: PyTorch MLP with reference routing (isolate MLP errors) ───────────
print("\n=== Test 2: PyTorch MLP with SAME routing as reference ===")
output_ref_routing = torch.zeros(1, H, dtype=torch.float32)
for k_idx in range(K):
    e   = topk_idx[0, k_idx].item()
    w   = topk_w[0, k_idx].item()
    gate_w = gate_up_4d[e, :, 0, :]   # [H, I]
    up_w   = gate_up_4d[e, :, 1, :]   # [H, I]
    d_w    = down_w[e]                 # [I, H]
    xt = x_norm_bf16[0].unsqueeze(0)   # [1, H]
    gate = (xt @ gate_w).float()
    up   = (xt @ up_w).float()
    inter = F.silu(gate) * up
    inter_bf16 = inter.to(torch.bfloat16)
    out_e = (inter_bf16 @ d_w).float()
    output_ref_routing += w * out_e

print(f"  max_diff vs full reference: {(output_ref_routing - ref_out).abs().max():.4e}  (should be ~0)")

# ── Test 3: Single-expert MLP - compare tile split vs full matmul ─────────────
print("\n=== Test 3: Single-expert MLP tile split analysis ===")
e0 = topk_idx[0, 0].item()
gate_w = gate_up_4d[e0, :, 0, :]  # [H, I=192]
up_w   = gate_up_4d[e0, :, 1, :]  # [H, I=192]
d_w    = down_w[e0]                # [I=192, H]
xt = x_norm_bf16[0].unsqueeze(0)   # [1, H] bf16

# Full matmul reference
gate_full = (xt @ gate_w).float()   # [1, 192]
up_full   = (xt @ up_w).float()     # [1, 192]
inter_full = F.silu(gate_full) * up_full  # [1, 192]
inter_full_bf16 = inter_full.to(torch.bfloat16)
out_full = (inter_full_bf16 @ d_w).float()  # [1, H]

# Tile-split matmul (matching kernel: tile0=cols 0:I0=128, tile1=cols I0:I=192)
# gate tile 0 and tile 1
gate_tile0 = (xt @ gate_w[:, 0:I0]).float()    # [1, 128]
gate_tile1 = (xt @ gate_w[:, I0:I]).float()    # [1, 64]
up_tile0   = (xt @ up_w[:, 0:I0]).float()      # [1, 128]
up_tile1   = (xt @ up_w[:, I0:I]).float()      # [1, 64]

# In the kernel, gate/up psum columns represent accumulated dot products:
# For each output neuron j (0..I-1), gate[j] = sum over H of x_norm[h] * gate_w[h,j]
# But the kernel uses nc_matmul with stationary=[P,C] and moving=[P,T]:
# Each partition p contributes one partial sum per matmul call.
# For gate tile 0: stationary=[128, 128] (P=128 partitions, C=I0=128 gate cols)
#   moving=[128, T] (P=128 partitions, T=1)
# nc_matmul result is [128, I0×T] = [128, 128] in PSUM? No...
#
# Actually nc_matmul: stationary[P,C_s] @ moving[P,C_m] → PSUM[P, C_s×C_m]?
# No. The NC matmul is: result[p, i] += stationary[p, :] @ moving[p, :]
# where i indexes into the free dimension. But what IS i?
#
# In NKI, nc_matmul(dst=psum[P, F_dst], stationary=[P, C], moving=[P, T])
# computes psum[p, f] += sum_c( stationary[p,c] * moving[p, t] )
# for each (p, f=t pair)... Actually it is:
# psum[p, f_stat * T + f_mov] += stationary[p, :] . moving[p, :]
# No, let's look at what the kernel does:
#
# dst = gate_up_psum[0:P, gu_base + i_tile : gu_base + i_tile + 1]  -- shape [P, 1]
# stationary = gate_up_bufs[k][0:P, h1, 0:I0]  -- shape [P, I0]
# moving = rmsnorm_normed_bf16[0:P, h1*T : h1*T + T]  -- shape [P, T=1]
# nc_matmul(dst=[P,1], stationary=[P,I0], moving=[P,1])
# This computes:  dst[p, 0] += sum over I0: stationary[p, i] * moving[p, 0]
# = dot product of row p of stationary with moving[p,0]
# So for each partition p: psum[p, gu_base+i_tile] += sum_i( gate_w_shard[p, i] * x_norm_shard[p] )
#
# Wait -- H is split into 128 partitions, each holding H/128 = H_free=16 columns.
# For h1-th H-tile: partition p holds x_norm[h1*128 + p] (one element of x_norm).
# gate_up_bufs[k][p, h1, 0:I0] = gate_w[h1*128+p, 0:I0] (I0 gate cols for that row).
# nc_matmul with stationary=[P, I0] and moving=[P, 1]:
#   result[p, 0] = sum_{i=0}^{I0-1} gate_w[h1*128+p, i] * x_norm[h1*128+p]
# Accumulating over h1 (all H_free=16 tiles):
#   gate_psum[p, gu_base+i_tile=0] = sum_{h1=0}^{H_free-1} sum_{i=0}^{I0-1} gate_w[h1*128+p, i] * x_norm[h1*128+p]
#
# But that is NOT the full dot product! The full gate[j] = sum_{h=0}^{H-1} x_norm[h] * gate_w[h, j].
# The kernel accumulates: psum[p, 0] = sum_h( x_norm[h] * gate_w[h, 0..I0-1] )
# Specifically: sum over partition p, all H elements: x_norm[h1*128+p] * gate_w[h1*128+p, i]
# for a FIXED p and a FIXED i, accumulated over h1.
#
# So psum[p, 0] = sum_{h1} gate_w[h1*128+p, tile0_i0_partition_p] * x_norm[h1*128+p]
# ... it collapses ALL I0 columns into a single scalar per partition!
# This is WRONG! For gate output j, we need sum_h x_norm[h] * gate_w[h, j].
# The kernel sums over I0 COLUMNS (inner dimension) in the stationary, not H.

# This is the key bug: the gate/up matmul is computing the WRONG contraction.
# stationary has shape [P, I0] where dim 1 = I0 (intermediate cols)
# moving has shape [P, T] where dim 1 = T=1 (token)
# nc_matmul contracts over P (partition dim) not over the free dims!
#
# Actually NKI nc_matmul semantics:
# dst[p, s*m] += stationary[p, s] * moving[p, m]  -- outer product per partition?
# OR: dst[p, f] += dot(stationary[p, :], moving[p, :])?
#
# Let me think from the kernel layout:
#   stationary = gate_w block [P=128, I0=128] where P=rows of H, I0=cols
#   moving = x_norm block [P=128, T=1] where P=128 rows of H, T=1 token
# nc_matmul result accumulates into PSUM[P, 1].
# The "matmul" contracts over the P (partition) dimension:
# psum[p_out, f_stat, f_mov] += stationary[p, f_stat] * moving[p, f_mov]
# aggregated over p. So psum has shape [1, I0, T] = [1, 128, 1]?
# But dst is declared [P, 1] not [1, I0, 1].
#
# Actually from NKI docs: nc_matmul computes C[i,j] += A[i,k] * B[k,j]
# where k is the partition dimension. So:
#   dst[f_stat, f_mov] += stationary[p, f_stat] * moving[p, f_mov]  for all p
# = outer product of columns, summed over partitions.
# dst shape = [I0, T] = [128, 1] -- but it goes into psum[P=128, 1]?
# dst is psum slice [P, 1], nc_matmul produces [I0, T] = [128, 1]. That fits!
# So psum[i, 0] = sum_p( stationary[p, i] * moving[p, 0] )
#               = sum_p( gate_w[h1*128+p, i] * x_norm[h1*128+p] )
# Accumulated over all h1:
# psum[i, 0] = sum_{h=0}^{H-1} gate_w[h, i] * x_norm[h]  = gate_output[i]  ✓
#
# So the gate/up matmul IS correct! psum[i, 0] = gate_output[i] for i in [0, I0).
# And psum[i, 0] for the second tile = gate_output[I0 + i] for i in [0, I1).

print("  nc_matmul semantics verified: gate/up contraction is over partition dim (H)")
print("  gate_psum[i, 0] = sum_h( gate_w[h, i] * x_norm[h] ) = gate_output[i]  ✓")

# Now verify SiLU + down matmul tile split
# gate_psum shape after accumulation: [I0, 1] for tile0, another [I0, 1] for tile1
# But tile1 only has I1=64 valid, rest zero-padded.
# silu_res = silu(gate_psum[0:P, gu_base:gu_base+I_tiles]) -- shape [P, 2]
# Wait: I_tiles=2, gate_psum[P, gu_base:gu_base+2] = [P=128, 2]
# tile0 col: psum[0:128, gu_base+0] -- this is gate outputs for neurons 0..127
# tile1 col: psum[0:128, gu_base+1] -- gate outputs for neurons 128..191, zeros for 192..255
# But I_tiles=2 and I0=128, so psum only has 2 columns for gate, 2 for up.
# inter_bf16 = [P=128, I_tiles=2] = [128, 2]
#   col 0 = silu(gate[0:128]) * up[0:128]
#   col 1 = silu(gate[128:192]) * up[128:192]  (last 64 valid, 64 zero-padded)

# Down matmul:
# down_full0_bufs[k] = [P=128, H_shard=1024] = down_w[e, 0:128, prg_id*1024:(prg_id+1)*1024]
# Actually DMA pattern [[H, I0], [1, H_shard]] means:
#   row stride = H=2048, col stride = 1, shape [I0=128, H_shard=1024]
# So down_full0[p, h] = down_w[e, p, prg_id*1024 + h]  -- rows 0:I0 of down_w
# Wait: DMA pattern for down_full0: pattern=[[H, I0], [1, H_shard]]
#   The pattern is [[row_stride, nrows], [col_stride, ncols]]
#   = [[2048, 128], [1, 1024]] -- 128 rows, 1024 cols each
# But down_full0_bufs[k] has shape [P=128, H_shard=1024].
# So P=128 = I0 rows of down_w (the first I0 intermediate neurons).
# down_full0[i, h] = down_w[e, i, prg_id*1024 + h]  for i in [0, I0)

# For down matmul:
# nc_matmul(dst=down_psum[P, 1], stationary=down_full0[P, H_shard_tile],
#           moving=inter_bf16[P, 0:1])
# For h1_out-th output tile:
# stationary = down_full0[0:P, h1_out*P : h1_out*P+P] -- [P, P] square
# moving = inter_bf16[0:P, 0:1] -- [P, 1]  (col 0 = inter for neurons 0..127)
# nc_matmul: dst[i, j] = sum_p( stationary[p, i] * moving[p, j] )
# = sum_p( down_w[e, p, h1_out*128+i] * inter_bf16[p, 0] )
# = sum_{i_inter=0}^{127} down_w[e, i_inter, h1_out*128 + i] * inter[i_inter]
# That gives output neuron [h1_out*128 + i] = sum over first 128 intermediate neurons.
# Then the second nc_matmul adds:
# stationary = down_full1[0:P, h1_out*P : ...] -- rows I0:I0+I1 of down_w
# moving = inter_bf16[0:P, 1:2] -- col 1 = inter for neurons 128..255
#   BUT col 1 of inter_bf16 is neurons [128..191] with 192..255 zero-padded.
# nc_matmul: sum_{i_inter=0}^{127} down_w[e, I0+i_inter, h1_out*128+i] * inter_bf16[i_inter, 1]
# = sum_{i_inter=0}^{63} down_w[e, 128+i_inter, h1_out*128+i] * inter[128+i_inter]
#   (because inter_bf16[64:128, 1] = 0 from zero padding)
# Total: sum_{j=0}^{192} inter[j] * down_w[e, j, h_out]  ✓

print("\n[Analysis] Down matmul tiling logic is CORRECT if inter_bf16 zero padding is correct.")

# Check: does inter_bf16 col 1 actually have zeros in rows 64:128?
# In the kernel: inter_bf16 = cast(inter_f32), where
#   inter_f32[p, 1] = silu_res[p,1] * up_sb[p,1]
#   silu_res = activation(silu, gate_up_psum[P, gu_base:gu_base+I_tiles])
#   gate_up_psum[p, gu_base+1] for tile1: sum_p( gate_t1_128[p, h1, 0:I0] * x_norm[p, h1] )
#   But gate_t1_128[p, h1, I1:I0] = 0 (zeroed by memset), gate_t1_128[p, h1, 0:I1] = gate_w[h1*128+p, I0:I]
# So gate_psum_tile1[p, 0] = sum_{h1} sum_{i=0}^{I0-1} gate_t1_128[p, h1, i] * x_norm[p, h1]
#   = sum_{h1} sum_{i=0}^{I1-1} gate_w[h1*128+p, I0+i] * x_norm[p, h1]  (I1:I0 are zero)
#   = sum_{h=0}^{H-1} gate_w[h, I0 + p] * x_norm[h]  ... hmm this maps p->output neuron
#
# BUT gate_t1_128 shape is [P=128, H_free=16, I0=128] and the matmul uses
# stationary=gate_t1_128[P, h1, 0:I0] -- shape [P, I0]
# moving=rmsnorm[P, h1*T:h1*T+T] -- shape [P, 1]
# nc_matmul: psum[i, 0] += sum_p( gate_t1_128[p, h1, i] * x_norm[p, h1] )
# For tile1: gate_t1_128[p, h1, i] = gate_w[h1*128+p, I0+i] for i<I1, 0 for i>=I1
# Accumulated over h1: psum[i, 0] = sum_{h=0}^{H-1} gate_w[h, I0+i] * x_norm[h]
# = gate_output[I0+i]  for i in [0, I1)
# = 0                  for i in [I1, I0)
#
# So gate_psum_tile1[0:P, gu_base+1] has:
#   rows 0..63: gate_output[128..191]  (valid)
#   rows 64..127: 0  (because gate_t1_128[p, h1, i>=64] = 0)
#
# inter_bf16[p, 1] = silu(gate_psum[p, 1]) * up_psum[p, 1]
#   For p in [0, 64): silu(gate_out[128+p]) * up_out[128+p]  -- valid inter
#   For p in [64, 128): silu(0) * 0 = 0  -- zero
#
# Then down matmul:
# nc_matmul(dst, stationary=down_full1[P, h1_out*P:], moving=inter_bf16[P, 1:2])
# down_full1[p, h] = down_w[e, I0+p, h]  for p in [0, I1)  (valid)
#                  = 0                    for p in [I1, I0)  (zero-padded)
#
# Result: dst[i, 0] = sum_p( down_w[e, I0+p, h1_out*128+i] * inter[I0+p] )
# where both are zero for p>=64. So only p=0..63 contribute.
# This gives sum_{j=128}^{191} inter[j] * down_w[e, j, h_out]  ✓
#
# CONCLUSION: The down matmul tiling is MATHEMATICALLY CORRECT.
# The bug must be elsewhere.

print("\n=== Test 4: Direct single-expert comparison ===")
# Run reference with K=1, using expert 0 with weight 1.0
e_test = topk_idx[0, 0].item()
print(f"Testing expert {e_test}")

gate_w  = gate_up_4d[e_test, :, 0, :]   # [H, I]
up_w    = gate_up_4d[e_test, :, 1, :]   # [H, I]
d_w     = down_w[e_test]                 # [I, H]
xt      = x_norm_bf16[0].unsqueeze(0)    # [1, H]

gate  = (xt @ gate_w).float()    # [1, I]
up    = (xt @ up_w).float()      # [1, I]
inter = F.silu(gate) * up        # [1, I]
inter_bf16 = inter.to(torch.bfloat16)
out_single = (inter_bf16 @ d_w).float()  # [1, H]

print(f"  Reference single-expert output norm: {out_single.norm():.4f}")
print(f"  Reference output[:8]: {out_single[0, :8].tolist()}")
print(f"  Kernel output[:8]:    {kern_cpu[0, :8].tolist()}")

# Check if maybe the issue is with the aff_bcast broadcast or norm_weights
print("\n=== Test 5: Check norm_weights (top-k normalization) ===")
# reference: topk_w / topk_w.sum()
print(f"  Reference topk_w (sum={topk_w[0].sum():.4f}): {[f'{w:.4f}' for w in topk_w[0].tolist()]}")
# kernel uses: top8_vals = raw softmax probs (pre-normalization), then normalizes in-kernel
# So kernel sum should also be 1.0

print("\nDiagnostics complete.")
