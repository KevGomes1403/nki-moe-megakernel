"""
Correctness test for v15d (Plan D: nc_matmul_mx MXFP8 Gate_Up).

Verification 1: BF16 accuracy (rtol=0.15, atol=0.15) vs torch bf16 reference
Verification 2: Quantization correctness (rtol=1e-2, atol=1e-2) vs Python MXFP8 reference
"""

import os
import sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"

import numpy as np
import torch
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v15d

# -----------------------------------------------------------------------
# Dimensions
# -----------------------------------------------------------------------
E    = v15d._E          # 128
H    = v15d._H          # 2048
I    = v15d._I          # 192
K    = v15d._K          # 8
B    = 1                # T=1 for TKG
GU   = v15d._GU_FLAT   # 384
N_H512    = v15d._N_H512     # 4
GU_SCALE_F = GU // 4           # = 96 (x4-grouped GU count used in weight building; NOT the scale HBM dim)
SCALE_P   = v15d._SCALE_P    # 16
PMAX      = v15d._PMAX       # 128
T_PAD     = v15d._T_PAD      # 4
Q_WIDTH   = v15d._Q_WIDTH    # 4
Q_HEIGHT  = v15d._Q_HEIGHT   # 8
GU_FLAT_W = v15d._GU_FLAT_W  # 384
H_FREE    = v15d._H_FREE     # 16

device = xm.xla_device()
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------
# Helper: compute MXFP8 block scales for [P, F] weight tensor
# Block = [Q_HEIGHT=8 P-rows, 4 F-elements], scale = uint8
# For weight in packed form [PMAX, GU_FLAT_W=384]:
#   scale_shape = [PMAX//8, GU_FLAT_W//4] = [16, 96]
# -----------------------------------------------------------------------
def compute_mxfp8_scales(w_pf, q_height=8, q_width=4, max_exp=8):
    """w_pf: float32 [P, F] → uint8 [P//q_height, F//q_width]"""
    P, F = w_pf.shape
    Pb = P // q_height
    Fb = F // q_width
    w_blocks = np.abs(w_pf.reshape(Pb, q_height, Fb, q_width))
    block_max = w_blocks.max(axis=(1, 3))   # [Pb, Fb]
    safe_max = np.where(block_max == 0, 1e-38, block_max)
    exp_vals = np.floor(np.log2(safe_max)).astype(np.int32)
    scale_u8 = np.clip(exp_vals - max_exp, 0, 255).astype(np.uint8)
    return scale_u8


def dequant_mxfp8(w_pf, scale_u8, q_height=8, q_width=4):
    """Dequantize w_pf using scale_u8, quantize to fp8 range then back."""
    P, F = w_pf.shape
    # expand scale
    scale_f = np.power(2.0, scale_u8.astype(np.float32))   # [P//8, F//4]
    scale_expand = np.repeat(np.repeat(scale_f, q_height, axis=0), q_width, axis=1)  # [P, F]
    w_scaled = w_pf / (scale_expand + 1e-38)
    w_fp8_clamped = np.clip(w_scaled, -448.0, 448.0)
    # round to fp8 via torch
    w_t = torch.tensor(w_fp8_clamped.flatten(), dtype=torch.float32)
    w_fp8 = w_t.to(torch.float8_e4m3fn).to(torch.float32).numpy().reshape(P, F)
    return w_fp8 * scale_expand


# -----------------------------------------------------------------------
# Generate test data
# -----------------------------------------------------------------------
print("Generating test tensors...")

inp_np  = np.random.randn(B, 1, H).astype(np.float32) * 0.1
gamma_np = np.ones((1, H), dtype=np.float32)
router_w_np = np.random.randn(H, E).astype(np.float32) * 0.01

# Logical gate_up weights: [E, GU=384, H=2048] fp32 (gate: GU[0:I], up: GU[I:2I])
# We use small values so quantization error is manageable
gate_up_w_logical = np.random.randn(E, GU, H).astype(np.float32) * 0.02
down_w_logical    = np.random.randn(E, I, H).astype(np.float32) * 0.02

# -----------------------------------------------------------------------
# Build new weight layout: [E, PMAX, N_H512, GU_FLAT_W] int8
# Mapping: weight_new[e, p, h512, gu] = fp8(gate_up_w_logical[e, gu, h512*PMAX + p])
#
# Note: "p" indexes 128 H-partition rows (each holds 4 H values via x4 packing)
# But in our Plan D, the weight is NOT x4 packed in H! The x4 packing is in GU.
# Actually: stationary [128, 96_x4] means x4 groups are in GU (output) direction.
# P=128 represents 128 individual H-contraction elements.
# So one H512 tile = 128 P elements × 4 H512 tiles = 512 H elements.
# But H=2048 → need 2048/128 = 16 tiles, not 4!
#
# Let me reconcile:
# H_free = 16 (each H_free tile = PMAX=128 H-elements)
# N_H512 = 4 (each H512 tile = 4 × H_free tiles = 4 × 128 = 512 H-elements)
# So H512 tile h512 covers H_free tiles h512*4..(h512+1)*4-1
# H elements: h512*512..(h512+1)*512-1
#
# For the weight at (e, h512, p):
# - h512 in 0..3 selects the H512 tile
# - p in 0..127 selects the P-row = one H-element within the H512 tile
# - But then only 4*128 = 512 H elements are covered!
#
# WAIT: The x4 packing IS in the H direction for the activation (moving tensor).
# For the weight (stationary), x4 packs the GU (free/output) direction.
# The moving activation [128_P, T_pad] has x4 packing in H (4 H-elements per P-row).
# The stationary weight [128_P, 96_GU_x4] has x4 packing in GU (4 GU per x4-group).
# So each matmul processes: 128 P-rows × 4 H-per-row (from activation x4) = 512 H elements.
# And produces: 96_GU_x4 × 4 = 384 GU neurons output.
#
# So the weight at (e, p, h512, gu_x4):
# - h512: H512 tile
# - p: H-partition (one x4 group of 4 H-elements in the activation)
# - gu_x4: output group (4 GU neurons packed)
# - But there are 4 H-elements per partition (from x4 activation), so which H element?
#
# In nc_matmul_mx: stationary[p, f] × moving[p, t] for each (p) → accumulate.
# Each p-row of the activation x4 element = 4 H-values.
# The corresponding p-row of the weight must have the SAME 4 H-values contracted.
# So weight[p, gu_x4] = {W(h=p*4+0, gu_x4), W(h=p*4+1, gu_x4), ...}?
# No — the weight doesn't have x4 in H; it's stationary with x4 in GU.
#
# The actual nc_matmul_mx operation:
# For each p in 0..127:
#   out[gu_x4, _, t] += weight[p, gu_x4] * activation[p, t]
# where each element is x4 (packed float8):
#   weight: 4 consecutive GU values per x4
#   activation: 4 consecutive H-values packed per x4 (but same p → same 4 H-values)
#
# This is: out[gu*4+q, _, t] += W[p, gu] * A[p*4+h_offset, t] summed over p and h_offset
# That's only 128 H elements per H512 tile, needing 16 tiles for H=2048.
# But N_H512=4! So only 512 H-elements would be covered!
#
# CONCLUSION: The x4 in the MOVING activation packs 4 H-values per partition row,
# giving 128*4=512 H elements per H512 tile, and 4 tiles * 512 = 2048 ✓.
# The STATIONARY weight x4 packs 4 GU neurons per group (output direction).
# So weight[p, gu_grp] covers 4 H-values contracted against 4 GU neurons.
#
# Weight layout [E, PMAX, N_H512, GU_FLAT_W=384]:
# weight_new[e, p, h512, gu] where:
#   p = 0..127 (H-partition: which of 128 groups of 4 H-elements in this H512 tile)
#   h512 = 0..3 (H512 tile)
#   gu = 0..383 (GU neuron index, raw; every 4 consecutive = one x4 group)
#   Corresponds to logical: gate_up_w_logical[e, gu, h512*512 + p*4 + {0,1,2,3}]
#   BUT: since stationary[p, gu_grp] is one x4 element covering 4 H-values at p*4+0..3,
#   we need to handle which of 4 H values within the partition group.
#   For weight: weight[p, gu_x4_grp] = the 4-H-value × 4-GU-value sub-matrix.
#   But nc_matmul_mx stationary dtype is x4 in one direction only!
#   The result is: out[gu_x4_grp, q_inner, t] += sum_p(stat[p, gu_x4_grp] · mov[p, t])
#   where · is the x4 inner product (contracting 4 H-values per x4).
#
# So the weight x4 element at [p, gu_x4_grp] contracts with activation x4 at [p, t]:
# Both have 4 H-values → inner product gives 1 scalar per (gu, t_token, h_offset_within_p)
# And the 4 different x4 positions in GU produce 4 separate outputs.
# The q_inner=4 in PSUM comes from 4 separate sub-matmuls in the I (output) direction.
#
# Bottom line: weight[p, gu] = fp8 for neuron gu, contracting with H-element (h512*512 + p*4 + which?)
# The "which" is determined by the activation's x4 packing.
# The activation at [p, t] holds H-values h512*512 + p*4 + 0..3.
# The weight at [p, gu] must hold W[gu, h512*512 + p*4 + 0..3] packed as x4 too? No!
#
# OK I need to step back. The x4 dtype for both stationary and moving:
# - Each x4 element = 4 float8 values packed together
# - nc_matmul_mx computes: dst[f_stat, q_inner, f_mov] += sum_p(stat[p, f_stat] ⊗ mov[p, f_mov])
#   where ⊗ is the x4 inner product (4 H-values each → 1 scalar? or 4 outputs?)
#
# From the reference code output shape [TILE_I, I_4=4, TILE_T]:
# TILE_I = I/4 = 48, I_4 = 4, TILE_T = T_pad
# The 4 in I_4 corresponds to the 4 inner q_width_I_idx iterations in _matmul_mx_accumulate
# Each call: dst[:48, q_w_idx, :T_pad] += stat[:, h512, weight_I_slice] · mov[:, h512, T_slice]
# weight_I_slice = 48 elements (x4 packed, 48*4=192 GU neurons), T_slice = T_pad x4 elements
#
# So each x4 multiply: stat[p, gu48] × mov[p, t]
# Both are x4 → the product contracts 4 H-values and 4 GU-neurons simultaneously?
# Or they're dot products in the H direction?
#
# From hardware perspective: x4 dtype means 4 elements are computed together as a block.
# For MXFP8×MXFP8: each x4 element = 4 separate float8 values.
# nc_matmul_mx with stationary[P=128, F_stat] and moving[P=128, F_mov]:
# Out[f_stat, _, f_mov] = sum_{p=0..127}(stat[p, f_stat] * mov[p, f_mov])
# where stat[p, f_stat] is one x4 element (4 float8 values, in the output direction)
# and mov[p, f_mov] is one x4 element (4 float8 values, in the token direction)
# → The H-dimension contraction is purely via the P summation (128 elements only per tile)
# The x4 in both directions creates a 4×4 outer product contribution per (p, f_stat, f_mov)
# giving a 4×4 matrix of outputs? But PSUM has shape [F_stat, 4, F_mov] not [F_stat, 4, F_mov, 4]
#
# FINAL UNDERSTANDING (from nkilib reference output shape [TILE_I, 4, TILE_T]):
# The 4 in the middle is NOT from x4 product; it's from 4 separate I sub-tiles (q_width_I_idx loop).
# Each call to nc_matmul_mx:
#   stationary: [128_P, F_stat] x4 → each x4 element = 4 float8 in the H-contraction direction
#   moving:     [128_P, F_mov] x4 → each x4 element = 4 float8 in the H-contraction direction
# The matmul contracts over P×4 = 512 H-elements per call.
# Output [F_stat, 1, F_mov] (one value in the middle, accumulated into the q_w_idx slot).
# 4 calls (q_width_I_idx=0..3) × F_stat elements × F_mov tokens = full output.
#
# So: stat[p, gu48] = 4 H-values (h512*512 + p + 0, 128, 256, 384)?
# No - stat[p, gu48] holds the weights for H-elements p*4+0..p*4+3 (4 consecutive H-elements)
# and GU neuron group gu48 (which holds 4 consecutive GU neurons when unpacked by the 4 sub-tiles).
#
# For sub-tile q_w_idx=0: weight_I_slice covers I/4 = 48 GU neurons (0..47)
#   stat[p, gu48_x4] covers H-elements p*4+0..p*4+3 for GU neurons 0..47 packed as x4 in H
# Wait, that would mean x4 in H direction for the stationary weight too!
#
# The nkilib comment says: "4_H packed in x4 dtype" — confirming x4 is in H direction.
# So: stat[p, f] = float8_e4m3fn_x4 where 4 = 4 H-elements, p=0..127 (partition/H-group index)
# Total H per call = 128 × 4 = 512 ✓
# And F_stat = I_chunk (the output neuron index within one sub-tile)
# After 4 sub-tiles: total output = 4 × F_stat = I ✓
#
# Plan D's weight layout [E, 128_P, N_H512=4, GU=384] int8 → view as x4 [128, 4, 96]:
# "96 x4-groups" in the GU direction → x4 packs GU, NOT H!
# But this contradicts the nkilib x4 in H direction!
#
# Plan D is using a DIFFERENT axis for x4 than the nkilib reference.
# Plan D: stationary x4 in GU → each [128_P, 96_x4] element covers 4 GU neurons, P covers H elements.
# Each H512 tile: 128 P × 1 H-per-P = 128 H elements → 4 tiles × 128 = 512? No, 4×128=512.
# But H=2048 → need 2048/128 = 16 tiles, not 4.
# Unless T_pad=4 handles this? But T_pad is for MOVING not STATIONARY...
#
# I believe Plan D has an error in the N_H512 calculation.
# Plan D says N_H512 = H/512 = 4, but with x4 in GU direction (not H):
# Each H512 tile loop covers 128 H-elements (not 512).
# So we'd need N_H512=16 (H/128), not 4.
#
# HOWEVER: The plan spec explicitly says N_H512=4 = H/512.
# This can only work if x4 IS in the H direction for the stationary weight too.
# Then: weight [128_P, N_H512=4, GU_x4=96] where x4 packs H → 128×4=512 H per tile, 4 tiles=2048 ✓
# And GU dimension: 96 x4-groups × each x4=4 GU neurons = 384 GU neurons ✓?
# But then each element [p, h512, gu_x4] covers 4 H-values AND one GU neuron??
# That doesn't make sense for a matrix multiply.
#
# Actually: in x4 dtype, each "element" IS 4 float8 values.
# For stationary [128, 96] x4: we have 128×96 x4-elements = 128×384 float8 values.
# The x4 packing can be in ANY logical direction — hardware just sees 4×128×96=49152 float8 values.
# The nc_matmul_mx semantics:
#   out[f, q, t] = sum_p(stat[p, f] · mov[p, t])
# where stat[p, f] is an x4 element (4 float8 values labeled q=0..3)
# and mov[p, t] is an x4 element (4 float8 values labeled r=0..3)
# The inner product: sum over p, and within each p, the 4 float8 in stat × 4 float8 in mov
# gives a 4×4 result matrix... but output is [f, 4, t] not [f, 4, t, 4].
# So there must be a specific contraction within the x4 pairs.
#
# ACTUAL nc_matmul_mx semantics (from nkilib use pattern):
# stat[p, f] × mov[p, t]: x4 dot product
# The 4 values in stat[p,f] and 4 values in mov[p,t] contract to 1 scalar.
# out[f, 1, t] += sum_p(dot4(stat[p,f], mov[p,t]))
# And the 4 in PSUM middle comes from 4 separate matmul calls (q_width_I_idx).
#
# If x4 in H direction:
# stat[p, f] = {W(4p, f), W(4p+1, f), W(4p+2, f), W(4p+3, f)} (4 consecutive H)
# mov[p, t]  = {A(4p,t),  A(4p+1,t),  A(4p+2,t),  A(4p+3,t)}  (4 consecutive H)
# dot4 = sum_{k=0}^{3} W(4p+k, f) * A(4p+k, t)
# Total: sum_p(dot4) = sum_{h=0}^{512-1} W(h, f) * A(h, t) ✓
#
# If x4 in GU direction for stationary but H direction for moving:
# stat[p, f_x4] = {W(p, 4f_x4+0), W(p, 4f_x4+1), W(p, 4f_x4+2), W(p, 4f_x4+3)} (4 consec GU)
# mov[p, t] = {A(4p, t), ...} (4 H values)
# dot4 = sum_{k=0}^{3} W(p, 4f+k) * A(4p+k, t)?
# That's a weird mixed contraction.
#
# CONCLUSION: x4 must be in the SAME direction for both tensors (H-contraction).
# Plan D weight layout: [E, 128_P, N_H512=4, GU=384] where GU=384 is raw, x4 packs H (in P direction).
# The HBM GU=384 flat dimension is unpacked (one float8 per GU neuron).
# The P=128 has 4 H-elements per partition (packed as x4 when viewed as x4 dtype).
# So view as x4 → [128_P, 4, 96]: P=128 (×4 H per row = 512 H), and 96 = 384/4.
# The 96 = F_stat dimension for one sub-tile of the I/GU dimension!
# With 4 sub-tiles (q_width_I_idx): 4 × 96 = 384 GU neurons ✓.
#
# FINALLY: weight_new[e, p, h512, gu] where:
#   p = H-partition (0..127), x4: covers H-elements h512*512 + p*4 + 0..3
#   h512 = H512 tile (0..3)
#   gu = GU neuron (0..383), one sub-byte per neuron (but stored as groups of 4 for x4)
#   Corresponds to: gate_up_w_logical[e, gu, h512*512 + p*4 + 0..3] (4 H-elements packed per P row)
#   With one float8 byte per gu, the 4 H-elements at position (e,p,h512,gu) are packed as x4
#   by grouping 4 CONSECUTIVE gu values: {gu, gu+1, gu+2, gu+3} at same H-position?? NO!
#   The x4 packing of H at position [p, h512] means:
#   The float8 at gu=j corresponds to W[j, h512*512+p*4+0] (first H element in this x4 group)
#   and the "x4" groups p=k with k packing 4 consecutive H elements {k*4, k*4+1, k*4+2, k*4+3}.
#   So the raw layout stores: for each (h512, p), GU=384 float8 values = weights for all 384 GU neurons
#   at the 4 H-elements h512*512+p*4+0..3 (one float8 per GU neuron, representing which H element?).
#   This is still ambiguous for a single float8 per (p, h512, gu) position.
#
# The key: for nc_matmul_mx, EACH x4 element is a VECTOR of 4 float8 values in one direction.
# So stat[p, f] is one x4 element = 4 float8 values for H direction:
# stat[p, f] = [W(h512*512+p*4+0, f), W(h512*512+p*4+1, f), W(h512*512+p*4+2, f), W(h512*512+p*4+3, f)]
# This is ONE x4-packed value at position (p, f).
# F = number of x4 elements = number of output neurons / 4 IF sub-tiling... or just output neurons?
#
# Let's just use F_stat = GU_FLAT_W = 384 for the ENTIRE output dimension (no x4 in GU).
# Then view as x4 [128, 4, 96]: dimension 1 (size 4) is the H x4 packing.
# stat.reshape(128, 4, 96) → stat[p, h4, gu96]:
#   p = H-group index (0..127)
#   h4 = H-within-group (0..3), this is the x4 inner dimension
#   gu96 = GU neuron index 0..95 (but 96 ≠ 384/4=96... wait 384/4=96 ✓)
# So F_stat = 96 (= GU/4)? And each matmul operates with F_stat=96 GU neurons?
# Then 4 × 96 = 384 GU after the q_width_I_idx loop? But Plan D doesn't have that loop!
#
# I think I've been overthinking this. Let me just use the same approach as nkilib:
# x4 in H direction, F = output neurons, loop over q_width (4 sub-tiles in output).
# Plan D says [128, 96_GU_x4] — the "x4" label refers to the dtype not the dimension.
# float8_e4m3fn_x4 means 4 float8 values packed into one "element" of this type.
# When viewed as [128, 96] float8_e4m3fn_x4:
#   Each of the 128*96 positions holds 4 float8 values.
#   Total float8 values = 128 * 96 * 4 = 49152.
#   Original int8 shape [128, 384]: 128 * 384 = 49152. ✓
#   The x4 packing is in which direction? The 4 comes from the last dim: 384 → 96 groups of 4.
#   So x4 packs consecutive GU neurons: w_x4[p, gu96] = {W(p, gu96*4), W(p, gu96*4+1), ...}.
# Now for nc_matmul_mx with stationary[128, 96] x4:
#   Each position [p, gu96] = 4 GU-neuron weights at H-partition p (which H-element?)
#   The dot product with moving [128, T_pad] x4 at [p, t_x4] = {A[p*4+0, t], ...}??
#   This is mixed H/GU in stationary vs H in moving.
#   This doesn't work as a proper matrix multiply!
#
# FINAL RESOLUTION: I'll follow the nkilib's actual working pattern exactly.
# x4 is in H-contraction direction for BOTH stationary and moving.
# Weight HBM: [E, 128_P, N_H512, GU] where GU=I_chunk (one output neuron per F position)
#   Each [p, h512, gu] is a float8_e4m3fn_x4 element representing W at (p, h512) for neuron gu
#   with 4 H-values packed: W[gu, h512*512+p*4+0..3].
#   Stored as raw int8: 4 bytes per position → GU raw bytes = 384 raw GU IF x4 in H.
#   But view as x4 [128, N_H512, GU/4=96]? No: GU_raw = 384, view as x4 gives [128, 4, 96].
#   That means the x4 packing is in the n_H512 direction (4 H tiles per x4)? That makes no sense.
#
# ACTUAL PLAN D INTERPRETATION:
# [E, 128, 4, 384] int8 → view as [128, 4, 96] x4
# The 4 in the middle IS n_H512! And 96 = GU/4!
# So x4 packs 4 CONSECUTIVE H512 TILES at once? That can't be right.
#
# OK LET ME JUST LOOK AT WHAT SHAPE nc_matmul_mx ACTUALLY ACCEPTS and
# use whatever works. I'll match the nkilib shape exactly:
# stationary [PMAX=128, I_chunk] float8_e4m3fn_x4 (x4 in H direction)
# Stored in HBM as [E, 128, N_H512, I_chunk] float8_e4m3fn, NOT as int8.
# But Plan D says int8! So we load as int8 then .view(x4).
# [128, 4, 384] int8 → [128, 4, 96] x4  where 96 = 384/4
# In this view: the 4 in dim 1 is n_H512 ✓, and dim 2 = 96 is GU_x4_groups (x4 packs GU).
# For nc_matmul_mx: stationary[:, i_h512, :] = [128, 96] x4 (x4 in GU direction).
# MOVING: [128, T_pad] x4 (x4 in H direction from quantize_mx).
# These have DIFFERENT x4 directions → unclear if hardware accepts this.
#
# Let me just try it and see if the compiler accepts it.
# Use F_stat=96, F_mov=T_pad, and see what shape PSUM output has.
# Based on Plan D spec: "PSUM output shape: [I_out_sz, 4, T_pad] in bf16" where I_out_sz=96.
# This matches: out[96, 4, T_pad] with 4 = q_width from sub-tiling.
# But Plan D doesn't loop over q_width_I_idx — it says "nc_matmul_mx call per H512 tile".
# So the 4 in the PSUM middle dimension comes from the hardware's inherent 4-way output of nc_matmul_mx.
#
# OK I'll just go with what Plan D says literally and trust it works.
# MXFP8 weight: [E, 128_P, N_H512=4, GU=384] int8 → view as [128, 4, 96] x4
# nc_matmul_mx: stationary=[128, 96] x4, moving=[128, T_pad] x4
# PSUM dst: [96, 4, T_pad]
# This is what we implement. Moving on.
# -----------------------------------------------------------------------

# Weight [E, PMAX, N_H512, GU_FLAT_W]: w_new[e,p,h512,gu] = fp8_byte(W_logical[e, gu, h512*PMAX+p])
# (P=128 H-elements per tile, 4 tiles * 128 = 512, but H=2048! So actually p ranges over 128 out of 512)
# Based on the plan: p indexes 128 H-partition groups, each group covers 4 H-elements (via x4 in H).
# w_new[e, p, h512, gu] corresponds to W_logical[e, gu, h512*512 + p*4 + {0,1,2,3}]?
# But we store only 1 float8 per (p, h512, gu) position...
#
# Actually the Plan D comment says "GU=384 stored as raw bytes, interpreted as 4 FP8-per-group internally"
# "4 FP8-per-group" refers to x4 packing in the GU direction:
# w_x4[p, h512, gu96_grp] = {W(p, h512, gu96_grp*4+0), ..., W(p, h512, gu96_grp*4+3)}
# And p = one H element (not a group of 4). So H coverage = PMAX * N_H512 = 128 * 4 = 512 ≠ 2048!
# This is wrong for H=2048. Plan D must have x4 in H direction.
#
# I'll build the weight following nkilib's convention: x4 in H.
# w_new[e, p, h512, gu] = fp8(W_logical[e, gu, h512*512 + p*4 + {H_inner}])
# But single byte per position... we store 4 bytes for 4 H-elements as 4 consecutive GU positions?
# No: for x4 in H direction, the 4 H-values for a given (p, gu) are packed as one x4 dtype element.
# In raw int8: that's 4 consecutive bytes at position (p, h512, gu*4 + 0..3)?
# Then gu_raw runs 0..383/4-1 = 0..95 groups, total 384 raw bytes = 96*4 ✓.
# w_raw[e, p, h512, gu_raw]: gu_raw = 0..383, where gu_raw = gu*4 + h_inner
# w_raw[e, p, h512, gu*4+h_inner] = fp8(W_logical[e, gu, h512*512 + p*4 + h_inner])
# This maps to: view as x4 [128, 4, 96] → [p, h512, gu] where gu is the outer x4 group index.
#
# FINAL WEIGHT GENERATION:
# gate_up_w_new[e, p, h512, gu_raw] = fp8_byte(W_logical[e, gu_raw//4, h512*512 + p*4 + (gu_raw%4)])
# Wait: gu_raw = gu*4 + h_inner, so gu = gu_raw//4 and h_inner = gu_raw%4.
# W_logical[e, gu, h512*512+p*4+h_inner] = W_logical[e, gu_raw//4, h512*512+p*4+(gu_raw%4)]
# Hmm, this interleaves H and GU dimensions in an unusual way.
#
# SIMPLEST CONSISTENT INTERPRETATION:
# Follow nkilib exactly. Weight HBM for one expert: [128_P, N_H512, I_chunk] x4
# where x4 packs H (the contraction direction).
# Raw bytes layout: [128_P, N_H512, I_chunk, 4_H] int8 (last dim = 4 H-values)
# = [128, 4, I_chunk, 4] total → but Plan D says [128, 4, 384] not [128, 4, I_chunk, 4].
# With I_chunk = GU_FLAT_W/4 = 96: [128, 4, 96, 4] = [128, 4, 384] raw bytes ✓
# The 384 raw bytes = I_chunk=96 * 4_H values per x4 element.
# view as x4: [128, 4, 96] ✓
# w_x4[p, h512, gu96] = {W(4H at p,h512) for GU-neuron gu96}
# = {W(gu96, h512*512+p*4+0), W(gu96, h512*512+p*4+1), W(gu96, h512*512+p*4+2), W(gu96, h512*512+p*4+3)}
# Stored as: w_raw[p, h512, gu96*4+0..3] = fp8 bytes of above 4 values.
print("Building MXFP8 weight layout (x4 in H direction)...")
# Weight HBM: [E, PMAX, N_H512, GU=384] float8_e4m3fn_x4 (x4 packs 4 H-values per P-row)
# w[e, p, h512, gu96*4 + h_inner] = fp8(W_logical[e, gu96, h512*512 + p*4 + h_inner])
gate_up_w_new = np.zeros((E, PMAX, N_H512, GU), dtype=np.uint8)  # uint8 for view as x4
gate_up_w_dequant = np.zeros((E, GU, H), dtype=np.float32)

for e in range(E):
    for h512 in range(N_H512):
        for p in range(PMAX):
            # 4 H-elements for this partition group (x4 packing in H direction)
            h_base = h512 * 512 + p * 4
            for gu96 in range(GU_SCALE_F):  # GU_SCALE_F = 96 x4-groups = 384 GU neurons total
                # 4 H-values for this x4 group: W[gu96, h_base+0..3]
                w4h = gate_up_w_logical[e, gu96, h_base:h_base + 4]  # [4] fp32
                # Convert to fp8 and store as raw bytes at [e, p, h512, gu96*4:gu96*4+4]
                w4h_t = torch.tensor(w4h, dtype=torch.float32)
                w4h_fp8 = w4h_t.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
                gate_up_w_new[e, p, h512, gu96 * 4:gu96 * 4 + 4] = w4h_fp8

print("Building weight scales...")
# Scale HBM: [E, SCALE_P=16, N_H512=4, GU=384] uint8
# From nkilib: scale_shape = [PMAX // Q_HEIGHT, n_H512, I] = [16, 4, 384]
# Block = [Q_HEIGHT=8 P-rows, 1 GU-neuron] = 8 × 4H = 32 float8 values per scale ✓
# scale[p_blk, h512, gu] = scale for P-rows p_blk*8..(p_blk+1)*8, H512 tile h512, GU neuron gu
gate_up_scales_new = np.zeros((E, SCALE_P, N_H512, GU), dtype=np.uint8)

for e in range(E):
    for h512 in range(N_H512):
        # Build [PMAX, GU_SCALE_F, 4] weight values (4 H-values per x4 group)
        w_tile = np.zeros((PMAX, GU_SCALE_F, 4), dtype=np.float32)
        for p in range(PMAX):
            h_base = h512 * 512 + p * 4
            for gu96 in range(GU_SCALE_F):
                w_tile[p, gu96, :] = gate_up_w_logical[e, gu96, h_base:h_base + 4]

        # Scale block: [Q_HEIGHT=8 P-rows, 1 GU-neuron × 4 H-values]
        # For each scale block, max abs over 8 P-rows × 4 H-values = 32 float8 values
        # w_tile: [128, 96, 4] → max per [8P, 1_gu96, 4H] = max per 32 values
        w_abs = np.abs(w_tile)  # [128, 96, 4]
        w_block = w_abs.reshape(SCALE_P, Q_HEIGHT, GU_SCALE_F, 4)  # [16, 8, 96, 4]
        block_max = w_block.max(axis=(1, 3))  # [16, 96] — max over 8P and 4H per gu96
        safe_max = np.where(block_max == 0, 1e-38, block_max)
        exp_vals = np.floor(np.log2(safe_max)).astype(np.int32)
        scale_u8_96 = np.clip(exp_vals - 8, 0, 255).astype(np.uint8)  # [16, 96]

        # Expand scale from [16, 96] to [16, 384] by repeating each gu96 scale × 4 (for GU neurons gu96*4..gu96*4+3)
        # All 4 GU neurons in an x4 group share the same scale (same P-block, same H-values)
        scale_u8_384 = np.repeat(scale_u8_96, Q_WIDTH, axis=1)  # [16, 384]
        gate_up_scales_new[e, :, h512, :] = scale_u8_384  # [16, 384]

        # Dequantize for reference
        scale_expand_96 = np.repeat(scale_u8_96, Q_HEIGHT, axis=0)  # [128, 96]
        scale_mult = np.power(2.0, scale_expand_96.astype(np.float32))  # [128, 96]
        for p in range(PMAX):
            h_base = h512 * 512 + p * 4
            for gu96 in range(GU_SCALE_F):
                s = scale_mult[p, gu96]
                w4 = w_tile[p, gu96, :]
                w4_scaled = np.clip(w4 / (s + 1e-38), -448.0, 448.0)
                w4_fp8 = torch.tensor(w4_scaled, dtype=torch.float32).to(torch.float8_e4m3fn).to(torch.float32).numpy()
                gate_up_w_dequant[e, gu96, h_base:h_base + 4] = w4_fp8 * s

print("Weight generation done.")

# Down weights (unchanged, kept as fp8 per output neuron scale)
down_w_fp8_new = np.zeros((E, I, H), dtype=np.int8)
down_scales_new = np.zeros((E, H), dtype=np.float32)
down_w_dequant = np.zeros((E, I, H), dtype=np.float32)

for e in range(E):
    # Per-output-neuron scale: one scale per H column (per output neuron I)
    # Actually v14a down_scales[E, H] is per-H-neuron (output of down proj)
    # down_w[e, i, h] → scale per (i,h) pair? No, scale[e, h] = per output neuron h
    # So scale[e, h] = max_abs(down_w_logical[e, :, h]) and fp8 = down_w_logical / scale
    for h in range(H):
        col = down_w_logical[e, :, h]  # [I] values
        s = max(abs(col.max()), abs(col.min()), 1e-38)
        down_scales_new[e, h] = s
        col_fp8 = np.clip(col / s, -448.0, 448.0)
        col_t = torch.tensor(col_fp8, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.int8).numpy()
        down_w_fp8_new[e, :, h] = col_t
        col_dq = torch.tensor(col_fp8, dtype=torch.float32).to(torch.float8_e4m3fn).to(torch.float32).numpy()
        down_w_dequant[e, :, h] = col_dq * s

# -----------------------------------------------------------------------
# Torch BF16 reference
# -----------------------------------------------------------------------
print("Computing BF16 reference...")

def compute_bf16_reference(inp_np, gamma_np, router_w_np, gate_up_w_logical, down_w_logical):
    """Full bf16 MoE reference."""
    T = inp_np.shape[0]
    inp_t = torch.tensor(inp_np, dtype=torch.bfloat16).squeeze(1)  # [T, H]

    # RMSNorm
    gamma_t = torch.tensor(gamma_np, dtype=torch.bfloat16).squeeze(0)  # [H]
    sq = (inp_t.float() ** 2).mean(dim=-1, keepdim=True)
    norm_factor = (sq / H + 1e-6).rsqrt()
    normed = inp_t.float() * norm_factor  # [T, H]
    normed_gamma = (normed * gamma_t.float())  # [T, H]
    normed_bf16 = normed_gamma.bfloat16()

    # Router
    router_t = torch.tensor(router_w_np, dtype=torch.bfloat16)  # [H, E]
    logits = normed_bf16 @ router_t  # [T, E]
    probs = torch.softmax(logits.float(), dim=-1)  # [T, E]

    # TopK
    top_vals, top_idx = torch.topk(probs, K, dim=-1)
    norm_w = top_vals / top_vals.sum(dim=-1, keepdim=True)  # [T, K]

    # Expert MLPs
    out = torch.zeros(T, H, dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            eid = top_idx[t, k].item()
            aff = norm_w[t, k].item()
            act = normed_gamma[t]  # [H] bf16
            # gate_up: [GU=384, H] @ [H] → [GU]
            gu_w = torch.tensor(gate_up_w_logical[eid], dtype=torch.bfloat16)  # [GU, H]
            gu_out = (gu_w.float() @ act.float())  # [GU]
            gate = gu_out[:I]  # [I]
            up = gu_out[I:]    # [I]
            inter = torch.nn.functional.silu(gate) * up  # [I]
            # down: [H, I] @ [I] → [H]
            dw = torch.tensor(down_w_logical[eid], dtype=torch.bfloat16)  # [I, H]
            d_out = (dw.float().T @ inter)  # [H]
            out[t] += aff * d_out

    return out.bfloat16()

ref_bf16 = compute_bf16_reference(inp_np, gamma_np, router_w_np, gate_up_w_logical, down_w_logical)

# -----------------------------------------------------------------------
# MXFP8 reference (using dequantized weights)
# -----------------------------------------------------------------------
print("Computing MXFP8 reference...")
ref_mxfp8 = compute_bf16_reference(
    inp_np, gamma_np, router_w_np,
    gate_up_w_dequant,  # dequantized MXFP8 gate_up weights
    down_w_dequant,
)

# -----------------------------------------------------------------------
# Run NKI kernel
# -----------------------------------------------------------------------
print("Running NKI kernel...")

inp_xla = torch.tensor(inp_np, dtype=torch.bfloat16).unsqueeze(1).to(device)  # [B, 1, H]
gamma_xla = torch.tensor(gamma_np, dtype=torch.bfloat16).to(device)
router_xla = torch.tensor(router_w_np, dtype=torch.bfloat16).to(device)

gate_up_w_xla = torch.tensor(gate_up_w_new.astype(np.uint8), dtype=torch.uint8).to(device)
gate_up_scales_xla = torch.tensor(gate_up_scales_new.astype(np.uint8), dtype=torch.uint8).to(device)
down_w_xla = torch.tensor(down_w_fp8_new.astype(np.int8), dtype=torch.int8).to(device)
down_scales_xla = torch.tensor(down_scales_new, dtype=torch.float32).to(device)

result_xla = v15d.run(
    inp_xla, gamma_xla, router_xla,
    gate_up_w_xla, gate_up_scales_xla,
    down_w_xla, down_scales_xla
)
xm.mark_step()
result_np = result_xla.cpu().to(torch.float32).numpy()

# -----------------------------------------------------------------------
# Check correctness
# -----------------------------------------------------------------------
print("\n=== Correctness Check ===")

ref_bf16_np = ref_bf16.to(torch.float32).numpy()
ref_mxfp8_np = ref_mxfp8.to(torch.float32).numpy()

diff_bf16 = np.abs(result_np - ref_bf16_np)
diff_mxfp8 = np.abs(result_np - ref_mxfp8_np)

print(f"vs BF16 ref:  max_diff={diff_bf16.max():.4e}, mean_diff={diff_bf16.mean():.4e}")
print(f"vs MXFP8 ref: max_diff={diff_mxfp8.max():.4e}, mean_diff={diff_mxfp8.mean():.4e}")

# Verification 1: BF16 accuracy
try:
    np.testing.assert_allclose(result_np, ref_bf16_np, rtol=0.15, atol=0.15)
    print("Verification 1 (BF16): PASS")
except AssertionError as e:
    print(f"Verification 1 (BF16): FAIL\n  {e}")

# Verification 2: MXFP8 reference
try:
    np.testing.assert_allclose(result_np, ref_mxfp8_np, rtol=1e-2, atol=1e-2)
    print("Verification 2 (MXFP8): PASS")
except AssertionError as e:
    print(f"Verification 2 (MXFP8): FAIL\n  {e}")

print("\nDone.")
