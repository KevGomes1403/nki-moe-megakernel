"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

v15e — Plan A [X3]: Fuse post-matmul activation chain via scalar_tensor_tensor
=============================================================================
Based on v15c (Plan C [F3] gate_up weight+scale packing preserved).

Key change vs v15c (targets VectorE bottleneck — was 55.1% / 40.1 μs):
  - Round 1 profile (v15c at 72.88 μs): VectorE = 55.1% is the new
    dominant engine. Much of the VectorE work is the post-matmul
    activation / scaling chain run per (expert, i_tile).

  - v15c per i_tile post-matmul chain was:
        gate_silu  = ScalarE activation(silu, gate_psum, scale=gate_scale)
        up_scaled  = ScalarE activation(copy, up_psum,  scale=up_scale)
        inter[t]   = VectorE tensor_tensor(gate_silu, up_scaled, multiply)
    and per expert the down chain was:
        combined_s = VectorE tensor_tensor(down_scale_bufs, aff_bcast, multiply)
        down_h_raw = ScalarE activation(copy, down_psum)
        down_h_all = VectorE tensor_tensor(down_h_raw, combined_s, multiply)

  - v15e fuses these using `nisa.scalar_tensor_tensor`, which computes
    `(data <op0> operand0) <op1> operand1` in one VectorE instruction.
    Per i_tile the up-scale + GLU multiply collapses into one VectorE op
    that also drains PSUM directly:
        inter[t] = VectorE scalar_tensor_tensor(
                       data=up_psum, op0=multiply, operand0=up_scale,
                       op1=multiply, operand1=gate_silu)
    Per expert the combined-scale + down multiply collapses into one
    VectorE op:
        down_h_all = VectorE scalar_tensor_tensor(
                         data=down_h_raw, op0=multiply, operand0=aff_col_buf,
                         op1=multiply, operand1=down_scale_bufs[k])
    (the separate `combined_down_scale` intermediate buffer is removed.)

  - Math is algebraically identical (FP multiplies commute); the VectorE
    instruction internally runs fp32 math, same precision as before.
    Net savings per expert: -2 ScalarE activation(copy,...) instructions
    (the per-i_tile up-scaled drains) and -1 VectorE tensor_tensor
    (the combined_down_scale computation). PSUM rule satisfied:
    `scalar_tensor_tensor` allows data=PSUM with operand1 in SBUF — and
    we use up_psum as data and gate_silu (SBUF) as operand1.

ALL v15c / v14a optimizations are preserved:
  - SBUF hoisting, affine_range tile-1 prefetch, ring buffers across waves,
    pre-read of 8 expert IDs, rearranged up-scale bufs, pre-transposed
    weights, int8-reinterpret HWDGE on gate_up, HWDGE on down_scales,
    merged gate_up weight+scale DMA layout. Only the post-matmul
    activation / multiply chain changes.

Key change vs v14a (carried forward from v15c):
  - Phase 0 profile showed ~15% of device time was spent on GpSimdE stalls
    serializing 10 indirect expert-weight / expert-scale DMAs.
  - v14a issues 5 separate indirect DMAs per expert (gate_up_w, gate_up_scales,
    down_w tile 0, down_w tile 1, down_scales). v15c/v15e reduce this to 4 by
    co-locating `gate_up_w[e, :, :]` with `gate_up_scales[e, :]` in HBM and
    loading them in a single indirect DMA per expert.
  - This drops the per-expert DMA count from 5 to 4, and cuts descriptor-
    generation work on the GpSimd/HWDGE pipelines.

Layout:
  gate_up_packed_w [E=128, H_free + 1 = 17, PMAX = 128, GU_FLAT = 384]  int8
    - Per expert, planes 0..H_free-1 (16 planes) contain the raw fp8 weight
      bytes identical to `gate_up_w[e, :, :].view(int8)` (partition-major:
      plane h1, partition p, element g  ≡  gate_up_w[e, h1*PMAX + p, g]).
    - Plane H_free (the 17th plane) carries the scales in rearranged form.
      Row p (partition p) of plane H_free contains, in its first 12 bytes,
      the three fp32 scales `gate_up_scales[e, j*PMAX + p]` for j=0,1,2
      (i.e. GU_J_BLOCKS = 3 values). The remaining GU_FLAT - 12 = 372 bytes
      of that row are unused padding.

DMA strategy (the HWDGE-compatible int8-reinterpret trick):
  - The packed tensor is loaded as int8 -> int8 (NO dtype cast), which is
    HWDGE-compatible (HWDGE requires matching src/dst dtype).
  - In SBUF, the int8 tile is `view()`-cast to float8_e4m3 for the matmul
    stationary slot (bit-level reinterpret — no data copy), and to float32
    for the scale-plane slice (again bit-level).
  - This means the MERGED DMA can use HWDGE, unlike the naive approach of
    loading with `dtype=nl.float8_e4m3` (which would force the whole DMA to
    SWDGE because it induces an i8 -> f8 element-type cast the HWDGE block
    does not support).

ALL v14a optimizations are preserved:
  - SBUF hoisting, affine_range tile-1 prefetch, ring buffers across waves,
    pre-read of 8 expert IDs, rearranged up-scale bufs, post-matmul
    combined-scale precompute, fp8 stationary weights, pre-quantize per-
    partition scales, prg_id H-sharding under LNC=2.
  - Only the gate_up HBM storage layout and Phase-1a/Phase-1b load sites
    change. down_w / down_scales are NOT merged — see "down packing not
    attempted" note below.

down packing not attempted:
  - down_w is already split into two tiles (rows 0:I0 and I0:I0+I1) with
    different SBUF destinations. Merging down_scales into one of those tiles
    would require restructuring both tiles' layouts, with a likely wasted-
    byte cost larger than the win from saving 2 DMAs per expert.
  - Plan C scope is gate_up packing. down remains as in v14a.

Interface:
  inp              [B=1, 1, H=2048]                 bf16
  gamma            [1, H=2048]                      bf16
  router_w         [H=2048, E=128]                  bf16
  gate_up_packed_w [E=128, H_free + 1 = 17, PMAX = 128, GU_FLAT = 384]  int8
                       (see layout description above)
  down_w           [E=128, I=192, H=2048]           int8 (fp8_e4m3fn bits)
  down_scales      [E=128, H=2048]                  fp32
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.tensor_view import TensorView

# Hardware constants
_PMAX = 128       # partition dimension max
_PSUM_FREE = 512  # PSUM free-dimension max on trn2

# Qwen3-30B-A3B at TP=4 fixed dims
_H = 2048    # hidden dim
_E = 128     # num experts
_K = 8       # top-K experts
_I = 192     # actual intermediate dim per TP rank (no padding anywhere)
_I0 = 128    # first tile  (full 128 rows)
_I1 = 64     # second tile (partial: 64 valid rows, 64 zero-padded)
_I_TILES = 2 # two I-dimension tiles
_EPS = 1e-6

# Flat gate+up combined width (native layout: gate cols 0:I, up cols I:2*I)
_GU_FLAT = 2 * _I   # = 384

# LNC=2 H-sharding constants (always launched with [2])
_N_PRGS = 2
_H_FREE = _H // _PMAX             # = 16 tiles of 128 each

# Full H sharding: no TP sharding at kernel level, LNC=2 gives 1024 per prg_id
_H_SHARD_TP = _H                  # = 2048 (full H, matches BF16 baseline)
_H_SHARD = _H // _N_PRGS          # = 1024 per LNC prg_id
_H_FREE_SHARD = _H_SHARD // _PMAX  # = 8
_H_SHARD_BLOCKS = _H_SHARD // _PMAX  # = 8 (for scale access)

# Router DMA batching: 16 tiles per DMA
_ROUTER_BATCH = 16

# 2-wave constants
_K_WAVE = 4  # experts per wave

# gate_up_scales: [E, GU=384] — per output neuron
# Still 3 j-blocks of 128 output neurons each
_GU_J_BLOCKS = _GU_FLAT // _PMAX   # = 3

# v15c packed layout constants
_GU_PACKED_PLANES = _H_FREE + 1    # = 17: 16 weight planes + 1 scale plane
_SCALE_PLANE = _H_FREE             # index of the scale plane (last plane)
# After int8 view -> fp32 view, last dim 384 -> 96 (since 4 int8 per fp32)
_GU_FLAT_FP32 = _GU_FLAT // 4      # = 96


@nki.jit
def qwen3_moe_fused_tkg(
    inp,                # [B, 1, H=2048]              bf16
    gamma,              # [1, H=2048]                 bf16
    router_w,           # [H=2048, E=128]              bf16  (router stays bf16)
    gate_up_packed_w,   # [E=128, H_free+1=17, PMAX=128, GU_FLAT=384] int8
                        #   (see module docstring for layout)
    down_w,             # [E=128, I=192,  H=2048]     int8 (fp8_e4m3fn bits)
    down_scales,        # [E=128, H=2048]             fp32
):
    """
    Fused RMSNorm + Router + TopK(K=8) + Expert MLP for Qwen3 MoE TKG.
    v15c: packs gate_up weight and gate_up scale into one indirect DMA per
    (expert, wave) via int8-reinterpret, enabling HWDGE on the merged load.

    Invoked as: qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_packed_w,
                                       down_w, down_scales)
    Returns: output [T, H=2048] bf16
    """
    B = inp.shape[0]
    T = B

    H = _H
    E = _E
    K = _K
    I = _I
    I0 = _I0
    I1 = _I1
    H_free = _H_FREE
    H_free_shard = _H_FREE_SHARD   # = 8
    H_shard = _H_SHARD             # = 1024

    I_tiles = _I_TILES

    prg_id = nl.program_id(axis=0)

    # -----------------------------------------------------------------------
    # Stage 1: RMSNorm
    # -----------------------------------------------------------------------
    inp_2d = inp.reshape((T, H))
    inp_2d_hbm_reshaped = inp_2d.reshape((H_free * T, _PMAX))
    inp_flat_sb = nl.ndarray((H_free * T, _PMAX), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_reshaped, dge_mode=3)

    inp_trans_psum = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)

    rmsnorm_out = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_out[...], op=nl.copy, data=inp_trans_psum[...])

    gamma_1d = gamma.reshape((H,))
    gamma_1d_hbm_reshaped = gamma_1d.reshape((H_free, _PMAX))
    gamma_flat_sb = nl.ndarray((H_free, _PMAX), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_flat_sb, src=gamma_1d_hbm_reshaped, dge_mode=3)
    gamma_trans_psum = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=gamma_trans_psum, data=gamma_flat_sb)
    gamma_sb = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.activation(gamma_sb[...], op=nl.copy, data=gamma_trans_psum[...])

    rmsnorm_sq = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(rmsnorm_sq[...], op=nl.square, data=rmsnorm_out[...])

    rmsnorm_reduced = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(rmsnorm_reduced[0:_PMAX, 0:T], nl.add, rmsnorm_sq[0:_PMAX, 0:H_free * T], axis=1)

    gamma_mult = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(gamma_mult[...], rmsnorm_out[...], gamma_sb[...], nl.multiply)

    sum_reduced_sb = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=sum_reduced_sb[0:1, 0:T], data=rmsnorm_reduced[0:_PMAX, 0:T], op=nl.add)

    norm_sum_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=norm_sum_sb[0:1, 0:T], src=sum_reduced_sb[0:1, 0:T])
    for g in nl.affine_range(4):
        nisa.nc_stream_shuffle(
            dst=norm_sum_sb[nl.ds(g * 32, 32), 0:T],
            src=norm_sum_sb[0:1, 0:T],
            shuffle_mask=[0] * 32,
        )

    eps_sb = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(eps_sb, value=_EPS)
    norm_factor_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        norm_factor_sb[0:_PMAX, 0:T],
        op=nl.rsqrt,
        data=norm_sum_sb[0:_PMAX, 0:T],
        scale=1.0 / H,
        bias=eps_sb[0:_PMAX, :],
    )

    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=2).broadcast(dim=2, size=H_free)
    rmsnorm_normed = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(rmsnorm_normed[...], gamma_mult[...], norm_factor_bcast.get_view(), nl.multiply)

    rmsnorm_normed_bf16 = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=rmsnorm_normed[...])

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul [T, H] @ [H, E=128] → logits [T, E]
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)
    router_w_wide_sb = nl.ndarray((_PMAX, _ROUTER_BATCH, E), dtype=inp.dtype, buffer=nl.sbuf)

    for h_chunk in nl.affine_range(H_free // _ROUTER_BATCH):
        nisa.dma_copy(
            dst=router_w_wide_sb,
            src=router_w.ap(
                pattern=[[E, _PMAX], [_PMAX * E, _ROUTER_BATCH], [1, E]],
                offset=h_chunk * _ROUTER_BATCH * _PMAX * E,
            ),
            dge_mode=3,
        )
        for h_sub in nl.affine_range(_ROUTER_BATCH):
            h1 = h_chunk * _ROUTER_BATCH + h_sub
            nisa.nc_matmul(
                dst=logits_psum[0:T, 0:E],
                stationary=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                moving=router_w_wide_sb[0:_PMAX, h_sub, 0:E],
            )

    logits_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(logits_sb[0:T, 0:E], op=nl.copy, data=logits_psum[0:T, 0:E])

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8)
    # -----------------------------------------------------------------------
    max_logit = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(max_logit[0:T, 0:1], nl.maximum, logits_sb[0:T, 0:E], axis=1)

    centered = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        centered[0:T, 0:E], data=logits_sb[0:T, 0:E],
        op0=nl.subtract, operand0=max_logit[0:T, 0:1],
    )

    exp_vals = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(exp_vals[0:T, 0:E], op=nl.exp, data=centered[0:T, 0:E])

    sum_exp = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(sum_exp[0:T, 0:1], nl.add, exp_vals[0:T, 0:E], axis=1)

    inv_sum_exp = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(inv_sum_exp[0:T, 0:1], op=nl.reciprocal, data=sum_exp[0:T, 0:1])

    probs = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        probs[0:T, 0:E], data=exp_vals[0:T, 0:E],
        op0=nl.multiply, operand0=inv_sum_exp[0:T, 0:1],
    )

    top8_vals = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_vals[0:T, 0:K], src=probs[0:T, 0:E])

    top8_idx = nl.ndarray((T, K), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=top8_idx[0:T, 0:K], data=probs[0:T, 0:E], vals=top8_vals[0:T, 0:K])

    # -----------------------------------------------------------------------
    # Stage 4: Selective-Expert MLP — 2-Wave Expert Processing
    # -----------------------------------------------------------------------
    # V12i: output_temp is [128, H_free_shard=8] (T=1 always in TKG)
    output_temp = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.affine_range(T):

        # ------------------------------------------------------------------
        # Allocate merged gate_up weight+scale buffers — Wave 0
        # int8 storage so HWDGE can load without a dtype cast; views into
        # fp8 (for matmul) and fp32 (for scales) are taken later.
        # ------------------------------------------------------------------
        gate_up_packed_buf0 = nl.ndarray((_PMAX, _GU_PACKED_PLANES, _GU_FLAT), dtype=nl.int8, buffer=nl.sbuf)
        gate_up_packed_buf1 = nl.ndarray((_PMAX, _GU_PACKED_PLANES, _GU_FLAT), dtype=nl.int8, buffer=nl.sbuf)
        gate_up_packed_buf2 = nl.ndarray((_PMAX, _GU_PACKED_PLANES, _GU_FLAT), dtype=nl.int8, buffer=nl.sbuf)
        gate_up_packed_buf3 = nl.ndarray((_PMAX, _GU_PACKED_PLANES, _GU_FLAT), dtype=nl.int8, buffer=nl.sbuf)
        gate_up_packed_bufs = [gate_up_packed_buf0, gate_up_packed_buf1, gate_up_packed_buf2, gate_up_packed_buf3]

        # fp8 views (weight region) — used as stationary for matmul
        gate_up_fp8_buf0_v = gate_up_packed_buf0.view(nl.float8_e4m3)
        gate_up_fp8_buf1_v = gate_up_packed_buf1.view(nl.float8_e4m3)
        gate_up_fp8_buf2_v = gate_up_packed_buf2.view(nl.float8_e4m3)
        gate_up_fp8_buf3_v = gate_up_packed_buf3.view(nl.float8_e4m3)
        gate_up_fp8_bufs = [gate_up_fp8_buf0_v, gate_up_fp8_buf1_v, gate_up_fp8_buf2_v, gate_up_fp8_buf3_v]

        # fp32 views (entire buffer) — scale plane is plane index _SCALE_PLANE
        gate_up_f32_buf0_v = gate_up_packed_buf0.view(nl.float32)
        gate_up_f32_buf1_v = gate_up_packed_buf1.view(nl.float32)
        gate_up_f32_buf2_v = gate_up_packed_buf2.view(nl.float32)
        gate_up_f32_buf3_v = gate_up_packed_buf3.view(nl.float32)
        gate_up_scale_full_f32 = [gate_up_f32_buf0_v, gate_up_f32_buf1_v, gate_up_f32_buf2_v, gate_up_f32_buf3_v]

        # up scale rearranged buffers for post-matmul scale
        up_scale_i0_buf0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_buf1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_buf2 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_buf3 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_bufs = [up_scale_i0_buf0, up_scale_i0_buf1, up_scale_i0_buf2, up_scale_i0_buf3]

        up_scale_i1_buf0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_buf1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_buf2 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_buf3 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_bufs = [up_scale_i1_buf0, up_scale_i1_buf1, up_scale_i1_buf2, up_scale_i1_buf3]

        # Consolidated gate scale buffers (extracted from packed tile) — [128, 3]
        # These are copied out of the scale plane so that downstream code that
        # indexes `gate_up_scale_bufs[k][p, i_tile]` still works identically
        # to v14a.
        gate_up_scale_buf0 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_buf1 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_buf2 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_buf3 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_bufs = [gate_up_scale_buf0, gate_up_scale_buf1, gate_up_scale_buf2, gate_up_scale_buf3]

        # down fp8 weight buffers (H_shard=1024 per prg_id)
        down_full0_fp8_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_bufs = [down_full0_fp8_buf0, down_full0_fp8_buf1, down_full0_fp8_buf2, down_full0_fp8_buf3]

        down_full1_fp8_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_bufs = [down_full1_fp8_buf0, down_full1_fp8_buf1, down_full1_fp8_buf2, down_full1_fp8_buf3]

        # down scale buffers: [128P, 8F]
        down_scale_buf0 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_buf1 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_buf2 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_buf3 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_bufs = [down_scale_buf0, down_scale_buf1, down_scale_buf2, down_scale_buf3]

        # ------------------------------------------------------------------
        # Allocate fp8 weight buffers — Wave 1 (separate to avoid data hazards)
        # ------------------------------------------------------------------
        gate_up_packed_w1_buf0 = nl.ndarray((_PMAX, _GU_PACKED_PLANES, _GU_FLAT), dtype=nl.int8, buffer=nl.sbuf)
        gate_up_packed_w1_buf1 = nl.ndarray((_PMAX, _GU_PACKED_PLANES, _GU_FLAT), dtype=nl.int8, buffer=nl.sbuf)
        gate_up_packed_w1_buf2 = nl.ndarray((_PMAX, _GU_PACKED_PLANES, _GU_FLAT), dtype=nl.int8, buffer=nl.sbuf)
        gate_up_packed_w1_buf3 = nl.ndarray((_PMAX, _GU_PACKED_PLANES, _GU_FLAT), dtype=nl.int8, buffer=nl.sbuf)
        gate_up_packed_w1_bufs = [gate_up_packed_w1_buf0, gate_up_packed_w1_buf1, gate_up_packed_w1_buf2, gate_up_packed_w1_buf3]

        gate_up_fp8_w1_buf0_v = gate_up_packed_w1_buf0.view(nl.float8_e4m3)
        gate_up_fp8_w1_buf1_v = gate_up_packed_w1_buf1.view(nl.float8_e4m3)
        gate_up_fp8_w1_buf2_v = gate_up_packed_w1_buf2.view(nl.float8_e4m3)
        gate_up_fp8_w1_buf3_v = gate_up_packed_w1_buf3.view(nl.float8_e4m3)
        gate_up_fp8_w1_bufs = [gate_up_fp8_w1_buf0_v, gate_up_fp8_w1_buf1_v, gate_up_fp8_w1_buf2_v, gate_up_fp8_w1_buf3_v]

        gate_up_f32_w1_buf0_v = gate_up_packed_w1_buf0.view(nl.float32)
        gate_up_f32_w1_buf1_v = gate_up_packed_w1_buf1.view(nl.float32)
        gate_up_f32_w1_buf2_v = gate_up_packed_w1_buf2.view(nl.float32)
        gate_up_f32_w1_buf3_v = gate_up_packed_w1_buf3.view(nl.float32)
        gate_up_scale_full_f32_w1 = [gate_up_f32_w1_buf0_v, gate_up_f32_w1_buf1_v, gate_up_f32_w1_buf2_v, gate_up_f32_w1_buf3_v]

        gate_up_scale_w1_buf0 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_w1_buf1 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_w1_buf2 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_w1_buf3 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_w1_bufs = [gate_up_scale_w1_buf0, gate_up_scale_w1_buf1, gate_up_scale_w1_buf2, gate_up_scale_w1_buf3]

        up_scale_i0_w1_buf0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_w1_buf1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_w1_buf2 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_w1_buf3 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_w1_bufs = [up_scale_i0_w1_buf0, up_scale_i0_w1_buf1, up_scale_i0_w1_buf2, up_scale_i0_w1_buf3]

        up_scale_i1_w1_buf0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_w1_buf1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_w1_buf2 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_w1_buf3 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_w1_bufs = [up_scale_i1_w1_buf0, up_scale_i1_w1_buf1, up_scale_i1_w1_buf2, up_scale_i1_w1_buf3]

        down_full0_fp8_w1_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_w1_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_w1_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_w1_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_w1_bufs = [down_full0_fp8_w1_buf0, down_full0_fp8_w1_buf1, down_full0_fp8_w1_buf2, down_full0_fp8_w1_buf3]

        down_full1_fp8_w1_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_w1_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_w1_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_w1_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_w1_bufs = [down_full1_fp8_w1_buf0, down_full1_fp8_w1_buf1, down_full1_fp8_w1_buf2, down_full1_fp8_w1_buf3]

        down_scale_w1_buf0 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_w1_buf1 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_w1_buf2 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_w1_buf3 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_w1_bufs = [down_scale_w1_buf0, down_scale_w1_buf1, down_scale_w1_buf2, down_scale_w1_buf3]

        # ------------------------------------------------------------------
        # Pre-allocate all 8 expert ID scratch buffers
        # ------------------------------------------------------------------
        eid_all_0 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_1 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_2 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_3 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_4 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_5 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_6 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_7 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_bufs = [eid_all_0, eid_all_1, eid_all_2, eid_all_3,
                        eid_all_4, eid_all_5, eid_all_6, eid_all_7]

        # zero_bias for post-matmul activation scale ([128,1] — for gate/up)
        zero_bias = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(zero_bias, value=0.0)

        # Zero-pad down_full1 rows I1:I0 (fp8 buffers) — Wave 0
        for k_pad in nl.affine_range(4):
            nisa.memset(down_full1_fp8_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        # Zero-pad down_full1_w1 rows I1:I0 (fp8 buffers) — Wave 1
        for k_pad in nl.affine_range(4):
            nisa.memset(down_full1_fp8_w1_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        # ------------------------------------------------------------------
        # V12i: gate_t1_128 and up_t1_128 with k-dimension for affine_range prefetch
        # Wave 0: [_K_WAVE=4, _PMAX=128, H_free=16, I0=128]
        # SBUF cost: 2 * 4 * 128 * 16 * 128 = 2 MB (fp8)
        # ------------------------------------------------------------------
        gate_t1_128_k0 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_k1 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_k2 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_k3 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_bufs = [gate_t1_128_k0, gate_t1_128_k1, gate_t1_128_k2, gate_t1_128_k3]

        up_t1_128_k0 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_k1 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_k2 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_k3 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_bufs = [up_t1_128_k0, up_t1_128_k1, up_t1_128_k2, up_t1_128_k3]

        # Zero-pad tile-1 I1..I0 region (Wave 0)
        for k_pad in nl.affine_range(_K_WAVE):
            nisa.memset(gate_t1_128_bufs[k_pad][0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
            nisa.memset(up_t1_128_bufs[k_pad][0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        # Wave 1 prefetch buffers
        gate_t1_128_w1_k0 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_w1_k1 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_w1_k2 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_w1_k3 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_w1_bufs = [gate_t1_128_w1_k0, gate_t1_128_w1_k1, gate_t1_128_w1_k2, gate_t1_128_w1_k3]

        up_t1_128_w1_k0 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_w1_k1 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_w1_k2 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_w1_k3 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_w1_bufs = [up_t1_128_w1_k0, up_t1_128_w1_k1, up_t1_128_w1_k2, up_t1_128_w1_k3]

        # Zero-pad tile-1 I1..I0 region (Wave 1)
        for k_pad in nl.affine_range(_K_WAVE):
            nisa.memset(gate_t1_128_w1_bufs[k_pad][0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
            nisa.memset(up_t1_128_w1_bufs[k_pad][0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        # ==================================================================
        # Pre-read all 8 expert IDs — 8 tiny 4-byte DMAs, complete nearly instantly
        # ==================================================================
        for k8 in nl.affine_range(8):
            nisa.dma_copy(
                dst=eid_all_bufs[k8][0:1, 0:1],
                src=top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k8),
            )

        # Phase 1a: Load Wave 0 experts — affine_range allows pipelining across k
        # v15c [F3]: Single merged DMA per expert for (gate_up_w + gate_up_scales).
        # int8 -> int8 (no dtype cast) so HWDGE can be used.
        for k in nl.affine_range(_K_WAVE):
            expert_id = eid_all_bufs[k].ap(pattern=[[1, 1], [1, 1]], offset=0)

            # Merged gate_up (weight + scale) load — SINGLE DMA, HWDGE
            # Packed HBM shape: [E, H_free + 1, PMAX, GU_FLAT] int8
            # Stride pattern mirrors original gate_up_w pattern but with
            # H_free + 1 planes instead of H_free.
            nisa.dma_copy(
                dst=gate_up_packed_bufs[k],
                src=gate_up_packed_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, _GU_PACKED_PLANES], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.hwdge,
            )

            # Load fp8 down_w tile 0 (rows 0:I0=128) for this shard (H_shard=1024)
            nisa.dma_copy(
                dst=down_full0_fp8_bufs[k],
                src=down_w.ap(
                    pattern=[[H, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3,
                ),
                dge_mode=nisa.dge_mode.swdge,
            )

            # Load fp8 down_w tile 1 (rows I0:I0+I1=64) for this shard
            nisa.dma_copy(
                dst=down_full1_fp8_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H_shard]],
                    offset=I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3,
                ),
                dge_mode=nisa.dge_mode.swdge,
            )

            # Load down_scales[e, prg_id*1024 : prg_id*1024+1024] → [128P, 8F]
            # fp32 -> fp32, no cast; safe for HWDGE.
            nisa.dma_copy(
                dst=down_scale_bufs[k],
                src=down_scales.ap(
                    pattern=[[1, _PMAX], [_PMAX, _H_SHARD_BLOCKS]],
                    offset=prg_id * _H_SHARD,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.hwdge,
            )

        # Phase 1b: Load Wave 1 experts — affine_range allows pipelining across k
        for k in nl.affine_range(_K_WAVE):
            kk = k + 4
            expert_id_w1 = eid_all_bufs[kk].ap(pattern=[[1, 1], [1, 1]], offset=0)

            # Merged gate_up (weight + scale) load — SINGLE DMA, HWDGE
            nisa.dma_copy(
                dst=gate_up_packed_w1_bufs[k],
                src=gate_up_packed_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, _GU_PACKED_PLANES], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.hwdge,
            )

            # Load fp8 down_w tile 0 for Wave 1
            nisa.dma_copy(
                dst=down_full0_fp8_w1_bufs[k],
                src=down_w.ap(
                    pattern=[[H, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3,
                ),
                dge_mode=nisa.dge_mode.swdge,
            )

            # Load fp8 down_w tile 1 for Wave 1
            nisa.dma_copy(
                dst=down_full1_fp8_w1_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H_shard]],
                    offset=I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3,
                ),
                dge_mode=nisa.dge_mode.swdge,
            )

            # Load down_scales for Wave 1 — HWDGE (fp32 -> fp32, no cast)
            nisa.dma_copy(
                dst=down_scale_w1_bufs[k],
                src=down_scales.ap(
                    pattern=[[1, _PMAX], [_PMAX, _H_SHARD_BLOCKS]],
                    offset=prg_id * _H_SHARD,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.hwdge,
            )

        # ------------------------------------------------------------------
        # Extract gate_up_scale_bufs[k] from the scale plane of each packed
        # buffer. After int8 -> fp32 view, the scale plane (plane index
        # _SCALE_PLANE) occupies the first 3 fp32 elements along the last
        # dim. We copy [PMAX, 3] fp32 out into gate_up_scale_bufs[k] so the
        # downstream code matches v14a exactly.
        # ------------------------------------------------------------------
        for k in nl.affine_range(_K_WAVE):
            nisa.tensor_copy(
                dst=gate_up_scale_bufs[k][0:_PMAX, 0:_GU_J_BLOCKS],
                src=gate_up_scale_full_f32[k][0:_PMAX, _SCALE_PLANE, 0:_GU_J_BLOCKS],
            )
        for k in nl.affine_range(_K_WAVE):
            nisa.tensor_copy(
                dst=gate_up_scale_w1_bufs[k][0:_PMAX, 0:_GU_J_BLOCKS],
                src=gate_up_scale_full_f32_w1[k][0:_PMAX, _SCALE_PLANE, 0:_GU_J_BLOCKS],
            )

        # ------------------------------------------------------------------
        # Compute norm_weights (overlaps with in-flight DMAs)
        # ------------------------------------------------------------------
        sum_topk = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(sum_topk[0:T, 0:1], nl.add, top8_vals[0:T, 0:K], axis=1)

        inv_sum_topk = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(inv_sum_topk[0:T, 0:1], op=nl.reciprocal, data=sum_topk[0:T, 0:1])

        norm_weights = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            norm_weights[0:T, 0:K], data=top8_vals[0:T, 0:K],
            op0=nl.multiply, operand0=inv_sum_topk[0:T, 0:1],
        )

        aff_bcast = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K], src=norm_weights[t:t + 1, 0:K])
        for g in nl.affine_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        # Assemble up scale rearranged buffers — Wave 0 (affine_range: depends on weight loads)
        for k in nl.affine_range(_K_WAVE):
            nisa.tensor_copy(
                dst=up_scale_i0_bufs[k][0:I1, 0:1],
                src=gate_up_scale_bufs[k][nl.ds(I1, I1), 1:2],
            )
            nisa.tensor_copy(
                dst=up_scale_i0_bufs[k][nl.ds(I1, I1), 0:1],
                src=gate_up_scale_bufs[k][0:I1, 2:3],
            )
            nisa.tensor_copy(
                dst=up_scale_i1_bufs[k][0:I1, 0:1],
                src=gate_up_scale_bufs[k][nl.ds(I1, I1), 2:3],
            )

        # Assemble up scale rearranged buffers — Wave 1 (affine_range: depends on weight loads)
        for k in nl.affine_range(_K_WAVE):
            nisa.tensor_copy(
                dst=up_scale_i0_w1_bufs[k][0:I1, 0:1],
                src=gate_up_scale_w1_bufs[k][nl.ds(I1, I1), 1:2],
            )
            nisa.tensor_copy(
                dst=up_scale_i0_w1_bufs[k][nl.ds(I1, I1), 0:1],
                src=gate_up_scale_w1_bufs[k][0:I1, 2:3],
            )
            nisa.tensor_copy(
                dst=up_scale_i1_w1_bufs[k][0:I1, 0:1],
                src=gate_up_scale_w1_bufs[k][nl.ds(I1, I1), 2:3],
            )

        # PSUM allocation for wave 0
        gate_up_psum = nl.ndarray((_PMAX, _K_WAVE * 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
        down_psum    = nl.ndarray((_PMAX, _K_WAVE * H_free_shard), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # ==================================================================
        # V12i Plan I: affine_range prefetch loop for tile-1 tensor_copies (Wave 0)
        # Uses fp8-viewed weight region from the packed tile.
        # ==================================================================
        for k in nl.affine_range(_K_WAVE):
            nisa.tensor_copy(
                dst=gate_t1_128_bufs[k][0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_fp8_bufs[k][0:_PMAX, 0:H_free, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128_bufs[k][0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_fp8_bufs[k][0:_PMAX, 0:H_free, nl.ds(I + I0, I1)],
            )

        # ------------------------------------------------------------------
        # V12i: Initialize output_temp to zero before Phase 2a loop.
        # ------------------------------------------------------------------
        nisa.memset(output_temp, value=0.0)

        # Phase 2a: Compute experts 0-3 (fp8 stationary)
        # Wave 1 DMA transfers overlap here since they were issued above.
        for k in nl.affine_range(_K_WAVE):
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            # Gate/Up matmul (fp8 stationary) — tile-1 data from prefetch buffers
            for h1 in nl.affine_range(H_free):
                for i_tile in nl.affine_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_fp8_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gate_up_fp8_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                    else:
                        g_stat = gate_t1_128_bufs[k][0:_PMAX, h1, 0:I0]
                        u_stat = up_t1_128_bufs[k][0:_PMAX, h1, 0:I0]
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                        stationary=g_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )

            # Post-matmul (v15e [X3]): fused gate drain+scale+SiLU on ScalarE,
            # then single VectorE scalar_tensor_tensor folding (up_psum * up_scale)
            # * gate_silu into one instruction per i_tile. This replaces the
            # per-i_tile ScalarE activation(copy, scale=up_scale) AND the
            # subsequent VectorE tensor_tensor GLU multiply with a single
            # VectorE op.
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)

            for i_tile in nl.affine_range(I_tiles):
                gate_scale_col = gate_up_scale_bufs[k][0:_PMAX, i_tile:i_tile + 1]

                # Fuse gate scale + SiLU into single ScalarE activation call
                gate_silu = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    gate_silu,
                    op=nl.silu,
                    data=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                    scale=gate_scale_col,
                    bias=zero_bias,
                )

                if i_tile == 0:
                    up_scale_col = up_scale_i0_bufs[k]
                else:
                    up_scale_col = up_scale_i1_bufs[k]

                # [X3] Fused: inter_f32 = (up_psum * up_scale_col) * gate_silu
                # PSUM rule OK: data=PSUM and operand1=gate_silu (SBUF).
                nisa.scalar_tensor_tensor(
                    dst=inter_f32[0:_PMAX, i_tile:i_tile + 1],
                    data=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                    op0=nl.multiply,
                    operand0=up_scale_col,
                    op1=nl.multiply,
                    operand1=gate_silu,
                )

            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            # Down matmul (fp8 stationary)
            for h1_out in nl.affine_range(H_free_shard):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full0_fp8_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full1_fp8_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            # Affinity column for this expert (Wave 0: use k) — scalar column
            # used in-place by scalar_tensor_tensor below, no broadcast needed.
            aff_col_buf = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=aff_col_buf, src=aff_bcast[0:_PMAX, k:k + 1])

            # Batch PSUM drain — ScalarE copy
            down_h_raw = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_h_raw,
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + _H_FREE_SHARD],
            )

            # [X3] Fused: down_h_all = (down_h_raw * aff_col_buf) * down_scale_bufs[k]
            # Replaces the previous combined_down_scale precompute + down_h_all
            # multiply (2 VectorE tensor_tensor ops) with a single VectorE op.
            down_h_all = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=down_h_all,
                data=down_h_raw,
                op0=nl.multiply,
                operand0=aff_col_buf,
                op1=nl.multiply,
                operand1=down_scale_bufs[k],
            )

            # V12i: Always-add to output_temp
            nisa.tensor_tensor(
                dst=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                data1=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                data2=down_h_all[0:_PMAX, 0:_H_FREE_SHARD],
                op=nl.add,
            )

        # ==================================================================
        # WAVE 1: Experts 4-7 — use dedicated w1 buffers (already loaded above)
        # ==================================================================

        # PSUM memset for wave 1
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # ==================================================================
        # V12i Plan I: affine_range prefetch loop for tile-1 tensor_copies (Wave 1)
        # ==================================================================
        for k in nl.affine_range(_K_WAVE):
            nisa.tensor_copy(
                dst=gate_t1_128_w1_bufs[k][0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_fp8_w1_bufs[k][0:_PMAX, 0:H_free, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128_w1_bufs[k][0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_fp8_w1_bufs[k][0:_PMAX, 0:H_free, nl.ds(I + I0, I1)],
            )

        # Phase 2b: Compute experts 4-7 (using w1 buffers)
        for k in nl.affine_range(_K_WAVE):
            kk = k + 4
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            for h1 in nl.affine_range(H_free):
                for i_tile in nl.affine_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_fp8_w1_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gate_up_fp8_w1_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                    else:
                        g_stat = gate_t1_128_w1_bufs[k][0:_PMAX, h1, 0:I0]
                        u_stat = up_t1_128_w1_bufs[k][0:_PMAX, h1, 0:I0]
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                        stationary=g_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )

            # Post-matmul (v15e [X3]): same fusion pattern as Wave 0.
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)

            for i_tile in nl.affine_range(I_tiles):
                gate_scale_col = gate_up_scale_w1_bufs[k][0:_PMAX, i_tile:i_tile + 1]

                # Fuse gate scale + SiLU into single ScalarE activation call
                gate_silu = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    gate_silu,
                    op=nl.silu,
                    data=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                    scale=gate_scale_col,
                    bias=zero_bias,
                )

                if i_tile == 0:
                    up_scale_col = up_scale_i0_w1_bufs[k]
                else:
                    up_scale_col = up_scale_i1_w1_bufs[k]

                # [X3] Fused: inter_f32 = (up_psum * up_scale_col) * gate_silu
                nisa.scalar_tensor_tensor(
                    dst=inter_f32[0:_PMAX, i_tile:i_tile + 1],
                    data=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                    op0=nl.multiply,
                    operand0=up_scale_col,
                    op1=nl.multiply,
                    operand1=gate_silu,
                )

            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            # Down matmul (fp8 stationary) — using w1 buffers
            for h1_out in nl.affine_range(H_free_shard):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full0_fp8_w1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full1_fp8_w1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            # Affinity column for this expert (Wave 1: use kk)
            aff_col_buf = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=aff_col_buf, src=aff_bcast[0:_PMAX, kk:kk + 1])

            # Batch PSUM drain — ScalarE copy
            down_h_raw = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_h_raw,
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + _H_FREE_SHARD],
            )

            # [X3] Fused: down_h_all = (down_h_raw * aff_col_buf) * down_scale_w1_bufs[k]
            down_h_all = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=down_h_all,
                data=down_h_raw,
                op0=nl.multiply,
                operand0=aff_col_buf,
                op1=nl.multiply,
                operand1=down_scale_w1_bufs[k],
            )

            nisa.tensor_tensor(
                dst=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                data1=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                data2=down_h_all[0:_PMAX, 0:_H_FREE_SHARD],
                op=nl.add,
            )

    # -----------------------------------------------------------------------
    # Stage 5: Transpose fp32->bf16, store to HBM
    # -----------------------------------------------------------------------
    output = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.shared_hbm)
    out_sb = nl.ndarray((T, H_shard), dtype=inp.dtype, buffer=nl.sbuf)

    for h1 in nl.affine_range(H_free_shard):
        tp_psum = nl.ndarray((T, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp_psum[0:T, 0:_PMAX], data=output_temp[0:_PMAX, h1:h1 + 1])
        nisa.activation(
            dst=out_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)],
            op=nl.copy,
            data=tp_psum[0:T, 0:_PMAX],
        )

    nisa.dma_copy(
        dst=output[0:T, nl.ds(prg_id * H_shard, H_shard)],
        src=out_sb[0:T, 0:H_shard],
    )

    return output


def pack_gate_up(gate_up_w_i8, gate_up_scales_f32):
    """Build the v15c packed gate_up HBM tensor.

    Input:
      gate_up_w_i8:       [E, H=2048, GU_FLAT=384] int8 (fp8_e4m3fn bits)
      gate_up_scales_f32: [E, GU_FLAT=384]         fp32

    Output:
      packed: [E, H_free + 1 = 17, PMAX = 128, GU_FLAT = 384] int8

    Layout per expert:
      planes [0 .. H_free-1]: weight planes, where
        packed[e, h1, p, g]  ==  gate_up_w_i8[e, h1*PMAX + p, g]
      plane [H_free] (the scale plane):
        For partition p, the first 12 bytes (as fp32) are the three scales
        gate_up_scales_f32[e, j*PMAX + p] for j=0,1,2.
        The remaining GU_FLAT - 12 = 372 bytes of that row are zeros.
    """
    import torch
    assert gate_up_w_i8.dtype == torch.int8
    assert gate_up_scales_f32.dtype == torch.float32
    E = gate_up_w_i8.shape[0]
    H = gate_up_w_i8.shape[1]
    GU = gate_up_w_i8.shape[2]
    PMAX = _PMAX
    H_FREE = H // PMAX
    GU_J_BLOCKS = GU // PMAX
    PLANES = H_FREE + 1
    assert gate_up_scales_f32.shape == (E, GU)

    # Allocate packed tensor [E, PLANES, PMAX, GU] int8, zero-initialized
    packed = torch.zeros(E, PLANES, PMAX, GU, dtype=torch.int8)

    # Weight planes: reshape from [E, H=H_FREE*PMAX, GU] -> [E, H_FREE, PMAX, GU]
    packed[:, :H_FREE, :, :] = gate_up_w_i8.view(E, H_FREE, PMAX, GU)

    # Scale plane: For each expert e, partition p, write 3 fp32 scales
    # into the first 12 bytes of the row.
    # scales[:, j*PMAX + p] for j=0,1,2 → reshape [E, GU_J_BLOCKS, PMAX]
    # then transpose to [E, PMAX, GU_J_BLOCKS] so partition-p sees 3 consecutive scales.
    scales_view = gate_up_scales_f32.view(E, GU_J_BLOCKS, PMAX).transpose(1, 2).contiguous()
    # shape: [E, PMAX, GU_J_BLOCKS=3] fp32
    scales_bytes = scales_view.view(torch.int8).view(E, PMAX, GU_J_BLOCKS * 4)  # [E, PMAX, 12]
    packed[:, H_FREE, :, :GU_J_BLOCKS * 4] = scales_bytes

    return packed.contiguous()


def run(inp, gamma, router_w, gate_up_packed_w, down_w, down_scales):
    """Run v15c kernel (Plan C [F3]: merged gate_up weight+scale DMA on trn3).

    Accepts:
      inp:              [B, 1, H=2048] bf16
      gamma:            [1, H=2048] bf16
      router_w:         [H=2048, E=128] bf16
      gate_up_packed_w: [E=128, H_free+1=17, PMAX=128, GU_FLAT=384] int8
                        (see module docstring; use `pack_gate_up` helper)
      down_w:           [E=128, I=192, H=2048] int8 (fp8_e4m3fn bits)
      down_scales:      [E=128, H=2048] fp32

    Returns: output [T, H=2048] bf16
    """
    import torch
    import torch_xla.core.xla_model as xm

    assert gate_up_packed_w.shape == (_E, _GU_PACKED_PLANES, _PMAX, _GU_FLAT), (
        f"gate_up_packed_w shape {gate_up_packed_w.shape}, expected "
        f"({_E}, {_GU_PACKED_PLANES}, {_PMAX}, {_GU_FLAT})"
    )
    assert down_w.shape == (_E, _I, _H), f"down_w shape {down_w.shape}"
    assert down_scales.shape == (_E, _H), f"down_scales shape {down_scales.shape}"

    xm.mark_step()

    outputs = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_packed_w, down_w, down_scales)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
