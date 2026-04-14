"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

v15d — Plan D: nc_matmul_mx MXFP8 Gate_Up Projection
=====================================================
Based on v14a (trn3 port of v12i).

Key changes vs v14a:
  - Gate_up weights stored as int8 raw bytes: [E=128, 128_H_P, n_H512=4, GU=384]
    Each element is float8_e4m3fn raw byte; x4 packs 4 H-contraction values per partition
  - Gate_up scales: [E=128, 16_scale_P, n_H512=4, GU_groups=96] uint8 (dense HBM)
    Loaded into sparse SBUF quadrant layout [128_P, n_H512, GU_groups]
  - BF16 activations quantized online via nisa.quantize_mx
    Layout adapter: [128_P, H_free*T] → [128_P, n_H512, T_pad, 4_H] → quantize → [128_P, n_H512*T_pad]
  - nc_matmul_mx replaces nc_matmul for gate_up; down projection unchanged.

Weight layout:
  gate_up_w:     [E=128, 128_H_P, n_H512=4, GU=384]    float8_e4m3fn_x4 (x4 packs H direction)
  gate_up_scales:[E=128, 16_scale_P, n_H512=4, GU=384] uint8 (unpacked GU dim; one scale per 8P×1GU block)

Scale layout:
  HBM: [E, 16_scale_P, n_H512, GU=384] — dense, unpacked GU dimension
  SBUF: [128_P, n_H512, GU=384] — sparse (rows 0-3, 32-35, 64-67, 96-99 have data)
  nc_matmul_mx uses offsets in unpacked GU units: 0, 96, 192, 288 (= q_idx * (GU//4))

Interface:
  gate_up_w      [E=128, 128_H_P, n_H512=4, GU=384]    float8_e4m3fn_x4
  gate_up_scales [E=128, 16_scale_P, n_H512=4, GU=384] uint8
  down_w         [E=128, I=192,  H=2048]   int8 (unchanged)
  down_scales    [E=128, H=2048]           fp32 (unchanged)
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.tensor_view import TensorView

# Hardware constants
_PMAX = 128       # partition dimension max
_PSUM_FREE = 512

# Qwen3-30B-A3B at TP=4 fixed dims
_H = 2048
_E = 128
_K = 8
_I = 192
_I0 = 128    # first I-tile (full)
_I1 = 64     # second I-tile (partial, zero-padded to 128)
_I_TILES = 2
_EPS = 1e-6

_GU_FLAT = 2 * _I   # = 384

_N_PRGS = 2
_H_FREE = _H // _PMAX             # = 16

_H_SHARD = _H // _N_PRGS          # = 1024
_H_FREE_SHARD = _H_SHARD // _PMAX  # = 8
_H_SHARD_BLOCKS = _H_SHARD // _PMAX  # = 8

_ROUTER_BATCH = 16
_K_WAVE = 4

# MXFP8 / nc_matmul_mx constants
# One H512 tile = 128_P × 4_H_packed = 512 unpacked H-contraction elements
_Q_WIDTH = 4      # _q_width: H elements packed per partition row
_Q_HEIGHT = 8     # _q_height: partitions per scale block
_N_H512 = _H // 512               # = 4  (H/512 tiles)
_T_PAD = 4                         # T=1 padded to 4 for quantize_mx
# After layout adapter: [128, n_H512, T_pad] quantized, F per tile = T_pad x4 elems
# GU_groups = GU_FLAT after quantize (weight x4 packs GU? No — weight x4 packs H)
# Weight stationary: [128_P, n_H512, GU_FLAT] — x4 packs H direction (4 H per row)
# So GU_FLAT = 384 is the raw F dimension of the weight (no x4 packing in GU direction)
# Each of the 384 raw FP8 bytes is one GU neuron's weight for one (P, H512) block
_GU_FLAT_W = _GU_FLAT              # = 384 (raw weight F dimension, unpacked)
# Scale: one scale per 8_P × 4_F — F is in GU direction (UNPACKED).
# x4 packing is in the H (P) direction, NOT in the GU/I direction.
# scale_P = 128 / 8 = 16, scale_F = GU_FLAT = 384 (same indexing as weight)
_SCALE_P = _PMAX // _Q_HEIGHT      # = 16

_QUAD_SIZE = 32   # SBUF partitions per quadrant
_SCALE_PER_QUAD = 4               # scale rows per quadrant
_N_QUADS = _PMAX // _QUAD_SIZE    # = 4

_GU_J_BLOCKS = _GU_FLAT // _PMAX  # = 3 (legacy, for down scales only)


@nki.jit
def qwen3_moe_fused_tkg(
    inp,              # [B, 1, H=2048]              bf16
    gamma,            # [1, H=2048]                 bf16
    router_w,         # [H=2048, E=128]              bf16
    gate_up_w,        # [E=128, 128_H_P, n_H512=4, GU=384]  float8_e4m3fn_x4 (x4 packs H direction)
    gate_up_scales,   # [E=128, 16_scale_P, n_H512=4, GU=384] uint8
    down_w,           # [E=128, I=192,  H=2048]       int8
    down_scales,      # [E=128, H=2048]               fp32
):
    """
    Fused RMSNorm + Router + TopK(K=8) + Expert MLP for Qwen3 MoE TKG.
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
    H_free_shard = _H_FREE_SHARD
    H_shard = _H_SHARD
    I_tiles = _I_TILES
    n_H512 = _N_H512   # = 4
    T_pad = _T_PAD     # = 4

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
    # Stage 2: Router matmul
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
    # MXFP8 Activation Quantization (once, shared across all experts)
    #
    # rmsnorm_normed_bf16: [128_P, H_free*T=16] bf16
    #   H_free=16, T=1: column h = H_free tile h, for token t=0
    #   Mapping: H512 tile h512 covers H_free tiles h512*4..(h512+1)*4-1
    #
    # Layout adapter for quantize_mx:
    #   Need [128_P, n_H512*T_pad*_Q_WIDTH=64] bf16 (x4 packs _Q_WIDTH=4 H values per P row)
    #   Allocate [128, n_H512=4, T_pad=4, _Q_WIDTH=4] bf16, zero it
    #   Copy real data: shfl[p, h512, 0, q] = rmsnorm_normed_bf16[p, h512*4+q]  (t=0 only)
    #   Then flatten to [128, 64] and quantize_mx
    #   Result: inp_qtz [128, n_H512*T_pad=16] float8_e4m3fn_x4
    #           inp_scale [128, n_H512*T_pad=16] uint8 (sparse quadrant layout)
    # -----------------------------------------------------------------------

    # Layout adapter: rmsnorm_normed_bf16 [128_P, H_free*T=16] → [128_P, 64] for quantize_mx
    # We need [128_P, n_H512=4, T_pad=4, _Q_WIDTH=4] = [128, 64] flat
    # Mapping: act_shfl[p, h512, 0, q] = rmsnorm_normed_bf16[p, h512*_Q_WIDTH + q]  (T=1 only)
    # rmsnorm_normed_bf16[p, col] where col = h512*4 + q (the H_free tile)
    # Use nc_transpose trick: reshape [128, 16] as needed and copy column-by-column
    # Simple approach: allocate [128, n_H512*T_pad*_Q_WIDTH=64] zeros, copy 16 real values
    act_flat = nl.ndarray((_PMAX, n_H512 * T_pad * _Q_WIDTH), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.memset(act_flat, value=0.0)
    # For each H512 tile h512, copy 4 columns (q=0..3) into t=0 slot
    # act_flat[p, h512*T_pad*_Q_WIDTH + 0*_Q_WIDTH + q] = rmsnorm_normed_bf16[p, h512*_Q_WIDTH + q]
    # Positions in act_flat for h512, t=0: h512*16 + 0*4 + 0..3 = h512*16 + 0..3
    # Positions in rmsnorm_normed_bf16: h512*4 + 0..3
    # So we copy a contiguous block of 4 columns from rmsnorm → act_flat at offset h512*16
    for h512 in nl.affine_range(n_H512):
        nisa.tensor_copy(
            dst=act_flat[0:_PMAX, nl.ds(h512 * T_pad * _Q_WIDTH, _Q_WIDTH)],
            src=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h512 * _Q_WIDTH, _Q_WIDTH)],
        )

    # Allocate quantized outputs: [128, n_H512*T_pad=16] x4
    inp_qtz_flat = nl.ndarray((_PMAX, n_H512 * T_pad), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    inp_scale_flat = nl.ndarray((_PMAX, n_H512 * T_pad), dtype=nl.uint8, buffer=nl.sbuf)

    nisa.quantize_mx(
        src=act_flat[0:_PMAX, 0:n_H512 * T_pad * _Q_WIDTH],
        dst=inp_qtz_flat[0:_PMAX, 0:n_H512 * T_pad],
        dst_scale=inp_scale_flat[0:_PMAX, 0:n_H512 * T_pad],
    )

    # Reshape to [128, n_H512=4, T_pad=4] for per-tile indexing in nc_matmul_mx
    inp_qtz_sb = inp_qtz_flat.reshape((_PMAX, n_H512, T_pad))
    inp_scale_sb = inp_scale_flat.reshape((_PMAX, n_H512, T_pad))

    # -----------------------------------------------------------------------
    # Stage 4: Selective-Expert MLP — 2-Wave Expert Processing
    # -----------------------------------------------------------------------
    output_temp = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.affine_range(T):

        # ---------------------------------------------------------------
        # Allocate gate_up weight SBUF buffers (new MXFP8 layout) — Wave 0
        # [128_P, n_H512=4, GU=384] float8_e4m3fn_x4 per expert (x4 packs H direction)
        # ---------------------------------------------------------------
        gu_w_buf0 = nl.ndarray((_PMAX, n_H512, _GU_FLAT_W), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        gu_w_buf1 = nl.ndarray((_PMAX, n_H512, _GU_FLAT_W), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        gu_w_buf2 = nl.ndarray((_PMAX, n_H512, _GU_FLAT_W), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        gu_w_buf3 = nl.ndarray((_PMAX, n_H512, _GU_FLAT_W), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        gu_w_bufs = [gu_w_buf0, gu_w_buf1, gu_w_buf2, gu_w_buf3]

        # Scale: sparse SBUF layout [128_P, n_H512=4, GU=384] uint8
        # x4 packing is in H (P) direction; GU index is UNPACKED (384 not 96)
        gu_scale_buf0 = nl.ndarray((_PMAX, n_H512, _GU_FLAT), dtype=nl.uint8, buffer=nl.sbuf)
        gu_scale_buf1 = nl.ndarray((_PMAX, n_H512, _GU_FLAT), dtype=nl.uint8, buffer=nl.sbuf)
        gu_scale_buf2 = nl.ndarray((_PMAX, n_H512, _GU_FLAT), dtype=nl.uint8, buffer=nl.sbuf)
        gu_scale_buf3 = nl.ndarray((_PMAX, n_H512, _GU_FLAT), dtype=nl.uint8, buffer=nl.sbuf)
        gu_scale_bufs = [gu_scale_buf0, gu_scale_buf1, gu_scale_buf2, gu_scale_buf3]

        # Down weights
        down_full0_fp8_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full0_fp8_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full0_fp8_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full0_fp8_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full0_fp8_bufs = [down_full0_fp8_buf0, down_full0_fp8_buf1, down_full0_fp8_buf2, down_full0_fp8_buf3]

        down_full1_fp8_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full1_fp8_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full1_fp8_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full1_fp8_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full1_fp8_bufs = [down_full1_fp8_buf0, down_full1_fp8_buf1, down_full1_fp8_buf2, down_full1_fp8_buf3]

        down_scale_buf0 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_buf1 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_buf2 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_buf3 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_bufs = [down_scale_buf0, down_scale_buf1, down_scale_buf2, down_scale_buf3]

        # Wave 1 buffers
        gu_w_w1_buf0 = nl.ndarray((_PMAX, n_H512, _GU_FLAT_W), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        gu_w_w1_buf1 = nl.ndarray((_PMAX, n_H512, _GU_FLAT_W), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        gu_w_w1_buf2 = nl.ndarray((_PMAX, n_H512, _GU_FLAT_W), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        gu_w_w1_buf3 = nl.ndarray((_PMAX, n_H512, _GU_FLAT_W), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        gu_w_w1_bufs = [gu_w_w1_buf0, gu_w_w1_buf1, gu_w_w1_buf2, gu_w_w1_buf3]

        gu_scale_w1_buf0 = nl.ndarray((_PMAX, n_H512, _GU_FLAT), dtype=nl.uint8, buffer=nl.sbuf)
        gu_scale_w1_buf1 = nl.ndarray((_PMAX, n_H512, _GU_FLAT), dtype=nl.uint8, buffer=nl.sbuf)
        gu_scale_w1_buf2 = nl.ndarray((_PMAX, n_H512, _GU_FLAT), dtype=nl.uint8, buffer=nl.sbuf)
        gu_scale_w1_buf3 = nl.ndarray((_PMAX, n_H512, _GU_FLAT), dtype=nl.uint8, buffer=nl.sbuf)
        gu_scale_w1_bufs = [gu_scale_w1_buf0, gu_scale_w1_buf1, gu_scale_w1_buf2, gu_scale_w1_buf3]

        down_full0_fp8_w1_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full0_fp8_w1_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full0_fp8_w1_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full0_fp8_w1_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full0_fp8_w1_bufs = [down_full0_fp8_w1_buf0, down_full0_fp8_w1_buf1, down_full0_fp8_w1_buf2, down_full0_fp8_w1_buf3]

        down_full1_fp8_w1_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full1_fp8_w1_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full1_fp8_w1_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full1_fp8_w1_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.int8, buffer=nl.sbuf)
        down_full1_fp8_w1_bufs = [down_full1_fp8_w1_buf0, down_full1_fp8_w1_buf1, down_full1_fp8_w1_buf2, down_full1_fp8_w1_buf3]

        down_scale_w1_buf0 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_w1_buf1 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_w1_buf2 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_w1_buf3 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_w1_bufs = [down_scale_w1_buf0, down_scale_w1_buf1, down_scale_w1_buf2, down_scale_w1_buf3]

        # Expert ID buffers
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

        zero_bias = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(zero_bias, value=0.0)

        # Zero-pad down_full1 partial rows
        for k_pad in nl.affine_range(4):
            nisa.memset(down_full1_fp8_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)
        for k_pad in nl.affine_range(4):
            nisa.memset(down_full1_fp8_w1_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        # Pre-read expert IDs
        for k8 in nl.affine_range(8):
            nisa.dma_copy(
                dst=eid_all_bufs[k8][0:1, 0:1],
                src=top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k8),
            )

        # ---------------------------------------------------------------
        # Phase 1a: Load Wave 0 expert weights (gate_up + down)
        # gate_up_w: [E=128, 128_H_P, n_H512=4, GU=384] int8
        # For indirect DGE (dge_mode=0): scalar_offset selects expert (dim 0)
        # DMA pattern reads [128_P, n_H512=4, GU=384] for one expert
        # ---------------------------------------------------------------
        for k in nl.affine_range(_K_WAVE):
            expert_id = eid_all_bufs[k].ap(pattern=[[1, 1], [1, 1]], offset=0)

            # gate_up weights: [E, 128_P, n_H512, GU_FLAT=384] float8_e4m3fn_x4
            # Stride order (innermost first): GU_FLAT, n_H512, 128_P, E
            # expert stride = 128 * n_H512 * GU_FLAT = 128 * 4 * 384 = 196608
            nisa.dma_copy(
                dst=gu_w_bufs[k][0:_PMAX, 0:n_H512, 0:_GU_FLAT_W],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT_W, _PMAX], [_PMAX * _GU_FLAT_W, n_H512], [1, _GU_FLAT_W]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3fn_x4,
                ),
                dge_mode=0,
            )

            # gate_up scales: [E, 16_scale_P, n_H512, GU=384] uint8
            # Sparse quadrant load: for each quadrant, copy 4 HBM rows → SBUF at quadrant offset
            # GU last dim = 384 (unpacked; x4 is in H/P direction not GU direction)
            for i_quad in nl.affine_range(_N_QUADS):
                # HBM offset: i_quad*4 rows of scale_P dimension
                # Each row of scale_P covers n_H512 * GU = 4 * 384 = 1536 elements
                nisa.dma_copy(
                    dst=gu_scale_bufs[k][nl.ds(i_quad * _QUAD_SIZE, _SCALE_PER_QUAD), 0:n_H512, 0:_GU_FLAT],
                    src=gate_up_scales.ap(
                        pattern=[[_GU_FLAT, _SCALE_PER_QUAD], [_SCALE_P * _GU_FLAT, n_H512], [1, _GU_FLAT]],
                        offset=i_quad * _SCALE_PER_QUAD * n_H512 * _GU_FLAT,
                        scalar_offset=expert_id,
                        indirect_dim=0,
                        dtype=nl.uint8,
                    ),
                    dge_mode=0,
                )

            # Down weights (unchanged from v14a)
            nisa.dma_copy(
                dst=down_full0_fp8_bufs[k],
                src=down_w.ap(
                    pattern=[[H, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.int8,
                ),
                dge_mode=0,
            )
            nisa.dma_copy(
                dst=down_full1_fp8_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H_shard]],
                    offset=I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.int8,
                ),
                dge_mode=0,
            )
            nisa.dma_copy(
                dst=down_scale_bufs[k],
                src=down_scales.ap(
                    pattern=[[1, _PMAX], [_PMAX, _H_SHARD_BLOCKS]],
                    offset=prg_id * _H_SHARD,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # Phase 1b: Load Wave 1 expert weights
        for k in nl.affine_range(_K_WAVE):
            kk = k + 4
            expert_id_w1 = eid_all_bufs[kk].ap(pattern=[[1, 1], [1, 1]], offset=0)

            nisa.dma_copy(
                dst=gu_w_w1_bufs[k][0:_PMAX, 0:n_H512, 0:_GU_FLAT_W],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT_W, _PMAX], [_PMAX * _GU_FLAT_W, n_H512], [1, _GU_FLAT_W]],
                    offset=0,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3fn_x4,
                ),
                dge_mode=0,
            )

            for i_quad in nl.affine_range(_N_QUADS):
                nisa.dma_copy(
                    dst=gu_scale_w1_bufs[k][nl.ds(i_quad * _QUAD_SIZE, _SCALE_PER_QUAD), 0:n_H512, 0:_GU_FLAT],
                    src=gate_up_scales.ap(
                        pattern=[[_GU_FLAT, _SCALE_PER_QUAD], [_SCALE_P * _GU_FLAT, n_H512], [1, _GU_FLAT]],
                        offset=i_quad * _SCALE_PER_QUAD * n_H512 * _GU_FLAT,
                        scalar_offset=expert_id_w1,
                        indirect_dim=0,
                        dtype=nl.uint8,
                    ),
                    dge_mode=0,
                )

            nisa.dma_copy(
                dst=down_full0_fp8_w1_bufs[k],
                src=down_w.ap(
                    pattern=[[H, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                    dtype=nl.int8,
                ),
                dge_mode=0,
            )
            nisa.dma_copy(
                dst=down_full1_fp8_w1_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H_shard]],
                    offset=I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                    dtype=nl.int8,
                ),
                dge_mode=0,
            )
            nisa.dma_copy(
                dst=down_scale_w1_bufs[k],
                src=down_scales.ap(
                    pattern=[[1, _PMAX], [_PMAX, _H_SHARD_BLOCKS]],
                    offset=prg_id * _H_SHARD,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # Norm weights
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

        # PSUM for down (wave 0)
        down_psum = nl.ndarray((_PMAX, _K_WAVE * H_free_shard), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(down_psum, value=0.0)
        nisa.memset(output_temp, value=0.0)

        # ---------------------------------------------------------------
        # Phase 2a: Experts 0-3 — nc_matmul_mx gate_up
        #
        # Weight: [128_P, n_H512=4, 96_GU_x4] x4 (x4 groups GU neurons)
        # nc_matmul_mx: stationary=[128, GU_x4_chunk], moving=[128, T_pad]
        #   dst PSUM: [GU_x4_chunk, 4, T_pad] in bf16
        #
        # We split GU into I_tiles=2 x-tiles:
        #   tile 0: gate GU_x4 0..31 (I0=128 neurons), up GU_x4 48..79
        #   tile 1: gate GU_x4 32..47 (I1=64 neurons), up GU_x4 80..95
        #
        # For eviction: gate_psum [32, 4, T_pad] has P=32, free=4*T_pad=16
        # AP constraint: p_step must equal free_dim = 16 ✓ (we'll use 16)
        # ---------------------------------------------------------------

        # Unpacked neuron offsets for gate and up in combined GU=384 space
        # gate neurons: 0..191 (I=192), up neurons: 192..383 (I=192)
        # _UP_START = 192 (unpacked up neuron start in GU dimension)
        # For I0=128 neurons per tile: 4 q_idx × 32 neurons = 128
        # For I1=64 neurons: 2 q_idx × 32 neurons = 64 (32-aligned offsets for TensorCopy)
        _UP_START = _GU_FLAT // 2    # = 192 (unpacked offset where up neurons begin)
        _GATE_I1_START = I0          # = 128 (unpacked offset where gate I1 tile begins)
        _IG0_CHUNK = I0 // _Q_WIDTH  # = 32  (neurons per q_idx for I0 tile; 4 q_idx × 32 = 128)
        _IG1_CHUNK = 32              # = 32  (neurons per q_idx for I1 tile; 2 iters × 32 = 64 = I1)
        _I1_Q_ITERS = I1 // _IG1_CHUNK  # = 2  (only 2 q_idx for I1, not Q_WIDTH=4)

        for k in nl.affine_range(_K_WAVE):
            d_base = k * H_free_shard

            # Buffer is already float8_e4m3fn_x4 (no .view() needed)
            # gu_w_bufs[k]: [128_P, n_H512=4, 384_GU] float8_e4m3fn_x4

            # Pre-compute fp8fn views of down weight buffers (view() requires full tensor)
            down_full0_fp8fn = down_full0_fp8_bufs[k].view(nl.float8_e4m3fn)
            down_full1_fp8fn = down_full1_fp8_bufs[k].view(nl.float8_e4m3fn)

            # nc_matmul_mx PSUM: [_PMAX, _Q_WIDTH, T_pad] = [128, 4, 4]
            # I0 tile: q_idx in 0..3, each call writes _IG0_CHUNK=32 neurons (unpacked offset)
            # Total: 4 × 32 = 128 = I0 ✓
            # I1 tile: 2 q_idx iterations of 32 neurons → 64 = I1 ✓
            # Using 2 iters so gather offsets {0, 32} are 32-aligned (valid for TensorCopy)
            gate0_psum = nl.ndarray((_PMAX, _Q_WIDTH, T_pad), dtype=nl.bfloat16, buffer=nl.psum)
            up0_psum   = nl.ndarray((_PMAX, _Q_WIDTH, T_pad), dtype=nl.bfloat16, buffer=nl.psum)
            gate1_psum = nl.ndarray((_PMAX, _I1_Q_ITERS, T_pad), dtype=nl.bfloat16, buffer=nl.psum)
            up1_psum   = nl.ndarray((_PMAX, _I1_Q_ITERS, T_pad), dtype=nl.bfloat16, buffer=nl.psum)

            for i_h512 in nl.sequential_range(n_H512):
                for q_idx in nl.sequential_range(_Q_WIDTH):
                    # Gate tile 0: unpacked neurons q_idx*32 .. q_idx*32+31 (gate: 0..127)
                    nisa.nc_matmul_mx(
                        dst=gate0_psum[0:_IG0_CHUNK, q_idx, 0:T_pad],
                        stationary=gu_w_bufs[k][0:_PMAX, i_h512, nl.ds(q_idx * _IG0_CHUNK, _IG0_CHUNK)],
                        moving=inp_qtz_sb[0:_PMAX, i_h512, 0:T_pad],
                        stationary_scale=gu_scale_bufs[k][0:_PMAX, i_h512, nl.ds(q_idx * _IG0_CHUNK, _IG0_CHUNK)],
                        moving_scale=inp_scale_sb[0:_PMAX, i_h512, 0:T_pad],
                    )
                    # Up tile 0: up neurons start at _UP_START=192, offset 192 + q_idx*32 (unpacked)
                    nisa.nc_matmul_mx(
                        dst=up0_psum[0:_IG0_CHUNK, q_idx, 0:T_pad],
                        stationary=gu_w_bufs[k][0:_PMAX, i_h512, nl.ds(_UP_START + q_idx * _IG0_CHUNK, _IG0_CHUNK)],
                        moving=inp_qtz_sb[0:_PMAX, i_h512, 0:T_pad],
                        stationary_scale=gu_scale_bufs[k][0:_PMAX, i_h512, nl.ds(_UP_START + q_idx * _IG0_CHUNK, _IG0_CHUNK)],
                        moving_scale=inp_scale_sb[0:_PMAX, i_h512, 0:T_pad],
                    )
                for q_idx in nl.sequential_range(_I1_Q_ITERS):
                    # Gate tile 1: gate neurons 128..191 at _GATE_I1_START=128 + q_idx*32 (unpacked)
                    nisa.nc_matmul_mx(
                        dst=gate1_psum[0:_IG1_CHUNK, q_idx, 0:T_pad],
                        stationary=gu_w_bufs[k][0:_PMAX, i_h512, nl.ds(_GATE_I1_START + q_idx * _IG1_CHUNK, _IG1_CHUNK)],
                        moving=inp_qtz_sb[0:_PMAX, i_h512, 0:T_pad],
                        stationary_scale=gu_scale_bufs[k][0:_PMAX, i_h512, nl.ds(_GATE_I1_START + q_idx * _IG1_CHUNK, _IG1_CHUNK)],
                        moving_scale=inp_scale_sb[0:_PMAX, i_h512, 0:T_pad],
                    )
                    # Up tile 1: up neurons 128..191 at _UP_START + _GATE_I1_START + q_idx*32 = 320+q_idx*32 (unpacked)
                    nisa.nc_matmul_mx(
                        dst=up1_psum[0:_IG1_CHUNK, q_idx, 0:T_pad],
                        stationary=gu_w_bufs[k][0:_PMAX, i_h512, nl.ds(_UP_START + _GATE_I1_START + q_idx * _IG1_CHUNK, _IG1_CHUNK)],
                        moving=inp_qtz_sb[0:_PMAX, i_h512, 0:T_pad],
                        stationary_scale=gu_scale_bufs[k][0:_PMAX, i_h512, nl.ds(_UP_START + _GATE_I1_START + q_idx * _IG1_CHUNK, _IG1_CHUNK)],
                        moving_scale=inp_scale_sb[0:_PMAX, i_h512, 0:T_pad],
                    )

            # Evict PSUM → SBUF using nkilib strided AP:
            # I0: psum [_PMAX, Q_WIDTH=4, T_pad], AP: [[Q_WIDTH*T_pad, chunk], [1, T_pad], [T_pad, Q_WIDTH]]
            # I1: psum [_PMAX, _I1_Q_ITERS=2, T_pad], AP: [[_I1_Q_ITERS*T_pad, chunk], [1, T_pad], [T_pad, _I1_Q_ITERS]]
            gate0_sb = nl.ndarray((_IG0_CHUNK, T_pad, _Q_WIDTH), dtype=nl.bfloat16, buffer=nl.sbuf)
            up0_sb   = nl.ndarray((_IG0_CHUNK, T_pad, _Q_WIDTH), dtype=nl.bfloat16, buffer=nl.sbuf)
            gate1_sb = nl.ndarray((_IG1_CHUNK, T_pad, _I1_Q_ITERS), dtype=nl.bfloat16, buffer=nl.sbuf)
            up1_sb   = nl.ndarray((_IG1_CHUNK, T_pad, _I1_Q_ITERS), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=gate0_sb[0:_IG0_CHUNK, 0:T_pad, 0:_Q_WIDTH],
                src=gate0_psum.ap([[_Q_WIDTH * T_pad, _IG0_CHUNK], [1, T_pad], [T_pad, _Q_WIDTH]]),
            )
            nisa.tensor_copy(
                dst=up0_sb[0:_IG0_CHUNK, 0:T_pad, 0:_Q_WIDTH],
                src=up0_psum.ap([[_Q_WIDTH * T_pad, _IG0_CHUNK], [1, T_pad], [T_pad, _Q_WIDTH]]),
            )
            nisa.tensor_copy(
                dst=gate1_sb[0:_IG1_CHUNK, 0:T_pad, 0:_I1_Q_ITERS],
                src=gate1_psum.ap([[_I1_Q_ITERS * T_pad, _IG1_CHUNK], [1, T_pad], [T_pad, _I1_Q_ITERS]]),
            )
            nisa.tensor_copy(
                dst=up1_sb[0:_IG1_CHUNK, 0:T_pad, 0:_I1_Q_ITERS],
                src=up1_psum.ap([[_I1_Q_ITERS * T_pad, _IG1_CHUNK], [1, T_pad], [T_pad, _I1_Q_ITERS]]),
            )

            # Gather token 0 into [_PMAX, 1] for down matmul.
            # gate0_sb[g, t, q] = gate neuron q*_IG0_CHUNK+g, token t
            # _IG0_CHUNK=32: 4 copies of 32 at offsets 0, 32, 64, 96 (all 32-aligned) ✓
            gate_t0 = nl.ndarray((_PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            up_t0   = nl.ndarray((_PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            # q_idx=0: neurons 0..31
            nisa.tensor_copy(dst=gate_t0[0:_IG0_CHUNK, 0:1], src=gate0_sb[0:_IG0_CHUNK, 0:1, 0:1])
            nisa.tensor_copy(dst=up_t0[0:_IG0_CHUNK, 0:1],   src=up0_sb[0:_IG0_CHUNK, 0:1, 0:1])
            # q_idx=1: neurons 32..63
            nisa.tensor_copy(dst=gate_t0[nl.ds(1 * _IG0_CHUNK, _IG0_CHUNK), 0:1], src=gate0_sb[0:_IG0_CHUNK, 0:1, 1:2])
            nisa.tensor_copy(dst=up_t0[nl.ds(1 * _IG0_CHUNK, _IG0_CHUNK), 0:1],   src=up0_sb[0:_IG0_CHUNK, 0:1, 1:2])
            # q_idx=2: neurons 64..95
            nisa.tensor_copy(dst=gate_t0[nl.ds(2 * _IG0_CHUNK, _IG0_CHUNK), 0:1], src=gate0_sb[0:_IG0_CHUNK, 0:1, 2:3])
            nisa.tensor_copy(dst=up_t0[nl.ds(2 * _IG0_CHUNK, _IG0_CHUNK), 0:1],   src=up0_sb[0:_IG0_CHUNK, 0:1, 2:3])
            # q_idx=3: neurons 96..127
            nisa.tensor_copy(dst=gate_t0[nl.ds(3 * _IG0_CHUNK, _IG0_CHUNK), 0:1], src=gate0_sb[0:_IG0_CHUNK, 0:1, 3:4])
            nisa.tensor_copy(dst=up_t0[nl.ds(3 * _IG0_CHUNK, _IG0_CHUNK), 0:1],   src=up0_sb[0:_IG0_CHUNK, 0:1, 3:4])

            # Gather tile 1 (I1=64 neurons): _IG1_CHUNK=32, 2 iters → offsets 0, 32 (32-aligned) ✓
            gate_t1 = nl.ndarray((_PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            up_t1   = nl.ndarray((_PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            # q_idx=0: neurons 0..31
            nisa.tensor_copy(dst=gate_t1[0:_IG1_CHUNK, 0:1], src=gate1_sb[0:_IG1_CHUNK, 0:1, 0:1])
            nisa.tensor_copy(dst=up_t1[0:_IG1_CHUNK, 0:1],   src=up1_sb[0:_IG1_CHUNK, 0:1, 0:1])
            # q_idx=1: neurons 32..63
            nisa.tensor_copy(dst=gate_t1[nl.ds(_IG1_CHUNK, _IG1_CHUNK), 0:1], src=gate1_sb[0:_IG1_CHUNK, 0:1, 1:2])
            nisa.tensor_copy(dst=up_t1[nl.ds(_IG1_CHUNK, _IG1_CHUNK), 0:1],   src=up1_sb[0:_IG1_CHUNK, 0:1, 1:2])

            # SiLU(gate) * up for tile 0
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            gate_silu0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_silu0[0:_PMAX, 0:1], op=nl.silu, data=gate_t0[0:_PMAX, 0:1])
            up_f32_0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(up_f32_0[0:_PMAX, 0:1], op=nl.copy, data=up_t0[0:_PMAX, 0:1])
            nisa.tensor_tensor(inter_f32[0:_PMAX, 0:1], gate_silu0[0:_PMAX, 0:1], up_f32_0[0:_PMAX, 0:1], nl.multiply)

            # SiLU(gate) * up for tile 1
            gate_silu1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_silu1[0:I1, 0:1], op=nl.silu, data=gate_t1[0:I1, 0:1])
            up_f32_1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(up_f32_1[0:I1, 0:1], op=nl.copy, data=up_t1[0:I1, 0:1])
            nisa.tensor_tensor(inter_f32[0:I1, 1:2], gate_silu1[0:I1, 0:1], up_f32_1[0:I1, 0:1], nl.multiply)

            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            # Down matmul: int8 (fp8 bytes) → pre-viewed as float8_e4m3fn → convert to bf16 → nc_matmul
            # This avoids mixing legacy fp8 (float8_e4m3) with OCP fp8 (float8_e4m3fn_x4).
            # Scale is applied to the PSUM output as before.
            for h1_out in nl.affine_range(H_free_shard):
                # I0 tile: [128_P, 128_F] fp8fn → bfloat16
                tile0_bf16 = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(tile0_bf16, op=nl.copy, data=down_full0_fp8fn[0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)])
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=tile0_bf16[0:_PMAX, 0:_PMAX],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                # I1 tile: [128_P, 128_F] fp8fn → bfloat16
                tile1_bf16 = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(tile1_bf16, op=nl.copy, data=down_full1_fp8fn[0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)])
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=tile1_bf16[0:_PMAX, 0:_PMAX],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            # Scale + accumulate output
            aff_col_buf = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=aff_col_buf, src=aff_bcast[0:_PMAX, k:k + 1])
            aff_col_view = TensorView(aff_col_buf).broadcast(dim=1, size=_H_FREE_SHARD)
            combined_down_scale = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(combined_down_scale, down_scale_bufs[k], aff_col_view.get_view(), nl.multiply)

            down_h_raw = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(down_h_raw, op=nl.copy, data=down_psum[0:_PMAX, d_base:d_base + _H_FREE_SHARD])

            down_h_all = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(down_h_all, down_h_raw, combined_down_scale, nl.multiply)

            nisa.tensor_tensor(
                dst=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                data1=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                data2=down_h_all[0:_PMAX, 0:_H_FREE_SHARD],
                op=nl.add,
            )

        # ---------------------------------------------------------------
        # Wave 1: Experts 4-7
        # ---------------------------------------------------------------
        nisa.memset(down_psum, value=0.0)

        for k in nl.affine_range(_K_WAVE):
            kk = k + 4
            d_base = k * H_free_shard

            # Buffer is already float8_e4m3fn_x4 (no .view() needed)
            down_full0_fp8fn_w1 = down_full0_fp8_w1_bufs[k].view(nl.float8_e4m3fn)
            down_full1_fp8fn_w1 = down_full1_fp8_w1_bufs[k].view(nl.float8_e4m3fn)

            gate0_psum = nl.ndarray((_PMAX, _Q_WIDTH, T_pad), dtype=nl.bfloat16, buffer=nl.psum)
            up0_psum   = nl.ndarray((_PMAX, _Q_WIDTH, T_pad), dtype=nl.bfloat16, buffer=nl.psum)
            gate1_psum = nl.ndarray((_PMAX, _I1_Q_ITERS, T_pad), dtype=nl.bfloat16, buffer=nl.psum)
            up1_psum   = nl.ndarray((_PMAX, _I1_Q_ITERS, T_pad), dtype=nl.bfloat16, buffer=nl.psum)

            for i_h512 in nl.sequential_range(n_H512):
                for q_idx in nl.sequential_range(_Q_WIDTH):
                    # Gate tile 0: unpacked neurons q_idx*32 .. q_idx*32+31 (gate: 0..127)
                    nisa.nc_matmul_mx(
                        dst=gate0_psum[0:_IG0_CHUNK, q_idx, 0:T_pad],
                        stationary=gu_w_w1_bufs[k][0:_PMAX, i_h512, nl.ds(q_idx * _IG0_CHUNK, _IG0_CHUNK)],
                        moving=inp_qtz_sb[0:_PMAX, i_h512, 0:T_pad],
                        stationary_scale=gu_scale_w1_bufs[k][0:_PMAX, i_h512, nl.ds(q_idx * _IG0_CHUNK, _IG0_CHUNK)],
                        moving_scale=inp_scale_sb[0:_PMAX, i_h512, 0:T_pad],
                    )
                    # Up tile 0: up neurons at _UP_START=192, offset 192 + q_idx*32 (unpacked)
                    nisa.nc_matmul_mx(
                        dst=up0_psum[0:_IG0_CHUNK, q_idx, 0:T_pad],
                        stationary=gu_w_w1_bufs[k][0:_PMAX, i_h512, nl.ds(_UP_START + q_idx * _IG0_CHUNK, _IG0_CHUNK)],
                        moving=inp_qtz_sb[0:_PMAX, i_h512, 0:T_pad],
                        stationary_scale=gu_scale_w1_bufs[k][0:_PMAX, i_h512, nl.ds(_UP_START + q_idx * _IG0_CHUNK, _IG0_CHUNK)],
                        moving_scale=inp_scale_sb[0:_PMAX, i_h512, 0:T_pad],
                    )
                for q_idx in nl.sequential_range(_I1_Q_ITERS):
                    # Gate tile 1: gate neurons 128..191 at _GATE_I1_START=128 + q_idx*32 (unpacked)
                    nisa.nc_matmul_mx(
                        dst=gate1_psum[0:_IG1_CHUNK, q_idx, 0:T_pad],
                        stationary=gu_w_w1_bufs[k][0:_PMAX, i_h512, nl.ds(_GATE_I1_START + q_idx * _IG1_CHUNK, _IG1_CHUNK)],
                        moving=inp_qtz_sb[0:_PMAX, i_h512, 0:T_pad],
                        stationary_scale=gu_scale_w1_bufs[k][0:_PMAX, i_h512, nl.ds(_GATE_I1_START + q_idx * _IG1_CHUNK, _IG1_CHUNK)],
                        moving_scale=inp_scale_sb[0:_PMAX, i_h512, 0:T_pad],
                    )
                    # Up tile 1: up neurons 128..191 at _UP_START + _GATE_I1_START + q_idx*32 = 320+q_idx*32 (unpacked)
                    nisa.nc_matmul_mx(
                        dst=up1_psum[0:_IG1_CHUNK, q_idx, 0:T_pad],
                        stationary=gu_w_w1_bufs[k][0:_PMAX, i_h512, nl.ds(_UP_START + _GATE_I1_START + q_idx * _IG1_CHUNK, _IG1_CHUNK)],
                        moving=inp_qtz_sb[0:_PMAX, i_h512, 0:T_pad],
                        stationary_scale=gu_scale_w1_bufs[k][0:_PMAX, i_h512, nl.ds(_UP_START + _GATE_I1_START + q_idx * _IG1_CHUNK, _IG1_CHUNK)],
                        moving_scale=inp_scale_sb[0:_PMAX, i_h512, 0:T_pad],
                    )

            gate0_sb = nl.ndarray((_IG0_CHUNK, T_pad, _Q_WIDTH), dtype=nl.bfloat16, buffer=nl.sbuf)
            up0_sb   = nl.ndarray((_IG0_CHUNK, T_pad, _Q_WIDTH), dtype=nl.bfloat16, buffer=nl.sbuf)
            gate1_sb = nl.ndarray((_IG1_CHUNK, T_pad, _I1_Q_ITERS), dtype=nl.bfloat16, buffer=nl.sbuf)
            up1_sb   = nl.ndarray((_IG1_CHUNK, T_pad, _I1_Q_ITERS), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=gate0_sb[0:_IG0_CHUNK, 0:T_pad, 0:_Q_WIDTH],
                src=gate0_psum.ap([[_Q_WIDTH * T_pad, _IG0_CHUNK], [1, T_pad], [T_pad, _Q_WIDTH]]),
            )
            nisa.tensor_copy(
                dst=up0_sb[0:_IG0_CHUNK, 0:T_pad, 0:_Q_WIDTH],
                src=up0_psum.ap([[_Q_WIDTH * T_pad, _IG0_CHUNK], [1, T_pad], [T_pad, _Q_WIDTH]]),
            )
            nisa.tensor_copy(
                dst=gate1_sb[0:_IG1_CHUNK, 0:T_pad, 0:_I1_Q_ITERS],
                src=gate1_psum.ap([[_I1_Q_ITERS * T_pad, _IG1_CHUNK], [1, T_pad], [T_pad, _I1_Q_ITERS]]),
            )
            nisa.tensor_copy(
                dst=up1_sb[0:_IG1_CHUNK, 0:T_pad, 0:_I1_Q_ITERS],
                src=up1_psum.ap([[_I1_Q_ITERS * T_pad, _IG1_CHUNK], [1, T_pad], [T_pad, _I1_Q_ITERS]]),
            )

            gate_t0 = nl.ndarray((_PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            up_t0   = nl.ndarray((_PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gate_t0[0:_IG0_CHUNK, 0:1], src=gate0_sb[0:_IG0_CHUNK, 0:1, 0:1])
            nisa.tensor_copy(dst=up_t0[0:_IG0_CHUNK, 0:1],   src=up0_sb[0:_IG0_CHUNK, 0:1, 0:1])
            nisa.tensor_copy(dst=gate_t0[nl.ds(1 * _IG0_CHUNK, _IG0_CHUNK), 0:1], src=gate0_sb[0:_IG0_CHUNK, 0:1, 1:2])
            nisa.tensor_copy(dst=up_t0[nl.ds(1 * _IG0_CHUNK, _IG0_CHUNK), 0:1],   src=up0_sb[0:_IG0_CHUNK, 0:1, 1:2])
            nisa.tensor_copy(dst=gate_t0[nl.ds(2 * _IG0_CHUNK, _IG0_CHUNK), 0:1], src=gate0_sb[0:_IG0_CHUNK, 0:1, 2:3])
            nisa.tensor_copy(dst=up_t0[nl.ds(2 * _IG0_CHUNK, _IG0_CHUNK), 0:1],   src=up0_sb[0:_IG0_CHUNK, 0:1, 2:3])
            nisa.tensor_copy(dst=gate_t0[nl.ds(3 * _IG0_CHUNK, _IG0_CHUNK), 0:1], src=gate0_sb[0:_IG0_CHUNK, 0:1, 3:4])
            nisa.tensor_copy(dst=up_t0[nl.ds(3 * _IG0_CHUNK, _IG0_CHUNK), 0:1],   src=up0_sb[0:_IG0_CHUNK, 0:1, 3:4])

            gate_t1 = nl.ndarray((_PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            up_t1   = nl.ndarray((_PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gate_t1[0:_IG1_CHUNK, 0:1], src=gate1_sb[0:_IG1_CHUNK, 0:1, 0:1])
            nisa.tensor_copy(dst=up_t1[0:_IG1_CHUNK, 0:1],   src=up1_sb[0:_IG1_CHUNK, 0:1, 0:1])
            nisa.tensor_copy(dst=gate_t1[nl.ds(_IG1_CHUNK, _IG1_CHUNK), 0:1], src=gate1_sb[0:_IG1_CHUNK, 0:1, 1:2])
            nisa.tensor_copy(dst=up_t1[nl.ds(_IG1_CHUNK, _IG1_CHUNK), 0:1],   src=up1_sb[0:_IG1_CHUNK, 0:1, 1:2])

            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            gate_silu0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_silu0[0:_PMAX, 0:1], op=nl.silu, data=gate_t0[0:_PMAX, 0:1])
            up_f32_0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(up_f32_0[0:_PMAX, 0:1], op=nl.copy, data=up_t0[0:_PMAX, 0:1])
            nisa.tensor_tensor(inter_f32[0:_PMAX, 0:1], gate_silu0[0:_PMAX, 0:1], up_f32_0[0:_PMAX, 0:1], nl.multiply)

            gate_silu1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_silu1[0:I1, 0:1], op=nl.silu, data=gate_t1[0:I1, 0:1])
            up_f32_1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(up_f32_1[0:I1, 0:1], op=nl.copy, data=up_t1[0:I1, 0:1])
            nisa.tensor_tensor(inter_f32[0:I1, 1:2], gate_silu1[0:I1, 0:1], up_f32_1[0:I1, 0:1], nl.multiply)

            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            for h1_out in nl.affine_range(H_free_shard):
                tile0_bf16 = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(tile0_bf16, op=nl.copy, data=down_full0_fp8fn_w1[0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)])
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=tile0_bf16[0:_PMAX, 0:_PMAX],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                tile1_bf16 = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(tile1_bf16, op=nl.copy, data=down_full1_fp8fn_w1[0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)])
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=tile1_bf16[0:_PMAX, 0:_PMAX],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            aff_col_buf = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=aff_col_buf, src=aff_bcast[0:_PMAX, kk:kk + 1])
            aff_col_view = TensorView(aff_col_buf).broadcast(dim=1, size=_H_FREE_SHARD)
            combined_down_scale = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(combined_down_scale, down_scale_w1_bufs[k], aff_col_view.get_view(), nl.multiply)

            down_h_raw = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(down_h_raw, op=nl.copy, data=down_psum[0:_PMAX, d_base:d_base + _H_FREE_SHARD])

            down_h_all = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(down_h_all, down_h_raw, combined_down_scale, nl.multiply)

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


def run(inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales):
    """Run v15d kernel.

    gate_up_w:     [E=128, 128_H_P, n_H512=4, GU=384]     float8_e4m3fn_x4 (x4 packs H direction)
    gate_up_scales:[E=128, 16_scale_P, n_H512=4, GU=384]  uint8 (unpacked GU dim, matches weight)
    down_w:        [E=128, I=192, H=2048]  int8
    down_scales:   [E=128, H=2048]         fp32
    """
    import torch
    import torch_xla.core.xla_model as xm

    assert gate_up_w.shape == (_E, _PMAX, _N_H512, _GU_FLAT), f"gate_up_w shape {gate_up_w.shape}"
    assert gate_up_scales.shape == (_E, _SCALE_P, _N_H512, _GU_FLAT), f"gate_up_scales shape {gate_up_scales.shape}"
    assert down_w.shape == (_E, _I, _H), f"down_w shape {down_w.shape}"
    assert down_scales.shape == (_E, _H), f"down_scales shape {down_scales.shape}"

    xm.mark_step()

    outputs = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
