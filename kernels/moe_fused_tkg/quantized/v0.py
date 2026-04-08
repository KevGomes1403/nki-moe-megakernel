"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

kernel_v29_fp8 — FP8 expert weights with dequant before compute
=================================================================
Based on v28f (Plan F: ROUTER_BATCH=16, 2-wave 4-buffer structure).

Changes vs v28f (Plan A: FP8 dequant before compute):
  A1: gate_up_w and down_w changed from bf16 to fp8_e4m3fn.
  A2: gate_up_scales [E, 2*I=384, H/128=16] fp32 added.
  A3: down_scales [E, H=2048, I/128=2] fp32 added.
  A4: After loading each expert's fp8 weights, dequantize to bf16 in SBUF
      by loading corresponding scales and multiplying.
  A5: Downstream matmul and compute logic identical to v28f.

Interface contract:
  gate_up_w  [E=128, H=2048, 2*I=384] fp8_e4m3fn
  gate_up_scales [E=128, 2*I=384, H/128=16] fp32
  down_w     [E=128, I=192,  H=2048]   fp8_e4m3fn
  down_scales [E=128, H=2048, I/128=2]  fp32

SBUF Budget (per partition):
  4 x gate_up_fp8 [16 x 384 x 1] = 24 KiB  (halved from bf16)
  4 x gate_up_bf16 [16 x 384 x 2] = 48 KiB  (dequant output)
  4 x gate_up_scales [48 x 4] = 0.75 KiB
  4 x down_full0 [1024 x 2]  =  8 KiB
  4 x down_full1 [1024 x 2]  =  8 KiB
  4 x down_scales [16 x 4] = 0.25 KiB
  gate_t1 + up_t1             =  8 KiB
  misc                        = ~2 KiB
  Total: ~99 KiB << 224 KiB

PSUM budget: gate_up_psum [128, 16] + down_psum [128, 32] = 48 cols < 512 max.
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
_H_FREE_SHARD = _H_FREE // _N_PRGS   # = 8
_H_SHARD = _H_FREE_SHARD * _PMAX     # = 1024

# Router DMA batching: 16 tiles per DMA (Plan F: doubled from 8, 1 DMA call total)
_ROUTER_BATCH = 16

# 2-wave constants
_K_WAVE = 4  # experts per wave

# FP8 scale dimension constants
# gate_up_scales: [E, H=2048, GU_FLAT//128=3]
#   Quantize per-block-of-128 along GU_FLAT columns.
#   Scale constant across 128 GU columns, varies per H row (partition dim).
#   SBUF scale_buf[p, h1, j_blk] = scales[e, h1*128+p, j_blk]
_GU_J_BLOCKS = _GU_FLAT // _PMAX   # = 3
# down_scales: [E, I=192, H//128=16]
#   Quantize per-block-of-128 along H columns.
#   Scale constant across 128 H columns, varies per I row (partition dim).
#   SBUF scale_buf[p, h_blk] = scales[e, p, prg_id*8+h_blk] for tile0 (p=I index 0..127)
_H_BLOCKS = _H // _PMAX            # = 16
_H_SHARD_BLOCKS = _H_SHARD // _PMAX  # = 8


@nki.jit
def qwen3_moe_fused_tkg(
    inp,              # [B, 1, H=2048]              bf16
    gamma,            # [1, H=2048]                 bf16
    router_w,         # [H=2048, E=128]              bf16  (router stays bf16)
    gate_up_w,        # [E=128, H=2048, 2*I=384]     int8 (reinterpreted as fp8_e4m3fn)
    gate_up_scales,   # [E=128, H=2048, GU/128=3]    fp32  per-block-of-128 along GU cols
    down_w,           # [E=128, I=192,  H=2048]       int8 (reinterpreted as fp8_e4m3fn)
    down_scales,      # [E=128, I=192,  H/128=16]     fp32  per-block-of-128 along H cols
):
    """
    Fused RMSNorm + Router + TopK(K=8) + Expert MLP for Qwen3 MoE TKG.
    Invoked as: qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w,
                                         gate_up_scales, down_w, down_scales)
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
    H_free_shard = _H_FREE_SHARD
    H_shard = _H_SHARD
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
    for g in nl.static_range(4):
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
            dge_mode=3,  # Plan E1: hw_dge (offset is linear in h_chunk)
        )
        for h_sub in nl.static_range(_ROUTER_BATCH):
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
    # Stage 4: Selective-Expert MLP — Plan G: 2-Wave Expert Processing
    #
    # Wave 0: Load & compute experts 0-3 (buffer indices 0-3)
    # Wave 1: Re-load & compute experts 4-7 (reusing buffer indices 0-3)
    # -----------------------------------------------------------------------
    output_temp = nl.ndarray((_PMAX, H_free_shard, T), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.static_range(T):

        # ------------------------------------------------------------------
        # Allocate 4 named SBUF buffers for fp8 weights (halved size vs bf16)
        # ------------------------------------------------------------------
        # fp8 SBUF buffers; int8 HBM is DMA'd directly into fp8 SBUF (raw bit copy, 1B each)
        gate_up_fp8_buf0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        gate_up_fp8_buf1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        gate_up_fp8_buf2 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        gate_up_fp8_buf3 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        gate_up_fp8_bufs = [gate_up_fp8_buf0, gate_up_fp8_buf1, gate_up_fp8_buf2, gate_up_fp8_buf3]

        # bf16 dequantized gate_up weight buffers (same shape as v28f)
        gate_up_buf0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.bfloat16, buffer=nl.sbuf)
        gate_up_buf1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.bfloat16, buffer=nl.sbuf)
        gate_up_buf2 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.bfloat16, buffer=nl.sbuf)
        gate_up_buf3 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.bfloat16, buffer=nl.sbuf)
        gate_up_bufs = [gate_up_buf0, gate_up_buf1, gate_up_buf2, gate_up_buf3]

        # gate_up scale buffers per expert: [128P, H_free=16, J_BLOCKS=3] fp32
        # Layout: scale_buf[p, h1, j_blk] = gate_up_scales[e, h1*128+p, j_blk]
        # Scale is constant across 128 GU columns for each (h1, j_blk) block.
        gate_up_scale_buf0 = nl.ndarray((_PMAX, H_free, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_buf1 = nl.ndarray((_PMAX, H_free, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_buf2 = nl.ndarray((_PMAX, H_free, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_buf3 = nl.ndarray((_PMAX, H_free, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_bufs = [gate_up_scale_buf0, gate_up_scale_buf1, gate_up_scale_buf2, gate_up_scale_buf3]

        # fp8 down weight buffers; DMA from int8 HBM into fp8 SBUF (raw bit copy)
        down_full0_fp8_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        down_full0_fp8_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        down_full0_fp8_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        down_full0_fp8_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        down_full0_fp8_bufs = [down_full0_fp8_buf0, down_full0_fp8_buf1, down_full0_fp8_buf2, down_full0_fp8_buf3]

        down_full1_fp8_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        down_full1_fp8_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        down_full1_fp8_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        down_full1_fp8_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        down_full1_fp8_bufs = [down_full1_fp8_buf0, down_full1_fp8_buf1, down_full1_fp8_buf2, down_full1_fp8_buf3]

        # bf16 dequantized down weight buffers (same shape as v28f)
        down_full0_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down_full0_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down_full0_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down_full0_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down_full0_bufs = [down_full0_buf0, down_full0_buf1, down_full0_buf2, down_full0_buf3]

        down_full1_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down_full1_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down_full1_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down_full1_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down_full1_bufs = [down_full1_buf0, down_full1_buf1, down_full1_buf2, down_full1_buf3]

        # down scale buffers per expert:
        #   tile0: [128P_I, H_shard_blocks=8] fp32
        #   tile1: [128P_I, H_shard_blocks=8] fp32 (only rows 0..63 valid)
        # Layout: scale_buf0[p, h_blk] = down_scales[e, p, prg_id*8+h_blk]  (I rows 0..127)
        #         scale_buf1[p, h_blk] = down_scales[e, 128+p, prg_id*8+h_blk] (I rows 128..191)
        # Scale is constant across 128 H columns for each (p, h_blk) block.
        down_scale0_buf0 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale0_buf1 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale0_buf2 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale0_buf3 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale0_bufs = [down_scale0_buf0, down_scale0_buf1, down_scale0_buf2, down_scale0_buf3]
        down_scale1_buf0 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale1_buf1 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale1_buf2 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale1_buf3 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale1_bufs = [down_scale1_buf0, down_scale1_buf1, down_scale1_buf2, down_scale1_buf3]

        # ------------------------------------------------------------------
        # Zero pad region (rows I1:I0 = 64:128) for 4 down_full1 buffers
        # ------------------------------------------------------------------
        for k_pad in range(4):
            nisa.memset(down_full1_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        # gate_t1_128/up_t1_128: single pair of reused buffers
        gate_t1_128 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.bfloat16, buffer=nl.sbuf)
        up_t1_128   = nl.ndarray((_PMAX, H_free, I0), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(gate_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
        nisa.memset(up_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        # ==================================================================
        # WAVE 0: Experts 0-3
        # ==================================================================

        # eid_scratch: [128, 1] int32 SBUF buffer for expert IDs.
        # Using a [128, 1] buffer preserves IndirectDimMaxIndex = 127 = E-1,
        # which is required for correct DGE bounds checking.
        eid_scratch = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)

        # Phase 1a: Load experts 0-3 fp8 weights + scales (20 DMAs)
        for k in nl.static_range(_K_WAVE):
            nisa.dma_copy(
                dst=eid_scratch[0:1, 0:1],
                src=top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k),
            )
            expert_id = eid_scratch.ap(pattern=[[1, 1], [1, 1]], offset=0)

            # Load fp8 gate_up weights: int8 HBM → fp8_e4m3fn SBUF
            # dtype=nl.float8_e4m3fn on the source ap() causes the DMA to reinterpret
            # int8 bits as fp8_e4m3fn, enabling correct fp8→fp32 conversion during dequant.
            nisa.dma_copy(
                dst=gate_up_fp8_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3fn,
                ),
                dge_mode=0,
            )

            # Load gate_up scales [H=2048, J_BLOCKS=3] fp32
            # HBM layout: gate_up_scales[E, H=2048, J_BLOCKS=3]
            # scale_buf[p, h1, j_blk] = gate_up_scales[e, h1*128+p, j_blk]
            nisa.dma_copy(
                dst=gate_up_scale_bufs[k],
                src=gate_up_scales.ap(
                    pattern=[[_GU_J_BLOCKS, _PMAX], [_PMAX * _GU_J_BLOCKS, H_free], [1, _GU_J_BLOCKS]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # Load fp8 down_w tile 0 (rows 0:I0=128) for this shard
            nisa.dma_copy(
                dst=down_full0_fp8_bufs[k],
                src=down_w.ap(
                    pattern=[[H, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3fn,
                ),
                dge_mode=0,
            )

            # Load fp8 down_w tile 1 (rows I0:I0+I1=64) for this shard
            nisa.dma_copy(
                dst=down_full1_fp8_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H_shard]],
                    offset=I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3fn,
                ),
                dge_mode=0,
            )

            # Load down scales tile0 [I0=128, H_shard_blocks=8] fp32
            # HBM layout: down_scales[E, I=192, H_BLOCKS=16]
            # scale_buf0[p, h_blk] = down_scales[e, p, prg_id*8+h_blk]
            nisa.dma_copy(
                dst=down_scale0_bufs[k],
                src=down_scales.ap(
                    pattern=[[_H_BLOCKS, _PMAX], [1, _H_SHARD_BLOCKS]],
                    offset=prg_id * _H_SHARD_BLOCKS,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # Load down scales tile1 [I1=64, H_shard_blocks=8] fp32
            nisa.dma_copy(
                dst=down_scale1_bufs[k][0:I1, 0:_H_SHARD_BLOCKS],
                src=down_scales.ap(
                    pattern=[[_H_BLOCKS, I1], [1, _H_SHARD_BLOCKS]],
                    offset=I0 * _H_BLOCKS + prg_id * _H_SHARD_BLOCKS,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # ------------------------------------------------------------------
        # Change 1: Compute norm_weights HERE (after Phase 1a DMAs issued)
        # This overlaps VectorE compute with in-flight DMAs.
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

        # aff_bcast: broadcast ALL K=8 affinities (used by both waves)
        aff_bcast = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K], src=norm_weights[t:t + 1, 0:K])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        # ------------------------------------------------------------------
        # Dequantize Wave 0 weights (fp8 → bf16) after loads complete
        # ------------------------------------------------------------------
        # Dequant is: bf16[p, h1, j] = fp8[p, h1, j] * scale[j, h1]
        # scale[j, h1] is stored in scale_buf: partition p covers j-indices [p, p+128, p+256]
        # because the HBM layout [GU_FLAT=384, H_scale=16] is packed to [128, 48]
        # with row p = j_blk*128 + something? Let's be precise:
        # gate_up_scale_buf[p, h1 + j_blk * H_scale_blocks]
        #   = gate_up_scales[e, j_blk*128+p, h1]   (if loaded with pattern [[GU_FLAT*H_scale, 128P], ...])
        # Pattern used: [[GU_FLAT*H_scale, PMAX], [PMAX*H_scale, GU_FLAT//PMAX], [1, H_scale]]
        # This means: for partition p, for j_blk in [0..2], for h1 in [0..15]:
        #   gate_up_scale_buf[p, j_blk * H_scale + h1] = gate_up_scales_flat[p + j_blk*128*H_scale + h1*128... wait
        # Let me re-derive. pattern[[s0,n0],[s1,n1],[s2,n2]] at offset 0:
        #   element [i0, i1, i2] = flat[i0*s0 + i1*s1 + i2*s2]
        # dst shape [128, GU_SCALE_FREE=48]:
        #   [p, f] = flat[p * (GU_FLAT*H_scale) + (f//H_scale)*(PMAX*H_scale) + (f%H_scale)*1]
        #          = flat[p*384*16 + (f//16)*128*16 + (f%16)]
        #          = flat[(p + (f//16)*128)*16 + f%16]
        # If flat is [GU_FLAT, H_scale] row-major: flat[j, h] = j*H_scale + h
        #   so flat[(p + j_blk*128)*16 + h1] = gate_up_scales[e, j=p+j_blk*128, h1=h1]
        # Therefore: gate_up_scale_buf[p, j_blk*16 + h1] = gate_up_scales[e, j_blk*128+p, h1]
        # And: gate_up_buf[p, h1, j] = fp8[...], where j ranges over GU_FLAT=384
        #   j = j_blk*128 + j_local,   j_blk in {0,1,2},  j_local in [0,128)
        # Dequant: gate_up_buf[p, h1, j_blk*128+j_local] *= scale_buf[j_local, j_blk*16 + h1]
        # Note the partition mismatch: scale index is j_local (not p).
        # The scale doesn't depend on partition p — it's per (j, h1) not per (p).
        # We need to broadcast scale over partition dim.
        # Since scale_buf[p, j_blk*16+h1] = scales[e, j_blk*128+p, h1] (partition-varying!),
        # the scale DOES vary with partition p... but that's the WRONG semantics.
        # Correct semantics: scale[j, h1] applies to ALL partition rows for column j.
        # But gate_up_buf has partition=128, free=[H_free=16, GU_FLAT=384].
        # The weight column j corresponds to free-dim j (not partition).
        # So scale should not vary with partition!
        # This is a key insight: gate_up_scale_buf must have the same scale value in all 128 rows.
        # We need to load scales differently. Instead of the indirect-DMA pattern above,
        # we should load scales as [1, GU_FLAT * H_scale] and replicate over partition.
        # But SBUF partition must be 128... Alternatively, load per-j-block scale into a
        # broadcast buffer.
        #
        # REVISED APPROACH: for each h1 in [0..H_free) and j in GU_FLAT, the dequant scale
        # is a SCALAR (same for all 128 partition rows). So we need scalar dequant per column.
        # Use nisa.tensor_scalar with different scalars per column (h1, j).
        # But nisa.tensor_scalar only takes a single scalar...
        #
        # BEST APPROACH: Use nisa.tensor_tensor with scale broadcast over partition dim.
        # Load scales as [1, H_free * GU_FLAT] in SBUF → replicate to [128, H_free * GU_FLAT].
        # But GU_FLAT=384 * H_free=16 = 6144 fp32 elements per row = 24 KiB... large.
        #
        # SIMPLEST CORRECT APPROACH: Process dequant h1-block by h1-block.
        # For each h1 in [0..H_free), load scale vector [GU_FLAT=384] for this h1 block,
        # then multiply gate_up_fp8_buf[..., h1, :] element-wise.
        # Scale [GU_FLAT] = 384 scalars, one per column j. Load into [1, GU_FLAT] SBUF,
        # use nl.multiply broadcasting: (128, 384) * (1, 384) → (128, 384).
        #
        # BUT SBUF partition must always be 128 — [1, GU_FLAT] is not valid for SBUF.
        # Use [128, GU_FLAT] where all 128 rows have the same scale values (broadcast at load).
        #
        # Actually: just load the h1-block scale into a [128, GU_FLAT] buffer using
        # a linear DMA (no indirect — expert-id is already determined, just use an index).
        # Problem: we don't know expert_id at compile time for the linear DMA address.
        # Solution: use the scale bufs already loaded per-expert with indirect DMA.
        # Re-think the scale buf layout to be per-h1-block accessible.
        #
        # FINAL APPROACH: Load scales as before into [128P, 48F] with the pattern.
        # Then dequant:
        #   scale_buf[p, j_blk*16 + h1] = scales[e, j_blk*128+p, h1]
        #   fp8_buf[p, h1, j_blk*128+j_p] should be scaled by scales[e, j_blk*128+j_p, h1]
        #   where j_p is the position within the 128-wide j_blk.
        #   scales[e, j_blk*128+j_p, h1] = scale_buf[j_p, j_blk*16+h1]
        # So for the element at partition=p, h1=h1, j=j_blk*128+j_p:
        #   scale = scale_buf[j_p, j_blk*16+h1]  (the j_p partition row!)
        # But we're at partition=p, not partition=j_p. This only works if p==j_p, i.e.,
        # the partition dimension is also the j dimension — which it is for a transpose!
        #
        # KEY INSIGHT: gate_up_fp8_buf has shape [128P, H_free, GU_FLAT].
        # Partition dimension p indexes the H-block rows (128 rows per block = one PMAX tile).
        # The j (column) dimension is in the free dimension.
        # So scale[j, h1] is NOT per-partition — it's per-column.
        # To dequant, we need to multiply each column j by its scalar.
        # This is a (128, H_free, GU_FLAT) * broadcast(1, H_free, GU_FLAT) operation.
        # We need scale broadcast over partition dim.
        #
        # REVISED SCALE LOAD: load scales into SBUF as [128P, H_free * GU_FLAT // 128F]
        # but all partition rows identical. We can't easily do this with dma_copy.
        #
        # PRACTICAL SOLUTION: Load each expert's gate_up scales for ONE h1-block at a time
        # using direct SBUF addressing (no indirect DMA), computed from the expert scratch buf.
        # Actually this gets complicated. Let me use the simplest working approach:
        #
        # SIMPLEST WORKING APPROACH:
        # Load gate_up scale for this expert: gate_up_scales[e, :, h1] is a [GU_FLAT=384] vector.
        # Store into a [GU_FLAT // T=384] scratchpad and replicate.
        # Use nisa.dma_copy with a reshape:
        #   scale_h1_flat[128, 3] = gate_up_scales[e, :, h1].reshape(3, 128).T
        # Then tensor_tensor can multiply:
        #   gate_up_buf[p, h1, j_blk*128+q] *= scale_h1_flat[q, j_blk]  → NO, wrong dim
        #
        # CORRECT FINAL APPROACH: Transpose the scales so partition=j:
        # gate_up_scales[e, j, h1]: shape [GU_FLAT=384, H_scale=16].
        # Load as [PMAX=128, GU_FLAT//PMAX * H_scale] = [128, 3*16=48] where
        # scale_buf[q, j_blk * H_scale + h1] = scales[e, j_blk*128+q, h1].
        # This is EXACTLY what the pattern above loads. And it IS partition-varying (q = j%128).
        #
        # Now for dequant of gate_up_fp8_buf[p, h1, j] where j = j_blk*128+j_q:
        #   scale = scale_buf[j_q, j_blk*16 + h1]   (q is j%128 = column residual)
        #   But we're at partition row p (row within H-block), NOT j_q.
        #
        # This is the fundamental problem: the scale varies over the free dimension (j),
        # but we're indexing the partition dim. We need the element at (partition=p, free=j)
        # to be multiplied by scale[j, h1], not scale[p, h1].
        #
        # The ONLY clean way to do this in NKI without a loop is to:
        # 1. Transpose gate_up_fp8_buf so partition = j dimension, then dequant.
        # 2. Or do it in a loop over h1 and j_blk, loading one scale scalar at a time.
        #
        # Let's use approach 2: loop over h1, j_blk. For each (h1, j_blk), get the 128
        # scalar scale values (one per j_q) and multiply the 128-wide fp8 column.
        # scale_buf[0:128, j_blk*16+h1] is a column vector [128, 1] of scales.
        # gate_up_fp8_buf[0:128, h1, j_blk*128:j_blk*128+128] is [128, 128].
        # Multiply: gate_up_buf[p, h1, j_blk*128+j_q] = fp8[p, h1, j_blk*128+j_q] * scale[j_q, ...]
        # In NKI: the free dim of fp8 buf is [H_free, GU_FLAT].
        # fp8_buf[:, h1, j_blk*128:j_blk*128+128] is [128P, 128F].
        # We want to multiply by scale_buf[:, j_blk*16+h1:j_blk*16+h1+1] which is [128P, 1F].
        # This multiplies partition row p by scale[p, j_blk*16+h1] = scales[e, j_blk*128+p, h1].
        # But we want to multiply free-dim col q by scales[e, j_blk*128+q, h1].
        # p ≠ q in general.
        #
        # The scale IS per-partition-of-j, so we need to treat the j-dimension as partition.
        # Solution: load a transposed scale: scale_transposed[j_q, 1] and use as "free" operand.
        #
        # WORKING SOLUTION: Do a TRANSPOSE of the 128-wide j_blk column before multiplying.
        # fp8 block: [128P, 128F] → transpose → [128P, 128F] with swapped roles.
        # Actually nc_transpose([128, 128]) → [128, 128] (same shape, swapped p/f).
        # Then: result[j_q, p] = fp8[p, j_q]  (after transpose, partition=j, free=h_row)
        # scale_buf[j_q, j_blk*16+h1] = scale per j_q
        # Multiply: result_scaled[j_q, p] = fp8_transposed[j_q, p] * scale_buf[j_q, j_blk*16+h1:1]
        # Then transpose back to get [128P, 128F] bf16 block.
        # This costs 2 transposes per (h1, j_blk) = 2 * 16 * 3 = 96 transposes per expert.
        # That's expensive.
        #
        # SIMPLEST CORRECT SOLUTION: Just do the dequant in a SBUF loop, reading one scale scalar
        # at a time. For each j_blk in [0..3) and h1 in [0..H_free), read scale_buf[0:128, j_blk*16+h1]
        # (which is [128, 1] = scales per j_q for this j_blk, h1) and multiply fp8 block.
        # Since scale varies by partition (== j_q), tensor_scalar with operand0=scale_buf[:, col] works.
        # Wait: nisa.tensor_scalar(dst, data, scalar0, op0) — scalar0 is a PYTHON SCALAR, not a tensor.
        # For tensor×tensor: nisa.tensor_tensor(dst, data1, data2, op) — data2 can be [128, 1].
        #
        # THIS WORKS: for each (h1, j_blk):
        #   scale_col = scale_buf[:, j_blk*16+h1:j_blk*16+h1+1]  # [128, 1]
        #   fp8_block = fp8_buf[:, h1, j_blk*128:j_blk*128+128]   # [128, 128]
        #   bf16_block = cast(fp8_block) * scale_col              # [128, 128] × [128, 1] → broadcast over free dim
        # Wait: broadcast [128, 1] → [128, 128] by repeating along free dim.
        # nisa.tensor_tensor broadcasts when one operand has free-dim 1? Let me check...
        # In NKI, tensor_tensor requires same shape or broadcasting rules.
        # The free dimension can be broadcast: [128P, 1F] broadcast to [128P, 128F].
        # So: nisa.tensor_tensor(dst=[128,128], data1=fp8_cast[128,128], data2=scale_col[128,1], nl.multiply)
        # should work with broadcast along free dim.
        #
        # Total ops: 3 j_blks × 16 h1 = 48 tensor_tensor per expert × 8 experts = 384 ops.
        # This is in the compute phase, overlapping with DMA. Should be acceptable.

        for k in nl.static_range(_K_WAVE):
            # Dequantize gate_up: [128P, H_free, GU_FLAT] fp8 → bf16
            # Single activation instruction: fp8 * per-partition-scale → bf16
            # scale=gate_up_scale_bufs[k][p, h1, j_blk] broadcasts [PMAX, 1] over free dim.
            GU_FLAT_BLOCKS = _GU_FLAT // _PMAX  # = 3
            for h1 in nl.static_range(H_free):
                for j_blk in nl.static_range(GU_FLAT_BLOCKS):
                    nisa.activation(
                        gate_up_bufs[k][0:_PMAX, h1, nl.ds(j_blk * _PMAX, _PMAX)],
                        op=nl.copy,
                        data=gate_up_fp8_bufs[k][0:_PMAX, h1, nl.ds(j_blk * _PMAX, _PMAX)],
                        scale=gate_up_scale_bufs[k][0:_PMAX, h1, j_blk:j_blk + 1],
                    )

            # Dequantize down tile 0: [I0=128P, H_shard] fp8 → bf16
            H_BLOCKS = H_shard // _PMAX  # = 8
            for h_blk in nl.static_range(H_BLOCKS):
                nisa.activation(
                    down_full0_bufs[k][0:_PMAX, nl.ds(h_blk * _PMAX, _PMAX)],
                    op=nl.copy,
                    data=down_full0_fp8_bufs[k][0:_PMAX, nl.ds(h_blk * _PMAX, _PMAX)],
                    scale=down_scale0_bufs[k][0:_PMAX, h_blk:h_blk + 1],
                )

            # Dequantize down tile 1: [I1=64, H_shard] fp8 → bf16
            for h_blk in nl.static_range(H_BLOCKS):
                nisa.activation(
                    down_full1_bufs[k][0:I1, nl.ds(h_blk * _PMAX, _PMAX)],
                    op=nl.copy,
                    data=down_full1_fp8_bufs[k][0:I1, nl.ds(h_blk * _PMAX, _PMAX)],
                    scale=down_scale1_bufs[k][0:I1, h_blk:h_blk + 1],
                )

        # PSUM allocation for wave 0 (4-expert capacity)
        gate_up_psum = nl.ndarray((_PMAX, _K_WAVE * 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
        down_psum    = nl.ndarray((_PMAX, _K_WAVE * H_free_shard), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # Phase 2a: Compute experts 0-3
        for k in nl.static_range(_K_WAVE):
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            # Tile-1 tensor_copy
            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free, nl.ds(I + I0, I1)],
            )

            # Gate/Up matmul
            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                    else:
                        g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]
                        u_stat = up_t1_128[0:_PMAX, h1, 0:I0]
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

            # Fuse SiLU directly from PSUM
            silu_res = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(silu_res, op=nl.silu, data=gate_up_psum[0:_PMAX, gu_base:gu_base + I_tiles])

            up_sb = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(up_sb, op=nl.copy, data=gate_up_psum[0:_PMAX, gu_base + I_tiles:gu_base + 2 * I_tiles])

            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(inter_f32, silu_res, up_sb, nl.multiply)

            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            # Down matmul
            for h1_out in nl.affine_range(H_free_shard):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full0_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            # Flush down PSUM -> SBUF, scale by expert affinity, accumulate
            down_result_sb = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free_shard],
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + H_free_shard],
            )
            down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],  # wave 0: global k = k (0-3)
            )

            if k == 0:
                nisa.tensor_copy(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    src=down_result_scaled[0:_PMAX, 0:H_free_shard],
                )
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data1=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data2=down_result_scaled[0:_PMAX, 0:H_free_shard],
                    op=nl.add,
                )

        # ==================================================================
        # WAVE 1: Experts 4-7
        # ==================================================================

        # NOTE: No down_full1 re-memset needed — DMA only writes [0:I1, :],
        # rows I1:I0 remain zeroed from initial memset before wave 0.

        # PSUM memset for wave 1
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # Phase 1b: Load experts 4-7 (reusing buffers 0-3)
        for k in nl.static_range(_K_WAVE):
            kk = k + 4  # global expert index
            nisa.dma_copy(
                dst=eid_scratch[0:1, 0:1],
                src=top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + kk),
            )
            expert_id = eid_scratch.ap(pattern=[[1, 1], [1, 1]], offset=0)

            # Load fp8 gate_up weights (int8 HBM → fp8_e4m3fn SBUF via dtype reinterpret)
            nisa.dma_copy(
                dst=gate_up_fp8_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3fn,
                ),
                dge_mode=0,
            )

            # Load gate_up scales [H=2048, J_BLOCKS=3] fp32 (same pattern as wave 0)
            nisa.dma_copy(
                dst=gate_up_scale_bufs[k],
                src=gate_up_scales.ap(
                    pattern=[[_GU_J_BLOCKS, _PMAX], [_PMAX * _GU_J_BLOCKS, H_free], [1, _GU_J_BLOCKS]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # Load fp8 down_w tile 0 (int8 HBM → fp8_e4m3fn SBUF via dtype reinterpret)
            nisa.dma_copy(
                dst=down_full0_fp8_bufs[k],
                src=down_w.ap(
                    pattern=[[H, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3fn,
                ),
                dge_mode=0,
            )

            # Load fp8 down_w tile 1 (int8 HBM → fp8_e4m3fn SBUF via dtype reinterpret)
            nisa.dma_copy(
                dst=down_full1_fp8_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H_shard]],
                    offset=I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3fn,
                ),
                dge_mode=0,
            )

            # Load down scales tile0 (same pattern as wave 0)
            nisa.dma_copy(
                dst=down_scale0_bufs[k],
                src=down_scales.ap(
                    pattern=[[_H_BLOCKS, _PMAX], [1, _H_SHARD_BLOCKS]],
                    offset=prg_id * _H_SHARD_BLOCKS,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # Load down scales tile1
            nisa.dma_copy(
                dst=down_scale1_bufs[k][0:I1, 0:_H_SHARD_BLOCKS],
                src=down_scales.ap(
                    pattern=[[_H_BLOCKS, I1], [1, _H_SHARD_BLOCKS]],
                    offset=I0 * _H_BLOCKS + prg_id * _H_SHARD_BLOCKS,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # Dequantize Wave 1 weights (single-instruction fp8*scale→bf16, same as wave 0)
        for k in nl.static_range(_K_WAVE):
            GU_FLAT_BLOCKS = _GU_FLAT // _PMAX  # = 3
            for h1 in nl.static_range(H_free):
                for j_blk in nl.static_range(GU_FLAT_BLOCKS):
                    nisa.activation(
                        gate_up_bufs[k][0:_PMAX, h1, nl.ds(j_blk * _PMAX, _PMAX)],
                        op=nl.copy,
                        data=gate_up_fp8_bufs[k][0:_PMAX, h1, nl.ds(j_blk * _PMAX, _PMAX)],
                        scale=gate_up_scale_bufs[k][0:_PMAX, h1, j_blk:j_blk + 1],
                    )

            H_BLOCKS = H_shard // _PMAX  # = 8
            for h_blk in nl.static_range(H_BLOCKS):
                nisa.activation(
                    down_full0_bufs[k][0:_PMAX, nl.ds(h_blk * _PMAX, _PMAX)],
                    op=nl.copy,
                    data=down_full0_fp8_bufs[k][0:_PMAX, nl.ds(h_blk * _PMAX, _PMAX)],
                    scale=down_scale0_bufs[k][0:_PMAX, h_blk:h_blk + 1],
                )

            for h_blk in nl.static_range(H_BLOCKS):
                nisa.activation(
                    down_full1_bufs[k][0:I1, nl.ds(h_blk * _PMAX, _PMAX)],
                    op=nl.copy,
                    data=down_full1_fp8_bufs[k][0:I1, nl.ds(h_blk * _PMAX, _PMAX)],
                    scale=down_scale1_bufs[k][0:I1, h_blk:h_blk + 1],
                )

        # Phase 2b: Compute experts 4-7
        for k in nl.static_range(_K_WAVE):
            kk = k + 4  # global expert index for affinity lookup
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            # Tile-1 tensor_copy
            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free, nl.ds(I + I0, I1)],
            )

            # Gate/Up matmul (identical to wave 0)
            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                    else:
                        g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]
                        u_stat = up_t1_128[0:_PMAX, h1, 0:I0]
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

            # SiLU + multiply + cast
            silu_res = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(silu_res, op=nl.silu, data=gate_up_psum[0:_PMAX, gu_base:gu_base + I_tiles])

            up_sb = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(up_sb, op=nl.copy, data=gate_up_psum[0:_PMAX, gu_base + I_tiles:gu_base + 2 * I_tiles])

            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(inter_f32, silu_res, up_sb, nl.multiply)

            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            # Down matmul
            for h1_out in nl.affine_range(H_free_shard):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full0_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            # Flush down PSUM -> SBUF, scale by expert affinity, accumulate
            down_result_sb = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free_shard],
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + H_free_shard],
            )
            down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, kk:kk + 1],  # wave 1: global k = kk (4-7)
            )

            # Always accumulate (output_temp already initialized by wave 0)
            nisa.tensor_tensor(
                dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                data1=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                data2=down_result_scaled[0:_PMAX, 0:H_free_shard],
                op=nl.add,
            )

    # -----------------------------------------------------------------------
    # Stage 5: Transpose fp32->bf16, store to HBM
    # -----------------------------------------------------------------------
    output = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.shared_hbm)
    out_sb = nl.ndarray((T, H_shard), dtype=inp.dtype, buffer=nl.sbuf)

    for h1 in nl.static_range(H_free_shard):
        tp_psum = nl.ndarray((T, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp_psum[0:T, 0:_PMAX], data=output_temp[0:_PMAX, h1, 0:T])
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
    """Run kernel_v29_fp8 with fp8 weight layouts.

    Accepts:
      gate_up_w: [E, H, 2*I=384] fp8_e4m3fn
      gate_up_scales: [E, 2*I=384, H//128=16] fp32
      down_w: [E, I=192, H=2048] fp8_e4m3fn
      down_scales: [E, H=2048, I//128=2] fp32

    Returns: output [T, H=2048] bf16
    """
    import torch
    import torch_xla.core.xla_model as xm

    # gate_up_w and down_w are passed as int8 (fp8 reinterpreted)
    assert gate_up_w.shape == (_E, _H, _GU_FLAT), f"gate_up_w shape {gate_up_w.shape}"
    assert gate_up_scales.shape == (_E, _H, _GU_J_BLOCKS), f"gate_up_scales shape {gate_up_scales.shape}"
    assert down_w.shape == (_E, _I, _H), f"down_w shape {down_w.shape}"
    assert down_scales.shape == (_E, _I, _H_BLOCKS), f"down_scales shape {down_scales.shape}"

    xm.mark_step()

    outputs = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
