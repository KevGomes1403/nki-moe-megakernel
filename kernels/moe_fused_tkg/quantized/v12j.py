"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

v12j — Plan J: fuse PSUM drain + scale multiplication into single activation call
==================================================================================
Based on v12i (affine_range prefetch for tile-1 tensor_copies).

Key changes vs v12i:
  V12j: Merge the PSUM drain activation + per-element scale tensor_tensor multiply
        into a single nisa.activation(scale=scale_col) call per H-shard column.

        Note: nisa.activation scale must be [N, 1] (not [N, F]), so we loop over
        the 8 H-shard columns instead of a single batched call. This fuses
        drain-from-PSUM + scale-multiply per column into one instruction.

        Before (v12i), per expert:
          Step A: nisa.activation(down_h_raw, op=nl.copy, data=down_psum[...])  # [128,8]
          Step B: nisa.tensor_tensor(down_h_all, down_h_raw, combined_scale, nl.multiply)
          Step C: nisa.tensor_tensor(output_temp, output_temp, down_h_all, nl.add)

        After (v12j), per expert × 8 columns:
          per-column scale: nisa.tensor_tensor(scale_col, down_scale[:,col], aff_col, nl.multiply)
          Step A+B fused:   nisa.activation(drain_col, op=copy, data=psum_col, scale=scale_col)
          Step C:           nisa.tensor_tensor(output_temp[:,col], ..., drain_col, nl.add)

        API constraint: activation(scale=...) requires scale=[N,1], not [N,F].
        The task description assumed [N,F] would work but the NKI API does not support it.

        Order (same as v12i):
          Pre-read 8 eid_all_bufs[0..7]
          Phase 1a: Load Wave 0 weights (affine_range, k=0..3)
          Phase 1b: Load Wave 1 weights (affine_range, k=0..3, kk=4..7)
          Compute norm_weights/aff_bcast
          Assemble up_scale bufs (static_range — depend on weight loads)
          Prefetch 2a (affine_range): fill gate_t1_128[k] and up_t1_128[k] for k=0..3
          Phase 2a: Compute experts 0-3 (static_range, memset-init output_temp)
          PSUM memset
          Prefetch 2b (affine_range): fill gate_t1_128_w1[k] and up_t1_128_w1[k] for k=0..3
          Phase 2b: Compute experts 4-7 (static_range, always-add)

Interface:
  gate_up_w  [E=128, H=2048, 2*I=384] fp8_e4m3fn (passed as int8)
  gate_up_scales [E=128, GU=384]                   fp32  per-output-neuron
  down_w     [E=128, I=192,  H=2048]   fp8_e4m3fn (passed as int8)
  down_scales [E=128, H=2048]                      fp32  per-output-neuron (full H)
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


@nki.jit
def qwen3_moe_fused_tkg(
    inp,              # [B, 1, H=2048]              bf16
    gamma,            # [1, H=2048]                 bf16
    router_w,         # [H=2048, E=128]              bf16  (router stays bf16)
    gate_up_w,        # [E=128, H=2048, 2*I=384]     int8 (reinterpreted as fp8_e4m3fn)
    gate_up_scales,   # [E=128, GU=384]               fp32  per-output-neuron
    down_w,           # [E=128, I=192,  H=2048]       int8 (reinterpreted as fp8_e4m3fn)
    down_scales,      # [E=128, H=2048]               fp32  per-output-neuron (full H)
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
            dge_mode=3,
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
    # Stage 4: Selective-Expert MLP — 2-Wave Expert Processing
    # -----------------------------------------------------------------------
    # V12i: output_temp is [128, H_free_shard=8] (T=1 always in TKG)
    output_temp = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.static_range(T):

        # ------------------------------------------------------------------
        # Allocate fp8 weight buffers — Wave 0
        # ------------------------------------------------------------------
        gate_up_fp8_buf0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_buf1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_buf2 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_buf3 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_bufs = [gate_up_fp8_buf0, gate_up_fp8_buf1, gate_up_fp8_buf2, gate_up_fp8_buf3]

        # gate_up scale buffers: [128P, 3F] — one scale per output neuron
        gate_up_scale_buf0 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_buf1 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_buf2 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_buf3 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_bufs = [gate_up_scale_buf0, gate_up_scale_buf1, gate_up_scale_buf2, gate_up_scale_buf3]

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
        gate_up_fp8_w1_buf0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_w1_buf1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_w1_buf2 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_w1_buf3 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_w1_bufs = [gate_up_fp8_w1_buf0, gate_up_fp8_w1_buf1, gate_up_fp8_w1_buf2, gate_up_fp8_w1_buf3]

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
        for k_pad in nl.static_range(4):
            nisa.memset(down_full1_fp8_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        # Zero-pad down_full1_w1 rows I1:I0 (fp8 buffers) — Wave 1
        for k_pad in nl.static_range(4):
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
        for k_pad in nl.static_range(_K_WAVE):
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
        for k_pad in nl.static_range(_K_WAVE):
            nisa.memset(gate_t1_128_w1_bufs[k_pad][0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
            nisa.memset(up_t1_128_w1_bufs[k_pad][0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        # ==================================================================
        # Pre-read all 8 expert IDs — 8 tiny 4-byte DMAs, complete nearly instantly
        # ==================================================================
        for k8 in nl.static_range(8):
            nisa.dma_copy(
                dst=eid_all_bufs[k8][0:1, 0:1],
                src=top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k8),
            )

        # Phase 1a: Load Wave 0 experts — affine_range allows pipelining across k
        for k in nl.affine_range(_K_WAVE):
            expert_id = eid_all_bufs[k].ap(pattern=[[1, 1], [1, 1]], offset=0)

            # Load fp8 gate_up weights
            nisa.dma_copy(
                dst=gate_up_fp8_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3,
                ),
                dge_mode=0,
            )

            # Load gate_up_scales[e, :] → [128P, 3F]
            nisa.dma_copy(
                dst=gate_up_scale_bufs[k],
                src=gate_up_scales.ap(
                    pattern=[[1, _PMAX], [_PMAX, _GU_J_BLOCKS]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
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
                    dtype=nl.float8_e4m3,
                ),
                dge_mode=0,
            )

            # Load down_scales[e, prg_id*1024 : prg_id*1024+1024] → [128P, 8F]
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

        # Phase 1b: Load Wave 1 experts — affine_range allows pipelining across k
        for k in nl.affine_range(_K_WAVE):
            kk = k + 4
            expert_id_w1 = eid_all_bufs[kk].ap(pattern=[[1, 1], [1, 1]], offset=0)

            # Load fp8 gate_up weights for Wave 1
            nisa.dma_copy(
                dst=gate_up_fp8_w1_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3,
                ),
                dge_mode=0,
            )

            # Load gate_up_scales for Wave 1
            nisa.dma_copy(
                dst=gate_up_scale_w1_bufs[k],
                src=gate_up_scales.ap(
                    pattern=[[1, _PMAX], [_PMAX, _GU_J_BLOCKS]],
                    offset=0,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                ),
                dge_mode=0,
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
                dge_mode=0,
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
                dge_mode=0,
            )

            # Load down_scales for Wave 1
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
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        # Assemble up scale rearranged buffers — Wave 0 (static_range: depends on weight loads)
        for k in nl.static_range(_K_WAVE):
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

        # Assemble up scale rearranged buffers — Wave 1 (static_range: depends on weight loads)
        for k in nl.static_range(_K_WAVE):
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
        # The compiler can pipeline setup of expert k+1 while expert k's
        # h1 matmul loop runs. Uses per-k allocated buffers to allow
        # affine_range iteration.
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
        # This removes the need for the `if k == 0` branch so both waves
        # can always-add, enabling affine_range on the outer expert loop.
        # ------------------------------------------------------------------
        nisa.memset(output_temp, value=0.0)

        # Phase 2a: Compute experts 0-3 (fp8 stationary)
        # Wave 1 DMA transfers overlap here since they were issued above.
        for k in nl.static_range(_K_WAVE):
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            # Gate/Up matmul (fp8 stationary) — tile-1 data from prefetch buffers
            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
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

            # Post-matmul: fused gate drain+scale+SiLU, up scale, multiply
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)

            for i_tile in nl.static_range(I_tiles):
                gate_scale_col = gate_up_scale_bufs[k][0:_PMAX, i_tile:i_tile + 1]

                # Fuse gate scale + SiLU into single activation call
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
                up_f32_scaled = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    up_f32_scaled,
                    op=nl.copy,
                    data=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                    scale=up_scale_col,
                    bias=zero_bias,
                )

                nisa.tensor_tensor(inter_f32[0:_PMAX, i_tile:i_tile + 1], gate_silu, up_f32_scaled, nl.multiply)

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

            # V12j Plan J: fuse PSUM drain + scale multiply into single activation per column.
            # nisa.activation scale must be [N, 1], so we loop over the 8 H-shard columns.
            # Each column: activation(dst[128,1], op=copy, data=psum_col[128,1],
            #               scale=combined_scale_col[128,1]) fuses drain + multiply.
            # Eliminates 1 tensor_tensor (VectorE multiply) + 1 down_h_raw SBUF buffer per expert.
            aff_col_buf = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=aff_col_buf, src=aff_bcast[0:_PMAX, k:k + 1])

            for h1_s in nl.static_range(_H_FREE_SHARD):
                # combined_down_scale_col = down_scale_bufs[k][:, h1_s] * aff_weight
                # computed per-column as [128, 1]
                down_scale_col = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    down_scale_col,
                    down_scale_bufs[k][0:_PMAX, h1_s:h1_s + 1],
                    aff_col_buf,
                    nl.multiply,
                )
                # Fused drain-from-PSUM + scale: replaces separate drain activation + tensor_tensor
                down_h_col = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    down_h_col,
                    op=nl.copy,
                    data=down_psum[0:_PMAX, d_base + h1_s:d_base + h1_s + 1],
                    scale=down_scale_col,
                )
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, h1_s:h1_s + 1],
                    data1=output_temp[0:_PMAX, h1_s:h1_s + 1],
                    data2=down_h_col[0:_PMAX, 0:1],
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
        for k in nl.static_range(_K_WAVE):
            kk = k + 4
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
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

            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)

            for i_tile in nl.static_range(I_tiles):
                gate_scale_col = gate_up_scale_w1_bufs[k][0:_PMAX, i_tile:i_tile + 1]

                # Fuse gate scale + SiLU into single activation call (Wave 1)
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
                up_f32_scaled = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    up_f32_scaled,
                    op=nl.copy,
                    data=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                    scale=up_scale_col,
                    bias=zero_bias,
                )

                nisa.tensor_tensor(inter_f32[0:_PMAX, i_tile:i_tile + 1], gate_silu, up_f32_scaled, nl.multiply)

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

            # V12j Plan J: fuse PSUM drain + scale multiply into single activation per column (Wave 1).
            # nisa.activation scale must be [N, 1], so we loop over the 8 H-shard columns.
            # Eliminates 1 tensor_tensor (VectorE multiply) + 1 down_h_raw SBUF buffer per expert.
            aff_col_buf = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=aff_col_buf, src=aff_bcast[0:_PMAX, kk:kk + 1])

            for h1_s in nl.static_range(_H_FREE_SHARD):
                # combined_down_scale_col = down_scale_w1_bufs[k][:, h1_s] * aff_weight
                down_scale_col = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    down_scale_col,
                    down_scale_w1_bufs[k][0:_PMAX, h1_s:h1_s + 1],
                    aff_col_buf,
                    nl.multiply,
                )
                # Fused drain-from-PSUM + scale: replaces separate drain activation + tensor_tensor
                down_h_col = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    down_h_col,
                    op=nl.copy,
                    data=down_psum[0:_PMAX, d_base + h1_s:d_base + h1_s + 1],
                    scale=down_scale_col,
                )
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, h1_s:h1_s + 1],
                    data1=output_temp[0:_PMAX, h1_s:h1_s + 1],
                    data2=down_h_col[0:_PMAX, 0:1],
                    op=nl.add,
                )

    # -----------------------------------------------------------------------
    # Stage 5: Transpose fp32->bf16, store to HBM
    # -----------------------------------------------------------------------
    output = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.shared_hbm)
    out_sb = nl.ndarray((T, H_shard), dtype=inp.dtype, buffer=nl.sbuf)

    # V12j: output_temp is [128, H_free_shard] — use [0:_PMAX, h1:h1+1] for [128,1] slice
    for h1 in nl.static_range(H_free_shard):
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
    """Run v12j kernel (Plan J: fuse PSUM drain + scale multiply into single activation call).

    Accepts:
      gate_up_w: [E, H, 2*I=384] fp8_e4m3fn (as int8)
      gate_up_scales: [E, GU=384] fp32  per-output-neuron
      down_w: [E, I=192, H=2048] fp8_e4m3fn (as int8)
      down_scales: [E, H=2048] fp32  per-output-neuron (full H)

    Returns: output [T, H=2048] bf16
    """
    import torch
    import torch_xla.core.xla_model as xm

    assert gate_up_w.shape == (_E, _H, _GU_FLAT), f"gate_up_w shape {gate_up_w.shape}"
    assert gate_up_scales.shape == (_E, _GU_FLAT), f"gate_up_scales shape {gate_up_scales.shape}"
    assert down_w.shape == (_E, _I, _H), f"down_w shape {down_w.shape}"
    assert down_scales.shape == (_E, _H), f"down_scales shape {down_scales.shape}"

    xm.mark_step()

    outputs = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
