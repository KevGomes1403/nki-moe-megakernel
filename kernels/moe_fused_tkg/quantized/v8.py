"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

v8 — H_shard Fix: full H=2048, 1024 per LNC prg_id
=====================================================
Based on v7 (Plan D: v6a + SiLU Fusion + Down-Scale+Affinity Fusion).

Key changes vs v7:
  V8: Fix H_shard constants to match BF16 baseline (no TP sharding at kernel level).
      _H_SHARD_TP = _H = 2048 (full H, not 512 TP shard)
      _H_SHARD = _H // _N_PRGS = 1024 per LNC prg_id (was 256 in v7)
      _H_FREE_SHARD = _H_SHARD // _PMAX = 8 (was 2 in v7)
      _H_SHARD_BLOCKS = _H_SHARD // _PMAX = 8 (was 2 in v7)
  SBUF budget (per partition):
    gate_up_fp8 × 4: 4 × (16×384) = 24,576 bytes
    down_full0_fp8 × 4: 4 × 1024 = 4,096 bytes
    down_full1_fp8 × 4: 4 × 1024 = 4,096 bytes
    down_scale × 4: 4 × 8 × 4 = 128 bytes
    gate_t1_128 + up_t1_128: 2 × (16×128) = 4,096 bytes (fp8)
    Various small buffers: ~10 KiB
    Total: ~37 KiB — well within 224 KiB ✓
  PSUM budget:
    down_psum: [128, _K_WAVE * H_free_shard] = [128, 4×8] = [128, 32] cols — within 512 max ✓

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
    H_free_shard = _H_FREE_SHARD   # = 2
    H_shard = _H_SHARD             # = 256

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
    output_temp = nl.ndarray((_PMAX, H_free_shard, T), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.static_range(T):

        # ------------------------------------------------------------------
        # Allocate fp8 weight buffers
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

        # zero_bias for post-matmul activation scale
        zero_bias = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(zero_bias, value=0.0)

        # Zero-pad down_full1 rows I1:I0 (fp8 buffers)
        for k_pad in nl.static_range(4):
            nisa.memset(down_full1_fp8_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        # gate_t1_128/up_t1_128: reused across k-iterations (fp8 for direct stationary use)
        gate_t1_128 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128   = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        nisa.memset(gate_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
        nisa.memset(up_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        # ==================================================================
        # WAVE 0: Experts 0-3
        # ==================================================================
        eid_scratch = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)

        # Phase 1a: Load experts 0-3 fp8 weights + scales
        for k in nl.static_range(_K_WAVE):
            nisa.dma_copy(
                dst=eid_scratch[0:1, 0:1],
                src=top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k),
            )
            expert_id = eid_scratch.ap(pattern=[[1, 1], [1, 1]], offset=0)

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

            # Load fp8 down_w tile 0 (rows 0:I0=128) for this shard (H_shard=256)
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

        # Assemble up scale rearranged buffers (same as v3)
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

        # PSUM allocation for wave 0
        gate_up_psum = nl.ndarray((_PMAX, _K_WAVE * 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
        down_psum    = nl.ndarray((_PMAX, _K_WAVE * H_free_shard), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # Phase 2a: Compute experts 0-3 (fp8 stationary)
        for k in nl.static_range(_K_WAVE):
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            # Tile-1 tensor_copy (fp8)
            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_fp8_bufs[k][0:_PMAX, 0:H_free, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_fp8_bufs[k][0:_PMAX, 0:H_free, nl.ds(I + I0, I1)],
            )

            # Gate/Up matmul (fp8 stationary)
            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_fp8_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gate_up_fp8_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
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

            # Post-matmul: fused gate drain+scale+SiLU, up scale, multiply
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)

            for i_tile in nl.static_range(I_tiles):
                gate_scale_col = gate_up_scale_bufs[k][0:_PMAX, i_tile:i_tile + 1]

                # V7.1: Fuse gate scale + SiLU into single activation call
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

            # V7.2: Precompute combined_down_scale = down_scale * affinity (Wave 0: use k)
            combined_down_scale = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            for h1_out_pre in nl.static_range(H_free_shard):
                nisa.tensor_tensor(
                    combined_down_scale[0:_PMAX, h1_out_pre:h1_out_pre + 1],
                    down_scale_bufs[k][0:_PMAX, h1_out_pre:h1_out_pre + 1],
                    aff_bcast[0:_PMAX, k:k + 1],
                    nl.multiply,
                )

            # Post-matmul scale for down (combined: down_scale * affinity) + accumulate
            for h1_out in nl.static_range(H_free_shard):
                down_h_final = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    down_h_final,
                    op=nl.copy,
                    data=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    scale=combined_down_scale[0:_PMAX, h1_out:h1_out + 1],
                    bias=zero_bias,
                )

                if k == 0:
                    nisa.tensor_copy(
                        dst=output_temp[0:_PMAX, h1_out:h1_out + 1, t:t + 1],
                        src=down_h_final[0:_PMAX, 0:1],
                    )
                else:
                    nisa.tensor_tensor(
                        dst=output_temp[0:_PMAX, h1_out:h1_out + 1, t:t + 1],
                        data1=output_temp[0:_PMAX, h1_out:h1_out + 1, t:t + 1],
                        data2=down_h_final[0:_PMAX, 0:1],
                        op=nl.add,
                    )

        # ==================================================================
        # WAVE 1: Experts 4-7
        # ==================================================================

        # PSUM memset for wave 1
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # Phase 1b: Load experts 4-7 (reusing buffers 0-3)
        for k in nl.static_range(_K_WAVE):
            kk = k + 4
            nisa.dma_copy(
                dst=eid_scratch[0:1, 0:1],
                src=top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + kk),
            )
            expert_id = eid_scratch.ap(pattern=[[1, 1], [1, 1]], offset=0)

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

        # Re-assemble up scale buffers for wave 1
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

        # Phase 2b: Compute experts 4-7
        for k in nl.static_range(_K_WAVE):
            kk = k + 4
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_fp8_bufs[k][0:_PMAX, 0:H_free, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_fp8_bufs[k][0:_PMAX, 0:H_free, nl.ds(I + I0, I1)],
            )

            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_fp8_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gate_up_fp8_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
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

            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)

            for i_tile in nl.static_range(I_tiles):
                gate_scale_col = gate_up_scale_bufs[k][0:_PMAX, i_tile:i_tile + 1]

                # V7.1: Fuse gate scale + SiLU into single activation call (Wave 1)
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

            # V7.2: Precompute combined_down_scale = down_scale * affinity (Wave 1: use kk)
            combined_down_scale = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            for h1_out_pre in nl.static_range(H_free_shard):
                nisa.tensor_tensor(
                    combined_down_scale[0:_PMAX, h1_out_pre:h1_out_pre + 1],
                    down_scale_bufs[k][0:_PMAX, h1_out_pre:h1_out_pre + 1],
                    aff_bcast[0:_PMAX, kk:kk + 1],
                    nl.multiply,
                )

            # Post-matmul scale for down (combined: down_scale * affinity) + accumulate
            for h1_out in nl.static_range(H_free_shard):
                down_h_final = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    down_h_final,
                    op=nl.copy,
                    data=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    scale=combined_down_scale[0:_PMAX, h1_out:h1_out + 1],
                    bias=zero_bias,
                )
                # Always accumulate (output_temp already initialized by wave 0)
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, h1_out:h1_out + 1, t:t + 1],
                    data1=output_temp[0:_PMAX, h1_out:h1_out + 1, t:t + 1],
                    data2=down_h_final[0:_PMAX, 0:1],
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
    """Run v8 kernel (Plan D + H_shard fix: full H=2048, 1024 per LNC prg_id).

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
