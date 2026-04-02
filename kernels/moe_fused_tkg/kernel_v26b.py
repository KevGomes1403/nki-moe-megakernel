"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

kernel_v26b — Plan B: Deferred PSUM Flush + Fused Activation(scale) + DMA-First Reorder
=============================================================
Based on v25e (Plan E: Combined micro-optimizations).

Changes vs v25e:
  Change 1 (DMA-First Reorder): Move Phase 1 DMAs before pad memsets so that
    memsets overlap with DMA in-flight. No memory conflicts since DMAs write
    to rows 0:64 while memsets target rows 64:128.

  Change 2 (Deferred PSUM Flush + Fused Activation(scale)): Instead of flushing
    each expert's PSUM immediately, defer the flush to the beginning of the next
    expert's iteration. The ScalarE activation(copy) then overlaps with VectorE
    tile-1 copies. Also fuses PSUM flush + affinity scale into one
    nisa.activation(..., scale=aff_bcast) call, eliminating a tensor_scalar.

  Change 3 (DMA Transpose in Stage 1): SKIPPED — dma_transpose requires 4D src.
    Falling back to original dma_copy + nc_transpose + activation(copy).

Interface contract (no repack required):
  gate_up_w  [E=128, H=2048, 2*I=384] bf16
  down_w     [E=128, I=192,  H=2048]   bf16
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
_I = 192     # actual intermediate dim per TP rank
_I0 = 128    # first tile  (full 128 rows)
_I1 = 64     # second tile (partial: 64 valid rows, 64 zero-padded)
_I_TILES = 2 # two I-dimension tiles
_EPS = 1e-6

# Flat gate+up combined width
_GU_FLAT = 2 * _I   # = 384

# LNC=2 H-sharding constants
_N_PRGS = 2
_H_FREE = _H // _PMAX             # = 16
_H_FREE_SHARD = _H_FREE // _N_PRGS   # = 8
_H_SHARD = _H_FREE_SHARD * _PMAX     # = 1024

# Router DMA batching
_ROUTER_BATCH = 4


@nki.jit
def qwen3_moe_fused_tkg(
    inp,        # [B, 1, H=2048]  bf16
    gamma,      # [1, H=2048]     bf16
    router_w,   # [H=2048, E=128] bf16
    gate_up_w,  # [E=128, H=2048, 2*I=384] bf16
    down_w,     # [E=128, I=192,  H=2048]  bf16
):
    """
    Fused RMSNorm + Router + TopK(K=8) + Expert MLP for Qwen3 MoE TKG.
    Invoked as: qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
    Returns: output [T, H=2048] bf16.
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
    # (dma_transpose requires 4D src; falling back to dma_copy+nc_transpose+activation)
    # -----------------------------------------------------------------------
    inp_2d = inp.reshape((T, H))

    # Load inp: HBM [H_free*T, _PMAX] → SBUF via dge_mode=3, transpose, flush
    inp_2d_hbm_reshaped = inp_2d.reshape((H_free * T, _PMAX))
    inp_flat_sb = nl.ndarray((H_free * T, _PMAX), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_reshaped, dge_mode=3)
    inp_trans_psum = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)
    rmsnorm_out = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_out[...], op=nl.copy, data=inp_trans_psum[...])

    # Load gamma: HBM [H_free, _PMAX] → SBUF → transpose → SBUF gamma_sb
    gamma_1d = gamma.reshape((H,))
    gamma_1d_hbm_reshaped = gamma_1d.reshape((H_free, _PMAX))
    gamma_flat_sb = nl.ndarray((H_free, _PMAX), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_flat_sb, src=gamma_1d_hbm_reshaped, dge_mode=3)
    gamma_trans_psum = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=gamma_trans_psum, data=gamma_flat_sb)
    gamma_sb = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.activation(gamma_sb[...], op=nl.copy, data=gamma_trans_psum[...])

    # RMSNorm: compute rms, apply norm and gamma
    # 1a. x^2
    rmsnorm_sq = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(rmsnorm_sq[...], op=nl.square, data=rmsnorm_out[...])

    # 1b. Reduce x^2 over H (axis=1) → [_PMAX, T]
    rmsnorm_reduced = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(rmsnorm_reduced[0:_PMAX, 0:T], nl.add, rmsnorm_sq[0:_PMAX, 0:H_free * T], axis=1)

    # 1c. gamma * input
    gamma_mult = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(gamma_mult[...], rmsnorm_out[...], gamma_sb[...], nl.multiply)

    # 1d. Cross-partition sum via GpSimdE
    sum_reduced_sb = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=sum_reduced_sb[0:1, 0:T], data=rmsnorm_reduced[0:_PMAX, 0:T], op=nl.add)

    # Broadcast [1, T] → [_PMAX, T]
    norm_sum_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=norm_sum_sb[0:1, 0:T], src=sum_reduced_sb[0:1, 0:T])
    for g in nl.static_range(4):
        nisa.nc_stream_shuffle(
            dst=norm_sum_sb[nl.ds(g * 32, 32), 0:T],
            src=norm_sum_sb[0:1, 0:T],
            shuffle_mask=[0] * 32,
        )

    # 1e. norm_factor = rsqrt(sum/H + eps)
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

    # 1f. rmsnorm_normed = gamma_mult * norm_factor
    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=2).broadcast(dim=2, size=H_free)
    rmsnorm_normed = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(rmsnorm_normed[...], gamma_mult[...], norm_factor_bcast.get_view(), nl.multiply)

    # Cast to bf16
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
            dge_mode=0,
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

    # TopK via DVE hardware
    top8_vals = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_vals[0:T, 0:K], src=probs[0:T, 0:E])

    top8_idx = nl.ndarray((T, K), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=top8_idx[0:T, 0:K], data=probs[0:T, 0:E], vals=top8_vals[0:T, 0:K])

    # -----------------------------------------------------------------------
    # Stage 4: Selective-Expert MLP — Plan B optimizations
    #
    # Change 1 (DMA-First): Phase 1 DMAs issued BEFORE pad memsets.
    # Change 2 (Deferred PSUM Flush): PSUM flush deferred to next expert's
    #   iteration start, fused with affinity scale via activation(..., scale=).
    # -----------------------------------------------------------------------
    output_temp = nl.ndarray((_PMAX, H_free_shard, T), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.static_range(T):

        # ------------------------------------------------------------------
        # Allocate K=8 named SBUF buffers
        # ------------------------------------------------------------------
        gate_up_buf0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf2 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf3 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf4 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf5 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf6 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf7 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_bufs = [gate_up_buf0, gate_up_buf1, gate_up_buf2, gate_up_buf3,
                        gate_up_buf4, gate_up_buf5, gate_up_buf6, gate_up_buf7]

        down_full0_buf0 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf1 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf2 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf3 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf4 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf5 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf6 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf7 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_bufs = [down_full0_buf0, down_full0_buf1, down_full0_buf2, down_full0_buf3,
                           down_full0_buf4, down_full0_buf5, down_full0_buf6, down_full0_buf7]

        down_full1_buf0 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf1 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf2 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf3 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf4 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf5 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf6 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf7 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_bufs = [down_full1_buf0, down_full1_buf1, down_full1_buf2, down_full1_buf3,
                           down_full1_buf4, down_full1_buf5, down_full1_buf6, down_full1_buf7]

        # ------------------------------------------------------------------
        # Change 1 (DMA-First): Phase 1 DMAs BEFORE pad memsets
        # Issue ALL K=8 × 3 = 24 DMAs first so memsets overlap with DMA.
        # ------------------------------------------------------------------
        for k in nl.static_range(K):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            # DMA 1: gate_up — ONE coalesced load per expert
            nisa.dma_copy(
                dst=gate_up_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # DMA 2a: down tile 0
            nisa.dma_copy(
                dst=down_full0_bufs[k],
                src=down_w.ap(
                    pattern=[[H, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # DMA 2b: down tile 1
            nisa.dma_copy(
                dst=down_full1_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H_shard]],
                    offset=I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # ------------------------------------------------------------------
        # Pad memsets — now overlap with DMA in-flight
        # ------------------------------------------------------------------
        # Zero pad region (rows I1:I0 = 64:128) for all K down_full1 buffers
        for k_pad in range(K):
            nisa.memset(down_full1_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        # gate_t1_128/up_t1_128: reused buffers
        gate_t1_128 = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        up_t1_128   = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        nisa.memset(gate_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
        nisa.memset(up_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        # ------------------------------------------------------------------
        # Per-expert PSUM slots, ONE memset each
        # ------------------------------------------------------------------
        gate_up_psum = nl.ndarray((_PMAX, K * 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
        down_psum    = nl.ndarray((_PMAX, K * H_free_shard), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # ------------------------------------------------------------------
        # Compute norm_weights (overlaps with DMAs in flight)
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

        # aff_bcast setup
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
        # Phase 2: Serial expert compute with deferred PSUM flush (Change 2)
        # ------------------------------------------------------------------
        for k in nl.static_range(K):
            # Per-expert PSUM base offsets
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            # ----------------------------------------------------------
            # Deferred flush from previous expert (Change 2)
            # k==0: no previous expert
            # k==1: flush expert 0's PSUM → initialize output_temp (tensor_copy)
            # k>1:  flush expert k-1's PSUM → accumulate (tensor_tensor add)
            # ----------------------------------------------------------
            if k == 1:
                prev_d = 0 * H_free_shard
                down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    down_result_scaled[0:_PMAX, 0:H_free_shard],
                    op=nl.copy,
                    data=down_psum[0:_PMAX, prev_d:prev_d + H_free_shard],
                    scale=aff_bcast[0:_PMAX, 0:1],
                )
                nisa.tensor_copy(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    src=down_result_scaled[0:_PMAX, 0:H_free_shard],
                )
            elif k > 1:
                prev_d = (k - 1) * H_free_shard
                down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    down_result_scaled[0:_PMAX, 0:H_free_shard],
                    op=nl.copy,
                    data=down_psum[0:_PMAX, prev_d:prev_d + H_free_shard],
                    scale=aff_bcast[0:_PMAX, (k - 1):k],
                )
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data1=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data2=down_result_scaled[0:_PMAX, 0:H_free_shard],
                    op=nl.add,
                )

            # Tile-1 tensor_copy using pre-loaded gate_up_bufs[k]
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

            # Flush up from PSUM → SBUF
            up_sb = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(up_sb, op=nl.copy, data=gate_up_psum[0:_PMAX, gu_base + I_tiles:gu_base + 2 * I_tiles])

            # SiLU(gate) * up → inter_f32
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(inter_f32, silu_res, up_sb, nl.multiply)

            # Cast to bf16 for down matmul
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

            # NO PSUM flush here — deferred to next iteration (Change 2)

        # ------------------------------------------------------------------
        # After loop: flush last expert K-1
        # ------------------------------------------------------------------
        last_d = (K - 1) * H_free_shard
        down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            down_result_scaled[0:_PMAX, 0:H_free_shard],
            op=nl.copy,
            data=down_psum[0:_PMAX, last_d:last_d + H_free_shard],
            scale=aff_bcast[0:_PMAX, (K - 1):K],
        )
        nisa.tensor_tensor(
            dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
            data1=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
            data2=down_result_scaled[0:_PMAX, 0:H_free_shard],
            op=nl.add,
        )

    # -----------------------------------------------------------------------
    # Stage 5: Transpose fp32→bf16, store to HBM
    # -----------------------------------------------------------------------
    output = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.hbm)
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


def run(inp, gamma, router_w, gate_up_w, down_w):
    """Run kernel_v26b with native weight layouts — no preprocessing required."""
    import torch
    import torch_xla.core.xla_model as xm

    if gate_up_w.dim() == 4:
        E, Hd, two, Iv = gate_up_w.shape
        if Iv != _I:
            gate_up_w = gate_up_w[:, :, :, :_I]
        gate_up_w = torch.cat([gate_up_w[:, :, 0, :], gate_up_w[:, :, 1, :]], dim=2)

    if gate_up_w.dim() == 3 and gate_up_w.shape[2] != _GU_FLAT:
        gate_up_w = gate_up_w[:, :, :_GU_FLAT]

    if down_w.shape[1] != _I:
        down_w = down_w[:, :_I, :]

    assert gate_up_w.shape == (
        _E, _H, _GU_FLAT
    ), f"gate_up_w shape {gate_up_w.shape} != ({_E}, {_H}, {_GU_FLAT})"
    assert down_w.shape == (
        _E, _I, _H
    ), f"down_w shape {down_w.shape} != ({_E}, {_I}, {_H})"

    xm.mark_step()

    outputs = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
