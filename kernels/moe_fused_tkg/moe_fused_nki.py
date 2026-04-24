"""
Fused Qwen3 MoE kernel with optional hoisted gamma and router prefetch.

If `gamma_sb_ready` or `router_w_wide_sb` is supplied, the caller owns those
SBUF tensors and the corresponding load path is skipped.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.tensor_view import TensorView
from nkilib.core.utils.allocator import SbufManager

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
_ROUTER_BATCH = 16

# 2-wave constants
_K_WAVE = 4  # experts per wave

def _qwen3_moe_sbuf_in_sbuf_out_hoisted(
    inp_sb,
    dtype,
    T,
    gamma,
    router_w,
    gate_up_w,
    down_w,
    sbm=None,
    gamma_sb_ready=None,
    router_w_wide_sb=None,
):
    """
    Fused RMSNorm + router + top-k + expert MLP.
    Returns column-major bf16 output in SBUF.
    """
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
    rmsnorm_normed_bf16 = sbm.alloc_heap((_PMAX, H_free * T), dtype, buffer=nl.sbuf, name="rmsnorm_normed_bf16")
    rmsnorm_normed = sbm.alloc_heap((_PMAX, H_free * T), nl.float32, buffer=nl.sbuf, name="rmsnorm_normed")

    sbm.open_scope(name="rmsnorm")

    rmsnorm_out = inp_sb

    if gamma_sb_ready is None:
        gamma_1d = gamma.reshape((H,))
        gamma_1d_hbm_reshaped = gamma_1d.reshape((H_free, _PMAX))
        gamma_flat_sb = sbm.alloc_stack((H_free, _PMAX), gamma.dtype, buffer=nl.sbuf, name="gamma_flat_sb")
        nisa.dma_copy(dst=gamma_flat_sb, src=gamma_1d_hbm_reshaped, dge_mode=3)
        gamma_trans_psum = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=gamma_trans_psum, data=gamma_flat_sb)
        gamma_sb = sbm.alloc_stack((_PMAX, H_free), gamma.dtype, buffer=nl.sbuf, name="gamma_sb")
        nisa.activation(gamma_sb[...], op=nl.copy, data=gamma_trans_psum[...])
    else:
        gamma_sb = gamma_sb_ready

    rmsnorm_sq = sbm.alloc_stack((_PMAX, H_free * T), nl.float32, buffer=nl.sbuf, name="rmsnorm_sq")
    nisa.activation(rmsnorm_sq[...], op=nl.square, data=rmsnorm_out[...])

    rmsnorm_reduced = sbm.alloc_stack((_PMAX, T), nl.float32, buffer=nl.sbuf, name="rmsnorm_reduced")
    nisa.tensor_reduce(rmsnorm_reduced[0:_PMAX, 0:T], nl.add, rmsnorm_sq[0:_PMAX, 0:H_free * T], axis=1)

    sum_reduced_sb = sbm.alloc_stack((1, T), nl.float32, buffer=nl.sbuf, name="sum_reduced_sb")
    nisa.tensor_partition_reduce(dst=sum_reduced_sb[0:1, 0:T], data=rmsnorm_reduced[0:_PMAX, 0:T], op=nl.add)

    norm_sum_sb = sbm.alloc_stack((_PMAX, T), nl.float32, buffer=nl.sbuf, name="norm_sum_sb")
    nisa.tensor_copy(dst=norm_sum_sb[0:1, 0:T], src=sum_reduced_sb[0:1, 0:T])
    for g in nl.static_range(4):
        nisa.nc_stream_shuffle(
            dst=norm_sum_sb[nl.ds(g * 32, 32), 0:T],
            src=norm_sum_sb[0:1, 0:T],
            shuffle_mask=[0] * 32,
        )

    eps_sb = sbm.alloc_stack((_PMAX, 1), nl.float32, buffer=nl.sbuf, name="eps_sb")
    nisa.memset(eps_sb, value=_EPS)
    norm_factor_sb = sbm.alloc_stack((_PMAX, T), nl.float32, buffer=nl.sbuf, name="norm_factor_sb")
    nisa.activation(
        norm_factor_sb[0:_PMAX, 0:T],
        op=nl.rsqrt,
        data=norm_sum_sb[0:_PMAX, 0:T],
        scale=1.0 / H,
        bias=eps_sb[0:_PMAX, :],
    )

    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=2).broadcast(dim=2, size=H_free)
    x_scaled = sbm.alloc_stack((_PMAX, H_free * T), nl.float32, buffer=nl.sbuf, name="x_scaled")
    nisa.tensor_tensor(x_scaled[...], rmsnorm_out[...], norm_factor_bcast.get_view(), nl.multiply)
    nisa.tensor_tensor(rmsnorm_normed[...], x_scaled[...], gamma_sb[...], nl.multiply)

    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=rmsnorm_normed[...])
    sbm.close_scope()

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul [T, H] @ [H, E=128] → logits [T, E]
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)

    sbm.open_scope(name="router_softmax")

    if router_w_wide_sb is None:
        _router_w_wide_sb = sbm.alloc_stack((_PMAX, _ROUTER_BATCH, E), dtype, buffer=nl.sbuf, name="router_w_wide_sb")
    else:
        _router_w_wide_sb = router_w_wide_sb

    for h_chunk in nl.affine_range(H_free // _ROUTER_BATCH):
        if router_w_wide_sb is None:
            nisa.dma_copy(
                dst=_router_w_wide_sb,
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
                moving=_router_w_wide_sb[0:_PMAX, h_sub, 0:E],
            )

    # NxDI materializes bf16 router logits, then runs the softmax chain in fp32.
    logits_bf16 = sbm.alloc_stack((T, E), dtype, buffer=nl.sbuf, name="logits_bf16")
    nisa.activation(logits_bf16[0:T, 0:E], op=nl.copy, data=logits_psum[0:T, 0:E])
    logits_sb = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="logits_sb")
    nisa.activation(logits_sb[0:T, 0:E], op=nl.copy, data=logits_bf16[0:T, 0:E])
    sbm.pop_heap()

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8)
    # -----------------------------------------------------------------------
    max_logit = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="max_logit")
    nisa.tensor_reduce(max_logit[0:T, 0:1], nl.maximum, logits_sb[0:T, 0:E], axis=1)

    centered = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="centered")
    nisa.tensor_scalar(
        centered[0:T, 0:E], data=logits_sb[0:T, 0:E],
        op0=nl.subtract, operand0=max_logit[0:T, 0:1],
    )

    exp_vals = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="exp_vals")
    nisa.activation(exp_vals[0:T, 0:E], op=nl.exp, data=centered[0:T, 0:E])

    sum_exp = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="sum_exp")
    nisa.tensor_reduce(sum_exp[0:T, 0:1], nl.add, exp_vals[0:T, 0:E], axis=1)

    inv_sum_exp = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="inv_sum_exp")
    nisa.activation(inv_sum_exp[0:T, 0:1], op=nl.reciprocal, data=sum_exp[0:T, 0:1])

    # Top-k runs on bf16-rounded logits.
    top8_logits_bf16 = sbm.alloc_heap((T, K), dtype, buffer=nl.sbuf, name="top8_logits_bf16")
    nisa.max8(dst=top8_logits_bf16[0:T, 0:K], src=logits_bf16[0:T, 0:E])

    top8_idx = sbm.alloc_heap((T, K), nl.uint32, buffer=nl.sbuf, name="top8_idx")
    nisa.nc_find_index8(dst=top8_idx[0:T, 0:K], data=logits_bf16[0:T, 0:E], vals=top8_logits_bf16[0:T, 0:K])

    top8_logits = sbm.alloc_stack((T, K), nl.float32, buffer=nl.sbuf, name="top8_logits")
    nisa.activation(top8_logits[0:T, 0:K], op=nl.copy, data=top8_logits_bf16[0:T, 0:K])

    # Recover softmax values for the selected logits.
    top8_centered = sbm.alloc_stack((T, K), nl.float32, buffer=nl.sbuf, name="top8_centered")
    nisa.tensor_scalar(
        top8_centered[0:T, 0:K], data=top8_logits[0:T, 0:K],
        op0=nl.subtract, operand0=max_logit[0:T, 0:1],
    )
    top8_exp = sbm.alloc_stack((T, K), nl.float32, buffer=nl.sbuf, name="top8_exp")
    nisa.activation(top8_exp[0:T, 0:K], op=nl.exp, data=top8_centered[0:T, 0:K])
    top8_vals = sbm.alloc_stack((T, K), nl.float32, buffer=nl.sbuf, name="top8_vals")
    nisa.tensor_scalar(
        top8_vals[0:T, 0:K], data=top8_exp[0:T, 0:K],
        op0=nl.multiply, operand0=inv_sum_exp[0:T, 0:1],
    )

    # NxDI renormalizes selected affinities in bf16.
    top8_vals_bf16 = sbm.alloc_heap((T, K), dtype, buffer=nl.sbuf, name="top8_vals_bf16")
    nisa.activation(top8_vals_bf16[0:T, 0:K], op=nl.copy, data=top8_vals[0:T, 0:K])
    sbm.close_scope()

    # -----------------------------------------------------------------------
    # Stage 4: Selective-Expert MLP — 2-Wave Expert Processing
    # -----------------------------------------------------------------------
    sbm.open_scope(name="expert_loop_outer")

    # output_temp: fp32 accumulator. HLO shows reference's torch.sum(bf16_tensor, dim=0) lowers
    # to a bf16 reduce.503 that the Neuron compiler implements with fp32-internal accumulation
    # (flags: --enable-mixed-precision-accumulation, --accumulate-on-alu-dtype) and a single
    # bf16 round at the end. Matching that = fp32 accumulator + one final bf16 cast at Stage 5.
    output_temp = sbm.alloc_stack((_PMAX, H_free_shard, T), nl.float32, buffer=nl.sbuf, name="output_temp")

    for t in nl.static_range(T):

        sbm.open_scope(name=f"token_{t}")

        # Token-scoped expert buffers reused across both waves.
        gate_up_buf0 = sbm.alloc_stack((_PMAX, H_free_shard, _GU_FLAT), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf0_t{t}")
        gate_up_buf1 = sbm.alloc_stack((_PMAX, H_free_shard, _GU_FLAT), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf1_t{t}")
        gate_up_buf2 = sbm.alloc_stack((_PMAX, H_free_shard, _GU_FLAT), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf2_t{t}")
        gate_up_buf3 = sbm.alloc_stack((_PMAX, H_free_shard, _GU_FLAT), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf3_t{t}")
        gate_up_bufs = [gate_up_buf0, gate_up_buf1, gate_up_buf2, gate_up_buf3]

        down_full0_buf0 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf0_t{t}")
        down_full0_buf1 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf1_t{t}")
        down_full0_buf2 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf2_t{t}")
        down_full0_buf3 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf3_t{t}")
        down_full0_bufs = [down_full0_buf0, down_full0_buf1, down_full0_buf2, down_full0_buf3]

        down_full1_buf0 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf0_t{t}")
        down_full1_buf1 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf1_t{t}")
        down_full1_buf2 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf2_t{t}")
        down_full1_buf3 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf3_t{t}")
        down_full1_bufs = [down_full1_buf0, down_full1_buf1, down_full1_buf2, down_full1_buf3]

        for k_pad in range(4):
            nisa.memset(down_full1_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        gate_t1_128 = sbm.alloc_stack((_PMAX, H_free_shard, I0), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_t1_128_t{t}")
        up_t1_128   = sbm.alloc_stack((_PMAX, H_free_shard, I0), gate_up_w.dtype, buffer=nl.sbuf, name=f"up_t1_128_t{t}")
        nisa.memset(gate_t1_128[0:_PMAX, 0:H_free_shard, nl.ds(I1, I1)], value=0.0)
        nisa.memset(up_t1_128[0:_PMAX, 0:H_free_shard, nl.ds(I1, I1)],   value=0.0)

        # ==================================================================
        # WAVE 0: Experts 0-3
        # ==================================================================

        for k in nl.static_range(_K_WAVE):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            nisa.dma_copy(
                dst=gate_up_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free_shard], [1, _GU_FLAT]],
                    offset=prg_id * H_free_shard * _PMAX * _GU_FLAT,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

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

        # Renormalize selected affinities in bf16.
        sum_topk = sbm.alloc_stack((T, 1), dtype, buffer=nl.sbuf, name=f"sum_topk_t{t}")
        nisa.tensor_reduce(sum_topk[0:T, 0:1], nl.add, top8_vals_bf16[0:T, 0:K], axis=1)

        inv_sum_topk = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name=f"inv_sum_topk_t{t}")
        nisa.activation(inv_sum_topk[0:T, 0:1], op=nl.reciprocal, data=sum_topk[0:T, 0:1])

        norm_weights = sbm.alloc_stack((T, K), dtype, buffer=nl.sbuf, name=f"norm_weights_t{t}")
        nisa.tensor_scalar(
            norm_weights[0:T, 0:K], data=top8_vals_bf16[0:T, 0:K],
            op0=nl.multiply, operand0=inv_sum_topk[0:T, 0:1],
        )

        norm_weights_f32 = sbm.alloc_stack((T, K), nl.float32, buffer=nl.sbuf, name=f"norm_weights_f32_t{t}")
        nisa.activation(norm_weights_f32[0:T, 0:K], op=nl.copy, data=norm_weights[0:T, 0:K])
        aff_bcast = sbm.alloc_stack((_PMAX, K), nl.float32, buffer=nl.sbuf, name=f"aff_bcast_t{t}")
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K], src=norm_weights_f32[t:t + 1, 0:K])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        gate_up_psum = nl.ndarray((_PMAX, _K_WAVE * 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
        down_psum    = nl.ndarray((_PMAX, _K_WAVE * H_free_shard), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        for k in nl.static_range(_K_WAVE):
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_free_shard, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free_shard, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_free_shard, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free_shard, nl.ds(I + I0, I1)],
            )

            for h1 in nl.affine_range(H_free_shard):
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
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((prg_id * H_free_shard + h1) * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((prg_id * H_free_shard + h1) * T, T)],
                    )

        gu_full_bf16_w0 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), dtype, buffer=nl.sbuf, name=f"gu_full_bf16_w0_t{t}")

        sbm.open_scope(name=f"w0_allreduce_t{t}")

        gu_send_w0 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), nl.float32, buffer=nl.sbuf, name=f"gu_send_w0_t{t}")
        gu_recv_w0 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), nl.float32, buffer=nl.sbuf, name=f"gu_recv_w0_t{t}")
        gu_full_w0 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), nl.float32, buffer=nl.sbuf, name=f"gu_full_w0_t{t}")

        nisa.activation(gu_send_w0, op=nl.copy, data=gate_up_psum[0:_PMAX, 0:_K_WAVE * 2 * I_tiles])

        nisa.sendrecv(
            send_to_rank=1 - prg_id,
            recv_from_rank=1 - prg_id,
            src=gu_send_w0,
            dst=gu_recv_w0,
            pipe_id=0,
        )

        nisa.tensor_tensor(gu_full_w0, gu_send_w0, gu_recv_w0, nl.add)
        nisa.activation(gu_full_bf16_w0, op=nl.copy, data=gu_full_w0[0:_PMAX, 0:_K_WAVE * 2 * I_tiles])

        sbm.close_scope()  # w0_allreduce

        for k in nl.static_range(_K_WAVE):
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            sbm.open_scope(name=f"w0_expert_{k}_t{t}")

            silu_res = sbm.alloc_stack((_PMAX, I_tiles), dtype, buffer=nl.sbuf, name=f"silu_res_w0k{k}_t{t}")
            nisa.activation(silu_res, op=nl.silu, data=gu_full_bf16_w0[0:_PMAX, gu_base:gu_base + I_tiles])

            up_sb = sbm.alloc_stack((_PMAX, I_tiles), dtype, buffer=nl.sbuf, name=f"up_sb_w0k{k}_t{t}")
            nisa.activation(up_sb, op=nl.copy, data=gu_full_bf16_w0[0:_PMAX, gu_base + I_tiles:gu_base + 2 * I_tiles])

            inter_bf16 = sbm.alloc_stack((_PMAX, I_tiles), dtype, buffer=nl.sbuf, name=f"inter_bf16_w0k{k}_t{t}")
            nisa.tensor_tensor(inter_bf16, silu_res, up_sb, nl.multiply)

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

            # Match NxDI's bf16 round after down-proj and after affinity scaling.
            down_result_bf16 = sbm.alloc_stack((_PMAX, H_free_shard), dtype, buffer=nl.sbuf, name=f"down_result_bf16_w0k{k}_t{t}")
            nisa.activation(
                down_result_bf16,
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + H_free_shard],
            )

            down_result_scaled_bf16 = sbm.alloc_stack((_PMAX, H_free_shard), dtype, buffer=nl.sbuf, name=f"down_result_scaled_bf16_w0k{k}_t{t}")
            nisa.tensor_scalar(
                down_result_scaled_bf16,
                data=down_result_bf16,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],  # wave 0: global k = k (0-3)
            )

            down_result_scaled = sbm.alloc_stack((_PMAX, H_free_shard), nl.float32, buffer=nl.sbuf, name=f"down_result_scaled_w0k{k}_t{t}")
            nisa.activation(down_result_scaled, op=nl.copy, data=down_result_scaled_bf16)

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

            sbm.close_scope()  # w0_expert_k

        # ==================================================================
        # WAVE 1: Experts 4-7
        # ==================================================================

        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        for k in nl.static_range(_K_WAVE):
            kk = k + 4  # global expert index
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + kk)

            nisa.dma_copy(
                dst=gate_up_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free_shard], [1, _GU_FLAT]],
                    offset=prg_id * H_free_shard * _PMAX * _GU_FLAT,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

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

        for k in nl.static_range(_K_WAVE):
            kk = k + 4  # global expert index for affinity lookup
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_free_shard, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free_shard, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_free_shard, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free_shard, nl.ds(I + I0, I1)],
            )

            for h1 in nl.affine_range(H_free_shard):
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
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((prg_id * H_free_shard + h1) * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((prg_id * H_free_shard + h1) * T, T)],
                    )

        gu_full_bf16_w1 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), dtype, buffer=nl.sbuf, name=f"gu_full_bf16_w1_t{t}")

        sbm.open_scope(name=f"w1_allreduce_t{t}")

        gu_send_w1 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), nl.float32, buffer=nl.sbuf, name=f"gu_send_w1_t{t}")
        gu_recv_w1 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), nl.float32, buffer=nl.sbuf, name=f"gu_recv_w1_t{t}")
        gu_full_w1 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), nl.float32, buffer=nl.sbuf, name=f"gu_full_w1_t{t}")

        nisa.activation(gu_send_w1, op=nl.copy, data=gate_up_psum[0:_PMAX, 0:_K_WAVE * 2 * I_tiles])

        nisa.sendrecv(
            send_to_rank=1 - prg_id,
            recv_from_rank=1 - prg_id,
            src=gu_send_w1,
            dst=gu_recv_w1,
            pipe_id=1,
        )

        nisa.tensor_tensor(gu_full_w1, gu_send_w1, gu_recv_w1, nl.add)
        nisa.activation(gu_full_bf16_w1, op=nl.copy, data=gu_full_w1[0:_PMAX, 0:_K_WAVE * 2 * I_tiles])

        sbm.close_scope()  # w1_allreduce

        for k in nl.static_range(_K_WAVE):
            kk = k + 4  # global expert index for affinity lookup
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            sbm.open_scope(name=f"w1_expert_{k}_t{t}")

            silu_res = sbm.alloc_stack((_PMAX, I_tiles), dtype, buffer=nl.sbuf, name=f"silu_res_w1k{k}_t{t}")
            nisa.activation(silu_res, op=nl.silu, data=gu_full_bf16_w1[0:_PMAX, gu_base:gu_base + I_tiles])

            up_sb = sbm.alloc_stack((_PMAX, I_tiles), dtype, buffer=nl.sbuf, name=f"up_sb_w1k{k}_t{t}")
            nisa.activation(up_sb, op=nl.copy, data=gu_full_bf16_w1[0:_PMAX, gu_base + I_tiles:gu_base + 2 * I_tiles])

            inter_bf16 = sbm.alloc_stack((_PMAX, I_tiles), dtype, buffer=nl.sbuf, name=f"inter_bf16_w1k{k}_t{t}")
            nisa.tensor_tensor(inter_bf16, silu_res, up_sb, nl.multiply)

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

            # Match NxDI's bf16 round after down-proj and after affinity scaling.
            down_result_bf16 = sbm.alloc_stack((_PMAX, H_free_shard), dtype, buffer=nl.sbuf, name=f"down_result_bf16_w1k{k}_t{t}")
            nisa.activation(
                down_result_bf16,
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + H_free_shard],
            )
            down_result_scaled_bf16 = sbm.alloc_stack((_PMAX, H_free_shard), dtype, buffer=nl.sbuf, name=f"down_result_scaled_bf16_w1k{k}_t{t}")
            nisa.tensor_scalar(
                down_result_scaled_bf16,
                data=down_result_bf16,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, kk:kk + 1],  # wave 1: global k = kk (4-7)
            )
            down_result_scaled = sbm.alloc_stack((_PMAX, H_free_shard), nl.float32, buffer=nl.sbuf, name=f"down_result_scaled_w1k{k}_t{t}")
            nisa.activation(down_result_scaled, op=nl.copy, data=down_result_scaled_bf16)

            nisa.tensor_tensor(
                dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                data1=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                data2=down_result_scaled[0:_PMAX, 0:H_free_shard],
                op=nl.add,
            )

            sbm.close_scope()  # w1_expert_k

        sbm.close_scope()  # token_t

    sbm.close_scope()  # expert_loop_outer

    sbm.pop_heap()  # top8_vals_bf16
    sbm.pop_heap()  # top8_idx
    sbm.pop_heap()  # top8_logits_bf16
    sbm.pop_heap()  # rmsnorm_normed_bf16

    sbm.open_scope(name="store")
    out_sb = sbm.alloc_stack((_PMAX, H_free_shard * T), dtype, buffer=nl.sbuf, name="out_sb")
    nisa.activation(
        out_sb[0:_PMAX, 0:H_free_shard * T],
        op=nl.copy,
        data=output_temp.reshape((_PMAX, H_free_shard * T))[0:_PMAX, 0:H_free_shard * T],
    )
    sbm.close_scope()  # store

    return out_sb  # [PMAX=128, H_free_shard*T=8] bf16 — column-major, partition-first
                   # caller must consume before any further sbm.alloc_stack


@nki.jit
def qwen3_moe_fused_tkg_sbuf_io(inp, gamma, router_w, gate_up_w, down_w):
    """Back-compat wrapper over the hoisted sub-kernel."""
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)

    B = inp.shape[0]
    T = B
    H_free = _H // _PMAX

    inp_2d = inp.reshape((T, _H))
    inp_2d_hbm_reshaped = inp_2d.reshape((H_free * T, _PMAX))

    sbm.open_scope(name="inp_load")
    inp_flat_sb = sbm.alloc_stack((H_free * T, _PMAX), inp.dtype, buffer=nl.sbuf, name="inp_flat_sb_wrap")
    nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_reshaped, dge_mode=3)
    inp_trans_psum = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)
    inp_sb = sbm.alloc_stack((_PMAX, H_free * T), inp.dtype, buffer=nl.sbuf, name="inp_sb_wrap")
    nisa.activation(inp_sb[...], op=nl.copy, data=inp_trans_psum[...])

    out_sb = _qwen3_moe_sbuf_in_sbuf_out_hoisted(
        inp_sb, inp.dtype, T, gamma, router_w, gate_up_w, down_w,
        sbm=sbm,
        gamma_sb_ready=None,
        router_w_wide_sb=None,
    )

    prg_id = nl.program_id(axis=0)
    output = nl.ndarray((T, _H), dtype=inp.dtype, buffer=nl.shared_hbm)

    sbm.open_scope(name="store_hbm")
    out_row_sb = sbm.alloc_stack((T, _H_SHARD), inp.dtype, buffer=nl.sbuf, name="out_row_sb")

    for h1 in nl.static_range(_H_FREE_SHARD):
        tp_psum = nl.ndarray((T, _PMAX), dtype=inp.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=tp_psum[0:T, 0:_PMAX],
            data=out_sb[0:_PMAX, nl.ds(h1 * T, T)],
        )
        nisa.activation(
            dst=out_row_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)],
            op=nl.copy,
            data=tp_psum[0:T, 0:_PMAX],
        )

    nisa.dma_copy(
        dst=output[0:T, nl.ds(prg_id * _H_SHARD, _H_SHARD)],
        src=out_row_sb[0:T, 0:_H_SHARD],
    )
    sbm.close_scope()  # store_hbm

    sbm.close_scope()  # inp_load

    return output


def run(inp, gamma, router_w, gate_up_w, down_w):
    """Run kernel_v30c_hoisted with native weight layouts — back-compat mode (both kwargs=None).

    Accepts gate_up_w as either:
      [E, H, 2*I=384]        — flat native (gate cols 0:I, up cols I:2I)
      [E, H, 2, I=192]       — 4D view (reshaped at zero cost)
      [E, H, 2, I_padded=256] — 4D padded view from test harness (sliced to I=192)

    down_w: [E, I=192, H=2048]       — native layout.
            [E, I_padded=256, H=2048] — padded layout from test harness (sliced to I=192).

    Returns: output [T, H=2048] bf16
    """
    import torch
    import torch_xla.core.xla_model as xm

    # Accept [E, H, 2, I_any] from qwen_fused_moe_tkg.py or test harness
    if gate_up_w.dim() == 4:
        E, Hd, two, Iv = gate_up_w.shape
        if Iv != _I:
            gate_up_w = gate_up_w[:, :, :, :_I]
        gate_up_w = torch.cat([gate_up_w[:, :, 0, :], gate_up_w[:, :, 1, :]], dim=2)

    # Accept [E, H, 2*I_any] flat — slice to _GU_FLAT if needed
    if gate_up_w.dim() == 3 and gate_up_w.shape[2] != _GU_FLAT:
        gate_up_w = gate_up_w[:, :, :_GU_FLAT]

    # Slice down_w if padded
    if down_w.shape[1] != _I:
        down_w = down_w[:, :_I, :]

    assert gate_up_w.shape == (
        _E, _H, _GU_FLAT
    ), f"gate_up_w shape {gate_up_w.shape} != ({_E}, {_H}, {_GU_FLAT})"
    assert down_w.shape == (
        _E, _I, _H
    ), f"down_w shape {down_w.shape} != ({_E}, {_I}, {_H})"

    xm.mark_step()

    outputs = qwen3_moe_fused_tkg_sbuf_io[2](inp, gamma, router_w, gate_up_w, down_w)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
