import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.allocator import SbufManager

from .rmsnorm_nki import _rmsnorm_sbuf_in_sbuf_out_hoisted
from .routertopk_nki import _routertopk_sbuf_in_sbuf_out

_PMAX    = 128
_H       = 2048
_E       = 128
_K       = 8
_I       = 192
_I0      = 128
_I1      = 64
_I_TILES = 2
_GU_FLAT = 2 * _I   # 384
_H_FREE  = _H // _PMAX  # 16
_H_FREE_SHARD = _H_FREE // 2
_H_SHARD = _H // 2
_GU_P    = 96
_GU_LNC_FLAT = 2 * _GU_P  # local gate + up shards packed as 192
_DOWN_P  = 96

_K_WAVE = 4  # experts per wave (two waves cover top-K=8)
_NORMALIZE_EPS_BF16 = 1.0018652574217413e-12  # bf16(1e-12) reinterpreted as fp32


def _experts_sbuf_in_sbuf_out_hshard_exact(
    rmsnorm_normed_bf16,
    top8_idx,
    top8_vals,
    dtype,
    T,
    gate_up_w,
    down_w,
    sbm=None,
    debug=False,
):
    """Baseline-compatible selective-expert path.

    nkilib's TKG selective MoE shards the hidden dimension across LNC=2 for
    gate/up, sums fp32 gate/up partials before activation, computes the down
    projection for only the local H shard, and accumulates experts sequentially
    in bf16. Keep that ordering here for bit-exactness to NxDI.
    """
    shard_id = nl.program_id(0)
    peer_lnc = 1 - shard_id
    h_free_start = shard_id * _H_FREE_SHARD
    h_hbm_start = shard_id * _H_SHARD

    sbm.open_scope(name="expert_loop_outer_exact")
    out_sb = sbm.alloc_stack((_PMAX, T, _H_FREE_SHARD), dtype, buffer=nl.sbuf, name="out_sb")
    nisa.memset(out_sb, value=0.0)

    for t in nl.static_range(T):
        sbm.open_scope(name=f"token_{t}")

        aff_bcast = sbm.alloc_stack((_PMAX, _K), nl.float32, buffer=nl.sbuf, name=f"aff_bcast_t{t}")
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:_K], src=top8_vals[t:t + 1, 0:_K])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:_K],
                src=aff_bcast[0:1, 0:_K],
                shuffle_mask=[0] * 32,
            )

        output_temp = sbm.alloc_stack(
            (_PMAX, _H_FREE_SHARD), dtype, buffer=nl.sbuf, name=f"output_temp_t{t}"
        )

        for k in nl.static_range(_K):
            sbm.open_scope(name=f"expert_{k}_t{t}")
            expert_id = top8_idx.ap(pattern=[[_K, 1], [1, 1]], offset=t * _K + k)

            gate_w0 = sbm.alloc_stack(
                (_PMAX, _H_FREE_SHARD, _I0), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_w0_k{k}_t{t}"
            )
            gate_w1 = sbm.alloc_stack(
                (_PMAX, _H_FREE_SHARD, _I0), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_w1_k{k}_t{t}"
            )
            up_w0 = sbm.alloc_stack(
                (_PMAX, _H_FREE_SHARD, _I0), gate_up_w.dtype, buffer=nl.sbuf, name=f"up_w0_k{k}_t{t}"
            )
            up_w1 = sbm.alloc_stack(
                (_PMAX, _H_FREE_SHARD, _I0), gate_up_w.dtype, buffer=nl.sbuf, name=f"up_w1_k{k}_t{t}"
            )

            h_gate_base = h_hbm_start * _GU_FLAT
            nisa.dma_copy(
                dst=gate_w0[0:_PMAX, 0:_H_FREE_SHARD, 0:_I0],
                src=gate_up_w.ap(
                    pattern=[[_H_FREE_SHARD * _GU_FLAT, _PMAX], [_GU_FLAT, _H_FREE_SHARD], [1, _I0]],
                    offset=h_gate_base,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )
            nisa.dma_copy(
                dst=gate_w1[0:_PMAX, 0:_H_FREE_SHARD, 0:_I1],
                src=gate_up_w.ap(
                    pattern=[[_H_FREE_SHARD * _GU_FLAT, _PMAX], [_GU_FLAT, _H_FREE_SHARD], [1, _I1]],
                    offset=h_gate_base + _I0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )
            nisa.dma_copy(
                dst=up_w0[0:_PMAX, 0:_H_FREE_SHARD, 0:_I0],
                src=gate_up_w.ap(
                    pattern=[[_H_FREE_SHARD * _GU_FLAT, _PMAX], [_GU_FLAT, _H_FREE_SHARD], [1, _I0]],
                    offset=h_gate_base + _I,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )
            nisa.dma_copy(
                dst=up_w1[0:_PMAX, 0:_H_FREE_SHARD, 0:_I1],
                src=gate_up_w.ap(
                    pattern=[[_H_FREE_SHARD * _GU_FLAT, _PMAX], [_GU_FLAT, _H_FREE_SHARD], [1, _I1]],
                    offset=h_gate_base + _I + _I0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            gate_psum0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.psum)
            gate_psum1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.psum)
            up_psum0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.psum)
            up_psum1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.psum)

            nisa.nc_matmul(
                dst=gate_psum0[0:_I0, 0:1],
                stationary=gate_w0[0:_PMAX, 0, 0:_I0],
                moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + 0) * T + t, 1)],
                accumulate=False,
            )
            nisa.nc_matmul(
                dst=gate_psum1[0:_I1, 0:1],
                stationary=gate_w1[0:_PMAX, 0, 0:_I1],
                moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + 0) * T + t, 1)],
                accumulate=False,
            )
            nisa.nc_matmul(
                dst=up_psum0[0:_I0, 0:1],
                stationary=up_w0[0:_PMAX, 0, 0:_I0],
                moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + 0) * T + t, 1)],
                accumulate=False,
            )
            nisa.nc_matmul(
                dst=up_psum1[0:_I1, 0:1],
                stationary=up_w1[0:_PMAX, 0, 0:_I1],
                moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + 0) * T + t, 1)],
                accumulate=False,
            )
            for h1 in nl.static_range(1, _H_FREE_SHARD):
                nisa.nc_matmul(
                    dst=gate_psum0[0:_I0, 0:1],
                    stationary=gate_w0[0:_PMAX, h1, 0:_I0],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + h1) * T + t, 1)],
                    accumulate=True,
                )
                nisa.nc_matmul(
                    dst=gate_psum1[0:_I1, 0:1],
                    stationary=gate_w1[0:_PMAX, h1, 0:_I1],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + h1) * T + t, 1)],
                    accumulate=True,
                )
                nisa.nc_matmul(
                    dst=up_psum0[0:_I0, 0:1],
                    stationary=up_w0[0:_PMAX, h1, 0:_I0],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + h1) * T + t, 1)],
                    accumulate=True,
                )
                nisa.nc_matmul(
                    dst=up_psum1[0:_I1, 0:1],
                    stationary=up_w1[0:_PMAX, h1, 0:_I1],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + h1) * T + t, 1)],
                    accumulate=True,
                )

            gate_fp32 = sbm.alloc_stack((_PMAX, _I_TILES), nl.float32, buffer=nl.sbuf, name=f"gate_fp32_k{k}_t{t}")
            up_fp32 = sbm.alloc_stack((_PMAX, _I_TILES), nl.float32, buffer=nl.sbuf, name=f"up_fp32_k{k}_t{t}")
            nisa.memset(gate_fp32, value=0.0)
            nisa.memset(up_fp32, value=0.0)
            nisa.activation(gate_fp32[0:_I0, 0:1], op=nl.copy, data=gate_psum0[0:_I0, 0:1])
            nisa.activation(gate_fp32[0:_I1, 1:2], op=nl.copy, data=gate_psum1[0:_I1, 0:1])
            nisa.activation(up_fp32[0:_I0, 0:1], op=nl.copy, data=up_psum0[0:_I0, 0:1])
            nisa.activation(up_fp32[0:_I1, 1:2], op=nl.copy, data=up_psum1[0:_I1, 0:1])

            gate_up_recv = sbm.alloc_stack(
                (_PMAX, _I_TILES), nl.float32, buffer=nl.sbuf, name=f"gate_up_recv_k{k}_t{t}"
            )
            nisa.sendrecv(
                src=gate_fp32,
                dst=gate_up_recv,
                send_to_rank=peer_lnc,
                recv_from_rank=peer_lnc,
                pipe_id=t * _K * 2 + k * 2,
            )
            nisa.tensor_tensor(dst=gate_fp32, data1=gate_fp32, data2=gate_up_recv, op=nl.add)
            nisa.sendrecv(
                src=up_fp32,
                dst=gate_up_recv,
                send_to_rank=peer_lnc,
                recv_from_rank=peer_lnc,
                pipe_id=t * _K * 2 + k * 2 + 1,
            )
            nisa.tensor_tensor(dst=up_fp32, data1=up_fp32, data2=gate_up_recv, op=nl.add)

            nisa.activation(dst=gate_fp32, op=nl.silu, data=gate_fp32)

            inter_bf16 = sbm.alloc_stack((_PMAX, _I_TILES), dtype, buffer=nl.sbuf, name=f"inter_bf16_k{k}_t{t}")
            nisa.memset(inter_bf16, value=0.0)
            nisa.tensor_tensor(
                dst=inter_bf16[0:_I0, 0:1],
                data1=gate_fp32[0:_I0, 0:1],
                data2=up_fp32[0:_I0, 0:1],
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=inter_bf16[0:_I1, 1:2],
                data1=gate_fp32[0:_I1, 1:2],
                data2=up_fp32[0:_I1, 1:2],
                op=nl.multiply,
            )

            down_w0_tile = sbm.alloc_stack((_PMAX, _PMAX), down_w.dtype, buffer=nl.sbuf, name=f"down_w0_tile_k{k}_t{t}")
            down_w1_tile = sbm.alloc_stack((_PMAX, _PMAX), down_w.dtype, buffer=nl.sbuf, name=f"down_w1_tile_k{k}_t{t}")
            down_psum = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.psum)
            for h1_out in nl.static_range(_H_FREE_SHARD):
                h_offset = h_hbm_start + h1_out
                nisa.dma_copy(
                    dst=down_w0_tile[0:_I0, 0:_PMAX],
                    src=down_w.ap(
                        pattern=[[_H, _I0], [_H_FREE_SHARD, _PMAX]],
                        offset=h_offset,
                        scalar_offset=expert_id,
                        indirect_dim=0,
                    ),
                    dge_mode=0,
                )
                nisa.dma_copy(
                    dst=down_w1_tile[0:_I1, 0:_PMAX],
                    src=down_w.ap(
                        pattern=[[_H, _I1], [_H_FREE_SHARD, _PMAX]],
                        offset=_I0 * _H + h_offset,
                        scalar_offset=expert_id,
                        indirect_dim=0,
                    ),
                    dge_mode=0,
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_w0_tile[0:_I0, 0:_PMAX],
                    moving=inter_bf16[0:_I0, 0:1],
                    accumulate=False,
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_w1_tile[0:_I1, 0:_PMAX],
                    moving=inter_bf16[0:_I1, 1:2],
                    accumulate=True,
                )

            down_sb = sbm.alloc_stack((_PMAX, _H_FREE_SHARD), dtype, buffer=nl.sbuf, name=f"down_sb_k{k}_t{t}")
            nisa.activation(down_sb, op=nl.copy, data=down_psum[0:_PMAX, 0:_H_FREE_SHARD])
            nisa.tensor_scalar(
                dst=down_sb,
                data=down_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k],
            )

            if k == 0:
                nisa.tensor_copy(dst=output_temp[0:_PMAX, 0:_H_FREE_SHARD], src=down_sb)
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                    data1=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                    data2=down_sb,
                    op=nl.add,
                )

            sbm.close_scope()

        for h1 in nl.static_range(_H_FREE_SHARD):
            nisa.tensor_copy(
                dst=out_sb[0:_PMAX, t, h1:h1 + 1],
                src=output_temp[0:_PMAX, h1:h1 + 1],
            )

        sbm.close_scope()

    sbm.close_scope()
    return out_sb


def _experts_sbuf_in_sbuf_out(
    rmsnorm_normed_bf16,  # [PMAX, H_free*T] bf16 SBUF
    top8_idx,             # [T, K] uint32 SBUF
    top8_vals_bf16,       # [T, K] bf16 SBUF
    dtype,
    T,
    gate_up_w,            # [E, H, GU_FLAT=384] bf16 HBM
    down_w,               # [E, I=192, H] bf16 HBM
    sbm=None,
    debug=False,
):
    """
    Expert-MLP body. With LNC=2, each program computes one 96-wide
    intermediate shard and exchanges down-projection partials before BF16
    materialization, matching the reference compiler's sharded matmul.

    Op sequence mirrors NxDI HLO §5e–§5j of nxdi_moe.md:
      - bf16 abs → bf16 sum → bf16 clamp(bf16(eps)) → bf16 divide lowering,
        followed by a bf16 affinity store.
      - Gate/Up: full-H bf16 matmul, fp32 PSUM.
      - GLU: SiLU(gate_fp32 PSUM) → bf16, up_fp32 PSUM → bf16,
        then bf16 multiply → bf16.
      - Down: full-H bf16 matmul, fp32 PSUM, with the I=192 contraction
        sharded as 2 x 96 across LNC=2.
      - Per-expert affinity multiply: tensor_scalar(fp32 down * fp32 scalar →
        fp32 dst), where the scalar is the bf16-rounded normalized affinity
        widened to fp32 only to satisfy tensor_scalar's operand contract.
      - Stage 5: tensor-engine reduce over K=8 with fp32 inputs and bf16 output.
    """
    I        = _I
    I_tiles  = _I_TILES
    H_free   = _H_FREE
    i_lnc    = nl.program_id(0)  # LNC=2 shards the 192-wide intermediate as 2 x 96.
    peer_lnc = 1 - i_lnc

    sbm.open_scope(name="expert_loop_outer")

    # Match the reference profile: down-projection partials stay fp32 through
    # affinity scaling and the top-k weighted sum, then the K-reduce casts to
    # bf16 once.
    out_sb = sbm.alloc_stack((_PMAX, T, 4, 4), dtype, buffer=nl.sbuf, name="out_sb")
    if debug:
        debug_gate_f32 = sbm.alloc_heap(
            (_GU_P, T * _K), nl.float32, buffer=nl.sbuf, name="debug_gate_f32"
        )
        debug_up_f32 = sbm.alloc_heap(
            (_GU_P, T * _K), nl.float32, buffer=nl.sbuf, name="debug_up_f32"
        )
        debug_inter_bf16 = sbm.alloc_heap(
            (_GU_P, T * _K), dtype, buffer=nl.sbuf, name="debug_inter_bf16"
        )
        debug_down_f32 = sbm.alloc_heap(
            (_PMAX, H_free * T * _K), nl.float32, buffer=nl.sbuf, name="debug_down_f32"
        )
        debug_weighted_f32 = sbm.alloc_heap(
            (_PMAX, H_free * T, _K), nl.float32, buffer=nl.sbuf, name="debug_weighted_f32"
        )

    for t in nl.static_range(T):

        sbm.open_scope(name=f"token_{t}")

        # ------------------------------------------------------------------
        # Per-token weight banks: 4 experts at a time, full H (no shard).
        # ------------------------------------------------------------------
        gate_up_buf0 = sbm.alloc_stack((_PMAX, H_free, _GU_LNC_FLAT), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf0_t{t}")
        gate_up_buf1 = sbm.alloc_stack((_PMAX, H_free, _GU_LNC_FLAT), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf1_t{t}")
        gate_up_buf2 = sbm.alloc_stack((_PMAX, H_free, _GU_LNC_FLAT), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf2_t{t}")
        gate_up_buf3 = sbm.alloc_stack((_PMAX, H_free, _GU_LNC_FLAT), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf3_t{t}")
        gate_up_bufs = [gate_up_buf0, gate_up_buf1, gate_up_buf2, gate_up_buf3]

        down_full0_buf0 = sbm.alloc_stack((_DOWN_P, _H), down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf0_t{t}")
        down_full0_buf1 = sbm.alloc_stack((_DOWN_P, _H), down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf1_t{t}")
        down_full0_buf2 = sbm.alloc_stack((_DOWN_P, _H), down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf2_t{t}")
        down_full0_buf3 = sbm.alloc_stack((_DOWN_P, _H), down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf3_t{t}")
        down_full0_bufs = [down_full0_buf0, down_full0_buf1, down_full0_buf2, down_full0_buf3]

        down_full1_buf0 = sbm.alloc_stack((_DOWN_P, _H), down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf0_t{t}")
        down_full1_buf1 = sbm.alloc_stack((_DOWN_P, _H), down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf1_t{t}")
        down_full1_buf2 = sbm.alloc_stack((_DOWN_P, _H), down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf2_t{t}")
        down_full1_buf3 = sbm.alloc_stack((_DOWN_P, _H), down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf3_t{t}")
        down_full1_bufs = [down_full1_buf0, down_full1_buf1, down_full1_buf2, down_full1_buf3]

        # ------------------------------------------------------------------
        # L1 normalize affinities (HLO §5e):
        #   %400 = bf16 abs(%399)
        #   %407 = fp32 reduce(%400, ADD)
        #   %411 = fp32 clamp(eps=bf16(1e-12), sum, +inf)
        #   %415 = bf16 divide(%399, broadcast(%411))
        # ------------------------------------------------------------------
        abs_topk = sbm.alloc_stack((T, _K), dtype, buffer=nl.sbuf, name=f"abs_topk_t{t}")
        nisa.activation(abs_topk[0:T, 0:_K], op=nl.abs, data=top8_vals_bf16[0:T, 0:_K])

        sum_topk = sbm.alloc_stack((T, 1), dtype, buffer=nl.sbuf, name=f"sum_topk_t{t}")
        nisa.tensor_reduce(sum_topk[0:T, 0:1], nl.add, abs_topk[0:T, 0:_K], axis=1)

        sum_topk_clamped = sbm.alloc_stack((T, 1), dtype, buffer=nl.sbuf, name=f"sum_topk_clamped_t{t}")
        nisa.tensor_scalar(
            sum_topk_clamped[0:T, 0:1],
            data=sum_topk[0:T, 0:1],
            op0=nl.maximum,
            operand0=_NORMALIZE_EPS_BF16,
        )

        # NKI requires activation scales to be fp32. The sum/clamp are still
        # BF16-rounded to match the HLO divide denominator.
        inv_sum_topk = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name=f"inv_sum_topk_t{t}")
        nisa.activation(inv_sum_topk[0:T, 0:1], op=nl.reciprocal, data=sum_topk_clamped[0:T, 0:1])

        norm_weights = sbm.alloc_stack((T, _K), dtype, buffer=nl.sbuf, name=f"norm_weights_t{t}")
        nisa.activation(
            norm_weights[0:T, 0:_K],
            op=nl.copy,
            data=top8_vals_bf16[0:T, 0:_K],
            scale=inv_sum_topk[0:T, 0:1],
        )

        # NxDI rounds normalized affinities to bf16, then uses a tensor-tensor
        # multiply with the fp32 down projection.
        aff_bcast = sbm.alloc_stack((_PMAX, _K), dtype, buffer=nl.sbuf, name=f"aff_bcast_t{t}")
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:_K], src=norm_weights[t:t + 1, 0:_K])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:_K],
                src=aff_bcast[0:1, 0:_K],
                shuffle_mask=[0] * 32,
            )
        aff_bcast_h = sbm.alloc_stack((_PMAX, H_free, _K), dtype, buffer=nl.sbuf, name=f"aff_bcast_h_t{t}")
        for h1 in nl.static_range(H_free):
            nisa.tensor_copy(
                dst=aff_bcast_h[0:_PMAX, h1:h1 + 1, 0:_K],
                src=aff_bcast[0:_PMAX, 0:_K].reshape((_PMAX, 1, _K)),
            )
        # PSUM allocations sized for one wave at full H.
        gate_up_psum = nl.ndarray((_GU_P, _K_WAVE * 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
        down_psum = nl.ndarray((_PMAX, _K * H_free), dtype=nl.float32, buffer=nl.psum)
        down_results_f32 = sbm.alloc_stack(
            (_PMAX, H_free, _K), nl.float32, buffer=nl.sbuf, name=f"down_results_f32_t{t}"
        )

        # ==================================================================
        # WAVE 0: experts 0-3
        # ==================================================================
        # Phase 1a: load experts 0-3 (full H per expert)
        for k in nl.static_range(_K_WAVE):
            expert_id = top8_idx.ap(pattern=[[_K, 1], [1, 1]], offset=t * _K + k)

            nisa.dma_copy(
                dst=gate_up_bufs[k][0:_PMAX, 0:H_free, 0:_GU_P],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_P]],
                    offset=i_lnc * _GU_P,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=gate_up_bufs[k][0:_PMAX, 0:H_free, _GU_P:_GU_LNC_FLAT],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_P]],
                    offset=I + i_lnc * _GU_P,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full0_bufs[k],
                src=down_w.ap(
                    pattern=[[_H, _DOWN_P], [1, _H]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full1_bufs[k],
                src=down_w.ap(
                    pattern=[[_H, _DOWN_P], [1, _H]],
                    offset=_DOWN_P * _H,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # Phase 2a: gate/up matmuls for the whole wave. Keeping the four
        # experts packed preserves the compiler schedule and SBUF reuse.
        for k in nl.static_range(_K_WAVE):
            gu_base = k * 2 * I_tiles

            nisa.nc_matmul(
                dst=gate_up_psum[0:_GU_P, gu_base + i_lnc:gu_base + i_lnc + 1],
                stationary=gate_up_bufs[k][0:_PMAX, 0, nl.ds(0, _GU_P)],
                moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(0, T)],
                accumulate=False,
            )
            for h1 in nl.affine_range(1, H_free):
                nisa.nc_matmul(
                    dst=gate_up_psum[0:_GU_P, gu_base + i_lnc:gu_base + i_lnc + 1],
                    stationary=gate_up_bufs[k][0:_PMAX, h1, nl.ds(0, _GU_P)],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    accumulate=True,
                )

            nisa.nc_matmul(
                dst=gate_up_psum[0:_GU_P, gu_base + I_tiles + i_lnc:gu_base + I_tiles + i_lnc + 1],
                stationary=gate_up_bufs[k][0:_PMAX, 0, nl.ds(_GU_P, _GU_P)],
                moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(0, T)],
                accumulate=False,
            )
            for h1 in nl.affine_range(1, H_free):
                nisa.nc_matmul(
                    dst=gate_up_psum[0:_GU_P, gu_base + I_tiles + i_lnc:gu_base + I_tiles + i_lnc + 1],
                    stationary=gate_up_bufs[k][0:_PMAX, h1, nl.ds(_GU_P, _GU_P)],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    accumulate=True,
                )

        # Phase 2b: SiLU + multiply, down, and LNC exchange per expert.
        for k in nl.static_range(_K_WAVE):
            d_base  = k * H_free
            gu_base = k * 2 * I_tiles

            sbm.open_scope(name=f"w0_expert_{k}_t{t}")

            if debug:
                nisa.activation(
                    debug_gate_f32[0:_GU_P, t * _K + k:t * _K + k + 1],
                    op=nl.copy,
                    data=gate_up_psum[0:_GU_P, gu_base + i_lnc:gu_base + i_lnc + 1],
                )
                nisa.activation(
                    debug_up_f32[0:_GU_P, t * _K + k:t * _K + k + 1],
                    op=nl.copy,
                    data=gate_up_psum[0:_GU_P, gu_base + I_tiles + i_lnc:gu_base + I_tiles + i_lnc + 1],
                )

            silu_res_bf16 = sbm.alloc_stack((_GU_P, 1), dtype, buffer=nl.sbuf, name=f"silu_res_bf16_w0k{k}_t{t}")
            nisa.activation(
                silu_res_bf16,
                op=nl.silu,
                data=gate_up_psum[0:_GU_P, gu_base + i_lnc:gu_base + i_lnc + 1],
            )
            inter_bf16 = sbm.alloc_stack((_GU_P, 1), dtype, buffer=nl.sbuf, name=f"inter_bf16_w0k{k}_t{t}")
            nisa.tensor_tensor(
                inter_bf16,
                silu_res_bf16,
                gate_up_psum[0:_GU_P, gu_base + I_tiles + i_lnc:gu_base + I_tiles + i_lnc + 1],
                nl.multiply,
            )
            if debug:
                nisa.tensor_copy(
                    dst=debug_inter_bf16[0:_GU_P, t * _K + k:t * _K + k + 1],
                    src=inter_bf16,
                )

            # Down matmul: each LNC computes one 96-wide I shard. The local
            # partial stays fp32 through weighting; LNC reduction happens after
            # the top-k accumulation.
            for h1_out in nl.affine_range(H_free):
                if i_lnc == 0:
                    nisa.nc_matmul(
                        dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                        stationary=down_full0_bufs[k][0:_DOWN_P, nl.ds(h1_out * _PMAX, _PMAX)],
                        moving=inter_bf16[0:_DOWN_P, 0:1],
                    )
                else:
                    nisa.nc_matmul(
                        dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                        stationary=down_full1_bufs[k][0:_DOWN_P, nl.ds(h1_out * _PMAX, _PMAX)],
                        moving=inter_bf16[0:_DOWN_P, 0:1],
                    )

            down_result_f32 = sbm.alloc_stack((_PMAX, H_free), nl.float32, buffer=nl.sbuf, name=f"down_result_f32_w0k{k}_t{t}")
            nisa.activation(
                down_result_f32,
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + H_free],
            )
            if debug:
                nisa.tensor_copy(
                    dst=debug_down_f32[0:_PMAX, nl.ds((t * _K + k) * H_free, H_free)],
                    src=down_result_f32,
                )
            down_peer_f32 = sbm.alloc_stack((_PMAX, H_free), nl.float32, buffer=nl.sbuf, name=f"down_peer_f32_w0k{k}_t{t}")
            nisa.sendrecv(
                src=down_result_f32,
                dst=down_peer_f32[0:_PMAX, 0:H_free],
                send_to_rank=peer_lnc,
                recv_from_rank=peer_lnc,
                pipe_id=t * _K + k,
            )
            if i_lnc == 0:
                nisa.tensor_tensor(
                    dst=down_result_f32,
                    data1=down_result_f32,
                    data2=down_peer_f32[0:_PMAX, 0:H_free],
                    op=nl.add,
                )
            else:
                nisa.tensor_tensor(
                    dst=down_result_f32,
                    data1=down_peer_f32[0:_PMAX, 0:H_free],
                    data2=down_result_f32,
                    op=nl.add,
                )

            nisa.tensor_copy(
                dst=down_results_f32[0:_PMAX, 0:H_free, k:k + 1],
                src=down_result_f32.reshape((_PMAX, H_free, 1)),
            )

            sbm.close_scope()  # w0_expert_k

        # ==================================================================
        # WAVE 1: experts 4-7 (reuse buffer banks)
        # ==================================================================
        for k in nl.static_range(_K_WAVE):
            kk = k + 4
            expert_id = top8_idx.ap(pattern=[[_K, 1], [1, 1]], offset=t * _K + kk)

            nisa.dma_copy(
                dst=gate_up_bufs[k][0:_PMAX, 0:H_free, 0:_GU_P],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_P]],
                    offset=i_lnc * _GU_P,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=gate_up_bufs[k][0:_PMAX, 0:H_free, _GU_P:_GU_LNC_FLAT],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_P]],
                    offset=I + i_lnc * _GU_P,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full0_bufs[k],
                src=down_w.ap(
                    pattern=[[_H, _DOWN_P], [1, _H]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full1_bufs[k],
                src=down_w.ap(
                    pattern=[[_H, _DOWN_P], [1, _H]],
                    offset=_DOWN_P * _H,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        for k in nl.static_range(_K_WAVE):
            gu_base = k * 2 * I_tiles

            nisa.nc_matmul(
                dst=gate_up_psum[0:_GU_P, gu_base + i_lnc:gu_base + i_lnc + 1],
                stationary=gate_up_bufs[k][0:_PMAX, 0, nl.ds(0, _GU_P)],
                moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(0, T)],
                accumulate=False,
            )
            for h1 in nl.affine_range(1, H_free):
                nisa.nc_matmul(
                    dst=gate_up_psum[0:_GU_P, gu_base + i_lnc:gu_base + i_lnc + 1],
                    stationary=gate_up_bufs[k][0:_PMAX, h1, nl.ds(0, _GU_P)],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    accumulate=True,
                )

            nisa.nc_matmul(
                dst=gate_up_psum[0:_GU_P, gu_base + I_tiles + i_lnc:gu_base + I_tiles + i_lnc + 1],
                stationary=gate_up_bufs[k][0:_PMAX, 0, nl.ds(_GU_P, _GU_P)],
                moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(0, T)],
                accumulate=False,
            )
            for h1 in nl.affine_range(1, H_free):
                nisa.nc_matmul(
                    dst=gate_up_psum[0:_GU_P, gu_base + I_tiles + i_lnc:gu_base + I_tiles + i_lnc + 1],
                    stationary=gate_up_bufs[k][0:_PMAX, h1, nl.ds(_GU_P, _GU_P)],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    accumulate=True,
                )

        for k in nl.static_range(_K_WAVE):
            kk = k + 4
            d_base  = kk * H_free
            gu_base = k * 2 * I_tiles

            sbm.open_scope(name=f"w1_expert_{k}_t{t}")

            if debug:
                nisa.activation(
                    debug_gate_f32[0:_GU_P, t * _K + kk:t * _K + kk + 1],
                    op=nl.copy,
                    data=gate_up_psum[0:_GU_P, gu_base + i_lnc:gu_base + i_lnc + 1],
                )
                nisa.activation(
                    debug_up_f32[0:_GU_P, t * _K + kk:t * _K + kk + 1],
                    op=nl.copy,
                    data=gate_up_psum[0:_GU_P, gu_base + I_tiles + i_lnc:gu_base + I_tiles + i_lnc + 1],
                )

            silu_res_bf16 = sbm.alloc_stack((_GU_P, 1), dtype, buffer=nl.sbuf, name=f"silu_res_bf16_w1k{k}_t{t}")
            nisa.activation(
                silu_res_bf16,
                op=nl.silu,
                data=gate_up_psum[0:_GU_P, gu_base + i_lnc:gu_base + i_lnc + 1],
            )
            inter_bf16 = sbm.alloc_stack((_GU_P, 1), dtype, buffer=nl.sbuf, name=f"inter_bf16_w1k{k}_t{t}")
            nisa.tensor_tensor(
                inter_bf16,
                silu_res_bf16,
                gate_up_psum[0:_GU_P, gu_base + I_tiles + i_lnc:gu_base + I_tiles + i_lnc + 1],
                nl.multiply,
            )
            if debug:
                nisa.tensor_copy(
                    dst=debug_inter_bf16[0:_GU_P, t * _K + kk:t * _K + kk + 1],
                    src=inter_bf16,
                )

            for h1_out in nl.affine_range(H_free):
                if i_lnc == 0:
                    nisa.nc_matmul(
                        dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                        stationary=down_full0_bufs[k][0:_DOWN_P, nl.ds(h1_out * _PMAX, _PMAX)],
                        moving=inter_bf16[0:_DOWN_P, 0:1],
                    )
                else:
                    nisa.nc_matmul(
                        dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                        stationary=down_full1_bufs[k][0:_DOWN_P, nl.ds(h1_out * _PMAX, _PMAX)],
                        moving=inter_bf16[0:_DOWN_P, 0:1],
                    )

            down_result_f32 = sbm.alloc_stack((_PMAX, H_free), nl.float32, buffer=nl.sbuf, name=f"down_result_f32_w1k{k}_t{t}")
            nisa.activation(
                down_result_f32,
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + H_free],
            )
            if debug:
                nisa.tensor_copy(
                    dst=debug_down_f32[0:_PMAX, nl.ds((t * _K + kk) * H_free, H_free)],
                    src=down_result_f32,
                )
            down_peer_f32 = sbm.alloc_stack((_PMAX, H_free), nl.float32, buffer=nl.sbuf, name=f"down_peer_f32_w1k{k}_t{t}")
            nisa.sendrecv(
                src=down_result_f32,
                dst=down_peer_f32[0:_PMAX, 0:H_free],
                send_to_rank=peer_lnc,
                recv_from_rank=peer_lnc,
                pipe_id=t * _K + kk,
            )
            if i_lnc == 0:
                nisa.tensor_tensor(
                    dst=down_result_f32,
                    data1=down_result_f32,
                    data2=down_peer_f32[0:_PMAX, 0:H_free],
                    op=nl.add,
                )
            else:
                nisa.tensor_tensor(
                    dst=down_result_f32,
                    data1=down_peer_f32[0:_PMAX, 0:H_free],
                    data2=down_result_f32,
                    op=nl.add,
                )

            nisa.tensor_copy(
                dst=down_results_f32[0:_PMAX, 0:H_free, kk:kk + 1],
                src=down_result_f32.reshape((_PMAX, H_free, 1)),
            )

            sbm.close_scope()  # w1_expert_k

        weighted_down_f32 = sbm.alloc_stack(
            (_PMAX, H_free, _K), nl.float32, buffer=nl.sbuf, name=f"weighted_down_f32_t{t}"
        )
        nisa.tensor_tensor(
            dst=weighted_down_f32[0:_PMAX, 0:H_free, 0:_K],
            data1=down_results_f32[0:_PMAX, 0:H_free, 0:_K],
            data2=aff_bcast_h[0:_PMAX, 0:H_free, 0:_K],
            op=nl.multiply,
        )
        nisa.tensor_reduce(
            dst=out_sb[0:_PMAX, t, 0:4, 0:4],
            op=nl.add,
            data=weighted_down_f32[0:_PMAX, 0:H_free, 0:_K].reshape((_PMAX, 4, 4, _K)),
            axis=3,
        )
        if debug:
            nisa.tensor_copy(
                dst=debug_weighted_f32[0:_PMAX, nl.ds(t * H_free, H_free), 0:_K],
                src=weighted_down_f32[0:_PMAX, 0:H_free, 0:_K],
            )

        sbm.close_scope()  # token_t

    sbm.close_scope()  # expert_loop_outer

    if debug:
        return out_sb, debug_gate_f32, debug_up_f32, debug_inter_bf16, debug_down_f32, debug_weighted_f32
    return out_sb  # [PMAX, T, 4, 4] bf16 — full H, column-major


@nki.jit
def experts_hbm(inp_normed, top8_idx_in, top8_vals_in, gate_up_w, down_w):
    """
    HBM wrapper for the unsharded Expert MLP stage.

    inp_normed:   [T, H=2048] bf16 HBM — RMSNorm-normalized input
    top8_idx_in:  [T, K=8]    uint32 HBM — top-K expert indices
    top8_vals_bf16: [T, K=8]  bf16 HBM — top-K softmax weights
    gate_up_w:    [E=128, H=2048, GU_FLAT=384] bf16 HBM
    down_w:       [E=128, I=192, H=2048] bf16 HBM
    Returns output [T, H=2048] bf16 HBM. Each program writes the full H output.
    With LNC=2 launch grids both cores produce the same value at the same HBM
    address (idempotent); with LNC=1 only one program runs.
    """
    sbm    = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    T      = inp_normed.shape[0]
    H_free = _H_FREE

    # --- Load inp_normed into SBUF as [PMAX, H_free*T] col-major ---
    inp_2d          = inp_normed.reshape((T, _H))
    inp_2d_hbm_flat = inp_2d.reshape((H_free * T, _PMAX))
    sbm.open_scope(name="inp_load")
    inp_flat_sb = sbm.alloc_stack((H_free * T, _PMAX), inp_normed.dtype, buffer=nl.sbuf, name="inp_flat_sb")
    nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_flat, dge_mode=3)
    inp_trans_psum = nl.ndarray((_PMAX, H_free * T), dtype=inp_normed.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)
    rmsnorm_normed_bf16 = sbm.alloc_heap((_PMAX, H_free * T), inp_normed.dtype, buffer=nl.sbuf, name="rmsnorm_normed_bf16")
    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=inp_trans_psum[...])

    # --- Load top8_idx and top8_vals into SBUF ---
    top8_idx = sbm.alloc_heap((T, _K), nl.uint32, buffer=nl.sbuf, name="top8_idx")
    nisa.dma_copy(dst=top8_idx[0:T, 0:_K], src=top8_idx_in[0:T, 0:_K])

    top8_vals_bf16 = sbm.alloc_heap((T, _K), inp_normed.dtype, buffer=nl.sbuf, name="top8_vals_bf16")
    nisa.dma_copy(dst=top8_vals_bf16[0:T, 0:_K], src=top8_vals_in[0:T, 0:_K])

    out_sb = _experts_sbuf_in_sbuf_out(
        rmsnorm_normed_bf16, top8_idx, top8_vals_bf16,
        inp_normed.dtype, T, gate_up_w, down_w, sbm=sbm,
    )

    # --- Store full H output to HBM: [PMAX, H_free*T] col-major → [T, H] row-major ---
    output = nl.ndarray((T, _H), dtype=inp_normed.dtype, buffer=nl.shared_hbm)
    sbm.open_scope(name="store_hbm")
    output_tiled = output.reshape((T, H_free, _PMAX))
    for t in nl.static_range(T):
        tp_psum = nl.ndarray((H_free, _PMAX), dtype=inp_normed.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=tp_psum[0:H_free, 0:_PMAX],
            data=out_sb[0:_PMAX, t, 0:4, 0:4].reshape((_PMAX, H_free)),
        )
        out_tile_sb = sbm.alloc_stack((H_free, _PMAX), inp_normed.dtype, buffer=nl.sbuf, name=f"out_tile_sb_t{t}")
        nisa.activation(
            dst=out_tile_sb[0:H_free, 0:_PMAX],
            op=nl.copy,
            data=tp_psum[0:H_free, 0:_PMAX],
        )
        nisa.dma_copy(
            dst=output_tiled[t, 0:H_free, 0:_PMAX],
            src=out_tile_sb[0:H_free, 0:_PMAX],
        )
    sbm.close_scope()  # store_hbm

    # Pop heaps in reverse allocation order
    sbm.pop_heap()  # top8_vals_bf16
    sbm.pop_heap()  # top8_idx
    sbm.pop_heap()  # rmsnorm_normed_bf16

    sbm.close_scope()  # inp_load

    return output


@nki.jit
def experts_debug_hbm(inp_normed, top8_idx_in, top8_vals_in, gate_up_w, down_w):
    sbm    = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    T      = inp_normed.shape[0]
    H_free = _H_FREE
    i_lnc  = nl.program_id(0)

    inp_2d          = inp_normed.reshape((T, _H))
    inp_2d_hbm_flat = inp_2d.reshape((H_free * T, _PMAX))
    sbm.open_scope(name="inp_load")
    inp_flat_sb = sbm.alloc_stack((H_free * T, _PMAX), inp_normed.dtype, buffer=nl.sbuf, name="inp_flat_sb")
    nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_flat, dge_mode=3)
    inp_trans_psum = nl.ndarray((_PMAX, H_free * T), dtype=inp_normed.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)
    rmsnorm_normed_bf16 = sbm.alloc_heap((_PMAX, H_free * T), inp_normed.dtype, buffer=nl.sbuf, name="rmsnorm_normed_bf16")
    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=inp_trans_psum[...])

    top8_idx = sbm.alloc_heap((T, _K), nl.uint32, buffer=nl.sbuf, name="top8_idx")
    nisa.dma_copy(dst=top8_idx[0:T, 0:_K], src=top8_idx_in[0:T, 0:_K])

    top8_vals_bf16 = sbm.alloc_heap((T, _K), inp_normed.dtype, buffer=nl.sbuf, name="top8_vals_bf16")
    nisa.dma_copy(dst=top8_vals_bf16[0:T, 0:_K], src=top8_vals_in[0:T, 0:_K])

    out_sb, debug_gate_f32, debug_up_f32, debug_inter_bf16, debug_down_f32, debug_weighted_f32 = _experts_sbuf_in_sbuf_out(
        rmsnorm_normed_bf16, top8_idx, top8_vals_bf16,
        inp_normed.dtype, T, gate_up_w, down_w, sbm=sbm, debug=True,
    )

    output = nl.ndarray((T, _H), dtype=inp_normed.dtype, buffer=nl.shared_hbm)
    debug_gate = nl.ndarray((_I, T * _K), dtype=nl.float32, buffer=nl.shared_hbm)
    debug_up = nl.ndarray((_I, T * _K), dtype=nl.float32, buffer=nl.shared_hbm)
    debug_inter = nl.ndarray((_I, T * _K), dtype=inp_normed.dtype, buffer=nl.shared_hbm)
    debug_down = nl.ndarray((_PMAX, H_free * T * _K), dtype=nl.float32, buffer=nl.shared_hbm)
    debug_weighted = nl.ndarray((_PMAX, H_free * T * _K), dtype=nl.float32, buffer=nl.shared_hbm)

    sbm.open_scope(name="store_hbm")
    output_tiled = output.reshape((T, H_free, _PMAX))
    for t in nl.static_range(T):
        tp_psum = nl.ndarray((H_free, _PMAX), dtype=inp_normed.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=tp_psum[0:H_free, 0:_PMAX],
            data=out_sb[0:_PMAX, t, 0:4, 0:4].reshape((_PMAX, H_free)),
        )
        out_tile_sb = sbm.alloc_stack((H_free, _PMAX), inp_normed.dtype, buffer=nl.sbuf, name=f"out_tile_sb_t{t}")
        nisa.activation(
            dst=out_tile_sb[0:H_free, 0:_PMAX],
            op=nl.copy,
            data=tp_psum[0:H_free, 0:_PMAX],
        )
        nisa.dma_copy(
            dst=output_tiled[t, 0:H_free, 0:_PMAX],
            src=out_tile_sb[0:H_free, 0:_PMAX],
        )

    for t in nl.static_range(T):
        for k in nl.static_range(_K):
            nisa.dma_copy(
                dst=debug_gate[nl.ds(i_lnc * _GU_P, _GU_P), t * _K + k:t * _K + k + 1],
                src=debug_gate_f32[0:_GU_P, t * _K + k:t * _K + k + 1],
            )
            nisa.dma_copy(
                dst=debug_up[nl.ds(i_lnc * _GU_P, _GU_P), t * _K + k:t * _K + k + 1],
                src=debug_up_f32[0:_GU_P, t * _K + k:t * _K + k + 1],
            )
            nisa.dma_copy(
                dst=debug_inter[nl.ds(i_lnc * _GU_P, _GU_P), t * _K + k:t * _K + k + 1],
                src=debug_inter_bf16[0:_GU_P, t * _K + k:t * _K + k + 1],
            )
            nisa.dma_copy(
                dst=debug_down[0:_PMAX, nl.ds((t * _K + k) * H_free, H_free)],
                src=debug_down_f32[0:_PMAX, nl.ds((t * _K + k) * H_free, H_free)],
            )
            nisa.dma_copy(
                dst=debug_weighted[0:_PMAX, nl.ds((t * _K + k) * H_free, H_free)],
                src=debug_weighted_f32[0:_PMAX, nl.ds(t * H_free, H_free), k],
            )
    sbm.close_scope()  # store_hbm

    sbm.pop_heap()  # debug_weighted_f32
    sbm.pop_heap()  # debug_down_f32
    sbm.pop_heap()  # debug_inter_bf16
    sbm.pop_heap()  # debug_up_f32
    sbm.pop_heap()  # debug_gate_f32
    sbm.pop_heap()  # top8_vals_bf16
    sbm.pop_heap()  # top8_idx
    sbm.pop_heap()  # rmsnorm_normed_bf16

    sbm.close_scope()  # inp_load

    return output, debug_gate, debug_up, debug_inter, debug_down, debug_weighted
