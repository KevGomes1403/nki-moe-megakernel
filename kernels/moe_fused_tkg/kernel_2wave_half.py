"""
Qwen3 MoE TKG kernel with H_HALF gate_up buffer halving (qwen3_moe_2wave_half).

Fix for NCC_IGCA044 (SBUF overflow): gate_up_buf is halved from
(_PMAX, H_FREE=16, GU_FLAT=384) to (_PMAX, H_HALF=8, GU_FLAT=384).
The H_FREE=16 h-tiles are processed in 2 sub-passes of H_HALF=8, reusing
the same half-sized buffers. This saves 28,672 bytes/partition.

Invoked as: qwen3_moe_2wave_half[2](inp, gamma, router_w, gate_up_w, down_w)
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.tensor_view import TensorView

# ---------------------------------------------------------------------------
# Constants (Qwen3-30B-A3B at TP=4, LNC=2)
# ---------------------------------------------------------------------------
_PMAX        = 128
H            = 2048
H_FREE       = H // _PMAX          # = 16
H_HALF       = H_FREE // 2         # = 8  (NEW: halved gate_up_buf h-dim)
N_PRGS       = 2
H_FREE_SHARD = H_FREE // N_PRGS    # = 8
H_SHARD      = H_FREE_SHARD * _PMAX  # = 1024
E            = 128
K_EXPERTS    = 8
K_WAVE       = 4
I_DIM        = 192
I0           = 128
I1           = 64
I_TILES      = 2
GU_FLAT      = 2 * I_DIM           # = 384
ROUTER_BATCH = 16
EPS          = 1e-6


@nki.jit
def qwen3_moe_2wave_half(inp, gamma, router_w, gate_up_w, down_w):
    """
    Qwen3 MoE TKG kernel with H_HALF gate_up buffer halving.

    Args:
        inp:       [B, H=2048] bf16
        gamma:     [1, H=2048] bf16
        router_w:  [H=2048, E=128] bf16
        gate_up_w: [E=128, H=2048, GU_FLAT=384] bf16
        down_w:    [E=128, I_DIM=192, H=2048] bf16

    Returns:
        output: [B, H=2048] bf16  (each LNC core writes its shard prg_id*H_SHARD)

    Invoked as: qwen3_moe_2wave_half[2](inp, gamma, router_w, gate_up_w, down_w)
    """
    B = inp.shape[0]
    T = B
    inp_dtype = inp.dtype
    prg_id = nl.program_id(axis=0)

    H_dim = _PMAX * H_FREE
    K = K_EXPERTS
    I = I_DIM
    H_shard = H_SHARD
    H_free_shard = H_FREE_SHARD

    # -----------------------------------------------------------------------
    # Stage 1: RMSNorm
    # -----------------------------------------------------------------------
    inp_2d = inp.reshape((T, H_dim))
    inp_2d_hbm_reshaped = inp_2d.reshape((H_FREE * T, _PMAX))
    inp_flat_sb = nl.ndarray((H_FREE * T, _PMAX), dtype=inp_dtype, buffer=nl.sbuf, name="inp_flat_sb")
    nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_reshaped, dge_mode=3)

    inp_trans_psum = nl.ndarray((_PMAX, H_FREE * T), dtype=inp_dtype, buffer=nl.psum, name="inp_trans_psum")
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)

    rmsnorm_out = nl.ndarray((_PMAX, H_FREE * T), dtype=inp_dtype, buffer=nl.sbuf, name="rmsnorm_out")
    nisa.activation(rmsnorm_out[...], op=nl.copy, data=inp_trans_psum[...])

    gamma_1d = gamma.reshape((H_dim,))
    gamma_1d_hbm_reshaped = gamma_1d.reshape((H_FREE, _PMAX))
    gamma_flat_sb = nl.ndarray((H_FREE, _PMAX), dtype=gamma.dtype, buffer=nl.sbuf, name="gamma_flat_sb")
    nisa.dma_copy(dst=gamma_flat_sb, src=gamma_1d_hbm_reshaped, dge_mode=3)
    gamma_trans_psum = nl.ndarray((_PMAX, H_FREE), dtype=gamma.dtype, buffer=nl.psum, name="gamma_trans_psum")
    nisa.nc_transpose(dst=gamma_trans_psum, data=gamma_flat_sb)
    gamma_sb = nl.ndarray((_PMAX, H_FREE), dtype=gamma.dtype, buffer=nl.sbuf, name="gamma_sb")
    nisa.activation(gamma_sb[...], op=nl.copy, data=gamma_trans_psum[...])

    rmsnorm_sq = nl.ndarray((_PMAX, H_FREE * T), dtype=nl.float32, buffer=nl.sbuf, name="rmsnorm_sq")
    nisa.activation(rmsnorm_sq[...], op=nl.square, data=rmsnorm_out[...])

    rmsnorm_reduced = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf, name="rmsnorm_reduced")
    nisa.tensor_reduce(rmsnorm_reduced[0:_PMAX, 0:T], nl.add, rmsnorm_sq[0:_PMAX, 0:H_FREE * T], axis=1)

    gamma_mult = nl.ndarray((_PMAX, H_FREE * T), dtype=nl.float32, buffer=nl.sbuf, name="gamma_mult")
    nisa.tensor_tensor(gamma_mult[...], rmsnorm_out[...], gamma_sb[...], nl.multiply)

    sum_reduced_sb = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf, name="sum_reduced_sb")
    nisa.tensor_partition_reduce(dst=sum_reduced_sb[0:1, 0:T], data=rmsnorm_reduced[0:_PMAX, 0:T], op=nl.add)

    norm_sum_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf, name="norm_sum_sb")
    nisa.tensor_copy(dst=norm_sum_sb[0:1, 0:T], src=sum_reduced_sb[0:1, 0:T])
    for g in nl.static_range(4):
        nisa.nc_stream_shuffle(
            dst=norm_sum_sb[nl.ds(g * 32, 32), 0:T],
            src=norm_sum_sb[0:1, 0:T],
            shuffle_mask=[0] * 32,
        )

    eps_sb = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="eps_sb")
    nisa.memset(eps_sb, value=EPS)
    norm_factor_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf, name="norm_factor_sb")
    nisa.activation(
        norm_factor_sb[0:_PMAX, 0:T],
        op=nl.rsqrt,
        data=norm_sum_sb[0:_PMAX, 0:T],
        scale=1.0 / H_dim,
        bias=eps_sb[0:_PMAX, :],
    )

    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=2).broadcast(dim=2, size=H_FREE)
    rmsnorm_normed = nl.ndarray((_PMAX, H_FREE * T), dtype=nl.float32, buffer=nl.sbuf, name="rmsnorm_normed")
    nisa.tensor_tensor(rmsnorm_normed[...], gamma_mult[...], norm_factor_bcast.get_view(), nl.multiply)

    rmsnorm_normed_bf16 = nl.ndarray((_PMAX, H_FREE * T), dtype=inp_dtype, buffer=nl.sbuf, name="rmsnorm_normed_bf16")
    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=rmsnorm_normed[...])

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum, name="logits_psum")
    router_w_wide_sb = nl.ndarray((_PMAX, ROUTER_BATCH, E), dtype=inp_dtype, buffer=nl.sbuf, name="router_w_wide_sb")

    for h_chunk in nl.affine_range(H_FREE // ROUTER_BATCH):
        nisa.dma_copy(
            dst=router_w_wide_sb,
            src=router_w.ap(
                pattern=[[E, _PMAX], [_PMAX * E, ROUTER_BATCH], [1, E]],
                offset=h_chunk * ROUTER_BATCH * _PMAX * E,
            ),
            dge_mode=3,
        )
        for h_sub in nl.static_range(ROUTER_BATCH):
            h1 = h_chunk * ROUTER_BATCH + h_sub
            nisa.nc_matmul(
                dst=logits_psum[0:T, 0:E],
                stationary=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                moving=router_w_wide_sb[0:_PMAX, h_sub, 0:E],
            )

    logits_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf, name="logits_sb")
    nisa.activation(logits_sb[0:T, 0:E], op=nl.copy, data=logits_psum[0:T, 0:E])

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8)
    # -----------------------------------------------------------------------
    max_logit = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf, name="max_logit")
    nisa.tensor_reduce(max_logit[0:T, 0:1], nl.maximum, logits_sb[0:T, 0:E], axis=1)

    centered = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf, name="centered")
    nisa.tensor_scalar(
        centered[0:T, 0:E], data=logits_sb[0:T, 0:E],
        op0=nl.subtract, operand0=max_logit[0:T, 0:1],
    )

    exp_vals = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf, name="exp_vals")
    nisa.activation(exp_vals[0:T, 0:E], op=nl.exp, data=centered[0:T, 0:E])

    sum_exp = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf, name="sum_exp")
    nisa.tensor_reduce(sum_exp[0:T, 0:1], nl.add, exp_vals[0:T, 0:E], axis=1)

    inv_sum_exp = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf, name="inv_sum_exp")
    nisa.activation(inv_sum_exp[0:T, 0:1], op=nl.reciprocal, data=sum_exp[0:T, 0:1])

    probs = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf, name="probs")
    nisa.tensor_scalar(
        probs[0:T, 0:E], data=exp_vals[0:T, 0:E],
        op0=nl.multiply, operand0=inv_sum_exp[0:T, 0:1],
    )

    top8_vals = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf, name="top8_vals")
    nisa.max8(dst=top8_vals[0:T, 0:K], src=probs[0:T, 0:E])

    top8_idx = nl.ndarray((T, K), dtype=nl.uint32, buffer=nl.sbuf, name="top8_idx")
    nisa.nc_find_index8(dst=top8_idx[0:T, 0:K], data=probs[0:T, 0:E], vals=top8_vals[0:T, 0:K])

    # -----------------------------------------------------------------------
    # Stage 4: Selective-Expert MLP — 2-Wave with H_HALF buffer halving
    # -----------------------------------------------------------------------
    # output_temp: accumulates weighted expert outputs for each token
    output_temp = nl.ndarray((_PMAX, H_free_shard, T), dtype=nl.float32, buffer=nl.sbuf, name="output_temp")

    for t in nl.static_range(T):

        # Half-sized gate_up buffers: (_PMAX, H_HALF=8, GU_FLAT=384) instead of H_FREE=16
        gate_up_buf0 = nl.ndarray((_PMAX, H_HALF, GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf0_{t}")
        gate_up_buf1 = nl.ndarray((_PMAX, H_HALF, GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf1_{t}")
        gate_up_buf2 = nl.ndarray((_PMAX, H_HALF, GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf2_{t}")
        gate_up_buf3 = nl.ndarray((_PMAX, H_HALF, GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf3_{t}")
        gate_up_bufs = [gate_up_buf0, gate_up_buf1, gate_up_buf2, gate_up_buf3]

        down_full0_buf0 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf0_{t}")
        down_full0_buf1 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf1_{t}")
        down_full0_buf2 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf2_{t}")
        down_full0_buf3 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf3_{t}")
        down_full0_bufs = [down_full0_buf0, down_full0_buf1, down_full0_buf2, down_full0_buf3]

        down_full1_buf0 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf0_{t}")
        down_full1_buf1 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf1_{t}")
        down_full1_buf2 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf2_{t}")
        down_full1_buf3 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf3_{t}")
        down_full1_bufs = [down_full1_buf0, down_full1_buf1, down_full1_buf2, down_full1_buf3]

        # Pad I1 region of down_full1 to zero (constant offset weight tile)
        for k_pad in range(4):
            nisa.memset(down_full1_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        # Scratch buffers for second I-tile (I1 elements zero-padded to I0)
        # Shape is (_PMAX, H_HALF, I0) — H_HALF instead of H_FREE
        gate_t1_128 = nl.ndarray((_PMAX, H_HALF, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_t1_128_{t}")
        up_t1_128   = nl.ndarray((_PMAX, H_HALF, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf, name=f"up_t1_128_{t}")

        # Routing weights for this token
        sum_topk = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"sum_topk_{t}")
        nisa.tensor_reduce(sum_topk[0:T, 0:1], nl.add, top8_vals[0:T, 0:K], axis=1)

        inv_sum_topk = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"inv_sum_topk_{t}")
        nisa.activation(inv_sum_topk[0:T, 0:1], op=nl.reciprocal, data=sum_topk[0:T, 0:1])

        norm_weights = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf, name=f"norm_weights_{t}")
        nisa.tensor_scalar(
            norm_weights[0:T, 0:K], data=top8_vals[0:T, 0:K],
            op0=nl.multiply, operand0=inv_sum_topk[0:T, 0:1],
        )

        aff_bcast = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf, name=f"aff_bcast_{t}")
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K], src=norm_weights[t:t + 1, 0:K])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        # gate_up_psum and down_psum are reused across both waves
        gate_up_psum = nl.ndarray((_PMAX, K_WAVE * 2 * I_TILES), dtype=nl.float32, buffer=nl.psum, name=f"gate_up_psum_{t}")
        down_psum    = nl.ndarray((_PMAX, K_WAVE * H_free_shard), dtype=nl.float32, buffer=nl.psum, name=f"down_psum_{t}")

        # ===================================================================
        # WAVE 0: Experts 0-3
        # ===================================================================

        # Step 1: Load down weights for all K_WAVE experts (outside h_half loop — UNCHANGED)
        for k in nl.static_range(K_WAVE):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            nisa.dma_copy(
                dst=down_full0_bufs[k],
                src=down_w.ap(
                    pattern=[[H_dim, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full1_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H_dim, I1], [1, H_shard]],
                    offset=I0 * H_dim + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # Step 2: Reset gate_up_psum BEFORE h_half loop (accumulates across both h_halves)
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # Step 3: h_half loop — load H_HALF rows at a time, accumulate gate_up_psum
        for h_half in nl.static_range(2):
            h_offset = h_half * H_HALF  # compile-time: 0 or 8

            # Load gate_up weights for this h_half (H_HALF rows instead of H_FREE)
            for k in nl.static_range(K_WAVE):
                expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

                nisa.dma_copy(
                    dst=gate_up_bufs[k],   # shape: (_PMAX, H_HALF, GU_FLAT)
                    src=gate_up_w.ap(
                        pattern=[[GU_FLAT, _PMAX], [_PMAX * GU_FLAT, H_HALF], [1, GU_FLAT]],
                        offset=h_offset * _PMAX * GU_FLAT,
                        scalar_offset=expert_id,
                        indirect_dim=0,
                    ),
                    dge_mode=0,
                )

            # Matmul accumulation for this h_half (gate_up_psum is NOT reset here)
            for k in nl.static_range(K_WAVE):
                gu_base = k * 2 * I_TILES

                # Pad I1 region of scratch buffers to zero for this k (must redo for each h_half)
                nisa.memset(gate_t1_128[0:_PMAX, 0:H_HALF, nl.ds(I1, I1)], value=0.0)
                nisa.memset(up_t1_128[0:_PMAX, 0:H_HALF, nl.ds(I1, I1)],   value=0.0)

                # Copy second I-tile (I1 elements) into zero-padded scratch
                nisa.tensor_copy(
                    dst=gate_t1_128[0:_PMAX, 0:H_HALF, 0:I1],
                    src=gate_up_bufs[k][0:_PMAX, 0:H_HALF, nl.ds(I0, I1)],
                )
                nisa.tensor_copy(
                    dst=up_t1_128[0:_PMAX, 0:H_HALF, 0:I1],
                    src=gate_up_bufs[k][0:_PMAX, 0:H_HALF, nl.ds(I + I0, I1)],
                )

                # Matmul: accumulate gate_up_psum from H_HALF input tiles
                for h1 in nl.affine_range(H_HALF):
                    for i_tile in nl.static_range(I_TILES):
                        if i_tile == 0:
                            g_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                            u_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                        else:
                            g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]
                            u_stat = up_t1_128[0:_PMAX, h1, 0:I0]
                        # Global h1 index: h_half * H_HALF is compile-time, h1 is affine
                        global_h1_offset = h_half * H_HALF * T
                        nisa.nc_matmul(
                            dst=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                            stationary=g_stat,
                            moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(global_h1_offset + h1 * T, T)],
                        )
                        nisa.nc_matmul(
                            dst=gate_up_psum[0:_PMAX, gu_base + I_TILES + i_tile:gu_base + I_TILES + i_tile + 1],
                            stationary=u_stat,
                            moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(global_h1_offset + h1 * T, T)],
                        )

        # Step 4: Silu + down projection (AFTER h_half loop, gate_up_psum fully accumulated)
        for k in nl.static_range(K_WAVE):
            gu_base = k * 2 * I_TILES
            d_base  = k * H_free_shard

            silu_res = nl.ndarray((_PMAX, I_TILES), dtype=nl.float32, buffer=nl.sbuf, name=f"w0_silu_res_{t}_{k}")
            nisa.activation(silu_res, op=nl.silu, data=gate_up_psum[0:_PMAX, gu_base:gu_base + I_TILES])

            up_sb = nl.ndarray((_PMAX, I_TILES), dtype=nl.float32, buffer=nl.sbuf, name=f"w0_up_sb_{t}_{k}")
            nisa.activation(up_sb, op=nl.copy, data=gate_up_psum[0:_PMAX, gu_base + I_TILES:gu_base + 2 * I_TILES])

            inter_f32 = nl.ndarray((_PMAX, I_TILES), dtype=nl.float32, buffer=nl.sbuf, name=f"w0_inter_f32_{t}_{k}")
            nisa.tensor_tensor(inter_f32, silu_res, up_sb, nl.multiply)

            inter_bf16 = nl.ndarray((_PMAX, I_TILES), dtype=inp_dtype, buffer=nl.sbuf, name=f"w0_inter_bf16_{t}_{k}")
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

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

            down_result_sb = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf, name=f"w0_down_result_sb_{t}_{k}")
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free_shard],
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + H_free_shard],
            )
            down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf, name=f"w0_down_result_scaled_{t}_{k}")
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],
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

        # ===================================================================
        # WAVE 1: Experts 4-7
        # ===================================================================

        # Step 1: Load down weights for Wave 1 experts
        for k in nl.static_range(K_WAVE):
            kk = k + 4
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + kk)

            nisa.dma_copy(
                dst=down_full0_bufs[k],
                src=down_w.ap(
                    pattern=[[H_dim, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full1_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H_dim, I1], [1, H_shard]],
                    offset=I0 * H_dim + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # Step 2: Reset gate_up_psum for Wave 1 (before its own h_half loop)
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # Step 3: h_half loop for Wave 1
        for h_half in nl.static_range(2):
            h_offset = h_half * H_HALF

            for k in nl.static_range(K_WAVE):
                kk = k + 4
                expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + kk)

                nisa.dma_copy(
                    dst=gate_up_bufs[k],
                    src=gate_up_w.ap(
                        pattern=[[GU_FLAT, _PMAX], [_PMAX * GU_FLAT, H_HALF], [1, GU_FLAT]],
                        offset=h_offset * _PMAX * GU_FLAT,
                        scalar_offset=expert_id,
                        indirect_dim=0,
                    ),
                    dge_mode=0,
                )

            for k in nl.static_range(K_WAVE):
                gu_base = k * 2 * I_TILES

                nisa.memset(gate_t1_128[0:_PMAX, 0:H_HALF, nl.ds(I1, I1)], value=0.0)
                nisa.memset(up_t1_128[0:_PMAX, 0:H_HALF, nl.ds(I1, I1)],   value=0.0)

                nisa.tensor_copy(
                    dst=gate_t1_128[0:_PMAX, 0:H_HALF, 0:I1],
                    src=gate_up_bufs[k][0:_PMAX, 0:H_HALF, nl.ds(I0, I1)],
                )
                nisa.tensor_copy(
                    dst=up_t1_128[0:_PMAX, 0:H_HALF, 0:I1],
                    src=gate_up_bufs[k][0:_PMAX, 0:H_HALF, nl.ds(I + I0, I1)],
                )

                for h1 in nl.affine_range(H_HALF):
                    for i_tile in nl.static_range(I_TILES):
                        if i_tile == 0:
                            g_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                            u_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                        else:
                            g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]
                            u_stat = up_t1_128[0:_PMAX, h1, 0:I0]
                        global_h1_offset = h_half * H_HALF * T
                        nisa.nc_matmul(
                            dst=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                            stationary=g_stat,
                            moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(global_h1_offset + h1 * T, T)],
                        )
                        nisa.nc_matmul(
                            dst=gate_up_psum[0:_PMAX, gu_base + I_TILES + i_tile:gu_base + I_TILES + i_tile + 1],
                            stationary=u_stat,
                            moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(global_h1_offset + h1 * T, T)],
                        )

        # Step 4: Silu + down projection for Wave 1
        for k in nl.static_range(K_WAVE):
            kk = k + 4
            gu_base = k * 2 * I_TILES
            d_base  = k * H_free_shard

            silu_res = nl.ndarray((_PMAX, I_TILES), dtype=nl.float32, buffer=nl.sbuf, name=f"w1_silu_res_{t}_{k}")
            nisa.activation(silu_res, op=nl.silu, data=gate_up_psum[0:_PMAX, gu_base:gu_base + I_TILES])

            up_sb = nl.ndarray((_PMAX, I_TILES), dtype=nl.float32, buffer=nl.sbuf, name=f"w1_up_sb_{t}_{k}")
            nisa.activation(up_sb, op=nl.copy, data=gate_up_psum[0:_PMAX, gu_base + I_TILES:gu_base + 2 * I_TILES])

            inter_f32 = nl.ndarray((_PMAX, I_TILES), dtype=nl.float32, buffer=nl.sbuf, name=f"w1_inter_f32_{t}_{k}")
            nisa.tensor_tensor(inter_f32, silu_res, up_sb, nl.multiply)

            inter_bf16 = nl.ndarray((_PMAX, I_TILES), dtype=inp_dtype, buffer=nl.sbuf, name=f"w1_inter_bf16_{t}_{k}")
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

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

            down_result_sb = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf, name=f"w1_down_result_sb_{t}_{k}")
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free_shard],
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + H_free_shard],
            )
            down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf, name=f"w1_down_result_scaled_{t}_{k}")
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, kk:kk + 1],
            )

            nisa.tensor_tensor(
                dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                data1=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                data2=down_result_scaled[0:_PMAX, 0:H_free_shard],
                op=nl.add,
            )

    # -----------------------------------------------------------------------
    # Stage 5: Transpose fp32->bf16, store to HBM shard
    # -----------------------------------------------------------------------
    # output: full [B, H=2048] bf16 — each LNC core writes its prg_id shard
    output = nl.ndarray((B, H), dtype=inp_dtype, buffer=nl.shared_hbm, name="output")

    # Stage 5a: transpose output_temp (_PMAX, H_free_shard, T) -> flat (T, H_shard)
    out_flat_sb = nl.ndarray((T, H_shard), dtype=inp_dtype, buffer=nl.sbuf, name="out_flat_sb")
    for h1 in nl.static_range(H_free_shard):
        tp_psum = nl.ndarray((T, _PMAX), dtype=nl.float32, buffer=nl.psum, name=f"tp_{h1}")
        nisa.nc_transpose(dst=tp_psum[0:T, 0:_PMAX], data=output_temp[0:_PMAX, h1, 0:T])
        nisa.activation(dst=out_flat_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)], op=nl.copy, data=tp_psum[0:T, 0:_PMAX])

    # Stage 5b: store this core's H_shard to the appropriate slice of output HBM
    # prg_id 0 -> output[:, 0:1024], prg_id 1 -> output[:, 1024:2048]
    if prg_id == 0:
        nisa.dma_copy(
            dst=output[0:T, 0:H_shard],
            src=out_flat_sb[0:T, 0:H_shard],
            dge_mode=nisa.dge_mode.hwdge,
        )
    else:
        nisa.dma_copy(
            dst=output[0:T, nl.ds(H_shard, H_shard)],
            src=out_flat_sb[0:T, 0:H_shard],
            dge_mode=nisa.dge_mode.hwdge,
        )

    # Barrier: wait for both cores to finish writing before returning
    nisa.core_barrier(data=output, cores=(0, 1))

    return output
