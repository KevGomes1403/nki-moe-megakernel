"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

kernel_v24a — Round 3 Plan A: Contiguous router DMA
=====================================================

Based on kernel_v22b (double-buffer expert weights pipeline).

Change vs v22b:
  Stage 2 router DMA: replace 3D non-contiguous dge_mode=0 pattern with flat
  2D contiguous slice loads using dge_mode=3.

  OLD (v22b):
    router_w_wide_sb [_PMAX, _ROUTER_BATCH, E]
    nisa.dma_copy with 3D ap() pattern, dge_mode=0
    → ~128 DMA packets per chunk × 4 chunks ≈ 512 router DMA packets

  NEW (v24a):
    router_tile_sb [_PMAX, E] — single tile buffer, reused across h_sub
    nisa.dma_copy with contiguous 2D slice per h_sub, dge_mode=3
    → 1 contiguous packet per h_sub × _ROUTER_BATCH × n_chunks ≈ 16 router DMA packets
    (vs ~512 strided packets in v22b)

  SBUF constraint: partition dim must be <= _PMAX=128. Cannot use a single
  [_ROUTER_BATCH*_PMAX, E] buffer as that would be 512 partition rows.
  Instead, use a single [_PMAX, E] tile buffer and load each h_sub tile
  separately inside the inner loop using contiguous slice addressing.

  Expected: ~500 fewer sw_dma_packets, measurable DMA overhead reduction.
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

# Flat gate+up combined width (native layout: gate cols 0:I, up cols I:2*I)
_GU_FLAT = 2 * _I   # = 384

# LNC=2 H-sharding constants (always launched with [2])
_N_PRGS = 2
_H_FREE = _H // _PMAX             # = 16 tiles of 128 each
_H_FREE_SHARD = _H_FREE // _N_PRGS   # = 8  (each core owns 8 H-tiles for output)
_H_SHARD = _H_FREE_SHARD * _PMAX     # = 1024

# Router DMA batching: 4 tiles per DMA → 4×32KB = 128KB per packet (contiguous)
_ROUTER_BATCH = 4  # H_FREE must be divisible by this


@nki.jit
def qwen3_moe_fused_tkg(
    inp,        # [B, 1, H=2048]  bf16
    gamma,      # [1, H=2048]     bf16
    router_w,   # [H=2048, E=128] bf16
    gate_up_w,  # [E=128, H=2048, 2*I=384] bf16  — NATIVE: gate cols 0:192, up cols 192:384
    down_w,     # [E=128, I=192,  H=2048]  bf16  — NATIVE: no shard split
):
    """
    Fused RMSNorm + Router + TopK(K=8) + Expert MLP for Qwen3 MoE TKG.
    Invoked as: qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
    where [2] means LNC=2 (two NeuronCores).
    Returns: output [T, H=2048] bf16 — complete, no partial sum or all-reduce needed.
    """
    B = inp.shape[0]
    T = B  # seq_len=1 for TKG, so tokens = batch

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

    # LNC program ID (0 or 1) — compile-time constant per NeuronCore
    prg_id = nl.program_id(axis=0)

    # -----------------------------------------------------------------------
    # Stage 1: RMSNorm
    # -----------------------------------------------------------------------
    inp_2d = inp.reshape((T, H))

    inp_2d_hbm_reshaped = inp_2d.reshape((H_free * T, _PMAX))
    inp_flat_sb = nl.ndarray((H_free * T, _PMAX), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=inp_flat_sb,
        src=inp_2d_hbm_reshaped,
        dge_mode=3,
    )

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
    #
    # v24a change: use contiguous 2D DMA (dge_mode=3) per tile instead of 3D
    # non-contiguous ap() pattern (dge_mode=0).
    #
    # router_w [H=2048, E=128] is row-major. Each H-tile (h1) is a contiguous
    # _PMAX×E = 128×128 block (32KB). Load per tile with dge_mode=3.
    #
    # SBUF partition constraint: par_dim <= _PMAX=128. Use a single tile buffer
    # router_tile_sb [_PMAX, E] and reload for each h_sub in the inner loop.
    # Each DMA covers exactly _PMAX*E*2=32KB contiguous bytes → 1 DMA packet.
    # Total: H_free=16 tiles × 1 packet = 16 router DMA packets
    # (vs ~512 strided packets in v22b using 3D ap() with dge_mode=0).
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)
    router_tile_sb = nl.ndarray((_PMAX, E), dtype=inp.dtype, buffer=nl.sbuf)

    for h_chunk in nl.affine_range(H_free // _ROUTER_BATCH):
        for h_sub in nl.static_range(_ROUTER_BATCH):
            h1 = h_chunk * _ROUTER_BATCH + h_sub
            nisa.dma_copy(
                dst=router_tile_sb,
                src=router_w[nl.ds(h1 * _PMAX, _PMAX), 0:E],
                dge_mode=3,
            )
            nisa.nc_matmul(
                dst=logits_psum[0:T, 0:E],
                stationary=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                moving=router_tile_sb[0:_PMAX, 0:E],
            )

    logits_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(logits_sb[0:T, 0:E], op=nl.copy, data=logits_psum[0:T, 0:E])

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8) + normalize weights
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

    sum_topk = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(sum_topk[0:T, 0:1], nl.add, top8_vals[0:T, 0:K], axis=1)

    inv_sum_topk = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(inv_sum_topk[0:T, 0:1], op=nl.reciprocal, data=sum_topk[0:T, 0:1])

    norm_weights = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        norm_weights[0:T, 0:K], data=top8_vals[0:T, 0:K],
        op0=nl.multiply, operand0=inv_sum_topk[0:T, 0:1],
    )

    # -----------------------------------------------------------------------
    # Stage 4: Selective-Expert MLP — Plan B: Double-buffer ping-pong
    #
    # Two separate 3D SBUF buffers per resource (slot 0 and slot 1).
    # nl.static_range unrolls the k-loop at compile time, so Python if/else
    # on k%2 selects between the two slot buffers statically.
    #
    # For each k:
    #   s_cur = k % 2  → slot holding DMA'd weights for expert k
    #   s_nxt = 1 - s_cur → slot to prefetch expert k+1 into
    #   Issue DMA for expert k+1 → s_nxt  (overlaps with expert k compute)
    #   Compute expert k from s_cur
    # -----------------------------------------------------------------------
    output_temp = nl.ndarray((_PMAX, H_free_shard, T), dtype=nl.float32, buffer=nl.sbuf)

    # Ping-pong slot buffers — two separate 3D SBUF tensors per weight matrix
    # gate_up_slot{0,1}: [_PMAX, H_free, _GU_FLAT] = [128, 16, 384]
    gate_up_slot0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
    gate_up_slot1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
    # down0_slot{0,1}: [_PMAX, H_shard] = [128, 1024]
    down0_slot0 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
    down0_slot1 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
    # down1_slot{0,1}: [I1, H_shard] = [64, 1024]  (only I1=64 valid rows used + pad region)
    # Use _PMAX for par_dim so DMA can write rows 0:I1 and pad region 0-filled
    down1_slot0 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
    down1_slot1 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)

    # gate_t1_128/up_t1_128: single pair reused across k iterations
    gate_t1_128 = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
    up_t1_128   = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)

    # Hoist PSUM buffers (per-expert memset stays inside k-loop)
    gate_up_psum = nl.ndarray((_PMAX, 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
    down_psum    = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.psum)

    for t in nl.static_range(T):

        # ------------------------------------------------------------------
        # Hoist ALL pad memsets before the pipelined loop
        # ------------------------------------------------------------------
        # Zero pad rows I1:_PMAX in both slot 0 and slot 1 down1 buffers
        nisa.memset(down1_slot0[nl.ds(I1, I1), 0:H_shard], value=0.0)
        nisa.memset(down1_slot1[nl.ds(I1, I1), 0:H_shard], value=0.0)

        # gate_t1_128 and up_t1_128 pad zeros (one per token, outside k-loop)
        nisa.memset(gate_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
        nisa.memset(up_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        # ------------------------------------------------------------------
        # aff_bcast setup
        # ------------------------------------------------------------------
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
        # Prolog: Issue DMA for expert 0 → slot 0
        # ------------------------------------------------------------------
        expert_id_0 = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + 0)

        nisa.dma_copy(
            dst=gate_up_slot0[0:_PMAX, 0:H_free, 0:_GU_FLAT],
            src=gate_up_w.ap(
                pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
                offset=0,
                scalar_offset=expert_id_0,
                indirect_dim=0,
            ),
            dge_mode=0,
        )
        nisa.dma_copy(
            dst=down0_slot0[0:_PMAX, 0:H_shard],
            src=down_w.ap(
                pattern=[[H, I0], [1, H_shard]],
                offset=prg_id * H_shard,
                scalar_offset=expert_id_0,
                indirect_dim=0,
            ),
            dge_mode=0,
        )
        nisa.dma_copy(
            dst=down1_slot0[0:I1, 0:H_shard],
            src=down_w.ap(
                pattern=[[H, I1], [1, H_shard]],
                offset=I0 * H + prg_id * H_shard,
                scalar_offset=expert_id_0,
                indirect_dim=0,
            ),
            dge_mode=0,
        )

        # ------------------------------------------------------------------
        # Pipelined loop: k=0..K-1
        # nl.static_range unrolls at compile time → Python if/else on k%2
        # resolves statically → correct slot selection without runtime branching.
        # ------------------------------------------------------------------
        for k in nl.static_range(K):
            s_cur = k % 2
            s_nxt = 1 - s_cur

            # Prefetch expert k+1 → slot s_nxt (issues DMA before compute for k starts)
            if k < K - 1:
                expert_id_k1 = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + (k + 1))

                if s_nxt == 0:
                    nisa.dma_copy(
                        dst=gate_up_slot0[0:_PMAX, 0:H_free, 0:_GU_FLAT],
                        src=gate_up_w.ap(
                            pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
                            offset=0,
                            scalar_offset=expert_id_k1,
                            indirect_dim=0,
                        ),
                        dge_mode=0,
                    )
                    nisa.dma_copy(
                        dst=down0_slot0[0:_PMAX, 0:H_shard],
                        src=down_w.ap(
                            pattern=[[H, I0], [1, H_shard]],
                            offset=prg_id * H_shard,
                            scalar_offset=expert_id_k1,
                            indirect_dim=0,
                        ),
                        dge_mode=0,
                    )
                    nisa.dma_copy(
                        dst=down1_slot0[0:I1, 0:H_shard],
                        src=down_w.ap(
                            pattern=[[H, I1], [1, H_shard]],
                            offset=I0 * H + prg_id * H_shard,
                            scalar_offset=expert_id_k1,
                            indirect_dim=0,
                        ),
                        dge_mode=0,
                    )
                else:  # s_nxt == 1
                    nisa.dma_copy(
                        dst=gate_up_slot1[0:_PMAX, 0:H_free, 0:_GU_FLAT],
                        src=gate_up_w.ap(
                            pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
                            offset=0,
                            scalar_offset=expert_id_k1,
                            indirect_dim=0,
                        ),
                        dge_mode=0,
                    )
                    nisa.dma_copy(
                        dst=down0_slot1[0:_PMAX, 0:H_shard],
                        src=down_w.ap(
                            pattern=[[H, I0], [1, H_shard]],
                            offset=prg_id * H_shard,
                            scalar_offset=expert_id_k1,
                            indirect_dim=0,
                        ),
                        dge_mode=0,
                    )
                    nisa.dma_copy(
                        dst=down1_slot1[0:I1, 0:H_shard],
                        src=down_w.ap(
                            pattern=[[H, I1], [1, H_shard]],
                            offset=I0 * H + prg_id * H_shard,
                            scalar_offset=expert_id_k1,
                            indirect_dim=0,
                        ),
                        dge_mode=0,
                    )

            # ------------------------------------------------------------------
            # Compute expert k from slot s_cur
            # ------------------------------------------------------------------

            # Select the right slot buffers for this iteration
            if s_cur == 0:
                gate_up_buf = gate_up_slot0
                down0_buf   = down0_slot0
                down1_buf   = down1_slot0
            else:
                gate_up_buf = gate_up_slot1
                down0_buf   = down0_slot1
                down1_buf   = down1_slot1

            # --- Tile-1 prep: copy gate/up partial tile → gate_t1_128/up_t1_128 ---
            for h1 in nl.static_range(H_free):
                nisa.tensor_copy(
                    dst=gate_t1_128[0:_PMAX, h1, 0:I1],
                    src=gate_up_buf[0:_PMAX, h1, nl.ds(I0, I1)],
                )
                nisa.tensor_copy(
                    dst=up_t1_128[0:_PMAX, h1, 0:I1],
                    src=gate_up_buf[0:_PMAX, h1, nl.ds(I + I0, I1)],
                )

            # --- Gate/Up matmul ---
            nisa.memset(gate_up_psum, value=0.0)

            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_buf[0:_PMAX, h1, nl.ds(0, I0)]   # gate t0
                        u_stat = gate_up_buf[0:_PMAX, h1, nl.ds(I, I0)]   # up t0
                    else:
                        g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]            # gate t1
                        u_stat = up_t1_128[0:_PMAX, h1, 0:I0]              # up t1
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, i_tile:i_tile + 1],
                        stationary=g_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, I_tiles + i_tile:I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )

            # --- Flush gate/up PSUM → SBUF, SiLU, inter ---
            gate_sb = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            up_sb   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_sb, op=nl.copy, data=gate_up_psum[0:_PMAX, 0:I_tiles])
            nisa.activation(up_sb,   op=nl.copy, data=gate_up_psum[0:_PMAX, I_tiles:2 * I_tiles])
            silu_res  = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(silu_res, op=nl.silu, data=gate_sb)
            nisa.tensor_tensor(inter_f32, silu_res, up_sb, nl.multiply)
            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            # --- Down matmul ---
            nisa.memset(down_psum, value=0.0)

            for h1_out in nl.affine_range(H_free_shard):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down0_buf[0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down1_buf[0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            # --- Flush down PSUM, scale by affinity, accumulate ---
            down_result_sb = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(down_result_sb, op=nl.copy, data=down_psum[0:_PMAX, 0:H_free_shard])
            down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                down_result_scaled, data=down_result_sb,
                op0=nl.multiply, operand0=aff_bcast[0:_PMAX, k:k + 1],
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
    """Run kernel_v24a with native weight layouts — no preprocessing required."""
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
