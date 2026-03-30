"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

kernel_v14a — Plan A: K-Expert Split Across Cores
==================================================

Changes from v10b:
  1. K=8 experts split across the two LNC cores instead of H-sharding the
     down projection:
       - Core 0 processes top-4 experts (indices 0..3 from top8_idx)
       - Core 1 processes top-4 experts (indices 4..7 from top8_idx)
  2. Each core now computes the FULL H=2048 output for its K/2=4 experts
     (no H-sharding, no sendrecv).
  3. Return tensor shape changes from [T, H] to [2, T, H]; run() sums
     output[0] + output[1] to produce the final [T, H] result.

Removed: _H_FREE_SHARD, _H_SHARD (no longer needed)
Added:   _K_PER_CORE = 4

SBUF budget estimate (per core):
  inp_flat_sb:         [16, 128] bf16         =  4 KB
  gamma_flat_sb:       [16, 128] bf16         =  4 KB
  rmsnorm_out:         [128, 16] bf16         =  4 KB
  rmsnorm_normed_bf16: [128, 16] bf16         =  4 KB
  output_temp:         [128, 16, 1] fp32      = 64 KB   (H_free=16, T=1)
  inter_buf:           [128, 2, 4] bf16       =  2 KB   (I_tiles=2, K_half=4)
  down_buf:            [128, 16, 4] fp32      = 64 KB   (H_free=16, K_half=4)
  fused_tile_3d:       [128, 16, 512] bf16    = 128 KB  (full H, gate+up fused, reused)
  down_tile_3d:        [128, 2, 2048] bf16    = 128 KB  (full H, reused)
  aff_bcast:           [128, 4] fp32          =  2 KB
  router_w_wide_sb:    [128, 512] bf16        = 128 KB  (4 tiles × 128 cols, router batch)
  out_sb:              [1, 2048] bf16         =  4 KB
  Note: fused_tile_3d and down_tile_3d don't coexist at peak;
        router_w_wide_sb is reused across router chunks.
        output_temp + down_buf together are 128 KB — this is the key tradeoff
        vs v10b's 64 KB combined. Within 224 KB limit.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.tensor_view import TensorView

# Hardware constants
_PMAX = 128      # partition dimension max
_PSUM_FREE = 512  # PSUM free-dimension max on trn2 (nl.tile_size.psum_fmax)

# Qwen3-30B-A3B at TP=4 fixed dims
_H = 2048    # hidden dim
_E = 128     # num experts
_K = 8       # top-K experts
_I = 256     # intermediate dim per TP rank (padded from 192)
_EPS = 1e-6

# LNC=2 K-split constants
_N_PRGS = 2
_K_PER_CORE = _K // 2   # = 4
_H_FREE = _H // _PMAX   # = 16 tiles of 128 each
_I0 = _PMAX             # = 128
_I_TILES = _I // _I0    # = 2

# Router DMA batching: 4 tiles per DMA → 4×32KB=128KB per packet
_ROUTER_BATCH = 4  # tiles per DMA batch; H_FREE must be divisible by this


@nki.jit
def qwen3_moe_fused_tkg(
    inp,        # [B, 1, H=2048] bf16
    gamma,      # [1, H=2048] bf16
    router_w,   # [H=2048, E=128] bf16
    gate_up_w,  # [E=128, H=2048, 2, I=256] bf16
    down_w,     # [E=128, I=256, H=2048] bf16
):
    """
    Fused RMSNorm + Router + TopK(K=8) + Expert MLP for Qwen3 MoE TKG.
    Invoked as: qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
    where [2] means LNC=2 (two cores).

    Plan A: K-expert split.
      Core 0 processes experts 0..3, Core 1 processes experts 4..7.
      Each core produces full-H output for its K/2 experts.
    Returns: output_partial [2, T, H=2048] bf16
             run() sums output_partial[0] + output_partial[1] -> [T, H]
    """
    B = inp.shape[0]
    T = B  # seq_len=1, so tokens = batch

    H = _H
    E = _E
    K = _K
    I = _I
    H_free = _H_FREE
    K_half = _K_PER_CORE   # = 4
    I0 = _I0
    I_tiles = _I_TILES

    # LNC program ID (0 or 1 for LNC=2)
    prg_id = nl.program_id(axis=0)

    # K-split: each core handles a different half of the top-K experts
    k_start = prg_id * K_half   # 0 for core 0, 4 for core 1

    # -----------------------------------------------------------------------
    # Stage 1: RMSNorm
    # Input: inp [B, 1, H] -> flatten to [T, H] -> SBUF [128, H_free*T]
    # Output: rmsnorm_normed_bf16 [128, H_free*T] (full H, both cores identical)
    #
    # Batch load: single 3D DMA for inp and gamma.
    # -----------------------------------------------------------------------
    inp_2d = inp.reshape((T, H))

    # Step a+b: Single DMA from HBM [H_free*T, _PMAX] -> SBUF [H_free*T, _PMAX]
    inp_2d_hbm_reshaped = inp_2d.reshape((H_free * T, _PMAX))
    inp_flat_sb = nl.ndarray((H_free * T, _PMAX), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=inp_flat_sb,
        src=inp_2d_hbm_reshaped,
        dge_mode=3,
    )

    # Step c: SBUF transpose [H_free*T, _PMAX] -> PSUM [_PMAX, H_free*T]
    inp_trans_psum = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)

    # Step d: PSUM [_PMAX, H_free*T] -> SBUF [_PMAX, H_free*T] = rmsnorm_out
    rmsnorm_out = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_out[...], op=nl.copy, data=inp_trans_psum[...])

    # Single DMA for gamma
    gamma_1d = gamma.reshape((H,))
    gamma_1d_hbm_reshaped = gamma_1d.reshape((H_free, _PMAX))
    gamma_flat_sb = nl.ndarray((H_free, _PMAX), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=gamma_flat_sb,
        src=gamma_1d_hbm_reshaped,
        dge_mode=3,
    )
    gamma_trans_psum = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=gamma_trans_psum, data=gamma_flat_sb)
    gamma_sb = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.activation(gamma_sb[...], op=nl.copy, data=gamma_trans_psum[...])

    # RMSNorm computation

    # 1a. x^2 [_PMAX, H_free*T]
    rmsnorm_sq = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(rmsnorm_sq[...], op=nl.square, data=rmsnorm_out[...])

    # 1b. Reduce x^2 over all H_free*T elements (axis=1) -> [_PMAX, T]
    rmsnorm_reduced = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(rmsnorm_reduced[0:_PMAX, 0:T], nl.add, rmsnorm_sq[0:_PMAX, 0:H_free * T], axis=1)

    # 1c. gamma * input [_PMAX, H_free*T]
    gamma_mult = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(gamma_mult[...], rmsnorm_out[...], gamma_sb[...], nl.multiply)

    # 1d. Reduce sum(x^2) across all 128 partitions via nc_matmul with all-ones
    matmul_const = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(matmul_const, value=1.0)
    final_psum = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(stationary=matmul_const, moving=rmsnorm_reduced, dst=final_psum)

    # 1e. Compute norm_factor = rsqrt(sum/H + eps)
    eps_sb = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(eps_sb, value=_EPS)
    norm_factor_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        norm_factor_sb[0:_PMAX, 0:T],
        op=nl.rsqrt,
        data=final_psum[0:_PMAX, 0:T],
        scale=1.0 / H,
        bias=eps_sb[0:_PMAX, :],
    )

    # 1f. rmsnorm_normed = gamma_mult * norm_factor [_PMAX, H_free*T]
    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=2).broadcast(dim=2, size=H_free)
    rmsnorm_normed = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(rmsnorm_normed[...], gamma_mult[...], norm_factor_bcast.get_view(), nl.multiply)

    # Convert to bf16 for matmul
    rmsnorm_normed_bf16 = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=rmsnorm_normed[...])

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul [T, H] @ [H, E=128] -> logits [T, E]
    #
    # Router weight batching — 4 tiles per DMA.
    # 16 separate 32KB DMAs → 4 × 128KB DMAs.
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)
    # Wide SBUF buffer: holds 4 consecutive router weight tiles at once.
    router_w_wide_sb = nl.ndarray((_PMAX, _ROUTER_BATCH, E), dtype=inp.dtype, buffer=nl.sbuf)

    for h_chunk in nl.affine_range(H_free // _ROUTER_BATCH):  # 4 outer iterations
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
    # Stage 3: Softmax + TopK(8) + normalize weights
    # -----------------------------------------------------------------------
    max_logit = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(max_logit[0:T, 0:1], nl.maximum, logits_sb[0:T, 0:E], axis=1)

    centered = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        centered[0:T, 0:E],
        data=logits_sb[0:T, 0:E],
        op0=nl.subtract,
        operand0=max_logit[0:T, 0:1],
    )

    exp_vals = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(exp_vals[0:T, 0:E], op=nl.exp, data=centered[0:T, 0:E])

    sum_exp = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(sum_exp[0:T, 0:1], nl.add, exp_vals[0:T, 0:E], axis=1)

    inv_sum_exp = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(inv_sum_exp[0:T, 0:1], op=nl.reciprocal, data=sum_exp[0:T, 0:1])

    probs = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        probs[0:T, 0:E],
        data=exp_vals[0:T, 0:E],
        op0=nl.multiply,
        operand0=inv_sum_exp[0:T, 0:1],
    )

    # TopK using DVE hardware
    top8_vals = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_vals[0:T, 0:K], src=probs[0:T, 0:E])

    top8_idx = nl.ndarray((T, K), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=top8_idx[0:T, 0:K], data=probs[0:T, 0:E], vals=top8_vals[0:T, 0:K])

    # Normalize top-K weights
    sum_topk = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(sum_topk[0:T, 0:1], nl.add, top8_vals[0:T, 0:K], axis=1)

    inv_sum_topk = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(inv_sum_topk[0:T, 0:1], op=nl.reciprocal, data=sum_topk[0:T, 0:1])

    norm_weights = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        norm_weights[0:T, 0:K],
        data=top8_vals[0:T, 0:K],
        op0=nl.multiply,
        operand0=inv_sum_topk[0:T, 0:1],
    )

    # -----------------------------------------------------------------------
    # Stage 4: Selective-Expert MLP (K-expert split version)
    #
    # Plan A strategy:
    #   Gate/Up: each core loads FULL H of gate/up weights for its K_half=4
    #            experts only. Each core computes full intermediate [128, I_tiles=2].
    #   Down: each core loads FULL H=2048 of down weights for its K_half experts.
    #         Each core writes the FULL output for its K_half experts, then
    #         accumulates into output_temp [128, H_free, T].
    #
    # No H-sharding, no sendrecv. Each core returns its partial sum over K_half
    # experts; the host sums the two partial outputs.
    # -----------------------------------------------------------------------

    E_shape, H_shape, _, I_shape = gate_up_w.shape
    gate_up_w_fused = gate_up_w.reshape((E_shape, H_shape, I_shape * 2))

    # output_temp: FULL H_free=16 tiles (full H output for this core's K_half experts)
    # No memset — k=0 uses tensor_copy to initialise
    output_temp = nl.ndarray((_PMAX, H_free, T), dtype=nl.float32, buffer=nl.sbuf)

    # K-slot buffers for two-pass pattern — K_half=4 slots
    inter_buf = nl.ndarray((_PMAX, I_tiles, K_half), dtype=inp.dtype, buffer=nl.sbuf)
    down_buf  = nl.ndarray((_PMAX, H_free, K_half), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.static_range(T):

        # ── Pre-broadcast K_half affinity weights for this core ─────────────────
        # norm_weights [T, K]: use k_start..k_start+K_half (this core's experts).
        aff_bcast = nl.ndarray((_PMAX, K_half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K_half], src=norm_weights[t:t + 1, nl.ds(k_start, K_half)])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K_half],
                src=aff_bcast[0:1, 0:K_half],
                shuffle_mask=[0] * 32,
            )

        # ── PASS 1: gate/up for K_half experts (full H, no I-shard) ──────────────
        # Use k_start offset so each core accesses its own experts from top8_idx.
        for k in nl.affine_range(K_half):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k_start + k)

            # 3D DMA for gate+up weights — load FULL H (all H_free=16 tiles)
            # gate_up_w_fused shape: [E=128, H=2048, I*2=512]
            fused_tile_3d = nl.ndarray((_PMAX, H_free, I * 2), dtype=gate_up_w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=fused_tile_3d,
                src=gate_up_w_fused.ap(
                    pattern=[[I * 2, _PMAX], [_PMAX * I * 2, H_free], [1, I * 2]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # PSUM for gate/up — full I_tiles=2
            gate_psum = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.psum)
            up_psum   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(gate_psum, value=0.0)
            nisa.memset(up_psum,   value=0.0)

            # Loop over full H_free=16, full I_tiles=2
            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    nisa.nc_matmul(
                        dst=gate_psum[0:_PMAX, i_tile:i_tile + 1],
                        stationary=fused_tile_3d[0:_PMAX, h1, nl.ds(i_tile * I0, I0)],
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=up_psum[0:_PMAX, i_tile:i_tile + 1],
                        stationary=fused_tile_3d[0:_PMAX, h1, nl.ds(I + i_tile * I0, I0)],
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )

            # Flush PSUM -> SBUF
            gate_sb = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            up_sb   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_sb, op=nl.copy, data=gate_psum)
            nisa.activation(up_sb,   op=nl.copy, data=up_psum)

            # SiLU activation + element-wise multiply
            silu_res  = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(silu_res, op=nl.silu, data=gate_sb)
            nisa.tensor_tensor(inter_f32, silu_res, up_sb, nl.multiply)

            # Store to K-slot buffer for Pass 2
            nisa.activation(
                inter_buf[0:_PMAX, 0:I_tiles, k:k + 1],
                op=nl.copy,
                data=inter_f32,
            )

        # ── PASS 2: down projection for K_half experts — FULL H (not H_shard) ──
        # Each core loads ALL H columns of down weights for its K_half experts.
        for k in nl.affine_range(K_half):
            expert_id_k = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k_start + k)

            # 3D DMA for down weights — FULL H (offset=0, no shard offset)
            # down_w shape: [E=128, I=256, H=2048]
            # Target: down_tile_3d[i0, i1, h] = down_w[expert_id, i1*I0 + i0, h]
            # AP pattern (3D):
            #   i0: stride = H = 2048,       count = I0 = 128
            #   i1: stride = I0*H = 262144,  count = I_tiles = 2
            #   h:  stride = 1,              count = H = 2048
            # offset = 0 — full H, no shard offset
            down_tile_3d = nl.ndarray((I0, I_tiles, H), dtype=down_w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=down_tile_3d,
                src=down_w.ap(
                    pattern=[[H, I0], [I0 * H, I_tiles], [1, H]],
                    offset=0,
                    scalar_offset=expert_id_k,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # Persistent PSUM covering all H_free=16 output tiles
            down_psum = nl.ndarray((_PMAX, _PSUM_FREE), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(down_psum, value=0.0)

            # Load inter for this expert from K-slot buffer
            inter_bf16_k = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=inter_bf16_k, src=inter_buf[0:_PMAX, 0:I_tiles, k:k + 1])

            # Matmul loop — FULL H_free=16 (not H_free_shard=8)
            for i_tile2 in nl.affine_range(I_tiles):
                for h1_out in nl.static_range(H_free):
                    nisa.nc_matmul(
                        dst=down_psum[0:_PMAX, nl.ds(h1_out * T, T)],
                        stationary=down_tile_3d[0:I0, i_tile2, nl.ds(h1_out * _PMAX, _PMAX)],
                        moving=inter_bf16_k[0:_PMAX, i_tile2:i_tile2 + 1],
                    )

            # Single PSUM flush per expert (full H_free columns)
            down_result_sb = nl.ndarray((_PMAX, H_free), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free],
                op=nl.copy,
                data=down_psum[0:_PMAX, 0:H_free],
            )

            # Scale by affinity weight (pre-broadcast, using k-th slot of K_half)
            down_result_scaled = nl.ndarray((_PMAX, H_free), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],
            )
            # Store to K-slot buffer for Pass 3
            nisa.tensor_copy(
                dst=down_buf[0:_PMAX, 0:H_free, k:k + 1],
                src=down_result_scaled[0:_PMAX, 0:H_free],
            )

        # ── PASS 3: accumulate K_half down results into output_temp ─────────────
        # Copy-for-k=0: no memset needed.
        for k in nl.static_range(K_half):
            if k == 0:
                nisa.tensor_copy(
                    dst=output_temp[0:_PMAX, 0:H_free, t:t + 1],
                    src=down_buf[0:_PMAX, 0:H_free, 0:1],
                )
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, 0:H_free, t:t + 1],
                    data1=output_temp[0:_PMAX, 0:H_free, t:t + 1],
                    data2=down_buf[0:_PMAX, 0:H_free, k:k + 1],
                    op=nl.add,
                )

    # -----------------------------------------------------------------------
    # Stage 5: Transpose, cast fp32->bf16, and store to this core's slot
    # output_temp [128, H_free, T] (fp32) -> output_partial[prg_id*T:(prg_id+1)*T, H] (bf16)
    # Each core writes its partial sum into its row-slice of a [2*T, H] tensor.
    # run() reshapes to [2, T, H] and sums the two halves.
    # -----------------------------------------------------------------------
    output_partial = nl.ndarray((2 * T, H), dtype=inp.dtype, buffer=nl.hbm)
    out_sb = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.sbuf)

    for h1 in nl.static_range(H_free):  # full H_free=16 tiles
        tp_psum = nl.ndarray((T, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp_psum[0:T, 0:_PMAX], data=output_temp[0:_PMAX, h1, 0:T])
        nisa.activation(
            dst=out_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)],
            op=nl.copy,
            data=tp_psum[0:T, 0:_PMAX],
        )

    nisa.dma_copy(
        dst=output_partial[nl.ds(prg_id * T, T), 0:H],
        src=out_sb[0:T, 0:H],
    )

    return output_partial


def run(inp, gamma, router_w, gate_up_w, down_w):
    """
    Run the fused Qwen3 MoE TKG kernel with LNC=2 K-expert split.

    Each core processes K/2=4 experts and writes a full-H partial sum into
    a [2*T, H] output tensor (row 0..T-1 = core 0, row T..2T-1 = core 1).
    Sum the two partial outputs to get the final [T, H] result.
    """
    output_partial = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
    if isinstance(output_partial, (tuple, list)):
        output_partial = output_partial[0]
    T = inp.shape[0]
    # Sum the two cores' contributions: [2*T, H] -> [T, H]
    return output_partial[0:T] + output_partial[T:2 * T]
