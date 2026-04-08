"""
Plan A optimized version of the fused MoE TKG kernel.

Changes vs kernel.py (Plan A — Persistent PSUM Accumulation Across H-Tiles):
  1. Gate/Up projection: declare one persistent PSUM per projection before the h1 loop,
     zero once with nisa.memset, let all H_free=16 tiles accumulate naturally, flush once.
     Eliminates 32 per-iteration PSUM→SBUF copies (was 16 h1 * 2 i_tile each).
  2. Down projection: declare one persistent PSUM [128, H_free_shard=8] before both loops,
     zero once, let all (i_tile2, h1_out) pairs accumulate naturally, flush once.
     Eliminates 16 per-iteration PSUM→SBUF copies (was 2 i_tile * 8 h1_out each).
  3. Output accumulation: copy on k=0, add on k>0 — removes the initial memset of
     output_temp and avoids one VectorE add-to-zero per (t, k=0).

All other code (RMSNorm, Router, TopK, weight loading, SiLU, affinity) is UNCHANGED.

DO NOT import or call nkilib kernel functions (moe_block_tkg, _moe_tkg, etc.)
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.tensor_view import TensorView

# Hardware constants
_PMAX = 128      # partition dimension max

# Qwen3-30B-A3B at TP=4 fixed dims
_H = 2048    # hidden dim
_E = 128     # num experts
_K = 8       # top-K experts
_I = 256     # intermediate dim per TP rank (padded from 192)
_EPS = 1e-6

# LNC=2 sharding constants (hardcoded, always launched with [2])
_N_PRGS = 2
_H_FREE = _H // _PMAX          # = 16 tiles of 128 each
_H_FREE_SHARD = _H_FREE // _N_PRGS  # = 8 (each core handles 8 output H-tiles for down proj)
_H_SHARD = _H_FREE_SHARD * _PMAX    # = 1024
_I0 = _PMAX     # = 128
_I_TILES = _I // _I0  # = 2


def _adaptive_dge(tv):
    """Return _DGE_NONE for static tensors, _DGE_DYN for dynamic-access TensorViews."""
    if isinstance(tv, TensorView) and tv.has_dynamic_access():
        return 0   # dynamic DMA
    return 3       # static DMA


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
    Returns: output [B, H=2048] bf16
    """
    B = inp.shape[0]
    T = B  # seq_len=1, so tokens = batch

    H = _H
    E = _E
    K = _K
    I = _I
    H_free = _H_FREE
    H_free_shard = _H_FREE_SHARD
    H_shard = _H_SHARD
    I0 = _I0
    I_tiles = _I_TILES

    # LNC program ID (0 or 1 for LNC=2)
    prg_id = nl.program_id(axis=0)

    # -----------------------------------------------------------------------
    # Stage 1: RMSNorm
    # Input: inp [B, 1, H] -> flatten to [T, H] -> SBUF [128, T, H_free]
    # Output: rmsnorm_out [128, T, H_free] (full H, both cores identical)
    # -----------------------------------------------------------------------
    inp_2d = inp.reshape((T, H))

    # Load [T, H] -> SBUF [128, T, H_free]
    # HBM layout: inp_2d[t, h] = inp_2d[t, p + h_free*128]
    # SBUF layout: rmsnorm_out[p, t, h_free]
    # We use a simple tile-by-tile copy to get correct layout
    rmsnorm_out = nl.ndarray((_PMAX, T, H_free), dtype=inp.dtype, buffer=nl.sbuf)
    for h1 in nl.affine_range(H_free):
        # Load inp tile [128, T]: inp_2d[0:T, h1*128:(h1+1)*128] -> transposed to [128, T]
        inp_tile_sb = nl.ndarray((_PMAX, T), dtype=inp.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=inp_tile_sb,
            src=inp_2d[0:T, nl.ds(h1 * _PMAX, _PMAX)],
            dge_mode=3,
        )
        # Store into rmsnorm_out[:, :, h1]
        nisa.tensor_copy(
            dst=rmsnorm_out[0:_PMAX, 0:T, h1:h1+1],
            src=inp_tile_sb[0:_PMAX, 0:T],
        )

    # Load gamma [1, H] -> SBUF [128, H_free]
    gamma_sb = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.sbuf)
    gamma_1d = gamma.reshape((H,))
    for h1 in nl.affine_range(H_free):
        gamma_tile_sb = nl.ndarray((_PMAX, 1), dtype=gamma.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=gamma_tile_sb,
            src=gamma_1d[nl.ds(h1 * _PMAX, _PMAX)],
            dge_mode=3,
        )
        nisa.tensor_copy(
            dst=gamma_sb[0:_PMAX, h1:h1+1],
            src=gamma_tile_sb[0:_PMAX, 0:1],
        )

    # RMSNorm computation
    # 1a. x^2 [128, T, H_free]
    rmsnorm_sq = nl.ndarray((_PMAX, T, H_free), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(rmsnorm_sq[...], op=nl.square, data=rmsnorm_out[...])

    # 1b. Reduce x^2 over H_free (axis=1 = last free dim in 3D tensor [128, T, H_free])
    rmsnorm_reduced = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(rmsnorm_reduced[...], nl.add, rmsnorm_sq[...], axis=1)

    # 1c. gamma * input [128, T, H_free]
    gamma_sb_bcast = TensorView(gamma_sb).expand_dim(dim=1).broadcast(dim=1, size=T)
    gamma_mult = nl.ndarray((_PMAX, T, H_free), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(gamma_mult[...], rmsnorm_out[...], gamma_sb_bcast.get_view(), nl.multiply)

    # 1d. Reduce sum(x^2) across all 128 partitions via nc_matmul with all-ones
    # stationary=all_ones [128, 128], moving=rmsnorm_reduced [128, T]
    # PSUM [Fs=128, Fm=T]: each row i of PSUM = sum_p( 1 * rmsnorm_reduced[p, t] ) = total sum(x^2)
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

    # 1f. rmsnorm_out_normed = gamma_mult * norm_factor [128, T, H_free]
    # norm_factor_sb [128, T] -> broadcast over H_free to [128, T, H_free]
    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=2).broadcast(dim=2, size=H_free)
    rmsnorm_normed = nl.ndarray((_PMAX, T, H_free), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(rmsnorm_normed[...], gamma_mult[...], norm_factor_bcast.get_view(), nl.multiply)

    # Convert to bf16 for matmul (matches reference which uses bf16 matmul)
    rmsnorm_normed_bf16 = nl.ndarray((_PMAX, T, H_free), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=rmsnorm_normed[...])

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul [T, H] @ [H, E=128] -> logits [T, E]
    # LHS/RHS swap: stationary=rmsnorm [128, T], moving=router_w [128, E]
    # PSUM[T, E]: T in PSUM partition dim, E in free dim -- directly usable for softmax
    # -----------------------------------------------------------------------
    # logits PSUM: [T=1, E=128] with T in partition, E in free
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)
    router_w_tile_sb = nl.ndarray((_PMAX, E), dtype=router_w.dtype, buffer=nl.sbuf)

    for h1 in nl.affine_range(H_free):
        nisa.dma_copy(
            dst=router_w_tile_sb,
            src=router_w[nl.ds(h1 * _PMAX, _PMAX), 0:E],
            dge_mode=3,
        )
        # stationary = rmsnorm tile [128, T]: P=128, Fs=T
        # moving     = router_w tile [128, E]: P=128, Fm=E
        # PSUM [Fs=T, Fm=E]: T in partition, E in free
        nisa.nc_matmul(
            dst=logits_psum[0:T, 0:E],
            stationary=rmsnorm_normed_bf16[0:_PMAX, 0:T, h1],
            moving=router_w_tile_sb[0:_PMAX, 0:E],
        )

    logits_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(logits_sb[0:T, 0:E], op=nl.copy, data=logits_psum[0:T, 0:E])

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8) + normalize weights
    # logits_sb [T, E]: T in partition, E in free
    # -----------------------------------------------------------------------
    # Stable softmax
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
    # Stage 4: Selective-Expert MLP
    #
    # Gate/Up projection: use FULL hidden [128, T, H_free] (all H channels).
    # Both cores compute IDENTICAL gate/up results independently.
    #
    # Down projection: each core uses its H_shard slice of down weights.
    # Core 0 -> output[:, 0:1024], Core 1 -> output[:, 1024:2048].
    #
    # nc_matmul LHS/RHS swap for gate/up:
    #   stationary = weight_tile [128, I0]  (partition=H0=128, free=I0=128)
    #   moving     = hidden_tile [128, 1]   (partition=H0=128, free=T=1)
    #   PSUM [I0=128, T=1]  at psum[0:I0, 0:T]
    #
    # nc_matmul LHS/RHS swap for down:
    #   stationary = down_w_tile [I0=128, H0=128]  (partition=I0, free=H0)
    #   moving     = inter_tile [I0=128, T=1]       (partition=I0, free=T)
    #   PSUM [H0=128, T=1]  at psum[0:H0, 0:T]
    # -----------------------------------------------------------------------

    # PLAN A CHANGE 3: output_temp declared without memset.
    # For k=0 we write directly (tensor_copy), for k>0 we accumulate (tensor_tensor add).
    # This avoids adding scaled expert output onto an all-zeros initial value.
    output_temp = nl.ndarray((_PMAX, H_free_shard, T), dtype=nl.float32, buffer=nl.sbuf)
    # NOTE: no nisa.memset here — k=0 uses tensor_copy to initialize

    for t in nl.static_range(T):
        for k in nl.static_range(K):
            # Dynamic expert ID for this (t, k) pair
            expert_id = top8_idx.ap(
                pattern=[[K, 1], [1, 1]],
                offset=t * K + k,
            )

            # ---- Gate/Up Projection: FULL hidden @ gate_up_w -> gate_out, up_out ----
            # PLAN A CHANGE 1: Persistent PSUMs declared OUTSIDE the h1 loop.
            # Shape [_PMAX=128, I_tiles=2]: column i_tile accumulates dot-products across
            # all H_free=16 hidden tiles. PSUM hardware adds to existing value on each
            # nc_matmul, so a single zero-then-accumulate replaces 32 copy+add ops.
            gate_psum = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.psum)
            up_psum   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.psum)
            # Zero once per expert so accumulation starts from 0, not leftover state
            nisa.memset(gate_psum, value=0.0)
            nisa.memset(up_psum,   value=0.0)

            # gate_w_view: select expert, select gate dim, reshape to [H_free, 128, I]
            gate_w_view = (
                TensorView(gate_up_w)
                .select(dim=0, index=expert_id)    # [H=2048, 2, I=256]
                .select(dim=1, index=0)            # [H=2048, I=256]
                .reshape_dim(dim=0, shape=[H_free, _PMAX])  # [H_free=16, 128, I=256]
            )

            up_w_view = (
                TensorView(gate_up_w)
                .select(dim=0, index=expert_id)    # [H=2048, 2, I=256]
                .select(dim=1, index=1)            # [H=2048, I=256]
                .reshape_dim(dim=0, shape=[H_free, _PMAX])  # [H_free=16, 128, I=256]
            )

            # Loop over ALL H_free tiles — nc_matmul naturally accumulates into gate_psum/up_psum
            for h1 in nl.static_range(H_free):
                # Load gate weight tile [128, I=256] — UNCHANGED from original
                gate_w_tile = nl.ndarray((_PMAX, I), dtype=gate_up_w.dtype, buffer=nl.sbuf)
                gate_w_h_view = gate_w_view.slice(dim=0, start=h1, end=h1+1).squeeze_dim(dim=0)
                nisa.dma_copy(
                    dst=gate_w_tile,
                    src=gate_w_h_view.get_view(),
                    dge_mode=_adaptive_dge(gate_w_h_view),
                )

                # Load up weight tile [128, I=256] — UNCHANGED from original
                up_w_tile = nl.ndarray((_PMAX, I), dtype=gate_up_w.dtype, buffer=nl.sbuf)
                up_w_h_view = up_w_view.slice(dim=0, start=h1, end=h1+1).squeeze_dim(dim=0)
                nisa.dma_copy(
                    dst=up_w_tile,
                    src=up_w_h_view.get_view(),
                    dge_mode=_adaptive_dge(up_w_h_view),
                )

                for i_tile in nl.static_range(I_tiles):
                    # Each call ADDS to the existing PSUM value in column i_tile.
                    # After all 16 h1 iterations, gate_psum[:, i_tile] holds the full
                    # dot product sum over all H_free hidden tiles for I-chunk i_tile.
                    # Removed: per-iteration g_psum/u_psum + activation(copy) + tensor_tensor(add)
                    nisa.nc_matmul(
                        dst=gate_psum[0:_PMAX, i_tile:i_tile+1],
                        stationary=gate_w_tile[0:_PMAX, nl.ds(i_tile * I0, I0)],
                        moving=rmsnorm_normed_bf16[0:_PMAX, 0:T, h1],
                    )
                    nisa.nc_matmul(
                        dst=up_psum[0:_PMAX, i_tile:i_tile+1],
                        stationary=up_w_tile[0:_PMAX, nl.ds(i_tile * I0, I0)],
                        moving=rmsnorm_normed_bf16[0:_PMAX, 0:T, h1],
                    )

            # Single PSUM→SBUF flush after ALL H_free tiles — was 32 copies, now 2.
            # Removed: nisa.memset(gate_result_sb/up_result_sb) — PSUM memset replaces them.
            gate_result_sb = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            up_result_sb   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_result_sb, op=nl.copy, data=gate_psum)
            nisa.activation(up_result_sb,   op=nl.copy, data=up_psum)

            # SiLU(gate) * up -> inter [128, I_tiles] fp32 — UNCHANGED
            silu_result = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(silu_result, op=nl.silu, data=gate_result_sb)
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(inter_f32, silu_result, up_result_sb, nl.multiply)

            # Convert intermediate to bf16 for down proj — UNCHANGED
            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            # ---- Down Projection ----
            # inter_bf16 [128, I_tiles=2]: partition=128 (I-elements), free=I_tiles
            # down_w[expert_id, :, prg_id*H_shard:(prg_id+1)*H_shard] = [I, H_shard]
            #
            # nc_matmul LHS/RHS swap:
            #   stationary = down_w_tile [I0=128, H0=128]
            #   moving     = inter_bf16  [I0=128, 1]
            #   PSUM       [H0=128, 1]

            down_w_view = (
                TensorView(down_w)
                .select(dim=0, index=expert_id)   # [I=256, H=2048]
                .slice(dim=1, start=prg_id * H_shard, end=(prg_id + 1) * H_shard)  # [I=256, H_shard]
            )

            # PLAN A CHANGE 2: Persistent PSUM for down projection.
            # Shape [128, H_free_shard=8]: column h1_out accumulates contributions from
            # both i_tile2=0 and i_tile2=1 iterations. Size = 128*8*4 = 4 KB.
            # Replaces 16 per-iteration (h1_psum + activation(copy) + tensor_tensor(add)) ops.
            down_psum = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.psum)
            # Zero once per expert before accumulation starts
            nisa.memset(down_psum, value=0.0)

            for h1_out in nl.static_range(H_free_shard):
                for i_tile2 in nl.static_range(I_tiles):
                    # Load [I0=128, H0=128] weight tile — UNCHANGED
                    down_w_tile = nl.ndarray((I0, _PMAX), dtype=down_w.dtype, buffer=nl.sbuf)
                    down_w_h1_view = (
                        down_w_view
                        .slice(dim=0, start=i_tile2 * I0, end=(i_tile2 + 1) * I0)
                        .slice(dim=1, start=h1_out * _PMAX, end=(h1_out + 1) * _PMAX)
                    )
                    nisa.dma_copy(
                        dst=down_w_tile,
                        src=down_w_h1_view.get_view(),
                        dge_mode=_adaptive_dge(down_w_h1_view),
                    )

                    # Accumulates into column h1_out.
                    # i_tile2=0: starts from 0 (due to memset above).
                    # i_tile2=1: hardware adds second partial product automatically.
                    # Removed: h1_psum per-iteration + activation(copy) + tensor_tensor(add)
                    nisa.nc_matmul(
                        dst=down_psum[0:_PMAX, h1_out:h1_out+1],
                        stationary=down_w_tile[0:I0, 0:_PMAX],
                        moving=inter_bf16[0:_PMAX, i_tile2:i_tile2+1],
                    )

            # Single flush for the entire down result — was 16 copies, now 1.
            # Removed: nisa.memset(down_result_sb) — PSUM memset replaces it.
            down_result_sb = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(down_result_sb, op=nl.copy, data=down_psum)

            # Apply affinity (POST_SCALE): multiply by norm_weights[t, k] — UNCHANGED
            # norm_weights [T, K] with T in partition, K in free
            # For T=1: norm_weights[0:1, k:k+1] -> scalar, broadcast to all 128 partitions
            aff_sb = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(aff_sb, value=0.0)
            # Copy the affinity scalar (partition 0 only)
            nisa.tensor_copy(dst=aff_sb[0:1, 0:1], src=norm_weights[t:t+1, k:k+1])
            # Broadcast partition 0 to all 128 partitions via nc_stream_shuffle
            for g in nl.static_range(4):
                nisa.nc_stream_shuffle(
                    dst=aff_sb[nl.ds(g * 32, 32), 0:1],
                    src=aff_sb[0:1, 0:1],
                    shuffle_mask=[0] * 32,
                )

            # Scale down result by affinity (POST_SCALE, keep fp32 for accumulation) — UNCHANGED
            down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_sb,
            )

            # PLAN A CHANGE 3: Copy-on-first, add-on-rest for output accumulation.
            # k=0: write directly to output_temp without adding to zero (no memset needed).
            # k>0: accumulate scaled expert contributions via add.
            if k == 0:
                # Direct copy for first expert — initializes output_temp for this t-slot
                nisa.tensor_copy(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t+1],
                    src=down_result_scaled[0:_PMAX, 0:H_free_shard],
                )
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t+1],
                    data1=output_temp[0:_PMAX, 0:H_free_shard, t:t+1],
                    data2=down_result_scaled[0:_PMAX, 0:H_free_shard],
                    op=nl.add,
                )

    # -----------------------------------------------------------------------
    # Stage 5: Transpose, cast fp32->bf16, and store output
    # output_temp [128, H_free_shard, T] (fp32) -> HBM output [T, H] (bf16)
    # Each core writes H_shard columns at offset prg_id*H_shard
    # -----------------------------------------------------------------------
    output = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.hbm)
    out_sb = nl.ndarray((T, H_shard), dtype=inp.dtype, buffer=nl.sbuf)

    for h1 in nl.static_range(H_free_shard):
        # output_temp[0:128, h1, 0:T] is [128, T] fp32
        # Transpose to [T, 128] via PSUM
        tp_psum = nl.ndarray((T, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp_psum[0:T, 0:_PMAX], data=output_temp[0:_PMAX, h1, 0:T])
        # Cast fp32->bf16 via activation copy from PSUM
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
    """
    Run the fused Qwen3 MoE TKG kernel (Plan A optimized) with LNC=2.

    Returns the expert MLP output after routing and accumulation.
    """
    outputs = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
