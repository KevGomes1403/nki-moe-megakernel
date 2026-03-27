"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).

v6a — Plan A: Unified K-loop (Eliminate 3-Pass Structure)

Key change: Replace the 3-pass structure (gate/up all-K → down all-K → accumulate)
with a single end-to-end K-loop per expert. Each iteration of the K-loop does:
  a) Gate/up projection (affine_range H_free_shard)
  b) Flush gate_psum → gate_sb, flush up_psum → up_sb
  c) sendrecv gate with pipe_id=0, sendrecv up with pipe_id=0
     (pipe_id=0 for ALL sendrecvs is correct — they run sequentially within static_range k)
  d) tensor_tensor add: full_gate = gate_sb + recv_gate, full_up = up_sb + recv_up
  e) silu(full_gate) * full_up → inter (local [128, I_tiles] SBUF)
  f) Down projection (affine_range i_tile2)
  g) Apply affinity: down_result_sb *= aff_bcast[0:_PMAX, k:k+1]
  h) Accumulate into output_temp: if k==0 → tensor_copy, else → tensor_tensor add

Benefits:
  - Removes inter_buf and down_buf (36KB of K-slot intermediate buffers freed)
  - Inter results are computed and consumed immediately — no K-slot SBUF storage
  - Separate PSUM banks for gate and up allow simultaneous accumulation

What is kept from v5a:
  - All RMSNorm logic (Stage 1, batch inp/gamma loads)
  - All Router logic (Stage 2, softmax, TopK, normalize)
  - Pre-broadcast affinity (aff_bcast computation before k-loop)
  - Fused gate_up_w reshape to [E, H, 2*I]
  - TensorView for dynamic expert weight selection
  - Down projection inner structure (loading down_w_full, affine_range i_tile2, PSUM→SBUF flush)
  - Stage 5 (output transpose and store)
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
    other = 1 - prg_id   # compile-time constant; the other LNC core's rank

    # -----------------------------------------------------------------------
    # Stage 1: RMSNorm
    # Input: inp [B, 1, H] -> flatten to [T, H] -> SBUF [128, H_free*T]
    # Output: rmsnorm_out [128, H_free*T] (full H, both cores identical)
    #
    # Batch load strategy (from v4a/v5a):
    #   a) Reshape inp_2d[T, H] -> [H_free*T, _PMAX] in HBM (contiguous, no copy).
    #   b) One DMA (no transpose): [H_free*T, _PMAX] -> SBUF [H_free*T, _PMAX]
    #   c) nc_transpose SBUF [H_free*T, _PMAX] -> PSUM [_PMAX, H_free*T]
    #   d) activation copy PSUM -> SBUF: rmsnorm_out [_PMAX, H_free*T]
    # -----------------------------------------------------------------------
    inp_2d = inp.reshape((T, H))

    # T=1 (TKG) assumption: H_free*T = H_free throughout this section.
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

    # Gamma batch load: replace 16-iteration loop with single DMA + transpose.
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
    # LHS/RHS swap: stationary=rmsnorm [128, T], moving=router_w [128, E]
    # PSUM[T, E]: T in PSUM partition dim, E in free dim -- directly usable for softmax
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)
    router_w_tile_sb = nl.ndarray((_PMAX, E), dtype=router_w.dtype, buffer=nl.sbuf)

    for h1 in nl.affine_range(H_free):
        nisa.dma_copy(
            dst=router_w_tile_sb,
            src=router_w[nl.ds(h1 * _PMAX, _PMAX), 0:E],
            dge_mode=3,
        )
        nisa.nc_matmul(
            dst=logits_psum[0:T, 0:E],
            stationary=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
            moving=router_w_tile_sb[0:_PMAX, 0:E],
        )

    logits_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(logits_sb[0:T, 0:E], op=nl.copy, data=logits_psum[0:T, 0:E])

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8) + normalize weights
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
    # v6a — Plan A: Unified K-loop (single pass per expert)
    # Eliminates inter_buf and down_buf (36KB freed).
    # Each k-iteration: gate/up → sendrecv → silu → down → affinity → accumulate.
    # -----------------------------------------------------------------------

    # Fuse gate and up dims for a single weight load per H-tile.
    # [E=128, H=2048, 2, I=256] -> [E=128, H=2048, 512]
    E_shape, H_shape, _, I_shape = gate_up_w.shape
    gate_up_w_fused = gate_up_w.reshape((E_shape, H_shape, I_shape * 2))

    # output_temp [128, H_free_shard, T] - accumulates this core's H output shard (fp32)
    # k=0 uses tensor_copy to initialize (avoids add-to-zero overhead, no memset needed).
    output_temp = nl.ndarray((_PMAX, H_free_shard, T), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.static_range(T):

        # ── Pre-broadcast all K affinity weights ────────────────────────────────
        # norm_weights has shape [T, K] with T in the partition dimension.
        # For T=1, partition 0 holds all K weights for this token.
        # Broadcast to all 128 partitions once (4 shuffles total) before k-loop.
        # aff_bcast[p, k] == norm_weights[t, k] for all partitions p.
        aff_bcast = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K], src=norm_weights[t:t + 1, 0:K])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        # ── Unified K-loop: gate/up + down + accumulate per expert ──────────────
        # static_range required: sendrecv pipe_ids must be issued in fixed order,
        # and the output accumulation across k carries a dependency.
        for k in nl.static_range(K):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            # --- Gate/Up projection ---
            fused_w_view_shard = (
                TensorView(gate_up_w_fused)
                .select(dim=0, index=expert_id)
                .slice(dim=0, start=prg_id * H_shard, end=(prg_id + 1) * H_shard)
                .reshape_dim(dim=0, shape=[H_free_shard, _PMAX])
            )

            # Separate PSUM tensors for gate and up — auto-allocated by compiler.
            # (nkilib uses gate_psum_base_bank=0, up_psum_base_bank=num_allocated_psums;
            #  here we rely on compiler auto-allocation to place them in distinct banks)
            gate_psum = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.psum)
            up_psum   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(gate_psum, value=0.0)
            nisa.memset(up_psum,   value=0.0)

            for h1 in nl.affine_range(H_free_shard):
                fused_tile = nl.ndarray((_PMAX, I * 2), dtype=gate_up_w.dtype, buffer=nl.sbuf)
                fused_h_view = fused_w_view_shard.slice(dim=0, start=h1, end=h1 + 1).squeeze_dim(dim=0)
                nisa.dma_copy(dst=fused_tile, src=fused_h_view.get_view(),
                              dge_mode=_adaptive_dge(fused_h_view))
                for i_tile in nl.static_range(I_tiles):
                    nisa.nc_matmul(
                        dst=gate_psum[0:_PMAX, i_tile:i_tile + 1],
                        stationary=fused_tile[0:_PMAX, nl.ds(i_tile * I0, I0)],
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((prg_id * H_free_shard + h1) * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=up_psum[0:_PMAX, i_tile:i_tile + 1],
                        stationary=fused_tile[0:_PMAX, nl.ds(I + i_tile * I0, I0)],
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((prg_id * H_free_shard + h1) * T, T)],
                    )

            # Flush PSUM -> SBUF (single flush per projection, after full H-shard accumulation)
            gate_sb = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            up_sb   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_sb, op=nl.copy, data=gate_psum)
            nisa.activation(up_sb,   op=nl.copy, data=up_psum)

            # All-reduce gate/up partial sums across LNC cores via NeuronLink SBUF-to-SBUF.
            # pipe_id=0 for ALL sendrecvs is correct — they run sequentially within
            # static_range k, so there is no pipe conflict. nkilib also uses pipe_id=0.
            recv_gate = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            recv_up   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.sendrecv(src=gate_sb, dst=recv_gate,
                          send_to_rank=other, recv_from_rank=other, pipe_id=0)
            nisa.sendrecv(src=up_sb,   dst=recv_up,
                          send_to_rank=other, recv_from_rank=other, pipe_id=0)

            # Combine partial sums: full_gate = local + received
            full_gate = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            full_up   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(full_gate, gate_sb, recv_gate, nl.add)
            nisa.tensor_tensor(full_up,   up_sb,   recv_up,   nl.add)

            # SiLU gate * up -> inter [_PMAX, I_tiles]
            # No intermediate K-slot buffer (v6a: compute and consume immediately)
            silu_res = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(silu_res, op=nl.silu, data=full_gate)
            nisa.tensor_tensor(inter_f32, silu_res, full_up, nl.multiply)

            # Cast inter to bf16 for down projection matmul
            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            # --- Down projection ---
            down_w_view = (
                TensorView(down_w)
                .select(dim=0, index=expert_id)
                .slice(dim=1, start=prg_id * H_shard, end=(prg_id + 1) * H_shard)
            )

            # Persistent PSUM — 1 bank covers all H_free_shard columns
            down_psum = nl.ndarray((_PMAX, _PSUM_FREE), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(down_psum, value=0.0)

            for i_tile2 in nl.affine_range(I_tiles):
                down_w_full = nl.ndarray((I0, H_free_shard * _PMAX), dtype=down_w.dtype, buffer=nl.sbuf)
                down_w_i_view = down_w_view.slice(dim=0, start=i_tile2 * I0, end=(i_tile2 + 1) * I0)
                nisa.dma_copy(
                    dst=down_w_full,
                    src=down_w_i_view.get_view(),
                    dge_mode=_adaptive_dge(down_w_i_view),
                )
                for h1_out in nl.static_range(H_free_shard):
                    nisa.nc_matmul(
                        dst=down_psum[0:_PMAX, nl.ds(h1_out * T, T)],
                        stationary=down_w_full[0:I0, nl.ds(h1_out * _PMAX, _PMAX)],
                        moving=inter_bf16[0:_PMAX, i_tile2:i_tile2 + 1],
                    )

            # Single PSUM flush per expert
            down_result_sb = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free_shard],
                op=nl.copy,
                data=down_psum[0:_PMAX, 0:H_free_shard],
            )

            # Apply affinity scale using pre-broadcast weights.
            # aff_bcast[:, k] already has norm_weights[t, k] replicated across all 128 partitions.
            nisa.tensor_scalar(
                dst=down_result_sb,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],
            )

            # Accumulate into output_temp — inline (no down_buf K-slot needed).
            # k=0: tensor_copy to initialize (avoids add-to-zero overhead, no memset).
            # k>0: tensor_tensor add.
            if k == 0:
                nisa.tensor_copy(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    src=down_result_sb[0:_PMAX, 0:H_free_shard],
                )
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data1=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data2=down_result_sb[0:_PMAX, 0:H_free_shard],
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
    outputs = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
