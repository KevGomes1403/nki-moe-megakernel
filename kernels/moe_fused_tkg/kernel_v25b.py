"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

kernel_v25b — Plan B: Batch gate_up 3D DMA + full-H down DMA
=============================================================

Interface contract (no repack required):
  gate_up_w  [E=128, H=2048, 2*I=384] bf16 — native layout from qwen.py/qwen_fused_moe_tkg.py
                                              cols 0:I   = gate weights
                                              cols I:2*I = up   weights
  down_w     [E=128, I=192,  H=2048]   bf16 — native layout from qwen.py/qwen_fused_moe_tkg.py

Change 1 — gate_up DMA: ONE 3D DMA per expert loads all H_free=16 H-tiles at once.
  3D pattern: [[_GU_FLAT, _PMAX], [_PMAX*_GU_FLAT, H_free], [1, _GU_FLAT]]
  Element (p, h1, c) → HBM offset: p*384 + h1*128*384 + c = row (h1*128+p), col c
  Total gate_up DMAs: 1 per expert × 8 = 8 (down from 128 in v20b).

Change 2 — down DMA: load full H=2048 (not H_shard=1024) per I-tile per expert.
  stride_p=H=count=H → fully contiguous. prg_id offsets into SBUF at matmul time.
  Total down packets: 1 per expert per tile × 8 × 2 = 16 (vs ~2048 in v19b).

Change 3 — Flat SBUF buffers for all 3 buffer types.

SBUF budget estimate (per partition lane, 224 KiB limit):
  gate_up_flat:    K × H_free × _GU_FLAT × 2 / _PMAX = 8×16×384×2/128 = 96 KB
  down_full0_flat: K × H × 2 / _PMAX = 8×2048×2/128 = 32 KB
  down_full1_flat: 32 KB
  other buffers:   ~46 KB (same as v19b)
  Stage 4 peak:    96 + 32 + 32 + 46 = 206 KB < 224 KB ✓
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
_GU_FLAT = 2 * _I   # = 384  (stride_p for coalesced DMA: 384 == count_col 384)

# LNC=2 H-sharding constants (always launched with [2])
_N_PRGS = 2
_H_FREE = _H // _PMAX             # = 16 tiles of 128 each
_H_FREE_SHARD = _H_FREE // _N_PRGS   # = 8  (each core owns 8 H-tiles for output)
_H_SHARD = _H_FREE_SHARD * _PMAX     # = 1024

# Router DMA batching: 4 tiles per DMA → 4×32KB = 128KB per packet
_ROUTER_BATCH = 4  # H_FREE must be divisible by this

# Plan B: flat gate_up stride per expert
# H_free H-tiles per expert, each _GU_FLAT wide
_GU_STRIDE_PER_EXPERT = _H_FREE * _GU_FLAT   # = 16 * 384 = 6144 cols per expert in flat gate_up buf


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

    gate_up_w is [E, H, 2*I=384] in native flat format (gate then up, no zero-pad).
    down_w is [E, I=192, H=2048] in native format (no shard pre-split).
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
    # Input: inp [B, 1, H] → flatten to [T, H] → SBUF [_PMAX, H_free*T]
    # Output: rmsnorm_normed_bf16 [_PMAX, H_free*T] (full H, both cores identical)
    #
    # Single 3D DMA with dge_mode=3 to load inp and gamma.
    # -----------------------------------------------------------------------
    inp_2d = inp.reshape((T, H))

    # Load inp: HBM [H_free*T, _PMAX] → SBUF [H_free*T, _PMAX] via dge_mode=3
    inp_2d_hbm_reshaped = inp_2d.reshape((H_free * T, _PMAX))
    inp_flat_sb = nl.ndarray((H_free * T, _PMAX), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=inp_flat_sb,
        src=inp_2d_hbm_reshaped,
        dge_mode=3,
    )

    # Transpose [H_free*T, _PMAX] → PSUM [_PMAX, H_free*T]
    inp_trans_psum = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)

    # PSUM [_PMAX, H_free*T] → SBUF rmsnorm_out
    rmsnorm_out = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_out[...], op=nl.copy, data=inp_trans_psum[...])

    # Load gamma: HBM [H_free, _PMAX] → SBUF → transpose → SBUF gamma_sb [_PMAX, H_free]
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

    # 1d. Cross-partition sum via GpSimdE (replaces 128×128 TensorE ones-matmul)
    sum_reduced_sb = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=sum_reduced_sb[0:1, 0:T], data=rmsnorm_reduced[0:_PMAX, 0:T], op=nl.add)

    # Broadcast [1, T] → [_PMAX, T]: copy to row 0 then shuffle to all 128 rows
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

    # 1f. rmsnorm_normed = gamma_mult * norm_factor (broadcast over H_free tiles)
    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=2).broadcast(dim=2, size=H_free)
    rmsnorm_normed = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(rmsnorm_normed[...], gamma_mult[...], norm_factor_bcast.get_view(), nl.multiply)

    # Cast to bf16 for router and expert matmuls
    rmsnorm_normed_bf16 = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=rmsnorm_normed[...])

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul [T, H] @ [H, E=128] → logits [T, E]
    #
    # Router weight batched DMA: 4 tiles per DMA (preserved from v13).
    # 4 × 128KB = 512KB per call vs 16 × 32KB; reduces DMA packet count 4×.
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

    # TopK via DVE hardware
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
        norm_weights[0:T, 0:K], data=top8_vals[0:T, 0:K],
        op0=nl.multiply, operand0=inv_sum_topk[0:T, 0:1],
    )

    # -----------------------------------------------------------------------
    # Stage 4: Selective-Expert MLP — Plan B: 2D coalesced gate_up + full-H down
    #
    # Phase 1: Issue ALL DMAs (gate_up: H_free 2D DMAs per expert × K experts;
    #          down: 2 full-H DMAs per expert × K experts) before any compute.
    #
    # Phase 2: Serial expert compute reads from pre-loaded flat SBUF buffers.
    #
    # gate_up_flat [_PMAX, K * H_free * _GU_FLAT]:
    #   Expert k occupies cols [k*_GU_STRIDE_PER_EXPERT : (k+1)*_GU_STRIDE_PER_EXPERT].
    #   Within expert k, H-tile h1 occupies cols [k*_GU_STRIDE_PER_EXPERT + h1*_GU_FLAT :
    #                                              k*_GU_STRIDE_PER_EXPERT + (h1+1)*_GU_FLAT].
    #   Each 2D DMA is [_PMAX, _GU_FLAT] fully contiguous (stride_p=_GU_FLAT=count).
    #
    # down_full0_flat / down_full1_flat [_PMAX, K * H]:
    #   Expert k occupies cols [k*H : (k+1)*H] (full H=2048 columns).
    #   DMA loads full H width → stride_p=H=count → 1 HW packet per I-tile per expert.
    #   prg_id selects H_shard at matmul time via offset into flat buf.
    # -----------------------------------------------------------------------
    output_temp = nl.ndarray((_PMAX, H_free_shard, T), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.static_range(T):

        # ------------------------------------------------------------------
        # Flat SBUF: one contiguous block for all K experts
        # gate_up: [_PMAX, K * H_free * _GU_FLAT] — H-tile contiguous within expert slot
        # down: load FULL H per expert per I-tile — avoids strided H-shard access
        # ------------------------------------------------------------------
        gate_up_flat    = nl.ndarray((_PMAX, K * H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        down_full0_flat = nl.ndarray((_PMAX, K * H),      dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_flat = nl.ndarray((_PMAX, K * H),      dtype=down_w.dtype, buffer=nl.sbuf)

        # ------------------------------------------------------------------
        # Hoist ALL pad memsets before the prefetch loop
        # ------------------------------------------------------------------
        # Zero pad region (rows I1:_PMAX) for ALL K experts in down_full1_flat in one instruction
        nisa.memset(down_full1_flat[nl.ds(I1, I1), 0:K * H], value=0.0)

        # gate_t1_128/up_t1_128: single pair of reused buffers (overwritten each k)
        gate_t1_128 = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        up_t1_128   = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        nisa.memset(gate_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
        nisa.memset(up_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        # Hoist PSUM buffers (per-expert memset stays inside k-loop)
        gate_up_psum = nl.ndarray((_PMAX, 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
        down_psum    = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.psum)

        # ------------------------------------------------------------------
        # aff_bcast setup (done before prefetch, overlaps with DMA loading)
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
        # Phase 1: Issue ALL DMAs before compute
        # For gate_up: ONE 3D DMA per expert loads all H_free=16 H-tiles at once
        #   3D pattern: [[_GU_FLAT, _PMAX], [_PMAX*_GU_FLAT, H_free], [1, _GU_FLAT]]
        # For down: full-H 2D DMA per expert per I-tile, step_p=H=count_col → 1 HW packet per I-tile
        # ------------------------------------------------------------------
        for k in nl.static_range(K):  # static_range: k is compile-time constant for scalar_offset
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            gate_expert_col = k * _GU_STRIDE_PER_EXPERT   # col base in gate_up_flat for expert k

            # ONE 3D DMA per expert: loads all H_free=16 H-tiles of [_PMAX, _GU_FLAT] in one shot
            # Total: [_PMAX, H_free * _GU_FLAT] = [128, 6144] elements per expert
            #
            # 3D pattern interpretation:
            #   Dim 0 (outermost): partition dim — step=_GU_FLAT, count=_PMAX=128
            #     → partition p starts at p * _GU_FLAT (p-th row of H_free H-tiles)
            #   Dim 1 (middle): H-tile dim — step=_PMAX*_GU_FLAT, count=H_free=16
            #     → H-tile h1 starts at h1 * _PMAX * _GU_FLAT (h1-th block of 128 rows)
            #   Dim 2 (innermost): column dim — step=1, count=_GU_FLAT=384
            #     → 384 contiguous columns
            #
            # Element (p, h1, c) maps to HBM offset: p*384 + h1*128*384 + c
            # = row (h1*128 + p), col c of the expert's [2048, 384] gate_up block
            #
            # In SBUF flat buffer: partition p gets free-dim elements [h1*384+c] for all h1,c
            # = exactly the gate_up_flat layout where expert k occupies
            #   cols [k*6144 : (k+1)*6144] with H-tile h1 at cols [k*6144 + h1*384 : k*6144 + (h1+1)*384]
            nisa.dma_copy(
                dst=gate_up_flat[0:_PMAX, nl.ds(k * H_free, H_free), 0:_GU_FLAT],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            down_expert_col = k * H   # col base in full-H down flat bufs for expert k

            # down tile 0: full H width, I0=128 rows — step_p=H=count_col → FULLY CONTIGUOUS
            # HBM: down_w[expert_id, 0:I0, 0:H] = contiguous block (I0 × H = 128×2048×2B)
            nisa.dma_copy(
                dst=down_full0_flat[0:_PMAX, nl.ds(down_expert_col, H)],
                src=down_w.ap(
                    # step_p=H=count_col → contiguous partition rows → 1 HW packet
                    pattern=[[H, I0], [1, H]],
                    offset=0,                    # start at I-row 0 for this expert
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # down tile 1: full H width, I1=64 rows — same coalesced pattern, offset to I-tile 1
            # Pad rows I1:_PMAX already zeroed by memset above
            nisa.dma_copy(
                dst=down_full1_flat[0:I1, nl.ds(down_expert_col, H)],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H]],
                    offset=I0 * H,               # skip I0=128 rows (I-tile 0)
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # ------------------------------------------------------------------
        # Phase 2: Serial expert compute — reads from pre-loaded flat SBUF buffers
        # ------------------------------------------------------------------
        for k in nl.static_range(K):
            gate_expert_htile = k * H_free   # H-tile base in gate_up_flat dim-1 for expert k
            down_expert_col = k * H

            # --- Tile-1 prep (gate_t1_128, up_t1_128) ---
            for h1 in nl.static_range(H_free):
                nisa.tensor_copy(
                    dst=gate_t1_128[0:_PMAX, h1, 0:I1],
                    src=gate_up_flat[0:_PMAX, gate_expert_htile + h1, nl.ds(I0, I1)],       # gate cols I0:I0+I1
                )
                nisa.tensor_copy(
                    dst=up_t1_128[0:_PMAX, h1, 0:I1],
                    src=gate_up_flat[0:_PMAX, gate_expert_htile + h1, nl.ds(I + I0, I1)],   # up cols I+I0:I+I0+I1
                )

            # --- Gate/Up matmul ---
            nisa.memset(gate_up_psum, value=0.0)

            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_flat[0:_PMAX, gate_expert_htile + h1, nl.ds(0,  I0)]  # gate t0 [128, 128]
                        u_stat = gate_up_flat[0:_PMAX, gate_expert_htile + h1, nl.ds(I,  I0)]  # up   t0 [128, 128]
                    else:
                        g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]   # gate t1 [128, 128]
                        u_stat = up_t1_128[0:_PMAX, h1, 0:I0]     # up   t1 [128, 128]
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

            # --- PSUM flush, SiLU, inter ---
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

            # --- Down matmul using full-H flat buffers ---
            # Within each expert's full-H block, select our shard (prg_id) and output tile (h1_out)
            # off = expert_base + shard_offset + tile_offset
            # prg_id=0 → cols 0:1024, prg_id=1 → cols 1024:2048 within expert
            nisa.memset(down_psum, value=0.0)

            for h1_out in nl.affine_range(H_free_shard):
                off = down_expert_col + prg_id * H_shard + h1_out * _PMAX
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_full0_flat[0:_PMAX, nl.ds(off, _PMAX)],   # [128, 128]
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_full1_flat[0:_PMAX, nl.ds(off, _PMAX)],   # [128, 128]
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            # --- Flush, scale, accumulate ---
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
    # output_temp [_PMAX, H_free_shard, T] → HBM output [T, H] bf16
    # Each core writes its H_shard columns at HBM offset prg_id*H_shard.
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
    """Run kernel_v25b with native weight layouts — no preprocessing required.

    Accepts gate_up_w as either:
      [E, H, 2*I=384]        — flat native (gate cols 0:I, up cols I:2I)
      [E, H, 2, I=192]       — 4D view as passed by qwen_fused_moe_tkg.py (reshaped at zero cost)
      [E, H, 2, I_padded=256] — 4D padded view from test harness (sliced to I=192)

    down_w: [E, I=192, H=2048]       — native layout.
            [E, I_padded=256, H=2048] — padded layout from test harness (sliced to I=192).

    Returns: output [T, H=2048] bf16 — complete, no partial sum needed.
    """
    import torch
    import torch_xla.core.xla_model as xm

    # Accept [E, H, 2, I_any] from qwen_fused_moe_tkg.py or test harness
    if gate_up_w.dim() == 4:
        E, Hd, two, Iv = gate_up_w.shape
        # Slice to actual I=192 if padded (e.g. I_padded=256 from test harness)
        if Iv != _I:
            gate_up_w = gate_up_w[:, :, :, :_I]
        # Reorder [E, H, 2, I] → [E, H, I, 2] → [E, H, 2*I] to get native flat layout
        # Native layout: gate cols 0:I, up cols I:2*I
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

    outputs = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
