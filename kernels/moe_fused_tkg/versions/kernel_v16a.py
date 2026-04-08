"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

kernel_v16a — Plan A: hoisted pad memsets outside k-loop
=========================================================

Interface contract (no repack required):
  gate_up_w  [E=128, H=2048, 2*I=384] bf16 — native layout from qwen.py/qwen_fused_moe_tkg.py
                                              cols 0:I   = gate weights
                                              cols I:2*I = up   weights
  down_w     [E=128, I=192,  H=2048]   bf16 — native layout from qwen.py/qwen_fused_moe_tkg.py

DMA coalescing strategy:
  gate_up:  ONE coalesced DMA per expert, loading [_PMAX, H_free, 2*I=384].
              stride_p = 2*I = 384 == count_col = 384 → fully coalesced.
            Two zero-padded tile-1 buffers (gate_t1_128, up_t1_128) hold the
              partial 64-col second tile, zero-filled at cols 64:128 so nc_matmul
              always sees a K=128 stationary. Filled via SBUF tensor_copy, not DMA.

  down:     ONE coalesced DMA per I-tile per expert, loading full H=2048.
              stride_p = H = 2048 == count_col = 2048 → fully coalesced.
            prg_id offsets into the SBUF at matmul time:
              h1_g = prg_id * H_free_shard + h1_out
            Tile 0: 128 rows  (I rows 0:128), full DMA.
            Tile 1:  64 valid rows (I rows 128:192) + 64 zero-padded rows.
              memset rows 64:128 then DMA rows 0:64.

Plan A optimisation (vs v15):
  Static pad-region memsets hoisted outside the k-loop (run once per token instead
  of once per expert):
    gate_t1_128 pad zeros [cols 64:128]: 8× memset → 1× memset per token
    up_t1_128   pad zeros [cols 64:128]: 8× memset → 1× memset per token
    down_full1  pad zeros [rows 64:128]: 8× memset → 1× memset per token
  gate_up_psum and down_psum allocations also hoisted; their per-expert
  nisa.memset(*, 0.0) calls remain inside k-loop (PSUM must be zeroed per expert).

run() is a direct pass-through — no repack, zero runtime preprocessing cost.
  Accepts gate_up_w as [E, H, 2, I] (4D from qwen_fused_moe_tkg.py view) and
  reshapes to [E, H, 2*I] (zero-cost), then calls the kernel directly.

SBUF budget estimate (per partition lane, 224 KiB limit):
  inp_flat_sb:        [16*T, _PMAX]   bf16  =   4 KB  (T=1)
  gamma_flat_sb:      [H_free, _PMAX] bf16  =   4 KB
  rmsnorm_out:        [_PMAX, 16*T]   bf16  =   4 KB
  rmsnorm_normed_bf16:[_PMAX, 16*T]   bf16  =   4 KB
  output_temp:        [_PMAX, 8, 1]   fp32  =   4 KB  (H_free_shard=8, T=1)
  aff_bcast:          [_PMAX, 8]      fp32  =   4 KB
  gate_up_tile:       [_PMAX,16,384]  bf16  =  12 KB  (per expert, reused)
  gate_t1_128:        [_PMAX,16,128]  bf16  =   4 KB  (hoisted, reused across experts)
  up_t1_128:          [_PMAX,16,128]  bf16  =   4 KB  (hoisted, reused across experts)
  down_full0:         [_PMAX, 2048]   bf16  =   4 KB  (hoisted, reused across experts)
  down_full1:         [_PMAX, 2048]   bf16  =   4 KB  (hoisted, reused across experts)
  router_w_wide_sb:   [_PMAX, 4, 128] bf16  =  64 KB  (4 tiles × H_free/4)
  out_sb:             [T, H_shard]    bf16  =   2 KB
  gate/up/inter SBUF: ~16 KB (PSUM flush + silu + inter)
  Total peak:        ~134 KB  — comfortable within 224 KiB.
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

    # 1d. Cross-partition reduction: sum(x^2) via matmul with ones
    matmul_const = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(matmul_const, value=1.0)
    final_psum = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(stationary=matmul_const, moving=rmsnorm_reduced, dst=final_psum)

    # 1e. norm_factor = rsqrt(sum/H + eps)
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
    # Stage 4: Selective-Expert MLP (single-pass K-loop)
    #
    # gate_up_w [E, H, 2*I=384] — NATIVE layout (gate cols 0:192, up cols 192:384)
    #   DMA: ONE coalesced load per expert, stride_p=384=count=384.
    #   tile 0 (K=128): gate cols 0:128, up cols 192:320 — taken directly from buffer.
    #   tile 1 (K=128): gate cols 128:192 (64 valid), up cols 320:384 (64 valid),
    #     each zero-padded to 128 cols via SBUF tensor_copy into gate_t1_128/up_t1_128.
    #
    # down_w [E, I=192, H=2048] — NATIVE layout (no shard pre-split)
    #   DMA: ONE coalesced full-H load per I-tile per expert.
    #     tile 0: stride_p=H=2048=count=2048 → fully coalesced.
    #     tile 1: 64 valid rows, memset rows 64:128 then DMA rows 0:64; same stride=count.
    #   prg_id offsets the H-shard in SBUF at matmul time:
    #     h1_g = prg_id * H_free_shard + h1_out
    # -----------------------------------------------------------------------
    output_temp = nl.ndarray((_PMAX, H_free_shard, T), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.static_range(T):

        # Broadcast K affinity weights to all partition rows
        aff_bcast = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K], src=norm_weights[t:t + 1, 0:K])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        # Plan A: hoist static pad-region memsets outside k-loop — zeros that never change between experts
        gate_t1_128 = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        up_t1_128   = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        nisa.memset(gate_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
        nisa.memset(up_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        down_full0 = nl.ndarray((_PMAX, H), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1 = nl.ndarray((_PMAX, H), dtype=down_w.dtype, buffer=nl.sbuf)
        nisa.memset(down_full1[nl.ds(I1, I1), 0:H], value=0.0)

        gate_up_psum = nl.ndarray((_PMAX, 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
        down_psum    = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.psum)

        # Single-pass K-loop: for each selected expert, run gate/up DMA + matmul +
        # down DMA + matmul + accumulate into output_temp in one pass.
        for k in nl.affine_range(K):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            # ----------------------------------------------------------------
            # DMA 1: gate_up tile — ONE coalesced load per expert
            # gate_up_w [E, H, 2*I=384]: element (e, h_row, col)
            #   3D AP: P-stride = 2*I = 384 (rows are 384-wide in HBM)
            #          H_free stride = _PMAX * 384 (skip 128 rows at a time)
            #          col stride = 1, count = 384
            # stride_p = 384 = count_col = 384 → FULLY COALESCED.
            # ----------------------------------------------------------------
            gate_up_tile = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=gate_up_tile,
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # Tile-1 buffers: zero-padded 128-col views for the partial 64-col tile.
            # gate cols 128:192 (I1=64 valid) → gate_t1_128 cols 0:64 valid, 64:128 zero.
            # up   cols 320:384 (I1=64 valid) → up_t1_128   cols 0:64 valid, 64:128 zero.
            # Pad region (cols I1:I0 = 64:128) was zeroed once before k-loop (Plan A).
            # SBUF tensor_copy: 64 valid cols from gate_up_tile into tile-1 buffers.
            # gate tile 1: cols I0:I = 128:192 in the flat combined buffer
            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_tile[0:_PMAX, 0:H_free, nl.ds(I0, I1)],
            )
            # up tile 1: cols I+I0:I+I = 320:384 in the flat combined buffer
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_tile[0:_PMAX, 0:H_free, nl.ds(I + I0, I1)],
            )

            # ----------------------------------------------------------------
            # DMA 2a: down tile 0 — I rows 0:128, full H=2048 (coalesced)
            # down_w [E, I=192, H=2048]: element (e, i, h)
            #   2D AP: P-stride = H = 2048 (adjacent I rows are 2048 apart)
            #          count_p = I0 = 128, col stride = 1, count_col = H = 2048
            # stride_p = H = 2048 = count_col = 2048 → FULLY COALESCED.
            # ----------------------------------------------------------------
            nisa.dma_copy(
                dst=down_full0,
                src=down_w.ap(
                    pattern=[[H, I0], [1, H]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # ----------------------------------------------------------------
            # DMA 2b: down tile 1 — I rows 128:192 (64 valid), zero-padded to 128
            # Same coalesced pattern; only 64 rows loaded; rows 64:128 zeroed before k-loop (Plan A).
            # offset = I0 * H = 128 * 2048 = 262144 (skip past tile-0 I rows)
            # ----------------------------------------------------------------
            # Pad region (rows I1:I0 = 64:128) was zeroed once before k-loop (Plan A).
            nisa.dma_copy(
                dst=down_full1[0:I1, 0:H],       # only write to rows 0:64
                src=down_w.ap(
                    pattern=[[H, I1], [1, H]],
                    offset=I0 * H,               # skip I0=128 rows into down_w
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # ----------------------------------------------------------------
            # Gate/Up matmul: [H, I] @ [H, T=1] → [I, T] in PSUM
            # PSUM layout: [_PMAX, 4] — cols 0,1 = gate tiles 0,1; cols 2,3 = up tiles 0,1
            # Each h1 loop iteration contributes one H-tile to the partial sum.
            # ----------------------------------------------------------------
            nisa.memset(gate_up_psum, value=0.0)

            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        # Tile 0: K=128 cols taken directly from the combined buffer
                        g_stat = gate_up_tile[0:_PMAX, h1, nl.ds(0, I0)]        # gate cols 0:128
                        u_stat = gate_up_tile[0:_PMAX, h1, nl.ds(I, I0)]         # up   cols 192:320
                    else:
                        # Tile 1: K=128 zero-padded (64 valid + 64 zeros)
                        g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]                  # gate tile 1
                        u_stat = up_t1_128[0:_PMAX, h1, 0:I0]                    # up   tile 1
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

            # Flush gate/up PSUM → SBUF
            gate_sb = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            up_sb   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_sb, op=nl.copy, data=gate_up_psum[0:_PMAX, 0:I_tiles])
            nisa.activation(up_sb,   op=nl.copy, data=gate_up_psum[0:_PMAX, I_tiles:2 * I_tiles])

            # SiLU(gate) * up → inter_f32 [_PMAX, I_tiles]
            # Rows 64:128 of I_tiles col 1 are zero (from tile-1 zero-padding) — correct.
            silu_res  = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(silu_res, op=nl.silu, data=gate_sb)
            nisa.tensor_tensor(inter_f32, silu_res, up_sb, nl.multiply)

            # Cast inter to bf16 for the down matmul
            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            # ----------------------------------------------------------------
            # Down matmul: [I, H] @ [I, T=1] → [H_shard, T] partial sum
            # Uses the full-H down tiles; prg_id selects the H shard at SBUF index time.
            # h1_g = prg_id * H_free_shard + h1_out → global H-tile for this core.
            # Stationary: down_full0/1 [_PMAX, H], slice [_PMAX, nl.ds(h1_g*128, 128)]
            # ----------------------------------------------------------------
            nisa.memset(down_psum, value=0.0)

            for h1_out in nl.affine_range(H_free_shard):
                # h1_g: global H-tile index for this core's prg_id
                h1_g = prg_id * H_free_shard + h1_out
                # Tile 0: I rows 0:128 (all 128 valid)
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_full0[0:_PMAX, nl.ds(h1_g * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                # Tile 1: I rows 128:192 (64 valid, rows 64:128 zero from pre-k-loop memset)
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_full1[0:_PMAX, nl.ds(h1_g * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            # Flush down PSUM → SBUF, scale by expert affinity
            down_result_sb = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free_shard],
                op=nl.copy,
                data=down_psum[0:_PMAX, 0:H_free_shard],
            )
            down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],
            )

            # Accumulate into output_temp (k==0: copy to initialise; k>0: add)
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
    """Run kernel_v16a with native weight layouts — no preprocessing required.

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
