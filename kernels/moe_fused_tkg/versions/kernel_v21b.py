"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

kernel_v21b — Plan B: Fuse SiLU into Gate PSUM Flush + Eliminate Down-Tile1 Pad Memsets
==========================================================================================
Based on kernel_v21a (Plan A + Plan F prefetch).

Plan B optimisation (vs v21a):
  Change 1: Fuse gate PSUM flush + SiLU into a single activation(op=nl.silu) call.
    Saves 1 activation call per expert × 8 experts = 8 ops.
  Change 2: Remove the 8 down_full1_bufs pad memsets.
    inter_scaled_bf16[64:128, 1] is provably always zero (gate_t1_128/up_t1_128 cols
    64:128 are zeroed → matmul result rows 64:128 = 0 → silu(0)=0 → inter=0).
    So down_full1_bufs[k][64:128, :] never contributes to the down matmul result.
    Saves 8 memset ops.
  Op delta vs v21a: −16 ops total.

Plan A optimisation (vs v19b):
  Single memset(down_psum, 0) BEFORE the k-loop; pre-scale inter_bf16 by affinity
  inside the k-loop so down matmuls accumulate across all experts without reset.
  After the k-loop: single flush + copy to output_temp.
  Op delta: −21 ops vs v19b.

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

Plan C optimisation (vs v16a):
  Within each k-loop iteration, issue the down_full0 and down_full1 DMAs immediately
  after the gate_up DMA, before the tensor_copy tile-1 prep and before the gate/up
  matmul. The down DMAs have no data dependency on gate_up_tile (they only require
  expert_id, already known). By moving them earlier, the DMA engine can pipeline
  gate_up + down0 + down1 together, allowing the down DMAs to complete during the
  tile-1 tensor_copy prep (~small) and/or the gate/up matmul (~48 μs TensorE window),
  rather than executing serially after the gate/up matmul is done.

Plan F optimisation (vs v18a):
  Issue ALL 24 DMAs (8 gate_up + 8 down_full0 + 8 down_full1) in a single prefetch
  loop BEFORE any compute, so the DMA engine pipelines all 24 packets simultaneously.
  Uses K=8 separate named SBUF buffers (Python lists) for gate_up_bufs,
  down_full0_bufs, down_full1_bufs — each [_PMAX, H_free/H_shard, _GU_FLAT/H_shard].
  First dim is always par_dim=_PMAX=128 (NKI requirement).
  Prefetch loop uses nl.static_range(K) to guarantee compile-time list indexing.
  Compute loop also uses nl.static_range(K) reading from the pre-loaded buffers.

SBUF budget estimate (per partition lane, 224 KiB limit):
  inp_flat_sb:           [16*T, _PMAX]   bf16  =   4 KB  (T=1)
  gamma_flat_sb:         [H_free, _PMAX] bf16  =   4 KB
  rmsnorm_out:           [_PMAX, 16*T]   bf16  =   4 KB
  rmsnorm_normed_bf16:   [_PMAX, 16*T]   bf16  =   4 KB
  output_temp:           [_PMAX, 8, 1]   fp32  =   4 KB  (H_free_shard=8, T=1)
  aff_bcast:             [_PMAX, 8]      fp32  =   4 KB
  gate_up_bufs (8×):     8×[_PMAX,16,384] bf16 =  96 KB
  gate_t1_128:           [_PMAX,16,128]  bf16  =   4 KB  (single, reused across experts)
  up_t1_128:             [_PMAX,16,128]  bf16  =   4 KB  (single, reused across experts)
  down_full0_bufs (8×):  8×[_PMAX,1024]  bf16  =  16 KB
  down_full1_bufs (8×):  8×[_PMAX,1024]  bf16  =  16 KB
  router_w_wide_sb:      [_PMAX, 4, 128] bf16  =  64 KB  (4 tiles × H_free/4)
  out_sb:                [T, H_shard]    bf16  =   2 KB
  gate/up/inter SBUF:    ~16 KB (PSUM flush + silu + inter)
  Total peak:            ~238 KB — note: router_w_wide_sb is freed before Stage 4,
                         peak in Stage 4 ≈ 174 KB < 224 KiB limit.
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
    # Stage 4: Selective-Expert MLP — Plan F: K=8 full prefetch
    #
    # Phase 1: Issue ALL 24 DMAs (8×gate_up + 8×down_full0 + 8×down_full1)
    #          in a single static_range(K) prefetch loop before any compute.
    #          Uses K=8 separate named SBUF buffers accessed via Python lists;
    #          static_range ensures list indices are compile-time constants.
    #
    # Phase 2: Serial expert compute reads from pre-loaded buffers (no DMAs).
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

        # ------------------------------------------------------------------
        # Allocate K=8 named SBUF buffers (par_dim=128 required as first dim)
        # Use explicit named allocations — list comprehensions are not supported
        # inside NKI kernel trace context. Access via Python list of named vars.
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
        # Hoist ALL pad memsets before the prefetch loop
        # ------------------------------------------------------------------
        # Plan B: down_full1_bufs pad memsets removed (provably safe — inter[64:128,1]=0 always)

        # gate_t1_128/up_t1_128: single pair of reused buffers (one per expert, overwritten each k)
        gate_t1_128 = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        up_t1_128   = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        nisa.memset(gate_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
        nisa.memset(up_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        # Hoist PSUM buffers; down_psum zeroed ONCE before k-loop (Plan A)
        gate_up_psum = nl.ndarray((_PMAX, 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
        down_psum    = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(down_psum, value=0.0)

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
        # Phase 1: Issue ALL K=8 × 3 = 24 DMAs in one static_range loop
        # static_range ensures list index is always a compile-time constant.
        # ------------------------------------------------------------------
        for k in nl.static_range(K):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            # DMA 1: gate_up — ONE coalesced load per expert [_PMAX, H_free, _GU_FLAT]
            # stride_p = 384 = count_col = 384 → FULLY COALESCED.
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

            # DMA 2a: down tile 0 — I rows 0:128, H_shard=1024 cols only (shard-aware)
            # down_w [E, I=192, H=2048]: P-stride=H=2048, count_p=I0=128, count_col=H_shard=1024
            # offset = prg_id * H_shard → load only this core's H columns.
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

            # DMA 2b: down tile 1 — I rows 128:192 (64 valid), H_shard=1024 cols only
            # Pad region (rows I1:I0 = 64:128) already zeroed above.
            # offset = I0 * H + prg_id * H_shard → skip tile-0 rows, then shard offset.
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
        # Phase 2: Serial expert compute — no DMAs, reads from pre-loaded buffers
        # ------------------------------------------------------------------
        for k in nl.static_range(K):
            # Tile-1 tensor_copy using pre-loaded gate_up_bufs[k]
            # gate cols 128:192 (I1=64 valid) → gate_t1_128 cols 0:64 valid, 64:128 zero.
            # up   cols 320:384 (I1=64 valid) → up_t1_128   cols 0:64 valid, 64:128 zero.
            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free, nl.ds(I + I0, I1)],
            )

            # ----------------------------------------------------------------
            # Gate/Up matmul: [H, I] @ [H, T=1] → [I, T] in PSUM
            # PSUM layout: [_PMAX, 4] — cols 0,1 = gate tiles 0,1; cols 2,3 = up tiles 0,1
            # ----------------------------------------------------------------
            nisa.memset(gate_up_psum, value=0.0)

            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        # Tile 0: K=128 cols taken directly from the combined buffer
                        g_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]   # gate cols 0:128
                        u_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]   # up   cols 192:320
                    else:
                        # Tile 1: K=128 zero-padded (64 valid + 64 zeros)
                        g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]               # gate tile 1
                        u_stat = up_t1_128[0:_PMAX, h1, 0:I0]                 # up   tile 1
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

            # Plan B Change 1: Fuse gate PSUM flush + SiLU into one activation call.
            # gate_silu_sb = silu(gate_psum) flushed directly; saves 1 activation per expert.
            gate_silu_sb = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            up_sb        = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_silu_sb, op=nl.silu, data=gate_up_psum[0:_PMAX, 0:I_tiles])
            nisa.activation(up_sb,        op=nl.copy, data=gate_up_psum[0:_PMAX, I_tiles:2 * I_tiles])

            # SiLU(gate) * up → inter_f32 [_PMAX, I_tiles]
            # Rows 64:128 of I_tiles col 1 are zero (from tile-1 zero-padding) — correct.
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(inter_f32, gate_silu_sb, up_sb, nl.multiply)

            # Pre-scale inter in fp32 by expert affinity, then cast to bf16 (Plan A)
            # Scale in fp32 to preserve precision before bf16 quantization.
            inter_scaled_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                inter_scaled_f32,
                data=inter_f32,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],
            )
            inter_scaled_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_scaled_bf16, op=nl.copy, data=inter_scaled_f32)

            # ----------------------------------------------------------------
            # Down matmul: [I, H] @ [I, T=1] → accumulate into shared down_psum
            # No per-expert memset; down_psum was zeroed once before the k-loop.
            # Uses pre-loaded down_full0_bufs[k]/down_full1_bufs[k].
            # ----------------------------------------------------------------
            for h1_out in nl.affine_range(H_free_shard):
                # Tile 0: I rows 0:128 (all 128 valid)
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_full0_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_scaled_bf16[0:_PMAX, 0:1],
                )
                # Tile 1: I rows 128:192 (64 valid, rows 64:128 zero from pre-k-loop memset)
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_full1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_scaled_bf16[0:_PMAX, 1:2],
                )

        # Plan A: single flush from accumulated down_psum after all experts
        down_result_sb = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            down_result_sb[0:_PMAX, 0:H_free_shard],
            op=nl.copy,
            data=down_psum[0:_PMAX, 0:H_free_shard],
        )
        nisa.tensor_copy(
            dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
            src=down_result_sb[0:_PMAX, 0:H_free_shard],
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
    """Run kernel_v21b with native weight layouts — no preprocessing required.

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
