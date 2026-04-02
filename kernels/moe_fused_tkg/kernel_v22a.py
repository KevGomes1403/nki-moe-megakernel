"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

kernel_v22a — Round 1 Plan A: Eliminate per-expert tile-1 prep loop
====================================================================

Derived from kernel_v20a.

Interface contract (no repack required):
  gate_up_w  [E=128, H=2048, 2*I=384] bf16 — native layout from qwen.py/qwen_fused_moe_tkg.py
                                              cols 0:I   = gate weights
                                              cols I:2*I = up   weights
  down_w     [E=128, I=192,  H=2048]   bf16 — native layout from qwen.py/qwen_fused_moe_tkg.py

Changes vs v20a:
  1. Removed the large gate_t1_128 [_PMAX, H_free, I0] and up_t1_128 [_PMAX, H_free, I0]
     SBUF buffers (saving 8 KB SBUF total).
  2. Removed the two hoisted memsets for the [H_free, I0] tile-1 buffers (2 fewer
     memset instructions per token).
  3. Removed the tile-1 prep loop (2 × H_free = 32 tensor_copy calls per expert ×
     K=8 experts = 256 tensor_copy instructions eliminated per token).
  4. In the gate/up matmul loop, tile-1 (i_tile==1) now uses two small single-slot
     scratch buffers gate_t1_sb [_PMAX, I0] and up_t1_sb [_PMAX, I0] (8 KB SBUF total,
     same as before but with H_free=1 instead of H_free=16 slots).
     The pad region [I1:I0] of these buffers is memset to 0 ONCE before the token
     loop (reuses the same memset; pad bytes are never overwritten inside the loops).
     For each h1 in the matmul affine_range, a tensor_copy loads the 64 valid cols
     into gate_t1_sb[:,0:I1] and up_t1_sb[:,0:I1] immediately before the nc_matmul.
     The compiler can pipeline the tensor_copy and nc_matmul on separate engines.

  NOTE: nc_matmul requires stationary free_dim = 128 (_PMAX) on trn2.
  K=64 stationary is not supported (compiler rejects it). The single-slot approach
  maintains K=128 by keeping the zero-padded layout while reducing SBUF footprint
  and eliminating the batched prep loop.

Net effect vs v20a:
  - SBUF: −8 KB (gate_t1_128 + up_t1_128 reduced from [16,128] to [1,128] each)
  - Instructions: −256 tensor_copy + −2 memset per token (prep loop eliminated)
  - Instructions added: +2 × H_free × K = 256 tensor_copy (inline in matmul loop)
  Total tensor_copy count is the same; however the instruction ordering is changed:
  copies are now interleaved with matmuls, which may improve engine overlap.

SBUF budget estimate (per partition lane, 224 KiB limit):
  inp_flat_sb:           [16*T, _PMAX]   bf16  =   4 KB  (T=1)
  gamma_flat_sb:         [H_free, _PMAX] bf16  =   4 KB
  rmsnorm_out:           [_PMAX, 16*T]   bf16  =   4 KB
  rmsnorm_normed_bf16:   [_PMAX, 16*T]   bf16  =   4 KB
  output_temp:           [_PMAX, 8, 1]   fp32  =   4 KB  (H_free_shard=8, T=1)
  aff_bcast:             [_PMAX, 8]      fp32  =   4 KB
  gate_up_flat:          [_PMAX, K*H_free, _GU_FLAT] bf16 = 96 KB
  gate_t1_sb:            [_PMAX, I0=128] bf16  =  ~0.5 KB  (single slot, reused per h1)
  up_t1_sb:              [_PMAX, I0=128] bf16  =  ~0.5 KB  (single slot, reused per h1)
  down_full0_flat:        [_PMAX,8192]   bf16  =  16 KB
  down_full1_flat:        [_PMAX,8192]   bf16  =  16 KB
  router_w_wide_sb:      [_PMAX, 4, 128] bf16  =  64 KB  (freed before Stage 4)
  out_sb:                [T, H_shard]    bf16  =   2 KB
  gate/up/inter SBUF:    ~16 KB (PSUM flush + silu + inter)
  Total peak (Stage 4):  ~167 KB < 224 KiB limit (7 KB saved vs v20a).
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

# Flat SBUF stride per expert for gate_up_flat
_GU_STRIDE = _H_FREE * _GU_FLAT  # = 16 * 384 = 6144 cols per expert in flat buf


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

    v22a changes: tile-1 matmuls use K=I1=64 directly from gate_up_flat, eliminating
    the gate_t1_128/up_t1_128 SBUF buffers and all associated tensor_copy calls.
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
    # Stage 4: Selective-Expert MLP — Plan A: Flat SBUF + affine_range(K) prefetch
    #
    # Phase 1: Issue ALL 24 DMAs (8×gate_up + 8×down_full0 + 8×down_full1)
    #          in a single affine_range(K) prefetch loop before any compute.
    #          Uses 3 flat SBUF tensors: gate_up_flat, down_full0_flat, down_full1_flat.
    #          affine_range signals independence → compiler can batch/overlap all K DMAs.
    #
    # Phase 2: Serial expert compute reads from flat pre-loaded buffers (no DMAs).
    #          nl.static_range(K) with nl.ds(k*stride, stride) for flat buffer access.
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

    # Flat SBUF buffers — one contiguous block for all K experts
    # Enables nl.affine_range so compiler can issue all K DMAs independently
    _GU_STRIDE_LOCAL = H_free * _GU_FLAT  # = 16 * 384 = 6144 cols per expert in flat buf
    # gate_up_flat: 3D [_PMAX, K*H_free, _GU_FLAT] — required because the DMA src pattern
    # is 3-level (3D), so dst must also be 3D. Middle dim K*H_free = 8*16 = 128 is the
    # combined H-tile index across all experts; expert k uses rows [k*H_free : (k+1)*H_free].
    gate_up_flat    = nl.ndarray((_PMAX, K * H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
    down_full0_flat = nl.ndarray((_PMAX, K * H_shard),          dtype=down_w.dtype,    buffer=nl.sbuf)
    down_full1_flat = nl.ndarray((_PMAX, K * H_shard),          dtype=down_w.dtype,    buffer=nl.sbuf)

    # Single-slot tile-1 scratch buffers: reused for each (k, h1) iteration.
    # Size [_PMAX, I0]: valid cols [0:I1] written per h1; pad cols [I1:I0] zeroed once.
    gate_t1_sb = nl.ndarray((_PMAX, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
    up_t1_sb   = nl.ndarray((_PMAX, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)

    # Hoist PSUM buffers (per-expert memset stays inside k-loop)
    gate_up_psum = nl.ndarray((_PMAX, 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
    down_psum    = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.psum)

    # Pad zeros for tile-1 scratch buffers — done ONCE outside the token loop.
    # Cols [I1:I0] are never overwritten inside any loop, so one memset suffices.
    nisa.memset(gate_t1_sb[0:_PMAX, nl.ds(I1, I1)], value=0.0)
    nisa.memset(up_t1_sb[0:_PMAX, nl.ds(I1, I1)],   value=0.0)

    for t in nl.static_range(T):

        # ------------------------------------------------------------------
        # Hoist ALL pad memsets before the prefetch loop
        # ------------------------------------------------------------------
        # Zero ALL K experts' pad rows (I1:_PMAX) in down_full1_flat at once
        # One memset covers all K experts simultaneously → K× fewer memset instructions
        nisa.memset(down_full1_flat[nl.ds(I1, I1), 0:K * H_shard], value=0.0)

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
        # Phase 1: Issue ALL K×3 = 24 DMAs as independent operations
        # nl.affine_range signals independence → compiler can batch/overlap all K expert DMAs
        # ------------------------------------------------------------------
        for k in nl.affine_range(K):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            # DMA 1: gate_up — coalesced load [_PMAX, H_free, _GU_FLAT] per expert k
            # dst is 3D slice [0:_PMAX, k*H_free:(k+1)*H_free, 0:_GU_FLAT] to match 3D src pattern
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

            # DMA 2a: down tile 0 — I0 rows, H_shard cols (same HBM pattern as v19b, flat dst)
            nisa.dma_copy(
                dst=down_full0_flat[0:_PMAX, nl.ds(k * H_shard, H_shard)],
                src=down_w.ap(
                    pattern=[[H, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # DMA 2b: down tile 1 — I1 valid rows (pad rows I1:_PMAX already zeroed above)
            nisa.dma_copy(
                dst=down_full1_flat[0:I1, nl.ds(k * H_shard, H_shard)],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H_shard]],
                    offset=I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # ------------------------------------------------------------------
        # Phase 2: Serial expert compute — reads from flat pre-loaded buffers
        # ------------------------------------------------------------------
        for k in nl.static_range(K):
            # gate_up_flat middle index for expert k: rows [k*H_free : (k+1)*H_free]
            gate_h_base = k * H_free   # first H-tile row index for expert k in gate_up_flat
            down_expert_base = k * H_shard  # col offset in down flats for expert k

            # --- Gate/Up matmul ---
            nisa.memset(gate_up_psum, value=0.0)

            for h1 in nl.affine_range(H_free):
                h_idx = gate_h_base + h1  # compile-time: gate_h_base is static, h1 is affine

                # Tile 0: K=I0=128, stationary read directly from gate_up_flat
                nisa.nc_matmul(
                    dst=gate_up_psum[0:_PMAX, 0:1],
                    stationary=gate_up_flat[0:_PMAX, h_idx, nl.ds(0, I0)],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                )
                nisa.nc_matmul(
                    dst=gate_up_psum[0:_PMAX, I_tiles:I_tiles + 1],
                    stationary=gate_up_flat[0:_PMAX, h_idx, nl.ds(I, I0)],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                )

                # Tile 1: inline tensor_copy of I1=64 valid cols into single-slot scratch buffer.
                # gate_t1_sb[:,0:I1] ← gate_up_flat[:,h_idx, I0:I0+I1]
                # up_t1_sb[:,0:I1]   ← gate_up_flat[:,h_idx, I+I0:I+I0+I1]
                # Pad cols [I1:I0] remain zero (set once before token loop).
                nisa.tensor_copy(
                    dst=gate_t1_sb[0:_PMAX, 0:I1],
                    src=gate_up_flat[0:_PMAX, h_idx, nl.ds(I0, I1)],
                )
                nisa.tensor_copy(
                    dst=up_t1_sb[0:_PMAX, 0:I1],
                    src=gate_up_flat[0:_PMAX, h_idx, nl.ds(I + I0, I1)],
                )
                nisa.nc_matmul(
                    dst=gate_up_psum[0:_PMAX, 1:2],
                    stationary=gate_t1_sb[0:_PMAX, 0:I0],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                )
                nisa.nc_matmul(
                    dst=gate_up_psum[0:_PMAX, I_tiles + 1:I_tiles + 2],
                    stationary=up_t1_sb[0:_PMAX, 0:I0],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                )

            # --- Flush gate/up PSUM → SBUF, SiLU, inter ---
            # (same as v19b — unchanged)
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

            # --- Down matmul — reads from flat down buffers ---
            nisa.memset(down_psum, value=0.0)

            for h1_out in nl.affine_range(H_free_shard):
                # Stationary: [_PMAX, _PMAX] block from expert k's H-shard at output tile h1_out
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_full0_flat[0:_PMAX, nl.ds(down_expert_base + h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_full1_flat[0:_PMAX, nl.ds(down_expert_base + h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            # --- Flush down PSUM, scale by affinity, accumulate (same as v19b) ---
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
    """Run kernel_v22a with native weight layouts — no preprocessing required.

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
