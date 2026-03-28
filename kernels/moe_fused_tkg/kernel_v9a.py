"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

kernel_v9a — I-dimension sharding + Shared-HBM all-reduce
==========================================================

Replaces sendrecv (v8a) with I-dimension sharding and a shared-HBM all-reduce.

v8a LNC strategy:
  Core 0: H[0:1024], Core 1: H[1024:2048] -> partial gate/up -> sendrecv -> full gate/up
  -> down via H_shard -> each core writes its H_shard of output

v9a LNC strategy:
  Core 0: I[0:128],   Core 1: I[128:256]  -> each core computes its own I-shard slice
  -> down via FULL H (no H-shard) -> each core stores full-H partial to shared HBM
  -> core_barrier -> each core loads both partials for its H_shard, adds, casts bf16
  -> writes final H_shard to output

Benefits:
  - No sendrecv: avoids compiler bug that breaks >1 MoE layers in a model graph
  - Shared-HBM all-reduce via core_barrier is compiler-safe across multiple layers

Preserved optimizations from v8a:
  1. Batch RMSNorm loads (single 3D DMA for inp and gamma) [v4a]
  2. Pre-broadcast affinity weights into aff_bcast[128, K] [v5a]
  3. 3D DMA pattern for weight loads (Plan D) — adapted to I-shard shapes
  4. Two-pass K-loop with affine_range on Pass 2 [v3b]
  5. Copy-for-k0 in Pass 3 (no memset) [v1b]

Constants change:
  _I_TILES_SHARD = _I_TILES // _N_PRGS  = 1 (each core handles 1 I-tile of 128 elements)

SBUF budget estimate (per core):
  inp_flat_sb:         [16, 128] bf16         =  4 KB
  gamma_flat_sb:       [16, 128] bf16         =  4 KB
  rmsnorm_out:         [128, 16] bf16         =  4 KB
  rmsnorm_normed_bf16: [128, 16] bf16         =  4 KB
  output_temp:         [128, 16, 1] fp32      = 64 KB   (full H, T=1)
  inter_buf:           [128, 1, 8] bf16       =  2 KB
  down_buf:            [128, 16, 8] fp32      = 64 KB   (full H_free, K=8)
  fused_tile_gate:     [128, 16, 128] bf16    = 64 KB  (per iteration, reused)
  fused_tile_up:       [128, 16, 128] bf16    = 64 KB  (per iteration, reused)
  down_tile:           [128, 16, 128] bf16    = 64 KB  (per iteration, reused)
  aff_bcast:           [128, 8] fp32          =  4 KB
  out_fp32_sb:         [1, 2048] fp32         =  8 KB  (stage 5)
  out_sb:              [1, 1024] bf16         =  2 KB  (stage 5)
  Various small SBUF tensors...
  Total peak: ~200 KB — within 224 KB limit (gate+up tiles don't coexist with down tile)
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import core_barrier
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
_H_FREE = _H // _PMAX           # = 16 tiles of 128 each
_H_FREE_SHARD = _H_FREE // _N_PRGS   # = 8 (each core owns 8 H-tiles for output)
_H_SHARD = _H_FREE_SHARD * _PMAX     # = 1024
_I0 = _PMAX      # = 128
_I_TILES = _I // _I0             # = 2
_I_TILES_SHARD = _I_TILES // _N_PRGS  # = 1 (each core handles 1 I-tile)


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
    Returns: (output [T, H=2048] bf16, partial_hbm [N_PRGS, H_free, _PMAX] fp32)
    Callers should use outputs[0]; partial_hbm is an all-reduce workspace returned
    for correct HBM allocation (shared_hbm tensors must be returned).
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
    I_tiles_shard = _I_TILES_SHARD  # = 1

    # LNC program ID (0 or 1 for LNC=2)
    prg_id = nl.program_id(axis=0)

    # -----------------------------------------------------------------------
    # Stage 1: RMSNorm
    # Input: inp [B, 1, H] -> flatten to [T, H] -> SBUF [128, H_free*T]
    # Output: rmsnorm_out [128, H_free*T] (full H, both cores identical)
    #
    # Plan A: Single DMA for inp and gamma (replaces 16x tiny DMAs each).
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

    # Plan A: Single DMA for gamma
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

    # RMSNorm computation — all intermediate tensors updated to 2D [_PMAX, H_free*T]

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
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)
    # Cast to inp.dtype (bf16) on load — router_w may be fp32 in e2e model,
    # but nc_matmul requires both inputs to share the same bit-width.
    router_w_tile_sb = nl.ndarray((_PMAX, E), dtype=inp.dtype, buffer=nl.sbuf)

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
    # v9a I-shard strategy:
    #   Core prg_id handles I-shard: I rows [prg_id*I0 : prg_id*I0+I0] (I0=128)
    #   - Gate/Up: load only I-shard columns from full H -> partial intermediate
    #   - No sendrecv needed: each core's partial is independent for down proj
    #   - Down: load I-shard rows of down weights, project onto FULL H
    #   - output_temp covers FULL H_free (not H_free_shard)
    #
    # Fuse gate/up weight dims: [E, H, 2, I] -> [E, H, I*2]
    # Gate cols: [0:I], Up cols: [I:I*2]
    # Core prg_id: gate cols [prg_id*I0 : prg_id*I0+I0], up cols [I+prg_id*I0 : I+prg_id*I0+I0]
    # -----------------------------------------------------------------------

    E_shape, H_shape, _, I_shape = gate_up_w.shape
    gate_up_w_fused = gate_up_w.reshape((E_shape, H_shape, I_shape * 2))

    # output_temp covers FULL H (H_free=16 tiles), not just H_free_shard=8
    # No memset — k=0 uses tensor_copy to initialise (avoids add-to-zero)
    output_temp = nl.ndarray((_PMAX, H_free, T), dtype=nl.float32, buffer=nl.sbuf)

    # K-slot intermediate and down buffers (v3b two-pass pattern)
    # inter_buf: 1 I-tile per core (I_tiles_shard=1)
    # down_buf: covers FULL H_free (H_free=16 tiles)
    inter_buf = nl.ndarray((_PMAX, I_tiles_shard, K), dtype=inp.dtype, buffer=nl.sbuf)
    down_buf  = nl.ndarray((_PMAX, H_free, K), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.static_range(T):

        # ── Pre-broadcast all K affinity weights ────────────────────────────────
        # norm_weights [T, K]: T in partition dim, K in free dim.
        # Broadcast partition 0's K weights to all 128 partitions once.
        aff_bcast = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K], src=norm_weights[t:t + 1, 0:K])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        # ── PASS 1: gate/up for ALL K experts (I-shard) ──────────────────────────
        # Each core loads only its I-shard columns from gate/up weights.
        # No sendrecv: core prg_id produces intermediate for I[prg_id*I0 : prg_id*I0+I0].
        for k in nl.static_range(K):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            # 3D DMA for gate weights (I-shard):
            # gate_up_w_fused shape: [E=128, H=2048, I*2=512]
            # Load gate cols for this core: fused col offset = prg_id * I0
            # AP pattern (3D):
            #   dim 0 (p):   stride = I*2 = 512,      count = _PMAX = 128
            #   dim 1 (h1):  stride = _PMAX*I*2 = 65536, count = H_free = 16
            #   dim 2 (col): stride = 1,               count = I0 = 128
            # offset = prg_id * I0  (within the 512-wide fused dim)
            fused_tile_gate = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=fused_tile_gate,
                src=gate_up_w_fused.ap(
                    pattern=[[I * 2, _PMAX], [_PMAX * I * 2, H_free], [1, I0]],
                    offset=prg_id * I0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # 3D DMA for up weights (I-shard):
            # Up cols offset = I + prg_id * I0 (past all I gate cols)
            fused_tile_up = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=fused_tile_up,
                src=gate_up_w_fused.ap(
                    pattern=[[I * 2, _PMAX], [_PMAX * I * 2, H_free], [1, I0]],
                    offset=I + prg_id * I0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # PSUM for gate/up (only 1 I-tile per core)
            gate_psum = nl.ndarray((_PMAX, I_tiles_shard), dtype=nl.float32, buffer=nl.psum)
            up_psum   = nl.ndarray((_PMAX, I_tiles_shard), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(gate_psum, value=0.0)
            nisa.memset(up_psum,   value=0.0)

            # affine_range enables DMA-compute overlap for h1 iterations
            for h1 in nl.affine_range(H_free):
                nisa.nc_matmul(
                    dst=gate_psum[0:_PMAX, 0:I_tiles_shard],
                    stationary=fused_tile_gate[0:_PMAX, h1, 0:I0],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                )
                nisa.nc_matmul(
                    dst=up_psum[0:_PMAX, 0:I_tiles_shard],
                    stationary=fused_tile_up[0:_PMAX, h1, 0:I0],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                )

            # Flush PSUM -> SBUF
            gate_sb = nl.ndarray((_PMAX, I_tiles_shard), dtype=nl.float32, buffer=nl.sbuf)
            up_sb   = nl.ndarray((_PMAX, I_tiles_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_sb, op=nl.copy, data=gate_psum)
            nisa.activation(up_sb,   op=nl.copy, data=up_psum)

            # SiLU activation + element-wise multiply (no sendrecv needed)
            silu_res  = nl.ndarray((_PMAX, I_tiles_shard), dtype=nl.float32, buffer=nl.sbuf)
            inter_f32 = nl.ndarray((_PMAX, I_tiles_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(silu_res, op=nl.silu, data=gate_sb)
            nisa.tensor_tensor(inter_f32, silu_res, up_sb, nl.multiply)

            # Store SiLU output to K-slot buffer for Pass 2
            nisa.activation(
                inter_buf[0:_PMAX, 0:I_tiles_shard, k:k + 1],
                op=nl.copy,
                data=inter_f32,
            )

        # ── PASS 2: down projection for ALL K experts — affine_range enables pipelining ──
        # Each core owns I_tiles_shard=1 I-tile, projects onto FULL H (H_free=16 tiles).
        # Independent across k: compiler overlaps expert k+1 DMA with expert k matmuls.
        for k in nl.affine_range(K):
            expert_id_k = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            # 3D DMA for down weights (I-shard):
            # down_w shape: [E=128, I=256, H=2048]
            # Core prg_id needs rows [prg_id*I0 : prg_id*I0+I0] of I dimension
            # Load: down_tile[I0=128, H_free=16, H0=128]
            # Element down_tile[i0, h1, h_p] -> down_w[expert, prg_id*I0 + i0, h1*128 + h_p]
            # AP pattern (3D):
            #   dim 0 (i0):  stride = H = 2048, count = I0 = 128
            #   dim 1 (h1):  stride = _PMAX = 128, count = H_free = 16
            #   dim 2 (h_p): stride = 1, count = _PMAX = 128
            # offset = prg_id * I0 * H  (skip to this core's I rows)
            down_tile = nl.ndarray((I0, H_free, _PMAX), dtype=down_w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=down_tile,
                src=down_w.ap(
                    pattern=[[H, I0], [_PMAX, H_free], [1, _PMAX]],
                    offset=prg_id * I0 * H,
                    scalar_offset=expert_id_k,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # Persistent PSUM covering all H_free output tiles
            # H_free=16 output columns (T=1) fit in one PSUM bank (_PSUM_FREE=512)
            down_psum = nl.ndarray((_PMAX, _PSUM_FREE), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(down_psum, value=0.0)

            # Load inter for this expert from K-slot buffer
            inter_bf16_k = nl.ndarray((_PMAX, I_tiles_shard), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=inter_bf16_k, src=inter_buf[0:_PMAX, 0:I_tiles_shard, k:k + 1])

            # Matmul loop over all H_free output tiles (static_range, persistent PSUM)
            # stationary = down_tile[I0, h1, _PMAX] -> view [_PMAX, I0] transposed
            # moving     = inter_bf16_k [_PMAX, I_tiles_shard=1]
            # PSUM[_PMAX, h1_out*T] accumulates over i0 dimension
            for h1_out in nl.static_range(H_free):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, nl.ds(h1_out * T, T)],
                    stationary=down_tile[0:I0, h1_out, 0:_PMAX],
                    moving=inter_bf16_k[0:_PMAX, 0:I_tiles_shard],
                )

            # Single PSUM flush — covers all H_free columns
            down_result_sb = nl.ndarray((_PMAX, H_free), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free],
                op=nl.copy,
                data=down_psum[0:_PMAX, 0:H_free],
            )

            # Scale by affinity weight (pre-broadcast)
            down_result_scaled = nl.ndarray((_PMAX, H_free), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],
            )
            # Store scaled result to K-slot buffer for Pass 3 accumulation
            nisa.tensor_copy(
                dst=down_buf[0:_PMAX, 0:H_free, k:k + 1],
                src=down_result_scaled[0:_PMAX, 0:H_free],
            )

        # ── PASS 3: accumulate K down results into output_temp ─────────────────
        # Copy-for-k0 optimization: no memset needed.
        for k in nl.static_range(K):
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
    # Stage 5: Shared-HBM All-reduce
    # Each core has partial sums for FULL H; sum via shared HBM + core_barrier.
    # Each core writes its H_shard of the final output.
    #
    # partial_hbm: [N_PRGS=2, H_free=16, _PMAX=128] fp32 allocated as shared_hbm
    # and returned (NKI requires shared_hbm tensors to be returned from the kernel
    # for the compiler to assign a valid HBM address).
    # Layout keeps partition=_PMAX=128 for DMA compatibility with output_temp.
    # output_temp[_PMAX, h1, 0:T=1] -> partial_hbm[prg_id, h1, 0:_PMAX] directly.
    # -----------------------------------------------------------------------

    # Allocate shared HBM partial buffer — must be returned for address assignment
    partial_hbm = nl.ndarray((_N_PRGS, H_free, _PMAX), dtype=nl.float32, buffer=nl.shared_hbm)

    # Step 1: Store output_temp tiles to this core's slot in shared HBM.
    # output_temp[0:_PMAX, h1, 0:T=1] is [_PMAX, 1] fp32 — store to partial_hbm[prg_id, h1, :]
    for h1 in nl.static_range(H_free):
        nisa.dma_copy(
            dst=partial_hbm[prg_id, h1, 0:_PMAX],
            src=output_temp[0:_PMAX, h1, 0:T],
            dge_mode=3,
        )

    # Step 2: Sync — wait for both cores to write their partials
    core_barrier(partial_hbm, cores=[0, 1])

    # Step 3: Each core reduces its H_shard slice: load both partials, add, cast bf16, store
    # Core prg_id handles H-tiles [prg_id*H_free_shard : (prg_id+1)*H_free_shard]
    output = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.shared_hbm)
    out_sb = nl.ndarray((T, H_shard), dtype=inp.dtype, buffer=nl.sbuf)

    for h1 in nl.static_range(H_free_shard):
        h1_global = prg_id * H_free_shard + h1

        # Load tile from core 0's partial: [_PMAX, T=1]
        partial0_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=partial0_sb[0:_PMAX, 0:T],
            src=partial_hbm[0, h1_global, 0:_PMAX],
            dge_mode=3,
        )
        # Load tile from core 1's partial: [_PMAX, T=1]
        partial1_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=partial1_sb[0:_PMAX, 0:T],
            src=partial_hbm[1, h1_global, 0:_PMAX],
            dge_mode=3,
        )
        # Add partials (fp32) and transpose to [T, _PMAX] for output layout
        sum_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(sum_sb, partial0_sb, partial1_sb, nl.add)
        # Transpose [_PMAX, T] -> PSUM [T, _PMAX], then cast fp32->bf16
        tp_psum = nl.ndarray((T, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp_psum[0:T, 0:_PMAX], data=sum_sb[0:_PMAX, 0:T])
        nisa.activation(
            dst=out_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)],
            op=nl.copy,
            data=tp_psum[0:T, 0:_PMAX],
        )

    nisa.dma_copy(
        dst=output[0:T, nl.ds(prg_id * H_shard, H_shard)],
        src=out_sb[0:T, 0:H_shard],
    )

    return output, partial_hbm


def run(inp, gamma, router_w, gate_up_w, down_w):
    """
    Run the fused Qwen3 MoE TKG kernel with LNC=2.

    Returns the expert MLP output after routing and accumulation.
    The kernel also returns a partial_hbm workspace tensor which is discarded here.
    """
    outputs = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
