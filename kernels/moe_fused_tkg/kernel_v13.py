"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

kernel_v13 — I=192 native support (no zero-padding required)
=============================================================

Changes from v10b:
  1. _I = 192 (actual intermediate dim per TP rank, was 256).
     I=192 = 128 + 64 → tile 0: full 128 rows, tile 1: partial 64 rows.
  2. Gate/up: separate 3D SBUF buffers gate_tile_3d and up_tile_3d, each
     [_PMAX, H_free, 256]. Each is loaded with 192 cols via DMA; the
     remaining 64 cols are memset to zero. This gives tile-1 matmul a
     128-wide stationary with cols 128:191 valid and 192:255 zero. ✓
     nc_matmul requires the stationary free dim = _PMAX = 128 exactly.
  3. Down projection: two separate 2D DMAs instead of one 3D DMA.
     down_tile1 is [128, H_shard] with rows 0:63 loaded, rows 64:127
     memset to zero (nc_matmul partition dim must be 128).
  4. run() slices gate_up_w to [:,:,:,:192] and down_w to [:,:192,:] so
     callers holding padded [I=256] tensors still work unchanged.

All other optimizations from v10b are preserved unchanged (router batching,
affine_range K-loops, DGE mode, copy-for-k0, etc.).

SBUF budget estimate (per core):
  inp_flat_sb:         [16, 128] bf16         =  4 KB
  gamma_flat_sb:       [16, 128] bf16         =  4 KB
  rmsnorm_out:         [128, 16] bf16         =  4 KB
  rmsnorm_normed_bf16: [128, 16] bf16         =  4 KB
  output_temp:         [128, 8, 1] fp32       = 32 KB   (H_free_shard=8, T=1)
  inter_buf:           [128, 2, 8] bf16       =  4 KB   (I_tiles=2, K=8)
  down_buf:            [128, 8, 8] fp32       = 32 KB   (H_free_shard=8, K=8)
  gate_tile_3d:        [128, 16, 256] bf16    = 64 KB   (full H, I_PAD=256, reused per expert)
  up_tile_3d:          [128, 16, 256] bf16    = 64 KB   (full H, I_PAD=256, reused per expert)
  down_tile0/1:        [128, 1024] bf16       = 32 KB   each (H_shard=1024, reused)
  aff_bcast:           [128, 8] fp32          =  4 KB
  router_w_wide_sb:    [128, 512] bf16        = 128 KB  (4 tiles × 128 cols, router batch)
  out_sb:              [1, 1024] bf16         =  2 KB
  Note: gate/up tiles and down tiles don't coexist; router_w_wide_sb reused across chunks.
  Total peak: ~160 KB — within 224 KB limit.
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
_I = 192     # actual intermediate dim per TP rank (no padding)
_I0 = 128    # first tile (full 128 rows)
_I1 = 64     # second tile (partial, only 64 valid rows)
_I_TILES = 2 # still 2 tiles
_EPS = 1e-6

# LNC=2 sharding constants (hardcoded, always launched with [2])
_N_PRGS = 2
_H_FREE = _H // _PMAX           # = 16 tiles of 128 each
_H_FREE_SHARD = _H_FREE // _N_PRGS   # = 8 (each core owns 8 H-tiles for output)
_H_SHARD = _H_FREE_SHARD * _PMAX     # = 1024

# Router DMA batching: 4 tiles per DMA -> 4x32KB=128KB per packet
_ROUTER_BATCH = 4  # tiles per DMA batch; H_FREE must be divisible by this


@nki.jit
def qwen3_moe_fused_tkg(
    inp,        # [B, 1, H=2048] bf16
    gamma,      # [1, H=2048] bf16
    router_w,   # [H=2048, E=128] bf16
    gate_up_w,  # [E=128, H=2048, 2, I=192] bf16
    down_w,     # [E=128, I=192, H=2048] bf16
):
    """
    Fused RMSNorm + Router + TopK(K=8) + Expert MLP for Qwen3 MoE TKG.
    Invoked as: qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
    where [2] means LNC=2 (two cores).
    Returns: output [T, H=2048] bf16
    Each core writes its H_shard independently; no synchronization needed.
    gate_up_w is [E, H, 2, I=192] — no padding; last dim is 192 exactly.
    down_w is [E, I=192, H] — no padding; I dim is 192 exactly.
    """
    B = inp.shape[0]
    T = B  # seq_len=1, so tokens = batch

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

    # LNC program ID (0 or 1 for LNC=2)
    prg_id = nl.program_id(axis=0)

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
    # Router weight batching — 4 tiles per DMA (preserved from v10b).
    # 16 separate 32KB DMAs -> 4 x 128KB DMAs.
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
        # 4 matmuls sharing the loaded wide tile
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
    # Stage 4: Selective-Expert MLP
    #
    # Gate/Up: each core loads FULL H of gate/up weights (H_free=16 tiles).
    #          Each core independently computes full intermediate [128, I_tiles=2].
    # Down: each core loads only H_shard=1024 columns (H_free_shard=8 tiles).
    #       Each core writes output[:, prg_id*H_shard:(prg_id+1)*H_shard].
    #
    # I=192 gate/up strategy:
    #   Separate 3D SBUF buffers for gate and up, each shape [_PMAX, H_free, I0*2=256].
    #   Zeroed once per expert; then loaded with I=192 actual columns (first 192 of 256).
    #   Columns 192:255 remain zero — giving correct zero-padded tile 1 for nc_matmul.
    #   Tile 0 uses nl.ds(0, 128), tile 1 uses nl.ds(128, 128) — both 128-wide. ✓
    #   gate_up_w_fused [E, H, I*2=384]: gate at cols 0:192, up at cols 192:384.
    # -----------------------------------------------------------------------

    E_shape, H_shape, _, I_shape = gate_up_w.shape
    gate_up_w_fused = gate_up_w.reshape((E_shape, H_shape, I_shape * 2))

    # Width of each padded gate/up 3D buffer: round up to I0*2 = 256.
    # This equals the v10b fused buffer half-width, enabling tile 0+1 = 128+128 slices.
    _I_PAD = _I0 * 2  # = 256

    # output_temp: H_free_shard tiles (this core's H output shard), fp32
    # No memset — k=0 uses tensor_copy to initialise (avoids add-to-zero)
    output_temp = nl.ndarray((_PMAX, H_free_shard, T), dtype=nl.float32, buffer=nl.sbuf)

    # K-slot buffers for two-pass pattern
    # inter_buf: full I_tiles=2 per core (no I-shard)
    # down_buf:  H_free_shard output tiles
    inter_buf = nl.ndarray((_PMAX, I_tiles, K), dtype=inp.dtype, buffer=nl.sbuf)
    down_buf  = nl.ndarray((_PMAX, H_free_shard, K), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.static_range(T):

        # -- Pre-broadcast all K affinity weights ----------------------------
        aff_bcast = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K], src=norm_weights[t:t + 1, 0:K])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        # -- PASS 1: gate/up for ALL K experts (full H, no I-shard, no sendrecv) --
        # affine_range(K) allows the compiler to overlap expert k+1's DMA with
        # expert k's matmul (preserved from v10b).
        for k in nl.affine_range(K):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            # Separate zero-padded 3D buffers for gate and up weights.
            # Each: [_PMAX, H_free, _I_PAD=256] with only first I=192 cols loaded.
            # Only the pad cols (192:256) need to be zeroed; loaded cols (0:192) are
            # overwritten by DMA. Zeroing only the pad region reduces memset work 4×.
            gate_tile_3d = nl.ndarray((_PMAX, H_free, _I_PAD), dtype=gate_up_w.dtype, buffer=nl.sbuf)
            up_tile_3d   = nl.ndarray((_PMAX, H_free, _I_PAD), dtype=gate_up_w.dtype, buffer=nl.sbuf)
            # Zero only the pad columns (I=192 to _I_PAD=256, i.e. 64 cols = I1).
            # Cols 0:I=192 are overwritten by the DMA below and need not be zeroed.
            nisa.memset(gate_tile_3d[0:_PMAX, 0:H_free, nl.ds(I, I1)], value=0.0)
            nisa.memset(up_tile_3d[0:_PMAX, 0:H_free, nl.ds(I, I1)],   value=0.0)

            # 3D DMA: gate weights — cols 0:I=192 of gate_up_w_fused per row
            # gate_up_w_fused layout [E, H, 384]: gate=cols 0:192, up=cols 192:384
            # AP pattern (3D):
            #   dim 0 (p):   stride = I*2 = 384,         count = _PMAX = 128
            #   dim 1 (h1):  stride = _PMAX*I*2 = 49152, count = H_free = 16
            #   dim 2 (col): stride = 1,                 count = I = 192
            # dst slice 0:I writes cols 0:191; cols 192:255 remain zero.
            nisa.dma_copy(
                dst=gate_tile_3d[0:_PMAX, 0:H_free, 0:I],
                src=gate_up_w_fused.ap(
                    pattern=[[I * 2, _PMAX], [_PMAX * I * 2, H_free], [1, I]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # 3D DMA: up weights — cols I:I*2=192:384 of gate_up_w_fused per row
            # AP offset = I (skip gate cols in each row of the fused layout).
            nisa.dma_copy(
                dst=up_tile_3d[0:_PMAX, 0:H_free, 0:I],
                src=gate_up_w_fused.ap(
                    pattern=[[I * 2, _PMAX], [_PMAX * I * 2, H_free], [1, I]],
                    offset=I,
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

            # Loop over full H_free=16 (not H_free_shard=8), full I_tiles=2
            # Tile 0: nl.ds(0,   I0=128) → cols 0:127   (128 valid gate/up cols) ✓
            # Tile 1: nl.ds(I0, I0=128) → cols 128:255  (64 valid + 64 zero)    ✓
            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    nisa.nc_matmul(
                        dst=gate_psum[0:_PMAX, i_tile:i_tile + 1],
                        stationary=gate_tile_3d[0:_PMAX, h1, nl.ds(i_tile * I0, I0)],
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=up_psum[0:_PMAX, i_tile:i_tile + 1],
                        stationary=up_tile_3d[0:_PMAX, h1, nl.ds(i_tile * I0, I0)],
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

        # -- PASS 2: down projection for ALL K experts ----------------------
        # Each core loads only its H_shard columns of down weights.
        # Two separate 2D DMAs per expert (tile 0 and tile 1) because I=192
        # is not a multiple of I0=128 (so the 3D stride pattern no longer works).
        #
        # down_w shape: [E=128, I=192, H=2048]
        # Tile 0: down_w rows 0:128   (shape [I0=128, H_shard=1024])
        # Tile 1: down_w rows 128:192 (shape [I1=64, H_shard=1024])
        #         Zero-padded to [I0=128, H_shard=1024] for the matmul.
        for k in nl.affine_range(K):
            expert_id_k = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            # ---- Tile 0 DMA: rows 0:128 of down_w for this expert ----
            # AP pattern (2D):
            #   row (p):   stride = H = 2048,  count = I0 = 128
            #   col (f):   stride = 1,         count = H_shard = 1024
            # offset = prg_id * H_shard (selects this core's H columns)
            down_tile0 = nl.ndarray((_I0, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=down_tile0,
                src=down_w.ap(
                    pattern=[[H, _I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id_k,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # ---- Tile 1 DMA: rows 128:192 of down_w for this expert ----
            # Zero the full tile first, then load only the 64 valid rows.
            # AP pattern (2D):
            #   row (p):   stride = H = 2048,  count = I1 = 64
            #   col (f):   stride = 1,         count = H_shard = 1024
            # offset = I0*H + prg_id*H_shard (skip past tile-0 rows)
            down_tile1 = nl.ndarray((_I0, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
            # Zero only rows I1:I0=64:127; rows 0:64 are overwritten by the DMA below.
            nisa.memset(down_tile1[nl.ds(_I1, _I1), 0:H_shard], value=0.0)
            nisa.dma_copy(
                dst=down_tile1[0:_I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H, _I1], [1, H_shard]],
                    offset=_I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id_k,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # Persistent PSUM covering all H_free_shard output tiles
            down_psum = nl.ndarray((_PMAX, _PSUM_FREE), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(down_psum, value=0.0)

            # Load inter for this expert from K-slot buffer
            inter_bf16_k = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=inter_bf16_k, src=inter_buf[0:_PMAX, 0:I_tiles, k:k + 1])

            # Matmul loop — reads from pre-loaded tile 0 and tile 1
            # down_tile0/1 shape: [I0=128, H_shard=1024]
            # For each h1_out tile, we read columns [h1_out*128 : (h1_out+1)*128].
            # Stationary (weight) shape per call: [I0=128, _PMAX=128].
            # Moving (inter) shape per call: [_PMAX=128, 1].
            # Note: down_tile layout is [I0 rows, H_shard cols], but nc_matmul
            # expects stationary[0:I0, nl.ds(h1_out*_PMAX, _PMAX)] — same as v10b.
            for h1_out in nl.static_range(H_free_shard):
                # Tile 0 contribution
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, nl.ds(h1_out * T, T)],
                    stationary=down_tile0[0:_I0, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16_k[0:_PMAX, 0:1],
                )
                # Tile 1 contribution (last 64 rows; rows 64:127 are zero from memset)
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, nl.ds(h1_out * T, T)],
                    stationary=down_tile1[0:_I0, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16_k[0:_PMAX, 1:2],
                )

            # Single PSUM flush per expert (H_free_shard columns)
            down_result_sb = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free_shard],
                op=nl.copy,
                data=down_psum[0:_PMAX, 0:H_free_shard],
            )

            # Scale by affinity weight (pre-broadcast)
            down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],
            )
            # Store to K-slot buffer for Pass 3
            nisa.tensor_copy(
                dst=down_buf[0:_PMAX, 0:H_free_shard, k:k + 1],
                src=down_result_scaled[0:_PMAX, 0:H_free_shard],
            )

        # -- PASS 3: accumulate K down results into output_temp -------------
        # Copy-for-k0: no memset needed.
        for k in nl.static_range(K):
            if k == 0:
                nisa.tensor_copy(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    src=down_buf[0:_PMAX, 0:H_free_shard, 0:1],
                )
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data1=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data2=down_buf[0:_PMAX, 0:H_free_shard, k:k + 1],
                    op=nl.add,
                )

    # -----------------------------------------------------------------------
    # Stage 5: Transpose, cast fp32->bf16, and store output
    # output_temp [128, H_free_shard, T] (fp32) -> HBM output [T, H] (bf16)
    # Each core writes its H_shard columns at offset prg_id*H_shard.
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
    """
    Run the fused Qwen3 MoE TKG kernel with LNC=2.

    Accepts gate_up_w of shape [E, H, 2, I_padded] where I_padded >= 192,
    and down_w of shape [E, I_padded, H] where I_padded >= 192.
    Slices to I=192 before passing to the kernel so that callers holding
    the standard I_padded=256 tensors work without modification.

    Returns the expert MLP output after routing and accumulation.
    """
    # Slice to actual I=192 (discard zero-padding if present)
    if gate_up_w.shape[-1] != _I:
        gate_up_w = gate_up_w[..., :_I]
    if down_w.shape[1] != _I:
        down_w = down_w[:, :_I, :]

    outputs = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
