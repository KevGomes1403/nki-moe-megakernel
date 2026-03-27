"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

Written from scratch following nkilib reference patterns.
DO NOT import or call nkilib kernel functions (moe_block_tkg, _moe_tkg, etc.)

LNC sharding strategy (hardcoded for LNC=2):
  - RMSNorm: full H, result [128, T, H_free=16] replicated on both cores
  - Router: full H, logits [T, E] replicated on both cores
  - TopK + softmax: identical on both cores
  - Gate/Up proj: each core uses H_SHARD half of hidden [128, T, H_free_shard]
                  -> each core computes partial gate/up [128, I_tiles]
                  -> partial sums exchanged via nisa.sendrecv (SBUF↔SBUF, no HBM)
                  -> full gate/up obtained by adding partial sums
  - Down proj: each core uses only H_shard slice of down weights
               -> core 0 writes output[:, 0:1024], core 1 writes output[:, 1024:2048]

Plan B changes vs v5a — Weight Tile Double-Buffering:
  Root cause: In v5a, fused_tile is declared INSIDE affine_range(H_free_shard) and
  down_w_full is declared INSIDE affine_range(I_tiles). The compiler sees N distinct
  SBUF allocations (one per iteration), preventing it from overlapping DMA for
  iteration i+1 with compute for iteration i.

  Fix: Pre-allocate 2 tiles BEFORE the relevant loops as ping-pong buffers.
  With 2 fixed SBUF regions, the compiler can schedule:
    "DMA tile[1] while TE uses tile[0]" (and vice versa).

  Change 1 — Gate/Up weight tile double-buffering:
    Pre-allocate wtile_0 and wtile_1 before the K-loop.
    Inside the h1 affine_range loop, cycle: h1%2==0 uses wtile_0, else wtile_1.
    With H_free_shard=8: h1=0,2,4,6 use wtile_0; h1=1,3,5,7 use wtile_1.
    Adjacent iterations always use different tiles — no aliasing.

  Change 2 — Down weight tile double-buffering:
    Pre-allocate down_tile_0 and down_tile_1 before the K-loop.
    Inside the i_tile2 affine_range loop, cycle: i_tile2%2==0 uses down_tile_0, else down_tile_1.
    With I_tiles=2: i_tile2=0 uses down_tile_0, i_tile2=1 uses down_tile_1. Perfect double-buffer.

v5a changes (inherited):
  - Batch RMSNorm input/gamma loads (32 tiny DMAs -> 2)
  - Pre-broadcast affinity weights (48 instructions -> 6 per token)
  - v3b two-pass K-loop with affine_range on down projection
  - v2 PSUM accumulation and H-shard gate/up + sendrecv all-reduce
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
    # Plan A change 1: Replace 16-iteration inp load loop (16 DMAs + 16 tensor_copies)
    # with a single DMA + single SBUF-transpose sequence.
    #
    # Strategy:
    #   a) Reshape inp_2d[T, H] -> [H_free*T, _PMAX] in HBM (contiguous, no copy).
    #      Element: [t*H_free+h1, p] = inp_2d[t, h1*128+p]
    #   b) One DMA (no transpose): [H_free*T, _PMAX] -> SBUF [H_free*T, _PMAX]
    #      (same shapes -> single DMA, partition=H_free*T=16, free=_PMAX=128)
    #   c) nc_transpose SBUF [H_free*T, _PMAX] -> PSUM [_PMAX, H_free*T]
    #      (partition 16 becomes free dim, free 128 becomes partition dim)
    #   d) activation copy PSUM -> SBUF: rmsnorm_out [_PMAX, H_free*T]
    #
    # Layout verification after step d:
    #   rmsnorm_out[p, t*H_free+h1] = inp_2d[t, h1*128+p]
    #   Original: inp_tile_sb[p, t] for tile h1 = inp_2d[t, h1*128+p] — same element!
    #   For T=1: rmsnorm_out[p, h1] = inp_2d[0, h1*128+p] ✓
    #
    # Net: 1 DMA replaces 16 DMAs (32 total across both tensors -> 2).
    # -----------------------------------------------------------------------
    inp_2d = inp.reshape((T, H))

    # T=1 (TKG) assumption: H_free*T = H_free throughout this section.
    # Step a+b: Single DMA from HBM [H_free*T, _PMAX] -> SBUF [H_free*T, _PMAX]
    # No transpose needed — same shapes. Replaces 16 x 256B calls with 1 x 4KB call.
    inp_2d_hbm_reshaped = inp_2d.reshape((H_free * T, _PMAX))
    inp_flat_sb = nl.ndarray((H_free * T, _PMAX), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=inp_flat_sb,
        src=inp_2d_hbm_reshaped,
        dge_mode=3,
    )
    # inp_flat_sb[t*H_free+h1, p] = inp_2d[t, h1*128+p]

    # Step c: SBUF transpose [H_free*T, _PMAX] -> PSUM [_PMAX, H_free*T]
    # nc_transpose swaps partition and free: partition(16)->free, free(128)->partition.
    # Output PSUM has partition=_PMAX=128, free=H_free*T=16 — ready for matmul use.
    inp_trans_psum = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)

    # Step d: PSUM [_PMAX, H_free*T] -> SBUF [_PMAX, H_free*T] = rmsnorm_out
    # Plan A change 1: rmsnorm_out is now 2D [_PMAX, H_free*T] instead of 3D [_PMAX, T, H_free].
    rmsnorm_out = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_out[...], op=nl.copy, data=inp_trans_psum[...])
    # rmsnorm_out[p, t*H_free+h1] = inp_2d[t, h1*128+p]
    # For T=1: rmsnorm_out[p, h1] = inp_2d[0, h1*128+p] ✓

    # Plan A change 2: Replace 16-iteration gamma load loop with a single DMA + transpose.
    # Same strategy as inp load above.
    #
    # Layout verification:
    #   gamma_1d[h] where h = h1*128 + p
    #   After reshape to [H_free, _PMAX]: element [h1, p] = gamma_1d[h1*128+p]
    #   After DMA + nc_transpose: gamma_sb[p, h1] = gamma_1d[h1*128+p]
    #   Original: gamma_tile_sb[p, 0] for tile h1 = gamma_1d[h1*128+p] — matches!
    gamma_1d = gamma.reshape((H,))
    # Step a+b: Single DMA [H_free, _PMAX] -> SBUF [H_free, _PMAX]
    gamma_1d_hbm_reshaped = gamma_1d.reshape((H_free, _PMAX))
    gamma_flat_sb = nl.ndarray((H_free, _PMAX), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=gamma_flat_sb,
        src=gamma_1d_hbm_reshaped,
        dge_mode=3,
    )
    # Step c: SBUF [H_free, _PMAX] -> PSUM [_PMAX, H_free]
    gamma_trans_psum = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=gamma_trans_psum, data=gamma_flat_sb)
    # Step d: PSUM -> SBUF
    gamma_sb = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.activation(gamma_sb[...], op=nl.copy, data=gamma_trans_psum[...])
    # gamma_sb[p, h1] = gamma_1d[h1*128+p] ✓

    # RMSNorm computation — all intermediate tensors updated to 2D [_PMAX, H_free*T]
    # T=1 (TKG) assumption: H_free*T = H_free throughout.

    # 1a. x^2 [_PMAX, H_free*T]
    # Previously: rmsnorm_sq [_PMAX, T, H_free]; now flattened last two dims.
    rmsnorm_sq = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(rmsnorm_sq[...], op=nl.square, data=rmsnorm_out[...])

    # 1b. Reduce x^2 over all H_free*T elements (axis=1).
    # Previously: reduced [_PMAX, T, H_free] -> [_PMAX, T] over axis=1 (H_free dim).
    # Now: reduce [_PMAX, H_free*T] -> [_PMAX, T] over axis=1 (all H_free*T elements).
    # For T=1: [128, 16] -> [128, 1], identical result to the old 3D reduce.
    rmsnorm_reduced = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(rmsnorm_reduced[0:_PMAX, 0:T], nl.add, rmsnorm_sq[0:_PMAX, 0:H_free * T], axis=1)

    # 1c. gamma * input [_PMAX, H_free*T]
    # Previously: gamma_sb_bcast was [_PMAX, T, H_free] via expand_dim+broadcast.
    # Now: gamma_sb is [_PMAX, H_free] and rmsnorm_out is [_PMAX, H_free*T].
    # T=1 (TKG) assumption: H_free*T = H_free, so both have shape [_PMAX, H_free].
    # No broadcast needed for T=1. For T>1 we would need to interleave gamma across T;
    # this optimization is scoped to T=1 TKG workloads only.
    gamma_mult = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    # T=1 (TKG): gamma_sb [_PMAX, H_free] == [_PMAX, H_free*T], direct multiply.
    nisa.tensor_tensor(gamma_mult[...], rmsnorm_out[...], gamma_sb[...], nl.multiply)

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

    # 1f. rmsnorm_normed = gamma_mult * norm_factor [_PMAX, H_free*T]
    # Previously: norm_factor_sb [128, T] broadcast over H_free to [128, T, H_free].
    # Now: rmsnorm_normed is [_PMAX, H_free*T] = [_PMAX, H_free] for T=1.
    # Broadcast norm_factor_sb [_PMAX, T] -> [_PMAX, H_free*T]:
    #   expand_dim(dim=1) -> [_PMAX, 1, T], but we need [_PMAX, H_free*T].
    # For T=1: norm_factor_sb [_PMAX, 1] broadcast over H_free to [_PMAX, H_free].
    # Use TensorView: expand last dim and broadcast over H_free dimension.
    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=2).broadcast(dim=2, size=H_free)
    rmsnorm_normed = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    # gamma_mult is [_PMAX, H_free*T]; norm_factor_bcast view is [_PMAX, T, H_free].
    # For T=1: both are effectively [_PMAX, H_free] — the flattened view matches.
    nisa.tensor_tensor(rmsnorm_normed[...], gamma_mult[...], norm_factor_bcast.get_view(), nl.multiply)

    # Convert to bf16 for matmul (matches reference which uses bf16 matmul)
    # Previously: rmsnorm_normed_bf16 [_PMAX, T, H_free]; now [_PMAX, H_free*T].
    rmsnorm_normed_bf16 = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=rmsnorm_normed[...])

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul [T, H] @ [H, E=128] -> logits [T, E]
    # LHS/RHS swap: stationary=rmsnorm [128, T], moving=router_w [128, E]
    # PSUM[T, E]: T in PSUM partition dim, E in free dim -- directly usable for softmax
    #
    # Downstream adaptation: rmsnorm_normed_bf16 is now [_PMAX, H_free*T].
    # Old index: rmsnorm_normed_bf16[0:_PMAX, 0:T, h1]  (shape [_PMAX, T])
    # New index: rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1*T, T)]  (shape [_PMAX, T])
    # For T=1: nl.ds(h1*1, 1) = nl.ds(h1, 1), so rmsnorm_normed_bf16[0:_PMAX, h1:h1+1] = [_PMAX, 1]
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
        # Old: rmsnorm_normed_bf16[0:_PMAX, 0:T, h1]
        # New: rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1*T, T)] — same T elements
        nisa.nc_matmul(
            dst=logits_psum[0:T, 0:E],
            stationary=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
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
    # Gate/Up projection: Change B — H-shard: each core uses only H_free_shard tiles.
    # Core 0: H rows [0:1024], core 1: H rows [1024:2048].
    # Partial intermediate sums exchanged via nisa.sendrecv (SBUF↔SBUF).
    # Full gate/up = local_partial + received_partial.
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
    #
    # Downstream adaptation for rmsnorm_normed_bf16:
    # Old: rmsnorm_normed_bf16[0:_PMAX, 0:T, prg_id * H_free_shard + h1]
    # New: rmsnorm_normed_bf16[0:_PMAX, nl.ds((prg_id * H_free_shard + h1) * T, T)]
    # For T=1: same as rmsnorm_normed_bf16[0:_PMAX, prg_id*H_free_shard+h1 : prg_id*H_free_shard+h1+1]
    # -----------------------------------------------------------------------

    # Change B + 1: Fuse gate and up dims for a single weight load per H-tile.
    # Original layout: [E=128, H=2048, 2, I=256] — gate at dim2=0, up at dim2=1
    # After reshape: [E=128, H=2048, 512] — gate in columns [:256], up in columns [256:512]
    # This halves dynamic DMA calls: one [128,512] load replaces two [128,256] loads.
    E_shape, H_shape, _, I_shape = gate_up_w.shape
    gate_up_w_fused = gate_up_w.reshape((E_shape, H_shape, I_shape * 2))

    # output_temp [128, H_free_shard, T] - accumulates this core's H output shard (fp32)
    # Change 3: No memset here — k=0 will use tensor_copy to initialise (avoids add-to-zero).
    output_temp = nl.ndarray((_PMAX, H_free_shard, T), dtype=nl.float32, buffer=nl.sbuf)

    # v3b: K-slot intermediate buffers enabling two-pass expert processing.
    # Pass 1 stores all K SiLU inputs; Pass 2 reads them independently.
    # inter_buf: [128, I_tiles=2, K=8] x bf16 = 4 KB
    # down_buf:  [128, H_free_shard=8, K=8] x fp32 = 32 KB
    inter_buf = nl.ndarray((_PMAX, I_tiles, K), dtype=inp.dtype, buffer=nl.sbuf)
    down_buf  = nl.ndarray((_PMAX, H_free_shard, K), dtype=nl.float32, buffer=nl.sbuf)

    # ── Plan B Change 1: Pre-allocate gate/up weight double-buffer tiles ────────
    # Declared BEFORE the K-loop so the compiler sees 2 fixed SBUF regions.
    # This allows it to overlap "DMA fused_tile for h1+1" with "TE compute for h1".
    # Each tile is [_PMAX=128, I*2=512] bf16 = 128*512*2 = 131072 bytes = 128 KB.
    # Two tiles = 256 KB total. With H_free_shard=8 iterations cycling between 2 tiles,
    # adjacent iterations (h1 and h1+1) always use different tiles — no aliasing.
    wtile_0 = nl.ndarray((_PMAX, I * 2), dtype=gate_up_w.dtype, buffer=nl.sbuf)
    wtile_1 = nl.ndarray((_PMAX, I * 2), dtype=gate_up_w.dtype, buffer=nl.sbuf)

    # ── Plan B Change 2: Pre-allocate down weight double-buffer tiles ───────────
    # Declared BEFORE the K-loop so the compiler sees 2 fixed SBUF regions.
    # Each tile is [I0=128, H_free_shard*_PMAX=1024] bf16 = 128*1024*2 = 262144 bytes = 256 KB.
    # Two tiles = 512 KB total. With I_tiles=2, i_tile2=0 uses down_tile_0 and
    # i_tile2=1 uses down_tile_1 — perfect double-buffer with no aliasing.
    # NOTE: If these tiles cause SBUF spill (spill_bytes > 0), remove down_tile_0/down_tile_1
    # and revert down_w_full to be declared inside the i_tile2 loop.
    down_tile_0 = nl.ndarray((I0, H_free_shard * _PMAX), dtype=down_w.dtype, buffer=nl.sbuf)
    down_tile_1 = nl.ndarray((I0, H_free_shard * _PMAX), dtype=down_w.dtype, buffer=nl.sbuf)

    for t in nl.static_range(T):

        # ── Pre-broadcast all K affinity weights ────────────────────────────────
        # norm_weights has shape [T, K] with T in the partition dimension.
        # For T=1, partition 0 holds all K weights for this token.
        # We broadcast them to all 128 partitions once here (4 shuffles total)
        # instead of repeating memset + tensor_copy + 4 shuffles per expert k
        # inside Pass 2 (which would cost 6×K instructions per token).
        # aff_bcast[p, k] == norm_weights[t, k] for all partitions p after the shuffles.
        aff_bcast = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(aff_bcast, value=0.0)
        # Copy all K weights from partition 0 (where norm_weights[t, :] lives)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K], src=norm_weights[t:t + 1, 0:K])
        # 4 shuffles broadcast partition 0's K values to all 128 partitions.
        # shuffle_mask=[0]*32 means every destination partition in the group
        # reads from source partition 0 of the group, replicating the K weights.
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        # ── PASS 1: gate/up for ALL K experts ──────────────────────────────────
        # static_range required: sendrecv pipe_ids must be issued in fixed order.
        # Stores SiLU inputs to inter_buf[:, :, k] for each expert k.
        for k in nl.static_range(K):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            fused_w_view_shard = (
                TensorView(gate_up_w_fused)
                .select(dim=0, index=expert_id)
                .slice(dim=0, start=prg_id * H_shard, end=(prg_id + 1) * H_shard)
                .reshape_dim(dim=0, shape=[H_free_shard, _PMAX])
            )

            gate_psum = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.psum)
            up_psum   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(gate_psum, value=0.0)
            nisa.memset(up_psum,   value=0.0)

            for h1 in nl.affine_range(H_free_shard):
                # Plan B Change 1: cycle between two pre-allocated tiles instead of
                # allocating a new tile each iteration. The compiler now sees 2 fixed
                # SBUF regions and can pipeline DMA for h1+1 with matmul for h1.
                # h1=0,2,4,6 -> wtile_0; h1=1,3,5,7 -> wtile_1
                fused_tile = wtile_0 if h1 % 2 == 0 else wtile_1
                fused_h_view = fused_w_view_shard.slice(dim=0, start=h1, end=h1 + 1).squeeze_dim(dim=0)
                nisa.dma_copy(dst=fused_tile, src=fused_h_view.get_view(),
                              dge_mode=_adaptive_dge(fused_h_view))
                for i_tile in nl.static_range(I_tiles):
                    nisa.nc_matmul(
                        dst=gate_psum[0:_PMAX, i_tile:i_tile + 1],
                        stationary=fused_tile[0:_PMAX, nl.ds(i_tile * I0, I0)],
                        # Old: rmsnorm_normed_bf16[0:_PMAX, 0:T, prg_id * H_free_shard + h1]
                        # New: index as [_PMAX, nl.ds(global_h1 * T, T)]
                        # For T=1: equivalent slice of the 2D [_PMAX, H_free*T] tensor
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((prg_id * H_free_shard + h1) * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=up_psum[0:_PMAX, i_tile:i_tile + 1],
                        stationary=fused_tile[0:_PMAX, nl.ds(I + i_tile * I0, I0)],
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((prg_id * H_free_shard + h1) * T, T)],
                    )

            gate_sb = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            up_sb   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_sb, op=nl.copy, data=gate_psum)
            nisa.activation(up_sb,   op=nl.copy, data=up_psum)

            recv_gate = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            recv_up   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.sendrecv(src=gate_sb, dst=recv_gate,
                          send_to_rank=other, recv_from_rank=other, pipe_id=k * 2)
            nisa.sendrecv(src=up_sb,   dst=recv_up,
                          send_to_rank=other, recv_from_rank=other, pipe_id=k * 2 + 1)

            full_gate = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            full_up   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(full_gate, gate_sb, recv_gate, nl.add)
            nisa.tensor_tensor(full_up,   up_sb,   recv_up,   nl.add)

            silu_res = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(silu_res, op=nl.silu, data=full_gate)
            nisa.tensor_tensor(inter_f32, silu_res, full_up, nl.multiply)

            # Store SiLU output to K-slot buffer — Pass 2 reads it
            nisa.activation(
                inter_buf[0:_PMAX, 0:I_tiles, k:k + 1],
                op=nl.copy,
                data=inter_f32,
            )

        # ── PASS 2: down projection for ALL K experts — affine_range enables pipelining ──
        # Independent across k: each reads inter_buf[:,:,k] (distinct), accesses
        # different expert_id HBM address, writes down_buf[:,:,k] (distinct).
        # Compiler overlaps expert k+1 down weight DMA with expert k matmuls.
        for k in nl.affine_range(K):
            expert_id_k = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            down_w_view = (
                TensorView(down_w)
                .select(dim=0, index=expert_id_k)
                .slice(dim=1, start=prg_id * H_shard, end=(prg_id + 1) * H_shard)
            )

            # Persistent PSUM (Plan 1 pattern) — 1 bank covers all H_free_shard columns
            down_psum = nl.ndarray((_PMAX, _PSUM_FREE), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(down_psum, value=0.0)

            # Load inter for this expert from K-slot buffer
            inter_bf16_k = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=inter_bf16_k, src=inter_buf[0:_PMAX, 0:I_tiles, k:k + 1])

            for i_tile2 in nl.affine_range(I_tiles):
                # Plan B Change 2: cycle between two pre-allocated down weight tiles.
                # The compiler now sees 2 fixed SBUF regions and can pipeline DMA for
                # i_tile2+1 with matmul for i_tile2. With I_tiles=2, this is a perfect
                # double-buffer: i_tile2=0 uses down_tile_0, i_tile2=1 uses down_tile_1.
                down_w_full = down_tile_0 if i_tile2 % 2 == 0 else down_tile_1
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
                        moving=inter_bf16_k[0:_PMAX, i_tile2:i_tile2 + 1],
                    )

            # Single flush per expert (Plan 1)
            down_result_sb = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free_shard],
                op=nl.copy,
                data=down_psum[0:_PMAX, 0:H_free_shard],
            )

            # Apply affinity scale using pre-broadcast weights.
            # aff_bcast[:, k] already has norm_weights[t, k] replicated across
            # all 128 partitions — no per-expert memset/tensor_copy/shuffle needed.
            down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],  # use pre-broadcast column for this expert
            )
            # Store scaled result to K-slot buffer for Pass 3 accumulation
            nisa.tensor_copy(
                dst=down_buf[0:_PMAX, 0:H_free_shard, k:k + 1],
                src=down_result_scaled[0:_PMAX, 0:H_free_shard],
            )

        # ── PASS 3: accumulate K down results into output_temp ─────────────────
        # Sequential (static_range) because output_temp carries across k.
        # Only K=8 tensor_tensor adds — negligible cost.
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
