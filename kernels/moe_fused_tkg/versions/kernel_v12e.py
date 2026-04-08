"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

kernel_v12e — Plan E: fused single k-loop + down DMA prefetch
=============================================================

Changes from v10b:
  1. Replace the 3-pass structure (Pass1 affine_range(K) + Pass2 affine_range(K) +
     Pass3 static_range(K)) with a single static_range(K) loop.
  2. Within each k iteration the order is:
       a. Get expert_id
       b. Issue gate/up DMA
       c. Issue down DMA IMMEDIATELY (before any matmul) — KEY change
          → down DMA executes in parallel with the gate/up matmul
       d. Gate/up matmul
       e. SiLU + inter (fp32 → bf16)
       f. Down matmul (using pre-loaded down_tile)
       g. Scale by affinity + accumulate into output_temp
          (copy-for-k=0, tensor_tensor(add) for k>0)
  3. Remove inter_buf [128P, 2, 8] (4 KB) and down_buf [128P, 8, 8] (32 KB).
     fused_tile and down_tile are allocated locally per k iteration.

All other optimizations from v10b are preserved unchanged.

v10b optimizations preserved:
  1. Batch RMSNorm loads (single 3D DMA for inp and gamma) [v4a]
  2. Pre-broadcast affinity weights into aff_bcast[128, K] [v5a]
  3. 3D DMA pattern for gate/up (full H) and down (H_shard) — Plan D adapted
  4. Router 4-tile DMA batch [v10b]
  5. Copy-for-k0 in accumulation (no memset) [v1b]
  6. router_w_tile_sb dtype=inp.dtype (bf16 cast fix from v9a)

SBUF budget estimate (per core):
  inp_flat_sb:         [16, 128] bf16         =  4 KB
  gamma_flat_sb:       [16, 128] bf16         =  4 KB
  rmsnorm_out:         [128, 16] bf16         =  4 KB
  rmsnorm_normed_bf16: [128, 16] bf16         =  4 KB
  output_temp:         [128, 8, 1] fp32       = 32 KB   (H_free_shard=8, T=1)
  aff_bcast:           [128, 8] fp32          =  4 KB
  router_w_wide_sb:    [128, 512] bf16        = 128 KB  (4 tiles × 128 cols, router batch)
  Per k iteration peak:
    fused_tile:        [128, 16, 512] bf16    = 128 KB
    down_tile:         [128, 2, 1024] bf16    =  64 KB (I0=128 in P-dim → 4 KB/partition)
  Note: fused_tile and down_tile don't coexist with router_w_wide_sb.
  Total peak: ~150 KB — within 224 KB limit.
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
_H_FREE = _H // _PMAX           # = 16 tiles of 128 each
_H_FREE_SHARD = _H_FREE // _N_PRGS   # = 8 (each core owns 8 H-tiles for output)
_H_SHARD = _H_FREE_SHARD * _PMAX     # = 1024
_I0 = _PMAX                           # = 128
_I_TILES = _I // _I0                  # = 2

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
    Returns: output [T, H=2048] bf16
    Each core writes its H_shard independently; no synchronization needed.
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
    # _PMAX=128 × (_ROUTER_BATCH*E=512) × 2 bytes = 131072 bytes = 128KB per DMA.
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)
    # Wide SBUF buffer: holds 4 consecutive router weight tiles at once.
    # 3D shape [_PMAX, _ROUTER_BATCH, E] so the AP descriptor matches src.
    router_w_wide_sb = nl.ndarray((_PMAX, _ROUTER_BATCH, E), dtype=inp.dtype, buffer=nl.sbuf)

    for h_chunk in nl.affine_range(H_free // _ROUTER_BATCH):  # 4 outer iterations
        # Single 128KB DMA instead of 4×32KB — fewer start overheads.
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
    # v12e strategy: single fused static_range(K) loop per token.
    # Within each k iteration:
    #   1. Issue gate/up DMA
    #   2. Issue down DMA immediately (KEY: before any matmul) → overlaps with gate/up matmul
    #   3. Gate/up matmul
    #   4. SiLU + inter
    #   5. Down matmul (uses pre-loaded down_tile)
    #   6. Scale + accumulate (copy-for-k=0, add-for-k>0)
    #
    # Fuse gate/up weight dims: [E, H, 2, I] -> [E, H, I*2]
    # Gate cols: [0:I], Up cols: [I:I*2]
    # Load ALL H rows (no H-shard offset for gate/up).
    # -----------------------------------------------------------------------

    E_shape, H_shape, _, I_shape = gate_up_w.shape
    gate_up_w_fused = gate_up_w.reshape((E_shape, H_shape, I_shape * 2))

    # output_temp: H_free_shard tiles (this core's H output shard), fp32
    # No memset — k=0 uses tensor_copy to initialise (avoids add-to-zero)
    output_temp = nl.ndarray((_PMAX, H_free_shard, T), dtype=nl.float32, buffer=nl.sbuf)

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

        # ── FUSED K-LOOP: gate/up DMA + down DMA prefetch + matmuls + accumulate ──
        # static_range(K) required: loop carries output_temp dependency across
        # iterations (copy-for-k=0 vs add-for-k>0 checked at trace time).
        for k in nl.static_range(K):
            # ─── STEP 1: Get expert ID for this k ──────────────────────────────
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            # ─── STEP 2: Issue gate/up DMA ─────────────────────────────────────
            # 3D DMA for gate+up weights — load FULL H (all H_free=16 tiles)
            # gate_up_w_fused shape: [E=128, H=2048, I*2=512]
            # Target: fused_tile[p, h1, col] = gate_up_w_fused[expert_id, h1*128 + p, col]
            # AP pattern (3D):
            #   dim 0 (p):   stride = I*2 = 512,         count = _PMAX = 128
            #   dim 1 (h1):  stride = _PMAX*I*2 = 65536, count = H_free = 16
            #   dim 2 (col): stride = 1,                 count = I*2 = 512
            # offset = 0 — load from start of expert (full H, no shard offset)
            fused_tile = nl.ndarray((_PMAX, H_free, I * 2), dtype=gate_up_w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=fused_tile,
                src=gate_up_w_fused.ap(
                    pattern=[[I * 2, _PMAX], [_PMAX * I * 2, H_free], [1, I * 2]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # ─── STEP 3: Issue down DMA IMMEDIATELY (KEY: before gate/up matmul) ──
            # This allows down DMA to execute in parallel with the gate/up matmul below.
            # 3D DMA for down weights — H_shard only (with prg_id offset)
            # down_w shape: [E=128, I=256, H=2048]
            # Target: down_tile[i0, i1, h] = down_w[expert_id, i1*I0 + i0, prg_id*H_shard + h]
            # AP pattern (3D):
            #   i0: stride = H = 2048,              count = I0 = 128
            #   i1: stride = I0*H = 262144,          count = I_tiles = 2
            #   h:  stride = 1,                      count = H_shard = 1024
            # offset = prg_id * H_shard (within expert's H row)
            down_tile = nl.ndarray((I0, I_tiles, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=down_tile,
                src=down_w.ap(
                    pattern=[[H, I0], [I0 * H, I_tiles], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # ─── STEP 4: Gate/up matmul (down DMA runs IN PARALLEL during this) ──
            # PSUM for gate/up — full I_tiles=2
            gate_psum = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.psum)
            up_psum   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(gate_psum, value=0.0)
            nisa.memset(up_psum,   value=0.0)

            # Loop over full H_free=16 (not H_free_shard=8), full I_tiles=2
            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    nisa.nc_matmul(
                        dst=gate_psum[0:_PMAX, i_tile:i_tile + 1],
                        stationary=fused_tile[0:_PMAX, h1, nl.ds(i_tile * I0, I0)],
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=up_psum[0:_PMAX, i_tile:i_tile + 1],
                        stationary=fused_tile[0:_PMAX, h1, nl.ds(I + i_tile * I0, I0)],
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )

            # ─── STEP 5: Flush PSUM -> SBUF, SiLU, compute inter (fp32 → bf16) ──
            gate_sb = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            up_sb   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_sb, op=nl.copy, data=gate_psum)
            nisa.activation(up_sb,   op=nl.copy, data=up_psum)

            # SiLU activation + element-wise multiply
            silu_res  = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(silu_res, op=nl.silu, data=gate_sb)
            nisa.tensor_tensor(inter_f32, silu_res, up_sb, nl.multiply)

            # Cast inter to bf16 for down matmul
            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            # ─── STEP 6: Down matmul using pre-loaded down_tile ─────────────────
            # Persistent PSUM covering all H_free_shard output tiles
            down_psum = nl.ndarray((_PMAX, _PSUM_FREE), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(down_psum, value=0.0)

            # Matmul loop — reads from pre-loaded 3D tile
            for i_tile2 in nl.affine_range(I_tiles):
                for h1_out in nl.static_range(H_free_shard):
                    nisa.nc_matmul(
                        dst=down_psum[0:_PMAX, nl.ds(h1_out * T, T)],
                        stationary=down_tile[0:I0, i_tile2, nl.ds(h1_out * _PMAX, _PMAX)],
                        moving=inter_bf16[0:_PMAX, i_tile2:i_tile2 + 1],
                    )

            # Single PSUM flush per expert (H_free_shard columns)
            down_result_sb = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free_shard],
                op=nl.copy,
                data=down_psum[0:_PMAX, 0:H_free_shard],
            )

            # ─── STEP 7: Scale by affinity + accumulate into output_temp ─────────
            # Scale by affinity weight (pre-broadcast)
            down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],
            )

            # Copy-for-k=0 (no memset needed), add-for-k>0
            # static_range(K) allows this Python-level if/else check at trace time
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
    # Stage 5: Transpose, cast fp32->bf16, and store output
    # output_temp [128, H_free_shard, T] (fp32) -> HBM output [T, H] (bf16)
    # Each core writes its H_shard columns at offset prg_id*H_shard.
    # No all-reduce needed — each core independently produced correct H_shard output.
    # -----------------------------------------------------------------------
    output = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.hbm)
    out_sb = nl.ndarray((T, H_shard), dtype=inp.dtype, buffer=nl.sbuf)

    for h1 in nl.static_range(H_free_shard):
        # output_temp[0:_PMAX, h1, 0:T] is [_PMAX, T] fp32
        # Transpose to [T, _PMAX] via PSUM, then cast fp32->bf16
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

    Returns the expert MLP output after routing and accumulation.
    """
    outputs = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
