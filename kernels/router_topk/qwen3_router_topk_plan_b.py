"""
Router Top-K kernel for Qwen3-30B-A3B CTE — Plan B.

Plan B swaps stationary/moving tensors relative to the original kernel:
  - Original: x_tile [T_TILE, H] is stationary,  w_tile [H, E] is moving
  - Plan B:   w_tile [H, E] is stationary,        x [H, T_local] is moving

This replaces 80 matmuls of [128,64]@[128,128] with 16 matmuls of
[128,128]@[128,320], growing the moving free dim from 64→320 (≤512 limit).

The nc_matmul result is [E=128, T_local=320] instead of [T_TILE, E], so a
transpose step converts it back to [T_local, E] before the existing softmax/topK logic.

All post-matmul logic (softmax, topK, scatter) is copied verbatim from the
original; the T-tile loop just slices [E, T_local] → [E, T_TILE_actual] → transpose.

Plan B (vs original Plan B): x is now [H, T] layout (identical to Plan A) to enable
single burst-DMA loading of x via .ap() strides — T_local tokens contiguous per
(partition, H-tile) row, eliminating stride-gather penalty.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import core_barrier

# --------------- Hardcoded Qwen3 constants ---------------
H = 2048          # hidden_size
E = 128           # num_experts
K = 8             # top-K experts per token
P = 128           # NeuronCore partition dimension (trn2)
NUM_H_TILES = H // P   # = 16
T_TILE = 128      # token tile size for post-matmul processing; P=128 supports this

# Module-level constant so the test harness can detect which HBM layout x uses.
# Plan B uses [H, T] layout for burst-DMA efficiency; original uses [T, H].
X_HBM_LAYOUT = "HT"


@nki.jit(platform_target="trn2")
def qwen3_router_topk_cte(
    x,                  # [H, T]    bf16  — hidden states, H-major for burst DMA (Plan B)
    w,                  # [H, E]    bf16  — router weight (transposed from [E, H])
    router_logits,      # [T, E]    float32  — output: raw logits before softmax
    expert_affinities,  # [T, E]    float32  — output: scattered, L1-normalized affinities
    expert_index,       # [T, K]    uint32   — output: top-K expert indices per token
):
    """
    Router top-K kernel — Plan B.

    Launched as: qwen3_router_topk_cte[2](x, w, router_logits, expert_affinities, expert_index)

    x is expected in [H, T] layout (H-major). This enables a single burst DMA
    that loads T_local contiguous tokens per (partition, H-tile) row without
    stride-gather penalties.

    Note: output parameters are accepted for interface compatibility but the
    kernel writes to nl.shared_hbm buffers and returns those.
    """
    T_full = x.shape[1]   # total tokens (dynamic); x is [H, T] so dim 1 is T

    # ----------------------------------------------------------------
    # LNC sharding: each core processes T_local = T/2 tokens
    # ----------------------------------------------------------------
    n_prgs = nl.num_programs(0)   # 2 when launched with [2]
    prg_id = nl.program_id(0)     # 0 or 1

    T_local = T_full // n_prgs     # tokens per core (320 for T=640, LNC=2)
    T_offset = prg_id * T_local    # token start index for this core

    # ----------------------------------------------------------------
    # Allocate output tensors as nl.shared_hbm.
    # Using nl.shared_hbm avoids a neuronx-cc compiler bug where multiple
    # nisa.dma_copy stores to the same parameter tensor cause an InstSave
    # assertion failure in the BIR address-rotation/dma-optimization passes.
    # ----------------------------------------------------------------
    rl_out  = nl.ndarray((T_full, E), dtype=nl.float32, buffer=nl.shared_hbm)
    ea_out  = nl.ndarray((T_full, E), dtype=nl.float32, buffer=nl.shared_hbm)
    ei_out  = nl.ndarray((T_full, K), dtype=nl.uint32,  buffer=nl.shared_hbm)

    # ----------------------------------------------------------------
    # expert_iota is constant — hoist outside T-tile loop
    # ----------------------------------------------------------------
    expert_iota = nl.ndarray((P, E), dtype=nl.uint32, buffer=nl.sbuf,
                             name="expert_iota")
    nisa.iota(dst=expert_iota, pattern=[[1, E]], offset=0, channel_multiplier=0)

    # ----------------------------------------------------------------
    # Plan B: load entire x into SBUF via single burst DMA.
    # x[H, T_full] is reorganized as x_sb[P, NUM_H_TILES, T_local].
    # .ap strides map x_sb[p, ht, t] → x[ht*P + p, T_offset + t]:
    #   partition dim (p): stride = T_full       (advancing p by 1: skip one full x row)
    #   h-tile dim (ht):   stride = P * T_full   (advancing ht by 1: skip P x rows)
    #   element dim (t):   stride = 1, size = T_local (T_local contiguous tokens)
    # offset = T_offset positions into the T dimension of x.
    # Within each (p, ht) row: T_local contiguous HBM reads — true burst DMA.
    # x stays OUTSIDE the T-tile loop — reused across T tiles.
    # ----------------------------------------------------------------
    x_sb = nl.ndarray((P, NUM_H_TILES, T_local), dtype=nl.bfloat16, buffer=nl.sbuf,
                      name="x_sb")
    # x[H, T_full] flat: element (h, t) at offset h*T_full + t.
    # x_sb[p, ht, t] maps to h = ht*P + p → offset (ht*P + p)*T_full + T_offset + t.
    #   partition stride (p): T_full       (shift by one x row)
    #   h-tile stride  (ht): P * T_full   (shift by P x rows)
    #   element stride  (t): 1            (contiguous)
    nisa.dma_copy(
        dst=x_sb,
        src=x.ap([[T_full, P], [P * T_full, NUM_H_TILES], [1, T_local]], offset=T_offset),
    )

    # ----------------------------------------------------------------
    # Plan B: load entire w into SBUF via single burst DMA.
    # w[H, E] is reorganized as w_sb[P, NUM_H_TILES, E].
    # .ap strides map w_sb[p, ht, e] → w[ht*P + p, e]:
    #   partition dim (p): stride = E      (skip one w row)
    #   h-tile dim (ht):   stride = P * E  (skip P w rows)
    #   element dim (e):   stride = 1, size = E (E contiguous elements)
    # Each (p, ht) row reads E=128 contiguous HBM bytes — burst, not gather.
    # ----------------------------------------------------------------
    w_sb = nl.ndarray((P, NUM_H_TILES, E), dtype=nl.bfloat16, buffer=nl.sbuf, name="w_sb")
    nisa.dma_copy(
        dst=w_sb,
        src=w.ap([[E, P], [P * E, NUM_H_TILES], [1, E]]),
    )

    # ----------------------------------------------------------------
    # Step 3: Single PSUM [E=128, T_local=320] accumulating all H-tiles.
    # Shape note: partition dim = E=128, free dim = T_local=320 ≤ 512 ✓
    # Plan B's distinguishing feature: moving free dim = T_local=320 ≥ T_TILE=128,
    # so ALL T_local tokens are processed in one PSUM (vs Plan A's per-T-tile PSUM).
    # nc_matmul with swapped stationary/moving:
    #   stationary = w_sb[:, ht, :] = [P=128, E=128]   (w h-tile)
    #   moving     = x_sb[:, ht, :] = [P=128, T_local=320]
    # computes: stationary^T @ moving = w_tile^T @ x_tile = [E=128, T_local=320]
    # and accumulates into the PSUM (zero-initialized by nl.zeros).
    # 16 total matmul calls (vs Plan A's 48 = 16 h-tiles × 3 t-tiles).
    # ----------------------------------------------------------------
    router_logits_psum = nl.zeros((E, T_local), dtype=nl.float32, buffer=nl.psum,
                                  name="rl_psum_full")

    for ht in range(NUM_H_TILES):  # Python range: ht is compile-time int for direct SBUF slice
        w_tile = nl.ndarray((P, E), dtype=nl.bfloat16, buffer=nl.sbuf,
                            name=f"w_tile_b_{ht}")
        # tensor_copy needed here to extract 2D slice from 3D w_sb before passing to
        # nc_matmul as stationary — nc_matmul stationary requires a 2D SBUF tensor.
        nisa.tensor_copy(dst=w_tile, src=w_sb[:, ht, :])
        nisa.nc_matmul(
            dst=router_logits_psum,
            stationary=w_tile,              # [P=128, E=128] — w h-tile as stationary
            moving=x_sb[:, ht, :],          # [P=128, T_local=320] — full T_local as moving
        )

    # ----------------------------------------------------------------
    # Step 4: Copy PSUM [E, T_local] → SBUF for T-tile slicing.
    # ----------------------------------------------------------------
    rl_sb = nl.ndarray((E, T_local), dtype=nl.float32, buffer=nl.sbuf,
                       name="rl_sb_full")
    nisa.tensor_copy(dst=rl_sb, src=router_logits_psum)

    # ----------------------------------------------------------------
    # Step 5: T-tile loop — transpose each [E, T_TILE_actual] slice and
    # run the existing softmax/topK/scatter pipeline per tile.
    # T_local=320 with T_TILE=128 → 3 tiles: [0,128), [128,256), [256,320)
    # ----------------------------------------------------------------
    num_t_tiles = (T_local + T_TILE - 1) // T_TILE   # ceiling division = 3

    for t_tile in range(num_t_tiles):
        T_TILE_actual = min(T_TILE, T_local - t_tile * T_TILE)
        t_off = T_offset + t_tile * T_TILE   # absolute token start in HBM

        # ----------------------------------------------------------------
        # Extract contiguous [E, T_TILE_actual] slice from rl_sb [E, T_local]
        # ----------------------------------------------------------------
        rl_slice = nl.ndarray((E, T_TILE_actual), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"rl_slice_{t_tile}")
        nisa.tensor_copy(
            dst=rl_slice,
            src=rl_sb[:, nl.ds(t_tile * T_TILE, T_TILE_actual)],
        )

        # ----------------------------------------------------------------
        # nc_transpose: [E=128, T_TILE_actual] SBUF → [T_TILE_actual, E=128] PSUM
        # This converts from E-major layout back to token-major layout.
        # ----------------------------------------------------------------
        rl_transposed_psum = nl.ndarray((T_TILE_actual, E), dtype=nl.float32,
                                        buffer=nl.psum, name=f"rl_trans_psum_{t_tile}")
        nisa.nc_transpose(dst=rl_transposed_psum, data=rl_slice)

        # Copy transposed PSUM → SBUF
        router_logits_sb = nl.ndarray((T_TILE_actual, E), dtype=nl.float32,
                                      buffer=nl.sbuf, name=f"rl_sb_{t_tile}")
        nisa.tensor_copy(dst=router_logits_sb, src=rl_transposed_psum)

        # Round fp32 PSUM accum to bf16 grid to match CPU bf16 matmul precision
        rl_bf16_tmp = nl.ndarray((T_TILE_actual, E), dtype=nl.bfloat16,
                                  buffer=nl.sbuf, name=f"rl_bf16_{t_tile}")
        nisa.tensor_copy(dst=rl_bf16_tmp, src=router_logits_sb)   # fp32 → bf16
        nisa.tensor_copy(dst=router_logits_sb, src=rl_bf16_tmp)   # bf16 → fp32

        # Store router_logits slice to HBM at token rows [t_off : t_off+T_TILE_actual]
        nisa.dma_copy(
            dst=rl_out.ap([[E, T_TILE_actual], [1, E]], offset=t_off * E),
            src=router_logits_sb,
        )

        # ----------------------------------------------------------------
        # Softmax (numerically stable)
        # ----------------------------------------------------------------
        affinities_sb = nl.ndarray((T_TILE_actual, E), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"affi_sb_{t_tile}")

        negmax_sb = nl.ndarray((T_TILE_actual, 1), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"negmax_sb_{t_tile}")
        nisa.tensor_reduce(
            dst=negmax_sb,
            op=nl.maximum,
            data=router_logits_sb,
            axis=1,
            negate=True,
            keepdims=True,
        )

        inv_sum_sb = nl.ndarray((T_TILE_actual, 1), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"inv_sum_sb_{t_tile}")
        nisa.activation(
            dst=affinities_sb,
            op=nl.exp,
            data=router_logits_sb,
            bias=negmax_sb,
            reduce_op=nl.add,
            reduce_res=inv_sum_sb,
        )
        nisa.reciprocal(dst=inv_sum_sb, data=inv_sum_sb)
        nisa.tensor_scalar(
            dst=affinities_sb,
            data=affinities_sb,
            op0=nl.multiply,
            operand0=inv_sum_sb,
        )

        # ----------------------------------------------------------------
        # Top-K selection
        # ----------------------------------------------------------------
        topk_vals_sb = nl.ndarray((T_TILE_actual, K), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"topk_vals_sb_{t_tile}")
        topk_idx_sb  = nl.ndarray((T_TILE_actual, K), dtype=nl.uint32, buffer=nl.sbuf,
                                  name=f"topk_idx_sb_{t_tile}")

        top8_buf = nl.ndarray((T_TILE_actual, 8), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"top8_buf_{t_tile}")
        nisa.max8(dst=top8_buf, src=affinities_sb)
        nisa.tensor_copy(dst=topk_vals_sb, src=top8_buf[:, :K])

        idx8_buf = nl.ndarray((T_TILE_actual, 8), dtype=nl.uint32, buffer=nl.sbuf,
                              name=f"idx8_buf_{t_tile}")
        nisa.nc_find_index8(dst=idx8_buf, data=affinities_sb, vals=top8_buf)
        nisa.tensor_copy(dst=topk_idx_sb, src=idx8_buf[:, :K])

        topk_idx_fp32_sb = nl.ndarray((T_TILE_actual, K), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"topk_idx_fp32_sb_{t_tile}")
        nisa.tensor_copy(dst=topk_idx_fp32_sb, src=topk_idx_sb)

        nisa.dma_copy(
            dst=ei_out.ap([[K, T_TILE_actual], [1, K]], offset=t_off * K),
            src=topk_idx_sb,
        )

        # ----------------------------------------------------------------
        # L1 normalization of top-K affinities
        # ----------------------------------------------------------------
        sum_topk_sb = nl.ndarray((T_TILE_actual, 1), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"sum_topk_sb_{t_tile}")
        nisa.tensor_reduce(
            dst=sum_topk_sb,
            op=nl.add,
            data=topk_vals_sb,
            axis=1,
            keepdims=True,
        )
        nisa.reciprocal(dst=sum_topk_sb, data=sum_topk_sb)

        topk_vals_norm_sb = nl.ndarray((T_TILE_actual, K), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"topk_vals_norm_sb_{t_tile}")
        nisa.tensor_scalar(
            dst=topk_vals_norm_sb,
            data=topk_vals_sb,
            op0=nl.multiply,
            operand0=sum_topk_sb,
        )

        # ----------------------------------------------------------------
        # One-hot scatter: build [T_TILE_actual, E] mask
        # ----------------------------------------------------------------
        mask_sb = nl.ndarray((T_TILE_actual, E), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"mask_sb_{t_tile}")
        nisa.memset(dst=mask_sb, value=0.0)

        check_buf = nl.ndarray((T_TILE_actual, E), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"check_buf_{t_tile}")
        for k_slot in nl.affine_range(K):
            nisa.tensor_scalar(
                dst=check_buf[:T_TILE_actual, :],
                op0=nl.equal,
                data=expert_iota[:T_TILE_actual, :],
                operand0=topk_idx_fp32_sb[:T_TILE_actual, k_slot],
            )
            nisa.tensor_tensor(
                dst=mask_sb[:T_TILE_actual, :],
                data1=mask_sb[:T_TILE_actual, :],
                op=nl.add,
                data2=check_buf[:T_TILE_actual, :],
            )

        # ----------------------------------------------------------------
        # Apply normalized affinities through the mask
        # ----------------------------------------------------------------
        nisa.tensor_scalar(
            dst=affinities_sb,
            data=affinities_sb,
            op0=nl.multiply,
            operand0=sum_topk_sb,
        )

        scattered_sb = nl.ndarray((T_TILE_actual, E), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"scattered_sb_{t_tile}")
        nisa.tensor_tensor(
            dst=scattered_sb,
            data1=mask_sb,
            op=nl.multiply,
            data2=affinities_sb,
        )

        # Store scattered expert_affinities slice to HBM
        nisa.dma_copy(
            dst=ea_out.ap([[E, T_TILE_actual], [1, E]], offset=t_off * E),
            src=scattered_sb,
        )

    # Barrier ensures both cores have written before caller reads full output.
    # core_barrier stays OUTSIDE the T-tile loop.
    core_barrier(ea_out, cores=[0, 1])

    return rl_out, ea_out, ei_out
