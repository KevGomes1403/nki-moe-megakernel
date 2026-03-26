"""
Router Top-K kernel for Qwen3-30B-A3B CTE, single TP rank (TP=4), LNC=2.

Math:
    logits[t, e]     = sum_h(x[t, h] * w[h, e])          # bf16 matmul, fp32 accum
    affinities[t, :] = softmax(logits[t, :])              # fp32, numerically stable
    topK_idx[t, :]   = argtop8(affinities[t, :])[:K]      # hardware max8+find_index8
    sum_topK[t]      = sum(affinities[t, topK_idx[t, :]]) # fp32 reduction
    out_affi[t, e]   = affinities[t, e] / sum_topK[t]     # if e in topK_idx, else 0

Shapes (CTE, B=1, S=128, TP=4):
    x:               [T, H]    = [T, 2048]    bf16   (T_local tiled in T_TILE=64 chunks)
    w:               [H, E]    = [2048, 128]  bf16   (caller transposes router.weight [E,H])
    router_logits:   [T, E]    = [T, 128]     float32
    expert_affinities:[T, E]   = [T, 128]     float32 (scattered + normalized)
    expert_index:    [T, K]    = [T, 8]       uint32

LNC=2 sharding:
    Each core handles T_local = T // 2 tokens starting at T_offset = prg_id * T_local.
    Results are written independently to HBM at [T_offset : T_offset+T_local].
    core_barrier synchronizes before caller reads the full output.

T-tiling:
    T_TILE=64 keeps the partition dimension within hardware limits (64 < P=128).
    T must be divisible by T_TILE * n_prgs = 64 * 2 = 128 (640 ✓).
    Each LNC core processes T_local/T_TILE = 5 tiles sequentially.
    Output tensors are nl.shared_hbm allocations (avoids compiler bug with
    multiple dma_copy stores to a passed-in output parameter tensor).

Key optimizations preserved from nkilib router_topk:
    1. w hoisted to SBUF once as [P, num_h_tiles, E] — reused across T tiles
    2. nc_matmul (TensorEngine) for [T_TILE, H] @ [H, E] contraction
    3. Numerically stable softmax: negate-max, fused exp+reduce, reciprocal
    4. Hardware max8 + nc_find_index8 for top-K in one pass (K <= 8)
    5. One-hot scatter via nisa.iota + K compare-accumulate passes
    6. LNC token sharding: T splits across 2 cores, independent HBM stores
    7. nisa.dma_copy for all bulk data movement (better pipeline than nl.load)
    Plan A: w loaded in single 524 KB DMA (replaces 16-packet affine_range loop)

Constraints:
    H = 2048, E = 128, K = 8  (hardcoded per CLAUDE.md)
    T divisible by T_TILE * n_prgs = 128
    T_local <= P (tiled in T_TILE=64 chunks)
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
T_TILE = 64       # token tile size; must be <= P=128; 320/64=5 tiles per LNC core


@nki.jit(platform_target="trn2")
def qwen3_router_topk_cte(
    x,                  # [T, H]    bf16  — hidden states after post_attn_layernorm
    w,                  # [H, E]    bf16  — router weight (transposed from [E, H])
    router_logits,      # [T, E]    float32  — output: raw logits before softmax
    expert_affinities,  # [T, E]    float32  — output: scattered, L1-normalized affinities
    expert_index,       # [T, K]    uint32   — output: top-K expert indices per token
):
    """
    Router top-K kernel specialized for Qwen3 CTE shapes, LNC=2.

    Launched as: qwen3_router_topk_cte[2](x, w, router_logits, expert_affinities, expert_index)

    Note: output parameters (router_logits, expert_affinities, expert_index) are
    accepted for interface compatibility but the kernel writes to nl.shared_hbm
    buffers internally and returns those.  The caller should use the returned
    tensors, not the passed-in output buffers.
    """
    T = x.shape[0]   # total tokens (dynamic)

    # ----------------------------------------------------------------
    # LNC sharding: each core processes T_local = T/2 tokens
    # ----------------------------------------------------------------
    n_prgs = nl.num_programs(0)   # 2 when launched with [2]
    prg_id = nl.program_id(0)     # 0 or 1

    # Each core owns a contiguous half of the token dimension
    T_local = T // n_prgs          # tokens per core (e.g. 320 for T=640, LNC=2)
    T_offset = prg_id * T_local    # token start index for this core

    num_t_tiles = T_local // T_TILE   # e.g. 320 // 64 = 5

    # ----------------------------------------------------------------
    # Allocate output tensors as nl.shared_hbm.
    # Using nl.shared_hbm (rather than writing to the passed-in output
    # parameter tensors) avoids a neuronx-cc compiler bug where multiple
    # nisa.dma_copy stores to the same parameter tensor cause an InstSave
    # assertion failure in the BIR address-rotation/dma-optimization passes.
    # ----------------------------------------------------------------
    rl_out  = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.shared_hbm)
    ea_out  = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.shared_hbm)
    ei_out  = nl.ndarray((T, K), dtype=nl.uint32,  buffer=nl.shared_hbm)

    # ----------------------------------------------------------------
    # Plan A: load entire w into SBUF in one 524 KB DMA.
    # w stays OUTSIDE the T-tile loop — same weights for all tiles.
    # ----------------------------------------------------------------
    w_sb = nl.ndarray((P, NUM_H_TILES * E), dtype=nl.bfloat16, buffer=nl.sbuf,
                      name="w_sb")
    nisa.dma_copy(dst=w_sb, src=w.reshape((P, NUM_H_TILES * E)))

    # expert_iota is constant — compute outside the T-tile loop
    expert_iota = nl.ndarray((P, E), dtype=nl.uint32, buffer=nl.sbuf,
                             name="expert_iota")
    nisa.iota(dst=expert_iota, pattern=[[1, E]], offset=0, channel_multiplier=0)

    # ----------------------------------------------------------------
    # T-tile loop: process T_local tokens in T_TILE-sized chunks.
    # Using plain Python range() so the loop is fully unrolled at trace
    # time — each iteration gets independently-named buffers.
    # ----------------------------------------------------------------
    w_tiles = []
    for ht in nl.affine_range(NUM_H_TILES):
        w_tile = nl.ndarray((P, E), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"w_tile_{ht}")
        nisa.dma_copy(
            dst=w_tile,
            src=w_sb.ap(
                [[NUM_H_TILES * E, P], [1, E]], offset=ht * E
            ),
        )
        w_tiles.append(w_tile)
            
            
    for t_tile in nl.affine_range(num_t_tiles):
        t_off = T_offset + t_tile * T_TILE   # absolute token offset in [T, H] tensor

        # ----------------------------------------------------------------
        # Matmul: [T_TILE, H] @ [H, E] → [T_TILE, E]
        # ----------------------------------------------------------------
        router_logits_psum = nl.zeros((T_TILE, E), dtype=nl.float32, buffer=nl.psum,
                                      name=f"rl_psum_{t_tile}")
        
        x_tiles = []
        for ht in nl.affine_range(NUM_H_TILES):
            x_tile = nl.ndarray((P, T_TILE), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"x_tile_{t_tile}_{ht}")
            nisa.dma_copy(
                dst=x_tile,
                src=x.ap(
                    [[NUM_H_TILES, P], [H, T_TILE]],
                    offset=t_off * H + ht,
                ),
            )
            x_tiles.append(x_tile)

        for ht in nl.affine_range(NUM_H_TILES):
            nisa.nc_matmul(dst=router_logits_psum, stationary=x_tiles[ht], moving=w_tiles[ht])

        # ----------------------------------------------------------------
        # Copy PSUM → SBUF and store router_logits slice to HBM
        # ----------------------------------------------------------------
        router_logits_sb = nl.ndarray((T_TILE, E), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"rl_sb_{t_tile}")
        nisa.tensor_copy(dst=router_logits_sb, src=router_logits_psum)

        # Round fp32 PSUM accum to bf16 grid to match CPU bf16 matmul precision
        rl_bf16_tmp = nl.ndarray((T_TILE, E), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"rl_bf16_{t_tile}")
        nisa.tensor_copy(dst=rl_bf16_tmp, src=router_logits_sb)   # fp32 → bf16
        nisa.tensor_copy(dst=router_logits_sb, src=rl_bf16_tmp)   # bf16 → fp32

        nisa.dma_copy(
            dst=rl_out.ap([[E, T_TILE], [1, E]], offset=t_off * E),
            src=router_logits_sb,
        )

        # ----------------------------------------------------------------
        # Softmax (numerically stable)
        # ----------------------------------------------------------------
        affinities_sb = nl.ndarray((T_TILE, E), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"affi_sb_{t_tile}")

        negmax_sb = nl.ndarray((T_TILE, 1), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"negmax_sb_{t_tile}")
        nisa.tensor_reduce(
            dst=negmax_sb,
            op=nl.maximum,
            data=router_logits_sb,
            axis=1,
            negate=True,
            keepdims=True,
        )

        inv_sum_sb = nl.ndarray((T_TILE, 1), dtype=nl.float32, buffer=nl.sbuf,
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
        topk_vals_sb = nl.ndarray((T_TILE, K), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"topk_vals_sb_{t_tile}")
        topk_idx_sb  = nl.ndarray((T_TILE, K), dtype=nl.uint32,  buffer=nl.sbuf,
                                  name=f"topk_idx_sb_{t_tile}")

        top8_buf = nl.ndarray((T_TILE, 8), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"top8_buf_{t_tile}")
        nisa.max8(dst=top8_buf, src=affinities_sb)
        nisa.tensor_copy(dst=topk_vals_sb, src=top8_buf[:, :K])

        idx8_buf = nl.ndarray((T_TILE, 8), dtype=nl.uint32, buffer=nl.sbuf,
                              name=f"idx8_buf_{t_tile}")
        nisa.nc_find_index8(dst=idx8_buf, data=affinities_sb, vals=top8_buf)
        nisa.tensor_copy(dst=topk_idx_sb, src=idx8_buf[:, :K])

        topk_idx_fp32_sb = nl.ndarray((T_TILE, K), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"topk_idx_fp32_sb_{t_tile}")
        nisa.tensor_copy(dst=topk_idx_fp32_sb, src=topk_idx_sb)

        nisa.dma_copy(
            dst=ei_out.ap([[K, T_TILE], [1, K]], offset=t_off * K),
            src=topk_idx_sb,
        )

        # ----------------------------------------------------------------
        # L1 normalization of top-K affinities
        # ----------------------------------------------------------------
        sum_topk_sb = nl.ndarray((T_TILE, 1), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"sum_topk_sb_{t_tile}")
        nisa.tensor_reduce(
            dst=sum_topk_sb,
            op=nl.add,
            data=topk_vals_sb,
            axis=1,
            keepdims=True,
        )
        nisa.reciprocal(dst=sum_topk_sb, data=sum_topk_sb)

        topk_vals_norm_sb = nl.ndarray((T_TILE, K), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"topk_vals_norm_sb_{t_tile}")
        nisa.tensor_scalar(
            dst=topk_vals_norm_sb,
            data=topk_vals_sb,
            op0=nl.multiply,
            operand0=sum_topk_sb,
        )

        # ----------------------------------------------------------------
        # One-hot scatter: build [T_TILE, E] mask
        # ----------------------------------------------------------------
        mask_sb = nl.ndarray((T_TILE, E), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"mask_sb_{t_tile}")
        nisa.memset(dst=mask_sb, value=0.0)

        check_buf = nl.ndarray((T_TILE, E), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"check_buf_{t_tile}")
        for k_slot in nl.affine_range(K):
            nisa.tensor_scalar(
                dst=check_buf[:T_TILE, :],
                op0=nl.equal,
                data=expert_iota[:T_TILE, :],
                operand0=topk_idx_fp32_sb[:T_TILE, k_slot],
            )
            nisa.tensor_tensor(
                dst=mask_sb[:T_TILE, :],
                data1=mask_sb[:T_TILE, :],
                op=nl.add,
                data2=check_buf[:T_TILE, :],
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

        scattered_sb = nl.ndarray((T_TILE, E), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"scattered_sb_{t_tile}")
        nisa.tensor_tensor(
            dst=scattered_sb,
            data1=mask_sb,
            op=nl.multiply,
            data2=affinities_sb,
        )

        # ----------------------------------------------------------------
        # Store scattered expert_affinities slice to HBM
        # ----------------------------------------------------------------
        nisa.dma_copy(
            dst=ea_out.ap([[E, T_TILE], [1, E]], offset=t_off * E),
            src=scattered_sb,
        )

    # Barrier ensures both cores have written their T_local rows before the
    # caller reads the full [T, E] expert_affinities tensor.
    # core_barrier stays OUTSIDE the T-tile loop.
    core_barrier(ea_out, cores=[0, 1])

    return rl_out, ea_out, ei_out
