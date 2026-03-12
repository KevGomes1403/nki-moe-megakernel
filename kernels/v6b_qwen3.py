"""
v6b MoE kernel adapted for Qwen3-30B-A3B integration.

Pre-gathered weight variant: the caller gathers K expert weights per token
in PyTorch (torch.index_select) and passes 3D tensors [T*K, H, 2*I] and
[T*K, I, H]. The kernel reshapes these to 2D internally (like the v3
all-expert kernel) and uses compile-time flat offsets. This completely
avoids DGE indirect addressing which causes out-of-bound errors in E2E.

All performance optimizations are preserved:
  - Coalesced gate+up DMA (full [P, two_I] row per h_t)
  - Coalesced down DMA (full [P, H] row per i_t)
  - Widened down nc_matmul ([P, H_PER_GROUP*P] moving blocks)
  - PSUM tile reuse across experts, SiLU+mul fusion

Hardcoded for Qwen3-30B-A3B with tp_degree=4 (padded):
    H = 2048, I = 256, K = 8, bf16 weights.
"""

import nki
import nki.language as nl
import nki.isa as nisa

# ── Qwen3-30B-A3B constants (tp=4, padded intermediate) ──────────────────
_I_SIZE = 256
_TWO_I  = 2 * _I_SIZE   # 512
_P      = 128
_NUM_I_TILES  = _I_SIZE // _P   # 2
_NUM_GU_TILES = _TWO_I  // _P   # 4


@nki.jit(platform_target="trn2")
def nki_moe_v6b_qwen3(
    hidden_states,        # [T, H]        bf16
    gate_up_gathered,     # [T*K, H, 2*I] bf16  — pre-gathered per-expert weights
    down_gathered,        # [T*K, I, H]   bf16  — pre-gathered per-expert weights
    routing_weights_k,    # [T, K]        float32
):
    """
    Sparse Token-Parallel MoE kernel with pre-gathered expert weights.

    The caller gathers K experts' weights per token using torch.index_select,
    yielding 3D tensors. This kernel reshapes them to 2D internally (following
    the pattern from the v3 all-expert kernel) and uses compile-time flat
    offsets, completely avoiding DGE indirect addressing.
    """
    T      = hidden_states.shape[0]
    H      = hidden_states.shape[1]
    K      = routing_weights_k.shape[1]
    P      = _P
    I_size = _I_SIZE
    two_I  = _TWO_I

    num_h_tiles  = H // P
    num_i_tiles  = _NUM_I_TILES
    num_gu_tiles = _NUM_GU_TILES

    # --- Widened down projection grouping ---
    H_PER_GROUP = min(4, num_h_tiles)
    num_h_groups = num_h_tiles // H_PER_GROUP

    # Reshape 3D pre-gathered weight tensors → 2D for flat .ap() indexing
    # (same pattern as v3 all-expert kernel on the full [E, H, 2I] tensor).
    # gate_up_gathered: [T*K, H, 2I] → [T*K*H, 2I]
    # down_gathered:    [T*K, I, H]   → [T*K*I, H]
    TK = gate_up_gathered.shape[0]   # T*K
    gate_up_flat = gate_up_gathered.reshape((TK * H, two_I))
    down_flat    = down_gathered.reshape((TK * I_size, H))

    # HBM output buffer
    output = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # =========================================================================
    # Pre-allocate PSUM tiles (reused across T and K loops).
    # =========================================================================
    gu_psum_tiles = []
    for _gu_t in nl.affine_range(num_gu_tiles):
        gu_psum_tiles.append(
            nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
        )

    down_psum_groups = []
    for _h_g in nl.affine_range(num_h_groups):
        down_psum_groups.append(
            nl.ndarray((1, H_PER_GROUP * P), dtype=nl.float32, buffer=nl.psum)
        )

    # =========================================================================
    # TOKEN LOOP — sequential
    # =========================================================================
    for t in range(T):

        # Load token t's hidden state tiles
        h_tiles_sb = []
        for h_t in nl.affine_range(num_h_tiles):
            h_off = h_t * P
            h_sb  = nl.ndarray((P, 1), dtype=hidden_states.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=h_sb,
                src=hidden_states.ap(
                    pattern=[[1, P], [H, 1]],
                    offset=t * H + h_off,
                ),
            )
            h_tiles_sb.append(h_sb)

        # Output accumulator (widened)
        out_accum = []
        for _h_g in nl.affine_range(num_h_groups):
            tmp = nl.ndarray((1, H_PER_GROUP * P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=tmp, value=0)
            out_accum.append(tmp)

        # -------------------------------------------------------------------
        # K-EXPERT LOOP — sequential (accumulates into out_accum)
        # -------------------------------------------------------------------
        for k in range(K):

            # Index into pre-gathered flat weight arrays.
            # gate_up_flat is [TK*H, 2I]: expert k of token t starts at row (t*K+k)*H.
            # down_flat is [TK*I, H]: expert k of token t starts at row (t*K+k)*I.
            gu_base = (t * K + k) * H       # row offset into gate_up_flat
            dw_base = (t * K + k) * I_size   # row offset into down_flat

            # Load routing weight for (t, k)
            rw_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=rw_scalar,
                src=routing_weights_k.ap(
                    pattern=[[K, 1], [1, 1]],
                    offset=t * K + k,
                ),
            )

            # ---------------------------------------------------------------
            # Stage 1: Gate+Up projection
            # ---------------------------------------------------------------

            for gu_t in nl.affine_range(num_gu_tiles):
                nisa.memset(dst=gu_psum_tiles[gu_t], value=0)

            for h_t in range(num_h_tiles):
                h_off  = h_t * P
                h_tile = h_tiles_sb[h_t]

                # Coalesced gate+up weight DMA: one [P, two_I] row.
                w_row = nl.ndarray((P, two_I), dtype=gate_up_flat.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=w_row,
                    src=gate_up_flat.ap(
                        pattern=[[two_I, P], [1, two_I]],
                        offset=(gu_base + h_off) * two_I,
                    ),
                )

                for gu_t in nl.affine_range(num_gu_tiles):
                    gu_off = gu_t * P
                    nisa.nc_matmul(
                        dst=gu_psum_tiles[gu_t],
                        stationary=w_row[0:P, gu_off:gu_off+P],
                        moving=h_tile,
                    )

            # PSUM → SBUF, SiLU + multiply + cast to bf16
            act_bf16_tiles = []
            for i_t in nl.affine_range(num_i_tiles):
                gate_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=gate_sb, src=gu_psum_tiles[i_t])

                up_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=up_sb, src=gu_psum_tiles[num_i_tiles + i_t])

                gate_act = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=gate_act, op=nl.silu, data=gate_sb, scale=1.0)

                act_f32 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=act_f32,
                    data1=gate_act,
                    data2=up_sb,
                    op=nl.multiply,
                )

                act_bf16 = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=act_bf16, src=act_f32)
                act_bf16_tiles.append(act_bf16)

            # ---------------------------------------------------------------
            # Stage 2: Down projection (coalesced DMA + widened matmul)
            # ---------------------------------------------------------------

            dw_rows = []
            for i_t in nl.affine_range(num_i_tiles):
                i_off = i_t * P
                dw_row = nl.ndarray((P, H), dtype=down_flat.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=dw_row,
                    src=down_flat.ap(
                        pattern=[[H, P], [1, H]],
                        offset=(dw_base + i_off) * H,
                    ),
                )
                dw_rows.append(dw_row)

            for h_g in nl.affine_range(num_h_groups):
                h_base = h_g * H_PER_GROUP * P
                nisa.memset(dst=down_psum_groups[h_g], value=0)

                for i_t in range(num_i_tiles):
                    nisa.nc_matmul(
                        dst=down_psum_groups[h_g],
                        stationary=act_bf16_tiles[i_t],
                        moving=dw_rows[i_t][0:P, h_base:h_base+H_PER_GROUP*P],
                    )

            for h_g in nl.affine_range(num_h_groups):
                down_sb_wide = nl.ndarray(
                    (1, H_PER_GROUP * P), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=down_sb_wide, src=down_psum_groups[h_g])

                scaled_wide = nl.ndarray(
                    (1, H_PER_GROUP * P), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=scaled_wide,
                    data=down_sb_wide,
                    op0=nl.multiply,
                    operand0=rw_scalar,
                )

                nisa.tensor_tensor(
                    dst=out_accum[h_g],
                    data1=out_accum[h_g],
                    data2=scaled_wide,
                    op=nl.add,
                )

        # -------------------------------------------------------------------
        # Store token t's output
        # -------------------------------------------------------------------
        for h_g in nl.affine_range(num_h_groups):
            h_base   = h_g * H_PER_GROUP * P
            out_cast = nl.ndarray(
                (1, H_PER_GROUP * P), dtype=hidden_states.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=out_cast, src=out_accum[h_g])
            nisa.dma_copy(
                dst=output.ap(
                    pattern=[[H, 1], [1, H_PER_GROUP * P]],
                    offset=t * H + h_base,
                ),
                src=out_cast,
            )

    return output
