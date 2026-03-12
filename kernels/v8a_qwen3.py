"""
v8a MoE kernel for Qwen3-30B-A3B integration — E2E-safe DGE with full weight tensors.

Fixes the DGE scalar_offset absolute-SBUF-address bug that caused v7a to produce
wrong outputs in E2E model compilation (see common-pitfalls.md: "DGE scalar_offset
absolute SBUF address bug"):

  BUG (v7a): expert_idx_sb.ap(offset=t*K+k) bakes an absolute SBUF address.
  In standalone tests the SBUF layout is predictable; in E2E compilation
  earlier model tensors shift the allocation so the DGE reads from activation
  residuals instead of expert indices → garbage expert selection.

  FIX (Plan A): Per k-iteration, copy the scalar expert index from HBM into
  eid_scratch[0,0], then reference via eid_scratch.ap(offset=0). The (128,1)
  shape preserves IndirectDimMaxIndex=127 for the compiler (NCC_IBIR030).
  Allocated outside the k loop so its SBUF base address is fixed; the DMA
  overwrites content before any DGE read each iteration.

Also incorporates:
  Plan B: Routing weights hoisted from HBM into SBUF before the k loop.
  One [1, K] DMA replaces K per-k scalar HBM round-trips (one per expert).

Accepts full [E, H, 2I] and [E, I, H] weight tensors directly — no
torch.index_select pre-gathering in the forward pass.

Optimizations preserved from v7a:
  - Coalesced gate+up DMA (one [P, two_I] HBM row per h_t, single DMA)
  - Coalesced down DMA (one [P, H] HBM row per i_t, single DMA)
  - Widened down nc_matmul ([P, H_PER_GROUP*P] moving blocks, fills 512-wide TE)
  - PSUM tile reuse across T and K loops

Shape constraints:
    T <= 128     (token count fits in partition dimension)
    H % 128 == 0
    I % 128 == 0
    E <= 128     (IndirectDimMaxIndex <= 127, derived from eid_scratch shape)
"""

import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit(platform_target="trn2")
def nki_moe_v8a_qwen3(
    hidden_states,        # [T, H]      bf16  — token hidden states
    gate_up_weights,      # [E, H, 2*I] bf16  — fused gate+up projection weights
    down_weights,         # [E, I, H]   bf16  — down projection weights
    expert_indices,       # [T, K]      int32 — top-K expert indices per token
    routing_weights_k,    # [T, K]      float32 — routing scale per (token, expert)
):
    """
    Sparse Token-Parallel MoE FFN kernel with E2E-safe DGE weight access.

    For each token t and each of its K selected experts:
      1. Gate+up: hidden[t] @ gate_up_weights[e]  → [2I]
      2. SiLU(gate) * up                          → [I]
      3. Down:    act @ down_weights[e]            → [H]
      4. output[t] += routing_weights_k[t,k] * down_result

    Kernel reads gate_up_weights and down_weights using DGE (indirect_dim=0)
    with a freshly-loaded per-k scalar offset to avoid the SBUF absolute-
    address baking bug (see module docstring and common-pitfalls.md).
    """
    T      = hidden_states.shape[0]
    H      = hidden_states.shape[1]
    # E shape dimension is only used by the compiler for IndirectDimMaxIndex
    two_I  = gate_up_weights.shape[2]
    I_size = two_I // 2
    K      = expert_indices.shape[1]
    P      = 128   # partition dimension size on trn2

    num_h_tiles  = H      // P   # hidden-dim P-wide tile count
    num_i_tiles  = I_size // P   # intermediate P-wide tile count
    num_gu_tiles = two_I  // P   # gate+up column tile count (2 × num_i_tiles)

    # Widened down matmul: group H_PER_GROUP h_tiles → one [P, H_PER_GROUP*P] nc_matmul.
    # H_PER_GROUP=4 fills the Tensor Engine's 512-wide moving dimension (hardware max).
    H_PER_GROUP  = min(4, num_h_tiles)
    num_h_groups = num_h_tiles // H_PER_GROUP

    # ── HBM output ────────────────────────────────────────────────────────────
    output = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # =========================================================================
    # Plan A: DGE expert-index scratch buffer.
    #
    # Allocated once here so its SBUF base address is fixed throughout the
    # kernel. Content is overwritten by DMA at the start of each k iteration
    # before any DGE read, so correctness is guaranteed.
    #
    # Shape (128, 1): IndirectDimMaxIndex = 127 = E-1 for Qwen3-30B-A3B (E=128).
    # This shape is required by the compiler (NCC_IBIR030: scalar_offset must
    # encode IndirectDimMaxIndex via its .ap() descriptor's first-dim count).
    # =========================================================================
    eid_scratch = nl.ndarray((128, 1), dtype=expert_indices.dtype, buffer=nl.sbuf)

    # =========================================================================
    # Pre-allocate PSUM tiles — reused across T and K loops to avoid repeated
    # SBUF allocation pressure.
    #   gu_psum_tiles[gu_t]   : [P, 1] f32 PSUM — gate+up result per gu column
    #   down_psum_groups[h_g] : [1, H_PER_GROUP*P] f32 PSUM — widened down result
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
    # TOKEN LOOP — sequential (one token at a time, SBUF buffers reused).
    # =========================================================================
    for t in range(T):

        # Load token t's hidden state tiles from HBM into SBUF.
        # nl.affine_range: DMA loads are independent → enables DMA-compute overlap.
        h_tiles_sb = []
        for h_t in nl.affine_range(num_h_tiles):
            h_off = h_t * P
            h_sb  = nl.ndarray((P, 1), dtype=hidden_states.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=h_sb,
                src=hidden_states.ap(
                    # partition: P rows of hidden_states[t], spaced H apart
                    # free:      1 element (scalar; hidden dim = partition dim here)
                    pattern=[[1, P], [H, 1]],
                    offset=t * H + h_off,
                ),
            )
            h_tiles_sb.append(h_sb)

        # Output accumulator — zeroed once; K expert contributions added below.
        out_accum = []
        for _h_g in nl.affine_range(num_h_groups):
            tmp = nl.ndarray((1, H_PER_GROUP * P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=tmp, value=0)
            out_accum.append(tmp)

        # -------------------------------------------------------------------
        # Plan B: Hoist all K routing weights for token t into SBUF before the
        # k loop. One [1, K] DMA replaces K per-k scalar HBM round-trips.
        # routing_weights_k[t, 0:K] is a contiguous row → single DMA suffices.
        # -------------------------------------------------------------------
        rw_k_sb = nl.ndarray((1, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=rw_k_sb,
            src=routing_weights_k.ap(
                # 1 partition reads K contiguous elements starting at row t
                pattern=[[K, 1], [1, K]],
                offset=t * K,
            ),
        )

        # -------------------------------------------------------------------
        # K-EXPERT LOOP — sequential: PSUM accumulation for gate+up and down
        # creates loop-carried dependencies so range() is correct here.
        # -------------------------------------------------------------------
        for k in range(K):

            # ---------------------------------------------------------------
            # Plan A: Load expert index e = expert_indices[t, k] from HBM
            # into eid_scratch[0, 0], then reference at offset=0.
            #
            # Why HBM .ap() is safe: HBM offsets resolve to absolute HBM
            # addresses, which are stable across compilations.
            #
            # Why SBUF .ap(offset=0) is safe: offset=0 always resolves to the
            # base of eid_scratch's SBUF allocation, which is fixed for the
            # lifetime of this kernel invocation.
            # ---------------------------------------------------------------
            nisa.dma_copy(
                dst=eid_scratch[0:1, 0:1],
                src=expert_indices.ap(
                    pattern=[[K, 1], [1, 1]],   # 1 element at position [t, k]
                    offset=t * K + k,
                ),
            )
            # offset=0: references eid_scratch base — never bakes a wrong address
            eid_offset = eid_scratch.ap(pattern=[[1, 1], [1, 1]], offset=0)

            # Read this expert's routing weight from SBUF (Plan B).
            rw_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=rw_scalar, src=rw_k_sb[0:1, k:k+1])

            # ---------------------------------------------------------------
            # Stage 1: Gate+Up projection
            #   hidden[t, :] @ gate_up_weights[e, :, :]  →  [2I]
            #   nc_matmul: dst += stationary.T @ moving
            #     stationary = w_row slice [P_h, P_gu]  (from coalesced row)
            #     moving     = h_tile       [P_h, 1]
            #     result     = [P_gu, 1]    stored in gu_psum_tiles[gu_t]
            # ---------------------------------------------------------------

            # Zero all gate+up PSUM accumulators before this expert's h_t loop.
            for gu_t in nl.affine_range(num_gu_tiles):
                nisa.memset(dst=gu_psum_tiles[gu_t], value=0)

            # h_t sequential: gu_psum_tiles accumulate across h_t (PSUM dep).
            for h_t in range(num_h_tiles):
                h_off  = h_t * P
                h_tile = h_tiles_sb[h_t]

                # Coalesced gate+up DMA: load full [P, two_I] row in one DMA.
                # DGE (indirect_dim=0): expert dim of gate_up_weights is resolved
                # at runtime via eid_offset (the (128,1) scratch loaded above).
                # offset = h_off * two_I: selects row h_t within expert e's [H, 2I].
                w_row = nl.ndarray((P, two_I), dtype=gate_up_weights.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=w_row,
                    src=gate_up_weights.ap(
                        # partition: P consecutive rows of gate_up[e, h_off:h_off+P, :]
                        # free:      two_I contiguous columns
                        pattern=[[two_I, P], [1, two_I]],
                        offset=h_off * two_I,   # row offset within expert's [H, 2I]
                        scalar_offset=eid_offset,
                        indirect_dim=0,
                    ),
                )

                # Slice the loaded [P, two_I] row for each gu_t's [P, P] tile.
                # nl.affine_range: gu_t slices are independent (different PSUM tiles).
                for gu_t in nl.affine_range(num_gu_tiles):
                    gu_off = gu_t * P
                    nisa.nc_matmul(
                        dst=gu_psum_tiles[gu_t],
                        stationary=w_row[0:P, gu_off:gu_off + P],
                        moving=h_tile,
                    )

            # PSUM → SBUF, SiLU(gate) * up, cast to bf16.
            # nl.affine_range: i_t tiles are independent (different PSUM sources).
            act_bf16_tiles = []
            for i_t in nl.affine_range(num_i_tiles):
                # Gate half:  gu_psum_tiles[i_t]
                # Up half:    gu_psum_tiles[i_t + num_i_tiles]
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

                # Cast f32 → bf16 once here (hoisted outside down loop).
                act_bf16 = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=act_bf16, src=act_f32)
                act_bf16_tiles.append(act_bf16)

            # ---------------------------------------------------------------
            # Stage 2: Down projection (coalesced DMA + widened matmul)
            #   act[I] @ down_weights[e, :, :]  →  [H]
            #   nc_matmul: dst += stationary.T @ moving
            #     stationary = act_tile  [P_i, 1]
            #     moving     = dw_slice  [P_i, H_PER_GROUP*P]  (widened)
            #     result     = [1, H_PER_GROUP*P]
            # ---------------------------------------------------------------

            # Pre-load all down weight rows for this expert (coalesced DMA).
            # nl.affine_range: i_t DMAs are independent (different HBM rows).
            dw_rows = []
            for i_t in nl.affine_range(num_i_tiles):
                i_off  = i_t * P
                dw_row = nl.ndarray((P, H), dtype=down_weights.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=dw_row,
                    src=down_weights.ap(
                        # partition: P consecutive rows of down[e, i_off:i_off+P, :]
                        # free:      H contiguous columns
                        pattern=[[H, P], [1, H]],
                        offset=i_off * H,       # row offset within expert's [I, H]
                        scalar_offset=eid_offset,
                        indirect_dim=0,
                    ),
                )
                dw_rows.append(dw_row)

            # Widened matmul: h_groups are independent (nl.affine_range),
            # i_t is sequential (PSUM accumulation dependency).
            for h_g in nl.affine_range(num_h_groups):
                h_base = h_g * H_PER_GROUP * P
                nisa.memset(dst=down_psum_groups[h_g], value=0)

                for i_t in range(num_i_tiles):
                    # Slice [P, H_PER_GROUP*P] from the pre-loaded [P, H] row.
                    # This fuses v6b's coalesced load with v6a's widened matmul:
                    # 512-wide moving dimension fills the Tensor Engine optimally.
                    nisa.nc_matmul(
                        dst=down_psum_groups[h_g],
                        stationary=act_bf16_tiles[i_t],
                        moving=dw_rows[i_t][0:P, h_base:h_base + H_PER_GROUP * P],
                    )

            # Scale by routing weight and accumulate into output accumulator.
            # nl.affine_range: h_g groups are independent.
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
                    operand0=rw_scalar,   # [1, 1] f32 from SBUF (Plan B)
                )

                nisa.tensor_tensor(
                    dst=out_accum[h_g],
                    data1=out_accum[h_g],
                    data2=scaled_wide,
                    op=nl.add,
                )

        # -------------------------------------------------------------------
        # Store token t's output: cast f32 → bf16, then DMA to HBM.
        # nl.affine_range: h_g stores are independent.
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
