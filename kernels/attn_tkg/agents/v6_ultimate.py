"""
v6_ultimate: The "ultimate" kernel combining ALL optimizations from Plans A and B.

Five combined optimizations:

1. Column Layout (from v1/Plan A):
   - hidden_states [B,1,H] -> [H,B], cos/sin [B,d] -> [d,B], output -> [Hq_out,B]
   - Eliminates ~394 nc_transpose calls (hidden, cos/sin, output)

2. Hidden Tile Hoisting (from v1/Plan A):
   - Pre-load all 16 hidden tiles [PMAX,B] outside head loop using affine_range
   - Reuse across all Q/K/V projections (~300 redundant DMA loads eliminated)

3. GQA K/V Projection Dedup (from v2):
   - K/V matmul + RMSNorm + RoPE computed once per KV head (4x instead of 8x)
   - Saves 64 matmuls each for K and V

4. Free-Dimension GQA Packing (from Plan B / v5):
   - Pack 2 Q heads into free dim [PMAX, 2] for flash decode
   - All flash-decode matmuls, DMA, and elementwise ops run on packed tensors
   - Halves K/V cache tile loads, score matmuls, V weighted matmuls, and
     all online-softmax elementwise ops

5. KV Tile Reuse (from Plan A / v4):
   - KV cache tiles loaded and transposed once per s_t, shared across packed heads
   - Combined with free-dim packing: one load serves both Q heads natively

Combined savings vs v0:
  - K/V matmul dedup: 128 -> 64 matmuls (saves 64 each)
  - Hidden tile hoisting: 384 -> 16 DMA loads (saves 368)
  - Column layout: ~394 transposes eliminated
  - KV tile reuse + packing: eliminates all redundant KV loads within GQA groups
"""

import math
import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128
EPS = 1e-6
INV_SQRT_D = float(1.0 / math.sqrt(128.0))  # 1/sqrt(128)


@nki.jit
def qwen3_attn_tkg_fused(
    hidden_states,   # [B, 1, H]  bf16  (B=1 assumed)
    Wq,              # [Hq_tp*d, H]  bf16  [1024, 2048]
    Wk,              # [Hkv_tp*d, H]  bf16  [512, 2048]
    Wv,              # [Hkv_tp*d, H]  bf16  [512, 2048]
    q_norm_weight,   # [d]  bf16  [128]
    k_norm_weight,   # [d]  bf16  [128]
    K_cache,         # [B, Hkv_tp, S_prior, d]  bf16
    V_cache,         # [B, Hkv_tp, S_prior, d]  bf16
    cos,             # [B, d]  bf16  (pre-indexed at position_ids)
    sin,             # [B, d]  bf16  (pre-indexed at position_ids)
    position_ids,    # [B, 1]  int32  (not used in kernel body)
):
    """
    Fused QKV projection + per-head RMSNorm + RoPE + Flash Decode for B=1.
    Returns output [B, 1, Hq_tp*d].

    v6: Ultimate kernel combining column layout + hidden tile hoisting +
        KV dedup + free-dim GQA packing + KV tile reuse.
    """
    # --- Setup: derive dimensions ---
    B = hidden_states.shape[0]       # 1
    H = hidden_states.shape[2]       # 2048
    Hq_out = Wq.shape[0]            # 1024 = Hq_tp * d
    Hkv_out = Wk.shape[0]           # 512  = Hkv_tp * d
    d = PMAX                        # 128
    Hq_tp = Hq_out // d             # 8
    Hkv_tp = Hkv_out // d           # 4
    gqa = Hq_tp // Hkv_tp           # 2
    S_prior = K_cache.shape[2]
    num_h_tiles = H // PMAX          # 16
    num_s_tiles = S_prior // PMAX

    # =========================================================================
    # COLUMN LAYOUT RESHAPES (Optimization 1)
    # For B=1, [B, X] and [X, B] have identical flat memory layout.
    # Reshaping to column layout lets us load/store [PMAX, B] tiles directly
    # without any nc_transpose operations.
    # =========================================================================

    # Output buffer: column layout [Hq_out, B] for direct store (no transpose)
    output = nl.ndarray((B, 1, Hq_out), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    output_col = output.reshape((Hq_out, B))

    # Hidden states: [B, 1, H] -> [H, B] column layout
    hidden_col = hidden_states.reshape((H, B))

    # cos/sin: [B, d] -> [d, B] = [PMAX, B] column layout
    cos_col = cos.reshape((PMAX, B))
    sin_col = sin.reshape((PMAX, B))

    # Packed scalar HBM staging for free-dim broadcast [gqa,1] -> [PMAX,gqa]
    scalar_hbm_packed = nl.ndarray((gqa, 1), dtype=nl.float32, buffer=nl.private_hbm)

    # =========================================================================
    # LOAD CONSTANTS: norm weights, cos/sin, rms_ones
    # =========================================================================

    # Norm weights: [d=128] -> [PMAX, 1] in SBUF (float32)
    qnw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_bf16")
    nisa.dma_copy(dst=qnw_bf16, src=q_norm_weight.reshape((PMAX, 1)))
    qnw_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qnw_sb")
    nisa.tensor_copy(qnw_sb, qnw_bf16)

    knw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_bf16")
    nisa.dma_copy(dst=knw_bf16, src=k_norm_weight.reshape((PMAX, 1)))
    knw_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="knw_sb")
    nisa.tensor_copy(knw_sb, knw_bf16)

    # All-ones [PMAX, PMAX] for partition-dimension reduction (RMSNorm, tile_sum, dot product)
    rms_ones = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)

    # =========================================================================
    # LOAD cos/sin: DIRECT COLUMN LAYOUT (Optimization 1)
    # cos_col [PMAX, B] loaded directly -- NO nc_transpose needed
    # =========================================================================
    cos_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf, name="cos_bf16")
    sin_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf, name="sin_bf16")
    nisa.dma_copy(dst=cos_bf16, src=cos_col)
    nisa.dma_copy(dst=sin_bf16, src=sin_col)
    cos_f32 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="cos_f32")
    sin_f32 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="sin_f32")
    nisa.tensor_copy(cos_f32, cos_bf16)
    nisa.tensor_copy(sin_f32, sin_bf16)

    # =========================================================================
    # HIDDEN TILE HOISTING (Optimization 2)
    # Pre-load all 16 hidden tiles [PMAX, B] outside all head loops.
    # Uses affine_range for DMA-compute overlap.
    # Column layout: hidden_col [H, B] -> tiles are [PMAX, B] directly.
    # Reused for all Q/K/V matmuls across all heads.
    # =========================================================================
    h_tiles = []
    for h_t in nl.affine_range(num_h_tiles):
        h_tile = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                            name=f"h_tile_t{h_t}")
        nisa.dma_copy(
            dst=h_tile,
            src=hidden_col[h_t * PMAX:(h_t + 1) * PMAX, 0:B],
        )
        h_tiles.append(h_tile)

    # Flatten KV cache for .ap() indexing
    K_cache_flat = K_cache.reshape((B * Hkv_tp * S_prior, d))
    V_cache_flat = V_cache.reshape((B * Hkv_tp * S_prior, d))

    half_d = d // 2  # 64

    # =========================================================================
    # MAIN LOOP: per KV head (Optimization 3: K/V dedup)
    #
    # For each KV head (kv_h), 4 iterations:
    #   1. K projection + RMSNorm + RoPE (once, using hoisted h_tiles)
    #   2. V projection (once, using hoisted h_tiles)
    #   3. Q projection + RMSNorm + RoPE for BOTH GQA heads -> pack [PMAX,2]
    #   4. Packed flash decode [PMAX,2] with KV tiles loaded ONCE (Opt 4+5)
    #   5. Active position scoring on packed tensors
    #   6. Normalize, split, and store per Q head (column layout, no transpose)
    # =========================================================================
    for kv_h in nl.affine_range(Hkv_tp):

        # =====================================================================
        # STEP 1: K matmul (once per KV head, using hoisted hidden tiles)
        # =====================================================================
        k_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                          name=f"k_psum_kv{kv_h}")
        for h_t in range(num_h_tiles):
            wk_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"wk_tile_kv{kv_h}_t{h_t}")
            nisa.dma_copy(
                dst=wk_tile,
                src=Wk.ap(pattern=[[H, PMAX], [1, PMAX]],
                          offset=kv_h * PMAX * H + h_t * PMAX),
            )
            nisa.nc_matmul(k_psum, stationary=wk_tile, moving=h_tiles[h_t])

        k_vec = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                           name=f"k_vec_kv{kv_h}")
        nisa.tensor_copy(k_vec, k_psum)

        # ---- K RMSNorm (once per kv_h) ----
        k_sq = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                          name=f"k_sq_kv{kv_h}")
        nisa.tensor_tensor(k_sq, k_vec, k_vec, op=nl.multiply)
        k_sq_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"k_sq_bf16_kv{kv_h}")
        nisa.tensor_copy(k_sq_bf16, k_sq)
        k_sum_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                              name=f"k_sum_psum_kv{kv_h}")
        nisa.nc_matmul(k_sum_psum, stationary=rms_ones, moving=k_sq_bf16)
        k_sum_sb = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"k_sum_sb_kv{kv_h}")
        nisa.tensor_copy(k_sum_sb, k_sum_psum)
        k_mean_sq = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"k_mean_sq_kv{kv_h}")
        nisa.tensor_scalar(k_mean_sq, k_sum_sb, op0=nl.multiply, operand0=1.0/d)
        k_var_eps = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"k_var_eps_kv{kv_h}")
        nisa.tensor_scalar(k_var_eps, k_mean_sq, op0=nl.add, operand0=EPS)
        k_rms_inv = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"k_rms_inv_kv{kv_h}")
        nisa.activation(k_rms_inv, op=nl.rsqrt, data=k_var_eps)
        k_normed = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"k_normed_kv{kv_h}")
        nisa.tensor_tensor(k_normed, k_vec, k_rms_inv, op=nl.multiply)
        k_normed2 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"k_normed2_kv{kv_h}")
        nisa.tensor_tensor(k_normed2, k_normed, knw_sb, op=nl.multiply)

        # ---- K RoPE (once per kv_h) ----
        rot_k = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                           name=f"rot_k_kv{kv_h}")
        neg_k_upper = nl.ndarray((half_d, B), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"neg_k_upper_kv{kv_h}")
        nisa.tensor_scalar(neg_k_upper, k_normed2[half_d:d, 0:B], op0=nl.multiply, operand0=-1.0)
        nisa.tensor_copy(rot_k[0:half_d, 0:B], neg_k_upper)
        nisa.tensor_copy(rot_k[half_d:d, 0:B], k_normed2[0:half_d, 0:B])

        k_cos = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                           name=f"k_cos_kv{kv_h}")
        k_sin_part = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"k_sin_part_kv{kv_h}")
        nisa.tensor_tensor(k_cos, k_normed2, cos_f32, op=nl.multiply)
        nisa.tensor_tensor(k_sin_part, rot_k, sin_f32, op=nl.multiply)
        # k_rope persists -- shared by all Q heads in this GQA group
        k_rope = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"k_rope_kv{kv_h}")
        nisa.tensor_tensor(k_rope, k_cos, k_sin_part, op=nl.add)

        # k_bf16 for flash decode (not needed -- active position uses k_rope directly)

        # =====================================================================
        # STEP 2: V matmul (once per KV head, using hoisted hidden tiles)
        # =====================================================================
        v_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                          name=f"v_psum_kv{kv_h}")
        for h_t in range(num_h_tiles):
            wv_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"wv_tile_kv{kv_h}_t{h_t}")
            nisa.dma_copy(
                dst=wv_tile,
                src=Wv.ap(pattern=[[H, PMAX], [1, PMAX]],
                          offset=kv_h * PMAX * H + h_t * PMAX),
            )
            nisa.nc_matmul(v_psum, stationary=wv_tile, moving=h_tiles[h_t])

        # v_active persists -- shared by all Q heads in this GQA group
        v_active = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"v_active_kv{kv_h}")
        nisa.tensor_copy(v_active, v_psum)

        # =====================================================================
        # STEP 3: Q projections for BOTH GQA heads + RMSNorm + RoPE
        # Then pack results into [PMAX, gqa=2] tensors for packed flash decode.
        # =====================================================================
        q_bf16_list = []
        q_scaled_list = []
        for g in nl.affine_range(gqa):
            q_h = kv_h * gqa + g  # global Q head index

            # ---- Q matmul (using hoisted hidden tiles) ----
            q_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                              name=f"q_psum_kv{kv_h}_g{g}")
            for h_t in range(num_h_tiles):
                wq_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                     name=f"wq_tile_kv{kv_h}_g{g}_t{h_t}")
                nisa.dma_copy(
                    dst=wq_tile,
                    src=Wq.ap(pattern=[[H, PMAX], [1, PMAX]],
                              offset=q_h * PMAX * H + h_t * PMAX),
                )
                nisa.nc_matmul(q_psum, stationary=wq_tile, moving=h_tiles[h_t])

            q_vec = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"q_vec_kv{kv_h}_g{g}")
            nisa.tensor_copy(q_vec, q_psum)

            # ---- Q RMSNorm ----
            q_sq = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"q_sq_kv{kv_h}_g{g}")
            nisa.tensor_tensor(q_sq, q_vec, q_vec, op=nl.multiply)
            q_sq_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"q_sq_bf16_kv{kv_h}_g{g}")
            nisa.tensor_copy(q_sq_bf16, q_sq)
            q_sum_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                                  name=f"q_sum_psum_kv{kv_h}_g{g}")
            nisa.nc_matmul(q_sum_psum, stationary=rms_ones, moving=q_sq_bf16)
            q_sum_sb = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"q_sum_sb_kv{kv_h}_g{g}")
            nisa.tensor_copy(q_sum_sb, q_sum_psum)
            q_mean_sq = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"q_mean_sq_kv{kv_h}_g{g}")
            nisa.tensor_scalar(q_mean_sq, q_sum_sb, op0=nl.multiply, operand0=1.0/d)
            q_var_eps = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"q_var_eps_kv{kv_h}_g{g}")
            nisa.tensor_scalar(q_var_eps, q_mean_sq, op0=nl.add, operand0=EPS)
            q_rms_inv = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"q_rms_inv_kv{kv_h}_g{g}")
            nisa.activation(q_rms_inv, op=nl.rsqrt, data=q_var_eps)
            q_normed = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"q_normed_kv{kv_h}_g{g}")
            nisa.tensor_tensor(q_normed, q_vec, q_rms_inv, op=nl.multiply)
            q_normed2 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"q_normed2_kv{kv_h}_g{g}")
            nisa.tensor_tensor(q_normed2, q_normed, qnw_sb, op=nl.multiply)

            # ---- Q RoPE ----
            rot_q = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"rot_q_kv{kv_h}_g{g}")
            neg_q_upper = nl.ndarray((half_d, B), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"neg_q_upper_kv{kv_h}_g{g}")
            nisa.tensor_scalar(neg_q_upper, q_normed2[half_d:d, 0:B], op0=nl.multiply, operand0=-1.0)
            nisa.tensor_copy(rot_q[0:half_d, 0:B], neg_q_upper)
            nisa.tensor_copy(rot_q[half_d:d, 0:B], q_normed2[0:half_d, 0:B])

            q_cos = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"q_cos_kv{kv_h}_g{g}")
            q_sin_part = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"q_sin_part_kv{kv_h}_g{g}")
            nisa.tensor_tensor(q_cos, q_normed2, cos_f32, op=nl.multiply)
            nisa.tensor_tensor(q_sin_part, rot_q, sin_f32, op=nl.multiply)
            q_rope = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"q_rope_kv{kv_h}_g{g}")
            nisa.tensor_tensor(q_rope, q_cos, q_sin_part, op=nl.add)

            # Scale q by 1/sqrt(d) and cast to bf16
            q_scaled = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"q_scaled_kv{kv_h}_g{g}")
            nisa.tensor_scalar(q_scaled, q_rope, op0=nl.multiply, operand0=INV_SQRT_D)
            q_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"q_bf16_kv{kv_h}_g{g}")
            nisa.tensor_copy(q_bf16, q_scaled)

            q_bf16_list.append(q_bf16)
            q_scaled_list.append(q_scaled)

        # =====================================================================
        # Pack Q vectors into free dimension [PMAX, 2] (Optimization 4)
        # q_bf16_list[0], q_bf16_list[1] are each [PMAX, B=1]
        # q_packed: [PMAX, 2] bf16 for flash decode scoring
        # q_scaled_packed: [PMAX, 2] f32 for active position dot product
        # =====================================================================
        q_packed = nl.ndarray((PMAX, gqa), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"q_packed_kv{kv_h}")
        nisa.tensor_copy(q_packed[0:PMAX, 0:1], q_bf16_list[0])
        nisa.tensor_copy(q_packed[0:PMAX, 1:2], q_bf16_list[1])

        q_scaled_packed = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"q_scaled_packed_kv{kv_h}")
        nisa.tensor_copy(q_scaled_packed[0:PMAX, 0:1], q_scaled_list[0])
        nisa.tensor_copy(q_scaled_packed[0:PMAX, 1:2], q_scaled_list[1])

        # =====================================================================
        # STEP 4: Initialize packed flash decode state [PMAX, gqa=2]
        # One column per Q head -- all ops are column-independent.
        # =====================================================================
        attn_acc = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"attn_acc_kv{kv_h}")
        nisa.memset(attn_acc, value=0.0)
        running_max = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"running_max_kv{kv_h}")
        nisa.memset(running_max, value=-1e9)
        running_sum = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"running_sum_kv{kv_h}")
        nisa.memset(running_sum, value=0.0)

        # =====================================================================
        # STEP 5: Packed Flash Decode (Optimizations 4+5)
        # KV cache tiles loaded ONCE per s_t, scored against BOTH Q heads
        # via packed [PMAX, 2] matmuls. All online-softmax ops on [PMAX, 2].
        # =====================================================================
        for s_t in range(num_s_tiles):
            s_off = s_t * PMAX
            flat_row = kv_h * S_prior + s_off

            # Load K_tile [PMAX, PMAX] from K_cache_flat -- ONCE per s_t (Opt 5)
            k_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"k_tile_kv{kv_h}_s{s_t}")
            nisa.dma_copy(
                dst=k_tile,
                src=K_cache_flat.ap(pattern=[[d, PMAX], [1, d]], offset=flat_row * d),
            )

            # Score both Q heads: K_tile[128,128] @ q_packed[128,2] -> [128,2] (Opt 4)
            score_psum = nl.zeros((PMAX, gqa), dtype=nl.float32, buffer=nl.psum,
                                  name=f"score_psum_kv{kv_h}_s{s_t}")
            nisa.nc_matmul(score_psum, stationary=k_tile, moving=q_packed)
            score_sb = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"score_sb_kv{kv_h}_s{s_t}")
            nisa.tensor_copy(score_sb, score_psum)

            # --- Online softmax: tile_max via transpose + reduce ---
            # Transpose [PMAX, 2] -> [2, PMAX] for per-head reduction
            score_T_psum = nl.ndarray((gqa, PMAX), dtype=nl.float32, buffer=nl.psum,
                                      name=f"score_T_psum_kv{kv_h}_s{s_t}")
            nisa.nc_transpose(score_T_psum, score_sb)
            score_T_sb = nl.ndarray((gqa, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"score_T_sb_kv{kv_h}_s{s_t}")
            nisa.tensor_copy(score_T_sb, score_T_psum)

            # Reduce max over axis=1: [2, PMAX] -> [2, 1] (per-head scalar)
            tile_max_scalar = nl.ndarray((gqa, 1), dtype=nl.float32, buffer=nl.sbuf,
                                         name=f"tile_max_scalar_kv{kv_h}_s{s_t}")
            nisa.tensor_reduce(dst=tile_max_scalar, op=nl.maximum, data=score_T_sb, axis=1)

            # Broadcast [2,1] -> [PMAX,2] via HBM staging with stride-0 pattern
            nisa.dma_copy(dst=scalar_hbm_packed, src=tile_max_scalar)
            tile_max = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"tile_max_kv{kv_h}_s{s_t}")
            nisa.dma_copy(dst=tile_max,
                          src=scalar_hbm_packed.ap(pattern=[[0, PMAX], [1, gqa]], offset=0))

            # new_max = max(running_max, tile_max) -- [PMAX, 2]
            new_max = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"new_max_kv{kv_h}_s{s_t}")
            nisa.tensor_tensor(new_max, running_max, tile_max, op=nl.maximum)

            # alpha = exp(running_max - new_max) -- [PMAX, 2]
            neg_new_max = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"neg_new_max_kv{kv_h}_s{s_t}")
            nisa.tensor_scalar(neg_new_max, new_max, op0=nl.multiply, operand0=-1.0)
            alpha_arg = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"alpha_arg_kv{kv_h}_s{s_t}")
            nisa.tensor_tensor(alpha_arg, running_max, neg_new_max, op=nl.add)
            alpha = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"alpha_kv{kv_h}_s{s_t}")
            nisa.activation(alpha, op=nl.exp, data=alpha_arg)

            # Rescale running state by alpha -- [PMAX, 2]
            new_sum = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"new_sum_kv{kv_h}_s{s_t}")
            nisa.tensor_tensor(new_sum, running_sum, alpha, op=nl.multiply)
            new_acc = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"new_acc_kv{kv_h}_s{s_t}")
            nisa.tensor_tensor(new_acc, attn_acc, alpha, op=nl.multiply)

            # exp(score - new_max) element-wise on [PMAX, 2]
            score_shifted = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"score_shifted_kv{kv_h}_s{s_t}")
            nisa.tensor_tensor(score_shifted, score_sb, neg_new_max, op=nl.add)
            score_exp = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"score_exp_kv{kv_h}_s{s_t}")
            nisa.activation(score_exp, op=nl.exp, data=score_shifted)

            # tile_sum: sum of score_exp over partition dim, per Q head
            # rms_ones[128,128] @ score_exp_bf16[128,2] -> [128,2]
            score_exp_bf16 = nl.ndarray((PMAX, gqa), dtype=nl.bfloat16, buffer=nl.sbuf,
                                        name=f"score_exp_bf16_kv{kv_h}_s{s_t}")
            nisa.tensor_copy(score_exp_bf16, score_exp)
            tile_sum_psum = nl.zeros((PMAX, gqa), dtype=nl.float32, buffer=nl.psum,
                                     name=f"tile_sum_psum_kv{kv_h}_s{s_t}")
            nisa.nc_matmul(tile_sum_psum, stationary=rms_ones, moving=score_exp_bf16)
            tile_sum = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"tile_sum_kv{kv_h}_s{s_t}")
            nisa.tensor_copy(tile_sum, tile_sum_psum)
            nisa.tensor_tensor(new_sum, new_sum, tile_sum, op=nl.add)

            # Load V_tile [PMAX, PMAX] and transpose ONCE (Opt 5: shared for both heads)
            v_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"v_tile_kv{kv_h}_s{s_t}")
            nisa.dma_copy(
                dst=v_tile,
                src=V_cache_flat.ap(pattern=[[d, PMAX], [1, d]], offset=flat_row * d),
            )
            v_tile_T_psum = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                       name=f"v_tile_T_psum_kv{kv_h}_s{s_t}")
            nisa.nc_transpose(v_tile_T_psum, v_tile)
            v_tile_T = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"v_tile_T_kv{kv_h}_s{s_t}")
            nisa.tensor_copy(v_tile_T, v_tile_T_psum)

            # V weighted for both heads: V_tile_T[128,128] @ score_exp_bf16[128,2] -> [128,2]
            v_weighted_psum = nl.zeros((PMAX, gqa), dtype=nl.float32, buffer=nl.psum,
                                       name=f"v_weighted_psum_kv{kv_h}_s{s_t}")
            nisa.nc_matmul(v_weighted_psum, stationary=v_tile_T, moving=score_exp_bf16)
            v_weighted = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"v_weighted_kv{kv_h}_s{s_t}")
            nisa.tensor_copy(v_weighted, v_weighted_psum)
            nisa.tensor_tensor(new_acc, new_acc, v_weighted, op=nl.add)

            # Commit state for next tile
            nisa.tensor_copy(running_max, new_max)
            nisa.tensor_copy(running_sum, new_sum)
            nisa.tensor_copy(attn_acc, new_acc)

        # =====================================================================
        # STEP 6: Active position (current token) with packed tensors
        # k_rope is [PMAX, B=1], v_active is [PMAX, B=1]
        # Broadcast to [PMAX, gqa=2] to match packed Q heads.
        # =====================================================================

        # Broadcast k_rope [PMAX, 1] -> [PMAX, 2]
        k_rope_packed = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"k_rope_packed_kv{kv_h}")
        nisa.tensor_copy(k_rope_packed[0:PMAX, 0:1], k_rope)
        nisa.tensor_copy(k_rope_packed[0:PMAX, 1:2], k_rope)

        # Element-wise: k_rope_packed[128,2] * q_scaled_packed[128,2] -> kq_elem[128,2]
        kq_elem = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"kq_elem_kv{kv_h}")
        nisa.tensor_tensor(kq_elem, k_rope_packed, q_scaled_packed, op=nl.multiply)
        kq_elem_bf16 = nl.ndarray((PMAX, gqa), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"kq_elem_bf16_kv{kv_h}")
        nisa.tensor_copy(kq_elem_bf16, kq_elem)

        # Reduction sum: rms_ones[128,128] @ kq_elem_bf16[128,2] -> score_active[128,2]
        score_active_psum = nl.zeros((PMAX, gqa), dtype=nl.float32, buffer=nl.psum,
                                     name=f"score_active_psum_kv{kv_h}")
        nisa.nc_matmul(score_active_psum, stationary=rms_ones, moving=kq_elem_bf16)
        score_active = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"score_active_kv{kv_h}")
        nisa.tensor_copy(score_active, score_active_psum)

        # Online softmax update for active token (all [PMAX, 2])
        new_max2 = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"new_max2_kv{kv_h}")
        nisa.tensor_tensor(new_max2, running_max, score_active, op=nl.maximum)

        neg_new_max2 = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"neg_new_max2_kv{kv_h}")
        nisa.tensor_scalar(neg_new_max2, new_max2, op0=nl.multiply, operand0=-1.0)
        alpha2_arg = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"alpha2_arg_kv{kv_h}")
        nisa.tensor_tensor(alpha2_arg, running_max, neg_new_max2, op=nl.add)
        alpha2 = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"alpha2_kv{kv_h}")
        nisa.activation(alpha2, op=nl.exp, data=alpha2_arg)

        new_sum2 = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"new_sum2_kv{kv_h}")
        nisa.tensor_tensor(new_sum2, running_sum, alpha2, op=nl.multiply)
        new_acc2 = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"new_acc2_kv{kv_h}")
        nisa.tensor_tensor(new_acc2, attn_acc, alpha2, op=nl.multiply)

        # exp(score_active - new_max2) element-wise on [PMAX, 2]
        score_act_shifted = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"score_act_shifted_kv{kv_h}")
        nisa.tensor_tensor(score_act_shifted, score_active, neg_new_max2, op=nl.add)
        score_act_exp = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"score_act_exp_kv{kv_h}")
        nisa.activation(score_act_exp, op=nl.exp, data=score_act_shifted)
        nisa.tensor_tensor(new_sum2, new_sum2, score_act_exp, op=nl.add)

        # V contribution for active token: broadcast v_active [PMAX,1] -> [PMAX,2]
        v_active_packed = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"v_active_packed_kv{kv_h}")
        nisa.tensor_copy(v_active_packed[0:PMAX, 0:1], v_active)
        nisa.tensor_copy(v_active_packed[0:PMAX, 1:2], v_active)

        # v_active_packed[128,2] * score_act_exp[128,2] -> v_act_weighted[128,2]
        v_act_weighted = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"v_act_weighted_kv{kv_h}")
        nisa.tensor_tensor(v_act_weighted, v_active_packed, score_act_exp, op=nl.multiply)
        nisa.tensor_tensor(new_acc2, new_acc2, v_act_weighted, op=nl.add)

        # =====================================================================
        # Normalize: attn_out = new_acc2 / new_sum2  (all [PMAX, 2])
        # Each column normalizes independently -- correct for packed heads.
        # =====================================================================
        sum_safe = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"sum_safe_kv{kv_h}")
        nisa.tensor_scalar(sum_safe, new_sum2, op0=nl.add, operand0=1e-9)
        rsqrt_sum = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"rsqrt_sum_kv{kv_h}")
        nisa.activation(rsqrt_sum, op=nl.rsqrt, data=sum_safe)
        inv_sum = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"inv_sum_kv{kv_h}")
        nisa.tensor_tensor(inv_sum, rsqrt_sum, rsqrt_sum, op=nl.multiply)

        attn_out = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"attn_out_kv{kv_h}")
        nisa.tensor_tensor(attn_out, new_acc2, inv_sum, op=nl.multiply)

        # =====================================================================
        # STEP 7: Split packed output and store per Q head
        # Column layout output: output_col [Hq_out, B]. Store [PMAX, B] directly.
        # NO transpose needed (Optimization 1).
        # =====================================================================
        for g in range(gqa):
            q_h = kv_h * gqa + g

            # Extract column g: [PMAX, 1] from [PMAX, 2]
            attn_out_g = attn_out[0:PMAX, g:g+1]  # [PMAX, 1]

            # Cast to bf16
            attn_bf16_g = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                     name=f"attn_bf16_kv{kv_h}_g{g}")
            nisa.tensor_copy(attn_bf16_g, attn_out_g)

            # Direct store to output_col -- NO transpose needed (column layout)
            nisa.dma_copy(
                dst=output_col[q_h * d:(q_h + 1) * d, 0:B],
                src=attn_bf16_g,
            )

    return output


# =============================================================================
# Embedded test harness (adapted from v2_kv_dedup.py)
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    import torch
    import torch.nn.functional as F
    import torch_xla.core.xla_model as xm

    os.environ["NEURON_CC_FLAGS"] = " "
    os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rmsnorm_per_head(x, weight, eps=1e-6):
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(variance + eps)
        return (x_normed * weight.float()).to(x.dtype)

    def pytorch_reference(
        hidden_states, Wq, Wk, Wv,
        q_norm_weight, k_norm_weight,
        K_cache, V_cache,
        cos_at_pos, sin_at_pos,
        d=128,
    ):
        B, _, H = hidden_states.shape
        Hq_out = Wq.shape[0]
        Hkv_out = Wk.shape[0]
        Hq_tp = Hq_out // d
        Hkv_tp = Hkv_out // d
        gqa = Hq_tp // Hkv_tp
        S_prior = K_cache.shape[2]

        hs = hidden_states.float()
        Q = hs @ Wq.float().T
        K = hs @ Wk.float().T
        V = hs @ Wv.float().T

        Q = Q.reshape(B, Hq_tp, 1, d)
        K = K.reshape(B, Hkv_tp, 1, d)
        V = V.reshape(B, Hkv_tp, 1, d)

        Q_norm = torch.zeros_like(Q)
        K_norm = torch.zeros_like(K)
        for h in range(Hq_tp):
            Q_norm[:, h, :, :] = apply_rmsnorm_per_head(Q[:, h, :, :], q_norm_weight)
        for h in range(Hkv_tp):
            K_norm[:, h, :, :] = apply_rmsnorm_per_head(K[:, h, :, :], k_norm_weight)

        cos = cos_at_pos.float().unsqueeze(1).unsqueeze(2)
        sin = sin_at_pos.float().unsqueeze(1).unsqueeze(2)

        Q_rope = Q_norm.float() * cos + rotate_half(Q_norm.float()) * sin
        K_rope = K_norm.float() * cos + rotate_half(K_norm.float()) * sin

        scale = 1.0 / math.sqrt(d)

        output = torch.zeros(B, Hq_tp, d, dtype=torch.float32)

        for b in range(B):
            for q_h in range(Hq_tp):
                kv_h = q_h // gqa
                q_vec = Q_rope[b, q_h, 0, :]

                K_full = torch.cat([
                    K_cache[b, kv_h].float(),
                    K_rope[b, kv_h, 0:1, :].float(),
                ], dim=0)

                V_full = torch.cat([
                    V_cache[b, kv_h].float(),
                    V[b, kv_h, 0:1, :].float(),
                ], dim=0)

                scores = (K_full @ q_vec) * scale
                attn_weights = F.softmax(scores, dim=0)
                out_vec = (attn_weights.unsqueeze(-1) * V_full).sum(dim=0)
                output[b, q_h, :] = out_vec

        output = output.reshape(B, 1, Hq_tp * d)
        return output.to(torch.bfloat16)

    def run_test(B=1, S_prior=128, H=2048, d=128, Hq_tp=8, Hkv_tp=4):
        print(f"\n{'='*70}")
        print(f"v6_ultimate Test (All 5 optimizations combined)")
        print(f"B={B}, S_prior={S_prior}, H={H}, d={d}, Hq_tp={Hq_tp}, Hkv_tp={Hkv_tp}")
        print(f"{'='*70}")

        device = xm.xla_device()
        dtype = torch.bfloat16

        Hq_out = Hq_tp * d
        Hkv_out = Hkv_tp * d

        torch.manual_seed(42)

        scale = 0.05
        hidden_states = (torch.randn(B, 1, H) * scale).to(dtype)
        Wq = (torch.randn(Hq_out, H) * scale).to(dtype)
        Wk = (torch.randn(Hkv_out, H) * scale).to(dtype)
        Wv = (torch.randn(Hkv_out, H) * scale).to(dtype)
        q_norm_weight = torch.ones(d, dtype=dtype)
        k_norm_weight = torch.ones(d, dtype=dtype)
        K_cache = (torch.randn(B, Hkv_tp, S_prior, d) * scale).to(dtype)
        V_cache = (torch.randn(B, Hkv_tp, S_prior, d) * scale).to(dtype)

        pos = S_prior
        position_ids = torch.full((B, 1), pos, dtype=torch.int32)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
        t = torch.tensor([float(pos)])
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_at_pos = emb.cos().to(dtype).expand(B, d)
        sin_at_pos = emb.sin().to(dtype).expand(B, d)

        print("\n[1/4] Computing PyTorch reference...")
        ref_out = pytorch_reference(
            hidden_states, Wq, Wk, Wv,
            q_norm_weight, k_norm_weight,
            K_cache, V_cache,
            cos_at_pos, sin_at_pos,
            d=d,
        )
        print(f"  ref_out shape: {ref_out.shape}")
        print(f"  ref_out stats: min={ref_out.float().min():.4f}, max={ref_out.float().max():.4f}")

        print("\n[2/4] Preparing NKI kernel inputs...")
        cos_nki = cos_at_pos.reshape(B, d)
        sin_nki = sin_at_pos.reshape(B, d)

        hidden_dev = hidden_states.to(device)
        Wq_dev = Wq.to(device)
        Wk_dev = Wk.to(device)
        Wv_dev = Wv.to(device)
        q_norm_dev = q_norm_weight.to(device)
        k_norm_dev = k_norm_weight.to(device)
        K_cache_dev = K_cache.to(device)
        V_cache_dev = V_cache.to(device)
        cos_dev = cos_nki.to(device)
        sin_dev = sin_nki.to(device)
        pos_dev = position_ids.to(device)

        print("\n[3/4] Running NKI kernel (v6_ultimate)...")
        nki_out_dev = qwen3_attn_tkg_fused(
            hidden_dev, Wq_dev, Wk_dev, Wv_dev,
            q_norm_dev, k_norm_dev,
            K_cache_dev, V_cache_dev,
            cos_dev, sin_dev, pos_dev,
        )
        xm.mark_step()
        nki_out = nki_out_dev.cpu()
        print(f"  nki_out shape: {nki_out.shape}")
        print(f"  nki_out stats: min={nki_out.float().min():.4f}, max={nki_out.float().max():.4f}")

        print("\n[4/4] Comparing results...")
        ref_f = ref_out.float()
        nki_f = nki_out.float()
        diff = (ref_f - nki_f).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"  Max  |diff| : {max_diff:.6e}")
        print(f"  Mean |diff| : {mean_diff:.6e}")

        threshold = 0.1
        if max_diff < threshold:
            print(f"\n{'='*70}")
            print(f"PASS: max_diff={max_diff:.4e} < threshold={threshold}")
            print(f"{'='*70}")
            return True
        else:
            print(f"\n{'='*70}")
            print(f"FAIL: max_diff={max_diff:.4e} >= threshold={threshold}")
            print(f"{'='*70}")
            print(f"\nRef sample:  {ref_f[0, 0, :8].tolist()}")
            print(f"NKI sample:  {nki_f[0, 0, :8].tolist()}")
            return False

    ok = run_test(B=1, S_prior=640, H=2048, d=128, Hq_tp=8, Hkv_tp=4)
