"""
v5_freedim_packing: Pack GQA Q heads into the free dimension for flash decode.

Optimization: Instead of running flash decode 8 times (once per Q head) with
[128, 1] tensors, pack the 2 Q heads per GQA group into [128, 2] and run
flash decode 4 times (once per KV head). This halves:
  - K cache tile DMA loads
  - V cache tile DMA loads + transposes
  - Score matmuls (K_tile @ q)
  - V weighted matmuls (V_tile_T @ score_exp)
  - All online-softmax elementwise ops

Free-dim packing correctness: all elementwise ops (exp, add, multiply, etc.)
and matmuls operate column-independently, so packing [128,1] into [128,2]
preserves per-head results. Reductions (max, sum) transpose to [2,128] and
reduce over axis=1, giving [2,1] per-head scalars correctly.

Builds on v2_kv_dedup (K/V deduplication across GQA groups).
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

    v5: Packs GQA Q heads (gqa=2) into free dimension [PMAX, 2] for flash decode.
    All flash decode matmuls, DMA loads, and elementwise ops run on packed tensors.
    """
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

    # Output buffer in HBM
    output = nl.ndarray((B, 1, Hq_out), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    output_2d = output.reshape((B, Hq_out))

    # Scalar staging buffer for flash-decode max/sum broadcast [gqa,1] -> [PMAX,gqa]
    scalar_hbm_packed = nl.ndarray((gqa, 1), dtype=nl.float32, buffer=nl.private_hbm)

    # -------------------------------------------------------------------------
    # Load norm weights: [d=128] -> [PMAX, 1] in SBUF (float32)
    # -------------------------------------------------------------------------
    qnw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_bf16")
    nisa.dma_copy(dst=qnw_bf16, src=q_norm_weight.reshape((PMAX, 1)))
    qnw_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qnw_sb")
    nisa.tensor_copy(qnw_sb, qnw_bf16)

    knw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_bf16")
    nisa.dma_copy(dst=knw_bf16, src=k_norm_weight.reshape((PMAX, 1)))
    knw_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="knw_sb")
    nisa.tensor_copy(knw_sb, knw_bf16)

    # -------------------------------------------------------------------------
    # All-ones [PMAX, PMAX] constant for partition-dimension reduction
    # Used in: RMSNorm sum-of-squares, tile_sum, score_active dot product
    # -------------------------------------------------------------------------
    rms_ones = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)

    # -------------------------------------------------------------------------
    # Load RoPE cos/sin: [B, d] -> [PMAX, B] in SBUF (float32)
    # -------------------------------------------------------------------------
    cos_row = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="cos_row")
    sin_row = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="sin_row")
    nisa.dma_copy(dst=cos_row, src=cos.reshape((B, PMAX)))
    nisa.dma_copy(dst=sin_row, src=sin.reshape((B, PMAX)))
    cos_T_psum = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.psum, name="cos_T_psum")
    sin_T_psum = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.psum, name="sin_T_psum")
    nisa.nc_transpose(cos_T_psum, cos_row)
    nisa.nc_transpose(sin_T_psum, sin_row)
    cos_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf, name="cos_bf16")
    sin_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf, name="sin_bf16")
    nisa.tensor_copy(cos_bf16, cos_T_psum)
    nisa.tensor_copy(sin_bf16, sin_T_psum)
    cos_f32 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="cos_f32")
    sin_f32 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="sin_f32")
    nisa.tensor_copy(cos_f32, cos_bf16)
    nisa.tensor_copy(sin_f32, sin_bf16)

    # -------------------------------------------------------------------------
    # Reshape hidden and flatten KV cache
    # -------------------------------------------------------------------------
    hidden_2d = hidden_states.reshape((B, H))  # [B, H]
    K_cache_flat = K_cache.reshape((B * Hkv_tp * S_prior, d))
    V_cache_flat = V_cache.reshape((B * Hkv_tp * S_prior, d))

    half_d = d // 2  # 64

    # =========================================================================
    # MAIN LOOP: Outer=KV head (4 iterations)
    #   - Compute K projection, K RMSNorm, K RoPE once per KV head
    #   - Compute V projection once per KV head
    #   - Compute both Q heads, pack into [PMAX, gqa=2]
    #   - Run packed flash decode with [PMAX, 2] tensors
    #   - Split and store output per Q head
    # =========================================================================
    for kv_h in nl.affine_range(Hkv_tp):
        # =====================================================================
        # K PROJECTION + RMSNorm + RoPE (once per KV head, unchanged from v2)
        # =====================================================================

        # ---- K matmul for kv_h ----
        k_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                          name=f"k_psum_kv{kv_h}")
        for h_t in nl.affine_range(num_h_tiles):
            h_off = h_t * PMAX
            h_row_k = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"h_row_k_kv{kv_h}_t{h_t}")
            nisa.dma_copy(
                dst=h_row_k,
                src=hidden_2d.ap(pattern=[[H, B], [1, PMAX]], offset=h_off),
            )
            h_tile_psum_k = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.psum,
                                       name=f"h_tile_psum_k_kv{kv_h}_t{h_t}")
            nisa.nc_transpose(h_tile_psum_k, h_row_k)
            h_tile_k = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"h_tile_k_kv{kv_h}_t{h_t}")
            nisa.tensor_copy(h_tile_k, h_tile_psum_k)

            wk_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"wk_tile_kv{kv_h}_t{h_t}")
            nisa.dma_copy(
                dst=wk_tile,
                src=Wk.ap(pattern=[[H, PMAX], [1, PMAX]], offset=kv_h * PMAX * H + h_off),
            )
            nisa.nc_matmul(k_psum, stationary=wk_tile, moving=h_tile_k)

        k_vec = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                           name=f"k_vec_kv{kv_h}")
        nisa.tensor_copy(k_vec, k_psum)

        # ---- K RMSNorm ----
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

        # ---- K RoPE ----
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
        k_rope = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"k_rope_kv{kv_h}")
        nisa.tensor_tensor(k_rope, k_cos, k_sin_part, op=nl.add)

        k_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                            name=f"k_bf16_kv{kv_h}")
        nisa.tensor_copy(k_bf16, k_rope)

        # =====================================================================
        # V PROJECTION (once per KV head, unchanged from v2)
        # =====================================================================
        v_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                          name=f"v_psum_kv{kv_h}")
        for h_t in range(num_h_tiles):
            h_off = h_t * PMAX
            h_row_v = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"h_row_v_kv{kv_h}_t{h_t}")
            nisa.dma_copy(
                dst=h_row_v,
                src=hidden_2d.ap(pattern=[[H, B], [1, PMAX]], offset=h_off),
            )
            h_tile_psum_v = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.psum,
                                       name=f"h_tile_psum_v_kv{kv_h}_t{h_t}")
            nisa.nc_transpose(h_tile_psum_v, h_row_v)
            h_tile_v = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"h_tile_v_kv{kv_h}_t{h_t}")
            nisa.tensor_copy(h_tile_v, h_tile_psum_v)

            wv_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"wv_tile_kv{kv_h}_t{h_t}")
            nisa.dma_copy(
                dst=wv_tile,
                src=Wv.ap(pattern=[[H, PMAX], [1, PMAX]], offset=kv_h * PMAX * H + h_off),
            )
            nisa.nc_matmul(v_psum, stationary=wv_tile, moving=h_tile_v)

        v_active = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"v_active_kv{kv_h}")
        nisa.tensor_copy(v_active, v_psum)

        # =====================================================================
        # STEP 1: Compute both Q heads and pack into [PMAX, gqa=2]
        # =====================================================================
        # We compute Q matmul + RMSNorm + RoPE + scale for each of the 2 Q heads,
        # storing results into lists, then pack into [PMAX, 2] tensors.

        q_bf16_list = []
        q_scaled_list = []
        for g in range(gqa):
            q_h = kv_h * gqa + g

            # ---- Q matmul for head q_h ----
            q_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                              name=f"q_psum_h{q_h}")
            for h_t in nl.affine_range(num_h_tiles):
                h_off = h_t * PMAX
                h_row_q = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                     name=f"h_row_q_h{q_h}_t{h_t}")
                nisa.dma_copy(
                    dst=h_row_q,
                    src=hidden_2d.ap(pattern=[[H, B], [1, PMAX]], offset=h_off),
                )
                h_tile_psum_q = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.psum,
                                           name=f"h_tile_psum_q_h{q_h}_t{h_t}")
                nisa.nc_transpose(h_tile_psum_q, h_row_q)
                h_tile_q = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                      name=f"h_tile_q_h{q_h}_t{h_t}")
                nisa.tensor_copy(h_tile_q, h_tile_psum_q)

                wq_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                     name=f"wq_tile_h{q_h}_t{h_t}")
                nisa.dma_copy(
                    dst=wq_tile,
                    src=Wq.ap(pattern=[[H, PMAX], [1, PMAX]], offset=q_h * PMAX * H + h_off),
                )
                nisa.nc_matmul(q_psum, stationary=wq_tile, moving=h_tile_q)

            q_vec = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"q_vec_h{q_h}")
            nisa.tensor_copy(q_vec, q_psum)

            # ---- Q RMSNorm ----
            q_sq = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"q_sq_h{q_h}")
            nisa.tensor_tensor(q_sq, q_vec, q_vec, op=nl.multiply)
            q_sq_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"q_sq_bf16_h{q_h}")
            nisa.tensor_copy(q_sq_bf16, q_sq)
            q_sum_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                                  name=f"q_sum_psum_h{q_h}")
            nisa.nc_matmul(q_sum_psum, stationary=rms_ones, moving=q_sq_bf16)
            q_sum_sb = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"q_sum_sb_h{q_h}")
            nisa.tensor_copy(q_sum_sb, q_sum_psum)
            q_mean_sq = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"q_mean_sq_h{q_h}")
            nisa.tensor_scalar(q_mean_sq, q_sum_sb, op0=nl.multiply, operand0=1.0/d)
            q_var_eps = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"q_var_eps_h{q_h}")
            nisa.tensor_scalar(q_var_eps, q_mean_sq, op0=nl.add, operand0=EPS)
            q_rms_inv = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"q_rms_inv_h{q_h}")
            nisa.activation(q_rms_inv, op=nl.rsqrt, data=q_var_eps)
            q_normed = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"q_normed_h{q_h}")
            nisa.tensor_tensor(q_normed, q_vec, q_rms_inv, op=nl.multiply)
            q_normed2 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"q_normed2_h{q_h}")
            nisa.tensor_tensor(q_normed2, q_normed, qnw_sb, op=nl.multiply)

            # ---- Q RoPE ----
            rot_q = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"rot_q_h{q_h}")
            neg_q_upper = nl.ndarray((half_d, B), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"neg_q_upper_h{q_h}")
            nisa.tensor_scalar(neg_q_upper, q_normed2[half_d:d, 0:B], op0=nl.multiply, operand0=-1.0)
            nisa.tensor_copy(rot_q[0:half_d, 0:B], neg_q_upper)
            nisa.tensor_copy(rot_q[half_d:d, 0:B], q_normed2[0:half_d, 0:B])

            q_cos = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"q_cos_h{q_h}")
            q_sin_part = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"q_sin_part_h{q_h}")
            nisa.tensor_tensor(q_cos, q_normed2, cos_f32, op=nl.multiply)
            nisa.tensor_tensor(q_sin_part, rot_q, sin_f32, op=nl.multiply)
            q_rope = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"q_rope_h{q_h}")
            nisa.tensor_tensor(q_rope, q_cos, q_sin_part, op=nl.add)

            # Scale q by 1/sqrt(d) and cast to bf16
            q_scaled = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"q_scaled_h{q_h}")
            nisa.tensor_scalar(q_scaled, q_rope, op0=nl.multiply, operand0=INV_SQRT_D)
            q_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"q_bf16_h{q_h}")
            nisa.tensor_copy(q_bf16, q_scaled)

            q_bf16_list.append(q_bf16)
            q_scaled_list.append(q_scaled)

        # =====================================================================
        # Pack Q results into [PMAX, gqa=2] tensors
        # q_bf16_list[0], q_bf16_list[1] are each [PMAX, 1] (since B=1)
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
        # STEP 2: Packed flash decode with [PMAX, gqa=2] tensors
        # All state vectors are [PMAX, 2] — one column per Q head.
        # =====================================================================

        # Initialize packed flash decode state
        attn_acc = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"attn_acc_kv{kv_h}")
        nisa.memset(attn_acc, value=0.0)
        running_max = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"running_max_kv{kv_h}")
        nisa.memset(running_max, value=-1e9)
        running_sum = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"running_sum_kv{kv_h}")
        nisa.memset(running_sum, value=0.0)

        # -----------------------------------------------------------------
        # KV cache tile loop: each tile loads K[128,128] and V[128,128] ONCE,
        # then scores/weights BOTH Q heads via packed [128,2] matmuls.
        # -----------------------------------------------------------------
        for s_t in nl.affine_range(num_s_tiles):
            s_off = s_t * PMAX
            flat_row = kv_h * S_prior + s_off

            # Load K_tile [PMAX, PMAX] from K_cache_flat (once per tile)
            k_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"k_tile_kv{kv_h}_s{s_t}")
            nisa.dma_copy(
                dst=k_tile,
                src=K_cache_flat.ap(pattern=[[d, PMAX], [1, d]], offset=flat_row * d),
            )

            # Score both Q heads: K_tile[128,128] @ q_packed[128,2] -> [128,2]
            score_psum = nl.zeros((PMAX, gqa), dtype=nl.float32, buffer=nl.psum,
                                  name=f"score_psum_kv{kv_h}_s{s_t}")
            nisa.nc_matmul(score_psum, stationary=k_tile, moving=q_packed)
            score_sb = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"score_sb_kv{kv_h}_s{s_t}")
            nisa.tensor_copy(score_sb, score_psum)

            # --- Online softmax: tile_max via transpose + reduce ---
            # Transpose [PMAX, 2] -> [2, PMAX] to get partition values into free dim
            score_T_psum = nl.ndarray((gqa, PMAX), dtype=nl.float32, buffer=nl.psum,
                                      name=f"score_T_psum_kv{kv_h}_s{s_t}")
            nisa.nc_transpose(score_T_psum, score_sb)
            score_T_sb = nl.ndarray((gqa, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"score_T_sb_kv{kv_h}_s{s_t}")
            nisa.tensor_copy(score_T_sb, score_T_psum)

            # Reduce max over axis=1: [2, PMAX] -> [2, 1]
            tile_max_scalar = nl.ndarray((gqa, 1), dtype=nl.float32, buffer=nl.sbuf,
                                         name=f"tile_max_scalar_kv{kv_h}_s{s_t}")
            nisa.tensor_reduce(dst=tile_max_scalar, op=nl.maximum, data=score_T_sb, axis=1)

            # Broadcast [2,1] -> [PMAX,2] via HBM staging with stride-0 pattern
            nisa.dma_copy(dst=scalar_hbm_packed, src=tile_max_scalar)
            tile_max = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"tile_max_kv{kv_h}_s{s_t}")
            nisa.dma_copy(dst=tile_max,
                          src=scalar_hbm_packed.ap(pattern=[[0, PMAX], [1, gqa]], offset=0))

            # new_max = max(running_max, tile_max)
            new_max = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"new_max_kv{kv_h}_s{s_t}")
            nisa.tensor_tensor(new_max, running_max, tile_max, op=nl.maximum)

            # alpha = exp(running_max - new_max)
            neg_new_max = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"neg_new_max_kv{kv_h}_s{s_t}")
            nisa.tensor_scalar(neg_new_max, new_max, op0=nl.multiply, operand0=-1.0)
            alpha_arg = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"alpha_arg_kv{kv_h}_s{s_t}")
            nisa.tensor_tensor(alpha_arg, running_max, neg_new_max, op=nl.add)
            alpha = nl.ndarray((PMAX, gqa), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"alpha_kv{kv_h}_s{s_t}")
            nisa.activation(alpha, op=nl.exp, data=alpha_arg)

            # Rescale running state by alpha
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
            # Each column sums independently (correct for packed heads)
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

            # Load V_tile [PMAX, PMAX] and transpose ONCE (shared for both Q heads)
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
        # STEP 3: Active position (current token) with packed tensors
        # k_rope is [PMAX, 1] (B=1), v_active is [PMAX, 1] (B=1)
        # Need to broadcast to [PMAX, 2] to match packed Q heads.
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
        # Each column sums independently (dot product per Q head)
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
        # Each column normalizes independently — correct for packed heads.
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
        # STEP 4: Split packed output and store per Q head
        # attn_out is [PMAX, 2] — extract each column [PMAX, 1], cast to bf16,
        # transpose to [1, PMAX], and DMA store to the correct head slot in output.
        # =====================================================================
        for g in range(gqa):
            q_h = kv_h * gqa + g

            # Extract column g: [PMAX, 1] slice from [PMAX, 2]
            attn_out_g = attn_out[0:PMAX, g:g+1]  # [PMAX, 1]

            # Cast to bf16
            attn_bf16_g = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                     name=f"attn_bf16_h{q_h}")
            nisa.tensor_copy(attn_bf16_g, attn_out_g)

            # Transpose [PMAX, 1] -> [1, PMAX] for DMA store
            attn_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                     name=f"attn_T_psum_h{q_h}")
            nisa.nc_transpose(attn_T_psum, attn_bf16_g)
            attn_T_sb = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"attn_T_sb_h{q_h}")
            nisa.tensor_copy(attn_T_sb, attn_T_psum)

            # Store to output: head q_h occupies columns [q_h*d, (q_h+1)*d)
            nisa.dma_copy(
                dst=output_2d.ap(pattern=[[Hq_out, B], [1, PMAX]], offset=q_h * d),
                src=attn_T_sb,
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
        print(f"v5_freedim_packing Test")
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

        print("\n[3/4] Running NKI kernel...")
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
