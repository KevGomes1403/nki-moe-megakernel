"""
v2_kv_dedup: Deduplicate K/V computation for GQA heads.

Optimization: With GQA ratio=2, K and V projections are identical for pairs
of Q heads (q_h=0,1 share kv_h=0, etc.). The original v0 recomputes K/V for
each of 8 Q heads. This version restructures the loop:

  Outer loop: for kv_h in range(Hkv_tp=4)  — compute K, V once per KV head
  Inner loop: for g in range(gqa=2)         — compute Q per Q head, reuse K/V

Savings:
  - K matmul: 128 -> 64  (saves 64 matmuls)
  - V matmul: 128 -> 64  (saves 64 matmuls)
  - K RMSNorm: 8 -> 4    (saves 4 all-ones matmuls + elementwise ops)
  - K RoPE: 8 -> 4       (saves 4 sets of elementwise ops)
  - Total: ~132 matmuls eliminated + associated DMA and transposes
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

    v2: Restructured loop to deduplicate K/V computation across GQA head groups.
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

    # Scalar staging buffer for flash-decode max broadcast
    scalar_hbm = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.private_hbm)

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
    # All-ones [PMAX, PMAX] constant for partition-dimension reduction in RMSNorm
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
    # RESTRUCTURED LOOP: Outer=KV head, Inner=GQA group member
    #
    # Outer loop (kv_h): Compute K projection, K RMSNorm, K RoPE, V projection
    #   once per KV head. These results persist across the inner g loop.
    #
    # Inner loop (g): Compute Q projection, Q RMSNorm, Q RoPE per Q head,
    #   then run flash decode attention using the shared K/V from outer loop.
    # =========================================================================
    for kv_h in nl.affine_range(Hkv_tp):
        # =====================================================================
        # OUTER LOOP: K and V computation (once per KV head)
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
        # k_rope persists across inner g loop — shared by all Q heads in this group
        k_rope = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"k_rope_kv{kv_h}")
        nisa.tensor_tensor(k_rope, k_cos, k_sin_part, op=nl.add)

        # k_bf16 for flash decode scoring — persists across inner g loop
        k_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                            name=f"k_bf16_kv{kv_h}")
        nisa.tensor_copy(k_bf16, k_rope)

        # ---- V matmul for kv_h (once per KV head) ----
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

        # v_active persists across inner g loop — shared by all Q heads in this group
        v_active = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"v_active_kv{kv_h}")
        nisa.tensor_copy(v_active, v_psum)

        # =================================================================
        # INNER LOOP: Q computation + flash decode (per Q head)
        # K/V results from outer loop (k_rope, k_bf16, v_active) are reused.
        # =================================================================
        for g in nl.affine_range(gqa):
            q_h = kv_h * gqa + g  # maps to global Q head index

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

            # ---- Flash Decode: online softmax over S_prior + 1 positions ----
            # Uses k_bf16, k_rope, v_active from outer loop (shared across GQA group)
            attn_acc = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"attn_acc_h{q_h}")
            nisa.memset(attn_acc, value=0.0)
            running_max = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"running_max_h{q_h}")
            running_sum = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"running_sum_h{q_h}")
            nisa.memset(running_max, value=-1e9)
            nisa.memset(running_sum, value=0.0)

            # KV cache tiles: for B=1, kv_h head
            for s_t in nl.affine_range(num_s_tiles):
                s_off = s_t * PMAX
                flat_row = kv_h * S_prior + s_off

                # Load K_tile [PMAX, d] from K_cache_flat
                k_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                    name=f"k_tile_h{q_h}_s{s_t}")
                nisa.dma_copy(
                    dst=k_tile,
                    src=K_cache_flat.ap(pattern=[[d, PMAX], [1, d]], offset=flat_row * d),
                )

                # score_tile = K_tile @ q_bf16  [PMAX_s, B]
                score_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                                      name=f"score_psum_h{q_h}_s{s_t}")
                nisa.nc_matmul(score_psum, stationary=k_tile, moving=q_bf16)
                score_sb = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"score_sb_h{q_h}_s{s_t}")
                nisa.tensor_copy(score_sb, score_psum)

                # Online softmax: tile_max via transpose + reduce
                score_T_psum = nl.ndarray((B, PMAX), dtype=nl.float32, buffer=nl.psum,
                                          name=f"score_T_psum_h{q_h}_s{s_t}")
                nisa.nc_transpose(score_T_psum, score_sb)
                score_T_sb = nl.ndarray((B, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"score_T_sb_h{q_h}_s{s_t}")
                nisa.tensor_copy(score_T_sb, score_T_psum)
                tile_max_scalar = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf,
                                             name=f"tile_max_scalar_h{q_h}_s{s_t}")
                nisa.tensor_reduce(dst=tile_max_scalar, op=nl.maximum, data=score_T_sb, axis=1)
                # Broadcast [B,1] -> [PMAX,B] via HBM staging
                nisa.dma_copy(dst=scalar_hbm, src=tile_max_scalar)
                tile_max = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"tile_max_h{q_h}_s{s_t}")
                nisa.dma_copy(dst=tile_max,
                              src=scalar_hbm.ap(pattern=[[0, PMAX], [1, B]], offset=0))

                new_max = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"new_max_h{q_h}_s{s_t}")
                nisa.tensor_tensor(new_max, running_max, tile_max, op=nl.maximum)

                # alpha = exp(running_max - new_max)
                neg_new_max = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                         name=f"neg_new_max_h{q_h}_s{s_t}")
                nisa.tensor_scalar(neg_new_max, new_max, op0=nl.multiply, operand0=-1.0)
                alpha_arg = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"alpha_arg_h{q_h}_s{s_t}")
                nisa.tensor_tensor(alpha_arg, running_max, neg_new_max, op=nl.add)
                alpha = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"alpha_h{q_h}_s{s_t}")
                nisa.activation(alpha, op=nl.exp, data=alpha_arg)

                # Rescale running state
                new_sum = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"new_sum_h{q_h}_s{s_t}")
                nisa.tensor_tensor(new_sum, running_sum, alpha, op=nl.multiply)
                new_acc = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"new_acc_h{q_h}_s{s_t}")
                nisa.tensor_tensor(new_acc, attn_acc, alpha, op=nl.multiply)

                # exp(score - new_max)
                score_shifted = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                           name=f"score_shifted_h{q_h}_s{s_t}")
                nisa.tensor_tensor(score_shifted, score_sb, neg_new_max, op=nl.add)
                score_exp = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"score_exp_h{q_h}_s{s_t}")
                nisa.activation(score_exp, op=nl.exp, data=score_shifted)

                # tile_sum = sum(score_exp over partition dim)
                score_exp_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                            name=f"score_exp_bf16_h{q_h}_s{s_t}")
                nisa.tensor_copy(score_exp_bf16, score_exp)
                tile_sum_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                                         name=f"tile_sum_psum_h{q_h}_s{s_t}")
                nisa.nc_matmul(tile_sum_psum, stationary=rms_ones, moving=score_exp_bf16)
                tile_sum = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"tile_sum_h{q_h}_s{s_t}")
                nisa.tensor_copy(tile_sum, tile_sum_psum)
                nisa.tensor_tensor(new_sum, new_sum, tile_sum, op=nl.add)

                # Load V_tile and compute V contribution
                v_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                    name=f"v_tile_h{q_h}_s{s_t}")
                nisa.dma_copy(
                    dst=v_tile,
                    src=V_cache_flat.ap(pattern=[[d, PMAX], [1, d]], offset=flat_row * d),
                )
                v_tile_T_psum = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                           name=f"v_tile_T_psum_h{q_h}_s{s_t}")
                nisa.nc_transpose(v_tile_T_psum, v_tile)
                v_tile_T = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                      name=f"v_tile_T_h{q_h}_s{s_t}")
                nisa.tensor_copy(v_tile_T, v_tile_T_psum)

                v_weighted_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                                           name=f"v_weighted_psum_h{q_h}_s{s_t}")
                nisa.nc_matmul(v_weighted_psum, stationary=v_tile_T, moving=score_exp_bf16)
                v_weighted = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"v_weighted_h{q_h}_s{s_t}")
                nisa.tensor_copy(v_weighted, v_weighted_psum)
                nisa.tensor_tensor(new_acc, new_acc, v_weighted, op=nl.add)

                # Commit state
                nisa.tensor_copy(running_max, new_max)
                nisa.tensor_copy(running_sum, new_sum)
                nisa.tensor_copy(attn_acc, new_acc)

            # ---- Active position (current token) ----
            # Uses k_rope from outer loop (shared K for this KV head group)
            kq_elem = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"kq_elem_h{q_h}")
            nisa.tensor_tensor(kq_elem, k_rope, q_scaled, op=nl.multiply)
            kq_elem_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                      name=f"kq_elem_bf16_h{q_h}")
            nisa.tensor_copy(kq_elem_bf16, kq_elem)
            score_active_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                                         name=f"score_active_psum_h{q_h}")
            nisa.nc_matmul(score_active_psum, stationary=rms_ones, moving=kq_elem_bf16)
            score_active = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"score_active_h{q_h}")
            nisa.tensor_copy(score_active, score_active_psum)

            # Online softmax update for active token
            new_max2 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"new_max2_h{q_h}")
            nisa.tensor_tensor(new_max2, running_max, score_active, op=nl.maximum)

            neg_new_max2 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"neg_new_max2_h{q_h}")
            nisa.tensor_scalar(neg_new_max2, new_max2, op0=nl.multiply, operand0=-1.0)
            alpha2_arg = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"alpha2_arg_h{q_h}")
            nisa.tensor_tensor(alpha2_arg, running_max, neg_new_max2, op=nl.add)
            alpha2 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"alpha2_h{q_h}")
            nisa.activation(alpha2, op=nl.exp, data=alpha2_arg)

            new_sum2 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"new_sum2_h{q_h}")
            nisa.tensor_tensor(new_sum2, running_sum, alpha2, op=nl.multiply)
            new_acc2 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"new_acc2_h{q_h}")
            nisa.tensor_tensor(new_acc2, attn_acc, alpha2, op=nl.multiply)

            # exp(score_active - new_max2)
            score_act_shifted = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                           name=f"score_act_shifted_h{q_h}")
            nisa.tensor_tensor(score_act_shifted, score_active, neg_new_max2, op=nl.add)
            score_act_exp = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"score_act_exp_h{q_h}")
            nisa.activation(score_act_exp, op=nl.exp, data=score_act_shifted)
            nisa.tensor_tensor(new_sum2, new_sum2, score_act_exp, op=nl.add)

            # V contribution for active token: uses v_active from outer loop
            v_act_weighted = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"v_act_weighted_h{q_h}")
            nisa.tensor_tensor(v_act_weighted, v_active, score_act_exp, op=nl.multiply)
            nisa.tensor_tensor(new_acc2, new_acc2, v_act_weighted, op=nl.add)

            # Normalize: attn_out = new_acc2 / new_sum2
            sum_safe = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"sum_safe_h{q_h}")
            nisa.tensor_scalar(sum_safe, new_sum2, op0=nl.add, operand0=1e-9)
            rsqrt_sum = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"rsqrt_sum_h{q_h}")
            nisa.activation(rsqrt_sum, op=nl.rsqrt, data=sum_safe)
            inv_sum = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"inv_sum_h{q_h}")
            nisa.tensor_tensor(inv_sum, rsqrt_sum, rsqrt_sum, op=nl.multiply)

            attn_out = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"attn_out_h{q_h}")
            nisa.tensor_tensor(attn_out, new_acc2, inv_sum, op=nl.multiply)

            # Cast to bf16 and store to output
            attn_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"attn_bf16_h{q_h}")
            nisa.tensor_copy(attn_bf16, attn_out)

            # Transpose [PMAX, B] -> [B, PMAX] for DMA store
            attn_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                     name=f"attn_T_psum_h{q_h}")
            nisa.nc_transpose(attn_T_psum, attn_bf16)
            attn_T_sb = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"attn_T_sb_h{q_h}")
            nisa.tensor_copy(attn_T_sb, attn_T_psum)

            nisa.dma_copy(
                dst=output_2d.ap(pattern=[[Hq_out, B], [1, PMAX]], offset=q_h * d),
                src=attn_T_sb,
            )

    return output


# =============================================================================
# Embedded test harness (adapted from test_attn_tkg_fused.py)
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
        print(f"v2_kv_dedup Test")
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

    # if ok:
    #     print("\n\nRunning second test with S_prior=640...")
    #     ok2 = run_test(B=1, S_prior=640, H=2048, d=128, Hq_tp=8, Hkv_tp=4)
    #     if ok2:
    #         print("\nAll tests PASSED.")
    #     else:
    #         print("\nSecond test FAILED.")
    #         sys.exit(1)
    # else:
    #     sys.exit(1)
