"""
Fused NKI kernel: QKV projection + per-head RMSNorm + RoPE + Flash Decode Attention
for Qwen3-30B-A3B token generation on Trainium2 (trn2, NeuronCore-v3).

Target: B=1 batch, one NeuronCore in a TP=4 setup.

Per TP rank shapes:
  hidden_states : [B, 1, H]         bf16   H=2048
  Wq            : [Hq_tp*d, H]      bf16   [1024, 2048]
  Wk            : [Hkv_tp*d, H]     bf16   [512,  2048]
  Wv            : [Hkv_tp*d, H]     bf16   [512,  2048]
  q_norm_weight : [d]               bf16   [128]
  k_norm_weight : [d]               bf16   [128]
  K_cache       : [B, Hkv_tp, S_prior, d] bf16
  V_cache       : [B, Hkv_tp, S_prior, d] bf16
  cos, sin      : [B, d]            bf16   (pre-indexed at position_ids)
  position_ids  : [B, 1]            int32  (kept for signature compatibility)
  Returns       : [B, 1, Hq_tp*d]   bf16

Key design:
  pmax = 128 (NeuronCore-v3 partition size = head_dim d)
  H is tiled in chunks of pmax=128  (num_h_tiles = H/128 = 16)
  For QKV matmul:
    nc_matmul: stationary=[PMAX, K], moving=[K, B]
    stationary = W_tile [PMAX, PMAX], moving = h_tile [PMAX, B]
    result [PMAX, B] in PSUM
  Flash decode: tile KV cache S_prior in chunks of PMAX=128
  GQA ratio gqa = Hq_tp / Hkv_tp = 8/4 = 2
  B=1 assumed throughout for simplicity (B dimension kept for generality).

  All tensors allocated inside loops use explicit unique name= parameters
  to avoid "Tensor name already in use" errors during NKI tracing.
"""

import math
import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128
EPS = 1e-6
INV_SQRT_D = float(1.0 / math.sqrt(128.0))  # 1/sqrt(128) ≈ 0.08838


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

    Layout convention:
      - Partition dim (axis 0) = head_dim d = 128 for all head-level tensors
      - Free dim    (axis 1) = B=1 or S_tile (for KV scoring)

    All tensors inside loops use explicit name= parameters to prevent
    duplicate tensor name errors during NKI tracing.
    """
    B = hidden_states.shape[0]       # 1
    H = hidden_states.shape[2]       # 2048
    Hq_out = Wq.shape[0]            # 1024 = Hq_tp * d
    Hkv_out = Wk.shape[0]           # 512  = Hkv_tp * d
    d = PMAX                        # 128
    Hq_tp = Hq_out // d             # 8
    Hkv_tp = Hkv_out // d           # 4
    gqa = Hq_tp // Hkv_tp           # 2
    S_prior = K_cache.shape[2]       # e.g. 128 or 640
    num_h_tiles = H // PMAX          # 16
    num_s_tiles = S_prior // PMAX   # tiles of KV cache positions

    # Output buffer in HBM
    output = nl.ndarray((B, 1, Hq_out), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    output_2d = output.reshape((B, Hq_out))

    # Scalar staging buffer in private HBM for the flash-decode max broadcast:
    # tile_max_scalar [B, 1] f32 is stored here, then re-loaded into [PMAX, B] SBUF
    # using a DMA pattern with outer stride=0 to replicate the single row to all partitions.
    # nl.private_hbm is allocatable inside the kernel body (unlike nl.shared_hbm).
    scalar_hbm = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.private_hbm)

    # -------------------------------------------------------------------------
    # Load norm weights: [d=128] -> [PMAX, 1] in SBUF (float32)
    # -------------------------------------------------------------------------
    qnw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_bf16")
    nisa.dma_copy(dst=qnw_bf16, src=q_norm_weight.reshape((PMAX, 1)))
    qnw_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qnw_sb")
    nisa.tensor_copy(qnw_sb, qnw_bf16)   # cast bf16->f32

    knw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_bf16")
    nisa.dma_copy(dst=knw_bf16, src=k_norm_weight.reshape((PMAX, 1)))
    knw_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="knw_sb")
    nisa.tensor_copy(knw_sb, knw_bf16)

    # -------------------------------------------------------------------------
    # All-ones [PMAX, PMAX] constant for partition-dimension reduction in RMSNorm.
    # nc_matmul(ones[PMAX,PMAX], x[PMAX,B]) -> result[PMAX,B] where each
    # partition row = sum_over_all_partitions(x), enabling broadcast-multiply.
    # -------------------------------------------------------------------------
    rms_ones = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)


    # -------------------------------------------------------------------------
    # Load RoPE cos/sin: [B, d] -> [PMAX, B] in SBUF (float32)
    # cos shape [B, d]: load as [B, PMAX] then transpose to [PMAX, B]
    # -------------------------------------------------------------------------
    cos_row = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="cos_row")
    sin_row = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="sin_row")
    nisa.dma_copy(dst=cos_row, src=cos.reshape((B, PMAX)))
    nisa.dma_copy(dst=sin_row, src=sin.reshape((B, PMAX)))
    # nc_transpose [B, PMAX] -> [PMAX, B] (writes to PSUM via Tensor Engine)
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

    # Flatten KV cache for .ap() indexing
    # K_cache: [B, Hkv_tp, S_prior, d] -> [B*Hkv_tp*S_prior, d]
    K_cache_flat = K_cache.reshape((B * Hkv_tp * S_prior, d))
    V_cache_flat = V_cache.reshape((B * Hkv_tp * S_prior, d))

    # =========================================================================
    # Per-head processing loop (plain Python range — no affine_range)
    # All tensors use explicit name= with q_h index to avoid name collisions.
    # For each Q head:
    #   1. QKV matmul (hidden @ Wq/Wk/Wv)
    #   2. RMSNorm on Q and K
    #   3. RoPE on Q and K
    #   4. Flash decode attention
    #   5. Store result to output
    # =========================================================================
    for q_h in range(Hq_tp):
        kv_h = q_h // gqa

        # ---- Q matmul for head q_h ----
        # Q: hidden [B, H] @ Wq[q_h*d:(q_h+1)*d, :].T -> [d, B] = [PMAX, B]
        q_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                          name=f"q_psum_h{q_h}")
        for h_t in range(num_h_tiles):
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

        # Copy Q from PSUM to SBUF (float32)
        q_vec = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                           name=f"q_vec_h{q_h}")
        nisa.tensor_copy(q_vec, q_psum)

        # ---- K matmul for kv_h head ----
        k_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                          name=f"k_psum_h{q_h}")
        for h_t in range(num_h_tiles):
            h_off = h_t * PMAX
            h_row_k = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"h_row_k_h{q_h}_t{h_t}")
            nisa.dma_copy(
                dst=h_row_k,
                src=hidden_2d.ap(pattern=[[H, B], [1, PMAX]], offset=h_off),
            )
            h_tile_psum_k = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.psum,
                                       name=f"h_tile_psum_k_h{q_h}_t{h_t}")
            nisa.nc_transpose(h_tile_psum_k, h_row_k)
            h_tile_k = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"h_tile_k_h{q_h}_t{h_t}")
            nisa.tensor_copy(h_tile_k, h_tile_psum_k)

            wk_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"wk_tile_h{q_h}_t{h_t}")
            nisa.dma_copy(
                dst=wk_tile,
                src=Wk.ap(pattern=[[H, PMAX], [1, PMAX]], offset=kv_h * PMAX * H + h_off),
            )
            nisa.nc_matmul(k_psum, stationary=wk_tile, moving=h_tile_k)

        k_vec = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                           name=f"k_vec_h{q_h}")
        nisa.tensor_copy(k_vec, k_psum)

        # ---- V matmul for kv_h head ----
        v_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                          name=f"v_psum_h{q_h}")
        for h_t in range(num_h_tiles):
            h_off = h_t * PMAX
            h_row_v = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"h_row_v_h{q_h}_t{h_t}")
            nisa.dma_copy(
                dst=h_row_v,
                src=hidden_2d.ap(pattern=[[H, B], [1, PMAX]], offset=h_off),
            )
            h_tile_psum_v = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.psum,
                                       name=f"h_tile_psum_v_h{q_h}_t{h_t}")
            nisa.nc_transpose(h_tile_psum_v, h_row_v)
            h_tile_v = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"h_tile_v_h{q_h}_t{h_t}")
            nisa.tensor_copy(h_tile_v, h_tile_psum_v)

            wv_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"wv_tile_h{q_h}_t{h_t}")
            nisa.dma_copy(
                dst=wv_tile,
                src=Wv.ap(pattern=[[H, PMAX], [1, PMAX]], offset=kv_h * PMAX * H + h_off),
            )
            nisa.nc_matmul(v_psum, stationary=wv_tile, moving=h_tile_v)

        v_active = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"v_active_h{q_h}")
        nisa.tensor_copy(v_active, v_psum)

        # ---- Q RMSNorm ----
        # Compute sum of squares over partition dim using all-ones matmul.
        # q_vec: [PMAX, B].  q_sq: [PMAX, B] = q_vec^2.
        # nc_matmul(ones[PMAX,PMAX], q_sq[PMAX,B]) -> q_sum_psum[PMAX,B]
        # where every partition row = sum_over_all_partitions(q_sq).
        q_sq = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                          name=f"q_sq_h{q_h}")
        nisa.tensor_tensor(q_sq, q_vec, q_vec, op=nl.multiply)
        q_sq_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"q_sq_bf16_h{q_h}")
        nisa.tensor_copy(q_sq_bf16, q_sq)
        q_sum_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                              name=f"q_sum_psum_h{q_h}")
        nisa.nc_matmul(q_sum_psum, stationary=rms_ones, moving=q_sq_bf16)
        # Copy PSUM -> SBUF then compute rsqrt(sum/d + eps) — shape [PMAX, B]
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
        # q_normed = q_vec * rms_inv * norm_weight  (all [PMAX, B] — shapes match)
        q_normed = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"q_normed_h{q_h}")
        nisa.tensor_tensor(q_normed, q_vec, q_rms_inv, op=nl.multiply)
        q_normed2 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"q_normed2_h{q_h}")
        nisa.tensor_tensor(q_normed2, q_normed, qnw_sb, op=nl.multiply)

        # ---- K RMSNorm ----
        k_sq = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                          name=f"k_sq_h{q_h}")
        nisa.tensor_tensor(k_sq, k_vec, k_vec, op=nl.multiply)
        k_sq_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"k_sq_bf16_h{q_h}")
        nisa.tensor_copy(k_sq_bf16, k_sq)
        k_sum_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                              name=f"k_sum_psum_h{q_h}")
        nisa.nc_matmul(k_sum_psum, stationary=rms_ones, moving=k_sq_bf16)
        k_sum_sb = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"k_sum_sb_h{q_h}")
        nisa.tensor_copy(k_sum_sb, k_sum_psum)
        k_mean_sq = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"k_mean_sq_h{q_h}")
        nisa.tensor_scalar(k_mean_sq, k_sum_sb, op0=nl.multiply, operand0=1.0/d)
        k_var_eps = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"k_var_eps_h{q_h}")
        nisa.tensor_scalar(k_var_eps, k_mean_sq, op0=nl.add, operand0=EPS)
        k_rms_inv = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"k_rms_inv_h{q_h}")
        nisa.activation(k_rms_inv, op=nl.rsqrt, data=k_var_eps)
        k_normed = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"k_normed_h{q_h}")
        nisa.tensor_tensor(k_normed, k_vec, k_rms_inv, op=nl.multiply)
        k_normed2 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"k_normed2_h{q_h}")
        nisa.tensor_tensor(k_normed2, k_normed, knw_sb, op=nl.multiply)

        # ---- RoPE: rotate_half and apply cos/sin ----
        half_d = d // 2   # 64

        # rotate_half(q_normed2): [PMAX, B]
        #   result[0:64, :] = -q_normed2[64:128, :]
        #   result[64:128, :] = q_normed2[0:64, :]
        rot_q = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                           name=f"rot_q_h{q_h}")
        neg_q_upper = nl.ndarray((half_d, B), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"neg_q_upper_h{q_h}")
        nisa.tensor_scalar(neg_q_upper, q_normed2[half_d:d, 0:B], op0=nl.multiply, operand0=-1.0)
        nisa.tensor_copy(rot_q[0:half_d, 0:B], neg_q_upper)
        nisa.tensor_copy(rot_q[half_d:d, 0:B], q_normed2[0:half_d, 0:B])

        # rotate_half(k_normed2)
        rot_k = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                           name=f"rot_k_h{q_h}")
        neg_k_upper = nl.ndarray((half_d, B), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"neg_k_upper_h{q_h}")
        nisa.tensor_scalar(neg_k_upper, k_normed2[half_d:d, 0:B], op0=nl.multiply, operand0=-1.0)
        nisa.tensor_copy(rot_k[0:half_d, 0:B], neg_k_upper)
        nisa.tensor_copy(rot_k[half_d:d, 0:B], k_normed2[0:half_d, 0:B])

        # q_rope = q_normed2 * cos + rot_q * sin
        q_cos = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                           name=f"q_cos_h{q_h}")
        q_sin_part = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"q_sin_part_h{q_h}")
        nisa.tensor_tensor(q_cos, q_normed2, cos_f32, op=nl.multiply)
        nisa.tensor_tensor(q_sin_part, rot_q, sin_f32, op=nl.multiply)
        q_rope = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"q_rope_h{q_h}")
        nisa.tensor_tensor(q_rope, q_cos, q_sin_part, op=nl.add)

        # k_rope = k_normed2 * cos + rot_k * sin
        k_cos = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                           name=f"k_cos_h{q_h}")
        k_sin_part = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"k_sin_part_h{q_h}")
        nisa.tensor_tensor(k_cos, k_normed2, cos_f32, op=nl.multiply)
        nisa.tensor_tensor(k_sin_part, rot_k, sin_f32, op=nl.multiply)
        k_rope = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"k_rope_h{q_h}")
        nisa.tensor_tensor(k_rope, k_cos, k_sin_part, op=nl.add)

        # Scale q by 1/sqrt(d) and cast to bf16
        q_scaled = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"q_scaled_h{q_h}")
        nisa.tensor_scalar(q_scaled, q_rope, op0=nl.multiply, operand0=INV_SQRT_D)
        q_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                            name=f"q_bf16_h{q_h}")
        nisa.tensor_copy(q_bf16, q_scaled)
        k_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf,
                            name=f"k_bf16_h{q_h}")
        nisa.tensor_copy(k_bf16, k_rope)

        # ---- Flash Decode: online softmax over S_prior + 1 positions ----
        # Output accumulator [PMAX, B] float32
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
        # K_cache_flat row offset: (0 * Hkv_tp + kv_h) * S_prior = kv_h * S_prior
        for s_t in range(num_s_tiles):
            s_off = s_t * PMAX
            flat_row = kv_h * S_prior + s_off   # for B=1

            # Load K_tile [PMAX, d] from K_cache_flat[flat_row:flat_row+PMAX, :]
            k_tile = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"k_tile_h{q_h}_s{s_t}")
            nisa.dma_copy(
                dst=k_tile,
                src=K_cache_flat.ap(pattern=[[d, PMAX], [1, d]], offset=flat_row * d),
            )

            # score_tile = K_tile @ q_bf16  [PMAX_s, B]
            # nc_matmul: stationary=k_tile [PMAX, d], moving=q_bf16 [d, B]
            score_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum,
                                  name=f"score_psum_h{q_h}_s{s_t}")
            nisa.nc_matmul(score_psum, stationary=k_tile, moving=q_bf16)
            score_sb = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"score_sb_h{q_h}_s{s_t}")
            nisa.tensor_copy(score_sb, score_psum)

            # Online softmax: compute tile_max [PMAX, B] by reducing over partition dim.
            # Step 1: transpose [PMAX,B] -> [B,PMAX], reduce max over free dim -> [B,1]
            score_T_psum = nl.ndarray((B, PMAX), dtype=nl.float32, buffer=nl.psum,
                                      name=f"score_T_psum_h{q_h}_s{s_t}")
            nisa.nc_transpose(score_T_psum, score_sb)
            score_T_sb = nl.ndarray((B, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"score_T_sb_h{q_h}_s{s_t}")
            nisa.tensor_copy(score_T_sb, score_T_psum)
            tile_max_scalar = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf,
                                         name=f"tile_max_scalar_h{q_h}_s{s_t}")
            nisa.tensor_reduce(dst=tile_max_scalar, op=nl.maximum, data=score_T_sb, axis=1)
            # Step 2: broadcast [B,1] -> [PMAX,B] via HBM staging with stride-0 DMA reload
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

            # Rescale: new_sum = running_sum * alpha, new_acc = attn_acc * alpha
            new_sum = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"new_sum_h{q_h}_s{s_t}")
            nisa.tensor_tensor(new_sum, running_sum, alpha, op=nl.multiply)
            new_acc = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"new_acc_h{q_h}_s{s_t}")
            nisa.tensor_tensor(new_acc, attn_acc, alpha, op=nl.multiply)

            # exp(score - new_max): score_shifted = score + (-new_max)
            score_shifted = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"score_shifted_h{q_h}_s{s_t}")
            nisa.tensor_tensor(score_shifted, score_sb, neg_new_max, op=nl.add)
            score_exp = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"score_exp_h{q_h}_s{s_t}")
            nisa.activation(score_exp, op=nl.exp, data=score_shifted)

            # tile_sum = sum(score_exp over partition dim) broadcast to [PMAX, B]
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
            # V_tile.T @ score_exp: stationary=[d, S_tile], moving=[S_tile, B]
            # nc_transpose v_tile [PMAX_s, d] -> [d, PMAX_s] in PSUM
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
        # score_active = dot(k_rope, q_scaled) — sum over partition dim, broadcast to [PMAX, B]
        # k_rope and q_scaled are [PMAX, B], elementwise multiply then sum over partition
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

        # exp(score_active - new_max2): score_act_shifted = score_active + neg_new_max2
        score_act_shifted = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"score_act_shifted_h{q_h}")
        nisa.tensor_tensor(score_act_shifted, score_active, neg_new_max2, op=nl.add)
        score_act_exp = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"score_act_exp_h{q_h}")
        nisa.activation(score_act_exp, op=nl.exp, data=score_act_shifted)
        nisa.tensor_tensor(new_sum2, new_sum2, score_act_exp, op=nl.add)

        # V contribution for active token: v_active [PMAX, B] * score_act_exp [PMAX, B]
        v_act_weighted = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"v_act_weighted_h{q_h}")
        nisa.tensor_tensor(v_act_weighted, v_active, score_act_exp, op=nl.multiply)
        nisa.tensor_tensor(new_acc2, new_acc2, v_act_weighted, op=nl.add)

        # Normalize: attn_out = new_acc2 / new_sum2
        # Use rsqrt(x)^2 = 1/x trick for scalar divisor — all [PMAX, B]
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

        # Cast to bf16 and store to output_2d[0:B, q_h*d:(q_h+1)*d]
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
