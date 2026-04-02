"""
v11a: Single-Pass Online Flash Decode

Changes from v10e:
  - Replaces the 4-loop 2-pass flash decode structure with a single
    sequential loop implementing the online softmax (flash attention) algorithm.
  - The four loops replaced:
      1. K-cache hoisting loop (k_cache_tiles)
      2. V-cache hoisting loop (v_cache_tiles)
      3. Pass1 loop (scores + mask + global_max, saved_scores[])
      4. Pass2 loop (exp(score - max) + V-weighted accumulation)
  - New: nl.sequential_range(NUM_S_TILES) loop carrying state:
      m       [GQA, 1]   running max per head
      v_acc   [PMAX, GQA] running weighted V sum
      sum_acc [PMAX, GQA] running softmax denominator
  - Benefits: eliminates saved_scores SBUF cost (5×4KB=20KB), eliminates
    pre-hoisted K/V tiles (2×10×32KB=640KB), loads K and V on-demand.

All other optimizations from v10e are preserved unchanged:
  Plan A static shape constants, Q-proj one-head-at-a-time, O-proj
  head-outer/h_blk-inner, all tp_broadcast patterns, hidden tile hoisting,
  Wo contiguous DMA, v10e position-id threshold masking.
"""

import math
import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128
F_MAX = 512
EPS = 1e-6
INV_SQRT_D = float(1.0 / math.sqrt(128.0))

import os
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "./output"

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

@nki.jit
def qwen3_attn_tkg_fused_oproj_v11a(
    hidden_states,   # [B, 1, H]        bf16  (B=1)
    Wq,              # [Hq_tp*d, H]     bf16  [1024, 2048]
    Wk,              # [Hkv_tp*d, H]    bf16  [128, 2048]  (Hkv_tp=1)
    Wv,              # [Hkv_tp*d, H]    bf16  [128, 2048]
    Wo,              # [Hq_tp*d, H]     bf16  [1024, 2048]  transposed o_proj weight
    q_norm_weight,   # [d]              bf16  [128]
    k_norm_weight,   # [d]              bf16  [128]
    K_cache,         # [B, 1, S_prior, d] bf16
    V_cache,         # [B, 1, S_prior, d] bf16
    cos,             # [B, d]           bf16
    sin,             # [B, d]           bf16
    position_ids,    # [B, 1]           int32 — decoding step (number of valid cache tokens)
):
    """
    Fused QKV + RMSNorm + RoPE + single-pass flash decode + output projection.
    Attention mask is generated on-chip from position_ids threshold.
    Returns (output, k_rope_out, v_out):
      - output:     [B, 1, H_out] bf16, where H_out = H = 2048
      - k_rope_out: [d, B] = [128, 1] bf16 — new token's K after RMSNorm+RoPE
      - v_out:      [d, B] = [128, 1] bf16 — new token's V (no RoPE)

    Single-pass online flash decode: carries running max m, weighted-V sum v_acc,
    and softmax denominator sum_acc through a single sequential loop over K-cache
    tiles. Eliminates the pre-hoisting of all K/V tiles and the saved_scores buffer.
    """
    # --- Dimensions ---
    B = hidden_states.shape[0]      # 1
    H = hidden_states.shape[2]      # 2048
    Hq_out = Wq.shape[0]            # 1024  = Hq_tp * d
    d = PMAX                        # 128
    Hq_tp = Hq_out // d             # 8
    Hkv_tp = 1                      # per-rank KV heads
    GQA = Hq_tp // Hkv_tp          # 8
    S_prior = K_cache.shape[2]
    num_h_tiles = H // PMAX         # 16
    num_s_tiles = S_prior // PMAX
    half_d = d // 2                 # 64

    H_wo = Wo.shape[1]              # 2048
    num_h_blocks = H_wo // F_MAX   # 4

    # =========================================================================
    # Plan A — Static shape constants
    # =========================================================================
    assert S_prior % PMAX == 0, f"S_prior={S_prior} must be a multiple of {PMAX}"
    NUM_S_TILES  = S_prior // PMAX   # trace-time constant; 5 for S_prior=640
    NUM_H_TILES  = 16   # H=2048 / PMAX=128
    HQ_TP_CONST  = 8    # Hq_tp fixed for this shape
    NUM_H_BLOCKS = 4    # H_wo=2048 / F_MAX=512
    assert H == NUM_H_TILES * PMAX, f"H={H} must be {NUM_H_TILES*PMAX}"
    assert Hq_tp == HQ_TP_CONST, f"Hq_tp={Hq_tp} must be {HQ_TP_CONST}"

    # =========================================================================
    # COLUMN LAYOUT RESHAPES
    # =========================================================================
    output = nl.ndarray((B, 1, H_wo), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    output_2d = output.reshape((1, H_wo))

    k_rope_out = nl.ndarray((B, d), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    v_out = nl.ndarray((B, d), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    hidden_col = hidden_states.reshape((H, B))
    cos_col = cos.reshape((PMAX, B))
    sin_col = sin.reshape((PMAX, B))

    # =========================================================================
    # LOAD CONSTANTS
    # =========================================================================
    qnw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_bf16")
    nisa.dma_copy(dst=qnw_bf16, src=q_norm_weight.reshape((PMAX, 1)))
    qnw_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qnw_sb")
    nisa.tensor_copy(qnw_sb, qnw_bf16)

    knw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_bf16")
    nisa.dma_copy(dst=knw_bf16, src=k_norm_weight.reshape((PMAX, 1)))
    knw_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="knw_sb")
    nisa.tensor_copy(knw_sb, knw_bf16)

    cos_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="cos_bf16")
    sin_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="sin_bf16")
    nisa.dma_copy(dst=cos_bf16, src=cos_col)
    nisa.dma_copy(dst=sin_bf16, src=sin_col)
    cos_f32 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="cos_f32")
    sin_f32 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="sin_f32")
    nisa.tensor_copy(cos_f32, cos_bf16)
    nisa.tensor_copy(sin_f32, sin_bf16)

    rms_ones = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)

    # =========================================================================
    # HIDDEN TILE HOISTING
    # =========================================================================
    h_all = nl.ndarray((PMAX, num_h_tiles), dtype=nl.bfloat16, buffer=nl.sbuf, name="h_all")
    nisa.dma_copy(
        dst=h_all,
        src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
    )

    # =========================================================================
    # WO WEIGHT HOISTING — nkilib-style contiguous DMA
    # =========================================================================
    Wo_reshaped = Wo.reshape((Hq_tp, d, H_wo))

    wo_sbuf = []
    for head in nl.affine_range(HQ_TP_CONST):
        wo_tile = nl.ndarray((PMAX, H_wo), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"wo_tile_h{head}")
        nisa.dma_copy(
            dst=wo_tile,
            src=Wo_reshaped.ap(
                pattern=[[H_wo, PMAX], [1, H_wo]],
                offset=head * PMAX * H_wo,
            ),
        )
        wo_sbuf.append(wo_tile)

    # =========================================================================
    # K PROJECTION (Hkv_tp=1)
    # =========================================================================
    wk_full = nl.ndarray((PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="wk_full")
    nisa.dma_copy(dst=wk_full, src=Wk)
    k_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="k_psum")
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(k_psum, stationary=wk_full[0:PMAX, h_t*PMAX:(h_t+1)*PMAX], moving=h_all[0:PMAX, h_t:h_t+1])

    k_vec = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_vec")
    nisa.tensor_copy(k_vec, k_psum)

    # K RMSNorm
    k_sq = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_sq")
    nisa.tensor_tensor(k_sq, k_vec, k_vec, op=nl.multiply)
    k_sq_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf, name="k_sq_bf16")
    nisa.tensor_copy(k_sq_bf16, k_sq)
    k_sum_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="k_sum_psum")
    nisa.nc_matmul(k_sum_psum, stationary=rms_ones, moving=k_sq_bf16)
    k_sum_sb = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_sum_sb")
    nisa.tensor_copy(k_sum_sb, k_sum_psum)
    k_mean_sq = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_mean_sq")
    nisa.tensor_scalar(k_mean_sq, k_sum_sb, op0=nl.multiply, operand0=1.0/d, op1=nl.add, operand1=EPS)
    k_rms_inv = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_rms_inv")
    nisa.activation(k_rms_inv, op=nl.rsqrt, data=k_mean_sq)
    k_normed = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_normed")
    nisa.tensor_tensor(k_normed, k_vec, k_rms_inv, op=nl.multiply)
    k_normed2 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_normed2")
    nisa.tensor_tensor(k_normed2, k_normed, knw_sb, op=nl.multiply)

    # K RoPE
    rot_k = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="rot_k")
    neg_k_upper = nl.ndarray((half_d, B), dtype=nl.float32, buffer=nl.sbuf, name="neg_k_upper")
    nisa.tensor_scalar(neg_k_upper, k_normed2[half_d:d, 0:B], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(rot_k[0:half_d, 0:B], neg_k_upper)
    nisa.tensor_copy(rot_k[half_d:d, 0:B], k_normed2[0:half_d, 0:B])
    k_cos = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_cos")
    k_sin_part = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_sin_part")
    nisa.tensor_tensor(k_cos, k_normed2, cos_f32, op=nl.multiply)
    nisa.tensor_tensor(k_sin_part, rot_k, sin_f32, op=nl.multiply)
    k_rope = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_rope")
    nisa.tensor_tensor(k_rope, k_cos, k_sin_part, op=nl.add)

    # Store k_rope to HBM
    k_rope_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf, name="k_rope_bf16")
    nisa.tensor_copy(k_rope_bf16, k_rope)
    k_rope_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name="k_rope_T_psum")
    nisa.nc_transpose(k_rope_T_psum, k_rope_bf16)
    k_rope_T_sb = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="k_rope_T_sb")
    nisa.tensor_copy(k_rope_T_sb, k_rope_T_psum)
    nisa.dma_copy(dst=k_rope_out, src=k_rope_T_sb)

    # =========================================================================
    # V PROJECTION (Hkv_tp=1)
    # =========================================================================
    wv_full = nl.ndarray((PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="wv_full")
    nisa.dma_copy(dst=wv_full, src=Wv)
    v_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="v_psum")
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(v_psum, stationary=wv_full[0:PMAX, h_t*PMAX:(h_t+1)*PMAX], moving=h_all[0:PMAX, h_t:h_t+1])

    v_active = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="v_active")
    nisa.tensor_copy(v_active, v_psum)

    # Store v_active to HBM
    v_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf, name="v_bf16")
    nisa.tensor_copy(v_bf16, v_active)
    v_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name="v_T_psum")
    nisa.nc_transpose(v_T_psum, v_bf16)
    v_T_sb = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="v_T_sb")
    nisa.tensor_copy(v_T_sb, v_T_psum)
    nisa.dma_copy(dst=v_out, src=v_T_sb)

    # =========================================================================
    # Q PROJECTIONS — one head at a time
    # =========================================================================
    q_packed_f32 = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_packed_f32")
    for q_h in nl.affine_range(HQ_TP_CONST):
        wq_head = nl.ndarray((PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"wq_head_{q_h}")
        nisa.dma_copy(dst=wq_head, src=Wq[q_h * PMAX:(q_h + 1) * PMAX, :])
        q_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name=f"q_psum_{q_h}")
        for h_t in nl.affine_range(NUM_H_TILES):
            nisa.nc_matmul(
                q_psum,
                stationary=wq_head[0:PMAX, h_t * PMAX:(h_t + 1) * PMAX],
                moving=h_all[0:PMAX, h_t:h_t + 1],
            )
        nisa.tensor_copy(q_packed_f32[0:PMAX, q_h:q_h + 1], q_psum)

    # =========================================================================
    # PACKED Q RMSNORM on [PMAX, GQA=8]
    # =========================================================================
    q_sq = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_sq")
    nisa.tensor_tensor(q_sq, q_packed_f32, q_packed_f32, op=nl.multiply)
    q_sq_bf16 = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name="q_sq_bf16")
    nisa.tensor_copy(q_sq_bf16, q_sq)
    q_sum_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="q_sum_psum")
    nisa.nc_matmul(q_sum_psum, stationary=rms_ones, moving=q_sq_bf16)
    q_sum_sb = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_sum_sb")
    nisa.tensor_copy(q_sum_sb, q_sum_psum)
    q_mean_sq = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_mean_sq")
    nisa.tensor_scalar(q_mean_sq, q_sum_sb, op0=nl.multiply, operand0=1.0/d, op1=nl.add, operand1=EPS)
    q_rms_inv = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_rms_inv")
    nisa.activation(q_rms_inv, op=nl.rsqrt, data=q_mean_sq)
    q_normed = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_normed")
    nisa.tensor_tensor(q_normed, q_packed_f32, q_rms_inv, op=nl.multiply)

    # tp_broadcast: qnw_sb[PMAX, 1] → qnw_gqa[PMAX, GQA]
    qnw_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="qnw_gqa_psum_T")
    nisa.nc_transpose(qnw_gqa_psum_T, qnw_sb.ap([[1, PMAX], [0, GQA]], offset=0))
    qnw_gqa_sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="qnw_gqa_sbuf_T")
    nisa.tensor_copy(qnw_gqa_sbuf_T, qnw_gqa_psum_T)
    qnw_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="qnw_gqa_psum")
    nisa.nc_transpose(qnw_gqa_psum, qnw_gqa_sbuf_T)
    qnw_gqa = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="qnw_gqa")
    nisa.tensor_copy(qnw_gqa, qnw_gqa_psum)

    q_normed2 = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_normed2")
    nisa.tensor_tensor(q_normed2, q_normed, qnw_gqa, op=nl.multiply)

    # =========================================================================
    # PACKED Q ROPE on [PMAX, GQA=8]
    # =========================================================================
    cos_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="cos_gqa_psum_T")
    nisa.nc_transpose(cos_gqa_psum_T, cos_f32.ap([[1, PMAX], [0, GQA]], offset=0))
    cos_gqa_sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="cos_gqa_sbuf_T")
    nisa.tensor_copy(cos_gqa_sbuf_T, cos_gqa_psum_T)
    cos_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="cos_gqa_psum")
    nisa.nc_transpose(cos_gqa_psum, cos_gqa_sbuf_T)
    cos_gqa = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="cos_gqa")
    nisa.tensor_copy(cos_gqa, cos_gqa_psum)

    sin_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="sin_gqa_psum_T")
    nisa.nc_transpose(sin_gqa_psum_T, sin_f32.ap([[1, PMAX], [0, GQA]], offset=0))
    sin_gqa_sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="sin_gqa_sbuf_T")
    nisa.tensor_copy(sin_gqa_sbuf_T, sin_gqa_psum_T)
    sin_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="sin_gqa_psum")
    nisa.nc_transpose(sin_gqa_psum, sin_gqa_sbuf_T)
    sin_gqa = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="sin_gqa")
    nisa.tensor_copy(sin_gqa, sin_gqa_psum)

    rot_q = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="rot_q")
    neg_q_upper = nl.ndarray((half_d, GQA), dtype=nl.float32, buffer=nl.sbuf, name="neg_q_upper")
    nisa.tensor_scalar(neg_q_upper, q_normed2[half_d:d, 0:GQA], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(rot_q[0:half_d, 0:GQA], neg_q_upper)
    nisa.tensor_copy(rot_q[half_d:d, 0:GQA], q_normed2[0:half_d, 0:GQA])

    q_cos = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_cos")
    q_sin_part = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_sin_part")
    nisa.tensor_tensor(q_cos, q_normed2, cos_gqa, op=nl.multiply)
    nisa.tensor_tensor(q_sin_part, rot_q, sin_gqa, op=nl.multiply)
    q_rope = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_rope")
    nisa.tensor_tensor(q_rope, q_cos, q_sin_part, op=nl.add)

    # Scale by 1/sqrt(d) and cast to bf16
    q_bf16 = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name="q_bf16")
    nisa.tensor_scalar(q_bf16, q_rope, op0=nl.multiply, operand0=INV_SQRT_D)

    # =========================================================================
    # ACTIVE POSITION SCORE
    # (kept identical to v10e: k_rope [PMAX,1] dot q_scaled [PMAX,GQA])
    # =========================================================================
    K_cache_2d = K_cache.reshape((S_prior, d))
    V_cache_2d = V_cache.reshape((S_prior, d))

    # tp_broadcast: k_rope[PMAX, 1] → k_rope_packed[PMAX, GQA]
    k_rope_packed_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="k_rope_packed_psum_T")
    nisa.nc_transpose(k_rope_packed_psum_T, k_rope.ap([[1, PMAX], [0, GQA]], offset=0))
    k_rope_packed_sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="k_rope_packed_sbuf_T")
    nisa.tensor_copy(k_rope_packed_sbuf_T, k_rope_packed_psum_T)
    k_rope_packed_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="k_rope_packed_psum")
    nisa.nc_transpose(k_rope_packed_psum, k_rope_packed_sbuf_T)
    k_rope_packed = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="k_rope_packed")
    nisa.tensor_copy(k_rope_packed, k_rope_packed_psum)

    kq_elem = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="kq_elem")
    nisa.tensor_tensor(kq_elem, k_rope_packed, q_bf16, op=nl.multiply)
    kq_elem_bf16 = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name="kq_elem_bf16")
    nisa.tensor_copy(kq_elem_bf16, kq_elem)
    score_active_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="score_active_psum")
    nisa.nc_matmul(score_active_psum, stationary=rms_ones, moving=kq_elem_bf16)
    score_active = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="score_active")
    nisa.tensor_copy(score_active, score_active_psum)

    # =========================================================================
    # v10e: Load position scalar for masking
    # =========================================================================
    position_ids_2d = position_ids.reshape((B, 1))
    pos_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name="pos_sb")
    nisa.dma_copy(dst=pos_sb, src=position_ids_2d[0:1, 0:1])
    pos_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf, name="pos_f32")
    nisa.tensor_copy(pos_f32, pos_sb)

    # =========================================================================
    # v10e: Build partition index [PMAX, 1] = [0.0, 1.0, ..., 127.0]
    # =========================================================================
    par_index_f32 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="par_index_f32")
    nisa.iota(par_index_f32, pattern=[[1, 1]], offset=0, channel_multiplier=1)

    # =========================================================================
    # SINGLE-PASS ONLINE FLASH DECODE (replaces 4-loop 2-pass structure)
    # Implements flash attention online softmax algorithm:
    #   For each tile: update running max, rescale accumulators, accumulate.
    # Loop is sequential because it carries state (m, v_acc, sum_acc).
    # =========================================================================

    # Running state: m (global max per head), v_acc (weighted V sum), sum_acc (softmax denom)
    m = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="m")
    nisa.memset(m, value=-1e9)
    v_acc = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="v_acc")
    nisa.memset(v_acc, value=0.0)
    sum_acc = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="sum_acc")
    nisa.memset(sum_acc, value=0.0)

    for s_t in nl.sequential_range(NUM_S_TILES):
        tile_start = s_t * PMAX  # compile-time constant

        # --- Load K tile and transpose ---
        k_raw = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"k_raw_{s_t}")
        nisa.dma_copy(dst=k_raw, src=K_cache_2d[s_t * PMAX:(s_t + 1) * PMAX, :])
        k_ct_psum = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name=f"k_ct_psum_{s_t}")
        nisa.nc_transpose(k_ct_psum, k_raw)
        k_ct = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"k_ct_{s_t}")
        nisa.tensor_copy(k_ct, k_ct_psum)

        # --- Compute score [PMAX, GQA] ---
        score_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"score_psum_{s_t}")
        nisa.nc_matmul(score_psum, stationary=k_ct, moving=q_bf16)
        score_sb = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score_sb_{s_t}")
        nisa.tensor_copy(score_sb, score_psum)

        # --- Compute v10e mask inline ---
        # neg_threshold = tile_start - pos  (scalar [1,1])
        neg_threshold = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"neg_threshold_{s_t}")
        nisa.tensor_scalar(neg_threshold, pos_f32,
                           op0=nl.multiply, operand0=-1.0,
                           op1=nl.add, operand1=float(tile_start))

        # Broadcast neg_threshold [1,1] → [PMAX,1]
        neg_thresh_psum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum, name=f"neg_thresh_psum_{s_t}")
        nisa.nc_transpose(neg_thresh_psum, neg_threshold.ap([[1, 1], [0, PMAX]], offset=0))
        neg_thresh_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"neg_thresh_sb_{s_t}")
        nisa.tensor_copy(neg_thresh_sb, neg_thresh_psum)

        # delta[p] = par_index_f32[p] + neg_thresh_sb[p] = p + (tile_start - pos)
        delta = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"delta_{s_t}")
        nisa.tensor_tensor(delta, par_index_f32, neg_thresh_sb, op=nl.add)

        relu_delta = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"relu_delta_{s_t}")
        nisa.activation(relu_delta, op=nl.relu, data=delta)

        clamped = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"clamped_{s_t}")
        nisa.tensor_scalar(clamped, relu_delta, op0=nl.minimum, operand0=1.0)

        mask_tile_f32 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"mask_tile_f32_{s_t}")
        nisa.tensor_scalar(mask_tile_f32, clamped, op0=nl.multiply, operand0=-1e9)

        # Broadcast mask [PMAX, 1] → [PMAX, GQA] using double-nc_transpose
        mask_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name=f"mask_gqa_psum_T_{s_t}")
        nisa.nc_transpose(mask_gqa_psum_T, mask_tile_f32.ap([[1, PMAX], [0, GQA]], offset=0))
        mask_gqa_sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name=f"mask_gqa_sbuf_T_{s_t}")
        nisa.tensor_copy(mask_gqa_sbuf_T, mask_gqa_psum_T)
        mask_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"mask_gqa_psum_{s_t}")
        nisa.nc_transpose(mask_gqa_psum, mask_gqa_sbuf_T)
        mask_gqa = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"mask_gqa_{s_t}")
        nisa.tensor_copy(mask_gqa, mask_gqa_psum)

        # Apply mask to scores
        score_masked = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score_masked_{s_t}")
        nisa.tensor_tensor(score_masked, score_sb, mask_gqa, op=nl.add)

        # --- Per-tile max: [PMAX, GQA] → transpose → [GQA, PMAX] → reduce → [GQA, 1] ---
        score_T_psum = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name=f"score_T_psum_{s_t}")
        nisa.nc_transpose(score_T_psum, score_masked)
        score_T_sb = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name=f"score_T_sb_{s_t}")
        nisa.tensor_copy(score_T_sb, score_T_psum)
        tile_max = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"tile_max_{s_t}")
        nisa.tensor_reduce(dst=tile_max, op=nl.maximum, data=score_T_sb, axis=1)

        # --- Online softmax rescaling ---
        # m_new = max(m, tile_max)
        m_new = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"m_new_{s_t}")
        nisa.tensor_tensor(m_new, m, tile_max, op=nl.maximum)

        # rescale_factor = exp(m - m_new)
        neg_m_new = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"neg_m_new_{s_t}")
        nisa.tensor_scalar(neg_m_new, m_new, op0=nl.multiply, operand0=-1.0)
        m_diff = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"m_diff_{s_t}")
        nisa.tensor_tensor(m_diff, m, neg_m_new, op=nl.add)
        rescale_g1 = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"rescale_g1_{s_t}")
        nisa.activation(rescale_g1, op=nl.exp, data=m_diff)

        # Broadcast rescale [GQA, 1] → [PMAX, GQA]
        rescale_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"rescale_psum_{s_t}")
        nisa.nc_transpose(rescale_psum, rescale_g1.ap([[1, GQA], [0, PMAX]], offset=0))
        rescale = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"rescale_{s_t}")
        nisa.tensor_copy(rescale, rescale_psum)

        # Apply rescaling to v_acc and sum_acc
        nisa.tensor_tensor(v_acc, v_acc, rescale, op=nl.multiply)
        nisa.tensor_tensor(sum_acc, sum_acc, rescale, op=nl.multiply)

        # Update running max: m = m_new
        nisa.tensor_copy(m, m_new)

        # --- Load V tile (column-transposed layout via ap()) ---
        v_ct = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"v_ct_{s_t}")
        nisa.dma_copy(
            dst=v_ct,
            src=V_cache_2d.ap(pattern=[[d, PMAX], [1, d]], offset=s_t * PMAX * d),
        )

        # exp_scores = exp(score_masked - m_new_broadcast)
        # Broadcast m_new [GQA, 1] → [PMAX, GQA]
        m_new_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"m_new_psum_{s_t}")
        nisa.nc_transpose(m_new_psum, m_new.ap([[1, GQA], [0, PMAX]], offset=0))
        m_new_broad = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"m_new_broad_{s_t}")
        nisa.tensor_copy(m_new_broad, m_new_psum)

        neg_m_new_broad = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"neg_m_new_broad_{s_t}")
        nisa.tensor_scalar(neg_m_new_broad, m_new_broad, op0=nl.multiply, operand0=-1.0)
        score_shifted = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score_shifted_{s_t}")
        nisa.tensor_tensor(score_shifted, score_masked, neg_m_new_broad, op=nl.add)

        exp_scores = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"exp_scores_{s_t}")
        nisa.activation(exp_scores, op=nl.exp, data=score_shifted)
        exp_scores_bf16 = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"exp_scores_bf16_{s_t}")
        nisa.tensor_copy(exp_scores_bf16, exp_scores)

        # sum_acc += sum(exp_scores) via rms_ones matmul
        tile_sum_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"tile_sum_psum_{s_t}")
        nisa.nc_matmul(tile_sum_psum, stationary=rms_ones, moving=exp_scores_bf16)
        tile_sum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"tile_sum_{s_t}")
        nisa.tensor_copy(tile_sum, tile_sum_psum)
        nisa.tensor_tensor(sum_acc, sum_acc, tile_sum, op=nl.add)

        # v_acc += V_tile @ exp_scores
        v_weighted_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"v_weighted_psum_{s_t}")
        nisa.nc_matmul(v_weighted_psum, stationary=v_ct, moving=exp_scores_bf16)
        v_weighted = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"v_weighted_{s_t}")
        nisa.tensor_copy(v_weighted, v_weighted_psum)
        nisa.tensor_tensor(v_acc, v_acc, v_weighted, op=nl.add)

    # =========================================================================
    # ACTIVE POSITION CONTRIBUTION (after single-pass loop)
    # m holds the running max from all K-cache tiles.
    # Incorporate active position into online softmax.
    # =========================================================================

    # Reduce score_active [PMAX, GQA] → [GQA, 1] max
    score_act_T_psum = nl.zeros((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="score_act_T_psum")
    nisa.nc_transpose(score_act_T_psum, score_active)
    score_act_T_sb = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="score_act_T_sb")
    nisa.tensor_copy(score_act_T_sb, score_act_T_psum)
    score_active_g1 = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="score_active_g1")
    nisa.tensor_reduce(dst=score_active_g1, op=nl.maximum, data=score_act_T_sb, axis=1)

    # m_new_act = max(m, score_active_max)
    m_new_act = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="m_new_act")
    nisa.tensor_tensor(m_new_act, m, score_active_g1, op=nl.maximum)

    # rescale_act = exp(m - m_new_act)
    neg_m_new_act = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="neg_m_new_act")
    nisa.tensor_scalar(neg_m_new_act, m_new_act, op0=nl.multiply, operand0=-1.0)
    m_diff_act = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="m_diff_act")
    nisa.tensor_tensor(m_diff_act, m, neg_m_new_act, op=nl.add)
    rescale_act_g1 = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="rescale_act_g1")
    nisa.activation(rescale_act_g1, op=nl.exp, data=m_diff_act)

    # Broadcast rescale_act [GQA, 1] → [PMAX, GQA]
    rescale_act_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="rescale_act_psum")
    nisa.nc_transpose(rescale_act_psum, rescale_act_g1.ap([[1, GQA], [0, PMAX]], offset=0))
    rescale_act = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="rescale_act")
    nisa.tensor_copy(rescale_act, rescale_act_psum)

    # Rescale existing accumulators
    nisa.tensor_tensor(v_acc, v_acc, rescale_act, op=nl.multiply)
    nisa.tensor_tensor(sum_acc, sum_acc, rescale_act, op=nl.multiply)

    # exp(score_active - m_new_act) [PMAX, GQA]
    m_new_act_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="m_new_act_psum")
    nisa.nc_transpose(m_new_act_psum, m_new_act.ap([[1, GQA], [0, PMAX]], offset=0))
    m_new_act_broad = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="m_new_act_broad")
    nisa.tensor_copy(m_new_act_broad, m_new_act_psum)

    neg_m_new_act_broad = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="neg_m_new_act_broad")
    nisa.tensor_scalar(neg_m_new_act_broad, m_new_act_broad, op0=nl.multiply, operand0=-1.0)
    score_act_shifted = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="score_act_shifted")
    nisa.tensor_tensor(score_act_shifted, score_active, neg_m_new_act_broad, op=nl.add)

    score_act_exp = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="score_act_exp")
    nisa.activation(score_act_exp, op=nl.exp, data=score_act_shifted)

    # sum_acc += sum(score_act_exp)
    nisa.tensor_tensor(sum_acc, sum_acc, score_act_exp, op=nl.add)

    # tp_broadcast: v_active[PMAX, 1] → v_act_packed[PMAX, GQA]
    v_act_packed_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="v_act_packed_psum_T")
    nisa.nc_transpose(v_act_packed_psum_T, v_active.ap([[1, PMAX], [0, GQA]], offset=0))
    v_act_packed_sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="v_act_packed_sbuf_T")
    nisa.tensor_copy(v_act_packed_sbuf_T, v_act_packed_psum_T)
    v_act_packed_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="v_act_packed_psum")
    nisa.nc_transpose(v_act_packed_psum, v_act_packed_sbuf_T)
    v_act_packed = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="v_act_packed")
    nisa.tensor_copy(v_act_packed, v_act_packed_psum)

    v_act_weighted = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="v_act_weighted")
    nisa.tensor_tensor(v_act_weighted, v_act_packed, score_act_exp, op=nl.multiply)
    nisa.tensor_tensor(v_acc, v_acc, v_act_weighted, op=nl.add)

    # --- Normalize: attn_out = v_acc / sum_acc ---
    sum_safe = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="sum_safe")
    nisa.tensor_scalar(sum_safe, sum_acc, op0=nl.add, operand0=1e-9)
    rsqrt_sum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="rsqrt_sum")
    nisa.activation(rsqrt_sum, op=nl.rsqrt, data=sum_safe)
    inv_sum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="inv_sum")
    nisa.tensor_tensor(inv_sum, rsqrt_sum, rsqrt_sum, op=nl.multiply)

    attn_out = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name="attn_out")
    nisa.tensor_tensor(attn_out, v_acc, inv_sum, op=nl.multiply)

    # =========================================================================
    # FUSED OUTPUT PROJECTION — head-outer, h_blk-inner loop order
    # =========================================================================
    res_psum_0 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_0")
    res_psum_1 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_1")
    res_psum_2 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_2")
    res_psum_3 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_3")
    for head in nl.affine_range(HQ_TP_CONST):
        nisa.nc_matmul(res_psum_0, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 0*F_MAX:1*F_MAX])
        nisa.nc_matmul(res_psum_1, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 1*F_MAX:2*F_MAX])
        nisa.nc_matmul(res_psum_2, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 2*F_MAX:3*F_MAX])
        nisa.nc_matmul(res_psum_3, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 3*F_MAX:4*F_MAX])
    out_sb_0 = nl.ndarray((1, F_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_sb_0")
    nisa.tensor_copy(out_sb_0, res_psum_0)
    nisa.dma_copy(dst=output_2d[0:1, 0*F_MAX:1*F_MAX], src=out_sb_0[0:1, 0:F_MAX])
    out_sb_1 = nl.ndarray((1, F_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_sb_1")
    nisa.tensor_copy(out_sb_1, res_psum_1)
    nisa.dma_copy(dst=output_2d[0:1, 1*F_MAX:2*F_MAX], src=out_sb_1[0:1, 0:F_MAX])
    out_sb_2 = nl.ndarray((1, F_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_sb_2")
    nisa.tensor_copy(out_sb_2, res_psum_2)
    nisa.dma_copy(dst=output_2d[0:1, 2*F_MAX:3*F_MAX], src=out_sb_2[0:1, 0:F_MAX])
    out_sb_3 = nl.ndarray((1, F_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_sb_3")
    nisa.tensor_copy(out_sb_3, res_psum_3)
    nisa.dma_copy(dst=output_2d[0:1, 3*F_MAX:4*F_MAX], src=out_sb_3[0:1, 0:F_MAX])

    return output, k_rope_out, v_out
