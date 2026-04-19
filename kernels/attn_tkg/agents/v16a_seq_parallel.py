"""
v16a_seq_parallel: v14b_kv_norm_pretransposed + S-tile sharding across LNC=2 cores.

Baseline v14b_kv_norm_pretransposed already does HEAD sharding (core 0: q-heads
0..3, core 1: q-heads 4..7) for Wq/Wo with a single end-of-kernel sendrecv
all-reduce on the o-proj output.

This variant *additionally* shards the flash-decode S axis. With NUM_S_TILES=8,
core 0 processes tiles [0,1,2,3] and core 1 processes tiles [4,5,6,7]. The
halved work covers:
  - early K/V cache DMA (one tile per s_t)
  - per-tile mask hoist (transpose-based broadcast)
  - Pass-1 (score nc_matmul + masked max reduction)
  - Pass-2 (softmax, v_weighted accumulate)

The active-position contribution (single score from current position) is only
added on core 0 (or n_prgs==1). A single end-of-pass-2 log-sum-exp merge over
three tensors (max, sum, v) combines the two cores' partial results before
normalization.

Other pieces unchanged from v14b_pretransposed:
  - Wq/Wk/Wv tile-transposed in HBM
  - Wq per-owned-head DMA
  - Wo per-owned-head DMA
  - pre-attn RMSNorm, KV-proj, Q-proj, Q/K rmsnorm+RoPE
  - Per-head o-proj + final sendrecv add
  - KV cache in-place scatter (guarded by prg_id)
"""

import math
import nki.language as nl
import nki.isa as nisa

PMAX = 128
F_MAX = 512
EPS = 1e-6
INV_SQRT_D = float(1.0 / math.sqrt(128.0))
PSUM_BANK_SIZE = 2048


def qwen3_attn_tkg_fused_oproj_v16a_seq_parallel(
    hidden_sb,
    Wq,
    Wk,
    Wv,
    Wo,
    q_norm_weight,
    k_norm_weight,
    gamma_pre_attn,
    K_cache,
    V_cache,
    cos,
    sin,
    position_ids,
    out_sb=None,
    sbm=None,
):
    assert sbm is not None, "sbm (SbufManager) is required"

    B = cos.shape[0]
    H = Wq.shape[1]
    Hq_out = Wq.shape[0]
    d = PMAX
    Hq_tp = Hq_out // d
    Hkv_tp = 1
    GQA = Hq_tp // Hkv_tp
    S_prior = K_cache.shape[2]
    num_h_tiles = H // PMAX

    half_d = d // 2

    H_wo = Wo.shape[1]
    num_h_blocks = H_wo // F_MAX

    assert S_prior % PMAX == 0
    NUM_S_TILES = S_prior // PMAX
    NUM_H_TILES = 16
    HQ_TP_CONST = 8
    NUM_H_BLOCKS = 4
    assert H == NUM_H_TILES * PMAX
    assert Hq_tp == HQ_TP_CONST

    cos_col = cos.reshape((PMAX, B))
    sin_col = sin.reshape((PMAX, B))

    sbm.open_scope("attn_outer")

    K_cache_2d = K_cache.reshape((S_prior, d))
    V_cache_2d = V_cache.reshape((S_prior, d))

    # -------------------------------------------------------------------------
    # LNC sharding — heads AND S-tiles
    # -------------------------------------------------------------------------
    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)

    if n_prgs == 1:
        owned_heads = [0, 1, 2, 3, 4, 5, 6, 7]
        owned_s_tiles = [0, 1, 2, 3, 4, 5, 6, 7]
    elif prg_id == 0:
        owned_heads = [0, 1, 2, 3]
        owned_s_tiles = [0, 1, 2, 3]
    else:
        owned_heads = [4, 5, 6, 7]
        owned_s_tiles = [4, 5, 6, 7]

    # =========================================================================
    # EARLY K/V CACHE DMA — only owned tiles
    # =========================================================================
    k_cache_tiles_hbm = []
    v_cache_tiles_hbm = []
    for s_t in owned_s_tiles:
        k_ct_early = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name=f"k_ct_early_{s_t}")
        nisa.dma_transpose(
            dst=k_ct_early.ap([[PMAX, PMAX], [1, 1], [1, 1], [1, PMAX]], offset=0),
            src=K_cache_2d.ap([[d, PMAX], [1, 1], [1, 1], [1, d]], offset=s_t * PMAX * d),
        )
        k_cache_tiles_hbm.append(k_ct_early)

        v_ct_early = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name=f"v_ct_early_{s_t}")
        nisa.dma_copy(
            dst=v_ct_early,
            src=V_cache_2d.ap(pattern=[[d, PMAX], [1, d]], offset=s_t * PMAX * d),
            dge_mode=nisa.dge_mode.hwdge,
        )
        v_cache_tiles_hbm.append(v_ct_early)

    # Wk/Wv hoist — still needed fully on both cores (kv projection is duplicated).
    wk_sb = sbm.alloc_stack((PMAX, NUM_H_TILES * d), nl.bfloat16, name="wk_sb")
    nisa.dma_copy(dst=wk_sb, src=Wk, dge_mode=nisa.dge_mode.hwdge)

    wv_sb = sbm.alloc_stack((PMAX, NUM_H_TILES * d), nl.bfloat16, name="wv_sb")
    nisa.dma_copy(dst=wv_sb, src=Wv, dge_mode=nisa.dge_mode.hwdge)

    # =========================================================================
    # CONSTANTS
    # =========================================================================
    qnw_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="qnw_bf16")
    nisa.dma_copy(dst=qnw_bf16, src=q_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
    qnw_sb = sbm.alloc_stack((PMAX, 1), nl.float32, name="qnw_sb")
    nisa.tensor_copy(qnw_sb, qnw_bf16)

    knw_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="knw_bf16")
    nisa.dma_copy(dst=knw_bf16, src=k_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
    knw_sb = sbm.alloc_stack((PMAX, 1), nl.float32, name="knw_sb")
    nisa.tensor_copy(knw_sb, knw_bf16)

    cos_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="cos_bf16")
    sin_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="sin_bf16")
    nisa.dma_copy(dst=cos_bf16, src=cos_col, dge_mode=nisa.dge_mode.hwdge)
    nisa.dma_copy(dst=sin_bf16, src=sin_col, dge_mode=nisa.dge_mode.hwdge)
    cos_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name="cos_f32")
    sin_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name="sin_f32")
    nisa.tensor_copy(cos_f32, cos_bf16)
    nisa.tensor_copy(sin_f32, sin_bf16)

    rms_ones = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)

    gpan_bf16 = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="gpan_bf16")
    nisa.dma_copy(dst=gpan_bf16,
                  src=gamma_pre_attn.reshape((H, 1)).ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge)
    gpan_f32 = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="gpan_f32")
    nisa.tensor_copy(gpan_f32, gpan_bf16)

    pos_write_i32_raw = sbm.alloc_stack((1, 1), nl.int32, name="pos_write_i32_raw")
    nisa.dma_copy(dst=pos_write_i32_raw,
                  src=position_ids.reshape((B, 1))[0:1, 0:1],
                  dge_mode=nisa.dge_mode.hwdge)
    pos_write_i32 = sbm.alloc_stack((1, 1), nl.uint32, name="pos_write_i32")
    nisa.tensor_copy(pos_write_i32, pos_write_i32_raw)

    h_all = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="h_all")

    # =========================================================================
    # PRE-ATTENTION RMSNORM (duplicated on both cores — cheap, avoids sendrecv)
    # =========================================================================
    sbm.open_scope("pre_attn_norm")

    h_f32 = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_f32")
    nisa.tensor_copy(h_f32, hidden_sb)

    h_sq = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_sq")
    nisa.tensor_tensor(h_sq, h_f32, h_f32, op=nl.multiply)

    h_sq_sum = sbm.alloc_stack((PMAX, 1), nl.float32, name="h_sq_sum")
    nisa.tensor_reduce(dst=h_sq_sum, op=nl.add, data=h_sq, axis=1)

    h_sq_sum_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="h_sq_sum_bf16")
    nisa.tensor_copy(h_sq_sum_bf16, h_sq_sum)
    h_sq_total_psum = nl.zeros((PMAX, 1), nl.float32, buffer=nl.psum)
    nisa.nc_matmul(h_sq_total_psum, stationary=rms_ones, moving=h_sq_sum_bf16)
    h_sq_total = sbm.alloc_stack((PMAX, 1), nl.float32, name="h_sq_total")
    nisa.tensor_copy(h_sq_total, h_sq_total_psum)

    h_mean_sq = sbm.alloc_stack((PMAX, 1), nl.float32, name="h_mean_sq")
    nisa.tensor_scalar(h_mean_sq, h_sq_total,
                       op0=nl.multiply, operand0=1.0 / H,
                       op1=nl.add, operand1=EPS)
    h_rms_inv = sbm.alloc_stack((PMAX, 1), nl.float32, name="h_rms_inv")
    nisa.activation(h_rms_inv, op=nl.rsqrt, data=h_mean_sq)

    h_rms_T_psum = nl.ndarray((num_h_tiles, PMAX), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_rms_T_psum, h_rms_inv.ap([[1, PMAX], [0, num_h_tiles]], offset=0))
    h_rms_T = sbm.alloc_stack((num_h_tiles, PMAX), nl.float32, name="h_rms_T")
    nisa.tensor_copy(h_rms_T, h_rms_T_psum)
    h_rms_expanded_psum = nl.ndarray((PMAX, num_h_tiles), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_rms_expanded_psum, h_rms_T)
    h_rms_expanded = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_rms_expanded")
    nisa.tensor_copy(h_rms_expanded, h_rms_expanded_psum)

    h_normed = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_normed")
    nisa.tensor_tensor(h_normed, h_f32, h_rms_expanded, op=nl.multiply)
    nisa.tensor_tensor(h_normed, h_normed, gpan_f32, op=nl.multiply)

    nisa.tensor_copy(h_all, h_normed)

    sbm.close_scope()  # pre_attn_norm

    # =========================================================================
    # Wq LOAD — all heads on both cores.
    # v16a shards S-tiles (not heads) in the attention pass.  Each core must
    # compute correct Q for all GQA heads so that the per-S-tile KV scores are
    # correct regardless of which tile-range is valid for a given position.
    # =========================================================================
    wq_head_sb = [None] * HQ_TP_CONST
    for q_h in range(HQ_TP_CONST):
        w = sbm.alloc_stack((PMAX, NUM_H_TILES * d), nl.bfloat16, name=f"wq_head_{q_h}")
        nisa.dma_copy(
            dst=w,
            src=Wq[q_h * PMAX:(q_h + 1) * PMAX, :],
            dge_mode=nisa.dge_mode.hwdge,
        )
        wq_head_sb[q_h] = w

    Wo_reshaped = Wo.reshape((Hq_tp, d, H_wo))

    # =========================================================================
    # Persistent tensors
    # =========================================================================
    k_rope = sbm.alloc_stack((PMAX, B), nl.float32, name="k_rope")
    k_rope_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_rope_bf16")
    k_rope_T_sb = sbm.alloc_stack((B, PMAX), nl.bfloat16, name="k_rope_T_sb")
    v_active = sbm.alloc_stack((PMAX, B), nl.float32, name="v_active")
    v_T_sb = sbm.alloc_stack((B, PMAX), nl.bfloat16, name="v_T_sb")
    q_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_bf16")

    k_rope_bf16_f32 = sbm.alloc_stack((PMAX, B), nl.float32, name="k_rope_bf16_f32")
    v_from_bf16 = sbm.alloc_stack((PMAX, B), nl.float32, name="v_from_bf16")

    wo_sbuf = [None] * HQ_TP_CONST
    for head in owned_heads:
        wo_tile = sbm.alloc_stack((PMAX, H_wo), nl.bfloat16, name=f"wo_tile_h{head}")
        wo_sbuf[head] = wo_tile

    # =========================================================================
    # KV PROJ
    # =========================================================================
    sbm.open_scope("kv_proj")

    k_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum)
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(k_psum, stationary=wk_sb[0:PMAX, h_t * d:(h_t + 1) * d], moving=h_all[0:PMAX, h_t:h_t + 1])

    k_vec = sbm.alloc_stack((PMAX, B), nl.float32, name="k_vec")
    nisa.tensor_copy(k_vec, k_psum)

    # K RMSNorm
    k_sq = sbm.alloc_stack((PMAX, B), nl.float32, name="k_sq")
    nisa.tensor_tensor(k_sq, k_vec, k_vec, op=nl.multiply)
    k_sq_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_sq_bf16")
    nisa.tensor_copy(k_sq_bf16, k_sq)
    k_sum_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(k_sum_psum, stationary=rms_ones, moving=k_sq_bf16)
    k_sum_sb = sbm.alloc_stack((PMAX, B), nl.float32, name="k_sum_sb")
    nisa.tensor_copy(k_sum_sb, k_sum_psum)
    k_mean_sq = sbm.alloc_stack((PMAX, B), nl.float32, name="k_mean_sq")
    nisa.tensor_scalar(k_mean_sq, k_sum_sb, op0=nl.multiply, operand0=1.0 / d, op1=nl.add, operand1=EPS)
    k_rms_inv = sbm.alloc_stack((PMAX, B), nl.float32, name="k_rms_inv")
    nisa.activation(k_rms_inv, op=nl.rsqrt, data=k_mean_sq)
    k_normed = sbm.alloc_stack((PMAX, B), nl.float32, name="k_normed")
    nisa.tensor_tensor(k_normed, k_vec, k_rms_inv, op=nl.multiply)
    k_normed2 = sbm.alloc_stack((PMAX, B), nl.float32, name="k_normed2")
    nisa.tensor_tensor(k_normed2, k_normed, knw_sb, op=nl.multiply)

    # K RoPE
    rot_k = sbm.alloc_stack((PMAX, B), nl.float32, name="rot_k")
    neg_k_upper = sbm.alloc_stack((half_d, B), nl.float32, name="neg_k_upper")
    nisa.tensor_scalar(neg_k_upper, k_normed2[half_d:d, 0:B], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(rot_k[0:half_d, 0:B], neg_k_upper)
    nisa.tensor_copy(rot_k[half_d:d, 0:B], k_normed2[0:half_d, 0:B])
    k_cos = sbm.alloc_stack((PMAX, B), nl.float32, name="k_cos")
    k_sin_part = sbm.alloc_stack((PMAX, B), nl.float32, name="k_sin_part")
    nisa.tensor_tensor(k_cos, k_normed2, cos_f32, op=nl.multiply)
    nisa.tensor_tensor(k_sin_part, rot_k, sin_f32, op=nl.multiply)
    nisa.tensor_tensor(k_rope, k_cos, k_sin_part, op=nl.add)

    nisa.tensor_copy(k_rope_bf16, k_rope)
    k_rope_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(k_rope_T_psum, k_rope_bf16)
    nisa.tensor_copy(k_rope_T_sb, k_rope_T_psum)
    nisa.tensor_copy(k_rope_bf16_f32, k_rope_bf16)

    # V projection
    v_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum)
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(v_psum, stationary=wv_sb[0:PMAX, h_t * d:(h_t + 1) * d], moving=h_all[0:PMAX, h_t:h_t + 1])

    nisa.tensor_copy(v_active, v_psum)

    v_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="v_bf16")
    nisa.tensor_copy(v_bf16, v_active)
    v_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(v_T_psum, v_bf16)
    nisa.tensor_copy(v_T_sb, v_T_psum)
    nisa.tensor_copy(v_from_bf16, v_bf16)

    sbm.close_scope()  # kv_proj

    # =========================================================================
    # Q PROJ
    # =========================================================================
    sbm.open_scope("q_proj")

    # Compute Q for ALL heads (not just owned_heads).
    # Each core needs correct Q for all GQA heads so that KV scores on its
    # owned S-tiles are accurate for every head before the log-sum-exp merge.
    q_psums = []
    for q_h in range(HQ_TP_CONST):
        q_p = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(q_p, value=0.0)
        q_psums.append(q_p)

    for h_t in nl.affine_range(NUM_H_TILES):
        for q_h in range(HQ_TP_CONST):
            nisa.nc_matmul(q_psums[q_h],
                           stationary=wq_head_sb[q_h][0:PMAX, h_t * d:(h_t + 1) * d],
                           moving=h_all[0:PMAX, h_t:h_t + 1])

    q_packed_f32 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_packed_f32")
    for q_h in range(HQ_TP_CONST):
        nisa.tensor_copy(q_packed_f32[0:PMAX, q_h:q_h + 1], q_psums[q_h])

    # Q RMSNorm
    q_sq = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_sq")
    nisa.tensor_tensor(q_sq, q_packed_f32, q_packed_f32, op=nl.multiply)
    q_sq_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_sq_bf16")
    nisa.tensor_copy(q_sq_bf16, q_sq)
    q_sum_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(q_sum_psum, stationary=rms_ones, moving=q_sq_bf16)
    q_sum_sb = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_sum_sb")
    nisa.tensor_copy(q_sum_sb, q_sum_psum)
    q_mean_sq = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_mean_sq")
    nisa.tensor_scalar(q_mean_sq, q_sum_sb, op0=nl.multiply, operand0=1.0 / d, op1=nl.add, operand1=EPS)
    q_rms_inv = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_rms_inv")
    nisa.activation(q_rms_inv, op=nl.rsqrt, data=q_mean_sq)
    q_normed = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_normed")
    nisa.tensor_tensor(q_normed, q_packed_f32, q_rms_inv, op=nl.multiply)

    qnw_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(qnw_gqa_psum_T, qnw_sb.ap([[1, PMAX], [0, GQA]], offset=0))
    qnw_gqa_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="qnw_gqa_sbuf_T")
    nisa.tensor_copy(qnw_gqa_sbuf_T, qnw_gqa_psum_T)
    qnw_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(qnw_gqa_psum, qnw_gqa_sbuf_T)
    qnw_gqa = sbm.alloc_stack((PMAX, GQA), nl.float32, name="qnw_gqa")
    nisa.tensor_copy(qnw_gqa, qnw_gqa_psum)

    q_normed2 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_normed2")
    nisa.tensor_tensor(q_normed2, q_normed, qnw_gqa, op=nl.multiply)

    # Q RoPE
    cos_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(cos_gqa_psum_T, cos_f32.ap([[1, PMAX], [0, GQA]], offset=0))
    cos_gqa_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="cos_gqa_sbuf_T")
    nisa.tensor_copy(cos_gqa_sbuf_T, cos_gqa_psum_T)
    cos_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(cos_gqa_psum, cos_gqa_sbuf_T)
    cos_gqa = sbm.alloc_stack((PMAX, GQA), nl.float32, name="cos_gqa")
    nisa.tensor_copy(cos_gqa, cos_gqa_psum)

    sin_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(sin_gqa_psum_T, sin_f32.ap([[1, PMAX], [0, GQA]], offset=0))
    sin_gqa_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="sin_gqa_sbuf_T")
    nisa.tensor_copy(sin_gqa_sbuf_T, sin_gqa_psum_T)
    sin_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(sin_gqa_psum, sin_gqa_sbuf_T)
    sin_gqa = sbm.alloc_stack((PMAX, GQA), nl.float32, name="sin_gqa")
    nisa.tensor_copy(sin_gqa, sin_gqa_psum)

    rot_q = sbm.alloc_stack((PMAX, GQA), nl.float32, name="rot_q")
    neg_q_upper = sbm.alloc_stack((half_d, GQA), nl.float32, name="neg_q_upper")
    nisa.tensor_scalar(neg_q_upper, q_normed2[half_d:d, 0:GQA], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(rot_q[0:half_d, 0:GQA], neg_q_upper)
    nisa.tensor_copy(rot_q[half_d:d, 0:GQA], q_normed2[0:half_d, 0:GQA])

    q_cos = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_cos")
    q_sin_part = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_sin_part")
    nisa.tensor_tensor(q_cos, q_normed2, cos_gqa, op=nl.multiply)
    nisa.tensor_tensor(q_sin_part, rot_q, sin_gqa, op=nl.multiply)
    q_rope = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_rope")
    nisa.tensor_tensor(q_rope, q_cos, q_sin_part, op=nl.add)

    nisa.tensor_scalar(q_bf16, q_rope, op0=nl.multiply, operand0=INV_SQRT_D)

    sbm.close_scope()  # q_proj

    # =========================================================================
    # WO WEIGHT HOIST
    # =========================================================================
    for head in owned_heads:
        nisa.dma_copy(
            dst=wo_sbuf[head],
            src=Wo_reshaped.ap(
                pattern=[[H_wo, PMAX], [1, H_wo]],
                offset=head * PMAX * H_wo,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )

    # =========================================================================
    # FLASH DECODE
    # =========================================================================
    k_rope_packed_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(k_rope_packed_psum_T, k_rope_bf16_f32.ap([[1, PMAX], [0, GQA]], offset=0))
    k_rope_packed_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="k_rope_packed_sbuf_T")
    nisa.tensor_copy(k_rope_packed_sbuf_T, k_rope_packed_psum_T)
    k_rope_packed_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(k_rope_packed_psum, k_rope_packed_sbuf_T)
    k_rope_packed = sbm.alloc_stack((PMAX, GQA), nl.float32, name="k_rope_packed")
    nisa.tensor_copy(k_rope_packed, k_rope_packed_psum)

    kq_elem = sbm.alloc_stack((PMAX, GQA), nl.float32, name="kq_elem")
    nisa.tensor_tensor(kq_elem, k_rope_packed, q_bf16, op=nl.multiply)
    kq_elem_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="kq_elem_bf16")
    nisa.tensor_copy(kq_elem_bf16, kq_elem)
    score_active_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(score_active_psum, stationary=rms_ones, moving=kq_elem_bf16)
    score_active = sbm.alloc_stack((PMAX, GQA), nl.float32, name="score_active")
    nisa.tensor_copy(score_active, score_active_psum)

    position_ids_2d = position_ids.reshape((B, 1))
    pos_f32 = sbm.alloc_stack((1, 1), nl.float32, name="pos_f32")
    nisa.tensor_copy(pos_f32, pos_write_i32)

    par_index_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name="par_index_f32")
    nisa.iota(par_index_f32, pattern=[[1, 1]], offset=0, channel_multiplier=1)

    # =========================================================================
    # KV HOIST — owned tiles only
    # =========================================================================
    sbm.open_scope("kv_hoist")

    k_cache_tiles = []
    mask_gqa_tiles = []
    v_cache_tiles = []

    for i_t in range(len(owned_s_tiles)):
        s_t = owned_s_tiles[i_t]
        k_ct = k_cache_tiles_hbm[i_t]

        mask_gqa_pre = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"mask_gqa_pre_{s_t}")

        sbm.open_scope(f"tile_{s_t}")

        tile_start = s_t * PMAX
        neg_threshold = sbm.alloc_stack((1, 1), nl.float32, name=f"neg_threshold_{s_t}")
        nisa.tensor_scalar(neg_threshold, pos_f32,
                           op0=nl.multiply, operand0=-1.0,
                           op1=nl.add, operand1=float(tile_start))

        neg_thresh_psum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(neg_thresh_psum, neg_threshold.ap([[1, 1], [0, PMAX]], offset=0))
        neg_thresh_sb = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"neg_thresh_sb_{s_t}")
        nisa.tensor_copy(neg_thresh_sb, neg_thresh_psum)

        delta = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"delta_{s_t}")
        nisa.tensor_tensor(delta, par_index_f32, neg_thresh_sb, op=nl.add)

        delta_eps = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"delta_eps_{s_t}")
        nisa.tensor_scalar(delta_eps, delta, op0=nl.add, operand0=1.0)
        relu_delta = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"relu_delta_{s_t}")
        nisa.activation(relu_delta, op=nl.relu, data=delta_eps)

        clamped = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"clamped_{s_t}")
        nisa.tensor_scalar(clamped, relu_delta, op0=nl.minimum, operand0=1.0)

        mask_tile_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"mask_tile_f32_{s_t}")
        nisa.tensor_scalar(mask_tile_f32, clamped, op0=nl.multiply, operand0=-1e9)

        mask_gqa_pre_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(mask_gqa_pre_psum_T, mask_tile_f32.ap([[1, PMAX], [0, GQA]], offset=0))
        mask_gqa_pre_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name=f"mask_gqa_pre_sbuf_T_{s_t}")
        nisa.tensor_copy(mask_gqa_pre_sbuf_T, mask_gqa_pre_psum_T)
        mask_gqa_pre_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(mask_gqa_pre_psum, mask_gqa_pre_sbuf_T)
        nisa.tensor_copy(mask_gqa_pre, mask_gqa_pre_psum)

        sbm.close_scope()  # tile_{s_t}

        k_cache_tiles.append(k_ct)
        mask_gqa_tiles.append(mask_gqa_pre)
        v_cache_tiles.append(v_cache_tiles_hbm[i_t])

    # =========================================================================
    # PASS 1 — owned S-tiles only
    # =========================================================================
    sbm.open_scope("pass1_active")

    global_max_g1 = sbm.alloc_stack((GQA, 1), nl.float32, name="global_max_g1")
    nisa.memset(global_max_g1, value=-1e9)

    # Active position contribution to max — only core 0 counts it, so the
    # corresponding exp term (added in pass-2) is self-consistent with this core's
    # local max.
    if n_prgs == 1 or prg_id == 0:
        score_act_T_psum = nl.zeros((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(score_act_T_psum, score_active)
        score_act_T_sb = sbm.alloc_stack((GQA, PMAX), nl.float32, name="score_act_T_sb")
        nisa.tensor_copy(score_act_T_sb, score_act_T_psum)
        score_active_g1 = sbm.alloc_stack((GQA, 1), nl.float32, name="score_active_g1")
        nisa.tensor_reduce(dst=score_active_g1, op=nl.maximum, data=score_act_T_sb, axis=1)
        nisa.tensor_tensor(global_max_g1, global_max_g1, score_active_g1, op=nl.maximum)

    saved_scores = []

    for i_t in range(len(owned_s_tiles)):
        s_t = owned_s_tiles[i_t]
        score_sb_masked = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"score_sb_masked_{s_t}")

        sbm.open_scope(f"p1_tile_{s_t}")

        score_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(score_psum, stationary=k_cache_tiles[i_t], moving=q_bf16)
        score_sb = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"score_sb_{s_t}")
        nisa.tensor_copy(score_sb, score_psum)

        mask_gqa = mask_gqa_tiles[i_t]
        nisa.tensor_tensor(score_sb_masked, score_sb, mask_gqa, op=nl.add)

        score_T_psum = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(score_T_psum, score_sb_masked)
        score_T_sb = sbm.alloc_stack((GQA, PMAX), nl.float32, name=f"score_T_sb_{s_t}")
        nisa.tensor_copy(score_T_sb, score_T_psum)

        tile_max_vec = sbm.alloc_stack((GQA, 1), nl.float32, name=f"tile_max_vec_{s_t}")
        nisa.tensor_reduce(dst=tile_max_vec, op=nl.maximum, data=score_T_sb, axis=1)
        nisa.tensor_tensor(global_max_g1, global_max_g1, tile_max_vec, op=nl.maximum)

        sbm.close_scope()  # p1_tile_{s_t}

        saved_scores.append(score_sb_masked)

    neg_max_g1 = sbm.alloc_stack((GQA, 1), nl.float32, name="neg_max_g1")
    nisa.tensor_scalar(neg_max_g1, global_max_g1, op0=nl.multiply, operand0=-1.0)

    neg_max_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(
        neg_max_psum,
        neg_max_g1.ap([[1, GQA], [0, PMAX]], offset=0),
    )
    neg_max = sbm.alloc_stack((PMAX, GQA), nl.float32, name="neg_max")
    nisa.tensor_copy(neg_max, neg_max_psum)

    attn_out = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="attn_out")

    # =========================================================================
    # PASS 2 — owned S-tiles only; add active-position on core 0.
    # =========================================================================
    sbm.open_scope("pass2")

    v_acc = sbm.alloc_stack((PMAX, GQA), nl.float32, name="v_acc")
    nisa.memset(v_acc, value=0.0)
    sum_acc = sbm.alloc_stack((PMAX, GQA), nl.float32, name="sum_acc")
    nisa.memset(sum_acc, value=0.0)

    for i_t in range(len(owned_s_tiles)):
        s_t = owned_s_tiles[i_t]
        score2_shifted = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"score2_shifted_{s_t}")
        nisa.tensor_tensor(score2_shifted, saved_scores[i_t], neg_max, op=nl.add)

        score2_exp = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"score2_exp_{s_t}")
        nisa.activation(score2_exp, op=nl.exp, data=score2_shifted)

        score2_exp_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name=f"score2_exp_bf16_{s_t}")
        nisa.tensor_copy(score2_exp_bf16, score2_exp)

        tile_sum_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(tile_sum_psum, stationary=rms_ones, moving=score2_exp_bf16)
        tile_sum = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"tile_sum_{s_t}")
        nisa.tensor_copy(tile_sum, tile_sum_psum)
        nisa.tensor_tensor(sum_acc, sum_acc, tile_sum, op=nl.add)

        v_weighted_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(v_weighted_psum, stationary=v_cache_tiles[i_t], moving=score2_exp_bf16)
        v_weighted = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"v_weighted_{s_t}")
        nisa.tensor_copy(v_weighted, v_weighted_psum)
        nisa.tensor_tensor(v_acc, v_acc, v_weighted, op=nl.add)

    # Active position contribution — core 0 only.
    if n_prgs == 1 or prg_id == 0:
        score_act_shifted = sbm.alloc_stack((PMAX, GQA), nl.float32, name="score_act_shifted")
        nisa.tensor_tensor(score_act_shifted, score_active, neg_max, op=nl.add)
        score_act_exp = sbm.alloc_stack((PMAX, GQA), nl.float32, name="score_act_exp")
        nisa.activation(score_act_exp, op=nl.exp, data=score_act_shifted)
        nisa.tensor_tensor(sum_acc, sum_acc, score_act_exp, op=nl.add)

        v_act_packed_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(v_act_packed_psum_T, v_from_bf16.ap([[1, PMAX], [0, GQA]], offset=0))
        v_act_packed_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="v_act_packed_sbuf_T")
        nisa.tensor_copy(v_act_packed_sbuf_T, v_act_packed_psum_T)
        v_act_packed_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(v_act_packed_psum, v_act_packed_sbuf_T)
        v_act_packed = sbm.alloc_stack((PMAX, GQA), nl.float32, name="v_act_packed")
        nisa.tensor_copy(v_act_packed, v_act_packed_psum)

        v_act_weighted = sbm.alloc_stack((PMAX, GQA), nl.float32, name="v_act_weighted")
        nisa.tensor_tensor(v_act_weighted, v_act_packed, score_act_exp, op=nl.multiply)
        nisa.tensor_tensor(v_acc, v_acc, v_act_weighted, op=nl.add)

    # =========================================================================
    # CROSS-CORE LOG-SUM-EXP MERGE
    #
    # Each core holds (local_max, sum_acc, v_acc) computed over a disjoint set of
    # positions. Merge by:
    #   merged_max = max(local_max, peer_max)
    #   sum_acc_new = sum_acc*exp(local_max - merged_max) + sum_peer*exp(peer_max - merged_max)
    #   v_acc_new   = v_acc  *exp(local_max - merged_max) + v_peer  *exp(peer_max - merged_max)
    # =========================================================================
    if n_prgs > 1:
        peer = 1 - prg_id

        max_peer = sbm.alloc_stack((GQA, 1), nl.float32, name="max_peer")
        sum_peer = sbm.alloc_stack((PMAX, GQA), nl.float32, name="sum_peer")
        v_peer = sbm.alloc_stack((PMAX, GQA), nl.float32, name="v_peer")

        nisa.sendrecv(src=global_max_g1, dst=max_peer,
                      send_to_rank=peer, recv_from_rank=peer, pipe_id=1)
        nisa.sendrecv(src=sum_acc, dst=sum_peer,
                      send_to_rank=peer, recv_from_rank=peer, pipe_id=2)
        nisa.sendrecv(src=v_acc, dst=v_peer,
                      send_to_rank=peer, recv_from_rank=peer, pipe_id=3)

        # merged_max: [GQA, 1]
        merged_max_g1 = sbm.alloc_stack((GQA, 1), nl.float32, name="merged_max_g1")
        nisa.tensor_tensor(merged_max_g1, global_max_g1, max_peer, op=nl.maximum)

        # Broadcast [GQA,1] -> [PMAX, GQA] via double-transpose (same pattern as
        # neg_max above). We need three broadcast tensors:
        #   merged_max_bc, local_max_bc, peer_max_bc
        # (NKI does not allow inner function definitions — inlined below.)

        merged_max_bc_psum_T = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(merged_max_bc_psum_T, merged_max_g1.ap([[1, GQA], [0, PMAX]], offset=0))
        merged_max_bc = sbm.alloc_stack((PMAX, GQA), nl.float32, name="merged_max_bc")
        nisa.tensor_copy(merged_max_bc, merged_max_bc_psum_T)

        local_max_bc_psum_T = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(local_max_bc_psum_T, global_max_g1.ap([[1, GQA], [0, PMAX]], offset=0))
        local_max_bc = sbm.alloc_stack((PMAX, GQA), nl.float32, name="local_max_bc")
        nisa.tensor_copy(local_max_bc, local_max_bc_psum_T)

        peer_max_bc_psum_T = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(peer_max_bc_psum_T, max_peer.ap([[1, GQA], [0, PMAX]], offset=0))
        peer_max_bc = sbm.alloc_stack((PMAX, GQA), nl.float32, name="peer_max_bc")
        nisa.tensor_copy(peer_max_bc, peer_max_bc_psum_T)

        # delta_local = local_max - merged_max ; delta_peer = peer_max - merged_max
        # Both are <= 0.
        neg_merged = sbm.alloc_stack((PMAX, GQA), nl.float32, name="neg_merged_bc")
        nisa.tensor_scalar(neg_merged, merged_max_bc, op0=nl.multiply, operand0=-1.0)
        delta_local = sbm.alloc_stack((PMAX, GQA), nl.float32, name="delta_local")
        nisa.tensor_tensor(delta_local, local_max_bc, neg_merged, op=nl.add)
        delta_peer = sbm.alloc_stack((PMAX, GQA), nl.float32, name="delta_peer")
        nisa.tensor_tensor(delta_peer, peer_max_bc, neg_merged, op=nl.add)

        exp_local = sbm.alloc_stack((PMAX, GQA), nl.float32, name="exp_local")
        nisa.activation(exp_local, op=nl.exp, data=delta_local)
        exp_peer = sbm.alloc_stack((PMAX, GQA), nl.float32, name="exp_peer")
        nisa.activation(exp_peer, op=nl.exp, data=delta_peer)

        sum_local_scaled = sbm.alloc_stack((PMAX, GQA), nl.float32, name="sum_local_scaled")
        nisa.tensor_tensor(sum_local_scaled, sum_acc, exp_local, op=nl.multiply)
        sum_peer_scaled = sbm.alloc_stack((PMAX, GQA), nl.float32, name="sum_peer_scaled")
        nisa.tensor_tensor(sum_peer_scaled, sum_peer, exp_peer, op=nl.multiply)
        nisa.tensor_tensor(sum_acc, sum_local_scaled, sum_peer_scaled, op=nl.add)

        v_local_scaled = sbm.alloc_stack((PMAX, GQA), nl.float32, name="v_local_scaled")
        nisa.tensor_tensor(v_local_scaled, v_acc, exp_local, op=nl.multiply)
        v_peer_scaled = sbm.alloc_stack((PMAX, GQA), nl.float32, name="v_peer_scaled")
        nisa.tensor_tensor(v_peer_scaled, v_peer, exp_peer, op=nl.multiply)
        nisa.tensor_tensor(v_acc, v_local_scaled, v_peer_scaled, op=nl.add)

    sum_safe = sbm.alloc_stack((PMAX, GQA), nl.float32, name="sum_safe")
    nisa.tensor_scalar(sum_safe, sum_acc, op0=nl.add, operand0=1e-9)
    rsqrt_sum = sbm.alloc_stack((PMAX, GQA), nl.float32, name="rsqrt_sum")
    nisa.activation(rsqrt_sum, op=nl.rsqrt, data=sum_safe)
    inv_sum = sbm.alloc_stack((PMAX, GQA), nl.float32, name="inv_sum")
    nisa.tensor_tensor(inv_sum, rsqrt_sum, rsqrt_sum, op=nl.multiply)

    nisa.tensor_tensor(attn_out, v_acc, inv_sum, op=nl.multiply)

    sbm.close_scope()  # pass2

    # =========================================================================
    # O_PROJ — per-owned-head; final sendrecv add unchanged.
    # =========================================================================
    sbm.open_scope("o_proj")

    NUM_OUT_COLS = H_wo // PMAX
    CHUNKS_PER_BLOCK = F_MAX // PMAX

    if out_sb is None:
        out_sb = sbm.alloc_stack((PMAX, NUM_OUT_COLS), nl.bfloat16, name="out_sb")

    out_sb_tmp = sbm.alloc_stack((1, F_MAX), nl.bfloat16, name="out_sb_tmp")
    col_tmp = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="col_tmp")
    chunk_T_psum = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.psum)

    res_psum_0 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum)
    res_psum_1 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum)
    res_psum_2 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum)
    res_psum_3 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum)

    for head in owned_heads:
        nisa.nc_matmul(res_psum_0, stationary=attn_out[0:PMAX, head:head + 1], moving=wo_sbuf[head][0:PMAX, 0 * F_MAX:1 * F_MAX])
        nisa.nc_matmul(res_psum_1, stationary=attn_out[0:PMAX, head:head + 1], moving=wo_sbuf[head][0:PMAX, 1 * F_MAX:2 * F_MAX])
        nisa.nc_matmul(res_psum_2, stationary=attn_out[0:PMAX, head:head + 1], moving=wo_sbuf[head][0:PMAX, 2 * F_MAX:3 * F_MAX])
        nisa.nc_matmul(res_psum_3, stationary=attn_out[0:PMAX, head:head + 1], moving=wo_sbuf[head][0:PMAX, 3 * F_MAX:4 * F_MAX])

    nisa.tensor_copy(out_sb_tmp, res_psum_0)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 0 * PMAX:1 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 0:1], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 1 * PMAX:2 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 1:2], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 2 * PMAX:3 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 2:3], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 3 * PMAX:4 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 3:4], col_tmp)

    nisa.tensor_copy(out_sb_tmp, res_psum_1)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 0 * PMAX:1 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 4:5], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 1 * PMAX:2 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 5:6], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 2 * PMAX:3 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 6:7], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 3 * PMAX:4 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 7:8], col_tmp)

    nisa.tensor_copy(out_sb_tmp, res_psum_2)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 0 * PMAX:1 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 8:9], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 1 * PMAX:2 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 9:10], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 2 * PMAX:3 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 10:11], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 3 * PMAX:4 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 11:12], col_tmp)

    nisa.tensor_copy(out_sb_tmp, res_psum_3)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 0 * PMAX:1 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 12:13], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 1 * PMAX:2 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 13:14], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 2 * PMAX:3 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 14:15], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 3 * PMAX:4 * PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 15:16], col_tmp)

    if n_prgs > 1:
        peer_buf = sbm.alloc_stack(out_sb.shape, nl.bfloat16, name="out_sb_peer")
        peer_id = 1 - prg_id
        nisa.sendrecv(
            src=out_sb,
            dst=peer_buf,
            send_to_rank=peer_id,
            recv_from_rank=peer_id,
            pipe_id=0,
        )
        nisa.tensor_tensor(out_sb, out_sb, peer_buf, op=nl.add)

    sbm.close_scope()  # o_proj
    sbm.close_scope()  # pass1_active
    sbm.close_scope()  # kv_hoist

    # =========================================================================
    # KV SCATTER — unchanged.
    # =========================================================================
    if n_prgs == 1 or prg_id == 0:
        nisa.dma_copy(
            dst=V_cache.reshape((B * S_prior, d)).ap(
                pattern=[[d, 1], [1, d]], offset=0,
                scalar_offset=pos_write_i32, indirect_dim=0,
            ),
            src=v_T_sb,
        )
    if n_prgs == 1 or prg_id == 1:
        nisa.dma_copy(
            dst=K_cache.reshape((B * S_prior, d)).ap(
                pattern=[[d, 1], [1, d]], offset=0,
                scalar_offset=pos_write_i32, indirect_dim=0,
            ),
            src=k_rope_T_sb,
        )

    return out_sb
