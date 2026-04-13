"""
v13bc_sbm: Sub-function version of qwen3_attn_tkg_fused_oproj_v13bc.

Changes from v13_BC:
  - No @nki.jit — sub-function called from inside a JIT kernel.
  - All nl.ndarray(..., buffer=nl.sbuf) → sbm.alloc_stack(...).
  - Main output in SBUF: returns out_sb [1, H_wo] bf16 in SBUF (no output HBM tensor).
  - k_rope_out and v_out stay nl.shared_hbm (unchanged).
  - No LNC sharding — no nl.program_id. Both cores produce full [1, H_wo=2048] output.
  - out_sb_0..3 intermediates eliminated: single out_sb_tmp [1, F_MAX] + activation copy.
  - Removed env-var block (belongs in harness).
  - Scope layout matches spec: attn_outer NOT closed — caller closes it.
"""

import math
import nki.language as nl
import nki.isa as nisa

PMAX = 128
F_MAX = 512
EPS = 1e-6
INV_SQRT_D = float(1.0 / math.sqrt(128.0))


def qwen3_attn_tkg_fused_oproj_v13bc(
    hidden_states,   # [B, 1, H]          bf16  (B=1)
    Wq,              # [Hq_tp*d, H]       bf16  [1024, 2048]
    Wk,              # [Hkv_tp*d, H]      bf16  [128, 2048]
    Wv,              # [Hkv_tp*d, H]      bf16  [128, 2048]
    Wo,              # [Hq_tp*d, H_wo]    bf16  [1024, 2048] transposed o_proj weight
    q_norm_weight,   # [d]                bf16  [128]
    k_norm_weight,   # [d]                bf16  [128]
    K_cache,         # [B, 1, S_prior, d] bf16
    V_cache,         # [B, 1, S_prior, d] bf16
    cos,             # [B, d]             bf16
    sin,             # [B, d]             bf16
    position_ids,    # [B, 1]             int32
    out_sb=None,     # optional [1, H_wo] bf16 sbuf tensor; alloc_stack'd if None
    sbm=None,        # required: SbufManager instance
):
    """
    Fused QKV + RMSNorm + RoPE + flash decode + output projection.
    Sub-function — no @nki.jit. Called from inside a JIT kernel that owns sbm.

    Returns (out_sb, k_rope_out, v_out):
      out_sb:     [1, H_wo=2048] bf16 in SBUF — valid until caller closes attn_outer scope
      k_rope_out: [B, d] = [1, 128] bf16 in shared_hbm
      v_out:      [B, d] = [1, 128] bf16 in shared_hbm

    IMPORTANT: Both LNC cores produce the full [1, 2048] output — no program_id sharding.
    Caller must consume out_sb before closing the scope that opened attn_outer.
    """
    assert sbm is not None, "sbm (SbufManager) is required"

    # --- Dimensions ---
    B = hidden_states.shape[0]      # 1
    H = hidden_states.shape[2]      # 2048
    Hq_out = Wq.shape[0]            # 1024  = Hq_tp * d
    d = PMAX                        # 128
    Hq_tp = Hq_out // d             # 8
    Hkv_tp = 1
    GQA = Hq_tp // Hkv_tp          # 8
    S_prior = K_cache.shape[2]
    num_h_tiles = H // PMAX         # 16

    half_d = d // 2                 # 64

    H_wo = Wo.shape[1]              # 2048
    num_h_blocks = H_wo // F_MAX    # 4

    assert S_prior % PMAX == 0, f"S_prior={S_prior} must be a multiple of {PMAX}"
    NUM_S_TILES  = S_prior // PMAX
    NUM_H_TILES  = 16
    HQ_TP_CONST  = 8
    NUM_H_BLOCKS = 4
    assert H == NUM_H_TILES * PMAX
    assert Hq_tp == HQ_TP_CONST

    # =========================================================================
    # K/V HBM outputs (shared_hbm — unchanged from v13_BC)
    # =========================================================================
    k_rope_out = nl.ndarray((B, d), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    v_out = nl.ndarray((B, d), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # =========================================================================
    # COLUMN LAYOUT RESHAPES
    # =========================================================================
    hidden_col = hidden_states.reshape((H, B))
    cos_col = cos.reshape((PMAX, B))
    sin_col = sin.reshape((PMAX, B))

    # =========================================================================
    # OPEN attn_outer SCOPE (NOT closed before return — caller closes it)
    # =========================================================================
    sbm.open_scope("attn_outer")

    # =========================================================================
    # LOAD CONSTANTS (allocated at attn_outer level)
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

    # =========================================================================
    # HIDDEN TILE HOISTING (attn_outer level)
    # =========================================================================
    h_all = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="h_all")
    nisa.dma_copy(
        dst=h_all,
        src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )

    # =========================================================================
    # Pre-load all 8 Wq head tiles (attn_outer level)
    # =========================================================================
    wq_heads = []
    for q_h in nl.affine_range(HQ_TP_CONST):
        wq_head_tile = sbm.alloc_stack((PMAX, H), nl.bfloat16, name=f"wq_head_early_{q_h}")
        nisa.dma_copy(dst=wq_head_tile, src=Wq[q_h * PMAX:(q_h + 1) * PMAX, :], dge_mode=nisa.dge_mode.hwdge)
        wq_heads.append(wq_head_tile)

    # Wo reshape
    Wo_reshaped = Wo.reshape((Hq_tp, d, H_wo))

    # =========================================================================
    # Allocate persistent tensors at attn_outer level
    # (survive into flash decode)
    # =========================================================================
    k_rope = sbm.alloc_stack((PMAX, B), nl.float32, name="k_rope")
    v_active = sbm.alloc_stack((PMAX, B), nl.float32, name="v_active")
    q_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_bf16")

    # wo_sbuf tiles (attn_outer level)
    wo_sbuf = []
    for head in nl.affine_range(HQ_TP_CONST):
        wo_tile = sbm.alloc_stack((PMAX, H_wo), nl.bfloat16, name=f"wo_tile_h{head}")
        wo_sbuf.append(wo_tile)

    # =========================================================================
    # KV PROJ SCOPE
    # =========================================================================
    sbm.open_scope("kv_proj")

    # K projection
    wk_full = sbm.alloc_stack((PMAX, H), nl.bfloat16, name="wk_full")
    nisa.dma_copy(dst=wk_full, src=Wk, dge_mode=nisa.dge_mode.hwdge)

    wv_full = sbm.alloc_stack((PMAX, H), nl.bfloat16, name="wv_full")
    nisa.dma_copy(dst=wv_full, src=Wv, dge_mode=nisa.dge_mode.hwdge)

    k_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="k_psum")
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(k_psum, stationary=wk_full[0:PMAX, h_t*PMAX:(h_t+1)*PMAX], moving=h_all[0:PMAX, h_t:h_t+1])

    k_vec = sbm.alloc_stack((PMAX, B), nl.float32, name="k_vec")
    nisa.tensor_copy(k_vec, k_psum)

    # K RMSNorm
    k_sq = sbm.alloc_stack((PMAX, B), nl.float32, name="k_sq")
    nisa.tensor_tensor(k_sq, k_vec, k_vec, op=nl.multiply)
    k_sq_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_sq_bf16")
    nisa.tensor_copy(k_sq_bf16, k_sq)
    k_sum_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="k_sum_psum")
    nisa.nc_matmul(k_sum_psum, stationary=rms_ones, moving=k_sq_bf16)
    k_sum_sb = sbm.alloc_stack((PMAX, B), nl.float32, name="k_sum_sb")
    nisa.tensor_copy(k_sum_sb, k_sum_psum)
    k_mean_sq = sbm.alloc_stack((PMAX, B), nl.float32, name="k_mean_sq")
    nisa.tensor_scalar(k_mean_sq, k_sum_sb, op0=nl.multiply, operand0=1.0/d, op1=nl.add, operand1=EPS)
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
    # Fill k_rope (outer alloc)
    nisa.tensor_tensor(k_rope, k_cos, k_sin_part, op=nl.add)

    # DMA k_rope → k_rope_out HBM
    k_rope_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_rope_bf16")
    nisa.tensor_copy(k_rope_bf16, k_rope)
    k_rope_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name="k_rope_T_psum")
    nisa.nc_transpose(k_rope_T_psum, k_rope_bf16)
    k_rope_T_sb = sbm.alloc_stack((B, PMAX), nl.bfloat16, name="k_rope_T_sb")
    nisa.tensor_copy(k_rope_T_sb, k_rope_T_psum)
    nisa.dma_copy(dst=k_rope_out, src=k_rope_T_sb, dge_mode=nisa.dge_mode.hwdge)

    # V projection
    v_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="v_psum")
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(v_psum, stationary=wv_full[0:PMAX, h_t*PMAX:(h_t+1)*PMAX], moving=h_all[0:PMAX, h_t:h_t+1])

    # Fill v_active (outer alloc)
    nisa.tensor_copy(v_active, v_psum)

    # DMA v_active → v_out HBM
    v_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="v_bf16")
    nisa.tensor_copy(v_bf16, v_active)
    v_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name="v_T_psum")
    nisa.nc_transpose(v_T_psum, v_bf16)
    v_T_sb = sbm.alloc_stack((B, PMAX), nl.bfloat16, name="v_T_sb")
    nisa.tensor_copy(v_T_sb, v_T_psum)
    nisa.dma_copy(dst=v_out, src=v_T_sb, dge_mode=nisa.dge_mode.hwdge)

    sbm.close_scope()  # kv_proj

    # =========================================================================
    # Q PROJ SCOPE
    # =========================================================================
    sbm.open_scope("q_proj")

    q_packed_f32 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_packed_f32")
    for q_h in nl.affine_range(HQ_TP_CONST):
        q_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name=f"q_psum_{q_h}")
        for h_t in nl.affine_range(NUM_H_TILES):
            nisa.nc_matmul(
                q_psum,
                stationary=wq_heads[q_h][0:PMAX, h_t * PMAX:(h_t + 1) * PMAX],
                moving=h_all[0:PMAX, h_t:h_t + 1],
            )
        nisa.tensor_copy(q_packed_f32[0:PMAX, q_h:q_h + 1], q_psum)

    # Q RMSNorm
    q_sq = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_sq")
    nisa.tensor_tensor(q_sq, q_packed_f32, q_packed_f32, op=nl.multiply)
    q_sq_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_sq_bf16")
    nisa.tensor_copy(q_sq_bf16, q_sq)
    q_sum_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="q_sum_psum")
    nisa.nc_matmul(q_sum_psum, stationary=rms_ones, moving=q_sq_bf16)
    q_sum_sb = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_sum_sb")
    nisa.tensor_copy(q_sum_sb, q_sum_psum)
    q_mean_sq = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_mean_sq")
    nisa.tensor_scalar(q_mean_sq, q_sum_sb, op0=nl.multiply, operand0=1.0/d, op1=nl.add, operand1=EPS)
    q_rms_inv = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_rms_inv")
    nisa.activation(q_rms_inv, op=nl.rsqrt, data=q_mean_sq)
    q_normed = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_normed")
    nisa.tensor_tensor(q_normed, q_packed_f32, q_rms_inv, op=nl.multiply)

    # Apply norm weight via tp_broadcast
    qnw_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="qnw_gqa_psum_T")
    nisa.nc_transpose(qnw_gqa_psum_T, qnw_sb.ap([[1, PMAX], [0, GQA]], offset=0))
    qnw_gqa_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="qnw_gqa_sbuf_T")
    nisa.tensor_copy(qnw_gqa_sbuf_T, qnw_gqa_psum_T)
    qnw_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="qnw_gqa_psum")
    nisa.nc_transpose(qnw_gqa_psum, qnw_gqa_sbuf_T)
    qnw_gqa = sbm.alloc_stack((PMAX, GQA), nl.float32, name="qnw_gqa")
    nisa.tensor_copy(qnw_gqa, qnw_gqa_psum)

    q_normed2 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_normed2")
    nisa.tensor_tensor(q_normed2, q_normed, qnw_gqa, op=nl.multiply)

    # Q RoPE — tp_broadcast cos/sin
    cos_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="cos_gqa_psum_T")
    nisa.nc_transpose(cos_gqa_psum_T, cos_f32.ap([[1, PMAX], [0, GQA]], offset=0))
    cos_gqa_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="cos_gqa_sbuf_T")
    nisa.tensor_copy(cos_gqa_sbuf_T, cos_gqa_psum_T)
    cos_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="cos_gqa_psum")
    nisa.nc_transpose(cos_gqa_psum, cos_gqa_sbuf_T)
    cos_gqa = sbm.alloc_stack((PMAX, GQA), nl.float32, name="cos_gqa")
    nisa.tensor_copy(cos_gqa, cos_gqa_psum)

    sin_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="sin_gqa_psum_T")
    nisa.nc_transpose(sin_gqa_psum_T, sin_f32.ap([[1, PMAX], [0, GQA]], offset=0))
    sin_gqa_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="sin_gqa_sbuf_T")
    nisa.tensor_copy(sin_gqa_sbuf_T, sin_gqa_psum_T)
    sin_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="sin_gqa_psum")
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

    # Scale + cast to bf16 → fills q_bf16 (outer alloc)
    nisa.tensor_scalar(q_bf16, q_rope, op0=nl.multiply, operand0=INV_SQRT_D)

    sbm.close_scope()  # q_proj

    # =========================================================================
    # WO WEIGHT HOISTING — DMA issued at attn_outer level (just before flash decode)
    # =========================================================================
    for head in nl.affine_range(HQ_TP_CONST):
        nisa.dma_copy(
            dst=wo_sbuf[head],
            src=Wo_reshaped.ap(
                pattern=[[H_wo, PMAX], [1, H_wo]],
                offset=head * PMAX * H_wo,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )

    # =========================================================================
    # TWO-PASS FLASH DECODE
    # =========================================================================
    K_cache_2d = K_cache.reshape((S_prior, d))
    V_cache_2d = V_cache.reshape((S_prior, d))

    # Active position score broadcast: k_rope [PMAX,1] → [PMAX,GQA]
    k_rope_packed_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="k_rope_packed_psum_T")
    nisa.nc_transpose(k_rope_packed_psum_T, k_rope.ap([[1, PMAX], [0, GQA]], offset=0))
    k_rope_packed_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="k_rope_packed_sbuf_T")
    nisa.tensor_copy(k_rope_packed_sbuf_T, k_rope_packed_psum_T)
    k_rope_packed_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="k_rope_packed_psum")
    nisa.nc_transpose(k_rope_packed_psum, k_rope_packed_sbuf_T)
    k_rope_packed = sbm.alloc_stack((PMAX, GQA), nl.float32, name="k_rope_packed")
    nisa.tensor_copy(k_rope_packed, k_rope_packed_psum)

    kq_elem = sbm.alloc_stack((PMAX, GQA), nl.float32, name="kq_elem")
    nisa.tensor_tensor(kq_elem, k_rope_packed, q_bf16, op=nl.multiply)
    kq_elem_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="kq_elem_bf16")
    nisa.tensor_copy(kq_elem_bf16, kq_elem)
    score_active_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="score_active_psum")
    nisa.nc_matmul(score_active_psum, stationary=rms_ones, moving=kq_elem_bf16)
    score_active = sbm.alloc_stack((PMAX, GQA), nl.float32, name="score_active")
    nisa.tensor_copy(score_active, score_active_psum)

    # Load position
    position_ids_2d = position_ids.reshape((B, 1))
    pos_sb = sbm.alloc_stack((1, 1), nl.int32, name="pos_sb")
    nisa.dma_copy(dst=pos_sb, src=position_ids_2d[0:1, 0:1], dge_mode=nisa.dge_mode.hwdge)
    pos_f32 = sbm.alloc_stack((1, 1), nl.float32, name="pos_f32")
    nisa.tensor_copy(pos_f32, pos_sb)

    par_index_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name="par_index_f32")
    nisa.iota(par_index_f32, pattern=[[1, 1]], offset=0, channel_multiplier=1)

    # =========================================================================
    # KV HOIST SCOPE (kept open through Pass 1 AND Pass 2)
    # =========================================================================
    sbm.open_scope("kv_hoist")

    k_cache_tiles = []
    mask_gqa_tiles = []
    v_cache_tiles = []

    for s_t in nl.affine_range(NUM_S_TILES):
        # Allocate k_cache_tiles[s_t] at kv_hoist level BEFORE inner scope
        k_ct = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name=f"k_ct_{s_t}")

        # Pre-broadcast mask at kv_hoist level BEFORE inner scope
        mask_gqa_pre = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"mask_gqa_pre_{s_t}")

        sbm.open_scope(f"tile_{s_t}")

        # Load K tile
        k_raw = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name=f"k_raw_{s_t}")
        nisa.dma_copy(dst=k_raw, src=K_cache_2d[s_t * PMAX:(s_t + 1) * PMAX, :], dge_mode=nisa.dge_mode.hwdge)

        # PE transpose
        k_ct_psum = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name=f"k_ct_psum_{s_t}")
        nisa.nc_transpose(k_ct_psum, k_raw)
        nisa.tensor_copy(k_ct, k_ct_psum)

        # Position-id threshold mask
        tile_start = s_t * PMAX
        neg_threshold = sbm.alloc_stack((1, 1), nl.float32, name=f"neg_threshold_{s_t}")
        nisa.tensor_scalar(neg_threshold, pos_f32,
                           op0=nl.multiply, operand0=-1.0,
                           op1=nl.add, operand1=float(tile_start))

        neg_thresh_psum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum, name=f"neg_thresh_psum_{s_t}")
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

        # Pre-broadcast mask [PMAX,1] → [PMAX,GQA] via double transpose
        mask_gqa_pre_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name=f"mask_gqa_pre_psum_T_{s_t}")
        nisa.nc_transpose(mask_gqa_pre_psum_T, mask_tile_f32.ap([[1, PMAX], [0, GQA]], offset=0))
        mask_gqa_pre_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name=f"mask_gqa_pre_sbuf_T_{s_t}")
        nisa.tensor_copy(mask_gqa_pre_sbuf_T, mask_gqa_pre_psum_T)
        mask_gqa_pre_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"mask_gqa_pre_psum_{s_t}")
        nisa.nc_transpose(mask_gqa_pre_psum, mask_gqa_pre_sbuf_T)
        nisa.tensor_copy(mask_gqa_pre, mask_gqa_pre_psum)

        sbm.close_scope()  # tile_{s_t}

        k_cache_tiles.append(k_ct)
        mask_gqa_tiles.append(mask_gqa_pre)

        # V cache tile at kv_hoist level (after inner scope)
        v_ct = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name=f"v_ct_{s_t}")
        nisa.dma_copy(
            dst=v_ct,
            src=V_cache_2d.ap(pattern=[[d, PMAX], [1, d]], offset=s_t * PMAX * d),
            dge_mode=nisa.dge_mode.hwdge,
        )
        v_cache_tiles.append(v_ct)

    # =========================================================================
    # PASS 1 SCOPE (active level — also holds attn_out and saved_scores)
    # =========================================================================
    sbm.open_scope("pass1_active")

    global_max_g1 = sbm.alloc_stack((GQA, 1), nl.float32, name="global_max_g1")
    nisa.memset(global_max_g1, value=-1e9)

    score_act_T_psum = nl.zeros((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="score_act_T_psum")
    nisa.nc_transpose(score_act_T_psum, score_active)
    score_act_T_sb = sbm.alloc_stack((GQA, PMAX), nl.float32, name="score_act_T_sb")
    nisa.tensor_copy(score_act_T_sb, score_act_T_psum)
    score_active_g1 = sbm.alloc_stack((GQA, 1), nl.float32, name="score_active_g1")
    nisa.tensor_reduce(dst=score_active_g1, op=nl.maximum, data=score_act_T_sb, axis=1)
    nisa.tensor_tensor(global_max_g1, global_max_g1, score_active_g1, op=nl.maximum)

    # saved_scores allocated at pass1_active level (live for Pass 2)
    saved_scores = []

    for s_t in nl.affine_range(NUM_S_TILES):
        # Allocate saved_scores[s_t] at pass1_active level BEFORE inner scope
        score_sb_masked = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"score_sb_masked_{s_t}")

        sbm.open_scope(f"p1_tile_{s_t}")

        score_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"score_psum_{s_t}")
        nisa.nc_matmul(score_psum, stationary=k_cache_tiles[s_t], moving=q_bf16)
        score_sb = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"score_sb_{s_t}")
        nisa.tensor_copy(score_sb, score_psum)

        mask_gqa = mask_gqa_tiles[s_t]
        nisa.tensor_tensor(score_sb_masked, score_sb, mask_gqa, op=nl.add)

        score_T_psum = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name=f"score_T_psum_{s_t}")
        nisa.nc_transpose(score_T_psum, score_sb_masked)
        score_T_sb = sbm.alloc_stack((GQA, PMAX), nl.float32, name=f"score_T_sb_{s_t}")
        nisa.tensor_copy(score_T_sb, score_T_psum)

        tile_max_vec = sbm.alloc_stack((GQA, 1), nl.float32, name=f"tile_max_vec_{s_t}")
        nisa.tensor_reduce(dst=tile_max_vec, op=nl.maximum, data=score_T_sb, axis=1)
        nisa.tensor_tensor(global_max_g1, global_max_g1, tile_max_vec, op=nl.maximum)

        sbm.close_scope()  # p1_tile_{s_t}

        saved_scores.append(score_sb_masked)

    # Negate global max and broadcast
    neg_max_g1 = sbm.alloc_stack((GQA, 1), nl.float32, name="neg_max_g1")
    nisa.tensor_scalar(neg_max_g1, global_max_g1, op0=nl.multiply, operand0=-1.0)

    neg_max_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="neg_max_psum")
    nisa.nc_transpose(
        neg_max_psum,
        neg_max_g1.ap([[1, GQA], [0, PMAX]], offset=0),
    )
    neg_max = sbm.alloc_stack((PMAX, GQA), nl.float32, name="neg_max")
    nisa.tensor_copy(neg_max, neg_max_psum)

    # attn_out allocated at pass1_active level (used in o_proj)
    attn_out = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="attn_out")

    # =========================================================================
    # PASS 2 SCOPE
    # =========================================================================
    sbm.open_scope("pass2")

    v_acc = sbm.alloc_stack((PMAX, GQA), nl.float32, name="v_acc")
    nisa.memset(v_acc, value=0.0)
    sum_acc = sbm.alloc_stack((PMAX, GQA), nl.float32, name="sum_acc")
    nisa.memset(sum_acc, value=0.0)

    for s_t in nl.affine_range(NUM_S_TILES):
        score2_shifted = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"score2_shifted_{s_t}")
        nisa.tensor_tensor(score2_shifted, saved_scores[s_t], neg_max, op=nl.add)

        score2_exp = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"score2_exp_{s_t}")
        nisa.activation(score2_exp, op=nl.exp, data=score2_shifted)

        score2_exp_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name=f"score2_exp_bf16_{s_t}")
        nisa.tensor_copy(score2_exp_bf16, score2_exp)

        tile_sum_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"tile_sum_psum_{s_t}")
        nisa.nc_matmul(tile_sum_psum, stationary=rms_ones, moving=score2_exp_bf16)
        tile_sum = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"tile_sum_{s_t}")
        nisa.tensor_copy(tile_sum, tile_sum_psum)
        nisa.tensor_tensor(sum_acc, sum_acc, tile_sum, op=nl.add)

        v_weighted_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"v_weighted_psum_{s_t}")
        nisa.nc_matmul(v_weighted_psum, stationary=v_cache_tiles[s_t], moving=score2_exp_bf16)
        v_weighted = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"v_weighted_{s_t}")
        nisa.tensor_copy(v_weighted, v_weighted_psum)
        nisa.tensor_tensor(v_acc, v_acc, v_weighted, op=nl.add)

    # Active position contribution
    score_act_shifted = sbm.alloc_stack((PMAX, GQA), nl.float32, name="score_act_shifted")
    nisa.tensor_tensor(score_act_shifted, score_active, neg_max, op=nl.add)
    score_act_exp = sbm.alloc_stack((PMAX, GQA), nl.float32, name="score_act_exp")
    nisa.activation(score_act_exp, op=nl.exp, data=score_act_shifted)
    nisa.tensor_tensor(sum_acc, sum_acc, score_act_exp, op=nl.add)

    # v_active broadcast: [PMAX,1] → [PMAX,GQA]
    v_act_packed_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="v_act_packed_psum_T")
    nisa.nc_transpose(v_act_packed_psum_T, v_active.ap([[1, PMAX], [0, GQA]], offset=0))
    v_act_packed_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="v_act_packed_sbuf_T")
    nisa.tensor_copy(v_act_packed_sbuf_T, v_act_packed_psum_T)
    v_act_packed_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="v_act_packed_psum")
    nisa.nc_transpose(v_act_packed_psum, v_act_packed_sbuf_T)
    v_act_packed = sbm.alloc_stack((PMAX, GQA), nl.float32, name="v_act_packed")
    nisa.tensor_copy(v_act_packed, v_act_packed_psum)

    v_act_weighted = sbm.alloc_stack((PMAX, GQA), nl.float32, name="v_act_weighted")
    nisa.tensor_tensor(v_act_weighted, v_act_packed, score_act_exp, op=nl.multiply)
    nisa.tensor_tensor(v_acc, v_acc, v_act_weighted, op=nl.add)

    # Normalize
    sum_safe = sbm.alloc_stack((PMAX, GQA), nl.float32, name="sum_safe")
    nisa.tensor_scalar(sum_safe, sum_acc, op0=nl.add, operand0=1e-9)
    rsqrt_sum = sbm.alloc_stack((PMAX, GQA), nl.float32, name="rsqrt_sum")
    nisa.activation(rsqrt_sum, op=nl.rsqrt, data=sum_safe)
    inv_sum = sbm.alloc_stack((PMAX, GQA), nl.float32, name="inv_sum")
    nisa.tensor_tensor(inv_sum, rsqrt_sum, rsqrt_sum, op=nl.multiply)

    # Fill attn_out (pass1_active alloc)
    nisa.tensor_tensor(attn_out, v_acc, inv_sum, op=nl.multiply)

    sbm.close_scope()  # pass2

    # =========================================================================
    # O_PROJ SCOPE
    # =========================================================================
    sbm.open_scope("o_proj")

    # Allocate or use caller-provided out_sb
    if out_sb is None:
        out_sb = sbm.alloc_stack((1, H_wo), nl.bfloat16, name="out_sb")

    out_sb_tmp = sbm.alloc_stack((1, F_MAX), nl.bfloat16, name="out_sb_tmp")

    # Pre-allocate all 4 output PSUMs
    res_psum_0 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_0")
    res_psum_1 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_1")
    res_psum_2 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_2")
    res_psum_3 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_3")

    for head in nl.affine_range(HQ_TP_CONST):
        nisa.nc_matmul(res_psum_0, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 0*F_MAX:1*F_MAX])
        nisa.nc_matmul(res_psum_1, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 1*F_MAX:2*F_MAX])
        nisa.nc_matmul(res_psum_2, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 2*F_MAX:3*F_MAX])
        nisa.nc_matmul(res_psum_3, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 3*F_MAX:4*F_MAX])

    # Copy each PSUM block → out_sb_tmp → out_sb slice (no DMA to HBM)
    nisa.tensor_copy(out_sb_tmp, res_psum_0)
    nisa.tensor_copy(out_sb[0:1, 0*F_MAX:1*F_MAX], out_sb_tmp[0:1, 0:F_MAX])

    nisa.tensor_copy(out_sb_tmp, res_psum_1)
    nisa.tensor_copy(out_sb[0:1, 1*F_MAX:2*F_MAX], out_sb_tmp[0:1, 0:F_MAX])

    nisa.tensor_copy(out_sb_tmp, res_psum_2)
    nisa.tensor_copy(out_sb[0:1, 2*F_MAX:3*F_MAX], out_sb_tmp[0:1, 0:F_MAX])

    nisa.tensor_copy(out_sb_tmp, res_psum_3)
    nisa.tensor_copy(out_sb[0:1, 3*F_MAX:4*F_MAX], out_sb_tmp[0:1, 0:F_MAX])

    sbm.close_scope()  # o_proj

    sbm.close_scope()  # pass1_active

    sbm.close_scope()  # kv_hoist

    # attn_outer NOT closed — caller is responsible
    return out_sb, k_rope_out, v_out
