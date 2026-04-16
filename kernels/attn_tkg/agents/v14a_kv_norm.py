"""
v13bc_kv_norm: Modified v13bc_sbm_tiled with two key changes:
  1. Accepts hidden_sb (SBUF tensor) instead of hidden_states (HBM), and applies
     input_layernorm (pre_attn_norm) inside the kernel using gamma_pre_attn.
  2. Writes K/V directly into K_cache/V_cache in-place via scatter DMA
     (using pos_write_i32 as scatter index), instead of allocating shared_hbm outputs.

Returns only out_sb (no k_rope_out / v_out).
"""

import math
import nki.language as nl
import nki.isa as nisa

PMAX = 128
F_MAX = 512
EPS = 1e-6
INV_SQRT_D = float(1.0 / math.sqrt(128.0))


def qwen3_attn_tkg_fused_oproj_v13bc_kv_norm(
    hidden_sb,        # [PMAX, num_h_tiles] = [128, 16]  bf16  SBUF — raw residual, column-major
    Wq,               # [Hq_tp*d, H]  = [1024, 2048]    bf16  HBM
    Wk,               # [Hkv_tp*d, H] = [128, 2048]     bf16  HBM
    Wv,               # [Hkv_tp*d, H] = [128, 2048]     bf16  HBM
    Wo,               # [Hq_tp*d, H_wo]= [1024, 2048]   bf16  HBM
    q_norm_weight,    # [d]           = [128]            bf16  HBM
    k_norm_weight,    # [d]           = [128]            bf16  HBM
    gamma_pre_attn,   # [H]           = [2048]           bf16  HBM  — input_layernorm.weight  NEW
    K_cache,          # [B, 1, S_prior, d]               bf16  HBM  mutated in-place
    V_cache,          # [B, 1, S_prior, d]               bf16  HBM  mutated in-place
    cos,              # [B, d]        = [1, 128]         bf16  HBM  (pre-indexed at position_ids)
    sin,              # [B, d]        = [1, 128]         bf16  HBM
    position_ids,     # [B, 1]        = [1, 1]           int32 HBM
    out_sb=None,      # optional [PMAX, H_wo//PMAX]      bf16  SBUF caller-provided
    sbm=None,         # required SbufManager
):
    """
    Fused pre-attn norm + QKV + RMSNorm + RoPE + KV cache scatter + flash decode + output projection.
    Sub-function — no @nki.jit. Called from inside a JIT kernel that owns sbm.

    Returns out_sb:
      out_sb:     [PMAX, H_wo//PMAX] = [128, 16] bf16 in SBUF — column-major layout.
                  Element mapping: out_sb[p, j] = linear_out[j * 128 + p]
                  Valid until caller closes attn_outer scope.

    K_cache and V_cache are updated in-place at position position_ids.

    IMPORTANT: Both LNC cores produce the full [128, 16] output — no program_id sharding.
    Caller must consume out_sb before closing the scope that opened attn_outer.
    """
    assert sbm is not None, "sbm (SbufManager) is required"

    # --- Dimensions ---
    B = cos.shape[0]    # 1
    H = Wq.shape[1]     # 2048
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
    # COLUMN LAYOUT RESHAPES
    # =========================================================================
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

    # gamma_pre_attn in [PMAX, num_h_tiles] tile layout — same column-major layout as h_all
    # gamma_pre_attn[H] → gpan_bf16[p, t] = gamma[t*PMAX + p]
    gpan_bf16 = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="gpan_bf16")
    nisa.dma_copy(dst=gpan_bf16,
                  src=gamma_pre_attn.reshape((H, 1)).ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge)
    gpan_f32 = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="gpan_f32")
    nisa.tensor_copy(gpan_f32, gpan_bf16)

    # write position for KV scatter (also reused for flash-decode mask)
    # int32 matches position_ids dtype; used as scalar_offset for scatter DMA
    # and converted to float32 for the causal mask computation
    pos_write_i32_raw = sbm.alloc_stack((1, 1), nl.int32, name="pos_write_i32_raw")
    nisa.dma_copy(dst=pos_write_i32_raw,
                  src=position_ids.reshape((B, 1))[0:1, 0:1],
                  dge_mode=nisa.dge_mode.hwdge)
    pos_write_i32 = sbm.alloc_stack((1, 1), nl.uint32, name="pos_write_i32")
    nisa.tensor_copy(pos_write_i32, pos_write_i32_raw)

    # =========================================================================
    # h_all: allocate shell, filled by pre_attn_norm scope below
    # =========================================================================
    h_all = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="h_all")
    # filled by pre_attn_norm scope below

    # =====================================================================
    # PRE-ATTENTION RMSNORM — input_layernorm(hidden_sb) → h_all
    # =====================================================================
    sbm.open_scope("pre_attn_norm")

    h_f32 = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_f32")
    nisa.tensor_copy(h_f32, hidden_sb)   # read raw residual from SBUF

    h_sq = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_sq")
    nisa.tensor_tensor(h_sq, h_f32, h_f32, op=nl.multiply)

    # Sum x² along free dim (16 tiles) → [PMAX, 1]
    h_sq_sum = sbm.alloc_stack((PMAX, 1), nl.float32, name="h_sq_sum")
    nisa.tensor_reduce(dst=h_sq_sum, op=nl.add, data=h_sq, axis=1)

    # Sum across 128 partitions → total Σx², replicated for each partition
    h_sq_sum_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="h_sq_sum_bf16")
    nisa.tensor_copy(h_sq_sum_bf16, h_sq_sum)
    h_sq_total_psum = nl.zeros((PMAX, 1), nl.float32, buffer=nl.psum)
    nisa.nc_matmul(h_sq_total_psum, stationary=rms_ones, moving=h_sq_sum_bf16)
    h_sq_total = sbm.alloc_stack((PMAX, 1), nl.float32, name="h_sq_total")
    nisa.tensor_copy(h_sq_total, h_sq_total_psum)

    # mean(x²) + eps → rsqrt
    h_mean_sq = sbm.alloc_stack((PMAX, 1), nl.float32, name="h_mean_sq")
    nisa.tensor_scalar(h_mean_sq, h_sq_total,
                       op0=nl.multiply, operand0=1.0/H,
                       op1=nl.add,      operand1=EPS)
    h_rms_inv = sbm.alloc_stack((PMAX, 1), nl.float32, name="h_rms_inv")
    nisa.activation(h_rms_inv, op=nl.rsqrt, data=h_mean_sq)

    # Broadcast h_rms_inv [PMAX,1] → [PMAX, num_h_tiles] via stride-0 ap + double transpose
    h_rms_T_psum = nl.ndarray((num_h_tiles, PMAX), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_rms_T_psum, h_rms_inv.ap([[1, PMAX], [0, num_h_tiles]], offset=0))
    h_rms_T = sbm.alloc_stack((num_h_tiles, PMAX), nl.float32, name="h_rms_T")
    nisa.tensor_copy(h_rms_T, h_rms_T_psum)
    h_rms_expanded_psum = nl.ndarray((PMAX, num_h_tiles), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_rms_expanded_psum, h_rms_T)
    h_rms_expanded = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_rms_expanded")
    nisa.tensor_copy(h_rms_expanded, h_rms_expanded_psum)

    # h_normed = h_f32 * h_rms_expanded * gpan_f32
    h_normed = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_normed")
    nisa.tensor_tensor(h_normed, h_f32, h_rms_expanded, op=nl.multiply)
    nisa.tensor_tensor(h_normed, h_normed, gpan_f32, op=nl.multiply)

    # Write into h_all (attn_outer alloc) — f32→bf16 downcast
    nisa.tensor_copy(h_all, h_normed)

    sbm.close_scope()  # pre_attn_norm

    # Pre-load Wq per-head via ONE dma_transpose per head (consolidates 16 strided DMAs per head).
    # Layout: wq_head_sb[q_h][p, h_t*d + f_d] = Wq[q_h*d + f_d, h_t*PMAX + p]
    # Source (per head q_h): Wq[q_h*d:(q_h+1)*d, 0:H] is [d=128, H=2048] contiguous (256 KB).
    # dma_transpose swaps {partition} and {inner-free}; middle free dim is preserved.
    # Slicing wq_head_sb[q_h][0:PMAX, h_t*d:(h_t+1)*d] yields the same [PMAX, PMAX] tile as before.
    wq_head_sb = []
    for q_h in nl.affine_range(HQ_TP_CONST):
        w = sbm.alloc_stack((PMAX, NUM_H_TILES * d), nl.bfloat16, name=f"wq_head_{q_h}")
        nisa.dma_transpose(
            dst=w.ap(
                pattern=[
                    [NUM_H_TILES * d, PMAX],  # partition: stride=free-size, size=PMAX (index p)
                    [1, 1],
                    [d, NUM_H_TILES],          # free middle: stride=d, size=num_h_tiles (index h_t)
                    [1, d],                     # free inner: stride=1, size=d (index f_d)
                ],
                offset=0,
            ),
            src=Wq.ap(
                pattern=[
                    [H, d],                     # partition: stride=H, size=d (index f_d → dst free-inner)
                    [1, 1],
                    [PMAX, NUM_H_TILES],        # free middle: stride=PMAX, size=num_h_tiles (index h_t, preserved)
                    [1, PMAX],                  # free inner: stride=1, size=PMAX (index p → dst partition)
                ],
                offset=q_h * d * H,
            ),
        )
        wq_head_sb.append(w)

    # Wo reshape
    Wo_reshaped = Wo.reshape((Hq_tp, d, H_wo))

    # =========================================================================
    # Allocate persistent tensors at attn_outer level
    # (survive into flash decode)
    # =========================================================================
    k_rope = sbm.alloc_stack((PMAX, B), nl.float32, name="k_rope")
    k_rope_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_rope_bf16")
    k_rope_T_sb = sbm.alloc_stack((B, PMAX), nl.bfloat16, name="k_rope_T_sb")
    v_active = sbm.alloc_stack((PMAX, B), nl.float32, name="v_active")
    v_T_sb = sbm.alloc_stack((B, PMAX), nl.bfloat16, name="v_T_sb")
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

    # K projection — ONE dma_transpose for all of Wk (consolidates 16 strided DMAs).
    # Layout: wk_sb[p, h_t*d + f_d] = Wk[f_d, h_t*PMAX + p]
    # Source Wk is [d=128, H=2048] contiguous (256 KB). Slicing [0:PMAX, h_t*d:(h_t+1)*d]
    # yields the same per-tile [PMAX, PMAX] stationary as before.
    wk_sb = sbm.alloc_stack((PMAX, NUM_H_TILES * d), nl.bfloat16, name="wk_sb")
    nisa.dma_transpose(
        dst=wk_sb.ap(
            pattern=[
                [NUM_H_TILES * d, PMAX],
                [1, 1],
                [d, NUM_H_TILES],
                [1, d],
            ],
            offset=0,
        ),
        src=Wk.ap(
            pattern=[
                [H, d],
                [1, 1],
                [PMAX, NUM_H_TILES],
                [1, PMAX],
            ],
            offset=0,
        ),
    )

    k_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum)
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(k_psum, stationary=wk_sb[0:PMAX, h_t*d:(h_t+1)*d], moving=h_all[0:PMAX, h_t:h_t+1])

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

    # Transpose k_rope to [B, PMAX] for KV cache scatter (k_rope_T_sb alloc'd at attn_outer)
    nisa.tensor_copy(k_rope_bf16, k_rope)
    k_rope_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(k_rope_T_psum, k_rope_bf16)
    nisa.tensor_copy(k_rope_T_sb, k_rope_T_psum)

    # V projection — ONE dma_transpose for all of Wv (consolidates 16 strided DMAs).
    # Same layout/semantics as Wk.
    wv_sb = sbm.alloc_stack((PMAX, NUM_H_TILES * d), nl.bfloat16, name="wv_sb")
    nisa.dma_transpose(
        dst=wv_sb.ap(
            pattern=[
                [NUM_H_TILES * d, PMAX],
                [1, 1],
                [d, NUM_H_TILES],
                [1, d],
            ],
            offset=0,
        ),
        src=Wv.ap(
            pattern=[
                [H, d],
                [1, 1],
                [PMAX, NUM_H_TILES],
                [1, PMAX],
            ],
            offset=0,
        ),
    )

    v_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum)
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(v_psum, stationary=wv_sb[0:PMAX, h_t*d:(h_t+1)*d], moving=h_all[0:PMAX, h_t:h_t+1])

    # Fill v_active (outer alloc)
    nisa.tensor_copy(v_active, v_psum)

    # Transpose v_active to [B, PMAX] for KV cache scatter (v_T_sb alloc'd at attn_outer)
    v_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="v_bf16")
    nisa.tensor_copy(v_bf16, v_active)
    v_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(v_T_psum, v_bf16)
    nisa.tensor_copy(v_T_sb, v_T_psum)

    sbm.close_scope()  # kv_proj

    # =========================================================================
    # Q PROJ SCOPE
    # =========================================================================
    sbm.open_scope("q_proj")

    # Q projection — use consolidated per-head wq_head_sb (slice per h_t)
    q_packed_f32 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_packed_f32")
    for q_h in nl.affine_range(HQ_TP_CONST):
        q_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum)
        for h_t in nl.affine_range(NUM_H_TILES):
            nisa.nc_matmul(q_psum,
                           stationary=wq_head_sb[q_h][0:PMAX, h_t*d:(h_t+1)*d],
                           moving=h_all[0:PMAX, h_t:h_t+1])
        nisa.tensor_copy(q_packed_f32[0:PMAX, q_h:q_h + 1], q_psum)

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
    nisa.tensor_scalar(q_mean_sq, q_sum_sb, op0=nl.multiply, operand0=1.0/d, op1=nl.add, operand1=EPS)
    q_rms_inv = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_rms_inv")
    nisa.activation(q_rms_inv, op=nl.rsqrt, data=q_mean_sq)
    q_normed = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_normed")
    nisa.tensor_tensor(q_normed, q_packed_f32, q_rms_inv, op=nl.multiply)

    # Apply norm weight via tp_broadcast
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

    # Q RoPE — tp_broadcast cos/sin
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

    # Active position score: broadcast k_rope [PMAX,1] → [PMAX,GQA], then element-wise * q_bf16,
    # then sum across partition dim via nc_matmul with rms_ones. Matches v13bc_sbm_tiled pattern.
    k_rope_packed_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(k_rope_packed_psum_T, k_rope.ap([[1, PMAX], [0, GQA]], offset=0))
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

    # Load position for causal mask computation (reuse pos_write_i32 loaded at attn_outer level)
    position_ids_2d = position_ids.reshape((B, 1))  # kept for reference
    pos_f32 = sbm.alloc_stack((1, 1), nl.float32, name="pos_f32")
    nisa.tensor_copy(pos_f32, pos_write_i32)  # int32 → float32

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
        k_ct_psum = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(k_ct_psum, k_raw)
        nisa.tensor_copy(k_ct, k_ct_psum)

        # Position-id threshold mask
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

        # mask positions j >= pos; the cache slot at pos has not been written yet
        # (new token contribution comes from active path). Shift by +1 so relu masks j >= pos.
        delta_eps = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"delta_eps_{s_t}")
        nisa.tensor_scalar(delta_eps, delta, op0=nl.add, operand0=1.0)
        relu_delta = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"relu_delta_{s_t}")
        nisa.activation(relu_delta, op=nl.relu, data=delta_eps)

        clamped = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"clamped_{s_t}")
        nisa.tensor_scalar(clamped, relu_delta, op0=nl.minimum, operand0=1.0)

        mask_tile_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"mask_tile_f32_{s_t}")
        nisa.tensor_scalar(mask_tile_f32, clamped, op0=nl.multiply, operand0=-1e9)

        # Pre-broadcast mask [PMAX,1] → [PMAX,GQA] via double transpose
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

    score_act_T_psum = nl.zeros((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
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

        score_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(score_psum, stationary=k_cache_tiles[s_t], moving=q_bf16)
        score_sb = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"score_sb_{s_t}")
        nisa.tensor_copy(score_sb, score_psum)

        mask_gqa = mask_gqa_tiles[s_t]
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

    # Negate global max and broadcast
    neg_max_g1 = sbm.alloc_stack((GQA, 1), nl.float32, name="neg_max_g1")
    nisa.tensor_scalar(neg_max_g1, global_max_g1, op0=nl.multiply, operand0=-1.0)

    neg_max_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
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

        tile_sum_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(tile_sum_psum, stationary=rms_ones, moving=score2_exp_bf16)
        tile_sum = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"tile_sum_{s_t}")
        nisa.tensor_copy(tile_sum, tile_sum_psum)
        nisa.tensor_tensor(sum_acc, sum_acc, tile_sum, op=nl.add)

        v_weighted_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(v_weighted_psum, stationary=v_cache_tiles[s_t], moving=score2_exp_bf16)
        v_weighted = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"v_weighted_{s_t}")
        nisa.tensor_copy(v_weighted, v_weighted_psum)
        nisa.tensor_tensor(v_acc, v_acc, v_weighted, op=nl.add)

    # Active position contribution (active token NOT in cache tiles — handled separately)
    score_act_shifted = sbm.alloc_stack((PMAX, GQA), nl.float32, name="score_act_shifted")
    nisa.tensor_tensor(score_act_shifted, score_active, neg_max, op=nl.add)
    score_act_exp = sbm.alloc_stack((PMAX, GQA), nl.float32, name="score_act_exp")
    nisa.activation(score_act_exp, op=nl.exp, data=score_act_shifted)
    nisa.tensor_tensor(sum_acc, sum_acc, score_act_exp, op=nl.add)

    # v_active broadcast: [PMAX,1] → [PMAX,GQA]
    v_act_packed_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(v_act_packed_psum_T, v_active.ap([[1, PMAX], [0, GQA]], offset=0))
    v_act_packed_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="v_act_packed_sbuf_T")
    nisa.tensor_copy(v_act_packed_sbuf_T, v_act_packed_psum_T)
    v_act_packed_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
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
    NUM_OUT_COLS = H_wo // PMAX      # 16
    CHUNKS_PER_BLOCK = F_MAX // PMAX  # 4

    if out_sb is None:
        out_sb = sbm.alloc_stack((PMAX, NUM_OUT_COLS), nl.bfloat16, name="out_sb")

    out_sb_tmp = sbm.alloc_stack((1, F_MAX), nl.bfloat16, name="out_sb_tmp")
    col_tmp = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="col_tmp")
    # Single PSUM buffer reused for all 16 transpositions (sequential, no overlap needed)
    chunk_T_psum = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.psum)

    # Pre-allocate all 4 output PSUMs
    res_psum_0 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum)
    res_psum_1 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum)
    res_psum_2 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum)
    res_psum_3 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum)

    for head in nl.affine_range(HQ_TP_CONST):
        nisa.nc_matmul(res_psum_0, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 0*F_MAX:1*F_MAX])
        nisa.nc_matmul(res_psum_1, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 1*F_MAX:2*F_MAX])
        nisa.nc_matmul(res_psum_2, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 2*F_MAX:3*F_MAX])
        nisa.nc_matmul(res_psum_3, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 3*F_MAX:4*F_MAX])

    # Transpose each 128-column chunk into a column of [128, 16]; no HBM roundtrip
    # Fully unrolled: 4 blocks x 4 chunks = 16 total columns
    # Block 0: res_psum_0 -> columns 0..3
    nisa.tensor_copy(out_sb_tmp, res_psum_0)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 0*PMAX:1*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 0:1], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 1*PMAX:2*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 1:2], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 2*PMAX:3*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 2:3], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 3*PMAX:4*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 3:4], col_tmp)
    # Block 1: res_psum_1 -> columns 4..7
    nisa.tensor_copy(out_sb_tmp, res_psum_1)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 0*PMAX:1*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 4:5], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 1*PMAX:2*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 5:6], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 2*PMAX:3*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 6:7], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 3*PMAX:4*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 7:8], col_tmp)
    # Block 2: res_psum_2 -> columns 8..11
    nisa.tensor_copy(out_sb_tmp, res_psum_2)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 0*PMAX:1*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 8:9], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 1*PMAX:2*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 9:10], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 2*PMAX:3*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 10:11], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 3*PMAX:4*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 11:12], col_tmp)
    # Block 3: res_psum_3 -> columns 12..15
    nisa.tensor_copy(out_sb_tmp, res_psum_3)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 0*PMAX:1*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 12:13], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 1*PMAX:2*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 13:14], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 2*PMAX:3*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 14:15], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 3*PMAX:4*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum)
    nisa.tensor_copy(out_sb[0:PMAX, 15:16], col_tmp)

    sbm.close_scope()  # o_proj

    sbm.close_scope()  # pass1_active

    sbm.close_scope()  # kv_hoist

    # =========================================================================
    # KV CACHE IN-PLACE SCATTER (after flash decode + o_proj)
    # LNC-split: V on core 0, K on core 1 — matches production attention_block_tkg pattern.
    # Both k_rope_T_sb and v_T_sb are allocated at attn_outer level (survive here).
    # =========================================================================
    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)

    if n_prgs == 1 or prg_id == 0:   # V on LNC=0
        nisa.dma_copy(
            dst=V_cache.reshape((B * S_prior, d)).ap(
                pattern=[[d, 1], [1, d]], offset=0,
                scalar_offset=pos_write_i32, indirect_dim=0,
            ),
            src=v_T_sb,
        )
    if n_prgs == 1 or prg_id == 1:   # K on LNC=1
        nisa.dma_copy(
            dst=K_cache.reshape((B * S_prior, d)).ap(
                pattern=[[d, 1], [1, d]], offset=0,
                scalar_offset=pos_write_i32, indirect_dim=0,
            ),
            src=k_rope_T_sb,
        )

    # attn_outer NOT closed — caller is responsible
    return out_sb
