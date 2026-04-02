"""
v13a: Extends v12d with LNC=2 optimization — Shard O-Projection by H_wo output columns.

Changes from v12d:
  LNC=2 O-projection sharding:
    - Both cores replicate all computation up through attn_out [128, 8].
    - Each core loads only half of the Wo weight (2MB instead of 4MB):
        Core 0: loads Wo[:, 0:1024],    writes output[0:1, 0:1024]
        Core 1: loads Wo[:, 1024:2048], writes output[0:1, 1024:2048]
    - Both cores write k_rope_out and v_out (idempotent — same value, 256 bytes each).
    - Kernel launched with qwen3_attn_tkg_fused_oproj_v13a[2](...).
    - 2 PSUMs instead of 4 in the O-projection (each core handles H_half=1024 wide).
    - Wo loading uses ap() with core_id * H_half offset to select the correct column shard.

All v12d optimizations preserved:
  Deferred Wo DMA (overlaps flash decode), Wq pre-load, early Wv DMA, Plan A static shape
  constants, Plan B K-cache contiguous load + PE transpose + saved_scores, Q-proj
  psum→q_packed_f32 direct copy, O-proj head-outer loop, all tp_broadcast patterns,
  Plan B saved scores, hidden tile hoisting, Wo contiguous ap() DMA pattern,
  v10e position_ids threshold masking.
"""

import math
import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128
F_MAX = 512
EPS = 1e-6
INV_SQRT_D = float(1.0 / math.sqrt(128.0))

@nki.jit
def qwen3_attn_tkg_fused_oproj_v13a(
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
    Fused QKV + RMSNorm + RoPE + flash decode + output projection.
    LNC=2: O-projection sharded across 2 cores by H_wo output columns.
    Attention mask is generated on-chip from position_ids threshold.
    Returns (output, k_rope_out, v_out):
      - output:     [B, 1, H_out] bf16, where H_out = H = 2048
      - k_rope_out: [d, B] = [128, 1] bf16 — new token's K after RMSNorm+RoPE
      - v_out:      [d, B] = [128, 1] bf16 — new token's V (no RoPE)

    Wo is passed as [Hq_out=1024, H_wo=2048] (caller transposes the weight).
    This enables contiguous DMA loading via nkilib-style ap() pattern.

    v13a changes (extends v12d):
      - core_id = nl.program_id(0): 0 or 1.
      - H_half = H_wo // 2 = 1024: each core handles half the output columns.
      - Wo loading: each core loads [PMAX, H_half] per head (2MB total instead of 4MB).
      - O-projection: 2 PSUMs (res_psum_0, res_psum_1) covering H_half columns.
      - Output stores: core 0 writes output[:, 0:H_half], core 1 writes output[:, H_half:H_wo].
      - k_rope_out and v_out: both cores write (idempotent — same value).
    """
    # --- LNC core identification ---
    core_id = nl.program_id(0)   # 0 or 1

    # --- Dimensions ---
    B = hidden_states.shape[0]      # 1
    H = hidden_states.shape[2]      # 2048
    Hq_out = Wq.shape[0]            # 1024  = Hq_tp * d
    d = PMAX                        # 128
    Hq_tp = Hq_out // d             # 8
    Hkv_tp = 1                      # per-rank KV heads (corrected)
    GQA = Hq_tp // Hkv_tp          # 8
    S_prior = K_cache.shape[2]
    num_h_tiles = H // PMAX         # 16
    num_s_tiles = S_prior // PMAX

    half_d = d // 2                 # 64

    # Output H: Wo is now [Hq_out=1024, H_wo=2048] (transposed), so H_wo is shape[1]
    H_wo = Wo.shape[1]              # 2048
    H_half = H_wo // 2              # 1024 — each core's shard width

    # =========================================================================
    # Plan A — Static shape constants
    # Replace runtime-derived loop bounds with compile-time integer constants.
    # =========================================================================
    assert S_prior % PMAX == 0, f"S_prior={S_prior} must be a multiple of {PMAX}"
    NUM_S_TILES  = S_prior // PMAX   # trace-time constant; 5 for S_prior=640
    NUM_H_TILES  = 16   # H=2048 / PMAX=128
    HQ_TP_CONST  = 8    # Hq_tp fixed for this shape
    NUM_H_BLOCKS = 2    # H_half=1024 / F_MAX=512  (per core, 2 blocks instead of 4)
    assert H == NUM_H_TILES * PMAX, f"H={H} must be {NUM_H_TILES*PMAX}"
    assert Hq_tp == HQ_TP_CONST, f"Hq_tp={Hq_tp} must be {HQ_TP_CONST}"

    # =========================================================================
    # COLUMN LAYOUT RESHAPES
    # =========================================================================
    # Output [B, 1, H_wo]: allocate in shared HBM (accessible by both cores)
    output = nl.ndarray((B, 1, H_wo), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # Reshape to [1, H_wo] for o_proj DMA stores.
    output_2d = output.reshape((1, H_wo))

    # K/V outputs for KV cache update
    # Use (B, d) = (1, 128) HBM shape so DMA can write in one contiguous packet
    # (partition=1, free=128) matching the known-working output DMA pattern.
    # Callers may view as (d, B) = (128, 1) via reshape.
    k_rope_out = nl.ndarray((B, d), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    v_out = nl.ndarray((B, d), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # Hidden: [B, 1, H] -> [H, B] column layout
    hidden_col = hidden_states.reshape((H, B))
    # cos/sin: [B, d] -> [PMAX, B]
    cos_col = cos.reshape((PMAX, B))
    sin_col = sin.reshape((PMAX, B))

    # =========================================================================
    # LOAD CONSTANTS
    # =========================================================================
    # Norm weights [128] -> [PMAX, 1] f32 in SBUF
    qnw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_bf16")
    nisa.dma_copy(dst=qnw_bf16, src=q_norm_weight.reshape((PMAX, 1)))
    qnw_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qnw_sb")
    nisa.tensor_copy(qnw_sb, qnw_bf16)

    knw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_bf16")
    nisa.dma_copy(dst=knw_bf16, src=k_norm_weight.reshape((PMAX, 1)))
    knw_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="knw_sb")
    nisa.tensor_copy(knw_sb, knw_bf16)

    # cos/sin in SBUF f32 [PMAX, 1] (B=1, so [PMAX, B] = [PMAX, 1])
    cos_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="cos_bf16")
    sin_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="sin_bf16")
    nisa.dma_copy(dst=cos_bf16, src=cos_col)
    nisa.dma_copy(dst=sin_bf16, src=sin_col)
    cos_f32 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="cos_f32")
    sin_f32 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="sin_f32")
    nisa.tensor_copy(cos_f32, cos_bf16)
    nisa.tensor_copy(sin_f32, sin_bf16)

    # All-ones [PMAX, PMAX] for reduction matmuls (RMSNorm sum-of-squares, softmax sums)
    rms_ones = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)

    # =========================================================================
    # HIDDEN TILE HOISTING
    # Pre-load all 16 hidden tiles [PMAX, 1] outside all loops. Reused for
    # all Q/K/V projections.
    # =========================================================================
    # Load entire hidden column as [PMAX, num_h_tiles] in one wide DMA.
    h_all = nl.ndarray((PMAX, num_h_tiles), dtype=nl.bfloat16, buffer=nl.sbuf, name="h_all")
    nisa.dma_copy(
        dst=h_all,
        src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
    )

    # =========================================================================
    # v12b Change 1: Pre-load all 8 Wq head tiles as 8 × 512KB DMAs issued early.
    # =========================================================================
    wq_heads = []
    for q_h in nl.affine_range(HQ_TP_CONST):
        wq_head_tile = nl.ndarray((PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"wq_head_early_{q_h}")
        nisa.dma_copy(dst=wq_head_tile, src=Wq[q_h * PMAX:(q_h + 1) * PMAX, :])
        wq_heads.append(wq_head_tile)

    # =========================================================================
    # WO WEIGHT RESHAPE (DMA hoisting deferred to just before flash decode)
    #
    # Wo is now passed as [Hq_out=1024, H_wo=2048] = [N*D, H] (caller transposes).
    # Reshape to [Hq_tp=8, d=128, H_wo=2048] so each head's slice is contiguous.
    # The actual SBUF hoisting (wo_sbuf loop) is issued just before flash decode
    # so the Wo DMA overlaps with flash decode TensorE/VectorE compute.
    # v13a: each core loads only [PMAX, H_half] per head (half the columns).
    # =========================================================================
    Wo_reshaped = Wo.reshape((Hq_tp, d, H_wo))  # logical [8, 128, 2048] view

    # =========================================================================
    # K PROJECTION (Hkv_tp=1, one KV head)
    # =========================================================================
    wk_full = nl.ndarray((PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="wk_full")
    nisa.dma_copy(dst=wk_full, src=Wk)

    # =========================================================================
    # v12b Change 2: Issue Wv DMA immediately after Wk DMA, before K projection loop.
    # =========================================================================
    wv_full = nl.ndarray((PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="wv_full")
    nisa.dma_copy(dst=wv_full, src=Wv)

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

    # Store k_rope to HBM for KV cache update (both cores write — idempotent).
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
    v_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="v_psum")
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(v_psum, stationary=wv_full[0:PMAX, h_t*PMAX:(h_t+1)*PMAX], moving=h_all[0:PMAX, h_t:h_t+1])

    v_active = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="v_active")
    nisa.tensor_copy(v_active, v_psum)

    # Store v_active to HBM for KV cache update (both cores write — idempotent).
    v_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf, name="v_bf16")
    nisa.tensor_copy(v_bf16, v_active)
    v_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name="v_T_psum")
    nisa.nc_transpose(v_T_psum, v_bf16)
    v_T_sb = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="v_T_sb")
    nisa.tensor_copy(v_T_sb, v_T_psum)
    nisa.dma_copy(dst=v_out, src=v_T_sb)

    # =========================================================================
    # Q PROJECTIONS — v12b Change 1: use pre-loaded wq_heads[] tiles
    # =========================================================================
    q_packed_f32 = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_packed_f32")
    for q_h in nl.affine_range(HQ_TP_CONST):
        q_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name=f"q_psum_{q_h}")
        for h_t in nl.affine_range(NUM_H_TILES):
            nisa.nc_matmul(
                q_psum,
                stationary=wq_heads[q_h][0:PMAX, h_t * PMAX:(h_t + 1) * PMAX],
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

    # tp_broadcast: qnw_sb[PMAX=128, 1] → qnw_gqa[PMAX=128, GQA=8]
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
    # Plan D: WO WEIGHT HOISTING — v13a: each core loads only H_half columns.
    # Core 0: Wo[:, 0:H_half], Core 1: Wo[:, H_half:H_wo].
    # ap() offset: head * PMAX * H_wo + core_id * H_half selects the correct shard.
    # Wo DMA issued just before flash decode to overlap with flash decode compute.
    # =========================================================================
    wo_sbuf = []
    for head in nl.affine_range(HQ_TP_CONST):
        wo_tile = nl.ndarray((PMAX, H_half), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"wo_tile_h{head}")
        nisa.dma_copy(
            dst=wo_tile,
            src=Wo_reshaped.ap(
                pattern=[[H_wo, PMAX], [1, H_half]],   # partition: stride H_wo, free: contiguous H_half
                offset=head * PMAX * H_wo + core_id * H_half,  # skip to this head+core shard
            ),
        )
        wo_sbuf.append(wo_tile)

    # =========================================================================
    # TWO-PASS FLASH DECODE
    # =========================================================================
    K_cache_2d = K_cache.reshape((S_prior, d))   # [S_prior, 128]
    V_cache_2d = V_cache.reshape((S_prior, d))   # [S_prior, 128]

    # --- Active position score: k_rope [PMAX,1] dot q_scaled [PMAX,GQA] ---
    # tp_broadcast: k_rope[PMAX=128, 1] → k_rope_packed[PMAX=128, GQA=8]
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
    # v10e: Load position scalar once for masking
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
    # Plan B — K-cache contiguous load + PE transpose
    # Hoist all K-cache tiles into SBUF before pass 1.
    # =========================================================================
    k_cache_tiles = []
    mask_tiles = []   # [PMAX, 1] f32 per tile: -1e9 for future/padding, 0 for valid
    for s_t in nl.affine_range(NUM_S_TILES):
        k_raw = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"k_raw_{s_t}")
        nisa.dma_copy(dst=k_raw, src=K_cache_2d[s_t * PMAX:(s_t + 1) * PMAX, :])

        k_ct_psum = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name=f"k_ct_psum_{s_t}")
        nisa.nc_transpose(k_ct_psum, k_raw)
        k_ct = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"k_ct_{s_t}")
        nisa.tensor_copy(k_ct, k_ct_psum)

        k_cache_tiles.append(k_ct)

        # v10e: Position-id threshold mask
        tile_start = s_t * PMAX  # Python int — compile-time constant per iteration

        neg_threshold = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"neg_threshold_{s_t}")
        nisa.tensor_scalar(neg_threshold, pos_f32,
                           op0=nl.multiply, operand0=-1.0,
                           op1=nl.add, operand1=float(tile_start))

        neg_thresh_psum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum, name=f"neg_thresh_psum_{s_t}")
        nisa.nc_transpose(neg_thresh_psum, neg_threshold.ap([[1, 1], [0, PMAX]], offset=0))
        neg_thresh_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"neg_thresh_sb_{s_t}")
        nisa.tensor_copy(neg_thresh_sb, neg_thresh_psum)

        delta = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"delta_{s_t}")
        nisa.tensor_tensor(delta, par_index_f32, neg_thresh_sb, op=nl.add)

        relu_delta = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"relu_delta_{s_t}")
        nisa.activation(relu_delta, op=nl.relu, data=delta)

        clamped = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"clamped_{s_t}")
        nisa.tensor_scalar(clamped, relu_delta, op0=nl.minimum, operand0=1.0)

        mask_tile_f32 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"mask_tile_f32_{s_t}")
        nisa.tensor_scalar(mask_tile_f32, clamped, op0=nl.multiply, operand0=-1e9)

        mask_tiles.append(mask_tile_f32)

    # Hoist all V-cache tiles into SBUF before pass 2.
    v_cache_tiles = []
    for s_t in nl.affine_range(NUM_S_TILES):
        v_ct = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"v_ct_{s_t}")
        nisa.dma_copy(
            dst=v_ct,
            src=V_cache_2d.ap(pattern=[[d, PMAX], [1, d]], offset=s_t * PMAX * d),
        )
        v_cache_tiles.append(v_ct)

    # --- Pass 1: find global max scalar across all K tiles + active position ---
    global_max_g1 = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="global_max_g1")
    nisa.memset(global_max_g1, value=-1e9)

    score_act_T_psum = nl.zeros((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="score_act_T_psum")
    nisa.nc_transpose(score_act_T_psum, score_active)
    score_act_T_sb = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="score_act_T_sb")
    nisa.tensor_copy(score_act_T_sb, score_act_T_psum)
    score_active_g1 = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="score_active_g1")
    nisa.tensor_reduce(dst=score_active_g1, op=nl.maximum, data=score_act_T_sb, axis=1)
    nisa.tensor_tensor(global_max_g1, global_max_g1, score_active_g1, op=nl.maximum)

    saved_scores = []

    for s_t in nl.affine_range(NUM_S_TILES):
        score_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"score_psum_{s_t}")
        nisa.nc_matmul(score_psum, stationary=k_cache_tiles[s_t], moving=q_bf16)
        score_sb = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score_sb_{s_t}")
        nisa.tensor_copy(score_sb, score_psum)

        mask_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name=f"mask_gqa_psum_T_{s_t}")
        nisa.nc_transpose(mask_gqa_psum_T, mask_tiles[s_t].ap([[1, PMAX], [0, GQA]], offset=0))
        mask_gqa_sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name=f"mask_gqa_sbuf_T_{s_t}")
        nisa.tensor_copy(mask_gqa_sbuf_T, mask_gqa_psum_T)
        mask_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"mask_gqa_psum_{s_t}")
        nisa.nc_transpose(mask_gqa_psum, mask_gqa_sbuf_T)
        mask_gqa = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"mask_gqa_{s_t}")
        nisa.tensor_copy(mask_gqa, mask_gqa_psum)

        score_sb_masked = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score_sb_masked_{s_t}")
        nisa.tensor_tensor(score_sb_masked, score_sb, mask_gqa, op=nl.add)

        saved_scores.append(score_sb_masked)

        score_T_psum = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name=f"score_T_psum_{s_t}")
        nisa.nc_transpose(score_T_psum, score_sb_masked)
        score_T_sb = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name=f"score_T_sb_{s_t}")
        nisa.tensor_copy(score_T_sb, score_T_psum)

        tile_max_vec = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"tile_max_vec_{s_t}")
        nisa.tensor_reduce(dst=tile_max_vec, op=nl.maximum, data=score_T_sb, axis=1)

        nisa.tensor_tensor(global_max_g1, global_max_g1, tile_max_vec, op=nl.maximum)

    # Negate compact global max.
    neg_max_g1 = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="neg_max_g1")
    nisa.tensor_scalar(neg_max_g1, global_max_g1, op0=nl.multiply, operand0=-1.0)

    neg_max_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="neg_max_psum")
    nisa.nc_transpose(
        neg_max_psum,
        neg_max_g1.ap([[1, GQA], [0, PMAX]], offset=0),
    )
    neg_max = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="neg_max")
    nisa.tensor_copy(neg_max, neg_max_psum)

    # --- Pass 2: use saved scores, exp(score - global_max), accumulate V ---
    v_acc = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="v_acc")
    nisa.memset(v_acc, value=0.0)
    sum_acc = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="sum_acc")
    nisa.memset(sum_acc, value=0.0)

    for s_t in nl.affine_range(NUM_S_TILES):
        score2_shifted = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score2_shifted_{s_t}")
        nisa.tensor_tensor(score2_shifted, saved_scores[s_t], neg_max, op=nl.add)

        score2_exp = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score2_exp_{s_t}")
        nisa.activation(score2_exp, op=nl.exp, data=score2_shifted)

        score2_exp_bf16 = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"score2_exp_bf16_{s_t}")
        nisa.tensor_copy(score2_exp_bf16, score2_exp)

        tile_sum_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"tile_sum_psum_{s_t}")
        nisa.nc_matmul(tile_sum_psum, stationary=rms_ones, moving=score2_exp_bf16)
        tile_sum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"tile_sum_{s_t}")
        nisa.tensor_copy(tile_sum, tile_sum_psum)
        nisa.tensor_tensor(sum_acc, sum_acc, tile_sum, op=nl.add)

        v_weighted_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"v_weighted_psum_{s_t}")
        nisa.nc_matmul(v_weighted_psum, stationary=v_cache_tiles[s_t], moving=score2_exp_bf16)
        v_weighted = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"v_weighted_{s_t}")
        nisa.tensor_copy(v_weighted, v_weighted_psum)
        nisa.tensor_tensor(v_acc, v_acc, v_weighted, op=nl.add)

    # --- Active position contribution ---
    score_act_shifted = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="score_act_shifted")
    nisa.tensor_tensor(score_act_shifted, score_active, neg_max, op=nl.add)
    score_act_exp = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="score_act_exp")
    nisa.activation(score_act_exp, op=nl.exp, data=score_act_shifted)
    nisa.tensor_tensor(sum_acc, sum_acc, score_act_exp, op=nl.add)

    # tp_broadcast: v_active[PMAX=128, 1] → v_act_packed[PMAX=128, GQA=8]
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

    # Cast attention output to bf16 for the matmul stationary operand
    attn_out = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name="attn_out")
    nisa.tensor_tensor(attn_out, v_acc, inv_sum, op=nl.multiply)

    # =========================================================================
    # FUSED OUTPUT PROJECTION — v13a LNC=2: each core handles H_half columns.
    # 2 PSUMs instead of 4; wo_sbuf[head] is [PMAX, H_half] (half width).
    # Core 0: writes output[:, 0:H_half]
    # Core 1: writes output[:, H_half:H_wo]
    # =========================================================================
    res_psum_0 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_0")
    res_psum_1 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_1")
    for head in nl.affine_range(HQ_TP_CONST):
        nisa.nc_matmul(res_psum_0, stationary=attn_out[0:PMAX, head:head+1],
                       moving=wo_sbuf[head][0:PMAX, 0:F_MAX])
        nisa.nc_matmul(res_psum_1, stationary=attn_out[0:PMAX, head:head+1],
                       moving=wo_sbuf[head][0:PMAX, F_MAX:H_half])

    # Store the 2 output blocks to the correct half of output_2d.
    # Use ap() with dynamic core_id offset — same pattern as Wo loading above.
    # output_2d is [1, H_wo=2048]; ap([[H_wo,1],[1,F_MAX]]) selects one F_MAX-wide strip.
    # Core 0: offset=0*H_half+0, 0*H_half+F_MAX  → columns [0:512], [512:1024]
    # Core 1: offset=1*H_half+0, 1*H_half+F_MAX  → columns [1024:1536], [1536:2048]
    out_sb_0 = nl.ndarray((1, F_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_sb_0")
    nisa.tensor_copy(out_sb_0, res_psum_0)
    nisa.dma_copy(
        dst=output_2d.ap(pattern=[[H_wo, 1], [1, F_MAX]], offset=core_id * H_half + 0),
        src=out_sb_0,
    )
    out_sb_1 = nl.ndarray((1, F_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_sb_1")
    nisa.tensor_copy(out_sb_1, res_psum_1)
    nisa.dma_copy(
        dst=output_2d.ap(pattern=[[H_wo, 1], [1, F_MAX]], offset=core_id * H_half + F_MAX),
        src=out_sb_1,
    )

    return output, k_rope_out, v_out
