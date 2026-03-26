"""
v9_tp_broadcast: Replace 128-iteration neg_max broadcast loop with ap(stride_f=0) pattern.

Changes from v8_wo_transposed:
  tp_broadcast for neg_max broadcast:
    The 128-iteration tensor_copy loop that replicated neg_max_g1 [GQA=8, 1]
    across PMAX=128 free columns (neg_max_wide [GQA=8, PMAX=128]) has been
    replaced with the nkilib production ap() pattern.

    Old approach (v8_wo_transposed):
      neg_max_wide = nl.ndarray((GQA, PMAX), ...)
      for f in nl.affine_range(PMAX):          # 128 tensor_copy instructions
          nisa.tensor_copy(neg_max_wide[0:GQA, f:f+1], neg_max_g1)
      neg_max_psum = nl.zeros((PMAX, GQA), ...)
      nisa.nc_transpose(neg_max_psum, neg_max_wide)
      neg_max = nl.ndarray((PMAX, GQA), ...)
      nisa.tensor_copy(neg_max, neg_max_psum)

    New approach (v9_tp_broadcast):
      Adopted from nkilib/core/utils/tp_broadcast.py.
      ap([[1, GQA], [0, PMAX]], offset=0) on neg_max_g1 [GQA=8, 1]:
        indexed[p, f] = flat[p*1 + f*0] = flat[p]  -> stride_f=0 broadcasts
        partition values across all PMAX=128 free columns.
      nc_transpose reads the [GQA=8, PMAX=128] view -> psum[PMAX=128, GQA=8].
      Replaces 128-loop + nc_transpose + tensor_copy (~130 instr) with 2 instructions.

  All other optimizations (v8 Wo contiguous DMA, Plan B cached scores, SBUF
  hoisting, hidden tile hoisting, packed Q RMSNorm+RoPE, packed K/V cache
  hoisting) preserved unchanged.

Preserved unchanged from v8_wo_transposed:
  Column layout reshapes, hidden tile hoisting, wk/wv/wq wide-row loads,
  packed Q RMSNorm + RoPE, rms_ones, global_max_g1 compact scalar,
  active-position score, normalization (rsqrt trick), Plan B saved_scores,
  test harness (with Wo.T.contiguous() passed to kernel).

Shape assumptions (Qwen3-30B-A3B, TP=4, Hkv_tp=1 per rank):
  Hq_tp=8, Hkv_tp=1, GQA=8, d=128, H=2048
  Wk: [128, 2048], Wv: [128, 2048], Wq: [1024, 2048]
  Wo: [1024, 2048]  (transposed o_proj weight: caller passes Wo.T.contiguous())
  K_cache: [1, 1, S_prior, 128], V_cache: [1, 1, S_prior, 128]
  Output: [1, 1, 2048] bf16
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
os.environ["XLA_IR_DEBUG"]= "1"
os.environ["XLA_HLO_DEBUG"]= "1"
os.environ["NEURON_RT_INSPECT_ENABLE"]= "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"]= "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]= "./output"

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

@nki.jit
def qwen3_attn_tkg_fused_oproj(
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
    position_ids,    # [B, 1]           int32 (unused in kernel body)
):
    """
    Fused QKV + RMSNorm + RoPE + flash decode + output projection.
    Returns [B, 1, H_out] where H_out = H = 2048 (no LNC sharding for now).

    Wo is passed as [Hq_out=1024, H_wo=2048] (caller transposes the weight).
    This enables contiguous DMA loading via nkilib-style ap() pattern.
    """
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

    # Output H: since no LNC, each core writes all H=2048 of Wo output
    # Wo is now [Hq_out=1024, H_wo=2048] (transposed), so H_wo is shape[1]
    H_wo = Wo.shape[1]              # 2048
    num_h_blocks = H_wo // F_MAX   # 4

    # =========================================================================
    # COLUMN LAYOUT RESHAPES
    # =========================================================================
    # Output [B, 1, H_wo]: allocate in HBM
    output = nl.ndarray((B, 1, H_wo), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # Reshape to [1, H_wo] for o_proj DMA stores.
    output_2d = output.reshape((1, H_wo))

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

    # Plan C1 NOTE: The original plan specified using nc_matmul outer-products
    # ([PMAX,1] @ [1,GQA]) for broadcasting.  However, the Trainium2 hardware
    # requires both operands of nc_matmul to share the same partition dimension
    # (par_dim must be equal).  [PMAX,1] has par=PMAX=128, [1,GQA] has par=1 —
    # they differ, so the compiler rejects them ("Fmap and Weight partitions must
    # match").  The original for-loop broadcasts from v6_ultimate are preserved
    # at all 6 sites as they are the correct working approach.

    # =========================================================================
    # HIDDEN TILE HOISTING
    # Pre-load all 16 hidden tiles [PMAX, 1] outside all loops. Reused for
    # all Q/K/V projections.
    # =========================================================================
    # Load entire hidden column as [PMAX, num_h_tiles] in one wide DMA.
    # hidden_col is [H, B] = [2048, 1] row-major; flat offset of [r,0] = r.
    # We want h_all[p, f] = hidden_col[f*PMAX + p, 0], i.e. flat offset = f*PMAX + p.
    # ap() pattern: partition p steps by 1 (count PMAX), free f steps by PMAX (count num_h_tiles).
    h_all = nl.ndarray((PMAX, num_h_tiles), dtype=nl.bfloat16, buffer=nl.sbuf, name="h_all")
    nisa.dma_copy(
        dst=h_all,
        src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
    )

    # =========================================================================
    # WO WEIGHT HOISTING — nkilib-style contiguous DMA
    #
    # Wo is now passed as [Hq_out=1024, H_wo=2048] = [N*D, H] (caller transposes).
    # Reshape to [Hq_tp=8, d=128, H_wo=2048] so each head's slice is contiguous.
    #
    # ap() pattern [[H_wo, PMAX], [1, H_wo]]:
    #   partition stride = H_wo = 2048 (one full output row apart)
    #   free stride = 1 (contiguous elements)
    # → 128 chunks × H_wo×2 = 4096 bytes each, stride 4096 bytes → 50% fill ratio
    # vs old [[1,128],[Hq_out,H_wo]]: 2048 chunks × 256 bytes, stride 2048 → 12.5%
    # 16× fewer DMA packets, 4× better fill ratio.
    # =========================================================================
    Wo_reshaped = Wo.reshape((Hq_tp, d, H_wo))  # logical [8, 128, 2048] view

    wo_sbuf = []
    for head in nl.affine_range(Hq_tp):
        wo_tile = nl.ndarray((PMAX, H_wo), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"wo_tile_h{head}")
        nisa.dma_copy(
            dst=wo_tile,
            src=Wo_reshaped.ap(
                pattern=[[H_wo, PMAX], [1, H_wo]],   # partition: stride H_wo, free: contiguous
                offset=head * PMAX * H_wo,             # skip head * [128 × 2048] elements
            ),
        )
        wo_sbuf.append(wo_tile)

    # =========================================================================
    # K PROJECTION (Hkv_tp=1, one KV head)
    # Wide row load: load entire Wk row [128, 2048] in 16 tiles of [128, 128],
    # then matmul each tile with corresponding hidden tile.
    # =========================================================================
    wk_full = nl.ndarray((PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="wk_full")
    nisa.dma_copy(dst=wk_full, src=Wk)
    k_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="k_psum")
    for h_t in nl.affine_range(num_h_tiles):
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
    # Fused tensor_scalar: multiply by 1/d then add eps
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

    # =========================================================================
    # V PROJECTION (Hkv_tp=1)
    # =========================================================================
    wv_full = nl.ndarray((PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="wv_full")
    nisa.dma_copy(dst=wv_full, src=Wv)
    v_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="v_psum")
    for h_t in nl.affine_range(num_h_tiles):
        nisa.nc_matmul(v_psum, stationary=wv_full[0:PMAX, h_t*PMAX:(h_t+1)*PMAX], moving=h_all[0:PMAX, h_t:h_t+1])

    v_active = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="v_active")
    nisa.tensor_copy(v_active, v_psum)

    # =========================================================================
    # Q PROJECTIONS — all 8 heads, then pack into [PMAX, GQA=8]
    # =========================================================================
    wq_heads = []
    for q_h in nl.affine_range(Hq_tp):
        wq_head = nl.ndarray((PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"wq_head_{q_h}")
        nisa.dma_copy(
            dst=wq_head,
            src=Wq[q_h * PMAX:(q_h + 1) * PMAX, :],
        )
        wq_heads.append(wq_head)

    q_psum_list = []
    for q_h in nl.affine_range(Hq_tp):
        q_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name=f"q_psum_{q_h}")
        for h_t in nl.affine_range(num_h_tiles):
            nisa.nc_matmul(q_psum, stationary=wq_heads[q_h][0:PMAX, h_t*PMAX:(h_t+1)*PMAX], moving=h_all[0:PMAX, h_t:h_t+1])
        q_psum_list.append(q_psum)

    # Move all Q PSUMs to SBUF
    q_vec_list = []
    for q_h in nl.affine_range(Hq_tp):
        q_vec = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name=f"q_vec_{q_h}")
        nisa.tensor_copy(q_vec, q_psum_list[q_h])
        q_vec_list.append(q_vec)

    # Pack Q into [PMAX, GQA] f32 for packed RMSNorm + RoPE
    q_packed_f32 = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_packed_f32")
    for q_h in nl.affine_range(Hq_tp):
        nisa.tensor_copy(q_packed_f32[0:PMAX, q_h:q_h+1], q_vec_list[q_h])

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
    # Fused: multiply by 1/d, add eps
    q_mean_sq = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_mean_sq")
    nisa.tensor_scalar(q_mean_sq, q_sum_sb, op0=nl.multiply, operand0=1.0/d, op1=nl.add, operand1=EPS)
    q_rms_inv = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_rms_inv")
    nisa.activation(q_rms_inv, op=nl.rsqrt, data=q_mean_sq)
    q_normed = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_normed")
    nisa.tensor_tensor(q_normed, q_packed_f32, q_rms_inv, op=nl.multiply)

    # Apply norm weight: qnw_sb [PMAX, 1] broadcast to [PMAX, GQA] before multiply.
    # for-loop broadcast (8 tensor_copy calls) — same as v6_ultimate.
    qnw_gqa = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="qnw_gqa")
    for g in nl.affine_range(GQA):
        nisa.tensor_copy(qnw_gqa[0:PMAX, g:g+1], qnw_sb)

    q_normed2 = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_normed2")
    nisa.tensor_tensor(q_normed2, q_normed, qnw_gqa, op=nl.multiply)

    # =========================================================================
    # PACKED Q ROPE on [PMAX, GQA=8]
    # =========================================================================
    # cos/sin [PMAX,1] broadcast to [PMAX,GQA] — for-loop (same as v6_ultimate).
    cos_gqa = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="cos_gqa")
    sin_gqa = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="sin_gqa")
    for g in nl.affine_range(GQA):
        nisa.tensor_copy(cos_gqa[0:PMAX, g:g+1], cos_f32)
        nisa.tensor_copy(sin_gqa[0:PMAX, g:g+1], sin_f32)

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

    # Scale by 1/sqrt(d) and cast to bf16 — this is the "scaled Q" for flash decode
    q_bf16 = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name="q_bf16")
    nisa.tensor_scalar(q_bf16, q_rope, op0=nl.multiply, operand0=INV_SQRT_D)

    # =========================================================================
    # TWO-PASS FLASH DECODE
    # =========================================================================
    K_cache_2d = K_cache.reshape((S_prior, d))   # [S_prior, 128]
    V_cache_2d = V_cache.reshape((S_prior, d))   # [S_prior, 128]

    # --- Active position score: k_rope [PMAX,1] dot q_scaled [PMAX,GQA] ---
    # k_rope [PMAX,1] broadcast to [PMAX,GQA] — for-loop (same as v6_ultimate).
    k_rope_packed = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="k_rope_packed")
    for g in nl.affine_range(GQA):
        nisa.tensor_copy(k_rope_packed[0:PMAX, g:g+1], k_rope)

    kq_elem = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="kq_elem")
    nisa.tensor_tensor(kq_elem, k_rope_packed, q_bf16, op=nl.multiply)
    kq_elem_bf16 = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name="kq_elem_bf16")
    nisa.tensor_copy(kq_elem_bf16, kq_elem)
    score_active_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="score_active_psum")
    nisa.nc_matmul(score_active_psum, stationary=rms_ones, moving=kq_elem_bf16)
    score_active = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="score_active")
    nisa.tensor_copy(score_active, score_active_psum)

    # Hoist all K-cache tiles into SBUF before pass 1 — reused in both passes.
    k_cache_tiles = []
    for s_t in nl.affine_range(num_s_tiles):
        k_ct = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"k_ct_{s_t}")
        nisa.dma_copy(
            dst=k_ct,
            src=K_cache_2d.ap(pattern=[[1, PMAX], [d, PMAX]], offset=s_t * PMAX * d),
        )
        k_cache_tiles.append(k_ct)

    # Hoist all V-cache tiles into SBUF before pass 2.
    v_cache_tiles = []
    for s_t in nl.affine_range(num_s_tiles):
        v_ct = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"v_ct_{s_t}")
        nisa.dma_copy(
            dst=v_ct,
            src=V_cache_2d.ap(pattern=[[d, PMAX], [1, d]], offset=s_t * PMAX * d),
        )
        v_cache_tiles.append(v_ct)

    # --- Pass 1: find global max scalar across all K tiles + active position ---
    global_max_g1 = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="global_max_g1")
    nisa.memset(global_max_g1, value=-1e9)

    # score_active [PMAX, GQA] → transpose → [GQA, PMAX] → reduce max over axis=1 → [GQA, 1]
    score_act_T_psum = nl.zeros((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="score_act_T_psum")
    nisa.nc_transpose(score_act_T_psum, score_active)
    score_act_T_sb = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="score_act_T_sb")
    nisa.tensor_copy(score_act_T_sb, score_act_T_psum)
    score_active_g1 = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="score_active_g1")
    nisa.tensor_reduce(dst=score_active_g1, op=nl.maximum, data=score_act_T_sb, axis=1)
    nisa.tensor_tensor(global_max_g1, global_max_g1, score_active_g1, op=nl.maximum)

    # ── Plan B: saved_scores list for Pass 2 reuse ─────────────────────────────
    # Collect score_sb from Pass 1 to avoid recomputing K×Q matmul in Pass 2.
    # Memory cost: GQA * PMAX * 4 bytes = 4 KB per tile, 20 KB for S_prior=640.
    saved_scores = []

    for s_t in nl.affine_range(num_s_tiles):
        # score [PMAX, GQA]: K_tile[PMAX,PMAX] @ q_bf16[PMAX,GQA]
        # name= suffixes use s_t to keep each iteration's tensor name unique —
        # the NKI compiler requires unique names even inside affine_range loops.
        score_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"score_psum_{s_t}")
        nisa.nc_matmul(score_psum, stationary=k_cache_tiles[s_t], moving=q_bf16)
        score_sb = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score_sb_{s_t}")
        nisa.tensor_copy(score_sb, score_psum)
        saved_scores.append(score_sb)   # cache score — reused in Pass 2, no re-matmul

        # Per-tile max reduction: transpose [PMAX,GQA] → [GQA,PMAX], reduce max → [GQA,1]
        score_T_psum = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name=f"score_T_psum_{s_t}")
        nisa.nc_transpose(score_T_psum, score_sb)
        score_T_sb = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name=f"score_T_sb_{s_t}")
        nisa.tensor_copy(score_T_sb, score_T_psum)

        tile_max_vec = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"tile_max_vec_{s_t}")
        nisa.tensor_reduce(dst=tile_max_vec, op=nl.maximum, data=score_T_sb, axis=1)

        # tile_max_vec [GQA, 1] and global_max_g1 [GQA, 1] — direct max, no broadcast needed.
        nisa.tensor_tensor(global_max_g1, global_max_g1, tile_max_vec, op=nl.maximum)

    # Negate compact global max.
    neg_max_g1 = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="neg_max_g1")
    nisa.tensor_scalar(neg_max_g1, global_max_g1, op0=nl.multiply, operand0=-1.0)

    # tp_broadcast: neg_max_g1[GQA=8, 1] → neg_max[PMAX=128, GQA=8]
    # Adopted from nkilib/core/utils/tp_broadcast.py production pattern.
    # ap([[1, GQA], [0, PMAX]], offset=0): indexed[p, f] = flat[p*1 + f*0] = flat[p]
    # → 8 partition values each broadcast across all PMAX=128 free columns.
    # nc_transpose reads the [GQA=8, PMAX=128] view and writes psum[PMAX=128, GQA=8].
    # Replaces 128-loop + nc_transpose + tensor_copy (~130 instr) with 2 instructions.
    neg_max_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="neg_max_psum")
    nisa.nc_transpose(
        neg_max_psum,
        neg_max_g1.ap([[1, GQA], [0, PMAX]], offset=0),
    )
    neg_max = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="neg_max")
    nisa.tensor_copy(neg_max, neg_max_psum)

    # --- Pass 2: use saved scores (no K matmul), exp(score - global_max), accumulate V ---
    v_acc = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="v_acc")
    nisa.memset(v_acc, value=0.0)
    sum_acc = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="sum_acc")
    nisa.memset(sum_acc, value=0.0)

    for s_t in nl.affine_range(num_s_tiles):
        # ── Plan B: No K matmul — reuse score cached from Pass 1 ──────────────
        # saved_scores[s_t] is SBUF→SBUF add only; the nc_matmul+tensor_copy
        # from Pass 1 is gone — for S_prior=640 this removes 5 matmuls.
        # Unique name= suffixes required by the NKI compiler across loop iterations.
        score2_shifted = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score2_shifted_{s_t}")
        nisa.tensor_tensor(score2_shifted, saved_scores[s_t], neg_max, op=nl.add)

        score2_exp = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score2_exp_{s_t}")
        nisa.activation(score2_exp, op=nl.exp, data=score2_shifted)

        score2_exp_bf16 = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"score2_exp_bf16_{s_t}")
        nisa.tensor_copy(score2_exp_bf16, score2_exp)

        # Accumulate softmax denominator: rms_ones @ score2_exp_bf16 → [PMAX, GQA] row-sum
        tile_sum_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"tile_sum_psum_{s_t}")
        nisa.nc_matmul(tile_sum_psum, stationary=rms_ones, moving=score2_exp_bf16)
        tile_sum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"tile_sum_{s_t}")
        nisa.tensor_copy(tile_sum, tile_sum_psum)
        nisa.tensor_tensor(sum_acc, sum_acc, tile_sum, op=nl.add)

        # V-weighted accumulation: stationary=v_cache_tiles[s_t], moving=score2_exp_bf16
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

    # v_active [PMAX,1] broadcast to [PMAX,GQA] — for-loop (same as v6_ultimate).
    v_act_packed = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="v_act_packed")
    for g in nl.affine_range(GQA):
        nisa.tensor_copy(v_act_packed[0:PMAX, g:g+1], v_active)

    v_act_weighted = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="v_act_weighted")
    nisa.tensor_tensor(v_act_weighted, v_act_packed, score_act_exp, op=nl.multiply)
    nisa.tensor_tensor(v_acc, v_acc, v_act_weighted, op=nl.add)

    # --- Normalize: attn_out = v_acc / sum_acc ---
    sum_safe = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="sum_safe")
    nisa.tensor_scalar(sum_safe, sum_acc, op0=nl.add, operand0=1e-9)
    # Use rsqrt trick: 1/x = rsqrt(x)^2 (avoids a native divide instruction)
    rsqrt_sum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="rsqrt_sum")
    nisa.activation(rsqrt_sum, op=nl.rsqrt, data=sum_safe)
    inv_sum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="inv_sum")
    nisa.tensor_tensor(inv_sum, rsqrt_sum, rsqrt_sum, op=nl.multiply)

    # Cast attention output to bf16 for the matmul stationary operand
    attn_out = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name="attn_out")
    nisa.tensor_tensor(attn_out, v_acc, inv_sum, op=nl.multiply)

    # =========================================================================
    # FUSED OUTPUT PROJECTION (v8 nkilib contiguous DMA pattern)
    #
    # attn_out [PMAX, GQA=8] bf16 — stationary per head, [PMAX, 1] slice.
    # wo_sbuf[head] = [PMAX=d, H_wo=2048] with wo_sbuf[head][p, f] = Wo_new[head*d+p, f].
    # nc_matmul: result[0, f] = sum_p attn_out[p, head] * Wo_new[head*d+p, f].
    # Accumulated over all 8 heads → output[f] = attn_out_flat @ Wo_new (= attn_out_flat @ Wo_old.T).
    #
    # Correctness check:
    #   old: wo_sbuf[h][p, f] = Wo_old[f, h*128+p]  → result = attn_out_flat @ Wo_old.T
    #   new: wo_sbuf[h][p, f] = Wo_new[h*128+p, f]  → result = attn_out_flat @ Wo_new
    #   Since Wo_new = Wo_old.T, both give the same result. ✓
    #
    # 4 output blocks of F_MAX=512 each — same as v6_ultimate.
    # =========================================================================
    for h_blk in nl.affine_range(num_h_blocks):
        # stationary=[PMAX,1], moving=[PMAX,F_MAX] → PSUM result=[1, F_MAX].
        # All 8 heads statically unrolled — nc_matmul accumulates into res_psum.
        res_psum = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name=f"res_psum_{h_blk}")
        nisa.nc_matmul(res_psum, stationary=attn_out[0:PMAX, 0:1], moving=wo_sbuf[0][0:PMAX, h_blk*F_MAX:(h_blk+1)*F_MAX])
        nisa.nc_matmul(res_psum, stationary=attn_out[0:PMAX, 1:2], moving=wo_sbuf[1][0:PMAX, h_blk*F_MAX:(h_blk+1)*F_MAX])
        nisa.nc_matmul(res_psum, stationary=attn_out[0:PMAX, 2:3], moving=wo_sbuf[2][0:PMAX, h_blk*F_MAX:(h_blk+1)*F_MAX])
        nisa.nc_matmul(res_psum, stationary=attn_out[0:PMAX, 3:4], moving=wo_sbuf[3][0:PMAX, h_blk*F_MAX:(h_blk+1)*F_MAX])
        nisa.nc_matmul(res_psum, stationary=attn_out[0:PMAX, 4:5], moving=wo_sbuf[4][0:PMAX, h_blk*F_MAX:(h_blk+1)*F_MAX])
        nisa.nc_matmul(res_psum, stationary=attn_out[0:PMAX, 5:6], moving=wo_sbuf[5][0:PMAX, h_blk*F_MAX:(h_blk+1)*F_MAX])
        nisa.nc_matmul(res_psum, stationary=attn_out[0:PMAX, 6:7], moving=wo_sbuf[6][0:PMAX, h_blk*F_MAX:(h_blk+1)*F_MAX])
        nisa.nc_matmul(res_psum, stationary=attn_out[0:PMAX, 7:8], moving=wo_sbuf[7][0:PMAX, h_blk*F_MAX:(h_blk+1)*F_MAX])
        # Cast PSUM → bf16 via tensor_copy to SBUF
        out_sb = nl.ndarray((1, F_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"out_sb_{h_blk}")
        nisa.tensor_copy(out_sb, res_psum)
        # Store [1, F_MAX] → output_2d[0, h_blk*F_MAX:(h_blk+1)*F_MAX]
        nisa.dma_copy(
            dst=output_2d[0:1, h_blk * F_MAX:(h_blk + 1) * F_MAX],
            src=out_sb[0:1, 0:F_MAX],
        )

    return output


# =============================================================================
# Test harness (v8: Wo passed as Wo.T.contiguous() to kernel)
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
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rmsnorm_per_head(x, weight, eps=1e-6):
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(variance + eps)
        return (x_normed * weight.float()).to(x.dtype)

    def pytorch_reference(
        hidden_states, Wq, Wk, Wv, Wo,
        q_norm_weight, k_norm_weight,
        K_cache, V_cache,
        cos_at_pos, sin_at_pos,
        d=128,
    ):
        """
        Reference implementation including fused output projection.
        K_cache: [B, 1, S_prior, d]
        V_cache: [B, 1, S_prior, d]
        Wo:      [H_out, Hq_tp*d] = [2048, 1024]  row-parallel o_proj (original shape)
        """
        B, _, H = hidden_states.shape
        Hq_out = Wq.shape[0]          # 1024
        Hkv_out = Wk.shape[0]         # 128
        Hq_tp = Hq_out // d           # 8
        Hkv_tp = Hkv_out // d         # 1
        gqa = Hq_tp // Hkv_tp         # 8
        S_prior = K_cache.shape[2]
        H_out = Wo.shape[0]            # 2048

        hs = hidden_states.float()
        Q = hs @ Wq.float().T          # [B, 1, 1024]
        K = hs @ Wk.float().T          # [B, 1, 128]
        V = hs @ Wv.float().T          # [B, 1, 128]

        Q = Q.reshape(B, Hq_tp, 1, d)
        K = K.reshape(B, Hkv_tp, 1, d)
        V = V.reshape(B, Hkv_tp, 1, d)

        Q_norm = torch.zeros_like(Q)
        K_norm = torch.zeros_like(K)
        for h in range(Hq_tp):
            Q_norm[:, h, :, :] = apply_rmsnorm_per_head(Q[:, h, :, :], q_norm_weight)
        for h in range(Hkv_tp):
            K_norm[:, h, :, :] = apply_rmsnorm_per_head(K[:, h, :, :], k_norm_weight)

        cos = cos_at_pos.float().unsqueeze(1).unsqueeze(2)   # [B, 1, 1, d]
        sin = sin_at_pos.float().unsqueeze(1).unsqueeze(2)

        Q_rope = Q_norm.float() * cos + rotate_half(Q_norm.float()) * sin
        K_rope = K_norm.float() * cos + rotate_half(K_norm.float()) * sin

        scale = 1.0 / math.sqrt(d)

        # Attention output: [B, Hq_tp, d]
        attn_output_heads = torch.zeros(B, Hq_tp, d, dtype=torch.float32)

        for b in range(B):
            for q_h in range(Hq_tp):
                kv_h = q_h // gqa  # = 0 for all heads (Hkv_tp=1)
                q_vec = Q_rope[b, q_h, 0, :]   # [d]

                K_full = torch.cat([
                    K_cache[b, kv_h].float(),           # [S_prior, d]
                    K_rope[b, kv_h, 0:1, :].float(),    # [1, d]
                ], dim=0)  # [S_prior+1, d]

                V_full = torch.cat([
                    V_cache[b, kv_h].float(),            # [S_prior, d]
                    V[b, kv_h, 0:1, :].float(),          # [1, d]
                ], dim=0)  # [S_prior+1, d]

                scores = (K_full @ q_vec) * scale       # [S_prior+1]
                attn_weights = F.softmax(scores, dim=0) # [S_prior+1]
                out_vec = (attn_weights.unsqueeze(-1) * V_full).sum(dim=0)  # [d]
                attn_output_heads[b, q_h, :] = out_vec

        # Reshape attn output to [B, 1, Hq_tp*d] for o_proj
        attn_output = attn_output_heads.reshape(B, 1, Hq_tp * d).to(torch.bfloat16)

        # Output projection: [B, 1, Hq_tp*d] @ Wo.T -> [B, 1, H_out]
        # Wo: [H_out, Hq_tp*d] — row-parallel (original untransposed shape)
        output = attn_output.float() @ Wo.float().T  # [B, 1, H_out]
        return output.to(torch.bfloat16)

    def run_test(B=1, S_prior=640, H=2048, d=128, Hq_tp=8, Hkv_tp=1):
        print(f"\n{'='*70}")
        print(f"v9_tp_broadcast Test")
        print(f"B={B}, S_prior={S_prior}, H={H}, d={d}, Hq_tp={Hq_tp}, Hkv_tp={Hkv_tp}")
        print(f"{'='*70}")

        device = xm.xla_device()
        dtype = torch.bfloat16

        Hq_out = Hq_tp * d       # 1024
        Hkv_out = Hkv_tp * d     # 128
        H_wo = H                  # 2048

        torch.manual_seed(42)
        scale = 0.05

        hidden_states = (torch.randn(B, 1, H) * scale).to(dtype)
        Wq = (torch.randn(Hq_out, H) * scale).to(dtype)
        Wk = (torch.randn(Hkv_out, H) * scale).to(dtype)
        Wv = (torch.randn(Hkv_out, H) * scale).to(dtype)
        Wo = (torch.randn(H_wo, Hq_out) * scale).to(dtype)  # [2048, 1024] original shape
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
            hidden_states, Wq, Wk, Wv, Wo,
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

        # Transpose Wo for the kernel: kernel expects [Hq_out=1024, H_wo=2048]
        Wo_kernel = Wo.T.contiguous()  # [1024, 2048]

        hidden_dev = hidden_states.to(device)
        Wq_dev = Wq.to(device)
        Wk_dev = Wk.to(device)
        Wv_dev = Wv.to(device)
        Wo_dev = Wo_kernel.to(device)  # pass transposed Wo to kernel
        q_norm_dev = q_norm_weight.to(device)
        k_norm_dev = k_norm_weight.to(device)
        K_cache_dev = K_cache.to(device)
        V_cache_dev = V_cache.to(device)
        cos_dev = cos_nki.to(device)
        sin_dev = sin_nki.to(device)
        pos_dev = position_ids.to(device)

        print("\n[3/4] Running NKI kernel (v9_tp_broadcast)...")
        nki_out_dev = qwen3_attn_tkg_fused_oproj[2](
            hidden_dev, Wq_dev, Wk_dev, Wv_dev, Wo_dev,
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

        import numpy as np
        try:
            np.testing.assert_allclose(
                nki_f.numpy(), ref_f.numpy(), rtol=1e-2, atol=1e-2
            )
            passed = True
        except (AssertionError, Exception) as e:
            print(f"  assert_allclose FAILED: {e}")
            passed = False

        threshold = 0.1
        if max_diff < threshold and passed:
            print(f"\n{'='*70}")
            print(f"PASS: max_diff={max_diff:.4e} < threshold={threshold}")
            print(f"{'='*70}")
            return True
        else:
            print(f"\n{'='*70}")
            print(f"FAIL: max_diff={max_diff:.4e} (threshold={threshold}), allclose={passed}")
            print(f"{'='*70}")
            print(f"\nRef sample:  {ref_f[0, 0, :8].tolist()}")
            print(f"NKI sample:  {nki_f[0, 0, :8].tolist()}")
            return False

    ok = run_test(B=1, S_prior=640, H=2048, d=128, Hq_tp=8, Hkv_tp=1)
    sys.exit(0 if ok else 1)
