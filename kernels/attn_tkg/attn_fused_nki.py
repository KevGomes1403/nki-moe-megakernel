"""
v14d_kv_norm_hoisted_weights: v14c_kv_norm_hoisted plus weight-prefetch kwargs.

Key difference over v14c:
  Four large weight matrices — Wk, Wv, Wq (per owned head), and Wo (per owned head) —
  can now be passed in as already-loaded bf16 SBUF tensors.  When a weight kwarg is not
  None, the corresponding sbm.alloc_stack + nisa.dma_copy calls are skipped and the
  caller-owned tensor is used directly.

  This is the building block for the multi-layer megakernel: call once outside the layer
  loop for layer-invariant weights (Wk, Wv are shared across layers in some designs), or
  stack across layers and feed the pre-loaded slices to each layer call, eliminating
  redundant HBM→SBUF DMAs for weights that don't change.

Calling conventions
-------------------
  - Back-compat mode (all weight kwargs = None): identical to v14c — alloc + DMA happens
    inside the function via sbm.alloc_stack + nisa.dma_copy.
  - Hoisted mode (some or all weight kwargs provided): caller is responsible for keeping
    the tensors live.  v14d will NOT pop (close_scope) / free them.

New kwargs (all default None)
-----------------------------
  wk_sb        : [PMAX, NH*d]           bf16 SBUF — skip Wk alloc+DMA when given
  wv_sb        : [PMAX, NH*d]           bf16 SBUF — skip Wv alloc+DMA when given
  wq_heads_sb  : list of len(owned_heads) [PMAX, NH*d] bf16 — skip Wq load loop when given
  wo_heads_sb  : list of len(owned_heads) [PMAX, H_wo] bf16 — skip Wo alloc+DMA loop when given

  Caller must order wq_heads_sb[i] / wo_heads_sb[i] to correspond to global head
  owned_heads[i] as computed internally from prg_id.

SBUF allocation
---------------
  All internal allocations still go through the caller-supplied sbm (SbufManager in
  manual/auto mode).  When the caller passes a pre-loaded tensor it must also have been
  allocated from the same sbm instance (or a compatible one).
"""

import math
import nki.language as nl
import nki.isa as nisa

PMAX = 128
F_MAX = 512
EPS = 1e-6
INV_SQRT_D = float(1.0 / math.sqrt(128.0))
PSUM_BANK_SIZE = 2048


def attn_fused_qwen(
    hidden_sb,           # [PMAX, num_h_tiles] = [128, 16]  bf16  SBUF
    Wq,                  # [Hq_tp*d, H]  = [1024, 2048]    bf16  HBM  tile-transposed
    Wk,                  # [Hkv_tp*d, H] = [128, 2048]     bf16  HBM  tile-transposed
    Wv,                  # [Hkv_tp*d, H] = [128, 2048]     bf16  HBM  tile-transposed
    Wo,                  # [Hq_tp*d, H_wo]= [1024, 2048]   bf16  HBM  plain .T layout
    q_norm_weight,       # [d]   bf16 HBM — used only if qnw_f32_sb is None
    k_norm_weight,       # [d]   bf16 HBM — used only if knw_f32_sb is None
    gamma_pre_attn,      # [H]   bf16 HBM — used only if gpan_f32_sb is None
    K_cache,             # [B, 1, S_prior, d]               bf16  HBM  mutated in-place
    V_cache,             # [B, 1, S_prior, d]               bf16  HBM  mutated in-place
    cos,                 # [B, d]        = [1, 128]         bf16  HBM — used only if cos_f32_sb is None
    sin,                 # [B, d]        = [1, 128]         bf16  HBM — used only if sin_f32_sb is None
    position_ids,        # [B, 1]        = [1, 1]           int32 HBM
    out_sb=None,         # optional [PMAX, H_wo//PMAX]      bf16  SBUF caller-provided
    sbm=None,            # required SbufManager
    # v14c: pre-loaded SBUF tensors (f32).  When provided, skip the DMA + bf16→f32 cast.
    qnw_f32_sb=None,     # [PMAX, 1]             f32  SBUF — replaces q_norm_weight load
    knw_f32_sb=None,     # [PMAX, 1]             f32  SBUF — replaces k_norm_weight load
    cos_f32_sb=None,     # [PMAX, B=1]           f32  SBUF — replaces cos load
    sin_f32_sb=None,     # [PMAX, B=1]           f32  SBUF — replaces sin load
    gpan_f32_sb=None,    # [PMAX, num_h_tiles=16] f32 SBUF — replaces gamma_pre_attn load
    # NEW v14d: pre-loaded weight SBUF tensors (bf16).
    wk_sb=None,          # [PMAX, NH*d]           bf16 SBUF — skip Wk alloc+DMA when given
    wv_sb=None,          # [PMAX, NH*d]           bf16 SBUF — skip Wv alloc+DMA when given
    wq_heads_sb=None,    # list of len(owned_heads) [PMAX, NH*d] bf16 — skip Wq load loop
    wo_heads_sb=None,    # list of len(owned_heads) [PMAX, H_wo] bf16 — skip Wo alloc+DMA loop
):
    """
    Fused pre-attn norm + QKV + RMSNorm + RoPE + KV scatter + flash decode + output projection.

    Weights Wq, Wk, Wv must be stored in tile-transposed layout:
      W_pt[head*d+p, tile*d+f] = W[head*d+f, tile*d+p]
    Produced by: W.reshape(n_heads, d, n_tiles, d).permute(0, 3, 2, 1).reshape(n_heads*d, H)

    When the *_f32_sb kwargs are provided (not None), the corresponding HBM load and
    bf16→f32 cast is skipped.  The caller owns those tensors and must keep them live.

    When wk_sb / wv_sb / wq_heads_sb / wo_heads_sb are provided, the corresponding
    HBM→SBUF weight DMAs are skipped.  The caller owns those tensors and must keep them live.
    """
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
    # OPEN attn_outer SCOPE
    # =========================================================================
    sbm.open_scope("attn_outer")

    K_cache_2d = K_cache.reshape((S_prior, d))
    V_cache_2d = V_cache.reshape((S_prior, d))

    # =========================================================================
    # EARLY K/V CACHE DMA — hoisted before all matmuls to hide HBM latency
    # =========================================================================
    k_cache_tiles_hbm = []
    v_cache_tiles_hbm = []

    for s_t in nl.affine_range(NUM_S_TILES):
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

    # =========================================================================
    # EARLY Wk/Wv DMA HOIST — conditionally skip when caller-provided
    # wk_sb[p, h_t*d+f_d] = Wk_pt[p, h_t*d+f_d] = Wk[f_d, h_t*d+p]
    # =========================================================================
    if wk_sb is None:
        wk_sb = sbm.alloc_stack((PMAX, NUM_H_TILES * d), nl.bfloat16, name="wk_sb")
        nisa.dma_copy(dst=wk_sb, src=Wk, dge_mode=nisa.dge_mode.hwdge)
    # else: use caller-provided wk_sb directly

    if wv_sb is None:
        wv_sb = sbm.alloc_stack((PMAX, NUM_H_TILES * d), nl.bfloat16, name="wv_sb")
        nisa.dma_copy(dst=wv_sb, src=Wv, dge_mode=nisa.dge_mode.hwdge)
    # else: use caller-provided wv_sb directly

    # =========================================================================
    # LOAD CONSTANTS — conditionally skip HBM load + f32 cast when pre-loaded
    # tensor is supplied by the caller.
    #
    # Convention:
    #   - When *_f32_sb is None  → allocate via sbm + DMA + cast (v14b behaviour)
    #   - When *_f32_sb provided → use directly; skip all allocations for this var
    # =========================================================================

    # --- q_norm_weight ---
    if qnw_f32_sb is None:
        # v14b path: load from HBM, cast to f32
        qnw_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="qnw_bf16")
        nisa.dma_copy(dst=qnw_bf16, src=q_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
        qnw_sb = sbm.alloc_stack((PMAX, 1), nl.float32, name="qnw_sb")
        nisa.tensor_copy(qnw_sb, qnw_bf16)
    else:
        # Hoisted path: use caller-provided f32 SBUF tensor directly
        qnw_sb = qnw_f32_sb

    # --- k_norm_weight ---
    if knw_f32_sb is None:
        knw_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="knw_bf16")
        nisa.dma_copy(dst=knw_bf16, src=k_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
        knw_sb = sbm.alloc_stack((PMAX, 1), nl.float32, name="knw_sb")
        nisa.tensor_copy(knw_sb, knw_bf16)
    else:
        knw_sb = knw_f32_sb

    # --- cos / sin ---
    if cos_f32_sb is None:
        cos_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="cos_bf16")
        sin_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="sin_bf16")
        nisa.dma_copy(dst=cos_bf16, src=cos_col, dge_mode=nisa.dge_mode.hwdge)
        nisa.dma_copy(dst=sin_bf16, src=sin_col, dge_mode=nisa.dge_mode.hwdge)
        cos_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name="cos_f32")
        nisa.tensor_copy(cos_f32, cos_bf16)
    else:
        cos_f32 = cos_f32_sb

    if sin_f32_sb is None:
        if cos_f32_sb is None:
            # Both were None — sin_bf16 already allocated above; reuse it.
            sin_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name="sin_f32")
            nisa.tensor_copy(sin_f32, sin_bf16)
        else:
            # cos was hoisted but sin was not — need to load sin independently.
            sin_bf16_indep = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="sin_bf16")
            nisa.dma_copy(dst=sin_bf16_indep, src=sin_col, dge_mode=nisa.dge_mode.hwdge)
            sin_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name="sin_f32")
            nisa.tensor_copy(sin_f32, sin_bf16_indep)
    else:
        sin_f32 = sin_f32_sb

    # --- gamma_pre_attn ---
    if gpan_f32_sb is None:
        gpan_bf16 = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="gpan_bf16")
        nisa.dma_copy(dst=gpan_bf16,
                      src=gamma_pre_attn.reshape((H, 1)).ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
                      dge_mode=nisa.dge_mode.hwdge)
        gpan_f32 = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="gpan_f32")
        nisa.tensor_copy(gpan_f32, gpan_bf16)
    else:
        gpan_f32 = gpan_f32_sb

    # --- position_ids (always loaded; tiny and positional, not layer-invariant) ---
    pos_write_i32_raw = sbm.alloc_stack((1, 1), nl.int32, name="pos_write_i32_raw")
    nisa.dma_copy(dst=pos_write_i32_raw,
                  src=position_ids.reshape((B, 1))[0:1, 0:1],
                  dge_mode=nisa.dge_mode.hwdge)
    pos_write_i32 = sbm.alloc_stack((1, 1), nl.uint32, name="pos_write_i32")
    nisa.tensor_copy(pos_write_i32, pos_write_i32_raw)

    # =========================================================================
    # h_all: shell allocated at attn_outer, filled by pre_attn_norm
    # =========================================================================
    h_all = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="h_all")

    # =========================================================================
    # SHARED RMSNorm ISA constants — allocated once, shared across 3 sites.
    # Match neuronxcc.nki._pre_prod_kernels.rmsnorm_tkg ISA constants exactly.
    # =========================================================================
    rms_zero_bias = sbm.alloc_stack((PMAX, 1), nl.float32, name="rms_zero_bias")
    nisa.memset(rms_zero_bias, value=0.0)
    rms_ones = sbm.alloc_stack((PMAX, PMAX), nl.float32, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)
    rms_eps_sb = sbm.alloc_stack((PMAX, 1), nl.float32, name="rms_eps_sb")
    nisa.memset(rms_eps_sb, value=EPS)

    # =====================================================================
    # PRE-ATTENTION RMSNORM — library ISA sequence
    # Input: hidden_sb bf16 [PMAX, num_h_tiles=16], H_total=2048
    # Changes from pre-R9:
    #   - activation(square, bias=zero) for squaring (scalar engine, single instruction)
    #   - fused activation(rsqrt, scale=1/H, bias=eps_sb) (library ISA, single instruction)
    # Cross-partition sum: nc_transpose chain (same as pre-R9)
    # =====================================================================
    sbm.open_scope("pre_attn_norm")

    # Load input to fp32
    h_f32 = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_f32")
    nisa.tensor_copy(h_f32, hidden_sb)

    # Step 1: Square via scalar-engine activation (library ISA)
    h_sq = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_sq")
    nisa.activation(h_sq, op=nl.square, data=h_f32, bias=rms_zero_bias)

    # Cross-partition sum via nc_transpose chain (as pre-R9)
    # 1. Transpose h_sq [PMAX, num_h_tiles] → [num_h_tiles, PMAX]
    h_sq_T_psum = nl.ndarray((num_h_tiles, PMAX), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_sq_T_psum, h_sq)
    h_sq_T_sb = sbm.alloc_stack((num_h_tiles, PMAX), nl.float32, name="h_sq_T_sb")
    nisa.tensor_copy(h_sq_T_sb, h_sq_T_psum)
    # 2. Sum over PMAX (free dim) → [num_h_tiles, 1]
    h_sq_sum_tiles = sbm.alloc_stack((num_h_tiles, 1), nl.float32, name="h_sq_sum_tiles")
    nisa.tensor_reduce(h_sq_sum_tiles, op=nl.add, data=h_sq_T_sb, axis=1)
    # 3. Transpose [num_h_tiles, 1] → [1, num_h_tiles] then sum → scalar [1, 1]
    h_sq_sum_tiles_T_psum = nl.ndarray((1, num_h_tiles), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_sq_sum_tiles_T_psum, h_sq_sum_tiles)
    h_sq_sum_tiles_T_sb = sbm.alloc_stack((1, num_h_tiles), nl.float32, name="h_sq_sum_tiles_T_sb")
    nisa.tensor_copy(h_sq_sum_tiles_T_sb, h_sq_sum_tiles_T_psum)
    h_sq_scalar = sbm.alloc_stack((1, 1), nl.float32, name="h_sq_scalar")
    nisa.tensor_reduce(h_sq_scalar, op=nl.add, data=h_sq_sum_tiles_T_sb, axis=1)
    # 4. Broadcast scalar [1, 1] → [PMAX, 1]
    h_sq_total_psum = nl.ndarray((PMAX, 1), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_sq_total_psum, h_sq_scalar.ap([[1, 1], [0, PMAX]], offset=0))
    h_sq_total = sbm.alloc_stack((PMAX, 1), nl.float32, name="h_sq_total")
    nisa.tensor_copy(h_sq_total, h_sq_total_psum)

    # Fused scale+bias+rsqrt in one scalar-engine instruction (library ISA)
    # rsqrt(total_sum * (1/H) + eps) where H=2048
    h_rms_inv = sbm.alloc_stack((PMAX, 1), nl.float32, name="h_rms_inv")
    nisa.activation(h_rms_inv, op=nl.rsqrt, data=h_sq_total, scale=1.0/H, bias=rms_eps_sb)

    # Broadcast rms_inv [PMAX,1] → [PMAX, num_h_tiles] via nc_transpose pattern
    h_rms_T_psum = nl.ndarray((num_h_tiles, PMAX), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_rms_T_psum, h_rms_inv.ap([[1, PMAX], [0, num_h_tiles]], offset=0))
    h_rms_T = sbm.alloc_stack((num_h_tiles, PMAX), nl.float32, name="h_rms_T")
    nisa.tensor_copy(h_rms_T, h_rms_T_psum)
    h_rms_expanded_psum = nl.ndarray((PMAX, num_h_tiles), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_rms_expanded_psum, h_rms_T)
    h_rms_expanded = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_rms_expanded")
    nisa.tensor_copy(h_rms_expanded, h_rms_expanded_psum)

    # Multiply x by rms_inv, apply gamma (same order as pre-R9)
    h_normed = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_normed")
    nisa.tensor_tensor(h_normed, h_f32, h_rms_expanded, op=nl.multiply)
    nisa.tensor_tensor(h_normed, h_normed, gpan_f32, op=nl.multiply)
    nisa.tensor_copy(h_all, h_normed)

    sbm.close_scope()  # pre_attn_norm

    # =========================================================================
    # LNC sharding
    # =========================================================================
    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)

    if n_prgs == 1:
        owned_heads = [0, 1, 2, 3, 4, 5, 6, 7]
    elif prg_id == 0:
        owned_heads = [0, 1, 2, 3]
    else:
        owned_heads = [4, 5, 6, 7]

    # =========================================================================
    # Wq LOAD — conditionally skip when caller-provided wq_heads_sb
    # Wq_pt[q_h*PMAX:(q_h+1)*PMAX, :] is [PMAX, H] and maps directly to
    # wq_head_sb[p, h_t*d+f_d] = Wq[q_h*d+f_d, h_t*PMAX+p] (same as dma_transpose would give)
    # =========================================================================
    wq_head_sb = [None] * HQ_TP_CONST
    if wq_heads_sb is None:
        for q_h in owned_heads:
            w = sbm.alloc_stack((PMAX, NUM_H_TILES * d), nl.bfloat16, name=f"wq_head_{q_h}")
            nisa.dma_copy(
                dst=w,
                src=Wq[q_h * PMAX:(q_h + 1) * PMAX, :],
                dge_mode=nisa.dge_mode.hwdge,
            )
            wq_head_sb[q_h] = w
    else:
        assert len(wq_heads_sb) == len(owned_heads), (
            f"wq_heads_sb length {len(wq_heads_sb)} must match owned_heads length {len(owned_heads)}"
        )
        for _idx in range(len(owned_heads)):
            wq_head_sb[owned_heads[_idx]] = wq_heads_sb[_idx]

    Wo_reshaped = Wo.reshape((Hq_tp, d, H_wo))

    # =========================================================================
    # Persistent tensors at attn_outer level
    # =========================================================================
    k_rope = sbm.alloc_stack((PMAX, B), nl.float32, name="k_rope")
    k_rope_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_rope_bf16")
    k_rope_T_sb = sbm.alloc_stack((B, PMAX), nl.bfloat16, name="k_rope_T_sb")
    v_active = sbm.alloc_stack((PMAX, B), nl.float32, name="v_active")
    v_T_sb = sbm.alloc_stack((B, PMAX), nl.bfloat16, name="v_T_sb")
    q_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_bf16")

    k_rope_bf16_f32 = sbm.alloc_stack((PMAX, B), nl.float32, name="k_rope_bf16_f32")
    v_from_bf16 = sbm.alloc_stack((PMAX, B), nl.float32, name="v_from_bf16")

    # Wo allocation — conditionally skip when caller-provided wo_heads_sb
    wo_sbuf = [None] * HQ_TP_CONST
    if wo_heads_sb is None:
        for head in owned_heads:
            wo_tile = sbm.alloc_stack((PMAX, H_wo), nl.bfloat16, name=f"wo_tile_h{head}")
            wo_sbuf[head] = wo_tile
    else:
        assert len(wo_heads_sb) == len(owned_heads), (
            f"wo_heads_sb length {len(wo_heads_sb)} must match owned_heads length {len(owned_heads)}"
        )
        for _idx in range(len(owned_heads)):
            wo_sbuf[owned_heads[_idx]] = wo_heads_sb[_idx]

    # =========================================================================
    # KV PROJ SCOPE
    # =========================================================================
    sbm.open_scope("kv_proj")

    k_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum)
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(k_psum, stationary=wk_sb[0:PMAX, h_t*d:(h_t+1)*d], moving=h_all[0:PMAX, h_t:h_t+1])

    k_vec = sbm.alloc_stack((PMAX, B), nl.float32, name="k_vec")
    nisa.tensor_copy(k_vec, k_psum)

    # Match PyTorch baseline: K-norm receives bf16 from the Wk matmul (ColumnParallelLinear
    # outputs bf16; CustomRMSNorm re-casts to fp32 internally). We emulate the same bf16
    # quantization by round-tripping the fp32 PSUM result through bf16 before the RMS math.
    k_vec_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_vec_bf16")
    nisa.tensor_copy(k_vec_bf16, k_vec)
    nisa.tensor_copy(k_vec, k_vec_bf16)

    # K RMSNorm — library ISA sequence (H=128=d, H_free=B=1)
    # Changes from pre-R9: activation(square) for squaring, fused activation(rsqrt)
    # Cross-partition sum: nc_transpose chain (as pre-R9, B=1)
    # Step 1: Square via scalar-engine activation (library ISA)
    k_sq = sbm.alloc_stack((PMAX, B), nl.float32, name="k_sq")
    nisa.activation(k_sq, op=nl.square, data=k_vec, bias=rms_zero_bias)
    # Cross-partition sum via nc_transpose chain (pre-R9 style, adapted for B=1)
    k_sq_T_psum = nl.ndarray((B, PMAX), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(k_sq_T_psum, k_sq)
    k_sq_T_sb = sbm.alloc_stack((B, PMAX), nl.float32, name="k_sq_T_sb")
    nisa.tensor_copy(k_sq_T_sb, k_sq_T_psum)
    k_sq_scalar = sbm.alloc_stack((B, 1), nl.float32, name="k_sq_scalar")
    nisa.tensor_reduce(k_sq_scalar, op=nl.add, data=k_sq_T_sb, axis=1)
    k_sum_psum = nl.ndarray((PMAX, B), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(k_sum_psum, k_sq_scalar.ap([[1, B], [0, PMAX]], offset=0))
    k_sum_sb = sbm.alloc_stack((PMAX, 1), nl.float32, name="k_sum_sb")
    nisa.tensor_copy(k_sum_sb, k_sum_psum)
    # Fused scale+bias+rsqrt in one scalar-engine instruction (library ISA, H=d=128)
    k_rms_inv = sbm.alloc_stack((PMAX, 1), nl.float32, name="k_rms_inv")
    nisa.activation(k_rms_inv, op=nl.rsqrt, data=k_sum_sb, scale=1.0/d, bias=rms_eps_sb)
    # Multiply k_vec by rms_inv, then apply gamma
    k_normed = sbm.alloc_stack((PMAX, B), nl.float32, name="k_normed")
    nisa.tensor_tensor(k_normed, k_vec, k_rms_inv, op=nl.multiply)
    k_normed2 = sbm.alloc_stack((PMAX, B), nl.float32, name="k_normed2")
    nisa.tensor_tensor(k_normed2, k_normed, knw_sb, op=nl.multiply)

    # Round k_normed2 through bf16 to match PyTorch's cast before RoPE
    k_normed2_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_normed2_bf16")
    nisa.tensor_copy(k_normed2_bf16, k_normed2)
    nisa.tensor_copy(k_normed2, k_normed2_bf16)

    # K RoPE — all ops in bf16 to match PyTorch reference
    cos_bf16_rope = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="cos_bf16_rope")
    nisa.tensor_copy(cos_bf16_rope, cos_f32)
    sin_bf16_rope = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="sin_bf16_rope")
    nisa.tensor_copy(sin_bf16_rope, sin_f32)

    rot_k = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="rot_k")
    neg_k_upper = sbm.alloc_stack((half_d, B), nl.bfloat16, name="neg_k_upper")
    nisa.tensor_scalar(neg_k_upper, k_normed2_bf16[half_d:d, 0:B], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(rot_k[0:half_d, 0:B], neg_k_upper)
    nisa.tensor_copy(rot_k[half_d:d, 0:B], k_normed2_bf16[0:half_d, 0:B])
    k_cos = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_cos")
    k_sin_part = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_sin_part")
    nisa.tensor_tensor(k_cos, k_normed2_bf16, cos_bf16_rope, op=nl.multiply)
    nisa.tensor_tensor(k_sin_part, rot_k, sin_bf16_rope, op=nl.multiply)
    nisa.tensor_tensor(k_rope_bf16, k_cos, k_sin_part, op=nl.add)
    nisa.tensor_copy(k_rope, k_rope_bf16)
    k_rope_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(k_rope_T_psum, k_rope_bf16)
    nisa.tensor_copy(k_rope_T_sb, k_rope_T_psum)
    nisa.tensor_copy(k_rope_bf16_f32, k_rope_bf16)

    # V projection
    v_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum)
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(v_psum, stationary=wv_sb[0:PMAX, h_t*d:(h_t+1)*d], moving=h_all[0:PMAX, h_t:h_t+1])

    nisa.tensor_copy(v_active, v_psum)

    v_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="v_bf16")
    nisa.tensor_copy(v_bf16, v_active)
    v_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(v_T_psum, v_bf16)
    nisa.tensor_copy(v_T_sb, v_T_psum)
    nisa.tensor_copy(v_from_bf16, v_bf16)

    sbm.close_scope()  # kv_proj

    # =========================================================================
    # Q PROJ SCOPE — PSUM bank interleaving to eliminate RAW stalls
    # =========================================================================
    sbm.open_scope("q_proj")

    q_psums = []
    for i in range(len(owned_heads)):
        q_h = owned_heads[i]
        q_p = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(q_p, value=0.0)
        q_psums.append(q_p)

    for h_t in nl.affine_range(NUM_H_TILES):
        for i in range(len(owned_heads)):
            q_h = owned_heads[i]
            nisa.nc_matmul(q_psums[i],
                           stationary=wq_head_sb[q_h][0:PMAX, h_t * d:(h_t + 1) * d],
                           moving=h_all[0:PMAX, h_t:h_t + 1])

    q_packed_f32 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_packed_f32")
    nisa.memset(q_packed_f32, value=0.0)
    for i in range(len(owned_heads)):
        q_h = owned_heads[i]
        nisa.tensor_copy(q_packed_f32[0:PMAX, q_h:q_h + 1], q_psums[i])

    # Match PyTorch baseline: Q-norm receives bf16 from the Wq matmul. Quantize the fp32
    # PSUM result through bf16 before the RMS math (same reasoning as K path).
    q_packed_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_packed_bf16")
    nisa.tensor_copy(q_packed_bf16, q_packed_f32)
    nisa.tensor_copy(q_packed_f32, q_packed_bf16)

    # Q RMSNorm — library ISA sequence (H=128=d, H_free=GQA)
    # First, broadcast qnw_sb [PMAX,1] → [PMAX,GQA] (needed before gamma multiply)
    qnw_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(qnw_gqa_psum_T, qnw_sb.ap([[1, PMAX], [0, GQA]], offset=0))
    qnw_gqa_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="qnw_gqa_sbuf_T")
    nisa.tensor_copy(qnw_gqa_sbuf_T, qnw_gqa_psum_T)
    qnw_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(qnw_gqa_psum, qnw_gqa_sbuf_T)
    qnw_gqa = sbm.alloc_stack((PMAX, GQA), nl.float32, name="qnw_gqa")
    nisa.tensor_copy(qnw_gqa, qnw_gqa_psum)

    # Q RMSNorm — library ISA sequence (H=128=d, H_free=GQA)
    # Changes from pre-R9: activation(square) for squaring, fused activation(rsqrt)
    # Cross-partition sum: nc_transpose chain (as pre-R9)
    # Step 1: Square via scalar-engine activation (library ISA)
    q_sq = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_sq")
    nisa.activation(q_sq, op=nl.square, data=q_packed_f32, bias=rms_zero_bias)
    # Cross-partition sum via nc_transpose chain (as pre-R9)
    q_sq_T_psum = nl.ndarray((GQA, PMAX), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(q_sq_T_psum, q_sq)
    q_sq_T_sb = sbm.alloc_stack((GQA, PMAX), nl.float32, name="q_sq_T_sb")
    nisa.tensor_copy(q_sq_T_sb, q_sq_T_psum)
    q_sq_sums = sbm.alloc_stack((GQA, 1), nl.float32, name="q_sq_sums")
    nisa.tensor_reduce(q_sq_sums, op=nl.add, data=q_sq_T_sb, axis=1)
    q_sum_psum = nl.ndarray((PMAX, GQA), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(q_sum_psum, q_sq_sums.ap([[1, GQA], [0, PMAX]], offset=0))
    q_sum_sb = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_sum_sb")
    nisa.tensor_copy(q_sum_sb, q_sum_psum)
    # q_sum_sb [PMAX, GQA] has the cross-partition sum per Q head.
    # But for RMSNorm we need one scalar per token (all heads share), not per head.
    # Take first column (all GQA partitions hold same sum per original pre-R9 logic):
    # Actually pre-R9 had per-head sum. Let me use the same path.
    # The mean_sq is over d=128 elements (one full head), not all GQA heads.
    # q_sum_sb holds sum_{partition}(q_sq[p, g]) for each g. We need rsqrt(sum/d + eps).
    # Fused scale+bias+rsqrt (library ISA, H=d=128)
    q_rms_inv = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_rms_inv")
    nisa.activation(q_rms_inv, op=nl.rsqrt, data=q_sum_sb, scale=1.0/d, bias=rms_eps_sb)
    # Multiply q by rms_inv (element-wise, [PMAX,GQA] × [PMAX,GQA]), then apply gamma
    q_normed = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_normed")
    nisa.tensor_tensor(q_normed, q_packed_f32, q_rms_inv, op=nl.multiply)
    q_normed2 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_normed2")
    nisa.tensor_tensor(q_normed2, q_normed, qnw_gqa, op=nl.multiply)

    # Round q_normed2 through bf16 to match PyTorch's cast before RoPE
    q_normed2_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_normed2_bf16")
    nisa.tensor_copy(q_normed2_bf16, q_normed2)
    nisa.tensor_copy(q_normed2, q_normed2_bf16)

    # Q RoPE — all ops in bf16 to match PyTorch reference
    cos_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(cos_gqa_psum_T, cos_f32.ap([[1, PMAX], [0, GQA]], offset=0))
    cos_gqa_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="cos_gqa_sbuf_T")
    nisa.tensor_copy(cos_gqa_sbuf_T, cos_gqa_psum_T)
    cos_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(cos_gqa_psum, cos_gqa_sbuf_T)
    cos_gqa_f32 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="cos_gqa_f32")
    nisa.tensor_copy(cos_gqa_f32, cos_gqa_psum)
    cos_gqa = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="cos_gqa")
    nisa.tensor_copy(cos_gqa, cos_gqa_f32)

    sin_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(sin_gqa_psum_T, sin_f32.ap([[1, PMAX], [0, GQA]], offset=0))
    sin_gqa_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="sin_gqa_sbuf_T")
    nisa.tensor_copy(sin_gqa_sbuf_T, sin_gqa_psum_T)
    sin_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(sin_gqa_psum, sin_gqa_sbuf_T)
    sin_gqa_f32 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="sin_gqa_f32")
    nisa.tensor_copy(sin_gqa_f32, sin_gqa_psum)
    sin_gqa = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="sin_gqa")
    nisa.tensor_copy(sin_gqa, sin_gqa_f32)

    rot_q = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="rot_q")
    neg_q_upper = sbm.alloc_stack((half_d, GQA), nl.bfloat16, name="neg_q_upper")
    nisa.tensor_scalar(neg_q_upper, q_normed2_bf16[half_d:d, 0:GQA], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(rot_q[0:half_d, 0:GQA], neg_q_upper)
    nisa.tensor_copy(rot_q[half_d:d, 0:GQA], q_normed2_bf16[0:half_d, 0:GQA])

    q_cos = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_cos")
    q_sin_part = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_sin_part")
    nisa.tensor_tensor(q_cos, q_normed2_bf16, cos_gqa, op=nl.multiply)
    nisa.tensor_tensor(q_sin_part, rot_q, sin_gqa, op=nl.multiply)
    q_rope = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_rope")
    nisa.tensor_tensor(q_rope, q_cos, q_sin_part, op=nl.add)

    # R2: softmax scale is applied post-matmul on scores (not pre-matmul on Q),
    # matching baseline's `matmul(Q, K) / sqrt(128)`. Keep q_bf16 = q_rope unchanged here.
    nisa.tensor_copy(q_bf16, q_rope)

    sbm.close_scope()  # q_proj

    # =========================================================================
    # WO WEIGHT HOISTING — conditionally skip DMA when caller-provided wo_heads_sb
    # =========================================================================
    if wo_heads_sb is None:
        for head in owned_heads:
            nisa.dma_copy(
                dst=wo_sbuf[head],
                src=Wo_reshaped.ap(
                    pattern=[[H_wo, PMAX], [1, H_wo]],
                    offset=head * PMAX * H_wo,
                ),
                dge_mode=nisa.dge_mode.hwdge,
            )
    # else: wo_sbuf already bound to caller-provided tensors above

    # =========================================================================
    # TWO-PASS FLASH DECODE
    # =========================================================================
    # Active score: k_rope_bf16.T @ q_bf16 → [1, GQA] (fp32-accumulated matmul matches ref MPA)
    score_act_1_psum = nl.zeros((1, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(score_act_1_psum, stationary=k_rope_bf16, moving=q_bf16)
    # spec §0.4 / §2 Step 7b: explicit CONVERT F32 PSUM → BF16 at PE boundary
    score_act_1_bf16 = sbm.alloc_stack((1, GQA), nl.bfloat16, name="score_act_1_bf16")
    nisa.tensor_copy(score_act_1_bf16, score_act_1_psum)
    # spec §2 Step 7b: scale in BF16
    score_act_1_scaled_bf16 = sbm.alloc_stack((1, GQA), nl.bfloat16, name="score_act_1_scaled_bf16")
    nisa.tensor_scalar(score_act_1_scaled_bf16, score_act_1_bf16, op0=nl.multiply, operand0=INV_SQRT_D)
    # spec §5.5: immediate upcast to F32 before broadcast/mask
    score_act_1_sb = sbm.alloc_stack((1, GQA), nl.float32, name="score_act_1_sb")
    nisa.tensor_copy(score_act_1_sb, score_act_1_scaled_bf16)
    # Broadcast [1, GQA] → [PMAX, GQA] via transpose-replicate (matches neg_max broadcast pattern)
    score_act_g_psum = nl.ndarray((GQA, 1), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(score_act_g_psum, score_act_1_sb)
    score_act_g_sb = sbm.alloc_stack((GQA, 1), nl.float32, name="score_act_g_sb")
    nisa.tensor_copy(score_act_g_sb, score_act_g_psum)
    score_active_psum = nl.ndarray((PMAX, GQA), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(score_active_psum, score_act_g_sb.ap([[1, GQA], [0, PMAX]], offset=0))
    score_active = sbm.alloc_stack((PMAX, GQA), nl.float32, name="score_active")
    nisa.tensor_copy(score_active, score_active_psum)

    position_ids_2d = position_ids.reshape((B, 1))
    pos_f32 = sbm.alloc_stack((1, 1), nl.float32, name="pos_f32")
    nisa.tensor_copy(pos_f32, pos_write_i32)

    par_index_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name="par_index_f32")
    nisa.iota(par_index_f32, pattern=[[1, 1]], offset=0, channel_multiplier=1)

    # =========================================================================
    # KV HOIST SCOPE
    # =========================================================================
    sbm.open_scope("kv_hoist")

    k_cache_tiles = []
    mask_gqa_tiles = []
    v_cache_tiles = []

    for s_t in nl.affine_range(NUM_S_TILES):
        k_ct = k_cache_tiles_hbm[s_t]

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
        # spec §5.5: mask sentinel is torch.finfo(torch.float32).min ≈ -3.4028235e38
        nisa.tensor_scalar(mask_tile_f32, clamped, op0=nl.multiply, operand0=-3.4028234663852886e38)

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
        v_cache_tiles.append(v_cache_tiles_hbm[s_t])

    # =========================================================================
    # PASS 1
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

    saved_scores = []

    for s_t in nl.affine_range(NUM_S_TILES):
        score_sb_masked = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"score_sb_masked_{s_t}")

        sbm.open_scope(f"p1_tile_{s_t}")

        score_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(score_psum, stationary=k_cache_tiles[s_t], moving=q_bf16)
        # spec §0.4 / §2 Step 7a: explicit CONVERT F32 PSUM → BF16 at PE boundary
        score_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name=f"score_bf16_{s_t}")
        nisa.tensor_copy(score_bf16, score_psum)
        # spec §2 Step 7a: scale in BF16 (`prior_scores_bf16 * softmax_scale`)
        score_scaled_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name=f"score_scaled_bf16_{s_t}")
        nisa.tensor_scalar(score_scaled_bf16, score_bf16, op0=nl.multiply, operand0=INV_SQRT_D)
        # spec §5.5: immediate upcast to F32 before mask
        score_sb = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"score_sb_{s_t}")
        nisa.tensor_copy(score_sb, score_scaled_bf16)

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
    # PASS 2 — two sub-passes to match baseline bf16 softmax quantization
    # =========================================================================
    sbm.open_scope("pass2")

    # ---- Sub-pass 2a: compute all exp tiles (fp32), accumulate fp32 denom [GQA,1] ----

    sum_acc_gqa = sbm.alloc_stack((GQA, 1), nl.float32, name="sum_acc_gqa")
    nisa.memset(sum_acc_gqa, value=0.0)

    exp_tiles = []

    for s_t in nl.affine_range(NUM_S_TILES):
        score2_shifted = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"score2_shifted_{s_t}")
        nisa.tensor_tensor(score2_shifted, saved_scores[s_t], neg_max, op=nl.add)

        score2_exp = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"score2_exp_{s_t}")
        nisa.activation(score2_exp, op=nl.exp, data=score2_shifted)
        exp_tiles.append(score2_exp)

        # Reduce exp tile [PMAX, GQA] → [GQA, 1] via transpose + tensor_reduce
        exp_T_psum = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(exp_T_psum, score2_exp)
        exp_T_sb = sbm.alloc_stack((GQA, PMAX), nl.float32, name=f"exp_T_sb_{s_t}")
        nisa.tensor_copy(exp_T_sb, exp_T_psum)
        tile_sum_gqa = sbm.alloc_stack((GQA, 1), nl.float32, name=f"tile_sum_gqa_{s_t}")
        nisa.tensor_reduce(dst=tile_sum_gqa, op=nl.add, data=exp_T_sb, axis=1)
        nisa.tensor_tensor(sum_acc_gqa, sum_acc_gqa, tile_sum_gqa, op=nl.add)

    # Active position exp contribution
    score_act_shifted = sbm.alloc_stack((PMAX, GQA), nl.float32, name="score_act_shifted")
    nisa.tensor_tensor(score_act_shifted, score_active, neg_max, op=nl.add)
    score_act_exp = sbm.alloc_stack((PMAX, GQA), nl.float32, name="score_act_exp")
    nisa.activation(score_act_exp, op=nl.exp, data=score_act_shifted)

    # score_act_exp is broadcast (all PMAX partitions hold same value per head).
    # Transpose → [GQA, PMAX], then take first column [GQA, 1] to get per-head exp value.
    score_act_exp_T_psum = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(score_act_exp_T_psum, score_act_exp)
    score_act_exp_T_sb = sbm.alloc_stack((GQA, PMAX), nl.float32, name="score_act_exp_T_sb")
    nisa.tensor_copy(score_act_exp_T_sb, score_act_exp_T_psum)
    exp_active_gqa = sbm.alloc_stack((GQA, 1), nl.float32, name="exp_active_gqa")
    nisa.tensor_copy(exp_active_gqa, score_act_exp_T_sb[0:GQA, 0:1])
    nisa.tensor_tensor(sum_acc_gqa, sum_acc_gqa, exp_active_gqa, op=nl.add)

    # ---- Reciprocal of denom: Vector RECIPROCAL (matches HLO `divide` BIR lowering, spec §0.2/§5.6) ----
    inv_denom_gqa = sbm.alloc_stack((GQA, 1), nl.float32, name="inv_denom_gqa")
    nisa.reciprocal(inv_denom_gqa, sum_acc_gqa)

    # Broadcast inv_denom_gqa [GQA, 1] → [PMAX, GQA] (same pattern as neg_max broadcast)
    inv_denom_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(
        inv_denom_psum,
        inv_denom_gqa.ap([[1, GQA], [0, PMAX]], offset=0),
    )
    inv_denom_bcast = sbm.alloc_stack((PMAX, GQA), nl.float32, name="inv_denom_bcast")
    nisa.tensor_copy(inv_denom_bcast, inv_denom_psum)

    # ---- Sub-pass 2b: normalize, cast to bf16, V-matmul ----

    v_acc = sbm.alloc_stack((PMAX, GQA), nl.float32, name="v_acc")
    nisa.memset(v_acc, value=0.0)

    for s_t in nl.affine_range(NUM_S_TILES):
        # softmax tile: fp32 multiply then cast to bf16
        softmax_fp32 = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"softmax_fp32_{s_t}")
        nisa.tensor_tensor(softmax_fp32, exp_tiles[s_t], inv_denom_bcast, op=nl.multiply)
        softmax_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name=f"softmax_bf16_{s_t}")
        nisa.tensor_copy(softmax_bf16, softmax_fp32)

        # V-matmul: bf16 softmax × bf16 V_cache → fp32 PSUM
        v_weighted_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(v_weighted_psum, stationary=v_cache_tiles[s_t], moving=softmax_bf16)
        v_weighted = sbm.alloc_stack((PMAX, GQA), nl.float32, name=f"v_weighted_{s_t}")
        nisa.tensor_copy(v_weighted, v_weighted_psum)
        nisa.tensor_tensor(v_acc, v_acc, v_weighted, op=nl.add)

    # Active: softmax_active = score_act_exp * inv_denom in fp32, cast bf16
    softmax_act_fp32 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="softmax_act_fp32")
    nisa.tensor_tensor(softmax_act_fp32, score_act_exp, inv_denom_bcast, op=nl.multiply)
    softmax_act_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="softmax_act_bf16")
    nisa.tensor_copy(softmax_act_bf16, softmax_act_fp32)

    # Broadcast V_active from [PMAX, B=1] to [PMAX, GQA] in fp32, then cast to bf16
    v_act_packed_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(v_act_packed_psum_T, v_from_bf16.ap([[1, PMAX], [0, GQA]], offset=0))
    v_act_packed_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="v_act_packed_sbuf_T")
    nisa.tensor_copy(v_act_packed_sbuf_T, v_act_packed_psum_T)
    v_act_packed_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(v_act_packed_psum, v_act_packed_sbuf_T)
    v_act_packed = sbm.alloc_stack((PMAX, GQA), nl.float32, name="v_act_packed")
    nisa.tensor_copy(v_act_packed, v_act_packed_psum)
    v_act_packed_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="v_act_packed_bf16")
    nisa.tensor_copy(v_act_packed_bf16, v_act_packed)

    # spec §2 Step 9 / §5.7: prior matmul output is BF16 — CONVERT F32 sum → BF16
    attn_prior_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="attn_prior_bf16")
    nisa.tensor_copy(attn_prior_bf16, v_acc)

    # bf16 * bf16 → fp32 (matches PE F32 PSUM for the active matmul)
    v_act_weighted_fp32 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="v_act_weighted_fp32")
    nisa.tensor_tensor(v_act_weighted_fp32, v_act_packed_bf16, softmax_act_bf16, op=nl.multiply)

    # spec §2 Step 9 / §5.7: active matmul output is BF16 — CONVERT F32 → BF16
    attn_active_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="attn_active_bf16")
    nisa.tensor_copy(attn_active_bf16, v_act_weighted_fp32)

    # spec §2 Step 9: attn_output = attn_prior + attn_active in BF16
    nisa.tensor_tensor(attn_out, attn_prior_bf16, attn_active_bf16, op=nl.add)

    sbm.close_scope()  # pass2

    # =========================================================================
    # O_PROJ SCOPE
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
        nisa.nc_matmul(res_psum_0, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 0*F_MAX:1*F_MAX])
        nisa.nc_matmul(res_psum_1, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 1*F_MAX:2*F_MAX])
        nisa.nc_matmul(res_psum_2, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 2*F_MAX:3*F_MAX])
        nisa.nc_matmul(res_psum_3, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 3*F_MAX:4*F_MAX])

    # Transpose each 128-column chunk — 4 blocks x 4 chunks = 16 columns
    nisa.tensor_copy(out_sb_tmp, res_psum_0)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 0*PMAX:1*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 0:1], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 1*PMAX:2*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 1:2], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 2*PMAX:3*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 2:3], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 3*PMAX:4*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 3:4], col_tmp)

    nisa.tensor_copy(out_sb_tmp, res_psum_1)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 0*PMAX:1*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 4:5], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 1*PMAX:2*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 5:6], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 2*PMAX:3*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 6:7], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 3*PMAX:4*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 7:8], col_tmp)

    nisa.tensor_copy(out_sb_tmp, res_psum_2)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 0*PMAX:1*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 8:9], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 1*PMAX:2*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 9:10], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 2*PMAX:3*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 10:11], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 3*PMAX:4*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 11:12], col_tmp)

    nisa.tensor_copy(out_sb_tmp, res_psum_3)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 0*PMAX:1*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 12:13], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 1*PMAX:2*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 13:14], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 2*PMAX:3*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 14:15], col_tmp)
    nisa.nc_transpose(chunk_T_psum, out_sb_tmp[0:1, 3*PMAX:4*PMAX])
    nisa.tensor_copy(col_tmp, chunk_T_psum); nisa.tensor_copy(out_sb[0:PMAX, 15:16], col_tmp)

    # =========================================================================
    # CROSS-CORE ALL-REDUCE (LNC=2)
    # =========================================================================
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
    # KV CACHE IN-PLACE SCATTER
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