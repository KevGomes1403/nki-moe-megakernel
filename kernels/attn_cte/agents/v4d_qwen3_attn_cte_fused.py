"""
Fused NKI attention CTE kernel for Qwen3.
Fuses: QKV linear projection + per-head RMSNorm + RoPE + causal flash attention.

v4d architecture (v4c + v4b hidden preload + two-pass softmax):
  Builds on v4c (pre-transposed K, 512-wide MM1, affine_select masking).

  v4b addition: preload all 80 hidden tiles into SBUF before loops.
    Eliminates 1,280 HBM hidden loads -> 80 (merged K+V ht loop).

  Two-pass softmax replaces flash attention:
    Pass 1: compute scores_all[PMAX, S_TOTAL=640] via 512-wide MM1 + affine_select.
    Pass 2: ONE broadcast of gmax[PMAX,1]->[PMAX,640] + exp + ONE broadcast of inv_sum.
    Pass 3: MM2 via 5 V-tile PSUM accumulations.
    DMA reduction: from 3 SBUF broadcasts per ki_batch (6 per qsi) -> 2 per qsi total.
    SBUF extra: scores_all[128,640] f32 = 320KB + attn_w[128,640] = 320KB per Q block.

Plan A optimization (Phase Fusion):
  Eliminates K/V staging through private_hbm entirely.

  v1c architecture (two separate phases):
    Phase 1: for si: compute K/V -> store to K_scratch/V_scratch (private_hbm)
    Phase 2: for kvh -> gi -> qsi -> ki: load K/V from K_scratch/V_scratch (private_hbm)
    Cost: ~440 HBM DMA ops (~14MB at 32KB/tile) purely for K/V staging.

  v2a architecture (single fused kvh loop):
    for kvh (affine_range):
      Phase 1 sub-loop: for si in range(num_s_tiles):
        compute K tile (krbf) and V tile (vbf) -> append to K_tiles[si], V_tiles[si]
      Phase 2 sub-loop: for gi (affine_range) -> for qsi (affine_range):
        flash attention uses K_tiles[ki] / V_tiles[ki] directly from SBUF (no HBM DMA)

  v3a architecture (v2a + full weight + cos/sin hoisting):
    Eliminates redundant weight and cos/sin DMA loads by hoisting invariant loads
    to outermost loop scope:
      - kv_cos_tiles / kv_sin_tiles: hoisted outside kvh loop (depend only on si)
      - Wq_tiles_hoisted: hoisted outside kvh loop (depend only on (qh, ht))
      - Wk_tiles_hoisted / Wv_tiles_hoisted: hoisted outside si loop (depend only on (kvh, ht))

  v4a architecture (v3a + pre-transposed K + affine_select masking):
    Change 1: K tiles stored pre-transposed [d, S_k] in Phase 1.
      Eliminates nc_transpose + tensor_copy from Phase 2 ki hot loop.
      Cost: one extra nc_transpose per (kvh, si) = 4*5=20 ops (vs 200 ops eliminated in Phase 2).
    Change 2: nisa.affine_select replaces attn_mask DMA load + tensor_add.
      Eliminates 200 HBM DMA loads (4 kvh * 2 gi * 5 qsi * 5 ki) from innermost loop.
      causal predicate: (qso + q_idx) >= (kso + k_idx), implemented as affine_select with
      pattern=[[-1, PMAX]], offset=(qso - kso), channel_multiplier=1, cmp_op=greater_equal.

  v4c architecture (v4a + 512-wide K matmul + PSUM-accumulated MM2):
    Builds on v4a (pre-transposed K tiles + affine_select masking).
    Additional change: batch 4 K tiles into K_wide[d=128, K_WIDE=512] per flash-attn iteration.
      MM1: one nc_matmul([S_q, 512]) per 4 ki -> score computation on wider tile
      MM2: 4 V tiles accumulated in a single PSUM (no intermediate PSUM->SBUF copy per V tile)
    For S=640 (num_s_tiles=5): 1 full batch of 4 + 1 tail tile = 2 outer iterations vs 5.

  SBUF budget per kvh iteration:
    K_tiles: num_s_tiles * [PMAX, PMAX] bf16 = 5 * 32KB = 160KB (for S=640)
    V_tiles: num_s_tiles * [PMAX, PMAX] bf16 = 160KB
    hidden_tiles: num_s_tiles * num_h_tiles * [PMAX, PMAX] bf16 = 5*16*32KB = 2.56MB (hoisted)
    scores_all: [PMAX, S_TOTAL=640] f32 = 320KB per qsi (temporary)
    attn_w: [PMAX, S_TOTAL=640] bf16 = 160KB per qsi (temporary)
    Total extra: ~3.5MB -- well within 24MB budget.

Plan C optimization (inherited from v1c):
  Broadcast [PMAX,1] -> [PMAX,PMAX] via SBUF .ap() pattern instead of HBM round-trip.
"""

import math
import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128
K_WIDE = 4 * PMAX  # 512 -- max free dim for nc_matmul moving input
EPS = 1e-6


@nki.jit
def qwen3_attn_cte_fused(
    hidden_states,   # [B, S, H] bf16, B=1 typical
    Wq,              # [Hq_tp*d, H] = [1024, 2048] bf16
    Wk,              # [Hkv_tp*d, H] = [512, 2048] bf16
    Wv,              # [Hkv_tp*d, H] = [512, 2048] bf16
    q_norm_weight,   # [d] = [128] bf16
    k_norm_weight,   # [d] = [128] bf16
    cos,             # [S, d] bf16 - RoPE cos per position
    sin,             # [S, d] bf16 - RoPE sin per position
):
    """Returns [B, S, Hq_tp*d] bf16"""
    B = hidden_states.shape[0]
    S = hidden_states.shape[1]
    H = hidden_states.shape[2]
    Hq_out = Wq.shape[0]
    Hkv_out = Wk.shape[0]
    d = PMAX
    Hq_tp = Hq_out // d
    Hkv_tp = Hkv_out // d
    gqa = Hq_tp // Hkv_tp
    num_h_tiles = H // PMAX
    num_s_tiles = S // PMAX
    half_d = d // 2
    scale = float(1.0 / math.sqrt(d))

    # v4c: batched flash attention constants
    num_full_batches = num_s_tiles // 4   # full batches of 4 K tiles
    ki_tail = num_s_tiles % 4             # remaining tiles (tail)

    # v4d: S_TOTAL for two-pass softmax score bank size
    S_TOTAL = S  # = num_s_tiles * PMAX = 640; used for two-pass softmax score bank size

    # PLAN A: Remove K_scratch and V_scratch private_hbm allocations entirely.
    # K/V tiles will live in SBUF Python lists within each kvh iteration.
    output = nl.ndarray((B, S, Hq_out), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    hidden_2d = hidden_states.reshape((B * S, H))
    output_2d = output.reshape((B * S, Hq_out))

    # PLAN C: Broadcast [PMAX,1] SBUF -> [PMAX,PMAX] SBUF using .ap() pattern on SBUF source.
    # The .ap(pattern=[[1,PMAX],[0,PMAX]], offset=0) descriptor tells the DMA engine to:
    #   - repeat the single free-dim element (dim0: step=1, count=PMAX replicating partition axis)
    #   - repeat the 1-wide free dim to PMAX (dim1: step=0, count=PMAX -- stride=0 = broadcast)
    # This is a SBUF-to-SBUF DMA with broadcast, avoiding ANY HBM round-trip.
    # Replaces the original bcast_hbm private_hbm staging buffer entirely.

    # Load Q norm weights as [PMAX,1] float32 in SBUF, then broadcast to [PMAX,PMAX] via .ap()
    qnw_1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qnw_1")
    qnw_bf16_tmp = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_bf16_tmp")
    nisa.dma_copy(dst=qnw_bf16_tmp, src=q_norm_weight.reshape((PMAX, 1)))
    nisa.tensor_copy(qnw_1, qnw_bf16_tmp)
    # BROADCAST: SBUF .ap() pattern replicates [PMAX,1] -> [PMAX,PMAX] without HBM round-trip
    qnw_sb = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="qnw_sb")
    nisa.dma_copy(dst=qnw_sb, src=qnw_1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

    # Load K norm weights similarly
    knw_1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="knw_1")
    knw_bf16_tmp = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_bf16_tmp")
    nisa.dma_copy(dst=knw_bf16_tmp, src=k_norm_weight.reshape((PMAX, 1)))
    nisa.tensor_copy(knw_1, knw_bf16_tmp)
    # BROADCAST: SBUF .ap() pattern replicates [PMAX,1] -> [PMAX,PMAX] without HBM round-trip
    knw_sb = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="knw_sb")
    nisa.dma_copy(dst=knw_sb, src=knw_1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

    rms_ones = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)

    # =========================================================================
    # v3a HOISTING SECTION: Load cos/sin and Wq once before the kvh loop.
    # These are invariant to kvh and (for cos/sin) to gi as well.
    # =========================================================================

    # Hoist K/V cos/sin: only depends on si, not on kvh.
    # Load all num_s_tiles cos and sin tiles once -- shared across all kvh iterations.
    kv_cos_tiles = []
    kv_sin_tiles = []
    for si in nl.affine_range(num_s_tiles):
        s_off = si * PMAX
        cos_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"hoist_cosbf_s{si}")
        nisa.dma_copy(dst=cos_bf, src=cos[s_off:s_off + PMAX, 0:PMAX])
        cos_f = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"hoist_cosf_s{si}")
        nisa.tensor_copy(cos_f, cos_bf)
        kv_cos_tiles.append(cos_f)

        sin_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"hoist_sinbf_s{si}")
        nisa.dma_copy(dst=sin_bf, src=sin[s_off:s_off + PMAX, 0:PMAX])
        sin_f = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"hoist_sinf_s{si}")
        nisa.tensor_copy(sin_f, sin_bf)
        kv_sin_tiles.append(sin_f)

    # Q cos/sin: same tiles as KV (qsi and si index the same [S, d] cos/sin array).
    q_cos_tiles = kv_cos_tiles  # reuse -- indexed by qsi
    q_sin_tiles = kv_sin_tiles  # reuse -- indexed by qsi

    # Hoist Wq: indexed by (qh, ht) = (kvh*gqa+gi, ht) only -- invariant to qsi.
    # Total: Hq_tp * num_h_tiles tiles stored in SBUF.
    # Use a 2D list (list-of-lists) instead of a dict to avoid NKI dict-mutation restriction.
    Wq_tiles_hoisted = []  # Wq_tiles_hoisted[qh][ht] = pre-transposed [PMAX, PMAX] bf16 SBUF tile
    for qh in nl.affine_range(Hq_tp):
        row = []
        for ht in nl.affine_range(num_h_tiles):
            ho = ht * PMAX
            wqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"hoist_wq_h{qh}_t{ht}")
            nisa.dma_copy(dst=wqb, src=Wq[qh*d:qh*d+PMAX, ho:ho+PMAX])
            wqTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                               name=f"hoist_wqTp_h{qh}_t{ht}")
            nisa.nc_transpose(wqTp, wqb)
            wqTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"hoist_wqTs_h{qh}_t{ht}")
            nisa.tensor_copy(wqTs, wqTp)
            row.append(wqTs)
        Wq_tiles_hoisted.append(row)

    # =========================================================================
    # v4d: Preload all hidden tiles into SBUF -- eliminates 1,280 HBM loads.
    # SBUF cost: num_s_tiles * num_h_tiles * [PMAX,PMAX] bf16 = 5*16*32KB = 2.56MB
    # Use range() (not affine_range) for both loops: nl.ndarray allocations inside
    # loops stored in Python lists require range() for correct list indexing.
    # =========================================================================
    hidden_tiles = []  # hidden_tiles[si][ht] = [PMAX,PMAX] bf16, raw (not pre-transposed)
    for si in range(num_s_tiles):
        s_off = si * PMAX
        row = []
        for ht in range(num_h_tiles):
            ho = ht * PMAX
            hb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"hoist_hb_s{si}_t{ht}")
            nisa.dma_copy(dst=hb, src=hidden_2d[s_off:s_off+PMAX, ho:ho+PMAX])
            row.append(hb)
        hidden_tiles.append(row)

    # =========================================================================
    # PLAN A: Single fused kvh loop (replaces separate Phase 1 + Phase 2 loops)
    # Outer loop uses affine_range to keep 4 kvh iterations independent.
    # =========================================================================
    for kvh in nl.affine_range(Hkv_tp):

        # -----------------------------------------------------------------
        # v3a: Hoist Wk/Wv outside the si loop.
        # Wk[kvh*d:kvh*d+PMAX, ho:ho+PMAX] depends only on (kvh, ht), not si.
        # Load and pre-transpose all weight tiles for this head once per kvh.
        # -----------------------------------------------------------------
        Wk_tiles_hoisted = []  # Wk_tiles_hoisted[ht] = [PMAX, PMAX] bf16 SBUF, pre-transposed
        Wv_tiles_hoisted = []

        for ht in nl.affine_range(num_h_tiles):
            ho = ht * PMAX
            # K weight tile
            wb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"hoist_wk_kv{kvh}_t{ht}")
            nisa.dma_copy(dst=wb, src=Wk[kvh*d:kvh*d+PMAX, ho:ho+PMAX])
            wTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                              name=f"hoist_wkTp_kv{kvh}_t{ht}")
            nisa.nc_transpose(wTp, wb)
            wTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"hoist_wkTs_kv{kvh}_t{ht}")
            nisa.tensor_copy(wTs, wTp)
            Wk_tiles_hoisted.append(wTs)

            # V weight tile
            wvb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"hoist_wv_kv{kvh}_t{ht}")
            nisa.dma_copy(dst=wvb, src=Wv[kvh*d:kvh*d+PMAX, ho:ho+PMAX])
            wvTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                               name=f"hoist_wvTp_kv{kvh}_t{ht}")
            nisa.nc_transpose(wvTp, wvb)
            wvTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"hoist_wvTs_kv{kvh}_t{ht}")
            nisa.tensor_copy(wvTs, wvTp)
            Wv_tiles_hoisted.append(wvTs)

        # -----------------------------------------------------------------
        # Phase 1 sub-loop: compute K/V tiles and store in SBUF Python lists.
        # v4d: merged K+V ht loop -- share hidden tile for both K and V matmul.
        # v4d: use hidden_tiles from SBUF -- no HBM DMA for hidden states.
        # v4c Change 1: K tiles stored pre-transposed [d, S_k] in SBUF.
        # -----------------------------------------------------------------
        K_tiles = []  # Python list: K_tiles[si] = krTs [PMAX, PMAX] bf16, pre-transposed [d, S_k]
        V_tiles = []  # Python list: V_tiles[si] = vbf  [PMAX, PMAX] bf16

        for si in nl.affine_range(num_s_tiles):
            s_off = si * PMAX

            # v3a: Use hoisted cos/sin tiles -- no HBM DMA per kvh iteration.
            cos_f = kv_cos_tiles[si]   # from SBUF -- no HBM DMA
            sin_f = kv_sin_tiles[si]   # from SBUF -- no HBM DMA

            # v4d: Merged K+V projection -- share hidden tile for both matmuls.
            # Eliminates duplicate hidden loads (V proj was loading identical tiles as K proj).
            kp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                          name=f"p1_kp_kv{kvh}_s{si}")
            vp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                          name=f"p1_vp_kv{kvh}_s{si}")
            for ht in nl.affine_range(num_h_tiles):
                hb = hidden_tiles[si][ht]   # from SBUF -- no HBM DMA
                hTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                  name=f"p1_hTp_kv{kvh}_s{si}_t{ht}")
                nisa.nc_transpose(hTp, hb)
                hTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"p1_hTs_kv{kvh}_s{si}_t{ht}")
                nisa.tensor_copy(hTs, hTp)
                nisa.nc_matmul(kp, stationary=hTs, moving=Wk_tiles_hoisted[ht])
                nisa.nc_matmul(vp, stationary=hTs, moving=Wv_tiles_hoisted[ht])

            kv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_kv_kv{kvh}_s{si}")
            nisa.tensor_copy(kv, kp)

            # K RMSNorm in transposed [d,S] layout
            kvb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_kvb_kv{kvh}_s{si}")
            nisa.tensor_copy(kvb, kv)
            kvTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                              name=f"p1_kvTp_kv{kvh}_s{si}")
            nisa.nc_transpose(kvTp, kvb)
            kvTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_kvTb_kv{kvh}_s{si}")
            nisa.tensor_copy(kvTb, kvTp)
            kvT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"p1_kvT_kv{kvh}_s{si}")
            nisa.tensor_copy(kvT, kvTb)

            ksq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"p1_ksq_kv{kvh}_s{si}")
            nisa.tensor_tensor(ksq, kvT, kvT, op=nl.multiply)
            ksqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_ksqb_kv{kvh}_s{si}")
            nisa.tensor_copy(ksqb, ksq)
            ksump = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                             name=f"p1_ksump_kv{kvh}_s{si}")
            nisa.nc_matmul(ksump, stationary=rms_ones, moving=ksqb)
            ksum = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_ksum_kv{kvh}_s{si}")
            nisa.tensor_copy(ksum, ksump)

            kmean = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_kmean_kv{kvh}_s{si}")
            nisa.tensor_scalar(kmean, ksum, op0=nl.multiply, operand0=1.0/d)
            kvar = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_kvar_kv{kvh}_s{si}")
            nisa.tensor_scalar(kvar, kmean, op0=nl.add, operand0=EPS)
            krinv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_krinv_kv{kvh}_s{si}")
            nisa.activation(krinv, op=nl.rsqrt, data=kvar)

            knT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"p1_knT_kv{kvh}_s{si}")
            nisa.tensor_tensor(knT, kvT, krinv, op=nl.multiply)
            # Apply norm weight (both [PMAX, PMAX] now)
            knwT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_knwT_kv{kvh}_s{si}")
            nisa.tensor_tensor(knwT, knT, knw_sb, op=nl.multiply)

            # Transpose back to [S,d]
            knwTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"p1_knwTb_kv{kvh}_s{si}")
            nisa.tensor_copy(knwTb, knwT)
            knp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                             name=f"p1_knp_kv{kvh}_s{si}")
            nisa.nc_transpose(knp, knwTb)
            knb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_knb_kv{kvh}_s{si}")
            nisa.tensor_copy(knb, knp)
            kn = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_kn_kv{kvh}_s{si}")
            nisa.tensor_copy(kn, knb)

            # K RoPE
            rotk = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_rotk_kv{kvh}_s{si}")
            negku = nl.ndarray((PMAX, half_d), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_negku_kv{kvh}_s{si}")
            nisa.tensor_scalar(negku, kn[0:PMAX, half_d:d], op0=nl.multiply, operand0=-1.0)
            nisa.tensor_copy(rotk[0:PMAX, 0:half_d], negku)
            nisa.tensor_copy(rotk[0:PMAX, half_d:d], kn[0:PMAX, 0:half_d])

            kcos = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_kcos_kv{kvh}_s{si}")
            nisa.tensor_tensor(kcos, kn, cos_f, op=nl.multiply)
            ksinp = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_ksinp_kv{kvh}_s{si}")
            nisa.tensor_tensor(ksinp, rotk, sin_f, op=nl.multiply)
            krope = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_krope_kv{kvh}_s{si}")
            nisa.tensor_tensor(krope, kcos, ksinp, op=nl.add)

            # v4a Change 1: Store K tile pre-transposed [d, S_k] in SBUF.
            # Eliminates nc_transpose + tensor_copy from Phase 2 ki hot loop.
            krbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_krbf_kv{kvh}_s{si}")
            nisa.tensor_copy(krbf, krope)
            # Pre-transpose: store K tile as [d, S_k] for direct use as MM1 moving input
            krT_psum = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                  name=f"p1_krTp_kv{kvh}_s{si}")
            nisa.nc_transpose(krT_psum, krbf)
            krTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_krTs_kv{kvh}_s{si}")
            nisa.tensor_copy(krTs, krT_psum)
            K_tiles.append(krTs)  # [d=128, S_k=128] -- ready for MM1 moving position without transpose

            # V projection: vp was accumulated in the merged ht loop above (no RMSNorm).
            vv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_vv_kv{kvh}_s{si}")
            nisa.tensor_copy(vv, vp)
            # PLAN A: Store V tile in SBUF (append to Python list), NOT to HBM scratch.
            vbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_vbf_kv{kvh}_s{si}")
            nisa.tensor_copy(vbf, vv)
            V_tiles.append(vbf)

        # -----------------------------------------------------------------
        # Phase 2 sub-loop: Per Q head two-pass softmax attention
        # K_tiles[ki] and V_tiles[ki] are already in SBUF -- no HBM DMA needed.
        # v4d: replaces flash attention with two-pass softmax:
        #   Pass 1: fill scores_all[PMAX, S_TOTAL] via 512-wide MM1 + affine_select
        #   Pass 2: global softmax (2 broadcasts total per qsi vs 3 per ki_batch in flash)
        #   Pass 3: MM2 accumulation over all V tiles into single PSUM
        # -----------------------------------------------------------------
        for gi in nl.affine_range(gqa):
            qh = kvh * gqa + gi

            for qsi in nl.affine_range(num_s_tiles):
                qso = qsi * PMAX

                # Q projection
                # v4d: Use preloaded hidden tiles from SBUF -- no HBM DMA.
                # v3a: Use hoisted Wq tiles -- no HBM DMA per qsi/kvh/gi iteration.
                qp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                              name=f"p2_qp_kv{kvh}_g{gi}_s{qsi}")
                for ht in nl.affine_range(num_h_tiles):
                    hb = hidden_tiles[qsi][ht]   # from SBUF -- no HBM DMA
                    hTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                      name=f"p2_hTp_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.nc_transpose(hTp, hb)
                    hTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                      name=f"p2_hTs_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.tensor_copy(hTs, hTp)
                    # v3a: Use hoisted Wq tile -- no HBM DMA per qsi/kvh/gi iteration.
                    nisa.nc_matmul(qp, stationary=hTs, moving=Wq_tiles_hoisted[qh][ht])

                qvec = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qvec_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvec, qp)

                # Q RMSNorm in transposed [d,S]
                qvb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"p2_qvb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvb, qvec)
                qvTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                  name=f"p2_qvTp_kv{kvh}_g{gi}_s{qsi}")
                nisa.nc_transpose(qvTp, qvb)
                qvTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"p2_qvTb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvTb, qvTp)
                qvT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_qvT_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvT, qvTb)

                qsq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_qsq_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qsq, qvT, qvT, op=nl.multiply)
                qsqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"p2_qsqb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qsqb, qsq)
                qsump = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                 name=f"p2_qsump_kv{kvh}_g{gi}_s{qsi}")
                nisa.nc_matmul(qsump, stationary=rms_ones, moving=qsqb)
                qsum = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qsum_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qsum, qsump)

                qmean = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_qmean_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qmean, qsum, op0=nl.multiply, operand0=1.0/d)
                qvar = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qvar_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qvar, qmean, op0=nl.add, operand0=EPS)
                qrinv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_qrinv_kv{kvh}_g{gi}_s{qsi}")
                nisa.activation(qrinv, op=nl.rsqrt, data=qvar)

                qnT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_qnT_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qnT, qvT, qrinv, op=nl.multiply)
                qnwT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qnwT_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qnwT, qnT, qnw_sb, op=nl.multiply)

                qnwTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"p2_qnwTb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qnwTb, qnwT)
                qnp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                 name=f"p2_qnp_kv{kvh}_g{gi}_s{qsi}")
                nisa.nc_transpose(qnp, qnwTb)
                qnb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"p2_qnb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qnb, qnp)
                qn = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"p2_qn_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qn, qnb)

                # Q RoPE -- v3a: use hoisted cos/sin tiles, no HBM DMA.
                cosqf = q_cos_tiles[qsi]  # from SBUF -- no HBM DMA
                sinqf = q_sin_tiles[qsi]  # from SBUF -- no HBM DMA

                rotq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_rotq_kv{kvh}_g{gi}_s{qsi}")
                negqu = nl.ndarray((PMAX, half_d), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_negqu_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(negqu, qn[0:PMAX, half_d:d], op0=nl.multiply, operand0=-1.0)
                nisa.tensor_copy(rotq[0:PMAX, 0:half_d], negqu)
                nisa.tensor_copy(rotq[0:PMAX, half_d:d], qn[0:PMAX, 0:half_d])

                qcos = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qcos_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qcos, qn, cosqf, op=nl.multiply)
                qsinp = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_qsinp_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qsinp, rotq, sinqf, op=nl.multiply)
                qrope = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_qrope_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qrope, qcos, qsinp, op=nl.add)

                qsc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_qsc_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qsc, qrope, op0=nl.multiply, operand0=scale)

                qscb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"p2_qscb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qscb, qsc)
                qTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                 name=f"p2_qTp_kv{kvh}_g{gi}_s{qsi}")
                nisa.nc_transpose(qTp, qscb)
                qTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"p2_qTs_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qTs, qTp)

                # ==============================================================
                # v4d Pass 1: Compute all scores + apply causal mask
                # Allocate score bank for all K positions: [PMAX, S_TOTAL] float32
                # SBUF cost: 128 * 640 * 4 = 320KB -- fits easily
                # ==============================================================
                scores_all = nl.ndarray((PMAX, S_TOTAL), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_scores_kv{kvh}_g{gi}_s{qsi}")

                # Process full batches of 4 K tiles -> 512-wide MM1
                for ki_b in nl.affine_range(num_full_batches):
                    batch_start = ki_b * 4
                    kso_batch   = batch_start * PMAX

                    # Build K_wide [PMAX, K_WIDE=512] from 4 pre-transposed K tiles
                    K_wide = nl.ndarray((PMAX, K_WIDE), dtype=nl.bfloat16, buffer=nl.sbuf,
                                         name=f"p2_Kw_kv{kvh}_g{gi}_s{qsi}_b{ki_b}")
                    nisa.tensor_copy(K_wide[0:PMAX, 0*PMAX:(0+1)*PMAX], K_tiles[batch_start + 0])
                    nisa.tensor_copy(K_wide[0:PMAX, 1*PMAX:(1+1)*PMAX], K_tiles[batch_start + 1])
                    nisa.tensor_copy(K_wide[0:PMAX, 2*PMAX:(2+1)*PMAX], K_tiles[batch_start + 2])
                    nisa.tensor_copy(K_wide[0:PMAX, 3*PMAX:(3+1)*PMAX], K_tiles[batch_start + 3])

                    scp_w = nl.zeros((PMAX, K_WIDE), dtype=nl.float32, buffer=nl.psum,
                                      name=f"p2_scpw_kv{kvh}_g{gi}_s{qsi}_b{ki_b}")
                    nisa.nc_matmul(scp_w, stationary=qTs, moving=K_wide)

                    # Copy PSUM -> SBUF
                    scs_w = nl.ndarray((PMAX, K_WIDE), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_scsw_kv{kvh}_g{gi}_s{qsi}_b{ki_b}")
                    nisa.tensor_copy(scs_w, scp_w)

                    # Apply causal mask via affine_select -> temp, then copy into scores_all slice
                    sm_tmp_w = nl.ndarray((PMAX, K_WIDE), dtype=nl.float32, buffer=nl.sbuf,
                                          name=f"p2_smtmpw_kv{kvh}_g{gi}_s{qsi}_b{ki_b}")
                    nisa.affine_select(
                        dst=sm_tmp_w,
                        on_true_tile=scs_w,
                        on_false_value=-3.4e38,
                        pattern=[[-1, K_WIDE]],
                        offset=qso - kso_batch,
                        channel_multiplier=1,
                        cmp_op=nl.greater_equal,
                    )
                    nisa.tensor_copy(scores_all[0:PMAX, kso_batch:kso_batch+K_WIDE], sm_tmp_w)

                # Tail tiles (1 tile for S=640, num_s_tiles=5, num_full_batches=1)
                for ki in nl.affine_range(ki_tail):
                    ki_abs = num_full_batches * 4 + ki
                    kso = ki_abs * PMAX

                    kTs2 = K_tiles[ki_abs]   # pre-transposed [d, S_k]
                    scp_t = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                      name=f"p2_scpt_kv{kvh}_g{gi}_s{qsi}_ki{ki}")
                    nisa.nc_matmul(scp_t, stationary=qTs, moving=kTs2)
                    scs_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_scst_kv{kvh}_g{gi}_s{qsi}_ki{ki}")
                    nisa.tensor_copy(scs_t, scp_t)
                    sm_tmp_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                          name=f"p2_smtmpt_kv{kvh}_g{gi}_s{qsi}_ki{ki}")
                    nisa.affine_select(
                        dst=sm_tmp_t,
                        on_true_tile=scs_t,
                        on_false_value=-3.4e38,
                        pattern=[[-1, PMAX]],
                        offset=qso - kso,
                        channel_multiplier=1,
                        cmp_op=nl.greater_equal,
                    )
                    nisa.tensor_copy(scores_all[0:PMAX, kso:kso+PMAX], sm_tmp_t)

                # ==============================================================
                # v4d Pass 2: Softmax on the full [PMAX, S_TOTAL] score bank
                # ==============================================================

                # Row-wise max over [PMAX, S_TOTAL] -> [PMAX, 1]
                row_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_rowmax_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=scores_all, axis=1)

                # Broadcast [PMAX,1] -> [PMAX, S_TOTAL]: ONE SBUF DMA op for the whole sequence
                neg_row_max_1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                            name=f"p2_nrm1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(neg_row_max_1, row_max, op0=nl.multiply, operand0=-1.0)
                neg_row_max_wide = nl.ndarray((PMAX, S_TOTAL), dtype=nl.float32, buffer=nl.sbuf,
                                               name=f"p2_nrmw_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=neg_row_max_wide,
                              src=neg_row_max_1.ap(pattern=[[1, PMAX], [0, S_TOTAL]], offset=0)) # High latency

                # score_shifted = scores_all + neg_row_max_wide
                sexp_all = nl.ndarray((PMAX, S_TOTAL), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_sexpall_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(sexp_all, scores_all, neg_row_max_wide, op=nl.add)

                # exp: sexp_all2 = exp(sexp_all)
                sexp_all2 = nl.ndarray((PMAX, S_TOTAL), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_sexpall2_kv{kvh}_g{gi}_s{qsi}")
                nisa.activation(sexp_all2, op=nl.exp, data=sexp_all)

                # Row sum -> [PMAX, 1]
                row_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_rowsum_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_reduce(dst=row_sum, op=nl.add, data=sexp_all2, axis=1)

                # Reciprocal sum via rsqrt^2 pattern (same as normalize block in v4c)
                ssafe = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_ssafe_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(ssafe, row_sum, op0=nl.add, operand0=1e-9)
                rsq = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_rsq_kv{kvh}_g{gi}_s{qsi}")
                nisa.activation(rsq, op=nl.rsqrt, data=ssafe)
                inv_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_invsum_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(inv_sum, rsq, rsq, op=nl.multiply)

                # Broadcast inv_sum [PMAX,1] -> [PMAX, S_TOTAL]
                inv_sum_wide = nl.ndarray((PMAX, S_TOTAL), dtype=nl.float32, buffer=nl.sbuf,
                                           name=f"p2_invwide_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=inv_sum_wide,
                              src=inv_sum.ap(pattern=[[1, PMAX], [0, S_TOTAL]], offset=0)) # High latency

                # Normalized attention weights: attn_w[PMAX, S_TOTAL]
                attn_w = nl.ndarray((PMAX, S_TOTAL), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"p2_attnw_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(attn_w, sexp_all2, inv_sum_wide, op=nl.multiply)

                # ==============================================================
                # v4d Pass 3: MM2 -- accumulate all V tiles into one PSUM
                # ==============================================================

                # Convert attn_w to bf16 for matmul
                attn_w_b = nl.ndarray((PMAX, S_TOTAL), dtype=nl.bfloat16, buffer=nl.sbuf,
                                       name=f"p2_attnwb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(attn_w_b, attn_w)

                # MM2: attn_w slice [PMAX,PMAX] transposed (stationary) x V tile [PMAX, d] (moving)
                # Accumulate all ki into one PSUM -- nc_matmul ADDs to existing PSUM content.
                vcp_all = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                    name=f"p2_vcpall_kv{kvh}_g{gi}_s{qsi}")
                for ki in range(num_s_tiles):   # Python range -- unrolled, accumulate in same PSUM
                    ki_o = ki * PMAX
                    # Extract [PMAX,PMAX] slice of attn_w_b for this V tile
                    aw_sl_b = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                          name=f"p2_awslb_kv{kvh}_g{gi}_s{qsi}_ki{ki}")
                    nisa.tensor_copy(aw_sl_b, attn_w_b[0:PMAX, ki_o:ki_o+PMAX])
                    # Transpose for stationary position: [S_q, S_k] -> [S_k, S_q]
                    aw_sl_Tp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                           name=f"p2_awslTp_kv{kvh}_g{gi}_s{qsi}_ki{ki}")
                    nisa.nc_transpose(aw_sl_Tp, aw_sl_b)
                    aw_sl_Ts = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                           name=f"p2_awslTs_kv{kvh}_g{gi}_s{qsi}_ki{ki}")
                    nisa.tensor_copy(aw_sl_Ts, aw_sl_Tp)
                    nisa.nc_matmul(vcp_all, stationary=aw_sl_Ts, moving=V_tiles[ki])

                # Copy final PSUM to SBUF and store to output
                aout_f = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"p2_aoutf_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(aout_f, vcp_all)
                aout_b = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                     name=f"p2_aoutb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(aout_b, aout_f)
                nisa.dma_copy(
                    dst=output_2d[qso:qso+PMAX, qh*d:qh*d+PMAX],
                    src=aout_b,
                )

    return output


# =============================================================================
# Test harness
# =============================================================================
if __name__ == "__main__":
    import os
    import sys
    import torch
    import torch.nn.functional as F

    os.environ["NEURON_CC_FLAGS"] = " "
    os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

    import torch_xla.core.xla_model as xm

    def pytorch_reference(hidden_states, Wq, Wk, Wv, q_norm_weight, k_norm_weight,
                          cos_vals, sin_vals, d=128):
        B, S, H = hidden_states.shape
        Hq_out = Wq.shape[0]
        Hkv_out = Wk.shape[0]
        Hq_tp = Hq_out // d
        Hkv_tp = Hkv_out // d
        gqa = Hq_tp // Hkv_tp
        scale_val = 1.0 / math.sqrt(d)

        hs = hidden_states.float()
        Q = (hs @ Wq.float().T).reshape(B, S, Hq_tp, d).transpose(1, 2)
        K = (hs @ Wk.float().T).reshape(B, S, Hkv_tp, d).transpose(1, 2)
        V = (hs @ Wv.float().T).reshape(B, S, Hkv_tp, d).transpose(1, 2)

        def rmsnorm(x, w, eps=1e-6):
            var = x.float().pow(2).mean(-1, keepdim=True)
            return (x.float() * torch.rsqrt(var + eps) * w.float())

        for h in range(Hq_tp):
            Q[:, h] = rmsnorm(Q[:, h], q_norm_weight)
        for h in range(Hkv_tp):
            K[:, h] = rmsnorm(K[:, h], k_norm_weight)

        cos_b = cos_vals.float().unsqueeze(0).unsqueeze(0)
        sin_b = sin_vals.float().unsqueeze(0).unsqueeze(0)

        def rotate_half(x):
            x1, x2 = x[..., :d // 2], x[..., d // 2:]
            return torch.cat((-x2, x1), dim=-1)

        Q = Q.float() * cos_b + rotate_half(Q.float()) * sin_b
        K = K.float() * cos_b + rotate_half(K.float()) * sin_b

        K_rep = K.repeat_interleave(gqa, dim=1)
        V_rep = V.float().repeat_interleave(gqa, dim=1)

        scores = (Q @ K_rep.transpose(-2, -1)) * scale_val
        causal = torch.triu(torch.full((S, S), float('-inf')), diagonal=1)
        scores = scores + causal

        P = torch.softmax(scores, dim=-1)
        out = P @ V_rep
        out = out.transpose(1, 2).reshape(B, S, Hq_tp * d)
        return out.to(torch.bfloat16)

    def run_test(B, S, H=2048, d=128, Hq_tp=8, Hkv_tp=4):
        print(f"\n{'=' * 70}")
        print(f"qwen3_attn_cte_fused v4d Test: B={B}, S={S}, H={H}, d={d}")
        print(f"Hq_tp={Hq_tp}, Hkv_tp={Hkv_tp}, gqa={Hq_tp // Hkv_tp}")
        print(f"{'=' * 70}")

        device = xm.xla_device()
        dtype = torch.bfloat16
        Hq_out = Hq_tp * d
        Hkv_out = Hkv_tp * d

        torch.manual_seed(42)
        sc = 0.02

        hidden_states = (torch.randn(B, S, H) * sc).to(dtype)
        Wq = (torch.randn(Hq_out, H) * sc).to(dtype)
        Wk = (torch.randn(Hkv_out, H) * sc).to(dtype)
        Wv = (torch.randn(Hkv_out, H) * sc).to(dtype)
        q_norm_weight = torch.ones(d, dtype=dtype)
        k_norm_weight = torch.ones(d, dtype=dtype)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
        t = torch.arange(S).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_v = emb.cos().to(dtype)
        sin_v = emb.sin().to(dtype)

        print("\n[1/4] Computing PyTorch reference...")
        ref_out = pytorch_reference(
            hidden_states, Wq, Wk, Wv,
            q_norm_weight, k_norm_weight,
            cos_v, sin_v, d=d,
        )
        print(f"  ref_out shape: {ref_out.shape}")
        print(f"  ref_out stats: min={ref_out.float().min():.6f}, max={ref_out.float().max():.6f}")

        print("\n[2/4] Moving tensors to device...")
        hidden_dev = hidden_states.to(device)
        Wq_dev = Wq.to(device)
        Wk_dev = Wk.to(device)
        Wv_dev = Wv.to(device)
        qnw_dev = q_norm_weight.to(device)
        knw_dev = k_norm_weight.to(device)
        cos_dev = cos_v.to(device)
        sin_dev = sin_v.to(device)

        print("\n[3/4] Running NKI kernel...")
        nki_out_dev = qwen3_attn_cte_fused(
            hidden_dev, Wq_dev, Wk_dev, Wv_dev,
            qnw_dev, knw_dev,
            cos_dev, sin_dev,
        )
        xm.mark_step()
        nki_out = nki_out_dev.cpu()
        print(f"  nki_out shape: {nki_out.shape}")
        print(f"  nki_out stats: min={nki_out.float().min():.6f}, max={nki_out.float().max():.6f}")

        print("\n[4/4] Comparing results...")
        ref_f = ref_out.float()
        nki_f = nki_out.float()
        diff = (ref_f - nki_f).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"  Max  |diff| : {max_diff:.6e}")
        print(f"  Mean |diff| : {mean_diff:.6e}")

        threshold = 0.05
        if max_diff < threshold:
            print(f"\n  PASS: max_diff={max_diff:.4e} < {threshold}")
            return True
        else:
            print(f"\n  FAIL: max_diff={max_diff:.4e} >= {threshold}")
            for h in range(Hq_tp):
                h_diff = diff[0, :, h * d:(h + 1) * d].max().item()
                print(f"    Head {h}: max_diff={h_diff:.4e}")
            print(f"\n  Ref sample:  {ref_f[0, 0, :8].tolist()}")
            print(f"  NKI sample:  {nki_f[0, 0, :8].tolist()}")
            return False

    all_pass = True
    for test_S in [640]:
        ok = run_test(B=1, S=test_S)
        if not ok:
            all_pass = False

    if all_pass:
        print(f"\n{'=' * 70}")
        print("ALL TESTS PASSED")
        print(f"{'=' * 70}")
    else:
        print(f"\n{'=' * 70}")
        print("SOME TESTS FAILED")
        print(f"{'=' * 70}")
        sys.exit(1)
