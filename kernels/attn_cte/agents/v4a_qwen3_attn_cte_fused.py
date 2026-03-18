"""
Fused NKI attention CTE kernel for Qwen3.
Fuses: QKV linear projection + per-head RMSNorm + RoPE + causal flash attention.

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

  SBUF budget per kvh iteration:
    K_tiles: num_s_tiles * [PMAX, PMAX] bf16 = 5 * 32KB = 160KB (for S=640)
    V_tiles: num_s_tiles * [PMAX, PMAX] bf16 = 160KB
    Computation temporaries: same as v1c
    Total extra: 320KB/kvh — well within 24MB budget.

Plan C optimization (inherited from v1c):
  Broadcast [PMAX,1] -> [PMAX,PMAX] via SBUF .ap() pattern instead of HBM round-trip.
"""

import math
import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128
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

    # PLAN A: Remove K_scratch and V_scratch private_hbm allocations entirely.
    # K/V tiles will live in SBUF Python lists within each kvh iteration.
    output = nl.ndarray((B, S, Hq_out), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    hidden_2d = hidden_states.reshape((B * S, H))
    output_2d = output.reshape((B * S, Hq_out))

    # PLAN C: Broadcast [PMAX,1] SBUF -> [PMAX,PMAX] SBUF using .ap() pattern on SBUF source.
    # The .ap(pattern=[[1,PMAX],[0,PMAX]], offset=0) descriptor tells the DMA engine to:
    #   - repeat the single free-dim element (dim0: step=1, count=PMAX replicating partition axis)
    #   - repeat the 1-wide free dim to PMAX (dim1: step=0, count=PMAX — stride=0 = broadcast)
    # This is a SBUF-to-SBUF DMA with broadcast, avoiding ANY HBM round-trip.
    # Replaces the original bcast_hbm private_hbm staging buffer entirely.

    # Load Q norm weights as [PMAX,1] float32 in SBUF, then broadcast to [PMAX,PMAX] via .ap()
    qnw_1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qnw_1")
    qnw_bf16_tmp = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_bf16_tmp")
    nisa.dma_copy(dst=qnw_bf16_tmp, src=q_norm_weight.reshape((PMAX, 1)))
    nisa.tensor_copy(qnw_1, qnw_bf16_tmp)
    # BROADCAST: SBUF .ap() pattern replicates [PMAX,1] -> [PMAX,PMAX] without HBM round-trip
    qnw_sb = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="qnw_sb")
    nisa.dma_copy(dst=qnw_sb, src=qnw_1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0)) # Long idle time

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
    # Load all num_s_tiles cos and sin tiles once — shared across all kvh iterations.
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
    q_cos_tiles = kv_cos_tiles  # reuse — indexed by qsi
    q_sin_tiles = kv_sin_tiles  # reuse — indexed by qsi

    # Hoist Wq: indexed by (qh, ht) = (kvh*gqa+gi, ht) only — invariant to qsi.
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
        # Phase 1 sub-loop: compute K/V tiles and store in SBUF Python lists
        # Use range() (not affine_range) so all si tiles are guaranteed written
        # before Phase 2 reads them. This enforces sequential ordering within kvh.
        # -----------------------------------------------------------------
        K_tiles = []  # Python list of SBUF tensors: K_tiles[si] = krTs [PMAX, PMAX] bf16 (pre-transposed [d, S_k])
        V_tiles = []  # Python list of SBUF tensors: V_tiles[si] = vbf  [PMAX, PMAX] bf16

        for si in nl.affine_range(num_s_tiles):
            s_off = si * PMAX

            # v3a: Use hoisted cos/sin tiles — no HBM DMA per kvh iteration.
            cos_f = kv_cos_tiles[si]   # from SBUF — no HBM DMA
            sin_f = kv_sin_tiles[si]   # from SBUF — no HBM DMA

            # K projection
            kp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                          name=f"p1_kp_kv{kvh}_s{si}")

            for ht in nl.affine_range(num_h_tiles):
                ho = ht * PMAX
                hb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"p1_hk_kv{kvh}_s{si}_t{ht}")
                nisa.dma_copy(dst=hb, src=hidden_2d[s_off:s_off+PMAX, ho:ho+PMAX]) # Long idle time
                hTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                 name=f"p1_hkTp_kv{kvh}_s{si}_t{ht}")
                nisa.nc_transpose(hTp, hb)
                hTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"p1_hkTs_kv{kvh}_s{si}_t{ht}")
                nisa.tensor_copy(hTs, hTp)
                # v3a: Use hoisted Wk tile — no HBM DMA per si iteration.
                nisa.nc_matmul(kp, stationary=hTs, moving=Wk_tiles_hoisted[ht])

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
            # Cost: one extra nc_transpose per (kvh, si) = 4*5=20 ops total
            # (vs 4*2*5*5=200 ops eliminated in Phase 2 ki loop).
            krbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_krbf_kv{kvh}_s{si}")
            nisa.tensor_copy(krbf, krope)
            # Pre-transpose: store K tile as [d, S_k] for direct use as MM1 moving input
            # — avoids nc_transpose in hot Phase 2 loop
            krT_psum = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                  name=f"p1_krTp_kv{kvh}_s{si}")
            nisa.nc_transpose(krT_psum, krbf)
            krTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_krTs_kv{kvh}_s{si}")
            nisa.tensor_copy(krTs, krT_psum)
            K_tiles.append(krTs)  # [d=128, S_k=128] — ready for MM1 moving position without transpose

            # V projection (no RMSNorm — correct per v1c)
            vp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                          name=f"p1_vp_kv{kvh}_s{si}")
            for ht in nl.affine_range(num_h_tiles):
                ho = ht * PMAX
                hvb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"p1_hv_kv{kvh}_s{si}_t{ht}")
                nisa.dma_copy(dst=hvb, src=hidden_2d[s_off:s_off+PMAX, ho:ho+PMAX])
                hvTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                  name=f"p1_hvTp_kv{kvh}_s{si}_t{ht}")
                nisa.nc_transpose(hvTp, hvb)
                hvTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"p1_hvTs_kv{kvh}_s{si}_t{ht}")
                nisa.tensor_copy(hvTs, hvTp)
                # v3a: Use hoisted Wv tile — no HBM DMA per si iteration.
                nisa.nc_matmul(vp, stationary=hvTs, moving=Wv_tiles_hoisted[ht])

            vv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_vv_kv{kvh}_s{si}")
            nisa.tensor_copy(vv, vp)
            # PLAN A: Store V tile in SBUF (append to Python list), NOT to HBM scratch.
            # vbf stays in SBUF; Phase 2 will use V_tiles[ki] directly.
            vbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_vbf_kv{kvh}_s{si}")
            nisa.tensor_copy(vbf, vv)
            V_tiles.append(vbf)

        # -----------------------------------------------------------------
        # Phase 2 sub-loop: Per Q head flash attention
        # K_tiles[ki] and V_tiles[ki] are already in SBUF — no HBM DMA needed.
        # All state kept as [PMAX, PMAX] to avoid broadcast issues.
        # Scalars (max, sum, alpha) are [PMAX, PMAX] with identical columns.
        # -----------------------------------------------------------------
        for gi in nl.affine_range(gqa):
            qh = kvh * gqa + gi

            for qsi in nl.affine_range(num_s_tiles):
                qso = qsi * PMAX

                # Q projection
                qp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                              name=f"p2_qp_kv{kvh}_g{gi}_s{qsi}")
                for ht in nl.affine_range(num_h_tiles):
                    ho = ht * PMAX
                    hqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                     name=f"p2_hq_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.dma_copy(dst=hqb, src=hidden_2d[qso:qso+PMAX, ho:ho+PMAX])
                    hqTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                      name=f"p2_hqTp_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.nc_transpose(hqTp, hqb)
                    hqTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                      name=f"p2_hqTs_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.tensor_copy(hqTs, hqTp)
                    # v3a: Use hoisted Wq tile — no HBM DMA per qsi/kvh/gi iteration.
                    nisa.nc_matmul(qp, stationary=hqTs, moving=Wq_tiles_hoisted[qh][ht])

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

                # Q RoPE — v3a: use hoisted cos/sin tiles, no HBM DMA.
                cosqf = q_cos_tiles[qsi]  # from SBUF — no HBM DMA
                sinqf = q_sin_tiles[qsi]  # from SBUF — no HBM DMA

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

                # Flash attention - all state [PMAX, PMAX]
                atacc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_atacc_kv{kvh}_g{gi}_s{qsi}")
                nisa.memset(atacc, value=0.0)
                # running_max and running_sum: [PMAX, PMAX] with identical columns
                rmax = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_rmax_kv{kvh}_g{gi}_s{qsi}")
                nisa.memset(rmax, value=-1e30)
                rsum = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_rsum_kv{kvh}_g{gi}_s{qsi}")
                nisa.memset(rsum, value=0.0)

                for ki in nl.affine_range(num_s_tiles):
                    kso = ki * PMAX

                    # v4a Change 1: K tile is pre-transposed [d, S_k] — no nc_transpose needed.
                    # K_tiles[ki] = krTs from Phase 1 si=ki iteration (same kvh).
                    kTs2 = K_tiles[ki]

                    # MM1: score[S_q, S_k]
                    scp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                   name=f"p2_scp_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.nc_matmul(scp, stationary=qTs, moving=kTs2)
                    scs = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"p2_scs_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_copy(scs, scp)

                    # v4a Change 2: Causal mask via affine_select — no HBM DMA, no tensor_add.
                    # predicate(q_idx, k_idx) = offset + channel_id * channel_multiplier + x * step_x
                    #   = (qso - kso) + q_idx * 1 + k_idx * (-1)
                    #   = (qso + q_idx) - (kso + k_idx)
                    # keep where affine_value >= 0, i.e., q_pos >= k_pos (causal)
                    # pattern=[[-1, PMAX]]: step_x=-1, iterating over k positions in free dim
                    sm = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_sm_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.affine_select(
                        dst=sm,
                        on_true_tile=scs,
                        on_false_value=-3.4e38,
                        pattern=[[-1, PMAX]],
                        offset=qso - kso,
                        channel_multiplier=1,
                        cmp_op=nl.greater_equal,
                    )

                    # tile_max: reduce over free dim (S_k), result [PMAX, 1]
                    # PLAN C: Broadcast [PMAX,1] -> [PMAX,PMAX] via nc_matmul outer product.
                    # Old: 2 DMA round-trips (SBUF->HBM->SBUF) per broadcast.
                    # New: tensor engine outer product [PMAX,1] x [1,PMAX] -> [PMAX,PMAX], no HBM.
                    tmax1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_tmax1_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_reduce(dst=tmax1, op=nl.maximum, data=sm, axis=1)
                    # BROADCAST: SBUF .ap() pattern replicates [PMAX,1] -> [PMAX,PMAX].
                    # Avoids HBM round-trip (was: SBUF->HBM->SBUF, 2 DMA ops per tile).
                    # pattern=[[1,PMAX],[0,PMAX]]: step=1 over PMAX partitions (unchanged),
                    # step=0 (broadcast) over PMAX free-dim columns.
                    tmax = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_tmax_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.dma_copy(dst=tmax,
                                  src=tmax1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0)) # Profile: Causes Idle Time

                    # new_max = max(rmax, tmax) [PMAX, PMAX]
                    nmax = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_nmax_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(nmax, rmax, tmax, op=nl.maximum)

                    # neg_new_max = -new_max [PMAX, PMAX]
                    nnmax = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_nnmax_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_scalar(nnmax, nmax, op0=nl.multiply, operand0=-1.0)

                    # alpha = exp(rmax - new_max) [PMAX, PMAX]
                    aarg = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_aarg_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(aarg, rmax, nnmax, op=nl.add)
                    alp = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"p2_alp_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.activation(alp, op=nl.exp, data=aarg)

                    # Rescale running state
                    nrs = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"p2_nrs_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(nrs, rsum, alp, op=nl.multiply)

                    nacc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_nacc_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(nacc, atacc, alp, op=nl.multiply)

                    # score_exp = exp(score - new_max)
                    sshift = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_sshift_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(sshift, sm, nnmax, op=nl.add)
                    sexp = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_sexp_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.activation(sexp, op=nl.exp, data=sshift)

                    # tile_sum: reduce, then broadcast to [PMAX,PMAX] via nc_matmul outer product.
                    # PLAN C: Same pattern as tmax above - avoids 2 HBM round-trips per tile.
                    tsum1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_tsum1_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_reduce(dst=tsum1, op=nl.add, data=sexp, axis=1)
                    # BROADCAST: SBUF .ap() pattern replicates [PMAX,1] -> [PMAX,PMAX].
                    # Avoids HBM round-trip (was: SBUF->HBM->SBUF, 2 DMA ops per tile).
                    tsum = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_tsum_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.dma_copy(dst=tsum,
                                  src=tsum1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0)) # Profiler: Causes Idle Time

                    nrs2 = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_nrs2_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(nrs2, nrs, tsum, op=nl.add)

                    # MM2
                    sexpb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                       name=f"p2_sexpb_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_copy(sexpb, sexp)
                    sexpTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                        name=f"p2_sexpTp_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.nc_transpose(sexpTp, sexpb)
                    sexpTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                        name=f"p2_sexpTs_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_copy(sexpTs, sexpTp)

                    # PLAN A: V tile already in SBUF — no HBM DMA needed.
                    # V_tiles[ki] = vbf from Phase 1 si=ki iteration (same kvh).
                    # vt is already bf16 in SBUF; use directly in nc_matmul (moving position).
                    vt = V_tiles[ki]

                    vcp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                   name=f"p2_vcp_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.nc_matmul(vcp, stationary=sexpTs, moving=vt)
                    vc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_vc_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_copy(vc, vcp)

                    nacc2 = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_nacc2_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(nacc2, nacc, vc, op=nl.add)

                    nisa.tensor_copy(rmax, nmax)
                    nisa.tensor_copy(rsum, nrs2)
                    nisa.tensor_copy(atacc, nacc2)

                # Normalize: attn_out = atacc / rsum
                ssafe = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_ssafe_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(ssafe, rsum, op0=nl.add, operand0=1e-9)
                rsq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_rsq_kv{kvh}_g{gi}_s{qsi}")
                nisa.activation(rsq, op=nl.rsqrt, data=ssafe)
                inv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_inv_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(inv, rsq, rsq, op=nl.multiply)

                aout = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_aout_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(aout, atacc, inv, op=nl.multiply)

                aoutb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"p2_aoutb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(aoutb, aout)

                nisa.dma_copy(
                    dst=output_2d[qso:qso+PMAX, qh*d:qh*d+PMAX],
                    src=aoutb,
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
        print(f"qwen3_attn_cte_fused v4a Test: B={B}, S={S}, H={H}, d={d}")
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
