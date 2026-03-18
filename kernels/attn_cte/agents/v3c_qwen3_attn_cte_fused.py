"""
Fused NKI attention CTE kernel for Qwen3.
Fuses: QKV linear projection + per-head RMSNorm + RoPE + causal flash attention.

Plan C optimization (Two-Pass Architecture):
  Separate K/V Precompute + attn_mask Preload

  v2a architecture (single fused kvh loop):
    for kvh (affine_range):
      Phase 1: for si in range(): compute K/V tiles -> K_tiles/V_tiles in SBUF
      Phase 2: for gi (affine_range) -> for qsi (affine_range): flash attention
    Problem: Phase 1 must use range() because Phase 2 reads K_tiles[ki] in same kvh body.

  v3c architecture (three separate sequential passes):
    Step 0: Preload all attn_mask tiles into SBUF (25 tiles, 1.6MB)
    Step 1: K/V precompute pass — for kvh -> for si: compute K_bank[kvh][si], V_bank[kvh][si]
    Step 2: Flash attention pass — for kvh (affine_range) -> gi (affine_range) -> qsi (affine_range)
              uses K_bank[kvh][ki], V_bank[kvh][ki], mask_tiles[qsi][ki] all from SBUF

  Unique contributions of Plan C:
    (1) Preloads attn_mask eliminating 200 DMA ops in the ki loop (5*5*8 = 200 loads)
    (2) Separates K/V compute into a standalone pass, enabling future affine_range on K/V outer loop

  SBUF budget:
    K_bank: 4 * 5 * 32KB = 640KB
    V_bank: 4 * 5 * 32KB = 640KB
    mask_tiles: 5 * 5 * 64KB = 1.6MB
    Total static: ~2.88MB + computation temporaries (~3MB) = ~6MB well within 24MB
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
    attn_mask,       # [S, S] float32 - 0 for valid, -3.4e38 for masked
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

    output = nl.ndarray((B, S, Hq_out), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    hidden_2d = hidden_states.reshape((B * S, H))
    output_2d = output.reshape((B * S, Hq_out))

    # Load Q norm weights as [PMAX,1] float32 in SBUF, then broadcast to [PMAX,PMAX] via .ap()
    qnw_1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qnw_1")
    qnw_bf16_tmp = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_bf16_tmp")
    nisa.dma_copy(dst=qnw_bf16_tmp, src=q_norm_weight.reshape((PMAX, 1)))
    nisa.tensor_copy(qnw_1, qnw_bf16_tmp)
    qnw_sb = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="qnw_sb")
    nisa.dma_copy(dst=qnw_sb, src=qnw_1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

    # Load K norm weights similarly
    knw_1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="knw_1")
    knw_bf16_tmp = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_bf16_tmp")
    nisa.dma_copy(dst=knw_bf16_tmp, src=k_norm_weight.reshape((PMAX, 1)))
    nisa.tensor_copy(knw_1, knw_bf16_tmp)
    knw_sb = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="knw_sb")
    nisa.dma_copy(dst=knw_sb, src=knw_1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

    rms_ones = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)

    # =========================================================================
    # Step 0: Preload all attn_mask tiles
    # mask_tiles[qsi][ki] = [PMAX, PMAX] float32 SBUF
    # SBUF cost: num_s_tiles * num_s_tiles * 64KB = 25 * 64KB = 1.6MB
    # These are invariant to kvh and gi — load once, reuse across all heads
    # =========================================================================
    mask_tiles = []
    for qsi in range(num_s_tiles):
        row = []
        for ki in range(num_s_tiles):
            qso = qsi * PMAX
            kso = ki * PMAX
            mk = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"preload_mask_{qsi}_{ki}")
            nisa.dma_copy(dst=mk, src=attn_mask[qso:qso + PMAX, kso:kso + PMAX])
            row.append(mk)
        mask_tiles.append(row)

    # =========================================================================
    # Step 1: K/V precompute pass
    # Build K_bank[kvh][si] and V_bank[kvh][si] — all in SBUF
    # SBUF cost: Hkv_tp * num_s_tiles * 2 * 32KB = 4 * 5 * 2 * 32KB = 1.28MB
    # =========================================================================
    K_bank = []
    V_bank = []
    for kvh in range(Hkv_tp):
        K_bank.append([])
        V_bank.append([])
        for si in range(num_s_tiles):
            s_off = si * PMAX

            # Load RoPE cos/sin for this sequence tile
            cos_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"kv_cosbf_h{kvh}_s{si}")
            nisa.dma_copy(dst=cos_bf, src=cos[s_off:s_off + PMAX, 0:PMAX])
            sin_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"kv_sinbf_h{kvh}_s{si}")
            nisa.dma_copy(dst=sin_bf, src=sin[s_off:s_off + PMAX, 0:PMAX])
            cos_f = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"kv_cosf_h{kvh}_s{si}")
            nisa.tensor_copy(cos_f, cos_bf)
            sin_f = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"kv_sinf_h{kvh}_s{si}")
            nisa.tensor_copy(sin_f, sin_bf)

            # ---- K projection matmul ----
            kp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                          name=f"kv_kp_h{kvh}_s{si}")
            for ht in range(num_h_tiles):
                ho = ht * PMAX
                hb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"kv_hk_h{kvh}_s{si}_t{ht}")
                nisa.dma_copy(dst=hb, src=hidden_2d[s_off:s_off + PMAX, ho:ho + PMAX])
                hTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                 name=f"kv_hkTp_h{kvh}_s{si}_t{ht}")
                nisa.nc_transpose(hTp, hb)
                hTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"kv_hkTs_h{kvh}_s{si}_t{ht}")
                nisa.tensor_copy(hTs, hTp)
                wb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"kv_wk_h{kvh}_s{si}_t{ht}")
                nisa.dma_copy(dst=wb, src=Wk[kvh * d:kvh * d + PMAX, ho:ho + PMAX])
                wTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                 name=f"kv_wkTp_h{kvh}_s{si}_t{ht}")
                nisa.nc_transpose(wTp, wb)
                wTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"kv_wkTs_h{kvh}_s{si}_t{ht}")
                nisa.tensor_copy(wTs, wTp)
                nisa.nc_matmul(kp, stationary=hTs, moving=wTs)

            kv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"kv_kv_h{kvh}_s{si}")
            nisa.tensor_copy(kv, kp)

            # K RMSNorm in transposed [d,S] layout
            kvb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"kv_kvb_h{kvh}_s{si}")
            nisa.tensor_copy(kvb, kv)
            kvTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                              name=f"kv_kvTp_h{kvh}_s{si}")
            nisa.nc_transpose(kvTp, kvb)
            kvTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"kv_kvTb_h{kvh}_s{si}")
            nisa.tensor_copy(kvTb, kvTp)
            kvT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"kv_kvT_h{kvh}_s{si}")
            nisa.tensor_copy(kvT, kvTb)

            ksq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"kv_ksq_h{kvh}_s{si}")
            nisa.tensor_tensor(ksq, kvT, kvT, op=nl.multiply)
            ksqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"kv_ksqb_h{kvh}_s{si}")
            nisa.tensor_copy(ksqb, ksq)
            ksump = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                             name=f"kv_ksump_h{kvh}_s{si}")
            nisa.nc_matmul(ksump, stationary=rms_ones, moving=ksqb)
            ksum = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"kv_ksum_h{kvh}_s{si}")
            nisa.tensor_copy(ksum, ksump)

            kmean = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"kv_kmean_h{kvh}_s{si}")
            nisa.tensor_scalar(kmean, ksum, op0=nl.multiply, operand0=1.0 / d)
            kvar = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"kv_kvar_h{kvh}_s{si}")
            nisa.tensor_scalar(kvar, kmean, op0=nl.add, operand0=EPS)
            krinv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"kv_krinv_h{kvh}_s{si}")
            nisa.activation(krinv, op=nl.rsqrt, data=kvar)

            knT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"kv_knT_h{kvh}_s{si}")
            nisa.tensor_tensor(knT, kvT, krinv, op=nl.multiply)
            knwT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"kv_knwT_h{kvh}_s{si}")
            nisa.tensor_tensor(knwT, knT, knw_sb, op=nl.multiply)

            # Transpose back to [S, d]
            knwTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"kv_knwTb_h{kvh}_s{si}")
            nisa.tensor_copy(knwTb, knwT)
            knp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                             name=f"kv_knp_h{kvh}_s{si}")
            nisa.nc_transpose(knp, knwTb)
            knb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"kv_knb_h{kvh}_s{si}")
            nisa.tensor_copy(knb, knp)
            kn = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"kv_kn_h{kvh}_s{si}")
            nisa.tensor_copy(kn, knb)

            # K RoPE
            rotk = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"kv_rotk_h{kvh}_s{si}")
            negku = nl.ndarray((PMAX, half_d), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"kv_negku_h{kvh}_s{si}")
            nisa.tensor_scalar(negku, kn[0:PMAX, half_d:d], op0=nl.multiply, operand0=-1.0)
            nisa.tensor_copy(rotk[0:PMAX, 0:half_d], negku)
            nisa.tensor_copy(rotk[0:PMAX, half_d:d], kn[0:PMAX, 0:half_d])

            kcos = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"kv_kcos_h{kvh}_s{si}")
            nisa.tensor_tensor(kcos, kn, cos_f, op=nl.multiply)
            ksinp = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"kv_ksinp_h{kvh}_s{si}")
            nisa.tensor_tensor(ksinp, rotk, sin_f, op=nl.multiply)
            krope = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"kv_krope_h{kvh}_s{si}")
            nisa.tensor_tensor(krope, kcos, ksinp, op=nl.add)

            # Store final K tile in SBUF
            krbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"kv_krbf_h{kvh}_s{si}")
            nisa.tensor_copy(krbf, krope)
            K_bank[kvh].append(krbf)

            # ---- V projection (no RMSNorm) ----
            vp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                          name=f"kv_vp_h{kvh}_s{si}")
            for ht in range(num_h_tiles):
                ho = ht * PMAX
                hvb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"kv_hv_h{kvh}_s{si}_t{ht}")
                nisa.dma_copy(dst=hvb, src=hidden_2d[s_off:s_off + PMAX, ho:ho + PMAX])
                hvTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                  name=f"kv_hvTp_h{kvh}_s{si}_t{ht}")
                nisa.nc_transpose(hvTp, hvb)
                hvTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"kv_hvTs_h{kvh}_s{si}_t{ht}")
                nisa.tensor_copy(hvTs, hvTp)
                wvb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"kv_wv_h{kvh}_s{si}_t{ht}")
                nisa.dma_copy(dst=wvb, src=Wv[kvh * d:kvh * d + PMAX, ho:ho + PMAX])
                wvTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                  name=f"kv_wvTp_h{kvh}_s{si}_t{ht}")
                nisa.nc_transpose(wvTp, wvb)
                wvTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"kv_wvTs_h{kvh}_s{si}_t{ht}")
                nisa.tensor_copy(wvTs, wvTp)
                nisa.nc_matmul(vp, stationary=hvTs, moving=wvTs)

            vv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"kv_vv_h{kvh}_s{si}")
            nisa.tensor_copy(vv, vp)
            vbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"kv_vbf_h{kvh}_s{si}")
            nisa.tensor_copy(vbf, vv)
            V_bank[kvh].append(vbf)

    # =========================================================================
    # Step 2: Flash attention pass
    # All K_bank and V_bank are ready in SBUF, all mask_tiles are ready in SBUF
    # Loop order: kvh (affine_range) -> gi (affine_range) -> qsi (affine_range) -> ki (range)
    # =========================================================================
    for kvh in nl.affine_range(Hkv_tp):
        for gi in nl.affine_range(gqa):
            qh = kvh * gqa + gi

            for qsi in nl.affine_range(num_s_tiles):
                qso = qsi * PMAX

                # Q projection
                qp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                              name=f"fa_qp_kv{kvh}_g{gi}_s{qsi}")
                for ht in range(num_h_tiles):
                    ho = ht * PMAX
                    hqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                     name=f"fa_hq_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.dma_copy(dst=hqb, src=hidden_2d[qso:qso + PMAX, ho:ho + PMAX])
                    hqTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                      name=f"fa_hqTp_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.nc_transpose(hqTp, hqb)
                    hqTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                      name=f"fa_hqTs_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.tensor_copy(hqTs, hqTp)
                    wqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                     name=f"fa_wq_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.dma_copy(dst=wqb, src=Wq[qh * d:qh * d + PMAX, ho:ho + PMAX])
                    wqTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                      name=f"fa_wqTp_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.nc_transpose(wqTp, wqb)
                    wqTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                      name=f"fa_wqTs_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.tensor_copy(wqTs, wqTp)
                    nisa.nc_matmul(qp, stationary=hqTs, moving=wqTs)

                qvec = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"fa_qvec_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvec, qp)

                # Q RMSNorm in transposed [d, S]
                qvb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"fa_qvb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvb, qvec)
                qvTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                  name=f"fa_qvTp_kv{kvh}_g{gi}_s{qsi}")
                nisa.nc_transpose(qvTp, qvb)
                qvTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"fa_qvTb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvTb, qvTp)
                qvT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"fa_qvT_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvT, qvTb)

                qsq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"fa_qsq_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qsq, qvT, qvT, op=nl.multiply)
                qsqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"fa_qsqb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qsqb, qsq)
                qsump = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                 name=f"fa_qsump_kv{kvh}_g{gi}_s{qsi}")
                nisa.nc_matmul(qsump, stationary=rms_ones, moving=qsqb)
                qsum = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"fa_qsum_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qsum, qsump)

                qmean = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"fa_qmean_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qmean, qsum, op0=nl.multiply, operand0=1.0 / d)
                qvar = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"fa_qvar_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qvar, qmean, op0=nl.add, operand0=EPS)
                qrinv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"fa_qrinv_kv{kvh}_g{gi}_s{qsi}")
                nisa.activation(qrinv, op=nl.rsqrt, data=qvar)

                qnT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"fa_qnT_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qnT, qvT, qrinv, op=nl.multiply)
                qnwT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"fa_qnwT_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qnwT, qnT, qnw_sb, op=nl.multiply)

                qnwTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"fa_qnwTb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qnwTb, qnwT)
                qnp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                 name=f"fa_qnp_kv{kvh}_g{gi}_s{qsi}")
                nisa.nc_transpose(qnp, qnwTb)
                qnb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"fa_qnb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qnb, qnp)
                qn = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"fa_qn_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qn, qnb)

                # Q RoPE
                cosqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"fa_cosqb_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=cosqb, src=cos[qso:qso + PMAX, 0:PMAX])
                sinqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"fa_sinqb_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=sinqb, src=sin[qso:qso + PMAX, 0:PMAX])
                cosqf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"fa_cosqf_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(cosqf, cosqb)
                sinqf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"fa_sinqf_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(sinqf, sinqb)

                rotq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"fa_rotq_kv{kvh}_g{gi}_s{qsi}")
                negqu = nl.ndarray((PMAX, half_d), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"fa_negqu_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(negqu, qn[0:PMAX, half_d:d], op0=nl.multiply, operand0=-1.0)
                nisa.tensor_copy(rotq[0:PMAX, 0:half_d], negqu)
                nisa.tensor_copy(rotq[0:PMAX, half_d:d], qn[0:PMAX, 0:half_d])

                qcos = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"fa_qcos_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qcos, qn, cosqf, op=nl.multiply)
                qsinp = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"fa_qsinp_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qsinp, rotq, sinqf, op=nl.multiply)
                qrope = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"fa_qrope_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qrope, qcos, qsinp, op=nl.add)

                qsc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"fa_qsc_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qsc, qrope, op0=nl.multiply, operand0=scale)

                qscb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"fa_qscb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qscb, qsc)
                qTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                 name=f"fa_qTp_kv{kvh}_g{gi}_s{qsi}")
                nisa.nc_transpose(qTp, qscb)
                qTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"fa_qTs_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qTs, qTp)

                # Flash attention - all state [PMAX, PMAX]
                atacc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"fa_atacc_kv{kvh}_g{gi}_s{qsi}")
                nisa.memset(atacc, value=0.0)
                rmax = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"fa_rmax_kv{kvh}_g{gi}_s{qsi}")
                nisa.memset(rmax, value=-1e30)
                rsum = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"fa_rsum_kv{kvh}_g{gi}_s{qsi}")
                nisa.memset(rsum, value=0.0)

                for ki in range(num_s_tiles):
                    # K tile from SBUF bank — no HBM DMA
                    kt = K_bank[kvh][ki]
                    kTp2 = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                      name=f"fa_kTp2_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.nc_transpose(kTp2, kt)
                    kTs2 = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                      name=f"fa_kTs2_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_copy(kTs2, kTp2)

                    # MM1: score[S_q, S_k]
                    scp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                   name=f"fa_scp_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.nc_matmul(scp, stationary=qTs, moving=kTs2)
                    scs = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"fa_scs_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_copy(scs, scp)

                    # Causal mask — from SBUF preload, no HBM DMA
                    mk = mask_tiles[qsi][ki]
                    sm = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"fa_sm_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(sm, scs, mk, op=nl.add)

                    # tile_max: reduce, then broadcast [PMAX,1] -> [PMAX,PMAX] via .ap()
                    tmax1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"fa_tmax1_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_reduce(dst=tmax1, op=nl.maximum, data=sm, axis=1)
                    tmax = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"fa_tmax_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.dma_copy(dst=tmax,
                                  src=tmax1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                    # new_max = max(rmax, tmax)
                    nmax = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"fa_nmax_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(nmax, rmax, tmax, op=nl.maximum)

                    # neg_new_max = -new_max
                    nnmax = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"fa_nnmax_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_scalar(nnmax, nmax, op0=nl.multiply, operand0=-1.0)

                    # alpha = exp(rmax - new_max)
                    aarg = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"fa_aarg_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(aarg, rmax, nnmax, op=nl.add)
                    alp = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"fa_alp_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.activation(alp, op=nl.exp, data=aarg)

                    # Rescale running state
                    nrs = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"fa_nrs_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(nrs, rsum, alp, op=nl.multiply)

                    nacc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"fa_nacc_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(nacc, atacc, alp, op=nl.multiply)

                    # score_exp = exp(score - new_max)
                    sshift = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"fa_sshift_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(sshift, sm, nnmax, op=nl.add)
                    sexp = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"fa_sexp_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.activation(sexp, op=nl.exp, data=sshift)

                    # tile_sum: reduce, then broadcast [PMAX,1] -> [PMAX,PMAX] via .ap()
                    tsum1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"fa_tsum1_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_reduce(dst=tsum1, op=nl.add, data=sexp, axis=1)
                    tsum = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"fa_tsum_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.dma_copy(dst=tsum,
                                  src=tsum1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                    nrs2 = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"fa_nrs2_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(nrs2, nrs, tsum, op=nl.add)

                    # MM2
                    sexpb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                       name=f"fa_sexpb_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_copy(sexpb, sexp)
                    sexpTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                        name=f"fa_sexpTp_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.nc_transpose(sexpTp, sexpb)
                    sexpTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                        name=f"fa_sexpTs_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_copy(sexpTs, sexpTp)

                    # V tile from SBUF bank — no HBM DMA
                    vt = V_bank[kvh][ki]

                    vcp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                   name=f"fa_vcp_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.nc_matmul(vcp, stationary=sexpTs, moving=vt)
                    vc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"fa_vc_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_copy(vc, vcp)

                    nacc2 = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"fa_nacc2_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(nacc2, nacc, vc, op=nl.add)

                    nisa.tensor_copy(rmax, nmax)
                    nisa.tensor_copy(rsum, nrs2)
                    nisa.tensor_copy(atacc, nacc2)

                # Normalize: attn_out = atacc / rsum
                ssafe = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"fa_ssafe_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(ssafe, rsum, op0=nl.add, operand0=1e-9)
                rsq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"fa_rsq_kv{kvh}_g{gi}_s{qsi}")
                nisa.activation(rsq, op=nl.rsqrt, data=ssafe)
                inv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"fa_inv_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(inv, rsq, rsq, op=nl.multiply)

                aout = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"fa_aout_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(aout, atacc, inv, op=nl.multiply)

                aoutb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"fa_aoutb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(aoutb, aout)

                nisa.dma_copy(
                    dst=output_2d[qso:qso + PMAX, qh * d:qh * d + PMAX],
                    src=aoutb,
                )

    return output


# =============================================================================
# Test harness (copied unchanged from v2a)
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
        print(f"qwen3_attn_cte_fused Test: B={B}, S={S}, H={H}, d={d}")
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

        causal_mask = torch.triu(torch.full((S, S), -3.4e38, dtype=torch.float32), diagonal=1)

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
        mask_dev = causal_mask.to(device)

        print("\n[3/4] Running NKI kernel...")
        nki_out_dev = qwen3_attn_cte_fused(
            hidden_dev, Wq_dev, Wk_dev, Wv_dev,
            qnw_dev, knw_dev,
            cos_dev, sin_dev, mask_dev,
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
