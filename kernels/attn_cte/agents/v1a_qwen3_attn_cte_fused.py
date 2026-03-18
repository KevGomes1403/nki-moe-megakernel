"""
Fused NKI attention CTE kernel for Qwen3.
Fuses: QKV linear projection + per-head RMSNorm + RoPE + causal flash attention.

Two-phase architecture:
  Phase 1: Project K/V + Norm + RoPE -> HBM scratch
  Phase 2: Per Q head - Project Q + Norm + RoPE + Flash Attention

Plan A optimisation (Phase 1):
  K/V projection weight tiles are invariant to sequence position (si).
  They were previously loaded inside si→kh→ht, causing num_s_tiles=5× redundant
  HBM reads per weight tile (640 extra 32KB loads ≈ 16.4MB).
  Plan A hoists the weight loads to kh→ht (once per KV head), storing the
  transposed tiles in SBUF arrays, and reuses them across all si iterations.
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
    K_scratch = nl.ndarray((Hkv_tp * S, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    V_scratch = nl.ndarray((Hkv_tp * S, d), dtype=nl.bfloat16, buffer=nl.private_hbm)

    hidden_2d = hidden_states.reshape((B * S, H))
    output_2d = output.reshape((B * S, Hq_out))

    # HBM staging buffer for [PMAX,1] -> [PMAX,PMAX] broadcast
    bcast_hbm = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.private_hbm)

    # Load norm weights as [PMAX,1] then broadcast to [PMAX,PMAX] via HBM staging
    qnw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_bf16")
    nisa.dma_copy(dst=qnw_bf16, src=q_norm_weight.reshape((PMAX, 1)))
    qnw_1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qnw_1")
    nisa.tensor_copy(qnw_1, qnw_bf16)
    nisa.dma_copy(dst=bcast_hbm, src=qnw_1)
    qnw_sb = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="qnw_sb")
    nisa.dma_copy(dst=qnw_sb, src=bcast_hbm.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

    knw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_bf16")
    nisa.dma_copy(dst=knw_bf16, src=k_norm_weight.reshape((PMAX, 1)))
    knw_1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="knw_1")
    nisa.tensor_copy(knw_1, knw_bf16)
    nisa.dma_copy(dst=bcast_hbm, src=knw_1)
    knw_sb = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="knw_sb")
    nisa.dma_copy(dst=knw_sb, src=bcast_hbm.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

    rms_ones = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)

    # =========================================================================
    # PHASE 1: Project K/V + Norm + RoPE -> HBM scratch
    #
    # Plan A restructuring: outer loop is now kh (KV head), not si (seq tile).
    # For each KV head we pre-load all num_h_tiles transposed Wk/Wv tiles once,
    # then reuse them across all num_s_tiles sequence iterations.
    # This eliminates num_s_tiles-1 redundant weight reloads per tile pair:
    #   Before: num_s_tiles × num_h_tiles loads = 5×16 = 80 per weight matrix
    #   After:  num_h_tiles loads = 16 per weight matrix  (5× reduction)
    # =========================================================================
    for kh in range(Hkv_tp):  # range() — ordering constraint for SBUF tile setup
        # -----------------------------------------------------------------
        # HOIST: Load all Wk weight tiles for this KV head once into SBUF.
        # These tiles are invariant to si (sequence tile), so loading them
        # inside the si loop caused num_s_tiles redundant DMA reads per tile.
        # Each tile is 128×128×2 bytes = 32 KB; hoisting saves
        # (num_s_tiles-1) × num_h_tiles × 32 KB = 4×16×32 KB ≈ 2 MB per head.
        # -----------------------------------------------------------------
        wkTs_tiles = []  # HOIST: pre-loaded Wk transposed tiles, indexed by ht
        for ht in range(num_h_tiles):
            ho = ht * PMAX
            wb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                            name=f"p1_wk_hoist_h{kh}_t{ht}")
            # HOIST: DMA from HBM once per KV head — not repeated for each si
            nisa.dma_copy(dst=wb, src=Wk[kh*d:kh*d+PMAX, ho:ho+PMAX])
            wTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                             name=f"p1_wkTp_hoist_h{kh}_t{ht}")
            nisa.nc_transpose(wTp, wb)
            wTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_wkTs_hoist_h{kh}_t{ht}")
            # HOIST: copy transposed tile to SBUF; stays live for all si
            nisa.tensor_copy(wTs, wTp)
            wkTs_tiles.append(wTs)

        # -----------------------------------------------------------------
        # HOIST: Load all Wv weight tiles for this KV head once into SBUF.
        # Same reasoning as Wk above — saves (num_s_tiles-1)×num_h_tiles DMA.
        # -----------------------------------------------------------------
        wvTs_tiles = []  # HOIST: pre-loaded Wv transposed tiles, indexed by ht
        for ht in range(num_h_tiles):
            ho = ht * PMAX
            wvb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_wv_hoist_h{kh}_t{ht}")
            # HOIST: DMA from HBM once per KV head — not repeated for each si
            nisa.dma_copy(dst=wvb, src=Wv[kh*d:kh*d+PMAX, ho:ho+PMAX])
            wvTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                              name=f"p1_wvTp_hoist_h{kh}_t{ht}")
            nisa.nc_transpose(wvTp, wvb)
            wvTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_wvTs_hoist_h{kh}_t{ht}")
            # HOIST: copy transposed tile to SBUF; stays live for all si
            nisa.tensor_copy(wvTs, wvTp)
            wvTs_tiles.append(wvTs)

        # si loop uses affine_range to preserve DMA-compute overlap for
        # hidden state loads (which do vary with si and cannot be hoisted).
        for si in nl.affine_range(num_s_tiles):
            s_off = si * PMAX

            cos_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"p1_cosbf_s{si}_h{kh}")
            nisa.dma_copy(dst=cos_bf, src=cos[s_off:s_off + PMAX, 0:PMAX])
            sin_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"p1_sinbf_s{si}_h{kh}")
            nisa.dma_copy(dst=sin_bf, src=sin[s_off:s_off + PMAX, 0:PMAX])
            cos_f = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_cosf_s{si}_h{kh}")
            nisa.tensor_copy(cos_f, cos_bf)
            sin_f = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_sinf_s{si}_h{kh}")
            nisa.tensor_copy(sin_f, sin_bf)

            # K projection — uses pre-loaded wkTs_tiles (no DMA inside si loop)
            kp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                          name=f"p1_kp_s{si}_h{kh}")

            for ht in range(num_h_tiles):
                ho = ht * PMAX
                hb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                name=f"p1_hk_s{si}_h{kh}_t{ht}")
                # Hidden state varies with si — must still DMA load here
                nisa.dma_copy(dst=hb, src=hidden_2d[s_off:s_off+PMAX, ho:ho+PMAX])
                hTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                 name=f"p1_hkTp_s{si}_h{kh}_t{ht}")
                nisa.nc_transpose(hTp, hb)
                hTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"p1_hkTs_s{si}_h{kh}_t{ht}")
                nisa.tensor_copy(hTs, hTp)
                # HOIST: use pre-loaded SBUF tile wkTs_tiles[ht] — no DMA needed
                nisa.nc_matmul(kp, stationary=hTs, moving=wkTs_tiles[ht])

            kv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_kv_s{si}_h{kh}")
            nisa.tensor_copy(kv, kp)

            # K RMSNorm in transposed [d,S] layout
            kvb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_kvb_s{si}_h{kh}")
            nisa.tensor_copy(kvb, kv)
            kvTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                              name=f"p1_kvTp_s{si}_h{kh}")
            nisa.nc_transpose(kvTp, kvb)
            kvTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_kvTb_s{si}_h{kh}")
            nisa.tensor_copy(kvTb, kvTp)
            kvT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"p1_kvT_s{si}_h{kh}")
            nisa.tensor_copy(kvT, kvTb)

            ksq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"p1_ksq_s{si}_h{kh}")
            nisa.tensor_tensor(ksq, kvT, kvT, op=nl.multiply)
            ksqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_ksqb_s{si}_h{kh}")
            nisa.tensor_copy(ksqb, ksq)
            ksump = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                             name=f"p1_ksump_s{si}_h{kh}")
            nisa.nc_matmul(ksump, stationary=rms_ones, moving=ksqb)
            ksum = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_ksum_s{si}_h{kh}")
            nisa.tensor_copy(ksum, ksump)

            kmean = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_kmean_s{si}_h{kh}")
            nisa.tensor_scalar(kmean, ksum, op0=nl.multiply, operand0=1.0/d)
            kvar = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_kvar_s{si}_h{kh}")
            nisa.tensor_scalar(kvar, kmean, op0=nl.add, operand0=EPS)
            krinv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_krinv_s{si}_h{kh}")
            nisa.activation(krinv, op=nl.rsqrt, data=kvar)

            knT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"p1_knT_s{si}_h{kh}")
            nisa.tensor_tensor(knT, kvT, krinv, op=nl.multiply)
            # Apply norm weight (both [PMAX, PMAX] now)
            knwT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_knwT_s{si}_h{kh}")
            nisa.tensor_tensor(knwT, knT, knw_sb, op=nl.multiply)

            # Transpose back to [S,d]
            knwTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"p1_knwTb_s{si}_h{kh}")
            nisa.tensor_copy(knwTb, knwT)
            knp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                             name=f"p1_knp_s{si}_h{kh}")
            nisa.nc_transpose(knp, knwTb)
            knb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_knb_s{si}_h{kh}")
            nisa.tensor_copy(knb, knp)
            kn = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_kn_s{si}_h{kh}")
            nisa.tensor_copy(kn, knb)

            # K RoPE
            rotk = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_rotk_s{si}_h{kh}")
            negku = nl.ndarray((PMAX, half_d), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_negku_s{si}_h{kh}")
            nisa.tensor_scalar(negku, kn[0:PMAX, half_d:d], op0=nl.multiply, operand0=-1.0)
            nisa.tensor_copy(rotk[0:PMAX, 0:half_d], negku)
            nisa.tensor_copy(rotk[0:PMAX, half_d:d], kn[0:PMAX, 0:half_d])

            kcos = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_kcos_s{si}_h{kh}")
            nisa.tensor_tensor(kcos, kn, cos_f, op=nl.multiply)
            ksinp = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_ksinp_s{si}_h{kh}")
            nisa.tensor_tensor(ksinp, rotk, sin_f, op=nl.multiply)
            krope = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_krope_s{si}_h{kh}")
            nisa.tensor_tensor(krope, kcos, ksinp, op=nl.add)

            krbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_krbf_s{si}_h{kh}")
            nisa.tensor_copy(krbf, krope)
            sr = kh * S + s_off
            nisa.dma_copy(dst=K_scratch[sr:sr+PMAX, 0:PMAX], src=krbf)

            # V projection — uses pre-loaded wvTs_tiles (no DMA inside si loop)
            vp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                          name=f"p1_vp_s{si}_h{kh}")
            for ht in range(num_h_tiles):
                ho = ht * PMAX
                hvb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"p1_hv_s{si}_h{kh}_t{ht}")
                # Hidden state varies with si — must still DMA load here
                nisa.dma_copy(dst=hvb, src=hidden_2d[s_off:s_off+PMAX, ho:ho+PMAX])
                hvTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                  name=f"p1_hvTp_s{si}_h{kh}_t{ht}")
                nisa.nc_transpose(hvTp, hvb)
                hvTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"p1_hvTs_s{si}_h{kh}_t{ht}")
                nisa.tensor_copy(hvTs, hvTp)
                # HOIST: use pre-loaded SBUF tile wvTs_tiles[ht] — no DMA needed
                nisa.nc_matmul(vp, stationary=hvTs, moving=wvTs_tiles[ht])

            vv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_vv_s{si}_h{kh}")
            nisa.tensor_copy(vv, vp)
            vbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_vbf_s{si}_h{kh}")
            nisa.tensor_copy(vbf, vv)
            nisa.dma_copy(dst=V_scratch[sr:sr+PMAX, 0:PMAX], src=vbf)

    # =========================================================================
    # PHASE 2: Per Q head flash attention
    # All state kept as [PMAX, PMAX] to avoid broadcast issues.
    # Scalars (max, sum, alpha) are [PMAX, PMAX] with identical columns.
    # =========================================================================
    for kvh in nl.affine_range(Hkv_tp):
        for gi in nl.affine_range(gqa):
            qh = kvh * gqa + gi

            for qsi in nl.affine_range(num_s_tiles):
                qso = qsi * PMAX

                # Q projection
                qp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                              name=f"p2_qp_kv{kvh}_g{gi}_s{qsi}")
                for ht in range(num_h_tiles):
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
                    wqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                     name=f"p2_wq_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.dma_copy(dst=wqb, src=Wq[qh*d:qh*d+PMAX, ho:ho+PMAX])
                    wqTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                      name=f"p2_wqTp_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.nc_transpose(wqTp, wqb)
                    wqTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                      name=f"p2_wqTs_kv{kvh}_g{gi}_s{qsi}_t{ht}")
                    nisa.tensor_copy(wqTs, wqTp)
                    nisa.nc_matmul(qp, stationary=hqTs, moving=wqTs)

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

                # Q RoPE
                cosqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"p2_cosqb_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=cosqb, src=cos[qso:qso+PMAX, 0:PMAX])
                sinqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"p2_sinqb_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=sinqb, src=sin[qso:qso+PMAX, 0:PMAX])
                cosqf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_cosqf_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(cosqf, cosqb)
                sinqf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_sinqf_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(sinqf, sinqb)

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

                for ki in range(num_s_tiles):
                    kso = ki * PMAX

                    kt = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                    name=f"p2_kt_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.dma_copy(dst=kt,
                                  src=K_scratch[kvh*S+kso:kvh*S+kso+PMAX, 0:PMAX])
                    kTp2 = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                      name=f"p2_kTp2_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.nc_transpose(kTp2, kt)
                    kTs2 = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                      name=f"p2_kTs2_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_copy(kTs2, kTp2)

                    # MM1: score[S_q, S_k]
                    scp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                   name=f"p2_scp_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.nc_matmul(scp, stationary=qTs, moving=kTs2)
                    scs = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"p2_scs_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_copy(scs, scp)

                    # Causal mask
                    mk = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_mk_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.dma_copy(dst=mk, src=attn_mask[qso:qso+PMAX, kso:kso+PMAX])
                    sm = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_sm_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_tensor(sm, scs, mk, op=nl.add)

                    # tile_max: reduce over free dim (S_k), result [PMAX, 1]
                    # Then broadcast to [PMAX, PMAX] via HBM staging
                    tmax1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_tmax1_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_reduce(dst=tmax1, op=nl.maximum, data=sm, axis=1)
                    nisa.dma_copy(dst=bcast_hbm, src=tmax1)
                    tmax = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_tmax_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.dma_copy(dst=tmax,
                                  src=bcast_hbm.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

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

                    # tile_sum: reduce, broadcast
                    tsum1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_tsum1_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_reduce(dst=tsum1, op=nl.add, data=sexp, axis=1)
                    nisa.dma_copy(dst=bcast_hbm, src=tsum1)
                    tsum = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_tsum_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.dma_copy(dst=tsum,
                                  src=bcast_hbm.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

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

                    vt = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                    name=f"p2_vt_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.dma_copy(dst=vt,
                                  src=V_scratch[kvh*S+kso:kvh*S+kso+PMAX, 0:PMAX])

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
