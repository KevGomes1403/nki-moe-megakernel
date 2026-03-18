"""
Fused NKI attention CTE kernel for Qwen3.
Fuses: QKV linear projection + per-head RMSNorm + RoPE + causal flash attention.

Stacks Plans B+C+D on top of v5b:

Plan B — TensorE Matmul Broadcast (pre-transposed weights, already in v5b):
  Wq, Wk, Wv stored pre-transposed in HBM [H, Hq/kv_tp*d] to load directly
  without nc_transpose. Also replaces SBUF-to-SBUF DMA broadcasts where possible.
  For the inner-loop broadcasts ([PMAX,1] → [PMAX,PMAX]), uses DMA .ap() from
  SBUF (proven approach from v5b) rather than nc_matmul which causes compiler errors.

Plan C — Free-Dimension RMSNorm:
  Replaces the transpose-heavy RMSNorm pattern with VectorE tensor_reduce in native
  [S,d] layout. Eliminates 4 nc_transpose + 4 tensor_copy ops per (kvh, si) for K,
  and 4 nc_transpose + 4 tensor_copy per (kvh, gi, qsi) for Q.
  Norm weights loaded as [1, PMAX] row, DMA-broadcast along partition axis to
  give qnw_sb[i,j] = weight[j] for all i (correct for [S,d] layout).

Plan D — Pre-Scored Flash Attention:
  Pre-computes ALL score tiles with affine_range before the sequential softmax loop.
  Decouples TensorE MM1 work from the serial VectorE dependency chain, enabling
  pipelining of all score computations.
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
    Wq,              # [H, Hq_tp*d] = [2048, 1024] bf16 — pre-transposed (Plan B)
    Wk,              # [H, Hkv_tp*d] = [2048, 512] bf16 — pre-transposed (Plan B)
    Wv,              # [H, Hkv_tp*d] = [2048, 512] bf16 — pre-transposed (Plan B)
    q_norm_weight,   # [d] = [128] bf16
    k_norm_weight,   # [d] = [128] bf16
    cos,             # [S, d] bf16 - RoPE cos per position
    sin,             # [S, d] bf16 - RoPE sin per position
):
    """Returns [B, S, Hq_tp*d] bf16"""
    B = hidden_states.shape[0]
    S = hidden_states.shape[1]
    H = hidden_states.shape[2]
    Hq_out = Wq.shape[1]    # Plan B: shape[1] (was shape[0] in v5a)
    Hkv_out = Wk.shape[1]   # Plan B: shape[1] (was shape[0] in v5a)
    d = PMAX
    Hq_tp = Hq_out // d
    Hkv_tp = Hkv_out // d
    gqa = Hq_tp // Hkv_tp
    num_h_tiles = H // PMAX
    num_s_tiles = S // PMAX
    half_d = d // 2
    scale = float(1.0 / math.sqrt(d))

    # Plan D: BATCH/num_full_batches computed here since they're used globally
    BATCH = 4
    num_full_batches = num_s_tiles // BATCH

    output = nl.ndarray((B, S, Hq_out), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    hidden_2d = hidden_states.reshape((B * S, H))
    output_2d = output.reshape((B * S, Hq_out))

    # =========================================================================
    # Plan C: Norm weight loading for [S,d] layout.
    # We need qnw_sb[i,j] = weight[j] for all i — weight varies along free dim (d).
    # Old v5b: qnw_sb[i,j] = weight[i] — weight varied along partition (for [d,S] layout).
    # Approach: load as [PMAX,1], broadcast via .ap() to get temp[p,f]=weight[p],
    #   then nc_transpose → qnw_sb[p,f] = temp[f,p] = weight[f] ✓
    # rms_ones removed (was used for old transpose+matmul reduction approach).
    # =========================================================================

    # Load Q norm weights: [PMAX,1] → broadcast [PMAX,PMAX] as temp[p,f]=weight[p]
    # then nc_transpose to get qnw_sb[p,f] = weight[f]
    qnw_1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qnw_1")
    qnw_bf16_tmp = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_bf16_tmp")
    nisa.dma_copy(dst=qnw_bf16_tmp, src=q_norm_weight.reshape((PMAX, 1)))
    nisa.tensor_copy(qnw_1, qnw_bf16_tmp)
    qnw_col = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="qnw_col")
    nisa.dma_copy(dst=qnw_col, src=qnw_1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))
    # qnw_col[p,f] = weight[p]; transpose → qnw_sb[p,f] = weight[f]
    qnw_col_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_col_bf")
    nisa.tensor_copy(qnw_col_bf, qnw_col)
    qnw_Tp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name="qnw_Tp")
    nisa.nc_transpose(qnw_Tp, qnw_col_bf)
    qnw_Tb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_Tb")
    nisa.tensor_copy(qnw_Tb, qnw_Tp)
    qnw_sb = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="qnw_sb")
    nisa.tensor_copy(qnw_sb, qnw_Tb)

    # Load K norm weights similarly
    knw_1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="knw_1")
    knw_bf16_tmp = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_bf16_tmp")
    nisa.dma_copy(dst=knw_bf16_tmp, src=k_norm_weight.reshape((PMAX, 1)))
    nisa.tensor_copy(knw_1, knw_bf16_tmp)
    knw_col = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="knw_col")
    nisa.dma_copy(dst=knw_col, src=knw_1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))
    knw_col_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_col_bf")
    nisa.tensor_copy(knw_col_bf, knw_col)
    knw_Tp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name="knw_Tp")
    nisa.nc_transpose(knw_Tp, knw_col_bf)
    knw_Tb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_Tb")
    nisa.tensor_copy(knw_Tb, knw_Tp)
    knw_sb = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="knw_sb")
    nisa.tensor_copy(knw_sb, knw_Tb)

    # =========================================================================
    # v3a HOISTING SECTION: Load cos/sin and Wq once before the kvh loop.
    # =========================================================================

    # Hoist K/V cos/sin: only depends on si, not on kvh.
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
    # Plan B: Wq is [H, Hq_tp*d] pre-transposed — load [H_tile, d] directly.
    Wq_tiles_hoisted = []  # Wq_tiles_hoisted[qh][ht] = [PMAX, PMAX] bf16 SBUF tile
    for qh in nl.affine_range(Hq_tp):
        row = []
        for ht in nl.affine_range(num_h_tiles):
            ho = ht * PMAX
            wqTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"hoist_wqTs_h{qh}_t{ht}")
            nisa.dma_copy(dst=wqTs, src=Wq[ho:ho+PMAX, qh*d:qh*d+PMAX])
            row.append(wqTs)
        Wq_tiles_hoisted.append(row)

    # =========================================================================
    # v5a Change 1: Hidden tile preload.
    # Preload all num_s_tiles * num_h_tiles hidden tiles into SBUF as
    # pre-transposed [PMAX, PMAX] bf16 tensors before the kvh loop.
    # =========================================================================
    hidden_tiles = []  # hidden_tiles[si][ht] = pre-transposed [PMAX, PMAX] bf16 SBUF tile
    for si in nl.affine_range(num_s_tiles):
        s_off = si * PMAX
        row = []
        for ht in nl.affine_range(num_h_tiles):
            ho = ht * PMAX
            hb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"hoist_hb_s{si}_t{ht}")
            nisa.dma_copy(dst=hb, src=hidden_2d[s_off:s_off+PMAX, ho:ho+PMAX])
            hTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                              name=f"hoist_hTp_s{si}_t{ht}")
            nisa.nc_transpose(hTp, hb)
            hTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"hoist_hTs_s{si}_t{ht}")
            nisa.tensor_copy(hTs, hTp)
            row.append(hTs)
        hidden_tiles.append(row)

    # =========================================================================
    # PLAN A: Single fused kvh loop (replaces separate Phase 1 + Phase 2 loops)
    # =========================================================================
    for kvh in nl.affine_range(Hkv_tp):

        # -----------------------------------------------------------------
        # v3a: Hoist Wk/Wv outside the si loop.
        # Plan B: Wk/Wv are [H, Hkv_tp*d] pre-transposed — load [H_tile, d] directly.
        # -----------------------------------------------------------------
        Wk_tiles_hoisted = []  # Wk_tiles_hoisted[ht] = [PMAX, PMAX] bf16 SBUF
        Wv_tiles_hoisted = []

        for ht in nl.affine_range(num_h_tiles):
            ho = ht * PMAX
            wkTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"hoist_wkTs_kv{kvh}_t{ht}")
            nisa.dma_copy(dst=wkTs, src=Wk[ho:ho+PMAX, kvh*d:kvh*d+PMAX])
            Wk_tiles_hoisted.append(wkTs)

            wvTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"hoist_wvTs_kv{kvh}_t{ht}")
            nisa.dma_copy(dst=wvTs, src=Wv[ho:ho+PMAX, kvh*d:kvh*d+PMAX])
            Wv_tiles_hoisted.append(wvTs)

        # -----------------------------------------------------------------
        # Phase 1 sub-loop: compute K/V tiles and store in SBUF Python lists.
        # -----------------------------------------------------------------
        K_tiles = []  # K_tiles[si] = krTs [PMAX, PMAX] bf16 (pre-transposed [d, S_k])
        V_tiles = []  # V_tiles[si] = vbf  [PMAX, PMAX] bf16

        for si in nl.affine_range(num_s_tiles):
            s_off = si * PMAX

            cos_f = kv_cos_tiles[si]
            sin_f = kv_sin_tiles[si]

            kp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                          name=f"p1_kp_kv{kvh}_s{si}")
            vp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                          name=f"p1_vp_kv{kvh}_s{si}")

            for ht in nl.affine_range(num_h_tiles):
                hTs = hidden_tiles[si][ht]
                nisa.nc_matmul(kp, stationary=hTs, moving=Wk_tiles_hoisted[ht])
                nisa.nc_matmul(vp, stationary=hTs, moving=Wv_tiles_hoisted[ht])

            kv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_kv_kv{kvh}_s{si}")
            nisa.tensor_copy(kv, kp)

            # ----------------------------------------------------------------
            # Plan C: K RMSNorm — free-dim VectorE reduction in [S,d] layout.
            # Eliminates 4 nc_transpose + 4 tensor_copy vs v5b.
            # ----------------------------------------------------------------
            ksq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"p1_ksq_kv{kvh}_s{si}")
            nisa.tensor_tensor(ksq, kv, kv, op=nl.multiply)
            ksum1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_ksum1_kv{kvh}_s{si}")
            nisa.tensor_reduce(dst=ksum1, op=nl.add, data=ksq, axis=1)
            kmean1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_kmean1_kv{kvh}_s{si}")
            nisa.tensor_scalar(kmean1, ksum1, op0=nl.multiply, operand0=1.0/d)
            kvar1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_kvar1_kv{kvh}_s{si}")
            nisa.tensor_scalar(kvar1, kmean1, op0=nl.add, operand0=EPS)
            krinv1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"p1_krinv1_kv{kvh}_s{si}")
            nisa.activation(krinv1, op=nl.rsqrt, data=kvar1)

            # Broadcast krinv1 [PMAX,1] → [PMAX,PMAX] via SBUF DMA .ap()
            krinv_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p1_krinvbc_kv{kvh}_s{si}")
            nisa.dma_copy(dst=krinv_bc,
                          src=krinv1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

            kn_c = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_knc_kv{kvh}_s{si}")
            nisa.tensor_tensor(kn_c, kv, krinv_bc, op=nl.multiply)
            knw = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"p1_knw_kv{kvh}_s{si}")
            nisa.tensor_tensor(knw, kn_c, knw_sb, op=nl.multiply)
            # knw is in [S, d] layout — ready for RoPE (no transpose needed)

            # K RoPE — uses knw directly (Plan C output in [S,d] layout)
            rotk = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_rotk_kv{kvh}_s{si}")
            negku = nl.ndarray((PMAX, half_d), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_negku_kv{kvh}_s{si}")
            nisa.tensor_scalar(negku, knw[0:PMAX, half_d:d], op0=nl.multiply, operand0=-1.0)
            nisa.tensor_copy(rotk[0:PMAX, 0:half_d], negku)
            nisa.tensor_copy(rotk[0:PMAX, half_d:d], knw[0:PMAX, 0:half_d])

            kcos = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_kcos_kv{kvh}_s{si}")
            nisa.tensor_tensor(kcos, knw, cos_f, op=nl.multiply)
            ksinp = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_ksinp_kv{kvh}_s{si}")
            nisa.tensor_tensor(ksinp, rotk, sin_f, op=nl.multiply)
            krope = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_krope_kv{kvh}_s{si}")
            nisa.tensor_tensor(krope, kcos, ksinp, op=nl.add)

            # v4a Change 1: Store K tile pre-transposed [d, S_k] in SBUF.
            krbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_krbf_kv{kvh}_s{si}")
            nisa.tensor_copy(krbf, krope)
            krT_psum = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                  name=f"p1_krTp_kv{kvh}_s{si}")
            nisa.nc_transpose(krT_psum, krbf)
            krTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_krTs_kv{kvh}_s{si}")
            nisa.tensor_copy(krTs, krT_psum)
            K_tiles.append(krTs)  # [d=128, S_k=128] — ready for MM1 moving position

            # V: copy out of vp PSUM
            vv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_vv_kv{kvh}_s{si}")
            nisa.tensor_copy(vv, vp)
            vbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_vbf_kv{kvh}_s{si}")
            nisa.tensor_copy(vbf, vv)
            V_tiles.append(vbf)

        # -----------------------------------------------------------------
        # Phase 2 sub-loop: Per Q head flash attention
        # K_tiles[ki] and V_tiles[ki] are already in SBUF.
        # -----------------------------------------------------------------
        for gi in nl.affine_range(gqa):
            qh = kvh * gqa + gi

            for qsi in nl.affine_range(num_s_tiles):
                qso = qsi * PMAX

                # Q projection
                qp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                              name=f"p2_qp_kv{kvh}_g{gi}_s{qsi}")
                for ht in nl.affine_range(num_h_tiles):
                    hqTs = hidden_tiles[qsi][ht]
                    nisa.nc_matmul(qp, stationary=hqTs, moving=Wq_tiles_hoisted[qh][ht])

                qvec = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qvec_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvec, qp)

                # ----------------------------------------------------------------
                # Plan C: Q RMSNorm — free-dim VectorE reduction in [S,d] layout.
                # Eliminates 4 nc_transpose + 4 tensor_copy vs v5b.
                # ----------------------------------------------------------------
                qsq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_qsq_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qsq, qvec, qvec, op=nl.multiply)
                qsum1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_qsum1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_reduce(dst=qsum1, op=nl.add, data=qsq, axis=1)
                qmean1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_qmean1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qmean1, qsum1, op0=nl.multiply, operand0=1.0/d)
                qvar1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_qvar1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qvar1, qmean1, op0=nl.add, operand0=EPS)
                qrinv1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_qrinv1_kv{kvh}_g{gi}_s{qsi}")
                nisa.activation(qrinv1, op=nl.rsqrt, data=qvar1)

                # Broadcast qrinv1 [PMAX,1] → [PMAX,PMAX] via SBUF DMA .ap()
                qrinv_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_qrinvbc_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=qrinv_bc,
                              src=qrinv1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                qn = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"p2_qn_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qn, qvec, qrinv_bc, op=nl.multiply)
                qnw = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_qnw_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qnw, qn, qnw_sb, op=nl.multiply)
                # qnw is in [S, d] layout — ready for RoPE

                # Q RoPE — uses qnw (Plan C output in [S,d] layout)
                cosqf = q_cos_tiles[qsi]
                sinqf = q_sin_tiles[qsi]

                rotq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_rotq_kv{kvh}_g{gi}_s{qsi}")
                negqu = nl.ndarray((PMAX, half_d), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_negqu_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(negqu, qnw[0:PMAX, half_d:d], op0=nl.multiply, operand0=-1.0)
                nisa.tensor_copy(rotq[0:PMAX, 0:half_d], negqu)
                nisa.tensor_copy(rotq[0:PMAX, half_d:d], qnw[0:PMAX, 0:half_d])

                qcos = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qcos_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qcos, qnw, cosqf, op=nl.multiply)
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

                # ================================================================
                # Plan D: Pre-compute ALL score tiles with affine_range.
                # TensorE can pipeline all MM1s since iterations are independent.
                # ================================================================
                masked_list = []
                for ki in nl.affine_range(num_s_tiles):
                    kso_ki = ki * PMAX
                    scp_ki = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                      name=f"p2_scpki_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.nc_matmul(scp_ki, stationary=qTs, moving=K_tiles[ki])
                    scs_ki = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_scski_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.tensor_copy(scs_ki, scp_ki)
                    sm_ki = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_smki_kv{kvh}_g{gi}_s{qsi}_k{ki}")
                    nisa.affine_select(
                        dst=sm_ki, on_true_tile=scs_ki, on_false_value=-3.4e38,
                        pattern=[[-1, PMAX]], offset=qso - kso_ki,
                        channel_multiplier=1, cmp_op=nl.greater_equal,
                    )
                    masked_list.append(sm_ki)

                # ================================================================
                # Online softmax using pre-computed scores (Plan D).
                # Broadcasts via SBUF DMA .ap() (proven from v5b).
                # ================================================================
                atacc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_atacc_kv{kvh}_g{gi}_s{qsi}")
                nisa.memset(atacc, value=0.0)
                rmax = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_rmax_kv{kvh}_g{gi}_s{qsi}")
                nisa.memset(rmax, value=-1e30)
                rsum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_rsum_kv{kvh}_g{gi}_s{qsi}")
                nisa.memset(rsum, value=0.0)

                # Batch loop: assemble 4 pre-computed masked scores into wide tile
                for ki_batch in range(num_full_batches):
                    sm_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                         name=f"p2_smw_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    for i in range(BATCH):
                        nisa.tensor_copy(sm_wide[0:PMAX, i*PMAX:(i+1)*PMAX],
                                         masked_list[ki_batch * BATCH + i])

                    # tile_max: reduce over BATCH*PMAX → [PMAX, 1]
                    tmax1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_tmax1w_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.tensor_reduce(dst=tmax1, op=nl.maximum, data=sm_wide, axis=1)

                    # new_max [PMAX, 1]
                    nmax1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_nmax1w_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.tensor_tensor(nmax1, rmax, tmax1, op=nl.maximum)

                    # neg_new_max [PMAX, 1]
                    nnmax1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_nnmax1w_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.tensor_scalar(nnmax1, nmax1, op0=nl.multiply, operand0=-1.0)

                    # alpha [PMAX, 1] = exp(rmax + neg_new_max)
                    aarg1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_aarg1w_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.tensor_tensor(aarg1, rmax, nnmax1, op=nl.add)
                    alp1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_alp1w_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.activation(alp1, op=nl.exp, data=aarg1)

                    # Rescale running sum [PMAX, 1]
                    nrs1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_nrs1w_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.tensor_tensor(nrs1, rsum, alp1, op=nl.multiply)

                    # Broadcast alp1 [PMAX,1] → [PMAX,PMAX] via SBUF DMA .ap()
                    alp_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_alpbc_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.dma_copy(dst=alp_bc,
                                  src=alp1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                    # Rescale output accumulator atacc [PMAX, PMAX] by alpha
                    nacc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_nacc_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.tensor_tensor(nacc, atacc, alp_bc, op=nl.multiply)

                    # Broadcast nnmax1 [PMAX,1] → [PMAX, BATCH*PMAX] via SBUF DMA .ap()
                    nnmax_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                            name=f"p2_nnmw_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.dma_copy(dst=nnmax_wide,
                                  src=nnmax1.ap(pattern=[[1, PMAX], [0, BATCH * PMAX]], offset=0))

                    # score_exp_wide = exp(sm_wide - new_max)
                    sshift_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                             name=f"p2_sshw_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.tensor_tensor(sshift_wide, sm_wide, nnmax_wide, op=nl.add)
                    sexp_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                           name=f"p2_sew_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.activation(sexp_wide, op=nl.exp, data=sshift_wide)

                    # tile_sum: reduce over BATCH*PMAX → [PMAX, 1]
                    tsum1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_tsum1w_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.tensor_reduce(dst=tsum1, op=nl.add, data=sexp_wide, axis=1)

                    # new running sum [PMAX, 1]
                    nrs2 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_nrs2w_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.tensor_tensor(nrs2, nrs1, tsum1, op=nl.add)

                    # MM2: accumulate BATCH V tiles into single PSUM
                    vcp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                   name=f"p2_vcpw_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    for i in range(BATCH):
                        sexp_i_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                               name=f"p2_seibf_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}_i{i}")
                        nisa.tensor_copy(sexp_i_bf, sexp_wide[0:PMAX, i*PMAX:(i+1)*PMAX])
                        sexpT_i = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                             name=f"p2_seTi_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}_i{i}")
                        nisa.nc_transpose(sexpT_i, sexp_i_bf)
                        sexpTs_i = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                              name=f"p2_seTsi_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}_i{i}")
                        nisa.tensor_copy(sexpTs_i, sexpT_i)
                        nisa.nc_matmul(vcp, stationary=sexpTs_i,
                                       moving=V_tiles[ki_batch * BATCH + i])

                    vc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_vcw_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.tensor_copy(vc, vcp)

                    nacc2 = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_nacc2w_kv{kvh}_g{gi}_s{qsi}_b{ki_batch}")
                    nisa.tensor_tensor(nacc2, nacc, vc, op=nl.add)

                    # Update running state
                    nisa.tensor_copy(rmax, nmax1)
                    nisa.tensor_copy(rsum, nrs2)
                    nisa.tensor_copy(atacc, nacc2)

                # Tail loop: handles remaining tiles (Plan D: use pre-computed masked scores)
                for ki_tail in range(num_full_batches * BATCH, num_s_tiles):
                    sm = masked_list[ki_tail]  # pre-computed score (Plan D)

                    # tile_max [PMAX, 1]
                    tmax1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_tmax1_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.tensor_reduce(dst=tmax1, op=nl.maximum, data=sm, axis=1)

                    # new_max [PMAX, 1]
                    nmax1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_nmax1_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.tensor_tensor(nmax1, rmax, tmax1, op=nl.maximum)

                    # neg_new_max [PMAX, 1]
                    nnmax1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_nnmax1_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.tensor_scalar(nnmax1, nmax1, op0=nl.multiply, operand0=-1.0)

                    # alpha [PMAX, 1] = exp(rmax + neg_new_max)
                    aarg1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_aarg1_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.tensor_tensor(aarg1, rmax, nnmax1, op=nl.add)
                    alp1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_alp1_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.activation(alp1, op=nl.exp, data=aarg1)

                    # Rescale running sum [PMAX, 1]
                    nrs1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_nrs1_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.tensor_tensor(nrs1, rsum, alp1, op=nl.multiply)

                    # Broadcast alp1 [PMAX,1] → [PMAX,PMAX] via SBUF DMA .ap()
                    alp_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_alpbc_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.dma_copy(dst=alp_bc,
                                  src=alp1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                    # Rescale output accumulator atacc [PMAX, PMAX] by alpha
                    nacc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_nacc_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.tensor_tensor(nacc, atacc, alp_bc, op=nl.multiply)

                    # Broadcast nnmax1 [PMAX,1] → [PMAX,PMAX] via SBUF DMA .ap()
                    nnmax_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                          name=f"p2_nnmbc_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.dma_copy(dst=nnmax_bc,
                                  src=nnmax1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                    # score_exp = exp(sm - new_max)
                    sshift = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_ssh_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.tensor_tensor(sshift, sm, nnmax_bc, op=nl.add)
                    sexp = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_se_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.activation(sexp, op=nl.exp, data=sshift)

                    # tile_sum [PMAX, 1]
                    tsum1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_tsum1_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.tensor_reduce(dst=tsum1, op=nl.add, data=sexp, axis=1)

                    # new running sum [PMAX, 1]
                    nrs2 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_nrs2_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.tensor_tensor(nrs2, nrs1, tsum1, op=nl.add)

                    # MM2
                    sexpb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                       name=f"p2_seb_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.tensor_copy(sexpb, sexp)
                    sexpTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                        name=f"p2_seTp_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.nc_transpose(sexpTp, sexpb)
                    sexpTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                        name=f"p2_seTs_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.tensor_copy(sexpTs, sexpTp)

                    vcp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                   name=f"p2_vcp_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.nc_matmul(vcp, stationary=sexpTs, moving=V_tiles[ki_tail])
                    vc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_vc_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.tensor_copy(vc, vcp)

                    nacc2 = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_nacc2_kv{kvh}_g{gi}_s{qsi}_kt{ki_tail}")
                    nisa.tensor_tensor(nacc2, nacc, vc, op=nl.add)

                    # Update running state
                    nisa.tensor_copy(rmax, nmax1)
                    nisa.tensor_copy(rsum, nrs2)
                    nisa.tensor_copy(atacc, nacc2)

                # Normalization
                ssafe1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_ssafe1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(ssafe1, rsum, op0=nl.add, operand0=1e-9)
                rsq1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_rsq1_kv{kvh}_g{gi}_s{qsi}")
                nisa.activation(rsq1, op=nl.rsqrt, data=ssafe1)
                inv1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_inv1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(inv1, rsq1, rsq1, op=nl.multiply)

                # Broadcast inv1 [PMAX,1] → [PMAX,PMAX] via SBUF DMA .ap()
                inv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_inv_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=inv,
                              src=inv1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

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
        print(f"qwen3_attn_cte_fused v6bcd Test: B={B}, S={S}, H={H}, d={d}")
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
        # Plan B: pass pre-transposed weights [H, Hq/kv_tp*d]
        Wq_dev = Wq.T.contiguous().to(device)   # [H, Hq_tp*d]
        Wk_dev = Wk.T.contiguous().to(device)   # [H, Hkv_tp*d]
        Wv_dev = Wv.T.contiguous().to(device)   # [H, Hkv_tp*d]
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
            print(f"\n  assert max_diff < 0.05  PASS: max_diff={max_diff:.4e} < {threshold}")
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
