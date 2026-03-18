"""
Fused NKI attention CTE kernel for Qwen3.
Fuses: QKV linear projection + per-head RMSNorm + RoPE + causal flash attention.

v8b architecture (v7ab + Plan B — Causal Compute Skipping):
  Builds on v7ab (score-softmax decoupling + narrow broadcast softmax) and adds
  causal compute skipping to eliminate score matmuls for (qsi, ki) pairs where
  ki > qsi (entirely masked by causal constraint).

  For causal attention with S=640 (num_s_tiles=5, BATCH=4):
    v7ab computed 10 score matmuls (5 wide + 5 tail for all qsi×ki pairs).
    v8b computes only 6 score matmuls by varying loop bounds per qsi:
      qsi=0: effective_batch=1, no tail → 1 matmul  (was 1 wide + 0 tail)
      qsi=1: effective_batch=2, no tail → 1 matmul  (was 1 wide + 0 tail)
      qsi=2: effective_batch=3, no tail → 1 matmul  (was 1 wide + 0 tail)
      qsi=3: effective_batch=4, no tail → 1 matmul  (was 1 wide + 0 tail)
      qsi=4: effective_batch=4, 1 tail  → 2 matmuls (was 1 wide + 1 tail)
    Total: 6 score matmuls (down from 10) — 40% reduction.

  Key implementation change:
    Pass 1 outer loop: range(num_s_tiles) instead of nl.affine_range(num_s_tiles).
      This makes qsi a concrete Python int at trace time, allowing variable inner
      loop bounds (effective_batch, tail range) to be computed per qsi.
    Pass 2 outer loop: same change for consistency and correct score tile indexing.

  Per-qsi variables:
    num_valid_k   = qsi + 1           — K tiles with any valid (unmasked) scores
    effective_batch = min(BATCH, num_valid_k) — tiles going into K_wide
    K_wide width  = effective_batch * PMAX  — varies per qsi
    tail range    = range(effective_batch, num_valid_k) — may be empty
    affine_select pattern width = effective_batch * PMAX — must match K_wide width

  SBUF layout: wide score tiles are now smaller for early qsi values, reducing
  peak SBUF pressure for the first few qsi iterations.

  Correctness: causal masking (affine_select) is still applied within valid tiles
  to handle the partially-masked diagonal tile (qsi == ki).

  All other optimizations from v7ab are preserved:
    - Phase fusion (single fused kvh loop, no K/V staging)
    - Pre-transposed weight layout (Plan B)
    - Hidden tile preload
    - Narrow broadcast softmax (v7b)
    - SBUF .ap() broadcast for norm weights and neg_max
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
    q_norm_weight,   # [d, 1] = [128, 1] bf16 — pre-shaped by caller to avoid NKI reshape on 1D HBM
    k_norm_weight,   # [d, 1] = [128, 1] bf16 — pre-shaped by caller to avoid NKI reshape on 1D HBM
    cos,             # [S, d] bf16 - RoPE cos per position
    sin,             # [S, d] bf16 - RoPE sin per position
    Hq_out=None,     # int: Hq_tp*d per TP rank; if None, falls back to Wq.shape[1] (standalone use)
    Hkv_out=None,    # int: Hkv_tp*d per TP rank; if None, falls back to Wk.shape[1] (standalone use)
):
    """Returns (output, K_cache, V_cache):
      output:  [B, S, Hq_tp*d] bf16 — attention output
      K_cache: [B, S, Hkv_tp*d] bf16 — post-RoPE, post-norm K values for KV cache update
      V_cache: [B, S, Hkv_tp*d] bf16 — projected V values (no norm/RoPE) for KV cache update"""
    B = hidden_states.shape[0]
    S = hidden_states.shape[1]
    H = hidden_states.shape[2]
    # Hq_out / Hkv_out are passed as Python ints from the model to avoid NKI tracer issues
    # with .shape[1] on TP-tagged weight tensors.  Fall back to shape access for standalone tests.
    if Hq_out is None:
        Hq_out = Wq.shape[1]    # Plan B: shape[1] (was shape[0] in v5a)
    if Hkv_out is None:
        Hkv_out = Wk.shape[1]   # Plan B: shape[1] (was shape[0] in v5a)
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

    # KV cache outputs: post-RoPE/norm K and projected V, both [B, S, Hkv_tp*d].
    # Stored in shared_hbm so the caller can update the KV cache after each decode step.
    K_cache = nl.ndarray((B, S, Hkv_out), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    V_cache = nl.ndarray((B, S, Hkv_out), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    hidden_2d = hidden_states.reshape((B * S, H))
    output_2d = output.reshape((B * S, Hq_out))
    K_cache_2d = K_cache.reshape((B * S, Hkv_out))  # [B*S, Hkv_tp*d] view for tiled stores
    V_cache_2d = V_cache.reshape((B * S, Hkv_out))  # [B*S, Hkv_tp*d] view for tiled stores

    # PLAN C: Broadcast [PMAX,1] SBUF -> [PMAX,PMAX] SBUF using .ap() pattern on SBUF source.
    # The .ap(pattern=[[1,PMAX],[0,PMAX]], offset=0) descriptor tells the DMA engine to:
    #   - repeat the single free-dim element (dim0: step=1, count=PMAX replicating partition axis)
    #   - repeat the 1-wide free dim to PMAX (dim1: step=0, count=PMAX — stride=0 = broadcast)
    # This is a SBUF-to-SBUF DMA with broadcast, avoiding ANY HBM round-trip.
    # Replaces the original bcast_hbm private_hbm staging buffer entirely.

    # Load Q norm weights as [PMAX,1] float32 in SBUF, then broadcast to [PMAX,PMAX] via .ap()
    qnw_1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qnw_1")
    qnw_bf16_tmp = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_bf16_tmp")
    nisa.dma_copy(dst=qnw_bf16_tmp, src=q_norm_weight)  # q_norm_weight is pre-shaped [PMAX,1]
    nisa.tensor_copy(qnw_1, qnw_bf16_tmp)
    # BROADCAST: SBUF .ap() pattern replicates [PMAX,1] -> [PMAX,PMAX] without HBM round-trip
    qnw_sb = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="qnw_sb")
    nisa.dma_copy(dst=qnw_sb, src=qnw_1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

    # Load K norm weights similarly
    knw_1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="knw_1")
    knw_bf16_tmp = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_bf16_tmp")
    nisa.dma_copy(dst=knw_bf16_tmp, src=k_norm_weight)  # k_norm_weight is pre-shaped [PMAX,1]
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
    # Plan B: Wq is [H, Hq_tp*d] pre-transposed — load [H_tile, d] directly, no nc_transpose needed.
    Wq_tiles_hoisted = []  # Wq_tiles_hoisted[qh][ht] = [PMAX, PMAX] bf16 SBUF tile
    for qh in nl.affine_range(Hq_tp):
        row = []
        for ht in nl.affine_range(num_h_tiles):
            ho = ht * PMAX
            # Plan B: load [H_tile, d] directly — Wq[ho:ho+PMAX, qh*d:qh*d+PMAX]
            # No nc_transpose or tensor_copy needed (was 3 ops, now 1 op per tile).
            wqTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"hoist_wqTs_h{qh}_t{ht}")
            nisa.dma_copy(dst=wqTs, src=Wq[ho:ho+PMAX, qh*d:qh*d+PMAX])
            row.append(wqTs)
        Wq_tiles_hoisted.append(row)

    # =========================================================================
    # v5a Change 1: Hidden tile preload.
    # Preload all num_s_tiles * num_h_tiles hidden tiles into SBUF as
    # pre-transposed [PMAX, PMAX] bf16 tensors before the kvh loop.
    # Eliminates redundant HBM loads of hidden tiles across all kvh iterations.
    # Both loops use affine_range for independent iterations -> DMA-compute overlap.
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
    # Outer loop uses affine_range to keep 4 kvh iterations independent.
    # =========================================================================
    for kvh in nl.affine_range(Hkv_tp):

        # -----------------------------------------------------------------
        # v3a: Hoist Wk/Wv outside the si loop.
        # Wk/Wv depend only on (kvh, ht), not si.
        # Plan B: Wk/Wv are [H, Hkv_tp*d] pre-transposed — load [H_tile, d] directly,
        # no nc_transpose needed (was 3 ops per tile, now 1 op per tile).
        # -----------------------------------------------------------------
        Wk_tiles_hoisted = []  # Wk_tiles_hoisted[ht] = [PMAX, PMAX] bf16 SBUF
        Wv_tiles_hoisted = []

        for ht in nl.affine_range(num_h_tiles):
            ho = ht * PMAX
            # K weight tile — Plan B: load [H_tile, d] directly from [H, Hkv_tp*d] layout
            wkTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"hoist_wkTs_kv{kvh}_t{ht}")
            nisa.dma_copy(dst=wkTs, src=Wk[ho:ho+PMAX, kvh*d:kvh*d+PMAX])
            Wk_tiles_hoisted.append(wkTs)

            # V weight tile — Plan B: load [H_tile, d] directly from [H, Hkv_tp*d] layout
            wvTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"hoist_wvTs_kv{kvh}_t{ht}")
            nisa.dma_copy(dst=wvTs, src=Wv[ho:ho+PMAX, kvh*d:kvh*d+PMAX])
            Wv_tiles_hoisted.append(wvTs)

        # -----------------------------------------------------------------
        # Phase 1 sub-loop: compute K/V tiles and store in SBUF Python lists.
        # v5a Change 2: Merged K+V ht loop — single pass reads hidden tiles
        # from SBUF (no HBM DMA) and accumulates both K and V projections.
        # Use affine_range for si (iterations are independent for accumulation
        # since each si writes to its own K_tiles[si]/V_tiles[si] entries).
        # -----------------------------------------------------------------
        K_tiles = []  # K_tiles[si] = krTs [PMAX, PMAX] bf16 (pre-transposed [d, S_k])
        V_tiles = []  # V_tiles[si] = vbf  [PMAX, PMAX] bf16

        for si in nl.affine_range(num_s_tiles):
            s_off = si * PMAX

            # v3a: Use hoisted cos/sin tiles — no HBM DMA per kvh iteration.
            cos_f = kv_cos_tiles[si]   # from SBUF — no HBM DMA
            sin_f = kv_sin_tiles[si]   # from SBUF — no HBM DMA

            # v5a Change 2: Merged K+V projection in single ht loop.
            # Both kp and vp are declared before the loop; each ht tile reads
            # hidden_tiles[si][ht] from SBUF (no HBM DMA) and accumulates both.
            kp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                          name=f"p1_kp_kv{kvh}_s{si}")
            vp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                          name=f"p1_vp_kv{kvh}_s{si}")

            for ht in nl.affine_range(num_h_tiles):
                hTs = hidden_tiles[si][ht]  # from SBUF — no HBM DMA
                # v3a: Use hoisted Wk/Wv tiles — no HBM DMA per si iteration.
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

            # Store post-RoPE, post-norm K tile to KV cache.
            # krbf is [PMAX, PMAX] bf16 in [S_tile, d] layout (non-transposed, unlike krTs).
            nisa.dma_copy(dst=K_cache_2d[s_off:s_off + PMAX, kvh * d:kvh * d + PMAX], src=krbf)

            # V: copy out of vp PSUM
            vv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_vv_kv{kvh}_s{si}")
            nisa.tensor_copy(vv, vp)
            vbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_vbf_kv{kvh}_s{si}")
            nisa.tensor_copy(vbf, vv)
            V_tiles.append(vbf)

            # Store V tile to KV cache (projected, no norm/RoPE).
            nisa.dma_copy(dst=V_cache_2d[s_off:s_off + PMAX, kvh * d:kvh * d + PMAX], src=vbf)

        # -----------------------------------------------------------------
        # Phase 2: Materialized softmax (v7ab — combined Plan A + Plan B)
        # v7a: Score-softmax decoupling — split into two passes.
        # v7b: Narrow broadcast + slices — applied within the softmax pass.
        # v8b: Causal compute skipping — outer loop uses range() not affine_range()
        #      so qsi is a concrete Python int, enabling variable loop bounds per qsi.
        # -----------------------------------------------------------------
        # BATCH: width of the wide score matmul (4 K tiles at once).
        # Clamp to num_s_tiles for small-S buckets (e.g. S=128 → BATCH=1).
        BATCH = min(4, num_s_tiles)

        for gi in nl.affine_range(gqa):
            qh = kvh * gqa + gi

            # =============================================================
            # v7a PASS 1: Pre-compute all score tiles (TensorE heavy).
            # v8b: Changed from nl.affine_range to range() so qsi is a
            #      concrete Python int at trace time, allowing per-qsi
            #      variable loop bounds for causal compute skipping.
            # =============================================================
            all_masked_wide = []   # all_masked_wide[qsi] = [PMAX, effective_batch*PMAX] f32 SBUF
            all_masked_tail = []   # all_masked_tail[qsi] = list of [PMAX, PMAX] f32 SBUF
            # Also store per-qsi effective_batch for Pass 2 to use
            all_effective_batch = []  # all_effective_batch[qsi] = int

            for qsi in range(num_s_tiles):  # range() not affine_range — qsi is a Python int
                qso = qsi * PMAX  # byte offset for this Q tile in the sequence dimension

                # v8b Causal Compute Skipping:
                # Under causal attention, K tile ki contributes non-masked scores only when ki <= qsi.
                # num_valid_k = qsi + 1 is the count of K tiles [0..qsi] that have valid scores.
                num_valid_k = qsi + 1  # Python int — number of causally-valid K tiles

                # How many K tiles go into the wide matmul (up to BATCH tiles).
                # For qsi=0: effective_batch=1 → K_wide is [PMAX, 128] (single tile, no waste).
                # For qsi=3+: effective_batch=4 → K_wide is [PMAX, 512] (full width, same as v7ab).
                effective_batch = min(BATCH, num_valid_k)  # Python int — varies per qsi

                all_effective_batch.append(effective_batch)  # save for Pass 2

                # === Q projection (identical to v7ab) ===
                qp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                              name=f"p2_qp_kv{kvh}_g{gi}_s{qsi}")
                for ht in nl.affine_range(num_h_tiles):
                    hqTs = hidden_tiles[qsi][ht]
                    nisa.nc_matmul(qp, stationary=hqTs, moving=Wq_tiles_hoisted[qh][ht])

                qvec = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qvec_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvec, qp)

                # === Q RMSNorm (identical to v7ab) ===
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

                # === Q RoPE (identical to v7ab) ===
                cosqf = q_cos_tiles[qsi]
                sinqf = q_sin_tiles[qsi]

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

                # Scale Q by 1/sqrt(d) for attention score computation
                qsc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_qsc_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qsc, qrope, op0=nl.multiply, operand0=scale)

                # Transpose Q to [d, S_q] for matmul with K (stationary position needs [d, S_q])
                qscb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"p2_qscb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qscb, qsc)
                qTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                 name=f"p2_qTp_kv{kvh}_g{gi}_s{qsi}")
                nisa.nc_transpose(qTp, qscb)
                qTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"p2_qTs_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qTs, qTp)

                # =============================================================
                # v8b CAUSAL COMPUTE SKIPPING: Wide score matmul
                # Instead of always using BATCH=4 tiles, use effective_batch
                # (at most num_valid_k = qsi+1 tiles) to skip fully-masked K tiles.
                #
                # K_wide shape: [PMAX, effective_batch * PMAX] — narrower for small qsi.
                #   qsi=0: [PMAX, 128]   (1 tile)
                #   qsi=1: [PMAX, 256]   (2 tiles)
                #   qsi=2: [PMAX, 384]   (3 tiles)
                #   qsi=3+: [PMAX, 512]  (4 tiles, same as v7ab)
                # =============================================================
                K_wide = nl.ndarray((PMAX, effective_batch * PMAX), dtype=nl.bfloat16,
                                    buffer=nl.sbuf,
                                    name=f"p2_Kwide_kv{kvh}_g{gi}_s{qsi}")
                # Only copy the effective_batch K tiles (ki=0..effective_batch-1)
                for i in range(effective_batch):
                    nisa.tensor_copy(K_wide[0:PMAX, i*PMAX:(i+1)*PMAX], K_tiles[i])

                # Wide matmul: Q [d, S_q] @ K_wide [d, effective_batch*S_k]
                # Output shape [PMAX, effective_batch*PMAX] — narrower for small qsi
                score_wide_psum = nl.zeros((PMAX, effective_batch * PMAX), dtype=nl.float32,
                                           buffer=nl.psum,
                                           name=f"p2_scwp_kv{kvh}_g{gi}_s{qsi}")
                nisa.nc_matmul(score_wide_psum, stationary=qTs, moving=K_wide)
                score_wide = nl.ndarray((PMAX, effective_batch * PMAX), dtype=nl.float32,
                                        buffer=nl.sbuf,
                                        name=f"p2_scw_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(score_wide, score_wide_psum)

                # Apply causal mask to wide scores.
                # affine_select pattern width must match the actual wide tile width.
                # offset=qso means: keep scores where (q_idx + qso) >= k_idx.
                masked_wide = nl.ndarray((PMAX, effective_batch * PMAX), dtype=nl.float32,
                                         buffer=nl.sbuf,
                                         name=f"p2_mw_kv{kvh}_g{gi}_s{qsi}")
                nisa.affine_select(
                    dst=masked_wide, on_true_tile=score_wide, on_false_value=-3.4e38,
                    # v8b: pattern width = effective_batch * PMAX (was always BATCH * PMAX in v7ab)
                    pattern=[[-1, effective_batch * PMAX]], offset=qso,
                    channel_multiplier=1, cmp_op=nl.greater_equal,
                )

                # =============================================================
                # v8b CAUSAL COMPUTE SKIPPING: Tail score tiles
                # Tail tiles are those in [effective_batch, num_valid_k).
                # In v7ab: tail_start = num_full_batches * BATCH (fixed for all qsi).
                # In v8b: effective_tail_start = effective_batch (per-qsi).
                #   - For qsi < BATCH-1: effective_tail_start = qsi+1 = num_valid_k → empty tail.
                #   - For qsi >= BATCH-1: effective_tail_start = BATCH, tail = [BATCH, qsi+1).
                # This skips ALL tail computation for early qsi values.
                # =============================================================
                masked_tail_tiles = []
                for ki_tail in range(effective_batch, num_valid_k):
                    # ki_tail is in [BATCH, qsi] — only enters when qsi >= BATCH
                    kso_t = ki_tail * PMAX  # byte offset for this K tile in the sequence dimension
                    scp_t = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                     name=f"p2_scpt_kv{kvh}_g{gi}_s{qsi}_t{ki_tail}")
                    nisa.nc_matmul(scp_t, stationary=qTs, moving=K_tiles[ki_tail])
                    scs_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_scst_kv{kvh}_g{gi}_s{qsi}_t{ki_tail}")
                    nisa.tensor_copy(scs_t, scp_t)
                    # Apply causal mask per tail tile; offset adjusted for this K tile's position
                    sm_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_smt_kv{kvh}_g{gi}_s{qsi}_t{ki_tail}")
                    nisa.affine_select(
                        dst=sm_t, on_true_tile=scs_t, on_false_value=-3.4e38,
                        pattern=[[-1, PMAX]], offset=qso - kso_t,
                        channel_multiplier=1, cmp_op=nl.greater_equal,
                    )
                    masked_tail_tiles.append(sm_t)

                # Store pre-computed masked score tiles for this qsi into outer lists
                all_masked_wide.append(masked_wide)
                all_masked_tail.append(masked_tail_tiles)

            # =============================================================
            # v7a PASS 2: Softmax + V accumulation (VectorE + DMA heavy).
            # Reads pre-computed masked scores from Pass 1.
            # v7b softmax: narrow [PMAX,PMAX] broadcast + per-slice processing.
            # v8b: Changed from nl.affine_range to range() to match Pass 1.
            #      Loops over only effective_batch slices (not always BATCH).
            # =============================================================
            for qsi in range(num_s_tiles):  # range() — qsi is a Python int for indexing
                qso = qsi * PMAX

                # Retrieve pre-computed masked score tiles and effective_batch for this qsi
                masked_wide = all_masked_wide[qsi]
                masked_tail_tiles = all_masked_tail[qsi]
                effective_batch = all_effective_batch[qsi]  # Python int — used for slice counts

                # --- Step 1: Global row maximum across all valid score tiles ---
                # tensor_reduce on the (possibly narrower) masked_wide
                row_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"p2_rowmax_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=masked_wide, axis=1)

                # Update row_max with maximums from each tail tile (may be empty for small qsi)
                for ti in range(len(masked_tail_tiles)):
                    sm_t = masked_tail_tiles[ti]
                    tmax_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_tmaxt_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_reduce(dst=tmax_t, op=nl.maximum, data=sm_t, axis=1)
                    nmax_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_nmaxt_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_tensor(nmax_t, row_max, tmax_t, op=nl.maximum)
                    nisa.tensor_copy(row_max, nmax_t)

                # --- Step 2: Broadcast neg_max, compute exp, accumulate ---
                neg_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"p2_negmax_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(neg_max, row_max, op0=nl.multiply, operand0=-1.0)

                # v7b: Single narrow broadcast of neg_max [PMAX,1] -> [PMAX,PMAX].
                # The same neg_max_bc is reused for all wide slices and tail tiles.
                neg_max_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_nmbc_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=neg_max_bc,
                              src=neg_max.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                # PSUM accumulator for exp(score) @ V (reused across all K tiles)
                atacc_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                      name=f"p2_atp_kv{kvh}_g{gi}_s{qsi}")

                # Initialize row_sum; will be set on first slice, accumulated thereafter.
                row_sum = None

                # v8b: Loop over effective_batch slices (not always BATCH).
                # For qsi=0: effective_batch=1 → only 1 slice.
                # For qsi>=3: effective_batch=4 → 4 slices (same as v7ab).
                for i in range(effective_batch):
                    # Extract the i-th [PMAX, PMAX] slice from the (possibly narrow) wide tile
                    slice_i = masked_wide[0:PMAX, i*PMAX:(i+1)*PMAX]

                    # Shift by neg_max: score_slice + (-max) = score_slice - max
                    shifted_i = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                           name=f"p2_shi_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_tensor(shifted_i, slice_i, neg_max_bc, op=nl.add)

                    # Compute exp(score - max) for this slice
                    sexp_i = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_sei_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.activation(sexp_i, op=nl.exp, data=shifted_i)

                    # Accumulate partial row sum: reduce this slice's exp values along axis=1
                    partial_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                             name=f"p2_psi_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_reduce(dst=partial_sum, op=nl.add, data=sexp_i, axis=1)

                    if i == 0:
                        # First slice: initialize row_sum directly
                        row_sum = partial_sum
                    else:
                        # Subsequent slices: add partial_sum into running row_sum
                        new_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                             name=f"p2_nsi_kv{kvh}_g{gi}_s{qsi}_i{i}")
                        nisa.tensor_tensor(new_sum, row_sum, partial_sum, op=nl.add)
                        row_sum = new_sum

                    # MM2 for this slice: transpose sexp_i and multiply with V_tiles[i]
                    sexp_i_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                           name=f"p2_seibf_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_copy(sexp_i_bf, sexp_i)
                    sexpT_i = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                         name=f"p2_seTi_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.nc_transpose(sexpT_i, sexp_i_bf)
                    sexpTs_i = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                          name=f"p2_seTsi_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_copy(sexpTs_i, sexpT_i)
                    # Accumulate: exp(score) @ V into attention output
                    nisa.nc_matmul(atacc_psum, stationary=sexpTs_i, moving=V_tiles[i])

                # v7b: Reuse neg_max_bc for tail tiles — no second broadcast needed.
                # v8b: Tail tiles loop over [effective_batch, num_valid_k) — same as Pass 1.
                for ti in range(len(masked_tail_tiles)):
                    sm_t = masked_tail_tiles[ti]
                    ki_abs = effective_batch + ti  # absolute K tile index for V_tiles lookup

                    # Shift by neg_max (reusing neg_max_bc from the single broadcast above)
                    shifted_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                           name=f"p2_sht_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_tensor(shifted_t, sm_t, neg_max_bc, op=nl.add)
                    sexp_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_set_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.activation(sexp_t, op=nl.exp, data=shifted_t)

                    # Accumulate this tail tile's exp sum into row_sum
                    tsum_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_tsumt_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_reduce(dst=tsum_t, op=nl.add, data=sexp_t, axis=1)
                    nrs_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_nrst_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_tensor(nrs_t, row_sum, tsum_t, op=nl.add)
                    row_sum = nrs_t

                    # MM2 for tail tile: transpose exp'd scores, matmul with V
                    sexp_t_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                           name=f"p2_setbf_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_copy(sexp_t_bf, sexp_t)
                    sexpT_t = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                         name=f"p2_seTt_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.nc_transpose(sexpT_t, sexp_t_bf)
                    sexpTs_t = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                          name=f"p2_seTst_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_copy(sexpTs_t, sexpT_t)
                    # V_tiles[ki_abs]: ki_abs = effective_batch + ti (the tail K tile index)
                    nisa.nc_matmul(atacc_psum, stationary=sexpTs_t, moving=V_tiles[ki_abs])

                # Copy accumulated attention output from PSUM to SBUF
                atacc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_atacc_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(atacc, atacc_psum)

                # --- Step 3: Normalize by row sum (softmax denominator) ---
                # Compute 1/row_sum via rsqrt(row_sum)^2, with epsilon for safety
                ssafe1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_ssafe1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(ssafe1, row_sum, op0=nl.add, operand0=1e-9)
                rsq1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_rsq1_kv{kvh}_g{gi}_s{qsi}")
                nisa.activation(rsq1, op=nl.rsqrt, data=ssafe1)
                inv1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_inv1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(inv1, rsq1, rsq1, op=nl.multiply)
                # Broadcast [PMAX,1] inverse -> [PMAX,PMAX] for element-wise normalization
                inv_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_inv_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=inv_bc,
                              src=inv1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                # Multiply accumulated V by 1/sum to get final softmax-weighted output
                aout = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_aout_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(aout, atacc, inv_bc, op=nl.multiply)

                # Cast to bf16 and store to HBM output
                aoutb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"p2_aoutb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(aoutb, aout)

                nisa.dma_copy(
                    dst=output_2d[qso:qso+PMAX, qh*d:qh*d+PMAX],
                    src=aoutb,
                )

    return output, K_cache, V_cache


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

        K_cache_ref = K.float().transpose(1, 2).reshape(B, S, Hkv_tp * d).to(torch.bfloat16)
        V_cache_ref = V.float().transpose(1, 2).reshape(B, S, Hkv_tp * d).to(torch.bfloat16)

        K_rep = K.repeat_interleave(gqa, dim=1)
        V_rep = V.float().repeat_interleave(gqa, dim=1)

        scores = (Q @ K_rep.transpose(-2, -1)) * scale_val
        causal = torch.triu(torch.full((S, S), float('-inf')), diagonal=1)
        scores = scores + causal

        P = torch.softmax(scores, dim=-1)
        out = P @ V_rep
        out = out.transpose(1, 2).reshape(B, S, Hq_tp * d)
        return out.to(torch.bfloat16), K_cache_ref, V_cache_ref

    def run_test(B, S, H=2048, d=128, Hq_tp=8, Hkv_tp=4):
        print(f"\n{'=' * 70}")
        print(f"qwen3_attn_cte_fused v8b Test: B={B}, S={S}, H={H}, d={d}")
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
        ref_out, K_cache_ref, V_cache_ref = pytorch_reference(
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
        nki_out_dev, K_cache_dev, V_cache_dev = qwen3_attn_cte_fused[2](
            hidden_dev, Wq_dev, Wk_dev, Wv_dev,
            qnw_dev, knw_dev,
            cos_dev, sin_dev,
        )
        xm.mark_step()
        nki_out = nki_out_dev.cpu()
        K_cache_nki = K_cache_dev.cpu()
        V_cache_nki = V_cache_dev.cpu()
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

        K_diff = (K_cache_ref.float() - K_cache_nki.float()).abs()
        V_diff = (V_cache_ref.float() - V_cache_nki.float()).abs()
        print(f"  K_cache Max  |diff|: {K_diff.max():.6e}")
        print(f"  K_cache Mean |diff|: {K_diff.mean():.6e}")
        print(f"  V_cache Max  |diff|: {V_diff.max():.6e}")
        print(f"  V_cache Mean |diff|: {V_diff.mean():.6e}")

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
        sys.exit(1)
