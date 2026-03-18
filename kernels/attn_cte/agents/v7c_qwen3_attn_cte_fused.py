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

  v5a architecture (v4a + hidden tile preload + merged K+V loop + 4x-wide flash attention):
    Change 1: Hidden tile preload.
      Preloads all num_s_tiles * num_h_tiles hidden tiles into SBUF as pre-transposed
      [PMAX, PMAX] bf16 before the kvh loop. Uses affine_range for both loops to enable
      DMA-compute overlap. Eliminates redundant HBM loads of hidden tiles across kvh.
    Change 2: Merged K+V ht loop in Phase 1.
      Single merged ht loop computes both K and V projections simultaneously using shared
      hidden tiles from SBUF (no HBM DMA per iteration).
    Change 3: [PMAX, 1] running state in flash attention.
      rmax and rsum are now [PMAX, 1] scalars (one per token row) rather than [PMAX, PMAX]
      (identical columns). Reduces SBUF footprint for running state by 128x.
    Change 4: 4x-wide batched ki loop in flash attention.
      Groups 4 consecutive K tiles into a single [PMAX, 4*PMAX] K_wide matmul, computing
      4 score columns in one MM1 pass. Reduces MM1 launch overhead by 4x. Tail loop handles
      remaining tiles using scalar [PMAX,1] running state arithmetic with broadcasts.
    Change 5: Normalization adapted for [PMAX, 1] rsum.
      Computes inv as [PMAX, 1] then broadcasts to [PMAX, PMAX] for final multiplication.

  SBUF budget per kvh iteration:
    hidden_tiles: num_s_tiles * num_h_tiles * [PMAX, PMAX] bf16 = 5*16*32KB = 2.5MB (for S=640, H=2048)
    K_tiles: num_s_tiles * [PMAX, PMAX] bf16 = 5 * 32KB = 160KB (for S=640)
    V_tiles: num_s_tiles * [PMAX, PMAX] bf16 = 160KB
    Computation temporaries: same as v4a
    Total extra: ~2.82MB/kvh — well within 24MB budget.

Plan B optimization (Pre-transposed Weight Layout):
  v5b architecture (v5a + pre-transposed Wq/Wk/Wv in HBM):
    Wq, Wk, Wv are stored pre-transposed in HBM so the kernel loads them directly
    into the correct [H_tile, d] layout without any nc_transpose + tensor_copy.

    v5a weight hoisting (per tile):
      dma_copy [d, H_tile] from HBM -> nc_transpose -> tensor_copy to SBUF (3 ops per tile)

    v5b weight hoisting (per tile):
      dma_copy [H_tile, d] from HBM directly to SBUF (1 op per tile)

    Eliminated ops:
      Wq: Hq_tp * num_h_tiles = 8 * 16 = 128 nc_transpose + 128 tensor_copy
      Wk: Hkv_tp * num_h_tiles = 4 * 16 = 64 nc_transpose + 64 tensor_copy
      Wv: Hkv_tp * num_h_tiles = 4 * 16 = 64 nc_transpose + 64 tensor_copy
      Total: 256 nc_transpose + 256 tensor_copy = 512 ops eliminated.

    New HBM weight layout:
      Wq: [H, Hq_tp*d]   (transposed from [Hq_tp*d, H])
      Wk: [H, Hkv_tp*d]  (transposed from [Hkv_tp*d, H])
      Wv: [H, Hkv_tp*d]  (transposed from [Hkv_tp*d, H])

    Caller passes Wq.T.contiguous(), Wk.T.contiguous(), Wv.T.contiguous() from PyTorch.

Plan C optimization (inherited from v1c):
  Broadcast [PMAX,1] -> [PMAX,PMAX] via SBUF .ap() pattern instead of HBM round-trip.

Plan C optimization — Free-Dim RMSNorm (v7c):
  Replaces the transposed-domain RMSNorm (2x nc_transpose + nc_matmul + many tensor_copy
  per invocation = 17 ops) with a free-dimension RMSNorm using tensor_reduce(axis=1) that
  works directly in [S,d] layout (8 ops per invocation).

  Old pattern per RMSNorm invocation (17 ops):
    1. bf16 cast -> nc_transpose -> 2x tensor_copy to get [d,S] f32
    2. square -> bf16 cast -> nc_matmul(ones) -> tensor_copy for sum
    3. mean, var, rsqrt
    4. normalize + weight in [d,S]
    5. bf16 cast -> nc_transpose -> 2x tensor_copy back to [S,d] f32

  New pattern per RMSNorm invocation (8 ops):
    1. square in native [S,d] layout
    2. tensor_reduce(add, axis=1) for sum along free dim -> [PMAX,1]
    3. mean, var, rsqrt
    4. broadcast rinv [PMAX,1] -> [PMAX,PMAX] via SBUF .ap()
    5. normalize + apply weight in [S,d] layout

  Norm weight layout change:
    v6a: qnw_sb/knw_sb are [PMAX,PMAX] with nw[p,f]=weight[p] (partition=d, free=S)
    v7c: qnw_sd/knw_sd are [PMAX,PMAX] with nw[s,f]=weight[f] (partition=S, free=d)
    Created once at setup by nc_transposing the existing bf16 weights.

  Total ops saved per RMSNorm: 17 - 8 = 9 ops.
  Applied to K RMSNorm (Phase 1) and Q RMSNorm (Phase 2).
  At S=640: K invocations = 4*5=20, Q invocations = 4*2*5=40 => 60 * 9 = 540 ops saved.

v6a architecture (v5b + materialized softmax — Plan A softmax):
  Replaces online flash attention (per-ki rescaling with [PMAX,1]->[PMAX,PMAX] DMA broadcasts)
  with two-pass materialized softmax:
    Pass 1: Compute ALL score tiles for the row, apply causal mask, find global row maximum.
    Pass 2: Broadcast neg_max, compute exp(score - max), accumulate exp-weighted V, sum row.
    Normalize: divide accumulated V by row sum.
  Eliminates ALL online rescaling broadcasts (alpha_bc, nnmax_bc per ki) from the inner loop.
  Uses 4x-wide score computation (moving free dim = 512) for first BATCH=4 K tiles.
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
    nisa.dma_copy(dst=qnw_sb, src=qnw_1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0)) # Profiler: Long Idle Time

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
    # v7c: Create [S,d]-compatible norm weights for free-dim RMSNorm.
    # qnw_sb/knw_sb are [PMAX,PMAX] with nw[p,f]=weight[p] (partition=d, free=S).
    # We need nw[s,f]=weight[f] (partition=S, free=d) for element-wise multiply
    # in native [S,d] layout. nc_transpose swaps partition and free dims.
    # Done ONCE before the kvh loop — 4 extra ops amortized across all 60 invocations.
    # =========================================================================

    # Q norm weight: transpose to [S,d]-compatible layout
    # nc_transpose requires bf16 input, so cast first
    qnw_sb_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_sb_bf")
    nisa.tensor_copy(qnw_sb_bf, qnw_sb)  # f32 -> bf16 cast for nc_transpose input
    qnw_sd_p = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name="qnw_sd_p")
    nisa.nc_transpose(qnw_sd_p, qnw_sb_bf)  # transpose [d,S]-compat -> [S,d]-compat in PSUM
    qnw_sd_b = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_sd_b")
    nisa.tensor_copy(qnw_sd_b, qnw_sd_p)  # PSUM -> SBUF bf16
    qnw_sd = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="qnw_sd")
    nisa.tensor_copy(qnw_sd, qnw_sd_b)  # bf16 -> f32 for use in tensor_tensor multiply

    # K norm weight: transpose to [S,d]-compatible layout
    knw_sb_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_sb_bf")
    nisa.tensor_copy(knw_sb_bf, knw_sb)  # f32 -> bf16 cast for nc_transpose input
    knw_sd_p = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name="knw_sd_p")
    nisa.nc_transpose(knw_sd_p, knw_sb_bf)  # transpose [d,S]-compat -> [S,d]-compat in PSUM
    knw_sd_b = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_sd_b")
    nisa.tensor_copy(knw_sd_b, knw_sd_p)  # PSUM -> SBUF bf16
    knw_sd = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="knw_sd")
    nisa.tensor_copy(knw_sd, knw_sd_b)  # bf16 -> f32 for use in tensor_tensor multiply

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

            # =============================================================
            # v7c: K RMSNorm — free-dim version using tensor_reduce(axis=1).
            # Works directly in [S,d] layout, no transpose needed.
            # Replaces 17-op transposed-domain RMSNorm with 8-op free-dim version.
            # =============================================================

            # Step 1: Square in native [S, d] layout (f32)
            ksq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"p1_ksq_kv{kvh}_s{si}")
            nisa.tensor_tensor(ksq, kv, kv, op=nl.multiply)  # element-wise x^2

            # Step 2: Sum along free dim (d) using tensor_reduce -> [PMAX, 1]
            ksum1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_ksum1_kv{kvh}_s{si}")
            nisa.tensor_reduce(dst=ksum1, op=nl.add, data=ksq, axis=1)  # sum over d

            # Step 3: Mean = sum / d, then add eps, then rsqrt
            kmean1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"p1_kmean1_kv{kvh}_s{si}")
            nisa.tensor_scalar(kmean1, ksum1, op0=nl.multiply, operand0=1.0/d)  # mean per row

            kvar1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_kvar1_kv{kvh}_s{si}")
            nisa.tensor_scalar(kvar1, kmean1, op0=nl.add, operand0=EPS)  # add epsilon

            krinv1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"p1_krinv1_kv{kvh}_s{si}")
            nisa.activation(krinv1, op=nl.rsqrt, data=kvar1)  # 1/sqrt(var+eps) per row

            # Step 4: Broadcast rinv [PMAX,1] -> [PMAX,PMAX] via SBUF .ap() pattern
            krinv_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p1_krinv_bc_kv{kvh}_s{si}")
            nisa.dma_copy(dst=krinv_bc,
                          src=krinv1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

            # Step 5: Normalize x * rinv, then apply weight in [S,d] layout
            kn = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_kn_kv{kvh}_s{si}")
            nisa.tensor_tensor(kn, kv, krinv_bc, op=nl.multiply)  # normalized

            knw = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"p1_knw_kv{kvh}_s{si}")
            nisa.tensor_tensor(knw, kn, knw_sd, op=nl.multiply)  # weight-applied

            # Rename to kn for downstream compatibility (K RoPE reads kn)
            kn = knw

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

            # V: copy out of vp PSUM
            vv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_vv_kv{kvh}_s{si}")
            nisa.tensor_copy(vv, vp)
            vbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_vbf_kv{kvh}_s{si}")
            nisa.tensor_copy(vbf, vv)
            V_tiles.append(vbf)

        # -----------------------------------------------------------------
        # Phase 2: Materialized softmax (Plan A — no flash attention)
        # Computes full score row, applies standard softmax, then V accumulation.
        # Eliminates ALL online rescaling broadcasts from the ki inner loop.
        # -----------------------------------------------------------------
        BATCH = 4
        num_full_batches = num_s_tiles // BATCH
        tail_start = num_full_batches * BATCH

        for gi in nl.affine_range(gqa):
            qh = kvh * gqa + gi

            for qsi in nl.affine_range(num_s_tiles):
                qso = qsi * PMAX

                # === Q projection (identical to v5b) ===
                qp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                              name=f"p2_qp_kv{kvh}_g{gi}_s{qsi}")
                for ht in nl.affine_range(num_h_tiles):
                    hqTs = hidden_tiles[qsi][ht]
                    nisa.nc_matmul(qp, stationary=hqTs, moving=Wq_tiles_hoisted[qh][ht])

                qvec = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qvec_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvec, qp)

                # =============================================================
                # v7c: Q RMSNorm — free-dim version using tensor_reduce(axis=1).
                # Works directly in [S,d] layout, no transpose needed.
                # Replaces 17-op transposed-domain RMSNorm with 8-op free-dim version.
                # =============================================================

                # Step 1: Square in native [S, d] layout (f32)
                qsq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_qsq_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qsq, qvec, qvec, op=nl.multiply)  # element-wise x^2

                # Step 2: Sum along free dim (d) using tensor_reduce -> [PMAX, 1]
                qsum1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_qsum1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_reduce(dst=qsum1, op=nl.add, data=qsq, axis=1)  # sum over d

                # Step 3: Mean = sum / d, then add eps, then rsqrt
                qmean1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_qmean1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qmean1, qsum1, op0=nl.multiply, operand0=1.0/d)  # mean per row

                qvar1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_qvar1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qvar1, qmean1, op0=nl.add, operand0=EPS)  # add epsilon

                qrinv1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_qrinv1_kv{kvh}_g{gi}_s{qsi}")
                nisa.activation(qrinv1, op=nl.rsqrt, data=qvar1)  # 1/sqrt(var+eps) per row

                # Step 4: Broadcast rinv [PMAX,1] -> [PMAX,PMAX] via SBUF .ap() pattern
                qrinv_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_qrinv_bc_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=qrinv_bc,
                              src=qrinv1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                # Step 5: Normalize x * rinv, then apply weight in [S,d] layout
                qn = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"p2_qn_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qn, qvec, qrinv_bc, op=nl.multiply)  # normalized

                qnw = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_qnw_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qnw, qn, qnw_sd, op=nl.multiply)  # weight-applied

                # Rename to qn for downstream compatibility (Q RoPE reads qn)
                qn = qnw

                # === Q RoPE (identical to v5b) ===
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

                # =============================================================
                # PLAN A: Materialized softmax (no online rescaling)
                # Step 1: Compute ALL score tiles and apply causal mask
                # Step 2: Find global row maximum
                # Step 3: Broadcast neg_max, compute exp, accumulate sum + V
                # Step 4: Normalize and store
                # =============================================================

                # --- Step 1: Score computation ---
                # Batch: assemble first 4 K tiles into K_wide [PMAX, 4*PMAX]
                # Use one wide nc_matmul (moving free dim = 512)
                K_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                    name=f"p2_Kwide_kv{kvh}_g{gi}_s{qsi}")
                for i in range(BATCH):
                    nisa.tensor_copy(K_wide[0:PMAX, i*PMAX:(i+1)*PMAX], K_tiles[i])

                score_wide_psum = nl.zeros((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.psum,
                                           name=f"p2_scwp_kv{kvh}_g{gi}_s{qsi}")
                nisa.nc_matmul(score_wide_psum, stationary=qTs, moving=K_wide)
                score_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_scw_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(score_wide, score_wide_psum)

                masked_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                         name=f"p2_mw_kv{kvh}_g{gi}_s{qsi}")
                nisa.affine_select(
                    dst=masked_wide, on_true_tile=score_wide, on_false_value=-3.4e38,
                    pattern=[[-1, BATCH * PMAX]], offset=qso,
                    channel_multiplier=1, cmp_op=nl.greater_equal,
                )

                # Tail tiles (remaining after batch)
                masked_tail_tiles = []
                for ki_tail in range(tail_start, num_s_tiles):
                    kso_t = ki_tail * PMAX
                    scp_t = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                     name=f"p2_scpt_kv{kvh}_g{gi}_s{qsi}_t{ki_tail}")
                    nisa.nc_matmul(scp_t, stationary=qTs, moving=K_tiles[ki_tail])
                    scs_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_scst_kv{kvh}_g{gi}_s{qsi}_t{ki_tail}")
                    nisa.tensor_copy(scs_t, scp_t)
                    sm_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_smt_kv{kvh}_g{gi}_s{qsi}_t{ki_tail}")
                    nisa.affine_select(
                        dst=sm_t, on_true_tile=scs_t, on_false_value=-3.4e38,
                        pattern=[[-1, PMAX]], offset=qso - kso_t,
                        channel_multiplier=1, cmp_op=nl.greater_equal,
                    )
                    masked_tail_tiles.append(sm_t)

                # --- Step 2: Global row maximum ---
                row_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"p2_rowmax_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=masked_wide, axis=1)

                for ti in range(len(masked_tail_tiles)):
                    sm_t = masked_tail_tiles[ti]
                    tmax_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_tmaxt_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_reduce(dst=tmax_t, op=nl.maximum, data=sm_t, axis=1)
                    nmax_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_nmaxt_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_tensor(nmax_t, row_max, tmax_t, op=nl.maximum)
                    nisa.tensor_copy(row_max, nmax_t)

                # --- Step 3: Broadcast neg_max, compute exp, accumulate ---
                neg_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"p2_negmax_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(neg_max, row_max, op0=nl.multiply, operand0=-1.0)

                # Broadcast neg_max [PMAX,1] → [PMAX, 4*PMAX] for wide tile
                neg_max_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                          name=f"p2_nmw_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=neg_max_wide,
                              src=neg_max.ap(pattern=[[1, PMAX], [0, BATCH * PMAX]], offset=0)) #Profiler: long idle time

                # Wide: shift, exp, sum
                shifted_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                          name=f"p2_shw_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(shifted_wide, masked_wide, neg_max_wide, op=nl.add)
                sexp_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_sew_kv{kvh}_g{gi}_s{qsi}")
                nisa.activation(sexp_wide, op=nl.exp, data=shifted_wide)
                row_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"p2_rsum_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_reduce(dst=row_sum, op=nl.add, data=sexp_wide, axis=1)

                # Wide MM2: accumulate 4 V tiles into single PSUM
                atacc_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                      name=f"p2_atp_kv{kvh}_g{gi}_s{qsi}")
                for i in range(BATCH):
                    sexp_i_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                           name=f"p2_seibf_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_copy(sexp_i_bf, sexp_wide[0:PMAX, i*PMAX:(i+1)*PMAX])
                    sexpT_i = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                         name=f"p2_seTi_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.nc_transpose(sexpT_i, sexp_i_bf)
                    sexpTs_i = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                          name=f"p2_seTsi_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_copy(sexpTs_i, sexpT_i)
                    nisa.nc_matmul(atacc_psum, stationary=sexpTs_i, moving=V_tiles[i])

                # Tail: broadcast neg_max [PMAX,1] → [PMAX,PMAX] for narrow tiles
                neg_max_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_nmbc_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=neg_max_bc,
                              src=neg_max.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0)) # Profiler: long idle time

                for ti in range(len(masked_tail_tiles)):
                    sm_t = masked_tail_tiles[ti]
                    ki_abs = tail_start + ti
                    shifted_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                           name=f"p2_sht_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_tensor(shifted_t, sm_t, neg_max_bc, op=nl.add)
                    sexp_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_set_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.activation(sexp_t, op=nl.exp, data=shifted_t)

                    tsum_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_tsumt_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_reduce(dst=tsum_t, op=nl.add, data=sexp_t, axis=1)
                    nrs_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_nrst_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_tensor(nrs_t, row_sum, tsum_t, op=nl.add)
                    nisa.tensor_copy(row_sum, nrs_t)

                    # MM2 for tail tile
                    sexp_t_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                           name=f"p2_setbf_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_copy(sexp_t_bf, sexp_t)
                    sexpT_t = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                         name=f"p2_seTt_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.nc_transpose(sexpT_t, sexp_t_bf)
                    sexpTs_t = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                          name=f"p2_seTst_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_copy(sexpTs_t, sexpT_t)
                    nisa.nc_matmul(atacc_psum, stationary=sexpTs_t, moving=V_tiles[ki_abs])

                atacc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_atacc_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(atacc, atacc_psum)

                # --- Step 4: Normalize ---
                ssafe1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_ssafe1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(ssafe1, row_sum, op0=nl.add, operand0=1e-9)
                rsq1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_rsq1_kv{kvh}_g{gi}_s{qsi}")
                nisa.activation(rsq1, op=nl.rsqrt, data=ssafe1)
                inv1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_inv1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(inv1, rsq1, rsq1, op=nl.multiply)
                inv_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_inv_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=inv_bc,
                              src=inv1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                aout = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_aout_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(aout, atacc, inv_bc, op=nl.multiply)

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
        print(f"qwen3_attn_cte_fused v7c Test: B={B}, S={S}, H={H}, d={d}")
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
        sys.exit(1)
