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

v6a architecture (v5b + materialized softmax — Plan A softmax):
  Replaces online flash attention (per-ki rescaling with [PMAX,1]->[PMAX,PMAX] DMA broadcasts)
  with two-pass materialized softmax:
    Pass 1: Compute ALL score tiles for the row, apply causal mask, find global row maximum.
    Pass 2: Broadcast neg_max, compute exp(score - max), accumulate exp-weighted V, sum row.
    Normalize: divide accumulated V by row sum.
  Eliminates ALL online rescaling broadcasts (alpha_bc, nnmax_bc per ki) from the inner loop.
  Uses 4x-wide score computation (moving free dim = 512) for first BATCH=4 K tiles.

v7ab architecture (v7a score-softmax decoupling + v7b narrow broadcast softmax):
  Combines two independent optimizations:

  From v7a (Plan A — Score-Softmax Decoupling):
    Splits the single qsi loop into two separate passes within each (kvh, gi) iteration:
      Score pass (affine_range): For each qsi, compute Q projection + RMSNorm + RoPE +
        scale + transpose, then compute all score tiles against K_tiles and apply causal
        mask. Store all masked score tiles (wide + tail) in Python lists of SBUF tensors.
      Softmax+V pass (affine_range): For each qsi, read pre-computed masked scores, compute
        row-max -> broadcast neg_max -> shift+exp -> sum -> V accumulation -> normalize -> store.
    Key insight: In the Score pass, all TensorE matmuls (16 Q proj + score matmuls) are
    independent across qsi and can be aggressively pipelined by the compiler. Previously
    they were blocked behind the serial softmax chain within each qsi.
    In the Softmax+V pass, DMA broadcasts can overlap across independent qsi iterations.

  From v7b (Plan B — Narrow Broadcast Softmax):
    Eliminates the expensive wide DMA broadcast [PMAX,1]->[PMAX,4*PMAX] in the softmax path.
    Instead:
      Change 1: Single narrow broadcast of neg_max [PMAX,1]->[PMAX,PMAX] once per qsi.
        Replaces the wide [PMAX,1]->[PMAX,4*PMAX] broadcast that caused DMA stalls.
      Change 2: Process wide score tile as 4 separate [PMAX,PMAX] slices, each shifted by
        the shared neg_max_bc. Accumulates partial row sums across slices.
      Change 3: Reuse the same neg_max_bc for tail tiles — no second broadcast needed.
        Eliminates the redundant second [PMAX,1]->[PMAX,PMAX] broadcast entirely.

  Combined effect:
    Score pass: TensorE-heavy ops (Q proj + score matmuls) fully decoupled from softmax.
    Softmax+V pass: Only 1 narrow [PMAX,PMAX] broadcast per qsi (vs 2 broadcasts in v7a).
    Total DMA broadcast budget: 1 narrow [PMAX,PMAX] per qsi (down from 1 wide + 1 narrow).

  SBUF budget: 5 wide [PMAX, 4*PMAX] f32 + 5x1 tail [PMAX, PMAX] f32 = 1.6MB additional.
  Total ~10MB of 24MB, fits within budget.

v8c architecture (v7ab + Three-Sub-Pass Softmax — Plan C broadcast pipelining):
  Problem: v7ab still has two DMA broadcast stalls per qsi iteration in Pass 2:
    - neg_max broadcast [PMAX,1]->[PMAX,PMAX] at line 692 (before exp())
    - inv broadcast [PMAX,1]->[PMAX,PMAX] at line 801 (before normalization)
  These create a DMA->compute->DMA->compute serial chain within each qsi, preventing
  the DMA engine from pipelining broadcasts across qsi iterations.

  Solution: Split Pass 2 into three independent sub-passes (each using affine_range):

    Sub-pass 2a — Row-max + Broadcast neg_max:
      For each qsi (affine_range — iterations independent):
        1. Compute row_max from masked_wide and masked_tail
        2. Compute neg_max = -row_max
        3. Broadcast neg_max [PMAX,1] -> [PMAX,PMAX] via .ap()
        4. Store neg_max_bc in all_neg_max_bc[qsi]
      With affine_range, the DMA engine pipelines broadcasts across qsi iterations.

    Sub-pass 2b — Exp + row_sum + V accumulation:
      For each qsi (affine_range — iterations independent):
        1. Read pre-computed neg_max_bc from all_neg_max_bc[qsi] — NO broadcast here
        2. Process wide + tail slices: shift, exp, partial row_sum, V matmul
        3. Copy atacc PSUM->SBUF, compute inv1 = rsqrt^2(row_sum + eps)
        4. Store all_atacc[qsi] and all_inv1[qsi]
      Pure compute sub-pass (VectorE + TensorE) without any DMA broadcast stalls.

    Sub-pass 2c — Normalize + Store:
      For each qsi (affine_range — iterations independent):
        1. Broadcast inv1[qsi] [PMAX,1] -> [PMAX,PMAX] via .ap()
        2. Multiply atacc[qsi] * inv_bc
        3. Cast to bf16 and DMA store to output HBM
      Again, affine_range pipelines inv broadcasts across qsi iterations.

  Key insight: By batching all DMA broadcasts into their own sub-passes with affine_range,
  the DMA engine can prefetch and pipeline broadcasts across qsi iterations. The heavy
  compute sub-pass (2b) runs without any DMA broadcast stalls.

  SBUF budget for additional pre-stored tensors (per gi iteration):
    all_neg_max_bc: num_s_tiles * [PMAX,PMAX] f32 = 5 * 64KB = 320KB
    all_atacc:      num_s_tiles * [PMAX,PMAX] f32 = 5 * 64KB = 320KB
    all_inv1:       num_s_tiles * [PMAX,1]    f32 = 5 * 512B  ≈ 2.5KB
    Total additional: ~642KB — fits within 24MB SBUF budget.
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
      V_cache: [B, S, Hkv_tp*d] bf16 — projected V values (no norm/RoPE) for KV cache update

    v8c Three-Sub-Pass Softmax architecture:
      Splits Pass 2 (Softmax+V) of v7ab into three affine_range sub-passes:
        2a: row_max + broadcast neg_max_bc (DMA-heavy, pipelined via affine_range)
        2b: exp + V accumulation (compute-heavy, no DMA broadcasts)
        2c: broadcast inv_bc + normalize + store (DMA-heavy, pipelined via affine_range)
      Eliminates serial DMA->compute->DMA->compute chain within each qsi by batching
      all DMA broadcasts into dedicated sub-passes that can pipeline across qsi.
    """
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
            # DMA store is scheduled within affine_range si loop — overlaps with Phase 2 compute.
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
            # vbf is [PMAX, PMAX] bf16 in [S_tile, d] layout — exact layout for KV cache.
            # DMA store overlaps with Phase 2 attention compute via affine_range scheduling.
            nisa.dma_copy(dst=V_cache_2d[s_off:s_off + PMAX, kvh * d:kvh * d + PMAX], src=vbf)

        # -----------------------------------------------------------------
        # Phase 2: Materialized softmax (v8c — Three-Sub-Pass Softmax)
        # Extends v7ab with sub-pass decomposition to pipeline DMA broadcasts.
        # -----------------------------------------------------------------
        # Clamp BATCH to available K tiles so small-S buckets (S<512) don't go out of bounds.
        # For S=128 → num_s_tiles=1, BATCH=1; for S>=512 → BATCH=4 (original behaviour).
        BATCH = min(4, num_s_tiles)
        num_full_batches = num_s_tiles // BATCH
        tail_start = num_full_batches * BATCH

        for gi in nl.affine_range(gqa):
            qh = kvh * gqa + gi

            # =============================================================
            # v7a PASS 1: Pre-compute all score tiles (TensorE heavy).
            # All Q projection matmuls and score matmuls across qsi are
            # independent, allowing the compiler to pipeline TensorE ops
            # aggressively without being blocked by softmax's serial chain.
            # =============================================================
            all_masked_wide = []   # all_masked_wide[qsi] = [PMAX, 4*PMAX] f32 SBUF
            all_masked_tail = []   # all_masked_tail[qsi] = list of [PMAX, PMAX] f32 SBUF

            for qsi in nl.affine_range(num_s_tiles):
                qso = qsi * PMAX

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
                # Score computation: wide batch (first BATCH=4 K tiles)
                # Assembles 4 pre-transposed K tiles [d, S_k] into K_wide [d, 4*S_k],
                # then one wide matmul computes all 4 score columns simultaneously.
                # =============================================================
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

                # Apply causal mask to wide scores: positions where q_idx >= k_idx pass through,
                # otherwise filled with -3.4e38 (effectively -inf for softmax).
                masked_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                         name=f"p2_mw_kv{kvh}_g{gi}_s{qsi}")
                nisa.affine_select(
                    dst=masked_wide, on_true_tile=score_wide, on_false_value=-3.4e38,
                    pattern=[[-1, BATCH * PMAX]], offset=qso,
                    channel_multiplier=1, cmp_op=nl.greater_equal,
                )

                # Score computation: tail tiles (remaining K tiles after the batch)
                masked_tail_tiles = []
                for ki_tail in range(tail_start, num_s_tiles):
                    kso_t = ki_tail * PMAX
                    scp_t = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                     name=f"p2_scpt_kv{kvh}_g{gi}_s{qsi}_t{ki_tail}")
                    nisa.nc_matmul(scp_t, stationary=qTs, moving=K_tiles[ki_tail])
                    scs_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_scst_kv{kvh}_g{gi}_s{qsi}_t{ki_tail}")
                    nisa.tensor_copy(scs_t, scp_t)
                    # Apply causal mask per tail tile, offset adjusted for this K tile's position
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
            # v8c PASS 2 — Three sub-passes to pipeline DMA broadcasts.
            #
            # v7ab had a serial chain per qsi:
            #   row_max -> DMA broadcast neg_max -> exp -> V matmul -> inv -> DMA broadcast inv -> normalize
            # This prevents DMA pipelining because each DMA is blocked by preceding compute.
            #
            # v8c breaks this chain into three independent affine_range loops:
            #   Sub-pass 2a: All row_max + neg_max broadcasts — DMA broadcasts pipelined across qsi
            #   Sub-pass 2b: All exp + V accum + inv compute — no DMA broadcasts, pure compute
            #   Sub-pass 2c: All inv broadcasts + normalize + store — DMA broadcasts pipelined across qsi
            # =============================================================

            # Python lists to carry intermediate results between sub-passes.
            # These are SBUF tensor handles — allocated inside the loops below.
            all_neg_max_bc = []  # all_neg_max_bc[qsi] = [PMAX,PMAX] f32 — broadcast neg_max from 2a
            all_atacc = []       # all_atacc[qsi]      = [PMAX,PMAX] f32 — accumulated attn output from 2b
            all_inv1 = []        # all_inv1[qsi]       = [PMAX,1]    f32 — 1/row_sum scalar from 2b

            # -------------------------------------------------------------
            # Sub-pass 2a: Row-max + Broadcast neg_max
            # All qsi iterations are independent (each reads its own masked scores
            # and writes its own neg_max_bc). affine_range lets the DMA engine
            # pipeline the [PMAX,1]->[PMAX,PMAX] broadcasts across qsi iterations.
            # -------------------------------------------------------------
            for qsi in nl.affine_range(num_s_tiles):
                # Retrieve pre-computed masked score tiles for this qsi (from Pass 1)
                masked_wide = all_masked_wide[qsi]
                masked_tail_tiles = all_masked_tail[qsi]

                # --- Compute global row maximum across all score tiles for this qsi ---
                # Initialize with maximum from the wide tile (covers BATCH K tiles at once)
                row_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"2a_rowmax_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=masked_wide, axis=1)

                # Update row_max with maximums from each tail tile
                for ti in range(len(masked_tail_tiles)):
                    sm_t = masked_tail_tiles[ti]
                    tmax_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"2a_tmaxt_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_reduce(dst=tmax_t, op=nl.maximum, data=sm_t, axis=1)
                    nmax_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"2a_nmaxt_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_tensor(nmax_t, row_max, tmax_t, op=nl.maximum)
                    nisa.tensor_copy(row_max, nmax_t)  # update running max in-place

                # Compute neg_max = -row_max for the shift-before-exp trick
                neg_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"2a_negmax_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(neg_max, row_max, op0=nl.multiply, operand0=-1.0)

                # Broadcast neg_max [PMAX,1] -> [PMAX,PMAX] via SBUF .ap() pattern.
                # v8c: This broadcast is now in its own affine_range sub-pass so the
                # DMA engine can prefetch/pipeline across qsi iterations (no compute blocking).
                neg_max_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"2a_nmbc_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=neg_max_bc,
                              src=neg_max.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                # Save for consumption in sub-pass 2b
                all_neg_max_bc.append(neg_max_bc)

            # -------------------------------------------------------------
            # Sub-pass 2b: Exp + row_sum + V accumulation
            # Reads neg_max_bc from all_neg_max_bc[qsi] — NO DMA broadcast in this loop.
            # Pure compute sub-pass: VectorE (exp/adds) + TensorE (V matmuls).
            # affine_range enables compiler to pipeline compute across qsi.
            # -------------------------------------------------------------
            for qsi in nl.affine_range(num_s_tiles):
                # Retrieve pre-computed masked scores and neg_max_bc (no DMA here)
                masked_wide = all_masked_wide[qsi]
                masked_tail_tiles = all_masked_tail[qsi]
                neg_max_bc = all_neg_max_bc[qsi]  # from sub-pass 2a — SBUF read, no broadcast

                # Accumulate exp(score) @ V into PSUM; will be copied to SBUF after all tiles
                atacc_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                      name=f"2b_atp_kv{kvh}_g{gi}_s{qsi}")

                # Initialize row_sum; first slice sets it, subsequent slices add to it
                row_sum = None

                # Process wide score tile as BATCH separate [PMAX,PMAX] slices.
                # Each slice is shifted by shared neg_max_bc then exp'd.
                # Partial row sums are accumulated across all BATCH slices.
                for i in range(BATCH):
                    # Extract the i-th [PMAX,PMAX] slice from the [PMAX, BATCH*PMAX] wide tile
                    slice_i = masked_wide[0:PMAX, i*PMAX:(i+1)*PMAX]

                    # Shift: score_slice - max = score_slice + neg_max_bc
                    shifted_i = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                           name=f"2b_shi_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_tensor(shifted_i, slice_i, neg_max_bc, op=nl.add)

                    # Compute exp(score - max) for this slice (numerically stable)
                    sexp_i = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"2b_sei_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.activation(sexp_i, op=nl.exp, data=shifted_i)

                    # Accumulate partial row sum: sum of exp values along the K dimension
                    partial_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                             name=f"2b_psi_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_reduce(dst=partial_sum, op=nl.add, data=sexp_i, axis=1)

                    if i == 0:
                        row_sum = partial_sum   # first slice: initialize row_sum
                    else:
                        # Subsequent slices: add partial sum into running row_sum
                        new_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                             name=f"2b_nsi_kv{kvh}_g{gi}_s{qsi}_i{i}")
                        nisa.tensor_tensor(new_sum, row_sum, partial_sum, op=nl.add)
                        row_sum = new_sum

                    # MM2: exp(score)^T @ V -> accumulate into atacc_psum
                    # Transpose exp scores [S_q, S_k] -> [S_k, S_q] for matmul stationary position
                    sexp_i_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                           name=f"2b_seibf_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_copy(sexp_i_bf, sexp_i)
                    sexpT_i = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                         name=f"2b_seTi_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.nc_transpose(sexpT_i, sexp_i_bf)
                    sexpTs_i = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                          name=f"2b_seTsi_kv{kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_copy(sexpTs_i, sexpT_i)
                    # Accumulate: exp(score) @ V into attention output psum
                    nisa.nc_matmul(atacc_psum, stationary=sexpTs_i, moving=V_tiles[i])

                # v7b Change 3 (preserved): Reuse neg_max_bc for tail tiles — no extra broadcast.
                # The same [PMAX,PMAX] neg_max_bc covers all tail tiles (same global max value).
                for ti in range(len(masked_tail_tiles)):
                    sm_t = masked_tail_tiles[ti]
                    ki_abs = tail_start + ti

                    # Shift tail score by neg_max (reuse neg_max_bc from sub-pass 2a)
                    shifted_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                           name=f"2b_sht_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_tensor(shifted_t, sm_t, neg_max_bc, op=nl.add)
                    sexp_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"2b_set_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.activation(sexp_t, op=nl.exp, data=shifted_t)

                    # Accumulate tail tile's exp sum into row_sum
                    tsum_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"2b_tsumt_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_reduce(dst=tsum_t, op=nl.add, data=sexp_t, axis=1)
                    nrs_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"2b_nrst_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_tensor(nrs_t, row_sum, tsum_t, op=nl.add)
                    row_sum = nrs_t

                    # MM2 for tail tile: transpose exp'd scores, matmul with corresponding V
                    sexp_t_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                           name=f"2b_setbf_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_copy(sexp_t_bf, sexp_t)
                    sexpT_t = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                         name=f"2b_seTt_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.nc_transpose(sexpT_t, sexp_t_bf)
                    sexpTs_t = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                          name=f"2b_seTst_kv{kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_copy(sexpTs_t, sexpT_t)
                    nisa.nc_matmul(atacc_psum, stationary=sexpTs_t, moving=V_tiles[ki_abs])

                # Copy accumulated attention output from PSUM to SBUF for sub-pass 2c
                atacc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"2b_atacc_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(atacc, atacc_psum)

                # Compute inv1 = 1/row_sum via rsqrt(row_sum + eps)^2
                # Using rsqrt trick: rsqrt(x)^2 = 1/x, eps guards against division by zero
                ssafe1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"2b_ssafe1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(ssafe1, row_sum, op0=nl.add, operand0=1e-9)
                rsq1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"2b_rsq1_kv{kvh}_g{gi}_s{qsi}")
                nisa.activation(rsq1, op=nl.rsqrt, data=ssafe1)
                inv1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"2b_inv1_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(inv1, rsq1, rsq1, op=nl.multiply)  # rsqrt^2 = 1/x

                # Save both atacc and inv1 for sub-pass 2c consumption
                all_atacc.append(atacc)
                all_inv1.append(inv1)

            # -------------------------------------------------------------
            # Sub-pass 2c: Normalize + Store
            # All qsi are independent (each reads its own atacc + inv1 and stores to distinct
            # output positions). affine_range pipelines the inv broadcasts across qsi.
            # -------------------------------------------------------------
            for qsi in nl.affine_range(num_s_tiles):
                qso = qsi * PMAX  # row offset in [B*S, Hq_out] output for this qsi tile

                # Retrieve pre-computed atacc and inv1 from sub-pass 2b
                atacc = all_atacc[qsi]
                inv1 = all_inv1[qsi]

                # Broadcast inv1 [PMAX,1] -> [PMAX,PMAX] via SBUF .ap() pattern.
                # v8c: Isolated in its own affine_range sub-pass so the DMA engine can
                # pipeline inv broadcasts across qsi iterations without compute blocking.
                inv_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"2c_invbc_kv{kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=inv_bc,
                              src=inv1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                # Normalize: multiply accumulated V by 1/row_sum (element-wise)
                aout = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"2c_aout_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(aout, atacc, inv_bc, op=nl.multiply)

                # Cast to bf16 (output dtype) and DMA store to HBM output buffer
                aoutb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"2c_aoutb_kv{kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(aoutb, aout)
                # Store to the appropriate [qso:qso+PMAX, qh*d:qh*d+PMAX] slice of output
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
        print(f"qwen3_attn_cte_fused v8c Test: B={B}, S={S}, H={H}, d={d}")
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

        print("\n[3/4] Running NKI kernel (LNC=2)...")
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
