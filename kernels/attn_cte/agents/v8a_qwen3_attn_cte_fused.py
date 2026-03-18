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
        Eliminates the redundant second [PMAX,PMAX] broadcast entirely.

  Combined effect:
    Score pass: TensorE-heavy ops (Q proj + score matmuls) fully decoupled from softmax.
    Softmax+V pass: Only 1 narrow [PMAX,PMAX] broadcast per qsi (vs 2 broadcasts in v7a).
    Total DMA broadcast budget: 1 narrow [PMAX,PMAX] per qsi (down from 1 wide + 1 narrow).

  SBUF budget: 5 wide [PMAX, 4*PMAX] f32 + 5x1 tail [PMAX, PMAX] f32 = 1.6MB additional.
  Total ~10MB of 24MB, fits within budget.

v8a architecture (v7ab + LNC=2 KVH work splitting):
  Splits the kvh loop across 2 NeuronCores via LNC=2 launch syntax.
  Each core processes exactly Hkv_tp/2 KV heads, starting at pid * (Hkv_tp/2).

  API:
    pid = nl.program_id(0)        # 0 or 1 — which core this instance is
    num_progs = nl.num_programs() # 2 when launched with kernel[2](...)

  Work distribution:
    - kvh range: [pid*kvh_per_core .. (pid+1)*kvh_per_core)
    - Wq tiles: only the qh indices belonging to this core's kvh range
      qh = kvh*gqa + gi, so each core loads gqa*kvh_per_core Q heads total
    - Wk/Wv tiles: loaded per-kvh, naturally partitioned
    - hidden tiles, cos/sin, norm weights: BOTH cores load independently
      (shared HBM reads, no conflict)
    - Output writes: each core writes output_2d[qso, qh*d:(qh+1)*d]
      where qh belongs exclusively to this core — non-overlapping
    - K_cache/V_cache writes: each core writes kvh partition — non-overlapping

  Throughput:
    Two cores each do half the kvh work in parallel -> ~2x throughput.
    No synchronization required: all outputs are non-overlapping HBM writes.
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

    v8a: Launched with kernel[2](...). Each NeuronCore handles Hkv_tp/2 KV heads.
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

    # v8a: Determine this core's identity and KV head assignment.
    # nl.program_id(0) returns 0 or 1 when launched with kernel[2](...).
    # nl.num_programs() returns 2, matching the launch grid size.
    pid = nl.program_id(0)           # which NeuronCore this instance runs on (0 or 1)
    num_progs = nl.num_programs()    # total cores = 2

    # Each core owns exactly kvh_per_core consecutive KV heads.
    # kvh_start is a compile-time Python int (pid and num_progs resolve at trace time).
    kvh_per_core = Hkv_tp // num_progs   # e.g. 4 // 2 = 2 heads per core
    kvh_start = pid * kvh_per_core       # core 0: 0, core 1: 2

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
    # v8a: Both cores load cos/sin independently — they're shared HBM reads with no conflict.
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
    #
    # v8a: Only hoist the Wq tiles for THIS core's qh range.
    # Core 0 handles kvh in [kvh_start, kvh_start+kvh_per_core), so it owns Q heads
    # qh in [kvh_start*gqa, (kvh_start+kvh_per_core)*gqa).
    # We build Wq_tiles_hoisted as a 2D list indexed [local_qh][ht] where
    # local_qh = qh - kvh_start*gqa in [0, gqa*kvh_per_core).
    # The global qh used for weight column offsets is (kvh_start*gqa + local_qh).
    qh_start = kvh_start * gqa            # first absolute Q-head index owned by this core
    num_qh_local = kvh_per_core * gqa     # how many Q heads this core owns

    Wq_tiles_hoisted = []  # Wq_tiles_hoisted[local_qh][ht] = [PMAX, PMAX] bf16 SBUF tile
    for local_qh in nl.affine_range(num_qh_local):
        global_qh = qh_start + local_qh   # absolute Q-head index in [0, Hq_tp)
        row = []
        for ht in nl.affine_range(num_h_tiles):
            ho = ht * PMAX
            # Plan B: load [H_tile, d] directly — Wq[ho:ho+PMAX, global_qh*d:global_qh*d+PMAX]
            # No nc_transpose or tensor_copy needed (was 3 ops, now 1 op per tile).
            wqTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"hoist_wqTs_h{local_qh}_t{ht}")
            nisa.dma_copy(dst=wqTs, src=Wq[ho:ho+PMAX, global_qh*d:global_qh*d+PMAX])
            row.append(wqTs)
        Wq_tiles_hoisted.append(row)

    # =========================================================================
    # v5a Change 1: Hidden tile preload.
    # Preload all num_s_tiles * num_h_tiles hidden tiles into SBUF as
    # pre-transposed [PMAX, PMAX] bf16 tensors before the kvh loop.
    # Eliminates redundant HBM loads of hidden tiles across all kvh iterations.
    # Both loops use affine_range for independent iterations -> DMA-compute overlap.
    # v8a: Both cores load ALL hidden tiles independently — shared HBM reads, no conflict.
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
    # Outer loop uses affine_range to keep kvh_per_core iterations independent.
    #
    # v8a: Loop only over this core's kvh partition [kvh_start, kvh_start+kvh_per_core).
    # The absolute KV head index is: kvh = kvh_start + local_kvh.
    # =========================================================================
    for local_kvh in nl.affine_range(kvh_per_core):
        # Compute absolute KV head index for HBM addressing.
        # kvh_start is a Python int resolved at trace time; local_kvh is the loop var.
        kvh = kvh_start + local_kvh   # absolute KV head index in [0, Hkv_tp)

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
                               name=f"hoist_wkTs_kv{local_kvh}_t{ht}")
            nisa.dma_copy(dst=wkTs, src=Wk[ho:ho+PMAX, kvh*d:kvh*d+PMAX])
            Wk_tiles_hoisted.append(wkTs)

            # V weight tile — Plan B: load [H_tile, d] directly from [H, Hkv_tp*d] layout
            wvTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"hoist_wvTs_kv{local_kvh}_t{ht}")
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
                          name=f"p1_kp_kv{local_kvh}_s{si}")
            vp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                          name=f"p1_vp_kv{local_kvh}_s{si}")

            for ht in nl.affine_range(num_h_tiles):
                hTs = hidden_tiles[si][ht]  # from SBUF — no HBM DMA
                # v3a: Use hoisted Wk/Wv tiles — no HBM DMA per si iteration.
                nisa.nc_matmul(kp, stationary=hTs, moving=Wk_tiles_hoisted[ht])
                nisa.nc_matmul(vp, stationary=hTs, moving=Wv_tiles_hoisted[ht])

            kv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_kv_kv{local_kvh}_s{si}")
            nisa.tensor_copy(kv, kp)

            # K RMSNorm in transposed [d,S] layout
            kvb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_kvb_kv{local_kvh}_s{si}")
            nisa.tensor_copy(kvb, kv)
            kvTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                              name=f"p1_kvTp_kv{local_kvh}_s{si}")
            nisa.nc_transpose(kvTp, kvb)
            kvTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_kvTb_kv{local_kvh}_s{si}")
            nisa.tensor_copy(kvTb, kvTp)
            kvT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"p1_kvT_kv{local_kvh}_s{si}")
            nisa.tensor_copy(kvT, kvTb)

            ksq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"p1_ksq_kv{local_kvh}_s{si}")
            nisa.tensor_tensor(ksq, kvT, kvT, op=nl.multiply)
            ksqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_ksqb_kv{local_kvh}_s{si}")
            nisa.tensor_copy(ksqb, ksq)
            ksump = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                             name=f"p1_ksump_kv{local_kvh}_s{si}")
            nisa.nc_matmul(ksump, stationary=rms_ones, moving=ksqb)
            ksum = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_ksum_kv{local_kvh}_s{si}")
            nisa.tensor_copy(ksum, ksump)

            kmean = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_kmean_kv{local_kvh}_s{si}")
            nisa.tensor_scalar(kmean, ksum, op0=nl.multiply, operand0=1.0/d)
            kvar = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_kvar_kv{local_kvh}_s{si}")
            nisa.tensor_scalar(kvar, kmean, op0=nl.add, operand0=EPS)
            krinv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_krinv_kv{local_kvh}_s{si}")
            nisa.activation(krinv, op=nl.rsqrt, data=kvar)

            knT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"p1_knT_kv{local_kvh}_s{si}")
            nisa.tensor_tensor(knT, kvT, krinv, op=nl.multiply)
            # Apply norm weight (both [PMAX, PMAX] now)
            knwT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_knwT_kv{local_kvh}_s{si}")
            nisa.tensor_tensor(knwT, knT, knw_sb, op=nl.multiply)

            # Transpose back to [S,d]
            knwTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                               name=f"p1_knwTb_kv{local_kvh}_s{si}")
            nisa.tensor_copy(knwTb, knwT)
            knp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                             name=f"p1_knp_kv{local_kvh}_s{si}")
            nisa.nc_transpose(knp, knwTb)
            knb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_knb_kv{local_kvh}_s{si}")
            nisa.tensor_copy(knb, knp)
            kn = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_kn_kv{local_kvh}_s{si}")
            nisa.tensor_copy(kn, knb)

            # K RoPE
            rotk = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_rotk_kv{local_kvh}_s{si}")
            negku = nl.ndarray((PMAX, half_d), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_negku_kv{local_kvh}_s{si}")
            nisa.tensor_scalar(negku, kn[0:PMAX, half_d:d], op0=nl.multiply, operand0=-1.0)
            nisa.tensor_copy(rotk[0:PMAX, 0:half_d], negku)
            nisa.tensor_copy(rotk[0:PMAX, half_d:d], kn[0:PMAX, 0:half_d])

            kcos = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"p1_kcos_kv{local_kvh}_s{si}")
            nisa.tensor_tensor(kcos, kn, cos_f, op=nl.multiply)
            ksinp = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_ksinp_kv{local_kvh}_s{si}")
            nisa.tensor_tensor(ksinp, rotk, sin_f, op=nl.multiply)
            krope = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"p1_krope_kv{local_kvh}_s{si}")
            nisa.tensor_tensor(krope, kcos, ksinp, op=nl.add)

            # v4a Change 1: Store K tile pre-transposed [d, S_k] in SBUF.
            krbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_krbf_kv{local_kvh}_s{si}")
            nisa.tensor_copy(krbf, krope)
            krT_psum = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                  name=f"p1_krTp_kv{local_kvh}_s{si}")
            nisa.nc_transpose(krT_psum, krbf)
            krTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                              name=f"p1_krTs_kv{local_kvh}_s{si}")
            nisa.tensor_copy(krTs, krT_psum)
            K_tiles.append(krTs)  # [d=128, S_k=128] — ready for MM1 moving position

            # Store post-RoPE, post-norm K tile to KV cache.
            # krbf is [PMAX, PMAX] bf16 in [S_tile, d] layout (non-transposed, unlike krTs).
            # v8a: Each core writes its own kvh partition — non-overlapping HBM writes.
            # kvh is the absolute KV head index, so kvh*d gives the correct column offset.
            nisa.dma_copy(dst=K_cache_2d[s_off:s_off + PMAX, kvh * d:kvh * d + PMAX], src=krbf)

            # V: copy out of vp PSUM
            vv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                            name=f"p1_vv_kv{local_kvh}_s{si}")
            nisa.tensor_copy(vv, vp)
            vbf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"p1_vbf_kv{local_kvh}_s{si}")
            nisa.tensor_copy(vbf, vv)
            V_tiles.append(vbf)

            # Store V tile to KV cache (projected, no norm/RoPE).
            # vbf is [PMAX, PMAX] bf16 in [S_tile, d] layout — exact layout for KV cache.
            # v8a: Each core writes its own kvh partition — non-overlapping HBM writes.
            nisa.dma_copy(dst=V_cache_2d[s_off:s_off + PMAX, kvh * d:kvh * d + PMAX], src=vbf)

        # -----------------------------------------------------------------
        # Phase 2: Materialized softmax (v7ab — combined Plan A + Plan B)
        # v7a: Score-softmax decoupling — split into two passes.
        # v7b: Narrow broadcast + 4 slices — applied within the softmax pass.
        # Pass 1 pre-computes all masked score tiles per qsi.
        # Pass 2 reads pre-computed scores and does softmax + V accumulation
        # using v7b's narrow broadcast approach.
        # -----------------------------------------------------------------
        # Clamp BATCH to available K tiles so small-S buckets (S<512) don't go out of bounds.
        # For S=128 → num_s_tiles=1, BATCH=1; for S>=512 → BATCH=4 (original behaviour).
        BATCH = min(4, num_s_tiles)
        num_full_batches = num_s_tiles // BATCH
        tail_start = num_full_batches * BATCH

        for gi in nl.affine_range(gqa):
            # v8a: Compute global qh from absolute kvh index (not local_kvh).
            # gi iterates [0, gqa) for each KV head; qh is the absolute Q head.
            qh = kvh * gqa + gi

            # v8a: Wq lookup uses local index: local_qh = (kvh - kvh_start)*gqa + gi.
            # local_qh maps into Wq_tiles_hoisted which was built with [local_qh][ht] indexing.
            local_qh = local_kvh * gqa + gi   # index into Wq_tiles_hoisted

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

                # === Q projection (identical to v6a) ===
                qp = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                              name=f"p2_qp_kv{local_kvh}_g{gi}_s{qsi}")
                for ht in nl.affine_range(num_h_tiles):
                    hqTs = hidden_tiles[qsi][ht]
                    # v8a: Use local_qh to index Wq_tiles_hoisted (built per-core).
                    nisa.nc_matmul(qp, stationary=hqTs, moving=Wq_tiles_hoisted[local_qh][ht])

                qvec = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qvec_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvec, qp)

                # === Q RMSNorm (identical to v6a) ===
                qvb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"p2_qvb_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvb, qvec)
                qvTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                  name=f"p2_qvTp_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.nc_transpose(qvTp, qvb)
                qvTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"p2_qvTb_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvTb, qvTp)
                qvT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_qvT_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qvT, qvTb)

                qsq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_qsq_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qsq, qvT, qvT, op=nl.multiply)
                qsqb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"p2_qsqb_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qsqb, qsq)
                qsump = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                 name=f"p2_qsump_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.nc_matmul(qsump, stationary=rms_ones, moving=qsqb)
                qsum = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qsum_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qsum, qsump)

                qmean = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_qmean_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qmean, qsum, op0=nl.multiply, operand0=1.0/d)
                qvar = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qvar_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qvar, qmean, op0=nl.add, operand0=EPS)
                qrinv = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_qrinv_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.activation(qrinv, op=nl.rsqrt, data=qvar)

                qnT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_qnT_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qnT, qvT, qrinv, op=nl.multiply)
                qnwT = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qnwT_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qnwT, qnT, qnw_sb, op=nl.multiply)

                qnwTb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"p2_qnwTb_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qnwTb, qnwT)
                qnp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                 name=f"p2_qnp_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.nc_transpose(qnp, qnwTb)
                qnb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"p2_qnb_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qnb, qnp)
                qn = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"p2_qn_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qn, qnb)

                # === Q RoPE (identical to v6a) ===
                cosqf = q_cos_tiles[qsi]
                sinqf = q_sin_tiles[qsi]

                rotq = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_rotq_kv{local_kvh}_g{gi}_s{qsi}")
                negqu = nl.ndarray((PMAX, half_d), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_negqu_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(negqu, qn[0:PMAX, half_d:d], op0=nl.multiply, operand0=-1.0)
                nisa.tensor_copy(rotq[0:PMAX, 0:half_d], negqu)
                nisa.tensor_copy(rotq[0:PMAX, half_d:d], qn[0:PMAX, 0:half_d])

                qcos = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_qcos_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qcos, qn, cosqf, op=nl.multiply)
                qsinp = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_qsinp_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qsinp, rotq, sinqf, op=nl.multiply)
                qrope = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_qrope_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(qrope, qcos, qsinp, op=nl.add)

                # Scale Q by 1/sqrt(d) for attention score computation
                qsc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"p2_qsc_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(qsc, qrope, op0=nl.multiply, operand0=scale)

                # Transpose Q to [d, S_q] for matmul with K (stationary position needs [d, S_q])
                qscb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  name=f"p2_qscb_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qscb, qsc)
                qTp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                 name=f"p2_qTp_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.nc_transpose(qTp, qscb)
                qTs = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"p2_qTs_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(qTs, qTp)

                # =============================================================
                # Score computation: wide batch (first BATCH=4 K tiles)
                # Assembles 4 pre-transposed K tiles [d, S_k] into K_wide [d, 4*S_k],
                # then one wide matmul computes all 4 score columns simultaneously.
                # =============================================================
                K_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                    name=f"p2_Kwide_kv{local_kvh}_g{gi}_s{qsi}")
                for i in range(BATCH):
                    nisa.tensor_copy(K_wide[0:PMAX, i*PMAX:(i+1)*PMAX], K_tiles[i])

                score_wide_psum = nl.zeros((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.psum,
                                           name=f"p2_scwp_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.nc_matmul(score_wide_psum, stationary=qTs, moving=K_wide)
                score_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_scw_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(score_wide, score_wide_psum)

                # Apply causal mask to wide scores: positions where q_idx >= k_idx pass through,
                # otherwise filled with -3.4e38 (effectively -inf for softmax).
                masked_wide = nl.ndarray((PMAX, BATCH * PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                         name=f"p2_mw_kv{local_kvh}_g{gi}_s{qsi}")
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
                                     name=f"p2_scpt_kv{local_kvh}_g{gi}_s{qsi}_t{ki_tail}")
                    nisa.nc_matmul(scp_t, stationary=qTs, moving=K_tiles[ki_tail])
                    scs_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_scst_kv{local_kvh}_g{gi}_s{qsi}_t{ki_tail}")
                    nisa.tensor_copy(scs_t, scp_t)
                    # Apply causal mask per tail tile, offset adjusted for this K tile's position
                    sm_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"p2_smt_kv{local_kvh}_g{gi}_s{qsi}_t{ki_tail}")
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
            # Reads pre-computed masked scores from Pass 1. Each qsi's
            # softmax is independent, so DMA broadcasts can overlap across
            # qsi iterations via affine_range scheduling.
            #
            # v7b softmax: Uses narrow [PMAX,PMAX] broadcast + 4 slices
            # instead of wide [PMAX,4*PMAX] broadcast.
            # =============================================================
            for qsi in nl.affine_range(num_s_tiles):
                qso = qsi * PMAX

                # Retrieve pre-computed masked score tiles for this qsi
                masked_wide = all_masked_wide[qsi]
                masked_tail_tiles = all_masked_tail[qsi]

                # --- Step 2: Global row maximum across all score tiles ---
                row_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"p2_rowmax_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=masked_wide, axis=1)

                # Update row_max with maximums from each tail tile
                for ti in range(len(masked_tail_tiles)):
                    sm_t = masked_tail_tiles[ti]
                    tmax_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_tmaxt_kv{local_kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_reduce(dst=tmax_t, op=nl.maximum, data=sm_t, axis=1)
                    nmax_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_nmaxt_kv{local_kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_tensor(nmax_t, row_max, tmax_t, op=nl.maximum)
                    nisa.tensor_copy(row_max, nmax_t)

                # --- Step 3: Broadcast neg_max, compute exp, accumulate ---
                neg_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"p2_negmax_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(neg_max, row_max, op0=nl.multiply, operand0=-1.0)

                # v7b Change 1: Single narrow broadcast of neg_max [PMAX,1] -> [PMAX,PMAX].
                # Replaces the expensive wide [PMAX,1] -> [PMAX, 4*PMAX] broadcast from v6a.
                # This is the ONLY broadcast per qsi iteration — reused for both wide slices and tail.
                neg_max_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_nmbc_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=neg_max_bc,
                              src=neg_max.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                # v7b Change 2: Process wide score tile as 4 separate [PMAX,PMAX] slices.
                # Each slice is shifted by the shared neg_max_bc and exp'd independently.
                # Row sums are accumulated across slices to get the full wide row_sum.
                atacc_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum,
                                      name=f"p2_atp_kv{local_kvh}_g{gi}_s{qsi}")

                # Initialize row_sum with the first slice, then accumulate remaining slices.
                # Using a Python variable to track the current row_sum tensor.
                row_sum = None

                for i in range(BATCH):
                    # Extract the i-th [PMAX, PMAX] slice from the wide masked score tile
                    slice_i = masked_wide[0:PMAX, i*PMAX:(i+1)*PMAX]

                    # Shift by neg_max: score_slice + (-max) = score_slice - max
                    shifted_i = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                           name=f"p2_shi_kv{local_kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_tensor(shifted_i, slice_i, neg_max_bc, op=nl.add)

                    # Compute exp(score - max) for this slice
                    sexp_i = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_sei_kv{local_kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.activation(sexp_i, op=nl.exp, data=shifted_i)

                    # Accumulate partial row sum: reduce this slice's exp values along axis=1
                    partial_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                             name=f"p2_psi_kv{local_kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_reduce(dst=partial_sum, op=nl.add, data=sexp_i, axis=1)

                    if i == 0:
                        # First slice: initialize row_sum directly
                        row_sum = partial_sum
                    else:
                        # Subsequent slices: add partial_sum into running row_sum
                        new_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                             name=f"p2_nsi_kv{local_kvh}_g{gi}_s{qsi}_i{i}")
                        nisa.tensor_tensor(new_sum, row_sum, partial_sum, op=nl.add)
                        row_sum = new_sum

                    # MM2 for this slice: transpose sexp_i and multiply with V_tiles[i]
                    sexp_i_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                           name=f"p2_seibf_kv{local_kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_copy(sexp_i_bf, sexp_i)
                    sexpT_i = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                         name=f"p2_seTi_kv{local_kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.nc_transpose(sexpT_i, sexp_i_bf)
                    sexpTs_i = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                          name=f"p2_seTsi_kv{local_kvh}_g{gi}_s{qsi}_i{i}")
                    nisa.tensor_copy(sexpTs_i, sexpT_i)
                    # Accumulate: exp(score) @ V into attention output
                    nisa.nc_matmul(atacc_psum, stationary=sexpTs_i, moving=V_tiles[i])

                # v7b Change 3: Reuse neg_max_bc for tail tiles — no second broadcast needed!
                # The same [PMAX, PMAX] neg_max_bc computed above works for all tail tiles
                # since it's the same neg_max value broadcast to the same [PMAX, PMAX] shape.
                for ti in range(len(masked_tail_tiles)):
                    sm_t = masked_tail_tiles[ti]
                    ki_abs = tail_start + ti

                    # Shift by neg_max (reusing neg_max_bc from the single broadcast above)
                    shifted_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                           name=f"p2_sht_kv{local_kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_tensor(shifted_t, sm_t, neg_max_bc, op=nl.add)
                    sexp_t = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_set_kv{local_kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.activation(sexp_t, op=nl.exp, data=shifted_t)

                    # Accumulate this tail tile's exp sum into row_sum
                    tsum_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                        name=f"p2_tsumt_kv{local_kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_reduce(dst=tsum_t, op=nl.add, data=sexp_t, axis=1)
                    nrs_t = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"p2_nrst_kv{local_kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_tensor(nrs_t, row_sum, tsum_t, op=nl.add)
                    row_sum = nrs_t

                    # MM2 for tail tile: transpose exp'd scores, matmul with V
                    sexp_t_bf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                           name=f"p2_setbf_kv{local_kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_copy(sexp_t_bf, sexp_t)
                    sexpT_t = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum,
                                         name=f"p2_seTt_kv{local_kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.nc_transpose(sexpT_t, sexp_t_bf)
                    sexpTs_t = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                          name=f"p2_seTst_kv{local_kvh}_g{gi}_s{qsi}_t{ti}")
                    nisa.tensor_copy(sexpTs_t, sexpT_t)
                    nisa.nc_matmul(atacc_psum, stationary=sexpTs_t, moving=V_tiles[ki_abs])

                # Copy accumulated attention output from PSUM to SBUF
                atacc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"p2_atacc_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_copy(atacc, atacc_psum)

                # --- Step 4: Normalize by row sum (softmax denominator) ---
                # Compute 1/row_sum via rsqrt(row_sum)^2, with epsilon for safety
                ssafe1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_ssafe1_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_scalar(ssafe1, row_sum, op0=nl.add, operand0=1e-9)
                rsq1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_rsq1_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.activation(rsq1, op=nl.rsqrt, data=ssafe1)
                inv1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_inv1_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(inv1, rsq1, rsq1, op=nl.multiply)

                # Broadcast [PMAX,1] inverse -> [PMAX,PMAX] for element-wise normalization
                inv_bc = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                    name=f"p2_inv_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.dma_copy(dst=inv_bc,
                              src=inv1.ap(pattern=[[1, PMAX], [0, PMAX]], offset=0))

                # Multiply accumulated V by 1/sum to get final softmax-weighted output
                aout = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"p2_aout_kv{local_kvh}_g{gi}_s{qsi}")
                nisa.tensor_tensor(aout, atacc, inv_bc, op=nl.multiply)

                # Cast to bf16 and store to HBM output.
                # v8a: qh is the ABSOLUTE Q head index (kvh*gqa+gi), so qh*d gives
                # the correct column offset into the shared output tensor.
                # Each core writes non-overlapping column ranges — no conflict.
                aoutb = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                                   name=f"p2_aoutb_kv{local_kvh}_g{gi}_s{qsi}")
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
        print(f"qwen3_attn_cte_fused v8a Test: B={B}, S={S}, H={H}, d={d}")
        print(f"Hq_tp={Hq_tp}, Hkv_tp={Hkv_tp}, gqa={Hq_tp // Hkv_tp}")
        print(f"LNC=2: core 0 handles kvh [0,{Hkv_tp//2}), core 1 handles kvh [{Hkv_tp//2},{Hkv_tp})")
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
        # v8a: Launch with [2] to assign 2 NeuronCores (LNC=2).
        # Each core receives pid=0 or pid=1 via nl.program_id(0).
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
