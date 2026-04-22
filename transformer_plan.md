Plan B — Cross-layer ring-buffered weight prefetch

  Detailed build-on-Plan-A spec. Three files change: transformer_qwen_multilayer.py, a new
  v14d_kv_norm_hoisted_weights.py (fork of v14c), and a new kernel_v30c_hoisted.py (fork of v30b). Sub-kernels gain
  "skip internal load" kwargs the same way v14c already does for constants.

  ---
  What we're prefetching

  Classify every per-layer weight by where it can live:

  ┌─────────────────────────────┬─────────────────────────────────────────────┬──────────────────────────────────┐
  │           Weight            │            Per-layer size (core)            │             Strategy             │
  ├─────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────┤
  │ Wk                          │ 512 KiB                                     │ 2-deep ring                      │
  ├─────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────┤
  │ Wv                          │ 512 KiB                                     │ 2-deep ring                      │
  ├─────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────┤
  │ Wq (4 owned heads × 512     │ 2 MiB                                       │ 2-deep ring                      │
  │ KiB)                        │                                             │                                  │
  ├─────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────┤
  │ Wo (4 owned heads × 512     │ 2 MiB                                       │ 2-deep ring                      │
  │ KiB)                        │                                             │                                  │
  ├─────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────┤
  │ router_w wide-SBUF form     │ 512 KiB                                     │ 2-deep ring                      │
  ├─────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────┤
  │ gpost (transposed gamma_sb) │ 4 KiB                                       │ full hoist (48× = 192 KiB)       │
  ├─────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────┤
  │ gate_up_w, down_w           │ dynamic (expert-indexed)                    │ stays inside MoE — cannot        │
  │                             │                                             │ prefetch                         │
  ├─────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────────┤
  │ K/V cache tiles             │ dynamic (pos-dependent but                  │ skip in this plan (Plan C)       │
  │                             │ layer-independent)                          │                                  │
  └─────────────────────────────┴─────────────────────────────────────────────┴──────────────────────────────────┘

  Resident SBUF budget on one core, additive to Plan A:
  - Rings: 2 × (512 K + 512 K + 2 M + 2 M + 512 K) = 11 MiB
  - Full-hoisted gpost: 192 KiB
  - Plan A constants already resident: ~450 KiB

  Total resident ≈ 11.6 MiB. With 32 MiB SBUF per core, leaves ~20 MiB for transient attention/MoE scratch (which
  today uses ~6–8 MiB peak). Fits.

  ---
  Changes to transformer_qwen_multilayer.py

  1. After the Plan A hoist block, add persistent ring allocations

  sbm.set_auto_alloc(True)   # rings live in the hoist scope, not per-layer

  # Owned head indices mirror v14c's LNC-sharding
  if prg_id == 0:
      OWNED = [0, 1, 2, 3]
  else:
      OWNED = [4, 5, 6, 7]
  NOH = len(OWNED)                 # 4

  # Two-slot ring buffers — block-dim 2 is the "slot", partition dim is PMAX
  Wk_ring = sbm.alloc_stack((2, PMAX, NH * PMAX),        dtype, name="Wk_ring")
  Wv_ring = sbm.alloc_stack((2, PMAX, NH * PMAX),        dtype, name="Wv_ring")
  # Wq / Wo are per-head — ring × owned_heads dims before par_dim
  Wq_ring = [[sbm.alloc_stack((PMAX, NH * PMAX), dtype, name=f"Wq_ring_s{s}_h{h}")
              for h in range(NOH)] for s in range(2)]
  Wo_ring = [[sbm.alloc_stack((PMAX, H),         dtype, name=f"Wo_ring_s{s}_h{h}")
              for h in range(NOH)] for s in range(2)]
  Router_ring = sbm.alloc_stack((2, PMAX, 16, 128),      dtype, name="Router_ring")

  # Full-hoisted gpost: pre-transposed `gamma_sb` for every layer
  gpost_bf16_all = sbm.alloc_stack((PMAX, num_layers * NH), dtype, name="gpost_bf16_all")
  for li in nl.affine_range(num_layers):
      # Replicates the DMA+transpose that v30b does internally for gamma_sb,
      # but writes into slice [li*NH:(li+1)*NH] instead of allocating per-layer.
      _load_gpost_into_slot(gpost_bf16_all[:, li*NH:(li+1)*NH], gpost_list[li], sbm)

  Helper _load_gpost_into_slot mirrors v30b lines 99–106 (DMA [H_free, PMAX], nc_transpose to [PMAX, H_free],
  activation copy into the slot).

  2. Pre-roll layer 0 into slot 0

  def _prefetch_layer_weights(slot, li):
      nisa.dma_copy(dst=Wk_ring[slot], src=Wk_list[li], dge_mode=nisa.dge_mode.hwdge)
      nisa.dma_copy(dst=Wv_ring[slot], src=Wv_list[li], dge_mode=nisa.dge_mode.hwdge)
      for hi, q_h in enumerate(OWNED):
          nisa.dma_copy(
              dst=Wq_ring[slot][hi],
              src=Wq_list[li][q_h * PMAX:(q_h + 1) * PMAX, :],
              dge_mode=nisa.dge_mode.hwdge,
          )
          # Wo AP pattern is the same one v14c uses after Wo.reshape((Hq_tp, d, H_wo))
          nisa.dma_copy(
              dst=Wo_ring[slot][hi],
              src=Wo_list[li].reshape((8, PMAX, H)).ap(
                  pattern=[[H, PMAX], [1, H]],
                  offset=q_h * PMAX * H,
              ),
              dge_mode=nisa.dge_mode.hwdge,
          )
      # Router wide-SBUF form — replicates v30b lines 157–165 for h_chunk=0
      # (and stride pattern indexed by slot instead of h_chunk).
      nisa.dma_copy(
          dst=Router_ring[slot],
          src=router_list[li].ap(
              pattern=[[128, PMAX], [PMAX * 128, 16], [1, 128]],
              offset=0,
          ),
          dge_mode=3,
      )

  _prefetch_layer_weights(0, 0)

  3. Restructure the layer loop

  for layer_idx in range(num_layers):
      cur   = layer_idx & 1
      nxt   = (layer_idx + 1) & 1

      # ---- issue (i+1)'s prefetch FIRST so it overlaps layer-i compute ----
      if layer_idx + 1 < num_layers:
          _prefetch_layer_weights(nxt, layer_idx + 1)

      # ---- Attention: consume pre-loaded weights from slot `cur` ----
      out_sb = qwen3_attn_tkg_fused_oproj_v13bc_kv_norm_hoisted_weights(
          hidden_sb=residual_sb,
          Wq=Wq_list[layer_idx],   # HBM fallback; unused when wq_heads_sb given
          Wk=Wk_list[layer_idx],
          Wv=Wv_list[layer_idx],
          Wo=Wo_list[layer_idx],
          K_cache=K_caches[layer_idx],
          V_cache=V_caches[layer_idx],
          cos=cos, sin=sin,
          position_ids=position_ids,
          q_norm_weight=qn_list[layer_idx],
          k_norm_weight=kn_list[layer_idx],
          gamma_pre_attn=gpre_list[layer_idx],
          sbm=sbm,
          # Plan A:
          qnw_f32_sb = qnw_f32_all[:, layer_idx:layer_idx+1],
          knw_f32_sb = knw_f32_all[:, layer_idx:layer_idx+1],
          cos_f32_sb = cos_f32_all,
          sin_f32_sb = sin_f32_all,
          gpan_f32_sb = gpan_f32_all[:, layer_idx*NH:(layer_idx+1)*NH],
          # Plan B:
          wk_sb         = Wk_ring[cur],
          wv_sb         = Wv_ring[cur],
          wq_heads_sb   = Wq_ring[cur],          # list length NOH
          wo_heads_sb   = Wo_ring[cur],          # list length NOH
          owned_heads   = OWNED,
      )

      # AllReduce #1 + residual add #1 (unchanged)

      moe_out_sb = _qwen3_moe_sbuf_in_sbuf_out_hoisted(
          inp_sb=residual_sb, dtype=dtype, T=T,
          gamma=gpost_list[layer_idx],         # HBM fallback
          router_w=router_list[layer_idx],     # HBM fallback
          gate_up_w=gate_up_list[layer_idx],
          down_w=down_list[layer_idx],
          sbm=sbm,
          # Plan B:
          gamma_sb_ready  = gpost_bf16_all[:, layer_idx*NH:(layer_idx+1)*NH],
          router_w_wide_sb = Router_ring[cur],
      )

      # SB2SB gather + residual add #2 (unchanged)

  Key invariant: the _prefetch_layer_weights(nxt, layer_idx + 1) call sits before the attention call but writes to nxt
   = (i+1)&1, while attention reads from cur = i&1. No dependency → compiler overlaps them. This is the standard
  "issue next-iteration DMA before consuming current-iteration slot" pattern from the SBUF guide §"#1 Hoist
  allocations outside loop-nests".

  4. Remove the per-layer while sbm.heap: pop_heap() for ring tensors

  The current pop_heap loop after MoE is fine — the rings are allocated in hoist scope (above the loop), not pushed to
   the heap inside the layer. But ensure the MoE sub-kernel's sbm.alloc_heap calls don't collide with the ring
  addresses; that's automatic since everything goes through the same sbm.

  ---
  Changes to attention — new v14d_kv_norm_hoisted_weights.py

  Fork v14c. Add kwargs:

  wk_sb=None,         # [PMAX, NH*d]   bf16 SBUF — skip wk_sb alloc+DMA when given
  wv_sb=None,         # [PMAX, NH*d]   bf16 SBUF
  wq_heads_sb=None,   # list of len(owned_heads) of [PMAX, NH*d] bf16 — skip Wq loop
  wo_heads_sb=None,   # list of len(owned_heads) of [PMAX, H_wo] bf16 — skip Wo loop
  owned_heads=None,   # list of ints; when wq_heads_sb is given, index i in the list
                      # maps to global head `owned_heads[i]`. The sub-kernel's own
                      # LNC shard logic becomes a sanity check.

  Specific blocks that change:

  Lines 135–143 of v14c (wk/wv internal DMA) — wrap with if wk_sb is None, else rebind local wk_sb = wk_sb (and same
  for wv_sb).

  Lines 228–240 of v14c (Wq head load loop) — replace the allocation + DMA loop with:
  if wq_heads_sb is None:
      wq_head_sb = [None] * HQ_TP_CONST
      for q_h in owned_heads:
          w = sbm.alloc_stack((PMAX, NH * d), nl.bfloat16, name=f"wq_head_{q_h}")
          nisa.dma_copy(dst=w, src=Wq[q_h*PMAX:(q_h+1)*PMAX, :], dge_mode=nisa.dge_mode.hwdge)
          wq_head_sb[q_h] = w
  else:
      wq_head_sb = [None] * HQ_TP_CONST
      for i, q_h in enumerate(owned_heads):
          wq_head_sb[q_h] = wq_heads_sb[i]   # caller-owned; no alloc, no DMA

  Wo loop (v14c lines 418–426) — same pattern. When wo_heads_sb is provided, skip the wo_sbuf[head] allocation (lines
  258–260) AND the DMA loop; just set wo_sbuf[q_h] = wo_heads_sb[i].

  Leave K/V cache DMA (lines 119–133) alone — Plan C territory.

  Add an assertion that owned_heads matches the internal computed list so a caller using different LNC sharding gets a
   clear failure.

  ---
  Changes to MoE — new kernel_v30c_hoisted.py

  Fork v30b. Add kwargs:

  gamma_sb_ready=None,   # [PMAX, H_free] bf16 SBUF — pre-loaded & pre-transposed
  router_w_wide_sb=None, # [PMAX, _ROUTER_BATCH=16, E=128] bf16 SBUF — replaces wide router DMA

  gamma block (lines 99–106) — wrap in if gamma_sb_ready is None. When provided, set gamma_sb = gamma_sb_ready and
  skip the DMA + nc_transpose + activation.

  Router stage 2 (lines 155–172) — the outer h_chunk loop ranges over H_free // _ROUTER_BATCH = 1, i.e. it's a single
  DMA in practice. When router_w_wide_sb is provided:
  if router_w_wide_sb is None:
      router_w_wide_sb = sbm.alloc_stack((PMAX, _ROUTER_BATCH, E), dtype, name="router_w_wide_sb")
      nisa.dma_copy(dst=router_w_wide_sb, src=router_w.ap(...), dge_mode=3)
  # Then the inner matmul loop (lines 166–172) consumes router_w_wide_sb as-is.

  Everything else in the MoE (gate/up/down per-expert DMA + compute) is unchanged because those are
  per-token/per-expert and cannot be cross-layer prefetched.

  ---
  Correctness invariants to preserve

  1. Layout match: Wq_ring[slot][hi] must have identical byte-layout to what v14c's internal Wq[q_h*PMAX:(q_h+1)*PMAX,
   :] + dma_copy produces. Both are contiguous [PMAX, NH*d] bf16 in the tile-transposed HBM layout — verify with a
  single-layer correctness test before hooking multi-layer.
  2. Wo AP pattern in the prefetch must match v14c lines 418–426 verbatim ([[H_wo, PMAX], [1, H_wo]], offset head *
  PMAX * H_wo).
  3. Router DMA pattern matches v30b lines 157–165. The batched _ROUTER_BATCH=16 form is exactly one tile, so one
  prefetch DMA suffices.
  4. gpost pre-transpose mirrors v30b lines 99–106. Verify that gpost_bf16_all[:, li*NH:(li+1)*NH] is indeed [PMAX,
  16] post-transposed.
  5. Slot i+1 writes don't race with slot i reads: true by definition since cur != nxt. But confirm no tensor aliases
  the two slots (they must be at different byte offsets — sbm.alloc_stack with a block-dim handles this; two separate
  alloc_stack calls for Wq_ring[0][hi] and Wq_ring[1][hi] also work and are simpler).
  6. No close_scope / pop_heap can reach into the hoist scope. Ensure the per-layer sbm.close_scope() / pop_heap calls
   are balanced against scopes opened inside the layer body only.

  ---
  Verification

  1. Single-layer baseline: run num_layers=1 end-to-end against the reference NxDI model via main.py --mode validate.
  Should match v14c+v30b exactly; this validates the skip-internal-load kwarg wiring.
  2. Two-layer ping-pong: num_layers=2 — this is the smallest case that exercises both slots of every ring and the
  prefetch-before-compute ordering. Run the same validation.
  3. Full 48 layers: e2e validate under tol_map from CLAUDE.md.
  4. Profile with NEURON_RT_INSPECT_*: expect dma_active_pct to drop and tensor_engine_pct to rise; device_time_us per
   layer should shrink because the prologue HBM stalls are now hidden.

  ---
  Suggested rollout order (de-risks separately)

  1. Fork v14c → v14d with the new kwargs behind default=None (back-compat). Land + verify with multilayer still
  passing None (no behavioural change).
  2. Fork v30b → v30c same way. Verify.
  3. Add Plan A-style full-hoist of gpost (no ring yet). Verify.
  4. Add Wk_ring + Wv_ring only (cheapest — 1 MiB total, easy to reason about). Verify.
  5. Add Wq_ring, Wo_ring, Router_ring. Verify.
  6. Measure.

  Want me to dispatch a subagent to implement step 1 (v14c → v14d) + step 2 (v30b → v30c) as a single change, keeping
  current behaviour, then come back to me before wiring the outer rings?