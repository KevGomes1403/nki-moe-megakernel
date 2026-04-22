# Plan [G] — Cross-Layer Expert Prefetch: Design & Integration Notes

**Status**: skeleton shipped as `kernels/moe_fused_tkg/quantized/v15g.py`
(structurally identical to v15c; no measurable within-kernel gain available).
Full [G] gain requires a multi-layer harness. This doc specifies the
interface, required changes, and blockers encountered during Round 2 Plan C.

**Baseline**: v15c at 72.29 μs (3-run mean, trn3).

**Target**: Fate-style (arXiv 2502.12224, 97% top-8 accuracy) speculative
prefetch of layer N+1's top-K experts during layer N's compute. Phase 1
matrix Δscore ≈ 439 (the single largest lever remaining).

---

## Why the within-kernel probes failed to yield a measurable win

v15c already does the "within-kernel Wave 0 → Wave 1 prefetch": all 8
expert DMAs are issued up front (Phase 1a then Phase 1b), then Wave 0
compute runs while Wave 1 DMAs drain. Three variants were tried in v15g
to expose additional overlap slack:

| Attempt | Change | device_time_us | Δ vs v15c |
|---|---|---|---|
| v15c baseline | — | 72.33 (single) / 72.29 (mean) | 0 |
| 1a: defer Wave 1 scale extract + up-scale assembly until after Wave 0 compute | Moves VectorE/ScalarE work into Wave 0 → Wave 1 seam | 78.60 | +8.7% |
| 1b: group DMAs by TYPE (all 8 gate_up, then tile-0, then tile-1, then scales) | HWDGE/SWDGE queues fill in parallel | 74.92 | +3.6% |
| 1c: hoist only Wave 1 gate_up DMAs ahead of Wave 0 down_w tiles | Fat Wave 1 DMAs start earliest | 75.20 | +4.0% |
| v15g (final, matches v15c ordering) | skeleton only, no change | 71.65 (mean of 3) | −0.9% (noise) |

**Pattern**: every reordering reduced gpsimd_engine_pct (30.4 → 24-27%) and
dma_active_pct (57.4 → 49.8-53.8%), but tensor_engine_pct dropped by a
similar amount and wall time increased. The compiler's default schedule on
v15c already effectively overlaps DMAs with compute via affine_range
prefetch + HWDGE/SWDGE dual-queue parallelism.

**Conclusion**: within a single isolated v15c kernel invocation, the
overlap ceiling is already hit. Extra overlap slack lives only ACROSS the
layer boundary (attention → RMSNorm → router → MoE, next layer).

---

## Predictor interface (for a future multi-layer harness)

### Input features

Per the Fate paper, a lightweight linear-projection head applied to the
current layer's post-layernorm hidden state (`rmsnorm_normed` in v15c)
achieves ~97% top-8 accuracy for the NEXT layer's expert set on Qwen3-like
MoE models. Minimum viable feature vector:

```
features:  [T=1, H=2048]  bf16   # rmsnorm_normed of current layer
             (already in SBUF at stage 2 of v15c)
```

### Predictor head

Offline-trained weight matrix stored alongside the decoder weights:

```
pred_w:    [H=2048, E=128]  bf16   # per-layer: pred_w[layer_idx]
                                    # shape matches router_w but is
                                    # a separate learned projection
                                    # for the NEXT layer
```

Compute (cheap — a single bf16 matmul reusing the same stationary input
as the current layer's router):

```
pred_logits = rmsnorm_normed_bf16 @ pred_w[layer_idx]    # [T, E=128]
pred_top8_idx = nc_find_index8(pred_logits, topk=8)      # [T, K=8]
```

This runs in parallel with the current layer's router+softmax+topk on
TensorE+VectorE — negligible added cost (same op count as the current
router, and the activations are already in SBUF).

### Ring buffer structure

Three SBUF ring positions instead of the current two (`*_bufs` and
`*_w1_bufs`):

```
gate_up_packed_ring[3][K_WAVE=4][...]    # 3 buffer slots
down_full0_fp8_ring[3][K_WAVE=4][...]
down_full1_fp8_ring[3][K_WAVE=4][...]
down_scale_ring[3][K_WAVE=4][...]
# plus analogous ring indexing for the w1-style "second-half-of-wave" bufs
```

Ring slot assignment per layer:
- slot (layer_idx % 3): consumed by current layer's Wave 0 compute
- slot ((layer_idx+1) % 3): filled by current layer's Wave 0 → Wave 1
  carry-over DMAs; consumed by current layer's Wave 1 compute
- slot ((layer_idx+2) % 3): speculative prefetch for layer+1, filled
  during current layer's Phase 2a/2b compute

This requires SBUF budget × 1.5 vs v15c. v15c's current SBUF footprint
for the 2-wave structure is ~6.4 MiB (gate_up_packed×8 buffers × 817 KiB +
down_w tiles + scales). Three-slot ring would push to ~9.6 MiB, well
within trn3's 32 MiB SBUF.

---

## Required kernel-level changes (checklist)

1. **Accept extra inputs**:
   - `pred_w: [H, E] bf16` (next-layer predictor weights)
   - `layer_idx: scalar int` (ring slot selector via runtime param)
   - `prev_layer_pred_top8_idx: [T, K] int32` (last layer's prediction,
     to pick which ring slot's prefetched weights to consume NOW).
   First layer has no prediction → use current-layer `top8_idx` directly
   (no prefetch speedup on layer 0).

2. **Add predictor compute stage** (between stage 2 router matmul and
   stage 3 softmax/topk):
   - Reuse `rmsnorm_normed_bf16` stationary
   - Single `nc_matmul` against `pred_w` → `pred_logits_psum`
   - `max8` + `nc_find_index8` for `pred_top8_idx` (same cost as current
     router's topk: negligible on VectorE).

3. **Restructure buffer allocation** to a 3-slot ring (see above).

4. **Splice in speculative prefetch DMAs** in the tail of Phase 2b:
   After the last Wave 1 matmul, before the stage-5 transpose/store, emit
   8 DMAs (4 Wave 0 + 4 Wave 1 for layer+1) into the ring slot
   `(layer_idx+2) % 3`. These DMAs overlap with the final PSUM drain + bf16
   output store, hiding the ~40 μs MoE DMA cost of layer+1 under layer N's
   ~5-10 μs of trailing compute. (Gap analysis: most of Fate's reported
   gain comes from overlapping weight DMAs with the PRECEDING attention
   kernel's compute, not the tail of MoE — see integration section.)

5. **Expose a correctness bypass**: when `prev_layer_pred_top8_idx` does
   NOT match the current layer's computed `top8_idx`, fall back to loading
   the required experts synchronously. The cache miss cost is the full
   v15c time; hit cost is ~25-30 μs (est.). At 97% accuracy, expected
   time = 0.97 × 30 + 0.03 × 72 ≈ 31.3 μs per layer.

---

## Required integration-level changes

The single biggest blocker for measuring real gain is that the prefetch
covers the PRECEDING layer's attention kernel — not within-MoE overlap.
The integration requires:

### Option A — multi-layer megakernel (preferred, aligns with project
CLAUDE.md "Multi-Layer Fused TKG Megakernel" effort)

Per CLAUDE.md, the ongoing effort extends v15c + v13bc attention into a
single NKI invocation running all decoder layers. The predictor DMA
splicing fits naturally:

- Layer N's MoE Phase 2b tail emits speculative DMAs for layer N+1's
  predicted top-K into the `(N+2) % 3` ring slot.
- Layer N+1's attention kernel compute (the big TensorE consumer) runs
  while those DMAs drain.
- By the time layer N+1's MoE Phase 1a would normally issue DMAs, the
  predicted experts are already resident — Phase 1a becomes a no-op
  (for the 97% hit case) and Phase 1b issues only the CORRECTIVE DMAs
  for any mispredicted slots.

This requires the megakernel to expose a per-layer `layer_idx` that can
be mapped to a ring slot inside the kernel.

### Option B — tight transformer-loop harness

If the megakernel is not available, a tighter Python/XLA loop that
invokes attention + MoE back-to-back without mark_step boundaries could
still get some gain — but the NKI JIT boundary is currently too
expensive to cross cheaply between layers. This option is lower-value.

---

## Specific blockers encountered during Round 2 Plan C

1. **Isolated-kernel ceiling**: three within-kernel DMA reordering
   variants all regressed. No within-kernel speculative-prefetch variant
   can produce a measurable gain without adding a 3rd buffer set AND
   simulating a "fake next layer" — the latter is not a real measurement.

2. **NEFF caching**: v15g with unchanged DMA ordering hits the same
   NEFF hash as v15c (cached compile), confirming structural equivalence.
   This is the cleanest no-regression proof point.

3. **Predictor cannot be stubbed in isolation**: a fake predictor
   producing random top-K + extra DMA round-trip would show only the
   OVERHEAD of the extra matmul + DMA descriptors (likely +3-5 μs on
   v15c), with no corresponding gain to offset. Measuring the full [G]
   balance requires the multi-layer harness.

4. **SBUF budget for 3-slot ring**: ~9.6 MiB vs v15c's ~6.4 MiB. Fits,
   but needs explicit per-slot declarations (no dynamic indexing of
   distinct `nl.ndarray` objects in NKI). This is an engineering-scale
   refactor: 3× repetition of every buffer declaration in stage 4.

---

## Estimated effort to land full [G]

| Phase | Scope | Effort |
|---|---|---|
| Predictor-head offline training | Qwen3-30B-A3B, single epoch, ~1 GPU-hour | 0.5 day |
| State-dict converter emits `pred_w` per layer | Python, pack along router_w | 0.5 day |
| Megakernel per-layer ring-slot wiring | `layer_idx` → ring slot math, 3× buffer decls | 2 days |
| Wave 1 DMA → speculative prefetch refactor | restructure Phase 2b tail, fallback path | 1.5 days |
| Correctness validation (bit-exact vs non-speculative reference) | test harness with forced top-K match | 0.5 day |
| End-to-end bench in megakernel harness | 24-layer loop, 3 runs, delta vs non-prefetch megakernel | 0.5 day |
| **Total** | | **~5.5 days** |

**Contingent on**: the megakernel landing (per CLAUDE.md, that work is
ongoing). Plan [G] cannot ship standalone.

---

## Recommendation

**Punt Plan [G] to Round 3**, after the multi-layer megakernel lands.
The three deliverables below represent the Round 2 Plan C outcome:

1. `kernels/moe_fused_tkg/quantized/v15g.py` — skeleton kernel,
   structurally identical to v15c, bit-exact, same performance.
2. `kernels/moe_fused_tkg/quantized/test_v15g.py` — correctness test
   (bit-exact vs v15c).
3. `kernels/moe_fused_tkg/quantized/bench_v15g_trn3.py` — benchmark.
4. This design doc (`docs/plan_g_cross_layer_prefetch_design.md`) —
   full interface spec for the multi-layer integration.

The three within-kernel reordering probes (Attempts 1a/1b/1c) are
documented in v15g.py's module docstring and this doc; they conclusively
establish the within-kernel ceiling and eliminate the single-kernel
variants from the Round 3 candidate list.
