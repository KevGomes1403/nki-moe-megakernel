# gpt-oss Megakernel TKG — Host-Overhead Roadmap

Goal: close the ~460 µs/token host-vs-device wall-time gap that makes the megakernel
**+7.8 % slower e2e** at p50 despite being **−9.4 % faster on-device** (5.234 ms vs 5.774 ms
TKG-bk=640). See `bk640_xla_vs_megakernel_comparison.md` for the source evidence.

## Phase 1 — Override `NeuronGptOssModel.forward` to bypass the per-layer loop

**Status: shipped (2026-05-11).** TKG p50 dropped 5.926 → 5.879 ms (-0.05 ms / -0.8 %),
TKG p99 dropped 6.492 → 6.401 ms (-0.09 ms / -1.4 %), e2e p50 dropped 4116.4 → 3989.9
ms (-126.5 ms / -3.1 %) on prompt 1. Per-call host savings are smaller than the ~460 µs
target gap; the remaining overhead is the weight-binding count (Phase 2) and the
49-output return tuple (Phase 3). Implementation: `NeuronGptOssModelV2.get_model_output`
overrides at the `get_model_output` level (cleaner than `forward` — reuses NxDI's mask
prep). `NeuronGptOssDecoderLayerV1` retains only the kernel-layout weight-param
registration; its `forward` and the `_tkg_kv_out` relay are gone. **Don't cache
`_gather_weights`** — Parameter `.data` is rebound per-trace, and a cached ref from
trace 1 trips `convert_parameters_to_constants` on trace 2 with "XLA data... async
operation in flight". Gathering every call is cheap because the heavy transposes /
bias-divide are pre-baked into the state dict.

**Lever 1.** Probable largest contributor; CLAUDE.md already lists this as the intended
end-state for the multi-layer fused TKG work. Worth doing independent of P0 outcome.

Current shape (`gpt_oss_with_megakernel.py` + NxDI `model_base.py:1340–1395`):
- `NeuronGptOssModelV1` keeps NxDI's `for idx, decoder_layer in enumerate(self.layers)`.
- Layer 0 invokes the megakernel and stashes outputs in `self._parent_model._tkg_kv_out`.
- Layers 1..23 each call `forward()` returning cached `(K_out[idx], V_out[idx])`.
- Trace builds 24 layer-iter graph nodes; `next_decoder_cache += kv` accumulates 48 K/V refs.

Target shape:
- New `NeuronGptOssModelV2(NeuronGptOssModel)` overrides `forward`.
- TKG path: embed → single megakernel call → norm → return `(Y, ())` (KV in-place).
- CTE path: delegate to `super().forward()` unchanged.
- `NeuronGptOssDecoderLayerV1` no longer needed; can be deleted along with the pass-through logic and `_tkg_kv_out` parent-attribute relay.

Tasks:
- [ ] **P1.1** Write `NeuronGptOssModelV2.forward` that branches on `is_for_context_encoding`.
      Reuse stock NxDI path for CTE; megakernel-direct path for TKG.
- [ ] **P1.2** Move weight-gathering off the layer and onto the model. Cache the gathered
      lists on the model after first call (or build them at `init_model` time once
      weights are loaded).
- [ ] **P1.3** Move mask + RoPE prep onto the model TKG path (same code as today's
      `NeuronGptOssDecoderLayerV1.forward`, just hoisted up one level).
- [ ] **P1.4** Delete `NeuronGptOssDecoderLayerV1` and `_tkg_kv_out` once V2 is wired.
- [ ] **P1.5** Verify accuracy via `python main.py --model gpt_oss --mode validate --enable-nki`.
- [ ] **P1.6** Re-run benchmark; record new TKG p50 / p99 / e2e p50 in this file.

**Acceptance:** TKG p50 drops below baseline's 5.46 ms, OR P0 diagnostics localize the
remaining gap to a different lever.

## Phase 2 — Stack per-layer weights into rank-3 tensors

**Lever 2.** Only do this if P0 says binding-count is the cost, or if Phase 1 doesn't close
the gap fully.

Current: `convert_gpt_oss_hf_to_neuron_state_dict` registers 24 `Wqkv_nki`, 24 `Wo_nki`, etc.
Kernel signature is ~370 explicit args. NEFF has correspondingly many input bindings.

Target:
- One stacked HBM tensor per weight type: `Wqkv_all [L, H, I]`, `gate_up_all [L, E, H, 2, I]`, etc.
- Kernel signature drops from ~370 to ~15. `_multilayer_body` indexes by layer with
  `nl.subview(Wqkv_all, layer_idx, axis=0)`.

Tasks:
- [ ] **P2.1** Standalone smoke test: confirm `nl.subview` on a stacked-weight tensor
      compiles into a single SBUF load (no extra HBM roundtrip vs. today's per-layer
      DMA). The comment at `transformer_gpt_oss.py:286–289` warns that
      "NKI classifies tuple/list args as scalars" — verify the alternative doesn't
      regress.
- [ ] **P2.2** Update `convert_gpt_oss_hf_to_neuron_state_dict` to emit stacked tensors.
- [ ] **P2.3** Update kernel signature + `_build_multilayer_kernel` codegen.
- [ ] **P2.4** Re-run validate + benchmark; record numbers.

**Acceptance:** No regression in on-device TKG total_time; observable drop in
`dmem_buf_batch_copyin` host time.

## Phase 3 — Mark K/V outputs as in-place, drop from return tuple

**Lever 3.** Smaller win (~50–150 µs estimated). Bundle with Phase 2 if you're touching
the kernel signature anyway.

Current: `transformer_gpt_oss.py:281`: `return (Y,) + tuple(K_post) + tuple(V_post)` — 49 outputs.
KV writes already happen in-place via `attention_block_tkg`'s scatter; the returns exist
only to keep NCC's data-dependency analysis happy.

Tasks:
- [ ] **P3.1** Investigate whether NKI / NCC supports an in-place-input annotation
      (a pragma or decorator) that lets the kernel mutate an HBM tensor and declare
      the side effect without including it in the return list. Check `nki.jit`
      kwargs and `nki.in_place` / similar in `nkilib` source.
- [ ] **P3.2** If supported: drop K/V from returns; assert KV state still updates in-place
      end-to-end via the validate path.
- [ ] **P3.3** If unsupported: at minimum, stop slicing `kernel_out[1:1+L]` /
      `kernel_out[1+L:1+2*L]` into Python lists at runtime. The current `_tkg_kv_out`
      stash is per-token Python work.

**Acceptance:** NEFF output count goes from 49 to 1; observable drop in
`nrt_async_sema_wait` host time.

## Phase 4 — Move mask + RoPE prep inside the kernel

**Lever 4.** Smallest win but cheap if you're already in the kernel.

Current `_prep_mask` (gpt_oss_with_megakernel.py:317–326): two `.contiguous()`
materializations × two masks × per token. Tagged `TODO(perf #2b)` in the source.

Current RoPE prep (lines 287–294): `.permute(2, 0, 1).contiguous()` × 2.

Tasks:
- [ ] **P4.1** Pass `position_ids` directly; have the kernel derive the active-token
      mask via `nl.iota + nl.where` into SBUF. Drops `mask_full` and `mask_window` as
      kernel inputs.
- [ ] **P4.2** Pass un-permuted `cos_cache` / `sin_cache`; permute inside the kernel
      while loading into SBUF (one SBUF op, no HBM roundtrip).

**Acceptance:** 4 fewer compiler-inserted TRANSPOSE matmuls in the device profile
(currently ~0.24 ms of "no-source-location" transpose flops); two fewer NEFF inputs.

## Diagnostics + checkpoints

After every phase:
- [ ] `python main.py --model gpt_oss --mode validate --enable-nki` — accuracy must pass.
- [ ] `python main.py --model gpt_oss --mode evaluate_single --enable-nki --benchmark`
      — record TKG p50/p99 and e2e p50 in the table below.
- [ ] Re-profile bk=640 TKG with `/neuron-nki-profile-querying`. Update bounds
      analysis report. Compare on-device total_time vs. host TKG p50; the gap is the
      remaining host-overhead.

### Score tracking

| Phase | Device total_time | Host TKG p50 | Host TKG p99 | e2e p50 | Δ vs baseline | Notes |
|---|---:|---:|---:|---:|---:|---|
| Baseline (XLA) | 5.774 ms | 5.463 ms | 5.743 ms | 3818.7 ms | (reference) | from `benchmark_report_base.json` |
| MK (current)   | 5.234 ms | 5.926 ms | 6.492 ms | 4116.4 ms | +297.7 ms | from `benchmark_report_mega.json` |
| MK + Phase 1   | 5.234 ms (unchanged) | 5.879 ms | 6.401 ms | 3989.9 ms | +171.2 ms | `evaluate_all --skip-compile` prompt 1; aggregate across 5 prompts: TKG p50 5.869 ms, p99 6.256 ms; Total Score 48.99 |
| MK + Phase 2   | — | — | — | — | — | TBD |
| MK + Phase 3   | — | — | — | — | — | TBD |
| MK + Phase 4   | — | — | — | — | — | TBD |

## Not on the roadmap (intentionally)

These were considered and rejected:

- **async-mode.** `NeuronConfig.async_mode` defaults to `False` and neither wrapper
  overrides it. Both runs are sync. Enabling async would shift both runs equally and
  not close the per-call host-time gap — different problem.
- **Reduce HBM bytes / quantize weights.** Both kernels read 1 GB / token and the
  kernel is DMA-bound at 55 %. MXFP4 (already pinned off for v1) and other weight
  quantization is its own Phase 5+ work and dwarfs the host-overhead win. Track in a
  separate doc when the host overhead is gone.
- **Bigger batch.** Out of scope for single-stream latency.
- **CTE optimization.** Megakernel CTE is already 4 ms *faster* than baseline
  (96.6 vs 100.7 ms p50). CTE runs once per request and is not the limiter.

## References

- `bk640_xla_vs_megakernel_comparison.md` — the analysis that motivated this roadmap.
- `bk640_baseline_xla_profile_analysis.md` — XLA baseline device profile.
- `bk640_megakernel_profile_analysis.md` — megakernel device profile.
- `bk2_profile_analysis.md` — prior-run megakernel bk=640 analysis (TP=8 effective).
- `transformer_gpt_oss.py` — the megakernel itself; orchestration is 15.6 µs / token, unchanged.
- `gpt_oss_with_megakernel.py` — the wrapper this roadmap modifies.
- `CLAUDE.md` (repo root, "Multi-Layer Fused TKG Megakernel") — notes the per-layer
  forward override is the intended end-state.
