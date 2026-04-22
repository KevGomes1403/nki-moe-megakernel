# Submission narrative — trn3 MoE TKG optimization (Phase 3 closeout, 2026-04-22)

## Core contribution

**int8-reinterpret HWDGE trick** — a novel trn3 optimization primitive for MXFP4-quantized MoE TKG.

The moving tile in the Qwen3 MoE TKG path is `[128P, T]` with `T=1`. `nc_matmul_mx` (MXFP_x4) requires a moving tile of `[128P, 512F]` and would waste 511/512 of its compute bandwidth at `T=1`, so the standard trn3 quantization path is off the table. HWDGE (hardware descriptor-gather engine) plus int8 reinterpretation lets us fold the scale multiplication into the DMA/compute pipeline without paying the `quantize_mx` bandwidth cost on the moving side, keeping the matmul on `nc_matmul` with fp8 stationary + bf16 moving while still consuming offline-MXFP4-packed weights. The net effect is a DMA-bound kernel that amortizes weight-load bytes down the scale axis rather than the dtype axis.

This primitive is not Qwen3-specific. It applies to any trn3 MoE TKG kernel where the moving tile is too narrow to justify on-device MXFP quantization but the weights can be pre-quantized offline. Documented as the Round 1 winner and the durable artifact of this work.

## Verified win

**v15c MoE kernel**: **72.88 μs**, **−17.2%** vs `v14a` baseline (**88.06 μs**), 1 layer, TP=4, LNC=2 shard, TKG path.

- **Correctness**: bit-exact against `v14a` (`max_diff = 0`) with both kernels independently validated as non-zero against a BF16 reference (`probe_v14a_v15c_zero_check.py`: absmean ~11.6, range [−49, 55]). Bit-exactness-as-a-contract-check is safe here because the reference has been independently validated — not a false positive (see L4 in `integration_findings.md`).
- **Harness**: `bench_v15c_trn3.py` + `benchmark.py` (`wrap_benchmark`, warmup=5, iters=50), `NEURON_PLATFORM_TARGET_OVERRIDE=trn3`, profile metrics captured in the Round 1 synthesis table.
- **Reproducibility**: `kernels/moe_fused_tkg/quantized/v15c.py` + `bench_v15c_trn3.py` + `test_v15c.py` committed and preserved.

Round 2 explored three further plans (v15e, v15f, v15g) against v15c. None produced measurable improvement; all kept as documented regressions under the <2% no-revert rule. **v15c remains the Phase 3 integration target.**

## Shipping stack — integration fixes that make v15c shippable on trn3

Getting v15c to compile end-to-end inside `qwen_complete.py` required three surgical fixes against NxDI/NKI behaviors. Each is documented with code-level patterns in `docs/integration_findings.md`.

1. **(b1) CTE shard-on-block config** — route `qwen_complete` CTE through `forward_blockwise` with `block_size=256`, `use_shard_on_block_dynamic_while=True`, and `block_sharding_strategy="PING_PONG"`. Without this, Phase 0's `block_size=8192` workaround forced the full `forward_all_experts` path and emitted 128 experts × 48 layers = 6144 expert sub-graphs, blowing past the protobuf 2 GB HLO serialization limit. All 3 CTE HLO buckets (ctx=128/256/640) compile cleanly under (b1).
2. **(v.1) Parameter promotion** — `register_buffer(..., persistent=True)` → `nn.Parameter(..., requires_grad=False)` for the three TKG-only quantized MoE buffers (`gate_up_packed_w_tkg`, `down_w_tkg`, `down_scales_tkg`). `neuronx_distributed/trace/model_builder.py:806` iterates `named_parameters()` only — not `named_buffers()` — so register_buffer tensors get captured as Python closures and materialize as `constant` HLO ops (often 4× upcast int8→fp32). Qwen3-30B's three TKG buffers × 48 layers produced 30.25 GB of `constant` HLO bytes, exceeding protobuf 2 GB. (v.1) collapses constants to <2 MB total; `ByteSize()` succeeds. Documented as **landmine L5**.
3. **(H1) in-place dtype hook** — `_apply_protect_quant` mutates `p.data = p.data.to(dtype)` instead of allocating a fresh `nn.Parameter(...)` on dtype mismatch during NxDI's cast ping-pong. Preserves `Parameter` identity in `tkg._parameters[...]` so (v.1) holds through repeated `model.to(...)` calls, and removes a minor memory leak in the cast path.

With (b1)+(v.1)+(H1) all installed, **qwen_complete compiles end-to-end on trn3 for the first time**: 6/6 NEFFs (3 CTE + 3 TKG) compile cleanly and the NEFF cache is preserved at `/var/tmp/neuron-compile-cache` (305 MB) + `~/qwen-30b-a3b/traced_model/model.pt` (85 MB).

## NKI landmines catalog — community-value deliverable

Seven silent-failure patterns uncovered during this integration. Each fails without a compile error and produces wrong or hung output. Full postmortems in `docs/integration_findings.md`.

| # | Landmine | Symptom | Fix |
|---|---|---|---|
| **L1** | `nisa.dma_copy(dst=reshape(...), ...)` | Returned tensor is zeros; DMA wrote to a temporary | Allocate HBM in target shape; reshape on return only |
| **L2** | `nisa.exponential` with `reduce_cmd=idle` | Downstream VectorE ops inherit corrupt accumulator; output zeros | `reduce_cmd=reset_reduce` (unverified) or stay on `nisa.activation(op=nl.exp)` |
| **L3** | `rtol=1e-2, atol=2e-2` when `max(|ref|) < atol` | Zero-output kernels pass `assert_allclose` | Triple-check: zero-check + range-check + allclose |
| **L4** | Bit-exact `max_diff=0` against unvalidated reference | Both kernels can be wrong-in-the-same-way | Spot-check reference against PyTorch fp32 at least once |
| **L5** | `register_buffer` tensors inlined as HLO constants | TKG HLO protobuf 2 GB overflow; 4× dtype upcast | Use `nn.Parameter(..., requires_grad=False)` instead |
| **L6** | `ModelBuilder.shard_checkpoint` RAM accumulation | SIGKILL at 127 GB during post-compile sharding | Streaming monkey-patch (installed in qwen_complete.py; superseded by L7) |
| **L7** | NxDI load-time dtype cast doubles tensors | SIGKILL before shard_weights runs | No runtime hook; requires NxDI change, offline checkpoint prep, or larger-DRAM host |

The catalog is as valuable as the kernel win. L5 in particular is a general NxDI gotcha that affects any tracer-reliant model with non-parameter model-scoped state; L6+L7 generalize to any Qwen3-30B-class deployment on a 124 GB-DRAM trn3 instance.

## Known limitation

**End-to-end TTFT / tok-s / NKI_FLOP_Ratio was not captured.** This is a host-memory infrastructure constraint, not a kernel problem.

- The `/tmp/phase0_e2e.log` run (2026-04-22 05:41 UTC) SIGKILL'd at t=1167.8s on prompt 0 (`returncode=-9`) during NxDI's load-time dtype cast pass (L7), before `shard_weights` was reached. The L6 streaming-shard patch was active (`[L6] NxDI streaming shard_checkpoint patch ACTIVE` at log line 22) but not load-bearing — the peak lived one stage upstream.
- DRAM ceiling on trn3.3xlarge (124 GB) + 32 GB `/swapfile` is insufficient for the Qwen3-30B bf16↔fp32 cast pass. Further headroom would require either invasive offline fp32-scales-in-checkpoint changes or a larger-DRAM instance.
- The v15c −17.2% **kernel-level win is verified and durable** — bit-exact correctness plus bench measurements. The **e2e translation is estimated, not measured**.

### Estimated e2e (placeholder, from isolated bench components)

```
implied_per_layer_us = v15c(72.88) + v13bc_sbm_tiled(53.71) + AR_est(12) + gap_est(25)
                     = 163.59 μs
implied_e2e_ms      = 48 × 163.59 μs + post_transformer_est
                    ≈ 7.85 ms + post_transformer_est
```

`AR_est` and `gap_est` are load-bearing placeholders (typical trn3 AR at H=2048 TP=4 and prior transformer TKG inter-op glue budgets respectively) — neither is measured at this model/config. See `kernels/moe_fused_tkg/quantized/OPTIMIZATION_LOG_TRN3.md` § "Phase 3 — ESTIMATED e2e" for the full component table and disclaimers. **Do not cite these numbers as submission results.**

## Reproducibility summary

- **Kernels**: `kernels/moe_fused_tkg/quantized/v15c.py` + `v14a.py` (reference) + `bench_v15c_trn3.py` + `test_v15c.py` + `probe_v14a_v15c_zero_check.py`.
- **Integration**: `qwen_complete.py` with (b1) + (v.1) + (H1) + L6 monkey-patch (gated by `NKI_STREAMING_SHARD_PATCH=1`).
- **Logs**: `kernels/moe_fused_tkg/quantized/OPTIMIZATION_LOG_TRN3.md` (full Round 1-3 history + Phase 3 CLOSEOUT) + `docs/integration_findings.md` (L1-L7 landmines + findings).
- **Preserved compile artifacts**: `/var/tmp/neuron-compile-cache` (305 MB) + `~/qwen-30b-a3b/traced_model/model.pt` (85 MB). A future session on a larger-DRAM host can `--skip-compile` and go directly to load/run.
