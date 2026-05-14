# gpt-oss megakernel study — context, batch, and SBUF pressure

**Branch:** `debug_gpt_oss` · **Date:** 2026-05-14 · **Model:** gpt-oss-20B
**Hardware:** trn3pd98.3xlarge (1 device, 8 LNC=1 cores or 4 LNC=2 cores, 144 GB device memory)
**Both modes:** TP=8 · LNC=1 · bf16 · single bucket = seq_len · prompt = 7 tokens · default sampling.

Each cell: 1 warmup + 10 timed iterations via NxDI's `Benchmark`. Cells named `_mega` use the megakernel (committed in `d6c1eb4`); cells named `_xla` use a **no-NKI XLA fallback path** (see "What 'XLA' means here" below for why).

## What "XLA" means here — no NKI kernels at all

The `_xla` cells in this study run with **all transformer-body NKI kernels disabled**. This was verified two ways:

1. The driver explicitly passes (for `mode == "xla"` only):
   - `moe_fused_nki_kernel_enabled=False`
   - `router_topk_nki_kernel_enabled=False`
   - `expert_mlp_nki_kernel_enabled=False`
   - `shared_mlp_nki_kernel_enabled=False`
   - `blockwise_matmul_config.use_torch_block_wise=True` (stripped by `GptOssBaselineNeuronConfig` for gpt-oss; the four NKI flags above are what actually take effect)
2. The saved `neuron_config.json` for `b1_s640_xla` confirms every NKI/kernel flag resolves to `False`:

| flag | resolved value | source |
|---|:-:|---|
| `attn_block_tkg_nki_kernel_enabled` | False | NxDI default |
| `attn_tkg_nki_kernel_enabled` | False | NxDI default |
| `attn_tkg_builtin_kernel_enabled` | False | NxDI default |
| `attn_block_cte_nki_kernel_enabled` | False | NxDI default |
| `attn_kernel_enabled` | None (off) | NxDI default |
| `qkv_kernel_enabled` | False | NxDI default |
| `qkv_nki_kernel_enabled` | False | NxDI default |
| `qkv_cte_nki_kernel_fuse_rope` | False | NxDI default |
| `mlp_kernel_enabled` | False | NxDI default |
| `mlp_tkg_nki_kernel_enabled` | False | NxDI default |
| `quantized_mlp_kernel_enabled` | False | NxDI default |
| `out_proj_kernel_enabled` | False | NxDI default |
| `rmsnorm_quantize_kernel_enabled` | False | NxDI default |
| `moe_fused_nki_kernel_enabled` | False | **driver explicit** |
| `router_topk_nki_kernel_enabled` | False | **driver explicit** |
| `expert_mlp_nki_kernel_enabled` | False | **driver explicit** |
| `shared_mlp_nki_kernel_enabled` | False | **driver explicit** |
| `use_index_calc_kernel` | False | NxDI default |
| `kv_cache_update_with_kernel` | False | NxDI default |
| `top_k_kernel_enabled` (sampling) | False | **driver explicit** |
| `disable_argmax_kernel` | False (i.e. argmax kernel not disabled) | NxDI default — argmax path not hit since `do_sample=True, top_k=20` |

So the `_xla` baseline runs the transformer entirely in HLO/torch (no NKI). NxDI's `GptOssNeuronConfig` does not auto-enable any NKI kernel — every NKI flag is off-by-default in `NeuronConfig.__init__`, and `GptOssBaselineNeuronConfig` (this repo's XLA subclass) does not flip any of them on. The four explicit `False` settings in the driver only matter to defend against future NxDI changes that might flip `moe_fused_nki_kernel_enabled` etc. to `True`-by-default.

**Implication for headline numbers:** mega vs `_xla` is therefore **mega vs no-NKI baseline**, not mega vs production XLA. A real "library baseline" on a ≥16-core trn3 instance would enable NxDI's `attn_block_tkg_nki_kernel_enabled`, `moe_fused_nki_kernel_enabled`, etc. (which hardcode LNC=2 and don't fit here at TP=8 LNC=1). On that hardware the mega-vs-library gap would shrink — the `_xla` numbers here are an **upper bound on mega's win**.

## Headline matrix (complete)

### Seq sweep at batch_size=1 — mega wins everywhere, gap compresses with context
| seq_len | xla TKG p50 | **mega TKG p50** | Δ p50 | xla TKG p99 | mega TKG p99 | xla tok/s | **mega tok/s** | Δ throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 640  | 8.62 ms | **5.88 ms** | **−31.8%** | 19.4 ms | 6.4 ms  | 110.9 | **170.4** | **+53.7%** |
| 2048 | 5.50 ms | **4.92 ms** | **−10.5%** | 17.4 ms | 15.5 ms | 139.4 | **156.5** | **+12.3%** |
| 4096 | 5.54 ms | **4.85 ms** | **−12.4%** | 8.5 ms  | 10.5 ms | 168.2 | **186.5** | **+10.9%** |
| 8192 | 5.78 ms | **4.98 ms** | **−13.8%** | 6.2 ms  | 8.4 ms  | 173.4 | **195.8** | **+12.9%** |

### Batch sweep at seq_len=640 — mega wins at bs=1, ties at bs=4, **LOSES at bs=8**
| batch | **xla TKG p50** | mega TKG p50 | Δ p50 | xla TKG p99 | mega TKG p99 | xla agg tok/s | mega agg tok/s | Δ throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.62 ms | **5.88 ms** | **−31.8%** | 19.4 ms | 6.4 ms | 110.9 | **170.4** | **+53.7%** |
| 4 | **10.44 ms** | 10.53 ms | +0.8% (tied) | 13.6 ms | 11.1 ms | 378.3 | 380.5 | +0.6% |
| 8 | **15.53 ms** | 17.65 ms | **+13.7% (mega SLOWER)** | 16.2 ms | 20.8 ms | **515.8** | 450.9 | **−12.6%** |
| 16 | _OOM during compile_ | _SBUF=596%, killed mid-compile_ | — | — | — | — | — | — |

### SBUF demand for `selective_expert_moe_tkg` (200 KB partition)
| cell | peak SBUF | status |
|---|---:|---|
| b1 (any seq_len ≥ 640) | **55%** (113 KB) | fits, room for prefetch |
| b4, s=640 | **163%** (335 KB) | **spill** — exceeds partition |
| b8, s=640 | **307%** (630 KB) | heavy spill |
| b16, s=640 | **596%** (1.22 MB) | extreme spill, likely unbenchable |

SBUF demand is **invariant in seq_len** (the MoE kernel handles only K=4 active expert-token routings per layer — KV cache lives in HBM) and **linear in batch_size**.

## Headline findings

### F1 — At bs=1, megakernel is 32% faster than XLA-fallback with 3× tighter p99 (s=640)
- mega TKG p50 5.88 ms vs XLA 8.62 ms · throughput 170 vs 111 tok/s (+54%)
- mega p99 6.4 ms vs XLA 19.4 ms (XLA p99/p50 = 2.3× heavy tail; mega = 1.09×)
- mega's edge comes from collapsing the 24 per-layer Python/HLO dispatches into one device-side call

### F2 — Megakernel TKG p50 is FLAT across seq_len at bs=1 (4.85–5.88 ms across 640→8192)
The surprise. Per-token cost does NOT grow as KV cache grows. **Long-context interactive decode does not degrade per-token latency** — at s=8192 mega still hits 5 ms/token (200 tok/s).

Both mega *and* XLA plateau on per-call TKG latency past s=2048 — at small seq the per-layer overhead dominates and shrinks with longer seq (more amortization within a call). The profile analysis (`bk2_profile_analysis.md`) shows attention is only ~0.45 ms of 5.88 ms wall at s=640; even 13× attention scaling would not push the mega past 5–10 ms.

### F3 — Megakernel CROSSES OVER at bs ≥ 4 due to SBUF spill
This is the headline negative finding:

| batch | mega TKG p50 | XLA TKG p50 | mega vs XLA |
|---:|---:|---:|---:|
| 1 | **5.88 ms** | 8.62 ms | mega **wins −32%** |
| 4 | 10.53 ms | **10.44 ms** | basically **tied** |
| 8 | 17.65 ms | **15.53 ms** | XLA **wins, mega +14% slower** |

The d6c1eb4 megakernel optimization (hoisted expert-weight DMA + 2-expert prefetch ring) sized the SBUF working set for K=4 active tokens at bs=1. SBUF demand scales with `B × K`: 55% (bs=1) → 163% (bs=4) → 307% (bs=8) → 596% (bs=16). Past 100% the compiler must spill the kernel's working set to HBM, paying DMA round-trips. **At bs=8 the spill cost is bad enough that the degraded XLA-torch path beats the megakernel.**

The megakernel as committed is a **bs=1 single-user / single-stream / agentic-decode** optimization — useful for interactive workloads, harmful for batched serving.

### F4 — Hardware fit forced a no-NKI XLA baseline on this 8-core instance
trn3pd98.3xlarge has 8 LNC=1 cores. TP=8 LNC=2 = 16 cores → impossible.

The library's optimized MoE NKI kernels (`moe_block_tkg`, `router_topk`, `expert_mlp`, etc.) **hardcode LNC=2** via `kernel_assert(dims.n_prgs == 2)`, so they can't run at TP=8 LNC=1. To make XLA runnable at all on this hardware we had to disable them. Combined with NxDI's default-off settings for every other NKI kernel (attention block / QKV / MLP / RMSNorm — see the table in the header), the resulting `_xla` cells use **no NKI kernels at all** (full flag table in the header).

→ The `_xla` baseline is pure HLO/torch — **not** the production XLA path you'd see on a ≥16-core trn3 instance with all NKI kernels enabled. **The mega vs `_xla` delta overstates mega's win vs production XLA on its native hardware.** Despite this handicap, the no-NKI XLA is still faster than mega at bs=8.

For a fair production comparison, repeat this study on a trn3 instance with ≥16 LNC=1 cores so XLA can run TP=8 LNC=2 with all NKI kernels enabled. On this 8-core box, the megakernel is the only path that uses TP=8 sharding at all — but only at bs=1 does it benefit from that.

### F5 — Compile time is comparable; mega has no compile-time penalty
| cell | "Finished building model" | notes |
|---|---:|---|
| b1, s640 (xla) | 290 s | solo, all NKI MoE off |
| b1, s2048 (mega) | 173 s | solo |
| b1, s4096 (mega) | 381 s | 5 parallel compiles |
| b1, s8192 (mega) | 596 s | 5 parallel compiles |
| b1, s8192 (xla) | 571 s | 5 parallel compiles |
| b4, s640 (xla) | 185 s | sequential (after lock clearing) |
| b8, s640 (xla) | 364 s | sequential |

### F6 — At long context, mega's p99 reverses — XLA has tighter tail at s=8192
- s=640: mega p99=6.4 ms, XLA p99=19.4 ms — mega 3× tighter
- s=2048: mega p99=15.5 ms, XLA p99=17.4 ms — mega slightly tighter
- s=4096: mega p99=10.5 ms, XLA p99=8.5 ms — XLA tighter
- s=8192: mega p99=8.4 ms, XLA p99=6.2 ms — XLA tighter

Hypothesis: at long context mega's per-iter wall is dominated by HBM DMA-bound MoE work (natural fluctuation from queue contention), while XLA's per-layer dispatch overhead becomes a relatively fixed cost. Mega's "single big kernel" gives less opportunity for the scheduler to absorb DMA jitter.

## Pros (where mega shines)
1. **Single-user interactive decode, any context up to 8K**: −10–32% TKG p50 vs XLA-fallback. Throughput +11–54%.
2. **Long-context decode at bs=1**: TKG p50 stays FLAT across 640→8192. Time per token does NOT degrade as conversation grows. Big win for chat / agentic workloads.
3. **Tight p50 tail at short context (s≤2048)**: 3× tighter than XLA-fallback at s=640.
4. **Engine utilization is high**: 95.4% active at bs=1/s=640 per profile analysis. Mostly DMA-serialization-limited (not bandwidth-limited; MBU 14.6%) — room to push further with deeper prefetch.
5. **Compile cost competitive** (~3 min solo, not noticeably worse than XLA).

## Cons (where mega does not shine)
1. **Batched serving (bs ≥ 4) is broken**: SBUF spill begins at bs=4 (163%), is severe at bs=8 (307%). **At bs=8 mega is slower than the degraded torch XLA-fallback.** As-committed, this kernel is *unusable* for continuous-batching production serving.
2. **Hardcoded TP=8 / LNC=1** in `gpt_oss_with_megakernel.py:55-57`. Won't run on production LNC=2 hardware without redesigning the attention block (kv_heads=1 constraint forces LNC=1). Leaves cores idle on larger trn3 instances.
3. **TTFT / prefill not addressed**: CTE uses NxDI stock path. mega CTE p50 matches XLA (96 ms at s=640 → 482 ms at s=8192). For long-prompt + short-decode, no benefit.
4. **bf16 only**: no mxfp4 / quantized inference path (`is_mxfp4_compute=False`).
5. **fused_qkv hard requirement**: not all checkpoints store this way.
6. **Sliding-window attention DP=1**: even though gpt-oss alternates sliding/full layers, sliding paths don't use additional data parallelism.
7. **p99 reverses at long context**: at s=8192 mega has WORSE p99 (8.4 ms) than XLA (6.2 ms). Single-big-kernel pattern absorbs less DMA jitter than per-layer dispatching does at long seq.

## SBUF / on-chip memory deep dive
- `selective_expert_moe_tkg` compiler-allocated partition: **200 KB**
- Demand at bs=1: 113 KB (55%) — the d6c1eb4 hoisted-weight + 2-expert-prefetch ring working set
- Demand scales **linearly with batch_size × K=4 active tokens**:
  - bs=1 → 113 KB (55%)
  - bs=4 → 335 KB (163%) — spill
  - bs=8 → 630 KB (307%) — heavy spill
  - bs=16 → 1.22 MB (596%) — extreme spill
- Demand is **flat in seq_len** at bs=1 — confirmed at s=640, s=2048, s=4096, s=8192, s=16384 (partial compile)
- Runtime SBUF peak across ALL partitions: **14.7%** at s=640 bs=1 (from profile analysis). The whole multi-layer megakernel uses very little SBUF — only the inner MoE selective-expert partition is locally heavy. Room exists for larger working sets.

## Profile breakdown (analysis agent, bs=1, s=640 mega)
- Tensor engine (LDWEIGHTS + MATMUL): **28.9%** wall
- DMA active (all queues): **49.6%** wall
- Vector / Scalar / GpSimd / CC-core: 16.7% / 10.7% / 6.6% / 0.2%
- Idle: **4.6%**
- HBM read per TKG iter: ~1006 MB
- Effective bandwidth: 172 GB/s (MBU 14.6% vs peak 1178 GB/s)
- **DMA serialization-limited, not bandwidth-limited.** Weight streaming waits gate the schedule.

Hot instructions (μs at s=640 mega):
| group | μs | source |
|---|---:|---|
| MoE gate_up matmul + ldweights | 3525 | `mlp_tkg_gate_up_projection.py:479` |
| MoE down matmul + ldweights | 2664 | `mlp_tkg_down_projection.py:326` |
| MoE gate_up DMA | 1150 | `mlp_tkg_gate_up_projection.py:470` |
| Attention O-projection | 884 | `output_projection_tkg.py:951` |
| Attention QKV | 342 | `qkv_tkg.py:1507` |

## Recommendations
1. **For interactive single-user decode at any context up to 8K**: deploy mega. It dominates XLA-fallback across the entire bs=1 sweep.
2. **For batched serving (bs > 1)**: do NOT deploy mega. SBUF spill makes it slower than even the degraded torch XLA at bs=8. The MoE kernel needs batch-aware SBUF tiling — either chunk K-dim per iteration or relax the hoist.
3. **For production LNC=2 hardware**: mega cannot run as-is. Redesigning the attention block to not hardcode `kv_heads=1` would unlock LNC=2. Conversely, XLA's NKI MoE kernels hardcode LNC=2 in the *opposite* direction. A 16-core trn3 instance would allow direct mega-vs-XLA comparison on equal hardware footing.
4. **Cross-over point**: at exactly bs=4 mega and XLA-fallback are tied (10.44 vs 10.53 ms). For workloads doing bs<4 decode, mega wins; bs≥4 needs XLA. This is the deployment decision boundary on this hardware.

## Addendum (2026-05-14) — isolating the SBUF spill source via `NKI_MOE_LEGACY_WEIGHT_LOAD`

To test whether the batch-scaling SBUF demand is owned by the d6c1eb4 hoist or by structural per-T state in the selective kernel, we added a module-level env-var toggle across the three vendored files touched by d6c1eb4:

- `nki_kernels/moe/selective_expert_impl.py` — `_MOE_LEGACY_WEIGHT_LOAD` reverts the exp 7 cross-expert prefetch ring; `use_prefetch_ring` becomes False.
- `nki_kernels/moe/mlp_tkg_gate_up_projection.py` — same env var reverts the exp 3 in-function hoisted gate/up tile (`use_hoisted_gate_up_load` becomes False); falls through to the per-HTile ring-buffer DMA path.
- `nki_kernels/moe/mlp_tkg_down_projection.py` — same env var reverts the exp 5 hoisted down tile; falls through to the per-HTile path.

Three legacy cells compiled at `seq=640` and benched alongside the hoist baseline:

### SBUF partition demand for `selective_expert_moe_tkg`

| batch | hoist (default) | **legacy (env=1)** |
|---:|---:|---:|
| 1 | 55% (113 KB) | **19% (40 KB)** |
| 4 | 163% (335 KB) — spill | **19% (40 KB)** |
| 8 | 307% (630 KB) — spill | **19% (40 KB)** |
| 16 | 596% (1.22 MB) — spill | (not run) |

**Legacy stays flat at ~40 KB regardless of batch.** The d6c1eb4 hoist owns 100% of the batch-scaling SBUF growth — every extra byte beyond 40 KB comes from the prefetch-ring slots + hoisted gate/up/down tiles, all of which are sized by `B × K × weight_chunk` in the hoist path and by `HTile` in the legacy path. The original report's "linear in batch" SBUF finding is a property of the hoist's allocation pattern, not of selective-expert TKG.

### Latency comparison (s=640, TKG p50, 10 iters)

| batch | hoist p50 | **legacy p50** | Δ | hoist tps | legacy tps |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.88 ms | **5.83 ms** | -0.05 ms (tied) | 170 | 172 |
| 4 | **10.53 ms** | 13.53 ms | +3.0 ms (hoist wins -22%) | **380** | 296 |
| 8 | **17.65 ms** | 23.41 ms | +5.8 ms (hoist wins -25%) | **451** | 342 |

### Findings

1. **The hoist owns the SBUF spill, full stop.** Verified across bs ∈ {1, 4, 8}. Selective-expert TKG itself has bounded SBUF footprint independent of batch.
2. **But spilling beats not-spilling at moderate batch.** Even at bs=8 with the hoist reporting 307% partition demand, the hoist is 25% faster than the legacy per-HTile-DMA path. The compiler handles the spill (presumably by HBM-resident expert weight chunks with overlapped reload) more efficiently than the legacy path's serialized per-HTile DMA pipeline. So reverting the hoist is the *wrong* fix for the batched-serving gap.
3. **The hoist's bs=1 win is essentially zero in the current compiler.** 5.88 vs 5.83 ms = 1% delta. The d6c1eb4 commit's -17% TKG p50 was measured against the pre-d6c1eb4 codebase; the per-HTile path has apparently improved enough since then that the hoist no longer provides the headline win it once did at bs=1. (It still helps a lot at bs=4/8 — see #2 — just not at bs=1.)
4. **The real bottleneck is compute reuse, not SBUF sizing.** Both hoist and legacy compute K matmuls per token at `[1, H] @ [H, I]` shape — tensor engine partition utilization 1/128. The XLA torch-blockwise baseline (no NKI) computes `[block_size, H] @ [H, I]` — full partition utilization, redundant compute on padding included. At bs=8 XLA-fallback beats both hoist (15.5 vs 17.6 ms) and legacy (15.5 vs 23.4 ms). See the "Why XLA wins at large batch" section below.

### Why XLA torch-blockwise beats both NKI selective variants at bs=8

The selective kernel (with or without our hoist) has the same loop structure: `for t in range(T): for k in range(K): [1, H] @ [H, I]`. Each weight load drives **one row** of matmul — the tensor engine's 128-row partition runs at 1/128 utilization. With K=4, E=32 active expert-token pairs at bs=8, that's 32 separate 1-row matmuls — 32 expert weight loads × 1 row of compute each.

The XLA torch-blockwise path uses `BlockwiseMatmulConfig.block_size=512` (visible in our saved `neuron_config.json`). It groups tokens routed to the same expert into a block, pads to block_size, and does **one** matmul of shape `[block_size, H] @ [H, I]`. At bs=8 × K=4 = 32 active pairs spread over 32 experts (~uniform routing), each active expert receives ~1 token but the matmul still runs at `[512, H] @ [H, I]` — 512× more rows of compute per weight load, on the same partition-full tensor engine cycle. Most of those 512 rows are zero-padded waste, but the tensor engine doesn't care — partition utilization is 100% either way.

Net: **selective trades compute for memory bandwidth (load weights K×T times, compute K×T × 1 rows); blockwise trades memory bandwidth for compute (load weights E times, compute E × block_size rows).** At bs=1 selective wins because total rows = K = 4 ≪ block_size = 512. At bs=8 blockwise pulls ahead because the wasted-compute factor `block_size / (B×K/E)` shrinks while the loaded-bytes-per-row factor flips in favor of blockwise.

The fix isn't to revert the hoist (that hurts at bs≥4 as Finding 2 shows) and isn't to add more hoisting (SBUF spills harder). The fix is to **change the loop shape**: process tokens grouped by expert with padding-amortized matmuls (the structure XLA uses), so each weight load drives a 128- or 512-row matmul instead of a 1-row matmul. This is the same structural argument the `all_expert_mx_impl.py:169-171` docstring is making when it says gpt-oss crosses over to all-expert at T=128-256: dense-over-experts dominates sparse-over-tokens once you have enough tokens to amortize.

### Artifacts (addendum)

- `b{1,4,8}_s640_mega_legacy.json` — bench reports
- `/home/ubuntu/models/study/b{1,4,8}_s640_mega_legacy/` — compiled NEFFs
- `/home/ubuntu/models/study/logs/b{1,4,8}_s640_mega_legacy{,.bench}.log` — compile + bench logs
- Toggle: `NKI_MOE_LEGACY_WEIGHT_LOAD=1` on the compile + bench environment (NEFF recompile required to switch)

## Open / unpursued
- **Profile capture at s=8192 mega**: confirm the seq-flat F2 finding mechanism (attention partial-K read pattern? bucket-padded attention reads avoiding full quadratic compute?).
- **s=16384 mega**: compile killed mid-process for memory reasons; bench data missing. SBUF demand expected to stay at 55% based on F2 extrapolation.
- **bs=16 mega**: compile not finished (596% SBUF). Bench would likely be very slow or fail.

## Artifacts
- `study_driver.py` — single-cell compile + benchmark
- `bench_all.sh` — find-compiled-then-bench script
- `aggregate.py` — auto-build the result tables from JSON reports
- `extract_sbm.py` — pull per-kernel SBUF stats from compile logs
- `extract_compile_time.py` — pull compile durations
- Per-cell JSONs: `b<B>_s<S>_<mega|xla>.json` in this dir
- Compiled artifacts: `/home/ubuntu/models/study/b<B>_s<S>_<mode>/`
- Compile logs: `/home/ubuntu/models/study/logs/b<B>_s<S>_<mode>.log`
- Bench logs: `/home/ubuntu/models/study/logs/b<B>_s<S>_<mode>.bench.log`
- Run matrix script: `run_matrix.sh` (sequential, kept as escape hatch — actually used parallel compile-only + sequential bench)
