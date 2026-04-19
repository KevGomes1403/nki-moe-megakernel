# v14a_kv_norm — Optimization Log

Profile source: `output/i-00fc9aa9608c2474c_pid_281025/1776300389037205267/` (token_generation_model, TP=4, trn3.3xlarge, Qwen3-MoE 48 layers, one decode step).
Date of analysis: 2026-04-16.

## Baseline metrics

- **E2E token-gen step:** 7.63 ms (NC4) / 7.65 ms (NC5), 0 errors, 100 collectives.
- **Per-layer steady state:** ~143 µs (layers 1–47), layer 0 = 154 µs (pipeline prologue cost).
- **Attention AR (AR1) ≈ 80 µs, MoE AR (AR2) ≈ 63 µs per layer** — attention half is ~25% heavier.
- **MFU 0.5%, arithmetic intensity 1.32** — heavily memory-bound (expected for TKG decode).
- **HBM read 2.68 GB/step, DMA active 64% of runtime.** Bottleneck is weight streaming, not compute.

## v14a engine mix (79,271 instructions / decode step, across 48 layers)

| Engine | Count | Active dur | Stall (evt_wait) |
|---|---|---|---|
| TensorMatrix (matmul) | 25,632 | 5.1 µs | 0.1 µs |
| Tensor (LDWEIGHTS) | 26,120 | 2.5 µs | 22.2 µs |
| Vector | 20,606 | 3.4 µs | 28.5 µs |
| **Sync (DMA triggers)** | 4,507 | 1.8 µs | **62.4 µs** |
| GpSimd | 677 | 0.1 µs | 15.0 µs |
| Scalar | 1,729 | 0.6 µs | 4.9 µs |

> `duration` is engine-local and parallelized — use only for relative weighting. `evt_wait_time` (stall) is the wall-clock-visible signal. **Sync-engine stall of 62 µs = DMA triggers waiting on semaphores**, i.e. consumers starved on weight / KV DMAs.

## Top stall hot-lines in v14a_kv_norm.py

| Line | Count | Stall | HBM R | Opcode | What it is |
|---|---|---|---|---|---|
| **199** | 1,246 | **17.4 µs** | 403 MB | `PSEUDO_DMA_TRIGGER` | `nisa.dma_transpose` of Wq per head (weight hoist) |
| **441** | 1,075 | 14.0 µs | 403 MB | `PSEUDO_DMA_DIRECT2D` | `nisa.dma_copy` Wo hoist (pre-flash-decode) |
| 647 | 817 | 12.3 µs | 0 | `TENSOR_TENSOR` | Pass-2 softmax `sum_acc` / `v_acc` add chain |
| 792 | 288 | 8.6 µs | 0 (12 KB W) | `ALU_OP` | V KV-cache indirect scatter |
| **366** | 24,815 | 7.0 µs | 0 | `LDWEIGHTS` | Q-projection `nc_matmul` inner |
| 504 | 545 | 6.9 µs | 15.7 MB | `PSEUDO_DMA_DIRECT2D` | K-cache tile load (per s_t) |
| 555 | 545 | 6.7 µs | 15.7 MB | `PSEUDO_DMA_DIRECT2D` | V-cache tile load (per s_t) |
| 800 | 288 | 6.3 µs | 0 (12 KB W) | `ALU_OP` | K KV-cache indirect scatter |
| 251 | 188 | 3.9 µs | 50.3 MB | `PSEUDO_DMA_TRIGGER` | (prior scope DMA — setup region) |
| 274 | 3,168 | 2.7 µs | 0 | `LDWEIGHTS` | Q RMSNorm small matmul |

## DMA size distribution (v14a only)

```
<128 B     96     tiny setup DMAs
<1 KB     480
<16 KB     96
<128 KB   960
>=128 KB 1728    ~66% of DMA traffic (healthy — bulk is thick)
```

Skinny DMAs (<512 B):
- Line 128: 96 × 8 B — scalar setup.
- **Lines 792 / 800: 48 × 256 B each** — KV-cache scatter. Intrinsic to token-by-token decode; single-slot writes per step.

## Prioritized optimization backlog

### P0 — Pre-transpose Wq in HBM at weight-prep time
- **Change:** replace `nisa.dma_transpose` at line 199 with plain `nisa.dma_copy` (`PSEUDO_DMA_DIRECT2D`). Requires storing Wq in the already-transposed layout in HBM (do the reshape once in the weight loader / checkpoint conversion, not per-step).
- **Expected gain:** ~10 µs stall reduction per layer → **~0.5 ms / step** across 48 layers.
- **Risk:** low. Layout-only change; no compute change. Must update weight loader and verify shape assumptions (Wq[q_h*d:(q_h+1)*d, 0:H] is currently `[d=128, H=2048]` contiguous 256 KB per head).
- **Validation:** re-run stall-attribution recipe, confirm line 199 stall drops to ~7 µs range (matching line 441 class).

### P1 — Double-buffer Wq/Wo across layer boundaries
- **Change:** in `qwen_fused_transformer_multilayer.py`, allocate **two** SBUF stacks for `wq_head_sb` (and optionally `wo_sbuf`) and ping-pong:
  - Layer N computes from buffer A.
  - At start of layer N attention, issue DMAs into buffer B for **layer N+1's** weights.
  - Layer N+1 computes from B, prefetches N+2 into A.
- **Why:** compiler `--enable-ccop-compute-overlap` pipelines at graph level, but inside a fused megakernel the compiler can't see across hand-written control flow. Explicit double-buffer exposes overlap.
- **Expected gain:** hides most of the remaining Wq/Wo DMA latency behind flash-decode compute. Target 143 µs → <130 µs steady-state.
- **Risk:** SBUF budget. Wq per head ≈ 256 KB × `HQ_TP_CONST`; two copies may be tight. If so, double-buffer Wq only (bigger stall).
- **Prereq:** do P0 first — removing the transpose simplifies the prefetch dependency chain.

### P2 — Consolidate Wo hoist into one wider DMA
- **Change:** line 440 loop `for head in nl.affine_range(HQ_TP_CONST): nisa.dma_copy(...)` → single `dma_copy` with a wider pattern covering all heads. Same total bytes, fewer descriptors, fewer triggers.
- **Expected gain:** small (~1–2 µs), cleanup-level.
- **Risk:** trivial.

### P3 — Move Wq/Wo hoist to top of `attn_outer`
- **Change:** reorder so both weight DMAs are issued at the very top of `attn_outer` rather than inside `q_proj` scope (line 199) and pre-flash-decode (line 441). Gives the DMA engine maximum runway.
- **Expected gain:** modest — the compiler scheduler already reorders somewhat, but source order matters for the BIR scheduler in fused megakernels.
- **Risk:** must verify SBUF lifetimes still intersect correctly across scope boundaries.

### P4 — Pass-2 softmax chain (line 647)
- **Change:** fuse `tile_sum`/`v_weighted` psum→sbuf copies with the subsequent `tensor_tensor` accumulate, or double-buffer `sum_acc` / `v_acc` across `s_t` iterations to overlap the dependency chain.
- **Expected gain:** potentially ~5–10 µs; requires careful analysis, since the chain is `nc_matmul → tensor_copy → tensor_tensor` with tight data dependency.
- **Risk:** medium — easy to break numerics.

### P5 — K/V cache tile-load batching (lines 504/555)
- **Change:** the `for s_t in NUM_TILES` loop currently issues serial ~15 KB DMAs. Investigate batching two tiles per DMA (widen pattern), or switching to a single strided `dma_transpose` over the full `S_prior × d` region loaded into a tiled SBUF.
- **Expected gain:** 5–8 µs if effective; depends on whether current K/V loads already overlap with softmax compute.
- **Risk:** medium; requires re-checking PE transpose dependencies (lines 508, 541, 545).

### Intrinsic / won't fix
- **KV scatter (792/800) — 256 B skinny DMAs.** Fundamental to token-by-token decode. The only path to reducing this is moving the scatter into a fused multilayer kernel that merges layer-N's scatter with layer-(N+1)'s KV read — scope of the megakernel plan, not v14a in isolation.
- **Line 366 `nc_matmul` Q-proj (7 µs stall, 24.8 K instrs).** Low per-instruction stall — matmul itself is fine; stall is absorbed startup cost waiting on Wq. Fixing P0/P1 dissolves this naturally.

## How to reproduce the analysis

1. Generate full JSON:
   ```bash
   neuron-profile view -n <neff> -s <ntff> \
     --output-format json --output-file /tmp/prof.json
   ```
2. Aggregate by `bir_debug_info_source_location`, filter to `v14a_kv_norm.py` — see `.claude/skills/neuron-profile/references/kernel_source_attribution.md` for copy-paste recipes (stall attribution, skinny DMAs, idle gaps, bytes-per-line).
3. Cross-check per-layer latency via CC-core TPB_TRIGGER gap analysis — see `references/layer_latency.md`.

## Change history

_Add one entry per optimization attempt below._

| Date | Change | Target line | Pre stall | Post stall | Per-layer Δ | Notes |
|---|---|---|---|---|---|---|
| 2026-04-16 | (baseline) | — | — | — | 143 µs | Initial profile analysis |

---

## Round 2 — 2026-04-17 · LNC-ceiling sweep on `v14b_kv_norm_pretransposed`

Working kernel: `v14b_kv_norm_pretransposed.py`. Baseline re-measured standalone: **device_time_us ≈ 75.9 µs** (50-iter bench, LNC=2).

Existing LNC pattern in baseline: head-shard Wq + Wo (4 of 8 heads per core) + one `sendrecv`-based all-reduce on the o-proj output. Wk, Wv, KV cache, pre-attn RMSNorm, and Q/K broadcast scratch all duplicated on both cores.

Standalone bench profile:
- VectorE 52.7% (~40 µs) — dominant
- TensorE 39.8% (~30 µs)
- DMA 47.8% (~35 µs)
- spill 163,840 B (unchanged across all variants)
- hbm_read 11,554 KiB, hbm_write 656.5 KiB

### Plans attempted (nkilib-pattern review)

After reading `nkilib/experimental/transformer/{transformer_tkg,attention_block_tkg,attention_block_tkg_sharding}.py`, the validated AWS pattern is: shard the hidden dim `H1_shard = H1 // n_prgs`, run **one** `all_reduce` per block boundary via `_sb2sb_all_reduce_gather` — not per-tensor `sendrecv` chains. Both LNC variants below applied this lens.

| Variant | File | Strategy | device_us | vs base | Outcome |
|---|---|---|---|---|---|
| v15a | `v15a_peer_split_kv.py` | Peer-split Wk/Wv + K/V cache; 14 `sendrecv`s | **100.3** | **−32%** | Regression — collectives serialize pipeline |
| v15a2 | `v15a2_h_shard_qkv.py` | nkilib-style H-shard of QKV (tiles 0–7 / 8–15); 4 `sendrecv`s | **85.4** | **−15%** | Regression — HBM-read −27% but still serialization-bound |
| v15b | `v15b_oproj_collapse.py` | Drop `col_tmp` intermediate in 16-col o-proj transpose | **74.96** | ~0% | Wash — compiler DCE'd the redundant `tensor_copy` |
| v15c | `v15c_broadcast_elim.py` | Replace 6 sites of `nc_transpose`→PSUM→transpose broadcast chains with stride-0 `.ap` inline broadcast inside `tensor_tensor` | **75.10** | **+1.1%** | Modest win; engine-time freed, DMA became critical path |

### Per-engine delta for v15c (the only net improvement)

| Metric | baseline | v15c | Δ |
|---|---|---|---|
| TensorE | 30.2 µs (39.8%) | 24.5 µs (32.6%) | **−5.7 µs** |
| VectorE | 40.0 µs (52.7%) | 35.8 µs (47.7%) | **−4.1 µs** |
| DMA active | 47.8% | **53.0%** | +5.2 pp |
| device_time | 75.9 µs | 75.1 µs | −0.8 µs |

~9.8 µs of engine time was freed but only ~0.8 µs surfaced as wall-clock — the freed cycles were already overlapped with DMA.

### v15c sites converted (broadcast `[PMAX,1] → [PMAX,GQA=8]`)

1. `qnw_gqa` — multiplied into `q_normed2`
2. `cos_gqa` — `q_normed2 * cos_gqa`
3. `sin_gqa` — `rot_q * sin_gqa`
4. `k_rope_packed` — `k_rope * q_bf16` (mixed fp32/bf16 accepted by `tensor_tensor`)
5. `v_act_packed` — `v_act * score_act_exp`
6. `mask_gqa_pre` — removed per-tile 2-transpose broadcast in the 8-iter `NUM_S_TILES` loop (largest single source of freed cycles)

AP pattern used: `X.ap(pattern=[[1, PMAX], [0, GQA]], offset=0)` inlined as the broadcast operand. Validator requires `[[part_stride, part_size], [free_stride, free_size]]` order.

`neg_max` broadcast (`[GQA,1] → [PMAX,GQA]`) left alone — that's a partition-dim broadcast, different pattern.

### Lessons (for future rounds)

1. **LNC-sharding floor is the existing head-shard + single final all-reduce.** On a kernel this small and already DMA-overlapped, every additional `sendrecv` on the hot path costs more than the HBM savings buy — confirmed across two independent shard strategies (peer-split and H-shard). Even a correctly-implemented H-shard à la nkilib regresses 15%.
2. **Engine-time wins don't convert 1:1 to device-time** once DMA becomes critical path. v15c freed ~10 µs of TensorE+VectorE; only ~1 µs came out in wall-clock because DMA now owns the schedule.
3. **Compiler already eliminates trivial intermediate copies** (`col_tmp` in o-proj). Don't spend plans on DCE-eligible rewrites.
4. **`tensor_tensor` accepts stride-0 `.ap` broadcasts across dtype mixes** (fp32 × bf16). This generalizes well — worth applying to any future `[PMAX, small]` broadcast site that is currently materialized via PSUM round-trip.

### New bottleneck after v15c

DMA (53.0% active, 11.5 MiB HBM read/core). To make further progress without serialization:
- FP8 / MXFP8 weight quantization on Wq/Wk/Wv/Wo — cuts weight DMA by 2–4× (trn3 supports 2× FP8 TFLOPS so no compute cost).
- Multi-layer fusion: reuse Wo or residual across layers in the megakernel context (see `docs/multilayer_fused_tkg_plan.md`).
- Weight ring-buffer / prefetch between layers.

**Best current variant: `v15c_broadcast_elim.py` @ 75.1 µs.**

---

## Round 3 — 2026-04-17 · v16a S-tile sequence-parallel (LNC=2)

Working kernel: `v16a_seq_parallel.py`. Goal: reduce per-core KV-cache DMA by sharding the flash-decode sequence axis across the two LNC cores, in addition to the existing head-sharding for Wq/Wo.

### Design

With `NUM_S_TILES=8` (S=640, PMAX=128):
- Core 0 owns S-tiles [0,1,2,3] and Q-heads [0,1,2,3]
- Core 1 owns S-tiles [4,5,6,7] and Q-heads [4,5,6,7]
- Active position (current token) accumulated on core 0 only
- Cross-core log-sum-exp merge via `nisa.sendrecv` (pipe_id=1,2,3 for max/sum/v)
- Final Wo all-reduce via `nisa.sendrecv` (pipe_id=0), unchanged from v14b

### Bugs fixed

| # | Root cause | Fix |
|---|---|---|
| 1 | `list(range(N))` returns a `range` object rejected by NKI compiler | Replaced with literal `[0,1,2,3,4,5,6,7]` |
| 2 | Python dicts with int keys rejected by NKI compiler | Converted all `{s_t: tensor}` dicts to lists |
| 3 | `enumerate()` not supported in NKI trace loops | Replaced with indexed `for i in range(len(…)): s_t = owned_s_tiles[i]` |
| 4 | Test file passed `Wv_np` (not pre-transposed) to device | Fixed to `Wv_pt` |
| 5 | `nki.simulate` hangs with `nisa.sendrecv` (single-threaded) | Rewrote test to use `torch_xla` on-device execution |
| 6 | **Design flaw**: Q was computed only for owned heads (non-owned heads had Q=0); S-tile sharding requires correct Q for ALL heads on each core so that each core's partial KV scores are correct before the log-sum-exp merge | Wq loaded for all 8 heads on both cores; Q projection computed for all 8 heads; removed head-ownership masking code that was attempting to paper over this |

**Bug 6 root cause detail**: In v14b (no S-tile sharding), both cores process ALL S-tiles. Non-owned heads have Q=0, but this only affects non-owned head columns in `attn_out`—which are never read in the per-owned-head Wo projection. With v16a S-tile sharding, core 1 holds tiles 4-7 which may be entirely masked for small position values. After the log-sum-exp merge, core 1's `attn_out` for its owned heads (4-7) is assembled from its masked tiles + core 0's tiles 0-3 via `sum_peer`/`v_peer`. But core 0's `sum_acc[h=4..7]` was zero (Q=0 → scores=0 but NOT what core 1 needs). Loading full Wq on both cores is the correct fix.

### Correctness results

| LNC | pos | attn_max | nk_max | nv_max | verdict |
|---|---|---|---|---|---|
| 1 | 0 | 2.44e-04 | 4.88e-04 | 2.44e-04 | PASS |
| 1 | 63 | 9.77e-04 | 9.77e-04 | 1.22e-04 | PASS |
| 1 | 64 | 9.77e-04 | 2.44e-04 | 2.44e-04 | PASS |
| 1 | 127 | 9.77e-04 | 4.88e-04 | 2.44e-04 | PASS |
| 1 | 128 | 9.77e-04 | 2.44e-04 | 2.44e-04 | PASS |
| 1 | 129 | 9.77e-04 | 2.44e-04 | 2.44e-04 | PASS |
| 1 | 255 | 4.88e-04 | 4.88e-04 | 1.22e-04 | PASS |
| 1 | 320 | 4.88e-04 | 4.88e-04 | 2.44e-04 | PASS |
| 2 | 0 | 2.44e-04 | 4.88e-04 | 2.44e-04 | PASS |
| 2 | 63 | 9.77e-04 | 9.77e-04 | 1.22e-04 | PASS |
| 2 | 64 | 9.77e-04 | 2.44e-04 | 2.44e-04 | PASS |
| 2 | 127 | 9.77e-04 | 4.88e-04 | 2.44e-04 | PASS |
| 2 | 128 | 9.77e-04 | 2.44e-04 | 2.44e-04 | PASS |
| 2 | 129 | 9.77e-04 | 2.44e-04 | 2.44e-04 | PASS |
| 2 | 255 | 4.88e-04 | 4.88e-04 | 1.22e-04 | PASS |
| 2 | 320 | 4.88e-04 | 4.88e-04 | 2.44e-04 | PASS |

**16/16 PASS** (atol=1e-3, rtol=1e-2 vs fp32-promoted PyTorch reference)

### Benchmark

| Metric | v14b_pretransposed (baseline) | v16a_seq_parallel (LNC=1) | v16a_seq_parallel (LNC=2) |
|---|---|---|---|
| device_time_us | 75.42 | 78.60 | 81.73 |
| TensorE % | 39.2% | 34.7% | 35.3% |
| VectorE % | 51.9% | 44.4% | 48.1% |
| DMA active % | 47.5% | 45.1% | 47.0% |
| spill_bytes | 163,840 | 163,840 | 131,072 |
| hbm_read_KiB | 11,554 | 10,193 | 15,026 |
| hbm_write_KiB | 656.5 | 652.5 | 656.5 |

### Analysis

v16a **regresses** vs v14b by +6.3 µs (LNC=2). The S-tile sharding saves ~50% KV-cache DMA (each core loads 4 tiles × PMAX×d instead of 8), but the bug fix required loading all 8 heads' Wq on both cores instead of 4 heads per core. This doubles the Wq DMA bandwidth — 15,026 KiB vs 11,554 KiB baseline, an increase of ~3,500 KiB. The added Wq DMA overwhelms the KV cache savings.

HBM read breakdown (LNC=2 v16a vs baseline):
- **KV cache** (tiles 0-3 per core): ~½ of baseline KV reads — savings ~1,920 KiB per core
- **Wq** (all 8 heads vs 4): doubled — extra ~3,500 KiB
- Net: **+1,580 KiB** extra HBM read → measured +3,472 KiB (additional scheduling overhead likely)

DMA is still the critical path at 47% active on both variants.

### Lessons

1. **S-tile sharding is only beneficial when Wq can also be head-sharded.** In the current design, S-tile sharding forces full Wq loads (all heads on both cores), negating the KV savings. A correct S-tile-only design would require broadcasting Q across cores via sendrecv before computing KV scores.
2. **`nki.simulate` cannot execute two-core `nisa.sendrecv`** — it runs single-threaded and deadlocks. Always use `torch_xla` on-device execution for LNC=2 correctness tests.
3. **Python `range`, `enumerate`, and int-keyed dicts are not valid** in NKI kernel trace loops. Use Python `range(N)` with index access, or explicit literal lists.

### Potential path to improvement

To recover v16a's intended savings without the Wq overhead:
- **Q exchange via sendrecv**: Core 0 projects Q for heads 0-3, core 1 for heads 4-7; exchange via one `sendrecv` (pipe_id=4); then each core computes KV scores for its owned tiles with complete Q. Net: 4 heads Wq each + 1 Q exchange sendrecv vs 8 heads Wq each.
- **FP8 / MXFP8 weight quantization** (Wq/Wk/Wv/Wo): 2× DMA reduction, compatible with any sharding strategy. Target: <38 µs from MBU ceiling.

**Best current variant: `v15c_broadcast_elim.py` @ 75.1 µs (v16a @ 81.7 µs is a regression).**
