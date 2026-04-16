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
