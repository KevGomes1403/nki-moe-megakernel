# DeltaNet decode-layer optimization — findings

Agentic optimization loop on the isolated DeltaNet decode layer, 2026-08-13.
Repo `/home/ubuntu/nki-moe-megakernel`, branch base `675c12e` (`qwen36-nkilib-fork`).

**Deliverable branch: `deltanet-opt` (commit `2983e6e`) — the one measured win.**
Patches also in `patches/`. Harness in `harness/`. Raw per-run metrics in `results/`.

---

## What was measured

Isolation target: `deltanet_attention_layer_state` (T=2 verify) —
norm → in_proj → conv → recurrence → gated norm → o_proj. This is the DeltaNet
attention half minus the TP all-reduce.

Harness (`harness/bench.sh <repo> <core> <tag>`, env `REPEATS=n`):
validate → emit NEFF (bf16) → capture NTFF ×n → ingest → analyze → `metrics.json`.

Correctness gate: `harness/validate.py`, atol=1e-5 rtol=1e-2, fp32, **cores=2**,
decode T=1 and verify T=2, over `o_out` / `candidate_states` / `conv_cand`.
Reuses the repo's own golden in
`megakernels/qwen3_6_moe/tests/test_deltanet_in_proj_out_fused_kernel.py`.

### Harness gotchas (all cost a run to find)
- `verify_T2` at **cores=1 does not compile even on baseline** — `recurrence.py`
  rank-1 update needs `W_full=1024` fp32 of PSUM (512 KB) vs a 256 KB bank.
  Cannot be a gate. The model deploys `deltanet_attention_layer_state[2]`.
- Profile in **bf16** (deployment dtype). fp32 inputs give a different regime
  entirely: 139.7 µs and 25.3 MB instead of ~95 µs and 17.19 MB.
- **`Summary.hbm_read_bytes` is authoritative for bytes.** A standalone capture
  does not packet-trace the large weight loads, so `DmaPacket` under-reports
  massively (0.08 MB vs the real 17.19 MB).
- `neuron-explorer capture -s` is a **path prefix, not a directory**.
- bf16 legitimately fails the 1e-5 gate (max_rel ~2.4) — inherent noise in the
  2048-deep in_proj contraction. Assert on fp32, report bf16.
- **Run-to-run spread is 2.5–7 µs.** Single-shot wall clock cannot resolve a
  3 µs win. Use `REPEATS>=5` and compare against a *same-session* control;
  the device drifts several µs over hours.

---

## THE decisive finding: this kernel is latency / hand-off bound, not throughput bound

Do not optimize by removing instructions or engine-busy time. Evidence:

- A variant with **3.0 µs less Tensor AND 5.3 µs less Vector instruction time
  ran 3.9 µs SLOWER** (plan 2's per-head block-diagonal matmul).
- Removing 42 transpose+copy pairs (`conv.py` MATMUL 100→58) left `conv.py`
  union-busy unchanged — they were already fully overlapped (plan 6).
- Batching the gated norm cut its work (TENSOR_TENSOR 5→3, ACTIVATE 5→3,
  busy 13.25→11.18 µs) and still regressed: the baseline already hides token 0's
  norm under token 1's recurrence, so only the last token's norm is exposed.
  Measured exposed chain 3.69 → 5.07 µs (plan 3).
- The one win **added** work (+24 matmuls, +18 DMAs per core) and won purely on
  pipelining (plan 1).

**Justify changes by exposed critical path / dependency-chain depth / overlap.**

---

## Hard NKI constraints discovered by measurement

- **Compute engines cannot address a non-zero partition base.** `nisa.tensor_copy`
  into `[1,128]` at partition h fails on Scalar, Vector *and* GpSimd:
  `Invalid access of 1 partitions starting at partition 1`. Only DMA and
  `nc_stream_shuffle` (bases 0/32/64/96) cross partitions. This also kills
  `tile_position=(32*h,0)` stationary packing. A single shuffle cannot broadcast
  across quadrants — the mask is per-quadrant.
- `nisa.dma_copy` requires **matching partition counts** src↔dst. Partitions
  cannot be folded into a row.
- `scalar_tensor_tensor` / `tensor_scalar` `operand0` must be a scalar or `(P,1)`
  tile — anything varying along the **free** axis is ineligible.
- `tensor_partition_reduce` runs on GpSimd; catastrophic here (+110 µs).
- **Every fp32 `nc_matmul` lowers to 2 LDWEIGHTS + 2 MATMUL.** `nc_transpose` is
  single-pass.
- **Vector cost is ~1 ns per free-dim column, independent of partition count**
  (`[128,512]` and `[1,512]` both ≈685 ns). Widening partitions is free;
  only shortening the free axis pays.
- Activation table map: `t0={COPY,SQUARE,RECIPROCAL_SQRT}`, `t1=SIGMOID`,
  `t2=SILU`, `t3=EXP`, `t4=SOFTPLUS`. A swap costs 1283 ns.
- SBUF free axis of the normed hidden is **shard-major**:
  `(H0, BxS, num_shards, H1_shard)`. The naive `reshape_dim(0,(H0,H1))` full-H
  view is a row permutation that **silently produces garbage** (validated at
  max_abs=2.46 before being caught). Correct view is
  `reshape_dim(0,(num_shards,H0,H1_shard)).permute((1,0,2,3))`, which is
  non-contiguous and cannot be flattened back to 3-D.

---

## Round 1 results

### Definitive head-to-head (quiet device, same logical core, 5 repeats each)

| | median | min-max | spread | no-compute | any-compute |
|---|---|---|---|---|---|
| baseline | 99.01 µs | 97.02-99.44 | 2.42 | 26.45 µs | 70.0% |
| **plan1** | **95.28 µs** | 94.16-95.94 | 1.78 | **8.29 µs** | 90.5% |

**−3.8%.** Note the mismatch: plan 1 removed **18.2 µs of stall** and bought only
**3.7 µs of wall**. Filling gaps exposed serialization underneath rather than
shortening the path. Consequences:

- Pure gap-filling is nearly exhausted — only ~8.3 µs of no-compute remains, so
  the ceiling on that strategy is ~87 µs.
- No single engine is near saturation (Tensor 59.3%, GpSimd 27.5%, Vector/Scalar
  lower) while any-compute is 90.5%. The engines take turns: a hand-off chain,
  one node at a time. **The remaining lever is engine-vs-engine concurrency —
  getting two engines to co-issue — not removing idle.**


| plan | change | outcome |
|---|---|---|
| **1 — WIN** | I-column-shard in_proj across LNC cores, deleting the `sendrecv`+add+`PSEUDO_CORE_BARRIER` seam | **−4.0%**, 94.89 vs 98.82 control. no-compute-engine **29.56 → 9.04 µs**, any-compute 70.1 → 90.5%, own spread 0.35 µs vs 3.1 |
| 2 — partial | (c) 4 `[128]·[128]→1×1` dot matmuls → `tensor_tensor`+`reduce`+`transpose` | −14 MATMUL, −14 LDWEIGHTS, −3.4 µs Tensor time, **no wall win** (inside noise). Kept in `patches/plan2c-*.patch` |
| 3 — negative | batch gated norm; reorder activations to cut `ACT_TABLE_LOAD` | all 4 variants regressed; reorder moved the counter **up** 18→19 and 18→20 |
| 6 — negative | conv taps channel-on-partition; hoist per-token v-gather DMAs | both regressed ~+5 µs |

### Plan 1 detail (`NKI_DELTANET_IN_PROJ_I_SHARD=1`, default off)
Each core contracts full H=2048 over its own **1552** of 3088 columns
(1544 exactly by consumption; 1552 shipped, taking the whole 16-wide `a|b` block
to avoid 8-byte DMA descriptors — 0.5% redundant FLOPs).

The win is **not** the ~2 µs sendrecv. It is pipelining: each column run is
matmul'd and evicted as its weight lands, moving `conv.py`'s first `nc_transpose`
from 42.74 → 24.60 µs and the recurrence's first matmul 63.02 → 58.07 µs.
Ordering matters — three variants measured; run-major/one-matmul-per-run won.

Disclosed costs: weight DMA goes 2 → 20 DMAs over 10240 short runs
(`DMA_DIRECT2D` 1.32 → 21.75 µs, GpSimd 14.5 → 27.5%); full-H contraction doubles
h1 tiles (LDWEIGHTS 6.72 → 15.65 µs, Tensor 41.7 → 59.3%). Both land on engines
with slack. Achieved BW unaffected (181.4 vs 173.8 GB/s).

### Refuted claims
- Plan 6 concluded the 17.7 µs weight-stream stall at 96% of peak BW was a
  **hard floor** for an isolated layer. Plan 1 cut no-compute to 9.04 µs. Not a floor.
- Plan 2 *proposed* moving `expA` adjacent to `exp` for ~1.28 µs/core. Plan 3
  *measured* that class of reorder: the counter went **up**, because the scheduler
  already hoists `exp(A_log)` into a 23.7 µs Scalar-idle window (free), while
  moving the gate block splits `_load_normed_qk`'s two rsqrts across a table
  boundary, turning one load into two. The SIGMOID table load also sits in that
  idle window and costs 0 ns of critical path.

---

## Open leads for round 2

1. Repack `proj_w` host-side to `[num_cores, H, I_core]` so each core's columns are
   row-contiguous — restores a 128-descriptor weight DMA (~−11 µs GpSimd/core).
   Needs a kernel-side `(hbm_start,out_start,size)` triple **and** test/model
   weight-prep changes (cross-file).
2. Pack q|k (256+256) into one 512-wide column tile: −16 matmuls, −16 LDWEIGHTS
   per core, now that Tensor is busiest at 59.3%.
3. `out_proj.py:86` still holds the **other** cross-core `sendrecv`, and
   `output_projection_tkg` matmuls are now the tail — likely the biggest remaining seam.
4. Gated norm runs on `[1, W]` (1 partition, 512 columns). In `[T*Hv, d]` layout
   each op touches 128 columns instead of 512 (~2.7 µs Vector). Needs a
   recurrence-output layout change; `nisa.activation_reduce` cannot substitute.

Vendored files (`qkv_tkg.py`, `output_projection_tkg.py`) are shared with GQA —
extensions must be **additive and env-gated off by default**.

---

## Round 2 — profile of the plan-1 base, and where the time actually is

Fresh capture of the I-column-shard build. **in_proj is the kernel.**

| phase | len | Tensor busy | DMA busy |
|---|---|---|---|
| instruction-stream head (no input DMA in flight) | 13.1 µs | 12% | 54% |
| RMSNorm prologue + wait for first `proj_w` chunk | 10.0 µs | 26% | 92% |
| **in_proj matmuls** | **52.1 µs** | **91.8%** | 100% |
| conv + recurrence + gated norm | 37.5 µs | 41% | 18% |
| **o_proj tail (zero non-o_proj compute in it)** | **21.4 µs** | 80% | 14% |

### The in_proj PE cost model (validated to 0.3%)
`2 (fp32 LOW_HIGH pair) × 16 (h1) × Σ(128 + i_size)` over runs `{256,256,512,512,16}`
= **70,144 cycles = 48.0 µs @1.46 GHz**, vs 47.87 µs measured.

`proj_w` streams to 63,031 ns; PE finishes at 75,238 ns. **PE is the binding
resource for the phase end, so PE cycles removed pay ~1:1 up to ~12.2 µs**, after
which the phase floors on the DMA end. Every round-2 saving is derived from this.

**`LDWEIGHTS` is 14.0 µs of that 48.0 µs** — `2 × 16 h1 × 5 tiles × 128 rows` —
and it is pure overhead for a rank-2 GEMV: the stationary is `[128, BxS=2]`, so
each load streams 128 rows to serve 2 useful columns.

### Two round-1 leads killed by measurement
- **Hoisting the `out_w` load**: already fully prefetched. Lands at 77,687; first
  o_proj MATMUL at 114,802. **37.1 µs of slack, 0 ns attributable stall.**
- **`out_proj.py:86` sendrecv as "the biggest remaining seam"**: lowers to a
  4,096 B transfer, **354 ns**, consumer `EVENT_SEMAPHORE` blocks 243/197 ns.
  Not a seam.

### Also rejected, with the measured reason
- **Operand swap** (weight stationary, hidden moving): cycles 70,144 → 54,080
  (−11.0 µs) but Tensor instruction count 320 → 832/core. At the measured
  per-instruction floor (~140 ns) the added 512 instructions cost 25–38 µs. Net loss.
- **Per-token split of the o_proj matmul**: cost is (128 + 512) cycles per
  (head, f_tile) whether `BxS` is 1 or 2 — splitting doubles o_proj PE to 28.0 µs
  to hide at most 14. Net loss.
- **bf16/tf32 for in_proj or o_proj**: would halve PE (profile confirms fp32 lowers
  to paired MATMULs, `fp32_mode = LOW_HIGH`). Blocked by the fp32 atol=1e-5 gate.
- **`dge_mode.none`/`hwdge` on weight DMAs**: `DMA_DIRECT2D` costs 21,750 ns of
  GpSimd, but GpSimd has a contiguous 60,165 ns idle window and all `proj_w`
  triggers retire by 25,151 while the stream runs to 63,031. Descriptor generation
  is not the limiter.
- **Folding the recurrence decay into the read matmul**: algebraically valid but
  chain length is identical (`decay→mm→delta` vs `mm→eg⊙pair→delta`), and Vector
  cost is per-column and partition-count-independent, so shrinking `[128,512]` to
  `[2,512]` buys nothing.
- **The 13.1 µs instruction-stream head**: real (9.7%), 236,856 B over 97 packets
  with no input DMA yet — but it is a per-NEFF startup cost that amortizes in the
  multi-layer megakernel.

### Round-2 plans dispatched
1. **h1-outer loop to elide repeated `LDWEIGHTS`** (stationary is identical across
   all 5 column tiles at a given h1) — up to −11.2 µs; grouped-safe variant −8.4 µs;
   packing fallback −2.8 µs. Kill: LDWEIGHTS/core 160 → must reach ≤64.
2. **Host-side `proj_w` repack to partition-contiguous** — 128 descriptors instead
   of 1,024 runs/DMA; 269 GB/s currently = 62% of the 435 GB/s peak. −5–7 µs, and
   it lowers the DMA floor that caps plans 1 and 3 by ~14 µs. Kill: stream span
   47,232 ns → must reach <38,000.
3. **Defer the `z` column tile's 14.0 µs of matmuls into the conv's Tensor-idle
   windows** (8.6 µs measured across two stretches; nothing before the gated norm
   consumes z). Must interleave per conv segment — each engine runs its queue in
   order, so one block emitted after the conv cannot back-fill an earlier wait.
   −6.8 µs. Kill: conv-window Tensor-idle 8,608 ns → must reach <3,000.
4. **Transpose the gated row inside the recurrence, deleting the `attn_sb`
   round-trip** — 1,518 ns with all four compute engines idle on both cores.
   −2.0–2.5 µs. Kill: that gap must fall below 400 ns.

**Composition:** plans 1 and 3 are NOT additive (both remove in_proj PE; together
they overshoot the ~12.2 µs realizable before the DMA floor). Plan 2 is what raises
that ceiling. Plans 1 and 3 also conflict structurally in `_qkv_projection_i_shard`.

**Measurement caveat:** a hardware 0.5-utilization throttle covered 44.4 µs of the
round-2 capture and inflates durations 1.69× (identical matmuls: 592 ns outside,
1,001 ns inside). Cross-session A/B is void; every arm needs a same-session control.

---

## Round 2 results

| plan | outcome |
|---|---|
| **defer `z` column run into the conv windows — WIN** | **−4.11 µs (−4.4%)**, control 93.83 → 89.72 (10 reps pooled per arm, non-overlapping distributions, throttle coverage 47.7% vs 47.3%). Bit-identical. Shipped as `NKI_DELTANET_IN_PROJ_Z_DEFER=1`. |
| h1-outer `LDWEIGHTS` elision — negative | Elision **works** (160 → 34/core) but costs more than it saves: **+3.33 µs**. |

### CORRECTION: `in_proj` is DMA-bound, not PE-bound — the earlier model was a throttle artifact

The round-2 planner measured PE ending 75,238 ns vs `proj_w` DMA 63,031 ns and
concluded PE was binding, so PE cycles would pay ~1:1. **That capture read 135.4 µs
wall with ~44 µs of 0.5-utilization throttle coverage, which inflates matmuls 1.69×**
(identical matmul: 592 ns unthrottled, 1,001 ns throttled). Two agents independently
re-measured unthrottled controls and found the opposite sign:

| | last in_proj MATMUL end | last INPUT DMA end | DMA − PE |
|---|---|---|---|
| agent A, ctrl-a/b/c (15 captures) | 44,756 / 45,543 / 44,444 ns | 50,083 / 50,269 / 50,032 ns | **+5.2 to +5.5 µs** |
| agent C control | 45,259 ns | 49,418 ns | **+4.2 µs** |

PE merged-busy is only 57–65% of span, and PE is idle 43–52% before the conv.
**There is no 1:1 PE→wall conversion.** Any plan justified by removing PE cycles
from in_proj should be re-derived. Always check the `Throttle` table before
trusting a phase-ordering claim.

### The `LDWEIGHTS` elision: it works, and it is a trap

The backend **does** elide repeated `LDWEIGHTS` when consecutive `nc_matmul`s
present a byte-identical stationary at a fixed `tile_position`:
**160 → 34 per core** (34 = 2 fp32 LOW/HIGH × 16 h1 + 2 — the ideal), LDWEIGHTS
instruction time 15.6 → 3.4 µs.

But pinning all tiles to `tile_position=(0,0)` collapses the **4 rotating PE column
groups** onto one and costs moving-stream throughput:
**1.38 → 1.08 columns/cycle (−28%)**, in_proj MATMUL window **24.7 → 31.4 µs (+6.6)**
despite issuing 126 fewer LDWEIGHTS. Net **+3.33 µs**.

**The two goals are structurally in tension: elision requires a fixed
`tile_position`; a fixed `tile_position` forfeits column-group rotation.**

Secondary effect (the briefed risk, confirmed): h1-outer defers all PSUM evictions,
pushing the q/k/v hand-off 25.5 → 53.7 µs so the conv can no longer overlap in_proj.
Grouping h1-outer over `{q,k,v}` only fixes the hand-off (`conv.py:195` starts 7 µs
*earlier* than control) but cannot overcome the throughput loss — lands at +0.17 µs,
a wash inside the 1.56 µs control drift.

### Mechanism of the `z`-deferral win (not the one predicted)
The plan assumed 14.0 µs of z PE work draining into a Tensor-bound conv stall.
Measured: z is **~3.8 µs** of PE, the compiler **already interleaves** in_proj
matmuls with conv transposes, and the conv-window stall is **Scalar/Vector-bound**
(five `ACT_TABLE_LOAD`s at 1,283 ns). The real mechanism: z's matmuls were stalling
the head of the Tensor queue on the tail of z's own 2 MiB weight DMA (a 1.87 µs
mid-run gap plus 2.9 µs before its first matmul). Removing them lets the conv's
transposes proceed while z streams. In-place block 25.1–26.5 → 16.8–18.5 µs;
recurrence's first Tensor instruction 58.4 → 51.4 µs.

### Two NKI frontend constraints (both agents hit these on first compile)
- **List comprehensions are rejected**: `error: unsupported expression`.
- **Inner `def`s are rejected**: `NKI does not support inner function definitions`.
  Pass `(module_level_fn, data)` tuples instead of closures.

### Round 2, remaining results

| plan | outcome | flag |
|---|---|---|
| **partition-contiguous `proj_w` repack — WIN** | **−3.43 µs (−3.6%)**, 94.52 → 91.09 pooled, disjoint distributions, **bit-identical on device** (`torch.equal` True) | `NKI_DELTANET_PROJW_PACKED=1` |
| **`attn_loc` tail restructure — WIN** | **−3.76 µs (−3.9%)**, 95.29 → 91.53, bit-identical. Dead hand-off **1262 → 54 ns** | `NKI_DELTANET_ATTN_LOC=1` |

### FINAL bottleneck model for `in_proj` (supersedes both earlier claims)

The weight stream **paces** in_proj but does not bound it. Measured on an
unthrottled control: in_proj's own `proj_w` stream ends at 41,573 ns, its last
matmul at 44,449 ns — **the phase tail is PE, +2.6 µs behind the stream**, and the
gap is *constant* (treatment: +2,733 ns). So:

- Stream-span reduction converts to phase-end reduction **~1:1**
  (`w_end` 41,573 → 39,449, `mm_end` 44,449 → 42,182).
- And to wall at **1.39×** (−2.47 µs of span → −3.43 µs of wall), because
  `out_w` also finishes earlier once it stops contending with an inefficient
  `proj_w` stream (full INPUT end 49,778 → 47,736 ns).

The earlier "in_proj is DMA-bound" reading mis-attributed the 49,418 ns INPUT tail:
that is **`out_w`** (o_proj weight, 2.10 MB/core, streaming from ~36.6 µs and
outliving the phase), not `proj_w`. Both earlier models were wrong in opposite
directions; this one is measured on both arms.

### The repack, quantified
`read_shape [[8 128]]` / `read_steps [[6176, 49408]]` → `[[128 1]]` / `[[8192, 1]]`.
**Descriptors per core 10,240 → 1,664 (6.2×)**; run bytes 512–1,024 → 4,096.
Stream span 26.3 → 23.9 µs, rate 240 → 267 GB/s.
Packet **count** is invariant (1,728 → 1,760): packets are byte-quantized at ~3.6 KB
against a fixed 6.36 MB transfer, so count ≈ bytes/3.6 KB regardless of layout.
**Descriptors are the metric, not packets.**

### Gate blind spots found the hard way
- **The fp32 gate cannot catch bf16-only backend failures.** Transposing z into a
  bf16 PSUM tile compiles at T=2 but fails the verifier at T=1
  (`checkMatmultOutputs`, 2-byte sub-word PSUM write) — only in the deployment dtype.
  Stage through an fp32 tile.
- **A feature wired into `deltanet_attention_layer` but not
  `deltanet_attention_layer_state` silently falls back and still passes the gate
  bit-identically.** The harness profiles the `_state` entrypoint. Verify a flag took
  effect from the profile's `read_steps`/`read_shape`, never from the gate.

### Production follow-up required for the repack (NOT done)
`build_deltanet_in_proj_fused` (`modeling_qwen36_a3b.py:3489`) must emit the packed
layout, and `self.in_proj_fused.weight` is a `RowParallelLinear` parameter **also
consumed by the PyTorch fallback** `_project_inputs` (`modeling_qwen36_a3b.py:461-462`).
It must not be mutated in place. Either carry the packed tensor as a second buffer
(+6.36 MB/core bf16 per layer at TP=4 — in_proj weight residency doubles) or port the
fallback to the packed layout and drop the `[H, I]` copy. That decision is open.

---

## Integration: the wins stack, and one does not

Merged commit `f6b8b16`. All arms measured on logical core 0, fully interleaved,
15 captures each for the headline / 10 each for the ladder.

### Headline — base vs stack

| arm | flags on top of `I_SHARD=1` | median | min-max | Δ vs A |
|---|---|---|---|---|
| A | — | 94.89 µs | 93.67–96.79 | — |
| **D** | `Z_DEFER` + `PROJW_PACKED` | **82.67 µs** | 81.24–84.07 | **−12.22 µs (−12.9%)** |
| F | + `ATTN_LOC` | 85.58 µs | 82.68–88.76 | −9.31 µs |

**Every D capture beat every A capture** (D max 84.07 < A min 93.67).

### Composition ladder

| arm | flags | median | Δ vs A |
|---|---|---|---|
| A | control | 94.50 | — |
| B | `Z_DEFER` | 91.40 | −3.11 |
| C | `PROJW_PACKED` | 91.12 | −3.38 |
| D | both | **82.73** | **−11.77** |

**D is superadditive.** Sum of parts −6.49 µs; measured −11.77 µs — an extra ~5.3 µs.
Achieved HBM bandwidth **181 → 208 GB/s**. Reproduced independently in the headline
ladder (−12.22). Mechanism: the repack shortens the weight stream, and the z-deferral
removes the matmuls that were stalling the Tensor queue on the tail of that same
stream — each makes the other's bottleneck cheaper to clear.

### `ATTN_LOC` wins alone and loses in the stack — DO NOT DEPLOY IT

Alone: −4.81 µs (95.57 → 90.75), reproducing its author's −3.77.
On top of D: **+2.13 µs** (D 83.19 → F 85.32), confirmed again at +2.91 in the
headline ladder. Profile deltas when adding it to D: `out_proj.py` union-busy drops
1.26 → 0.20 µs exactly as designed, but `recurrence.py`+`norm_gate.py` busy rises
~5.8 µs and all-engines-idle rises 5.20 → 7.47 µs. Its per-token transposes are free
in the unstacked schedule and exposed in the stacked one.

**Not a throttle artifact**: F has the LOWEST throttle coverage of any arm
(39.9/39.4% vs A's 46.9/53.7%) and is still 2.9 µs slower than D.

**Deploy `I_SHARD` + `Z_DEFER` + `PROJW_PACKED`. Leave `ATTN_LOC` off.**

### Correctness of the stack
All six tensors (`o_out`, `final_state`, `new_conv_state` at T=1; `o_out`,
`candidate_states`, `conv_cand` at T=2) dumped fp32 and compared with `torch.equal`:
**B, C, D and FULL are all `True`, max_abs_diff 0.0 vs A.** The bf16 error fingerprint
is byte-identical across all seven arms, so bit-identity holds in the deployment dtype.

### Flag effect verified from the profile, not the gate
- `PROJW_PACKED` off → `read_shape [[8 128]]`, `read_steps [[6176, 49408]]`.
  On → `[[128 1]]` with `[[16384, 1]]`/`[[8192, 1]]`; the strided pattern is gone.
- `Z_DEFER` off → last `qkv_tkg.py` MATMUL at conv+32.2k ns.
  On → conv+50.7k (unpacked) / conv+42.1k (packed), MATMUL count unchanged at 160.

### The merge interaction that had to be resolved
Both features edit `_qkv_projection_i_shard`. Resolution: the packed branch goes in
the **weight-load loop**, which runs unconditionally for every segment including
deferred ones, and both paths end at the same `(H0, H1, i_size)` view — packed
directly, unpacked via `flatten_dims(1, 2)`. The deferred `pending` tuple carries the
already-sliced `w_sb`, so the drained z run picks up the packed layout automatically
(confirmed by `[[8192, 1]]` descriptors in the D profile). Both derive their runs from
the single `in_proj_column_shard`, keeping host-side pack order and kernel-side offset
consistent regardless of which flags are on.

---

## Round 3 — probe of the stacked build (`929217c`, all three flags on)

**81.73 µs median**, 210.6 GB/s. Wall decomposes into five windows:

| window | span | ANY-busy | idle | binding | what |
|---|---|---|---|---|---|
| `[0.0, 12.6]` | 12.60 | 4.72 | **7.88** | — | NEFF/runtime preamble, all `?`-sourced. **Not kernel-addressable.** |
| `[12.6, 19.6]` | 7.00 | — | — | serial chain | interleave load → RMSNorm → first in_proj matmul |
| `[19.6, 33.6]` | 14.04 | 13.86 | 0.18 | **Tensor 79.7%** | 64 in_proj matmuls. At its floor. |
| `[33.6, 50.4]` | 16.76 | 14.43 | 2.33 | **Scalar 82.1%** | conv MAC/SiLU + q/k l2norm. 7.70 µs is `ACT_TABLE_LOAD`. |
| `[50.4, 69.7]` | 19.25 | 19.16 | **0.09** | **Vector 77.4% / Tensor 69.6%** | recurrence token loop. Fully packed. |
| `[69.7, 78.3]` | 8.69 | 7.67 | 1.02 | Tensor 65% | o_proj, 16 serial matmuls |
| `[78.3, 81.7]` | 3.39 | 1.42 | 2.04 | — | epilogue / cross-core barrier |

**~16 µs of prologue+epilogue is a hard floor** — all `?`-sourced `DRAIN`/`EVENT_SEMAPHORE`/
`TENSOR_LOAD`/`SET_ORDERING_MODE`, constant across every arm, not reachable from kernel source.

Three round-2 beliefs died here: (1) the `proj_w` stream **no longer paces** — it ends at
23.02 µs vs the last in_proj matmul at 33.64, so 10.6 µs of slack, not 2.6; `PROJW_PACKED`
already ate that lever. (2) "the SIGMOID load costs 0 ns of critical path" is false on this
build — it is followed immediately by the largest in-body idle hole, `[40.15, 42.33]` = 2.18 µs.
(3) The bottleneck is no longer in_proj at all.

**Core imbalance (new):** core 1 lags throughout and sets the wall clock — its in_proj matmuls
end at 36.24 vs core 0's 33.64, its o_proj runs 9.08 µs vs 6.60, and it takes 20.5 µs of
`activity_1` throttle in `[58.98, 79.46]` vs core 0's 13.65. Total-activity reductions help
core 1 more than core-0 numbers suggest.

**`ACT_TABLE_LOAD` is exactly 1283 ns whether throttled or not** — table-load removals are
throttle-immune and convert un-discounted. Tensor/matmul removals in throttled windows need
~1.69× discounting when reasoning about work (not about window occupancy).

### Why `ATTN_LOC` cannot be recovered cheaply (resolved)
`out_proj_compose` transposes `attn_sb[0:T, h*d:(h+1)*d]` — **batched over T**. Per-token you
cannot batch over T, so the count is `Hv_core × T` = 8 instead of `Hv_core` = 4. Those 8
transposes cost ~3.7 µs of Tensor in a window with **0.09 µs of idle**, pushing Tensor 69.6%
→ ~89%. The transposes must follow the gated row, and the only window with Tensor slack
(`[33.6, 50.4]`) is before the gated rows exist. **Prerequisite for a retry is Tensor headroom
inside the token loop.**

### Round 3, plan 1 — de-thrash the Scalar activation table: NEGATIVE

| arm | items | median | GB/s |
|---|---|---|---|
| ctl1 / ctl2 | none | **81.65 / 82.15** | 210.8 / 209.6 |
| a|b-first + z-index + reorder + tap-engine | 1,2,4,5 | 87.35 | 197.1 |
| a|b-first + z-index | 1,2 | 85.33 | 201.7 |
| reorder + tap-engine | 4,5 | 88.71 | 194.2 |

Controls bracket the treatments; no drift. Bit-identical on all six tensors in every arm.

**The premise held and the plan still failed.** a|b-first genuinely moved the `a_sb`/`b_sb`
DMAs from 35.98–37.31 → **24.60–28.20 µs** — but **the gating chain did not hoist with it**:
SOFTPLUS's table load still fires at ~35.0 µs. Availability was never what pinned the chain.

Whole-run `ACT_TABLE_LOAD` on Scalar pcore0 is **10 in both arms** — the reorder *relocated*
loads out of the window rather than eliminating them. In-window Scalar busy fell 14.26 → 7.31 µs
but in-window Tensor rose 8.48 → 15.25 µs and whole-run ANY-busy 69.33 → 73.61: the cost moved
onto the larger engine.

**Conclusion for anyone resuming this line:** the 7.70 µs of in-window `ACT_TABLE_LOAD` is NOT
reachable by making `a`/`b` available earlier. The loads are pinned by something downstream of
availability; establish what before spending another reorder.

### New NKI constraint
`nisa.tensor_copy(..., engine=nisa.engine.gpsimd)` **does not compile when the source is PSUM**:
`tensor_copy src must be in [sbuf], got psum`. Both conv tap transposes evict from PSUM, so
GpSimd is not a legal target regardless of its 0.00% occupancy.
