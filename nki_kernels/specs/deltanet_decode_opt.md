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
