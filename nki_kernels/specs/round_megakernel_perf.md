# Round megakernel — profile findings

Profile of the fused speculation round (`qwen36_round_megakernel`) at spec_len=2, TP=4, LNC=2,
seq_len=128, trn2. Build `/home/ubuntu/models/qwen36_a3b_round_mk`.

Findings are ranked by estimated wall-clock saving across the whole round. Every number below is
measured from the trace unless labelled an estimate.

## Artifacts

- Capture: `/home/ubuntu/profiles/round_mk/` — NEFF, 4 ranks of NTFF, ingested parquet under
  `ne-data/`, plus a re-runnable `stage_partition.py`.
- Viewable copy: `parquet_files/` at the repo root (see its README for the viewer command).
- Steady-state iteration (`--num-exec=3 --profile-nth-exec=2`), DGE notifications enabled.

## Method and its limits

Stage identity is **not** in the profile. The `name_prefix` tags (`d1_`, `v_L{i}_`, `d2_`, `*_lm_`)
do not survive tracing: `bir_instruction_name` holds compiler ids, `layer` is `Unknown`,
`hlo_name` is empty, `stack_frame_ids` is NULL. The partition is temporal, anchored on
`nki_source_location` file/line landmarks, which are populated on 97.7% of instructions. It is
inference from source lines, not a label.

Other caveats that shaped the numbers:

- `DmaPacketAggregated` busy percentages are wrong here — one `qSyncDynamicHW` row spans
  603,452→8,575,461 ns carrying 128 bytes and reads as a 100%-busy queue. All bandwidth figures
  come from per-packet `DmaPacket` sums.
- Inputs are synthetic (neuron-explorer allocated its own IO), so expert-selection *distribution*
  is not meaningful. Transfer *volume* is data-independent under fixed-count top-k.
- The NEFF carries no compiler metrics; `model_flops` / `mfu_hlo_estimated_percent` unavailable.
- Ingest logged "DGE packet count exceeds number of DMA trace entries" for engines 0, 21, 29.
- Engine busy is interval-merged `ActiveTime`, per physical core. "any" is the union across cores.

## The round

Device total 9,495,665 ns (rank 0). Host p50 is 10.31 ms, so ~0.8 ms is runtime dispatch outside
the trace. Stage percentages are of the device total and do not sum to the host number.

| stage | ns | % round |
| --- | --- | --- |
| D1 draft (T=1, MTP layer + eh_proj + vocab head) | 1,058,976 | 11.15 |
| — of which `d1_lm_` | 478,424 | 5.04 |
| V verify trunk (40 layers) | 7,512,866 | 79.12 |
| — 30 DeltaNet layers | 5,978,000 | 62.96 |
| — 10 GQA layers | 1,535,000 | 16.17 |
| — `v_lm_` vocab head (runs concurrently inside D2's window) | 505,262 | 5.32 |
| D2 replay (T=1, headless) | 923,823 | 9.73 |
| — of which D2's own work | 246,096 | 2.59 |
| — of which XLA round epilogue | 144,823 | 1.53 |

D2's window is mostly not D2: the verify vocab head occupies 533 µs of it, legal because D2 depends
on the residual and not on the token id.

**Rank 0's D1 skew is an artifact.** D1 is 1.059 ms on rank 0 vs 0.890/0.889/0.904 ms on ranks
1/2/3, but all four ranks *trigger* their first collective at the same point (155.9–156.5k ns).
Rank 0's entry barrier is 186.5 µs vs 17–26 µs; it enters the NEFF ~165 µs early and idles. V and D2
agree across ranks to within 12 µs. Do not chase it.

## Nothing is saturated

Whole round: PE 28.8%, DVE 24.6%, ACT 17.6%, POOL 17.9%, SP 2.9%. All 32 DMA engines sit at 39–51%
and are balanced within 3 points. **Union of all compute engines busy is 74.2%** — 25.8% of the
round has no compute engine running anywhere.

V moves 1.935 GB at 258 GB/s against a measured sustainable peak of 635 GB/s (20 µs window) /
707 (10 µs) / 781 (5 µs). The DMA floor for V is ~2.7–3.0 ms; it takes 7.51 ms.

This is a serialization-bound kernel, not a bandwidth- or compute-bound one. Where V's 1.959 ms of
all-engine idle goes:

| cause | µs | share |
| --- | --- | --- |
| TP all-reduces | 1,080 | 55% |
| in_proj/qkv LNC sendrecv + core barriers | ~582 | 30% |
| conv candidate-store DMA wait | 209 | 11% |
| D1→V argmax all-gather seam | 62 | 3% |
| scheduler gaps < 400 ns | ~26 | 1% |

## Two hypotheses that did not survive

**`nc_transpose` cost is negligible.** In V: 7,584 transpose-mode instructions, 1,617 µs of
*instruction* time but only 320 µs (pcore0) / 330 µs (pcore1) of **merged PE-busy = 4.3% of V**, on
an engine at 29% occupancy. `flops_transpose / raw_flops = 2.48%`. In the draft path the total is
~22 µs (0.23% of round): `natural_to_tp2013` 5.8 µs (D1) / 5.6 µs (D2), `tp2013_to_natural` 9.0 µs,
`head_major` transposes 0.9–1.5 µs. The "16 transposes serialized on one PSUM tile" concern does not
materialize — the LDWEIGHTS/MATMUL pairs issue ~62 ns apart and pipeline, whole chain 1.5 µs.
`embed.py:108`'s 396 µs *span* in D1 is two calls separated by a collective, not transpose cost.

**Fragmented DMA is not a bandwidth problem.** 86,486 sub-64-byte packets in V (excluding
instruction fetch) carry 0.69 MB total and ~28 µs per engine across V. The 4-byte packets attributed
to `routed_experts_nki.py:85/:95` and `recurrence.py:451` are DMA *completion writes* accompanying
512 KB weight loads — the actual data packets there are 4,096 B. The contiguous-slab fix worked.

What does hurt is **latency**: a few small transfers and collectives that everything else waits on.

## Findings

### 1. 80 serialized 8 KB TP all-reduces — 400–900 µs
`nki_kernels/megakernel/collectives.py:57` (`all_reduce_gather_h`), `collectives.py:107`
(`all_reduce_gather_free_block`), called from `qwen36_verify_megakernel.py:187` and `:198`, twice per
layer × 40 layers.

80 × `AllReduce / Mesh / 4096 bf16`, 8,192 B in and out, mean **13,331 ns**, total 1,066,457 ns.
Merged busy equals the sum of durations — **strictly serialized, zero overlap**. Exposed idle is
1,079,527 ns = **11.4% of the round**; `AR_attn` runs at 6.7% any-engine busy, `AR_moe` at 14.7%.
Effective rate 0.6 GB/s: entirely fixed latency. Add ~145 µs of `cc_trigger_start_delay`.

No per-layer growth (mean by octile 13.09/13.20/13.82/13.52/13.03 µs) — the prior design note's
13.45→27.77 µs rise does not reproduce. One op decomposes as `SEMAPHORE_30_WAIT_EQ_8` 4,746 ns,
`SEMAPHORE_23_WAIT_EQ_24` 1,385 ns, ~16 mesh events at 18–25 ns, and ~5 µs of unattributed gaps.

Fixes, cheapest first: (a) **overlap** — DMA engines are 34%/15% busy in those windows, so issue the
next layer's attention weights and this layer's router/shared-expert loads into them explicitly
rather than trusting the scheduler; scheduling-only, low risk. (b) Halve the count by folding the
attention partial and the MoE partial into one all-reduce per layer, which needs the post-attention
RMSNorm's sum-of-squares computed locally first — a real algebraic restructure that changes
reduction ordering. (c) Check whether a 4-rank `nisa.sendrecv` ring beats nccl mesh at 8 KB.

### 2. MoE expert prefetch ring is only 2 deep — 270–440 µs
`nki_kernels/moe/components/routed_experts_nki.py:37` (`PREFETCH_SLOTS`), `:373` (prime),
`:380-386` (k+1 prefetch), loads at `:85` (gate/up slab) and `:95` (down).

Routed-expert phase is 47.75 µs/layer, 26.0 MB, **544 GB/s with DMA engines 81.6% busy** — 1.915 ms
total (25.5% of V). Peak sustainable is 635–707 GB/s, so ~18% of the engines' time is idle inside
the phase. Packets are 4,096 B modal; 50 × 524,288-byte aggregates per layer, exactly as designed.

Byte volume is near-irreducible: 2 tokens × 8 experts × (H·2I + I·H) × 2 B = 24 MB/layer minimum
against 26.2 MB moved (expected distinct experts across 2 tokens ≈ 15.75 of 16). But all 8 indices
are known the instant `router_topk` returns, while the loop issues them one ahead. Raise
`PREFETCH_SLOTS` to 4 (~6 MB SBUF; `SbufUsage` peaks at 16%, so there is room), hoist the token's
slab loads above the compute loop, and issue the `down` DMA concurrently with the two slab DMAs
rather than after them.

*Unknown:* synthetic inputs mean cross-token expert correlation is unmeasured. If real routing
correlates between adjacent speculative tokens, dedup could cut a further 10–25% of these bytes.

### 3. qkv PSUM accumulate runs as a DVE chain — 200–400 µs (high risk)
`nkilib/core/qkv/qkv_tkg.py:1526-1536`, reached from `deltanet/components/in_proj.py:28` and the GQA
qkv projection.

DeltaNet norm+in_proj is 20.8 µs/layer moving 6.6 MB at 318 GB/s with **DMA only 45.6% busy**; GQA
norm+qkv is worse at 192 GB/s / 26.6%. Per layer both cores, `qkv_tkg.py:1531` is 52.3
`TENSOR_TENSOR` at 615 ns = **32.2 µs of DVE**, making DVE the busiest engine (40%) in a phase that
should be DMA-bound. At 635 GB/s the 8.29 MB weight is 13 µs against 20.8 µs observed.

Accumulating in PSUM via `nc_matmul(..., accumulate=True)` across the array-tiled partitions removes
the DVE chain. **This is vendored nkilib shared with GQA and other models** — it would have to be a
local fork under `nki_kernels/`, and PSUM bank pressure needs re-checking.

### 4. Greedy argmax scans the vocab on one SBUF partition — 150–250 µs
`nki_kernels/lm_head/components/lm_head.py:155` (chunk `tensor_reduce`), `:168` (`nc_find_index8`),
`MAX_REDUCE_WIDTH` at `:43`.

`logits_sb` is `[T, V_core] = [1, 31040]` — a single partition. 15,520 elements in 16,309 ns =
**0.95 elem/ns, i.e. 1 of 128 DVE lanes**. In D1 this is a 56 µs tail with PE at 0.0% and DVE at
94.6%. In the verify head it is 4 × 16,309 + 4 × 16,314 = **130.5 µs of DVE**.

Chunk 0's reduce already overlaps the matmul, but chunk 1's reduce and both `nc_find_index8` calls
wait on the global `core_max`, exposing ~49 µs. Run `nc_find_index8` per chunk against that chunk's
own max so each chunk's search issues as soon as its logits exist, then select the winning chunk at
the end; raise the chunk count so the exposed remainder is small. The reversed-index trick at
`:184-193` already gives lowest-index-within-chunk, so tie semantics carry over with the same
`r = V_core - g` treatment.

Pays three times: D1's tail, the verify head, and D2's blocked start.

### 5. DeltaNet recurrence is a serial cross-engine chain — 150–350 µs
`nki_kernels/deltanet/components/recurrence.py:479,492,495`, `norm_gate.py:50,52`.

37.8 µs/layer (1.134 ms over 30 layers) at 98.7% *union* busy but PE 27% / DVE 30% / ACT 29%
individually — a hand-off chain, never two engines at once on a core. DMA sits at 7%.

The DVE ops are already at peak: `tensor_tensor` on [128, 1024] fp32 at ~690 ns is 1.05
elem/lane/cycle. They can only be made **fewer**, not faster. Step 1 (`Sp = src * eg`, `:479`) and
step 4 (`Sp += Kbeta*bcast`, `:492`/`:495`) collapse into one `nisa.scalar_tensor_tensor`; likewise
`norm_gate.py:50` and `:52` collapse by pre-scaling `gamma`. Numerically identical, low risk,
~85–125 µs.

bf16 recurrent state would halve every DVE op (~300 µs) but changes the numerics of a 40-layer
recurrent carry — not without an accuracy study.

### 6. in_proj LNC sendrecv blocks conv start — 150–300 µs (high risk)
`nki_kernels/deltanet/components/in_proj.py:28` → `qkv_tkg.py:1212` (sendrecv), `:1222` (add).

~9.2 µs of fully-idle wall per DeltaNet layer. Traced in layer 5: last `qkv_tkg.py:1531` ends at rel
20,998, then nothing until a 3,084 ns `PSEUDO_CORE_BARRIER` at rel 27,136, then `:1222` at 30,248,
then conv. Across V, `PSEUDO_CORE_BARRIER` is 978 rows / 585 µs of GpSimd time. Gap classification:
30 gaps averaging 5,506 ns ending at `conv.py:125`, 17 at 2,724 ns ending at `conv.py:150`.

The in_proj H-shard forces a full cross-core round trip before any conv work can begin. Either shard
in_proj by output channel — `conv_qkv_sbuf` already head-shards q/k/v downstream, so each core could
compute only the channels it convolves and skip the sendrecv entirely — or start conv on the
locally-complete channel segments. Touches shard geometry; verify TP=4 × LNC=2 shapes carefully.

### 7. Router weight loaded on the critical path — 150–300 µs
`nki_kernels/moe/vendored/router_topk.py:488` (matmul), `:1922` (weight DMA);
`nki_kernels/moe/components/post_attn_norm.py` → `rmsnorm_tkg.py:385,437,445`.

Post-attn norm + router is 17.4 µs/layer (694 µs total, 9.2% of V) with **DMA at 31% and no engine
above 17%** — a pure dependency chain to produce a 256-wide logit vector for 2 tokens and pick 8
indices. It sits between the attention all-reduce and the expert stream, so every nanosecond here
delays finding 2. The router weight is static per layer and only 1 MB (2.8 µs at peak): prefetch it
during the preceding attention phase, where DMA is 7% busy. Scheduling-only, low risk.

### 8. ACT table thrash in norm_gate — 100–200 µs
`nki_kernels/deltanet/components/norm_gate.py:56` (`silu`), with `:39` `SQUARE` / `:45` `RSQRT`.

1,060 `ACT_TABLE_LOAD` in V, every one exactly 1,283 ns, 680 µs merged-busy per pcore = 9.1% of V.
13.25 loads per layer per core. 182,844 ns (13.4%) of that has no non-scalar engine running.

The traced per-token alternation is `SQUARE, RSQRT, [LOAD] ... [LOAD] SILU, [LOAD] SQUARE+RSQRT,
[LOAD] SILU, ...` — the norm_gate pair costs 2 loads per token per core = 5.13 µs/layer/core =
**154 µs over 30 layers**, landing inside the finding-5 serial chain where ACT is on the critical
path. `z` for all T tokens is available up front from `proj_sb`, so hoist `silu(z)` out of the
per-token loop and compute the whole `[T, W]` block once; `norm_gate_row` takes a pre-computed gate.

The other ~9 loads per layer per core are already in engine shadow (86.6% of table-load time has a
non-scalar engine running).

### 9. XLA glue around the launch — 90–190 µs
`megakernels/qwen3_6_moe/modeling_qwen36_a3b.py:4709` (`_round_token_gen_forward`), `:4687`
(`_round_epilogue`), plus NxDI `model_base.py:2952` (`_tkg_postprocessor`).

**Prologue, 163 µs.** GpSimd is 87.5% busy through D1's first 175 µs, occupied entirely by glue: 120
`DMA_DIRECT2D` (77.9 µs), 60 `DMA_INDIRECT` (62.1 µs), 120 `TENSOR_LOAD` (35.6 µs). Those 188 DMA
instructions move **62,464 bytes — 0.4 GB/s**, pure descriptor overhead against a 616 GB/s roof.
The consequence is hard serialization: `gather_embed_rows`' indirect DMA cannot issue until
138,804 ns, and the first RMSNorm not until 170,433. The draft does nothing for 163 µs.

Reorder so only D1's critical tensors (position ids, cos/sin, mask, MTP cache handles) are produced
before the launch, deferring `_verify_prologue` and `_collect_verify_weights` past it. Python
ordering only — but it helps only if the compiler's schedule follows trace order, so re-profile
rather than assume.

**Epilogue, 145 µs.** Same signature after the kernel finishes: 130 `DMA_DIRECT2D` on GpSimd
(102 µs), 424 `STREAM_SHUFFLE` in NxDI's `_tkg_postprocessor` (62 µs), 89 more `DMA_DIRECT2D`
(55 µs), 120 `COPY_PREDICATED` (54 µs), PE at 0.3%. `_round_epilogue` computes acceptance and
updates the rolling hidden — correctness-critical, so changes need the acceptance gate re-run.

### 10. conv candidate store — ~100 µs
`nki_kernels/deltanet/components/conv.py:238` (`nisa.dma_copy` of `cand_blk` → `conv_cand`).

360 `Sync EVENT_SEMAPHORE` waits in V (12 per DeltaNet layer), mean 1,822 ns, 656 µs total, of which
**209 µs (31.9%) is fully idle**. The DMA moves 2,048 bytes in 1,781 ns = 1.15 GB/s — pure latency.
Batch the T tokens' candidate windows into one DMA per segment (3 instead of 6 per core), or defer
the candidate store past the recurrence so the wait falls in shadow. Local, low risk.

### 11. `all_gather_argmax` issues two 8-byte gathers back to back — 50–80 µs
`nki_kernels/megakernel/collectives.py:172-173`.

Per fold site: `AllGather Ring`, 2 elements fp32, 8 bytes in. D1 measures 26,729 + 22,158 ns with
`cc_trigger_start_delay = 28,873` on the second — it was triggered and then waited for the first to
drain. **54 µs to agree on one token id.** Union-busy in that window is 4.9%. The same pair at V's
seam costs 51.6 µs and gates `embed_compose`, which shows 106,901 ns of wait on `embed.py:61`.

Pack `val` and `idx` into one contiguous `[T, 2]` fp32 tile and issue a single `nccl.all_gather`
with `collective_dim=1`, then de-interleave. This is already the repo idiom — `lm_head.py:224-233`
does exactly this for the LNC exchange. Update the per-rank column stride at `collectives.py:157-163`.
Pays at all three fold sites.

### 12. eh_proj's two serial TP all-gathers — 44–70 µs
`nki_kernels/eh_proj/components/eh_proj.py:156` and `:195`.

Both are `AllGather Ring`, 2,048 B in / 8,192 B out: D1 24,896 + 34,491 ns, D2 22,235 + 24,192 ns.
0.08 GB/s — entirely latency. The eh_proj phase runs at 45.1% (D1) / 52.5% (D2) union-busy and the
idle is these two collectives.

Restructuring row-parallel (all-reduce an 8-byte sum-of-squares pair so both RMSNorms get global
statistics on their local shard, contract locally, all-reduce the result) removes a round trip. This
invalidates the layout-exactness argument in the module docstring at `eh_proj.py:19-23` and changes
`eh_w`'s row/shard contract — re-gate `eh_proj_fwd` before trusting it.

### 13. `head_major_from_tp2013`'s 128-byte DMA batch — 12–20 µs
`nki_kernels/lm_head/components/lm_head.py:92-97`, docstring claim at `:74`.

64 `DMA_DIRECT2D` on the Sync engine, uniformly **128 bytes each**, 40,951 ns of instruction time,
22,704 ns merged wall — 8 KB in 23 µs = 0.35 GB/s. It is exposed: MATMUL counts per 10 µs bucket run
0, 7, 56, 71, 95, 89 against a full rate of ~90, so the head ramps at 0–60% for its first ~35 µs.

The docstring's "G*n_prgs*T ≤ 64 small DMAs, negligible against a ~127 MB/core weight stream" does
not hold at T=1: 23 µs against a 424 µs stream is 5.4%, and it sits at the front where it gates the
pipe. The verify head at T=2 is worse in absolute terms (134 instructions, 44,501 ns merged).

Consecutive `b` write to contiguous destination partitions, so one strided DMA per `(s, t)` can
replace the `G=16` inner loop, cutting 32 DMAs/core to 2. Failing that, hoist the permutation ahead
of the weight prefetch — `output_projection_tkg` starts its weight DMAs ~400 µs earlier. The comment
at `:71-74` deliberately splits partition base from free stride; confirm the DMA engine accepts both
in one descriptor, and gate on `test_lm_head_kernel.py`.

## Per-layer budgets

Segment boundaries are landmark-derived and tile each layer window by construction, so totals are
exact; attribution carries ±5% on GQA (segment overlap) and ±2 µs per DeltaNet boundary.

### Steady-state DeltaNet layer — 195.9 µs mean (n=27), ×30 = 5.978 ms

| segment | ns | % layer | PE | DVE | ACT | any | MB | GB/s | dma% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pre-attn norm + in_proj matmuls | 20,771 | 10.6 | 15 | 40 | 7 | 100 | 6.6 | 318 | 46 |
| in_proj LNC sendrecv reduce | 2,846 | 1.5 | 20 | 10 | 0 | 66 | 0.4 | 263 | 21 |
| conv (incl. residual sendrecv wait) | 23,365 | 11.9 | 24 | 18 | 23 | 62 | 6.5 | 265 | 41 |
| recurrence + gated norm | 37,815 | 19.3 | 27 | 30 | 29 | 99 | 1.5 | 41 | 7 |
| o_proj | 2,328 | 1.2 | 33 | 7 | 27 | 97 | 0.0 | 2 | 1 |
| pre-all-reduce slack | 5,954 | 3.0 | 42 | 3 | 7 | 93 | 0.0 | 8 | 0 |
| **AR_attn (TP all-reduce)** | 16,421 | 8.4 | 0 | 0 | 0 | **7** | 4.0 | 240 | 34 |
| post-attn norm + router | 17,392 | 8.9 | 15 | 12 | 16 | 81 | 3.5 | 200 | 31 |
| **routed experts** | 47,733 | 24.4 | 34 | 12 | 12 | 94 | 26.0 | 545 | 82 |
| shared expert + combine | 3,615 | 1.8 | 0 | 48 | 35 | 99 | 0.1 | 20 | 9 |
| **AR_moe (TP all-reduce)** | 14,412 | 7.4 | 0 | 6 | 0 | **15** | 1.1 | 71 | 15 |
| residual add + tail | 3,255 | 1.7 | 0 | 3 | 6 | 28 | 0.0 | 7 | 3 |

### Steady-state GQA layer — ~155 µs, ×10 = 1.535 ms

| segment | ns | PE | DVE | ACT | any | MB | GB/s | dma% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pre-attn norm + qkv matmuls | 19,564 | 14 | 43 | 11 | 100 | 3.7 | 192 | 27 |
| attention core (qk_norm+rope+attention_tkg) | 29,223 | 14 | 26 | 25 | 80 | 10.4 | 354 | 53 |
| o_proj | 7,921 | 43 | 4 | 11 | 93 | 0.8 | 23 | 11 |
| **AR_attn (TP)** | 14,844 | 0 | 0 | 2 | **8** | 0.7 | 38 | 11 |
| post-attn norm + router | 17,231 | 15 | 13 | 17 | 83 | 1.4 | 79 | 12 |
| **routed experts** | 48,269 | 34 | 11 | 12 | 93 | 25.8 | 535 | 80 |
| shared expert + combine | 3,652 | 0 | 46 | 35 | 96 | 0.0 | 10 | 16 |
| **AR_moe (TP)** | 14,413 | 0 | 6 | 0 | **15** | 1.0 | 66 | 14 |
| residual add + tail | 3,207 | 0 | 3 | 6 | 26 | 0.0 | 4 | 3 |

Segments sum to 160.0 µs vs a 155 µs window — ~4% over-count from segment overlap.

### D1 draft — 890,256 ns (rank 1, skew-free)

| phase | ns | % D1 |
| --- | --- | --- |
| XLA prologue (GpSimd-saturated, draft blocked) | 163,100 | 18.3 |
| eh_proj (AG + 2×RMSNorm + GEMV + AG + tp2013) | 86,900 | 9.8 |
| GQA layer incl. o_proj + TP all-reduce | 86,000 | 9.7 |
| MoE incl. TP all-reduce | 74,500 | 8.4 |
| vocab weight stream | 423,753 | 47.6 |
| argmax tail (post-matmul, 1 DVE partition) | 56,021 | 6.3 |

### D2 replay — 923,823 ns (rank 0)

| phase | ns | % round | union-busy |
| --- | --- | --- | --- |
| blocked on verify vocab head | 532,904 | 5.61 | 94.4 / 98.0 / 42.7 |
| D2 eh_proj | 107,096 | 1.13 | 52.5 |
| D2 GQA (incl. o_proj + AR) | 52,000 | 0.55 | 40.6 |
| D2 MoE (incl. AR) | 68,000 | 0.72 | 87.3 |
| D2 megakernel epilogue/store | 19,000 | 0.20 | 38.5 |
| XLA round epilogue | 144,823 | 1.53 | 92.7 |

## Not worth pursuing

- **The two vocab heads are not defects.** D1's head moves 246.85 MB — exactly
  2 cores × 31,040 × 2,048 × 2 B, no redundant traffic — at 571 GB/s, and the round's peak 20 µs
  bandwidth (663 GB/s) occurs inside that window. It is at ~93% of the demonstrated roof. The only
  lever is bytes: fp8 weight would save ~200 µs per head, but greedy argmax is tie-sensitive and
  `test_lm_head_kernel.py` gates lowest-index tie-break against torch, so quantizing moves which
  token wins near-ties, which moves the accept rate.
- **The 1-column `nc_matmul` in the routed experts.** `routed_experts_nki.py:146/147/183` issue 384
  per core per layer, each with a single moving column, at ~1/128 of PE peak. Irrelevant while that
  phase is 82% DMA-busy — fixing the matmul shape saves nothing until finding 2 is fixed.
- **The 4-byte token-id seam** (`qwen36_round_megakernel.py:202-203`): 4 instructions, 238 ns.
- **The three `core_barrier`s** in the round epilogue (`qwen36_round_megakernel.py:306-308`): 5,138 ns
  of instruction time. Their `evt_wait_time_ns` (87k/43k/59k) is waiting on the peer core's real
  work, not barrier overhead.
- **Conv-tap transposes.** `conv.py:125` and `:150` (24 + 18 per layer) are the only transposes with
  a clean fix — pre-permuting `dn_conv_weight` host-side — worth maybe 60–90 µs, and it changes a
  checkpoint layout. Everything else in the transpose set is dependency-shadowed.
- **SBUF/PSUM pressure.** `SbufUsage.avg_util_percent` peaks at 0.164 over V; `qSyncSpillReload0`
  moves 2.1 MB in all of V. Nothing spills in a way that costs time.
- **DMA engine imbalance.** All 32 engines within 3 points (38–42% mean, 40–46% max).
- **Unattributed instructions.** 2.3% of the round has NULL `nki_source_location`; 96% of their
  duration is `EVENT_SEMAPHORE` and `WRITE` — synchronization, not a blind spot.

## Accounting

D1 and D2 are 100% accounted at phase level. V is 100% accounted at segment level. What remains
undecomposed: the ~6.6 µs of fixed latency inside each nccl mesh all-reduce (finding 1), and the
instruction-level critical path inside the 37.8 µs recurrence segment (finding 5), where all three
engines sit near 30% and no single one is the bottleneck.
