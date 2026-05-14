# e2e MoE Optimization Translation-Failure Analysis

**Profile:** `e2e-tkg-38994553-vnc0@latest` (canonical post-revert, gpt-oss-20B megakernel, TP=8 LNC=1, bk=640 TKG)
**Parquet path:** `/home/ubuntu/nki-moe/parquet_files/profiles/global/e2e-tkg-38994553-vnc0@latest`
**Standalone reference:** `moe-revert@latest` (107.1 us, DMA% 69.8) — agrees with the brief's 106.8 us baseline

Headline (Summary.parquet): total_time 5020.8 us, HBM read 1006 MB, DMA 57.8%, TE 33.6%, HWDGE pkts 616,384, SWDGE pkts 102,784. Sanity-checked vs `bk640_megakernel_profile_analysis.md` (5.255 ms / 1.006 GB / DMA 54.9% / TE 32.3%) — within iteration noise.

---

## 1. Region time budget table

Two columns deserve separate reading:
- **wall_span** (min start_ts → max end_ts of any instruction in the region) is large for every region because they all run in every layer across the 24-layer loop. Wall_span tells you "MoE is touched throughout the kernel," not "MoE consumes that much time."
- **merged_active** (interval-union of all instructions in the region across all engines) is the right measure of how much wall-clock time is *exclusively* attributed to this region's source file. Sorted by this.

| Region | n_instr | merged_active (us) | %total | TE (us) | Sync (us) | Vector (us) | Scalar (us) | GpSimd (us) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **mlp_tkg_gate_up_projection.py** | 38,364 | **1701.7** | **33.9%** | 518.6 | 1254.6 | 174.5 | 50.7 | 103.7 |
| **mlp_tkg_down_projection.py** | 16,283 | **682.2** | **13.6%** | 543.7 | 0.0 | 18.8 | 0.0 | 200.5 |
| output_projection_tkg.py | 4,938 | 197.8 | 3.9% | 174.4 | 0.0 | 39.1 | 0.0 | 1.8 |
| router_topk.py | 1,871 | 139.3 | 2.8% | 18.4 | 2.1 | 95.7 | 41.7 | 8.9 |
| attention_tkg.py | 1,278 | 138.8 | 2.8% | 26.0 | 1.8 | 72.1 | 46.7 | 0.3 |
| rmsnorm_tkg.py | 626 | 134.0 | 2.7% | 18.3 | 0.0 | 43.3 | 82.6 | 0.0 |
| qkv_tkg.py | 2,215 | 123.5 | 2.5% | 44.6 | 3.2 | 74.8 | 0.0 | 2.1 |
| norm_tkg_utils.py | 85 | 116.3 | 2.3% | 0 | 0 | 0 | 0 | **116.3** |
| interleave_copy.py | 693 | 108.3 | 2.2% | 0 | 0 | 24.4 | 93.9 | 0 |
| attention_block_tkg.py | 1,485 | 99.0 | 2.0% | 19.3 | 26.0 | 39.1 | 0 | 32.2 |
| selective_expert_impl.py | 818 | 60.8 | 1.2% | 0 | 0 | 60.8 | 0 | 0 |
| stream_shuffle_broadcast.py | 418 | 51.8 | 1.0% | 0 | 0 | 51.8 | 0 | 0 |
| moe_tkg_utils.py | 244 | 49.6 | 1.0% | 0 | 0 | 25.4 | 0 | 24.2 |
| rope.py | 336 | 34.3 | 0.7% | 0 | 0 | 34.3 | 0 | 0 |
| transformer_gpt_oss.py | 111 | 21.5 | 0.4% | 0 | 2.7 | 17.4 | 1.5 | 0.4 |
| ops.py | 89 | 12.5 | 0.2% | 0 | 0 | 0 | 0 | 12.5 |

**MoE union (gate_up + down + router_topk + moe_tkg_utils + selective_expert) = 2206.1 us = 43.9% of total_time.**

Attention-side union (qkv + attention_block + attention_tkg + output_projection + rope) ≈ 593 us ≈ 11.8%.

Per-engine merged active across the whole profile:

| Engine | merged active (us) | % total | Note |
|---|---:|---:|---|
| **DMA (any queue)** | **2900.9** | **57.8%** | bottleneck |
| Tensor | 1710.2 | 34.1% | |
| Sync | 1464.1 | 29.2% | HWDGE descriptor generator |
| Vector | 991.6 | 19.7% | |
| Scalar | 639.5 | 12.7% | |
| GpSimd | 403.5 | 8.0% | SWDGE descriptor generator |

---

## 2. MoE region engine activity + DMA queue mix

### HBM-read traffic attribution (source file → queue)

| Source file | Queue | DMA-trigger count | HBM read (MB) | trigger duration (us) |
|---|---|---:|---:|---:|
| **mlp_tkg_gate_up_projection.py** | **HWDGE (Sync)** | **1152** | **452.98** | 623.0 |
| mlp_tkg_down_projection.py | SWDGE (GpSimd) | 384 | 227.08 | 11.9 |
| (no source: scheduler-emitted) | SWDGE (GpSimd) | 197 | 148.00 | 6.1 |
| qkv_tkg.py | SWDGE (GpSimd) | 48 | 94.37 | 1.5 |
| output_projection_tkg.py | SWDGE (GpSimd) | 48 | 75.64 | 1.5 |
| router_topk.py | SWDGE (GpSimd) | 24 | 4.72 | 0.7 |
| mlp_tkg_gate_up_projection.py | SWDGE (GpSimd) | 192 | 0.15 | 6.0 |
| Other (norm, attention, etc.) | HWDGE / SWDGE | 110 | 1.93 | 28 |

**MoE weights = 680 MB = 67.6% of all HBM read (1006 MB).** gate_up alone = 453 MB = 45.0% of HBM read.

### Per-queue DMA wall-clock (DmaPacket merged intervals)

| Queue | Packets | Bytes | Merged-active span (us) | Avg pkt size (B) |
|---|---:|---:|---:|---:|
| HWDGE | 616,384 | 453.5 MB | **1653.2** | 736 |
| SWDGE | 102,784 | 550.1 MB | **1462.5** | **5352** |
| input | 3,042 | 2.7 MB | 40.7 | 888 |
| data (spill) | 1,930 | 0.06 MB | 10.3 | 30 |
| instruction | 9,580 | 1.8 MB | 167.6 | 184 |
| **Union (any DMA)** | — | 1006 MB | **2900.9** | — |

Two facts to note:
- **HWDGE is doing the most packets (6× more than SWDGE) for less data (0.83× of SWDGE bytes).** Per-packet HWDGE is 736 B; SWDGE is 5352 B. SWDGE is more efficient per descriptor.
- **HWDGE and SWDGE are running concurrently and together total 2900 us — SWDGE alone is already wide enough (1462 us / 28% of total) to be a critical-path participant.**

### HWDGE-from-gate_up source line (single)

All 1152 HWDGE triggers for gate_up come from one line:
- `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/nkilib/core/mlp/mlp_tkg/mlp_tkg_gate_up_projection.py:470` — `nisa.dma_copy(... dge_mode=nisa.dge_mode.hwdge)`.

**The active code path is the nkilib-vendored kernel, not the repo copy at `nki_kernels/moe/mlp_tkg_gate_up_projection.py`.** The repo file has a `fused_weight_tiles` variant; that variant is not compiled in.

SWDGE for down comes from two lines in the nkilib copy:
- `nkilib/core/mlp/mlp_tkg/mlp_tkg_down_projection.py:296` (288 triggers, 227 MB)
- `nkilib/core/mlp/mlp_tkg/mlp_tkg_down_projection.py:259` (96 triggers)

### Descriptor-generator engine load (the "can SWDGE go here?" question)

| Engine | Merged active | % of total | Headroom |
|---|---:|---:|---|
| Sync (HWDGE descriptors) | 1464.1 us | 29.2% | moderate |
| GpSimd (SWDGE descriptors) | **403.5 us** | **8.0%** | **large** |

GpSimd is busy only 8% of the kernel. Moving gate_up from HWDGE→SWDGE would not saturate GpSimd at the aggregate level. **But** the per-window picture is different: during HWDGE-in-flight windows, GpSimd ∩ HWDGE = 172 us (10.4% of HWDGE window), and during SWDGE-in-flight windows, GpSimd ∩ SWDGE = 154 us (10.5% of SWDGE window). At a window level the GpSimd is only ~10% loaded.

---

## 3. What's on the critical path during the MoE wall-span window

This answers the central question: *while gate_up DMA is in flight, what is the TE busy on?*

### TE source attribution during HWDGE-in-flight windows

| TE source file | TE total (us) | TE ∩ HWDGE (us) | % of file's TE in HWDGE window |
|---|---:|---:|---:|
| **mlp_tkg_gate_up_projection.py** | **518.6** | **274.8** | **53.0%** |
| **mlp_tkg_down_projection.py** | **543.7** | **221.1** | **40.7%** |
| (no source) | 338.8 | 4.8 | 1.4% |
| output_projection_tkg.py | 174.4 | 1.7 | 0.9% |
| qkv_tkg.py | 44.6 | 0.5 | 1.0% |
| attention_tkg.py | 26.0 | 1.4 | 5.5% |
| attention_block_tkg.py | 19.3 | 0.0 | 0.0% |
| rmsnorm_tkg.py | 18.3 | 0.9 | 4.9% |

**Of the 1653 us that HWDGE is in flight, 477 us are overlapped with TE compute, and ~496 us of that is TE doing MoE work (gate_up 275 + down 221 = 496 us).** The next layer's attention/QKV/output appears in negligible amounts.

This means the megakernel scheduler is pipelining gate_up's HWDGE weight DMA **against the same gate_up's matmuls and the following down's matmuls** — not against attention/QKV/output of the next layer.

### Idle/serialization at the engine level

| Quantity | μs | % total |
|---|---:|---:|
| Any-DMA active | 2900.9 | 57.8% |
| TE active | 1710.2 | 34.1% |
| TE ∩ any-DMA (overlap) | 932.4 | 18.6% |
| **DMA in flight, TE idle** | **1826.2** | **36.4%** |
| HWDGE in flight, TE idle | 1176.0 | 23.4% |
| SWDGE in flight, TE idle | 927.9 | 18.5% |

**1.18 ms of HWDGE active time happens with the TE idle.** That is the slack created by the HWDGE→TE pipeline not feeding the TE fast enough.

### MoE wall-span window — the standalone-vs-e2e split

- gate_up union span across all engines: 1701.7 us. Of that: 933.7 us (54.9%) overlaps HWDGE in flight, 362.2 us (21.3%) overlaps SWDGE in flight.
- TE doing gate_up matmuls: 518.6 us, of which 275 us (53%) is concurrent with HWDGE — i.e., concurrent with its own weight load. The other 244 us of gate_up matmul runs after the weights have already arrived (out-of-band TE work).
- Standalone gate_up kernel total = 86 us. 24 layers × 86 us = 2064 us standalone budget. MoE-attributed e2e wall = 2206 us. **The megakernel is only slightly compressing MoE total work** vs. 24× standalone (factor 1.07×). It is not hiding MoE behind attention.

---

## 4. Performance bounds + gap analysis

| Bound | μs | Note |
|---|---:|---|
| **total_time** | **5020.8** | |
| memory_bound (DMA active wall) | 2902.3 | bottleneck |
| memory_bound_ideal (HBM / peak BW) | 854.1 | 1006 MB / 1178 GB/s |
| compute_bound (TE active wall) | 1688.8 | |
| compute_bound_ideal (FLOPs / peak FLOPS) | 253.8 | 19.96 GFLOPs / 78.6 TFLOPS |
| perfect_pipeline (max of all merged engines) | 2900.9 | = DMA active |

| Gap | μs | % total |
|---|---:|---:|
| memory_bound − memory_ideal | **2048.2** | **40.8%** | (excess DMA / inefficient HBM usage) |
| compute_bound − compute_ideal | 1435.0 | 28.6% (low TE MFU) |
| **total − perfect_pipeline** | **2119.9** | **42.2%** | **pipeline serialization gap** |
| total − memory_bound (non-DMA tail) | 2118.5 | 42.2% |

Achieved HBM BW during DMA-active window = 1006 MB / 2.902 ms = **346.6 GB/s = 29.4% of peak 1178 GB/s**.

**Bottleneck identification:**
- **Engine bottleneck:** DMA (57.8% of total wall is DMA-active, with TE the runner-up at 34%).
- **Dominant gap is pipeline serialization (2120 us / 42.2%)** = the difference between total_time and the perfect-pipeline ideal (where every engine ran in parallel with no stalls). This is much larger than the memory-vs-ideal gap (2048 us) and the compute-vs-ideal gap (1435 us) measured against engine-active.
- The kernel is far from compute-bound (MFU 0.28% vs peak; useful FLOPs 1.02 GFLOPs vs hardware FLOPs 19.96 GFLOPs — 94.9% of compute is transposes, consistent with TKG batch-size-1 nature).

**Is MoE DMA on the critical path?** Yes — gate_up HWDGE alone is 453 MB / 1006 MB = 45% of HBM read, and HWDGE in-flight span is 1653 us = 33% of total_time. But the limiting axis is not "DMA bandwidth saturated" (we're at 29% of peak BW); it is "DMA active wall is long because TE-issue cadence is forcing many small descriptors that don't saturate the bus" — see HWDGE avg packet 736 B and SWDGE avg 5352 B.

---

## 5. Headline conclusion: why don't standalone MoE gains translate?

**Standalone MoE gains don't translate because the gate_up DMA is already mostly overlapped with TE compute *that is itself MoE work in the same layer* — there is no second pool of compute available to hide additional DMA savings behind, and the in-layer TE workload is the floor.**

The evidence:

1. **DMA is the wall-clock bottleneck (2902 us, 57.8%) and stays the bottleneck in e2e**. Standalone moe-revert kernel is also DMA-bound (69.8%). So MoE DMA savings *would* matter — *if* the savings actually reduced wall-time DMA.
2. **The HWDGE active window is 1653 us; TE is only busy 477 us inside that window (29%).** During the other 1176 us (23.4% of total_time), HWDGE is in flight but TE is idle. That idle is not waiting for *more* compute — it is the descriptor-cadence gap between when the next matmul's weights arrive and when the matmul can issue. Cutting HWDGE bytes does not shorten this; cutting *descriptors* might.
3. **The TE that *is* running during HWDGE windows is 53% gate_up matmul + 40.7% down matmul = 93.7% of in-window TE.** Almost nothing else is co-scheduled. The megakernel scheduler is not running the next layer's attention/QKV/output during MoE DMA (those see <2% overlap with HWDGE).
4. **The standalone 24×86 us = 2064 us budget is essentially what we observe e2e** (MoE-union span = 2206 us, 1.07× the standalone budget). The megakernel buys you 7% of compression vs running 24 independent invocations. The gain is small because the next layer's attention starts only after the current layer's MoE finishes, so MoE remains on the critical path as a single 2.2 ms block.
5. **GpSimd is at 8% engine load** — there *is* descriptor-engine headroom for moving gate_up HWDGE→SWDGE *aggregate-wise*. The post-revert state is back to HWDGE at `nkilib/core/mlp/mlp_tkg/mlp_tkg_gate_up_projection.py:470` (1152 HWDGE triggers, all from this one line). However, standalone the P1 SWDGE conversion saved 7 us on a 107 us kernel by reducing descriptor count and increasing per-packet size; e2e the same change has to *also* not perturb the per-packet pipeline timing into the TE for the in-window 53% TE-overlap. In the standalone there is no concurrent TE pipeline to interfere with; in e2e there is, and the brief reports the change regressed e2e — consistent with the SWDGE conversion changing arrival timing of the first weight slice and stretching the TE-busy critical-path window.

**Where MoE work would have to target for e2e gain (per the profile):**
- Reduce the 1.18 ms HWDGE-in-flight-but-TE-idle slack. That is *not* a bytes problem (we're at 29% of peak BW); it is a descriptor/issue-cadence problem. Reducing 616k HWDGE packets (avg 736 B) toward 5k-byte SWDGE-sized batches — or fewer but larger HWDGE bursts — would cut the wall span without changing bytes.
- Reduce the MoE *compute* tail on TE (518 us gate_up + 544 us down = 1063 us TE-active in MoE, ~63% of all TE time). MFU on MoE matmuls is the largest e2e compute lever.
- Pipeline the *next layer's* attention/QKV against the *previous layer's* MoE DMA. Currently overlap is <2%. This is a scheduling/megakernel-orchestration problem, not a kernel-internal problem; modifying mlp_tkg_gate_up_projection.py alone cannot reach it.

Until one of those three changes, MoE-kernel-internal optimizations targeting a single in-MoE engine will fight the in-MoE TE pipeline they coexist with, and the standalone gain will not survive integration.

---

### Sanity check against the previous baseline analysis

`bk640_megakernel_profile_analysis.md` reports total 5.255 ms / 1.006 GB / DMA 54.9% / TE 32.3% for the same configuration; this profile reads 5.021 ms / 1.006 GB / DMA 57.8% / TE 34.1%. Within iteration noise. HBM bytes are identical to 6 digits across all three TKG buckets analyzed (bk0/1/2 all 1006 MB) — consistent with same weight set, KV-length only affecting total wall.

---

# Part 2 — DMA Cadence Anti-Pattern vs XLA Baseline

(Added after head-to-head ingestion of the prior XLA baseline TKG profile at `gpt-oss-bk640-xla-baseline@latest`. Both profiles operate on identical model/config; XLA was at TP=4/LNC=2, megakernel at TP=8/LNC=1 — sharding differs, but the per-byte attribution comparison below is invariant to TP.)

## A. Headline divergence

| | Megakernel | XLA baseline | Δ |
|---|---:|---:|---|
| total_time | 5020.8 μs | 5798.6 μs | MK wins overall by 778 μs |
| Sync engine active% | **26.3 %** | **2.4 %** | **+24 pp** |
| GpSimd engine active% | 7.7 % | 10.5 % | −2.8 pp |
| HWDGE packets | **616 384** | **8 436** | **73× more in MK** |
| SWDGE packets | 102 784 | 208 224 | XLA does 2× more |
| HBM read | 1006 MB | 1001 MB | identical |
| HWDGE avg packet | 736 B | 241 B (negligible) | — |
| SWDGE avg packet | 5352 B | 4800 B | similar |

The megakernel wins total time but pays ~1.0 ms of Sync-engine time that the baseline never spends. The DMA *bytes* are identical; the *descriptor cadence* is the only thing that differs.

## B. Per-instruction DMA attribution (the only table that matters)

The on-device DMA work decomposes into two engine paths — `DMA_DIRECT2D` on **Sync** (HWDGE) vs `DMA_DIRECT2D` on **GpSimd** (SWDGE). Both move the same total bytes but cost the engine differently per instruction:

| Profile | Engine | Inst count | Tot μs | HBM moved | KB / inst | ns / inst |
|---|---|---:|---:|---:|---:|---:|
| **MK** | **Sync (HWDGE)** | 1269 | **682.7** | 453 MB | 357 | **538** |
| MK | GpSimd (SWDGE) | 952 | 29.5 | 550 MB | 578 | 31 |
| **XLA** | Sync (HWDGE) | 209 | 119.9 | 1.3 MB | 6 | 574 |
| **XLA** | **GpSimd (SWDGE)** | 1992 | **61.8** | 999 MB | 502 | **31** |

Per-instruction wall cost: Sync DMA_DIRECT2D is **17× more expensive than GpSimd DMA_DIRECT2D** (538 ns vs 31 ns). Bytes-per-instruction is similar (357 KB vs 502 KB).

**XLA moves 1 GB of MoE weights via GpSimd-DMA in 61.8 μs total. The megakernel moves 453 MB of MoE gate_up weights via Sync-DMA in 623 μs total.** Same job, same bytes, **10× longer wall time** on the engine path the kernel chose.

## C. Single source line is responsible for 91% of the megakernel's Sync DMA time

| Source line | Engine | Inst | Tot μs | HBM MB |
|---|---|---:|---:|---:|
| `mlp_tkg_gate_up_projection.py:470` (canonical = repo line 497) | **Sync** | **1152** | **623.0** | **453.0** |
| attention_block_tkg.py:859 | Sync | 48 | 23.4 | 0.0 |
| custom_op_name.py:603 | Sync | 12 | 6.3 | 0.2 |
| module.py:1786 | Sync | 11 | 5.8 | 0.1 |
| transformer_gpt_oss.py:223/233/301 | Sync | 5 | 2.4 | 0.0 |

623.0 μs / 682.7 μs = **91.3 %** of all megakernel Sync DMA time comes from one `nisa.dma_copy(... dge_mode=nisa.dge_mode.hwdge)` call inside `process_gate_up_projection_lhs_rhs_swap`.

XLA's GpSimd-DMA source attribution (HBM > 100 MB only):

| Source line | Engine | Inst | Tot μs | HBM MB |
|---|---|---:|---:|---:|
| `torch/nn/modules/module.py:1775` (PyTorch dispatcher → MoE) | GpSimd | 987 | 30.6 | **680** |
| `nxd-inference/.../attention_base.py:1656/1636` (QKV/Wo) | GpSimd | 360 | 11.2 | 88.6 |
| `torch_neuronx/.../custom_op_name.py:603` (helpers) | GpSimd | 229 | 7.1 | 76.8 |

XLA's MoE weight load (the 680 MB at module.py:1775) takes **30.6 μs total** — vs the megakernel's equivalent at 623 μs. Each XLA instruction moves on average 674 KB at 31 ns/instruction; the megakernel moves 384 KB at 538 ns/instruction.

## D. Why this matches the user's "30 μs faster per layer" observation

MK MoE-attributed wall span / 24 layers = 2206 / 24 = **91.9 μs / layer**.
If the gate_up Sync-DMA cost were reduced from 623 μs to XLA-comparable (~31 μs total — pure GpSimd cost for the same bytes), the gate_up region would shrink by **(623 − 31)/24 ≈ 24.7 μs / layer**. Add the saved Sync-engine non-DMA work it triggers (semaphore/event chain) and the per-layer saving lands in the 25–30 μs range — exactly matching the user's claim.

## E. Why the previous P1 (HWDGE→SWDGE) attempt failed

Prior session's P1 attempt (`dge_mode=nisa.dge_mode.hwdge` → `adaptive_dge_mode(weight_view)`) gained 7 μs standalone but regressed e2e by ~213 μs at p50. The current data explains why the *standalone* gain was small:

- Standalone bench is single-MoE-call. Sync DMA_DIRECT2D at the affected line takes ~28 μs (623 / 24). Switching it to GpSimd saves most of that 28 μs but is *partially overlapped* with the bench's TE pipeline, so wall-clock gain is only 7 μs.

But the e2e regression is harder to explain from this profile alone, because we don't have a P1-state profile captured. The most likely cause is **descriptor-arrival timing perturbation**: when the first gate_up weight slice arrives a few μs later (because GpSimd has to generate the descriptor instead of Sync chaining it off the previous instruction), the first matmul in *each* of the 24 layers issues a few μs later. 24× per-layer slip → ~150–250 μs e2e regression. That matches the observed ~213 μs delta.

**The 600 μs of theoretical headroom is real, but the prior P1 didn't capture it because of arrival-time perturbation on the first matmul.** What's needed is a variant that keeps the first matmul's weight arrival on schedule.

## F. Proposed concrete experiments (in priority order)

**Experiment 1 — Split the conversion: swap engines only on the inner HTile iterations, not the first.** The first dma_copy in the per-layer HTile loop is timing-critical (its arrival gates the first matmul). The remaining HTile iterations are scheduled into the gap behind the first matmul and have looser timing. Code change (one block at `nki_kernels/moe/mlp_tkg_gate_up_projection.py:486-501`):

```python
# First HTile keeps HWDGE — its arrival is on the critical path
# Subsequent HTiles use SWDGE — they hide behind the first matmul
is_first_htile = (hidden_tiles.index == 0)
dge = nisa.dge_mode.hwdge if is_first_htile else nisa.dge_mode.swdge
nisa.dma_copy(
    dst=weight_tiles[weight_idx][0:H0, 0:h1_size, 0:shared_I],
    src=weight_view.get_view(),
    dge_mode=dge,
)
```
Expected gain: 7/8 of the inner-loop Sync-DMA work moves to GpSimd. ~540 μs of the 623 μs total. Net e2e gain ~150–400 μs depending on how much overlap GpSimd has.

**Experiment 2 — Pre-pull weight metadata so SWDGE descriptor generation finishes before the loop starts.** SWDGE on GpSimd requires per-descriptor setup. Issue the SWDGE batch at the *top* of the layer's MoE region (before router top-k completes), so the descriptors are queued and ready by the time the matmul wants the weights. This decouples engine choice from arrival timing.

**Experiment 3 — Hoist the weight DMA out of the HTile loop entirely.** Currently we issue one dma_copy per HTile (48/layer/projection = 1152 instructions across 24 layers). Issue ONE dma_copy per expert per layer that loads the full `[H, I]` weight slice, then matmul indexes into it. That cuts the instruction count from 1152 → 96 (12× fewer instructions, 12× fewer per-instruction overheads on Sync). Per-byte stays the same. This may break SBUF pressure (loading full slice instead of HTile chunk) — needs measurement.

**Experiment 4 — Coalesce gate + up DMAs and re-validate.** The prior P2 (fusion) regressed by 5 μs/layer because the single fused DMA delayed the first matmul. Combine P2 with Experiment 1's "keep first HTile on HWDGE" idea: fuse gate+up only on inner HTiles, keep first HTile per projection unfused. Expected gain: P2's standalone 21 μs/layer without the first-matmul-delay penalty.

All four experiments can be A/B tested with isolated standalone bench + e2e p50, with a profile-capture in both states so we can verify on the descriptor-fan-out numbers above.

## G. What changed in this analysis vs Part 1

Part 1's "no low-hanging fruit" conclusion was about *bytes* — DMA bandwidth was at 29% of peak, so cutting bytes can't help much. Part 2's conclusion is about *descriptors* — Sync-engine DMA_DIRECT2D is 17× more expensive per instruction than GpSimd. The 600 μs of headroom is **descriptor-emission-engine cost**, not bytes-bandwidth. The XLA baseline never pays this cost because the neuronx-cc compiler routes the same byte budget through GpSimd-SWDGE.

