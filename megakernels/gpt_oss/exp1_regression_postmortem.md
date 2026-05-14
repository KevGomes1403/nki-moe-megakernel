# Experiment-1 (HWDGE/SWDGE split) — e2e regression postmortem

**Profiles compared**
- canonical (post-revert, HWDGE-only): `/home/ubuntu/nki-moe/parquet_files/profiles/global/e2e-tkg-38994553-vnc0@latest`
- exp1 (1st HTile HWDGE, HTiles 1..7 SWDGE) — bk0 TKG, vnc_0: `/home/ubuntu/nki-moe/parquet_files/profiles/global/e2e-exp1-tkg-140480@latest`

The exp1 NEFF was the cached `--skip-compile` artifact (the agent's recompile after the bench but before the code revert). Three TKG NEFFs were ingested for this run (graph IDs 140480414712667, 517727976607229, 537895261305882); per `total_time` they correspond to bk0 / bk1 / bk2 = 5637 / 5797 / 6107 us. We analyze bk0 (smallest = matches steady-state TKG used at p50).

## Headline diff (table)

| Metric                                | canonical (HWDGE only) | exp1 (split) | delta             |
| ------------------------------------- | ----------------------:| ------------:| -----------------:|
| `total_time`                          | 5020.8 us              | 5637.3 us    | **+616.5 us**     |
| `dma_active_time_percent`             | 57.81 %                | 51.76 %      | -6.05 pp          |
| `tensor_engine_active_time_percent`   | 33.64 %                | 30.37 %      | -3.27 pp          |
| `sync_engine_active_time_percent`     | 26.28 %                | **5.25 %**   | **-21.03 pp**     |
| `gpsimd_engine_active_time_percent`   | 7.70 %                 | **16.20 %**  | **+8.50 pp**      |
| `hardware_dynamic_dma_packet_count`   | 616 384                | 109 504      | -506 880 (-82 %)  |
| `software_dynamic_dma_packet_count`   | 102 784                | 210 304      | +107 520 (+105 %) |
| `hardware_dynamic_dma_active_time`    | 1655.0 us              | 294.5 us     | **-1360.5 us**    |
| `software_dynamic_dma_active_time`    | 1463.9 us              | 2556.4 us    | **+1092.5 us**    |
| `hbm_read_bytes`                      | 1006.0 MB              | 1006.0 MB    | 0                 |
| `matmul_instruction_count`            | 34 658                 | 34 658       | 0                 |

The DMA traffic itself is identical; only the engine that issues it shifted.

## Per-engine + per-source diff

### Interval-merged active time per engine (whole trace)

| engine  | canonical (us) | exp1 (us) | delta (us)     |
| ------- | -------------: | --------: | -------------: |
| Tensor  | 1710.2         | 1734.9    | +24.7          |
| Sync    | 1464.1         | 527.4     | **-936.7**     |
| GpSimd  | 403.5          | 942.5     | **+539.0**     |
| Vector  | 991.6          | 1003.5    | +11.9          |
| Scalar  | 639.5          | 638.7     | -0.8           |
| **net** |                |           | **-361.9**     |

Engine-time bookkeeping shows a net DECREASE of 362 us, yet wall grew +617 us. The lost time is wall-clock idle — engines waiting on each other, not engines doing more work.

### MoE-region wall span and merged engine occupancy

MoE-region wall (gate_up + down + selective_expert + router_topk):

| MoE engine occupancy   | canonical   | exp1        | delta      |
| ---------------------- | -----------:| -----------:| ----------:|
| MoE wall span          | 4392.6 us   | 5013.0 us   | **+620.4** |
| Tensor merged          | 1300.3 us   | 1325.1 us   | +24.8      |
| Sync merged            | 1435.9 us   | 500.4 us    | -935.5     |
| GpSimd merged          | 378.0 us    | 918.7 us    | +540.7     |
| Vector merged          | 732.4 us    | 745.6 us    | +13.2      |
| Scalar merged          | 318.1 us    | 318.8 us    | +0.7       |

The +620 us of e2e regression is **entirely inside the MoE region**. Non-MoE instruction time changed by +99 us Sync, +4 us GpSimd, ~0 elsewhere.

### Top-line per-source attribution shift (gate_up path)

The gate_up DMA op moved from Sync to a mix of GpSimd + (1/8) Sync:

| source line                                                  | canonical                  | exp1                              |
| ------------------------------------------------------------ | -------------------------- | --------------------------------- |
| `nkilib/.../mlp_tkg_gate_up_projection.py:470` (canon source) | Sync 1254.6 us             | — (path replaced)                 |
| `nki_kernels/moe/mlp_tkg_gate_up_projection.py:506`           | —                          | **GpSimd 541.4 us + Sync 218.1 us** |

Within the **gate_up** region:
| engine, opcode              | canonical    | exp1         | delta    |
| --------------------------- | -----------: | -----------: | -------: |
| Sync DMA_DIRECT2D           | 623.0 us     | 109.0 us     | **-514** |
| Sync TENSOR_LOAD            | 381.6 us     | 69.7 us      | -311.9   |
| Sync MOVE                   | 80.4 us      | 0 us         | -80.4    |
| Sync ALU_OP                 | 148.9 us     | 22.0 us      | -126.9   |
| **Sync subtotal in gate_up**| **1233.9 us**| **200.7 us** | **-1033** |
| GpSimd TENSOR_LOAD          | 69.6 us      | 392.9 us     | +323.3   |
| GpSimd ALU_OP               | 20.1 us      | 142.0 us     | +121.9   |
| GpSimd MOVE                 | 8.9 us       | 53.1 us      | +44.2    |
| GpSimd DMA_DIRECT2D         | ~0           | 35.7 us      | +35.7    |
| GpSimd EVENT_SEMAPHORE      | 0            | 17.3 us      | +17.3    |
| **GpSimd subtotal in gate_up**| ~99 us     | **641.0 us** | **+542** |

The standalone bench prediction (Sync gate_up wall 623 → ~78 us with 1/8 staying) tracks: Sync DMA_DIRECT2D dropped to **109 us** in gate_up. The Sync work moved cleanly to GpSimd.

### TE pipeline pressure

| Tensor-engine wait pattern (gate_up)        | canonical   | exp1         | delta     |
| ------------------------------------------- | ----------: | -----------: | --------: |
| LDWEIGHTS evt_wait_time total                | 2152.4 us   | 2397.8 us    | **+245.4** |
| LDWEIGHTS with wait > 0                      | 13262 / 14016 | 13530 / 14016 | +268    |
| MATMUL evt_wait_time total                   | 16.2 us     | 16.9 us      | +0.7      |

TE is waiting more on weight DMAs to arrive (+245 us on LDWEIGHTS alone in gate_up). Whole-trace matmul-with-wait count grew 21772 → 22469 (+697).

### Region-by-region wall spans

| region (source-file match) | canonical span | exp1 span | delta   |
| -------------------------- | -------------: | --------: | ------: |
| gate_up                    | 4304.1 us      | 4605.9 us | +301.8  |
| down                       | 4271.9 us      | 4568.1 us | +296.2  |
| selective_expert           | 4325.5 us      | 4627.8 us | +302.3  |
| router_topk                | 4262.8 us      | 4867.7 us | +604.9  |
| rmsnorm                    | 4212.1 us      | 4819.7 us | +607.6  |
| attention_block_tkg        | 4234.3 us      | 4840.0 us | +605.7  |
| qkv                        | 4230.5 us      | 4836.2 us | +605.7  |
| out_proj                   | 4236.0 us      | 4841.7 us | +605.7  |

Non-MoE regions look "slower" only because their wall span subsumes the MoE region. Non-MoE instruction time itself is unchanged. The actual slowdown is concentrated in the MoE region.

### The smoking gun: DMA engine parallelism inside MoE window

Per-microsecond occupancy of the two dynamic DMA queues during the MoE window:

| MoE-window state           | canonical               | exp1                    |
| -------------------------- | ----------------------: | ----------------------: |
| Both HWDGE and SWDGE busy  | **810 us (18.4 %)**     | **298 us (5.9 %)**      |
| HWDGE only                 | 1454 us (33.1 %)        | 205 us (4.1 %)          |
| SWDGE only                 | 726 us (16.5 %)         | **2717 us (54.2 %)**    |
| Neither                    | 1403 us (31.9 %)        | 1794 us (35.8 %)        |
| HWDGE merged wall          | 1646.3 us               | 287.9 us                |
| SWDGE merged wall          | 1152.4 us               | 2243.8 us               |
| Union (HW ∪ SW)            | 2442.7 us (55.6 % wall) | 2454.0 us (49.0 % wall) |

The union of DMA-active time **barely changed (+11 us)**. What changed is the parallelism: canonical pumped both DMA engines simultaneously for **810 us**; exp1 only does so for **298 us** (-512 us of overlap). The first-HTile-only-on-HWDGE schedule does not produce enough HWDGE work to overlap the bulk SWDGE traffic — SWDGE runs essentially serial and the saved Sync time isn't recovered as wall-clock improvement.

## Which hypothesis holds (cite specific numbers)

- **H1 — GpSimd became the bottleneck: PARTIAL.** GpSimd merged-active grew 378.0 → 918.7 us in MoE (+540), and its instruction count more than doubled (gpsimd_engine_instruction_count rose roughly in line). But **GpSimd is at only 18.3 % of MoE wall** (918.7 / 5013) — far from saturated. GpSimd alone is not the bottleneck.

- **H2 — DMA arrival timing shifted: SUPPORTED.** LDWEIGHTS wait time in gate_up grew 2152.4 → 2397.8 us (+245.4 us), and the wait>0 count grew (+268 LDWEIGHTS, +697 matmuls trace-wide). The first weight tile still rides HWDGE, but subsequent tiles ride the now-saturated SWDGE — so the second/third weight LDWEIGHTS of each gate_up tile waits longer.

- **H3 — Inter-DMA gap pattern changed: SUPPORTED.** SWDGE inter-issue gap compressed 48.7 → 26.7 ns (engine is back-pressured), HWDGE inter-issue gap stretched 8.1 → 51.3 ns (engine has 6x less work to do). Consistent with the engine-volume shift.

- **H4 — TE issue cadence regressed: SUPPORTED but small.** TE matmul start-ts gaps sum 4924.8 → 5225.0 us (+300 us); matmuls with positive wait 21772 → 22469 (+697). TE stalls more, but the *cause* is upstream DMA, not the TE engine itself.

- **H5 — Compiler scheduler made global decisions worse: NOT SUPPORTED.** Non-MoE instruction time by engine: Sync +99 us, GpSimd +4 us, others essentially unchanged. The MoE region accounts for 620 us of the 617 us total regression. This is a local effect, not a global scheduler artifact.

**Root cause (the actual mechanism, evidence-backed):** Canonical kept BOTH dynamic DMA queues pumping in parallel for **810 us of the MoE window (18.4 %)**. Exp1 reduced parallel-DMA time to **298 us (5.9 %)** because there's no longer enough HWDGE work (only 1/8 of weight tiles) to overlap the bulk SWDGE traffic. Total DMA bytes and the union-wall of DMA work are unchanged (2443 us vs 2454 us inside MoE), but the work is now serialized onto one engine. The lost parallelism (~512 us) closely matches the MoE wall regression (+620 us); the rest is downstream TE stalls (+245 us LDWEIGHTS waits, +25 us TE merged-active).

**This is H2 + H3 combined**, manifesting as a DMA pipelining inefficiency — not a single-engine bottleneck. H1 is misleading because GpSimd has plenty of headroom; H4 is a downstream symptom.

## Recommended next experiment

The standalone bench measured the **gate_up sub-region in isolation**, where the predicted Sync savings dominated because the DMA queues were not already running in parallel with another workload. The e2e bench reveals the canonical schedule was using BOTH DMA queues simultaneously, and the experiment broke that parallelism.

Recommended path forward, in order:

1. **Do not retry "HTile k onward on SWDGE" with smaller k.** Pushing the split to k=4 or k=2 trades the same parallelism in the opposite direction; the 810-us co-running window relies on roughly even HW/SW volumes, not on 1/8-vs-7/8 or 1/2-vs-1/2 statically.

2. **Try volume-balanced staggering across HTiles, not a binary split.** Alternate HWDGE / SWDGE per HTile (HW for HTiles {0,2,4,6}, SW for {1,3,5,7}) so each engine carries 4 HTiles. This preserves the parallel-DMA pattern that canonical exploits, while still amortizing Sync descriptor cost across two engines. Predict: HWDGE wall ~= SWDGE wall ~= ~1300-1500 us, "both busy" should recover to ~700-900 us, MoE wall back near 4400 us.

3. **Re-measure with the same recompile path.** Canonical was profiled from a separate compile; exp1 from this run's compile. Re-run canonical with the env exactly as now (compile then `--skip-compile` capture) before declaring any deltas < ~50 us meaningful. The 617-us regression is well above noise but it's worth bracketing.

4. **If (2) also regresses**, the problem is engine pairing semantics on this MLP shape: SWDGE descriptor latency is higher than HWDGE for the gate_up tile pattern, and the LDWEIGHTS chain in gate_up depends on tile-2 onward arriving without slack. In that case the fix is upstream: pre-fetch the first 2 HTiles before the first matmul (deepen the prologue), not change which engine carries them.

Files modified to capture this analysis: none — exp1 NEFF was already on disk and the canonical profile was pre-ingested.
