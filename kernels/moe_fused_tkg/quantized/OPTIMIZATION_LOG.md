# FP8 MoE Fused TKG Kernel — Optimization Log

Kernel: `qwen3_moe_fused_tkg` — fused RMSNorm + Router + TopK(8) + Expert MLP for Qwen3-30B-A3B (TP=4, LNC=2, T=1 TKG).
Hardware: trn2 (NeuronCore-v3). BF16 baseline target: ~100 μs.

---

## Starting Point: v5 Profiling

```
total_time     = 131.60 μs
scalar_engine  = 83.7%  (110.19 μs)  ← dominant bottleneck
vector_engine  = 30.1%  (39.56 μs)
tensor_engine  = 30.9%  (40.61 μs)
dma_active     = 47.7%  (62.80 μs)
hbm_read       = 14136 KiB
spill_bytes    = 0
```

**Root cause**: v5 dequantizes FP8 weights to BF16 before every matmul. This produces 24 large `nisa.activation(op=nl.copy)` calls (3 per expert × 8 experts) that run on ScalarE and dominate execution. Key call sizes: gate_up `[128P, 16×384F]` = 6144 elements/partition per expert.

---

## Round 1

### Plan A — FP8 Stationary Matmul (`float8_e4m3`) ✅ WINNER

**Idea**: Eliminate fp8→bf16 dequant by using FP8 weight tensors directly as the `stationary` argument in `nisa.nc_matmul`. Apply weight dequant scale post-matmul via existing `nisa.activation(data=psum, scale=weight_scale)` drain — unchanged.

**Key finding**: `nl.float8_e4m3fn` as stationary fails with a BIR verification error. `nl.float8_e4m3` (without `fn`) maps to `float8e4` which the BIR verifier accepts. The compiler's error message for the `fn` variant listed `float8e4` as allowed — this was the hint.

**Changes (v6a)**:
- Removed `gate_up_buf{0-3}`, `down_full0_buf{0-3}`, `down_full1_buf{0-3}` (bf16 dequant buffers)
- Removed all 24 large `nisa.activation(fp8→bf16)` calls
- Changed `gate_t1_128` / `up_t1_128` dtype to `float8_e4m3`
- All matmul stationary args now reference `*_fp8_bufs[k]` directly
- Down tile padding zero-fill preserved on fp8 buffers

**Result**: 131.60 → 125.62 μs (−4.5%). ScalarE drops from 110 μs of dequant work.

**Correctness note**: The original test used `rng.integers(-127, 127)` cast to `int8`, which reinterprets some bytes as the fp8 NaN bit pattern (`0xFF`), causing all-NaN outputs in both ref and candidate — making the test vacuous. Fix: use proper FP8 quantization (`w.clamp(-240,240).to(float8_e4m3fn).view(int8)`) with derived per-neuron scales. After fix, v6a matches v5 with `max_diff=0.0`.

---

### Plan B — Fuse SiLU + Affinity into Post-Matmul Activations ✅ IMPROVEMENT

**Idea**: Per expert per i_tile, two separate ScalarE calls (`copy+scale` then `silu`) can be fused into one. Per expert per h1_out, two calls (`activation(scale)` + `tensor_scalar(affinity)`) can be replaced with one `activation(combined_scale)`.

**Changes (v6b, applied to base v5)**:
- Fused `activation(op=copy, scale=gate_scale)` + `activation(op=silu)` → `activation(op=silu, scale=gate_scale)`
- Precomputed `combined_down_scale[128P, 2F] = down_scale × affinity_weight`, then drained PSUM with combined scale in one call
- Eliminates 2 ScalarE calls per expert per i_tile (gate chain) and 1 call per expert per h1_out (down chain)

**Result**: 136.94 → 127.36 μs (−7.0% vs same-run v5 baseline). ScalarE 80.2% → 79.9%.

---

### Plan C — Merge Load+Dequant Loops per Expert ❌ REGRESSION

**Idea**: Two separate `for k in static_range(4)` loops (all loads, then all dequants) prevent DMA/ScalarE overlap. Merging them into one loop (load k, dequant k, repeat) should let hardware pipeline DMA[k+1] with ScalarE[k].

**Result**: 136.94 → 142.69 μs (+4.2% regression). The loop merge disrupted the compiler's scheduling; the combined loop body created tighter data dependencies and reduced the compiler's freedom to pipeline. Loop boundaries in NKI `static_range` don't guarantee better overlap when the loop body itself becomes a longer dependency chain.

**Lesson**: Merging loops in NKI does not reliably improve DMA/ScalarE overlap. The compiler handles inter-loop scheduling already, and loop merging can hurt by creating longer serialized instruction sequences.

---

## Round 2

### Plan D — Combine A + B (v7) ✅ BEST RESULT

**Idea**: Apply Plan B's fusions on top of v6a (which already has FP8 stationary). The improvements are orthogonal — A eliminates large dequant work, B reduces the small post-matmul call count.

**Changes (v7, based on v6a)**:
- SiLU fusion (Change 1): `activation(op=silu, scale=gate_scale)` replaces 2-call chain. `gate_f32_scaled` buffer eliminated.
- Affinity fusion (Change 2): `combined_down_scale` precomputed via `tensor_tensor`, PSUM drained with combined scale in one call. `down_h_sbuf` + `tensor_scalar` eliminated.
- Both Wave 0 and Wave 1 compute loops updated (Wave 1 uses `kk=k+4` for affinity index).

**Result**: 125.62 → 120.44 μs (−4.1% vs v6a, −8.5% vs v5 original). ScalarE: 83.7% → 40%. DMA is now the leading engine at 47.5%.

```
v7 profile:
  device_time_us    = 120.44
  tensor_engine     = 36.5%
  scalar_engine     = 40.0%
  vector_engine     = 40.0%
  dma_active        = 47.5%
  hbm_read          = 14136 KiB
  spill_bytes       = 0
```

---

## Summary Table

| Version | device_time_us | vs v5 orig | Bottleneck | Key change |
|---------|---------------|-----------|------------|------------|
| v5 | 131.60 | — | ScalarE 83.7% | Baseline FP8 w/ bf16 dequant |
| v6a | 125.62 | −4.5% | ScalarE↓, DMA | FP8 stationary (`float8_e4m3`) |
| v6b | 127.36 | −3.2% | ScalarE 79.9% | SiLU+affinity fusion (on v5) |
| v6c | 142.69 | **+8.4%** | ScalarE 76%, DMA | Loop merge — REGRESSION |
| **v7** | **120.44** | **−8.5%** | DMA 47.5% | A+B combined — best |

BF16 baseline target: ~100 μs. v7 is at 120 μs, ~20% above target.

---

## What Worked

1. **`float8_e4m3` (without `fn`) as matmul stationary** — the single biggest unlock. `float8_e4m3fn` fails BIR verification; `float8_e4m3` passes. Eliminates the dominant ScalarE bottleneck entirely (6144 elements/partition per expert → 0 dequant cost). Always try both FP8 variants before assuming FP8 stationary is unsupported.

2. **Fusing `activation(copy+scale)` + `activation(silu)` → `activation(silu, scale=...)`** — `nisa.activation` accepts both `op` and `scale` simultaneously. One ScalarE call instead of two, plus one fewer intermediate SBUF buffer allocation per expert per i_tile.

3. **Precomputing combined scale (down_scale × affinity) before the drain loop** — eliminates a `tensor_scalar` VectorE call per h1_out per expert and reduces the critical path length through the post-matmul chain.

## What Didn't Work

1. **`float8_e4m3fn` as stationary** — BIR verifier rejects it despite listing `float8e4` as valid. The `fn` (finite) variant is a different internal type.

2. **Merging load+dequant loops** — theoretical DMA/ScalarE overlap benefit did not materialize. The NKI compiler already overlaps engines across separate loops; merging disrupts its scheduling freedom and creates longer in-loop dependency chains. Do not merge loops expecting overlap improvements.

3. **FP8 stationary with `perf_mode=double_row`** — not tested (requires moving tensor also in FP8). With BF16 moving (activation), double_row is unavailable regardless of stationary dtype.

---

## Round 4 (quantized/ kernels)

### Plan D — FP8 Activation Quantization + double_row attempt (v12d) ❌ REGRESSION

**Idea**: Quantize RMSNorm output to FP8 per-token using global absmax scale (computed via tensor_reduce over free dim + tensor_partition_reduce with max op across 128 partitions). Use FP8 moving in gate_up nc_matmul to enable `perf_mode=double_row` for 2× TensorE throughput.

**Key findings**:

1. **`double_row` is blocked at T=1**: BIR verifier requires "second dim of input AP must have Num=2, Size%16==0". With T=1, the moving slice `rmsnorm_normed_fp8[128P, 1]` has Access Pattern `[[16,128],[1,1]]` with Num=1. `double_row` is a TKG-phase (batch matmul) feature only — it requires the moving tensor to have ≥2 tokens simultaneously. The hardware contraction doubles K from 128→256, requiring two adjacent rows.

2. **`tensor_partition_reduce` with `nl.maximum`**: Works when called as `nisa.tensor_partition_reduce(dst, nl.maximum, data)` (positional, not keyword-style `op=nl.maximum`). The `_percent` fields in NTFF JSON are fractions (0.0-1.0), not percentages — multiply by 100 for display.

3. **Per-partition scale is mathematically wrong**: Computing per-partition absmax and using different scales across 128 partitions is incorrect for nc_matmul. The moving tensor's K contraction sums `sum_k(stationary[p_out, k] * moving[k, t])` where each K value comes from a different partition. For correct dequantization, all K values must share the same scale. Must use global (cross-partition) absmax.

4. **FP8 activation quantization adds ~3-5% absolute error**: The global absmax/240 scale means many activation values use only a fraction of FP8 range, losing precision. Max absolute error ~0.9 on values up to ~27 (relative ~3.4%). Exceeds the planned 2% tolerance.

5. **v12d is slower than v11c**: 112.29 μs vs 104.40 μs (+7.9 μs). The FP8 activation quantization adds 7 extra SBUF operations (abs, tensor_reduce, partition_reduce, nc_stream_shuffle×4, scale, reciprocal, tensor_tensor×2, activation cast) to the critical path. Without double_row benefit, the overhead is net negative.

**Result**: 104.40 → 112.29 μs (+7.6% regression). Plan D fails for TKG (T=1) workloads.

```
v12d profile:
  device_time_us    = 112.29
  tensor_engine     = 43.2%
  scalar_engine     = 42.3%
  vector_engine     = 50.7%
  dma_active        = 62.0%
  hbm_read          = 16488 KiB   (vs 14136 KiB for v11c — extra activation data)
  spill_bytes       = 0
```

**Root cause of Plan D failure**: The TKG setting (T=1) is fundamentally incompatible with `perf_mode=double_row`. The hardware requires K to be doubled from 128→256, which needs 2 tokens in the moving tensor. Plan D only works for batch inference (T≥2) or prefill (longer sequences).

---

## Round 5 (scheduling optimizations)

### Plan G — Precomputed combined_down_scales (v12g) ❌ REGRESSION

**Idea**: Move per-expert `combined_down_scale` precomputation from inside Phase 2a loop body to before Phase 2a, where it overlaps with in-flight Wave 1 DMA.

**Result**: 101.00 → 106.98 μs (+6.0% regression). Compiler was already scheduling these ops inside DMA windows; pulling them earlier breaks that overlap.

---

### Plan H — Single PSUM for All 8 Experts (v12h) ❌ REGRESSION

**Idea**: Double gate_up_psum [16F→32F] and down_psum [32F→64F] to hold all 8 experts, eliminating the two nisa.memset calls between Wave 0 and Wave 1.

**Result**: 101.00 → 109.29 μs (+8.2% regression). Larger PSUM adds overhead; the memsets were not a meaningful bottleneck.

---

### Plan I — affine_range Tile-1 Prefetch (v12i) ✅ WINNER

**Idea**: Extract `gate_t1_128` / `up_t1_128` tensor_copy setup into a dedicated `affine_range(_K_WAVE)` prefetch loop before the static_range compute loop. Gives compiler freedom to pipeline tile-1 copies for expert k+1 while expert k's h1 matmul runs.

**Key changes**:
- Per-expert tile-1 scratch buffers: `gate_t1_128_bufs[0..3]` and `up_t1_128_bufs[0..3]`, each `[128P, H_free=16, I0=128]` fp8 (~1 MB per wave)
- Prefetch loop: `for k in nl.affine_range(_K_WAVE)` → tensor_copy gate_t1 + up_t1 for all h1 into per-k buffers
- Compute loop (static_range): references pre-filled per-k buffers
- `if k == 0` output branch replaced by memset(output_temp, 0) + always-add

**Result**: 104.40 → 90.54 μs (−13.3%). All engines run simultaneously at 44–48%.

```
v12i profile:
  device_time_us    = 90.54
  tensor_engine     = 44.1%
  vector_engine     = 48.4%
  scalar_engine     = 44.6%
  dma_active        = 62.3%
  hbm_read          = 16488 KiB
  spill_bytes       = 0
```

**Why it works**: `affine_range` lets the compiler see that tile-1 copies for expert k+1 are independent of expert k's matmul, interleaving them with the h1 loop's matmul calls. Separate per-expert buffers (no aliasing) make this pipelining legal.

---

## Round 6 (Plan L — MXFP4 / Alternative)

### Phase 1: nc_matmul_mx / MXFP4 Probe ❌ BLOCKED ON TRN2

**Finding**: `nisa.nc_matmul_mx` is available in the NKI API (trn2 SDK) and `nl.float4_e2m1fn_x4` dtype exists, but compilation fails with:

```
[INTERNAL_ERROR] [NCC_IBIR530] MatmultMx is not supported on arch Trn2,
must be Trn3 or greater
```

MXFP4 matrix multiplication is a Trn3+ feature only. This blocks the entire Plan L MXFP4 path.

Also confirmed: `perf_mode=double_row` (tried in v12d) requires T≥2 moving tensor — incompatible with TKG T=1.

**API signature found**: `nisa.nc_matmul_mx(dst, stationary, moving, stationary_scale, moving_scale, tile_position=None, tile_size=None)` — stationary/moving must be `float4_e2m1fn_x4` (4-packed), scale tiles must be `uint8`. Available dtype: `nl.float4_e2m1fn_x4`.

Note: `nisa.quantize_mx` only outputs `float8_e5m2_x4` or `float8_e4m3fn_x4` (FP8), not FP4. No on-device FP4 quantization is possible on trn2.

---

### Plan L Alternative 2: Split gate_up loading (v12k) ❌ REGRESSION

**Idea**: Instead of one large gate_up DMA [128P, H_free=16, GU=384] per expert + SBUF tensor_copy for tile-1 (the v12i approach), split into 4 targeted DMA sub-loads per expert:
- gate tile-0: cols [0:I0=128]  → gate0_fp8_bufs [128P, H_free, I0]
- gate tile-1: cols [I0:I0+I1]  → gate_t1_fp8_bufs [128P, H_free, I0] (pre-zeroed, I1 rows filled)
- up tile-0:   cols [I:I+I0]    → up0_fp8_bufs [128P, H_free, I0]
- up tile-1:   cols [I+I0:I+I0+I1] → up_t1_fp8_bufs [128P, H_free, I0] (pre-zeroed, I1 rows filled)

This eliminates 32 tensor_copy SBUF operations (the affine_range prefetch loops from v12i).

**Result**: 92.24 → 126.57 μs (+37.2% regression). DMA time increased from 57.65 μs to 88.31 μs.

```
v12k profile:
  device_time_us    = 126.57
  tensor_engine     = 42.0%  (53.14 μs)
  dma_active        = 69.8%  (88.31 μs)  ← +53% more DMA time
  scalar_engine     = 40.5%
  hbm_read          = 16488 KiB  (unchanged)
  spill_bytes       = 0
```

**Root cause**: Splitting 1 DMA call into 4 per expert quadruples GpSimdE descriptor generation overhead. Each indirect DMA call requires GpSimdE to compute descriptor from scalar_offset. v12k: 8 DMA calls × 8 experts × 2 waves = 128 GpSimdE descriptor ops vs v12i's 5 × 8 × 2 = 80. The extra 48 descriptor generations add ~30 μs. The v12i SBUF tensor_copies were free (overlapping with DMA); eliminating them while adding DMA call overhead is net negative.

**Lesson**: The v12i approach (1 combined gate_up DMA + SBUF copies for tile-1) minimizes GpSimdE descriptor overhead. Do not split into sub-loads to save SBUF copies when DMA is already the bottleneck.

---

## Round 7 (Plan M — Preloaded Scales)

### Plan M — Preload All Expert Scales to Eliminate Indirect Scale DMAs ❌ BLOCKED BY COMPILER

**Idea**: Of the 40 indirect DMA calls (dge_mode=0 / GpSimdE) per kernel launch, 16 are for tiny scale loads (8 gate_up_scales × 8 experts + 8 down_scales × 8 experts). Each scale DMA only moves 1.5KB–4KB but still requires a GpSimdE descriptor op to compute `expert_id × stride`. Plan: preload ALL experts' scales (128 experts total) via 2 direct DMA calls (dge_mode=3, no indirection), then extract per-expert slices from SBUF.

**Implementation attempted**:
1. `all_gu_scales_sb = nl.ndarray((_PMAX, E*3=384), float32, buffer=nl.sbuf)` — 192KB SBUF
2. `all_down_scales_sb = nl.ndarray((_PMAX, E*8=1024), float32, buffer=nl.sbuf)` — 512KB SBUF
3. Load both via `nisa.dma_copy(..., dge_mode=3)` — direct DMA, no GpSimdE
4. Per-expert extraction: `nisa.tensor_copy(dst=gate_up_scale_bufs[k], src=all_gu_scales_sb.ap(..., scalar_offset=expert_id, indirect_dim=0))`

**Compiler error (step 4 fails)**:
```
[INTERNAL_ERROR] [NCC_INLA001] Dynamic Access is not allowed in the instruction.
```

SBUF-to-SBUF `tensor_copy` with runtime `scalar_offset` (runtime expert_id) is rejected by neuronx-cc. The NKI compiler does not support dynamic SBUF address calculation.

**Root cause**: There is no way to extract per-expert data from an SBUF preload buffer using a runtime index. All SBUF access patterns must be statically-known at compile time.

**Fallback (v12m)**: v12m documents this finding and falls back to v12i behavior (same dge_mode=0 indirect DMAs for scale loads). Functionally identical to v12i.

**Benchmark**:
```
v12m (= v12i) profile:
  device_time_us    = 90.94
  tensor_engine     = 45.6%  (41.43 μs)
  dma_active        = 62.4%  (56.74 μs)
  scalar_engine     = 47.0%  (42.71 μs)
  hbm_read          = 16488.0 KiB
  spill_bytes       = 0
```

(v12i fresh run: 89.52 μs. Difference is within measurement noise.)

**Conclusion**: The 16 indirect scale DMAs cannot be eliminated on current neuronx-cc toolchain because:
1. SBUF dynamic indexing with runtime expert_id is not allowed by the compiler
2. HBM direct DMA (dge_mode=3) requires statically-known offsets — incompatible with runtime expert_id
3. There is no intermediate hardware mechanism that avoids GpSimdE for runtime-indexed data

**v12i remains the practical ceiling** for this workload on Trn2.

---

## Summary Table (all versions)

| Version | device_time_us | vs v5 orig | Bottleneck | Key change |
|---------|---------------|-----------|------------|------------|
| v5 | 131.60 | — | ScalarE 83.7% | BF16 dequant baseline |
| v7 | 120.44 | −8.5% | DMA 47.5% | FP8 stationary + fused scales |
| v11c | 104.40 | −20.7% | DMA 62.3% | affine_range DMA + batched output |
| **v12i** | **92.24** | **−29.9%** | DMA 62.5% | affine_range tile-1 prefetch |
| v12k | 126.57 | +3.8% vs v5 | DMA 69.8% | Split gate_up load — REGRESSION |
| v12m | ~90–91 | same as v12i | DMA 62.4% | Plan M probe — SBUF dynamic indexing blocked |

---

## Remaining Bottleneck and Future Directions

After v12i (92.24 μs), DMA is the critical path at 62.5% (~57.65 μs). HBM reads are fixed at ~16.5 MiB (8 expert weight matrices). All engines run at 42–48%, indicating a well-saturated schedule.

- **MXFP4 weights (Trn3+)** — nc_matmul_mx is unavailable on Trn2. Halving HBM traffic to ~8 MiB requires Trn3 hardware. This is the single largest remaining lever but hardware-blocked.
- **Hardware DGE for expert loads** — current loads use `dge_mode=0` (software/GpSimdE) because of `scalar_offset`. Increasing per-expert DMA call count is counter-productive (v12k: +37% regression). Decreasing calls below 5 per expert would require packing data formats.
- **DMA call minimization** — 5 DMA calls per expert is near-minimal. Merging gate_up+scales into 1 tensor could reduce to 4 but the savings are tiny (scales = 1.5KB vs 768KB weights).
- **Scale preloading (Plan M)** — BLOCKED. SBUF dynamic indexing not supported by neuronx-cc. 16 indirect scale DMAs cannot be eliminated without compiler support for runtime SBUF addressing.
- **FP8 activation + double_row** — double_row requires T≥2. Not applicable to TKG (T=1).
- **v12i is the hardware ceiling** — At 92.24 μs with 16488 KiB HBM reads, v12i is effectively at the DMA bandwidth limit for 8-expert FP8 MoE on Trn2 at T=1.
