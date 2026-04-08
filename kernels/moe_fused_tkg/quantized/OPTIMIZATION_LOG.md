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

## Remaining Bottleneck and Future Directions

After v7, DMA is the critical engine at 47.5% (62+ μs). HBM reads are fixed at 14136 KiB (8 expert weight matrices per token — unavoidable for correctness). Remaining opportunities:

- **Quantize activations to FP8** — enables `perf_mode=double_row` for 2× TensorE throughput. Requires adding per-token absmax quantization (tiny ScalarE cost for T=1) and adjusting post-matmul scale. Would also allow trying `float8_e4m3` as moving.
- **Hardware DGE for expert loads** — current loads use `dge_mode=0` (software/GpSimdE) because of `scalar_offset`. The indirect DMA generates descriptors on GpSimdE, adding latency. No known workaround without changing the expert-ID indirection pattern.
- **Reduce per-expert DMA call count** — currently 5 DMA calls per expert (gate_up_fp8, gate_up_scales, down0_fp8, down1_fp8, down_scales). Packing scales alongside weights could reduce this to 3 calls at the cost of reformatting the weight layout offline.
