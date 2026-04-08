# Attention TKG Kernel Optimization Log

Kernel: `qwen3_attn_tkg_fused_oproj` — fused QKV + RMSNorm + RoPE + flash decode + O-proj  
Hardware: trn2, NEURON_LOGICAL_NC_CONFIG=2 (2 physical NeuronCores → 56MB SBUF)  
Shapes: B=1, H=2048, d=128, Hq_out=1024 (8 heads), S_prior=640 (5 tiles)

---

## Baseline: v12e

| Metric | Value |
|--------|-------|
| device_time_us | 65.33 |
| tensor_engine_pct | ~40% |
| dma_active_pct | ~50% |
| vector_engine_pct | 35.6% (NTFF) |
| scalar_engine_pct | 23.5% (NTFF) |
| spill_bytes | 0 |
| hbm_read_KiB | 9541.0 |
| hbm_write_KiB | 4.5 |
| mfu_estimated | 0.00% |

**Bottleneck**: Mixed DMA + compute. The kernel loads ~9.3MB of weights. At 3TB/s peak that's ~3.1μs theoretical minimum, but we're at 65μs — ~21× gap. The kernel is latency-bound rather than bandwidth-bound, indicating high DMA dispatch overhead and suboptimal DMA-compute overlap.

All engines are running simultaneously (DMA 50%, TensorE 40%, VectorE 35%, ScalarE 23%) but none dominate — classic pipelined but stall-ridden kernel where each stage must wait for the previous.

---

## Round 1

### Plans

| Plan | Title | Hypothesis |
|------|-------|------------|
| A | Hoist all Wo DMAs to kernel prologue | Earlier issue of 8×512KB Wo loads overlaps with more compute; reduces DMA active time |
| B | hw_dge on all dma_copy calls | Hardware DGE descriptor generation overlaps with ScalarE; reduces effective DMA overhead |
| C | Pre-broadcast mask tiles during hoisting | Moves 5×(2 nc_transpose + 2 tensor_copy) out of Pass 1 hot loop into pre-hoisting phase |

### Results

| Plan | device_time_us | tensor_eng% | dma% | spill | vs baseline | Notes |
|------|---------------|-------------|------|-------|-------------|-------|
| A (hoist Wo DMAs) | 64.01 | 44.1% | 52.8% | 0 | −2% (noise) | Wo was already well-overlapped |
| **B (hw_dge)** | **61.43** | — | — | 0 | **−6%** | Best; dge_mode enum is `hwdge` (no underscore) |
| C (pre-broadcast mask) | 64.15 | 43.4% | 53.5% | 0 | −1.8% | Marginal but real |
| **F (B+C combined)** | **59.47** | 46.3% | 53.9% | 0 | **−9%** | Best composite → new baseline |

### Findings

Plan B (hw_dge) is the dominant win. Plan C adds a small additional improvement. Combined they give −9% (65.33 → 59.47 μs). Hoisting Wo earlier (Plan A) did nothing because the compiler was already overlapping it effectively.

---

## Round 2 (baseline: v13_BC = 59.47 μs)

### Plans

| Plan | Title | Result |
|------|-------|--------|
| D | LNC=2 Wo sharding (each core loads 2MB instead of 4MB) | +14.6% REGRESSION |
| E | Mask fusion (fold constants, combine clamp+scale) | +16.9% REGRESSION |
| G | dma_transpose for K-cache (replace dma_copy+nc_transpose) | +13.5% REGRESSION |

### Findings

All Round 2 attempts regressed. **Critical pattern discovered**: this kernel is DMA-bound (DMA at ~54% of total time, TensorE at ~46%). The regression mechanism is consistent: removing TensorE instructions shifts those operations to the DMA engine (e.g., dma_transpose), or disrupts the compiler's instruction scheduling that overlaps DMA with compute. The kernel is highly sensitive to the exact instruction sequence.

**Key rule**: In this kernel, DMA is the bottleneck, TensorE has slack. Do NOT move work from TensorE to DMA. Do NOT remove TensorE instructions hoping to "free" TensorE — TensorE is already underutilized and serves as latency-filling compute to hide DMA latency.

The correct optimization direction is either:
1. Reduce DMA overhead/descriptor cost (hw_dge did this — already done)
2. Reduce HBM bandwidth (fewer reads — hard without algorithm change)
3. Improve DMA pipelining (overlap more DMA with compute)

---

---

## Round 3 (baseline: v13_BC = 59.47 μs)

| Plan | device_time_us | vs v13_BC | Notes |
|------|---------------|-----------|-------|
| G (dma_transpose K-cache) | 67.69 | +13.5% | Moving nc_transpose work to DMA engine adds DMA pressure |
| H (merge K/V hoisting loops) | 61.43 | +3.3% | Separate affine_range loops already pipelined by compiler |
| I (stride-0 DMA broadcast cos/sin/qnw) | 76.72 | +29% | Stride-0 reads HBM 8× per element — massive HBM amplification |

### Findings
All Round 3 plans regressed. The consistent pattern: any change that adds DMA work (even small) worsens performance. The compiler's instruction scheduling in v13_BC is near-optimal for the DMA pipeline.

---

## Optimizations That Worked (cumulative)

| Optimization | File | Improvement | Mechanism |
|---|---|---|---|
| Hardware DGE (`dge_mode=hwdge`) | v13_planB | −6% | DMA descriptor generation overlaps with ScalarE compute |
| Pre-broadcast mask during hoisting | v13_planC | −1.8% | Moves 5×4 ops out of Pass 1 hot loop |
| **Combined B+C** | **v13_BC** | **−9%** | **Best result: 65.33 → 59.47 μs** |

---

## Optimizations That Did Not Work

| Plan | Regression | Root cause |
|---|---|---|
| Hoist Wo DMAs to prologue (Plan A) | Neutral | Wo was already overlapping effectively |
| LNC=2 Wo output sharding (Plan D) | +15% | LNC coordination overhead > 2MB DMA savings |
| Mask computation scalar fusion (Plan E) | +17% | Disrupted compiler's DMA/compute overlap scheduling |
| dma_transpose for K-cache (Plan G) | +14% | Moves TensorE work to DMA engine — worsens DMA bottleneck |
| Merge K/V cache hoisting into one loop (Plan H) | +3% | Compiler already pipelining separate loops optimally |
| Stride-0 DMA broadcast for cos/sin/qnw (Plan I) | +29% | Stride-0 re-reads HBM 8× per element (8× bandwidth amplification) |
| Online softmax (single-pass) | N/A (rejected) | User-rejected: tried before, doesn't work |

---

## Architecture Notes

- `nl.affine_range` loops are unrolled by the compiler — tensors inside get unique names.
- `nc_transpose` is used heavily for broadcast: `[P,1].ap([[1,P],[0,GQA]])` → `nc_transpose` → `[GQA,P]` → `tensor_copy` → `nc_transpose` → `[P,GQA]`. Two nc_transpose + 2 tensor_copy per broadcast.
- `rms_ones [PMAX, PMAX]` is used for column-sum reductions via nc_matmul (computes per-partition sum broadcast to all output partitions).
- Wo DMA uses `ap()` pattern that is logically contiguous (128 consecutive rows × 2048 cols per head) despite the ap() syntax.
- `NEURON_LOGICAL_NC_CONFIG=2` gives 56MB SBUF — all 9.3MB of weights + K/V cache fit comfortably.
