# MoE Fused TKG Kernel — Optimization Log

**Kernel**: `qwen3_moe_fused_tkg` — RMSNorm + Router + TopK(8) + Expert MLPs, fused end-to-end
**Target**: ≤100 μs on trn2 (NeuronCore-v3, LNC=2)
**Reference**: `kernel_v19b.py` (correctness baseline)
**Current best**: `kernel_v28f.py` — **98.15 μs**

---

## Fixed Constants (Do Not Change)

```
B=1, H=2048, E=128, K=8, I=192, I0=128, I1=64
_PMAX=128, H_shard=1024, H_free=16, H_free_shard=8
gate_up_w: [E=128, H=2048, 2*I=384] bf16
down_w:    [E=128, I=192,  H=2048]  bf16
Invoked as: qwen3_moe_fused_tkg[2](...)  — LNC=2
```

**Hard constraints**:
- Cannot use `core_barrier` or `sendrecv` — compiler bug
- Cannot change input/output interface
- sw_dma_packet_count = **8320** (fixed — 24 indirect DMAs × ~347 packets each)
- hbm_read = **31760 KiB** (fixed — 8 experts × 3 weight tensors each)

---

## Current Best: kernel_v27d Structure

```
[RMSNorm + Router + Softmax + TopK]   ← prologue, ~15-20 μs
Phase 1a: 12 indirect DMAs (experts 0-3)
  norm_weights + aff_bcast (9 instructions, overlaps DMA)
  PSUM memset ×2 (wave 0)
Phase 2a: compute experts 0-3 (loop k=0..3)
  tile_copy → gate/up matmul → silu/up/multiply/cast → down matmul → flush+scale+accumulate
  [NO re-memset between waves for down_full1 — KEY OPTIMIZATION]
  PSUM memset ×2 (wave 1)
Phase 1b: 12 indirect DMAs (experts 4-7)
Phase 2b: compute experts 4-7 (loop k=0..3)
[Transpose + store output]
```

**SBUF budget** (~74 KiB of 224 KiB):
- 4× gate_up_buf [128, 16, 384] = 48 KiB
- 4× down_full0_buf [128, 1024] = 8 KiB
- 4× down_full1_buf [128, 1024] = 8 KiB
- gate_t1_128 + up_t1_128 = 8 KiB
- misc ≈ 2 KiB

**PSUM budget**: gate_up_psum [128, 16] + down_psum [128, 32] = 48 cols

---

## What Worked

| Version | Change | Result | Savings |
|---------|--------|--------|---------|
| v20b | Baseline | 117.46 μs | — |
| v26g | 2-wave structure, 4 buffer sets (vs 8) | 104.89 μs | −12.57 μs |
| v27d | Remove 4 redundant inter-wave down_full1 pad re-memsets | 103.48 μs | −1.41 μs |
| v28e | Router DMA: `dge_mode=0` → `dge_mode=3`, `_ROUTER_BATCH` 4→8 | 98.74 μs | −4.74 μs |
| **v28f** | **`_ROUTER_BATCH` 8→16 (1 router DMA call instead of 2)** | **98.15 μs** | **−0.59 μs** |

**Why v27d works**: The DMA for each expert writes only to `down_full1_bufs[k][0:I1, :]` (rows 0:64). Rows 64:128 are zeroed once before wave 0 and never touched by any DMA. So the 4 re-memsets between waves were pure overhead with no effect on correctness.

**Why v26g works**: Reducing from 8 buffer sets to 4 freed ~96 KiB of SBUF, dramatically improving the compiler's instruction scheduling quality. SBUF pressure is the primary lever.

---

## What Failed (Do Not Retry)

### Structural / DMA changes

| Version | Idea | Result | Why it failed |
|---------|------|--------|---------------|
| v27e | 4 waves of 2 experts, 4 buffers per wave | 106.81 μs | More PSUM memsets, shorter DMA batches per phase |
| v27c | 4 waves of 2 experts, 2 buffer sets | 113.00 μs | 6 DMAs/wave instead of 12 — DMA engine underutilized |
| v27b | 1 expert at a time, double-buffered | 115.88 μs | 3 DMAs/wave — severe DMA pipeline starvation |
| v27i | Prefetch wave-1 DMAs inside wave-0 compute loop | 105.54 μs | Disrupts compiler's instruction scheduling |
| v27g | Move PSUM memsets to after Phase 1b DMAs | 110.06 μs | Severely disrupts scheduler (any reordering hurts) |

**Root cause pattern**: The NKI compiler's instruction scheduler is extremely sensitive to code ordering. Any reordering — even if semantically equivalent — disrupts the schedule and regresses performance.

### Compute instruction changes

| Version | Idea | Result | Why it failed |
|---------|------|--------|---------------|
| v27a | Separate PSUM per wave (96 PSUM cols) | 107.15 μs | Extra PSUM size hurts scheduler |
| v27f | Per-expert PSUM (12 cols), per-expert memset | 106.64 μs | More memset instructions always worse |
| v27h | `nisa.scalar_tensor_tensor` fused scale+accumulate | 109.05 μs | Switches to scalar engine, disrupts pipeline |
| v28i | `dma_transpose` for inp/gamma loading (replace 6 instr with 2) | 101.99 μs | Moves TensorE/VectorE work onto DMA engine (already bottleneck), net loss |

### Round 5 — v28f baseline (98.15 μs)

| Version | Idea | Result | Why it failed |
|---------|------|--------|---------------|
| v29j | `nl.hbm` → `nl.shared_hbm` (required) + remove dead `aff_bcast` memset | 100.05 μs | +1.9 μs — shared_hbm slightly changes DMA schedule; removing the memset didn't help |
| v29k | Single wave of 8 experts (merge 2×4 waves into 1×8) | 98.99 μs | PSUM cols doubled (96 vs 48), consistent with v27a lesson: larger PSUM hurts |
| v29m | Merged down DMA: prepad down_w [E,192,H]→[E,256,H], single DMA per expert | 100.98 μs | 3D DMA pattern generates MORE sw_dma_packets (8192 vs 7936), not fewer — the NKI compiler split the 3D indirect DMA into multiple descriptor chains |

**Round 5 key lessons**:
- `nl.shared_hbm` output costs ~1-2 μs vs `nl.hbm` due to different DMA scheduling (but is required for correctness)
- Increasing PSUM cols from 48 to 96 (single-wave 8 experts) regresses — confirms v27a finding
- A 3D indirect DMA pattern `[[H, 128], [128*H, 2], [1, H_shard]]` does NOT reduce sw_dma_packets; the compiler appears to split it into ≥2 descriptor chains, producing MORE packets than two separate 2D DMAs
- The `aff_bcast` memset appears technically redundant (all 128 rows overwritten) but removing it hurts — likely a scheduling dependency the compiler relies on
- DMA batching depth (12/phase) is still the confirmed sweet spot; merging to 8/phase (Plan M removes 1 DMA type) or expanding to 24/phase (Plan K) both regress

**Key lessons**:
- Every added instruction hurts; every removed instruction (if truly redundant) helps
- Changing instruction TYPE (e.g., tensor_tensor → scalar_tensor_tensor) disrupts engine pipeline scheduling
- Fewer PSUM columns is not always better — 48 cols (v27d) beats 12 cols (v27f) and 96 cols (v27a)
- DMA batching depth matters: 12 DMAs/phase is the sweet spot; going lower (6, 3) or higher (24) causes regressions
- 3D indirect DMA patterns can generate MORE packets than equivalent 2D patterns — prefer simple 2D patterns for sw_dge DMAs

---

## Key Technical Facts

### DMA
- All 24 expert DMAs use `dge_mode=0` (software DGE, indirect addressing via `scalar_offset`)
- Hardware DGE (`dge_mode=3`) does NOT support `scalar_offset` — cannot be used for expert loads
- sw_dma_packet_count=8320 is fixed regardless of instruction ordering
- Minimum HBM transfer time ≈ 40 μs at 800 GB/s; DMA active time ≈ 65 μs (extra 25 μs = DGE overhead)

### PSUM
- `PSUM` buffers must be zeroed with `nisa.memset` before `nc_matmul` accumulates into them
- `PSUM` buffers must be copied to SBUF via `nisa.activation(op=nl.copy)` before use
- `nisa.activation(scale=tensor)` is INVALID — `scale` must be a compile-time scalar float

### nc_matmul / I-dim tiling
- Contraction dim K must be ≤128 and the stationary must be [128, K]
- I=192 requires 2 tiles: I0=128 (full) + I1=64 (zero-padded to 128 in gate_t1_128/up_t1_128)
- The zero-padding in gate_t1_128/up_t1_128 (columns 64:128) is set once and reused across both waves — do NOT re-memset
- The zero-padding in down_full1_bufs (rows 64:128) is set once before wave 0 and reused — do NOT re-memset between waves

### Profiler numbers (v27d approximate)
- tensor_engine_pct ≈ 41%, active ≈ 43 μs
- dma_active_pct ≈ 63%, active ≈ 65 μs
- spill_bytes = 0
- These overlap: kernel is dual-bottlenecked (DMA and compute run concurrently)

---


---

## Ideas Not Yet Explored

- **`activation_reduce` fusions** (see "Remaining Candidate Optimizations" below): replace `activation(square)` + `tensor_reduce(add)` with single `activation_reduce`; same for softmax exp+sum. Low risk, ~1-2 μs potential. Note: `activation_reduce(op=nl.square, reduce_op=nl.add)` was attempted in an earlier round and hit `[NCC_IXCG864] ISA check failed` — verify compiler support before retrying.
- **FP8 double-performance mode**: major correctness risk, needs quantization infrastructure
- **Different SBUF tiling for gate_up**: currently loads full [H_free, GU_FLAT] per expert; smaller tiles with higher reuse could reduce SBUF pressure but may hurt DMA efficiency
- **LNC shard along I dimension**: not viable — I/2=96 < K_min=128 for nc_matmul

## Remaining Candidate Optimizations (Untested, Low-Risk)

These are instruction REMOVALS in the prologue — the safe pattern.

1. **`nisa.activation_reduce` for RMSNorm square+sum**: Replace `activation(square)` + `tensor_reduce(add)` with a single `activation_reduce(square, add)`. Saves: 1 instruction, 1 full pass over [128, 16] data. **Caveat**: compiler rejected `activation_reduce(op=nl.square, reduce_op=nl.add)` in earlier testing with ISA error — verify.

2. **`nisa.activation_reduce` for softmax exp+sum**: Replace `activation(exp)` + `tensor_reduce(add)` with `activation_reduce(exp, add)`. Saves: 1 instruction, 1 full pass over [1, 128] data.

3. **Remove `sum_reduced_sb` intermediate**: Write `tensor_partition_reduce` result directly into `norm_sum_sb[0:1, :]` and eliminate the allocation + tensor_copy. Saves: 1 instruction.

Estimated combined savings: ~1–2 μs. Likely insufficient to close the remaining gap to ≤95 μs without the shared_hbm overhead factored in (which adds ~1.9 μs).
