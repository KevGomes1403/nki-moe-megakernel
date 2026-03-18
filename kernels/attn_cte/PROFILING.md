# v0

## Overall Stats

| Metric | Value |
| --- | --- |
| Model Flops | 0.000 FLOPS |
| Total Time | 36.46 ms |
| Mm Arithmetic Intensity | 109 |
| Peak Flops Bandwidth Ratio | 109.837 FLOPS |
| Event Count | 35,538 |
| Trace Count | 6,760,193 |
| Neuroncore Cycle Count | 43,749,528 |
| Transpose Flops | 13.086 GFLOPS |
| Hardware Flops | 20.384 GFLOPS |

---

## Tensor Engine Stats

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 920.19 µs |
| Mfu Estimated Percent | 0.13% |
| Tensor Engine Instruction Time | 1.99 ms |
| Hfu Estimated Percent | 0.36% |
| Tensor Engine Active Time Percent | 2.52% |
| Mfu Max Achievable Estimated Percent | 49.45% |
| Matmul Instruction Count | 4,860 |
| Tensor Engine Instruction Count | 10,188 |

---

## DMA / Memory Stats

| Metric | Value |
| --- | --- |
| Mbu Min Read Util Percent | 0.05% |
| Mbu Estimated Percent | 0.26% |
| DMA Active Time | 27.22 ms |
| DMA Transfer Time | 33.00 ms |
| DMA Packet Time | 51.74 ms |
| DMA Active Time Percent | 74.66% |
| Dynamic DMA Size Percent | 98.91% |
| Dynamic DMA Packet Percent | 100.00% |
| Static DMA Packet Count | 50 |
| DMA Queue Count | 51 |
| DMA Transfer Count | 1,946 |
| Psum Read Sbuf Write Count | 3,260 |

# Round 4

## Optimization Summary

Baseline: v3a — 4.91ms, 83.61% DMA active, 14.24% TensorE active, 938 DMA transfers, 51.31ms DMA Packet Time (latency-bound micro-transfers)

---

Plan A — v4a_qwen3_attn_cte_fused.py

Targets the two "Profiler:" idle-gap lines directly.

```python
┌──────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│              Change              │                                                             What's eliminated                                                             │
├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Pre-transpose K tiles in Phase 1 │ nc_transpose + tensor_copy chain before every MM1 — 200 ops removed from the innermost (kvh×gi×qsi×ki) loop, at cost of 20 ops in Phase 1 │
├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ nisa.affine_select causal mask   │ dma_copy(mk, attn_mask[...]) + tensor_tensor(score+mask) — 200 HBM DMA loads (12.8MB) eliminated, attn_mask removed from signature        │
└──────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

The affine_select predicate (qso + q_idx) ≥ (kso + k_idx) encodes as pattern=[[-1, PMAX]], offset=qso-kso, channel_multiplier=1. Correctness: PASS max_diff=2.44e-04

---

Plan B — v4b_qwen3_attn_cte_fused.py

v3a + v3b combined: eliminates all redundant hidden HBM loads.

Preloads all 80 hidden tiles (5 si × 16 ht × 32KB = 2.56MB) into SBUF before the main loops. Merges K and V projection ht loops so each hidden tile feeds both matmuls. Eliminates
~1,200 HBM loads from the hot path:

- Phase 1 K-proj: 320 loads → 0
- Phase 1 V-proj: 320 loads → 0 (was duplicate of K-proj)
- Phase 2 Q-proj: 640 loads → 0

Correctness: PASS max_diff=2.44e-04

---

Plan C — v4c_qwen3_attn_cte_fused.py

v4a + 512-wide K tiles + PSUM-accumulated MM2.

Batches 4 K tiles into K_wide[d=128, 512], does one MM1 producing scores[128, 512], runs flash attention rescaling (max/exp/sum) on the wider tile (4× fewer iterations), then
accumulates 4 V tiles into a single PSUM without intermediate PSUM→SBUF copies. For S=640: 1 batch of 4 + 1 tail tile = 2 flash-attn outer iterations instead of 5. The affine_select
offset for the 512-wide tile is qso - kso_batch where kso_batch = batch_start * PMAX.

Correctness: PASS max_diff=2.44e-04

## Data (v4a)

### Overall Stats

| Metric | Value |
| --- | --- |
| Model Flops | 0.000 FLOPS |
| Total Time | 4.74 ms |
| Peak Flops Bandwidth Ratio | 109.837 FLOPS |
| Mm Arithmetic Intensity | 389 |
| Event Count | 20,744 |
| Neuroncore Cycle Count | 5,682,719 |
| Trace Count | 6,648,681 |
| Transpose Flops | 8.036 GFLOPS |
| Hardware Flops | 14.999 GFLOPS |

---

### Vector Engine Stats

| Metric | Value |
| --- | --- |
| Vector Engine Active Time | 1.04 ms |
| Vector Engine Instruction Time | 1.36 ms |
| Vector Engine Active Time Percent | 22.02% |
| Vector Engine Instruction Count | 6,408 |

---

### Tensor Engine Stats

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 650.93 µs |
| Tensor Engine Instruction Time | 1.49 ms |
| Mfu Estimated Percent | 0.93% |
| Hfu Estimated Percent | 2.01% |
| Tensor Engine Active Time Percent | 13.75% |
| Mfu Max Achievable Estimated Percent | 100.00% |
| Matmul Instruction Count | 3,576 |
| Tensor Engine Instruction Count | 7,530 |

---

### DMA / Memory Stats

| Metric | Value |
| --- | --- |
| Mbu Min Read Util Percent | 0.33% |
| DMA Active Time | 4.09 ms |
| DMA Transfer Time | 4.46 ms |
| Mbu Estimated Percent | 0.53% |
| DMA Packet Time | 50.74 ms |
| DMA Active Time Percent | 86.38% |
| Dynamic DMA Size Percent | 98.54% |
| Dynamic DMA Packet Percent | 100.00% |
| DMA Queue Count | 35 |
| Static DMA Packet Count | 45 |
| DMA Transfer Count | 950 |
| Psum Read Sbuf Write Count | 2,176 |

# Round 5

## Optimization Summary

Baseline: v4a — 4.74ms, 86.38% DMA active, 13.75% TensorE active, 950 DMA transfers, 50.74ms DMA Packet Time.
Root bottlenecks: (1) 400 broadcast DMA stalls in ki inner loop (lines 551/598 in v4a — latency-bound serial chain: VectorE reduce → DMA broadcast → VectorE ops, 5 ki iters × 4 kvh × 2 gi × 5 qsi = 200 iters × 2 broadcasts = 400 stalls); (2) 1,280 hidden HBM loads in hot path (Phase 1 K, Phase 1 V, Phase 2 Q all redundantly loading the same hidden tiles across kvh heads).

---

### Plan A — v5a_qwen3_attn_cte_fused.py

Compound: hidden preload + merged K+V Phase 1 loop + 4×-wide flash attention + [PMAX,1] running state

- **Change 1 — Hidden tile preload**: Preloads all 80 hidden tiles (5si×16ht×32KB=2.56MB) as pre-transposed [PMAX,PMAX] bf16 into SBUF before the kvh loop using affine_range. Phase 1 K, Phase 1 V, and Phase 2 Q all read from SBUF — eliminates 1,280 hot-path HBM loads.
- **Change 2 — Merged K+V ht loop**: Single ht loop in Phase 1 where `hidden_tiles[si][ht]` feeds both `nc_matmul(kp)` and `nc_matmul(vp)` — halves the inner ht loop iterations in Phase 1.
- **Change 3 — [PMAX, 1] running state**: `rmax` and `rsum` changed from [PMAX, PMAX] (with identical columns) to [PMAX, 1] scalars, eliminating 128× wasteful identical-column maintenance. Broadcasts to [PMAX, PMAX] and [PMAX, 4*PMAX] done explicitly only when needed (alp for atacc rescale, nnmax for score shift).
- **Change 4 — 4×-wide batched ki loop**: Batches 4 K tiles into K_wide [PMAX, 512], one MM1, one affine_select with `pattern=[[-1, 4*PMAX]]`, one reduce+2 broadcasts per 4-tile batch. For S=640: 1 batch + 1 tail = 2 ki iterations instead of 5. Broadcast DMAs: 400 → 160 (2.5×). MM2 accumulates 4 V tiles into single PSUM (no intermediate SBUF copies).
- **Change 5 — Normalization**: rsum [PMAX,1] → inv computed at [PMAX,1] then broadcast to [PMAX,PMAX] for final multiply.

Correctness: PASS max_diff=2.44e-04

---

### Plan B — v5b_qwen3_attn_cte_fused.py

Plan A + pre-transposed weight layout change (interface modification).

All of v5a changes, plus:
- **Weight layout**: Wq `[Hq_tp*d, H]` → `[H, Hq_tp*d]`, Wk/Wv `[Hkv_tp*d, H]` → `[H, Hkv_tp*d]`. Caller passes `W.T.contiguous()`.
- **Wq hoisting**: `dma_copy(wqTs, Wq[ho:ho+PMAX, qh*d:qh*d+PMAX])` directly — no nc_transpose + tensor_copy. Saves 8×16=128 nc_transpose + 128 tensor_copy ops.
- **Wk/Wv hoisting**: same pattern per kvh — saves 4×16×2=128 nc_transpose + 128 tensor_copy ops.
- Total ops eliminated from kernel setup: 512 (256 nc_transpose + 256 tensor_copy).

Correctness: PASS max_diff=2.44e-04

# Round 5

## Optimization Summary

Plan A — v5a_qwen3_attn_cte_fused.py

Five compound changes vs v4a, targeting the two major bottlenecks simultaneously:

```python
┌──────────────────────────────────────────────┬──────────────────────────────────────────────────────────────┐
│                    Change                    │                      What's eliminated                       │
├──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ Hidden tile preload (80 tiles into SBUF)     │ 1,280 HBM DMA loads from hot path (Phase 1 K + V, Phase 2 Q) │
├──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ Merged K+V ht loop in Phase 1                │ 50% of Phase 1 inner-loop iterations                         │
├──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ [PMAX,1] running state (rmax, rsum)          │ 128× wasteful identical-column [PMAX,PMAX] maintenance       │
├──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 4×-wide ki batching (K_wide [128,512])       │ ki iterations 5→2, broadcast DMA stalls 400→160 (2.5×)       │
├──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ Single-PSUM MM2 (4 nc_matmuls into one PSUM) │ 3 intermediate SBUF copies per batch                         │
└──────────────────────────────────────────────┴──────────────────────────────────────────────────────────────┘
```

The affine_select causal mask extends naturally to pattern=[[-1, 4*PMAX]] for the wide tile. Function signature unchanged — drop-in replacement.

Plan B — v5b_qwen3_attn_cte_fused.py

Everything in Plan A, plus pre-transposed weight layout:

- New signature: Wq [H, Hq_tp*d], Wk [H, Hkv_tp*d], Wv [H, Hkv_tp*d]
- Caller: pass Wq.T.contiguous() etc. (you said you can change this at compile time)
- Eliminates: 256 nc_transpose + 256 tensor_copy = 512 ops from kernel setup (Wq hoisting, Wk/Wv hoisting per kvh)

Both pass max_diff=2.44e-04 for B=1, S=640.

## Data (v5a)

### Overall Stats

| Metric | Value |
| --- | --- |
| Model Flops | 0.000 FLOPS |
| Total Time | 4.23 ms |
| Peak Flops Bandwidth Ratio | 109.837 FLOPS |
| Mm Arithmetic Intensity | 566 |
| Event Count | 11,407 |
| Neuroncore Cycle Count | 5,074,837 |
| Trace Count | 5,313,464 |
| Transpose Flops | 3.003 GFLOPS |
| Hardware Flops | 10.167 GFLOPS |

---

### Vector Engine Stats

| Metric | Value |
| --- | --- |
| Vector Engine Active Time | 631.95 µs |
| Vector Engine Instruction Time | 758.87 µs |
| Vector Engine Active Time Percent | 14.94% |
| Vector Engine Instruction Count | 3,441 |

---

### DMA / Memory Stats

| Metric | Value |
| --- | --- |
| DMA Active Time | 3.28 ms |
| DMA Transfer Time | 3.33 ms |
| Mbu Min Read Util Percent | 0.37% |
| Mbu Estimated Percent | 0.42% |
| DMA Packet Time | 39.37 ms |
| DMA Active Time Percent | 77.48% |
| Dynamic DMA Size Percent | 98.75% |
| Dynamic DMA Packet Percent | 100.00% |
| Static DMA Packet Count | 31 |
| DMA Queue Count | 35 |
| DMA Transfer Count | 590 |
| Psum Read Sbuf Write Count | 904 |

---

### Memory Traffic

| Metric | Value |
| --- | --- |
| Software Dynamic DMA Packet Count | 12,416 |
| Weight Size Bytes | 16.38 KB |
| DMA Transfer Average Bytes | 57.21 KB |
| Psum Read Bytes | 340.99 KB |
| Psum Read Sbuf Write Bytes | 341.01 KB |
| Static DMA Size | 428.05 KB |
| Psum Write Bytes | 620.54 KB |
| HBM Write Bytes | 1.31 MB |
| DMA Active Cycles | 3,932,113 |
| Hardware Dynamic DMA Packet Count | 5,278,880 |
| HBM Read Bytes | 11.34 MB |
| Inputs And Weights Size Bytes | 11.35 MB |
| Software Dynamic DMA Size | 12.67 MB |
| Hardware Dynamic DMA Size | 21.12 MB |
| SBUF Read Bytes | 24.33 MB |
| SBUF Write Bytes | 33.21 MB |
| DMA Transfer Total Bytes | 33.75 MB |

## Data (v5b)

### Overall Stats

| Metric | Value |
| --- | --- |
| Model Flops | 0.000 FLOPS |
| Total Time | 3.95 ms |
| Peak Flops Bandwidth Ratio | 109.837 FLOPS |
| Mm Arithmetic Intensity | 566 |
| Event Count | 11,390 |
| Neuroncore Cycle Count | 4,744,681 |
| Trace Count | 5,312,431 |
| Transpose Flops | 1.929 GFLOPS |
| Hardware Flops | 9.093 GFLOPS |

---

### Tensor Engine Stats

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 332.38 µs |
| Tensor Engine Instruction Time | 890.78 µs |
| Mfu Estimated Percent | 1.15% |
| Hfu Estimated Percent | 1.46% |
| Tensor Engine Active Time Percent | 8.41% |
| Mfu Max Achievable Estimated Percent | 100.00% |
| Matmul Instruction Count | 2,048 |
| Tensor Engine Instruction Count | 4,212 |

---

### DMA / Memory Stats

| Metric | Value |
| --- | --- |
| DMA Active Time | 3.26 ms |
| DMA Transfer Time | 3.40 ms |
| Mbu Min Read Util Percent | 0.40% |
| Mbu Estimated Percent | 0.45% |
| DMA Packet Time | 39.74 ms |
| DMA Active Time Percent | 82.35% |
| Dynamic DMA Size Percent | 98.60% |
| Dynamic DMA Packet Percent | 100.00% |
| DMA Queue Count | 35 |
| Static DMA Packet Count | 35 |
| DMA Transfer Count | 590 |
| Psum Read Sbuf Write Count | 648 |

---

### Memory Traffic

| Metric | Value |
| --- | --- |
| Software Dynamic DMA Packet Count | 12,416 |
| Weight Size Bytes | 16.38 KB |
| DMA Transfer Average Bytes | 57.21 KB |
| Psum Read Sbuf Write Bytes | 271.39 KB |
| Psum Read Bytes | 275.46 KB |
| Static DMA Size | 478.22 KB |
| Psum Write Bytes | 555.01 KB |
| HBM Write Bytes | 1.31 MB |
| DMA Active Cycles | 3,907,274 |
| Hardware Dynamic DMA Packet Count | 5,278,880 |
| HBM Read Bytes | 11.34 MB |
| Inputs And Weights Size Bytes | 11.35 MB |
| Software Dynamic DMA Size | 12.67 MB |
| Hardware Dynamic DMA Size | 21.12 MB |
| SBUF Read Bytes | 24.20 MB |
| SBUF Write Bytes | 33.14 MB |
| DMA Transfer Total Bytes | 33.75 MB |

# Round 6

## No Flash Attention

Plan A — Full-Width Score Materialization (No Flash Attention)

Bottleneck targeted: 240 .ap() DMA broadcast stalls in the online flash attention ki loop (lines 592,

604, 712, 723, 787)

Root cause: Online softmax requires per-ki-iteration rescaling of atacc, rmax, rsum, each needing a

[PMAX,1]→[PMAX,PMAX] broadcast. The serial dependency chain VectorE→ScalarE→DMA→VectorE creates pipeline
bubbles that cannot be overlapped because range() (not affine_range) is forced by the loop-carried

dependency.

Change: Replace the online flash attention with full score-row materialization + standard two-pass

softmax.

### Overall Stats

| Metric | Value |
| --- | --- |
| Model FLOPs | 0.000 FLOPS |
| Total Time | 3.23 ms |
| Peak FLOPs Bandwidth Ratio | 109.837 FLOPS |
| MM Arithmetic Intensity | 566 |
| Event Count | 9,496 |
| Neuroncore Cycle Count | 3,879,611 |
| Trace Count | 3,997,419 |
| Transpose FLOPs | 1.929 GFLOPS |
| Hardware FLOPs | 9.093 GFLOPS |

### Tensor Engine

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 315.82 µs |
| Tensor Engine Instruction Time | 890.84 µs |
| MFU Estimated Percent | 1.41% |
| HFU Estimated Percent | 1.79% |
| Tensor Engine Active Time Percent | 9.77% |
| MFU Max Achievable Estimated Percent | 100.00% |
| Matmul Instruction Count | 2,048 |
| Tensor Engine Instruction Count | 4,216 |

### Vector Engine

| Metric | Value |
| --- | --- |
| Vector Engine Active Time | 482.66 µs |
| Vector Engine Instruction Time | 580.15 µs |
| Vector Engine Active Time Percent | 14.93% |
| Vector Engine Instruction Count | 2,461 |

### DMA / Memory Stats

| Metric | Value |
| --- | --- |
| DMA Active Time | 2.48 ms |
| DMA Transfer Time | 2.59 ms |
| MBU Min Read Util Percent | 0.49% |
| MBU Estimated Percent | 0.55% |
| DMA Packet Time | 30.02 ms |
| DMA Active Time Percent | 76.84% |
| Dynamic DMA Size Percent | 98.35% |
| Dynamic DMA Packet Percent | 100.00% |
| DMA Queue Count | 35 |
| Static DMA Packet Count | 36 |
| DMA Transfer Count | 510 |
| PSUM Read SBUF Write Count | 648 |
| Software Dynamic DMA Packet Count | 12,416 |
| Weight Size Bytes | 16.38 KB |
| DMA Transfer Average Bytes | 55.90 KB |
| PSUM Read Bytes | 275.46 KB |
| PSUM Read SBUF Write Bytes | 281.61 KB |
| Static DMA Size | 479.25 KB |
| PSUM Write Bytes | 555.01 KB |
| HBM Write Bytes | 1.31 MB |
| DMA Active Cycles | 2,981,031 |
| Hardware Dynamic DMA Packet Count | 3,966,880 |
| HBM Read Bytes | 11.34 MB |
| Inputs and Weights Size Bytes | 11.35 MB |
| Software Dynamic DMA Size | 12.67 MB |
| Hardware Dynamic DMA Size | 15.87 MB |
| SBUF Read Bytes | 18.92 MB |
| SBUF Write Bytes | 27.86 MB |
| DMA Transfer Total Bytes | 28.51 MB |

---

## Flash Attn

v6bcd — Plans C+D: Free-dim RMSNorm + Pre-scored Flash Attention

File: kernels/attn_cte/agents/v6bcd_qwen3_attn_cte_fused.py
Correctness: PASS, max_diff=2.44e-04

Changes from v5b:

- Plan C: RMSNorm uses tensor_reduce(axis=1) in native [S,d] layout — eliminates 2 nc_transpose + 1 nc_matmul + 4 tensor_copy per invocation (60 invocations total = ~600 ops removed). Norm weights transposed at setup to [S,d]-compatible layout.
- Plan D: All score tiles pre-computed with nl.affine_range(num_s_tiles) before sequential softmax loop. Decouples TensorE MM1 pipelining from serial VectorE chain.
- Plan B: nc_matmul-based broadcast failed — compiler requires matching partition dimensions between stationary and moving inputs ([PMAX,1] × [1,PMAX] is invalid). DMA .ap() broadcasts retained.

Plan B failure note

The nc_matmul(stationary=[PMAX,1], moving=[1,PMAX]) approach hits a hardware/compiler constraint: "Fmap
and Weight partitions must match." The partition dimension of both inputs must be PMAX. This makes
TensorE-based column→matrix broadcast infeasible with the current API. The DMA .ap() broadcast remains
the only mechanism for [PMAX,1]→[PMAX,PMAX] expansion.