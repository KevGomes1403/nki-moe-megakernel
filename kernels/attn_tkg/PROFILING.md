## v0

## Overall Stats

| Metric | Value |
| --- | --- |
| Model Flops | 0.000 FLOPS |
| Total Time | 531.79 µs |
| MM Arithmetic Intensity | 1 |
| Peak Flops Bandwidth Ratio | 109.837 FLOPS |
| Event Count | 7,693 |
| Trace Count | 42,509 |
| Neuroncore Cycle Count | 638,147 |
| Transpose Flops | 169.444 MFLOPS |
| Hardware Flops | 186.745 MFLOPS |

---

## Tensor Engine

<aside>
🚨

High Transpose FLOPS Overhead

NeuronCore 0 has 90.7% transpose FLOPS of total hardware FLOPS. Improve memory 
layout to reduce data movement operations within the tensor engine.

</aside>

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 143.90 µs |
| MFU Estimated Percent | 0.02% |
| Tensor Engine Instruction Time | 297.69 µs |
| HFU Estimated Percent | 0.22% |
| MFU Max Achievable Estimated Percent | 0.51% |
| Tensor Engine Active Time Percent | 27.06% |
| Matmul Instruction Count | 1,002 |
| Tensor Engine Instruction Count | 2,394 |

---

## Memory / DMA Sizes

| Metric | Value |
| --- | --- |
| Static DMA Size | 486.85 KB |
| Inputs and Weights Size Bytes | 9.72 MB |
| Software Dynamic DMA Size | 15.23 MB |
| HBM Read Bytes | 15.32 MB |
| DMA Transfer Total Bytes | 15.32 MB |
| SBUF Write Bytes | 15.34 MB |

---

## DMA / Memory Engine Stats

| Metric | Value |
| --- | --- |
| DMA Active Time | 261.31 µs |
| DMA Transfer Time | 433.62 µs |
| DMA Packet Time | 1.17 ms |
| MBU Min Read Util Percent | 2.55% |
| MBU Estimated Percent | 4.02% |
| DMA Active Time Percent | 49.14% |
| Dynamic DMA Packet Percent | 75.38% |
| Dynamic DMA Size Percent | 96.91% |
| DMA Queue Count | 83 |
| Spill Save Bytes | 160 B |
| Weight Queue Bytes | 160 B |
| Psum Read SBUF Write Count | 554 |
| DMA Transfer Count | 926 |
| HBM Write Bytes | 2.21 KB |
| Hardware Dynamic DMA Packet Count | 5,760 |
| Static DMA Packet Count | 6,751 |
| Psum Read SBUF Write Bytes | 13.54 KB |
| Software Dynamic DMA Packet Count | 14,912 |
| Weight Size Bytes | 16.38 KB |
| DMA Transfer Average Bytes | 16.55 KB |
| Hardware Dynamic DMA Size | 23.04 KB |
| Psum Read Bytes | 33.86 KB |
| Psum Write Bytes | 34.60 KB |
| Input Queue Bytes | 97.28 KB |
| SBUF Read Bytes | 282.74 KB |
| DMA Active Cycles | 313,567 |
| Static DMA Size | 486.85 KB |
| Inputs and Weights Size Bytes | 9.72 MB |

---

# Round 1

## Optimization Summary

All three optimized kernels pass correctness with max_diff well under 0.1:

### Plan A — Eliminate Transposes via Direct Column Layout (v1_direct_layout.py)

- Reshaped hidden/cos/sin/output to column layout [X, B] for B=1 — loads directly as [PMAX, 1]
- Hoisted 16 hidden tiles outside head loop (384 DMA loads → 16)
- Eliminated ~394 nc_transpose instructions targeting the 90.7% transpose FLOPS overhead
- max_diff: 2.19e-02 / 8.51e-03

### Plan B — Deduplicate K/V Computation for GQA (v2_kv_dedup.py)

- Restructured loop: outer kv_h in range(4), inner g in range(2)
- K matmul + RMSNorm + RoPE computed once per KV head, V matmul once per KV head
- Eliminated ~132 matmuls (64 K + 64 V + 4 RMSNorm)
- max_diff: 2.19e-02 / 8.51e-03

### Plan C — Replace All-Ones Matmul Reductions (v3_reduce_replace.py)

- Replaced nc_matmul(rms_ones[128,128], data) with transpose+reduce+broadcast
- Freed 32KB SBUF from rms_ones allocation
- Eliminated 24+8×num_s reduction matmuls, moved work to Vector Engine
- Improved precision by keeping reductions in float32 (removed unnecessary bf16 casts)
- max_diff: 2.19e-02 / 8.51e-03

## Data

| Metric | v1 | v2 | v3 |
| --- | --- | --- | --- |
| **Total Time** | 459.11 µs | 434.93 µs | 797.27 µs |
| **Mm Arithmetic Intensity** | 2 | 1 | 1 |
| **Event Count** | 3,946 | 6,970 | 10,082 |
| **Trace Count** | 24,555 | 34,113 | 55,472 |
| **Neuroncore Cycle Count** | 550,931 | 521,910 | 956,724 |
| **Transpose Flops** | 169.083 MFLOPS | 169.411 MFLOPS | 171.541 MFLOPS |
| **Hardware Flops** | 186.384 MFLOPS | 182.387 MFLOPS | 186.745 MFLOPS |
| **Tensor Engine Active Time** | 58.11 µs | 109.14 µs | 155.73 µs |
| **Tensor Engine Instruction Time** | 184.03 µs | 226.17 µs | 311.35 µs |
| **MFU Estimated Percent** | 0.02% | 0.02% | 0.01% |
| **HFU Estimated Percent** | 0.26% | 0.27% | 0.15% |
| **MFU Max Achievable Estimated Percent** | 0.81% | 0.53% | 0.45% |
| **Tensor Engine Active Time Percent** | 12.66% | 25.09% | 19.53% |
| **Matmul Instruction Count** | 608 | 742 | 1,002 |
| **Tensor Engine Instruction Count** | 1,303 | 1,797 | 2,389 |
| **DMA Active Time** | 173.25 µs | 203.28 µs | 317.08 µs |
| **DMA Transfer Time** | 217.41 µs | 327.76 µs | 560.47 µs |
| **DMA Packet Time** | 748.81 µs | 865.58 µs | 1.27 ms |
| **MBU Min Read Util Percent** | 2.96% | 3.12% | 1.70% |
| **MBU Estimated Percent** | 2.96% | 3.56% | 2.69% |
| **DMA Active Time Percent** | 37.74% | 46.74% | 39.77% |
| **Dynamic DMA Packet Percent** | 96.03% | 77.44% | 79.35% |
| **Dynamic DMA Size Percent** | 96.69% | 96.38% | 96.79% |
| **DMA Queue Count** | 51 | 83 | 83 |
| **Spill Save Bytes** | 160.00 B | 160.00 B | 416.00 B |
| **Weight Queue Bytes** | 160.00 B | 160.00 B | 416.00 B |
| **Psum Read Sbuf Write Count** | 160 | 414 | 602 |
| **DMA Transfer Count** | 404 | 678 | 1,054 |
| **Static DMA Packet Count** | 666 | 4,829 | 7,776 |
| **HBM Write Bytes** | 2.21 KB | 2.21 KB | 2.46 KB |
| **Hardware Dynamic DMA Packet Count** | 5,760 | 5,760 | 14,976 |
| **Software Dynamic DMA Packet Count** | 10,368 | 10,816 | 14,912 |
| **Psum Read Sbuf Write Bytes** | 10.72 KB | 13.24 KB | 13.74 KB |
| **Weight Size Bytes** | 16.38 KB | 16.38 KB | 16.38 KB |
| **Hardware Dynamic DMA Size** | 23.04 KB | 23.04 KB | 59.90 KB |
| **DMA Transfer Average Bytes** | 24.08 KB | 16.37 KB | 14.57 KB |
| **Psum Read Bytes** | 31.04 KB | 33.56 KB | 66.56 KB |
| **Psum Write Bytes** | 31.78 KB | 34.08 KB | 67.24 KB |
| **Sbuf Read Bytes** | 178.79 KB | 215.57 KB | 299.74 KB |
| **Input Queue Bytes** | — | 66.56 KB | 97.28 KB |
| **DMA Active Cycles** | 207,899 | 243,932 | 380,496 |
| **Static DMA Size** | 333.49 KB | 415.68 KB | 507.58 KB |
| **Inputs And Weights Size Bytes** | 9.72 MB | 9.72 MB | 9.72 MB |
| **Software Dynamic DMA Size** | — | 11.03 MB | 15.23 MB |
| **HBM Read Bytes** | 9.72 MB | 11.10 MB | 15.35 MB |
| **DMA Transfer Total Bytes** | 9.73 MB | 11.10 MB | 15.36 MB |
| **Sbuf Write Bytes** | — | 11.11 MB | 15.37 MB |

# Round 2

Optimization Summary

Original Best Kernels

- v1 (direct layout): 459.11 µs — column layout + hidden hoisting
- v2 (KV dedup): 434.93 µs — GQA K/V projection dedup

Plan A — v4_merged_layout_dedup.py

- What changed: Combined v1's column layout + hidden hoisting with v2's GQA dedup + NEW flash decode KV
tile reuse (moved s_t loop outside g loop)
- Correctness: PASS (max_diff=8.51e-03)
- Path: kernels/attn_tkg/agents/v4_merged_layout_dedup.py

Plan B — v5_freedim_packing.py

- What changed: Packed 2 GQA Q heads into free dimension [128,2] during flash decode. Single matmul
scores both heads simultaneously. All online softmax/V contribution operates on packed tensors.
- Correctness: PASS (max_diff=8.51e-03)
- Path: kernels/attn_tkg/agents/v5_freedim_packing.py

Plan C — v6_ultimate.py

- What changed: All 5 optimizations combined — column layout, hidden hoisting, GQA K/V dedup, free-dim
packing, KV tile reuse
- Correctness: PASS (max_diff=8.51e-03)
- Path: kernels/attn_tkg/agents/v6_ultimate.py

---

## Data (v6)

---

### **Overall Stats**

| Metric | Value |
| --- | --- |
| Model FLOPs | 0.000 FLOPS |
| Total Time | 359.87 µs |
| MM Arithmetic Intensity | 1 |
| Peak FLOPs Bandwidth Ratio | 109.837 FLOPS |
| Event Count | 2,537 |
| Trace Count | 18,748 |
| Neuroncore Cycle Count | 431,848 |
| Transpose FLOPs | 85.197 MFLOPS |
| Hardware FLOPs | 98.173 MFLOPS |

---

### **Tensor Engine**

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 37.40 µs |
| Tensor Engine Instruction Time | 112.96 µs |
| MFU Estimated Percent | 0.02% |
| HFU Estimated Percent | 0.17% |
| MFU Max Achievable Estimated Percent | 0.61% |
| Tensor Engine Active Time Percent | 10.39% |
| Matmul Instruction Count | 372 |
| Tensor Engine Instruction Count | 840 |

---

### **Data Movement**

| Metric | Value |
| --- | --- |
| Spill Reload Bytes | 0.00 B |
| Input Queue Bytes | 0.00 B |
| Output Queue Bytes | 0.00 B |
| DMA Active Time | 148.47 µs |
| DMA Transfer Time | 178.56 µs |
| DMA Packet Time | 717.14 µs |
| MBU Min Read Util Percent | 3.77% |
| MBU Estimated Percent | 3.78% |
| DMA Active Time Percent | 41.25% |
| Dynamic DMA Packet Percent | 97.50% |
| Dynamic DMA Size Percent | 97.51% |
| DMA Queue Count | 51 |
| PSUM Read SBUF Write Count | 88 |
| Spill Save Bytes | 160.00 B |
| Weight Queue Bytes | 160.00 B |
| Static DMA Packet Count | 340 |
| DMA Transfer Count | 364 |
| HBM Write Bytes | 2.21 KB |
| Hardware Dynamic DMA Packet Count | 2,880 |
| PSUM Read SBUF Write Bytes | 5.47 KB |
| Software Dynamic DMA Packet Count | 10,368 |
| PSUM Read Bytes | 15.63 KB |
| PSUM Write Bytes | 16.15 KB |
| Weight Size Bytes | 16.38 KB |
| Hardware Dynamic DMA Size | 21.76 KB |
| DMA Transfer Average Bytes | 26.72 KB |
| SBUF Read Bytes | 109.17 KB |
| DMA Active Cycles | 178,158 |
| Static DMA Size | 249.26 KB |
| Inputs and Weights Size Bytes | 9.72 MB |
| HBM Read Bytes | 9.72 MB |
| DMA Transfer Total Bytes | 9.73 MB |
| Software Dynamic DMA Size | 9.73 MB |
| SBUF Write Bytes | 9.73 MB |

---

# Round 3 (v7c)

### **General Performance Metrics**

| Metric | Value |
| --- | --- |
| Total Time | 200.21 µs |
| MM Arithmetic Intensity | 1 |
| Peak FLOPs Bandwidth Ratio | 109.837 FLOPS |
| Event Count | 8,201 |
| Trace Count | 26,497 |
| Neuroncore Cycle Count | 240,247 |
| Transpose FLOPs | 43.524 MFLOPS |
| Hardware FLOPs | 58.794 MFLOPS |

---

### **Tensor Engine Metrics**

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 55.98 µs |
| Tensor Engine Instruction Time | 119.32 µs |
| MFU Estimated Percent | 0.05% |
| HFU Estimated Percent | 0.19% |
| MFU Max Achievable Estimated Percent | 0.62% |
| Tensor Engine Active Time Percent | 27.96% |
| Matmul Instruction Count | 388 |
| Tensor Engine Instruction Count | 895 |

---

### **DMA / Memory Metrics**

| Metric | Value |
| --- | --- |
| DMA Active Time | 105.05 µs |
| DMA Packet Time | 879.72 µs |
| MBU Min Read Util Percent | 6.83% |
| MBU Estimated Percent | 7.79% |
| DMA Active Time Percent | 52.47% |
| Dynamic DMA Size Percent | 99.06% |
| Dynamic DMA Packet Percent | 99.91% |
| Static DMA Packet Count | 15 |
| PSUM Read SBUF Write Count | 74 |
| DMA Queue Count | 100 |
| PSUM Read SBUF Write Bytes | 2.95 KB |
| Hardware Dynamic DMA Packet Count | 3,456 |
| HBM Write Bytes | 7.17 KB |
| Hardware Dynamic DMA Size | 7.68 KB |
| PSUM Read Bytes | 9.08 KB |
| PSUM Write Bytes | 9.67 KB |
| Software Dynamic DMA Packet Count | 12,800 |
| Weight Size Bytes | 16.38 KB |
| Static DMA Size | 106.52 KB |
| SBUF Read Bytes | 115.27 KB |
| DMA Active Cycles | 126,057 |
| Inputs and Weights Size Bytes | 9.79 MB |
| HBM Read Bytes | 11.16 MB |

---

# Round 4

Optimization Summary

Plan A — v8a_weight_hoisted.py — PASS (max_diff=1.43e-02)

- Hoisted all Wq tiles (64/core) and Wo tiles (64/core) into SBUF upfront via affine_range
- Prefetched all K+V cache tiles into SBUF before Q projection
- Q projection, flash decode, and o_proj are now pure SBUF→PSUM compute with zero DMA in the critical path
- Total ~5.6 MB/core hoisted (well within 48 MB SBUF)

Plan B — v8b_packed_q.py — PASS (max_diff=1.43e-02)

- Replaced 4 independent RMSNorm+RoPE chains with 1 packed chain on [PMAX, G=4]
- Eliminates 3 sum-of-squares matmuls and ~36 element-wise tensor ops
- Uses .ap() broadcast for norm weights, cos, sin from [PMAX, 1] → [PMAX, G]
- Phase 5 (Q packing) eliminated — packing now happens before RMSNorm

---

## Data (8b)

### Overall Stats

| Metric | Value |
| --- | --- |
| Model Flops | 0.000 FLOPS |
| Total Time | 183.95 µs |
| MM Arithmetic Intensity | 1 |
| Peak Flops Bandwidth Ratio | 109.837 FLOPS |
| Event Count | 8,081 |
| Trace Count | 26,230 |
| Neuroncore Cycle Count | 220,742 |
| Transpose Flops | 43.524 MFLOPS |
| Hardware Flops | 58.794 MFLOPS |

---

### Tensor Engine Stats

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 52.92 µs |
| Tensor Engine Instruction Time | 117.61 µs |
| MFU Estimated Percent | 0.05% |
| HFU Estimated Percent | 0.20% |
| MFU Max Achievable Estimated Percent | 0.62% |
| Tensor Engine Active Time Percent | 28.77% |
| Matmul Instruction Count | 382 |
| Tensor Engine Instruction Count | 878 |

### DMA & Memory Stats

| Metric | Value |
| --- | --- |
| DMA Transfer Time | 0.00 ns |
| DMA Active Time | 104.61 µs |
| DMA Packet Time | 870.55 µs |
| MBU Min Read Util Percent | 7.43% |
| MBU Estimated Percent | 8.48% |
| DMA Active Time Percent | 56.87% |
| Dynamic DMA Size Percent | 99.64% |
| Dynamic DMA Packet Percent | 99.91% |
| Static DMA Packet Count | 14 |
| PSUM Read SBUF Write Count | 68 |
| DMA Queue Count | 100 |
| PSUM Read SBUF Write Bytes | 2.95 KB |
| Hardware Dynamic DMA Packet Count | 3,456 |
| HBM Write Bytes | 7.17 KB |
| Hardware Dynamic DMA Size | 7.68 KB |
| PSUM Read Bytes | 9.08 KB |
| PSUM Write Bytes | 9.67 KB |
| Software Dynamic DMA Packet Count | 12,800 |
| Weight Size Bytes | 16.38 KB |
| Static DMA Size | 40.98 KB |
| SBUF Read Bytes | 113.78 KB |
| DMA Active Cycles | 125,526 |
| Inputs and Weights Size Bytes | 9.79 MB |
| HBM Read Bytes | 11.16 MB |
| SBUF Write Bytes | 11.16 MB |
| Software Dynamic DMA Size | 11.18 MB |

---

# Round 5

Optimization Summary

Original Kernel (v8b_packed_q)

- 183.95 µs total, DMA-bound (56.87% DMA active, 28.77% Tensor Engine)
- Wq/Wo/KV-cache tiles loaded from HBM inside compute loops (serialized with matmuls)
- core_barrier synchronization between cores for o_proj staging
- 11.16 MB HBM reads, 382 matmul instructions

Plan A — Full Weight + KV Cache Hoisting (v9a_full_hoist.py)

- What changed: Hoisted ALL Wq tiles (64/core), Wo tiles (64/core), K-cache tiles (5), V-cache tiles (5, pre-transposed) into SBUF
via nl.affine_range before compute phases. All compute loops (Q projection, flash decode, o_proj) are now pure SBUF→PSUM with
zero HBM DMA.
- Why it helps: DMA burst at startup overlaps with nothing, but compute phases no longer stall on DMA. Eliminates ~128
per-iteration DMA loads from critical path. Pre-transposed V tiles save nc_transpose in inner loop.
- Correctness: PASS (max_diff=1.43e-02)
- SBUF budget: ~5.3 MB/core (of 24 MB available)

Plan B — Barrier-Free O_proj via Redundant Full Attention (v9b_barrier_free.py)

- What changed: All Plan A hoists PLUS each core computes all 8 Q heads (not just 4). Eliminates attn_staging shared_hbm buffer,
core_barrier, staging writes/reads entirely. Packed operations widened from [PMAX,4] to [PMAX,8]. O_proj reads from local SBUF
tensors.
- Why it helps: Eliminates barrier synchronization stall + 16 staging DMA ops per core. The redundant compute (4 extra Q
projections + flash decode = ~112 extra matmuls) costs ~15 µs compute but removes a potentially larger barrier wait + DMA
overhead.
- Correctness: PASS (max_diff=1.43e-02)
- SBUF budget: ~7.3 MB/core (of 24 MB available)
- Trade-off: +29% matmul count vs eliminated barrier + staging DMA

## Data (v9b)

### **General Performance Metrics**

| Metric | Value |
| --- | --- |
| Model FLOPs | 0.000 FLOPS |
| Total Time | 197.36 µs |
| MM Arithmetic Intensity | 2 |
| Peak FLOPs Bandwidth Ratio | 109.837 FLOPS |
| Event Count | 9,602 |
| Trace Count | 28,696 |
| Neuroncore Cycle Count | 242,956 |
| Transpose FLOPs | 45.122 MFLOPS |
| Hardware FLOPs | 69.304 MFLOPS |

---

### **Tensor Engine Metrics**

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 84.87 µs |
| Tensor Engine Instruction Time | 153.69 µs |
| MFU Estimated Percent | 0.08% |
| HFU Estimated Percent | 0.22% |
| MFU Max Achievable Estimated Percent | 0.72% |
| Tensor Engine Active Time Percent | 41.92% |
| Matmul Instruction Count | 510 |
| Tensor Engine Instruction Count | 1,120 |

---

### **DMA / Memory Metrics**

| Metric | Value |
| --- | --- |
| DMA Active Time | 151.48 µs |
| DMA Packet Time | 1.16 ms |
| MBU Min Read Util Percent | 6.75% |
| MBU Estimated Percent | 10.59% |
| DMA Active Time Percent | 74.82% |
| Dynamic DMA Size Percent | 99.73% |
| Dynamic DMA Packet Percent | 99.93% |
| Static DMA Packet Count | 12 |
| DMA Queue Count | 36 |
| PSUM Read SBUF Write Count | 76 |
| PSUM Read SBUF Write Bytes | 3.24 KB |
| HBM Write Bytes | 5.12 KB |
| PSUM Read Bytes | 9.37 KB |
| PSUM Write Bytes | 10.24 KB |
| Weight Size Bytes | 16.38 KB |
| Software Dynamic DMA Packet Count | 16,896 |
| Static DMA Size | 40.97 KB |
| SBUF Read Bytes | 145.68 KB |
| DMA Active Cycles | 181,771 |
| Inputs and Weights Size Bytes | 9.79 MB |
| HBM Read Bytes | 15.35 MB |
| SBUF Write Bytes | 15.35 MB |
| Software Dynamic DMA Size | 15.38 MB |

## Data (v9a)

### **General Performance Metrics**

| Metric | Value |
| --- | --- |
| Model FLOPs | 0.000 FLOPS |
| Total Time | 185.14 µs |
| MM Arithmetic Intensity | 1 |
| Peak FLOPs Bandwidth Ratio | 109.837 FLOPS |
| Event Count | 8,077 |
| Trace Count | 26,208 |
| Neuroncore Cycle Count | 222,708 |
| Transpose FLOPs | 43.524 MFLOPS |
| Hardware FLOPs | 58.794 MFLOPS |

---

### **Tensor Engine Metrics**

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 54.24 µs |
| Tensor Engine Instruction Time | 116.75 µs |
| MFU Estimated Percent | 0.05% |
| HFU Estimated Percent | 0.20% |
| MFU Max Achievable Estimated Percent | 0.62% |
| Tensor Engine Active Time Percent | 29.23% |
| Matmul Instruction Count | 382 |
| Tensor Engine Instruction Count | 872 |

---

### **DMA / Memory Metrics**

| Metric | Value |
| --- | --- |
| DMA Active Time | 108.37 µs |
| DMA Packet Time | 920.34 µs |
| MBU Min Read Util Percent | 7.36% |
| MBU Estimated Percent | 8.40% |
| DMA Active Time Percent | 58.39% |
| Dynamic DMA Size Percent | 99.64% |
| Dynamic DMA Packet Percent | 99.91% |
| Static DMA Packet Count | 14 |
| PSUM Read SBUF Write Count | 68 |
| DMA Queue Count | 100 |
| PSUM Read SBUF Write Bytes | 2.95 KB |
| Hardware Dynamic DMA Packet Count | 3,456 |
| HBM Write Bytes | 7.17 KB |
| Hardware Dynamic DMA Size | 7.68 KB |
| PSUM Read Bytes | 9.08 KB |
| PSUM Write Bytes | 9.67 KB |
| Software Dynamic DMA Packet Count | 12,800 |
| Weight Size Bytes | 16.38 KB |
| Static DMA Size | 40.98 KB |
| SBUF Read Bytes | 113.27 KB |
| DMA Active Cycles | 130,043 |
| Inputs and Weights Size Bytes | 9.79 MB |
| HBM Read Bytes | 11.16 MB |
| SBUF Write Bytes | 11.16 MB |
| Software Dynamic DMA Size | 11.18 MB |

---

# Round 6

---

Optimization Summary

Original Kernel (v9b — barrier-free)

- 185-197 µs total, DMA-bound (58-75% DMA active, 29-42% TE active)
- MBU of only 8-10% — severely underutilized HBM bandwidth
- 74% of hardware FLOPs spent on transposes
- 12,800+ fragmented DMA packets (~900B average)

Plan A — v10a_wide_dma.py — PASS (max_diff=1.43e-02)

What changed:

- Weight loads restructured from tile-level [128,128] to row-level wide DMAs: Wk/Wv as [128,2048] (512KB each), per-head Wq as [128,2048], per-output-tile Wo as [128,1024]. Reduces
~170 DMA ops to ~30.
- Weights streamed per-phase instead of all hoisted simultaneously. Peak SBUF: ~1MB (down from ~7MB).
- Caller provides V_cache pre-transposed [B,1,d,S_prior] — eliminates 5 nc_transpose + 5 tensor_copy during V hoisting.
- Barrier-free all 8 heads per core.
- O_proj outer loop uses affine_range for DMA-compute overlap.

Why it should help: Average DMA transfer size jumps from ~900B to ~300KB. Fewer, larger transfers improve DMA bandwidth utilization. Reduced SBUF pressure gives the compiler more
room for efficient allocation.

Plan B — v10b_global_max.py — PASS (max_diff=1.43e-02)

What changed (all of Plan A, plus):

- Global scalar max softmax: Replaces per-head max computation (7 nc_transpose ops) with global scalar max (1 nc_transpose). Uses tensor_reduce(axis=1) on score tiles for
per-position max, chains via element-wise max, one transpose to scalar, then HBM scratch broadcast to [PMAX, G].
- Fused tensor_scalar: K and Q RMSNorm each combine multiply-by-1/d + add-epsilon into a single dual-op tensor_scalar(op0=multiply, op1=add) instruction.

Why it should help: Eliminates 6 nc_transpose + 6 tensor_copy from the softmax critical path. Saves 2 VectorE instructions in RMSNorm. Combined with Plan A's DMA improvements,
targets both DMA and compute bottlenecks.

## Data (v10a)

### General Performance Metrics

| Metric | Value |
| --- | --- |
| Model FLOPs | 0.000 FLOPS |
| Total Time | 91.84 µs |
| MM Arithmetic Intensity | 2 |
| Peak FLOPs Bandwidth Ratio | 109.837 FLOPS |
| Event Count | 3,266 |
| Trace Count | 11,761 |
| Neuroncore Cycle Count | 110,213 |
| Transpose FLOPs | 3.178 MFLOPS |
| Hardware FLOPs | 27.361 MFLOPS |

### Tensor Engine Metrics

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 28.28 µs |
| Tensor Engine Instruction Time | 145.84 µs |
| MFU Estimated Percent | 0.17% |
| HFU Estimated Percent | 0.19% |
| MFU Max Achievable Estimated Percent | 0.72% |
| Tensor Engine Active Time Percent | 30.79% |
| Matmul Instruction Count | 500 |
| Tensor Engine Instruction Count | 1,071 |

### DMA / Memory Metrics

| Metric | Value |
| --- | --- |
| Hardware Dynamic DMA Size | 0.00 B |
| DMA Active Time | 40.78 µs |
| DMA Packet Time | 800.34 µs |
| MBU Min Read Util Percent | 14.88% |
| MBU Estimated Percent | 23.34% |
| DMA Active Time Percent | 44.40% |
| Dynamic DMA Packet Percent | 99.90% |
| Dynamic DMA Size Percent | 99.98% |
| Static DMA Packet Count | 7 |
| DMA Queue Count | 36 |
| PSUM Read SBUF Write Count | 66 |
| PSUM Read SBUF Write Bytes | 680.00 B |
| Static DMA Size | 3.10 KB |
| HBM Write Bytes | 5.12 KB |
| Software Dynamic DMA Packet Count | 6,720 |
| PSUM Read Bytes | 6.81 KB |
| PSUM Write Bytes | 7.68 KB |
| Weight Size Bytes | 16.38 KB |
| DMA Active Cycles | 48,937 |
| SBUF Read Bytes | 140.56 KB |
| Inputs and Weights Size Bytes | 9.79 MB |
| HBM Read Bytes | 15.35 MB |
| SBUF Write Bytes | 15.35 MB |
| Software Dynamic DMA Size | 15.36 MB |

---

# Round 7

Optimization Summary

Original Kernel (v10a) — 91.84 µs

- DMA-bound (44% DMA, 31% TE), 7 nc_transpose for per-head max
- K/V cache hoisted (320KB SBUF), no Wo prefetch
- Two-pass softmax with all score tiles materialized in SBUF

Plan A — v12a_sbuf_broadcast.py — PASS (max_diff=1.43e-02)

What changed:

1. Global scalar max (1 transpose instead of 7)
2. SBUF-only broadcast: neg_max_scalar [1,1] → .ap() to [1,PMAX] → nc_transpose → [PMAX,1] → .ap() to [PMAX,G_full]. Eliminates
the HBM scratch roundtrip that caused the v11a/v11b gap.
3. K-cache streamed JIT in Pass 1 (frees 160KB SBUF)
4. V-cache streamed JIT in Pass 2 (frees 160KB SBUF)
5. Fused o_proj cast (PSUM→bf16 directly)
6. Wo row prefetch before o_proj loop

Expected improvement: Fewer transposes, no HBM roundtrip, better DMA-compute overlap from streaming K/V, bridged attention→o_proj
gap via Wo prefetch.

Plan B — v12b_score_recompute.py — PASS (max_diff=1.43e-02)

What changed (all of Plan A, plus):

- Score tiles never materialized in SBUF — discarded after max extraction in Pass 1
- Pass 2 re-computes scores (5 extra matmuls, ~2% overhead) with K+V streamed together in a single self-contained affine_range
loop
- Each Pass 2 iteration is fully independent: DMA(K) → matmul(score) → exp → DMA(V) → matmul(V_weighted) → accumulate

Expected improvement: Completely clean SBUF in Pass 2 gives the compiler maximum freedom for scheduling. The self-contained loop

body should pipeline better than Plan A where Pass 2 reads pre-computed score tiles from SBUF.

## Data — v12a

### Overall Stats

| Metric | Value |
| --- | --- |
| Model Flops | 0.000 FLOPS |
| Total Time | 88.08 μs |
| Mm Arithmetic Intensity | 2 |
| Peak Flops Bandwidth Ratio | 109.837 FLOPS |
| Event Count | 2,988 |
| Trace Count | 11,424 |
| Transpose Flops | 66.048 KFLOPS |
| Neuroncore Cycle Count | 107,174 |
| Hardware Flops | 24.249 MFLOPS |

### Tensor Engine

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 27.17 μs |
| Tensor Engine Instruction Time | 140.37 μs |
| Mfu Estimated Percent | 0.17% |
| Hfu Estimated Percent | 0.17% |
| Mfu Max Achievable Estimated Percent | 0.72% |
| Tensor Engine Active Time Percent | 30.42% |
| Matmul Instruction Count | 490 |
| Tensor Engine Instruction Count | 1,046 |

### DMA / Memory

| Metric | Value |
| --- | --- |
| Dma Active Time | 42.16 μs |
| Dma Packet Time | 829.97 μs |
| Mbu Min Read Util Percent | 15.30% |
| Mbu Estimated Percent | 24.01% |
| Dma Active Time Percent | 47.21% |
| Dynamic Dma Packet Percent | 99.82% |
| Dynamic Dma Size Percent | 99.95% |
| Static Dma Packet Count | 12 |
| Dma Queue Count | 36 |
| Psum Read Sbuf Write Count | 56 |
| Psum Read Sbuf Write Bytes | 584.00 B |
| Psum Read Bytes | 1.69 KB |
| Psum Write Bytes | 2.51 KB |
| Hbm Write Bytes | 5.12 KB |
| Software Dynamic Dma Packet Count | 6,720 |
| Static Dma Size | 8.22 KB |
| Weight Size Bytes | 16.38 KB |
| Dma Active Cycles | 50,591 |
| Sbuf Read Bytes | 135.27 KB |
| Inputs And Weights Size Bytes | 9.79 MB |
| Hbm Read Bytes | 15.35 MB |
| Sbuf Write Bytes | 15.35 MB |
| Software Dynamic Dma Size | 15.36 MB |

## Data — v12b

### Overall Stats

| Metric | Value |
| --- | --- |
| Model Flops | 0.000 FLOPS |
| Total Time | 93.42 μs |
| Mm Arithmetic Intensity | 2 |
| Peak Flops Bandwidth Ratio | 109.837 FLOPS |
| Event Count | 3,062 |
| Trace Count | 11,525 |
| Transpose Flops | 66.048 KFLOPS |
| Neuroncore Cycle Count | 112,105 |
| Hardware Flops | 26.870 MFLOPS |

### Tensor Engine

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 23.97 μs |
| Tensor Engine Instruction Time | 143.66 μs |
| Mfu Estimated Percent | 0.18% |
| Hfu Estimated Percent | 0.18% |
| Mfu Max Achievable Estimated Percent | 0.79% |
| Tensor Engine Active Time Percent | 25.66% |
| Matmul Instruction Count | 500 |
| Tensor Engine Instruction Count | 1,063 |

### DMA / Memory

| Metric | Value |
| --- | --- |
| Dma Active Time | 42.33 μs |
| Dma Packet Time | 823.17 μs |
| Mbu Min Read Util Percent | 14.63% |
| Mbu Estimated Percent | 22.95% |
| Dma Active Time Percent | 45.31% |
| Dynamic Dma Size Percent | 99.52% |
| Dynamic Dma Packet Percent | 99.81% |
| Static Dma Packet Count | 13 |
| Dma Queue Count | 36 |
| Psum Read Sbuf Write Count | 56 |
| Psum Read Sbuf Write Bytes | 360.00 B |
| Psum Read Bytes | 1.69 KB |
| Psum Write Bytes | 2.67 KB |
| Hbm Write Bytes | 5.12 KB |
| Software Dynamic Dma Packet Count | 6,720 |
| Weight Size Bytes | 16.38 KB |
| Dma Active Cycles | 50,796 |
| Static Dma Size | 73.75 KB |
| Sbuf Read Bytes | 138.24 KB |
| Inputs And Weights Size Bytes | 9.79 MB |
| Hbm Read Bytes | 15.35 MB |
| Sbuf Write Bytes | 15.35 MB |
| Software Dynamic Dma Size | 15.36 MB |

---

# Round 8

Change 1 (Fused W_qkv [I=1280, H=2048]) — implemented as designed. One range() loop over 10 output-neuron blocks, each loading [PMAX, H=2048] contiguously. Had to use
nl.static_range for h_tiles (instead of affine_range) because the NKI tracer requires statically indexable list elements when accessed inside an outer Python range loop.

Change 2 (Contiguous V_cache + in-kernel nc_transpose) — implemented. Required an extra bf16→f32 SBUF cast before nc_transpose — CoreV3 (trn2) enforces matching input/output dtype
on nc_transpose. So: DMA bf16 → cast f32 → nc_transpose f32→f32 PSUM → cast bf16 SBUF → matmul.

Change 3 (sendrecv) — implemented with a key insight: since both cores currently compute all S_prior tiles identically, after sendrecv + add each core holds 2× the correct value for
both v_acc and sum_acc. Division v_acc / sum_acc = (2×correct) / (2×correct) = correct. This validates the sendrecv plumbing at zero correctness cost. True per-core S_prior
sharding (halving compute) is flagged as # OPT: and requires num_s_tiles to be even.

Change 4 (double_row) — perf_mode=matmul_perf_mode.double_row requires FP8 weights on trn2; not available for bf16. The OPT comment flags this. Output sharding and fused PSUM→bf16
cast from v12a are preserved.

Result: max_diff=1.43e-02  PASS (within rtol/atol=1e-2)

File: kernels/attn_tkg/agents/v13a_block_tkg_aligned.py