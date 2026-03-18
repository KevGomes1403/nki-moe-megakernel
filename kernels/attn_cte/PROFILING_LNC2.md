# Round 7

## Data (v7ab — LNC=2)

### Overall Stats

| Metric | Value |
| --- | --- |
| Model Flops | 0.000 FLOPS |
| Total Time | 1.48 ms |
| Peak Flops Bandwidth Ratio | 109.837 FLOPS |
| Mm Arithmetic Intensity | 514 |
| Event Count | 18,576 |
| Neuroncore Cycle Count | 1,791,321 |
| Trace Count | 2,744,307 |
| Transpose Flops | 3.859 GFLOPS |
| Hardware Flops | 18.187 GFLOPS |

### Vector Engine

| Metric | Value |
| --- | --- |
| Vector Engine Active Time | 772.74 µs |
| Vector Engine Instruction Time | 1.27 ms |
| Vector Engine Active Time Percent | 51.77% |
| Vector Engine Instruction Count | 5,540 |

### Tensor Engine

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 528.94 µs |
| Tensor Engine Instruction Time | 1.77 ms |
| MFU Estimated Percent | 6.10% |
| HFU Estimated Percent | 7.75% |
| Tensor Engine Active Time Percent | 35.43% |
| MFU Max Achievable Estimated Percent | 100.00% |
| Matmul Instruction Count | 4,096 |
| Tensor Engine Instruction Count | 8,352 |

### DMA / Memory Activity

| Metric | Value |
| --- | --- |
| Dma Active Time | 1.24 ms |
| Dma Transfer Time | 2.19 ms |
| Mbu Min Read Util Percent | 1.06% |
| Dma Packet Time | 23.19 ms |
| Mbu Estimated Percent | 2.61% |
| Dma Active Time Percent | 82.93% |
| Dynamic Dma Size Percent | 98.44% |
| Dynamic Dma Packet Percent | 100.00% |
| Static Dma Packet Count | 47 |
| Dma Queue Count | 68 |
| Dma Transfer Count | 1,017 |
| Psum Read Sbuf Write Count | 1,296 |

### Memory Sizes & Transfers

| Metric | Value |
| --- | --- |
| Weight Size Bytes | 16.38 KB |
| Software Dynamic Dma Packet Count | 27,328 |
| Dma Transfer Average Bytes | 37.89 KB |
| Psum Read Bytes | 550.91 KB |
| Psum Read Sbuf Write Bytes | 571.39 KB |
| Static Dma Size | 612.38 KB |
| Psum Write Bytes | 1.11 MB |
| Dma Active Cycles | 1,485,497 |
| Hardware Dynamic Dma Packet Count | 2,680,518 |

### HBM / SBUF Traffic

| Metric | Value |
| --- | --- |
| Hbm Write Bytes | 5.21 MB |
| Hardware Dynamic Dma Size | 10.72 MB |
| Inputs And Weights Size Bytes | 11.35 MB |
| Sbuf Read Bytes | 19.47 MB |
| Hbm Read Bytes | 22.68 MB |
| Software Dynamic Dma Size | 27.91 MB |
| Sbuf Write Bytes | 34.77 MB |
| Dma Transfer Total Bytes | 38.54 MB |

# Round 8

## Optimization Summary

Original Kernel (v7ab at LNC=2)

- 1.48ms total, DMA-bound (82.93% active)
- TE/VE severely underutilized (35%/52%) — starved waiting on DMA
- Two .ap() broadcasts (lines 692, 801) create idle gaps on compute engines

Plan A — LNC=2 KVH Work Splitting (v8a)

- Splits kvh loop across 2 physical NeuronCores via program_id(0)
- Each core processes Hkv_tp/2 KV heads with its own DMA/TE/VE engines
- Wq hoisting partitioned per-core; hidden/cos/sin shared
- Correctness: PASS (max_diff=2.44e-04)

Plan B — Causal Compute Skipping (v8b)

- Uses concrete qsi (Python range()) to set variable inner loop bounds
- effective_batch = min(BATCH, qsi+1) — only computes valid K tiles
- Score matmuls reduced from 10 to 6 for S=640 (40% fewer)
- Correctness: PASS (max_diff=2.44e-04)

Plan C — Three-Sub-Pass Softmax (v8c)

- Splits Softmax+V into: (2a) row-max + broadcast, (2b) exp + V-accum, (2c) normalize + store
- Each sub-pass uses affine_range for DMA pipelining across qsi
- Removes DMA broadcasts from the compute-critical path
- Correctness: PASS (max_diff=2.44e-04)

---

## Data (v8a)

### Overall Stats

| Metric | Value |
| --- | --- |
| Model Flops | 0.000 FLOPS |
| Total Time | 783.02 µs |
| Peak Flops Bandwidth Ratio | 109.837 FLOPS |
| Mm Arithmetic Intensity | 424 |
| Event Count | 10,037 |
| Neuroncore Cycle Count | 943,228 |
| Trace Count | 1,413,813 |
| Transpose Flops | 2.265 GFLOPS |
| Hardware Flops | 9.429 GFLOPS |

### Vector Engine

| Metric | Value |
| --- | --- |
| Vector Engine Active Time | 401.87 µs |
| Vector Engine Instruction Time | 656.65 µs |
| Vector Engine Active Time Percent | 51.13% |
| Vector Engine Instruction Count | 2,883 |

### Tensor Engine

| Metric | Value |
| --- | --- |
| Tensor Engine Active Time | 279.66 µs |
| Tensor Engine Instruction Time | 893.72 µs |
| MFU Estimated Percent | 5.79% |
| HFU Estimated Percent | 7.63% |
| Tensor Engine Active Time Percent | 35.58% |
| MFU Max Achievable Estimated Percent | 100.00% |
| Matmul Instruction Count | 2,128 |
| Tensor Engine Instruction Count | 4,413 |

### DMA / Memory Activity

| Metric | Value |
| --- | --- |
| Dma Active Time | 627.28 µs |
| Dma Transfer Time | 1.19 ms |
| Dma Packet Time | 12.50 ms |
| Mbu Min Read Util Percent | 2.02% |
| Mbu Estimated Percent | 3.00% |
| Dma Active Time Percent | 79.80% |
| Dynamic Dma Size Percent | 98.03% |
| Dynamic Dma Packet Percent | 100.00% |
| Static Dma Packet Count | 38 |
| Dma Queue Count | 68 |
| Dma Transfer Count | 603 |
| Psum Read Sbuf Write Count | 728 |

### Memory Sizes & Transfers

| Metric | Value |
| --- | --- |
| Weight Size Bytes | 16.38 KB |
| Software Dynamic Dma Packet Count | 16,608 |
| Dma Transfer Average Bytes | 37.12 KB |
| Psum Read Bytes | 295.94 KB |
| Psum Read Sbuf Write Bytes | 306.18 KB |
| Static Dma Size | 449.56 KB |
| Psum Write Bytes | 575.49 KB |
| Dma Active Cycles | 752,732 |
| Hardware Dynamic Dma Packet Count | 1,377,600 |

### HBM / SBUF Traffic

| Metric | Value |
| --- | --- |
| Hbm Write Bytes | 2.59 MB |
| Hardware Dynamic Dma Size | 5.51 MB |
| Sbuf Read Bytes | 9.89 MB |
| Inputs And Weights Size Bytes | 11.35 MB |
| Hbm Read Bytes | 14.29 MB |
| Software Dynamic Dma Size | 16.91 MB |
| Sbuf Write Bytes | 20.49 MB |
| Dma Transfer Total Bytes | 22.38 MB |