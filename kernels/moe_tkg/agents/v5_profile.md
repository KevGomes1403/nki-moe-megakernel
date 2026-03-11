# Round 1

### Overall Stats

| Metric | Value |
| --- | --- |
| Model FLOPs | 0.000 FLOPS |
| Total Time | 1.00 ms |
| MM Arithmetic Intensity | 1 |
| Peak FLOPs Bandwidth Ratio | 109.837 FLOPS |
| Event Count | 14,759 |
| Trace Count | 48,701 |
| Neuroncore Cycle Count | 1,203,203 |
| Hardware FLOPs | 25.166 MFLOPS |

### Data Movement — Neuron Core 0 (v5)

| Metric | Value |
| --- | --- |
| Spill Reload Bytes | 0.00 B |
| Spill Save Bytes | 0.00 B |
| Weight Queue Bytes | 0.00 B |
| Output Queue Bytes | 0.00 B |
| Hardware Dynamic DMA Packet Count | 0 |
| Hardware Dynamic DMA Size | 0.00 B |
| MBU Percent | 3.51% |
| Dynamic DMA Packet Percent | 98.39% |
| Dynamic DMA Size Percent | 99.78% |
| PSUM Read SBUF Write Count | 180 |
| Static DMA Packet Count | 410 |
| DMA Transfer Count | 809 |
| HBM Write Bytes | 4.10 KB |
| Input Queue Bytes | 5.76 KB |
| Software Dynamic DMA Packet Count | 25,088 |
| DMA Transfer Average Bytes | 31.12 KB |
| Static DMA Size | 55.96 KB |
| PSUM Read SBUF Write Bytes | 65.74 KB |
| PSUM Read Bytes | 90.34 KB |
| PSUM Write Bytes | 135.38 KB |
| SBUF Read Bytes | 306.55 KB |
| DMA Transfer Time | 321.55 µs |
| DMA Packet Time | 1.69 ms |
| HBM Read Bytes | 25.17 MB |
| DMA Transfer Total Bytes | 25.17 MB |
| Software Dynamic DMA Size | 25.22 MB |
| SBUF Write Bytes | 25.29 MB |

### Tensor Engine Stats

| Metric | Value |
| --- | --- |
| HFU Estimated Percent | 0.02% |
| MFU Estimated Percent | 0.02% |
| Tensor Engine Active Time | 167.59 µs |
| Tensor Engine Instruction Time | 239.67 µs |
| MFU Max Achievable Estimated Percent | 0.46% |
| Tensor Engine Active Time Percent | 16.71% |
| Matmul Instruction Count | 768 |
| Tensor Engine Instruction Count | 1,700 |

# Round 2


## Original Kernel (v5_token_parallel_tkg.py)

Token-parallel sparse MoE kernel processing one token at a time, with K experts per token via dynamic .ap() indexing. Profiled at T=1,
H=2048, I=256, E=128, K=8 in 1.00 ms. Primary bottleneck: DMA overhead — 25,088 software DMA packets for 809 transfers (31
packets/transfer), with DMA Packet Time (1.69 ms) being 5.3x the actual DMA Transfer Time (321 µs). Tensor Engine utilization only 16.71%.
Arithmetic intensity = 1 FLOP/byte (extremely memory-bound at T=1).

## Plan A — Coalesce Gate+Up Weight DMAs (v5a_coalesce_gateup_dma.py)

- What changed: Gate+up weight loads changed from [128, 128] per (h_t, gu_t) pair to [128, 512] per h_t. One DMA loads the full 2I-wide
weight row; nc_matmul slices the pre-loaded SBUF buffer.
- Why it helps: Reduces gate_up DMA count from 64 to 16 per expert (512 → 128 total). Fewer, larger DMA transfers reduce packet scheduling
overhead.
- Correctness: PASS (max_diff=9.77e-04 on Qwen3 shapes)
- No deviations from plan.

## Plan B — Widen Down Matmul Moving Dimension (v5b_widen_down_matmul.py)

- What changed: Down projection batches 4 h_tiles per nc_matmul using moving=[128, 512] instead of [128, 128]. PSUM, out_accum, and output
store all widened to [1, 4P]. Loop count drops from 16 to 4 h_groups.
- Why it helps: 4x fewer matmul instructions (256 → 64) AND 4x fewer down DMA loads (256 → 64). Each matmul instruction uses the full
512-wide Tensor Engine capacity. Output store also coalesced (16 → 4 DMAs).
- Correctness: PASS (max_diff=1.95e-03 on Qwen3 shapes)
- No deviations from plan. H_PER_GROUP=min(4, num_h_tiles) handles small H values.

## Plan C — Batch-Load Routing Weights (v5c_batch_routing_weights.py)

- What changed: Routing weights loaded in one [1, K] DMA per token instead of K individual 4-byte DMAs. SBUF slice at partition 0 used
directly in tensor_scalar, avoiding the non-zero partition constraint.
- Why it helps: Eliminates K-1=7 tiny DMA transfers per token and their associated packet overhead.

## Overall Stats

| Metric | v5c | v5b | v5a |
| --- | --- | --- | --- |
| Model Flops | 0.000 FLOPS | 0.000 FLOPS | 0.000 FLOPS |
| Total Time | 1.00 ms | 751.88 µs | 530.66 µs |
| Mm Arithmetic Intensity | 1 | 1 | 1 |
| Peak Flops Bandwidth Ratio | 109.837 FLOPS | 109.837 FLOPS | 109.837 FLOPS |
| Event Count | 14,764 | 10,933 | 8,512 |
| Trace Count | 48,576 | 37,106 | 29,521 |
| Neuroncore Cycle Count | 1,203,864 | 902,257 | 636,787 |
| Hardware Flops | 25.166 MFLOPS | 25.166 MFLOPS | 25.166 MFLOPS |

## Tensor Engine

| Metric | v5c | v5b | v5a |
| --- | --- | --- | --- |
| Hfu Estimated Percent | 0.02% | 0.02% | 0.03% |
| Mfu Estimated Percent | 0.02% | 0.02% | 0.03% |
| Tensor Engine Active Time | 168.78 µs | 137.14 µs | 103.36 µs |
| Tensor Engine Instruction Time | 240.57 µs | 194.45 µs | 233.67 µs |
| Mfu Max Achievable Estimated Percent | 0.46% | 0.46% | 0.46% |
| Tensor Engine Active Time Percent | 16.82% | 18.24% | 19.48% |
| Matmul Instruction Count | 768 | 576 | 768 |
| Tensor Engine Instruction Count | 1,704 | 1,257 | 1,687 |

## DMA / Memory System

| Metric | v5c | v5b | v5a |
| --- | --- | --- | --- |
| Dma Active Time | 312.26 µs | 274.00 µs | 212.81 µs |
| Dma Transfer Time | 325.31 µs | 289.29 µs | 223.09 µs |
| Dma Packet Time | 1.73 ms | 1.52 ms | 1.53 ms |
| Mbu Estimated Percent | 3.50% | 4.68% | 6.63% |
| Dma Active Time Percent | 31.13% | 36.44% | 40.10% |
| Mbu Min Read Util Percent | 56.06% | 74.80% | 105.98% |
| Dynamic Dma Packet Percent | 98.78% | 98.87% | 97.20% |
| Dynamic Dma Size Percent | 99.13% | 99.20% | 98.87% |
| Dma Queue Count | 51 | 51 | 51 |
| Psum Read Sbuf Write Count | 180 | 48 | 180 |
| Static Dma Packet Count | 310 | 229 | 427 |
| Dma Transfer Count | 802 | 605 | 425 |

## Memory Traffic

| Metric | v5c | v5b | v5a |
| --- | --- | --- | --- |
| Hbm Write Bytes | 4.10 KB | 4.10 KB | 4.10 KB |
| Input Queue Bytes | 4.16 KB | 4.16 KB | 4.16 KB |
| Weight Size Bytes | 16.38 KB | 16.38 KB | 16.38 KB |
| Software Dynamic Dma Packet Count | 25,088 | 19,968 | 14,848 |
| Dma Transfer Average Bytes | 31.39 KB | 41.61 KB | 59.23 KB |
| Psum Read Sbuf Write Bytes | 65.74 KB | 65.60 KB | 65.74 KB |
| Psum Read Bytes | 90.34 KB | 65.60 KB | 90.34 KB |
| Psum Write Bytes | 135.38 KB | 66.56 KB | 135.38 KB |
| Static Dma Size | 220.38 KB | 203.68 KB | 287.39 KB |
| Sbuf Read Bytes | 306.55 KB | 210.11 KB | 306.53 KB |

## Total Data Movement

| Metric | v5c | v5b | v5a |
| --- | --- | --- | --- |
| Dma Active Cycles | 374,711 | 328,795 | 255,376 |
| Hbm Read Bytes | 25.17 MB | 25.17 MB | 25.17 MB |
| Dma Transfer Total Bytes | 25.17 MB | 25.17 MB | 25.17 MB |
| Software Dynamic Dma Size | 25.22 MB | 25.21 MB | 25.20 MB |
| Sbuf Write Bytes | 25.29 MB | 25.24 MB | 25.29 MB |
| Inputs And Weights Size Bytes | 402.67 MB | 402.67 MB | 402.67 MB |
