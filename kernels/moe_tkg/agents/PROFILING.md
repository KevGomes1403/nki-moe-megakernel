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

# Round 3

### Original Kernel (v5a — 530.66 µs)

Token-parallel sparse MoE with coalesced gate+up weight DMAs. Bottleneck: DMA packet scheduling overhead (1.53 ms packet time
vs 223 µs transfer time). 425 DMA transfers with 14,848 software packets. Tensor Engine only 19.48% active. Down projection
alone accounts for 256/425 = 60% of all DMAs.

### Plan A — Widen Down Matmul Moving Dimension (v6a_widen_down_on_v5a.py)

- What changed: Down projection batches 4 h_tiles per nc_matmul using moving=[P, 4P]=[128, 512]. Weight DMA loads [P, 512] blocks. PSUM, accumulators, and output store all widened to [1, 4P].
- Why it helps: Down DMAs reduced from 256 → 64 (4×). Down matmuls reduced from 256 → 64 (4×). Total DMAs: ~233 (45% reduction from 425). Each matmul uses the full 512-wide Tensor Engine capacity.
- Correctness: PASS (max_diff=9.77e-04)
- No deviations from plan.

### Plan B — Coalesce Down Weight DMA (v6b_coalesce_down_dma.py)

- What changed: Down weight rows loaded as full [P, H=2048] per i_tile in one DMA, then sliced [P, P] for each h_t's matmul.
Both i_tile rows pre-loaded in affine_range before the matmul loop.
- Why it helps: Down DMAs reduced from 256 → 16 (16×). Total DMAs: ~185 (56% reduction from 425). DMA transfer average size increases dramatically.
- Correctness: PASS (max_diff=9.77e-04)
- Deviation: Originally planned i_t-outer/h_t-inner loop flip, but compiler error (NCC_IGCA057) forced keeping h_t-outer structure with pre-loaded rows. Same DMA reduction achieved.

### Plan C — Transpose Gate+Up Matmul Layout (v6c_transpose_gateup.py)

- What changed: Swapped stationary/moving in gate+up nc_matmul: stationary=h_tile[128,1], moving=w_row[128,512] → result
[1,512]. One matmul per h_tile covers all gate+up outputs. Post-processing uses [1,I] tensors, with tiled nc_transpose
(4×[1,32]→[32,1]) to reshape activations back to [P,1] for unchanged down projection.
- Why it helps: Gate+up matmuls reduced from 512 → 128 (4×). Net ~368 fewer TE instructions. PSUM allocations reduced from
4×[P,1] to 1×[1,512].

## Profiling

### Overall Stats Comparison

| Metric | v6a | v6b | v6c |
| --- | --- | --- | --- |
| Model Flops | 0.000 FLOPS | 0.000 FLOPS | 0.000 FLOPS |
| Total Time | 278.65 µs | 274.76 µs | 547.00 µs |
| MM Arithmetic Intensity | 1 | 1 | 1 |
| Peak Flops Bandwidth Ratio | 109.837 FLOPS | 109.837 FLOPS | 109.837 FLOPS |
| Event Count | 4,792 | 4,898 | 8,402 |
| Trace Count | 17,902 | 18,038 | 28,543 |
| Neuroncore Cycle Count | 334,378 | 329,707 | 656,405 |
| Hardware Flops | 25.166 MFLOPS | 25.166 MFLOPS | 25.166 MFLOPS |

---

### Tensor Engine Comparison

| Metric | v6a | v6b | v6c |
| --- | --- | --- | --- |
| Tensor Engine Active Time | 76.59 µs | 69.64 µs | 146.45 µs |
| Tensor Engine Instruction Time | 188.92 µs | 234.37 µs | 177.06 µs |
| HFU Estimated Percent | 0.06% | 0.06% | 0.03% |
| MFU Estimated Percent | 0.06% | 0.06% | 0.03% |
| MFU Max Achievable Estimated Percent | 0.46% | 0.46% | 0.46% |
| Tensor Engine Active Time Percent | 27.49% | 25.35% | 26.77% |
| Matmul Instruction Count | 576 | 768 | 384 |
| Tensor Engine Instruction Count | 1,252 | 1,535 | 910 |

---

### DMA / Memory Comparison

| Metric | v6a | v6b | v6c |
| --- | --- | --- | --- |
| DMA Active Time | 172.54 µs | 137.83 µs | 212.01 µs |
| DMA Transfer Time | 181.09 µs | 161.76 µs | 223.87 µs |
| DMA Packet Time | 1.35 ms | 1.35 ms | 1.55 ms |
| MBU Estimated Percent | 12.62% | 12.80% | 6.43% |
| DMA Active Time Percent | 61.92% | 50.17% | 38.76% |
| Dynamic DMA Packet Percent | 97.70% | 95.50% | 97.24% |
| Dynamic DMA Size Percent | 99.20% | 99.13% | 99.13% |
| MBU Min Read Util Percent | 201.83% | 204.69% | 102.81% |
| DMA Queue Count | 51 | 51 | 51 |
| DMA Transfer Count | 221 | 185 | 425 |
| Static DMA Packet Count | 229 | 422 | 422 |
| PSUM Read SBUF Write Count | 48 | 180 | 197 |
| HBM Write Bytes | 4.10 KB | 4.10 KB | 4.10 KB |
| Input Queue Bytes | 4.16 KB | 4.16 KB | 4.16 KB |
| Software Dynamic DMA Packet Count | 9,728 | 8,960 | 14,848 |
| Weight Size Bytes | 16.38 KB | 16.38 KB | 16.38 KB |
| PSUM Read Bytes | 65.60 KB | 65.76 KB | 108.54 KB |
| PSUM Read SBUF Write Bytes | 65.60 KB | 65.74 KB | 108.54 KB |
| PSUM Write Bytes | 66.56 KB | 111.82 KB | 260.10 KB |
| DMA Transfer Average Bytes | 113.91 KB | 136.08 KB | 59.23 KB |
| Static DMA Size | 203.68 KB | 220.83 KB | 220.83 KB |
| DMA Active Cycles | 207,043 | 165,399 | 254,412 |
| SBUF Read Bytes | 210.11 KB | 307.90 KB | 334.55 KB |
| HBM Read Bytes | 25.17 MB | 25.17 MB | 25.17 MB |
| DMA Transfer Total Bytes | 25.17 MB | 25.17 MB | 25.17 MB |
| Software Dynamic DMA Size | 25.18 MB | 25.18 MB | 25.20 MB |
| SBUF Write Bytes | 25.24 MB | 25.29 MB | 25.34 MB |
| Inputs and Weights Size Bytes | 402.67 MB | 402.67 MB | 402.67 MB |

# Round 4

## Optimization Summary

### Original Kernels (v6a/v6b — ~275 µs)

Both are DMA-bound token-gen MoE kernels for Qwen3 (T=1, H=2048, I=256, E=128, K=8). v6a uses widened down matmul (576 matmuls, 221 DMAs). v6b uses coalesced down DMA (768 matmuls, 185 DMAs). Both read 25.17 MB from HBM with ~13% MBU and ~26% TE utilization.

### Plan A — Fuse Coalesced Down DMA + Widened Down Matmul → v7a

- File: kernels/moe_tkg/agents/v7a_fused_coalesce_widen_down.py
- What changed: Keeps v6b's pre-loaded [P, H] down weight rows, but slices [P, 4P]=[128, 512] for widened matmuls
instead of [P, P]. PSUM, accumulators, and output store all widened to [1, 4P]. Loop over 4 h_groups instead of 16
h_tiles.
- Expected effect: Down matmuls 256→64 (−192), output store DMAs 16→4 (−12). Best of both v6a and v6b — fewest DMAs
AND fewest matmuls.
- Correctness: PASS (max_diff=9.77e-04)

### Plan B — Pre-load All Gate+Up Weight Rows → v7b

- File: kernels/moe_tkg/agents/v7b_preload_gateup_rows.py
- What changed: Gate+up weight rows pre-loaded in affine_range before the sequential matmul loop, instead of loading one row at a time inside it. All 16 DMAs per expert become independent.
- Expected effect: Better DMA-compute overlap — compiler can batch-schedule all 16 gate+up DMAs. TE stall gaps between matmuls should shrink.
- Correctness: PASS (max_diff=9.77e-04)

### Plan C — Coalesce Hidden State Load → v7c

- File: kernels/moe_tkg/agents/v7c_coalesce_hidden_dma.py
- What changed: 16 separate [P, 1] hidden state DMAs replaced by single [P, 16]=[128, 16] DMA (4KB). Matmuls slice from the coalesced buffer.
- Expected effect: 15 fewer DMA transfers per token, ~660 fewer software DMA packets. Marginal improvement.
- Correctness: PASS (max_diff=9.77e-04)

## Data

### Overall Stats

| Metric | v7a | v7b | v7c |
| --- | --- | --- | --- |
| Model FLOPs | 0.000 FLOPS | 0.000 FLOPS | 0.000 FLOPS |
| Total Time | 220.81 µs | 274.66 µs | 261.64 µs |
| MM Arithmetic Intensity | 1 | 1 | 1 |
| Peak FLOPs Bandwidth Ratio | 109.837 FLOPS | 109.837 FLOPS | 109.837 FLOPS |
| Event Count | 3,993 | 4,899 | 4,735 |
| Trace Count | 15,893 | 18,039 | 17,337 |
| Neuroncore Cycle Count | 264,975 | 329,594 | 313,963 |
| Hardware FLOPs | 25.166 MFLOPS | 25.166 MFLOPS | 25.166 MFLOPS |

### Tensor Engine

| Metric | v7a | v7b | v7c |
| --- | --- | --- | --- |
| Tensor Engine Active Time | 65.36 µs | 69.93 µs | 69.68 µs |
| Tensor Engine Instruction Time | 188.40 µs | 234.53 µs | 234.80 µs |
| MFU Estimated Percent | 0.07% | 0.06% | 0.06% |
| HFU Estimated Percent | 0.07% | 0.06% | 0.06% |
| MFU Max Achievable Estimated Percent | 0.46% | 0.46% | 0.46% |
| Tensor Engine Active Time Percent | 29.60% | 25.46% | 26.63% |
| Matmul Instruction Count | 576 | 768 | 768 |
| Tensor Engine Instruction Count | 1,197 | 1,535 | 1,512 |

### DMA / Memory / Buffer Statistics

| Metric | v7a | v7b | v7c |
| --- | --- | --- | --- |
| DMA Active Time | 133.16 µs | 139.22 µs | 134.29 µs |
| DMA Transfer Time | 157.52 µs | 164.66 µs | 148.18 µs |
| DMA Packet Time | 1.33 ms | 1.33 ms | 1.34 ms |
| MBU Estimated Percent | 15.92% | 12.80% | 13.44% |
| DMA Active Time Percent | 60.30% | 50.69% | 51.33% |
| Dynamic DMA Packet Percent | 97.51% | 95.50% | 95.27% |
| Dynamic DMA Size Percent | 99.20% | 99.13% | 99.13% |
| MBU Min Read Util Percent | 254.69% | 204.76% | 214.95% |
| DMA Queue Count | 51 | 51 | 51 |
| DMA Transfer Count | 173 | 185 | 170 |
| PSUM Read SBUF Write Count | 48 | 180 | 180 |
| Static DMA Packet Count | 229 | 422 | 422 |
| HBM Write Bytes | 4.10 KB | 4.10 KB | 4.10 KB |
| Input Queue Bytes | 4.16 KB | 4.16 KB | 4.16 KB |
| Software Dynamic DMA Packet Count | 8,960 | 8,960 | 8,496 |
| Weight Size Bytes | 16.38 KB | 16.38 KB | 16.38 KB |
| PSUM Read Bytes | 65.60 KB | 65.76 KB | 65.76 KB |
| PSUM Read SBUF Write Bytes | 65.60 KB | 65.74 KB | 65.74 KB |
| PSUM Write Bytes | 66.56 KB | 111.82 KB | 111.82 KB |
| DMA Transfer Average Bytes | 145.51 KB | 136.08 KB | 148.08 KB |
| DMA Active Cycles | 159,788 | 167,058 | 161,152 |
| Static DMA Size | 203.68 KB | 220.83 KB | 220.83 KB |
| SBUF Read Bytes | 210.02 KB | 307.90 KB | 307.88 KB |
| HBM Read Bytes | 25.17 MB | 25.17 MB | 25.17 MB |
| DMA Transfer Total Bytes | 25.17 MB | 25.17 MB | 25.17 MB |
| Software Dynamic DMA Size | 25.18 MB | 25.18 MB | 25.18 MB |
| SBUF Write Bytes | 25.24 MB | 25.29 MB | 25.29 MB |
| Inputs and Weights Size Bytes | 402.67 MB | 402.67 MB | 402.67 MB |