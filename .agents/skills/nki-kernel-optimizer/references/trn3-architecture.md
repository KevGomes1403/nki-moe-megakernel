# Trainium3 Architecture

Trainium3 is the fourth-generation purpose-built ML chip from AWS. A Trainium3 device contains **eight NeuronCore-v4 cores**. Like Trainium2, it supports Logical NeuronCore Configuration (LNC), which lets you combine the compute and memory resources of multiple physical NeuronCores into a single logical NeuronCore.

---

## Chip-Level Specs

### Compute (per chip)

| Precision | TFLOPS |
|-----------|--------|
| MXFP8 / MXFP4 | 2,517 |
| BF16 / FP16 / TF32 | 671 |
| FP16/BF16/TF32 sparse | 2,517 |
| FP32 | 183 |

### Memory

| | Value |
|-|-------|
| HBM capacity | 144 GiB |
| HBM bandwidth | 4.9 TB/sec |
| SBUF capacity (per NeuronCore-v4) | 32 MiB on-chip SRAM |

### Data Movement

| | Value |
|-|-------|
| DMA bandwidth | 4.9 TB/sec |
| NeuronLink-v4 (inter-device) | 2.56 TB/sec per device |

### Collective Communication

16 CC-Cores orchestrate collective communication among Trainium3 devices, both within a server and across servers.

---

## Trainium2 vs Trainium3 Comparison

| Feature | Trainium2 | Trainium3 | Factor |
|---------|-----------|-----------|--------|
| MXFP4 (TFLOPS) | — | 2,517 | — |
| FP8 (TFLOPS) | 1,299 | 2,517 | 2x |
| BF16/FP16/TF32 (TFLOPS) | 667 | 671 | 1x |
| FP32 (TFLOPS) | 181 | 183 | 1x |
| HBM capacity (GiB) | 96 | 144 | 1.5x |
| HBM bandwidth (TB/sec) | 2.9 | 4.9 | 1.7x |
| SBUF capacity (MiB) | 224 | 256 | 1.14x |
| Inter-chip interconnect (GB/sec/chip) | 1,280 | 2,560 | 2x |
| DMA bandwidth (TB/sec) | 3.5 | 4.9 | 1.4x |

---

## NeuronCore-v4 Architecture

NeuronCore-v4 is the fourth-generation NeuronCore. It is a fully-independent heterogeneous compute unit with four main engines — **Tensor, Vector, Scalar, and GPSIMD** — plus software-managed on-chip SRAM.

Supports: control flow, dynamic shapes, programmable rounding mode (RNE & Stochastic Rounding).

### On-Chip SRAM

- **32 MiB per NeuronCore-v4**
- Software-managed for data locality and prefetch
- New near-memory accumulation: DMA engines can perform read-add-write into existing SRAM in a single transfer

### Tensor Engine

- Systolic array; optimized for GEMM, CONV, Transpose
- Supported input dtypes: MXFP8, MXFP4, FP16, BF16, TF32, FP32
- Output dtype: FP32 or BF16
- Per-core throughput: **315 MXFP8/MXFP4 TFLOPS**, 79 BF16/FP16/TF32 TFLOPS, 20 FP32 TFLOPS
- MXFP4 is converted to MXFP8 before compute; programmer-defined mapping
- **Structured sparsity**: up to 315 TFLOPS for FP16/BF16/TF32 with M:N patterns (4:16, 4:12, 4:8, 2:8, 2:4, 1:4, 1:2)

### Vector Engine

- Optimized for operations where each output element depends on multiple inputs (axpy, LayerNorm, Pooling)
- **1.2 TFLOPS FP32** per core
- Supported dtypes: FP8, FP16, BF16, TF32, FP32, INT8, INT16, INT32
- New in v4:
  - Online quantization from BF16/FP16 → MXFP8 (useful between MLP layers)
  - Fast exponential at 4× the throughput of Scalar Engine (useful for self-attention softmax)

### Scalar Engine

- Optimized for element-wise ops (each output depends on one input element)
- **1.2 TFLOPS FP32** per core
- Supported dtypes: FP8, FP16, BF16, TF32, FP32, INT8, INT16, INT32

### GPSIMD Engine

- Eight fully-programmable **512-bit wide vector processors** per NeuronCore
- Execute general-purpose C/C++ code
- Direct access to on-chip SRAM
- Used for custom operators running directly on the NeuronCores

---

## NKI Kernel Implications

- **Partition dimension**: NeuronCore-v4 has 128 partition lanes (same as v3; `nl.par_dim(128)` still applies).
- **SBUF is larger** (32 MiB vs ~28 MiB on trn2) — more room for SBUF-resident activations and hoisting.
- **MXFP8/MXFP4**: double the FP8 TFLOPS vs trn2 — quantized kernels see a 2× compute ceiling increase.
- **Near-memory accumulation**: DMA read-add-write into SBUF in one transfer — useful for fused residual adds without a separate load.
- **NeuronLink-v4**: 2× inter-device bandwidth enables more aggressive tensor-parallel patterns.
- **LNC=2**: combining two NeuronCore-v4s gives 64 MiB SBUF and doubles TFLOPS — required for `shared_hbm` kernels.
