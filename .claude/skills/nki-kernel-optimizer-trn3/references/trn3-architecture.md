# Trainium3 (NeuronCore-v4) Architecture Reference

---

## Device Generations

| Instance | NeuronCore | Notes |
|---|---|---|
| `trn1`, `trn1n`, `inf2` | NeuronCore-v2 | Baseline |
| `trn2` | NeuronCore-v3 | Previous generation |
| `trn3` | NeuronCore-v4 | **Current — target for this skill** |

---

## Device Overview (trn3 vs trn2)

| Resource | trn3 | trn2 |
|---|---|---|
| NeuronCores per chip | 8 (NeuronCore-v4) | 8 (NeuronCore-v3) |
| HBM capacity | 144 GiB | 96 GiB |
| HBM bandwidth | 4.7 TB/s | 3 TB/s |
| DMA engines per device | 128 | 128 |
| CC-Cores | 20 | 20 |
| NeuronLink | v4 | v3 |
| Process node | N3P | N5 |

---

## Memory per NeuronCore

| Memory | Capacity | Notes |
|---|---|---|
| SBUF | 32 MiB | 128 partitions × 256 KiB (↑ from 28 MiB on trn2) |
| PSUM | 2 MiB | Unchanged from prior generations |
| HBM | Shared device pool | Accessed via DMA |

**Partition dimension**: always **128** on trn3. Free dimension is all remaining dimensions.

---

## Compute Engines

### Tensor Engine (TensorE)

Peak throughput: **315 MXFP8/MXFP4 / 79 BF16/FP16/TF32 / 20 FP32 TFLOPS**.
Data-path: `8×128 MXFP8 input` (4x wider than trn2's 4×128 FP8 path), `2×128 BF16/FP16`, `5×128 sparse`, `1×128 output`.

**MXFP8/MXFP4 matmul** (NeuronCore-v4 exclusive)
- Use `nisa.nc_matmul_mx(dst, stationary, moving, stationary_scale, moving_scale)`.
- Performs dequantization + matmul in a single instruction — **4x throughput** vs BF16.
- Scaling group: 32 elements along contraction dimension share one uint8 scale.
- Stationary tile: `[128P, 128F]` (MXFP_x4 packs 4 elements per value → 512 logical elements).
- Moving tile: `[128P, 512F]` (MXFP_x4 packs 4 elements → 2048 logical elements max with BF16 output).
- See `mxfp-quantization.md` for full layout and API details.

**Background transpose** (NeuronCore-v4 exclusive)
- TensorE can run a transpose operation **in parallel** with another matmul.
- No extra cycles for transpose when pipelined with matmul.
- Exploit by scheduling transpose after issuing the matmul instruction.

**BF16 output with rounding control** (NeuronCore-v4)
- `nc_matmul_mx` can output BF16 directly to PSUM with RNE or stochastic rounding.
- Use stochastic rounding for training kernels to avoid systematic bias.

**Comparison with trn2 FP8 double-row mode**:
- trn2 used `perf_mode=nisa.matmul_perf_mode.double_row` for 2x FP8 throughput.
- **trn3 replaces this with native MXFP8/MXFP4 hardware** — do not use `double_row` on trn3.
- trn3 MXFP path gives 4x BF16 throughput vs trn2's 2x.

---

### Vector Engine (VectorE)

Peak: **1.2 TFLOPS FP32** (↑ from 1.0 TFLOPS on trn2). Frequency: 1.2 GHz (↑ from 0.96 GHz).
Data-path: `512 BF16/FP16/FP8 input/output`; `256 other types`.

**MXFP quantization** (NeuronCore-v4 exclusive)
- `nisa.quantize_mx(dst_data, src_bf16, dst_scale)` — converts BF16/FP16 → MXFP8 + uint8 scales.
- Runs on VectorE; TensorE runs at 2× clock frequency, so quantize can be a bottleneck for fully on-device paths.
- See `mxfp-quantization.md` for layout and usage.

**Fast exponential** (NeuronCore-v4 exclusive)
- `nisa.exponential` on trn3 runs at **4× throughput** vs `nisa.activation(..., op=nl.exp)`.
- Prefer `nisa.exponential` over `nisa.activation` for exp-heavy kernels (softmax, attention) on trn3.

**XORWOW PRNG** (NeuronCore-v4 exclusive)
- Higher quality than the LFSR algorithm used in prior generations.
- Use `nisa.rng` / `nisa.rand2` — same API, hardware automatically uses XORWOW on trn3.
- Supports state save/restore for reproducible runs.

**Performance mode** (inherited from trn2, still applies)
- 4× throughput for `tensor_copy` / `tensor_scalar` when: both tensors SBUF, both BF16/FP16, inner-most free dim contiguous.
- 2× throughput: same dtype but one tensor in PSUM, or non-contiguous inner-most dim.

---

### Scalar Engine (ScalarE)

Peak: **1.2 TFLOPS FP32** (↑ from prior, frequency 1.2 GHz).
Data-path: `256 BF16/FP16/FP8 input/output`; `128 other types` (doubled vs trn2).

**tensor_scalar and tensor_copy on ScalarE** (NeuronCore-v4 exclusive)
- ScalarE can now execute `nisa.tensor_scalar` and `nisa.tensor_copy` (previously VectorE-only).
- Enables better pipelining: VectorE and ScalarE can run different ops simultaneously.

**Activation2 instruction** (NeuronCore-v4 exclusive)
- Supports flexible bias operations and multiple reduction types.
- More composable than the trn2 `activation` instruction.

---

### GpSimd Engine (GpSimdE)

Frequency: 1.2 GHz. Data-path: `128 input/output all types` (same as trn2).
8 fully programmable processors for arbitrary ML operators.

---

## Data Movement

### DMA — New in NeuronCore-v4

**SBUF/PSUM indirect addressing (gather/scatter)**
- NeuronCore-v4 DMA natively supports indirect access into SBUF/PSUM.
- Enables on-device gather/scatter without GpSimdE software intervention.
- Use `nisa.dma_copy` with indirect addressing parameters.

**SBUF Read-Add-Write**
- DMA can read an SBUF tensor, add to it, and write back in a single pass.
- Enables on-the-fly tensor accumulation near memory — useful for expert accumulation in MoE.

**DMA Traffic Shaping**
- 4 service classes for bandwidth allocation.
- Priority-assign DMA transactions to avoid head-of-line blocking in latency-sensitive paths.

### DMA Transpose (inherited, still valid)

```python
# HBM → SBUF transpose: swap most-minor HBM dim into SBUF partition dim
sbuf_dst = nisa.dma_transpose(src=hbm_src)   # [512, 128] → [128, 512]
```

---

## Key Design Rules for trn3 Kernels

1. **Partition dim is 128** — tile shapes must respect this (same as trn2).
2. **Prefer MXFP8 for matmul-bound kernels** — `nc_matmul_mx` gives 4× BF16 throughput.
3. **Quantize offline when possible** — static weights should be quantized once at load time; don't re-quantize per inference.
4. **Pipeline quantize + matmul** — on-device quantization (`quantize_mx`) runs on VectorE; overlap with TensorE matmul on prior tile.
5. **Use fast exp** — `nisa.exponential` on trn3 is 4× faster; replace `nisa.activation(nl.exp)` in attention/softmax.
6. **Background transpose** — schedule matmul instruction first, then issue transpose — TensorE runs both in parallel.
7. **SBUF Read-Add-Write** — use for expert accumulation in MoE to reduce explicit load-add-store sequences.
8. **VectorE + ScalarE parallelism** — both can access SBUF simultaneously; pipeline post-matmul ops across both engines.
9. **Do not use** `double_row` perf mode — that is a trn2 feature; trn3 achieves the same via native MXFP hardware.
10. **Bucket sizes 128-aligned** for MoE kernels (same constraint as trn2).
