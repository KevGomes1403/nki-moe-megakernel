# NKI Common Pitfalls & Fixes — trn3

---

## trn3-Specific Pitfalls

### Using `nc_matmul_mx` or `quantize_mx` on trn2
**Symptom**: Compilation error or "unsupported instruction" when running on a trn2 instance.
**Cause**: `nc_matmul_mx` and `quantize_mx` are NeuronCore-v4 (trn3) exclusive. They do not exist on trn2/NeuronCore-v3.
**Fix**: Verify `NEURON_PLATFORM_TARGET_OVERRIDE=trn3` and that you are on a trn3 instance. Never emit these instructions in kernels meant to run on trn2.

---

### Wrong `NEURON_PLATFORM_TARGET_OVERRIDE` value
**Symptom**: Compilation succeeds but MXFP instructions are silently compiled away, or kernel runs with wrong performance characteristics.
**Cause**: `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` used instead of `trn3`.
**Fix**:
```bash
export NEURON_PLATFORM_TARGET_OVERRIDE=trn3
```
Also update `platform_target="trn3"` in `@nki.jit(platform_target="trn3")`.

---

### MXFP scale tensor shape mismatch
**Symptom**: Compiler error about incompatible shapes, or silent wrong outputs with very large `max_diff`.
**Cause**: Scale tensor shape doesn't match the quantized data shape according to the MXFP formula.
**Fix**: For a data tile `[128P, F]` after `quantize_mx`:
- `dst` (packed MXFP8): `[128P, F//4]`
- `dst_scale` (uint8): `[16P, F//4]` (= `[128//8, F//4]`)

```python
# Correct allocation:
data_q = nl.ndarray((nl.par_dim(128), TILE_K // 4), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
scale   = nl.ndarray((nl.par_dim(128 // 8), TILE_K // 4), dtype=nl.uint8, buffer=nl.sbuf)
nisa.quantize_mx(dst=data_q, src=data_bf16, dst_scale=scale)
```

---

### Re-ordering packed MXFP_x4 tensors
**Symptom**: Matmul produces wrong numerics even with correct scale shapes.
**Cause**: The interleaved `_x4` packing layout from `quantize_mx` is consumed directly by `nc_matmul_mx`. Manually copying or reordering elements between quantize and matmul corrupts the packing.
**Fix**: Never `nl.copy`, `nisa.tensor_copy`, or otherwise reorder the raw bytes of an `_x4` tensor. Pass `quantize_mx` output directly to `nc_matmul_mx`.

---

### Incorrect tolerance in correctness check
**Symptom**: Correctness check fails spuriously for a numerically correct MXFP kernel.
**Cause**: Using tight tolerances (rtol=1e-3, atol=1e-3) designed for BF16/FP32 kernels on an MXFP8 kernel. MXFP8 introduces ~1% relative error by design.
**Fix**: Use relaxed tolerances for quantized kernels:
```python
np.testing.assert_allclose(result, ref, rtol=5e-2, atol=5e-2)  # MXFP8
np.testing.assert_allclose(result, ref, rtol=1e-1, atol=1e-1)  # MXFP4
```
Compare against a BF16 reference, not FP32.

---

### Quantizing on HBM (outside SBUF)
**Symptom**: Compiler error about memory location for `quantize_mx`.
**Cause**: `quantize_mx` requires all tensors (src, dst, dst_scale) to be in **SBUF**.
**Fix**: Always load the BF16 source into SBUF first, then quantize, then use the quantized SBUF tile.

```python
# Wrong — src is HBM tensor
nisa.quantize_mx(dst=q, src=weight_hbm, dst_scale=scale)  # ERROR

# Correct
w_tile = nl.load(weight_hbm[...])                          # load into SBUF
nisa.quantize_mx(dst=q, src=w_tile, dst_scale=scale)       # all SBUF
```

---

### Using trn2 FP8 double-row mode on trn3
**Symptom**: No compiler error (code compiles) but you don't get the expected performance.
**Cause**: `perf_mode=nisa.matmul_perf_mode.double_row` is a trn2-only feature. On trn3, this mode either has no effect or behaves differently. The correct path for high throughput on trn3 is `nc_matmul_mx`.
**Fix**: Remove `double_row` perf mode and migrate to `nc_matmul_mx` for 4× throughput.

---

### Partition dim not a multiple of 32 for MXFP ops
**Symptom**: `quantize_mx` or `nc_matmul_mx` compiler error about partition dimension.
**Cause**: Both APIs require partition dim to be a multiple of 32 (due to the 8P × 4F scaling group).
**Fix**: Always use `nl.par_dim(128)` for full-width tiles. If using sub-128 partition tiles, ensure the size is 32, 64, or 96.

---

## Inherited Pitfalls (from trn2, still apply)

### NCC_IBIR158 — illegal SBUF/PSUM access pattern
**Error**: `NCC_IBIR158: ...`
**Cause**: SBUF/PSUM `.ap()` first tuple step must equal total free-dimension element count.
**Fix**: Ensure step matches free-dim, or use direct slice indexing.

### DGE `scalar_offset` absolute SBUF address bug
**Symptom**: Correct in standalone, wrong in E2E model.
**Fix**: Copy element to scratch at `offset=0`; always reference `offset=0` (see nki-syntax-quickref.md).

### "TensorScalarPtr arith immediate dtype must be fp32"
**Cause**: BF16 affinities in POST_SCALE mode.
**Fix**: `expert_affinities = expert_affinities.to(torch.float32)`.

### Compiler error after adding `affine_range`
**Cause**: `affine_range` on a loop with a carried dependency.
**Fix**: Use `sequential_range` for accumulation loops.

### Output tensor stays zero
**Fix**:
```python
# Always: PSUM → SBUF → HBM
nl.copy(sbuf_tmp, psum_result)
nl.store(output_hbm[...], sbuf_tmp)
```

### NaN/Inf outputs
- BF16 overflow: add scaling before accumulation.
- MXFP saturation: inputs outside `[-448, 448]` (FP8 E4M3) will saturate; clip or normalize.
- Uninitialized PSUM: always use `nl.zeros(...)`.

---

## Environment & Runtime

### "NeuronCore busy" / process conflicts
**Cause**: Another Python process holds the NeuronCores.
**Fix**: Kill all other Python processes using Neuron devices. NeuronCores are exclusive.

### `nki.benchmark` not working
**Cause**: `nki.benchmark` does not work in the `pytorch_2_9_nxd_inference` environment.
**Fix**: Use the `wrap_benchmark` harness from `scripts/benchmark.py` (see benchmarking-api.md).

### Env vars set after imports
**Symptom**: `last_result` is None, no NTFF written.
**Fix**: Set all `NEURON_RT_INSPECT_*` vars **before** any `import torch` or `import nki`.

---

## Debugging Protocol

When a kernel fails to compile:
1. Strip back to bare minimum (no fusion, no `affine_range`, no MXFP quantization).
2. Add MXFP quantization first (just `quantize_mx` + `nc_matmul_mx` on small shapes).
3. Add fusions and tiling one at a time.
4. Check scale tensor shapes at every step.
5. Use `print()` for shapes — prints run at compile time.

When a quantized kernel gives wrong results:
1. First verify the BF16 baseline is correct.
2. Run on tiny shapes (M=128, K=128, N=128) — minimum MXFP tile size.
3. Check scale tensor shapes exactly: `[P//8, F//4]`.
4. Ensure no reordering of `_x4` packed tensors.
5. Use relaxed tolerances — if `max_diff < 0.1`, the kernel is likely correct.
6. Compare scale tensor values directly: quantize a known input manually and verify scales.
