# NKI Common Pitfalls & Fixes

---

## Compiler Errors

### NCC_IBIR158 — illegal SBUF/PSUM access pattern
**Error**: `NCC_IBIR158: ...`
**Cause**: SBUF/PSUM `.ap()` call violates the contiguous-partition constraint: the first tuple's step must equal the total free-dimension element count. Reading every-other partition or any non-contiguous partition stride triggers this error.
**Fix**: Ensure the first tuple step matches the free-dim size, or use direct slice indexing instead:
```python
# ILLEGAL — step=64 skips partitions on a (128, 32) tensor (free_dim=32, step must be 32)
t.ap(pattern=[[64, 64], [1, 32]], offset=0)

# LEGAL — step=32 matches free_dim
t.ap(pattern=[[32, 128], [1, 32]], offset=0)

# ALTERNATIVE — for simple column access, direct slice avoids .ap() entirely
col = sbuf_tensor[0:T, expert_id:expert_id+1]
```

### NCC_IBIR030 — `scalar_offset` without `.ap()` descriptor
**Error**: `NCC_IBIR030: ...`
**Cause**: Passing an SBUF tensor directly (or a 1×1 slice) as `scalar_offset` to a DGE instruction without an `.ap()` descriptor. The compiler requires `.ap()` to derive `IndirectDimMaxIndex`.
**Fix**: Wrap the SBUF tensor in `.ap()` with a shape that encodes the max expert index:
```python
# WRONG — compiler can't derive IndirectDimMaxIndex
nisa.dge_gather(..., scalar_offset=eid_scratch[0:1, 0:1])

# RIGHT — (128,1) shape → IndirectDimMaxIndex=127=E-1
eid_offset = eid_scratch.ap(pattern=[[1,1],[1,1]], offset=0)
nisa.dge_gather(..., scalar_offset=eid_offset)
```

### DGE `scalar_offset` absolute SBUF address bug (silent wrong results in E2E)
**Symptom**: Kernel passes standalone correctness tests but produces wrong outputs (e.g., expert indices ≥ E, garbage scatter/gather) when run inside a full model.
**Cause**: Using a computed `offset` (e.g., `offset=t*K+k`) in `.ap()` on an SBUF tensor bakes in an *absolute* SBUF address at compile time. In standalone tests the tensor is at a predictable SBUF location, so it resolves correctly. In E2E model compilation, earlier model tensors shift the allocation, so the baked address points to residual activations instead of the intended data.
**Fix**: Copy the single element of interest into a dedicated scratch tensor at its base (partition 0, element 0), then reference with `offset=0`:
```python
# WRONG — offset=t*K+k bakes wrong absolute address in E2E
eid_offset = expert_idx_sb.ap(pattern=[[K,1],[1,1]], offset=t*K+k)

# CORRECT — load element fresh from HBM, reference at offset=0
eid_scratch = nl.ndarray((128, 1), dtype=expert_indices.dtype, buffer=nl.sbuf)
nisa.dma_copy(
    dst=eid_scratch[0:1, 0:1],
    src=expert_indices.ap(pattern=[[K,1],[1,1]], offset=t*K+k),  # HBM .ap() is safe
)
eid_offset = eid_scratch.ap(pattern=[[1,1],[1,1]], offset=0)  # always offset=0
```
The `(128,1)` shape preserves `IndirectDimMaxIndex=127` for the compiler.

### "TensorScalarPtr arith immediate dtype must be fp32"
**Cause**: Using `bfloat16` affinities/scalars in POST_SCALE mode (e.g., MoE with POST_SCALE).
**Fix**: Cast affinities to `float32` before passing to the kernel:
```python
expert_affinities = expert_affinities.to(torch.float32)
```

### Compiler error after adding `affine_range`
**Cause**: `affine_range` used on a loop that carries a data dependency (accumulation into PSUM, sequential expert loop, etc.).
**Fix**: Revert to `sequential_range` for that loop. Only use `affine_range` when you are certain iterations are independent.

### Shape mismatch in `nc_matmul`
**Cause**: Wrong tensor layout — `stationary` must be `[par_dim, K]`, `moving` must be `[K, free_dim]`.
**Fix**: Verify layouts and add `nisa.nc_transpose` if needed before calling `nc_matmul`.

---

## Incorrect Results

### Output tensor stays zero / all zeros
**Causes** (in order of likelihood):
1. Forgot to call `nl.store(output_hbm, result_sbuf)`.
2. Stored to an SBUF tensor instead of the HBM output argument.
3. PSUM buffer not copied to SBUF before storing (PSUM→HBM direct store is not valid).

**Fix**:
```python
# Always: PSUM → SBUF → HBM
nl.copy(sbuf_tmp, psum_result)     # PSUM → SBUF
nl.store(output_hbm[...], sbuf_tmp) # SBUF → HBM
```

### NaN / Inf outputs
**Causes**:
- Float overflow: bf16 range is ~65504 max. Add scaling before accumulation.
- Division by zero in softmax denominator — add epsilon: `nl.maximum(denom, 1e-9)`.
- Uninitialized PSUM — always memset before use: `nl.zeros(...)` or explicit init.

### Wrong numerics (large max_diff)
**Causes**:
- Mixed precision accumulation: intermediate in bf16 when fp32 is needed.
- Incorrect reduction axis.
- Missing scale factor (e.g., `1/sqrt(d_k)` in attention).

---

## Performance Issues

### Kernel compiles but is slow
**Checklist**:
1. Are all independent DMA loads inside `affine_range` loops? (Enables prefetching)
2. Are there repeated `nl.load` calls for the same HBM tensor? (Use hoisting or caching)
3. Is SBUF overflowing? (Check `spill_save_bytes` in neuron-profile)
4. Are intermediate buffers declared outside inner loops? (Declare inside to avoid spilling — see performance-guide.md Opt #2 gotcha)

### DMA not overlapping with compute
**Cause**: Using `sequential_range` on loops with independent DMA loads.
**Fix**: Switch to `affine_range` on the outer tile loop.

### Poor MFU on matmul
**Causes**:
- Tile size too small — increase free-dimension tile size.
- Stationary matrix not held in SBUF across multiple moving tiles (missing loop reordering).

---

## Environment & Runtime

### "NeuronCore busy" / process conflicts
**Cause**: Another Python process is holding the NeuronCores.
**Fix**: Kill all other Python processes using Neuron devices, then re-run. NeuronCores are exclusive — only one process at a time.

### `nki.benchmark` not working
**Cause**: `nki.benchmark` does not work in the `pytorch_2_9_nxd_inference` environment.
**Fix**: Use wall-clock timing instead:
```python
import time, torch
def bench(fn, *args, warmup=5, iters=20):
    for _ in range(warmup): fn(*args)
    torch.xla.sync()
    t0 = time.perf_counter()
    for _ in range(iters): fn(*args)
    torch.xla.sync()
    print(f"{(time.perf_counter()-t0)/iters*1e3:.3f} ms/iter")
```

### `NEURON_PLATFORM_TARGET_OVERRIDE` not set
**Symptom**: Kernel compiles for wrong device or fails with architecture mismatch.
**Fix**: Always export before running:
```bash
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
```

---

## Debugging Protocol

When a kernel fails to compile:
1. Strip back to bare minimum (no fusion, no `affine_range`, no advanced tiling).
2. Add one optimization at a time, compiling after each addition.
3. Check shapes/dtypes at every intermediate step.
4. Use `print()` liberally — remember prints execute at compile time, so they're free for debugging shape issues.
5. For `nki.isa` failures: check tile/layout constraints in the architecture guide.

When a kernel gives wrong results:
1. Run on tiny shapes (e.g., T=2, H=64) to make inspection easier.
2. Print intermediate SBUF tensor values at compile time to verify shapes are correct.
3. Add `nl.device_print` for runtime values (expensive, debug-only).
4. Compare step-by-step against a NumPy reference.
