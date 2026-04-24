# SBUF Allocation: The All-or-Nothing Constraint

Source: `nki/backends/mlir_tracer/context.py`, `nkilib/core/utils/allocator.py`

---

## The Rule

**Every SBUF (and PSUM) tensor in a kernel must use either manual (fixed-address) allocation or automatic allocation — never both.**

Mixing modes causes `NCC_EGCA111: Memory allocation failed` at compile time, even when total SBUF usage is well below the 22 MiB limit.

---

## Two Modes

### Automatic allocation (default)

```python
x = nl.ndarray((PMAX, N), dtype=nl.bfloat16, buffer=nl.sbuf, name="x")
```

NCC performs register-coloring and assigns addresses. Tensors with non-overlapping live ranges can share SBUF space automatically.

### Manual (fixed-address) allocation

```python
sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("my_kernel"))
sbm.set_auto_alloc(False)
x = sbm.alloc((H0, N), dtype, buffer=nl.sbuf, name="x")
# emits: nl.ndarray(..., address=(base_partition, offset))
```

Tensor is pinned to a specific byte offset. **Required only** for nccl peer-core communication buffers whose SBUF address must be known at compile time for `all_reduce`.

---

## Why Mixing Fails

When mixing occurs, NCC's register-coloring cannot determine valid placements because part of SBUF is opaquely reserved by fixed-address tensors. With spilling disabled for NKI kernels, the compiler reports `NCC_EGCA111`. The error is attributed to the tensor with the **longest live range** — not the actual cause — making this class of bug misleading to diagnose.

---

## Fix Strategies

### Option A — Use `create_auto_alloc_manager` (recommended)

nkilib's canonical pattern: wrap the entire SBUF range in a `BufferManager` and allocate everything through it. Addresses are fixed at compile time and visible to register-coloring.

```python
from nkilib.core.utils.allocator import create_auto_alloc_manager

sbm = create_auto_alloc_manager()  # wraps full SBUF range
x   = sbm.alloc_stack((PMAX, N), dtype=nl.bfloat16, buffer=nl.sbuf)
# internally emits nl.ndarray(..., address=(partition, offset))
```

### Option B — Remove fixed-address requirement

If the nccl operation can be restructured to not require a compile-time-known SBUF address, remove the `BufferManager` entirely and let all tensors use automatic allocation.

---

## Key Facts

| Property | Value |
|----------|-------|
| Total usable SBUF per partition (trn2 / gen2) | 176,128 bytes (172 KiB) |
| Partitions (PMAX) | 128 |
| Total SBUF across all partitions | ~22 MiB |
| LNC=2 (two physical NCs fused) | ~44 MiB visible |
| Spilling disabled for NKI kernels | Yes — no fallback once allocation fails |
| Error code (mixing modes) | `NCC_EGCA111` (or `NCC_IGCA044` in older versions) |

---

## Diagnosis Checklist

If you see `NCC_EGCA111` and total allocation looks fine:

1. Search the kernel (including all called helper functions) for `nl.ndarray(..., buffer=nl.sbuf)` — these are automatic.
2. Search for `BufferManager`, `sbm.alloc`, `address=(...)` — these are manual.
3. If both exist anywhere in the kernel call graph, that is the cause.
4. The tensor named in the error is a symptom (longest live range), not the root.

---

## Reference

- Constraint enforcement: `nki/backends/mlir_tracer/context.py`
- Allocator: `nkilib/core/utils/allocator.py` (`BufferManager`, `create_auto_alloc_manager`)
- Canonical usage: `nkilib/experimental/transformer/attention_block_tkg.py` → `_rms_norm_inplace`
- SBUF size constants: `nki/language/tile_size.py`
