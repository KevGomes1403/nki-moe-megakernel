# NKI SBUF Allocation: The All-or-Nothing Constraint

## Overview

NKI enforces a hard rule: **within a single kernel, every SBUF (and PSUM) tensor must use either manual (fixed-address) allocation or automatic allocation — never both.** Mixing the two modes will cause `NCC_EGCA111: Memory allocation failed` at compile time, even when far less than the available 28 MiB of SBUF is in use.

---

## Two Allocation Modes

### Automatic allocation

```python
x = nl.ndarray((PMAX, N), dtype=nl.bfloat16, buffer=nl.sbuf, name="x")
```

The NCC backend performs register-coloring and assigns the address. Tensors can be reused / overlapped across non-overlapping live ranges.

### Manual (fixed-address) allocation

```python
sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("my_kernel"))
sbm.set_auto_alloc(False)
x = sbm.alloc((H0, N), dtype, buffer=nl.sbuf, name="x")
# internally emits: nl.ndarray(..., address=(base_partition, offset))
```

The tensor is pinned to a specific byte offset in SBUF. Required for tensors whose SBUF address must be known at compile time (e.g., nccl peer-core communication buffers for `all_reduce`).

---

## Why Mixing Fails

The constraint is enforced in:

```
/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/nki/backends/mlir_tracer/context.py
```

```python
def record_manual_allocation(self, memspace: MemSpace) -> None:
    if self._has_auto_sbuf_psum:
        raise ValueError(
            "Cannot mix manual and automatic allocations for SBUF/PSUM. "
            "Either ALL SBUF/PSUM ndarrays must use address=..., or NONE. "
            "An automatic allocation was already made earlier in the kernel."
        )

def record_auto_allocation(self, memspace: MemSpace) -> None:
    if self._has_manual_sbuf_psum:
        raise ValueError(
            "Cannot mix manual and automatic allocations for SBUF/PSUM. "
            "Either ALL SBUF/PSUM ndarrays must use address=..., or NONE. "
            "A manual allocation (with address=...) was already made earlier in the kernel."
        )
```

When mixing occurs, the NCC allocator is placed in an incoherent state. It can't perform valid register coloring for the auto-allocated tensors because part of SBUF is opaquely reserved by fixed-address tensors. With no valid placement found and spilling disabled for NKI kernels, the compiler reports `NCC_EGCA111` (or `NCC_IGCA044` in earlier versions), attributing the failure to whichever tensor has the longest live range — even though that tensor is not the actual cause.

---

## Affected Kernels

Both `transformer_tkg_qwen.py` and `attn_tkg_qwen_v2.py` have this issue.

### `kernels/transformer/transformer_tkg_qwen.py`

```python
# Line 51
SBM_SIZE_BYTES = 32 * 1024  # fixed SBUF region for nccl all_reduce tensors

# Lines 1210–1252 (outer kernel function)
sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("transformer_tkg_qwen"))
sbm.set_auto_alloc(False)
attn_sharded_sb = sbm.alloc((H0, BxS * h1_shard), dtype, buffer=nl.sbuf, name="attn_sharded")  # MANUAL
moe_sharded_sb  = sbm.alloc((H0, BxS * h1_shard), dtype, buffer=nl.sbuf, name="moe_sharded")   # MANUAL

# Lines 256, 867–870, ... (inner functions / loop body)
rms_ones      = nl.ndarray((PMAX, PMAX), ..., buffer=nl.sbuf)   # AUTO  ← conflict
gate_up_buf0  = nl.ndarray((_PMAX, H_FREE, GU_FLAT), ..., buffer=nl.sbuf)  # AUTO  ← conflict
# ... many more auto-allocated tensors
```

### `kernels/attn_tkg/attn_tkg_qwen_v2.py`

```python
# Line 42
SBM_SIZE_BYTES = 8 * 1024

# Lines 684–699 (outer kernel function)
sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("attn_tkg_qwen_v2"))
sbm.set_auto_alloc(False)
attn_sharded_sb = sbm.alloc((H0, BxS * h1_shard), dtype, buffer=nl.sbuf, name="attn_sharded")  # MANUAL

# Lines 209, 237–435, ... (inner function _qwen3_attn_tkg_v2)
rms_ones  = nl.ndarray((PMAX, PMAX), ..., buffer=nl.sbuf)   # AUTO  ← conflict
qnw_bf16  = nl.ndarray((PMAX, 1),    ..., buffer=nl.sbuf)   # AUTO  ← conflict
# ... many more auto-allocated tensors
```

The error message names `rms_ones` because it has the longest live range and is the last tensor the allocator attempts to spill. It is a symptom, not the root cause.

---

## Fix Strategy

There are two valid approaches. Both are all-or-nothing within a kernel.

### Option A: Put everything under `BufferManager` (all manual)

Use a single `BufferManager` spanning the full available SBUF
(`nl.tile_size.total_available_sbuf_size`) and allocate every SBUF tensor
through it. The nkilib canonical pattern:

```python
from nkilib.core.utils.allocator import create_auto_alloc_manager

sbm = create_auto_alloc_manager()           # wraps the full SBUF range
x   = sbm.alloc_stack((PMAX, N), dtype=nl.bfloat16, buffer=nl.sbuf)
```

`alloc_stack` with `use_auto_alloc=False` emits `nl.ndarray(..., address=(partition, offset))`.
All addresses are compiler-visible and register-coloring works normally.

### Option B: Use `all_gather`/`all_reduce` without a fixed-address buffer

If the nccl operation can be restructured so it does not require a pre-known
SBUF address, all tensors can remain auto-allocated and the `BufferManager`
can be removed entirely. This is the simpler option when the communication
primitive supports it.

---

## Key Facts

| Property | Value |
|---|---|
| Total usable SBUF per partition (trn2 / gen2) | 176,128 bytes (172 KiB) |
| Number of partitions (PMAX) | 128 |
| Total SBUF across all partitions | ~22 MiB |
| LNC=2 sharding (two logical NCs) | ~22 MiB × 2 = ~44 MiB visible |
| Error when mixing allocations | `NCC_EGCA111` |
| Spilling disabled for NKI kernels | Yes — no fallback once allocation fails |

---

## Reference

- Constraint implementation: `nki/backends/mlir_tracer/context.py`
- Allocator implementation: `nkilib/core/utils/allocator.py` (`BufferManager`, `SbufManager`, `create_auto_alloc_manager`)
- Canonical nkilib usage example: `nkilib/experimental/transformer/attention_block_tkg.py` → `_rms_norm_inplace` (uses `sbm.alloc_stack` for the ones matrix)
- SBUF sizing constants: `nki/language/tile_size.py`
