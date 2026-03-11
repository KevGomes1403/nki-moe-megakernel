# NKI Overview — Architecture & Programming Model

Source: AWS Neuron SDK documentation

---

## What is NKI?

Neuron Kernel Interface (NKI) is an open-source tool for developing kernels for Trainium hardware. Three components:

1. **NKI Programming Interface** — `nki.lang` (high-level tile programming, numpy/Triton-like) and `nki.isa` (direct hardware instruction access).
2. **NKI Compiler** — MLIR-based, converts NKI kernel code into optimized hardware instructions. Maintains developer-specified execution order and memory allocation.
3. **NKI Library (NKI-Lib)** — Ready-to-use optimized kernels (open source, Apache 2.0).

GitHub repos:
- Compiler: https://github.com/aws-neuron/nki
- Library: https://github.com/aws-neuron/nki-library
- Samples: https://github.com/aws-neuron/nki-samples

---

## Hardware Architecture

### NeuronCore Engines
Each NeuronCore has four specialized engines:

| Engine | Handles |
|--------|---------|
| **Tensor Engine** | Matrix multiplications |
| **Vector Engine** | Multi-input vector ops, reductions |
| **Scalar Engine** | Element-wise non-linear functions (hardware-accelerated) |
| **GpSimd Engine** | General-purpose programmable processors for custom ops |

**Collective Communication Engines (CC-Cores)** handle AllReduce/AllGather between NeuronCores and chips while computation continues.

### Memory Hierarchy

| Level | Type | Notes |
|-------|------|-------|
| **HBM** | High Bandwidth Memory (device memory) | Persistent storage, slowest |
| **SBUF** | State Buffer (on-chip SRAM) | Software-managed scratchpad for active computation |
| **PSUM** | Partial Sum Buffer | Near-memory accumulation of matmul results |

Unlike GPUs, Trainium uses **software-managed** memory hierarchy. NKI exposes all NISA primitives to control allocation and data movement explicitly.

### Device Generations
| Instance | Generation | Notes |
|----------|-----------|-------|
| `trn1`, `trn1n`, `inf2` | Trainium/Inferentia2 (NeuronCore-v2) | Baseline architecture |
| `trn2` | Trainium2 (NeuronCore-v3) | Enhanced; default for this environment |
| `trn3` | Trainium3 | Latest generation |

---

## NKI APIs

### `nki.lang` (high-level)
Memory allocation, tensor indexing, control flow — familiar to numpy/Triton users.

```python
import nki.language as nl

# Allocate on-chip (SBUF)
x_sbuf = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)

# Allocate off-chip (HBM)
y_hbm = nl.ndarray(shape, dtype=nl.float16, buffer=nl.shared_hbm)
```

### `nki.isa` (low-level)
Direct hardware instruction access — full control over instruction selection, scheduling, allocation.

```python
import nki.isa as nisa

# Matrix multiply on Tensor Engine
nisa.nc_matmul(dst=output, stationary=weights, moving=inputs)

# Element-wise op on Vector Engine
nisa.tensor_tensor(dst=output, data1=x, data2=y, op=nl.add)
```

Both APIs are designed to work together: `nki.lang` for indexing/memory, `nki.isa` for hardware-level precision.

---

## Loop Types (Scheduling Hints)

These are hints to the compiler — the compiler always ensures correctness regardless of choice.

### Sequential Range (default)
```python
for i in nl.sequential_range(8):
    result = process_tile(result_from_previous)  # may carry deps
```
Compiler does NOT reorder or parallelize. Use when in doubt.

### Affine Range (independent iterations)
```python
for i in nl.affine_range(8):
    process_tile(i)  # compiler can pipeline/unroll
```
Use when you are confident no inter-iteration dependencies exist. Enables DMA-compute overlap.

### Dynamic Range (runtime bounds)
```python
for i in nl.dynamic_range(lower, upper):
    process_tensor(t[i])  # evaluated on device at runtime
```
Use when bounds are unknown at compile time.

---

## Core Programming Model

NKI uses a **sequential programming model** — operations run in written order, but the compiler may reorder independent operations for parallelism.

**Critical distinction: compile-time vs. runtime**
- `print()`, Python conditionals, loop bounds → evaluated at **compile time**
- `nki.isa.*` calls → generate **runtime** hardware operations

```python
@nki.jit
def my_fn(x, y):
    print(f"shape: {x.shape}")           # compile-time print
    nisa.tensor_tensor(out, x, y, op=nl.add)  # runtime op
```

---

## Tensor Management

Tensors are on-device memory regions with queryable metadata: `dtype`, `shape`, `address`, `offset`, `pattern`, `buffer`.

```python
t = nl.ndarray((128, 128), nl.float16, nl.sbuf)
u = t.reshape((128, 2, 64))  # same memory, different view

# Indexing (numpy-compatible)
u = t[0, 0, 10]       # single element
u = t[:, 0, :]        # slice
u = t[::2, :, ::2]    # step indexing
u = t[0, ...]         # ellipsis
```

**SBUF layout**: 2D — first dimension = partition dimension, remaining = free dimension.
`nl.par_dim(128)` marks the partition dimension size.

---

## Compilation Flow

```
@nki.jit function
        ↓
    NKI Compiler (MLIR)
        ↓
    NKI IR  →  Neuron Graph Compiler
                    ↓
              NEFF executable
```

The `@nki.jit` decorator triggers the NKI compiler during Python tracing.
In PyTorch, use `@nki_op("mylib::my_op")` to register as a custom operator.

---

## Using NKI Kernels

Three approaches (in order of effort):
1. **Automatic** — Neuron compiler injects NKI kernels during model compilation (zero code changes).
2. **NKI Library** — Import and call pre-built kernels from `nki-library`.
3. **Custom kernels** — Write from scratch with `nki.lang` / `nki.isa`.

```python
# Custom kernel example
import torch
from torch_neuronx import nki_op
import nki.language as nl

@nki.jit
def my_kernel(in_ptr, out_ptr):
    # kernel implementation
    pass

@nki_op("mylib::my_op", mutates_args={})
def my_op(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    my_kernel(x, out)
    return out
```
