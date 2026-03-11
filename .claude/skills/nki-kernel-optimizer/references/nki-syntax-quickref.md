# NKI Syntax Quick Reference

---

## Imports

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
```

---

## Kernel Declaration

```python
@nki.jit
def my_kernel(input_hbm: nl.ndarray, output_hbm: nl.ndarray):
    ...

# With platform target
my_kernel_jit = nki.jit(my_kernel, platform_target="trn2")
result = my_kernel_jit(input_tensor)

# Trace mode (for dispatch wrappers — returns 0, not the output tensor)
status = nki.jit(wrapper_fn, mode="trace")(arg1=..., arg2=...)
```

---

## Memory Allocation

```python
# SBUF (on-chip scratchpad) — fast, software-managed
buf = nl.ndarray((nl.par_dim(128), 512), dtype=nl.float32, buffer=nl.sbuf)

# PSUM (partial sum buffer — near matmul output)
psum = nl.ndarray((nl.par_dim(128), 512), dtype=nl.float32, buffer=nl.psum)

# HBM (device memory — inputs/outputs)
out = nl.ndarray(shape, dtype=nl.float16, buffer=nl.shared_hbm)

# Reshape (same memory, different view)
reshaped = buf.reshape((128, 8, 64))
```

**Partition dimension**: always the first tensor dimension; `nl.par_dim(128)` on trn2.
**Free dimension**: remaining dimensions, laid out contiguously.

---

## Data Movement

```python
# HBM → SBUF
tile = nl.load(hbm_tensor[i, ...])                   # basic load
tile = nl.load(hbm_tensor[i, ...], dtype=nl.bfloat16) # with cast

# SBUF → HBM
nl.store(hbm_tensor[i, ...], tile)

# SBUF → SBUF (copy)
nl.copy(dst=sbuf_b, src=sbuf_a)

# DMA copy (nisa, more direct)
nisa.dma_copy(dst=dst_tensor, src=src_tensor)
```

**Rule**: `.ap()` is only valid on HBM tensors. Never call `.ap()` on SBUF tensors.

---

## Loop Types

```python
# Sequential — safe default, no reordering
for i in nl.sequential_range(N):
    ...

# Affine — independent iterations, enables DMA-compute overlap and unrolling
for i in nl.affine_range(N):
    ...

# Static — compile-time unroll hint (similar to affine)
for i in nl.static_range(N):
    ...

# Dynamic — runtime bounds, runs on device
for i in nl.dynamic_range(lower, upper):
    ...

# Plain Python range — equivalent to sequential_range for compile-time constants
for i in range(N):
    ...
```

---

## Compute — `nki.lang` (high-level)

```python
# Elementwise
c = nl.add(a, b)
c = nl.multiply(a, b)
c = nl.exp(a)
c = nl.sqrt(a)
c = nl.rsqrt(a)           # reciprocal sqrt
c = nl.maximum(a, b)

# Reduction along free dimension
s = nl.sum(a, axis=[1])
m = nl.max(a, axis=[1])

# Matmul (returns PSUM tensor)
out_psum = nl.matmul(lhs, rhs, transpose_x=False)

# Cast
a_bf16 = nl.cast(a, dtype=nl.bfloat16)

# Softmax
s = nl.softmax(a, axis=[1])
```

---

## Compute — `nki.isa` (low-level, hardware-direct)

```python
# Matrix multiply on Tensor Engine
# stationary: [par_dim, K], moving: [K, free_dim], dst: [par_dim, free_dim] in PSUM
nisa.nc_matmul(dst=psum_out, stationary=weight_sbuf, moving=input_sbuf)

# Elementwise tensor-tensor
nisa.tensor_tensor(dst=out, data1=a, data2=b, op=nl.add)
nisa.tensor_tensor(dst=out, data1=a, data2=b, op=nl.multiply)

# Tensor-scalar
nisa.tensor_scalar(dst=out, data=a, scalar0=scale, op0=nl.multiply)

# Activation (Scalar Engine)
nisa.activation(dst=out, data=a, op=nl.exp)
nisa.activation(dst=out, data=a, op=nl.rsqrt)

# Copy PSUM → SBUF
nisa.tensor_copy(dst=sbuf_out, src=psum_in)

# Transpose
nisa.nc_transpose(dst=out, data=inp)

# Barrier (sync all NeuronCores)
nisa.barrier()

# Collective communication
nisa.sendrecv()
```

---

## Indexing

```python
# Single element
x = t[0, 0]

# Slice
x = t[i:i+128, :]

# Step
x = t[::2, :]

# Ellipsis
x = t[0, ...]

# Column slice (SBUF)
col = rw_sb[0:T, expert_id:expert_id+1]   # use direct slice, NOT .ap()
```

---

## Dynamic Control Flow (on-device)

```python
# Registers for runtime conditions
reg = nisa.register_alloc(initial_value)
nisa.register_load(reg, tensor)    # load from tensor
nisa.register_store(tensor, reg)   # store to tensor
nisa.register_move(dst_reg, src_reg)

# Dynamic while loop
while reg:
    compute(...)
    nisa.register_load(reg, cond_tensor)  # update condition
```

---

## Device Print (runtime debug)

```python
# NOT regular print() — that runs at compile time
nl.device_print("val =", tensor)
```

---

## Common Data Types

| NKI dtype | Description |
|-----------|-------------|
| `nl.float32` / `nl.fp32` | 32-bit float |
| `nl.bfloat16` / `nl.bf16` | BFloat16 |
| `nl.float16` / `nl.fp16` | Float16 |
| `nl.int32` | 32-bit integer |
| `nl.int8` | 8-bit integer |

**Important**: POST_SCALE MoE mode requires `float32` affinities — `bfloat16` causes compiler error.

---

## Kernel Invocation Patterns

### Pattern A: `@nki.jit` decorated (e.g., `attention_cte`)
```python
output = my_kernel(input1, input2, ...)   # direct call
```

### Pattern B: Trace-mode dispatch wrapper (e.g., `moe_cte`)
```python
status = nki.jit(wrapper, mode="trace")(hidden_states=..., spec=spec)
# Returns int 0 — output tensor is NOT the return value
```

### Pattern C: Standard `nki.jit` with Python-dispatch wrapper (e.g., `moe_tkg`)
```python
kernel_jit = nki.jit(my_fn, platform_target="trn2")
output = kernel_jit(input=..., flag=True, ...)   # returns output tensor
# Boolean args evaluated at trace time
```
