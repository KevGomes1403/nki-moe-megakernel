# NKI Syntax Reference and Authoring Guide

This file is a practical reference for writing and reviewing NKI kernels. It focuses on the syntax and programming model you’re most likely to need when generating or improving kernels.

It is not a full API reference. It is a field guide for authoring.

## 1) Mental model

NKI kernels are written in Python syntax, but they are **compiled** and run on Neuron hardware.

Two consequences matter a lot:

1. Most ordinary Python expressions in the kernel are evaluated at **compile time**.
2. Calls that lower to hardware operations, especially `nki.isa.*`, correspond to **runtime device work**.

That distinction affects debugging, control flow, and performance tuning.

### Compile-time vs runtime example

```python
@nki.jit
def my_kernel(x, y):
    print(f"x shape = {x.shape}")   # compile-time diagnostic
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    z = nl.load(x) + nl.load(y)
    nl.store(out, z)                # runtime effect
    return out
````

The `print(...)` helps during compilation; it is not device-side output.

---

## 2) Imports and naming conventions

Typical imports:

```python
from neuronxcc.nki import jit, benchmark
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np
```

Common conventions used in generated code:

* `*_tensor` means a tensor handle passed into or returned from the kernel.
* short names like `x`, `y`, `acc`, `tmp` usually represent loaded tiles or intermediates.
* `out_tensor` / `dst_tensor` for outputs.
* `sbuf_*` names for on-chip scratch tiles.
* `psum_*` names for accumulation buffers.

---

## 3) Kernel skeletons

### 3.1 Minimal kernel skeleton

```python
from neuronxcc.nki import jit
import neuronxcc.nki.language as nl

@jit
def add_one(x_tensor):
    y_tensor = nl.ndarray(x_tensor.shape, dtype=x_tensor.dtype, buffer=nl.shared_hbm)

    x = nl.load(x_tensor)
    y = x + 1

    nl.store(y_tensor, y)
    return y_tensor
```

This is the default structure to start from:

1. Allocate output tensors explicitly.
2. Load inputs explicitly.
3. Compute.
4. Store explicitly.
5. Return output tensor(s).

### 3.2 Multi-input skeleton

```python
@jit
def add_kernel(a_tensor, b_tensor):
    assert a_tensor.shape == b_tensor.shape
    assert a_tensor.dtype == b_tensor.dtype

    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    a = nl.load(a_tensor)
    b = nl.load(b_tensor)
    out = a + b

    nl.store(out_tensor, out)
    return out_tensor
```

### 3.3 Working-set-in-SBUF skeleton

```python
@jit
def fused_kernel(x_tensor):
    out_tensor = nl.ndarray(x_tensor.shape, dtype=x_tensor.dtype, buffer=nl.shared_hbm)

    x_sbuf = nl.ndarray(x_tensor.shape, dtype=x_tensor.dtype, buffer=nl.sbuf)
    tmp_sbuf = nl.ndarray(x_tensor.shape, dtype=x_tensor.dtype, buffer=nl.sbuf)

    x = nl.load(x_tensor)
    # optionally materialize / operate on-chip
    tmp = x * x + 1

    nl.store(out_tensor, tmp)
    return out_tensor
```

### 3.4 Benchmark wrapper skeleton

```python
from neuronxcc.nki import benchmark
import neuronxcc.nki.language as nl

@benchmark(warmup=10, iters=100, save_neff_name="file.neff", save_trace_name="profile.ntff")
def add_bench(a_tensor, b_tensor):
    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    a = nl.load(a_tensor)
    b = nl.load(b_tensor)
    out = a + b

    nl.store(out_tensor, out)
    return out_tensor
```

Remember: benchmark results are for performance measurement, not correctness validation.

---

## 4) Tensors and memory placement

Tensors are the core value type in NKI. A tensor is a reference to device memory plus metadata.

Common tensor metadata:

* `t.shape`
* `t.dtype`
* `t.buffer`
* `t.address`
* `t.offset`
* `t.pattern`

The most commonly used fields in authoring are `shape`, `dtype`, and `buffer`.

### 4.1 Creating tensors with `nl.ndarray`

```python
t = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)
u = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.shared_hbm)
```

This allocates a tensor with the given shape, dtype, and backing memory.

### 4.2 Important buffers

#### `nl.shared_hbm`

Use for:

* input tensors arriving from outside the kernel
* output tensors returned to the caller
* large data that cannot stay on-chip

Typical pattern:

```python
out_tensor = nl.ndarray(x_tensor.shape, dtype=x_tensor.dtype, buffer=nl.shared_hbm)
```

#### `nl.sbuf`

Use for:

* hot working tiles
* intermediates that should remain on-chip
* staging buffers to reduce HBM traffic

Typical pattern:

```python
tile = nl.ndarray((128, 512), dtype=nl.float16, buffer=nl.sbuf)
```

#### `nl.psum`

Use for:

* accumulation associated with tensor-engine style compute
* partial sums that should not spill unnecessarily

Use only when it fits the algorithm and instruction mix.

### 4.3 Rule of thumb for placement

* Inputs/outputs: usually HBM
* Reused intermediates: SBUF
* Matmul accumulations: PSUM where applicable

A common mistake is to allocate outputs in HBM and also do all intermediate computation effectively in HBM, which leaves performance on the table.

---

## 5) Loading and storing

### 5.1 Loading inputs

```python
x = nl.load(x_tensor)
```

`nl.load(...)` brings data into the computational context. In simple generated examples, this is usually how you start computation from a passed tensor.

### 5.2 Storing outputs

```python
nl.store(out_tensor, out)
```

Most kernels must end with an explicit store into an output tensor that lives in a caller-visible buffer, usually `nl.shared_hbm`.

Common mistake:

* compute `out`
* return the wrong tensor object
* forget `nl.store(...)`

### 5.3 Full pattern

```python
@jit
def relu_kernel(x_tensor):
    out_tensor = nl.ndarray(x_tensor.shape, dtype=x_tensor.dtype, buffer=nl.shared_hbm)

    x = nl.load(x_tensor)
    out = x * (x > 0)

    nl.store(out_tensor, out)
    return out_tensor
```

---

## 6) Reshape, views, and lower-level allocation

### 6.1 Reshape

```python
t = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)
u = t.reshape((128, 2, 64))
```

This creates an alternate view of the same memory.

Use reshape when:

* the data is already laid out compatibly
* you need a different logical interpretation
* you do not want to copy

### 6.2 Pointer / region based allocation

For more control, you can define memory regions and create views on top of them.

Conceptually:

```python
region = nl.sbuf.ptr(size=(128, 64))
t = region.view(nl.float16, (128, 32))
```

You can also define offsets to control relative placement:

```python
region1 = nl.sbuf.ptr(size=(128, 64), offset=(0, 0))
region2 = nl.sbuf.ptr(size=(128, 64), offset=(0, 64))
t1 = region1.view(nl.float16, (128, 32))
t2 = region2.view(nl.float16, (128, 32))
```

Use lower-level allocation when:

* you want explicit scratch layout
* you want neighboring tensors in memory
* you want buffer reuse patterns that `ndarray` alone does not express cleanly

---

## 7) Indexing and access patterns

NKI tensor indexing is close to NumPy syntax, but it also corresponds to hardware access behavior.

### 7.1 Basic indexing

```python
u = t[0, 0, 10]
v = t[:, 0, :]
w = t[::2, :, ::2]
z = t[0, ..., :]
```

Supported styles you’ll use most:

* integer indexing
* slices
* ellipsis

### 7.2 Shape intuition

If `t.shape == (64, 64, 64)`, then:

```python
u = t[0, ...]
```

typically gives:

```python
u.shape == (64, 64)
```

### 7.3 Partition dimension convention

SBUF is logically partitioned. By convention, the **first tensor dimension** corresponds to the partition dimension.

That means the first dimension is especially important for tiling and layout. When generating kernels, you should think carefully about what dimension should become the first dimension of an on-chip tile.

### 7.4 Access pattern metadata

Advanced code may inspect:

```python
print(t.offset)
print(t.pattern)
```

Or explicitly construct hardware access patterns. Most generated kernels should avoid that unless necessary.

---

## 8) Arithmetic and elementwise expressions

In many kernels, `nki.language` expressions are the simplest starting point.

### 8.1 Straightforward elementwise compute

```python
x = nl.load(x_tensor)
y = nl.load(y_tensor)

z = x + y
w = x * y
u = x - y
```

### 8.2 Slightly more complex fused expressions

```python
tmp = x * scale + bias
out = tmp * (tmp > 0)
```

This is where fusion often starts: chain several simple operations before storing to HBM.

### 8.3 Generated-kernel guideline

Prefer `nl`-level expressions first when:

* the operation is elementwise or straightforward
* you are building a correct baseline
* there is no clear need for explicit ISA control yet

Drop to `nisa.*` when the algorithm or performance target requires more direct hardware mapping.

---

## 9) Assertions and validation inside kernels

Assertions are useful for guarding assumptions during compilation.

```python
assert x_tensor.shape == y_tensor.shape
assert x_tensor.dtype == y_tensor.dtype
```

Use assertions to make generated kernels fail early on:

* shape mismatches
* unsupported dtypes
* tile assumptions
* divisibility requirements

Example:

```python
assert x_tensor.shape[0] % 128 == 0, "expected partition dimension multiple of 128"
```

Do not overdo assertions, but include the ones that make invariants explicit.

---

## 10) Control flow: compile-time loops

NKI supports ordinary Python-like control flow, but most of it is compile-time control flow unless you explicitly use dynamic constructs.

### 10.1 Ordinary `for` loops

```python
for i in range(4):
    ...
```

This is generally compile-time structured control flow.

### 10.2 `nl.sequential_range(...)`

Use this as the safe default when there may be dependencies between iterations.

```python
for i in nl.sequential_range(8):
    acc = process_step(acc, i)
```

Use sequential loops when:

* iteration `i` depends on iteration `i - 1`
* you are unsure whether reordering is safe
* you want the most conservative semantics

### 10.3 `nl.affine_range(...)`

Use this only when iterations are independent and reordering/unrolling is safe.

```python
for i in nl.affine_range(8):
    process_tile(i)
```

Typical use cases:

* independent tiles
* repeated computation over independent blocks
* situations where unrolling or pipelining should be legal

Common mistake:

* changing a loop from sequential to affine without verifying no loop-carried dependency exists

### 10.4 `nl.static_range(...)`

This is another hint-style loop form in some NKI contexts. Use it when the iteration structure is static and known, but the main practical distinction in generated code is usually between sequential and affine.

### 10.5 Rule of thumb

* Start with `nl.sequential_range(...)`
* Move to `nl.affine_range(...)` only after confirming independence

---

## 11) Dynamic control flow and registers

Dynamic control flow runs on the device rather than being fully unrolled/expanded at compile time.

This is more advanced, but important for some kernels.

### 11.1 Dynamic loops

```python
for i in nl.dynamic_range(10):
    process_tensor(t[i])
```

This means the loop executes on-device.

### 11.2 Register allocation and usage

Typical APIs include:

* `nisa.register_alloc(...)`
* `nisa.register_move(...)`
* `nisa.register_load(...)`
* `nisa.register_store(...)`

Conceptual example:

```python
reg = nisa.register_alloc(1)
```

or loading from a tensor:

```python
count = nisa.register_alloc(count_tensor)
```

### 11.3 Dynamic loop with register upper bound

```python
count = nisa.register_alloc(count_tensor)

for i in nl.dynamic_range(count):
    process_tensor(t[i])
```

### 11.4 Dynamic `while` loop

```python
cond = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
reg = nisa.register_alloc(1)

while reg:
    # do work that updates cond
    nisa.register_load(reg, cond)
```

Use dynamic control flow when:

* loop trip count is not known statically
* control depends on device-side computed values
* the algorithm genuinely needs runtime branching/iteration

Do not introduce dynamic control flow unless you need it; it is harder to reason about and optimize.

---

## 12) ISA-level operations

The `nki.isa` layer gives you lower-level control and more direct mapping to hardware instructions.

Use it when you need stronger control over:

* instruction selection
* tensor engine usage
* data movement
* barriers / synchronization
* runtime control flow
* core-to-core primitives

### 12.1 Example: matmul-style primitive

```python
nisa.nc_matmul(dst=output, stationary=a_tile, moving=b_tile)
```

When using ISA primitives:

* be explicit about destination tensors
* verify dtype and shape constraints
* verify layout and buffer constraints
* keep the code readable with comments about intended engine usage

### 12.2 Example: tensor-tensor op pattern

```python
nisa.tensor_tensor(dst=output, data1=x, data2=y, op=nl.add)
```

Pattern to remember:

* destination is explicit
* source tensors are explicit
* operation is explicit

### 12.3 Barrier / communication examples

```python
nisa.barrier()
nisa.sendrecv()
```

These belong in advanced kernels where inter-core coordination matters.

### 12.4 When to stay at `nl` level

Stay at `nl` level when:

* you can express the math simply
* you are building the first correct version
* explicit ISA selection is not yet needed

### 12.5 When to drop to `nisa`

Drop to `nisa` when:

* a tensor-engine operation should be explicit
* you need register-based control flow
* you need explicit synchronization or communication
* profile evidence says the high-level version is leaving performance on the table

---

## 13) Memory movement strategy patterns

A good NKI kernel is often a good data-movement plan.

### 13.1 Bad pattern: repeated HBM traffic

```python
# conceptual anti-pattern
x = nl.load(x_tensor)
tmp1 = f(x)
# store too early
nl.store(tmp_tensor, tmp1)
# later reload for another stage
tmp2 = nl.load(tmp_tensor)
out = g(tmp2)
nl.store(out_tensor, out)
```

This often introduces unnecessary HBM traffic.

### 13.2 Better pattern: keep intermediates on-chip

```python
x = nl.load(x_tensor)
tmp = f(x)
out = g(tmp)
nl.store(out_tensor, out)
```

### 13.3 Buffer choice checklist

Before you allocate a tensor, ask:

* Does this tensor need to be visible outside the kernel? → HBM
* Is this a reused intermediate? → SBUF
* Is this a partial accumulation near tensor-engine compute? → PSUM

---

## 14) Tiling patterns

Most performance-sensitive kernels need explicit thinking about tiles.

### 14.1 Basic tile loop structure

```python
for i in nl.sequential_range(num_tiles):
    x_tile = ...
    y_tile = ...
    out_tile = ...
```

### 14.2 Independent tile processing

```python
for i in nl.affine_range(num_tiles):
    process_tile(i)
```

This is only correct if each tile is independent.

### 14.3 Generated-code guidance

When generating a tiled kernel, always make these things explicit in comments:

* tile shape
* what dimension is the partition dimension
* where the tile lives
* whether tiles are independent
* whether the loop form is sequential or affine and why

Example:

```python
# Tile shape: (128, 512)
# First dim is partition dim in SBUF
# Iterations are independent across tiles, so affine_range is legal
for tile_idx in nl.affine_range(num_tiles):
    ...
```

---

## 15) Correctness harness patterns

A good generated deliverable should include a correctness test outside the kernel.

### 15.1 Minimal structure

```python
import numpy as np

def reference_impl(x, y):
    return x + y

x = np.random.randn(128, 1024).astype(np.float32)
y = np.random.randn(128, 1024).astype(np.float32)

out_ref = reference_impl(x, y)
out_kernel = my_kernel(x, y)

np.testing.assert_allclose(out_kernel, out_ref, rtol=1e-5, atol=1e-5)
```

### 15.2 Why correctness must be separate from benchmark

`nki.benchmark` is for performance statistics. Its output should not be treated as numerically validated output for correctness checking.

Always include:

* deterministic inputs
* reference implementation
* `assert_allclose`
* explicit tolerances

---

## 16) Benchmarking patterns

### 16.1 Minimal benchmark example

```python
from neuronxcc.nki import benchmark
import neuronxcc.nki.language as nl
import numpy as np

@benchmark(warmup=10, iters=100, save_neff_name="file.neff", save_trace_name="profile.ntff")
def add_kernel(a_tensor, b_tensor):
    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    a = nl.load(a_tensor)
    b = nl.load(b_tensor)
    out = a + b

    nl.store(out_tensor, out)
    return out_tensor

a = np.zeros((128, 1024), dtype=np.float32)
b = np.random.random_sample((128, 1024)).astype(np.float32)

_ = add_kernel(a, b)

metrics = add_kernel.benchmark_result.nc_latency
print("p50(us) =", metrics.get_latency_percentile(50))
print("p99(us) =", metrics.get_latency_percentile(99))
```

### 16.2 Useful benchmark options

* `warmup`
* `iters`
* `save_neff_name`
* `save_trace_name`
* `additional_compile_opt`

### 16.3 What to report

At minimum:

* shape
* dtype
* device generation
* p50
* p99
* any compiler flags used

---

## 17) Common authoring patterns

### 17.1 Elementwise unary kernel

```python
@jit
def square_kernel(x_tensor):
    out_tensor = nl.ndarray(x_tensor.shape, dtype=x_tensor.dtype, buffer=nl.shared_hbm)

    x = nl.load(x_tensor)
    out = x * x

    nl.store(out_tensor, out)
    return out_tensor
```

### 17.2 Elementwise binary kernel

```python
@jit
def mul_kernel(a_tensor, b_tensor):
    assert a_tensor.shape == b_tensor.shape
    assert a_tensor.dtype == b_tensor.dtype

    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    a = nl.load(a_tensor)
    b = nl.load(b_tensor)
    out = a * b

    nl.store(out_tensor, out)
    return out_tensor
```

### 17.3 Fused pointwise kernel

```python
@jit
def fused_affine_relu(x_tensor, scale, bias):
    out_tensor = nl.ndarray(x_tensor.shape, dtype=x_tensor.dtype, buffer=nl.shared_hbm)

    x = nl.load(x_tensor)
    tmp = x * scale + bias
    out = tmp * (tmp > 0)

    nl.store(out_tensor, out)
    return out_tensor
```

### 17.4 Skeleton with explicit loop hint

```python
@jit
def tiled_kernel(x_tensor):
    out_tensor = nl.ndarray(x_tensor.shape, dtype=x_tensor.dtype, buffer=nl.shared_hbm)

    for tile_idx in nl.sequential_range(4):
        # load/process/store per tile
        pass

    return out_tensor
```

---

## 18) Common mistakes and how to avoid them

### Mistake 1: forgetting `nl.store(...)`

Symptom: kernel appears to run but output is wrong or not materialized.

Fix: always end each output path with an explicit store.

### Mistake 2: returning an on-chip temporary as the final result

Symptom: caller expects an HBM-visible output tensor, but the kernel returns the wrong object.

Fix: allocate a proper output tensor in `nl.shared_hbm`, store into it, and return it.

### Mistake 3: using `nl.affine_range(...)` on dependent loops

Symptom: incorrect results or bad compiler behavior.

Fix: switch back to `nl.sequential_range(...)` unless independence is proven.

### Mistake 4: unnecessary HBM roundtrips

Symptom: poor benchmark numbers dominated by movement.

Fix: keep intermediates in SBUF and fuse straightforward compute.

### Mistake 5: relying on benchmark output for correctness

Fix: benchmark separately from numerical validation.

### Mistake 6: not documenting tile assumptions

Fix: comment tile shapes, dependency assumptions, and why the loop form is legal.

---

## 19) Practical generation checklist

When writing or improving a kernel, verify each item:

* [ ] Inputs and outputs are explicitly typed by shape/dtype assumptions
* [ ] Output allocation is explicit
* [ ] Buffer choice is intentional
* [ ] Loads and stores are explicit
* [ ] Loop form is justified
* [ ] High-level math is correct before low-level tuning
* [ ] ISA usage is explained where present
* [ ] Correctness harness exists
* [ ] Benchmark harness exists if performance matters

---

## 20) Preferred output structure for generated responses

When generating NKI code for a user, the deliverable should usually include:

1. The kernel
2. A short explanation of memory placement
3. A correctness test
4. A benchmark snippet
5. Tuning notes

Recommended structure:

```python
# 1) kernel

# 2) correctness harness

# 3) benchmark harness
```

And then a prose section:

* why this buffer layout was chosen
* why loops are sequential or affine
* what to measure next
* what to tune next

---

## 21) Quick-reference summary

### Use `nl` for:

* baseline kernels
* simple elementwise math
* readable load/compute/store structure
* first correct implementation

### Use `nisa` for:

* explicit tensor-engine ops
* explicit vector/tensor instructions
* register-based dynamic control flow
* synchronization / communication
* performance-critical low-level control

### Use `nl.shared_hbm` for:

* inputs/outputs
* large external tensors

### Use `nl.sbuf` for:

* hot tiles
* scratch buffers
* reused intermediates

### Use `nl.psum` for:

* accumulation where tensor-engine style compute warrants it

### Use `nl.sequential_range(...)` when:

* there may be loop-carried dependencies
* you are not sure

### Use `nl.affine_range(...)` when:

* iterations are truly independent
* you want compiler freedom for unrolling/pipelining

### Always:

* separate correctness from benchmarking
* comment tile assumptions
* explain buffer placement
* measure after every optimization step
