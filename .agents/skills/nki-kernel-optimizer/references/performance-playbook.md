NKI Performance Optimizations
==============================

A well-optimized NKI kernel should be either **compute-bound** (a compute engine active 90%+ of execution time) or
**memory-bound** (device memory bandwidth utilization 60%+). The optimizations below are organized into three categories
to help reach one of those two endpoints.

**Note**: This document contains NKI Beta 1 syntax which uses `nl.load` and `nl.store`. They are meant to demonstrate general optimization principles - use NKI Beta 2 syntax in your implementation, which uses `nisa.dma_copy`.

---

## 1. Improving Arithmetic Intensity

Arithmetic intensity = compute ops / bytes read from HBM. When it's too low, compute engines starve waiting on DMA.
The two common causes are **input data reloading** and **intermediate data spilling**.

### Opt #1: Exploit temporal locality to minimize input data reloading

**Problem**: The same HBM input tile is loaded multiple times across iterations. For example, in a tiled matmul the
`lhsT[m, k]` tile gets loaded once per `n` iteration instead of once per `m` iteration.

**Solution**: Hoist loads to the outermost loop that covers all consumers of that tile. Load the tile once into SBUF,
then reuse it across the inner loops.

```python
# Bad: lhsT tile reloaded N//TILE_N times
for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
        for k in nl.affine_range(K // TILE_K):
            lhsT_tile = nl.load(lhsT[k*TILE_K:(k+1)*TILE_K, m*TILE_M:(m+1)*TILE_M])
            ...

# Good: lhsT tile loaded once per m
for m in nl.affine_range(M // TILE_M):
    lhsT_tiles = [nl.load(lhsT[k*TILE_K:(k+1)*TILE_K, m*TILE_M:(m+1)*TILE_M])
                  for k in nl.affine_range(K // TILE_K)]
    for n in nl.affine_range(N // TILE_N):
        for k in nl.affine_range(K // TILE_K):
            nisa.nc_matmul(..., stationary=lhsT_tiles[k], ...)
```

**Trade-off**: Keeping more tiles in SBUF simultaneously increases memory pressure and can cause spilling (Opt #2).
Choose blocking dimensions so the working set fits in SBUF.

---

### Opt #2: Fuse operations to minimize intermediate data spilling

**Problem**: Sequential operators applied to a large tensor force intermediate results to be written to HBM between
operators, then reloaded for the next operator. This doubles (or more) the data movement.

**Solution**: Fuse operators in a single loop: load a tile, apply all operators in sequence, store the final result.

```python
# Bad: op0 output spills to HBM, reloaded for op1
for tile in kernel_in_hbm:
    tile_sbuf = nl.load(tile)
    op0_out = op0(tile_sbuf)
    nl.store(op0_out_hbm, op0_out)      # spill

for tile in op0_out_hbm:
    tile_sbuf = nl.load(tile)           # reload
    op1_out = op1(tile_sbuf)
    nl.store(kernel_out_hbm, op1_out)

# Good: fused, no intermediate HBM traffic
for tile in kernel_in_hbm:
    tile_sbuf = nl.load(tile)
    op0_out = op0(tile_sbuf)
    op1_out = op1(op0_out)
    nl.store(kernel_out_hbm, op1_out)
```

A prime example is `matmul → softmax → matmul` in self-attention: the intermediate attention score matrix would
overflow SBUF without fusion.

**Gotcha**: Declare buffers *inside* the inner loop to avoid compiler-inserted spills:

```python
# May cause spilling — compiler sees the full [2,4,128,512] buffer live at once
buf = nl.ndarray((2, 4, nl.par_dim(128), 512), buffer=nl.sbuf)
for i0 in nl.affine_range(2):
    for i1 in nl.affine_range(4):
        buf[i0, i1, ...] = nl.load(...)

# Better — each iteration allocates a fresh tile
for i0 in nl.affine_range(2):
    for i1 in nl.affine_range(4):
        buf = nl.ndarray((nl.par_dim(128), 512), buffer=nl.sbuf)
        buf[...] = nl.load(...)
```

---

## 2. Optimizing Compute Efficiency

### Opt #3: Overlap execution across compute engines (pipelining)

**Problem**: When operator A runs on ScalarE and operator B runs on VectorE in sequence, ScalarE sits idle while
VectorE runs and vice versa.

**Solution**: Tile the computation so ScalarE can immediately start processing tile N+1 while VectorE processes
tile N. This creates a software pipeline across engines.

Example: for `X → op0 (ScalarE) → Y → op1 (VectorE) → Z`, split X into tiles so ScalarE produces each Y tile
and VectorE can start op1 on it before ScalarE finishes the full op0.

Choose tile size carefully: too small → pipeline overhead dominates; too large → SBUF pressure and poor overlap.

---

### Opt #4: Overlap data loading with computation

**Problem**: Compute engines sit idle waiting for DMA to finish loading the next tile from HBM.

**Solution**: Use `nl.affine_range` on the load loop so the compiler can schedule DMA for tile N+1 while compute
processes tile N. DMA engines run in parallel with compute engines.

```python
# affine_range signals to compiler that iterations are independent → DMA-compute overlap
for i in nl.affine_range(num_tiles):
    tile = nl.load(input[i * TILE:(i + 1) * TILE])
    result = compute(tile)
    nl.store(output[i * TILE:(i + 1) * TILE], result)
```

If DMA duration cannot be fully hidden behind compute even after maximizing overlap, the DMA itself may be
inefficient — see Opt #9.

---

### Opt #5a: Use sufficiently large tiles in the free dimension

**Problem**: Many back-to-back instructions with tiny free-dimension sizes (e.g., 1 element/partition) dominate
execution because each instruction carries ~100-cycle static overhead regardless of how little work it does.

**Solution**: Increase the free dimension of instruction inputs. NeuronCore compute engines are efficient with
at least 128 elements per partition. Use `nl.affine_range` tiling or reformulate the operator to batch more
elements per instruction.

**Trade-off**: Larger tiles improve engine efficiency but may conflict with engine pipelining (Opt #3) and
increase SBUF pressure (Opt #2).

---

### Opt #5b: Use sufficiently large tiles in the partition dimension

**Problem**: Instructions that span fewer than 128 partitions under-utilize compute engines, since SBUF/PSUM
partitions map 1:1 to parallel vector lanes. Two 64-partition instructions run serially when they could be
combined into a single 128-partition instruction.

**Solution**: "Partition vectorization" — combine multiple narrow operations into one wide operation spanning
all 128 partitions.

```python
# Bad: two 64-partition reductions, run serially
mm_tile0 = nisa.nc_matmul(...)   # partitions 0-63
mm_tile1 = nisa.nc_matmul(...)   # partitions 0-63 (reuses same partitions)
reduce0 = nisa.tensor_reduce(mm_tile0, ...)
reduce1 = nisa.tensor_reduce(mm_tile1, ...)

# Good: one 128-partition reduction — 2x faster
mm_tile = nl.zeros((128, ...), np.float32, buffer=nl.psum)
mm_tile[nl.arange(64)[:, None], ...]    = nisa.nc_matmul(...)  # partitions 0-63
mm_tile[64 + nl.arange(64)[:, None], ...] = nisa.nc_matmul(...)  # partitions 64-127
reduce = nisa.tensor_reduce(mm_tile, ...)  # single instruction over all 128 partitions
```

---

### Opt #6: Combine instructions

**Problem**: Chained element-wise scalar/vector ops (e.g., multiply → add → exp) each require a separate
instruction, each touching the data independently.

**Solution**: Use low-level `nki.isa` APIs that combine multiple operations in a single pipelined instruction.

```python
# Bad: 3 separate instructions, each reads/writes data
scaled   = nl.multiply(data, scale)
shifted  = nl.add(scaled, bias)
exp_out  = nl.exp(shifted)

# Good: 1 instruction — scale, bias, and exp in a single ScalarE pipeline pass
exp_out = nisa.activation(nl.exp, data, bias=bias, scale=scale)
```

This is ~3x faster: the engine pipelines all three operations internally while reading the data only once.

---

### Opt #7 (TensorE): Use fast weight load — prefer short tensors as moving

**Problem**: In a matrix multiplication where one dimension is much smaller than 128 (e.g., matrix-vector
products during token generation), mapping the short tensor to the **stationary** position in `nc_matmul`
leads to slow `LoadStationary` throughput.

**Solution**: Map the short tensor to the **moving** position. TensorE's `LoadStationary` can execute up to
4x faster than `MultiplyMoving` for the same data volume ("Fast LoadStationary"), so the moving instruction
becomes the bottleneck less often when the short tensor is there.

Concretely: to compute `A × B` where A is short, instead of `nc_matmul(A.T, B)`, call `nc_matmul(B, A.T)`,
which computes `(B.T × A.T).T = A × B`. The output is transposed, so account for this downstream.

---

### Opt #8 (TensorE): Reduce tensor transposes

**Problem**: PF-transposes (swapping partition ↔ free dimension) consume TensorE cycles doing no useful
computation. They arise when the layout produced by one instruction doesn't match what the next instruction
expects.

**Two types and their fixes:**

**IO transposes** (input/output tensor layout mismatch):
- Choose the IO tensor layout in HBM to match what the first/last NKI compute API expects.
- Avoid `nl.load_transpose2d` in memory-bound kernels (low DMA bandwidth); prefer `nl.load` + `nisa.nc_transpose`.

**Intermediate transposes** (layout mismatch between ops in the kernel):

1. *Swap stationary/moving tensors*: e.g., in `linear → layernorm`, mapping the weight to moving and feature
   map to stationary can produce output in the layout `nisa.bn_stats` expects, eliminating the transpose.

2. *Use an alternative engine*: e.g., `RMSNorm` summation can run on either VectorE (free-dim reduce) or
   TensorE (partition-dim reduce via `nisa.nc_matmul`). Choose whichever avoids a transpose given the
   surrounding operator layout.

---

## 3. Optimizing Data Movement Efficiency

### Opt #9: Perform sufficiently large DMA transfers

**Problem**: Many small DMA transfers have high per-transfer overhead. Transfer sizes below ~32 KiB on
Trainium/Inferentia2 achieve well below peak DMA bandwidth.

**Solution**: Maximize partition and free dimension sizes in `nl.load` / `nl.store`. Aim for each transfer
to move at least 32 KiB.

```python
# 128 partitions × 1024 elements × 4B = 512 KiB — good bandwidth efficiency
i_p, i_f = nl.mgrid[0:128, 0:1024]
data_tile = nl.load(in_tensor[i_p, i_f])
```

When blocking large matrices, ensure block sizes are large enough to keep individual DMA transfers above the
efficiency threshold. Overly fine-grained tiling (e.g., TILES_IN_BLOCK_K = 1) produces many small transfers.

---

### Opt #10: Minimize DMA transposes (`nl.load_transpose2d`)

**Problem**: `nl.load_transpose2d` performs an on-the-fly transpose in the DMA engine at significantly lower
bandwidth than a regular `nl.load`.

**Solution**:
- For compute-bound kernels: acceptable if transposes are unavoidable.
- For memory-bound kernels: replace with `nl.load()` + `nisa.nc_transpose()` (TensorE transpose after load),
  or restructure IO tensor layout so no transpose is needed (see Opt #8).

---

## 4. Practical Patterns (from Fused Mamba)

These learnings generalize beyond Mamba but the kernel is a concrete illustration of several optimizations working together.

### Opt #11: Loop reordering to prioritize reuse of the largest tensors

**Problem**: When two loop dimensions have conflicting reuse — e.g., tensor `delta/u` is reused across `state_size`
iterations while `B/C` are reused across `channels` iterations — the inner loop determines what gets reloaded.
If `channels` is the inner loop, `delta/u` (large: `[channels, seq_len]`) get reloaded `state_size` times.

**Solution**: Make the dimension with the larger reuse benefit the *outer* loop and hoist the loads above the inner loop.
When `channels >> state_size`, move `channels` to the outer loop so `delta/u` load once per channel tile:

```python
# Bad: delta/u reloaded state_size times per channel tile
for i_state in nl.affine_range(state_size):
    for i_ch in nl.affine_range(n_channel_tiles):
        delta_i = nl.load(delta[..., ch_start:ch_end, :])   # reloaded every state
        u_i     = nl.load(u[..., ch_start:ch_end, :])

# Good: delta/u loaded once per channel tile, reused across all states
for i_ch in nl.affine_range(n_channel_tiles):
    delta_i = nl.load(delta[..., ch_start:ch_end, :])       # loaded once
    u_i     = nl.load(u[..., ch_start:ch_end, :])
    for i_state in nl.affine_range(state_size):
        # use delta_i and u_i
```

As a bonus, loop reordering often reveals loop fusion opportunities: two sibling loops over the same dimension
can be merged, reducing intermediate buffer sizes by that dimension factor.

---

### Opt #12: Use `nisa.tensor_tensor_scan` for associative scans

**Problem**: A naive associative scan over `seq_len` produces `seq_len` many instructions each operating on
a single element per partition. With chained data dependencies, each instruction stalls until the previous
completes. The instruction overhead (~100 cycles/instruction) dominates.

```python
# Bad: seq_len instructions, each with 1 element/partition, serial dependency chain
for i in nl.sequential_range(seq_len - 1):
    scan[..., i+1] = nisa.tensor_scalar(
        deltaA[..., i+1], op0=nl.multiply, operand0=scan[..., i],
        op1=nl.add, operand1=deltaBu[..., i+1])
```

**Solution**: Use `nisa.tensor_tensor_scan`, a single VectorE instruction that performs the entire scan
internally, caching intermediate state without going through SBUF:

```python
# Good: single instruction, all seq_len positions processed internally by VectorE
scan_i = nisa.tensor_tensor_scan(deltaA_i, deltaBu_i, initial=0,
                                  op0=np.multiply, op1=np.add)
```

When `seq_len` must be tiled, pass the last element of the previous tile as `initial` for the next tile,
and use `static_range` (not `affine_range`) since the loop carries a dependency:

```python
scan_init = nl.zeros((channel_psize, 1), ...)
for i_seq in static_range(seq_len // seq_len_tile):
    scan_i = nisa.tensor_tensor_scan(deltaA[..., seq_slice], deltaBu[..., seq_slice],
                                      initial=scan_init, op0=np.multiply, op1=np.add)
    scan_init = scan_i[..., seq_len_tile - 1]  # carry last state to next tile
```

Use `static_range` (not `affine_range`) for any loop with a carried dependency — `affine_range` signals
independent iterations and the compiler may reorder or vectorize them, breaking the dependency.

---

### Opt #13: Tile the free dimension proactively to prevent SBUF spilling

**Problem**: Increasing a free dimension (e.g., `seq_len`) scales the size of *every* tensor that has that
dimension. Once the working set exceeds SBUF capacity, the compiler spills tensors to HBM, introducing
unscheduled DMA traffic that blocks compute engines. SBUF usage measured at 50% can still trigger spilling
due to fragmentation.

**Solution**: Add a tile loop over the large free dimension *before* spilling becomes a problem. Choose a
tile size where the previous implementation achieved high compute utilization (e.g., `seq_len_tile = 512`
worked well → use that as the tile size when `seq_len` grows to 8K).

Place the tile loop at the right level: inserting it as the innermost loop reduces SBUF usage the least
(large tensors stay live across the full loop); moving it toward the outermost loop tiles more tensors and
reduces SBUF pressure more aggressively. Start with the innermost position and move outward if spilling persists.

---

### Opt #14: Maintain consistent partition layout across an operator chain

**Problem**: Choosing different partition/free dimension assignments for the same logical axis in consecutive
operators forces transposes between them. Transposes consume TensorE cycles and can cascade through the chain.

**Solution**: Fix a layout convention at the start and verify each operator in the chain is compatible with
it before writing the kernel. In Mamba: `channels` is always the partition dimension. This choice:
- Enables `nisa.tensor_scalar` (free-dim broadcast of `A_i`) instead of `nisa.tensor_tensor` + broadcast
- Allows `nisa.activation` (Step 1 multiply + Step 2 exp in one ScalarE instruction) because both steps
  share the same layout
- Avoids all intermediate transposes across 6 operator steps

When two operators require conflicting layouts, pick the layout for the operator that runs most often or is
on the performance-critical engine, and accept one transpose at the boundary.

---

## Quick Reference

| Goal | Key Opts |
|---|---|
| Reduce HBM reads | #1 (hoist loads), #2 (fuse ops) |
| Keep compute engines busy | #3 (pipeline engines), #4 (overlap DMA+compute) |
| Improve instruction efficiency | #5a/b (tile sizes), #6 (combine instructions) |
| TensorE-specific | #7 (fast weight load), #8 (reduce transposes) |
| Improve DMA bandwidth | #9 (large transfers), #10 (avoid DMA transposes) |
| Loop structure | #11 (loop reorder for reuse), #13 (tile free dim) |
| Scan / sequential ops | #12 (tensor_tensor_scan, static_range) |
| Layout consistency | #14 (uniform partition dim across op chain) |
