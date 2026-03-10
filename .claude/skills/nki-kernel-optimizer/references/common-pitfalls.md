# Common Pitfalls and Fixes

## 1) Output is “garbage” after benchmarking
Cause: `nki.benchmark` runs a NEFF in a tight loop and does not use the provided runtime inputs for correctness.
Fix: Use a separate correctness harness; treat benchmark output as undefined.

## 2) Slow kernel with lots of HBM traffic
Symptoms: Profiling shows DMA dominates.
Fix:
- Tile and move hot data into SBUF
- Fuse pointwise ops around reductions/matmuls
- Avoid storing intermediates to HBM

## 3) Incorrect results after switching to affine loops
Cause: loop iterations had dependencies; affine allowed reordering/unrolling.
Fix: revert to `nl.sequential_range` for dependency-carrying loops, or rewrite to remove dependencies.

## 4) Unexpected dtype behavior
Cause: implicit casts or mixed dtypes.
Fix: make casts explicit; enforce dtype in allocations.

## 5) Compiler errors on ISA ops
Cause: tile/layout/dtype constraints not satisfied.
Fix: adjust tile sizes; reshape/relayout; ensure buffers match instruction requirements.
