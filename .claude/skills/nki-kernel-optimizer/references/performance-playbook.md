# Performance Playbook for NKI Kernels

This doc is deliberately operational. It’s meant to guide iteration.

## 1) Decide what you are optimizing
- Latency (p50/p99) for a fixed shape
- Throughput (tokens/s, elements/s) across a batch of shapes
Write down the exact metric and the exact shapes up front.

## 2) Classify the kernel
- Pure elementwise (bandwidth-bound)
- Reduction (bandwidth + vector engine + scheduling)
- Matmul (tensor engine; tiling dominates)
- Fused attention primitive (matmul + softmax + elementwise)

## 3) First-order wins (do these before “clever” tricks)
### Reduce HBM traffic
- Fuse simple pointwise ops around reductions/matmuls.
- Keep intermediates in SBUF; avoid store+reload cycles.

### Tile to fit on-chip
- Choose a tile that fits SBUF with room for double-buffering if you pipeline.
- Prefer contiguous access in the free dimension.

### Pick correct buffers
- HBM for inputs/outputs, SBUF for working set.
- PSUM for tensor-engine accumulation.

## 4) Scheduling tactics
- Start sequential; move to affine only after confirming independence.
- Pipeline: overlap DMA (HBM->SBUF) with compute where possible.
- Avoid large live ranges; reuse scratch buffers.

## 5) When to use nki.isa
Use ISA when:
- You need tensor-engine matmul or a hardware-optimized primitive.
- You need explicit barriers/synchronization or advanced data movement.
Stay in `nl` when:
- It’s straightforward elementwise math and the compiler maps well.

## 6) Read the profile like a compiler engineer
- High DMA time: reduce HBM traffic, increase reuse, fuse.
- Tensor engine underutilized in matmul: fix tiling/layout, increase compute per tile.
- Vector engine saturated: reduce reduction passes, fuse elementwise, adjust partitioning.

## 7) Optimization checklist (copy/paste into PRs)
- [ ] Baseline kernel + correctness test exists
- [ ] Benchmark harness reports p50/p99
- [ ] Each change explains: what, why, measured delta
- [ ] Memory placement is explicit and justified
- [ ] Loop types (sequential vs affine) are intentional
