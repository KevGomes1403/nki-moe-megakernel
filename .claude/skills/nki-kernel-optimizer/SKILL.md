---
name: nki-kernel-optimizer
description: Generates, reviews, and optimizes AWS Neuron NKI kernels with correct nki.lang/nki.isa syntax and a performance-first approach. Use when the user asks to "write an NKI kernel", "optimize/improve an NKI kernel", "fix NKI syntax errors", "reduce latency on Trainium/Inferentia2", "choose tiling/scheduling for NKI", or "benchmark/profile an NKI kernel"
---

# NKI Kernel Optimizer

## Purpose
This skill helps you:
1) Generate NKI kernels from a math spec / PyTorch block.
2) Fix syntax/semantic issues in existing NKI kernels.
3) Improve performance by choosing good tiling, memory placement, and loop scheduling.
4) Benchmark and profile using `nki.benchmark` and traces (NEFF/NTFF).

## When to use (activation cues)
Use this skill when the user:
- Provides NKI kernel code and asks to fix errors or improve latency/throughput.
- Asks for an NKI implementation of an op (attention pieces, layernorm/rmsnorm, softmax, GEMM variants, elementwise+reduction).
- Mentions Trainium/Inferentia2/Trainium2/Trainium3, NeuronCores, SBUF/PSUM/HBM.
- Wants a benchmarking harness with `nki.benchmark` or profiling traces.

## Operating principles
- **Correctness first, then speed.** Produce a correctness-checkable kernel (separate from benchmark) before tuning.
- **Make data movement explicit.** Prefer predictable allocations and explicit loads/stores; avoid hidden HBM roundtrips.
- **Tile to fit on-chip.** Keep hot working sets in SBUF/PSUM; stream from HBM only when needed.
- **Use the right engine.** Prefer tensor engine instructions for matmul, vector engine for reductions/elementwise, scalar engine for nonlinearities.
- **Iterate with measurements.** Every optimization should be tied to a benchmark delta.

## Standard workflow

### Step 0 — Intake (ask only what you need)
Collect:
- Target device generation: inf2 / trn1 (NeuronDevice v2), trn2, trn3.
- Kernel category: elementwise, reduction, fused, matmul, attention primitive.
- Shapes/dtypes/layout constraints; any alignment or tile-size requirements you already know.
- Performance objective: latency p50, p99, or throughput; acceptable numerical error.

### Step 1 — Establish a minimal spec
Write a compact, testable spec:
- Inputs/outputs with shapes & dtype.
- Exact math (including scaling, masking, eps, etc.).
- Expected tolerances (rtol/atol).
- Any fusion boundaries (what must be inside the kernel vs outside).

If the user provides PyTorch, derive the math spec and identify which parts can be fused safely.

### Step 2 — Generate a baseline NKI kernel (clear + correct)
Use this baseline structure unless the op demands otherwise:
1) Allocate outputs explicitly using `nl.ndarray(..., buffer=...)`.
2) Load inputs via `nl.load(...)`.
3) Compute using `nl` ops where possible; drop to `nki.isa` for performance-critical primitives.
4) Store outputs via `nl.store(...)`.
5) Return the output tensor(s).

Baseline quality gates:
- No missing stores.
- Output buffers are correct (HBM vs SBUF vs PSUM where appropriate).
- Shapes are consistent at every step.
- Loops: default to `nl.sequential_range()` unless you’re sure iterations are independent.

### Step 3 — Correctness harness (separate from benchmarking)
Produce a separate Python test that:
- Generates deterministic inputs (seeded).
- Runs a reference implementation in NumPy (or framework CPU) and compares.
- Uses `np.testing.assert_allclose` with chosen rtol/atol and reports max error.

Do NOT rely on `nki.benchmark` output for correctness, since the benchmark path does not use the actual runtime inputs for correctness checks.

### Step 4 — Performance pass (structured optimization)
Apply optimizations in this order, measuring after each:
1) **HBM traffic reduction**
   - Fuse pointwise ops around reductions/matmul when safe.
   - Use SBUF tiles; keep intermediate results on-chip.
2) **Tiling & layout**
   - Choose partition dimension (SBUF first dimension) and tile sizes that maximize reuse.
   - Avoid strided/irregular accesses in free dimension.
3) **Loop scheduling**
   - Switch eligible loops to `nl.affine_range()` to unlock unrolling/pipelining.
   - Keep dependency-carrying loops sequential.
4) **Instruction selection**
   - Use `nki.isa.nc_matmul` / `nki.isa.tensor_tensor` when they match hardware-friendly tile shapes.
   - Prefer combined ops if available (e.g., fused vector ops) to reduce instruction count.
5) **Allocation & lifetime**
   - Reuse SBUF regions where safe.
   - Avoid large transient allocations; keep SKILL output readable.

Always explain:
- What changed (diff-level summary).
- Why it should help (bandwidth vs compute vs scheduling).
- What to measure (which percentile / which shapes).

### Step 5 — Benchmarking with `nki.benchmark`
Provide a benchmark wrapper using the decorator:
- Use `warmup` and `iters`.
- Optionally emit NEFF and trace (NTFF) to enable profiling.
- Record p50 and p99 and compare to baseline.

See `references/benchmarking-recipes.md`.

### Step 6 — Profiling & diagnosis
When traces are available:
- Attribute time to engines (tensor/vector/scalar/gpsimd) and DMA.
- Identify whether you are compute-bound or memory-bound.
- Recommend the next optimization step accordingly.

## Output format expectations
When producing code, always include:
1) The kernel function.
2) A correctness test snippet.
3) A benchmark snippet (if requested), including how to read p50/p99.
4) A short “tuning notes” section with what to try next.

## Common pitfalls (fix automatically)
- Returning an SBUF tensor when caller expects HBM output.
- Allocating outputs in HBM but doing all math on HBM (missing SBUF tiling).
- Using `nl.affine_range` on dependency-carrying loops (causes incorrect results or compiler pessimization).
- Incorrect dtype promotion or missing casts.
- Misunderstanding compile-time vs runtime behavior (e.g., print statements).

See `references/common-pitfalls.md`.

## Examples (user prompts that should trigger this skill)
- "Write an NKI kernel for RMSNorm on Trainium2."
- "Here’s my NKI kernel; it compiles but is slow. Improve tiling and scheduling."
- "Fix this NKI syntax error around nl.ndarray / nl.store."
- "Create a microbenchmark with nki.benchmark and emit a trace."

## Troubleshooting
If the kernel fails to compile:
1) Reduce to baseline (no fusion, sequential loops).
2) Validate shapes/dtypes at every intermediate.
3) Move advanced details into `nki.isa` only where necessary.
4) If a specific ISA op fails, check tile/layout constraints and adjust.

If the benchmark is noisy:
- Increase `iters`, keep inputs stable, and record multiple runs.

## Reference docs bundled with this skill
- `references/nki-syntax-quickref.md`
- `references/performance-playbook.md`
- `references/benchmarking-recipes.md`
- `references/common-pitfalls.md`
- `references/templates.md`
