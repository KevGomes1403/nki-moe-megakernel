# Templates

## Template A — “kernel + correctness + benchmark” bundle

Use this template for most deliverables.

### 1) Kernel
- `@nki.jit` kernel function
- explicit buffers for I/O and working tiles
- explicit load/store

### 2) Correctness
- deterministic input generation
- reference implementation
- `assert_allclose` with chosen tolerances

### 3) Benchmark
- `@nki.benchmark` wrapper (or separate function)
- report p50/p99 and compile opts

### 4) Tuning notes
- next experiments
- hypotheses
- what to measure
