---
name: nki-kernel-optimizer
description: |
  Generates, reviews, and optimizes AWS Neuron NKI (Neuron Kernel Interface) kernels.
  Trigger when the user asks to write, fix, optimize, benchmark, or profile an NKI kernel,
  mentions Trainium/Inferentia, SBUF/PSUM/HBM, nki.lang, nki.isa, or neuron-profile.
---

# NKI Kernel Optimizer Skill

## Environment Setup (ALWAYS do this before running any kernel)

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
```

**Hardware**: AWS Trainium 2/3
NeuronCores are exclusive: only one Python process can hold them at a time. Run kernels sequentially.

---

## Reference Docs

| File | Contents |
|------|----------|
| `references/nki-syntax-quickref.md` | nki.lang / nki.isa cheatsheet, common patterns |
| `references/performance-playbook.md` | Structured step-by-step optimization workflow |
| `references/benchmarking-api.md` | Benchmarking harness API, env setup, `BenchmarkResult` fields, metric interpretation |
| `references/common-pitfalls.md` | Compiler errors, dtype gotchas, scheduling mistakes |
| `references/logical-neuron-cores.md` | Combining two physical NeuronCores into one logical NeuronCore |
| `references/trn2-architecture.md` | NeuronCore-v3 hardware specs: engines, memory, DMA transpose, DGE, FP8 double-perf mode |
| `references/trn3-architecture.md` | NeuronCore-v4 hardware specs: chip-level perf, MXFP8/MXFP4, 32 MiB SBUF, near-memory accumulation, GPSIMD, trn2→trn3 comparison |
| `references/fp8-mxfp-quantization.md` | FP8 row/static/MX quantization patterns with exact NKI calls, MXFP scale layout, `double_row` matmul, RMSNorm fusion |
| `references/sbuf-allocation.md` | All-or-nothing auto vs manual SBUF allocation rule; `NCC_EGCA111` cause and fix; `create_auto_alloc_manager` pattern |
| `references/dma-patterns.md` | `dma_transpose` vs `nc_transpose`, `.ap()` stride patterns, scalar broadcast, `tp_broadcast`, weight ring buffer, PSUM bank interleaving |

Read the relevant reference(s) before generating or modifying any kernel code.

---

## Intake — Collect Before Starting

Ask only what you need:

1. **Device generation**: trn1/inf2 (NeuronCore-v2), trn2, or trn3. Default to **trn2** unless specified.
2. **Op category**: elementwise, reduction, fused, matmul/GEMM, attention primitive, MoE, custom.
3. **Shapes & dtypes**: all input/output tensor shapes, dtypes, any alignment constraints.
4. **Existing code**: the kernel to optimize (required for optimization; not required for new kernels).
5. **PyTorch reference**: a pure-PyTorch function that computes the correct output — **required**. If the user does not provide one, ask for it before proceeding. All correctness checks run against this reference, never against the original NKI kernel.

Profiling data from a prior round (if any) will be provided by the orchestrator — do not ask the user for it.

---

## Optimization Loop (Iterative Rounds)

This skill operates as an **iterative orchestrator/subagent loop**. The orchestrator plans and synthesizes; subagents implement and benchmark. Repeat until performance is satisfactory.

```
REPEAT each round:
  1. Orchestrator: analyze current kernel + profiling summaries from prior round
                   → generate N distinct optimization plans
  2. Dispatch N subagents sequentially (hardware constraint: one at a time)
     Each subagent:
       a. Implement the assigned plan
       b. Loop until assert_allclose passes (correctness loop)
       c. Benchmark the passing kernel using the benchmarking harness
       d. Return a structured summary to the orchestrator
  3. Orchestrator: collect N summaries → synthesize findings
                   → decide: continue with a new round or stop
UNTIL performance target is met or no further plans are promising
```

---

## Step 1 — Math Spec

Write a compact, unambiguous spec:
- Inputs/outputs: shape, dtype, memory location (HBM vs SBUF).
- Exact math (scaling, masking, epsilon, etc.).
- Fusion boundaries: what must be fused vs. what can be separate.
- Tolerances: rtol/atol for correctness check.

If the user supplies PyTorch code, derive the spec and annotate which parts can be safely fused.

---

## Step 2 — Profiling Analysis (Orchestrator)

**First round**: characterize the baseline kernel by reading `references/benchmarking-api.md` and running the harness on the unmodified kernel before planning. Extract and record:
- `device_time_us`, `tensor_engine_pct`, `dma_active_pct`, `spill_bytes`
- `mfu_estimated_percent`, `hbm_read_bytes`, `hbm_write_bytes`

**Subsequent rounds**: use the profiling summaries returned by the previous round's subagents.

Classify the bottleneck:
- **Compute-bound**: `tensor_engine_pct` ≥ 90% — reduce FLOPs or increase arithmetic intensity.
- **Memory-bound**: `dma_active_pct` high, engines low — reduce HBM traffic, improve tiling.
- **Spill-bound**: `spill_bytes` > 0 — SBUF overflow, reduce tile size or hoist allocations.
- **Stall-bound**: all engines low, DMA low — scheduling or dependency issue.

Document which metrics support the characterization. This drives the plans in Step 3.

---

## Step 3 — Optimization Planning (Orchestrator)

Generate **exactly N optimization plans** (default N=3). Each plan must:

- Be **independent** — implementable without relying on the other plans.
- Be **specific** — reference exact loop variables, tensor names, tile sizes, API calls, and profiling metrics that motivate the change. Vague guidance ("tile better") is not acceptable.
- Target a **distinct bottleneck or axis of improvement**.
- State the **hypothesis**: which metric should improve and in which direction.

Present each plan in this format:

```
### Plan A — <Short Title>

**Bottleneck targeted**: <specific metric from profiling data>
**Root cause**: <why this metric is high / low>
**Change**: <exact code-level change — loop restructure, tile size, API swap, fusion boundary, etc.>
**Expected effect**: <which profiler metric improves, and why>
**Correctness risk**: <any numerical or shape concern to watch for>
**Verification**: assert_allclose(rtol=X, atol=Y) against the PyTorch reference — state the tolerances and any dtype considerations
```

Do not begin any implementation until all plans are written.

---

## Step 4 — Sequential Subagent Dispatch

Dispatch subagents **one at a time** (hardware constraint: NeuronCores are exclusive).

### Subagent Charter

> You are implementing **Plan [A/B/C]** exactly as specified. Read `references/benchmarking-api.md` before starting.
>
> **Goals**:
> 1. **Faithfulness**: implement the plan as written. Do not introduce unplanned changes.
> 2. **Correctness**: the kernel must be numerically equivalent to the original. Do not exit the correctness loop until `assert_allclose` passes.
> 3. **Benchmarking**: after correctness passes, benchmark the kernel using `wrap_benchmark` or `nki_benchmark`. Copy `scripts/benchmark.py` from the skill root into the current workspace before importing (see `references/benchmarking-api.md`).
> 4. **Reporting**: return a structured summary to the orchestrator (format below).

### Subagent Internal Loop

```
REPEAT:
  1. Implement the plan change on the current kernel code.
  2. Run the correctness harness (see Step 5).
  3. IF assert_allclose fails:
       - Diagnose the numerical discrepancy (shape mismatch, dtype cast, accumulation order, etc.)
       - Fix only what is needed to restore correctness; do not expand scope.
       - Return to step 1.
  4. UNTIL assert_allclose passes.

THEN:
  5. Benchmark the passing kernel (see Step 5, benchmarking section).
  6. Output the structured summary (see below).
```

A subagent **may not exit** while `assert_allclose` is failing or before benchmarking is complete.

### Subagent Output Format

```
### Plan [A/B/C] — <Title>

**Correctness**: PASS  max_diff=X.XXe-XX

**Benchmark results**:
- device_time_us:       X.XX
- tensor_engine_pct:    X.X%
- dma_active_pct:       X.X%
- spill_bytes:          X
- mfu_estimated_pct:    X.X%
- hbm_read_KiB:         X.X
- hbm_write_KiB:        X.X

**Implementation note**: <what was changed, any deviation from the plan (must be flagged and justified), residual risk>

**Remaining bottleneck**: <which metric is still limiting and why>
```

Also include the complete, runnable kernel code with inline comments.

---

## Step 5 — Correctness Harness and Benchmarking

### Correctness

Always compare against the **PyTorch reference function provided at intake** — never against the original NKI kernel. Use the same seed for all subagents across all rounds.

```python
import numpy as np
import torch

rng = np.random.default_rng(42)
x = rng.random(shape).astype(np.float32)  # replace with actual shape

# pytorch_reference is the function provided by the user at intake
ref = pytorch_reference(torch.tensor(x)).numpy()
result = optimized_kernel(torch.tensor(x).to("xla")).cpu().numpy()

np.testing.assert_allclose(result, ref, rtol=1e-3, atol=1e-3)
print(f"max_diff={np.abs(result - ref).max():.2e}  PASS")
```

### Benchmarking (after correctness passes)

Read `references/benchmarking-api.md` for full details. Minimal pattern:

```python
import os, sys

# Set BEFORE any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out"
)
# Copy scripts/benchmark.py from the skill root into this directory first:
# cp <skill-root>/scripts/benchmark.py .
from benchmark import wrap_benchmark

my_kernel = wrap_benchmark(my_kernel, warmup=5, iters=50)
my_kernel(*inputs)

r = my_kernel.last_result
# extract r.device_time_us, r.tensor_engine_pct, r.dma_active_pct,
# r.spill_bytes, r.prof['mfu_estimated_percent'], etc.
```

---

## Step 6 — Orchestrator Synthesis

After collecting all subagent summaries for the round, produce:

```
## Round [N] Synthesis

### Baseline (Round 1 only)
- device_time_us: X.XX — bottleneck: <classification>

### Results

| Plan | device_time_us | tensor_eng% | dma% | spill | mfu% | vs baseline |
|------|---------------|-------------|------|-------|------|-------------|
| A    | ...           | ...         | ...  | ...   | ...  | ...         |
| B    | ...           | ...         | ...  | ...   | ...  | ...         |
| C    | ...           | ...         | ...  | ...   | ...  | ...         |

### Analysis
- Which plans improved the target metric? Did the hypothesis hold?
- Which plans had unexpected effects (positive or negative)?
- What is the new bottleneck after the best plan(s)?

### Next Round Decision
- Continue: new bottleneck identified, plans drafted for Round [N+1]
- Stop: performance target met / no further plans are promising
```

If continuing, carry the best kernel variant from this round as the baseline for the next round and go back to Step 2.

---

## Code Documentation Standards

All kernel code produced by this skill must meet these standards:

- **Block comments** before each logical section (load, compute, store).
- **Inline comments** on any non-obvious indexing, tile size choice, or API selection — explain the *hardware reason*.
- **Named constants** for tile sizes and magic numbers — no bare literals without a comment.
- **Dtype annotations** on all `nl.ndarray` allocations.

---

## Quick Rules (apply automatically)

- Always `source` the venv and set `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` before running.
- Set all `NEURON_RT_INSPECT_*` env vars **before any neuron/torch_xla import** — setting them after is a silent no-op.
- `expert_affinities` and other accumulation-path tensors must be `float32` (not `bf16`) for POST_SCALE MoE patterns.
- `nl.par_dim(128)` is the partition dimension size on trn2; tile your partition dimension accordingly.
- Use `nl.affine_range` for DMA loads that are independent; keep accumulation loops sequential.
- When debugging compiler errors: reduce to minimal baseline (no fusion, no affine_range), then re-add optimizations one at a time.
- `PSUM` buffers must be copied to SBUF before storing to HBM.
- `.ap()` works on HBM and SBUF/PSUM tensors; see `nki-syntax-quickref.md` for restrictions and the DGE `scalar_offset` address pitfall.
- Subagents run sequentially — never dispatch two subagents in parallel.
- Every SBUF/PSUM tensor in a kernel must use either all-automatic or all-manual allocation — never both. Mixing causes `NCC_EGCA111`. The named tensor in the error is a symptom (longest live range), not the cause. Fix: use `create_auto_alloc_manager()` from nkilib.
- Prefer `dma_transpose` over `nc_transpose` whenever the source is in HBM — it's free (DMA engine) and avoids an intermediate SBUF allocation. `nc_transpose` is for SBUF→PSUM only.
- For FP8 kernels, use `perf_mode=nisa.matmul_perf_mode.double_row` on `nc_matmul` when both inputs are FP8 — doubles effective throughput.
- MXFP scale layout in SBUF is sparse: valid rows are only at `[0-3, 32-35, 64-67, 96-99]` per 128-partition tile. Load with the 4-rows-per-quadrant DMA pattern; never load densely into SBUF.
- Hoist all quantization scale loads (`nisa.dma_copy` for scales) to outside all loops. Loading inside the inner tile loop causes per-tile HBM traffic that destroys bandwidth.
- FP8 quantization requires `[T, 1]` scale shape (not `[T]`) for `nisa.activation` broadcasting, and a two-sided clamp to `[-240, 240]` before the dtype cast.
- Use `stream_shuffle_broadcast` for scalars that must reach all partitions — load once to SBUF `[0,0]`, then broadcast. Avoids ~128 serialized `.ap()` calls.
- PSUM bank interleaving: assign accumulator tiles via `address=(0, (tile_idx % 4) * PSUM_BANK_SIZE)` to eliminate read-after-write stalls when multiple tiles are in flight.
