---
name: nki-kernel-optimizer-trn3
description: |
  Generates, reviews, and optimizes AWS Neuron NKI kernels for Trainium3 (trn3) / NeuronCore-v4.
  Specializes in MXFP8/MXFP4 quantization (nc_matmul_mx, quantize_mx) and NeuronCore-v4 features.
  DEFAULT skill for all NKI kernel work in this repo (hardware is trn3).
  Trigger when the user asks to write, fix, optimize, benchmark, or profile any NKI kernel,
  mentions Trainium/SBUF/PSUM/HBM/nki.lang/nki.isa/neuron-profile, OR mentions trn3,
  NeuronCore-v4, MXFP8, MXFP4, microscaling, nc_matmul_mx, or quantize_mx.
  Prefer this over nki-kernel-optimizer unless the user explicitly targets trn2.
---

# NKI Kernel Optimizer — Trainium3 (trn3)

## Environment Setup (ALWAYS do this before running any kernel)

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn3
```

**Hardware**: trn3 instance — 8 NeuronCores (NeuronCore-v4 / gen4).
NeuronCores are exclusive: only one Python process can hold them at a time. Run kernels sequentially.

---

## Reference Docs

| File | Contents |
|------|----------|
| `references/trn3-architecture.md` | NeuronCore-v4 hardware specs: engines, memory, DMA, new trn3-only features |
| `references/mxfp-quantization.md` | MXFP8/MXFP4 quantization workflow, `nc_matmul_mx`, `quantize_mx` APIs, layout requirements |
| `references/performance-playbook.md` | Structured optimization workflow, trn3-specific opts including MXFP quantization path |
| `references/benchmarking-api.md` | Benchmarking harness API, env setup, `BenchmarkResult` fields, metric interpretation |
| `references/nki-syntax-quickref.md` | nki.lang / nki.isa cheatsheet with trn3-specific APIs |
| `references/common-pitfalls.md` | Compiler errors, dtype gotchas, MXFP layout mistakes, scheduling issues |

Read the relevant reference(s) before generating or modifying any kernel code.

---

## Competition Constraint (always applies in this repo)

**Batch size = 1, short input sequences (TKG / single-token inference).**

This means:
- The moving tensor for all matmuls is `[128P, T]` where **T = 1** (one token column).
- `nc_matmul_mx` (MXFP_x4) requires moving tile `[128P, 512F]` — with T=1 you would waste 511/512 of compute bandwidth. **Do NOT propose `nc_matmul_mx` for the gate/up/down projections in TKG kernels.**
- All kernels are **DMA-bound**, not compute-bound. Weight-loading dominates. Optimize for HBM traffic reduction and DMA/VectorE overlap, not FLOPs.
- Quantize-on-device (`quantize_mx` activations) gives no benefit here because the moving tile is too small to amortize the quantization overhead.
- The correct matmul path for TKG is: **fp8 stationary (offline pre-quantized) + bf16 moving + `nc_matmul`** — not `nc_matmul_mx`.

---

## Intake — Collect Before Starting

Ask only what you need:

1. **Op category**: elementwise, reduction, fused, matmul/GEMM, attention primitive, MoE, custom.
2. **Shapes & dtypes**: all input/output tensor shapes, dtypes, any alignment constraints.
3. **Quantization target**: are weights pre-quantized (offline) or quantized on-device? MXFP8 or MXFP4?
4. **Existing code**: the kernel to optimize (required for optimization; not required for new kernels).

Profiling data from a prior round (if any) will be provided by the orchestrator — do not ask the user for it.

---

## Quantization Strategy Decision (resolve before planning)

Before writing or optimizing any matmul kernel on trn3, decide the quantization path:

| Scenario | Strategy |
|----------|----------|
| Weights are static (model weights) | **Offline**: pre-quantize with `quantize_mx` at model load time, store MXFP8+scales in HBM |
| Activations are dynamic (per-token) | **On-device**: run `quantize_mx` inside the kernel per tile before `nc_matmul_mx` |
| Both weights and activations change | **Fully on-device**: quantize both stationary and moving inside the kernel |
| BF16 baseline kernel, switching to MXFP | **Migration**: add quantize step before each `nc_matmul`, verify correctness with relaxed tolerances |

Document the chosen strategy in the math spec. The strategy determines tile sizing, SBUF budget, and the load/quantize/matmul pipeline structure.

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
- Exact math (scaling, masking, epsilon, quantization mode, etc.).
- Fusion boundaries: what must be fused vs. what can be separate.
- Quantization strategy (from decision table above).
- Tolerances: rtol/atol for correctness check. **Note**: MXFP quantization introduces ~1e-2 error; use relaxed tolerances (rtol=5e-2, atol=5e-2) for quantized kernels.

If the user supplies PyTorch code, derive the spec and annotate which parts can be safely fused.

---

## Step 2 — Profiling Analysis (Orchestrator)

**First round**: characterize the baseline kernel by reading `references/benchmarking-api.md` and running the harness on the unmodified kernel before planning. Extract and record:
- `device_time_us`, `tensor_engine_pct`, `dma_active_pct`, `spill_bytes`
- `mfu_estimated_percent`, `hbm_read_bytes`, `hbm_write_bytes`

**Subsequent rounds**: use the profiling summaries returned by the previous round's subagents.

Classify the bottleneck:
- **Compute-bound**: `tensor_engine_pct` ≥ 90% — reduce FLOPs or switch to MXFP quantized path (4x TensorE throughput).
- **Memory-bound**: `dma_active_pct` high, engines low — reduce HBM traffic, use offline quantization to halve weight load bytes.
- **Quantize-bound**: VectorE high while TensorE is idle — `quantize_mx` is a bottleneck, pipeline or move to offline quantization.
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
**Change**: <exact code-level change — quantization strategy, loop restructure, tile size, API swap, fusion boundary, etc.>
**Expected effect**: <which profiler metric improves, and why>
**Correctness risk**: <any numerical or shape concern to watch for; note relaxed tolerances for MXFP>
**Verification**: assert_allclose(rtol=X, atol=Y) — explain what to compare against
```

Do not begin any implementation until all plans are written.

---

## Step 4 — Sequential Subagent Dispatch

Dispatch subagents **one at a time** (hardware constraint: NeuronCores are exclusive).

### Subagent Charter

> You are implementing **Plan [A/B/C]** exactly as specified. Read `references/benchmarking-api.md` and `references/mxfp-quantization.md` before starting.
>
> **Goals**:
> 1. **Faithfulness**: implement the plan as written. Do not introduce unplanned changes.
> 2. **Correctness**: the kernel must be numerically equivalent to the original within the specified tolerances. For MXFP kernels use relaxed tolerances (rtol=5e-2, atol=5e-2). Do not exit the correctness loop until `assert_allclose` passes.
> 3. **Benchmarking**: after correctness passes, benchmark the kernel using `wrap_benchmark` or `nki_benchmark`. Copy `scripts/benchmark.py` from the skill root into the current workspace before importing (see `references/benchmarking-api.md`).
> 4. **Reporting**: return a structured summary to the orchestrator (format below).

### Subagent Internal Loop

```
REPEAT:
  1. Implement the plan change on the current kernel code.
  2. Run the correctness harness (see Step 5).
  3. IF assert_allclose fails:
       - Diagnose the numerical discrepancy (shape mismatch, dtype cast, MXFP layout, scale alignment, etc.)
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

**Correctness**: PASS  max_diff=X.XXe-XX  (tolerances: rtol=X, atol=Y)

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

Run against the **original unmodified kernel** (or a BF16 reference) to establish the reference baseline first. Use the same seed for all subagents.

For MXFP quantized kernels, compare against a BF16 reference (not FP32) and use relaxed tolerances:

```python
import numpy as np
import torch

rng = np.random.default_rng(42)
# Use fp32 input, scale to reasonable range to avoid MXFP saturation
x = (rng.random(shape).astype(np.float32) - 0.5) * 2.0

ref = pytorch_bf16_reference(torch.tensor(x))
result = optimized_mxfp_kernel(torch.tensor(x).to(torch.bfloat16).to("xla")).cpu().float().numpy()

# MXFP quantization error is ~1e-2 — use relaxed tolerances
np.testing.assert_allclose(result, ref, rtol=5e-2, atol=5e-2)
print(f"max_diff={np.abs(result - ref).max():.2e}  PASS")
```

For non-quantized kernels use the standard tight tolerances: rtol=1e-3, atol=1e-3.

### Benchmarking (after correctness passes)

Read `references/benchmarking-api.md` for full details. Minimal pattern:

```python
import os, sys

# Set BEFORE any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"   # trn3, not trn2
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out"
)
# Copy scripts/benchmark.py from the skill root into this directory first
from benchmark import wrap_benchmark

my_kernel = wrap_benchmark(my_kernel, warmup=5, iters=50)
my_kernel(*inputs)

r = my_kernel.last_result
# extract r.device_time_us, r.tensor_engine_pct, r.dma_active_pct, etc.
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
- Did the MXFP quantization path achieve the expected 4x TensorE speedup vs BF16?

### Next Round Decision
- Continue: new bottleneck identified, plans drafted for Round [N+1]
- Stop: performance target met / no further plans are promising
```

If continuing, carry the best kernel variant from this round as the baseline for the next round and go back to Step 2.

---

## Code Documentation Standards

All kernel code produced by this skill must meet these standards:

- **Block comments** before each logical section (quantize, load, compute, store).
- **Inline comments** on any non-obvious indexing, tile size choice, or API selection — explain the *hardware reason*.
- **Named constants** for tile sizes and magic numbers — no bare literals without a comment.
- **Dtype annotations** on all `nl.ndarray` allocations, especially MXFP packed types.
- **Scale tensor shapes** must be explicitly commented — explain the `[P//8, F//4]` layout.

---

## Quick Rules (apply automatically)

- Always `source` the venv and set `NEURON_PLATFORM_TARGET_OVERRIDE=trn3` before running.
- Set all `NEURON_RT_INSPECT_*` env vars **before any neuron/torch_xla import** — setting them after is a silent no-op.
- `nl.par_dim(128)` is the partition dimension size on trn3 (unchanged from trn2).
- `nc_matmul_mx` is **trn3-only** — do not use on trn2. Check platform before emitting.
- `quantize_mx` is **trn3-only** — do not use on trn2. Check platform before emitting.
- MXFP scaling group is **32 elements** along the contraction dimension.
- MXFP tile sizes: stationary `[128P, 128F]`, moving `[128P, 512F]` (MXFP_x4 effective contraction = 512).
- For offline weight quantization, call `quantize_mx` once at load time and cache MXFP8+scales in HBM.
- Use `affine_range` for independent DMA loads; keep accumulation loops `sequential_range`.
- `PSUM` buffers must be copied to SBUF before storing to HBM.
- BF16 output from `nc_matmul_mx` goes directly to PSUM — copy to SBUF before downstream ops.
- Background TensorE transpose (trn3 feature): the TensorE can run a transpose in parallel with another matmul — exploit this to hide transpose latency.
- VectorE fast exp (trn3): 4x throughput vs `activation` instruction — prefer `nisa.exponential` on trn3.
- Subagents run sequentially — never dispatch two subagents in parallel.
