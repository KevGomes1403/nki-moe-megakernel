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

**Hardware**: trn2 instance — 4 NeuronCores (NeuronCore-v3 / gen3).
NeuronCores are exclusive: only one Python process can hold them at a time. Run kernels sequentially.

---

## Reference Docs

| File | Contents |
|------|----------|
| `references/nki-overview.md` | Architecture, APIs, programming model |
| `references/nki-syntax-quickref.md` | nki.lang / nki.isa cheatsheet, common patterns |
| `references/performance-playbook.md` | Structured step-by-step optimization workflow |
| `references/benchmarking-recipes.md` | Wall-clock timing, trace emission, profiling hooks |
| `references/common-pitfalls.md` | Compiler errors, dtype gotchas, scheduling mistakes |
| `references/templates.md` | Ready-to-use kernel scaffolding and harness templates |
| `references/logical-neuron-cores.md` | Combining two physical NeuronCores into one logical NeuronCore |

Read the relevant reference(s) before generating or modifying any kernel code.

---

## Intake — Collect Before Starting

Ask only what you need:

1. **Device generation**: trn1/inf2 (NeuronCore-v2), trn2, or trn3.
   Default to **trn2** unless specified.
2. **Op category**: elementwise, reduction, fused, matmul/GEMM, attention primitive, MoE, custom.
3. **Shapes & dtypes**: all input/output tensor shapes, dtypes, any alignment constraints.
4. **Profiling data**: neuron-profile trace metrics (engine utilization, spill bytes, DMA activity, etc.). These drive the optimization plans — do not proceed to planning without them.
5. **Existing code**: the kernel to optimize (required).

---

## Standard Workflow

### Step 1 — Math Spec

Write a compact, unambiguous spec:
- Inputs/outputs: shape, dtype, memory location (HBM vs SBUF).
- Exact math (scaling, masking, epsilon, etc.).
- Fusion boundaries: what must be fused vs. what can be separate.
- Tolerances: rtol/atol for correctness check.

If the user supplies PyTorch code, derive the spec and annotate which parts can be safely fused.

---

### Step 2 — Profiling Analysis

Before planning, analyze the provided profiling data and characterize the kernel's current bottleneck:

- **Compute-bound**: ≥90% engine utilization on bottleneck engine.
- **Memory-bound**: high `dma_active_time_percent`, low engine utilization, significant HBM traffic.
- **Spill-bound**: non-zero `spill_save_bytes` / `spill_reload_bytes` — SBUF overflow forcing eviction.
- **Stall-bound**: low utilization on all engines with no DMA overlap — scheduling or dependency issue.

Document which metrics from the trace support your characterization. This framing drives the three plans in Step 3.

---

### Step 3 — Optimization Planning (Three Distinct Plans)

Generate **exactly three optimization plans**. Each plan must:

- Be **independent** — implementable without relying on the other two plans.
- Be **specific** — reference exact loop variables, tensor names, tile sizes, API calls, and profiling metrics that motivate the change. Vague guidance ("tile better") is not acceptable.
- Target a **distinct bottleneck or axis of improvement** (e.g., one plan may address memory bandwidth, another engine selection, another loop structure).
- State the **hypothesis**: which metric in the profiler should improve and in what direction.

Present each plan in this format:

```
### Plan A — <Short Title>

**Bottleneck targeted**: <specific metric from profiling data>
**Root cause**: <why this metric is high / low>
**Change**: <exact code-level change — loop restructure, tile size, API swap, fusion boundary, etc.>
**Expected effect**: <which profiler metric improves, and why>
**Correctness risk**: <any numerical or shape concern to watch for>
**Verification**: assert_allclose(rtol=X, atol=Y) — explain what to compare against
```

Do not begin any implementation until all three plans are written and confirmed (implicitly or explicitly) by the user.

---

### Step 4 — Sequential Subagent Dispatch

Dispatch three subagents **one at a time** (hardware allows only one process at a time).
Each subagent operates under the following mandate:

#### Subagent Charter

> You are implementing **Plan [A/B/C]** exactly as specified. Your goals are:
> 1. **Faithfulness**: implement the plan as written. Do not introduce unplanned changes, even if you think they would help.
> 2. **Correctness**: the kernel must be numerically equivalent to the original before you finish. Do not mark the task complete until `assert_allclose` passes.
> 3. **Documentation**: every non-obvious line must have an inline comment explaining *why*, not just *what*.

Each subagent must follow this internal loop:

```
REPEAT:
  1. Implement the plan change on the current kernel code.
  2. Run the correctness harness (see Step 5).
  3. IF assert_allclose fails:
       - Diagnose the numerical discrepancy (shape mismatch, dtype cast, accumulation order, etc.)
       - Fix only what is needed to restore correctness; do not expand scope.
       - Return to step 1.
  4. UNTIL assert_allclose passes.
THEN: output the final kernel with documentation and a correctness confirmation line.
```

A subagent **may not exit** while `assert_allclose` is failing. Partial implementations are not acceptable outputs.

Output from each subagent:
- Complete, runnable kernel code (with inline comments).
- Final correctness confirmation: `max_diff=X.XXe-XX  PASS`.
- A one-paragraph implementation note: what was changed, any deviation from the plan (must be flagged and justified), and any residual risk.

---

### Step 5 — Correctness Harness

This harness is used by every subagent. It must be run against the **original unmodified kernel** to establish the reference baseline first.

```python
import numpy as np
import torch

# Deterministic inputs — use the same seed for all three subagents
rng = np.random.default_rng(42)
x = rng.random(shape).astype(np.float32)  # replace `shape` with actual shape

# Reference: run the PyTorch reference to get the expected outputs
ref = pytorch_kernel(torch.tensor(x))

# Optimized result
result = optimized_kernel(torch.tensor(x).to("xla")).cpu().numpy()

# Numerical equivalence check
np.testing.assert_allclose(result, ref, rtol=1e-3, atol=1e-3)
print(f"max_diff={np.abs(result - ref).max():.2e}  PASS")
```

**Important**: all three subagents compare against the same `ref` (the original kernel output), not against each other. This keeps correctness checks independent and reproducible.

---

### Step 6 — Final Summary

After all three subagents have completed, produce a structured summary:

```
## Optimization Summary

### Original Kernel
- Brief description of what it did and its profiled bottleneck(s).

### Plan A — <Title>
- What changed (code-level).
- Why it was expected to help (hardware rationale).
- Correctness: PASS (max_diff=X.XXe-XX).
- Implementation notes / any deviations from the plan.

### Plan B — <Title>
- (same structure)

### Plan C — <Title>
- (same structure)

### Recommendations
- Which plan(s) are most promising to profile next and why.
- Any interactions between plans that could be combined in a follow-up.
- Any remaining risks or open questions.
```

The user will profile the three optimized kernels themselves. Do not include any wall-clock timing, `nki.benchmark` calls, or synthetic latency estimates.

---

## Code Documentation Standards

All kernel code produced by this skill (baseline or optimized) must meet these standards:

- **Block comments** before each logical section (load, compute, store).
- **Inline comments** on any non-obvious indexing, tile size choice, or API selection — explain the *hardware reason* (e.g., "nl.affine_range here because DMA iterations are independent, enabling pipelining").
- **Named constants** for tile sizes and magic numbers — no bare literals without a comment.
- **Dtype annotations** on all `nl.ndarray` allocations.

---

## Quick Rules (apply automatically)

- Always `source` the venv and set `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` before running.
- `expert_affinities` and other accumulation-path tensors must be `float32` (not `bf16`) for POST_SCALE MoE patterns.
- `nl.par_dim(128)` is the partition dimension size on trn2; tile your partition dimension accordingly.
- Use `nl.affine_range` for DMA loads that are independent; keep accumulation loops sequential.
- When debugging compiler errors: reduce to minimal baseline (no fusion, no affine_range), then re-add optimizations one at a time.
- `PSUM` buffers must be copied to SBUF before storing to HBM.
- `.ap()` works on HBM and SBUF/PSUM tensors; see `nki-syntax-quickref.md` for restrictions and the DGE `scalar_offset` address pitfall.
- No benchmarking in this workflow — the user profiles externally.