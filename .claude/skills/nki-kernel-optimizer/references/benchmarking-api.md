# Benchmarking API Reference

The harness is in `scripts/benchmark.py` (relative to this skill's root).

**Before benchmarking, copy `benchmark.py` into your current workspace:**

```bash
cp "$(git rev-parse --show-toplevel)/.claude/skills/nki-kernel-optimizer/scripts/benchmark.py" .
```

Then import it as a local module — no path manipulation needed.

---

## Required Setup — Must Happen Before Any Neuron Import

```python
import os

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "/abs/path/to/bench_out"  # must be absolute
```

These must be set **before** `import torch`, `import torch_xla`, or any `nki` import. Setting them after is a silent no-op — the runtime will not capture the NTFF.

---

## Style A — `wrap_benchmark` (preferred for development)

Apply after `@nki.jit`. The wrapper is transparent: same call signature, same return value.

```python
import nki
from benchmark import wrap_benchmark  # benchmark.py must be in the same directory

@nki.jit(platform_target="trn2")
def my_kernel(a, b):
    ...

my_kernel = wrap_benchmark(my_kernel, warmup=5, iters=50)

# First call: compiles, runs, captures NTFF, prints device metrics.
result_tensor = my_kernel(a, b)

# Access device metrics after the call:
r = my_kernel.last_result
print(r.device_time_us)      # hardware execution time in μs
print(r.tensor_engine_pct)   # tensor engine utilization %
print(r.dma_active_pct)      # DMA active time %
print(r.spill_bytes)         # SBUF spill bytes (save + reload)
```

At script exit, the harness prints the `neuron-bench exec` command for end-to-end latency measurement. You do not need to run this command — copy the device metrics from `last_result` for your report.

---

## Style B — `nki_benchmark` (one-shot)

```python
from benchmark import nki_benchmark

result = nki_benchmark(my_kernel, a, b, warmup=5, iters=50)
# result is a BenchmarkResult or None if artifact collection failed
```

---

## `BenchmarkResult` Fields

| Field | Type | Description |
|-------|------|-------------|
| `device_time_us` | `float` | Hardware execution time (μs) from NTFF `total_time` |
| `tensor_engine_pct` | `float` | `tensor_engine_active_time_percent` from NTFF |
| `dma_active_pct` | `float` | `dma_active_time_percent` from NTFF |
| `spill_bytes` | `int` | `spill_save_bytes + spill_reload_bytes` |
| `prof` | `dict` | Full raw summary dict from `neuron-profile view --output-format summary-json` |

Additional metrics available via `result.prof`:

| Key | Meaning |
|-----|---------|
| `total_time` | Execution time in seconds |
| `total_active_time` | Time at least one engine was active |
| `tensor_engine_active_time` | Tensor (matmul) engine active time (s) |
| `vector_engine_active_time` | Vector engine active time (s) |
| `scalar_engine_active_time` | Scalar engine active time (s) |
| `dma_active_time` | DMA active time (s) |
| `mfu_estimated_percent` | Model FLOPs utilization estimate |
| `mbu_estimated_percent` | Memory bandwidth utilization estimate |
| `mm_arithmetic_intensity` | Arithmetic intensity of matmul ops |
| `hbm_read_bytes` | Bytes read from HBM |
| `hbm_write_bytes` | Bytes written to HBM |
| `spill_save_bytes` | Bytes spilled from SBUF to HBM |
| `spill_reload_bytes` | Bytes reloaded from HBM into SBUF |

---

## Reading Metrics for Optimization Decisions

| Observation | Interpretation |
|-------------|---------------|
| `tensor_engine_pct` ≥ 90% | Compute-bound on matmul engine — reduce FLOPs or increase arithmetic intensity |
| `dma_active_pct` high, engines low | Memory-bound — reduce HBM traffic, improve tiling, overlap DMA with compute |
| `spill_bytes` > 0 | SBUF overflow — reduce tile size or hoist allocations out of inner loops |
| All engines low, DMA low | Stall-bound — dependency chain or scheduling issue |
| `mfu_estimated_percent` low despite high engine utilization | Low arithmetic intensity — tiles are too small for the matmul engine |

---

## Full Benchmark Script Template

```python
import os
import sys

# MUST be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out"
)

# benchmark.py must be copied into the same directory as this script
from benchmark import wrap_benchmark

import torch
import torch_xla.core.xla_model as xm
import nki
import nki.language as nl


@nki.jit(platform_target="trn2")
def my_kernel(a, b):
    ...

my_kernel = wrap_benchmark(my_kernel, warmup=5, iters=50)

device = xm.xla_device()
a = torch.randn(128, 1024, dtype=torch.bfloat16).to(device)
b = torch.randn(128, 1024, dtype=torch.bfloat16).to(device)

my_kernel(a, b)

r = my_kernel.last_result
if r:
    print(f"device_time_us       = {r.device_time_us:.2f}")
    print(f"tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
    print(f"dma_active_pct       = {r.dma_active_pct:.1f}%")
    print(f"spill_bytes          = {r.spill_bytes}")
    print(f"mfu_estimated        = {r.prof.get('mfu_estimated_percent', 0):.2f}%")
    print(f"hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"hbm_write_KiB        = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")
```

---

## Common Pitfalls

- **Env vars set after imports**: runtime already initialized, NTFF will not be written. Reorder imports.
- **`NEURON_RT_INSPECT_OUTPUT_DIR` is relative**: use `os.path.abspath(...)` — the NRT resolves this at startup, not at call time.
- **Running neuron-bench while XLA holds the cores**: the harness prints the command but does not run it. This is intentional — run the printed command after the script exits.
- **`last_result` is `None`**: the NEFF was not found, usually because `NEURON_RT_INSPECT_OUTPUT_DIR` was not set before imports.
- **Multiple kernels in one script**: each `wrap_benchmark` call tracks its own NEFF via snapshot diffing. Run one kernel per `wrap_benchmark` invocation per script to avoid artifact ambiguity.
