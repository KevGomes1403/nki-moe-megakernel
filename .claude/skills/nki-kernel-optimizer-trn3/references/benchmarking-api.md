# Benchmarking API Reference (trn3)

The harness is in `scripts/benchmark.py` (relative to this skill's root).

**Before benchmarking, copy `benchmark.py` into your current workspace:**

```bash
cp "$(git rev-parse --show-toplevel)/.claude/skills/nki-kernel-optimizer-trn3/scripts/benchmark.py" .
```

The trn3 harness adds `vector_engine_pct` to `BenchmarkResult` (quantize-bound diagnostic) and prints a trn3-specific bottleneck hint. Use this version, not the trn2 one.

---

## Required Setup — Must Happen Before Any Neuron Import

```python
import os

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"   # ← trn3, not trn2
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
from benchmark import wrap_benchmark

@nki.jit(platform_target="trn3")
def my_kernel(a, b):
    ...

my_kernel = wrap_benchmark(my_kernel, warmup=5, iters=50)

result_tensor = my_kernel(a, b)

r = my_kernel.last_result
print(r.device_time_us)      # hardware execution time in μs
print(r.tensor_engine_pct)   # tensor engine utilization %
print(r.dma_active_pct)      # DMA active time %
print(r.spill_bytes)         # SBUF spill bytes (save + reload)
```

---

## Style B — `nki_benchmark` (one-shot)

```python
from benchmark import nki_benchmark

result = nki_benchmark(my_kernel, a, b, warmup=5, iters=50)
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

Additional metrics via `result.prof`:

| Key | Meaning |
|-----|---------|
| `total_time` | Execution time in seconds |
| `tensor_engine_active_time` | TensorE active time (s) — high = compute-bound |
| `vector_engine_active_time` | VectorE active time (s) — high = quantize-bound or elementwise-bound |
| `scalar_engine_active_time` | ScalarE active time (s) |
| `dma_active_time` | DMA active time (s) |
| `mfu_estimated_percent` | Model FLOPs utilization estimate |
| `mbu_estimated_percent` | Memory bandwidth utilization estimate |
| `mm_arithmetic_intensity` | Arithmetic intensity of matmul ops |
| `hbm_read_bytes` | Bytes read from HBM |
| `hbm_write_bytes` | Bytes written to HBM |
| `spill_save_bytes` | Bytes spilled from SBUF to HBM |
| `spill_reload_bytes` | Bytes reloaded from HBM into SBUF |

---

## Reading Metrics for Optimization Decisions (trn3-specific)

| Observation | Interpretation | Action |
|-------------|---------------|--------|
| `tensor_engine_pct` ≥ 90% | Compute-bound on TensorE | Check if already on MXFP path; if BF16, migrate to `nc_matmul_mx` for 4× |
| `vector_engine_active_time` high, TensorE low | Quantize-bound (`quantize_mx` dominates) | Move to offline weight quantization; pipeline quantize-of-activations with prior tile's matmul |
| `dma_active_pct` high, engines low | Memory-bound | Use MXFP8 weights (halves HBM bytes); increase tile size; overlap DMA with compute |
| `spill_bytes` > 0 | SBUF overflow | Reduce tile size; SBUF is 32 MiB on trn3 (was 28 MiB), use the extra 4 MiB for larger tiles |
| All engines low, DMA low | Stall-bound | Dependency chain or scheduling issue |
| `mfu_estimated_percent` low despite high TensorE | Low arithmetic intensity | Tiles too small for matmul engine; increase K or N tile |

---

## Full Benchmark Script Template (trn3)

```python
import os
import sys

# MUST be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out"
)

from benchmark import wrap_benchmark

import torch
import torch_xla.core.xla_model as xm
import nki
import nki.language as nl


@nki.jit(platform_target="trn3")
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
    print(f"vector_engine_pct    = {r.prof.get('vector_engine_active_time_percent', 0):.1f}%")
    print(f"spill_bytes          = {r.spill_bytes}")
    print(f"mfu_estimated        = {r.prof.get('mfu_estimated_percent', 0):.2f}%")
    print(f"hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"hbm_write_KiB        = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")
```

---

## Common Pitfalls

- **Wrong platform target**: using `trn2` instead of `trn3` in `NEURON_PLATFORM_TARGET_OVERRIDE` — `nc_matmul_mx` will fail to compile.
- **Env vars set after imports**: runtime already initialized, NTFF will not be written. Reorder imports.
- **`NEURON_RT_INSPECT_OUTPUT_DIR` is relative**: use `os.path.abspath(...)`.
- **`last_result` is `None`**: the NEFF was not found — check `NEURON_RT_INSPECT_OUTPUT_DIR` was set before imports.
- **Multiple kernels in one script**: run one kernel per `wrap_benchmark` invocation to avoid artifact ambiguity.
