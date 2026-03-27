# Kernels

This directory is a workspace for Qwen 3 MoE kernel work on Trainium/Neuron. It contains several kernel families that are being incrementally optimized, validated, and profiled rather than a single polished package. The center of gravity today is the fused MoE path for the Qwen3-30B-A3B shape, alongside router and attention kernels that isolate major subproblems.

## What Is In Here

- `moe_fused_tkg/`: the main fused MoE workspace. The current kernel fuses RMSNorm, router matmul, softmax/top-k, and selective expert MLP execution into one kernel specialized for Qwen3-30B-A3B with TP=4 and LNC=2.
- `router_topk/`: a standalone router kernel for Qwen3 CTE shapes. It computes logits, stable softmax, top-k expert indices, and scattered normalized affinities.
- `attn_tkg/`: token-generation attention work. The profiling notes show an optimization path focused on reducing transpose overhead, deduplicating KV work for GQA, and improving KV reuse.
- `attn_cte/`: context-encoding attention work. The kernel is split into a K/V staging phase and a Q + flash-attention phase, with later optimizations aimed at reducing DMA traffic and widening flash-attention tiles.
- `benchmarking_workspace/`: shared benchmarking utilities for `nki.*` kernels, including NTFF parsing and `neuron-bench` command generation.
- `nkilib/` and `nki_moe.py`: earlier or more general-purpose kernel experiments and references.

## Characteristics Of The Kernels

Most kernels in this workspace are deliberately shape-specialized for Qwen3 rather than written as generic operators. Common patterns across the codebase:

- Hardcoded Qwen3 dimensions such as hidden size `2048`, `128` experts, and top-k `8`.
- Explicit Trainium-oriented tiling around `PMAX=128`.
- LNC-aware sharding, especially `LNC=2`, where work is split across two NeuronCores.
- Aggressive fusion to keep intermediate tensors off HBM when possible.
- Repeated focus on DMA pressure, transpose overhead, and matmul layout as the main performance constraints.
- Versioned kernels (`kernel_v1a.py`, `kernel_v3b.py`, `v5a`, `v8a`, etc.) used as optimization checkpoints instead of replacing earlier iterations.

In the fused MoE kernel specifically, the current implementation:

- Runs end-to-end MoE inference for the Qwen3 token-generation case.
- Replicates RMSNorm and routing work across both LNC programs.
- Uses full-hidden gate/up projection on each core, then shards only the down projection output across the hidden dimension.
- Uses hardware top-k primitives (`max8` / index lookup) and accumulates expert outputs in fp32 before writing bf16 output.

## Optimization Style

The profiling notes make the optimization strategy fairly consistent:

- Eliminate redundant HBM loads by preloading or hoisting tiles into SBUF.
- Remove transpose-heavy layouts when direct column or pre-transposed layouts are possible.
- Replace repeated small DMA or broadcast chains with wider batched operations.
- Preserve correctness with per-version tests before keeping the faster variant.

For example:

- `attn_tkg/PROFILING.md` shows a progression from high transpose overhead to layouts that hoist hidden tiles, deduplicate KV work for GQA, and pack more work into the free dimension.
- `attn_cte/PROFILING.md` focuses on cutting hot-path HBM loads and reducing flash-attention loop overhead with wider K tiles.
- `moe_fused_tkg/` keeps many `kernel_v*` files so individual optimization steps can be benchmarked and compared directly.

## Running

These kernels assume a Neuron/Trainium environment. At minimum, use a Neuron-enabled Python environment and set platform targeting before importing `torch_xla` or Neuron modules.

Typical setup:

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
```

Common entrypoints:

```bash
python kernels/moe_fused_tkg/test_correctness.py
python kernels/moe_fused_tkg/test_correctness.py kernel_v1a.py
python kernels/moe_fused_tkg/test_correctness.py /abs/path/to/kernel.py
python kernels/router_topk/test_qwen3_router_topk_cte.py
python kernels/attn_tkg/test_attn_tkg_fused.py
```

## Benchmarking

The reusable benchmark harness lives in [benchmark.py](/home/ubuntu/nki-moe/kernels/benchmarking_workspace/benchmark.py). The `moe_fused_tkg/test_correctness.py` script is one example of how to use it, but the API is meant to work for your own kernels too.

### Setup

Before benchmarking a new kernel, make the harness available next to your benchmark script or import it from this repo:

```python
from kernels.benchmarking_workspace.benchmark import wrap_benchmark, nki_benchmark
```

The benchmark environment variables must be set before any `torch`, `torch_xla`, or `nki` import:

```python
import os

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "/abs/path/to/bench_out"
```

`NEURON_RT_INSPECT_OUTPUT_DIR` should be absolute. If these are set after Neuron or XLA imports, profiling will silently fail because the runtime has already initialized.

### Preferred API: `wrap_benchmark`

Use `wrap_benchmark` when you are iterating on one kernel during development. Apply it after `@nki.jit`; the wrapped function keeps the same call signature and return value.

```python
import os

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.abspath("./_bench_out")

import torch
import torch_xla.core.xla_model as xm
import nki

from kernels.benchmarking_workspace.benchmark import wrap_benchmark


@nki.jit(platform_target="trn2")
def my_kernel(a, b):
    ...


my_kernel = wrap_benchmark(my_kernel, warmup=5, iters=50)

device = xm.xla_device()
a = torch.randn(128, 1024, dtype=torch.bfloat16).to(device)
b = torch.randn(128, 1024, dtype=torch.bfloat16).to(device)

out = my_kernel(a, b)

r = my_kernel.last_result
if r:
    print(f"device_time_us    = {r.device_time_us:.2f}")
    print(f"tensor_engine_pct = {r.tensor_engine_pct:.1f}%")
    print(f"dma_active_pct    = {r.dma_active_pct:.1f}%")
    print(f"spill_bytes       = {r.spill_bytes}")
```

What happens on the first call:

- The kernel compiles and runs.
- The harness snapshots generated artifacts and finds the new NEFF and NTFF.
- `neuron-profile view --output-format summary-json` is used to extract device metrics from the NTFF.
- A `BenchmarkResult` is stored on `my_kernel.last_result`.
- A `neuron-bench exec ...` command is printed at process exit for optional end-to-end latency measurement after XLA releases the NeuronCores.

### One-shot API: `nki_benchmark`

Use `nki_benchmark` when you want a single explicit benchmark call instead of mutating the kernel symbol:

```python
result = nki_benchmark(my_kernel, a, b, warmup=5, iters=50)
if result:
    print(result.device_time_us)
```

This is just a thin one-shot wrapper around `wrap_benchmark`.

### `BenchmarkResult`

The main fields exposed by the API are:

- `device_time_us`: hardware execution time in microseconds from NTFF `total_time`.
- `tensor_engine_pct`: tensor-engine active percentage.
- `dma_active_pct`: DMA active percentage.
- `spill_bytes`: `spill_save_bytes + spill_reload_bytes`.
- `prof`: the full parsed summary dict from `neuron-profile`.

Useful raw keys in `result.prof`:

- `total_time`
- `total_active_time`
- `tensor_engine_active_time`
- `vector_engine_active_time`
- `scalar_engine_active_time`
- `dma_active_time`
- `mfu_estimated_percent`
- `mbu_estimated_percent`
- `mm_arithmetic_intensity`
- `hbm_read_bytes`
- `hbm_write_bytes`
- `spill_save_bytes`
- `spill_reload_bytes`

### How To Interpret The Metrics

- High `tensor_engine_pct` means the kernel is compute-heavy on the matmul engine.
- High `dma_active_pct` with low engine utilization usually means the kernel is memory-bound and dominated by HBM traffic or DMA scheduling.
- Non-zero `spill_bytes` means SBUF pressure is too high and tile sizes or allocation lifetimes need attention.
- Low engine utilization and low DMA utilization together usually point to dependency-chain or scheduling stalls.
- Low `mfu_estimated_percent` even when engines are busy usually means arithmetic intensity is still poor.

### Benchmarking Your Own Kernel

Recommended workflow:

1. Write a small standalone benchmark script for one kernel variant.
2. Set the benchmark env vars before any Neuron-related import.
3. Move inputs to `xm.xla_device()`.
4. Use `wrap_benchmark` for normal iteration.
5. Read `last_result` and compare device metrics across variants.
6. Run the printed `neuron-bench exec ...` command after the script exits if you also want end-to-end latency.

Minimal template:

```python
import os

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.abspath("./_bench_out")

import torch
import torch_xla.core.xla_model as xm
import nki

from kernels.benchmarking_workspace.benchmark import wrap_benchmark


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
    print(f"device_time_us = {r.device_time_us:.2f}")
    print(f"mfu_estimated  = {r.prof.get('mfu_estimated_percent', 0):.2f}%")
    print(f"hbm_read_KiB   = {r.prof.get('hbm_read_bytes', 0) / 1024:.1f}")
    print(f"hbm_write_KiB  = {r.prof.get('hbm_write_bytes', 0) / 1024:.1f}")
```

### Common Pitfalls

- Env vars set after imports: NTFF capture will not start.
- Relative `NEURON_RT_INSPECT_OUTPUT_DIR`: use an absolute path.
- Running `neuron-bench` before the script exits: XLA may still hold the cores.
- `last_result is None`: artifact discovery failed, usually because profiling env vars were set too late.
- Benchmarking many kernels in one script: artifact matching becomes ambiguous. Prefer one wrapped kernel per script when comparing variants.

### Existing Repo Entry Points

If you want working examples in this repo, start with:

```bash
python kernels/moe_fused_tkg/test_correctness.py --benchmark
python kernels/moe_fused_tkg/test_correctness.py kernel_v5a.py --benchmark --warmup 5 --iters 50
```

Those scripts already configure profiling output, move tensors to XLA, and call the shared benchmark harness.

## Practical Workflow

When adding a new kernel version:

1. Start from the nearest `kernel_v*` or attention/router variant.
2. Keep the shape assumptions explicit instead of half-generalizing them.
3. Validate numerics with the local correctness harness.
4. Capture profile data before and after the change.
5. Keep the old version if it documents a meaningful optimization step.

This directory is best read as an optimization notebook in code form: small specialized kernels, explicit hardware assumptions, and versioned experiments that track how the Qwen 3 MoE path is being improved over time.
