# Benchmarking Recipes (nki.benchmark)

This file focuses on practical usage patterns.

## Install prerequisites (host)
Benchmarking requires an AWS trn/inf instance with NeuronDevices and aws-neuronx-tools installed.

## Minimal benchmark wrapper

```python
from neuronxcc.nki import benchmark
import neuronxcc.nki.language as nl
import numpy as np

@benchmark(warmup=10, iters=100, save_neff_name="file.neff", save_trace_name="profile.ntff")
def add(a_tensor, b_tensor):
    c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    a = nl.load(a_tensor)
    b = nl.load(b_tensor)
    c = a + b
    nl.store(c_tensor, c)
    return c_tensor

a = np.zeros([128, 1024], dtype=np.float32)
b = np.random.random_sample([128, 1024]).astype(np.float32)
_ = add(a, b)

metrics = add.benchmark_result.nc_latency
print("p50(us) =", metrics.get_latency_percentile(50))
print("p99(us) =", metrics.get_latency_percentile(99))
```

## Options you should expose in experiments
- `warmup`: increase for kernels with JIT/initialization effects
- `iters`: increase for stable percentiles
- `additional_compile_opt`: pass compiler flags for specific experiments
- `save_neff_name` / `save_trace_name`: save artifacts for profile analysis

## Interpretation notes
- Percentiles are in microseconds (us).
- `neuron-bench` includes host<->device transfer time.
- The benchmark does not use the actual runtime inputs for correctness checks.
  Always do correctness testing separately.
