from neuronxcc.nki import benchmark
import neuronxcc.nki.language as nl
import numpy as np

@benchmark(warmup=10, iters = 100, save_neff_name='file.neff', save_trace_name='profile.ntff')
def nki_tensor_tensor_add(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  a = nl.load(a_tensor)
  b = nl.load(b_tensor)

  c = a + b

  nl.store(c_tensor, c)

  return c_tensor

a = np.zeros([128, 1024], dtype=np.float32)
b = np.random.random_sample([128, 1024]).astype(np.float32)
c = nki_tensor_tensor_add(a, b)

metrics = nki_tensor_tensor_add.benchmark_result.nc_latency
print("latency.p50 = " + str(metrics.get_latency_percentile(50)))
print("latency.p99 = " + str(metrics.get_latency_percentile(99)))