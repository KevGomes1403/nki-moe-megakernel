"""
Demo: benchmark a simple tensor-add NKI kernel.
All kernel code uses nki.* namespace. No Python timing used.

Run:
    python tensor_add.py
"""
import os

# Must be set BEFORE any neuron/torch_xla imports
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out"
)

import torch
import torch_xla.core.xla_model as xm
import nki
import nki.language as nl
import nki.isa as nisa

from benchmark import wrap_benchmark, nki_benchmark


@nki.jit(platform_target="trn2")
def nki_tensor_add(a_tensor, b_tensor):
    """Element-wise add: c = a + b  ([128, 1024] float32)."""
    c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.hbm)
    a_sbuf   = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.sbuf)
    b_sbuf   = nl.ndarray(b_tensor.shape, dtype=b_tensor.dtype, buffer=nl.sbuf)
    c_sbuf   = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_sbuf, src=a_tensor)
    nisa.dma_copy(dst=b_sbuf, src=b_tensor)
    nisa.tensor_tensor(dst=c_sbuf, data1=a_sbuf, data2=b_sbuf, op=nl.add)
    nisa.dma_copy(dst=c_tensor, src=c_sbuf)
    return c_tensor


if __name__ == "__main__":
    device = xm.xla_device()
    a = torch.zeros(128, 1024, dtype=torch.float32).to(device)
    b = torch.rand(128, 1024, dtype=torch.float32).to(device)

    print("Benchmarking nki_tensor_add...")
    result = nki_benchmark(nki_tensor_add, a, b, warmup=10, iters=100)

    if result:
        print(f"device time = {result.device_time_us:.2f} μs  (hardware trace)")
