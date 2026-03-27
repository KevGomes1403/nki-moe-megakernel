"""
Print complete result.prof for kernel_v8a.
"""
import os
import sys
import json

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output"
)

sys.path.insert(0, "/home/ubuntu/nki-moe")

from kernels.benchmarking_workspace.benchmark import nki_benchmark

import torch
import torch_xla.core.xla_model as xm

# Load kernel
import importlib.util
spec = importlib.util.spec_from_file_location(
    "kernel_v8a",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_v8a.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
run_kernel = mod.run

# Create inputs (same as test_correctness.py)
torch.manual_seed(42)
B = 1
H, E, K, I = 2048, 128, 8, 256

inp      = torch.randn(B, 1, H, dtype=torch.bfloat16)
gamma    = torch.randn(1, H, dtype=torch.bfloat16)
router_w = torch.randn(H, E, dtype=torch.bfloat16)
gate_up_w = torch.randn(E, H, 2, I, dtype=torch.bfloat16)
down_w   = torch.randn(E, I, H, dtype=torch.bfloat16)

device = xm.xla_device()
inp_xla      = inp.to(device)
gamma_xla    = gamma.to(device)
router_xla   = router_w.to(device)
gate_up_xla  = gate_up_w.to(device)
down_xla     = down_w.to(device)

result = nki_benchmark(
    run_kernel,
    inp_xla, gamma_xla, router_xla, gate_up_xla, down_xla,
    warmup=5,
    iters=50,
)

if result and result.prof:
    print("\n=== Complete result.prof JSON ===")
    print(json.dumps(result.prof, indent=2))
else:
    print("No profile data available")
