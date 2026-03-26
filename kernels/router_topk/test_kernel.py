"""
Usage:
    python test_kernel.py <module_name>

    <module_name> is the Python module (without .py) in the same directory,
    exporting a function named `qwen3_router_topk_cte`.

Examples:
    python test_kernel.py qwen3_router_topk_cte
    python test_kernel.py qwen3_router_topk_plan_a
    python test_kernel.py qwen3_router_topk_plan_b

The kernel under test is compared against a PyTorch reference that matches the
Qwen3 router math exactly:
    logits       = (x @ w) cast to bf16 then back to fp32   [bf16 round-trip]
    affinities   = softmax(logits, dim=-1)
    topk_idx     = top-K indices of affinities per token
    sum_topk     = sum of top-K affinity values per token
    out_affi[e]  = affinities[e] / sum_topk  if e in topk_idx, else 0
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch_xla.core.xla_model as xm

import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

os.environ["NEURON_CC_FLAGS"] = " "
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"]= "1"
os.environ["XLA_HLO_DEBUG"]= "1"
os.environ["NEURON_RT_INSPECT_ENABLE"]= "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"]= "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]= "./output"

sys.path.insert(0, str(Path(__file__).parent))

KERNEL_FN = "qwen3_router_topk_cte"
T, H, E, K = 640, 2048, 128, 8
LNC       = 2


def pytorch_reference(x: torch.Tensor, w: torch.Tensor, k: int):
    """
    Pure PyTorch Qwen3 router reference (CPU, fp32).

    Mirrors the kernel math:
      1. bf16 matmul with fp32 accumulation, then bf16 round-trip to match
         the kernel's PSUM->bf16->fp32 cast that pins logits to the bf16 grid.
      2. Numerically stable softmax over experts.
      3. top-K selection; L1-normalise the top-K affinities.
      4. Scatter normalised values back into a [T, E] tensor (zeros elsewhere).
    """
    # Step 1: logits — fp32 matmul then bf16 round-trip
    # x and w are bf16; upcasting to fp32 for the matmul mimics fp32 PSUM accumulation.
    logits = (x.float() @ w.float()).bfloat16().float()  # [T, E]

    # Step 2: softmax over experts
    affinities = torch.softmax(logits, dim=-1)  # [T, E]

    # Step 3: top-K indices and values
    topk_vals, topk_idx = torch.topk(affinities, k, dim=-1)  # [T, K] each

    # Step 4: L1-normalise top-K affinities, scatter into [T, E]
    sum_topk = topk_vals.sum(dim=-1, keepdim=True)  # [T, 1]
    normed   = topk_vals / sum_topk                 # [T, K]
    out_affi = torch.zeros_like(affinities)
    out_affi.scatter_(1, topk_idx, normed)          # [T, E]

    return logits, out_affi, topk_idx


def load_kernel(module_name: str):
    mod = importlib.import_module(module_name)
    return getattr(mod, KERNEL_FN)


def main():
    parser = argparse.ArgumentParser(description="Correctness test for router_topk kernels")
    parser.add_argument("module", help="Module name to test (e.g. qwen3_router_topk_plan_a)")
    args = parser.parse_args()

    device = xm.xla_device()

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((T, H)).astype(np.float16)
    w_np = rng.standard_normal((H, E)).astype(np.float16)

    x_cpu = torch.tensor(x_np, dtype=torch.bfloat16)
    w_cpu = torch.tensor(w_np, dtype=torch.bfloat16)

    # PyTorch reference on CPU
    rl_ref, ea_ref, ei_ref = pytorch_reference(x_cpu, w_cpu, K)
    rl_ref_np = rl_ref.numpy()
    ea_ref_np = ea_ref.numpy()
    ei_ref_np = ei_ref.numpy().astype(np.uint32)

    # NKI kernel on device
    x_t  = x_cpu.to(device)
    w_t  = w_cpu.to(device)
    rl_d = torch.zeros(T, E, dtype=torch.float32).to(device)
    ea_d = torch.zeros(T, E, dtype=torch.float32).to(device)
    ei_d = torch.zeros(T, K, dtype=torch.int32).to(device)

    print(f"Reference : PyTorch (CPU)")
    print(f"Under test: {args.module}")

    kernel = load_kernel(args.module)
    rl_t, ea_t, ei_t = kernel[LNC](x_t, w_t, rl_d, ea_d, ei_d)
    xm.mark_step()
    rl_t_np = rl_t.cpu().numpy()
    ea_t_np = ea_t.cpu().numpy()
    ei_t_np = ei_t.cpu().numpy().astype(np.uint32)

    np.testing.assert_allclose(rl_t_np, rl_ref_np, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(ea_t_np, ea_ref_np, rtol=1e-2, atol=1e-2)
    np.testing.assert_array_equal(ei_t_np, ei_ref_np)

    print(f"router_logits  max_diff={np.abs(rl_t_np - rl_ref_np).max():.2e}  PASS")
    print(f"expert_affi    max_diff={np.abs(ea_t_np - ea_ref_np).max():.2e}  PASS")
    print(f"expert_index   exact_match  PASS")


if __name__ == "__main__":
    main()
