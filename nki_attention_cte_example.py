"""
Simple Working NKI Kernel Test - attention_cte
"""

import os

import torch
import torch_xla.core.xla_model as xm

from nkilib.core.attention.attention_cte import attention_cte
from nkilib.core.attention.attention_cte_torch import attention_cte_torch_ref


def test_attention_cte():
    """Instantiate and validate attention_cte against a PyTorch reference."""
    print("=" * 70)
    print("NKI attention_cte Test")
    print("=" * 70)

    # Ensure NKI tracing can identify the target on this host.
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

    device = xm.xla_device()
    print(f"\nUsing device: {device}")
    print(f"NEURON_PLATFORM_TARGET_OVERRIDE={os.environ['NEURON_PLATFORM_TARGET_OVERRIDE']}")

    # Keep shapes small so compile/run is fast.
    bs, seqlen_q, seqlen_k, d = 1, 16, 16, 32
    print(f"Shapes: q=({bs}, {seqlen_q}, {d}), k=({bs}, {d}, {seqlen_k}), v=({bs}, {seqlen_k}, {d})")

    torch.manual_seed(0)
    q_cpu = torch.randn(bs, seqlen_q, d, dtype=torch.bfloat16)
    k_cpu = torch.randn(bs, d, seqlen_k, dtype=torch.bfloat16)
    v_cpu = torch.randn(bs, seqlen_k, d, dtype=torch.bfloat16)

    q = q_cpu.to(device)
    k = k_cpu.to(device)
    v = v_cpu.to(device)

    print("\n[1/3] Computing reference (PyTorch)...")
    ref_output = attention_cte_torch_ref(
        q_cpu,
        k_cpu,
        v_cpu,
        causal_mask=True,
        tp_q=True,
        tp_k=False,
        tp_out=False,
    )

    print("[2/3] Computing NKI kernel...")
    nki_output = attention_cte(
        q,
        k,
        v,
        causal_mask=True,
        tp_q=True,
        tp_k=False,
        tp_out=False,
    )

    print("[3/3] Comparing...")
    ref_fp32 = ref_output.to(torch.float32)
    nki_fp32 = nki_output.cpu().to(torch.float32)
    diff = torch.abs(ref_fp32 - nki_fp32)

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print("\nResults:")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    # BF16 kernel output is expected to have small numerical differences vs fp32 reference.
    tol = 2e-2
    if max_diff < tol:
        print("\n" + "=" * 70)
        print("SUCCESS! attention_cte instantiates and executes on trn2.")
        print("=" * 70)
        return True

    print("\nFAILED - Numerical mismatch exceeds tolerance")
    return False


if __name__ == "__main__":
    test_attention_cte()
