"""
Numerical equivalence test for qwen3_router_topk_cte kernel.

Compares NKI kernel output against a PyTorch golden reference
for Qwen3-30B-A3B CTE shapes at TP=4, LNC=2.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
    python test_qwen3_router_topk_cte.py
"""

import os
import numpy as np
import torch

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_CC_FLAGS"] = " "
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"]= "1"
os.environ["XLA_HLO_DEBUG"]= "1"
os.environ["NEURON_RT_INSPECT_ENABLE"]= "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"]= "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]= "./output"

# torch_xla must be imported before any .to("xla") call
import torch_xla.core.xla_model as xm  # noqa: E402

from qwen3_router_topk_cte import qwen3_router_topk_cte  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (Qwen3-30B-A3B, TP=4)
# ---------------------------------------------------------------------------
H = 2048   # hidden_size
E = 128    # num_experts
K = 8      # top-K experts per token


# ---------------------------------------------------------------------------
# PyTorch golden reference
# ---------------------------------------------------------------------------
def pytorch_reference(x_np, w_np):
    """
    Args:
        x_np: [T, H] float32 (we cast to bf16 for matmul fidelity)
        w_np: [H, E] float32

    Returns:
        logits:      [T, E] float32
        affinities:  [T, E] float32  (scattered, L1-normalized)
        indices:     [T, K] int64
    """
    x = torch.tensor(x_np, dtype=torch.bfloat16)
    w = torch.tensor(w_np, dtype=torch.bfloat16)

    # Router logits: x @ w (in float32 for precision)
    logits = torch.matmul(x.float(), w.float())   # [T, E]

    # Softmax over expert dimension
    affinities_full = torch.softmax(logits, dim=-1)  # [T, E]

    # Top-K selection
    topk_vals, topk_idx = torch.topk(affinities_full, k=K, dim=-1)  # [T, K]

    # L1 normalization of top-K weights
    sum_topk = topk_vals.sum(dim=-1, keepdim=True)   # [T, 1]
    topk_vals_norm = topk_vals / sum_topk             # [T, K]

    # Scatter normalized weights back to [T, E] (zero everywhere else)
    affinities_scattered = torch.zeros_like(affinities_full)
    affinities_scattered.scatter_(-1, topk_idx, topk_vals_norm)

    return (
        logits.numpy(),
        affinities_scattered.float().numpy(),
        topk_idx.numpy().astype(np.uint32),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_inputs(T, rng):
    """Create deterministic bf16-range inputs."""
    x_f32 = rng.standard_normal((T, H)).astype(np.float32) * 0.1
    w_f32 = rng.standard_normal((H, E)).astype(np.float32) * 0.02
    # Cast to bf16 values (to match hardware precision)
    x_bf16 = x_f32.astype(np.float32)
    w_bf16 = w_f32.astype(np.float32)
    return x_bf16, w_bf16


def run_nki_kernel(x_np, w_np, T):
    """Run the NKI kernel on XLA device, return outputs as numpy."""
    x_t = torch.tensor(x_np, dtype=torch.bfloat16).to("xla")
    w_t = torch.tensor(w_np, dtype=torch.bfloat16).to("xla")

    # Allocate output tensors on XLA device
    logits_t = torch.zeros((T, E), dtype=torch.float32).to("xla")
    affi_t = torch.zeros((T, E), dtype=torch.float32).to("xla")
    idx_t = torch.zeros((T, K), dtype=torch.int32).to("xla")

    # Launch with LNC=2
    logits_t, affi_t, idx_t = qwen3_router_topk_cte[2](x_t, w_t, logits_t, affi_t, idx_t)

    xm.mark_step()

    return (
        logits_t.cpu().float().numpy(),
        affi_t.cpu().float().numpy(),
        idx_t.cpu().numpy().astype(np.uint32),
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def test_case(T, rng, label=""):
    print(f"\n{'='*60}")
    print(f"Test: T={T}  {label}")
    print(f"{'='*60}")

    x_np, w_np = make_inputs(T, rng)
    ref_logits, ref_affi, ref_idx = pytorch_reference(x_np, w_np)
    nki_logits, nki_affi, nki_idx = run_nki_kernel(x_np, w_np, T)

    # --- router_logits ---
    logits_diff = np.abs(nki_logits - ref_logits)
    print(f"router_logits   max_diff={logits_diff.max():.3e}  "
          f"mean_diff={logits_diff.mean():.3e}")
    np.testing.assert_allclose(nki_logits, ref_logits, rtol=1e-2, atol=1e-2,
                                err_msg="router_logits mismatch")
    print("router_logits   PASS")

    # --- expert_affinities (scattered) ---
    # Compare only the non-zero positions (top-K positions can have small numerical
    # diffs due to bf16 matmul ordering; zero positions must be exactly zero)
    affi_diff = np.abs(nki_affi - ref_affi)
    print(f"expert_affinities max_diff={affi_diff.max():.3e}  "
          f"mean_diff={affi_diff.mean():.3e}")
    np.testing.assert_allclose(nki_affi, ref_affi, rtol=1e-2, atol=1e-3,
                                err_msg="expert_affinities mismatch")
    print("expert_affinities PASS")

    # --- expert_index ---
    # Top-K indices can differ when affinities are nearly equal (tie-breaking).
    # We verify that:
    #   (a) The set of selected experts per token is identical (no ties), OR
    #   (b) The corresponding affinity values are numerically very close (ties)
    idx_match = (nki_idx == ref_idx).all(axis=-1)  # [T] bool
    mismatched_tokens = np.where(~idx_match)[0]
    if len(mismatched_tokens) > 0:
        for t in mismatched_tokens:
            nki_set = set(nki_idx[t].tolist())
            ref_set = set(ref_idx[t].tolist())
            diff_experts = nki_set.symmetric_difference(ref_set)
            for e in diff_experts:
                assert abs(ref_affi[t, e]) < 1e-4, (
                    f"Token {t}: expert index mismatch with non-negligible affinity "
                    f"diff_experts={diff_experts} ref_affi={ref_affi[t, e]:.4e}"
                )
        print(f"expert_index    {len(mismatched_tokens)} tie-broken tokens (OK)  PASS")
    else:
        print("expert_index    PASS  (exact match)")

    print(f"\nmax_diff={affi_diff.max():.2e}  PASS")
    return affi_diff.max()


def main():
    rng = np.random.default_rng(42)

    max_diffs = []
    # Standard Qwen3 CTE shape (B=1, S=640)
    max_diffs.append(test_case(T=640, rng=rng, label="standard CTE B=1 S=640"))

    print(f"\n{'='*60}")
    print("All tests PASSED")
    print(f"Worst case max_diff = {max(max_diffs):.2e}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
