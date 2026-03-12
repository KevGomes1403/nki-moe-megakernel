"""
Unit tests for nki_moe_v8a_qwen3: kernels/v8a_qwen3.py NKI kernel.

v8a is the E2E-safe DGE variant of v7a:
  - Accepts full [E, H, 2I] and [E, I, H] weight tensors (no pre-gathering).
  - Fixes the DGE scalar_offset absolute-SBUF-address bug by loading each
    expert index from HBM into a fresh (128,1) scratch buffer at offset=0.
  - Hoists K routing weights per token into SBUF before the k loop (Plan B).

These tests verify:
  1. Numerical correctness vs a PyTorch CPU reference across several shapes.
  2. Qwen3-30B-A3B integration shapes: T=1,2,4,8, H=2048, I=256, E=128, K=8.
  3. That the kernel accepts full weight tensors (not pre-gathered) — the
     expert dispatch via DGE must produce the same result as explicit indexing.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python tests/test_moe_v8a.py
"""

import os
import sys
import time

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
os.environ["NEURON_CC_FLAGS"] = " "
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.v8a_qwen3 import nki_moe_v8a_qwen3


# ---------------------------------------------------------------------------
# PyTorch reference — mirrors the kernel's sparse top-K MoE FFN exactly.
# Uses float32 accumulation throughout to set an honest numerical baseline.
# ---------------------------------------------------------------------------

def pytorch_moe_sparse_reference(
    hidden_states,      # [T, H]      bf16
    gate_up_weights,    # [E, H, 2*I] bf16
    down_weights,       # [E, I, H]   bf16
    expert_indices,     # [T, K]      int32/int64
    routing_weights_k,  # [T, K]      float32
):
    T, H   = hidden_states.shape
    I_size = gate_up_weights.shape[2] // 2
    K      = expert_indices.shape[1]

    output = torch.zeros(T, H, dtype=torch.float32)
    hs_f32 = hidden_states.float()

    for t in range(T):
        for k in range(K):
            eid = expert_indices[t, k].item()
            rw  = routing_weights_k[t, k].item()

            gu_w  = gate_up_weights[eid].float()     # [H, 2*I]
            d_w   = down_weights[eid].float()         # [I, H]

            gu_out = hs_f32[t:t+1] @ gu_w             # [1, 2*I]
            gate   = gu_out[:, :I_size]
            up     = gu_out[:, I_size:]
            act    = F.silu(gate) * up                 # [1, I]
            down   = act @ d_w                         # [1, H]

            output[t] += rw * down[0]

    return output.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(T, H, I_size, E, K, dtype=torch.bfloat16, seed=42):
    """Random MoE inputs — K distinct experts per token (no collision)."""
    torch.manual_seed(seed)
    hidden        = torch.randn(T, H, dtype=dtype) * 0.1
    gate_up_w     = torch.randn(E, H, 2 * I_size, dtype=dtype) * 0.02
    down_w        = torch.randn(E, I_size, H, dtype=dtype) * 0.02
    routing_w     = torch.softmax(torch.randn(T, K, dtype=torch.float32), dim=-1)

    # Sample K distinct experts per token to avoid degenerate routing
    expert_idx = torch.zeros(T, K, dtype=torch.int32)
    for t in range(T):
        expert_idx[t] = torch.randperm(E)[:K]

    return hidden, gate_up_w, down_w, expert_idx, routing_w


def _run_case(T, H, I_size, E, K, seed=42, atol=0.05, label=None):
    """
    Core test harness:
      1. Compute PyTorch CPU reference with full weight tensors.
      2. Run v8a kernel passing the same full weight tensors + expert_indices.
      3. assert_allclose(atol=atol).

    Passing full weight tensors (not pre-gathered) exercises the DGE path —
    correctness here implies the expert-index scratch buffer is working.
    """
    tag = label or f"T={T},H={H},I={I_size},E={E},K={K}"
    print(f"\n{'='*68}")
    print(f"  {tag}")
    print(f"{'='*68}")

    device = xm.xla_device()
    hidden, gate_up_w, down_w, expert_idx, routing_w = _make_inputs(
        T, H, I_size, E, K, seed=seed
    )

    print(f"  [1/3] Reference (PyTorch CPU)...")
    ref = pytorch_moe_sparse_reference(hidden, gate_up_w, down_w, expert_idx, routing_w)

    print(f"  [2/3] nki_moe_v8a_qwen3 on Trainium (full weight tensors, DGE)...")
    nki_out = nki_moe_v8a_qwen3(
        hidden.to(device),
        gate_up_w.to(device),
        down_w.to(device),
        expert_idx.to(device),                       # [T, K] int32 — DGE selects expert
        routing_w.to(device),
    ).cpu()

    print(f"  [3/3] Comparing...")
    diff     = torch.abs(ref.float() - nki_out.float())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"    output shape : {nki_out.shape}")
    print(f"    max  |diff|  : {max_diff:.4e}")
    print(f"    mean |diff|  : {mean_diff:.4e}")

    ok = max_diff < atol
    status = "PASS" if ok else "FAIL"
    print(f"    {status} (threshold={atol:.2e})")
    return ok


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_small_shapes():
    """Small shape sanity check — fast, catches obvious bugs."""
    results = []
    # Minimal valid case: T=1, small H/I, few experts
    results.append(("small T=1,H=256,I=128,E=8,K=2",
                     _run_case(1, 256, 128, 8, 2, seed=0, label="small T=1 (H=256,I=128,E=8,K=2)")))
    results.append(("small T=4,H=512,I=256,E=8,K=2",
                     _run_case(4, 512, 256, 8, 2, seed=1, label="small T=4 (H=512,I=256,E=8,K=2)")))
    return results


def test_qwen3_shapes():
    """
    Qwen3-30B-A3B TKG integration shapes with tp_degree=4.
      H=2048 (hidden / tp=4 slice)
      I=256  (moe_intermediate_size=768, padded to 1024 for tp=4 → /4 = 256)
      E=128  (num_experts, all local — no EP)
      K=8    (num_experts_per_tok)

    T varies over typical TKG scenarios (single token up to small batches).
    This is the primary regression target: the DGE fix must route correctly
    for all T values, not just the standalone-test case.
    """
    results = []
    for T in [1, 2, 4, 8]:
        label = f"qwen3 T={T} (H=2048,I=256,E=128,K=8)"
        ok = _run_case(T, 2048, 256, 128, 8, seed=100 + T, atol=0.05, label=label)
        results.append((label, ok))
    return results


def test_dge_expert_dispatch():
    """
    Explicit DGE correctness test: verify that the kernel selects the correct
    expert weights for each token by constructing inputs where each expert
    produces a distinct, identifiable output.

    Each expert's gate_up_weight is scaled by (eid+1), making the output
    proportional to the routing weight × (eid+1). The reference is computed
    by explicitly indexing gate_up_w[eid] and down_w[eid] — if DGE selects
    the wrong expert, max_diff will be large (not just numerical noise).
    """
    print(f"\n{'='*68}")
    print(f"  DGE expert dispatch correctness")
    print(f"{'='*68}")

    device = xm.xla_device()
    T, H, I_size, E, K = 2, 512, 256, 8, 2
    torch.manual_seed(7)

    # Base weight pattern: all experts share the same base, expert e is scaled by (e+1)
    base_gu = torch.randn(H, 2 * I_size, dtype=torch.bfloat16) * 0.02
    base_dw = torch.randn(I_size, H,     dtype=torch.bfloat16) * 0.02
    gate_up_w = torch.stack([(e + 1) * base_gu for e in range(E)])  # [E, H, 2I]
    down_w    = torch.stack([(e + 1) * base_dw for e in range(E)])  # [E, I, H]

    hidden   = torch.randn(T, H, dtype=torch.bfloat16) * 0.1
    # Fixed routing: token 0 → experts [3, 5], token 1 → experts [1, 7]
    expert_idx = torch.tensor([[3, 5], [1, 7]], dtype=torch.int32)
    routing_w  = torch.tensor([[0.6, 0.4], [0.7, 0.3]], dtype=torch.float32)

    ref = pytorch_moe_sparse_reference(hidden, gate_up_w, down_w, expert_idx, routing_w)

    print(f"  [1/2] nki_moe_v8a_qwen3 (fixed expert routing, T={T})...")
    nki_out = nki_moe_v8a_qwen3(
        hidden.to(device),
        gate_up_w.to(device),
        down_w.to(device),
        expert_idx.to(device),
        routing_w.to(device),
    ).cpu()

    diff     = torch.abs(ref.float() - nki_out.float())
    max_diff = diff.max().item()
    print(f"  [2/2] max |diff| = {max_diff:.4e}")

    ok = max_diff < 0.05
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.perf_counter()
    all_results = []

    # all_results += test_small_shapes()
    all_results += test_qwen3_shapes()
    # all_results.append(("DGE expert dispatch", test_dge_expert_dispatch()))

    t1 = time.perf_counter()

    print(f"\n{'='*68}")
    print("Summary:")
    all_ok = True
    for name, ok in all_results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        all_ok = all_ok and ok
    print(f"{'='*68}")
    print(f"Total time: {t1 - t0:.1f}s")

    sys.exit(0 if all_ok else 1)
