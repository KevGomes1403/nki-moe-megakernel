"""
Unit test for nki_moe_v6b_qwen3: kernels/v6b_qwen3.py NKI kernel.

Tests that the kernel produces numerically correct results vs a PyTorch reference
that performs the same router -> expert FFN computation.

The kernel expects pre-gathered expert weights [T*K, H, 2*I] and [T*K, I, H].
Routing (top-k selection) and weight gathering are done in PyTorch before
invoking the NKI kernel for the expert FFN.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python tests/test_moe_v6b_integration.py
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

from kernels.v6b_qwen3 import nki_moe_v6b_qwen3


def pytorch_moe_reference(hidden_states, router_weight, gate_up_weight, down_weight,
                           top_k, normalize):
    """
    Full MoE reference: linear router -> softmax -> topk -> expert FFN.
    Uses float64 softmax to match RouterTopK behavior.
    """
    T, H = hidden_states.shape
    I = gate_up_weight.shape[2] // 2

    # Router: cast input to float32
    logits = hidden_states.float() @ router_weight.float().T   # [T, E]

    # float64 softmax
    affinities = F.softmax(logits, dim=1, dtype=torch.float64)  # [T, E]
    _, expert_index = torch.topk(logits, top_k, dim=1)          # [T, K]
    routing_weights = torch.gather(affinities, 1, expert_index)  # [T, K] float64

    if normalize:
        routing_weights = F.normalize(routing_weights.float(), p=1.0, dim=1)
    routing_weights = routing_weights.float()

    # Expert FFN
    output = torch.zeros(T, H, dtype=torch.float32)
    hs_f32 = hidden_states.float()

    for t in range(T):
        for k in range(top_k):
            eid = expert_index[t, k].item()
            rw = routing_weights[t, k].item()

            gu_w = gate_up_weight[eid].float()   # [H, 2*I]
            d_w = down_weight[eid].float()        # [I, H]

            gu_out = hs_f32[t:t+1] @ gu_w         # [1, 2*I]
            gate = gu_out[:, :I]
            up = gu_out[:, I:]
            act = F.silu(gate) * up                # [1, I]
            down = act @ d_w                        # [1, H]

            output[t] += rw * down[0]

    return output.to(hidden_states.dtype), expert_index, routing_weights


def gather_expert_weights(gate_up_weight, down_weight, expert_index):
    """
    Gather K expert weights per token for the pre-gathered kernel interface.

    Args:
        gate_up_weight: [E, H, 2*I] bf16
        down_weight:    [E, I, H]   bf16
        expert_index:   [T, K]      int64

    Returns:
        gate_up_gathered: [T*K, H, 2*I] bf16
        down_gathered:    [T*K, I, H]   bf16
    """
    flat_indices = expert_index.reshape(-1)          # [T*K]
    gate_up_gathered = gate_up_weight[flat_indices]  # [T*K, H, 2*I]
    down_gathered = down_weight[flat_indices]        # [T*K, I, H]
    return gate_up_gathered, down_gathered


def test_v6b_qwen3_kernel(T, H, I, E, K, normalize=True, seed=42):
    """Test nki_moe_v6b_qwen3 against PyTorch reference."""
    print(f"\n{'='*70}")
    print(f"v6b_qwen3 kernel Test: T={T}, H={H}, I={I}, E={E}, K={K}, normalize={normalize}")
    print(f"{'='*70}")

    device = xm.xla_device()
    torch.manual_seed(seed)
    scale = 0.02

    router_weight  = torch.randn(E, H, dtype=torch.float32) * scale
    gate_up_weight = torch.randn(E, H, 2*I, dtype=torch.bfloat16) * scale
    down_weight    = torch.randn(E, I, H, dtype=torch.bfloat16) * scale
    hidden         = torch.randn(T, H, dtype=torch.bfloat16) * 0.1

    # Reference — also provides expert_index and routing_weights for the kernel
    print("[1/3] Computing reference (PyTorch CPU)...")
    ref_output, expert_index, routing_weights = pytorch_moe_reference(
        hidden, router_weight, gate_up_weight, down_weight, K, normalize
    )
    print(f"  Expert assignments: {expert_index.tolist()}")

    # Gather weights: [T*K, H, 2*I] and [T*K, I, H]
    gate_up_gathered, down_gathered = gather_expert_weights(
        gate_up_weight, down_weight, expert_index
    )

    # Run kernel (already @nki.jit decorated — call directly)
    print("[2/3] Running nki_moe_v6b_qwen3 on device...")
    nki_output = nki_moe_v6b_qwen3(
        hidden.to(device),
        gate_up_gathered.to(device),
        down_gathered.to(device),
        routing_weights.to(device),
    )
    nki_output = nki_output.cpu()

    # Compare
    print("[3/3] Comparing...")
    diff = torch.abs(ref_output.float() - nki_output.float())
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\n  Output shape:        {nki_output.shape}")
    print(f"  Max  |output diff|:  {max_diff:.6e}")
    print(f"  Mean |output diff|:  {mean_diff:.6e}")

    threshold = 0.1
    if max_diff < threshold:
        print(f"\n  PASS (max_diff={max_diff:.2e} < {threshold})")
        return True
    else:
        print(f"\n  FAIL (max_diff={max_diff:.4e} > {threshold})")
        return False


def test_v6b_qwen3_3d_input(T, H, I, E, K, seed=99):
    """Test that 3D input [B, 1, H] works (reshaped to 2D for kernel, back to 3D after)."""
    print(f"\n{'='*70}")
    print(f"v6b_qwen3 3D Input Test: B={T}, S=1, H={H}")
    print(f"{'='*70}")

    device = xm.xla_device()
    torch.manual_seed(seed)
    scale = 0.02

    router_weight  = torch.randn(E, H, dtype=torch.float32) * scale
    gate_up_weight = torch.randn(E, H, 2*I, dtype=torch.bfloat16) * scale
    down_weight    = torch.randn(E, I, H, dtype=torch.bfloat16) * scale
    hidden_3d      = torch.randn(T, 1, H, dtype=torch.bfloat16) * 0.1

    # Kernel requires [T, H]; reshape [T, 1, H] → [T, H]
    hidden_2d = hidden_3d.reshape(T, H)

    # Routing on 2D hidden
    _, expert_index, routing_weights = pytorch_moe_reference(
        hidden_2d, router_weight, gate_up_weight, down_weight, K, normalize=True
    )
    gate_up_gathered, down_gathered = gather_expert_weights(
        gate_up_weight, down_weight, expert_index
    )

    output_2d = nki_moe_v6b_qwen3(
        hidden_2d.to(device),
        gate_up_gathered.to(device),
        down_gathered.to(device),
        routing_weights.to(device),
    )
    output = output_2d.cpu().reshape(T, 1, H)

    assert output.shape == (T, 1, H), f"Expected shape ({T}, 1, {H}), got {output.shape}"
    assert output.dtype == torch.bfloat16
    assert not torch.all(output == 0), "Output is all zeros"

    print(f"  Output shape: {output.shape}  OK")
    print(f"  Output range: [{output.float().min():.4f}, {output.float().max():.4f}]")
    print(f"\n  PASS")
    return True


if __name__ == "__main__":
    t0 = time.perf_counter()
    results = []

    # results.append(("small (T=4,H=256,I=256,E=8,K=2)",
    #                  test_v6b_qwen3_kernel(T=4, H=256, I=256, E=8, K=2, normalize=True)))

    results.append(("qwen3 (T=1,H=2048,I=256,E=128,K=8)",
                     test_v6b_qwen3_kernel(T=1, H=2048, I=256, E=128, K=8, normalize=True, seed=123)))

    # results.append(("3D input (B=4,S=1)",
                    #  test_v6b_qwen3_3d_input(T=4, H=256, I=256, E=8, K=2)))

    t1 = time.perf_counter()

    print(f"\n{'='*70}")
    print("Summary:")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        all_ok = all_ok and ok
    print(f"{'='*70}")
    print(f"Total time: {t1-t0:.1f}s")

    sys.exit(0 if all_ok else 1)
