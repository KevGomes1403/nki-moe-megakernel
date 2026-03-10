"""
NKI moe_tkg Kernel Validation Test

Validates the moe_tkg (Mixture of Experts token generation) kernel from
nkilib/core/moe/moe_tkg/moe_tkg.py against a PyTorch reference.

Invocation pattern
------------------
moe_tkg has no @nki.jit decorator but contains only Python-level dispatch (if/else
on boolean args) + NKI sub-kernel calls. Standard nki.jit (not mode="trace") works:

    moe_tkg_jit = nki.jit(moe_tkg, platform_target="trn2")
    output = moe_tkg_jit(hidden_input, ..., is_all_expert=False, ...)

The bool args (is_all_expert, is_mx_kernel) are evaluated at trace time, so the
kernel compiles to a single specialised KLIR and returns the [T, H] output tensor.

Dimensions
----------
  T    total tokens  (batch_size * seq_len, must be <= 128 for non-MX mode)
  H    hidden dimension  (must be divisible by pmax=128)
  I    intermediate (expert FFN) dimension
  E    number of experts  (E > 1 required; local == global for single-device test)
  K    top-K experts selected per token  (K <= 16)

Test configuration
------------------
  Selective-expert mode (is_all_expert=False), ExpertAffinityScaleMode.NO_SCALE
  T=4, H=512, I=512, E=4, K=2  -- small, comfortably within all hardware limits.
"""

import os

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

import nki
from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode


def moe_tkg_torch_ref(
    hidden_input,    # [T, H]  float32
    gate_up_weights, # [E, H, 2, I]  float32
    down_weights,    # [E, I, H]  float32
    expert_index,    # [T, K]  int64
):
    """
    PyTorch reference for moe_tkg selective-expert, NO_SCALE mode.

    For each token t and each selected expert k:
        gate_proj  = silu(hidden[t] @ gate_w[e])
        up_proj    = hidden[t] @ up_w[e]
        expert_out = (gate_proj * up_proj) @ down_w[e]
        output[t] += expert_out   (no affinity scaling with NO_SCALE)
    """
    T, H = hidden_input.shape
    output = torch.zeros(T, H, dtype=torch.float32)

    for t in range(T):
        for k in range(expert_index.shape[1]):
            e = int(expert_index[t, k].item())
            h = hidden_input[t].float()                    # [H]
            gate_w = gate_up_weights[e, :, 0, :].float()  # [H, I]
            up_w   = gate_up_weights[e, :, 1, :].float()  # [H, I]
            down_w = down_weights[e].float()               # [I, H]

            gate_proj  = F.silu(h @ gate_w)               # [I]
            up_proj    = h @ up_w                          # [I]
            expert_out = (gate_proj * up_proj) @ down_w   # [H]

            output[t] += expert_out

    return output


def test_moe_tkg():
    print("=" * 70)
    print("NKI moe_tkg Test (selective-expert, NO_SCALE, trn2)")
    print("=" * 70)

    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    device = xm.xla_device()
    print(f"\nUsing device: {device}")
    print(f"NEURON_PLATFORM_TARGET_OVERRIDE={os.environ['NEURON_PLATFORM_TARGET_OVERRIDE']}")

    # --- Shapes ---
    T = 4    # tokens  (<= 128 for non-MX selective mode)
    H = 512  # hidden  (divisible by pmax=128)
    I = 512  # intermediate
    E = 128    # local experts (E > 1; equals global E for single-device test)
    K = 8    # top-K experts per token (K <= 16)

    print(f"\nDimensions: T={T}, H={H}, I={I}, E={E}, K={K}")
    print(f"  hidden_input:           [{T}, {H}]  bf16")
    print(f"  expert_gate_up_weights: [{E}, {H}, 2, {I}]  bf16")
    print(f"  expert_down_weights:    [{E}, {I}, {H}]  bf16")
    print(f"  expert_affinities:      [{T}, {E}]  bf16")
    print(f"  expert_index:           [{T}, {K}]  int32")

    torch.manual_seed(42)
    scale = 0.1  # small scale for numerical stability in BF16

    hidden_cpu    = torch.randn(T, H,    dtype=torch.bfloat16) * scale
    gate_up_cpu   = torch.randn(E, H, 2, I, dtype=torch.bfloat16) * scale
    down_cpu      = torch.randn(E, I, H, dtype=torch.bfloat16) * scale
    affinities_cpu = torch.ones(T, E,   dtype=torch.bfloat16)  # uniform; not used with NO_SCALE

    # Assign each token to 2 distinct experts (cyclic assignment)
    expert_index_cpu = torch.zeros(T, K, dtype=torch.int32)
    for t in range(T):
        expert_index_cpu[t, 0] = t % E
        expert_index_cpu[t, 1] = (t + 1) % E

    print(f"\nExpert assignments (token -> experts):")
    for t in range(T):
        print(f"  token {t}: experts {expert_index_cpu[t].tolist()}")

    # --- Reference ---
    print("\n[1/3] Computing reference (PyTorch)...")
    ref_output = moe_tkg_torch_ref(
        hidden_input=hidden_cpu,
        gate_up_weights=gate_up_cpu,
        down_weights=down_cpu,
        expert_index=expert_index_cpu,
    )

    # --- NKI kernel ---
    # moe_tkg dispatches with Python-level if/else on boolean args (is_all_expert,
    # is_mx_kernel) that are evaluated at trace time, so standard nki.jit works and
    # returns the output tensor (unlike mode="trace" which returns int 0).
    moe_tkg_jit = nki.jit(moe_tkg, platform_target="trn2")

    print("[2/3] Computing NKI kernel (moe_tkg)...")
    nki_output = moe_tkg_jit(
        hidden_input=hidden_cpu.to(device),
        expert_gate_up_weights=gate_up_cpu.to(device),
        expert_down_weights=down_cpu.to(device),
        expert_affinities=affinities_cpu.to(device),
        expert_index=expert_index_cpu.to(device),
        is_all_expert=False,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.NO_SCALE,
        activation_fn=ActFnType.SiLU,
    )

    # --- Compare ---
    print("[3/3] Comparing...")
    ref_fp32 = ref_output.float()
    nki_fp32 = nki_output.cpu().float()
    diff = (nki_fp32 - ref_fp32).abs()

    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nResults:")
    print(f"  NKI output shape: {nki_output.shape}")
    print(f"  NKI output dtype: {nki_output.dtype}")
    print(f"  Max difference:   {max_diff:.6e}")
    print(f"  Mean difference:  {mean_diff:.6e}")

    tol = 5e-2  # BF16 accumulation tolerance
    if max_diff < tol:
        print("\n" + "=" * 70)
        print("SUCCESS! moe_tkg API validated on trn2:")
        print("  * moe_tkg trace kernel compiles and executes")
        print("  * Output matches PyTorch reference within BF16 tolerance")
        print("=" * 70)
        return True

    print("\nFAILED - Numerical mismatch exceeds tolerance")
    return False


if __name__ == "__main__":
    test_moe_tkg()
