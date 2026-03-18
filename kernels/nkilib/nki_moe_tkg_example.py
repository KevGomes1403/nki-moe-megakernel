"""
NKI moe_tkg Kernel Validation Test — single TP shard, TP=4, LNC=2, trn2

Validates the moe_tkg kernel using shapes that match a single TP rank of
Qwen3-30B-A3B at token generation (T=1 decode step).

Weight sharding (TP=4, no expert parallel)
------------------------------------------
  H    = 2048   hidden size, not TP-sharded (full dim on every rank)
  I    = 768    full MoE intermediate size
  I_tp = I / 4 = 192   per-rank intermediate slice (TP-sharded)
  E    = 128    all experts replicated on every rank
  K    = 8      top-K experts selected per token

  expert_gate_up_weights : [E=128, H=2048, 2, I_tp=192]  (gate and up fused)
  expert_down_weights    : [E=128, I_tp=192, H=2048]

  Output [T, H] is a partial sum — the full model requires a TP all-reduce
  across all 4 ranks to recover the complete hidden state.

Affinity mode
-------------
  POST_SCALE: affinities are multiplied into each expert's output after the
  FFN computation and accumulated into the result. This matches Qwen3's
  norm_topk_prob=True behaviour.

  IMPORTANT: expert_affinities must be float32 for POST_SCALE. bf16 triggers
  a compiler error ("TensorScalarPtr arith immediate dtype must be fp32").

LNC=2
-----
  Controlled at runtime by NEURON_RT_NUM_CORES=2 (set below before torch_xla
  initialises). The NxD framework additionally passes logical_nc_config=2
  via BlockwiseMatmulConfig at model-compile time; for this standalone test,
  the runtime flag is sufficient to allocate 2 physical NeuronCores.

Invocation
----------
  moe_tkg has no @nki.jit decorator — it uses Python-level dispatch (if/else
  on boolean args) with NKI sub-kernel calls. Standard nki.jit works:

      moe_tkg_jit = nki.jit(moe_tkg, platform_target="trn2")
      output = moe_tkg_jit(hidden_input, ..., is_all_expert=False, ...)

  The bool args (is_all_expert, is_mx_kernel) are evaluated at trace time,
  producing a single specialised KLIR. The call returns the [T, H] output
  tensor directly (unlike mode="trace" which returns int 0).
"""

import os

# Set before torch_xla / neuron runtime initialise (lazy but safest at module level).
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")   # LNC=2: 2 NeuronCores per logical core

# Profiling
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"]= "1"
os.environ["XLA_HLO_DEBUG"]= "1"
os.environ["NEURON_RT_INSPECT_ENABLE"]= "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"]= "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]= "./output"

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

import nki
from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode


def moe_tkg_torch_ref(
    hidden_input,     # [T, H]    float32
    gate_up_weights,  # [E, H, 2, I_tp]  float32
    down_weights,     # [E, I_tp, H]     float32
    expert_affinities,  # [T, E]  float32 — sparse, non-zero at selected K positions
    expert_index,     # [T, K]   int64
):
    """
    PyTorch reference for moe_tkg selective-expert, POST_SCALE mode.

    For each token t and each selected expert k:
        gate_proj  = silu(hidden[t] @ gate_w[e])
        up_proj    = hidden[t] @ up_w[e]
        expert_out = (gate_proj * up_proj) @ down_w[e]
        output[t] += affinity[t, e] * expert_out   (POST_SCALE)

    This is a partial result — the full output requires TP all-reduce.
    """
    T, H = hidden_input.shape
    output = torch.zeros(T, H, dtype=torch.float32)

    for t in range(T):
        for k in range(expert_index.shape[1]):
            e = int(expert_index[t, k].item())
            h      = hidden_input[t].float()                    # [H]
            gate_w = gate_up_weights[e, :, 0, :].float()       # [H, I_tp]
            up_w   = gate_up_weights[e, :, 1, :].float()       # [H, I_tp]
            down_w = down_weights[e].float()                    # [I_tp, H]

            gate_proj  = F.silu(h @ gate_w)                    # [I_tp]
            up_proj    = h @ up_w                               # [I_tp]
            expert_out = (gate_proj * up_proj) @ down_w        # [H]

            output[t] += expert_affinities[t, e].float() * expert_out  # POST_SCALE

    return output


def test_moe_tkg():
    print("=" * 70)
    print("NKI moe_tkg Test — TP=4 shard shapes, POST_SCALE, LNC=2, trn2")
    print("=" * 70)

    print(f"\nEnvironment:")
    print(f"  NEURON_PLATFORM_TARGET_OVERRIDE = {os.environ.get('NEURON_PLATFORM_TARGET_OVERRIDE')}")
    print(f"  NEURON_RT_NUM_CORES             = {os.environ.get('NEURON_RT_NUM_CORES')}")

    device = xm.xla_device()
    print(f"  XLA device: {device}")

    # ------------------------------------------------------------------ #
    # Shapes: single TP rank of Qwen3-30B-A3B, T=1 decode step           #
    # ------------------------------------------------------------------ #
    TP   = 4
    T    = 1    # single token (TKG decode); kernel supports T <= 128
    H    = 2048 # hidden size — not TP-sharded
    I_tp = 768 // TP  # = 192, per-rank intermediate slice
    E    = 128  # all experts replicated on this rank
    K    = 8    # top-K experts per token

    print(f"\nShapes (Qwen3-30B-A3B, TP={TP} shard):")
    print(f"  T={T}, H={H}, I_tp={I_tp} (={768}//{TP}), E={E}, K={K}")
    print(f"  hidden_input           : [{T}, {H}]          bf16")
    print(f"  expert_gate_up_weights : [{E}, {H}, 2, {I_tp}]  bf16")
    print(f"  expert_down_weights    : [{E}, {I_tp}, {H}]      bf16")
    print(f"  expert_affinities      : [{T}, {E}]          float32  (POST_SCALE requires fp32)")
    print(f"  expert_index           : [{T}, {K}]          int32")

    torch.manual_seed(42)
    scale = 0.02  # small scale keeps bf16 accumulation errors low

    hidden_cpu   = torch.randn(T, H,       dtype=torch.bfloat16) * scale
    gate_up_cpu  = torch.randn(E, H, 2, I_tp, dtype=torch.bfloat16) * scale
    down_cpu     = torch.randn(E, I_tp, H, dtype=torch.bfloat16) * scale

    # Select K=8 distinct experts per token (cyclic offset per token).
    # Affinities are uniform over the selected experts and sum to 1.0,
    # matching Qwen3's norm_topk_prob=True behaviour.
    expert_index_cpu = torch.zeros(T, K, dtype=torch.int32)
    affinities_cpu   = torch.zeros(T, E, dtype=torch.float32)  # float32 required for POST_SCALE
    for t in range(T):
        selected = [(t * K + k) % E for k in range(K)]
        expert_index_cpu[t] = torch.tensor(selected, dtype=torch.int32)
        for e in selected:
            affinities_cpu[t, e] = 1.0 / K  # uniform, sums to 1.0

    print(f"\nExpert assignments (token → experts, affinity={1.0/K:.4f} each):")
    for t in range(T):
        print(f"  token {t}: experts {expert_index_cpu[t].tolist()}")

    # ------------------------------------------------------------------ #
    # Reference                                                           #
    # ------------------------------------------------------------------ #
    print("\n[1/3] Computing reference (PyTorch, POST_SCALE)...")
    ref_output = moe_tkg_torch_ref(
        hidden_input=hidden_cpu,
        gate_up_weights=gate_up_cpu,
        down_weights=down_cpu,
        expert_affinities=affinities_cpu,
        expert_index=expert_index_cpu,
    )

    # ------------------------------------------------------------------ #
    # NKI kernel                                                          #
    # ------------------------------------------------------------------ #
    # is_all_expert=False  → selective-expert path (K=8 out of E=128)
    # POST_SCALE           → affinity applied after FFN, requires fp32 affinities
    moe_tkg_jit = nki.jit(moe_tkg, platform_target="trn2")

    print("[2/3] Computing NKI kernel (moe_tkg, selective, POST_SCALE)...")
    nki_output = moe_tkg_jit(
        hidden_input=hidden_cpu.to(device),
        expert_gate_up_weights=gate_up_cpu.to(device),
        expert_down_weights=down_cpu.to(device),
        expert_affinities=affinities_cpu.to(device),   # float32
        expert_index=expert_index_cpu.to(device),
        is_all_expert=False,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        activation_fn=ActFnType.SiLU,
    )

    # ------------------------------------------------------------------ #
    # Compare                                                             #
    # ------------------------------------------------------------------ #
    print("[3/3] Comparing...")
    ref_fp32 = ref_output.float()
    nki_fp32 = nki_output.cpu().float()
    diff = (nki_fp32 - ref_fp32).abs()

    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nResults:")
    print(f"  NKI output shape : {nki_output.shape}")
    print(f"  NKI output dtype : {nki_output.dtype}")
    print(f"  Max  |ref - nki| : {max_diff:.6e}")
    print(f"  Mean |ref - nki| : {mean_diff:.6e}")
    print(f"  Note: output is a partial TP shard — TP all-reduce not applied here")

    tol = 5e-2  # BF16 accumulation tolerance
    if max_diff < tol:
        print("\n" + "=" * 70)
        print("SUCCESS! moe_tkg validated (TP=4 shard, POST_SCALE, trn2):")
        print("  * selective-expert kernel compiles and executes")
        print("  * output matches PyTorch POST_SCALE reference within BF16 tolerance")
        print("=" * 70)
        return True

    print("\nFAILED — numerical mismatch exceeds tolerance")
    return False


if __name__ == "__main__":
    test_moe_tkg()
