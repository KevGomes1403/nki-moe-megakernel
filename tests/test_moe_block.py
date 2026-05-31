"""Hardware block test for NeuronMoEBlock.

Compares NxDI's compiled MoE block against the HF reference
(Qwen3_5MoeSparseMoeBlock) on trn3 with TP=4, LNC=2.

Reference is vendored at models/qwen3_6_moe/_hf_reference/ from
huggingface/transformers main; the dimensions are deliberately tiny so the
test runs in a few minutes.
"""

import os
import sys
from pathlib import Path

import torch

# Repo root on sys.path so `models.qwen3_6_moe.*` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Vendored HF reference (patched for transformers 4.57.6 compat).
_HF_REF = Path(__file__).resolve().parent.parent / "_hf_reference"
if str(_HF_REF) not in sys.path:
    sys.path.insert(0, str(_HF_REF))

from block_testing_utils import test_block_correctness
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig

from models.qwen3_6_moe.modeling_qwen36_a3b import (
    NeuronMoEBlock,
    Qwen36A3BInferenceConfig,
)
from modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
from configuration_qwen3_5_moe import Qwen3_5MoeConfig


# ---------------------------------------------------------------------------
# Tiny shape for fast turnaround. A3B-shaped (256 experts, top-8) would
# compile for tens of minutes; this exercises the same code paths.
# ---------------------------------------------------------------------------

BATCH = 1
SEQ_LEN = 16
HIDDEN = 64
HEAD_DIM = 16
MOE_INTERMEDIATE = 128
SHARED_INTERMEDIATE = 128
NUM_EXPERTS = 4
TOP_K = 2

TP_DEGREE = 4
LNC = 2


def _build_neuron_config() -> Qwen36A3BInferenceConfig:
    nc = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        logical_nc_config=LNC,
    )
    return Qwen36A3BInferenceConfig(
        neuron_config=nc,
        # text-stack shape -- tiny
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_theta=10_000,
        hidden_act="silu",
        # DeltaNet -- not exercised here but config validator requires
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        # MoE
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=MOE_INTERMEDIATE,
        shared_expert_intermediate_size=SHARED_INTERMEDIATE,
        norm_topk_prob=True,
        # MTP off for this test
        mtp_num_hidden_layers=0,
    )


def _build_hf_reference_config() -> Qwen3_5MoeConfig:
    return Qwen3_5MoeConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_theta=10_000,
        hidden_act="silu",
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=MOE_INTERMEDIATE,
        shared_expert_intermediate_size=SHARED_INTERMEDIATE,
        norm_topk_prob=True,
    )


# ---------------------------------------------------------------------------
# Weight sync. The HF reference holds fused expert tensors with HF's layout;
# NxDI uses the same fused layout but with transposed inner dims. The shared
# expert + sigmoid gate live under matching key names so they pass through.
# ---------------------------------------------------------------------------

WEIGHT_MAPPING = {
    # Router
    "gate.weight": "moe.router.linear_router.weight",
    # Fused routed experts -- shape transform handled in sync_fn below.
    "experts.gate_up_proj": "moe.expert_mlps.mlp_op.gate_up_proj.weight",
    "experts.down_proj": "moe.expert_mlps.mlp_op.down_proj.weight",
    # Shared expert (owned by NeuronMoEBlock)
    "shared_expert.gate_proj.weight": "shared_expert.gate_proj.weight",
    "shared_expert.up_proj.weight": "shared_expert.up_proj.weight",
    "shared_expert.down_proj.weight": "shared_expert.down_proj.weight",
    # Sigmoid gate over the shared expert
    "shared_expert_gate.weight": "shared_expert_gate.weight",
}


def sync_moe_weights(reference_block, checkpoint_path, weight_mapping, verbose):
    """HF expert layout -> NxDI expert layout transpose.

    HF Qwen3_5MoeExperts:
        gate_up_proj  shape (E, 2*I, H)
        down_proj     shape (E, H, I)
    NxDI ExpertMLPsV2:
        gate_up_proj  shape (E, H, 2*I)
        down_proj     shape (E, I, H)
    Both other weights pass through with the standard sync.
    """
    neuron_sd = torch.load(checkpoint_path)
    ref_sd = reference_block.state_dict()
    count = 0

    for ref_key, neuron_key in weight_mapping.items():
        if not neuron_key.startswith("block."):
            neuron_key = f"block.{neuron_key}"
        if ref_key not in ref_sd or neuron_key not in neuron_sd:
            if verbose:
                print(f"  skip {ref_key} -> {neuron_key}: missing")
            continue

        t = ref_sd[ref_key].to(neuron_sd[neuron_key].dtype).contiguous().clone()

        # Transpose expert tensors into NxDI's layout.
        if ref_key == "experts.gate_up_proj":
            # (E, 2I, H) -> (E, H, 2I)
            t = t.transpose(1, 2).contiguous()
        elif ref_key == "experts.down_proj":
            # (E, H, I) -> (E, I, H)
            t = t.transpose(1, 2).contiguous()

        if t.shape != neuron_sd[neuron_key].shape:
            if verbose:
                print(f"  shape mismatch {ref_key} {t.shape} vs {neuron_key} {neuron_sd[neuron_key].shape}")
            continue

        neuron_sd[neuron_key] = t
        count += 1
        if verbose:
            print(f"  synced {ref_key} -> {neuron_key} {tuple(t.shape)}")

    torch.save(neuron_sd, checkpoint_path)
    return count


# ---------------------------------------------------------------------------
# Test inputs. Distinct per-position values so the router actually has work
# to do across tokens (per the block-translator rule that structured inputs
# must exercise the structure under test).
# ---------------------------------------------------------------------------


def _make_inputs():
    torch.manual_seed(123)
    sample = torch.randn(BATCH, SEQ_LEN, HIDDEN, dtype=torch.bfloat16) * 0.5
    example = torch.zeros(BATCH, SEQ_LEN, HIDDEN, dtype=torch.bfloat16)
    return [(example,)], [(sample,)], [(sample,)]


def main():
    example_inputs, test_inputs, reference_inputs = _make_inputs()

    cfg = _build_neuron_config()
    hf_cfg = _build_hf_reference_config()

    test_block_correctness(
        neuron_block_class=NeuronMoEBlock,
        pytorch_block_class=Qwen3_5MoeSparseMoeBlock,
        weight_mapping=WEIGHT_MAPPING,
        config=cfg,
        pytorch_init_kwargs={"config": hf_cfg},
        example_inputs=example_inputs,
        test_inputs=test_inputs,
        reference_inputs=reference_inputs,
        checkpoint_name="qwen36_a3b_moe_block.pt",
        seed=42,
        use_moe=True,
        verbose=True,
        sync_weights_fn=sync_moe_weights,
    )


if __name__ == "__main__":
    main()
