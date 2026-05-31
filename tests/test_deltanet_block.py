"""Hardware block test for NeuronGatedDeltaNet.

Compares NxDI's compiled gated DeltaNet kernel against the HF reference
(Qwen3_5MoeGatedDeltaNet) on trn3 with TP=4, LNC=2.

The NKI DeltaNet kernels work in 128-token chunks (P_MAX=128 in
nki_deltanet_fused.py), so we use seq_len=128 -- exactly one chunk -- to
exercise the fused-prefill path without padding artifacts.
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
    NeuronGatedDeltaNet,
    Qwen36A3BInferenceConfig,
    reorder_deltanet_qkv_channels_for_tp,
    reorder_deltanet_qkv_for_tp,
)
from modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet
from configuration_qwen3_5_moe import Qwen3_5MoeConfig


# Dimensions -- chosen so a single fused chunk (P_MAX=128) covers the input
# while keeping head sizes large enough to be representative of A3B.
BATCH = 1
SEQ_LEN = 128
HIDDEN = 256
HEAD_DIM = 64           # GQA head_dim (unused by this block, but config requires)
NUM_V_HEADS = 8
NUM_K_HEADS = 4
LINEAR_KV_HEAD_DIM = 128  # matches the kernel's P_MAX/head_dim contract
CONV_KERNEL = 4

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
        # DeltaNet -- the block under test
        linear_num_value_heads=NUM_V_HEADS,
        linear_num_key_heads=NUM_K_HEADS,
        linear_key_head_dim=LINEAR_KV_HEAD_DIM,
        linear_value_head_dim=LINEAR_KV_HEAD_DIM,
        linear_conv_kernel_dim=CONV_KERNEL,
        # MoE -- not exercised, but required by config validator
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        norm_topk_prob=True,
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
        linear_num_value_heads=NUM_V_HEADS,
        linear_num_key_heads=NUM_K_HEADS,
        linear_key_head_dim=LINEAR_KV_HEAD_DIM,
        linear_value_head_dim=LINEAR_KV_HEAD_DIM,
        linear_conv_kernel_dim=CONV_KERNEL,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        norm_topk_prob=True,
    )


# Same names on both sides except for the conv/A_log/dt_bias rewraps and
# the TP-shuffled QKV projection -- handled in sync_fn.
WEIGHT_MAPPING = {
    "in_proj_qkv.weight": "in_proj_qkv.weight",
    "in_proj_z.weight": "in_proj_z.weight",
    "in_proj_a.weight": "in_proj_a.weight",
    "in_proj_b.weight": "in_proj_b.weight",
    "conv1d.weight": "conv1d_weight.weight",
    "A_log": "A_log_weight.weight",
    "dt_bias": "dt_bias_weight.weight",
    "norm.weight": "norm.weight",
    "out_proj.weight": "out_proj.weight",
}


def sync_deltanet_weights(reference_block, checkpoint_path, weight_mapping, verbose):
    """Apply the DeltaNet-specific shape transforms + TP-shuffle on copy."""
    neuron_sd = torch.load(checkpoint_path)
    ref_sd = reference_block.state_dict()
    count = 0

    reorder_args = (
        TP_DEGREE,
        NUM_K_HEADS,
        NUM_V_HEADS,
        LINEAR_KV_HEAD_DIM,
        LINEAR_KV_HEAD_DIM,
    )

    for ref_key, neuron_key in weight_mapping.items():
        if not neuron_key.startswith("block."):
            neuron_key = f"block.{neuron_key}"
        if ref_key not in ref_sd or neuron_key not in neuron_sd:
            if verbose:
                print(f"  skip {ref_key} -> {neuron_key}: missing")
            continue

        t = ref_sd[ref_key].to(neuron_sd[neuron_key].dtype).contiguous().clone()

        if ref_key == "conv1d.weight":
            # HF: (conv_dim, 1, K) ; NxDI: (conv_dim, K) after TP-shuffle.
            t = t.squeeze(1).contiguous()
            if TP_DEGREE > 1:
                t = reorder_deltanet_qkv_channels_for_tp(t, *reorder_args)
        elif ref_key in ("A_log", "dt_bias"):
            t = t.reshape(-1, 1).contiguous()
        elif ref_key == "in_proj_qkv.weight" and TP_DEGREE > 1:
            t = reorder_deltanet_qkv_for_tp(t, *reorder_args)

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


# The HF and NxDI forward signatures differ. HF: (x, cache_params, attention_mask).
# NxDI: (x, attention_mask, position_ids, past_key_value, ...) -> 4-tuple.
# Wrap the HF reference to take a single tensor input matching the harness.
class _HFReferenceWrapper(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0)

    def forward(self, hidden_states):
        return self.block(hidden_states, cache_params=None, attention_mask=None)

    def state_dict(self, *args, **kwargs):
        return self.block.state_dict(*args, **kwargs)


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
        neuron_block_class=NeuronGatedDeltaNet,
        pytorch_block_class=_HFReferenceWrapper,
        weight_mapping=WEIGHT_MAPPING,
        config=cfg,
        neuron_init_kwargs={"layer_idx": 0},
        pytorch_init_kwargs={"config": hf_cfg},
        example_inputs=example_inputs,
        test_inputs=test_inputs,
        reference_inputs=reference_inputs,
        checkpoint_name="qwen36_a3b_deltanet_block.pt",
        seed=42,
        use_moe=True,
        verbose=True,
        sync_weights_fn=sync_deltanet_weights,
    )


if __name__ == "__main__":
    main()
