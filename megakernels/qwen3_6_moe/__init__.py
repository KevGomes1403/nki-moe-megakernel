"""Qwen3.6-35B-A3B NxDI port (text-only)."""

from .modeling_qwen36_a3b import (
    HybridDeltaNetCacheManager,
    NeuronGatedDeltaNet,
    NeuronMoEBlock,
    NeuronMTPHead,
    NeuronQwen36A3BAttention,
    NeuronQwen36A3BDecoderLayer,
    NeuronQwen36A3BForCausalLM,
    NeuronQwen36A3BModel,
    Qwen36A3BInferenceConfig,
    Qwen36A3BMRoPEEmbedding,
    convert_qwen36_a3b_hf_to_neuron_state_dict,
)

__all__ = [
    "Qwen36A3BInferenceConfig",
    "Qwen36A3BMRoPEEmbedding",
    "NeuronGatedDeltaNet",
    "NeuronQwen36A3BAttention",
    "NeuronMoEBlock",
    "NeuronMTPHead",
    "NeuronQwen36A3BDecoderLayer",
    "NeuronQwen36A3BModel",
    "NeuronQwen36A3BForCausalLM",
    "HybridDeltaNetCacheManager",
    "convert_qwen36_a3b_hf_to_neuron_state_dict",
]
