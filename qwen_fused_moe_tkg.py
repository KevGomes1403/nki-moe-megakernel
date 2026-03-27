# coding=utf-8
"""
Qwen3 MoE model with kernel_v8a fused MoE TKG kernel.

Standalone variant of qwen_with_router_nki.py with the TKG MoE path replaced
by a direct call to kernels.moe_fused_tkg.kernel_v8a.qwen3_moe_fused_tkg.

Changes vs qwen_with_router_nki.py:
  - TKG path: self.mlp(...) replaced by kernel_v8a.qwen3_moe_fused_tkg[2](...)
  - CTE path: unchanged (self.mlp as before)
  - Weight layout: reuses initialize_moe_module weights exactly as loaded;
    gate_up_proj.weight is reshaped [E, H, 2*I] -> [E, H, 2, I] inline

All tensor inputs to the kernel have .data appended for XLA tracing compatibility.
"""

import warnings
from typing import Optional, Tuple

import torch
from torch import nn

from qwen_with_router_nki import (
    NeuronQwen3MoeDecoderLayerWithNKI,
    NeuronQwen3MoeModelWithNKI,
    NeuronQwen3MoeForCausalLM as _NeuronQwen3MoeForCausalLMBase,
)
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

from kernels.moe_fused_tkg import kernel_v8a


# ---------------------------------------------------------------------------
# Decoder layer — TKG MoE path replaced with kernel_v8a
# ---------------------------------------------------------------------------

class NeuronQwen3MoeDecoderLayerFusedTKG(NeuronQwen3MoeDecoderLayerWithNKI):
    """Decoder layer that calls kernel_v8a.qwen3_moe_fused_tkg for TKG, CTE unchanged."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead."
            )

        residual = hidden_states

        qkv_fused_rmsnorm = None
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states

        is_tkg = past_key_value is not None
        if is_tkg:
            # kernel_v8a handles RMSNorm + Router + TopK + Expert MLPs internally.
            # inp: [B, 1, H] (pre-norm hidden states)
            # gamma: [1, H]  (post-attention RMSNorm weight)
            # router_w: [H, E] float32 (weight_T set up by _install_nki_router)
            # gate_up_w: [E, H, 2, I]  (reshaped from stored [E, H, 2*I])
            # down_w: [E, I, H]
            tkg = self.mlp.moe_fused_tkg
            gate_up_w_fused = tkg.expert_mlps.mlp_op.gate_up_proj.weight  # [E, H, 2*I]
            E, H, two_I = gate_up_w_fused.shape
            gate_up_w = gate_up_w_fused.view(E, H, 2, two_I // 2)         # [E, H, 2, I]

            moe_out = kernel_v8a.qwen3_moe_fused_tkg[2](
                hidden_states.data,
                self.post_attention_layernorm.weight.unsqueeze(0).data,    # [1, H]
                tkg.router.weight_T.data,                                  # [H, E] float32
                gate_up_w.data,                                            # [E, H, 2, I]
                tkg.expert_mlps.mlp_op.down_proj.weight.data,             # [E, I, H]
            )
            # kernel returns [B, H]; restore seq dim to match residual [B, 1, H]
            hidden_states = moe_out.unsqueeze(1)
        else:
            # CTE path: unchanged
            hidden_states = self.mlp(hidden_states, padding_mask)[0]

        hidden_states = residual + hidden_states

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Model — swaps in the new decoder layer
# ---------------------------------------------------------------------------

class NeuronQwen3MoeModelFusedTKG(NeuronQwen3MoeModelWithNKI):
    def init_model(self, config):
        super().init_model(config)
        self.layers = nn.ModuleList([
            NeuronQwen3MoeDecoderLayerFusedTKG(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])


# ---------------------------------------------------------------------------
# CausalLM entry point
# ---------------------------------------------------------------------------

class NeuronQwen3MoeForCausalLM(_NeuronQwen3MoeForCausalLMBase):
    """Qwen3 MoE CausalLM with kernel_v8a fused MoE TKG kernel."""

    _model_cls = NeuronQwen3MoeModelFusedTKG
