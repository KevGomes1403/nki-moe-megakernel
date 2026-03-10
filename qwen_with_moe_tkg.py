# coding=utf-8
"""Qwen3 MoE with direct nkilib moe_tkg kernel for token generation.

Context encoding (prefill):   uses attention_cte from qwen_with_attention_cte.py (unchanged).
Token generation (decode):    for the MoE MLP, bypasses the NxD MoEFusedTKG wrapper and calls
                               nkilib's moe_tkg kernel directly, analogous to how attention_cte
                               is called directly for prefill.

Architecture-specific notes (Qwen3-30B-A3B, config.json):
  H  = 2048   (hidden_size)
  I  = 768    (moe_intermediate_size); per TP rank: I_tp = 768/tp_degree
  E  = 128    (num_experts)
  K  = 8      (num_experts_per_tok)
  norm_topk_prob = True  ->  router softmax probs are re-normalized over selected K experts
  activation   = SiLU

moe_tkg shape constraints (non-MX, selective-expert mode):
  T  <= 128  (batch tokens during decode)
  H  divisible by 128
  E  > 1
  K  <= 16
"""

import logging
from typing import Optional, Tuple

import nki
import torch
from torch import nn

from neuronx_distributed.parallel_layers import mappings, parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

# Import base classes from the attention_cte variant; rename to avoid collision.
from qwen_with_attention_cte import (
    Qwen3MoeInferenceConfig,
    NeuronQwen3MoEAttention,
    NeuronQwen3MoeModel as _BaseModel,
    NeuronQwen3MoeForCausalLM as _BaseCausalLM,
    get_rmsnorm_cls,
    convert_qwen3_moe_hf_to_neuron_state_dict,
)

try:
    from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg as _moe_tkg_fn
    from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
    _moe_tkg_available = True
except ImportError:
    _moe_tkg_fn = None
    _moe_tkg_available = False

logger = logging.getLogger("Neuron")

# Compile once at module load; nki.jit caches the traced KLIR per unique shape set.
_moe_tkg_jit = nki.jit(_moe_tkg_fn, platform_target="trn2") if _moe_tkg_available else None

# One-time log flags (avoid log spam in the decode loop).
_moe_tkg_active_logged = False
_moe_tkg_skip_logged = False
_moe_tkg_error_logged = False


# ---------------------------------------------------------------------------
# Decoder layer with direct moe_tkg call for token generation
# ---------------------------------------------------------------------------

class NeuronQwen3MoeTKGDecoderLayer(nn.Module):
    """Decoder layer that wires attention_cte (CTE) and moe_tkg (TKG) together.

    For context encoding (q_len > 1):  falls through to the standard MoE MLP.
    For token generation (q_len == 1): calls nkilib moe_tkg directly.
    """

    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3MoEAttention(config=config)

        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        # Standard MoE module – used for CTE and as the weight store for TKG.
        # We do NOT use init_tkg_module=True; moe_tkg is called directly below.
        self.mlp = initialize_moe_module(config=config)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens

        # Whether to use direct moe_tkg for decode (can be disabled via neuron_config).
        self.moe_tkg_direct_enabled = (
            _moe_tkg_available
            and getattr(config.neuron_config, "moe_tkg_direct_enabled", True)
        )

        # Qwen3 normalizes the top-K router probs to sum to 1 before scaling expert outputs.
        self.normalize_top_k_affinities = config.neuron_config.normalize_top_k_affinities
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

    # ------------------------------------------------------------------
    # moe_tkg helper
    # ------------------------------------------------------------------

    def _forward_moe_tkg(self, hidden_states: torch.Tensor, bsz: int) -> torch.Tensor:
        """Call nkilib moe_tkg directly for one decode step.

        Args:
            hidden_states: [bsz, 1, H]  post-RMSNorm
            bsz:           batch size (= number of tokens T during decode)

        Returns:
            [bsz, 1, H]  MoE output (after TP all-reduce)
        """
        global _moe_tkg_active_logged
        if not _moe_tkg_active_logged:
            logger.warning("moe_tkg NKI kernel ACTIVE for token generation (direct nkilib call)")
            _moe_tkg_active_logged = True

        T = bsz
        H = hidden_states.shape[-1]
        hidden_2d = hidden_states.reshape(T, H)  # [T, H]

        # --- Router: get affinities and top-K indices ---
        # RouterTopK.forward returns (router_logits, expert_affinities, expert_index).
        #   expert_affinities: [T, E]  full softmax probabilities (not yet top-K normalized)
        #   expert_index:      [T, K]  int64 indices of the K selected experts
        _, affinities_full, expert_index = self.mlp.router(hidden_2d)

        # moe_tkg expects int32 indices.
        expert_index_i32 = expert_index.to(torch.int32)

        if self.normalize_top_k_affinities:
            # Qwen3 norm_topk_prob=True: re-normalize K selected probs to sum to 1.
            topk_probs = affinities_full.gather(1, expert_index)           # [T, K]
            topk_normalized = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # [T, K]
            # Sparse [T, E]: zero everywhere except at selected expert positions.
            affinities_sparse = torch.zeros_like(affinities_full)
            affinities_sparse.scatter_(1, expert_index, topk_normalized)
        else:
            affinities_sparse = affinities_full  # use raw softmax values

        # --- Weights (TP-sharded on the intermediate dimension I) ---
        # Each TP rank holds a slice of I: I_tp = I / tp_degree.
        # gate_up_proj.weight: [E, H, 2*I_tp]
        # down_proj.weight:    [E, I_tp, H]
        gate_up_w = self.mlp.expert_mlps.mlp_op.gate_up_proj.weight  # [E, H, 2*I_tp]
        down_w    = self.mlp.expert_mlps.mlp_op.down_proj.weight      # [E, I_tp, H]

        E, H2, two_I_tp = gate_up_w.shape
        I_tp = two_I_tp // 2

        # moe_tkg expects gate+up separated: [E, H, 2, I_tp].
        gate_up_4d = gate_up_w.view(E, H, 2, I_tp)

        # --- nkilib moe_tkg kernel call ---
        # POST_SCALE: output[t] += affinity[t,e] * expert_out(hidden[t], weights[e])
        # is_all_expert=False: selective mode, only K experts are processed per token.
        # Note: expert_affinities must be float32 — the ISA tensor_scalar op for POST_SCALE
        # requires fp32 for the affinity operand (bf16 triggers a compiler error).
        output = _moe_tkg_jit(
            hidden_input=hidden_2d,
            expert_gate_up_weights=gate_up_4d,
            expert_down_weights=down_w,
            expert_affinities=affinities_sparse.to(dtype=torch.float32),  # [T, E] float32
            expert_index=expert_index_i32,                                # [T, K]
            is_all_expert=False,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
        )  # [T, H]

        # --- TP all-reduce ---
        # Each rank produced a partial result over its I_tp shard; sum across TP ranks.
        output = mappings.reduce_from_tensor_model_parallel_region(
            output, process_group=parallel_state.get_world_group()
        )

        return output.reshape(bsz, 1, H)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple:
        global _moe_tkg_skip_logged, _moe_tkg_error_logged

        # ---- Attention block (mirrors qwen_with_attention_cte decoder layer) ----
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

        # ---- MoE block ----
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        bsz, q_len, _ = hidden_states.shape

        # Token generation: single query token per sequence.
        if q_len == 1 and self.moe_tkg_direct_enabled:
            try:
                hidden_states = self._forward_moe_tkg(hidden_states, bsz)
            except Exception as exc:
                if not _moe_tkg_error_logged:
                    logger.warning(
                        "moe_tkg direct call failed; falling back to standard MoE MLP. error=%s", exc
                    )
                    _moe_tkg_error_logged = True
                hidden_states = self.mlp(hidden_states, padding_mask)[0]
        else:
            # Context encoding or moe_tkg disabled: standard MoE MLP.
            if not _moe_tkg_skip_logged and q_len > 1:
                logger.debug("CTE (q_len=%d): using standard MoE MLP (moe_tkg is TKG-only)", q_len)
                _moe_tkg_skip_logged = True
            hidden_states = self.mlp(hidden_states, padding_mask)[0]

        hidden_states = residual + hidden_states
        hidden_states = ModuleMarkerEndWrapper()(hidden_states)

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Model: override only init_model to swap the decoder layer class
# ---------------------------------------------------------------------------

class NeuronQwen3MoeTKGModel(_BaseModel):
    """NeuronQwen3MoeModel with moe_tkg decoder layers for token generation."""

    def init_model(self, config: Qwen3MoeInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        # Key change: use NeuronQwen3MoeTKGDecoderLayer instead of default.
        self.layers = nn.ModuleList(
            [NeuronQwen3MoeTKGDecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


# ---------------------------------------------------------------------------
# CausalLM: point to the new model class
# ---------------------------------------------------------------------------

class NeuronQwen3MoeForCausalLM(_BaseCausalLM):
    """Qwen3 MoE CausalLM with direct moe_tkg token-generation kernel."""

    _model_cls = NeuronQwen3MoeTKGModel

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: Qwen3MoeInferenceConfig) -> dict:
        return convert_qwen3_moe_hf_to_neuron_state_dict(state_dict, config)
