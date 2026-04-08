# coding=utf-8
"""
Qwen3 MOE model for NXD inference — with offline blockwise FP8 expert quantization.

Extends qwen.py by quantizing MoE expert weights (gate_up_proj, down_proj) to FP8
inside convert_qwen3_moe_hf_to_neuron_state_dict, after the fused neuron-layout tensors
are built.  All other weights (attention, router, norms, embeddings) are unchanged.

Quantization scheme
-------------------
Format  : FP8 E4M3FN (torch.float8_e4m3fn, exact per-spec via PyTorch cast)
Block   : 128 values along the contraction / input-feature dimension
Dim     : dim 1 of each fused tensor (H for gate_up_proj, I for down_proj)

Fused tensor shapes (neuron layout, already transposed from HF):
  gate_up_proj : [E, H, 2·I]   — matmul: tokens[T,H] @ w[H,2·I] → contraction over H
  down_proj    : [E, I,  H ]   — matmul: act  [T,I] @ w[I, H ] → contraction over I

To block over dim 1 on a 3-D tensor we:
  1. Permute: [E, D_contract, D_out] → [E, D_out, D_contract]
  2. Reshape the last dim into [n_blocks, block_size], padding if needed
  3. Compute per-block amax; derive scale = amax / qmax  (scale=1 for zero blocks)
  4. Cast normalized block to float8_e4m3fn (PyTorch handles rounding + clamping)
  5. Store FP8 weight and float32 scales alongside BF16 weights in the state dict

Scale tensor shapes:
  gate_up_proj scales : [E, 2·I, ceil(H / 128)]
  down_proj    scales : [E,  H,  ceil(I / 128)]

Stored state-dict keys (added alongside the weight keys):
  layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight        → torch.float8_e4m3fn
  layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight_scale  → torch.float32
  layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight           → torch.float8_e4m3fn
  layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight_scale     → torch.float32
"""

import torch

from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeInferenceConfig

torch.manual_seed(0)

import gc
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import math

from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode
from torch import nn
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import Qwen3MoeForCausalLM
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig, SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP, MOE_TKG_MK_INTERMEDIATE_PER_TP
from neuronx_distributed_inference.models.model_wrapper import CONTEXT_ENCODING_MODEL_TAG, TOKEN_GENERATION_MODEL_TAG
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

_flash_fwd_call = nki_jit()(attention_isa_kernel)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

# ---------------------------------------------------------------------------
# Blockwise FP8 quantization helpers
# ---------------------------------------------------------------------------
_FP8_QMAX = 448.0          # max finite magnitude for E4M3FN (OFP8 spec)
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_BLOCK_SIZE = 128


def _quantize_expert_fused_weight(
    weight: torch.Tensor,
    contract_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Blockwise FP8 quantization of a 3-D expert fused weight tensor.

    Parameters
    ----------
    weight : torch.Tensor
        Shape [E, D_contract, D_out] in neuron layout.
        Examples:
          gate_up_proj → [E, H,  2·I],  contract_dim=1  (H = hidden_size)
          down_proj    → [E, I,   H ],  contract_dim=1  (I = intermediate_size)
    contract_dim : int
        The dimension to block over.  Always 1 for both fused expert tensors.

    Returns
    -------
    fp8_weight : torch.Tensor, dtype=torch.float8_e4m3fn
        Same shape as input weight: [E, D_contract, D_out].
        Stored in the original layout so NXD's shape-validation in
        shard_weights_with_cache passes unchanged.
    scales : torch.Tensor, dtype=torch.float32
        Shape [E, D_out, n_blocks]  (one scale per 128-value block along D_contract).

    Notes on layout
    ---------------
    Blocking is computed in permuted space [E, D_out, D_contract] so that
    scales index as [expert, output_feature, block_index].  The normalized
    values are then permuted back to the original [E, D_contract, D_out]
    layout before casting to FP8, keeping the stored shape identical to
    the BF16 weight NXD expects.
    """
    assert weight.ndim == 3, f"Expected 3-D tensor, got shape {weight.shape}"
    E, D_c, D_out = weight.shape[0], weight.shape[contract_dim], weight.shape[2 if contract_dim == 1 else 1]

    # Work in float32 for numerical stability; make contiguous.
    w = weight.contiguous().float()          # [E, D_contract, D_out]

    # 1. Permute so D_contract is last: [E, D_out, D_contract]
    w = w.permute(0, 2, 1).contiguous()     # [E, D_out, D_contract]

    # 2. Pad D_contract to a multiple of block_size along the last dim.
    n_blocks = math.ceil(D_c / _FP8_BLOCK_SIZE)
    pad_amount = n_blocks * _FP8_BLOCK_SIZE - D_c
    if pad_amount > 0:
        w = torch.nn.functional.pad(w, (0, pad_amount))  # pad last dim

    # 3. Reshape into blocks: [E, D_out, n_blocks, block_size]
    w = w.reshape(E, D_out, n_blocks, _FP8_BLOCK_SIZE)

    # 4. Per-block scale: amax over block_size axis (dim=-1).
    amax = w.abs().amax(dim=-1)                          # [E, D_out, n_blocks]
    scales = torch.where(
        amax == 0,
        torch.ones_like(amax),
        amax / _FP8_QMAX,
    ).float()

    # 5. Normalize (in blocked view), then restore original shape before casting.
    normalized = (w / scales.unsqueeze(-1)).clamp(-_FP8_QMAX, _FP8_QMAX)

    # 6. Flatten blocks back: [E, D_out, n_blocks * block_size] → strip pad → [E, D_out, D_c]
    normalized = normalized.reshape(E, D_out, n_blocks * _FP8_BLOCK_SIZE)
    if pad_amount > 0:
        normalized = normalized[..., :D_c]

    # 7. Permute back to original layout: [E, D_contract, D_out]
    normalized = normalized.permute(0, 2, 1).contiguous()

    # Cast to FP8 — shape matches original weight, NXD shape check passes.
    fp8_weight = normalized.to(_FP8_DTYPE)

    return fp8_weight, scales


def _dequantize_expert_fused_weight(
    fp8_weight: torch.Tensor,
    scales: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Reconstruct BF16/FP32 weight from FP8 weight + scales.

    fp8_weight : [E, D_contract, D_out]  — original shape, FP8 dtype
    scales     : [E, D_out, n_blocks]
    """
    E, D_c, D_out = fp8_weight.shape
    n_blocks = scales.shape[2]

    # Permute to [E, D_out, D_contract] to align with scales layout.
    w = fp8_weight.float().permute(0, 2, 1).contiguous()   # [E, D_out, D_c]

    # Pad D_contract to n_blocks * block_size if needed.
    pad_amount = n_blocks * _FP8_BLOCK_SIZE - D_c
    if pad_amount > 0:
        w = torch.nn.functional.pad(w, (0, pad_amount))

    # Reshape into blocks and multiply by scales.
    w = w.reshape(E, D_out, n_blocks, _FP8_BLOCK_SIZE)
    w = w * scales.unsqueeze(-1)                            # [E, D_out, n_blocks, 128]

    # Flatten and strip padding.
    w = w.reshape(E, D_out, n_blocks * _FP8_BLOCK_SIZE)
    if pad_amount > 0:
        w = w[..., :D_c]

    # Permute back to [E, D_contract, D_out].
    return w.permute(0, 2, 1).contiguous().to(target_dtype)

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE


# Get the modules_to_not_convert from the neuron configs
def get_modules_to_not_convert(neuron_config: MoENeuronConfig):
    return getattr(neuron_config, "modules_to_not_convert", None)


def _helper_concat_and_delete_qkv(qwen_state_dict: Dict[str, Any], layer_num: int, attr: str):
    """
    Helper function to concatenate and delete QKV attributes for fusedqkv (weight or scale).
    Args:
        qwen_state_dict: The state dictionary containing model weights
        layer_num: The index of the layer to process
        attr: The attribute to process ('weight' or 'scale')
    """
    qwen_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            qwen_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            qwen_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            qwen_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ],
    )
    del qwen_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del qwen_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del qwen_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(qwen_state_dict: Dict[str, Any], cfg: InferenceConfig):
    """
    This function concats the qkv weights and scales to a Wqkv weight and scale for fusedqkv, and deletes the qkv weights.
    """
    mods_to_not_conv = get_modules_to_not_convert(cfg.neuron_config)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for l in range(cfg.num_hidden_layers):  # noqa: E741
        _helper_concat_and_delete_qkv(qwen_state_dict, l, "weight")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized
        ) and f"layers.{l}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(qwen_state_dict, l, "scale")

    gc.collect()

    return qwen_state_dict


def maybe_dequantize_layer(neuron_state_dict, config):
    scale_layers = []
    for layer_key in neuron_state_dict.keys():
        if "_scale_inv" in layer_key:
            scales = neuron_state_dict[layer_key]
            scale_layers.append(layer_key)
            fp8_layer_name = layer_key.replace("_scale_inv", "")
            fp8_layer = neuron_state_dict[fp8_layer_name]
            block_size = config.quantization_config["weight_block_size"]
            scales_expanded = scales.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=1)
            scaled_layer = fp8_layer.to(torch.float32) * scales_expanded.to(torch.float32)
            neuron_state_dict[fp8_layer_name] = scaled_layer.to(config.neuron_config.torch_dtype)

    # delete scale layers
    for scale_layer in scale_layers:
        del neuron_state_dict[scale_layer]


def convert_qwen3_moe_hf_to_neuron_state_dict(neuron_state_dict, config):
    """
    Helper function which converts the huggingface checkpoints to state dictionary compatible with the stucture of the neuron MoE model.
    """
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    # dequantize layers if needed
    maybe_dequantize_layer(neuron_state_dict, config)

    # to facilitate rank usage in base model
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    for l in range(config.num_hidden_layers):  # noqa: E741
        # To facilitate rank usage in attention
        neuron_state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )

        # Rename the q_norm, k_norm names
        neuron_state_dict[f"layers.{l}.self_attn.k_layernorm.weight"] = (
            neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"]

        # Rename the q_norm, k_norm names
        neuron_state_dict[f"layers.{l}.self_attn.q_layernorm.weight"] = (
            neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"]

        # Copy router weights
        neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
            neuron_state_dict[f"layers.{l}.mlp.gate.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.mlp.gate.weight"]

        intermediate_size, hidden_size = neuron_state_dict[
            f"layers.{l}.mlp.experts.0.gate_proj.weight"
        ].shape
        device = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"].device
        dtype = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"].dtype

        # copy the MLP parameters
        gate_up_proj = torch.empty(
            config.num_experts,
            hidden_size,
            2 * intermediate_size,
            dtype=dtype,
            device=device,
        )
        for e in range(config.num_experts):
            # Copy gate_proj and up_proj after concatenation
            gate_proj_weights = (
                neuron_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"]
                .T.detach()
                .clone()
            )
            up_proj_weights = (
                neuron_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"]
                .T.detach()
                .clone()
            )

            gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
            gate_proj_slice = torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size)
            gate_proj_slice.copy_(gate_proj_weights)
            up_proj_slice = torch.narrow(
                gate_up_proj_slice, 2, intermediate_size, intermediate_size
            )
            up_proj_slice.copy_(up_proj_weights)

            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"]
            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"]

        # padding gate_up_proj on intermediate size
        pad_size = getattr(config, "moe_intermediate_pad_size", 0)
        if pad_size > 0:
            gate_up_proj = gate_up_proj.reshape(config.num_experts, hidden_size, 2, -1)
            # padding right on gate_up_proj: (num_experts, hidden_size, 2, intermediate_size)
            gate_up_proj = torch.nn.functional.pad(gate_up_proj, (0, pad_size))
            gate_up_proj = gate_up_proj.reshape(config.num_experts, hidden_size, -1)
        # Quantize gate_up_proj to FP8 (block over H, the contraction dim = dim 1).
        # gate_up_proj shape: [E, H, 2·I]
        # Quantize gate_up_proj to FP8 (block over H, the contraction dim = dim 1).
        # Stored shape matches original [E, H, 2·I] so NXD shape validation passes.
        gate_up_fp8, gate_up_scales = _quantize_expert_fused_weight(gate_up_proj, contract_dim=1)
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_fp8
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight_scale"] = gate_up_scales
        del gate_up_proj

        down_proj = torch.empty(
            config.num_experts,
            intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        for e in range(config.num_experts):
            # Copy down_proj
            down_proj_weights = (
                neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"]
                .T.detach()
                .clone()
            )
            down_proj_slice = torch.narrow(down_proj, 0, e, 1)
            down_proj_slice.copy_(down_proj_weights)
            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"]

        # padding down_proj on intermediate size
        if pad_size > 0:
            # padding bottom on down_proj: (num_experts, intermediate_size, hidden_size)
            down_proj = torch.nn.functional.pad(down_proj, (0, 0, 0, pad_size))

        # Quantize down_proj to FP8 (block over I, the contraction dim = dim 1).
        # Stored shape matches original [E, I, H] so NXD shape validation passes.
        down_fp8, down_scales = _quantize_expert_fused_weight(down_proj, contract_dim=1)
        del down_proj
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_fp8
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight_scale"] = down_scales

        gc.collect()

    if config.neuron_config.fused_qkv:
        neuron_state_dict = convert_state_dict_to_fused_qkv(neuron_state_dict, config)

    return neuron_state_dict


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return Qwen3MoeRMSNorm if cpu_mode() else CustomRMSNorm


class Qwen3MoeInferenceConfig(InferenceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Qwen3-MoE config has `num_experts` instead of `num_local_experts`
        # We need to add `num_local_experts` as it is expected by `initialize_moe_module`
        self.num_local_experts = self.num_experts
        # Qwen3-MoE has no shared experts
        self.n_shared_experts = 0
        # ExpertMLPsV2 reads moe_intermediate from config.intermediate_size

        # check whether need to pad intermediate size
        self.maybe_pad_intermediate()

        # enable moe_fused_nki_kernel
        self.enable_moe_fused_nki_kernel()

        self.intermediate_size = self.moe_intermediate_size
        # We need router dtype to be FP32 for accuracy
        self.neuron_config.router_config.dtype = torch.float32
        # HF uses softmax (non-configurable) act for Qwen3-MoE
        self.neuron_config.router_config.act_fn = "softmax"
        # Set DISABLE_NUMERIC_CC_TOKEN=1 for Qwen3 MoE as a workaround
        # for the extra add/multiple in all-gather/reduce-scatter CC ops
        # https://github.com/pytorch/xla/pull/3825 (openxla PR https://github.com/openxla/xla/pull/7677 not accepted)
        self.neuron_config.disable_numeric_cc_token = True
        # Qwen3 normalizes top k affinities
        self.neuron_config.normalize_top_k_affinities = True

    def maybe_pad_intermediate(self):
        moe_tp_degree = self.neuron_config.moe_tp_degree
        I_TP = self.moe_intermediate_size // moe_tp_degree
        if getattr(self.neuron_config.blockwise_matmul_config, "use_shard_on_intermediate_dynamic_while", False):
            # If shard-on-I enabled, check the intermediate size per tp is divisible by SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP
            if I_TP % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded_moe_intermediate_size = math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP) * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP * moe_tp_degree
                self.moe_intermediate_pad_size = max(padded_moe_intermediate_size - self.moe_intermediate_size, 0)
                # set moe_intermediate_size to padded size
                self.moe_intermediate_size = padded_moe_intermediate_size

    def enable_moe_fused_nki_kernel(self):
        I_TP = self.moe_intermediate_size // self.neuron_config.moe_tp_degree
        # if moe_fused_nki_kernel_enabled is enabled and the intermeidiate_size_per_tp is divisible by MOE_TKG_MK_INTERMEDIATE_PER_TP
        if getattr(self.neuron_config, "moe_fused_nki_kernel_enabled", False) and I_TP % MOE_TKG_MK_INTERMEDIATE_PER_TP == 0:
            self.moe_fused_nki_kernel_enabled = True

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "max_position_embeddings",
            "moe_intermediate_size",
            "norm_topk_prob",
            "num_attention_heads",
            "num_experts",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_scaling",
            "rope_theta",
            "tie_word_embeddings",
            "vocab_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


class NeuronQwen3MoEAttention(NeuronAttentionBase):
    def __init__(self, config: Qwen3MoeInferenceConfig):
        rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            # qk_norm in the base class is different from Qwen3RMSNorm
            use_qk_norm=False,
        )

        # Override q_layernorm and k_layernorm with RMSNorm
        self.q_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)
        self.k_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronQwen3MoEAttention has to be initialized in a distributed env. Please use neuronx_distributed"
                " module to initialize a distributed env."
            )


class NeuronQwen3MoeDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3MoEAttention(config=config)
        self.moe_fused_nki_kernel_enabled = getattr(config, "moe_fused_nki_kernel_enabled", False)

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        if self.moe_fused_nki_kernel_enabled:
            self.mlp = initialize_moe_module(
                config=config, rmsnorm=self.post_attention_layernorm, init_tkg_module=True
            )
        else:
            self.mlp = initialize_moe_module(
                config=config,
            )

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.FloatTensor`, *optional*):
                position ids of size `(batch_size, sequence_length)`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        qkv_fused_rmsnorm = None
        # We wrap input_layernorm/self_attn/post_attention_layernorm with module markers start/end
        # as a hint for compiler's modular-flow to avoid layer boundries in-between decoder layer components
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MoE
        residual = hidden_states
        if not self.moe_fused_nki_kernel_enabled:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, padding_mask)[0]
        hidden_states = residual + hidden_states

        # End module marker
        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronQwen3MoeModel(NeuronBaseModel):
    """
    NeuronQwen3MoeModel extends the Qwen3MoeModel to be traceable.
    The forward function of this class is traced.
    """

    def setup_attr_for_model(self, config: Qwen3MoeInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

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
        self.layers = nn.ModuleList(
            [
                NeuronQwen3MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


class NeuronQwen3MoeForCausalLM(NeuronBaseForCausalLM):
    """
    This class can be used as Qwen3MoeForCausalLM
    """

    _model_cls = NeuronQwen3MoeModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return Qwen3MoeForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return Qwen3MoeInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: Qwen3MoeInferenceConfig) -> dict:
        return convert_qwen3_moe_hf_to_neuron_state_dict(state_dict, config)

    # Wraps NeuronBaseForCausalLM.enable_context_encoding() to add compile_tag.
    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    # Wraps NeuronBaseForCausalLM.enable_token_generation() to add compile_tag.
    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self):
        # Set compiler optimization level based on model tag
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            # Disable Modular flow for TKG graph with EP enabled as it causes perf degradation
            optimization_level = "-O3" if self.neuron_config.moe_ep_degree > 1 else "-O1"
        compiler_args = f"--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer {optimization_level}"
        # Add flags for cc-overlap
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        )
        compiler_args += " --auto-cast=none"
        # Enable vector-offset DGE
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        if self.neuron_config.scratchpad_page_size:
            compiler_args += (
                f" --hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size} "
            )

        if self.neuron_config.attn_block_tkg_nki_kernel_enabled:
            assert (
                self.neuron_config.attn_block_tkg_nki_kernel_cascaded_attention
            ), "If using attn_block_tkg_nki_kernel_enabled for Qwen3MoE you must also use attn_block_tkg_nki_kernel_cascaded_attention"
            # Enabled RMSNorm pre-RoPE in the Attn TKG MK
            self.neuron_config.pre_rope_rmsnorm = True
            # When enabling the Cascaded Attn TKG MK we will run over 5 million instructions on E2E
            compiler_args += " --internal-max-instruction-limit=15000000"

        return compiler_args


def generate(skip_compile=False):
    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)

    if not skip_compile:
        neuron_config = MoENeuronConfig(
            tp_degree=4,
            batch_size=1,
            max_context_length=128,
            seq_len=1024,
            on_device_sampling_config=OnDeviceSamplingConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95),
            enable_bucketing=False,
            flash_decoding_enabled=False
        )
        config = Qwen3MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )        
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token
        # Compile and save model.
        print("\nCompiling and saving model...")
        model = NeuronQwen3MoeForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronQwen3MoeForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate outputs.
    print("\nGenerating outputs...")
    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")

