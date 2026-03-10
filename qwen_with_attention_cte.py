# coding=utf-8
""" Qwen3 MOE model for NXD inference. This is a re-implementation of the NxDI source code for Qwen3 MOE, provided here for easy kernel development."""

import torch

from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeInferenceConfig

torch.manual_seed(0)

import gc
import logging
import shlex
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn.functional as F
import math
import nki

from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronx_distributed.parallel_layers import mappings, parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode
from torch import nn
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import Qwen3MoeForCausalLM
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig, SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP, MOE_TKG_MK_INTERMEDIATE_PER_TP
from neuronx_distributed_inference.models.model_wrapper import CONTEXT_ENCODING_MODEL_TAG, TOKEN_GENERATION_MODEL_TAG
from neuronx_distributed_inference.modules.attention.attention_base import (
    FlashAttentionStrategy,
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

from nkilib.core.attention.attention_cte import attention_cte

try:
    from nkilib.core.attention.attention_tkg import attention_tkg
    from nkilib.core.attention.attention_tkg_utils import AttnTKGConfig
    from nkilib.core.utils.allocator import SbufManager
except ImportError:
    attention_tkg = None
    AttnTKGConfig = None
    SbufManager = None

_flash_fwd_call = nki_jit()(attention_isa_kernel)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

# nki_moe_fused kernel: one-time log flags
_moe_fused_active_logged = False
_moe_fused_skip_logged = False
_moe_fused_error_logged = False

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE
logger = logging.getLogger("Neuron")
_attn_cte_active_logged = False
_attn_cte_skip_logged = False
_attn_cte_error_logged = False
_attn_tkg_active_logged = False
_attn_tkg_skip_logged = False
_attn_tkg_error_logged = False
_attn_tkg_posid_mismatch_logged = False

_ATTN_TKG_MAX_BS = 128
_ATTN_TKG_MAX_Q_HEAD = 16
_ATTN_TKG_MAX_S_ACTIVE = 8
_ATTN_TKG_MAX_S_PRIOR = 32 * 1024
_ATTN_TKG_SB_RANGE_BYTES = 200 * 1024


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
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

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
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

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
        self.moe_fused_nki_kernel_enabled = True
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
        
        # self.neuron_config.attn_block_tkg_nki_kernel_enabled = True
        # self.neuron_config.attn_block_tkg_nki_kernel_cascaded_attention = True
        # self.neuron_config.fused_qkv = True
        # self.neuron_config.qkv_kernel_enabled = True
        self._snap_tkg_buckets_to_128()

    def _snap_tkg_buckets_to_128(self):
        """Cascaded TKG kernel requires s_prior % 128 == 0 for s_prior > 256.
        Round up any non-compliant values to the next multiple of 128.

        generate_buckets_for_tkg has two paths:
          - enable_bucketing=False: uses max_length directly (ignores token_generation_buckets)
          - enable_bucketing=True:  uses token_generation_buckets if set, else generate_buckets(128, max_length)
        Both paths must be covered, so we snap max_length/seq_len AND token_generation_buckets.
        """
        nc = self.neuron_config

        def snap128(n):
            return n if n <= 256 or n % 128 == 0 else ((n + 127) // 128) * 128

        # Snap explicit bucket list (enable_bucketing=True path when user passes --token-generation-buckets)
        if nc.token_generation_buckets is not None:
            nc.token_generation_buckets = sorted(set(snap128(b) for b in nc.token_generation_buckets))

        # Snap max_length and seq_len — these drive the bucket ceiling in all paths
        for attr in ("max_length", "seq_len"):
            v = getattr(nc, attr, None)
            if v and v > 256 and v % 128 != 0:
                snapped = snap128(v)
                setattr(nc, attr, snapped)
                logger.warning(
                    "attn_block_tkg: %s=%d is not a multiple of 128; snapping to %d "
                    "(cascaded kernel requires s_prior %% 128 == 0)",
                    attr, v, snapped,
                )

    def _get_effective_moe_tp_degree(self):
        moe_tp_degree = self.neuron_config.moe_tp_degree
        # For the common TP-only setup, reuse TP degree for MoE TP unless caller overrides it.
        if moe_tp_degree == 1 and self.neuron_config.tp_degree > 1:
            moe_tp_degree = self.neuron_config.tp_degree
        return moe_tp_degree

    def maybe_pad_intermediate(self):
        moe_tp_degree = self._get_effective_moe_tp_degree()
        I_TP = self.moe_intermediate_size // moe_tp_degree
        if getattr(self.neuron_config.blockwise_matmul_config, "use_shard_on_intermediate_dynamic_while", False):
            # If shard-on-I enabled, check the intermediate size per tp is divisible by SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP
            if I_TP % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded_moe_intermediate_size = math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP) * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP * moe_tp_degree
                self.moe_intermediate_pad_size = max(padded_moe_intermediate_size - self.moe_intermediate_size, 0)
                # set moe_intermediate_size to padded size
                self.moe_intermediate_size = padded_moe_intermediate_size

    def enable_moe_fused_nki_kernel(self):
        requested = getattr(self.neuron_config, "moe_fused_nki_kernel_enabled", None)
        self.moe_fused_nki_kernel_enabled = False
        if requested is not True:
            return

        moe_tp_degree = self._get_effective_moe_tp_degree()
        I_TP = self.moe_intermediate_size // moe_tp_degree
        if I_TP % MOE_TKG_MK_INTERMEDIATE_PER_TP == 0:
            self.moe_fused_nki_kernel_enabled = True
            return

        msg = (
            "Cannot enable fused MoE kernel: intermediate_size_per_tp "
            f"({I_TP}) must be divisible by {MOE_TKG_MK_INTERMEDIATE_PER_TP}. "
            "Set `moe_tp_degree` accordingly or disable `moe_fused_nki_kernel_enabled`."
        )
        raise ValueError(msg)

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
        self.attn_cte_kernel_enabled = getattr(config.neuron_config, "attn_cte_kernel_enabled", True)
        self.attn_tkg_kernel_enabled = getattr(config.neuron_config, "attn_tkg_kernel_enabled", True)
        self.attn_tkg_use_pos_id = getattr(config.neuron_config, "attn_tkg_use_pos_id", False)
        self._tkg_decode_skip_rope_active = False
        self._tkg_decode_rotary_position_ids = None

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronQwen3MoEAttention has to be initialized in a distributed env. Please use neuronx_distributed"
                " module to initialize a distributed env."
            )

    def _attention_tkg_static_disable_reason(self, q_len: int) -> Optional[str]:
        if not hasattr(self, "attention_tokengen_kernel_builtin"):
            return "attention_tokengen_kernel_builtin unavailable"
        if not self.attn_tkg_kernel_enabled:
            return "attn_tkg_kernel_enabled is False"
        if self.cp_degree > 1:
            return "cp_degree > 1"
        if self.attention_chunk_size is not None:
            return "attention_chunk_size is set"
        if self.sliding_window:
            return "sliding_window attention is enabled"
        if self.flash_decoding_enabled:
            return "flash_decoding is enabled"
        if self.neuron_config.is_chunked_prefill:
            return "chunked prefill is enabled"
        if self.head_dim == 128 and not getattr(self.neuron_config, "attn_tkg_allow_head_dim_128", False):
            return "head_dim=128 attention_tkg path disabled (set attn_tkg_allow_head_dim_128=True to override)"
        if self.head_dim > 128:
            return "head_dim > 128 (attention_tkg limit)"
        if self.head_dim % 64 != 0:
            return "head_dim not divisible by 64 (attention_tkg fused RoPE requirement)"
        if q_len <= 0 or q_len > _ATTN_TKG_MAX_S_ACTIVE:
            return f"s_active={q_len} is outside [1, {_ATTN_TKG_MAX_S_ACTIVE}]"
        if self.num_key_value_heads <= 0 or self.num_heads % self.num_key_value_heads != 0:
            return "num_heads must be divisible by num_key_value_heads for grouped mapping"
        q_heads_per_kv = self.num_heads // self.num_key_value_heads
        if q_heads_per_kv > _ATTN_TKG_MAX_Q_HEAD:
            return f"q_head per kv ({q_heads_per_kv}) exceeds attention_tkg max {_ATTN_TKG_MAX_Q_HEAD}"
        return None

    def _attention_tkg_decode_disable_reason(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor]],
        active_mask: Optional[torch.Tensor],
    ) -> Optional[str]:
        if past_key_value is None or len(past_key_value) < 2:
            return "past_key_value missing"
        if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
            return "Q/K/V must be rank-4 tensors"

        bsz = Q.shape[0]
        q_len = Q.shape[2]
        static_reason = self._attention_tkg_static_disable_reason(q_len)
        if static_reason is not None:
            return static_reason

        if K.shape[0] != bsz or V.shape[0] != bsz:
            return "Q/K/V batch mismatch"
        if K.shape[1] != self.num_key_value_heads or V.shape[1] != self.num_key_value_heads:
            return "K/V key-value head count mismatch"
        if Q.shape[1] != self.num_heads:
            return "Q head count mismatch"
        if Q.shape[3] != self.head_dim or K.shape[3] != self.head_dim or V.shape[3] != self.head_dim:
            return "Q/K/V head_dim mismatch"

        if attention_mask is None or attention_mask.ndim != 4:
            return "attention_mask must be rank-4 for attention_tkg"
        if attention_mask.shape[0] != bsz:
            return "attention_mask batch mismatch"
        if attention_mask.shape[2] != q_len:
            return "attention_mask q_len mismatch"
        s_prior = int(attention_mask.shape[-1])
        if s_prior <= 0:
            return "attention_mask has empty prior length"
        if s_prior > _ATTN_TKG_MAX_S_PRIOR:
            return f"curr_sprior={s_prior} exceeds attention_tkg max {_ATTN_TKG_MAX_S_PRIOR}"
        if s_prior > 256 and s_prior % 128 != 0:
            return "curr_sprior must be <=256 or divisible by 128"

        K_prior, V_prior = past_key_value[0], past_key_value[1]
        if K_prior.ndim != 4 or V_prior.ndim != 4:
            return "K/V cache must be rank-4 tensors"
        if K_prior.shape[0] < bsz or V_prior.shape[0] < bsz:
            return "K/V cache batch is smaller than active batch"
        if K_prior.shape[1] != self.num_key_value_heads or V_prior.shape[1] != self.num_key_value_heads:
            return "K/V cache key-value head count mismatch"

        if self.k_cache_transposed:
            s_prior_full = int(K_prior.shape[-1])
            if K_prior.shape[2] != self.head_dim:
                return "transposed K cache head_dim mismatch"
        else:
            s_prior_full = int(K_prior.shape[-2])
            if K_prior.shape[-1] != self.head_dim:
                return "non-transposed K cache head_dim mismatch"
        if V_prior.shape[-1] != self.head_dim:
            return "V cache head_dim mismatch"
        if s_prior > s_prior_full:
            return "curr_sprior exceeds KV cache capacity"
        if s_prior_full > _ATTN_TKG_MAX_S_PRIOR:
            return f"full_sprior={s_prior_full} exceeds attention_tkg max {_ATTN_TKG_MAX_S_PRIOR}"

        q_heads_per_kv = self.num_heads // self.num_key_value_heads
        bs_for_kernel = bsz * self.num_key_value_heads
        if bs_for_kernel > _ATTN_TKG_MAX_BS:
            return f"flattened kernel batch {bs_for_kernel} exceeds attention_tkg max {_ATTN_TKG_MAX_BS}"
        if bs_for_kernel * q_heads_per_kv * q_len > 128:
            return "batch*q_head*s_active exceeds attention_tkg fused RoPE partition limit"

        if active_mask is not None:
            if active_mask.ndim != 4:
                return "active_mask must be rank-4 when provided"
            if active_mask.shape[0] != bsz or active_mask.shape[2] != q_len or active_mask.shape[3] != q_len:
                return "active_mask shape mismatch"
            if active_mask.shape[1] not in (1, self.num_heads):
                return "active_mask head dimension must be 1 or num_heads"

        if attention_mask.shape[1] not in (1, self.num_heads):
            return "attention_mask head dimension must be 1 or num_heads"

        return None

    def _prepare_tokengen_fallback_qk(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        rotary_position_ids: Optional[torch.LongTensor] = None,
        use_polar_compatible_rope: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._tkg_decode_skip_rope_active:
            return Q, K
        rope_position_ids = rotary_position_ids if rotary_position_ids is not None else position_ids
        if rope_position_ids is None:
            return Q, K

        Q_rope, K_rope, _, _ = self.apply_rotary_embedding(
            Q,
            K,
            V,
            rope_position_ids,
            cos_cache=None,
            sin_cache=None,
            use_polar_compatible_rope=use_polar_compatible_rope,
        )
        K.copy_(K_rope)
        return Q_rope, K

    def _attention_cte_disable_reason(self, q_len: int, bsz: int) -> Optional[str]:
        if attention_cte is None:
            return "attention_cte import unavailable"
        if not self.attn_cte_kernel_enabled:
            return "attn_cte_kernel_enabled is False"
        if self.cp_degree > 1:
            return "cp_degree > 1"
        if self.attention_chunk_size is not None:
            return "attention_chunk_size is set"
        if self.sliding_window:
            return "sliding_window attention is enabled"
        if self.head_dim > 128:
            return "head_dim > 128 (attention_cte limit)"
        if q_len > 36864:
            return "q_len > 36864 (attention_cte limit)"
        if bsz > 32:
            return "batch size > 32 (attention_cte limit)"
        return None

    def _extract_uniform_prior_used_len(
        self, attention_mask: Optional[torch.Tensor], bsz: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None

        mask_i32 = attention_mask.to(torch.int32)
        if mask_i32.ndim == 1:
            per_batch = mask_i32.sum().reshape(1)
        else:
            counts = mask_i32.sum(dim=-1)
            if counts.ndim == 1:
                per_batch = counts
            else:
                flat = counts.reshape(counts.shape[0], -1)
                min_counts = flat.min(dim=1).values
                max_counts = flat.max(dim=1).values
                if not torch.equal(min_counts, max_counts):
                    return None
                per_batch = min_counts

        if per_batch.shape[0] == 1 and bsz > 1:
            per_batch = per_batch.expand(bsz)
        if per_batch.shape[0] != bsz:
            return None

        if not torch.equal(per_batch.min(), per_batch.max()):
            return None

        return per_batch[:1].to(device=device, dtype=torch.int32)

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        skip_rope=False,
        residual=None,
        use_polar_compatible_rope=False,
    ):
        tkg_skip_rope = False
        self._tkg_decode_rotary_position_ids = None
        if past_key_value is not None and not skip_rope and position_ids is not None:
            q_len = hidden_states.shape[1]
            tkg_skip_rope = self._attention_tkg_static_disable_reason(q_len) is None
            if tkg_skip_rope:
                self._tkg_decode_rotary_position_ids = position_ids
        self._tkg_decode_skip_rope_active = tkg_skip_rope

        return super().prep_qkv_tensors(
            position_ids=position_ids,
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            skip_rope=skip_rope or tkg_skip_rope,
            residual=residual,
            use_polar_compatible_rope=use_polar_compatible_rope,
        )

    def _build_attention_tkg_mask(
        self,
        attention_mask: torch.Tensor,
        active_mask: Optional[torch.Tensor],
        bsz: int,
        s_prior: int,
        q_len: int,
        q_heads_per_kv: int,
    ) -> torch.Tensor:
        mask = attention_mask.to(torch.uint8)
        if mask.shape[1] == 1:
            mask = mask.expand(-1, self.num_heads, -1, -1)
        mask = mask.permute(3, 0, 1, 2).contiguous()

        if active_mask is None:
            active = torch.ones(
                (bsz, 1, q_len, q_len), dtype=torch.uint8, device=attention_mask.device
            ).tril(diagonal=0)
        else:
            active = active_mask.to(torch.uint8)
        if active.shape[1] == 1:
            active = active.expand(-1, self.num_heads, -1, -1)
        active = active.permute(3, 0, 1, 2).contiguous()
        mask[-q_len:, :, :, :] = active

        mask = mask.reshape(s_prior, bsz, self.num_key_value_heads, q_heads_per_kv, q_len)
        mask = mask.reshape(s_prior, bsz * self.num_key_value_heads, q_heads_per_kv, q_len)
        return mask.contiguous()

    def _build_attention_tkg_active_mask(
        self,
        active_mask: Optional[torch.Tensor],
        bsz: int,
        q_len: int,
        q_heads_per_kv: int,
        device: torch.device,
    ) -> torch.Tensor:
        if active_mask is None:
            mask = torch.ones((bsz, 1, q_len, q_len), dtype=torch.uint8, device=device).tril(diagonal=0)
        else:
            mask = active_mask.to(torch.uint8)

        if mask.shape[1] == 1:
            mask = mask.expand(-1, self.num_heads, -1, -1)
        mask = mask.permute(3, 0, 1, 2).contiguous()
        mask = mask.reshape(q_len, bsz, self.num_key_value_heads, q_heads_per_kv, q_len)
        mask = mask.reshape(q_len, bsz * self.num_key_value_heads, q_heads_per_kv, q_len)
        return mask.contiguous()

    def attention_tokengen(
        self,
        Q,
        K,
        V,
        attention_mask,
        position_ids,
        past_key_value,
        active_mask,
        **kwargs,
    ):
        global _attn_tkg_active_logged, _attn_tkg_skip_logged, _attn_tkg_error_logged, _attn_tkg_posid_mismatch_logged

        use_polar_compatible_rope = kwargs.get("use_polar_compatible_rope", False)
        rotary_position_ids = self._tkg_decode_rotary_position_ids
        if (
            rotary_position_ids is None
            or position_ids is None
            or rotary_position_ids.shape != position_ids.shape
        ):
            rotary_position_ids = position_ids

        disable_reason = self._attention_tkg_decode_disable_reason(
            Q=Q,
            K=K,
            V=V,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            active_mask=active_mask,
        )
        if position_ids is None and disable_reason is None:
            disable_reason = "position_ids missing for attention_tkg fused RoPE"
        if rotary_position_ids is None and disable_reason is None:
            disable_reason = "rotary_position_ids missing for attention_tkg fused RoPE"

        if disable_reason is not None:
            if not _attn_tkg_skip_logged:
                logger.warning(
                    "attention_tkg NKI kernel not used; falling back to base token-gen attention: %s",
                    disable_reason,
                )
                _attn_tkg_skip_logged = True
            Q_fallback, K_fallback = self._prepare_tokengen_fallback_qk(
                Q=Q,
                K=K,
                V=V,
                position_ids=position_ids,
                rotary_position_ids=rotary_position_ids,
                use_polar_compatible_rope=use_polar_compatible_rope,
            )
            return super().attention_tokengen(
                Q_fallback,
                K_fallback,
                V,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                **kwargs,
            )

        bsz = Q.shape[0]
        K_prior_full, V_prior_full = past_key_value[0][:bsz], past_key_value[1][:bsz]
        pkv_kernel = (K_prior_full, V_prior_full)
        if len(past_key_value) > 2:
            pkv_kernel = tuple([K_prior_full, V_prior_full, *past_key_value[2:]])
        pos_ids_kernel = position_ids[:bsz]
        rope_pos_ids_kernel = rotary_position_ids[:bsz]

        try:
            if not _attn_tkg_active_logged:
                logger.warning(
                    "attention_tkg NKI kernel is ACTIVE for token generation via builtin fused-TKG runtime (use_pos_id=%s)",
                    self.attn_tkg_use_pos_id,
                )
                _attn_tkg_active_logged = True
            logger.debug("Using attention_tkg NKI kernel for token generation")
            if not torch.equal(rope_pos_ids_kernel, pos_ids_kernel) and not _attn_tkg_posid_mismatch_logged:
                logger.warning(
                    "attention_tkg using rotary_position_ids for RoPE and position_ids for causal offset in token generation"
                )
                _attn_tkg_posid_mismatch_logged = True

            attn_output, k_output = self.attention_tokengen_kernel_builtin(
                Q=Q,
                K=K,
                V=V,
                position_ids=pos_ids_kernel,
                past_key_value=pkv_kernel,
                attention_mask=attention_mask[:bsz],
                active_mask=active_mask[:bsz] if active_mask is not None else None,
                rotary_position_ids=rope_pos_ids_kernel,
            )

            # Propagate fused-RoPE K from kernel output into in-flight K so base KV update stores roped keys.
            if k_output is None:
                raise RuntimeError("attention_tkg builtin kernel returned empty k_out")
            K.copy_(k_output.to(dtype=K.dtype))
            return attn_output

        except Exception as exc:
            if not _attn_tkg_error_logged:
                logger.warning(
                    "attention_tkg NKI token-gen failed; falling back to base token-gen attention. error=%s",
                    exc,
                )
                _attn_tkg_error_logged = True
            Q_fallback, K_fallback = self._prepare_tokengen_fallback_qk(
                Q=Q,
                K=K,
                V=V,
                position_ids=position_ids,
                rotary_position_ids=rotary_position_ids,
                use_polar_compatible_rope=use_polar_compatible_rope,
            )
            return super().attention_tokengen(
                Q_fallback,
                K_fallback,
                V,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                **kwargs,
            )

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask):
        global _attn_cte_active_logged, _attn_cte_skip_logged, _attn_cte_error_logged

        disable_reason = self._attention_cte_disable_reason(q_len, bsz)
        if disable_reason is not None:
            if not _attn_cte_skip_logged:
                logger.warning("attention_cte NKI kernel not used; falling back to base prefill: %s", disable_reason)
                _attn_cte_skip_logged = True
            return super().perform_prefill(Q, K, V, q_len, bsz, attention_mask)

        try:
            if not _attn_cte_active_logged:
                logger.warning("attention_cte NKI kernel is ACTIVE for context encoding prefill")
                _attn_cte_active_logged = True
            logger.debug("Using attention_cte NKI kernel for context encoding prefill")
            q = Q.reshape((bsz * self.num_heads, q_len, self.head_dim)).to(self.torch_dtype)
            k = (
                K.reshape((bsz * self.num_key_value_heads, q_len, self.head_dim))
                .permute(0, 2, 1)
                .to(self.torch_dtype)
            )
            v = V.reshape((bsz * self.num_key_value_heads, q_len, self.head_dim)).to(self.torch_dtype)

            attn_output = attention_cte(
                q,
                k,
                v,
                scale=1.0 / math.sqrt(self.head_dim),
                causal_mask=attention_mask is not None,
                tp_q=True,
                tp_k=False,
                tp_out=True,
            )
            attn_output = attn_output.reshape((bsz, self.num_heads, self.head_dim, q_len))
            return attn_output, FlashAttentionStrategy.UNSHARDED_KERNEL
        except Exception as exc:
            if not _attn_cte_error_logged:
                logger.warning(
                    "attention_cte NKI prefill failed; falling back to base prefill. error=%s",
                    exc,
                )
                _attn_cte_error_logged = True
            return super().perform_prefill(Q, K, V, q_len, bsz, attention_mask)

    def perform_prefix_prefill(self, Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask):
        global _attn_cte_active_logged, _attn_cte_skip_logged, _attn_cte_error_logged

        disable_reason = self._attention_cte_disable_reason(q_len, bsz)
        if disable_reason is not None:
            if not _attn_cte_skip_logged:
                logger.warning("attention_cte NKI kernel not used; falling back to base prefix prefill: %s", disable_reason)
                _attn_cte_skip_logged = True
            return super().perform_prefix_prefill(
                Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask
            )
        if past_key_value is None or len(past_key_value) < 2:
            return super().perform_prefix_prefill(
                Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask
            )

        K_prior, V_prior = past_key_value[0], past_key_value[1]
        if K_prior.ndim != 4 or V_prior.ndim != 4:
            return super().perform_prefix_prefill(
                Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask
            )

        prior_used_len = self._extract_uniform_prior_used_len(attention_mask, bsz, Q.device)
        if prior_used_len is None:
            return super().perform_prefix_prefill(
                Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask
            )

        s_prior = K_prior.shape[-1] if self.k_cache_transposed else K_prior.shape[-2]
        if int(prior_used_len.item()) > s_prior:
            return super().perform_prefix_prefill(
                Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask
            )

        try:
            if not _attn_cte_active_logged:
                logger.warning("attention_cte NKI kernel is ACTIVE for prefix-caching context encoding")
                _attn_cte_active_logged = True
            logger.debug("Using attention_cte NKI kernel for prefix-caching context encoding")
            q = Q.reshape((bsz * self.num_heads, q_len, self.head_dim)).to(self.torch_dtype)
            k = (
                K.reshape((bsz * self.num_key_value_heads, q_len, self.head_dim))
                .permute(0, 2, 1)
                .to(self.torch_dtype)
            )
            v = V.reshape((bsz * self.num_key_value_heads, q_len, self.head_dim)).to(self.torch_dtype)

            if self.k_cache_transposed:
                k_prior = K_prior.reshape((bsz * self.num_key_value_heads, self.head_dim, s_prior))
            else:
                k_prior = K_prior.reshape((bsz * self.num_key_value_heads, s_prior, self.head_dim)).permute(
                    0, 2, 1
                )
            k_prior = k_prior.to(self.torch_dtype)
            v_prior = V_prior.reshape((bsz * self.num_key_value_heads, s_prior, self.head_dim)).to(
                self.torch_dtype
            )

            attn_output = attention_cte(
                q,
                k,
                v,
                scale=1.0 / math.sqrt(self.head_dim),
                causal_mask=True,
                k_prior=k_prior,
                v_prior=v_prior,
                prior_used_len=prior_used_len,
                tp_q=True,
                tp_k=False,
                tp_out=True,
            )
            attn_output = attn_output.reshape((bsz, self.num_heads, self.head_dim, q_len))
            return attn_output, FlashAttentionStrategy.UNSHARDED_KERNEL
        except Exception as exc:
            if not _attn_cte_error_logged:
                logger.warning(
                    "attention_cte NKI prefix prefill failed; falling back to base prefix prefill. error=%s",
                    exc,
                )
                _attn_cte_error_logged = True
            return super().perform_prefix_prefill(
                Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask
            )


class NeuronQwen3MoeDecoderLayer(nn.Module):
    """Decoder layer with attention_cte for prefill and nki_moe_fused for token generation.

    CTE (q_len > 1): Uses the standard NxD MoE module (initialize_moe_module).
    TKG (q_len == 1): Calls nki_moe_fused kernel directly, bypassing the standard
        MoE module for higher performance during token generation.
    """

    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3MoEAttention(config=config)
        self.moe_fused_nki_kernel_enabled = getattr(config, "moe_fused_nki_kernel_enabled", True)

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Standard MoE module — used for CTE and as the weight store for TKG.
        # rmsnorm is applied internally by self.mlp, so callers pass un-normed input.
        # The nki_moe_fused path applies norm explicitly since it bypasses self.mlp.
        self.mlp = initialize_moe_module(
            config=config, rmsnorm=self.post_attention_layernorm,  init_tkg_module=True
        )

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens

        # nki_moe_fused configuration
        self.normalize_top_k_affinities = config.neuron_config.normalize_top_k_affinities
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Check if nki_moe_fused can be used: I_tp must be padded to multiple of 128.
        # moe_tp_degree = getattr(config.neuron_config, "moe_tp_degree", 1)
        # if moe_tp_degree == 1 and config.neuron_config.tp_degree > 1:
        #     moe_tp_degree = config.neuron_config.tp_degree
        # I_tp = config.moe_intermediate_size // moe_tp_degree
        # self._i_tp = I_tp
        # self._i_tp_padded = ((I_tp + 127) // 128) * 128
        # self._i_tp_needs_pad = (I_tp % 128 != 0)
        # self.moe_fused_direct_enabled = getattr(
        #     config.neuron_config, "moe_fused_direct_enabled", True
        # )

    # ------------------------------------------------------------------
    # nki_moe_fused forward helper for token generation
    # ------------------------------------------------------------------

    # def _forward_moe_fused(self, hidden_states: torch.Tensor, bsz: int) -> torch.Tensor:
    #     """Call nki_moe_fused for one decode step.

    #     Args:
    #         hidden_states: [bsz, 1, H] post-RMSNorm
    #         bsz: batch size (= number of tokens T during decode)

    #     Returns:
    #         [bsz, 1, H] MoE output (after TP all-reduce)
    #     """
    #     global _moe_fused_active_logged
    #     if not _moe_fused_active_logged:
    #         logger.warning("nki_moe_fused NKI kernel ACTIVE for token generation")
    #         _moe_fused_active_logged = True

    #     T = bsz
    #     H = hidden_states.shape[-1]
    #     device = hidden_states.device
    #     hidden_2d = hidden_states.reshape(T, H)  # [T, H]

    #     # --- Router: get affinities and top-K indices ---
    #     _, affinities_full, expert_index = self.mlp.router(hidden_2d)
    #     # affinities_full: [T, E] (softmax over all experts)
    #     # expert_index:     [T, K] int64

    #     E = self.num_experts

    #     # Build routing_weights [T, E]: sparse tensor with normalized top-K probs.
    #     # Non-selected experts have weight 0, so the kernel skips their contribution.
    #     if self.normalize_top_k_affinities:
    #         topk_probs = affinities_full.gather(1, expert_index)           # [T, K]
    #         topk_normalized = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    #         routing_weights = torch.zeros(T, E, device=device, dtype=torch.float32)
    #         routing_weights.scatter_(1, expert_index, topk_normalized.to(torch.float32))
    #     else:
    #         routing_weights = affinities_full.to(torch.float32)

    #     # --- Weights (TP-sharded on intermediate dimension I) ---
    #     gate_up_w = self.mlp.expert_mlps.mlp_op.gate_up_proj.weight  # [E, H, 2*I_tp]
    #     down_w    = self.mlp.expert_mlps.mlp_op.down_proj.weight      # [E, I_tp, H]

    #     # Pad intermediate dimension to multiple of 128 if needed.
    #     if self._i_tp_needs_pad:
    #         pad_size = self._i_tp_padded - self._i_tp
    #         # gate_up_w: [E, H, 2*I_tp] → split into gate [E,H,I_tp] and up [E,H,I_tp],
    #         # pad each, then concat back
    #         gate_w = gate_up_w[:, :, :self._i_tp]
    #         up_w = gate_up_w[:, :, self._i_tp:]
    #         gate_w = F.pad(gate_w, (0, pad_size))
    #         up_w = F.pad(up_w, (0, pad_size))
    #         gate_up_w = torch.cat([gate_w, up_w], dim=2)  # [E, H, 2*I_tp_padded]
    #         # down_w: [E, I_tp, H] → pad on dim=1
    #         down_w = F.pad(down_w, (0, 0, 0, pad_size))   # [E, I_tp_padded, H]

    #     # --- nki_moe_fused kernel call ---
    #     output = _nki_moe_fused(
    #         hidden_2d,         # [T, H] bf16
    #         gate_up_w,         # [E, H, 2*I_tp_padded] bf16
    #         down_w,            # [E, I_tp_padded, H] bf16
    #         routing_weights,   # [T, E] float32
    #     )  # [T, H]

    #     # --- TP all-reduce ---
    #     output = mappings.reduce_from_tensor_model_parallel_region(
    #         output, process_group=parallel_state.get_world_group()
    #     )

    #     return output.reshape(bsz, 1, H)

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
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        global _moe_fused_skip_logged, _moe_fused_error_logged

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        qkv_fused_rmsnorm = None
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

        # ---- MoE block ----
        # self.mlp applies post_attention_layernorm internally.
        # The nki_moe_fused path needs explicit norm since it bypasses self.mlp.
      
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
        # Cascaded TKG kernel requires every bucket to be a multiple of 128
        # (n_sprior_tile = ceil(s_prior/128) but accesses up to n_sprior_tile*128-1).
        # Snap the bucket list that was just generated by generate_buckets_for_tkg.
        tkg_nc = self.token_generation_model.config.neuron_config
        def _snap128(b):
            return b if b <= 256 or b % 128 == 0 else ((b + 127) // 128) * 128
        if tkg_nc.buckets and not isinstance(tkg_nc.buckets[0], list):
            snapped = sorted(set(_snap128(b) for b in tkg_nc.buckets))
            if snapped != tkg_nc.buckets:
                logger.warning(
                    "attn_block_tkg: snapping TKG buckets %s → %s "
                    "(cascaded kernel requires s_prior %% 128 == 0)",
                    tkg_nc.buckets, snapped,
                )
                tkg_nc.buckets = snapped

    def get_compiler_args(self):
        # Build args as a list so each element is a complete, unambiguous argument.
        # shlex.join() quotes any values that contain spaces, and the base class
        # calls shlex.split() before passing to the compiler, so the round-trip is safe.
        args = [
            "--enable-saturate-infinity",
            "--enable-mixed-precision-accumulation",
            "--model-type", "transformer",
        ]

        # Set compiler optimization level based on model tag
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=4",
                "--vectorize-strided-dma",
                "--enable-scalar-dge-vectorization",
                "--enable-dmacopy-transpose",
            ]
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O3"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=1",
                "--vectorize-strided-dma",
                "--enable-scalar-dge-vectorization",
                "--enable-dmacopy-transpose",
                "--eager-tkg-vectorize-dma",
                "--enable-dge-on-indirect-dma",
                "--enable-dge-on-vector-indirect-dma",
            ]
        else:
            optimization_level = "-O1"
            tensorizer_opts = []

        if tensorizer_opts:
            # Join sub-options into a single value; shlex.join quotes the resulting
            # element so spaces inside the value survive the base class's shlex.split().
            args.append(f"--tensorizer-options={' '.join(tensorizer_opts)}")

        args.append(optimization_level)
        args += ["--auto-cast=none", "--auto-cast-type=bf16"]
        # Enable vector-offset DGE (two separate tokens)
        args += ["--internal-enable-dge-levels", "vector_dynamic_offsets"]
        # hlo2tensorizer sub-options (value contains spaces; shlex.join will quote it)
        args.append("--internal-hlo2tensorizer-options=--verify-hlo=true")

        if self.neuron_config.scratchpad_page_size:
            args.append(f"--hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size}")

        if self.neuron_config.attn_block_tkg_nki_kernel_enabled:
            assert (
                self.neuron_config.attn_block_tkg_nki_kernel_cascaded_attention
            ), "If using attn_block_tkg_nki_kernel_enabled for Qwen3MoE you must also use attn_block_tkg_nki_kernel_cascaded_attention"
            # Enabled RMSNorm pre-RoPE in the Attn TKG MK
            self.neuron_config.pre_rope_rmsnorm = True
            # When enabling the Cascaded Attn TKG MK we will run over 5 million instructions on E2E
            args.append("--internal-max-instruction-limit=15000000")

        return shlex.join(args)


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
