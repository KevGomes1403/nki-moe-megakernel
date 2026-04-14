# coding=utf-8
"""
Qwen3 MoE model with transformer_qwen3_moe_tkg_v2_jit for TKG.

CTE path (past_key_value is None):
    Standard flash attention + standard MoE (self.mlp unchanged).

TKG path (past_key_value is not None):
    Full fused transformer layer via transformer_qwen3_moe_tkg_v2_jit:
    v13bc attention (plain layout) + bare AllReduce #1 + MoE + AllReduce #2.

Weight layouts:
  Attention (plain layout for v13bc, no tile-transpose):
    q_proj.weight   [Hq_tp*d, H]  = [1024, 2048]  reused directly from NeuronAttentionBase
    k_proj.weight   [Hkv_tp*d, H] = [128,  2048]  reused directly
    v_proj.weight   [Hkv_tp*d, H] = [128,  2048]  reused directly
    Wo_nki.weight   [Hq_tp*d, H]  = [1024, 2048]  plain T (o_proj.weight.T)
  MoE (native layout):
    gate_up_proj.weight  [E, H, 2*I_tp]  gate cols 0:I_tp, up cols I_tp:2*I_tp
    down_proj.weight     [E, I_tp, H]

NOTE: The kernel's residual-add step 9 is currently commented out in
transformer_qwen_v3_v2.py, so Y will be zeros. TKG path is for compilation
testing only.
"""

import gc
import shlex
import warnings
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.modules.moe.moe_configs import BlockwiseMatmulConfig
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeInferenceConfig,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

from kernels.transformer.transformer_qwen import transformer_qwen3_moe_tkg_v2_jit

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE

import os
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"  
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"

# ---------------------------------------------------------------------------
# Neuron config
# ---------------------------------------------------------------------------

class Qwen3MoEV2TestNeuronConfig(MoENeuronConfig):
    """MoENeuronConfig for transformer_qwen3_moe_tkg_v2_jit test."""

    def __init__(self, **kwargs):
        if "blockwise_matmul_config" not in kwargs:
            kwargs["blockwise_matmul_config"] = BlockwiseMatmulConfig.from_kwargs(
                block_size=256,
                logical_nc_config=2,
                normalize_top_k_affinities=True,
                use_shard_on_block_dynamic_while=True,
                block_sharding_strategy="PING_PONG",
            )
        # Disable KV cache slicing so the kernel receives the full [B, 1, S_prior, d] cache.
        kwargs["attn_tkg_nki_kernel_enabled"] = True
        kwargs.setdefault("fused_qkv", False)
        kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(**kwargs)
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_rmsnorm_cls():
    return Qwen3MoeRMSNorm if cpu_mode() else CustomRMSNorm


def get_modules_to_not_convert(neuron_config: MoENeuronConfig):
    return getattr(neuron_config, "modules_to_not_convert", None)


def maybe_dequantize_layer(neuron_state_dict, config):
    scale_layers = []
    for layer_key in neuron_state_dict.keys():
        if "_scale_inv" in layer_key:
            scales = neuron_state_dict[layer_key]
            scale_layers.append(layer_key)
            fp8_layer_name = layer_key.replace("_scale_inv", "")
            fp8_layer = neuron_state_dict[fp8_layer_name]
            block_size = config.quantization_config["weight_block_size"]
            scales_expanded = (
                scales.repeat_interleave(block_size[0], dim=0)
                      .repeat_interleave(block_size[1], dim=1)
            )
            scaled_layer = fp8_layer.to(torch.float32) * scales_expanded.to(torch.float32)
            neuron_state_dict[fp8_layer_name] = scaled_layer.to(config.neuron_config.torch_dtype)
    for scale_layer in scale_layers:
        del neuron_state_dict[scale_layer]


# ---------------------------------------------------------------------------
# State dict converter
# ---------------------------------------------------------------------------

def convert_qwen3_moe_hf_to_neuron_state_dict(neuron_state_dict, config):
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"
    maybe_dequantize_layer(neuron_state_dict, config)
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    for l in range(config.num_hidden_layers):  # noqa: E741
        neuron_state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )
        # Rename q/k norm weights.
        neuron_state_dict[f"layers.{l}.self_attn.k_layernorm.weight"] = (
            neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"]
        neuron_state_dict[f"layers.{l}.self_attn.q_layernorm.weight"] = (
            neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"]

        # Wo_nki: plain transpose of o_proj weight for v13bc kernel.
        # q_proj, k_proj, v_proj are reused directly — already plain [out_tp, H] after TP sharding.
        o_proj_w = neuron_state_dict[f"layers.{l}.self_attn.o_proj.weight"]   # [H, Hq]
        neuron_state_dict[f"layers.{l}.self_attn.Wo_nki.weight"] = (
            o_proj_w.T.contiguous()
        )

        # Router weight: HF gate.weight [E, H] → linear_router.weight.
        neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
            neuron_state_dict[f"layers.{l}.mlp.gate.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.mlp.gate.weight"]

        # MoE weights: native layout [E, H, 2*I] and [E, I, H].
        intermediate_size, hidden_size = neuron_state_dict[
            f"layers.{l}.mlp.experts.0.gate_proj.weight"
        ].shape
        device = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"].device
        dtype = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"].dtype
        gate_up_proj = torch.empty(
            config.num_experts, hidden_size, 2 * intermediate_size, dtype=dtype, device=device,
        )
        for e in range(config.num_experts):
            gate_proj_weights = (
                neuron_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"].T.detach().clone()
            )
            up_proj_weights = (
                neuron_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"].T.detach().clone()
            )
            gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
            gate_proj_slice = torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size)
            gate_proj_slice.copy_(gate_proj_weights)
            up_proj_slice = torch.narrow(gate_up_proj_slice, 2, intermediate_size, intermediate_size)
            up_proj_slice.copy_(up_proj_weights)
            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"]
            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"]
        pad_size = getattr(config, "moe_intermediate_pad_size", 0)
        if pad_size > 0:
            gate_up_proj = gate_up_proj.reshape(config.num_experts, hidden_size, 2, -1)
            gate_up_proj = torch.nn.functional.pad(gate_up_proj, (0, pad_size))
            gate_up_proj = gate_up_proj.reshape(config.num_experts, hidden_size, -1)
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        down_proj = torch.empty(
            config.num_experts, intermediate_size, hidden_size, dtype=dtype, device=device,
        )
        for e in range(config.num_experts):
            down_proj_weights = (
                neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"].T.detach().clone()
            )
            down_proj_slice = torch.narrow(down_proj, 0, e, 1)
            down_proj_slice.copy_(down_proj_weights)
            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"]
        if pad_size > 0:
            down_proj = torch.nn.functional.pad(down_proj, (0, 0, 0, pad_size))
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        gc.collect()

    return neuron_state_dict


# ---------------------------------------------------------------------------
# NKI attention (v2: only Wo_nki, plain layout for Q/K/V)
# ---------------------------------------------------------------------------

class NeuronQwen3MoEAttentionV2(NeuronAttentionBase):
    """
    Qwen3 MoE attention for the v2 transformer kernel test.

    CTE: delegates to NeuronAttentionBase.forward() (use_qk_norm=False).
    TKG: not called — the decoder layer invokes transformer_qwen3_moe_tkg_v2_jit
         directly, accessing q_proj/k_proj/v_proj weights from this class.

    Only Wo_nki is added as an extra weight holder; q_proj, k_proj, v_proj from
    NeuronAttentionBase are already in the plain [out_tp, H] layout that v13bc expects.
    """

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
            use_qk_norm=False,
        )
        self.q_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)
        self.k_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronQwen3MoEAttentionV2 must be initialized in a distributed env."
            )

        # Wo_nki: ColumnParallelLinear holder for o_proj.weight.T (plain transpose, no tile-transpose).
        # q/k/v weights are accessed via get_qkv_proj().q_proj/.k_proj/.v_proj (inside GQA).
        _dtype = config.neuron_config.torch_dtype
        _H = config.hidden_size
        _Hq_full = config.num_attention_heads * config.head_dim
        self._nki_d = config.head_dim
        self.Wo_nki = ColumnParallelLinear(_H, _Hq_full, bias=False, gather_output=False, dtype=_dtype)


# ---------------------------------------------------------------------------
# Decoder layer — v2 fused transformer TKG
# ---------------------------------------------------------------------------

class NeuronQwen3MoeDecoderLayerV2(nn.Module):
    """Decoder layer with transformer_qwen3_moe_tkg_v2_jit for TKG.

    CTE: standard flash attention + standard MoE (self.mlp unchanged).
    TKG: full fused transformer kernel (v13bc attention + AllReduce + MoE + AllReduce).

    NOTE: The kernel's residual-add step 9 is currently commented out in
    transformer_qwen_v3_v2.py, so Y will be zeros. TKG path is for compilation
    testing only.
    """

    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3MoEAttentionV2(config=config)

        _dtype = config.neuron_config.torch_dtype
        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps).to(_dtype)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps).to(_dtype)

        self.mlp = initialize_moe_module(config=config)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = False
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens

        # Replica groups for nccl.all_reduce inside the transformer kernel.
        # tp_groups are [[0, 1, ..., tp_degree-1]] — nccl.ReplicaGroup expects List[List[int]].
        # Derived from tp_degree to avoid torch.distributed calls during CTE mock tracing.
        self._replica_groups = (list(range(config.neuron_config.tp_degree)),)

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
        is_tkg = past_key_value is not None

        if is_tkg:
            # --- TKG: full fused transformer kernel ---
            hidden_states = ModuleMarkerStartWrapper()(hidden_states)
            X_residual_tkg = hidden_states   # original X (pre-norm) for residual adds
            hidden_states = self.input_layernorm(hidden_states)

            cos_cache, sin_cache = self.self_attn.rotary_emb(hidden_states, position_ids)
            cos_at_pos = cos_cache.squeeze(1)   # [B, d]
            sin_at_pos = sin_cache.squeeze(1)   # [B, d]

            K_cache, V_cache = past_key_value

            # Attention weights — plain layout, no tile-transpose needed for v13bc.
            # q/k/v live inside self.self_attn.qkv_proj (GQA), accessed via get_qkv_proj().
            _qkv = self.self_attn.get_qkv_proj()
            Wq = _qkv.q_proj.weight                  # [Hq_tp*d, H]
            Wk = _qkv.k_proj.weight                  # [Hkv_tp*d, H]
            Wv = _qkv.v_proj.weight                  # [Hkv_tp*d, H]
            Wo = self.self_attn.Wo_nki.weight        # [Hq_tp*d, H]  o_proj.weight.T

            q_norm_weight = self.self_attn.q_layernorm.weight   # [d]
            k_norm_weight = self.self_attn.k_layernorm.weight   # [d]

            # gamma_moe: post-attention layernorm weight for MoE RMSNorm inside kernel.
            gamma_moe = self.post_attention_layernorm.weight.unsqueeze(0)  # [1, H]

            # router_w: [H, E] — linear_router.weight is [E, H], transpose to [H, E].
            router_w = self.mlp.router.linear_router.weight.T.contiguous()
            gate_up_w = self.mlp.expert_mlps.mlp_op.gate_up_proj.weight   # [E, H, 2*I_tp]
            down_w    = self.mlp.expert_mlps.mlp_op.down_proj.weight       # [E, I_tp, H]

            # Full fused transformer layer:
            #   normed_X → attn (v13bc) → AllReduce #1 → residual #1 (original X + attn)
            #   → post_attn_norm → MoE → AllReduce+gather #2 → residual #2 → Y
            Y = transformer_qwen3_moe_tkg_v2_jit[2](
                hidden_states,
                X_residual_tkg,
                Wq, Wk, Wv, Wo,
                q_norm_weight, k_norm_weight,
                K_cache, V_cache,
                cos_at_pos, sin_at_pos,
                position_ids.to(torch.int32),
                gamma_moe,
                router_w, gate_up_w, down_w,
                replica_groups=self._replica_groups,
            )
            # present_key_value: kernel writes K/V to shared_hbm internally;
            # pass through cache unchanged for NxDI framework compatibility.
            present_key_value = past_key_value

            Y = ModuleMarkerEndWrapper()(Y)
            return (Y, present_key_value, cos_cache, sin_cache, None)

        else:
            # --- CTE: standard path ---
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
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, padding_mask)[0]
            hidden_states = residual + hidden_states

            hidden_states = ModuleMarkerEndWrapper()(hidden_states)
            return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NeuronQwen3MoeModelV2(NeuronBaseModel):
    def setup_attr_for_model(self, config: Qwen3MoeInferenceConfig):
        self.on_device_sampling = True
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
        self.layers = nn.ModuleList([
            NeuronQwen3MoeDecoderLayerV2(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


# ---------------------------------------------------------------------------
# CausalLM entry point
# ---------------------------------------------------------------------------

class NeuronQwen3MoeForCausalLMV2(NeuronBaseForCausalLM):
    """Qwen3 MoE CausalLM with transformer_qwen3_moe_tkg_v2_jit for TKG."""

    _model_cls = NeuronQwen3MoeModelV2

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return Qwen3MoeForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_neuron_config_cls(cls):
        return Qwen3MoEV2TestNeuronConfig

    @classmethod
    def get_config_cls(cls):
        return Qwen3MoeInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: Qwen3MoeInferenceConfig) -> dict:
        return convert_qwen3_moe_hf_to_neuron_state_dict(state_dict, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self):
        args = [
            "--enable-saturate-infinity",
            "--enable-mixed-precision-accumulation",
            "--model-type",
            "transformer",
        ]
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=4",
                "--vectorize-strided-dma",
                "--enable-scalar-dge-vectorization",
            ]
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O3"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=1",
                "--vectorize-strided-dma",
                "--enable-scalar-dge-vectorization",
                "--eager-tkg-vectorize-dma",
                "--enable-dge-on-indirect-dma",
                "--enable-dge-on-vector-indirect-dma",
            ]
        else:
            optimization_level = "-O1"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=4",
                "--vectorize-strided-dma",
                "--enable-scalar-dge-vectorization",
            ]

        if tensorizer_opts:
            args.append(f"--tensorizer-options={' '.join(tensorizer_opts)}")

        args.append(optimization_level)
        args.append("--auto-cast=none")
        args += ["--internal-enable-dge-levels", "vector_dynamic_offsets"]
        args.append("--internal-hlo2tensorizer-options=--verify-hlo=true")

        if self.neuron_config.scratchpad_page_size:
            args.append(f"--hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size}")

        return shlex.join(args)


# Alias expected by main.py's `qwen.NeuronQwen3MoeForCausalLM` convention.
NeuronQwen3MoeForCausalLM = NeuronQwen3MoeForCausalLMV2
