# coding=utf-8
"""
Qwen3 MoE model — NxDI stock attention + per-layer MoE NKI kernel.

A/B isolation variant of `qwen_fused_transformer_multilayer.py`. The 48-layer
fused TKG megakernel is replaced with two per-layer pieces:
  * Attention: NxDI's stock `NeuronAttentionBase` (qwen.py-style — including
    its standard fused QKV NKI kernel and Python KV scatter).
  * MoE: `qwen3_moe_tkg_singlelayer_jit` — a thin HBM-in/HBM-out wrapper
    around `_qwen3_moe_sbuf_in_sbuf_out_hoisted`. RMSNorm (post_attention
    layernorm) is fused inside the wrapper. TP all-reduce + LNC gather are
    handled by `_sb2sb_all_reduce_gather` inside the wrapper.

Goal: determine whether the divergence observed end-to-end against NxDI is
attributable to the fused attention path (v14d) or the MoE path (v30c).
With this variant, only the MoE path differs from the NxDI reference.

Performance note: each per-layer kernel call eats HBM round-trips for the
residual that the megakernel kept SBUF-resident across all 48 layers. Plus
there are 48 separate kernel launches per token. This is an accuracy probe,
not a perf-competitive variant.

CTE: stock per-layer Python path (unchanged from qwen.py / V2).
TKG: per-layer flow with NxDI attention and the MoE wrapper kernel.

Debug capture mirrors the existing V3 file: layer-0 `attn_out`, `kv_k`,
`kv_v`, `final_hidden_states` are registered when
`tensor_capture_config` is provided in the neuron config.
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
from qwen import Qwen3MoeInferenceConfig    
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

from kernels.transformer.moe_tkg_single_layer import qwen3_moe_tkg_singlelayer_jit

try:
    from neuronx_distributed.utils.tensor_capture import register_tensor as _register_tensor
except ImportError:
    def _register_tensor(name, tensor):
        pass

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE

import os
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"


# ---------------------------------------------------------------------------
# Neuron config — drops the megakernel-specific attn flags so NxDI runs its
# stock attention path with normal Python KV scatter.
# ---------------------------------------------------------------------------

class Qwen3MoENxdAttnMoeKernelNeuronConfig(MoENeuronConfig):
    """MoENeuronConfig for the NxDI-attn + MoE-NKI-kernel variant."""

    def __init__(self, **kwargs):
        if not isinstance(kwargs.get("blockwise_matmul_config"), BlockwiseMatmulConfig):
            kwargs["blockwise_matmul_config"] = BlockwiseMatmulConfig.from_kwargs(
                use_torch_block_wise=True,
            )
        kwargs.setdefault("fused_qkv", False)
        # tensor_capture_config is not understood by OnDeviceSamplingConfig — pop it
        # before constructing OnDeviceSamplingConfig and put it back afterwards.
        _tcc = kwargs.pop("tensor_capture_config", None)
        kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(**kwargs)
        if _tcc is not None:
            kwargs["tensor_capture_config"] = _tcc
        kwargs["output_logits"] = True
        kwargs["async_mode"] = True
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
# State dict converter — qwen.py-style. No tile-transposed Wq/Wk/Wv/Wo
# (NxDI stock attention consumes the original q_proj/k_proj/v_proj/o_proj).
# Router weight is fp32-cast to match `router_config.dtype = torch.float32`.
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
        neuron_state_dict[f"layers.{l}.self_attn.k_layernorm.weight"] = (
            neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"]
        neuron_state_dict[f"layers.{l}.self_attn.q_layernorm.weight"] = (
            neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"]

        neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
            neuron_state_dict[f"layers.{l}.mlp.gate.weight"].detach().clone().to(torch.float32)
        )
        del neuron_state_dict[f"layers.{l}.mlp.gate.weight"]

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
# Attention — verbatim NxDI stock from qwen.py. RMSNorm overrides for q/k.
# ---------------------------------------------------------------------------

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
            use_qk_norm=False,
        )

        # Override q/k layernorm with RMSNorm (NeuronAttentionBase's qk_norm
        # is a different normalization — Qwen3 uses RMSNorm on q and k).
        self.q_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)
        self.k_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronQwen3MoEAttention has to be initialized in a distributed env."
            )


# ---------------------------------------------------------------------------
# Decoder layer — single per-layer flow.
#   CTE: stock NxDI Python MoE (matches qwen.py).
#   TKG: NxDI attention + per-layer MoE wrapper kernel.
# ---------------------------------------------------------------------------

class NeuronQwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3MoEAttention(config=config)

        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        # Plain Python MoE module — used for CTE. On TKG we bypass `self.mlp`
        # entirely and call the wrapper kernel with the underlying weights.
        self.mlp = initialize_moe_module(config=config)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens
        self.debug_tensor_capture = (
            getattr(config.neuron_config, "tensor_capture_config", None) is not None
        )

        # Replica groups for the wrapper kernel's nccl all-reduce.
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

        # Discriminate TKG vs CTE from past_key_value (matches V3 docstring
        # convention) so we don't have to consume an explicit kwarg —
        # everything else flows through to self_attn unchanged, just like qwen.py.
        is_tkg = past_key_value is not None

        residual = hidden_states

        qkv_fused_rmsnorm = None
        # Match qwen.py exactly: capture residual BEFORE the module marker.
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)

        # NxDI stock attention — handles QKV proj, RoPE, KV scatter, attention,
        # o_proj, and TP all-reduce internally. Forward all remaining kwargs
        # (is_for_context_encoding, kv_mgr, etc.) as-is so NxDI's CTE/TKG
        # dispatch matches qwen.py.
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )
        if self.debug_tensor_capture and self.layer_idx == 0:
            _register_tensor("attn_out", hidden_states)
            _register_tensor("kv_k", present_key_value[0])
            _register_tensor("kv_v", present_key_value[1])
        hidden_states = residual + hidden_states

        residual = hidden_states

        if is_tkg:
            # MoE wrapper kernel — fuses post_attention_layernorm internally
            # via `gpost`, so do NOT apply self.post_attention_layernorm here.
            router_w  = self.mlp.router.linear_router.weight.T.contiguous()  # [H, E]
            gate_up_w = self.mlp.expert_mlps.mlp_op.gate_up_proj.weight       # [E, H, 2*I_pad]
            down_w    = self.mlp.expert_mlps.mlp_op.down_proj.weight          # [E, I_pad, H]
            gpost     = self.post_attention_layernorm.weight.unsqueeze(0)     # [1, H]

            moe_out = qwen3_moe_tkg_singlelayer_jit[2](
                hidden_states, gpost, router_w, gate_up_w, down_w,
                replica_groups=self._replica_groups,
            )
            hidden_states = residual + moe_out
        else:
            # CTE: stock NxDI Python MoE (matches qwen.py's else branch when
            # moe_fused_nki_kernel_enabled is False).
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, padding_mask)[0]
            hidden_states = residual + hidden_states

        if self.debug_tensor_capture and self.layer_idx == 0:
            _register_tensor("final_hidden_states", hidden_states)

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Pre-transposed LM head (perf only — orthogonal to the A/B test).
# ---------------------------------------------------------------------------

class PreTransposedColumnParallelLinear(ColumnParallelLinear):
    """Stores lm_head weight as [hidden, vocab_per_partition] instead of
    [vocab_per_partition, hidden], so HBM loads avoid transpose_mode=ENABLED."""

    def set_weight_and_bias_config(self):
        self.weight_shape = (self.input_size, self.output_size_per_partition)
        self.weight_partition_dim = 1
        self.bias_shape = None

    def preshard_hook(self, model_state_dict, prefix):
        if prefix.endswith("weight"):
            w = model_state_dict[prefix]
            model_state_dict[prefix] = w.t().contiguous()

    def forward(self, input, slice_indices=None):
        input_parallel = self._cpl_maybe_input_copy_to_tp_region(input)
        weight = self.weight[:, slice_indices] if slice_indices is not None else self.weight
        output_parallel = torch.matmul(input_parallel, weight)
        return self._cpl_maybe_gather_output(output_parallel)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NeuronQwen3MoeNxdAttnMoeKernelModel(NeuronBaseModel):
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
            NeuronQwen3MoeDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = PreTransposedColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            bias=False,
        )


# ---------------------------------------------------------------------------
# CausalLM entry point
# ---------------------------------------------------------------------------

class NeuronQwen3MoeForCausalLMNxdAttnMoeKernel(NeuronBaseForCausalLM):
    """Qwen3 MoE CausalLM with NxDI stock attention + per-layer MoE NKI kernel."""

    _model_cls = NeuronQwen3MoeNxdAttnMoeKernelModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return Qwen3MoeForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_neuron_config_cls(cls):
        return Qwen3MoENxdAttnMoeKernelNeuronConfig

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
            f"--lnc={self.neuron_config.logical_nc_config}",
        ]
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=4",
                "--vectorize-strided-dma",
                "--enable-scalar-dge-vectorization",
            ]
            hlo2tensorizer_extra = "--modular-flow-mac-threshold=10"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            # Per-layer kernels (no megakernel) — no need for the giant
            # instruction limit or the in-place-KV-scatter verifier override.
            optimization_level = "-O1"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=4",
                "--vectorize-strided-dma",
                "--eager-tkg-vectorize-dma",
                "--enable-dge-on-indirect-dma",
                "--enable-dge-on-vector-indirect-dma",
            ]
            hlo2tensorizer_extra = ""
        else:
            optimization_level = "-O1"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=4",
                "--vectorize-strided-dma",
                "--enable-scalar-dge-vectorization",
            ]
            hlo2tensorizer_extra = ""

        if tensorizer_opts:
            args.append(f"--tensorizer-options={' '.join(tensorizer_opts)}")

        args.append(optimization_level)
        args.append("--auto-cast=none")
        args += ["--internal-enable-dge-levels", "vector_dynamic_offsets"]

        if self.neuron_config.scratchpad_page_size:
            args.append(f"--hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size}")

        if hlo2tensorizer_extra:
            args.append(f"--internal-hlo2tensorizer-options={hlo2tensorizer_extra} --verify-hlo=true")

        return shlex.join(args)


def make_debug_dump_hook(save_path="debug_tensors.pt", tkg_save_path="debug_dump_tkg.pt"):
    """Returns a tensor_capture_hook that saves layer-0 intermediates on the
    first two forward calls:
      call 1 (CTE)        → save_path       (default: debug_tensors.pt)
      call 2 (first TKG)  → tkg_save_path   (default: debug_dump_tkg.pt)
    Subsequent calls are silently ignored.
    Tensor names: attn_out, kv_k, kv_v, final_hidden_states.
    """
    _DEBUG_TENSOR_NAMES = ["attn_out", "kv_k", "kv_v", "final_hidden_states"]
    _count = [0]

    def hook(model, captured_tensors):
        _count[0] += 1
        if _count[0] > 2:
            return
        path = save_path if _count[0] == 1 else tkg_save_path
        tag  = "CTE" if _count[0] == 1 else "TKG step 1"
        data = {
            name: t.detach().cpu()
            for name, t in zip(_DEBUG_TENSOR_NAMES, captured_tensors)
        }
        torch.save(data, path)
        print(f"Debug tensors ({tag}) saved to {path}:")
        for name, t in data.items():
            print(f"  {name}: shape={tuple(t.shape)} dtype={t.dtype}")

    return hook


# Aliases expected by the same `qwen.NeuronQwen3MoeForCausalLM` import convention.
NeuronQwen3MoeForCausalLM = NeuronQwen3MoeForCausalLMNxdAttnMoeKernel
NeuronQwen3MoeModel = NeuronQwen3MoeNxdAttnMoeKernelModel
