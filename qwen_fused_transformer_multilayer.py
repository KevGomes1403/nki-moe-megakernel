# coding=utf-8
"""
Qwen3 MoE model with a 48-layer fused TKG megakernel.

CTE: stock per-layer path (unchanged from V2 / baseline).
TKG: all decoder layers run inside `transformer_qwen3_moe_tkg_multilayer_jit[2]`
     — a single NKI kernel invocation that keeps the residual SBUF-resident
     across layer boundaries and writes each layer's K/V in place.

Integration strategy (to avoid overriding NxDI's `forward`):
  - The kernel dispatch is done from the **first decoder layer**. When layer 0
    is called with `past_key_value is not None` (TKG), it:
      1. gathers all per-layer weights from `self._parent_model.layers`,
      2. gathers all K/V cache tensors from `kv_mgr.past_key_values`
         (flat ParameterList: [K0, V0, K1, V1, ...]),
      3. calls the megakernel once (mutating all K/V caches in place),
      4. returns Y as its own layer output.
  - Layers 1..N-1 on TKG are pass-throughs: they return the hidden state
    unchanged. Their compile-time footprint is a no-op.
  - Kernel does NOT apply final RMSNorm — NxDI's `self.norm` is applied
    post-loop inside `get_model_output`, so we inherit that behavior cleanly.

Flag / open items (also in kernel docstring):
  - `attn_block_tkg_nki_kernel_cache_update=True` is set on the neuron config
    so NxDI skips its Python KV scatter and trailing `kv_mgr.update_cache`.
  - Pass-through layers on TKG still participate in the traced graph. The
    compiler should constant-fold them into no-ops; verify in HLO before
    declaring the integration free.
  - Router weight pre-transpose is deferred — we `.T.contiguous()` at
    forward-time like V2 (traced once per layer, becomes a compile-time
    constant in the NEFF). Converter stays identical to V2.
  - Only TKG B=1 is handled by the megakernel. Speculative decoding / B>1
    must fall back to the per-layer path (not wired yet).
"""

import gc
import shlex
import warnings
import weakref
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
    TensorCaptureConfig,
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

from kernels.transformer.transformer_qwen_multilayer import (
    transformer_qwen3_moe_tkg_multilayer_jit,
    get_multilayer_kernel_jit,
)
from kernels.transformer.transformer_qwen_multilayer_debug import (
    get_multilayer_debug_kernel_jit,
)

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
# Neuron config
# ---------------------------------------------------------------------------

class Qwen3MoEV3MultilayerNeuronConfig(MoENeuronConfig):
    """MoENeuronConfig for the 48-layer fused TKG megakernel."""

    def __init__(self, **kwargs):
        if not isinstance(kwargs.get("blockwise_matmul_config"), BlockwiseMatmulConfig):
            kwargs["blockwise_matmul_config"] = BlockwiseMatmulConfig.from_kwargs(
                use_torch_block_wise=True,
                # block_size=256,
                # logical_nc_config=2,
                # normalize_top_k_affinities=True,
                # use_shard_on_block_dynamic_while=True,
                # block_sharding_strategy="PING_PONG",
            )
        # Full-KV cache layout expected by v14a (no slicing).
        kwargs["attn_tkg_nki_kernel_enabled"] = True
        # Megakernel scatters K/V in place → NxDI must skip its Python scatter
        # and the trailing kv_mgr.update_cache (gated on this flag).
        kwargs["attn_block_tkg_nki_kernel_cache_update"] = True
        kwargs.setdefault("fused_qkv", False)
        # Pop tensor_capture_config before passing to OnDeviceSamplingConfig — it
        # doesn't accept that kwarg and will raise TypeError if it sees it.
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
# State dict converter — identical to V2 (router pre-transpose deferred to
# forward-time per the plan's §2.3 "only change" optimization; leave for now).
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

        o_proj_w = neuron_state_dict[f"layers.{l}.self_attn.o_proj.weight"]   # [H, Hq]
        neuron_state_dict[f"layers.{l}.self_attn.Wo_nki.weight"] = o_proj_w.T.contiguous()

        # Tile-transposed layout for pretransposed attention kernel:
        #   W_pt[head*d+p, tile*d+f] = W[head*d+f, tile*d+p]
        #   Produced by: W.reshape(n_heads, d, n_tiles, d).permute(0, 3, 2, 1).reshape(n_heads*d, H)
        _d        = config.head_dim                       # 128
        _nh       = config.num_attention_heads            # 32
        _nkv      = config.num_key_value_heads            # 4
        _H_cfg    = config.hidden_size                    # 2048
        _nh_tiles = _H_cfg // _d                          # 16

        q_proj_w = neuron_state_dict[f"layers.{l}.self_attn.q_proj.weight"]   # [Hq, H]
        neuron_state_dict[f"layers.{l}.self_attn.Wq_nki.weight"] = (
            q_proj_w.reshape(_nh, _d, _nh_tiles, _d)
            .permute(0, 3, 2, 1).reshape(_nh * _d, _H_cfg).contiguous()
        )

        k_proj_w = neuron_state_dict[f"layers.{l}.self_attn.k_proj.weight"]   # [nkv*d, H]
        neuron_state_dict[f"layers.{l}.self_attn.Wk_nki.weight"] = (
            k_proj_w.reshape(_nkv, _d, _nh_tiles, _d)
            .permute(0, 3, 2, 1).reshape(_nkv * _d, _H_cfg).contiguous()
        )

        v_proj_w = neuron_state_dict[f"layers.{l}.self_attn.v_proj.weight"]   # [nkv*d, H]
        neuron_state_dict[f"layers.{l}.self_attn.Wv_nki.weight"] = (
            v_proj_w.reshape(_nkv, _d, _nh_tiles, _d)
            .permute(0, 3, 2, 1).reshape(_nkv * _d, _H_cfg).contiguous()
        )

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
# Attention (V3 — identical in shape/weights to V2; TKG dispatch done at model
# level, not in this class).
# ---------------------------------------------------------------------------

class NeuronQwen3MoEAttentionV3(NeuronAttentionBase):
    def __init__(self, config: Qwen3MoeInferenceConfig):
        rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        # Temporarily mask attn_block_tkg_nki_kernel_enabled during super().__init__()
        # so init_gqa_properties constructs the CTE o_proj with out_proj_kernel_enabled=False,
        # matching qwen.py's plain o_proj matmul. Without this, the deprecated-alias
        # auto-promotion (config.py:459-467) forces out_proj_kernel_enabled=True on the CTE
        # path (attention_base.py:376), which transposes the weight and routes CTE through
        # _kernel_o_proj (a different numerics path than the reference).
        # V3's TKG layer 0 dispatches the megakernel directly and layers 1..L-1 return early,
        # so NxDI's self_attn is never called on TKG; the instance attr
        # self.attn_block_tkg_nki_kernel_enabled being False is harmless.
        # The config-level flag stays True so model_base.py's update_kv_per_layer gate
        # (which requires attn_block_tkg_nki_kernel_cache_update + _enabled) still fires.
        _saved_block_flag = config.neuron_config.attn_block_tkg_nki_kernel_enabled
        config.neuron_config.attn_block_tkg_nki_kernel_enabled = False
        try:
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
        finally:
            config.neuron_config.attn_block_tkg_nki_kernel_enabled = _saved_block_flag
        self.q_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)
        self.k_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronQwen3MoEAttentionV3 must be initialized in a distributed env."
            )

        _dtype = config.neuron_config.torch_dtype
        _H = config.hidden_size
        _Hq_full = config.num_attention_heads * config.head_dim
        _Hkv_full = config.num_key_value_heads * config.head_dim
        self._nki_d = config.head_dim
        self.Wq_nki = ColumnParallelLinear(_H, _Hq_full,  bias=False, gather_output=False, dtype=_dtype)
        self.Wk_nki = ColumnParallelLinear(_H, _Hkv_full, bias=False, gather_output=False, dtype=_dtype)
        self.Wv_nki = ColumnParallelLinear(_H, _Hkv_full, bias=False, gather_output=False, dtype=_dtype)
        self.Wo_nki = ColumnParallelLinear(_H, _Hq_full,  bias=False, gather_output=False, dtype=_dtype)


# ---------------------------------------------------------------------------
# Decoder layer — dispatches the megakernel from idx=0 on TKG.
# ---------------------------------------------------------------------------

class NeuronQwen3MoeDecoderLayerV3(nn.Module):
    """
    CTE path: stock per-layer forward (input_layernorm → self_attn → residual
              → post_attention_layernorm → mlp → residual).
    TKG path: layer 0 runs the megakernel over **all** layers and returns Y;
              layers 1..N-1 are pass-throughs.
    """

    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3MoEAttentionV3(config=config)

        _dtype = config.neuron_config.torch_dtype
        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = initialize_moe_module(config=config)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = False
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens
        self.debug_tensor_capture = getattr(config.neuron_config, "tensor_capture_config", None) is not None

        # Replica groups for nccl.all_reduce inside the megakernel.
        # Derived from tp_degree to avoid torch.distributed calls during mock tracing.
        self._replica_groups = (list(range(config.neuron_config.tp_degree)),)
        self._num_hidden_layers = config.num_hidden_layers

        # Set by parent model.init_model so layer 0 can reach sibling layers' weights.
        # Stored as weakref so nn.Module doesn't register the parent as a child
        # (which would create a reference cycle and break train()/.eval() recursion).
        self._parent_model_ref = None

    @property
    def _parent_model(self):
        return self._parent_model_ref() if self._parent_model_ref is not None else None

    # ---- helpers for layer 0's megakernel dispatch ----

    def _gather_weights_from_parent(self):
        """Stack per-layer weights along a new leading layer dim.

        Each returned tensor is [L, ...]; the kernel slices via Wq_all[i]
        (NKI integer indexing on HBM reduces the layer dim at compile time).
        NKI's frontend rejects list/tuple-of-tensor args at the top level —
        stacking sidesteps that by passing a single tensor per weight type.
        """
        # All per-layer weights are passed UN-stacked. Stacking + integer slicing
        # produces an HBM view whose outer-dim stride is the full per-layer size
        # (e.g. 1024*2048=2097152 for Wq), and the sub-kernels' access-pattern
        # derivation can't represent that — it trips NCC_IBIR158/243 OOB.
        # Pass each layer's weight as its own top-level NKI tensor (same pattern
        # as KV caches).
        layers = self._parent_model.layers
        Wq, Wk, Wv, Wo = [], [], [], []
        qn, kn = [], []
        gpre = []
        gpost = []
        router, gate_up, down = [], [], []
        for l in layers:
            Wq.append(l.self_attn.Wq_nki.weight)
            Wk.append(l.self_attn.Wk_nki.weight)
            Wv.append(l.self_attn.Wv_nki.weight)
            Wo.append(l.self_attn.Wo_nki.weight)
            qn.append(l.self_attn.q_layernorm.weight)
            kn.append(l.self_attn.k_layernorm.weight)
            gpre.append(l.input_layernorm.weight)
            gpost.append(l.post_attention_layernorm.weight.unsqueeze(0))    # [1, H]
            router.append(l.mlp.router.linear_router.weight.T.contiguous())
            gate_up.append(l.mlp.expert_mlps.mlp_op.gate_up_proj.weight)
            down.append(l.mlp.expert_mlps.mlp_op.down_proj.weight)
        return (Wq, Wk, Wv, Wo, qn, kn, gpre, gpost, router, gate_up, down)

    def _gather_kv_caches(self, kv_mgr):
        """kv_mgr.past_key_values is flat [K0, V0, K1, V1, ...].

        Returned as flat lists; caller spreads them into individual positional
        args so each KV cache registers as its own top-level NKI tensor input
        (required for in-place scatter DMA aliasing).
        """
        L = self._num_hidden_layers
        Ks = [kv_mgr.past_key_values[2 * i]     for i in range(L)]
        Vs = [kv_mgr.past_key_values[2 * i + 1] for i in range(L)]
        return Ks, Vs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        kv_mgr=None,
        idx: int = 0,
        is_for_context_encoding: bool = True,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead."
            )

        is_tkg = not is_for_context_encoding

        # ------------------------------------------------------------------
        # TKG: layer 0 runs the megakernel, later layers pass through.
        # ------------------------------------------------------------------
        if is_tkg:
            L = self._num_hidden_layers
            if self.layer_idx == 0:
                assert self._parent_model is not None, \
                    "V3 layer 0 needs _parent_model set by NeuronQwen3MoeModelV3.init_model"
                assert kv_mgr is not None, \
                    "V3 TKG path requires kv_mgr (update_kv_per_layer gate must be on)"

                hidden_states = ModuleMarkerStartWrapper()(hidden_states)

                # RoPE cos/sin at position — same shape trick as V2.
                cos_cache, sin_cache = self.self_attn.rotary_emb(hidden_states, position_ids)
                cos_at_pos = cos_cache.squeeze(1)   # [B, d]
                sin_at_pos = sin_cache.squeeze(1)

                (Wq_list, Wk_list, Wv_list, Wo_list,
                 qn_list, kn_list, gpre_list, gpost_list,
                 router_list, gate_up_list, down_list) = self._gather_weights_from_parent()
                K_caches, V_caches = self._gather_kv_caches(kv_mgr)

                _kernel_fn = (
                    get_multilayer_debug_kernel_jit(L)[2]
                    if self.debug_tensor_capture
                    else get_multilayer_kernel_jit(L)[2]
                )
                kernel_out = _kernel_fn(
                    hidden_states,
                    *Wq_list, *Wk_list, *Wv_list, *Wo_list,
                    *qn_list, *kn_list, *gpre_list, *gpost_list,
                    *router_list, *gate_up_list, *down_list,
                    *K_caches, *V_caches,
                    cos_at_pos, sin_at_pos, position_ids.to(torch.int32),
                    replica_groups=self._replica_groups,
                )
                Y = kernel_out[0]
                if self.debug_tensor_capture:
                    # Debug kernel returns (Y, attn_out_l0, final_hidden_states_l0,
                    #                       K0..K_{L-1}, V0..V_{L-1}).
                    attn_out_l0            = kernel_out[1]
                    final_hidden_states_l0 = kernel_out[2]
                    K_out = list(kernel_out[3     : 3 + L])
                    V_out = list(kernel_out[3 + L : 3 + 2 * L])
                    _register_tensor("attn_out",            attn_out_l0)
                    _register_tensor("kv_k",                K_out[0])
                    _register_tensor("kv_v",                V_out[0])
                    _register_tensor("final_hidden_states", final_hidden_states_l0)
                else:
                    # Production kernel returns (Y, K0..K_{L-1}, V0..V_{L-1}).
                    K_out = list(kernel_out[1     : 1 + L])
                    V_out = list(kernel_out[1 + L : 1 + 2 * L])
                Y = ModuleMarkerEndWrapper()(Y)

                # Stash the post-scatter KV handles so pass-through layers can
                # forward them into next_decoder_cache (NxDI aliases those to
                # kv_mgr.past_key_values[i]).
                self._parent_model._tkg_kv_out = (K_out, V_out)

                return (Y, (K_out[0], V_out[0]), cos_cache, sin_cache, None)

            # Layers 1..L-1 on TKG: pure pass-through. Forward the kernel's
            # returned KV handles (set by layer 0) so next_decoder_cache aliases
            # the mutated HBM buffers, not the pre-kernel inputs.
            K_out, V_out = self._parent_model._tkg_kv_out
            return (hidden_states,
                    (K_out[self.layer_idx], V_out[self.layer_idx]),
                    None, None, None)

        # ------------------------------------------------------------------
        # CTE: stock path (unchanged).
        # ------------------------------------------------------------------
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        residual = hidden_states

        qkv_fused_rmsnorm = None
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
        if self.debug_tensor_capture and self.layer_idx == 0:
            _register_tensor("attn_out", hidden_states)
            _register_tensor("kv_k", present_key_value[0])
            _register_tensor("kv_v", present_key_value[1])
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, padding_mask)[0]
        hidden_states = residual + hidden_states
        if self.debug_tensor_capture and self.layer_idx == 0:
            _register_tensor("final_hidden_states", hidden_states)

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Pre-transposed LM head
# ---------------------------------------------------------------------------

class PreTransposedColumnParallelLinear(ColumnParallelLinear):
    """Stores lm_head weight as [hidden, vocab_per_partition] instead of
    [vocab_per_partition, hidden], so HBM loads require no transpose_mode=ENABLED."""

    def set_weight_and_bias_config(self):
        self.weight_shape = (self.input_size, self.output_size_per_partition)
        self.weight_partition_dim = 1
        self.bias_shape = None

    def preshard_hook(self, model_state_dict, prefix):
        if prefix.endswith("weight"):
            w = model_state_dict[prefix]  # HF shape: [vocab, hidden]
            model_state_dict[prefix] = w.t().contiguous()  # → [hidden, vocab]
        # partition_dim=1 → NxDI shards along vocab axis automatically

    def forward(self, input, slice_indices=None):
        input_parallel = self._cpl_maybe_input_copy_to_tp_region(input)
        # weight is [hidden, vocab_per_partition] — matmul directly, no .t()
        weight = self.weight[:, slice_indices] if slice_indices is not None else self.weight
        output_parallel = torch.matmul(input_parallel, weight)
        return self._cpl_maybe_gather_output(output_parallel)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NeuronQwen3MoeModelV3(NeuronBaseModel):
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
            NeuronQwen3MoeDecoderLayerV3(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        # Layer 0 reaches sibling layers for its megakernel dispatch.
        for layer in self.layers:
            layer._parent_model_ref = weakref.ref(self)

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

class NeuronQwen3MoeForCausalLMV3(NeuronBaseForCausalLM):
    """Qwen3 MoE CausalLM with 48-layer fused TKG megakernel."""

    _model_cls = NeuronQwen3MoeModelV3

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return Qwen3MoeForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_neuron_config_cls(cls):
        return Qwen3MoEV3MultilayerNeuronConfig

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
            # model_wrapper adds this only on the default (compiler_args=None) path;
            # force modular flow here too so NKI mac-ops don't defeat graph partitioner.
            hlo2tensorizer_extra = "--modular-flow-mac-threshold=10"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            # TKG graph is massively larger with L=48 unrolled; use -O1 while
            # iterating, bump to -O3 when stable. Plan §1.6 warns 20-60 min at -O3.
            optimization_level = "-O3"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                # Plan §1.6: try =2 / =4 to overlap layer i+1 attn AR with layer i MoE.
                "--cc-pipeline-tiling-factor=4",
                "--vectorize-strided-dma",
                "--eager-tkg-vectorize-dma",
                "--enable-dge-on-indirect-dma",
                "--enable-dge-on-vector-indirect-dma",
            ]
            hlo2tensorizer_extra = ""
            # NCC-6661: NKI in-place KV scatter triggers the backend verifier; disable it.
            # model_wrapper adds this automatically only when compiler_args=None.
            args.append("--internal-backend-options=--enable-verifier=false")
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
        # Larger unrolled TKG graph may trip the default instruction limit.
        args.append("--internal-max-instruction-limit=30000000")

        if self.neuron_config.scratchpad_page_size:
            args.append(f"--hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size}")

        # model_wrapper always appends --internal-hlo2tensorizer-options after get_compiler_args();
        # do NOT add it here or it will appear twice. Pass extra hlo2tensorizer flags via the
        # config's layer_boundary_markers / modular-flow path instead (see model_wrapper:85-162).
        # The one exception is hlo2tensorizer_extra which model_wrapper won't add on the custom path.
        if hlo2tensorizer_extra:
            args.append(f"--internal-hlo2tensorizer-options={hlo2tensorizer_extra} --verify-hlo=true")

        return shlex.join(args)


def make_debug_dump_hook(save_path="debug_tensors.pt", tkg_save_path="debug_dump_tkg.pt"):
    """Returns a tensor_capture_hook that saves layer-0 intermediates on the first two
    forward calls:
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


# Alias expected by main.py's `qwen.NeuronQwen3MoeForCausalLM` convention.
NeuronQwen3MoeForCausalLM = NeuronQwen3MoeForCausalLMV3
