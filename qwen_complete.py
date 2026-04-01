# coding=utf-8
"""
Qwen3 MoE model with NKI-fused attention TKG (v10e) and NKI-fused MoE TKG (kernel_v19b).

CTE path (past_key_value is None):
    Standard flash attention + standard MoE (self.mlp unchanged).

TKG path (past_key_value is not None):
    Attention: qwen3_attn_tkg_fused_oproj_v10e — fused QKV proj + per-head RMSNorm
               + RoPE + flash decode + output proj. Mask generated on-chip from position_ids.
    MoE:       kernel_v19b.qwen3_moe_fused_tkg — fused RMSNorm + Router + TopK(8)
               + Expert MLPs in one NKI kernel.

Weight layouts (set up by convert_qwen3_moe_hf_to_neuron_state_dict):
  Attention (tile-transposed for v10e Plan A):
    Wq_nki.weight  [Hq_tp*d, H]  = [1024, 2048]  reshape→permute(0,3,2,1)
    Wk_nki.weight  [d, H]        = [128,  2048]  reshape→permute(0,3,2,1)
    Wv_nki.weight  [d, H]        = [128,  2048]  reshape→permute(0,3,2,1)
    Wo_nki.weight  [Hq_tp*d, H]  = [1024, 2048]  plain T, no tile-transpose
  MoE (native layout for kernel_v19b):
    gate_up_proj.weight  [E, H, 2*I=384]  gate cols 0:I, up cols I:2I
    down_proj.weight     [E, I=192, H]    no shard pre-split
"""

import gc
import logging
import math
import shlex
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Neuron")
logger.setLevel(logging.DEBUG)

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, GenerationConfig, Qwen3MoeForCausalLM
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from neuronx_distributed.parallel_layers import mappings, parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding, RowParallelLinear
from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed.modules.moe.moe_configs import BlockwiseMatmulConfig
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed_inference.models.config import (
    InferenceConfig,
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
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
    NeuronAttentionBaseOutput,
)
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)

torch.manual_seed(0)

from kernels.attn_tkg.agents.v10e import qwen3_attn_tkg_fused_oproj_v10e
from kernels.router_topk.qwen3_router_topk_plan_a import qwen3_router_topk_cte
from kernels.moe_fused_tkg import kernel_v19b as custom_moe_fused_kernel

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE

# Router kernel shape constants — must match the model config
_ROUTER_H = 2048   # hidden_size
_ROUTER_E = 128    # num_experts
_ROUTER_K = 8      # top_k


# ---------------------------------------------------------------------------
# Neuron config
# ---------------------------------------------------------------------------

class Qwen3MoEWithRouterNeuronConfig(MoENeuronConfig):
    """MoENeuronConfig with NKI-CTE blockwise defaults and normalize_top_k_affinities=True."""

    def __init__(self, **kwargs):
        if "blockwise_matmul_config" not in kwargs:
            kwargs["blockwise_matmul_config"] = BlockwiseMatmulConfig.from_kwargs(
                block_size=128,
                logical_nc_config=2,
                skip_dma_token=True,
                skip_dma_weight=True,
                normalize_top_k_affinities=True,
            )
        # Disable KV cache slicing so the kernel receives the full [B, 1, S_prior, d] cache.
        kwargs.setdefault("attn_tkg_nki_kernel_enabled", True)
        kwargs.setdefault("fused_qkv", False)
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_rmsnorm_cls():
    return Qwen3MoeRMSNorm if cpu_mode() else CustomRMSNorm


def get_modules_to_not_convert(neuron_config: MoENeuronConfig):
    return getattr(neuron_config, "modules_to_not_convert", None)


def _helper_concat_and_delete_qkv(qwen_state_dict: Dict[str, Any], layer_num: int, attr: str):
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
    for scale_layer in scale_layers:
        del neuron_state_dict[scale_layer]


def _build_interleaved_q_perm(num_attention_heads: int, head_dim: int, tp_degree: int) -> torch.Tensor:
    """Column permutation converting contiguous Q-head sharding to interleaved.

    Rank r gets global Q heads r, tp+r, 2*tp+r, ... (interleaved across tp ranks).
    """
    perm = []
    for r in range(tp_degree):
        for k in range(num_attention_heads // tp_degree):
            g = k * tp_degree + r  # global Q head index
            perm.extend(range(g * head_dim, g * head_dim + head_dim))
    return torch.tensor(perm, dtype=torch.long)


def convert_qwen3_moe_hf_to_neuron_state_dict(neuron_state_dict, config):
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"
    maybe_dequantize_layer(neuron_state_dict, config)
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    # Interleaved Q-head permutation, built once and reused for every layer.
    q_perm = _build_interleaved_q_perm(
        config.num_attention_heads, config.head_dim, config.neuron_config.tp_degree
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

        # Attention weights: tile-transposed for v10e Plan A.
        # reshape W → [heads, d, num_h_tiles, d], permute(0,3,2,1).
        _d        = config.head_dim                           # 128
        _nh       = config.num_attention_heads                # 32
        _nkv      = config.num_key_value_heads                # 4
        _H_cfg    = config.hidden_size                        # 2048
        _nh_tiles = _H_cfg // _d                              # 16

        q_proj_w = neuron_state_dict[f"layers.{l}.self_attn.q_proj.weight"]   # [Hq, H]
        W_tiled  = (q_proj_w
                    .reshape(_nh, _d, _nh_tiles, _d)
                    .permute(0, 3, 2, 1)
                    .reshape(_nh * _d, _H_cfg))
        neuron_state_dict[f"layers.{l}.self_attn.Wq_nki.weight"] = (
            W_tiled.contiguous()
        )

        o_proj_w = neuron_state_dict[f"layers.{l}.self_attn.o_proj.weight"]   # [H, Hq]
        neuron_state_dict[f"layers.{l}.self_attn.Wo_nki.weight"] = (
            o_proj_w.T.contiguous()
        )

        k_proj_w = neuron_state_dict[f"layers.{l}.self_attn.k_proj.weight"]   # [nkv*d, H]
        neuron_state_dict[f"layers.{l}.self_attn.Wk_nki.weight"] = (
            k_proj_w.reshape(_nkv, _d, _nh_tiles, _d).permute(0, 3, 2, 1)
            .reshape(_nkv * _d, _H_cfg).contiguous()
        )

        v_proj_w = neuron_state_dict[f"layers.{l}.self_attn.v_proj.weight"]   # [nkv*d, H]
        neuron_state_dict[f"layers.{l}.self_attn.Wv_nki.weight"] = (
            v_proj_w.reshape(_nkv, _d, _nh_tiles, _d).permute(0, 3, 2, 1)
            .reshape(_nkv * _d, _H_cfg).contiguous()
        )

        # Router weight: HF gate.weight [E, H] → linear_router.weight.
        neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
            neuron_state_dict[f"layers.{l}.mlp.gate.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.mlp.gate.weight"]

        # MoE weights: native layout for kernel_v19b (gate_up_proj, down_proj).
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
        down_proj = torch.empty(config.num_experts, intermediate_size, hidden_size, dtype=dtype, device=device)
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
    if config.neuron_config.fused_qkv:
        neuron_state_dict = convert_state_dict_to_fused_qkv(neuron_state_dict, config)
    return neuron_state_dict


def _format_blockwise_debug(blockwise_matmul_config: BlockwiseMatmulConfig):
    if blockwise_matmul_config is None:
        return "<none>"
    skip_dma = getattr(blockwise_matmul_config, "skip_dma", None)
    return (
        f"block_size={blockwise_matmul_config.block_size}, "
        f"logical_nc_config={blockwise_matmul_config.logical_nc_config}, "
        f"skip_dma_token={blockwise_matmul_config.skip_dma_token}, "
        f"skip_dma_weight={blockwise_matmul_config.skip_dma_weight}, "
        f"use_shard_on_block_dynamic_while={blockwise_matmul_config.use_shard_on_block_dynamic_while}, "
        f"use_shard_on_intermediate_dynamic_while={blockwise_matmul_config.use_shard_on_intermediate_dynamic_while}, "
        f"use_block_parallel={blockwise_matmul_config.use_block_parallel}, "
        f"skip_dma={skip_dma}, "
        f"obj_id={hex(id(blockwise_matmul_config))}"
    )


def _log_wrapper_blockwise(prefix: str, wrapper):
    cfg = getattr(wrapper, "neuron_config", None)
    if cfg is None:
        cfg = getattr(getattr(wrapper, "config", None), "neuron_config", None)
    if cfg is None:
        print(f"{prefix}: no neuron_config available")
        return
    bw_cfg = getattr(cfg, "blockwise_matmul_config", None)
    print(f"{prefix}: { _format_blockwise_debug(bw_cfg) }")
    model = getattr(wrapper, "model", None)
    if model is not None:
        model_cfg = getattr(model, "neuron_config", None)
        if model_cfg is None:
            model_cfg = getattr(getattr(model, "config", None), "neuron_config", None)
        if model_cfg is not None:
            model_bw = getattr(model_cfg, "blockwise_matmul_config", None)
            print(f"{prefix} model: { _format_blockwise_debug(model_bw) }")


# ---------------------------------------------------------------------------
# NKI router
# ---------------------------------------------------------------------------

class NKIRouterTopK(RouterTopK):
    """RouterTopK with forward replaced by the NKI router kernel.

    Uses self.weight_T [H, E] pre-transposed by RouterBase (store_transposed_weights=True).
    ea_out is L1-normalized top-K affinities in [T, E] (normalize_top_k_affinities=True).
    """

    def forward(self, hidden_states):
        # Flatten any leading dims to (T, H).
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        T, H = hidden_states.shape
        E, K = self.num_experts, self.top_k

        assert H == _ROUTER_H, f"hidden_size mismatch: kernel expects {_ROUTER_H}, got {H}"
        assert E == _ROUTER_E, f"num_experts mismatch: kernel expects {_ROUTER_E}, got {E}"
        assert K == _ROUTER_K, f"top_k mismatch: kernel expects {_ROUTER_K}, got {K}"

        # Kernel: x=[H, T], w=[H, E].
        x = hidden_states.T.contiguous()
        w = self.weight_T

        rl = torch.empty((T, E), dtype=torch.float32, device=hidden_states.device)
        ea = torch.empty((T, E), dtype=torch.float32, device=hidden_states.device)
        ei = torch.empty((T, K), dtype=torch.int32,   device=hidden_states.device)

        rl, ea, ei = qwen3_router_topk_cte[2](x.data, w.data, rl, ea, ei)

        router_logits     = rl.to(hidden_states.dtype)
        expert_affinities = ea.to(hidden_states.dtype)
        expert_index      = ei.to(torch.long)

        return router_logits, expert_affinities, expert_index


def _install_nki_router(mlp) -> None:
    """Swap mlp.router in-place with NKIRouterTopK, sharing all weights.

    The TKG fused kernel requires moe_fused_tkg.router.weight_T in float32
    (router_mm_dtype=float32). We cast it here and patch _apply to prevent
    model.to(bfloat16) from undoing this.

    invoke_preshard_hook stops at MoEFusedTKG (its preshard_hook is a no-op),
    so we patch moe_fused_tkg.preshard_hook directly to populate weight_T from
    the checkpoint's linear_router.weight.
    """
    old = mlp.router

    # Cast TKG router weight_T to float32.
    if hasattr(old, 'weight_T'):
        old.weight_T = nn.Parameter(old.weight_T.data.to(torch.float32))

    # Protect weight_T from being cast back by model.to(bfloat16).
    _orig_apply = old._apply
    def _apply_protect_weight_T(fn):
        _orig_apply(fn)
        if 'weight_T' in old._parameters and old._parameters['weight_T'].dtype != torch.float32:
            old._parameters['weight_T'] = nn.Parameter(
                old._parameters['weight_T'].data.to(torch.float32),
                requires_grad=False,
            )
        return old
    old._apply = _apply_protect_weight_T

    def _tkg_preshard_hook(model_state_dict, prefix):
        # prefix = "…mlp.moe_fused_tkg.weight"
        mlp_prefix    = prefix.removesuffix("moe_fused_tkg.weight")
        tkg_prefix    = mlp_prefix + "moe_fused_tkg."
        original_key  = mlp_prefix + "router.linear_router.weight"
        transposed_key = tkg_prefix + "router.weight_T"
        if original_key in model_state_dict:
            model_state_dict[transposed_key] = (
                model_state_dict[original_key]
                .detach().transpose(0, 1).clone().to(torch.float32)
            )
        elif transposed_key in model_state_dict:
            model_state_dict[transposed_key] = model_state_dict[transposed_key].to(torch.float32)
    mlp.moe_fused_tkg.preshard_hook = _tkg_preshard_hook

    nki_router = NKIRouterTopK(
        num_experts=old.num_experts,
        top_k=old.top_k,
        hidden_size=old.linear_router.in_features,
        act_fn=old.act_fn,
        sequence_parallel_enabled=old.sequence_parallel_enabled,
        sequence_dimension=old.sequence_dimension,
        dtype=old.dtype,
        device=old.device,
        bias=old.bias,
        tensor_model_parallel_group=old.tensor_parallel_group,
        jitter_eps=old.jitter_eps,
        store_transposed_weights=True,
        apply_act_fn_over_topk=old.apply_act_fn_over_topk,
    )
    nki_router.linear_router = old.linear_router
    nki_router.weight_T = nn.Parameter(old.linear_router.weight.detach().T.clone())
    mlp.router = nki_router


# ---------------------------------------------------------------------------
# NKI-fused TKG attention (v10e)
# ---------------------------------------------------------------------------

class NeuronQwen3MoEAttentionWithNKITKG(NeuronAttentionBase):
    """
    Qwen3 MoE attention using the v10e NKI fused kernel for TKG.

    CTE: delegates to NeuronAttentionBase.forward() (use_qk_norm=False).
    TKG: fused QKV proj + per-head RMSNorm + RoPE + flash decode + o_proj,
         followed by all-reduce. Mask generated on-chip from position_ids.
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
                "NeuronQwen3MoEAttentionWithNKITKG must be initialized in a distributed env."
            )

        # NKI weight holders: ColumnParallelLinear, already sharded to [out_tp, H] per rank.
        _dtype = config.neuron_config.torch_dtype
        _H = config.hidden_size
        _Hq_full = config.num_attention_heads * config.head_dim
        _Hkv_full = config.num_key_value_heads * config.head_dim
        self._nki_d = config.head_dim
        self.Wq_nki = ColumnParallelLinear(_H, _Hq_full, bias=False, gather_output=False, dtype=_dtype)
        self.Wo_nki = ColumnParallelLinear(_H, _Hq_full, bias=False, gather_output=False, dtype=_dtype)
        self.Wk_nki = ColumnParallelLinear(_H, _Hkv_full, bias=False, gather_output=False, dtype=_dtype)
        self.Wv_nki = ColumnParallelLinear(_H, _Hkv_full, bias=False, gather_output=False, dtype=_dtype)

        logger.debug(
            "NKI TKG attn init: H=%d  Hq_full=%d  Hkv_full=%d  tp_degree=%d",
            _H, _Hq_full, _Hkv_full, config.neuron_config.tp_degree,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        **kwargs,
    ):
        if past_key_value is not None:
            return self._nki_tkg_forward(hidden_states, position_ids, past_key_value)
        # CTE: use NeuronAttentionBase default flash attention
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

    def _nki_tkg_forward(
        self,
        hidden_states: torch.Tensor,       # [B, 1, H]
        position_ids: torch.LongTensor,    # [B, 1]
        past_key_value,                    # (K_cache [B, 1, S_prior, d], V_cache [B, 1, S_prior, d])
    ) -> NeuronAttentionBaseOutput:
        B = hidden_states.shape[0]
        d = self._nki_d

        cos_cache, sin_cache = self.rotary_emb(hidden_states, position_ids)
        cos_at_pos = cos_cache.squeeze(1)   # [B, d]
        sin_at_pos = sin_cache.squeeze(1)   # [B, d]

        Wq = self.Wq_nki.weight
        Wk = self.Wk_nki.weight
        Wv = self.Wv_nki.weight
        Wo = self.Wo_nki.weight

        K_cache, V_cache = past_key_value

        # Fused QKV + RMSNorm + RoPE + flash decode + o_proj (row-parallel).
        # Mask generated on-chip from position_ids threshold (v10e).
        output, k_rope_out, v_out = qwen3_attn_tkg_fused_oproj_v10e[2](
            hidden_states.data,
            Wq.data,
            Wk.data,
            Wv.data,
            Wo.data,
            self.q_layernorm.weight.data,
            self.k_layernorm.weight.data,
            K_cache.data,
            V_cache.data,
            cos_at_pos,
            sin_at_pos,
            position_ids.to(torch.int32),
        )

        output = reduce_from_tensor_model_parallel_region(output)

        k_new = k_rope_out.reshape(B, 1, 1, d)
        v_new = v_out.reshape(B, 1, 1, d)

        return NeuronAttentionBaseOutput(
            hidden_states=output,
            present_key_value=(k_new, v_new),
            cos_cache=cos_cache,
            sin_cache=sin_cache,
        )


# ---------------------------------------------------------------------------
# Decoder layer — v10e attention TKG + kernel_v19b MoE TKG
# ---------------------------------------------------------------------------

class NeuronQwen3MoeDecoderLayerComplete(nn.Module):
    """Decoder layer with v10e NKI attention TKG and kernel_v19b NKI MoE TKG.

    CTE: standard flash attention + standard MoE (self.mlp unchanged).
    TKG: v10e fused attention kernel + kernel_v19b fused MoE kernel.
    """

    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3MoEAttentionWithNKITKG(config=config)

        _dtype = config.neuron_config.torch_dtype
        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps).to(_dtype)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps).to(_dtype)

        self.mlp = initialize_moe_module(
            config=config, rmsnorm=self.post_attention_layernorm, init_tkg_module=True
        )
        _install_nki_router(self.mlp)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = False
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
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead."
            )

        residual = hidden_states

        # Attention block — v10e handles TKG internally, CTE delegates to base.
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

        # MoE block — kernel_v19b for TKG, standard mlp for CTE.
        residual = hidden_states
        is_tkg = past_key_value is not None
        if is_tkg:
            # kernel_v19b handles RMSNorm + Router + TopK(8) + Expert MLPs internally.
            # inp:      [B, 1, H]         — pre-norm hidden states
            # gamma:    [1, H]            — post-attention RMSNorm weight
            # router_w: [H, E] float32   — weight_T, set up by _install_nki_router
            # gate_up:  [E, H, 2*I=384]  — native layout (gate cols 0:I, up cols I:2I)
            # down_w:   [E, I=192, H]    — native layout, no shard pre-split
            tkg = self.mlp.moe_fused_tkg
            moe_out = custom_moe_fused_kernel.qwen3_moe_fused_tkg[2](
                hidden_states.data,
                self.post_attention_layernorm.weight.unsqueeze(0).data,        # [1, H]
                tkg.router.weight_T.data,                                      # [H, E] float32
                tkg.expert_mlps.mlp_op.gate_up_proj.weight.data,               # [E, H, 2*I=384]
                tkg.expert_mlps.mlp_op.down_proj.weight.data,                  # [E, I=192, H]
            )                                                                  # returns [T, H] bf16
            if isinstance(moe_out, (tuple, list)):
                moe_out = moe_out[0]
            # TP all-reduce: each rank produced a partial sum over its I shard.
            moe_out = mappings.reduce_from_tensor_model_parallel_region(
                moe_out, process_group=parallel_state.get_world_group()
            )
            hidden_states = moe_out.unsqueeze(1)   # restore seq dim to match residual [B, 1, H]
        else:
            hidden_states = self.mlp(hidden_states, padding_mask)[0]

        hidden_states = residual + hidden_states

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NeuronQwen3MoeModelComplete(NeuronBaseModel):
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
        self.layers = nn.ModuleList([
            NeuronQwen3MoeDecoderLayerComplete(config, layer_idx)
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

class NeuronQwen3MoeForCausalLMComplete(NeuronBaseForCausalLM):
    """Qwen3 MoE CausalLM with v10e fused attention TKG and kernel_v19b fused MoE TKG."""

    _model_cls = NeuronQwen3MoeModelComplete

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return Qwen3MoeForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_neuron_config_cls(cls):
        return Qwen3MoEWithRouterNeuronConfig

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
                "--enable-dmacopy-transpose",
            ]
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O3"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=2",
                "--vectorize-strided-dma",
                "--enable-scalar-dge-vectorization",
                "--enable-dmacopy-transpose",
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
                "--enable-dmacopy-transpose",
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
NeuronQwen3MoeForCausalLM = NeuronQwen3MoeForCausalLMComplete
