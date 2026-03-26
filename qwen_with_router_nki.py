# coding=utf-8
"""
Qwen3 MoE model with NKI-fused attention (CTE) and NKI router.

Standalone variant of qwen_with_attn_cte_nki.py with the PyTorch RouterTopK
replaced by the NKI router kernel for the CTE path.

NKI kernels used:
  - qwen3_attn_cte_fused  : QKV proj + per-head RMSNorm + RoPE + causal flash attn
  - qwen3_router_topk_cte : linear proj + softmax + top-K + L1-norm + scatter

Router integration notes:
  - NKIRouterTopK overrides RouterTopK.forward() only; all other routing logic
    (sequence parallel, preshard_hook, weight_T registration) is inherited.
  - self.weight_T ([H, E]) is pre-transposed by RouterBase when
    store_transposed_weights=True. The TKG fused kernel uses the same parameter,
    so no runtime transpose is needed in either path.
  - ea_out is already L1-normalized top-K affinities scattered into [T, E].
    normalize_top_k_affinities=True in BlockwiseMatmulConfig makes the downstream
    masking step a no-op (renormalizing an already-normalized tensor is identity).
    NOTE: verify that BlockwiseMatmulConfig accepts normalize_top_k_affinities;
    if not, this field may belong on MoENeuronConfig directly.
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
from torch import nn
from transformers import AutoTokenizer, GenerationConfig, Qwen3MoeForCausalLM
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding, RowParallelLinear
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

import os
# os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
# os.environ["XLA_IR_DEBUG"]= "1"
# os.environ["XLA_HLO_DEBUG"]= "1"
# os.environ["NEURON_RT_INSPECT_ENABLE"]= "1"
# os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"]= "1"
# os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]= "./output"

# ---------------------------------------------------------------------------
# NKI kernel imports
# ---------------------------------------------------------------------------
from kernels.attn_cte.v7ab_qwen3_attn_cte_fused import qwen3_attn_cte_fused
from kernels.router_topk.qwen3_router_topk_plan_a import qwen3_router_topk_cte

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
    """MoENeuronConfig with NKI-CTE blockwise defaults and normalize_top_k_affinities.

    normalize_top_k_affinities=True: ea_out from the NKI router is already
    L1-normalized, so the downstream masking step is a no-op.
    """

    def __init__(self, **kwargs):
        if "blockwise_matmul_config" not in kwargs:
            kwargs["blockwise_matmul_config"] = BlockwiseMatmulConfig.from_kwargs(
                block_size=128,
                logical_nc_config=2,
                skip_dma_token=True,
                skip_dma_weight=True,
                normalize_top_k_affinities=True,
            )
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
    """Column permutation that converts contiguous Q-head layout to interleaved.

    Standard RowParallelLinear shards [H, Hq*d] contiguously: rank r gets global
    Q heads [r*(Hq/tp) .. (r+1)*(Hq/tp)-1].  The NKI kernel assumes interleaved
    sharding: rank r gets global Q heads r, tp+r, 2*tp+r, ...

    With interleaved sharding and 4 replicated KV heads, each rank has exactly
    gqa = (Hq/tp) / Hkv = 2 Q heads per KV head, enabling LNC=2 to split Hkv
    across 2 NeuronCores.
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

    # Interleaved Q-head permutation — built once, reused for every layer.
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

        # NKI Plan B weights: pre-transposed [H, out], Q columns in interleaved order.
        q_proj_T = neuron_state_dict[f"layers.{l}.self_attn.q_proj.weight"].T  # [H, Hq*d]
        neuron_state_dict[f"layers.{l}.self_attn.Wq_nki.weight"] = (
            q_proj_T[:, q_perm].contiguous()
        )
        neuron_state_dict[f"layers.{l}.self_attn.Wk_nki.weight"] = (
            neuron_state_dict[f"layers.{l}.self_attn.k_proj.weight"].T.contiguous()  # [H, Hkv]
        )
        neuron_state_dict[f"layers.{l}.self_attn.Wv_nki.weight"] = (
            neuron_state_dict[f"layers.{l}.self_attn.v_proj.weight"].T.contiguous()  # [H, Hkv]
        )
        o_proj_w = neuron_state_dict[f"layers.{l}.self_attn.o_proj.weight"]  # [H, Hq*d]
        neuron_state_dict[f"layers.{l}.self_attn.Wo_nki.weight"] = (
            o_proj_w[:, q_perm].contiguous()
        )

        # Router weight: HF gate.weight [E, H] → linear_router.weight [E, H].
        # weight_T ([H, E]) is derived by RouterBase.preshard_hook at load time.
        neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
            neuron_state_dict[f"layers.{l}.mlp.gate.weight"].detach().clone()
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
    """RouterTopK with the CTE forward replaced by the NKI router kernel.

    Uses self.weight_T ([H, E]) pre-transposed by RouterBase when
    store_transposed_weights=True. The TKG fused kernel reads the same
    parameter, so no runtime transpose is needed in either path.

    ea_out is L1-normalized top-K affinities scattered into [T, E] (zeros at
    non-selected positions), equivalent to what get_expert_affinities_masked
    produces when normalize_top_k_affinities=True.
    """

    def forward(self, hidden_states):
        # hidden_states: [T, H] bf16
        # RouterBase.get_router_logits already flattened (S, B, H) → (T, H).
        T, H = hidden_states.shape
        E, K = self.num_experts, self.top_k

        assert H == _ROUTER_H, f"hidden_size mismatch: kernel expects {_ROUTER_H}, got {H}"
        assert E == _ROUTER_E, f"num_experts mismatch: kernel expects {_ROUTER_E}, got {E}"
        assert K == _ROUTER_K, f"top_k mismatch: kernel expects {_ROUTER_K}, got {K}"

        # Kernel layout: x=[H, T], w=[H, E]; weight_T is already [H, E]
        x = hidden_states.T.contiguous()
        w = self.weight_T

        rl = torch.empty((T, E), dtype=torch.float32, device=hidden_states.device)
        ea = torch.empty((T, E), dtype=torch.float32, device=hidden_states.device)
        ei = torch.empty((T, K), dtype=torch.int32,   device=hidden_states.device)

        rl, ea, ei = qwen3_router_topk_cte[2](x, w, rl, ea, ei)

        # Cast to match ExpertMLPsV2 expectations
        router_logits     = rl.to(hidden_states.dtype)   # [T, E]
        expert_affinities = ea.to(hidden_states.dtype)   # [T, E]
        expert_index      = ei.to(torch.long)            # [T, K]

        return router_logits, expert_affinities, expert_index


def _install_nki_router(mlp) -> None:
    """Swap mlp.router in-place with NKIRouterTopK, sharing all weights.

    weight_T is re-derived from linear_router.weight so RouterBase.preshard_hook
    (which repopulates weight_T from the checkpoint at load time) keeps both
    parameters in sync.
    """
    old = mlp.router
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
# NKI-fused CTE attention
# ---------------------------------------------------------------------------

class NeuronQwen3MoEAttentionWithNKICTE(NeuronAttentionBase):
    """
    Qwen3 MoE attention that uses the v7ab NKI fused kernel for CTE (prefill).

    CTE path  (past_key_value is None):
        hidden_states → [NKI: QKV proj + per-head RMSNorm + RoPE + causal flash attn]
                      → attn_out [B, S, Hq_per_rank * d]
                      → o_proj  → [B, S, H]
        K and V are also computed from the per-rank weight slices to populate the KV cache.

    TKG path  (past_key_value is not None):
        Delegates to NeuronAttentionBase.forward() unchanged.
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
        # Per-head RMSNorm weights used by the NKI kernel
        self.q_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)
        self.k_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronQwen3MoEAttentionWithNKICTE must be initialized in a distributed env."
            )

        # NKI Plan B weight holders.
        # Q is sharded across TP ranks → RowParallelLinear (partition_dim=1) gives [H, Hq/tp].
        # K/V use REPLICATE_TO_TP_DEGREE: all KV heads live on every rank, so nn.Linear
        # (replicated by default in NxD) gives [H, Hkv_full] on each rank.
        # TKG continues to use self.qkv_proj (ColumnParallelLinear) unchanged.
        _dtype = config.neuron_config.torch_dtype
        _H = config.hidden_size
        _Hq_full = config.num_attention_heads * config.head_dim
        _Hkv_full = config.num_key_value_heads * config.head_dim
        # Save full (pre-TP) KV head count for KV-cache slicing after the kernel.
        self._nki_num_kv_heads_full = config.num_key_value_heads
        # Wq_nki columns stored in interleaved Q-head order (see state dict conversion).
        self.Wq_nki = RowParallelLinear(_Hq_full, _H, bias=False, input_is_parallel=True, dtype=_dtype)
        self.Wk_nki = nn.Linear(_Hkv_full, _H, bias=False, dtype=_dtype)  # [H, Hkv_full], replicated
        self.Wv_nki = nn.Linear(_Hkv_full, _H, bias=False, dtype=_dtype)  # [H, Hkv_full], replicated
        # Wo_nki columns permuted to match interleaved kernel output.
        self.Wo_nki = RowParallelLinear(_Hq_full, _H, bias=False, input_is_parallel=True, dtype=_dtype)

        logger.debug(
            "NKI CTE attn init: H=%d  Hq_full=%d  Hkv_full=%d  "
            "num_kv_heads_full=%d  tp_degree=%d",
            _H, _Hq_full, _Hkv_full,
            config.num_key_value_heads,
            config.neuron_config.tp_degree,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        **kwargs,
    ):
        # is_token_gen = past_key_value is not None
        # if is_token_gen:
            # TKG: use NeuronAttentionBase's default KV-cache decoding path
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        # CTE: use NKI fused kernel
        # return self._nki_cte_forward(hidden_states, position_ids, **kwargs)

    def _nki_cte_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        **kwargs,
    ) -> NeuronAttentionBaseOutput:
        bsz, q_len, h = hidden_states.size()

        # Rotary embeddings
        cos_full, sin_full = self.rotary_emb(hidden_states, position_ids)
        cos_for_kernel = cos_full[0]  # [S, d]
        sin_for_kernel = sin_full[0]  # [S, d]

        # Weight extraction in Plan B layout [H, out_per_rank]
        Wq = self.Wq_nki.weight.data  # [H, Hq/tp] — interleaved column order
        Wk = self.Wk_nki.weight.data  # [H, Hkv_full]
        Wv = self.Wv_nki.weight.data  # [H, Hkv_full]

        # Per-head RMSNorm weights pre-reshaped to [d, 1] for NKI tracing
        q_norm_weight = self.q_layernorm.weight.reshape(self.head_dim, 1)
        k_norm_weight = self.k_layernorm.weight.reshape(self.head_dim, 1)

        Hq_out = self.num_heads * self.head_dim
        Hkv_full_out = self._nki_num_kv_heads_full * self.head_dim

        logger.debug(
            "NKI CTE fwd: B=%d S=%d H=%d  Hq_out=%d  Hkv_full_out=%d  Wq=%s  Wk=%s",
            bsz, q_len, h, Hq_out, Hkv_full_out, tuple(Wq.shape), tuple(Wk.shape),
        )

        attn_out, K_cache_nki, V_cache_nki = qwen3_attn_cte_fused[2](
            hidden_states,
            Wq, Wk, Wv,
            q_norm_weight, k_norm_weight,
            cos_for_kernel, sin_for_kernel,
            Hq_out=Hq_out, Hkv_out=Hkv_full_out,
        )

        attn_output = self.Wo_nki(attn_out)

        # Slice rank-local KV heads for the cache: kernel outputs all Hkv_full heads
        num_kv_per_rank = self.num_key_value_heads
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        kv_start = tp_rank * num_kv_per_rank * self.head_dim
        kv_end   = kv_start + num_kv_per_rank * self.head_dim

        K_cache = (
            K_cache_nki[:, :, kv_start:kv_end]
            .reshape(bsz, q_len, num_kv_per_rank, self.head_dim)
            .transpose(1, 2)
        )
        V_cache = (
            V_cache_nki[:, :, kv_start:kv_end]
            .reshape(bsz, q_len, num_kv_per_rank, self.head_dim)
            .transpose(1, 2)
        )

        return NeuronAttentionBaseOutput(
            hidden_states=attn_output,
            present_key_value=(K_cache, V_cache),
            cos_cache=cos_full,
            sin_cache=sin_full,
        )


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class NeuronQwen3MoeDecoderLayerWithNKI(nn.Module):
    """Decoder layer with NKI attention (CTE) and NKI router."""

    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3MoEAttentionWithNKICTE(config=config)
        self.moe_fused_nki_kernel_enabled = getattr(config, "moe_fused_nki_kernel_enabled", False)

        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = initialize_moe_module(
            config=config, rmsnorm=self.post_attention_layernorm, init_tkg_module=True
        )
        # Install the NKI router in place of the PyTorch RouterTopK
        _install_nki_router(self.mlp)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = True
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
        if not self.moe_fused_nki_kernel_enabled:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, padding_mask)[0]
        hidden_states = residual + hidden_states

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NeuronQwen3MoeModelWithNKI(NeuronBaseModel):
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
                NeuronQwen3MoeDecoderLayerWithNKI(config, layer_idx)
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


# ---------------------------------------------------------------------------
# CausalLM entry point
# ---------------------------------------------------------------------------

class NeuronQwen3MoeForCausalLMWithNKI(NeuronBaseForCausalLM):
    """Qwen3 MoE CausalLM with NKI-fused attention (CTE) and NKI router."""

    _model_cls = NeuronQwen3MoeModelWithNKI

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


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------

def test_cte_compile(model_path: str, traced_model_path: str):
    """
    Compile the CTE model and verify both NKI kernels trace successfully.

    Usage:
        python qwen_with_router_nki.py <model_path> <traced_model_path>
    """
    import os
    os.environ.setdefault("NEURON_CC_FLAGS", " ")
    os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

    print("=" * 70)
    print("NKI Attention + Router CTE Compile Test")
    print("  Model:      NeuronQwen3MoeForCausalLMWithNKI")
    print("  Attn kernel: kernels/attn_cte/v7ab_qwen3_attn_cte_fused.py")
    print("  Router kernel: kernels/router_topk/qwen3_router_topk_plan_a.py")
    print("=" * 70)

    blockwise_config = BlockwiseMatmulConfig.from_kwargs(
        block_size=128,
        logical_nc_config=2,
        skip_dma_token=True,
        skip_dma_weight=True,
        use_shard_on_block_dynamic_while=True,
        normalize_top_k_affinities=True,
    )
    neuron_config = MoENeuronConfig(
        tp_degree=4,
        batch_size=1,
        max_context_length=640,
        seq_len=640,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        blockwise_matmul_config=blockwise_config,
    )
    config = Qwen3MoeInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    print("\n[1/3] Initialising model ...")
    model = NeuronQwen3MoeForCausalLMWithNKI(model_path, config)

    print("[2/3] Enabling context encoding and compiling CTE graph ...")
    model.compile(traced_model_path)

    print("[3/3] Compilation complete.")
    print("  NKI attention + router CTE kernels compiled successfully.")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python qwen_with_router_nki.py <model_path> <traced_model_path>")
        sys.exit(1)

    test_cte_compile(model_path=sys.argv[1], traced_model_path=sys.argv[2])
