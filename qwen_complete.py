# coding=utf-8
"""
Qwen3 MoE model with NKI-fused attention TKG (v17_fast_exp) and NKI-fused MoE TKG (v15c).

CTE path (past_key_value is None):
    Standard flash attention + standard MoE (self.mlp unchanged; bf16 gate_up/down retained
    alongside the TKG-only int8 buffers).

TKG path (past_key_value is not None):
    Attention: v17_fast_exp (Plan B [X1], 53.28 us) — fused QKV proj + per-head RMSNorm
               + RoPE + flash decode + output proj, with nisa.exponential fast-exp.
               Wrapped in qwen3_attn_tkg_v17_wrapper to match v10e's [B, 1, H] output shape.
    MoE:       v15c (Plan C [F3], 72.88 us) — fused RMSNorm + Router + TopK(8) + Expert MLPs
               with FP8-packed gate_up weights (merged weight+scale DMA, HWDGE offload).

Weight layouts (set up by convert_qwen3_moe_hf_to_neuron_state_dict):
  Attention (tile-transposed for v17/v10e Plan A):
    Wq_nki.weight  [Hq_tp*d, H]  = [1024, 2048]  reshape->permute(0,3,2,1)
    Wk_nki.weight  [d, H]        = [128,  2048]  reshape->permute(0,3,2,1)
    Wv_nki.weight  [d, H]        = [128,  2048]  reshape->permute(0,3,2,1)
    Wo_nki.weight  [Hq_tp*d, H]  = [1024, 2048]  plain T, no tile-transpose
  MoE (v15c native layout):
    gate_up_proj.weight    [E, H, 2*I=384]            bf16  (kept for CTE)
    down_proj.weight       [E, I=192, H]              bf16  (kept for CTE)
    gate_up_packed_w_tkg   [E, H_free+1=17, 128, 384] int8  (v15c packed FP8 + scales, TKG-only)
    down_w_tkg             [E, I=192, H]              int8  (FP8 bytes, TKG-only)
    down_scales_tkg        [E, H]                     fp32  (FP8 scales, TKG-only)
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

from kernels.router_topk.qwen3_router_topk_plan_a import qwen3_router_topk_cte
# Round 2 [Z.1]: MoE TKG upgrade v19b -> v15c (72.88 us, bit-exact vs v14a, Plan C [F3] merged DMA).
from kernels.moe_fused_tkg.quantized import v15c as custom_moe_fused_kernel
# Round 2 [Z.1]: Attention TKG upgrade v10e -> v17_fast_exp (53.28 us, nisa.exponential fast-exp).
# v17_fast_exp exports a sub-function; we wrap it in a @nki.jit that produces
# v10e-compatible [B, 1, H_wo] output so qwen_complete's call site is unchanged.
from kernels.moe_fused_tkg.quantized._qwen_integration import (
    qwen3_attn_tkg_v17_wrapper as qwen3_attn_tkg_fused_oproj_v10e,
    quantize_and_pack_gate_up,
    quantize_down,
)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE

# Router kernel shape constants — must match the model config
_ROUTER_H = 2048   # hidden_size
_ROUTER_E = 128    # num_experts
_ROUTER_K = 8      # top_k


# ---------------------------------------------------------------------------
# [L6] NxDI streaming shard_checkpoint monkey-patch (Round 3 [M1b] workaround)
# ---------------------------------------------------------------------------
#
# LANDMINE L6 (integration_findings.md):
#   neuronx_distributed/trace/model_builder.py:827-858 `shard_checkpoint`
#   accumulates ALL TP-rank sharded checkpoint dicts in a single Python list
#   before returning. For Qwen3-30B-A3B with TP=4, each sharded dict is ~15 GB
#   bf16 → the loop holds 4 × 15 GB = 60 GB on top of the 60 GB original
#   checkpoint, producing a ~120 GB peak that SIGKILLs the process on a
#   124 GB-DRAM trn3.3xlarge even with 32 GB swap (observed 04:34 UTC 2026-04-22:
#   mem=127GB + swap=32GB saturated, child_rss=124GB).
#
#   NxDI exposes NO config knob (`save_sharded_checkpoint`, `skip_sharding`)
#   that disables the per-rank accumulation — both flags only toggle WHERE
#   sharding happens (compile vs load), not HOW the loop buffers ranks.
#
# FIX: replace `ModelBuilder.shard_checkpoint` with a streaming version that
#   flushes each rank to disk via `save_file(...)` and drops the in-RAM dict
#   before the next rank starts. The upstream caller at
#   application_base.py:254 ignores the return value, so we return [].
#
# GATED by NKI_STREAMING_SHARD_PATCH env var (default "1"). Set to "0" to
#   disable and fall back to stock NxDI behavior.
#
# Requires `save_sharded_checkpoint=True` on the NeuronConfig so that:
#   (a) compile-time `shard_weights` calls `shard_checkpoint(serialize_path=...)`
#       (application_base.py:252-254), which triggers our streaming write.
#   (b) load-time `load_weights` reads per-rank from disk
#       (application_base.py:389-399) — avoids a second accumulating call.
import os as _os

if _os.environ.get("NKI_STREAMING_SHARD_PATCH", "1") == "1":
    import gc as _gc
    import time as _time
    import torch as _torch
    from neuronx_distributed.trace.model_builder import ModelBuilder as _NxDModelBuilder
    from neuronx_distributed.trace.trace import (
        _mock_parallel_state as _mock_ps,  # noqa: F401  (keep import parity with upstream)
        preprocess_checkpoint as _preprocess_ckpt,
    )
    from neuronx_distributed.utils.model_utils import init_on_device as _init_on_device
    from neuronx_distributed.trace.mock_torchdist import mock_distributed as _mock_distributed
    from neuronx_distributed.parallel_layers import parallel_state as _parallel_state
    from safetensors.torch import save_file as _save_file

    _ORIG_SHARD_CHECKPOINT = _NxDModelBuilder.shard_checkpoint

    def _streaming_shard_checkpoint(self, serialize_path=None):
        """Streaming replacement for ModelBuilder.shard_checkpoint (L6 workaround).

        Mirrors upstream (NxDI 2.9, model_builder.py:817-858) but flushes each
        rank's sharded checkpoint to disk via save_file() and drops the in-RAM
        copy BEFORE the next rank starts. Caller ignores the return value
        (application_base.py:254), so we return []."""
        if serialize_path is None:
            # If caller didn't request disk write, we can't stream — fall back
            # to upstream behavior to preserve semantics for other code paths.
            return _ORIG_SHARD_CHECKPOINT(self, serialize_path=serialize_path)
        if not _os.path.exists(serialize_path):
            _os.makedirs(serialize_path)

        source_model_key = list(self.model_collection.keys())[0]
        model_container = self.model_collection[source_model_key]
        logger.info(
            f"[L6 streaming shard] Sharding weights for ranks: "
            f"{self.start_rank_id}...{self.start_rank_id + self.local_ranks_size - 1}"
        )
        t0 = _time.monotonic()
        with _mock_distributed(world_size=self.world_size), _init_on_device(
            _torch.device("meta"), force_custom_init_on_device=True
        ):
            _torch.distributed.init_process_group(
                backend="xla", rank=0, world_size=self.world_size
            )
            _parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=self.tp_degree,
                pipeline_model_parallel_size=self.pp_degree,
                expert_model_parallel_size=self.ep_degree,
                skip_collective_init=True,
                lnc_size=self.logical_nc_config,
            )
            if self.init_custom_process_group_fn:
                self.init_custom_process_group_fn()

            model_container.model_instance.load_module()
            func_kwargs = (
                {}
                if model_container.bucket_config is None
                else model_container.bucket_config.get_func_kwargs_for_bucket_rank(0)
            )
            if "bucket_rank" in func_kwargs:
                func_kwargs.pop("bucket_rank")
            model, _io_aliases = model_container.model_instance.get(0, **func_kwargs)
            checkpoint = self.checkpoint_loader()
            _preprocess_ckpt(model, checkpoint)
            self.cast_weights(checkpoint, model, "")

            for rank in range(self.start_rank_id, self.start_rank_id + self.local_ranks_size):
                # shard_weights_with_cache calls checkpoint.copy() (shallow),
                # then get_sharded_checkpoint mutates the copy in place (new
                # tensors for parallel layers), then save_file writes to disk.
                sharded = self.shard_weights_with_cache(
                    rank, model, checkpoint, serialize_path
                )
                # CRITICAL: drop the sharded dict NOW, not after the loop.
                del sharded
                _gc.collect()
                logger.info(
                    f"[L6 streaming shard] rank {rank} flushed to disk, "
                    f"elapsed={_time.monotonic() - t0:.1f}s"
                )

            _parallel_state.destroy_model_parallel()
            _torch.distributed.destroy_process_group()
        logger.info(
            f"[L6 streaming shard] Done sharding (streaming) in "
            f"{_time.monotonic() - t0:.1f}s"
        )
        return []  # caller (application_base.py:254) ignores return value

    _NxDModelBuilder.shard_checkpoint = _streaming_shard_checkpoint
    logger.info("[L6] NxDI streaming shard_checkpoint patch ACTIVE")
else:
    logger.info("[L6] NxDI streaming shard_checkpoint patch DISABLED via env")


# ---------------------------------------------------------------------------
# Neuron config
# ---------------------------------------------------------------------------

class Qwen3MoEWithRouterNeuronConfig(MoENeuronConfig):
    """MoENeuronConfig with NKI-CTE blockwise defaults and normalize_top_k_affinities=True."""

    def __init__(self, **kwargs):
        if "blockwise_matmul_config" not in kwargs:
            # Round 3 [M1a] fix: route CTE through forward_blockwise with
            # use_shard_on_block_dynamic_while=True so the dispatcher picks
            # _call_bwmm_shard_on_block_kernel (beta2 nkilib `bwmm_shard_on_block`,
            # installed — verified via `probe` on this SDK and STEP A smoke test
            # on qwen_fused_transformer, which compiled all CTE HLO buckets
            # cleanly without the protobuf 2 GB error).
            #
            # Replaces Phase 0's block_size=8192 workaround, which was motivated
            # by the apparent "missing private blockwise_mm kernel" narrative but
            # actually created the real problem: forward_all_experts has a Python
            # `for e in range(num_experts)` loop that unrolls into 128 expert
            # sub-ops × 48 layers = 6144 HLO sub-ops → protobuf serialization fail.
            #
            # Config pattern mirrored from `qwen_fused_transformer.py:78-86` —
            # the block_size/dynamic_while/block_sharding_strategy triple is the
            # (b1) path from `docs/m1_investigation.md`. skip_dma_token/weight
            # and normalize_top_k_affinities preserved from the prior config.
            kwargs["blockwise_matmul_config"] = BlockwiseMatmulConfig.from_kwargs(
                block_size=256,
                logical_nc_config=2,
                skip_dma_token=True,
                skip_dma_weight=True,
                normalize_top_k_affinities=True,
                use_shard_on_block_dynamic_while=True,
                block_sharding_strategy="PING_PONG",
            )
        # Disable KV cache slicing so the kernel receives the full [B, 1, S_prior, d] cache.
        kwargs.setdefault("attn_tkg_nki_kernel_enabled", True)
        kwargs.setdefault("fused_qkv", False)
        # Round 3 [M1b]: enable on-compile disk write + per-rank load from disk
        # so L6 streaming shard patch (above) can flush each rank and avoid the
        # ~120 GB accumulated-shard peak that SIGKILLs on 124 GB trn3.3xlarge.
        kwargs.setdefault("save_sharded_checkpoint", True)
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

        # Round 2 [Z.2]: Offline FP8 quantization + v15c packing for the TKG kernel.
        # The bf16 gate_up_proj / down_proj params stay in the state_dict for the CTE
        # path (which uses the standard MoE module). The packed int8 buffers below are
        # loaded into TKG-only keys via the preshard hook (see _install_nki_router).
        #
        # gate_up_packed_w layout (see v15c.pack_gate_up):
        #   [E, H_free+1=17, PMAX=128, GU=384] int8
        #     planes 0..15  = fp8_e4m3fn bytes of gate_up weight
        #     plane 16      = fp32 scales (first 12 bytes per row = 3 scales)
        gate_up_packed_tkg = quantize_and_pack_gate_up(gate_up_proj.detach())
        neuron_state_dict[
            f"layers.{l}.mlp.moe_fused_tkg.gate_up_packed_w_tkg"
        ] = gate_up_packed_tkg

        # TODO(Round 2 Plan B [F3b]): down_w is currently int8 bytes + fp32 scales
        # (v14a / v15c layout). If [F3b] lands a new down_w layout (e.g. packed or
        # sharded), swap the call below to the new packer and update the TKG call
        # site in NeuronQwen3MoeDecoderLayerComplete.forward accordingly.
        down_w_tkg_i8, down_scales_tkg_f32 = quantize_down(down_proj.detach())
        neuron_state_dict[
            f"layers.{l}.mlp.moe_fused_tkg.down_w_tkg"
        ] = down_w_tkg_i8
        neuron_state_dict[
            f"layers.{l}.mlp.moe_fused_tkg.down_scales_tkg"
        ] = down_scales_tkg_f32
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


def _register_tkg_quant_buffers(mlp, config) -> None:
    """Register TKG-only quantized MoE weights on ``mlp.moe_fused_tkg``.

    v15c expects:
      - gate_up_packed_w_tkg: [E, H_free+1=17, PMAX=128, GU=384] int8
      - down_w_tkg:           [E, I, H]                          int8
      - down_scales_tkg:      [E, H]                              fp32

    These are loaded from the state dict at keys populated by
    ``convert_qwen3_moe_hf_to_neuron_state_dict`` (see quantize_and_pack_gate_up
    / quantize_down). The existing bf16 gate_up_proj / down_proj parameters on
    ``expert_mlps.mlp_op`` remain untouched for the CTE path.

    IMPLEMENTATION NOTE (Round 3 [M1a'] fix — 2026-04-21):
    These tensors MUST be registered as ``nn.Parameter`` (with requires_grad=
    False for inference), not ``register_buffer``. NxDI's
    ``neuronx_distributed/trace/model_builder.py:806`` iterates
    ``model.named_parameters()`` to collect trace inputs — buffers are NOT
    included. Buffers get captured as Python closures in the forward, and
    XLA materializes them as ``constant`` HLO ops (upcast int8 → fp32,
    4× bloat). For Qwen3-30B that pushes TKG HLO from ~200 MB to > 30 GB,
    blowing past protobuf's 2 GB SerializeToString limit. The fix is simply
    Parameter registration — same as ``weight_T`` in ``_install_nki_router``.
    See ``docs/integration_findings.md`` landmine L5 for the full writeup.
    """
    E = config.num_experts
    H = config.hidden_size
    # config.moe_intermediate_size is post-pad (see Qwen3MoeInferenceConfig.maybe_pad_intermediate);
    # config.moe_intermediate_pad_size is the pad delta. I_padded equals moe_intermediate_size.
    I_padded = getattr(config, "moe_intermediate_size", None)
    if I_padded is None:
        I_padded = config.intermediate_size

    GU_FLAT = 2 * I_padded                       # 2*I (gate||up along last dim)
    PMAX = 128
    H_FREE = H // PMAX
    PLANES = H_FREE + 1                          # 17

    # Shapes match the output of _qwen_integration.quantize_and_pack_gate_up /
    # quantize_down given the [E, H, 2*I] / [E, I, H] inputs after any pad.
    tkg = mlp.moe_fused_tkg
    # Register as nn.Parameter(requires_grad=False) so NxDI trace captures
    # these as model inputs (HLO parameter ops), not inlined constants. The
    # zero-init here is a placeholder; real values come from state_dict().
    tkg.gate_up_packed_w_tkg = nn.Parameter(
        torch.zeros(E, PLANES, PMAX, GU_FLAT, dtype=torch.int8),
        requires_grad=False,
    )
    tkg.down_w_tkg = nn.Parameter(
        torch.zeros(E, I_padded, H, dtype=torch.int8),
        requires_grad=False,
    )
    tkg.down_scales_tkg = nn.Parameter(
        torch.zeros(E, H, dtype=torch.float32),
        requires_grad=False,
    )

    # Protect int8/fp32 Parameter dtypes from model.to(bfloat16). Mirrors the
    # pattern used for router.weight_T in _install_nki_router — operates on
    # tkg._parameters (was tkg._buffers when these were register_buffer).
    #
    # H1 (Round 3 fix, 2026-04-22): use in-place `p.data = p.data.to(dtype)`
    # instead of creating a fresh nn.Parameter(...) on dtype mismatch. The
    # in-place update preserves the Parameter identity in tkg._parameters[...]
    # so NxDI's state_dict loader doesn't lose its reference during the cast
    # ping-pong between `model.to(bfloat16)` (coerces fp32 → bf16) and this
    # hook (restores back to fp32/int8). Recreation was also compounding
    # memory pressure during the sharded-checkpoint write phase by holding
    # stale Parameter objects until Python GC.
    _orig_apply = tkg._apply
    def _apply_protect_quant(fn):
        _orig_apply(fn)
        for name, want_dtype in [
            ("gate_up_packed_w_tkg", torch.int8),
            ("down_w_tkg", torch.int8),
            ("down_scales_tkg", torch.float32),
        ]:
            p = tkg._parameters.get(name)
            if p is not None and p.dtype != want_dtype:
                p.data = p.data.to(want_dtype)
        return tkg
    tkg._apply = _apply_protect_quant


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
        _register_tkg_quant_buffers(self.mlp, config)

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

        # MoE block — v15c for TKG, standard mlp for CTE.
        residual = hidden_states
        is_tkg = past_key_value is not None
        if is_tkg:
            # v15c handles RMSNorm + Router + TopK(8) + Expert MLPs internally.
            # inp:              [B, 1, H]                            — pre-norm hidden states
            # gamma:            [1, H]                                — post-attention RMSNorm weight
            # router_w:         [H, E] float32                        — weight_T, set up by _install_nki_router
            # gate_up_packed_w: [E, H_free+1=17, 128, 384] int8      — FP8 weights + fp32 scales packed offline
            # down_w:           [E, I=192, H] int8                    — FP8 weight bytes
            # down_scales:      [E, H] fp32                           — FP8 per-H scales
            # TODO(Round 2 Plan B [F3b]): when the new down_w layout lands, update
            # the down_w_tkg / down_scales_tkg buffers to match and adjust the call here.
            tkg = self.mlp.moe_fused_tkg
            moe_out = custom_moe_fused_kernel.qwen3_moe_fused_tkg[2](
                hidden_states.data,
                self.post_attention_layernorm.weight.unsqueeze(0).data,        # [1, H]
                tkg.router.weight_T.data,                                      # [H, E] float32
                tkg.gate_up_packed_w_tkg.data,                                 # [E, 17, 128, 384] int8 (v15c packed)
                tkg.down_w_tkg.data,                                           # [E, I=192, H] int8 (fp8 bytes)
                tkg.down_scales_tkg.data,                                      # [E, H] fp32 (fp8 scales)
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
