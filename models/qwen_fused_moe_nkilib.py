# coding=utf-8
""" Qwen3 MOE model for NXD inference. This is a re-implementation of the NxDI source code for Qwen3 MOE, provided here for easy kernel development."""
import torch

from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed.modules.moe.moe_configs import BlockwiseMatmulConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeInferenceConfig


_QWEN_FORCED_CONFIG_FLAGS = {
    "moe_fused_nki_kernel_enabled": True,
    "qkv_nki_kernel_enabled": True,
    "qkv_kernel_enabled": True,
    "output_logits": True,
    "async_mode": True,
    "fused_qkv": True,
    "attn_block_tkg_nki_kernel_enabled": True,
    "attn_block_tkg_nki_kernel_cascaded_attention": True,
    "attn_block_tkg_nki_kernel_cache_update": True,
}

# nkilib attention TKG kernel asserts s_prior % p_max == 0 after LNC=2 sharding,
# i.e. bucket_size % 256 == 0. Default bucket [128, 256, 640] for max_length=640
# fails on 640. Override with 256-aligned buckets covering the full 640 range.
_QWEN_FORCED_TKG_BUCKETS = [128, 256, 512, 768]


_QWEN_FORCED_SAMPLING_FLAGS = {
    "do_sample": False,
    "top_k": 1,
    "top_p": 1.0,
    "temperature": 1.0,
    "dynamic": False,
    "top_k_kernel_enabled": False,
}


def _get_sampling_kwarg(kwargs, name, default):
    value = kwargs.get(name, default)
    return default if value is None else value


def _force_qwen_greedy_on_device_sampling_config(config):
    for flag, value in _QWEN_FORCED_SAMPLING_FLAGS.items():
        setattr(config, flag, value)
    return config


def _make_qwen_forced_on_device_sampling_config(kwargs):
    existing_config = kwargs.get("on_device_sampling_config")
    if existing_config is not None:
        return _force_qwen_greedy_on_device_sampling_config(existing_config)

    sampling_kwargs = {
        flag: _get_sampling_kwarg(kwargs, flag, value)
        for flag, value in _QWEN_FORCED_SAMPLING_FLAGS.items()
    }
    global_topk = kwargs.get("global_topk")
    if global_topk is not None:
        sampling_kwargs["global_topk"] = global_topk
    sampling_dp_degree = kwargs.get("sampling_dp_degree")
    if sampling_dp_degree is not None:
        sampling_kwargs["sampling_dp_degree"] = sampling_dp_degree
    return _force_qwen_greedy_on_device_sampling_config(
        OnDeviceSamplingConfig(**sampling_kwargs)
    )


def _make_qwen_forced_blockwise_matmul_config():
    return BlockwiseMatmulConfig.from_kwargs(
        use_torch_block_wise=False,
        block_size=256,
        logical_nc_config=2,
        use_shard_on_block_dynamic_while=True,
        block_sharding_strategy="PING_PONG",
    )


def _inject_qwen_forced_config_flags():
    if getattr(MoENeuronConfig, "_qwen_forced_config_patch", False):
        return

    original_init = MoENeuronConfig.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.update(_QWEN_FORCED_CONFIG_FLAGS)
        kwargs["on_device_sampling_config"] = _make_qwen_forced_on_device_sampling_config(kwargs)
        kwargs["blockwise_matmul_config"] = _make_qwen_forced_blockwise_matmul_config()
        kwargs["disable_normalize_top_k_affinities"] = False
        kwargs["token_generation_buckets"] = list(_QWEN_FORCED_TKG_BUCKETS)
        # NxDI asserts buckets[-1] <= max_length AND sizes the KV cache S-dim
        # from max_length (kv_cache_manager.py:194-235). The TKG NKI kernel
        # asserts curr_sprior <= full_sprior, so max_length must cover the
        # largest bucket. Affects both submitted model and baseline (same patch).
        kwargs["max_length"] = max(kwargs.get("max_length") or 0, max(_QWEN_FORCED_TKG_BUCKETS))

        original_init(self, *args, **kwargs)

        for flag, value in _QWEN_FORCED_CONFIG_FLAGS.items():
            setattr(self, flag, value)
        if self.on_device_sampling_config is None:
            self.on_device_sampling_config = OnDeviceSamplingConfig(
                do_sample=False,
                top_k=1,
                top_p=1.0,
                temperature=1.0,
                dynamic=False,
                top_k_kernel_enabled=False,
            )
        else:
            self.on_device_sampling_config = _force_qwen_greedy_on_device_sampling_config(
                self.on_device_sampling_config
            )
        self.on_device_sampling = True

    MoENeuronConfig.__init__ = patched_init
    MoENeuronConfig._qwen_forced_config_patch = True
    MoENeuronConfig._qwen_forced_original_init = original_init


_inject_qwen_forced_config_flags()

import os
# os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
# os.environ["XLA_IR_DEBUG"]= "1"
# os.environ["XLA_HLO_DEBUG"]= "1"
# os.environ["NEURON_RT_INSPECT_ENABLE"]= "1"
# os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"]= "1"
# os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]= "./output"
# os.environ["BASE_COMPILE_WORK_DIR"] = "./baseline_compiler_dir/"
torch.manual_seed(0)

import gc
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
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

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig, MOE_TKG_MK_INTERMEDIATE_PER_TP
from neuronx_distributed_inference.models.model_wrapper import CONTEXT_ENCODING_MODEL_TAG, TOKEN_GENERATION_MODEL_TAG
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase, QKNormPlacement, EPDispatchOption
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
# from kernels.moe_fused_tkg.moe_fused_nki_nkilib import moe_fused_tkg

# MoE NKI
import nki.isa as nisa
import nki.language as nl
import nki.collectives as nccl
from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg as _moe_tkg
from nkilib.core.router_topk.router_topk import (
    XSBLayout_tp2013__1,
    router_topk as _router_topk,
)
from nkilib.core.subkernels.rmsnorm_tkg import rmsnorm_tkg as _rmsnorm_tkg
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.tensor_view import TensorView
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode, QuantizationType, RouterActFnType
from nkilib.core.utils.interleave_copy import interleave_copy
from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg as _attention_block_tkg_call

def _qwen3_moe_sbuf_in_sbuf_out_custom_rms_nkilib(
    rmsnorm_out,          # [PMAX, T, H_FREE] dtype in SBUF (already RMSNorm'd, nkilib layout)
    dtype,
    T,
    router_w,             # [H, E] HBM
    gate_up_w,            # [E, H, 2*I] HBM
    down_w,               # [E, I, H] HBM
    sbm=None,
):
    """
    nkilib router_topk and moe_tkg kernels used by NxDI's moe_fused_nki_kernel_enabled
    Qwen TKG path. Caller produces rmsnorm_out via nkilib rmsnorm_tkg.
    """
    prg_id = nl.program_id(axis=0)

    router_in = rmsnorm_out
    if router_w.dtype != rmsnorm_out.dtype:
        router_in = sbm.alloc_stack(
            (_PMAX, T, _H_FREE), router_w.dtype, buffer=nl.sbuf, name="router_in"
        )
        nisa.tensor_copy(dst=router_in[0:_PMAX, 0:T, 0:_H_FREE], src=rmsnorm_out[0:_PMAX, 0:T, 0:_H_FREE])

    expert_index = nl.ndarray((T, _K), dtype=nl.uint32, buffer=nl.sbuf, name="expert_index")
    expert_affinities = nl.ndarray((T, _E), dtype=nl.float32, buffer=nl.sbuf, name="expert_affinities")
    # nkilib currently requires router_logits to receive a store even when the
    # caller does not need logits. Keep that required scratch write in SBUF.
    router_logits_scratch = nl.ndarray((T, _E), dtype=dtype, buffer=nl.sbuf, name="router_logits_scratch")

    router_outputs = _router_topk(
        x=router_in,
        w=router_w,
        w_bias=None,
        router_logits=router_logits_scratch,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        act_fn=RouterActFnType.SOFTMAX,
        k=_K,
        x_hbm_layout=0,
        x_sb_layout=XSBLayout_tp2013__1,
        router_pre_norm=True,
        norm_topk_prob=True,
        use_column_tiling=True,
        use_indirect_dma_scatter=False,
        return_eager_affi=False,
        use_PE_broadcast_w_bias=False,
        shard_on_tokens=T > 1,
        skip_store_expert_index=False,
        skip_store_router_logits=False,
    )
    expert_index = router_outputs[1]
    expert_affinities = router_outputs[2]

    if T > 1:
        expert_mlp_in = rmsnorm_out
    else:
        expert_mlp_in = nl.ndarray(
            (_PMAX, T, _H_FREE_SHARD), dtype=dtype, buffer=nl.sbuf, name="expert_mlp_in"
        )
        nisa.tensor_copy(
            dst=expert_mlp_in[0:_PMAX, 0:T, 0:_H_FREE_SHARD],
            src=rmsnorm_out[0:_PMAX, 0:T, nl.ds(prg_id * _H_FREE_SHARD, _H_FREE_SHARD)],
        )

    return _moe_tkg(
        hidden_input=expert_mlp_in,
        expert_gate_up_weights=gate_up_w.reshape((_E, _H, 2, _I)),
        expert_down_weights=down_w,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        is_all_expert=False,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        activation_fn=ActFnType.SiLU,
        output_dtype=dtype,
        output_in_sbuf=True,
    )


def _store_h_sharded_sbuf_output(out_sb, output, dtype, T, sbm):
    prg_id = nl.program_id(axis=0)
    out_tile_sb = sbm.alloc_stack((T, _H_SHARD), dtype, buffer=nl.sbuf, name="out_tile_sb")

    for t in nl.static_range(T):
        for h1 in nl.static_range(_H_FREE_SHARD):
            tp_psum = nl.ndarray((1, _PMAX), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=tp_psum[0:1, 0:_PMAX],
                data=out_sb[0:_PMAX, t, h1 : h1 + 1],
            )
            interleave_copy(
                dst=out_tile_sb.ap(
                    pattern=[[_H_SHARD, 1], [_H_FREE_SHARD, _PMAX]],
                    offset=t * _H_SHARD + h1,
                ),
                src=tp_psum[0:1, 0:_PMAX],
                index=h1,
            )
        if prg_id == 0:
            nisa.dma_copy(dst=output[t : t + 1, 0:_H_SHARD], src=out_tile_sb[t : t + 1, 0:_H_SHARD])
        else:
            nisa.dma_copy(dst=output[t : t + 1, _H_SHARD:_H], src=out_tile_sb[t : t + 1, 0:_H_SHARD])


def _store_t_sharded_sbuf_output(out_sb, output, dtype, T, sbm):
    prg_id = nl.program_id(axis=0)
    t_first_shard = T // _N_PRGS
    t_second_shard = T - t_first_shard
    t_local = t_first_shard if prg_id == 0 else t_second_shard
    t_offset = 0 if prg_id == 0 else t_first_shard

    out_tile_sb = sbm.alloc_stack((t_local, _H), dtype, buffer=nl.sbuf, name="out_tile_sb")
    for local_t in nl.static_range(t_local):
        global_t = t_offset + local_t
        for h1 in nl.static_range(_H_FREE):
            tp_psum = nl.ndarray((1, _PMAX), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=tp_psum[0:1, 0:_PMAX],
                data=out_sb[0:_PMAX, global_t, h1 : h1 + 1],
            )
            interleave_copy(
                dst=out_tile_sb.ap(
                    pattern=[[_H, 1], [_H_FREE, _PMAX]],
                    offset=local_t * _H + h1,
                ),
                src=tp_psum[0:1, 0:_PMAX],
                index=h1,
            )
        nisa.dma_copy(dst=output[global_t : global_t + 1, 0:_H], src=out_tile_sb[local_t : local_t + 1, 0:_H])


def moe_fused_tkg(inp, gamma, router_w, gate_up_w, down_w, replica_groups=None):
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)

    T = inp.shape[0]
    prg_id = nl.program_id(axis=0)
    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    sbm.open_scope(name="moe_fused_tkg")

    rmsnorm_out = sbm.alloc_stack(
        (_PMAX, T, _H_FREE), inp.dtype, buffer=nl.sbuf, name="rmsnorm_out"
    )
    _rmsnorm_tkg(
        input=inp.reshape((1, T, _H)),
        gamma=gamma,
        output=rmsnorm_out,
        eps=1e-6,
        hidden_dim_tp=False,
        single_core_forced=(T > 1),
        sbm=sbm,
    )

    # MoE produces a TP-partial SBUF tensor (sum over this rank's experts only).
    # output_in_sbuf=True returns shape == hidden_input.shape; for T=1 our
    # expert_mlp_in is [PMAX, 1, H_FREE_SHARD], so moe_out_sb matches that.
    # Layout is channel-interleaved: moe_out_sb[p, t, h1] holds the value at
    # HBM-equivalent H position prg_id*H_SHARD + p*H_FREE_SHARD + h1.
    moe_out_sb = _qwen3_moe_sbuf_in_sbuf_out_custom_rms_nkilib(
        rmsnorm_out,
        inp.dtype,
        T,
        router_w,
        gate_up_w,
        down_w,
        sbm=sbm,
    )

    # Load each LNC core's owned H_SHARD slice of the residual (= pre-RMSNorm
    # `inp`) into SBUF in the same CHANNEL-INTERLEAVED layout the MoE output
    # uses (rmsnorm_tkg docstring lines 79-86 for the convention pseudocode).
    #
    # Layout: dst[p, t, h1] = HBM[t, prg_id*H_SHARD + p*H_FREE_SHARD + h1]
    #   → partition p indexes the HIGH bits within the shard (stride H_FREE_SHARD)
    #   → h1 indexes the LOW bits (stride 1)
    # _store_h_sharded_sbuf_output consumes this same convention.
    free_size = T * _H_FREE_SHARD
    inp_flat = inp.reshape((T * _H,))

    residual_sb = sbm.alloc_stack(
        (_PMAX, free_size), inp.dtype, buffer=nl.sbuf, name="residual_sb"
    )
    # Pattern levels (innermost → outermost source iteration):
    #   level 0: stride H_FREE_SHARD, count PMAX → partition dim p
    #   level 1: stride 1,            count H_FREE_SHARD → h1 (free, inner)
    #   level 2: stride H,            count T → t (free, outer)
    h_load_pattern = [[_H_FREE_SHARD, _PMAX], [1, _H_FREE_SHARD], [_H, T]]
    h_load_offset = prg_id * _H_SHARD
    nisa.dma_copy(
        dst=residual_sb,
        src=inp_flat.ap(pattern=h_load_pattern, offset=h_load_offset),
        dge_mode=nisa.dge_mode.hwdge,
    )

    # AR across TP in SBUF: TP-partial → TP-summed. Same accumulation order as
    # `reduce_from_tensor_model_parallel_region` (bf16 sum across TP). Reshape
    # the [PMAX, T, H_FREE_SHARD] MoE output to 2D for the SBUF collective
    # (nccl.all_reduce only accepts 2D SBUF tensors).
    ar_sb = nl.zeros(
        (_PMAX, free_size), dtype=inp.dtype, buffer=nl.sbuf, name="ar_sb"
    )
    nccl.all_reduce(
        dsts=[ar_sb],
        srcs=[moe_out_sb.reshape((_PMAX, free_size))],
        op=nl.add,
        replica_group=rg,
    )

    # Residual add in SBUF: bf16 add, same op/dtype/order as the previous
    # external `residual + hidden_states`. Both operands are channel-interleaved
    # so the per-element add covers matching H positions.
    nisa.tensor_tensor(dst=ar_sb, data1=ar_sb, data2=residual_sb, op=nl.add)

    # Store the post-residual H-shard back to HBM. _store_h_sharded_sbuf_output
    # expects channel-interleaved layout, which we just produced.
    output = nl.ndarray((T, _H), dtype=inp.dtype, buffer=nl.shared_hbm)
    _store_h_sharded_sbuf_output(
        ar_sb.reshape((_PMAX, T, _H_FREE_SHARD)), output, inp.dtype, T, sbm
    )

    sbm.close_scope()

    return output


def attn_fused_tkg(
    inp,                       # [T, H] HBM — pre-attn hidden_states; X for kernel and residual.
    gamma,                     # [1, H] HBM — input_layernorm.weight (rmsnorm_X gamma).
    q_norm_gamma,              # [1, d_head] HBM — q_layernorm.weight, used as pre-RoPE Q gamma.
    k_norm_gamma,              # [1, d_head] HBM — k_layernorm.weight, used as pre-RoPE K gamma.
    W_qkv,                     # HBM — fused QKV proj weights (transposed for kernel).
    W_out,                     # HBM — o_proj weights.
    K_cache,                   # HBM — KV cache (in-place updated when update_cache=True).
    V_cache,                   # HBM — KV cache.
    cos_table_T,               # [d_head//2, max_length] bf16 HBM — precomputed RoPE cos table (transposed for partition-mapped gather).
    sin_table_T,               # [d_head//2, max_length] bf16 HBM — precomputed RoPE sin table.
    attention_mask,            # HBM — already permuted to kernel format by host.
    position_ids,              # [B, S_total] int32 HBM — sliced to [B,1] inside kernel (host cast int64→int32 since NKI has no int64).
    rms_norm_eps,              # float — gamma RMSNorm eps.
    qk_norm_eps,               # float — pre-RoPE QK norm eps (Qwen3 uses it).
    softmax_scale,             # float or None — caller passes (1 / self.softmax_scale).
    K_cache_transposed,        # bool.
    is_pre_rope_qk_norm,       # bool — Qwen3 = True.
    rope_contiguous_layout,    # bool — Qwen3 = True (Llama3-style halves).
    update_cache,              # bool — True for our config (cache update inside kernel).
    replica_groups=None,
):
    """
    Fused attn TKG: attention_block_tkg + TP all-reduce + pre-attn residual add,
    all in SBUF (no HBM round-trip between the kernel's o_proj output and the
    AR/residual stages). Mirrors moe_fused_tkg's shape.

    Composition pattern (per nkilib/experimental/transformer/transformer_tkg.py:90,
    231): attention_block_tkg's @nki.jit body is composable when called from an
    outer non-jitted function with caller-managed sbm. The outer function is
    jitted at the call site (_attn_fused_nkilib_call below) — double-jit causes
    a stack overflow.

    Layout:
      - attention_block_tkg with transposed_out=True, out_in_sb=True returns
        [H_1=PMAX, H_2=H_FREE_SHARD, B*S=T] in SBUF (output_projection_tkg.py:118-130).
        Mapping: h = prg_id*H_SHARD + h_1*H_FREE_SHARD + h_2, where prg_id is the
        LNC core index. Channel-interleaved within this core's H shard.
      - For T=1 this matches the MoE convention [PMAX, T, H_FREE_SHARD] exactly
        (T dim collapses) so we can reuse moe_fused_tkg's residual DMA pattern
        and _store_h_sharded_sbuf_output. T>1 (speculation) would need a
        layout-aware residual load — not supported yet, asserted below.

    Bit-accuracy: same op/dtype/order as the host-side
    `reduce_from_tensor_model_parallel_region(attn_out)` + `residual + attn_out`
    chain. nccl.all_reduce with op=nl.add over the TP replica group, then bf16
    tensor_tensor add — identical to what moe_fused_tkg already proves works.

    KV cache: kernel writes K_cache/V_cache HBM in-place when update_cache=True.
    Returned tuple's K/V slots are the post-update cache views.
    """
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)

    T = inp.shape[0]
    # T=1 layout assumption — see docstring. Speculation (T>1) needs work.
    assert T == 1, f"attn_fused_tkg currently assumes T=B*S_tkg=1, got T={T}"

    prg_id = nl.program_id(axis=0)
    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    sbm.open_scope(name="attn_fused_tkg")

    # Phase 1 — kv_cache_update_idx slice on-device. Host now passes the full
    # int32 position_ids (cast happens on host because NKI rejects int64
    # inputs); the kernel slices [:, 0:1] and writes a contiguous [B, 1] int32
    # buffer in shared_hbm. The inner attn kernel asserts
    # kv_cache_update_idx.shape == (B, 1) (attention_block_tkg.py:1272-1273) and
    # DMAs it into a uint32 SBUF tile (:1290-1291). T=B*S_tkg=1 with S_tkg=1 →
    # B=1. Both LNC programs run this and write the same value to shared_hbm;
    # benign race (same value, no ordering needed before the inner-kernel read
    # in this same scope).
    B = 1
    kv_idx_sb = sbm.alloc_stack((B, 1), dtype=nl.int32, buffer=nl.sbuf, name="kv_idx_sb")
    nisa.dma_copy(dst=kv_idx_sb, src=position_ids[:, 0:1])
    kv_cache_update_idx_hbm = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=kv_cache_update_idx_hbm, src=kv_idx_sb)

    # Phase 2 — RoPE cos/sin via table gather. Tables are precomputed on host
    # via the same RotaryEmbedding.forward path used previously per-step
    # (fp32 inv_freq @ pos → cos/sin → bf16 cast), so values are bit-identical
    # to torch. We gather column `pos` from each table directly into a
    # [d_head//2, 1] SBUF tile (partition i ↔ frequency i), then DMA into a
    # [d_head//2, B, S_tkg] shared_hbm scratch in attention_block_tkg's
    # expected layout (attention_block_tkg.py:189-190). Avoids the bit-accuracy
    # gap we'd hit using on-device nl.cos/nl.sin or nisa.activation(op=nl.sin).
    # Uses Phase 1's kv_idx_sb as the int32 scalar offset; for our forced TKG
    # config rotary_position_ids == position_ids (host wrapper asserts).
    HALF_D = _PMAX // 2
    S_TKG = 1
    max_len = cos_table_T.shape[1]

    cos_sb = sbm.alloc_stack((HALF_D, 1), dtype=cos_table_T.dtype, buffer=nl.sbuf, name="rope_cos_sb")
    nisa.dma_copy(
        dst=cos_sb,
        src=cos_table_T.ap(
            pattern=[[max_len, HALF_D], [1, 1]],
            offset=0,
            scalar_offset=kv_idx_sb,
            indirect_dim=1,
        ),
    )
    sin_sb = sbm.alloc_stack((HALF_D, 1), dtype=sin_table_T.dtype, buffer=nl.sbuf, name="rope_sin_sb")
    nisa.dma_copy(
        dst=sin_sb,
        src=sin_table_T.ap(
            pattern=[[max_len, HALF_D], [1, 1]],
            offset=0,
            scalar_offset=kv_idx_sb,
            indirect_dim=1,
        ),
    )
    cos_hbm = nl.ndarray((HALF_D, B, S_TKG), dtype=cos_table_T.dtype, buffer=nl.shared_hbm, name="rope_cos_hbm")
    sin_hbm = nl.ndarray((HALF_D, B, S_TKG), dtype=sin_table_T.dtype, buffer=nl.shared_hbm, name="rope_sin_hbm")
    nisa.dma_copy(dst=cos_hbm.reshape((HALF_D, 1)), src=cos_sb)
    nisa.dma_copy(dst=sin_hbm.reshape((HALF_D, 1)), src=sin_sb)

    # attention_block_tkg expects X as [B, S_tkg, H]; flatten T → 1×T×H.
    X_hbm = inp.reshape((1, T, _H))

    # Fused attn block. transposed_out=True + out_in_sb=True keeps the o_proj
    # output in SBUF in the channel-interleaved layout described above. sbm is
    # shared so the inner kernel's SBUF lives in our scope.
    attn_kernel_out = _attention_block_tkg_call(
        X=X_hbm,
        X_hidden_dim_actual=None,
        rmsnorm_X_enabled=True,
        rmsnorm_X_eps=rms_norm_eps,
        rmsnorm_X_gamma=gamma,
        W_qkv=W_qkv,
        bias_qkv=None,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=is_pre_rope_qk_norm,
        rmsnorm_QK_pre_rope_eps=qk_norm_eps if is_pre_rope_qk_norm else 0.0,
        rmsnorm_QK_pre_rope_W_Q=q_norm_gamma if is_pre_rope_qk_norm else None,
        rmsnorm_QK_pre_rope_W_K=k_norm_gamma if is_pre_rope_qk_norm else None,
        cos=cos_hbm,
        sin=sin_hbm,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=0.0,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=K_cache_transposed,
        active_blocks_table=None,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=None,
        softmax_scale=softmax_scale,
        update_cache=update_cache,
        kv_cache_update_idx=kv_cache_update_idx_hbm,
        W_out=W_out,
        bias_out=None,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=True,
        out_in_sb=True,
        sbm=sbm,
    )
    attn_out_sb = attn_kernel_out[0]  # [PMAX, H_FREE_SHARD, T] SBUF, TP-partial.

    # Free attention block's heap allocations before AR + residual stages
    # (transformer_tkg.py:273-274 does the same to reclaim SBUF).
    while sbm.heap:
        sbm.pop_heap()

    # nccl.all_reduce wants 2D SBUF [PMAX, free]. For T=1 this is [PMAX, H_FREE_SHARD]
    # — same shape MoE uses post-AR. Same bf16 sum order as the host AR primitive.
    free_size = _H_FREE_SHARD * T
    ar_sb = nl.zeros(
        (_PMAX, free_size), dtype=inp.dtype, buffer=nl.sbuf, name="attn_ar_sb"
    )
    nccl.all_reduce(
        dsts=[ar_sb],
        srcs=[attn_out_sb.reshape((_PMAX, free_size))],
        op=nl.add,
        replica_group=rg,
    )

    # Load residual (= pre-attn `inp`) for this LNC core's H shard into SBUF
    # in the same channel-interleaved layout as attn_out_sb. Pattern levels
    # (innermost → outermost source iteration) match moe_fused_tkg's load:
    #   level 0: stride H_FREE_SHARD, count PMAX → partition p (high bits)
    #   level 1: stride 1,            count H_FREE_SHARD → h1 (low bits within shard)
    #   level 2: stride H,            count T → outer t (no-op for T=1)
    inp_flat = inp.reshape((T * _H,))
    residual_sb = sbm.alloc_stack(
        (_PMAX, free_size), inp.dtype, buffer=nl.sbuf, name="attn_residual_sb"
    )
    h_load_pattern = [[_H_FREE_SHARD, _PMAX], [1, _H_FREE_SHARD], [_H, T]]
    nisa.dma_copy(
        dst=residual_sb,
        src=inp_flat.ap(pattern=h_load_pattern, offset=prg_id * _H_SHARD),
        dge_mode=nisa.dge_mode.hwdge,
    )

    # Residual add in SBUF: bf16 add, same op/dtype/order as previous external
    # `residual + hidden_states` add at qwen_with_nki.py:1187.
    nisa.tensor_tensor(dst=ar_sb, data1=ar_sb, data2=residual_sb, op=nl.add)

    # Store post-residual H-shard back to HBM. _store_h_sharded_sbuf_output
    # consumes the same channel-interleaved layout we produced.
    output = nl.ndarray((T, _H), dtype=inp.dtype, buffer=nl.shared_hbm)
    _store_h_sharded_sbuf_output(
        ar_sb.reshape((_PMAX, T, _H_FREE_SHARD)), output, inp.dtype, T, sbm
    )

    sbm.close_scope()

    return output


def attn_fused_tkg_phase1_only(
    inp,
    gamma,
    q_norm_gamma,
    k_norm_gamma,
    W_qkv,
    W_out,
    K_cache,
    V_cache,
    cos,                       # [d_head//2, B, S_tkg] bf16 HBM — host-computed via attn.rotary_emb.
    sin,                       # [d_head//2, B, S_tkg] bf16 HBM — host-computed via attn.rotary_emb.
    attention_mask,
    position_ids,              # [B, S_total] int32 HBM — sliced to [B,1] inside kernel for Phase 1.
    rms_norm_eps,
    qk_norm_eps,
    softmax_scale,
    K_cache_transposed,
    is_pre_rope_qk_norm,
    rope_contiguous_layout,
    update_cache,
    replica_groups=None,
):
    """Phase 1-only variant of attn_fused_tkg for A/B testing against the
    Phase 1+2 path (which adds on-device cos/sin table gather).

    Identical to attn_fused_tkg except:
      - cos/sin are passed in directly as HBM (computed by host via
        attn.rotary_emb), not gathered from precomputed tables on-device.
      - Phase 1's int32 kv_cache_update_idx scratch path is preserved.

    Toggled via _PHASE2_ROPE_ENABLED at module scope. Lets us bisect whether
    residual logit drift is from Phase 2 (table gather) or Phase 1 (kv_idx
    scratch) without ripping either out.
    """
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)

    T = inp.shape[0]
    assert T == 1, f"attn_fused_tkg_phase1_only currently assumes T=B*S_tkg=1, got T={T}"

    prg_id = nl.program_id(axis=0)
    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    sbm.open_scope(name="attn_fused_tkg_phase1_only")

    # Phase 1 — kv_cache_update_idx slice on-device (same as the Phase 1+2 kernel).
    B = 1
    kv_idx_sb = sbm.alloc_stack((B, 1), dtype=nl.int32, buffer=nl.sbuf, name="kv_idx_sb")
    nisa.dma_copy(dst=kv_idx_sb, src=position_ids[:, 0:1])
    kv_cache_update_idx_hbm = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=kv_cache_update_idx_hbm, src=kv_idx_sb)

    # No Phase 2 — cos/sin come straight from host as HBM tensors.

    X_hbm = inp.reshape((1, T, _H))
    attn_kernel_out = _attention_block_tkg_call(
        X=X_hbm,
        X_hidden_dim_actual=None,
        rmsnorm_X_enabled=True,
        rmsnorm_X_eps=rms_norm_eps,
        rmsnorm_X_gamma=gamma,
        W_qkv=W_qkv,
        bias_qkv=None,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=is_pre_rope_qk_norm,
        rmsnorm_QK_pre_rope_eps=qk_norm_eps if is_pre_rope_qk_norm else 0.0,
        rmsnorm_QK_pre_rope_W_Q=q_norm_gamma if is_pre_rope_qk_norm else None,
        rmsnorm_QK_pre_rope_W_K=k_norm_gamma if is_pre_rope_qk_norm else None,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=0.0,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=K_cache_transposed,
        active_blocks_table=None,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=None,
        softmax_scale=softmax_scale,
        update_cache=update_cache,
        kv_cache_update_idx=kv_cache_update_idx_hbm,
        W_out=W_out,
        bias_out=None,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=True,
        out_in_sb=True,
        sbm=sbm,
    )
    attn_out_sb = attn_kernel_out[0]

    while sbm.heap:
        sbm.pop_heap()

    free_size = _H_FREE_SHARD * T
    ar_sb = nl.zeros(
        (_PMAX, free_size), dtype=inp.dtype, buffer=nl.sbuf, name="attn_ar_sb"
    )
    nccl.all_reduce(
        dsts=[ar_sb],
        srcs=[attn_out_sb.reshape((_PMAX, free_size))],
        op=nl.add,
        replica_group=rg,
    )

    inp_flat = inp.reshape((T * _H,))
    residual_sb = sbm.alloc_stack(
        (_PMAX, free_size), inp.dtype, buffer=nl.sbuf, name="attn_residual_sb"
    )
    h_load_pattern = [[_H_FREE_SHARD, _PMAX], [1, _H_FREE_SHARD], [_H, T]]
    nisa.dma_copy(
        dst=residual_sb,
        src=inp_flat.ap(pattern=h_load_pattern, offset=prg_id * _H_SHARD),
        dge_mode=nisa.dge_mode.hwdge,
    )

    nisa.tensor_tensor(dst=ar_sb, data1=ar_sb, data2=residual_sb, op=nl.add)

    output = nl.ndarray((T, _H), dtype=inp.dtype, buffer=nl.shared_hbm)
    _store_h_sharded_sbuf_output(
        ar_sb.reshape((_PMAX, T, _H_FREE_SHARD)), output, inp.dtype, T, sbm
    )

    sbm.close_scope()
    return output


# Toggle for Phase 2 (on-device RoPE cos/sin via precomputed table gather).
# True  → kernel does table gather; host passes rope_*_table_T buffers.
# False → host computes cos/sin via attn.rotary_emb (un-fused-hoist style)
#         and passes them directly; kernel skips Phase 2 prep.
# Used for A/B bisecting Phase 2 vs Phase 1 contributions to logit drift.
_PHASE2_ROPE_ENABLED = False


_flash_fwd_call = nki_jit()(attention_isa_kernel)
_moe_fused_nkilib_call = nki.jit(moe_fused_tkg)
_attn_fused_nkilib_call = nki.jit(attn_fused_tkg)
_attn_fused_nkilib_call_phase1_only = nki.jit(attn_fused_tkg_phase1_only)


def _nki_data(tensor):
    return tensor.data if tensor is not None and hasattr(tensor, "data") else tensor


# Module-level cache for RoPE tables. Keyed on (head_dim, max_length,
# rope_theta, dtype) so different configs don't collide. The first decoder
# layer's __init__ triggers the XLA precompute (~few seconds for torch_xla
# spinup); the remaining 47 layers reuse the same Python tensor objects, so
# NxDI's tracer sees one storage for the constant — no per-layer HBM copy.
_ROPE_TABLE_CACHE = {}


def _precompute_rope_tables_xla(head_dim, max_length, rope_theta, dtype):
    """Run RotaryEmbedding on the XLA device and return CPU bf16 tables.

    Why XLA and not CPU: the un-fused reference path (NxDI's runtime
    attn.rotary_emb) compiles to Neuron's hardware cos/sin, while CPU torch
    uses libm's polynomial approximation. Empirically these disagree by
    1 bf16 ULP on ~0.1% of values for Qwen3 (head_dim=128, max_length=768),
    which accumulates to >0.5 normalized logit error after 48 RoPE
    invocations. Running the precompute on XLA produces values bit-identical
    to runtime so the gathered cos/sin in the kernel match what the un-fused
    path would have produced.

    Returns CPU tensors so subsequent register_buffer calls and tracing work
    in the standard way (no XLA-resident lifetime concerns).
    """
    import torch_xla.core.xla_model as xm

    half_d = head_dim // 2
    device = xm.xla_device()
    rope_emb = RotaryEmbedding(
        head_dim, max_position_embeddings=max(max_length, 32768), base=rope_theta
    ).to(device)
    positions = torch.arange(max_length, dtype=torch.long, device=device).unsqueeze(0)
    dummy_x = torch.empty((1, 1, head_dim), dtype=dtype, device=device)
    cos_full, sin_full = rope_emb(dummy_x, positions)
    cos_T = cos_full[0, :, :half_d].t().contiguous()
    sin_T = sin_full[0, :, :half_d].t().contiguous()
    xm.mark_step()
    return cos_T.cpu(), sin_T.cpu()


def _build_rope_tables(rotary_emb, max_length, head_dim, dtype):
    """Get cos/sin RoPE tables, precomputing on XLA once and caching.

    Mirrors what the host previously did per-step:
        cos_full, sin_full = rotary_emb(hidden_states, position_ids)
        cos = cos_full[..., :half_d].permute(2, 0, 1)  # [half_d, B, S]
    Bit-identical to runtime because the precompute also runs on XLA.

    Layout: tables transposed to [head_dim//2, max_length] for direct
    partition-mapped gather in the kernel (partition i ↔ frequency i, scalar
    offset = pos selects the column).

    Cache: first decoder layer triggers the XLA precompute; layers 1-47 hit
    the cache and reuse the same tensor objects, so NxDI sees one constant
    storage rather than 48 duplicates.
    """
    rope_theta = rotary_emb.base
    cache_key = (head_dim, max_length, rope_theta, str(dtype))
    if cache_key not in _ROPE_TABLE_CACHE:
        _ROPE_TABLE_CACHE[cache_key] = _precompute_rope_tables_xla(
            head_dim, max_length, rope_theta, dtype
        )
    return _ROPE_TABLE_CACHE[cache_key]


SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

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
            # If shard-on-I enabled, check the intermediate size per tp is divisible by SHARD_ON_INTERMEDIATE_DIMENTION_PER_TP
            if I_TP % SHARD_ON_INTERMEDIATE_DIMENTION_PER_TP != 0:
                padded_moe_intermediate_size = math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENTION_PER_TP) * SHARD_ON_INTERMEDIATE_DIMENTION_PER_TP * moe_tp_degree
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

    def attention_block_tokengen_nki_kernel(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        update_kv_per_layer: bool = True,
        active_block_table: Optional[torch.Tensor] = None,
        use_polar_compatible_rope: bool = False,
    ):
        # Hoisted from NeuronAttentionBase.attention_block_tokengen_nki_kernel so
        # the attn TKG kernel call site lives in this file (mirroring the MoE
        # fused TKG hoist). Body is byte-for-byte equivalent to the upstream
        # method; only the kernel-handle name changes.
        bsz, s_tkg, h = hidden_states.shape
        h_out = h // 2 if self.is_eagle3_draft else h
        num_q_heads = self.num_heads

        rmsnorm_enabled = rmsnorm is not None
        W_gamma = rmsnorm.weight.data.unsqueeze(0) if rmsnorm is not None else None

        rope_contiguous_layout = not use_polar_compatible_rope

        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(hidden_states, rotary_position_ids)
                cos_cache = cos_cache[..., : cos_cache.shape[-1] // 2].permute(2, 0, 1)
                sin_cache = sin_cache[..., : sin_cache.shape[-1] // 2].permute(2, 0, 1)
        else:
            cos_cache = None
            sin_cache = None

        attention_mask = attention_mask.expand(-1, num_q_heads, -1, -1)
        expected_active_mask_shape = (bsz, 1, s_tkg, s_tkg)
        if s_tkg == 1:
            active_mask = torch.ones(
                expected_active_mask_shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        else:
            assert (
                active_mask.shape == expected_active_mask_shape
            ), f"{active_mask.shape} != {expected_active_mask_shape}"
        active_mask = active_mask.expand(-1, num_q_heads, -1, -1)
        attention_mask[:, :, :, -s_tkg:] = active_mask
        attention_mask = attention_mask.permute(3, 0, 1, 2)

        K_prior, V_prior = past_key_value[:2]
        K_prior = K_prior.data
        V_prior = V_prior.data
        update_cache_in_kernel = update_kv_per_layer and self.attn_block_tkg_nki_kernel_cache_update
        sink = self.get_learned_sinks().data.unsqueeze(-1) if self.learned_sinks_size is not None else None
        kv_cache_update_idx = position_ids[:, :1].to(torch.int32)

        W_out = self.get_o_proj().o_proj.weight.data
        if self.o_bias:
            W_out_bias_param = self.get_o_proj().o_proj.bias / self.tp_degree
            W_out_bias = W_out_bias_param.data.unsqueeze(0)
        else:
            W_out_bias = None

        has_qk_layernorm = self.q_layernorm is not None and self.k_layernorm is not None
        qk_norm_eps = self.rms_norm_eps if self.rms_norm_eps else 1e-6

        is_pre_rope_qk_norm = has_qk_layernorm and self.qk_norm_placement == QKNormPlacement.PRE_ROPE
        rmsnorm_QK_pre_rope_W_Q = self.q_layernorm.weight.data.unsqueeze(0) if is_pre_rope_qk_norm else None
        rmsnorm_QK_pre_rope_W_K = self.k_layernorm.weight.data.unsqueeze(0) if is_pre_rope_qk_norm else None

        is_post_rope_qk_norm = has_qk_layernorm and self.qk_norm_placement == QKNormPlacement.POST_ROPE
        rmsnorm_QK_post_rope_W_Q = self.q_layernorm.weight.data.unsqueeze(0) if is_post_rope_qk_norm else None
        rmsnorm_QK_post_rope_W_K = self.k_layernorm.weight.data.unsqueeze(0) if is_post_rope_qk_norm else None

        attn_output, K, V = _attention_block_tkg_call[self.logical_nc_config](
            X=hidden_states,
            X_hidden_dim_actual=getattr(self.config, "original_hidden_size", None),
            rmsnorm_X_enabled=rmsnorm_enabled,
            rmsnorm_X_eps=self.rms_norm_eps,
            rmsnorm_X_gamma=W_gamma,
            W_qkv=self.get_qkv_proj().Wqkv.weight.data,
            bias_qkv=self.get_qkv_proj().Wqkv.bias.data.unsqueeze(0) if self.qkv_bias else None,
            quantization_type_qkv=QuantizationType.NONE,
            weight_dequant_scale_qkv=None,
            input_dequant_scale_qkv=None,
            rmsnorm_QK_pre_rope_enabled=is_pre_rope_qk_norm,
            rmsnorm_QK_pre_rope_eps=qk_norm_eps if is_pre_rope_qk_norm else 0.0,
            rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
            rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
            cos=cos_cache,
            sin=sin_cache,
            rope_contiguous_layout=rope_contiguous_layout,
            rmsnorm_QK_post_rope_enabled=is_post_rope_qk_norm,
            rmsnorm_QK_post_rope_eps=qk_norm_eps if is_post_rope_qk_norm else 0.0,
            rmsnorm_QK_post_rope_W_Q=rmsnorm_QK_post_rope_W_Q,
            rmsnorm_QK_post_rope_W_K=rmsnorm_QK_post_rope_W_K,
            K_cache_transposed=self.k_cache_transposed,
            active_blocks_table=active_block_table.to(torch.uint32) if active_block_table is not None else None,
            K_cache=K_prior,
            V_cache=V_prior,
            attention_mask=attention_mask,
            sink=sink,
            softmax_scale=None if self.softmax_scale is None else (1 / self.softmax_scale),
            update_cache=update_cache_in_kernel,
            kv_cache_update_idx=kv_cache_update_idx,
            W_out=W_out,
            bias_out=W_out_bias,
            quantization_type_out=QuantizationType.NONE,
            weight_dequant_scale_out=None,
            input_dequant_scale_out=None,
            transposed_out=False,
            out_in_sb=False,
        )

        attn_output = attn_output.reshape((bsz, s_tkg, h_out))
        # Qwen forced config: SP off, DP=1 → only the AR_AG branch is reachable.
        # Other dispatch options remain in the upstream method for general models.
        assert not self.sequence_parallel_enabled
        assert self.ep_dispatch_cc_option == EPDispatchOption.AR_AG
        attn_output = mappings.reduce_from_tensor_model_parallel_region(
            attn_output, process_group=self.tensor_model_parallel_group
        )

        if not update_cache_in_kernel:
            K = K.permute(1, 0, 2) if self.k_cache_transposed else K.permute(1, 2, 0)
            K = K.unsqueeze(1)

        return attn_output, (K, V), cos_cache, sin_cache


class NeuronQwen3MoeDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3MoEAttention(config=config)
        self.moe_fused_nki_kernel_enabled = getattr(config, "moe_fused_nki_kernel_enabled", False)
        self.attn_fused_nki_kernel_enabled = getattr(
            config.neuron_config, "attn_block_tkg_nki_kernel_enabled", False
        )

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

        # Replica groups for nccl.all_reduce inside the fused MoE TKG kernel.
        # Format mirrors qwen_fused_transformer_multilayer.py — derived from
        # tp_degree to avoid torch.distributed calls during mock tracing.
        self._replica_groups = (list(range(config.neuron_config.tp_degree)),)

        # Phase 2 — precompute RoPE cos/sin tables once at __init__. Tables are
        # bit-identical to host-side `attn.rotary_emb(x, pos)[..., :half].permute(2,0,1)`
        # because we call the same RotaryEmbedding.forward (fp32 matmul + cos/sin
        # then bf16 cast). At kernel runtime we gather column `pos` from each
        # table, eliminating the per-step host trig op without introducing any
        # on-device trig (nl.cos/nl.sin aren't bit-accurate to torch).
        # Layout: [head_dim//2, max_length] bf16 contiguous — transposed for
        # direct partition-mapped gather (partition i ↔ frequency i, scalar
        # offset = pos selects the column).
        self._rope_cos_table_T, self._rope_sin_table_T = _build_rope_tables(
            self.self_attn.rotary_emb,
            max_length=config.neuron_config.max_length,
            head_dim=config.head_dim,
            dtype=config.neuron_config.torch_dtype,
        )
        self.register_buffer("rope_cos_table_T", self._rope_cos_table_T, persistent=False)
        self.register_buffer("rope_sin_table_T", self._rope_sin_table_T, persistent=False)

    def _attn_fused_nkilib_tkg(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value,
        active_mask: Optional[torch.Tensor] = None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Fused attn TKG path: replaces input_layernorm + self_attn(...) + (residual
        + attn_out) with a single _attn_fused_nkilib_call invocation. Kernel does
        input RMSNorm + QKV + attn + o_proj + TP all-reduce + residual-add in SBUF.

        Host-side prep mirrors NeuronQwen3MoEAttention.attention_block_tokengen_nki_kernel
        for the mask permute. RoPE cos/sin (Phase 2) is now done via on-device
        gather from precomputed tables registered on this layer
        (rope_cos_table_T / rope_sin_table_T) — the host no longer calls
        attn.rotary_emb. kv_cache_update_idx (Phase 1) is derived on-device from
        position_ids. Weights are pulled from self.self_attn so they're the exact
        param instances NxDI sets up.

        KV cache is updated in-place inside the kernel; returned present_key_value
        references the same buffers as past_key_value (matches the un-fused path's
        update_cache_in_kernel=True branch in attention_base.py:1378-1381).
        """
        attn = self.self_attn
        bsz, s_tkg, h = hidden_states.shape
        num_q_heads = attn.num_heads

        # Phase 2 toggle. When ON: RoPE is gathered from precomputed tables on
        # device (self.rope_*_table_T). When OFF: host computes cos/sin via
        # attn.rotary_emb verbatim from the un-fused hoist and passes them as
        # HBM into the phase1-only kernel variant. Lets us bisect Phase 1 vs
        # Phase 2 contributions to logit drift.
        rotary_position_ids = kwargs.get("rotary_position_ids", None)
        if rotary_position_ids is None:
            rotary_position_ids = position_ids
        if _PHASE2_ROPE_ENABLED and rotary_position_ids is not position_ids:
            assert torch.equal(rotary_position_ids, position_ids), (
                "Phase 2 RoPE table-gather assumes rotary_position_ids == position_ids; "
                "got distinct tensors with different values."
            )
        use_polar_compatible_rope = kwargs.get("use_polar_compatible_rope", False)
        rope_contiguous_layout = not use_polar_compatible_rope

        # Attention mask prep (override:1143-1157)
        attention_mask = attention_mask.expand(-1, num_q_heads, -1, -1)
        expected_active_mask_shape = (bsz, 1, s_tkg, s_tkg)
        if s_tkg == 1:
            active_mask = torch.ones(
                expected_active_mask_shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        else:
            assert (
                active_mask.shape == expected_active_mask_shape
            ), f"{active_mask.shape} != {expected_active_mask_shape}"
        active_mask = active_mask.expand(-1, num_q_heads, -1, -1)
        attention_mask[:, :, :, -s_tkg:] = active_mask
        attention_mask = attention_mask.permute(3, 0, 1, 2)

        # KV cache + update idx (override:1159-1164). NKI doesn't accept int64
        # tensors at all (no int64 in supported dtype list), so the int32 cast
        # has to stay on host. The kernel still does the [B, 1] slice + scratch
        # materialization on-device — this is the part Phase 1 actually moves.
        K_cache = past_key_value[0].data
        V_cache = past_key_value[1].data
        position_ids_i32 = position_ids.to(torch.int32)

        # Weights and norm gammas (override:1166-1182)
        gamma = self.input_layernorm.weight.data.unsqueeze(0)
        has_qk_layernorm = attn.q_layernorm is not None and attn.k_layernorm is not None
        is_pre_rope_qk_norm = (
            has_qk_layernorm and attn.qk_norm_placement == QKNormPlacement.PRE_ROPE
        )
        q_norm_gamma = attn.q_layernorm.weight.data.unsqueeze(0) if is_pre_rope_qk_norm else None
        k_norm_gamma = attn.k_layernorm.weight.data.unsqueeze(0) if is_pre_rope_qk_norm else None
        W_qkv = attn.get_qkv_proj().Wqkv.weight.data
        W_out = attn.get_o_proj().o_proj.weight.data

        qk_norm_eps = attn.rms_norm_eps if attn.rms_norm_eps else 1e-6
        softmax_scale = None if attn.softmax_scale is None else (1 / attn.softmax_scale)

        # Wrapper takes [B*S_tkg, H]; kernel itself reshapes to [1, T, H].
        inp_2d = hidden_states.reshape(-1, self.hidden_size)

        if _PHASE2_ROPE_ENABLED:
            output_2d = _attn_fused_nkilib_call[attn.logical_nc_config](
                inp_2d,
                gamma,
                q_norm_gamma,
                k_norm_gamma,
                W_qkv,
                W_out,
                K_cache,
                V_cache,
                self.rope_cos_table_T,
                self.rope_sin_table_T,
                attention_mask,
                position_ids_i32,
                attn.rms_norm_eps,
                qk_norm_eps,
                softmax_scale,
                attn.k_cache_transposed,
                is_pre_rope_qk_norm,
                rope_contiguous_layout,
                True,  # update_cache — forced by attn_block_tkg_nki_kernel_cache_update flag
                replica_groups=self._replica_groups,
            )
        else:
            # Phase 1-only path: host computes cos/sin via attn.rotary_emb
            # (verbatim from un-fused hoist, attention_base.py:1136-1141), then
            # passes HBM bf16 [d_head//2, B, S_tkg] tensors to the kernel.
            cos_cache_h, sin_cache_h = attn.rotary_emb(hidden_states, rotary_position_ids)
            cos_cache_h = cos_cache_h[..., : cos_cache_h.shape[-1] // 2].permute(2, 0, 1)
            sin_cache_h = sin_cache_h[..., : sin_cache_h.shape[-1] // 2].permute(2, 0, 1)
            output_2d = _attn_fused_nkilib_call_phase1_only[attn.logical_nc_config](
                inp_2d,
                gamma,
                q_norm_gamma,
                k_norm_gamma,
                W_qkv,
                W_out,
                K_cache,
                V_cache,
                cos_cache_h,
                sin_cache_h,
                attention_mask,
                position_ids_i32,
                attn.rms_norm_eps,
                qk_norm_eps,
                softmax_scale,
                attn.k_cache_transposed,
                is_pre_rope_qk_norm,
                rope_contiguous_layout,
                True,
                replica_groups=self._replica_groups,
            )
        output = output_2d.reshape(bsz, s_tkg, h)

        # In-place KV update — return the same cache buffers as present_key_value.
        present_key_value = (past_key_value[0], past_key_value[1])

        # cos_cache/sin_cache are returned only to satisfy NxDI's per-call cache
        # plumbing convention (forward() unpacks them). With Phase 2 the kernel
        # owns RoPE via tables, so they're no longer meaningful — return None.
        return output, present_key_value, None, None

    def _moe_fused_nkilib_tkg(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        hidden_states_2d = hidden_states.reshape(-1, self.hidden_size)

        # Keep the same weights NxDI's token-gen MoE wrapper owns; only swap the NKI entry point.
        gamma = _nki_data(self.post_attention_layernorm.weight.unsqueeze(0))
        router_w = _nki_data(self.mlp.router.weight_T)
        gate_up_w = _nki_data(self.mlp.expert_mlps.mlp_op.gate_up_proj.weight)
        down_w = _nki_data(self.mlp.expert_mlps.mlp_op.down_proj.weight)
        logical_nc_config = self.mlp.moe_fused_tkg.logical_nc_config

        # Kernel does the TP all-reduce + residual add internally (in SBUF),
        # so the returned tensor is the post-residual hidden state.
        output_2d = _moe_fused_nkilib_call[logical_nc_config](
            hidden_states_2d,
            gamma,
            router_w,
            gate_up_w,
            down_w,
            replica_groups=self._replica_groups,
        )
        return output_2d.reshape(hidden_states_shape)

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

        # We wrap input_layernorm/self_attn/post_attention_layernorm with module markers start/end
        # as a hint for compiler's modular-flow to avoid layer boundries in-between decoder layer components
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)

        # Self Attention
        if self.attn_fused_nki_kernel_enabled and past_key_value is not None:
            # Fused-NKI attn TKG path: kernel does input_layernorm + QKV + attn
            # + o_proj + TP all-reduce + pre-attn residual add internally
            # (residual stashed in SBUF at kernel entry, added back after the
            # in-SBUF AR). Mirrors the MoE wrapper below.
            hidden_states, present_key_value, cos_cache, sin_cache = self._attn_fused_nkilib_tkg(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                **kwargs,
            )
        else:
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
            hidden_states = residual + hidden_states

        # MoE
        if self.moe_fused_nki_kernel_enabled and past_key_value is not None:
            # Fused-NKI TKG path: kernel does post_attention_layernorm + MoE
            # + TP all-reduce + residual add internally (residual stashed in
            # SBUF at kernel entry, added back after the in-SBUF AR).
            hidden_states = self._moe_fused_nkilib_tkg(hidden_states)
        else:
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
