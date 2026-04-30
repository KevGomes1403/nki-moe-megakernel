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
    "output_logits": True,
    "async_mode": True,
    "fused_qkv": True
}


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
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
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
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode, RouterActFnType
from nkilib.core.utils.interleave_copy import interleave_copy


_PMAX = 128
_H    = 2048
_H_FREE = _H // _PMAX            # 16
_N_PRGS = 2
_H_FREE_SHARD = _H_FREE // _N_PRGS  # 8
_H_SHARD = _H_FREE_SHARD * _PMAX    # 1024
_EPS  = 1e-6


def _rmsnorm_sbuf_in_sbuf_out_hoisted(
    inp_sb,               # [PMAX, H_free*T] bf16 in SBUF
    dtype,
    T,
    gamma,                # [1, H] bf16 HBM
    sbm=None,
    gamma_sb_ready=None,  # [PMAX, H_free] bf16 SBUF pre-loaded
    debug_rmsnorm_out=None,
):
    """
    RMSNorm sub-kernel. inp_sb already in SBUF.

    Heap lifecycle (caller's responsibility):
      alloc_heap rmsnorm_normed_bf16 [PMAX, H_free*T] bf16  — returned; caller pops.
      alloc_heap rmsnorm_normed [PMAX, H_free*T] fp32        — popped before returning.

    Returns rmsnorm_normed_bf16 [PMAX, H_free*T] bf16 heap tensor.
    """
    H      = _H
    H_free = _H_FREE
    B      = T
    prg_id = nl.program_id(axis=0)

    # heap: bf16 normed — lives through router + expert stages; caller pops
    rmsnorm_normed_bf16 = sbm.alloc_heap((_PMAX, H_free * T), dtype, buffer=nl.sbuf, name="rmsnorm_normed_bf16")
    # heap: fp32 normed — only needed for gamma multiply; popped before returning
    rmsnorm_normed      = sbm.alloc_heap((_PMAX, H_free * T), nl.float32, buffer=nl.sbuf, name="rmsnorm_normed")

    sbm.open_scope(name="rmsnorm")

    rmsnorm_out = inp_sb

    # --- gamma load (skipped when gamma_sb_ready is supplied) ---
    if gamma_sb_ready is None:
        gamma_sb = sbm.alloc_stack((_PMAX, H_free), gamma.dtype, buffer=nl.sbuf, name="gamma_sb")
        gamma_1d = gamma.reshape((H,))
        for h1 in nl.static_range(H_free):
            shard = h1 // _H_FREE_SHARD
            h2 = h1 - shard * _H_FREE_SHARD
            nisa.dma_copy(
                dst=gamma_sb[0:_PMAX, h1:h1 + 1],
                src=gamma_1d.ap(
                    pattern=[[_H_FREE_SHARD, _PMAX], [1, 1]],
                    offset=shard * _H_SHARD + h2,
                ),
                dge_mode=3,
            )
    else:
        gamma_sb = gamma_sb_ready  # caller-owned — do NOT free

    rmsnorm_sq = sbm.alloc_stack((_PMAX, H_free * T), nl.float32, buffer=nl.sbuf, name="rmsnorm_sq")
    nisa.activation(rmsnorm_sq[...], op=nl.square, data=rmsnorm_out[...])

    # Within-partition reduce: [PMAX, H_free*T] → [PMAX, T]
    rmsnorm_reduced = sbm.alloc_stack((_PMAX, T), nl.float32, buffer=nl.sbuf, name="rmsnorm_reduced")
    nisa.tensor_reduce(rmsnorm_reduced[0:_PMAX, 0:T], nl.add, rmsnorm_sq[0:_PMAX, 0:H_free * T], axis=1)

    # Cross-partition reduce via reduce-as-MATMUL with an all-ones [PMAX, PMAX] stationary
    # operand. Matches AwsNeuronRmsNorm's HLO lowering (nxdi_moe.md §10.8).
    mm_ones = sbm.alloc_stack((_PMAX, _PMAX), nl.float32, buffer=nl.sbuf, name="rmsnorm_mm_ones")
    nisa.memset(mm_ones, value=1.0)
    final_reduced_psum = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=final_reduced_psum[0:_PMAX, 0:T],
        stationary=mm_ones[0:_PMAX, 0:_PMAX],
        moving=rmsnorm_reduced[0:_PMAX, 0:T],
    )

    eps_sb = sbm.alloc_stack((_PMAX, 1), nl.float32, buffer=nl.sbuf, name="eps_sb")
    nisa.memset(eps_sb, value=_EPS)
    norm_factor_sb = sbm.alloc_stack((_PMAX, T), nl.float32, buffer=nl.sbuf, name="norm_factor_sb")
    nisa.activation(
        norm_factor_sb[0:_PMAX, 0:T],
        op=nl.rsqrt,
        data=final_reduced_psum[0:_PMAX, 0:T],
        scale=1.0 / H,
        bias=eps_sb[0:_PMAX, :],
    )

    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=2).broadcast(dim=2, size=H_free)
    x_scaled = sbm.alloc_stack((_PMAX, H_free * T), nl.float32, buffer=nl.sbuf, name="x_scaled")
    nisa.tensor_tensor(x_scaled[...], rmsnorm_out[...], norm_factor_bcast.get_view(), nl.multiply)
    # rmsnorm_normed is a heap tensor (allocated before this scope) — write into it directly
    nisa.tensor_tensor(rmsnorm_normed[...], x_scaled[...], gamma_sb[...], nl.multiply)

    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=rmsnorm_normed[...])

    sbm.close_scope()  # frees rmsnorm stack tensors (not gamma_sb_ready — caller-owned)

    # Debug: DMA rmsnorm_normed_bf16 [PMAX, H_free*T] → HBM [T, H] (prg_id 0 only).
    if debug_rmsnorm_out is not None:
        if prg_id == 0:
            debug_flat = debug_rmsnorm_out.reshape((B, H))
            for _t in nl.static_range(B):
                for _h1 in nl.static_range(H_free):
                    _dbg_psum = nl.ndarray((1, _PMAX), dtype=dtype, buffer=nl.psum)
                    nisa.nc_transpose(
                        _dbg_psum,
                        rmsnorm_normed_bf16[0:_PMAX, _h1 * B + _t : _h1 * B + _t + 1],
                    )
                    _dbg_sb = nl.ndarray((1, _PMAX), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(_dbg_sb, _dbg_psum)
                    nisa.dma_copy(
                        dst=debug_flat[_t : _t + 1, _h1 * _PMAX : (_h1 + 1) * _PMAX],
                        src=_dbg_sb,
                        dge_mode=nisa.dge_mode.hwdge,
                    )

    sbm.pop_heap()  # rmsnorm_normed fp32 — not needed past this point

    return rmsnorm_normed_bf16  # [PMAX, H_free*T] bf16 heap; caller pops


@nki.jit
def rmsnorm_hbm(inp, gamma, hoisted_gamma=False, debug_rmsnorm_out=None):
    """
    HBM wrapper for RMSNorm.

    inp:   [T, H=2048] bf16 HBM
    gamma: [1, H=2048] bf16 HBM
    Returns output [T, H=2048] bf16 HBM (RMSNorm applied; both LNC cores
    produce identical results, prg_id==0 writes the output).
    """
    sbm    = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    T      = inp.shape[0]
    H_free = _H_FREE
    prg_id = nl.program_id(axis=0)

    # --- Load inp into SBUF ---
    sbm.open_scope(name="inp_load")
    inp_2d = inp.reshape((T, _H))
    inp_sb = sbm.alloc_stack((_PMAX, H_free * T), inp.dtype, buffer=nl.sbuf, name="inp_sb")
    for t in nl.static_range(T):
        for h1 in nl.static_range(H_free):
            shard = h1 // _H_FREE_SHARD
            h2 = h1 - shard * _H_FREE_SHARD
            nisa.dma_copy(
                dst=inp_sb[0:_PMAX, h1 * T + t:h1 * T + t + 1],
                src=inp_2d.ap(
                    pattern=[[_H_FREE_SHARD, _PMAX], [1, 1]],
                    offset=t * _H + shard * _H_SHARD + h2,
                ),
                dge_mode=3,
            )

    # --- Optionally pre-load gamma into SBUF ---
    gamma_sb_ready = None
    if hoisted_gamma:
        sbm.open_scope(name="gamma_hoist")
        gamma_sb_ready     = sbm.alloc_stack((_PMAX, H_free), gamma.dtype, buffer=nl.sbuf, name="gamma_sb_hoist")
        gamma_1d           = gamma.reshape((_H,))
        for h1 in nl.static_range(H_free):
            shard = h1 // _H_FREE_SHARD
            h2 = h1 - shard * _H_FREE_SHARD
            nisa.dma_copy(
                dst=gamma_sb_ready[0:_PMAX, h1:h1 + 1],
                src=gamma_1d.ap(
                    pattern=[[_H_FREE_SHARD, _PMAX], [1, 1]],
                    offset=shard * _H_SHARD + h2,
                ),
                dge_mode=3,
            )

    rmsnorm_normed_bf16 = _rmsnorm_sbuf_in_sbuf_out_hoisted(
        inp_sb, inp.dtype, T, gamma, sbm=sbm,
        gamma_sb_ready=gamma_sb_ready,
        debug_rmsnorm_out=debug_rmsnorm_out,
    )

    # --- Store output to HBM: [PMAX, H_free*T] col-major → [T, H] row-major ---
    output = nl.ndarray((T, _H), dtype=inp.dtype, buffer=nl.shared_hbm)
    sbm.open_scope(name="store_hbm")
    out_row_sb = sbm.alloc_stack((T, _H), inp.dtype, buffer=nl.sbuf, name="out_row_sb")
    for h1 in nl.static_range(H_free):
        tp_psum = nl.ndarray((T, _PMAX), dtype=inp.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=tp_psum[0:T, 0:_PMAX],
            data=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
        )
        nisa.activation(
            dst=out_row_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)],
            op=nl.copy,
            data=tp_psum[0:T, 0:_PMAX],
        )
    if prg_id == 0:
        nisa.dma_copy(dst=output[0:T, 0:_H], src=out_row_sb[0:T, 0:_H])
    sbm.close_scope()  # store_hbm

    sbm.pop_heap()  # rmsnorm_normed_bf16

    if hoisted_gamma:
        sbm.close_scope()  # gamma_hoist
    sbm.close_scope()  # inp_load

    return output

# Hardware constants
_PMAX = 128

# Qwen3-30B-A3B at TP=4 fixed dims
_H = 2048
_E = 128
_K = 8
_I = 192
_H_FREE = _H // _PMAX

# LNC=2 H-sharding constants
_N_PRGS = 2
_H_FREE_SHARD = _H_FREE // _N_PRGS
_H_SHARD = _H_FREE_SHARD * _PMAX


def _flattened_custom_rmsnorm_to_nkilib_layout(rmsnorm_flat, dtype, T, sbm):
    """Convert custom RMSNorm [P, H_free*T] layout to nkilib [P, T, H_free]."""
    rmsnorm_out = sbm.alloc_stack(
        (_PMAX, T, _H_FREE), dtype, buffer=nl.sbuf, name="rmsnorm_out_nkilib_layout"
    )
    for h1 in nl.static_range(_H_FREE):
        nisa.tensor_copy(
            dst=rmsnorm_out[0:_PMAX, 0:T, h1],
            src=rmsnorm_flat[0:_PMAX, h1 * T : h1 * T + T],
        )
    return rmsnorm_out


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


_flash_fwd_call = nki_jit()(attention_isa_kernel)
_moe_fused_nkilib_call = nki.jit(moe_fused_tkg)


def _nki_data(tensor):
    return tensor.data if tensor is not None and hasattr(tensor, "data") else tensor


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

        # Replica groups for nccl.all_reduce inside the fused MoE TKG kernel.
        # Format mirrors qwen_fused_transformer_multilayer.py — derived from
        # tp_degree to avoid torch.distributed calls during mock tracing.
        self._replica_groups = (list(range(config.neuron_config.tp_degree)),)

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
