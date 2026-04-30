# coding=utf-8
"""Qwen3 MoE model with the local fused MoE TKG NKI kernel."""

import nki
import torch

from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeInferenceConfig


def _inject_moe_fused_nki_config_flag():
    if getattr(MoENeuronConfig, "_qwen_with_nki_moe_fused_patch", False):
        return

    original_init = MoENeuronConfig.__init__

    def patched_init(self, *args, **kwargs):
        # main.py forwards argparse defaults, so the missing backend flag arrives
        # as False. Force the repository-supported flag for both model configs.
        kwargs["moe_fused_nki_kernel_enabled"] = True
        original_init(self, *args, **kwargs)
        self.moe_fused_nki_kernel_enabled = True

    MoENeuronConfig.__init__ = patched_init
    MoENeuronConfig._qwen_with_nki_moe_fused_patch = True
    MoENeuronConfig._qwen_with_nki_original_init = original_init


_inject_moe_fused_nki_config_flag()

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

from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

# Try except for the compatibility with older compiler version
# try:
#     from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
# except ImportError:
#     from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
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
# _flash_fwd_call = nki_jit()(attention_isa_kernel)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE

_LNC = 2
_MOE_H = 2048

# MoE Kernel
import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.tensor_view import TensorView
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


_PMAX = 128
_H    = 2048
_E    = 128
_K    = 8
_H_FREE = _H // _PMAX          # 16
_N_PRGS = 2
_H_FREE_SHARD = _H_FREE // _N_PRGS  # 8
_H_SHARD = _H_FREE_SHARD * _PMAX    # 1024
_ROUTER_BATCH = 16


def _routertopk_sbuf_in_sbuf_out(
    rmsnorm_normed_bf16,    # [PMAX, H_free*T] bf16 heap in SBUF
    dtype,
    T,
    router_w,               # [H, E=128] bf16 HBM
    sbm=None,
    router_w_wide_sb=None,  # [PMAX, ROUTER_BATCH, E] bf16 SBUF pre-loaded
    debug_router_logits=None,  # [T, E] HBM, optional
):
    """
    Router matmul + Softmax + TopK(8) sub-kernel.
    rmsnorm_normed_bf16 must already be a live heap tensor in SBUF (from _rmsnorm_sbuf_in_sbuf_out_hoisted).

    Heap lifecycle (caller's responsibility):
      alloc_heap top8_logits_bf16 [T, K] bf16
      alloc_heap top8_idx         [T, K] uint32
      alloc_heap top8_vals_bf16   [T, K] bf16

    Returns (top8_logits_bf16, top8_idx, top8_vals_bf16).
    Caller pops in reverse order: top8_vals_bf16, top8_idx, top8_logits_bf16.
    rmsnorm_normed_bf16 is NOT popped here (expert stage still needs it).
    """
    H      = _H
    E      = _E
    K      = _K
    H_free = _H_FREE

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul [T, H] @ [H, E] → logits [T, E]
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((_PMAX, E), dtype=nl.float32, buffer=nl.psum)

    sbm.open_scope(name="router_softmax")

    # --- router wide DMA (skipped when router_w_wide_sb is supplied) ---
    if router_w_wide_sb is None:
        _router_w_wide_sb = sbm.alloc_stack((_PMAX, _ROUTER_BATCH, E), dtype, buffer=nl.sbuf, name="router_w_wide_sb")
    else:
        _router_w_wide_sb = router_w_wide_sb  # caller-owned — do NOT free

    for h_chunk in nl.affine_range(H_free // _ROUTER_BATCH):
        if router_w_wide_sb is None:
            for h_sub_load in nl.static_range(_ROUTER_BATCH):
                h1_load = h_chunk * _ROUTER_BATCH + h_sub_load
                shard = h1_load // _H_FREE_SHARD
                h2 = h1_load - shard * _H_FREE_SHARD
                nisa.dma_copy(
                    dst=_router_w_wide_sb[0:_PMAX, h_sub_load, 0:E],
                    src=router_w.ap(
                        pattern=[[_H_FREE_SHARD * E, _PMAX], [1, E]],
                        offset=shard * _H_SHARD * E + h2 * E,
                    ),
                    dge_mode=3,
                )
        for h_sub in nl.static_range(_ROUTER_BATCH):
            h1 = h_chunk * _ROUTER_BATCH + h_sub
            column_tile = h1 % 4
            column_offset = column_tile * 32
            nisa.nc_matmul(
                dst=logits_psum[nl.ds(column_offset, T), 0:E],
                stationary=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                moving=_router_w_wide_sb[0:_PMAX, h_sub, 0:E],
                tile_position=(0, column_offset),
                tile_size=(_PMAX, 32),
            )

    logits_sb = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="logits_sb")
    nisa.tensor_copy(dst=logits_sb[0:T, 0:E], src=logits_psum[0:T, 0:E])
    for column_tile in nl.static_range(1, 4):
        column_offset = column_tile * 32
        nisa.tensor_tensor(
            dst=logits_sb[0:T, 0:E],
            data1=logits_sb[0:T, 0:E],
            data2=logits_psum[nl.ds(column_offset, T), 0:E],
            op=nl.add,
        )
    if debug_router_logits is not None:
        nisa.dma_copy(dst=debug_router_logits[0:T, 0:E], src=logits_sb[0:T, 0:E])

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8)
    # -----------------------------------------------------------------------
    max_logit = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="max_logit")
    nisa.tensor_reduce(
        max_logit[0:T, 0:1],
        nl.maximum,
        logits_sb[0:T, 0:E],
        axis=1,
        negate=True,
        keepdims=True,
    )

    exp_vals = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="exp_vals")
    sum_exp = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="sum_exp")
    nisa.activation(
        dst=exp_vals[0:T, 0:E],
        op=nl.exp,
        data=logits_sb[0:T, 0:E],
        bias=max_logit[0:T, 0:1],
        reduce_op=nl.add,
        reduce_res=sum_exp[0:T, 0:1],
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )

    inv_sum_exp = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="inv_sum_exp")
    nisa.reciprocal(dst=inv_sum_exp[0:T, 0:1], data=sum_exp[0:T, 0:1])

    softmax_probs = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="softmax_probs")
    nisa.tensor_scalar(
        softmax_probs[0:T, 0:E], data=exp_vals[0:T, 0:E],
        op0=nl.multiply, operand0=inv_sum_exp[0:T, 0:1],
    )

    top8_logits = sbm.alloc_heap((T, K), nl.float32, buffer=nl.sbuf, name="top8_logits")
    nisa.max8(dst=top8_logits[0:T, 0:K], src=softmax_probs[0:T, 0:E])

    top8_idx = sbm.alloc_heap((T, K), nl.uint32, buffer=nl.sbuf, name="top8_idx")
    nisa.nc_find_index8(dst=top8_idx[0:T, 0:K], data=softmax_probs[0:T, 0:E], vals=top8_logits[0:T, 0:K])

    sum_topk = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="sum_topk")
    nisa.tensor_reduce(sum_topk[0:T, 0:1], nl.add, top8_logits[0:T, 0:K], axis=1, keepdims=True)
    inv_sum_topk = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="inv_sum_topk")
    nisa.reciprocal(dst=inv_sum_topk[0:T, 0:1], data=sum_topk[0:T, 0:1])

    top8_vals = sbm.alloc_heap((T, K), nl.float32, buffer=nl.sbuf, name="top8_vals")
    nisa.tensor_scalar(
        dst=top8_vals[0:T, 0:K],
        data=top8_logits[0:T, 0:K],
        op0=nl.multiply,
        operand0=inv_sum_topk[0:T, 0:1],
    )

    sbm.close_scope()  # frees router/softmax stack tensors (not router_w_wide_sb — caller-owned)

    return top8_logits, top8_idx, top8_vals


_PMAX    = 128
_H       = 2048
_E       = 128
_K       = 8
_I       = 192
_I0      = 128
_I1      = 64
_I_TILES = 2
_GU_FLAT = 2 * _I   # 384
_H_FREE  = _H // _PMAX  # 16
_H_FREE_SHARD = _H_FREE // 2
_H_SHARD = _H // 2
_GU_P    = 96
_GU_LNC_FLAT = 2 * _GU_P  # local gate + up shards packed as 192
_DOWN_P  = 96

_K_WAVE = 4  # experts per wave (two waves cover top-K=8)
_NORMALIZE_EPS_BF16 = 1.0018652574217413e-12  # bf16(1e-12) reinterpreted as fp32


def _experts_sbuf_in_sbuf_out_hshard_exact(
    rmsnorm_normed_bf16,
    top8_idx,
    top8_vals,
    dtype,
    T,
    gate_up_w,
    down_w,
    sbm=None,
    debug=False,
):
    """Baseline-compatible selective-expert path.

    nkilib's TKG selective MoE shards the hidden dimension across LNC=2 for
    gate/up, sums fp32 gate/up partials before activation, computes the down
    projection for only the local H shard, and accumulates experts sequentially
    in bf16. Keep that ordering here for bit-exactness to NxDI.
    """
    shard_id = nl.program_id(0)
    peer_lnc = 1 - shard_id
    h_free_start = shard_id * _H_FREE_SHARD
    h_hbm_start = shard_id * _H_SHARD

    sbm.open_scope(name="expert_loop_outer_exact")
    out_sb = sbm.alloc_stack((_PMAX, T, _H_FREE_SHARD), dtype, buffer=nl.sbuf, name="out_sb")
    nisa.memset(out_sb, value=0.0)

    for t in nl.static_range(T):
        sbm.open_scope(name=f"token_{t}")

        aff_bcast = sbm.alloc_stack((_PMAX, _K), nl.float32, buffer=nl.sbuf, name=f"aff_bcast_t{t}")
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:_K], src=top8_vals[t:t + 1, 0:_K])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:_K],
                src=aff_bcast[0:1, 0:_K],
                shuffle_mask=[0] * 32,
            )

        output_temp = sbm.alloc_stack(
            (_PMAX, _H_FREE_SHARD), dtype, buffer=nl.sbuf, name=f"output_temp_t{t}"
        )

        for k in nl.static_range(_K):
            sbm.open_scope(name=f"expert_{k}_t{t}")
            expert_id = top8_idx.ap(pattern=[[_K, 1], [1, 1]], offset=t * _K + k)

            gate_w0 = sbm.alloc_stack(
                (_PMAX, _H_FREE_SHARD, _I0), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_w0_k{k}_t{t}"
            )
            gate_w1 = sbm.alloc_stack(
                (_PMAX, _H_FREE_SHARD, _I0), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_w1_k{k}_t{t}"
            )
            up_w0 = sbm.alloc_stack(
                (_PMAX, _H_FREE_SHARD, _I0), gate_up_w.dtype, buffer=nl.sbuf, name=f"up_w0_k{k}_t{t}"
            )
            up_w1 = sbm.alloc_stack(
                (_PMAX, _H_FREE_SHARD, _I0), gate_up_w.dtype, buffer=nl.sbuf, name=f"up_w1_k{k}_t{t}"
            )

            h_gate_base = h_hbm_start * _GU_FLAT
            nisa.dma_copy(
                dst=gate_w0[0:_PMAX, 0:_H_FREE_SHARD, 0:_I0],
                src=gate_up_w.ap(
                    pattern=[[_H_FREE_SHARD * _GU_FLAT, _PMAX], [_GU_FLAT, _H_FREE_SHARD], [1, _I0]],
                    offset=h_gate_base,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )
            nisa.dma_copy(
                dst=gate_w1[0:_PMAX, 0:_H_FREE_SHARD, 0:_I1],
                src=gate_up_w.ap(
                    pattern=[[_H_FREE_SHARD * _GU_FLAT, _PMAX], [_GU_FLAT, _H_FREE_SHARD], [1, _I1]],
                    offset=h_gate_base + _I0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )
            nisa.dma_copy(
                dst=up_w0[0:_PMAX, 0:_H_FREE_SHARD, 0:_I0],
                src=gate_up_w.ap(
                    pattern=[[_H_FREE_SHARD * _GU_FLAT, _PMAX], [_GU_FLAT, _H_FREE_SHARD], [1, _I0]],
                    offset=h_gate_base + _I,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )
            nisa.dma_copy(
                dst=up_w1[0:_PMAX, 0:_H_FREE_SHARD, 0:_I1],
                src=gate_up_w.ap(
                    pattern=[[_H_FREE_SHARD * _GU_FLAT, _PMAX], [_GU_FLAT, _H_FREE_SHARD], [1, _I1]],
                    offset=h_gate_base + _I + _I0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            gate_psum0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.psum)
            gate_psum1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.psum)
            up_psum0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.psum)
            up_psum1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.psum)

            nisa.nc_matmul(
                dst=gate_psum0[0:_I0, 0:1],
                stationary=gate_w0[0:_PMAX, 0, 0:_I0],
                moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + 0) * T + t, 1)],
                accumulate=False,
            )
            nisa.nc_matmul(
                dst=gate_psum1[0:_I1, 0:1],
                stationary=gate_w1[0:_PMAX, 0, 0:_I1],
                moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + 0) * T + t, 1)],
                accumulate=False,
            )
            nisa.nc_matmul(
                dst=up_psum0[0:_I0, 0:1],
                stationary=up_w0[0:_PMAX, 0, 0:_I0],
                moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + 0) * T + t, 1)],
                accumulate=False,
            )
            nisa.nc_matmul(
                dst=up_psum1[0:_I1, 0:1],
                stationary=up_w1[0:_PMAX, 0, 0:_I1],
                moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + 0) * T + t, 1)],
                accumulate=False,
            )
            for h1 in nl.static_range(1, _H_FREE_SHARD):
                nisa.nc_matmul(
                    dst=gate_psum0[0:_I0, 0:1],
                    stationary=gate_w0[0:_PMAX, h1, 0:_I0],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + h1) * T + t, 1)],
                    accumulate=True,
                )
                nisa.nc_matmul(
                    dst=gate_psum1[0:_I1, 0:1],
                    stationary=gate_w1[0:_PMAX, h1, 0:_I1],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + h1) * T + t, 1)],
                    accumulate=True,
                )
                nisa.nc_matmul(
                    dst=up_psum0[0:_I0, 0:1],
                    stationary=up_w0[0:_PMAX, h1, 0:_I0],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + h1) * T + t, 1)],
                    accumulate=True,
                )
                nisa.nc_matmul(
                    dst=up_psum1[0:_I1, 0:1],
                    stationary=up_w1[0:_PMAX, h1, 0:_I1],
                    moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((h_free_start + h1) * T + t, 1)],
                    accumulate=True,
                )

            gate_fp32 = sbm.alloc_stack((_PMAX, _I_TILES), nl.float32, buffer=nl.sbuf, name=f"gate_fp32_k{k}_t{t}")
            up_fp32 = sbm.alloc_stack((_PMAX, _I_TILES), nl.float32, buffer=nl.sbuf, name=f"up_fp32_k{k}_t{t}")
            nisa.memset(gate_fp32, value=0.0)
            nisa.memset(up_fp32, value=0.0)
            nisa.activation(gate_fp32[0:_I0, 0:1], op=nl.copy, data=gate_psum0[0:_I0, 0:1])
            nisa.activation(gate_fp32[0:_I1, 1:2], op=nl.copy, data=gate_psum1[0:_I1, 0:1])
            nisa.activation(up_fp32[0:_I0, 0:1], op=nl.copy, data=up_psum0[0:_I0, 0:1])
            nisa.activation(up_fp32[0:_I1, 1:2], op=nl.copy, data=up_psum1[0:_I1, 0:1])

            gate_up_recv = sbm.alloc_stack(
                (_PMAX, _I_TILES), nl.float32, buffer=nl.sbuf, name=f"gate_up_recv_k{k}_t{t}"
            )
            nisa.sendrecv(
                src=gate_fp32,
                dst=gate_up_recv,
                send_to_rank=peer_lnc,
                recv_from_rank=peer_lnc,
                pipe_id=t * _K * 2 + k * 2,
            )
            nisa.tensor_tensor(dst=gate_fp32, data1=gate_fp32, data2=gate_up_recv, op=nl.add)
            nisa.sendrecv(
                src=up_fp32,
                dst=gate_up_recv,
                send_to_rank=peer_lnc,
                recv_from_rank=peer_lnc,
                pipe_id=t * _K * 2 + k * 2 + 1,
            )
            nisa.tensor_tensor(dst=up_fp32, data1=up_fp32, data2=gate_up_recv, op=nl.add)

            nisa.activation(dst=gate_fp32, op=nl.silu, data=gate_fp32)

            inter_bf16 = sbm.alloc_stack((_PMAX, _I_TILES), dtype, buffer=nl.sbuf, name=f"inter_bf16_k{k}_t{t}")
            nisa.memset(inter_bf16, value=0.0)
            nisa.tensor_tensor(
                dst=inter_bf16[0:_I0, 0:1],
                data1=gate_fp32[0:_I0, 0:1],
                data2=up_fp32[0:_I0, 0:1],
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=inter_bf16[0:_I1, 1:2],
                data1=gate_fp32[0:_I1, 1:2],
                data2=up_fp32[0:_I1, 1:2],
                op=nl.multiply,
            )

            down_w0_tile = sbm.alloc_stack((_PMAX, _PMAX), down_w.dtype, buffer=nl.sbuf, name=f"down_w0_tile_k{k}_t{t}")
            down_w1_tile = sbm.alloc_stack((_PMAX, _PMAX), down_w.dtype, buffer=nl.sbuf, name=f"down_w1_tile_k{k}_t{t}")
            down_psum = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.psum)
            for h1_out in nl.static_range(_H_FREE_SHARD):
                h_offset = h_hbm_start + h1_out
                nisa.dma_copy(
                    dst=down_w0_tile[0:_I0, 0:_PMAX],
                    src=down_w.ap(
                        pattern=[[_H, _I0], [_H_FREE_SHARD, _PMAX]],
                        offset=h_offset,
                        scalar_offset=expert_id,
                        indirect_dim=0,
                    ),
                    dge_mode=0,
                )
                nisa.dma_copy(
                    dst=down_w1_tile[0:_I1, 0:_PMAX],
                    src=down_w.ap(
                        pattern=[[_H, _I1], [_H_FREE_SHARD, _PMAX]],
                        offset=_I0 * _H + h_offset,
                        scalar_offset=expert_id,
                        indirect_dim=0,
                    ),
                    dge_mode=0,
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_w0_tile[0:_I0, 0:_PMAX],
                    moving=inter_bf16[0:_I0, 0:1],
                    accumulate=False,
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_w1_tile[0:_I1, 0:_PMAX],
                    moving=inter_bf16[0:_I1, 1:2],
                    accumulate=True,
                )

            down_sb = sbm.alloc_stack((_PMAX, _H_FREE_SHARD), dtype, buffer=nl.sbuf, name=f"down_sb_k{k}_t{t}")
            nisa.activation(down_sb, op=nl.copy, data=down_psum[0:_PMAX, 0:_H_FREE_SHARD])
            nisa.tensor_scalar(
                dst=down_sb,
                data=down_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k],
            )

            if k == 0:
                nisa.tensor_copy(dst=output_temp[0:_PMAX, 0:_H_FREE_SHARD], src=down_sb)
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                    data1=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                    data2=down_sb,
                    op=nl.add,
                )

            sbm.close_scope()

        for h1 in nl.static_range(_H_FREE_SHARD):
            nisa.tensor_copy(
                dst=out_sb[0:_PMAX, t, h1:h1 + 1],
                src=output_temp[0:_PMAX, h1:h1 + 1],
            )

        sbm.close_scope()

    sbm.close_scope()
    return out_sb



# Hardware constants
_PMAX = 128       # partition dimension max
_PSUM_FREE = 512  # PSUM free-dimension max on trn2

# Qwen3-30B-A3B at TP=4 fixed dims
_H = 2048    # hidden dim
_E = 128     # num experts
_K = 8       # top-K experts
_I = 192     # actual intermediate dim per TP rank
_I0 = 128    # first tile  (full 128 rows)
_I1 = 64     # second tile (partial: 64 valid rows, 64 zero-padded)
_I_TILES = 2 # two I-dimension tiles
_EPS = 1e-6

# Flat gate+up combined width
_GU_FLAT = 2 * _I   # = 384

# LNC=2 H-sharding constants
_N_PRGS = 2
_H_FREE = _H // _PMAX             # = 16
_H_FREE_SHARD = _H_FREE // _N_PRGS   # = 8
_H_SHARD = _H_FREE_SHARD * _PMAX     # = 1024

# Router DMA batching
_ROUTER_BATCH = 16

# 2-wave constants
_K_WAVE = 4  # experts per wave


def _qwen3_moe_sbuf_in_sbuf_out_hoisted(
    inp_sb,               # [PMAX=128, H_free*T] bf16 — already in SBUF (no HBM load needed)
    dtype,                # bf16 — explicit since inp.dtype is no longer available
    T,                    # int — number of tokens
    gamma,                # [1, H=2048] bf16 — HBM
    router_w,             # [H=2048, E=128] bf16 — HBM
    gate_up_w,            # [E=128, H=2048, 384] bf16 — HBM
    down_w,               # [E=128, 192, H=2048] bf16 — HBM
    sbm=None,             # required: SbufManager instance
    gamma_sb_ready=None,  # [PMAX, H_free=16] bf16 SBUF — pre-loaded & pre-transposed gamma
                          # When supplied, skip gamma_flat_sb alloc+DMA,
                          # gamma_trans_psum alloc+nc_transpose,
                          # gamma_sb alloc+activation copy.
    router_w_wide_sb=None,
    debug_rmsnorm_out=None,  # [T, H=2048] bf16 HBM — when provided, DMA rmsnorm_normed_bf16 here
):
    """
    Fused RMSNorm + Router + TopK(K=8) + Expert MLP for Qwen3 MoE TKG.
    Sub-kernel — no @nki.jit decorator. Called from inside a jitted function.
    inp_sb: [PMAX, H_free*T] bf16 already in SBUF.
    Returns: out_sb [PMAX=128, H_free*T=16*T] bf16 in SBUF — column-major, partition-first.
             caller must consume before any further sbm.alloc_stack

    New kwargs (all default None):
      gamma_sb_ready   : pre-loaded [PMAX, H_free] bf16 SBUF tensor — skips gamma DMA chain
    """
    # -----------------------------------------------------------------------
    # Stage 1: RMSNorm
    # -----------------------------------------------------------------------
    rmsnorm_normed_bf16 = _rmsnorm_sbuf_in_sbuf_out_hoisted(
        inp_sb, dtype, T, gamma, sbm=sbm,
        gamma_sb_ready=gamma_sb_ready,
        debug_rmsnorm_out=debug_rmsnorm_out,
    )
    # heap stack: [rmsnorm_normed_bf16]

    # -----------------------------------------------------------------------
    # Stage 2+3: Router matmul + Softmax + TopK(8)
    # -----------------------------------------------------------------------
    router_in = sbm.alloc_stack(
        (_PMAX, _H_FREE * T), router_w.dtype, buffer=nl.sbuf, name="router_in"
    )
    nisa.tensor_copy(dst=router_in[0:_PMAX, 0:_H_FREE * T], src=rmsnorm_normed_bf16[0:_PMAX, 0:_H_FREE * T])
    top8_logits_bf16, top8_idx, top8_vals_bf16 = _routertopk_sbuf_in_sbuf_out(
        router_in, router_w.dtype, T, router_w, sbm=sbm,
        router_w_wide_sb=router_w_wide_sb,
    )
    # heap stack: [rmsnorm_normed_bf16, top8_logits_bf16, top8_idx, top8_vals_bf16]

    # -----------------------------------------------------------------------
    # Stage 4+5: Expert MLP + output cast
    # -----------------------------------------------------------------------
    out_sb = _experts_sbuf_in_sbuf_out_hshard_exact(
        rmsnorm_normed_bf16, top8_idx, top8_vals_bf16,
        dtype, T, gate_up_w, down_w, sbm=sbm,
    )

    # Free heap in reverse order of allocation
    sbm.pop_heap()  # top8_vals_bf16
    sbm.pop_heap()  # top8_idx
    sbm.pop_heap()  # top8_logits_bf16
    sbm.pop_heap()  # rmsnorm_normed_bf16

    return out_sb  # [PMAX=128, T, H_free_shard=8] bf16 — local H shard, partition-first
                   # caller must consume before any further sbm.alloc_stack


# ---------------------------------------------------------------------------
# Raw kernel body (identical to test_v30c_vs_nxdi_trn3.py)
# ---------------------------------------------------------------------------
def moe_fused_tkg(inp, gamma, router_w, gate_up_w, down_w):
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)

    T = inp.shape[0]

    sbm.open_scope(name="inp_load")
    inp_2d = inp.reshape((T, _H))
    inp_sb = sbm.alloc_stack(
        (_PMAX, _H_FREE * T), inp.dtype, buffer=nl.sbuf, name="inp_sb"
    )
    for t in nl.static_range(T):
        for h1 in nl.static_range(_H_FREE):
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

    out_sb = _qwen3_moe_sbuf_in_sbuf_out_hoisted(
        inp_sb, inp.dtype, T,
        gamma, router_w, gate_up_w, down_w,
        sbm=sbm,
        gamma_sb_ready=None,
    )

    prg_id = nl.program_id(axis=0)
    output = nl.ndarray((T, _H), dtype=inp.dtype, buffer=nl.shared_hbm)

    sbm.open_scope(name="store_hbm")
    out_tile_sb = sbm.alloc_stack(
        (T, _H_SHARD), inp.dtype, buffer=nl.sbuf, name="out_tile_sb"
    )

    for t in nl.static_range(T):
        for h1 in nl.static_range(_H_FREE_SHARD):
            tp_psum = nl.ndarray((1, _PMAX), dtype=inp.dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=tp_psum[0:1, 0:_PMAX],
                data=out_sb[0:_PMAX, t, h1:h1 + 1],
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
            nisa.dma_copy(
                dst=output[t:t + 1, 0:_H_SHARD],
                src=out_tile_sb[t:t + 1, 0:_H_SHARD],
            )
        else:
            nisa.dma_copy(
                dst=output[t:t + 1, _H_SHARD:_H],
                src=out_tile_sb[t:t + 1, 0:_H_SHARD],
            )
    sbm.close_scope()  # store_hbm
    sbm.close_scope()  # inp_load

    return output


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
    Qwen decoder layer with the local fused MoE NKI kernel used for TKG.
    """

    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3MoEAttention(config=config)

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.mlp = initialize_moe_module(
            config=config,
            rmsnorm=self.post_attention_layernorm,
            init_tkg_module=True,
        )
        self._moe_tkg = nki.jit(moe_fused_tkg)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens
        self.config = config

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
        if self.config.neuron_config.is_prefill_stage is False:
            gamma = self.post_attention_layernorm.weight.unsqueeze(0)
            router_w = self.mlp.router.weight_T.contiguous()
            gate_up_w = self.mlp.expert_mlps.mlp_op.gate_up_proj.weight
            down_w = self.mlp.expert_mlps.mlp_op.down_proj.weight

            orig_shape = hidden_states.shape
            inp_2d = hidden_states.reshape(-1, _MOE_H)
            out_2d = self._moe_tkg[2](inp_2d, gamma, router_w, gate_up_w, down_w)
            out_2d = reduce_from_tensor_model_parallel_region(
                out_2d, process_group=parallel_state.get_world_group()
            )
            hidden_states = out_2d.reshape(orig_shape)
        else:
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
