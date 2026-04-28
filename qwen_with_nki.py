# coding=utf-8
"""Qwen3 MoE model with token generation routed through v14d Attn TKG stages.

Everything outside the customized attention path follows the installed NxDI Qwen3
MoE implementation. Context encoding uses the standard NxDI attention path; token
generation uses local NKI kernels for pre-attention RMSNorm and QKV/RoPE/KV-write,
then uses the baseline NxDI attention and output projection path.
"""

import warnings
from typing import Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch import nn

from transformers import AutoTokenizer, GenerationConfig, Qwen3MoeForCausalLM
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
)
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
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
    NeuronQwen3MoEAttention,
    Qwen3MoeInferenceConfig,
    convert_qwen3_moe_hf_to_neuron_state_dict,
    get_rmsnorm_cls,
)
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
    KV_CACHE_PAD_FOR_SEQ_IDS_MASKING,
)
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)
from nkilib.core.utils.allocator import create_auto_alloc_manager
from nkilib.core.utils.logging import Logger


V14D_PMAX = 128
V14D_EPS = 1e-6
V14D_ROPE_INV_FREQ_VALUES = (
    1, 0.80584216117858887, 0.64938163757324219, 0.52329909801483154,
    0.42169651389122009, 0.33982083201408386, 0.2738419771194458, 0.22067341208457947,
    0.17782793939113617, 0.14330126345157623, 0.11547819525003433, 0.093057207763195038,
    0.074989423155784607, 0.060429640114307404, 0.048696752637624741, 0.03924189880490303,
    0.03162277489900589, 0.025482967495918274, 0.020535251125693321, 0.016548171639442444,
    0.013335213996469975, 0.010746078565716743, 0.0086596431210637093, 0.0069783059880137444,
    0.0056234132498502731, 0.0045315837487578392, 0.0036517411936074495, 0.002942727180197835,
    0.0023713738191872835, 0.0019109529675915837, 0.0015399265103042126, 0.0012409377377480268,
    0.0010000000474974513, 0.00080584216630086303, 0.00064938160357996821, 0.00052329909522086382,
    0.0004216965171508491, 0.00033982083550654352, 0.00027384195709601045, 0.00022067340614739805,
    0.00017782794020604342, 0.00014330125122796744, 0.00011547820031410083, 9.305720595875755e-05,
    7.4989424319937825e-05, 6.0429640143411234e-05, 4.8696751036914065e-05, 3.92418987757992e-05,
    3.1622777896700427e-05, 2.5482968339929357e-05, 2.0535249859676696e-05, 1.6548170606256463e-05,
    1.3335214134713169e-05, 1.0746078260126524e-05, 8.6596428445773199e-06, 6.978305918892147e-06,
    5.6234134717669804e-06, 4.5315837269299664e-06, 3.6517412809189409e-06, 2.9427271783788456e-06,
    2.3713737391517498e-06, 1.9109529603156261e-06, 1.5399265294036013e-06, 1.2409377632138785e-06,
)

PMAX = V14D_PMAX
DEBUG_RETURN_QSCORES = False


def _attn_rmsnorm_sbuf_in_sbuf_out(
    hidden_sb,
    gpan_f32,
    rms_zero_bias,
    rms_eps_sb,
    H,
    num_h_tiles,
    sbm=None,
):
    """Pre-attention RMSNorm subkernel operating entirely on SBUF tensors."""
    assert sbm is not None, "sbm (SbufManager) is required"

    h_all = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="h_all")

    sbm.open_scope("pre_attn_norm")

    h_f32 = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_f32")
    nisa.tensor_copy(h_f32, hidden_sb)

    h_sq = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_sq")
    nisa.activation(h_sq, op=nl.square, data=h_f32, bias=rms_zero_bias)

    h_sq_T_psum = nl.ndarray((num_h_tiles, PMAX), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_sq_T_psum, h_sq)
    h_sq_T_sb = sbm.alloc_stack((num_h_tiles, PMAX), nl.float32, name="h_sq_T_sb")
    nisa.tensor_copy(h_sq_T_sb, h_sq_T_psum)
    h_sq_sum_tiles = sbm.alloc_stack((num_h_tiles, 1), nl.float32, name="h_sq_sum_tiles")
    nisa.tensor_reduce(h_sq_sum_tiles, op=nl.add, data=h_sq_T_sb, axis=1)
    h_sq_sum_tiles_T_psum = nl.ndarray((1, num_h_tiles), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_sq_sum_tiles_T_psum, h_sq_sum_tiles)
    h_sq_sum_tiles_T_sb = sbm.alloc_stack((1, num_h_tiles), nl.float32, name="h_sq_sum_tiles_T_sb")
    nisa.tensor_copy(h_sq_sum_tiles_T_sb, h_sq_sum_tiles_T_psum)
    h_sq_scalar = sbm.alloc_stack((1, 1), nl.float32, name="h_sq_scalar")
    nisa.tensor_reduce(h_sq_scalar, op=nl.add, data=h_sq_sum_tiles_T_sb, axis=1)
    h_sq_total_psum = nl.ndarray((PMAX, 1), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_sq_total_psum, h_sq_scalar.ap([[1, 1], [0, PMAX]], offset=0))
    h_sq_total = sbm.alloc_stack((PMAX, 1), nl.float32, name="h_sq_total")
    nisa.tensor_copy(h_sq_total, h_sq_total_psum)

    h_rms_inv = sbm.alloc_stack((PMAX, 1), nl.float32, name="h_rms_inv")
    nisa.activation(h_rms_inv, op=nl.rsqrt, data=h_sq_total, scale=1.0/H, bias=rms_eps_sb)

    h_rms_T_psum = nl.ndarray((num_h_tiles, PMAX), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_rms_T_psum, h_rms_inv.ap([[1, PMAX], [0, num_h_tiles]], offset=0))
    h_rms_T = sbm.alloc_stack((num_h_tiles, PMAX), nl.float32, name="h_rms_T")
    nisa.tensor_copy(h_rms_T, h_rms_T_psum)
    h_rms_expanded_psum = nl.ndarray((PMAX, num_h_tiles), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(h_rms_expanded_psum, h_rms_T)
    h_rms_expanded = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_rms_expanded")
    nisa.tensor_copy(h_rms_expanded, h_rms_expanded_psum)

    h_normed = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="h_normed")
    nisa.tensor_tensor(h_normed, h_f32, h_rms_expanded, op=nl.multiply)
    nisa.tensor_tensor(h_normed, h_normed, gpan_f32, op=nl.multiply)
    nisa.tensor_copy(h_all, h_normed)

    sbm.close_scope()

    return h_all


def _attn_qkv_rope_kvwrite_sbuf_in_sbuf_out(
    h_all,
    wk_sb,
    wv_sb,
    wq_head_sb,
    qnw_sb,
    knw_sb,
    cos_f32,
    sin_f32,
    pos_write_i32,
    K_cache,
    V_cache,
    rms_zero_bias,
    rms_ones,
    rms_eps_sb,
    H,
    Hq_tp,
    S_prior,
    owned_heads,
    sbm=None,
):
    """QKV projection, Q/K RMSNorm, RoPE, and in-place active K/V cache write."""
    assert sbm is not None, "sbm (SbufManager) is required"

    B = cos_f32.shape[1]
    d = PMAX
    GQA = Hq_tp
    half_d = d // 2
    NUM_H_TILES = H // PMAX

    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)

    k_rope = sbm.alloc_stack((PMAX, B), nl.float32, name="k_rope")
    k_rope_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_rope_bf16")
    k_rope_T_sb = sbm.alloc_stack((B, PMAX), nl.bfloat16, name="k_rope_T_sb")
    v_active = sbm.alloc_stack((PMAX, B), nl.float32, name="v_active")
    v_T_sb = sbm.alloc_stack((B, PMAX), nl.bfloat16, name="v_T_sb")
    q_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_bf16")

    k_rope_bf16_f32 = sbm.alloc_stack((PMAX, B), nl.float32, name="k_rope_bf16_f32")
    v_from_bf16 = sbm.alloc_stack((PMAX, B), nl.float32, name="v_from_bf16")
    if DEBUG_RETURN_QSCORES:
        q_pre_rope_dbg = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_pre_rope_dbg")
        q_raw_dbg = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_raw_dbg")

    sbm.open_scope("kv_proj")

    k_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum)
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(
            k_psum,
            stationary=wk_sb[0:PMAX, h_t * d:(h_t + 1) * d],
            moving=h_all[0:PMAX, h_t:h_t + 1],
        )

    k_vec = sbm.alloc_stack((PMAX, B), nl.float32, name="k_vec")
    nisa.tensor_copy(k_vec, k_psum)

    k_sq = sbm.alloc_stack((PMAX, B), nl.float32, name="k_sq")
    nisa.activation(k_sq, op=nl.square, data=k_vec, bias=rms_zero_bias)
    k_sum_psum = nl.ndarray((PMAX, B), nl.float32, buffer=nl.psum)
    nisa.nc_matmul(k_sum_psum, stationary=rms_ones, moving=k_sq)
    k_sum_sb = sbm.alloc_stack((PMAX, 1), nl.float32, name="k_sum_sb")
    nisa.tensor_copy(k_sum_sb, k_sum_psum)
    k_rms_inv = sbm.alloc_stack((PMAX, 1), nl.float32, name="k_rms_inv")
    nisa.activation(k_rms_inv, op=nl.rsqrt, data=k_sum_sb, scale=1.0/d, bias=rms_eps_sb)
    k_normed = sbm.alloc_stack((PMAX, B), nl.float32, name="k_normed")
    nisa.tensor_tensor(k_normed, k_vec, k_rms_inv, op=nl.multiply)
    k_normed2 = sbm.alloc_stack((PMAX, B), nl.float32, name="k_normed2")
    nisa.tensor_tensor(k_normed2, k_normed, knw_sb, op=nl.multiply)

    k_normed2_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_normed2_bf16")
    nisa.tensor_copy(k_normed2_bf16, k_normed2)
    nisa.tensor_copy(k_normed2, k_normed2_bf16)

    rot_k = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="rot_k")
    neg_k_upper = sbm.alloc_stack((half_d, B), nl.bfloat16, name="neg_k_upper")
    nisa.tensor_scalar(neg_k_upper, k_normed2_bf16[half_d:d, 0:B], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(rot_k[0:half_d, 0:B], neg_k_upper)
    nisa.tensor_copy(rot_k[half_d:d, 0:B], k_normed2_bf16[0:half_d, 0:B])
    k_cos = sbm.alloc_stack((PMAX, B), nl.float32, name="k_cos")
    k_sin_part = sbm.alloc_stack((PMAX, B), nl.float32, name="k_sin_part")
    nisa.tensor_tensor(k_cos, k_normed2_bf16, cos_f32, op=nl.multiply)
    nisa.tensor_tensor(k_sin_part, rot_k, sin_f32, op=nl.multiply)
    k_rope_f32 = sbm.alloc_stack((PMAX, B), nl.float32, name="k_rope_f32")
    nisa.tensor_tensor(k_rope_f32, k_cos, k_sin_part, op=nl.add)
    nisa.tensor_copy(k_rope_bf16, k_rope_f32)
    nisa.tensor_copy(k_rope, k_rope_bf16)
    k_rope_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(k_rope_T_psum, k_rope_bf16)
    nisa.tensor_copy(k_rope_T_sb, k_rope_T_psum)
    nisa.tensor_copy(k_rope_bf16_f32, k_rope_bf16)

    v_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum)
    if n_prgs == 1:
        for h_t in nl.affine_range(NUM_H_TILES):
            nisa.nc_matmul(
                v_psum,
                stationary=wv_sb[0:PMAX, h_t * d:(h_t + 1) * d],
                moving=h_all[0:PMAX, h_t:h_t + 1],
            )
        nisa.tensor_copy(v_active, v_psum)
    else:
        for h_t_local in nl.affine_range(NUM_H_TILES // 2):
            if prg_id == 0:
                h_t = h_t_local
            else:
                h_t = h_t_local + NUM_H_TILES // 2
            nisa.nc_matmul(
                v_psum,
                stationary=wv_sb[0:PMAX, h_t * d:(h_t + 1) * d],
                moving=h_all[0:PMAX, h_t:h_t + 1],
            )

        v_local = sbm.alloc_stack((PMAX, B), nl.float32, name="v_local")
        v_peer = sbm.alloc_stack((PMAX, B), nl.float32, name="v_peer")
        nisa.tensor_copy(v_local, v_psum)
        nisa.sendrecv(src=v_local, dst=v_peer, send_to_rank=1 - prg_id, recv_from_rank=1 - prg_id, pipe_id=0)
        if prg_id == 0:
            nisa.tensor_tensor(v_active, v_peer, v_local, op=nl.add)
        else:
            nisa.tensor_tensor(v_active, v_local, v_peer, op=nl.add)

    v_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="v_bf16")
    nisa.tensor_copy(v_bf16, v_active)
    v_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(v_T_psum, v_bf16)
    nisa.tensor_copy(v_T_sb, v_T_psum)
    nisa.tensor_copy(v_from_bf16, v_bf16)

    sbm.close_scope()

    sbm.open_scope("q_proj")

    q_psums = []
    for i in range(len(owned_heads)):
        q_p = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(q_p, value=0.0)
        q_psums.append(q_p)

    for h_t in nl.affine_range(NUM_H_TILES):
        for i in range(len(owned_heads)):
            q_h = owned_heads[i]
            nisa.nc_matmul(
                q_psums[i],
                stationary=wq_head_sb[q_h][0:PMAX, h_t * d:(h_t + 1) * d],
                moving=h_all[0:PMAX, h_t:h_t + 1],
            )

    q_packed_f32 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_packed_f32")
    nisa.memset(q_packed_f32, value=0.0)
    for i in range(len(owned_heads)):
        q_h = owned_heads[i]
        nisa.tensor_copy(q_packed_f32[0:PMAX, q_h:q_h + 1], q_psums[i])
    if DEBUG_RETURN_QSCORES:
        nisa.tensor_copy(q_raw_dbg, q_packed_f32)

    qnw_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(qnw_gqa_psum_T, qnw_sb.ap([[1, PMAX], [0, GQA]], offset=0))
    qnw_gqa_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="qnw_gqa_sbuf_T")
    nisa.tensor_copy(qnw_gqa_sbuf_T, qnw_gqa_psum_T)
    qnw_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(qnw_gqa_psum, qnw_gqa_sbuf_T)
    qnw_gqa = sbm.alloc_stack((PMAX, GQA), nl.float32, name="qnw_gqa")
    nisa.tensor_copy(qnw_gqa, qnw_gqa_psum)

    q_sq = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_sq")
    nisa.activation(q_sq, op=nl.square, data=q_packed_f32, bias=rms_zero_bias)
    q_sum_psum = nl.ndarray((PMAX, GQA), nl.float32, buffer=nl.psum)
    nisa.nc_matmul(q_sum_psum, stationary=rms_ones, moving=q_sq)
    q_sum_sb = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_sum_sb")
    nisa.tensor_copy(q_sum_sb, q_sum_psum)
    q_rms_inv = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_rms_inv")
    nisa.activation(q_rms_inv, op=nl.rsqrt, data=q_sum_sb, scale=1.0/d, bias=rms_eps_sb)
    q_normed = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_normed")
    nisa.tensor_tensor(q_normed, q_packed_f32, q_rms_inv, op=nl.multiply)
    q_normed2 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_normed2")
    nisa.tensor_tensor(q_normed2, q_normed, qnw_gqa, op=nl.multiply)

    q_normed2_bf16 = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_normed2_bf16")
    nisa.tensor_copy(q_normed2_bf16, q_normed2)
    nisa.tensor_copy(q_normed2, q_normed2_bf16)
    if DEBUG_RETURN_QSCORES:
        nisa.tensor_copy(q_pre_rope_dbg, q_normed2_bf16)

    cos_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(cos_gqa_psum_T, cos_f32.ap([[1, PMAX], [0, GQA]], offset=0))
    cos_gqa_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="cos_gqa_sbuf_T")
    nisa.tensor_copy(cos_gqa_sbuf_T, cos_gqa_psum_T)
    cos_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(cos_gqa_psum, cos_gqa_sbuf_T)
    cos_gqa_f32 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="cos_gqa_f32")
    nisa.tensor_copy(cos_gqa_f32, cos_gqa_psum)

    sin_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(sin_gqa_psum_T, sin_f32.ap([[1, PMAX], [0, GQA]], offset=0))
    sin_gqa_sbuf_T = sbm.alloc_stack((GQA, PMAX), nl.float32, name="sin_gqa_sbuf_T")
    nisa.tensor_copy(sin_gqa_sbuf_T, sin_gqa_psum_T)
    sin_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(sin_gqa_psum, sin_gqa_sbuf_T)
    sin_gqa_f32 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="sin_gqa_f32")
    nisa.tensor_copy(sin_gqa_f32, sin_gqa_psum)

    rot_q = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="rot_q")
    neg_q_upper = sbm.alloc_stack((half_d, GQA), nl.bfloat16, name="neg_q_upper")
    nisa.tensor_scalar(neg_q_upper, q_normed2_bf16[half_d:d, 0:GQA], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(rot_q[0:half_d, 0:GQA], neg_q_upper)
    nisa.tensor_copy(rot_q[half_d:d, 0:GQA], q_normed2_bf16[0:half_d, 0:GQA])

    q_cos = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_cos")
    q_sin_part = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_sin_part")
    nisa.tensor_tensor(q_cos, q_normed2_bf16, cos_gqa_f32, op=nl.multiply)
    nisa.tensor_tensor(q_sin_part, rot_q, sin_gqa_f32, op=nl.multiply)
    q_rope_f32 = sbm.alloc_stack((PMAX, GQA), nl.float32, name="q_rope_f32")
    nisa.tensor_tensor(q_rope_f32, q_cos, q_sin_part, op=nl.add)
    q_rope = sbm.alloc_stack((PMAX, GQA), nl.bfloat16, name="q_rope")
    nisa.tensor_copy(q_rope, q_rope_f32)

    nisa.tensor_copy(q_bf16, q_rope)

    sbm.close_scope()

    if n_prgs == 1 or prg_id == 0:
        nisa.dma_copy(
            dst=V_cache.reshape((B * S_prior, d)).ap(
                pattern=[[d, 1], [1, d]], offset=0,
                scalar_offset=pos_write_i32, indirect_dim=0,
            ),
            src=v_T_sb,
        )
    if n_prgs == 1 or prg_id == 1:
        nisa.dma_copy(
            dst=K_cache.reshape((B * S_prior, d)).ap(
                pattern=[[d, 1], [1, d]], offset=0,
                scalar_offset=pos_write_i32, indirect_dim=0,
            ),
            src=k_rope_T_sb,
        )

    return q_bf16, k_rope_bf16, v_active


class Qwen3V14DAttnNeuronConfig(MoENeuronConfig):
    """Neuron config for the direct v14d token-generation attention path."""

    def __init__(self, **kwargs):
        # The local kernel updates K/V in place, so NxDI must skip its own
        # Python-side KV cache update path.
        kwargs["attn_tkg_nki_kernel_enabled"] = True
        kwargs["attn_block_tkg_nki_kernel_cache_update"] = True
        kwargs.setdefault("fused_qkv", False)

        # Force the same leaderboard-friendly runtime mode used by the
        # multilayer submission: sample on device and run the model async.
        tensor_capture_config = kwargs.pop("tensor_capture_config", None)
        on_device_sampling_config = kwargs.pop("on_device_sampling_config", None)
        if on_device_sampling_config is None:
            on_device_sampling_config = OnDeviceSamplingConfig(**kwargs)
        kwargs["on_device_sampling_config"] = on_device_sampling_config
        if tensor_capture_config is not None:
            kwargs["tensor_capture_config"] = tensor_capture_config
        kwargs["output_logits"] = True
        kwargs["async_mode"] = True
        super().__init__(**kwargs)


def _v14d_load_bsh_hidden_to_sbuf(hidden_states, hidden_size, batch, sbm):
    hidden_tiles = hidden_size // V14D_PMAX
    hidden_sb = sbm.alloc_stack(
        (V14D_PMAX, hidden_tiles),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name="hidden_sb",
    )
    nisa.dma_copy(
        dst=hidden_sb,
        src=hidden_states.reshape((hidden_size, batch)).ap(
            pattern=[[1, V14D_PMAX], [V14D_PMAX, hidden_tiles]],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )
    return hidden_sb


def _v14d_store_sbuf_to_bsh(output, out_sb, hidden_size, batch, sbm):
    out_tiles = hidden_size // V14D_PMAX
    out_flat = output.reshape((batch, hidden_size))
    for tile_idx in nl.affine_range(out_tiles):
        out_tile_psum = nl.ndarray((1, V14D_PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(out_tile_psum, out_sb[0:V14D_PMAX, tile_idx:tile_idx + 1])
        out_tile = sbm.alloc_stack(
            (1, V14D_PMAX),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
            name=f"out_tile_{tile_idx}",
        )
        nisa.tensor_copy(out_tile, out_tile_psum)
        nisa.dma_copy(
            dst=out_flat[0:batch, tile_idx * V14D_PMAX:(tile_idx + 1) * V14D_PMAX],
            src=out_tile,
            dge_mode=nisa.dge_mode.hwdge,
        )


def _v14d_load_gamma_f32(gamma, hidden_size, sbm):
    hidden_tiles = hidden_size // V14D_PMAX
    gamma_bf16 = sbm.alloc_stack(
        (V14D_PMAX, hidden_tiles),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name="gamma_bf16",
    )
    nisa.dma_copy(
        dst=gamma_bf16,
        src=gamma.reshape((hidden_size, 1)).ap(
            pattern=[[1, V14D_PMAX], [V14D_PMAX, hidden_tiles]],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )
    gamma_f32 = sbm.alloc_stack(
        (V14D_PMAX, hidden_tiles),
        dtype=nl.float32,
        buffer=nl.sbuf,
        name="gamma_f32",
    )
    nisa.tensor_copy(gamma_f32, gamma_bf16)
    return gamma_f32


def _v14d_load_vector_weight_f32(weight, name, sbm):
    weight_bf16 = sbm.alloc_stack(
        (V14D_PMAX, 1),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name=f"{name}_bf16",
    )
    nisa.dma_copy(
        dst=weight_bf16,
        src=weight.reshape((V14D_PMAX, 1)),
        dge_mode=nisa.dge_mode.hwdge,
    )
    weight_f32 = sbm.alloc_stack(
        (V14D_PMAX, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
        name=f"{name}_f32",
    )
    nisa.tensor_copy(weight_f32, weight_bf16)
    return weight_f32


def _v14d_alloc_rms_constants(sbm):
    rms_zero_bias = sbm.alloc_stack((V14D_PMAX, 1), nl.float32, buffer=nl.sbuf, name="rms_zero_bias")
    nisa.memset(rms_zero_bias, value=0.0)
    rms_ones = sbm.alloc_stack((V14D_PMAX, V14D_PMAX), nl.float32, buffer=nl.sbuf, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)
    rms_eps_sb = sbm.alloc_stack((V14D_PMAX, 1), nl.float32, buffer=nl.sbuf, name="rms_eps_sb")
    nisa.memset(rms_eps_sb, value=V14D_EPS)
    return rms_zero_bias, rms_ones, rms_eps_sb


def _v14d_load_position(position_ids, batch, sbm, name_prefix="pos"):
    pos_write_i32_raw = sbm.alloc_stack(
        (1, 1),
        nl.int32,
        buffer=nl.sbuf,
        name=f"{name_prefix}_write_i32_raw",
    )
    nisa.dma_copy(
        dst=pos_write_i32_raw,
        src=position_ids.reshape((batch, 1))[0:1, 0:1],
        dge_mode=nisa.dge_mode.hwdge,
    )
    pos_write_i32 = sbm.alloc_stack(
        (1, 1),
        nl.uint32,
        buffer=nl.sbuf,
        name=f"{name_prefix}_write_i32",
    )
    nisa.tensor_copy(pos_write_i32, pos_write_i32_raw)
    return pos_write_i32


def _v14d_reconstruct_rope(position_ids, batch, sbm):
    half_d = V14D_PMAX // 2
    pos_write_i32 = _v14d_load_position(position_ids, batch, sbm, name_prefix="rope_pos")
    rope_pos_scalar = sbm.alloc_stack((1, 1), nl.float32, buffer=nl.sbuf, name="rope_pos_scalar")
    nisa.tensor_copy(rope_pos_scalar, pos_write_i32)
    rope_pos_psum = nl.ndarray((V14D_PMAX, 1), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(rope_pos_psum, rope_pos_scalar.ap([[1, 1], [0, V14D_PMAX]], offset=0))
    rope_pos = sbm.alloc_stack((V14D_PMAX, 1), nl.float32, buffer=nl.sbuf, name="rope_pos")
    nisa.tensor_copy(rope_pos, rope_pos_psum)

    rope_inv_freq_T = sbm.alloc_stack((1, V14D_PMAX), nl.float32, buffer=nl.sbuf, name="rope_inv_freq_T")
    for rope_i in range(half_d):
        rope_value = V14D_ROPE_INV_FREQ_VALUES[rope_i]
        nisa.memset(rope_inv_freq_T[0:1, rope_i:rope_i + 1], value=rope_value)
        nisa.memset(rope_inv_freq_T[0:1, rope_i + half_d:rope_i + half_d + 1], value=rope_value)
    rope_inv_freq_psum = nl.ndarray((V14D_PMAX, 1), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(rope_inv_freq_psum, rope_inv_freq_T)
    rope_inv_freq = sbm.alloc_stack((V14D_PMAX, 1), nl.float32, buffer=nl.sbuf, name="rope_inv_freq")
    nisa.tensor_copy(rope_inv_freq, rope_inv_freq_psum)
    rope_angle = sbm.alloc_stack((V14D_PMAX, 1), nl.float32, buffer=nl.sbuf, name="rope_angle")
    nisa.tensor_tensor(rope_angle, rope_inv_freq, rope_pos, op=nl.multiply)
    return pos_write_i32, nl.cos(rope_angle), nl.sin(rope_angle)


def _v14d_owned_heads():
    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)
    if n_prgs == 1:
        return [0, 1, 2, 3, 4, 5, 6, 7]
    if prg_id == 0:
        return [0, 1, 2, 3]
    return [4, 5, 6, 7]


@nki.jit
def _qwen3_tkg_v14d_rmsnorm_kernel(
    hidden_states: nl.ndarray,
    gamma_pre_attn: nl.ndarray,
) -> nl.ndarray:
    """HBM-facing wrapper around the SBUF pre-attention RMSNorm subkernel."""

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[2]
    hidden_tiles = hidden_size // V14D_PMAX

    sbm = create_auto_alloc_manager(logger=Logger("qwen-attn-tkg-v14d-rmsnorm"))
    sbm.open_scope(name="qwen_attn_tkg_v14d_rmsnorm_wrapper")

    hidden_sb = _v14d_load_bsh_hidden_to_sbuf(hidden_states, hidden_size, batch, sbm)
    gamma_f32 = _v14d_load_gamma_f32(gamma_pre_attn, hidden_size, sbm)
    rms_zero_bias, _, rms_eps_sb = _v14d_alloc_rms_constants(sbm)
    out_sb = _attn_rmsnorm_sbuf_in_sbuf_out(
        hidden_sb,
        gamma_f32,
        rms_zero_bias,
        rms_eps_sb,
        hidden_size,
        hidden_tiles,
        sbm=sbm,
    )

    out = nl.ndarray(
        (batch, 1, hidden_size),
        dtype=hidden_states.dtype,
        buffer=nl.shared_hbm,
        name="qwen_attn_tkg_v14d_normed",
    )
    _v14d_store_sbuf_to_bsh(out, out_sb, hidden_size, batch, sbm)

    sbm.close_scope()
    return out


@nki.jit
def _qwen3_tkg_v14d_qkv_rope_kernel(
    normed_states: nl.ndarray,
    Wq: nl.ndarray,
    Wk: nl.ndarray,
    Wv: nl.ndarray,
    q_norm_weight: nl.ndarray,
    k_norm_weight: nl.ndarray,
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    position_ids: nl.ndarray,
    rotary_position_ids: nl.ndarray,
) -> tuple[nl.ndarray, nl.ndarray, nl.ndarray, nl.ndarray, nl.ndarray]:
    """HBM-facing wrapper around the QKV/RoPE/KV-write subkernel."""

    batch = normed_states.shape[0]
    hidden_size = Wq.shape[1]
    hidden_tiles = hidden_size // V14D_PMAX
    hq_tp = Wq.shape[0] // V14D_PMAX
    seq_len = K_cache.shape[2]

    sbm = create_auto_alloc_manager(logger=Logger("qwen-attn-tkg-v14d-qkv-rope"))
    sbm.open_scope(name="qwen_attn_tkg_v14d_qkv_rope_wrapper")

    h_all = _v14d_load_bsh_hidden_to_sbuf(normed_states, hidden_size, batch, sbm)
    qnw_f32 = _v14d_load_vector_weight_f32(q_norm_weight, "qnw", sbm)
    knw_f32 = _v14d_load_vector_weight_f32(k_norm_weight, "knw", sbm)
    pos_write_i32 = _v14d_load_position(position_ids, batch, sbm, name_prefix="cache_pos")
    _, cos_f32, sin_f32 = _v14d_reconstruct_rope(rotary_position_ids, batch, sbm)
    rms_zero_bias, rms_ones, rms_eps_sb = _v14d_alloc_rms_constants(sbm)

    wk_sb = sbm.alloc_stack(
        (V14D_PMAX, hidden_tiles * V14D_PMAX),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name="wk_sb",
    )
    nisa.dma_copy(dst=wk_sb, src=Wk, dge_mode=nisa.dge_mode.hwdge)
    wv_sb = sbm.alloc_stack(
        (V14D_PMAX, hidden_tiles * V14D_PMAX),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name="wv_sb",
    )
    nisa.dma_copy(dst=wv_sb, src=Wv, dge_mode=nisa.dge_mode.hwdge)

    owned_heads = _v14d_owned_heads()
    wq_head_sb = [None] * 8
    for q_h in owned_heads:
        wq_tile = sbm.alloc_stack(
            (V14D_PMAX, hidden_tiles * V14D_PMAX),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
            name=f"wq_head_{q_h}",
        )
        nisa.dma_copy(
            dst=wq_tile,
            src=Wq[q_h * V14D_PMAX:(q_h + 1) * V14D_PMAX, :],
            dge_mode=nisa.dge_mode.hwdge,
        )
        wq_head_sb[q_h] = wq_tile

    q_bf16, k_rope_bf16, v_active = _attn_qkv_rope_kvwrite_sbuf_in_sbuf_out(
        h_all,
        wk_sb,
        wv_sb,
        wq_head_sb,
        qnw_f32,
        knw_f32,
        cos_f32,
        sin_f32,
        pos_write_i32,
        K_cache,
        V_cache,
        rms_zero_bias,
        rms_ones,
        rms_eps_sb,
        hidden_size,
        hq_tp,
        seq_len,
        owned_heads,
        sbm=sbm,
    )

    q_output = nl.ndarray(
        (batch, hq_tp, 1, V14D_PMAX),
        dtype=normed_states.dtype,
        buffer=nl.shared_hbm,
        name="qwen_attn_tkg_v14d_Q",
    )
    k_output = nl.ndarray(
        (batch, 1, 1, V14D_PMAX),
        dtype=normed_states.dtype,
        buffer=nl.shared_hbm,
        name="qwen_attn_tkg_v14d_K",
    )
    v_output = nl.ndarray(
        (batch, 1, 1, V14D_PMAX),
        dtype=normed_states.dtype,
        buffer=nl.shared_hbm,
        name="qwen_attn_tkg_v14d_V",
    )

    q_out_2d = q_output.reshape((hq_tp, V14D_PMAX))
    for head in owned_heads:
        q_head_psum = nl.ndarray((1, V14D_PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(q_head_psum, q_bf16[0:V14D_PMAX, head:head + 1])
        q_head_sb = sbm.alloc_stack((1, V14D_PMAX), nl.bfloat16, buffer=nl.sbuf, name=f"q_head_out_{head}")
        nisa.tensor_copy(q_head_sb, q_head_psum)
        nisa.dma_copy(dst=q_out_2d[head:head + 1, 0:V14D_PMAX], src=q_head_sb, dge_mode=nisa.dge_mode.hwdge)

    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)
    if n_prgs == 1 or prg_id == 1:
        k_psum = nl.ndarray((1, V14D_PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(k_psum, k_rope_bf16)
        k_sb = sbm.alloc_stack((1, V14D_PMAX), nl.bfloat16, buffer=nl.sbuf, name="k_out_sb")
        nisa.tensor_copy(k_sb, k_psum)
        nisa.dma_copy(dst=k_output.reshape((1, V14D_PMAX)), src=k_sb, dge_mode=nisa.dge_mode.hwdge)
    if n_prgs == 1 or prg_id == 0:
        v_bf16 = sbm.alloc_stack((V14D_PMAX, batch), nl.bfloat16, buffer=nl.sbuf, name="v_out_bf16")
        nisa.tensor_copy(v_bf16, v_active)
        v_psum = nl.ndarray((1, V14D_PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(v_psum, v_bf16)
        v_sb = sbm.alloc_stack((1, V14D_PMAX), nl.bfloat16, buffer=nl.sbuf, name="v_out_sb")
        nisa.tensor_copy(v_sb, v_psum)
        nisa.dma_copy(dst=v_output.reshape((1, V14D_PMAX)), src=v_sb, dge_mode=nisa.dge_mode.hwdge)

    sbm.close_scope()
    return q_output, k_output, v_output, K_cache, V_cache


def _tile_transpose_v14d_weight(attn, weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    return (
        weight.reshape(
            num_heads,
            attn.head_dim,
            attn.hidden_size // attn.head_dim,
            attn.head_dim,
        )
        .permute(0, 3, 2, 1)
        .reshape(num_heads * attn.head_dim, attn.hidden_size)
        .contiguous()
    )


def _get_v14d_qkv_weights(attn):
    qkv_proj = attn.get_qkv_proj()
    q_size = attn.num_heads * attn.head_dim
    kv_size = attn.num_key_value_heads * attn.head_dim

    if attn.fused_qkv:
        qkv_weight = qkv_proj.Wqkv.weight.data
        if qkv_weight.shape[0] == attn.hidden_size:
            qkv_weight = qkv_weight.transpose(0, 1)
        Wq, Wk, Wv = torch.tensor_split(qkv_weight, (q_size, q_size + kv_size), dim=0)
    else:
        Wq = qkv_proj.q_proj.weight.data
        Wk = qkv_proj.k_proj.weight.data
        Wv = qkv_proj.v_proj.weight.data
        if Wq.shape[0] == attn.hidden_size:
            Wq = Wq.transpose(0, 1)
            Wk = Wk.transpose(0, 1)
            Wv = Wv.transpose(0, 1)

    return (
        _tile_transpose_v14d_weight(attn, Wq, attn.num_heads),
        _tile_transpose_v14d_weight(attn, Wk, attn.num_key_value_heads),
        _tile_transpose_v14d_weight(attn, Wv, attn.num_key_value_heads),
    )


def _get_v14d_rope(attn, hidden_states, rotary_position_ids, cos_cache, sin_cache):
    if cos_cache is None or sin_cache is None or cos_cache.shape[-1] != attn.head_dim:
        cos_cache, sin_cache = attn.rotary_emb(hidden_states, rotary_position_ids)

    return (
        cos_cache.reshape(hidden_states.shape[0], attn.head_dim),
        sin_cache.reshape(hidden_states.shape[0], attn.head_dim),
        cos_cache,
        sin_cache,
    )


def _slice_v14d_cache_for_baseline_attention(attn, K_cache, V_cache, attention_mask):
    """Return the logical KV slice expected by baseline compute_for_token_gen."""

    s_prior = attention_mask.shape[-1]
    k_seq_dim = 3 if attn.k_cache_transposed else 2
    v_seq_dim = 2
    k_cache_len = K_cache.shape[k_seq_dim]
    v_cache_len = V_cache.shape[v_seq_dim]

    if k_cache_len == s_prior and v_cache_len == s_prior:
        return K_cache, V_cache

    if attn.neuron_config.padding_side == "right":
        K_attn = torch.ops.aten.slice(K_cache, dim=k_seq_dim, start=0, end=s_prior)
        V_attn = torch.ops.aten.slice(V_cache, dim=v_seq_dim, start=0, end=s_prior)
    else:
        K_attn = torch.ops.aten.slice(K_cache, dim=k_seq_dim, start=k_cache_len - s_prior, end=k_cache_len)
        V_attn = torch.ops.aten.slice(V_cache, dim=v_seq_dim, start=v_cache_len - s_prior, end=v_cache_len)

    return K_attn, V_attn


def _attention_tkg_v14d(
    attn,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_value: Tuple[torch.Tensor],
    rmsnorm: nn.Module,
    active_mask: Optional[torch.Tensor] = None,
    adapter_ids=None,
    cos_cache: Optional[torch.Tensor] = None,
    sin_cache: Optional[torch.Tensor] = None,
    rotary_position_ids: Optional[torch.LongTensor] = None,
    active_block_table: Optional[torch.Tensor] = None,
    use_polar_compatible_rope: bool = False,
):
    if attn.sequence_parallel_enabled and attn.tensor_model_parallel_group is not None:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states,
            attn.sequence_dimension,
            process_group=attn.tensor_model_parallel_group,
        )

    bsz, s_tkg, hidden_size = hidden_states.shape
    assert bsz == 1, "v14d token-generation attention currently expects batch size 1"
    assert s_tkg == 1, "v14d token-generation attention currently expects one token"
    assert hidden_size == attn.hidden_size, "v14d token-generation attention expects unpadded hidden states"
    assert attn.tp_degree == 4, "v14d token-generation attention expects TP=4"
    assert attn.hidden_size == 16 * V14D_PMAX, "v14d token-generation attention expects hidden_size=2048"
    assert attn.head_dim == V14D_PMAX, "v14d token-generation attention hardcodes head_dim=128"
    assert attn.num_key_value_heads == 1, "v14d token-generation attention expects one local KV head"
    assert attn.num_heads == 8, "v14d token-generation attention expects eight local Q heads"
    assert attn.logical_nc_config in (1, 2), "v14d token-generation attention supports LNC=1 or LNC=2"
    assert not attn.k_cache_transposed, "v14d token-generation attention expects non-transposed K cache"
    assert not attn.o_bias, "v14d token-generation attention does not apply o_proj bias"
    assert attn.get_learned_sinks() is None, "v14d token-generation attention does not support learned sinks"
    assert active_block_table is None, "v14d token-generation attention does not support block KV cache"
    assert not use_polar_compatible_rope, "v14d token-generation attention expects contiguous RoPE layout"
    assert rmsnorm is not None, "v14d token-generation attention fuses pre-attention RMSNorm"
    assert position_ids is not None, "v14d token-generation attention requires position_ids"
    assert attention_mask is not None, "baseline token-generation attention requires attention_mask"
    assert past_key_value is not None, "v14d token-generation attention requires KV cache"
    assert attn.q_layernorm is not None and attn.k_layernorm is not None, "v14d token-generation attention requires Q/K RMSNorm"

    if rotary_position_ids is None:
        rotary_position_ids = position_ids

    K_prior, V_prior = past_key_value[:2]
    assert K_prior.shape[0] == 1 and V_prior.shape[0] == 1, "v14d token-generation attention expects batch-1 KV cache"
    assert K_prior.shape[1] == 1 and V_prior.shape[1] == 1, "v14d token-generation attention expects one local KV head"
    assert K_prior.shape[2] % V14D_PMAX == 0, "v14d token-generation attention expects 128-aligned KV buckets"
    assert K_prior.shape[-1] == V14D_PMAX and V_prior.shape[-1] == V14D_PMAX, "v14d token-generation attention expects head_dim=128 KV cache"
    Wq, Wk, Wv = _get_v14d_qkv_weights(attn)
    _, _, cos_cache, sin_cache = _get_v14d_rope(
        attn,
        hidden_states,
        rotary_position_ids,
        cos_cache,
        sin_cache,
    )

    original_dtype = hidden_states.dtype
    normed_states = _qwen3_tkg_v14d_rmsnorm_kernel[attn.logical_nc_config](
        hidden_states.to(attn.torch_dtype),
        rmsnorm.weight.data,
    )

    Q, K, V, K_prior, V_prior = _qwen3_tkg_v14d_qkv_rope_kernel[attn.logical_nc_config](
        normed_states=normed_states,
        Wq=Wq,
        Wk=Wk,
        Wv=Wv,
        q_norm_weight=attn.q_layernorm.weight.data,
        k_norm_weight=attn.k_layernorm.weight.data,
        K_cache=K_prior.data,
        V_cache=V_prior.data,
        position_ids=position_ids.to(torch.int32),
        rotary_position_ids=rotary_position_ids.to(torch.int32),
    )

    K_attn, V_attn = _slice_v14d_cache_for_baseline_attention(attn, K_prior, V_prior, attention_mask)

    attn_output = attn.compute_for_token_gen(
        Q,
        K,
        V,
        position_ids,
        (K_attn, V_attn),
        attention_mask,
        active_mask,
        is_prefix_caching=attn.neuron_config.is_prefix_caching,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, s_tkg, attn.num_heads * attn.head_dim)
    attn_output = attn.get_o_proj()(attn_output, adapter_ids=adapter_ids)
    attn_output = attn_output.to(original_dtype)

    return attn_output, (K_prior, V_prior), cos_cache, sin_cache


class NeuronQwen3MoeDecoderLayer(nn.Module):
    """Decoder layer using the installed NxDI Qwen modules plus local TKG attention."""

    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Keep the config-level TKG flags enabled so NxDI skips its own KV
        # cache update path, but mask the block-TKG dispatch flag while
        # constructing the baseline attention module to preserve the standard
        # CTE o_proj path.
        _saved_block_flag = config.neuron_config.attn_block_tkg_nki_kernel_enabled
        config.neuron_config.attn_block_tkg_nki_kernel_enabled = False
        try:
            self.self_attn = NeuronQwen3MoEAttention(config=config)
        finally:
            config.neuron_config.attn_block_tkg_nki_kernel_enabled = _saved_block_flag
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
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        is_token_gen = past_key_value is not None or kwargs.get("get_kv_per_layer", False)
        if kwargs.get("windowed_context_encoding_window_idx", -1) >= 0:
            is_token_gen = False
        if self.self_attn.neuron_config.is_prefix_caching:
            q_len = hidden_states.size(1)
            if self.sequence_parallel_enabled:
                q_len *= self.self_attn.tensor_model_parallel_group.size()
            is_token_gen = is_token_gen and q_len < 128

        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        if is_token_gen:
            assert self.input_layernorm is not None, "v14d token-generation attention requires input_layernorm"
            assert not self.self_attn.neuron_config.flash_decoding_enabled, "v14d token-generation attention does not support flash decoding"
            assert not self.self_attn.neuron_config.is_chunked_prefill, "v14d token-generation attention does not support chunked prefill"

            tkg_position_ids = position_ids
            if self.self_attn.neuron_config.is_block_kv_layout:
                tkg_position_ids = kwargs["scatter_index"]

            tkg_past_key_value = past_key_value
            if kwargs.get("get_kv_per_layer", False):
                kv_mgr = kwargs.get("kv_mgr")
                assert kv_mgr is not None, "v14d token-generation attention requires a KV cache manager"
                # Match NxDI's block-TKG path: first let the cache manager run
                # its normal per-layer lookup, then pass the direct backing cache
                # to the in-place NKI kernel.
                kv_mgr.get_kv_by_layer_id(**kwargs)
                if self.self_attn.dp_degree > 1:
                    tkg_past_key_value = kv_mgr.get_kv_by_layer_id(
                        idx=kwargs["idx"],
                        kvcache_buffer=kwargs["kvcache_buffer"],
                        seq_len=hidden_states.size(1),
                        skip_slice=True,
                    )
                else:
                    tkg_past_key_value = kv_mgr._fetch_cache(
                        idx=kwargs["idx"],
                        kvcache_buffer=kwargs["kvcache_buffer"],
                    )

            if self.self_attn.neuron_config.apply_seq_ids_mask:
                seq_ids = kwargs.get("seq_ids")
                assert seq_ids is not None, "seq_ids is required when apply_seq_ids_mask is enabled"
                position_ids_invalid = (
                    tkg_past_key_value[1].shape[2] - KV_CACHE_PAD_FOR_SEQ_IDS_MASKING
                ) + torch.arange(
                    tkg_position_ids.shape[-1],
                    device=tkg_position_ids.device,
                    dtype=tkg_position_ids.dtype,
                ).reshape(1, -1).broadcast_to(tkg_position_ids.shape)
                seq_ids_mask = torch.ge(seq_ids, torch.full_like(seq_ids, 0))
                seq_ids_mask = seq_ids_mask.reshape(-1, 1).broadcast_to(tkg_position_ids.shape)
                tkg_position_ids = torch.where(seq_ids_mask, tkg_position_ids, position_ids_invalid)

            hidden_states, present_key_value, cos_cache, sin_cache = _attention_tkg_v14d(
                self.self_attn,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=tkg_position_ids,
                past_key_value=tkg_past_key_value,
                rmsnorm=self.input_layernorm,
                active_mask=kwargs.get("active_mask"),
                adapter_ids=kwargs.get("adapter_ids"),
                cos_cache=kwargs.get("cos_cache"),
                sin_cache=kwargs.get("sin_cache"),
                rotary_position_ids=kwargs.get("rotary_position_ids"),
                active_block_table=kwargs.get("active_block_table"),
                use_polar_compatible_rope=kwargs.get("use_polar_compatible_rope", False),
            )
        else:
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

        residual = hidden_states
        if not self.moe_fused_nki_kernel_enabled:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, padding_mask)[0]
        hidden_states = residual + hidden_states

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronQwen3MoeModel(NeuronBaseModel):
    """Qwen3 MoE model using the local decoder layer."""

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
    """Qwen3 MoE CausalLM using the local model class."""

    _model_cls = NeuronQwen3MoeModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return Qwen3MoeForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_neuron_config_cls(cls):
        return Qwen3V14DAttnNeuronConfig

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
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O3" if self.neuron_config.moe_ep_degree > 1 else "-O1"
        compiler_args = f"--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer {optimization_level}"
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        )
        compiler_args += " --auto-cast=none"
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        if self.neuron_config.scratchpad_page_size:
            compiler_args += (
                f" --hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size} "
            )
        return compiler_args


def generate(model_path, traced_model_path, skip_compile=False):
    generation_config = GenerationConfig.from_pretrained(model_path)

    if not skip_compile:
        neuron_config = MoENeuronConfig(
            tp_degree=4,
            batch_size=1,
            max_context_length=128,
            seq_len=1024,
            on_device_sampling_config=OnDeviceSamplingConfig(
                do_sample=True,
                temperature=0.6,
                top_k=20,
                top_p=0.95,
            ),
            enable_bucketing=False,
            flash_decoding_enabled=False,
        )
        config = Qwen3MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token

        print("\nCompiling and saving model...")
        model = NeuronQwen3MoeForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    print("\nLoading model from compiled checkpoint...")
    model = NeuronQwen3MoeForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    print("\nGenerating outputs...")
    prompt = "Give me a short introduction to large language models."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
    )
    output_tokens = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")
