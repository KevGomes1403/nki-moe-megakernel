# coding=utf-8
"""
Qwen3 MoE model with kernel_v15 fused MoE TKG kernel.

Standalone file — no imports from qwen_with_router_nki.py.

Changes vs qwen_with_router_nki.py:
  - TKG path: self.mlp(...) replaced by kernel_v15.qwen3_moe_fused_tkg[2](...)
  - CTE path: unchanged (self.mlp as before)
  - Weight layout: weights are stored in native layout and passed directly to
    the kernel — no repack or zero-padding required.

kernel_v15 uses native weight layouts with coalesced DMA (no preprocessing):
  - gate_up_w: [E, H, 2*I=384] bf16 — native flat layout (gate cols 0:I, up cols I:2I)
  - down_w: [E, I=192, H=2048] bf16 — native layout, no shard pre-split
  - Returns complete [T, H] output — LNC=2 H-sharding handled internally by the kernel.

I=192 is the per-TP-rank intermediate dim. TP all-reduce is still required after the kernel.

All tensor inputs to the kernel have .data appended for XLA tracing compatibility.
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

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.tensor_view import TensorView

torch.manual_seed(0)

import os

from kernels.router_topk.qwen3_router_topk_plan_a import qwen3_router_topk_cte

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE

# Router kernel shape constants — must match the model config
_ROUTER_H = 2048   # hidden_size
_ROUTER_E = 128    # num_experts
_ROUTER_K = 8      # top_k

# ---------------------------------------------------------------------------
# NKI Kernels
# ---------------------------------------------------------------------------

# MoE Fused TKG -- v19b
@nki.jit
def qwen3_moe_fused_tkg(
    inp,        # [B, 1, H=2048]  bf16
    gamma,      # [1, H=2048]     bf16
    router_w,   # [H=2048, E=128] bf16
    gate_up_w,  # [E=128, H=2048, 2*I=384] bf16  — NATIVE: gate cols 0:192, up cols 192:384
    down_w,     # [E=128, I=192,  H=2048]  bf16  — NATIVE: no shard split
):
    """
    Fused RMSNorm + Router + TopK(K=8) + Expert MLP for Qwen3 MoE TKG.
    Invoked as: qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, down_w)
    where [2] means LNC=2 (two NeuronCores).
    Returns: output [T, H=2048] bf16 — complete, no partial sum or all-reduce needed.

    gate_up_w is [E, H, 2*I=384] in native flat format (gate then up, no zero-pad).
    down_w is [E, I=192, H=2048] in native format (no shard pre-split).
    """
    B = inp.shape[0]
    T = B  # seq_len=1 for TKG, so tokens = batch

    H = _H
    E = _E
    K = _K
    I = _I
    I0 = _I0
    I1 = _I1
    H_free = _H_FREE
    H_free_shard = _H_FREE_SHARD
    H_shard = _H_SHARD
    I_tiles = _I_TILES

    # LNC program ID (0 or 1) — compile-time constant per NeuronCore
    prg_id = nl.program_id(axis=0)

    # -----------------------------------------------------------------------
    # Stage 1: RMSNorm
    # Input: inp [B, 1, H] → flatten to [T, H] → SBUF [_PMAX, H_free*T]
    # Output: rmsnorm_normed_bf16 [_PMAX, H_free*T] (full H, both cores identical)
    #
    # Single 3D DMA with dge_mode=3 to load inp and gamma.
    # -----------------------------------------------------------------------
    inp_2d = inp.reshape((T, H))

    # Load inp: HBM [H_free*T, _PMAX] → SBUF [H_free*T, _PMAX] via dge_mode=3
    inp_2d_hbm_reshaped = inp_2d.reshape((H_free * T, _PMAX))
    inp_flat_sb = nl.ndarray((H_free * T, _PMAX), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=inp_flat_sb,
        src=inp_2d_hbm_reshaped,
        dge_mode=3,
    )

    # Transpose [H_free*T, _PMAX] → PSUM [_PMAX, H_free*T]
    inp_trans_psum = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)

    # PSUM [_PMAX, H_free*T] → SBUF rmsnorm_out
    rmsnorm_out = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_out[...], op=nl.copy, data=inp_trans_psum[...])

    # Load gamma: HBM [H_free, _PMAX] → SBUF → transpose → SBUF gamma_sb [_PMAX, H_free]
    gamma_1d = gamma.reshape((H,))
    gamma_1d_hbm_reshaped = gamma_1d.reshape((H_free, _PMAX))
    gamma_flat_sb = nl.ndarray((H_free, _PMAX), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_flat_sb, src=gamma_1d_hbm_reshaped, dge_mode=3)
    gamma_trans_psum = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=gamma_trans_psum, data=gamma_flat_sb)
    gamma_sb = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.activation(gamma_sb[...], op=nl.copy, data=gamma_trans_psum[...])

    # RMSNorm: compute rms, apply norm and gamma
    # 1a. x^2
    rmsnorm_sq = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(rmsnorm_sq[...], op=nl.square, data=rmsnorm_out[...])

    # 1b. Reduce x^2 over H (axis=1) → [_PMAX, T]
    rmsnorm_reduced = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(rmsnorm_reduced[0:_PMAX, 0:T], nl.add, rmsnorm_sq[0:_PMAX, 0:H_free * T], axis=1)

    # 1c. gamma * input
    gamma_mult = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(gamma_mult[...], rmsnorm_out[...], gamma_sb[...], nl.multiply)

    # 1d. Cross-partition sum via GpSimdE (replaces 128×128 TensorE ones-matmul)
    sum_reduced_sb = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=sum_reduced_sb[0:1, 0:T], data=rmsnorm_reduced[0:_PMAX, 0:T], op=nl.add)

    # Broadcast [1, T] → [_PMAX, T]: copy to row 0 then shuffle to all 128 rows
    norm_sum_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=norm_sum_sb[0:1, 0:T], src=sum_reduced_sb[0:1, 0:T])
    for g in nl.static_range(4):
        nisa.nc_stream_shuffle(
            dst=norm_sum_sb[nl.ds(g * 32, 32), 0:T],
            src=norm_sum_sb[0:1, 0:T],
            shuffle_mask=[0] * 32,
        )

    # 1e. norm_factor = rsqrt(sum/H + eps)
    eps_sb = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(eps_sb, value=_EPS)
    norm_factor_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        norm_factor_sb[0:_PMAX, 0:T],
        op=nl.rsqrt,
        data=norm_sum_sb[0:_PMAX, 0:T],
        scale=1.0 / H,
        bias=eps_sb[0:_PMAX, :],
    )

    # 1f. rmsnorm_normed = gamma_mult * norm_factor (broadcast over H_free tiles)
    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=2).broadcast(dim=2, size=H_free)
    rmsnorm_normed = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(rmsnorm_normed[...], gamma_mult[...], norm_factor_bcast.get_view(), nl.multiply)

    # Cast to bf16 for router and expert matmuls
    rmsnorm_normed_bf16 = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=rmsnorm_normed[...])

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul [T, H] @ [H, E=128] → logits [T, E]
    #
    # Router weight batched DMA: 4 tiles per DMA (preserved from v13).
    # 4 × 128KB = 512KB per call vs 16 × 32KB; reduces DMA packet count 4×.
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)
    router_w_wide_sb = nl.ndarray((_PMAX, _ROUTER_BATCH, E), dtype=inp.dtype, buffer=nl.sbuf)

    for h_chunk in nl.affine_range(H_free // _ROUTER_BATCH):
        nisa.dma_copy(
            dst=router_w_wide_sb,
            src=router_w.ap(
                pattern=[[E, _PMAX], [_PMAX * E, _ROUTER_BATCH], [1, E]],
                offset=h_chunk * _ROUTER_BATCH * _PMAX * E,
            ),
            dge_mode=0,
        )
        for h_sub in nl.static_range(_ROUTER_BATCH):
            h1 = h_chunk * _ROUTER_BATCH + h_sub
            nisa.nc_matmul(
                dst=logits_psum[0:T, 0:E],
                stationary=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                moving=router_w_wide_sb[0:_PMAX, h_sub, 0:E],
            )

    logits_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(logits_sb[0:T, 0:E], op=nl.copy, data=logits_psum[0:T, 0:E])

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8) + normalize weights
    # -----------------------------------------------------------------------
    max_logit = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(max_logit[0:T, 0:1], nl.maximum, logits_sb[0:T, 0:E], axis=1)

    centered = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        centered[0:T, 0:E], data=logits_sb[0:T, 0:E],
        op0=nl.subtract, operand0=max_logit[0:T, 0:1],
    )

    exp_vals = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(exp_vals[0:T, 0:E], op=nl.exp, data=centered[0:T, 0:E])

    sum_exp = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(sum_exp[0:T, 0:1], nl.add, exp_vals[0:T, 0:E], axis=1)

    inv_sum_exp = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(inv_sum_exp[0:T, 0:1], op=nl.reciprocal, data=sum_exp[0:T, 0:1])

    probs = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        probs[0:T, 0:E], data=exp_vals[0:T, 0:E],
        op0=nl.multiply, operand0=inv_sum_exp[0:T, 0:1],
    )

    # TopK via DVE hardware
    top8_vals = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_vals[0:T, 0:K], src=probs[0:T, 0:E])

    top8_idx = nl.ndarray((T, K), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=top8_idx[0:T, 0:K], data=probs[0:T, 0:E], vals=top8_vals[0:T, 0:K])

    # Normalize top-K weights
    sum_topk = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(sum_topk[0:T, 0:1], nl.add, top8_vals[0:T, 0:K], axis=1)

    inv_sum_topk = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(inv_sum_topk[0:T, 0:1], op=nl.reciprocal, data=sum_topk[0:T, 0:1])

    norm_weights = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        norm_weights[0:T, 0:K], data=top8_vals[0:T, 0:K],
        op0=nl.multiply, operand0=inv_sum_topk[0:T, 0:1],
    )

    # -----------------------------------------------------------------------
    # Stage 4: Selective-Expert MLP — Plan F: K=8 full prefetch
    #
    # Phase 1: Issue ALL 24 DMAs (8×gate_up + 8×down_full0 + 8×down_full1)
    #          in a single static_range(K) prefetch loop before any compute.
    #          Uses K=8 separate named SBUF buffers accessed via Python lists;
    #          static_range ensures list indices are compile-time constants.
    #
    # Phase 2: Serial expert compute reads from pre-loaded buffers (no DMAs).
    #
    # gate_up_w [E, H, 2*I=384] — NATIVE layout (gate cols 0:192, up cols 192:384)
    #   DMA: ONE coalesced load per expert, stride_p=384=count=384.
    #   tile 0 (K=128): gate cols 0:128, up cols 192:320 — taken directly from buffer.
    #   tile 1 (K=128): gate cols 128:192 (64 valid), up cols 320:384 (64 valid),
    #     each zero-padded to 128 cols via SBUF tensor_copy into gate_t1_128/up_t1_128.
    #
    # down_w [E, I=192, H=2048] — NATIVE layout (no shard pre-split)
    #   DMA: ONE coalesced full-H load per I-tile per expert.
    #     tile 0: stride_p=H=2048=count=2048 → fully coalesced.
    #     tile 1: 64 valid rows, memset rows 64:128 then DMA rows 0:64; same stride=count.
    #   prg_id offsets the H-shard in SBUF at matmul time:
    #     h1_g = prg_id * H_free_shard + h1_out
    # -----------------------------------------------------------------------
    output_temp = nl.ndarray((_PMAX, H_free_shard, T), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.static_range(T):

        # ------------------------------------------------------------------
        # Allocate K=8 named SBUF buffers (par_dim=128 required as first dim)
        # Use explicit named allocations — list comprehensions are not supported
        # inside NKI kernel trace context. Access via Python list of named vars.
        # ------------------------------------------------------------------
        gate_up_buf0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf2 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf3 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf4 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf5 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf6 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_buf7 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        gate_up_bufs = [gate_up_buf0, gate_up_buf1, gate_up_buf2, gate_up_buf3,
                        gate_up_buf4, gate_up_buf5, gate_up_buf6, gate_up_buf7]

        down_full0_buf0 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf1 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf2 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf3 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf4 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf5 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf6 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_buf7 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full0_bufs = [down_full0_buf0, down_full0_buf1, down_full0_buf2, down_full0_buf3,
                           down_full0_buf4, down_full0_buf5, down_full0_buf6, down_full0_buf7]

        down_full1_buf0 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf1 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf2 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf3 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf4 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf5 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf6 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_buf7 = nl.ndarray((_PMAX, H_shard), dtype=down_w.dtype, buffer=nl.sbuf)
        down_full1_bufs = [down_full1_buf0, down_full1_buf1, down_full1_buf2, down_full1_buf3,
                           down_full1_buf4, down_full1_buf5, down_full1_buf6, down_full1_buf7]

        # ------------------------------------------------------------------
        # Hoist ALL pad memsets before the prefetch loop
        # ------------------------------------------------------------------
        # Zero pad region (rows I1:I0 = 64:128) for all K down_full1 buffers
        for k_pad in range(K):  # static Python range — compile-time unroll
            nisa.memset(down_full1_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        # gate_t1_128/up_t1_128: single pair of reused buffers (one per expert, overwritten each k)
        gate_t1_128 = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        up_t1_128   = nl.ndarray((_PMAX, H_free, I0), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        nisa.memset(gate_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
        nisa.memset(up_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        # Hoist PSUM buffers (per-expert memset stays inside k-loop)
        gate_up_psum = nl.ndarray((_PMAX, 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
        down_psum    = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.psum)

        # ------------------------------------------------------------------
        # aff_bcast setup (done before prefetch, overlaps with DMA loading)
        # ------------------------------------------------------------------
        aff_bcast = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K], src=norm_weights[t:t + 1, 0:K])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        # ------------------------------------------------------------------
        # Phase 1: Issue ALL K=8 × 3 = 24 DMAs in one static_range loop
        # static_range ensures list index is always a compile-time constant.
        # ------------------------------------------------------------------
        for k in nl.static_range(K):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            # DMA 1: gate_up — ONE coalesced load per expert [_PMAX, H_free, _GU_FLAT]
            # stride_p = 384 = count_col = 384 → FULLY COALESCED.
            nisa.dma_copy(
                dst=gate_up_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # DMA 2a: down tile 0 — I rows 0:128, H_shard=1024 cols only (shard-aware)
            # down_w [E, I=192, H=2048]: P-stride=H=2048, count_p=I0=128, count_col=H_shard=1024
            # offset = prg_id * H_shard → load only this core's H columns.
            nisa.dma_copy(
                dst=down_full0_bufs[k],
                src=down_w.ap(
                    pattern=[[H, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # DMA 2b: down tile 1 — I rows 128:192 (64 valid), H_shard=1024 cols only
            # Pad region (rows I1:I0 = 64:128) already zeroed above.
            # offset = I0 * H + prg_id * H_shard → skip tile-0 rows, then shard offset.
            nisa.dma_copy(
                dst=down_full1_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H_shard]],
                    offset=I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # ------------------------------------------------------------------
        # Phase 2: Serial expert compute — no DMAs, reads from pre-loaded buffers
        # ------------------------------------------------------------------
        for k in nl.static_range(K):
            # Tile-1 tensor_copy using pre-loaded gate_up_bufs[k]
            # gate cols 128:192 (I1=64 valid) → gate_t1_128 cols 0:64 valid, 64:128 zero.
            # up   cols 320:384 (I1=64 valid) → up_t1_128   cols 0:64 valid, 64:128 zero.
            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free, nl.ds(I + I0, I1)],
            )

            # ----------------------------------------------------------------
            # Gate/Up matmul: [H, I] @ [H, T=1] → [I, T] in PSUM
            # PSUM layout: [_PMAX, 4] — cols 0,1 = gate tiles 0,1; cols 2,3 = up tiles 0,1
            # ----------------------------------------------------------------
            nisa.memset(gate_up_psum, value=0.0)

            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        # Tile 0: K=128 cols taken directly from the combined buffer
                        g_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]   # gate cols 0:128
                        u_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]   # up   cols 192:320
                    else:
                        # Tile 1: K=128 zero-padded (64 valid + 64 zeros)
                        g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]               # gate tile 1
                        u_stat = up_t1_128[0:_PMAX, h1, 0:I0]                 # up   tile 1
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, i_tile:i_tile + 1],
                        stationary=g_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, I_tiles + i_tile:I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )

            # Flush gate/up PSUM → SBUF
            gate_sb = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            up_sb   = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(gate_sb, op=nl.copy, data=gate_up_psum[0:_PMAX, 0:I_tiles])
            nisa.activation(up_sb,   op=nl.copy, data=gate_up_psum[0:_PMAX, I_tiles:2 * I_tiles])

            # SiLU(gate) * up → inter_f32 [_PMAX, I_tiles]
            # Rows 64:128 of I_tiles col 1 are zero (from tile-1 zero-padding) — correct.
            silu_res  = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(silu_res, op=nl.silu, data=gate_sb)
            nisa.tensor_tensor(inter_f32, silu_res, up_sb, nl.multiply)

            # Cast inter to bf16 for the down matmul
            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            # ----------------------------------------------------------------
            # Down matmul: [I, H] @ [I, T=1] → [H_shard, T] partial sum
            # Uses pre-loaded down_full0_bufs[k]/down_full1_bufs[k].
            # ----------------------------------------------------------------
            nisa.memset(down_psum, value=0.0)

            for h1_out in nl.affine_range(H_free_shard):
                # Tile 0: I rows 0:128 (all 128 valid)
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_full0_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                # Tile 1: I rows 128:192 (64 valid, rows 64:128 zero from pre-k-loop memset)
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, h1_out:h1_out + 1],
                    stationary=down_full1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            # Flush down PSUM → SBUF, scale by expert affinity
            down_result_sb = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free_shard],
                op=nl.copy,
                data=down_psum[0:_PMAX, 0:H_free_shard],
            )
            down_result_scaled = nl.ndarray((_PMAX, H_free_shard), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],
            )

            # Accumulate into output_temp (k==0: copy to initialise; k>0: add)
            if k == 0:
                nisa.tensor_copy(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    src=down_result_scaled[0:_PMAX, 0:H_free_shard],
                )
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data1=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data2=down_result_scaled[0:_PMAX, 0:H_free_shard],
                    op=nl.add,
                )

    # -----------------------------------------------------------------------
    # Stage 5: Transpose fp32→bf16, store to HBM
    # output_temp [_PMAX, H_free_shard, T] → HBM output [T, H] bf16
    # Each core writes its H_shard columns at HBM offset prg_id*H_shard.
    # -----------------------------------------------------------------------
    output = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.hbm)
    out_sb = nl.ndarray((T, H_shard), dtype=inp.dtype, buffer=nl.sbuf)

    for h1 in nl.static_range(H_free_shard):
        tp_psum = nl.ndarray((T, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp_psum[0:T, 0:_PMAX], data=output_temp[0:_PMAX, h1, 0:T])
        nisa.activation(
            dst=out_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)],
            op=nl.copy,
            data=tp_psum[0:T, 0:_PMAX],
        )

    nisa.dma_copy(
        dst=output[0:T, nl.ds(prg_id * H_shard, H_shard)],
        src=out_sb[0:T, 0:H_shard],
    )

    return output

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
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj
        down_proj = torch.empty(config.num_experts, intermediate_size, hidden_size, dtype=dtype, device=device)
        for e in range(config.num_experts):
            down_proj_weights = (
                neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"].T.detach().clone()
            )
            down_proj_slice = torch.narrow(down_proj, 0, e, 1)
            down_proj_slice.copy_(down_proj_weights)
            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"]
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

    Input hidden_states may have leading dims (S, B, H); they are flattened to
    (T, H) before the kernel call, matching RouterBase.get_router_logits behaviour.
    """

    def forward(self, hidden_states):
        # Flatten (S, B, H) or any leading dims → (T, H), matching RouterBase.get_router_logits
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
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

        rl, ea, ei = qwen3_router_topk_cte[2](x.data, w.data, rl, ea, ei)

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

    The TKG fused kernel casts hidden states to float32 internally before the
    router matmul (MoEFusedTKGConfig.router_mm_dtype defaults to torch.float32).
    weight_T on the TKG router (moe_fused_tkg.router, which is the old router)
    must also be float32 to satisfy the kernel's x.dtype == w.dtype assertion.
    This is done here before the swap so moe_fused_tkg.router.weight_T is float32
    at both compile time and runtime (preshard_hook is also patched to preserve this).
    """
    old = mlp.router

    # Cast the TKG router's weight_T to float32 to match router_mm_dtype=float32.
    # moe_fused_tkg.router IS old, so this fixes the TKG path without touching our router.
    if hasattr(old, 'weight_T'):
        old.weight_T = nn.Parameter(old.weight_T.data.to(torch.float32))

    # Override _apply on old so that model.to(bfloat16) (called in model_wrapper.py
    # after construction) does not cast weight_T back to bfloat16.
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

    # Patch moe_fused_tkg.preshard_hook to derive weight_T at load time.
    #
    # invoke_preshard_hook (trace.py) walks the module tree and, when it finds a
    # module with preshard_hook, calls it and *returns immediately* without
    # recursing into children. MoEFusedTKG.preshard_hook is a no-op pass, so the
    # framework calls that and stops — it never reaches old.preshard_hook on the
    # child router. We must therefore patch moe_fused_tkg directly.
    #
    # The framework calls moe_fused_tkg.preshard_hook(state_dict, prefix) where
    # prefix = "…mlp.moe_fused_tkg.weight". From that we can reconstruct:
    #   checkpoint key : "…mlp.router.linear_router.weight"
    #   target key     : "…mlp.moe_fused_tkg.router.weight_T"
    def _tkg_preshard_hook(model_state_dict, prefix):
        # prefix = "…mlp.moe_fused_tkg.weight"
        mlp_prefix = prefix.removesuffix("moe_fused_tkg.weight")   # "…mlp."
        tkg_prefix  = mlp_prefix + "moe_fused_tkg."                 # "…mlp.moe_fused_tkg."

        original_key  = mlp_prefix + "router.linear_router.weight"  # checkpoint key
        transposed_key = tkg_prefix + "router.weight_T"             # target key
        if original_key in model_state_dict:
            model_state_dict[transposed_key] = (
                model_state_dict[original_key]
                .detach().transpose(0, 1).clone().to(torch.float32)
            )
        elif transposed_key in model_state_dict:
            # Compiled checkpoint already has weight_T — just ensure float32.
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


# ---------------------------------------------------------------------------
# Decoder layer — base (CTE path unchanged, TKG uses standard mlp)
# ---------------------------------------------------------------------------

class NeuronQwen3MoeDecoderLayerWithNKI(nn.Module):
    """Decoder layer with NKI attention (CTE) and NKI router."""

    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3MoEAttentionWithNKICTE(config=config)
        self.moe_fused_nki_kernel_enabled = getattr(config, "moe_fused_nki_kernel_enabled", False)

        _dtype = config.neuron_config.torch_dtype
        # Cast to target dtype at construction so XLA traces with bfloat16 from the start.
        # Without this, Qwen3MoeRMSNorm initializes its weight as float32 (torch.ones default),
        # which propagates float32 hidden states into the TKG fused kernel and triggers
        # an x.dtype != w.dtype assertion against the bfloat16 weight_T.
        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps).to(_dtype)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps).to(_dtype)

        self.mlp = initialize_moe_module(
            config=config, rmsnorm=self.post_attention_layernorm, init_tkg_module=True
        )
        # Install the NKI router in place of the PyTorch RouterTopK.
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
        # if not self.moe_fused_nki_kernel_enabled:
        #     hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, padding_mask)[0]
        hidden_states = residual + hidden_states

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Decoder layer — TKG MoE path uses kernel_v15 (tile-interleaved, I=192 native)
# ---------------------------------------------------------------------------

class NeuronQwen3MoeDecoderLayerFusedTKG(NeuronQwen3MoeDecoderLayerWithNKI):
    """Decoder layer that calls kernel_v15.run() for TKG, CTE unchanged.

    Gate/up weights are sliced from the shared gate_up_proj.weight [E, H, 2*I]
    at trace time via .view(E, H, 2, I) + .contiguous(), avoiding 9 GiB of
    per-layer static parameter duplicates across 48 MoE layers.
    """

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

        is_tkg = past_key_value is not None
        if is_tkg:
            # kernel_v15 handles RMSNorm + Router + TopK(8) + Expert MLPs internally.
            # inp:      [B, 1, H]         — pre-norm hidden states
            # gamma:    [1, H]            — post-attention RMSNorm weight
            # router_w: [H, E] bf16       — weight_T, set up by _install_nki_router
            # gate_up:  [E, H, 2*I=384]  — native layout (gate cols 0:I, up cols I:2I)
            # down_w:   [E, I=192, H]    — native layout, no shard pre-split
            tkg = self.mlp.moe_fused_tkg
            moe_out = qwen3_moe_fused_tkg[2](
                hidden_states.data,
                self.post_attention_layernorm.weight.unsqueeze(0).data,        # [1, H]
                tkg.router.weight_T.data,                                      # [H, E] bf16
                tkg.expert_mlps.mlp_op.gate_up_proj.weight.data,               # [E, H, 2*I=384]
                tkg.expert_mlps.mlp_op.down_proj.weight.data,                  # [E, I=192, H]
            )                                                                  # returns [T, H] bf16
            if isinstance(moe_out, (tuple, list)):
                moe_out = moe_out[0]
            # TP all-reduce: each rank produced a partial sum over its I shard;
            # sum across TP ranks to get the full output.
            moe_out = mappings.reduce_from_tensor_model_parallel_region(
                moe_out, process_group=parallel_state.get_world_group()
            )
            # restore seq dim to match residual [B, 1, H]
            hidden_states = moe_out.unsqueeze(1)
        else:
            # CTE path: unchanged
            hidden_states = self.mlp(hidden_states, padding_mask)[0]

        hidden_states = residual + hidden_states

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NeuronQwen3MoeModel(NeuronBaseModel):
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
            NeuronQwen3MoeDecoderLayerFusedTKG(config, layer_idx)
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

class NeuronQwen3MoeForCausalLM(NeuronBaseForCausalLM):
    """Qwen3 MoE CausalLM with kernel_v15 fused MoE TKG kernel."""

    _model_cls = NeuronQwen3MoeModel

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
