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

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import core_barrier
from nkilib.core.utils.tensor_view import TensorView

import os
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"  
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
# from kernels.attn_tkg.agents.v10e import qwen3_attn_tkg_fused_oproj_v10e
# from kernels.router_topk.qwen3_router_topk_plan_a import qwen3_router_topk_cte
# from kernels.moe_fused_tkg import kernel_v19b as custom_moe_fused_kernel

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE

# Router kernel shape constants — must match the model config
_ROUTER_H = 2048   # hidden_size
_ROUTER_E = 128    # num_experts
_ROUTER_K = 8      # top_k

# ---------------------------------------------------------------------------
# NKI Kernels
# ---------------------------------------------------------------------------

# --------------- Hardcoded Qwen3 constants ---------------
H = 2048          # hidden_size
E = 128           # num_experts
K = 8             # top-K experts per token
P = 128           # NeuronCore partition dimension (trn2)
NUM_H_TILES = H // P   # = 16
# Plan A: T_TILE=128 fills the full partition dimension (P=128).
# The original T_TILE=64 left the upper half of PSUM empty; 128 doubles utilization.
T_TILE = 128      # was 64

@nki.jit(platform_target="trn2")
def qwen3_router_topk_cte(
    x,                  # [H, T]    bf16  — hidden states, H-major for burst DMA (Plan A)
    w,                  # [H, E]    bf16  — router weight (transposed from [E, H])
    router_logits,      # [T, E]    float32  — output: raw logits before softmax
    expert_affinities,  # [T, E]    float32  — output: scattered, L1-normalized affinities
    expert_index,       # [T, K]    uint32   — output: top-K expert indices per token
):
    """
    Router top-K kernel specialized for Qwen3 CTE shapes, LNC=2. Plan A variant.

    Launched as: qwen3_router_topk_cte[2](x, w, router_logits, expert_affinities, expert_index)

    x is expected in [H, T] layout (H-major). This enables a single burst DMA
    that loads T_local contiguous tokens per (partition, H-tile) row without
    stride-gather penalties.

    Note: output parameters (router_logits, expert_affinities, expert_index) are
    accepted for interface compatibility but the kernel writes to nl.shared_hbm
    buffers internally and returns those.  The caller should use the returned
    tensors, not the passed-in output buffers.
    """
    T = x.shape[1]   # total tokens (dynamic); x is [H, T] so dim 1 is T

    # ----------------------------------------------------------------
    # LNC sharding: each core processes T_local = T/2 tokens
    # ----------------------------------------------------------------
    n_prgs = nl.num_programs(0)   # 2 when launched with [2]
    prg_id = nl.program_id(0)     # 0 or 1

    # Each core owns a contiguous half of the token dimension
    T_local = T // n_prgs          # tokens per core (e.g. 320 for T=640, LNC=2)
    T_offset = prg_id * T_local    # token start index for this core

    # Ceiling division: handles the case where T_local is not divisible by T_TILE.
    # For T_local=320, T_TILE=128: ceil(320/128) = 3 tiles (128, 128, 64 tokens).
    num_t_tiles = (T_local + T_TILE - 1) // T_TILE

    # ----------------------------------------------------------------
    # Allocate output tensors as nl.shared_hbm.
    # Using nl.shared_hbm (rather than writing to the passed-in output
    # parameter tensors) avoids a neuronx-cc compiler bug where multiple
    # nisa.dma_copy stores to the same parameter tensor cause an InstSave
    # assertion failure in the BIR address-rotation/dma-optimization passes.
    # ----------------------------------------------------------------
    rl_out  = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.shared_hbm)
    ea_out  = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.shared_hbm)
    ei_out  = nl.ndarray((T, K), dtype=nl.uint32,  buffer=nl.shared_hbm)

    # ----------------------------------------------------------------
    # Plan A: load entire w into SBUF via single burst DMA.
    # w[H, E] is reorganized as w_sb[P, NUM_H_TILES, E].
    # .ap strides: partition stride = NUM_H_TILES*E (skip one full P-row worth),
    #              h-tile stride    = E              (skip one E-block within a P-row),
    #              element stride   = 1              (E contiguous elements).
    # Mapping: w_sb[p, ht, e] = w[ht*P + p, e]
    # Each (p, ht) row reads E=128 contiguous HBM bytes — burst, not gather.
    # w stays OUTSIDE the T-tile loop — same weights for all tiles.
    # ----------------------------------------------------------------
    T_full = x.shape[1]  # full T dimension of x [H, T] = 640
    w_sb = nl.ndarray((P, NUM_H_TILES, E), dtype=nl.bfloat16, buffer=nl.sbuf, name="w_sb")
    # w.ap strides map w_sb[p, ht, e] → w[ht*P + p, e]:
    # w[H, E] flat layout: element (h, e) is at offset h*E + e.
    # We want h = ht*P + p, so offset = (ht*P + p)*E + e.
    #   partition dim (p): stride = E   (advancing p by 1: (ht*P+p)*E → (ht*P+p+1)*E, delta=E)
    #   h-tile dim  (ht):  stride = P*E (advancing ht by 1: delta = P*E)
    #   element dim (e):   stride = 1,  size = E (E contiguous elements per row of w)
    nisa.dma_copy(
        dst=w_sb,
        src=w.ap([[E, P], [P * E, NUM_H_TILES], [1, E]]),
    )

    # expert_iota is constant — compute outside the T-tile loop
    expert_iota = nl.ndarray((P, E), dtype=nl.uint32, buffer=nl.sbuf,
                             name="expert_iota")
    nisa.iota(dst=expert_iota, pattern=[[1, E]], offset=0, channel_multiplier=0)

    # ----------------------------------------------------------------
    # Plan A: load x into SBUF via single burst DMA.
    # x[H, T] is reorganized as x_sb[P, NUM_H_TILES, T_local].
    # .ap strides map x_sb[p, ht, t] → x[ht*P + p, T_offset + t]:
    #   partition dim (p): stride = NUM_H_TILES*T_full (one full row of x[H,T] = T_full
    #                               elements, repeated NUM_H_TILES times per partition)
    #   h-tile dim (ht):   stride = P*T_full (each H-tile spans P rows of x)
    #   element dim (t):   stride = 1, size = T_local (T_local contiguous tokens)
    # offset = T_offset positions into the T dimension of x.
    # Within each (p, ht) row: T_local contiguous HBM reads — true burst DMA.
    # x stays OUTSIDE the T-tile loop — reused across T tiles.
    # ----------------------------------------------------------------
    x_sb = nl.ndarray((P, NUM_H_TILES, T_local), dtype=nl.bfloat16, buffer=nl.sbuf,
                      name="x_sb")
    # x[H, T_full] viewed as x_sb[P, NUM_H_TILES, T_local]:
    #   partition stride = T_full        (x row p=0 starts at 0, p=1 starts at T_full,
    #                                     but each partition covers NUM_H_TILES H-indices
    #                                     interleaved: h=p, h=p+P, h=p+2P, ...)
    #   Actually the mapping ht*P + p means h-tile is the slow index, partition is fast.
    #   So stride for ht (slow dim in x):  P * T_full  (skip P rows of x to advance ht)
    #   Stride for p  (fast dim in x):     T_full      (skip one row of x to advance p)
    #   Stride for t  (innermost):         1            (contiguous T elements)
    nisa.dma_copy(
        dst=x_sb,
        src=x.ap([[T_full, P], [P * T_full, NUM_H_TILES], [1, T_local]], offset=T_offset),
    )

    # ----------------------------------------------------------------
    # T-tile loop: process T_local tokens in T_TILE-sized chunks.
    # Using plain Python range() (not nl.affine_range) because T_TILE_actual
    # varies per iteration — each tile gets independently-named buffers.
    # ----------------------------------------------------------------
    for t_tile in range(num_t_tiles):
        # Absolute token offset in the full [T] output dimension
        t_off = T_offset + t_tile * T_TILE

        # Handle the last (potentially partial) tile.
        # For T_local=320, T_TILE=128: tiles 0,1 have 128 tokens, tile 2 has 64.
        T_TILE_actual = min(T_TILE, T_local - t_tile * T_TILE)

        # ----------------------------------------------------------------
        # Matmul: [T_TILE_actual, H] @ [H, E] → [T_TILE_actual, E]
        # PSUM dimension is [T_TILE_actual, E]; T_TILE_actual <= P=128 ✓
        # nc_matmul args: stationary=[P, T_TILE_actual], moving=[P, E]
        # stationary=x_sb[:, ht, nl.ds(t_tile*T_TILE, T_TILE_actual)] gives
        # a [P, T_TILE_actual] slice directly from SBUF — no tensor_copy needed.
        # The ht loop uses plain Python range (not nl.affine_range) so ht is a
        # compile-time integer, enabling direct index into the x_sb/w_sb arrays.
        # ----------------------------------------------------------------
        router_logits_psum = nl.zeros((T_TILE_actual, E), dtype=nl.float32,
                                      buffer=nl.psum, name=f"rl_psum_{t_tile}")

        for ht in range(NUM_H_TILES):  # Python range: ht is compile-time int for direct SBUF slice
            nisa.nc_matmul(
                dst=router_logits_psum,
                stationary=x_sb[:, ht, nl.ds(t_tile * T_TILE, T_TILE_actual)],  # [P, T_TILE_actual]
                moving=w_sb[:, ht, :],                                           # [P, E]
            )

        # ----------------------------------------------------------------
        # Copy PSUM → SBUF and store router_logits slice to HBM
        # ----------------------------------------------------------------
        router_logits_sb = nl.ndarray((T_TILE_actual, E), dtype=nl.float32,
                                      buffer=nl.sbuf, name=f"rl_sb_{t_tile}")
        nisa.tensor_copy(dst=router_logits_sb, src=router_logits_psum)

        # Round fp32 PSUM accum to bf16 grid to match CPU bf16 matmul precision
        rl_bf16_tmp = nl.ndarray((T_TILE_actual, E), dtype=nl.bfloat16, buffer=nl.sbuf,
                                 name=f"rl_bf16_{t_tile}")
        nisa.tensor_copy(dst=rl_bf16_tmp, src=router_logits_sb)   # fp32 → bf16
        nisa.tensor_copy(dst=router_logits_sb, src=rl_bf16_tmp)   # bf16 → fp32

        # t_off = T_offset + t_tile * T_TILE is the absolute token index, so
        # t_off * E is the correct HBM byte-offset for rl_out[t_off:t_off+T_TILE_actual, :]
        nisa.dma_copy(
            dst=rl_out.ap([[E, T_TILE_actual], [1, E]], offset=t_off * E),
            src=router_logits_sb,
        )

        # ----------------------------------------------------------------
        # Softmax (numerically stable)
        # ----------------------------------------------------------------
        affinities_sb = nl.ndarray((T_TILE_actual, E), dtype=nl.float32, buffer=nl.sbuf,
                                   name=f"affi_sb_{t_tile}")

        negmax_sb = nl.ndarray((T_TILE_actual, 1), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"negmax_sb_{t_tile}")
        nisa.tensor_reduce(
            dst=negmax_sb,
            op=nl.maximum,
            data=router_logits_sb,
            axis=1,
            negate=True,
            keepdims=True,
        )

        inv_sum_sb = nl.ndarray((T_TILE_actual, 1), dtype=nl.float32, buffer=nl.sbuf,
                                name=f"inv_sum_sb_{t_tile}")
        nisa.activation(
            dst=affinities_sb,
            op=nl.exp,
            data=router_logits_sb,
            bias=negmax_sb,
            reduce_op=nl.add,
            reduce_res=inv_sum_sb,
        )
        nisa.reciprocal(dst=inv_sum_sb, data=inv_sum_sb)
        nisa.tensor_scalar(
            dst=affinities_sb,
            data=affinities_sb,
            op0=nl.multiply,
            operand0=inv_sum_sb,
        )

        # ----------------------------------------------------------------
        # Top-K selection
        # ----------------------------------------------------------------
        topk_vals_sb = nl.ndarray((T_TILE_actual, K), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"topk_vals_sb_{t_tile}")
        topk_idx_sb  = nl.ndarray((T_TILE_actual, K), dtype=nl.uint32,  buffer=nl.sbuf,
                                  name=f"topk_idx_sb_{t_tile}")

        top8_buf = nl.ndarray((T_TILE_actual, 8), dtype=nl.float32, buffer=nl.sbuf,
                              name=f"top8_buf_{t_tile}")
        nisa.max8(dst=top8_buf, src=affinities_sb)
        nisa.tensor_copy(dst=topk_vals_sb, src=top8_buf[:, :K])

        idx8_buf = nl.ndarray((T_TILE_actual, 8), dtype=nl.uint32, buffer=nl.sbuf,
                              name=f"idx8_buf_{t_tile}")
        nisa.nc_find_index8(dst=idx8_buf, data=affinities_sb, vals=top8_buf)
        nisa.tensor_copy(dst=topk_idx_sb, src=idx8_buf[:, :K])

        topk_idx_fp32_sb = nl.ndarray((T_TILE_actual, K), dtype=nl.float32, buffer=nl.sbuf,
                                      name=f"topk_idx_fp32_sb_{t_tile}")
        nisa.tensor_copy(dst=topk_idx_fp32_sb, src=topk_idx_sb)

        # t_off * K is the correct HBM byte-offset for ei_out[t_off:t_off+T_TILE_actual, :]
        nisa.dma_copy(
            dst=ei_out.ap([[K, T_TILE_actual], [1, K]], offset=t_off * K),
            src=topk_idx_sb,
        )

        # ----------------------------------------------------------------
        # L1 normalization of top-K affinities
        # ----------------------------------------------------------------
        sum_topk_sb = nl.ndarray((T_TILE_actual, 1), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"sum_topk_sb_{t_tile}")
        nisa.tensor_reduce(
            dst=sum_topk_sb,
            op=nl.add,
            data=topk_vals_sb,
            axis=1,
            keepdims=True,
        )
        nisa.reciprocal(dst=sum_topk_sb, data=sum_topk_sb)

        topk_vals_norm_sb = nl.ndarray((T_TILE_actual, K), dtype=nl.float32, buffer=nl.sbuf,
                                       name=f"topk_vals_norm_sb_{t_tile}")
        nisa.tensor_scalar(
            dst=topk_vals_norm_sb,
            data=topk_vals_sb,
            op0=nl.multiply,
            operand0=sum_topk_sb,
        )

        # ----------------------------------------------------------------
        # One-hot scatter: build [T_TILE_actual, E] mask.
        # expert_iota[:T_TILE_actual, :] slices the hoisted iota buffer.
        # ----------------------------------------------------------------
        mask_sb = nl.ndarray((T_TILE_actual, E), dtype=nl.float32, buffer=nl.sbuf,
                             name=f"mask_sb_{t_tile}")
        nisa.memset(dst=mask_sb, value=0.0)

        check_buf = nl.ndarray((T_TILE_actual, E), dtype=nl.float32, buffer=nl.sbuf,
                               name=f"check_buf_{t_tile}")
        for k_slot in nl.affine_range(K):
            nisa.tensor_scalar(
                dst=check_buf[:T_TILE_actual, :],
                op0=nl.equal,
                data=expert_iota[:T_TILE_actual, :],
                operand0=topk_idx_fp32_sb[:T_TILE_actual, k_slot],
            )
            nisa.tensor_tensor(
                dst=mask_sb[:T_TILE_actual, :],
                data1=mask_sb[:T_TILE_actual, :],
                op=nl.add,
                data2=check_buf[:T_TILE_actual, :],
            )

        # ----------------------------------------------------------------
        # Apply normalized affinities through the mask
        # ----------------------------------------------------------------
        nisa.tensor_scalar(
            dst=affinities_sb,
            data=affinities_sb,
            op0=nl.multiply,
            operand0=sum_topk_sb,
        )

        scattered_sb = nl.ndarray((T_TILE_actual, E), dtype=nl.float32, buffer=nl.sbuf,
                                  name=f"scattered_sb_{t_tile}")
        nisa.tensor_tensor(
            dst=scattered_sb,
            data1=mask_sb,
            op=nl.multiply,
            data2=affinities_sb,
        )

        # ----------------------------------------------------------------
        # Store scattered expert_affinities slice to HBM
        # ----------------------------------------------------------------
        nisa.dma_copy(
            dst=ea_out.ap([[E, T_TILE_actual], [1, E]], offset=t_off * E),
            src=scattered_sb,
        )

    # Barrier ensures both cores have written their T_local rows before the
    # caller reads the full [T, E] expert_affinities tensor.
    # core_barrier stays OUTSIDE the T-tile loop.
    core_barrier(ea_out, cores=[0, 1])

    return rl_out, ea_out, ei_out

PMAX = 128
F_MAX = 512
EPS = 1e-6
INV_SQRT_D = float(1.0 / math.sqrt(128.0))

@nki.jit(platform_target="trn2")
def qwen3_attn_tkg_fused_oproj_v10e(
    hidden_states,   # [B, 1, H]        bf16  (B=1)
    Wq,              # [Hq_tp*d, H]     bf16  [1024, 2048]
    Wk,              # [Hkv_tp*d, H]    bf16  [128, 2048]  (Hkv_tp=1)
    Wv,              # [Hkv_tp*d, H]    bf16  [128, 2048]
    Wo,              # [Hq_tp*d, H]     bf16  [1024, 2048]  transposed o_proj weight
    q_norm_weight,   # [d]              bf16  [128]
    k_norm_weight,   # [d]              bf16  [128]
    K_cache,         # [B, 1, S_prior, d] bf16
    V_cache,         # [B, 1, S_prior, d] bf16
    cos,             # [B, d]           bf16
    sin,             # [B, d]           bf16
    position_ids,    # [B, 1]           int32 — decoding step (number of valid cache tokens)
):
    """
    Fused QKV + RMSNorm + RoPE + flash decode + output projection.
    Attention mask is generated on-chip from position_ids threshold.
    Returns (output, k_rope_out, v_out):
      - output:     [B, 1, H_out] bf16, where H_out = H = 2048
      - k_rope_out: [d, B] = [128, 1] bf16 — new token's K after RMSNorm+RoPE
      - v_out:      [d, B] = [128, 1] bf16 — new token's V (no RoPE)

    Wo is passed as [Hq_out=1024, H_wo=2048] (caller transposes the weight).
    This enables contiguous DMA loading via nkilib-style ap() pattern.

    On-chip masking (v10e): For each K-cache tile s_t, computes an exact binary
    mask using position_ids:
      tile_start = s_t * PMAX  (compile-time)
      row_global = tile_start + p  (p = partition index 0..127)
      mask[p] = 0     if row_global < pos   (valid: token already in cache)
      mask[p] = -1e9  if row_global >= pos  (future/padding)
    Uses relu + clamp idiom: relu(p - (pos - tile_start)) clamped to [0,1] * -1e9.
    """
    # --- Dimensions ---
    B = hidden_states.shape[0]      # 1
    H = hidden_states.shape[2]      # 2048
    Hq_out = Wq.shape[0]            # 1024  = Hq_tp * d
    d = PMAX                        # 128
    Hq_tp = Hq_out // d             # 8
    Hkv_tp = 1                      # per-rank KV heads (corrected)
    GQA = Hq_tp // Hkv_tp          # 8
    S_prior = K_cache.shape[2]
    num_h_tiles = H // PMAX         # 16
    num_s_tiles = S_prior // PMAX
    half_d = d // 2                 # 64

    # Output H: since no LNC, each core writes all H=2048 of Wo output
    # Wo is now [Hq_out=1024, H_wo=2048] (transposed), so H_wo is shape[1]
    H_wo = Wo.shape[1]              # 2048
    num_h_blocks = H_wo // F_MAX   # 4

    # =========================================================================
    # Plan A — Static shape constants
    # Replace runtime-derived loop bounds with compile-time integer constants.
    # =========================================================================
    assert S_prior % PMAX == 0, f"S_prior={S_prior} must be a multiple of {PMAX}"
    NUM_S_TILES  = S_prior // PMAX   # trace-time constant; 5 for S_prior=640
    NUM_H_TILES  = 16   # H=2048 / PMAX=128
    HQ_TP_CONST  = 8    # Hq_tp fixed for this shape
    NUM_H_BLOCKS = 4    # H_wo=2048 / F_MAX=512
    assert H == NUM_H_TILES * PMAX, f"H={H} must be {NUM_H_TILES*PMAX}"
    assert Hq_tp == HQ_TP_CONST, f"Hq_tp={Hq_tp} must be {HQ_TP_CONST}"

    # =========================================================================
    # COLUMN LAYOUT RESHAPES
    # =========================================================================
    # Output [B, 1, H_wo]: allocate in HBM
    output = nl.ndarray((B, 1, H_wo), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # Reshape to [1, H_wo] for o_proj DMA stores.
    output_2d = output.reshape((1, H_wo))

    # K/V outputs for KV cache update
    # Use (B, d) = (1, 128) HBM shape so DMA can write in one contiguous packet
    # (partition=1, free=128) matching the known-working output DMA pattern.
    # Callers may view as (d, B) = (128, 1) via reshape.
    k_rope_out = nl.ndarray((B, d), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    v_out = nl.ndarray((B, d), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # Hidden: [B, 1, H] -> [H, B] column layout
    hidden_col = hidden_states.reshape((H, B))
    # cos/sin: [B, d] -> [PMAX, B]
    cos_col = cos.reshape((PMAX, B))
    sin_col = sin.reshape((PMAX, B))

    # =========================================================================
    # LOAD CONSTANTS
    # =========================================================================
    # Norm weights [128] -> [PMAX, 1] f32 in SBUF
    qnw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnw_bf16")
    nisa.dma_copy(dst=qnw_bf16, src=q_norm_weight.reshape((PMAX, 1)))
    qnw_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qnw_sb")
    nisa.tensor_copy(qnw_sb, qnw_bf16)

    knw_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="knw_bf16")
    nisa.dma_copy(dst=knw_bf16, src=k_norm_weight.reshape((PMAX, 1)))
    knw_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="knw_sb")
    nisa.tensor_copy(knw_sb, knw_bf16)

    # cos/sin in SBUF f32 [PMAX, 1] (B=1, so [PMAX, B] = [PMAX, 1])
    cos_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="cos_bf16")
    sin_bf16 = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="sin_bf16")
    nisa.dma_copy(dst=cos_bf16, src=cos_col)
    nisa.dma_copy(dst=sin_bf16, src=sin_col)
    cos_f32 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="cos_f32")
    sin_f32 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="sin_f32")
    nisa.tensor_copy(cos_f32, cos_bf16)
    nisa.tensor_copy(sin_f32, sin_bf16)

    # All-ones [PMAX, PMAX] for reduction matmuls (RMSNorm sum-of-squares, softmax sums)
    rms_ones = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)

    # Plan C1 NOTE: The original plan specified using nc_matmul outer-products
    # ([PMAX,1] @ [1,GQA]) for broadcasting.  However, the Trainium2 hardware
    # requires both operands of nc_matmul to share the same partition dimension
    # (par_dim must be equal).  [PMAX,1] has par=PMAX=128, [1,GQA] has par=1 —
    # they differ, so the compiler rejects them ("Fmap and Weight partitions must
    # match").  The original for-loop broadcasts from v6_ultimate are preserved
    # at all 6 sites as they are the correct working approach.

    # =========================================================================
    # HIDDEN TILE HOISTING
    # Pre-load all 16 hidden tiles [PMAX, 1] outside all loops. Reused for
    # all Q/K/V projections.
    # =========================================================================
    # Load entire hidden column as [PMAX, num_h_tiles] in one wide DMA.
    # hidden_col is [H, B] = [2048, 1] row-major; flat offset of [r,0] = r.
    # We want h_all[p, f] = hidden_col[f*PMAX + p, 0], i.e. flat offset = f*PMAX + p.
    # ap() pattern: partition p steps by 1 (count PMAX), free f steps by PMAX (count num_h_tiles).
    h_all = nl.ndarray((PMAX, num_h_tiles), dtype=nl.bfloat16, buffer=nl.sbuf, name="h_all")
    nisa.dma_copy(
        dst=h_all,
        src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
    )

    # =========================================================================
    # WO WEIGHT HOISTING — nkilib-style contiguous DMA
    #
    # Wo is now passed as [Hq_out=1024, H_wo=2048] = [N*D, H] (caller transposes).
    # Reshape to [Hq_tp=8, d=128, H_wo=2048] so each head's slice is contiguous.
    #
    # ap() pattern [[H_wo, PMAX], [1, H_wo]]:
    #   partition stride = H_wo = 2048 (one full output row apart)
    #   free stride = 1 (contiguous elements)
    # → 128 chunks × H_wo×2 = 4096 bytes each, stride 4096 bytes → 50% fill ratio
    # vs old [[1,128],[Hq_out,H_wo]]: 2048 chunks × 256 bytes, stride 2048 → 12.5%
    # 16× fewer DMA packets, 4× better fill ratio.
    # =========================================================================
    Wo_reshaped = Wo.reshape((Hq_tp, d, H_wo))  # logical [8, 128, 2048] view

    wo_sbuf = []
    for head in nl.affine_range(HQ_TP_CONST):
        wo_tile = nl.ndarray((PMAX, H_wo), dtype=nl.bfloat16, buffer=nl.sbuf,
                             name=f"wo_tile_h{head}")
        nisa.dma_copy(
            dst=wo_tile,
            src=Wo_reshaped.ap(
                pattern=[[H_wo, PMAX], [1, H_wo]],   # partition: stride H_wo, free: contiguous
                offset=head * PMAX * H_wo,             # skip head * [128 × 2048] elements
            ),
        )
        wo_sbuf.append(wo_tile)

    # =========================================================================
    # K PROJECTION (Hkv_tp=1, one KV head)
    # Wide row load: load entire Wk row [128, 2048] in 16 tiles of [128, 128],
    # then matmul each tile with corresponding hidden tile.
    # =========================================================================
    wk_full = nl.ndarray((PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="wk_full")
    nisa.dma_copy(dst=wk_full, src=Wk)
    k_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="k_psum")
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(k_psum, stationary=wk_full[0:PMAX, h_t*PMAX:(h_t+1)*PMAX], moving=h_all[0:PMAX, h_t:h_t+1])

    k_vec = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_vec")
    nisa.tensor_copy(k_vec, k_psum)

    # K RMSNorm
    k_sq = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_sq")
    nisa.tensor_tensor(k_sq, k_vec, k_vec, op=nl.multiply)
    k_sq_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf, name="k_sq_bf16")
    nisa.tensor_copy(k_sq_bf16, k_sq)
    k_sum_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="k_sum_psum")
    nisa.nc_matmul(k_sum_psum, stationary=rms_ones, moving=k_sq_bf16)
    k_sum_sb = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_sum_sb")
    nisa.tensor_copy(k_sum_sb, k_sum_psum)
    # FALLBACK: nisa.activation(bias=float) rejected by compiler ("expecting tensor access, got float").
    # Keeping original two-instruction form.
    k_mean_sq = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_mean_sq")
    nisa.tensor_scalar(k_mean_sq, k_sum_sb, op0=nl.multiply, operand0=1.0/d, op1=nl.add, operand1=EPS)
    k_rms_inv = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_rms_inv")
    nisa.activation(k_rms_inv, op=nl.rsqrt, data=k_mean_sq)
    k_normed = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_normed")
    nisa.tensor_tensor(k_normed, k_vec, k_rms_inv, op=nl.multiply)
    k_normed2 = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_normed2")
    nisa.tensor_tensor(k_normed2, k_normed, knw_sb, op=nl.multiply)

    # K RoPE
    rot_k = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="rot_k")
    neg_k_upper = nl.ndarray((half_d, B), dtype=nl.float32, buffer=nl.sbuf, name="neg_k_upper")
    nisa.tensor_scalar(neg_k_upper, k_normed2[half_d:d, 0:B], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(rot_k[0:half_d, 0:B], neg_k_upper)
    nisa.tensor_copy(rot_k[half_d:d, 0:B], k_normed2[0:half_d, 0:B])
    k_cos = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_cos")
    k_sin_part = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_sin_part")
    nisa.tensor_tensor(k_cos, k_normed2, cos_f32, op=nl.multiply)
    nisa.tensor_tensor(k_sin_part, rot_k, sin_f32, op=nl.multiply)
    k_rope = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="k_rope")
    nisa.tensor_tensor(k_rope, k_cos, k_sin_part, op=nl.add)

    # Store k_rope to HBM for KV cache update. k_rope is [PMAX, B] f32 SBUF.
    # Transpose to [B, PMAX] = [1, 128] so DMA can write one contiguous packet.
    # k_rope_out is (B, d) = (1, 128) in HBM. Callers reshape to [B, 1, 1, d].
    k_rope_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf, name="k_rope_bf16")
    nisa.tensor_copy(k_rope_bf16, k_rope)
    k_rope_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name="k_rope_T_psum")
    nisa.nc_transpose(k_rope_T_psum, k_rope_bf16)
    k_rope_T_sb = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="k_rope_T_sb")
    nisa.tensor_copy(k_rope_T_sb, k_rope_T_psum)
    nisa.dma_copy(dst=k_rope_out, src=k_rope_T_sb)

    # =========================================================================
    # V PROJECTION (Hkv_tp=1)
    # =========================================================================
    wv_full = nl.ndarray((PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="wv_full")
    nisa.dma_copy(dst=wv_full, src=Wv)
    v_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="v_psum")
    for h_t in nl.affine_range(NUM_H_TILES):
        nisa.nc_matmul(v_psum, stationary=wv_full[0:PMAX, h_t*PMAX:(h_t+1)*PMAX], moving=h_all[0:PMAX, h_t:h_t+1])

    v_active = nl.ndarray((PMAX, B), dtype=nl.float32, buffer=nl.sbuf, name="v_active")
    nisa.tensor_copy(v_active, v_psum)

    # Store v_active to HBM for KV cache update. v_active is [PMAX, B] f32 SBUF.
    # Transpose to [B, PMAX] = [1, 128] so DMA can write one contiguous packet.
    # v_out is (B, d) = (1, 128) in HBM. Callers reshape to [B, 1, 1, d].
    v_bf16 = nl.ndarray((PMAX, B), dtype=nl.bfloat16, buffer=nl.sbuf, name="v_bf16")
    nisa.tensor_copy(v_bf16, v_active)
    v_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name="v_T_psum")
    nisa.nc_transpose(v_T_psum, v_bf16)
    v_T_sb = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="v_T_sb")
    nisa.tensor_copy(v_T_sb, v_T_psum)
    nisa.dma_copy(dst=v_out, src=v_T_sb)

    # =========================================================================
    # Q PROJECTIONS — Change 1: one head at a time to reduce peak SBUF.
    # Instead of hoisting all 8 wq_head[128,2048] tiles simultaneously (~4MB),
    # we load, compute, and pack each head sequentially (peak: 1×512KB).
    # Fuses the psum→q_vec→q_packed_f32 chain into a single tensor_copy per head.
    # =========================================================================
    q_packed_f32 = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_packed_f32")
    for q_h in nl.affine_range(HQ_TP_CONST):
        # Load one head's weight row [128, 2048] — released after tensor_copy below
        wq_head = nl.ndarray((PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"wq_head_{q_h}")
        nisa.dma_copy(dst=wq_head, src=Wq[q_h * PMAX:(q_h + 1) * PMAX, :])
        # Accumulate matmul over all 16 hidden tiles into psum [PMAX, B=1]
        q_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name=f"q_psum_{q_h}")
        for h_t in nl.affine_range(NUM_H_TILES):
            nisa.nc_matmul(
                q_psum,
                stationary=wq_head[0:PMAX, h_t * PMAX:(h_t + 1) * PMAX],
                moving=h_all[0:PMAX, h_t:h_t + 1],
            )
        # Directly copy psum → q_packed_f32[:, q_h] — skips intermediate q_vec buffer
        nisa.tensor_copy(q_packed_f32[0:PMAX, q_h:q_h + 1], q_psum)

    # =========================================================================
    # PACKED Q RMSNORM on [PMAX, GQA=8]
    # =========================================================================
    q_sq = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_sq")
    nisa.tensor_tensor(q_sq, q_packed_f32, q_packed_f32, op=nl.multiply)
    q_sq_bf16 = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name="q_sq_bf16")
    nisa.tensor_copy(q_sq_bf16, q_sq)
    q_sum_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="q_sum_psum")
    nisa.nc_matmul(q_sum_psum, stationary=rms_ones, moving=q_sq_bf16)
    q_sum_sb = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_sum_sb")
    nisa.tensor_copy(q_sum_sb, q_sum_psum)
    # FALLBACK: nisa.activation(bias=float) rejected by compiler ("expecting tensor access, got float").
    # Keeping original two-instruction form.
    q_mean_sq = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_mean_sq")
    nisa.tensor_scalar(q_mean_sq, q_sum_sb, op0=nl.multiply, operand0=1.0/d, op1=nl.add, operand1=EPS)
    q_rms_inv = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_rms_inv")
    nisa.activation(q_rms_inv, op=nl.rsqrt, data=q_mean_sq)
    q_normed = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_normed")
    nisa.tensor_tensor(q_normed, q_packed_f32, q_rms_inv, op=nl.multiply)

    # Apply norm weight: qnw_sb [PMAX, 1] broadcast to [PMAX, GQA] before multiply.
    # for-loop broadcast (8 tensor_copy calls) — same as v6_ultimate.
    # tp_broadcast: qnw_sb[PMAX=128, 1] → qnw_gqa[PMAX=128, GQA=8]
    qnw_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="qnw_gqa_psum_T")
    nisa.nc_transpose(qnw_gqa_psum_T, qnw_sb.ap([[1, PMAX], [0, GQA]], offset=0))
    qnw_gqa_sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="qnw_gqa_sbuf_T")
    nisa.tensor_copy(qnw_gqa_sbuf_T, qnw_gqa_psum_T)
    qnw_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="qnw_gqa_psum")
    nisa.nc_transpose(qnw_gqa_psum, qnw_gqa_sbuf_T)
    qnw_gqa = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="qnw_gqa")
    nisa.tensor_copy(qnw_gqa, qnw_gqa_psum)

    q_normed2 = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_normed2")
    nisa.tensor_tensor(q_normed2, q_normed, qnw_gqa, op=nl.multiply)

    # =========================================================================
    # PACKED Q ROPE on [PMAX, GQA=8]
    # =========================================================================
    # tp_broadcast: cos_f32[PMAX=128, 1] → cos_gqa[PMAX=128, GQA=8]
    # Step 1: ap(stride_f=0) + nc_transpose → [GQA=8, PMAX=128] transposed
    # Step 2: nc_transpose back → [PMAX=128, GQA=8]
    # Replaces 8 tensor_copy with 2 nc_transpose + 2 tensor_copy per tensor.
    cos_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="cos_gqa_psum_T")
    nisa.nc_transpose(cos_gqa_psum_T, cos_f32.ap([[1, PMAX], [0, GQA]], offset=0))
    cos_gqa_sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="cos_gqa_sbuf_T")
    nisa.tensor_copy(cos_gqa_sbuf_T, cos_gqa_psum_T)
    cos_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="cos_gqa_psum")
    nisa.nc_transpose(cos_gqa_psum, cos_gqa_sbuf_T)
    cos_gqa = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="cos_gqa")
    nisa.tensor_copy(cos_gqa, cos_gqa_psum)

    sin_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="sin_gqa_psum_T")
    nisa.nc_transpose(sin_gqa_psum_T, sin_f32.ap([[1, PMAX], [0, GQA]], offset=0))
    sin_gqa_sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="sin_gqa_sbuf_T")
    nisa.tensor_copy(sin_gqa_sbuf_T, sin_gqa_psum_T)
    sin_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="sin_gqa_psum")
    nisa.nc_transpose(sin_gqa_psum, sin_gqa_sbuf_T)
    sin_gqa = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="sin_gqa")
    nisa.tensor_copy(sin_gqa, sin_gqa_psum)

    rot_q = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="rot_q")
    neg_q_upper = nl.ndarray((half_d, GQA), dtype=nl.float32, buffer=nl.sbuf, name="neg_q_upper")
    nisa.tensor_scalar(neg_q_upper, q_normed2[half_d:d, 0:GQA], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(rot_q[0:half_d, 0:GQA], neg_q_upper)
    nisa.tensor_copy(rot_q[half_d:d, 0:GQA], q_normed2[0:half_d, 0:GQA])

    q_cos = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_cos")
    q_sin_part = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_sin_part")
    nisa.tensor_tensor(q_cos, q_normed2, cos_gqa, op=nl.multiply)
    nisa.tensor_tensor(q_sin_part, rot_q, sin_gqa, op=nl.multiply)
    q_rope = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="q_rope")
    nisa.tensor_tensor(q_rope, q_cos, q_sin_part, op=nl.add)

    # Scale by 1/sqrt(d) and cast to bf16 — this is the "scaled Q" for flash decode
    q_bf16 = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name="q_bf16")
    nisa.tensor_scalar(q_bf16, q_rope, op0=nl.multiply, operand0=INV_SQRT_D)

    # =========================================================================
    # TWO-PASS FLASH DECODE
    # =========================================================================
    K_cache_2d = K_cache.reshape((S_prior, d))   # [S_prior, 128]
    V_cache_2d = V_cache.reshape((S_prior, d))   # [S_prior, 128]

    # --- Active position score: k_rope [PMAX,1] dot q_scaled [PMAX,GQA] ---
    # tp_broadcast: k_rope[PMAX=128, 1] → k_rope_packed[PMAX=128, GQA=8]
    k_rope_packed_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="k_rope_packed_psum_T")
    nisa.nc_transpose(k_rope_packed_psum_T, k_rope.ap([[1, PMAX], [0, GQA]], offset=0))
    k_rope_packed_sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="k_rope_packed_sbuf_T")
    nisa.tensor_copy(k_rope_packed_sbuf_T, k_rope_packed_psum_T)
    k_rope_packed_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="k_rope_packed_psum")
    nisa.nc_transpose(k_rope_packed_psum, k_rope_packed_sbuf_T)
    k_rope_packed = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="k_rope_packed")
    nisa.tensor_copy(k_rope_packed, k_rope_packed_psum)

    kq_elem = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="kq_elem")
    nisa.tensor_tensor(kq_elem, k_rope_packed, q_bf16, op=nl.multiply)
    kq_elem_bf16 = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name="kq_elem_bf16")
    nisa.tensor_copy(kq_elem_bf16, kq_elem)
    score_active_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="score_active_psum")
    nisa.nc_matmul(score_active_psum, stationary=rms_ones, moving=kq_elem_bf16)
    score_active = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="score_active")
    nisa.tensor_copy(score_active, score_active_psum)

    # =========================================================================
    # v10e: Load position scalar once for masking
    # position_ids [B, 1] int32 → pos_sb [1, 1] int32 → pos_f32 [1, 1] f32
    # pos = number of valid tokens currently in the K/V cache.
    # All cache rows with global index >= pos are masked to -1e9.
    # =========================================================================
    # Reshape position_ids to [B, 1] = [1, 1]; load into SBUF as int32
    position_ids_2d = position_ids.reshape((B, 1))
    pos_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name="pos_sb")
    nisa.dma_copy(dst=pos_sb, src=position_ids_2d[0:1, 0:1])
    # Cast int32 → float32 for arithmetic below
    pos_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf, name="pos_f32")
    nisa.tensor_copy(pos_f32, pos_sb)

    # =========================================================================
    # v10e: Build partition index [PMAX, 1] = [0.0, 1.0, ..., 127.0]
    # nisa.iota fills dst such that dst[p, 0] = p (hardware partition index).
    # This is used to compute per-row deltas for threshold masking below.
    # Built once here and reused across all NUM_S_TILES iterations.
    # =========================================================================
    # nisa.iota generates: dst[channel_id, 0] = offset + channel_id * channel_multiplier
    # With offset=0, channel_multiplier=1, pattern=[[1,1]]: dst[p, 0] = p (0..127).
    # The GpSimd engine casts the integer result to the dst dtype (float32 here).
    par_index_f32 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="par_index_f32")
    nisa.iota(par_index_f32, pattern=[[1, 1]], offset=0, channel_multiplier=1)

    # =========================================================================
    # Plan B — K-cache contiguous load + PE transpose
    # Hoist all K-cache tiles into SBUF before pass 1 — reused in both passes.
    # v10e: compute on-chip mask_tile_f32 [PMAX, 1] from position_ids threshold.
    # =========================================================================
    k_cache_tiles = []
    mask_tiles = []   # [PMAX, 1] f32 per tile: -1e9 for future/padding, 0 for valid
    for s_t in nl.affine_range(NUM_S_TILES):
        # Step 1: Load K tile as natural [S_tile, d] row-major — 1 contiguous 32KB packet per tile
        # k_raw[p, f] = K_cache_2d[s_t*128 + p, f]  (no stride, no scatter)
        k_raw = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"k_raw_{s_t}")
        nisa.dma_copy(dst=k_raw, src=K_cache_2d[s_t * PMAX:(s_t + 1) * PMAX, :])

        # Step 2: PE transpose to get k_ct[p, f] = K_cache_2d[s_t*128 + f, p]
        # nc_transpose maps [P, F] → [F, P]: k_ct_psum[p_out, f_out] = k_raw[f_out, p_out]
        #   = K_cache_2d[s_t*128 + f_out, p_out]  — identical to original ap() result
        # CoreV3+ requires matching dtype for nc_transpose: use bf16 psum.
        k_ct_psum = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name=f"k_ct_psum_{s_t}")
        nisa.nc_transpose(k_ct_psum, k_raw)
        k_ct = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"k_ct_{s_t}")
        nisa.tensor_copy(k_ct, k_ct_psum)

        k_cache_tiles.append(k_ct)

        # ── v10e: Position-id threshold mask ────────────────────────────────
        # tile_start = s_t * PMAX is a compile-time Python int (nl.affine_range
        # unrolls statically, so s_t is a constant per loop iteration).
        # For local row p, global index = tile_start + p.
        # valid iff (tile_start + p) < pos  ⟺  p < (pos - tile_start).
        #
        # threshold_local = pos_f32 - tile_start  [1, 1] f32
        # delta[p] = par_index_f32[p] - threshold_local
        #            < 0 for valid rows, >= 0 for future/padding rows
        # relu_delta[p] = max(delta[p], 0) → 0 for valid, positive for invalid
        # clamped[p] = min(relu_delta[p], 1.0) → binary {0, 1}
        # mask_tile_f32[p] = clamped[p] * (-1e9) → 0 for valid, -1e9 for invalid
        tile_start = s_t * PMAX  # Python int — compile-time constant per iteration

        # Op 1: neg_threshold[0,0] = tile_start - pos  (scalar, [1,1] f32)
        # delta[p] = p - (pos - tile_start) = p + (tile_start - pos) = p + neg_threshold
        # Use two-op tensor_scalar: pos_f32 * (-1) + tile_start
        neg_threshold = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"neg_threshold_{s_t}")
        nisa.tensor_scalar(neg_threshold, pos_f32,
                           op0=nl.multiply, operand0=-1.0,
                           op1=nl.add, operand1=float(tile_start))

        # Op 2a: broadcast neg_threshold [1,1] → [PMAX,1] using nc_transpose pattern.
        # ap([[1,1],[0,PMAX]]): 1 partition, PMAX free copies (step=0 repeats the value).
        # nc_transpose [1,PMAX] → [PMAX,1]: each of PMAX partitions gets the same value.
        # This is identical to the neg_max_g1 broadcast pattern used elsewhere in the kernel.
        neg_thresh_psum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum, name=f"neg_thresh_psum_{s_t}")
        nisa.nc_transpose(neg_thresh_psum, neg_threshold.ap([[1, 1], [0, PMAX]], offset=0))
        neg_thresh_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"neg_thresh_sb_{s_t}")
        nisa.tensor_copy(neg_thresh_sb, neg_thresh_psum)

        # Op 2b: per-row delta = par_index_f32 + neg_thresh_sb (both [PMAX,1])
        # delta[p] = p + (tile_start - pos) → negative for valid rows (p < pos-tile_start)
        delta = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"delta_{s_t}")
        nisa.tensor_tensor(delta, par_index_f32, neg_thresh_sb, op=nl.add)

        # Op 3: relu — zero out valid rows (delta < 0), keep positive for invalid rows
        relu_delta = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"relu_delta_{s_t}")
        nisa.activation(relu_delta, op=nl.relu, data=delta)

        # Op 4: clamp to [0, 1] — binary step function
        clamped = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"clamped_{s_t}")
        nisa.tensor_scalar(clamped, relu_delta, op0=nl.minimum, operand0=1.0)

        # Op 5: scale to -1e9 — valid rows: 0, future/padding rows: -1e9
        mask_tile_f32 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"mask_tile_f32_{s_t}")
        nisa.tensor_scalar(mask_tile_f32, clamped, op0=nl.multiply, operand0=-1e9)

        mask_tiles.append(mask_tile_f32)

    # Hoist all V-cache tiles into SBUF before pass 2.
    v_cache_tiles = []
    for s_t in nl.affine_range(NUM_S_TILES):
        v_ct = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"v_ct_{s_t}")
        nisa.dma_copy(
            dst=v_ct,
            src=V_cache_2d.ap(pattern=[[d, PMAX], [1, d]], offset=s_t * PMAX * d),
        )
        v_cache_tiles.append(v_ct)

    # --- Pass 1: find global max scalar across all K tiles + active position ---
    global_max_g1 = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="global_max_g1")
    nisa.memset(global_max_g1, value=-1e9)

    # score_active [PMAX, GQA] → transpose → [GQA, PMAX] → reduce max over axis=1 → [GQA, 1]
    score_act_T_psum = nl.zeros((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="score_act_T_psum")
    nisa.nc_transpose(score_act_T_psum, score_active)
    score_act_T_sb = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="score_act_T_sb")
    nisa.tensor_copy(score_act_T_sb, score_act_T_psum)
    score_active_g1 = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="score_active_g1")
    nisa.tensor_reduce(dst=score_active_g1, op=nl.maximum, data=score_act_T_sb, axis=1)
    nisa.tensor_tensor(global_max_g1, global_max_g1, score_active_g1, op=nl.maximum)

    # ── Plan B: saved_scores list for Pass 2 reuse ─────────────────────────────
    # Collect score_sb_masked from Pass 1 to avoid recomputing K×Q matmul in Pass 2.
    # Memory cost: GQA * PMAX * 4 bytes = 4 KB per tile, 20 KB for S_prior=640.
    saved_scores = []

    for s_t in nl.affine_range(NUM_S_TILES):
        # score [PMAX, GQA]: K_tile[PMAX,PMAX] @ q_bf16[PMAX,GQA]
        # name= suffixes use s_t to keep each iteration's tensor name unique —
        # the NKI compiler requires unique names even inside affine_range loops.
        score_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"score_psum_{s_t}")
        nisa.nc_matmul(score_psum, stationary=k_cache_tiles[s_t], moving=q_bf16) # Depends on DMA
        score_sb = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score_sb_{s_t}")
        nisa.tensor_copy(score_sb, score_psum)

        # ── v10e: On-chip mask from mask_tiles[s_t] (computed in hoisting loop) ──
        # mask_tiles[s_t] is [PMAX, 1] f32: 0 for valid rows, -1e9 for future/padding.

        # Broadcast [PMAX, 1] → [PMAX, GQA] using double-nc_transpose pattern
        # (same pattern as qnw_gqa broadcast, lines 327-334 in v10b)
        mask_gqa_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name=f"mask_gqa_psum_T_{s_t}")
        nisa.nc_transpose(mask_gqa_psum_T, mask_tiles[s_t].ap([[1, PMAX], [0, GQA]], offset=0))
        mask_gqa_sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name=f"mask_gqa_sbuf_T_{s_t}")
        nisa.tensor_copy(mask_gqa_sbuf_T, mask_gqa_psum_T)
        mask_gqa_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"mask_gqa_psum_{s_t}")
        nisa.nc_transpose(mask_gqa_psum, mask_gqa_sbuf_T)
        mask_gqa = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"mask_gqa_{s_t}")
        nisa.tensor_copy(mask_gqa, mask_gqa_psum)

        # Apply mask: future/padding positions get score -1e9, exp(-1e9 - max) ≈ 0
        score_sb_masked = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score_sb_masked_{s_t}")
        nisa.tensor_tensor(score_sb_masked, score_sb, mask_gqa, op=nl.add)

        saved_scores.append(score_sb_masked)   # cache masked score — reused in Pass 2, no re-matmul

        # Per-tile max reduction: transpose [PMAX,GQA] → [GQA,PMAX], reduce max → [GQA,1]
        score_T_psum = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name=f"score_T_psum_{s_t}")
        nisa.nc_transpose(score_T_psum, score_sb_masked)
        score_T_sb = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name=f"score_T_sb_{s_t}")
        nisa.tensor_copy(score_T_sb, score_T_psum)

        tile_max_vec = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"tile_max_vec_{s_t}")
        nisa.tensor_reduce(dst=tile_max_vec, op=nl.maximum, data=score_T_sb, axis=1)

        # tile_max_vec [GQA, 1] and global_max_g1 [GQA, 1] — direct max, no broadcast needed.
        nisa.tensor_tensor(global_max_g1, global_max_g1, tile_max_vec, op=nl.maximum)

    # Negate compact global max.
    neg_max_g1 = nl.ndarray((GQA, 1), dtype=nl.float32, buffer=nl.sbuf, name="neg_max_g1")
    nisa.tensor_scalar(neg_max_g1, global_max_g1, op0=nl.multiply, operand0=-1.0)

    # tp_broadcast: neg_max_g1[GQA=8, 1] → neg_max[PMAX=128, GQA=8]
    # Adopted from nkilib/core/utils/tp_broadcast.py production pattern.
    # ap([[1, GQA], [0, PMAX]], offset=0): indexed[p, f] = flat[p*1 + f*0] = flat[p]
    # → 8 partition values each broadcast across all PMAX=128 free columns.
    # nc_transpose reads the [GQA=8, PMAX=128] view and writes psum[PMAX=128, GQA=8].
    # Replaces 128-loop + nc_transpose + tensor_copy (~130 instr) with 2 instructions.
    neg_max_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="neg_max_psum")
    nisa.nc_transpose(
        neg_max_psum,
        neg_max_g1.ap([[1, GQA], [0, PMAX]], offset=0),
    )
    neg_max = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="neg_max")
    nisa.tensor_copy(neg_max, neg_max_psum)

    # --- Pass 2: use saved scores (no K matmul), exp(score - global_max), accumulate V ---
    v_acc = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="v_acc")
    nisa.memset(v_acc, value=0.0)
    sum_acc = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="sum_acc")
    nisa.memset(sum_acc, value=0.0)

    for s_t in nl.affine_range(NUM_S_TILES):
        # ── Plan B: No K matmul — reuse masked score cached from Pass 1 ──────────────
        # saved_scores[s_t] is already masked (score_sb_masked); the nc_matmul+tensor_copy
        # from Pass 1 is gone — for S_prior=640 this removes 5 matmuls.
        # Unique name= suffixes required by the NKI compiler across loop iterations.
        score2_shifted = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score2_shifted_{s_t}")
        nisa.tensor_tensor(score2_shifted, saved_scores[s_t], neg_max, op=nl.add)

        score2_exp = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"score2_exp_{s_t}")
        nisa.activation(score2_exp, op=nl.exp, data=score2_shifted)

        score2_exp_bf16 = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"score2_exp_bf16_{s_t}")
        nisa.tensor_copy(score2_exp_bf16, score2_exp)

        # Accumulate softmax denominator: rms_ones @ score2_exp_bf16 → [PMAX, GQA] row-sum
        tile_sum_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"tile_sum_psum_{s_t}")
        nisa.nc_matmul(tile_sum_psum, stationary=rms_ones, moving=score2_exp_bf16)
        tile_sum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"tile_sum_{s_t}")
        nisa.tensor_copy(tile_sum, tile_sum_psum)
        nisa.tensor_tensor(sum_acc, sum_acc, tile_sum, op=nl.add)

        # V-weighted accumulation: stationary=v_cache_tiles[s_t], moving=score2_exp_bf16
        v_weighted_psum = nl.zeros((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name=f"v_weighted_psum_{s_t}")
        nisa.nc_matmul(v_weighted_psum, stationary=v_cache_tiles[s_t], moving=score2_exp_bf16)
        v_weighted = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name=f"v_weighted_{s_t}")
        nisa.tensor_copy(v_weighted, v_weighted_psum)
        nisa.tensor_tensor(v_acc, v_acc, v_weighted, op=nl.add)

    # --- Active position contribution ---
    score_act_shifted = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="score_act_shifted")
    nisa.tensor_tensor(score_act_shifted, score_active, neg_max, op=nl.add)
    score_act_exp = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="score_act_exp")
    nisa.activation(score_act_exp, op=nl.exp, data=score_act_shifted)
    nisa.tensor_tensor(sum_acc, sum_acc, score_act_exp, op=nl.add)

    # tp_broadcast: v_active[PMAX=128, 1] → v_act_packed[PMAX=128, GQA=8]
    v_act_packed_psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum, name="v_act_packed_psum_T")
    nisa.nc_transpose(v_act_packed_psum_T, v_active.ap([[1, PMAX], [0, GQA]], offset=0))
    v_act_packed_sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf, name="v_act_packed_sbuf_T")
    nisa.tensor_copy(v_act_packed_sbuf_T, v_act_packed_psum_T)
    v_act_packed_psum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum, name="v_act_packed_psum")
    nisa.nc_transpose(v_act_packed_psum, v_act_packed_sbuf_T)
    v_act_packed = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="v_act_packed")
    nisa.tensor_copy(v_act_packed, v_act_packed_psum)

    v_act_weighted = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="v_act_weighted")
    nisa.tensor_tensor(v_act_weighted, v_act_packed, score_act_exp, op=nl.multiply)
    nisa.tensor_tensor(v_acc, v_acc, v_act_weighted, op=nl.add)

    # --- Normalize: attn_out = v_acc / sum_acc ---
    # FALLBACK: nisa.activation(bias=float) rejected by compiler ("expecting tensor access, got float").
    # Keeping original two-instruction form.
    sum_safe = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="sum_safe")
    nisa.tensor_scalar(sum_safe, sum_acc, op0=nl.add, operand0=1e-9)
    # Use rsqrt trick: 1/x = rsqrt(x)^2 (avoids a native divide instruction)
    rsqrt_sum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="rsqrt_sum")
    nisa.activation(rsqrt_sum, op=nl.rsqrt, data=sum_safe)
    inv_sum = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf, name="inv_sum")
    nisa.tensor_tensor(inv_sum, rsqrt_sum, rsqrt_sum, op=nl.multiply)

    # Cast attention output to bf16 for the matmul stationary operand
    attn_out = nl.ndarray((PMAX, GQA), dtype=nl.bfloat16, buffer=nl.sbuf, name="attn_out")
    nisa.tensor_tensor(attn_out, v_acc, inv_sum, op=nl.multiply)

    # =========================================================================
    # FUSED OUTPUT PROJECTION — Change 2: head-outer, h_blk-inner loop order.
    #
    # Pre-allocate all 4 output PSUMs upfront so all h_blk blocks accumulate
    # simultaneously across the head loop. Each head's wo_sbuf is fully consumed
    # (all 4 blocks) before moving to the next head — better SBUF access locality.
    # =========================================================================
    res_psum_0 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_0")
    res_psum_1 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_1")
    res_psum_2 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_2")
    res_psum_3 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_3")
    for head in nl.affine_range(HQ_TP_CONST):
        # All 4 output blocks for this head — fully consumes wo_sbuf[head] before next head
        nisa.nc_matmul(res_psum_0, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 0*F_MAX:1*F_MAX])
        nisa.nc_matmul(res_psum_1, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 1*F_MAX:2*F_MAX])
        nisa.nc_matmul(res_psum_2, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 2*F_MAX:3*F_MAX])
        nisa.nc_matmul(res_psum_3, stationary=attn_out[0:PMAX, head:head+1], moving=wo_sbuf[head][0:PMAX, 3*F_MAX:4*F_MAX])
    # Store all 4 output blocks after the head loop completes
    out_sb_0 = nl.ndarray((1, F_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_sb_0")
    nisa.tensor_copy(out_sb_0, res_psum_0)
    nisa.dma_copy(dst=output_2d[0:1, 0*F_MAX:1*F_MAX], src=out_sb_0[0:1, 0:F_MAX])
    out_sb_1 = nl.ndarray((1, F_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_sb_1")
    nisa.tensor_copy(out_sb_1, res_psum_1)
    nisa.dma_copy(dst=output_2d[0:1, 1*F_MAX:2*F_MAX], src=out_sb_1[0:1, 0:F_MAX])
    out_sb_2 = nl.ndarray((1, F_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_sb_2")
    nisa.tensor_copy(out_sb_2, res_psum_2)
    nisa.dma_copy(dst=output_2d[0:1, 2*F_MAX:3*F_MAX], src=out_sb_2[0:1, 0:F_MAX])
    out_sb_3 = nl.ndarray((1, F_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_sb_3")
    nisa.tensor_copy(out_sb_3, res_psum_3)
    nisa.dma_copy(dst=output_2d[0:1, 3*F_MAX:4*F_MAX], src=out_sb_3[0:1, 0:F_MAX])

    return output, k_rope_out, v_out



# Hardware constants
_PMAX = 128       # partition dimension max
_PSUM_FREE = 512  # PSUM free-dimension max on trn2

# Qwen3-30B-A3B at TP=4 fixed dims
_H = 2048    # hidden dim
_E = 128     # num experts
_K = 8       # top-K experts
_I = 192     # actual intermediate dim per TP rank (no padding anywhere)
_I0 = 128    # first tile  (full 128 rows)
_I1 = 64     # second tile (partial: 64 valid rows, 64 zero-padded)
_I_TILES = 2 # two I-dimension tiles
_EPS = 1e-6

# Flat gate+up combined width (native layout: gate cols 0:I, up cols I:2*I)
_GU_FLAT = 2 * _I   # = 384  (stride_p for coalesced DMA: 384 == count_col 384)

# LNC=2 H-sharding constants (always launched with [2])
_N_PRGS = 2
_H_FREE = _H // _PMAX             # = 16 tiles of 128 each
_H_FREE_SHARD = _H_FREE // _N_PRGS   # = 8  (each core owns 8 H-tiles for output)
_H_SHARD = _H_FREE_SHARD * _PMAX     # = 1024

# Router DMA batching: 4 tiles per DMA → 4×32KB = 128KB per packet
_ROUTER_BATCH = 4  # H_FREE must be divisible by this

@nki.jit(platform_target="trn2")
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
        kwargs["attn_tkg_nki_kernel_enabled"] = True
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
            moe_out = qwen3_moe_fused_tkg[2](
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
