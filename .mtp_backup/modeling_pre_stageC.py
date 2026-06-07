"""NxDI port of Qwen3.6-35B-A3B (text stack).

Architecture
------------
40 decoder layers in a repeating [DN, DN, DN, GQA] x 10 pattern:
  * 30 Gated DeltaNet layers (linear recurrent attention, causal conv1d on QKV,
    delta-rule update, gated output)
  * 10 Standard GQA layers (16 Q heads, 2 KV heads, head_dim=256, partial RoPE
    on 64 of 256 dims, sigmoid output gate)

Every layer's FFN is a MoE block: 256 routed experts (top-8) + 1 always-on
shared expert, both with intermediate dim 512. Routing uses fp32 softmax
with normalized top-k affinities.

A multi-token-prediction (MTP) head sits alongside the main stack -- one
extra decoder layer + RMSNorms + an LM head -- and is invoked off the main
traced forward as a self-speculative drafter.

State / caching
---------------
GQA layers use NxDI's KVCacheManager.
DeltaNet layers carry their recurrent state + 1-D conv state as nn.Parameter
buffers and return dummy KV tuples from forward.
"""

import gc
import math
import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode

try:
    from nki import jit as nki_jit  # NKI 0.3.0+ (SDK 2.29)
except ImportError:
    from torch_neuronx.xla_impl.ops import nki_jit  # NKI 0.2.x (SDK 2.28)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from .nki_kernels.nki_deltanet import deltanet_recurrent_fwd as _deltanet_nki_kernel
from .nki_kernels.nki_deltanet import (
    deltanet_recurrent_fwd_state as _deltanet_nki_kernel_state,
)
from .nki_kernels.nki_deltanet_chunked import (
    deltanet_chunk_step as _deltanet_nki_chunk_step,
)
from .nki_kernels.nki_deltanet_fused import (
    deltanet_fused_chunked_fwd as _deltanet_fused_kernel,
)
from .nki_kernels.nki_deltanet_fused import (
    _make_lower_mask,
    _make_lower_mask_diag,
    _make_identity,
)

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
)
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    DecoderModelInstance,
    ModelWrapper,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

logger = logging.getLogger(__name__)

# Tag for the on-device MTP draft head, traced as a third compiled graph
# alongside the context-encoding and token-generation graphs.
MTP_HEAD_MODEL_TAG = "mtp_draft_head_model"

# Tag for the verify backbone graph (n_active=2), traced as an additional
# compiled graph that runs the 2-token block seeded from committed decode state
# and emits per-position logits. See NeuronVerifyModel / VerifyModelWrapper.
VERIFY_MODEL_TAG = "verify_backbone_model"

try:
    _flash_fwd_call = nki_jit()(attention_isa_kernel)
except TypeError:
    from torch_neuronx.xla_impl.ops import nki_jit as _torch_xla_nki_jit

    _flash_fwd_call = _torch_xla_nki_jit()(attention_isa_kernel)

# Option B: Direct nkilib flash attention for head_dim > 128
USE_NKILIB_KERNEL = os.environ.get("USE_NKILIB_KERNEL", "0") == "1"

_nkilib_flash_attn = None
if USE_NKILIB_KERNEL:
    try:
        import neuronxcc.nki as _nki
        from neuronx_distributed_inference.modules.attention.attention_base import (
            peel_decorations as _peel_decorations,
            get_platform_target as _get_platform_target,
        )
        from neuronxcc.nki.compiler import (
            skip_middle_end_transformations as _skip_middle_end,
            enable_stack_allocator as _enable_stack_allocator,
        )

        import importlib

        _fork_path = "/home/ubuntu/nki-library-fork/nkilib_src"
        if os.path.isdir(_fork_path) and _fork_path not in sys.path:
            sys.path.insert(0, _fork_path)
        _to_remove = [k for k in sys.modules if k.startswith("nkilib")]
        for k in _to_remove:
            del sys.modules[k]
        import nki.language as _stub_nl
        import neuronxcc.nki.language as _real_nl

        for _attr in [
            "NKIObject",
            "float8_e4m3fn",
            "float8_e4m3fn_x4",
            "float8_e5m2_x4",
            "float4_e2m1fn_x4",
        ]:
            if not hasattr(_real_nl, _attr) and hasattr(_stub_nl, _attr):
                setattr(_real_nl, _attr, getattr(_stub_nl, _attr))
        from nkilib.core.attention.attention_cte import (
            attention_cte as _attention_cte_raw,
            _MAX_HEAD_DIM,
        )

        assert _MAX_HEAD_DIM == 256, (
            f"nkilib fork has _MAX_HEAD_DIM={_MAX_HEAD_DIM}, expected 256. "
            f"System nkilib may have been loaded instead of fork."
        )
        logger.info(
            f"Loaded nkilib attention_cte from fork (_MAX_HEAD_DIM={_MAX_HEAD_DIM})"
        )

        _raw_fn = _peel_decorations(_attention_cte_raw)
        _platform = _get_platform_target()
        _nkilib_flash_attn = _nki.jit(
            _raw_fn,
            mode="torchxla",
            platform_target=_platform,
            show_compiler_tb=True,
            debug_kernel=True,
        )
        _nkilib_flash_attn = _skip_middle_end(_nkilib_flash_attn)
        _nkilib_flash_attn = _enable_stack_allocator(
            _nkilib_flash_attn, log_level=logging.INFO
        )
        logger.info("Option B: nkilib flash attention loaded for head_dim > 128")
    except Exception as e:
        logger.warning(f"Option B: Failed to load nkilib flash attention: {e}")
        import traceback as _tb

        _tb.print_exc()
        _nkilib_flash_attn = None

# Option A: Detect if patch_attn_kernel was imported
NKILIB_PATCH_ACTIVE = False
try:
    from importlib import import_module as _import_module

    _attn_mod = _import_module("neuronxcc.nki._pre_prod_kernels.attn_fwd")
    if hasattr(_attn_mod, "_original_attention_nki_kernel_adapter"):
        NKILIB_PATCH_ACTIVE = True
        logger.info("Option A detected: _pre_prod_kernels patched with nkilib kernel")
except Exception:
    pass


# ============================================================
# Newton-Raphson Refined RMSNorm
# ============================================================
USE_NEWTON_RMSNORM = os.environ.get("USE_NEWTON_RMSNORM") == "1"
USE_PYTHON_RMSNORM = os.environ.get("USE_PYTHON_RMSNORM") == "1"


class NewtonRMSNorm(nn.Module):
    """RMSNorm with Newton-Raphson refined rsqrt for improved numerical accuracy."""

    def __init__(self, hidden_size=None, eps=1e-6):
        super().__init__()
        self.weight = None
        if hidden_size is not None:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.hidden_size = hidden_size
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        original_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        y = torch.rsqrt(variance + self.variance_epsilon)
        y = y * (3.0 - (variance + self.variance_epsilon) * y * y) * 0.5
        result = x * y
        if self.weight is not None:
            result = result * self.weight.float()
        return result.to(original_dtype)


def get_rmsnorm_cls():
    if cpu_mode() or USE_PYTHON_RMSNORM:
        return Qwen3MoeRMSNorm
    return NewtonRMSNorm if USE_NEWTON_RMSNORM else CustomRMSNorm


def l2norm(x, dim=-1, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)


def commit_accept(
    accept_count,
    recurrent_candidates,
    conv_candidates,
    gqa_kv_block,
    mtp_kv_block=None,
):
    """Host-side reject/update rule for the multi-token-prediction verify block.

    Given ``accept_count`` committed positions (``1`` on reject of the draft,
    ``2`` on accept), select the carried state matching that count and truncate
    the sliceable caches to it. This is pure tensor indexing with no device
    specifics so it can run identically on host and device.

    Parameters
    ----------
    accept_count : int
        Number of verified tokens to commit. ``1`` keeps only the real token
        (draft rejected); ``2`` keeps real + draft (draft accepted). Generalises
        to a ``k+1`` block where ``accept_count`` ranges over ``1..k+1``.
    recurrent_candidates : list of Tensor
        Per-DeltaNet-layer candidate states from
        :meth:`NeuronGatedDeltaNet.verify_block_candidates`, each ``[B, S, H,
        Kd, Vd]`` where axis-1 is the candidate axis (``S_i`` = state after
        consuming token ``i``).
    conv_candidates : list of Tensor
        Per-DeltaNet-layer causal-conv windows from
        :meth:`NeuronGatedDeltaNet.conv_window_candidates`, each ``[B, S,
        conv_dim, K-1]`` with the same candidate axis-1.
    gqa_kv_block : Tensor or sequence of Tensor
        Full-attention KV cache block(s) spanning the verify positions on the
        sequence axis (axis -2 in the usual ``[B, n_heads, S, head_dim]``
        layout). Truncated to the first ``accept_count`` positions.
    mtp_kv_block : Tensor or sequence of Tensor, optional
        Draft-head KV cache block(s), sliced the same way when present.

    Returns
    -------
    dict
        ``{"accept_count", "recurrent_states", "conv_states", "gqa_kv",
        "mtp_kv"}`` where ``recurrent_states``/``conv_states`` are the selected
        per-layer states (candidate axis removed) and the ``*_kv`` entries are
        the truncated cache blocks (``None`` if not supplied).
    """
    # Candidate axis-1 index: accept_count tokens committed -> S_{accept_count-1},
    # the state after consuming the last committed token. At k=1 this is a plain
    # index (S_0 on reject, S_1 on accept).
    cand_idx = accept_count - 1

    recurrent_states = [cand[:, cand_idx] for cand in recurrent_candidates]
    conv_states = [cand[:, cand_idx] for cand in conv_candidates]

    def _truncate_kv(kv):
        if kv is None:
            return None
        if isinstance(kv, (list, tuple)):
            return type(kv)(k[..., :accept_count, :] for k in kv)
        return kv[..., :accept_count, :]

    return {
        "accept_count": accept_count,
        "recurrent_states": recurrent_states,
        "conv_states": conv_states,
        "gqa_kv": _truncate_kv(gqa_kv_block),
        "mtp_kv": _truncate_kv(mtp_kv_block),
    }


# ============================================================
# Gated DeltaNet Module (Linear Recurrent Attention)
# ============================================================


class NeuronGatedDeltaNet(nn.Module):
    """Gated DeltaNet linear-recurrent attention.

    HF weight layout (dimensions parameterised by config):
        in_proj_qkv : (2*key_dim + value_dim, hidden_size)
        in_proj_z   : (value_dim, hidden_size)
        in_proj_a   : (num_v_heads, hidden_size)
        in_proj_b   : (num_v_heads, hidden_size)
        conv1d      : (conv_dim, 1, conv_kernel_size)
        A_log       : (num_v_heads,)
        dt_bias     : (num_v_heads,)
        norm        : (head_v_dim,)
        out_proj    : (hidden_size, value_dim)
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        tc = config

        self.hidden_size = tc.hidden_size
        self.tp_degree = tc.neuron_config.tp_degree
        self.global_num_v_heads = tc.linear_num_value_heads
        self.global_num_k_heads = tc.linear_num_key_heads
        self.head_k_dim = tc.linear_key_head_dim  # 128
        self.head_v_dim = tc.linear_value_head_dim  # 128
        if self.global_num_v_heads % self.tp_degree != 0:
            raise ValueError(
                f"linear_num_value_heads={self.global_num_v_heads} must be divisible "
                f"by tp_degree={self.tp_degree}"
            )
        if self.global_num_k_heads % self.tp_degree != 0:
            raise ValueError(
                f"linear_num_key_heads={self.global_num_k_heads} must be divisible "
                f"by tp_degree={self.tp_degree}"
            )
        self.num_v_heads = self.global_num_v_heads // self.tp_degree
        self.num_k_heads = self.global_num_k_heads // self.tp_degree
        self.global_key_dim = self.head_k_dim * self.global_num_k_heads  # 2048
        self.global_value_dim = self.head_v_dim * self.global_num_v_heads
        self.key_dim = self.head_k_dim * self.num_k_heads  # 512 at TP=4
        self.value_dim = self.head_v_dim * self.num_v_heads  # 1536 at TP=4
        self.conv_kernel_size = tc.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.rms_norm_eps = tc.rms_norm_eps
        self.use_hybrid_cache_manager = getattr(tc, "use_hybrid_cache_manager", False)
        self.use_qwen_hybrid_chunked_prefill = getattr(
            tc, "use_qwen_hybrid_chunked_prefill", False
        )
        self.use_qwen_hybrid_chunked_prefill_nki = getattr(
            tc, "use_qwen_hybrid_chunked_prefill_nki", False
        )

        # KV cache dummy shape info
        self.head_dim = tc.head_dim  # 256
        tp_degree = tc.neuron_config.tp_degree
        raw_kv_heads = tc.num_key_value_heads
        if raw_kv_heads < tp_degree:
            replicated_kv_heads = tp_degree
        else:
            replicated_kv_heads = raw_kv_heads
        self.kv_heads_per_rank = replicated_kv_heads // tp_degree

        # Conv1d on concatenated QKV (NOT Z).  Store the depthwise kernel in a
        # ColumnParallelLinear parameter container so NxD's checkpoint sharder
        # can split it by output channel.  Forward still uses it as Conv1d
        # weight after unsqueezing the singleton input-channel dimension.
        self.global_conv_dim = self.global_key_dim * 2 + self.global_value_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim  # 2560 at TP=4
        self.conv1d_weight = ColumnParallelLinear(
            self.conv_kernel_size,
            self.global_conv_dim,
            bias=False,
            gather_output=False,
        )

        # Input/output projections are the large DeltaNet tensors.  Shard them
        # with tensor parallelism; convert_qwen36_a3b_hf_to_neuron_state_dict()
        # reorders in_proj_qkv into per-rank [Q_local | K_local | V_local]
        # blocks before NxD slices the output dimension.
        self.in_proj_qkv = ColumnParallelLinear(
            self.hidden_size,
            self.global_key_dim * 2 + self.global_value_dim,
            bias=False,
            gather_output=False,
        )
        self.in_proj_z = ColumnParallelLinear(
            self.hidden_size,
            self.global_value_dim,
            bias=False,
            gather_output=False,
        )
        self.in_proj_b = ColumnParallelLinear(
            self.hidden_size,
            self.global_num_v_heads,
            bias=False,
            gather_output=False,
        )
        self.in_proj_a = ColumnParallelLinear(
            self.hidden_size,
            self.global_num_v_heads,
            bias=False,
            gather_output=False,
        )

        # Same parameter-container pattern for per-value-head decay vectors.
        # These are used as vectors in forward but sharded by output dim during
        # checkpoint conversion/loading.
        self.dt_bias_weight = ColumnParallelLinear(
            1,
            self.global_num_v_heads,
            bias=False,
            gather_output=False,
        )
        self.A_log_weight = ColumnParallelLinear(
            1,
            self.global_num_v_heads,
            bias=False,
            gather_output=False,
        )

        # Output norm and projection
        self.norm = Qwen3MoeRMSNorm(self.head_v_dim, eps=self.rms_norm_eps)
        self.out_proj = RowParallelLinear(
            self.global_value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
        )

        # State buffers for CTE -> TKG carry-over
        alloc_batch_size = getattr(config.neuron_config, "max_batch_size", 1)
        self._phase_batch_size = getattr(config.neuron_config, "batch_size", 1)
        self.recurrent_state_buffer = nn.Parameter(
            torch.zeros(
                alloc_batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                dtype=config.neuron_config.torch_dtype,
            ),
            requires_grad=False,
        )
        self.conv_state_buffer = nn.Parameter(
            torch.zeros(
                alloc_batch_size,
                self.conv_dim,
                self.conv_kernel_size - 1,
                dtype=config.neuron_config.torch_dtype,
            ),
            requires_grad=False,
        )

    def _conv1d_weight(self):
        return self.conv1d_weight.weight.unsqueeze(1)

    def _dt_bias(self):
        return self.dt_bias_weight.weight.squeeze(1)

    def _A_log(self):
        return self.A_log_weight.weight.squeeze(1)

    def _recurrent_step(self, query, key, value, g, beta, recurrent_state):
        """Single-step recurrent update for token generation."""
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        scale = 1.0 / (query.shape[-1] ** 0.5)
        query = query * scale

        q_t = query[:, :, 0]
        k_t = key[:, :, 0]
        v_t = value[:, :, 0]
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, 0].unsqueeze(-1)

        new_state = recurrent_state * g_t
        kv_mem = (new_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        new_state = new_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)

        return output.unsqueeze(2), new_state

    def verify_block_candidates(self, query, key, value, g, beta, initial_state):
        """Run the k+1 verify block as exact single-step recurrences seeded from
        ``initial_state``, emitting a per-position state checkpoint after each
        token so the host can select the carried state by its accept count.

        For the k=1 (2-token) block the steps are:
            step 0 from ``initial_state`` over position 0 (the real token),
            step 1 from step-0's state over position 1 (the draft token).

        Reuses :meth:`_recurrent_step` (the exact single-step recurrence, no
        chunked-decay split) so there is no decay-window overflow. Inputs are the
        already projected / conv'd / headed tensors for the block in the same
        ``[B, H, S, dim]`` layout ``_recurrent_step`` expects (g/beta ``[B, H, S]``);
        they are sliced one position at a time.

        Returns ``(out_stack, S_stack)`` where

            out_stack : [B, H, S, Vd]  -- per-position recurrence outputs
            S_stack   : [B, S, H, Kd, Vd]  -- candidate states [S_0, S_1, ...]
                        stacked on a new axis-1; ``S_i`` is the recurrent state
                        after consuming token ``i``.

        Generic over block length: with k tokens it stacks k checkpoints; k=1
        (two candidates) is the current case.
        """
        num_positions = query.shape[2]
        recurrent_state = initial_state
        outputs = []
        states = []
        for i in range(num_positions):
            out_i, recurrent_state = self._recurrent_step(
                query[:, :, i : i + 1],
                key[:, :, i : i + 1],
                value[:, :, i : i + 1],
                g[:, :, i : i + 1],
                beta[:, :, i : i + 1],
                recurrent_state,
            )
            outputs.append(out_i)
            states.append(recurrent_state)

        # out_i is [B, H, 1, Vd]; concat along the position axis -> [B, H, S, Vd].
        out_stack = torch.cat(outputs, dim=2)
        # Each state is [B, H, Kd, Vd]; stack on a new candidate axis-1.
        S_stack = torch.stack(states, dim=1)
        return out_stack, S_stack

    def conv_window_candidates(self, conv_seed, mixed):
        """Per-position causal conv windows for the verify block.

        ``conv_seed`` is the carried conv state ``[B, conv_dim, K-1]`` (the window
        feeding the first block token) and ``mixed`` is the block's conv input
        ``[B, conv_dim, S]`` (channels-first, pre-activation, same tensor whose
        tail is gathered into ``new_conv_state`` in forward()). The conv window
        *after* token ``i`` is the ``K-1`` columns of ``cat([conv_seed, mixed])``
        ending at token ``i``'s column, i.e. ``conv_input[:, :, i+1 : i+K]``.

        Returns ``[B, S, conv_dim, K-1]`` -- one window per block position, so the
        host can carry the conv state matching its chosen accept count.
        """
        state_len = self.conv_kernel_size - 1
        conv_input = torch.cat([conv_seed, mixed], dim=-1)
        num_positions = mixed.shape[-1]
        windows = [
            conv_input[:, :, i + 1 : i + 1 + state_len] for i in range(num_positions)
        ]
        return torch.stack(windows, dim=1)

    def verify_block(self, hidden_states, seq_ids):
        """Verify-mode forward over the k+1 (=2 at k=1) token block.

        Seeded from the COMMITTED state in ``recurrent_state_buffer`` /
        ``conv_state_buffer`` (read-only — this method never writes them), it
        reproduces the projection / causal-conv / gating of :meth:`forward` over
        the short block, then emits a per-position state checkpoint after each
        token instead of committing a single final state. The exact single-step
        recurrence (:meth:`verify_block_candidates` -> :meth:`_recurrent_step`) is
        used rather than the chunked NKI kernels: at 2 tokens it is exact and
        cheap, and a host loop selects ``S_{accept_count}`` after verify.

        Returns
        -------
        attn_out : [B, S, H]
            Output projection per block position (same as ``forward``'s output).
        S_stack : [B, S, num_v_heads, head_k_dim, head_v_dim]
            Candidate recurrent states ``[S_1 .. S_S]`` (state after each token).
        conv_cand : [B, S, conv_dim, K-1]
            Candidate causal-conv windows after each token.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Projections (mirrors forward()'s standard path).
        qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        query = qkv[..., : self.key_dim]
        key = qkv[..., self.key_dim : self.key_dim * 2]
        value = qkv[..., self.key_dim * 2 :]

        # Causal conv over the block, seeded from the committed conv window.
        mixed = torch.cat([query, key, value], dim=-1).transpose(1, 2)  # [B, conv_dim, S]
        conv_seed = torch.index_select(self.conv_state_buffer, 0, seq_ids)
        conv_input = torch.cat([conv_seed, mixed], dim=-1)
        w = self._conv1d_weight().squeeze(1)  # [conv_dim, K]
        conv_out = torch.zeros_like(mixed)
        for k in range(self.conv_kernel_size):
            conv_out = conv_out + w[:, k].unsqueeze(0).unsqueeze(-1) * conv_input[
                :, :, k : k + seq_len
            ]
        mixed_post_conv = F.silu(conv_out)

        # Per-position conv windows (candidate conv states) -- needs the
        # channels-first `mixed` and the seed, before transposing back.
        conv_cand = self.conv_window_candidates(conv_seed, mixed)

        mixed_post_conv = mixed_post_conv.transpose(1, 2)  # [B, S, conv_dim]
        query = mixed_post_conv[..., : self.key_dim]
        key = mixed_post_conv[..., self.key_dim : self.key_dim * 2]
        value = mixed_post_conv[..., self.key_dim * 2 :]

        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self._A_log().float().exp() * F.softplus(a.float() + self._dt_bias())

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = (
                query.unsqueeze(3)
                .expand(-1, -1, -1, rep, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )
            key = (
                key.unsqueeze(3)
                .expand(-1, -1, -1, rep, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )

        query = query.transpose(1, 2).contiguous().float()
        key = key.transpose(1, 2).contiguous().float()
        value = value.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()

        # Seed the recurrence from the committed recurrent state (read-only).
        initial_state = torch.index_select(
            self.recurrent_state_buffer, 0, seq_ids
        ).float()

        out_stack, S_stack = self.verify_block_candidates(
            query, key, value, g, beta, initial_state
        )

        # Output: norm, z-gate, project (mirrors forward()).
        output = out_stack.to(hidden_states.dtype)
        output = output.transpose(1, 2).contiguous()  # [B, S, H, Vd]
        output = output.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = self.norm(output)
        z_gate = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = output * F.silu(z_gate)
        output = output.reshape(batch_size, seq_len, self.value_dim)
        output = self.out_proj(output)

        return output, S_stack, conv_cand

    def _nki_recurrent_forward(self, query, key, value, g, beta):
        """Full-sequence recurrent forward using NKI kernel for context encoding."""
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        BH = B * H
        query_flat = query.reshape(BH, S, k_dim).contiguous()
        key_flat = key.reshape(BH, S, k_dim).contiguous()
        value_flat = value.reshape(BH, S, v_dim).contiguous()

        g_flat = g.reshape(BH, S).unsqueeze(-1).expand(-1, -1, v_dim).contiguous()
        beta_flat = beta.reshape(BH, S).unsqueeze(-1).expand(-1, -1, v_dim).contiguous()

        outputs = []
        states = []
        for bh in range(BH):
            out_bh, state_bh = _deltanet_nki_kernel_state(
                query_flat[bh],
                key_flat[bh],
                value_flat[bh],
                g_flat[bh],
                beta_flat[bh],
            )
            outputs.append(out_bh)
            states.append(state_bh)

        output = torch.stack(outputs, dim=0)
        output = output.reshape(B, H, S, v_dim)

        final_state = torch.stack(states, dim=0)
        final_state = final_state.reshape(B, H, k_dim, v_dim)

        return output, final_state

    def _nki_chunked_forward(
        self, query, key, value, g, beta, output_final_state=False, initial_state=None
    ):
        """Chunked NKI kernel forward for context encoding (prefill)."""
        chunk_size = 128

        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        pad_size = (chunk_size - S % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))
        total_seq_len = S + pad_size

        num_chunks = total_seq_len // chunk_size
        g_reshaped = g.reshape(B, H, num_chunks, chunk_size)
        g_cs = g_reshaped.cumsum(dim=-1)
        g_last_per_chunk = g_cs[:, :, :, -1:]
        g_last_expanded = g_last_per_chunk.expand(-1, -1, -1, chunk_size)

        query_chunks = query.reshape(B, H, num_chunks, chunk_size, k_dim)
        key_chunks = key.reshape(B, H, num_chunks, chunk_size, k_dim)
        value_chunks = value.reshape(B, H, num_chunks, chunk_size, v_dim)

        beta_chunks = (
            beta.reshape(B, H, num_chunks, chunk_size)
            .unsqueeze(-1)
            .expand(-1, -1, -1, -1, v_dim)
        )
        gc_chunks = g_cs.unsqueeze(-1).expand(-1, -1, -1, -1, v_dim)
        gl_chunks = g_last_expanded.unsqueeze(-1).expand(-1, -1, -1, -1, v_dim)

        BH = B * H
        query_chunks = query_chunks.reshape(
            BH, num_chunks, chunk_size, k_dim
        ).contiguous()
        key_chunks = key_chunks.reshape(BH, num_chunks, chunk_size, k_dim).contiguous()
        value_chunks = value_chunks.reshape(
            BH, num_chunks, chunk_size, v_dim
        ).contiguous()
        beta_chunks = beta_chunks.reshape(
            BH, num_chunks, chunk_size, v_dim
        ).contiguous()
        gc_chunks = gc_chunks.reshape(BH, num_chunks, chunk_size, v_dim).contiguous()
        gl_chunks = gl_chunks.reshape(BH, num_chunks, chunk_size, v_dim).contiguous()

        device = query.device
        lower_mask = torch.tril(
            torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=device),
            diagonal=-1,
        )
        identity_mat = torch.eye(chunk_size, dtype=torch.float32, device=device)
        lower_mask_diag = torch.tril(
            torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=device),
            diagonal=0,
        )

        initial_state_flat = None
        if initial_state is not None:
            initial_state_flat = initial_state.reshape(BH, k_dim, v_dim).float().contiguous()

        all_outputs = []
        all_states = []
        for bh in range(BH):
            if initial_state_flat is None:
                state = torch.zeros(k_dim, v_dim, dtype=torch.float32, device=device)
            else:
                state = initial_state_flat[bh]

            head_chunks = []
            for c_idx in range(num_chunks):
                q_chunk = query_chunks[bh, c_idx].contiguous()
                k_chunk = key_chunks[bh, c_idx].contiguous()
                v_chunk = value_chunks[bh, c_idx].contiguous()
                beta_chunk = beta_chunks[bh, c_idx].contiguous()
                gc_chunk = gc_chunks[bh, c_idx].contiguous()
                gl_chunk = gl_chunks[bh, c_idx].contiguous()

                out_chunk, state = _deltanet_nki_chunk_step(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    beta_chunk,
                    gc_chunk,
                    gl_chunk,
                    state,
                    lower_mask,
                    identity_mat,
                    lower_mask_diag,
                )
                head_chunks.append(out_chunk)

            head_output = torch.cat(head_chunks, dim=0)
            all_outputs.append(head_output)
            all_states.append(state)

        output = torch.stack(all_outputs, dim=0)
        output = output.reshape(B, H, total_seq_len, v_dim)
        output = output[:, :, :S]

        if output_final_state:
            final_state = torch.stack(all_states, dim=0)
            last_recurrent_state = final_state.reshape(B, H, k_dim, v_dim)
        else:
            last_recurrent_state = None

        return output, last_recurrent_state

    def _fused_chunked_forward(
        self, query, key, value, g, beta, output_final_state=False
    ):
        """Fused single-kernel chunked forward for CTE — SSD-style.

        Processes all chunks in a single NKI kernel call per (B,H) pair.
        State persists in SBUF across chunks (no HBM round-trips).
        Cumsum of g computed in-kernel via tensor_tensor_scan.

        This is the optimized version of _nki_chunked_forward with:
          1. Single kernel call per (B,H) instead of B*H*num_chunks
          2. State in SBUF across all chunks (biggest perf win)
          3. In-kernel cumsum (avoids PyTorch cumsum overhead)
          4. tensor_scalar for broadcasts (no explicit loops)
        """
        chunk_size = 128

        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        # Pad sequence to multiple of chunk_size
        pad_size = (chunk_size - S % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))
        total_seq_len = S + pad_size
        # Pass raw per-token log-decay. The fused NKI kernel forms decay as
        # exp(cumsum(g)_i - cumsum(g)_j), so no pre-kernel clamp is needed.

        BH = B * H
        # Flatten to (BH, S, dim) for per-(b,h) kernel calls
        query_flat = query.reshape(BH, total_seq_len, k_dim).contiguous()
        key_flat = key.reshape(BH, total_seq_len, k_dim).contiguous()
        value_flat = value.reshape(BH, total_seq_len, v_dim).contiguous()

        # g and beta: (BH, S) -> (BH, S, 1) for the kernel's (S, 1) input layout
        g_flat = g.reshape(BH, total_seq_len).unsqueeze(-1).contiguous()
        beta_flat = beta.reshape(BH, total_seq_len).unsqueeze(-1).contiguous()

        # Create constant mask tensors (shared across all B*H calls)
        device = query.device
        lower_mask = torch.tensor(
            _make_lower_mask(), dtype=torch.float32, device=device
        )
        identity_mat = torch.tensor(
            _make_identity(), dtype=torch.float32, device=device
        )
        lower_mask_diag = torch.tensor(
            _make_lower_mask_diag(), dtype=torch.float32, device=device
        )

        all_outputs = []
        all_states = []
        for bh in range(BH):
            out_bh, state_bh = _deltanet_fused_kernel(
                query_flat[bh],  # (S, 128)
                key_flat[bh],  # (S, 128)
                value_flat[bh],  # (S, 128)
                g_flat[bh],  # (S, 1) — RAW g, not cumsum
                beta_flat[bh],  # (S, 1) — sigmoid(b)
                lower_mask,  # (128, 128)
                identity_mat,  # (128, 128)
                lower_mask_diag,  # (128, 128)
            )
            all_outputs.append(out_bh)
            all_states.append(state_bh)

        output = torch.stack(all_outputs, dim=0)
        output = output.reshape(B, H, total_seq_len, v_dim)
        output = output[:, :, :S]

        if output_final_state:
            final_state = torch.stack(all_states, dim=0)
            last_recurrent_state = final_state.reshape(B, H, k_dim, v_dim)
        else:
            last_recurrent_state = None

        return output, last_recurrent_state

    def _sequential_forward(self, query, key, value, g, beta, output_final_state=False):
        """Sequential full-sequence gated delta rule for CTE.

        Uses the same per-step recurrence as _recurrent_step but loops over the
        full sequence.  Avoids the slice-assignment loop in _chunk_forward that
        may compile incorrectly on Neuron/XLA.
        """
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        state = query.new_zeros(B, H, k_dim, v_dim)
        all_outputs = []
        for t in range(S):
            q_t = query[:, :, t]  # (B, H, K)
            k_t = key[:, :, t]  # (B, H, K)
            v_t = value[:, :, t]  # (B, H, V)
            beta_t = beta[:, :, t].unsqueeze(-1)  # (B, H, 1)
            g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

            # Gated delta rule
            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # (B, H, V)
            delta = (v_t - kv_mem) * beta_t  # (B, H, V)
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # (B, H, K, V)

            o_t = (state * q_t.unsqueeze(-1)).sum(dim=-2)  # (B, H, V)
            all_outputs.append(o_t.unsqueeze(2))

        output = torch.cat(all_outputs, dim=2)  # (B, H, S, V)
        final_state = state if output_final_state else None
        return output, final_state

    def _chunk_forward(
        self, query, key, value, g, beta, output_final_state=False, initial_state=None
    ):
        """Chunk-based forward for context encoding (prefill)."""
        chunk_size = 64

        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        pad_size = (chunk_size - S % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))
        total_seq_len = S + pad_size

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        num_chunks = total_seq_len // chunk_size
        query = query.reshape(B, H, num_chunks, chunk_size, k_dim)
        key = key.reshape(B, H, num_chunks, chunk_size, k_dim)
        value = value.reshape(B, H, num_chunks, chunk_size, v_dim)
        k_beta = k_beta.reshape(B, H, num_chunks, chunk_size, k_dim)
        v_beta = v_beta.reshape(B, H, num_chunks, chunk_size, v_dim)
        g = g.reshape(B, H, num_chunks, chunk_size)

        mask = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=0,
        )

        g = g.cumsum(dim=-1)
        decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()

        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        if initial_state is None:
            last_recurrent_state = torch.zeros(
                B, H, k_dim, v_dim, dtype=query.dtype, device=query.device
            )
        else:
            last_recurrent_state = initial_state.to(dtype=query.dtype)
        core_attn_out = torch.zeros_like(value)
        mask2 = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=1,
        )

        for i in range(num_chunks):
            q_i = query[:, :, i]
            k_i = key[:, :, i]
            v_i = value[:, :, i]

            attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(
                mask2, 0
            )

            v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
            v_new = v_i - v_prime

            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn_i @ v_new

            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (
                    k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
                ).transpose(-1, -2)
                @ v_new
            )

        core_attn_out = core_attn_out.reshape(B, H, -1, v_dim)
        core_attn_out = core_attn_out[:, :, :S]

        if not output_final_state:
            last_recurrent_state = None

        return core_attn_out, last_recurrent_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        **kwargs,
    ):
        """Forward pass compatible with NxDI decoder layer interface."""
        batch_size, seq_len, _ = hidden_states.shape

        seq_ids = kwargs.get("seq_ids", None)
        qwen_chunked_prefill_active = (
            self.use_qwen_hybrid_chunked_prefill
            and past_key_value is not None
            and seq_len > 1
        )
        is_decode = past_key_value is not None and not qwen_chunked_prefill_active

        # Padding mask for DeltaNet: [B, S, 1] with 1.0 for real tokens, 0.0 for padding.
        # Passed from get_model_output where it's computed from input_ids != pad_token_id.
        # Embeddings are already zeroed for padding tokens; this mask additionally
        # zeros the decay gate so the recurrent state is preserved unchanged
        # through padding positions (no spurious decay).
        valid_mask_1d = kwargs.get("deltanet_padding_mask", None)  # [B, S, 1] or None
        hybrid_cache_active = self.use_hybrid_cache_manager
        recurrent_state_cache = None
        conv_state_cache = None
        if hybrid_cache_active and past_key_value is not None:
            recurrent_state_cache, conv_state_cache = past_key_value

        # Project inputs
        deltanet_fp32 = os.environ.get("DELTANET_FP32") == "1"
        if deltanet_fp32 and isinstance(self.in_proj_qkv, nn.Linear):
            hs_f32 = hidden_states.float()
            qkv = F.linear(hs_f32, self.in_proj_qkv.weight.float()).to(
                hidden_states.dtype
            )
            z = F.linear(hs_f32, self.in_proj_z.weight.float()).to(hidden_states.dtype)
            b = F.linear(hs_f32, self.in_proj_b.weight.float()).to(hidden_states.dtype)
            a = F.linear(hs_f32, self.in_proj_a.weight.float()).to(hidden_states.dtype)
        else:
            qkv = self.in_proj_qkv(hidden_states)
            z = self.in_proj_z(hidden_states)
            b = self.in_proj_b(hidden_states)
            a = self.in_proj_a(hidden_states)

        # Split QKV
        query = qkv[..., : self.key_dim]
        key = qkv[..., self.key_dim : self.key_dim * 2]
        value = qkv[..., self.key_dim * 2 :]

        # Causal Conv1d on QKV
        mixed = torch.cat([query, key, value], dim=-1)
        mixed = mixed.transpose(1, 2)

        if is_decode:
            if conv_state_cache is not None:
                conv_state = conv_state_cache[:batch_size]
            elif seq_ids is not None:
                conv_state = torch.index_select(self.conv_state_buffer, 0, seq_ids)
            else:
                conv_state = self.conv_state_buffer[:batch_size]
            conv_input = torch.cat([conv_state, mixed], dim=-1)

            w = self._conv1d_weight().squeeze(1)
            conv_out = torch.zeros_like(mixed)
            for k in range(4):
                conv_out = (
                    conv_out
                    + w[:, k].unsqueeze(0).unsqueeze(-1) * conv_input[:, :, k : k + 1]
                )
            mixed_post_conv = F.silu(conv_out)

            new_conv_state = torch.cat([conv_state[:, :, 1:], mixed], dim=-1)
            alloc_bs = self.conv_state_buffer.shape[0]
            if hybrid_cache_active:
                new_conv_state = new_conv_state.to(self.conv_state_buffer.dtype)
            elif seq_ids is not None:
                # BS=1 optimization: scatter to index 0 of size-1 buffer = direct replacement
                # Add buffer dependency for input_output_alias
                new_conv_state = (
                    new_conv_state.to(self.conv_state_buffer.dtype)
                    + self.conv_state_buffer * 0
                )
            elif batch_size < alloc_bs:
                pad_size = alloc_bs - batch_size
                new_conv_state = torch.cat(
                    [
                        new_conv_state,
                        self.conv_state_buffer[batch_size:] * 0,
                    ],
                    dim=0,
                )
            else:
                new_conv_state = new_conv_state + self.conv_state_buffer * 0
        else:
            if qwen_chunked_prefill_active and conv_state_cache is not None:
                conv_state = conv_state_cache[:batch_size]
                if position_ids is not None:
                    reset_mask = (position_ids[:, :1].long() == 0).to(
                        dtype=conv_state.dtype, device=conv_state.device
                    )
                    conv_state = conv_state * (1.0 - reset_mask[:, None, :])
                conv_input = torch.cat([conv_state, mixed], dim=-1)
                w = self._conv1d_weight().squeeze(1)
                conv_out = torch.zeros_like(mixed)
                for k in range(self.conv_kernel_size):
                    conv_out = conv_out + w[:, k].unsqueeze(0).unsqueeze(-1) * conv_input[
                        :, :, k : k + seq_len
                    ]
                mixed_post_conv = F.silu(conv_out)
                if valid_mask_1d is not None:
                    state_len = self.conv_kernel_size - 1
                    num_valid = valid_mask_1d.squeeze(-1).sum(dim=-1, keepdim=True).long()
                    idx_base = (state_len + num_valid - state_len).clamp(min=0)
                    offsets = torch.arange(state_len, device=mixed.device).unsqueeze(0)
                    gather_idx = idx_base + offsets
                    gather_idx = gather_idx.unsqueeze(1).expand(-1, self.conv_dim, -1)
                    new_conv_state = torch.gather(conv_input, 2, gather_idx)
                else:
                    new_conv_state = conv_input[:, :, -self.conv_kernel_size + 1 :].contiguous()
            else:
                mixed_post_conv = F.silu(
                    F.conv1d(
                        mixed,
                        self._conv1d_weight(),
                        bias=None,
                        padding=self.conv_kernel_size - 1,
                        groups=self.conv_dim,
                    )[:, :, :seq_len]
                )

                if valid_mask_1d is not None:
                    # valid_mask_1d is [B, S, 1]; count valid tokens per batch
                    num_valid = (
                        valid_mask_1d.squeeze(-1).sum(dim=-1, keepdim=True).long()
                    )  # [B, 1]
                    idx_base = num_valid - 3
                    idx_base = idx_base.clamp(min=0)
                    offsets = torch.arange(3, device=mixed.device).unsqueeze(0)
                    gather_idx = idx_base + offsets  # [B, 3]
                    gather_idx = gather_idx.unsqueeze(1).expand(-1, self.conv_dim, -1)
                    new_conv_state = torch.gather(mixed, 2, gather_idx)
                else:
                    new_conv_state = mixed[:, :, -3:].contiguous()

            alloc_bs = self.conv_state_buffer.shape[0]
            if hybrid_cache_active:
                new_conv_state = new_conv_state.to(self.conv_state_buffer.dtype)
            elif seq_ids is not None:
                # BS=1 optimization: scatter to index 0 = direct replacement
                new_conv_state = (
                    new_conv_state.to(self.conv_state_buffer.dtype)
                    + self.conv_state_buffer * 0
                )
            elif batch_size < alloc_bs:
                pad_size = alloc_bs - batch_size
                new_conv_state = torch.cat(
                    [
                        new_conv_state,
                        torch.zeros(
                            pad_size,
                            self.conv_dim,
                            self.conv_kernel_size - 1,
                            dtype=new_conv_state.dtype,
                            device=new_conv_state.device,
                        ),
                    ],
                    dim=0,
                )
                new_conv_state = new_conv_state + self.conv_state_buffer * 0
            else:
                new_conv_state = new_conv_state + self.conv_state_buffer * 0

        mixed_post_conv = mixed_post_conv.transpose(1, 2)

        # Zero out conv1d output for padding positions.
        # Conv1d with kernel_size=4 leaks real token info into the first
        # few padding positions.  Zeroing here ensures Q, K, V are exactly
        # zero for all padding positions so the recurrence is unaffected.
        if valid_mask_1d is not None:
            mixed_post_conv = (
                mixed_post_conv * valid_mask_1d
            )  # [B, S, conv_dim] * [B, S, 1]

        query = mixed_post_conv[..., : self.key_dim]
        key = mixed_post_conv[..., self.key_dim : self.key_dim * 2]
        value = mixed_post_conv[..., self.key_dim * 2 :]

        # Reshape to heads
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # Compute gating
        beta = b.sigmoid()
        g = -self._A_log().float().exp() * F.softplus(a.float() + self._dt_bias())

        if valid_mask_1d is not None:
            # Zero g for padding → alpha=exp(0)=1 → state preserved through padding
            # Zero beta for padding → no state update from padding tokens
            mask_2d = valid_mask_1d.squeeze(-1).float()  # [B, S]
            g = g * mask_2d.unsqueeze(-1)
            beta = beta * mask_2d.unsqueeze(-1)

        # Expand K heads to match V heads (16 -> 48) using expand+reshape
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads  # 3
            query = (
                query.unsqueeze(3)
                .expand(-1, -1, -1, rep, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )
            key = (
                key.unsqueeze(3)
                .expand(-1, -1, -1, rep, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )

        # Transpose to (B, H, S, dim)
        query = query.transpose(1, 2).contiguous().float()
        key = key.transpose(1, 2).contiguous().float()
        value = value.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()

        if is_decode:
            # TKG: single-step recurrent update
            if recurrent_state_cache is not None:
                recurrent_state = recurrent_state_cache[:batch_size].float()
            elif seq_ids is not None:
                recurrent_state = torch.index_select(
                    self.recurrent_state_buffer, 0, seq_ids
                ).float()
            else:
                recurrent_state = self.recurrent_state_buffer[:batch_size].float()

            output, new_state = self._recurrent_step(
                query, key, value, g, beta, recurrent_state
            )
            new_state_bf16 = new_state.to(self.recurrent_state_buffer.dtype)
            alloc_bs = self.recurrent_state_buffer.shape[0]
            if hybrid_cache_active:
                new_rec_state = new_state_bf16
            elif seq_ids is not None:
                # BS=1 optimization: scatter to index 0 of size-1 buffer = direct replacement
                # Add buffer dependency for input_output_alias
                new_rec_state = new_state_bf16 + self.recurrent_state_buffer * 0
            elif batch_size < alloc_bs:
                new_rec_state = torch.cat(
                    [
                        new_state_bf16,
                        self.recurrent_state_buffer[batch_size:] * 0,
                    ],
                    dim=0,
                )
            else:
                new_rec_state = new_state_bf16 + self.recurrent_state_buffer * 0
        else:
            # CTE: chunked NKI kernel by default. It forms the within-chunk
            # decay as exp(gc[i] - gc[j]) (always <= 1), which is numerically
            # stable. The fused kernel uses the split form exp(gc[i])*exp(-gc[j]);
            # with this checkpoint's gating (exp(A_log) up to ~72), a single
            # token's cumulative log-decay reaches < -88, so exp(-gc) overflows
            # float32 -> inf -> NaN logits -> degenerate (all-EOS) generation.
            # Override with env vars for debugging/benchmarking.
            use_nki_fused = os.environ.get("USE_NKI_FUSED") == "1"
            use_nki = os.environ.get("USE_NKI") == "1"
            use_sequential = os.environ.get("DELTANET_SEQUENTIAL") == "1"
            use_pytorch_chunk = os.environ.get("USE_PYTORCH_CHUNK") == "1"
            # Default-on unless another path is explicitly requested.
            use_nki_chunked = os.environ.get("USE_NKI_CHUNKED", "1") != "0" and not (
                use_nki_fused or use_nki or use_sequential or use_pytorch_chunk
            )

            if qwen_chunked_prefill_active and recurrent_state_cache is not None:
                initial_state = recurrent_state_cache[:batch_size].float()
                if position_ids is not None:
                    reset_mask = (position_ids[:, :1].long() == 0).to(
                        dtype=initial_state.dtype, device=initial_state.device
                    )
                    initial_state = initial_state * (1.0 - reset_mask[:, :, None, None])
                if self.use_qwen_hybrid_chunked_prefill_nki:
                    output, final_state = self._nki_chunked_forward(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        output_final_state=True,
                        initial_state=initial_state,
                    )
                else:
                    output, final_state = self._chunk_forward(
                        query,
                        key,
                        value,
                        g,
                        beta,
                        output_final_state=True,
                        initial_state=initial_state,
                    )
            elif use_pytorch_chunk:
                output, final_state = self._chunk_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            elif use_nki_chunked:
                output, final_state = self._nki_chunked_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            elif use_nki:
                output, final_state = self._nki_recurrent_forward(
                    query, key, value, g, beta
                )
            elif use_sequential:
                output, final_state = self._sequential_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            elif use_nki_fused:
                output, final_state = self._fused_chunked_forward(
                    query, key, value, g, beta, output_final_state=True
                )
            else:
                output, final_state = self._fused_chunked_forward(
                    query, key, value, g, beta, output_final_state=True
                )

            if final_state is not None:
                final_state_bf16 = final_state.to(self.recurrent_state_buffer.dtype)
                alloc_bs = self.recurrent_state_buffer.shape[0]
                if hybrid_cache_active:
                    new_rec_state = final_state_bf16
                elif seq_ids is not None:
                    # BS=1 optimization: scatter to index 0 of size-1 buffer = direct replacement
                    # Add buffer dependency for input_output_alias
                    new_rec_state = final_state_bf16 + self.recurrent_state_buffer * 0
                elif batch_size < alloc_bs:
                    new_rec_state = torch.cat(
                        [
                            final_state_bf16,
                            torch.zeros(
                                alloc_bs - batch_size,
                                self.num_v_heads,
                                self.head_k_dim,
                                self.head_v_dim,
                                dtype=final_state_bf16.dtype,
                                device=final_state_bf16.device,
                            ),
                        ],
                        dim=0,
                    )
                    new_rec_state = new_rec_state + self.recurrent_state_buffer * 0
                else:
                    new_rec_state = final_state_bf16 + self.recurrent_state_buffer * 0
            else:
                new_rec_state = self.recurrent_state_buffer * 1

        # Output: norm, gate, project
        output = output.to(hidden_states.dtype)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = self.norm(output)
        z_gate = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        output = output * F.silu(z_gate)
        output = output.reshape(batch_size, seq_len, self.value_dim)
        output = self.out_proj(output)

        if hybrid_cache_active:
            return output, (new_rec_state, new_conv_state), new_rec_state, new_conv_state

        # Return dummy KV for KVCacheManager
        dummy_k = torch.zeros(
            batch_size,
            self.kv_heads_per_rank,
            seq_len,
            self.head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        dummy_v = torch.zeros_like(dummy_k)

        return output, (dummy_k, dummy_v), new_rec_state, new_conv_state


# ============================================================
# InferenceConfig (Dense -- no MoE)
# ============================================================


class Qwen36A3BInferenceConfig(InferenceConfig):
    """Hybrid DeltaNet + GQA decoder with 256-expert MoE and an MTP head."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("num_hidden_layers", 40)

        # Hybrid layer dispatch: [3 DeltaNet + 1 GQA] repeated.
        if "layer_types" not in kwargs and not any(
            hasattr(a, "layer_types") for a in args if hasattr(a, "__dict__")
        ):
            num_layers = kwargs["num_hidden_layers"]
            if num_layers % 4 != 0:
                raise ValueError(
                    f"hybrid layer count must be divisible by 4, got {num_layers}"
                )
            layer_types: list = []
            for _ in range(num_layers // 4):
                layer_types.extend(
                    ["linear_attention"] * 3 + ["full_attention"]
                )
            kwargs.setdefault("layer_types", layer_types)

        # DeltaNet
        kwargs.setdefault("linear_num_value_heads", 32)
        kwargs.setdefault("linear_num_key_heads", 16)
        kwargs.setdefault("linear_key_head_dim", 128)
        kwargs.setdefault("linear_value_head_dim", 128)
        kwargs.setdefault("linear_conv_kernel_dim", 4)
        kwargs.setdefault("use_hybrid_cache_manager", False)
        kwargs.setdefault("use_qwen_hybrid_chunked_prefill", False)
        kwargs.setdefault("use_qwen_hybrid_chunked_prefill_nki", False)

        # Surface the pre-final-norm trunk hidden state as a trailing forward
        # output (last position only). Inert unless explicitly enabled; consumed
        # by the draft/MTP head wiring.
        kwargs.setdefault("output_trunk_hidden", False)

        # Register the verify backbone (n_active=2) as an additional compiled
        # graph. Inert unless explicitly enabled (A3B_ENABLE_VERIFY=1).
        kwargs.setdefault("enable_verify_backbone", False)

        # MoE
        kwargs.setdefault("num_experts", 256)
        kwargs.setdefault("num_experts_per_tok", 8)
        kwargs.setdefault("moe_intermediate_size", 512)
        kwargs.setdefault("shared_expert_intermediate_size", 512)
        kwargs.setdefault("norm_topk_prob", True)
        kwargs.setdefault("router_aux_loss_coef", 0.001)
        # ExpertMLPsV2 reads config.intermediate_size for routed + shared experts.
        # Seed it before super().__init__ so the validator passes.
        kwargs.setdefault("intermediate_size", kwargs["moe_intermediate_size"])

        # MTP
        kwargs.setdefault("mtp_num_hidden_layers", 1)
        kwargs.setdefault("mtp_use_dedicated_embeddings", False)

        super().__init__(*args, **kwargs)

        # Attention output gate
        self.attn_output_gate = getattr(self, "attn_output_gate", True)

        # Trailing trunk-hidden output flag
        self.output_trunk_hidden = getattr(self, "output_trunk_hidden", False)

        # Verify backbone graph flag
        self.enable_verify_backbone = getattr(self, "enable_verify_backbone", False)

        # Partial RoPE
        self.partial_rotary_factor = getattr(self, "partial_rotary_factor", 0.25)
        self.rope_dim = int(self.head_dim * self.partial_rotary_factor)

        # mRoPE (multimodal RoPE) for any VL extension
        rope_params = getattr(self, "rope_parameters", {}) or {}
        self.mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        self.mrope_interleaved = rope_params.get("mrope_interleaved", True)

        # Both routed and shared expert dims must match (single intermediate_size).
        if self.moe_intermediate_size != self.shared_expert_intermediate_size:
            raise ValueError(
                "routed and shared expert intermediate sizes must match; got "
                f"{self.moe_intermediate_size} vs {self.shared_expert_intermediate_size}"
            )
        self.num_local_experts = self.num_experts  # EP=1
        # Shared expert + sigmoid output gate are handled by NeuronMoEBlock
        # directly (NxDI SharedExperts has no per-token gate).
        self.n_shared_experts = 0

        # MoE router: fp32 softmax with normalized affinities.
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "softmax"
        self.neuron_config.disable_numeric_cc_token = True
        self.neuron_config.normalize_top_k_affinities = bool(self.norm_topk_prob)

        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "vocab_size",
            "linear_num_value_heads",
            "linear_num_key_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "layer_types",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "mtp_num_hidden_layers",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


# ============================================================
# Attention (standard GQA for 16 of 64 layers)
# With output gate: q_proj is 2x sized, split into (query, gate)
# With partial RoPE: only first rope_dim dimensions get rotary
# ============================================================


class Qwen36A3BMRoPEEmbedding(nn.Module):
    """Multimodal Rotary Position Embedding (mRoPE).

    Handles 3D position information (temporal, height, width) for VL paths.
    Position IDs have shape (3, batch_size, seq_len) for T/H/W dimensions.
    For text-only (2D position_ids), broadcasts to 3D with identical positions.
    """

    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim  # 256
        self.rope_dim = config.rope_dim
        self.mrope_section = config.mrope_section  # [11, 11, 10]
        self.mrope_interleaved = getattr(config, "mrope_interleaved", True)
        self.rope_theta = config.rope_theta

        # Validate mrope_section sums to rope_dim // 2 = 32
        assert sum(self.mrope_section) == self.rope_dim // 2, (
            f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)}, "
            f"expected {self.rope_dim // 2}"
        )

    def forward(self, x, position_ids_3d):
        """Compute cos/sin from 3D position IDs.

        Args:
            x: hidden_states (for device/dtype inference)
            position_ids_3d: (3, batch_size, seq_len) -- T, H, W positions

        Returns:
            cos: (batch_size, seq_len, rope_dim)
            sin: (batch_size, seq_len, rope_dim)
        """
        device = x.device
        dtype = torch.float32

        if position_ids_3d.ndim == 2:
            position_ids_3d = position_ids_3d[None, ...].expand(
                3, position_ids_3d.shape[0], -1
            )

        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.rope_dim, 2, dtype=dtype, device=device)
                / self.rope_dim
            )
        )
        inv_freq = inv_freq[None, None, :, None].expand(
            3, position_ids_3d.shape[1], -1, 1
        )
        positions = position_ids_3d[:, :, None, :].float()
        freqs = (inv_freq.float() @ positions).transpose(2, 3)

        # Match HF Qwen3.6 mRoPE layout exactly: start from the temporal
        # frequencies, then splice H/W frequencies into interleaved positions.
        freqs_t = freqs[0]
        if self.mrope_interleaved:
            for dim, offset in enumerate((1, 2), start=1):
                length = self.mrope_section[dim] * 3
                idx = slice(offset, length, 3)
                freqs_t[..., idx] = freqs[dim, ..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)

        return cos, sin


class NeuronQwen36A3BAttention(NeuronAttentionBase):
    """Standard GQA attention with output gate and partial RoPE.

    16 Q heads, 2 KV heads (8:1 GQA), head_dim=256.
    q_proj is doubled (query + gate), split at load time.
    Only the first rope_dim=64 of head_dim=256 gets rotary encoding.

    Uses NeuronAttentionBase infrastructure for QKV projection, KV cache,
    RoPE, and attention computation. Overrides forward() to insert the
    sigmoid output gate between attention output and o_proj.
    """

    def __init__(self, config):
        # Partial RoPE: create mRoPE embedding with rope_dim (64)
        self.rope_dim = config.rope_dim

        # Create QK norm modules (will be passed to base class)
        rms_norm_eps = config.rms_norm_eps
        q_ln = get_rmsnorm_cls()(config.head_dim, rms_norm_eps)
        k_ln = get_rmsnorm_cls()(config.head_dim, rms_norm_eps)

        # Partial RoPE: use standard RotaryEmbedding.
        # For VL with 3D mRoPE positions, cos/sin are pre-computed externally in
        # get_model_output() using Qwen36A3BMRoPEEmbedding and passed as cos_cache/sin_cache.
        rotary_emb = RotaryEmbedding(
            self.rope_dim,  # Only 64 dims get rotary embedding
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
            rms_norm_eps=rms_norm_eps,
            use_qk_norm=False,
            q_layernorm=q_ln,
            k_layernorm=k_ln,
        )

        # Separate mRoPE module for VL 3D position_ids
        self.mrope_emb = Qwen36A3BMRoPEEmbedding(config)

        # Output gate projection: hidden_size -> num_heads * head_dim
        # Populated from the second half of q_proj during state dict conversion.
        self.output_gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
            gather_output=False,
        )

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        """Partial RoPE: only apply rotary embedding to first rope_dim dimensions.

        Q shape: (B, H, S, head_dim) where head_dim=256
        cos/sin shape: (B, S, rope_dim) where rope_dim=64 (from RotaryEmbedding(dim=64))

        Split Q/K along last dim into:
          q_rope (first 64 dims) -- apply RoPE
          q_pass (remaining 192 dims) -- pass through unchanged
        """
        from neuronx_distributed_inference.modules.attention.utils import (
            apply_rotary_pos_emb,
        )

        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

        # Split into rope and pass-through portions
        Q_orig_dtype = Q.dtype
        q_rope = Q[..., : self.rope_dim]  # (B, H, S, 64)
        q_pass = Q[..., self.rope_dim :]  # (B, H, S, 192)
        k_rope = K[..., : self.rope_dim]
        k_pass = K[..., self.rope_dim :]

        # Apply RoPE only to the rope portion
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos_cache, sin_cache)

        # Concatenate back (ensure bf16 is maintained)
        Q = torch.cat([q_rope, q_pass], dim=-1).to(Q_orig_dtype)
        K = torch.cat([k_rope, k_pass], dim=-1).to(Q_orig_dtype)

        return Q, K, cos_cache, sin_cache

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask=None):
        """Prefill path with NKI flash attention for head_dim=256."""
        head_dim = Q.shape[-1]

        # Option B: nkilib flash attention for head_dim > 128
        if _nkilib_flash_attn is not None:
            q_contig = Q.contiguous()
            k_contig = K.contiguous()
            v_contig = V.contiguous()
            scale = 1.0 / math.sqrt(head_dim)
            result = _nkilib_flash_attn(
                q_contig, k_contig, v_contig, scale=scale, use_causal_mask=True
            )
            return result, None

        # Option A: kernel patched globally
        if NKILIB_PATCH_ACTIVE:
            return _flash_fwd_call(Q, K, V, use_causal_mask=True), None

        # Fallback: softmax path (use 3D tensors to avoid compiler ICE with 4D patterns)
        if head_dim > 128:
            # GQA: expand K/V heads to match Q heads
            num_q_heads = Q.shape[1]
            num_kv_heads = K.shape[1]
            if num_q_heads != num_kv_heads:
                kv_rep = num_q_heads // num_kv_heads
                K = (
                    K.unsqueeze(2)
                    .expand(-1, -1, kv_rep, -1, -1)
                    .reshape(bsz, num_q_heads, q_len, head_dim)
                )
                V = (
                    V.unsqueeze(2)
                    .expand(-1, -1, kv_rep, -1, -1)
                    .reshape(bsz, num_q_heads, q_len, head_dim)
                )
            # Reshape to 3D (B*H, S, d) to avoid neuronx-cc codegen ICE with 4D
            # attention weight tensors (NCC_INLA001: Expected 2D tensor but got 4D AP)
            Q_3d = Q.reshape(bsz * num_q_heads, q_len, head_dim)
            K_3d = K.reshape(bsz * num_q_heads, q_len, head_dim)
            V_3d = V.reshape(bsz * num_q_heads, q_len, head_dim)
            attn_weights = torch.bmm(Q_3d, K_3d.transpose(-1, -2)) / math.sqrt(head_dim)
            # Build causal mask for 3D: (1, S, S) broadcast over B*H
            causal_mask = torch.triu(
                torch.full(
                    (q_len, q_len),
                    -65504.0,
                    dtype=attn_weights.dtype,
                    device=attn_weights.device,
                ),
                diagonal=1,
            ).unsqueeze(0)
            attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                Q.dtype
            )
            attn_output = torch.bmm(attn_weights, V_3d)
            # Reshape back to 4D (B, H, S, d)
            return attn_output.reshape(bsz, num_q_heads, q_len, head_dim), None

        return _flash_fwd_call(Q, K, V, use_causal_mask=True), None

    def perform_qwen_chunked_prefill(self, Q, K, V, past_key_value, position_ids):
        """Exact chunked CTE over the full decode cache.

        The current chunk K/V tensors are scattered into the full cache at
        absolute position_ids, then attention for this chunk is computed over
        all cache positions up to the chunk end. This keeps full-attention
        layers correct when model-local chunked prefill feeds context in
        multiple CTE-bucket calls.
        """
        k_cache, v_cache = past_key_value
        B, q_heads, q_len, head_dim = Q.shape
        kv_heads = K.shape[1]
        cache_len = k_cache.shape[2]

        pos = position_ids.long()
        k_index = pos[:, None, :, None].expand(B, kv_heads, q_len, head_dim)
        k_cache = torch.scatter(k_cache, dim=2, index=k_index, src=K.to(k_cache.dtype))
        v_cache = torch.scatter(v_cache, dim=2, index=k_index, src=V.to(v_cache.dtype))

        if q_heads != kv_heads:
            kv_rep = q_heads // kv_heads
            K_full = (
                k_cache.unsqueeze(2)
                .expand(-1, -1, kv_rep, -1, -1)
                .reshape(B, q_heads, cache_len, head_dim)
            )
            V_full = (
                v_cache.unsqueeze(2)
                .expand(-1, -1, kv_rep, -1, -1)
                .reshape(B, q_heads, cache_len, head_dim)
            )
        else:
            K_full = k_cache
            V_full = v_cache

        attn_weights = torch.matmul(Q, K_full.transpose(-1, -2)) / math.sqrt(head_dim)
        cache_positions = torch.arange(cache_len, device=position_ids.device).view(1, 1, 1, -1)
        causal_mask = cache_positions <= pos[:, None, :, None]
        attn_weights = attn_weights.masked_fill(~causal_mask, -65504.0)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
        return torch.matmul(attn_weights, V_full)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        adapter_ids=None,
        active_mask=None,
        **kwargs,
    ):
        """Forward with output gate applied BEFORE o_proj.

        Override NeuronAttentionBase.forward() to insert the sigmoid gate
        between the attention output and o_proj, matching the HF reference:
          gate = sigmoid(gate_proj(pre_attn_hidden))
          attn_output = attn_output * gate
          attn_output = o_proj(attn_output)
        """
        bsz, q_len, _ = hidden_states.shape

        # Use standard 2D position_ids for prep_qkv_tensors.
        rope_pos_ids = position_ids

        # Compute gate from input hidden states (before QKV projection)
        gate = self.output_gate_proj(hidden_states)  # (B, S, num_heads * head_dim)

        # Standard QKV prep (projections, QK norm, RoPE)
        Q, K, V, cos_cache, sin_cache, _residual = self.prep_qkv_tensors(
            rope_pos_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
        )

        qwen_chunked_prefill_active = (
            past_key_value is not None
            and q_len > 1
            and getattr(self.config, "use_qwen_hybrid_chunked_prefill", False)
        )

        if past_key_value is None:
            # Context encoding (prefill)
            attn_output, _flash_strategy = self.perform_prefill(
                Q, K, V, q_len, bsz, attention_mask
            )
        elif qwen_chunked_prefill_active:
            attn_output = self.perform_qwen_chunked_prefill(
                Q, K, V, past_key_value, position_ids
            )
        else:
            # Token generation (decode)
            tkg_mask = attention_mask
            if tkg_mask is not None and tkg_mask.ndim == 2:
                tkg_mask = tkg_mask.unsqueeze(1).unsqueeze(2)  # (B, S) -> (B, 1, 1, S)
            attn_output = self.compute_for_token_gen(
                Q, K, V, position_ids, past_key_value, tkg_mask, active_mask
            )

        # attn_output is (B, H, S, head_dim) -- transpose to (B, S, H*head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Apply sigmoid output gate BEFORE o_proj (matching HF reference)
        attn_output = attn_output * torch.sigmoid(gate)

        # Apply o_proj
        attn_output = self.get_o_proj()(attn_output, adapter_ids=adapter_ids)

        # DEBUG knob (default off): neutralize the GQA attention contribution
        # to localize a structural NaN to attention vs. MoE. Keeps K/V/cos/sin
        # valid so the graph/cache stay well-formed.
        if os.environ.get("A3B_BYPASS_ATTN") == "1":
            attn_output = torch.zeros_like(attn_output)

        # Ensure K, V are in model dtype (bf16) for KV cache update
        # (prevents mixed-precision dynamic-update-slice in neuronx-cc)
        K = K.to(self.torch_dtype)
        V = V.to(self.torch_dtype)
        past_key_value = (K, V)
        return attn_output, past_key_value, cos_cache, sin_cache


# ============================================================
# MoE FFN block (every layer)
# ============================================================


class NeuronMoEBlock(nn.Module):
    """Routed top-k experts + sigmoid-gated shared expert.

    NxDI's `SharedExperts` has no per-token output gate, so the shared
    expert lives in this module directly rather than inside the MoE call.
    Layout:
        moe                      : NxDI MoE module, routed-only (n_shared=0)
        shared_expert.gate_proj  : (H -> shared_I)  column-parallel
        shared_expert.up_proj    : (H -> shared_I)  column-parallel
        shared_expert.down_proj  : (shared_I -> H)  row-parallel
        shared_expert_gate       : (H -> 1)         column-parallel
    Forward: routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x).
    """

    def __init__(self, config: "Qwen36A3BInferenceConfig"):
        super().__init__()
        self.config = config
        self.moe = initialize_moe_module(config=config)

        # Always-on shared expert, owned here so we can apply the sigmoid gate.
        si = config.shared_expert_intermediate_size
        h = config.hidden_size
        self.shared_expert = SharedExpertMLP(h, si)
        # Scalar gate per token. Output is 1, so a parallel linear can't
        # shard along the output dim -- replicate across ranks instead.
        self.shared_expert_gate = nn.Linear(h, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # DEBUG knob (default off): neutralize the MoE FFN contribution to
        # localize a structural NaN to MoE vs. attention. With this set the
        # decoder layer's FFN branch adds zero, so finite logits => MoE was
        # the NaN source.
        if os.environ.get("A3B_BYPASS_MOE") == "1":
            return torch.zeros_like(hidden_states)

        is_spec = (
            self.config.neuron_config.enable_fused_speculation
            and not self.config.neuron_config.is_prefill_stage
        )
        routed = self.moe(
            hidden_states, padding_mask, is_speculative_decoding=is_spec,
        )[0]
        shared = self.shared_expert(hidden_states)
        gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        return routed + gate * shared


class SharedExpertMLP(nn.Module):
    """Always-on shared expert: SwiGLU FFN sized by `shared_expert_intermediate_size`."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, input_is_parallel=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# MTP draft head (self-speculative decoding)
# ============================================================


class NeuronMTPHead(nn.Module):
    """One extra decoder layer that predicts token t+2 from (hidden_t, embed_{t+1}).

    Layout (matches the HF checkpoint):
        embed_norm    : RMSNorm over embed(t+1)
        hidden_norm   : RMSNorm over last main hidden at t
        eh_proj       : Linear(2H, H)
        decoder_layer : one Qwen36A3BDecoderLayer
        final_norm    : RMSNorm
        mtp_lm_head   : ColumnParallelLinear(H, vocab)

    Invoked off the traced main forward by the speculative-decoding driver.
    """

    def __init__(
        self,
        config: "Qwen36A3BInferenceConfig",
        layer_idx: Optional[int] = None,
        layer_type: str = "full_attention",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = (
            layer_idx if layer_idx is not None else config.num_hidden_layers
        )
        self.layer_type = layer_type

        rms_cls = get_rmsnorm_cls()
        h = config.hidden_size
        eps = config.rms_norm_eps

        self.embed_norm = rms_cls(h, eps=eps)
        self.hidden_norm = rms_cls(h, eps=eps)
        self.eh_proj = ColumnParallelLinear(
            2 * h, h, bias=False, gather_output=True,
        )

        # Stand up the decoder layer with the requested attention type by
        # presenting a shallow proxy config whose layer_types extends to the
        # MTP slot.
        forged = _config_with_layer_type(config, self.layer_idx, self.layer_type)
        self.decoder_layer = NeuronQwen36A3BDecoderLayer(forged, self.layer_idx)

        self.final_norm = rms_cls(h, eps=eps)
        self.mtp_lm_head = ColumnParallelLinear(
            h, config.vocab_size, bias=False, gather_output=True,
        )

    def draft_step(
        self,
        prev_hidden: torch.Tensor,
        next_input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
    ):
        """Draft the t+2 logits and surface the draft layer's updated KV.

        Returns ``(draft_logits, present_key_value)`` where ``present_key_value``
        is the ``(K, V)`` tuple the draft decoder layer produced (a dummy KV for a
        linear-attention slot). The on-device draft graph aliases this KV back into
        its own one-layer cache; pure host callers can ignore it.
        """
        # Concat order is [embed | hidden]: the eh_proj weight is loaded straight
        # from the checkpoint's `mtp.fc` (no column repacking), whose input is
        # laid out as [normed embedding of t+1 | normed trunk hidden at t]. Each
        # norm is applied to its own tensor before the join.
        combined = torch.cat(
            [self.embed_norm(next_input_embeds), self.hidden_norm(prev_hidden)],
            dim=-1,
        )
        h_in = self.eh_proj(combined)
        out = self.decoder_layer(
            h_in,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
        )
        draft_logits = self.mtp_lm_head(self.final_norm(out[0]))
        present_key_value = out[1]
        return draft_logits, present_key_value


def _config_with_layer_type(config, layer_idx: int, layer_type: str):
    """Shallow proxy of `config` with layer_types[layer_idx] = layer_type."""

    class _Proxy(config.__class__):  # type: ignore[misc]
        pass

    proxy = _Proxy.__new__(_Proxy)
    proxy.__dict__.update(config.__dict__)
    base = list(config.layer_types)
    if len(base) <= layer_idx:
        base = base + [layer_type] * (layer_idx + 1 - len(base))
    base[layer_idx] = layer_type
    proxy.layer_types = base
    return proxy


# ============================================================
# Decoder Layer (hybrid dispatch -- DeltaNet or GQA, then MoE)
# ============================================================


class NeuronQwen36A3BDecoderLayer(nn.Module):
    """Hybrid decoder layer: dispatches to DeltaNet or standard attention, then MoE FFN."""

    def __init__(self, config: Qwen36A3BInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.layer_idx = layer_idx
        self.config = config

        if self.layer_type == "linear_attention":
            self.linear_attn = NeuronGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = NeuronQwen36A3BAttention(config=config)

        self.mlp = NeuronMoEBlock(config)

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        padding_mask=None,
        cos_cache=None,
        sin_cache=None,
        verify_mode=False,
        **kwargs,
    ):
        residual = hidden_states

        # Layer-boundary markers are XLA custom calls; skip them on the host so
        # the layer runs under cpu_mode (device tracing is unaffected).
        if not cpu_mode():
            hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)

        if verify_mode and self.layer_type == "linear_attention":
            # Verify path: run the 2-token block as an exact single-step
            # recurrence seeded read-only from the live committed DeltaNet
            # buffers. The recurrent S-candidates and conv windows are collected
            # for the host's accept/reject state selection (Stage B.2); in Stage
            # B.1 they are surfaced as a side payload but not aliased.
            seq_ids = kwargs.get("seq_ids", None)
            attn_out, S_stack, conv_cand = self.linear_attn.verify_block(
                hidden_states, seq_ids
            )
            hidden_states = residual + attn_out
            # Dummy KV so the verify graph's private KVCacheManager keeps a
            # uniform per-layer KV list (DeltaNet layers carry no GQA cache).
            b, s, _ = attn_out.shape
            dummy_k = torch.zeros(
                b,
                self.linear_attn.kv_heads_per_rank,
                s,
                self.linear_attn.head_dim,
                dtype=attn_out.dtype,
                device=attn_out.device,
            )
            present_key_value = (dummy_k, torch.zeros_like(dummy_k))
            deltanet_states = (S_stack, conv_cand)
        elif self.layer_type == "linear_attention":
            # DeltaNet path
            attn_out, dummy_kv, new_rec_state, new_conv_state = self.linear_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                **kwargs,
            )
            hidden_states = residual + attn_out
            present_key_value = dummy_kv
            deltanet_states = (
                None
                if getattr(self.config, "use_hybrid_cache_manager", False)
                else (new_rec_state, new_conv_state)
            )
        else:
            deltanet_states = None
            # Standard attention path
            hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                **kwargs,
            )
            hidden_states = residual + hidden_states

        # Dense MLP FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if not cpu_mode():
            hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (
            hidden_states,
            present_key_value,
            cos_cache,
            sin_cache,
            None,
            deltanet_states,
        )
        return outputs


# ============================================================
# Hybrid Cache Manager (opt-in)
# ============================================================


class HybridDeltaNetCacheManager(KVCacheManager):
    """Layer-type-aware cache manager: DeltaNet (recurrent+conv state) or GQA (K/V)."""

    def __init__(self, config: Qwen36A3BInferenceConfig, num_kv_head, **kwargs):
        self.layer_types = list(config.layer_types)
        self._validate_hybrid_config(config)
        super().__init__(config, num_kv_head=num_kv_head, **kwargs)

        dtype = (
            config.neuron_config.attention_dtype
            if config.neuron_config.attention_dtype is not None
            else config.neuron_config.torch_dtype
        )
        cache_dtype = getattr(self, "cache_dtype", dtype)
        max_batch_size = (
            config.neuron_config.kv_cache_batch_size
            + config.neuron_config.kv_cache_padding_size
        )
        tp_degree = config.neuron_config.tp_degree
        if config.linear_num_value_heads % tp_degree != 0:
            raise ValueError(
                f"linear_num_value_heads={config.linear_num_value_heads} must be divisible "
                f"by tp_degree={tp_degree}"
            )
        if config.linear_num_key_heads % tp_degree != 0:
            raise ValueError(
                f"linear_num_key_heads={config.linear_num_key_heads} must be divisible "
                f"by tp_degree={tp_degree}"
            )
        local_num_value_heads = config.linear_num_value_heads // tp_degree
        local_num_key_heads = config.linear_num_key_heads // tp_degree
        recurrent_shape = [
            max_batch_size,
            local_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        ]
        conv_dim = (
            2 * local_num_key_heads * config.linear_key_head_dim
            + local_num_value_heads * config.linear_value_head_dim
        )
        conv_shape = [
            max_batch_size,
            conv_dim,
            config.linear_conv_kernel_dim - 1,
        ]

        params = []
        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "linear_attention":
                params.append(
                    nn.Parameter(torch.zeros(recurrent_shape, dtype=dtype), requires_grad=False)
                )
                params.append(
                    nn.Parameter(torch.zeros(conv_shape, dtype=dtype), requires_grad=False)
                )
            else:
                k_shape = self.k_shapes[layer_idx] if hasattr(self, "k_shapes") else self.k_shape
                v_shape = self.v_shapes[layer_idx] if hasattr(self, "v_shapes") else self.v_shape
                params.append(
                    nn.Parameter(torch.zeros(k_shape, dtype=cache_dtype), requires_grad=False)
                )
                params.append(
                    nn.Parameter(torch.zeros(v_shape, dtype=cache_dtype), requires_grad=False)
                )

        self.past_key_values = nn.ParameterList(params)

    @staticmethod
    def _validate_hybrid_config(config: Qwen36A3BInferenceConfig):
        nc = config.neuron_config
        unsupported = []
        if nc.is_block_kv_layout:
            unsupported.append("block KV layout")
        if getattr(nc, "kv_quant_config", None) is not None or getattr(nc, "kv_cache_quant", False):
            unsupported.append("KV cache quantization")
        if nc.enable_fused_speculation or nc.speculation_length > 0 or nc.is_medusa:
            unsupported.append("speculative decoding")
        if getattr(nc, "enable_eagle_speculation", False) or getattr(nc, "is_eagle_draft", False):
            unsupported.append("EAGLE speculation")
        if nc.flash_decoding_enabled:
            unsupported.append("flash decoding")
        if nc.attention_dp_degree > 1:
            unsupported.append("attention data parallelism")
        if nc.kv_cache_tiling:
            unsupported.append("KV cache tiling")
        if nc.padding_side != "right":
            unsupported.append("left padding")
        if nc.is_continuous_batching:
            unsupported.append("continuous batching")
        if unsupported:
            raise ValueError(
                "HybridDeltaNetCacheManager v1 does not support: "
                + ", ".join(unsupported)
            )

    def _is_deltanet_layer(self, idx: int) -> bool:
        return self.layer_types[idx] == "linear_attention"

    def get_seq_length(self, past_key_values=None):
        for idx, layer_type in enumerate(self.layer_types):
            if layer_type != "linear_attention":
                if past_key_values is None:
                    _, v_cache = self._fetch_cache(idx)
                elif len(past_key_values) == len(self.past_key_values):
                    v_cache = past_key_values[2 * idx + 1]
                else:
                    v_cache = past_key_values[idx][1]
                return v_cache.shape[2]
        return 0

    def get_deltanet_state_by_layer_id(self, idx, kvcache_buffer=None, seq_ids=None):
        recurrent_state, conv_state = self._fetch_cache(idx, kvcache_buffer)
        if seq_ids is not None:
            cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids)
            recurrent_state = torch.index_select(recurrent_state, dim=0, index=cache_idx)
            conv_state = torch.index_select(conv_state, dim=0, index=cache_idx)
        elif self.kv_cache_padding_size > 0:
            recurrent_state = recurrent_state[: -self.kv_cache_padding_size]
            conv_state = conv_state[: -self.kv_cache_padding_size]
        return recurrent_state, conv_state

    def get_cache(
        self,
        seq_len: int,
        skip_slice=False,
        kvcache_buffer=None,
        seq_ids=None,
        windowed_context_encoding_window_idx=-1,
        **kwargs,
    ):
        past_key_values = []
        for idx in range(len(self.past_key_values) // 2):
            if self._is_deltanet_layer(idx):
                past_key_values.append(
                    list(self.get_deltanet_state_by_layer_id(idx, kvcache_buffer, seq_ids))
                )
            else:
                past_key_values.append(
                    list(
                        self.get_kv_by_layer_id(
                            idx=idx,
                            skip_slice=skip_slice,
                            seq_len=seq_len,
                            kvcache_buffer=kvcache_buffer,
                            seq_ids=seq_ids,
                            windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                            **kwargs,
                        )
                    )
                )
        return past_key_values

    def update_cache(
        self,
        is_for_context_encoding: bool,
        seq_ids: torch.Tensor,
        position_ids: torch.Tensor,
        new_key_values: List[torch.Tensor],
        seq_len: int,
        scatter_index=None,
        kv_active_mask=None,
        kvcache_buffer=None,
        windowed_context_encoding_window_idx: int = -1,
        **kwargs,
    ):
        updated_cache = []
        for idx, kv_per_layer in enumerate(new_key_values):
            if self._is_deltanet_layer(idx):
                recurrent_state, conv_state = self.update_deltanet_state_by_layer_id(
                    idx=idx,
                    seq_ids=seq_ids,
                    state_per_layer=kv_per_layer,
                    kvcache_buffer=kvcache_buffer,
                )
            elif kwargs.get("qwen_chunked_prefill_update", False):
                recurrent_state, conv_state = self.update_qwen_chunked_kv_by_layer_id(
                    idx=idx,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    kv_per_layer=kv_per_layer,
                    kvcache_buffer=kvcache_buffer,
                    valid_mask=kwargs.get("qwen_chunked_valid_mask", None),
                )
            else:
                recurrent_state, conv_state = self.update_kv_by_layer_id(
                    idx=idx,
                    is_for_context_encoding=is_for_context_encoding,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    kv_per_layer=kv_per_layer,
                    seq_len=seq_len,
                    scatter_index=scatter_index,
                    kv_active_mask=kv_active_mask,
                    kvcache_buffer=kvcache_buffer,
                    windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                    **kwargs,
                )
            updated_cache.append(recurrent_state)
            updated_cache.append(conv_state)
        return updated_cache

    def update_qwen_chunked_kv_by_layer_id(
        self,
        idx: int,
        seq_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_per_layer: Tuple[torch.Tensor, torch.Tensor],
        kvcache_buffer=None,
        valid_mask=None,
    ):
        latest_k, latest_v = kv_per_layer
        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)
        latest_k = latest_k.to(k_cache.dtype)
        latest_v = latest_v.to(v_cache.dtype)

        if seq_ids is not None:
            cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids)
            selected_k = torch.index_select(k_cache, dim=0, index=cache_idx)
            selected_v = torch.index_select(v_cache, dim=0, index=cache_idx)
        else:
            cache_idx = None
            selected_k = k_cache[: latest_k.shape[0]]
            selected_v = v_cache[: latest_v.shape[0]]

        pos = position_ids.long()
        k_index = pos[:, None, :, None].expand_as(latest_k)
        v_index = pos[:, None, :, None].expand_as(latest_v)

        if valid_mask is not None:
            valid = valid_mask.to(torch.bool)[:, None, :, None]
            old_k = torch.gather(selected_k, dim=2, index=k_index)
            old_v = torch.gather(selected_v, dim=2, index=v_index)
            latest_k = torch.where(valid, latest_k, old_k)
            latest_v = torch.where(valid, latest_v, old_v)

        updated_k = torch.scatter(selected_k, dim=2, index=k_index, src=latest_k)
        updated_v = torch.scatter(selected_v, dim=2, index=v_index, src=latest_v)

        if cache_idx is not None:
            k_row_index = cache_idx.view(-1, 1, 1, 1).expand_as(updated_k)
            v_row_index = cache_idx.view(-1, 1, 1, 1).expand_as(updated_v)
            k_cache = torch.scatter(k_cache, dim=0, index=k_row_index, src=updated_k)
            v_cache = torch.scatter(v_cache, dim=0, index=v_row_index, src=updated_v)
            return k_cache, v_cache

        if updated_k.shape[0] == k_cache.shape[0]:
            return updated_k + k_cache * 0, updated_v + v_cache * 0

        pad_rows = k_cache.shape[0] - updated_k.shape[0]
        if pad_rows > 0:
            updated_k = torch.cat([updated_k, k_cache[updated_k.shape[0] :] * 0], dim=0)
            updated_v = torch.cat([updated_v, v_cache[updated_v.shape[0] :] * 0], dim=0)
        return updated_k + k_cache * 0, updated_v + v_cache * 0

    def update_deltanet_state_by_layer_id(
        self,
        idx: int,
        seq_ids: torch.Tensor,
        state_per_layer: Tuple[torch.Tensor, torch.Tensor],
        kvcache_buffer=None,
    ):
        latest_recurrent, latest_conv = state_per_layer
        recurrent_cache, conv_cache = self._fetch_cache(idx, kvcache_buffer)
        latest_recurrent = latest_recurrent.to(recurrent_cache.dtype)
        latest_conv = latest_conv.to(conv_cache.dtype)

        if latest_recurrent.shape[0] == recurrent_cache.shape[0] and seq_ids is None:
            return (
                latest_recurrent + recurrent_cache * 0,
                latest_conv + conv_cache * 0,
            )

        if seq_ids is not None:
            cache_idx = self.get_cache_update_index_for_seq_ids(seq_ids)
            recurrent_index = cache_idx.view(-1, 1, 1, 1).expand_as(latest_recurrent)
            conv_index = cache_idx.view(-1, 1, 1).expand_as(latest_conv)
            recurrent_cache = torch.scatter(
                input=recurrent_cache,
                dim=0,
                index=recurrent_index,
                src=latest_recurrent,
            )
            conv_cache = torch.scatter(
                input=conv_cache,
                dim=0,
                index=conv_index,
                src=latest_conv,
            )
            return recurrent_cache, conv_cache

        pad_size = recurrent_cache.shape[0] - latest_recurrent.shape[0]
        if pad_size > 0:
            latest_recurrent = torch.cat(
                [latest_recurrent, recurrent_cache[latest_recurrent.shape[0] :] * 0],
                dim=0,
            )
            latest_conv = torch.cat(
                [latest_conv, conv_cache[latest_conv.shape[0] :] * 0],
                dim=0,
            )
        return latest_recurrent + recurrent_cache * 0, latest_conv + conv_cache * 0


# ============================================================
# Model
# ============================================================


class NeuronQwen36A3BModel(NeuronBaseModel):
    def setup_attr_for_model(self, config: Qwen36A3BInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Qwen36A3BInferenceConfig):
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
                NeuronQwen36A3BDecoderLayer(config, layer_idx)
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

        # mRoPE embedding for VL paths.
        self.mrope_emb = Qwen36A3BMRoPEEmbedding(config)

        # MTP draft head, called off the traced main forward by the
        # speculative-decoding driver in inference_qwen36_a3b.py.
        self.mtp_head = (
            NeuronMTPHead(config) if config.mtp_num_hidden_layers > 0 else None
        )

        # When the trunk hidden is surfaced (MTP draft input), carry it through an
        # input_output-aliased buffer so it rides the SAME path as the DeltaNet
        # state tensors (see _deltanet_state_params) rather than becoming a fresh
        # non-aliased output -- which would desync the NxD trace output packer.
        self.trunk_hidden_buffer = None
        if (
            getattr(config, "output_trunk_hidden", False)
            or config.mtp_num_hidden_layers > 0
        ):
            alloc_bs = getattr(config.neuron_config, "max_batch_size", 1)
            self.trunk_hidden_buffer = nn.Parameter(
                torch.zeros(
                    alloc_bs,
                    1,
                    config.hidden_size,
                    dtype=config.neuron_config.torch_dtype,
                ),
                requires_grad=False,
            )

        # Verify-backbone candidate scratch buffers (Stage B.2). Created on the
        # MAIN model -- NOT only on NeuronVerifyModel -- when the verify graph is
        # enabled, because NxD's shared StateInitializer derives its zero-init
        # state keys from the FIRST traced graph's metaneff only (CTE). A state
        # key that exists solely in the verify graph (a verify-only nn.Parameter)
        # is never created, so the verify graph's initialize() fails with
        # "Missing state tensor". Registering these buffers on the main model too
        # makes their state keys appear in the CTE/TKG metaneffs (emitted there as
        # zero passthrough), so the shared StateInitializer allocates them and the
        # verify graph finds them. The main graphs never write them (passthrough),
        # so this is byte-identical when the flag is off and inert for decode when
        # on. These are the dedicated candidate scratch buffers; the verify graph
        # writes the real S/conv candidates + per-position trunk hidden into them.
        self.recurrent_cand_buffers = None
        self.conv_cand_buffers = None
        self.verify_trunk_buffer = None
        if getattr(config, "enable_verify_backbone", False):
            self._init_verify_candidate_buffers(config)

    def _init_verify_candidate_buffers(self, config: Qwen36A3BInferenceConfig):
        """Allocate the per-DeltaNet-layer recurrent S-stack / conv-window
        candidate buffers + the single per-position trunk-hidden buffer.

        Dims are read from each DeltaNet layer at runtime (num_v_heads,
        head_k_dim, head_v_dim, conv_dim, conv_kernel_size). At TP4/B1 this is
        recurrent [1,2,8,128,128], conv [1,2,2560,3], trunk [1,2,H]. seq_len=2 is
        the k+1 verify block (k=1).
        """
        alloc_bs = getattr(config.neuron_config, "max_batch_size", 1)
        seq_len = 2
        torch_dtype = config.neuron_config.torch_dtype

        rec_cands = []
        conv_cands = []
        for layer in self.layers:
            if not hasattr(layer, "linear_attn"):
                continue
            dn = layer.linear_attn
            rec_cands.append(
                nn.Parameter(
                    torch.zeros(
                        alloc_bs,
                        seq_len,
                        dn.num_v_heads,
                        dn.head_k_dim,
                        dn.head_v_dim,
                        dtype=torch_dtype,
                    ),
                    requires_grad=False,
                )
            )
            conv_cands.append(
                nn.Parameter(
                    torch.zeros(
                        alloc_bs,
                        seq_len,
                        dn.conv_dim,
                        dn.conv_kernel_size - 1,
                        dtype=torch_dtype,
                    ),
                    requires_grad=False,
                )
            )
        self.recurrent_cand_buffers = nn.ParameterList(rec_cands)
        self.conv_cand_buffers = nn.ParameterList(conv_cands)
        self.verify_trunk_buffer = nn.Parameter(
            torch.zeros(alloc_bs, seq_len, config.hidden_size, dtype=torch_dtype),
            requires_grad=False,
        )

    @property
    def _verify_candidate_params(self):
        """Candidate scratch buffers in alias order: per DeltaNet layer the
        recurrent S-stack buffer then the conv-window buffer (interleaved in
        layer order), then the single per-position trunk-hidden buffer.

        Surfaced as input_output-aliased outputs AFTER the seed-passthrough
        states (``_deltanet_state_params``). Empty when the verify graph is
        disabled. Disjoint from ``_deltanet_state_params`` by construction --
        these are dedicated scratch buffers, never the live recurrent/conv seed
        buffers.
        """
        if self.verify_trunk_buffer is None:
            return []
        params = []
        for rec, conv in zip(self.recurrent_cand_buffers, self.conv_cand_buffers):
            params.append(rec)
            params.append(conv)
        params.append(self.verify_trunk_buffer)
        return params

    def init_inference_optimization(self, config: Qwen36A3BInferenceConfig):
        super().init_inference_optimization(config)
        if getattr(config, "use_hybrid_cache_manager", False):
            self.kv_mgr = HybridDeltaNetCacheManager(
                config,
                num_kv_head=self.num_key_value_heads,
                global_rank=self.rank_util,
                attention_chunk_size=self.attention_chunk_size,
                sliding_window=self.sliding_window,
                windowed_context_encoding_size=self.windowed_context_encoding_size,
                layer_to_cache_size_mapping=self.layer_to_cache_size_mapping,
            )

    @property
    def _deltanet_state_params(self):
        """Return DeltaNet state nn.Parameters in alias order."""
        params = []
        for layer in self.layers:
            if hasattr(layer, "linear_attn"):
                params.append(layer.linear_attn.recurrent_state_buffer)
                params.append(layer.linear_attn.conv_state_buffer)
        # Trunk hidden rides the same alias path as the last "state" entry.
        if getattr(self, "trunk_hidden_buffer", None) is not None:
            params.append(self.trunk_hidden_buffer)
        return params

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
        """Scatter vision embeddings into text input embeddings at image token positions."""
        _, max_positions, embedding_dim = inputs_embeds.shape
        h_new = inputs_embeds.clone()
        vision_flat = vision_embeddings.view(-1, embedding_dim)
        positions_flat = vision_mask.view(-1)
        h_new.view(-1, embedding_dim).index_put_(
            (positions_flat,), vision_flat, accumulate=False
        )
        return h_new

    def get_model_output(
        self,
        input_ids=None,
        seq_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        active_mask=None,
        inputs_embeds=None,
        prev_hidden=None,
        adapter_ids=None,
        rotary_position_ids=None,
        update_cache=False,
        is_for_context_encoding=False,
        vision_embeddings=None,
        vision_mask=None,
        local_attn_mask=None,
        windowed_context_encoding_window_idx=-1,
        padding_mask=None,
        **kwargs,
    ):
        """Override to collect DeltaNet state tensors from decoder layers."""
        batch_size, seq_length = input_ids.shape[:2]
        if self.config.neuron_config.layer_boundary_markers:
            input_ids = ModuleMarkerStartWrapper()(input_ids)

        past_key_values_length = 0
        if past_key_values is not None:
            if hasattr(self.kv_mgr, "get_seq_length"):
                past_key_values_length = self.kv_mgr.get_seq_length(past_key_values)
            else:
                past_key_values_length = past_key_values[0][1].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # CRITICAL: Zero out embeddings for padding tokens so DeltaNet recurrence
        # is not polluted. DeltaNet has no attention mask -- it processes all
        # sequence positions through a linear recurrence.  Padding tokens have
        # real embedding vectors which corrupt the recurrence state.
        # The mask is [B, S, 1] float with 1.0 for real tokens, 0.0 for padding.
        if (
            is_for_context_encoding
            and attention_mask is not None
            and attention_mask.ndim == 2
        ):
            deltanet_padding_mask = attention_mask.unsqueeze(-1).to(
                inputs_embeds.dtype
            )
        else:
            deltanet_padding_mask = (
                (input_ids != self.padding_idx).unsqueeze(-1).to(inputs_embeds.dtype)
            )
        if is_for_context_encoding:
            inputs_embeds = inputs_embeds * deltanet_padding_mask

        # Vision embedding injection. Text-only calls still pass dummy vision
        # tensors to keep the traced input signature stable; those tensors have
        # one dummy entry per text token and must not overwrite text embeddings.
        if (vision_embeddings is not None) and (vision_mask is not None):
            if vision_embeddings.dtype != self.config.neuron_config.torch_dtype:
                vision_embeddings = vision_embeddings.to(
                    self.config.neuron_config.torch_dtype
                )
            has_real_vision_inputs = (
                vision_embeddings.ndim == 3
                and vision_mask.ndim == 3
                and vision_embeddings.shape[1] != seq_length
            )
            if is_for_context_encoding and has_real_vision_inputs:
                inputs_embeds = self.encode_vision_to_input(
                    inputs_embeds, vision_embeddings, vision_mask
                )
            elif is_for_context_encoding and vision_embeddings.numel() > 0:
                inputs_embeds = inputs_embeds + vision_embeddings.sum() * 0
                inputs_embeds = (
                    inputs_embeds + vision_mask.sum().to(inputs_embeds.dtype) * 0
                )

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        hidden_states = inputs_embeds

        # Get KV cache for TKG and for model-local chunked CTE.
        use_qwen_chunked_prefill = (
            is_for_context_encoding
            and getattr(self.config, "use_qwen_hybrid_chunked_prefill", False)
        )
        cache_size = (
            self.config.neuron_config.seq_len
            if use_qwen_chunked_prefill
            else self.n_positions
        )
        if (not is_for_context_encoding) or use_qwen_chunked_prefill:
            if self.kv_mgr is not None:
                past_key_values = self.kv_mgr.get_cache(
                    seq_ids=seq_ids,
                    seq_len=cache_size,
                    is_for_context_encoding=is_for_context_encoding,
                    windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                    **kwargs,
                )

        # Decoder layers
        next_decoder_cache = ()
        deltanet_state_tensors = []
        cos_cache = None
        sin_cache = None

        # Convert 2D attention_mask to 4D causal mask for CTE
        if (
            attention_mask is not None
            and attention_mask.ndim == 2
            and is_for_context_encoding
        ):
            causal = torch.ones(
                (seq_length, seq_length),
                dtype=torch.bool,
                device=attention_mask.device,
            ).tril()
            padding_4d = attention_mask[:, None, None, :].to(torch.bool)
            attention_mask = (causal[None, None, :, :] & padding_4d).to(
                attention_mask.dtype
            )

        # Pre-compute MRoPE-interleaved cos/sin. Qwen3.6 always uses the
        # MRoPE form (even for text-only inputs); the HF reference broadcasts
        # 2D position_ids to (3, B, S) with identical positions then runs the
        # interleaved-MRoPE mixing. Plain RoPE on the first 64 dims is NOT
        # equivalent and produces garbage logits.
        if rotary_position_ids is not None and rotary_position_ids.ndim == 3:
            cos_cache, sin_cache = self.mrope_emb(inputs_embeds, rotary_position_ids)
        elif cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.mrope_emb(inputs_embeds, position_ids)

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                seq_ids=seq_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                adapter_ids=adapter_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                rotary_position_ids=rotary_position_ids,
                kv_mgr=self.kv_mgr,
                get_kv_per_layer=False,
                update_kv_per_layer=False,
                idx=idx,
                is_for_context_encoding=is_for_context_encoding,
                seq_len=cache_size,
                residual=None,
                local_mask=local_attn_mask,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                padding_mask=padding_mask,
                deltanet_padding_mask=deltanet_padding_mask,
                qwen_chunked_prefill_update=use_qwen_chunked_prefill,
                qwen_chunked_valid_mask=deltanet_padding_mask.squeeze(-1)
                if use_qwen_chunked_prefill
                else None,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            kv = layer_outputs[1]
            next_decoder_cache += (kv,)
            cos_cache, sin_cache = layer_outputs[2:4]

            # Collect DeltaNet state tensors
            deltanet_states = layer_outputs[5] if len(layer_outputs) > 5 else None
            if deltanet_states is not None:
                deltanet_state_tensors.append(deltanet_states[0])
                deltanet_state_tensors.append(deltanet_states[1])

        # Update KV cache
        if update_cache:
            next_decoder_cache = self.kv_mgr.update_cache(
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=next_decoder_cache,
                seq_len=cache_size,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                qwen_chunked_prefill_update=use_qwen_chunked_prefill,
                qwen_chunked_valid_mask=deltanet_padding_mask.squeeze(-1)
                if use_qwen_chunked_prefill
                else None,
                **kwargs,
            )

        # MTP consumes the trunk hidden state *before* the final norm (it applies
        # its own pre_fc norm). Stash it so forward() can surface it for the draft
        # head. Shape [B, n_active, H]; kept when the MTP head is present or when
        # output_trunk_hidden requests it be surfaced as a trailing output.
        if self.mtp_head is not None or getattr(
            self.config, "output_trunk_hidden", False
        ):
            self._mtp_prenorm_hidden = hidden_states

        hidden_states = self.norm(hidden_states)

        self._deltanet_updated_states = deltanet_state_tensors

        return (hidden_states, next_decoder_cache)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden=None,
        adapter_ids=None,
        accepted_indices=None,
        current_length=None,
        medusa_mask=None,
        scatter_index=None,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
        tile_q_indices=None,
        tile_block_tables=None,
        tile_masks=None,
        inputs_embeds=None,
        kv_cache=None,
        active_mask=None,
        rotary_position_id=None,
        vision_embeddings=None,
        vision_mask=None,
    ):
        """Override base forward to append DeltaNet state tensors to output."""
        prev_hidden = self.set_none_if_empty(prev_hidden)
        adapter_ids = self.set_none_if_empty(adapter_ids)
        accepted_indices = self.set_none_if_empty(accepted_indices)
        current_length = self.set_none_if_empty(current_length)
        medusa_mask = self.set_none_if_empty(medusa_mask)
        scatter_index = self.set_none_if_empty(scatter_index)
        slot_mapping = self.set_none_if_empty(slot_mapping)
        active_block_table = self.set_none_if_empty(active_block_table)
        num_queries = self.set_none_if_empty(num_queries)
        computed_context_lens = self.set_none_if_empty(computed_context_lens)
        tile_q_indices = self.set_none_if_empty(tile_q_indices)
        tile_block_tables = self.set_none_if_empty(tile_block_tables)
        tile_masks = self.set_none_if_empty(tile_masks)
        inputs_embeds = self.set_none_if_empty(inputs_embeds)
        kv_cache = self.set_none_if_empty(kv_cache)
        active_mask = self.set_none_if_empty(active_mask)
        rotary_position_id = self.set_none_if_empty(rotary_position_id)
        vision_embeddings = self.set_none_if_empty(vision_embeddings)
        vision_mask = self.set_none_if_empty(vision_mask)

        is_for_context_encoding = position_ids.shape[-1] != 1 and not (
            hasattr(self.neuron_config, "speculation_length")
            and position_ids.shape[-1] == self.neuron_config.speculation_length
        )

        seq_ids = seq_ids.to(torch.int32)
        attn_mask = attention_mask

        hidden_states, updated_kv_cache = self.get_model_output(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,
            adapter_ids=adapter_ids,
            rotary_position_ids=rotary_position_id,
            update_cache=True,
            is_for_context_encoding=is_for_context_encoding,
            padding_mask=None,
            active_block_table=active_block_table,
            scatter_index=slot_mapping
            if getattr(self, "is_block_kv_layout", False)
            else scatter_index,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )

        batch_size = input_ids.shape[0]
        if not getattr(self, "sliced_hidden", False):
            if not is_for_context_encoding:
                pass
            else:
                if getattr(self.config, "use_qwen_hybrid_chunked_prefill", False):
                    if attention_mask is not None and attention_mask.ndim == 2:
                        index = (
                            attention_mask.to(torch.long).sum(dim=1, keepdim=True)
                            - 1
                        ).clamp(min=0)
                    else:
                        index = (
                            (input_ids != self.padding_idx)
                            .sum(dim=1, keepdim=True)
                            .long()
                            - 1
                        ).clamp(min=0)
                else:
                    index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if hasattr(self.lm_head, "pad_size"):
            if self.lm_head.gather_output:
                rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
                world_size = 1
            else:
                from neuronx_distributed.parallel_layers import parallel_state

                rank_id = self.rank_util.get_rank()
                world_size = torch.distributed.get_world_size(
                    group=self.lm_head.tensor_parallel_group
                )
            from neuronx_distributed_inference.models.model_base import (
                mask_padded_logits,
            )

            logits = mask_padded_logits(
                logits, rank_id, world_size, pad_size=self.lm_head.pad_size
            )

        if self.on_device_sampling:
            res = self._sample_on_device(
                logits, sampling_params, False, is_for_context_encoding
            )
        else:
            res = logits

        outputs = [res]
        if self.neuron_config.output_logits:
            outputs += [logits]
        outputs += updated_kv_cache

        # Surface the pre-final-norm trunk hidden (MTP draft input) by appending it
        # to the DeltaNet state-tensor list, so it is carried via the SAME
        # input_output_alias path as the recurrent/conv states (its buffer is the
        # last entry of _deltanet_state_params). Routing it through the aliased
        # path -- instead of a fresh non-aliased output -- keeps the NxD trace
        # output packer layout consistent. Slice the last real position -> [B,1,H],
        # mirroring the logits last-position gather so prefill picks the true last
        # valid token.
        if (
            getattr(self.config, "output_trunk_hidden", False)
            and getattr(self, "trunk_hidden_buffer", None) is not None
            and hasattr(self, "_mtp_prenorm_hidden")
            and hasattr(self, "_deltanet_updated_states")
        ):
            trunk_hidden = self._mtp_prenorm_hidden
            if is_for_context_encoding and trunk_hidden.shape[1] > 1:
                if getattr(self.config, "use_qwen_hybrid_chunked_prefill", False):
                    if attention_mask is not None and attention_mask.ndim == 2:
                        trunk_index = (
                            attention_mask.to(torch.long).sum(dim=1, keepdim=True)
                            - 1
                        ).clamp(min=0)
                    else:
                        trunk_index = (
                            (input_ids != self.padding_idx)
                            .sum(dim=1, keepdim=True)
                            .long()
                            - 1
                        ).clamp(min=0)
                else:
                    trunk_index = torch.max(
                        position_ids, dim=1, keepdim=True
                    ).indices
                trunk_index = trunk_index.unsqueeze(1).expand(
                    batch_size, 1, self.hidden_size
                )
                trunk_hidden = torch.gather(trunk_hidden, dim=1, index=trunk_index)
            else:
                trunk_hidden = trunk_hidden[:, -1:, :]

            # Match the buffer shape [alloc_bs, 1, H] + add the alias dependency.
            trunk_hidden = trunk_hidden.to(self.trunk_hidden_buffer.dtype)
            alloc_bs = self.trunk_hidden_buffer.shape[0]
            if batch_size < alloc_bs:
                trunk_hidden = torch.cat(
                    [trunk_hidden, self.trunk_hidden_buffer[batch_size:] * 0], dim=0
                )
            trunk_hidden = trunk_hidden + self.trunk_hidden_buffer * 0
            self._deltanet_updated_states = list(self._deltanet_updated_states) + [
                trunk_hidden
            ]

        # Append DeltaNet state tensors (+ trunk hidden) for input_output_aliases.
        if (
            not getattr(self.config, "use_hybrid_cache_manager", False)
            and hasattr(self, "_deltanet_updated_states")
        ):
            outputs += self._deltanet_updated_states

        # When the verify backbone graph is enabled, emit its candidate scratch
        # buffers as trailing zero PASSTHROUGH outputs HERE too (CTE/TKG), so
        # their state keys appear in the first graph's metaneff and NxD's shared
        # StateInitializer zero-allocates them (see init_inference_optimization).
        # The main graphs never write candidates -- this is inert passthrough that
        # keeps the state-key set consistent across graphs. Flag-off: skipped, so
        # the decode output layout is byte-identical.
        if (
            not getattr(self.config, "use_hybrid_cache_manager", False)
            and getattr(self, "verify_trunk_buffer", None) is not None
        ):
            for param in self._verify_candidate_params:
                outputs.append(param + param * 0)

        return outputs


# ============================================================
# MTP draft graph (third compiled graph)
# ============================================================


def _config_for_mtp_draft(config: Qwen36A3BInferenceConfig):
    """Deep-copy `config` collapsed to a single full-attention layer.

    The MTP draft graph owns a one-layer KV cache (the draft decoder layer is a
    standard full-attention layer), so the cache manager is built from a config
    whose `num_hidden_layers == 1` and `layer_types == ["full_attention"]`.
    """
    import copy as _copy

    draft_config = _copy.deepcopy(config)
    draft_config.num_hidden_layers = 1
    draft_config.layer_types = ["full_attention"]
    return draft_config


class NeuronMTPDraftModel(NeuronBaseModel):
    """Traceable wrapper around the MTP draft head for its own compiled graph.

    Reuses the main model's tied embeddings and the `NeuronMTPHead` (one
    eh_proj + decoder layer + norms + LM head) to draft the t+2 token from the
    trunk hidden at t and the embedding of the committed token t+1. Carries a
    private one-layer KV cache so the draft layer's attention can be aliased
    independently of the main graph.

    The forward signature mirrors NeuronQwen36A3BModel.forward's 24 positional
    args verbatim so the shared NxDModelExecutor sees a UNIFORM input arity
    across all three compiled graphs (CTE/TKG/MTP). Only three slots are read:
    ``input_ids`` (the committed next-token ids [B,1]), ``prev_hidden`` (the
    trunk hidden [B,1,H] at arg index 5), and ``position_ids``/``seq_ids``. All
    other slots are empty placeholders. The non-empty prev_hidden at idx 5 (vs
    the empty [0] placeholder the TKG path passes there) is what lets the
    shape-router disambiguate this graph from token-generation.

    Returns (draft_logits, draft_k, draft_v) where the trailing KV is the draft
    layer's updated one-layer cache, aliased back into self.draft_kv_mgr by
    MTPDraftModelInstance.get().

    The KV cache manager is named ``draft_kv_mgr`` (NOT the base
    ``kv_mgr``) on purpose. NxD's shared StateInitializer keys every graph's
    INPUT_STATE by the parameter's dotted checkpoint name, so a draft cache
    named ``kv_mgr.past_key_values.0/1`` would COLLIDE with the full stack's
    layer-0 GQA buffer (same name across graphs => same physical HBM). The
    distinct ``draft_kv_mgr.past_key_values.*`` key gives the draft graph a
    truly-private one-layer cache that cannot clobber the live GQA cache the
    verify graph reads.
    """

    def setup_attr_for_model(self, config: Qwen36A3BInferenceConfig):
        self.on_device_sampling = False
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Qwen36A3BInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Tied to the main model's embeddings (same key `embed_tokens.weight`).
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )

        # The draft head (eh_proj + one decoder layer + norms + LM head).
        self.mtp_head = NeuronMTPHead(config)

        # mRoPE cos/sin for the draft layer's GQA attention (matches the main
        # stack, which always uses the interleaved mRoPE form).
        self.mrope_emb = Qwen36A3BMRoPEEmbedding(config)

    def init_inference_optimization(self, config: Qwen36A3BInferenceConfig):
        # Build a private one-layer KV cache manager instead of the full stack's.
        # Named ``draft_kv_mgr`` so its state keys (draft_kv_mgr.past_key_values.*)
        # are DISTINCT from the full stack's layer-0 GQA buffer
        # (kv_mgr.past_key_values.0/1); otherwise the shared StateInitializer
        # binds them to the same physical HBM and the draft graph clobbers the
        # live GQA cache the verify graph reads.
        self.draft_kv_mgr = KVCacheManager(
            _config_for_mtp_draft(config),
            num_kv_head=self.num_key_value_heads,
            global_rank=self.rank_util,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden=None,
        adapter_ids=None,
        accepted_indices=None,
        current_length=None,
        medusa_mask=None,
        scatter_index=None,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
        tile_q_indices=None,
        tile_block_tables=None,
        tile_masks=None,
        inputs_embeds=None,
        kv_cache=None,
        active_mask=None,
        rotary_position_id=None,
        vision_embeddings=None,
        vision_mask=None,
    ):
        # Only three slots carry real data on this graph: input_ids holds the
        # committed next-token ids [B,1], prev_hidden (idx 5) holds the trunk
        # hidden [B,1,H], and position_ids/seq_ids are as on the main model.
        # The rest are empty placeholders (kept only to match the main model's
        # 24-arg arity for the shared executor).
        prev_hidden = self.set_none_if_empty(prev_hidden)
        next_token_ids = input_ids
        seq_ids = seq_ids.to(torch.int32)
        position_ids = position_ids.view(-1, next_token_ids.shape[1]).long()

        next_embeds = self.embed_tokens(next_token_ids)

        # One-layer KV for the draft full-attention layer.
        past_key_values = self.draft_kv_mgr.get_cache(
            seq_ids=seq_ids,
            seq_len=self.n_positions,
            is_for_context_encoding=False,
        )
        past_key_value = past_key_values[0] if past_key_values is not None else None

        cos_cache, sin_cache = self.mrope_emb(next_embeds, position_ids)

        # Decode-style causal mask over the PRIOR cache positions only.
        # compute_for_token_gen decomposes attention into a prior term (the
        # cached KV, gated by this mask) and an active term (the current token,
        # always attended). The current token's KV is written to slot
        # `position_ids` only AFTER this forward (update_cache below), so at
        # attention time slots >= the current position are unwritten garbage and
        # must be masked out. Per the NxD contract, mask[..., i] is True iff
        # i < computed_context_len (== current position).
        cache_positions = torch.arange(
            self.n_positions, device=next_token_ids.device
        ).view(1, -1)
        attention_mask = cache_positions < position_ids.view(-1, 1)

        draft_logits, present_key_value = self.mtp_head.draft_step(
            prev_hidden,
            next_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
        )
        draft_logits = draft_logits.float()

        updated_kv = self.draft_kv_mgr.update_cache(
            is_for_context_encoding=False,
            seq_ids=seq_ids,
            position_ids=position_ids,
            new_key_values=[present_key_value],
            seq_len=self.n_positions,
        )

        return [draft_logits, *updated_kv]


# ============================================================
# Verify backbone graph (n_active=2)
# ============================================================


class NeuronVerifyModel(NeuronQwen36A3BModel):
    """Traceable full 40-layer backbone over the 2-token verify block.

    Runs ``[x_{t+1}, x_{t+2}^draft]`` (n_active=2) seeded from the COMMITTED
    decode state and emits per-position logits ``[B, 2, vocab]``. Reuses the
    main model's ``init_model`` (embeddings, layers, norm, lm_head, mrope_emb),
    so the shared sharded checkpoint loads verbatim.

    State handling:
      * DeltaNet (linear_attention) layers run ``verify_mode=True``: each calls
        ``NeuronGatedDeltaNet.verify_block`` which seeds READ-ONLY from the live
        ``recurrent_state_buffer``/``conv_state_buffer`` (the same buffers the
        TKG graph commits to). verify_block never writes them, so a verify pass
        cannot corrupt the live recurrent/conv state. The per-position S/conv
        candidates it returns are collected but DISCARDED in Stage B.1 (the
        candidate aliasing is Stage B.2).
      * full_attention layers use the existing q_len>1 token-generation path
        against this model's OWN private ``KVCacheManager`` (built below), so a
        rejected draft's GQA K/V never touches the live TKG cache. In Stage B.1
        the private cache starts zeroed (no committed GQA prior); the 2-token
        block attends to itself via the intra-block causal active mask.

    The 24-arg forward signature matches NeuronQwen36A3BModel.forward verbatim so
    the shared NxDModelExecutor sees a uniform input arity across CTE/TKG/MTP/
    Verify. Only idx0 input_ids[B,2], idx2 position_ids[B,2], idx3 seq_ids, and
    idx5 prev_hidden (router disambiguator, ignored after set_none_if_empty) are
    read.

    VERIFY-GRAPH OUTPUT LAYOUT (index -> meaning), consumed by the Stage C host
    loop. Let L = #linear_attention (DeltaNet) layers (=30 at the production
    config), G = #full_attention layers (=10), num_kv = 2 * num_hidden_layers
    (=80; K and V per layer, including dummy zeros for DeltaNet slots),
    num_seed = 2*L + (1 if trunk_hidden_buffer else 0) (=60 standalone; +1 under
    A3B_ENABLE_MTP because the inherited trunk_hidden_buffer joins
    _deltanet_state_params). All shapes at TP4/B1; alloc_bs=1, seq_len=2.

      idx 0                       : logits          [B, 2, vocab]   (float)
      idx 1 .. num_kv             : *updated_kv     K,V per layer [B, kv, 2, hd]
                                    (private GQA cache; DeltaNet slots dummy 0)
      next num_seed entries       : SEED passthrough (recurrent_state_buffer /
                                    conv_state_buffer per DeltaNet layer, then the
                                    inherited trunk_hidden_buffer if present) --
                                    returned UNCHANGED (read-only seeding).
      next (2*L + 1) entries      : CANDIDATES (Stage B.2), in this order:
          for each DeltaNet layer (layer order):
            recurrent_cand  [B, 2, num_v_heads, head_k_dim, head_v_dim]
                            (=[1,2,8,128,128]); [:,0]=S1 (state after pos 0),
                            [:,1]=S2 (state after pos 1)
            conv_cand       [B, 2, conv_dim, K-1] (=[1,2,2560,3]); per-position
                            causal-conv windows
          then 1x verify_trunk  [B, 2, H] (=[1,2,2048]); pre-final-norm trunk
                            hidden, BOTH positions (no last-position gather):
                            [:,0]=h_{t+1}, [:,1]=h_{t+2}

    So the candidate block begins at index 1 + num_kv + num_seed and has
    2*L recurrent/conv entries (interleaved per layer) followed by 1 trunk entry.
    """

    def init_inference_optimization(self, config: Qwen36A3BInferenceConfig):
        # Private full-stack KV cache manager (default, non-hybrid). The DeltaNet
        # layers write dummy zeros into their slots in verify mode; full_attention
        # layers write real GQA K/V. Isolated from the main graph's kv_mgr so a
        # rejected draft never corrupts the live TKG cache.
        #
        # The candidate scratch buffers (recurrent_cand_buffers / conv_cand_buffers
        # / verify_trunk_buffer) are created in NeuronQwen36A3BModel.__init__ (the
        # shared base), gated by enable_verify_backbone, so they are ALSO present
        # on the CTE/TKG graphs. NxD's StateInitializer only zero-inits state keys
        # found in the FIRST graph's metaneff, so a verify-only state key would
        # fail with "Missing state tensor"; registering them on the main model
        # (emitted there as zero passthrough) puts their keys in the CTE metaneff.
        self.kv_mgr = KVCacheManager(
            config,
            num_kv_head=self.num_key_value_heads,
            global_rank=self.rank_util,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden=None,
        adapter_ids=None,
        accepted_indices=None,
        current_length=None,
        medusa_mask=None,
        scatter_index=None,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
        tile_q_indices=None,
        tile_block_tables=None,
        tile_masks=None,
        inputs_embeds=None,
        kv_cache=None,
        active_mask=None,
        rotary_position_id=None,
        vision_embeddings=None,
        vision_mask=None,
    ):
        # Router disambiguator only; never read for compute.
        prev_hidden = self.set_none_if_empty(prev_hidden)

        seq_ids = seq_ids.to(torch.int32)
        batch_size, seq_len = input_ids.shape[:2]  # seq_len == 2
        position_ids = position_ids.view(-1, seq_len).long()

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # mRoPE cos/sin for the full_attention layers (interleaved form, matching
        # the main stack which always uses MRoPE even for text-only).
        cos_cache, sin_cache = self.mrope_emb(inputs_embeds, position_ids)

        # Private per-layer KV cache (decode-style, full stack).
        past_key_values = self.kv_mgr.get_cache(
            seq_ids=seq_ids,
            seq_len=self.n_positions,
            is_for_context_encoding=False,
        )

        # Prior mask over cached positions for compute_for_token_gen: True iff a
        # cache slot was written before this token's position. In Stage B.1 the
        # private cache is zeroed so there is no real prior, but the mask is per
        # the NxD contract (mask[..., i] True iff i < computed_context_len).
        cache_positions = torch.arange(
            self.n_positions, device=input_ids.device
        ).view(1, 1, 1, -1)
        prior_mask = cache_positions < position_ids.view(batch_size, 1, seq_len, 1)

        # Intra-block causal active mask [B, 1, 2, 2] over the 2 active tokens:
        # token i attends to active tokens j <= i.
        block_idx = torch.arange(seq_len, device=input_ids.device)
        verify_active_mask = (
            (block_idx.view(1, 1, seq_len, 1) >= block_idx.view(1, 1, 1, seq_len))
            .expand(batch_size, 1, seq_len, seq_len)
        )

        next_decoder_cache = ()
        # Per-DeltaNet-layer candidates collected in layer order (one entry per
        # linear_attention layer): (S_stack [B,2,num_v_heads,Kd,Vd], conv_cand
        # [B,2,conv_dim,K-1]). Routed into the dedicated scratch buffers below.
        layer_candidates = []
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            layer_outputs = decoder_layer(
                hidden_states,
                seq_ids=seq_ids,
                attention_mask=prior_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=verify_active_mask,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                verify_mode=True,
            )
            hidden_states = layer_outputs[0]
            next_decoder_cache += (layer_outputs[1],)
            cos_cache, sin_cache = layer_outputs[2:4]
            # layer_outputs[5] is the deltanet-states slot: (S_stack, conv_cand)
            # for linear_attention layers, None for full_attention layers.
            if layer_outputs[5] is not None:
                layer_candidates.append(layer_outputs[5])

        # Update the private KV cache (full_attention layers' GQA K/V; DeltaNet
        # slots get dummy zeros). Isolated from the live TKG cache.
        updated_kv = self.kv_mgr.update_cache(
            is_for_context_encoding=False,
            seq_ids=seq_ids,
            position_ids=position_ids,
            new_key_values=next_decoder_cache,
            seq_len=self.n_positions,
        )

        # Pre-final-norm trunk hidden, BOTH block positions (the MTP draft input;
        # the draft head applies its own pre_fc norm). No last-position gather --
        # the host needs h_{t+1} (pos 0) and h_{t+2} (pos 1) per the verify block
        # semantics. Captured before self.norm to mirror the main stack's
        # _mtp_prenorm_hidden.
        verify_trunk_hidden = hidden_states

        hidden_states = self.norm(hidden_states)

        # Per-position logits: run lm_head on the FULL [B, 2, H] hidden (no
        # last-position gather) -> [B, 2, vocab].
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if hasattr(self.lm_head, "pad_size"):
            if self.lm_head.gather_output:
                rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
                world_size = 1
            else:
                rank_id = self.rank_util.get_rank()
                world_size = torch.distributed.get_world_size(
                    group=self.lm_head.tensor_parallel_group
                )
            from neuronx_distributed_inference.models.model_base import (
                mask_padded_logits,
            )

            logits = mask_padded_logits(
                logits, rank_id, world_size, pad_size=self.lm_head.pad_size
            )

        # Surface the DeltaNet SEED buffers (recurrent + conv) as trailing
        # input_output-aliased PASSTHROUGH outputs, returned UNCHANGED. These
        # buffers are not in the checkpoint -- aliasing them marks them as
        # zero-initialized device STATE (like the main stack's
        # _deltanet_state_params) so NxD does not try to load them as weights.
        # verify_block reads them read-only; emitting them unchanged keeps the
        # verify pass non-destructive (the returned values are the same zeros the
        # buffers already hold; the host discards them). These are the SEED
        # buffers only -- the S/conv candidates go to their own scratch buffers
        # below.
        passthrough_states = []
        for param in self._deltanet_state_params:
            passthrough_states.append(param + param * 0)

        # Surface the per-position S/conv CANDIDATES + per-position trunk hidden
        # (Stage B.2). Each candidate is routed into its OWN dedicated scratch
        # buffer and surfaced as an input_output-aliased output -- a candidate is
        # both returned to the host AND written in place to that scratch buffer.
        # ``cand + scratch * 0`` adds the alias dependency (matching value, marks
        # the buffer as the alias target) WITHOUT reading the buffer's contents
        # into the candidate. Risk #1: candidates alias ONLY these scratch
        # buffers, never the live/seed recurrent/conv buffers above. Cast to the
        # buffer dtype (S_stack/conv_cand are computed in float32 inside
        # verify_block) and pad to alloc_bs so the alias shapes match exactly.
        # The number/order of layer_candidates matches recurrent_cand_buffers /
        # conv_cand_buffers (both built by iterating self.layers' linear_attn).
        alloc_bs = self.verify_trunk_buffer.shape[0]

        def _alias_cand(value, scratch):
            value = value.to(scratch.dtype)
            cur_bs = value.shape[0]
            if cur_bs < alloc_bs:
                value = torch.cat([value, scratch[cur_bs:] * 0], dim=0)
            return value + scratch * 0

        candidate_outputs = []
        for (S_stack, conv_cand), rec_buf, conv_buf in zip(
            layer_candidates, self.recurrent_cand_buffers, self.conv_cand_buffers
        ):
            candidate_outputs.append(_alias_cand(S_stack, rec_buf))
            candidate_outputs.append(_alias_cand(conv_cand, conv_buf))
        candidate_outputs.append(
            _alias_cand(verify_trunk_hidden, self.verify_trunk_buffer)
        )

        return [logits, *updated_kv, *passthrough_states, *candidate_outputs]


# ============================================================
# State Dict Converter
# ============================================================


def reorder_deltanet_qkv_for_tp(
    qkv_weight: torch.Tensor,
    tp_degree: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> torch.Tensor:
    """Pack [Q_all | K_all | V_all] into per-rank Q/K/V blocks.

    ColumnParallelLinear slices the first dimension contiguously. DeltaNet
    needs each rank to receive its local query, key, and value heads
    together, so the full HF tensor is repacked as:
    [rank0 Q | rank0 K | rank0 V | rank1 Q | rank1 K | rank1 V | ...].
    """
    if num_k_heads % tp_degree != 0:
        raise ValueError(
            f"linear_num_key_heads={num_k_heads} must be divisible by tp_degree={tp_degree}"
        )
    if num_v_heads % tp_degree != 0:
        raise ValueError(
            f"linear_num_value_heads={num_v_heads} must be divisible by tp_degree={tp_degree}"
        )

    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    q_weight = qkv_weight[:key_dim].reshape(num_k_heads, head_k_dim, -1)
    k_weight = qkv_weight[key_dim : 2 * key_dim].reshape(num_k_heads, head_k_dim, -1)
    v_weight = qkv_weight[2 * key_dim : 2 * key_dim + value_dim].reshape(
        num_v_heads, head_v_dim, -1
    )
    local_k_heads = num_k_heads // tp_degree
    local_v_heads = num_v_heads // tp_degree
    blocks = []
    for rank in range(tp_degree):
        blocks.append(
            q_weight[rank * local_k_heads : (rank + 1) * local_k_heads].reshape(-1, qkv_weight.shape[1])
        )
        blocks.append(
            k_weight[rank * local_k_heads : (rank + 1) * local_k_heads].reshape(-1, qkv_weight.shape[1])
        )
        blocks.append(
            v_weight[rank * local_v_heads : (rank + 1) * local_v_heads].reshape(-1, qkv_weight.shape[1])
        )
    return torch.cat(blocks, dim=0).contiguous()


def reorder_deltanet_qkv_channels_for_tp(
    channel_tensor: torch.Tensor,
    tp_degree: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> torch.Tensor:
    """Per-rank repack of a first-dimension Q/K/V channel tensor (e.g. conv1d weight)."""
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    q_tensor = channel_tensor[:key_dim]
    k_tensor = channel_tensor[key_dim : 2 * key_dim]
    v_tensor = channel_tensor[2 * key_dim : 2 * key_dim + value_dim]
    local_key_dim = key_dim // tp_degree
    local_value_dim = value_dim // tp_degree
    blocks = []
    for rank in range(tp_degree):
        blocks.append(q_tensor[rank * local_key_dim : (rank + 1) * local_key_dim])
        blocks.append(k_tensor[rank * local_key_dim : (rank + 1) * local_key_dim])
        blocks.append(v_tensor[rank * local_value_dim : (rank + 1) * local_value_dim])
    return torch.cat(blocks, dim=0).contiguous()


def _convert_full_attention_block(sd, prefix, config):
    """GQA per-layer conversions on `{prefix}.self_attn.*`, in place.

    Splits the gated q_proj into query + sigmoid output gate, renames the q/k
    norms, and seeds rank_util. Shared by the main layers and the MTP head's
    decoder layer.
    """
    sd[f"{prefix}.self_attn.rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    for src, dst in (("q_norm", "q_layernorm"), ("k_norm", "k_layernorm")):
        key = f"{prefix}.self_attn.{src}.weight"
        if key in sd:
            sd[f"{prefix}.self_attn.{dst}.weight"] = sd.pop(key).detach().clone()

    # q_proj is doubled and interleaved per head: [head_i query | head_i gate].
    q_proj_key = f"{prefix}.self_attn.q_proj.weight"
    if q_proj_key in sd:
        num_heads, head_dim = config.num_attention_heads, config.head_dim
        w = sd.pop(q_proj_key).reshape(num_heads, head_dim * 2, config.hidden_size)
        sd[q_proj_key] = w[:, :head_dim, :].reshape(num_heads * head_dim, config.hidden_size)
        sd[f"{prefix}.self_attn.output_gate_proj.weight"] = (
            w[:, head_dim:, :].reshape(num_heads * head_dim, config.hidden_size)
        )

    if config.neuron_config.fused_qkv:
        q, k, v = (f"{prefix}.self_attn.{p}_proj.weight" for p in ("q", "k", "v"))
        if q in sd:
            sd[f"{prefix}.self_attn.Wqkv.weight"] = torch.cat([sd[q], sd[k], sd[v]])
            for key in (q, k, v):
                del sd[key]


def _convert_moe_block(sd, prefix, config):
    """MoE FFN conversions on `{prefix}.mlp.*`, in place.

    Renames the router and transposes the fused expert tensors into the layout
    ExpertMLPsV2 expects: gate_up_proj (E, H, 2I) and down_proj (E, I, H). The
    sigmoid-gated shared expert passes through under its HF names.
    """
    gate_key = f"{prefix}.mlp.gate.weight"
    if gate_key in sd:
        sd[f"{prefix}.mlp.moe.router.linear_router.weight"] = sd.pop(gate_key).detach().clone()

    for hf_suffix, nxdi_suffix in (
        ("experts.gate_up_proj", "moe.expert_mlps.mlp_op.gate_up_proj.weight"),
        ("experts.down_proj", "moe.expert_mlps.mlp_op.down_proj.weight"),
    ):
        src = f"{prefix}.mlp.{hf_suffix}"
        if src in sd:
            sd[f"{prefix}.mlp.{nxdi_suffix}"] = sd.pop(src).transpose(1, 2).contiguous()


def convert_qwen36_a3b_hf_to_neuron_state_dict(neuron_state_dict, config):
    """Convert HF Qwen3.6-A3B weights to NxDI keys / shapes.

    Per-layer mappings:

    DeltaNet (linear_attention):
        in_proj_qkv is repacked across TP ranks; conv1d, A_log, dt_bias
        are folded into ColumnParallelLinear parameter containers
        (conv1d_weight.weight, A_log_weight.weight, dt_bias_weight.weight).

    GQA (full_attention):
        q_proj is doubled (Q + output gate, interleaved per head): split
        into q_proj + output_gate_proj. q_norm/k_norm renamed to
        q_layernorm/k_layernorm. Optionally fused into Wqkv.

    MoE FFN (all layers):
        mlp.gate -> mlp.moe.router.linear_router
        mlp.experts.{0..N-1}.{gate,up,down}_proj are stacked + transposed
        into mlp.moe.expert_mlps.mlp_op.{gate_up_proj,down_proj}.
        mlp.shared_expert.{gate,up,down}_proj : pass-through (owned by
        NeuronMoEBlock, applied with a sigmoid gate over the output).
        mlp.shared_expert_gate.weight : pass-through.

    MTP (single layer, tied embeddings/LM head):
        mtp.fc                                    -> mtp_head.eh_proj
        mtp.pre_fc_norm_embedding                 -> mtp_head.embed_norm
        mtp.pre_fc_norm_hidden                    -> mtp_head.hidden_norm
        mtp.norm                                  -> mtp_head.final_norm
        mtp.layers.0.input_layernorm              -> mtp_head.decoder_layer.input_layernorm
        mtp.layers.0.self_attn.{q,k,v,o}_proj     -> mtp_head.decoder_layer.self_attn.{q,k,v,o}_proj
        mtp.layers.0.self_attn.{q,k}_norm         -> mtp_head.decoder_layer.self_attn.{q,k}_norm
        mtp.layers.0.post_attention_layernorm     -> mtp_head.decoder_layer.post_attention_layernorm
        mtp.layers.0.mlp.<rest>                   -> mtp_head.decoder_layer.mlp.<rest>
        lm_head.weight (main)                     -> mtp_head.mtp_lm_head.weight (tied copy, no mtp.lm_head in ckpt)
        (then _convert_full_attention_block + _convert_moe_block run on mtp_head.decoder_layer)

    All Qwen RMSNorm weights are converted from `(1 + w)` to standard form.
    """
    # Add rank_util
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0,
        config.neuron_config.tp_degree,
        dtype=torch.int32,
    )

    # HF norms use `output = norm(x) * (1 + weight)` with weight ~ 0; NxDI
    # RMSNorm uses `output = norm(x) * weight` with weight ~ 1. Add 1 to convert.
    norm_keys_to_convert = []
    for l in range(config.num_hidden_layers):
        norm_keys_to_convert.append(f"layers.{l}.input_layernorm.weight")
        norm_keys_to_convert.append(f"layers.{l}.post_attention_layernorm.weight")
        if config.layer_types[l] == "full_attention":
            norm_keys_to_convert.append(f"layers.{l}.self_attn.q_norm.weight")
            norm_keys_to_convert.append(f"layers.{l}.self_attn.k_norm.weight")
    norm_keys_to_convert.append("norm.weight")

    for nk in norm_keys_to_convert:
        if nk in neuron_state_dict:
            old_val = neuron_state_dict[nk]
            neuron_state_dict[nk] = old_val.float() + 1.0
            if "layers.0." in nk or nk == "norm.weight":
                logger.debug(
                    f"[NORM FIX] {nk}: mean {old_val.float().mean():.4f} -> {neuron_state_dict[nk].mean():.4f}"
                )
        else:
            if "layers.0." in nk or nk == "norm.weight":
                logger.warning(f"[NORM FIX] key not found: {nk}")

    for l in range(config.num_hidden_layers):
        layer_type = config.layer_types[l]

        # === DeltaNet layers ===
        if layer_type == "linear_attention":
            tp = config.neuron_config.tp_degree
            reorder_args = (
                tp,
                config.linear_num_key_heads,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            )
            qkv_key = f"layers.{l}.linear_attn.in_proj_qkv.weight"
            if qkv_key in neuron_state_dict and tp > 1:
                neuron_state_dict[qkv_key] = reorder_deltanet_qkv_for_tp(
                    neuron_state_dict[qkv_key], *reorder_args,
                )

            conv_key = f"layers.{l}.linear_attn.conv1d.weight"
            conv_weight_key = f"layers.{l}.linear_attn.conv1d_weight.weight"
            if conv_key in neuron_state_dict:
                conv_weight = neuron_state_dict.pop(conv_key)
                if tp > 1:
                    conv_weight = reorder_deltanet_qkv_channels_for_tp(
                        conv_weight, *reorder_args,
                    )
                neuron_state_dict[conv_weight_key] = conv_weight.squeeze(1).contiguous()

            for vector_name in ("A_log", "dt_bias"):
                vector_key = f"layers.{l}.linear_attn.{vector_name}"
                vector_weight_key = f"layers.{l}.linear_attn.{vector_name}_weight.weight"
                if vector_key in neuron_state_dict:
                    neuron_state_dict[vector_weight_key] = (
                        neuron_state_dict.pop(vector_key).reshape(-1, 1).contiguous()
                    )

        # === Attention layers ===  (GQA; DeltaNet handled above)
        if layer_type == "full_attention":
            _convert_full_attention_block(neuron_state_dict, f"layers.{l}", config)

        # MoE FFN (every layer). The sigmoid-gated shared expert passes through
        # under its HF names (owned by NeuronMoEBlock).
        _convert_moe_block(neuron_state_dict, f"layers.{l}", config)

        gc.collect()

    # MTP head (Qwen3-Next layout). The checkpoint ships `mtp.fc`,
    # `mtp.pre_fc_norm_{embedding,hidden}`, `mtp.norm`, and one full-attention
    # decoder layer under `mtp.layers.0.*`; it ships neither a draft embedding
    # nor a draft LM head, so both are tied to the main model. When MTP is
    # disabled, drop the keys so they don't trip load_state_dict.
    if config.mtp_num_hidden_layers == 0:
        for k in [k for k in neuron_state_dict if k.startswith("mtp.")]:
            neuron_state_dict.pop(k)
    else:
        top_rename = {
            "mtp.fc.weight": "mtp_head.eh_proj.weight",
            "mtp.pre_fc_norm_embedding.weight": "mtp_head.embed_norm.weight",
            "mtp.pre_fc_norm_hidden.weight": "mtp_head.hidden_norm.weight",
            "mtp.norm.weight": "mtp_head.final_norm.weight",
        }
        for old, new in top_rename.items():
            if old in neuron_state_dict:
                neuron_state_dict[new] = neuron_state_dict.pop(old).detach().clone()

        layer_pfx = "mtp.layers.0."
        for k in [k for k in neuron_state_dict if k.startswith(layer_pfx)]:
            neuron_state_dict[f"mtp_head.decoder_layer.{k[len(layer_pfx):]}"] = (
                neuron_state_dict.pop(k).detach().clone()
            )
        for k in [k for k in neuron_state_dict if k.startswith("mtp.")]:
            neuron_state_dict.pop(k)  # drop any stray mtp.* (none expected)

        # (1 + w) for every MTP RMSNorm (head norms + the draft layer's norms).
        for nk in (
            "mtp_head.embed_norm.weight",
            "mtp_head.hidden_norm.weight",
            "mtp_head.final_norm.weight",
            "mtp_head.decoder_layer.input_layernorm.weight",
            "mtp_head.decoder_layer.post_attention_layernorm.weight",
            "mtp_head.decoder_layer.self_attn.q_norm.weight",
            "mtp_head.decoder_layer.self_attn.k_norm.weight",
        ):
            if nk in neuron_state_dict:
                neuron_state_dict[nk] = neuron_state_dict[nk].float() + 1.0

        # The draft layer is a standard full-attention + MoE layer.
        _convert_full_attention_block(neuron_state_dict, "mtp_head.decoder_layer", config)
        _convert_moe_block(neuron_state_dict, "mtp_head.decoder_layer", config)

        # Tie the draft LM head to the main model's (no mtp.lm_head in checkpoint).
        neuron_state_dict["mtp_head.mtp_lm_head.weight"] = (
            neuron_state_dict["lm_head.weight"].detach().clone()
        )

    return neuron_state_dict


# ============================================================
# Custom ModelWrapper and DecoderModelInstance for DeltaNet state aliasing
# ============================================================


class Qwen36A3BDecoderModelInstance(DecoderModelInstance):
    """Custom DecoderModelInstance that adds DeltaNet state buffers to input_output_aliases."""

    def get(self, bucket_rank, **kwargs):
        """Override to add DeltaNet state aliases after KV cache aliases."""
        module, input_output_aliases = super().get(bucket_rank, **kwargs)

        num_output_from_trace = 1 if not self.neuron_config.output_logits else 2

        if module.kv_mgr is not None:
            num_kv = len(module.kv_mgr.past_key_values)
        else:
            num_kv = 0

        state_start_idx = num_output_from_trace + num_kv

        if (
            not getattr(module.config, "use_hybrid_cache_manager", False)
            and hasattr(module, "_deltanet_state_params")
        ):
            seed_params = list(module._deltanet_state_params)
            for i, param in enumerate(seed_params):
                input_output_aliases[param] = state_start_idx + i

            # Verify candidate scratch buffers, emitted by the main forward as
            # trailing zero passthrough when the verify graph is enabled. Aliased
            # AFTER the seed states so their state keys are registered on CTE/TKG
            # too (NxD's StateInitializer only zero-inits keys from the first
            # graph's metaneff). Empty / no-op when verify is disabled.
            cand_start_idx = state_start_idx + len(seed_params)
            for i, param in enumerate(module._verify_candidate_params):
                input_output_aliases[param] = cand_start_idx + i

        return module, input_output_aliases


class Qwen36A3BModelWrapper(ModelWrapper):
    """Custom ModelWrapper for VL support with mRoPE and vision inputs."""

    def get_model_instance(self):
        return Qwen36A3BDecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )

    def input_generator(self):
        """Generate inputs including mrope_position_ids, vision_embeddings, and vision_mask."""
        base_inputs = super().input_generator()
        extended_inputs = []

        for bucket_inputs in base_inputs:
            input_ids = bucket_inputs[0]
            batch_size = input_ids.shape[0]
            n_active_tokens = input_ids.shape[1]

            is_cte = n_active_tokens > 1

            if is_cte:
                mrope_position_ids = (
                    torch.arange(0, n_active_tokens, dtype=torch.int32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(3, batch_size, -1)
                    .contiguous()
                )

                vision_embeddings = torch.zeros(
                    (batch_size, n_active_tokens, self.config.hidden_size),
                    dtype=self.config.neuron_config.torch_dtype,
                )
                vision_mask = torch.full(
                    (batch_size, n_active_tokens, 1),
                    fill_value=n_active_tokens - 1,
                    dtype=torch.int32,
                )
            else:
                mrope_position_ids = torch.zeros((0,), dtype=torch.int32)
                vision_embeddings = torch.zeros(
                    (0,), dtype=self.config.neuron_config.torch_dtype
                )
                vision_mask = torch.zeros((0,), dtype=torch.int32)

            padded = list(bucket_inputs)
            while len(padded) < 21:
                padded.append(torch.zeros((0,), dtype=torch.int32))
            padded.append(mrope_position_ids)  # position 21
            padded.append(vision_embeddings)  # position 22
            padded.append(vision_mask)  # position 23

            extended_inputs.append(tuple(padded))

        return extended_inputs

    def pad_inputs(self, *args, pad_type="first_fit"):
        """Override to pad mrope_position_ids and vision inputs to bucket size."""
        orig_mrope = args[21] if len(args) >= 22 else None
        orig_vis_emb = args[22] if len(args) >= 23 else None
        orig_vis_mask = args[23] if len(args) >= 24 else None

        padded_args = super().pad_inputs(*args, pad_type=pad_type)

        if len(padded_args) >= 24 and orig_mrope is not None:
            padded_seq_len = padded_args[0].shape[1]
            batch_size = padded_args[0].shape[0]
            is_cte = padded_seq_len > 1

            if is_cte:
                current_mrope = orig_mrope
                current_vis_emb = orig_vis_emb
                current_vis_mask = orig_vis_mask

                if (
                    current_mrope.ndim == 3
                    and current_mrope.shape[-1] != padded_seq_len
                ):
                    orig_len = current_mrope.shape[-1]
                    pad_size = padded_seq_len - orig_len
                    last_pos = current_mrope[:, :, -1:]
                    pad_offsets = torch.arange(
                        1, pad_size + 1, dtype=current_mrope.dtype
                    )
                    pad_offsets = (
                        pad_offsets.unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
                    )
                    mrope_pad = last_pos + pad_offsets
                    mrope_position_ids = torch.cat([current_mrope, mrope_pad], dim=-1)
                elif current_mrope.ndim == 3:
                    mrope_position_ids = current_mrope
                else:
                    mrope_position_ids = (
                        torch.arange(0, padded_seq_len, dtype=torch.int32)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand(3, batch_size, -1)
                        .contiguous()
                    )

                if (
                    current_vis_emb is not None
                    and current_vis_emb.ndim == 3
                    and current_vis_emb.shape[1] < padded_seq_len
                ):
                    pad_emb = torch.zeros(
                        (
                            batch_size,
                            padded_seq_len - current_vis_emb.shape[1],
                            current_vis_emb.shape[2],
                        ),
                        dtype=current_vis_emb.dtype,
                    )
                    vision_embeddings = torch.cat([current_vis_emb, pad_emb], dim=1)
                elif current_vis_emb is not None and current_vis_emb.ndim == 3:
                    vision_embeddings = current_vis_emb[:, :padded_seq_len]
                else:
                    vision_embeddings = torch.zeros(
                        (batch_size, padded_seq_len, self.config.hidden_size),
                        dtype=self.config.neuron_config.torch_dtype,
                    )

                if (
                    current_vis_mask is not None
                    and current_vis_mask.ndim == 3
                    and current_vis_mask.shape[1] < padded_seq_len
                ):
                    pad_mask = torch.full(
                        (batch_size, padded_seq_len - current_vis_mask.shape[1], 1),
                        fill_value=padded_seq_len - 1,
                        dtype=torch.int32,
                    )
                    vision_mask = torch.cat([current_vis_mask, pad_mask], dim=1)
                elif current_vis_mask is not None and current_vis_mask.ndim == 3:
                    vision_mask = current_vis_mask[:, :padded_seq_len]
                else:
                    vision_mask = torch.full(
                        (batch_size, padded_seq_len, 1),
                        fill_value=padded_seq_len - 1,
                        dtype=torch.int32,
                    )

                padded_args = (
                    *padded_args[:21],
                    mrope_position_ids,
                    vision_embeddings,
                    vision_mask,
                )

                padded_args = list(padded_args)
                padded_args[23] = padded_args[23].clamp(max=padded_seq_len - 1)
                padded_args = tuple(padded_args)

        return padded_args


# ============================================================
# MTP draft graph wrapper / instance
# ============================================================


class MTPDraftModelInstance(DecoderModelInstance):
    """DecoderModelInstance for the MTP draft graph.

    The draft forward returns ``[draft_logits, draft_k, draft_v]`` -- a single
    trace output followed by this graph's own one-layer KV. Alias that KV back
    into ``module.draft_kv_mgr.past_key_values`` so the runtime updates it in
    place, independently of the main graph's caches.
    """

    def get(self, bucket_rank, **kwargs):
        if bucket_rank is not None:
            self.module.n_positions = self.neuron_config.buckets[bucket_rank]

        self.input_output_aliases = {}
        # The draft forward emits exactly one non-KV output (the draft logits).
        num_output_from_trace = 1

        past_key_values = self.module.draft_kv_mgr.past_key_values
        for i in range(len(past_key_values)):
            self.input_output_aliases[past_key_values[i]] = num_output_from_trace + i

        return self.module, self.input_output_aliases


class MTPHeadModelWrapper(ModelWrapper):
    """ModelWrapper that traces NeuronMTPDraftModel as a standalone graph.

    The input signature -- prev_hidden [B,1,H] bf16, next_token_ids [B,1] int32,
    position_ids [B,1] int32, seq_ids [B] int32 -- is deliberately distinct from
    the token-generation graph (whose first input is int32 input_ids [B,1]) so
    the runtime can disambiguate by shape; the host helper also passes
    model_name=MTP_HEAD_MODEL_TAG explicitly.
    """

    def get_model_instance(self):
        return MTPDraftModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )

    def input_generator(self):
        """Produce the SAME 24-arg positional layout as the main model wrapper.

        The shared NxDModelExecutor requires a uniform input arity across all
        compiled graphs, so this draft graph must take the identical 24 args as
        NeuronQwen36A3BModel.forward (input_ids, attention_mask, position_ids,
        seq_ids, sampling_params, prev_hidden, adapter_ids, 14 empty
        placeholders, rotary_position_id, vision_embeddings, vision_mask).

        NxD's runtime router selects a graph by matching the FULL example-input
        shape signature (built here at trace time) against the runtime call's
        shapes, so input_generator() and draft() MUST be shape-identical in ALL
        24 slots. Since NeuronMTPDraftModel.forward reads ONLY idx0 (input_ids),
        idx2 (position_ids), idx3 (seq_ids) and idx5 (prev_hidden) -- it builds
        its own attention_mask internally and never touches sampling_params --
        every other slot (idx1, idx4, idx6..23) is an EMPTY placeholder here, and
        draft() emits the byte-identical layout. We therefore build the 24-tuple
        explicitly rather than reuse the base ModelWrapper 7-tuple (whose idx1
        attention_mask = ones([B,bucket]) and idx4 sampling_params = ones([B,S])
        would NOT match draft()'s empties and would misroute at runtime).

        Slot layout (index -> contents):
            0  input_ids        ones([B,1] int32)  -- dummy committed token
            1  attention_mask   empty (built on-device in forward)
            2  position_ids     ones([B,1] int32)
            3  seq_ids          arange(B) int32
            4  sampling_params  empty (never read; on_device_sampling False)
            5  prev_hidden      zeros([B,1,H] torch_dtype)  -- the DISTINGUISHING
                                input (non-empty here vs empty [0] on TKG)
            6..20  empty int32 placeholders (adapter_ids + 14 unused slots)
            21 rotary_position_id  empty int32
            22 vision_embeddings   empty torch_dtype
            23 vision_mask         empty int32
        """
        inputs = []
        for bucket in self.neuron_config.buckets:
            batch_size = self.neuron_config.batch_size

            torch_dtype = self.config.neuron_config.torch_dtype

            # Fresh empty object PER slot. extract() dedups tensors by identity;
            # XLA tracing gives every slot a distinct input (HLO-gen flattener
            # layout = 24 uniques), so reusing one empty object would dedup the
            # placeholders and break the flattener's `assert self.layout ==
            # layout` at build_flattener_map. CTE/TKG do the same (fresh per slot).
            def _empty_i32():
                return torch.zeros((0,), dtype=torch.int32)

            def _empty_f():
                return torch.zeros((0,), dtype=torch_dtype)

            input_ids = torch.ones((batch_size, 1), dtype=torch.int32)
            position_ids = torch.ones((batch_size, 1), dtype=torch.int32)
            seq_ids = torch.arange(0, batch_size, dtype=torch.int32)
            prev_hidden = torch.zeros(
                (batch_size, 1, self.config.hidden_size),
                dtype=torch_dtype,
            )

            args24 = [
                input_ids,      # 0  input_ids (dummy committed token)
                _empty_i32(),   # 1  attention_mask (built on-device)
                position_ids,   # 2  position_ids
                seq_ids,        # 3  seq_ids
                _empty_i32(),   # 4  sampling_params (never read by draft forward)
                prev_hidden,    # 5  prev_hidden (distinguishing input)
            ]
            # idx 6..20: empty int32 placeholders (adapter_ids + 14 unused slots).
            args24 += [_empty_i32() for _ in range(15)]
            args24.append(_empty_i32())  # 21 rotary_position_id
            args24.append(_empty_f())    # 22 vision_embeddings
            args24.append(_empty_i32())  # 23 vision_mask

            inputs.append(tuple(args24))
        return inputs

    def forward(self, *args, pad_type="first_fit"):
        """Run the draft graph directly.

        The draft graph takes the same 24 positional args as the main model but
        a fixed-shape decode-style bucket, so bypass the base wrapper's
        pad/route machinery and call the compiled module with only int64->int32
        normalization.
        """
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict()."
            )
        args = self.convert_int64_to_int32(*args)
        return self._forward(*args)


# ============================================================
# Verify backbone graph wrapper / instance
# ============================================================


class VerifyModelInstance(DecoderModelInstance):
    """DecoderModelInstance for the verify backbone graph.

    The verify forward returns ``[logits, *updated_kv]`` -- a single trace
    output (the per-position logits [B,2,vocab]) followed by this graph's own
    full-stack private KV cache. Alias that KV back into
    ``module.kv_mgr.past_key_values`` so the runtime updates it in place,
    isolated from the main graph's caches. No DeltaNet state buffer is aliased
    in Stage B.1 (verify_block is read-only; candidate aliasing is Stage B.2).
    """

    def get(self, bucket_rank, **kwargs):
        if bucket_rank is not None:
            self.module.n_positions = self.neuron_config.buckets[bucket_rank]

        self.input_output_aliases = {}
        # The verify forward emits exactly one non-KV output (the logits),
        # followed by this graph's private KV cache, then the DeltaNet state
        # buffers (recurrent + conv) as passthrough outputs.
        num_output_from_trace = 1

        past_key_values = self.module.kv_mgr.past_key_values
        for i in range(len(past_key_values)):
            self.input_output_aliases[past_key_values[i]] = num_output_from_trace + i

        # Alias the DeltaNet SEED buffers AFTER the KV. The verify forward emits
        # them unchanged (read-only seeding); aliasing them marks them as
        # zero-initialized device STATE so they are not required from the
        # checkpoint (they hold carry-over state, not trained weights). These are
        # this verify graph's OWN buffers (separate from the live TKG/CTE graph),
        # so aliasing them cannot corrupt live decode state. NO live state buffer
        # from another graph appears here.
        state_start_idx = num_output_from_trace + len(past_key_values)
        seed_params = list(self.module._deltanet_state_params)
        seed_ids = set(id(p) for p in seed_params)
        for i, param in enumerate(seed_params):
            self.input_output_aliases[param] = state_start_idx + i

        # Stage B.2: alias the S/conv CANDIDATE scratch buffers + per-position
        # trunk-hidden buffer AFTER the seed buffers. Output layout (per the
        # verify forward) is [logits, *KV, *seed_passthrough, *candidates], so the
        # candidate output indices start right after the seed passthrough block.
        cand_start_idx = state_start_idx + len(seed_params)
        cand_params = list(self.module._verify_candidate_params)
        # Risk #1: a candidate must alias ONLY its dedicated scratch buffer, never
        # a live/seed recurrent/conv buffer -- aliasing a live buffer would
        # silently commit a candidate and corrupt decode. Assert the candidate
        # buffers are disjoint from the seed buffers before wiring them.
        for param in cand_params:
            assert id(param) not in seed_ids, (
                "verify candidate buffer aliases a live/seed DeltaNet state "
                "buffer; candidates MUST use dedicated scratch buffers only "
                "(Risk #1)"
            )
        for i, param in enumerate(cand_params):
            self.input_output_aliases[param] = cand_start_idx + i

        return self.module, self.input_output_aliases


class VerifyModelWrapper(ModelWrapper):
    """ModelWrapper that traces NeuronVerifyModel as a standalone graph.

    The input signature -- input_ids [B,2] int32, position_ids [B,2] int32,
    seq_ids [B] int32, prev_hidden [B,2,H] torch_dtype at idx5 -- distinguishes
    the verify graph from CTE (n_active=bucket, prev_hidden empty), TKG (n_active
    =1, prev_hidden empty) and the MTP draft head (n_active=1, prev_hidden
    [B,1,H]). The non-empty prev_hidden [B,2,H] at idx5 is the router
    disambiguator; the host verify() helper also passes model_name explicitly.
    """

    def get_model_instance(self):
        return VerifyModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )

    def input_generator(self):
        """Produce the SAME 24-arg positional layout as the main model wrapper,
        at n_active=2.

        The shared NxDModelExecutor requires a uniform input arity across all
        compiled graphs, so this verify graph takes the identical 24 args as
        NeuronQwen36A3BModel.forward. NxD's runtime router selects a graph by
        matching the FULL example-input shape signature against the runtime
        call, so input_generator() and verify() MUST be byte-shape-identical in
        ALL 24 slots. NeuronVerifyModel.forward reads ONLY idx0 (input_ids[B,2]),
        idx2 (position_ids[B,2]), idx3 (seq_ids) and idx5 (prev_hidden[B,2,H],
        router disambiguator), so every other slot is an EMPTY placeholder, and
        verify() emits the byte-identical layout.

        Slot layout (index -> contents):
            0  input_ids        ones([B,2] int32)
            1  attention_mask   empty (built on-device in forward)
            2  position_ids     ones([B,2] int32)
            3  seq_ids          arange(B) int32
            4  sampling_params  empty (never read; on_device_sampling off)
            5  prev_hidden      zeros([B,2,H] torch_dtype)  -- DISTINGUISHING
            6..20  empty int32 placeholders
            21 rotary_position_id  empty int32
            22 vision_embeddings   empty torch_dtype
            23 vision_mask         empty int32
        """
        inputs = []
        for bucket in self.neuron_config.buckets:
            batch_size = self.neuron_config.batch_size
            torch_dtype = self.config.neuron_config.torch_dtype

            # Fresh empty object PER slot (extract() dedups by identity; reusing
            # one empty would break the flattener's layout assert).
            def _empty_i32():
                return torch.zeros((0,), dtype=torch.int32)

            def _empty_f():
                return torch.zeros((0,), dtype=torch_dtype)

            input_ids = torch.ones((batch_size, 2), dtype=torch.int32)
            position_ids = torch.ones((batch_size, 2), dtype=torch.int32)
            seq_ids = torch.arange(0, batch_size, dtype=torch.int32)
            prev_hidden = torch.zeros(
                (batch_size, 2, self.config.hidden_size),
                dtype=torch_dtype,
            )

            args24 = [
                input_ids,      # 0  input_ids [B,2]
                _empty_i32(),   # 1  attention_mask (built on-device)
                position_ids,   # 2  position_ids [B,2]
                seq_ids,        # 3  seq_ids
                _empty_i32(),   # 4  sampling_params (never read)
                prev_hidden,    # 5  prev_hidden [B,2,H] (router disambiguator)
            ]
            args24 += [_empty_i32() for _ in range(15)]  # 6..20
            args24.append(_empty_i32())  # 21 rotary_position_id
            args24.append(_empty_f())    # 22 vision_embeddings
            args24.append(_empty_i32())  # 23 vision_mask

            inputs.append(tuple(args24))
        return inputs

    def forward(self, *args, pad_type="first_fit"):
        """Run the verify graph directly.

        Fixed-shape decode-style bucket, so bypass the base wrapper's pad/route
        machinery and call the compiled module with only int64->int32 norm.
        """
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict()."
            )
        args = self.convert_int64_to_int32(*args)
        return self._forward(*args)


# ============================================================
# Top-Level Model
# ============================================================


class NeuronQwen36A3BForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronQwen36A3BModel

    def get_model_wrapper_cls(self):
        """Return custom ModelWrapper with DeltaNet state aliasing."""
        return Qwen36A3BModelWrapper

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load HF model weights.

        The model is a VL model (Qwen3_5ForConditionalGeneration) but we
        only need the text backbone.
        """
        from transformers import AutoModelForCausalLM

        kwargs.setdefault("trust_remote_code", True)
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return Qwen36A3BInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """Strip VL wrapper prefix and convert to NxDI format."""
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("language_model."):
                new_k = k.replace("language_model.", "", 1)
                new_sd[new_k] = v
            elif k.startswith("model.language_model."):
                new_k = k.replace("model.language_model.", "", 1)
                new_sd[new_k] = v
            elif k.startswith("model.visual") or k.startswith("visual"):
                continue  # Skip vision encoder (text-only port)
            elif k.startswith("model."):
                new_sd[k.replace("model.", "", 1)] = v
            else:
                # lm_head.*, mtp.*, and anything else passes through.
                new_sd[k] = v

        return convert_qwen36_a3b_hf_to_neuron_state_dict(new_sd, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()
        # Register the MTP draft head as a third compiled graph when present.
        # Done here (after CTE+TKG) so the draft graph is always last in
        # self.models and the main two graphs are byte-identical to the
        # MTP-disabled build.
        if getattr(self.config, "mtp_num_hidden_layers", 0) > 0:
            self.enable_mtp_head()
        # Register the verify backbone (n_active=2) as an additional compiled
        # graph when enabled. Independent of MTP for standalone bring-up
        # (A3B_ENABLE_VERIFY=1 without A3B_ENABLE_MTP => CTE+TKG+Verify). Done
        # after enable_mtp_head() and BEFORE the priority clear-loop so the
        # verify wrapper is included in the WLO disable.
        if getattr(self.config, "enable_verify_backbone", False):
            self.enable_verify_backbone()
        if (
            getattr(self.config, "mtp_num_hidden_layers", 0) > 0
            or getattr(self.config, "enable_verify_backbone", False)
        ):
            # NxD weight-layout optimization (WLO) propagates the priority graph's
            # optimal layout to every other HLO via the `hlo-opt` binary, which
            # crashes (std::invalid_argument: stoi) on the extra graph's HLO.
            # WLO runs only when some graph is marked priority, so clear the
            # priority flag on all graphs -> _should_optimize_layout() is False ->
            # NxD skips the pass before hlo-opt runs. WLO is perf-only; the build
            # stays correct.
            for wrapper in self.models:
                wrapper.priority_model_idx = None

    def enable_mtp_head(self):
        """Register NeuronMTPDraftModel as the third compiled graph.

        Mirrors enable_token_generation's decode-style config (batch_size =
        tkg_batch_size, n_active_tokens = 1) but traces NeuronMTPDraftModel via
        MTPHeadModelWrapper. The shared sharded checkpoint already carries the
        mtp_head.* and embed_tokens.* weights this graph reuses (the CTE module
        instantiates mtp_head whenever mtp_num_hidden_layers > 0).
        """
        import copy

        from neuronx_distributed_inference.modules import autobucketing

        self.compile_tag = MTP_HEAD_MODEL_TAG
        new_config = copy.deepcopy(self.config)
        new_config.neuron_config.batch_size = self.neuron_config.tkg_batch_size
        new_config.neuron_config.n_active_tokens = 1
        new_config.neuron_config.bucket_n_active_tokens = False
        new_config.neuron_config.sequence_parallel_enabled = False
        new_config.neuron_config.is_prefill_stage = False
        new_config.neuron_config.buckets = autobucketing.generate_buckets_for_tkg(
            new_config
        )

        self.mtp_head_model = MTPHeadModelWrapper(
            config=new_config,
            model_cls=NeuronMTPDraftModel,
            tag=MTP_HEAD_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.mtp_head_model)
        # Restore the default compile_tag for any subsequent get_compiler_args().
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG

    def enable_verify_backbone(self):
        """Register NeuronVerifyModel as an additional compiled graph (n_active=2).

        Mirrors enable_mtp_head's decode-style config (batch_size =
        tkg_batch_size) but traces the full 40-layer NeuronVerifyModel at
        n_active_tokens=2 via VerifyModelWrapper. The shared sharded checkpoint
        already carries every weight this graph reuses (embeddings, layers, norm,
        lm_head), so no extra weights are loaded.
        """
        import copy

        from neuronx_distributed_inference.modules import autobucketing

        self.compile_tag = VERIFY_MODEL_TAG
        new_config = copy.deepcopy(self.config)
        new_config.neuron_config.batch_size = self.neuron_config.tkg_batch_size
        new_config.neuron_config.n_active_tokens = 2
        new_config.neuron_config.bucket_n_active_tokens = False
        new_config.neuron_config.sequence_parallel_enabled = False
        new_config.neuron_config.is_prefill_stage = False
        new_config.neuron_config.buckets = autobucketing.generate_buckets_for_tkg(
            new_config
        )

        self.verify_model = VerifyModelWrapper(
            config=new_config,
            model_cls=NeuronVerifyModel,
            tag=VERIFY_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.verify_model)
        # Restore the default compile_tag for any subsequent get_compiler_args().
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG

    def draft(self, prev_hidden, next_token_ids, position_ids, seq_ids):
        """Host entry point for the on-device MTP draft graph.

        Drafts the t+2 logits from the trunk hidden at t (prev_hidden, [B,1,H])
        and the committed token t+1 (next_token_ids, [B,1]). Routes to the MTP
        graph by tag; its input signature is shape-distinct from TKG, but passing
        model_name keeps the runtime unambiguous. Returns the draft logits.
        """
        if getattr(self, "mtp_head_model", None) is None:
            raise RuntimeError(
                "MTP draft graph is not registered; build with "
                "mtp_num_hidden_layers > 0 (A3B_ENABLE_MTP=1)."
            )

        # Build the SAME 24-arg positional tuple the MTP graph was traced with
        # (see MTPHeadModelWrapper.input_generator). The committed next-token
        # ids go in idx 0 (input_ids), position_ids in idx 2, seq_ids in idx 3,
        # and the trunk hidden in idx 5 (prev_hidden); every other slot is an
        # empty placeholder. The non-empty prev_hidden at idx 5 distinguishes
        # this graph from the token-generation graph for the shape-router.
        #
        # NxD's runtime router matches the FULL per-slot shape signature against
        # the traced example inputs, so these slots must be byte-identical in
        # shape/dtype to MTPHeadModelWrapper.input_generator(). idx1
        # (attention_mask) and idx4 (sampling_params) are therefore both EMPTY
        # int32 here: the draft forward builds its own mask and never reads
        # sampling_params (on_device_sampling is False for this model).
        # Fresh empty object per slot (see input_generator): extract() dedups by
        # identity, so reusing one empty would diverge from the traced layout.
        torch_dtype = self.config.neuron_config.torch_dtype

        def _empty_i32():
            return torch.zeros((0,), dtype=torch.int32)

        def _empty_f():
            return torch.zeros((0,), dtype=torch_dtype)

        args24 = [
            next_token_ids,  # 0  input_ids (committed next token)
            _empty_i32(),    # 1  attention_mask (built on-device)
            position_ids,    # 2  position_ids
            seq_ids,         # 3  seq_ids
            _empty_i32(),    # 4  sampling_params (unused by draft forward)
            prev_hidden,     # 5  prev_hidden (trunk hidden, distinguishing input)
        ]
        # idx 6..20: empty int32 placeholders (adapter_ids + 14 unused slots).
        args24 += [_empty_i32() for _ in range(15)]
        args24.append(_empty_i32())  # 21 rotary_position_id
        args24.append(_empty_f())    # 22 vision_embeddings
        args24.append(_empty_i32())  # 23 vision_mask

        outputs = self.mtp_head_model(*args24)
        # is_neuron(): bare logits; CPU: [logits, draft_k, draft_v].
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs

    def verify(self, prev_hidden, block_token_ids, position_ids, seq_ids):
        """Host entry point for the on-device verify backbone graph.

        Runs the 2-token block ``block_token_ids`` [B,2] (= [x_{t+1},
        x_{t+2}^draft]) at positions ``position_ids`` [B,2], seeded from the
        committed decode state, and returns the verify graph's output list
        ``[logits, *updated_kv]`` where ``logits`` is per-position [B,2,vocab].

        ``prev_hidden`` [B,2,H] is the router disambiguator (its value is not
        used for compute). The 24-arg call tuple MUST be byte-shape-identical to
        VerifyModelWrapper.input_generator() in all 24 slots; the non-empty
        prev_hidden [B,2,H] at idx5 plus model_name keep the runtime
        unambiguous. Fresh empty object per slot (extract() dedups by identity).
        """
        if getattr(self, "verify_model", None) is None:
            raise RuntimeError(
                "Verify backbone graph is not registered; build with "
                "enable_verify_backbone (A3B_ENABLE_VERIFY=1)."
            )

        torch_dtype = self.config.neuron_config.torch_dtype

        def _empty_i32():
            return torch.zeros((0,), dtype=torch.int32)

        def _empty_f():
            return torch.zeros((0,), dtype=torch_dtype)

        args24 = [
            block_token_ids,  # 0  input_ids [B,2]
            _empty_i32(),     # 1  attention_mask (built on-device)
            position_ids,     # 2  position_ids [B,2]
            seq_ids,          # 3  seq_ids
            _empty_i32(),     # 4  sampling_params (unused by verify forward)
            prev_hidden,      # 5  prev_hidden [B,2,H] (router disambiguator)
        ]
        # idx 6..20: empty int32 placeholders.
        args24 += [_empty_i32() for _ in range(15)]
        args24.append(_empty_i32())  # 21 rotary_position_id
        args24.append(_empty_f())    # 22 vision_embeddings
        args24.append(_empty_i32())  # 23 vision_mask

        outputs = self.verify_model(*args24)
        # is_neuron(): bare logits; CPU: [logits, *updated_kv].
        if isinstance(outputs, (list, tuple)):
            return list(outputs)
        return [outputs]

    def _copy_past_key_values(self, outputs):
        """Override to also copy DeltaNet state buffers on CPU."""
        super()._copy_past_key_values(outputs)
        if getattr(self.config, "use_hybrid_cache_manager", False):
            return

        num_output_from_trace = 1
        if (
            self.neuron_config.output_logits
            and self.neuron_config.on_device_sampling_config
        ):
            num_output_from_trace = 2

        if (
            hasattr(self, "token_generation_model")
            and self.token_generation_model is not None
        ):
            tkg_model = self.token_generation_model.model
            cte_model = self.context_encoding_model.model
        else:
            return

        if tkg_model.kv_mgr is not None:
            num_kv = len(tkg_model.kv_mgr.past_key_values)
        else:
            num_kv = 0

        state_start = num_output_from_trace + num_kv

        tkg_params = getattr(tkg_model, "_deltanet_state_params", [])
        cte_params = getattr(cte_model, "_deltanet_state_params", [])

        if len(tkg_params) > 0 and state_start + len(tkg_params) <= len(outputs):
            for i, (tkg_param, cte_param) in enumerate(zip(tkg_params, cte_params)):
                new_state = outputs[state_start + i]
                tkg_param.data = new_state
                cte_param.data = new_state

    def get_required_kwargs(self):
        """Return extra kwargs for HF generation loop."""
        return ["llava_args"]

    def _get_model_outputs(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden,
        adapter_ids,
        medusa_args,
        llava_args,
        slot_mapping=None,
        block_table=None,
        full_context_lens=None,
        computed_context_lens=None,
        tf_args=None,
    ):
        """Override to pass all 24 positional args explicitly."""
        is_prefill = self._is_prefill(position_ids) or (
            getattr(self.config, "use_qwen_hybrid_chunked_prefill", False)
            and input_ids.shape[-1] > 1
        )

        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        if llava_args and len(llava_args) >= 2:
            vision_embeddings = llava_args[0]
            vision_mask = llava_args[1]
            if len(llava_args) >= 3:
                mrope_position_ids = llava_args[2]
            else:
                mrope_position_ids = None
        elif is_prefill:
            vision_embeddings = torch.zeros(
                (batch_size, seq_len, self.config.hidden_size),
                dtype=self.config.neuron_config.torch_dtype,
            )
            vision_mask = torch.full(
                (batch_size, seq_len, 1),
                fill_value=seq_len - 1,
                dtype=torch.int32,
            )
            mrope_position_ids = None
        else:
            vision_embeddings = torch.zeros((0,), dtype=torch.float32)
            vision_mask = torch.zeros((0,), dtype=torch.int32)
            mrope_position_ids = None

        if is_prefill:
            if mrope_position_ids is None:
                mrope_position_ids = (
                    torch.arange(0, seq_len, dtype=torch.int32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(3, batch_size, -1)
                    .contiguous()
                )
        else:
            mrope_position_ids = torch.zeros((0,), dtype=torch.int32)

        empties = [torch.empty(0) for _ in range(14)]

        if is_prefill:
            ctx_bs = self.context_encoding_model.neuron_config.batch_size
            output_logits = []

            for cb in range(0, batch_size, ctx_bs):
                cb_end = min(cb + ctx_bs, batch_size)
                actual_chunk = cb_end - cb

                chunk_input_ids = input_ids[cb:cb_end]
                chunk_attn_mask = attention_mask[cb:cb_end]
                chunk_pos_ids = position_ids[cb:cb_end]
                chunk_seq_ids = seq_ids[cb:cb_end]
                chunk_sampling = sampling_params[cb:cb_end]
                chunk_prev_hidden = (
                    prev_hidden[cb:cb_end]
                    if prev_hidden is not None
                    and hasattr(prev_hidden, "ndim")
                    and prev_hidden.ndim > 0
                    and prev_hidden.shape[0] > 0
                    else prev_hidden
                )
                chunk_adapter_ids = (
                    adapter_ids[cb:cb_end]
                    if adapter_ids is not None
                    and hasattr(adapter_ids, "ndim")
                    and adapter_ids.ndim > 0
                    and adapter_ids.shape[0] > 0
                    else adapter_ids
                )

                if mrope_position_ids.ndim == 3:
                    chunk_mrope = mrope_position_ids[:, cb:cb_end, :]
                else:
                    chunk_mrope = mrope_position_ids

                if vision_embeddings.ndim == 3:
                    chunk_vis_emb = vision_embeddings[cb:cb_end]
                    chunk_vis_mask = vision_mask[cb:cb_end]
                else:
                    chunk_vis_emb = vision_embeddings
                    chunk_vis_mask = vision_mask

                if actual_chunk < ctx_bs:
                    pad_n = ctx_bs - actual_chunk
                    chunk_input_ids = torch.cat(
                        [chunk_input_ids, chunk_input_ids[:1].expand(pad_n, -1)], dim=0
                    )
                    chunk_attn_mask = torch.cat(
                        [chunk_attn_mask, chunk_attn_mask[:1].expand(pad_n, -1)], dim=0
                    )
                    chunk_pos_ids = torch.cat(
                        [chunk_pos_ids, chunk_pos_ids[:1].expand(pad_n, -1)], dim=0
                    )
                    pad_seq = torch.arange(
                        batch_size, batch_size + pad_n, dtype=chunk_seq_ids.dtype
                    )
                    chunk_seq_ids = torch.cat([chunk_seq_ids, pad_seq], dim=0)
                    chunk_sampling = torch.cat(
                        [chunk_sampling, chunk_sampling[:1].expand(pad_n, -1)], dim=0
                    )
                    if (
                        chunk_prev_hidden is not None
                        and hasattr(chunk_prev_hidden, "ndim")
                        and chunk_prev_hidden.ndim > 0
                        and chunk_prev_hidden.shape[0] > 0
                    ):
                        chunk_prev_hidden = torch.cat(
                            [
                                chunk_prev_hidden,
                                chunk_prev_hidden[:1].expand(pad_n, -1),
                            ],
                            dim=0,
                        )
                    if (
                        chunk_adapter_ids is not None
                        and hasattr(chunk_adapter_ids, "ndim")
                        and chunk_adapter_ids.ndim > 0
                        and chunk_adapter_ids.shape[0] > 0
                    ):
                        chunk_adapter_ids = torch.cat(
                            [
                                chunk_adapter_ids,
                                chunk_adapter_ids[:1].expand(pad_n, -1),
                            ],
                            dim=0,
                        )
                    if chunk_mrope.ndim == 3:
                        chunk_mrope = torch.cat(
                            [chunk_mrope, chunk_mrope[:, :1, :].expand(-1, pad_n, -1)],
                            dim=1,
                        )
                    if chunk_vis_emb.ndim == 3:
                        chunk_vis_emb = torch.cat(
                            [
                                chunk_vis_emb,
                                torch.zeros(
                                    (pad_n,) + chunk_vis_emb.shape[1:],
                                    dtype=chunk_vis_emb.dtype,
                                ),
                            ],
                            dim=0,
                        )
                        chunk_vis_mask = torch.cat(
                            [
                                chunk_vis_mask,
                                torch.full(
                                    (pad_n,) + chunk_vis_mask.shape[1:],
                                    fill_value=seq_len - 1,
                                    dtype=chunk_vis_mask.dtype,
                                ),
                            ],
                            dim=0,
                        )

                chunk_out = self.context_encoding_model(
                    chunk_input_ids,
                    chunk_attn_mask,
                    chunk_pos_ids,
                    chunk_seq_ids,
                    chunk_sampling,
                    chunk_prev_hidden,
                    chunk_adapter_ids,
                    *empties,
                    chunk_mrope,
                    chunk_vis_emb,
                    chunk_vis_mask,
                )
                if actual_chunk < ctx_bs:
                    chunk_out = chunk_out[:actual_chunk]
                output_logits.append(chunk_out)

            outputs = (
                torch.cat(output_logits, dim=0)
                if len(output_logits) > 1
                else output_logits[0]
            )
            self.kv_cache_populated = True
            is_run_on_neuron = self.context_encoding_model.is_neuron()
        else:
            outputs = self.token_generation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
                *empties,
                mrope_position_ids,
                vision_embeddings,
                vision_mask,
            )
            is_run_on_neuron = self.token_generation_model.is_neuron()

        return outputs, is_run_on_neuron

    def get_compiler_args(self):
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        else:
            optimization_level = "-O1"

        compiler_args = (
            "--enable-saturate-infinity "
            "--enable-mixed-precision-accumulation "
            f"--model-type transformer {optimization_level} "
            "--auto-cast=none "
        )
        return compiler_args
