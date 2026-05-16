# coding=utf-8
"""GPT-OSS-20B with the 24-layer fused TKG megakernel.

CTE: stock per-layer NxDI path.
TKG: `NeuronGptOssModelV2.get_model_output` bypasses NxDI's per-layer loop and
calls the megakernel once for all 24 layers. KV cache updates are in-place via
the kernel's scatter.
"""

import math
import os
import shlex
import warnings
from typing import Optional, Tuple

import torch
from torch import nn

# attention_block_tkg hard-codes kv_heads=1; gpt-oss has 8 KV heads → TP=8.
# trn3 has 8 physical cores per chip, so TP=8 only fits at LNC=1.
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3")

from neuronx_distributed.parallel_layers.layers import (  # noqa: E402
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig  # noqa: E402
from neuronx_distributed_inference.models.gpt_oss.modeling_gpt_oss import (  # noqa: E402
    GptOssInferenceConfig,
    GptOssNeuronConfig,
    NeuronGptOssDecoderLayer,
    NeuronGptOssForCausalLM as _BaseNeuronGptOssForCausalLM,
    NeuronGptOssModel,
    get_updated_configs,
)
from neuronx_distributed_inference.models.layer_boundary_marker import (  # noqa: E402
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from neuronx_distributed_inference.models.model_wrapper import (  # noqa: E402
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from neuronx_distributed_inference.modules.lora_serving import is_lora_module  # noqa: E402

# Module-level imports for the MX CTE monkey-patched fns. NKI's parser doesn't
# allow `import` inside function bodies, and disambiguating aliases trips its
# name resolution — match the names used in tests/gpt_oss/test_mxfp4_cte.py.
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402
from nkilib.core.utils.kernel_assert import kernel_assert  # noqa: E402
from nkilib.core.moe.moe_cte.moe_cte_utils import (  # noqa: E402
    PSUM_SIZE,
    TILE_SIZE,
    div_ceil,
)


# --- MXFP4 CTE kernel integration -------------------------------------------
# We call nkilib's bwmm_shard_on_block_mx directly for the MoE in CTE under MX,
# bypassing NxDI's BlockwiseMatmulNKIFunc dispatcher (which doesn't support the
# gpt-oss swizzled MX scale layout). The kernel itself supports our layout but
# is shipped LNC=2-only; two monkey-patches at module level make it run at
# LNC=1 (collectives inside the kernel are already gated `if num_shards == 2`).
# See tests/gpt_oss/test_mxfp4_cte.py for the standalone correctness test.

_MX_CTE_PATCHES_APPLIED = False


def _mx_check_kernel_compatibility(dims, configs):
    """Drop-in for bwmm_shard_on_block_mx.check_kernel_compatibility that
    omits the LNC=2 hard requirement; everything else still checked."""
    kernel_assert(dims.B % 128 == 0, "Blocksize must be a multiple of 128")
    kernel_assert(512 <= dims.H <= 8192, f"Hidden dims must be in [512, 8192], found {dims.H}")
    kernel_assert(dims.H % PSUM_SIZE == 0, f"H must be multiple of {PSUM_SIZE}, found {dims.H}")
    kernel_assert(dims.I % 16 == 0, f"down_proj I must be divisible by 16, found {dims.I}")
    kernel_assert(configs.is_tensor_update_accumulating, "Only topK > 1 supported")
    if configs.use_dynamic_while:
        kernel_assert(dims.cond_vec_len == dims.N + 2, "cond vec must be N+2")


def _mx_output_init_lnc1_safe(output, dims):
    """Drop-in for bwmm_shard_on_block_mx.output_initialization that always
    uses the 3D-aware sharded path. The upstream 2D path (num_shards==1
    fallback) OOB-fails because output is allocated 3D (2, T+1, H)."""
    T_loc = dims.T
    H_loc = dims.H
    for tile_idx in range(div_ceil(T_loc, TILE_SIZE)):
        num_p = min(TILE_SIZE, T_loc - tile_idx * TILE_SIZE)
        zeros = nl.ndarray((TILE_SIZE, H_loc), dtype=output.dtype, buffer=nl.sbuf)
        nisa.memset(zeros, value=0)
        nisa.dma_copy(
            dst=output.ap(
                pattern=[[H_loc, num_p], [1, H_loc]],
                offset=dims.shard_id * T_loc * H_loc + tile_idx * TILE_SIZE * H_loc,
            ),
            src=zeros.ap(pattern=[[H_loc, num_p], [1, H_loc]], offset=0),
        )


def _apply_mx_cte_lnc1_patches():
    """Idempotent: patch bwmm_shard_on_block_mx for LNC=1 once."""
    global _MX_CTE_PATCHES_APPLIED
    if _MX_CTE_PATCHES_APPLIED:
        return
    from nkilib.core.moe.moe_cte import bwmm_shard_on_block_mx as _km
    _km.check_kernel_compatibility = _mx_check_kernel_compatibility
    _km.output_initialization = _mx_output_init_lnc1_safe
    _MX_CTE_PATCHES_APPLIED = True


def _mx_forward_blockwise(self, hidden_states, expert_affinities, expert_index,
                          expert_affinities_masked_full=None, padding_mask=None):
    """Drop-in for ExpertMLPsV2.forward_blockwise under MXFP4.

    Bypasses NxDI's `can_use_blockwise_matmul_nki` dispatcher (which routes to
    `BlockwiseMatmulNKIFunc` whose dequant fallback doesn't support gpt-oss's
    swizzled MX layout). Mirrors the routing-prep that the original
    forward_blockwise does, then calls `bwmm_shard_on_block_mx` directly at
    LNC=1 grid via the two module-level monkey-patches above.
    """
    import math
    import nki
    from nkilib.core.moe.moe_cte.bwmm_shard_on_block_mx import bwmm_shard_on_block_mx
    from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
    from nkilib.core.moe.moe_cte.moe_cte_utils import SkipMode
    from neuronx_distributed.modules.moe.blockwise import (
        augment_inputs_for_padded_blockwise_matmul,
    )

    _apply_mx_cte_lnc1_patches()

    mlp_op = self.get_mlp_op()
    total_tokens, hidden_size = hidden_states.shape
    local_experts = self.routed_experts_mlp_config.num_experts
    block_size = self.blockwise_matmul_config.block_size
    top_k = self.routed_experts_mlp_config.top_k

    # num_blocks: same formula as expert_mlps_v2.forward_blockwise.
    num_blocks = (
        math.ceil((total_tokens * top_k - (local_experts - 1)) / block_size)
        + (local_experts - 1)
    )
    num_blocks = min(num_blocks, total_tokens * top_k)

    expert_mask = self.get_expert_mask(expert_index, local_experts)
    expert_affinities_masked = self.get_expert_affinities_masked(
        expert_affinities, expert_mask,
        self.routed_experts_mlp_config.normalize_top_k_affinities,
    )
    expert_mask, expert_affinities_masked = self.mask_padding_tokens(
        expert_mask, expert_affinities_masked, padding_mask,
    )

    block_to_expert, token_position_to_id = self.get_blockwise_expert_and_token_mapping(
        total_tokens=total_tokens,
        num_blocks=num_blocks,
        expert_mask=expert_mask,
        expert_index=expert_index,
        block_size=block_size,
        device=hidden_states.device,
        enable_spmd_rank=False,
        spmd_rank=None,
        tensor_parallel_group=self.moe_tensor_model_parallel_group,
        optimized_block_to_token_mapping=(
            self.blockwise_matmul_config.optimized_block_to_token_mapping,
        ),
        parallelize_token_to_block_mapping=(
            self.blockwise_matmul_config.parallelize_token_to_block_mapping
        ),
        pad_num_blocks_to_even=False,
    )

    # T -> T+1 padding row.
    output = torch.zeros(
        total_tokens, hidden_size,
        device=hidden_states.device, dtype=hidden_states.dtype,
    )
    output, hidden_padded, tok2id_padded, aff_padded = (
        augment_inputs_for_padded_blockwise_matmul(
            output, hidden_states, token_position_to_id, expert_affinities_masked,
        )
    )

    # Reshape gate_up bias to the kernel's expected [E, I_p, 2, n_I_tiles, 4].
    # The bias arrives flat [E, 2*I_per_rank] from the NxDI MX layout transform.
    E = mlp_op.gate_up_proj.weight.shape[0]
    I_per_rank = (
        self.routed_experts_mlp_config.intermediate_size
        // self.moe_tensor_model_parallel_group.size()
    )
    n_I_tiles = max(1, math.ceil(I_per_rank / 512))
    I_TP_block_size = I_per_rank // n_I_tiles
    gate_up_bias_view = mlp_op.gate_up_proj.bias.view(
        E, I_TP_block_size // 4, 2, n_I_tiles, 4,
    )

    # Scales: NxDI's model_wrapper.load_module unconditionally upcasts any
    # param ending in "scale" to float32 (model_wrapper.py:1473-1475). Our MX
    # block-scales (E8M0) are uint8 with integer values in [0, 255] — cast
    # back so the kernel sees the expected uint8 dtype.
    gate_up_scale_u8 = mlp_op.gate_up_proj.scale.to(torch.uint8)
    down_scale_u8 = mlp_op.down_proj.scale.to(torch.uint8)

    # Launch kernel at LNC=1 grid.
    kernel_jit = nki.jit(bwmm_shard_on_block_mx)
    kernel_out = kernel_jit[1](
        hidden_states=hidden_padded,
        expert_affinities_masked=aff_padded.view(-1, 1),
        gate_up_proj_weight=mlp_op.gate_up_proj.weight,
        down_proj_weight=mlp_op.down_proj.weight,
        token_position_to_id=tok2id_padded.to(torch.int32),
        block_to_expert=block_to_expert.to(torch.int32),
        conditions=None,
        gate_and_up_proj_bias=gate_up_bias_view,
        down_proj_bias=mlp_op.down_proj.bias,
        gate_up_proj_scale=gate_up_scale_u8,
        down_proj_scale=down_scale_u8,
        block_size=block_size,
        gate_up_activations_T=None,
        down_activations=None,
        activation_function=ActFnType.Swish,
        skip_dma=SkipMode(False, False),
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        gate_clamp_upper_limit=7.0,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=7.0,
        up_clamp_lower_limit=-7.0,
    )
    # Output: (2, T+1, H). With num_shards=1 only shard 0 is written.
    return kernel_out[0, :total_tokens, :]


# ----------------------------------------------------------------------------


class GptOssV1MultilayerNeuronConfig(GptOssNeuronConfig):
    """GptOssNeuronConfig with the megakernel TKG path enabled."""

    def __init__(self, **kwargs):
        if kwargs.get("tp_degree", None) not in (None, 8):
            warnings.warn(
                f"GPT-OSS megakernel requires tp_degree=8; was {kwargs['tp_degree']}."
            )
        kwargs["tp_degree"] = 8
        kwargs["logical_nc_config"] = 1
        kwargs["logical_neuron_cores"] = 1
        # moe_tp_degree=tp_degree shards the I dim — without it each rank holds
        # 32 experts × 3072 H × 6144 (2*I) bf16 ≈ 32 GB just for gate_up.
        kwargs["moe_tp_degree"] = 8

        kwargs.setdefault("padded_hidden_size", 3072)        # 2880 → next mul of 128
        kwargs.setdefault("padded_intermediate_size", 3072)
        kwargs.setdefault("is_mxfp4_compute", False)
        kwargs.setdefault("sliding_window_attention_dp_degree", 1)

        # Megakernel TKG uses fused QKV in NKI layout regardless of mode.
        kwargs["attn_tkg_nki_kernel_enabled"] = True
        kwargs["attn_block_tkg_nki_kernel_cache_update"] = True
        kwargs["fused_qkv"] = True

        is_mx = kwargs.get("is_mxfp4_compute", False)
        if is_mx:
            # Uniformly-shuffled residual across the megakernel; +1 folded into
            # up_bias by the state-dict converter (matches NxDI's bf16 preshard
            # hook). router_dtype bf16 matches the bf16 router weight; default
            # fp32 would mismatch the matmul dtype check.
            kwargs.setdefault("is_full_model_shuffled", True)
            kwargs.setdefault("hidden_act_bias", 1.0)
            kwargs["moe_fused_nki_kernel_enabled"] = True
            kwargs["router_dtype"] = "bfloat16"
        else:
            kwargs.setdefault("is_full_model_shuffled", False)

        _tcc = kwargs.pop("tensor_capture_config", None)
        kwargs.setdefault("global_topk", kwargs.get("top_k", 64))
        kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(**kwargs)
        if _tcc is not None:
            kwargs["tensor_capture_config"] = _tcc

        if not is_mx:
            # main.py sets blockwise_matmul_config for the MX CTE NKI path;
            # bf16 doesn't use it.
            kwargs.pop("blockwise_matmul_config", None)
        super().__init__(**kwargs)


def convert_gpt_oss_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
    # Public converter does pad→convert; the private one skips pad. Pad first
    # so weights match the compiled hidden_size=3072.
    _BaseNeuronGptOssForCausalLM._pad_hf_state_dict(state_dict, config)
    state_dict = _BaseNeuronGptOssForCausalLM._convert_hf_format_state_dict(state_dict, config)

    # NxDI's loader walks expert_mlps via two paths under MX (moe.expert_mlps
    # and moe.moe_fused_tkg.expert_mlps — same Python object). Mirror the
    # keys so both lookups resolve.
    if config.neuron_config.is_mxfp4_compute:
        for l in range(config.num_hidden_layers):
            for proj in ("gate_up_proj", "down_proj"):
                for attr in ("weight", "scale", "bias"):
                    src = f"layers.{l}.feed_forward.moe.expert_mlps.mlp_op.{proj}.{attr}"
                    dst = f"layers.{l}.feed_forward.moe.moe_fused_tkg.expert_mlps.mlp_op.{proj}.{attr}"
                    if src in state_dict:
                        state_dict[dst] = state_dict[src]

    # Pre-bake the per-step Wqkv / Wo transposes and the Wo bias / tp divide
    # that the megakernel needs. Same layout for bf16 and MX.
    tp = config.neuron_config.tp_degree
    for l in range(config.num_hidden_layers):
        wqkv_key = f"layers.{l}.self_attn.Wqkv.weight"
        state_dict[f"layers.{l}.self_attn.Wqkv_nki.weight"] = (
            state_dict[wqkv_key].transpose(0, 1).contiguous()
        )
        wo_key = f"layers.{l}.self_attn.o_proj.weight"
        state_dict[f"layers.{l}.self_attn.Wo_nki.weight"] = (
            state_dict[wo_key].transpose(0, 1).contiguous()
        )
        # o_proj.bias is replicated; divide by tp so the kernel's AR-sum
        # reproduces the bias exactly.
        bo_key = f"layers.{l}.self_attn.o_proj.bias"
        state_dict[f"layers.{l}.self_attn.Wo_nki_bias"] = (
            state_dict[bo_key] / tp
        ).contiguous()

    return state_dict


class NeuronGptOssDecoderLayerV1(NeuronGptOssDecoderLayer):
    """Stock decoder layer + extra kernel-layout weight params for the
    megakernel TKG path. CTE uses the stock layer forward; TKG is bypassed
    at the model level so no forward override is needed here.
    """

    def __init__(self, config, layer_idx, rotary_cache_manager):
        # Mask attn_block_tkg_nki_kernel_enabled during super().__init__() so
        # GroupQueryAttention_O builds o_proj as a plain Linear (matching the
        # kernel-layout transpose in the state-dict converter). Restored after
        # init so NxDI's update_kv_per_layer gate fires on TKG.
        _saved = config.neuron_config.attn_block_tkg_nki_kernel_enabled
        config.neuron_config.attn_block_tkg_nki_kernel_enabled = False
        try:
            super().__init__(config, layer_idx, rotary_cache_manager)
        finally:
            config.neuron_config.attn_block_tkg_nki_kernel_enabled = _saved

        if config.neuron_config.is_mxfp4_compute:
            self._setup_mxfp4(config)

        # Wqkv_nki / Wo_nki / Wo_nki_bias are consumed by the megakernel TKG
        # path. The state-dict converter populates them.
        _dtype = config.neuron_config.torch_dtype
        H = config.hidden_size
        d_head = config.head_dim
        Hq_full = config.num_attention_heads * d_head
        Hkv_full = config.num_key_value_heads * d_head
        I_full = Hq_full + 2 * Hkv_full

        # RowParallelLinear shards dim 1 of weight → per-rank [H, I_per_rank],
        # matching the kernel layout. bias=False; QKV bias is read from the
        # original qkv.Wqkv.bias path which is already [I_per_rank].
        self.self_attn.Wqkv_nki = RowParallelLinear(
            input_size=I_full,
            output_size=H,
            bias=False,
            input_is_parallel=False,
            dtype=_dtype,
        )
        # The per-rank loader (trace.py:804) routes weights tagged `fused_qkv`
        # through create_local_weight_qkv, which splits Q/K/V independently
        # along partition_dim before chunking each per rank. Without this the
        # plain contiguous shard interleaves Q and K/V across ranks (wrong
        # attention output). create_local_weight_qkv accepts any partition_dim
        # so it works with our transposed [H, I_full] layout sharded on dim 1.
        setattr(self.self_attn.Wqkv_nki.weight, "fused_qkv", True)
        setattr(self.self_attn.Wqkv_nki.weight, "num_attention_heads",
                config.num_attention_heads)
        setattr(self.self_attn.Wqkv_nki.weight, "num_key_value_heads",
                config.num_key_value_heads)
        setattr(self.self_attn.Wqkv_nki.weight, "head_dim", d_head)
        # ColumnParallelLinear shards dim 0 → per-rank [Hq_per_rank, H], the
        # kernel's expected Wo layout.
        self.self_attn.Wo_nki = ColumnParallelLinear(
            input_size=H,
            output_size=Hq_full,
            bias=False,
            gather_output=False,
            dtype=_dtype,
        )
        # o_proj bias is replicated across ranks (RowParallel adds pre-AR);
        # store the divided copy as a plain Parameter (not sharded).
        self.self_attn.Wo_nki_bias = nn.Parameter(
            torch.empty(H, dtype=_dtype), requires_grad=False
        )

    def _setup_mxfp4(self, config):
        """Swap MoE projections to QuantizedExpertFused*, override scale
        params to the gpt-oss state-dict layout, and bind the CTE forward
        override that calls bwmm_shard_on_block_mx directly.
        """
        from neuronx_distributed.quantization.quantize import convert
        from neuronx_distributed.quantization.quantization_config import (
            QuantizationType, QuantizedDtype, ScaleDtype,
        )
        convert(
            self.feed_forward.moe.expert_mlps.mlp_op,
            q_config={
                "quantization_type": QuantizationType.BLOCKWISE_SYMMETRIC,
                "quantized_dtype": QuantizedDtype.F4E2M1FN_X4,
                "scale_dtype": ScaleDtype.F8E8M0,
                "block_axis": [1],
                "block_size": [32],
            },
            inplace=True,
        )

        # NxDI's QuantizedLinear._setup_for_scale picks block_size=32 along
        # block_axis=1 of the weight, giving scale shape (E, 4, 2, n_H, 1536).
        # gpt-oss state-dict layout is (E, 16, 2, n_H, I_per_rank) — same
        # bytes, different labels. Override the Parameters to match.
        mlp_op = self.feed_forward.moe.expert_mlps.mlp_op
        E = mlp_op.gate_up_proj._n_local_experts
        H = config.hidden_size
        I_per_rank = config.intermediate_size // config.neuron_config.moe_tp_degree
        tp = config.neuron_config.tp_degree
        n_H_tiles = max(1, math.ceil(H / 512))
        n_I_tiles = max(1, math.ceil(I_per_rank / 512))
        q_per_H_tile = 512 // 32       # 16
        q_per_I_tile = I_per_rank // 32  # 12 for gpt-oss I_per_rank=384

        def _attach_shard_meta(p, partition_dim):
            setattr(p, "tensor_model_parallel", True)
            setattr(p, "partition_dim", partition_dim)
            setattr(p, "partition_stride", 1)
            setattr(p, "num_partitions", tp)

        mlp_op.gate_up_proj.scale = nn.Parameter(
            torch.empty(E, q_per_H_tile, 2, n_H_tiles, I_per_rank, dtype=torch.uint8),
            requires_grad=False,
        )
        _attach_shard_meta(mlp_op.gate_up_proj.scale, partition_dim=4)
        mlp_op.down_proj.scale = nn.Parameter(
            torch.empty(E, q_per_I_tile, n_I_tiles, H, dtype=torch.uint8),
            requires_grad=False,
        )
        _attach_shard_meta(mlp_op.down_proj.scale, partition_dim=1)

        # CTE MoE: bypass NxDI's torch dispatcher (which doesn't accept the
        # swizzled MX layout) and call bwmm_shard_on_block_mx directly.
        expert_mlps = self.feed_forward.moe.expert_mlps
        expert_mlps.forward_blockwise = _mx_forward_blockwise.__get__(
            expert_mlps, type(expert_mlps)
        )


class NeuronGptOssModelV1(NeuronGptOssModel):
    """Overrides get_model_output to bypass NxDI's per-layer loop on TKG.
    CTE delegates to the stock path; TKG runs the multilayer megakernel,
    which scatters KV in-place into self.kv_mgr.past_key_values.
    """

    def init_model(self, config):
        super().init_model(config)
        # Replace stock decoder layers with V1 layers (kernel-layout weights).
        rotary_cache_manager = self.layers[0].rotary_cache_manager
        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList([
            NeuronGptOssDecoderLayerV1(conf, idx, rotary_cache_manager)
            for idx, conf in enumerate(updated_configs)
        ])
        self._num_hidden_layers = config.num_hidden_layers
        self._replica_groups = (list(range(config.neuron_config.tp_degree)),)
        self._num_heads_per_rank = (
            config.num_attention_heads // config.neuron_config.tp_degree
        )

    @staticmethod
    def _pick_qkv(attn):
        if hasattr(attn, "tkg_qkv_proj"):
            return attn.tkg_qkv_proj
        return attn.qkv_proj

    @staticmethod
    def _pick_sinks(attn):
        if hasattr(attn, "tkg_learned_sinks"):
            return attn.tkg_learned_sinks.sink
        return attn.learned_sinks.sink

    def _gather_weights(self):
        """Gather per-layer weights in kernel-expected layouts.

        Re-gathered every TKG call: caching nn.Parameter refs across traces
        trips `convert_parameters_to_constants`. The heavy transposes / bias
        divides are baked into the state dict by the converter; this is just
        cheap reshape views.
        """
        cfg = self.config
        H_padded = cfg.hidden_size
        I_per_rank = cfg.intermediate_size // cfg.neuron_config.moe_tp_degree
        E_count = cfg.num_local_experts
        is_mx = cfg.neuron_config.is_mxfp4_compute

        Wqkv, bqkv, Wo, bo = [], [], [], []
        sinks = []
        gpre, gpost = [], []
        router_w, router_b = [], []
        gate_up, gate_up_b, down, down_b = [], [], [], []
        gate_up_s, down_s = [], []

        # MX selective expects gate_up bias as [E, I_p, 2, n_I_tiles, 4].
        # NxDI's converter stores it flat [E, 2*I_per_rank] after per-rank
        # shard, where the implied axes are [q_blocks, Q_HEIGHT, 2, n_I_tiles,
        # Q_WIDTH] and I_p = q_blocks * Q_HEIGHT.
        Q_HEIGHT, Q_WIDTH = 8, 4
        n_I_tiles = max(1, I_per_rank // 512)
        q_blocks_per_I_tile = (I_per_rank // n_I_tiles) // (Q_HEIGHT * Q_WIDTH)
        I_p = q_blocks_per_I_tile * Q_HEIGHT

        for l in self.layers:
            attn = l.self_attn
            qkv = self._pick_qkv(attn)
            Wqkv.append(attn.Wqkv_nki.weight)
            bqkv.append(qkv.Wqkv.bias.reshape(1, -1))
            Wo.append(attn.Wo_nki.weight)
            bo.append(attn.Wo_nki_bias.reshape(1, -1))
            sinks.append(self._pick_sinks(attn).reshape(-1, 1))

            gpre.append(l.input_layernorm.weight.unsqueeze(0))
            gpost.append(l.post_attention_layernorm.weight.unsqueeze(0))

            moe = l.feed_forward.moe
            router_w.append(moe.router.weight_T)
            router_b.append(moe.router.linear_router.bias.reshape(1, -1))

            mlp_op = moe.expert_mlps.mlp_op
            if is_mx:
                # NxDI's MX layout has gate_up.weight already in
                # [E, 128, 2, n_H_tiles, I_per_rank] uint16 (fp4_x4 packed).
                # Scales must be uint8 — NxDI's model_wrapper.py force-upcasts
                # any param ending in "scale" to fp32, so cast back here.
                gate_up.append(mlp_op.gate_up_proj.weight)
                gate_up_s.append(mlp_op.gate_up_proj.scale.to(torch.uint8))
                # Flat [E, q_blocks*Q_HEIGHT*2*n_I_tiles*Q_WIDTH] →
                # view as [E, I_p, 2, n_I_tiles, Q_WIDTH].
                gu_b = mlp_op.gate_up_proj.bias.reshape(
                    E_count, q_blocks_per_I_tile, Q_HEIGHT, 2, n_I_tiles, Q_WIDTH
                ).reshape(E_count, I_p, 2, n_I_tiles, Q_WIDTH)
                gate_up_b.append(gu_b)
                down.append(mlp_op.down_proj.weight)
                down_s.append(mlp_op.down_proj.scale.to(torch.uint8))
                down_b.append(mlp_op.down_proj.bias)
            else:
                # gate at [..., 0, :], up at [..., 1, :] — matches NxDI's
                # convert_gate_up_proj which does cat((gate, up), dim=1).
                gu_w = mlp_op.gate_up_proj.weight
                gate_up.append(gu_w.reshape(E_count, H_padded, 2, I_per_rank))
                gu_b = mlp_op.gate_up_proj.bias
                gate_up_b.append(gu_b.reshape(E_count, 2, I_per_rank))
                down.append(mlp_op.down_proj.weight)
                down_b.append(mlp_op.down_proj.bias)

        return (Wqkv, bqkv, Wo, bo, sinks,
                gpre, gpost,
                router_w, router_b,
                gate_up, gate_up_b, down, down_b,
                gate_up_s, down_s)

    def _gather_kv_caches(self):
        # past_key_values is flat [K0, V0, K1, V1, ...]. SWA layers have
        # max_len=sliding_window-1; full layers have max_len=max_length.
        L = self._num_hidden_layers
        Ks = [self.kv_mgr.past_key_values[2 * i] for i in range(L)]
        Vs = [self.kv_mgr.past_key_values[2 * i + 1] for i in range(L)]
        return Ks, Vs

    def get_model_output(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        active_mask=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        update_cache: bool = False,
        is_for_context_encoding: bool = False,
        vision_embeddings=None,
        vision_mask=None,
        deepstack_vision_embeds=None,
        local_attn_mask: Optional[torch.Tensor] = None,
        windowed_context_encoding_window_idx: int = -1,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # CTE: delegate to NxDI's stock path. Under MX the MoE block is
        # routed through our _mx_forward_blockwise override.
        if is_for_context_encoding:
            return super().get_model_output(
                input_ids=input_ids,
                seq_ids=seq_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                active_mask=active_mask,
                inputs_embeds=inputs_embeds,
                prev_hidden=prev_hidden,
                adapter_ids=adapter_ids,
                rotary_position_ids=rotary_position_ids,
                update_cache=update_cache,
                is_for_context_encoding=is_for_context_encoding,
                vision_embeddings=vision_embeddings,
                vision_mask=vision_mask,
                deepstack_vision_embeds=deepstack_vision_embeds,
                local_attn_mask=local_attn_mask,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                padding_mask=padding_mask,
                **kwargs,
            )

        # TKG: embed → megakernel → norm → return.
        from megakernels.gpt_oss.transformer_gpt_oss import (
            get_multilayer_kernel_jit,
        )

        if inputs_embeds is None:
            inputs_embeds = (
                self.embed_tokens(input_ids)
                if not is_lora_module(self.embed_tokens)
                else self.embed_tokens(input_ids, adapter_ids=adapter_ids)
            )
        hidden_states = inputs_embeds
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)

        # gpt-oss emits [B, S, d_head] with the second half duplicated; pass
        # the un-permuted half-d slice and let the kernel do the
        # (B, S, d//2) -> (d//2, B, S) permute via a strided DMA on load.
        cos_cache, sin_cache = self.layers[0].self_attn.rotary_emb(
            hidden_states, position_ids
        )
        d_head = self.config.head_dim
        cos_unperm = cos_cache[..., : d_head // 2].contiguous()
        sin_unperm = sin_cache[..., : d_head // 2].contiguous()

        (Wqkv_l, bqkv_l, Wo_l, bo_l, sinks_l,
         gpre_l, gpost_l,
         router_w_l, router_b_l,
         gate_up_l, gate_up_b_l, down_l, down_b_l,
         gate_up_s_l, down_s_l) = self._gather_weights()
        K_caches, V_caches = self._gather_kv_caches()

        L = self._num_hidden_layers
        is_mx = self.config.neuron_config.is_mxfp4_compute
        scale_args = (*gate_up_s_l, *down_s_l) if is_mx else ()
        kernel_out = get_multilayer_kernel_jit(L, mxfp4=is_mx)[1](
            hidden_states,
            *Wqkv_l, *bqkv_l, *Wo_l, *bo_l,
            *sinks_l,
            *gpre_l, *gpost_l,
            *router_w_l, *router_b_l,
            *gate_up_l, *gate_up_b_l, *down_l, *down_b_l,
            *scale_args,
            *K_caches, *V_caches,
            cos_unperm, sin_unperm,
            position_ids.to(torch.int32),
            # SWA wraps the cache: index = position % (sliding_window-1).
            (position_ids.to(torch.int32)
             % (self.config.sliding_window - 1)),
            num_heads_per_rank=self._num_heads_per_rank,
            replica_groups=self._replica_groups,
        )
        Y = kernel_out[0]
        K_out = kernel_out[1     : 1 + L]
        V_out = kernel_out[1 + L : 1 + 2 * L]

        Y = ModuleMarkerEndWrapper()(Y)

        hidden_states = self.norm(Y)

        # Flat [K0, V0, K1, V1, ...] — matches NxDI's update_kv_per_layer
        # convention for downstream consumers.
        next_decoder_cache = []
        for i in range(L):
            next_decoder_cache.append(K_out[i])
            next_decoder_cache.append(V_out[i])

        return (hidden_states, next_decoder_cache)


class NeuronGptOssForCausalLM(_BaseNeuronGptOssForCausalLM):
    """GPT-OSS-20B CausalLM with the 24-layer fused TKG megakernel."""

    _model_cls = NeuronGptOssModelV1

    @classmethod
    def get_neuron_config_cls(cls):
        return GptOssV1MultilayerNeuronConfig

    @classmethod
    def get_config_cls(cls):
        return GptOssInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        return convert_gpt_oss_hf_to_neuron_state_dict(state_dict, config)

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
            "--model-type", "transformer",
            f"--lnc={self.neuron_config.logical_nc_config}",
        ]
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=4",
                "--vectorize-strided-dma",
                "--enable-scalar-dge-vectorization",
            ]
            hlo2tensorizer_extra = "--modular-flow-mac-threshold=10"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O1"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=4",
                "--vectorize-strided-dma",
                "--eager-tkg-vectorize-dma",
                "--enable-dge-on-indirect-dma",
                "--enable-dge-on-vector-indirect-dma",
            ]
            hlo2tensorizer_extra = ""
            # NCC-6661: NKI in-place KV scatter trips the backend verifier.
            args.append("--internal-backend-options=--enable-verifier=false")
        else:
            optimization_level = "-O1"
            tensorizer_opts = [
                "--enable-ccop-compute-overlap",
                "--cc-pipeline-tiling-factor=4",
                "--vectorize-strided-dma",
                "--enable-scalar-dge-vectorization",
            ]
            hlo2tensorizer_extra = ""

        if tensorizer_opts:
            args.append(f"--tensorizer-options={' '.join(tensorizer_opts)}")
        args.append(optimization_level)
        args.append("--auto-cast=none")
        args += ["--internal-enable-dge-levels", "vector_dynamic_offsets"]
        args.append("--internal-max-instruction-limit=30000000")

        if self.neuron_config.scratchpad_page_size:
            args.append(
                f"--hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size}"
            )

        if hlo2tensorizer_extra:
            args.append(
                f"--internal-hlo2tensorizer-options={hlo2tensorizer_extra} --verify-hlo=true"
            )

        return shlex.join(args)
