# coding=utf-8
"""GPT-OSS-20B with the 24-layer fused TKG megakernel.

CTE: stock per-layer NxDI path.
TKG: layer 0 dispatches the megakernel for all 24 layers; layers 1..L-1 are
pass-throughs that forward the kernel's KV outputs into next_decoder_cache.
"""

import os
import shlex
import warnings
import weakref
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
        kwargs.setdefault("is_mxfp4_compute", False)         # phase 1: bf16
        kwargs.setdefault("is_full_model_shuffled", False)
        kwargs.setdefault("sliding_window_attention_dp_degree", 1)
        # Megakernel scatters KV in place; NxDI must skip its Python scatter.
        kwargs["attn_tkg_nki_kernel_enabled"] = True
        kwargs["attn_block_tkg_nki_kernel_cache_update"] = True
        # gpt-oss has QKV bias; fused projection matches attention_block_tkg.
        kwargs["fused_qkv"] = True

        _tcc = kwargs.pop("tensor_capture_config", None)
        kwargs.setdefault("global_topk", kwargs.get("top_k", 64))
        kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(**kwargs)
        if _tcc is not None:
            kwargs["tensor_capture_config"] = _tcc

        kwargs.pop("blockwise_matmul_config", None)
        super().__init__(**kwargs)


def convert_gpt_oss_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
    # Public converter does pad→convert; the private one skips pad. Pad first
    # so weights match the compiled hidden_size=3072.
    _BaseNeuronGptOssForCausalLM._pad_hf_state_dict(state_dict, config)
    state_dict = _BaseNeuronGptOssForCausalLM._convert_hf_format_state_dict(state_dict, config)

    # Bake the per-step transposes / bias-divide that _gather_weights_from_parent
    # would otherwise do every TKG forward. New keys are loaded into the
    # NeuronGptOssDecoderLayerV1 modules registered below.
    tp = config.neuron_config.tp_degree
    for l in range(config.num_hidden_layers):
        # Wqkv: [I_full, H] → [H, I_full]; RowParallelLinear shards dim 1 →
        # per-rank [H, I_per_rank] which is the kernel's expected layout.
        wqkv_key = f"layers.{l}.self_attn.Wqkv.weight"
        state_dict[f"layers.{l}.self_attn.Wqkv_nki.weight"] = (
            state_dict[wqkv_key].transpose(0, 1).contiguous()
        )

        # o_proj weight: [H, Hq_full] → [Hq_full, H]; ColumnParallelLinear
        # shards dim 0 → per-rank [Hq_per_rank, H], matching the kernel.
        wo_key = f"layers.{l}.self_attn.o_proj.weight"
        state_dict[f"layers.{l}.self_attn.Wo_nki.weight"] = (
            state_dict[wo_key].transpose(0, 1).contiguous()
        )

        # o_proj.bias is replicated [H] (RowParallel adds it pre-AR). Divide
        # by tp once so the kernel's AR-sum reproduces the bias exactly.
        bo_key = f"layers.{l}.self_attn.o_proj.bias"
        state_dict[f"layers.{l}.self_attn.Wo_nki_bias"] = (
            state_dict[bo_key] / tp
        ).contiguous()

    # NB: router transpose is handled by NxDI's RouterTopK(store_transposed_weights=True)
    # which auto-registers `router.weight_T` (shape [H, E]) on device. We just
    # read it from _gather_weights_from_parent.

    return state_dict


class NeuronGptOssDecoderLayerV1(NeuronGptOssDecoderLayer):
    """Layer 0 dispatches the megakernel; layers 1..L-1 are pass-throughs."""

    def __init__(self, config, layer_idx, rotary_cache_manager):
        # Mask attn_block_tkg_nki_kernel_enabled during super().__init__() so
        # GroupQueryAttention_O builds o_proj as a plain Linear (weight stored
        # as [H, Hq_per_rank], untransposed). Restore after init so
        # model_base.py's update_kv_per_layer gate still fires on TKG.
        _saved = config.neuron_config.attn_block_tkg_nki_kernel_enabled
        config.neuron_config.attn_block_tkg_nki_kernel_enabled = False
        try:
            super().__init__(config, layer_idx, rotary_cache_manager)
        finally:
            config.neuron_config.attn_block_tkg_nki_kernel_enabled = _saved
        self._num_hidden_layers = config.num_hidden_layers
        self._replica_groups = (list(range(config.neuron_config.tp_degree)),)
        # Set by NeuronGptOssModelV1.init_model. Weakref to avoid module cycle.
        self._parent_model_ref = None

        # Pre-baked kernel-layout weights. Receive values from
        # convert_gpt_oss_hf_to_neuron_state_dict so _gather_weights_from_parent
        # can skip the per-step .T.contiguous() / bias divide.
        _dtype = config.neuron_config.torch_dtype
        H = config.hidden_size
        d_head = config.head_dim
        Hq_full = config.num_attention_heads * d_head
        Hkv_full = config.num_key_value_heads * d_head
        I_full = Hq_full + 2 * Hkv_full

        # RowParallelLinear shards dim 1 of weight → per-rank [H, I_per_rank],
        # matching the kernel layout. bias=False; we read the QKV bias from the
        # original (unaltered) qkv.Wqkv.bias path which is already [I_per_rank].
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

    @property
    def _parent_model(self):
        return self._parent_model_ref() if self._parent_model_ref is not None else None

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

    def _gather_weights_from_parent(self):
        """Gather per-layer weights, reshaped/transposed to kernel layouts."""
        cfg = self.config
        H_padded = cfg.hidden_size
        I_per_rank = cfg.intermediate_size // cfg.neuron_config.moe_tp_degree
        E_count = cfg.num_local_experts

        Wqkv, bqkv, Wo, bo = [], [], [], []
        sinks = []
        gpre, gpost = [], []
        router_w, router_b = [], []
        gate_up, gate_up_b, down, down_b = [], [], [], []

        for l in self._parent_model.layers:
            attn = l.self_attn
            qkv = self._pick_qkv(attn)
            # Pre-baked by convert_gpt_oss_hf_to_neuron_state_dict: weight in
            # kernel layout [H, I_per_rank], bias divided by tp_degree.
            Wqkv.append(attn.Wqkv_nki.weight)
            bqkv.append(qkv.Wqkv.bias.reshape(1, -1))
            Wo.append(attn.Wo_nki.weight)
            bo.append(attn.Wo_nki_bias.reshape(1, -1))
            sinks.append(self._pick_sinks(attn).reshape(-1, 1))

            gpre.append(l.input_layernorm.weight.unsqueeze(0))
            gpost.append(l.post_attention_layernorm.weight.unsqueeze(0))

            moe = l.feed_forward.moe
            # store_transposed_weights=True (set via init_tkg_module) auto-
            # registers router.weight_T with shape [H, E].
            router_w.append(moe.router.weight_T)
            router_b.append(moe.router.linear_router.bias.reshape(1, -1))

            mlp_op = moe.expert_mlps.mlp_op
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
                gate_up, gate_up_b, down, down_b)

    def _gather_kv_caches(self, kv_mgr):
        # past_key_values is flat [K0, V0, K1, V1, ...]. SWA layers have
        # max_len=sliding_window-1; full layers have max_len=max_length.
        L = self._num_hidden_layers
        Ks = [kv_mgr.past_key_values[2 * i] for i in range(L)]
        Vs = [kv_mgr.past_key_values[2 * i + 1] for i in range(L)]
        return Ks, Vs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        kv_mgr=None,
        idx: int = 0,
        is_for_context_encoding: bool = True,
        **kwargs,
    ):
        is_tkg = not is_for_context_encoding

        if is_tkg:
            from megakernels.gpt_oss.transformer_gpt_oss import (
                get_multilayer_kernel_jit,
            )

            L = self._num_hidden_layers
            if self.layer_idx == 0:
                assert self._parent_model is not None
                assert kv_mgr is not None

                hidden_states = ModuleMarkerStartWrapper()(hidden_states)

                # gpt-oss uses one rotary; pass the same cos/sin to both slots.
                cos_cache, sin_cache = self.self_attn.rotary_emb(
                    hidden_states, position_ids
                )
                # NxDI emits [B, S, d_head] with the second half duplicated.
                # Kernel wants [d_head//2, B, S].
                d_head = self.config.head_dim
                cos_at_pos = cos_cache[..., : d_head // 2].permute(2, 0, 1).contiguous()
                sin_at_pos = sin_cache[..., : d_head // 2].permute(2, 0, 1).contiguous()

                (Wqkv_l, bqkv_l, Wo_l, bo_l, sinks_l,
                 gpre_l, gpost_l,
                 router_w_l, router_b_l,
                 gate_up_l, gate_up_b_l, down_l, down_b_l) = (
                    self._gather_weights_from_parent()
                )
                K_caches, V_caches = self._gather_kv_caches(kv_mgr)

                num_heads_per_rank = (
                    self.config.num_attention_heads
                    // self.config.neuron_config.tp_degree
                )
                s_tkg = hidden_states.shape[1]

                # Mask layout: [past_cache_mask | active_mask] with active=1 at
                # the last s_tkg slots, then [S_total, B, H, S_tkg]. Mirrors
                # stock NxDI's mask-prep (attention_base.py:1247-1263).
                # TODO(perf #2b): the .expand→.contiguous head broadcast and
                # .permute→.contiguous below still materialize per step. If
                # attention_block_tkg can consume a stride-0 head view we can
                # drop the materialization.
                def _prep_mask(m):
                    if m is None:
                        return None
                    m = m.expand(-1, num_heads_per_rank, -1, -1).contiguous()
                    # Scalar fill avoids allocating a torch.ones tensor every
                    # step. Caching the tensor on self breaks tracing — the
                    # cached XLA tensor escapes the trace boundary and trips
                    # convert_parameters_to_constants on a later pass.
                    m[:, :, :, -s_tkg:] = 1.0
                    return m.permute(3, 0, 1, 2).contiguous()

                mask_full = _prep_mask(attention_mask)
                mask_window = _prep_mask(local_mask) if local_mask is not None else mask_full

                kernel_out = get_multilayer_kernel_jit(L)[1](
                    hidden_states,
                    *Wqkv_l, *bqkv_l, *Wo_l, *bo_l,
                    *sinks_l,
                    *gpre_l, *gpost_l,
                    *router_w_l, *router_b_l,
                    *gate_up_l, *gate_up_b_l, *down_l, *down_b_l,
                    *K_caches, *V_caches,
                    cos_at_pos, sin_at_pos,
                    cos_at_pos, sin_at_pos,
                    mask_window, mask_full,
                    position_ids.to(torch.int32),
                    # SWA wraps the cache: index = position % (sliding_window-1).
                    (position_ids.to(torch.int32)
                     % (self.config.sliding_window - 1)),
                    replica_groups=self._replica_groups,
                )
                Y = kernel_out[0]
                K_out = list(kernel_out[1     : 1 + L])
                V_out = list(kernel_out[1 + L : 1 + 2 * L])

                Y = ModuleMarkerEndWrapper()(Y)

                self._parent_model._tkg_kv_out = (K_out, V_out)
                return (Y, (K_out[0], V_out[0]), cos_cache, sin_cache, None)

            K_out, V_out = self._parent_model._tkg_kv_out
            return (hidden_states,
                    (K_out[self.layer_idx], V_out[self.layer_idx]),
                    None, None, None)

        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            local_mask=local_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            **kwargs,
        )


class NeuronGptOssModelV1(NeuronGptOssModel):
    def init_model(self, config):
        super().init_model(config)
        # NxDI built `layers` with stock NeuronGptOssDecoderLayer; rebuild with
        # V1, sharing the rotary_cache_manager from the first stock layer.
        rotary_cache_manager = self.layers[0].rotary_cache_manager
        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList([
            NeuronGptOssDecoderLayerV1(conf, idx, rotary_cache_manager)
            for idx, conf in enumerate(updated_configs)
        ])
        for layer in self.layers:
            layer._parent_model_ref = weakref.ref(self)


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
