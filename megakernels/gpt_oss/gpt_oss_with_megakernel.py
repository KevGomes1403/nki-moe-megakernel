# coding=utf-8
"""GPT-OSS-20B with the 24-layer fused TKG megakernel.

CTE: stock per-layer NxDI path.
TKG: `NeuronGptOssModelV2.get_model_output` bypasses NxDI's per-layer loop and
calls the megakernel once for all 24 layers. KV cache updates are in-place via
the kernel's scatter.
"""

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

    # Bake the per-step transposes / bias-divide that the model-level gather
    # would otherwise do every TKG forward. New keys load into the
    # NeuronGptOssDecoderLayerV1 weight params registered below.
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
    # read it from the model gather.

    return state_dict


class NeuronGptOssDecoderLayerV1(NeuronGptOssDecoderLayer):
    """Stock decoder layer + extra kernel-layout weight params.

    The megakernel needs Wqkv / Wo in a different layout than NxDI's defaults
    and Wo bias pre-divided by tp_degree. Register the kernel-layout params on
    the layer so `convert_gpt_oss_hf_to_neuron_state_dict` can route values
    into them through the state-dict loader. No `forward` override: the V2
    model bypasses the per-layer loop entirely on TKG, and CTE uses the stock
    `NeuronGptOssDecoderLayer.forward`.
    """

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


class NeuronGptOssModelV1(NeuronGptOssModel):
    """Overrides `get_model_output` to bypass the per-layer loop on TKG.

    CTE path delegates to `super().get_model_output(...)` unchanged.

    TKG path: embed → mask + RoPE prep → single megakernel call (all 24 layers
    fused) → final RMSNorm → return `(hidden_states, flat_kv_list)`. The
    megakernel scatters KV in-place into `self.kv_mgr.past_key_values`, so the
    returned cache list is just references to the (now-updated) buffer
    tensors — matching the `update_kv_per_layer=True` convention NxDI's loop
    builds incrementally.
    """

    def init_model(self, config):
        super().init_model(config)
        # NxDI built `layers` with stock NeuronGptOssDecoderLayer; rebuild with
        # V1 layers (which carry the kernel-layout weight params), sharing the
        # rotary_cache_manager from the first stock layer.
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
        """Gather per-layer weights, reshaped/transposed to kernel layouts.

        Re-gathered every TKG call. Caching across calls breaks tracing:
        nn.Parameter `.data` gets re-wrapped as a per-trace XLA tensor, so a
        cached ref from an earlier trace trips `convert_parameters_to_constants`
        ("XLA data... while an async operation is in flight"). The gather
        itself is cheap — the .T.contiguous() / bias-divide are baked into
        the state dict by `convert_gpt_oss_hf_to_neuron_state_dict`.
        """
        cfg = self.config
        H_padded = cfg.hidden_size
        I_per_rank = cfg.intermediate_size // cfg.neuron_config.moe_tp_degree
        E_count = cfg.num_local_experts

        Wqkv, bqkv, Wo, bo = [], [], [], []
        sinks = []
        gpre, gpost = [], []
        router_w, router_b = [], []
        gate_up, gate_up_b, down, down_b = [], [], [], []

        for l in self.layers:
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

        # TKG fast path: embed → megakernel → norm → return.
        from megakernels.gpt_oss.transformer_gpt_oss import (
            get_multilayer_kernel_jit,
        )

        if inputs_embeds is None:
            inputs_embeds = (
                self.embed_tokens(input_ids)
                if not is_lora_module(self.embed_tokens)
                else self.embed_tokens(input_ids, adapter_ids=adapter_ids)
            )
        # TKG never runs with sequence parallelism, so the SP scatter from
        # NxDI's get_model_output is a no-op here.
        hidden_states = inputs_embeds

        hidden_states = ModuleMarkerStartWrapper()(hidden_states)

        # gpt-oss uses one rotary; pass the un-permuted half-d slice and let
        # the kernel do the (B, S, d//2) -> (d//2, B, S) permute via a strided
        # DMA on load (Phase 4). NxDI emits [B, S, d_head] with the second
        # half duplicated; we only need the first half.
        cos_cache, sin_cache = self.layers[0].self_attn.rotary_emb(
            hidden_states, position_ids
        )
        d_head = self.config.head_dim
        cos_unperm = cos_cache[..., : d_head // 2].contiguous()
        sin_unperm = sin_cache[..., : d_head // 2].contiguous()

        # Phase 4: attention_mask / local_attn_mask are no longer passed —
        # the kernel derives both masks from position_ids via iota+tensor_scalar.

        (Wqkv_l, bqkv_l, Wo_l, bo_l, sinks_l,
         gpre_l, gpost_l,
         router_w_l, router_b_l,
         gate_up_l, gate_up_b_l, down_l, down_b_l) = self._gather_weights()
        K_caches, V_caches = self._gather_kv_caches()

        L = self._num_hidden_layers
        kernel_out = get_multilayer_kernel_jit(L)[1](
            hidden_states,
            *Wqkv_l, *bqkv_l, *Wo_l, *bo_l,
            *sinks_l,
            *gpre_l, *gpost_l,
            *router_w_l, *router_b_l,
            *gate_up_l, *gate_up_b_l, *down_l, *down_b_l,
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

        # NxDI's loop builds next_decoder_cache by `next_decoder_cache += kv`
        # per layer when `update_kv_per_layer=True` (set because we have
        # attn_block_tkg_nki_kernel_cache_update=True and is_for_context_encoding=False).
        # Result is a flat list [K0, V0, K1, V1, ...] — reproduce the same.
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
