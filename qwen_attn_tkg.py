# coding=utf-8
"""Qwen3 MoE model with token generation routed through v14d Attn TKG.

Everything outside the attention module follows qwen.py.  Context encoding uses the
standard NxDI attention path; token generation calls the local attention kernel
directly instead of relying on NxDI's attn_block_tkg_nki_kernel_enabled dispatch.
"""

import warnings
from typing import Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch import nn

from qwen import *  # noqa: F401,F403 - re-export the baseline qwen.py surface.

from kernels.attn_tkg.attn_fused_nki import (
    PMAX as V14D_PMAX,
    qwen3_attn_tkg_fused_oproj_v13bc_kv_norm_hoisted_weights,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region_with_dim,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    reduce_scatter_to_tensor_model_parallel_region_with_dim,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    EPDispatchOption,
)
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
    KV_CACHE_PAD_FOR_SEQ_IDS_MASKING,
)
from neuronx_distributed_inference.modules.attention.attention_process_groups import (
    get_data_parallel_attention_dp_group,
)
from nkilib.core.utils.allocator import create_auto_alloc_manager
from nkilib.core.utils.logging import Logger


@nki.jit
def _qwen3_attn_tkg_v14d_kernel(
    hidden_states: nl.ndarray,
    Wq: nl.ndarray,
    Wk: nl.ndarray,
    Wv: nl.ndarray,
    Wo: nl.ndarray,
    q_norm_weight: nl.ndarray,
    k_norm_weight: nl.ndarray,
    gamma_pre_attn: nl.ndarray,
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    cos: nl.ndarray,
    sin: nl.ndarray,
    position_ids: nl.ndarray,
) -> nl.ndarray:
    """HBM-facing wrapper around the SBUF-resident v14d attention subkernel."""

    hidden_size = Wq.shape[1]
    hidden_tiles = hidden_size // V14D_PMAX
    out_hidden_size = Wo.shape[1]
    out_hidden_tiles = out_hidden_size // V14D_PMAX

    sbm = create_auto_alloc_manager(logger=Logger("qwen-attn-tkg-v14d"))
    sbm.open_scope(name="qwen_attn_tkg_v14d_wrapper")
    hidden_sb = sbm.alloc_stack(
        (V14D_PMAX, hidden_tiles),
        dtype=hidden_states.dtype,
        buffer=nl.sbuf,
        name="hidden_sb",
    )
    nisa.dma_copy(
        dst=hidden_sb,
        src=hidden_states.reshape((hidden_size, 1)).ap(
            pattern=[[1, V14D_PMAX], [V14D_PMAX, hidden_tiles]],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )
    out_sb = sbm.alloc_stack(
        (V14D_PMAX, out_hidden_tiles),
        dtype=hidden_states.dtype,
        buffer=nl.sbuf,
        name="out_sb",
    )

    qwen3_attn_tkg_fused_oproj_v13bc_kv_norm_hoisted_weights(
        hidden_sb=hidden_sb,
        Wq=Wq,
        Wk=Wk,
        Wv=Wv,
        Wo=Wo,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        gamma_pre_attn=gamma_pre_attn,
        K_cache=K_cache,
        V_cache=V_cache,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        out_sb=out_sb,
        sbm=sbm,
    )

    out = nl.ndarray(
        (1, 1, out_hidden_size),
        dtype=hidden_states.dtype,
        buffer=nl.shared_hbm,
        name="qwen_attn_tkg_v14d_out",
    )
    nisa.dma_copy(
        dst=out.reshape((out_hidden_size, 1)).ap(
            pattern=[[1, V14D_PMAX], [V14D_PMAX, out_hidden_tiles]],
            offset=0,
        ),
        src=out_sb,
    )
    sbm.close_scope()  # attn_outer scope intentionally left open by the subkernel.
    sbm.close_scope()  # qwen_attn_tkg_v14d_wrapper
    return out


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


def _attention_tkg_v14d(
    attn,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_value: Tuple[torch.Tensor],
    rmsnorm: nn.Module,
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
    Wo = attn.get_o_proj().o_proj.weight.data
    cos, sin, cos_cache, sin_cache = _get_v14d_rope(
        attn,
        hidden_states,
        rotary_position_ids,
        cos_cache,
        sin_cache,
    )

    attn_output = _qwen3_attn_tkg_v14d_kernel[attn.logical_nc_config](
        hidden_states=hidden_states.to(attn.torch_dtype),
        Wq=Wq,
        Wk=Wk,
        Wv=Wv,
        Wo=Wo,
        q_norm_weight=attn.q_layernorm.weight.data,
        k_norm_weight=attn.k_layernorm.weight.data,
        gamma_pre_attn=rmsnorm.weight.data,
        K_cache=K_prior.data,
        V_cache=V_prior.data,
        cos=cos,
        sin=sin,
        position_ids=position_ids.to(torch.int32),
    )

    if attn.sequence_parallel_enabled:
        attn_output = reduce_scatter_to_sequence_parallel_region(
            attn_output,
            1,
            process_group=attn.tensor_model_parallel_group,
        )
    else:
        if attn.ep_dispatch_cc_option == EPDispatchOption.AR_AG:
            attn_output = reduce_from_tensor_model_parallel_region(
                attn_output,
                process_group=attn.tensor_model_parallel_group,
            )
        elif attn.ep_dispatch_cc_option == EPDispatchOption.RS_AG:
            attn_output = reduce_scatter_to_tensor_model_parallel_region_with_dim(
                attn_output,
                partition_dim=0,
                process_group=attn.tensor_model_parallel_group,
            )
        elif attn.ep_dispatch_cc_option == EPDispatchOption.AG_AR:
            attn_output = gather_from_tensor_model_parallel_region_with_dim(
                attn_output,
                gather_dim=0,
                process_group=get_data_parallel_attention_dp_group(),
            )
        else:
            raise ValueError(f"Unknown EPDispatchOption: {attn.ep_dispatch_cc_option}")

    return attn_output, (K_prior, V_prior), cos_cache, sin_cache


class NeuronQwen3MoeDecoderLayer(nn.Module):
    """Decoder layer copied from qwen.py, using the local attention module only."""

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
                position_ids=tkg_position_ids,
                past_key_value=tkg_past_key_value,
                rmsnorm=self.input_layernorm,
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
    """NeuronQwen3MoeModel copied from qwen.py with the local decoder layer."""

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
    """Qwen3 MoE CausalLM copied from qwen.py with the local model class."""

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


def generate(skip_compile=False):
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
