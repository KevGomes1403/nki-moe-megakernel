import os
import sys
from typing import Dict

import nki
import nki.isa as nisa
import nki.language as nl
import torch
import torch.nn as nn
from nkilib.core.utils.allocator import SbufManager

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from common import (  # noqa: E402
    B,
    D,
    H,
    HQ_TP,
    LNC,
    N_TILES,
    PMAX,
    S,
    capture_weight_store,
    make_config,
    tile_transpose,
    _scale_module_params,
)
from kernel_v14d_debug import (  # noqa: E402
    EPS,
    ROPE_INV_FREQ_VALUES,
    _attn_attention_oproj_sbuf_in_sbuf_out,
    _attn_qkv_rope_kvwrite_sbuf_in_sbuf_out,
    _attn_rmsnorm_sbuf_in_sbuf_out,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm  # noqa: E402
from qwen import NeuronQwen3MoEAttention  # noqa: E402


def capture_attn_weight_store(attn: NeuronQwen3MoEAttention) -> Dict[str, torch.Tensor]:
    return {
        "q_proj_weight": attn.qkv_proj.q_proj.weight.detach().bfloat16().cpu(),
        "k_proj_weight": attn.qkv_proj.k_proj.weight.detach().bfloat16().cpu(),
        "v_proj_weight": attn.qkv_proj.v_proj.weight.detach().bfloat16().cpu(),
        "o_proj_weight": attn.o_proj.o_proj.weight.detach().bfloat16().cpu(),
        "q_norm_weight": attn.q_layernorm.weight.detach().bfloat16().cpu(),
        "k_norm_weight": attn.k_layernorm.weight.detach().bfloat16().cpu(),
    }


def _load_bsh_hidden_to_sbuf(hidden_states, hidden_dim, batch, sbm):
    num_h_tiles = hidden_dim // PMAX
    hidden_col = hidden_states.reshape((hidden_dim, batch))
    hidden_sb = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="hidden_sb")
    nisa.dma_copy(
        dst=hidden_sb,
        src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )
    return hidden_sb


def _store_sbuf_to_bsh(output, out_sb, out_hidden_dim, batch, sbm):
    num_out_cols = out_hidden_dim // PMAX
    output_flat = output.reshape((batch, out_hidden_dim))
    for j in range(num_out_cols):
        col_psum = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(col_psum, out_sb[0:PMAX, j:j + 1])
        col_sb = sbm.alloc_stack((1, PMAX), nl.bfloat16, name=f"col_sb_{j}")
        nisa.tensor_copy(col_sb, col_psum)
        nisa.dma_copy(
            dst=output_flat[0:batch, j * PMAX:(j + 1) * PMAX],
            src=col_sb,
            dge_mode=nisa.dge_mode.hwdge,
        )


def _load_vector_weight_f32(weight, name, sbm):
    weight_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name=f"{name}_bf16")
    nisa.dma_copy(dst=weight_bf16, src=weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
    weight_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"{name}_f32")
    nisa.tensor_copy(weight_f32, weight_bf16)
    return weight_f32


def _load_gamma_f32(gamma, hidden_dim, sbm):
    num_h_tiles = hidden_dim // PMAX
    gamma_bf16 = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="gamma_bf16")
    nisa.dma_copy(
        dst=gamma_bf16,
        src=gamma.reshape((hidden_dim, 1)).ap(
            pattern=[[1, PMAX], [PMAX, num_h_tiles]],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )
    gamma_f32 = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="gamma_f32")
    nisa.tensor_copy(gamma_f32, gamma_bf16)
    return gamma_f32


def _alloc_rms_constants(sbm):
    rms_zero_bias = sbm.alloc_stack((PMAX, 1), nl.float32, name="rms_zero_bias")
    nisa.memset(rms_zero_bias, value=0.0)
    rms_ones = sbm.alloc_stack((PMAX, PMAX), nl.float32, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)
    rms_eps_sb = sbm.alloc_stack((PMAX, 1), nl.float32, name="rms_eps_sb")
    nisa.memset(rms_eps_sb, value=EPS)
    return rms_zero_bias, rms_ones, rms_eps_sb


def _load_position(position_ids, batch, sbm):
    pos_write_i32_raw = sbm.alloc_stack((1, 1), nl.int32, name="pos_write_i32_raw")
    nisa.dma_copy(
        dst=pos_write_i32_raw,
        src=position_ids.reshape((batch, 1))[0:1, 0:1],
        dge_mode=nisa.dge_mode.hwdge,
    )
    pos_write_i32 = sbm.alloc_stack((1, 1), nl.uint32, name="pos_write_i32")
    nisa.tensor_copy(pos_write_i32, pos_write_i32_raw)
    return pos_write_i32


def _reconstruct_rope(position_ids, batch, sbm):
    half_d = PMAX // 2
    pos_write_i32 = _load_position(position_ids, batch, sbm)
    rope_pos_scalar = sbm.alloc_stack((1, 1), nl.float32, name="rope_pos_scalar")
    nisa.tensor_copy(rope_pos_scalar, pos_write_i32)
    rope_pos_psum = nl.ndarray((PMAX, 1), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(rope_pos_psum, rope_pos_scalar.ap([[1, 1], [0, PMAX]], offset=0))
    rope_pos = sbm.alloc_stack((PMAX, 1), nl.float32, name="rope_pos")
    nisa.tensor_copy(rope_pos, rope_pos_psum)

    rope_inv_freq_T = sbm.alloc_stack((1, PMAX), nl.float32, name="rope_inv_freq_T")
    for rope_i in range(half_d):
        rope_value = ROPE_INV_FREQ_VALUES[rope_i]
        nisa.memset(rope_inv_freq_T[0:1, rope_i:rope_i + 1], value=rope_value)
        nisa.memset(rope_inv_freq_T[0:1, rope_i + half_d:rope_i + half_d + 1], value=rope_value)
    rope_inv_freq_psum = nl.ndarray((PMAX, 1), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(rope_inv_freq_psum, rope_inv_freq_T)
    rope_inv_freq = sbm.alloc_stack((PMAX, 1), nl.float32, name="rope_inv_freq")
    nisa.tensor_copy(rope_inv_freq, rope_inv_freq_psum)
    rope_angle = sbm.alloc_stack((PMAX, 1), nl.float32, name="rope_angle")
    nisa.tensor_tensor(rope_angle, rope_inv_freq, rope_pos, op=nl.multiply)
    return pos_write_i32, nl.cos(rope_angle), nl.sin(rope_angle)


def _owned_heads():
    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)
    if n_prgs == 1:
        return [0, 1, 2, 3, 4, 5, 6, 7]
    if prg_id == 0:
        return [0, 1, 2, 3]
    return [4, 5, 6, 7]


def _load_q_weights(Wq, owned_heads, sbm):
    wq_head_sb = [None] * HQ_TP
    for q_h in owned_heads:
        w = sbm.alloc_stack((PMAX, N_TILES * PMAX), nl.bfloat16, name=f"wq_head_{q_h}")
        nisa.dma_copy(
            dst=w,
            src=Wq[q_h * PMAX:(q_h + 1) * PMAX, :],
            dge_mode=nisa.dge_mode.hwdge,
        )
        wq_head_sb[q_h] = w
    return wq_head_sb


def _load_o_weights(Wo, owned_heads, out_hidden_dim, sbm):
    hq_tp = Wo.shape[0] // PMAX
    Wo_reshaped = Wo.reshape((hq_tp, PMAX, out_hidden_dim))
    wo_sbuf = [None] * HQ_TP
    for head in owned_heads:
        wo_tile = sbm.alloc_stack((PMAX, out_hidden_dim), nl.bfloat16, name=f"wo_tile_h{head}")
        nisa.dma_copy(
            dst=wo_tile,
            src=Wo_reshaped.ap(
                pattern=[[out_hidden_dim, PMAX], [1, out_hidden_dim]],
                offset=head * PMAX * out_hidden_dim,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )
        wo_sbuf[head] = wo_tile
    return wo_sbuf


def _load_cache_tiles(K_cache, V_cache, seq_len, sbm):
    K_cache_2d = K_cache.reshape((seq_len, PMAX))
    V_cache_2d = V_cache.reshape((seq_len, PMAX))
    k_cache_tiles_hbm = []
    v_cache_tiles_hbm = []
    for s_t in nl.affine_range(seq_len // PMAX):
        k_ct = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name=f"k_ct_{s_t}")
        nisa.dma_transpose(
            dst=k_ct.ap([[PMAX, PMAX], [1, 1], [1, 1], [1, PMAX]], offset=0),
            src=K_cache_2d.ap([[PMAX, PMAX], [1, 1], [1, 1], [1, PMAX]], offset=s_t * PMAX * PMAX),
        )
        k_cache_tiles_hbm.append(k_ct)

        v_ct = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name=f"v_ct_{s_t}")
        nisa.dma_copy(
            dst=v_ct,
            src=V_cache_2d.ap(pattern=[[PMAX, PMAX], [1, PMAX]], offset=s_t * PMAX * PMAX),
            dge_mode=nisa.dge_mode.hwdge,
        )
        v_cache_tiles_hbm.append(v_ct)
    return k_cache_tiles_hbm, v_cache_tiles_hbm


def _rmsnorm_stage_raw(hidden_states, gamma, output):
    batch = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[2]
    num_h_tiles = hidden_dim // PMAX
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("rmsnorm_stage_wrapper")

    hidden_sb = _load_bsh_hidden_to_sbuf(hidden_states, hidden_dim, batch, sbm)
    gamma_f32 = _load_gamma_f32(gamma, hidden_dim, sbm)
    rms_zero_bias, _, rms_eps_sb = _alloc_rms_constants(sbm)
    out_sb = _attn_rmsnorm_sbuf_in_sbuf_out(
        hidden_sb,
        gamma_f32,
        rms_zero_bias,
        rms_eps_sb,
        hidden_dim,
        num_h_tiles,
        sbm=sbm,
    )
    _store_sbuf_to_bsh(output, out_sb, hidden_dim, batch, sbm)

    sbm.close_scope()
    return output


def _qkv_rope_stage_raw(
    normed_states,
    Wq,
    Wk,
    Wv,
    q_norm_weight,
    k_norm_weight,
    K_cache,
    V_cache,
    position_ids,
    q_output,
    k_output,
    v_output,
):
    batch = normed_states.shape[0]
    hidden_dim = Wq.shape[1]
    hq_tp = Wq.shape[0] // PMAX
    seq_len = K_cache.shape[2]

    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("qkv_rope_stage_wrapper")

    h_all = _load_bsh_hidden_to_sbuf(normed_states, hidden_dim, batch, sbm)
    qnw_f32 = _load_vector_weight_f32(q_norm_weight, "qnw", sbm)
    knw_f32 = _load_vector_weight_f32(k_norm_weight, "knw", sbm)
    pos_write_i32, cos_f32, sin_f32 = _reconstruct_rope(position_ids, batch, sbm)
    rms_zero_bias, rms_ones, rms_eps_sb = _alloc_rms_constants(sbm)

    wk_sb = sbm.alloc_stack((PMAX, N_TILES * PMAX), nl.bfloat16, name="wk_sb")
    nisa.dma_copy(dst=wk_sb, src=Wk, dge_mode=nisa.dge_mode.hwdge)
    wv_sb = sbm.alloc_stack((PMAX, N_TILES * PMAX), nl.bfloat16, name="wv_sb")
    nisa.dma_copy(dst=wv_sb, src=Wv, dge_mode=nisa.dge_mode.hwdge)

    owned_heads = _owned_heads()
    wq_head_sb = _load_q_weights(Wq, owned_heads, sbm)

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
        hidden_dim,
        hq_tp,
        seq_len,
        owned_heads,
        sbm=sbm,
    )

    q_out_2d = q_output.reshape((hq_tp, PMAX))
    for head in owned_heads:
        q_head_psum = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(q_head_psum, q_bf16[0:PMAX, head:head + 1])
        q_head_sb = sbm.alloc_stack((1, PMAX), nl.bfloat16, name=f"q_head_out_{head}")
        nisa.tensor_copy(q_head_sb, q_head_psum)
        nisa.dma_copy(dst=q_out_2d[head:head + 1, 0:PMAX], src=q_head_sb, dge_mode=nisa.dge_mode.hwdge)

    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)
    if n_prgs == 1 or prg_id == 1:
        k_psum = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(k_psum, k_rope_bf16)
        k_sb = sbm.alloc_stack((1, PMAX), nl.bfloat16, name="k_out_sb")
        nisa.tensor_copy(k_sb, k_psum)
        nisa.dma_copy(dst=k_output.reshape((1, PMAX)), src=k_sb, dge_mode=nisa.dge_mode.hwdge)
    if n_prgs == 1 or prg_id == 0:
        v_bf16 = sbm.alloc_stack((PMAX, batch), nl.bfloat16, name="v_out_bf16")
        nisa.tensor_copy(v_bf16, v_active)
        v_psum = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(v_psum, v_bf16)
        v_sb = sbm.alloc_stack((1, PMAX), nl.bfloat16, name="v_out_sb")
        nisa.tensor_copy(v_sb, v_psum)
        nisa.dma_copy(dst=v_output.reshape((1, PMAX)), src=v_sb, dge_mode=nisa.dge_mode.hwdge)

    sbm.close_scope()
    return q_output, k_output, v_output, K_cache, V_cache


def _attention_oproj_stage_raw(
    q_input,
    k_input,
    v_input,
    Wo,
    K_cache,
    V_cache,
    position_ids,
    output,
):
    batch = q_input.shape[0]
    hq_tp = q_input.shape[1]
    out_hidden_dim = Wo.shape[1]
    seq_len = K_cache.shape[2]

    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("attention_oproj_stage_wrapper")

    q_col = q_input.reshape((hq_tp * PMAX, batch))
    q_bf16 = sbm.alloc_stack((PMAX, hq_tp), nl.bfloat16, name="q_bf16")
    nisa.dma_copy(
        dst=q_bf16,
        src=q_col.ap(pattern=[[1, PMAX], [PMAX, hq_tp]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )

    k_rope_bf16 = sbm.alloc_stack((PMAX, batch), nl.bfloat16, name="k_rope_bf16")
    nisa.dma_copy(dst=k_rope_bf16, src=k_input.reshape((PMAX, batch)), dge_mode=nisa.dge_mode.hwdge)

    v_bf16 = sbm.alloc_stack((PMAX, batch), nl.bfloat16, name="v_bf16")
    nisa.dma_copy(dst=v_bf16, src=v_input.reshape((PMAX, batch)), dge_mode=nisa.dge_mode.hwdge)
    v_active = sbm.alloc_stack((PMAX, batch), nl.float32, name="v_active")
    nisa.tensor_copy(v_active, v_bf16)

    pos_write_i32 = _load_position(position_ids, batch, sbm)
    rms_ones = sbm.alloc_stack((PMAX, PMAX), nl.float32, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)
    k_cache_tiles_hbm, v_cache_tiles_hbm = _load_cache_tiles(K_cache, V_cache, seq_len, sbm)

    owned_heads = _owned_heads()
    wo_sbuf = _load_o_weights(Wo, owned_heads, out_hidden_dim, sbm)
    out_sb = _attn_attention_oproj_sbuf_in_sbuf_out(
        q_bf16,
        k_rope_bf16,
        v_active,
        k_cache_tiles_hbm,
        v_cache_tiles_hbm,
        wo_sbuf,
        rms_ones,
        pos_write_i32,
        out_hidden_dim,
        owned_heads,
        sbm=sbm,
    )
    _store_sbuf_to_bsh(output, out_sb, out_hidden_dim, batch, sbm)

    sbm.close_scope()
    return output


class RefRmsNormStageModule(nn.Module):
    def __init__(self, config, seed=42, weight_scale=1.0, _weight_store=None):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.input_layernorm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype)
        _scale_module_params(self, weight_scale)
        if _weight_store is not None and not _weight_store and self.input_layernorm.weight.device.type != "meta":
            _weight_store["input_layernorm_weight"] = self.input_layernorm.weight.detach().bfloat16().cpu()

    def forward(self, hidden_states):
        return self.input_layernorm(hidden_states)


class KernelRmsNormStageModule(nn.Module):
    def __init__(self, config, seed=42, weight_scale=1.0, _weight_store=None):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.input_layernorm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype)
        _scale_module_params(self, weight_scale)
        self._kernel = nki.jit(_rmsnorm_stage_raw)
        if _weight_store is not None and not _weight_store and self.input_layernorm.weight.device.type != "meta":
            _weight_store["input_layernorm_weight"] = self.input_layernorm.weight.detach().bfloat16().cpu()

    def forward(self, hidden_states):
        output = torch.zeros_like(hidden_states)
        return self._kernel[LNC](hidden_states, self.input_layernorm.weight, output)


class RefQkvRopeStageModule(nn.Module):
    def __init__(self, config, seed=42, weight_scale=1.0, _weight_store=None):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.attn = NeuronQwen3MoEAttention(config).to(dtype)
        _scale_module_params(self, weight_scale)
        if _weight_store is not None and not _weight_store and self.attn.qkv_proj.q_proj.weight.device.type != "meta":
            _weight_store.update(capture_attn_weight_store(self.attn))

    def forward(self, normed_states, position_ids, K_cache, V_cache):
        Q, K, V, _, _, _ = self.attn.prep_qkv_tensors(
            position_ids,
            normed_states,
            past_key_value=(K_cache, V_cache),
        )
        bsz, n_kv, _, d = K_cache.shape
        K_new = K.reshape(bsz, n_kv, 1, d).to(K_cache.device)
        V_new = V.reshape(bsz, n_kv, 1, d).to(V_cache.device)
        cache_idx = position_ids.to(torch.long).reshape(bsz, 1, 1, 1).expand(bsz, n_kv, 1, d)
        return (
            Q.squeeze(2),
            K.squeeze(2),
            V.squeeze(2),
            K_cache.scatter(2, cache_idx, K_new),
            V_cache.scatter(2, cache_idx, V_new),
        )


class KernelQkvRopeStageModule(nn.Module):
    def __init__(self, config, seed=42, weight_scale=1.0, _weight_store=None):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.attn = NeuronQwen3MoEAttention(config).to(dtype)
        _scale_module_params(self, weight_scale)
        self._kernel = nki.jit(_qkv_rope_stage_raw)
        if _weight_store is not None and not _weight_store and self.attn.qkv_proj.q_proj.weight.device.type != "meta":
            _weight_store.update(capture_attn_weight_store(self.attn))

    def forward(self, normed_states, position_ids, K_cache, V_cache):
        Wq_tt = tile_transpose(self.attn.qkv_proj.q_proj.weight, HQ_TP, D, N_TILES)
        Wk_tt = tile_transpose(self.attn.qkv_proj.k_proj.weight, 1, D, N_TILES)
        Wv_tt = tile_transpose(self.attn.qkv_proj.v_proj.weight, 1, D, N_TILES)
        q_output = torch.zeros((B, HQ_TP, D), dtype=normed_states.dtype, device=normed_states.device)
        k_output = torch.zeros((B, 1, D), dtype=normed_states.dtype, device=normed_states.device)
        v_output = torch.zeros((B, 1, D), dtype=normed_states.dtype, device=normed_states.device)
        return self._kernel[LNC](
            normed_states,
            Wq_tt,
            Wk_tt,
            Wv_tt,
            self.attn.q_layernorm.weight,
            self.attn.k_layernorm.weight,
            K_cache,
            V_cache,
            position_ids.to(torch.int32),
            q_output,
            k_output,
            v_output,
        )


class RefAttentionOProjStageModule(nn.Module):
    def __init__(self, config, seed=42, weight_scale=1.0, _weight_store=None):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.attn = NeuronQwen3MoEAttention(config).to(dtype)
        _scale_module_params(self, weight_scale)
        if _weight_store is not None and not _weight_store and self.attn.qkv_proj.q_proj.weight.device.type != "meta":
            _weight_store.update(capture_attn_weight_store(self.attn))

    def forward(self, q_input, k_input, v_input, attention_mask, position_ids, K_cache, V_cache):
        Q = q_input.reshape(B, HQ_TP, 1, D)
        K = k_input.reshape(B, 1, 1, D)
        V = v_input.reshape(B, 1, 1, D)
        attn_output = self.attn.compute_for_token_gen(
            Q,
            K,
            V,
            position_ids,
            (K_cache, V_cache),
            attention_mask,
            active_mask=None,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, 1, HQ_TP * D)
        return self.attn.get_o_proj()(attn_output)


class KernelAttentionOProjStageModule(nn.Module):
    def __init__(self, config, seed=42, weight_scale=1.0, _weight_store=None):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.attn = NeuronQwen3MoEAttention(config).to(dtype)
        _scale_module_params(self, weight_scale)
        self._kernel = nki.jit(_attention_oproj_stage_raw)
        if _weight_store is not None and not _weight_store and self.attn.qkv_proj.q_proj.weight.device.type != "meta":
            _weight_store.update(capture_attn_weight_store(self.attn))

    def forward(self, q_input, k_input, v_input, attention_mask, position_ids, K_cache, V_cache):
        del attention_mask
        Wo = self.attn.o_proj.o_proj.weight.T.contiguous()
        output = torch.zeros((B, 1, H), dtype=q_input.dtype, device=q_input.device)
        return self._kernel[LNC](
            q_input,
            k_input,
            v_input,
            Wo,
            K_cache,
            V_cache,
            position_ids.to(torch.int32),
            output,
        )


def make_tp1_config():
    cfg = make_config()
    cfg.neuron_config.tp_degree = 1
    cfg.num_attention_heads = HQ_TP
    cfg.num_key_value_heads = 1
    return cfg
