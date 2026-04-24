import math
from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

import ml_dtypes
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch.nn as nn
from nkilib.core.utils.allocator import SbufManager

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from kernel_v14d_debug import (
    qwen3_attn_tkg_fused_oproj_v13bc_kv_norm_hoisted_weights as _v14d_kernel,
)
from qwen import Qwen3MoeInferenceConfig, NeuronQwen3MoEAttention


MODEL_PATH = "/home/ubuntu/models/Qwen3-30B-A3B"

PMAX = 128
B = 1
H = 2048
D = 128
HQ_TP = 8
S = 640
N_TILES = H // D
LNC = 2
TP_DEGREE = 4
MOE_TP_DEGREE = 1

DEFAULT_POSITIONS = [0, 1, 2, 63, 64, 127, 128, 255, 319, 511, 639]
DEFAULT_HIDDEN_SCALES = [0.1, 0.5, 1.0, 3.0]
DEFAULT_CACHE_SCALES = [0.1, 1.0, 3.0]
DEFAULT_DISTS = ["normal", "student-t5", "laplace"]

COMPILER_ARGS = (
    "--enable-saturate-infinity --enable-mixed-precision-accumulation "
    "--model-type transformer -O1 "
    "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
)

WEIGHT_KEYS = (
    "input_layernorm_weight",
    "q_proj_weight",
    "k_proj_weight",
    "v_proj_weight",
    "o_proj_weight",
    "q_norm_weight",
    "k_norm_weight",
)


def make_config() -> Qwen3MoeInferenceConfig:
    # Mirror main_new.py defaults as closely as possible for the token-generation
    # attention path, except that the test harness intentionally forces LNC=2.
    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        moe_tp_degree=MOE_TP_DEGREE,
        batch_size=1,
        seq_len=S,
        flash_decoding_enabled=False,
        logical_nc_config=LNC,
        enable_bucketing=True,
        fused_qkv=False,
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
        mlp_kernel_enabled=False,
        attn_tkg_nki_kernel_enabled=False,
        attn_block_tkg_nki_kernel_enabled=False,
    )
    cfg = Qwen3MoeInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )
    return cfg


def tile_transpose(weight: torch.Tensor, n_heads: int, d_dim: int, n_tiles: int) -> torch.Tensor:
    return (
        weight.reshape(n_heads, d_dim, n_tiles, d_dim)
        .permute(0, 3, 2, 1)
        .reshape(n_heads * d_dim, n_tiles * d_dim)
        .contiguous()
    )


def _scale_module_params(module: nn.Module, weight_scale: float) -> None:
    if weight_scale == 1.0:
        return
    with torch.no_grad():
        for param in module.parameters():
            if param.is_floating_point():
                param.mul_(weight_scale)


def capture_weight_store(input_layernorm: nn.Module, attn: NeuronQwen3MoEAttention) -> Dict[str, torch.Tensor]:
    return {
        "input_layernorm_weight": input_layernorm.weight.detach().bfloat16().cpu(),
        "q_proj_weight": attn.qkv_proj.q_proj.weight.detach().bfloat16().cpu(),
        "k_proj_weight": attn.qkv_proj.k_proj.weight.detach().bfloat16().cpu(),
        "v_proj_weight": attn.qkv_proj.v_proj.weight.detach().bfloat16().cpu(),
        "o_proj_weight": attn.o_proj.o_proj.weight.detach().bfloat16().cpu(),
        "q_norm_weight": attn.q_layernorm.weight.detach().bfloat16().cpu(),
        "k_norm_weight": attn.k_layernorm.weight.detach().bfloat16().cpu(),
    }


def compare_weight_stores(
    ref_weight_store: Dict[str, torch.Tensor],
    kern_weight_store: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    diffs = {}
    for key in WEIGHT_KEYS:
        diffs[key] = (
            ref_weight_store[key].float() - kern_weight_store[key].float()
        ).abs().max().item()
    return diffs


def build_rotary_emb(config: Qwen3MoeInferenceConfig) -> RotaryEmbedding:
    return RotaryEmbedding(
        config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    )


def get_cos_sin_at_pos(
    rotary_emb: RotaryEmbedding,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        cos_full, sin_full = rotary_emb(hidden_states, position_ids)
    return cos_full.squeeze(1).bfloat16(), sin_full.squeeze(1).bfloat16()


class RefAttnModule(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeInferenceConfig,
        seed: int = 42,
        weight_scale: float = 1.0,
        _weight_store: Dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.input_layernorm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype)
        self.attn = NeuronQwen3MoEAttention(config).to(dtype)
        _scale_module_params(self, weight_scale)
        if _weight_store is not None and not _weight_store and self.input_layernorm.weight.device.type != "meta":
            _weight_store.update(capture_weight_store(self.input_layernorm, self.attn))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
    ):
        normed = self.input_layernorm(hidden_states)
        hidden_out, kv, _, _ = self.attn(
            hidden_states=normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=(K_cache, V_cache),
        )
        return hidden_out, kv[0], kv[1]


def _v14d_attn_raw(
    hidden_states,
    Wq,
    Wk,
    Wv,
    Wo,
    q_norm_weight,
    k_norm_weight,
    gamma_pre_attn,
    K_cache,
    V_cache,
    cos,
    sin,
    position_ids,
    output,
):
    batch = cos.shape[0]
    hidden_dim = Wq.shape[1]
    out_hidden_dim = Wo.shape[1]
    num_h_tiles = hidden_dim // PMAX
    num_out_cols = out_hidden_dim // PMAX
    hq_tp_local = Wq.shape[0] // PMAX

    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)
    if n_prgs == 1:
        owned_heads = [0, 1, 2, 3, 4, 5, 6, 7]
    elif prg_id == 0:
        owned_heads = [0, 1, 2, 3]
    else:
        owned_heads = [4, 5, 6, 7]

    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("wrapper")

    hidden_col = hidden_states.reshape((hidden_dim, batch))
    hidden_sb = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="hidden_sb")
    nisa.dma_copy(
        dst=hidden_sb,
        src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )

    cos_col = cos.reshape((PMAX, batch))
    sin_col = sin.reshape((PMAX, batch))

    qnw_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="qnw_bf16")
    nisa.dma_copy(dst=qnw_bf16, src=q_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
    qnw_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name="qnw_f32")
    nisa.tensor_copy(qnw_f32, qnw_bf16)

    knw_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="knw_bf16")
    nisa.dma_copy(dst=knw_bf16, src=k_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
    knw_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name="knw_f32")
    nisa.tensor_copy(knw_f32, knw_bf16)

    cos_bf16 = sbm.alloc_stack((PMAX, batch), nl.bfloat16, name="cos_bf16")
    nisa.dma_copy(dst=cos_bf16, src=cos_col, dge_mode=nisa.dge_mode.hwdge)
    cos_f32 = sbm.alloc_stack((PMAX, batch), nl.float32, name="cos_f32")
    nisa.tensor_copy(cos_f32, cos_bf16)

    sin_bf16 = sbm.alloc_stack((PMAX, batch), nl.bfloat16, name="sin_bf16")
    nisa.dma_copy(dst=sin_bf16, src=sin_col, dge_mode=nisa.dge_mode.hwdge)
    sin_f32 = sbm.alloc_stack((PMAX, batch), nl.float32, name="sin_f32")
    nisa.tensor_copy(sin_f32, sin_bf16)

    gpan_bf16 = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="gpan_bf16")
    nisa.dma_copy(
        dst=gpan_bf16,
        src=gamma_pre_attn.reshape((hidden_dim, 1)).ap(
            pattern=[[1, PMAX], [PMAX, num_h_tiles]],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )
    gpan_f32 = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="gpan_f32")
    nisa.tensor_copy(gpan_f32, gpan_bf16)

    wk_sb = sbm.alloc_stack((PMAX, num_h_tiles * PMAX), nl.bfloat16, name="wk_sb")
    nisa.dma_copy(dst=wk_sb, src=Wk, dge_mode=nisa.dge_mode.hwdge)

    wv_sb = sbm.alloc_stack((PMAX, num_h_tiles * PMAX), nl.bfloat16, name="wv_sb")
    nisa.dma_copy(dst=wv_sb, src=Wv, dge_mode=nisa.dge_mode.hwdge)

    wq_heads = []
    for q_head in owned_heads:
        w = sbm.alloc_stack((PMAX, num_h_tiles * PMAX), nl.bfloat16, name=f"wq_h_{q_head}")
        nisa.dma_copy(dst=w, src=Wq[q_head * PMAX:(q_head + 1) * PMAX, :], dge_mode=nisa.dge_mode.hwdge)
        wq_heads.append(w)

    wo_heads = []
    for q_head in owned_heads:
        wo_tile = sbm.alloc_stack((PMAX, out_hidden_dim), nl.bfloat16, name=f"wo_h_{q_head}")
        nisa.dma_copy(
            dst=wo_tile,
            src=Wo.reshape((hq_tp_local, PMAX, out_hidden_dim)).ap(
                pattern=[[out_hidden_dim, PMAX], [1, out_hidden_dim]],
                offset=q_head * PMAX * out_hidden_dim,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )
        wo_heads.append(wo_tile)

    out_sb = _v14d_kernel(
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
        sbm=sbm,
        qnw_f32_sb=qnw_f32,
        knw_f32_sb=knw_f32,
        cos_f32_sb=cos_f32,
        sin_f32_sb=sin_f32,
        gpan_f32_sb=gpan_f32,
        wk_sb=wk_sb,
        wv_sb=wv_sb,
        wq_heads_sb=wq_heads,
        wo_heads_sb=wo_heads,
    )

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

    sbm.close_scope()
    sbm.close_scope()
    return output, K_cache, V_cache


class KernelAttnModule(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeInferenceConfig,
        seed: int = 42,
        weight_scale: float = 1.0,
        _weight_store: Dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.input_layernorm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype)
        self.attn = NeuronQwen3MoEAttention(config).to(dtype)
        _scale_module_params(self, weight_scale)
        self._attn_jit = nki.jit(_v14d_attn_raw)
        if _weight_store is not None and not _weight_store and self.input_layernorm.weight.device.type != "meta":
            _weight_store.update(capture_weight_store(self.input_layernorm, self.attn))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
    ):
        del attention_mask
        Wq_raw = self.attn.qkv_proj.q_proj.weight
        Wk_raw = self.attn.qkv_proj.k_proj.weight
        Wv_raw = self.attn.qkv_proj.v_proj.weight
        Wo_raw = self.attn.o_proj.o_proj.weight
        q_norm = self.attn.q_layernorm.weight
        k_norm = self.attn.k_layernorm.weight
        gamma_pre_attn = self.input_layernorm.weight

        Wq_tt = tile_transpose(Wq_raw, HQ_TP, D, N_TILES)
        Wk_tt = tile_transpose(Wk_raw, 1, D, N_TILES)
        Wv_tt = tile_transpose(Wv_raw, 1, D, N_TILES)
        Wo_k = Wo_raw.T.contiguous()

        cos_at_pos, sin_at_pos = get_cos_sin_at_pos(self.attn.rotary_emb, hidden_states, position_ids)
        pos_i32 = position_ids.to(torch.int32)
        output = torch.zeros_like(hidden_states)

        out, K_out, V_out = self._attn_jit[LNC](
            hidden_states,
            Wq_tt,
            Wk_tt,
            Wv_tt,
            Wo_k,
            q_norm,
            k_norm,
            gamma_pre_attn,
            K_cache,
            V_cache,
            cos_at_pos,
            sin_at_pos,
            pos_i32,
            output,
        )
        return out, K_out, V_out


def _to_bf16_np(tensor: torch.Tensor) -> np.ndarray:
    return np.array(tensor.detach().cpu().float().numpy(), dtype=ml_dtypes.bfloat16)


class KernelSimModule:
    def __init__(self, config: Qwen3MoeInferenceConfig, weight_store: Dict[str, torch.Tensor]):
        self._rotary_emb = build_rotary_emb(config)
        self._Wq_tt = _to_bf16_np(tile_transpose(weight_store["q_proj_weight"], HQ_TP, D, N_TILES))
        self._Wk_tt = _to_bf16_np(tile_transpose(weight_store["k_proj_weight"], 1, D, N_TILES))
        self._Wv_tt = _to_bf16_np(tile_transpose(weight_store["v_proj_weight"], 1, D, N_TILES))
        self._Wo = _to_bf16_np(weight_store["o_proj_weight"].T.contiguous())
        self._q_norm = _to_bf16_np(weight_store["q_norm_weight"])
        self._k_norm = _to_bf16_np(weight_store["k_norm_weight"])
        self._gamma_pre_attn = _to_bf16_np(weight_store["input_layernorm_weight"])
        self._sim = nki.simulate(nki.jit(_v14d_attn_raw)[LNC])

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ):
        if cos is None or sin is None:
            cos, sin = get_cos_sin_at_pos(self._rotary_emb, hidden_states, position_ids)

        output_np = np.zeros(tuple(hidden_states.shape), dtype=ml_dtypes.bfloat16)
        out_np, K_out_np, V_out_np = self._sim(
            _to_bf16_np(hidden_states),
            self._Wq_tt,
            self._Wk_tt,
            self._Wv_tt,
            self._Wo,
            self._q_norm,
            self._k_norm,
            self._gamma_pre_attn,
            _to_bf16_np(K_cache),
            _to_bf16_np(V_cache),
            _to_bf16_np(cos),
            _to_bf16_np(sin),
            position_ids.detach().cpu().numpy().astype(np.int32, copy=True),
            output_np,
        )
        return (
            torch.from_numpy(np.array(out_np, dtype=np.float32)).bfloat16(),
            torch.from_numpy(np.array(K_out_np, dtype=np.float32)).bfloat16(),
            torch.from_numpy(np.array(V_out_np, dtype=np.float32)).bfloat16(),
        )


def generate_cases(
    n_samples: int,
    positions: Sequence[int] = DEFAULT_POSITIONS,
    hidden_scales: Sequence[float] = DEFAULT_HIDDEN_SCALES,
    cache_scales: Sequence[float] = DEFAULT_CACHE_SCALES,
    distributions: Sequence[str] = DEFAULT_DISTS,
) -> List[Dict[str, float | int | str]]:
    combos = list(product(positions, hidden_scales, cache_scales, distributions))
    cases = []
    for i in range(n_samples):
        pos, hidden_scale, cache_scale, dist = combos[i % len(combos)]
        cases.append(
            {
                "sample_idx": i,
                "seed": 200 + i,
                "position": pos,
                "hidden_scale": hidden_scale,
                "cache_scale": cache_scale,
                "dist": dist,
            }
        )
    return cases


def _sample_dist(rng: np.random.Generator, shape: Tuple[int, ...], scale: float, dist: str) -> np.ndarray:
    if dist == "normal":
        values = rng.standard_normal(shape)
    elif dist == "student-t5":
        df = 5.0
        values = rng.standard_t(df, size=shape)
        values = values * math.sqrt((df - 2.0) / df)
    elif dist == "laplace":
        values = rng.laplace(0.0, 1.0 / math.sqrt(2.0), size=shape)
    else:
        raise ValueError(f"unknown distribution: {dist}")
    return (values * scale).astype(np.float32)


def make_sample(case: Dict[str, float | int | str]):
    rng = np.random.default_rng(int(case["seed"]))
    hidden = torch.from_numpy(_sample_dist(rng, (B, 1, H), float(case["hidden_scale"]), str(case["dist"]))).bfloat16()
    K_cache = torch.from_numpy(_sample_dist(rng, (B, 1, S, D), float(case["cache_scale"]), str(case["dist"]))).bfloat16()
    V_cache = torch.from_numpy(_sample_dist(rng, (B, 1, S, D), float(case["cache_scale"]), str(case["dist"]))).bfloat16()
    position_ids = torch.tensor([[int(case["position"])]], dtype=torch.int32)

    ref_mask = torch.zeros(B, 1, 1, S, dtype=torch.bool)
    pos = int(case["position"])
    if pos > 0:
        ref_mask[:, :, :, :pos] = True
    kern_mask = torch.zeros(B, 1, 1, S, dtype=torch.bool)
    return hidden, ref_mask, kern_mask, position_ids, K_cache, V_cache


def flatten_outputs(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().cpu().float().reshape(-1) for tensor in outputs], dim=0)


def compute_sample_stats(expected: Sequence[torch.Tensor], actual: Sequence[torch.Tensor]) -> Dict[str, float]:
    expected_flat = flatten_outputs(expected)
    actual_flat = flatten_outputs(actual)
    diff = actual_flat - expected_flat
    abs_diff = diff.abs()
    rel_diff = abs_diff / expected_flat.abs().clamp_min(1e-12)
    return {
        "max_abs_err": abs_diff.max().item(),
        "max_rel_err": rel_diff.max().item(),
        "mean_signed": diff.mean().item(),
        "std_signed": diff.std().item(),
        "attn_max_abs_err": (actual[0].float() - expected[0].float()).abs().max().item(),
        "K_max_abs_err": (actual[1].float() - expected[1].float()).abs().max().item(),
        "V_max_abs_err": (actual[2].float() - expected[2].float()).abs().max().item(),
    }


def outputs_close(
    expected: Sequence[torch.Tensor],
    actual: Sequence[torch.Tensor],
    atol: float,
    rtol: float,
) -> bool:
    try:
        for exp, act in zip(expected, actual):
            torch.testing.assert_close(exp, act, atol=atol, rtol=rtol, check_device=False)
        return True
    except AssertionError:
        return False


def format_case(case: Dict[str, float | int | str]) -> str:
    return (
        f"sample={case['sample_idx']} pos={case['position']} "
        f"hscale={case['hidden_scale']} cscale={case['cache_scale']} dist={case['dist']}"
    )


def dump_failure_sample(
    dump_path: str,
    case: Dict[str, float | int | str],
    sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ref_outputs: Sequence[torch.Tensor],
    kern_outputs: Sequence[torch.Tensor],
    weight_store: Dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    weight_scale: float,
    seed: int,
    tag: str,
    stats: Dict[str, float],
) -> None:
    hidden, ref_mask, _, position_ids, K_cache, V_cache = sample
    torch.save(
        {
            "hidden_states": hidden.cpu(),
            "ref_attention_mask": ref_mask.cpu(),
            "position_ids": position_ids.cpu(),
            "K_cache_in": K_cache.cpu(),
            "V_cache_in": V_cache.cpu(),
            "cos": cos.cpu(),
            "sin": sin.cpu(),
            "ref_out": ref_outputs[0].cpu(),
            "ref_K": ref_outputs[1].cpu(),
            "ref_V": ref_outputs[2].cpu(),
            "kern_out": kern_outputs[0].cpu(),
            "kern_K": kern_outputs[1].cpu(),
            "kern_V": kern_outputs[2].cpu(),
            "weight_scale": weight_scale,
            "seed": seed,
            "case": dict(case),
            "dump_tag": tag,
            "stats": dict(stats),
            **{key: value.cpu() for key, value in weight_store.items()},
        },
        dump_path,
    )
