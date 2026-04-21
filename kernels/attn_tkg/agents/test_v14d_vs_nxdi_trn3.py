"""
test_v14d_vs_nxdi_trn3.py

Verify v14d_kv_norm_hoisted_weights matches NeuronQwen3MoEAttention — the exact
attention class used by the full Qwen3 MoE TKG model.  Both are compiled with
build_module at TP=1 / LNC=2 using the production TKG compiler args.

Weights are identical: both module __init__s call torch.manual_seed(42) and
create input_layernorm then NeuronQwen3MoEAttention in the same order.

Dims: H=2048, d=128, Hq_tp=8 (32/TP=4), Hkv_tp=1 (4/TP=4), S=640.

Tolerance (bf16-native, no fp32 promotion): atol=1e-5, rtol=1e-2.
Checks: attention output [B,1,H] and KV cache row at position_ids.

Run:
    python kernels/attn_tkg/agents/test_v14d_vs_nxdi_trn3.py
"""

import argparse
import os, sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/ubuntu/nki-moe")
sys.path.insert(0, _HERE)

import tempfile
import numpy as np
import torch
import torch.nn as nn
import nki
import nki.language as nl
import nki.isa as nisa
from nkilib.core.utils.allocator import SbufManager

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

from qwen import Qwen3MoeInferenceConfig, NeuronQwen3MoEAttention
from v14d_kv_norm_hoisted_weights import (
    qwen3_attn_tkg_fused_oproj_v13bc_kv_norm_hoisted_weights as _v14d_kernel,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/ubuntu/models/Qwen3-30B-A3B"
PMAX      = 128
B, H, d, Hq_tp, S = 1, 2048, 128, 8, 640
N_TILES   = H // d   # 16
LNC       = 2

# Verbatim from NeuronQwen3MoeForCausalLM.get_compiler_args(), TKG, no EP.
COMPILER_ARGS = (
    "--enable-saturate-infinity --enable-mixed-precision-accumulation "
    "--model-type transformer -O1 "
    "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def make_config() -> Qwen3MoeInferenceConfig:
    neuron_config = MoENeuronConfig(
        tp_degree=1,
        batch_size=1,
        seq_len=S,
        flash_decoding_enabled=False,
        logical_nc_config=LNC,
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
    # Simulate TP=4 rank 0: per-rank head counts
    cfg.num_attention_heads = Hq_tp   # 32 / 4 = 8
    cfg.num_key_value_heads = 1        # 4  / 4 = 1
    return cfg


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def tile_transpose(W: torch.Tensor, n_heads: int, d_dim: int, n_tiles: int) -> torch.Tensor:
    return (W.reshape(n_heads, d_dim, n_tiles, d_dim)
              .permute(0, 3, 2, 1)
              .reshape(n_heads * d_dim, n_tiles * d_dim)
              .contiguous())


def make_attention_mask(pos: int) -> torch.Tensor:
    m = torch.zeros(B, 1, 1, S, dtype=torch.bfloat16)
    m[:, :, :, pos + 1:] = float("-inf")
    return m


# ---------------------------------------------------------------------------
# Reference module
# NeuronQwen3MoEAttention with input_layernorm prepended, matching the exact
# call sequence of NeuronQwen3MoeDecoderLayer.forward().
# ---------------------------------------------------------------------------
class RefAttnModule(nn.Module):
    def __init__(self, config: Qwen3MoeInferenceConfig, seed: int = 42):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.input_layernorm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype)
        self.attn = NeuronQwen3MoEAttention(config).to(dtype)

    def forward(
        self,
        hidden_states,   # [B, 1, H]        bf16
        attention_mask,  # [B, 1, 1, S]     bf16
        position_ids,    # [B, 1]            int32
        K_cache,         # [B, 1, S, d]      bf16  (updated in-place)
        V_cache,         # [B, 1, S, d]      bf16  (updated in-place)
    ):
        normed = self.input_layernorm(hidden_states)
        hidden_out, kv, _, _ = self.attn(
            hidden_states=normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=(K_cache, V_cache),
        )
        return hidden_out, kv[0], kv[1]


# ---------------------------------------------------------------------------
# v14d wrapper — raw (no @nki.jit), mirrors test_v14d_full_hoisted_trn3.py.
# Wrapped in KernelAttnModule with nki.jit so it compiles inside build_module.
# ---------------------------------------------------------------------------
def _v14d_attn_raw(
    hidden_states,    # [B, 1, H]
    Wq,               # [Hq_tp*d, H]       bf16 tile-transposed
    Wk,               # [PMAX, N_TILES*d]   bf16 tile-transposed
    Wv,               # [PMAX, N_TILES*d]   bf16 tile-transposed
    Wo,               # [Hq_tp*d, H]        bf16 plain .T layout
    q_norm_weight,    # [d]                 bf16
    k_norm_weight,    # [d]                 bf16
    gamma_pre_attn,   # [H]                 bf16
    K_cache,          # [B, 1, S, d]        bf16  mutated in-place
    V_cache,          # [B, 1, S, d]        bf16  mutated in-place
    cos,              # [B, d]              bf16
    sin,              # [B, d]              bf16
    position_ids,     # [B, 1]              int32
    output,           # [B, 1, H]           bf16  written in-place
):
    B_loc        = cos.shape[0]
    H_dim        = Wq.shape[1]
    H_wo         = Wo.shape[1]
    num_h_tiles  = H_dim // PMAX
    num_out_cols = H_wo // PMAX
    Hq_tp_loc    = Wq.shape[0] // PMAX

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

    # Load hidden into SBUF column layout
    hidden_col = hidden_states.reshape((H_dim, B_loc))
    hidden_sb  = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="hidden_sb")
    nisa.dma_copy(
        dst=hidden_sb,
        src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )

    cos_col = cos.reshape((PMAX, B_loc))
    sin_col = sin.reshape((PMAX, B_loc))

    # Hoist scalar norm/RoPE constants into f32 SBUF
    qnw_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="qnw_bf16")
    nisa.dma_copy(dst=qnw_bf16, src=q_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
    qnw_f32  = sbm.alloc_stack((PMAX, 1), nl.float32,  name="qnw_f32")
    nisa.tensor_copy(qnw_f32, qnw_bf16)

    knw_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="knw_bf16")
    nisa.dma_copy(dst=knw_bf16, src=k_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
    knw_f32  = sbm.alloc_stack((PMAX, 1), nl.float32,  name="knw_f32")
    nisa.tensor_copy(knw_f32, knw_bf16)

    cos_bf16 = sbm.alloc_stack((PMAX, B_loc), nl.bfloat16, name="cos_bf16")
    nisa.dma_copy(dst=cos_bf16, src=cos_col, dge_mode=nisa.dge_mode.hwdge)
    cos_f32  = sbm.alloc_stack((PMAX, B_loc), nl.float32,  name="cos_f32")
    nisa.tensor_copy(cos_f32, cos_bf16)

    sin_bf16 = sbm.alloc_stack((PMAX, B_loc), nl.bfloat16, name="sin_bf16")
    nisa.dma_copy(dst=sin_bf16, src=sin_col, dge_mode=nisa.dge_mode.hwdge)
    sin_f32  = sbm.alloc_stack((PMAX, B_loc), nl.float32,  name="sin_f32")
    nisa.tensor_copy(sin_f32, sin_bf16)

    gpan_bf16 = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="gpan_bf16")
    nisa.dma_copy(
        dst=gpan_bf16,
        src=gamma_pre_attn.reshape((H_dim, 1)).ap(
            pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )
    gpan_f32  = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="gpan_f32")
    nisa.tensor_copy(gpan_f32, gpan_bf16)

    # Hoist weight matrices into bf16 SBUF
    wk_sb = sbm.alloc_stack((PMAX, num_h_tiles * PMAX), nl.bfloat16, name="wk_sb")
    nisa.dma_copy(dst=wk_sb, src=Wk, dge_mode=nisa.dge_mode.hwdge)

    wv_sb = sbm.alloc_stack((PMAX, num_h_tiles * PMAX), nl.bfloat16, name="wv_sb")
    nisa.dma_copy(dst=wv_sb, src=Wv, dge_mode=nisa.dge_mode.hwdge)

    wq_heads = []
    for q_h in owned_heads:
        w = sbm.alloc_stack((PMAX, num_h_tiles * PMAX), nl.bfloat16, name=f"wq_h_{q_h}")
        nisa.dma_copy(dst=w, src=Wq[q_h * PMAX:(q_h + 1) * PMAX, :], dge_mode=nisa.dge_mode.hwdge)
        wq_heads.append(w)

    wo_heads = []
    for q_h in owned_heads:
        wo_tile = sbm.alloc_stack((PMAX, H_wo), nl.bfloat16, name=f"wo_h_{q_h}")
        nisa.dma_copy(
            dst=wo_tile,
            src=Wo.reshape((Hq_tp_loc, PMAX, H_wo)).ap(
                pattern=[[H_wo, PMAX], [1, H_wo]],
                offset=q_h * PMAX * H_wo,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )
        wo_heads.append(wo_tile)

    out_sb = _v14d_kernel(
        hidden_sb=hidden_sb,
        Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        gamma_pre_attn=gamma_pre_attn,
        K_cache=K_cache, V_cache=V_cache,
        cos=cos, sin=sin, position_ids=position_ids,
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

    # Write out_sb [PMAX, num_out_cols] -> output [B_loc, 1, H_wo]
    output_flat = output.reshape((B_loc, H_wo))
    for j in range(num_out_cols):
        col_psum = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(col_psum, out_sb[0:PMAX, j:j + 1])
        col_sb = sbm.alloc_stack((1, PMAX), nl.bfloat16, name=f"col_sb_{j}")
        nisa.tensor_copy(col_sb, col_psum)
        nisa.dma_copy(
            dst=output_flat[0:B_loc, j * PMAX:(j + 1) * PMAX],
            src=col_sb,
            dge_mode=nisa.dge_mode.hwdge,
        )

    sbm.close_scope()
    sbm.close_scope()
    return output, K_cache, V_cache


# ---------------------------------------------------------------------------
# Kernel module
# Same weight creation order/seed as RefAttnModule → identical weights.
# Extracts weights, tile-transposes, and calls v14d via nki.jit.
# ---------------------------------------------------------------------------
class KernelAttnModule(nn.Module):
    def __init__(self, config: Qwen3MoeInferenceConfig, seed: int = 42):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.input_layernorm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype)
        self.attn = NeuronQwen3MoEAttention(config).to(dtype)
        self._attn_jit = nki.jit(_v14d_attn_raw)

    def forward(
        self,
        hidden_states,   # [B, 1, H]
        attention_mask,  # [B, 1, 1, S]  (not used by v14d; causal mask is implicit)
        position_ids,    # [B, 1]         int32
        K_cache,         # [B, 1, S, d]
        V_cache,         # [B, 1, S, d]
    ):
        # Weight extraction — qkv_proj path for fused_qkv=False
        Wq_raw = self.attn.qkv_proj.q_proj.weight     # [Hq_tp*d, H]  = [1024, 2048]
        Wk_raw = self.attn.qkv_proj.k_proj.weight     # [d, H]         = [128,  2048]
        Wv_raw = self.attn.qkv_proj.v_proj.weight     # [d, H]         = [128,  2048]
        Wo_raw = self.attn.o_proj.o_proj.weight        # [H, Hq_tp*d]  = [2048, 1024]
        qn     = self.attn.q_layernorm.weight          # [d]
        kn     = self.attn.k_layernorm.weight          # [d]
        gpan   = self.input_layernorm.weight           # [H]

        Wq_tt = tile_transpose(Wq_raw, Hq_tp, d, N_TILES)  # [1024, 2048] tile-transposed
        Wk_tt = tile_transpose(Wk_raw, 1,     d, N_TILES)  # [128,  2048] tile-transposed
        Wv_tt = tile_transpose(Wv_raw, 1,     d, N_TILES)
        Wo_k  = Wo_raw.T.contiguous()                       # [Hq_tp*d, H] = [1024, 2048]

        # Rotary embeddings at the current positions
        cos_full, sin_full = self.attn.rotary_emb(hidden_states, position_ids)
        cos_at_pos = cos_full.squeeze(1)    # [B, d]
        sin_at_pos = sin_full.squeeze(1)    # [B, d]

        pos_i32 = position_ids.to(torch.int32)
        output  = torch.zeros_like(hidden_states)   # [B, 1, H]

        out, K_out, V_out = self._attn_jit[LNC](
            hidden_states,
            Wq_tt, Wk_tt, Wv_tt, Wo_k,
            qn, kn, gpan,
            K_cache, V_cache,
            cos_at_pos, sin_at_pos, pos_i32, output,
        )
        return out, K_out, V_out


# ---------------------------------------------------------------------------
# Build inputs / expected_outputs for validate_accuracy
# ---------------------------------------------------------------------------
def make_inputs_and_expected(kern_traced, positions):
    """
    Run kern_traced over all positions to collect (inputs, expected_outputs).

    kern_traced returns (attn_out, K_full, V_full) with K_full shape (B,1,S,d).
    ref_traced (NxDI) returns (attn_out, K_new, V_new) with K_new shape (B,1,1,d).
    Slice K_full/V_full at slot `pos` so shapes match for validate_accuracy.
    """
    inputs_list   = []
    expected_list = []

    for i, pos in enumerate(positions):
        rng = np.random.default_rng(100 + i)
        SC  = 0.1

        def r(*shape):
            return torch.from_numpy((rng.standard_normal(shape) * SC).astype(np.float32)).bfloat16()

        hidden = r(B, 1, H)
        K_init = r(B, 1, S, d)
        V_init = r(B, 1, S, d)
        # v14d kernel ignores the mask (implicit causal); pass bool for dtype consistency.
        kern_mask = torch.zeros(B, 1, 1, S, dtype=torch.bool)
        # NxDI uses torch.where(attention_mask, scores, min_val) WITHOUT .to(torch.bool),
        # so float masks have inverted semantics (0→False→masked, -inf→True→kept).
        # The E2E model passes bool masks: True = include in prior (positions 0..pos-1),
        # False = exclude (positions pos..S-1).  We must match that convention here.
        ref_mask = torch.zeros(B, 1, 1, S, dtype=torch.bool)
        if pos > 0:
            ref_mask[:, :, :, :pos] = True
        pos_t  = torch.tensor([[pos]], dtype=torch.int32)

        inp = (hidden, kern_mask, pos_t, K_init.clone(), V_init.clone())
        attn_out, K_full, V_full = kern_traced(*inp)

        # Slice to (B,1,1,d) to match NxDI's return shape
        K_new = K_full[:, :, pos:pos + 1, :]
        V_new = V_full[:, :, pos:pos + 1, :]

        inputs_list.append((hidden, ref_mask, pos_t, K_init.clone(), V_init.clone()))
        expected_list.append((attn_out, K_new, V_new))
        print(f"  collected pos={pos}", flush=True)

    return inputs_list, expected_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _SEED = args.seed

    cfg = make_config()
    print(
        f"Config: H={cfg.hidden_size} d={cfg.head_dim} "
        f"Hq={cfg.num_attention_heads} Hkv={cfg.num_key_value_heads} "
        f"S={S} TP=1 LNC={LNC}",
        flush=True,
    )

    example_hidden = torch.zeros(B, 1, H,    dtype=torch.bfloat16)
    example_mask   = torch.zeros(B, 1, 1, S, dtype=torch.bool)   # pos=0: empty prior
    example_pos    = torch.zeros(B, 1,        dtype=torch.int32)
    example_K      = torch.zeros(B, 1, S, d, dtype=torch.bfloat16)
    example_V      = torch.zeros(B, 1, S, d, dtype=torch.bfloat16)
    example_inputs = [(example_hidden, example_mask, example_pos, example_K, example_V)]

    positions = [0, 1, 63, 64, 127, 128, 129, 255, 320]

    with tempfile.TemporaryDirectory(prefix="v14d_vs_nxdi_") as workdir:
        print("\nCompiling RefAttnModule...", flush=True)
        ref_traced = build_module(
            module_cls=RefAttnModule,
            example_inputs=example_inputs,
            module_init_kwargs={"config": cfg, "seed": _SEED},
            tp_degree=1,
            logical_nc_config=LNC,
            compiler_args=COMPILER_ARGS,
            compiler_workdir=os.path.join(workdir, "ref_workdir"),
        )
        print("RefAttnModule compile PASS\n", flush=True)

        print("Compiling KernelAttnModule...", flush=True)
        kern_traced = build_module(
            module_cls=KernelAttnModule,
            example_inputs=example_inputs,
            module_init_kwargs={"config": cfg, "seed": _SEED},
            tp_degree=1,
            logical_nc_config=LNC,
            compiler_args=COMPILER_ARGS,
            compiler_workdir=os.path.join(workdir, "kern_workdir"),
        )
        print("KernelAttnModule compile PASS\n", flush=True)

        print("=" * 65)
        print("  v14d vs NeuronQwen3MoEAttention — trn3, LNC=2")
        print("=" * 65, flush=True)

        print("\nCollecting kernel outputs for all positions...", flush=True)
        inputs_list, expected_list = make_inputs_and_expected(kern_traced, positions)

        print("\nRunning validate_accuracy (ref_traced vs kernel expected)...", flush=True)
        validate_accuracy(
            ref_traced,
            inputs_list,
            expected_outputs=expected_list,
            assert_close_kwargs={"atol": 1e-5, "rtol": 1e-2, "check_device": False},
        )

    print("\nOVERALL: PASS")


if __name__ == "__main__":
    main()
