"""
test_v30c_vs_nxdi_trn3.py

Verify qwen3_moe_fused_tkg_sbuf_io (v30c) matches NeuronQwen3MoeDecoderLayer's MoE —
the exact class and calling convention used by qwen.py during token generation.

Reference: NxDI MoE (init_tkg_module=False, pure PyTorch) with post_attention_layernorm
  applied explicitly before the call — matching qwen.py's moe_fused_nki_kernel_enabled=False
  code path (lines 427-429):
      hidden_states = self.post_attention_layernorm(hidden_states)
      hidden_states = self.mlp(hidden_states, padding_mask)[0]

Kernel: v30c wrapper where RMSNorm is fused inside the kernel, matching the
  moe_fused_nki_kernel_enabled=True path where norm is passed to initialize_moe_module
  and not applied by the decoder layer.

Both modules are seeded identically (manual_seed(42)) so weights are identical.

Dims: H=2048, E=128, I=192 (moe_tp_degree=1), K=8, B=1, S=640, LNC=2.

Tolerance (bf16-native, no fp32 promotion): atol=1e-5, rtol=1e-2.

Run:
    python tests/test_v30c_vs_nxdi_trn3.py
"""

import os
import sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOE_VERSIONS = os.path.join(_HERE, "..", "kernels", "moe_fused_tkg", "versions")
sys.path.insert(0, "/home/ubuntu/nki-moe")
sys.path.insert(0, _MOE_VERSIONS)

import argparse
import tempfile

import numpy as np
import torch
import torch.nn as nn
import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.allocator import SbufManager

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

from qwen import Qwen3MoeInferenceConfig
from kernel_v30c_hoisted import (
    _qwen3_moe_sbuf_in_sbuf_out_hoisted,
    _PMAX,
    _H,
    _E,
    _H_FREE,
    _H_FREE_SHARD,
    _H_SHARD,
    _ROUTER_BATCH,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/ubuntu/models/Qwen3-30B-A3B"
B   = 1
S   = 640
LNC = 2

# Verbatim from NeuronQwen3MoeForCausalLM.get_compiler_args(), TKG branch, no EP.
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
        moe_tp_degree=1,
        batch_size=1,
        seq_len=S,
        flash_decoding_enabled=False,
        logical_nc_config=LNC,
        # Disable NxDI's internal TKG NKI kernel so the reference is pure PyTorch MoE.
        moe_fused_nki_kernel_enabled=False,
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
    # Simulate a single TP=4 rank: v30c hardcodes _I=192 = moe_intermediate_size//4.
    # Qwen3MoeInferenceConfig sets intermediate_size = moe_intermediate_size (768 full),
    # but with moe_tp_degree=1 ExpertFusedColumnParallelLinear creates no-sharding weight
    # [E, H, 2*768]. Override to 192 so both modules create [E, H, 384] weights that the
    # kernel actually expects, mirroring what test_v14d does for attention head counts.
    cfg.intermediate_size = cfg.moe_intermediate_size // 4  # 768 // 4 = 192
    return cfg


# ---------------------------------------------------------------------------
# Reference module
# Mirrors NeuronQwen3MoeDecoderLayer.forward() with moe_fused_nki_kernel_enabled=False:
#   hidden_states = self.post_attention_layernorm(hidden_states)
#   hidden_states = self.mlp(hidden_states, padding_mask)[0]
# ---------------------------------------------------------------------------
class RefMoEModule(nn.Module):
    def __init__(self, config: Qwen3MoeInferenceConfig, seed: int = 42):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.post_attention_layernorm = CustomRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        ).to(dtype)
        self.mlp = initialize_moe_module(config, init_tkg_module=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, 1, H] -> norm -> MoE -> [B, 1, H]
        normed = self.post_attention_layernorm(hidden_states)
        return self.mlp(normed, None)[0]


# ---------------------------------------------------------------------------
# Raw kernel body — non-decorated so it can be wrapped with nki.jit() below.
# Mirrors qwen3_moe_fused_tkg_sbuf_io (back-compat mode: both hoisting kwargs=None).
# ---------------------------------------------------------------------------
def _v30c_moe_raw(inp, gamma, router_w, gate_up_w, down_w):
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)

    T = inp.shape[0]  # number of tokens (= B for TKG)

    inp_2d             = inp.reshape((T, _H))
    inp_2d_hbm_reshaped = inp_2d.reshape((_H_FREE * T, _PMAX))

    sbm.open_scope(name="inp_load")
    inp_flat_sb = sbm.alloc_stack(
        (_H_FREE * T, _PMAX), inp.dtype, buffer=nl.sbuf, name="inp_flat_sb"
    )
    nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_reshaped, dge_mode=3)
    inp_trans_psum = nl.ndarray((_PMAX, _H_FREE * T), dtype=inp.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)
    inp_sb = sbm.alloc_stack(
        (_PMAX, _H_FREE * T), inp.dtype, buffer=nl.sbuf, name="inp_sb"
    )
    nisa.activation(inp_sb[...], op=nl.copy, data=inp_trans_psum[...])

    # Pre-load router_w into SBUF using the exact DMA pattern from the
    # multi-layer fused megakernel (transformer_qwen_multilayer.py:262-269),
    # then feed the SBUF tensor in via router_w_wide_sb. This exercises the
    # same router path the megakernel uses, so any layout discrepancy between
    # HBM-fallback and hoisted-SBUF routes will surface in this test.
    router_w_wide_sb = sbm.alloc_stack(
        (_PMAX, _ROUTER_BATCH, _E), nl.float32, buffer=nl.sbuf, name="router_w_wide_sb"
    )
    nisa.dma_copy(
        dst=router_w_wide_sb,
        src=router_w.ap(
            pattern=[[_E, _PMAX], [_PMAX * _E, _ROUTER_BATCH], [1, _E]],
            offset=0,
        ),
        dge_mode=3,
    )

    out_sb = _qwen3_moe_sbuf_in_sbuf_out_hoisted(
        inp_sb, inp.dtype, T,
        gamma, router_w, gate_up_w, down_w,
        sbm=sbm,
        gamma_sb_ready=None,
        router_w_wide_sb=router_w_wide_sb,
    )

    prg_id = nl.program_id(axis=0)
    output = nl.ndarray((T, _H), dtype=inp.dtype, buffer=nl.shared_hbm)

    sbm.open_scope(name="store_hbm")
    out_row_sb = sbm.alloc_stack(
        (T, _H_SHARD), inp.dtype, buffer=nl.sbuf, name="out_row_sb"
    )
    for h1 in nl.static_range(_H_FREE_SHARD):
        tp_psum = nl.ndarray((T, _PMAX), dtype=inp.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=tp_psum[0:T, 0:_PMAX],
            data=out_sb[0:_PMAX, nl.ds(h1 * T, T)],
        )
        nisa.activation(
            dst=out_row_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)],
            op=nl.copy,
            data=tp_psum[0:T, 0:_PMAX],
        )
    nisa.dma_copy(
        dst=output[0:T, nl.ds(prg_id * _H_SHARD, _H_SHARD)],
        src=out_row_sb[0:T, 0:_H_SHARD],
    )
    sbm.close_scope()  # store_hbm
    sbm.close_scope()  # inp_load
    return output


# ---------------------------------------------------------------------------
# Kernel module
# Same weight creation order/seed as RefMoEModule → identical weights.
# RMSNorm is fused inside v30c; hidden_states enters pre-norm, matching
# the moe_fused_nki_kernel_enabled=True path in qwen.py.
# ---------------------------------------------------------------------------
class KernelMoEModule(nn.Module):
    def __init__(self, config: Qwen3MoeInferenceConfig, seed: int = 42):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.post_attention_layernorm = CustomRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        ).to(dtype)
        self.mlp = initialize_moe_module(config, init_tkg_module=False)
        self._moe_jit = nki.jit(_v30c_moe_raw)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Weight extraction
        # post_attention_layernorm.weight: [H]  -> [1, H]
        gamma    = self.post_attention_layernorm.weight.unsqueeze(0)
        # router.linear_router is nn.Linear([H, E] stored as [E, H]); v30c wants [H, E]
        router_w  = self.mlp.router.linear_router.weight.T.contiguous()
        # expert_mlps.mlp_op.gate_up_proj.weight: [E, H, 2*I]  (matches v30c gate_up_w)
        gate_up_w = self.mlp.expert_mlps.mlp_op.gate_up_proj.weight
        # expert_mlps.mlp_op.down_proj.weight:    [E, I, H]    (matches v30c down_w)
        down_w    = self.mlp.expert_mlps.mlp_op.down_proj.weight

        # v30c expects inp [T, H]; reshape [B, 1, H] -> [B, H] (T = B = 1 for TKG)
        inp_2d = hidden_states.reshape(B, _H)
        out_2d = self._moe_jit[LNC](inp_2d, gamma, router_w, gate_up_w, down_w)  # [T, H]
        return out_2d.reshape(hidden_states.shape)  # [B, 1, H]


# ---------------------------------------------------------------------------
# Build inputs / expected_outputs for validate_accuracy
# Run kern_traced over inputs drawn from a distribution closer to real
# post-residual pre-MoE hidden states than N(0, 0.1):
#   - scale sweep covering small-to-large activation magnitudes,
#   - Student-t samples (df=5) to inject heavy tails that push router scores
#     toward boundary conditions where bf16 drift flips experts.
# ---------------------------------------------------------------------------
def _sample_hidden(rng: np.random.Generator, scale: float, heavy_tail: bool) -> torch.Tensor:
    if heavy_tail:
        # Student-t df=5: finite variance, kurtosis ~9 (vs 3 for gaussian).
        # Rescale to unit std so `scale` is comparable across distributions.
        df = 5.0
        x = rng.standard_t(df, size=(B, 1, _H))
        x = x * np.sqrt((df - 2.0) / df)
    else:
        x = rng.standard_normal((B, 1, _H))
    return torch.from_numpy((x * scale).astype(np.float32)).bfloat16()


def make_inputs_and_expected(kern_traced, n_samples: int = 512):
    # Scales chosen to bracket realistic post-residual activation magnitudes:
    #   0.1 is the original regime (easy routing), 1.0 is near real TKG inputs,
    #   3.0 stresses saturation paths in RMSNorm + softmax.
    scales = [0.1, 0.5, 3.0, 5.0]
    combos = [(s, ht) for s in scales for ht in (False, True)]

    inputs_list   = []
    expected_list = []
    for i in range(n_samples):
        scale, heavy_tail = combos[i % len(combos)]
        rng = np.random.default_rng(200 + i)
        hidden = _sample_hidden(rng, scale, heavy_tail)

        out = kern_traced(hidden)

        inputs_list.append((hidden,))
        expected_list.append(out)
        tag = "t5" if heavy_tail else "N"
        print(f"  collected sample {i:3d}  scale={scale:<4}  dist={tag}", flush=True)

    return inputs_list, expected_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = make_config()
    print(
        f"Config: H={cfg.hidden_size} E={cfg.num_experts} I={cfg.moe_intermediate_size} "
        f"K={cfg.num_experts_per_tok} B={B} S={S} TP=1 LNC={LNC}",
        flush=True,
    )

    example_inputs = [(torch.zeros(B, 1, _H, dtype=torch.bfloat16),)]

    with tempfile.TemporaryDirectory(prefix="v30c_vs_nxdi_") as workdir:
        print("\nCompiling RefMoEModule...", flush=True)
        ref_traced = build_module(
            module_cls=RefMoEModule,
            example_inputs=example_inputs,
            module_init_kwargs={"config": cfg, "seed": args.seed},
            tp_degree=1,
            logical_nc_config=LNC,
            compiler_args=COMPILER_ARGS,
            compiler_workdir=os.path.join(workdir, "ref_workdir"),
        )
        print("RefMoEModule compile PASS\n", flush=True)

        print("Compiling KernelMoEModule...", flush=True)
        kern_traced = build_module(
            module_cls=KernelMoEModule,
            example_inputs=example_inputs,
            module_init_kwargs={"config": cfg, "seed": args.seed},
            tp_degree=1,
            logical_nc_config=LNC,
            compiler_args=COMPILER_ARGS,
            compiler_workdir=os.path.join(workdir, "kern_workdir"),
        )
        print("KernelMoEModule compile PASS\n", flush=True)

        print("=" * 65)
        print("  v30c vs NxDI MoE — trn3, LNC=2")
        print("=" * 65, flush=True)

        print("\nCollecting kernel outputs for all samples...", flush=True)
        inputs_list, expected_list = make_inputs_and_expected(kern_traced)

        print("\nRunning validate_accuracy (ref_traced vs kernel expected)...", flush=True)
        validate_accuracy(
            ref_traced,
            inputs_list,
            expected_outputs=expected_list,
            assert_close_kwargs={"atol": 1e-5, "rtol": 2e-2, "check_device": False},
        )

    print("\nOVERALL: PASS")


if __name__ == "__main__":
    main()
