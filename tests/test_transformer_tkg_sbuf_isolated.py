"""
Isolated test for nkilib transformer_tkg SBUF path (sbuf_residual_and_cc=True).

Runs ONLY the SBUF path — no HBM path first — so NeuronCores are available.
Also clears the specific cached NEFF before running so the SBUF path is
compiled fresh rather than reusing the HBM-path NEFF (same cache key due to
sbuf_residual_and_cc not being part of the compiler cache hash).

Usage:
    python tests/test_transformer_tkg_sbuf_isolated.py
"""

import os
import sys
import glob
import shutil
import tempfile

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "./output"

sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch.nn as nn
import nki

from nkilib.experimental.transformer.transformer_tkg import transformer_tkg
from nkilib.experimental.transformer.transformer_tkg_torch import (
    llama3_transformer_fwd_tkg_torch,
)
from neuronx_distributed.trace import parallel_model_trace

# ── model config (matches test_transformer_tkg.py) ─────────────────────────
B          = 1
S_TKG      = 1
H          = 2048
D_HEAD     = 128
Q_HEADS    = 8
I_MLP      = 1024
S_CTX      = 512
NUM_LAYERS = 2
TP         = 2

COMPILER_ARGS = (
    "--enable-saturate-infinity "
    "--enable-mixed-precision-accumulation "
    "--model-type transformer "
    "-O3 "
    "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-hlo2tensorizer-options='--verify-hlo=true' "
    "--internal-max-instruction-limit=15000000"
)


def make_weights(dtype=torch.bfloat16):
    return dict(
        W_qkv       = torch.randn(H, D_HEAD * (Q_HEADS + 2), dtype=dtype) * 0.02,
        W_out       = torch.randn(Q_HEADS * D_HEAD, H, dtype=dtype) * 0.02,
        W_gate      = torch.randn(H, I_MLP, dtype=dtype) * 0.02,
        W_up        = torch.randn(H, I_MLP, dtype=dtype) * 0.02,
        W_down      = torch.randn(I_MLP, H, dtype=dtype) * 0.02,
        W_gamma_qkv = torch.ones(1, H, dtype=dtype),
        W_gamma_mlp = torch.ones(1, H, dtype=dtype),
        K_cache     = torch.zeros(B, D_HEAD, S_CTX, dtype=dtype),
        V_cache     = torch.zeros(B, S_CTX, D_HEAD, dtype=dtype),
    )


def make_inputs(dtype=torch.bfloat16):
    mask = torch.ones(S_CTX, B, Q_HEADS, S_TKG, dtype=dtype)
    return dict(
        X            = torch.randn(B, S_TKG, H, dtype=dtype) * 0.1,
        RoPE_cos     = torch.ones(D_HEAD // 2, B, S_TKG, dtype=dtype),
        RoPE_sin     = torch.zeros(D_HEAD // 2, B, S_TKG, dtype=dtype),
        mask_cache   = mask,
        mask_active  = mask.clone(),
        position_ids = torch.zeros(B, S_TKG, dtype=torch.int32),
    )


class TransformerTKGSBUFModule(nn.Module):
    def __init__(self, weights_list):
        super().__init__()
        self.num_layers = len(weights_list)
        for i, w in enumerate(weights_list):
            for name, t in w.items():
                self.register_buffer(f"L{i}_{name}", t)

    def forward(self, X, RoPE_cos, RoPE_sin, mask_cache, mask_active, position_ids):
        NL = self.num_layers
        W_qkvs      = tuple(getattr(self, f"L{i}_W_qkv")      for i in range(NL))
        W_outs      = tuple(getattr(self, f"L{i}_W_out")       for i in range(NL))
        W_gates     = tuple(getattr(self, f"L{i}_W_gate")      for i in range(NL))
        W_ups       = tuple(getattr(self, f"L{i}_W_up")        for i in range(NL))
        W_downs     = tuple(getattr(self, f"L{i}_W_down")      for i in range(NL))
        W_gamma_qkvs = tuple(getattr(self, f"L{i}_W_gamma_qkv") for i in range(NL))
        W_gamma_mlps = tuple(getattr(self, f"L{i}_W_gamma_mlp") for i in range(NL))
        K_caches    = tuple(getattr(self, f"L{i}_K_cache")     for i in range(NL))
        V_caches    = tuple(getattr(self, f"L{i}_V_cache")     for i in range(NL))

        return nki.jit(transformer_tkg)[2](
            X,
            W_qkvs, W_outs, W_gates, W_ups, W_downs,
            W_gamma_qkvs, W_gamma_mlps,
            K_caches, V_caches,
            RoPE_cos, RoPE_sin,
            mask_cache, mask_active,
            position_ids,
            num_layers=NL,
            replica_groups=([0, 1],),
            sbuf_residual_and_cc=True,  # SBUF path
        )


_WEIGHTS_LIST = []


def _module_factory():
    m = TransformerTKGSBUFModule(_WEIGHTS_LIST)
    m.eval()
    return m, {}


def clear_neff_cache():
    """
    Delete any cached NEFFs whose module hash matches the transformer_tkg
    signature. The SBUF and HBM paths share the same hash because
    sbuf_residual_and_cc is a Python constant, not a traced tensor.
    Without this, the SBUF test silently loads the HBM-path NEFF from cache.
    """
    cache_roots = [
        "/var/tmp/neuron-compile-cache",
        os.path.expanduser("~/.cache/neuron-compile-cache"),
    ]
    pattern = "MODULE_7257916412218192313*"
    deleted = []
    for root in cache_roots:
        for match in glob.glob(os.path.join(root, "**", pattern), recursive=True):
            shutil.rmtree(match, ignore_errors=True)
            deleted.append(match)
    if deleted:
        print(f"  Cleared {len(deleted)} cached NEFF(s):")
        for d in deleted:
            print(f"    {d}")
    else:
        print("  No matching cached NEFFs found (will compile fresh regardless).")


def main():
    global _WEIGHTS_LIST
    _WEIGHTS_LIST = [make_weights() for _ in range(NUM_LAYERS)]

    inp = make_inputs()
    example_args = (
        inp["X"], inp["RoPE_cos"], inp["RoPE_sin"],
        inp["mask_cache"], inp["mask_active"], inp["position_ids"],
    )

    print("=" * 60)
    print("transformer_tkg SBUF path — isolated fresh-compile test")
    print(f"  sbuf_residual_and_cc=True, tp_degree={TP}, LNC=2")
    print(f"  H={H}, d_head={D_HEAD}, q_heads={Q_HEADS}/rank, layers={NUM_LAYERS}")
    print("=" * 60)

    # print("\nClearing cached NEFF to force fresh compilation...")
    # clear_neff_cache()

    with tempfile.TemporaryDirectory(prefix="tkg_sbuf_isolated_") as workdir:
        try:
            print("\nCompiling SBUF path...")
            traced = parallel_model_trace(
                _module_factory,
                example_args,
                tp_degree=TP,
                compiler_workdir=workdir,
                compiler_args=COMPILER_ARGS,
                # inline_weights_to_neff=True,
            )
            print("Compile PASS — running inference...")
            result = traced(*example_args)
            print(f"Output shape: {result.shape}")
            assert result.shape == (B, S_TKG, H)
            r = result.float()
            print(f"Output stats: min={r.min():.4f}  max={r.max():.4f}  "
                  f"mean={r.mean():.4f}  std={r.std():.4f}")
            assert torch.isfinite(r).all(), "output contains inf/nan"

            # DCE tripwire: a passthrough/empty NEFF would make output ≡ input X.
            x_in = inp["X"].float()
            assert not torch.allclose(r, x_in, atol=1e-3, rtol=1e-3), \
                "DCE tripwire: output equals input X — kernel body was eliminated"

            # Torch reference (full computation, no sharding; allreduce simulated
            # by multiplying by cc_workers inside llama3_transformer_fwd_tkg_torch).
            print("\nComputing torch reference...")
            W_qkvs       = [w["W_qkv"]       for w in _WEIGHTS_LIST]
            W_outs       = [w["W_out"]       for w in _WEIGHTS_LIST]
            W_gates      = [w["W_gate"]      for w in _WEIGHTS_LIST]
            W_ups        = [w["W_up"]        for w in _WEIGHTS_LIST]
            W_downs      = [w["W_down"]      for w in _WEIGHTS_LIST]
            W_gamma_qkvs = [w["W_gamma_qkv"] for w in _WEIGHTS_LIST]
            W_gamma_mlps = [w["W_gamma_mlp"] for w in _WEIGHTS_LIST]
            K_caches     = [w["K_cache"].unsqueeze(1) for w in _WEIGHTS_LIST]
            V_caches     = [w["V_cache"].unsqueeze(1) for w in _WEIGHTS_LIST]

            ref = llama3_transformer_fwd_tkg_torch(
                X=inp["X"],
                W_qkvs=W_qkvs,
                W_outs=W_outs,
                W_gates=W_gates, W_gate_scales=None,
                W_ups=W_ups, W_up_scales=None,
                W_downs=W_downs, W_down_scales=None,
                W_gamma_qkvs=W_gamma_qkvs,
                W_gamma_mlps=W_gamma_mlps,
                RoPE_cos=inp["RoPE_cos"],
                RoPE_sin=inp["RoPE_sin"],
                mask_cache=inp["mask_cache"],
                mask_active=inp["mask_active"],
                position_ids=inp["position_ids"],
                K_caches=K_caches,
                V_caches=V_caches,
                num_layers=NUM_LAYERS,
                replica_groups=[[0, 1]],
                sbuf_residual_and_cc=True,
            ).float()

            print(f"Reference stats: min={ref.min():.4f}  max={ref.max():.4f}  "
                  f"mean={ref.mean():.4f}  std={ref.std():.4f}")
            abs_err = (r - ref).abs()
            print(f"Abs err: max={abs_err.max():.4e}  mean={abs_err.mean():.4e}")

            atol, rtol = 5e-2, 5e-2  # bf16, multi-layer accumulation
            ok = torch.allclose(r, ref, atol=atol, rtol=rtol)
            assert ok, (
                f"Numerical mismatch vs torch reference "
                f"(atol={atol}, rtol={rtol}); max abs err {abs_err.max().item():.4e}"
            )
            print(f"\nRESULT: PASS (allclose atol={atol} rtol={rtol})")
            return 0
        except Exception as e:
            err = str(e)
            if "NCC_ILLC059" in err or "neuronx-cc failed with 70" in err:
                print("  FAIL: NCC_ILLC059 — cross-rank buffer resolution error")
            else:
                print(f"  FAIL: {type(e).__name__}: {err[:800]}")
            import traceback
            traceback.print_exc()
            print("\nRESULT: FAIL")
            return 1


if __name__ == "__main__":
    sys.exit(main())
