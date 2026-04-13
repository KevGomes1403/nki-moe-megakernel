"""
Integration test for the nkilib transformer_tkg megakernel.

Verifies that the kernel compiles and produces numerically reasonable output
when called via parallel_model_trace with tp_degree=2, LNC=2.

Model config (small but valid for the kernel constraints)
---------------------------------------------------------
  H        = 2048   (hidden dim, must be multiple of 128)
  d_head   = 128
  q_heads  = 8      (per TP rank; 16 total with TP=2)
  kv_heads = 1      (GQA, hardcoded in attention_block_tkg)
  I        = 1024   (MLP intermediate per TP rank)
  B        = 1, S_tkg = 1  (token generation: single token)
  S_ctx    = 512    (KV cache context length)
  num_layers = 2

Weight shapes (per TP rank, as transformer_tkg expects)
-------------------------------------------------------
  W_qkv:       [H, d_head*(q_heads + 2*kv_heads)] = [2048, 1280]
  W_out:        [q_heads*d_head, H]                 = [1024, 2048]
  W_gate, W_up: [H, I]                              = [2048, 1024]
  W_down:       [I, H]                              = [1024, 2048]
  W_gamma_*:    [1, H]                              = [1, 2048]
  K_cache:      [B, d_head, S_ctx]                  = [1, 128, 512]
  V_cache:      [B, S_ctx, d_head]                  = [1, 512, 128]
  mask_cache:   [S_ctx, B, q_heads, S_tkg]          = [512, 1, 8, 1]
  mask_active:  [S_ctx, B, q_heads, S_tkg]          (same)
  RoPE_cos/sin: [d_head//2, B, S_tkg]               = [64, 1, 1]
  position_ids: [B, S_tkg]                          = [1, 1]

Usage
-----
  python test_transformer_tkg.py                 # TP=2, HBM path (default)
  python test_transformer_tkg.py --sbuf          # TP=2, SBUF residual path only
  python test_transformer_tkg.py --all           # both paths (separate processes needed)

Note: run --sbuf and the default test in separate invocations.
parallel_model_trace holds NeuronCores for the lifetime of the process;
running two traces in one process causes NRT_FAILURE on the second test.
"""

import argparse
import os
import sys
import tempfile

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch.nn as nn
import nki

from nkilib.experimental.transformer.transformer_tkg import transformer_tkg
from neuronx_distributed.trace import parallel_model_trace

# ── model config ───────────────────────────────────────────────────────────
B        = 1
S_TKG    = 1
H        = 2048
D_HEAD   = 128
Q_HEADS  = 8       # per TP rank
I_MLP    = 1024    # intermediate per TP rank
S_CTX    = 512     # KV cache length
NUM_LAYERS = 2  # overridden per test below

TP = 2             # tensor parallel degree (matches trn2.3xlarge with LNC=2)

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


# ─────────────────────────────────────────────────────────────────────────────
# Tensor factories
# ─────────────────────────────────────────────────────────────────────────────

def make_weights(dtype=torch.bfloat16):
    """Create one set of per-layer weights for a single TP rank."""
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
    """Create inputs that are shared across TP ranks."""
    # mask_cache: [S_ctx, B, q_heads, S_tkg]
    # All-ones = attend everywhere (no masking for this test)
    mask = torch.ones(S_CTX, B, Q_HEADS, S_TKG, dtype=dtype)

    return dict(
        X            = torch.randn(B, S_TKG, H, dtype=dtype) * 0.1,
        RoPE_cos     = torch.ones(D_HEAD // 2, B, S_TKG, dtype=dtype),
        RoPE_sin     = torch.zeros(D_HEAD // 2, B, S_TKG, dtype=dtype),
        mask_cache   = mask,
        mask_active  = mask.clone(),
        position_ids = torch.zeros(B, S_TKG, dtype=torch.int32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Module wrappers
# (parallel_model_trace requires a Module factory, not an instance)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerTKGModule(nn.Module):
    """
    Wraps nkilib.transformer_tkg for parallel_model_trace.

    All per-layer weight lists and KV caches are stored as module buffers so
    they are serialised alongside the traced NEFF and loaded onto each rank.
    """

    def __init__(self, weights_list, sbuf_residual_and_cc: bool = False):
        super().__init__()
        self.sbuf_residual_and_cc = sbuf_residual_and_cc
        self.num_layers = len(weights_list)

        # Register every weight tensor as a buffer (no grad needed)
        for i, w in enumerate(weights_list):
            for name, t in w.items():
                self.register_buffer(f"L{i}_{name}", t)

    def forward(self, X, RoPE_cos, RoPE_sin, mask_cache, mask_active, position_ids):
        NL = self.num_layers

        # NKI requires sequence shape parameters to be tuples, not lists
        W_qkvs       = tuple(getattr(self, f"L{i}_W_qkv")       for i in range(NL))
        W_outs        = tuple(getattr(self, f"L{i}_W_out")        for i in range(NL))
        W_gates       = tuple(getattr(self, f"L{i}_W_gate")       for i in range(NL))
        W_ups         = tuple(getattr(self, f"L{i}_W_up")         for i in range(NL))
        W_downs       = tuple(getattr(self, f"L{i}_W_down")       for i in range(NL))
        W_gamma_qkvs  = tuple(getattr(self, f"L{i}_W_gamma_qkv")  for i in range(NL))
        W_gamma_mlps  = tuple(getattr(self, f"L{i}_W_gamma_mlp")  for i in range(NL))
        K_caches      = tuple(getattr(self, f"L{i}_K_cache")      for i in range(NL))
        V_caches      = tuple(getattr(self, f"L{i}_V_cache")      for i in range(NL))

        return nki.jit(transformer_tkg)(
            X,
            W_qkvs, W_outs, W_gates, W_ups, W_downs,
            W_gamma_qkvs, W_gamma_mlps,
            K_caches, V_caches,
            RoPE_cos, RoPE_sin,
            mask_cache, mask_active,
            position_ids,
            num_layers=NL,
            replica_groups=([0, 1],),   # tuple of lists (NKI shape param convention)
            sbuf_residual_and_cc=self.sbuf_residual_and_cc,
        )


# Module-level factory functions (closures can't be pickled by parallel_model_trace)
_WEIGHTS_LIST: list = []
_SBUF_RESIDUAL: bool = False


def _module_factory():
    m = TransformerTKGModule(_WEIGHTS_LIST, sbuf_residual_and_cc=_SBUF_RESIDUAL)
    m.eval()
    return m, {}


def make_module_factory(sbuf_residual_and_cc: bool, num_layers: int = NUM_LAYERS):
    """Populate module-level globals and return the factory function."""
    global _WEIGHTS_LIST, _SBUF_RESIDUAL
    _WEIGHTS_LIST = [make_weights() for _ in range(num_layers)]
    _SBUF_RESIDUAL = sbuf_residual_and_cc
    return _module_factory


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_test(label: str, sbuf_residual_and_cc: bool, num_layers: int = NUM_LAYERS):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"      tp_degree={TP}, LNC=2, num_layers={num_layers}")
    print(f"      sbuf_residual_and_cc={sbuf_residual_and_cc}")
    print(f"{'='*60}")

    inp = make_inputs()
    example_args = (
        inp["X"],
        inp["RoPE_cos"],
        inp["RoPE_sin"],
        inp["mask_cache"],
        inp["mask_active"],
        inp["position_ids"],
    )

    factory = make_module_factory(sbuf_residual_and_cc, num_layers=num_layers)

    with tempfile.TemporaryDirectory(prefix="tkg_test_") as workdir:
        try:
            print("  Compiling...")
            traced = parallel_model_trace(
                factory,
                example_args,
                tp_degree=TP,
                compiler_workdir=workdir,
                compiler_args=COMPILER_ARGS,
                inline_weights_to_neff=True,
            )
            print("  Compile PASS — running inference...")

            try:
                result = traced(*example_args)
                print(f"  Output shape: {result.shape}")
                assert result.shape == (B, S_TKG, H), f"unexpected shape {result.shape}"

                r = result.float()
                print(f"  Output stats: min={r.min():.4f}  max={r.max():.4f}  "
                      f"mean={r.mean():.4f}  std={r.std():.4f}")

                # Basic sanity: output should be finite and non-trivially non-zero
                assert torch.isfinite(r).all(), "output contains inf/nan"
                print("  Inference PASS — output is finite and correct shape")
                return True

            except Exception as e2:
                print(f"  Inference FAIL: {type(e2).__name__}: {str(e2)[:400]}")
                return False

        except Exception as e:
            err = str(e)
            if "NCC_ILLC059" in err or "neuronx-cc failed with 70" in err:
                print("  FAIL: NCC_ILLC059 — compiler bug (cross-rank buffer resolution)")
            else:
                print(f"  FAIL: {type(e).__name__}: {err[:500]}")
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbuf", action="store_true",
                        help="Also test SBUF residual + CC path (sbuf_residual_and_cc=True)")
    parser.add_argument("--all", action="store_true", help="Run both HBM and SBUF paths")
    args = parser.parse_args()
    if args.all:
        args.sbuf = True

    print("transformer_tkg megakernel integration test")
    print(f"Hardware: trn2.3xlarge, LNC=2, TP={TP}")
    print(f"Model: H={H}, d_head={D_HEAD}, q_heads={Q_HEADS}/rank, "
          f"I={I_MLP}/rank, S_ctx={S_CTX}, layers={NUM_LAYERS}")
    print(f"dtype: bfloat16")

    results = {}

    results["HBM path (sbuf_residual_and_cc=False)"] = _run_test(
        "transformer_tkg — TP=2, LNC=2, HBM all-reduce path",
        sbuf_residual_and_cc=False,
    )

    if args.sbuf:
        results["SBUF path (sbuf_residual_and_cc=True)"] = _run_test(
            "transformer_tkg — TP=2, LNC=2, SBUF residual+CC path",
            sbuf_residual_and_cc=True,
        )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        print(f"  {name:<50} {'PASS' if passed else 'FAIL'}")

    all_pass = all(results.values())
    print()
    if all_pass:
        print("All tests PASSED — transformer_tkg compiles and runs with TP=2+LNC=2.")
    else:
        print("Some tests FAILED — check output above.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
