"""
Integration test for transformer_tkg_qwen megakernel.
Single Qwen3-style transformer layer: attention + MoE, SBUF allreduce.

Usage:
  python tests/test_transformer_tkg_qwen.py
"""
import os, sys, tempfile
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch.nn as nn
import nki
from neuronx_distributed.trace import parallel_model_trace

B, S_TKG, H = 1, 1, 2048
D_HEAD  = 128
Q_HEADS = 8       # per TP rank
I_MLP   = 192
E       = 128
S_CTX   = 512     # KV cache len, multiple of 128
TP      = 2

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
        Wq       = torch.randn(Q_HEADS * D_HEAD, H, dtype=dtype) * 0.02,
        Wk       = torch.randn(D_HEAD, H, dtype=dtype) * 0.02,
        Wv       = torch.randn(D_HEAD, H, dtype=dtype) * 0.02,
        Wo       = torch.randn(Q_HEADS * D_HEAD, H, dtype=dtype) * 0.02,  # full [1024,2048]
        q_norm_w = torch.ones(D_HEAD, dtype=dtype),
        k_norm_w = torch.ones(D_HEAD, dtype=dtype),
        K_cache  = torch.zeros(B, 1, S_CTX, D_HEAD, dtype=dtype),
        V_cache  = torch.zeros(B, 1, S_CTX, D_HEAD, dtype=dtype),
        gamma_mlp= torch.ones(1, H, dtype=dtype),
        router_w = torch.randn(H, E, dtype=dtype) * 0.02,
        gate_up_w= torch.randn(E, H, 2 * I_MLP, dtype=dtype) * 0.02,
        down_w   = torch.randn(E, I_MLP, H, dtype=dtype) * 0.02,
    )

class TransformerTKGQwenModule(nn.Module):
    def __init__(self, weights):
        super().__init__()
        for name, t in weights.items():
            self.register_buffer(name, t)

    def forward(self, X, cos, sin, position_ids):
        from kernels.transformer.transformer_tkg_qwen import transformer_tkg_qwen
        return nki.jit(transformer_tkg_qwen)(
            X, cos, sin, position_ids,
            self.Wq, self.Wk, self.Wv, self.Wo,
            self.q_norm_w, self.k_norm_w,
            self.K_cache, self.V_cache,
            self.gamma_mlp, self.router_w, self.gate_up_w, self.down_w,
            replica_groups=([0, 1],),
        )


class _WeightsFactory:
    """Picklable factory that carries weights dict so it works with spawn multiprocessing."""
    def __init__(self, weights):
        self.weights = weights

    def __call__(self):
        m = TransformerTKGQwenModule(self.weights)
        m.eval()
        return m, {}


def make_module_factory():
    weights = make_weights()
    return _WeightsFactory(weights)

def make_inputs(dtype=torch.bfloat16):
    return dict(
        X            = torch.randn(B, S_TKG, H, dtype=dtype) * 0.1,
        cos          = torch.ones(B, D_HEAD, dtype=dtype),
        sin          = torch.zeros(B, D_HEAD, dtype=dtype),
        position_ids = torch.zeros(B, 1, dtype=torch.int32),
    )

def run_test():
    inp = make_inputs()
    example_args = (inp["X"], inp["cos"], inp["sin"], inp["position_ids"])
    factory = make_module_factory()

    with tempfile.TemporaryDirectory(prefix="tkg_qwen_test_") as workdir:
        print("Compiling transformer_tkg_qwen...")
        try:
            traced = parallel_model_trace(
                factory, example_args, tp_degree=TP,
                compiler_workdir=workdir,
                compiler_args=COMPILER_ARGS,
                inline_weights_to_neff=True,
            )
            print("Compile PASS — running inference...")
            result = traced(*example_args)
            print(f"Output shape: {result.shape}")
            assert result.shape == (B, S_TKG, H), f"shape mismatch: {result.shape}"
            r = result.float()
            print(f"Stats: min={r.min():.4f} max={r.max():.4f} mean={r.mean():.4f} std={r.std():.4f}")
            assert torch.isfinite(r).all(), "output contains inf/nan"
            print("PASS")
            return True
        except Exception as e:
            print(f"FAIL: {type(e).__name__}: {str(e)[:600]}")
            import traceback; traceback.print_exc()
            return False

if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)
