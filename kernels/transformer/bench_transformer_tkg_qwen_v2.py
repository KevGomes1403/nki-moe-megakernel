"""
Benchmark script for transformer_tkg_qwen_v2 NKI kernel (H_HALF fix).

Sets environment variables BEFORE any neuron/torch_xla imports.
"""

import os
import sys

# MUST be set before any neuron/torch_xla imports
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v2"
)

sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch.nn as nn
import nki
from neuronx_distributed.trace import parallel_model_trace

from kernels.transformer.transformer_tkg_qwen_v2 import transformer_tkg_qwen_v2
from kernels.transformer.benchmark import wrap_benchmark

# ---------------------------------------------------------------------------
# Fixed shapes (match kernel constants)
# ---------------------------------------------------------------------------
B, S_TKG, H = 1, 1, 2048
D_HEAD  = 128
Q_HEADS = 8        # per TP rank
I_MLP   = 192
E       = 128
K       = 8        # top-K experts
S_CTX   = 512      # KV cache entries
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


def make_weights_and_inputs():
    torch.manual_seed(42)
    weights = dict(
        Wq       = torch.randn(Q_HEADS * D_HEAD, H, dtype=torch.bfloat16) * 0.02,
        Wk       = torch.randn(D_HEAD, H, dtype=torch.bfloat16) * 0.02,
        Wv       = torch.randn(D_HEAD, H, dtype=torch.bfloat16) * 0.02,
        Wo       = torch.randn(Q_HEADS * D_HEAD, H, dtype=torch.bfloat16) * 0.02,
        q_norm_w = torch.ones(D_HEAD, dtype=torch.bfloat16),
        k_norm_w = torch.ones(D_HEAD, dtype=torch.bfloat16),
        K_cache  = torch.zeros(B, 1, S_CTX, D_HEAD, dtype=torch.bfloat16),
        V_cache  = torch.zeros(B, 1, S_CTX, D_HEAD, dtype=torch.bfloat16),
        gamma_mlp= torch.ones(1, H, dtype=torch.bfloat16),
        router_w = torch.randn(H, E, dtype=torch.bfloat16) * 0.02,
        gate_up_w= torch.randn(E, H, 2 * I_MLP, dtype=torch.bfloat16) * 0.02,
        down_w   = torch.randn(E, I_MLP, H, dtype=torch.bfloat16) * 0.02,
    )
    inputs = dict(
        X            = torch.randn(B, S_TKG, H, dtype=torch.bfloat16) * 0.1,
        cos          = torch.ones(B, D_HEAD, dtype=torch.bfloat16),
        sin          = torch.zeros(B, D_HEAD, dtype=torch.bfloat16),
        position_ids = torch.zeros(B, 1, dtype=torch.int32),
    )
    return weights, inputs


class TKGModule(nn.Module):
    def __init__(self, weights):
        super().__init__()
        for k, v in weights.items():
            self.register_buffer(k, v)

    def forward(self, X, cos, sin, position_ids):
        kernel_fn = nki.jit(transformer_tkg_qwen_v2)
        kernel_fn = wrap_benchmark(kernel_fn, warmup=5, iters=50)
        return kernel_fn(
            X, cos, sin, position_ids,
            self.Wq, self.Wk, self.Wv, self.Wo,
            self.q_norm_w, self.k_norm_w,
            self.K_cache, self.V_cache,
            self.gamma_mlp, self.router_w, self.gate_up_w, self.down_w,
            replica_groups=([0, 1],),
        )


class _WeightsFactory:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self):
        m = TKGModule(self.weights)
        m.eval()
        return m, {}


def main():
    print("=" * 60)
    print("transformer_tkg_qwen_v2 benchmark")
    print("=" * 60)

    weights, inputs = make_weights_and_inputs()

    example_args = (
        inputs["X"], inputs["cos"], inputs["sin"], inputs["position_ids"]
    )
    factory = _WeightsFactory(weights)

    workdir = "/tmp/tkg_qwen_v2_bench_workdir"
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"], exist_ok=True)

    print(f"\nCompiling transformer_tkg_qwen_v2 kernel (tp_degree=2), workdir={workdir}...")
    traced = parallel_model_trace(
        factory, example_args, tp_degree=TP,
        compiler_workdir=workdir,
        compiler_args=COMPILER_ARGS,
        inline_weights_to_neff=True,
    )
    print("Compilation SUCCEEDED.")

    print("\nRunning benchmark...")
    traced(*example_args)


if __name__ == "__main__":
    main()
