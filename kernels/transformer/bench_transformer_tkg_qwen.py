"""
Benchmark script for transformer_tkg_qwen megakernel.

Sets NEURON_RT_INSPECT env vars before any neuron imports, then runs
inference 10 times and reports device metrics via the benchmark harness.
"""
import os, sys

# MUST be set before any neuron/torch_xla imports
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "/home/ubuntu/nki-moe/kernels/transformer/_bench_out"

sys.path.insert(0, "/home/ubuntu/nki-moe")
sys.path.insert(0, "/home/ubuntu/nki-moe/.claude/skills/nki-kernel-optimizer/scripts")

import torch
import torch.nn as nn
import nki
import tempfile
import time
from neuronx_distributed.trace import parallel_model_trace

# Import benchmark harness
from benchmark import wrap_benchmark, nki_benchmark

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
        Wo       = torch.randn(Q_HEADS * D_HEAD, H, dtype=dtype) * 0.02,
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
    """Picklable factory that carries weights dict."""
    def __init__(self, weights):
        self.weights = weights

    def __call__(self):
        m = TransformerTKGQwenModule(self.weights)
        m.eval()
        return m, {}


def make_inputs(dtype=torch.bfloat16):
    return dict(
        X            = torch.randn(B, S_TKG, H, dtype=dtype) * 0.1,
        cos          = torch.ones(B, D_HEAD, dtype=dtype),
        sin          = torch.zeros(B, D_HEAD, dtype=dtype),
        position_ids = torch.zeros(B, 1, dtype=torch.int32),
    )


def run_benchmark():
    inp = make_inputs()
    example_args = (inp["X"], inp["cos"], inp["sin"], inp["position_ids"])
    factory = _WeightsFactory(make_weights())

    with tempfile.TemporaryDirectory(prefix="tkg_qwen_bench_") as workdir:
        print("Compiling transformer_tkg_qwen for benchmarking...")
        traced = parallel_model_trace(
            factory, example_args, tp_degree=TP,
            compiler_workdir=workdir,
            compiler_args=COMPILER_ARGS,
            inline_weights_to_neff=True,
        )
        print("Compile done. Running warmup + benchmark iterations...")

        # Warmup runs
        N_WARMUP = 3
        N_BENCH  = 10
        for i in range(N_WARMUP):
            inp = make_inputs()
            args = (inp["X"], inp["cos"], inp["sin"], inp["position_ids"])
            _ = traced(*args)
        print(f"Warmup ({N_WARMUP} iters) done.")

        # Timed runs
        times_ms = []
        for i in range(N_BENCH):
            inp = make_inputs()
            args = (inp["X"], inp["cos"], inp["sin"], inp["position_ids"])
            t0 = time.perf_counter()
            out = traced(*args)
            # Force sync
            _ = out.cpu()
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            times_ms.append(elapsed_ms)
            print(f"  iter {i+1:2d}: {elapsed_ms:.3f} ms")

        import statistics
        print(f"\n=== Wall-clock Latency ({N_BENCH} iters, after {N_WARMUP} warmup) ===")
        print(f"  min   = {min(times_ms):.3f} ms")
        print(f"  max   = {max(times_ms):.3f} ms")
        print(f"  mean  = {statistics.mean(times_ms):.3f} ms")
        print(f"  median= {statistics.median(times_ms):.3f} ms")
        print(f"  stdev = {statistics.stdev(times_ms):.3f} ms")

        return traced


if __name__ == "__main__":
    run_benchmark()
