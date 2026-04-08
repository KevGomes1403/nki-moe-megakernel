"""
Does nkc.all_reduce work end-to-end with NEURON_LOGICAL_NC_CONFIG=1?

Earlier result: compile with LNC=1 succeeded (no NCC_ILLC059), but the
NEFF had "Logical Core Size = 2" and runtime failed with LNC mismatch.

This test runs compile AND inference under LNC=1 to confirm the full picture.
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"

sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch, torch.nn as nn
import nki, nki.isa as nisa, nki.language as nl, nki.collectives as nkc
from neuronx_distributed.trace import parallel_model_trace
import tempfile

HIDDEN, PAR, FREE = 2048, 128, 16

@nki.jit
def sbuf_ar(x):
    P, F = x.shape
    s_in  = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    s_out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=s_in, src=x)
    nkc.all_reduce(srcs=[s_in], dsts=[s_out],
                   replica_group=nkc.ReplicaGroup([[0, 1]]), op=nl.add)
    out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=s_out)
    return out

class M(nn.Module):
    def forward(self, x):
        return sbuf_ar(x.reshape(PAR, FREE)).reshape(1, HIDDEN)

def factory(): m = M(); m.eval(); return m, {}

def main():
    x = torch.randn(1, HIDDEN, dtype=torch.bfloat16)
    with tempfile.TemporaryDirectory() as d:
        print("Compiling with NEURON_LOGICAL_NC_CONFIG=1 ...")
        try:
            traced = parallel_model_trace(factory, (x,), tp_degree=2,
                compiler_workdir=d,
                compiler_args="--target=trn2 --model-type transformer -O1",
                inline_weights_to_neff=True)
            print("Compile PASS")
        except Exception as e:
            err = str(e)
            if "NCC_ILLC059" in err or "neuronx-cc failed with 70" in err:
                print("Compile FAIL: NCC_ILLC059")
            else:
                print(f"Compile FAIL: {err[:200]}")
            return

        print("Running inference with NEURON_LOGICAL_NC_CONFIG=1 ...")
        try:
            result = traced(x).cpu().float()
            expected = 2.0 * x.float()
            rel = (result - expected).abs().max() / expected.abs().max()
            print(f"Inference PASS  rel_diff={rel:.3e}  correct={rel < 0.02}")
        except Exception as e:
            err = str(e)
            if "Logical Core Size" in err:
                print(f"Inference FAIL: {err[:300]}")
                print("→ NEFF compiled for LNC=2 but runtime is LNC=1")
            else:
                print(f"Inference FAIL: {err[:300]}")

if __name__ == "__main__":
    main()
