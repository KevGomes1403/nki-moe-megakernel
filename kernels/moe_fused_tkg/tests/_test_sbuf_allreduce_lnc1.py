"""
LNC=1 isolation test for nkc.all_reduce on SBUF.

Key question: does SBUF all_reduce compile and run when NEURON_LOGICAL_NC_CONFIG=2
(the default for trn2 / qwen.py), but without the LNC=2+TP error seen when
parallel_model_trace itself runs under NEURON_LOGICAL_NC_CONFIG=2?

Approach: compile with NEURON_LOGICAL_NC_CONFIG=2 (no override), observe whether
NCC_ILLC059 occurs.  The LNC=2 is the only mode the runtime supports on this machine.
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
# Use default NEURON_LOGICAL_NC_CONFIG (=2, the system default for trn2)
# Do NOT override to LNC=1 here.
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch, torch.nn as nn
import nki, nki.isa as nisa, nki.language as nl, nki.collectives as nkc
from neuronx_distributed.trace import parallel_model_trace
import tempfile

HIDDEN = 2048
PAR = 128
FREE = HIDDEN // PAR

@nki.jit
def sbuf_ar_tp2_lnc_default(x):
    """SBUF all-reduce with default LNC (LNC=2 on trn2)."""
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
        return sbuf_ar_tp2_lnc_default(x.reshape(PAR, FREE)).reshape(1, HIDDEN)

def factory():
    m = M(); m.eval(); return m, {}

def main():
    x = torch.randn(1, HIDDEN, dtype=torch.bfloat16)
    with tempfile.TemporaryDirectory() as d:
        print("Tracing with parallel_model_trace(tp_degree=2) and default LNC=2...")
        try:
            traced = parallel_model_trace(factory, (x,), tp_degree=2,
                compiler_workdir=d,
                compiler_args="--target=trn2 --model-type transformer -O1",
                inline_weights_to_neff=True)
            print("Trace PASS")
            result = traced(x).cpu().float()
            expected = 2.0 * x.float()
            rel = (result - expected).abs().max() / expected.abs().max()
            print(f"rel_diff={rel:.3e}  PASS={rel < 0.02}")
        except Exception as e:
            print(f"FAIL: {type(e).__name__}: {str(e)[:300]}")

if __name__ == "__main__":
    main()
