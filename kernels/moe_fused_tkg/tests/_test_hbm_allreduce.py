"""
Test nkc.all_reduce on HBM (shared_hbm) via parallel_model_trace(tp_degree=2, LNC=2).

This tests whether the HBM-based collective compiles and runs correctly.
If SBUF all-reduce is blocked by NCC_ILLC059, HBM all-reduce is the fallback
that still allows fusing the CC op into the NKI kernel graph.
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch, torch.nn as nn
import nki, nki.isa as nisa, nki.language as nl, nki.collectives as nkc
from neuronx_distributed.trace import parallel_model_trace
import tempfile

HIDDEN = 2048
PAR = 128
FREE = HIDDEN // PAR

@nki.jit
def hbm_ar_tp2(x):
    """
    HBM all-reduce: load → SBUF compute (identity here) → store to HBM → CC allreduce on HBM.

    This avoids SBUF cross-rank address resolution (which causes NCC_ILLC059).
    The CC operation works on HBM tensors, which ARE visible to the linker across ranks.

    In a real MoE kernel this would be fused after the down-projection matmul
    result is in SBUF/PSUM, then stored to HBM, then all-reduced in-kernel.
    """
    P, F = x.shape

    # ── Compute on SBUF (identity pass-through here) ──────────────────────
    sbuf_buf = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf_buf, src=x)

    # ── Store result to HBM for CC all-reduce ─────────────────────────────
    hbm_src = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=hbm_src, src=sbuf_buf)

    # ── CC all-reduce on HBM — sum across TP=2 ranks ─────────────────────
    # hbm_ar is an intermediate HBM buffer (not the kernel's returned IO tensor).
    # The collective cannot write to the returned output tensor directly.
    hbm_ar = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nkc.all_reduce(
        srcs=[hbm_src],
        dsts=[hbm_ar],
        replica_group=nkc.ReplicaGroup([[0, 1]]),
        op=nl.add,
    )
    # Copy all-reduce result into the returned output tensor.
    out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=hbm_ar)
    return out

class M(nn.Module):
    def forward(self, x):
        return hbm_ar_tp2(x.reshape(PAR, FREE)).reshape(1, HIDDEN)

def factory():
    m = M(); m.eval(); return m, {}

def main():
    x = torch.randn(1, HIDDEN, dtype=torch.bfloat16)
    with tempfile.TemporaryDirectory() as d:
        print("Testing HBM all-reduce: parallel_model_trace(tp_degree=2, LNC=2)...")
        try:
            traced = parallel_model_trace(factory, (x,), tp_degree=2,
                compiler_workdir=d,
                compiler_args="--target=trn2 --model-type transformer -O1",
                inline_weights_to_neff=True)
            print("Trace / compile PASS")
            result = traced(x).cpu().float()
            expected = 2.0 * x.float()
            rel = (result - expected).abs().max() / expected.abs().max()
            print(f"rel_diff={rel:.3e}  PASS={rel < 0.02}")
        except Exception as e:
            print(f"FAIL: {type(e).__name__}: {str(e)[:400]}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
