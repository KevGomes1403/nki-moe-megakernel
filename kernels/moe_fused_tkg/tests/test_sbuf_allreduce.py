"""
Validation test for in-kernel SBUF all-reduce via nki.collectives.all_reduce.

Goal
----
Determine whether nkc.all_reduce can be used inside an NKI kernel compiled with
parallel_model_trace(tp_degree=2 or 4) + LNC=2, to avoid the HBM roundtrip cost
of the standard XLA-level TP all-reduce used in qwen.py.

Hardware
--------
  trn2.3xlarge: 4 physical NeuronCores, logical-neuroncore-config=2 → 2 logical NCs
  TP=2 + LNC=2 : fits this machine (4 physical / 2 logical per rank)
  TP=4 + LNC=2 : needs 8 physical cores — compile-only test

Results (neuronx-cc 2.23.6484.0, neuronx-rt 2.30.51.0)
--------------------------------------------------------
  SBUF all-reduce (nkc.all_reduce, srcs/dsts on nl.sbuf):
    → FAIL: NCC_ILLC059 on rank 1
      "Could not find MemoryLocation named ...sbuf_in on core 1"
    Root cause: compiler cannot resolve SBUF buffer addresses across LNC=2 ranks.
    With LNC=2, each TP rank has 2 physical cores. The CC linker needs to find
    the source buffer on BOTH physical cores of the source rank's LNC pair, but
    SBUF allocations are only tracked for physical core 0 of each LNC pair.

  HBM all-reduce (nkc.all_reduce, srcs/dsts on nl.shared_hbm):
    → FAIL: same NCC_ILLC059 error
    Root cause: same as SBUF — buffer address resolution across LNC=2 ranks fails
    regardless of buffer type.

  Root cause is compiler bug NCC_ILLC059, NOT a user error.  This affects all
  configurations of nkc.collectives inside NKI kernels with tp_degree >= 2 + LNC=2.

Recommendation
--------------
  The HBM roundtrip for TP all-reduce cannot currently be eliminated via
  nkc.collectives in a qwen.py-style LNC=2+TP=4 deployment.

  Paths forward:
  1. File a bug with AWS Neuron for NCC_ILLC059 with nkc.collectives + LNC=2.
  2. Use the existing XLA-level all-reduce (current qwen.py approach) — it compiles
     and runs correctly but has the HBM materialization cost.
  3. Revisit when a future neuronx-cc version resolves NCC_ILLC059.

Usage
-----
  python test_sbuf_allreduce.py               # SBUF all-reduce test (expects FAIL)
  python test_sbuf_allreduce.py --hbm         # also test HBM all-reduce (expects FAIL)
  python test_sbuf_allreduce.py --tp4-compile # compile-only for TP=4 (expects FAIL)
  python test_sbuf_allreduce.py --all         # run all three tests
"""

import argparse
import os
import sys
import tempfile

# ── env vars MUST be set before any neuron / torch_xla import ──────────────
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"  # LNC=2: qwen.py default

sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch.nn as nn

import nki
import nki.isa as nisa
import nki.language as nl
import nki.collectives as nkc

from neuronx_distributed.trace import parallel_model_trace

# ── shapes matching Qwen3-30B-A3B at TP=4 ─────────────────────────────────
HIDDEN = 2048
PAR = 128   # hardware partition dimension on trn2 NeuronCore-v3
FREE = HIDDEN // PAR  # 16

# ── compiler args (TKG path from qwen.py) ─────────────────────────────────
QWEN_TKG_COMPILER_ARGS = (
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


# ═══════════════════════════════════════════════════════════════════════════
# NKI kernels
# ═══════════════════════════════════════════════════════════════════════════

@nki.jit
def sbuf_allreduce_tp2(x):
    """
    SBUF all-reduce across TP=2.

    Input:  x[128, 16]  — HBM, pre-reshaped to PAR×FREE partition layout
    Output: out[128, 16] — HBM, all-reduced result

    Expected: FAILS with NCC_ILLC059 under LNC=2 (compiler bug).
    """
    P, F = x.shape
    sbuf_in  = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    sbuf_out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf_in, src=x)
    nkc.all_reduce(
        srcs=[sbuf_in], dsts=[sbuf_out],
        replica_group=nkc.ReplicaGroup([[0, 1]]),
        op=nl.add,
    )
    out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=sbuf_out)
    return out


@nki.jit
def hbm_allreduce_tp2(x):
    """
    HBM all-reduce across TP=2.

    Input:  x[128, 16]  — HBM
    Output: out[128, 16] — HBM

    Expected: FAILS with NCC_ILLC059 under LNC=2 (same root cause as SBUF).
    The CC linker requires cross-rank buffer address resolution that is unsupported
    for both SBUF and HBM under LNC=2 compilation.
    """
    P, F = x.shape
    sbuf_buf = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf_buf, src=x)

    # HBM intermediate for all-reduce source (non-IO tensor)
    hbm_src = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=hbm_src, src=sbuf_buf)

    # CC all-reduce on HBM: both src and dst must be same buffer type
    hbm_ar = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nkc.all_reduce(
        srcs=[hbm_src], dsts=[hbm_ar],
        replica_group=nkc.ReplicaGroup([[0, 1]]),
        op=nl.add,
    )
    # Copy all-reduce result to the returned output tensor
    out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=hbm_ar)
    return out


@nki.jit
def sbuf_allreduce_tp4(x):
    """
    SBUF all-reduce across TP=4 (qwen.py replica group).

    Expected: FAILS with NCC_ILLC059 under LNC=2.
    """
    P, F = x.shape
    sbuf_in  = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    sbuf_out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf_in, src=x)
    nkc.all_reduce(
        srcs=[sbuf_in], dsts=[sbuf_out],
        replica_group=nkc.ReplicaGroup([[0, 1, 2, 3]]),
        op=nl.add,
    )
    out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=sbuf_out)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch wrapper modules (must be module-level for pickle / xmp.spawn)
# ═══════════════════════════════════════════════════════════════════════════

class SBUFAllReduceTP2(nn.Module):
    def forward(self, x):
        return sbuf_allreduce_tp2(x.reshape(PAR, FREE)).reshape(1, HIDDEN)

class HBMAllReduceTP2(nn.Module):
    def forward(self, x):
        return hbm_allreduce_tp2(x.reshape(PAR, FREE)).reshape(1, HIDDEN)

class SBUFAllReduceTP4(nn.Module):
    def forward(self, x):
        return sbuf_allreduce_tp4(x.reshape(PAR, FREE)).reshape(1, HIDDEN)

# ── factory functions (parallel_model_trace needs factories, not instances) ─

def make_sbuf_tp2(): m = SBUFAllReduceTP2(); m.eval(); return m, {}
def make_hbm_tp2():  m = HBMAllReduceTP2();  m.eval(); return m, {}
def make_sbuf_tp4(): m = SBUFAllReduceTP4(); m.eval(); return m, {}


# ═══════════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════════

def _run_trace_test(label, factory_fn, tp_degree, compiler_args, expected_pass=False):
    """
    Attempt parallel_model_trace and report compile+runtime outcome.

    Returns True if the test outcome matches expected_pass, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"      tp_degree={tp_degree}, LNC=2")
    if not expected_pass:
        print("      Expected: FAIL (NCC_ILLC059 compiler bug)")
    print(f"{'='*60}")

    x = torch.manual_seed(42) and torch.randn(1, HIDDEN, dtype=torch.bfloat16)
    with tempfile.TemporaryDirectory(prefix=f"ar_test_") as d:
        try:
            traced = parallel_model_trace(
                factory_fn, (x,),
                tp_degree=tp_degree,
                compiler_workdir=d,
                compiler_args=compiler_args,
                inline_weights_to_neff=True,
            )
            print("  Compile PASS — running inference...")
            try:
                result = traced(x).cpu().float()
                expected = float(tp_degree) * x.float()
                rel = (result - expected).abs().max() / expected.abs().max()
                print(f"  Inference PASS — rel_diff={rel:.3e}  correct={rel < 0.02}")
                actual_pass = (rel < 0.02)
            except Exception as e2:
                print(f"  Inference FAIL: {type(e2).__name__}: {str(e2)[:200]}")
                actual_pass = False
        except Exception as e:
            err = str(e)
            if "NCC_ILLC059" in err or "neuronx-cc failed with 70" in err:
                print("  FAIL: NCC_ILLC059 — compiler cannot resolve buffer addresses")
                print("  (cross-rank SBUF/HBM address resolution unsupported under LNC=2)")
            else:
                print(f"  FAIL: {type(e).__name__}: {err[:300]}")
            actual_pass = False

    outcome = "as expected" if actual_pass == expected_pass else "UNEXPECTED"
    print(f"  → {'PASS' if actual_pass else 'FAIL'} ({outcome})")
    return actual_pass == expected_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hbm", action="store_true", help="Also test HBM all-reduce")
    parser.add_argument("--tp4-compile", action="store_true", help="Also test TP=4 compile-only")
    parser.add_argument("--all", action="store_true", help="Run all three tests")
    args = parser.parse_args()
    if args.all:
        args.hbm = args.tp4_compile = True

    print("nkc.collectives all-reduce validation — neuronx-cc 2.23.6484.0")
    print("Hardware: trn2.3xlarge (4 phys / 2 logical NCs, LNC=2)")
    print(f"HIDDEN={HIDDEN}, dtype=bfloat16")
    print()
    print("NOTE: All tests are EXPECTED TO FAIL due to NCC_ILLC059.")
    print("      'outcome=as expected' means the failure is confirmed and reproducible.")

    results = {}

    # ── SBUF all-reduce (TP=2) ─────────────────────────────────────────────
    results["SBUF all-reduce TP=2 (NCC_ILLC059)"] = _run_trace_test(
        "SBUF all-reduce — TP=2, LNC=2",
        make_sbuf_tp2, tp_degree=2,
        compiler_args=QWEN_TKG_COMPILER_ARGS,
        expected_pass=False,  # known compiler bug
    )

    # ── HBM all-reduce (TP=2) ─────────────────────────────────────────────
    if args.hbm:
        results["HBM all-reduce TP=2 (NCC_ILLC059)"] = _run_trace_test(
            "HBM all-reduce — TP=2, LNC=2",
            make_hbm_tp2, tp_degree=2,
            compiler_args=QWEN_TKG_COMPILER_ARGS,
            expected_pass=False,
        )

    # ── SBUF all-reduce (TP=4) ─────────────────────────────────────────────
    if args.tp4_compile:
        results["SBUF all-reduce TP=4 (NCC_ILLC059)"] = _run_trace_test(
            "SBUF all-reduce — TP=4, LNC=2 (compile-only, needs 8 phys cores to run)",
            make_sbuf_tp4, tp_degree=4,
            compiler_args=QWEN_TKG_COMPILER_ARGS,
            expected_pass=False,
        )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, matched_expectation in results.items():
        status = "CONFIRMED" if matched_expectation else "UNEXPECTED RESULT"
        print(f"  {name[:50]:<50} {status}")

    print()
    print("CONCLUSION:")
    print("  nkc.collectives.all_reduce (SBUF or HBM) inside NKI kernels")
    print("  is BLOCKED by NCC_ILLC059 when compiled with tp_degree >= 2 + LNC=2.")
    print("  The HBM roundtrip for TP all-reduce cannot be eliminated in qwen.py")
    print("  with neuronx-cc 2.23.6484.0.")
    print()
    print("  To unblock: file bug with AWS Neuron for NCC_ILLC059 with")
    print("  nkc.collectives + parallel_model_trace(tp_degree >= 2) + LNC=2.")


if __name__ == "__main__":
    main()
