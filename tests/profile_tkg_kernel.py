"""Emit a NEFF for the HBM-in/HBM-out DeltaNet TKG recurrence kernel for profiling.

Runs ONE config (the way the model invokes it per rank) so neuron-explorer can
capture/replay the NKI kernel's NEFF. Reference math is NOT computed (we only
want the kernel's NEFF). Inputs are the real per-rank decode/verify shapes at
the NEW 10-arg contract (all f32):
    q,k (Hk,T,d); v (Hv,T,d); a,b (T,Hv); A_log,dt_bias (Hv,);
    z (T,Hv*d) head-major; norm_weight (d,); init_state (Hv,d,d)

Usage:
    python -m models.qwen3_6_moe.tests.profile_tkg_kernel <decode|verify> <outdir>
      decode -> deltanet_tkg_fwd        (T=1 commit)
      verify -> deltanet_tkg_fwd_state  (T=2 verify, per-position candidate states)
"""

import os
import sys

# Profiling env vars MUST be set before importing torch_xla.
_OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "./output"
# Pre-release trn3 silicon reports its platform as "trn3pre"; the NEFF must be
# compiled for that exact target (and LNC=2, as the model runs) or the runtime
# rejects it as a newer/invalid arch revision.
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3pre")
os.environ.setdefault("NEURON_CC_FLAGS", "--target trn3pre --lnc 2")
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = _OUTDIR
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
# Source-level attribution + DMA packet tables for the perf-bug re-verify.
os.environ.setdefault("XLA_IR_DEBUG", "1")
os.environ.setdefault("XLA_HLO_DEBUG", "1")
os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch_xla.core.xla_model as xm  # noqa: E402

from models.qwen3_6_moe.nki_kernels.deltanet.components.recurrence import (  # noqa: E402
    deltanet_tkg_fwd,
    deltanet_tkg_fwd_state,
)

DIM = 128
HK = 4  # q/k heads per rank
HV = 8  # v heads per rank (rep = HV // HK = 2)


def main():
    mode = sys.argv[1]  # decode | verify
    T = 1 if mode == "decode" else 2

    torch.manual_seed(0)
    # New contract: raw pre-glue inputs, all f32. The kernel folds in l2norm,
    # gating, GQA replication, output RMSNorm, and z-gate.
    q = torch.randn(HK, T, DIM)
    k = torch.randn(HK, T, DIM)
    v = torch.randn(HV, T, DIM) * 0.5
    a = torch.randn(T, HV)
    b = torch.randn(T, HV)
    A_log = torch.randn(HV) * 0.5
    dt_bias = torch.randn(HV) * 0.5
    z = torch.randn(T, HV * DIM) * 0.5
    norm_weight = torch.randn(DIM) * 0.2 + 1.0
    init = torch.randn(HV, DIM, DIM) * 0.1

    dev = xm.xla_device()
    args = [
        t.float().contiguous().to(dev)
        for t in (q, k, v, a, b, A_log, dt_bias, z, norm_weight, init)
    ]
    fn = deltanet_tkg_fwd if mode == "decode" else deltanet_tkg_fwd_state

    # A couple of executions so the NEFF is emitted and warm.
    for _ in range(3):
        out, st = fn(*args)
        xm.mark_step()
    _ = out.cpu(), st.cpu()
    print(f"[profile_tkg] emitted NEFF for {mode} Hk={HK} Hv={HV} T={T} -> {_OUTDIR}")


if __name__ == "__main__":
    main()
