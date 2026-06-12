"""Emit a NEFF for the HBM-in/HBM-out DeltaNet TKG kernel for profiling.

Runs ONE config (the way the model invokes it per rank) so neuron-explorer can
capture/replay the NKI kernel's NEFF. Reference math is NOT computed (we only
want the kernel's NEFF). Inputs are realistic decode/verify shapes.

Usage:
    python -m models.qwen3_6_moe.tests.profile_tkg_kernel <decode|verify> <BH> <S> <outdir>
"""

import math
import os
import sys

# Profiling env vars MUST be set before importing torch_xla.
_OUTDIR = sys.argv[4] if len(sys.argv) > 4 else "./output"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = _OUTDIR
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch_xla.core.xla_model as xm  # noqa: E402

from models.qwen3_6_moe.nki_kernels.nki_deltanet_tkg import (  # noqa: E402
    deltanet_tkg_fwd,
    deltanet_tkg_fwd_state,
)

DIM = 128


def main():
    mode = sys.argv[1]  # decode | verify
    BH = int(sys.argv[2])
    S = int(sys.argv[3])

    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(BH, S, DIM), p=2, dim=-1) / math.sqrt(DIM)
    k = torch.nn.functional.normalize(torch.randn(BH, S, DIM), p=2, dim=-1)
    v = torch.randn(BH, S, DIM) * 0.5
    g = -torch.nn.functional.softplus(torch.randn(BH, S)).unsqueeze(-1)  # (BH, S, 1)
    beta = torch.sigmoid(torch.randn(BH, S)).unsqueeze(-1)  # (BH, S, 1)
    init = torch.randn(BH, DIM, DIM) * 0.1

    dev = xm.xla_device()
    args = [t.float().contiguous().to(dev) for t in (q, k, v, g, beta, init)]
    fn = deltanet_tkg_fwd if mode == "decode" else deltanet_tkg_fwd_state

    # A couple of executions so the NEFF is emitted and warm.
    for _ in range(3):
        out, st = fn(*args)
        xm.mark_step()
    _ = out.cpu(), st.cpu()
    print(f"[profile_tkg] emitted NEFF for {mode} BH={BH} S={S} -> {_OUTDIR}")


if __name__ == "__main__":
    main()
