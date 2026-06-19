"""Emit a NEFF for the HBM-in/HBM-out DeltaNet causal conv1d TKG kernel for profiling.

Runs ONE config (the way the model invokes it per rank) so neuron-explorer can
capture/replay the NKI kernel's NEFF. Reference math is NOT computed (we only
want the kernel's NEFF). Inputs are the real per-rank decode/verify conv shapes
at the NEW contract (qkv token-major [T, conv_dim]):
    qkv        (T, conv_dim)    raw in_proj_qkv output, token-major: cat(q,k,v) on free axis
    conv_state (conv_dim, K-1)  carried window
    conv_weight(conv_dim, K)    per-channel taps

Usage:
    python -m models.qwen3_6_moe.tests.profile_conv_kernel <decode|verify> <outdir>
      decode -> deltanet_conv_tkg_fwd       (T=1 commit)
      verify -> deltanet_conv_tkg_fwd_cand  (T=2 verify)
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

from models.qwen3_6_moe.nki_kernels.nki_deltanet_conv_tkg import (  # noqa: E402
    deltanet_conv_tkg_fwd,
    deltanet_conv_tkg_fwd_cand,
)

CONV_DIM = 2048  # per-rank conv_dim = 2*key_dim + value_dim (TP=4)
KEY_DIM = 512  # per-rank key_dim; value_dim = CONV_DIM - 2*KEY_DIM
K = 4  # linear_conv_kernel_dim
STATE_W = K - 1  # carried window width


def main():
    mode = sys.argv[1]  # decode | verify
    T = 1 if mode == "decode" else 2

    torch.manual_seed(0)
    # New contract: all f32, qkv is token-major [T, conv_dim].
    qkv = torch.randn(T, CONV_DIM).float()
    conv_state = torch.randn(CONV_DIM, STATE_W).float()
    conv_weight = torch.randn(CONV_DIM, K).float()

    dev = xm.xla_device()
    args = [t.contiguous().to(dev) for t in (qkv, conv_state, conv_weight)]
    if mode == "decode":
        fn = deltanet_conv_tkg_fwd
    else:
        fn = deltanet_conv_tkg_fwd_cand

    # A couple of executions so the NEFF is emitted and warm. [2] shards across both cores.
    for _ in range(3):
        qkv_out, last = fn[2](*args, KEY_DIM)
        xm.mark_step()
    _ = qkv_out.cpu(), last.cpu()
    print(f"[profile_conv] emitted NEFF for {mode} T={T} -> {_OUTDIR}")


if __name__ == "__main__":
    main()
