"""Emit a NEFF for the FUSED DeltaNet conv + gated-delta recurrence TKG kernel for profiling.

Runs ONE config (the way the model invokes it per rank) so neuron-explorer can
capture/replay the NKI kernel's NEFF. Reference math is NOT computed (we only
want the kernel's NEFF). Inputs are the real per-rank decode/verify shapes for
the value-head-sharded fused entrypoints (all f32):
    qkv         (T, conv_dim)     raw in_proj_qkv output, token-major: cat(q,k,v) on free axis
    conv_state  (conv_dim, K-1)   carried conv window
    conv_weight (conv_dim, K)     per-channel taps
    key_dim     python int        q/k segment width (Hk=key_dim//128)
    a, b        (T, Hv)           raw in_proj_a / in_proj_b (token-major, head on free)
    A_log       (Hv,)             per-head decay param
    dt_bias     (Hv,)             per-head decay bias
    init_state  (Hv, 128, 128)    carried recurrent state

Usage:
    python -m models.qwen3_6_moe.tests.profile_fused_kernel <decode|verify> <outdir>
      decode -> deltanet_fused_tkg_fwd        (T=1 commit)
      verify -> deltanet_fused_tkg_fwd_state  (T=2 verify)
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
# Source-level attribution + DMA packet tables for the perf analysis.
os.environ.setdefault("XLA_IR_DEBUG", "1")
os.environ.setdefault("XLA_HLO_DEBUG", "1")
os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch_xla.core.xla_model as xm  # noqa: E402

from models.qwen3_6_moe.nki_kernels.nki_deltanet_fused_tkg import (  # noqa: E402
    deltanet_fused_tkg_fwd,
    deltanet_fused_tkg_fwd_state,
)

CONV_DIM = 2048  # per-rank conv_dim = 2*key_dim + value_dim (TP=4)
KEY_DIM = 512  # per-rank key_dim; value_dim = CONV_DIM - 2*KEY_DIM
K = 4  # linear_conv_kernel_dim
STATE_W = K - 1  # carried conv window width
HV = (CONV_DIM - 2 * KEY_DIM) // 128  # value-head count (8)
D = 128  # head_dim


def main():
    mode = sys.argv[1]  # decode | verify
    T = 1 if mode == "decode" else 2

    torch.manual_seed(0)
    # All f32; qkv is token-major [T, conv_dim], a/b token-major [T, Hv].
    qkv = torch.randn(T, CONV_DIM).float()
    conv_state = torch.randn(CONV_DIM, STATE_W).float()
    conv_weight = torch.randn(CONV_DIM, K).float()
    a = torch.randn(T, HV).float()
    b = torch.randn(T, HV).float()
    A_log = torch.randn(HV).float()
    dt_bias = torch.randn(HV).float()
    init_state = torch.randn(HV, D, D).float()

    dev = xm.xla_device()
    host = (qkv, conv_state, conv_weight, a, b, A_log, dt_bias, init_state)
    qkv_d, conv_state_d, conv_weight_d, a_d, b_d, A_log_d, dt_bias_d, init_state_d = (
        t.contiguous().to(dev) for t in host
    )
    if mode == "decode":
        fn = deltanet_fused_tkg_fwd
    else:
        fn = deltanet_fused_tkg_fwd_state

    # A couple of executions so the NEFF is emitted and warm. [2] shards across both cores.
    for _ in range(3):
        outs = fn[2](
            qkv_d, conv_state_d, conv_weight_d, KEY_DIM,
            a_d, b_d, A_log_d, dt_bias_d, init_state_d,
        )
        xm.mark_step()
    _ = [o.cpu() for o in outs]
    print(f"[profile_fused] emitted NEFF for {mode} T={T} -> {_OUTDIR}")


if __name__ == "__main__":
    main()
