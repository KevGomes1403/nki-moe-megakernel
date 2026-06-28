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
    z           (T, Hv*128)       optional in_proj_z gate -> enables gated per-head RMSNorm
    gamma       (128,)            optional per-head norm weight -> enables gated per-head RMSNorm

By default this exercises the gated per-head RMSNorm path (z/gamma passed). Pass
"raw" as the 3rd arg to profile the bare recurrence (z/gamma None) for an A/B baseline.

Usage:
    python -m models.qwen3_6_moe.tests.profile_fused_kernel <decode|verify> <outdir> [raw] [bf16]
      decode -> deltanet_fused_tkg_fwd        (T=1 commit)
      verify -> deltanet_fused_tkg_fwd_state  (T=2 verify)
      raw    -> skip z/gamma (no gated RMSNorm) for an A/B baseline
      bf16   -> bf16 inputs (default f32) for an fp32-vs-bf16 perf A/B
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

from models.qwen3_6_moe.nki_kernels.deltanet.decode.fused_layer import (  # noqa: E402
    deltanet_fused_tkg_fwd,
    deltanet_fused_tkg_fwd_state,
)

CONV_DIM = 2048  # per-rank conv_dim = 2*key_dim + value_dim (TP=4)
KEY_DIM = 512  # per-rank key_dim; value_dim = CONV_DIM - 2*KEY_DIM
K = 4  # linear_conv_kernel_dim
STATE_W = K - 1  # carried conv window width
HV = (CONV_DIM - 2 * KEY_DIM) // 128  # value-head count (8)
D = 128  # head_dim
VALUE_DIM = HV * D  # 1024


def main():
    mode = sys.argv[1]  # decode | verify
    flags = sys.argv[3:]
    raw = "raw" in flags
    dtype = torch.bfloat16 if "bf16" in flags else torch.float32
    T = 1 if mode == "decode" else 2

    torch.manual_seed(0)
    # qkv is token-major [T, conv_dim], a/b token-major [T, Hv]. dtype = f32 (default)
    # or bf16 -- matching the model's activation dtype for an fp32-vs-bf16 perf A/B.
    qkv = torch.randn(T, CONV_DIM).to(dtype)
    conv_state = torch.randn(CONV_DIM, STATE_W).to(dtype)
    conv_weight = torch.randn(CONV_DIM, K).to(dtype)
    a = torch.randn(T, HV).to(dtype)
    b = torch.randn(T, HV).to(dtype)
    A_log = torch.randn(HV).to(dtype)
    dt_bias = torch.randn(HV).to(dtype)
    init_state = torch.randn(HV, D, D).to(dtype)
    # Gated per-head RMSNorm inputs (the "with rmsnorm" path): z token-major
    # [T, Hv*128], gamma per-head [128]. Skipped under `raw` for an A/B baseline.
    z = torch.randn(T, VALUE_DIM).to(dtype)
    gamma = torch.randn(D).to(dtype)

    dev = xm.xla_device()
    host = (qkv, conv_state, conv_weight, a, b, A_log, dt_bias, init_state, z, gamma)
    (
        qkv_d,
        conv_state_d,
        conv_weight_d,
        a_d,
        b_d,
        A_log_d,
        dt_bias_d,
        init_state_d,
        z_d,
        gamma_d,
    ) = (t.contiguous().to(dev) for t in host)
    if mode == "decode":
        fn = deltanet_fused_tkg_fwd
    else:
        fn = deltanet_fused_tkg_fwd_state

    # A couple of executions so the NEFF is emitted and warm. [2] shards across both cores.
    for _ in range(3):
        if raw:
            outs = fn[2](
                qkv_d,
                conv_state_d,
                conv_weight_d,
                KEY_DIM,
                a_d,
                b_d,
                A_log_d,
                dt_bias_d,
                init_state_d,
            )
        else:
            outs = fn[2](
                qkv_d,
                conv_state_d,
                conv_weight_d,
                KEY_DIM,
                a_d,
                b_d,
                A_log_d,
                dt_bias_d,
                init_state_d,
                z=z_d,
                gamma=gamma_d,
                eps=1e-6,
            )
        xm.mark_step()
    _ = [o.cpu() for o in outs]
    path = "raw" if raw else "rmsnorm"
    dt = "bf16" if dtype is torch.bfloat16 else "f32"
    print(f"[profile_fused] emitted NEFF for {mode} T={T} ({path}, {dt}) -> {_OUTDIR}")


if __name__ == "__main__":
    main()
