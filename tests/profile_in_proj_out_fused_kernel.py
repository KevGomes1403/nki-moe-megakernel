"""Emit a NEFF for the FULL end-to-end DeltaNet TKG kernel (input RMSNorm + 4-way projection +
conv + gated-delta recurrence + gated per-head RMSNorm + output projection), SBUF-resident in one
launch, for profiling.

This is the o_proj-FUSED sibling of profile_in_proj_fused_kernel.py: identical inputs/dims plus the
out_w [value_dim, hidden] o_proj weight transpose, swapping to the o_proj-fused entrypoints. Dims are
kept identical to that script so the two profiles are directly comparable (the delta isolates the
gather + transpose + output_projection_tkg matmul that fusion adds). Reference math is NOT computed.

    hidden      (1, T, 2048)      layer input (pre-norm)
    proj_w      (2048, 3088)      fused [H, I] projection weight (qkv|z|a|b concatenated, transposed)
    gamma       (1, 2048)         input RMSNorm weight ([1, H] for qkv_tkg)
    eps         python float      input RMSNorm epsilon
    conv_state  (2048, K-1)       carried conv window
    conv_weight (2048, K)         per-channel taps
    key_dim     python int        q/k segment width
    A_log,dt_bias (8,)            per-head decay params
    init_state  (8, 128, 128)     carried recurrent state
    z_gamma     (128,)            gated per-head RMSNorm weight
    out_w       (1024, 2048)      o_proj weight transpose ([value_dim, hidden])
    z_eps       python float      gated per-head RMSNorm epsilon

Usage:
    python -m models.qwen3_6_moe.tests.profile_in_proj_out_fused_kernel <decode|verify> <outdir> [bf16]
      decode -> deltanet_attention_layer        (T=1 commit)
      verify -> deltanet_attention_layer_state  (T=2 verify)
      bf16   -> bf16 inputs (default f32) for an fp32-vs-bf16 perf A/B
"""

import os
import sys

# Profiling env vars MUST be set before importing torch_xla.
_OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "./output"
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3pre")
os.environ.setdefault("NEURON_CC_FLAGS", "--target trn3pre --lnc 2")
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = _OUTDIR
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
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
    deltanet_attention_layer,
    deltanet_attention_layer_state,
)

HIDDEN = 2048
CONV_DIM = 2048  # per-rank conv_dim = 2*key_dim + value_dim (TP=4)
KEY_DIM = 512
VALUE_DIM = CONV_DIM - 2 * KEY_DIM  # 1024
K = 4
STATE_W = K - 1
HV = VALUE_DIM // 128  # value-head count (8)
HEAD_DIM = 128
NUM_V_HEADS = HV
I_DIM = CONV_DIM + VALUE_DIM + 2 * NUM_V_HEADS  # 3088
EPS = 1e-6


def main():
    mode = sys.argv[1]  # decode | verify
    dtype = torch.bfloat16 if "bf16" in sys.argv[3:] else torch.float32
    T = 1 if mode == "decode" else 2

    torch.manual_seed(0)
    hidden = torch.randn(1, T, HIDDEN).to(dtype)
    w_qkv = torch.randn(CONV_DIM, HIDDEN)
    w_z = torch.randn(VALUE_DIM, HIDDEN)
    w_a = torch.randn(NUM_V_HEADS, HIDDEN)
    w_b = torch.randn(NUM_V_HEADS, HIDDEN)
    proj_w = (
        torch.cat([w_qkv, w_z, w_a, w_b], dim=0).t().contiguous().to(dtype)
    )  # [H, I]
    gamma = torch.randn(1, HIDDEN).to(dtype)
    conv_state = torch.randn(CONV_DIM, STATE_W).to(dtype)
    conv_weight = torch.randn(CONV_DIM, K).to(dtype)
    A_log = torch.randn(HV).to(dtype)
    dt_bias = torch.randn(HV).to(dtype)
    init_state = torch.randn(HV, HEAD_DIM, HEAD_DIM).to(dtype)
    z_gamma = torch.randn(HEAD_DIM).to(dtype)
    out_w = torch.randn(VALUE_DIM, HIDDEN).to(dtype)  # o_proj weight transpose

    dev = xm.xla_device()
    h = hidden.contiguous().to(dev)
    w = proj_w.contiguous().to(dev)
    g = gamma.contiguous().to(dev)
    conv_state_d = conv_state.contiguous().to(dev)
    conv_weight_d = conv_weight.contiguous().to(dev)
    A_log_d = A_log.contiguous().to(dev)
    dt_bias_d = dt_bias.contiguous().to(dev)
    init_state_d = init_state.contiguous().to(dev)
    z_gamma_d = z_gamma.contiguous().to(dev)
    out_w_d = out_w.contiguous().to(dev)

    if mode == "decode":
        fn = deltanet_attention_layer
    else:
        fn = deltanet_attention_layer_state

    # A couple of executions so the NEFF is emitted and warm. [2] shards across both cores.
    for _ in range(3):
        outs = fn[2](
            h,
            w,
            g,
            EPS,
            conv_state_d,
            conv_weight_d,
            KEY_DIM,
            A_log_d,
            dt_bias_d,
            init_state_d,
            z_gamma_d,
            out_w_d,
            EPS,
        )
        xm.mark_step()
    _ = [o.cpu() for o in outs]
    dt = "bf16" if dtype is torch.bfloat16 else "f32"
    print(
        f"[profile_in_proj_out_fused] emitted NEFF for {mode} T={T} ({dt}) -> {_OUTDIR}"
    )


if __name__ == "__main__":
    main()
