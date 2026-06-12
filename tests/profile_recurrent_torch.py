"""Emit an XLA NEFF for the batched PyTorch DeltaNet recurrence (head-to-head).

Mirrors NeuronGatedDeltaNet._recurrent_step's post-l2norm math, vectorized over
(B, H) exactly as the model's decode path runs it -- one XLA graph, all heads in
batched tensor ops. Lets us compare device total_time against the NKI kernel.

Usage:
    python -m models.qwen3_6_moe.tests.profile_recurrent_torch <H> <S> <outdir>
"""

import math
import os
import sys

_OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "./output"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = _OUTDIR
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

import torch  # noqa: E402
import torch_xla.core.xla_model as xm  # noqa: E402

DIM = 128


def step(state, q_t, k_t, v_t, g_t, beta_t):
    """One gated-delta step. state (B,H,Kd,Vd); q/k/v_t (B,H,Kd); g/beta_t (B,H)."""
    ns = state * g_t.exp().unsqueeze(-1).unsqueeze(-1)
    kv = (ns * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv) * beta_t.unsqueeze(-1)
    ns = ns + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    out = (ns * q_t.unsqueeze(-1)).sum(dim=-2)
    return out, ns


def main():
    H = int(sys.argv[1])
    S = int(sys.argv[2])
    B = 1

    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(B, H, S, DIM), p=2, dim=-1) / math.sqrt(DIM)
    k = torch.nn.functional.normalize(torch.randn(B, H, S, DIM), p=2, dim=-1)
    v = torch.randn(B, H, S, DIM) * 0.5
    g = -torch.nn.functional.softplus(torch.randn(B, H, S))
    beta = torch.sigmoid(torch.randn(B, H, S))
    state0 = torch.randn(B, H, DIM, DIM) * 0.1

    dev = xm.xla_device()
    q, k, v, g, beta, state0 = (t.float().to(dev) for t in (q, k, v, g, beta, state0))

    for _ in range(3):
        state = state0
        out = None
        for t in range(S):
            out, state = step(state, q[:, :, t], k[:, :, t], v[:, :, t], g[:, :, t], beta[:, :, t])
        xm.mark_step()
    _ = out.cpu(), state.cpu()
    print(f"[profile_recurrent_torch] emitted NEFF for H={H} S={S} -> {_OUTDIR}")


if __name__ == "__main__":
    main()
