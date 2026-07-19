"""Isolated correctness test for the fused DeltaNet NKI kernel.

Calls `deltanet_fused_chunked_fwd` DIRECTLY (no NeuronGatedDeltaNet module)
on a single (S=128, d=128) chunk and compares against a pure-PyTorch
sequential recurrence (ground truth). This separates "is the kernel
correctly instantiated + numerically correct" from any module-level
projection / gating bug.

The decay magnitude is swept so we can see exactly where (if anywhere) the
kernel diverges or NaNs, and correlate it with the float32 overflow of the
per-column decay factor exp(-cumsum(g)).

All results print as scalar lines (max_abs_err / has_nan / max_exp_neg_gc)
so they survive a flaky display.

Run:
    python -m models.qwen3_6_moe.tests.test_deltanet_fused_kernel
"""

import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.qwen3_6_moe.nki_kernels.deltanet.prefill.chunked_fused import (  # noqa: E402
    deltanet_fused_chunked_fwd,
    _make_lower_mask,
    _make_identity,
    _make_lower_mask_diag,
)

CHUNK = 128


def ref_sequential(q, k, v, g, beta):
    """Naive per-token gated delta rule. q,k: (S,d); v: (S,dv); g,beta: (S,).

    Matches the kernel's contract: q already l2-normed and scaled by
    1/sqrt(d); k l2-normed; g raw per-token log-decay (<=0); beta in (0,1).
    """
    S, d = q.shape
    dv = v.shape[-1]
    state = torch.zeros(d, dv, dtype=torch.float64)
    out = torch.zeros(S, dv, dtype=torch.float64)
    qd, kd, vd = q.double(), k.double(), v.double()
    gd, bd = g.double(), beta.double()
    for t in range(S):
        state = state * torch.exp(gd[t])
        delta = vd[t] - kd[t] @ state
        state = state + bd[t] * torch.outer(kd[t], delta)
        out[t] = qd[t] @ state
    return out


def make_inputs(seed, g_scale, beta_loc, d=128, dv=128, S=CHUNK):
    torch.manual_seed(seed)
    q = torch.randn(S, d)
    k = torch.randn(S, d)
    v = torch.randn(S, dv) * 0.5
    q = torch.nn.functional.normalize(q, p=2, dim=-1) / math.sqrt(d)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    # raw per-token log-decay, <= 0
    g = -torch.nn.functional.softplus(torch.randn(S)) * g_scale
    beta = torch.sigmoid(torch.randn(S) + beta_loc)
    return q, k, v, g, beta


def run_kernel(q, k, v, g, beta):
    # The @nki.jit kernel must run on an XLA (Neuron) device, not on plain CPU
    # FloatTensors. Move every kernel input to the XLA device as contiguous
    # float32, call the kernel, then bring the output back to CPU before the
    # .double() reference comparison.
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()

    lower_mask = torch.tensor(_make_lower_mask(), dtype=torch.float32)
    identity = torch.tensor(_make_identity(), dtype=torch.float32)
    lower_mask_diag = torch.tensor(_make_lower_mask_diag(), dtype=torch.float32)

    q_d = q.to(dtype=torch.float32).contiguous().to(dev)
    k_d = k.to(dtype=torch.float32).contiguous().to(dev)
    v_d = v.to(dtype=torch.float32).contiguous().to(dev)
    g_d = g[:, None].to(dtype=torch.float32).contiguous().to(dev)
    beta_d = beta[:, None].to(dtype=torch.float32).contiguous().to(dev)
    lower_mask_d = lower_mask.contiguous().to(dev)
    identity_d = identity.contiguous().to(dev)
    lower_mask_diag_d = lower_mask_diag.contiguous().to(dev)

    out = deltanet_fused_chunked_fwd(
        q_d, k_d, v_d,
        g_d, beta_d,
        lower_mask_d, identity_d, lower_mask_diag_d,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.cpu()


def main():
    regimes = [
        ("gentle_decay  ", dict(seed=0, g_scale=0.05, beta_loc=0.0)),
        ("moderate_decay", dict(seed=1, g_scale=0.15, beta_loc=0.0)),
        ("tiny_init_0p02", dict(seed=2, g_scale=0.69, beta_loc=0.0)),  # ~ normal_(0,.02)
        ("strong_decay  ", dict(seed=3, g_scale=1.5, beta_loc=0.0)),
    ]
    for name, kw in regimes:
        q, k, v, g, beta = make_inputs(**kw)
        gc = torch.cumsum(g, dim=0)
        max_exp_neg_gc = torch.exp(-gc).max().item()
        ref = ref_sequential(q, k, v, g, beta)
        try:
            ker = run_kernel(q, k, v, g, beta).double()
            has_nan = bool(torch.isnan(ker).any())
            abs_err = (ker - ref).abs()
            denom = ref.abs().clamp_min(1e-4)
            print(f"[{name}] gc_min={gc.min().item():.1f} "
                  f"max_exp(-gc)={max_exp_neg_gc:.2e} "
                  f"kernel_nan={has_nan} "
                  f"max_abs_err={abs_err.max().item():.3e} "
                  f"max_rel_err={(abs_err/denom).max().item():.3e}")
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] gc_min={gc.min().item():.1f} "
                  f"max_exp(-gc)={max_exp_neg_gc:.2e} EXCEPTION "
                  f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
