"""Probe: does the fused DeltaNet kernel NaN on padding-heavy input?

The isolated kernel test (test_deltanet_fused_kernel.py) only exercised an
all-real S=128 chunk. The full model instead feeds the kernel a padding-heavy
chunk: N real tokens at the front, the rest zeroed (q/k/v=0, g=0, beta=0 via
the modeling-side valid_mask). The length-sweep showed the model NaNs even at
1 real token, so this probe replicates exactly that kernel-input regime to see
whether the kernel itself produces the NaN (structurally, not via decay).

Run:  NEURON_RT_VISIBLE_CORES=0 python -m models.qwen3_6_moe.tests.probe_kernel_padding
"""

import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.qwen3_6_moe.nki_kernels.nki_deltanet_fused import (  # noqa: E402
    deltanet_fused_chunked_fwd,
    _make_lower_mask,
    _make_identity,
    _make_lower_mask_diag,
)

CHUNK = 128


def make_inputs(n_real, seed=0, d=128, dv=128, S=CHUNK):
    """N real tokens at the front (gentle decay), the rest zeroed exactly as
    the modeling valid_mask does (q/k/v=0, g=0, beta=0)."""
    torch.manual_seed(seed)
    q = torch.randn(S, d)
    k = torch.randn(S, d)
    v = torch.randn(S, dv) * 0.5
    q = torch.nn.functional.normalize(q, p=2, dim=-1) / math.sqrt(d)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    g = -torch.nn.functional.softplus(torch.randn(S)) * 0.1  # gentle
    beta = torch.sigmoid(torch.randn(S))
    if n_real < S:
        # Modeling zeros conv output then l2-norms -> l2norm(0)=0 at padding.
        q[n_real:] = 0.0
        k[n_real:] = 0.0
        v[n_real:] = 0.0
        g[n_real:] = 0.0       # zeroed decay -> exp(0)=1, state preserved
        beta[n_real:] = 0.0    # zeroed write gate
    return q, k, v, g, beta


def run_kernel(q, k, v, g, beta):
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    lm = torch.tensor(_make_lower_mask(), dtype=torch.float32).to(dev)
    idn = torch.tensor(_make_identity(), dtype=torch.float32).to(dev)
    lmd = torch.tensor(_make_lower_mask_diag(), dtype=torch.float32).to(dev)
    out = deltanet_fused_chunked_fwd(
        q.float().contiguous().to(dev),
        k.float().contiguous().to(dev),
        v.float().contiguous().to(dev),
        g[:, None].float().contiguous().to(dev),
        beta[:, None].float().contiguous().to(dev),
        lm, idn, lmd,
    )
    if isinstance(out, (tuple, list)):
        out, state = out[0], out[1]
    else:
        state = None
    out = out.cpu()
    state = state.cpu() if state is not None else None
    return out, state


def main():
    for n_real in (128, 64, 5, 2, 1):
        q, k, v, g, beta = make_inputs(n_real)
        try:
            out, state = run_kernel(q, k, v, g, beta)
            out_nan = bool(torch.isnan(out).any())
            st_nan = bool(torch.isnan(state).any()) if state is not None else None
            # NaN confined to real rows, or leaking into padding rows?
            real_nan = bool(torch.isnan(out[:n_real]).any())
            pad_nan = bool(torch.isnan(out[n_real:]).any()) if n_real < CHUNK else False
            print(f"[n_real={n_real:3d}] out_nan={out_nan} "
                  f"(real_rows_nan={real_nan} pad_rows_nan={pad_nan}) "
                  f"state_nan={st_nan} "
                  f"out_absmax={out[:n_real].abs().max().item():.3e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[n_real={n_real:3d}] EXCEPTION {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
