"""Isolated correctness test for the batched DeltaNet TKG NKI kernel.

Calls `deltanet_tkg_fwd` / `deltanet_tkg_fwd_state` DIRECTLY (no
NeuronGatedDeltaNet module) and compares against a float64 sequential
gated-delta recurrence (ground truth) that mirrors
`NeuronGatedDeltaNet._recurrent_step`.

Staged from simplest to hardest so a failure localizes:
  1. one head, S=1            -> single-step decode == reference
  2. one head, S=2, states    -> per-position candidate states == reference
  3. all heads (BH=12), S=2   -> batched-over-heads correctness + isolation
  4. batch B>1 (BH=8), S=3    -> multi-batch path

Tolerance: atol=1e-5, rtol=1e-2 on every output (per-token output AND every
candidate state).

Run:
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    python -m models.qwen3_6_moe.tests.test_deltanet_tkg_kernel
"""

import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.qwen3_6_moe.nki_kernels.nki_deltanet_tkg import (  # noqa: E402
    deltanet_tkg_fwd,
    deltanet_tkg_fwd_sbuf,
    deltanet_tkg_fwd_state,
)

DIM = 128  # k_dim = v_dim = P_MAX (kernel constraint)
ATOL = 1e-5
RTOL = 1e-2


# ---------------------------------------------------------------------------
# float64 ground truth: per-(batch, head) sequential gated delta rule.
# Mirrors NeuronGatedDeltaNet._recurrent_step exactly.
# ---------------------------------------------------------------------------
def ref_recurrence(q, k, v, g, beta, init):
    """q,k,v: (S, dim); g,beta: (S,); init: (dim, dim). All comparisons in f64.

    Returns (out (S, dim), states (S, dim, dim)) where states[t] is the
    recurrent state AFTER consuming token t.
    """
    S, dim = q.shape
    qd, kd, vd = q.double(), k.double(), v.double()
    gd, bd = g.double(), beta.double()
    state = init.double().clone()
    out = torch.zeros(S, dim, dtype=torch.float64)
    states = torch.zeros(S, dim, dim, dtype=torch.float64)
    for t in range(S):
        state = state * torch.exp(gd[t])
        kv = kd[t] @ state  # (dim,) : sum_i k[i] * state[i, j]
        delta = (vd[t] - kv) * bd[t]
        state = state + torch.outer(kd[t], delta)  # state[i,j] += k[i]*delta[j]
        out[t] = qd[t] @ state  # sum_i q[i] * state[i, j]
        states[t] = state
    return out, states


def make_inputs(BH, S, seed, dim=DIM):
    """Batched, kernel-contract inputs.

    Returns q,k,v,g_bc,beta_bc (BH,S,dim) and init (BH,dim,dim) for the kernel,
    plus scalar g,beta (BH,S) for the reference. q is l2-normed and scaled by
    1/sqrt(dim); k is l2-normed; g is raw per-token log-decay (<=0).
    """
    torch.manual_seed(seed)
    q = torch.randn(BH, S, dim)
    k = torch.randn(BH, S, dim)
    v = torch.randn(BH, S, dim) * 0.5
    q = torch.nn.functional.normalize(q, p=2, dim=-1) / math.sqrt(dim)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    g = -torch.nn.functional.softplus(torch.randn(BH, S))  # raw log-decay <= 0
    beta = torch.sigmoid(torch.randn(BH, S))
    init = torch.randn(BH, dim, dim) * 0.1
    # Kernel contract: g/beta are one scalar per (head, token) -> (BH, S, 1).
    g_s = g.unsqueeze(-1).contiguous()
    beta_s = beta.unsqueeze(-1).contiguous()
    return q, k, v, g_s, beta_s, init, g, beta


def run_kernel(fn, q, k, v, g_bc, beta_bc, init):
    """Move inputs to the Neuron device, call the @nki.jit kernel, return CPU tensors."""
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    args = [
        q.float().contiguous().to(dev),
        k.float().contiguous().to(dev),
        v.float().contiguous().to(dev),
        g_bc.float().contiguous().to(dev),
        beta_bc.float().contiguous().to(dev),
        init.float().contiguous().to(dev),
    ]
    out, state = fn(*args)
    return out.cpu(), state.cpu()


def _metrics(ker, ref):
    ker = ker.double()
    abs_err = (ker - ref).abs()
    denom = ref.abs().clamp_min(1e-4)
    return abs_err.max().item(), (abs_err / denom).max().item()


def _check(name, ker, ref):
    max_abs, max_rel = _metrics(ker, ref)
    ok = torch.allclose(ker.double(), ref, atol=ATOL, rtol=RTOL)
    status = "PASS" if ok else "FAIL"
    print(f"[{name}] {status}  max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}")
    assert ok, f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} exceeds atol={ATOL} rtol={RTOL}"


def _ref_batched(q, k, v, g, beta, init):
    """Run ref_recurrence per bh; return (out (BH,S,dim), states (BH,S,dim,dim))."""
    BH = q.shape[0]
    outs, sts = [], []
    for bh in range(BH):
        o, s = ref_recurrence(q[bh], k[bh], v[bh], g[bh], beta[bh], init[bh])
        outs.append(o)
        sts.append(s)
    return torch.stack(outs, 0), torch.stack(sts, 0)


def run_stage_final(name, BH, S, seed):
    """deltanet_tkg_fwd: check per-token output and the final state."""
    q, k, v, g_bc, beta_bc, init, g, beta = make_inputs(BH, S, seed)
    ker_out, ker_final = run_kernel(deltanet_tkg_fwd, q, k, v, g_bc, beta_bc, init)
    ref_out, ref_states = _ref_batched(q, k, v, g, beta, init)
    _check(f"{name}:output", ker_out, ref_out)
    _check(f"{name}:final_state", ker_final, ref_states[:, -1])


def run_stage_candidates(name, BH, S, seed):
    """deltanet_tkg_fwd_state: check output and EVERY per-position candidate state."""
    q, k, v, g_bc, beta_bc, init, g, beta = make_inputs(BH, S, seed)
    ker_out, ker_states = run_kernel(
        deltanet_tkg_fwd_state, q, k, v, g_bc, beta_bc, init
    )
    ref_out, ref_states = _ref_batched(q, k, v, g, beta, init)
    _check(f"{name}:output", ker_out, ref_out)
    _check(f"{name}:candidate_states", ker_states, ref_states)


def run_stage_sbuf(name, BH, S, seed):
    """deltanet_tkg_fwd_sbuf: SBUF-output path. Output is the packed (S, BH*dim)
    SBUF layout: element [t, h*dim + j] holds token (bh=h, t)'s output[j].
    Compare to the same-laid-out ref."""
    q, k, v, g_bc, beta_bc, init, g, beta = make_inputs(BH, S, seed)
    ker_out, ker_final = run_kernel(deltanet_tkg_fwd_sbuf, q, k, v, g_bc, beta_bc, init)
    ref_out, ref_states = _ref_batched(q, k, v, g, beta, init)
    # ref_out (BH,S,dim) -> (S, BH*dim): element [t, h*dim + j] = ref_out[h, t, j].
    ref_packed = ref_out.permute(1, 0, 2).reshape(S, BH * ref_out.shape[-1])
    _check(f"{name}:output_sbuf", ker_out, ref_packed)
    _check(f"{name}:final_state", ker_final, ref_states[:, -1])


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_stage1_single_head_s1():
    run_stage_final("stage1_BH1_S1", BH=1, S=1, seed=0)


def test_stage2_single_head_s2_candidates():
    run_stage_candidates("stage2_BH1_S2", BH=1, S=2, seed=1)


def test_stage3_all_heads_s2():
    run_stage_candidates("stage3_BH12_S2", BH=12, S=2, seed=2)


def test_stage4_batch_s3():
    run_stage_candidates("stage4_BH8_S3", BH=8, S=3, seed=3)


def test_stage5_sbuf_output():
    run_stage_sbuf("stage5_BH12_S2_sbuf", BH=12, S=2, seed=4)


def main():
    run_stage_final("stage1_BH1_S1", BH=1, S=1, seed=0)
    run_stage_candidates("stage2_BH1_S2", BH=1, S=2, seed=1)
    run_stage_candidates("stage3_BH12_S2", BH=12, S=2, seed=2)
    run_stage_candidates("stage4_BH8_S3", BH=8, S=3, seed=3)
    run_stage_sbuf("stage5_BH12_S2_sbuf", BH=12, S=2, seed=4)
    print("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
