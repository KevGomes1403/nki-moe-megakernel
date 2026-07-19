"""Isolated correctness test for the batched DeltaNet TKG NKI kernel (new I/O contract).

Calls `deltanet_tkg_fwd` / `deltanet_tkg_fwd_state` DIRECTLY (no NeuronGatedDeltaNet module) and
compares against a float64 sequential gated-delta recurrence (ground truth) that folds in the glue
the kernel absorbs: l2norm+scale of q/k, GQA head replication (value-head h <- k-head h//rep),
beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias). The kernel emits the RAW per-token recurrence
output (head-major); the output RMSNorm over j and the z-gate (*silu(z)) are now the CALLER's job
(applied in PyTorch downstream) and are NOT part of this golden.

New contract (per rank, bs=1): q,k are [Hk,T,d]; v is [Hv,T,d]; a,b are [T,Hv]; A_log,dt_bias are
[Hv]; init_state is [Hv,d,d]. Outputs: attn_out [T,Hv*d] head-major (raw recurrence output), and
state ([Hv,d,d] final / [T,Hv,d,d] candidates).

Staged simplest -> hardest so a failure localizes:
  1. Hk=Hv=1, S=1            -> single-step decode (rep=1)
  2. Hk=Hv=2, S=2, candidates -> per-position candidate states (rep=1 sanity, isolates GQA)
  3. Hk=4, Hv=8, S=2, cand    -> real GQA shapes (rep=2)
  4. Hk=4, Hv=8, S=3, cand    -> longer block, GQA

Tolerance: atol=1e-5, rtol=1e-2 on every output.

Run (pin to core 0):
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_deltanet_tkg_kernel
"""

import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.qwen3_6_moe.nki_kernels.deltanet.components.recurrence import (  # noqa: E402
    deltanet_tkg_fwd,
    deltanet_tkg_fwd_state,
)

DIM = 128  # d = P_MAX (kernel constraint)
ATOL = 1e-5
RTOL = 1e-2


# ---------------------------------------------------------------------------
# float64 ground truth: per-(value-head) recurrence with the input glue folded in.
# Emits the RAW head-major recurrence output (no output RMSNorm, no z-gate).
# ---------------------------------------------------------------------------
def ref_full(q, k, v, a, b, A_log, dt_bias, init):
    """Reference for the kernel's raw recurrence output, in f64.

    Shapes: q,k (Hk,T,d); v (Hv,T,d); a,b (T,Hv); A_log,dt_bias (Hv,); init (Hv,d,d).
    Returns attn_out (T, Hv*d) head-major and states (T, Hv, d, d). The output RMSNorm over j and
    the z-gate are NOT applied here -- they are the caller's job (PyTorch downstream).
    """
    Hk, T, d = q.shape
    Hv = v.shape[0]
    rep = Hv // Hk
    qd, kd, vd = q.double(), k.double(), v.double()
    ad, bd = a.double(), b.double()
    A_logd, dtb = A_log.double(), dt_bias.double()

    # l2norm + scale.
    qn = torch.nn.functional.normalize(qd, p=2, dim=-1) / math.sqrt(d)
    kn = torch.nn.functional.normalize(kd, p=2, dim=-1)
    # gating.
    beta = torch.sigmoid(bd)  # (T, Hv)
    g = -torch.exp(A_logd) * torch.nn.functional.softplus(ad + dtb)  # (T, Hv)

    states = torch.zeros(Hv, T, d, d, dtype=torch.float64)
    out_raw = torch.zeros(Hv, T, d, dtype=torch.float64)
    for h in range(Hv):
        kh = h // rep
        state = init[h].double().clone()
        for t in range(T):
            state = state * torch.exp(g[t, h])
            kv = kn[kh, t] @ state
            delta = (vd[h, t] - kv) * beta[t, h]
            state = state + torch.outer(kn[kh, t], delta)
            out_raw[h, t] = qn[kh, t] @ state
            states[h, t] = state

    # Raw head-major output: (Hv, T, d) -> (T, Hv, d) -> (T, Hv*d). No RMSNorm, no z-gate.
    attn_out = out_raw.permute(1, 0, 2).contiguous().reshape(T, Hv * d)  # (T, Hv*d)
    states = states.permute(1, 0, 2, 3).contiguous()  # (T, Hv, d, d)
    return attn_out, states


def make_inputs(Hk, Hv, S, seed, d=DIM):
    """Raw, new-contract inputs (pre-glue). Realistic ranges: g via the real formula (<=0 region),
    beta via sigmoid, small init."""
    torch.manual_seed(seed)
    rep = Hv // Hk
    assert Hv % Hk == 0 and rep >= 1
    q = torch.randn(Hk, S, d)
    k = torch.randn(Hk, S, d)
    v = torch.randn(Hv, S, d) * 0.5
    a = torch.randn(S, Hv)
    b = torch.randn(S, Hv)
    A_log = torch.randn(Hv) * 0.5  # exp(A_log) ~ O(1)
    dt_bias = torch.randn(Hv) * 0.5
    init = torch.randn(Hv, d, d) * 0.1
    return q, k, v, a, b, A_log, dt_bias, init


def run_kernel(fn, q, k, v, a, b, A_log, dt_bias, init, cores=1):
    """Move inputs to the Neuron device, launch the @nki.jit kernel on ``cores`` NeuronCores
    (value-head SPMD shard), return CPU tensors. cores=1 is the unsharded single-core path."""
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    args = [
        q.float().contiguous().to(dev),
        k.float().contiguous().to(dev),
        v.float().contiguous().to(dev),
        a.float().contiguous().to(dev),
        b.float().contiguous().to(dev),
        A_log.float().contiguous().to(dev),
        dt_bias.float().contiguous().to(dev),
        init.float().contiguous().to(dev),
    ]
    attn_out, state = fn[cores](*args)
    return attn_out.cpu(), state.cpu()


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


def run_stage_final(name, Hk, Hv, S, seed, cores=1):
    """deltanet_tkg_fwd: check attn_out and the final state."""
    inp = make_inputs(Hk, Hv, S, seed)
    ker_out, ker_final = run_kernel(deltanet_tkg_fwd, *inp, cores=cores)
    ref_out, ref_states = ref_full(*inp)
    _check(f"{name}:attn_out", ker_out, ref_out)
    _check(f"{name}:final_state", ker_final, ref_states[-1])


def run_stage_candidates(name, Hk, Hv, S, seed, cores=1):
    """deltanet_tkg_fwd_state: check attn_out and EVERY per-position candidate state."""
    inp = make_inputs(Hk, Hv, S, seed)
    ker_out, ker_states = run_kernel(deltanet_tkg_fwd_state, *inp, cores=cores)
    ref_out, ref_states = ref_full(*inp)
    _check(f"{name}:attn_out", ker_out, ref_out)
    _check(f"{name}:candidate_states", ker_states, ref_states)


def run_shard_equality(name, Hk, Hv, S, seed):
    """Launch the same inputs single-core and LNC=2 (value-head shard); assert bit-identical
    outputs -- proves the shard reproduces the unsharded result exactly."""
    inp = make_inputs(Hk, Hv, S, seed)
    out1, st1 = run_kernel(deltanet_tkg_fwd_state, *inp, cores=1)
    out2, st2 = run_kernel(deltanet_tkg_fwd_state, *inp, cores=2)
    ok = torch.equal(out1, out2) and torch.equal(st1, st2)
    status = "PASS" if ok else "FAIL"
    print(f"[{name}:cores1_vs_cores2] {status}")
    assert ok, f"{name}: cores=1 and cores=2 outputs differ (shard is not bit-identical)"


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_stage1_single_head_s1():
    run_stage_final("stage1_Hk1Hv1_S1", Hk=1, Hv=1, S=1, seed=0)


def test_stage2_rep1_s2_candidates():
    run_stage_candidates("stage2_Hk2Hv2_S2", Hk=2, Hv=2, S=2, seed=1)


def test_stage3_gqa_s2_candidates():
    run_stage_candidates("stage3_Hk4Hv8_S2", Hk=4, Hv=8, S=2, seed=2, cores=2)


def test_stage4_gqa_s3_candidates():
    run_stage_candidates("stage4_Hk4Hv8_S3", Hk=4, Hv=8, S=3, seed=3, cores=2)


def test_stage3_shard_bit_identical():
    run_shard_equality("stage3_Hk4Hv8_S2", Hk=4, Hv=8, S=2, seed=2)


def main():
    # Stages 1 (Hv=1) and 2: cores=1 -- the n=1 regression (unsharded path byte-identical).
    run_stage_final("stage1_Hk1Hv1_S1", Hk=1, Hv=1, S=1, seed=0, cores=1)
    run_stage_candidates("stage2_Hk2Hv2_S2", Hk=2, Hv=2, S=2, seed=1, cores=1)
    # Stages 3 and 4 (Hk=4, Hv=8): cores=2 -- the real LNC=2 value-head shard (Hv_loc=4, rep=2).
    run_stage_candidates("stage3_Hk4Hv8_S2", Hk=4, Hv=8, S=2, seed=2, cores=2)
    run_stage_candidates("stage4_Hk4Hv8_S3", Hk=4, Hv=8, S=3, seed=3, cores=2)
    # Shard is bit-identical to single-core.
    run_shard_equality("stage3_Hk4Hv8_S2", Hk=4, Hv=8, S=2, seed=2)
    print("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
