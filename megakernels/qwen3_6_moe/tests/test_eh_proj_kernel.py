"""Isolated correctness test for the MTP draft front end (embed -> two RMSNorms -> eh_proj).

Exercises ``eh_proj_fwd`` on real hardware via torch_xla. The component replaces the XLA prologue of
``NeuronMTPHead.draft_step`` -- ``cat([embed_norm(emb), hidden_norm(prev_hidden)], -1) @ eh_proj.T``
-- and returns the tp2013 residual tile the draft decoder layer seeds from; the isolation twin
stores it back through the tp2013 inverse so the host reads natural [B, S, H_out].

The design under test assembles the concatenated normed tile and runs ONE ``qkv_tkg(NO_NORM)`` over
the full 2H=4096 contraction. Every other in-repo call site is H=2048, so the wide call is the
load-bearing thing this test settles.

Every gate runs at BOTH n_prgs=1 and n_prgs=2 (production launches at LNC=2). At n_prgs=1 the
tp2013 shard term is identically zero and ``qkv_tkg`` contracts the whole 2H on one core; at
n_prgs=2 core 0 contracts the embed half and core 1 the hidden half, so only n_prgs=2 evaluates the
shard map and the internal cross-core combine.

tp_degree is 1 throughout, so rg=None and both all-gathers are the identity; the TP branch is NOT
covered (see Note).

Gates (each at T in {1, 2} x n_prgs in {1, 2}):
  G1  fp32 numerics    allclose(kernel_fp32, oracle_fp32, atol=1e-5, rtol=1e-2)   -- the pass gate
  G2  bf16 numerics    max-abs error vs the fp32 oracle, reported informationally

The oracle is the draft_step prologue in fp32 on CPU (never on the XLA device, which would emit a
second NEFF): rms(x) = x * rsqrt(mean(x.float()**2, -1) + eps), each half normed over its own
H=2048, concatenated [embed | hidden], then a single fp32 matmul against eh_w.

Note:
    ``all_gather_embed_h`` with a real replica group is untestable in a single-process launch --
    collectives comms are uninitialized and fail at NEFF load -- so only its rg=None identity branch
    is covered here.

Run (LNC=2; covers both n_prgs=1 and n_prgs=2):
    cd /home/ubuntu/nki-moe && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    NEURON_CC_FLAGS="--target trn2 --lnc 2" \
    python -m megakernels.qwen3_6_moe.tests.test_eh_proj_kernel
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nki_kernels.eh_proj import eh_proj_fwd  # noqa: E402

V = 4096  # fast-iteration vocab; the gather is V-agnostic
H = 2048  # hidden_size; tp_degree=1 here, so the per-rank width is the full width
EPS = 1e-6
T_CASES = (1, 2)  # 1 = MTP draft step, 2 = replay
N_PRGS_CASES = (1, 2)  # 2 is production (LNC=2); 1 degenerates tp2013 to tp102
ATOL, RTOL = 1e-5, 1e-2


def make_inputs(t, dtype, seed):
    """Draft-step inputs at the shapes and scales the real front end sees."""
    torch.manual_seed(seed)
    ids = torch.randint(0, V, (1, t), dtype=torch.int32)
    embed_w = (torch.randn(V, H) * 0.02).to(dtype).contiguous()
    prev_hidden = (torch.randn(1, t, H) * 0.5).to(dtype).contiguous()
    gamma_e = (torch.randn(1, H) * 0.02 + 1.0).to(dtype).contiguous()
    gamma_h = (torch.randn(1, H) * 0.02 + 1.0).to(dtype).contiguous()
    eh_w = (torch.randn(2 * H, H) * 0.02).to(dtype).contiguous()
    return ids, embed_w, prev_hidden, gamma_e, gamma_h, eh_w


def _rms(x, gamma):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * gamma.float()


def oracle(ids, embed_w, prev_hidden, gamma_e, gamma_h, eh_w):
    """``draft_step``'s first three lines, in fp32 on CPU."""
    emb = F.embedding(ids.long().reshape(-1), embed_w)
    hid = prev_hidden.reshape(-1, H)
    combined = torch.cat([_rms(emb, gamma_e), _rms(hid, gamma_h)], dim=-1)
    return (combined @ eh_w.float()).reshape(1, -1, eh_w.shape[1])


def _dev():
    import torch_xla.core.xla_model as xm

    return xm.xla_device()


def run_kernel(args, n_prgs):
    dev = [a.to(_dev()) for a in args]
    return eh_proj_fwd[n_prgs](*dev, EPS, 1).cpu()


def _errors(got, want):
    diff = (got.float() - want).abs()
    denom = want.abs().clamp_min(1e-12)
    return diff.max().item(), (diff / denom).max().item()


def run_case(tag, t, dtype, seed, n_prgs, gate):
    args = make_inputs(t, dtype, seed)
    got = run_kernel(args, n_prgs)
    want = oracle(*args)
    max_abs, max_rel = _errors(got, want)
    # Rounding the oracle to the IO dtype -- the error floor no kernel can beat.
    floor = (want.to(dtype).float() - want).abs().max().item()
    ok = torch.allclose(got.float(), want, atol=ATOL, rtol=RTOL)
    status = ("PASS" if ok else "FAIL") if gate else "INFO"
    print(
        f"[{tag}] {status}  shape={tuple(got.shape)} dtype={got.dtype}  "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} dtype_floor={floor:.3e} "
        f"allclose={ok}"
    )
    if gate:
        assert ok, f"{tag}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_g1_fp32():
    """G1: the pass gate -- fp32 in, fp32 out, allclose(atol=1e-5, rtol=1e-2)."""
    for t in T_CASES:
        for n in N_PRGS_CASES:
            run_case(f"G1/fp32/T{t}/n{n}", t, torch.float32, 11, n, gate=True)


def test_g2_bf16():
    """G2: the production dtype, reported informationally against the fp32 oracle."""
    for t in T_CASES:
        for n in N_PRGS_CASES:
            run_case(f"G2/bf16/T{t}/n{n}", t, torch.bfloat16, 11, n, gate=False)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ALL CASES PASSED")


if __name__ == "__main__":
    main()
