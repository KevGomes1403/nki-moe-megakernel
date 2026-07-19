"""Isolated correctness test for the LM head TKG building block.

Exercises ``lm_head_fwd`` (final RMSNorm -> vocab-parallel lm_head matmul -> per-rank greedy argmax)
on real hardware via torch_xla. The kernel serves two call sites that differ only in T: the verify
pass (T=2) and the MTP draft step (T=1). ``output_projection_tkg`` LNC-shards the vocab, so at grid
[2] each core owns V_core = V_rank/2 columns and the two core-local winners are combined by
sendrecv; the harness emits per-core (max, index) rows so the test can also confirm they agree.

Precision is bf16 by design -- the model's lm_head is a bare bf16 torch.matmul -- so the bf16 case is
gated against the measured bf16 quantization floor, not against fp32.

Gates:
  P1 fp32  kernel logits vs fp32 CPU oracle, allclose(atol=1e-5, rtol=1e-2)          HARD
  P1 bf16  kernel logits vs the measured bf16 quantization floor * BF16_HEADROOM
  P2a      kernel token vs argmax of the KERNEL'S OWN logits -- exact. This is the gate that
           actually tests the reduction (including the lowest-index tie-break).
  P2b      kernel token vs the PyTorch lm_head(norm(h)) + argmax path -- exact, EXCEPT a mismatch is
           only a bug when the top-2 logits differ by more than the bf16 floor (else a legitimate tie).

Vocab sizes. At the real V_rank only bf16 on 2 cores fits SBUF (see spec §7.6), so the matrix runs
twice: the production config at the real vocab, and the FULL dtype x T x cores matrix -- including
the fp32 HARD gate at its literal tolerance -- at SMALL_VOCAB, which still exercises the multi-chunk
reduce. Nothing is relaxed but the vocab width.

Run (LNC=2, two logical cores):
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_lm_head_kernel

Single-core cases pin NEURON_RT_VISIBLE_CORES=2.
"""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.qwen3_6_moe.nki_kernels.lm_head.components.lm_head import (  # noqa: E402
    lm_head_fwd,
)

H = 2048  # hidden_size
V_RANK = 62080  # vocab_global(248320) / TP(4); pad_size is 0
# Fits fp32 logits in SBUF on 1 core and still yields >1 reduce chunk there (V_core > 2^14).
SMALL_VOCAB = 20480
EPS = 1e-6  # config.rms_norm_eps

# FP32 hard gate (the literal task tolerance).
ATOL = 1e-5
RTOL = 1e-2
# bf16 gate: allow the kernel error to reach the bf16 output-rounding floor with headroom for the
# fp32 reduction-order difference vs torch.
BF16_HEADROOM = 3.0


def make_inputs(t, vocab, seed):
    """Random fp32 inputs: hidden [1, T, H], gamma = randn(H)*0.02 + 1.0 (standard form, mimicking
    the +1 checkpoint convention), lm_head weight [H, vocab] in TransposedColumnParallelLinear's
    stored [in, out] layout."""
    torch.manual_seed(seed)
    hidden = torch.randn(1, t, H)
    gamma = torch.randn(H) * 0.02 + 1.0
    lm_head_w = torch.randn(H, vocab) * (H**-0.5)
    return hidden, gamma, lm_head_w


def golden(hidden, gamma, lm_head_w, io_dtype):
    """CPU oracle: RMSNorm (plain rsqrt, fp32 accumulate, matching runtime CustomRMSNorm) then the
    lm_head matmul. Inputs are rounded to io_dtype FIRST -- the IO contract -- and the math runs in
    fp32. Returns fp32 logits [T, vocab]."""
    x = hidden.to(io_dtype).float().reshape(-1, H)
    g = gamma.to(io_dtype).float().reshape(-1)
    w = lm_head_w.to(io_dtype).float()
    inv = (x.square().mean(dim=-1, keepdim=True) + EPS).rsqrt()
    normed = (x * inv) * g
    return normed @ w


def run_kernel(hidden, gamma, lm_head_w, io_dtype, cores):
    """Move inputs to the Neuron device, launch on `cores` logical cores, return
    (logits fp32 [T, vocab], max [cores, T] fp32, idx [cores, T] int64)."""
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    hd = hidden.to(io_dtype).contiguous().to(dev)
    gd = gamma.reshape(1, H).to(io_dtype).contiguous().to(dev)
    wd = lm_head_w.to(io_dtype).contiguous().to(dev)
    logits, max_out, idx_out = lm_head_fwd[cores](hd, gd, wd, EPS)
    return logits.float().cpu(), max_out.float().cpu(), idx_out.cpu().long()


def _max_abs(a, b):
    return (a.double() - b.double()).abs().max().item()


def _max_rel(a, b):
    denom = b.double().abs().clamp_min(1e-4)
    return ((a.double() - b.double()).abs() / denom).max().item()


def _check_fp32(name, ker, ref):
    """P1 fp32 HARD gate: torch.allclose at the literal task tolerance."""
    max_abs = _max_abs(ker, ref)
    max_rel = _max_rel(ker, ref)
    ok = torch.allclose(ker.double(), ref.double(), atol=ATOL, rtol=RTOL)
    print(
        f"[{name}] {'PASS' if ok else 'FAIL'}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
        f"(atol={ATOL} rtol={RTOL})"
    )
    assert ok, (
        f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} exceeds atol={ATOL} rtol={RTOL}"
    )


def _check_bf16(name, ker, ref):
    """P1 bf16 gate: max_abs(kernel_bf16 - oracle_fp32) <= bf16 quantization floor * headroom, the
    floor being the max abs change from rounding the SAME oracle tensor to bf16."""
    max_abs = _max_abs(ker, ref)
    floor = _max_abs(ref.to(torch.bfloat16).float(), ref)
    gate = floor * BF16_HEADROOM
    ok = max_abs <= gate
    print(
        f"[{name}] {'PASS' if ok else 'FAIL'}  max_abs={max_abs:.3e}  "
        f"bf16_floor={floor:.3e}  gate={gate:.3e} (floor*{BF16_HEADROOM:g})"
    )
    assert ok, f"{name}: max_abs={max_abs:.3e} exceeds bf16 floor gate {gate:.3e}"


def _check_cores_agree(name, max_out, idx_out):
    """Every logical core must report the same per-rank winner (checks the sendrecv LNC combine)."""
    cores = max_out.shape[0]
    for c in range(1, cores):
        assert torch.equal(idx_out[c], idx_out[0]), (
            f"{name}: core {c} index {idx_out[c].tolist()} != core 0 {idx_out[0].tolist()}"
        )
        assert torch.equal(max_out[c], max_out[0]), (
            f"{name}: core {c} max {max_out[c].tolist()} != core 0 {max_out[0].tolist()}"
        )
    print(f"[{name}] PASS  all {cores} core(s) agree on (max, index)")


def _check_p2a(name, ker_logits, ker_idx, ker_max):
    """P2a: the kernel's token must be the argmax of the kernel's OWN logits -- exact. torch.argmax
    returns the lowest index on ties, which is exactly what the reduction is built to reproduce."""
    want_idx = ker_logits.argmax(dim=-1)
    got_idx = ker_idx[0]
    ok = torch.equal(got_idx, want_idx)
    print(
        f"[{name}] {'PASS' if ok else 'FAIL'}  kernel_idx={got_idx.tolist()}  "
        f"argmax(kernel_logits)={want_idx.tolist()}"
    )
    assert ok, f"{name}: kernel index {got_idx.tolist()} != {want_idx.tolist()}"

    want_max = ker_logits.gather(-1, want_idx.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(ker_max[0], want_max), (
        f"{name}: kernel max {ker_max[0].tolist()} != {want_max.tolist()}"
    )


def _check_p2b(name, ker_idx, ref_logits, io_dtype):
    """P2b: the kernel's token must match the PyTorch lm_head(norm(h)) + argmax path -- exact, unless
    the reference's top-2 logits are within the bf16 quantization floor, in which case which one wins
    is not determined by the kernel and the mismatch is a legitimate tie."""
    ref_idx = ref_logits.argmax(dim=-1)
    got_idx = ker_idx[0]
    if torch.equal(got_idx, ref_idx):
        print(f"[{name}] PASS  kernel_idx={got_idx.tolist()} == argmax(torch_logits)")
        return

    floor = _max_abs(ref_logits.to(torch.bfloat16).float(), ref_logits)
    top2 = ref_logits.topk(2, dim=-1).values
    margins = (top2[:, 0] - top2[:, 1]).abs()
    tie = margins <= floor
    mismatch = got_idx != ref_idx
    unexplained = mismatch & ~tie
    ok = not bool(unexplained.any())
    print(
        f"[{name}] {'PASS (tie)' if ok else 'FAIL'}  kernel_idx={got_idx.tolist()}  "
        f"torch_idx={ref_idx.tolist()}  top2_margin={margins.tolist()}  bf16_floor={floor:.3e} "
        f"[io={io_dtype}]"
    )
    assert ok, (
        f"{name}: token mismatch at positions {unexplained.nonzero().flatten().tolist()} with "
        f"top-2 margin {margins.tolist()} > bf16 floor {floor:.3e}"
    )


def run_case(name, t, cores, io_dtype, seed, vocab=V_RANK):
    hidden, gamma, lm_head_w = make_inputs(t=t, vocab=vocab, seed=seed)
    ref = golden(hidden, gamma, lm_head_w, io_dtype)
    ker_logits, ker_max, ker_idx = run_kernel(hidden, gamma, lm_head_w, io_dtype, cores)

    if io_dtype == torch.float32:
        _check_fp32(f"{name}:P1", ker_logits, ref)
    else:
        _check_bf16(f"{name}:P1", ker_logits, ref)
    _check_cores_agree(f"{name}:cores", ker_max, ker_idx)
    _check_p2a(f"{name}:P2a", ker_logits, ker_idx, ker_max)
    _check_p2b(f"{name}:P2b", ker_idx, ref, io_dtype)


# ---------------------------------------------------------------------------
# pytest entrypoints
#   full vocab: the production config (bf16, 2 cores) -- the only one that fits SBUF at V_RANK
#   small vocab: the complete dtype x T x cores matrix, including the fp32 HARD gate
# ---------------------------------------------------------------------------
def test_full_bf16_t1_c2():
    run_case("full/bf16/T1/c2", t=1, cores=2, io_dtype=torch.bfloat16, seed=7)


def test_full_bf16_t2_c2():
    run_case("full/bf16/T2/c2", t=2, cores=2, io_dtype=torch.bfloat16, seed=8)


def test_small_fp32_t1_c1():
    run_case(
        "small/fp32/T1/c1",
        t=1,
        cores=1,
        io_dtype=torch.float32,
        seed=1,
        vocab=SMALL_VOCAB,
    )


def test_small_fp32_t2_c1():
    run_case(
        "small/fp32/T2/c1",
        t=2,
        cores=1,
        io_dtype=torch.float32,
        seed=2,
        vocab=SMALL_VOCAB,
    )


def test_small_fp32_t1_c2():
    run_case(
        "small/fp32/T1/c2",
        t=1,
        cores=2,
        io_dtype=torch.float32,
        seed=3,
        vocab=SMALL_VOCAB,
    )


def test_small_fp32_t2_c2():
    run_case(
        "small/fp32/T2/c2",
        t=2,
        cores=2,
        io_dtype=torch.float32,
        seed=4,
        vocab=SMALL_VOCAB,
    )


def test_small_bf16_t1_c1():
    run_case(
        "small/bf16/T1/c1",
        t=1,
        cores=1,
        io_dtype=torch.bfloat16,
        seed=5,
        vocab=SMALL_VOCAB,
    )


def test_small_bf16_t2_c1():
    run_case(
        "small/bf16/T2/c1",
        t=2,
        cores=1,
        io_dtype=torch.bfloat16,
        seed=6,
        vocab=SMALL_VOCAB,
    )


def test_small_bf16_t1_c2():
    run_case(
        "small/bf16/T1/c2",
        t=1,
        cores=2,
        io_dtype=torch.bfloat16,
        seed=7,
        vocab=SMALL_VOCAB,
    )


def test_small_bf16_t2_c2():
    run_case(
        "small/bf16/T2/c2",
        t=2,
        cores=2,
        io_dtype=torch.bfloat16,
        seed=8,
        vocab=SMALL_VOCAB,
    )


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ALL CASES PASSED")


if __name__ == "__main__":
    main()
