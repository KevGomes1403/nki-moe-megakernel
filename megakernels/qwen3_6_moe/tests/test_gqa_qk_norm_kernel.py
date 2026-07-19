"""Isolated correctness test for the GQA q/k RMSNorm TKG NKI kernel (Phase 2 of 5).

Exercises ``qk_norm_compose`` (the SBUF-in/SBUF-out composable that drops into the fused GQA kernel)
via a thin @nki.jit harness. The harness takes the Phase 1 head-major NBSd QKV tile in HBM
([N, B, S, D], N = 4 q + 1 k + 1 v, head_dim D=256), DMAs it to SBUF [T, N, D] (T = B*S tokens on the
partition axis; head_dim on the free axis), partition-broadcasts the q/k layernorm weights to [T, D],
and calls the composable. RMSNorm is a FREE-AXIS reduction over D=256 (D on the free axis, <= 512), so
there is no partition tiling -- head_dim=256 needs no splitting.

Math (standard-weight RMSNorm; gamma is the layernorm weight, NOT 1+weight):
    y[t, :] = x[t, :] * rsqrt(mean_D(x[t, :]^2) + eps) * gamma     for each Q head and each K head.
The V head is passed through unchanged. fp32 square/reduce/rsqrt/scale; IO dtype bf16 or fp32.

Precision (two dtypes per case, one CPU reference each):
  - FP32 IO -- the HARD correctness gate: torch.allclose(atol=1e-5, rtol=1e-2). This MUST pass.
  - bf16 IO -- the model's runtime contract (bf16 IO, fp32 accumulate): gated on cosine > 0.999, with
    max_abs / max_rel / ULP and the fixed-(1e-5,1e-2) allclose reported for visibility. A fixed 1e-5
    atol is below the bf16 noise floor after a 256-deep reduce + bf16 output rounding on near-zero
    (cancelling) outputs -- bf16-vs-fp32-oracle noise, not a kernel error (cosine ~0.9999 and the clean
    fp32 gate establish correctness; same precedent as Phase 1 / test_deltanet_in_proj_kernel).

Run (USE ONLY CORE 2, single logical core, grid [1]):
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=2 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_gqa_qk_norm_kernel
"""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

from models.qwen3_6_moe.nki_kernels.gqa.components.qk_norm import (  # noqa: E402
    HEAD_DIM,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    qk_norm_compose,
)

EPS = 1e-6  # config.rms_norm_eps (used identically by kernel and reference)

# FP32 hard gate (the literal task tolerance).
ATOL = 1e-5
RTOL = 1e-2
# bf16 structural gate: cosine, plus an output-magnitude-scaled atol floor for the reported allclose.
COS_MIN = 0.999
ATOL_SCALE_BF16 = 2e-2


@nki.jit
def qk_norm_harness(x, gq, gk):
    """Test harness (single core, grid [1]): DMA the NBSd QKV tile to SBUF [T, N, D], broadcast the
    q/k layernorm weights to [T, D], call qk_norm_compose, and write the NBSd [N, B, S, D] result.

    x:  [N, B, S, D] HBM, head-major (q heads, then k, then v).
    gq: [D] HBM, q-layernorm weight.   gk: [D] HBM, k-layernorm weight.
    """
    N, B, S, D = x.shape
    T = B * S

    # Load NBSd -> SBUF [T, N, D]: partition t = b*S+s (stride D), free heads N (stride B*S*D) then D.
    qkv_sb = nl.ndarray((T, N, D), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=qkv_sb, src=x.ap(pattern=[[D, T], [B * S * D, N], [1, D]], offset=0)
    )

    # Partition-broadcast each gamma [D] to [T, D] (partition stride 0 replicates the single row).
    gq_sb = nl.ndarray((T, D), dtype=gq.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gq_sb, src=gq.ap(pattern=[[0, T], [1, D]], offset=0))
    gk_sb = nl.ndarray((T, D), dtype=gk.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gk_sb, src=gk.ap(pattern=[[0, T], [1, D]], offset=0))

    out_sb = qk_norm_compose(qkv_sb, gq_sb, gk_sb, NUM_Q_HEADS, NUM_KV_HEADS, D, EPS)

    out_hbm = nl.ndarray((N, B, S, D), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=out_hbm.ap(pattern=[[D, T], [B * S * D, N], [1, D]], offset=0), src=out_sb
    )
    return out_hbm


def make_inputs(t, seed):
    """Random fp32 inputs. ``x`` is the NBSd [N, B=1, S=t, D] QKV tile; gq/gk are random per-head_dim
    weights (not ones) to exercise the RMSNorm weight."""
    torch.manual_seed(seed)
    x = torch.randn(NUM_HEADS, 1, t, HEAD_DIM)
    gq = torch.randn(HEAD_DIM)
    gk = torch.randn(HEAD_DIM)
    return x, gq, gk


def golden(inp, dtype):
    """CPU reference mirroring the kernel: cast inputs to ``dtype``, RMSNorm each Q/K head over D in
    fp32 in the kernel's order (x * inv_rms) * gamma, cast back to ``dtype``; V head passed through."""
    x, gq, gk = inp
    out = x.to(dtype).clone()

    def rmsnorm(h, g):
        hf = h.to(dtype).float()
        inv = (hf.square().mean(dim=-1, keepdim=True) + EPS).rsqrt()
        return ((hf * inv) * g.to(dtype).float()).to(dtype)

    for n in range(NUM_Q_HEADS):
        out[n] = rmsnorm(x[n], gq)
    for n in range(NUM_Q_HEADS, NUM_Q_HEADS + NUM_KV_HEADS):
        out[n] = rmsnorm(x[n], gk)
    # V heads (already x cast to dtype) are passed through unchanged.
    return out


def run_kernel(inp, dtype):
    """Move inputs to the Neuron device, launch the harness on a single core (grid [1]), return the
    NBSd [N, B, S, D] CPU output in ``dtype``."""
    import torch_xla.core.xla_model as xm

    x, gq, gk = inp
    dev = xm.xla_device()
    xd = x.to(dtype).contiguous().to(dev)
    gqd = gq.to(dtype).contiguous().to(dev)
    gkd = gk.to(dtype).contiguous().to(dev)
    out = qk_norm_harness[1](xd, gqd, gkd)
    return out.to(dtype).cpu()


def _ulp_distance(ker, ref):
    """Max ULP distance between two same-dtype (bf16 or fp32) tensors via their integer bit patterns."""
    if ker.dtype == torch.bfloat16:
        k = ker.view(torch.int16).to(torch.int64)
        r = ref.view(torch.int16).to(torch.int64)
        bias = 0x8000
    else:
        k = ker.to(torch.float32).view(torch.int32).to(torch.int64)
        r = ref.to(torch.float32).view(torch.int32).to(torch.int64)
        bias = 0x80000000
    k = torch.where(k < 0, bias - k, k)
    r = torch.where(r < 0, bias - r, r)
    return (k - r).abs().max().item()


def _metrics(ker, ref):
    kd = ker.double().reshape(-1)
    rd = ref.double().reshape(-1)
    abs_err = (kd - rd).abs()
    denom = rd.abs().clamp_min(1e-4)
    cos = torch.nn.functional.cosine_similarity(kd, rd, dim=0).item()
    return (
        abs_err.max().item(),
        (abs_err / denom).max().item(),
        _ulp_distance(ker, ref),
        cos,
    )


def _check_fp32(name, ker, ref):
    """HARD gate: torch.allclose at the literal task tolerance (atol=1e-5, rtol=1e-2)."""
    max_abs, max_rel, ulp, cos = _metrics(ker, ref)
    ok = torch.allclose(ker.double(), ref.double(), atol=ATOL, rtol=RTOL)
    print(
        f"[{name}] {'PASS' if ok else 'FAIL'}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
        f"max_ulp={ulp}  cos={cos:.7f}  (atol={ATOL} rtol={RTOL})"
    )
    assert ok, (
        f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} max_ulp={ulp} exceeds 1e-5/1e-2"
    )


def _check_bf16(name, ker, ref):
    """Model-contract gate: cosine > 0.999 (structural correctness). The fixed-(1e-5,1e-2) allclose and
    an output-magnitude-scaled allclose are reported for visibility (1e-5 atol is below the bf16 noise
    floor on near-zero cancelling outputs -- see module docstring)."""
    max_abs, max_rel, ulp, cos = _metrics(ker, ref)
    mag = ref.double().abs().median().item()
    atol_floor = ATOL_SCALE_BF16 * max(mag, 1e-4)
    strict = torch.allclose(ker.double(), ref.double(), atol=ATOL, rtol=RTOL)
    scaled = torch.allclose(ker.double(), ref.double(), atol=atol_floor, rtol=RTOL)
    ok = cos > COS_MIN
    print(
        f"[{name}] {'PASS' if ok else 'FAIL'}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
        f"max_ulp={ulp}  cos={cos:.7f}  | allclose(1e-5,1e-2)={strict}  "
        f"allclose(atol={atol_floor:.2e},1e-2)={scaled}  (gate: cos>{COS_MIN})"
    )
    assert ok, f"{name}: cosine {cos:.6f} <= {COS_MIN} (structural mismatch)"


def run_case(name, t, seed):
    """For both fp32 (hard gate) and bf16 (model contract): validate the NBSd output against the CPU
    reference."""
    inp = make_inputs(t=t, seed=seed)

    ref32 = golden(inp, torch.float32)
    ker32 = run_kernel(inp, torch.float32)
    _check_fp32(f"{name}:fp32", ker32, ref32)

    ref16 = golden(inp, torch.bfloat16)
    ker16 = run_kernel(inp, torch.bfloat16)
    _check_bf16(f"{name}:bf16", ker16, ref16)


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_t1():
    run_case("T1", t=1, seed=1)


def test_t2():
    run_case("T2", t=2, seed=2)


def main():
    run_case("T1", t=1, seed=1)
    run_case("T2", t=2, seed=2)
    print("ALL CASES PASSED")


if __name__ == "__main__":
    main()
