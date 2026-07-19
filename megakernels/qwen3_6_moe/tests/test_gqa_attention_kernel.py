"""Correctness test for the head_dim=256 GQA token-generation attention.

Validates the VENDORED + PATCHED nkilib ``attention_tkg`` (wrapped by
``components.attention.gqa_attention_d256``) against:
  (1) a torch SDPA-style golden (GQA replicate + causal over prior+active,
      dtype-rounded inputs / fp32 accumulate), AND
  (2) the from-scratch ``attention_fresh_ref.gqa_attention_core`` kernel (a
      device-validated, bit-exact reference) -- the two kernels must agree.

The vendored kernel preserves the AWS bf16 QK^T / online-softmax / P.V compute
path verbatim; only the head_dim-on-partition layout is tiled into
D_TILES = ceil(d_head / 128) partition tiles so d_head=256 fits the PE array.

Contract differences exercised here:
  * The vendored kernel does NOT scale Q (fuse_rope=False) -- the harness PRE-
    SCALES Q by 1/sqrt(d). The fresh kernel scales Q internally, so it gets the
    UNSCALED Q. Both compute the same attention; outputs are cross-checked.
  * The vendored kernel requires curr_sprior (== full KV length L, prior+active)
    to be a MULTIPLE OF 128 (AWS asserts s_prior % 128 == 0 per shard), and a
    multiple of 256 when s_prior is sharded across 2 cores. This is a fundamental
    AWS-kernel requirement, not a limitation of the head_dim patch -- so the L
    values here are the 128-multiples nearest the spec's {~8, ~200, ~600}.
  * The active tokens occupy the LAST s_active slots of the L-length KV; the
    caller supplies the full causal mask (1=keep).

GATES:
  * FP32 IO path: torch.allclose(atol=1e-5, rtol=1e-2) MUST pass (hard gate).
  * bf16 IO path: report max_abs/max_rel/ULP + cosine; pass on allclose OR on the
    documented bf16 absolute floor (near-zero attention outputs make a fixed
    1e-5 atol unreachable in bf16 -- same precedent as the other GQA phases).
  * Cross-check: vendored vs fresh_ref outputs agree (per-dtype floor).

Run (CORES 0,1):
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_gqa_attention_kernel
"""

import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

from models.qwen3_6_moe.nki_kernels.gqa.components.attention import (  # noqa: E402
    gqa_attention_d256,
)
from models.qwen3_6_moe.nki_kernels.gqa.components.attention_fresh_ref import (  # noqa: E402
    gqa_attention_core,
)

# Qwen3.6 per-shard GQA decode dims: head_dim=256 (> 128 partition cap), 4 query
# heads sharing 1 kv head.
HEAD_DIM = 256  # D
Q_HEADS = 4  # query heads per kv head (GQA group)
P_MAX = 128

ATOL = 1e-5
RTOL = 1e-2
# bf16 absolute floor (~1 bf16 ULP at an O(1) attention output: 2^-8 ~= 3.9e-3).
# Attention outputs are weighted averages of V ~ N(0,1); some elements land near
# zero, where a ~1-ULP bf16 absolute difference (PE fp32-accumulate vs torch's
# reduction order, and unnormalized-exp rounding) blows up in RELATIVE terms,
# making a pure rtol gate unreachable for bf16 short of bit-exactness. A case
# passes when it clears allclose OR when the worst absolute error is within this
# floor (<< the O(0.1)+ error any real logic bug would produce).
BF16_ATOL = 4e-3
# Cross-check (vendored vs fresh_ref): the two kernels differ in softmax-
# normalization order (vendored normalizes AFTER P.V on bf16-rounded exp; fresh
# normalizes before), so bf16 cross-diffs sit at the bf16 floor too.
CROSS_BF16_ATOL = 8e-3


def div_ceil(n, d):
    return (n + d - 1) // d


D_TILES = div_ceil(HEAD_DIM, P_MAX)


# ---------------------------------------------------------------------------
# Device harnesses
# ---------------------------------------------------------------------------
@nki.jit
def vendored_harness(q_in, k_active_in, k_prior, v_prior, v_active, mask, T):
    """Harness for the VENDORED attention_tkg composable.

    DMAs q/k_active HBM inputs into the head_dim-tiled SBUF layout, calls
    gqa_attention_d256 (KV streamed from HBM by the AWS kernel), writes the
    head_dim-on-partition SBUF output to HBM. Launch on [cores] programs; the
    kernel auto-selects s_prior sharding from the SPMD grid size.

    q_in [D, H*T] PRE-SCALED by 1/sqrt(D); k_active_in [D, T]; k_prior [1,1,D,L];
    v_prior [1,1,L,D]; v_active [1,1,T,D]; mask [L,1,H,T] uint8. Returns
    out_hbm [D, H*T] (element [d, h*T+t] = attn_out[t,h,d])."""
    D, Tq = q_in.shape
    L = k_prior.shape[3]
    H = Tq // T

    q_sb = nl.ndarray((P_MAX, D_TILES, Tq), dtype=q_in.dtype, buffer=nl.sbuf)
    for dt in range(D_TILES):
        d_sz = min(P_MAX, D - dt * P_MAX)
        nisa.dma_copy(
            dst=q_sb[0:d_sz, dt, :], src=q_in[dt * P_MAX : dt * P_MAX + d_sz, :]
        )

    Tk = k_active_in.shape[1]
    k_active_sb = nl.ndarray(
        (P_MAX, D_TILES, Tk), dtype=k_active_in.dtype, buffer=nl.sbuf
    )
    for dt in range(D_TILES):
        d_sz = min(P_MAX, D - dt * P_MAX)
        nisa.dma_copy(
            dst=k_active_sb[0:d_sz, dt, :],
            src=k_active_in[dt * P_MAX : dt * P_MAX + d_sz, :],
        )

    out_sb = nl.ndarray((P_MAX, D_TILES, Tq), dtype=q_in.dtype, buffer=nl.sbuf)
    gqa_attention_d256(
        q_sb=q_sb,
        k_active_sb=k_active_sb,
        k_prior=k_prior,
        v_prior=v_prior,
        v_active=v_active,
        mask=mask,
        out_sb=out_sb,
        bs=1,
        q_head=H,
        s_active=T,
        curr_sprior=L,
        head_dim=D,
    )

    out_hbm = nl.ndarray((D, Tq), dtype=q_in.dtype, buffer=nl.shared_hbm)
    for dt in range(D_TILES):
        d_sz = min(P_MAX, D - dt * P_MAX)
        nisa.dma_copy(
            dst=out_hbm[dt * P_MAX : dt * P_MAX + d_sz, :], src=out_sb[0:d_sz, dt, :]
        )
    return out_hbm


@nki.jit
def fresh_harness(q_in, k_in, v_in, T):
    """Harness for the from-scratch fresh_ref gqa_attention_core (head-sharded).

    q_in [D, H*T] UNSCALED (the fresh kernel scales by 1/sqrt(D) internally);
    k_in [D, L]; v_in [L, D]. Returns out_hbm [D, H*T]."""
    D, Tq = q_in.shape
    L = k_in.shape[1]
    n = nl.num_programs(0)
    c = nl.program_id(0)

    L_TILES = div_ceil(L, P_MAX)
    qh = Tq // T
    qh_local = qh // n
    tq_local = T * qh_local
    col0 = c * tq_local

    q_sb = nl.ndarray((P_MAX, D_TILES, tq_local), dtype=q_in.dtype, buffer=nl.sbuf)
    for dt in range(D_TILES):
        d_sz = min(P_MAX, D - dt * P_MAX)
        nisa.dma_copy(
            dst=q_sb[0:d_sz, dt, 0:tq_local],
            src=q_in[dt * P_MAX : dt * P_MAX + d_sz, col0 : col0 + tq_local],
        )

    k_sb = nl.ndarray((P_MAX, D_TILES, L), dtype=k_in.dtype, buffer=nl.sbuf)
    for dt in range(D_TILES):
        d_sz = min(P_MAX, D - dt * P_MAX)
        nisa.dma_copy(
            dst=k_sb[0:d_sz, dt, 0:L], src=k_in[dt * P_MAX : dt * P_MAX + d_sz, 0:L]
        )

    v_sb = nl.ndarray((P_MAX, L_TILES, D), dtype=v_in.dtype, buffer=nl.sbuf)
    for lc in range(L_TILES):
        c0 = lc * P_MAX
        cp = min(P_MAX, L - c0)
        nisa.dma_copy(dst=v_sb[0:cp, lc, 0:D], src=v_in[c0 : c0 + cp, 0:D])

    out_sb = gqa_attention_core(q_sb, k_sb, v_sb, T)

    out_hbm = nl.ndarray((D, Tq), dtype=q_in.dtype, buffer=nl.shared_hbm)
    for dt in range(D_TILES):
        d_sz = min(P_MAX, D - dt * P_MAX)
        nisa.dma_copy(
            dst=out_hbm[dt * P_MAX : dt * P_MAX + d_sz, col0 : col0 + tq_local],
            src=out_sb[0:d_sz, dt, 0:tq_local],
        )
    return out_hbm


# ---------------------------------------------------------------------------
# Inputs / reference
# ---------------------------------------------------------------------------
def make_inputs(T, L, seed):
    """Random fp32 logical inputs. q [T,H,D]; full K/V [L,D] (prior then active).

    The active tokens occupy the last T rows of the L-length KV (prior_len = L-T).
    L must be a multiple of 128 (AWS kernel requirement)."""
    assert L % P_MAX == 0, f"L must be a multiple of {P_MAX}, got {L}"
    assert L > T, f"L (={L}) must exceed T (={T})"
    torch.manual_seed(seed)
    q = torch.randn(T, Q_HEADS, HEAD_DIM)
    k_full = torch.randn(L, HEAD_DIM)
    v_full = torch.randn(L, HEAD_DIM)
    return q, k_full, v_full


def to_vendored_inputs(inp, T, torch_dtype):
    """Build the vendored kernel's HBM inputs (Q pre-scaled by 1/sqrt(D))."""
    q, k_full, v_full = inp
    L = k_full.shape[0]
    scale = 1.0 / math.sqrt(HEAD_DIM)
    # q_in[d, h*T+t] = (scale * Q[t,h,d]); pre-scaled because fuse_rope=False.
    q_in = (
        (q * scale)
        .permute(2, 1, 0)
        .reshape(HEAD_DIM, Q_HEADS * T)
        .contiguous()
        .to(torch_dtype)
    )
    k_active_in = (
        k_full[L - T :, :].transpose(0, 1).contiguous().to(torch_dtype)
    )  # [D, T]
    k_prior = (
        k_full.transpose(0, 1).reshape(1, 1, HEAD_DIM, L).contiguous().to(torch_dtype)
    )
    v_prior = v_full.reshape(1, 1, L, HEAD_DIM).contiguous().to(torch_dtype)
    v_active = (
        v_full[L - T :, :].reshape(1, 1, T, HEAD_DIM).contiguous().to(torch_dtype)
    )
    mask = build_mask(T, L)  # [L,1,H,T] uint8
    return q_in, k_active_in, k_prior, v_prior, v_active, mask


def to_fresh_inputs(inp, torch_dtype):
    """Build the fresh kernel's HBM inputs (Q UNSCALED; kernel scales internally)."""
    q, k_full, v_full = inp
    q_in = (
        q.permute(2, 1, 0)
        .reshape(HEAD_DIM, Q_HEADS * q.shape[0])
        .contiguous()
        .to(torch_dtype)
    )
    k_in = k_full.transpose(0, 1).contiguous().to(torch_dtype)  # [D, L]
    v_in = v_full.contiguous().to(torch_dtype)  # [L, D]
    return q_in, k_in, v_in


def build_mask(T, L):
    """Causal mask [L,1,H,T] uint8 (1=keep). Active token t (slot L-T+t) attends
    key slot j iff j <= (L-T)+t; the whole prior is visible. Same for all heads."""
    prior_len = L - T
    j = torch.arange(L).view(L, 1)
    t = torch.arange(T).view(1, T)
    keep = (j <= (prior_len + t)).to(torch.uint8)  # [L, T]
    mask = keep.view(L, 1, 1, T).expand(L, 1, Q_HEADS, T).contiguous()
    return mask


def golden(inp, T, torch_dtype):
    """SDPA-style reference that MIRRORS the vendored kernel's precision sequence so
    the gate atol=1e-5/rtol=1e-2 holds for bf16 too (the only residual is matmul
    reduction order, <= ~1 bf16 ULP = 0.39% < rtol). For fp32 it is the exact
    reference. Sequence (matches attention_tkg, fuse_rope=False, non-FA):
      scores = (bf16(Q*scale)) @ bf16(K)              [fp32 accumulate]
      mask -> max -> e = exp(scores - max)            [fp32]
      e_b  = bf16(e)                                  [qk_io_type rounding]
      s    = sum_j e_b                                [fp32 sum of bf16 exp]
      out  = (e_b @ bf16(V)) / s                      [normalize AFTER P.V]
    Causal: active token t (slot L-T+t) attends key j <= (L-T)+t."""
    q, k_full, v_full = inp
    L = k_full.shape[0]
    prior_len = L - T
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # Q is pre-scaled by 1/sqrt(d) BEFORE rounding (the harness rounds q*scale).
    qs = (q * scale).to(torch_dtype).float()
    kb = k_full.to(torch_dtype).float()
    vb = v_full.to(torch_dtype).float()

    l_idx = torch.arange(L).view(1, L)
    thresh = (prior_len + torch.arange(T)).view(T, 1)
    addmask = torch.where(l_idx <= thresh, 0.0, float("-inf"))  # [T, L]

    out = torch.empty(T, Q_HEADS, HEAD_DIM)
    for h in range(Q_HEADS):
        scores = (qs[:, h, :] @ kb.transpose(0, 1)) + addmask  # [T, L] fp32
        m = scores.max(dim=-1, keepdim=True).values
        e = torch.exp(scores - m)  # [T, L] fp32
        e_b = e.to(torch_dtype).float()  # bf16-rounded exp (qk_io_type)
        s = e_b.sum(dim=-1, keepdim=True)  # fp32 sum of bf16 exp
        out[:, h, :] = (e_b @ vb) / s  # P.V (fp32) then normalize
    return out.to(torch_dtype)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_vendored(inp, T, cores, torch_dtype):
    import torch_xla.core.xla_model as xm

    q_in, k_active_in, k_prior, v_prior, v_active, mask = to_vendored_inputs(
        inp, T, torch_dtype
    )
    dev = xm.xla_device()
    out_hbm = vendored_harness[cores](
        q_in.to(dev),
        k_active_in.to(dev),
        k_prior.to(dev),
        v_prior.to(dev),
        v_active.to(dev),
        mask.to(dev),
        T,
    )
    out = (
        out_hbm.to(torch_dtype)
        .cpu()
        .reshape(HEAD_DIM, Q_HEADS, T)
        .permute(2, 1, 0)
        .contiguous()
    )
    return out


def run_fresh(inp, T, cores, torch_dtype):
    import torch_xla.core.xla_model as xm

    q_in, k_in, v_in = to_fresh_inputs(inp, torch_dtype)
    dev = xm.xla_device()
    out_hbm = fresh_harness[cores](q_in.to(dev), k_in.to(dev), v_in.to(dev), T)
    out = (
        out_hbm.to(torch_dtype)
        .cpu()
        .reshape(HEAD_DIM, Q_HEADS, T)
        .permute(2, 1, 0)
        .contiguous()
    )
    return out


# ---------------------------------------------------------------------------
# Metrics / checks
# ---------------------------------------------------------------------------
def _ulp_distance(ker, ref):
    k = ker.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    r = ref.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    k = torch.where(k < 0, 0x8000 - k, k)
    r = torch.where(r < 0, 0x8000 - r, r)
    return (k - r).abs().max().item()


def _metrics(ker, ref):
    kd = ker.double()
    rd = ref.double()
    abs_err = (kd - rd).abs()
    denom = rd.abs().clamp_min(1e-4)
    cos = torch.nn.functional.cosine_similarity(
        kd.reshape(-1), rd.reshape(-1), dim=0
    ).item()
    return (
        abs_err.max().item(),
        (abs_err / denom).max().item(),
        _ulp_distance(ker, ref),
        cos,
    )


def _check(name, ker, ref, floor, is_fp32):
    max_abs, max_rel, ulp, cos = _metrics(ker, ref)
    allclose = torch.allclose(ker.double(), ref.double(), atol=ATOL, rtol=RTOL)
    if is_fp32:
        ok = allclose
        gate = "allclose(fp32-gate)" if allclose else "FAILED"
    else:
        ok = allclose or (max_abs <= floor)
        gate = "allclose" if allclose else ("bf16-floor" if ok else "FAILED")
    status = "PASS" if ok else "FAIL"
    print(
        f"[{name}] {status} via {gate}  max_abs_err={max_abs:.3e}  "
        f"max_rel_err={max_rel:.3e}  max_ulp={ulp}  cos={cos:.6f}  "
        f"(atol={ATOL} rtol={RTOL} floor={floor})"
    )
    assert ok, (
        f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} max_ulp={ulp} "
        f"cos={cos:.6f} exceeds the gate"
    )


def run_case(name, T, L, cores, seed, torch_dtype, cross_check=True):
    """Run vendored (and fresh_ref) for T tokens, KV length L on `cores` cores;
    assert vendored matches the golden and (optionally) the fresh_ref kernel."""
    is_fp32 = torch_dtype == torch.float32
    floor = 0.0 if is_fp32 else BF16_ATOL
    inp = make_inputs(T=T, L=L, seed=seed)
    ref = golden(inp, T, torch_dtype)

    ker = run_vendored(inp, T, cores=cores, torch_dtype=torch_dtype)
    _check(f"{name}/vendored-vs-golden", ker, ref, floor, is_fp32)

    if cross_check:
        fresh = run_fresh(inp, T, cores=cores, torch_dtype=torch_dtype)
        cross_floor = 0.0 if is_fp32 else CROSS_BF16_ATOL
        _check(f"{name}/vendored-vs-fresh", ker, fresh, cross_floor, is_fp32)
    return ker


# ---------------------------------------------------------------------------
# Case table. L is a multiple of 128 (AWS requirement); 256-multiple for the
# s_prior-sharded cores=2 cases (L >= 256). Covers single + multi KV tile and
# (in main) a multi-chunk MM1 case (L > 4096).
# ---------------------------------------------------------------------------
_FP32 = torch.float32
_BF16 = torch.bfloat16


# pytest entrypoints (fp32 hard gate)
def test_fp32_t1_l128_cores1():
    run_case("fp32_T1_L128_c1", T=1, L=128, cores=1, seed=1, torch_dtype=_FP32)


def test_fp32_t2_l128_cores1():
    run_case("fp32_T2_L128_c1", T=2, L=128, cores=1, seed=2, torch_dtype=_FP32)


def test_fp32_t1_l128_cores2():
    run_case("fp32_T1_L128_c2", T=1, L=128, cores=2, seed=1, torch_dtype=_FP32)


def test_fp32_t2_l256_cores2():
    run_case("fp32_T2_L256_c2", T=2, L=256, cores=2, seed=4, torch_dtype=_FP32)


def test_fp32_t1_l256_cores1():
    run_case("fp32_T1_L256_c1", T=1, L=256, cores=1, seed=3, torch_dtype=_FP32)


def test_fp32_t2_l512_cores2():
    run_case("fp32_T2_L512_c2", T=2, L=512, cores=2, seed=5, torch_dtype=_FP32)


def test_fp32_t1_l640_cores1():
    run_case("fp32_T1_L640_c1", T=1, L=640, cores=1, seed=6, torch_dtype=_FP32)


# pytest entrypoints (bf16 metrics, floor gate)
def test_bf16_t1_l128_cores1():
    run_case("bf16_T1_L128_c1", T=1, L=128, cores=1, seed=1, torch_dtype=_BF16)


def test_bf16_t2_l256_cores2():
    run_case("bf16_T2_L256_c2", T=2, L=256, cores=2, seed=4, torch_dtype=_BF16)


def test_bf16_t1_l640_cores1():
    run_case("bf16_T1_L640_c1", T=1, L=640, cores=1, seed=6, torch_dtype=_BF16)


def main():
    print("=== FP32 IO (hard gate: allclose atol=1e-5 rtol=1e-2) ===")
    run_case("fp32_T1_L128_c1", T=1, L=128, cores=1, seed=1, torch_dtype=_FP32)
    run_case("fp32_T2_L128_c1", T=2, L=128, cores=1, seed=2, torch_dtype=_FP32)
    run_case("fp32_T1_L128_c2", T=1, L=128, cores=2, seed=1, torch_dtype=_FP32)
    run_case("fp32_T2_L256_c2", T=2, L=256, cores=2, seed=4, torch_dtype=_FP32)
    run_case("fp32_T1_L256_c1", T=1, L=256, cores=1, seed=3, torch_dtype=_FP32)
    run_case("fp32_T2_L512_c2", T=2, L=512, cores=2, seed=5, torch_dtype=_FP32)
    run_case("fp32_T1_L640_c1", T=1, L=640, cores=1, seed=6, torch_dtype=_FP32)

    print("\n=== bf16 IO (metrics + bf16-floor gate) ===")
    run_case("bf16_T1_L128_c1", T=1, L=128, cores=1, seed=1, torch_dtype=_BF16)
    run_case("bf16_T2_L128_c1", T=2, L=128, cores=1, seed=2, torch_dtype=_BF16)
    run_case("bf16_T1_L128_c2", T=1, L=128, cores=2, seed=1, torch_dtype=_BF16)
    run_case("bf16_T2_L256_c2", T=2, L=256, cores=2, seed=4, torch_dtype=_BF16)
    run_case("bf16_T1_L256_c1", T=1, L=256, cores=1, seed=3, torch_dtype=_BF16)
    run_case("bf16_T2_L512_c2", T=2, L=512, cores=2, seed=5, torch_dtype=_BF16)
    run_case("bf16_T1_L640_c1", T=1, L=640, cores=1, seed=6, torch_dtype=_BF16)

    print("\n=== multi-chunk MM1 (L > 4096, cores=1) ===")
    run_case(
        "fp32_T1_L4224_c1",
        T=1,
        L=4224,
        cores=1,
        seed=7,
        torch_dtype=_FP32,
        cross_check=False,
    )

    print("\nALL CASES PASSED")


if __name__ == "__main__":
    main()
