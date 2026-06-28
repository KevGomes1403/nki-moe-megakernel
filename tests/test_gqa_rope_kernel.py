"""Isolated correctness test for the GQA partial-RoPE (+mRoPE) TKG NKI kernel (Phase 3 of 5).

Exercises ``rope_partial_compose`` (the SBUF-in/SBUF-out composable that will drop into the fused GQA
kernel) via a thin @nki.jit harness defined here. The harness DMAs the NBSd qk_norm output
``x_in`` [N, B, S, D] (N=6 = [q0,q1,q2,q3,k0,v0], head_dim D=256) into SBUF as a [T, N, D] tile
(T = B*S tokens on the PARTITION axis; N heads then head_dim on the FREE axis -- the qk_norm output
layout), builds the per-token cos/sin [T, rope_dim], calls the composable (which broadcasts cos/sin
across the NUM_ROPE_HEADS=5 rope heads internally), and transposing-stores the [T, N, D] result back
to NBSd. The composable applies PARTIAL rotary embedding -- rotate the first rope_dim=64 of
head_dim=256, pass the remaining 192 through -- to each Q head and the K head; V is passed through.

Rotation convention (matched bit-exactly to the model):
  - NeuronQwen36A3BAttention.apply_rotary_embedding (modeling_qwen36_a3b.py ~:1807-1841): q_rope =
    Q[..., :64], apply_rotary_pos_emb(q_rope, k_rope, cos, sin), then cat with Q[..., 64:].
  - NxDI apply_rotary_pos_emb / _rotate_half (attention/utils.py :233-249): out = x*cos +
    rotate_half(x)*sin, rotate_half(x) = cat(-x[half:], x[:half]); dim i pairs with i+32.
  - cos/sin built here exactly as Qwen36A3BMRoPEEmbedding.forward (modeling_qwen36_a3b.py ~:1705-1750)
    with config mrope_section [11,11,10], rope_dim=64, rope_theta=1e7: emb = cat(freqs_t, freqs_t),
    cos = emb.cos(), sin = emb.sin(). We feed the precomputed cos/sin to BOTH kernel and golden; the
    kernel only applies them.

Precision (two dtypes per case):
  - FP32 IO -- the HARD correctness gate: torch.allclose(atol=1e-5, rtol=1e-2). RoPE is elementwise and
    the kernel is fp32-internal, so this is tight and MUST pass.
  - bf16 IO -- the model's runtime contract (bf16 IO, fp32 internal then bf16 cast, mirrored by the
    golden). Gated on cosine > 0.999; max_abs / max_rel / ULP and the fixed-(1e-5,1e-2) and an
    output-magnitude-scaled allclose are reported. A fixed 1e-5 atol is below the bf16 representation
    granularity (~value*2^-8), so it is reported, not gated.

Cases: T=1 (position [13]) and T=2 (distinct positions [13, 27]) so the rotation is non-trivial.

Run (USE CORE 3 ONLY):
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=3 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_gqa_rope_kernel
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

from models.qwen3_6_moe.nki_kernels.gqa.components.rope import (  # noqa: E402
    HEAD_DIM,
    NUM_HEADS,
    NUM_ROPE_HEADS,
    ROPE_DIM,
    rope_partial_compose,
)

# Qwen3.6 GQA per-rank (TP=4) RoPE config (config.rope_parameters).
ROPE_THETA = 10_000_000.0  # 1e7
MROPE_SECTION = (11, 11, 10)  # sums to rope_dim // 2 = 32

# FP32 hard gate (the literal task tolerance).
ATOL = 1e-5
RTOL = 1e-2
# bf16 structural gate: cosine, plus an output-magnitude-scaled atol floor for the reported allclose.
COS_MIN = 0.999
ATOL_SCALE_BF16 = 2e-2


@nki.jit
def rope_harness(x_in, cos_in, sin_in):
    """DMA the NBSd input [N, B, S, D] into a [T, N, D] SBUF tile (T=B*S tokens on partition, N heads
    then head_dim on free -- the qk_norm output layout), build per-token cos/sin [T, rope_dim], call
    rope_partial_compose, and transposing-store the [T, N, D] result back to NBSd [N, B, S, D] in
    shared_hbm. dtype-agnostic (bf16 or fp32 from the inputs). Launch [1]."""
    n, b, s, d = x_in.shape
    t = b * s
    rope_dim = cos_in.shape[1]

    # Transposing load: x_sb[tok, head, d] = x_in[head, 0, tok, d]. NBSd [N,1,S,D] is row-major, so
    # element (head, 0, tok, d) has linear offset head*(t*d) + tok*d + d. AP order matches the dst
    # iteration (partition=tok, then free head, then free d).
    x_sb = nl.ndarray((t, n, d), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x_in.ap(pattern=[[d, t], [t * d, n], [1, d]], offset=0))

    # Per-token cos/sin [T, rope_dim] (contiguous); the composable broadcasts across heads internally.
    cos_sb = nl.ndarray((t, rope_dim), dtype=cos_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=cos_sb, src=cos_in.ap(pattern=[[rope_dim, t], [1, rope_dim]], offset=0)
    )
    sin_sb = nl.ndarray((t, rope_dim), dtype=sin_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=sin_sb, src=sin_in.ap(pattern=[[rope_dim, t], [1, rope_dim]], offset=0)
    )

    out_sb = rope_partial_compose(x_sb, cos_sb, sin_sb)

    # Transposing store back to NBSd: out_hbm[head, 0, tok, d] = out_sb[tok, head, d] (inverse of the
    # load AP), so the golden/to_nbsd comparison runs in NBSd [N, 1, t, D].
    out_hbm = nl.ndarray((n, b, s, d), dtype=out_sb.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=out_hbm.ap(pattern=[[d, t], [t * d, n], [1, d]], offset=0), src=out_sb
    )
    return out_hbm


def build_cos_sin(position_ids):
    """mRoPE cos/sin exactly as Qwen36A3BMRoPEEmbedding.forward (text path: 3D positions identical).

    Args:
        position_ids: 1D long tensor [S] of token positions (B=1).

    Returns:
        cos, sin: fp32 tensors [S, rope_dim] (rope_dim = ROPE_DIM = 64, the duplicated cat(freqs,
        freqs)).
    """
    s = position_ids.shape[0]
    # Text path: broadcast the same positions across the 3 (T/H/W) mRoPE axes.
    pos3d = position_ids[None, None, :].expand(3, 1, s).float()  # (3, B=1, S)
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM)
    )  # (rope_dim/2 = 32,)
    inv_freq = inv_freq[None, None, :, None].expand(3, 1, -1, 1)  # (3, 1, 32, 1)
    positions = pos3d[:, :, None, :]  # (3, 1, 1, S)
    freqs = (inv_freq @ positions).transpose(2, 3)  # (3, 1, S, 32)
    freqs_t = freqs[0].clone()  # (1, S, 32)
    # Interleaved H/W splice (no-op for identical positions; kept to match the convention exactly).
    for dim, offset in ((1, 1), (2, 2)):
        length = MROPE_SECTION[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim][..., idx]
    emb = torch.cat((freqs_t, freqs_t), dim=-1)  # (1, S, 64)
    return emb.cos()[0], emb.sin()[0]  # (S, 64), (S, 64)


def make_inputs(t, seed):
    """Random fp32 NBSd input x [N, B=1, S=t, D] and per-token cos/sin [t, rope_dim].

    Returns:
        x: [N, 1, t, D] fp32; cos, sin: [t, rope_dim] fp32; position_ids: [t].
    """
    torch.manual_seed(seed)
    x = torch.randn(NUM_HEADS, 1, t, HEAD_DIM)
    # Distinct, non-trivial token positions so the rotation is non-trivial.
    position_ids = torch.arange(t) * 14 + 13  # t=1 -> [13]; t=2 -> [13, 27]
    cos, sin = build_cos_sin(position_ids)
    return x, cos, sin


def to_nbsd(out, t):
    """Kernel output (already NBSd [N, B=1, S=t, D]) -> NBSd [N, 1, t, D] (identity reshape)."""
    return out.reshape(NUM_HEADS, 1, t, HEAD_DIM)


def golden(inp, dtype):
    """CPU reference mirroring the kernel: dtype-round the inputs, rotate the first ROPE_DIM of each
    Q/K head in fp32 (rotate_half), pass the tail and the V head through, cast back to dtype.

    Matches NeuronQwen36A3BAttention.apply_rotary_embedding + NxDI apply_rotary_pos_emb."""
    x, cos, sin = inp
    t = x.shape[2]
    half = ROPE_DIM // 2
    # dtype-round inputs (IO contract), then fp32 for the rotation.
    out = x.to(dtype).float().clone()  # [N, 1, t, D]
    c = cos.to(dtype).float()  # [t, rope_dim]
    s = sin.to(dtype).float()
    for n in range(
        NUM_ROPE_HEADS
    ):  # q0..q3, k0 (v0 = head NUM_ROPE_HEADS passes through)
        xr = out[n, 0, :, :ROPE_DIM]  # [t, rope_dim]
        x1 = xr[:, :half]
        x2 = xr[:, half:]
        rot_half = torch.cat((-x2, x1), dim=-1)  # [t, rope_dim]
        out[n, 0, :, :ROPE_DIM] = xr * c + rot_half * s
    return out.to(dtype)


def run_kernel(inp, dtype):
    """Move dtype inputs to the Neuron device, launch on CORE 3 (grid [1]), return NBSd CPU output."""
    import torch_xla.core.xla_model as xm

    x, cos, sin = inp
    t = x.shape[2]
    dev = xm.xla_device()
    x_d = x.to(dtype).contiguous().to(dev)
    cos_d = cos.to(dtype).contiguous().to(dev)
    sin_d = sin.to(dtype).contiguous().to(dev)
    out = rope_harness[1](x_d, cos_d, sin_d)  # NBSd [N, B, S, D]
    return to_nbsd(out.to(dtype).cpu(), t)


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
    """Model-contract gate: cosine > 0.999. The fixed-(1e-5,1e-2) allclose and an output-magnitude
    -scaled allclose are reported for visibility (a fixed 1e-5 atol is below bf16 granularity)."""
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


def _check_passthrough(name, ker, inp, dtype):
    """The V head (NUM_ROPE_HEADS) and the [ROPE_DIM:] tail of every head must be bit-identical to the
    dtype-rounded input (pass-through is a copy, not arithmetic)."""
    x = inp[0].to(dtype)
    v_ok = torch.equal(ker[NUM_ROPE_HEADS], x[NUM_ROPE_HEADS])
    tail_ok = torch.equal(ker[..., ROPE_DIM:], x[..., ROPE_DIM:])
    print(
        f"[{name}:passthrough] v_head_bit_identical={v_ok}  rope_tail_bit_identical={tail_ok}"
    )
    assert v_ok, f"{name}: V head was modified (must pass through)"
    assert tail_ok, (
        f"{name}: head_dim tail [{ROPE_DIM}:] was modified (must pass through)"
    )


def run_case(name, t, seed):
    """For both fp32 (hard gate) and bf16 (model contract): validate the rotated output against the
    NBSd reference and assert the V head / rope-tail pass through bit-exactly."""
    inp = make_inputs(t=t, seed=seed)

    # FP32: hard correctness gate at the literal task tolerance.
    ref32 = golden(inp, torch.float32)
    ker32 = run_kernel(inp, torch.float32)
    _check_fp32(f"{name}:fp32", ker32, ref32)
    _check_passthrough(f"{name}:fp32", ker32, inp, torch.float32)

    # bf16: model runtime contract, gated on cosine; metrics reported.
    ref16 = golden(inp, torch.bfloat16)
    ker16 = run_kernel(inp, torch.bfloat16)
    _check_bf16(f"{name}:bf16", ker16, ref16)
    _check_passthrough(f"{name}:bf16", ker16, inp, torch.bfloat16)


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
