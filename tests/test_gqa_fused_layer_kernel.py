"""On-device correctness test for the fully fused GQA token-generation layer kernel.

Exercises ``gqa_fused_tkg_fwd`` (one LNC2 @nki.jit launch fusing qkv projection -> q/k RMSNorm ->
partial RoPE -> attention -> sigmoid gate -> output projection, all intermediates in SBUF) against a
PyTorch reference that mirrors the kernel's precision sequence stage-by-stage.

PRIMARY path (norm_in=True): the kernel receives RAW hidden and runs the pre-attention input RMSNorm
IN-KERNEL (nkilib rmsnorm_tkg -> SBUF-resident [128,T,16] normed tile), feeding BOTH projections
(qkv + gate) directly from SBUF with zero HBM round-trip; then q/k-RMSNorms over head_dim,
partial-RoPEs the first rope_dim, scales Q by 1/sqrt(D), runs causal GQA attention over
(prior-from-cache + active), applies sigmoid(gate) before the o_proj, and returns the per-rank o_proj
PARTIAL [T, H]. The pre-normed regression path (norm_in=False, gamma_in=None) is the backward-compatible
path the model still uses (hidden already normed in HBM).

GATES (cosine is BANNED repo-wide as a gate/metric):
  * FP32 IO: torch.allclose(atol=1e-5, rtol=1e-2) -- the HARD gate. Also validates the spec-6.1 SBUF
    column-order match between rmsnorm's output and qkv_tkg's H-shard slicing (a wrong permutation is
    an O(1) mismatch, not 1e-6).
  * BF16 IO: max_abs(kernel_bf16 - oracle_fp32) <= floor * BF16_HEADROOM, floor = the end-to-end
    output-rounding step max_abs(oracle_fp32.to(bf16) - oracle_fp32). max_abs and max_rel reported for
    every case and both dtypes.

Cases (required): cores in {1,2}, T in {1,2}, L in {128,256} -- incl. the c2/L256 s_prior-sharded case
and the c2/L128 deployment case.

Run (CORES 0,1):
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_gqa_fused_layer_kernel
"""

import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.qwen3_6_moe.nki_kernels.gqa.decode.fused_layer import (  # noqa: E402
    gqa_fused_tkg_fwd,
)

# Per-rank (TP=4) Qwen3.6 GQA decode dims (mirror the kernel module constants).
P_MAX = 128
HIDDEN = 2048  # H
HEAD_DIM = 256  # D
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
NUM_HEADS = NUM_Q_HEADS + 2 * NUM_KV_HEADS  # 6 head-major [q0|q1|q2|q3|k0|v0]
ROPE_DIM = 64
I_DIM = NUM_HEADS * HEAD_DIM  # 1536
GATE_DIM = NUM_Q_HEADS * HEAD_DIM  # 1024
VALUE_DIM = NUM_Q_HEADS * HEAD_DIM  # 1024
EPS = 1e-6
SCALE = 1.0 / math.sqrt(HEAD_DIM)

# mRoPE config (config.rope_parameters); identical to the rope phase test.
ROPE_THETA = 10_000_000.0
MROPE_SECTION = (11, 11, 10)

ATOL = 1e-5
RTOL = 1e-2
# bf16 gate (cosine BANNED repo-wide): max_abs(kernel_bf16 - oracle_fp32) <= floor * BF16_HEADROOM,
# floor = max_abs(oracle_fp32.to(bf16) - oracle_fp32) is the end-to-end output-rounding step. Headroom
# covers the accumulated bf16 rounding of the full input-norm->qkv->qk-norm->rope->attention->o_proj
# pipeline (many bf16-rounded matmuls/weights) on top of that single output-rounding floor. The
# measured worst-case ratio (achieved / floor) across all cases/tensors is ~4.4x (active_k) and ~3.1x
# for the o output; 8.0 leaves ~1.8x safety for seed variation while staying a meaningful gate.
BF16_HEADROOM = 8.0


# ---------------------------------------------------------------------------
# mRoPE cos/sin (exact copy of the rope phase test's build_cos_sin)
# ---------------------------------------------------------------------------
def build_cos_sin(position_ids):
    """mRoPE cos/sin exactly as Qwen36A3BMRoPEEmbedding.forward (text path). Returns [S, rope_dim]."""
    s = position_ids.shape[0]
    pos3d = position_ids[None, None, :].expand(3, 1, s).float()
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM)
    )
    inv_freq = inv_freq[None, None, :, None].expand(3, 1, -1, 1)
    positions = pos3d[:, :, None, :]
    freqs = (inv_freq @ positions).transpose(2, 3)  # (3, 1, S, 32)
    freqs_t = freqs[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        length = MROPE_SECTION[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim][..., idx]
    emb = torch.cat((freqs_t, freqs_t), dim=-1)  # (1, S, 64)
    return emb.cos()[0], emb.sin()[0]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def make_inputs(T, L, seed):
    """Random fp32 logical inputs (B=1). Weights are fan-in scaled so activations stay O(1).

    Returns hidden [1,T,H] (RAW, pre-norm), qkv_w [H,I], gate_w [H,G], gamma_q/k [D], cos/sin
    [T,rope_dim], prior_k/prior_v [L-T,D], o_proj_w [value_dim,H], gamma_in [H]. ``gamma_in`` is the
    input_layernorm weight in STANDARD form (~1.0; the +1 checkpoint convention already applied) --
    used only by the in-kernel-norm path. ``hidden`` is RAW: the norm path applies the input RMSNorm
    (kernel and golden), the pre-normed regression path consumes it as-is. The active tokens sit at
    sequence positions [L-T, L); the prior occupies the first L-T KV slots."""
    assert L % P_MAX == 0 and L > T
    torch.manual_seed(seed)
    hidden = torch.randn(1, T, HIDDEN)
    qkv_w = torch.randn(HIDDEN, I_DIM) / math.sqrt(HIDDEN)
    gate_w = torch.randn(HIDDEN, GATE_DIM) / math.sqrt(HIDDEN)
    gamma_q = torch.randn(HEAD_DIM)
    gamma_k = torch.randn(HEAD_DIM)
    o_proj_w = torch.randn(VALUE_DIM, HIDDEN) / math.sqrt(VALUE_DIM)
    gamma_in = (
        torch.randn(HIDDEN) * 0.02 + 1.0
    )  # input_layernorm.weight, standard form (~1.0)
    position_ids = torch.arange(L - T, L)  # active token absolute positions
    cos, sin = build_cos_sin(position_ids)
    prior_k = torch.randn(L - T, HEAD_DIM)
    prior_v = torch.randn(L - T, HEAD_DIM)
    return (
        hidden,
        qkv_w,
        gate_w,
        gamma_q,
        gamma_k,
        cos,
        sin,
        prior_k,
        prior_v,
        o_proj_w,
        gamma_in,
    )


def build_mask(T, L):
    """Causal mask [L,1,q_heads,T] uint8 (1=keep): active token t (slot L-T+t) attends key j iff
    j <= (L-T)+t; the whole prior is visible. Same for all heads."""
    prior_len = L - T
    j = torch.arange(L).view(L, 1)
    t = torch.arange(T).view(1, T)
    keep = (j <= (prior_len + t)).to(torch.uint8)  # [L, T]
    return keep.view(L, 1, 1, T).expand(L, 1, NUM_Q_HEADS, T).contiguous()


# ---------------------------------------------------------------------------
# PyTorch reference (fp32 exact; bf16 mirrors the kernel precision sequence)
# ---------------------------------------------------------------------------
def _rd(x, dtype):
    """Round to the IO dtype then back to fp32 (models a bf16/fp32 IO store; no-op for fp32)."""
    return x.to(dtype).float()


def _rope(x, cos, sin, dtype):
    """Partial rotate_half RoPE on the first ROPE_DIM dims of each head of x [T, heads, D]."""
    half = ROPE_DIM // 2
    out = x.clone()
    c = _rd(cos, dtype)
    s = _rd(sin, dtype)
    for h in range(x.shape[1]):
        xr = out[:, h, :ROPE_DIM]
        x1 = xr[:, :half]
        x2 = xr[:, half:]
        rot = torch.cat([-x2, x1], dim=-1)
        out[:, h, :ROPE_DIM] = _rd(xr * c + rot * s, dtype)
    return out


def golden(inp, T, L, dtype, norm_in):
    """Reference outputs mirroring the kernel stage-by-stage at the IO precision.

    When ``norm_in`` the input RMSNorm over H is applied first (fp32 internal, mirroring the in-kernel
    ``rmsnorm_tkg`` -> SBUF normed_sb feeding both projections); otherwise ``hidden`` is consumed as the
    already-normed input (the pre-normed regression path). ``gamma_in`` is bf16-rounded in the bf16
    oracle (via ``_rd``), matching the kernel's bf16 gamma input.

    Returns ``(o [T, H], active_k [1,1,D,T] BHDS, active_v [1,1,T,D] BHSD)`` -- the o_proj partial and
    the post-norm/RoPE active K / projected active V (the tensors NxDI scatters into the caches)."""
    (
        hidden,
        qkv_w,
        gate_w,
        gamma_q,
        gamma_k,
        cos,
        sin,
        prior_k,
        prior_v,
        o_proj_w,
        gamma_in,
    ) = inp
    D = HEAD_DIM

    h_io = _rd(hidden.reshape(T, HIDDEN), dtype)  # hidden as stored in HBM (bf16/fp32)
    if norm_in:
        # Pre-attention RMSNorm over full H (fp32 internal), gamma at IO dtype -> normed IO tile.
        x32 = h_io.float()
        g_in = _rd(gamma_in, dtype)
        inv = (x32.square().mean(-1, keepdim=True) + EPS).rsqrt()
        nh = _rd((x32 * inv) * g_in, dtype)
    else:
        nh = h_io
    qkv = _rd(nh @ _rd(qkv_w, dtype), dtype)  # [T, I]
    gate = _rd(nh @ _rd(gate_w, dtype), dtype)  # [T, G]

    q = qkv[:, : NUM_Q_HEADS * D].reshape(T, NUM_Q_HEADS, D)
    k = qkv[:, NUM_Q_HEADS * D : (NUM_Q_HEADS + NUM_KV_HEADS) * D].reshape(
        T, NUM_KV_HEADS, D
    )
    v = qkv[:, (NUM_Q_HEADS + NUM_KV_HEADS) * D :].reshape(T, NUM_KV_HEADS, D)

    gq = _rd(gamma_q, dtype)
    gk = _rd(gamma_k, dtype)

    def rms(x, g):
        inv = (x.square().mean(-1, keepdim=True) + EPS).rsqrt()
        return _rd((x * inv) * g, dtype)

    q = rms(q, gq)
    k = rms(k, gk)
    q = _rope(q, cos, sin, dtype)
    k = _rope(k, cos, sin, dtype)
    # Active K/V exactly as scattered into the caches: K is post-norm+RoPE (pre-scale), V is projected.
    active_k = k[:, 0, :].transpose(0, 1).reshape(1, 1, D, T).contiguous()  # BHDS
    active_v = v[:, 0, :].reshape(1, 1, T, D).contiguous()  # BHSD
    q = _rd(q * SCALE, dtype)  # pre-scale then round, mirrors q_sb

    # Full K/V (GQA: 1 kv head replicated across the q heads); causal over prior + active.
    fk = torch.cat([_rd(prior_k, dtype), k[:, 0, :]], dim=0)  # [L, D]
    fv = torch.cat([_rd(prior_v, dtype), v[:, 0, :]], dim=0)  # [L, D]

    prior_len = L - T
    l_idx = torch.arange(L).view(1, L)
    thresh = (prior_len + torch.arange(T)).view(T, 1)
    addmask = torch.where(l_idx <= thresh, 0.0, float("-inf"))  # [T, L]

    attn = torch.empty(T, NUM_Q_HEADS, D)
    for h in range(NUM_Q_HEADS):
        scores = q[:, h, :] @ fk.transpose(0, 1) + addmask  # [T, L] fp32
        m = scores.max(dim=-1, keepdim=True).values
        e = torch.exp(scores - m)
        e_b = _rd(e, dtype)  # bf16-rounded exp (qk_io_type), fp32 for fp32
        s = e_b.sum(dim=-1, keepdim=True)
        attn[:, h, :] = (e_b @ fv) / s  # P.V (fp32) then normalize after
    attn = _rd(attn, dtype)  # attention output IO dtype

    attn_flat = attn.reshape(T, NUM_Q_HEADS * D)  # [T, value_dim] head-major
    sig = torch.sigmoid(gate.float())  # fp32 sigmoid
    gated = _rd(attn_flat * sig, dtype)  # [T, value_dim]
    o = _rd(gated @ _rd(o_proj_w, dtype), dtype)  # [T, H]
    return o.to(dtype), active_k.to(dtype), active_v.to(dtype)


# ---------------------------------------------------------------------------
# Device runner
# ---------------------------------------------------------------------------
def run_kernel(inp, T, L, cores, dtype, norm_in):
    """Build the cache buffers + mask, launch gqa_fused_tkg_fwd on `cores` cores.

    When ``norm_in`` the input_layernorm weight is passed as ``gamma_in=`` [1,H] so the input RMSNorm
    runs in-kernel (SBUF-resident, feeding both projections); otherwise gamma_in stays None (the
    pre-normed regression path -- the model's current call).

    Returns ``(o_out [T,H], active_k [1,1,D,T], active_v [1,1,T,D])``."""
    import torch_xla.core.xla_model as xm

    (
        hidden,
        qkv_w,
        gate_w,
        gamma_q,
        gamma_k,
        cos,
        sin,
        prior_k,
        prior_v,
        o_proj_w,
        gamma_in,
    ) = inp

    k_cache = torch.zeros(1, 1, HEAD_DIM, L)
    v_cache = torch.zeros(1, 1, L, HEAD_DIM)
    k_cache[0, 0, :, 0 : L - T] = prior_k.transpose(0, 1)  # BHDS prior
    v_cache[0, 0, 0 : L - T, :] = prior_v  # BHSD prior
    mask = build_mask(T, L)

    dev = xm.xla_device()
    gamma_in_dev = None
    if norm_in:
        gamma_in_dev = gamma_in.reshape(1, HIDDEN).to(dtype).contiguous().to(dev)
    o, active_k, active_v = gqa_fused_tkg_fwd[cores](
        hidden.to(dtype).contiguous().to(dev),
        qkv_w.to(dtype).contiguous().to(dev),
        gate_w.to(dtype).contiguous().to(dev),
        gamma_q.to(dtype).contiguous().to(dev),
        gamma_k.to(dtype).contiguous().to(dev),
        cos.to(dtype).contiguous().to(dev),
        sin.to(dtype).contiguous().to(dev),
        k_cache.to(dtype).contiguous().to(dev),
        v_cache.to(dtype).contiguous().to(dev),
        mask.to(dev),
        o_proj_w.to(dtype).contiguous().to(dev),
        EPS,
        gamma_in=gamma_in_dev,
    )
    return o.to(dtype).cpu(), active_k.to(dtype).cpu(), active_v.to(dtype).cpu()


# ---------------------------------------------------------------------------
# Metrics / checks
# ---------------------------------------------------------------------------
def _metrics(ker, ref):
    """max_abs and max_rel (rel = |k-r| / max(|r|, 1e-4)) vs the reference."""
    kd = ker.double()
    rd = ref.double()
    abs_err = (kd - rd).abs()
    denom = rd.abs().clamp_min(1e-4)
    return abs_err.max().item(), (abs_err / denom).max().item()


def _bf16_floor(ref_fp32):
    """End-to-end output-rounding floor: max_abs(oracle_fp32.to(bf16) - oracle_fp32)."""
    r = ref_fp32.float()
    return (r.to(torch.bfloat16).float() - r).abs().max().item()


def _check(name, ker, ref_fp32, dtype):
    """Gate one tensor against the fp32 oracle (used for o AND active K/V).

    fp32 IO: HARD gate torch.allclose(atol=1e-5, rtol=1e-2).
    bf16 IO: max_abs(kernel_bf16 - oracle_fp32) <= floor * BF16_HEADROOM (cosine BANNED). ref_fp32 is
    the fp32 oracle for BOTH dtypes -- bf16 is gated against the ideal fp32 result, not a bf16 mirror.
    Prints max_abs and max_rel for every case."""
    max_abs, max_rel = _metrics(ker, ref_fp32)
    if dtype == torch.float32:
        ok = torch.allclose(ker.double(), ref_fp32.double(), atol=ATOL, rtol=RTOL)
        gate = f"allclose(atol={ATOL} rtol={RTOL})"
        extra = ""
    else:
        floor = _bf16_floor(ref_fp32)
        limit = floor * BF16_HEADROOM
        ratio = max_abs / max(floor, 1e-30)
        ok = max_abs <= limit
        gate = f"max_abs<=floor*{BF16_HEADROOM:g}"
        extra = f"  floor={floor:.3e}  limit={limit:.3e}  ratio={ratio:.2f}x"
    status = "PASS" if ok else "FAIL"
    print(
        f"[{name}] {status}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  gate={gate}{extra}"
    )
    assert ok, (
        f"{name}: gate {gate} failed (max_abs={max_abs:.3e} max_rel={max_rel:.3e})"
    )


def run_case(name, T, L, cores, seed, dtype, norm_in=True):
    """Run one case. norm_in=True exercises the in-kernel RMSNorm path (primary); False is the
    pre-normed regression path. The bf16 gate compares against the fp32 oracle (same inputs)."""
    inp = make_inputs(T=T, L=L, seed=seed)
    ker_o, ker_k, ker_v = run_kernel(
        inp, T, L, cores=cores, dtype=dtype, norm_in=norm_in
    )
    ref_o, ref_k, ref_v = golden(
        inp, T, L, torch.float32, norm_in=norm_in
    )  # fp32 oracle
    _check(name, ker_o, ref_o, dtype)
    _check(f"{name}.active_k", ker_k, ref_k, dtype)
    _check(f"{name}.active_v", ker_v, ref_v, dtype)
    return ker_o


# ---------------------------------------------------------------------------
# pytest entrypoints -- in-kernel RMSNorm path (norm_in=True) is the PRIMARY focus.
# fp32 = HARD gate allclose(atol=1e-5, rtol=1e-2); bf16 = max_abs <= floor * BF16_HEADROOM.
# Matrix mirrors the pre-normed matrix: cores in {1,2}, T in {1,2}, L in {128,256}, incl. the
# c2/L256 s_prior-sharded case and the c2/L128 deployment case.
# ---------------------------------------------------------------------------
def test_norm_fp32_t1_c1_l128():
    run_case("norm_fp32_T1_L128_c1", T=1, L=128, cores=1, seed=1, dtype=torch.float32)


def test_norm_fp32_t2_c1_l128():
    run_case("norm_fp32_T2_L128_c1", T=2, L=128, cores=1, seed=2, dtype=torch.float32)


def test_norm_fp32_t1_c2_l256():
    run_case("norm_fp32_T1_L256_c2", T=1, L=256, cores=2, seed=3, dtype=torch.float32)


def test_norm_fp32_t2_c2_l256():
    run_case("norm_fp32_T2_L256_c2", T=2, L=256, cores=2, seed=4, dtype=torch.float32)


def test_norm_fp32_t1_c2_l128():
    run_case("norm_fp32_T1_L128_c2", T=1, L=128, cores=2, seed=5, dtype=torch.float32)


def test_norm_fp32_t2_c2_l128():
    run_case("norm_fp32_T2_L128_c2", T=2, L=128, cores=2, seed=6, dtype=torch.float32)


def test_norm_bf16_t1_c1_l128():
    run_case("norm_bf16_T1_L128_c1", T=1, L=128, cores=1, seed=1, dtype=torch.bfloat16)


def test_norm_bf16_t2_c2_l256():
    run_case("norm_bf16_T2_L256_c2", T=2, L=256, cores=2, seed=4, dtype=torch.bfloat16)


def test_norm_bf16_t1_c2_l128():
    run_case("norm_bf16_T1_L128_c2", T=1, L=128, cores=2, seed=5, dtype=torch.bfloat16)


# Backward-compat regression: gamma_in=None -> the pre-normed HBM path the model still uses.
def test_prenorm_fp32_t1_c1_l128():
    run_case(
        "prenorm_fp32_T1_L128_c1",
        T=1,
        L=128,
        cores=1,
        seed=1,
        dtype=torch.float32,
        norm_in=False,
    )


def test_prenorm_fp32_t2_c2_l256():
    run_case(
        "prenorm_fp32_T2_L256_c2",
        T=2,
        L=256,
        cores=2,
        seed=4,
        dtype=torch.float32,
        norm_in=False,
    )


def main():
    print(
        "=== IN-KERNEL NORM PATH -- FP32 IO (HARD gate: allclose atol=1e-5 rtol=1e-2) ==="
    )
    run_case("norm_fp32_T1_L128_c1", T=1, L=128, cores=1, seed=1, dtype=torch.float32)
    run_case("norm_fp32_T2_L128_c1", T=2, L=128, cores=1, seed=2, dtype=torch.float32)
    run_case("norm_fp32_T1_L256_c2", T=1, L=256, cores=2, seed=3, dtype=torch.float32)
    run_case("norm_fp32_T2_L256_c2", T=2, L=256, cores=2, seed=4, dtype=torch.float32)
    run_case("norm_fp32_T1_L128_c2", T=1, L=128, cores=2, seed=5, dtype=torch.float32)
    run_case("norm_fp32_T2_L128_c2", T=2, L=128, cores=2, seed=6, dtype=torch.float32)

    print(
        f"\n=== IN-KERNEL NORM PATH -- BF16 IO (gate: max_abs <= floor * {BF16_HEADROOM:g}) ==="
    )
    run_case("norm_bf16_T1_L128_c1", T=1, L=128, cores=1, seed=1, dtype=torch.bfloat16)
    run_case("norm_bf16_T2_L128_c1", T=2, L=128, cores=1, seed=2, dtype=torch.bfloat16)
    run_case("norm_bf16_T1_L256_c2", T=1, L=256, cores=2, seed=3, dtype=torch.bfloat16)
    run_case("norm_bf16_T2_L256_c2", T=2, L=256, cores=2, seed=4, dtype=torch.bfloat16)
    run_case("norm_bf16_T1_L128_c2", T=1, L=128, cores=2, seed=5, dtype=torch.bfloat16)
    run_case("norm_bf16_T2_L128_c2", T=2, L=128, cores=2, seed=6, dtype=torch.bfloat16)

    print(
        "\n=== REGRESSION -- pre-normed path (gamma_in=None; the model's current call) ==="
    )
    run_case(
        "prenorm_fp32_T1_L128_c1",
        T=1,
        L=128,
        cores=1,
        seed=1,
        dtype=torch.float32,
        norm_in=False,
    )
    run_case(
        "prenorm_fp32_T2_L256_c2",
        T=2,
        L=256,
        cores=2,
        seed=4,
        dtype=torch.float32,
        norm_in=False,
    )

    print("\nALL CASES PASSED")


if __name__ == "__main__":
    main()
