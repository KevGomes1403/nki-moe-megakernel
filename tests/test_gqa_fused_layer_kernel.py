"""On-device correctness test for the fully fused GQA token-generation layer kernel.

Exercises ``gqa_fused_tkg_fwd`` (one LNC2 @nki.jit launch fusing qkv projection -> q/k RMSNorm ->
partial RoPE -> attention -> sigmoid gate -> output projection, all intermediates in SBUF) against a
PyTorch reference that mirrors the kernel's precision sequence stage-by-stage.

The kernel receives ALREADY-RMSNorm'd hidden (the decoder applies input_layernorm), computes the gate
from the same normed hidden, q/k-RMSNorms over head_dim, partial-RoPEs the first rope_dim, scales Q by
1/sqrt(D), runs causal GQA attention over (prior-from-cache + active), applies sigmoid(gate) before the
o_proj, and returns the per-rank o_proj PARTIAL [T, H] (TP all-reduce deferred). Active K/V are written
into the caches in place.

GATES:
  * FP32 IO: torch.allclose(atol=1e-5, rtol=1e-2) -- the HARD gate (literal requirement).
  * BF16 IO: report max_abs / max_rel / ULP / cosine; pass on allclose OR a magnitude-aware cosine
    floor (the runtime is bf16, so cosine must be high; the hard numeric gate is fp32).

Cases (required): cores=1 {T=1, T=2} at L=128; cores=2 {T=1, T=2} at L=256 (exercises attention
s_prior-sharding AND the no-sendrecv out_proj seam -- the riskiest path).

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
COSINE_FLOOR = 0.999  # bf16 magnitude-aware (scale-invariant) gate


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

    Returns hidden [1,T,H] (pre-normed), qkv_w [H,I], gate_w [H,G], gamma_q/k [D], cos/sin [T,rope_dim],
    prior_k/prior_v [L-T,D], o_proj_w [value_dim,H]. The active tokens sit at sequence positions
    [L-T, L); the prior occupies the first L-T KV slots."""
    assert L % P_MAX == 0 and L > T
    torch.manual_seed(seed)
    hidden = torch.randn(1, T, HIDDEN)
    qkv_w = torch.randn(HIDDEN, I_DIM) / math.sqrt(HIDDEN)
    gate_w = torch.randn(HIDDEN, GATE_DIM) / math.sqrt(HIDDEN)
    gamma_q = torch.randn(HEAD_DIM)
    gamma_k = torch.randn(HEAD_DIM)
    o_proj_w = torch.randn(VALUE_DIM, HIDDEN) / math.sqrt(VALUE_DIM)
    position_ids = torch.arange(L - T, L)  # active token absolute positions
    cos, sin = build_cos_sin(position_ids)
    prior_k = torch.randn(L - T, HEAD_DIM)
    prior_v = torch.randn(L - T, HEAD_DIM)
    return hidden, qkv_w, gate_w, gamma_q, gamma_k, cos, sin, prior_k, prior_v, o_proj_w


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


def golden(inp, T, L, dtype):
    """Reference outputs mirroring the kernel stage-by-stage at the IO precision.

    Returns ``(o [T, H], active_k [1,1,D,T] BHDS, active_v [1,1,T,D] BHSD)`` -- the o_proj partial and
    the post-norm/RoPE active K / projected active V (the tensors NxDI scatters into the caches)."""
    hidden, qkv_w, gate_w, gamma_q, gamma_k, cos, sin, prior_k, prior_v, o_proj_w = inp
    D = HEAD_DIM

    nh = _rd(hidden.reshape(T, HIDDEN), dtype)
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
def run_kernel(inp, T, L, cores, dtype):
    """Build the cache buffers + mask, launch gqa_fused_tkg_fwd on `cores` cores.

    Returns ``(o_out [T,H], active_k [1,1,D,T], active_v [1,1,T,D])``."""
    import torch_xla.core.xla_model as xm

    hidden, qkv_w, gate_w, gamma_q, gamma_k, cos, sin, prior_k, prior_v, o_proj_w = inp

    k_cache = torch.zeros(1, 1, HEAD_DIM, L)
    v_cache = torch.zeros(1, 1, L, HEAD_DIM)
    k_cache[0, 0, :, 0 : L - T] = prior_k.transpose(0, 1)  # BHDS prior
    v_cache[0, 0, 0 : L - T, :] = prior_v  # BHSD prior
    mask = build_mask(T, L)

    dev = xm.xla_device()
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
    )
    return o.to(dtype).cpu(), active_k.to(dtype).cpu(), active_v.to(dtype).cpu()


# ---------------------------------------------------------------------------
# Metrics / checks
# ---------------------------------------------------------------------------
def _ulp_distance(ker, ref):
    k = ker.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    r = ref.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    k = torch.where(k < 0, 0x8000 - k, k)
    r = torch.where(r < 0, 0x8000 - r, r)
    return (k - r).abs().max().item()


def _cosine(ker, ref):
    kd = ker.double().flatten()
    rd = ref.double().flatten()
    denom = (kd.norm() * rd.norm()).clamp_min(1e-30)
    return (kd @ rd / denom).item()


def _metrics(ker, ref):
    kd = ker.double()
    rd = ref.double()
    abs_err = (kd - rd).abs()
    denom = rd.abs().clamp_min(1e-4)
    return (
        abs_err.max().item(),
        (abs_err / denom).max().item(),
        _ulp_distance(ker, ref),
        _cosine(ker, ref),
    )


def _check(name, ker, ref, dtype):
    max_abs, max_rel, ulp, cos = _metrics(ker, ref)
    allclose = torch.allclose(ker.double(), ref.double(), atol=ATOL, rtol=RTOL)
    if dtype == torch.float32:
        ok = allclose  # HARD gate
        gate = f"allclose(atol={ATOL} rtol={RTOL})"
    else:
        ok = allclose or (cos >= COSINE_FLOOR)
        gate = "allclose" if allclose else f"cosine>={COSINE_FLOOR}"
    status = "PASS" if ok else "FAIL"
    print(
        f"[{name}] {status}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
        f"max_ulp={ulp}  cos={cos:.6f}  allclose={allclose}  gate={gate}"
    )
    assert ok, f"{name}: gate {gate} failed (max_abs={max_abs:.3e} cos={cos:.6f})"


def _check_active(name, ker, ref, dtype):
    """Active K/V check: fp32 hard gate (allclose atol=1e-5 rtol=1e-2); bf16 reports max_abs only."""
    max_abs = (ker.double() - ref.double()).abs().max().item()
    if dtype == torch.float32:
        ok = torch.allclose(ker.double(), ref.double(), atol=ATOL, rtol=RTOL)
        print(
            f"[{name}] {'PASS' if ok else 'FAIL'}  max_abs={max_abs:.3e}  shape={tuple(ker.shape)}"
        )
        assert ok, f"{name}: active K/V mismatch (max_abs={max_abs:.3e})"
    else:
        print(f"[{name}] (bf16) max_abs={max_abs:.3e}  shape={tuple(ker.shape)}")


def run_case(name, T, L, cores, seed, dtype):
    inp = make_inputs(T=T, L=L, seed=seed)
    ker_o, ker_k, ker_v = run_kernel(inp, T, L, cores=cores, dtype=dtype)
    ref_o, ref_k, ref_v = golden(inp, T, L, dtype)
    _check(name, ker_o, ref_o, dtype)
    _check_active(f"{name}.active_k", ker_k, ref_k, dtype)
    _check_active(f"{name}.active_v", ker_v, ref_v, dtype)
    return ker_o


# ---------------------------------------------------------------------------
# pytest entrypoints (fp32 hard gate)
# ---------------------------------------------------------------------------
def test_fp32_t1_cores1():
    run_case("fp32_T1_L128_c1", T=1, L=128, cores=1, seed=1, dtype=torch.float32)


def test_fp32_t2_cores1():
    run_case("fp32_T2_L128_c1", T=2, L=128, cores=1, seed=2, dtype=torch.float32)


def test_fp32_t1_cores2():
    run_case("fp32_T1_L256_c2", T=1, L=256, cores=2, seed=3, dtype=torch.float32)


def test_fp32_t2_cores2():
    run_case("fp32_T2_L256_c2", T=2, L=256, cores=2, seed=4, dtype=torch.float32)


# Deployment path: L=128 on a 2-core launch (seq_len=128 bucket). Attention runs unsharded
# (s_prior < 256, batch=1), while qkv/o_proj stay H-sharded across the 2 cores.
def test_fp32_t1_cores2_l128():
    run_case("fp32_T1_L128_c2", T=1, L=128, cores=2, seed=5, dtype=torch.float32)


def test_fp32_t2_cores2_l128():
    run_case("fp32_T2_L128_c2", T=2, L=128, cores=2, seed=6, dtype=torch.float32)


# pytest entrypoints (bf16 metrics, cosine floor)
def test_bf16_t1_cores1():
    run_case("bf16_T1_L128_c1", T=1, L=128, cores=1, seed=1, dtype=torch.bfloat16)


def test_bf16_t2_cores2():
    run_case("bf16_T2_L256_c2", T=2, L=256, cores=2, seed=4, dtype=torch.bfloat16)


def main():
    print("=== FP32 IO (hard gate: allclose atol=1e-5 rtol=1e-2) ===")
    run_case("fp32_T1_L128_c1", T=1, L=128, cores=1, seed=1, dtype=torch.float32)
    run_case("fp32_T2_L128_c1", T=2, L=128, cores=1, seed=2, dtype=torch.float32)
    run_case("fp32_T1_L256_c2", T=1, L=256, cores=2, seed=3, dtype=torch.float32)
    run_case("fp32_T2_L256_c2", T=2, L=256, cores=2, seed=4, dtype=torch.float32)
    run_case("fp32_T1_L128_c2", T=1, L=128, cores=2, seed=5, dtype=torch.float32)
    run_case("fp32_T2_L128_c2", T=2, L=128, cores=2, seed=6, dtype=torch.float32)

    print("\n=== BF16 IO (cosine floor gate; metrics reported) ===")
    run_case("bf16_T1_L128_c1", T=1, L=128, cores=1, seed=1, dtype=torch.bfloat16)
    run_case("bf16_T2_L128_c1", T=2, L=128, cores=1, seed=2, dtype=torch.bfloat16)
    run_case("bf16_T1_L256_c2", T=1, L=256, cores=2, seed=3, dtype=torch.bfloat16)
    run_case("bf16_T2_L256_c2", T=2, L=256, cores=2, seed=4, dtype=torch.bfloat16)

    print("\nALL CASES PASSED")


if __name__ == "__main__":
    main()
