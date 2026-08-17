"""T=1 pre-flight for ``gqa_fused_tkg_fwd`` (draft-step shape) with a realistic partial-cache mask.

The verify megakernel only ever launches this kernel at T=2; the draft step launches it at T=1. This
covers T in {1, 2} x launch cores in {1, 2} at fp32, design A (kv_write_idx=None) with the in-kernel
input RMSNorm path (gamma_in given) -- what the draft megakernel uses.

Unlike the T=2 harness (whole prior visible), the KV cache here is only partially committed: slots
[0, committed_len) hold real context, slots [committed_len, L-T) hold garbage that the mask must
exclude, and the active tokens occupy the last T slots of the L tile. A mask bug shows up as an O(1)
mismatch rather than a rounding-level one.

Gate: torch.allclose(kernel_fp32, oracle_fp32, atol=1e-5, rtol=1e-2). max_abs / max_rel reported per
config. Oracle is the T=2 harness's stage-by-stage fp32 reference with the mask term replaced.

Run (CORES 0,1):
    cd /home/ubuntu/nki-moe && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    NEURON_CC_FLAGS="--target trn2 --lnc 2" \
    python -m megakernels.qwen3_6_moe.tests.test_gqa_fused_layer_t1_kernel
"""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nki_kernels.gqa.decode.fused_layer import gqa_fused_tkg_fwd  # noqa: E402

from megakernels.qwen3_6_moe.tests.test_gqa_fused_layer_kernel import (  # noqa: E402
    ATOL,
    EPS,
    HEAD_DIM,
    HIDDEN,
    NUM_Q_HEADS,
    RTOL,
    SCALE,
    _metrics,
    _rd,
    _rope,
    make_inputs,
)

L_CTX = 256  # KV cache length
COMMITTED_LEN = 150  # real committed context; [COMMITTED_LEN, L-T) is garbage


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------
def build_keep(T, L, committed_len):
    """[L, T] bool keep-map: prior slot j kept iff j < committed_len; active slot L-T+t kept causally.

    The active tokens live in the LAST T slots of the L tile -- that is where the kernel writes the
    freshly projected K, so slot L-1 must be kept for the final active token."""
    j = torch.arange(L).view(L, 1)
    t = torch.arange(T).view(1, T)
    prior_keep = (j < committed_len) & (j < L - T)
    active_keep = (j >= L - T) & (j <= (L - T) + t)
    return prior_keep | active_keep


def build_mask(keep):
    """[L, T] bool -> [L, 1, q_heads, T] uint8 kernel mask (1=keep), same for all heads."""
    L, T = keep.shape
    return (
        keep.to(torch.uint8).view(L, 1, 1, T).expand(L, 1, NUM_Q_HEADS, T).contiguous()
    )


# ---------------------------------------------------------------------------
# Oracle (fp32; mirrors the T=2 harness stage-by-stage, mask term generalized)
# ---------------------------------------------------------------------------
def golden(inp, T, L, keep, dtype):
    """Reference (o [T,H], active_k [1,1,D,T], active_v [1,1,T,D]) with the in-kernel input norm applied."""
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

    h_io = _rd(hidden.reshape(T, HIDDEN), dtype)
    x32 = h_io.float()
    inv = (x32.square().mean(-1, keepdim=True) + EPS).rsqrt()
    nh = _rd((x32 * inv) * _rd(gamma_in, dtype), dtype)

    qkv = _rd(nh @ _rd(qkv_w, dtype), dtype)
    gate = _rd(nh @ _rd(gate_w, dtype), dtype)
    q = qkv[:, : NUM_Q_HEADS * D].reshape(T, NUM_Q_HEADS, D)
    k = qkv[:, NUM_Q_HEADS * D : (NUM_Q_HEADS + 1) * D].reshape(T, 1, D)
    v = qkv[:, (NUM_Q_HEADS + 1) * D :].reshape(T, 1, D)

    def rms(x, g):
        i = (x.square().mean(-1, keepdim=True) + EPS).rsqrt()
        return _rd((x * i) * g, dtype)

    q = _rope(rms(q, _rd(gamma_q, dtype)), cos, sin, dtype)
    k = _rope(rms(k, _rd(gamma_k, dtype)), cos, sin, dtype)
    active_k = k[:, 0, :].transpose(0, 1).reshape(1, 1, D, T).contiguous()
    active_v = v[:, 0, :].reshape(1, 1, T, D).contiguous()
    q = _rd(q * SCALE, dtype)

    fk = torch.cat([_rd(prior_k, dtype), k[:, 0, :]], dim=0)  # [L, D]
    fv = torch.cat([_rd(prior_v, dtype), v[:, 0, :]], dim=0)  # [L, D]
    addmask = torch.where(keep.t(), 0.0, float("-inf"))  # [T, L]

    attn = torch.empty(T, NUM_Q_HEADS, D)
    for h in range(NUM_Q_HEADS):
        scores = q[:, h, :] @ fk.transpose(0, 1) + addmask
        e = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
        e_b = _rd(e, dtype)
        attn[:, h, :] = (e_b @ fv) / e_b.sum(dim=-1, keepdim=True)
    attn = _rd(attn, dtype)

    gated = _rd(attn.reshape(T, NUM_Q_HEADS * D) * torch.sigmoid(gate.float()), dtype)
    o = _rd(gated @ _rd(o_proj_w, dtype), dtype)
    return o.to(dtype), active_k.to(dtype), active_v.to(dtype)


# ---------------------------------------------------------------------------
# Device runner
# ---------------------------------------------------------------------------
def run_kernel(inp, T, L, keep, cores, dtype):
    """Launch gqa_fused_tkg_fwd on `cores` cores, design A + in-kernel input norm."""
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
    k_cache[0, 0, :, 0 : L - T] = prior_k.transpose(0, 1)
    v_cache[0, 0, 0 : L - T, :] = prior_v

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
        build_mask(keep).to(dev),
        o_proj_w.to(dtype).contiguous().to(dev),
        EPS,
        gamma_in=gamma_in.reshape(1, HIDDEN).to(dtype).contiguous().to(dev),
    )
    return o.to(dtype).cpu(), active_k.to(dtype).cpu(), active_v.to(dtype).cpu()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def _check(name, ker, ref):
    max_abs, max_rel = _metrics(ker, ref)
    ok = torch.allclose(ker.double(), ref.double(), atol=ATOL, rtol=RTOL)
    print(
        f"[{name}] {'PASS' if ok else 'FAIL'}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
    )
    assert ok, f"{name}: allclose(atol={ATOL} rtol={RTOL}) failed (max_abs={max_abs:.3e})"


def run_case(name, T, cores, seed, L=L_CTX, committed_len=COMMITTED_LEN):
    """One fp32 config: partial-cache mask, in-kernel input norm, design A."""
    inp = make_inputs(T=T, L=L, seed=seed)
    keep = build_keep(T, L, committed_len)
    ker_o, ker_k, ker_v = run_kernel(inp, T, L, keep, cores=cores, dtype=torch.float32)
    ref_o, ref_k, ref_v = golden(inp, T, L, keep, torch.float32)
    _check(name, ker_o, ref_o)
    _check(f"{name}.active_k", ker_k, ref_k)
    _check(f"{name}.active_v", ker_v, ref_v)
    return ker_o


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_fp32_t1_c2():
    run_case("gqa_fp32_T1_c2", T=1, cores=2, seed=11)


def test_fp32_t1_c1():
    run_case("gqa_fp32_T1_c1", T=1, cores=1, seed=11)


def test_fp32_t2_c2():
    run_case("gqa_fp32_T2_c2", T=2, cores=2, seed=12)


def test_fp32_t2_c1():
    run_case("gqa_fp32_T2_c1", T=2, cores=1, seed=12)


def main():
    print(
        f"=== GQA fused layer -- fp32, L={L_CTX}, committed_len={COMMITTED_LEN}, "
        "in-kernel input norm, design A ==="
    )
    run_case("gqa_fp32_T1_c2", T=1, cores=2, seed=11)
    run_case("gqa_fp32_T1_c1", T=1, cores=1, seed=11)
    run_case("gqa_fp32_T2_c2", T=2, cores=2, seed=12)
    run_case("gqa_fp32_T2_c1", T=2, cores=1, seed=12)
    print("\nALL GQA T=1/T=2 CASES PASSED")


if __name__ == "__main__":
    main()
