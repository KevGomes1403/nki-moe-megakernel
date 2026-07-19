"""On-device test for the in-place KV-cache-write path (design B) of the fused GQA TKG kernel.

Extends the design-A test (test_gqa_fused_layer_kernel) by passing ``kv_write_idx`` so the kernel
scatters the post-norm/RoPE active K/V into the caches in place at slots [idx : idx+T] and ALSO
returns the mutated (k_cache, v_cache) handles. Reuses that module's make_inputs / build_mask /
golden so the reference is identical. The scatter runs AFTER attention reads the prior, so o_out is
unchanged by the write and is gated against the same golden.

Checks per case:
  * o_out: fp32 allclose(atol=1e-5, rtol=1e-2) HARD gate; bf16 reports max_abs only (no cosine).
  * cache window [idx:idx+T] == returned active_k / active_v, bit-for-bit (same SBUF source).
  * cache window [idx:idx+T] == fp32-reference active K/V (fp32 hard gate; bf16 max_abs only).
  * cache slots OUTSIDE the window are unchanged from the input cache (no stray writes).
A tail idx (idx=L-T, the deployment slot) AND an interior idx (overwrites prior, proves
scalar_offset + the read-before-write ordering) are both exercised. A kv_write_idx=None case
confirms the default (design-A) path still returns 3 tensors and matches golden.

Coverage: T in {1,2} x cores in {1,2}; cores=2 exercises the LNC2 write-gating (V on prg 0, K on
prg 1 -> each cache written exactly once).

Run (CORES 0,1):
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_gqa_fused_layer_inplace_kv
"""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.qwen3_6_moe.nki_kernels.gqa.decode.fused_layer import (  # noqa: E402
    gqa_fused_tkg_fwd,
)
from models.qwen3_6_moe.tests.test_gqa_fused_layer_kernel import (  # noqa: E402
    ATOL,
    EPS,
    HEAD_DIM,
    RTOL,
    build_mask,
    golden,
    make_inputs,
)

# Arbitrary non-aligned interior slot: overwrites prior data, proving scalar_offset + write ordering.
INTERIOR_IDX = 17


# ---------------------------------------------------------------------------
# Checks (cosine-free, per repo policy)
# ---------------------------------------------------------------------------
def _gate(name, ker, ref, dtype):
    """fp32 -> allclose(atol=1e-5, rtol=1e-2) HARD gate; bf16 -> report max_abs only (no cosine)."""
    max_abs = (ker.double() - ref.double()).abs().max().item()
    if dtype == torch.float32:
        ok = torch.allclose(ker.double(), ref.double(), atol=ATOL, rtol=RTOL)
        print(
            f"[{name}] {'PASS' if ok else 'FAIL'}  max_abs={max_abs:.3e}  shape={tuple(ker.shape)}"
        )
        assert ok, (
            f"{name}: allclose(atol={ATOL}, rtol={RTOL}) failed (max_abs={max_abs:.3e})"
        )
    else:
        print(f"[{name}] (bf16) max_abs={max_abs:.3e}  shape={tuple(ker.shape)}")


def _bitexact(name, a, b):
    """Assert two CPU tensors are bit-for-bit identical (same shape + dtype + values)."""
    ok = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    print(f"[{name}] {'PASS' if ok else 'FAIL'}  bit-exact  shape={tuple(a.shape)}")
    assert ok, f"{name}: not bit-exact"


# ---------------------------------------------------------------------------
# Device runners
# ---------------------------------------------------------------------------
def run_kernel_inplace(inp, T, L, cores, dtype, idx):
    """Launch with kv_write_idx=idx; return CPU (o, active_k, active_v, ret_k, ret_v, init_k, init_v)."""
    import torch_xla.core.xla_model as xm

    hidden, qkv_w, gate_w, gamma_q, gamma_k, cos, sin, prior_k, prior_v, o_proj_w, _ = (
        inp
    )
    k_cache = torch.zeros(1, 1, HEAD_DIM, L)
    v_cache = torch.zeros(1, 1, L, HEAD_DIM)
    k_cache[0, 0, :, 0 : L - T] = prior_k.transpose(
        0, 1
    )  # BHDS prior in the first L-T slots
    v_cache[0, 0, 0 : L - T, :] = prior_v  # BHSD prior in the first L-T slots
    mask = build_mask(T, L)
    init_k = k_cache.to(
        dtype
    ).clone()  # caches as the kernel sees them (untouched slots must match)
    init_v = v_cache.to(dtype).clone()
    kv_write_idx = torch.tensor(
        [[idx]], dtype=torch.int32
    )  # [B,1] int32 write-start slot (B==1)

    dev = xm.xla_device()
    o, active_k, active_v, ret_k, ret_v = gqa_fused_tkg_fwd[cores](
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
        kv_write_idx.to(dev),
    )
    return (
        o.to(dtype).cpu(),
        active_k.to(dtype).cpu(),
        active_v.to(dtype).cpu(),
        ret_k.to(dtype).cpu(),
        ret_v.to(dtype).cpu(),
        init_k,
        init_v,
    )


def run_kernel_none(inp, T, L, cores, dtype):
    """Launch with kv_write_idx omitted (default); return the raw kernel return tuple + CPU o_out."""
    import torch_xla.core.xla_model as xm

    hidden, qkv_w, gate_w, gamma_q, gamma_k, cos, sin, prior_k, prior_v, o_proj_w, _ = (
        inp
    )
    k_cache = torch.zeros(1, 1, HEAD_DIM, L)
    v_cache = torch.zeros(1, 1, L, HEAD_DIM)
    k_cache[0, 0, :, 0 : L - T] = prior_k.transpose(0, 1)
    v_cache[0, 0, 0 : L - T, :] = prior_v
    mask = build_mask(T, L)

    dev = xm.xla_device()
    out = gqa_fused_tkg_fwd[cores](
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
    return out, out[0].to(dtype).cpu()


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def run_case_inplace(name, T, L, cores, seed, dtype, idx):
    """One in-place case: o_out gate + cache-window equality + no-stray-write checks at slot idx."""
    inp = make_inputs(T=T, L=L, seed=seed)
    o, active_k, active_v, ret_k, ret_v, init_k, init_v = run_kernel_inplace(
        inp, T, L, cores, dtype, idx
    )
    ref_o, ref_k, ref_v = golden(inp, T, L, dtype, norm_in=False)

    _gate(f"{name}.o_out", o, ref_o, dtype)  # the in-place write must not perturb o_out

    k_slot = ret_k[0, 0, :, idx : idx + T].contiguous()  # [D, T] BHDS window
    v_slot = ret_v[0, 0, idx : idx + T, :].contiguous()  # [T, D] BHSD window
    _bitexact(
        f"{name}.kcache==active_k", k_slot, active_k[0, 0, :, :]
    )  # same SBUF source
    _bitexact(f"{name}.vcache==active_v", v_slot, active_v[0, 0, :, :])
    _gate(f"{name}.kcache_vs_ref", k_slot.reshape(1, 1, HEAD_DIM, T), ref_k, dtype)
    _gate(f"{name}.vcache_vs_ref", v_slot.reshape(1, 1, T, HEAD_DIM), ref_v, dtype)

    rk, ik = (
        ret_k.clone(),
        init_k.clone(),
    )  # blank the written window in both; the rest must match
    rk[0, 0, :, idx : idx + T] = 0
    ik[0, 0, :, idx : idx + T] = 0
    _bitexact(f"{name}.kcache_no_stray", rk, ik)
    rv, iv = ret_v.clone(), init_v.clone()
    rv[0, 0, idx : idx + T, :] = 0
    iv[0, 0, idx : idx + T, :] = 0
    _bitexact(f"{name}.vcache_no_stray", rv, iv)


def run_case_none(name, T, L, cores, seed, dtype):
    """Default path (kv_write_idx=None): kernel returns 3 tensors and o_out matches golden."""
    inp = make_inputs(T=T, L=L, seed=seed)
    out, o = run_kernel_none(inp, T, L, cores, dtype)
    assert len(out) == 3, f"{name}: default path must return 3 tensors, got {len(out)}"
    ref_o, _, _ = golden(inp, T, L, dtype, norm_in=False)
    _gate(f"{name}.o_out", o, ref_o, dtype)


# ---------------------------------------------------------------------------
# pytest entrypoints (fp32 hard gate)
# ---------------------------------------------------------------------------
def test_inplace_fp32_t1_c1_tail():
    run_case_inplace(
        "ip_fp32_T1_L128_c1_tail",
        T=1,
        L=128,
        cores=1,
        seed=11,
        dtype=torch.float32,
        idx=127,
    )


def test_inplace_fp32_t1_c1_interior():
    run_case_inplace(
        "ip_fp32_T1_L128_c1_int",
        T=1,
        L=128,
        cores=1,
        seed=11,
        dtype=torch.float32,
        idx=INTERIOR_IDX,
    )


def test_inplace_fp32_t2_c1_tail():
    run_case_inplace(
        "ip_fp32_T2_L128_c1_tail",
        T=2,
        L=128,
        cores=1,
        seed=12,
        dtype=torch.float32,
        idx=126,
    )


def test_inplace_fp32_t2_c1_interior():
    run_case_inplace(
        "ip_fp32_T2_L128_c1_int",
        T=2,
        L=128,
        cores=1,
        seed=12,
        dtype=torch.float32,
        idx=INTERIOR_IDX,
    )


def test_inplace_fp32_t1_c2_tail():
    run_case_inplace(
        "ip_fp32_T1_L256_c2_tail",
        T=1,
        L=256,
        cores=2,
        seed=13,
        dtype=torch.float32,
        idx=255,
    )


def test_inplace_fp32_t1_c2_interior():
    run_case_inplace(
        "ip_fp32_T1_L256_c2_int",
        T=1,
        L=256,
        cores=2,
        seed=13,
        dtype=torch.float32,
        idx=INTERIOR_IDX,
    )


def test_inplace_fp32_t2_c2_tail():
    run_case_inplace(
        "ip_fp32_T2_L256_c2_tail",
        T=2,
        L=256,
        cores=2,
        seed=14,
        dtype=torch.float32,
        idx=254,
    )


def test_inplace_fp32_t2_c2_interior():
    run_case_inplace(
        "ip_fp32_T2_L256_c2_int",
        T=2,
        L=256,
        cores=2,
        seed=14,
        dtype=torch.float32,
        idx=INTERIOR_IDX,
    )


def test_inplace_none_default_green():
    run_case_none(
        "none_fp32_T1_L128_c1", T=1, L=128, cores=1, seed=11, dtype=torch.float32
    )


# bf16 (max_abs reported; bit-exact cache==active still a hard gate)
def test_inplace_bf16_t2_c2_tail():
    run_case_inplace(
        "ip_bf16_T2_L256_c2_tail",
        T=2,
        L=256,
        cores=2,
        seed=14,
        dtype=torch.bfloat16,
        idx=254,
    )


# ---------------------------------------------------------------------------
def main():
    print(
        "=== In-place KV write: FP32 IO (hard gate: allclose atol=1e-5 rtol=1e-2) ==="
    )
    # cores=1 (single program does both V and K writes).
    run_case_inplace(
        "ip_fp32_T1_L128_c1_tail",
        T=1,
        L=128,
        cores=1,
        seed=11,
        dtype=torch.float32,
        idx=127,
    )
    run_case_inplace(
        "ip_fp32_T1_L128_c1_int",
        T=1,
        L=128,
        cores=1,
        seed=11,
        dtype=torch.float32,
        idx=INTERIOR_IDX,
    )
    run_case_inplace(
        "ip_fp32_T2_L128_c1_tail",
        T=2,
        L=128,
        cores=1,
        seed=12,
        dtype=torch.float32,
        idx=126,
    )
    run_case_inplace(
        "ip_fp32_T2_L128_c1_int",
        T=2,
        L=128,
        cores=1,
        seed=12,
        dtype=torch.float32,
        idx=INTERIOR_IDX,
    )

    # cores=2 (LNC2 write-gating: V on prg 0, K on prg 1; each cache written exactly once).
    run_case_inplace(
        "ip_fp32_T1_L256_c2_tail",
        T=1,
        L=256,
        cores=2,
        seed=13,
        dtype=torch.float32,
        idx=255,
    )
    run_case_inplace(
        "ip_fp32_T1_L256_c2_int",
        T=1,
        L=256,
        cores=2,
        seed=13,
        dtype=torch.float32,
        idx=INTERIOR_IDX,
    )
    run_case_inplace(
        "ip_fp32_T2_L256_c2_tail",
        T=2,
        L=256,
        cores=2,
        seed=14,
        dtype=torch.float32,
        idx=254,
    )
    run_case_inplace(
        "ip_fp32_T2_L256_c2_int",
        T=2,
        L=256,
        cores=2,
        seed=14,
        dtype=torch.float32,
        idx=INTERIOR_IDX,
    )

    print("\n=== Default path unchanged (kv_write_idx=None -> 3 returns) ===")
    run_case_none(
        "none_fp32_T1_L128_c1", T=1, L=128, cores=1, seed=11, dtype=torch.float32
    )
    run_case_none(
        "none_fp32_T2_L256_c2", T=2, L=256, cores=2, seed=14, dtype=torch.float32
    )

    print(
        "\n=== In-place KV write: BF16 IO (max_abs reported; cache==active bit-exact) ==="
    )
    run_case_inplace(
        "ip_bf16_T1_L128_c1_tail",
        T=1,
        L=128,
        cores=1,
        seed=11,
        dtype=torch.bfloat16,
        idx=127,
    )
    run_case_inplace(
        "ip_bf16_T2_L256_c2_int",
        T=2,
        L=256,
        cores=2,
        seed=14,
        dtype=torch.bfloat16,
        idx=INTERIOR_IDX,
    )

    print("\nALL CASES PASSED")


if __name__ == "__main__":
    main()
