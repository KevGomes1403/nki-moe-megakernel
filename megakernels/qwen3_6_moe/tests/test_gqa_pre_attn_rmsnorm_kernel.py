"""Isolated correctness test for the GQA pre-attention RMSNorm TKG building block.

Exercises ``pre_attn_rmsnorm_compose`` (the caller glue over nkilib ``rmsnorm_tkg`` that later drops
into the fused GQA kernel before Stage 0) via a thin @nki.jit harness launched on ONE logical core with
grid [2] (LNC=2). The harness RMSNorms raw HBM hidden [B, T, H] into the SBUF [H0=128, T, H1=16] tile
that qkv_tkg consumes, then DMAs it to HBM purely so the host can read it (the megakernel omits this
store). Grid [2] -> num_H_shards = lnc = 2, so the output H1 columns are laid out as
[shard0_H2(8) | shard1_H2(8)] -- the exact qkv_tkg SBUF-input layout the fused kernel feeds. At T <= 2
< SHARDING_THRESHOLD(18) both cores compute the full replicated norm (no BxS shard, no sendrecv).

Math (standard-weight RMSNorm; gamma is input_layernorm.weight, NOT 1+weight):
    y[t, :] = x[t, :] * rsqrt(mean_H(x[t, :]^2) + eps) * gamma
fp32 square/reduce/rsqrt/scale; IO dtype bf16 or fp32.

Numeric gates (repo rule -- cosine similarity BANNED as a gate):
  - FP32 IO -- HARD gate: torch.allclose(atol=1e-5, rtol=1e-2). MUST pass.
  - bf16 IO -- max_abs(kernel_bf16 - oracle_fp32) <= measured bf16 quantization floor * headroom, where
    the floor = max_abs(oracle_fp32.bf16() - oracle_fp32) on the same tensor and the oracle uses the
    SAME bf16-rounded inputs/gamma (avoids a false fail from gamma rounding).

Run (USE ONLY CORE 2, one logical core, grid [2], LNC=2):
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=2 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_gqa_pre_attn_rmsnorm_kernel
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

from models.qwen3_6_moe.nki_kernels.gqa.components.pre_attn_norm import (  # noqa: E402
    H,
    H0,
    pre_attn_rmsnorm_compose,
)

EPS = 1e-6  # config.rms_norm_eps (used identically by kernel and reference)
LNC = 2  # launch grid / num_H_shards -> [shard0 | shard1] output column layout

# FP32 hard gate (the literal task tolerance).
ATOL = 1e-5
RTOL = 1e-2
# bf16 gate: allow the kernel error to reach the bf16 output-rounding floor with headroom for the
# fp32 reduction-order difference vs torch.
BF16_HEADROOM = 3.0


@nki.jit
def pre_attn_rmsnorm_harness(hidden, gamma):
    """Single-launch [2] harness: RMSNorm raw hidden [B, T, H] into SBUF [H0, T, H1], DMA to HBM so
    the host can read it. hidden: [B, T, H] HBM (raw). gamma: [1, H] HBM (standard-form weight)."""
    B, S, hdim = hidden.shape
    T = B * S
    h1 = hdim // H0

    normed_sb = pre_attn_rmsnorm_compose(hidden, gamma, eps=EPS, hidden_actual=hdim)

    normed_hbm = nl.ndarray((H0, T, h1), dtype=hidden.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=normed_hbm, src=normed_sb)
    return normed_hbm


def make_inputs(t, seed):
    """Random fp32 inputs: hidden [B=1, T=t, H], gamma = randn(H)*0.02 + 1.0 (standard form, mimicking
    the +1 checkpoint convention)."""
    torch.manual_seed(seed)
    hidden = torch.randn(1, t, H)
    gamma = torch.randn(H) * 0.02 + 1.0
    return hidden, gamma


def to_kernel_layout(y_th):
    """Transform an fp32 [T, H] RMSNorm result into rmsnorm_tkg's [H0, T, H1] output layout with
    num_H_shards = LNC (the docstring pseudocode): split H into LNC shards, reshape each to
    [T, H0, H2] and transpose to [H0, T, H2], concatenate the shards on the H-tile axis."""
    t, hdim = y_th.shape
    h2 = hdim // H0 // LNC  # H1 tiles per shard (8)
    h_half = hdim // LNC  # 1024
    parts = []
    for s in range(LNC):
        seg = y_th[:, s * h_half : (s + 1) * h_half]  # [T, H//LNC]
        seg = seg.reshape(t, H0, h2).permute(1, 0, 2)  # [H0, T, H2]
        parts.append(seg)
    return torch.cat(parts, dim=2)  # [H0, T, H1]


def golden(hidden, gamma, io_dtype):
    """CPU RMSNorm oracle matching runtime CustomRMSNorm (plain rsqrt, fp32 accumulate). Rounds the
    inputs to ``io_dtype`` first (the IO contract), computes in fp32, returns the fp32 [T, H] result in
    the kernel's [H0, T, H1] output layout."""
    x = hidden.to(io_dtype).float().reshape(-1, hidden.shape[-1])  # [T, H]
    g = gamma.to(io_dtype).float().reshape(-1)  # [H]
    inv = (x.square().mean(dim=-1, keepdim=True) + EPS).rsqrt()
    y = (x * inv) * g  # [T, H] fp32
    return to_kernel_layout(y)  # [H0, T, H1]


def run_kernel(hidden, gamma, io_dtype):
    """Move inputs to the Neuron device, launch the harness on one logical core with grid [2] (LNC=2),
    return the [H0, T, H1] output cast to fp32 for comparison."""
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    hd = hidden.to(io_dtype).contiguous().to(dev)
    gd = gamma.reshape(1, H).to(io_dtype).contiguous().to(dev)
    out = pre_attn_rmsnorm_harness[2](hd, gd)
    return out.float().cpu()


def _max_abs(a, b):
    return (a.double() - b.double()).abs().max().item()


def _max_rel(a, b):
    denom = b.double().abs().clamp_min(1e-4)
    return ((a.double() - b.double()).abs() / denom).max().item()


def _check_fp32(name, ker, ref):
    """HARD gate: torch.allclose at the literal task tolerance (atol=1e-5, rtol=1e-2)."""
    max_abs = _max_abs(ker, ref)
    max_rel = _max_rel(ker, ref)
    ok = torch.allclose(ker.double(), ref.double(), atol=ATOL, rtol=RTOL)
    print(
        f"[{name}] {'PASS' if ok else 'FAIL'}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
        f"(atol={ATOL} rtol={RTOL})"
    )
    assert ok, (
        f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} exceeds atol=1e-5 rtol=1e-2"
    )


def _check_bf16(name, ker, ref):
    """Model-contract gate: max_abs(kernel_bf16 - oracle_fp32) <= bf16 quantization floor * headroom.
    The floor is the max abs change from rounding the SAME oracle to bf16."""
    max_abs = _max_abs(ker, ref)
    max_rel = _max_rel(ker, ref)
    floor = _max_abs(ref.to(torch.bfloat16).float(), ref)
    gate = floor * BF16_HEADROOM
    ok = max_abs <= gate
    print(
        f"[{name}] {'PASS' if ok else 'FAIL'}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
        f"bf16_floor={floor:.3e}  gate={gate:.3e} (floor*{BF16_HEADROOM:g})"
    )
    assert ok, f"{name}: max_abs={max_abs:.3e} exceeds bf16 floor gate {gate:.3e}"


def run_case(name, t, seed):
    """For both fp32 (hard gate) and bf16 (model contract), validate the [H0, T, H1] output."""
    hidden, gamma = make_inputs(t=t, seed=seed)

    ref32 = golden(hidden, gamma, torch.float32)
    ker32 = run_kernel(hidden, gamma, torch.float32)
    _check_fp32(f"{name}:fp32", ker32, ref32)

    ref16 = golden(hidden, gamma, torch.bfloat16)
    ker16 = run_kernel(hidden, gamma, torch.bfloat16)
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
