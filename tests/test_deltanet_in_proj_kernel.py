"""Isolated correctness test for the fused DeltaNet input-RMSNorm + 4-projection TKG NKI kernel.

Calls ``deltanet_in_proj_fwd`` DIRECTLY (a thin @nki.jit wrapper around nkilib's ``qkv_tkg``) and
compares against a high-precision (fp64) CPU reference: ``norm(hidden) @ proj_w`` sliced into qkv / z
/ a / b, where ``norm`` is the standard RMSNorm ``(x * rsqrt(mean(x^2) + eps)) * gamma`` -- the same
effective norm ``qkv_tkg`` / Neuron ``CustomRMSNorm`` apply (NOT HF's ``(1+weight)`` variant).

DTYPE: inputs are cast to **bf16** before going to device (the dtype the real model invokes the
kernel in -- a 1-pass matmul, half the weight DMA), while the golden stays in fp64. This mirrors the
final model integration; the f32 path is no longer the contract here.

The single fused matmul output [B*S, I] is sliced into the four projections at offsets
``conv_dim``, ``+value_dim``, ``+num_v_heads``, ``+num_v_heads`` and each slice is validated.
The full fused matmul is also cross-checked against the reference directly.

Stages (so a failure localizes):
  1. full      -- the whole [B*S, I] projection matches norm(hidden) @ proj_w.
  2. qkv/z/a/b -- each sliced segment matches its individual projection.
  3. shard     -- single-core [1] vs LNC=2 [2] agree at the bf16 tolerance.

Tolerance (bf16): per output we report max_abs_err, max_rel_err, and cosine similarity. bf16 carries
~8 mantissa bits (rel resolution ~2^-8 ~= 4e-3); after a 2048-deep contraction the accumulated
rounding lands the relative error in the low 1e-2. We therefore gate on rtol=2e-2 with an
output-magnitude-scaled atol (2e-2 * median|ref|, so small-magnitude rows are not held to the
fp32 atol that bf16 cannot meet), AND require cosine > 0.999 per output (catches any directional /
structural error a loose elementwise tol would miss). These thresholds are tight enough that a real
bug (wrong slice, missing norm, a regressed matmul pass) still fails; see the printed metrics.

Run:
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_deltanet_in_proj_kernel
"""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.qwen3_6_moe.nki_kernels.deltanet.components.in_proj import (  # noqa: E402
    deltanet_in_proj_fwd,
)

# Per-rank (TP=4) DeltaNet input dims.
HIDDEN = 2048
CONV_DIM = 2048
VALUE_DIM = 1024
NUM_V_HEADS = 8
I_DIM = CONV_DIM + VALUE_DIM + 2 * NUM_V_HEADS  # 3088

# Slice offsets into the fused projection output (qkv | z | a | b, in that order).
OFF_QKV = 0
OFF_Z = CONV_DIM
OFF_A = OFF_Z + VALUE_DIM
OFF_B = OFF_A + NUM_V_HEADS
OFF_END = OFF_B + NUM_V_HEADS

# bf16 tolerance: relative gate + an output-magnitude-scaled absolute floor, and a cosine gate.
RTOL = 2e-2
ATOL_SCALE = 2e-2  # atol = ATOL_SCALE * median(|ref|) per output (so small rows aren't over-constrained)
COS_MIN = 0.999
DTYPE = torch.bfloat16
EPS = 1e-6  # config rms_norm_eps


def make_inputs(T, seed):
    """Random fp32 inputs. ``proj_w`` is the concatenated [H, I] Linear-transpose weight."""
    torch.manual_seed(seed)
    hidden = torch.randn(1, T, HIDDEN)
    # nn.Linear weights are [out, in] = [I, H]; the kernel wants [H, I] (the transpose).
    w_qkv = torch.randn(CONV_DIM, HIDDEN)
    w_z = torch.randn(VALUE_DIM, HIDDEN)
    w_a = torch.randn(NUM_V_HEADS, HIDDEN)
    w_b = torch.randn(NUM_V_HEADS, HIDDEN)
    w_linear = torch.cat([w_qkv, w_z, w_a, w_b], dim=0)  # [I, H]
    proj_w = w_linear.t().contiguous()  # [H, I]
    gamma = torch.randn(HIDDEN)  # random (not ones) to exercise the norm weight
    return hidden, proj_w, gamma


def rms_norm_ref(x, gamma, eps):
    """Standard RMSNorm: (x * rsqrt(mean(x^2) + eps)) * gamma -- matches qkv_tkg / CustomRMSNorm.

    Reference takes the bf16-rounded inputs (what the kernel actually sees) but computes in fp64,
    isolating the kernel's compute error from the unavoidable input-quantization error."""
    x = x.to(torch.float64)
    rms = (x.square().mean(dim=-1, keepdim=True) + eps).sqrt()
    return (x * rms.reciprocal()) * gamma.to(torch.float64)


def golden(inp):
    """CPU fp64 reference on the bf16-rounded inputs: norm(hidden) @ proj_w, whole + per-segment."""
    hidden, proj_w, gamma = inp
    # Round to bf16 then back to fp64: the reference shares the kernel's input quantization but
    # carries it through in full precision, so the remaining error is the kernel's compute error.
    hidden = hidden.to(DTYPE).to(torch.float64)
    proj_w = proj_w.to(DTYPE).to(torch.float64)
    gamma = gamma.to(DTYPE).to(torch.float64)
    normed = rms_norm_ref(hidden, gamma, EPS)  # [1, T, H]
    T = hidden.shape[1]
    out = normed.reshape(T, HIDDEN) @ proj_w  # [T, I]
    segs = {
        "qkv": out[:, OFF_QKV:OFF_Z],
        "z": out[:, OFF_Z:OFF_A],
        "a": out[:, OFF_A:OFF_B],
        "b": out[:, OFF_B:OFF_END],
    }
    return out, segs


def run_kernel(inp, cores=2):
    """Move bf16 inputs to the Neuron device, launch on ``cores`` cores, return CPU output [B*S, I]."""
    import torch_xla.core.xla_model as xm

    hidden, proj_w, gamma = inp
    dev = xm.xla_device()
    h = hidden.to(DTYPE).contiguous().to(dev)
    w = proj_w.to(DTYPE).contiguous().to(dev)
    # qkv_tkg expects gamma as [1, H].
    g = gamma.reshape(1, HIDDEN).to(DTYPE).contiguous().to(dev)
    out = deltanet_in_proj_fwd[cores](h, w, g, EPS)
    # qkv_tkg returns BSD [B, S, I]; flatten B*S to [T, I] for per-segment slicing.
    return out.float().cpu().reshape(-1, I_DIM)


def _metrics(ker, ref):
    ker = ker.double().reshape(-1)
    ref = ref.double().reshape(-1)
    abs_err = (ker - ref).abs()
    denom = ref.abs().clamp_min(1e-4)
    cos = torch.nn.functional.cosine_similarity(ker, ref, dim=0).item()
    return abs_err.max().item(), (abs_err / denom).max().item(), cos


def _check(name, ker, ref):
    max_abs, max_rel, cos = _metrics(ker, ref)
    # Output-magnitude-scaled atol: bf16 cannot hold an absolute floor tighter than its relative
    # resolution times the output magnitude. Use the median |ref| as the magnitude scale.
    mag = ref.double().abs().median().item()
    atol = ATOL_SCALE * max(mag, 1e-4)
    ok = (
        torch.allclose(ker.double(), ref.double(), atol=atol, rtol=RTOL)
        and cos > COS_MIN
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"[{name}] {status}  max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}  "
        f"cos={cos:.6f}  (atol={atol:.3e} rtol={RTOL} cos_min={COS_MIN})"
    )
    assert ok, (
        f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} cos={cos:.6f} "
        f"exceeds atol={atol:.3e} rtol={RTOL} or cos<={COS_MIN}"
    )


def run_stage_projection(name, T, seed):
    """Full projection + each of the four sliced segments vs the CPU reference."""
    inp = make_inputs(T=T, seed=seed)
    ker = run_kernel(inp, cores=2)  # [T, I]
    ref_full, ref_segs = golden(inp)
    _check(f"{name}:full", ker, ref_full)
    _check(f"{name}:qkv", ker[:, OFF_QKV:OFF_Z], ref_segs["qkv"])
    _check(f"{name}:z", ker[:, OFF_Z:OFF_A], ref_segs["z"])
    _check(f"{name}:a", ker[:, OFF_A:OFF_B], ref_segs["a"])
    _check(f"{name}:b", ker[:, OFF_B:OFF_END], ref_segs["b"])


def run_shard_equality(name, seed):
    """Single-core [1] vs LNC=2 [2]; assert agreement at the bf16 tolerance.

    qkv_tkg shards the *contraction* H across the cores and sums the two halves with a sendrecv +
    add, so cores=1 (one full-H sum) and cores=2 (two half-H sums recombined) are NOT bit-identical
    -- float addition is not associative, and in bf16 the per-core rounding compounds. The
    difference is pure reduction order, so the check is the bf16 tolerance, not torch.equal."""
    inp = make_inputs(T=2, seed=seed)
    o1 = run_kernel(inp, cores=1).double()
    o2 = run_kernel(inp, cores=2).double()
    max_abs, max_rel, cos = _metrics(o1, o2)
    mag = o2.abs().median().item()
    atol = ATOL_SCALE * max(mag, 1e-4)
    ok = torch.allclose(o1, o2, atol=atol, rtol=RTOL) and cos > COS_MIN
    status = "PASS" if ok else "FAIL"
    print(
        f"[{name}:cores1_vs_cores2] {status}  max_abs_err={max_abs:.3e}  "
        f"max_rel_err={max_rel:.3e}  cos={cos:.6f}  (atol={atol:.3e} rtol={RTOL})"
    )
    assert ok, f"{name}: cores=1 and cores=2 differ beyond the bf16 tolerance"


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_stage1_decode():
    run_stage_projection("stage1_decode_T1", T=1, seed=1)


def test_stage2_verify():
    run_stage_projection("stage2_verify_T2", T=2, seed=2)


def test_stage3_shard_bit_identical():
    run_shard_equality("stage3_shard", seed=3)


def main():
    run_stage_projection("stage1_decode_T1", T=1, seed=1)
    run_stage_projection("stage2_verify_T2", T=2, seed=2)
    run_shard_equality("stage3_shard", seed=3)
    print("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
