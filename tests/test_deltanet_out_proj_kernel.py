"""Isolated correctness test for the DeltaNet output-projection (o_proj) TKG NKI kernel.

Exercises ``out_proj_compose`` (the value-head-sharded composable that will drop into the fused
kernel) via a thin @nki.jit harness defined here. The harness provides the SBUF input the real
composable consumes: each LNC core DMA-copies its value-head slice ``attn_in[:, c*W_core:(c+1)*W_core]``
from HBM into SBUF ``[T, W_core]`` (Layout A, head-major) and calls ``out_proj_compose``. The
composable gathers all heads across cores (sendrecv), transposes to head_dim-on-partition, and runs
``output_projection_tkg`` which H-shards the [T, hidden] output across the cores. The TP all-reduce is
deferred, so this validates the per-rank o_proj matmul + the gather/transpose layout.

DTYPE: bf16 io with fp32 matmul accumulate (the model's contract). The reference rounds attn_in/out_w
to bf16, computes (attn_in.float() @ out_w.float()) accumulating in fp32, and casts back to bf16 --
mirroring the kernel's precision for the tightest comparison.

Tolerance: bf16 atol=1e-5, rtol=1e-2 against the bf16 reference; we also report max-abs-diff and the
max ULP distance per case. A matmul reduction is not bit-identical to torch's, but lands within ~1 ULP.

Run:
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_deltanet_out_proj_kernel
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

from models.qwen3_6_moe.nki_kernels.nki_deltanet_out_proj import (  # noqa: E402
    out_proj_compose,
)

# A3B-realistic per-rank (TP=4) DeltaNet o_proj dims.
HEAD_DIM = 128  # d
HV = 8  # value heads per rank
VALUE_DIM = HV * HEAD_DIM  # 1024
HIDDEN = 2048  # o_proj output (full hidden)

ATOL = 1e-5
RTOL = 1e-2
DTYPE = torch.bfloat16


@nki.jit
def out_proj_harness(attn_in, out_w):
    """Test harness: each core DMA-copies its value-head slice of attn_in [T, value_dim] into SBUF
    [T, W_core] (Layout A head-major) and calls out_proj_compose. Returns [T, hidden] HBM. Launch [n]."""
    T, value_dim = attn_in.shape
    n = nl.num_programs(0)
    c = nl.program_id(0)
    W_core = value_dim // n
    col_off = c * W_core

    attn_sb = nl.ndarray((T, W_core), dtype=attn_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=attn_sb[0:T, 0:W_core],
        src=attn_in.ap(pattern=[[value_dim, T], [1, W_core]], offset=col_off),
    )
    return out_proj_compose(attn_sb, out_w)


def make_inputs(T, seed):
    """Random fp32 inputs. ``out_w`` is the [value_dim, hidden] transpose of the o_proj Linear weight."""
    torch.manual_seed(seed)
    attn_in = torch.randn(T, VALUE_DIM)
    out_w = torch.randn(VALUE_DIM, HIDDEN)
    return attn_in, out_w


def golden(inp):
    """bf16-rounded inputs, fp32 matmul accumulate, cast back to bf16 -- mirrors the kernel."""
    attn_in, out_w = inp
    a = attn_in.to(DTYPE)
    w = out_w.to(DTYPE)
    out = (a.float() @ w.float()).to(DTYPE)
    return out


def run_kernel(inp, cores):
    """Move bf16 inputs to the Neuron device, launch on ``cores`` cores (value-head shard), return
    the assembled full [T, hidden] CPU output (bf16)."""
    import torch_xla.core.xla_model as xm

    attn_in, out_w = inp
    dev = xm.xla_device()
    a = attn_in.to(DTYPE).contiguous().to(dev)
    w = out_w.to(DTYPE).contiguous().to(dev)
    out = out_proj_harness[cores](a, w)
    return out.to(DTYPE).cpu()


def _ulp_distance(ker, ref):
    """Max ULP distance between two bf16 tensors (compared via their fp32 upcasts' bit patterns)."""
    k = ker.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    r = ref.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    # Map to a monotonic ordering so the integer gap equals the count of representable bf16 steps.
    k = torch.where(k < 0, 0x8000 - k, k)
    r = torch.where(r < 0, 0x8000 - r, r)
    return (k - r).abs().max().item()


def _metrics(ker, ref):
    kd = ker.double()
    rd = ref.double()
    abs_err = (kd - rd).abs()
    denom = rd.abs().clamp_min(1e-4)
    return abs_err.max().item(), (abs_err / denom).max().item(), _ulp_distance(ker, ref)


def _check(name, ker, ref):
    max_abs, max_rel, ulp = _metrics(ker, ref)
    ok = torch.allclose(ker.double(), ref.double(), atol=ATOL, rtol=RTOL)
    status = "PASS" if ok else "FAIL"
    print(
        f"[{name}] {status}  max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}  "
        f"max_ulp={ulp}  (atol={ATOL} rtol={RTOL})"
    )
    assert ok, (
        f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} max_ulp={ulp} exceeds gate"
    )


def run_case(name, T, cores, seed):
    """Run the o_proj kernel on ``cores`` cores for ``T`` tokens; assert the assembled [T, hidden]
    matches the bf16 reference within the gate."""
    inp = make_inputs(T=T, seed=seed)
    ker = run_kernel(inp, cores=cores)
    ref = golden(inp)
    _check(f"{name}", ker, ref)
    return ker


def run_shard_equality(name, T, seed):
    """Compare cores=1 vs cores=2 assembled outputs; report whether bit-identical (not required)."""
    inp = make_inputs(T=T, seed=seed)
    o1 = run_kernel(inp, cores=1)
    o2 = run_kernel(inp, cores=2)
    bit_identical = torch.equal(o1, o2)
    max_abs, max_rel, ulp = _metrics(o1.double(), o2.double())
    print(
        f"[{name}:cores1_vs_cores2] bit_identical={bit_identical}  "
        f"max_abs_err={max_abs:.3e}  max_ulp={ulp}"
    )


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_t1_cores1():
    run_case("T1_cores1", T=1, cores=1, seed=1)


def test_t1_cores2():
    run_case("T1_cores2", T=1, cores=2, seed=1)


def test_t2_cores1():
    run_case("T2_cores1", T=2, cores=1, seed=2)


def test_t2_cores2():
    run_case("T2_cores2", T=2, cores=2, seed=2)


def test_shard_equality_t1():
    run_shard_equality("shard_T1", T=1, seed=1)


def test_shard_equality_t2():
    run_shard_equality("shard_T2", T=2, seed=2)


def main():
    run_case("T1_cores1", T=1, cores=1, seed=1)
    run_case("T1_cores2", T=1, cores=2, seed=1)
    run_case("T2_cores1", T=2, cores=1, seed=2)
    run_case("T2_cores2", T=2, cores=2, seed=2)
    run_shard_equality("shard_T1", T=1, seed=1)
    run_shard_equality("shard_T2", T=2, seed=2)
    print("ALL CASES PASSED")


if __name__ == "__main__":
    main()
