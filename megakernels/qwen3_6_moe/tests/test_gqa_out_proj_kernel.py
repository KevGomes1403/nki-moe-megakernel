# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated correctness test for the GQA attention output-projection (o_proj) TKG NKI kernel.

Exercises ``out_proj_compose`` (the head-sharded composable that drops into the fused GQA kernel) via
a thin @nki.jit harness. The harness reproduces the Phase 4 attention core's output layout:
``attn_sb [P_MAX, D_TILES, Tq]`` (head_dim on partition, split into D_TILES=2 tiles of 128; Tq head-major)
by DMA-copying each (q-head, d-tile) 128-wide sub-head of ``attn_in [T, value_dim]`` straight onto the
partition axis. It then calls ``out_proj_compose``, which reorders the sub-heads into
output_projection_tkg's [d, 1, N, T] layout, gathers the other core's sub-heads via sendrecv, and runs
the H-sharded matmul. The TP all-reduce is deferred, so this validates the per-rank o_proj matmul plus
the sub-head reorder / gather layout.

Per-rank (TP=4) Qwen3.6 GQA o_proj dims: q_heads=4, head_dim=256 -> value_dim = 4*256 = 1024 presented
as N = 8 sub-heads of 128; hidden = 2048. out_w is the [value_dim, hidden] transpose of the o_proj
nn.Linear weight, row-indexed by value_dim, so it feeds output_projection_tkg's [N*D, H] weight unreshaped.

Precision / gates:
  - FP32 IO (literal correctness gate): kernel fp32 matmul vs CPU fp32 reference, MUST pass
    torch.allclose(atol=1e-5, rtol=1e-2).
  - BF16 IO: reference rounds inputs to bf16, accumulates in fp32, casts back to bf16 (mirrors the
    kernel). A 1024-deep bf16 reduction cannot hit a fixed 1e-5 atol, so bf16 is gated on a
    magnitude-aware floor (cosine similarity); max_abs / max_rel / ULP / cosine are all reported.

Optional output gate: a separate case folds in gated = attn_in * sigmoid(gate) before the matmul,
exercising the real attn -> gate -> o_proj tail. Tested in both FP32 (hard gate) and BF16.

Run:
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_gqa_out_proj_kernel
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

from models.qwen3_6_moe.nki_kernels.gqa.components.out_proj import (  # noqa: E402
    out_proj_compose,
)

# A3B-realistic per-rank (TP=4) GQA o_proj dims.
P_MAX = 128
HEAD_DIM = 256  # D (one q-head) -> D_TILES = 2 sub-heads of 128
D_TILES = HEAD_DIM // P_MAX  # 2
Q_HEADS = 4  # query heads per rank
VALUE_DIM = Q_HEADS * HEAD_DIM  # 1024 -> N = 8 sub-heads of 128
HIDDEN = 2048  # o_proj output (full hidden)

ATOL = 1e-5
RTOL = 1e-2
COSINE_FLOOR = 0.999  # bf16 magnitude-aware gate (scale-invariant)


def _build_attn_sb(src_hbm, T, dst):
    """DMA each (q-head, d-tile) 128-wide sub-head of src_hbm [T, value_dim] onto the partition axis,
    producing the Phase 4 attention layout dst [P_MAX, D_TILES, Tq] for this core's q-heads.

    dst[d_in, d_tile, h_local*T + t] = src_hbm[t, (c*qh_local + h_local)*HEAD_DIM + d_tile*P_MAX + d_in].
    """
    _, value_dim = src_hbm.shape
    n = nl.num_programs(0)
    c = nl.program_id(0)
    q_heads = value_dim // HEAD_DIM
    qh_local = q_heads // n
    for h_local in range(qh_local):
        for d_tile in range(D_TILES):
            base = (c * qh_local + h_local) * HEAD_DIM + d_tile * P_MAX
            # Partition stride 1: 128 consecutive value_dim elements -> 128 partitions (head_dim on
            # partition). Free stride value_dim: step one token. Strided transpose-load (test only).
            nisa.dma_copy(
                dst=dst[0:P_MAX, d_tile, h_local * T : h_local * T + T],
                src=src_hbm.ap(pattern=[[1, P_MAX], [value_dim, T]], offset=base),
            )


@nki.jit
def out_proj_harness(attn_in, out_w):
    """Build attn_sb in the Phase 4 attention layout from attn_in [T, value_dim] and run the o_proj.
    Returns [T, hidden] HBM (per-rank partial). Launch [n]."""
    T, value_dim = attn_in.shape
    n = nl.num_programs(0)
    qh_local = (value_dim // HEAD_DIM) // n
    Tq = qh_local * T

    attn_sb = nl.ndarray((P_MAX, D_TILES, Tq), dtype=attn_in.dtype, buffer=nl.sbuf)
    _build_attn_sb(attn_in, T, attn_sb)
    return out_proj_compose(attn_sb, out_w, T)


@nki.jit
def out_proj_harness_gated(attn_in, out_w, gate_in):
    """As out_proj_harness, plus a sigmoid output gate: gated = attn_in * sigmoid(gate_in) before the
    matmul. gate_in is presented in the same [T, value_dim] logical layout as attn_in."""
    T, value_dim = attn_in.shape
    n = nl.num_programs(0)
    qh_local = (value_dim // HEAD_DIM) // n
    Tq = qh_local * T

    attn_sb = nl.ndarray((P_MAX, D_TILES, Tq), dtype=attn_in.dtype, buffer=nl.sbuf)
    _build_attn_sb(attn_in, T, attn_sb)
    gate_sb = nl.ndarray((P_MAX, D_TILES, Tq), dtype=gate_in.dtype, buffer=nl.sbuf)
    _build_attn_sb(gate_in, T, gate_sb)
    return out_proj_compose(attn_sb, out_w, T, gate_sb=gate_sb)


def make_inputs(T, seed, gated):
    """Random fp32 inputs. out_w is the [value_dim, hidden] transpose of the o_proj Linear weight."""
    torch.manual_seed(seed)
    attn_in = torch.randn(T, VALUE_DIM)
    out_w = torch.randn(VALUE_DIM, HIDDEN)
    gate_in = torch.randn(T, VALUE_DIM) if gated else None
    return attn_in, out_w, gate_in


def golden(inp, dtype):
    """Reference o_proj. FP32: pure fp32. BF16: round inputs to bf16, fp32 accumulate, cast back to
    bf16 -- mirrors the kernel's precision for the tightest comparison. The sigmoid gate (when present)
    is computed in fp32 and applied before the matmul, matching the kernel."""
    attn_in, out_w, gate_in = inp
    a = attn_in.to(dtype)
    w = out_w.to(dtype)
    if gate_in is not None:
        g = gate_in.to(dtype)
        sig = torch.sigmoid(g.float())
        a = (a.float() * sig).to(dtype)
    out = (a.float() @ w.float()).to(dtype)
    return out


def run_kernel(inp, cores, dtype):
    """Move inputs to the Neuron device, launch on ``cores`` cores (q-head shard), return the assembled
    full [T, hidden] CPU output."""
    import torch_xla.core.xla_model as xm

    attn_in, out_w, gate_in = inp
    dev = xm.xla_device()
    a = attn_in.to(dtype).contiguous().to(dev)
    w = out_w.to(dtype).contiguous().to(dev)
    if gate_in is not None:
        g = gate_in.to(dtype).contiguous().to(dev)
        out = out_proj_harness_gated[cores](a, w, g)
    else:
        out = out_proj_harness[cores](a, w)
    return out.to(dtype).cpu()


def _ulp_distance(ker, ref):
    """Max ULP distance between two bf16 tensors (compared via their int16 bit patterns)."""
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
        ok = allclose  # literal correctness gate
        gate = f"allclose(atol={ATOL} rtol={RTOL})"
    else:
        ok = (
            cos >= COSINE_FLOOR
        )  # magnitude-aware floor for the 1024-deep bf16 reduction
        gate = f"cosine>={COSINE_FLOOR}"
    status = "PASS" if ok else "FAIL"
    print(
        f"[{name}] {status}  max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}  "
        f"max_ulp={ulp}  cosine={cos:.6f}  allclose={allclose}  gate={gate}"
    )
    assert ok, f"{name}: gate {gate} failed (max_abs={max_abs:.3e} cosine={cos:.6f})"


def run_case(name, T, cores, dtype, seed, gated=False):
    """Run the o_proj kernel on ``cores`` cores for ``T`` tokens; assert the assembled [T, hidden]
    matches the reference within the dtype-appropriate gate."""
    inp = make_inputs(T=T, seed=seed, gated=gated)
    ker = run_kernel(inp, cores=cores, dtype=dtype)
    ref = golden(inp, dtype)
    _check(name, ker, ref, dtype)
    return ker


def run_shard_equality(name, T, dtype, seed):
    """Compare cores=1 vs cores=2 assembled outputs; report whether bit-identical (not required)."""
    inp = make_inputs(T=T, seed=seed, gated=False)
    o1 = run_kernel(inp, cores=1, dtype=dtype)
    o2 = run_kernel(inp, cores=2, dtype=dtype)
    bit_identical = torch.equal(o1, o2)
    max_abs, _, ulp, cos = _metrics(o1, o2)
    print(
        f"[{name}:cores1_vs_cores2] bit_identical={bit_identical}  "
        f"max_abs_err={max_abs:.3e}  max_ulp={ulp}  cosine={cos:.6f}"
    )


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_fp32_t1_cores1():
    run_case("fp32_T1_cores1", T=1, cores=1, dtype=torch.float32, seed=1)


def test_fp32_t1_cores2():
    run_case("fp32_T1_cores2", T=1, cores=2, dtype=torch.float32, seed=1)


def test_fp32_t2_cores1():
    run_case("fp32_T2_cores1", T=2, cores=1, dtype=torch.float32, seed=2)


def test_fp32_t2_cores2():
    run_case("fp32_T2_cores2", T=2, cores=2, dtype=torch.float32, seed=2)


def test_bf16_t1_cores1():
    run_case("bf16_T1_cores1", T=1, cores=1, dtype=torch.bfloat16, seed=1)


def test_bf16_t1_cores2():
    run_case("bf16_T1_cores2", T=1, cores=2, dtype=torch.bfloat16, seed=1)


def test_bf16_t2_cores1():
    run_case("bf16_T2_cores1", T=2, cores=1, dtype=torch.bfloat16, seed=2)


def test_bf16_t2_cores2():
    run_case("bf16_T2_cores2", T=2, cores=2, dtype=torch.bfloat16, seed=2)


def test_gated_fp32_t2_cores1():
    run_case(
        "gated_fp32_T2_cores1", T=2, cores=1, dtype=torch.float32, seed=3, gated=True
    )


def test_gated_fp32_t2_cores2():
    run_case(
        "gated_fp32_T2_cores2", T=2, cores=2, dtype=torch.float32, seed=3, gated=True
    )


def test_gated_bf16_t2_cores2():
    run_case(
        "gated_bf16_T2_cores2", T=2, cores=2, dtype=torch.bfloat16, seed=3, gated=True
    )


def test_shard_equality_fp32_t1():
    run_shard_equality("shard_fp32_T1", T=1, dtype=torch.float32, seed=1)


def test_shard_equality_fp32_t2():
    run_shard_equality("shard_fp32_T2", T=2, dtype=torch.float32, seed=2)


def main():
    print("=== FP32 IO (literal correctness gate: allclose) ===")
    run_case("fp32_T1_cores1", T=1, cores=1, dtype=torch.float32, seed=1)
    run_case("fp32_T1_cores2", T=1, cores=2, dtype=torch.float32, seed=1)
    run_case("fp32_T2_cores1", T=2, cores=1, dtype=torch.float32, seed=2)
    run_case("fp32_T2_cores2", T=2, cores=2, dtype=torch.float32, seed=2)

    print("=== BF16 IO (magnitude-aware cosine gate; metrics reported) ===")
    run_case("bf16_T1_cores1", T=1, cores=1, dtype=torch.bfloat16, seed=1)
    run_case("bf16_T1_cores2", T=1, cores=2, dtype=torch.bfloat16, seed=1)
    run_case("bf16_T2_cores1", T=2, cores=1, dtype=torch.bfloat16, seed=2)
    run_case("bf16_T2_cores2", T=2, cores=2, dtype=torch.bfloat16, seed=2)

    print("=== Output gate fold-in (gated = attn_in * sigmoid(gate)) ===")
    run_case(
        "gated_fp32_T2_cores1", T=2, cores=1, dtype=torch.float32, seed=3, gated=True
    )
    run_case(
        "gated_fp32_T2_cores2", T=2, cores=2, dtype=torch.float32, seed=3, gated=True
    )
    run_case(
        "gated_bf16_T2_cores2", T=2, cores=2, dtype=torch.bfloat16, seed=3, gated=True
    )

    print("=== cores=1 vs cores=2 equality ===")
    run_shard_equality("shard_fp32_T1", T=1, dtype=torch.float32, seed=1)
    run_shard_equality("shard_fp32_T2", T=2, dtype=torch.float32, seed=2)
    run_shard_equality("shard_bf16_T2", T=2, dtype=torch.bfloat16, seed=2)

    print("ALL CASES PASSED")


if __name__ == "__main__":
    main()
