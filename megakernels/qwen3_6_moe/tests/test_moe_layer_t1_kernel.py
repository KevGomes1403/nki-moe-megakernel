"""T=1 pre-flight for ``moe_layer_fwd`` (draft-step shape), the model-facing @nki.jit entrypoint.

The verify megakernel only ever launches this at T=2. At T=1 with two LNC cores the expert work split
cannot token-shard and falls back to the H-shard path (``moe_h_shard``), where each core owns half the
tp2013 free axis -- gate/up cross-core reduced, down disjoint. This covers T in {1, 2} x launch cores
in {1, 2} at fp32 against the full NeuronMoEBlock-equivalent oracle.

Output is read back as HBM [1, T, H] natural (``output_bsh=True``), so the check is independent of the
SBUF H-permutation.

Gate: torch.allclose(kernel_fp32, oracle_fp32, atol=1e-5, rtol=1e-2). max_abs / max_rel per config.

Run (CORES 0,1):
    cd /home/ubuntu/nki-moe && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    NEURON_CC_FLAGS="--target trn2 --lnc 2" \
    python -m megakernels.qwen3_6_moe.tests.test_moe_layer_t1_kernel
"""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nki_kernels.moe.components.moe_layer import moe_layer_fwd  # noqa: E402

from megakernels.qwen3_6_moe.tests.test_moe_layer_kernel import (  # noqa: E402
    ATOL,
    E_FULL,
    EPS,
    HIDDEN,
    K_FULL,
    RTOL,
    _args,
    _metrics,
    make_inputs,
    oracle_layer,
    repack,
)


def run_layer(inp, T, cores, dtype, K):
    """Launch moe_layer_fwd on `cores` cores; returns the per-rank partial as [T, H]."""
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    args = _args(inp, repack(inp), dtype, dev)
    out = moe_layer_fwd[cores](*args, EPS, K)
    return out.to(torch.float32).cpu().reshape(T, HIDDEN)


def _check(name, ker, ref):
    max_abs, max_rel = _metrics(ker, ref)
    ok = torch.allclose(ker.double(), ref.double(), atol=ATOL, rtol=RTOL)
    print(
        f"[{name}] {'PASS' if ok else 'FAIL'}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
    )
    assert ok, f"{name}: allclose(atol={ATOL} rtol={RTOL}) failed (max_abs={max_abs:.3e})"


def run_case(name, T, cores, seed, E=E_FULL, K=K_FULL):
    inp = make_inputs(T=T, E=E, K=K, seed=seed)
    ref, _ = oracle_layer(inp, K)
    _check(name, run_layer(inp, T, cores=cores, dtype=torch.float32, K=K), ref)


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_fp32_t1_c2():
    run_case("moe_fp32_T1_c2", T=1, cores=2, seed=21)  # H-shard fallback


def test_fp32_t1_c1():
    run_case("moe_fp32_T1_c1", T=1, cores=1, seed=21)


def test_fp32_t2_c2():
    run_case("moe_fp32_T2_c2", T=2, cores=2, seed=22)  # token-shard


def test_fp32_t2_c1():
    run_case("moe_fp32_T2_c1", T=2, cores=1, seed=22)


def main():
    print("=== MoE layer (moe_layer_fwd) -- fp32, E=256, k=8, HBM [1,T,H] readback ===")
    run_case("moe_fp32_T1_c2", T=1, cores=2, seed=21)
    run_case("moe_fp32_T1_c1", T=1, cores=1, seed=21)
    run_case("moe_fp32_T2_c2", T=2, cores=2, seed=22)
    run_case("moe_fp32_T2_c1", T=2, cores=1, seed=22)
    print("\nALL MoE T=1/T=2 CASES PASSED")


if __name__ == "__main__":
    main()
