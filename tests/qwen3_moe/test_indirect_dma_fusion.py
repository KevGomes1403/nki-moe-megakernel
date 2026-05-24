"""Parity test for the NKI_MOE_INDIRECT_DMA_FUSION=1 path.

Phase 1 of the indirect-DMA fusion plan fuses the per-expert gate+up weight
DMAs in ``nki_kernels/moe/selective_expert_impl.py`` so that one DMA per
expert (against the contiguous ``[H, 2*I]`` fused TensorView) replaces two
DMAs per expert (against the two ``[H, I]`` per-projection views, whose
contiguous-inner is only ``I*bf16=384B`` on Qwen3-30B-A3B).

This test exercises both paths on the SAME input and asserts the outputs
match within bf16 ulp.

Because ``NKI_MOE_INDIRECT_DMA_FUSION`` is read at module import time
inside ``selective_expert_impl.py`` AND inside the compiler caches the
@nki.jit kernel build, we run each path in a dedicated SUBPROCESS to
guarantee a clean re-import / re-compile.

The kernel under test (``_indirect_dma_fusion_kernel.py``) is a minimal
HBM-input wrapper around ``moe_tkg`` with ``is_all_expert=False`` and
``POST_SCALE`` affinities — i.e., the same code path the speculative
megakernel exercises for the MoE block, minus the rmsnorm/router/AR-gather
scaffolding (which is unaffected by Phase 1).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


# --------------------------------------------------------------------------
# Test shapes — close to Qwen3-30B-A3B at TP=4 (I_PER_RANK = 768/4 = 192).
# We bump I to 256 (multiple of I0=128) because the vendored down-projection
# hoisted-weight path requires `I % 128 == 0` (it does
# `reshape_dim(dim=0, shape=(num_total_128_tiles_per_I, I0))` which asserts
# that the source dim equals `num_total_128_tiles_per_I * I0`). The fusion
# change only touches the gate/up path; using I=256 still exercises the
# fused-vs-baseline divergence we need to test.
# --------------------------------------------------------------------------

H = 2048
E = 128
TOP_K = 8
I_PER_EXPERT = 256


# --------------------------------------------------------------------------
# bf16 <-> numpy plumbing
#
# np.savez does NOT preserve the ml_dtypes.bfloat16 dtype tag across save/load
# (it becomes opaque ``|V2``). We serialize bf16 as raw uint16 bytes plus a
# dtype tag and reconstruct in the runner.
# --------------------------------------------------------------------------

def _torch_bf16_to_uint16(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.bfloat16
    return t.contiguous().view(torch.uint16).numpy()


def _torch_to_numpy_preserving(t: torch.Tensor):
    """Return (np_array, dtype_tag). dtype_tag tells the runner how to recover."""
    if t.dtype == torch.bfloat16:
        return _torch_bf16_to_uint16(t), "bfloat16_u16"
    if t.dtype == torch.float32:
        return t.contiguous().numpy(), "float32"
    if t.dtype == torch.uint32:
        return t.contiguous().numpy().astype(np.uint32), "uint32"
    if t.dtype == torch.int32:
        return t.contiguous().numpy().astype(np.uint32), "uint32"
    raise ValueError(f"Unhandled dtype: {t.dtype}")


def _numpy_to_torch_bf16(arr: np.ndarray) -> torch.Tensor:
    import ml_dtypes
    if arr.dtype == ml_dtypes.bfloat16:
        u16 = arr.view(np.uint16)
        return torch.from_numpy(u16.copy()).view(torch.bfloat16)
    if arr.dtype == np.uint16:
        return torch.from_numpy(arr.copy()).view(torch.bfloat16)
    return torch.from_numpy(arr.copy())


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _build_random_inputs(T: int, seed: int):
    """Build random Qwen3-MoE-shaped inputs for the parity test."""
    g = torch.Generator().manual_seed(seed)
    # Hidden in canonical small magnitude (post-RMSNorm scale).
    hidden = (torch.randn(T, H, generator=g, dtype=torch.float32) * 0.05).to(torch.bfloat16)
    # Random gate_up weights laid out [E, H, 2, I] (the moe_tkg-expected layout).
    gate_up_w = (torch.randn(E, H, 2, I_PER_EXPERT, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    down_w = (torch.randn(E, I_PER_EXPERT, H, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    # Build a sparse top-K affinity vector: per-token K experts picked at random.
    expert_index = torch.zeros(T, TOP_K, dtype=torch.uint32)
    expert_affinities = torch.zeros(T, E, dtype=torch.float32)
    for t in range(T):
        perm = torch.randperm(E, generator=g)[:TOP_K]
        logits = torch.randn(TOP_K, generator=g, dtype=torch.float32)
        probs = torch.softmax(logits, dim=0)
        probs = probs / probs.sum()
        expert_index[t] = perm.to(torch.uint32)
        for k in range(TOP_K):
            expert_affinities[t, perm[k].item()] = probs[k]
    return {
        "hidden": hidden,
        "gate_up_w": gate_up_w,
        "down_w": down_w,
        "expert_affinities": expert_affinities,
        "expert_index": expert_index,
    }


@pytest.fixture(scope="module")
def inputs_T1():
    """Single-token (TKG) inputs."""
    return _build_random_inputs(T=1, seed=0xC0FFEE01)


@pytest.fixture(scope="module")
def inputs_T4():
    """Multi-token inputs (speculative-decode-like). T=4 exercises the
    cross-token loop in selective_expert_impl plus the prefetch-ring
    re-priming per token. T=4 is divisible by N_PRGS=2 so the shard_on_T
    branch is also exercised."""
    return _build_random_inputs(T=4, seed=0xC0FFEE02)


# --------------------------------------------------------------------------
# Subprocess runner
# --------------------------------------------------------------------------

# Inline script that loads the kernel and runs nki.simulate. Uses environment
# variable NKI_MOE_INDIRECT_DMA_FUSION (set by the parent test) to flip path.
_RUNNER_SCRIPT = r"""
import os
import sys
import numpy as np
import ml_dtypes
from pathlib import Path

repo_root = Path(os.environ["MOE_FUSION_TEST_REPO_ROOT"])
tests_dir = repo_root / "tests" / "qwen3_moe"
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(tests_dir))

import nki

# Install simulator all_reduce single-rank patch directly here so we don't
# pull in the broken test_attention_vs_hf module.
def _install_simulator_allreduce_patch():
    from nki.backends import simulator as _sim
    def _all_reduce_single_rank(dsts, srcs, reduce_op, replica_group, cc_dim):
        groups = replica_group
        if len(groups) != 1 or any(len(g) != 1 for g in groups):
            raise NotImplementedError("only single-rank supported")
        for dst, src in zip(dsts, srcs):
            data = src.get_data()
            from nki.backends.simulator.dtypes import to_numpy_dtype
            dst_np_dtype = to_numpy_dtype(dst.dtype)
            if data.dtype != dst_np_dtype:
                data = data.astype(dst_np_dtype)
            dst.set_data(data)
    _sim.all_reduce = _all_reduce_single_rank
_install_simulator_allreduce_patch()

# Import AFTER env is set so module-level flag reads pick up the right value.
from _indirect_dma_fusion_kernel import moe_fusion_kernel
from nki_kernels.moe.selective_expert_impl import (
    _MOE_INDIRECT_DMA_FUSION,
    _MOE_FUSION_ENABLED,
)

print(f"# [runner] NKI_MOE_INDIRECT_DMA_FUSION env={os.environ.get('NKI_MOE_INDIRECT_DMA_FUSION', '<unset>')}", flush=True)
print(f"# [runner] _MOE_INDIRECT_DMA_FUSION module flag = {_MOE_INDIRECT_DMA_FUSION}", flush=True)
print(f"# [runner] _MOE_FUSION_ENABLED module flag      = {_MOE_FUSION_ENABLED}", flush=True)

inputs_path = os.environ["MOE_FUSION_TEST_INPUTS"]
output_path = os.environ["MOE_FUSION_TEST_OUTPUT"]

# Load and reconstruct dtypes. We stored bf16 arrays as raw uint16 with a
# sidecar dtype tag '<name>__dtype' = 'bfloat16_u16'.
npz = np.load(inputs_path, allow_pickle=False)

def _load(name):
    arr = npz[name]
    tag = str(npz[name + "__dtype"])
    if tag == "bfloat16_u16":
        return arr.view(ml_dtypes.bfloat16)
    if tag == "float32":
        return arr.astype(np.float32)
    if tag == "uint32":
        return arr.astype(np.uint32)
    raise ValueError(f"Unknown dtype tag: {tag}")

hidden = _load("hidden")
gate_up_w = _load("gate_up_w")
down_w = _load("down_w")
expert_affinities = _load("expert_affinities")
expert_index = _load("expert_index")

print(f"# [runner] hidden {hidden.shape} {hidden.dtype}", flush=True)
print(f"# [runner] gate_up_w {gate_up_w.shape} {gate_up_w.dtype}", flush=True)
print(f"# [runner] down_w {down_w.shape} {down_w.dtype}", flush=True)
print(f"# [runner] expert_affinities {expert_affinities.shape} {expert_affinities.dtype}", flush=True)
print(f"# [runner] expert_index {expert_index.shape} {expert_index.dtype}", flush=True)

result = nki.simulate(moe_fusion_kernel[2])(
    hidden, gate_up_w, down_w, expert_affinities, expert_index,
    replica_groups=([0],),
)
out_np = result[0]
print(f"# [runner] output {out_np.shape} {out_np.dtype}", flush=True)

# Store output as uint16 if it's bf16, with dtype tag for round-trip.
if out_np.dtype == ml_dtypes.bfloat16:
    np.savez(output_path, output=out_np.view(np.uint16), output__dtype=np.array("bfloat16_u16"))
elif out_np.dtype == np.float32:
    np.savez(output_path, output=out_np, output__dtype=np.array("float32"))
else:
    np.savez(output_path, output=out_np, output__dtype=np.array(str(out_np.dtype)))
"""


def _save_inputs(inputs: dict, path: Path):
    """Serialize input tensors with explicit dtype tags so the subprocess
    can rebuild bf16/fp32 arrays cleanly across np.savez."""
    payload = {}
    for name, t in inputs.items():
        arr, tag = _torch_to_numpy_preserving(t)
        payload[name] = arr
        payload[name + "__dtype"] = np.array(tag)
    np.savez(path, **payload)


def _run_kernel_with_flag(flag_value: str, inputs: dict, tmp_path: Path):
    """Run the moe_fusion_kernel in a subprocess with the given env flag value.

    Returns the kernel output as a numpy array (bf16 reconstructed from uint16).
    """
    import ml_dtypes  # noqa: F401 — needed at output load time
    inputs_npz = tmp_path / f"inputs_{flag_value}.npz"
    output_npz = tmp_path / f"output_{flag_value}.npz"

    _save_inputs(inputs, inputs_npz)

    repo_root = Path(__file__).parent.parent.parent
    env = os.environ.copy()
    env["NKI_MOE_INDIRECT_DMA_FUSION"] = flag_value
    # Ensure other related flags are at their defaults so they don't perturb.
    env.setdefault("NKI_MOE_ENABLE_FUSION", "0")
    env.setdefault("NKI_MOE_LEGACY_GATE_UP_WEIGHT_LOAD", "0")
    env["MOE_FUSION_TEST_REPO_ROOT"] = str(repo_root)
    env["MOE_FUSION_TEST_INPUTS"] = str(inputs_npz)
    env["MOE_FUSION_TEST_OUTPUT"] = str(output_npz)

    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        msg = (
            f"Subprocess failed with NKI_MOE_INDIRECT_DMA_FUSION={flag_value}.\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )
        raise RuntimeError(msg)

    # Echo runner stdout into the pytest log for visibility.
    sys.stdout.write(proc.stdout)
    if proc.stderr.strip():
        sys.stderr.write(proc.stderr)

    npz = np.load(output_npz)
    out = npz["output"]
    tag = str(npz["output__dtype"])
    import ml_dtypes
    if tag == "bfloat16_u16":
        return out.view(ml_dtypes.bfloat16)
    if tag == "float32":
        return out
    return out


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_indirect_dma_fusion_T1_parity(inputs_T1, tmp_path):
    """T=1 (single-token TKG) parity between baseline and fused paths.

    Acceptance: bit-identical at bf16 if possible; otherwise within the
    same atol=rtol=1e-2 strict tolerance the existing
    ``test_moe_vs_hf.py`` uses.
    """
    out_baseline = _run_kernel_with_flag("0", inputs_T1, tmp_path)
    out_fused = _run_kernel_with_flag("1", inputs_T1, tmp_path)

    assert out_baseline.shape == out_fused.shape, (
        f"shape mismatch: baseline={out_baseline.shape} fused={out_fused.shape}"
    )
    assert out_baseline.dtype == out_fused.dtype, (
        f"dtype mismatch: baseline={out_baseline.dtype} fused={out_fused.dtype}"
    )

    out_baseline_t = _numpy_to_torch_bf16(out_baseline)
    out_fused_t = _numpy_to_torch_bf16(out_fused)

    # Compute multiple metrics for diagnostic output.
    a = out_baseline_t.to(torch.float32)
    b = out_fused_t.to(torch.float32)
    diff = (a - b).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    # bf16 ulp at magnitude ~1.0: ~1/128 ~= 0.0078.
    print(f"[T=1] max_abs={max_abs:.6e}  mean_abs={mean_abs:.6e}")

    # Tight tolerance: same compiler, same math, just different DMA pattern.
    ATOL = 1e-2
    RTOL = 1e-2
    torch.testing.assert_close(
        out_fused_t.to(torch.float32),
        out_baseline_t.to(torch.float32),
        atol=ATOL,
        rtol=RTOL,
        msg=lambda m: (
            f"NKI_MOE_INDIRECT_DMA_FUSION=1 output diverges from baseline "
            f"(atol={ATOL}, rtol={RTOL}). max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}.\n{m}"
        ),
    )

    if max_abs == 0.0:
        print(f"[T=1] BIT-EXACT match between baseline and fused paths")
    else:
        print(
            f"[T=1] Outputs differ by max_abs={max_abs:.6e} (bf16 ulp ~7.8e-3 at magnitude 1.0); "
            f"within tolerances atol={ATOL}, rtol={RTOL}."
        )


def test_indirect_dma_fusion_T4_parity(inputs_T4, tmp_path):
    """T=4 (multi-token speculative-decode-like) parity between baseline and
    fused paths. Exercises the per-token outer loop and prefetch-ring
    re-priming, ensuring that the fused slot is correctly re-loaded for
    each token's K=0 expert."""
    out_baseline = _run_kernel_with_flag("0", inputs_T4, tmp_path)
    out_fused = _run_kernel_with_flag("1", inputs_T4, tmp_path)

    assert out_baseline.shape == out_fused.shape, (
        f"shape mismatch: baseline={out_baseline.shape} fused={out_fused.shape}"
    )
    assert out_baseline.dtype == out_fused.dtype, (
        f"dtype mismatch: baseline={out_baseline.dtype} fused={out_fused.dtype}"
    )

    out_baseline_t = _numpy_to_torch_bf16(out_baseline)
    out_fused_t = _numpy_to_torch_bf16(out_fused)

    a = out_baseline_t.to(torch.float32)
    b = out_fused_t.to(torch.float32)
    diff = (a - b).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"[T=4] max_abs={max_abs:.6e}  mean_abs={mean_abs:.6e}")

    ATOL = 1e-2
    RTOL = 1e-2
    torch.testing.assert_close(
        out_fused_t.to(torch.float32),
        out_baseline_t.to(torch.float32),
        atol=ATOL,
        rtol=RTOL,
        msg=lambda m: (
            f"NKI_MOE_INDIRECT_DMA_FUSION=1 output diverges from baseline at T=4 "
            f"(atol={ATOL}, rtol={RTOL}). max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}.\n{m}"
        ),
    )

    if max_abs == 0.0:
        print(f"[T=4] BIT-EXACT match between baseline and fused paths")
    else:
        print(
            f"[T=4] Outputs differ by max_abs={max_abs:.6e} (bf16 ulp ~7.8e-3 at magnitude 1.0); "
            f"within tolerances atol={ATOL}, rtol={RTOL}."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
