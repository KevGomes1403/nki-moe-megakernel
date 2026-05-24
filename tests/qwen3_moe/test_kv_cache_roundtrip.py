"""Multi-step KV-cache read/write round-trip test for the speculative megakernel.

WHY THIS TEST EXISTS
--------------------
``test_attention_vs_hf.py`` runs the attention kernel for a SINGLE decode step
against a pre-filled cache. It proves the per-step compute is correct, but it
never feeds a *kernel-written* cache slot back into a subsequent step. The
end-to-end failure under investigation — output that is coherent for the first
token then degenerates into repetition over ~20-30 generated tokens — is
exactly the signature of a KV-cache write/read round-trip bug, and only a
multi-step test can catch it.

WHAT THIS TEST DOES
-------------------
Drives ``attention_only_kernel`` (the exact layer-0 attention call site of
``transformer_qwen3_moe_speculative._multilayer_body``) for N consecutive
decode steps. Each step:
  * generates a fresh, independent hidden state ``X_step`` (the "next token");
  * runs the kernel, which scatters K/V into the cache at ``position_ids``;
  * runs the golden PyTorch reference (``_attention_reference``, imported from
    ``test_attention_vs_hf`` so the two tests cannot drift apart);
  * propagates the *kernel's own* returned cache into the next step.

Because step k reads cache slots written by steps < k, this exercises the
write-then-read round trip directly. ``test_kv_cache_write_lands_at_position``
additionally sweeps the write index across the LNC=2 s_prior shard boundary
(slot 128) and the bucket edge (slot 254).

TOLERANCES
----------
A cache slot holds a post-RoPE K (or raw V) value of ~unit magnitude; one bf16
ULP there is ~1.5e-2, and RoPE rounding is position-dependent (identity at
pos=0, a full ULP by pos~128). So the MEAN abs error is the structural signal:
a slot that is mis-scattered (wrong index / wrong layout) reads stale or zero
data, which blows the mean to O(1). The MAX abs error only bounds bf16 ULP
noise and is held loose. This mirrors the Q-dump diagnostic calibration in
test_attention_vs_hf.py. Do NOT loosen ``CACHE_MEAN_ATOL`` to make a failure
pass — that is the scatter-correctness signal.

INTERPRETING FAILURES
---------------------
Each decode step's ``X_step`` is independent, so the kernel's K/V *write* is a
pure function of that step's input — it does NOT accumulate error across steps.
A correct kernel therefore shows a FLAT per-step error curve. A per-step error
that GROWS with step index is a genuine round-trip bug: a slot written at step
k is being misread at step k+1 (wrong index, wrong layout, or stale data). The
per-step error table is printed (run with ``-s``) so the shape of the curve is
visible.

This test runs on ``nki.simulate`` (CPU NumPy, LNC=2) — same execution path as
test_attention_vs_hf. It issues one simulation per decode step, so it is slower
than the single-step tests; tune ``N_STEPS`` / the parametrize lists below if
iteration time matters.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the sibling modules importable when pytest treats tests/qwen3_moe as a
# package (it has __init__.py).
sys.path.insert(0, str(Path(__file__).parent))

import pytest
import torch

# Importing test_attention_vs_hf installs the simulator all_reduce patch (a
# module-level side effect) AND single-sources the golden reference + kernel
# runner so this test cannot diverge from the single-step test.
from test_attention_vs_hf import (  # noqa: E402
    ATOL_LOOSE,
    RTOL_LOOSE,
    _attention_reference,
    _build_cos_sin_at_pos,
    _max_abs_err,
    _mean_abs_err,
    _run_nki_kernel,
    _seeded,
)
from _attention_only_kernel import (  # noqa: E402
    D_HEAD,
    H,
    NUM_KV_HEADS_TP,
    NUM_Q_HEADS_TP,
)

# --------------------------------------------------------------------------
# Fixed test geometry (B=1, S_tkg=1 — the only TKG shape the megakernel runs).
# S_MAX=256 is the minimum LNC=2-valid bucket (multiple of 256); the s_prior
# axis is sharded across 2 cores at slot 128.
# --------------------------------------------------------------------------

B         = 1
S_TKG     = 1
S_MAX     = 256
DTYPE     = torch.bfloat16
SEED      = 0xCACE
ROPE_BASE = 1_000_000.0

# See module docstring "TOLERANCES". MEAN catches scatter/layout bugs; MAX
# only bounds bf16 ULP noise on unit-magnitude post-RoPE values.
CACHE_MEAN_ATOL = 5e-3
CACHE_MAX_ATOL  = 5e-2          # loose ceiling (~3 bf16 ULP); MEAN is the
                                # real scatter-correctness signal
Y_ATOL          = ATOL_LOOSE    # 1e-2 — attention output is accurate to <5e-3

# Avoid amplifying near-zero bf16 noise when step 0's error is tiny.
GROWTH_FLOOR  = 2.0e-3
GROWTH_FACTOR = 3.0


@pytest.fixture(scope="module")
def weights():
    """Deterministic layer-0 attention weights, shared across all steps/tests."""
    g = torch.Generator().manual_seed(SEED)
    return {
        "Wq": _seeded(g, (NUM_Q_HEADS_TP  * D_HEAD, H), DTYPE, scale=0.02),
        "Wk": _seeded(g, (NUM_KV_HEADS_TP * D_HEAD, H), DTYPE, scale=0.02),
        "Wv": _seeded(g, (NUM_KV_HEADS_TP * D_HEAD, H), DTYPE, scale=0.02),
        "Wo": _seeded(g, (NUM_Q_HEADS_TP  * D_HEAD, H), DTYPE, scale=0.02),
        "qn":   torch.ones(D_HEAD, dtype=DTYPE) + _seeded(g, (D_HEAD,), DTYPE, scale=0.01),
        "kn":   torch.ones(D_HEAD, dtype=DTYPE) + _seeded(g, (D_HEAD,), DTYPE, scale=0.01),
        "gpre": torch.ones(H,      dtype=DTYPE) + _seeded(g, (H,),      DTYPE, scale=0.01),
    }


def _make_inputs(weights, X, K_cache, V_cache, pos):
    """Assemble the input dict consumed by ``_run_nki_kernel``."""
    cos, sin = _build_cos_sin_at_pos(pos, B, D_HEAD, DTYPE, base=ROPE_BASE)
    return {
        "X": X,
        "Wq": weights["Wq"], "Wk": weights["Wk"],
        "Wv": weights["Wv"], "Wo": weights["Wo"],
        "qn": weights["qn"], "kn": weights["kn"], "gpre": weights["gpre"],
        "K_cache": K_cache, "V_cache": V_cache,
        "cos_at_pos": cos, "sin_at_pos": sin,
        "position_ids": torch.full((B, 1), pos, dtype=torch.int32),
        "pos": pos,  # ignored by _run_nki_kernel (not a tensor)
    }


def _empty_cache():
    return torch.zeros(B, NUM_KV_HEADS_TP, S_MAX, D_HEAD, dtype=DTYPE)


def _check_cache_region(nki, ref, label):
    """Assert a cache region matches the golden reference. Returns (mean, max).

    MEAN strict: a mis-scattered slot reads stale/zero data -> mean -> O(1).
    MAX loose:   only bounds position-dependent bf16 RoPE rounding (~ULP).
    """
    mean_e = _mean_abs_err(ref, nki)
    max_e  = _max_abs_err(ref, nki)
    assert mean_e <= CACHE_MEAN_ATOL, (
        f"{label}: mean_abs_err={mean_e:.3e} > {CACHE_MEAN_ATOL:.0e}. A cache "
        f"slot holds stale / wrong data — scatter index or layout bug, not "
        f"bf16 noise (max_abs_err={max_e:.3e})."
    )
    assert max_e <= CACHE_MAX_ATOL, (
        f"{label}: max_abs_err={max_e:.3e} > {CACHE_MAX_ATOL:.0e} "
        f"(~2 bf16 ULP at unit magnitude). mean_abs_err={mean_e:.3e}."
    )
    return mean_e, max_e


# ==========================================================================
# Test 1 — the kernel's in-place scatter must land at the correct slot for
# every position, including the LNC=2 s_prior shard boundary (slot 128) and
# the bucket edge (slot 254). Single step; cache pre-filled with random prior
# context at [0, pos).
# ==========================================================================

@pytest.mark.parametrize("pos", [0, 1, 64, 127, 128, 200, 254])
def test_kv_cache_write_lands_at_position(weights, pos):
    g = torch.Generator().manual_seed(SEED ^ (0x1000 + pos))
    X = _seeded(g, (B, S_TKG, H), DTYPE, scale=0.02)

    K_cache = _empty_cache()
    V_cache = _empty_cache()
    if pos > 0:
        K_cache[:, :, :pos, :] = _seeded(g, (B, NUM_KV_HEADS_TP, pos, D_HEAD), DTYPE, scale=0.1)
        V_cache[:, :, :pos, :] = _seeded(g, (B, NUM_KV_HEADS_TP, pos, D_HEAD), DTYPE, scale=0.1)
    K_orig = K_cache.clone()
    V_orig = V_cache.clone()

    inp = _make_inputs(weights, X, K_cache, V_cache, pos)

    Y_ref, K_ref, V_ref = _attention_reference(
        X=X,
        Wq=weights["Wq"], Wk=weights["Wk"], Wv=weights["Wv"], Wo=weights["Wo"],
        qn=weights["qn"], kn=weights["kn"], gpre=weights["gpre"],
        K_cache=K_orig.clone(), V_cache=V_orig.clone(),
        cos_at_pos=inp["cos_at_pos"], sin_at_pos=inp["sin_at_pos"],
        pos=pos,
    )
    Y_nki, K_nki, V_nki, _ = _run_nki_kernel(inp)

    # The freshly written slot must hold the post-RoPE K / raw V.
    k_mean, k_max = _check_cache_region(
        K_nki[:, :, pos:pos + 1, :], K_ref[:, :, pos:pos + 1, :], f"K@pos={pos}")
    v_mean, v_max = _check_cache_region(
        V_nki[:, :, pos:pos + 1, :], V_ref[:, :, pos:pos + 1, :], f"V@pos={pos}")
    y_err = _max_abs_err(Y_ref, Y_nki)
    print(f"[write pos={pos:3d}]  K mean={k_mean:.3e} max={k_max:.3e}   "
          f"V mean={v_mean:.3e} max={v_max:.3e}   Y={y_err:.3e}")

    # Every other slot must be byte-for-byte untouched — a stray write to a
    # neighbouring slot (off-by-one, wrong LNC shard offset) shows up here.
    if pos > 0:
        torch.testing.assert_close(
            K_nki[:, :, :pos, :].to(torch.float32),
            K_orig[:, :, :pos, :].to(torch.float32),
            atol=0.0, rtol=0.0,
            msg=f"K_cache[:, :, :{pos}, :] modified by the scatter.",
        )
        torch.testing.assert_close(
            V_nki[:, :, :pos, :].to(torch.float32),
            V_orig[:, :, :pos, :].to(torch.float32),
            atol=0.0, rtol=0.0,
            msg=f"V_cache[:, :, :{pos}, :] modified by the scatter.",
        )
    torch.testing.assert_close(
        K_nki[:, :, pos + 1:, :].to(torch.float32),
        K_orig[:, :, pos + 1:, :].to(torch.float32),
        atol=0.0, rtol=0.0,
        msg=f"K_cache[:, :, {pos + 1}:, :] modified by the scatter.",
    )
    torch.testing.assert_close(
        V_nki[:, :, pos + 1:, :].to(torch.float32),
        V_orig[:, :, pos + 1:, :].to(torch.float32),
        atol=0.0, rtol=0.0,
        msg=f"V_cache[:, :, {pos + 1}:, :] modified by the scatter.",
    )

    torch.testing.assert_close(
        Y_nki.to(torch.float32), Y_ref.to(torch.float32),
        atol=Y_ATOL, rtol=RTOL_LOOSE,
        msg=f"Attention output wrong at pos={pos}.",
    )


# ==========================================================================
# Test 2 — the round trip. Run N consecutive decode steps, propagating the
# kernel's OWN returned cache. Step k reads slots written by steps < k, so a
# write/read layout or index mismatch surfaces as a per-step error that grows
# with step index. Parametrized over start positions that (a) stay low, (b)
# cross the LNC=2 s_prior shard boundary at 128, (c) sit near the bucket edge.
# ==========================================================================

N_STEPS = 10


@pytest.mark.parametrize("p0", [4, 120, 240])
def test_kv_cache_multistep_roundtrip(weights, p0):
    assert p0 + N_STEPS - 1 < S_MAX - 1, "decode positions must stay below S_MAX-1"

    # Prefill: random prior context at [0, p0). The kernel cache and the
    # reference cache start identical and are then propagated independently.
    g = torch.Generator().manual_seed(SEED ^ (p0 << 8))
    K_init = _empty_cache()
    V_init = _empty_cache()
    if p0 > 0:
        K_init[:, :, :p0, :] = _seeded(g, (B, NUM_KV_HEADS_TP, p0, D_HEAD), DTYPE, scale=0.1)
        V_init[:, :, :p0, :] = _seeded(g, (B, NUM_KV_HEADS_TP, p0, D_HEAD), DTYPE, scale=0.1)

    K_nki, V_nki = K_init.clone(), V_init.clone()
    K_ref, V_ref = K_init.clone(), V_init.clone()

    y_errs: list[float] = []
    cache_means: list[float] = []

    print(f"\n[multistep p0={p0}]  N_STEPS={N_STEPS}")
    for step in range(N_STEPS):
        pos = p0 + step

        # Fresh, independent "next token" hidden state for this step.
        gx = torch.Generator().manual_seed(SEED ^ (p0 << 8) ^ (step + 1))
        X = _seeded(gx, (B, S_TKG, H), DTYPE, scale=0.02)

        inp = _make_inputs(weights, X, K_nki, V_nki, pos)

        # Golden reference advances its own cache.
        Y_ref, K_ref, V_ref = _attention_reference(
            X=X,
            Wq=weights["Wq"], Wk=weights["Wk"], Wv=weights["Wv"], Wo=weights["Wo"],
            qn=weights["qn"], kn=weights["kn"], gpre=weights["gpre"],
            K_cache=K_ref, V_cache=V_ref,
            cos_at_pos=inp["cos_at_pos"], sin_at_pos=inp["sin_at_pos"],
            pos=pos,
        )

        # Kernel advances its own cache — this is the round trip under test.
        Y_nki, K_nki, V_nki, _ = _run_nki_kernel(inp)

        y_err = _max_abs_err(Y_ref, Y_nki)
        # Compare EVERY written slot [0, pos], not just the freshly written
        # one — a slot corrupted at an earlier step is caught here.
        k_mean, k_max = _check_cache_region(
            K_nki[:, :, :pos + 1, :], K_ref[:, :, :pos + 1, :],
            f"step {step} K[0:{pos + 1}]")
        v_mean, v_max = _check_cache_region(
            V_nki[:, :, :pos + 1, :], V_ref[:, :, :pos + 1, :],
            f"step {step} V[0:{pos + 1}]")
        cache_mean = max(k_mean, v_mean)
        y_errs.append(y_err)
        cache_means.append(cache_mean)

        print(f"  step {step:2d}  pos={pos:3d}:  Y_err={y_err:.3e}   "
              f"cache_mean={cache_mean:.3e}  "
              f"(K mean/max={k_mean:.2e}/{k_max:.2e}  "
              f"V mean/max={v_mean:.2e}/{v_max:.2e})")

        # Hard per-step bound on Y. A round-trip bug blows the very first step
        # that reads a kernel-written slot (step 1, pos=p0+1).
        assert y_err <= Y_ATOL, (
            f"step {step} (pos={pos}): attention output diverged "
            f"(err={y_err:.3e} > {Y_ATOL:.0e}). The kernel is misreading a "
            f"cache slot written by an earlier step — KV-cache round-trip bug."
        )

    # Round-trip invariant: because each step's X is independent, the kernel's
    # K/V write does not accumulate error. A correct kernel therefore has a
    # FLAT per-step error curve. Monotonic growth == a slot written at step k
    # is being misread at step k+1.
    assert y_errs[-1] <= max(y_errs[0], GROWTH_FLOOR) * GROWTH_FACTOR, (
        f"per-step Y error grew across the decode sequence "
        f"({y_errs[0]:.3e} -> {y_errs[-1]:.3e}); a flat curve is expected for a "
        f"correct kernel. Growth indicates kernel-written cache slots are read "
        f"back incorrectly. Full curve: "
        f"{', '.join(f'{e:.2e}' for e in y_errs)}"
    )
    assert cache_means[-1] <= max(cache_means[0], GROWTH_FLOOR) * GROWTH_FACTOR, (
        f"per-step cache mean error grew across the decode sequence "
        f"({cache_means[0]:.3e} -> {cache_means[-1]:.3e}). Full curve: "
        f"{', '.join(f'{e:.2e}' for e in cache_means)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
