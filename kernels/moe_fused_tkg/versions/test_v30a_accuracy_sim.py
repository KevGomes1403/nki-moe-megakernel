"""
Accuracy test for kernel_v30a_sbuf_io using the NKI CPU simulator.

Compares qwen3_moe_fused_tkg_sbuf_io against a pure PyTorch fp32 reference
without requiring Trainium hardware.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    NKI_PRECISE_FP=1 python test_v30a_accuracy_sim.py
"""

import os
import sys

# Must be before any NKI/neuron import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ.setdefault("NKI_PRECISE_FP", "1")  # use real bf16 precision in sim

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nki
from kernel_v30a_sbuf_io import qwen3_moe_fused_tkg_sbuf_io
from torch_neuronx.testing.validation import neuron_allclose

# ── Fixed Qwen3-30B-A3B dims ──────────────────────────────────────────────────
_H   = 2048
_E   = 128
_K   = 8
_I   = 192
_EPS = 1e-6


# ── Pure PyTorch reference ────────────────────────────────────────────────────

def pytorch_moe_reference(inp, gamma, router_w, gate_up_w, down_w, verbose=False):
    """
    Reference implementation of fused RMSNorm + Router + MoE MLP.

    Args:
        inp:       [T, 1, H] or [T, H]  — any float dtype, will be cast to f32
        gamma:     [1, H] or [H]
        router_w:  [H, E]
        gate_up_w: [E, H, 2*I]   (gate cols 0:I, up cols I:2I)
        down_w:    [E, I, H]
    Returns:
        output:    [T, H] float32
    """
    x = inp.float().reshape(-1, _H)       # [T, H]
    T = x.shape[0]
    g  = gamma.float().reshape(_H)        # [H]
    rw = router_w.float()                 # [H, E]
    gu = gate_up_w.float()                # [E, H, 2*I]
    dw = down_w.float()                   # [E, I, H]

    # ── RMSNorm ──
    ss         = x.pow(2).sum(dim=-1, keepdim=True)   # [T, 1]  (sum, not mean)
    norm_factor = torch.rsqrt(ss / _H + _EPS)          # [T, 1]
    x_normed   = x * norm_factor * g                   # [T, H]

    if verbose:
        print(f"  RMSNorm  max|x_normed|={x_normed.abs().max():.4f}")

    # ── Router softmax ──
    logits = x_normed @ rw                             # [T, E]
    probs  = F.softmax(logits, dim=-1)                 # [T, E]

    # ── TopK(8) ──
    top8_vals, top8_idx = torch.topk(probs, k=_K, dim=-1)   # [T, K]

    # ── Normalize affinities (Qwen3 style) ──
    norm_weights = top8_vals / top8_vals.sum(dim=-1, keepdim=True)  # [T, K]

    if verbose:
        print(f"  TopK idx[0]={top8_idx[0].tolist()}")
        print(f"  norm_weights[0]={norm_weights[0].tolist()}")

    # ── Selective-expert MLPs ──
    output = torch.zeros(T, _H, dtype=torch.float32)
    for t in range(T):
        for k in range(_K):
            e = int(top8_idx[t, k].item())
            w = float(norm_weights[t, k].item())
            gate_e = x_normed[t] @ gu[e, :, :_I]   # [I]
            up_e   = x_normed[t] @ gu[e, :, _I:]   # [I]
            inter  = F.silu(gate_e) * up_e          # [I]
            out_e  = inter @ dw[e]                  # [H]
            output[t] += w * out_e

    return output  # [T, H] float32


# ── Accuracy metrics ──────────────────────────────────────────────────────────

def report_accuracy(name, actual, expected, rtol=1e-2, atol=1e-2):
    """Print per-element accuracy summary and return pass/fail."""
    actual   = np.asarray(actual, dtype=np.float32).ravel()
    expected = np.asarray(expected, dtype=np.float32).ravel()

    abs_diff = np.abs(actual - expected)
    rel_diff = abs_diff / (np.abs(expected) + 1e-8)

    max_abs  = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    max_rel  = float(rel_diff.max())
    p99_abs  = float(np.percentile(abs_diff, 99))

    has_nan = bool(np.any(np.isnan(actual)))
    has_inf = bool(np.any(np.isinf(actual)))

    print(f"\n── {name} ──")
    if has_nan:
        print("  FAIL: output contains NaN (uninitialized memory read)")
        return False
    if has_inf:
        print("  FAIL: output contains Inf")
        return False

    print(f"  max_abs_diff : {max_abs:.4e}")
    print(f"  mean_abs_diff: {mean_abs:.4e}")
    print(f"  p99_abs_diff : {p99_abs:.4e}")
    print(f"  max_rel_diff : {max_rel:.4e}")

    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        print(f"  PASS  rtol={rtol}, atol={atol}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {e}")
        return False


# ── Main test ─────────────────────────────────────────────────────────────────

def run_test(seed=42, verbose=False):
    rng = np.random.default_rng(seed)

    def make_bf16(shape, scale=0.1):
        arr = rng.standard_normal(shape).astype(np.float32) * scale
        return torch.tensor(arr).to(torch.bfloat16)

    T = 1
    inp_t       = make_bf16((T, 1, _H))
    gamma_t     = make_bf16((1, _H))
    router_w_t  = make_bf16((_H, _E))
    gate_up_w_t = make_bf16((_E, _H, 2 * _I))
    down_w_t    = make_bf16((_E, _I, _H))

    # ── PyTorch reference (fp32 precision) ──
    print("Running PyTorch fp32 reference...")
    ref_f32 = pytorch_moe_reference(
        inp_t, gamma_t, router_w_t, gate_up_w_t, down_w_t, verbose=verbose
    )
    ref_np = ref_f32.numpy()
    print(f"  Reference output shape: {ref_np.shape}")
    print(f"  Reference sample [0,:4]: {ref_np[0, :4]}")

    # Cast reference to bf16 for apples-to-apples comparison with kernel output
    ref_bf16 = ref_f32.to(torch.bfloat16).float().numpy()

    # ── NKI CPU simulator ──
    print("\nRunning NKI CPU simulator (LNC=2)...")
    # torch bfloat16 → ml_dtypes bfloat16 → numpy (required because NumPy has no native bf16)
    import ml_dtypes
    def to_bf16_np(t: torch.Tensor) -> np.ndarray:
        return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)

    try:
        sim_fn = nki.simulate(qwen3_moe_fused_tkg_sbuf_io)[2]
        sim_out = sim_fn(
            to_bf16_np(inp_t),
            to_bf16_np(gamma_t),
            to_bf16_np(router_w_t),
            to_bf16_np(gate_up_w_t),
            to_bf16_np(down_w_t),
        )
    except Exception as exc:
        print(f"\nSimulator ERROR: {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        return False, None, None

    if isinstance(sim_out, (list, tuple)):
        sim_out = sim_out[0]
    sim_np = np.asarray(sim_out, dtype=np.float32)
    print(f"  Simulator output shape: {sim_np.shape}")
    print(f"  Simulator sample [0,:4]: {sim_np[0, :4]}")

    # ── Comparisons ──
    all_pass = True

    # 1. np.testing.assert_allclose vs fp32 reference
    all_pass &= report_accuracy(
        "Simulator vs fp32 reference",
        sim_np, ref_np,
        rtol=1.6e-2, atol=0.05,
    )

    # 2. np.testing.assert_allclose vs bf16-rounded reference
    all_pass &= report_accuracy(
        "Simulator vs bf16-rounded reference",
        sim_np, ref_bf16,
        rtol=1.6e-2, atol=1e-2,
    )

    # 3. NXDI neuron_allclose (uses max-normalized rtol — matches compiler validation)
    actual_t   = torch.tensor(sim_np)
    expected_t = torch.tensor(ref_np)
    nc = neuron_allclose(actual_t, expected_t, rtol=1.6e-2, atol=1e-5)
    print(f"\n── NXDI neuron_allclose (rtol={1.6e-2}, atol=1e-5) ──")
    print(f"  allclose       : {nc.allclose}")
    print(f"  num_mismatches : {nc.num_mismatches} / {actual_t.numel()}")
    print(f"  max_rel_error  : {nc.max_rel_error:.4e}")
    print(f"  max_abs_error  : {nc.max_abs_error:.4e}")
    if nc.allclose:
        print("  PASS")
    else:
        print("  FAIL")
        all_pass = False

    # 4. Sanity: all-zero output means DMA layout bug
    if np.allclose(sim_np, 0.0, atol=1e-6):
        print("\n  FAIL: simulator output is all zeros — likely a DMA layout bug")
        all_pass = False

    return all_pass, sim_np, ref_np


def run_multi_seed(seeds=(42, 7, 123, 999, 12345)):
    print("=" * 70)
    print("kernel_v30a_sbuf_io — Accuracy Test (NKI CPU Simulator, trn3)")
    print("=" * 70)

    results = []
    for seed in seeds:
        print(f"\n{'─'*70}")
        print(f"Seed {seed}")
        print(f"{'─'*70}")
        ok, _sim, _ref = run_test(seed=seed, verbose=(seed == seeds[0]))
        results.append((seed, bool(ok)))

    print("\n" + "═" * 70)
    print("SUMMARY")
    print("═" * 70)
    all_pass = True
    for seed, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  seed={seed:6d}: {status}")
        all_pass &= ok

    print("═" * 70)
    if all_pass:
        print("OVERALL: PASS — kernel_v30a_sbuf_io is numerically correct")
    else:
        print("OVERALL: FAIL — numerical errors detected (see above)")
    print("═" * 70)
    return all_pass


if __name__ == "__main__":
    ok = run_multi_seed()
    sys.exit(0 if ok else 1)
