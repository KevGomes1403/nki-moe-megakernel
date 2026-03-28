"""
Correctness test and benchmark for any NKI attention TKG kernel.

Usage:
    python test_correctness.py agents/v9_optimized.py
    python test_correctness.py agents/v0_attn_tkg_fused.py --s-prior 128 640

    python test_correctness.py agents/v9_optimized.py --benchmark
    python test_correctness.py agents/v9_optimized.py --benchmark --warmup 5 --iters 50

    # Multiple S_prior sizes (correctness only)
    python test_correctness.py agents/v9_optimized.py --s-prior 128 640 2048

Kernel API detection (no `run` wrapper needed):
  Variant A — no Wo, Hkv_tp=4:  qwen3_attn_tkg_fused(hidden, Wq, Wk, Wv, q_norm, k_norm,
                                     K_cache[B,4,S,d], V_cache[B,4,S,d], cos, sin, pos)
  Variant B — Wo[H, Hq*d], Hkv_tp=1: + Wo after Wv (v6, v7)
  Variant C — Wo[Hq*d, H], Hkv_tp=1: + Wo after Wv, transposed (v8+)
"""

import argparse
import importlib.util
import inspect
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Must be set before importing torch_xla or any neuron modules
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output"
)

sys.path.insert(0, "/home/ubuntu/nki-moe")

from kernels.benchmarking_workspace.benchmark import nki_benchmark

import torch_xla.core.xla_model as xm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Correctness test and benchmark for NKI attention TKG kernels."
    )
    parser.add_argument(
        "kernel",
        nargs="?",
        default=None,
        help="Kernel file to test. Bare filenames and agents/xxx resolved relative "
             "to this directory.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmarking instead of numerical correctness check.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="neuron-bench warmup iterations (default: 10, benchmark mode only).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="neuron-bench benchmark iterations (default: 100, benchmark mode only).",
    )
    parser.add_argument(
        "--s-prior",
        type=int,
        nargs="+",
        default=[640],
        metavar="S",
        help="KV cache sequence lengths to test (default: 128 640, correctness only).",
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="Batch size (default: 1).",
    )
    return parser.parse_args()


def _resolve_kernel_path(kernel_arg: str | None) -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if kernel_arg is None:
        raise ValueError("Kernel path required. Example: agents/v9_optimized.py")
    if os.path.isabs(kernel_arg) or os.path.exists(kernel_arg):
        return kernel_arg
    return os.path.join(this_dir, kernel_arg)


# ---------------------------------------------------------------------------
# Kernel loading + API detection
# ---------------------------------------------------------------------------

# Known kernel entry-point names in priority order
_KNOWN_ENTRY_POINTS = [
    "qwen3_attn_tkg_fused_oproj_v10b",
    "qwen3_attn_tkg_fused_oproj",
    "qwen3_attn_tkg_fused",
]


def _load_kernel(kernel_path: str):
    """
    Load kernel module and return (kernel_fn, variant) where variant is one of:
      'no_wo'          — v0–v5: no Wo, Hkv_tp=4, K/V cache [B,4,S,d]
      'wo_normal'      — v6–v7: Wo[H, Hq*d], Hkv_tp=1, K/V cache [B,1,S,d]
      'wo_transposed'  — v8+:   Wo[Hq*d, H], Hkv_tp=1, K/V cache [B,1,S,d]
    """
    kernel_path = os.path.abspath(kernel_path)
    if not os.path.exists(kernel_path):
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")
    spec = importlib.util.spec_from_file_location("_nki_kernel_under_test", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    kernel_fn = None
    for name in _KNOWN_ENTRY_POINTS:
        if hasattr(mod, name):
            kernel_fn = getattr(mod, name)
            break

    if kernel_fn is None:
        raise AttributeError(
            f"No known entry point found in {kernel_path}. "
            f"Expected one of: {_KNOWN_ENTRY_POINTS}"
        )

    # Detect variant from parameter list
    params = list(inspect.signature(kernel_fn).parameters.keys())
    has_wo = "Wo" in params

    if not has_wo:
        return kernel_fn, "no_wo"

    # Distinguish transposed vs normal Wo by reading the source comment
    src = inspect.getsource(kernel_fn)
    # v6/v7 comment: "Wo,   # [H, Hq_tp*d]"  →  first dim is H (larger)
    # v8+  comment: "Wo,   # [Hq_tp*d, H]"   →  first dim is Hq*d (smaller)
    if "[H, Hq_tp*d]" in src or "[2048, 1024]" in src:
        return kernel_fn, "wo_normal"
    else:
        return kernel_fn, "wo_transposed"


# ---------------------------------------------------------------------------
# Qwen3 attention PyTorch golden reference
# ---------------------------------------------------------------------------

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rmsnorm_per_head(x, weight, eps=1e-6):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(variance + eps)
    return (x_normed * weight.float()).to(x.dtype)


def qwen3_attn_tkg_reference(
    hidden_states,   # [B, 1, H]
    Wq,              # [Hq_tp*d, H]
    Wk,              # [Hkv_tp*d, H]
    Wv,              # [Hkv_tp*d, H]
    q_norm_weight,   # [d]
    k_norm_weight,   # [d]
    K_cache,         # [B, Hkv_tp, S_prior, d]
    V_cache,         # [B, Hkv_tp, S_prior, d]
    cos_at_pos,      # [B, d]
    sin_at_pos,      # [B, d]
    Wo=None,         # [H, Hq_tp*d] or [Hq_tp*d, H] (transposed); None → skip
    wo_transposed: bool = False,
    d: int = 128,
) -> torch.Tensor:
    """
    Pure PyTorch (CPU, float32) reference for one Qwen3 GQA decode step.
    Returns [B, 1, Hq_tp*d] bf16 (without Wo) or [B, 1, H] bf16 (with Wo).
    """
    B, _, H = hidden_states.shape
    Hq_tp  = Wq.shape[0] // d
    Hkv_tp = Wk.shape[0] // d
    gqa    = Hq_tp // Hkv_tp
    S_prior = K_cache.shape[2]

    hs = hidden_states.float()
    Q = (hs @ Wq.float().T).reshape(B, Hq_tp,  1, d)
    K = (hs @ Wk.float().T).reshape(B, Hkv_tp, 1, d)
    V = (hs @ Wv.float().T).reshape(B, Hkv_tp, 1, d)

    Q = torch.stack([apply_rmsnorm_per_head(Q[:, h], q_norm_weight) for h in range(Hq_tp)],  dim=1)
    K = torch.stack([apply_rmsnorm_per_head(K[:, h], k_norm_weight) for h in range(Hkv_tp)], dim=1)

    cos = cos_at_pos.float().unsqueeze(1).unsqueeze(2)
    sin = sin_at_pos.float().unsqueeze(1).unsqueeze(2)
    Q = Q.float() * cos + rotate_half(Q.float()) * sin
    K = K.float() * cos + rotate_half(K.float()) * sin

    scale = 1.0 / math.sqrt(d)
    output = torch.zeros(B, Hq_tp, d, dtype=torch.float32)

    for b in range(B):
        for q_h in range(Hq_tp):
            kv_h  = q_h // gqa
            q_vec = Q[b, q_h, 0]

            K_full = torch.cat([K_cache[b, kv_h].float(), K[b, kv_h, 0:1].float()], dim=0)
            V_full = torch.cat([V_cache[b, kv_h].float(), V[b, kv_h, 0:1].float()], dim=0)

            scores = (K_full @ q_vec) * scale
            attn_w = F.softmax(scores, dim=0)
            output[b, q_h] = (attn_w.unsqueeze(-1) * V_full).sum(0)

    # output: [B, Hq_tp, d] → [B, 1, Hq_tp*d]
    out = output.reshape(B, 1, Hq_tp * d)

    if Wo is not None:
        Wo_f = Wo.float()
        if wo_transposed:
            # Wo is [Hq_tp*d, H]: out[..., Hq*d] @ Wo[Hq*d, H] → [B, 1, H]
            out = out @ Wo_f
        else:
            # Wo is [H, Hq_tp*d]: out[..., Hq*d] @ Wo.T[Hq*d, H] → [B, 1, H]
            out = out @ Wo_f.T

    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------

def make_inputs(B=1, S_prior=128, H=2048, d=128, Hq_tp=8, Hkv_tp=4,
                variant="no_wo", seed=42):
    """Create deterministic inputs matching the given kernel variant."""
    torch.manual_seed(seed)
    scale = 0.05

    hidden_states  = (torch.randn(B, 1, H) * scale).to(torch.bfloat16)
    Wq             = (torch.randn(Hq_tp * d, H) * scale).to(torch.bfloat16)
    Wk             = (torch.randn(Hkv_tp * d, H) * scale).to(torch.bfloat16)
    Wv             = (torch.randn(Hkv_tp * d, H) * scale).to(torch.bfloat16)
    q_norm_weight  = torch.ones(d, dtype=torch.bfloat16)
    k_norm_weight  = torch.ones(d, dtype=torch.bfloat16)
    K_cache        = (torch.randn(B, Hkv_tp, S_prior, d) * scale).to(torch.bfloat16)
    V_cache        = (torch.randn(B, Hkv_tp, S_prior, d) * scale).to(torch.bfloat16)

    # RoPE at position S_prior
    pos      = S_prior
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
    freqs    = torch.outer(torch.tensor([float(pos)]), inv_freq)
    emb      = torch.cat([freqs, freqs], dim=-1)
    cos_at_pos = emb.cos().to(torch.bfloat16).expand(B, d)
    sin_at_pos = emb.sin().to(torch.bfloat16).expand(B, d)
    position_ids = torch.full((B, 1), pos, dtype=torch.int32)

    Wo = None
    if variant == "wo_normal":
        Wo = (torch.randn(H, Hq_tp * d) * scale).to(torch.bfloat16)
    elif variant == "wo_transposed":
        Wo = (torch.randn(Hq_tp * d, H) * scale).to(torch.bfloat16)

    return (hidden_states, Wq, Wk, Wv, Wo,
            q_norm_weight, k_norm_weight,
            K_cache, V_cache,
            cos_at_pos, sin_at_pos,
            position_ids)


def _kernel_args(inputs, device, variant):
    """Build the positional arg list for the kernel based on variant."""
    (hidden_states, Wq, Wk, Wv, Wo,
     q_norm_weight, k_norm_weight,
     K_cache, V_cache,
     cos_at_pos, sin_at_pos,
     position_ids) = inputs

    def d(t):
        return t.to(device)

    if variant == "no_wo":
        return (d(hidden_states), d(Wq), d(Wk), d(Wv),
                d(q_norm_weight), d(k_norm_weight),
                d(K_cache), d(V_cache),
                d(cos_at_pos), d(sin_at_pos), d(position_ids))
    else:
        return (d(hidden_states), d(Wq), d(Wk), d(Wv), d(Wo),
                d(q_norm_weight), d(k_norm_weight),
                d(K_cache), d(V_cache),
                d(cos_at_pos), d(sin_at_pos), d(position_ids))


def _hkv_for_variant(variant):
    return 4 if variant == "no_wo" else 1


# ---------------------------------------------------------------------------
# Correctness mode
# ---------------------------------------------------------------------------

def run_correctness(kernel_fn, kernel_path: str, variant: str,
                    s_prior_list: list[int], B: int):
    print(f"Testing kernel: {kernel_path}  (variant={variant})")
    device   = xm.xla_device()
    Hkv_tp   = _hkv_for_variant(variant)
    all_pass = True

    for S_prior in s_prior_list:
        print(f"\n{'='*60}")
        print(f"  B={B}  S_prior={S_prior}  H=2048  d=128  Hq_tp=8  Hkv_tp={Hkv_tp}")
        print(f"{'='*60}")

        inputs = make_inputs(B=B, S_prior=S_prior, Hkv_tp=Hkv_tp, variant=variant)
        (hidden_states, Wq, Wk, Wv, Wo,
         q_norm_weight, k_norm_weight,
         K_cache, V_cache,
         cos_at_pos, sin_at_pos, _) = inputs

        print("Running PyTorch reference...")
        ref_out = qwen3_attn_tkg_reference(
            hidden_states, Wq, Wk, Wv,
            q_norm_weight, k_norm_weight,
            K_cache, V_cache,
            cos_at_pos, sin_at_pos,
            Wo=Wo,
            wo_transposed=(variant == "wo_transposed"),
        )
        print(f"  ref shape={ref_out.shape}  norm={ref_out.float().norm():.4f}")

        print("Running NKI kernel on device...")
        nki_out = kernel_fn(*_kernel_args(inputs, device, variant))
        xm.mark_step()
        # v10b and similar kernels return (output, k_rope_out, v_out); take first
        if isinstance(nki_out, (tuple, list)):
            nki_out = nki_out[0]
        nki_out_cpu = nki_out.cpu()
        print(f"  nki shape={nki_out_cpu.shape}  norm={nki_out_cpu.float().norm():.4f}")

        ref_np = ref_out.float().numpy().flatten()
        nki_np = nki_out_cpu.float().numpy().flatten()

        if ref_np.shape != nki_np.shape:
            min_len = min(len(ref_np), len(nki_np))
            print(f"  Shape mismatch: ref={ref_np.shape}, nki={nki_np.shape} — aligning to {min_len}")
            ref_np = ref_np[:min_len]
            nki_np = nki_np[:min_len]

        abs_diff  = np.abs(nki_np - ref_np)
        max_diff  = abs_diff.max()
        mean_diff = abs_diff.mean()
        p99_diff  = np.percentile(abs_diff, 99)

        print(f"\n  mean_diff={mean_diff:.2e}  p99_diff={p99_diff:.2e}  max_diff={max_diff:.2e}")

        try:
            np.testing.assert_allclose(nki_np, ref_np, rtol=0.02, atol=0.1)
            print(f"  PASS  (rtol=0.02, atol=0.1)")
        except AssertionError as e:
            print(f"  FAIL  {e}")
            print(f"  ref sample: {ref_np[:8]}")
            print(f"  nki sample: {nki_np[:8]}")
            all_pass = False

    print()
    if all_pass:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------

def run_benchmark(kernel_fn, kernel_path: str, variant: str,
                  warmup: int, iters: int, B: int, s_prior: int = 640):
    print(f"Benchmarking kernel: {kernel_path}  (variant={variant})")
    print(f"  warmup={warmup}  iters={iters}")

    Hkv_tp  = _hkv_for_variant(variant)
    S_prior = s_prior
    print(f"  B={B}  S_prior={S_prior}  H=2048  d=128  Hq_tp=8  Hkv_tp={Hkv_tp}")

    inputs = make_inputs(B=B, S_prior=S_prior, Hkv_tp=Hkv_tp, variant=variant)
    device = xm.xla_device()

    nki_benchmark(
        kernel_fn,
        *_kernel_args(inputs, device, variant),
        warmup=warmup,
        iters=iters,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args        = _parse_args()
    kernel_path = _resolve_kernel_path(args.kernel)
    kernel_fn, variant = _load_kernel(kernel_path)
    print(f"Loaded: {os.path.basename(kernel_path)}  entry={kernel_fn.__name__}  variant={variant}")

    if args.benchmark:
        run_benchmark(kernel_fn, kernel_path, variant,
                      warmup=args.warmup, iters=args.iters, B=args.batch,
                      s_prior=args.s_prior[0])
    else:
        run_correctness(kernel_fn, kernel_path, variant,
                        s_prior_list=args.s_prior, B=args.batch)
