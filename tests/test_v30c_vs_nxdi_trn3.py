"""
test_v30c_vs_nxdi_trn3.py

Verify qwen3_moe_fused_tkg_sbuf_io (v30c) matches NeuronQwen3MoeDecoderLayer's MoE —
the exact class and calling convention used by qwen.py during token generation.

Reference: NxDI MoE (init_tkg_module=False, pure PyTorch) with post_attention_layernorm
  applied explicitly before the call — matching qwen.py's moe_fused_nki_kernel_enabled=False
  code path (lines 427-429):
      hidden_states = self.post_attention_layernorm(hidden_states)
      hidden_states = self.mlp(hidden_states, padding_mask)[0]

Kernel: v30c wrapper where RMSNorm is fused inside the kernel, matching the
  moe_fused_nki_kernel_enabled=True path where norm is passed to initialize_moe_module
  and not applied by the decoder layer.

Both modules are seeded identically (manual_seed(42)) so weights are identical.

Dims: H=2048, E=128, I=192 (tp_degree=1, moe_tp_degree=1), K=8, B=1, S=640, LNC=2.

Tolerance (bf16-native, no fp32 promotion): atol=1e-5, rtol=1e-2.

Run:
    python tests/test_v30c_vs_nxdi_trn3.py
"""

import os
import sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOE_VERSIONS = os.path.join(_HERE, "..", "kernels", "moe_fused_tkg", "versions")
sys.path.insert(0, "/home/ubuntu/nki-moe")
sys.path.insert(0, _MOE_VERSIONS)
sys.path.insert(0, _HERE)

import argparse
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch_neuronx
import nki

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

from kernels.moe_fused_tkg.moe_fused_nki import moe_fused_tkg
from qwen import Qwen3MoeInferenceConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/ubuntu/models/Qwen3-30B-A3B"
B   = 1
S   = 640
LNC = 2
_H = 2048    # hidden dim
# The v30c kernel assumes a single TP=4 rank's MoE shard (I_TP = 768 / 4 = 192).
# NxDI with tp_degree=1 would hand it the un-sharded I=768 weights, which makes
# the kernel's `.ap()` patterns (strided for the 2*I=384 row width) read the
# wrong bytes out of a 2*I=1536-wide row. Override the config so both the ref
# and the kernel see a model whose per-expert intermediate is already one
# TP-shard's worth of work — i.e. simulate rank 0 directly.
_I_TP = 192

# Verbatim from NeuronQwen3MoeForCausalLM.get_compiler_args(), default path, no EP.
COMPILER_ARGS = (
    "--enable-saturate-infinity --enable-mixed-precision-accumulation "
    "--model-type transformer -O1 "
    "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def make_config() -> Qwen3MoeInferenceConfig:
    neuron_config = MoENeuronConfig(
        tp_degree=1,
        moe_tp_degree=1,
        batch_size=1,
        seq_len=S,
        flash_decoding_enabled=False,
        logical_nc_config=LNC,
        enable_bucketing=True,
        # Disable NxDI's internal TKG NKI kernel so the reference is pure PyTorch MoE.
        moe_fused_nki_kernel_enabled=False,
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
        mlp_kernel_enabled=False,
        attn_tkg_nki_kernel_enabled=False,
        attn_block_tkg_nki_kernel_enabled=False,
    )
    cfg = Qwen3MoeInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )
    # Shrink the MoE intermediate dim to one TP=4 shard so NxDI initializes
    # expert weights at the layout the kernel expects ([E, H, 2*192] and
    # [E, 192, H]). Qwen3MoeInferenceConfig.__init__ sets
    # `self.intermediate_size = self.moe_intermediate_size` at construction, so
    # both attributes need the override to stay consistent.
    cfg.moe_intermediate_size = _I_TP
    cfg.intermediate_size = _I_TP
    return cfg


# ---------------------------------------------------------------------------
# Reference module
# Mirrors NeuronQwen3MoeDecoderLayer.forward() with moe_fused_nki_kernel_enabled=False:
#   hidden_states = self.post_attention_layernorm(hidden_states)
#   hidden_states = self.mlp(hidden_states, padding_mask)[0]
# ---------------------------------------------------------------------------
class RefMoEModule(nn.Module):
    def __init__(self, config: Qwen3MoeInferenceConfig, seed: int = 42, weight_scale: float = 1.0, _weight_store: dict = None):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.post_attention_layernorm = CustomRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        ).to(dtype)
        self.mlp = initialize_moe_module(config, init_tkg_module=False).to(dtype)
        if weight_scale != 1.0:
            with torch.no_grad():
                for p in self.parameters():
                    if p.is_floating_point():
                        p.mul_(weight_scale)
        # ExpertMLPs.forward dispatches on self.training: training=True -> forward_all_experts,
        # training=False -> forward_selective_loading (the real TKG path the kernel models).
        # build_module traces modules as-is, and nn.Module defaults to training=True, so without
        # this call the reference would compute all 128 experts (zero-masked, re-normalized),
        # which rounds differently than computing only the top-8. Force eval mode to match
        # the kernel's selective-loading semantics.
        self.eval()
        if (
            _weight_store is not None
            and not _weight_store
            and self.post_attention_layernorm.weight.device.type != "meta"
        ):
            with torch.no_grad():
                _weight_store["gamma"]    = self.post_attention_layernorm.weight.unsqueeze(0).bfloat16().cpu()
                _weight_store["router_w"] = self.mlp.router.linear_router.weight.T.contiguous().bfloat16().cpu()
                _weight_store["gate_up"]  = self.mlp.expert_mlps.mlp_op.gate_up_proj.weight.bfloat16().cpu()
                _weight_store["down"]     = self.mlp.expert_mlps.mlp_op.down_proj.weight.bfloat16().cpu()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, 1, H] -> norm -> MoE -> [B, 1, H]
        normed = self.post_attention_layernorm(hidden_states)
        return self.mlp(normed, None)[0]


# ---------------------------------------------------------------------------
# Kernel module
# Same weight creation order/seed as RefMoEModule → identical weights.
# RMSNorm is fused inside v30c; hidden_states enters pre-norm, matching
# the moe_fused_nki_kernel_enabled=True path in qwen.py.
# ---------------------------------------------------------------------------
class KernelMoEModule(nn.Module):
    def __init__(self, config: Qwen3MoeInferenceConfig, seed: int = 42, weight_scale: float = 1.0, _weight_store: dict = None):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        self._dtype = dtype
        torch.manual_seed(seed)
        self.post_attention_layernorm = CustomRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        ).to(dtype)
        self.mlp = initialize_moe_module(config, init_tkg_module=False).to(dtype)
        if weight_scale != 1.0:
            with torch.no_grad():
                for p in self.post_attention_layernorm.parameters():
                    if p.is_floating_point():
                        p.mul_(weight_scale)
                for p in self.mlp.parameters():
                    if p.is_floating_point():
                        p.mul_(weight_scale)
        self._moe_jit = nki.jit(moe_fused_tkg)
        # Capture bf16 CPU copies of all kernel weight tensors on the first real
        # instantiation (not AOT-mode meta-tensor re-instantiations).
        if (
            _weight_store is not None
            and not _weight_store
            and self.post_attention_layernorm.weight.device.type != "meta"
        ):
            with torch.no_grad():
                _weight_store["gamma"]    = self.post_attention_layernorm.weight.unsqueeze(0).bfloat16().cpu()
                _weight_store["router_w"] = self.mlp.router.linear_router.weight.T.contiguous().bfloat16().cpu()
                _weight_store["gate_up"]  = self.mlp.expert_mlps.mlp_op.gate_up_proj.weight.bfloat16().cpu()
                _weight_store["down"]     = self.mlp.expert_mlps.mlp_op.down_proj.weight.bfloat16().cpu()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Weight extraction
        # post_attention_layernorm.weight: [H]  -> [1, H]
        gamma    = self.post_attention_layernorm.weight.unsqueeze(0)
        # router.linear_router is nn.Linear([H, E] stored as [E, H]); v30c wants [H, E]
        router_w  = self.mlp.router.linear_router.weight.T.contiguous().to(self._dtype)
        # expert_mlps.mlp_op.gate_up_proj.weight: [E, H, 2*I]  (matches v30c gate_up_w)
        gate_up_w = self.mlp.expert_mlps.mlp_op.gate_up_proj.weight
        # expert_mlps.mlp_op.down_proj.weight:    [E, I, H]    (matches v30c down_w)
        down_w    = self.mlp.expert_mlps.mlp_op.down_proj.weight

        # v30c expects inp [T, H]; reshape [B, 1, H] -> [B, H] (T = B = 1 for TKG)
        inp_2d = hidden_states.reshape(B, _H)
        out_2d = self._moe_jit[LNC](inp_2d, gamma, router_w, gate_up_w, down_w)  # [T, H]
        return out_2d.reshape(hidden_states.shape)  # [B, 1, H]


# ---------------------------------------------------------------------------
# Build inputs / expected_outputs for validate_accuracy
# Sweep 5 activation scales x 3 distributions so routing sees a broad mix of
# magnitude regimes, tail heaviness, and boundary-near topk decisions:
#   - 5 scales [0.1, 0.5, 1.0, 3.0, 5.0] — from easy-routing to RMSNorm/softmax saturation
#   - Normal: baseline Gaussian (kurtosis=3)
#   - Student-t df=5: finite variance, kurtosis ~9 — heavier tails push router toward ties
#   - Laplace: kurtosis=6, double-exponential — intermediate heavy tail, different shape
# ---------------------------------------------------------------------------
_SCALES = [0.1, 0.5, 1.0, 3.0, 5.0]
_DISTRIBUTIONS = ["normal", "student-t5", "laplace"]


def _sample_hidden(rng: np.random.Generator, scale: float, dist: str) -> torch.Tensor:
    if dist == "normal":
        x = rng.standard_normal((B, 1, _H))
    elif dist == "student-t5":
        df = 5.0
        x = rng.standard_t(df, size=(B, 1, _H))
        x = x * np.sqrt((df - 2.0) / df)  # rescale to unit std
    elif dist == "laplace":
        # Laplace(0, b) has std=b*sqrt(2); pick b=1/sqrt(2) for unit std.
        x = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=(B, 1, _H))
    else:
        raise ValueError(f"unknown dist: {dist}")
    return torch.from_numpy((x * scale).astype(np.float32)).bfloat16()


def make_inputs_and_expected(kern_traced, n_samples: int = 2048):
    """Collect kernel outputs across the scale/distribution sweep.

    Cycles through all 5*3=15 combos and wraps around if n_samples exceeds that,
    with a unique seed per sample (200+i) so each sample is a fresh draw.
    """
    combos = [(s, d) for s in _SCALES for d in _DISTRIBUTIONS]

    inputs_list   = []
    expected_list = []
    combos_list   = []
    for i in range(n_samples):
        scale, dist = combos[i % len(combos)]
        rng = np.random.default_rng(200 + i)
        hidden = _sample_hidden(rng, scale, dist)

        out = kern_traced(hidden)

        inputs_list.append((hidden,))
        expected_list.append(out)
        combos_list.append((scale, dist))
        if i % 64 == 0 or i == n_samples - 1:
            print(f"  collected sample {i:4d} / {n_samples}  scale={scale:<4}  dist={dist}", flush=True)

    return inputs_list, expected_list, combos_list


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def _percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _compute_sample_stats(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    """Per-sample stats on expected (kernel) vs actual (ref).

    All in fp32 for stable math. Returns scalars.
    """
    e = expected.float().flatten()
    a = actual.float().flatten()
    diff = e - a
    abs_diff = diff.abs()
    max_abs_a = a.abs().max().item()

    max_abs_err = abs_diff.max().item()
    # Relative err uses |actual| as denom (matches torch_neuronx.assert_close semantics).
    # Clamp denom away from zero to avoid 0/0 — values below max_abs_err/1e6 contribute nothing useful.
    denom = a.abs().clamp_min(1e-30)
    max_rel_err = (abs_diff / denom).max().item()
    mean_signed = diff.mean().item()
    std_signed = diff.std().item()
    # bf16 ulp floor: at the max-magnitude value, one bf16 ulp is max_abs * 2^-7.
    # Use this as the "noise floor" tolerance.
    ulp_atol = max_abs_a * (2.0 ** -7) if max_abs_a > 0 else 0.0
    return {
        "max_abs_err":  max_abs_err,
        "max_rel_err":  max_rel_err,
        "mean_signed":  mean_signed,
        "std_signed":   std_signed,
        "max_abs_ref":  max_abs_a,
        "ulp_atol":     ulp_atol,
    }


def _check_tolerance(expected: torch.Tensor, actual: torch.Tensor, atol: float, rtol: float) -> bool:
    try:
        torch_neuronx.testing.assert_close(
            expected, actual, atol=atol, rtol=rtol, check_device=False,
        )
        return True
    except AssertionError:
        return False


def _fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "0/0 (nan%)"
    return f"{n}/{total} ({100.0 * n / total:5.2f}%)"


def _dump_failure_sample(
    dump_path: str,
    hidden: torch.Tensor,
    actual: torch.Tensor,
    expected: torch.Tensor,
    ref_weight_store: dict,
    kern_weight_store: dict,
    args,
    ws: float,
    i: int,
    combo,
    sample_stats: dict,
    tag: str,
) -> None:
    torch.save(
        {
            "input":        hidden.cpu(),
            "gamma":        kern_weight_store["gamma"],
            "router_w":     kern_weight_store["router_w"],
            "gate_up":      kern_weight_store["gate_up"],
            "down":         kern_weight_store["down"],
            "ref_gamma":    ref_weight_store["gamma"],
            "ref_router_w": ref_weight_store["router_w"],
            "ref_gate_up":  ref_weight_store["gate_up"],
            "ref_down":     ref_weight_store["down"],
            "ref_out":      actual.cpu(),
            "kern_out":     expected.cpu(),
            "weight_scale": ws,
            "seed":         args.seed,
            "sample_idx":   i,
            "combo":        combo,
            "dump_tag":     tag,
            "stats": {
                "max_abs_err": float(sample_stats["max_abs_err"]),
                "max_rel_err": float(sample_stats["max_rel_err"]),
                "mean_signed": float(sample_stats["mean_signed"]),
                "std_signed":  float(sample_stats["std_signed"]),
                "ulp_atol":    float(sample_stats["ulp_atol"]),
                "pass_strict": bool(sample_stats["pass_strict"]),
                "pass_loose":  bool(sample_stats["pass_loose"]),
                "pass_ulp":    bool(sample_stats["pass_ulp"]),
            },
        },
        dump_path,
    )


def _print_scale_summary(ws: float, stats: list, combos: list) -> None:
    """Print a per-weight-scale summary table. stats is a list of per-sample dicts."""
    total = len(stats)
    if total == 0:
        print(f"  weight_scale={ws:g}: NO SAMPLES", flush=True)
        return

    # Failure counts under each tolerance band
    strict_fail = sum(1 for s in stats if not s["pass_strict"])
    loose_fail  = sum(1 for s in stats if not s["pass_loose"])
    ulp_fail    = sum(1 for s in stats if not s["pass_ulp"])

    max_abs_errs = [s["max_abs_err"] for s in stats]
    max_rel_errs = [s["max_rel_err"] for s in stats]
    mean_signed  = [s["mean_signed"] for s in stats]

    print(f"\n--- weight_scale={ws:g}: summary over {total} samples ---", flush=True)
    print(f"  Failure rate by tolerance band:", flush=True)
    print(f"    strict  (atol=1e-5, rtol=0       ): {_fmt_pct(strict_fail, total)}", flush=True)
    print(f"    loose   (atol=1e-5, rtol=1e-2    ): {_fmt_pct(loose_fail,  total)}  <-- CLAUDE.md bf16-native tol", flush=True)
    print(f"    bf16-ulp (atol=max|out|*2^-7     ): {_fmt_pct(ulp_fail,    total)}  <-- noise-floor tolerance", flush=True)
    print(f"  Max |err| distribution: mean={np.mean(max_abs_errs):.6g}  "
          f"p50={_percentile(max_abs_errs, 50):.6g}  "
          f"p95={_percentile(max_abs_errs, 95):.6g}  "
          f"max={max(max_abs_errs):.6g}", flush=True)
    print(f"  Max rel err distribution: mean={np.mean(max_rel_errs):.6g}  "
          f"p50={_percentile(max_rel_errs, 50):.6g}  "
          f"p95={_percentile(max_rel_errs, 95):.6g}  "
          f"max={max(max_rel_errs):.6g}", flush=True)
    bias_ratio = abs(np.mean(mean_signed)) / (np.std(mean_signed) + 1e-30)
    print(f"  Mean signed diff (bias check): mean={np.mean(mean_signed):+.6g}  "
          f"std={np.std(mean_signed):.6g}  "
          f"bias_ratio=|mean|/std={bias_ratio:.3f}  "
          f"({'ZERO-MEAN NOISE' if bias_ratio < 0.3 else 'POSSIBLE SYSTEMATIC BIAS'})", flush=True)

    # Break down failures by (scale, distribution) combo at the strict + loose bars
    breakdown = {}
    for st, (sc, di) in zip(stats, combos):
        key = (sc, di)
        d = breakdown.setdefault(key, {"total": 0, "strict_fail": 0, "loose_fail": 0})
        d["total"] += 1
        if not st["pass_strict"]:
            d["strict_fail"] += 1
        if not st["pass_loose"]:
            d["loose_fail"] += 1
    print(f"  Breakdown by (scale, distribution):", flush=True)
    print(f"    {'scale':<6} {'dist':<12} {'samples':>8} {'strict_fail':>14} {'loose_fail':>14}", flush=True)
    for (sc, di) in sorted(breakdown.keys()):
        d = breakdown[(sc, di)]
        print(f"    {sc:<6} {di:<12} {d['total']:>8} "
              f"{_fmt_pct(d['strict_fail'], d['total']):>14} "
              f"{_fmt_pct(d['loose_fail'],  d['total']):>14}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weight-scales",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[2.0],
        metavar="W1,W2,...",
        help="Comma-separated weight scale factors",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2048,
        help="Samples per weight-scale (cycled through 5 scales x 3 distributions = 15 combos)",
    )
    parser.add_argument(
        "--dump-dir",
        default=".",
        metavar="DIR",
        help="Directory to write failure dump files (default: current dir)",
    )
    parser.add_argument(
        "--loose-fail-threshold",
        type=float,
        default=0.01,
        help="Exit non-zero if loose-tolerance failure rate exceeds this fraction (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--dump-first-strict-per-ws",
        action="store_true",
        help="Dump the first strict failure for each weight_scale instead of only the first global strict failure.",
    )
    parser.add_argument(
        "--dump-worst-loose-per-ws",
        action="store_true",
        help="Dump the worst loose-tolerance failure for each weight_scale.",
    )
    parser.add_argument(
        "--dump-top-loose",
        type=int,
        default=0,
        metavar="N",
        help="After the sweep, dump the top-N loose failures by max_abs_err across all weight_scales.",
    )
    args = parser.parse_args()

    cfg = make_config()
    print(
        f"Config: H={cfg.hidden_size} E={cfg.num_experts} I={cfg.intermediate_size} "
        f"K={cfg.num_experts_per_tok} B={B} S={S} TP={cfg.neuron_config.tp_degree} "
        f"moe_tp={cfg.neuron_config.moe_tp_degree} LNC={LNC}",
        flush=True,
    )
    print(f"Weight scales: {args.weight_scales}  n_samples/scale: {args.n_samples}  "
          f"(combos: {len(_SCALES)} scales x {len(_DISTRIBUTIONS)} distributions = {len(_SCALES) * len(_DISTRIBUTIONS)})",
          flush=True)

    example_inputs = [(torch.zeros(B, 1, _H, dtype=torch.bfloat16),)]

    # Aggregate stats across all weight scales. Optional representative dumps are
    # captured per-weight-scale and/or as top-N global loose failures.
    aggregate_stats: list = []
    dump_records: list = []
    global_first_strict_dumped = False

    with tempfile.TemporaryDirectory(prefix="v30c_vs_nxdi_") as workdir:
        for ws in args.weight_scales:
            ws_tag = f"ws{ws:g}"
            print(f"\n{'=' * 65}", flush=True)
            print(f"  weight_scale={ws:g}", flush=True)
            print(f"{'=' * 65}", flush=True)

            print(f"\nCompiling RefMoEModule (weight_scale={ws:g})...", flush=True)
            ref_weight_store: dict = {}
            ref_traced = build_module(
                module_cls=RefMoEModule,
                example_inputs=example_inputs,
                module_init_kwargs={"config": cfg, "seed": args.seed, "weight_scale": ws, "_weight_store": ref_weight_store},
                tp_degree=cfg.neuron_config.tp_degree,
                logical_nc_config=LNC,
                compiler_args=COMPILER_ARGS,
                compiler_workdir=os.path.join(workdir, f"ref_workdir_{ws_tag}"),
            )
            print("RefMoEModule compile PASS\n", flush=True)

            print(f"Compiling KernelMoEModule (weight_scale={ws:g})...", flush=True)
            kern_weight_store: dict = {}
            kern_traced = build_module(
                module_cls=KernelMoEModule,
                example_inputs=example_inputs,
                module_init_kwargs={"config": cfg, "seed": args.seed, "weight_scale": ws, "_weight_store": kern_weight_store},
                tp_degree=cfg.neuron_config.tp_degree,
                logical_nc_config=LNC,
                compiler_args=COMPILER_ARGS,
                compiler_workdir=os.path.join(workdir, f"kern_workdir_{ws_tag}"),
            )
            print("KernelMoEModule compile PASS\n", flush=True)

            print("Comparing captured ref/kernel weights...", flush=True)
            for name in ("gamma", "router_w", "gate_up", "down"):
                ref_w = ref_weight_store[name].float()
                kern_w = kern_weight_store[name].float()
                max_abs = (ref_w - kern_w).abs().max().item()
                print(f"  {name:<8} max_abs_diff={max_abs:.6g}", flush=True)
            print("", flush=True)

            print("Collecting kernel outputs...", flush=True)
            inputs_list, expected_list, combos_list = make_inputs_and_expected(
                kern_traced, n_samples=args.n_samples,
            )

            print("\nValidating sample by sample (ref_traced vs kernel expected), "
                  "collecting stats across all samples...", flush=True)
            scale_stats: list = []
            scale_first_strict_dumped = False
            scale_worst_loose = None
            for i, ((hidden,), expected) in enumerate(zip(inputs_list, expected_list)):
                actual = ref_traced(hidden)

                sample_stats = _compute_sample_stats(expected, actual)
                # Tolerance checks — each independently so we can count failures in each band.
                sample_stats["pass_strict"] = _check_tolerance(expected, actual, atol=1e-5, rtol=0.0)
                sample_stats["pass_loose"]  = _check_tolerance(expected, actual, atol=1e-5, rtol=1e-2)
                # bf16-ulp tolerance: sample-specific atol = max|out| * 2^-7.
                sample_stats["pass_ulp"] = _check_tolerance(
                    expected, actual,
                    atol=max(sample_stats["ulp_atol"], 1e-5),
                    rtol=0.0,
                )
                sample_stats["weight_scale"] = ws
                sample_stats["sample_idx"]   = i
                sample_stats["combo"]        = combos_list[i]
                scale_stats.append(sample_stats)

                if not sample_stats["pass_loose"]:
                    record = {
                        "weight_scale": ws,
                        "ws_tag": ws_tag,
                        "sample_idx": i,
                        "combo": combos_list[i],
                        "hidden": hidden.cpu(),
                        "actual": actual.cpu(),
                        "expected": expected.cpu(),
                        "sample_stats": dict(sample_stats),
                        "ref_weight_store": ref_weight_store,
                        "kern_weight_store": kern_weight_store,
                    }
                    dump_records.append(record)
                    if (
                        scale_worst_loose is None
                        or sample_stats["max_abs_err"] > scale_worst_loose["sample_stats"]["max_abs_err"]
                    ):
                        scale_worst_loose = record

                dump_first_strict = (
                    (args.dump_first_strict_per_ws and not scale_first_strict_dumped)
                    or ((not args.dump_first_strict_per_ws) and (not global_first_strict_dumped))
                )
                if dump_first_strict and not sample_stats["pass_strict"]:
                    tag = "first_strict"
                    dump_path = os.path.join(args.dump_dir, f"{tag}_{ws_tag}_s{i}.pt")
                    _dump_failure_sample(
                        dump_path=dump_path,
                        hidden=hidden,
                        actual=actual,
                        expected=expected,
                        ref_weight_store=ref_weight_store,
                        kern_weight_store=kern_weight_store,
                        args=args,
                        ws=ws,
                        i=i,
                        combo=combos_list[i],
                        sample_stats=sample_stats,
                        tag=tag,
                    )
                    scale_first_strict_dumped = True
                    global_first_strict_dumped = True
                    scale, dist = combos_list[i]
                    print(f"\n  Dumped {tag}: sample {i} (scale={scale}, dist={dist})", flush=True)
                    print(f"    dump written to: {dump_path}", flush=True)
                    print(f"    Debug with simulator:", flush=True)
                    print(f"      NKI_PRECISE_FP=1 python tests/test_v30c_simulate.py --load-dump {dump_path}", flush=True)
                    print(f"  (continuing to collect full-sweep statistics...)\n", flush=True)

                if (i + 1) % 256 == 0 or i == len(inputs_list) - 1:
                    print(f"    validated {i + 1:4d} / {len(inputs_list)} samples", flush=True)

            if args.dump_worst_loose_per_ws and scale_worst_loose is not None:
                tag = "worst_loose"
                i = scale_worst_loose["sample_idx"]
                scale, dist = scale_worst_loose["combo"]
                dump_path = os.path.join(args.dump_dir, f"{tag}_{ws_tag}_s{i}.pt")
                _dump_failure_sample(
                    dump_path=dump_path,
                    hidden=scale_worst_loose["hidden"],
                    actual=scale_worst_loose["actual"],
                    expected=scale_worst_loose["expected"],
                    ref_weight_store=ref_weight_store,
                    kern_weight_store=kern_weight_store,
                    args=args,
                    ws=ws,
                    i=i,
                    combo=scale_worst_loose["combo"],
                    sample_stats=scale_worst_loose["sample_stats"],
                    tag=tag,
                )
                print(
                    f"\n  Dumped {tag} for weight_scale={ws:g}: sample {i} "
                    f"(scale={scale}, dist={dist}, max_abs_err={scale_worst_loose['sample_stats']['max_abs_err']:.6g})",
                    flush=True,
                )
                print(f"    dump written to: {dump_path}", flush=True)
                print(f"    Debug with simulator:", flush=True)
                print(f"      NKI_PRECISE_FP=1 python tests/test_v30c_simulate.py --load-dump {dump_path}", flush=True)

            _print_scale_summary(ws, scale_stats, combos_list)
            aggregate_stats.extend(scale_stats)

    # ------------------------------------------------------------------
    # Final aggregate summary
    # ------------------------------------------------------------------
    total = len(aggregate_stats)
    print(f"\n{'=' * 65}", flush=True)
    print(f"  AGGREGATE SUMMARY: {total} samples across "
          f"{len(args.weight_scales)} weight_scales x "
          f"{len(_SCALES)} scales x {len(_DISTRIBUTIONS)} distributions",
          flush=True)
    print(f"{'=' * 65}", flush=True)

    strict_fail = sum(1 for s in aggregate_stats if not s["pass_strict"])
    loose_fail  = sum(1 for s in aggregate_stats if not s["pass_loose"])
    ulp_fail    = sum(1 for s in aggregate_stats if not s["pass_ulp"])

    max_abs_errs = [s["max_abs_err"] for s in aggregate_stats]
    max_rel_errs = [s["max_rel_err"] for s in aggregate_stats]
    mean_signed  = [s["mean_signed"] for s in aggregate_stats]

    print(f"  strict   (atol=1e-5, rtol=0       ): {_fmt_pct(strict_fail, total)}", flush=True)
    print(f"  loose    (atol=1e-5, rtol=1e-2    ): {_fmt_pct(loose_fail,  total)}", flush=True)
    print(f"  bf16-ulp (atol=max|out|*2^-7      ): {_fmt_pct(ulp_fail,    total)}", flush=True)
    print(f"  Aggregate max |err|: "
          f"p50={_percentile(max_abs_errs, 50):.6g}  "
          f"p95={_percentile(max_abs_errs, 95):.6g}  "
          f"p99={_percentile(max_abs_errs, 99):.6g}  "
          f"max={max(max_abs_errs):.6g}", flush=True)
    print(f"  Aggregate max rel err: "
          f"p50={_percentile(max_rel_errs, 50):.6g}  "
          f"p95={_percentile(max_rel_errs, 95):.6g}  "
          f"p99={_percentile(max_rel_errs, 99):.6g}  "
          f"max={max(max_rel_errs):.6g}", flush=True)
    bias_ratio = abs(np.mean(mean_signed)) / (np.std(mean_signed) + 1e-30)
    print(f"  Aggregate bias: mean_signed={np.mean(mean_signed):+.6g}  "
          f"std={np.std(mean_signed):.6g}  "
          f"|mean|/std={bias_ratio:.3f}  "
          f"({'ZERO-MEAN NOISE' if bias_ratio < 0.3 else 'POSSIBLE SYSTEMATIC BIAS'})", flush=True)

    if args.dump_top_loose > 0 and dump_records:
        ranked = sorted(
            dump_records,
            key=lambda r: (
                r["sample_stats"]["max_abs_err"],
                r["sample_stats"]["max_rel_err"],
            ),
            reverse=True,
        )
        topn = ranked[: args.dump_top_loose]
        print(f"\n  Top loose failures selected for dumping: {len(topn)}", flush=True)
        for rank, record in enumerate(topn, start=1):
            ws = record["weight_scale"]
            ws_tag = record["ws_tag"]
            i = record["sample_idx"]
            scale, dist = record["combo"]
            dump_path = os.path.join(args.dump_dir, f"toploose_r{rank}_{ws_tag}_s{i}.pt")
            _dump_failure_sample(
                dump_path=dump_path,
                hidden=record["hidden"],
                actual=record["actual"],
                expected=record["expected"],
                ref_weight_store=record["ref_weight_store"] if "ref_weight_store" in record else {},
                kern_weight_store=record["kern_weight_store"],
                args=args,
                ws=ws,
                i=i,
                combo=record["combo"],
                sample_stats=record["sample_stats"],
                tag=f"toploose_r{rank}",
            )
            print(
                f"    r{rank}: weight_scale={ws:g} sample={i} "
                f"(scale={scale}, dist={dist}, max_abs_err={record['sample_stats']['max_abs_err']:.6g})",
                flush=True,
            )
            print(f"      dump written to: {dump_path}", flush=True)
            print(f"      NKI_PRECISE_FP=1 python tests/test_v30c_simulate.py --load-dump {dump_path}", flush=True)

    loose_fail_rate = loose_fail / total if total > 0 else 0.0
    print(f"\n  Loose-tolerance fail rate: {100.0 * loose_fail_rate:.3f}%  "
          f"(gate: {100.0 * args.loose_fail_threshold:.2f}%)", flush=True)
    if loose_fail_rate > args.loose_fail_threshold:
        print(f"  OVERALL: FAIL — loose-tol fail rate {100.0 * loose_fail_rate:.3f}% > "
              f"{100.0 * args.loose_fail_threshold:.2f}% gate",
              flush=True)
        raise AssertionError(
            f"loose-tol fail rate {100.0 * loose_fail_rate:.3f}% exceeds gate "
            f"{100.0 * args.loose_fail_threshold:.2f}%"
        )
    print(f"  OVERALL: PASS (loose-tol fail rate within gate)", flush=True)


if __name__ == "__main__":
    main()
