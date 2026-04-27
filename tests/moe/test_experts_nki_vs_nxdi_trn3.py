"""
test_experts_nki_vs_nxdi_trn3.py

Verify the standalone experts_hbm NKI kernel against NxDI's selective-loading
ExpertMLPsV2 path, with router and RMSNorm removed from the comparison.

Inputs:
  hidden_states: [T, H=2048] bf16, already RMSNorm-normalized
  top8_idx:      [T, K=8] int32, precomputed chosen expert indices
  top8_vals:     [T, K=8] bf16, gathered softmax affinities at top8_idx

Reference:
  ExpertMLPsV2.forward_selective_loading(hidden_states, full_affinities, top8_idx)

Kernel:
  kernels.moe_fused_tkg.experts_nki.experts_hbm(hidden_states, top8_idx, top8_vals,
                                                gate_up_w, down_w)

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python tests/moe/test_experts_nki_vs_nxdi_trn3.py
"""

import argparse
import contextlib
import os
import sys
import tempfile

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import nki
import numpy as np
import torch
import torch.nn as nn
import torch_neuronx

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.testing import build_module

from kernels.moe_fused_tkg.experts_nki import experts_hbm
from qwen import Qwen3MoeInferenceConfig


MODEL_PATH = "/home/ubuntu/models/Qwen3-30B-A3B"
LNC = 2
H = 2048
E = 128
K = 8
I_TP = 192

COMPILER_ARGS = (
    "--enable-saturate-infinity --enable-mixed-precision-accumulation "
    "--model-type transformer -O1 "
    "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
)


def make_config() -> Qwen3MoeInferenceConfig:
    neuron_config = MoENeuronConfig(
        tp_degree=1,
        moe_tp_degree=1,
        batch_size=1,
        seq_len=1,
        flash_decoding_enabled=False,
        logical_nc_config=LNC,
        enable_bucketing=True,
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
    # Match the production fused kernel's per-TP expert shard layout:
    # gate_up [E, H, 2*192], down [E, 192, H].
    cfg.moe_intermediate_size = I_TP
    cfg.intermediate_size = I_TP
    return cfg


def _capture_weights(module, store: dict) -> None:
    if store is None or store or module.mlp.expert_mlps.mlp_op.gate_up_proj.weight.device.type == "meta":
        return
    with torch.no_grad():
        store["gate_up"] = module.mlp.expert_mlps.mlp_op.gate_up_proj.weight.bfloat16().cpu()
        store["down"] = module.mlp.expert_mlps.mlp_op.down_proj.weight.bfloat16().cpu()


class RefExpertsModule(nn.Module):
    def __init__(self, config: Qwen3MoeInferenceConfig, seed: int = 42, weight_scale: float = 1.0, _weight_store: dict = None):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.mlp = initialize_moe_module(config, init_tkg_module=False).to(dtype)
        if weight_scale != 1.0:
            with torch.no_grad():
                for p in self.mlp.parameters():
                    if p.is_floating_point():
                        p.mul_(weight_scale)
        self.eval()
        _capture_weights(self, _weight_store)

    def forward(self, hidden_states: torch.Tensor, top8_idx: torch.Tensor, top8_vals: torch.Tensor) -> torch.Tensor:
        expert_index = top8_idx.to(torch.long)
        full_affinities = torch.zeros(
            hidden_states.shape[0], E, dtype=top8_vals.dtype, device=hidden_states.device
        )
        full_affinities = full_affinities.scatter(1, expert_index, top8_vals)
        return self.mlp.expert_mlps.forward_selective_loading(hidden_states, full_affinities, expert_index)


class KernelExpertsModule(nn.Module):
    def __init__(self, config: Qwen3MoeInferenceConfig, seed: int = 42, weight_scale: float = 1.0, _weight_store: dict = None):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.mlp = initialize_moe_module(config, init_tkg_module=False).to(dtype)
        if weight_scale != 1.0:
            with torch.no_grad():
                for p in self.mlp.parameters():
                    if p.is_floating_point():
                        p.mul_(weight_scale)
        self._experts_jit = nki.jit(experts_hbm)
        self.eval()
        _capture_weights(self, _weight_store)

    def forward(self, hidden_states: torch.Tensor, top8_idx: torch.Tensor, top8_vals: torch.Tensor) -> torch.Tensor:
        gate_up_w = self.mlp.expert_mlps.mlp_op.gate_up_proj.weight
        down_w = self.mlp.expert_mlps.mlp_op.down_proj.weight
        return self._experts_jit[LNC](hidden_states, top8_idx, top8_vals, gate_up_w, down_w)


HIDDEN_SCALES = [0.1, 0.5, 1.0, 3.0, 5.0]
HIDDEN_DISTS = ["normal", "student-t5", "laplace"]
AFFINITY_MODES = ["uniform", "dirichlet", "peaky", "tiny"]


def _sample_hidden(rng: np.random.Generator, tokens: int, scale: float, dist: str) -> torch.Tensor:
    if dist == "normal":
        x = rng.standard_normal((tokens, H))
    elif dist == "student-t5":
        df = 5.0
        x = rng.standard_t(df, size=(tokens, H))
        x = x * np.sqrt((df - 2.0) / df)
    elif dist == "laplace":
        x = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=(tokens, H))
    else:
        raise ValueError(f"unknown hidden distribution: {dist}")
    return torch.from_numpy((x * scale).astype(np.float32)).bfloat16()


def _sample_topk(rng: np.random.Generator, tokens: int, mode: str):
    top8_idx = np.empty((tokens, K), dtype=np.int32)
    top8_vals = np.empty((tokens, K), dtype=np.float32)
    for t in range(tokens):
        top8_idx[t] = rng.choice(E, size=K, replace=False).astype(np.int32)
        if mode == "uniform":
            vals = rng.uniform(0.05, 1.0, size=K)
        elif mode == "dirichlet":
            vals = rng.dirichlet(np.ones(K) * 0.7)
        elif mode == "peaky":
            vals = rng.uniform(1.0e-3, 0.05, size=K)
            vals[rng.integers(0, K)] = rng.uniform(0.5, 1.0)
        elif mode == "tiny":
            vals = rng.uniform(1.0e-14, 5.0e-12, size=K)
        else:
            raise ValueError(f"unknown affinity mode: {mode}")
        top8_vals[t] = vals.astype(np.float32)
    return torch.from_numpy(top8_idx), torch.from_numpy(top8_vals).bfloat16()


def make_inputs_and_expected(kern_traced, tokens: int, n_samples: int):
    combos = [(s, d, a) for s in HIDDEN_SCALES for d in HIDDEN_DISTS for a in AFFINITY_MODES]
    inputs_list = []
    expected_list = []
    combos_list = []
    for i in range(n_samples):
        scale, dist, affinity_mode = combos[i % len(combos)]
        rng = np.random.default_rng(200 + i)
        hidden = _sample_hidden(rng, tokens, scale, dist)
        top8_idx, top8_vals = _sample_topk(rng, tokens, affinity_mode)
        expected = kern_traced(hidden, top8_idx, top8_vals)
        inputs_list.append((hidden, top8_idx, top8_vals))
        expected_list.append(expected)
        combos_list.append((scale, dist, affinity_mode))
        if i % 64 == 0 or i == n_samples - 1:
            print(
                f"  collected sample {i:4d} / {n_samples}  "
                f"scale={scale:<4}  dist={dist:<10}  affinities={affinity_mode}",
                flush=True,
            )
    return inputs_list, expected_list, combos_list


def _check_tolerance(expected: torch.Tensor, actual: torch.Tensor, atol: float, rtol: float) -> bool:
    try:
        torch_neuronx.testing.assert_close(expected, actual, atol=atol, rtol=rtol, check_device=False)
        return True
    except AssertionError:
        return False


def _sample_stats(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    e = expected.float().flatten()
    a = actual.float().flatten()
    diff = e - a
    abs_diff = diff.abs()
    max_abs_ref = a.abs().max().item()
    return {
        "max_abs_err": abs_diff.max().item(),
        "max_rel_err": (abs_diff / a.abs().clamp_min(1.0e-30)).max().item(),
        "mean_signed": diff.mean().item(),
        "std_signed": diff.std().item(),
        "ulp_atol": max_abs_ref * (2.0 ** -7) if max_abs_ref > 0 else 0.0,
    }


def _fmt_pct(n: int, total: int) -> str:
    return f"{n}/{total} ({100.0 * n / total:5.2f}%)" if total else "0/0 (nan%)"


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def _bf16_bits(t: torch.Tensor) -> str:
    return f"0x{int(t.detach().cpu().view(torch.int16).item()) & 0xffff:04x}"


def _print_failure_details(
    sample_idx: int,
    combo: tuple,
    hidden: torch.Tensor,
    top8_idx: torch.Tensor,
    top8_vals: torch.Tensor,
    kernel_out: torch.Tensor,
    nxdi_out: torch.Tensor,
    max_coords: int = 8,
) -> None:
    kernel_cpu = kernel_out.detach().cpu()
    nxdi_cpu = nxdi_out.detach().cpu()
    abs_diff = (kernel_cpu.float() - nxdi_cpu.float()).abs()
    failing = torch.nonzero(abs_diff > 1.0e-5, as_tuple=False)
    flat_order = torch.argsort(abs_diff.flatten(), descending=True)

    print(
        f"\n  STRICT FAILURE sample={sample_idx} combo={combo} "
        f"num_coords={failing.shape[0]} max_abs={abs_diff.max().item():.6g}",
        flush=True,
    )
    print(f"    top8_idx={top8_idx.detach().cpu().tolist()}", flush=True)
    print(f"    top8_vals={top8_vals.detach().cpu().float().tolist()}", flush=True)

    shown = 0
    for flat_idx in flat_order.tolist():
        if shown >= max_coords:
            break
        t = flat_idx // kernel_cpu.shape[1]
        h = flat_idx % kernel_cpu.shape[1]
        err = abs_diff[t, h].item()
        if err <= 1.0e-5:
            break
        k_val = kernel_cpu[t, h]
        n_val = nxdi_cpu[t, h]
        print(
            "    "
            f"coord=(t={t}, h={h}) "
            f"kernel={float(k_val): .8g} {_bf16_bits(k_val)} "
            f"nxdi={float(n_val): .8g} {_bf16_bits(n_val)} "
            f"diff={float((k_val.float() - n_val.float()).item()):+.8g} "
            f"hidden={float(hidden.detach().cpu()[t, h]): .8g} {_bf16_bits(hidden.detach().cpu()[t, h])}",
            flush=True,
        )
        shown += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--weight-scales", type=lambda s: [float(x) for x in s.split(",")], default=[2.0])
    parser.add_argument("--loose-fail-threshold", type=float, default=0.01)
    parser.add_argument("--strict-fail-threshold", type=float, default=None)
    parser.add_argument("--dump-failures", type=int, default=0)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--ref-compiler-args", type=str, default=None)
    parser.add_argument("--kernel-compiler-args", type=str, default=None)
    args = parser.parse_args()

    cfg = make_config()
    print(
        f"Config: H={cfg.hidden_size} E={cfg.num_experts} I={cfg.intermediate_size} "
        f"K={cfg.num_experts_per_tok} T={args.tokens} LNC={LNC}",
        flush=True,
    )

    example_inputs = [
        (
            torch.zeros(args.tokens, H, dtype=torch.bfloat16),
            torch.zeros(args.tokens, K, dtype=torch.int32),
            torch.ones(args.tokens, K, dtype=torch.bfloat16),
        )
    ]

    aggregate_stats = []
    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
        workdir_cm = contextlib.nullcontext(args.workdir)
    elif args.keep_workdir:
        workdir_cm = contextlib.nullcontext(tempfile.mkdtemp(prefix="experts_vs_nxdi_"))
    else:
        workdir_cm = tempfile.TemporaryDirectory(prefix="experts_vs_nxdi_")

    with workdir_cm as workdir:
        print(f"Compiler workdir: {workdir}", flush=True)
        for ws in args.weight_scales:
            ws_tag = f"ws{ws:g}"
            print(f"\n{'=' * 65}\n  weight_scale={ws:g}\n{'=' * 65}", flush=True)

            print(f"\nCompiling RefExpertsModule (weight_scale={ws:g})...", flush=True)
            ref_weight_store = {}
            ref_traced = build_module(
                module_cls=RefExpertsModule,
                example_inputs=example_inputs,
                module_init_kwargs={"config": cfg, "seed": args.seed, "weight_scale": ws, "_weight_store": ref_weight_store},
                tp_degree=cfg.neuron_config.tp_degree,
                logical_nc_config=LNC,
                compiler_args=args.ref_compiler_args or COMPILER_ARGS,
                compiler_workdir=os.path.join(workdir, f"ref_workdir_{ws_tag}"),
            )
            print("RefExpertsModule compile PASS\n", flush=True)

            print(f"Compiling KernelExpertsModule (weight_scale={ws:g})...", flush=True)
            kern_weight_store = {}
            kern_traced = build_module(
                module_cls=KernelExpertsModule,
                example_inputs=example_inputs,
                module_init_kwargs={"config": cfg, "seed": args.seed, "weight_scale": ws, "_weight_store": kern_weight_store},
                tp_degree=cfg.neuron_config.tp_degree,
                logical_nc_config=LNC,
                compiler_args=args.kernel_compiler_args or COMPILER_ARGS,
                compiler_workdir=os.path.join(workdir, f"kern_workdir_{ws_tag}"),
            )
            print("KernelExpertsModule compile PASS\n", flush=True)

            print("Comparing captured expert weights...", flush=True)
            for name in ("gate_up", "down"):
                max_abs = (ref_weight_store[name].float() - kern_weight_store[name].float()).abs().max().item()
                print(f"  {name:<8} max_abs_diff={max_abs:.6g}", flush=True)

            print("\nCollecting kernel outputs...", flush=True)
            inputs_list, expected_list, combos_list = make_inputs_and_expected(kern_traced, args.tokens, args.n_samples)

            print("\nValidating sample by sample (ref_traced vs kernel expected)...", flush=True)
            scale_stats = []
            dumped_failures = 0
            for i, ((hidden, top8_idx, top8_vals), expected) in enumerate(zip(inputs_list, expected_list)):
                actual = ref_traced(hidden, top8_idx, top8_vals)
                stats = _sample_stats(expected, actual)
                stats["pass_strict"] = _check_tolerance(expected, actual, atol=1.0e-5, rtol=0.0)
                stats["pass_loose"] = _check_tolerance(expected, actual, atol=1.0e-5, rtol=1.0e-2)
                stats["pass_ulp"] = _check_tolerance(expected, actual, atol=max(stats["ulp_atol"], 1.0e-5), rtol=0.0)
                stats["combo"] = combos_list[i]
                stats["weight_scale"] = ws
                scale_stats.append(stats)
                if not stats["pass_strict"] and dumped_failures < args.dump_failures:
                    _print_failure_details(i, combos_list[i], hidden, top8_idx, top8_vals, expected, actual)
                    dumped_failures += 1
                if (i + 1) % 256 == 0 or i == len(inputs_list) - 1:
                    print(f"    validated {i + 1:4d} / {len(inputs_list)} samples", flush=True)

            total = len(scale_stats)
            strict_fail = sum(1 for s in scale_stats if not s["pass_strict"])
            loose_fail = sum(1 for s in scale_stats if not s["pass_loose"])
            ulp_fail = sum(1 for s in scale_stats if not s["pass_ulp"])
            max_abs_errs = [s["max_abs_err"] for s in scale_stats]
            max_rel_errs = [s["max_rel_err"] for s in scale_stats]
            mean_signed = [s["mean_signed"] for s in scale_stats]
            print(f"\n--- weight_scale={ws:g}: summary over {total} samples ---", flush=True)
            print(f"  strict   (atol=1e-5, rtol=0       ): {_fmt_pct(strict_fail, total)}", flush=True)
            print(f"  loose    (atol=1e-5, rtol=1e-2    ): {_fmt_pct(loose_fail, total)}", flush=True)
            print(f"  bf16-ulp (atol=max|out|*2^-7      ): {_fmt_pct(ulp_fail, total)}", flush=True)
            print(
                f"  Max |err|: p50={_percentile(max_abs_errs, 50):.6g}  "
                f"p95={_percentile(max_abs_errs, 95):.6g}  max={max(max_abs_errs):.6g}",
                flush=True,
            )
            print(
                f"  Max rel err: p50={_percentile(max_rel_errs, 50):.6g}  "
                f"p95={_percentile(max_rel_errs, 95):.6g}  max={max(max_rel_errs):.6g}",
                flush=True,
            )
            bias_ratio = abs(np.mean(mean_signed)) / (np.std(mean_signed) + 1.0e-30)
            print(
                f"  Bias: mean_signed={np.mean(mean_signed):+.6g}  "
                f"std={np.std(mean_signed):.6g}  |mean|/std={bias_ratio:.3f}",
                flush=True,
            )
            aggregate_stats.extend(scale_stats)

    total = len(aggregate_stats)
    strict_fail = sum(1 for s in aggregate_stats if not s["pass_strict"])
    strict_fail_rate = strict_fail / total if total > 0 else 0.0
    loose_fail = sum(1 for s in aggregate_stats if not s["pass_loose"])
    loose_fail_rate = loose_fail / total if total > 0 else 0.0
    print(f"\n  Strict fail rate: {100.0 * strict_fail_rate:.3f}%", flush=True)
    if args.strict_fail_threshold is not None and strict_fail_rate > args.strict_fail_threshold:
        raise AssertionError(
            f"strict fail rate {100.0 * strict_fail_rate:.3f}% exceeds gate "
            f"{100.0 * args.strict_fail_threshold:.2f}%"
        )
    print(f"\n  Loose-tolerance fail rate: {100.0 * loose_fail_rate:.3f}%  "
          f"(gate: {100.0 * args.loose_fail_threshold:.2f}%)", flush=True)
    if loose_fail_rate > args.loose_fail_threshold:
        raise AssertionError(
            f"loose-tol fail rate {100.0 * loose_fail_rate:.3f}% exceeds gate "
            f"{100.0 * args.loose_fail_threshold:.2f}%"
        )
    print("  OVERALL: PASS (loose-tol fail rate within gate)", flush=True)


if __name__ == "__main__":
    main()
