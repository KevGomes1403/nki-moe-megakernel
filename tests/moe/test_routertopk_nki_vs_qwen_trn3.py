"""
test_routertopk_nki_vs_qwen_trn3.py

Verify the standalone routertopk_hbm NKI kernel against the Qwen3 MoE
RouterTopK path used by qwen.py.

Inputs:
  hidden_states: [T, H=2048] bf16, already RMSNorm-normalized
  router_w:      [H=2048, E=128] transposed router weight for the NKI kernel

Reference:
  qwen.py -> initialize_moe_module(...).router(hidden_states)

Kernel:
  kernels.moe_fused_tkg.routertopk_nki.routertopk_hbm(hidden_states, router_w)

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python tests/moe/test_routertopk_nki_vs_qwen_trn3.py
"""

import argparse
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

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.testing import build_module

from kernels.moe_fused_tkg.routertopk_nki import routertopk_hbm
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
    # Keep expert tensors small; this test only exercises the router.
    cfg.moe_intermediate_size = I_TP
    cfg.intermediate_size = I_TP
    return cfg


def _capture_router_weight(router, store: dict, transposed_weight: torch.Tensor = None) -> None:
    if store is None or store:
        return
    weight = transposed_weight.T if transposed_weight is not None else router.linear_router.weight
    if weight.device.type == "meta":
        return
    with torch.no_grad():
        store["router"] = weight.detach().float().cpu()


class RefRouterTopKModule(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeInferenceConfig,
        seed: int = 42,
        weight_scale: float = 1.0,
        _weight_store: dict = None,
    ):
        super().__init__()
        torch.manual_seed(seed)
        dtype = config.neuron_config.torch_dtype
        self.router = initialize_moe_module(config, init_tkg_module=False).router.to(dtype)
        if weight_scale != 1.0:
            with torch.no_grad():
                self.router.linear_router.weight.mul_(weight_scale)
        self.eval()
        _capture_router_weight(self.router, _weight_store)

    def forward(self, hidden_states: torch.Tensor):
        _, expert_affinities, expert_index = self.router(hidden_states)
        return expert_index.to(torch.int32), expert_affinities


class KernelRouterTopKModule(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeInferenceConfig,
        seed: int = 42,
        weight_scale: float = 1.0,
        hoisted_router_w: bool = False,
        _weight_store: dict = None,
    ):
        super().__init__()
        torch.manual_seed(seed)
        dtype = config.neuron_config.torch_dtype
        self.router = initialize_moe_module(config, init_tkg_module=False).router.to(dtype)
        self.router_w = nn.Parameter(self.router.linear_router.weight.detach().T.clone())
        if weight_scale != 1.0:
            with torch.no_grad():
                self.router.linear_router.weight.mul_(weight_scale)
                self.router_w.mul_(weight_scale)
        self.hoisted_router_w = hoisted_router_w
        self._routertopk_jit = nki.jit(routertopk_hbm)
        self.eval()
        _capture_router_weight(self.router, _weight_store, transposed_weight=self.router_w)

    def forward(self, hidden_states: torch.Tensor):
        if self.hoisted_router_w:
            return self._routertopk_jit[LNC](hidden_states, self.router_w, True)
        return self._routertopk_jit[LNC](hidden_states, self.router_w)


HIDDEN_SCALES = [0.1, 0.5, 1.0, 3.0, 5.0]
HIDDEN_DISTS = ["normal", "student-t5", "laplace"]


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


def make_inputs_and_reference(ref_traced, tokens: int, n_samples: int):
    combos = [(s, d) for s in HIDDEN_SCALES for d in HIDDEN_DISTS]
    inputs_list = []
    expected_list = []
    combos_list = []
    for i in range(n_samples):
        scale, dist = combos[i % len(combos)]
        rng = np.random.default_rng(200 + i)
        hidden = _sample_hidden(rng, tokens, scale, dist)
        expected = ref_traced(hidden)
        inputs_list.append(hidden)
        expected_list.append(expected)
        combos_list.append((scale, dist))
        if i % 64 == 0 or i == n_samples - 1:
            print(
                f"  collected sample {i:4d} / {n_samples}  "
                f"scale={scale:<4}  dist={dist:<10}",
                flush=True,
            )
    return inputs_list, expected_list, combos_list


def _sample_stats(expected, actual) -> dict:
    expected_idx, expected_affinities = expected
    actual_idx, actual_vals = actual
    expected_idx = expected_idx.to(torch.int32).cpu()
    actual_idx = actual_idx.to(torch.int32).cpu()
    expected_affinities = expected_affinities.cpu()
    expected_vals = torch.gather(expected_affinities, 1, expected_idx.to(torch.long))
    actual_vals = actual_vals.cpu()

    diff = expected_vals.float() - actual_vals.float()
    abs_diff = diff.abs()
    idx_mismatch = expected_idx.ne(actual_idx)
    val_mismatch = expected_vals.ne(actual_vals)
    return {
        "idx_mismatch_count": int(idx_mismatch.sum().item()),
        "val_mismatch_count": int(val_mismatch.sum().item()),
        "max_abs_err": abs_diff.max().item(),
        "mean_signed": diff.mean().item(),
        "expected_idx": expected_idx,
        "actual_idx": actual_idx,
        "expected_vals": expected_vals,
        "actual_vals": actual_vals,
    }


def _fmt_pct(n: int, total: int) -> str:
    return f"{n}/{total} ({100.0 * n / total:5.2f}%)" if total else "0/0 (nan%)"


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def _parse_hoist_modes(modes: str):
    parsed = []
    labels = {
        "plain": ("plain", False),
        "false": ("plain", False),
        "0": ("plain", False),
        "hoisted": ("hoisted", True),
        "true": ("hoisted", True),
        "1": ("hoisted", True),
    }
    for raw_mode in modes.split(","):
        mode = raw_mode.strip().lower()
        if not mode:
            continue
        if mode not in labels:
            raise ValueError(f"unknown hoist mode: {raw_mode}")
        label, enabled = labels[mode]
        if (label, enabled) not in parsed:
            parsed.append((label, enabled))
    if not parsed:
        raise ValueError("at least one hoist mode is required")
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--weight-scales", type=lambda s: [float(x) for x in s.split(",")], default=[1.0])
    parser.add_argument(
        "--hoist-modes",
        default="plain,hoisted",
        help="comma-separated modes: plain,hoisted",
    )
    args = parser.parse_args()
    hoist_modes = _parse_hoist_modes(args.hoist_modes)

    cfg = make_config()
    print(
        f"Config: H={cfg.hidden_size} E={cfg.num_experts} I={cfg.intermediate_size} "
        f"K={cfg.num_experts_per_tok} T={args.tokens} LNC={LNC}",
        flush=True,
    )

    example_inputs = [(torch.zeros(args.tokens, H, dtype=torch.bfloat16),)]
    aggregate_stats = []
    first_failure = None

    with tempfile.TemporaryDirectory(prefix="routertopk_vs_qwen_") as workdir:
        for ws in args.weight_scales:
            ws_tag = f"ws{ws:g}"
            print(f"\n{'=' * 65}\n  weight_scale={ws:g}\n{'=' * 65}", flush=True)

            print(f"\nCompiling RefRouterTopKModule (weight_scale={ws:g})...", flush=True)
            ref_weight_store = {}
            ref_traced = build_module(
                module_cls=RefRouterTopKModule,
                example_inputs=example_inputs,
                module_init_kwargs={
                    "config": cfg,
                    "seed": args.seed,
                    "weight_scale": ws,
                    "_weight_store": ref_weight_store,
                },
                tp_degree=cfg.neuron_config.tp_degree,
                logical_nc_config=LNC,
                compiler_args=COMPILER_ARGS,
                compiler_workdir=os.path.join(workdir, f"ref_workdir_{ws_tag}"),
            )
            print("RefRouterTopKModule compile PASS\n", flush=True)

            print("Collecting reference outputs...", flush=True)
            inputs_list, expected_list, combos_list = make_inputs_and_reference(
                ref_traced, args.tokens, args.n_samples
            )

            for hoist_label, hoisted_router_w in hoist_modes:
                mode_tag = f"{ws_tag}_{hoist_label}"
                print(
                    f"\nCompiling KernelRouterTopKModule "
                    f"(weight_scale={ws:g}, mode={hoist_label})...",
                    flush=True,
                )
                kern_weight_store = {}
                kern_traced = build_module(
                    module_cls=KernelRouterTopKModule,
                    example_inputs=example_inputs,
                    module_init_kwargs={
                        "config": cfg,
                        "seed": args.seed,
                        "weight_scale": ws,
                        "hoisted_router_w": hoisted_router_w,
                        "_weight_store": kern_weight_store,
                    },
                    tp_degree=cfg.neuron_config.tp_degree,
                    logical_nc_config=LNC,
                    compiler_args=COMPILER_ARGS,
                    compiler_workdir=os.path.join(workdir, f"kern_workdir_{mode_tag}"),
                )
                print("KernelRouterTopKModule compile PASS\n", flush=True)

                print("Comparing captured router weights...", flush=True)
                max_abs = (
                    ref_weight_store["router"] - kern_weight_store["router"]
                ).abs().max().item()
                print(f"  router max_abs_diff={max_abs:.6g}", flush=True)

                print("\nValidating sample by sample (kernel vs qwen RouterTopK)...", flush=True)
                mode_stats = []
                for i, (hidden, expected) in enumerate(zip(inputs_list, expected_list)):
                    actual = kern_traced(hidden)
                    stats = _sample_stats(expected, actual)
                    stats["combo"] = combos_list[i]
                    stats["weight_scale"] = ws
                    stats["mode"] = hoist_label
                    mode_stats.append(stats)
                    if first_failure is None and (
                        stats["idx_mismatch_count"] or stats["val_mismatch_count"]
                    ):
                        first_failure = (i, stats)
                    if (i + 1) % 256 == 0 or i == len(inputs_list) - 1:
                        print(f"    validated {i + 1:4d} / {len(inputs_list)} samples", flush=True)

                total = len(mode_stats)
                idx_fail = sum(1 for s in mode_stats if s["idx_mismatch_count"])
                val_fail = sum(1 for s in mode_stats if s["val_mismatch_count"])
                idx_mismatches = sum(s["idx_mismatch_count"] for s in mode_stats)
                val_mismatches = sum(s["val_mismatch_count"] for s in mode_stats)
                max_abs_errs = [s["max_abs_err"] for s in mode_stats]
                mean_signed = [s["mean_signed"] for s in mode_stats]
                print(
                    f"\n--- weight_scale={ws:g}, mode={hoist_label}: summary over {total} samples ---",
                    flush=True,
                )
                print(f"  index sample failures: {_fmt_pct(idx_fail, total)}", flush=True)
                print(f"  value sample failures: {_fmt_pct(val_fail, total)}", flush=True)
                print(f"  index element mismatches: {idx_mismatches}", flush=True)
                print(f"  value element mismatches: {val_mismatches}", flush=True)
                print(
                    f"  Max |err|: p50={_percentile(max_abs_errs, 50):.6g}  "
                    f"p95={_percentile(max_abs_errs, 95):.6g}  max={max(max_abs_errs):.6g}",
                    flush=True,
                )
                print(
                    f"  Bias: mean_signed={np.mean(mean_signed):+.6g}  "
                    f"std={np.std(mean_signed):.6g}",
                    flush=True,
                )
                aggregate_stats.extend(mode_stats)

    total = len(aggregate_stats)
    idx_fail = sum(1 for s in aggregate_stats if s["idx_mismatch_count"])
    val_fail = sum(1 for s in aggregate_stats if s["val_mismatch_count"])
    if idx_fail or val_fail:
        print(f"\n  Exact index failures: {_fmt_pct(idx_fail, total)}", flush=True)
        print(f"  Exact value failures: {_fmt_pct(val_fail, total)}", flush=True)
        if first_failure is not None:
            sample_idx, stats = first_failure
            print(
                f"  First failure: sample={sample_idx} mode={stats['mode']} "
                f"weight_scale={stats['weight_scale']} combo={stats['combo']}",
                flush=True,
            )
            print(f"    expected_idx={stats['expected_idx']}", flush=True)
            print(f"    actual_idx  ={stats['actual_idx']}", flush=True)
            print(f"    expected_vals={stats['expected_vals']}", flush=True)
            print(f"    actual_vals  ={stats['actual_vals']}", flush=True)
        raise AssertionError("routertopk_hbm differs from qwen RouterTopK")

    print("\n  OVERALL: PASS (exact index and bf16 value match)", flush=True)


if __name__ == "__main__":
    main()
