"""
Hardware sweep for the SBUF decode-attention + output-projection subkernel.

The Q/K/V active tensors are generated directly so this test isolates the final
attention and O-proj stage from RMSNorm and QKV projection.

Run:
  python tests/attn/test_subkernel_attention_oproj_vs_qwen_trn3.py
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
sys.path.insert(0, _HERE)

import torch
from neuronx_distributed_inference.utils.testing import build_module

from common import (
    B,
    COMPILER_ARGS,
    D,
    DEFAULT_EDGE_CASES,
    DEFAULT_GRID_SAMPLE_COUNT,
    H,
    HQ_TP,
    S,
    format_case,
    generate_cases,
    make_sample,
    outputs_close,
)
from subkernel_common import KernelAttentionOProjStageModule, RefAttentionOProjStageModule, make_tp1_config


def _fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "0/0 (nan%)"
    return f"{n}/{total} ({100.0 * n / total:5.2f}%)"


def _stats(expected, actual) -> float:
    return (actual.detach().cpu().float() - expected.detach().cpu().float()).abs().max().item()


def _check_weights(ref_weight_store, kern_weight_store) -> None:
    diff = (ref_weight_store["o_proj_weight"].float() - kern_weight_store["o_proj_weight"].float()).abs().max().item()
    print(f"  o_proj_weight max_abs_diff={diff:.6g}", flush=True)
    if diff != 0.0:
        raise AssertionError("reference and kernel modules did not initialize identically")


def _make_qkv_inputs(hidden: torch.Tensor):
    q = hidden[:, :, 0:HQ_TP * D].reshape(B, HQ_TP, D).contiguous()
    k_start = HQ_TP * D
    v_start = k_start + D
    k = hidden[:, :, k_start:k_start + D].reshape(B, 1, D).contiguous()
    v = hidden[:, :, v_start:v_start + D].reshape(B, 1, D).contiguous()
    return q, k, v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-scales", type=lambda s: [float(x) for x in s.split(",")], default=[1.0])
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_GRID_SAMPLE_COUNT,
        help="Number of regular grid cases to run. Use 0 for one full grid pass.",
    )
    parser.add_argument("--no-edge-cases", action="store_true")
    parser.add_argument("--print-failures", action="store_true")
    parser.add_argument("--compiler-workdir", default=None)
    args = parser.parse_args()

    cfg = make_tp1_config()
    example_q = torch.zeros(B, HQ_TP, D, dtype=torch.bfloat16)
    example_k = torch.zeros(B, 1, D, dtype=torch.bfloat16)
    example_v = torch.zeros(B, 1, D, dtype=torch.bfloat16)
    example_mask = torch.zeros(B, 1, 1, S, dtype=torch.bool)
    example_pos = torch.zeros(B, 1, dtype=torch.int32)
    example_K = torch.zeros(B, 1, S, D, dtype=torch.bfloat16)
    example_V = torch.zeros(B, 1, S, D, dtype=torch.bfloat16)
    example_inputs = [(example_q, example_k, example_v, example_mask, example_pos, example_K, example_V)]

    total_strict_fail = 0
    total_loose_fail = 0
    total_samples = 0

    workdir_ctx = (
        contextlib.nullcontext(args.compiler_workdir)
        if args.compiler_workdir is not None
        else tempfile.TemporaryDirectory(prefix="subkernel_attention_oproj_")
    )
    with workdir_ctx as workdir:
        if workdir is not None:
            os.makedirs(workdir, exist_ok=True)
        for weight_scale in args.weight_scales:
            print(f"\n{'=' * 72}", flush=True)
            print(f"attention_oproj weight_scale={weight_scale:g}", flush=True)
            print(f"{'=' * 72}", flush=True)

            ref_weight_store = {}
            ref_traced = build_module(
                module_cls=RefAttentionOProjStageModule,
                example_inputs=example_inputs,
                module_init_kwargs={
                    "config": cfg,
                    "seed": args.seed,
                    "weight_scale": weight_scale,
                    "_weight_store": ref_weight_store,
                },
                tp_degree=1,
                logical_nc_config=2,
                compiler_args=COMPILER_ARGS,
                compiler_workdir=os.path.join(workdir, f"ref_ws{weight_scale:g}"),
            )

            kern_weight_store = {}
            kern_traced = build_module(
                module_cls=KernelAttentionOProjStageModule,
                example_inputs=example_inputs,
                module_init_kwargs={
                    "config": cfg,
                    "seed": args.seed,
                    "weight_scale": weight_scale,
                    "_weight_store": kern_weight_store,
                },
                tp_degree=1,
                logical_nc_config=2,
                compiler_args=COMPILER_ARGS,
                compiler_workdir=os.path.join(workdir, f"kern_ws{weight_scale:g}"),
            )
            _check_weights(ref_weight_store, kern_weight_store)

            strict_fail = 0
            loose_fail = 0
            cases = generate_cases(args.n_samples, include_edge_cases=not args.no_edge_cases)
            print(
                f"  cases={len(cases)} regular={len(cases) - (0 if args.no_edge_cases else len(DEFAULT_EDGE_CASES))} "
                f"edge={0 if args.no_edge_cases else len(DEFAULT_EDGE_CASES)}",
                flush=True,
            )
            for case in cases:
                hidden, ref_mask, _, position_ids, K_cache, V_cache = make_sample(case)
                q, k, v = _make_qkv_inputs(hidden)
                expected = ref_traced(q, k, v, ref_mask, position_ids, K_cache.clone(), V_cache.clone())
                actual = kern_traced(q, k, v, ref_mask, position_ids, K_cache.clone(), V_cache.clone())
                pass_strict = outputs_close((expected,), (actual,), atol=1e-5, rtol=0.0)
                pass_loose = outputs_close((expected,), (actual,), atol=1e-5, rtol=1e-2)
                strict_fail += 0 if pass_strict else 1
                loose_fail += 0 if pass_loose else 1
                max_abs = _stats(expected, actual)
                if args.print_failures and (not pass_strict or not pass_loose):
                    verdict = "FAIL" if not pass_loose else "STRICT"
                    print(f"  {format_case(case)} out={max_abs:.3e} {verdict}", flush=True)
                if (case["sample_idx"] + 1) % 32 == 0 or case is cases[-1]:
                    print(f"  {format_case(case)} out={max_abs:.3e}", flush=True)

            total_samples += len(cases)
            total_strict_fail += strict_fail
            total_loose_fail += loose_fail
            print(f"\nSummary for weight_scale={weight_scale:g}", flush=True)
            print(f"  strict: {_fmt_pct(strict_fail, len(cases))}", flush=True)
            print(f"  loose : {_fmt_pct(loose_fail, len(cases))}", flush=True)

    print(f"\nAggregate strict: {_fmt_pct(total_strict_fail, total_samples)}", flush=True)
    print(f"Aggregate loose : {_fmt_pct(total_loose_fail, total_samples)}", flush=True)
    if total_loose_fail:
        raise AssertionError(f"loose tolerance failed for {total_loose_fail} / {total_samples} samples")
    print("OVERALL: PASS", flush=True)


if __name__ == "__main__":
    main()
