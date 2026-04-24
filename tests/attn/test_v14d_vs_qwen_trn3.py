"""
Hardware accuracy sweep for the test-local v14d attention kernel copy.

Reference:
  qwen.py -> NeuronQwen3MoEAttention with input_layernorm applied explicitly.

Kernel:
  tests/attn/kernel_v14d_debug.py wrapped by KernelAttnModule in common.py.

Run:
  python tests/attn/test_v14d_vs_qwen_trn3.py
  python tests/attn/test_v14d_vs_qwen_trn3.py --n-samples 32 --weight-scales 1.0,2.0
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
sys.path.insert(0, _HERE)

from neuronx_distributed_inference.utils.testing import build_module

from common import (
    B,
    COMPILER_ARGS,
    D,
    H,
    LNC,
    S,
    TP_DEGREE,
    KernelAttnModule,
    RefAttnModule,
    build_rotary_emb,
    compare_weight_stores,
    compute_sample_stats,
    dump_failure_sample,
    format_case,
    generate_cases,
    get_cos_sin_at_pos,
    make_config,
    make_sample,
    outputs_close,
)
import torch


def _fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "0/0 (nan%)"
    return f"{n}/{total} ({100.0 * n / total:5.2f}%)"


def _print_weight_diffs(ref_weight_store, kern_weight_store) -> None:
    print("Comparing captured reference/kernel weights...", flush=True)
    diffs = compare_weight_stores(ref_weight_store, kern_weight_store)
    for key, max_abs in diffs.items():
        print(f"  {key:<22} max_abs_diff={max_abs:.6g}", flush=True)
    if any(value != 0.0 for value in diffs.values()):
        raise AssertionError("reference and kernel modules did not initialize identically")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weight-scales",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[1.0],
        metavar="W1,W2,...",
        help="Comma-separated parameter scale factors applied before compilation",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=192,
        help="Number of sampled attention cases per weight scale",
    )
    parser.add_argument(
        "--dump-dir",
        default=".",
        metavar="DIR",
        help="Directory to write failure dumps",
    )
    parser.add_argument(
        "--dump-first-strict-per-ws",
        action="store_true",
        help="Dump the first strict failure for each weight scale instead of only the first global one",
    )
    args = parser.parse_args()

    cfg = make_config()
    rotary_emb = build_rotary_emb(cfg)
    print(
        f"Config: H={H} d={D} Hq={cfg.num_attention_heads} Hkv={cfg.num_key_value_heads} "
        f"S={S} TP={TP_DEGREE} LNC={LNC}",
        flush=True,
    )
    print(f"weight_scales={args.weight_scales}  n_samples/scale={args.n_samples}", flush=True)

    example_hidden = torch.zeros(B, 1, H, dtype=torch.bfloat16)
    example_mask = torch.zeros(B, 1, 1, S, dtype=torch.bool)
    example_pos = torch.zeros(B, 1, dtype=torch.int32)
    example_K = torch.zeros(B, 1, S, D, dtype=torch.bfloat16)
    example_V = torch.zeros(B, 1, S, D, dtype=torch.bfloat16)
    example_inputs = [(example_hidden, example_mask, example_pos, example_K, example_V)]

    total_loose_fail = 0
    total_strict_fail = 0
    total_samples = 0
    global_first_strict_dumped = False

    with tempfile.TemporaryDirectory(prefix="v14d_attn_vs_qwen_") as workdir:
        for weight_scale in args.weight_scales:
            scale_tag = f"ws{weight_scale:g}"
            print(f"\n{'=' * 72}", flush=True)
            print(f"weight_scale={weight_scale:g}", flush=True)
            print(f"{'=' * 72}", flush=True)

            ref_weight_store = {}
            print("Compiling RefAttnModule...", flush=True)
            ref_traced = build_module(
                module_cls=RefAttnModule,
                example_inputs=example_inputs,
                module_init_kwargs={
                    "config": cfg,
                    "seed": args.seed,
                    "weight_scale": weight_scale,
                    "_weight_store": ref_weight_store,
                },
                tp_degree=TP_DEGREE,
                logical_nc_config=LNC,
                compiler_args=COMPILER_ARGS,
                compiler_workdir=os.path.join(workdir, f"ref_{scale_tag}"),
            )
            print("RefAttnModule compile PASS\n", flush=True)

            kern_weight_store = {}
            print("Compiling KernelAttnModule...", flush=True)
            kern_traced = build_module(
                module_cls=KernelAttnModule,
                example_inputs=example_inputs,
                module_init_kwargs={
                    "config": cfg,
                    "seed": args.seed,
                    "weight_scale": weight_scale,
                    "_weight_store": kern_weight_store,
                },
                tp_degree=TP_DEGREE,
                logical_nc_config=LNC,
                compiler_args=COMPILER_ARGS,
                compiler_workdir=os.path.join(workdir, f"kern_{scale_tag}"),
            )
            print("KernelAttnModule compile PASS\n", flush=True)

            _print_weight_diffs(ref_weight_store, kern_weight_store)

            cases = generate_cases(args.n_samples)
            strict_fail = 0
            loose_fail = 0
            first_scale_dumped = False

            for case in cases:
                hidden, ref_mask, kern_mask, position_ids, K_cache, V_cache = make_sample(case)
                ref_outputs = ref_traced(
                    hidden,
                    ref_mask,
                    position_ids,
                    K_cache.clone(),
                    V_cache.clone(),
                )
                kern_outputs = kern_traced(
                    hidden,
                    kern_mask,
                    position_ids,
                    K_cache.clone(),
                    V_cache.clone(),
                )

                stats = compute_sample_stats(kern_outputs, ref_outputs)
                pass_strict = outputs_close(kern_outputs, ref_outputs, atol=1e-5, rtol=0.0)
                pass_loose = outputs_close(kern_outputs, ref_outputs, atol=1e-5, rtol=1e-2)

                if not pass_strict:
                    strict_fail += 1
                if not pass_loose:
                    loose_fail += 1

                should_dump = (
                    (args.dump_first_strict_per_ws and not first_scale_dumped)
                    or ((not args.dump_first_strict_per_ws) and (not global_first_strict_dumped))
                )
                if should_dump and not pass_strict:
                    cos, sin = get_cos_sin_at_pos(rotary_emb, hidden, position_ids)
                    tag = "first_strict"
                    dump_path = os.path.join(
                        args.dump_dir,
                        f"{tag}_{scale_tag}_s{case['sample_idx']}.pt",
                    )
                    dump_failure_sample(
                        dump_path=dump_path,
                        case=case,
                        sample=(hidden, ref_mask, kern_mask, position_ids, K_cache, V_cache),
                        ref_outputs=ref_outputs,
                        kern_outputs=kern_outputs,
                        weight_store=kern_weight_store,
                        cos=cos,
                        sin=sin,
                        weight_scale=weight_scale,
                        seed=args.seed,
                        tag=tag,
                        stats={
                            **stats,
                            "pass_strict": pass_strict,
                            "pass_loose": pass_loose,
                        },
                    )
                    first_scale_dumped = True
                    global_first_strict_dumped = True
                    print(f"\nDumped {tag}: {format_case(case)}", flush=True)
                    print(f"  dump written to: {dump_path}", flush=True)
                    print("  replay with:", flush=True)
                    print(
                        f"    NKI_PRECISE_FP=1 python tests/attn/test_v14d_simulate.py --load-dump {dump_path}",
                        flush=True,
                    )
                    print("", flush=True)

                if (case["sample_idx"] + 1) % 32 == 0 or case["sample_idx"] == len(cases) - 1:
                    verdict = "PASS" if pass_loose else "FAIL"
                    print(
                        f"  {format_case(case)}  "
                        f"attn={stats['attn_max_abs_err']:.3e} "
                        f"K={stats['K_max_abs_err']:.3e} "
                        f"V={stats['V_max_abs_err']:.3e} "
                        f"{verdict}",
                        flush=True,
                    )

            total_samples += len(cases)
            total_strict_fail += strict_fail
            total_loose_fail += loose_fail
            print(f"\nSummary for weight_scale={weight_scale:g}", flush=True)
            print(f"  strict: {_fmt_pct(strict_fail, len(cases))}", flush=True)
            print(f"  loose : {_fmt_pct(loose_fail, len(cases))}", flush=True)

    print(f"\n{'=' * 72}", flush=True)
    print("Aggregate summary", flush=True)
    print(f"{'=' * 72}", flush=True)
    print(f"  strict: {_fmt_pct(total_strict_fail, total_samples)}", flush=True)
    print(f"  loose : {_fmt_pct(total_loose_fail, total_samples)}", flush=True)

    if total_loose_fail:
        raise AssertionError(f"loose tolerance failed for {total_loose_fail} / {total_samples} samples")
    print("OVERALL: PASS", flush=True)


if __name__ == "__main__":
    main()
