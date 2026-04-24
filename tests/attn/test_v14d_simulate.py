"""
CPU simulator harness for the test-local v14d attention kernel copy.

Default mode:
  Compiles the Qwen attention reference module once, captures its weights, then
  runs the same sampled cases through nki.simulate at LNC=2.

Replay mode:
  Loads a dump written by test_v14d_vs_qwen_trn3.py and replays that exact case
  against both the recorded hardware-kernel output and recorded reference output.

Run:
  NKI_PRECISE_FP=1 python tests/attn/test_v14d_simulate.py
  NKI_PRECISE_FP=1 python tests/attn/test_v14d_simulate.py --n-samples 32
  NKI_PRECISE_FP=1 python tests/attn/test_v14d_simulate.py --load-dump ./first_strict_ws1_s7.pt
"""

import argparse
import os
import sys
import tempfile

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NKI_PRECISE_FP"] = "1"

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
    KernelSimModule,
    RefAttnModule,
    compute_sample_stats,
    format_case,
    generate_cases,
    make_config,
    make_sample,
    outputs_close,
)
import torch


def _print_component_summary(prefix: str, stats: dict) -> None:
    print(
        f"{prefix} attn={stats['attn_max_abs_err']:.3e} "
        f"K={stats['K_max_abs_err']:.3e} "
        f"V={stats['V_max_abs_err']:.3e} "
        f"max={stats['max_abs_err']:.3e}",
        flush=True,
    )


def _run_from_dump(dump_path: str) -> None:
    print(f"Loading dump: {dump_path}", flush=True)
    dump = torch.load(dump_path, map_location="cpu", weights_only=True)
    cfg = make_config()
    weight_store = {key: dump[key].cpu() for key in (
        "input_layernorm_weight",
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
        "o_proj_weight",
        "q_norm_weight",
        "k_norm_weight",
    )}
    kern_sim = KernelSimModule(cfg, weight_store)

    hidden = dump["hidden_states"].cpu()
    position_ids = dump["position_ids"].cpu()
    K_cache = dump["K_cache_in"].cpu()
    V_cache = dump["V_cache_in"].cpu()
    cos = dump["cos"].cpu()
    sin = dump["sin"].cpu()

    sim_outputs = kern_sim(hidden, position_ids, K_cache, V_cache, cos=cos, sin=sin)
    ref_outputs = (dump["ref_out"].cpu(), dump["ref_K"].cpu(), dump["ref_V"].cpu())
    kern_outputs = (dump["kern_out"].cpu(), dump["kern_K"].cpu(), dump["kern_V"].cpu())

    print(f"Case: {dump.get('case', {})}", flush=True)
    print("", flush=True)

    print("--- simulator vs recorded hardware kernel output ---", flush=True)
    sim_vs_kern = compute_sample_stats(kern_outputs, sim_outputs)
    _print_component_summary("  delta:", sim_vs_kern)
    print(
        f"  verdict: {'MATCH' if outputs_close(kern_outputs, sim_outputs, atol=1e-5, rtol=0.0) else 'MISMATCH'}",
        flush=True,
    )

    print("\n--- simulator vs recorded reference output ---", flush=True)
    sim_vs_ref = compute_sample_stats(ref_outputs, sim_outputs)
    _print_component_summary("  delta:", sim_vs_ref)
    print(
        f"  verdict: {'MATCH' if outputs_close(ref_outputs, sim_outputs, atol=1e-5, rtol=1e-2) else 'MISMATCH'}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=192)
    parser.add_argument("--load-dump", default=None, metavar="PATH")
    args = parser.parse_args()

    if args.load_dump:
        _run_from_dump(args.load_dump)
        return

    cfg = make_config()
    print(
        f"Config: H={H} d={D} Hq={cfg.num_attention_heads} Hkv={cfg.num_key_value_heads} "
        f"S={S} TP={TP_DEGREE} LNC={LNC}",
        flush=True,
    )
    print(
        f"weight_scale={args.weight_scale:g}  n_samples={args.n_samples}  "
        f"NKI_PRECISE_FP={os.environ.get('NKI_PRECISE_FP')}",
        flush=True,
    )

    example_hidden = torch.zeros(B, 1, H, dtype=torch.bfloat16)
    example_mask = torch.zeros(B, 1, 1, S, dtype=torch.bool)
    example_pos = torch.zeros(B, 1, dtype=torch.int32)
    example_K = torch.zeros(B, 1, S, D, dtype=torch.bfloat16)
    example_V = torch.zeros(B, 1, S, D, dtype=torch.bfloat16)
    example_inputs = [(example_hidden, example_mask, example_pos, example_K, example_V)]

    with tempfile.TemporaryDirectory(prefix="v14d_attn_sim_") as workdir:
        print("\nCompiling RefAttnModule...", flush=True)
        weight_store = {}
        ref_traced = build_module(
            module_cls=RefAttnModule,
            example_inputs=example_inputs,
            module_init_kwargs={
                "config": cfg,
                "seed": args.seed,
                "weight_scale": args.weight_scale,
                "_weight_store": weight_store,
            },
            tp_degree=TP_DEGREE,
            logical_nc_config=LNC,
            compiler_args=COMPILER_ARGS,
            compiler_workdir=os.path.join(workdir, "ref"),
        )
        print("RefAttnModule compile PASS\n", flush=True)

        print("Building KernelSimModule...", flush=True)
        kern_sim = KernelSimModule(cfg, weight_store)
        print("KernelSimModule ready\n", flush=True)

        strict_fail = 0
        loose_fail = 0
        cases = generate_cases(args.n_samples)
        for case in cases:
            hidden, ref_mask, _, position_ids, K_cache, V_cache = make_sample(case)
            ref_outputs = ref_traced(
                hidden,
                ref_mask,
                position_ids,
                K_cache.clone(),
                V_cache.clone(),
            )
            sim_outputs = kern_sim(
                hidden,
                position_ids,
                K_cache.clone(),
                V_cache.clone(),
            )

            stats = compute_sample_stats(ref_outputs, sim_outputs)
            pass_strict = outputs_close(ref_outputs, sim_outputs, atol=1e-5, rtol=0.0)
            pass_loose = outputs_close(ref_outputs, sim_outputs, atol=1e-5, rtol=1e-2)
            if not pass_strict:
                strict_fail += 1
            if not pass_loose:
                loose_fail += 1

            if (case["sample_idx"] + 1) % 32 == 0 or case["sample_idx"] == len(cases) - 1 or not pass_loose:
                verdict = "PASS" if pass_loose else "FAIL"
                print(f"  {format_case(case)}  {verdict}", flush=True)
                _print_component_summary("    delta:", stats)

        print(f"\nstrict_fail={strict_fail}/{len(cases)}", flush=True)
        print(f"loose_fail={loose_fail}/{len(cases)}", flush=True)
        if loose_fail:
            raise AssertionError(f"loose tolerance failed for {loose_fail} / {len(cases)} samples")
        print("OVERALL: PASS", flush=True)


if __name__ == "__main__":
    main()
