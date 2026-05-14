"""Thin compile+benchmark driver for the gpt-oss megakernel study.

Bypasses main.py's evaluate_single dual-compile flow so we can compile ONE
model (mega or XLA) into a dedicated artifact dir, then benchmark it.

Usage:
  python study_driver.py --mode {nki|xla} --seq-len N --batch-size B \
      --compiled-out /path/to/artifact_dir --report-out /path/to/report.json \
      [--skip-compile] [--prompt "..."] [--prompt "..."]

Prints benchmark_report.json to --report-out and stdout.
"""
import argparse
import copy
import importlib
import json
import os
import sys
import time

# Force trn3 + LNC=1 before anything Neuron-y imports.
# trn3pd98.3xlarge has 8 LNC=1 cores (or 4 LNC=2 cores). TP=8 + LNC=2 doesn't
# fit and crashes at load with c10::Error.
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3")
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"

# Add repo root to path so megakernels.* import
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from neuronx_distributed_inference.models.config import (
    OnDeviceSamplingConfig,
    to_torch_dtype,
)
from neuronx_distributed_inference.modules.generation.sampling import (
    prepare_sampling_params,
)
from neuronx_distributed_inference.utils.benchmark import (
    Benchmark,
    create_submodule_latency_collectors,
    generate_report,
    register_latency_collectors,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)
from neuronx_distributed_inference.utils.random import set_random_seed
from transformers import AutoTokenizer, GenerationConfig


_MODEL_REGISTRY = {
    "qwen3_moe": (
        "megakernels.qwen3_moe.qwen",
        "megakernels.qwen3_moe.qwen_with_megakernel",
        "NeuronQwen3MoeForCausalLM",
        "neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe",
    ),
    "gpt_oss": (
        "megakernels.gpt_oss.gpt_oss",
        "megakernels.gpt_oss.gpt_oss_with_megakernel",
        "NeuronGptOssForCausalLM",
        "neuronx_distributed_inference.models.gpt_oss.modeling_gpt_oss",
    ),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt_oss", choices=list(_MODEL_REGISTRY))
    p.add_argument("--mode", choices=["nki", "xla"], required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="If None, set to seq_len - input_len")
    p.add_argument("--tp-degree", type=int, default=8)
    p.add_argument("--logical-nc-config", type=int, default=1)
    p.add_argument("--model-path", default="/home/ubuntu/models/gpt-oss-20b")
    p.add_argument("--compiled-out", type=str, required=True)
    p.add_argument("--report-out", type=str, required=True)
    p.add_argument("--skip-compile", action="store_true")
    p.add_argument("--compile-only", action="store_true",
                   help="Compile then exit; do not load to device or benchmark.")
    p.add_argument("--bench-only", action="store_true",
                   help="Alias for --skip-compile.")
    p.add_argument("--prompt", dest="prompts", action="append", default=None)
    p.add_argument("--on-device-sampling", action="store_true", default=True)
    p.add_argument("--torch-dtype", type=to_torch_dtype, default="bfloat16")
    p.add_argument("--platform-target", default="trn3")
    return p.parse_args()


def build_config_kwargs(args, prompts):
    """Build NeuronConfig kwargs mirroring main.py's compile path."""
    kwargs = {
        "model_path": args.model_path,
        "compiled_model_path": args.compiled_out,
        "tp_degree": args.tp_degree,
        "logical_nc_config": args.logical_nc_config,
        "torch_dtype": args.torch_dtype,
        "seq_len": args.seq_len,
        "max_length": args.seq_len,
        "batch_size": args.batch_size,
        "max_batch_size": args.batch_size,
        "tkg_batch_size": args.batch_size,
        "ctx_batch_size": args.batch_size,
        "max_context_length": args.seq_len,
        "padding_side": "right",
        "platform_target": args.platform_target,
        "enable_bucketing": True,
        # Single-bucket compile for both CTE and TKG. Avoids auto-bucketing
        # producing seq=128 TKG buckets that mismatch seq_len during tracing.
        "token_generation_buckets": [args.seq_len],
        "context_encoding_buckets": [args.seq_len],
        # Leave attention_dtype/rpl_reduce_dtype at None (auto). Explicit
        # bf16 in earlier driver caused load-time c10::Error on XLA fresh compile.
        "cc_pipeline_tiling_factor": 2,
        # XLA MoE: use_torch_block_wise=True bypasses moe_block_tkg NKI kernel
        # (which requires LNC=2; we run LNC=1 to fit TP=8 on 8-core trn3 instance).
        # moe_fused_nki_kernel_enabled=False disables the *TKG* fused NKI MoE
        # kernel (which also hardcodes LNC=2 via the moe_block_tkg primitive).
        # Mega has its own NKI MoE path so this only affects xla mode.
        **({"blockwise_matmul_config": {"use_torch_block_wise": True},
            "moe_fused_nki_kernel_enabled": False,
            "router_topk_nki_kernel_enabled": False,
            "expert_mlp_nki_kernel_enabled": False,
            "shared_mlp_nki_kernel_enabled": False}
           if args.mode == "xla" else {}),
        # Mode-specific MoE config. XLA path uses moe_tp_degree=1 (per existing
        # baseline neuron_config.json); mega class forces moe_tp_degree=8.
        # Setting only for xla — let mega class own it for nki mode.
        **({"moe_tp_degree": 1, "fused_qkv": False} if args.mode == "xla" else {}),
        "moe_ep_degree": 1,
        # KV cache + paged-attention: single block covering seq_len, batch=B.
        # Default pa_block_size is seq_len so explicit set matches.
        "pa_block_size": args.seq_len,
        "pa_num_blocks": args.batch_size,
        "kv_cache_batch_size": args.batch_size,
        # Kernel flags are owned by the per-model NeuronConfig subclass
        # (GptOssV1MultilayerNeuronConfig for mega; stock for XLA).
    }
    if args.max_new_tokens is not None:
        kwargs["max_new_tokens"] = args.max_new_tokens
    if args.on_device_sampling:
        # Set kwargs that OnDeviceSamplingConfig is built from. The mega class
        # rebuilds OnDeviceSamplingConfig(**kwargs) so we have to set fields via
        # kwargs, not via a pre-built ODS object.
        kwargs.update({
            "top_k": 20,
            "top_p": 0.95,
            "temperature": 0.6,
            "global_topk": 20,
            "do_sample": True,
            "dynamic": False,
            "top_k_kernel_enabled": False,
            "on_device_sampling": True,
        })
        # Also build one explicit ODS in case xla path uses it; mega ignores this
        # via __init__ override.
        sampling_cfg = OnDeviceSamplingConfig(
            top_k=20, top_p=0.95, temperature=0.6, global_topk=20,
            do_sample=True, dynamic=False, top_k_kernel_enabled=False,
        )
        kwargs["on_device_sampling_config"] = sampling_cfg
    return kwargs


def prepare_inference(model_cls, kwargs, skip_compile, compile_only=False):
    config_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    print("Building neuron_config with:", {
        k: config_kwargs.get(k) for k in
        ["batch_size","seq_len","tp_degree","attn_block_tkg_nki_kernel_enabled"]
    })
    neuron_config = model_cls.get_neuron_config_cls()(**config_kwargs)
    print("Resolved kernel flags:",
          {k: getattr(neuron_config, k, None) for k in
           ["attn_block_tkg_nki_kernel_enabled","attn_tkg_nki_kernel_enabled",
            "mlp_tkg_nki_kernel_enabled","fused_qkv","moe_tp_degree",
            "logical_nc_config","tp_degree"]})
    config = model_cls.get_config_cls()(
        neuron_config, load_config=load_pretrained_config(kwargs["model_path"])
    )
    model = model_cls(kwargs["model_path"], config)

    if not skip_compile:
        t0 = time.monotonic()
        print(f"Compiling -> {kwargs['compiled_model_path']}")
        model.compile(kwargs["compiled_model_path"], debug=False)
        print(f"Compile took {time.monotonic()-t0:.1f}s")

    if compile_only:
        print("Compile-only: exiting before device load.")
        return None, None, None

    print(f"Loading from {kwargs['compiled_model_path']}")
    model.load(kwargs["compiled_model_path"])

    tokenizer = AutoTokenizer.from_pretrained(kwargs["model_path"], padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(kwargs["compiled_model_path"])
    neuron_config.pad_token_id = tokenizer.pad_token_id

    gen_config = GenerationConfig.from_pretrained(kwargs["model_path"])
    gen_config.update(do_sample=False, top_k=1, pad_token_id=tokenizer.pad_token_id)

    return model, tokenizer, gen_config


def benchmark_sampling(model, tokenizer, generation_config, prompts):
    neuron_config = model.neuron_config
    sampling_params = prepare_sampling_params(
        batch_size=neuron_config.batch_size,
        top_k=[1] * neuron_config.batch_size,
        top_p=[1.0] * neuron_config.batch_size,
        temperature=[1.0] * neuron_config.batch_size,
    )
    modified = copy.deepcopy(generation_config)
    if model.on_device_sampling:
        modified.eos_token_id = []

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    neuron_config.max_new_tokens = neuron_config.seq_len - input_ids.shape[1]
    print(f"prompt_len={input_ids.shape[1]} max_new_tokens={neuron_config.max_new_tokens}")

    input_param = dict(
        input_ids=input_ids,
        generation_config=modified,
        attention_mask=attention_mask,
        min_new_tokens=neuron_config.max_new_tokens,
        max_new_tokens=neuron_config.max_new_tokens,
        top_k=1,
        do_sample=False,
        sampling_params=sampling_params,
        max_length=neuron_config.max_length,
    )

    collectors = create_submodule_latency_collectors(model)
    gen_model = HuggingFaceGenerationAdapter(model)

    def post_warmup():
        register_latency_collectors(collectors, model)

    # Default Benchmark.num_runs=20; reduce to 10 to cap long-seq bench time
    # (at seq=8192 each iter generates ~8k tokens ≈ 50 sec).
    num_runs = int(os.environ.get("STUDY_NUM_RUNS", "10"))
    e2e = Benchmark(gen_model.generate, input_param, preprocess_func=model.reset,
                    post_warmup_func=post_warmup, num_runs=num_runs)
    e2e.run()

    report = {
        "e2e_model": generate_report(
            e2e.latency_list, neuron_config.max_length,
            neuron_config.max_batch_size, n_runs=e2e.num_runs
        )
    }
    for key, col in collectors.items():
        tokens_len = neuron_config.max_length
        if key == "context_encoding_model":
            tokens_len = neuron_config.seq_len - neuron_config.max_new_tokens
        elif key == "token_generation_model":
            tokens_len = neuron_config.max_new_tokens
        report[key] = generate_report(
            col.latency_list, tokens_len,
            neuron_config.max_batch_size, e2e.num_runs
        )
    return report


def main():
    args = parse_args()
    set_random_seed(0)

    if not args.prompts:
        # Default prompt — 10ish tokens; rest is decode
        args.prompts = ["I believe the meaning of life is"] * args.batch_size
    elif len(args.prompts) == 1 and args.batch_size > 1:
        args.prompts = args.prompts * args.batch_size
    assert len(args.prompts) == args.batch_size, \
        f"prompts={len(args.prompts)} != batch_size={args.batch_size}"

    xla_mod, nki_mod, cls_name, _ = _MODEL_REGISTRY[args.model]
    target_mod_name = nki_mod if args.mode == "nki" else xla_mod
    print(f"Loading {target_mod_name}")
    target_cls = getattr(importlib.import_module(target_mod_name), cls_name)

    os.makedirs(args.compiled_out, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)

    cfg_kwargs = build_config_kwargs(args, args.prompts)
    skip_compile = args.skip_compile or args.bench_only
    model, tok, gen_cfg = prepare_inference(
        target_cls, cfg_kwargs, skip_compile, compile_only=args.compile_only
    )
    if args.compile_only:
        # Save meta only; bench will be a follow-up call.
        with open(args.report_out, "w") as f:
            json.dump({"__study_meta__": {
                "mode": args.mode, "seq_len": args.seq_len,
                "batch_size": args.batch_size, "tp_degree": args.tp_degree,
                "logical_nc_config": args.logical_nc_config,
                "platform_target": args.platform_target,
                "compiled_out": args.compiled_out,
                "compile_only": True,
            }}, f, indent=2)
        print(f"Compile-only stub written: {args.report_out}")
        return

    report = benchmark_sampling(model, tok, gen_cfg, args.prompts)
    report["__study_meta__"] = {
        "mode": args.mode,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "tp_degree": args.tp_degree,
        "logical_nc_config": args.logical_nc_config,
        "platform_target": args.platform_target,
        "compiled_out": args.compiled_out,
    }
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written: {args.report_out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
