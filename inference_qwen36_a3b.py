"""End-to-end inference driver for Qwen3.6-35B-A3B on Trainium.

Default config: trn3, TP=4, LNC=2, bf16, seq_len=128, greedy sampling.
Compiles once into --compiled-path, then loads and generates for a few
short prompts to validate that the port produces sensible tokens.

Usage:
    python inference_qwen36_a3b.py \\
        --model-path /home/ubuntu/models/Qwen3.6-35B-A3B \\
        --compiled-path /home/ubuntu/models/qwen36_a3b_traced

    --max-new-tokens defaults to filling the window (seq_len - prompt_len).

Environment variables (override defaults without flags):
    A3B_TP_DEGREE   : tensor-parallel degree (default 4)
    A3B_SEQ_LEN     : max sequence length (default 128)
    USE_PYTORCH_CHUNK=1 : fall back to PyTorch DeltaNet path (debugging)
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import transformers
from transformers import AutoTokenizer, GenerationConfig

# Make `models.qwen3_6_moe.*` importable when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Route decode through `forward_all_experts` instead of `forward_selective_loading`.
# Selective loading does an indirect gather over expert weights that hits an OOB
# DMA on trn3 for our 256-expert / batch=1 / top_k=8 shape; all-experts is
# equivalent in FLOPs but uses a contiguous batched matmul. This mirrors the
# upstream qwen3_moe demo which picks tkg_batch_size so the same branch fires.
import neuronx_distributed.modules.moe.expert_mlps_v2 as _expert_mlps_v2  # noqa: E402
_expert_mlps_v2.DEFAULT_SELECTIVE_LOADING_THRESHOLD = 0.0

from neuronx_distributed_inference.models.config import (  # noqa: E402
    MoENeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import (  # noqa: E402
    HuggingFaceGenerationAdapter,
)
from neuronx_distributed_inference.modules.generation.sampling import (  # noqa: E402
    prepare_sampling_params,
)
from neuronx_distributed_inference.utils.benchmark import (  # noqa: E402
    Benchmark,
    create_submodule_latency_collectors,
    generate_report,
    register_latency_collectors,
)

from models.qwen3_6_moe.modeling_qwen36_a3b import (  # noqa: E402
    NeuronQwen36A3BForCausalLM,
    Qwen36A3BInferenceConfig,
)


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Q: What color is the sky?\nA:",
    "def fibonacci(n):\n    ",
    "Once upon a time, in a small villa,",
]


def build_inference_config(
    model_path: str,
    tp_degree: int,
    seq_len: int,
    mtp_spec_decode: bool = False,
) -> Qwen36A3BInferenceConfig:
    """Build the NxDI config from the HF config.json on disk."""
    with open(os.path.join(model_path, "config.json")) as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)

    config_dict = dict(text_config)
    config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    if "rope_parameters" in text_config:
        config_dict["rope_theta"] = text_config["rope_parameters"].get(
            "rope_theta", 10_000_000
        )
    config_dict.setdefault("tie_word_embeddings", False)

    # block_size must exceed (seq_len * num_experts_per_tok) so prefill takes
    # forward_all_experts instead of forward_blockwise (the NKI blockwise
    # kernel is not bundled in this SDK build).
    blockwise_block_size = max(2048, seq_len * 8 * 2)

    # MTP self-speculative decoding enables the draft head + verify backbone
    # graphs and host-side argmax. Controlled by the mtp_spec_decode flag; the
    # legacy A3B_ENABLE_* env vars still work for the dev/diagnostic scripts.
    enable_mtp = mtp_spec_decode or os.environ.get("A3B_ENABLE_MTP") == "1"
    enable_verify = mtp_spec_decode or os.environ.get("A3B_ENABLE_VERIFY") == "1"
    host_argmax = mtp_spec_decode or os.environ.get("A3B_RETURN_LOGITS") == "1"

    on_device_sampling = None if host_argmax else OnDeviceSamplingConfig(top_k=1)

    neuron_config = MoENeuronConfig(
        tp_degree=tp_degree,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=on_device_sampling,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
        blockwise_matmul_config={"block_size": blockwise_block_size},
    )

    # Draft head (3rd graph) + verify backbone (4th graph, n_active=2). Enabling
    # MTP disables NxD weight-layout optimization (priority_model_idx cleared in
    # enable_token_generation) -- the WLO hlo-opt pass crashes on the draft HLO.
    config_dict["mtp_num_hidden_layers"] = 1 if enable_mtp else 0
    config_dict["output_trunk_hidden"] = (
        enable_mtp or os.environ.get("A3B_OUTPUT_TRUNK_HIDDEN") == "1"
    )
    if enable_verify:
        config_dict["enable_verify_backbone"] = True

    return Qwen36A3BInferenceConfig(neuron_config=neuron_config, **config_dict)


def maybe_compile(model_path: str, compiled_path: str, inf_config) -> None:
    """Compile only if no model.pt is already present."""
    neff_path = os.path.join(compiled_path, "model.pt")
    if os.path.exists(neff_path):
        print(f"Found compiled artifacts at {compiled_path}; skipping compile.")
        return
    print(f"Compiling Qwen3.6-A3B to {compiled_path} (this can take 10-20 minutes)...")
    t0 = time.time()
    model = NeuronQwen36A3BForCausalLM(model_path, inf_config)
    model.compile(compiled_path)
    del model
    gc.collect()
    print(f"  compile done in {time.time() - t0:.1f}s")


def load_model(compiled_path: str) -> NeuronQwen36A3BForCausalLM:
    print(f"Loading compiled model from {compiled_path}...")
    model = NeuronQwen36A3BForCausalLM(compiled_path)
    model.load(compiled_path)
    return model


def resolve_max_new_tokens(model, prompt_len: int, requested: Optional[int] = None) -> int:
    """New-token budget that fills the sequence window: seq_len - prompt_len.

    With no explicit --max-new-tokens, generate until the sequence is full; a
    given value is capped to the remaining room.
    """
    room = model.neuron_config.seq_len - prompt_len
    return room if requested is None else min(requested, room)


def generate(
    model, tokenizer, prompt: str, max_new_tokens: Optional[int] = None
) -> tuple:
    """Greedy generate for a single prompt; return (token_ids, decoded_text).

    ``max_new_tokens=None`` fills the sequence window (seq_len - prompt_len).
    This is a correctness/sanity pass only -- timing is done separately by
    benchmark_device(), which measures on-device submodule latency.
    """
    gen_config = GenerationConfig(
        do_sample=False,  # greedy: matches OnDeviceSamplingConfig(top_k=1)
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_config.transformers_version = transformers.__version__

    adapter = HuggingFaceGenerationAdapter(model)
    adapter.generation_config.transformers_version = transformers.__version__

    inputs = tokenizer(prompt, padding=True, return_tensors="pt")
    new_tokens = resolve_max_new_tokens(model, inputs.input_ids.shape[1], max_new_tokens)

    out_ids = adapter.generate(
        inputs.input_ids,
        generation_config=gen_config,
        attention_mask=inputs.attention_mask,
        max_new_tokens=new_tokens,
    )

    full_ids = out_ids[0].tolist()
    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    return full_ids, text


def benchmark_device(
    model, tokenizer, prompt: str, max_new_tokens: Optional[int], num_runs: int = 20
) -> dict:
    """On-device latency/throughput benchmark (replaces host wall-clock timing).

    Mirrors NxDI's `benchmark_sampling`: one warm-up run (excluded), then
    `num_runs` timed runs with `model.reset()` between them. Latency collectors
    are registered as forward pre/post hooks on the compiled submodule wrappers
    *after* warm-up, so each measured interval brackets a blocking on-device
    execution of that submodule (context-encoding vs token-generation) rather
    than a single Python-side `time.time()` around `generate()`. Reports
    p50/p90/p99 latency and throughput for the e2e run and per submodule.
    """
    neuron_config = model.neuron_config

    inputs = tokenizer(prompt, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    prompt_len = input_ids.shape[1]

    # Force fixed-length runs so every iteration does identical work.
    new_tokens = resolve_max_new_tokens(model, prompt_len, max_new_tokens)
    neuron_config.max_new_tokens = new_tokens

    gen_config = GenerationConfig(
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_config.transformers_version = transformers.__version__
    # On-device sampling can't honor min_new_tokens to override EOS, so drop EOS
    # to guarantee a full-length decode every run.
    if model.on_device_sampling:
        gen_config.eos_token_id = []

    sampling_params = prepare_sampling_params(
        batch_size=neuron_config.batch_size,
        top_k=[1],
        top_p=[1.0],
        temperature=[1.0],
    )

    input_param = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "generation_config": gen_config,
        "max_new_tokens": new_tokens,
        "min_new_tokens": new_tokens,
        "top_k": 1,
        "do_sample": False,
        "sampling_params": sampling_params,
    }

    latency_collectors = create_submodule_latency_collectors(model)

    def post_warmup_func():
        # Register after warm-up so one-time costs aren't recorded.
        register_latency_collectors(latency_collectors, model)

    adapter = HuggingFaceGenerationAdapter(model)
    adapter.generation_config.transformers_version = transformers.__version__

    bench = Benchmark(
        adapter.generate,
        input_param,
        num_runs=num_runs,
        preprocess_func=model.reset,
        post_warmup_func=post_warmup_func,
    )
    bench.run()

    report = {
        "e2e_model": generate_report(
            bench.latency_list,
            neuron_config.max_length,
            neuron_config.max_batch_size,
            n_runs=bench.num_runs,
        )
    }
    # Per-submodule reports keyed by the wrapper tag strings.
    for key, collector in latency_collectors.items():
        tokens_len = neuron_config.max_length
        if key == "context_encoding_model":
            tokens_len = prompt_len
        elif key == "token_generation_model":
            tokens_len = new_tokens
        report[key] = generate_report(
            collector.latency_list,
            tokens_len,
            neuron_config.max_batch_size,
            n_runs=bench.num_runs,
        )

    model.reset()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="HF checkpoint dir")
    parser.add_argument(
        "--compiled-path",
        default=os.environ.get("A3B_COMPILED_PATH"),
        help="Compiled artifacts dir (default depends on --mtp-spec-decode)",
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=int(os.environ.get("A3B_TP_DEGREE", "4")),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=int(os.environ.get("A3B_SEQ_LEN", "128")),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="New tokens to generate (default: fill the window, seq_len - prompt_len)",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=DEFAULT_PROMPTS,
        help="Prompts to generate (default: a built-in sanity set)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=int(os.environ.get("A3B_BENCH_RUNS", "10")),
        help="Timed runs for the on-device benchmark (after one warm-up run)",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Only run the sanity generations; skip the device benchmark",
    )
    parser.add_argument(
        "--mtp-spec-decode",
        action="store_true",
        help="Generate with MTP self-speculative decoding (draft + verify graphs)",
    )
    parser.add_argument(
        "--check-equivalence",
        action="store_true",
        help="With --mtp-spec-decode: also run plain greedy and report any divergence",
    )
    args = parser.parse_args()

    # Header label only; the real per-prompt budget is seq_len - prompt_len when
    # --max-new-tokens is unset.
    mnt_desc = (
        "seq_len - prompt_len" if args.max_new_tokens is None else str(args.max_new_tokens)
    )

    compiled_path = args.compiled_path or (
        "/home/ubuntu/models/qwen36_a3b_specdec_traced"
        if args.mtp_spec_decode
        else "/home/ubuntu/models/qwen36_a3b_traced"
    )

    inf_config = build_inference_config(
        args.model_path, args.tp_degree, args.seq_len,
        mtp_spec_decode=args.mtp_spec_decode,
    )
    maybe_compile(args.model_path, compiled_path, inf_config)
    model = load_model(compiled_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mtp_spec_decode:
        from models.qwen3_6_moe.mtp_spec_decode import spec_decode

        print("\n" + "=" * 70)
        print(f"MTP SPECULATIVE DECODE")
        print("=" * 70)
        for prompt in args.prompts:
            prompt_len = len(tokenizer(prompt).input_ids)
            new_tokens = resolve_max_new_tokens(model, prompt_len, args.max_new_tokens)
            ids, accept_rate = spec_decode(
                model, tokenizer, inf_config, prompt, new_tokens
            )
            new_ids = ids[len(tokenizer(prompt).input_ids):]
            print(f"\n--- prompt ---\n{prompt}")
            print(f"--- completion ({len(new_ids)} tok, accept_rate={accept_rate:.1%}) ---")
            print(f"decoded: {tokenizer.decode(new_ids, skip_special_tokens=True)!r}")
            if args.check_equivalence:
                ref_ids, _ = generate(model, tokenizer, prompt, args.max_new_tokens)
                ref_new = ref_ids[len(tokenizer(prompt).input_ids):]
                # The loop may overshoot by one (an accept commits 2 tokens), so
                # compare over the common length, not the exact count.
                n = min(len(new_ids), len(ref_new))
                div = next((i for i in range(n) if new_ids[i] != ref_new[i]), None)
                if div is None:
                    print(f"  equivalence vs plain greedy: bit-identical ({n} tokens)")
                else:
                    print(
                        f"  equivalence vs plain greedy: diverges at new-token idx {div} "
                        "(may be a bf16 near-tie)"
                    )
        return

    print("\n" + "=" * 70)
    print(f"GENERATING (max_new_tokens={mnt_desc}, greedy)")
    print("=" * 70)
    for prompt in args.prompts:
        token_ids, text = generate(model, tokenizer, prompt, args.max_new_tokens)
        prompt_ids = tokenizer(prompt).input_ids
        new_ids = token_ids[len(prompt_ids):]
        n_new = len(new_ids)
        print(f"\n--- prompt ---\n{prompt}")
        print(f"--- completion ({n_new} tok) ---")
        print(f"new_ids: {new_ids}")
        print(f"new_tokens: {tokenizer.convert_ids_to_tokens(new_ids)}")
        print(f"decoded (raw): {tokenizer.decode(new_ids, skip_special_tokens=False)!r}")
        print(f"decoded (clean): {tokenizer.decode(new_ids, skip_special_tokens=True)!r}")

    if args.skip_benchmark:
        return

    # On-device benchmark: warm up once, then time `num_runs` fixed-length runs
    # with per-submodule forward hooks. Replaces host-side time.time() timing.
    bench_prompt = args.prompts[0]
    print("\n" + "=" * 70)
    print(
        f"BENCHMARK (max_new_tokens={mnt_desc}, {args.num_runs} runs, "
        f"1 warm-up) on: {bench_prompt!r}"
    )
    print("=" * 70)
    report = benchmark_device(
        model, tokenizer, bench_prompt, args.max_new_tokens, num_runs=args.num_runs,
    )
    for key, sub in report.items():
        if sub is None:
            continue
        print(
            f"{key:24s} "
            f"p50={sub['latency_ms_p50']:8.2f}ms  "
            f"p90={sub['latency_ms_p90']:8.2f}ms  "
            f"p99={sub['latency_ms_p99']:8.2f}ms  "
            f"avg={sub['latency_ms_avg']:8.2f}ms  "
            f"throughput={sub['throughput']:7.1f} tok/s"
        )
    print("\n(full report)")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
