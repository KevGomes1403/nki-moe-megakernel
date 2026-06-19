"""End-to-end inference driver for Qwen3.6-35B-A3B on Trainium.

Default config: trn3pre, TP=4, LNC=2, bf16, seq_len=128, greedy sampling.
Compiles once into --compiled-path, then loads and generates for a few
short prompts to validate that the port produces sensible tokens.

Usage:
    python inference_qwen36_a3b.py \\
        --model-path /home/ubuntu/models/Qwen3.6-35B-A3B \\
        --compiled-path /home/ubuntu/models/qwen36_a3b_traced

    --max-new-tokens defaults to filling the window (seq_len - prompt_len).

Decode/spec graphs gather only the routed experts (NxDI selective loading);
prefill streams all experts, which is optimal at 128 tokens x top-8.

Environment variables (override defaults without flags):
    A3B_TP_DEGREE   : tensor-parallel degree (default 4)
    A3B_SEQ_LEN     : max sequence length (default 128)
    A3B_SKIP_SHARD=1 : trace graphs only; reuse an existing weights/ dir
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

# Pre-release trn3 silicon reports its platform as "trn3pre"
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3pre")

import torch
import transformers
from transformers import AutoTokenizer, GenerationConfig

# Make `models.qwen3_6_moe.*` importable when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from neuronx_distributed_inference.models.config import (  # noqa: E402
    FusedSpecNeuronConfig,
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
    LatencyCollector,
    create_submodule_latency_collectors,
    generate_report,
    register_forward_latency_collector,
    register_latency_collectors,
)

from models.qwen3_6_moe.modeling_qwen36_a3b import (  # noqa: E402
    NeuronQwen36A3BForCausalLM,
    NeuronQwen36MTPDraftForCausalLM,
    Qwen36A3BInferenceConfig,
    Qwen36SpecTarget,
)


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Q: What color is the sky?\nA:",
    "def fibonacci(n):\n    ",
    "Once upon a time, in a small villa,",
]


def _make_neuron_config(tp_degree, seq_len, blockwise_block_size, spec_decode):
    """MoENeuronConfig shared by the target and draft; adds the EAGLE fused-spec
    flags (speculation_length=2 == k=1) when speculating."""
    spec_kwargs = (
        dict(enable_eagle_speculation=True, speculation_length=2, is_eagle3=False)
        if spec_decode
        else {}
    )
    return MoENeuronConfig(
        tp_degree=tp_degree,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),  # greedy
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
        # A3B_SKIP_SHARD=1: trace graphs only; reuse an existing weights/ dir
        # (e.g. symlinked from another build with identical sharding).
        skip_sharding=os.environ.get("A3B_SKIP_SHARD") == "1",
        blockwise_matmul_config={"block_size": blockwise_block_size},
        **spec_kwargs,
    )


def build_inference_config(
    model_path: str,
    tp_degree: int,
    seq_len: int,
    spec_decode: bool = False,
    tkg_attention_kernel: bool = False,
) -> Qwen36A3BInferenceConfig:
    """Build the NxDI config from the HF config.json on disk.

    ``spec_decode`` builds the fused EAGLE speculation config: the backbone is the
    verify-mode target and the MTP head is a separate self-speculative draft
    (shared checkpoint). One CTE graph + one fused-speculation graph.
    """
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
    # Route the DeltaNet verify/decode recurrence through the TKG NKI kernel.
    # Carried by both the target and (harmlessly) the draft config below.
    config_dict["use_tkg_attention_kernel"] = tkg_attention_kernel

    # block_size must exceed (seq_len * num_experts_per_tok) so prefill takes
    # forward_all_experts instead of forward_blockwise (the NKI blockwise
    # kernel is not bundled in this SDK build).
    blockwise_block_size = max(2048, seq_len * 8 * 2)

    fused_spec_config = None
    if spec_decode:
        # Draft owns the MTP head (mtp_num_hidden_layers=1); same checkpoint.
        draft_dict = dict(config_dict)
        draft_dict["mtp_num_hidden_layers"] = 1
        draft_config = Qwen36A3BInferenceConfig(
            neuron_config=_make_neuron_config(
                tp_degree, seq_len, blockwise_block_size, spec_decode=True
            ),
            **draft_dict,
        )
        fused_spec_config = FusedSpecNeuronConfig(
            worker_cls=Qwen36SpecTarget,
            draft_config=draft_config,
            draft_model_path=model_path,
            draft_model_cls=NeuronQwen36MTPDraftForCausalLM,
        )

    # The target backbone carries no MTP head (the draft does).
    config_dict["mtp_num_hidden_layers"] = 0
    return Qwen36A3BInferenceConfig(
        neuron_config=_make_neuron_config(
            tp_degree, seq_len, blockwise_block_size, spec_decode=spec_decode
        ),
        fused_spec_config=fused_spec_config,
        **config_dict,
    )


def _presharded_weights_complete(compiled_path: str, tp_degree: int) -> bool:
    """True iff every rank's sharded checkpoint is present. Sharding runs at the
    tail of compile() and can be killed partway, leaving e.g. tp0/tp1 but not
    tp2/tp3."""
    weights_dir = os.path.join(compiled_path, "weights")
    return all(
        os.path.exists(os.path.join(weights_dir, f"tp{rank}_sharded_checkpoint.safetensors"))
        for rank in range(tp_degree)
    )


def maybe_compile(model_path: str, compiled_path: str, inf_config) -> None:
    """Compile if no model.pt is present; re-shard if the presharded set is
    incomplete.

    Sharding doesn't need the NEFFs, so an interrupted shard (model.pt plus a
    partial weights/ dir) is recovered by re-sharding alone -- no recompile.
    Delete weights/ to force a re-shard even when complete (model.pt stays, so
    no graph rebuild).
    """
    neuron_config = inf_config.neuron_config
    neff_path = os.path.join(compiled_path, "model.pt")

    if not os.path.exists(neff_path):
        print(f"Compiling Qwen3.6-A3B to {compiled_path} (this can take 10-20 minutes)...")
        t0 = time.time()
        model = NeuronQwen36A3BForCausalLM(model_path, inf_config)
        model.compile(compiled_path)
        del model
        gc.collect()
        print(f"  compile done in {time.time() - t0:.1f}s")
        return

    print(f"Found compiled artifacts at {compiled_path}; skipping compile.")
    if (
        neuron_config.save_sharded_checkpoint
        and not neuron_config.skip_sharding
        and not _presharded_weights_complete(compiled_path, neuron_config.tp_degree)
    ):
        print(f"Presharded weights incomplete; re-sharding into {compiled_path}/weights/ ...")
        t0 = time.time()
        model = NeuronQwen36A3BForCausalLM(model_path, inf_config)
        model.shard_weights(compiled_path)
        del model
        gc.collect()
        print(f"  re-shard done in {time.time() - t0:.1f}s")


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

    # Fused speculation runs through HF assisted decoding; prompt_lookup_num_tokens
    # selects that generation mode (NxDI then dispatches to _fused_assisted_decoding).
    spec_kwargs = {}
    if model.neuron_config.enable_fused_speculation:
        spec_kwargs["prompt_lookup_num_tokens"] = model.neuron_config.speculation_length

    out_ids = adapter.generate(
        inputs.input_ids,
        generation_config=gen_config,
        attention_mask=inputs.attention_mask,
        max_new_tokens=new_tokens,
        **spec_kwargs,
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
    # Fused speculation runs through HF assisted decoding (see generate());
    # prompt_lookup_num_tokens selects that mode so the benchmark exercises the
    # fused graph instead of crashing in the plain _sample path.
    if neuron_config.enable_fused_speculation:
        input_param["prompt_lookup_num_tokens"] = neuron_config.speculation_length

    latency_collectors = create_submodule_latency_collectors(model)
    # NxDI's helper doesn't cover the fused-spec decode graph; collect it too.
    if neuron_config.enable_fused_speculation and hasattr(model, "fused_spec_model"):
        latency_collectors["fused_speculation_model"] = LatencyCollector()

    def post_warmup_func():
        # Register after warm-up so one-time costs aren't recorded.
        register_latency_collectors(latency_collectors, model)
        if "fused_speculation_model" in latency_collectors:
            register_forward_latency_collector(
                latency_collectors["fused_speculation_model"], model.fused_spec_model
            )

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
    # Per-submodule reports keyed by the wrapper tag strings. The fused-spec graph
    # is the decode submodule (replaces token_generation_model).
    for key, collector in latency_collectors.items():
        tokens_len = neuron_config.max_length
        if key == "context_encoding_model":
            tokens_len = prompt_len
        elif key in ("token_generation_model", "fused_speculation_model"):
            tokens_len = new_tokens
        report[key] = generate_report(
            collector.latency_list,
            tokens_len,
            neuron_config.max_batch_size,
            n_runs=bench.num_runs,
        )

    # Realized speculation gain: tokens committed per fused-graph round (1..spec_len).
    # The fused collector records one latency per round, so rounds = entries/run.
    if neuron_config.enable_fused_speculation:
        fused = latency_collectors.get("fused_speculation_model")
        if fused is not None and len(fused.latency_list):
            rounds_per_run = len(fused.latency_list) / bench.num_runs
            report["e2e_model"]["tokens_per_round"] = round(new_tokens / rounds_per_run, 3)

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
        help="Build/run with fused MTP self-speculative decoding (one fused graph)",
    )
    parser.add_argument(
        "--tkg-attention-kernel",
        action="store_true",
        help="Use the DeltaNet TKG NKI kernel for the verify/decode recurrence "
        "(default: PyTorch). Pair with --mtp-spec-decode to exercise verify.",
    )
    args = parser.parse_args()

    # Header label only; the real per-prompt budget is seq_len - prompt_len when
    # --max-new-tokens is unset.
    mnt_desc = (
        "seq_len - prompt_len" if args.max_new_tokens is None else str(args.max_new_tokens)
    )

    compiled_path = args.compiled_path or (
        "/home/ubuntu/models/qwen36_a3b_fused_selective"
        if args.mtp_spec_decode
        else "/home/ubuntu/models/qwen36_a3b_traced"
    )

    inf_config = build_inference_config(
        args.model_path, args.tp_degree, args.seq_len,
        spec_decode=args.mtp_spec_decode,
        tkg_attention_kernel=args.tkg_attention_kernel,
    )
    if args.tkg_attention_kernel:
        # Hard-fail if the flag didn't reach the configs, so we never silently
        # benchmark the PyTorch fallback while believing we tested the kernel.
        assert getattr(inf_config, "use_tkg_attention_kernel", False), (
            "use_tkg_attention_kernel did not propagate to the target config"
        )
        if inf_config.fused_spec_config is not None:
            assert getattr(
                inf_config.fused_spec_config.draft_config,
                "use_tkg_attention_kernel",
                False,
            ), "use_tkg_attention_kernel did not propagate to the draft (MTP) config"
        print("[assert] TKG attention kernel ENABLED on target + draft configs")
    maybe_compile(args.model_path, compiled_path, inf_config)
    model = load_model(compiled_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Fused spec decode dispatches through the same generate() path (NxDI's
    # _fused_assisted_decoding); only the build differs.
    mode = "FUSED SPEC DECODE" if args.mtp_spec_decode else "GREEDY"
    if args.tkg_attention_kernel:
        mode += " + TKG-KERNEL"
    print("\n" + "=" * 70)
    print(f"GENERATING (max_new_tokens={mnt_desc}, {mode})")
    print("=" * 70)
    for prompt in args.prompts:
        token_ids, _ = generate(model, tokenizer, prompt, args.max_new_tokens)
        new_ids = token_ids[len(tokenizer(prompt).input_ids):]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
        print(f"\n{prompt}")
        print(response)

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
    tokens_per_round = report["e2e_model"].get("tokens_per_round")
    if tokens_per_round is not None:
        print(
            f"\nfused speculation: {tokens_per_round:.2f} tokens/round "
            f"(spec_len={model.neuron_config.speculation_length}; 1.0 = no acceptance)"
        )
    print("\n(full report)")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
