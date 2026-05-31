"""End-to-end inference driver for Qwen3.6-35B-A3B on Trainium.

Default config: trn3, TP=4, LNC=2, bf16, seq_len=128, greedy sampling.
Compiles once into --compiled-path, then loads and generates for a few
short prompts to validate that the port produces sensible tokens.

Usage:
    python inference_qwen36_a3b.py \\
        --model-path /home/ubuntu/models/Qwen3.6-35B-A3B \\
        --compiled-path /home/ubuntu/models/qwen36_a3b_traced \\
        --max-new-tokens 32

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

    on_device_sampling = (
        None if os.environ.get("A3B_RETURN_LOGITS") == "1"
        else OnDeviceSamplingConfig(top_k=1)
    )

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

    # Disable MTP for this first end-to-end pass: the draft head is not
    # wired into the traced forward yet, so leaving it on just adds an extra
    # decoder layer's worth of compile time + weights for no inference win.
    config_dict["mtp_num_hidden_layers"] = 0

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


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple:
    """Greedy generate for a single prompt; return (token_ids, decoded_text, latency_s)."""
    gen_config = GenerationConfig(
        do_sample=False,  # greedy: matches OnDeviceSamplingConfig(top_k=1)
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_config.transformers_version = transformers.__version__

    adapter = HuggingFaceGenerationAdapter(model)
    adapter.generation_config.transformers_version = transformers.__version__

    inputs = tokenizer(prompt, padding=True, return_tensors="pt")

    t0 = time.time()
    out_ids = adapter.generate(
        inputs.input_ids,
        generation_config=gen_config,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
    )
    elapsed = time.time() - t0

    full_ids = out_ids[0].tolist()
    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    return full_ids, text, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="HF checkpoint dir")
    parser.add_argument(
        "--compiled-path",
        default=os.environ.get("A3B_COMPILED_PATH", "/home/ubuntu/models/qwen36_a3b_traced"),
        help="Where to write / read compiled Neuron artifacts",
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
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=DEFAULT_PROMPTS,
        help="Prompts to generate (default: a built-in sanity set)",
    )
    args = parser.parse_args()

    inf_config = build_inference_config(
        args.model_path, args.tp_degree, args.seq_len,
    )
    maybe_compile(args.model_path, args.compiled_path, inf_config)
    model = load_model(args.compiled_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n" + "=" * 70)
    print(f"GENERATING ({args.max_new_tokens} new tokens, greedy)")
    print("=" * 70)
    for prompt in args.prompts:
        token_ids, text, elapsed = generate(
            model, tokenizer, prompt, args.max_new_tokens,
        )
        prompt_ids = tokenizer(prompt).input_ids
        new_ids = token_ids[len(prompt_ids):]
        n_new = len(new_ids)
        tps = n_new / elapsed if elapsed > 0 else 0.0
        print(f"\n--- prompt ---\n{prompt}")
        print(f"--- completion ({n_new} tok, {elapsed:.2f}s, {tps:.1f} tok/s) ---")
        print(f"new_ids: {new_ids}")
        print(f"new_tokens: {tokenizer.convert_ids_to_tokens(new_ids)}")
        print(f"decoded (raw): {tokenizer.decode(new_ids, skip_special_tokens=False)!r}")
        print(f"decoded (clean): {tokenizer.decode(new_ids, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
