# NKI Megakernels for MoE Models on Trainium

A collection of full-model NKI megakernels for MoE inference on AWS Trainium. Each model's entire decoder stack runs in a single NKI jit invocation — residual stays in Sbuf, KV caches are scattered in place, and there are no HBM round-trips between layers.

## Models

| Model | Path | Hardware | Status |
|---|---|---|---|
| Qwen3-30B-A3B | [`megakernels/qwen3_moe/`](megakernels/qwen3_moe/) | trn2, trn3 (TP=4, LNC=2) | 1.76× over XLA baseline ([results](megakernels/qwen3_moe/README.md#results)) |
| GPT-OSS-20B | [`megakernels/gpt_oss/`](megakernels/gpt_oss/) | trn3 (TP=8, LNC=1) | In progress |

Each model lives in its own folder under `megakernels/` and has its own README with results, design notes, and run instructions.

## Layout

```
megakernels/<model>/
  <model>.py                    # Baseline XLA NxDI subclass
  <model>_with_megakernel.py    # NKI-enabled NxDI subclass (--enable-nki)
  transformer_<model>.py        # The multilayer megakernel itself
  ...                           # Any model-specific NKI subkernels

nki_kernels/                    # Shared NKI primitives
  attention/                    # attention_block_tkg + helpers
  moe/                          # moe_tkg, router_topk
  norm/                         # rmsnorm_tkg

tests/<model>/                  # Per-model unit tests for kernels
main.py                         # CLI: generate / validate / benchmark
```

The shared primitives in `nki_kernels/` are vendored from `nkilib` so that per-layer invocations don't produce duplicate op names — see [`nki_kernels/_vendor_meta.md`](nki_kernels/_vendor_meta.md).

## Setup

Tested with AWS Neuron SDK v2.27.

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
git clone https://github.com/KevGomes1403/nki-moe && cd nki-moe
```

## Running

`main.py` dispatches by `--model` and toggles the megakernel with `--enable-nki`:

```bash
# Baseline (XLA)
python main.py --model qwen3_moe --mode generate \
  --model-path ~/qwen-30b-a3b/hf_model \
  --compiled-model-path ~/qwen-30b-a3b/traced_model \
  --prompt "What is the capital of France?"

# Megakernel
python main.py --model qwen3_moe --mode generate --enable-nki \
  --model-path ~/qwen-30b-a3b/hf_model \
  --compiled-model-path ~/qwen-30b-a3b/traced_nki_model \
  --prompt "What is the capital of France?"

# Benchmark
python main.py --model qwen3_moe --mode evaluate-single --enable-nki \
  --model-path ~/qwen-30b-a3b/hf_model \
  --compiled-model-path ~/qwen-30b-a3b/traced_nki_model
```

Swap `--model qwen3_moe` for `--model gpt_oss` to run the other model. See each model's README for weight download instructions and supported flags.

When switching between baseline and megakernel, clear the compile cache:

```bash
rm -rf <compiled-model-path> /var/tmp/neuron-compile-cache/*
```

## Adding a new model

1. Create `megakernels/<model>/` with `<model>.py`, `<model>_with_megakernel.py`, and `transformer_<model>.py`.
2. Reuse `nki_kernels/{attention,moe,norm}/` where possible; put model-specific subkernels alongside the megakernel.
3. Register the model in `main.py`'s `_MODEL_REGISTRY`.
4. Add tests under `tests/<model>/`.

## License

See [LICENSE](LICENSE).
