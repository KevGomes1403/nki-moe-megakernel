# Qwen3.6-35B-A3B on AWS Trainium

A self-contained port of the text stack of `Qwen/Qwen3.6-35B-A3B` to AWS Trainium
(trn3) using NxD Inference (NxDI). The model has roughly 35B total parameters with
about 3B active per token (the "A3B" mixture-of-experts configuration). This
directory contains the modeling code, the DeltaNet NKI kernels, an inference driver,
and diagnostics.

## Model architecture

The decoder has 40 layers arranged in a hybrid pattern: three Gated DeltaNet (linear
attention) layers followed by one Grouped-Query Attention (full attention) layer,
repeated 10 times. This yields 30 DeltaNet layers and 10 GQA layers. Every layer uses
a Mixture-of-Experts feed-forward block.

Global settings:
- `hidden_size` 2048, `vocab_size` 248320, untied input and output embeddings.
- RMSNorm with `eps` 1e-6 and the `(1 + weight)` convention (weights are stored as
  deviations from one; the weight converter adds 1.0 to the standard norms).

Grouped-Query Attention (full attention layers):
- 16 query heads, 2 key/value heads (8:1 grouping), `head_dim` 256.
- Per-head q/k RMSNorm applied before rotary embedding.
- Partial rotary embedding on the first 64 of 256 head dimensions
  (`partial_rotary_factor` 0.25), using interleaved multimodal RoPE
  (`mrope_section` [11, 11, 10]) with `rope_theta` 1e7. For text-only input this
  reduces to standard RoPE on the first 64 dimensions.
- A sigmoid output gate is applied to the attention output before the output
  projection.

Gated DeltaNet (linear attention layers):
- 32 value heads, 16 key heads, key and value head dimension 128.
- Causal depthwise 1D convolution (kernel size 4) over the Q, K, V stream.
- Gated delta-rule recurrence with a gated output RMSNorm.

Mixture-of-Experts feed-forward (every layer):
- 256 routed experts, top-8 routing. Routing softmax is taken over all experts, the
  top-8 are selected, and their weights are renormalized to sum to one.
- One always-on shared expert whose output is scaled by a per-token sigmoid gate.
- `moe_intermediate_size` and `shared_expert_intermediate_size` are both 512.

A multi-token-prediction (MTP) head exists in the checkpoint but is disabled in this
deployment (`mtp_num_hidden_layers` is set to 0).

## Hardware and parallelism

- Target platform: trn3. Validated on a `trn3pd98.3xlarge` instance, which exposes one
  Neuron device with 4 NeuronCores.
- Logical NeuronCore configuration (LNC): 2.
- Tensor parallel degree (TP): 4. Expert parallel degree (EP): 1.
- Data type: bfloat16. Default sequence length: 128. Batch size: 1.

A TP=4 build occupies all 4 NeuronCores, so only one full-model job can run on the
device at a time. To run isolated single-core kernel work in parallel, set
`NEURON_RT_VISIBLE_CORES` to distinct core IDs (for example `0` and `1`).

## Prerequisites

1. Activate the Neuron inference virtual environment:

   ```bash
   source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
   ```

2. The Hugging Face checkpoint must be present at
   `/home/ubuntu/models/Qwen3.6-35B-A3B`.

3. Run all commands from the repository root so that `models.qwen3_6_moe.*` imports
   resolve:

   ```bash
   cd /home/ubuntu/trainium-model-translation
   ```

## Running inference

The driver compiles the model on the first run if the compiled directory is empty,
then loads it and generates greedily for a set of prompts. If the directory already
contains `model.pt`, the compile step is skipped.

```bash
A3B_RETURN_LOGITS=1 python -m models.qwen3_6_moe.inference_qwen36_a3b \
  --model-path /home/ubuntu/models/Qwen3.6-35B-A3B \
  --compiled-path /home/ubuntu/models/qwen36_a3b_traced \
  --max-new-tokens 24
```

Expected behavior:
- First run: compilation takes roughly 20 minutes and writes a presharded checkpoint
  (about 65 GB) into the compiled directory.
- Subsequent runs: weight loading takes about 8 to 9 minutes, then generation runs at
  roughly 20 to 22 tokens per second.

Useful flags and variables:
- `--prompts "Prompt one" "Prompt two"`: override the built-in sanity prompts.
- `--max-new-tokens N`: number of tokens to generate per prompt.
- `A3B_SEQ_LEN`: compiled sequence length (default 128).
- `A3B_TP_DEGREE`: tensor parallel degree (default 4).
- `A3B_RETURN_LOGITS=1`: build with on-device sampling disabled, so the model returns
  logits and greedy selection is done on the host. This matches how the current
  compiled directory was built. Omit it to build an on-device-sampling variant, which
  forces a one-time recompile into the target directory.

Example with custom prompts:

```bash
A3B_RETURN_LOGITS=1 python -m models.qwen3_6_moe.inference_qwen36_a3b \
  --model-path /home/ubuntu/models/Qwen3.6-35B-A3B \
  --compiled-path /home/ubuntu/models/qwen36_a3b_traced \
  --prompts "The capital of France is" "Explain gravity in one sentence." \
  --max-new-tokens 32
```

## Implementation notes

DeltaNet prefill kernel. The default prefill path uses the chunked NKI kernel
(`nki_kernels/nki_deltanet_chunked.py`), which forms the within-chunk decay as
`exp(gc[i] - gc[j])` and is numerically stable. The fused kernel
(`nki_kernels/nki_deltanet_fused.py`) uses the split form `exp(gc[i]) * exp(-gc[j])`,
which overflows float32 for this checkpoint's gating magnitude and produces NaN
logits. Do not select the fused path for this model. Token generation (decode) uses a
single-step recurrence that is stable by construction.

MoE execution. The selective-loading and blockwise MoE paths are not used. Decode and
prefill both route through `forward_all_experts`, which is set up in the inference
driver and config. This avoids an out-of-bounds gather in selective loading and a
blockwise kernel that is not available in this SDK build.

Storage. Each presharded TP=4 build is about 65 GB. Compile into a single directory
and reuse it. Do not create a separate compiled directory per experiment, since
several builds will exhaust local disk.

The NKI kernels in `nki_kernels/` are taken unmodified from a Neuron Distributed
Inference (NxDI) pull request authored by an AWS engineer
([aws-neuron/neuronx-distributed-inference#164](https://github.com/aws-neuron/neuronx-distributed-inference/pull/164)).
They are treated as validated code and are not edited.

## Files

- `modeling_qwen36_a3b.py`: configuration, DeltaNet, GQA, MoE block, MTP head, hybrid
  cache manager, decoder layer, model, causal LM wrapper, and the weight converter.
- `nki_kernels/`: DeltaNet NKI kernels (`nki_deltanet.py` recurrent,
  `nki_deltanet_chunked.py` per-chunk and stable, `nki_deltanet_fused.py` fused).
- `inference_qwen36_a3b.py`: end-to-end driver (compile, load, greedy generate).
- `diag_logits.py`: prefill logits and top-k diagnostic.
- `tests/`: CPU weight-conversion tests, hardware block tests for the MoE and DeltaNet
  blocks, and kernel and decay diagnostics (`probe_kernel_padding.py`,
  `probe_real_decay.py`, `scan_weights_finite.py`, `diag_bisect.py`).

## Tests

CPU weight-conversion tests:

```bash
python -m pytest models/qwen3_6_moe/tests/test_a3b_weight_conversion.py
```

Hardware block tests (require a Neuron device, TP=4, LNC=2):

```bash
python -m models.qwen3_6_moe.tests.test_moe_block
python -m models.qwen3_6_moe.tests.test_deltanet_block
```

## Configuration reference

| Setting | Value |
| --- | --- |
| Layers | 40 (30 DeltaNet, 10 GQA), pattern [DeltaNet x3, GQA] x10 |
| Hidden size | 2048 |
| Vocabulary | 248320 |
| Attention heads | 16 query, 2 key/value, head_dim 256 |
| Partial rotary | first 64 of 256 dims, interleaved MRoPE, theta 1e7 |
| DeltaNet | 32 value heads, 16 key heads, head dim 128, conv kernel 4 |
| Experts | 256 routed, top-8, plus 1 sigmoid-gated shared expert |
| Expert intermediate | 512 (routed and shared) |
| Data type | bfloat16 |
| Tensor parallel | 4 |
| Logical NeuronCore | 2 |
| Sequence length | 128 (default) |
