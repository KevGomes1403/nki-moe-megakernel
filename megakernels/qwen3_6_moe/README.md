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

A multi-token-prediction (MTP) head (one extra decoder layer over the concatenated
last hidden and next-token embedding) is used for self-speculative decoding via
NxDI's fused EAGLE speculation: the MTP head is the draft, the backbone verifies,
and the draft-verify-accept loop runs in one fused on-device graph
(`speculation_length` 2, k=1). The plain non-speculative build keeps it disabled
(`mtp_num_hidden_layers` 0).

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

The driver compiles the model on the first run if the compiled directory has no
`model.pt`, then loads it, generates greedily for a set of prompts, and runs the
on-device benchmark. The primary mode is fused MTP speculative decoding:

```bash
python -m models.qwen3_6_moe.inference_qwen36_a3b \
  --model-path /home/ubuntu/models/Qwen3.6-35B-A3B \
  --mtp-spec-decode
```

This defaults to the prebuilt selective-decode directory
`/home/ubuntu/models/qwen36_a3b_fused_selective`. Omit `--mtp-spec-decode` for the
plain greedy build (defaults to `/home/ubuntu/models/qwen36_a3b_traced`).

Expected behavior:
- First run: compilation takes roughly 10-20 minutes (plus presharding, about 65 GB,
  unless `A3B_SKIP_SHARD=1` reuses an existing `weights/`).
- Subsequent runs: weight loading takes about 6 to 9 minutes, then generation.

Useful flags and variables:
- `--prompts "Prompt one" "Prompt two"`: override the built-in sanity prompts.
- `--max-new-tokens N`: tokens per prompt (default fills the window).
- `--skip-benchmark`: sanity generations only.
- `--num-runs N` / `A3B_BENCH_RUNS`: timed benchmark runs (default 10).
- `A3B_SEQ_LEN`: compiled sequence length (default 128).
- `A3B_TP_DEGREE`: tensor parallel degree (default 4).
- `A3B_SKIP_SHARD=1`: trace graphs only; reuse an existing `weights/` dir.

## Performance

Measured with the on-device benchmark (`--mtp-spec-decode`, prompt "The capital of
France is", 123 new tokens filling the 128-token window, 10 timed runs after one
warm-up, greedy):

| Metric | Value |
| --- | --- |
| End-to-end p50 (123 new tokens) | 990 ms (129 tok/s) |
| Prefill (context encoding) p50 | 301 ms |
| Fused speculation round p50 | 9.4 ms |
| Decode throughput | 186 tok/s |
| Acceptance | 1.76 tokens/round (spec_len 2) |

## Implementation notes

DeltaNet prefill kernel. The default prefill path uses the chunked-step NKI kernel
(`nki_kernels/deltanet/prefill/chunked_step.py`), which forms the within-chunk decay
as `exp(gc[i] - gc[j])` and is numerically stable. The fused kernel
(`nki_kernels/deltanet/prefill/chunked_fused.py`) uses the split form `exp(gc[i]) * exp(-gc[j])`,
which overflows float32 for this checkpoint's gating magnitude and produces NaN
logits. Do not select the fused path for this model. Token generation (decode) uses a
single-step recurrence that is stable by construction.

MoE execution. Prefill routes through `forward_all_experts` (128 tokens x top-8 touches
4x the expert count, so all-experts is optimal). Decode uses NxDI's default selective
loading: the fused decode graph gathers only the routed experts (~16/256 slices per layer
per round), about 3x faster per round than streaming all experts, with token-identical
output. The blockwise kernel is not available in this SDK build.

Storage. Each presharded TP=4 build is about 65 GB. Compile into a single directory
and reuse it. New graph variants over identical weights can skip resharding: pass
`A3B_SKIP_SHARD=1` and symlink an existing `weights/` into the new compiled directory
(see `/home/ubuntu/models/qwen36_a3b_fused_selective`).

The NKI kernels in `nki_kernels/` are taken unmodified from a Neuron Distributed
Inference (NxDI) pull request authored by an AWS engineer
([aws-neuron/neuronx-distributed-inference#164](https://github.com/aws-neuron/neuronx-distributed-inference/pull/164)).
They are treated as validated code and are not edited.

## Files

- `modeling_qwen36_a3b.py`: configuration, DeltaNet, GQA, MoE block, MTP head, hybrid
  cache manager, decoder layer, model, causal LM wrapper, fused-speculation classes,
  and the weight converter.
- `nki_kernels/`: DeltaNet NKI kernels organized by stage and regime under
  `deltanet/` (`components/`, `decode/`, `prefill/`); see `nki_kernels/README.md`
  for the full map. Consumers import entrypoints from the `nki_kernels` package root.
- `inference_qwen36_a3b.py`: end-to-end driver (compile, load, generate, benchmark).
- `tests/`: CPU weight-conversion tests, hardware block tests for the MoE and DeltaNet
  blocks, the MTP state-rule oracle (`test_mtp_state_rule.py`), and kernel/decay
  diagnostics (`probe_kernel_padding.py`, `probe_real_decay.py`,
  `scan_weights_finite.py`).

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
