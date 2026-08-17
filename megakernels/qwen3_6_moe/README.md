# Qwen3.6-35B-A3B on AWS Trainium

Port of `Qwen/Qwen3.6-35B-A3B` (35B total / ~3B active MoE) to trn2 via NxDI, with
the entire self-speculative decode round fused into a single NKI megakernel.

Architecture in one line: 40 layers in a [DeltaNet x3, GQA] x10 hybrid pattern, each
with a 256-expert top-8 MoE FFN plus a sigmoid-gated shared expert; hidden 2048,
vocab 248320, bf16, tensor parallel 4, batch 1, two NeuronCores fused per logical
core (LNC=2). A multi-token-prediction (MTP) head drafts one extra token per step,
verified by the backbone through NxDI's fused EAGLE speculation.

## Benchmarks

Speculation megakernel (`--mtp-spec-decode --speculation-megakernel`), greedy, new
tokens filling the window, 10 timed runs after warm-up. Decode counts the speculation
graph alone (tokens per round divided by round latency); end-to-end includes prefill
and the host generation loop.

| seq_len | round p50 | tokens/round | decode | end-to-end |
| --- | --- | --- | --- | --- |
| 128 | 9.62 ms | 1.78 | 185 tok/s | 142 tok/s |
| 512 | 9.75 ms | 1.98 | 203 tok/s | 160 tok/s |
| 1024 | 9.76 ms | 1.99 | 204 tok/s | 163 tok/s |
| 2048 | 9.84 ms | 1.91 | 194 tok/s | 157 tok/s |

Round latency is nearly flat in sequence length: the only work that grows with the
window is GQA attention over the KV cache, and only 10 of 40 layers use it.
Tokens/round is prompt-dependent; small differences between rows are noise.

### Against the XLA baseline

Same fused-speculation structure and identical blockwise prefill; the baseline runs
NxDI's stock XLA/torch graphs (PyTorch DeltaNet recurrence, selective-loading MoE)
with every NKI kernel disabled.

| seq_len | XLA round p50 | megakernel round p50 | XLA decode | megakernel decode |
| --- | --- | --- | --- | --- |
| 1024 | 14.76 ms | **9.76 ms (-34%)** | 127 tok/s | **204 tok/s (1.60x)** |
| 2048 | 14.96 ms | **9.84 ms (-34%)** | 131 tok/s | **194 tok/s (1.48x)** |

The XLA round is flat in sequence length too, so the megakernel's win is fusion —
launch overhead and HBM traffic removed — not attention scaling. The decode ratio
moves with per-run acceptance noise; the stable per-round latency gap is -34%.

### Layerwise NKI

The same GQA, DeltaNet and MoE block kernels launched separately (82 NKI calls per
round), with NxDI holding the collectives, residual adds, KV scatter and accept/reject.
Embedding, eh_proj and LM head stay XLA. Run with `--tkg-attention-kernel
--use-moe-layer-kernel` and the megakernel flags off.

| seq_len | round p50 | tokens/round | decode | end-to-end |
| --- | --- | --- | --- | --- |
| 1024 | 10.62 ms | 1.94 | 182 tok/s | 149 tok/s |
| 2048 | 10.54 ms | 1.98 | 188 tok/s | 153 tok/s |

Device-time profiles at 1024: megakernel 9.051 ms, layerwise 9.808 ms.

## What's novel here

### Speculation megakernel

One NKI launch runs the entire speculative-decoding round — the draft step, the
verify pass (all 40 layers + LM head + greedy argmax) and the MTP replay — spanning
both logical cores (`nki_kernels/megakernel/qwen36_round_megakernel.py`).

- **The residual never leaves SBUF**, from embedding to token id, across all 40
  layers and all three stages.
- **Collectives run in-kernel**: TP all-reduces, cross-core gathers, and the
  vocab-argmax reduction.
- **Caches update in place** — KV and MTP, no host-side scatter.
- **Stage seams carry zero HBM traffic**: the drafted token id reaches the verifier
  through an on-chip buffer, the verify hidden state reaches the replay through one
  on-chip transpose, and because all three stages share a launch, the replay can
  read cache entries the draft stage just wrote.
- Verify-only and draft-only megakernels exist as standalone building blocks.

### Fused DeltaNet decode kernel

The linear-attention layer as one launch (`nki_kernels/deltanet/`).

- Input projection → causal convolution → gated delta-rule recurrence → gated
  RMSNorm → output projection, value-heads sharded across the two logical cores.
- When verifying speculated tokens it emits **per-token candidate states**, so the
  host commits the recurrent and convolution state of whichever token was accepted.

### Fused GQA decode layer

The full-attention layer with every intermediate in SBUF (`nki_kernels/gqa/`).

- QKV and sigmoid-gate projections, q/k RMSNorm, partial rotary embedding,
  attention over the HBM KV cache, gate apply and output projection.
- The 256-wide head dimension runs as two 128-partition tiles.

## Implementation notes

- Prefill MoE must use the blockwise kernel above seq_len 512. The all-experts path
  materializes an experts x tokens x hidden intermediate per layer (1 MB per token
  per layer at bf16), which fails compilation on host RAM at 1024 and on device HBM
  at 4096 — the KV cache is negligible by comparison (80 MB at 4096). The driver
  configures this by default: `block_size` 128 (with 256 experts the worst-case
  block count has a floor of 255, so small blocks waste far less),
  `use_shard_on_block_dynamic_while`, and `PING_PONG` block sharding. This SDK build
  lacks the default shard-on-hidden blockwise kernel. Blockwise output has not yet
  been diffed against the all-experts path.
- DeltaNet prefill uses the chunked-step kernel; the fused variant overflows fp32
  for this checkpoint's gating magnitude (NaN logits) — do not select it.
- Decode MoE uses NxDI selective loading (~16/256 expert slices per layer per round),
  ~3x faster than streaming all experts, token-identical output.
- RMSNorm uses the `(1 + weight)` convention; the weight converter adds 1.0.
- Presharded TP=4 weights are ~65 GB; reuse a `weights/` dir across graph variants
  with `A3B_SKIP_SHARD=1` plus a symlink.

## Run

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd /home/ubuntu/nki-moe
python -m megakernels.qwen3_6_moe.inference_qwen36_a3b \
  --model-path /home/ubuntu/models/Qwen3.6-35B-A3B \
  --mtp-spec-decode --speculation-megakernel
```

First compile takes 10-20 minutes; later runs load in ~6-9 minutes. Key flags:
`--seq-len`, `--max-new-tokens`, `--skip-benchmark`, `--num-runs`. Tests live in
`tests/` (CPU weight-conversion via pytest; MoE/DeltaNet block tests need the
device).

## Files

- `modeling_qwen36_a3b.py` — config, DeltaNet/GQA/MoE blocks, MTP head, hybrid cache,
  fused-speculation classes, weight converter.
- `inference_qwen36_a3b.py` — compile / load / generate / benchmark driver.
- `../../nki_kernels/` — the NKI kernels (`megakernel/`, `deltanet/`, `gqa/`, `moe/`,
  `embed/`, `lm_head/`).
