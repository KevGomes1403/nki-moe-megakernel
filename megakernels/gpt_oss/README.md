# GPT-OSS-20B Megakernel for Trainium

A full-model NKI megakernel for GPT-OSS-20B decode inference on AWS Trainium 3. All 24 decoder layers (alternating sliding-window / full attention) run in a single NKI jit invocation — no HBM round-trips between layers.

## Results

End-to-end on Trainium 3 (`trn3pd98.3xlarge`, TP=8, LNC=1, bf16):

### bs=1 — megakernel wins across the full single-stream context range

| seq_len | XLA TKG p50 | **Megakernel TKG p50** | Δ | XLA tok/s | **Megakernel tok/s** |
|---:|---:|---:|---:|---:|---:|
| 640  | 8.62 ms | **5.88 ms** | **−32%** | 110.9 | **170.4** |
| 2048 | 5.50 ms | **4.92 ms** | **−11%** | 139.4 | **156.5** |
| 4096 | 5.54 ms | **4.85 ms** | **−12%** | 168.2 | **186.5** |
| 8192 | 5.78 ms | **4.98 ms** | **−14%** | 173.4 | **195.8** |

Per-token latency is **flat across the full 640→8192 context window**: at bs=1 the megakernel sustains ~5 ms/token (≈200 tok/s) regardless of KV cache size. Long-context interactive decode does not degrade.

### Where the megakernel does not win — batch sweep at seq=640

| batch | XLA TKG p50 | Megakernel TKG p50 | verdict |
|---:|---:|---:|---|
| 1 | 8.62 ms | **5.88 ms** | mega **−32%** |
| 4 | 10.44 ms | 10.53 ms | tied |
| 8 | **15.53 ms** | 17.65 ms | mega **+14% slower** |

The committed kernel is a **bs=1 single-stream / single-user / agentic-decode** optimization. At bs ≥ 4 the selective-expert MoE primitive's SBUF working set grows linearly with batch (55% partition demand at bs=1 → 307% at bs=8), and the resulting spill plus the kernel's per-token-per-expert matmul shape (`[1, H] @ [H, I]`, partition utilization 1/128) make it slower than the torch-blockwise XLA path at bs=8. See `study_results/megakernel_study.md` for the full characterization.

**Caveat on the XLA baseline.** This hardware has only 8 LNC=1 cores, which is not enough for TP=8 LNC=2 (= 16 cores). The library's optimized MoE NKI kernels (`moe_block_tkg`, `expert_mlp`, etc.) hardcode LNC=2, so to run XLA at TP=8 here we disabled them and use the torch-blockwise MoE path. The "XLA" baseline in this table is therefore no-NKI HLO/torch — slower than what production XLA would deliver on a ≥16-core trn3. The mega-vs-XLA gap in this table is an **upper bound** on mega's win vs a real LNC=2 XLA deployment.

## Design

The megakernel fuses the entire 24-layer GPT-OSS-20B decoder stack (input layernorm through residual add) into two subkernels per layer, chained across all layers with the residual hidden state kept in SBUF throughout.

**Attention subkernel** — RMSNorm → fused QKV → RoPE → KV cache update (in-place) → attention (sliding-window / full alternation) → output projection → AllReduce.

**MoE subkernel** — RMSNorm → router top-k (E=32, K=4) → selective expert weight loading (with hoisted weight DMA + 2-expert prefetch ring for single-stream decode) → gate/up/down expert GEMMs → affinity-weighted sum → AllReduce.

TP=8 is used throughout. KV heads are sharded across the 8 LNC=1 cores. `SbufManager` handles on-chip memory allocation; weight double-buffering across layers eliminates the per-layer HBM round-trip cost present in XLA.

## MXFP4 expert weights

The megakernel also supports gpt-oss-20b's native **MXFP4** expert weights (the format the model ships in) via `--mxfp4`. Selected by `is_mxfp4_compute=True`, this swaps the bf16 MoE primitive for the MX selective-expert kernel and routes per-layer expert weights through `nc_matmul_mx` / `quantize_mx` with block-size-32 scales.

Two NKI primitives under `nki_kernels/moe/` back this path:

- `selective_expert_mx_impl.py` — selective-expert MoE TKG with MXFP4 gate/up + down projections (vendored from `nkilib`, with a `name_prefix` kwarg added so 24 per-layer instantiations don't collide on NKI's duplicate-op-name checker).
- `down_projection_mx_shard_H.py` — H-sharded MXFP4 down projection sub-kernel used by the above.

**Caveat — HBM round-trip per layer.** The current MX MoE kernel asserts `output_in_sbuf=False` and requires its hidden input in HBM (in the model's shuffled-H layout). The megakernel therefore inserts a per-layer `dma_copy` of the MoE input out to `shared_hbm` and the MoE output back into SBUF on either side of the MX call (see `transformer_gpt_oss.py:434-470`). The bf16 path, by contrast, keeps the MoE input/output entirely in SBUF. This round-trip is the dominant overhead of the MX path today and is the next thing to remove — it would require teaching the MX selective-expert kernel to accept an SBUF hidden tensor and emit an SBUF output.

### Running MXFP4

```bash
# Megakernel with native MXFP4 weights
python main.py --model gpt_oss --mode generate --enable-nki --mxfp4 \
  --model-path ~/models/gpt-oss-20b \
  --compiled-model-path ~/models/gpt-oss-20b/traced_nki_mx_model \
  --prompt "What is the capital of France?"

# Benchmark
python main.py --model gpt_oss --mode evaluate-single --enable-nki --mxfp4 \
  --model-path ~/models/gpt-oss-20b \
  --compiled-model-path ~/models/gpt-oss-20b/traced_nki_mx_model
```

Notes:

- `--mxfp4` forces `is_full_model_shuffled=True` (the residual layout the MX MoE kernel expects) and routes CTE through the NKI shard-on-intermediate blockwise path — NxDI's torch blockwise dequant does not support gpt-oss's swizzled MX weight layout.
- Use a **separate compile cache directory** from the bf16 megakernel (e.g. `traced_nki_mx_model` vs `traced_nki_model`); the NEFFs are not interchangeable. Clear the global cache when switching paths:

  ```bash
  rm -rf ~/models/gpt-oss-20b/traced_nki_mx_model /var/tmp/neuron-compile-cache/*
  ```

## Setup

Tested on Trainium 3 (`trn3pd98.3xlarge`) with AWS Neuron SDK v2.27.

```bash
# 1. Activate the Neuron venv
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# 2. Clone and enter the repo
git clone https://github.com/KevGomes1403/nki-moe && cd nki-moe

# 3. Download the model weights
pip install huggingface_hub[cli]
huggingface-cli download openai/gpt-oss-20b --local-dir ~/models/gpt-oss-20b
```

## Running

```bash
# Baseline (XLA, no NKI MoE kernels — see "Caveat" above)
python main.py --model gpt_oss --mode generate \
  --model-path ~/models/gpt-oss-20b \
  --compiled-model-path ~/models/gpt-oss-20b/traced_model \
  --prompt "What is the capital of France?"

# Megakernel
python main.py --model gpt_oss --mode generate --enable-nki \
  --model-path ~/models/gpt-oss-20b \
  --compiled-model-path ~/models/gpt-oss-20b/traced_nki_model \
  --prompt "What is the capital of France?"

# Benchmark
python main.py --model gpt_oss --mode evaluate-single --enable-nki \
  --model-path ~/models/gpt-oss-20b \
  --compiled-model-path ~/models/gpt-oss-20b/traced_nki_model
```

When switching between baseline and megakernel, clear the compile cache:

```bash
rm -rf ~/models/gpt-oss-20b/traced_nki_model /var/tmp/neuron-compile-cache/*
```

## Toggles

The MoE primitives expose two env vars (read at module import, NEFF recompile required to switch):

- `NKI_MOE_ENABLE_FUSION=1` — enable the fused gate+up dma_copy (halves descriptor count, ~13 µs win at H=3072, I=384, K=4, bs=1).
- `NKI_MOE_LEGACY_WEIGHT_LOAD=1` — revert to upstream nkilib's per-HTile weight DMA (no exp 3/5/6/7 hoist/prefetch). Caps `selective_expert_moe_tkg` SBUF demand at ~19% regardless of batch. Slower at bs ≥ 4 — used for A/B testing the hoist's contribution to the SBUF spill, not a production setting.

## Repository Layout

```
gpt_oss.py                    # Baseline XLA model subclass
gpt_oss_with_megakernel.py    # Megakernel model (--enable-nki)
transformer_gpt_oss.py        # Top-level 24-layer megakernel
study_results/                # Seq/batch sweep data + the legacy-toggle ablation
  megakernel_study.md         # Full characterization across seq_len, batch, SBUF
  study_driver.py             # Single-cell compile + bench driver
```

Shared NKI primitives live in `nki_kernels/{attention,moe,norm}/` (vendored from nkilib; see `nki_kernels/_vendor_meta.md`).
