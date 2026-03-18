# NxDI Compute-Communication Overlap Config

## Background

In tensor-parallel inference, each layer does:
1. **Compute** — matrix multiply on each NeuronCore
2. **Communicate** — all-gather or reduce-scatter to sync across NeuronCores

By default these are sequential. CC overlap pipelines them: tile N+1 computes while tile N communicates, hiding latency.

The three FSDP env vars (`NEURON_NXD_FSDP_CC_MULTISTREAM`, `NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT`, `NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT`) are **training-side** and do not exist in NxDI inference.

---

## Compiler Flags

Set in `get_compiler_args()` or applied automatically via `model_wrapper.py`:

```
--enable-ccop-compute-overlap         # enables the overlap optimization
--cc-pipeline-tiling-factor=N         # matches cc_pipeline_tiling_factor in NeuronConfig
--vectorize-strided-dma               # DMA optimization (skipped for seqlen >= 128k on trn2)
```

---

## NeuronConfig Fields

### `cc_pipeline_tiling_factor` (default: `2`)

Splits each layer's weight matrix into N tiles and pipelines compute+communicate across them:

```
factor=1:  [compute all] → [communicate all]
factor=2:  [compute tile1] → [compute tile2 + communicate tile1] → [communicate tile2]
```

- Higher factor → more latency hidden, but smaller tiles add overhead
- **TKG is always forced to 1** — single-token matmuls are too small to benefit from tiling
- Must match `--cc-pipeline-tiling-factor` in compiler args

### `seq_len_threshold_for_cc_tiling` (default: `16384`)

CC tiling only activates for CTE when `seqlen >= threshold`. Below this, the kernel uses a flat grid (equivalent to factor=1).

Short sequences have small enough matmuls that tiling overhead exceeds savings. At 16k+ tokens the tradeoff flips.

### `disable_numeric_cc_token` (default: `False`)

The Neuron runtime uses a numeric scalar "CC token" to enforce ordering between compute and collective ops. In Qwen3 MoE, XLA inserts extra multiply/add ops around this token that corrupt output numerics.

Setting `True` sets `DISABLE_NUMERIC_CC_TOKEN=1`, switching to a non-numeric ordering mechanism. Only needed for models that hit this XLA bug (Qwen3 MoE hardcodes it to `True`).

### `ep_dispatch_cc_option` (default: `"AR_AG"`)

For Expert Parallelism: controls the collective communication pattern used to route token hidden states to the correct expert core and gather results back.

| Option | Pattern | Constraint |
|--------|---------|------------|
| `AR_AG` | AllReduce (TP group) → AllGather (DP group) | None; works for all configs |
| `RS_AG` | ReduceScatter (TP group) → AllGather (global) | Requires `tp_degree <= tkg_batch_size` |
| `AG_AR` | AllGather (DP group) → AllReduce (TP group) | Different overlap profile |

Each option has different bandwidth and overlap characteristics depending on hardware topology and batch size.

### `switch_cc` (default: `False`)

Controls the NeuronCore mesh topology used for collective communication routing.

- `False`: default trn2 NeuronLink mesh wiring
- `True`: alternative 8×8 mesh mapped through a switch fabric, which can reduce congestion at large TP degrees

Only relevant at TP=32+. Changes which physical NeuronCores talk to which during all-gather/reduce-scatter.

---

## NKI Kernel Grid Integration

In NKI kernels (attention, MLP), the grid must be wrapped with `CCPipeline` for tiling to take effect:

```python
from neuronxcc.nki.typing import CCPipeline, nc

# CTE kernel (long seqlen):
if seqlen > neuron_config.seq_len_threshold_for_cc_tiling:
    grid = (CCPipeline(neuron_config.cc_pipeline_tiling_factor) * nc(logical_nc_config),)
else:
    grid = (nc(logical_nc_config),)

# TKG kernel: always flat grid (factor forced to 1)
grid = (nc(logical_nc_config),)
```

---

## Quick Reference

| Field | Default | When to change |
|-------|---------|----------------|
| `cc_pipeline_tiling_factor` | `2` | Increase to 4 for very long seqlen CTE; leave 1 for TKG |
| `seq_len_threshold_for_cc_tiling` | `16384` | Lower if you see benefit at shorter seqlens |
| `disable_numeric_cc_token` | `False` | Set `True` for MoE models with XLA numeric CC bug |
| `ep_dispatch_cc_option` | `"AR_AG"` | Try `RS_AG` for MoE if `tp_degree <= batch_size` |
| `switch_cc` | `False` | Set `True` for large TP on switch-fabric hardware |
