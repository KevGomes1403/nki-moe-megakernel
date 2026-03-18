# MoE CTE Optimization Analysis: Qwen3-30B-A3B, TP=4, seqlen=640

**Context:** MLSys 2026 contest baseline — CTE latency ~407ms, attention ~50ms, MoE+other ~357ms.

---

## Empirical Foundation

### Per-expert token load

```
tokens/expert = seqlen × top_k / num_experts
             = 640 × 8 / 128
             = 40 tokens per expert (average)
```

### Expert weight sizes (per expert, BF16)

```
gate_up: H × 2I = 2048 × 1536 × 2 bytes = 6.29 MB
down:    I × H  = 768  × 2048 × 2 bytes = 3.14 MB
total:   9.44 MB per expert
```

### GEMM B-tile waste from block_size=512

NKI PSUM is pmax×pmax = 128×128, so B (token) dimension is tiled in 128-row chunks:

```
block_size=512 → 4 B-tiles per expert (512/128), ~40 real tokens → 3 of 4 tiles are pure padding
block_size=128 → 1 B-tile per expert (128/128), ~40 real tokens → 1 tile with 31% fill
```

### Arithmetic intensity

```
FLOPs per expert (40 tokens):
  gate:  40 × 2048 × 768 × 2 = 126 MFLOPS
  up:    40 × 2048 × 768 × 2 = 126 MFLOPS
  down:  40 × 768  × 2048 × 2 = 126 MFLOPS
  total: 378 MFLOPS

Weight bytes per expert: 9.44 MB
Arithmetic intensity: 378 MFLOPS / 9.44 MB = 40 FLOPS/byte

trn2 ridgeline: 190 TFLOPS / 820 GB/s = 232 FLOPS/byte
```

The MoE is deeply **memory-bandwidth-bound** (40 vs 232 FLOPS/byte).

### Theoretical minimum time

With EP=4 (32 local experts per rank):

```
Weight load per layer per rank: 32 × 9.44 MB = 302 MB
At 820 GB/s: 302 MB / 820 GB/s ≈ 368 µs/layer
For 48 layers: ~17.7 ms
```

Measured MoE time: ~357ms. **Gap: ~20x from the memory-bandwidth bound.** All design decisions aim to close this gap.

---

## Design Decision 1: Reduce `block_size` 512 → 128 (config change)

**Problem:** `block_size=512` runs 4 B-tile GEMM passes per expert. With only ~40 real tokens per expert, 3 of 4 passes compute on zeros. This is 4x unnecessary GEMM work.

**Fix:**

```python
BlockwiseMatmulConfig.from_kwargs(
    block_size=128,   # was 512
    logical_nc_config=2,
    skip_dma_token=True,
    skip_dma_weight=True,
)
```

**Quantitative case:**

| block_size | B-tiles/expert | Padding fraction | GEMM tile passes (32 experts/layer) |
|---|---|---|---|
| 512 | 4 | 92% | 128 |
| 128 | 1 | 69% | 32 |

- **4x reduction in GEMM compute per layer**
- Secondary benefit: SBUF footprint drops from ~3.57 MB to ~2.07 MB per NC, freeing ~1.9 MB for double-buffering (Design Decision 2)

**Constraint check:** `block_size` must be a multiple of pmax=128. ✓

Since the kernel is memory-bandwidth-bound, the GEMM savings matter most when DMA and compute cannot overlap (current case — see DD2). Regardless, the SBUF benefit alone makes this change worthwhile.

---

## Design Decision 2: DMA-Compute Overlap for Gate+Up Weight Loading (custom kernel)

**Problem:** The existing `bwmm_shard_on_I` CTE kernel has no double-buffering for gate+up weights. From the kernel's inner loop:

```python
# Current CTE behavior — sequential DMA → compute:
for expert in range(num_local_experts):
    for h_t in range(num_h_tiles):         # sequential loop
        DMA gate_up_weight[expert, h_t]    # blocks compute
        nc_matmul(psum, gate_up_w, tokens) # blocks next DMA
```

DMA and compute are serialized. Per-expert time = DMA_time + GEMM_time.

**Fix:** Adapt the v7b TKG technique (which already exists in `kernels/moe_tkg/agents/v7b_preload_gateup_rows.py`) to the CTE kernel. Pre-load all gate+up rows for expert `e+1` while computing expert `e`'s down projection:

```python
# Target CTE double-buffered pattern:
for expert in range(num_local_experts):
    # Pre-load gate+up for NEXT expert (all h_tiles in parallel via affine_range)
    for h_t in nl.affine_range(num_h_tiles):
        gate_up_next[h_t] = DMA gate_up_weight[expert+1, h_t]

    # Compute current expert's down projection (overlaps with DMA above)
    for i_t in range(num_i_tiles):
        nc_matmul(down_psum, activation[i_t], down_weight[i_t])

    gate_up_cur, gate_up_next = gate_up_next, gate_up_cur
```

**Quantitative case:**

```
Weight DMA per expert:  9.44 MB / 820 GB/s = 11.5 µs
GEMM compute per expert (40 tokens, block_size=128):
  gate+up: 40 × 2048 × 384 × 2 / (95 TFLOPS/NC) = 0.66 µs/NC
  down:    40 × 384  × 2048 × 2 / (95 TFLOPS/NC) = 0.66 µs/NC

Sequential (current):  11.5 + 0.66 ≈ 12.2 µs/expert
Double-buffered:       max(11.5, 0.66) ≈ 11.5 µs/expert  (~6% per expert)
For 32 experts/layer:  390 µs → 368 µs
For 48 layers:         18.7 ms → 17.7 ms (approaches bandwidth bound)
```

**Why this requires DD1 first:** With `block_size=512`, the token tile alone is 2 MB per NC, leaving no room for a second weight buffer in 4 MB SBUF. With `block_size=128`, the token tile is 0.5 MB, freeing 1.5 MB for a second gate+up buffer (1.57 MB needed).

**SBUF budget with block_size=128, LNC=2 (I_local=384):**

```
Token tile:          128 × 2048 × 2 bytes = 0.50 MB
gate_up buffer A:    2048 × 384  × 2 bytes = 1.57 MB
gate_up buffer B:    2048 × 384  × 2 bytes = 1.57 MB
down weight:         384  × 2048 × 2 bytes = 1.57 MB (single buffer, serial load ok)
PSUM (128×128 fp32): 128  × 128  × 4 bytes = 0.06 MB
Total:               ~5.27 MB across 4 MB SBUF
```

Down weight double-buffering fits if gate+up buffers are released before down loads. Exact feasibility depends on SBUF allocation order in the NKI compiler — verify empirically.

---

## Design Decision 3: `use_shard_on_block_dynamic_while` (config, free for shorter seqs)

**Problem:** `skip_dma_weight=True` only skips weight DMA when a block is *entirely* padding (all -1 token indices). At seqlen=640 with 40 tokens/expert, no expert block is entirely empty — all 32 local experts get tokens. `skip_dma_weight` never fires for this seqlen.

**Fix:**

```python
BlockwiseMatmulConfig.from_kwargs(
    block_size=128,
    logical_nc_config=2,
    skip_dma_token=True,
    skip_dma_weight=True,
    use_shard_on_block_dynamic_while=True,  # add this
)
```

This enables a kernel variant that checks at runtime whether a block is non-empty before issuing any DMA or compute.

**When it matters:**

| seqlen | tokens/expert (avg) | Empty expert blocks/rank | Dynamic while skip rate |
|---|---|---|---|
| 640 | 40 | ~0% | negligible |
| 128 | 8 | ~40% | ~40% of blocks skipped |
| 32 | 2 | ~80% | ~80% of blocks skipped |

If the contest benchmarks multiple sequence-length buckets (common in LLM inference evaluation), this is a free win for shorter prompts. Zero kernel code required.

---

## Design Decision 4: Fuse RMSNorm + Router + Blockwise Kernel (custom kernel, high effort)

**Problem:** For CTE, `hidden_states` is read from HBM at least 3 separate times before the blockwise GEMM:

1. `post_attention_layernorm` (standalone XLA op)
2. Router linear: `(640, 2048) @ (2048, 128)` (separate XLA op)
3. Blockwise token gather inside the NKI kernel

Additionally, the token-to-expert mapping (one-hot creation, cumsum over (640,128) tensor, scatter/gather) runs as pure PyTorch/XLA ops generating intermediate tensors that round-trip through HBM. This happens 48 times (once per layer).

**Target architecture — fused CTE MoE kernel:**

```
hidden_states (HBM, read once)
    ↓ DMA into SBUF
    → RMSNorm (in-register, SBUF only)
    → Router linear: (640, 2048) @ (2048, 128) → logits (SBUF)
    → softmax (fp32, in SBUF) + topk → expert indices + affinities (SBUF)
    → token→expert assignment (cumsum in SBUF, no HBM round-trip)
    → blockwise GEMM with double-buffered weight loading
    → output (HBM, written once)
```

**Quantitative case:**

```
HBM reads saved per layer:
  - post_attention_layernorm input:  640 × 2048 × 2 = 2.5 MB
  - router linear input:             640 × 2048 × 2 = 2.5 MB
  Total: 5 MB/layer × 48 layers = 240 MB

At 820 GB/s: 293 µs absolute savings

Indirect savings:
  - Eliminates 48 × 3+ separate XLA kernel launches
  - Eliminates (640, 128) = 80 KB intermediate tensors × 4 (one-hot, masked
    affinities, cumsum, token-position map) × 48 layers = significant HBM traffic
  - Removes Python framework dispatch overhead per layer
```

The absolute bandwidth saving (~0.3ms) is modest, but framework overhead across 48 layers may account for a significant fraction of the 20x gap from bandwidth bound.

**Implementation notes:**
- Router weight `(2048, 128)` fits in SBUF: 2048 × 128 × 2 = 0.5 MB ✓
- Router output `(640, 128)` fits in SBUF: 640 × 128 × 2 = 0.16 MB ✓
- topk over 128 entries requires a custom sort or threshold pass in NKI — non-trivial
- `norm_topk_prob=True` requires L1 normalization after topk (8 values — trivial)
- Largest challenge: NKI topk. Consider approximating with a fixed sort network for K=8, E=128.

---

## Design Decision 5: FP32 Router Activation (code change, low effort)

**Problem:** `RouterTopK` runs the softmax in `float64` by default, then casts to BF16. FP64 arithmetic runs at ~47.5 TFLOPS on trn2 vs 190 TFLOPS for BF16/FP32 — roughly 4x slower.

```
Router FLOPs per layer: 640 × 2048 × 128 × 2 = 335 MFLOPS (linear)
                      + 640 × 128 × ~10 ops    = ~820 KFLOPS (softmax)
```

**Fix:** `norm_topk_prob=True` (already set in config) normalizes affinities over the selected K=8 experts rather than all 128, which reduces the precision sensitivity of the full softmax. FP32 is sufficient. Change the router dtype or patch the routing code to avoid FP64.

---

## Priority Summary

| # | Decision | Type | Effort | Key Metric |
|---|---|---|---|---|
| 1 | `block_size=128` | Config | Trivial | 4x GEMM waste reduction; unlocks DD2 |
| 2 | Double-buffer gate+up DMA in CTE kernel | Custom kernel | Medium | Approach bandwidth bound |
| 3 | `use_shard_on_block_dynamic_while=True` | Config | Trivial | Free win for seqlen < 640 |
| 4 | Fused RMSNorm + Router + Blockwise kernel | Custom kernel | High | Eliminate multi-pass HBM + launch overhead |
| 5 | FP32 router instead of FP64 | Code change | Low | ~4x router compute speedup |

**Implementation order:** DD1 + DD3 first (config-only, zero risk, immediate measurement). Then DD2 (CTE adaptation of existing v7b TKG pattern). Then DD5. Then DD4 if time permits.

---

## Key Shapes Reference (TP=4, EP=4)

| Quantity | Value |
|---|---|
| H (hidden) | 2048 |
| I (expert intermediate) | 768 |
| I_local (per NC, LNC=2) | 384 |
| E (total experts) | 128 |
| E_local (per rank, EP=4) | 32 |
| K (active experts/token) | 8 |
| seqlen (CTE) | 640 |
| tokens/expert (avg) | 40 |
| Expert weight per expert | 9.44 MB |
| Expert weights per rank/layer | 302 MB |
| Arithmetic intensity | 40 FLOPS/byte |
| trn2 ridgeline | 232 FLOPS/byte |
| Theoretical min time (bw-bound, 48 layers) | ~17.7 ms |
| Measured MoE time | ~357 ms |
