# Attention TKG Theoretical Lower Bound & Optimization Report

## 1. Ground-Truth Shapes at TP=4

With `REPLICATE_TO_TP_DEGREE` for KV heads and TP=4:

| Tensor | Shape | Bytes (bf16) |
|--------|-------|-------------|
| `hidden_states` | `[1, 1, 2048]` | 4 KB |
| `Wq` (sharded) | `[8×128, 2048]` = `[1024, 2048]` | **2 MB** |
| `Wk` (replicated → 1 KV head/rank) | `[128, 2048]` | 256 KB |
| `Wv` | `[128, 2048]` | 256 KB |
| `Wo` (sharded) | `[2048, 1024]` | **2 MB** |
| K-cache | `[1, 1, S_prior, 128]` | 160 KB (S_prior=640) |
| V-cache | same | 160 KB |
| **Total HBM** | | **~4.8 MB** |

> **Critical finding**: The custom NKI kernels (v0–v6) all benchmark with `Hkv_tp=4`, i.e., they process all 4 KV heads per rank. The real TP=4 shape only has **1 KV head per rank**. This inflates benchmark time by ~4× on the KV path and makes the GQA ratio 2 instead of the real 8. The 430–530µs benchmarks are measuring the wrong problem.

---

## 2. Theoretical Lower Bound

### Memory Bandwidth Analysis (trn2)

trn2 per-NC HBM bandwidth: **~900 GB/s per NeuronCore pair** (LNC=2 gets the full chip bandwidth). With TP=4 using 4 separate chips, each rank has its own HBM, so there is no bandwidth sharing penalty.

| Component | Bytes | Time @ 900 GB/s |
|-----------|-------|-----------------|
| Wq + Wk + Wv (weights) | 2.5 MB | **2.8 µs** |
| Wo (output proj) | 2.0 MB | **2.2 µs** |
| KV cache K+V | 320 KB | **0.36 µs** |
| hidden + output | ~8 KB | negligible |
| **Total** | **~4.8 MB** | **~5.4 µs** |

### Compute Analysis

For B=1, s_active=1, GQA=8, S_prior=640:

| Op | FLOPs |
|----|-------|
| Wq projection: `[2048]×[2048,1024]` | 4.2 M |
| Wk projection: `[2048]×[2048,128]` | 0.5 M |
| Wv projection: same | 0.5 M |
| 8 Q heads × KV dot (S_prior=640) | 1.3 M |
| 8 Q heads × V weighted sum | 1.3 M |
| Wo projection: `[1024]×[1024,2048]` | 4.2 M |
| **Total** | **~12 M** |

At 23 TFLOPS (bf16) per NeuronCore: 12M / 23T = **0.5 µs** compute-bound.

The operation is **completely memory-bandwidth bound**, not compute-bound. The roofline is ~5.4 µs ideal, ~7–8 µs achievable (accounting for all-reduce + DMA overhead).

### Current vs Ideal

| | Time |
|---|---|
| Current compiled (nkilib attn_tkg) | **~40 µs** |
| Roofline (all weights from HBM) | **~5.4 µs** |
| Practical target (with engineering) | **~10–15 µs** |

The **5–7× gap** between compiled and roofline is explained by: weight loads not being pipelined optimally, serial head processing, transpose overhead, and suboptimal DMA scheduling. The compiled nkilib kernel does not statically pin weights in SBUF between steps.

---

## 3. Design Decisions

### Decision 1: Fix the Per-Rank KV Head Count

**Problem**: Custom kernels use `Hkv_tp=4` (all KV heads replicated). The real TKG shape per rank with TP=4 + `REPLICATE_TO_TP_DEGREE` is `Hkv_tp=1` (1 KV head/rank), giving **GQA=8**.

**Action**: Rewrite the kernel with `Hkv_tp=1`:
- KV cache reads drop from `4×S×128×2=640KB` to `1×S×128×2=160KB`
- KV flash decode loop iterates over 1 KV head instead of 4
- Estimated impact on benchmarks: **4× reduction in KV bandwidth**, ~100µs → ~25µs at the kernel level

---

### Decision 2: GQA=8 Free-Dim Packing → Single KV Cache Pass

**Current (v6)**: Packs `gqa=2` Q heads into `[PMAX, 2]` free dim. With the wrong `Hkv_tp=4`, runs 4 KV head outer loops.

**Correct target**: With `Hkv_tp=1, gqa=8`, pack all 8 Q heads into `[PMAX, 8]` in the free dimension. The entire KV cache pass (the dominant cost for long context) is executed **once** for all 8 Q heads simultaneously:

```python
# Score: k_tile [128, 128] @ q_packed [128, 8] → [128, 8]  — valid free-dim width
# V weighted: v_tile_T [128, 128] @ score_exp [128, 8] → [128, 8]
```

This is correct because attention is independent across Q heads sharing the same KV head. The matmul `k_tile[128×128] × q_packed[128×8]` is valid in NKI (free dim ≤ PMAX). Online softmax operates column-independently on `[128, 8]`.

**Estimated impact**: 8× reduction in KV cache DMA and KV matmul count vs the naive per-head loop.

---

### Decision 3: Eliminate HBM Bounce in Scalar Broadcasts

**Problem**: The current kernels do:
```python
# score [PMAX, 8] → reduce → scalar [1,1] → dma to private_hbm → dma back → [PMAX, 8]
```
This SBUF→HBM→SBUF round-trip is a latency bottleneck in the flash decode inner loop. The nkilib production kernel avoids this entirely.

**Action**: Use `nisa.stream_shuffle_broadcast`:
```python
# Reduce to [1,1] scalar in SBUF, then:
nisa.dma_copy(dst=scalar_sb[0, 0], src=tile_max_scalar_sbuf)
stream_shuffle_broadcast(src=scalar_sb, dst=broadcast_sb)  # broadcast across all 128 partitions
```
This keeps data on-chip (SBUF→SBUF) via the NeuronCore's shuffle network, avoiding the HBM roundtrip entirely.

**Estimated impact**: Eliminates 2× HBM round-trips per KV tile per head. At S_prior=640 (5 KV tiles), 8 Q heads: saves 80 HBM bounce operations per TKG step.

---

### Decision 4: dma_transpose Instead of nc_transpose

**Problem**: v0–v6 all use `nisa.nc_transpose` to rotate tensors. On trn2 (NeuronCore-v3 / Gen3), `nisa.dma_transpose` is a hardware-accelerated transpose that happens in the DMA engine while data is being transferred — **zero extra Tensor Engine cycles**.

**Action**: Replace `nc_transpose(dst, src)` patterns (which tie up the Tensor Engine) with `dma_transpose` when loading from HBM. For V tiles:
```python
# Instead of:
nisa.dma_copy(dst=v_tile, src=V_cache[...])  # load
nisa.nc_transpose(v_tile_T, v_tile)           # separate transpose (tensor engine)

# Use (Gen3 / trn2 only):
nisa.dma_copy(dst=v_tile_T, src=V_cache[...], dst_layout=nl.layout.col_major)
```
The v6 column-layout trick (reshaping source to `[PMAX, B]`) already avoids one transpose by using the correct HBM layout. Extend this to all remaining transposes including V-tile loads. The nkilib `attention_tkg` uses this pattern internally.

**Estimated impact**: Frees the Tensor Engine from all transpose work, allowing 100% occupancy for matmuls. Removes the 90.7% transpose FLOPS overhead seen in v0 profiling.

---

### Decision 5: 3-Stage Q-Group Pipeline (Overlap MM1 with MM2)

**Problem**: The current kernels process heads sequentially: compute Q RMSNorm+RoPE → KV flash decode → store, fully serialized.

**Action**: Implement the 3-stage pipeline used by the production nkilib kernel:

```
Iteration i:   [V weighted sum (MM2) + store output for group i-1]
Iteration i+1: [exp(scores) + softmax update for group i]
Iteration i+2: [Q projection + K·Q^T (MM1) for group i+1]
```

With 8 Q heads split into sub-groups of 4, pipeline the sub-group iterations. More importantly: pipeline the **QKV projection** of the next step's hidden state with the **flash decode** of the current step. Since they are independent computations (flash decode reads KV cache, projection reads weights), they can run on DMA and compute engines simultaneously.

**Estimated impact**: Hides QKV projection latency (2.8µs weight reads) behind KV flash decode computation, potentially saving 2–4µs.

---

### Decision 6: PSUM Bank Interleaving for Zero RAW Stalls

**Problem**: When `nc_matmul` accumulates into the same PSUM address across loop iterations, read-after-write stalls occur. This serializes the inner `h_tiles` loop.

**Action**: Assign each KV-tile matmul accumulator to a different PSUM bank using modulo-4 interleaving:
```python
k_score_psum = nl.zeros((PMAX, 8), dtype=nl.float32, buffer=nl.psum,
                         address=(0, (s_t % 4) * PSUM_BANK_SIZE))
```
This is exactly what the nkilib `attention_tkg` kernel does. It eliminates the pipeline bubble between accumulate and read in the inner flash decode loop.

**Estimated impact**: Could recover 10–30% hidden cycles in the KV scan loop.

---

### Decision 7: Fuse QKV + Attention + o_proj into One Kernel

**Current architecture**: `NeuronAttentionBase.forward()` calls separate ops: `qkv_proj` (ColumnParallelLinear), then the GQA module (nkilib attention_tkg), then `o_proj` (RowParallelLinear + all-reduce). Each is a separate kernel launch with separate HBM read/write transactions.

**Action**: Write a single NKI kernel that takes `hidden_states` as input and produces the post-o_proj output, keeping all intermediate activations (Q, K, V, attn_out) in SBUF. The key benefit:
- `Wq, Wk, Wv, Wo` are loaded from HBM once into SBUF and stay there for the entire computation
- No intermediate HBM writes — Q, K, V tensors never touch HBM
- Kernel launch overhead reduced from 3+ launches to 1

**SBUF footprint check** (trn2 SBUF = 48 MB per NC):

| Tensor | Size |
|--------|------|
| Wq | 2 MB |
| Wk | 256 KB |
| Wv | 256 KB |
| Wo | 2 MB |
| h_tiles (hidden, hoisted) | 32 KB |
| Q heads `[PMAX, 8]` | ~8 KB |
| KV tile × 2 | 64 KB |
| Flash decode accumulators | 16 KB |
| **Total** | **~4.6 MB** — fits easily in 48 MB |

This is the single largest architectural win available. Eliminates ~2.5MB of weight re-loads per layer per step (the dominant cost).

**Estimated impact**: Reduces effective memory traffic from 4.8MB → 0.5MB (only KV cache + hidden + output), targeting ~1–2µs execution time for the fused kernel body.

---

### Decision 8: LNC=2 Grid for TKG Attention Kernel

**Current**: The TKG path through `NeuronAttentionBase` uses the grid the nkilib kernel configures. The `BlockwiseMatmulConfig(logical_nc_config=2)` only affects the MoE kernel, not the attention TKG path.

**Action**: Explicitly enable LNC=2 for the custom TKG attention NKI kernel:
```python
from neuronxcc.nki.typing import nc
grid = (nc(2),)  # 2 NeuronCores in the logical NC pair
```
With LNC=2 and 8 Q heads, assign 4 Q heads to each NC. Both NCs independently:
- Process their 4 Q projections (reading non-overlapping Wq rows: `[0:512]` and `[512:1024]`)
- Run flash decode against the same 1 KV head (both NCs read the same KV cache rows — replicated reads are hardware-efficient on LNC=2 shared HBM)

The doubled memory bandwidth from 2 NCs working in parallel significantly reduces the bottleneck on the weight-load bound.

**Estimated impact**: Near-linear 2× speedup on the weight-load-bound portion.

---

### Decision 9: Compiler Flag Tuning for TKG

In `get_compiler_args()`, the TKG path already uses `-O3` with several flags. Verify and consider:

```python
# Already present and correct:
"--enable-ccop-compute-overlap"           # DMA-compute overlap
"--eager-tkg-vectorize-dma"               # DMA vectorization for single-token ops
"--enable-dge-on-indirect-dma"            # Dynamic gather for KV cache addressing
"--enable-dge-on-vector-indirect-dma"

# Candidates to investigate:
"--internal-enable-hoist-sb-allocations"  # May help with weight SBUF pinning
"--enable-strided-dma-hoist"              # Hoists strided DMA patterns to static offsets
```

Also: if the custom TKG kernel declares weight tensors as `nl.static_hbm` (constant weights), the compiler can pre-stage them at compile time, skipping DMA generation at runtime entirely.

---

### Decision 10: Context-Length-Aware Bucketing

The KV cache cost scales linearly with S_prior:

| S_prior | KV bytes | KV time @ 900 GB/s | Total roofline |
|---------|----------|---------------------|----------------|
| 128 | 64 KB | 0.07 µs | ~5.1 µs |
| 640 | 320 KB | 0.36 µs | ~5.4 µs |
| 2048 | 1 MB | 1.1 µs | ~6.3 µs |
| 8192 | 4 MB | 4.4 µs | ~10 µs |

For short contexts (early TKG steps), KV is not the bottleneck — weights dominate. For long contexts (>2k), KV bandwidth begins to compete with weight loads.

**Action**: Use many small buckets at short context lengths (where bucket granularity matters most for padding waste) and fewer coarse buckets at long contexts (where KV bandwidth dominates and is unavoidable anyway). With `max_context_length=640`, the KV contribution is minor and the fused-kernel design targets the weight-dominated regime.

---

## 4. Implementation Priority Ranking

| # | Decision | Estimated Gain | Difficulty |
|---|----------|----------------|------------|
| 1 | Fix Hkv_tp=1 in custom kernel benchmarks | 4× benchmark accuracy | Low |
| 2 | Fuse QKV+Attn+o_proj (weight SBUF pinning) | 3–5× (dominant win) | High |
| 3 | GQA=8 free-dim packing `[PMAX, 8]` | 8× KV reads reduction | Medium |
| 4 | `dma_transpose` instead of `nc_transpose` | 10–30% cycle reduction | Low |
| 5 | `stream_shuffle_broadcast` for scalars | 5–15% latency reduction | Medium |
| 6 | LNC=2 grid with head split | 1.5–2× on weight loads | Medium |
| 7 | PSUM bank interleaving | 10–20% pipeline stall reduction | Low |
| 8 | 3-stage Q group pipeline | 10–20% overlap gain | High |
| 9 | Compiler flags | 5–10% | Low |
| 10 | Context bucketing | 5–15% at short ctx | Low |

---

## 5. Minimum Viable Target

With decisions 1–5 implemented (fixing shapes + fused kernel + GQA packing + Gen3 transpose + on-chip broadcast):
- Wq+Wk+Wv+Wo loads: **eliminated** if kernel is fused and weights are SBUF-pinned
- KV reads: 320 KB (unavoidable for S_prior=640) at 900 GB/s = 0.36 µs
- All-reduce (4 KB, 4 ranks): ~1–2 µs
- RMSNorm + RoPE + flash decode compute: <0.5 µs
- **Realistic target: ~5–8 µs** (down from 40 µs) — a 5–8× improvement

The **critical path** is: KV cache bandwidth + all-reduce latency. Both are irreducible at the hardware level. Achieving sub-5 µs would require either INT8 KV cache quantization (halves KV bandwidth) or hardware-level flash decode acceleration beyond what NKI currently exposes.
