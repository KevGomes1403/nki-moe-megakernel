# MoE TKG Optimization Analysis: Qwen3-30B-A3B, TP=4, LNC=2, trn2

**Context:** MLSys 2026 contest. Current benchmarked baseline (`benchmark_report_v8a_128_block.json`):

| Metric | Value |
|--------|-------|
| CTE latency (p50) | **196.4 ms** |
| TKG latency (p50) | **10.06 ms / step** |
| TKG throughput | **99.5 tokens/sec** |
| E2E latency (p50) | 10,019 ms |

The v8a kernel with `block_size=128` closed the CTE gap significantly (from ~407ms). TKG at ~10ms/step is the primary remaining target.

---

## 1. Current TKG MoE Execution Path

In `qwen_with_attn_cte_nki.py`, the TKG MoE path is:

```
NeuronQwen3MoeDecoderLayerWithNKIAttn.forward()
  ├─ post_attention_layernorm(hidden)                    ← separate XLA op (moe_fused_nki_kernel_enabled=False)
  └─ self.mlp = initialize_moe_module(init_tkg_module=True)
       └─ MoEFusedTKG (NxD wrapper)
            └─ _can_use_nki_kernel():
                 perc_experts_loaded = T×K / E = 1×8/128 = 6.25%
                 → selective-load path
                      └─ nkilib selective_expert_impl._selective_expert_moe_tkg()
```

The **router** (`RouterTopK`) is a separate XLA op: `[T, H] × [H, E=128]` linear + float64 softmax + topK. It writes `[T, 128]` affinities and `[T, 8]` indices to HBM before the expert kernel launches.

Profiler: **MoE TKG layer ≈ 100 µs**. With 48 layers: ~4.8 ms of the 10.06 ms TKG step.

---

## 2. Arithmetic Intensity Analysis

**Shapes per TP rank, T=1 decode:**

| Quantity | Value |
|----------|-------|
| H (hidden) | 2048 |
| I_tp (intermediate per TP rank) | 768 / 4 = **192** |
| E (experts, replicated on every rank) | 128 |
| K (selected per token) | 8 |

**Weight bytes loaded for K=8 selected experts (bf16):**

```
gate+up : K × H × 2 × I_tp × 2 B  =  8 × 2048 × 2 × 192 × 2  =  12.0 MB
down    : K × I_tp × H × 2 B       =  8 ×  192 × 2048 × 2     =   6.0 MB
total   :                                                         18.0 MB
```

**Compute (T=1):**

```
gate + up : 8 × 2 × 2048 × 192 × 2 MACs = 12.6 GFLOPS
down      : 8 × 192 × 2048 × 2 MACs     =  6.3 GFLOPS
total     :                               18.9 GFLOPS
```

**Arithmetic intensity: 18.9 GFLOPS / 18.0 MB = ~1.05 FLOPS/byte**

trn2 ridgeline: **232 FLOPS/byte** (190 TFLOPS / 820 GB/s). This is **220× below** the compute-bound threshold — a pure HBM bandwidth problem.

**Theoretical floor at estimated ~450 GB/s per chip with LNC=2:**

```
18.0 MB / 450 GB/s ≈ 40 µs per layer
```

Measured: **100 µs per layer → ~40% HBM bandwidth utilization.** 2.5× gap before the bandwidth wall, and quantization can move the wall further.

---

## 3. Design Decisions

---

### Decision 1 — MxFP4 Expert Weight Quantization ★ Highest Impact

**What:** Quantize expert gate/up/down weights from bf16 to MxFP4 (4 bits per element, microscaling). nkilib already supports this via `selective_expert_mx_impl.py`.

**Why:** Bandwidth-bound kernels scale linearly with bytes moved:

```
bf16 weight bytes:  18.0 MB  →  expected time: ~100 µs
MxFP4 weight bytes:  4.5 MB  →  expected time: ~25 µs/layer
48 layers: 4.8 ms → ~1.2 ms
```

**How:** Pass pre-quantized weights (`dtype=float4_e2m1fn_x4`) and per-128-element block scales to `moe_tkg`. Quantization is done offline in `convert_qwen3_moe_hf_to_neuron_state_dict`. The nkilib kernel handles dequantization via the MxFP4 ISA path internally.

**Risk:** Accuracy loss — requires calibration and validation against the CPU reference logit check. Start with Decision 1b (FP8) as a safer stepping stone.

---

### Decision 1b — FP8 Row-Scale Quantization (safer fallback)

**What:** Quantize expert weights to FP8 with per-row scales. nkilib supports `ExpertQuantizationType.ROW`.

```
bf16 bytes: 18.0 MB → FP8 bytes: 9.0 MB + ~0.1 MB scales
Expected time: ~50 µs/layer
48 layers: 4.8 ms → ~2.4 ms
```

FP8 per-row has better dynamic range and fewer calibration artifacts than MxFP4. Use as a checkpoint before committing to MxFP4.

---

### Decision 2 — Direct nkilib Call (eliminate NxD wrapper)

**What:** Replace `initialize_moe_module(init_tkg_module=True)` → `MoEFusedTKG` with a direct `nki.jit(moe_tkg)` call, exactly as in `qwen_with_moe_tkg.py:_forward_moe_tkg()`.

**Why:** `MoEFusedTKG` adds Python dispatch overhead, argument validation, NxD routing logic, and type checks on every forward call. At 100 µs, even 5–10 µs of host-side overhead is a visible fraction. The direct-call pattern also gives full control over:
- `expert_affinities` dtype (must be float32 for POST_SCALE)
- `expert_index` int32 format
- Quantization scale passing (needed for Decisions 1/1b)

**How:** Merge `qwen_with_moe_tkg.py`'s `_forward_moe_tkg()` into `qwen_with_attn_cte_nki.py`. CTE remains via standard MoE; TKG dispatches direct via `q_len == 1` check.

**Effort:** Low. The code already exists in `qwen_with_moe_tkg.py`.

---

### Decision 3 — Enable `moe_fused_nki_kernel_enabled=True`

**What:** Set `moe_fused_nki_kernel_enabled=True` in the config so the MoEFusedTKG (or direct nkilib kernel) fuses `post_attention_layernorm` internally instead of applying it as a separate XLA op.

**Current state (`moe_fused_nki_kernel_enabled=False`):**
1. `post_attention_layernorm(hidden)` → writes `[1, 2048]` = 4 KB to HBM
2. Router reads this, writes `[1, 128]` affinities to HBM
3. nkilib reads from HBM again

**With fusion:** hidden stays in SBUF through LN → router → expert loop. The `if not self.moe_fused_nki_kernel_enabled:` gate at decoder line 603 already handles the LN correctly — enabling the flag removes the separate LN call.

The absolute HBM saving (4 KB/layer × 48) is modest at T=1, but eliminating 2 XLA op launches per layer removes CPU dispatch overhead.

---

### Decision 4 — Fuse Router Computation into NKI Expert Kernel

**What:** Include the router `[T, H] × [H, E=128]` linear + softmax + topK inside the NKI kernel, before the expert GEMM loop.

**Why:**

```
Router weight DMA:    H × E × 2 B = 2048 × 128 × 2 = 512 KB / layer
At 450 GB/s:          ~1.1 µs
XLA kernel launch:    ~2–5 µs
Sync point overhead:  router must complete before expert indices are known → sequential dispatch

Total saving:         ~3–6 µs/layer → 1.5–2.9 ms total over 48 layers
```

The router weight (512 KB) fits comfortably in SBUF alongside the hidden state (4 KB).

**Implementation sketch:**

```
NKI kernel receives: hidden [T, H], W_router [H, E=128]
  Step 1: hidden → SBUF
  Step 2: W_router → SBUF (512 KB)
  Step 3: logits = hidden @ W_router  →  [T, E] in SBUF
  Step 4: softmax(logits, fp32)       →  [T, E]
  Step 5: topK(E=128, K=8)            →  expert_indices [T,K], affinities [T,K]
  Step 6: norm_topk_prob: affinities /= sum(affinities)
  Step 7: expert GEMM loop (existing kernel logic)
```

**Key NKI challenge:** topK over 128 elements. For K=8 out of E=128, a partial sort (select K largest) in register is feasible — 128 elements is small enough for a register-level selection sort. This is a strong innovation score candidate.

---

### Decision 5 — Double-Buffer Expert Weight Loading

**What:** Pre-load the **next** expert's gate+up weights while computing the **current** expert's down projection, using `nl.affine_range` on independent DMA loads.

**Kernel structure (target):**

```python
# Pre-load gate+up for expert 0
for h_t in nl.affine_range(num_h_tiles):
    gate_up_buf[0][h_t] = nki.dma_copy(gate_up_weights[e0, h_t])

for k in range(K):  # sequential — output accumulation dep
    # [parallel] Pre-load gate+up for expert k+1
    for h_t in nl.affine_range(num_h_tiles):
        gate_up_buf[(k+1)%2][h_t] = nki.dma_copy(gate_up_weights[e_{k+1}, h_t])

    # compute gate+up matmul for expert k (DMA of k+1 overlaps here)
    ...
    # SiLU × up

    # [parallel] Pre-load all down weight rows
    for i_t in nl.affine_range(num_i_tiles):
        down_buf[i_t] = nki.dma_copy(down_weights[e_k, i_t])

    # compute down matmul using pre-loaded rows
    ...
    # scale by affinity, accumulate
```

**Quantitative case (T=1, H=2048, I_tp=192):**

```
Gate+up per expert: 2048 × 192 × 2 × 2 B = 1.57 MB → DMA: 1.57 MB / 450 GB/s ≈ 3.5 µs
Down per expert:    192 × 2048 × 2 B = 0.75 MB → 1.7 µs
Compute (T=1):      negligible (~0.08 µs)

Sequential: K × (DMA_gate_up + DMA_down) = 8 × 5.2 µs ≈ 41.6 µs
Double-buffered:    K × max(DMA_gate_up, DMA_down) ≈ 8 × 3.5 µs ≈ 28 µs
```

**SBUF budget (T=1):**

```
Hidden state:        1 × 2048 × 2 B = 4 KB
gate_up buffer A:    2048 × 192 × 2 B = 0.75 MB
gate_up buffer B:    2048 × 192 × 2 B = 0.75 MB  ← second buffer for overlap
down buffer:         192 × 2048 × 2 B = 0.75 MB
PSUM tiles:          a few 128×128×4 B = ~0.25 MB
Total: ~2.5 MB  →  fits in 4 MB SBUF with LNC=2
```

This is the v7b pattern (`kernels/moe_tkg/agents/v7b_preload_gateup_rows.py`). The v6b production kernel (`kernels/v6b_qwen3.py`) has coalesced down DMAs but not the full double-buffer between experts.

**Why the custom kernel may lag nkilib:** nkilib uses `stream_shuffle` (inter-partition data movement without HBM), `local_gather` for affinity collection (T≤16 fast path), and GPSIMD-aligned SBUF addresses. If any load inside the K-loop uses `range()` instead of `affine_range()`, the compiler serializes those iterations and DMA-compute overlap is lost.

---

### Decision 6 — Enable Dynamic-While Flags (free config win)

**What:** Uncomment the two flags currently disabled in `Qwen3MoEAttnCTENeuronConfig`:

```python
BlockwiseMatmulConfig.from_kwargs(
    block_size=128,
    logical_nc_config=2,
    skip_dma_token=True,
    skip_dma_weight=True,
    use_shard_on_block_dynamic_while=True,        # was commented out
    use_shard_on_intermediate_dynamic_while=True,  # was commented out
)
```

**Why:** These enable a kernel variant that checks at runtime whether a block is non-empty before issuing any DMA or compute. For CTE at shorter prompt lengths:

| seqlen | tokens/expert (avg) | Empty expert blocks/rank | Skip rate |
|--------|---------------------|--------------------------|-----------|
| 640 | 40 | ~0% | negligible |
| 128 | 8 | ~40% | ~40% |
| 32 | 2 | ~80% | ~80% |

If the contest benchmarks multiple sequence-length buckets, this is a free win. Zero kernel code required.

---

### Decision 7 — T=1 Kernel Specialization

**What:** In the custom NKI kernel, add a compile-time `T == 1` specialization that:
- Removes the outer T-loop entirely
- Loads the single hidden state as a flat `[H]` vector (no token-dimension SBUF broadcast)
- Uses a simplified affinity gather (K values from direct SBUF index, no partition alignment)
- Accumulates into a single `[H]` PSUM tile

**Why:** nkilib's selective_expert_impl handles variable T (1 to 128) through general loops and the `gather_expert_affinities` utility's T≤16 fast path. A T=1-specialized kernel eliminates all T-dimension overhead: loop setup, partition-alignment padding, and T-broadcast in the affinity/output tiles.

**Effort:** Medium — requires a separate kernel variant, but T=1 is the dominant decode case and reduces instruction count meaningfully.

---

### Decision 8 — Attention TKG NKI Kernel

**What:** The TKG attention path currently uses `NeuronAttentionBase.forward()` — the standard NxD KV-cache path. The `kernels/attn_tkg/agents/` directory contains a progression through `v6_ultimate`. Integrating a fused NKI TKG attention kernel (QKV projection + per-head RMSNorm + RoPE + flash decode) would eliminate HBM round-trips for intermediate Q/K/V tensors.

**Why:** This is the natural complement to the CTE NKI attention (`v7ab`). The per-head QK-norm is Qwen3-specific and not fused in any reference kernel — strong innovation score candidate. See `docs/optimization_roadmap.md` Priority 1 for full analysis.

---

### Decision 9 — FP32 Router (2-line fix)

**What:** `RouterTopK` runs the softmax in float64 by default. trn2 runs FP64 at ~47.5 TFLOPS vs 190 TFLOPS for FP32 — 4× slower for the same op count.

```python
# Force FP32 before softmax in the router forward
logits = (hidden @ router_weight).to(torch.float32)
affinities = torch.softmax(logits, dim=-1)
```

With `norm_topk_prob=True` (already set in Qwen3 config), the renormalization touches only K=8 values — FP32 precision is sufficient.

Router FLOPs at T=1: `1 × 2048 × 128 × 2 = 524K MACs` — tiny in absolute terms, but FP64 stalls on the scalar unit can create pipeline bubbles.

---

### Decision 10 — AllReduce / CC Overlap

**What:** After each TKG MoE layer, a TP AllReduce sums partial `[1, 2048]` results across 4 ranks (8 KB of data). AllReduce startup latency (~5–20 µs) dominates over bandwidth at this size.

**Options:**
- **`--cc-pipeline-tiling-factor=1`** for TKG (currently 2): at T=1 there is no useful token-dimension tiling, and tiling factor 2 adds unnecessary slicing overhead.
- **Overlap AllReduce[L] with LN+Router[L+1]**: the `--enable-ccop-compute-overlap` flag (already set) enables this in the compiler. Verify it is effective by checking profiler traces for AR/compute overlap after each MoE layer.

---

## 4. Summary Table

| # | Decision | Type | BW Impact | Layer Target | Effort | Risk |
|---|----------|------|-----------|--------------|--------|------|
| **1** | MxFP4 expert weights | State dict + config | 4× reduction | ~25 µs | Medium | Accuracy; validate |
| **1b** | FP8 row-scale weights | State dict + config | 2× reduction | ~50 µs | Low-Med | Low |
| **2** | Direct nkilib call (no NxD wrapper) | Code (import from qwen_with_moe_tkg.py) | Host overhead only | −5–10 µs | Low | Very low |
| **3** | `moe_fused_nki_kernel_enabled=True` | Config flag | Tiny (T=1) | −2–3 µs | Very low | Low |
| **4** | Fuse router into NKI kernel | Custom kernel | Saves 512 KB/layer | −6–8 µs | Medium | Medium (topK in NKI) |
| **5** | Double-buffer gate+up DMA | Custom kernel | Overlaps DMA+compute | ~28 µs | Medium | Low (v7b exists) |
| **6** | Enable dynamic-while flags | Config (uncomment 2 lines) | Skips empty blocks | Free for T>1 | Very low | Very low |
| **7** | T=1 kernel specialization | Custom kernel | Reduces instruction overhead | −5–10 µs | Medium | Low |
| **8** | NKI TKG attention kernel | Custom kernel | Fuses QKV+RoPE+decode | −20–40 µs | High | Medium |
| **9** | FP32 router | 2-line code change | None | −1–3 µs | Very low | Very low |
| **10** | `cc-pipeline-tiling-factor=1` for TKG | Compiler flag | None | −2–5 µs | Very low | Low |

---

## 5. Recommended Implementation Order

### Phase 1 — Low risk, immediate (1–2 days)
1. **Decision 2**: Merge direct nkilib call from `qwen_with_moe_tkg.py` into `qwen_with_attn_cte_nki.py`. Foundation for all subsequent changes.
2. **Decision 9**: FP32 router. Two-line change.
3. **Decision 3**: Enable `moe_fused_nki_kernel_enabled=True`, verify no double-LN.
4. **Decision 6**: Uncomment `use_shard_on_block_dynamic_while=True`.
5. **Decision 10**: Try `--cc-pipeline-tiling-factor=1` for TKG and measure.

### Phase 2 — Medium effort, major gains (3–5 days)
6. **Decision 1b**: FP8 quantization — add to state dict conversion, thread scales through to nkilib call.
7. **Decision 5**: Debug and integrate v7b double-buffering kernel. Profile to confirm `nl.affine_range` is on all independent DMA loads.

### Phase 3 — High effort, high reward (innovation score)
8. **Decision 4**: Fused router NKI kernel. Strongest novelty candidate for the 15% innovation score.
9. **Decision 1**: Upgrade FP8 → MxFP4 using `selective_expert_mx_impl` path.
10. **Decision 8**: NKI TKG attention kernel integration (see `optimization_roadmap.md`).

---

## 6. Key Shapes Reference (TP=4, LNC=2)

| Quantity | Value |
|----------|-------|
| H (hidden) | 2048 |
| I (expert intermediate, full) | 768 |
| I_tp (per TP rank) | 192 |
| E (total experts) | 128 |
| K (active experts/token) | 8 |
| Expert weight per expert (bf16) | 2 × 2048×192×2 + 192×2048×2 = **2.25 MB** |
| Expert weights loaded per token (K=8) | **18.0 MB** |
| Arithmetic intensity (T=1) | **1.05 FLOPS/byte** |
| trn2 ridgeline | 232 FLOPS/byte |
| Theoretical min (BW-bound, 450 GB/s) | **~40 µs/layer** |
| Measured (current) | **~100 µs/layer** |
| Measured TKG step (48 layers total) | **10.06 ms** |
| MoE TKG share of step | ~4.8 ms / 10.06 ms ≈ **48%** |
