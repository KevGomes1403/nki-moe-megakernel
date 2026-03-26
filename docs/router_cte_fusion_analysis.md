# Router + CTE Expert Kernel Fusion Analysis

Analysis of optimization opportunities for integrating the custom NKI router kernel
(`kernels/router_topk/qwen3_router_topk_plan_a.py`) with the library blockwise CTE
expert kernel on trn2, LNC=2, TP=4.

---

## Single `@nki.jit` Fusion: Why It Is Blocked

You cannot wrap the library blockwise expert kernel inside your own `@nki.jit`. NKI kernels
cannot call other `@nki.jit` functions; the compiler compiles each kernel independently.
`BlockwiseMatmulNKIFunc.apply()` is a Python-level `torch.autograd.Function` that dispatches
to one of seven `@nki.jit` kernels (`blockwise_mm_baseline_shard_hidden`,
`bwmm_shard_on_block`, etc.) based on config — all sealed in
`neuronxcc.nki._pre_prod_kernels`.

Even if that were not the case, `nl.shared_hbm` **is** HBM. There is no mechanism in the
compiler to keep `rl_out`/`ea_out`/`ei_out` in SBUF across two separately-launched `@nki.jit`
kernels. The data must round-trip through HBM regardless of how the Python-side call is
structured.

---

## The Real Pipeline Bottleneck

The current CTE MoE execution has three distinct HBM round-trips between router and experts:

```
Router kernel (nki.jit)
  → ea_out [T, E]   f32  (T×128×4 B)   ← scatter, mostly zeros
  → ei_out [T, K]   u32  (T×8×4 B)
  → rl_out [T, E]   f32  (T×128×4 B)   ← not consumed downstream
      ↓ HBM write
get_blockwise_expert_and_token_mapping  ← Python or NKI kernel (reads from HBM)
  → token_position_to_id [N*B]  i32
  → block_to_expert      [N]    i32
      ↓ HBM write
blockwise expert kernel (nki.jit)
  ← hidden_states [T, H]               (gathers per-block via token_position_to_id)
  ← expert_affinities_masked [T, E]    (gathers per-block per-expert)
```

The `[T, E]` scatter in the router is the most wasteful step. For T=640, K=8: writing
640×128×4B = 320KB just so the expert kernel can read back 640×8 useful affinity values via
sparse gather. `rl_out` (logits) is never consumed by anything downstream.

---

## Optimization 1: Fuse Token Mapping Into the Router Kernel

The router already has `ei_out [T_TILE, K]` in SBUF when it finishes each T-tile. At that
point it has all information needed to compute `block_to_expert [N]` and
`token_position_to_id [N*B]` before writing anything to HBM.

The histogram computation is nearly free because `mask_sb [T_TILE, E]` (the one-hot scatter
buffer) already encodes expert membership. Summing over the T dimension gives
`tokens_per_expert [E]` for that tile:

```python
# mask_sb [T_TILE, E] is already computed for the scatter step
tile_hist [1, E] = nisa.tensor_reduce(mask_sb, op=nl.add, axis=0)
# accumulate into persistent hist_sb [1, E] across T-tiles
```

After all T-tiles, compute per-expert prefix sums in SBUF to get block start offsets.
From the prefix sums + `ei_out`, scatter token indices into `token_position_to_id [N*B]`
and write it once to HBM at the end.

**Updated output contract:**

| Tensor | Shape | dtype | Notes |
|--------|-------|-------|-------|
| `ea_out` | `[T, E]` | f32 | unchanged, library kernel compatible |
| `ei_out` | `[T, K]` | u32 | unchanged |
| `block_to_expert` | `[N]` | i32 | **NEW** |
| `token_position_to_id` | `[N*B]` | i32 | **NEW** |

`rl_out` can be dropped entirely (nothing downstream reads it).

**Size check:** N = ceil(T×K / B) + E − 1. For T=640, K=8, B=128, E=128:
N ≤ ceil(5120/128) + 127 = 167. `token_position_to_id [167×128] = 21 KB` — fits easily in
SBUF. This eliminates `get_blockwise_expert_and_token_mapping` (PyTorch or NKI variant)
entirely, removing one full HBM round-trip of index data plus the separate kernel launch
overhead.

---

## Optimization 2: Eliminate the Scatter, Output Compact Affinity Format

The current router pipeline per T-tile:

1. Softmax → `affinities [T_TILE, E]`
2. Top-K → `topk_vals [T_TILE, K]`, `topk_idx [T_TILE, K]`
3. L1-normalize → `topk_vals_norm [T_TILE, K]`
4. One-hot scatter → `mask_sb [T_TILE, E]`
5. `mask_sb × affinities` → `scattered_sb [T_TILE, E]` → HBM `ea_out [T, E]`

Steps 4–5 and writing the full `[T, E]` scatter exist solely to convert the compact `[T, K]`
result into the format the library expert kernel expects.

With a custom expert kernel, steps 4–5 can be dropped entirely. Output instead:

```
affinities_compact [T, K]   f32   ← L1-normalized, K values per token
expert_index       [T, K]   u32   ← already output
```

The expert kernel looks up `affinities_compact[token_id, slot_for_this_expert]` instead of
`ea_out[token_id, expert_id]`. This is a `[T, K]` gather vs a `[T, E]` gather — 16× fewer
affinity bytes read from HBM (20 KB vs 320 KB for T=640).

---

## Optimization 3: H-Major Token Pre-Sort (Eliminates Gather in Expert Kernel)

The router already loads `x_sb [P, NUM_H_TILES, T_local]` — tokens are in SBUF in H-major
layout. After computing `token_position_to_id`, each token's sorted destination is known.

Tokens can be written to HBM in **pre-sorted, block-aligned layout**:
`sorted_hidden [N*B, H]` where block `n` holds `B` tokens all assigned to
`block_to_expert[n]`. The expert kernel then reads blocks sequentially
(`sorted_hidden[n*B:(n+1)*B, :]`) instead of gathering from
`hidden_states[token_position_to_id[n*B:(n+1)*B], :]`.

This requires a custom expert kernel (library interface expects `hidden_states [T, H]` +
`token_position_to_id`). `token_position_to_id` degenerates to an iota (0..N*B-1) and can
be dropped.

---

## What a Fully Custom Fused CTE MoE Kernel Would Look Like

```python
@nki.jit(platform_target="trn2")
def qwen3_cte_moe_fused(
    x,              # [H, T]      bf16  — H-major (reuses router layout)
    router_w,       # [H, E]      bf16  — pre-transposed
    gate_up_w,      # [E, H, 2*I] bf16
    down_w,         # [E, I, H]   bf16
):
    # Phase 1 (router):   logits → softmax → top-K → L1-norm  [all SBUF]
    # Phase 2 (sort):     histogram + prefix-sum → token_position_to_id in SBUF
    # Phase 3 (permute):  write sorted_hidden [N*B, H] to HBM once
    # Phase 4 (experts):  per-block [B, H] @ [H, 2I] → SiLU → [B, I] @ [I, H]
    # Phase 5 (scatter):  weighted accumulate back to output [T, H]
```

### Why Phase 3 still needs HBM

Expert weights at E=128, I=768 (Qwen3-30B-A3B intermediate after TP=4):
`gate_up = 128 × 2048 × 1536 × 2B = 512MB`. This dominates all other memory traffic.
The kernel is weight-load bound regardless of token layout. The HBM bandwidth savings from
eliminating the router scatter are real but second-order.

With LNC=2, the library's `shard_hidden` variant shards H across both cores, halving weight
loads per core. Any custom fused kernel needs to replicate this sharding.

### Where fusion actually helps

The expert matmul time is roughly fixed by weight-load bandwidth. The optimization budget is
in the metadata traffic between router and expert: `ea_out [T, E]` at 320KB,
`token_position_to_id [N*B]` at 21KB, and the index computation kernel launch. For a
20–30 µs expert kernel, these round-trips are a measurable fraction of total MoE latency.

---

## Optimization Roadmap

| Priority | Optimization | Complexity | Benefit | Notes |
|----------|-------------|------------|---------|-------|
| **1** | Drop `rl_out` write | Trivial | −320KB HBM write | Nothing reads logits downstream |
| **2** | Fuse token mapping into router | Medium | Eliminates separate kernel launch + index round-trip | Histogram from `mask_sb`, prefix sum, scatter `token_pos_to_id` in SBUF |
| **3** | Compact `[T, K]` affinity output | Medium (needs custom expert kernel) | 16× affinity bandwidth reduction | 320KB → 20KB, drops scatter step |
| **4** | Pre-sort tokens in router for sequential expert access | High (needs custom expert kernel) | Eliminates gather in expert kernel | Write `sorted_hidden [N*B, H]` directly |
| **5** | Full fused CTE MoE kernel | Very high | Maximum fusion, single launch | Expert weight I/O still dominates; diminishing returns |

**Highest-leverage near-term step is Priority 2.** The scatter and `ea_out [T, E]` output
are the correct format for the library kernel and should be kept. But computing
`block_to_expert` and `token_position_to_id` inside the router — using `mask_sb` and
`topk_idx` already in SBUF — eliminates the entire separate index computation step (whether
PyTorch or the NKI `find_nonzero_indices` / `indexed_flatten` path).

---

## File References

| File | Relevant Location |
|------|-------------------|
| `kernels/router_topk/qwen3_router_topk_plan_a.py` | `qwen3_router_topk_cte()` — `mask_sb` at line 330, `topk_idx_sb` at line 279 |
| `neuronx_distributed/modules/moe/blockwise.py` | `BlockwiseMatmulNKIFunc.forward()` line 445; kernel dispatch lines 86–104 |
| `neuronx_distributed/modules/moe/expert_mlps_v2.py` | `forward_blockwise()` line 692; `get_blockwise_expert_and_token_mapping()` line 1206; `get_blockwise_expert_and_token_mapping_kernel()` line 1080 |
| `neuronx_distributed/modules/moe/routing.py` | `RouterTopK.forward()` line 192 |
| `qwen_with_router_nki.py` | `NKIRouterTopK` class; `BlockwiseMatmulConfig` setup with `normalize_top_k_affinities=True` |
