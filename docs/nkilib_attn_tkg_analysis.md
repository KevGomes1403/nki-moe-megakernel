# nkilib attention_tkg.py — Deep Analysis for Qwen3 Shapes

Source: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/nkilib/core/attention/attention_tkg.py`
tp_broadcast utility: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/nkilib/core/utils/tp_broadcast.py`
Analyzed: 2026-03-26

---

## Target Shapes

Qwen3-30B-A3B, TP=4, B=1 (token generation):
- Hq_tp=8, Hkv_tp=1, GQA=8, d=128, H=2048
- Wq: [1024, 2048], Wk: [128, 2048], Wv: [128, 2048], Wo: [1024, 2048]
- S_prior ≈ 640 tokens, PMAX=128, F_MAX=512

---

## Section 1: Complete Optimization Techniques in nkilib

### A. QKV Projection Structure

**1.1 Horizontal Weight Tiling with Wide-Row Loads**
- Load full weight row `[d, H]` in single wide DMA, then tile against hidden in 128-sized chunks
- Wk/Wv: load `[128, 2048]` entirely into SBUF via `nisa.dma_copy(dst=wk_full, src=Wk)`
- Tiling loop: `for h_t in nl.affine_range(num_h_tiles)` with `num_h_tiles = H // PMAX = 16`
- Matmul: `nisa.nc_matmul(k_psum, stationary=wk_full[0:PMAX, h_t*PMAX:(h_t+1)*PMAX], moving=h_all[0:PMAX, h_t:h_t+1])`
- Reuses weight row across all hidden tiles; avoids repeated weight DMA

**1.2 Hidden State Hoisting**
- Pre-load entire hidden dimension outside all loops as `h_all[PMAX, num_h_tiles]`
- ap() pattern: `hidden_col.ap([[1, PMAX], [PMAX, num_h_tiles]], offset=0)`
  - Partition stride=1, free stride=128 (contiguous 128-element chunks)
- Same `h_all` reused for Wq, Wk, Wv projections without re-DMA

**1.3 Per-Head Weight Hoisting for Wq**
- Load one head per iteration via `for q_h in nl.affine_range(Hq_tp):`
- `nisa.dma_copy(dst=wq_head, src=Wq[q_h * PMAX:(q_h + 1) * PMAX, :])`
- Separate `q_psum_list[q_h]` for each head enables subsequent packing into `[PMAX, GQA]`

### B. Softmax / Flash-Decode Structure

**1.4 Two-Pass Flash Decode with Saved Scores (v8)**
- Pass 1: compute `score[PMAX, GQA]` for each K-cache tile, save to `saved_scores[s_t]`, compute per-tile max
- Pass 2: reuse saved scores (no re-matmul), compute `exp(score - global_max)`, accumulate V-weighted sum
- nkilib uses proper Flash Attention tiling for S_prior > 8K

**1.5 Cascaded Max Reduction in nkilib (lines 1808-1902)**
- Step 2.1: strided reduce from `[PMAX, tile_n_sprior * s_active_bqh]` → `[PMAX, s_active_bqh]`
- Step 2.2: nc_transpose to PSUM `[s_active_bqh_tile, PMAX]`
- Step 2.3: LNC2 sendrecv for cross-NC max reduction (GPSIMD DMA if available)
- Step 2.3.3: Final reduction with `negate=True` to save one exp negation later
- Result stored as compact `qk_max_buf[GQA=8, 2]` (local + cross-NC)

**1.6 GQA Broadcasting WITHOUT HBM — tp_broadcast (lines 1988-2021)**

**THIS IS THE KEY PATTERN.** nkilib achieves `[GQA=8, 1] → [PMAX=128, GQA=8]` entirely in SBUF+PSUM:

```python
# From tp_broadcast.py (full source):
def tp_broadcast(src, dst, src_offset, psum_address=None):
    """
    Transposes then broadcasts src[0:1, :] onto all partitions of dst.
    Uses ap() with stride_f=0 on SBUF tensor as nc_transpose source.
    No HBM intermediate buffers.
    """
    p_dim, f_dim = src.shape       # e.g., GQA=8, 1
    broadcast_dim, tp_dim = dst.shape  # e.g., PMAX=128, GQA=8

    # Allocate PSUM for transpose output
    tp_psum = nl.ndarray((broadcast_dim, tp_dim), nl.float32, buffer=nl.psum, address=psum_address)

    # ap() with stride_f=0: dst[p, f] = flat[p*f_dim + f*0] = flat[p]
    # → each partition p value broadcast across all broadcast_dim free columns
    nisa.nc_transpose(
        tp_psum[...],
        src.ap([[f_dim, p_dim], [0, broadcast_dim]], offset=src_offset)
    )

    nisa.tensor_copy(dst[0:broadcast_dim, 0:tp_dim], src=tp_psum)
```

**How it works:**
- `src[GQA=8, 1]` with ap `[[1, GQA], [0, PMAX]]`:
  - `indexed[p, f] = flat[p*1 + f*0] = flat[p]` → partition p value broadcast to all PMAX free cols
  - Produces a `[GQA=8, PMAX=128]` view of the data
- `nc_transpose` flips `[GQA=8, PMAX=128]` → `psum[PMAX=128, GQA=8]`
- `tensor_copy` → SBUF `neg_max[PMAX=128, GQA=8]`

Result: **2 instructions instead of 128-loop + nc_transpose + tensor_copy (~130 instructions)**

For the reverse direction `[PMAX=128, 1] → [PMAX=128, GQA=8]` (cos/sin/qnw/k_rope):
- ap `[[1, PMAX], [0, GQA]]` on `[PMAX=128, 1]` → `[PMAX=128, GQA=8]` indexed view
- nc_transpose: input `[PMAX=128, GQA=8]` → psum `[GQA=8, PMAX=128]` (transposed form)
- Second nc_transpose: `[GQA=8, PMAX=128]` → `[PMAX=128, GQA=8]` (final shape)
- Total: 2 nc_transposes + 2 tensor_copies = 4 instructions vs 8 tensor_copies

### C. K/V Cache Loading

**1.8 K-Cache DMA Transpose Pattern**
- ap() pattern: `K_cache_2d.ap([[1, PMAX], [d, PMAX]], offset=s_t * PMAX * d)`
- DMA converts `[d, s_prior]` layout to `[PMAX, PMAX]` tiles on-the-fly
- Tiles hoisted outside both passes; reused in Pass 1 (matmul) and Pass 2 (score saved)

**1.9 V-Cache Sequential Load**
- ap() pattern: `V_cache_2d.ap([[d, PMAX], [1, d]], offset=s_t * PMAX * d)`
- Different stride order for contiguous sequential DMA vs K's transpose

**1.10 nkilib K_prior — Three Paths**
- Path A: Block KV cache with indirect DMA via `vector_offset` for paged attention
- Path B: Pre-transposed K_prior — flat load, no transpose
- Path C: Non-transposed — DMA-transpose or PE-transpose

### D. Output Projection

**1.12 Contiguous DMA for Wo (v8's key innovation)**
- Wo passed as `[Hq_out=1024, H_wo=2048]` (transposed by caller)
- Reshape to `[Hq_tp=8, d=128, H_wo=2048]`
- ap() pattern: `[[H_wo, PMAX], [1, H_wo]]` → 128 chunks × 4096B, 50% fill ratio
- vs old scatter pattern: 2048 chunks × 256B, 12.5% fill ratio → 16× fewer DMA packets

**1.13 Output Matmul — Current v8 Structure (suboptimal)**
- PSUM `[1, F_MAX=512]` per block, 4 h_blk blocks × 8 heads unrolled = 32 nc_matmuls
- stationary=`attn_out[PMAX, 1]` (tiny), moving=`wo_tile[PMAX, F_MAX]` (large)
- 4 separate tensor_copy + 4 DMA stores

### E. Normalization

**1.14 Reciprocal via Rsqrt Trick**
- `1/x = rsqrt(x)^2` — avoids native divide, uses vector activation path
- Both v8 and nkilib use this

**1.15 nisa.activation() with bias and scale**
- Signature: `nisa.activation(dst, op, data, bias=..., scale=...)`
- Example from nkilib line 1073-1078:
  ```python
  nisa.activation(fa_correction, nl.exp, prev_running_max,
                  bias=fa_running_max, scale=-1.0)
  # Computes: exp(prev_running_max - fa_running_max) in one ScalarE pass
  ```
- Fuses subtract + exp (or add + rsqrt, etc.) into single instruction

**1.16 Matmul-Based Row Sum**
- `nc_matmul(rms_ones[PMAX, PMAX] @ data[PMAX, GQA])` for softmax denominator
- Single instruction vs explicit loop

### F. SBUF/PSUM Management

**1.17 SbufManager Scoping**
- `sbm.open_scope()` / `sbm.close_scope()` for nested lifetime management
- Bank-aware PSUM allocation: `address=(0, (i % psum_b_max) * psum_f_max_bytes)`

**1.25 PSUM Free Dim Budget for our shapes**
- `s_active_bqh = 1 * 8 * 1 = 8` (B=1, Hq_tp=8, s_active=1)
- MM1 group size 4K: `(4096/128) * 8 = 256 < 512` ✓ (fits in PSUM free dim budget)
- QK buffer: `[PMAX=128, 3*8=24]` = 12KB SBUF (for S_prior/NC=320, 3 tiles of 128)

---

## Section 2: nkilib vs v8 Comparison

| Feature | nkilib | v8 |
|---|---|---|
| Flash Attention tiling | ✓ (for S_prior > 8K) | ✗ single-tile (works for S≤8K) |
| LNC2 sharding (sprior) | ✓ splits S across 2 NCs | ✗ single NC |
| FP8 KV cache | ✓ | ✗ BF16 only |
| Block/paged KV cache | ✓ indirect DMA | ✗ dense only |
| tp_broadcast (SBUF-only) | ✓ | ✗ uses 128-loop |
| Contiguous Wo DMA | ✓ (v8 improved this) | ✓ |
| Saved scores (Plan B) | ✗ recomputes per tile | ✓ |
| Static GQA=8 unrolling | ✗ generic loops | ✓ |
| activation(bias=, scale=) | ✓ fuses ops | ✗ separate tensor_scalar + activation |

### Gaps in v8 vs nkilib (not critical for S_prior=640)
1. No Flash Attention — safe up to S≤8K, fails beyond
2. No LNC2 sharding — single NC, half available memory bandwidth
3. No block KV — cannot handle paged/sparse caches
4. No FP8 support

---

## Section 3: Qwen3 Shape Specifics in nkilib Code Paths

```
B=1, S_prior=640, Hq_tp=8, Hkv_tp=1, GQA=8, d=128, H=2048

LNC2 Sharding Decision:
  s_prior_sharded = True (640 >= 256 and s_active_bqh=8 <= 128)
  sprior_n_prgs = 2 → each NC processes 320 tokens

Per-NC tensor shapes:
  q_sb[128, 8]           — Q for all 8 heads, B=1
  k_prior[128, 320]      — K cache, half of 640
  v_prior[128, 320]      — V cache, half of 640
  qk[128, 24]            — QK^T (3 tiles × 8 heads)
  qk_max_buf[8, 2]       — compact max (local + cross-NC)
  neg_max[128, 8]        — broadcast form for exp subtraction

PSUM budget:
  MM1 group: 4K → 32 * 8 = 256 free dim elements < 512 ✓
  QK PSUM: [128, 24] = 12KB
  Wo output PSUM: [1, 512] per block = 2KB

SBUF estimate per NC:
  K/V cache: 320 * 128 * 2 * 2 = 163KB
  Wo matrix: 8 * 128 * 2048 * 2 = 4MB
  Attention intermediates: ~50KB
  Total: ~4.2MB (fits in Trainium2 SBUF ≈ 16MB)
```

---

## Section 4: tp_broadcast — Exact Mechanism

```python
# File: nkilib/core/utils/tp_broadcast.py

def tp_broadcast(src, dst, src_offset, psum_address=None):
    """
    src: [P, F] SBUF tensor (e.g. [GQA=8, 1] or [PMAX=128, 1])
    dst: [broadcast_dim, tp_dim] SBUF tensor where tp_dim == P

    Result: dst[b, p] = src[p, src_offset] for all b
    i.e., each partition's value is broadcast across all broadcast_dim free cols
    """
    p_dim, f_dim = src.shape
    broadcast_dim, tp_dim = dst.shape

    tp_psum = nl.ndarray((broadcast_dim, tp_dim), nl.float32, buffer=nl.psum, address=psum_address)

    # ap() stride_f=0 → broadcasts partition values across free dim:
    # indexed[p, f] = flat[p * f_dim + f * 0] = flat[p * f_dim + src_offset]
    nisa.nc_transpose(
        tp_psum[...],
        src.ap([[f_dim, p_dim], [0, broadcast_dim]], offset=src_offset)
    )
    nisa.tensor_copy(dst[0:broadcast_dim, 0:tp_dim], src=tp_psum)
```

### Application to v8 Broadcasts

| Broadcast needed | src shape | dst shape | Pattern | Instructions saved |
|---|---|---|---|---|
| neg_max_g1 → neg_max | [GQA=8, 1] | [PMAX=128, GQA=8] | ap([[1,GQA],[0,PMAX]]) + nc_transpose | ~128 (128-loop → 2) |
| cos_f32 → cos_gqa | [PMAX=128, 1] | [PMAX=128, GQA=8] | ap([[1,PMAX],[0,GQA]]) + 2×nc_transpose | 4 (8-loop → 4) |
| sin_f32 → sin_gqa | [PMAX=128, 1] | [PMAX=128, GQA=8] | same | 4 |
| qnw_sb → qnw_gqa | [PMAX=128, 1] | [PMAX=128, GQA=8] | same | 4 |
| k_rope → k_rope_packed | [PMAX=128, 1] | [PMAX=128, GQA=8] | same | 4 |
| v_active → v_act_packed | [PMAX=128, 1] | [PMAX=128, GQA=8] | same | 4 |

### Two-Step for [PMAX, 1] → [PMAX, GQA] (cos/sin/qnw/k_rope/v_act)

```python
# Step 1: ap(stride_f=0) + nc_transpose → [GQA=8, PMAX=128] (transposed intermediate)
psum_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.psum)
nisa.nc_transpose(psum_T, src[PMAX, 1].ap([[1, PMAX], [0, GQA]], offset=0))
sbuf_T = nl.ndarray((GQA, PMAX), dtype=nl.float32, buffer=nl.sbuf)
nisa.tensor_copy(sbuf_T, psum_T)

# Step 2: nc_transpose [GQA=8, PMAX=128] → [PMAX=128, GQA=8]
psum_final = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.psum)
nisa.nc_transpose(psum_final, sbuf_T)
dst = nl.ndarray((PMAX, GQA), dtype=nl.float32, buffer=nl.sbuf)
nisa.tensor_copy(dst, psum_final)
```

---

## Section 5: Additional Optimization Opportunities from nkilib

### nisa.activation with bias parameter
Fuses `tensor_scalar(add/subtract) + activation` into a single ScalarE pass:
```python
# Instead of:
sum_safe = nisa.tensor_scalar(sum_acc, op0=nl.add, operand0=1e-9)
rsqrt_sum = nisa.activation(op=nl.rsqrt, data=sum_safe)

# Use:
rsqrt_sum = nisa.activation(op=nl.rsqrt, data=sum_acc, bias=1e-9)
# Computes: rsqrt(sum_acc + 1e-9) in a single instruction
```

Same applies to Q/K RMSNorm:
```python
# Instead of:
q_mean_sq = nisa.tensor_scalar(q_sum_sb, op0=nl.multiply, operand0=1.0/d, op1=nl.add, operand1=EPS)
q_rms_inv = nisa.activation(op=nl.rsqrt, data=q_mean_sq)

# Use:
q_rms_inv = nisa.activation(op=nl.rsqrt, data=q_sum_sb, scale=1.0/d, bias=EPS)
# Computes: rsqrt(q_sum_sb * (1/d) + EPS) in a single instruction
```

### Q Projection One-at-a-Time (SBUF pressure reduction)
Current v8 hoists all 8 wq_head[128, 2048] tiles simultaneously = ~4MB SBUF.
Process one head at a time: load→matmul→copy-to-q_packed→next-head.
Reduces peak SBUF for Q weights from 4MB to 512KB.

### O-proj Loop Restructuring
Current: h_blk outer (4) × 8 heads unrolled = 32 matmuls, 4 DMA stores.
Alternative: head outer (8) × h_blk inner (4) with 4 pre-allocated PSUMs.
Better SBUF access locality; each head's wo_sbuf fully consumed before next head.
