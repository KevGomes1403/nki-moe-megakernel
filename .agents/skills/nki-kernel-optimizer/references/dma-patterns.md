# DMA Optimization Patterns

Source: `nkilib/core/attention/attention_cte.py`, `attention_tkg.py`, `mlp_tkg_gate_up_projection.py`, `tp_broadcast.py`

---

## `dma_transpose` vs `nc_transpose`

These are not interchangeable — they operate on different memory spaces.

| | `nisa.dma_transpose` | `nisa.nc_transpose` |
|---|---|---|
| Source | HBM | SBUF |
| Destination | SBUF | PSUM |
| Cost | Free (DMA engine) | Tensor engine cycles |
| When to use | Transpose during HBM→SBUF load | SBUF↔PSUM moves |

**Always prefer `dma_transpose` when the source is in HBM.** It avoids an intermediate SBUF allocation and uses the DMA engine instead of the tensor engine.

```python
# Good: transpose happens in DMA engine during load — no separate pass, no extra SBUF
nisa.dma_transpose(src=weight_hbm[...], dst=weight_sb.ap(pattern=[...]))

# Costly: load first, then burn tensor engine cycles for transpose
nisa.dma_copy(src=weight_hbm[...], dst=tmp_sb[...])
nisa.nc_transpose(data=tmp_sb[...], dst=result_psum[...])
```

---

## Address Patterns (`.ap()`)

`.ap()` generates a logical view of a tensor using a stride pattern. The pattern encodes `[[stride_p, size_p], [stride_f, size_f]]` plus an offset.

### Strided load pattern

```python
# Load contiguous 128-element chunks from a 2D tensor:
# hidden_col.ap([[1, PMAX], [PMAX, num_h_tiles]], offset=0)
# → partition stride=1 (contiguous), free stride=128 (step between tiles)
hidden_all = hidden_col.ap([[1, PMAX], [PMAX, num_h_tiles]], offset=0)
```

### KV cache transpose-on-load

```python
# K cache: [d, s_prior] → [PMAX, PMAX] tiles on-the-fly during DMA
K_tile = K_cache.ap([[1, PMAX], [d, PMAX]], offset=s_t * PMAX * d)

# V cache: different stride order for sequential (non-transposed) load
V_tile = V_cache.ap([[d, PMAX], [1, d]], offset=s_t * PMAX * d)
```

### Stride-0 broadcast

Setting `stride_f=0` makes every free-dim element read the same partition value:

```python
# ap([[f_dim, p_dim], [0, broadcast_dim]]):
# indexed[p, f] = flat[p * f_dim + f * 0] = flat[p]
# → partition p value replicated across all broadcast_dim free columns
src.ap([[f_dim, p_dim], [0, broadcast_dim]], offset=src_offset)
```

This is the key building block for `tp_broadcast` (see below).

### Dynamic scalar offset

When the offset is runtime-determined, use `scalar_offset` to avoid baking in absolute addresses that break end-to-end compilation:

```python
# ind_offset is a uint32 SBUF tensor computed at runtime
tensor.ap(pattern=[[stride_p, size_p], [stride_f, size_f]],
          scalar_offset=ind_offset)
```

---

## Scalar Broadcast (`stream_shuffle_broadcast`)

When a scalar must be broadcast to all partitions, avoid repeated `.ap()` calls that serialize DMA streams. Load once to SBUF `[0, 0]`, then broadcast:

```python
# Single scalar DMA — one stream
nisa.dma_copy(dst=sink_sb[0, 0], src=sink[batch_id, 0])
# Broadcast to all partitions in SBUF — no HBM re-read
stream_shuffle_broadcast(src=sink_sb, dst=sink_sb)
```

This replaces ~128 serialized `.ap()` calls (one per partition) with 2 instructions.

---

## TP Broadcast Without HBM (`tp_broadcast`)

Source: `nkilib/core/utils/tp_broadcast.py`

Broadcast `[GQA=8, 1]` → `[PMAX=128, GQA=8]` entirely in SBUF+PSUM, no HBM round-trip.

```python
def tp_broadcast(src, dst, src_offset, psum_address=None):
    p_dim, f_dim = src.shape           # e.g., GQA=8, 1
    broadcast_dim, tp_dim = dst.shape  # e.g., PMAX=128, GQA=8

    tp_psum = nl.ndarray((broadcast_dim, tp_dim), nl.float32,
                          buffer=nl.psum, address=psum_address)

    # stride_f=0 broadcasts partition p value across all broadcast_dim free cols
    nisa.nc_transpose(
        tp_psum[...],
        src.ap([[f_dim, p_dim], [0, broadcast_dim]], offset=src_offset)
    )
    nisa.tensor_copy(dst[0:broadcast_dim, 0:tp_dim], src=tp_psum)
```

**2 instructions** instead of ~130 (128-loop + transpose + tensor_copy). Used in nkilib for GQA broadcasting of max/sum accumulators.

For the reverse direction `[PMAX=128, 1] → [PMAX=128, GQA=8]` (cos/sin/qnw/k_rope): 2 `nc_transpose` + 2 `tensor_copy` = 4 instructions vs 8 `tensor_copy` calls.

---

## Weight Ring Buffer (Double-Buffering)

To hide DMA latency, allocate N weight tile slots and cycle through them while overlapping DMA load with compute on the previous slot.

```python
# Allocate N slots
weight_tiles = [
    sbm.alloc_stack((H0, I), name=f"gate_w_{i}", dtype=fp8_dtype)
    for i in range(tiles.num_allocated_w_tile)
]

# Ring buffer: load slot N while computing with slot N-1
for i_tile in nl.affine_range(n_I_tiles):
    slot = i_tile % tiles.num_allocated_w_tile
    nisa.dma_copy(dst=weight_tiles[slot][...], src=weight_hbm[i_tile, ...])
    nisa.nc_matmul(..., stationary=weight_tiles[(slot - 1) % N][...], ...)
```

---

## Weight Row Hoisting

Load the entire weight row once, then tile against hidden in 128-element chunks. Avoids re-issuing weight DMA per hidden tile.

```python
# Load full Wk/Wv row [d=128, H=2048] into SBUF once
nisa.dma_copy(dst=wk_full[0:PMAX, 0:H], src=Wk[0:PMAX, 0:H])

# Tile loop: reuse wk_full, vary hidden slice
for h_t in nl.affine_range(num_h_tiles):   # num_h_tiles = H // PMAX = 16
    nisa.nc_matmul(
        k_psum,
        stationary=wk_full[0:PMAX, h_t*PMAX:(h_t+1)*PMAX],
        moving=h_all[0:PMAX, h_t:h_t+1],
    )
```

---

## Wo Contiguous DMA (Output Projection)

The output projection weight `Wo` is typically stored as `[Hq_out, H_wo]`. Reshaping and using a strided AP pattern yields 50% DMA fill ratio vs 12.5% for the naive scatter pattern.

```python
# Reshape [Hq_out=1024, H_wo=2048] → [Hq_tp=8, d=128, H_wo=2048]
# AP: [[H_wo, PMAX], [1, H_wo]]
# → 128 chunks × H_wo bytes, 50% fill ratio
# vs old scatter: 2048 chunks × 256B, 12.5% fill ratio (16× more packets)
wo_tile = Wo.reshape((Hq_tp, d, H_wo)).ap([[H_wo, PMAX], [1, H_wo]])
```

---

## DMA Transpose for Scale Loading

When row-wise scales are stored transposed in HBM (`[H, I]`), use `dma_transpose` to load them into `[T, 1]` SBUF layout without an intermediate allocation:

```python
dequant_scale_view = dequant_scale.slice(...).reshape_dim(dim=0, shape=(num_128_I_tiles, I0))
nisa.dma_transpose(
    src=dequant_scale_view.slice(dim=0, start=0, end=num_128_I_tiles).get_view(),
    dst=dequant_tile.ap(pattern=[
        [dequant_tile.shape[1], I0],
        [1, 1], [1, 1], [1, num_128_I_tiles],
    ]),
)
```

---

## PSUM Bank Interleaving

Assign accumulator tiles to different PSUM banks by using modulo-4 addressing. Eliminates read-after-write stalls when multiple matmul tiles are in flight.

```python
# 4-way bank interleaving: each K tile lands in a different bank
res_psum = nl.ndarray((T, H1), dtype=nl.float32, buffer=nl.psum,
                       address=(0, (k_tile_idx % 4) * PSUM_BANK_SIZE))
```

Use this whenever you have multiple accumulator tiles live simultaneously (e.g. K tile loop in attention, I tile loop in MLP).

---

## `affine_range` vs `range()` for DMA

- **`nl.affine_range`**: for DMA loads that are independent across iterations — compiler can issue them in parallel
- **`range()`** / sequential loop: when iterations have data dependencies (e.g. online softmax, flash attention sections)

Using `affine_range` on independent DMA loads lets the compiler overlap multiple HBM reads with compute. Using it where there are real dependencies produces incorrect results.
