# DeltaNet Prefill (CTE) — Chunked Forward Implementation Notes

Design rationale for `deltanet/prefill/chunked_fused.py` and `deltanet/prefill/chunked_step.py`.

---

## 1. Which path is live

`chunked_step.py` is the **default prefill path**.

`chunked_fused.py` is faster but **gated OFF for this checkpoint**: it uses the split decay form
`exp(gc[i]) * exp(-gc[j])`, which overflows fp32 at this checkpoint's gating magnitude and yields
NaN logits. `chunked_step.py` instead uses a stable triangular solve for the intra-chunk
correction.

Keep the fused path around: it is correct for checkpoints with smaller gating magnitude, and the
overflow is a numerics issue in the decay factorization, not a bug in the chunk algebra.

---

## 2. Fused-kernel optimizations over the per-chunk-step kernel

SSD-style architecture: processes ALL chunks for one (batch, head) pair in a single NKI kernel call.

1. **One kernel call per (B, H)** instead of `B*H*num_chunks` calls.
2. **State resident in SBUF across all chunks** — no HBM state read/write per chunk, so no
   round-trip for inter-chunk state propagation.
3. **In-kernel cumsum** via `tensor_tensor_scan`, instead of a PyTorch cumsum on the host.
4. **Masks and constants loaded once** and reused across chunks.
5. **`tensor_scalar` for partition-broadcast**, removing the explicit broadcast loops.
6. **`nc_transpose` (Vector engine) for the 128x128 transposes** instead of
   `nc_matmul(moving=eye)` (Tensor engine), which frees the TE for actual math.

---

## 3. Why `chunked_step.py` takes full tiles

No sequence-indexed DMA inside the kernel — all inputs and outputs are full 128x128 tiles, and the
caller loops over chunks in PyTorch, passing state between calls.

This avoids a DMA out-of-bounds issue seen with `nl.sequential_range` plus slice indexing in the
NxDI model-compilation context. Changing this to an in-kernel chunk loop means re-testing that
path specifically.

---

## 4. Shared mathematical framework

Chunk size = `k_dim` = `v_dim` = 128 = `P_MAX`, one tile per chunk.

Per-chunk Neumann-series power-doubling for the intra-chunk correction:

```
A          = -QK_decay * lower_mask
N          = (I+A)(I+A^2)(I+A^4)...(I+A^64)     6 rounds
value_corr = N @ v_beta
k_cumdecay = N @ (k_beta * exp(gc))
```

Inter-chunk state propagation:

```
v_prime    = k_cumdecay @ state
v_new      = value_corr - v_prime
attn_inter = (q * exp(gc)) @ state
attn_intra = (q @ k^T) * decay_mask * lower_mask_diag
output     = attn_inter + attn_intra @ v_new
state      = exp(g_last) * (state + k_raw_decay^T @ v_new)
```

The 6 power-doubling rounds cover a 128-wide chunk: `2^6 = 64`, and `A` is strictly lower
triangular so `A^128 = 0`; 6 rounds reach the full nilpotent order for this chunk size.
