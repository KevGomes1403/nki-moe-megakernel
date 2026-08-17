# MoE TKG (Qwen3.6-A3B) — Layout, Sharding & DMA Notes

Design rationale for `nki_kernels/moe/`. The kernels carry the dataflow and the contract; the
layout algebra, sharding decisions and DMA reasoning live here.

Status: **IMPLEMENTED & on-device validated.** TP=4, LNC=2. Per rank: `H=2048` (`H0=128`, `H1=16`),
`E=256`, `K=8`, routed `I=128`, shared `I_s=128`, EP=1 with all 256 experts replicated.

---

## 1. The tp2013 SBUF H-permutation

Every SBUF hidden tile in the MoE path is `[H0, T, H1]` in the **tp2013** permutation, which is what
the attention kernels emit for their residual — so a megakernel can share **one** SBUF residual
across attention and MoE.

With `n_s` H-shards (`n_s = n_prgs`) and `H2 = H1 // n_s`, the free index

```
f = s*H2 + h2      <->      H-column  s*(H0*H2) + h0*H2 + h2
```

At `n_s == 1` this degenerates **exactly** to tp102 (`H = h0*H1 + f`), so single-core needs no
separate path. Only the access-pattern index math into the (unmoved) HBM weights depends on the
permutation; no weight is ever repacked.

`rmsnorm_tkg` keys the emitted permutation off `lnc`, so `single_core_forced=False` gives tp2013 and
`True` forces tp102 on every core. That flag is independent of whether rmsnorm shards the BxS work
— it does not, below `SHARDING_THRESHOLD`, so at `T<=2` both cores compute the full norm.

The HBM contract (`[B,S,H]` in, `[1,T,H]` out) is layout-independent.

---

## 2. Work split vs. SBUF layout — two independent decisions

These are never welded together:

**Work split** (`moe_token_shard` / `moe_h_shard`): the experts **token-shard** across the LNC cores
whenever `cores > 1`, `T > 1`, and the config is not the big one — each core then owns a token slice
over the **full** H. When the token-shard cannot engage (`T == 1`) the cores **H-shard** instead,
each owning half the tp2013 free axis. Exactly one of the two is active at `cores > 1`. This mirrors
nkilib's `shard_on_h_disabled = shard_on_T` switch.

**SBUF layout**: always tp2013, regardless of which work split is active (see §1).

### What the H-shard costs

Core `p` owns free indices `[p*H2, (p+1)*H2)`, i.e. the contiguous H-column block
`[p*H0*H2, (p+1)*H0*H2)`:

- **gate/up** contract only their half of H. H is the *contraction* dim, so this needs **one
  cross-core reduce before the activation**. Routed: one fused `[I,2]` fp32 `sendrecv`+add reduces
  gate and up together. Shared: one fused `[I_s, 2T]` fp32 `sendrecv`+add.
- **down** writes disjoint output columns. H is the *output* dim, so **no reduce**.

The shared expert deliberately uses the same work split as the routed path, so `shared_local` keeps
the same per-core layout as `routed_local` and the gated sum is a plain per-core add.

---

## 3. Why the routed loop is raw NKI: DMA fragmentation

`moe_tkg`'s selective path loads each selected expert's gate and up weights as **two separate
strided `[H, I]` views** — `[E,H,2,I].select(dim=0,e).select(dim=1, GATE/UP)`. Because gate and up
are interleaved on the middle `2` axis, the innermost contiguous DMA run is only `I` elements:
**256 B at bf16**. That fragments the load into tens of thousands of tiny packets, leaving the sync
engine bound and the tensor engine starved.

The fix in `routed_experts_nki.py` loads each selected expert's gate+up as **one contiguous
`[H0, H1, 2I]` slab** — inner run `2I`, **512 B at bf16, half the descriptors** — then slices
`gate = slab[:, :, 0:I]` and `up = slab[:, :, I:2I]` in SBUF on the free axis.

Two further overlaps ride on that:

1. The slab load is **split per H-shard `s`**, so the gate/up matmul over the first shard can start
   before the second lands.
2. A **2-slot cross-expert prefetch ring** issues expert `k+1`'s weight DMAs while expert `k`
   computes.

This path is always on — there is no legacy or env-gated fallback. It reproduces the `moe_tkg`
selective contract exactly: top-K experts per token, SiLU activation, POST_SCALE by the
already-L1-normalized affinity with **no re-normalization**, summed over the K experts. The stored
`[E,H,2,I]` gate/up and `[E,I,H]` down weight layouts are unchanged.

---

## 4. Why the shared expert cannot reuse `mlp_tkg`

nkilib's `mlp_tkg` gate/up loader hardcodes
`weight.reshape_dim(dim=0, shape=(H0, H1_shard))` — the **tp102** H-permutation.

tp2013 needs H viewed as `(s, h0, h2)` and then **permuted** so `h0` is the partition axis.
`reshape_dim` cannot express a permutation, and no flag toggles it. Hence the shared expert is
written in raw NKI, structurally one routed expert without the expert index or affinity scale,
reusing the same AP idioms as `routed_experts_nki`.

Its weight loads:

- gate/up are **separate `[H, I_s]`** tensors, so each partition's `H2` rows coalesce into an
  `H2 * I_s` run (**2 KB at bf16**).
- down is the contiguous `[I_s, H]` row load, which is permutation-agnostic.

---

## 5. Norm once, and the single all-reduce

Per `modeling_qwen36_a3b.py`'s `NeuronMoEBlock`, all four consumers — router, routed experts, shared
expert, sigma-gate — read the **same** post-attention-normed hidden. So the norm runs once and
`normed_sb` is shared by every downstream composable, with zero HBM round-trip.

The layer returns `combined_local`, a per-rank partial, and the model applies a **single**
`reduce_from_tensor_model_parallel_region`. That is valid because the sigma-gate is
rank-replicated:

```
AR(routed) + g*AR(shared) == AR(routed + g*shared)
```

No cross-rank all-reduce happens inside the kernel (that is the megakernel/model boundary; a future
megakernel could use `nki.collectives.all_reduce` on the SBUF tile directly).

### sigma-gate

`g = sigmoid(normed .h sigma_gate_w)` is an `H->1` projection contracting the full H, which lives
split as the `H0` partition x `H1` free axes of `normed_sb`. The `[H,1]` weight is loaded through the
same tp2013 AP (`w_sb[h0, s*H2 + h2] = w[s*H0*H2 + h0*H2 + h2]`), so each free-index matmul contracts
`H0` with matching operands. The output is a `[1, T]` row (M=1), ready for the partition-broadcast in
the gated sum.

### gated sum

`g` is `[1, T]` but `T` is the **middle** axis of `[H0,T,H1]`, so it is broadcast across the `H0`
partitions with one ones-matmul (`[H0,T] = ones[1,H0].T @ g[1,T]`), then multiplied into each `H1`
slice before adding routed.

It operates on this core's free-index block only: at `cores=2 / T>1` routed and shared hold their
token slice over the full H (`f_offset=0`, `H1_local=H1`); at `cores=2 / T==1` they hold their
H-shard. Either way only the core's own slice is valid, which the caller's per-core store selects.
