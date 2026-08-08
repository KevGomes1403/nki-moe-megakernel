# Fused MoE Layer — SBUF-Resident TKG Building Block (Spec & Plan)

Status: **PROPOSED — not yet implemented.**
Scope: the whole post-attention FFN of every Qwen3.6-A3B decoder layer — today
`post_attention_layernorm` (PyTorch, `modeling_qwen36_a3b.py:2450,2426`) feeding
`NeuronMoEBlock` (`modeling_qwen36_a3b.py:2199-2256`). Goal: run the post-attn RMSNorm so its
output **persists in SBUF (no HBM round-trip)**, then router + routed experts + the always-on
sigmoid-gated shared expert, producing the layer's MoE output with a **single TP all-reduce**.
Standalone isolation-testable now; later an SBUF-only megakernel composable next to the
GQA/DeltaNet attention blocks.

**Verdict up front:** compose four `nkilib` sub-kernels directly —
`rmsnorm_tkg` + `router_topk` + `moe_tkg`(selective) + `mlp_tkg`(shared expert) — plus a tiny
hand-written σ-gate and a gated-sum. **Do NOT use `moe_block/moe_block_tkg.py`**: its shared
expert is a literal `# TODO / pass` (`moe_block_tkg.py:384-386`), its utils hard-assert shared
expert unsupported (`moe_block_tkg_utils.py:230-231`), it pins LNC-2 (`:263`), and it routes the
normed hidden through HBM (`moe_block_tkg.py:254,285` `nl.shared_hbm`) — exactly the round-trip
we want to kill. Every primitive it would call is reusable on its own.

## Config (verified against `modeling_qwen36_a3b.py:1610-1653`, not the HTML)
H=2048, E=256, K=8, `moe_intermediate_size`=512, `shared_expert_intermediate_size`=512,
`norm_topk_prob`=True, router fp32 **softmax over E → top-8 → L1-normalize** (`:1649-1653`).
TP=4 ⇒ routed **I/rank=128** and shared **I_s/rank=128** (both intermediates sharded on I; EP=1,
all 256 experts replicated per rank, `:1644`). Router weight is **rank-replicated** (full [E,H]).
HTML §4.1 dims cross-check clean. bs=1, T∈{1,2}, bf16.

---

## 1. Contract

**Inputs** (per rank, TP=4):
| name | shape | dtype | layout / buffer |
|---|---|---|---|
| `hidden` (raw post-attn residual) | `[B,S,H]`=[1,T,2048] | bf16 | HBM (isolation) **or** SBUF `[H0,T,H1]` (megakernel). **This IS the residual — must survive untouched.** |
| `gamma` (post_attn_layernorm.weight, +1-folded) | `[1,H]` | fp32 | HBM |
| `router_w` | `[H,E]`=[2048,256] | bf16 | HBM, rank-replicated (load-transposed from stored `[E,H]`) |
| `expert_gate_up_w` | `[E,H,2,I]`=[256,2048,2,128] | bf16 | HBM (already converted, `:3506-3512`) |
| `expert_down_w` | `[E,I,H]`=[256,128,2048] | bf16 | HBM (already converted) |
| `shared_gate_up_w` | `[H,2,I_s]`=[2048,2,128] | bf16 | HBM (load-repacked) |
| `shared_down_w` | `[I_s,H]`=[128,2048] | bf16 | HBM (load-repacked) |
| `shared_gate_w` (σ-gate) | `[H,1]`=[2048,1] | bf16 | HBM, rank-replicated (load-transposed from `[1,H]`) |
| `eps` | scalar | — | — |

**Output:** `moe_out` for the residual add — `[B,S,H]` (isolation) or SBUF `[H0,T,H1]`
(megakernel). The caller does `hidden_states = residual + moe_out`.

**All-reduce boundary:** the block produces the **per-rank partial** `routed_local + gate*shared_local`.
- Isolation: return the partial; caller applies torch `reduce_from_tensor_model_parallel_region`
  (preserves the existing `_defer_moe_allreduce` identity, `modeling:2185-2196,2244-2256`).
- Megakernel: in-kernel `nki.collectives.all_reduce` on the SBUF tile (the `transformer_tkg` idiom).

Single all-reduce only — valid because the σ-gate is rank-replicated:
`AR(routed) + gate·AR(shared) == AR(routed + gate·shared)`.

---

## 2. Stage dataflow (SBUF-resident throughout)

```
hidden [B,S,H] / [H0,T,H1]
   │  rmsnorm_tkg(gamma, output_in_sbuf)              ← NORM ONCE
   ▼
normed_sb [H0,T,H1]  (H0=128, H1=H/128 = 16; LNC-2 ⇒ H1_shard=8)   ── shared by all 3 consumers
   ├───────────────┬────────────────────────────────────────┐
   ▼               ▼                                          ▼
router_topk      moe_tkg(is_all_expert=False)            mlp_tkg(shared, NO_NORM, SiLU)
 → affinities[T,E](sb)                                    → shared_local [H0,T,H1] (sb)
 → index[T,K](sb)   → routed_local [H0,T,H1] (sb)
 → eager[T,K](sb) ──────────┘ (POST_SCALE)
   │
   ▼  σ-gate:  g = sigmoid( normed_sb ·ₕ shared_gate_w )  → [T,1]   (tiny H→1 matmul)
   ▼
combined_sb[H0,T,H1] = routed_local + broadcast(g) * shared_local
   ▼  all_reduce  (torch in isolation | nki.collectives in megakernel)
moe_out [H0,T,H1] / [B,S,H]
```

**Every tile's layout:**
- `normed_sb` **`[H0,T,H1]`** — the single chosen intermediate layout. Accepted with **ZERO reshape** by:
  router_topk (`x_sb_layout`, a permutation of `[128,T,H/128]`, `router_topk.py:98,109`), moe_tkg
  (`hidden_input` SBUF `[H0,T,H1]`, `moe_tkg.py:103`), and the shared gate_up matmul (contraction-first:
  H on the partition axis, which is what `process_gate_up_projection` wants). `rmsnorm_tkg` emits exactly
  this transposed layout for "efficient downstream sharded matmul" (`rmsnorm_tkg.py:89,187,206`).
- `routed_local`, `shared_local`, `combined_sb`, `moe_out` — all `[H0,T,H1]` (moe_tkg/mlp_tkg with
  `output_in_sbuf=True` return `hidden_input.shape`, `moe_tkg.py:303-304`).
- **The one unavoidable wrinkle:** the σ-gate `g` is per-token `[T,1]`, but the gated sum lives in
  `[H0,T,H1]` where T is the *middle* axis. `g` must broadcast across both H0 (partition) and H1 (free)
  while indexed by T — materialize `g` as a `[H0,T,1]`-broadcastable tile (one partition-broadcast copy)
  then `tensor_tensor` multiply with free-axis broadcast on H1. A few cheap ops; no HBM, no transpose.
- No other transpose anywhere in the SBUF path.

---

## 3. nkilib reuse table

| kernel `file:line` | verdict | what the caller adds |
|---|---|---|
| `subkernels/rmsnorm_tkg.py:45` `rmsnorm_tkg` | **✓ drop-in** | pass `gamma=[1,H]`, request SBUF out → `normed_sb [H0,T,H1]`. Already the chosen `pre_attn_rmsnorm.md` reuse, mirror it for post-attn. |
| `router_topk/router_topk.py:57` `router_topk` | **✓ drop-in (args)** | `act_fn=SOFTMAX`, `router_pre_norm=True` (softmax-before-topk = ACT1), `norm_topk_prob=True` (L1) → **exactly matches** `modeling:1649-1653`. `x`=`normed_sb` (set `x_sb_layout` to its permutation), `w`=`router_w[H,E]`, `k=8`. SBUF outputs. Set `return_eager_affi=True` for moe_tkg's `expert_affinities_eager`. Pseudocode `router_topk.py:152-167` confirms order. |
| `moe/moe_tkg/moe_tkg.py:51` `moe_tkg` | **✓ drop-in (args)** | `is_all_expert=False` (selective top-8 of 256), `hidden_input=normed_sb`, `output_in_sbuf=True`, `activation_fn=SiLU`, `expert_affinities_scaling_mode=POST_SCALE`, `expert_affinities_eager=eager`. Weight layouts `[E,H,2,I]`/`[E,I,H]` already match the checkpoint conversion. Internally `NormType.NO_NORM` (`moe_tkg.py:252`) ⇒ consumes pre-normed input — confirms norm-once. → `routed_local`. |
| `mlp/mlp_tkg/mlp_tkg.py:304` `mlp_tkg` | **⚠ adapt = shared expert** | This is the "reuse a non-selective MLP kernel" answer (Q2b) — **no from-scratch matmul chain**. Feed `normed_sb` (`input_in_sbuf=True`), `NormType.NO_NORM`, `SiLU`, `store_output_in_sbuf=True`. Needs shared weights repacked contraction-first (§5). → `shared_local`. |
| `moe_block/moe_block_tkg.py:41` `moe_block_tkg` | **✗ do not use** | shared expert `# TODO/pass` (`:384-386`), utils assert rejects it (`moe_block_tkg_utils.py:230-231`), LNC-2 pin (`:263`), HBM norm round-trip (`:254,285`). Compose the four kernels above instead. |
| σ-gate (`sigmoid(H→1)`) | **✗ gap (tiny custom)** | `nisa.nc_matmul(normed_sb, shared_gate_w[H,1])` → `[T,1]`, then `nl.sigmoid`. ~one matmul; not worth a wrapper. |
| gated sum + AR | **caller glue** | broadcast-multiply + add in `[H0,T,H1]` (§2 wrinkle); then torch AR (isolation) / `nki.collectives.all_reduce` (megakernel). |

`return`s and SBUF-vs-HBM auto-detection: router_topk and moe_tkg auto-detect buffer type of their
in/out tensors, so SBUF wiring is just passing SBUF `nl.ndarray`s.

---

## 4. Shared-expert build plan (the main custom piece)

The shared expert is a plain SwiGLU FFN: `down(silu(gate(x)) * up(x))`, `I_s/rank=128`
(`SharedExpertMLP`, `modeling:2259-2287`). It is **not** in the fused nkilib block — but it is just a
single-expert MLP, so **reuse `mlp_tkg`** rather than hand-writing matmuls:

1. `mlp_tkg(params)` with `params.hidden_tensor=normed_sb` (`input_in_sbuf=True`),
   `gate/up/down` = shared weights, `normalization_type=NO_NORM`, `activation_fn=SiLU`,
   `store_output_in_sbuf=True`, `shard_on_h` (T≤2 ⇒ no T-sharding, `mlp_tkg.py:325`). Down-proj returns
   the **per-rank partial** (no internal reduce) → `shared_local [H0,T,H1]`. This reuses the exact
   `process_gate_up_projection`/`process_down_projection` primitives the routed experts use.
2. **σ-gate** (custom, tiny): `g = sigmoid(normed_sb ·ₕ shared_gate_w)` — a `[H,1]` projection
   contracting over H (natural in `[H0,T,H1]`: contract the partition+free H), then `nl.sigmoid`.
   Rank-replicated, so identical on every rank. Output `[T,1]`.
3. **Gated sum:** `combined = routed_local + broadcast(g)·shared_local` (§2 broadcast), **before** the
   single AR.

Fallback if `mlp_tkg`'s separate-vs-fused gate/up handling proves awkward: a hand matmul chain via
`experimental/primitives/blas` — but `mlp_tkg` is the validated, faster choice; treat blas as plan B only.

---

## 5. Weight-layout plan (load-time repack > runtime reshape — repo precedent)

| tensor | stored (today) | kernel wants | action |
|---|---|---|---|
| router `linear_router.weight` | `[E,H]`=[256,2048] (`modeling:3502`) | `[H,E]` (`router_topk.py:99`) | **load-time transpose**, rank-replicated |
| routed `gate_up_proj` | `[E,H,2I]`→reshape `[E,H,2,I]` | `[E,H,2,I]` (`moe_tkg.py:105`) | **already done** by `_convert_moe_block` (`:3506-3512`) — no change |
| routed `down_proj` | `[E,I,H]` (`:3508,3512`) | `[E,I,H]` (`moe_tkg.py:107`) | **already done** — no change |
| shared `gate_proj`+`up_proj` | each `[I_s,H]` (ColumnParallel, `:2264-2275`) | fused `[H,2,I_s]` (contraction-first) | **load-time repack**: transpose each `[I_s,H]→[H,I_s]`, stack on the `2` axis |
| shared `down_proj` | `[H,I_s]` (RowParallel, `:2278`) | `[I_s,H]` | **load-time transpose** |
| σ `shared_expert_gate.weight` | `[1,H]` (`:2226`) | `[H,1]` | **load-time transpose**, rank-replicated |

The three shared repacks transpose the torch nn.Linear (out-first) layout into the contraction-first
layout the TKG matmuls need — exactly the DeltaNet `in_proj_fused` precedent
(`MEMORY: deltanet_weight_layout`; `modeling:3596-3608`). Add them to `_convert_moe_block`
**gated behind the kernel-path flag** (e.g. `use_moe_layer_kernel`): the repacked layout is incompatible
with the existing `ColumnParallelLinear`/`RowParallelLinear` torch path, so changing it unconditionally
breaks today's model and the checkpoint contract (requires delete+recompile+reshard). For the isolation
test, do the repack in the harness.

---

## 6. Isolation test & numeric gate; megakernel handoff

**Oracle:** CPU `NeuronMoEBlock`-equivalent (`modeling:2228-2256`) — post-attn RMSNorm → routed top-8
(fp32 softmax/normalize) + σ-gated shared expert, the gated sum returned as the per-rank partial (run
the oracle TP-sharded to match, or full + compare post-AR). Reuse `*_torch.py` refs
(`router_topk_torch.py`, `moe_tkg_torch.py`, `mlp_torch.py`) as component cross-checks only.

**Gate (repo rules — `MEMORY: no_cosine_gates`):**
- **fp32 hard gate:** `allclose` (atol≈1e-5/rtol≈1e-2) on the per-rank partial AND the post-AR output.
- **bf16:** gate on `max_abs` error vs the **measured bf16 quantization floor** (run the oracle in bf16,
  take the observed deviation as the floor); **no cosine similarity** anywhere.
- Sub-checks: router top-8 **index** exact-match + affinity allclose (routing is discrete — a wrong
  index is a hard fail, not a tolerance miss); σ-gate scalar allclose.
- T∈{1,2}, bs=1, TP=4, LNC=2, bf16. Test file `tests/test_moe_layer_kernel.py`.

**Megakernel handoff:**
- **In** (from attn building block): residual+attn_out. Keep the residual stream as SBUF `[H0,T,H1]`
  across the layer — `rmsnorm_tkg` consumes `[H0,BxS,H1]` SBUF directly (`rmsnorm_tkg.py:187`), so the
  attn block hands its `[H0,T,H1]` output straight in with **no transpose**.
- **Out** (to next layer): `moe_out [H0,T,H1]`; caller does `residual + moe_out` elementwise in
  `[H0,T,H1]`, and the next layer's input `rmsnorm_tkg` again consumes `[H0,T,H1]` — **no transpose
  across the whole layer boundary.** For isolation only, emit `[B,S,H]` (one trailing transpose, or
  `moe_tkg outp_layout=B_S_H` HBM out) to match the torch reference.

---

## 7. Open questions / risks (honest)

1. **`mlp_tkg` fused vs separate gate/up.** `process_gate_up_projection` is shared with moe_tkg, which
   passes a **fused** `[E,H,2,I]`. Need to confirm `mlp_tkg`/`MLPParameters` accept *separate* shared
   `gate`/`up` `[I,H]` views, or whether to fuse them to `[H,2,I]` at load (§5 already plans the fused
   form — low risk, but verify the exact `MLPParameters` field expectations before coding).
2. **σ-gate broadcast in `[H0,T,H1]`.** The only layout friction (§2). Confirm the cheapest primitive
   (partition-broadcast copy + free-broadcast `tensor_tensor`) vs folding `g` into the shared-expert
   intermediate `[I,T]` before down-proj (T as free there too — same broadcast, different axis). Pick by
   op count.
3. **router_topk `x_sb_layout` value.** Four SBUF layouts exist (`router_topk.py:109`, codes 0-3); must
   pick the one matching `rmsnorm_tkg`'s emitted `[H0,T,H1]` permutation exactly, else router silently
   reloads/transposes. Verify against `router_topk_input_x_load`.
4. **POST_SCALE vs eager affinities.** Confirm moe_tkg selective applies the top-k affinity weighting
   correctly with `POST_SCALE` + `expert_affinities_eager`; the affinity normalization (`norm_topk_prob`)
   already happens in router_topk, so moe_tkg must NOT re-normalize.
5. **LNC sharding under one grid.** In isolation each kernel self-sharded under LNC-2; in the megakernel
   all four run under one SPMD `n_prgs=2` grid — confirm `get_verified_program_sharding_info` composes
   (H1_shard=8, T≤2 ⇒ no token-sharding in any stage).
6. **Checkpoint contract churn.** The shared-expert repack (§5) changes stored weight shapes → must be
   flag-gated and documented as a reshard, like the DeltaNet in_proj fusion.
```
