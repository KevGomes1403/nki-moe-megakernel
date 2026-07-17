# Target-model verify-trunk megakernel

Status: PROPOSED — not yet implemented. Audited against installed nkilib 2026-07-11.

Fuses the 40-layer verify trunk (30 DeltaNet + 10 GQA, MoE on every layer) into a single
NKI invocation with the residual SBUF-resident and the TP all-reduces in-kernel.

Out of scope (later slices): LM head, argmax, accept, state commit, MTP draft head, CTE prefill.

---

## 1. Contract

Replaces the layer loop in `Qwen36SpecTarget._verify_forward` (`modeling_qwen36_a3b.py:4252-4271`)
for the TKG verify pass (T=2). Per rank, TP=4, LNC=2, bf16.

**In**
| Arg | Shape | Notes |
|---|---|---|
| `hidden` | `[1, T, H=2048]` | post-embedding, pre-layer-0 |
| DeltaNet weights ×30 | `proj_w [H,3088]`, `gamma [1,H]`, `conv_weight [2048,4]`, `A_log [8]`, `dt_bias [8]`, `z_gamma [128]`, `out_w [1024,H]` | layouts already correct |
| DeltaNet state ×30 | `conv_state [2048,3]`, `init_state [8,128,128]` | read-only seed |
| GQA weights ×10 | `qkv_w [H,1536]`, `gate_w [H,1024]`, `gamma_q/k [256]`, `o_proj_w [1024,H]`, `gamma_in [1,H]` | `gamma_in` now passed (norm moves in-kernel) |
| GQA cache ×10 | `k_cache [1,1,256,L]` BHDS, `v_cache [1,1,L,256]` BHSD | in-place scatter, design B |
| MoE weights ×40 | `gamma [1,H]`, `router_w [H,256]`, `expert_gate_up_w [256,H,2,128]`, `expert_down_w [256,128,H]`, `sigma_gate_w [H,1]`, `shared_{gate,up,down}_w` | layouts already correct |
| Shared | `cos [T,64]`, `sin [T,64]`, `mask [L,1,4,T]` uint8, `position_ids`, `kv_write_idx` | hoisted once, reused across the 10 GQA layers |
| kwarg | `replica_groups=([0,1,2,3],)` | **traced constant** — `nccl.ReplicaGroup` is a plain `List[List[int]]` dataclass (`nki/collectives/utils.py:8`), no `torch.distributed` in the path |

**Out**
`(Y [1,T,H], *S_cand[30], *conv_cand[30], *K_post[10], *V_post[10])`

- `Y` is pre-final-norm. NxDI applies `model.norm` + `lm_head` Python-side, as today.
- `S_cand [T,8,128,128]` fp32 / `conv_cand [T,2048,3]` are written to HBM **per layer** and consumed
  unchanged by the existing XLA `_commit_deltanet` (`:4303`).
- `K_post`/`V_post` are the post-scatter cache handles. **They must be returned or NCC dead-stores
  the in-place scatter.**

---

## 2. Body

```
rg          = nccl.ReplicaGroup(replica_groups)
residual_sb = load(hidden)              # [128, T*16] bf16, tp2013

for i in range(N):                      # N = 4 (group) → 40 (full)
    sbm.set_name_prefix(f"L{i}_")
    if layer_type[i] == "linear_attention":
        attn_sb = deltanet_layer_compose(residual_sb, W[i], S_cand[i], conv_cand[i])
    else:
        attn_sb = gqa_layer_compose(residual_sb, W[i], K[i], V[i], cos, sin, mask, kv_write_idx)
    residual_sb += sb2sb_all_reduce_gather(attn_sb, rg)        # TP AR + LNC H-gather

    moe_sb = moe_layer_compose(residual_sb, W_moe[i])
    residual_sb += sb2sb_all_reduce_gather_tokens(moe_sb, rg)  # TP AR + LNC token-gather

store(residual_sb → Y)
return (Y, *S_cand, *conv_cand, *K_post, *V_post)
```

`residual_sb` is `[128, T, 16]` **tp2013**, full-H, identical on both LNC cores — bf16, 8 KB.
Build and validate at N=4, then raise to 40.

---

## 3. Work items

### W1 · SBUF-in / SBUF-out refactor of the three composables

| Kernel | Reality today | Needed |
|---|---|---|
| `deltanet/decode/fused_layer.py::attention_layer_compose` | **SBUF-in already works** — never destructures `hidden`, forwards it to `qkv_tkg`, which sniffs `.buffer == nl.sbuf` (`qkv_tkg.py:482`). Out is HBM. | flip `OUT_IN_SB=True` (`deltanet/components/out_proj.py:104`). One line. |
| `moe/components/moe_layer.py::moe_layer_compose` | **SBUF-out already works** (`output_in_sbuf=True` is the default, `:204`). In is HBM-only. | drop the `B,S,H = hidden.shape` unpack + asserts in `post_attn_norm.py:38-79`; `rmsnorm_tkg` already accepts SBUF. |
| `gqa/decode/fused_layer.py::gqa_fused_compose` | HBM in/out **and an HBM round-trip mid-kernel** — see W1b. | the real work. |

### W1b · GQA: killing the `active_v` HBM round-trip — **DONE** (2026-07-13, all 3 GQA device tests pass)

`gqa_fused_compose` allocated `active_v` as `shared_hbm`, DMA'd the already-SBUF-resident V head into it,
then passed it **back into attention** as `v_active`. That was SBUF→HBM→SBUF in the middle of the layer,
×10 layers. `OUT_IN_SB` does not touch it.

Shipped: `v_in_sb` flag on `AttnTKGConfig` + one patch site in our vendored `attention_tkg.py`; a single
contiguous `v_active_sb [T, HEAD_DIM]` SBUF tile now feeds **both** the `active_v` HBM output DMA and
attention. `active_k`/`active_v` remain **write-only** HBM outputs — the design-A 3-tuple and design-B
5-tuple contracts and the model wiring at `:2158` are unchanged. Dropping the allocs entirely is
design-B-only and belongs with the megakernel refactor (below). Record kept for the megakernel work:

**No new vendoring needed — we already own `gqa/vendored/attention_tkg.py`** (forked for the d256 patch).
On our flat-KV config (`block_len=0` ⇒ `is_block_kv=False`), `v_active` is read in exactly **two** places:

| Site | Use | Action |
|---|---|---|
| `_compute_tile_params:835` | `v_active.dtype` (fp8 probe) | none — works on an SBUF tile |
| `_compute_pv_matmul_and_store:3394-3406` (`else` = non-block-KV) | one `nisa.dma_copy` HBM→`v_sb` bottom-right corner | **the only patch site** |
| `_setup_block_kv_cache:1028` (`v_active.reshape`) | block-KV only | never entered (`active_blocks_table=None`) |

**The patch** (add `v_in_sb: bool = False` to `AttnTKGConfig`, mirroring `qk_in_sb`/`out_in_sb`):
replace that one DMA's HBM source with the caller's SBUF `[T, d_head]` tile — which is just
`roped[0:T, V_HEAD, 0:D]`, already in SBUF at `fused_layer.py:283`.

The destination is `v_sb[p_max - s_active :, -d_head:]`, so the copy **shifts partitions**
(source partitions `0..T-1` → dest `126..127`). Compute engines are partition-aligned, so this must be
an **SBUF→SBUF `nisa.dma_copy`**, not a `tensor_copy`. That still removes both HBM hops; the residual
descriptor moves ~1 KB. (`cfg.use_gpsimd_sb2sb=True` already routes nkilib's other SB2SB copies this way.)

**Then `active_k` and `active_v` disappear entirely.** They exist only to serve (a) the design-A outputs
that NxDI scatters host-side, and (b) attention's `v_active` input. With design B on (in-place scatter,
`scatter_kv_cache_inplace:102-150`, already built and tested in `test_gqa_fused_layer_inplace_kv.py`) the
kernel does its own scatter from the SBUF `k_active_sb`/`roped`, and we return the mutated cache handles.
Delete both `shared_hbm` allocs → **zero HBM allocation inside the GQA compose.**

The `gamma_in` in-kernel input norm already exists but is unused (model pre-norms in torch, `:2520`).
Turn it on; delete the torch norm.

### W2 · Hidden-state layout contract — **DONE** (2026-07-13; 27/27 MoE tests pass, fp32 allclose(1e-5,1e-2))

Shipped: MoE runs end-to-end in tp2013 via ONE parameterized index map
`h(h0,f) = (f//H2)·H0·H2 + h0·H2 + (f%H2)`, `n_s = n_prgs`, `H2 = H1//n_s` — which **degenerates exactly to
tp102 at n_prgs=1**, so cores=1 and cores=2 share one code path with no branch. Token-sharding unchanged.
Numerically inert (layer T=2/c2: 1.922e-04 before and after). Shared expert rewritten raw-NKI.
Bonus: the cores=2/T=1 silent-wrong-answer landmine is fixed (align_T1_c2 4.285e+00 → 2.086e-05).

**Measured cost: +2.1 µs (+2.9%) on the standalone fused MoE layer** (70.95 → 73.03 µs, 3 captures each,
non-overlapping spreads). Expert-weight DMA is byte-identical and still 6144 × 4096 B — the 129→71 µs
coalescing fix survives. The +2.1 µs sits in the expert-stream (+1.71) and epilogue (+0.63) phases as added
DMA *idle*, not DMA busy (+0.6 only); the prologue got 0.3 µs *faster*. Prime suspect is the shared
expert's move from `mlp_tkg`'s hwdge path to raw-NKI SWDGE (GpSimd 20.5 → 25.7 µs/core, Sync 5.0 → 0.1),
even though that DMA itself got faster (5.11 → 4.16 µs). Not a blocker; worth one look. Note `nki-moe` hit
SWDGE scheduler contention too.

**`router_w` pre-permute: DO NOT DO IT — the prize is measured at ZERO.** The predicted descriptor split is
real and exactly as forecast (8 KB × 128 → 4 KB × 256 packets, byte-identical), but it costs **−0.02 µs**,
inside the ±0.13 µs spread: per-packet DMA throughput is already saturated at 4 KB, so halving the run
length is free. The +4.2% seen on the *isolated routed* kernel does not reproduce on the fused layer.
Scratch this idea.

Original derivation retained below.

### W2 (original) · Hidden-state layout contract

H=2048, H0=128 partitions, H1=16 free, LNC n_s=2, H2=8:

- **tp102** — `h = h0·16 + j`
- **tp2013** — free `f = s·8 + h2`, `h = s·1024 + h0·8 + h2`; an H-contraction shard is a free-slice.

**Residual contract: SBUF `[128, T, 16]`, tp2013, full-H, identical on both LNC cores.** 8 KB.

**The one real conflict: attention is tp2013 end-to-end; MoE is tp102 end-to-end.** Every MoE consumer
bakes in tp102 — `routed_experts_nki`'s slab AP, the shared expert's `mlp_tkg` weight view,
`sigma_gate`'s `reshape((H0,H1))`.

**Resolution — change the MoE weight ACCESS PATTERNS, not the weights.** The HBM tensors are shared with
the XLA prefill path and cannot be permuted; a decode-only replica is unaffordable (384 MiB/rank/layer).
But tp102 lives only in how the kernels *read* HBM.

| Consumer | Change | Cost |
|---|---|---|
| `router_topk` | pass `x_sb_layout=X_SB_LAYOUT_TP2013` | **free** — it has a layout knob |
| routed gate/up slab (`routed_experts_nki.py:65-74`) | split the 2-way DMA on `s`, not on `h1`-halves | **DMA-neutral** (below) |
| routed down (`routed_experts_nki.py:137`) | stationary `[[H,I0],[H1,H0]], offset=h1` → `[[H,I0],[H2,H0]], offset=s*1024+h2` | **zero** — SBUF→PE load |
| `sigma_gate` (`moe_layer.py:64`) | `reshape((H0,H1))` → tp2013 view | trivial |
| shared expert | no tp2013 path — see below | the one real item |
| matmul loops | **none** | same `j`↔`h` map both sides |

**The routed gate/up AP is DMA-neutral — the load-bearing check.** Current `[[4096,128],[256,8],[1,256]]`
has dim1 step (256) == dim2 extent (256), so dims 1–2 **coalesce**: the true contiguous run is **4 KB**,
not the 512 B its docstring claims. The tp2013 form `[[2048,128],[262144,2],[256,8],[1,256]]` also
coalesces to **4 KB** with the **same descriptor count** (128 × 2 × 4 KB = 1 MB/expert). Two numbers:

| | partition step | 2nd DMA offset |
|---|---|---|
| current (tp102, split on h1) | 4096 | 2048 |
| tp2013 (split on s) | 2048 | 262144 |

**The 129→71 µs DMA fix is preserved.** Cost accepted: tp2013 forecloses an *unsplit* 8 KB-run gate/up
load — but the 2-way split was chosen for pipelining, so that was already given up.

**Shared expert — write it raw-NKI.** `mlp_tkg_gate_up_projection.py:180` hardcodes
`weight.reshape_dim(dim=0, shape=(H0, H1_shard))` with **no layout parameter**, so a core cannot address
a full 16-wide tp2013 tile. We already bypass `mlp_tkg`'s public wrapper (it's broken for our SBUF +
NO_NORM case — `shared_expert.py:12-16`), it's one SwiGLU at `[H,128]/[H,128]/[128,H]`, and the machinery
exists in `routed_experts_nki`. Raw-NKI beats patching the vendored view.

**Traps**
- `rmsnorm_tkg`'s H-layout keys off **`lnc`**, not `do_shard`/`num_shards` (`rmsnorm_tkg.py:218`). Under
  LNC=2 it emits tp2013 *unless* `single_core_forced=True` — the only reason MoE is tp102 today.
  Flipping that flag off is the entire MoE-side activation change.
- **In-place norm can corrupt the residual.** `transformer_gpt_oss.py:319-323` copies the residual to
  scratch before attention because the fused RMSNorm writes back in place. Both our DeltaNet and GQA fuse
  their input norm through `qkv_tkg`. **Verify on our SBUF residual before trusting a 40-layer accumulation.**
- `output_projection_tkg(TRANSPOSE_OUT=True)` nests **h-outer / token-inner**, the reverse of rmsnorm/qkv.
  `_sb2sb_all_reduce_gather` is the adapter that flips it back — don't hand-roll it.
- `combined`'s foreign-token columns are **garbage-but-populated**, not stale — `gated_sum` writes all T
  columns from uninitialized `routed_local`/`shared_local`. Never read them.
- tp2013 is LNC-coupled (degenerates to tp102 at LNC=1). Derive it from `n_prgs`, never hardcode.

### W2b · MoE keeps token-sharding — gather on T, not H
`moe_h_shard_mode` (`routed_experts.py:52-66`) token-shards at T=2: core 0 owns token 0, core 1 owns
token 1, each computing the **full H**. **Keep it.** Its output is a per-rank partial exactly like
attention's; it just needs its LNC gather on the **token** axis. Core *c* all-reduces its own token tile
`[128,16]` across the 4 TP ranks, then one `sendrecv` swaps tokens. The AR operand is the same size
either way, so the collective costs identically.

Do **not** switch MoE to shard-on-H. The byte math is exactly break-even:

| mode | expert loads/core | bytes/load | per core |
|---|---|---|---|
| shard-on-T | 1 token × K=8 = 8 | 1.5 MiB (full expert) | **12 MiB** |
| shard-on-H | 2 tokens × K=8 = 16 | 0.75 MiB (half expert) | **12 MiB** |

Shard-on-H buys **zero** bandwidth at T=2 — only the output layout, which W2 gets for free anyway.
Against that it costs 32 cross-core `sendrecv`s/layer, and `routed_experts_nki.py` has **no shard mode at
all** (no `shard_id`, no `H1_offset`, full-slab DMA, zero collectives) — porting it means re-deriving the
fused `[H0,H1,2I]` slab split through a prefetch ring not structured for it. A good way to give back the
129→71 µs win.

**Landmine → assert:** at T=1/cores=2, `single_core_forced=False` ⇒ the norm emits tp2013 while
`routed_experts_nki`, the shared-expert weight view, and `sigma_gate` all read tp102. **Silent wrong
answer.** Doesn't affect the verify trunk (always T=2), but it must fail loudly.

### W3 · Per-layer name prefixing (prerequisite — but 4 files, not a tree)
Tracing a subkernel N× collides on op names and will not compile.

**The mechanism already exists upstream**: `BufferManager.set_name_prefix()/get_name_prefix()`
(`nkilib/core/utils/allocator.py:546-550`), applied automatically inside `alloc()`/`alloc_stack()`/
`alloc_heap()`. `transformer_tkg.py:221` already de-collides 40 layers with `f"L{layer_idx}_attn_"`.
`rmsnorm_tkg`, `qkv_tkg`, `output_projection_tkg`, and `attention_tkg` all accept a caller-supplied `sbm`,
so they de-collide **for free** if we thread one manager through.

**We do not need nki-moe's 19-file wholesale vendor** — it vendored `moe_tkg` + `attention_block_tkg`,
and we use neither (our routed experts are raw NKI; our attention is already forked).

Vendor set — 4 files:

| File | Why | Severity |
|---|---|---|
| `router_topk` | **No `sbm` parameter at all** (`router_topk.py:57-77`). Hardcoded `name='router_w_sb'` (`:340`), `"ones_mask_sb"` (`:360`), `f"IOTA_offset_{offset}"` (`:996`). Nothing external can reach them. | **hard blocker** |
| `rmsnorm_tkg` | Takes `sbm`, but leaks `name="tin_raw"`/`"tin_perm"` (`:111,116`) — and that's on the **SBUF-input** path, exactly the one we use. | blocker |
| `qkv_tkg` | Takes `sbm`, but two `shared_hbm` tensors use bare literals `"fused_hidden_shared_hbm"`/`"fused_hidden_hbm"` (`:712-719`). | blocker (verify we hit that path) |
| `gqa/vendored/attention_tkg.py` | **Already ours.** ~15 unprefixed DMA names remain (`:1736,1770,3252,3292,3327,3404,3767,3770,3775,3784,3804`). Finish the rewrite here — and land W1b's `v_in_sb` in the same pass. | blocker |

Check `output_projection_tkg` for stray literals; add only if found. The rewrite is mechanical:
`name="x"` → `name=f"{sbm.get_name_prefix()}x"`, hoisting the prefix out of hot loops.

Our own composables hardcode prefixes (`"gqa_qkv_"`, `"gqa_gate_"` at `fused_layer.py:204-206`;
`"shared_mlp_"` at `shared_expert.py:185`). Replace with a threaded `name_prefix`/`sbm` argument.

### W4 · In-kernel TP all-reduce, and removing the framework's
- `_sb2sb_all_reduce_gather` (`nkilib/experimental/transformer/transformer_tkg.py:62-88`) exists and is a
  **drop-in for the attention path**. But it **hardcodes an H-gather** (dest offset
  `prg_id * BxS * H1_shard`, `:70`) and **LNC=2** (`other_lnc = 1 - prg_id`, `:74`).
  **MoE needs a token-gather twin — that's a new function, not a parameter.**
- **`sendrecv` and `core_barrier` are NOT in `nki.collectives`** — they are ISA ops
  (`nki/isa/lnc.py:82` and `:18`). `nki.collectives` exposes only `all_reduce`, `all_gather`,
  `reduce_scatter`, `all_to_all`, `collective_permute`, `rank_id`, `ReplicaGroup`.
- Collective constraints (`nki/collectives/ops.py`): SBUF operands must be **exactly 2-D** (`:59`) with a
  **contiguous free dim** (`:39-46`), no PSUM (`:53`), no HBM/SBUF mixing (`:54`), one tensor per SBUF call
  (`:66`), `op ∈ {add, minimum, maximum}` (`:102`).
- **Delete the three framework all-reduces or you double-reduce**:
  `reduce_from_tensor_model_parallel_region` at `:729` (DeltaNet), `:2174` (GQA), `:2306` (MoE).
  Reuse the `_defer_moe_allreduce` identity-patch idiom (`:2186`).

### W5 · `exec()`-generated flat signature
NKI's frontend classifies tuple/list top-level args as scalars → no HBM binding → aliasing breaks. Every
weight and cache must be its own positional arg, codegen'd via `exec` + `linecache` (working precedent:
`~/nki-moe/megakernels/qwen3_moe/transformer_qwen3_moe_speculative.py:602-668` — populate
`linecache.cache` *before* `compile` so the frontend can introspect the generated source). Stacked-by-layer
tensors are also out — the outer-dim stride trips `NCC_IBIR158/243` OOB.

Arg count: ~68 at N=4, **~675 at N=40** (vs 532 for the 48-layer Qwen3 kernel — same order).

### W6 · Model integration
Layer-0 dispatch, layers 1..N-1 pass through, forwarding the kernel's returned KV/state handles so NxDI
aliases the mutated HBM buffers rather than the pre-kernel ones (precedent:
`~/nki-moe/megakernels/qwen3_moe/qwen_with_megakernel.py:411-486` — layer 0 stashes `_tkg_kv_out` on the
parent; layers 1+ return `(hidden, (K_out[i], V_out[i]), ...)`). Parent access via `weakref`.
`_commit_deltanet` and the aliased-state region (`:4490`) stay untouched.

### W7 · Hoisted per-round invariants
`cos`/`sin` (`mrope_emb`) and the uint8 mask (`_build_kernel_mask`, `:2112`) are rebuilt *inside* each GQA
layer's forward. Hoist both out of the loop and index by layer type (precedent:
`~/nki-moe/megakernels/gpt_oss/transformer_gpt_oss.py:290-313`).

⚠ Speculative-KV ordering trap: the in-place scatter runs *after* attention reads the cache, so slots
`[base_pos, base_pos+T)` hold stale data when attention runs. The mask must key off `base_pos`, and the
active tokens' K must come from the SBUF `k_active` region, not the cache. Silent if you get it wrong.

### W8 · Compile scale
Expect `--internal-max-instruction-limit=30000000`,
`--internal-backend-options=--enable-verifier=false` (NCC-6661: in-place KV scatter trips the verifier),
`-O1` while iterating (`-O3` was 20–60 min for the 48-layer homogeneous kernel; ours is heterogeneous with
DeltaNet recurrence + a 16-iteration expert loop per layer). `--enable-ccop-compute-overlap` +
`--cc-pipeline-tiling-factor=4` to overlap layer i+1's attention AR with layer i's MoE.

---

## 4. SBUF budget

trn3 = 32 MiB/core, **~30.0 MiB usable** (`nki/language/tile_size.py`). SBUF is private per LNC core.

| Item (worst-case layer: DeltaNet + MoE) | per rank | per core |
|---|---|---|
| Attention weights (DeltaNet, H-sharded) | 16.1 MiB | 8.05 MiB |
| Routed experts (16 gathered) | 24 MiB | 12 MiB |
| Shared expert | 1.5 MiB | 1.5 MiB |
| Router + sigma (replicated) | 1.0 MiB | 1.0 MiB |
| **One layer resident** | 42.6 MiB | **≈ 22.6 MiB (75% of usable)** |

**Two layers = ~45 MiB/core → does not fit. There is no cross-layer weight ring.** (`nki-moe` v1 built
one; v2 — the live kernel — dropped it.) Consequences:

- Weights stream per layer. Only **intra-layer** overlap is available: prefetch expert *e+1*'s 1.5 MiB
  while the PE array chews expert *e*. `routed_experts_nki.py` already has this ring. (`nki-moe` had to
  *disable* its equivalent — SWDGE-on-up contended with the 48L/LNC=2 scheduler. Re-measure on our shapes.)
- **Do not hold the 30 layers' candidate states in SBUF** — 7.5–15 MiB/core, collides with the weights.
  Write to HBM per layer, as today.
- Residual (8 KB), in_proj output (12 KB), router logits (2 KB) are noise.

---

## 5. Traffic

Per rank, per verify pass, bf16:

| Stream | MB |
|---|---|
| 30 × DeltaNet (16.09 MiB each) | 506 |
| 10 × GQA (14.01 MiB each) | 147 |
| 40 × MoE routed, 16 experts × 1.5 MiB | 1,007 |
| 40 × MoE shared + router + sigma | 105 |
| **Verify trunk total** | **≈ 1.76 GB** |

Two corrections to `megakernel_scoping.html:509-513`: it omits the per-layer shared expert + router
(~105 MB), and the "union ≤14 experts" saving **does not materialize** in either shard mode — the expert
loop is `(token, k)`-nested with no dedup, so an expert picked by both tokens is loaded twice. Deduping
would need an SBUF broadcast between cores; a later optimization.

The win is not FLOPs — it is killing 120 kernel launches → 1, 80 XLA all-reduces → in-kernel CC ops that
overlap with compute, and ~120 HBM round-trips of the hidden state. Note `nki-moe` measured AR cost
*rising* in-kernel (13.45 → 27.77 µs/layer) and still netted 1.76× because sync-engine (−34 µs) and DMA
(−25 µs) fell further.

---

## 6. Phasing

- **P0 · Unblock.** W3 (vendor 4 files + prefix; `router_topk` first — it is the only hard blocker),
  W1 + W1b (SBUF seams; `v_in_sb` patch kills the GQA round-trip), W2 (tp2013 access patterns + raw-NKI
  shared expert). Gate: every per-block test still green, **plus a new test that traces one composable
  twice in one kernel and compiles** — the cheapest possible proof the prefix scheme works.
- **P1 · One group.** N=4 `[Δ,Δ,Δ,GQA]` with SBUF residual + in-kernel AR (W4, W5). Gate: bit-comparable
  to the torch stack for those 4 layers. Grouping is free — there is no cross-layer ring to lose — so this
  de-risks compile scale at zero perf cost.
- **P2 · Full trunk.** N=4 → 40; layer-0 dispatch + pass-throughs (W6); hoist invariants (W7); strip the
  framework ARs. Gate: token-identical E2E vs the current round.
- **P3 · Optimize.** CC overlap flags, expert prefetch ring re-measure, mask/RoPE hoisting.

---

## 7. Risks

| Risk | Sev | Note |
|---|---|---|
| `router_topk` has no prefix hook | **H** | The one hard blocker. Vendor it first; everything else waits. |
| Fused input norm corrupts the SBUF residual | M | W2 trap. Precedent hit it (`transformer_gpt_oss.py:319-323`). Test at N=2 before N=40. |
| MoE token-gather AR is a new function, not a config | M | W4. `_sb2sb_all_reduce_gather` is H-gather + LNC2 hardcoded. |
| Shared expert has no tp2013 path | M | W2. Write it raw-NKI. |
| Compile time / instruction limit at N=40 | M | P1 at N=4 tells us early. |
| DCE eats the in-place KV scatter / state writes | M | Return every mutated handle. |
| Numerics drift across 40 fused layers | M | Per-block tests + a whole-trunk test vs the torch layer stack. |
