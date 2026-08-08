# eh_proj — MTP Draft Front End (Spec & Plan)

Status: **PROPOSED — not yet implemented.**
Scope: the `NeuronMTPHead` front end that today runs in PyTorch (`modeling_qwen36_a3b.py:2462-2466`):

```python
combined = torch.cat([embed_norm(next_input_embeds), hidden_norm(prev_hidden)], dim=-1)
h_in     = eh_proj(combined)          # ColumnParallelLinear(2H, H, gather_output=True)
```

`h_in` seeds the draft decoder layer's residual. Goal: a composable `eh_proj_compose` that takes the
embedding (SBUF, from the embed gather) and `prev_hidden` (HBM), and produces the tp2013 residual
tile in SBUF — the front third of the draft megakernel, between `embed` and `gqa_fused_compose`.

Reuse: `qkv_tkg` does the matmul, `gather_embed_rows`/`all_gather_embed_h`/`natural_to_tp2013` do the
data movement. The only new math is a **natural-layout RMSNorm** (~40 lines, the `rms_norm_over_free`
idiom extended from D=256 to H=2048 by chunking the free reduce).

---

## 0. Ground-truth facts verified

- **Concat order is `[embed | hidden]`** (`modeling_qwen36_a3b.py:2458-2465`): weight rows `[0, H)`
  contract the normed embedding, rows `[H, 2H)` the normed trunk hidden. The comment there is
  explicit — the order matches the checkpoint's `mtp.fc` with no column repacking.
- **Two independent RMSNorms, each over its own full H.** `embed_norm` and `hidden_norm`
  (`:2406-2407`) are plain `get_rmsnorm_cls()` modules: replicated (not TP-sharded), full `[1, H]`
  gamma, standard form at runtime (the `(1+w)` conversion happens at checkpoint load — same fact
  chain as `specs/pre_attn_rmsnorm.md` §0). fp32 internal, bf16 out.
- **`eh_proj` is `ColumnParallelLinear(2H, H, gather_output=True)`** (`:2408-2413`): per-rank weight
  `[H/TP, 2H] = [512, 4096]` (nn.Linear `[out, in]`), rank r owns output columns `[r·512, (r+1)·512)`,
  output gathered to full H on every rank. The kernel consumes it **transposed, contraction-first**
  `[2H, H/TP] = [4096, 512]` — the same convention as `dn_proj_w` (`deltanet/components/in_proj.py:9`,
  "qkv_tkg wants contraction H first"); transpose at weight-prep, not in-kernel.
- **The embedding is H-sharded, not vocab-sharded** (`ParallelEmbedding(shard_across_embedding=True)`,
  `modeling_qwen36_a3b.py:4068-4074`): per-rank table `[V, H/TP]`, lookup is a row gather, full H is
  rebuilt by all-gather. Already implemented: `embed/components/embed.py:49,73`.
- **The draft never needs the embedding as a residual tile.** Unlike the verify megakernel (where
  `embed_compose` seeds the trunk residual), the draft residual is `h_in` = eh_proj's *output*. So the
  draft front end uses only `gather_embed_rows` + `all_gather_embed_h` (natural `[T, H]`) and skips
  `embed_compose`'s `natural_to_tp2013` transposes entirely. `return_natural=True`
  (`embed.py:126,141`) anticipated the consumer; the tp2013 half of that call is dead work here.
- **`qkv_tkg` accepts H=4096.** SBUF input `[H0=128, BxS, H1]` derives `H = H0·H1` from the tile
  (`qkv_tkg.py:489-491`); the only constraints are `H % 128 == 0`, weight dim0 `== H`, and
  `H1 % num_shards == 0` (`:533`) — all satisfied at `H1=32`, `num_shards ∈ {1,2}`.
- **`qkv_tkg`'s LNC-2 shard map splits exactly on the two halves.** The weight shard view reshapes
  dim 0 as `(num_shards, H0, H1_shard)` and selects `shard_id` (`qkv_tkg.py:339-343`); at H=4096,
  LNC=2 that is rows `[0, 2048)` on core 0 and `[2048, 4096)` on core 1 — core 0 contracts the embed
  half, core 1 the hidden half. Within a shard the (tile column ↔ weight row) map is
  `h = s·2048 + h0·16 + h2`: each half-block is tp102-ordered, and the whole tile is exactly what
  `natural_to_tp2013(concat, n_prgs)` produces for H=4096. At `num_shards=1` the map degenerates to
  `h = h0·32 + f`, which the same helper also produces at `n_prgs=1` — the assembly is
  layout-correct for both LNC counts **by construction**, with no layout branch in our code.
- **`qkv_tkg` output dtype follows input dtype** (`qkv_tkg.py:1377`) — bf16 in, bf16 out, fp32 PSUM
  accumulate over the full contraction. `fused_add` is an HBM-residual add, unusable for combining
  half-matmuls (`:484,537`).

---

## 1. Contract

### Dims (A3B, TP=4, LNC=2, bs=1)
`H = 2048`, `2H = 4096`, `H0 = 128`, `H1 = 16` (per half), `H_rank = H/TP = 512`. `T = B·S` =
**1 (draft step ①)** / **2 (replay ③)**. Hard caps: `T ≤ 128`, `B = 1`, `n_prgs ∈ {1, 2}`.

| Tensor | Role | Shape | Dtype | Layout / provenance |
|---|---|---|---|---|
| `ids_sb` | token ids | SBUF `[T, 1]` int32 | int32 | from `load_token_ids_to_sbuf` (host ids) or the fused argmax |
| `embed_w` | H-sharded table | HBM `[V=248320, H_rank=512]` | bf16 | `ParallelEmbedding.weight` verbatim |
| `prev_hidden` | trunk hidden h_t (…h_{t+1}) | HBM `[B, T, H]` | bf16 | rolling buffer (①) / rolling ⊕ verify hidden (③) — see §3 |
| `gamma_e` | `embed_norm.weight` | HBM `[1, H]` | bf16 | replicated, standard form |
| `gamma_h` | `hidden_norm.weight` | HBM `[1, H]` | bf16 | replicated, standard form |
| `eh_w` | `eh_proj.weight` transposed | HBM `[2H=4096, H_rank=512]` | bf16 | contraction-first; rows `[0,2048)` = embed half |
| `rg`, `tp_degree` | TP collective config | — | — | `rg=None` ⇒ identity (TP=1), the megakernel convention |
| **`residual`** | **output: decoder residual seed** | SBUF `[H0, T·H1] = [128, T·16]` | bf16 | tp2013(n_prgs) — the `gqa_fused_compose` input layout |

The output tile IS `h_in`: the draft decoder layer's residual base (attn and MoE partials are added
into it), exactly as `load_residual_to_sbuf`'s output is for the verify trunk.

### Dataflow

```
ids_sb [T,1] ── gather_embed_rows ──► emb_local [T,512] ── all_gather_embed_h ──► emb_nat [T,2048]
                                                                                       │
prev_hidden HBM [B,T,2048] ── dma_copy ──► hid_nat [T,2048]                            │
                                                                                       ▼
rms_norm_natural(emb_nat, gamma_e) ──► concat_nat[:, 0:2048]      ┐
rms_norm_natural(hid_nat, gamma_h) ──► concat_nat[:, 2048:4096]   ┘ [T, 4096] bf16 (≤16 KB)
                                                                                       │
natural_to_tp2013(concat_nat, concat_sb [128,T,32], n_prgs)                            ▼
qkv_tkg(hidden=concat_sb, qkv_w=eh_w [4096,512], NO_NORM, BSD, output_in_sbuf=True) ──► [T,512]
                                                                                       │
all_gather_embed_h([T,512], rg, tp) ──► h_in_nat [T,2048]                              ▼
natural_to_tp2013(h_in_nat, residual [128, T·16], n_prgs) ──► residual seed
```

Two collectives total (embed gather-H, output gather-H), both `[T, ·]` 2-D dense SBUF operands —
the `nccl.all_gather(collective_dim=1)` idiom already validated in `embed.py:73-89`.

---

## 2. Design: one wide NO_NORM contraction (recommended)

Concat-then-matmul admits two decompositions:

- **(A) Two fused-norm `qkv_tkg` calls + add.** Split `eh_w` into halves `[2048, 512]`; each half is
  one `qkv_tkg(RMS_NORM)` call (the `in_proj_compose` idiom, `in_proj.py:23`); `tensor_tensor(add)`
  the two `[T, 512]` results.
- **(B) Norm the halves separately, assemble the concatenated normed tile, ONE
  `qkv_tkg(NO_NORM)` over H=4096.** The norm is a new natural-layout helper (§0 shows the assembly
  is layout-exact for the wide call).

**Recommend B.** Concrete reasons:

1. **B matches the XLA baseline's rounding structure bit-for-bit-in-shape.** Baseline: norm in fp32
   → round normed halves to bf16 → cat → one bf16 matmul with fp32 accumulate over 4096. B is
   identical: `rms_norm_natural` rounds to bf16 into `concat_nat`, then one fp32-PSUM contraction.
   A differs twice — no bf16 rounding of the normed values (norm stays fp32 inside `qkv_tkg`), plus
   an extra bf16 rounding at the add of the two partials. Given that draft-token agreement drives
   acceptance rate (bf16-ulp drift is the standing failure mode on this model), rounding-structure
   parity is worth more than reusing the fused-norm path.
2. **B is one `qkv_tkg` call, not two** — fewer instruction issues on a serialization-bound decode
   round, and the weight stays whole (`[4096, 512]` consumed verbatim; A needs it pre-split).
3. **LNC-2 work split falls out for free**: core 0 contracts the embed half, core 1 the hidden half
   (§0, shard map) — A puts both cores through both calls.
4. **B's new math is small and precedented.** `rms_norm_natural([T, H])` = `rms_norm_over_free`
   (`gqa/components/qk_norm.py:42-74`: square → free reduce → `rsqrt(scale=1/H, bias=eps)` →
   `scalar_tensor_tensor`) with the single ≤512-wide reduce replaced by a chunked one (view sq as
   `[T, 4, 512]`, reduce innermost → `[T, 4]`, reduce again → `[T, 1]`). Gamma is partition-broadcast
   to `[T, H]` once (T ≤ 2 rows), same as `gamma_q_sb`/`gamma_k_sb` in `qk_norm_compose`.

Why not `rmsnorm_tkg`: its input/output live partition-major `[128, T, H1]`, but both norm inputs
here are natural `[T, H]` (the all-gather emits token-major; `prev_hidden` loads token-major). Using
it would force natural→partition transposes *before* the norm and buy nothing — the transposes are
needed once either way, and doing them after lets one `natural_to_tp2013` call serve the assembled
concat. (Same reasoning the GQA path uses for qk-norm/RoPE before its single transpose.)

### Rejected

- **(C) Reshard `eh_proj` row-parallel** (contraction-sharded: each rank contracts its local embed
  H-slice pre-gather + a hidden H-slice, `all_reduce(add)` the `[T, H]` partials). Skips the embed
  all-gather, but: needs a checkpoint-side weight reshard, needs cross-rank sum-of-squares exchanges
  for both norms (statistics span full H), and swaps a `[T, 512]`-per-rank gather for a `[T, 2048]`
  all-reduce. Three collectives and a model change to save one tiny gather. No.
- **(D) Leave the front end in XLA** and start the draft megakernel at `h_in`. Costs one HBM
  round-trip of `h_in` + keeps `embed`/norm/proj as XLA launches — the launch floor (~12 µs/NEFF)
  is the whole reason the draft megakernel exists.
- **`embed_compose(return_natural=True)`** as the embed front: computes the tp2013 residual tile the
  draft never uses (16–32 dead `nc_transpose`s). Call `gather_embed_rows` + `all_gather_embed_h`
  directly; no changes to `embed.py` needed.

---

## 3. `prev_hidden`: HBM vs SBUF (decisive)

**While the draft is its own NEFF launch, `prev_hidden` must come from HBM — always, both
invocations.** SBUF does not persist across NEFF launches (each kernel's SBUF is allocated by
graph-coloring for that kernel alone; nothing survives between the HOPs of a traced graph —
already stated in `specs/pre_attn_rmsnorm.md` §4). "The verify graph delivers the hidden in SBUF"
is not a thing that can exist across a launch boundary.

And the HBM delivery **already exists, zero extra work**:

- **Draft ① (T=1):** `h_t` comes from the rolling hidden buffer (`modeling_qwen36_a3b.py:4518`) —
  written by the *previous round's* graph, so HBM by definition.
- **Replay ③ (T=2):** `prev_hidden = cat[orig_hidden, verify_hidden[:, :1]]` (`:4577-4579`).
  `verify_hidden` is the verify megakernel's `output` tensor — the kernel already stores the
  pre-final-norm residual to HBM (`store_residual_to_hbm`, `qwen36_verify_megakernel.py:53`)
  precisely because ① and ③ need it as the draft seed. The host-side concat stays in XLA (trivial).

The SBUF read becomes real only in the endgame single-NEFF round (draft + verify + replay in one
launch). Even then it is **mixed-source**: `h_{t+1}` is the live verify residual tile (SBUF,
tp2013), but `h_t` crossed the round boundary and still arrives via HBM. Design consequence:

- **v1 accepts `prev_hidden` as HBM `[B, T, H]` only.** One `dma_copy` to natural `[T, H]` SBUF.
- The seam is reserved: `rms_norm_natural` takes any natural `[T, H]` SBUF tile, so a future fused
  round passes a caller-assembled tile (copy `h_t` in from HBM next to `h_{t+1}`). The open cost is
  that the verify residual is tp2013 *partition-major*, so producing its natural `[1, H]` row takes
  an inverse transpose (16 `nc_transpose`s) — deferred to the fused-round design (§7.3), not paid now.
- **Verify-trunk prerequisite for the fused round:** once the standalone draft megakernel works, the
  verify megakernel grows a `keep_residual_in_sbuf` flag — skip `store_residual_to_hbm` and hand the
  live residual tile to the in-launch replay so `h_{t+1}` never round-trips HBM. Default-off so the
  standalone verify contract stays byte-identical (the `out_in_sb` convention).

---

## 4. Reuse table

| `file:line` | What | Verdict | Caller adds |
|---|---|---|---|
| `embed/components/embed.py:37` `load_token_ids_to_sbuf` | host ids → `[T,1]` SBUF | drop-in ✓ | first-step adapter only |
| `embed/components/embed.py:49` `gather_embed_rows` | one indirect-DMA row gather | drop-in ✓ | — |
| `embed/components/embed.py:73` `all_gather_embed_h` | `[T, W_rank] → [T, W·tp]` gather on dim 1 | drop-in ✓ ×2 | reused for the eh_proj output gather too (it is width-generic) |
| `embed/components/embed.py:92` `natural_to_tp2013` | `[T, H] → [H0, T·H1]` transposes | drop-in ✓ ×2 | called at H=4096 (concat) and H=2048 (residual); handles any `n_prgs` |
| `gqa/components/qk_norm.py:42` `rms_norm_over_free` | free-axis RMSNorm idiom | **extend** | new `rms_norm_natural`: chunked free reduce (H=2048 > 512), gamma partition-broadcast `[T, H]` |
| `core/qkv/qkv_tkg.py` `qkv_tkg` NO_NORM SBUF path | the 4096-contraction | drop-in ✓ | `hidden=[128,T,32]`, `qkv_w=eh_w`, `NormType.NO_NORM`, `QKVOutputLayout.BSD`, `output_in_sbuf=True`, prefixed `sbm` |
| `core/utils/common_types.py` `NormType`, `QKVOutputLayout` | enums | drop-in ✓ | — |
| `deltanet/components/in_proj.py:23` `in_proj_compose` | fused-norm call idiom | **not used** | Design A only (rejected §2); kept as fallback if the H=4096 call surprises |
| `core/subkernels/rmsnorm_tkg.py` | partition-major norm | **not used** | wrong orientation for natural inputs (§2) |

New code: `nki_kernels/eh_proj/components/eh_proj.py` — `rms_norm_natural`, `eh_proj_compose`, and a
`@nki.jit` `eh_proj_fwd` isolation twin. Directory mirrors `embed/` (sibling composable stage).

---

## 5. Isolation test plan

`eh_proj_fwd(input_ids, embed_w, prev_hidden, gamma_e, gamma_h, eh_w, eps=1e-6, tp_degree=1)` →
`h_in [B, T, H]` HBM via the tp2013-inverse store (the `embed_fwd` pattern, `embed.py:183-197`:
core 0 stores, `core_barrier`). TP=1 in isolation (`rg=None`, full-width `embed_w [V, 2048]`,
`eh_w [4096, 2048]`) — the collectives are identity there; TP=4 is exercised in the megakernel
integration tests, as with `embed`. Launch `[2]` and `[1]` (both `n_prgs`).

1. **Inputs:** random bf16 `prev_hidden [1, T, 2048]`, real-vocab-range ids `[1, T]`, random bf16
   table rows; `gamma_{e,h} = randn(2048)·0.02 + 1.0`; T ∈ {1, 2}.
2. **CPU oracle (fp32):** `cat([rms(emb)·γ_e, rms(hid)·γ_h], -1) @ eh_w_full.T` with
   `rms(x) = x·rsqrt(mean(x², -1) + eps)` — i.e. `NeuronMTPHead.draft_step`'s first three lines run
   in fp32. Oracle uses the same bf16-rounded gammas and a bf16-rounded normed intermediate for the
   bf16 gate (rounding-structure parity is the point of Design B).
3. **Numeric gates (repo rules — cosine similarity BANNED):**
   - fp32 run: `allclose(kernel_fp32, oracle_fp32, rtol=1e-2, atol=1e-5)`.
   - bf16 run: `max_abs(kernel − oracle_fp32) ≤` measured bf16 floor
     (`max_abs(oracle_fp32.bf16() − oracle_fp32)` + headroom).
4. **Layout round-trip:** compare against the oracle at both `n_prgs` values — this is the test that
   catches a tp2013 assembly mistake (§0's shard-map claim is load-bearing; verify it on device).
5. **Torch-path cross-check:** run `Qwen36MTPDraft.forward`'s XLA front end on the same inputs and
   diff — the ulp histogram should match the bf16 gate, confirming baseline parity.

---

## 6. Draft-megakernel integration

Position in `qwen36_draft_megakernel` (single decoder layer, launched twice per round graph):

```
ids → eh_proj_compose → residual [128, T·16]
    → gqa_fused_compose(gamma_in=input_layernorm, kv_write_idx=…)  → all_reduce_gather_h  → residual +=
    → moe_layer_compose                                            → all_reduce_gather_tokens → residual +=
    → store hidden (pre-final-norm, HBM)                            # the ①-launch carry / ③ unused
    → [with_lm_head only] lm_head_compose → all_gather_argmax → ids
```

- **Name-prefixing:** the round graph traces this kernel twice (① T=1, ③ T=2); thread a per-launch
  `name_prefix` through every subcall (the `L{i}_` machinery from the verify megakernel). All
  eh_proj intermediates are SBUF-only — no HBM scratch, so the cross-launch scratch-aliasing hazard
  does not apply to this composable.
- **Weight prep:** transpose `eh_proj.weight` to `[4096, 512]` at flatten time (the `dn_proj_w`
  precedent); `gamma_e`/`gamma_h` pass as `[1, 2048]` (standard form — no `+1` in-kernel).
- **T is a launch-shape property**, not a kernel branch: the same compose serves ① and ③; only the
  `with_lm_head` tail differs between the two builds (compile-time key, like `layer_is_gqa`).

---

## 7. Open questions / risks

1. **`qkv_tkg` at H=4096 is assert-clean but unproven on device.** The validation block admits it
   (§0) and the shard math checks out on paper, but every in-repo call site is H=2048. First device
   run of the isolation test settles it; Design A (two H=2048 fused-norm calls + add) is the
   zero-risk fallback if anything in the tiling assumes H ≤ 2048.
2. **`qkv_tkg`'s internal cross-core reduce at `num_shards=2`.** Each core contracts one half, so
   the full `[T, 512]` result requires the kernel's internal LNC combine. This is the same proven
   path every H=2048/LNC=2 call in the repo exercises — inherited, not re-derived — but worth one
   explicit check that the *output* (not just each shard's slice) is valid on both cores, since the
   downstream all-gather runs replicated on both.
3. **Fused-round SBUF seam (future).** When replay merges into the verify NEFF (behind the verify
   trunk's `keep_residual_in_sbuf` flag, §3), `h_{t+1}` should feed `rms_norm_natural` straight from
   the verify residual — which is tp2013 partition-major, not natural. Options when that lands: 16
   inverse `nc_transpose`s for the one row, or a tp2013-input norm variant + a weight-row
   permutation of the hidden half at load. Out of scope for v1 (§3).
4. **`nccl.all_gather` op-name collision across the two launches.** The verify megakernel's
   collectives de-collide via `name_prefix`; confirm `all_gather_embed_h`'s collective op gets the
   prefix treatment too when called four times per round graph (2 launches × 2 gathers).
5. **bf16 gamma rounding** is folded into the bf16-floor gate with same-rounded-gamma oracle
   (§5.2) — listed to mirror `specs/pre_attn_rmsnorm.md` §6.3, not a separate risk.
