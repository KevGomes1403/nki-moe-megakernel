# qwen36_draft_megakernel — MTP Draft Head in One Launch (Spec & Plan)

Status: **PROPOSED — not yet implemented.**
Scope: the whole `Qwen36MTPDraft.forward` decode path (`modeling_qwen36_a3b.py:4087-4180`) as one NKI
launch: token ids + trunk hidden in, draft token ids + carry hidden + mutated KV cache out. The round
graph traces it twice — **draft ① at T=1** (token needed) and **replay ③ at T=2** (only the KV
writes are consumed). Prefill (`is_cte`) stays XLA.

Composition (every stage exists and is device-validated; see §3):

```
residual = eh_proj_compose(ids, embed_w, prev_hidden, γ_e, γ_h, eh_w)          # [128, T*16] seed
attn     = gqa_fused_compose(residual, …, gamma_in=γ_attn,
                             kv_write_idx=pos, out_in_sb=True)                  # in-place KV (design B)
residual += all_reduce_gather_h(attn)
moe      = moe_layer_compose(residual, …, output_in_sbuf=True)
residual += all_reduce_gather_free_block(moe, f_offset, f_len)                  # §2.3
store_residual_to_hbm(hidden, residual)                                         # pre-final-norm carry
if with_lm_head:                                                                # ① build only
    rank_max, rank_idx, _ = lm_head_compose(residual, final_gamma, lm_head_w)
    tokens = all_gather_argmax(rank_max, rank_idx, rg)
return (tokens?, hidden, k_cache, v_cache)
```

---

## 0. Ground-truth facts verified

- **The two launches differ only in T and which outputs are consumed.** ① (T=1): the round uses
  `sampled` and `hidden` (`modeling_qwen36_a3b.py:4543-4551`); its KV write is redundant at k=1
  (③ column 0 rewrites slot t with a bit-identical payload — `orig_hidden` is stashed at `:4533` so
  the recompute is exact; the MTP architecture doc states ③ could run alone at k=1). ③ (T=2): the
  round consumes ONLY `model_output[1:-1]` — the KV (`:4590`); `sampled` and `hidden` of ③ are dead.
- **The MTP cache is a standard one-layer full-attention cache** (`_config_for_mtp_draft`, `:3381`);
  the layer is shape-identical to a trunk GQA layer (per-rank 4Q/1KV, D=256, H=2048, 256-expert MoE).
- **T=1 is validated on device for every composable** (fp32 `allclose(atol=1e-5, rtol=1e-2)`):
  eh_proj (`test_eh_proj_kernel.py`, T∈{1,2} × LNC∈{1,2}), GQA fused layer with in-kernel input norm
  (`test_gqa_fused_layer_t1_kernel.py`), MoE layer including the T=1 **H-shard fallback** at LNC=2
  (`test_moe_layer_t1_kernel.py`), LM head (`specs/lm_head.md`, T∈{1,2} designed, device-validated).
- **The in-place KV scatter (design B) is validated standalone**
  (`test_gqa_fused_layer_inplace_kv.py`): T∈{1,2}, tail and interior `kv_write_idx`, and the
  read-before-write ordering is asserted (attention output matches a golden computed against the
  PRE-scatter cache). The scatter runs after the prior read; active K/V feed attention from SBUF
  (`v_in_sb`); LNC2 splits V on core 0 / K on core 1 (`gqa/decode/fused_layer.py:102`).
- **Mask conventions at T=1** (pinned by the pre-flight): active tokens occupy the LAST T slots of
  the L-length tile and their trailing mask bits must be kept (at T=1 that single bit is
  `mask[L-1]`); the committed/prior split is expressed only through the mask — the kernel reads the
  whole cache tile, and garbage in `[committed_len, L-T)` provably does not leak when masked.
  cos/sin must be indexed by **token position** (t … t+T-1), not slot position — the unit test
  conflates them (cache full), the shell must not.
- **MoE work split branches on T** (`moe/components/routed_experts.py:71,88`): T=2 → token-shard
  (each core owns a token, full H); T=1 → H-shard (each core owns free indices `[s·H2, (s+1)·H2)`,
  i.e. contiguous H-columns `[0,1024)` / `[1024,2048)`). The valid region of `combined` differs
  accordingly — the reduce-gather must follow it (§2.3).
- **LM head + argmax coexistence with the layer composables is proven**: the verify megakernel
  already runs `lm_head_compose` (MANUAL 224 KiB `BufferManager`) after auto-alloc'd layers in one
  launch (`megakernel/qwen36_verify_megakernel.py:215` tail).

---

## 1. Contract

### Dims (A3B, per-rank TP=4, LNC=2, bs=1)
`H = 2048`, `T ∈ {1 (①), 2 (③)}`, `L = n_positions` (`% 128 == 0`; `% 256` if s_prior-sharded),
`V_rank = 62080`. bf16 IO, fp32 internal. `B = 1`, `n_prgs ∈ {1, 2}`, `rg = None` ⇒ TP=1 identity.

### Inputs (flat positional order — single layer, no codegen loop)

| Group | Tensor | Shape | Notes |
|---|---|---|---|
| step | `input_ids` | HBM `[B, T]` int32 | ①: last committed token; ③: candidate pair |
| step | `prev_hidden` | HBM `[B, T, H]` | ①: rolling buffer; ③: `cat[orig, verify_hidden[:, :1]]` (host concat) |
| step | `kv_write_idx` | HBM `[B, 1]` int32 | `position_ids[:, :1]` — scatter base slot |
| step | `cos`, `sin` | HBM `[T, 64]` | at token positions, host-computed (MRoPE) |
| step | `mask` | HBM `[L, 1, 4, T]` uint8 | §2.4 recipe |
| eh_proj | `embed_w` | `[V, H/TP]` | `ParallelEmbedding` verbatim |
| eh_proj | `gamma_e`, `gamma_h` | `[1, H]` | `embed_norm` / `hidden_norm`, standard form |
| eh_proj | `eh_w` | `[2H, H/TP]` | transposed at flatten (dn_proj_w convention) |
| gqa | `gamma_in` | `[1, H]` | `decoder_layer.input_layernorm` |
| gqa | `qkv_w`, `gate_w` | `[2048, 1536]`, `[2048, 1024]` | head-major `[q0..q3|k0|v0]` |
| gqa | `gamma_q`, `gamma_k` | `[256]` | qk-norm |
| gqa | `o_proj_w` | `[1024, 2048]` | |
| gqa | `k_cache`, `v_cache` | `[1, 1, 256, L]` BHDS / `[1, 1, L, 256]` BHSD | **mutated in place** |
| moe | `moe_gamma`, `router_w`, `gate_up_w`, `down_w`, `sigma_gate_w`, `shared_{gate,up,down}_w` | as verify (`MOE_FIELDS`) | E=256, k=8, I=I_s=128 |
| head | `final_gamma` | `[1, H]` | `mtp_head.final_norm` — **with_lm_head build only** |
| head | `lm_head_w` | `[2048, V_rank]` | `mtp_lm_head` verbatim — **with_lm_head build only** |
| cfg | `eps`, `replica_groups` | scalars / tuple | trace-time |

### Outputs

| Build | Returns | Consumed by round |
|---|---|---|
| `with_lm_head=True` (①) | `(tokens [B,T] int32, hidden [B,T,H], k_cache, v_cache)` | tokens → candidate ids; hidden → ③'s `prev_hidden`; caches → aliasing only |
| `with_lm_head=False` (③) | `(hidden [B,T,H], k_cache, v_cache)` | caches → the round's draft KV (`:4590`); hidden dead but returned (DCE guard uniformity) |

Mutated cache handles are ALWAYS returned — an unreturned mutated buffer is dead-stored by NCC
(the verify kernel's 81-tensor lesson).

---

## 2. Design decisions

### 2.1 In-place KV scatter (design B) is the primary path — decisive

Chosen over emit-and-XLA-scatter (design A). Grounds:

- **Validated**: the standalone in-place test covers both T values, both write-position classes, and
  the read-before-write ordering (§0). The megakernel adds composition, not new scatter mechanics.
- **Correct without rollback**: every read is masked by committed length, and a rejected slot is
  rewound-to and overwritten before any read (the round's own self-cleaning invariant). In-place
  adds no new invariant.
- **The cross-launch write-after-write is benign by value at k=1.** ① writes slot t; ③ rewrites
  slot t with a bit-identical payload (§0) and additionally writes t+1, which ① never touches. There
  is no read-after-write between the launches either: ③'s slot-t+1 column attends to slot t's K from
  its own SBUF active tile (intra-block active mask), not from the cache. So ANY scheduler order of
  the two launches' writes yields the same final cache — the compiler/scheduler needs only ordinary
  aliased-buffer dependence tracking, no extra fences. Consequence: ①'s returned cache handles may
  be dropped by the graph (its write is redundant at k=1); **③'s returned handles are the round's
  draft KV**, which is exactly what the model already consumes (`:4590`).
- **Model-side**: the NKI branch of `Qwen36MTPDraft.forward` skips `kv_mgr.update_cache` and returns
  the kernel's mutated handles as `updated_kv` (the aliased state region pattern already used for
  DeltaNet committed states). This is the NxDI-dependency cut, scoped to the draft cache only.

Revisit trigger: at k > 1 the "benign by value" argument must be re-derived per slot (③'s early
columns correct speculative-hidden entries — values then differ from ①'s writes and launch order
matters; the aliased-handle threading discipline returns).

### 2.2 `with_lm_head` is a compile-time build key

`build_draft_megakernel(with_lm_head: bool)` returns a cached jitted wrapper (the
`build_verify_megakernel` pattern, keyed on a bool instead of the layer tuple). The ③ build omits
`final_gamma`/`lm_head_w` from its signature entirely — the LM head streams ~127 MB/core of weight,
pure waste on a launch whose logits are dead. T is shape-derived, not a key.

### 2.3 New helper: `all_reduce_gather_free_block` (generalizes the MoE gather)

The verify megakernel's `all_reduce_gather_tokens` slices the valid token block
`flat[:, T_offset·H1 : (T_offset+T_len)·H1]` before the TP all-reduce + LNC sendrecv. At T=1 the
MoE valid region is the **H-shard** free block `[s·H2, (s+1)·H2)` instead — a contiguous free slice
of a different offset/length. Both are the same operation on a contiguous free block; generalize to
`all_reduce_gather_free_block(tile, f_offset, f_len, rg)` and derive `(f_offset, f_len)` at trace
time from the same shard decision the MoE composable makes (`moe_token_shard` / `moe_h_shard`,
`routed_experts.py:71,88` — T is a trace-time constant, so this is a Python branch, not a kernel
branch). `all_reduce_gather_tokens` becomes the `f_offset = T_offset·H1` special case. The dense
copy-before-collective (nccl needs a packed 2-D operand) carries over unchanged.

Helper relocation: `load/store_residual_to_hbm`, `all_reduce_gather_h`, the generalized free-block
gather, and `all_gather_argmax` move from `qwen36_verify_megakernel.py` to
`megakernel/collectives.py`; the verify file re-imports them (import repoint only, no math change).

### 2.4 Host-side step inputs (XLA, trivial)

- **Mask** `[L, 1, 4, T]` uint8, from `position_ids` (committed length `t = position_ids[0]`):
  `keep[j, i] = (j < t and j < L−T) or (L−T ≤ j ≤ L−T+i)` — committed prior plus causal trailing
  active slots. The last-slot bits are load-bearing (§0). Slots `[t, L−T)` masked off.
- **cos/sin** at token positions `t … t+T−1` (MRoPE host computation, as today).
- **prev_hidden** for ③: `cat[orig_hidden, verify_hidden[:, :1]]` stays a host concat.
- **kv_write_idx** = `position_ids[:, :1].int()`.

### 2.5 Residual & norm placement

Identical to one verify GQA layer: `gqa_fused_compose` norms in-kernel (`gamma_in`), MoE norms
in-kernel (`moe_gamma`), the eh_proj output tile IS the residual base, adds are in-place
`tensor_tensor` on the tp2013 tile. `hidden` is stored **pre-final-norm** (the carry contract);
`lm_head_compose` applies `final_gamma` itself.

### Rejected

- **Design A first, flip B later**: B is already validated at unit level and A would build the
  `update_cache` plumbing only to delete it; the k=1 WAW analysis (§2.1) removes A's remaining
  safety argument.
- **One build with a runtime lm_head skip**: NKI has no runtime branch; two cached builds are the
  established pattern and drop the dead weight args from the ③ signature.
- **In-kernel MRoPE / mask generation**: tiny host tensors, per-launch varying; not worth kernel
  surface. Same call as the verify path.

---

## 3. Reuse table

| Piece | Where | Status | Shell adds |
|---|---|---|---|
| `eh_proj_compose` | `eh_proj/components/eh_proj.py:101` | device-validated | `out_sb`, `name_prefix` seams already present |
| `gqa_fused_compose` + in-place scatter | `gqa/decode/fused_layer.py:153,102` | device-validated (incl. B) | pass `kv_write_idx`, `gamma_in`, `out_in_sb=True`, prefix |
| `moe_layer_compose` | `moe/components/moe_layer.py:174` | device-validated at T∈{1,2} | `output_in_sbuf=True`, prefix |
| `lm_head_compose` + `all_gather_argmax` | `lm_head/components/lm_head.py:287`, megakernel `:147` | device-validated | with_lm_head build only |
| `all_reduce_gather_h` | megakernel `:68` | proven ×40/layer | relocate to `collectives.py` |
| `all_reduce_gather_tokens` | megakernel `:107` | proven | **generalize** → `all_reduce_gather_free_block` (§2.3) |
| `load/store_residual_to_hbm` | megakernel `:38,53` | proven | relocate |
| flat-arg builder | `flatten_megakernel_args` `:444` | pattern | new `flatten_draft_args` — fixed order, no per-layer codegen |
| build cache | `build_verify_megakernel` `:493` | pattern | `build_draft_megakernel(with_lm_head)` — plain `@nki.jit` wrapper is fine (no exec-codegen needed at fixed arity), keep the not-double-jitted rule |

New files: `megakernel/qwen36_draft_megakernel.py`, `megakernel/collectives.py` (relocations),
plus the import repoint inside `qwen36_verify_megakernel.py`.

---

## 4. Isolation test plan

`tests/test_draft_megakernel.py`, oracle = `Qwen36MTPDraft.forward` math run in torch fp32 (embed →
eh_proj front → decoder layer → final norm → lm_head → argmax), TP=1, LNC launch `[2]` and `[1]`.

1. **① case (T=1, with_lm_head)**: random weights, prior cache with `t ≈ 150`, `L = 256`.
   Gates — `hidden`: fp32 `allclose(atol=1e-5, rtol=1e-2)`; `tokens`: **exact** int match vs torch
   argmax (vocab-parallel tie-break must match torch lowest-index — already the lm_head contract);
   caches: written slots `[t, t+1)` match oracle K/V post-norm/RoPE to the same allclose, all other
   slots **bit-identical to their pre-launch contents** (the in-place scatter touched nothing else).
2. **③ case (T=2, no lm_head)**: `prev_hidden = [h_t, h_{t+1}]` pair, candidate ids pair. Gates —
   caches at `[t, t+2)` + untouched-slot check; `hidden` allclose (returned though dead in the round).
3. **Round chaining smoke**: run ① then ③ against the same cache buffers in one process and assert
   the final cache equals the ③-only oracle (empirically confirms the §2.1 benign-WAW claim).
4. bf16 informational run (max_abs vs floor). No cosine similarity anywhere.

---

## 5. Round-graph integration

- `Qwen36MTPDraft.forward` gets the NKI branch for the decode widths (`q == 1` or `q == spec_len`);
  `is_cte` keeps the XLA path unchanged. The branch: flatten args → `build_draft_megakernel(q == 1)`
  → return `[sampled, *kernel_cache_handles, hidden]` (skips `self.kv_mgr.update_cache`; ③'s handles
  flow to `flat_draft_cache` exactly as today's `:4590` expects).
- Two launches per traced round ⇒ per-launch `name_prefix` (e.g. `"d1_"`, `"d2_"`) threaded through
  every composable and collective — including the eh_proj `nccl.all_gather` ops (eh_proj spec §7.4).
- No HBM scratch in the shell (eh_proj is SBUF-only; GQA/MoE stream weights): the only aliased HBM
  the two launches share is the KV cache pair, covered by §2.1.

---

## 6. Open questions / risks

1. **SBUF budget of the ① build**: one GQA+MoE layer (~22.6 MiB/core in the verify worst case is
   DeltaNet+MoE; GQA+MoE is smaller — attention weights 8.05 + experts 12 + shared/router ~2.5 ≈
   22.6 MiB/core too) **plus** lm_head's 224 KiB manual arena plus eh_proj tiles (< 0.5 MiB) plus
   the `[T, V_core]` logits tile (~62 KiB fp32 at T=1). Fits on paper under ~30 MiB usable;
   first compile will tell. Mitigation if tight: lm_head weight streams (already manual-alloc), and
   the expert prefetch ring depth is the tunable.
2. **Auto-alloc + MANUAL `BufferManager` coexistence** is proven in the verify kernel but the draft
   adds eh_proj's auto-alloc sbm in the same launch; expected fine (separate managers, disjoint
   arenas) — verify at first compile.
3. **③'s dead `hidden`/logits at T=2**: the ③ build already drops the lm_head; storing `hidden` is
   kept for signature uniformity and costs one 8 KB DMA. Acceptable.
4. **Interaction of the in-place scatter with the LNC K/V split** (V on core 0, K on core 1) and the
   `prg_id`-gated duplicate-write guard is unit-tested standalone; the composition test (§4.3) is
   the guard against a megakernel-context regression.
5. **k > 1 future**: §2.1's benign-WAW derivation is k=1-specific; the revisit trigger is documented
   there.
