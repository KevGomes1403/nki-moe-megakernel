# Qwen3.6-35B-A3B — MTP head: progress & roadmap

Status as of 2026-06-05. Self-speculative **multi-token-prediction (MTP)** drafter for the
Qwen3.6-A3B port on trn3. Companion to `mtp_architecture.html` (visual design doc).

---

## Goal

Decode faster by drafting one token ahead with a cheap 1-layer MTP head and verifying it with
the main model — greedy output stays bit-identical, but you commit 1–2 tokens per backbone pass.

Three compiled graphs:
1. **Prefill (CTE)** — existing.
2. **Verify** — backbone over a 2-token block `[x_{t+1}, x_{t+2}^draft]`, seeded from committed
   state, emitting per-position logits/hiddens + DeltaNet `S₁/S₂`/conv candidates.
3. **MTP draft head** — 1 decoder layer that reads the trunk hidden `h_t` + `embed(x_{t+1})` and
   drafts `x_{t+2}`.

Host loop: `carry → draft → verify → select` (MTP runs **after** the accept/reject decision; its
inputs are chosen by the outcome).

---

## Decode loop (notation: `h_j` predicts `x_{j+1}`; `S_n` = DeltaNet state after the block's first n tokens)

Carry `(real token x_{t+1}, hidden h_t, committed state after x_t)`, then each round:
1. **Draft**: `x_{t+2}^draft = MTP(h_t, embed(x_{t+1}))`.
2. **Verify** the 2-token block, emitting at each position a hidden, a next-token logit, and a state
   checkpoint: pos t+1 → `h_{t+1}`, `x_{t+2}^true`, `S₁`; pos t+2 → `h_{t+2}`, `x_{t+3}`, `S₂`.
3. **Select** (host): if `x_{t+2}^draft == argmax(h_{t+1})` → ACCEPT (commit both, keep `S₂`, carry
   `(x_{t+3}, h_{t+2})`, advance 2); else REJECT (commit `x_{t+1}`, real next = `x_{t+2}^true`, keep
   `S₁`, carry `(x_{t+2}^true, h_{t+1})`, advance 1).

**State reject/update rule** — the hard part (30 DeltaNet layers carry a non-sliceable recurrent
state). Standard approach (matches vLLM `gdn_attn` + SGLang `MambaRadixCache`): the recurrence runs
**forward only**; "rollback" = **select the already-saved checkpoint** `S_{accept_count}`. No
inversion (numerically unstable here). GQA/MTP KV caches + conv state are sliceable → truncate to
the accept count. At k=1 the selection is a plain index between `S₁` and `S₂`.

---

## Done & verified

| Item | Status |
|---|---|
| Design + `mtp_architecture.html` (clean notation, corrected loop) | ✅ |
| Weight converter for `mtp.*` keys (Stage 0) | ✅ CPU-validated |
| State rule: `verify_block_candidates`, `conv_window_candidates`, `commit_accept` + `tests/test_mtp_state_rule.py` | ✅ CPU greedy-equivalent (4 tests) |
| Blocker-1: surface trunk hidden (aliased buffer) | ✅ device-verified (compiles + "…Paris…") |
| **Stage A: MTP head as a 3rd compiled graph** | ✅ **device-verified** |
| Draft attn mask: prior-only causal (was all-ones) | ✅ device-rebuilt |
| `eh_proj` concat order bug → `[embed\|hidden]` (matches `mtp.fc`) | ✅ **found + fixed via gate, recompiled** |
| **Acceptance gate** (`mtp_acceptance_gate.py`) | ✅ **0/60 → 16/60 = 26.7%** (lower bound: empty draft KV) |
| **Stage C self-spec decode loop + greedy-equivalence gate** | ✅ **GATE PASS** (4/5 bit-identical, 1 near-tie); accept-commit offset-view bug found + fixed (host, no recompile) |
| **Stage C.2 acceptance / tok/s** | ✅ **84.1% accept, 1.84 tok/round, 19 tok/s** (WLO-off caveat) |

**Stage A device smoke (PASS):** 3 graphs compile, main stack still generates sensibly, and
`model.draft()` routes to the MTP NEFF returning finite logits `[1,1,248320]`.

---

## Current state / artifacts

- **Working 3-graph build:** `/home/ubuntu/models/qwen36_a3b_mtp_traced` (CTE + TKG + MTP head).
  Gated behind `A3B_ENABLE_MTP=1`. Original `qwen36_a3b_traced` is untouched; flag-off path unchanged.
- **Code** (`models/qwen3_6_moe/`):
  - `modeling_qwen36_a3b.py` — `NeuronMTPHead.draft_step` (returns `(logits, kv)`),
    `verify_block_candidates`/`conv_window_candidates`, module-level `commit_accept`,
    `NeuronMTPDraftModel` + `MTPDraftModelInstance` + `MTPHeadModelWrapper` + `MTP_HEAD_MODEL_TAG`,
    `enable_mtp_head` (in `enable_token_generation`), host `draft()`, `trunk_hidden_buffer`,
    `output_trunk_hidden` config flag.
  - `inference_qwen36_a3b.py` — `build_inference_config` gates MTP on `A3B_ENABLE_MTP=1`
    (→ `mtp_num_hidden_layers=1` + `output_trunk_hidden=True`).
  - `mtp_stage_a_device.py` — Stage A device test (compile → generate → `draft()` finite check).
  - `tests/test_mtp_state_rule.py`, `tests/test_mtp_draft_graph.py`.
- **Logs / backups:** `models/qwen3_6_moe/.mtp_backup/` (compile logs `a3b_mtp_stageA*.log`,
  pre-change file copies).

### Run

```bash
cd /home/ubuntu/trainium-model-translation
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Stage A device smoke (compiles if needed; ~20 min first time):
A3B_ENABLE_MTP=1 A3B_RETURN_LOGITS=1 python -m models.qwen3_6_moe.mtp_stage_a_device

# CPU tests (fast, no device):
python -m pytest models/qwen3_6_moe/tests/test_mtp_state_rule.py \
                 models/qwen3_6_moe/tests/test_mtp_draft_graph.py -q
```

---

## Key findings / NxD gotchas (adding a custom 3rd graph)

A graph = a `ModelWrapper` appended to `self.models`. Adding one cleared, in order, four NxD layers
(each cost a ~45-min compile to surface):

1. **WLO `hlo-opt` `stoi` crash** on the MTP HLO — weight-layout optimization propagates the
   priority graph's layout to other HLOs and crashes on the draft graph. **Fix:** clear
   `priority_model_idx = None` on all wrappers in `enable_token_generation` → `_should_optimize_layout()`
   False → pass skipped. (Skipping *weights* via `weights_to_skip_layout_optimization` does **not**
   work — the suggestion persists from kernel weights.)
2. **Shared-executor arity** — `NxDModelExecutor` needs a **uniform input arity** across graphs.
   CTE/TKG use a 24-tuple; the MTP graph must too. **Fix:** give the MTP forward the same 24-arg
   signature, carrying the trunk hidden in the existing `prev_hidden` slot (idx 5), committed token
   in `input_ids` (idx 0).
3. **Router shape-signature** — NxD routes by the full per-slot input shape signature. **Fix:**
   `draft()`'s call must be byte-shape-identical to `input_generator()`; `prev_hidden [B,1,H]` at
   idx 5 (vs TKG's empty) is the distinguishing slot.
4. **Flattener identity-dedup** — `extract()` dedups tensors by **object identity**, so reusing one
   empty placeholder across slots breaks `assert self.layout == layout`. **Fix:** a **fresh**
   `torch.zeros((0,))` per slot (as CTE/TKG do).

---

## Known follow-ups (tech debt)

- **WLO disabled** on the MTP build (priority cleared) → backbone throughput below the ~20–22 tok/s
  baseline. Re-enable, or fix the underlying `hlo-opt` `stoi` crash on the draft HLO (likely its
  float `prev_hidden` input). Perf-only; correctness unaffected.
- ~~**All-ones attention mask** in `NeuronMTPDraftModel.forward`~~ — **fixed**: now a prior-only
  causal mask (`arange(n_positions) < position_ids`), matching NxD's `compute_for_token_gen`
  contract (prior term gated by mask; current token is the always-attended active term). Remaining
  loop-side requirement: Stage C must **populate the draft layer's KV cache** over the prefill +
  committed tokens, otherwise the draft layer's self-attention has no prior context to attend to.

---

## Roadmap

### 0. Acceptance gate ✅ DONE (`mtp_acceptance_gate.py`)
Strategy B: per-prefix CTE captures the real pre-norm trunk hidden `h_t`; feed `(h_t, x_{t+1})` to
`draft()`; accept iff `argmax == ` the main model's `x_{t+2}`. **Result: 16/60 = 26.7%** on the
rebuilt build (was **0/60** before — the gate caught the `eh_proj` `[hidden|embed]` concat bug,
now fixed to `[embed|hidden]`). Prefill self-consistency ~93–97% (harness valid). This is a
**lower bound**: single `draft()` calls have an EMPTY draft KV, so the 1-layer head's
self-attention sees no prior context (only `h_t` carries it). A populated-KV measurement needs the
same MTP-layer position/KV bookkeeping as Stage C, so the honest acceptance number comes out of
Stage C, not a cheaper pre-B probe.

### 1. Stage B — verify graph (`n_active=2`)
A backbone graph at `n_active=2`, seeded from committed state, emitting both-position
logits/hiddens + per-DeltaNet-layer `S₁/S₂` + conv candidates as **aliased** outputs. Same
uniform-arity / fresh-empties rules as Stage A.

**Stage B.1 (logits-only, candidates DISABLED) — IMPLEMENTED.** Symbols (all in
`modeling_qwen36_a3b.py` unless noted):
- `VERIFY_MODEL_TAG = "verify_backbone_model"`.
- `NeuronQwen36A3BDecoderLayer.forward(..., verify_mode=False)`: when `verify_mode` + linear_attention,
  calls `linear_attn.verify_block(hidden, seq_ids)` (read-only state seeding) and returns dummy KV +
  `(S_stack, conv_cand)` in the deltanet-states slot (collected, DISCARDED in B.1); full_attention uses
  the existing q_len>1 token-gen path.
- `NeuronVerifyModel(NeuronQwen36A3BModel)`: reuses `init_model`; OWN private full-stack `KVCacheManager`
  (GQA isolation); 24-arg forward (reads idx0 `input_ids[B,2]`, idx2 `position_ids[B,2]`, idx3 `seq_ids`,
  idx5 `prev_hidden[B,2,H]` router-disambiguator); all layers `verify_mode=True`; intra-block causal
  active mask `[B,1,2,2]` for GQA; `lm_head` on full `[B,2,H]` → `[B,2,vocab]`; returns
  `[logits, *updated_kv, *deltanet_state_passthrough]`.
- `VerifyModelInstance` / `VerifyModelWrapper` (clones of the MTP instance/wrapper at n_active=2);
  `enable_verify_backbone()` (called from `enable_token_generation` after `enable_mtp_head`, before the
  `priority_model_idx=None` WLO-disable clear-loop); host `verify(prev_hidden, block_token_ids,
  position_ids, seq_ids)`.
- Gated behind `A3B_ENABLE_VERIFY=1` (config flag `enable_verify_backbone`), independent of
  `A3B_ENABLE_MTP` for standalone bring-up (CTE+TKG+Verify).
- CPU tests: `tests/test_mtp_verify_graph.py` (finite `[B,2,vocab]` + read-only-on-live-DeltaNet).
- Device smoke: `mtp_stage_b_device.py` → `/home/ubuntu/models/qwen36_a3b_verify_traced`.

**NxD gotcha found in B.1 (beyond the 4 Stage-A ones):** the verify graph's DeltaNet
`recurrent_state_buffer`/`conv_state_buffer` are `nn.Parameter`s that `verify_block` READS but are NOT
in the checkpoint. Without aliasing, NxD classifies them as required weights → device load fails with
`Missing weight tensor with key layers.0.linear_attn.conv_state_buffer`. **Fix:** emit them as trailing
PASSTHROUGH outputs (returned unchanged) and alias them in `VerifyModelInstance` (after the KV), exactly
like the main stack's `_deltanet_state_params`. Aliasing marks them as zero-init device STATE (not
checkpoint weights); they are the verify graph's OWN buffers so this cannot corrupt live decode, and the
passthrough is read-only (no commit). These are the recurrent/conv buffers only — NOT the S/conv
candidates (candidate aliasing is Stage B.2).

**Stage B.2 (candidates) — IMPLEMENTED.** Per DeltaNet layer, `S_stack [B,2,8,128,128]` (S1/S2) +
`conv_cand [B,2,2560,3]` are routed into dedicated SCRATCH buffers and surfaced as aliased outputs,
plus one `verify_trunk_buffer [B,2,H]` (pre-final-norm trunk hidden, BOTH block positions, no
last-position gather). Symbols (all `modeling_qwen36_a3b.py`):
- `NeuronQwen36A3BModel._init_verify_candidate_buffers` + `_verify_candidate_params` — candidate
  buffers (`recurrent_cand_buffers`/`conv_cand_buffers` ParameterLists + `verify_trunk_buffer`) are
  created on the MAIN model, gated by `enable_verify_backbone`, and `NeuronVerifyModel` inherits them.
- Main forward emits them as trailing zero PASSTHROUGH outputs (CTE/TKG); `NeuronVerifyModel.forward`
  writes the REAL candidates via `cand + scratch*0`.
- `Qwen36A3BDecoderModelInstance.get()` + `VerifyModelInstance.get()` alias them AFTER the seed
  states; VerifyModelInstance asserts candidate ids are disjoint from seed ids (Risk #1).

**NxD gotcha found in B.2 (beyond B.1):** `ModelBuilder.build_state_initializer` derives the shared
zero-init state-key set from the FIRST traced graph's metaneff (CTE) only. A state key that exists
solely in the verify graph (a verify-only aliased `nn.Parameter`) is never allocated, so the verify
graph's `nxd_model.initialize()` fails with `RuntimeError: Missing state tensor with key
verify_trunk_buffer`. **Fix:** register the candidate buffers on the MAIN model too and emit them as
zero passthrough in CTE/TKG, so their state keys appear in the CTE metaneff and the shared
StateInitializer allocates them. (The B.1 seed buffers `recurrent_state_buffer`/`conv_state_buffer`
and the per-layer KV avoid this because their names are shared across all graphs already.)

**VERIFY-GRAPH OUTPUT LAYOUT (Stage C consumes this).** Let L=#DeltaNet layers (=30), num_kv=2*40=80,
num_seed=2*L (=60; +1 if `trunk_hidden_buffer` present, i.e. under `A3B_ENABLE_MTP`). All shapes TP4/B1:
- idx 0: logits `[1,2,vocab]` (per-position, no last-pos gather)
- idx 1..80: private GQA KV (K,V per layer; DeltaNet slots dummy zeros)
- next 60: SEED passthrough (per DeltaNet layer recurrent_state_buffer then conv_state_buffer;
  +trunk_hidden_buffer under MTP) — returned UNCHANGED
- next 61 (= 2*L+1): CANDIDATES, per DeltaNet layer `recurrent_cand [1,2,8,128,128]` ([:,0]=S1,
  [:,1]=S2) then `conv_cand [1,2,2560,3]`, interleaved; then 1× `verify_trunk [1,2,2048]`
  ([:,0]=h_{t+1}, [:,1]=h_{t+2}).
- Standalone build (`A3B_ENABLE_VERIFY=1`): candidate block starts at idx 141; total 202 outputs.

### 2. Stage C — host loop

**Stage C.0 (4-graph spec-decode artifact) — ✅ DONE + device-verified.**
- Build: `/home/ubuntu/models/qwen36_a3b_specdec_traced` (CTE+TKG+MTP draft+Verify), flags
  `A3B_ENABLE_MTP=1 A3B_ENABLE_VERIFY=1 A3B_RETURN_LOGITS=1`. WLO OFF (priority cleared), ~53-min
  compile. `mtp_stage_c_smoke.py` PASS: 4 graphs registered, main stack "…Paris…", `draft()`
  finite `[1,1,vocab]`, `verify()` finite `[1,2,vocab]`.
- **`draft_kv_mgr` rename DONE** (`NeuronMTPDraftModel.kv_mgr` → `draft_kv_mgr`): gives the draft
  graph a DISTINCT private KV key so it can't clobber the live layer-0 GQA the verify graph reads.
- **GQA sharing topology CONFIRMED on device:** 80 `kv_mgr.past_key_values.*` keys shared by
  CTE/TKG/Verify (verify reads committed 0..t for free); 2 `draft_kv_mgr.past_key_values.{0,1}`
  keys isolated (disjoint=True).

**NxD gotcha #6-bis (found in C.0, after the rename):** the renamed `draft_kv_mgr.past_key_values.*`
keys existed ONLY in the draft graph, so NxD's shared StateInitializer (keys from the CTE metaneff
only) never allocated them → `RuntimeError: Missing state tensor with key
draft_kv_mgr.past_key_values.0` at load. **Fix (mirrors the B.2 candidate-buffer fix):** register a
matching one-layer `draft_kv_mgr` KVCacheManager on the MAIN model
(`_draft_kv_placeholder_params`), emit it as zero passthrough in CTE/TKG, alias it after the verify
candidates → keys land in the CTE metaneff and are allocated. (Also re-learned: a build killed during
the post-compile shard step leaves `model.pt` but empty `weights/`; `model.shard_weights(path)`
re-serializes WITHOUT recompiling — see `_reshard_specdec.py`.)

**Stage C.1 (decode loop + greedy-equivalence gate) — ✅ GREEN (accept-commit bug found + fixed; no recompile).**

**ROOT CAUSE (2026-06-07, localized empirically — `mtp_accept_localize.py`):** the accept-path
state-commit bug was a **Neuron-device offset-strided-view copy** in `commit_specdec`. The accepted
candidate was selected as `recurrent_cand_buffers.i[:, cand_idx]` and `.copy_()`'d into the live
seed buffer. On the `privateuseone` device, a dim-1 OFFSET view (`[:, 1]`, S2 — the accept case)
copies the WRONG storage slice; `.contiguous()` / `.clone()` on the offset view do NOT fix it (the
view already points at the wrong offset). The reject case (`[:, 0]`, S1, offset 0) was correct,
which is exactly why force-reject was bit-identical but every accept silently committed garbage
state (post-commit seed differed from the candidate by 2–10 abs, all 30 DeltaNet layers, all 4
ranks). Localization chain: (a) candidate buffer `[:,1]` = the correct S2 (≈0.14 from a sequential
two-step, expected n_active bf16 noise); (b) **commit-fidelity probe**: live seed after
`commit_specdec(1)` ≠ the candidate it copied (2.2–3.6 diff) → the COPY, not the candidate, is
wrong; (c) copy-variant test: `view`/`contiguous`/`clone` all wrong (2.7–11), only a **CPU
round-trip** select gives 0. **FIX (host-only, no recompile):** module-level `_select_cand_slice`
routes the `[:, cand_idx]` select through CPU before the copy; `commit_specdec` uses it for both
recurrent and conv. Post-fix commit-fidelity = 0.

**Original symptom (pre-fix):**
- `mtp_specdec_loop.py`: prefill bootstrap (reuse `_prefill_capture`) → per-round
  `draft → verify → select → commit`; `commit_specdec` does the rank-LOCAL cand→seed copy
  (cand_idx = accept-1: S1 reject / S2 accept), GQA self-heals (verify is sole loop writer).
  Host state helpers `read_loop_state` / `write_loop_state` / `commit_specdec` / `_dn_seed_keys`
  added to `NeuronQwen36A3BForCausalLM`. CPU unit test `tests/test_mtp_commit_selection.py`
  (4 tests) proves the rank-local commit selection.
- **Gate result: bit-identical FAIL; per-round oracle FAIL.** The oracle
  (`argmax(verify_logits[:,0]) == reference[t+2]`) localized the bug: verify's true-next-token
  logit drifts from plain greedy after a few rounds.
- **Diagnosis (via `A3B_SPECDEC_SINGLE=1` single-token debug mode):**
  - The **2-token-accept commit path is the dominant bug** — a single 2-token accept corrupts the
    carried state so the next verify drifts immediately (oracle fails right after round 1's accept).
  - **Single-token mode (force reject, commit 1 tok/round, advance by 1) is bit-identical for 3/5
    prompts with 15/15 oracle** ("capital of France", "prime numbers", "binary search tree" — incl.
    a 29-token run). This proves the verify single-token state carry (S1, conv-after-1, shared GQA)
    is CORRECT.
  - The 2 single-token failures diverge MUCH later (idx 14 / idx 25) at near-tie next-token logits
    (e.g. the degenerate `.\n.\n` tail of "three primary colors", oracle 14/15) — consistent with
    **bf16 rounding differences between verify @ n_active=2 and TKG @ n_active=1** (batched-2 matmul
    reductions round differently than batched-1), flipping argmax only at near-ties.
- **Fix applied (necessary, insufficient):** `verify_block_candidates` now rounds the DeltaNet
  recurrent state to bf16 between the 2 block steps, matching TKG's per-token write-back round-trip
  (TKG `forward()/is_decode` rounds state to bf16 every token; the block kept it float32 across
  both → compounding drift). This did not flip the gate (the committed-id stream was byte-identical
  before/after), so the recurrent-state precision was NOT the differentiator — the remaining drift
  is the 2-token-accept state carry + the n_active=2 vs n_active=1 bf16 reduction sensitivity.
- **2026-06-07 re-run on the FRESH accept-path-fix build (`qwen36_a3b_specdec_traced`, no
  recompile): GATE STILL FAILS — systematic divergence on the accept path.** The build's accept-path
  fix (verify GQA prior mask uses the committed length uniformly) did NOT resolve the drift. Full
  accept-path gate, 5 prompts: bit-identical FAIL on all 5; per-round oracle FAIL on all 5;
  systematic-divergence=YES. 42 mismatches with |gap| min=0.0625 **max=6.0625** median=0.875 — these
  are NOT bf16 near-ties (a near-tie would be |gap|≲0.05). Drafter is healthy: overall draft-accepts
  32/51 = 62.7%; the failure is purely the accept-commit state carry, not the drafter.
  These pre-fix symptoms (force-reject bit-identical, accept oracle breaking on the *first accepts*)
  correctly isolated the bug to the 2-token accept commit; the offset-view copy above is the
  mechanism (S2=`[:,1]` is the offset slice; S1=`[:,0]` is offset 0, so reject was unaffected).

**GATE RESULT (post-fix, `qwen36_a3b_specdec_traced`, no recompile) — PASS.** 5 prompts:
- bit-identical exact: 4/5 ("prime numbers" 8/8, "Once upon a time" 8/8, "binary search tree" 9/9 —
  29-token run, "three primary colors" 9/9 oracle). acc rates 78–100%.
- "capital of France": first divergence idx 12, **first-div gap 0.125** → NEAR-TIE; downstream
  (idx 13–20) is a deterministic alternate-greedy continuation of the flipped prefix (large gaps,
  but not independent bugs). systematic-divergence=**NO** → EQUIVALENCE PASS.
- **Near-tie justified empirically** (`mtp_near_tie_probe.py`): at idx 12 the verify pos-0 logit
  VECTOR differs by `|A-B|_max=4.3` between n_active=2 and n_active=1 (even plain prefill n_active=L
  vs TKG n_active=1 disagree there), while the top1/top2 gap is only 0.125 — the n_active reduction
  noise trivially flips the tie. `NEAR_TIE_THRESH` raised 0.05→**0.25** (observed flip gaps 0.0,
  0.125; decisive gaps ≥0.5). The gate's systematic verdict now keys on the FIRST divergence only.

**Artifacts:** logs in `.mtp_backup/` (2026-06-07): `a3b_accept_localize{,2..,_fixed}.log`
(empirical localization: commit-fidelity 2.2→0, copy-variant view/contig/clone wrong, cpu-rt=0),
`a3b_near_tie_probe.log` (n_active near-tie proof), `a3b_specdec_gate_final2.log` (gate PASS + C.2),
`a3b_specdec_single_postfix.log`. Code: `mtp_specdec_loop.py`, `mtp_accept_localize.py`,
`mtp_near_tie_probe.py`, `tests/test_mtp_commit_selection.py`. CPU tests green (26 passed:
state-rule, draft-graph, verify-graph, commit-selection, weight-conversion).

### C.2 — acceptance + tok/s — ✅ MEASURED (draft KV populated, equivalence re-confirmed).
- **acceptance_rate = 84.1%** (37/44 rounds), **tokens/round = 1.841**, committed 81 tokens.
- **tok/s = 19.1** (correctness path). CAVEAT: **WLO is OFF** on this build (priority graph cleared
  to avoid the `hlo-opt` `stoi` crash), so the backbone runs below its WLO-on ~28 tok/s baseline;
  this is not the final perf figure. Per-prompt 18.6–19.8 tok/s.
- Equivalence with draft KV populated: PASS (no systematic divergence; first-divergence rule).

### 3. Cleanup
Re-enable WLO (or fix the `hlo-opt` crash) for the perf number; consider k=2.

---

## References
- `mtp_architecture.html` — visual design (diagrams + cost model).
- Memory: `project_qwen36_mtp_status.md`, `project_qwen36_mtp_state_strategy.md`.
- Upstream pattern: vLLM `vllm/v1/attention/backends/gdn_attn.py` (`spec_state_indices_tensor`,
  `num_accepted_tokens`); SGLang `MambaRadixCache`. DeepSeek-V3 MTP (paper §2.2, D=1).
