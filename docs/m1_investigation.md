# [M1] Investigation — CTE HLO-explosion fix

**Date**: 2026-04-21 (resume session)
**Scope**: READ-ONLY investigation per user instruction. Decision tree only. No code written, no path chosen.

---

## Problem recap

`qwen_complete.py` currently sets `BlockwiseMatmulConfig(block_size=8192)`. For CTE (ctx=10-402, top_k=8):

- `total_tokens × top_k < block_size` (e.g. 402·8 = 3216 < 8192) → `forward_all_experts` at `expert_mlps_v2.py:1494`.
- `forward_all_experts` does a **Python for-loop** `for e in range(num_experts)` (lines 383-389), unrolling into **128 scaled-add sub-ops per layer × 48 layers = 6144 sub-ops**.
- The resulting HLO exceeds protobuf's 2 GB `SerializeToString` limit at `torch_neuronx/xla_impl/trace.py:179`.

Splitting [M1] into:
- **[M1a]** HLO size fix — critical path; must succeed for any e2e measurement.
- **[M1b]** Drop redundant bf16 CTE MoE weights — ~30 GB DRAM save; handled by swap today; nice-to-have.

---

## New discoveries this investigation

### D1 — Which kernel does `forward_blockwise` actually call?

`blockwise.py:1006-1015`, inside `BlockwiseMatmulNKIFunc.forward`:

```python
if self.training and use_shard_hidden:
    output, ... = _call_training_shard_hidden_kernel(args)
elif self.training:
    output, ... = _call_training_kernel(args)
elif use_shard_on_intermediate_dynamic_while:     # inference path #1
    output = _call_shard_on_intermediate_kernel(args)
elif use_shard_on_block_dynamic_while:            # inference path #2
    output = _call_bwmm_shard_on_block_kernel(args)
else:                                             # inference fallback (CURRENT qwen_complete PATH)
    output, ... = _call_shard_hidden_kernel(args) # raises NotImplementedError
```

`_call_shard_hidden_kernel` at `blockwise.py:267-269` is the default fallback and *always* raises. It's only reachable when:
- CTE inference mode, AND
- neither `use_shard_on_intermediate_dynamic_while` nor `use_shard_on_block_dynamic_while` is True in the `BlockwiseMatmulConfig`.

That is the configuration `qwen_complete.py` uses today — but the dispatcher never reaches this branch anyway because **`forward_blockwise` itself is not selected** when `block_size=8192` (the condition `total_tokens × top_k < block_size` short-circuits to `forward_all_experts`).

### D2 — The beta2 nkilib kernels ARE installed

Probe results (`import_nki_beta2` on this SDK):

| NKI beta2 kernel | module_name | Import status |
|---|---|---|
| `bwmm_shard_on_block` | `moe.moe_cte.bwmm_shard_on_block` | **OK** |
| `bwmm_shard_on_block_mx` | `moe.moe_cte.bwmm_shard_on_block_mx` | **OK** |
| `blockwise_mm_baseline_shard_intermediate` | `moe.moe_cte.bwmm_shard_on_I` | **OK** |
| `blockwise_mm_baseline_shard_intermediate_hybrid` | `moe.moe_cte.bwmm_shard_on_I` | **OK** |
| `moe_cte` | `moe.moe_cte.moe_cte` | **OK** |

All inference-path kernels are available. The only missing kernels are the **training** ones (`blockwise_mm_baseline_shard_hidden`, `blockwise_mm_bwd*`) — and those are warnings at import time that don't block inference.

The "missing private kernel" narrative from Phase 0 was misleading. The actual issue wasn't a missing kernel — it was **a config choice that routed CTE away from the installed kernels** into the expert-looping `forward_all_experts` path.

### D3 — `qwen_fused_transformer.py` already proves the config-only fix

`qwen_fused_transformer.py:78-86` uses:
```python
BlockwiseMatmulConfig.from_kwargs(
    block_size=256,
    logical_nc_config=2,
    normalize_top_k_affinities=True,
    use_shard_on_block_dynamic_while=True,
    block_sharding_strategy="PING_PONG",
)
```

With `block_size=256` and `use_shard_on_block_dynamic_while=True`:
- `total_tokens × top_k >= block_size` (e.g. 402·8 = 3216 > 256) → `forward_blockwise`
- `can_use_blockwise_matmul_nki(..., use_shard_on_block_dynamic_while=True)` → True (beta2 `bwmm_shard_on_block` is installed)
- dispatch lands at `_call_bwmm_shard_on_block_kernel` (`blockwise.py:1013`, uses `_bwmm_shard_on_block_nki_call`)
- ONE NKI kernel call per layer, NOT a Python 128-expert loop

Whether `qwen_fused_transformer` CTE has been compiled end-to-end on this SDK is unknown to this session; its docstring only comments on the TKG path being "for compilation testing only" (the v2_jit residual-add is commented out). The CTE config is an existing, code-visible proof of the pattern.

---

## Decision tree — 5 unblock paths

### (a) Change `BlockwiseMatmulConfig.block_size` only (no flag change)

**Mechanism**: Lower block_size so the dispatcher picks `forward_blockwise` instead of `forward_all_experts`. Leave `use_shard_on_block_dynamic_while=False` (default).

**Does it fix [M1a]?** **NO.** `forward_blockwise` with all flags off lands at `_call_shard_hidden_kernel` (the `else` branch at line 1015), which `raise NotImplementedError`. The model will compile the HLO but fail at runtime. Even earlier: `can_use_blockwise_matmul_nki` returns False when both `use_shard_on_intermediate` and `use_shard_on_block_dynamic_while` are False, because `glu_supported/bias_supported/clamp_supported` are all gated on one of those flags (`blockwise.py:758`). Without them, `can_use_blockwise_matmul_nki` returns False → dispatcher falls through to `elif self.training: forward_all_experts` or else `torch_blockwise_matmul_inference`. The torch fallback is slow and may still materialize a Python loop over tokens.

**Does it fix [M1b]?** No. Block_size doesn't touch weight layout.

**Preserves v15c?** Yes — CTE-side only. TKG path (which uses v15c) is untouched.

**Hours**: 15 min to test, but likely hits the `_call_shard_hidden_kernel` NotImplementedError or the `can_use_blockwise_matmul_nki → False` torch-fallback path. **Not viable alone.**

**New blockers**: would need to set a `blockwise_nki_autograd_cls` override in config, or combine with flags (which makes it path (b)).

---

### (b) Route CTE through forward_blockwise + flags (config-only change)

Three sub-variants based on which flag is set. All land on `forward_blockwise` → `BlockwiseMatmulNKIFunc.forward` → an installed nkilib kernel.

#### (b1) `use_shard_on_block_dynamic_while=True` + `block_size=256`

**Mechanism**: `BlockwiseMatmulConfig.from_kwargs(block_size=256, use_shard_on_block_dynamic_while=True, block_sharding_strategy="PING_PONG", ...)`. Dispatcher hits `_call_bwmm_shard_on_block_kernel` (line 1013) → `bwmm_shard_on_block` nkilib kernel (installed, verified via probe D2). `qwen_fused_transformer.py` already uses this config.

**Does it fix [M1a]?** **YES.** One kernel call per layer instead of a 128-expert Python loop. HLO drops from ~6144 expert sub-graphs/forward to ~48 kernel calls/forward. Typical reduction: **>100× fewer HLO ops for MoE**, comfortably under the 2 GB protobuf limit.

**Does it fix [M1b]?** **NO.** `_call_bwmm_shard_on_block_kernel` still consumes `mlp_op.gate_up_proj.weight` and `mlp_op.down_proj.weight` in bf16 (the stock NxDI weights). The int8 TKG buffers we already register stay as-is for TKG. Total DRAM usage unchanged — still ~2× MoE weight storage from Risk #3.

**Preserves v15c correctness?** **YES.** v15c is a TKG-only kernel. CTE config change doesn't touch TKG dispatch. TKG path in qwen_complete.py continues to use `custom_moe_fused_kernel.qwen3_moe_fused_tkg` (v15c) unchanged.

**Hours**: **2-4 hours**. One config change in qwen_complete.py:105-112 (replace block_size=8192 with the above kwargs). Run `python3 -c "import qwen_complete"` to verify imports still work. Clear caches, run the existing launcher with `--skip-accuracy-check` for one prompt as smoke test (~15 min compile). If smoke passes, 5-prompt bench → STEP 3 + STEP 4.

**New blockers**:
- `bwmm_shard_on_block` kernel may have shape constraints incompatible with Qwen3-30B's config (H=2048, intermediate_tp=48 per TP=4 → I=192/tp=48 if tp_degree=4). `can_use_blockwise_matmul_nki` calls `check_blockwise_mm_kernel_compatibility` which could reject our shapes. If so, fall to (b2) or (b3).
- `block_sharding_strategy="PING_PONG"` vs `"HI_LO"` (default) may have different compatibility; `qwen_fused_transformer.py` uses "PING_PONG", which is a known-working combination with `use_shard_on_block_dynamic_while=True`.
- qwen_fused_transformer.py's CTE compile hasn't been independently verified in this session; only the config pattern is shown.

**Verdict**: **strongest candidate.** Minimal edit, config-only, proven pattern, clear kernel availability, preserves v15c.

#### (b2) `use_shard_on_intermediate_dynamic_while=True` + small block_size

**Mechanism**: dispatcher hits `_call_shard_on_intermediate_kernel` (line 1011) → `blockwise_mm_baseline_shard_intermediate` nkilib kernel (installed).

**Does it fix [M1a]?** Yes — same path logic as (b1), different kernel.

**Does it fix [M1b]?** No — same as (b1).

**Preserves v15c?** Yes — CTE-only.

**Hours**: 2-4 hours; same edit pattern as (b1). Possibly requires different sharding configuration — the intermediate-sharded kernel has different I-dim constraints than block-sharded.

**New blockers**: No public qwen module in this repo uses this variant. Less proven than (b1). May hit a different shape-compat rejection.

**Verdict**: fallback if (b1) fails on shape compatibility.

#### (b3) Custom `blockwise_nki_autograd_cls`

**Mechanism**: Write a subclass of `BlockwiseMatmulNKIFunc` that overrides `forward` to call one of the installed nkilib kernels directly, bypassing the dispatcher's `else: _call_shard_hidden_kernel` fallback. Pass the class via `BlockwiseMatmulConfig(blockwise_nki_autograd_cls=MyClass)`.

**Does it fix [M1a]?** Yes, if the chosen kernel is installed and shape-compatible.

**Does it fix [M1b]?** No (same reason as b1/b2).

**Hours**: **1-2 days** — more invasive; own autograd.Function forward/backward, exact arg packing to match nkilib's kernel signature.

**Verdict**: only useful if both (b1) AND (b2) fail on shape compatibility and we need to hand-pick a kernel call.

---

### (c) Write a custom NKI CTE MoE kernel that bypasses NxDI

**Mechanism**: Replace the stock NxDI MoE `self.mlp` for CTE with a hand-written NKI kernel analogous to v15c but adapted for T>1. Either adapt v15c internally (many T=1 assumptions), or write from scratch using nkilib's moe_block_tkg as a template adjusted for the CTE case.

**Does it fix [M1a]?** Yes — one kernel call per layer, no Python expert loop.

**Does it fix [M1b]?** **YES, potentially.** A custom kernel can consume the int8-packed TKG buffers directly, eliminating the bf16 duplicates. Biggest win on DRAM.

**Preserves v15c?** Yes — v15c stays TKG-only. New kernel is CTE-only.

**Hours**: **3-5 days minimum**. Includes: shape adaptation (T=1 → T=402 max), SBUF budget re-check (larger T = larger intermediate tiles), correctness vs stock NxDI reference, triple-check smoke test, integration into qwen_complete's CTE path.

**New blockers**:
- v15c's int8-reinterpret HWDGE trick relies on T=1 moving tile; for T>1 the layout may need revision.
- CTE expert-distribution handling (multiple tokens hit different top-K experts) — the blockwise sharding logic is what bwmm_shard_on_block already handles; re-implementing in NKI is non-trivial.
- LNC=2 CTE sharding strategy (different from TKG's).

**Verdict**: overkill for Round 3 unless (b1/b2/b3) all fail. Revisit if we want [M1b] addressed standalone.

---

### (d) Compiler flag or HLO-sharding trick to split serialization

**Mechanism**: Make `torch_neuronx/xla_impl/trace.py:179` handle HLOs larger than 2 GB. Either:
- Replace `hlo.SerializeToString()` with chunked serialization + file-backed passing.
- Use protobuf's arena-based serialization (not exposed in the Python API on protobuf 7.34.1 + upb).
- Set a compiler/runtime flag that skips the metaneff serialization and uses a different on-disk format.

**Does it fix [M1a]?** Maybe — addresses the serialization limit without changing model structure.

**Does it fix [M1b]?** No.

**Preserves v15c?** Yes.

**Hours**: **unknown, SDK-level**. Requires patching `torch_neuronx` internals. Even if done, the HLO on disk is still multi-GB — downstream `neuronx-cc` compile-time memory may blow up on the file even if serialization succeeds. Risk that the serialization is only the first wall.

**New blockers**: SDK-owned code; patching it is unsupported. Updates get overwritten on SDK upgrade. Fragile.

**Verdict**: last resort. Flag as out-of-band SDK issue; do NOT attempt as a Phase 3 unblock.

---

### (e) Install the missing `neuronxcc.nki._private.blockwise_mm` module

**Mechanism**: SDK upgrade or targeted install so `neuronxcc.nki._private.blockwise_mm.blockwise_mm_baseline_shard_hidden` is importable.

**What's actually missing**: the `NKIImport(name='blockwise_mm_baseline_shard_hidden', module_name='blockwise_mm')` tried three paths:
- `neuronxcc.nki._pre_prod_kernels.blockwise_mm.blockwise_mm_baseline_shard_hidden` — doesn't have this symbol (only `blockwise_mm_baseline_while_shard_hidden` with `_while_`).
- `neuronxcc.nki._private_kernels.blockwise_mm.blockwise_mm_baseline_shard_hidden` — doesn't have this symbol (same reason — has `_while_` variant).
- `neuronxcc.nki._private.blockwise_mm.blockwise_mm_baseline_shard_hidden` — the `_private` directory exists at `/opt/.../neuronxcc/nki/_private/` but does NOT contain a `blockwise_mm` module.

This is a **name mismatch between NxDI and neuronxcc**: NxDI expects `blockwise_mm_baseline_shard_hidden` (no `_while_`); neuronxcc ships `blockwise_mm_baseline_while_shard_hidden` (with `_while_`). Version drift.

**Does it fix [M1a]?** **NOT NEEDED.** `_call_shard_hidden_kernel` is only reached when `use_shard_on_block_dynamic_while=False` AND `use_shard_on_intermediate_dynamic_while=False`. Path (b) avoids this code path entirely. Fixing the missing symbol only unblocks a code branch we don't need.

**Does it fix [M1b]?** No.

**Hours**: out-of-band SDK/pip upgrade; unknown whether a compatible pinned-version combo exists that matches this NxDI.

**Verdict**: flag as out-of-band. Not on the Round 3 critical path. Phase 0 Block #3's "missing kernel" diagnosis was a red herring — the real issue was never the missing kernel, it was the Phase 0 `block_size=8192` workaround routing CTE away from the installed kernels.

---

## Summary decision table

| Path | Fixes [M1a]? | Fixes [M1b]? | Preserves v15c? | Hours | New blockers | Verdict |
|---|:---:|:---:|:---:|:---:|---|---|
| **(a)** block_size only | No | No | Yes | 0.25 | hits NotImplementedError | **Reject** |
| **(b1)** `use_shard_on_block_dynamic_while=True` + block_size=256 | **YES** | No | Yes | **2-4** | kernel shape compat | **Recommended** |
| (b2) `use_shard_on_intermediate_dynamic_while=True` | Yes | No | Yes | 2-4 | less proven than b1 | Fallback if b1 fails |
| (b3) custom `blockwise_nki_autograd_cls` | Yes | No | Yes | 8-16 | extra code | Fallback if b1/b2 fail |
| (c) custom NKI CTE MoE kernel | Yes | **Yes** | Yes | 24-40 | T>1 SBUF, LNC=2 CTE sharding | Overkill for Round 3 |
| (d) patch `torch_neuronx` trace.py | Maybe | No | Yes | unknown | SDK internals | Out-of-band |
| (e) install missing `_private.blockwise_mm` | n/a | No | Yes | out-of-band | SDK upgrade | Out-of-band red herring |

---

## Phase 0 Block D narrative update

Phase 0 attributed the CTE HLO compile failure to "missing `neuronxcc.nki._private.blockwise_mm` kernel." That's half-true (the symbol IS missing), but it was not the actual cause of the failure we hit in Phase 3 STEP 2. The actual cause:

1. Phase 0 harness fix set `block_size=8192` — motivated by correctly observing that `_call_shard_hidden_kernel` raises NotImplementedError for the default config. The workaround routed CTE away from `forward_blockwise` entirely to avoid that error.
2. But `forward_all_experts` — the chosen fallback — has its OWN Python expert loop that unrolls into 6144 HLO sub-ops, which exceeds protobuf 2 GB.
3. The actually-available beta2 inference kernels (`bwmm_shard_on_block`, `blockwise_mm_baseline_shard_intermediate*`) were never tried because the right config flags (`use_shard_on_block_dynamic_while=True`) weren't set.

**Correction needed in OPTIMIZATION_LOG_TRN3.md**: the Phase 0 Harness Fix #3 should be annotated with a TODO to swap to `(b1)` in Round 3. This is purely a log-clarity fix, not a behavioral change.

---

## Open questions (stop, wait for user pick)

1. Is (b1) the path you want to try first, or do you want (c) for the additional [M1b] win (despite 10× the effort)?
2. If (b1) is picked and hits a kernel-shape-compatibility rejection, should the subagent auto-fall-back to (b2), or stop and report?
3. [M1b] by itself (without the HLO fix) is lower-priority now that swap is on. Do you want it tracked as a separate post-[M1a] Round 3/4 candidate, or dropped?
4. Does the Round 1-2 `qwen_fused_transformer.py` path need to be run end-to-end to confirm (b1) works in practice before we commit to the change in qwen_complete.py? (I'd recommend yes, since we have the evidence it exists but not that it compiles on this SDK.)

**Stopping at writeup per user instruction. No code written, no path chosen.**

---

## UPDATE — (b1) applied, STEP C result (2026-04-21)

STEP A (smoke test on `qwen_fused_transformer`) **PASSED**: all 3 CTE HLO buckets (ctx=128/256/640) + all 3 TKG HLO buckets ([1,1] × 3) generated cleanly. No protobuf error. (b1) config is viable on this SDK.

STEP B (edit `qwen_complete.py:114-134`) **PASSED**: config changed from `block_size=8192` to the (b1) triple (`block_size=256, use_shard_on_block_dynamic_while=True, block_sharding_strategy="PING_PONG"`). Import smoke test `python3 -c "import qwen_complete"` PASS.

STEP C (compile qwen_complete + 1 prompt e2e, cache cleared) — **PARTIAL PASS / NEW BLOCKER**:
- CTE ctx=128, 256, 640 HLOs all compiled cleanly (9.8 / 10.0 / 10.7 s each). **(b1) resolves Block D for CTE.**
- TKG [1,1] HLO serialization: **same `google.protobuf.message.EncodeError: Failed to serialize proto` at `torch_neuronx/xla_impl/trace.py:179`**. Launcher finished at 731.5 s with returncode=1, no Final Score.

### New finding — Block D' (TKG HLO size)

`qwen_complete.py`'s TKG path emits **3 independent NKI custom-call ops per layer** (`v13bc_sbm_tiled` attn + `qwen3_router_topk_plan_a` router + `v15c` MoE) plus surrounding NxDI decoder-layer glue (residuals, 2× RMSNorm, 2× AllReduce). × 48 layers = ~144 NKI custom-call ops + associated bf16 HLO ops.

`qwen_fused_transformer.py`'s TKG path emits **1 fused NKI custom-call per layer** (`transformer_qwen3_moe_tkg_v2_jit`). × 48 layers = 48 ops.

The 3× difference in NKI-op-count is sufficient to push qwen_complete's TKG HLO past the protobuf 2 GB limit, even though qwen_fused_transformer's stays under.

### Revised Round 3 picture

`[M1a]` (CTE HLO) is resolved by (b1) for qwen_complete. `[M1a']` (TKG HLO) is newly surfaced and blocks e2e measurement separately. Options tracked in the Phase 3 STEP C close-out reply — (C1) accept blocker, (C2) swap TKG to fused kernel + fix residual-add, (C3) write megakernel [H], (C4) investigate HLO size per op before deciding.

---

## C4 probe results (2026-04-21)

### STEP 1 — HLO size probe

Monkey-patched `torch_neuronx.xla_impl.trace.hlo_metaneff` to dump ByteSize + per-op breakdown before each `SerializeToString()` call. Ran against `qwen_complete` with (b1) config.

**CTE (ctx=128, 256, 640)**: all ~13 MB HLOs, 32,846 instructions each. **Under 2 GB by 150×.** (b1) completely resolves CTE HLO size. The "Phase 3 Block D CTE issue was from forward_all_experts unrolling 6144 expert ops" narrative is now confirmed retroactively — with (b1) routing CTE to the fused nkilib `bwmm_shard_on_block` kernel instead, only ~1000 `call` ops remain.

**TKG ([1,1])**: `hlo.ByteSize()` **fails with "Failed to serialize proto"** — confirms TKG HLO > 2 GB. Only **8,047 instructions** total (FEWER than CTE) but dominated by **`constant` ops totaling 30.25 GB** (530 constants, avg 57 MB).

Top 10 largest individual TKG ops:

```
constant  xlaconst334   427.82 MB    (IDs: 271, 278, 285, 292, 299, 306, 313, 320, 327, 334)
```

All 10 are same-size `constant` ops with stride-of-7 IDs — suggests **per-layer weight tensors being embedded as HLO constants** rather than passed as `parameter` ops.

CTE's 3,996 constants averaged 448 B each (1.79 MB total); TKG's 530 constants average 57 MB each (30.25 GB total). **Same NxDI trace infrastructure, vastly different constant handling** between CTE and TKG paths. Something specific to the TKG forward in qwen_complete causes weights to be captured as constant literals.

**Earlier hypothesis WRONG**: the "3× NKI-op-count" explanation from my STEP C writeup is not the root cause. TKG has FEWER ops than CTE; the issue is per-op SIZE inflation driven by constant-as-data embedding.

### STEP 2 — Y-residual bug scope

- `qwen_fused_transformer.py`'s docstring references `transformer_qwen_v3_v2.py` — **file does not exist**.
- Actual file `kernels/transformer/transformer_qwen.py:205` has **active `nisa.tensor_tensor(dst=output_sb, data1=residual_moe_sb, data2=moe_gathered_sb, op=nl.add)` — residual-add at Step 9 is NOT commented out**.
- Caveat appears stale. Fix scope: 0-2 hours (verify + update docstring).
- **Important**: `transformer_qwen.py` uses `kernel_v30a_sbuf_io` for MoE and `v13bc_sbm_tiled` for attention — **not v15c**. Swapping qwen_complete's TKG to this fused kernel would **lose the Round 1 v15c −17.2% win** unless we also swap the MoE implementation inside transformer_qwen.py (extra 4-8 hours).

### STEP 3 — Decision recommendation (awaiting user pick)

New root cause identified: **weight-as-constant inlining in qwen_complete TKG** — fits user rubric option (iv) "surprise re-examination" but points to a surgical fix (option v below) rather than C1/C2/C3.

**Ranked options**:
1. **(v.1) Promote TKG-only buffers to `nn.Parameter`** — 2 hours; targets the constant-inlining hypothesis directly; preserves v15c + v13bc.
2. **(v.2) Drop bf16 CTE duplicates + use int8 for both CTE and TKG** — 4 hours; addresses [M1b] as a side effect.
3. **(v.3) Rewire the TKG call site** so weights flow as explicit args, not `self.*` captures — 2-4 hours.
4. **(ii) C2 fused-kernel swap** — fallback if (v) fails. Y-bug stale. **Loses v15c win** unless we also swap inside transformer_qwen.py.
5. **C1 closeout** — only if all above fail.

Recommendation: **(v.1) first**, with a 30-min verification read to confirm the constant-inlining hypothesis on a specific tensor class before editing.

---
