# Integration findings — Phase 3 STEP 1a

**Date**: 2026-04-21
**Context**: wiring v15c MoE + an attention TKG kernel into `qwen_complete.py` for the first real trn3 e2e measurement.

---

## Finding 1 — `v17_fast_exp` produces all-zero output

**Severity**: Critical. Round 1 Plan B [X1] is a **false-positive win**, not a real kernel improvement.

**Evidence**:
- `bench_v17_fast_exp.py` passes correctness at `rtol=1e-2, atol=2e-2` with `max_abs_err=1.65e-02` — exactly the magnitude of the reference attention output (`max |ref| = 0.0165`).
- `probe_v17_passthrough.py`: running v17_fast_exp's sub-function and copying `out_sb` AS-IS to HBM produces `range=[0.000000, 0.000000] absmean=0.0000e+00`.
- `probe_v13bc_passthrough.py` (same test on the v13bc baseline): produces `range=[-0.014, 0.013] absmean=3.0e-03` — the baseline is correct.

**Root cause**: the `nisa.exponential` swap in the two softmax sites (v17_fast_exp.py Pass 2, lines 518 + 539) silently corrupts the sub-function output. The kernel compiles and executes without error, but produces zeros. `nisa.activation(op=nl.exp, data=score_shifted)` and `nisa.exponential(dst=..., src=score_shifted)` are NOT bitwise-equivalent on this SDK when used with `max_value=0.0`.

**Impact on Round 1 log**:
- Plan B's 53.28 μs "win" is the speed of computing and storing nothing.
- The actual attention TKG baseline on trn3 is **v13bc_sbm_tiled at 53.71 μs**. Round 1 did not improve attention.
- `qwen_complete.py` is updated in Phase 3 to import from `v13bc_sbm_tiled`, not `v17_fast_exp`.

**Remediation open item**: debugging `nisa.exponential` vs `nisa.activation(op=nl.exp)` semantic difference on this SDK is a Round 3 candidate (**[X1b]**) if we want to pursue [X1] correctly.

### Postmortem — where exactly v17_fast_exp breaks

The diff `v13bc_sbm_tiled.py` → `v17_fast_exp.py` is **exactly two lines** — Pass 2 softmax exp and the active-position exp:

```diff
- nisa.activation(score2_exp, op=nl.exp, data=score2_shifted)
+ nisa.exponential(dst=score2_exp, src=score2_shifted)

- nisa.activation(score_act_exp, op=nl.exp, data=score_act_shifted)
+ nisa.exponential(dst=score_act_exp, src=score_act_shifted)
```

Both inputs are `(PMAX=128, GQA=8) float32` SBUF tiles. Both outputs are same shape/dtype. Semantically the two calls should compute the same thing (`exp(score_shifted - 0.0) == exp(score_shifted)`).

**Reading `nki/isa/misc.py:37-130`**, `nisa.exponential` is trn3-only and computes `dst[i] = exp(src[i] - max_value)` on VectorE. Critically, the docs say:

> "Even when reduce_cmd is set to idle [the default], the accumulator state may still be modified. Always use reset_reduce after any Vector Engine operation that ran with idle mode to ensure consistent behavior."
>
> "The accumulator registers are shared for other Vector Engine accumulation instructions such as `nki.isa.range_select`, `nki.isa.select_reduce`, and `nki.isa.tensor_scalar_cumulative`."

**Three candidate root causes, in decreasing order of likelihood**:

1. **Shared VectorE accumulator state corruption.** The two `nisa.exponential` calls run with `reduce_cmd=idle` (default). Subsequent VectorE ops in the softmax (`nisa.tensor_tensor`, `nisa.tensor_copy`) may inherit a corrupted accumulator register. If any downstream op implicitly reads the accumulator (e.g., during internal precision-handling paths), the softmax would produce wrong values that compound through the normalization divide and the V·weights matmul. End-state: output zeros or NaN masked to zero by some downstream clamp. The doc's own warning about "always reset_reduce after idle-mode VectorE ops" strongly suggests this hazard is real.

2. **`max_value=0.0` scalar default not handled for fp32 src.** `max_value` docs say it "can be a scalar or vector of shape `(src.shape[0], 1)`, supported dtypes: float32." The scalar-zero default may have an untested code path on this SDK revision. Inputs to the softmax in v17 are fp32; ref is fp32; but the SDK may special-case scalar-zero in a broken way.

3. **NKI keyword-arg regression** — `nisa.exponential(dst=..., src=...)` may not bind correctly if the compiler expects positional args. Unlikely given the signature matches.

**Correct fix to try** (Round 3 candidate [X1b]):
```python
# Option A: force accumulator reset on every exp call
nisa.exponential(dst=score_exp, src=score_shifted,
                 reduce_cmd=nisa.reduce_cmd_enum.reset_reduce)
# If the reset doesn't help, try passing an explicit scalar max_value:
nisa.exponential(dst=score_exp, src=score_shifted, max_value=0.0,
                 reduce_cmd=nisa.reduce_cmd_enum.reset_reduce)
# Option B: fold max subtraction into nisa.exponential's max_value arg
# (requires max_value shape (src.shape[0], 1) — won't fit GQA-dim max)
# Option C: revert to nisa.activation(op=nl.exp) — known-correct baseline
```

**Until a correct pattern is demonstrated**, [X1] is off the table. `v17_fast_exp.py` stays in the repo as a documented false-positive — do NOT import from qwen_complete or any production wiring.

---

## Known NKI landmines

A running list of NKI SDK / NxDI behaviors that silently produce wrong output without compile errors. Every future kernel/wrapper author needs to see this section.

### L1 — `nisa.dma_copy(dst=reshape(...), ...)` does not alias to the declared output

**Observed in**: Phase 3 STEP 1a, `qwen3_attn_tkg_v17_wrapper` in `_qwen_integration.py`.

**Pattern that fails silently**:
```python
output = nl.ndarray((B, 1, H_wo), dtype=nl.bfloat16, buffer=nl.shared_hbm)
output_2d = output.reshape((1, H_wo))
...
nisa.dma_copy(dst=output_2d.reshape((NUM_OUT_COLS, PMAX)), src=transposed_sb, ...)
return output, ...  # returns ZEROS — DMA wrote to a temporary, not to `output`
```

**Safe pattern**:
```python
output_flat = nl.ndarray((NUM_OUT_COLS, PMAX), buffer=nl.shared_hbm)  # target shape
nl.store(output_flat, transposed_sb)                                   # no reshape on dst
return output_flat.reshape((B, 1, H_wo)), ...                          # reshape on RETURN
```

**Rule**: never reshape the `dst` argument of `nisa.dma_copy` or `nl.store`. Allocate HBM in the target shape; reshape after the kernel returns.

### L2 — `nisa.exponential` with `reduce_cmd=idle` (default) corrupts downstream VectorE

**Observed in**: v17_fast_exp (see Finding 1 above).

**Pattern that fails silently**:
```python
nisa.exponential(dst=exp_out, src=scores_shifted)   # reduce_cmd=idle by default
nisa.tensor_tensor(...)                              # any later VectorE op reads corrupt accumulator
```

**Mitigation (unverified — needs [X1b] probe)**:
```python
nisa.exponential(dst=exp_out, src=scores_shifted,
                 reduce_cmd=nisa.reduce_cmd_enum.reset_reduce)
```

Until [X1b] verifies the mitigation, **prefer `nisa.activation(op=nl.exp, data=x)`** which is known-correct.

### L3 — Correctness tolerance `rtol=1e-2, atol=2e-2` is useless when `max(|ref|) < atol`

**Observed in**: `bench_v17_fast_exp.py`, `bench_v13bc_sbm_tiled.py`, `bench_v13bc_tkg_layout.py` (all attention benches).

Attention output values are ~1e-2 magnitude given typical bench weights (`Wq, Wk, Wv, Wo = 0.02 × randn`). `atol=2e-2` ≥ `max(|ref|)`, so zero output vs ref satisfies `atol` unconditionally and the test PASSES even when the kernel is broken.

**Rule**: every wrapper/integration smoke test uses the triple-check pattern:
```python
assert got_absmean > 1e-4                  # zero-check
assert 0.5*ref_max < got_max < 2.0*ref_max # range-check
np.testing.assert_allclose(got, ref, ...)  # element-wise
```

### L4 — Bit-exact `assert_allclose(max_diff=0)` is safe ONLY vs a known-correct reference kernel

**Observed implicitly in**: v14a ↔ v15c ↔ v15e ↔ v15f ↔ v15g bit-exact comparisons.

If both the reference kernel AND the candidate produce the SAME wrong output (e.g., both emit zeros), `max_diff=0` passes. Bit-exact comparison is a PASS only if the reference has been independently validated as non-zero with correct magnitude. Spot-check references with a PyTorch fp32 comparison at least once.

**v14a / v15c verified correct**: `probe_v14a_v15c_zero_check.py` — both produce `absmean ~11.6, range [-49, 55]`. The Round 1 + Round 2 MoE bit-exact comparisons are valid.

### L5 — `register_buffer` tensors get inlined as HLO constants (NxDI trace doesn't iterate `named_buffers()`)

**Observed in**: Phase 3 Round 3 [M1a'] — qwen_complete's TKG HLO exceeded protobuf 2 GB.

**Mechanism**: `neuronx_distributed/trace/model_builder.py:806` iterates `model.named_parameters()` to build the set of trace inputs. **It does NOT iterate `named_buffers()`.** Tensors registered via `tkg.register_buffer("xxx", tensor, persistent=True)` are therefore NOT trace inputs. The forward body accesses `self.xxx` as a Python closure variable → XLA materializes those tensors as **`constant` HLO ops**. For int8/fp32 buffers this also triggers a **4× upcast to fp32** during materialization.

**Proof — HLO probe data**:
```
qwen_complete TKG HLO (8,047 instructions):
  constant  30.25 GB total  (530 ops, avg 57 MB)
  top-10 individual ops: all 427.82 MB (exactly int8 [128,17,128,384] × 4 bytes/fp32)

qwen_complete CTE HLO (32,846 instructions, same model):
  constant  1.79 MB total   (3,996 ops, avg 448 B)
```

CTE has 3,996 constants averaging 448 B each because its weights flow through stock NxDI `self.mlp(...)` — those are `nn.Parameter` entries, captured as parameters by NxDI's trace. TKG hits the register_buffer path and explodes.

**Pattern that fails silently**:
```python
tkg.register_buffer("weight", tensor, persistent=True)
# later in forward():
output = nki_kernel(..., tkg.weight.data, ...)   # HLO gets `constant` op sized 4×tensor
```

**Safe pattern**:
```python
tkg.weight = nn.Parameter(tensor, requires_grad=False)
# same forward — now HLO gets `parameter` op (by reference, not by value)
```

**Rule**: for any tensor that is (a) model-scoped state, (b) accessed inside the `forward()` that will be traced by NxDI, and (c) not subject to gradients, use `nn.Parameter(..., requires_grad=False)` — NOT `register_buffer`. State_dict serialization works identically for both.

**Dtype protection note**: if you have an `_apply` hook protecting buffer dtypes from `model.to(bfloat16)` coercion (as in `_install_nki_router`'s `weight_T` protection), swap `tkg._buffers[name]` → `tkg._parameters[name]` to match the new registration — and wrap in `nn.Parameter(...data.to(dtype), requires_grad=False)` since assigning a plain tensor to `_parameters[...]` is an error.

### L6 — `ModelBuilder.shard_checkpoint` accumulates all TP-rank sharded dicts in RAM before returning

**Observed in**: Phase 3 STEP 2 retry (2026-04-22 04:34 UTC). SIGKILL'd at 127 GB RSS + 32 GB swap saturated on trn3.3xlarge.

**Mechanism**: `neuronx_distributed/trace/model_builder.py:827-858` `shard_checkpoint` builds a Python list holding **all TP-rank** sharded checkpoint dicts before returning. For Qwen3-30B-A3B with TP=4, each sharded dict is ~15 GB bf16; the loop peaks at 4×15 GB = 60 GB **on top of** the original checkpoint still resident in scope (~60 GB). Net peak ≈ 120 GB, which OOMs on a 124 GB instance with 32 GB swap.

NxDI exposes **no config knob** that toggles the accumulation — `save_sharded_checkpoint` and `skip_sharding` only pick WHERE sharding happens (compile vs load), not HOW the loop buffers ranks.

**Fix pattern** (installed in `qwen_complete.py:111-232`, gated by `NKI_STREAMING_SHARD_PATCH=1`): monkey-patch `ModelBuilder.shard_checkpoint` with a streaming version that calls `save_file(sharded, path)` per rank and `del sharded; gc.collect()` between ranks, returning `[]` (caller at `application_base.py:254` ignores the return value).

**Status**: **Installed, correct — but not exercised at runtime** because the process SIGKILL'd upstream during the load-time dtype cast pass (see L7). The patch remains installed for future use on an instance where L7 is not the dominant peak.

### L7 — NxDI load-time dtype cast pass doubles precision-varying tensors in RAM before any disk write

**Observed in**: Phase 3 CLOSEOUT run (2026-04-22 05:41 UTC). SIGKILL at t=1167.8s during the cast pass, **before** L6's streaming-shard patch was reached. `~/qwen-30b-a3b/traced_model/weights/` was empty at death.

**Mechanism**: `neuronx_distributed_inference/models/application_base.py:655` iterates the loaded checkpoint and emits warnings of the form:

```
UserWarning: Found torch.float32 weights in checkpoint: layers.N.mlp.moe_fused_tkg.down_scales_tkg. Will convert to torch.bfloat16
```

followed by model-side casts:

```
Neuron: casting layers.N.mlp.moe_fused_tkg.down_scales_tkg from torch.bfloat16 to torch.float32
```

For each precision-varying tensor the cast allocates a fresh tensor at the new dtype while the original remains held by the enclosing `checkpoint: Dict[str, Tensor]`. For Qwen3-30B-A3B with 48 layers × quant scales buffers + the full bf16 checkpoint retained, peak DRAM exceeds the 124 GB ceiling on trn3.3xlarge (32 GB swapfile insufficient). **The peak occurs before `shard_weights` ever runs**, so the L6 streaming-shard fix does not help.

**No streaming hook** is exposed in NxDI for this pass. Fix options:
1. **Upstream NxDI change** — add a streaming cast that mutates `checkpoint[k]` in place (`.data = .data.to(dtype)`) and `gc.collect()` between keys. Not attempted in this session.
2. **Offline checkpoint preparation** — pre-cast quantization scales to fp32 in the `hf_model/` checkpoint so the runtime cast pass is a no-op for those keys. Invasive; requires changing the (H1) `_apply_protect_quant` contract and the preshard hook.
3. **Larger-DRAM instance** — straightforward but requires instance-class change beyond trn3.3xlarge.

**Rule**: before declaring a Qwen3-30B-class model trn3.3xlarge-ready, audit every entry in `named_buffers()` and `named_parameters()` for dtype mismatch against the loaded checkpoint. Any mismatch triggers the cast pass. If cumulative mismatched-tensor bytes exceed ~30 GB, expect a load-time DRAM spike in the 1.3×–2× checkpoint-size range.

### L6 ↔ L7 cross-reference

| Aspect | **L6** | **L7** |
|---|---|---|
| Stage | `shard_checkpoint` (post-compile, pre-load) | Load-time dtype cast (before `shard_checkpoint`) |
| NxDI location | `neuronx_distributed/trace/model_builder.py:827-858` | `neuronx_distributed_inference/models/application_base.py:655` |
| Peak source | Python list holding all TP-rank dicts | Original + converted tensors co-resident in checkpoint dict |
| Fixable by us? | Yes — `ModelBuilder.shard_checkpoint` monkey-patch | No — no Python hook point exposed |
| Status | Installed, correct, **not exercised** (superseded) | **Unresolved blocker** on trn3.3xlarge-class instances |

The L6 patch is preserved in `qwen_complete.py` and becomes load-bearing if the workflow runs on a host where L7's peak fits (e.g., larger-DRAM instance or after upstream NxDI adds streaming casts).

---

## Finding 2 — loose correctness tolerance masks critical bugs

**Severity**: Process bug. Affected every attention-kernel correctness check that used `rtol=1e-2, atol=2e-2`.

**Mechanism**: attention output values are small (~1e-2 magnitude given the typical bench's Wq/Wk/Wv/Wo = 0.02 × random). `atol=2e-2` is GREATER than `max |ref|`, so zero output vs ref satisfies `atol` regardless of content. `rtol` is multiplicative and degrades when ref is close to zero.

**Proof**: `v17_fast_exp` bench reports `max_abs_err=1.65e-02` — that IS `max |ref|`. The kernel is returning zeros and the test passes.

**Required fix — triple-check pattern for ALL future wrapper/integration smoke tests**:

```python
got_absmean = float(np.abs(got).mean())
ref_absmean = float(np.abs(ref).mean())
got_max, ref_max = float(np.abs(got).max()), float(np.abs(ref).max())

# 1. zero-check: output must not be identically zero
assert got_absmean > 1e-4, f"[HARD FAIL] wrapper output is zero (absmean={got_absmean:.2e})"

# 2. range-check: magnitude not off by >2x either way
assert got_max > 0.5 * ref_max, f"[HARD FAIL] got max {got_max:.4f} << ref max {ref_max:.4f}"
assert got_max < 2.0 * ref_max, f"[HARD FAIL] got max {got_max:.4f} >> ref max {ref_max:.4f}"

# 3. element-wise allclose (match the kernel's advertised tolerance)
np.testing.assert_allclose(got, ref, rtol=1e-2, atol=2e-2)
```

The zero-check + range-check catch pathological outputs (zero, saturated, shape-misaligned) that the raw `allclose` cannot distinguish when `|ref|` is small.

**Thresholds to tune**:
- zero-check `> 1e-4`: safe for attention outputs of magnitude ≥ 1e-3; raise threshold for kernels with larger outputs.
- range-check 0.5–2.0×: assumes magnitude correctness matters; relax only when you know semantic transformation changes magnitude.

**MoE correctness checks already use bit-exact (`max_diff=0`) comparisons against a reference kernel** (v14a ↔ v15c). Bit-exact is immune to this false-positive mode because zero output on BOTH kernels would produce `max_diff=0` coincidentally only if the reference is also zero — which `probe_v14a_v15c_zero_check.py` has now confirmed is NOT the case. v14a and v15c produce `absmean ~11.6, range [-49, 55]` — genuine MoE outputs.

---

## Finding 3 — wrapper `reshape` on `nisa.dma_copy` destination produces zero output

**Severity**: Design bug in the Round 2 [Z.1+Z.2] integration agent's initial wrapper.

**Mechanism observed**: `output_2d.reshape((NUM_OUT_COLS, PMAX))` as the `dst` of `nisa.dma_copy` does not actually alias to the declared output tensor. The reshape may create a temporary buffer that is not visible to the return value.

**Evidence**: β option whole-tile transpose + `nisa.dma_copy(dst=output_2d.reshape(...), ...)` produced all-zero output (same as α per-column loop with the same reshape-on-dst pattern). Replacing with:
```python
output_flat = nl.ndarray((NUM_OUT_COLS, PMAX), buffer=shared_hbm)
nl.store(output_flat, transposed_sb)
return output_flat.reshape((B, 1, H_wo)), ...
```
produced correct non-zero output (absmean 3.0e-3).

**Rule**: do NOT reshape the `dst` argument of `nisa.dma_copy` or `nl.store`. Allocate HBM in the target shape, or reshape only on the **return** (PyTorch/XLA side, after the kernel function exits).

---

## Remaining risks (from [Z.1+Z.2] integration agent's original 5-item list)

| # | Risk | Status post-Phase-3-STEP-1a |
|---|---|---|
| 1 | v17 wrapper output-order untested on device | **RESOLVED NEGATIVELY**: wrapper was broken, v17_fast_exp itself was broken. Reverted to v13bc_sbm_tiled. |
| 2 | Buffer dtype preservation through `.to(bfloat16)` | Not yet exercised; will surface at first compile. |
| 3 | Memory cost ~2× MoE weight storage | Not yet exercised; compile-time OOM unlikely but possible. |
| 4 | Preshard behavior (non-parameter buffers) | Not yet exercised; most likely place for a compile-time failure. |
| 5 | `config.moe_intermediate_size` post-pad interpretation | Sanity-verified by integration agent; not a known blocker. |

All 4 remaining risks only surface during full-model compile (Phase 3 STEP 2). Proceeding with eyes open.
