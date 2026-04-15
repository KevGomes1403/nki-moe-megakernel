# NKI Debugging Learnings

Patterns and pitfalls collected while debugging NKI kernels on Trainium. Each entry is a concrete failure mode that cost real time, with the observable symptom, the root cause, and the fix.

---

## 1. `nc_matmul` contracts on the **partition dim** — laying weights out wrong silently produces the wrong linear map

### Symptom
Kernel output differs from PyTorch reference by a large factor (e.g. 4×–10× magnitude), but the kernel compiles, runs, and produces output that "looks" structurally correct (right shape, right sign distribution). On scaled-down inputs (`W *= 0.02`) the error can be small enough to squeak under a loose tolerance and *pass the test* — the bug stays latent.

### Root cause
`nisa.nc_matmul(result, stationary, moving)` computes `result[i, j] = sum_k stationary[k, i] * moving[k, j]` where `k` is the **partition dimension** of both operands (the contraction axis). If the weight tile is DMA-loaded such that partition = output-dim (`d`) instead of partition = contraction-dim (slice of `H`), the matmul is effectively `Wᵀ · h` rather than `W · h` — a completely different linear map that happens to have the right output shape.

Concretely: for `k = Wk · x` with `Wk: [d, H]`, a direct `nisa.dma_copy(dst=wk_full, src=Wk)` where `wk_full: [PMAX=d, H]` puts partition=d. `nc_matmul(stationary=wk_full_slice [P=d, F=H_slice], moving=h_tile [P=h_within_chunk, F=1])` then contracts over partition, pairing `d`-indices with `h`-indices — garbage.

### Fix
Load weight tiles with an explicit DMA-transpose pattern so partition dim = contraction dim:

```python
# Instead of: nisa.dma_copy(dst=wk_full, src=Wk)       # partition = d (WRONG)
# Do this, per tile, with partition = H-slice (CORRECT):
nisa.dma_copy(
    dst=wk_tile,                                       # [PMAX=H-slice, d]
    src=Wk.ap(pattern=[[1, PMAX], [H, PMAX]],
              offset=h_t * PMAX),                      # wk_tile[p, f] = Wk[f, h_t*PMAX + p]
    dge_mode=nisa.dge_mode.hwdge,
)
```

### How to notice it earlier
- When writing a new kernel, **write down the partition-dim semantics of every SBUF tensor** and verify both operands of every `nc_matmul` name the same semantic axis for their partition dim. If a `[d, H]` weight and an `[h_tile, 1]` activation both have partition-count 128, that's a *shape* match but not a *semantic* match.
- Test with **raw std-normal weights at least once**. Scaled-down tests (`W *= 0.02`) tighten absolute tolerances but hide magnitude bugs. Run both regimes — tight-tolerance/scaled for precision, raw-scale for structural sanity (with loosened tolerance).

---

## 2. `@nki.jit` in-place HBM mutations must be **returned** from the wrapper

### Symptom
Kernel writes to an HBM input tensor via indirect DMA (e.g. KV cache scatter). Compile and run succeed. The scatter DMA is visible in the NEFF. But after the kernel call, reading the input tensor from the host shows the *original* values — the mutation never landed.

### Root cause
`@nki.jit` / XLA aliasing: if the mutated tensor is only an input and isn't returned, XLA's SSA model doesn't know to propagate the mutation back. The DMA writes into a region that the caller's view never re-reads.

### Fix
Return the mutated tensors from the wrapper alongside the "real" output:

```python
@nki.jit
def wrapper(..., K_cache, V_cache, ..., output):
    ...
    return output, K_cache, V_cache   # even though scatter is "in place"
```

Then unpack on the call site: `out, K_out, V_out = wrapper[2](...)`.

### How to notice it earlier
If your kernel mutates an HBM input and the post-call read looks unchanged, check the return signature before blaming the DMA. A one-line diagnostic: scatter a known constant and read it back — if the constant isn't visible, it's an aliasing problem, not a pattern problem.

---

## 3. Indirect-DMA `scalar_offset` tensor dtype must survive HWDGE's same-type rule

### Symptom
Compile fails with:
```
'nisa.dma_copy' op HWDGE requires same src/dst element type, got src='i32' vs dst='ui32'.
Use SWDGE or NoDGE for type casting.
```

### Root cause
`nisa.dma_copy(..., scalar_offset=tensor, indirect_dim=k)` wants an unsigned-integer SBUF tensor (matches what `nkilib`'s reference `update_kv_cache` uses: `nl.uint32`). `position_ids` arrives from NxDI as `int32`. A single HWDGE copy can't cast across signed/unsigned, so loading `position_ids` directly into a `uint32` SBUF tensor fails.

### Fix
Two-step load: DMA into an `int32` SBUF tensor, then `tensor_copy` (which can cross dtypes) into a `uint32` SBUF tensor used as `scalar_offset`:

```python
pos_i32 = sbm.alloc_stack((1, 1), nl.int32, name="pos_i32")
nisa.dma_copy(dst=pos_i32, src=position_ids[0:1, 0:1], dge_mode=nisa.dge_mode.hwdge)
pos_u32 = sbm.alloc_stack((1, 1), nl.uint32, name="pos_u32")
nisa.tensor_copy(pos_u32, pos_i32)   # signed→unsigned cast lives on a compute engine, not DMA
```

### Related
`dge_mode.swdge` / `dge_mode.no_dge` also allow cross-type DMA but are slower and aren't always available for indirect patterns. The compute-engine cast is usually the right tool.

---

## 4. Causal-mask off-by-one in TKG flash-decode (`j == pos` must be masked)

### Symptom
Attention output magnitude is ~10× the reference. All 2048/2048 elements mismatch. Random-looking relative errors (not a clean scale factor). Tight early indication: re-scaling weights changes the magnitude ratio but not the mismatch count.

### Root cause
In TKG, the new token's K/V is computed in-kernel but **not yet scattered into the cache** when the flash-decode tile loop runs — the cache slot at index `pos` still holds stale/uninitialized values. The correct pattern is:

- **Cache-tile loop** masks `j >= pos` (including `pos`) with `-1e9` — those rows are garbage.
- **Active path** separately computes `q · k_new` and `softmax_weight · v_new` for the single new token and adds them into the accumulator.

If the mask uses `relu(j - pos)` directly, `j == pos` maps to 0 (unmasked), the garbage cache row leaks into the softmax, AND the active path also fires — double-counting with a random addend.

### Fix
Match the reference idiom (`v13_BC.py:546-549`, `attention_block_tkg`, `v13bc_sbm_tiled.py:408-411`):

```python
delta     = par_index_f32 + neg_thresh_sb           # j - pos
delta_eps = delta + 1.0                             # the critical +1 shift
relu_d    = relu(delta_eps)
clamped   = min(relu_d, 1.0)                        # binary {0,1}
mask      = clamped * -1e9                          # -1e9 iff j >= pos
```

Why `+1`: `delta = -1` at `j = pos-1` (last valid) → `delta_eps = 0` → `relu = 0` → unmasked ✓. `delta = 0` at `j = pos` → `delta_eps = 1` → `relu = 1` → masked ✓.

### How to notice it earlier
When porting a TKG kernel, always write the contract on the TOP of the file:
> "Cache loop masks `j >= pos`. Active path supplies the new token."
If the mask math and the active-path wiring don't both agree with that contract, the kernel is wrong even if it looks correct.

---

## 5. bf16 noise floor vs. assertion tolerance — don't hide it with scaling unless you know why

### Symptom
Test asserts `rtol=1e-2, atol=1e-2` against a PyTorch fp32 reference, kernel bf16. With raw std-normal inputs, output values have std ~√H (e.g. 45 for H=2048). 1% relative error on magnitude-45 values is ~0.5 absolute — exactly the bf16 mantissa precision floor (~7 bits → ~0.8% relative). The test fails on scattered elements even when the kernel is correct.

### Mitigation
Scale weights to initialization-realistic magnitudes (e.g. `W *= 0.02` for `std = 2/√fan_in`-style init). This keeps output values in a regime where bf16 quantization is well below the 1% tolerance. The `test_v13bc_sbm_tiled.py` / `test_v13bc_kv_norm.py` convention is `W *= 0.02`, `cache *= 0.1` — realistic for Qwen3-MoE.

### Don't abuse this
Scaling is the right move only when:
- The numerical regime matches production (real model weights are O(σ_init)).
- The assertion tolerance reflects bf16, not fp32.

If you find yourself lowering scale to pass a test for unclear reasons — stop. You're masking a real bug. Run an unscaled sanity check with a loosened tolerance (`atol=0.05 * std_of_output`) to verify the kernel is structurally right before tightening.

---

## 6. SBUF scope / BufferManager lifetime across in-kernel KV scatter

### Symptom
Kernel compiles but scatter value is wrong, or scatter reads the wrong position.

### Root cause checklist
- The SBUF tensor holding the K/V to be scattered (`k_rope_T_sb`, `v_T_sb` here) **must be allocated at a scope that survives until after flash-decode** if the scatter happens at end-of-kernel. If it's inside a nested `kv_proj` scope that closes before the scatter DMA, the compiler's BufferManager will have reused that SBUF region for something else.
- The `scalar_offset` tensor (`pos_write_i32` / `pos_write_u32`) must be allocated and filled at a scope that survives both the mask-computation use (pass 1 setup) and the scatter use (end of kernel). The outermost `attn_outer` scope is the usual home.

### How to notice it earlier
Add `sbm.stat_log()`-equivalent diagnostics before each close-scope. If you see an unexpectedly low max_usage mid-kernel, something live is being dropped early.

---

## 7. Hardware rules that bit during this session

- **LNC=2 program split**: if the kernel guards `if n_prgs == 1 or prg_id == 0: ...` / `== 1: ...`, the caller MUST launch with `kernel[2](...)` AND the environment must permit it (`NEURON_LOGICAL_NC_CONFIG=2` on trn2; trn3 defaults). Forgetting the env var on a guarded kernel silently drops one LNC branch — half the scatters never happen.
- **Both LNC cores writing the same output**: two cores issuing the identical DMA to the same `output` HBM is a benign race on value but not on ordering — watch for `-Oaliasing` warnings in compile logs.
- **`@nki.jit`'s `[n_prgs]` subscript is not a shape hint — it's a program-count directive**; the kernel's `nl.num_programs(0)` reflects it, and guards based on `prg_id` will be silently dead if you forget.

---

## 8. Reading logs: what "passes" actually means

- A `.neff` in `/var/tmp/neuron-compile-cache/` from a prior session will be reused — **stale kernels can appear to "pass" a test** because the compile cache never invalidates on syntax-ambiguous edits. When in doubt: `rm -rf /var/tmp/neuron-compile-cache/*` or run with a unique `--cache-dir`.
- `CORRECTNESS: PASS` in a sibling test does not mean the kernel body is correct, only that the sibling's inputs and tolerances happened to not expose a latent bug. See §1.

---

## Debugging playbook

When a TKG-style fused attention kernel fails correctness:

1. **Isolate the three channels**: attention output, `new_k` readout, `new_v` readout. Which fails first tells you whether scatter, projection, or flash-decode is broken.
2. **Print max/std of reference vs. kernel output** before looking at element-wise diffs — a uniform scale factor points to a missing/extra `1/√d`, a doubled active path, or a matmul axis bug.
3. **Diff against the known-good sibling kernel** (here `v13bc_sbm_tiled.py`). Account for every line that differs. "It's a copy-paste with one extra scope" is often a lie.
4. **Revert one variable at a time** (scaling, LNC count, mask form) and re-run. A 5-minute cycle here saves hours of fruitless theorizing.
5. **Check the return signature** of the `@nki.jit` wrapper before accusing the DMA.
6. When the bug is finally found, **write it down here** — NKI's silent-failure modes are numerous and mostly undocumented.
