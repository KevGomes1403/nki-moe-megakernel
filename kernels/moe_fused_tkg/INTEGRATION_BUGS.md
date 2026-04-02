# Integration Bug Analysis: Repetitive Output in qwen_fused_moe_tkg.py

**Symptom**: Model generates coherent but repetitive text during token generation (TKG path). Base model (`qwen.py`) does not exhibit this behavior. Double all-reduce was ruled out via profiler.

---

## Bug 1: Float32 Router Weights in a BF16 Kernel

### Where
`qwen_fused_moe_tkg.py`, `_install_nki_router()` (line ~380) and the TKG forward pass (line ~676).

### What happens
`_install_nki_router` explicitly casts `moe_fused_tkg.router.weight_T` to float32:

```python
old.weight_T = nn.Parameter(old.weight_T.data.to(torch.float32))
```

This is correct for the *standard NXD TKG kernel*, which requires float32 router weights (`router_mm_dtype=torch.float32`). But the custom NKI kernel (`kernel_v20a`) declares its router weight SBUF as `dtype=inp.dtype` (bf16):

```python
router_w_wide_sb = nl.ndarray((_PMAX, _ROUTER_BATCH, E), dtype=inp.dtype, buffer=nl.sbuf)
```

When `tkg.router.weight_T` (float32) is passed to the NKI kernel, XLA likely auto-casts it to bf16 before the DMA. The router matmul then runs in bf16 precision (both operands are bf16, accumulating to float32 PSUM), whereas the base model computes router logits in float32.

### Why this causes repetition
With 128 experts and top-8 selection, bf16 has ~2× less precision in logits than float32. At the margins (experts with very similar logits), different experts get selected compared to the base model. With bf16-quantized softmax outputs, ties between expert probabilities are also more likely, which can make `nisa.nc_find_index8` return an unexpected index for one of the 8 positions.

These small per-layer routing errors compound over 94 layers: the hidden state drifts away from the true distribution, the next-token logits become increasingly peaked over a subset of tokens, and the model enters a repetition loop.

### Fix / Verification
The router weight passed to the custom NKI kernel should be bf16, not float32. Concretely:
- Either keep `moe_fused_tkg.router.weight_T` as bf16 (and update `_install_nki_router` to NOT cast it to float32 — the float32 requirement only applies to the standard NXD TKG kernel, which is being bypassed in the TKG path)
- Or add an explicit `.to(torch.bfloat16)` cast at the call site before passing to the kernel

**Debugging shortcut**: Log `top8_idx` values from the kernel vs. the base model's router for the same input. Even 1-2 mismatches out of 8 per layer is enough to explain the drift.

---

## Bug 2: BF16 Precision in Router Logits (Compounding with Bug 1)

### Where
`kernel_v20a.py`, Stage 2 (router matmul) and Stage 3 (softmax + TopK).

### What happens
Even if the weight dtype issue above is fixed, the router matmul uses bf16 operands:

```python
nisa.nc_matmul(
    dst=logits_psum[0:T, 0:E],           # float32 PSUM
    stationary=rmsnorm_normed_bf16[...],  # bf16
    moving=router_w_wide_sb[...],         # bf16 SBUF
)
```

The base model's standard NXD TKG kernel promotes hidden states to float32 before the router matmul. This kernel computes the matmul in bf16, accumulating to float32. The logits land in float32 PSUM (`logits_sb` is float32), so the softmax runs in float32 — but the matmul precision is bf16.

The practical impact: for experts with logit differences < ~0.01 (common in a 128-expert model), bf16 rounding can flip which expert wins top-8 selection.

### Fix / Verification
Consider using float32 for the router matmul stationary and moving operands if the hardware supports it, or accept the precision difference and verify it does not affect output quality at scale.

---

## Bug 3: Systematic MLP Computation Error (Newly Confirmed)

### Where
`kernel_v20a.py`, Stage 4 (expert MLP: gate/up/down matmuls, SiLU, output accumulation).

### What happens
Exhaustive correctness testing against `reference.py` (PyTorch CPU, seed sweep 0–9, B=1) shows:
- **0/10 seeds pass** `allclose(rtol=0.05, atol=0.05)`
- max_diff ranges from 0.125 to 0.5 across all seeds
- Failures are **systematic** — they occur even on seeds where the kernel's bf16 router and the reference fp32 router select identical top-8 experts

Because the routing is identical on most seeds yet the outputs still diverge, the error must originate in the MLP computation itself, not in the router. Candidates:

- LNC sharding of `down_w`: each NeuronCore owns `H_shard = 1024` output columns, selected by `prg_id * H_shard` offset. If the shard boundary or reassembly is off, the output columns are wrong regardless of routing.
- PSUM accumulation order across H-tiles in the down matmul (see `affine_range(H_free_shard)` loop).
- `inter_bf16` precision loss before the down matmul: intermediate is cast to bf16 from fp32 `inter_f32`, discarding ~7 bits of SiLU output precision.
- Output scaling (`aff_bcast` broadcast logic or `norm_weights` computation).

### Why this causes repetition
Same mechanism as Bugs 1/2: wrong MLP outputs corrupt the hidden state per layer, and over 94 layers the distribution collapses. Test 5 (single-expert swap) quantifies the sensitivity: one wrong expert index yields max_diff ≈ 26 vs output magnitude ≈ 0.1 — a complete corruption of the token contribution.

### Fix / Verification
This bug must be isolated before addressing Bugs 1/2. Suggested bisect:
1. Patch `reference.py` to use the same bf16 inter cast: `inter = F.silu(gate.to(bfloat16)) * up.to(bfloat16)` — if diffs shrink, the cast is the cause.
2. Run with a single expert (force topk_idx[0,0] to a known expert, K=1) and compare the down matmul output against a manual `x_norm_bf16 @ gate_w, silu, @ down_w` in PyTorch.
3. Check prg_id=0 vs prg_id=1 output slices independently to isolate LNC shard reassembly.

Test script: `kernels/moe_fused_tkg/tests/verify_v20a_vs_pytorch.py`

---

## Bug 4: Expert Routing Sensitivity (Empirical)

### What the verification tests showed
Test 5 in `verify_v20a_vs_pytorch.py`: swapping a single expert at position k=0 (seed=1) produces:
- max_diff = **26** (vs output magnitude ~0.1)
- mean_diff = **5.7**

This confirms that routing errors are **not benign** — a single wrong expert in 8 corrupts the output catastrophically. Over 94 layers even a 1-in-20-seed routing mismatch rate would cause visible output drift.

---

## Summary Table

| Bug | Root cause | Confirmed? | Likely impact | Fix |
|-----|-----------|-----------|---------------|-----|
| Float32 router weights | `weight_T` cast to fp32 for standard NXD TKG kernel, passed as-is to custom bf16 NKI kernel | Partially (1/20 synthetic seeds flip 2/8 indices) | Wrong expert selection at margins, compounds over 94 layers | Keep `weight_T` as bf16 for the custom kernel path |
| BF16 router matmul precision | Kernel computes router GEMM in bf16 vs fp32 in base model | Yes (~0.025 logit error, flips indices at margins) | Amplifies bug 1, causes marginal expert mismatches | Accept or promote router operands to fp32 |
| Systematic MLP computation error | Unknown — down matmul sharding, inter bf16 cast, or output accumulation | **Yes — 0/10 seeds pass vs PyTorch reference** | Corrupts output on every token regardless of routing | Bisect via single-expert test, check LNC shard reassembly |
| Expert routing sensitivity | Single wrong expert completely corrupts token output (max_diff=26) | Yes (empirical) | Catastrophic per-layer error amplification | Fix routing precision bugs; treat as compounding factor |
