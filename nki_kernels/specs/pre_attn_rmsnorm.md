# Pre-Attention RMSNorm — SBUF-Resident TKG Building Block (Spec & Plan)

Status: **PROPOSED — not yet implemented.**
Scope: the GQA (`full_attention`) decoder layer's `input_layernorm`, which today runs in PyTorch
(`modeling_qwen36_a3b.py:2450`) and writes a normed `[B,S,H]` tensor to HBM that the fused GQA TKG
kernel then reads back (`gqa/decode/fused_layer.py:190,204`, both `qkv_tkg` calls use
`NormType.NO_NORM` = "hidden is pre-normed"). Goal: run the norm so its output **persists in SBUF**
and feeds the GQA kernel with **zero HBM round-trip**, as a standalone isolation-testable composable
that later drops into an SBUF-only megakernel.

Reuse: **`nkilib.core.subkernels.rmsnorm_tkg.rmsnorm_tkg` does 100% of the math**. No new kernel math
is written — only caller glue (one `rmsnorm_tkg` call + a model-side norm skip + one new `gamma`
kernel arg).

---

## 0. Ground-truth facts verified (corrections to the brief)

- **Weight is in standard form at runtime, NOT `(1+w)`.** The `(1+w) → w_std` conversion is applied
  once at checkpoint load (`modeling_qwen36_a3b.py:3571-3574`, `neuron_state_dict[nk] = old_val + 1.0`).
  So `input_layernorm.weight` at inference is the standard gamma — feed it directly, no `+1` in-kernel.
  Precedent: the DeltaNet verify path already passes `gamma = input_layernorm.weight.reshape(1, H)`
  straight into its fused kernel (`modeling_qwen36_a3b.py:710`).
- **The gate DOES consume normed hidden (brief is correct).** In `_forward_tkg_kernel` the kernel
  receives the already-normed `hidden_states` and both projections (qkv at `:190`, gate at `:204`) read
  it. In the non-kernel path `gate = self.output_gate_proj(hidden_states)` (`:2049`) also reads normed
  hidden (norm applied at `:2450` before `self_attn`). So both projections want the *same* normed tile.
- **`input_layernorm` is NOT TP-sharded.** It is a plain `get_rmsnorm_cls()(config.hidden_size, ...)`
  module (`:2423`), replicated on every rank; gamma is full `[1, H=2048]`, norm is over full H,
  computed identically (replicated) on each rank.
- **Runtime norm = plain RMSNorm, fp32 internal.** Default `get_rmsnorm_cls()` returns `CustomRMSNorm`
  (`RmsNorm.apply`, plain rsqrt, fp32 accumulate) — matches `rmsnorm_tkg` (`inter_dtype=nl.float32`,
  `nl.rsqrt`, `scale=1/H`, `bias=eps`). `NewtonRMSNorm` (Newton-refined rsqrt) is opt-in only
  (`USE_NEWTON_RMSNORM`) and is **not** the default — do not replicate its refinement in-kernel.
- **eps** = `config.rms_norm_eps` (passed as `self.rms_norm_eps`); pass it explicitly (kernel default is
  `1e-6`; Qwen value is `1e-6`).

---

## 1. Contract

### Dims (A3B, TP=4, LNC=2, bs=1)
`H = 2048`, `H0 = 128`, `H1 = H/H0 = 16`. `T = B·S` = **1 (decode)** / **2 (verify)**.

| Tensor | Role | Shape | Dtype | Layout |
|---|---|---|---|---|
| `hidden` (raw) | input, pre-norm | HBM `[B,1,H]=[1,1,2048]` *(T=1)* / `[1,2,2048]` *(T=2)*; or SBUF `[128, T, 16]` (megakernel) | bf16 | `[B,S,H]` HBM, or `[H0,T,H1]` SBUF |
| `gamma` | input_layernorm.weight | HBM `[1, H]=[1,2048]` | bf16/fp32 | replicated, full H (not sharded) |
| `eps` | scalar | — | fp32 | `config.rms_norm_eps` |
| **`normed`** | **output (SBUF-resident)** | SBUF `[128, T, 16]` | bf16 | `[H0, T, H1]` — the exact `qkv_tkg` SBUF-input layout |
| `hidden` (raw) | **untouched** | unchanged | bf16 | preserved for residual (see §5) |

### SBUF handoff layout to `qkv_tkg`
`qkv_tkg` accepts SBUF input as `[H0=128, BxS, H1]` (asserts `H0==128`, `qkv_tkg.py:489-490`) and, in
its `NO_NORM` path, **slices its own H-shard** off dim 2: `[:, :, shard_id*H1_shard : (shard_id+1)*H1_shard]`
(`qkv_tkg.py:867-869`, `H1_shard = 8` at LNC=2). So `normed` must be the **full** `[128, T, 16]` tile,
replicated on both cores; each core slices its 8 H1-tiles. `rmsnorm_tkg`'s output is exactly
`[128, BxS, H//128]` (`rmsnorm_tkg.py:70,84`) → **zero reshape**, drop-in.

**Layout-match is guaranteed by construction**, not by luck: `qkv_tkg`'s *own* built-in `RMS_NORM` path
calls this same `_rmsnorm_tkg(input, output=[H0,BxS,H1])` and then slices dim 2 by shard
(`qkv_tkg.py:879-909`). Hoisting that call out and feeding the result back in via `NO_NORM` reproduces
the validated internal dataflow byte-for-byte, shared across both projection calls.

---

## 2. Design recommendation: **B (norm once into a shared SBUF tile)** — decisive

Two designs:

- **(A) Flip both `qkv_tkg` calls to `RMS_NORM`** (pass `norm_w = input_layernorm.weight`). One-line
  change, max reuse — **but norms H=2048 twice** (once per projection call) and, critically, is
  **megakernel-incorrect** (see below).
- **(B) Call `rmsnorm_tkg` once** → `normed [128,T,16]` SBUF tile, feed **both** `qkv_tkg` calls with
  `NO_NORM`. Norms once; literally "persist the normed output in SBUF for the GQA kernel."

**Recommend B.** Reasons, concrete:

1. **A clobbers the residual in the megakernel (disqualifying).** `qkv_tkg`'s `RMS_NORM` path with an
   SBUF input writes the normed result **in-place over the input tile**: `hidden_sb = hidden` then
   `_rmsnorm_tkg(input=hidden, output=hidden_sb)` (`qkv_tkg.py:880-897`). In an SBUF-only megakernel the
   raw hidden tile IS the residual — A destroys it on the first call, and the second (gate) call then
   double-norms the already-normed tile. B keeps raw and normed in separate tiles by construction.
2. **A doubles the norm work.** Redundant cost of A = one extra full RMSNorm over `[128, T, 16]`:
   `nisa.activation(square)` + `tensor_reduce(H1)` + `tensor_tensor(gamma)` + `nc_matmul(128×128 reduce)`
   + `rsqrt` + `tensor_tensor(final)` ≈ **3 elementwise passes over 2048·T elements + 1 H1-reduce + 1
   128×128 matmul + ~7 instruction issues**, *per layer, every decode step*. At T=1 the FLOPs are
   trivial (~64 vector cycles), but the decode round is **serialization / instruction-issue bound** (no
   engine >27% of wall, per profiling notes), so the **~7 redundant instructions/GQA-layer** are the
   real cost B removes — and they recur across all `full_attention` layers.
3. **B's extra cost is exactly what the megakernel wants.** One extra SBUF tile
   `[128, T, 16]` bf16 = `128·16·T·2` = **4 KB (T=1) / 8 KB (T=2)** — negligible vs ~24 MB SBUF — held
   briefly. That tile is the persisted normed output the megakernel composes downstream. No layout cost:
   §1 proves zero reshape.

Net: B norms once, is the only residual-safe option for the SBUF megakernel, and its only "extra" is a
4–8 KB tile that the design needs anyway. A's only advantage (one fewer call site) is outweighed.

---

## 3. nkilib reuse table

| `file:line` | What | Verdict | Caller adds |
|---|---|---|---|
| `core/subkernels/rmsnorm_tkg.py:45` `rmsnorm_tkg` | the norm math (square→reduce→rsqrt→gamma), fp32 internal, BxS<18 auto single-core | **drop-in ✓** | call once: `rmsnorm_tkg(input=raw, gamma=input_layernorm.weight, output=normed_sb[128,T,16], eps=cfg.rms_norm_eps, hidden_actual=H)` |
| `core/qkv/qkv_tkg.py:867-869` `NO_NORM` SBUF-input path | slices its H-shard from the full `[128,T,16]` tile, projects | **drop-in ✓** (the handoff) | feed `hidden=normed_sb`, `norm_type=NO_NORM`, `output_in_sbuf=True` (already set in `fused_layer.py:194,201,208,215`) |
| `core/qkv/qkv_tkg.py:879-909` `_fused_norm_and_load` `RMS_NORM` path | qkv's built-in fused norm (= Design A) | **avoid ⚠** | not used: double-norms across the 2 calls and writes normed in-place over raw SBUF input (clobbers residual). Use only as proof the output layout matches (§1). |
| `core/utils/common_types.py:30-34` `NormType` | `NO_NORM=0` enum | **drop-in ✓** | import (already imported in `fused_layer.py:41`) |
| `rmsnorm_tkg.py:39` `SHARDING_THRESHOLD=18` | LNC-shard gate | **drop-in ✓** | nothing — T≤2 < 18 ⇒ `do_shard=False`, each core computes the full replicated norm, no sendrecv (`rmsnorm_tkg.py:212-214`). Matches the unsharded `input_layernorm`. |

Hand-written glue (no library analog): (a) the model-side change to **skip `input_layernorm` for GQA**
and pass raw hidden + `gamma` (mirrors the existing verify-DeltaNet skip at `:2449,2459`); (b) the single
`rmsnorm_tkg` call + new `gamma` kernel arg in `gqa/decode/fused_layer.py`. No kernel math.

---

## 4. Isolation test plan

Build/test the composable standalone. SBUF cannot persist across NEFF launches, so in isolation the
kernel ends with a `dma_copy(normed_sb → normed_hbm[128,T,16])` (or `[T,H]`) purely so the test can read
it; the megakernel build omits that store and hands `normed_sb` to `qkv_tkg` in-SBUF.

1. **Inputs:** random bf16 `hidden [1,T,2048]` (T∈{1,2}), `gamma = randn(2048)*0.02 + 1.0` (standard
   form, mimicking the `+1` checkpoint convention), `eps=1e-6`.
2. **Kernel:** `rmsnorm_tkg(input=hidden, gamma=gamma[None,:], output=normed_sb, eps, hidden_actual=2048)`
   → DMA `normed_sb` to HBM; un-permute `[128,T,16] → [T,2048]` on host to compare.
3. **CPU oracle:** `Qwen3MoeRMSNorm(2048, eps)` with `weight=gamma`, run in **fp32** on the same input:
   `x32 = x.float(); out = x32 * rsqrt(mean(x32^2,-1,keepdim)+eps) * weight`. (Matches the runtime
   `CustomRMSNorm`/`RmsNorm.apply`; **do not** use `NewtonRMSNorm` — different rsqrt.)
4. **Numeric gate (repo rules — cosine similarity BANNED):**
   - **Hard gate:** fp32 run of the kernel path `allclose(kernel_fp32, oracle_fp32, rtol=1e-4, atol=1e-5)`.
   - **bf16 gate:** `max_abs(kernel_bf16 − oracle_fp32) ≤` the measured bf16 quantization floor
     (compute the floor as `max_abs(oracle_fp32.bf16() − oracle_fp32)` on the same tensor, with small
     headroom), NOT cosine.

---

## 5. Megakernel integration handoff

**SBUF normed tile → GQA composable, zero HBM round-trip:**
- In `gqa/decode/fused_layer.py::gqa_fused_compose`, before Stage 0, insert one
  `normed_sb = rmsnorm_tkg(hidden_raw, gamma, out=[128,T,16], eps, hidden_actual=H)`; replace the two
  `qkv_tkg(hidden=hidden, norm_type=NO_NORM, ...)` (`:190` qkv, `:204` gate) with
  `qkv_tkg(hidden=normed_sb, norm_type=NO_NORM, ...)`. Both already pass `output_in_sbuf=True`. Add a new
  `gamma` parameter to `gqa_fused_compose` / `gqa_fused_tkg_fwd` and thread `input_layernorm.weight`
  through `_forward_tkg_kernel` (`:2156`). The kernel's `hidden` arg becomes **raw** (docstring at
  `fused_layer.py:339` "PRE-NORMED hidden" flips to "raw, in-kernel normed").
- **What changes in `fused_layer.py`:** the two `NO_NORM` calls are **fed from the SBUF `normed_sb`**
  (Design B), *not* flipped to `RMS_NORM` (Design A is rejected, §2).

**Raw hidden survives for the residual:**
- Today the residual is added in PyTorch: `residual = hidden_states` (raw, `:2442`) and
  `hidden_states = residual + attn_out` (`:2498`). The norm consumes raw and produces normed in a
  *separate* tile; raw is never written. So the model must **stop applying `input_layernorm` for GQA**
  (extend the existing skip condition at `:2449` to `full_attention` too) and pass raw `hidden_states` +
  `gamma` to `_forward_tkg_kernel`. `residual` stays the raw pre-norm hidden exactly as now → residual
  semantics preserved bit-for-bit.
- In the eventual full megakernel, the residual add also moves on-device: raw hidden lives in its own
  SBUF tile (Design B never clobbers it), normed lives in `normed_sb`; after o_proj, `raw + attn_out`
  is an SBUF add. This is downstream of this building block.

---

## 6. Open questions / risks (honest)

1. **Megakernel raw-hidden SBUF layout.** For the *current* integration raw hidden is HBM `[B,S,H]`
   and `rmsnorm_tkg`'s HBM-input path produces the qkv-matching column order (proven: qkv's own norm
   path uses it). For the *SBUF-input* megakernel case, `rmsnorm_tkg` skips the H-shard-interleaving
   `load_input_to_sbuf` and trusts the input tile's existing `[128,T,16]` column order
   (`rmsnorm_tkg.py:672-673`). So **the upstream composable that produces the raw SBUF hidden must emit
   the same H1 column order qkv_tkg's weight expects.** Verify when wiring the megakernel; until then the
   HBM-input path is unambiguous. **Open.**
2. **Where the norm lives — inside `gqa_fused_tkg_fwd` vs a separate launch.** This spec puts the
   `rmsnorm_tkg` call *inside* the GQA fused kernel (one launch, SBUF handoff). A standalone NEFF is for
   isolation testing only. Confirm we don't want a separately-launchable norm composable for reuse by
   the MoE `post_attention_layernorm` path (likely yes later, but out of scope here).
3. **bf16 gamma precision.** `input_layernorm.weight` is stored bf16; the `+1.0` conversion is done in
   fp32 then cast. `rmsnorm_tkg` loads gamma at its native dtype and multiplies in fp32. The bf16 gamma
   rounding is folded into the bf16-floor gate (§4) — not a separate risk, but the oracle must use the
   **same bf16-rounded gamma** to avoid a false fail.
4. **Single-core replication assumption.** Relies on `BxS=T≤2 < SHARDING_THRESHOLD=18` so both cores
   compute the full norm. If a future verify uses T>18 (won't at k=1) the sharded path engages and the
   sendrecv cost returns; fine, still correct, just note the threshold.
5. **No `RMS_NORM_SKIP_GAMMA`.** `NormType` has a `SKIP_GAMMA=3` variant (`common_types.py:34`); not
   relevant here (Qwen GQA has a real gamma), listed only to confirm it was considered and rejected.
