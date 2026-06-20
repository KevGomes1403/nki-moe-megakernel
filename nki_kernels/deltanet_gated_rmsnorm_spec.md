# DeltaNet Gated Per-Head RMSNorm — TKG Kernel Spec & Implementation Plan

Status: **IMPLEMENTED & on-device validated** (kernel `nki_deltanet_norm_gate.py`, integrated into the
fused TKG kernel behind optional `z`/`gamma`/`eps`; test `tests/test_deltanet_fused_tkg_kernel.py` PASSes
vs HF `Qwen3_5MoeRMSNormGated` at atol=1e-5/rtol=1e-2). No precedent in nkilib — custom.
Scope: the back-half of the DeltaNet layer between the fused conv+recurrence kernel and `out_proj`.
This is the `attn_output()` pre-projection step in `modeling_qwen36_a3b.py:466-480`.

> **API reality (as-built, supersedes the "NKI 0.3.0" notes below):** the installed SDK is **NKI 0.4.0**,
> where **`nisa.rsqrt` does not exist**. The kernel uses `nisa.activation(op=nl.rsqrt, data=, scale=1/d,
> bias=eps)` — the same form the existing recurrence kernel uses (`nki_deltanet_tkg.py:186`), which also
> folds mean-scale+eps into one Scalar-engine instruction (the optimization §7 wanted). `activation_reduce`
> can't do the per-head reduce (it collapses the whole free dim), so square+`tensor_reduce(axis=(2,))` is
> used. `nl.silu` is available. Treat the "use `nisa.rsqrt`" lines in §5/§7/§8 as obsolete.

---

## 1. What this kernel computes

After the fused conv+recurrence kernel emits the raw head-major recurrence output, the model does
(`modeling_qwen36_a3b.py:474-479`):

```python
output = attn_raw.to(dtype)                                   # f32 -> bf16 round
output = output.reshape(B, S, num_v_heads, head_v_dim)        # head-major view, head_v_dim=128
output = self.norm(output)                                    # per-HEAD RMSNorm over head_v_dim (Qwen3MoeRMSNorm)
z_gate = z.reshape(B, S, num_v_heads, head_v_dim)
output = output * F.silu(z_gate)                              # SiLU gate from in_proj_z
output = output.reshape(B, S, value_dim)
# -> out_proj(output)   (separate kernel; NOT part of this spec)
```

So one fused op: **per-head RMSNorm over the 128-wide value-head dim, followed by a SiLU gate by `z`.**
This is a `RMSNormGated`-style operation, but nkilib has no kernel for it (its `rmsnorm_tkg` is
full-hidden, ungated). Hence "no precedent → build it."

Math, per (token `t`, value-head `h`), over `j = 0..127`:

```
x   = attn_raw[t, h, :]                       # 128-vec, recurrence output (f32)
var = mean_j(x_j^2)                           # scalar per (t,h)
y_j = x_j * rsqrt(var + eps)                  # plain rsqrt (NOT Newton-Raphson)
n_j = y_j * gamma_j                           # gamma = self.norm.weight, shape [128], shared across heads
o_j = n_j * silu(z[t, h, j])                  # silu(u) = u * sigmoid(u)
```

`eps = 1e-6` (config `rms_norm_eps`). `gamma` is a single `[128]` vector shared by all heads.

### Numerics parity (match `Qwen3MoeRMSNorm`, NOT the Newton variant)
- `self.norm` is hardcoded `Qwen3MoeRMSNorm` (`modeling_qwen36_a3b.py:372`) → **plain `rsqrt`**, no
  Newton-Raphson refinement. Do **not** add the `y*(3 - var*y*y)*0.5` correction here. (The NR norms in
  this model are the decoder/attention norms via `get_rmsnorm_cls()`; they don't touch this path.)
- Reference rounds `attn_raw` (f32) → `bf16` **before** the norm (`output = attn_raw.to(dtype)`), then
  `Qwen3MoeRMSNorm` upcasts to f32 for the variance, normalizes, casts back to bf16, multiplies by the
  bf16 `gamma`, then multiplies by `silu(bf16 z)`. For bit-parity the kernel should: round input to bf16
  first, accumulate the variance in **f32**, produce the normalized value in bf16, gamma-multiply in bf16,
  gate in bf16. (A pure-f32 path is marginally more accurate but won't bit-match; validate against the
  bf16 reference.)

---

## 2. Per-rank / per-core constraints (this model, TP=4, LNC=2)

Ground-truth from `config.json` (`text_config`): `hidden_size=2048`, `linear_num_value_heads=32`,
`linear_num_key_heads=16`, `linear_value_head_dim=128`, `linear_conv_kernel_dim=4`, `rms_norm_eps=1e-6`,
40 layers. **Ignore the stale `1536`/`48-head` numbers in the modeling-file comments.**

| Quantity | Global | Per rank (TP=4) | Per core (LNC=2 value-head shard) |
|---|---|---|---|
| value heads `Hv` | 32 | **8** | **4** (`Hv_core`) |
| head dim `d` | 128 | 128 | 128 |
| `value_dim` / `W` | 4096 | **1024** | **512** (`W_core = Hv_core*d`) |
| `gamma` | `[128]` | `[128]` (replicated) | `[128]` (replicated) |
| tokens `T` | — | 1 (decode) / 2 (verify) | same |

Key consequences — and the headline simplification vs nkilib's `rmsnorm_tkg`:

- **The norm is per-head, so it is entirely core-local and rank-local.** Each value head's 128-wide
  statistic lives on one core. There is **no cross-core `sendrecv` and no cross-rank all-reduce** — unlike
  nkilib `rmsnorm_tkg`, which normalizes the full hidden and therefore needs a partition/`shard_on_h`
  reduction across cores (`rmsnorm_tkg.py` `sendrecv` path). Drop all of that machinery. The kernel
  mirrors the recurrence's value-head SPMD shard: launch `[2]`, each core handles its own `Hv_core=4`
  heads and writes a disjoint column slice of the full-shape output.
- `gamma` is replicated (no sharding); load once per core.
- Partition occupancy is inherently low (`T·Hv` ∈ {1·8, 2·8} per rank, or {4, 8} per core) — this is the
  nature of TKG (few tokens). The layout choice below is about getting the cheapest correct reduction at
  this small size, not about saturating 128 partitions.

---

## 3. Layout analysis — where `head_dim` goes

The recurrence emits each token's output as a `[1, W]` row, **head-major on the free axis**
(`nki_deltanet_tkg.py:422-431`: `O_row[0, h*d+j]`). The standalone input `attn_raw` is the full-shape
`[T, W]` head-major HBM tensor (`nki_deltanet_fused_tkg.py:97`). Two viable layouts:

### Layout A — `head_dim` on the FREE axis: `[P = T·Hv, F = d=128]`  ← recommended for v1 / standalone
The head-major `[T, W]` tensor is **contiguous**: linear index `t*W + h*d + j = (t*Hv + h)*d + j`. So
`[T, W]` reshapes to `[T·Hv, 128]` (partition = `t*Hv + h`, free = `j`) with **zero transpose** — a free
reshape on load. Then:
- sum-of-squares = free-axis reduce over 128 → `[T·Hv, 1]`
- `rsqrt(mean + eps)`, broadcast-multiply over free, gamma `[1,128]` broadcast over partition, silu-gate.
- Output `[T·Hv, 128]` reshapes back to `[T, W]` (contiguous) for the standalone.

This mirrors nkilib's `rmsnorm_tkg_th` (T-on-partition, H-on-free, free-axis reduce). It also drops in at
the recurrence's per-token output seam with **zero transpose**: each token's `O_row [1, W]`
(`nki_deltanet_tkg.py:422-426`) is already `[1, Hv, d]` head-major, so the per-head reduce is a free-axis
reduce over `d` on a 1-partition tile (T=1) — see §5. **Recommended for the fused integration as long as
`out_proj` is external** (see the boundary note below).

### Layout B — `head_dim` on the PARTITION axis: `[P = d=128, F = T·Hv]`  ← only for the FULL megakernel
Transpose the recurrence's output so the 128 head-dim lands on partitions, then:
- sum-of-squares = **partition-axis reduce via a ones-`nc_matmul`** (`stationary=ones[128,1]`,
  `moving=x²[128, T·Hv]` → `[1, T·Hv]` in PSUM) — the technique nkilib uses for its H0 reduce
  (`rmsnorm_tkg.py:267`) and that the recurrence already has (`nki_deltanet_tkg.py:90
  partition_broadcast_psum`, `red_p`, `ones_row`).
- `rsqrt`, broadcast back over 128 partitions, gamma `[128,1]` per-partition multiply, silu-gate.
- Output `[128, T·Hv]` has `head_dim` on partition → feeds `output_projection_tkg` with no further transpose.

### The deciding boundary: `out_proj` is external
`out_proj` is a **RowParallelLinear with a cross-rank all-reduce** — it cannot live inside this
value-head-sharded, collective-free kernel, so it stays a separate framework op. Therefore the fused
kernel **must materialize its output to HBM** at the kernel boundary regardless of layout, and `out_proj`
reads it back from HBM. Consequences:
- **While `out_proj` is external (now): use Layout A.** The kernel writes the gated-normed output to HBM
  head-major — byte-identical placement to today's `attn_out` — with **no transpose**. Layout B's
  `head_dim`-on-partition output buys nothing here because the HBM round-trip to `out_proj` erases it; the
  transpose is pure cost.
- **Layout B pays off only once `out_proj` is also in-kernel** (the true megakernel: recurrence → norm →
  matmul → all-reduce in one launch). At that point head_dim-on-partition feeds the matmul transpose-free.
  That is a separate, larger piece; do not pre-pay its transpose now.

**Bottom line:** integrate with Layout A at the `O_row` seam today; revisit Layout B when `out_proj` moves
in-kernel.

---

## 4. Kernel API (NKI 0.3.0) — integrate directly, one composable helper

Goal: fuse the norm+gate into the existing conv+recurrence kernel **from the start** (no separate kernel in
the model's forward). The math lives in **one composable SBUF helper** that is dropped into the recurrence's
per-token output seam; a thin standalone `@nki.jit` wrapper exposes the *same* helper over HBM for unit
testing. This is not "v1 then v2" — it is one function, validated cheaply, then composed.

```python
# Composable helper — the actual math. Plain function (NO @nki.jit), called inside the recurrence loop.
def norm_gate_row(o_row, z_row, gamma_sb, eps, d):
    """In-place gated per-head RMSNorm of ONE token's recurrence output.
    o_row   [1, W_core] f32   -- this token's O_row (head-major; W_core = Hv_core*d), in SBUF
    z_row   [1, W_core] bf16  -- this token's in_proj_z slice (same head-major layout), in SBUF
    gamma_sb[1, d]      bf16  -- per-head norm weight, shared across heads (loaded once)
    returns gated [1, W_core] bf16 -- viewed as [1, Hv_core, d], per-head RMSNorm * silu(z)
    Layout A: head_dim on the free axis; per-head reduce = free-axis reduce over d. No transpose."""

# Thin standalone harness — same helper, HBM in/out, for CPU-torch validation only.
@nki.jit
def deltanet_gated_rmsnorm(attn_raw, gamma, z, eps):
    """attn_raw (T, W) f32, gamma (d,), z (T, W) bf16 -> out (T, W) bf16.
    Loads each row to SBUF, calls norm_gate_row, stores. Launch [2] (value-head shard)."""
```

Plumbing into the fused kernel (the integration surface, all in `nki_kernels/`):
- Thread `z`, `gamma`, `eps` through `deltanet_fused_tkg_fwd` / `..._fwd_state` → `fused_compose` →
  `gated_delta_rule_tkg`. Today none of these carry `z`/`gamma` (`nki_deltanet_fused_tkg.py:57,83,108`).
- In `gated_delta_rule_tkg`, slice `z` by the same `col_off` the output write uses
  (`nki_deltanet_tkg.py:428-431`); `gamma` is replicated (load once per core). Call `norm_gate_row` on
  `O_row` right before the DMA at line 430, and write the **gated** row instead of the raw row.
- Model side (`modeling_qwen36_a3b.py`): `fused_attn_nki` passes `z`/`gamma`/`eps` into the kernel;
  `attn_output()` (lines 466-480) collapses to just `out_proj(...)`. Keep the torch norm/gate behind the
  existing flag for A/B validation, then delete once parity holds.

---

## 5. Implementation plan (integration-first)

### Step 1 — write `norm_gate_row` (Layout A) + the standalone `@nki.jit` harness
Per token (`o_row [1, W_core]` viewed as `[1, Hv_core, d]`):
1. `kernel_assert` (not Python `assert`): `W_core % d == 0`, `gamma.shape == (d,)`, `d == 128`.
2. Round to bf16 then keep an f32 view of `o_row` (parity: reference rounds `attn_raw`→bf16 before norm).
3. Square in f32: `nisa.activation(dst=sq, data=o_f32, op=nl.square)` → `[1, Hv_core, d]`.
4. Per-head reduce over the innermost `d`: `nisa.tensor_reduce(dst=sumsq[1,Hv_core,1], data=sq, op=nl.add,
   axis=(2,))`. (If 0.3.0 has `nisa.activation_reduce(op=nl.square, reduce_op=nl.add)`, fuse 3+4 — the
   `rmsnorm_tkg_th` optimization; confirm via /neuron-nki-docs.) **Mind the 3D `axis=` (0.3.0 fixes Beta-2
   axis bugs) — reduce the real `d` axis, not `axis=1`.**
5. `nisa.tensor_scalar(dst=stat, data=sumsq, op0=nl.multiply, operand0=1.0/d, op1=nl.add, operand1=eps)`
   then `nisa.rsqrt(dst=inv, data=stat)`. **NKI 0.3.0: `nisa.rsqrt`, not `nisa.activation(op=nl.rsqrt)`.**
   (If `nisa.rsqrt` takes `scale=`/`bias=`, fold the mul+add to match nkilib's single-instruction rsqrt.)
6. Normalize: broadcast `inv [1,Hv_core,1]` over `d`, multiply by the bf16-rounded value; cast bf16.
7. gamma: multiply by `gamma_sb [1,1,d]` broadcast over `Hv_core` (`nisa.tensor_tensor`, `nl.multiply`).
8. Gate: `nisa.activation(dst=sz, data=z_row, op=nl.silu)` (or `nl.sigmoid` then multiply if `nl.silu`
   absent), then `nisa.tensor_tensor(out, out, sz, op=nl.multiply)`. Return `[1, W_core]` bf16.
9. Standalone harness: loop `t`, load `attn_raw[t]`/`z[t]` rows, call the helper, store. Validate vs CPU
   torch (Section 6) for `T∈{1,2}`, single-core `[1]` then sharded `[2]`.

### Step 2 — compose into the recurrence (the integration)
1. Plumb `z`/`gamma`/`eps` through the entrypoints → `fused_compose` → `gated_delta_rule_tkg` (§4).
2. Load `gamma` once per core; per token, slice `z_row = z[t, col_off:col_off+W_core]` into SBUF.
3. At `nki_deltanet_tkg.py:424-433`, replace the raw `O_row` write with
   `gated = norm_gate_row(O_row, z_row, gamma_sb, eps, d)` then DMA `gated` to the same `attn_out` columns.
   **No transpose, no extra HBM round-trip** — the kernel now emits the gated-normed output directly.
4. Validate the fused kernel output against torch `attn_output(attn_raw, z)` (norm+gate only), then against
   the full layer.

### Step 3 — model wiring + cleanup
- `fused_attn_nki` passes `z`/`gamma`/`eps`; `attn_output()` becomes `return self.out_proj(gated)`.
- Keep the torch norm/gate behind `use_tkg_attention_kernel` for one A/B cycle; remove after parity.

### Deferred — Layout B (only with in-kernel `out_proj`)
Not now. When `out_proj` is brought in-kernel (RowParallel matmul + all-reduce), switch `norm_gate_row` to
the head_dim-on-partition form (§3 Layout B: one transpose from the recurrence, ones-`nc_matmul` partition
reduce reusing `partition_broadcast_psum`, output feeds the matmul transpose-free). Until then it is pure
added transpose cost — see §3's boundary note.

---

## 6. Validation plan

- **Reference on CPU, not XLA** (avoid extra NEFFs): replicate `attn_output` exactly — `Qwen3MoeRMSNorm`
  per-head over 128 (plain rsqrt) + `F.silu(z)` gate, with the f32→bf16 input round.
- **Stage-wise** (store intermediate to HBM, compare): (a) reshaped load == input; (b) per-head `var`;
  (c) normalized `y`; (d) `*gamma`; (e) `*silu(z)`. Only advance when each matches.
- **Shapes:** `(T=1, Hv=8/rank, Hv_core=4)` decode and `(T=2)` verify; bf16 I/O, f32 accumulate.
  Also a single-core `[1]` launch for isolation before the `[2]` shard.
- **Metrics:** atol/rtol + max-abs-diff + cosine-sim (don't rely on one). Expect tight bf16 agreement;
  the dominant error source is the bf16 input round, which the reference also incurs.
- Cross-check the `[2]` value-head shard writes disjoint, complete columns (no overlap/gap at `col_off`).

---

## 7. Optimizations preserved from nkilib (and what's intentionally dropped)

Preserved:
- **f32 intermediates** for square/reduce/rsqrt, bf16 I/O (`rmsnorm_tkg.py:225 inter_dtype=nl.float32`).
- **Fused square+reduce** via `activation_reduce` when available (installed `rmsnorm_tkg_th`), else
  `activation(square)`+`tensor_reduce` (bundled `process_rmsnorm_tile:237,247`).
- **Single-pass mean·scale + eps then rsqrt** — nkilib does it in one `rsqrt` activation with `scale`+`bias`
  (`rmsnorm_tkg.py:273-281`); under NKI 0.3.0 reproduce as `tensor_scalar(mul,add)` + `nisa.rsqrt`, or fold
  into `nisa.rsqrt(scale=,bias=)` if supported.
- **eps materialized once** (`rmsnorm_tkg.py:373-375`).
- **PSUM-direct partition reduce via ones-`nc_matmul`** for Layout B (`rmsnorm_tkg.py:267`) — also already
  in the recurrence, so reuse `partition_broadcast_psum`/`ones_row`/`red_p`.
- **Contiguous DMA** (head-major output is already contiguous; Layout A load is a free reshape, no strided
  access).

Intentionally dropped (don't fit per-head, per-core scope):
- **Cross-core `sendrecv`/`shard_on_h` reduction** — unnecessary: per-head stats are core-local.
- **Full-hidden `validate_shapes` contract** (`H%128==0` over the whole hidden) — we normalize per 128-head.
- **MX/quant fused paths** — `out_proj` here uses the float path; no quant coupling.

---

## 8. Open questions / risks to resolve at implementation time

1. **`nisa.rsqrt` scale/bias support** — does 0.3.0 `nisa.rsqrt` take `scale=`/`bias=` to fold the
   `mean*1/d + eps` step into one instruction? If not, the `tensor_scalar`+`rsqrt` pair is the parity-safe
   form. Check via /neuron-nki-docs.
2. **`nl.silu` availability** — if absent, use `nl.sigmoid` then `tensor_tensor` multiply (`u*sigmoid(u)`).
3. **`activation_reduce` in 0.3.0** — confirm the fused square+add reduce exists; otherwise two ops.
4. **Layout-B transpose cost vs Layout-A double-transpose** — measure on device; the recommendation
   (B for fused) assumes the single recurrence→norm transpose beats norm→out_proj transpose + the T>1
   free→partition collapse. Cheap to A/B test once v1 is correct.
5. **bf16-round-first parity** — confirm whether matching the reference's pre-norm bf16 round is required
   for the model's accuracy gate, or whether a pure-f32 norm passes (slightly more accurate, not
   bit-identical).
```
