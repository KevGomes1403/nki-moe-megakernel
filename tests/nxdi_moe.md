# How NxDI Qwen3-MoE Token-Gen Executes RMSNorm + MoE on Trainium3

Source profile: `output/baseline/1776976164165154117/138311351493953_vnc_0.ntff` against `output/baseline/1776976164165154117/neff_138311351493953_vnc_2.neff` (model `token_generation_model/_tp0_bk2/model.MODULE_069222236a5ea438f22b+e91e8f48.neff`, instance `trn3.3xlarge`, TP=4, batch-bucket = 2 tokens).

This is the layout the compiler chose for the **baseline** (non-NKI) NxDI lowering of Qwen3-MoE on NeuronCore-v3. All numbers below were taken directly from `neuron-profile view --output-format json` and `neuron-profile show-session`, and constants (eps, scales, memsets) were decoded from the operand strings of individual instructions.

---

## 1. Reference graph (per decoder layer)

```
hidden_states (bf16, replicated across TP)
  └─ post_attention_layernorm   (CustomRMSNorm, eps=1e-6)            <-- AwsNeuronRmsNorm
       │
       ▼
  router (TP-replicated, dense Linear, weight stored in fp32)
       │
       ▼  router_logits  (fp32, NOT cast back to bf16 here)
       │
       ├──► softmax over E=128                                       <-- AwsNeuronSoftmax
       │       │
       │       ▼  expert_affinities (cast to bf16 at end)
       │
       └──► topk(router_logits, k=8)                                 <-- AwsNeuronTopK
                │
                ▼   expert_index (int32/long), top-k logit values unused
       gather expert_affinities[t, expert_index[t]]                  -> chosen_expert_affinities (bf16)
       F.normalize(chosen_expert_affinities, p=1, dim=1, eps=1e-12)  -> normalized weights (bf16)

  per token t (T loop, T = 2 in this NEFF):
       gate_up = ExpertFusedColumnParallelLinear[selective load]( hidden_states[t], expert_index[t] )
                  --> (top_k=8, 1, 2*I/TP)   [I=768, TP=4 → 2*I/TP = 384]
       gate, up = chunk(gate_up, 2, dim=-1)                           # (8,1,192) each
       activated = SiLU(gate) * up                                    # bf16
       down = ExpertFusedRowParallelLinear[selective load]( activated, expert_index[t] )  
                  --> partial (top_k=8, 1, H)  [H=2048]
       weighted = sum_over_topk( down * chosen_expert_affinities[t].unsqueeze(1) )  -> (H,)
  output = stack(weighted, dim=0)                                     -> (T, H)

  cross_replica_sum AllReduce across TP=4   <-- final all-reduce only here
  residual add
```

The MoE.forward chosen path is `forward_selective_loading` (because this is token-gen, T=2 is small). That is the per-token Python loop you can read in `expert_mlps_v2.py:595`.

Top-k indices are computed from **raw fp32 router_logits**, not from the softmax output (`routing.py:204`). `apply_activation_fn` is computed in fp64 in the eager source, but XLA→Neuron lowers it to a single `AwsNeuronSoftmax` custom-call that operates on bf16 with fp32 accumulators (see §3.2).

Logical per-TP-rank shapes are H=2048, I=768, E=128, top_k=8, TP=4, T=2 (`bk2`). There is still no expert-parallelism in this NEFF, but the compiled physical execution is **not** single-core per TP rank: it uses LNC=2 virtual-core sharding inside each logical TP rank.

### 1.1 LNC=2 physical placement and sharding

The NEFF header reports:

| Field | Value |
|---|---:|
| `number_of_neuroncores_per_lnc` | `2` |
| `number_of_logical_neuroncores` | `1` |
| `number_of_cc_participants` | `4` |
| enabled features | includes `virtual-core`, `dge`, `hw-dge` |

The token-gen NTFF has two physical subgraphs for this one logical core:

| subgraph | physical core | LNC | trace count | tensor instr | gpsimd instr |
|---|---:|---:|---:|---:|---:|
| `sg00` | `ND0 NC4` | `0` | `243330` | `67147` | `42011` |
| `sg01` | `ND0 NC5` | `0` | `237376` | `66652` | `41833` |

This means the compiler is using two physical NeuronCores as one virtual/logical core. The MoE work is **sharded**, not replicated in full:

- MoE-attributed instruction counts are balanced across the pair: `sg00=103134`, `sg01=101886`.
- For a representative layer (`NeuronQwen3MoeDecoderLayer[.49]`), router and expert matmuls appear on both subgraphs with the same HLO names but different SBUF/PSUM addresses.
- The large matmuls are split across output/channel tiles. Examples for layer 49:

| HLO | Meaning | `sg00` | `sg01` | Logical total |
|---|---|---:|---:|---:|
| `%dot.9` | router linear | 16 MATMUL | 16 MATMUL | 32 MATMUL |
| `%dot.10` | gate/up projection | 256 MATMUL | 256 MATMUL | 512 MATMUL |
| `%dot.11` | down projection | 128 MATMUL | 128 MATMUL | 256 MATMUL |

The `channels=96` SiLU/multiply instructions on both `sg00` and `sg01` are the clearest sign of physical sharding: the logical per-TP intermediate is `I/TP = 192`, and each physical core handles one 96-wide shard. This is why the same logical HLO can appear on both subgraphs without meaning the full MoE is duplicated on both cores.

The split is asymmetric for scalar/vector post-processing:

- `sg00` carries the full softmax custom-call for the router affinities.
- top-k appears on both subgraphs, so both physical shards have the selected expert indices needed for local expert weight gathers.
- the visible top-k weighted-sum reduction and the MoE TP all-reduce trigger are mostly attributed to `sg00`.

I did **not** find explicit physical-core `sendrecv` operations between `sg00` and `sg01`. The attributed instruction table has no `SEND`, `RECV`, `SENDRECV`, `send_recv`, `receive`, `DGE`-named instruction, `IMCPY`, or `PSEUDO_DMA_*` opcodes/operands, and the raw CC trace exposes `TPB_TRIGGER`, `DMA_ADVANCE`, semaphores, and `ALGO_MESH_*` events rather than send/recv-named events. The cross-core exchange needed by virtual-core execution is therefore compiler-managed and not represented as an explicit sendrecv collective in this profile.

The explicit collectives are TP collectives: the NEFF has `number_of_cc_participants=4`, and the MoE collectives are `xla__cross_replica_sum` under `MoE/.../reduce_from_tensor_model_parallel_region`. There are 48 unique MoE all-reduce HLOs and 48 attention all-reduce HLOs in this token-gen graph.

---

## 2. RMSNorm (`CustomRMSNorm`) — instruction-by-instruction

Framework wrapper (`neuronx_distributed_inference/modules/custom_calls.py:8`):

```python
def forward(self, hidden_states):                # hidden_states: bf16
    original_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)        # _to_copy upcast
    result = RmsNorm.apply(hidden_states, self.weight, eps=1e-6, dim=-1)
    return result.to(original_dtype)             # cast back to bf16
```

The `_to_copy` upcast is fused away by the compiler. In SBUF, **x is held in bf16** and engines read it as bf16; the upcast happens implicitly inside the engine MAC.

Decoded constants in the operand strings (verified by hex):

| memset/imm value | hex / float | meaning |
|---|---|---|
| `1065353216`        | `0x3F800000` = `1.0f`               | all-ones weight vector for the partition-reduce matmul |
| `897988541`         | `0x358637BD` ≈ `9.9999999747e-7`    | `eps = 1e-6` (config `rms_norm_eps`) |
| `scale = 0.000488`  | `1/2048` exactly (bf16 == fp32 here) | `1/H`, where `H = 2048` is the **full** hidden size |

The exact instruction sequence emitted **per CustomRMSNorm op** (single tile, channels=128, partition lanes = 128):

| # | engine | opcode | dtype path | role |
|---|---|---|---|---|
| 1 | Vector | `MEMSET`            | fp32 ← 1.0     | initialise the all-ones partition-reduce weight (128 lanes) |
| 2 | Vector | `MEMSET`            | fp32 ← 1e-6    | stage `eps` for the rsqrt bias |
| 3 | Scalar | `TENSOR_TENSOR MULTIPLY` | fp32 = fp32(γ) * bf16(x) | precompute `γ·x` (kept in fp32 in SBUF), used as multiplicand in step 8. **γ is held in fp32 SBUF**, the cast `bf16→fp32` happens once at load. |
| 4 | Scalar | `ACTIVATE SQUARE`   | fp32 = SQUARE(bf16(x)) | per-partition squares of x. `bias_ptr=fp16@…`, `scale=1.0`, `imm=0` ⇒ `out = (1.0·x + 0)^2 + 0`. |
| 5 | Vector | `TENSOR_REDUCE ADD dim=X` | fp32 ← Σ fp32 squares (along free axis) | within-partition sum of x² (16 elements per lane: 2048 / 128 = 16). |
| 6 | Tensor | `LDWEIGHTS` + `MATMUL` (`fp32_mode=LOW_HIGH`, `acc_flags=1`, `psum_zero=2048`) | fp32 ← Σ across 128 lanes | cross-partition sum **using the matmul-with-ones trick**, fp32 emulated. The 1×128 partial sums are matmul'd by a 128-vector of `1.0f` to produce a single scalar broadcast back to all 128 lanes via PSUM. |
| 7 | Tensor | `LDWEIGHTS` + `MATMUL` (`fp32_mode=HIGH`, `acc_flags=2`) | fp32 += high-half product | second matmul of the two-pass FP32 emulation: lane sums are split into bf16 LOW/HIGH halves; each half is multiplied by the all-ones bf16 weight and accumulated in PSUM. **Order matters**: LOW first (`acc_flags=1`, zero+accum), HIGH second (`acc_flags=2`, accum). |
| 8 | Scalar | `ACTIVATE RECIPROCAL_SQRT` (`scale=1/2048`, `bias_ptr=fp32(1e-6)`) | fp32 ← `rsqrt(scale·sum + bias)` | computes `1/sqrt(mean(x²)+eps)`. **Fused** order: `rsqrt(s·x + b)`, where `s = 1/H = 1/2048` and `b = 1e-6`. Result kept in PSUM. |
| 9 | Scalar | `ACTIVATE COPY` (`scale=[psum_addr]`, `imm=0`) | bf16 ← cast(fp32(γ·x) · rsqrt) | Multiplies the precomputed `γ·x` (step 3, fp32) by the per-token `rsqrt` (per‑partition scale loaded from PSUM `0x2000800`) and casts the result down to bf16 in one engine pass. |
| 10 | Scalar | `COPY` int32 | bf16 lane shuffle | output reformat / strided write to consumer SBUF tile. |

### What you must replicate, bit-exactly, for an RMSNorm kernel

1. **Square the bf16 input directly.** `square = x^2` is computed by `ACTIVATE SQUARE` consuming bf16 and producing fp32. `(bf16(x))^2 → fp32` is exact (no rounding because `bf16→fp32` is bit-extension and fp32 holds the product exactly for x²).
2. **Within-partition reduction:** plain fp32 `TENSOR_REDUCE ADD` along the free axis. Order is left-to-right within the engine; for K=16 elements per lane, this is a single one-shot reduce, so no associativity ambiguity at this stage.
3. **Cross-partition reduction:** **fp32 emulated matmul** with an all-ones vector, `LOW` then `HIGH`. Do **not** use a Vector-engine partition reduction (it would lose precision). The compiler is explicit about this — both `MATMUL` ops have `fp32_mode=LOW_HIGH` and write to the same PSUM offset.
4. **Mean and rsqrt are fused:** `rsqrt(scale·sum + eps)` with `scale = 1.0f/H` (exact bf16, exact fp32) and `eps = 1.0e-6` (`0x358637BD`). Do not compute `mean = sum * (1/H)` in a separate step then add eps — the engine fuses it via the `bias_ptr`/`scale` operands of `ACTIVATE RECIPROCAL_SQRT`.
5. **Final scale-and-cast is one op:** `out = bf16( (γ·x_fp32) * rsqrt_fp32 )` is one `ACTIVATE COPY` whose runtime `scale` operand is loaded from PSUM (`scale=[0x2000800]`). γ is loaded into SBUF as fp32 once.
6. **γ load:** even though HF stores `Qwen3MoeRMSNorm.weight` as bf16, the compiler upcasts to fp32 at load time; the multiply in step 3 takes `src0=fp32` from SBUF address `0x3ff00`. If your kernel reads γ as bf16 and casts inside, you'll match — the `bf16→fp32` cast is bit-extension.

### Things that are **not** done
- No `mean(x²)` then `+ eps` then `rsqrt` in three separate ops. The compiler fuses to `rsqrt(scale·x + b)` (1 instruction).
- No use of `RECIPROCAL` followed by `SQRT`. Always `RECIPROCAL_SQRT`.
- No promotion of γ during `MULTIPLY` — γ is **already** fp32 in SBUF.

---

## 3. MoE — exact precision and op chain

### 3.1 Router linear (`get_router_logits`)

Module: `RouterTopK` over a `Linear` of shape `(num_experts=128, hidden_size=2048)`. **`router_config.dtype = torch.float32` is forced** in `Qwen3MoeInferenceConfig` (line 263 of `modeling_qwen3_moe.py`), so the weight is stored fp32 on the host. In this token-gen NEFF, the router matmul produces the full 128-logit row on each logical TP rank; the profiler shows 128-wide tensor tiles and no pre-softmax router AllReduce.

What happens on hardware (per decoder layer, `bk2` token bucket):

- LDWEIGHTS: `bfloat16@…[partition,free,K] 128*128`  — the fp32 weight is fed to the tensor engine in **bf16** halves (no `fp32_mode=LOW_HIGH` is set; native bf16 matmul to fp32 PSUM).
- MATMUL: `src=bf16(hidden), dst=fp32(PSUM), psum_zero=2048, acc_flags=1` for the first K-tile, then `acc_flags=0` for subsequent K-tiles. The logical layer has **32 LDWEIGHTS+MATMUL pairs** for the `bk2` bucket, split as 16 on `sg00` and 16 on `sg01`.
- The PSUM result (fp32) is consumed directly by the softmax custom-call without rounding to bf16 first — the HLO chain after this matmul is `convert.X` (fp32 → bf16) then the softmax custom-call. That `convert` is fused into the next op's source dtype.

Bit-exact recipe for a fused router NKI kernel:
- Multiply `bf16(hidden_states) × bf16(router_weight_T)` with **fp32 PSUM accumulation**.
- K dimension reduces along partition + free; the compiler chose a single 128-partition × 128-free tile and stationary-weight semantics (`tag_weight_mode=WEIGHT_ONLY`).
- Do **not** apply fp32 LOW_HIGH emulation on the router matmul (the compiler did not). The router weight is bf16 from the engine’s view.
- After the matmul, the next consumer (softmax) reads the fp32 PSUM directly, but the topk consumer reads the same value (cast to bf16). If your kernel persists `router_logits` to SBUF for both consumers, **persist as bf16 once** — that is what the compiler does (see `convert.380` and `reshape.341` in the HLO names).

### 3.2 Softmax (`apply_activation_fn`, `AwsNeuronSoftmax`)

Source code says `F.softmax(weights, dim=1, dtype=torch.float64)`. Lowered to a single custom-call `xla___op_SoftmaxForwardImpl_custom-call`. Sequence on hardware (one decoder layer):

| # | engine | opcode | dtype | role |
|---|---|---|---|---|
| 1 | (DMA) | `ACT_TABLE_LOAD table_sel=2` | — | load EXP lookup table |
| 2 | DMA | `DMA_DIRECT2D` | bf16 | bring router_logits tile into SBUF (twice — one per `nd` half) |
| 3 | Vector | `TENSOR_REDUCE op=MAX dim=X` | bf16 → fp32 | per-row max |
| 4 | Scalar | `ACTIVATE COPY scale=-1.0` | fp32 → fp32 | negate max ⇒ `-m` |
| 5 | Scalar | `ACTIVATE EXP bias_ptr=fp32(-m), scale=1.0` | bf16 → fp32 | `exp(scale·x + bias) = exp(x − m)` per element |
| 6 | Vector | `TENSOR_REDUCE op=ADD dim=X` | fp32 → fp32 | denominator `Z = Σ exp` |
| 7 | Vector | `RECIPROCAL` | fp32 → fp32 | `1/Z` |
| 8 | Scalar | `ACTIVATE COPY scale=[1/Z]` (next ACTIVATE) | fp32 → bf16 | multiply each `exp(...)` by `1/Z` and cast to bf16 |

So although the eager Python is fp64, **on Neuron softmax is bf16 input → fp32 internal → bf16 output**. The intermediate `exp` is fp32 (kept in SBUF at `0x1804948`, 128 fp32 elements per partition). Subtracting the max is done by computing `−m` (a single `ACTIVATE COPY scale=-1`) and then folding `exp(x + (-m))` via the `bias_ptr` operand of `ACTIVATE EXP`.

Bit-exact recipe:
- One row max (bf16 → fp32), negate, fuse into EXP via `bias_ptr` of the activate.
- Single fp32 reduction over the **full E=128** router row. There is no pre-softmax router AllReduce in this trace; `convert.380` is the bf16 cast of the full router logits before the softmax custom-call.
- One `RECIPROCAL` into a per-row scalar; final ACTIVATE multiplies and casts to bf16.

### 3.3 TopK (`AwsNeuronTopK`, k=8)

Per the profile, this uses the dedicated topk hardware ops:

```
MAX8           src=bf16,    dst=bf16   (8 winners by value, per row, 1 round)
MATCH_VALUE_LOAD                       (set up for index search)
FIND_INDEX8    src=bf16,    dst=uint32 (8 indices matching the 8 max values)
```

The compiler emits two pairs `(MAX8, MATCH_VALUE_LOAD, FIND_INDEX8)` per layer: one for each token in the bk2 batch. Inputs are **bf16 router_logits** (note: top-k is over the bf16 cast of the router logits, **not** over softmax outputs — confirms `routing.py:204`).

For a bit‑exact kernel, you must call `MAX8` / `FIND_INDEX8` on the **bf16-rounded router logits**, not on fp32. If you keep router_logits in fp32 in your kernel for use by softmax, cast a bf16 copy before calling top-k — otherwise tie-break behaviour and rounding will diverge. (Qwen3’s router is fp32 in HF, but the casted bf16 is what’s used for top-k selection on hardware.)

### 3.4 Selective expert load + gather

For each of the 8 chosen experts per token, NxDI's `forward_selective_loading` does an HBM gather of just that expert's gate/up and down weight tiles, using the topk indices. In the trace this shows up as a chain of GpSimd `GATHER`, `STREAM_SHUFFLE`, `POOL_BUFFER_LOAD`, `LOAD_MASK_SELECT`, `DMA_DIRECT2D`. The expert indices are first translated through `aten.index.Tensor` (gather) → `aten__add_add` → `aten__where_select` (predicated copy) to map global `expert_index` to the local TP shard's slot.

The expert weights live in HBM laid out as `(E_local, hidden, 2*I_local)` for gate_up_proj and `(E_local, I_local, hidden)` for down_proj, both bf16. There is no quantization of expert weights here.

### 3.5 Fused gate_up matmul (`ExpertFusedColumnParallelLinear`, _activation)

HLO: `%dot.X = dot(%reshape.X, %gather.X)`. Shapes (per token, per chosen expert):

```
input:  (1, H=2048)            bf16
weight: (H=2048, 2*I/TP = 384) bf16   <-- gathered for that expert
output: (1, 384)               fp32 PSUM
```

512 LDWEIGHTS+MATMUL pairs per decoder layer per logical TP rank, split across the LNC=2 physical-core pair as 256 on `sg00` and 256 on `sg01`. With `T=2`, top_k=8, that is 2·8 = 16 expert matmuls per layer across the logical rank. Tile: **128 partition × 96 free**, and each physical core handles a 96-wide shard of the logical `I/TP = 192` gate or up half. `acc_flags=1` zeros PSUM at the start of K and `acc_flags=0` continues; native bf16 matmul to fp32 PSUM (no fp32 emulation).

Then split: `gate, up = chunk(...)`. The split is **purely by SBUF address arithmetic** — no real data movement; the activation custom-call just reads the gate half from `slice.X` of the matmul output.

### 3.6 SiLU + gating (`xla___op_SiluForwardImpl_custom-call`, then mul)

```
ACT_TABLE_LOAD                    (SiLU LUT)
ACTIVATE      SiLU(gate_fp32) -> bf16 SBUF
TENSOR_TENSOR MULTIPLY            bf16 × bf16 -> bf16   (silu(gate) * up)
```

For Qwen3-MoE, `_glu = True`, `_glu_type = GLU` (default), `hidden_act_scaling_factor = 1.0`, `hidden_act_bias = 0`, no `gate_clamp`, no `up_clamp` — so `_activation` reduces to **`silu(gate) * up`** (no SwiGLU re-multiplication, no clamps). This matches `experts.py:227`:
```python
return self._activation_fn(self.hidden_act_scaling_factor * gate) * (up + self.hidden_act_bias)
```
with the scaling factor and bias being identity values.

Bit-exact recipe:
1. Read gate from PSUM (fp32) into the activate engine, apply `ACTIVATE SiLU` with the standard SiLU LUT (`activation_fn = silu`). Output **bf16**.
2. Do not re-cast gate to bf16 then back — the compiler does fp32-PSUM → bf16 in one ACTIVATE.
3. Multiply the bf16 silu(gate) by the bf16 up (which itself came from the same matmul output PSUM, cast to bf16 the same way). The MULTIPLY is `TENSOR_TENSOR op=MULTIPLY` with bf16 inputs and bf16 output.

### 3.7 Down projection (`ExpertFusedRowParallelLinear`)

HLO: `%dot.X = dot(%reshape.X, %gather.X)` for the down weight. Shapes (per token, per chosen expert):

```
input:  (1, I/TP = 192)    bf16   (output of silu(gate)*up)
weight: (I/TP=192, H=2048) bf16   gathered for that expert
output: (1, H=2048)        fp32 PSUM   <-- partial, before TP all-reduce
```

256 LDWEIGHTS+MATMUL pairs per decoder layer per logical TP rank, split as 128 on `sg00` and 128 on `sg01`. Tile **96 partition × 128 free**, `acc_flags=3 psum_zero=2048` — i.e. each per-output-tile matmul zeroes PSUM and accumulates. Native bf16 matmul to fp32 PSUM. The PSUM bank addresses (`0x2003800`, `0x2003804`, `0x2003808`, … on `sg00`; different corresponding PSUM addresses on `sg01`) increment by 4 bytes per matmul: each matmul writes to a different free-dim tile of the output, accumulating K through the free dim of the input.

This is partial output; the TP all-reduce comes last (§3.9).

### 3.8 Routing-weight normalization and per-token expert weighted sum

Computed as:
```
chosen_expert_affinities = expert_affinities[t, expert_index[t]]    # (top_k=8,) bf16
denom = clamp(min=1e-12, sum_bf16(chosen_expert_affinities)*)       # fp32 internally
weight = chosen_expert_affinities / denom                           # bf16 (F.normalize p=1)
output_t = Σ_k  down_partial[k] * weight[k]                         # (H,)
```

In the trace, the order is:

| Op | Engine | What it is |
|---|---|---|
| `aten.index.Tensor / gather`     | GpSimd `GATHER`   | gather affinities at top-k positions |
| `aten.sum.dim_IntList / reduce.X` | Scalar `TENSOR_REDUCE op=ADD dim=X` | sum across top-k (8 elements) into fp32 |
| `aten.clamp_min`                  | Vector `TENSOR_SCALAR ops=MAX,MIN` | `clamp_min(sum, 1e-12)`; the `imm` is the **bf16-rounded** value of `1e-12` ≈ `1.0018653e-12` (`F.normalize`'s default eps) |
| `aten.div.Tensor`                 | Vector `RECIPROCAL` then `ACTIVATE COPY scale=[recip]` | compute `1/denom` once (fp32) then scale the gathered weights to bf16 |
| `forward_selective_loading[…]/aten.sum.dim_IntList` (the **second** sum) | Vector `TENSOR_REDUCE op=ADD dim=X` (top-k=8 → 1) **+** Tensor `MATMUL transpose_mode=ENABLED` (cross-partition all-ones reduce) **+** `COPY` (fp32→bf16 cast) | weighted sum of the 8 down_proj partials, finally cast to bf16 |

Two important subtleties:
- `clamp_min` uses `imm = 1.0018653e-12` which is exactly `bf16(1e-12)` cast back to fp32. **Use the bf16-rounded constant**, not a literal `1e-12`. (Default `eps` of `torch.nn.functional.normalize` is `1e-12`; the compiler rounds the constant through bf16 because `clamp` fuses with bf16 inputs.)
- The "sum" of a `(top_k=8, H/128_partitions)` tile is implemented as **(a)** within-partition reduce over `top_k` then **(b)** matmul-with-ones for the partition-axis reduce — same trick used in RMSNorm. This is necessary because the H=2048 output is laid across the 128 partition lanes (16 elements per lane), so a clean reduction over the lane axis is implemented as a transposed matmul against a vector of ones (`transpose_mode=ENABLED`). Importantly, the cast to bf16 happens **after** the matmul reduce — the topk weighted sum is accumulated in fp32 PSUM, then cast.

### 3.9 All-reduce (`reduce_from_tensor_model_parallel_region`)

Single TP AllReduce per decoder layer for the stacked `bk2` MoE output, after the per-token weighted sums are written to bf16 SBUF. HLO: `%all-reduce.X = all-reduce(%reshape.X, %get-tuple-element.X)`. CC core executes this; the compiler observed `disable_numeric_cc_token=True` (set by `Qwen3MoeInferenceConfig` line 269) so the **collective receive does not include the spurious extra add/multiply** that the default openxla CC token path would inject. This matters for bit-exactness: with `disable_numeric_cc_token=True`, AllReduce is a plain elementwise sum — the receiving side's residual chain sees the unaltered AllReduce result.

This is a TP=4 collective, not a visible LNC-local two-way reduction. In the profile, the all-reduce trigger is mostly attributed to `sg00`, while both `sg00` and `sg01` contribute physical-core shards of the MoE matmul work before that point.

---

## 4. End-to-end accumulation order (what to honour bit-for-bit)

1. **Pre-norm residual stream is bf16.** Everything entering the MoE block, including the residual that flows around it, is bf16.
2. **RMSNorm:** all squaring, summing, mean, eps, rsqrt are **fp32**, with cross-partition reduction via the **all-ones LOW_HIGH matmul trick**. Output is **bf16**.
3. **Router matmul:** bf16 hidden × bf16 router_weight → **fp32 PSUM**; cast to bf16 once for use by softmax & topk.
4. **Softmax:** subtract-max in fp32, EXP in fp32 (bf16 input), Σ in fp32, ÷Σ via fp32 reciprocal, final cast to **bf16**.
5. **TopK** is over the bf16 router_logits, not over softmax probabilities.
6. **Affinity normalization (`F.normalize`):** Σ in fp32, `clamp_min(_, bf16(1e-12))`, `1/(...)` in fp32, multiply each top‑k affinity by the reciprocal, cast result to bf16.
7. **Expert MLP:** bf16 inputs, bf16 weights, **fp32 PSUM accumulation** in both gate_up and down. Activations between matmuls are **bf16**: silu(gate_fp32→bf16) × up_bf16 → bf16.
8. **Top-k weighted reduction:** `down_partial_fp32` (still in PSUM) × `weight_bf16`, summed across top‑k in fp32, cross-partition reduced via the matmul-with-ones, cast to bf16.
9. **AllReduce** in bf16 across TP=4, plain sum (`disable_numeric_cc_token=True`). 
10. Residual add (bf16 + bf16 → bf16) in the post-MoE residual.

Anything done in a different order (e.g. casting router_logits to bf16 before softmax instead of letting the softmax ingest bf16 from a `convert`, or casting the topk-weighted sum to bf16 **before** the cross-partition reduce instead of after) **will produce a different bf16 result** even though it should be mathematically equivalent.

---

## 5. Key constants (decoded from `MEMSET`/`imm` operands)

| Name | Where | Bit pattern | Value | Notes |
|---|---|---|---|---|
| `rms_norm_eps` | RMSNorm | `0x358637BD` | `9.999999974…e-7` (fp32) | from `config.rms_norm_eps = 1e-6` |
| `rms_inv_H` | RMSNorm activate scale | `1/2048` exactly | bf16 == fp32 here | hidden = 2048 |
| `ones_for_reduce` | RMSNorm + per-token weighted sum | `0x3F800000` | `1.0f` | weight vector for partition-reduce matmul |
| `normalize_eps` | F.normalize p=1 | bf16-rounded `1e-12` ≈ `1.001865e-12` | clamp_min imm | default eps of `F.normalize` |
| `softmax max-negate scale` | Softmax | `-1.0f` | activate scale | turns `m` into `-m` for fused EXP bias |
| `disable_numeric_cc_token` | AllReduce | bool=True | — | `Qwen3MoeInferenceConfig` line 269; AllReduce is a plain sum |

---

## 6. Practical guidance for a bit-exact NKI kernel

1. **Match the engine that produces each value.** SQUARE, EXP, RECIPROCAL_SQRT, RECIPROCAL must come from the Activate/Vector engine ops — not from emulated `x*x`, `2^(x*ln2_e)`, etc. Their outputs round in subtly different ways.
2. **Don’t flatten cross-partition reductions to Vector-engine adds.** The compiler intentionally uses Tensor-engine matmul-with-ones (with FP32 LOW_HIGH for RMSNorm; native bf16 for the topk weighted sum). A bf16 partition reduce on the Vector engine **will not match** a fp32-emulated partition reduce.
3. **Pin γ as fp32 in SBUF.** Even though HF stores it bf16. The MULTIPLY in step 3 of RMSNorm reads `src0=fp32`. If your kernel feeds bf16 γ, the result is still bit-identical because `bf16→fp32` is bit-extension, but you must cast γ once before the multiply, not let bf16 γ flow through a `TENSOR_TENSOR` MULTIPLY with bf16 src0.
4. **Use the bf16-rounded `1e-12` constant** in `F.normalize`, not the fp32 literal. (`bits = (struct.unpack('I', struct.pack('f', 1e-12))[0] + 0x8000) & 0xFFFF0000`.)
5. **Cast router logits to bf16 once, share the bf16 copy with both softmax and topk.** Calling top-k on fp32 router_logits will diverge on tie-prone tokens.
6. **Keep PSUM live across the topk-weighted sum.** Down-proj partials must remain in fp32 PSUM (or be reloaded as fp32 from a fp32 SBUF copy) when multiplied by the bf16 affinity weight; cast to bf16 only after the cross-partition matmul reduce. Casting to bf16 between down-proj and weighted sum changes the result.
7. **Skip the CC token numeric tweak.** The compiler set `disable_numeric_cc_token=True`. AllReduce is a plain elementwise sum; do not insert any extra `+0` or `*1` on the receive side as some openxla paths do.
8. **Activation flags are part of the answer.** `acc_flags=1` (zero+accum), `acc_flags=2` (accum HIGH after LOW), `acc_flags=3` (zero+accum, single phase), `psum_zero=2048` — these encode whether PSUM is reset and whether you are in the LOW or HIGH half of a fp32-emulated matmul. If your NKI kernel uses `nl.matmul(... , psum_acc=...)` with the wrong combination, a single decoder layer can drift by 1‑2 bf16 ulps and compound.

---

## 7. One-shot per-layer summary (what to fuse together for a kernel)

If you want a single fused kernel for the post-attention residual MoE block, the natural fuse boundaries the compiler implies are:

- **Block A (RMSNorm):** SBUF in: residual_bf16 + γ_fp32. SBUF out: norm_bf16. Internals: SQUARE, within-partition reduce, fp32 matmul-with-ones cross-partition reduce, fused `rsqrt(scale·x+eps)`, fused multiply-cast.
- **Block B (Router + Softmax + TopK):** SBUF in: norm_bf16 + router_w_bf16. PSUM out (fp32) → bf16 SBUF for both consumers. Softmax uses the bf16 copy; TopK uses the same bf16 copy.
- **Block C (per-token expert loop, T=2):** for each `t`: HBM gather of expert tiles (gate_up + down) for the 8 chosen experts; gate_up matmul (bf16×bf16→fp32 PSUM); SiLU(gate)·up (bf16); down matmul (bf16×bf16→fp32 PSUM, accumulated across the input intermediate axis). Multiply by bf16 affinity weight, fp32 reduce across top‑k (within-partition vector reduce), fp32 matmul-with-ones cross‑partition reduce, cast to bf16.
- **Block D (AllReduce + residual add):** plain bf16 AllReduce → bf16 add.

That mirrors the existing `kernels/moe_fused_tkg/moe_fused_nki.py` factoring; the constants and op orderings above are the ones to honour to obtain bit-exact agreement with the baseline NEFF.
