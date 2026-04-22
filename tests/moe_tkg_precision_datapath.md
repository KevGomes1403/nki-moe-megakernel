# MoE Token-Generation Precision Datapath

**Scope:** `qwen.py` running token generation (S=1) with `moe_fused_nki_kernel_enabled=False`
(`init_tkg_module=False`) and compiler flag `--enable-mixed-precision-accumulation`.

This is the reference execution path that a replacement NKI kernel must match.

---

## Code path in `qwen.py`

`NeuronQwen3MoeDecoderLayer.forward()` with `moe_fused_nki_kernel_enabled=False`:

```python
# lines 427-430 of qwen.py
if not self.moe_fused_nki_kernel_enabled:
    hidden_states = self.post_attention_layernorm(hidden_states)   # explicit RMSNorm
hidden_states = self.mlp(hidden_states, padding_mask)[0]
```

`self.mlp` is `MoEV2` initialized without `init_tkg_module`.
`self.moe_fused_tkg` is `None`, so `MoEV2.forward()` goes to `_forward_compute_bound()`.
For S=1 (token generation), `ExpertMLPsV2.forward()` dispatches to
`forward_selective_loading()` or `forward_all_experts()` — both are pure PyTorch matmuls
compiled to Neuron with `--enable-mixed-precision-accumulation`.

---

## Step 1 — RMSNorm (`CustomRMSNorm`)

Source: `neuronx_distributed_inference/modules/custom_calls.py`

```python
def forward(self, hidden_states):
    original_dtype = hidden_states.dtype          # bfloat16
    hidden_states = hidden_states.to(torch.float32)
    result = RmsNorm.apply(hidden_states, self.weight, self.variance_epsilon, ...)
    return result.to(original_dtype)              # cast back to bfloat16
```

| | dtype |
|---|---|
| Input | **bf16** |
| Cast inside forward | **float32** |
| `RmsNorm.apply` (variance, reciprocal sqrt, γ multiply) | **float32** |
| Output | **bf16** (cast from float32) |

γ weight is bf16; it is promoted to float32 inside the Neuron custom op.

---

## Step 2 — Router linear projection

Source: `neuronx_distributed/modules/moe/routing.py`, `RouterTopK.get_router_logits()`

```python
hidden_states = hidden_states.to(dtype=self.linear_router.weight.dtype)  # cast to float32
router_logits = self.linear_router(hidden_states)                         # float32 matmul
hidden_states = hidden_states.to(dtype=original_hidden_dtype)            # cast back to bf16
```

Router weights are initialized with `dtype=router_config.dtype = torch.float32`
(set in `Qwen3MoeInferenceConfig.__init__`).

| | dtype |
|---|---|
| `hidden_states` input | bf16 |
| Cast before matmul | **float32** |
| Router weight (`linear_router.weight`) | **float32** |
| Matmul accumulation | float32 |
| `router_logits` output | **float32** |

`--enable-mixed-precision-accumulation` has no additional effect here because both
operands are already float32.

---

## Step 3 — Router activation (softmax) and top-K

Source: `routing.py`, `RouterTopK.apply_activation_fn()` and `RouterTopK.forward()`

```python
# Qwen3 uses act_fn="softmax"
expert_affinities = F.softmax(weights, dim=1, dtype=torch.float64)  # float32 → float64
expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)  # float64 → bf16
```

| | dtype |
|---|---|
| `router_logits` input | float32 |
| Softmax compute | **float64** (explicit `dtype=torch.float64` in `F.softmax`) |
| Cast after softmax | **bf16** |
| `expert_index` (top-K indices) | int64 |

---

## Step 4 — Expert affinity normalization (`norm_topk_prob=True`)

Source: `expert_mlps_v2.py`, `ExpertMLPsV2.get_expert_affinities_masked()`

```python
if normalize_top_k_affinities:
    expert_affinities_masked = F.normalize(expert_affinities_masked, p=1.0, dim=1)
```

At this point `expert_affinities_masked` is **bf16** (result of Step 3 cast).
`F.normalize` operates in the tensor's dtype.

| | dtype |
|---|---|
| Input affinities | bf16 |
| L1 norm computation | **bf16** |
| Normalized affinities | **bf16** |

---

## Step 5 — Expert MLP (gate-up projection, SiLU, down projection)

Source: `experts.py`, `Experts._activation()` and the compiled matmuls.
Called via `forward_selective_loading` or `forward_all_experts` (both invoke the same
`Experts.forward()` which calls `gate_up_proj` → activation → `down_proj`).

All weight tensors are `torch_dtype = bfloat16`.

### Without `--enable-mixed-precision-accumulation` (baseline)

Every matmul is bf16×bf16 → bf16, accumulation in bf16 (hardware default).

### With `--enable-mixed-precision-accumulation` (actual)

The compiler promotes matmul accumulators to float32 for every bf16 matmul in the
compiled graph. The **Python-visible tensor dtype remains bf16 throughout** — the
promotion is invisible at the Python/XLA level; only the internal Neuron ISA
instruction changes.

| Operation | Input dtype | Weight dtype | Accumulation | Output dtype |
|---|---|---|---|---|
| Gate-up projection | **bf16** | bf16 | **float32** (compiler) | **bf16** |
| SiLU activation | bf16 | — | — | **bf16** |
| Gate × Up (elementwise) | bf16 | — | — | **bf16** |
| Down projection | **bf16** | bf16 | **float32** (compiler) | **bf16** |
| Affinity × expert output | bf16 | — | — | **bf16** |
| Expert accumulation (`+=`) | bf16 | — | — | **bf16** |

Key difference from the NKI TKG megakernel: SiLU is applied to a **bf16** tensor
(not float32 SBUF). The gate×up intermediate is bf16. The down-projection input is bf16.

---

## Step 6 — All-Reduce (TP)

`reduce_from_tensor_model_parallel_region` preserves dtype. Output is **bf16**.

---

## Complete datapath

```
hidden_states [B, 1, H]  bf16
       │
       ▼  CustomRMSNorm
       │  bf16 → float32 (explicit cast) → RmsNorm op → bf16
       │
rmsnorm_out [B, 1, H]  bf16
       │
       ├──── Router ────────────────────────────────────────────
       │     bf16 cast to float32 (router weight dtype)
       │     float32 @ float32 → router_logits  float32
       │     F.softmax(..., dtype=float64) → affinities  float64
       │     .to(bf16) → expert_affinities  bf16
       │     F.normalize (bf16) → normalized_affinities  bf16
       │     torch.topk → expert_index  int64
       │
       └──── Expert MLP ─────────────────────────────────────────
             gate_up: bf16 @ bf16 → [float32 accum] → bf16
             SiLU:    bf16 → bf16
             gate×up: bf16 * bf16 → bf16
             down:    bf16 @ bf16 → [float32 accum] → bf16
             × aff:   bf16 * bf16 → bf16
             Σ expt:  bf16 + bf16 → bf16

output [B, 1, H]  bf16
```

`[float32 accum]` = internal to the compiled Neuron op; tensor remains bf16 at Python level.

---

## Precision to match in a replacement NKI kernel

| Operation | Must match |
|---|---|
| RMSNorm variance | float32 intermediate, bf16 output |
| Router matmul | float32×float32 inputs and weights |
| Softmax | float32 input; numerically equivalent to float64 (high-precision exp/sum) |
| Affinity normalization | bf16 L1-norm |
| Gate-up matmul | bf16×bf16, float32 PSUM accumulation |
| SiLU | applied to bf16 (not float32) |
| Gate×up product | bf16 |
| Down matmul | bf16×bf16, float32 PSUM accumulation |
| Affinity scale | bf16 × bf16 → bf16 |
| Expert accumulation | bf16 + bf16 |

The critical difference from the NKI TKG megakernel (`init_tkg_module=True`) is that
**SiLU, gate×up, and the down-projection input are all bf16 here**, whereas the
megakernel explicitly keeps those intermediates in float32 SBUF.
