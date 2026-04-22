# Token Generation Attention — `qwen.py` Dataflow & Precision Reference

**Assumes:** `attn_block_tkg_nki_kernel_enabled=False` (default in `qwen.py`'s `generate()` config).  
**Hardware:** Trainium3 (trn3), TP=4.  
**Model:** Qwen3-30B-A3B — `hidden_size=2048`, `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=128`.

---

## GQA Sharding: `REPLICATE_TO_TP_DEGREE`

With TP=4, the Wqkv is built around 4 unique KV heads (not 8). Each rank owns a disjoint slice of Q heads but an independent KV projection:

```
Q heads per rank:  16 / 4 = 4
KV heads per rank: 1  (4 unique KV heads across ranks, GQA ratio on-rank = 4:1)

Wqkv weight per rank:  [2048, 768]
  Q slice:  4 × 128 = 512
  K slice:  1 × 128 = 128
  V slice:  1 × 128 = 128
  Total:              768

o_proj weight per rank: [512, 2048]
  (RowParallelLinear — input sharded, output full hidden_size)
```

For the NKI kernel the Wqkv is CPU-transposed to `[768, 2048]` and o_proj to `[2048, 512]`, but that path is disabled here.

---

## Per-Step Dataflow (bs=1, seq=1 for TKG)

### Step 1 — Input LayerNorm (`input_layernorm`, CustomRMSNorm)

```
Input:   hidden_states  (1, 1, 2048)  bf16
Compute: cast to fp32, rms = sqrt(mean(x²) + eps), x_norm = x/rms * γ, cast to bf16
Output:  (1, 1, 2048)  bf16
```

Weight `γ`: shape `(2048,)`, dtype fp32.

### Step 2 — QKV Projection (ColumnParallelLinear, fused Wqkv)

```
Input:   (1, 1, 2048)  bf16
Weight:  Wqkv  [2048, 768]  bf16
Compute: matmul in bf16, no bias
Output:  (1, 1, 768)  bf16  → split:
  Q  (1, 1, 512)
  K  (1, 1, 128)
  V  (1, 1, 128)
```

Split indices (from `attention_base.py`):
```
q_end = num_attention_heads * head_dim / tp_degree  = 16*128/4 = 512
k_end = q_end + num_key_value_heads * head_dim / tp_degree = 512 + 8*128/4 = 768
```

### Step 3 — QK Layernorm PRE-ROPE (`q_layernorm`, `k_layernorm`, CustomRMSNorm)

Applied per-head on `head_dim=128`. This is **before** RoPE.

```
Reshape Q:  (1, 1, 512) → (1, 1, 4, 128)
Apply q_layernorm on dim=-1 (head_dim):
  cast to fp32, rms = sqrt(mean(x²) + eps), x_norm = x/rms * γ_q, cast to bf16
  γ_q shape: (128,)
Reshape K:  (1, 1, 128) → (1, 1, 1, 128)
Apply k_layernorm same way, γ_k shape: (128,)

Transpose to BHSD layout:
  Q  (1, 4, 1, 128)  bf16
  K  (1, 1, 1, 128)  bf16
  V  (1, 1, 1, 128)  bf16  [no norm on V]
```

### Step 4 — RoPE (`apply_rotary_pos_emb`)

```
cos_cache: (1, 1, 64)  bf16   [head_dim//2 = 64 frequencies]
sin_cache: (1, 1, 64)  bf16

q_rot = q * cos + rotate_half(q) * sin   [all bf16, no upcast]
k_rot = k * cos + rotate_half(k) * sin

Output:
  Q  (1, 4, 1, 128)  bf16
  K  (1, 1, 1, 128)  bf16
```

### Step 5 — KV Cache Update

```
Write K_new  (1, 1, 1, 128)  bf16  →  K_cache at [:, :, position_id, :]
Write V_new  (1, 1, 1, 128)  bf16  →  V_cache at [:, :, position_id, :]

K_cache layout (k_cache_transposed=False):  (1, 1, s_prior, 128)  bf16
V_cache layout:                             (1, 1, s_prior, 128)  bf16
```

s_prior = number of already-decoded positions (includes current token after write).

### Step 6 — GQA Expansion (`repeat_kv`)

```
K_cache  (1, 1, s_prior, 128)  →  K_exp  (1, 4, s_prior, 128)  bf16
V_cache  (1, 1, s_prior, 128)  →  V_exp  (1, 4, s_prior, 128)  bf16
```

Each KV head is repeated 4× to serve all 4 Q heads on this rank.

### Step 7 — Attention Scores

```
K_T = K_exp.transpose(2, 3)                         # (1, 4, 128, s_prior)  bf16
prior_scores = Q @ K_T / sqrt(128)                  # (1, 4, 1, s_prior)    bf16
prior_scores = prior_scores.to(float32)             # explicit upcast before masking

softmax_scale = 1 / sqrt(head_dim) = 1 / sqrt(128) ≈ 0.08838
```

The scale is applied to Q before the matmul (i.e., `Q_scaled = Q * scale` then `Q_scaled @ K_T`), which is equivalent.

For the active (current) token:
```
K_active = repeat_kv(K, 4)                          # (1, 4, 1, 128)  bf16
active_scores = Q @ K_active.transpose(2,3) / sqrt(128)  # (1, 4, 1, 1)  bf16
active_scores = active_scores.to(float32)
```

### Step 8 — Causal Masking

```
attention_mask: boolean, shape (1, 4, 1, s_prior)
prior_scores  = where(attention_mask, prior_scores, -inf)   # float32
```

### Step 9 — Softmax (manual, log-sum-exp)

Implemented in `manual_softmax` (`utils.py`):

```python
max_prior  = max(prior_scores,  dim=-1, keepdim=True)   # (1,4,1,1)  fp32
max_active = max(active_scores, dim=-1, keepdim=True)   # (1,4,1,1)  fp32
global_max = maximum(max_prior, max_active)             # (1,4,1,1)  fp32

exp_prior  = exp(prior_scores  - global_max)            # (1,4,1,s_prior)  fp32
exp_active = exp(active_scores - global_max)            # (1,4,1,1)        fp32

denom = exp_prior.sum(dim=-1, keepdim=True) + exp_active.sum(dim=-1, keepdim=True)

softmax_prior  = (exp_prior  / denom).to(bf16)          # (1,4,1,s_prior)  bf16
softmax_active = (exp_active / denom).to(bf16)          # (1,4,1,1)        bf16
```

All intermediate computations in fp32. Output cast to bf16 before the output matmul.

### Step 10 — Attention Output

```
attn_prior  = softmax_prior  @ V_exp      # (1,4,1,s_prior) @ (1,4,s_prior,128) = (1,4,1,128)  bf16
attn_active = softmax_active @ V_active   # (1,4,1,1)       @ (1,4,1,128)       = (1,4,1,128)  bf16
attn_output = attn_prior + attn_active    # (1,4,1,128)  bf16
```

### Step 11 — Reshape + Output Projection (RowParallelLinear)

```
attn_output: (1,4,1,128) → reshape → (1,1,512)  bf16
W_out per rank: [512, 2048]  bf16
output = attn_output @ W_out.T            # (1,1,2048)  bf16, no bias

AllReduce across TP=4 ranks              → (1,1,2048)  bf16  (sum contributions)
```

### Step 12 — Residual Add (in `NeuronQwen3MoeDecoderLayer.forward`)

```
hidden_states = residual + attn_output   # (1,1,2048)  bf16
```

---

## Precision Summary Table

| Operation | Compute dtype | Output dtype | Notes |
|-----------|--------------|--------------|-------|
| Input LayerNorm | fp32 | bf16 | CustomRMSNorm; γ stored fp32 |
| Wqkv matmul | bf16 | bf16 | no bias |
| QK LayerNorm (per-head, PRE-ROPE) | fp32 | bf16 | on head_dim=128; applied before RoPE |
| RoPE | bf16 | bf16 | no upcast |
| KV cache read/write | bf16 | bf16 | |
| GQA repeat_kv | bf16 | bf16 | 1 KV head → 4 copies |
| QK matmul | bf16 | bf16 → fp32 | immediately upcast after matmul |
| Causal masking | fp32 | fp32 | |
| Softmax (exp, sum, div) | fp32 | fp32 → bf16 | cast to bf16 before output matmul |
| Attention output matmul | bf16 | bf16 | |
| o_proj matmul | bf16 | bf16 | no bias |
| AllReduce | bf16 | bf16 | |
| Residual add | bf16 | bf16 | |

---

## Key Invariants

1. **QK norm is PRE-ROPE** — applied per-head on `head_dim=128` after QKV split, before rotary embedding.
2. **Softmax is fp32** — scores are bf16 from the matmul, explicitly cast to fp32 before masking and softmax.
3. **Manual softmax with global max** — prior and active scores share a single global max for numerical stability across both parts.
4. **GQA ratio on-rank is 4:1** — 4 Q heads, 1 KV head per rank (not the global 2:1 ratio).
5. **softmax_scale = 1/sqrt(128) ≈ 0.08838** — applied to Q before the score matmul.
6. **No bias** in Wqkv or o_proj.
7. **KV cache layout** (default): `(batch, kv_heads, seq, head_dim)` — not transposed.

---

## Source Locations

| Component | File | Key lines |
|-----------|------|-----------|
| TKG dispatch | `attention_base.py` | ~1753 |
| `compute_for_token_gen` | `attention_base.py` | ~1383–1461 |
| `manual_softmax` | `modules/attention/utils.py` | ~252–270 |
| QK norm application (`move_heads_front`) | `modules/attention/utils.py` | ~200–216 |
| GQA sharding / Wqkv init | `modules/attention/gqa.py` | ~411–420 |
| RoPE | `modules/attention/utils.py` | ~306–343 |
| Qwen3 attention class | `qwen.py` | 313–341 |
| Decoder layer forward | `qwen.py` | 378–436 |
