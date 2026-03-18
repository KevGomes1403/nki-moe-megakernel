# Qwen3-30B-A3B Shape Reference for NKI Kernel Development

**Model**: Qwen3-30B-A3B (`Qwen3MoeForCausalLM`)
**Hardware**: AWS Trainium2 (NeuronCore-v3 / trn2)
**Tensor Parallelism**: TP=4
**Dtype**: bfloat16 (activations, weights), float32 (router logits/affinities)

---

## 1. Global Model Parameters

| Symbol | Parameter | Value |
|--------|-----------|-------|
| `H`    | hidden_size | 2048 |
| `Hq`   | num_attention_heads | 32 |
| `Hkv`  | num_key_value_heads | 4 |
| `d`    | head_dim | 128 |
| `I`    | moe_intermediate_size (per expert) | 768 |
| `E`    | num_experts | 128 |
| `K`    | num_experts_per_tok | 8 |
| `L`    | num_hidden_layers | 48 |
| `V`    | vocab_size | 151936 |
| `S_max`| max_position_embeddings | 40960 |

**TP-derived dimensions (TP=4):**

| Symbol | Description | Value |
|--------|-------------|-------|
| `Hq_tp` | Q heads per TP rank | 32/4 = **8** |
| `Hkv_tp` | KV heads per TP rank (replicated via `REPLICATE_TO_TP_DEGREE`) | **4** (all heads on every rank) |
| `I_tp`  | MoE intermediate per TP rank | 768/4 = **192** |
| `GQA_ratio` | Q heads per KV head per TP rank | 8/4 = **2** |

> **KV Replication Note**: `GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE`. With 4 KV heads and TP=4, each TP rank holds all 4 KV heads (fully replicated). Q is sharded: each TP rank holds 8 of 32 Q heads.

---

## 2. Variable Dimensions at Runtime

| Symbol | Description | CTE (context encoding) | TKG (token generation) |
|--------|-------------|------------------------|------------------------|
| `B`    | batch size | 1 (default) | 1 (default) |
| `S`    | active sequence length (q_len) | ≤ max_context_length (default 128) | 1 |
| `T`    | total tokens = B × S | B × S | B × 1 = B |
| `S_prior` | KV cache tokens in use | 0 | ≤ seq_len (default 640) |

Default run config (from `main.py`): `--seq-len 640 --batch-size 1 --tp-degree 4`

---

## 3. Embedding Layer

| Tensor | Global Shape | Per-TP Shape | Notes |
|--------|-------------|--------------|-------|
| `embed_tokens.weight` | [V, H] = [151936, 2048] | [151936, 512] | Sharded across H (`shard_across_embedding=True`) |
| Input token IDs | [B, S] | [B, S] | Not sharded |
| Embedded output | [B, S, H] | [B, S, 512] | Each TP rank holds partial H; all-gathered before layer 0 |

---

## 4. Per-Decoder-Layer Shapes (×48 layers)

### 4.1 Layer Norm Weights

| Module | Shape | Notes |
|--------|-------|-------|
| `input_layernorm.weight` | [H] = [2048] | RMSNorm; replicated on all TP ranks |
| `post_attention_layernorm.weight` | [H] = [2048] | RMSNorm; replicated on all TP ranks |

### 4.2 Self-Attention — Projection Weights

| Weight | Global Shape | Per-TP Shape | Notes |
|--------|-------------|--------------|-------|
| `q_proj.weight` | [Hq×d, H] = [4096, 2048] | [1024, 2048] | Column-parallel; shards Q heads across TP |
| `k_proj.weight` | [Hkv×d, H] = [512, 2048] | [512, 2048] | Replicated (all KV heads on every rank) |
| `v_proj.weight` | [Hkv×d, H] = [512, 2048] | [512, 2048] | Replicated |
| `o_proj.weight` | [H, Hq×d] = [2048, 4096] | [2048, 1024] | Row-parallel; each TP holds Hq_tp heads |
| `q_layernorm.weight` | [d] = [128] | [128] | Per-head RMSNorm; replicated |
| `k_layernorm.weight` | [d] = [128] | [128] | Per-head RMSNorm; replicated |

> **Fused QKV variant** (`--fused-qkv`): `Wqkv.weight` shape is `[(Hq+2×Hkv)×d, H] = [5120, 2048]` globally, with the Q portion sharded ([1024, 2048]) and KV replicated ([1024, 2048]) per TP rank.

### 4.3 Self-Attention — Activation Tensors

#### Context Encoding (CTE, q_len = S)

| Tensor | Shape (per-TP rank) | Notes |
|--------|---------------------|-------|
| `hidden_states` (input) | [B, S, H] = [B, S, 2048] | Full H pre-attention |
| `Q` after q_proj + q_layernorm | [B, S, Hq_tp×d] = [B, S, 1024] | Sharded |
| `K` after k_proj + k_layernorm | [B, S, Hkv×d] = [B, S, 512] | Replicated (Hkv_tp=4) |
| `V` after v_proj | [B, S, Hkv×d] = [B, S, 512] | Replicated (Hkv_tp=4) |
| `Q` reshaped for attention | [B, Hq_tp, S, d] = [B, 8, S, 128] | |
| `K` reshaped for attention | [B, Hkv_tp, S, d] = [B, 4, S, 128] | |
| `V` reshaped for attention | [B, Hkv_tp, S, d] = [B, 4, S, 128] | |
| Attention output | [B, Hq_tp, d, S] = [B, 8, 128, S] | Output of `FlashAttentionStrategy.UNSHARDED_KERNEL` |
| After o_proj + all-reduce | [B, S, H] = [B, S, 2048] | Full H restored |

**`attention_cte` kernel call signature** (from `perform_prefill`):

```python
q:  [B*Hq_tp, S, d]    = [B*8,  S,   128]   # (batch×heads, seq, head_dim)
k:  [B*Hkv_tp, d, S]   = [B*4,  128, S  ]   # transposed: (batch×kv_heads, head_dim, seq)
v:  [B*Hkv_tp, S, d]   = [B*4,  S,   128]
# output → reshaped to [B, Hq_tp, d, S] = [B, 8, 128, S]
```

**With prefix caching** (`perform_prefix_prefill`), adds:

```python
k_prior: [B*Hkv_tp, d, S_prior]  = [B*4, 128, S_prior]
v_prior: [B*Hkv_tp, S_prior, d]  = [B*4, S_prior, 128]
prior_used_len: [1]  # scalar int32
```

#### Token Generation (TKG, q_len = 1)

| Tensor | Shape (per-TP rank) | Notes |
|--------|---------------------|-------|
| `hidden_states` (input) | [B, 1, H] = [B, 1, 2048] | |
| `Q` | [B, Hq_tp, 1, d] = [B, 8, 1, 128] | |
| `K` (active, pre-cache) | [B, Hkv_tp, 1, d] = [B, 4, 1, 128] | |
| `V` (active) | [B, Hkv_tp, 1, d] = [B, 4, 1, 128] | |
| `K_cache` (prior) | [B, Hkv_tp, S_prior, d] = [B, 4, S_prior, 128] | or transposed `[B, 4, d, S_prior]` if `k_cache_transposed` |
| `V_cache` (prior) | [B, Hkv_tp, S_prior, d] = [B, 4, S_prior, 128] | |
| `attention_mask` | [B, 1, 1, S_prior] | Causal mask over prior tokens |

**`attention_tkg` kernel constraints** (from `_attention_tkg_decode_disable_reason`):

| Constraint | Value |
|------------|-------|
| Max batch×kv_heads | 128 → max B = 128 / Hkv_tp = 32 |
| Max q_len (S_active) | 8 |
| Max S_prior | 32768 |
| S_prior must be | ≤256 OR divisible by 128 |
| head_dim | must be divisible by 64 |
| head_dim=128 | disabled by default (set `attn_tkg_allow_head_dim_128=True`) |

---

### 4.4 MoE Block — Router Weights

| Tensor | Global Shape | Per-TP Shape | Notes |
|--------|-------------|--------------|-------|
| `mlp.router.linear_router.weight` | [E, H] = [128, 2048] | [128, 2048] | Replicated on all TP ranks; float32 output |

### 4.5 MoE Block — Expert Weights (after state-dict conversion)

Expert weights are stored in fused form after `convert_qwen3_moe_hf_to_neuron_state_dict`:

| Tensor | Global Shape | Per-TP Shape | Memory (BF16, per TP) |
|--------|-------------|--------------|----------------------|
| `mlp.expert_mlps.mlp_op.gate_up_proj.weight` | [E, H, 2×I] = [128, 2048, 1536] | [128, 2048, 384] | 128×2048×384×2 ≈ **192 MB** |
| `mlp.expert_mlps.mlp_op.down_proj.weight` | [E, I, H] = [128, 768, 2048] | [128, 192, 2048] | 128×192×2048×2 ≈ **96 MB** |

> **Weight layout**: `gate_up_proj[e, h, :]` = concat(gate_proj[e,h,:], up_proj[e,h,:]) along last dim. Gate is [:I_tp], up is [I_tp:2*I_tp].

### 4.6 MoE Block — Activation Tensors

#### Context Encoding (CTE, T = B×S tokens)

| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `hidden_states` (post-norm) | [T, H] = [T, 2048] | bf16 | Input to router and expert MLPs |
| Router logits | [T, E] = [T, 128] | float32 | Pre-softmax |
| `affinities` | [T, E] = [T, 128] | float32 | Post-softmax, before top-K |
| `expert_index` | [T, K] = [T, 8] | int64 | Top-K expert indices |
| `routing_weights` (sparse) | [T, E] = [T, 128] | float32 | K non-zeros per row (normalized) |
| gate+up intermediate | [T_active, 2×I_tp] | bf16 | Only activated tokens×experts |
| After SiLU+multiply | [T_active, I_tp] | bf16 | |
| After down_proj | [T, H] | bf16 | Accumulated from K experts |
| After all-reduce | [T, H] = [T, 2048] | bf16 | Sum across TP ranks |

#### Token Generation (TKG, T = B tokens)

Same shapes as CTE but with T = B (typically B=1 for latency-optimized decode).

**`nki_moe_fused` kernel call signature** (from `_forward_moe_fused`):

```python
hidden_2d:       [T, H]      = [B, 2048]    # bf16
gate_up_w:       [E, H, 2*I_tp] = [128, 2048, 384]  # bf16, padded to 2*I_tp_padded if I_tp%128≠0
down_w:          [E, I_tp, H]   = [128, 192, 2048]   # bf16, padded if needed
routing_weights: [T, E]      = [B, 128]     # float32, sparse K-of-E
# output: [T, H] = [B, 2048]  bf16
```

> **Padding note**: The code pads I_tp to the next multiple of 128 before the kernel call. With I_tp=192 (already divisible by 128? No: 192/128=1.5), it pads to 256. So actual kernel shapes are:
> - `gate_up_w`: [128, 2048, **512**] (2×256)
> - `down_w`: [128, **256**, 2048]

---

## 5. Final Norm and LM Head

| Module | Global Shape | Per-TP Shape | Notes |
|--------|-------------|--------------|-------|
| `norm.weight` (RMSNorm) | [H] = [2048] | [2048] | Replicated |
| `lm_head.weight` | [H, V] = [2048, 151936] | [2048, 37984] | Column-parallel; V/TP = 151936/4 = 37984 |
| LM head output (pre-gather) | [B, V/TP] = [B, 37984] | [B, 37984] | |
| LM head output (post-gather) | [B, V] = [B, 151936] | — | Gathered unless on-device sampling |

---

## 6. KV Cache

Stored per-layer, per-TP rank:

| Tensor | Shape | Dtype | Memory per layer (BF16, B=1, S_max=640) |
|--------|-------|-------|----------------------------------------|
| `K_cache` | [B, Hkv_tp, S_max, d] = [B, 4, 640, 128] | bf16 | 1×4×640×128×2 = **655 KB** |
| `V_cache` | [B, Hkv_tp, S_max, d] = [B, 4, 640, 128] | bf16 | **655 KB** |
| Total KV cache (48 layers) | — | — | 48 × 2 × 655 KB ≈ **61 MB** per TP rank |

> With `k_cache_transposed=True`: K_cache shape becomes [B, Hkv_tp, d, S_max] = [B, 4, 128, 640].

---

## 7. Complete Data Flow Summary

```
Input IDs [B, S]
    ↓ embed_tokens
[B, S, H] bf16
    ↓ × 48 decoder layers
    ├─ input_layernorm → [B, S, H]
    ├─ Self-Attention
    │   ├─ q_proj → [B, S, Hq_tp×d]  [B, S, 1024]
    │   ├─ k_proj → [B, S, Hkv×d]    [B, S, 512]
    │   ├─ v_proj → [B, S, Hkv×d]    [B, S, 512]
    │   ├─ q_layernorm, k_layernorm (per head, d=128)
    │   ├─ RoPE embeddings
    │   ├─ CTE: attention_cte(q[B*8,S,128], k[B*4,128,S], v[B*4,S,128])
    │   │   → [B, 8, 128, S]
    │   ├─ TKG: attention_tkg/flash_attention
    │   │   Q[B,8,1,128], K[B,4,1,128], K_cache[B,4,S_prior,128]
    │   └─ o_proj + all-reduce → [B, S, H]
    ├─ residual add → [B, S, H]
    ├─ post_attention_layernorm → [B, S, H]  (applied internally by mlp)
    └─ MoE MLP
        ├─ router: [T,H]→[T,128] logits → [T,8] top-K indices + [T,128] weights
        ├─ CTE: standard MoE module (initialize_moe_module)
        ├─ TKG: nki_moe_fused(hidden[T,H], gate_up[128,H,384], down[128,192,H], weights[T,128])
        └─ all-reduce → [B, S, H]
    ↓ norm → [B, S, H]
    ↓ lm_head → [B, S, V/TP] → gather → [B, S, V]
Output logits [B, S, V]
```

---

## 8. Competition Kernel Target Shapes

### Priority 1: MoE Token Generation (highest impact)

The `nki_moe_fused` kernel is the dominant compute in TKG.

| Input | Shape | Dtype |
|-------|-------|-------|
| `hidden` | [T, 2048] | bf16, T=B (1–32 typical) |
| `gate_up_w` | [128, 2048, 512] | bf16 (padded from 384) |
| `down_w` | [128, 256, 2048] | bf16 (padded from 192) |
| `routing_weights` | [T, 128] | float32 |
| **Output** | [T, 2048] | bf16 |

**FLOPs per token (single layer)**: K × (2×H×2I + 2×H×I) = 8 × 3×768×2048 ≈ **75 MFLOPs** (unsharded); **19 MFLOPs** per TP rank. Across all 48 layers: **3.6 GFLOPs** per token unsharded.

### Priority 2: Attention Context Encoding (second highest impact)

| Input | Shape | Dtype |
|-------|-------|-------|
| `q` | [B×8, S, 128] | bf16 |
| `k` | [B×4, 128, S] | bf16 |
| `v` | [B×4, S, 128] | bf16 |
| **Output** | [B×8, S, 128] → reshaped [B, 8, 128, S] | bf16 |

Typical CTE shapes (B=1, S=128): q=[8,128,128], k=[4,128,128], v=[4,128,128].

### Priority 3: MoE Context Encoding

Same expert weights as TKG; T = B×S (up to ~128 tokens), same weight shapes.

### Priority 4: QKV Projection + RMSNorm (fused)

| Kernel opportunity | Weight | Input | Output |
|-------------------|--------|-------|--------|
| Fused QKV + Q/K norm | Wqkv [5120, 2048] | [B, S, 2048] | Q [B,S,1024], K [B,S,512], V [B,S,512] |

### Priority 5: Attention Token Generation

| Input | Shape | Dtype |
|-------|-------|-------|
| `Q` | [B, 8, 1, 128] | bf16 |
| `K_active` | [B, 4, 1, 128] | bf16 |
| `K_prior` | [B, 4, S_prior, 128] | bf16 |
| `V_prior` | [B, 4, S_prior, 128] | bf16 |
| **Output** | [B, 1, 1024] | bf16 |

---

## 9. Numerical Dtype Summary

| Operation | Input Dtype | Output Dtype | Notes |
|-----------|-------------|--------------|-------|
| All weights | bf16 | — | Stored as bf16 |
| All activations | bf16 | bf16 | |
| Router logits | bf16 → float32 | float32 | Cast for precision |
| Expert affinities | float32 | float32 | Softmax in fp32 |
| Routing weights | float32 | float32 | Must stay fp32 for `nki_moe_fused` (POST_SCALE mode) |
| Attention scale | float32 | — | `1/√128 ≈ 0.0884` |
| RMSNorm eps | float32 | — | `1e-6` |

---

## 10. Quick Reference: Key Numbers

```
H=2048  Hq=32  Hkv=4  d=128  I=768  E=128  K=8  L=48  V=151936

Per TP (TP=4):
  Hq_tp=8   Hkv_tp=4 (replicated)   I_tp=192 (→ padded to 256 for kernel)

Single TKG step, B=1:
  hidden:   [1, 2048]   bf16
  gate_up:  [128, 2048, 512]  bf16   (~201 MB total weights on chip)
  down:     [128, 256, 2048]  bf16   (~100 MB total weights on chip)
  output:   [1, 2048]   bf16

Single CTE step, B=1, S=128:
  q:  [8, 128, 128]  bf16
  k:  [4, 128, 128]  bf16
  v:  [4, 128, 128]  bf16
```
