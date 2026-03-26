# NKI Router TopK Integration Plan

Integrating `kernels/router_topk/qwen3_router_topk_plan_a.py` into `qwen_with_attn_cte_nki.py`
to replace the PyTorch `RouterTopK` computation with the NKI kernel.

---

## Execution Path Summary

```
hidden_states [T, H]
    ↓
[MoE._forward_compute_bound()]  ← model.py:154
    ↓
[RouterTopK.forward()]          ← routing.py:192   ← TARGET: REPLACE THIS
    ├─ get_router_logits()      → router_logits [T, E]
    ├─ apply_activation_fn()    → expert_affinities [T, E] (softmax, float64→dtype)
    └─ torch.topk()             → expert_index [T, top_k]
    ↓
[ExpertMLPsV2.forward_blockwise()]  ← expert_mlps_v2.py:692
    ├─ get_expert_mask(expert_index)                         → expert_mask [T, E]
    ├─ get_expert_affinities_masked(affinities, mask, ...)   → expert_affinities_masked [T, E]
    ├─ get_blockwise_expert_and_token_mapping(...)           → block_to_expert [N,], token_pos [N*B,]
    └─ BlockwiseMatmulNKIFunc.apply(...)                     → output [T, H]
```

---

## Shape/Dtype Interface Mismatches

| Item | NXD `RouterTopK` expects | Your NKI kernel produces |
|------|--------------------------|--------------------------|
| Input `x` | `[T, H]` bf16 | `[H, T]` bf16 (transposed) |
| Input `w` | `Linear([H, E])` stored as `[E, H]` | `[H, E]` (transposed) |
| `router_logits` out | `[T, E]` bf16 | `[T, E]` float32 |
| `expert_affinities` out | `[T, E]` full softmax, hidden_states.dtype | `[T, E]` L1-normalized scattered top-K, float32 |
| `expert_index` out | `[T, K]` int64 | `[T, K]` uint32 |
| Kernel launch | — | `[2]` (LNC=2, requires 2 NeuronCores) |

**Critical semantic difference**: `ea_out` from the kernel is not the raw softmax over all E experts.
It is the L1-normalized top-K values scattered back into a `[T, E]` tensor (zeros at non-selected
positions). This is exactly what `get_expert_affinities_masked(..., normalize_top_k_affinities=True)`
would produce downstream — so the NXD masking step becomes a safe no-op if
`normalize_top_k_affinities=True` is set in the blockwise config.

---

## Minimal Integration Plan

### Step 1: Create a custom router class in `qwen_with_attn_cte_nki.py`

Subclass `RouterTopK` and override `forward()`. Use `self.weight_T` (`[H, E]`) which
`RouterBase` already creates as a registered parameter when `store_transposed_weights=True`
(routing.py:74-75). This is the same parameter the TKG fused kernel uses, so both paths
share one pre-transposed copy — no runtime transpose needed anywhere.

```python
from neuronx_distributed.modules.moe.routing import RouterTopK
from kernels.router_topk.qwen3_router_topk_plan_a import qwen3_router_topk_cte
import torch

class NKIRouterTopK(RouterTopK):
    def forward(self, hidden_states):
        # hidden_states: [T, H] bf16
        T, H = hidden_states.shape

        # NKI kernel expects [H, T] and [H, E]
        x = hidden_states.T.contiguous()  # [H, T] bf16
        w = self.weight_T                 # [H, E] bf16 — pre-transposed, no runtime .T needed

        # Allocate output buffers
        E = self.num_experts
        K = self.top_k
        rl = torch.empty((T, E), dtype=torch.float32, device=x.device)
        ea = torch.empty((T, E), dtype=torch.float32, device=x.device)
        ei = torch.empty((T, K), dtype=torch.int32,   device=x.device)

        # Launch NKI kernel (LNC=2)
        rl, ea, ei = qwen3_router_topk_cte[2](x, w, rl, ea, ei)

        # Dtype conversions to match NXD interface
        router_logits     = rl.to(hidden_states.dtype)  # [T, E]
        expert_affinities = ea.to(hidden_states.dtype)  # [T, E] already masked+normalized
        expert_index      = ei.to(torch.long)           # [T, K] int64

        return router_logits, expert_affinities, expert_index
```

### Step 2: Swap the router after MoE module construction

In `NeuronQwen3MoeDecoderLayerWithNKIAttn.__init__` (line 554), after `initialize_moe_module`
creates `self.mlp`:

```python
# After line 556: self.mlp = initialize_moe_module(...)
# Swap in the NKI router, preserving the Linear weight and weight_T:
old_router = self.mlp.router
nki_router = NKIRouterTopK(
    num_experts=old_router.num_experts,
    top_k=old_router.top_k,
    hidden_size=old_router.linear_router.in_features,
    act_fn=old_router.act_fn,
    store_transposed_weights=True,   # ensures weight_T is created and kept in sync
)
nki_router.linear_router = old_router.linear_router  # share the weight tensor
nki_router.weight_T = old_router.weight_T            # share the pre-transposed tensor
self.mlp.router = nki_router
```

This preserves weight loading — the state dict still maps to `mlp.router.linear_router.weight`
as line 248 of the convert function sets up. `weight_T` is a separate registered parameter
already handled by NXD's checkpoint logic.

### Step 3: Ensure `normalize_top_k_affinities=True` in blockwise config

Since `ea_out` is already L1-normalized top-K scattered back, the downstream masking step must
treat renormalization as a no-op. Set this in `Qwen3MoEAttnCTENeuronConfig`:

```python
class Qwen3MoEAttnCTENeuronConfig(MoENeuronConfig):
    @classmethod
    def get_blockwise_matmul_config_defaults(cls):
        defaults = super().get_blockwise_matmul_config_defaults()
        defaults["normalize_top_k_affinities"] = True  # ea_out already normalized
        return defaults
```

---

## Hardcoded Constants to Verify

The kernel (`plan_a.py:68-80`) hardcodes:

| Constant | Value | What to check |
|----------|-------|---------------|
| `H` | 2048 | Qwen3-30B hidden_size |
| `E` | 128 | num_experts |
| `K` | 8 | top_k |
| `T_TILE` | 128 | Must divide T evenly, or ceiling-division path handles remainder |

If these are wrong for the model config, the kernel will silently produce wrong shapes or crash.
Add runtime assertions before the kernel call in `NKIRouterTopK.forward()`.

---

## What You Don't Need to Change

- **State dict conversion** (`convert_qwen3_moe_hf_to_neuron_state_dict`): weight path
  `mlp.router.linear_router.weight` is preserved since `NKIRouterTopK` inherits `linear_router`.
  `weight_T` is a separate registered parameter already handled by NXD's checkpoint logic.
- **Blockwise NKI matmul kernel**: receives `expert_affinities_masked` from the downstream masking
  step; the router just feeds into that.
- **`ExpertMLPsV2.forward_blockwise()`**: the interface contract is fully met by returning the
  correct shapes and dtypes from `NKIRouterTopK.forward()`.
- **TKG fused kernel path**: it already uses `self.router.weight_T` (`[H, E]`) directly
  (moe_fused_tkg.py:306, moe_fused_tkg_mx.py:341). Sharing `weight_T` between `NKIRouterTopK`
  and the TKG kernel is both correct and zero-cost.

## TKG/CTE Weight Layout Compatibility

`RouterBase` maintains two weight representations when `store_transposed_weights=True`:

| Parameter | Shape | Used by |
|-----------|-------|---------|
| `linear_router.weight` | `[E, H]` | Standard nn.Linear forward (fallback path in routing.py:100) |
| `weight_T` | `[H, E]` | TKG fused kernels (moe_fused_tkg.py:306), NKI router (this plan) |

Both are registered parameters and both are saved/loaded from checkpoints. The state dict
conversion does not need to change. Storing `linear_router.weight` as `[H, E]` natively
(as previously considered) would break the TKG path by making `weight_T = linear_router.weight.T`
= `[E, H]` — the wrong shape for the TKG kernel. Use `weight_T` instead.

---

## Risk Summary

| Risk | Severity | Mitigation |
|------|----------|-----------|
| `ea_out` double-normalization | Medium | Set `normalize_top_k_affinities=True`; L1-norm of already-L1-normalized tensor is identity |
| Hardcoded H/E/K in kernel | High | Add runtime assertions in `NKIRouterTopK.forward()` before kernel call |
| LNC=2 vs model's `logical_nc_config` | Medium | Kernel is launched with `[2]` fixed; ensure model config matches |
| uint32 → int64 index conversion | Low | Explicit `.to(torch.long)` handles this |
| Sequence parallel reshape | Medium | `RouterBase.get_router_logits()` reshapes `[S, B, H] → [T, H]` before calling `.forward()`; the NKI router overrides `.forward()` so it already receives `[T, H]` |
| `store_transposed_weights=False` on old router | Medium | If `initialize_moe_module` created the router without `store_transposed_weights=True`, `old_router.weight_T` won't exist; ensure the MoE config sets this flag, or compute `weight_T` manually at swap time |

---

## File References

| File | Relevant Location |
|------|-------------------|
| `neuronx_distributed/modules/moe/model.py` | `MoE._forward_compute_bound()` line 154, `MoE.forward()` line 260 |
| `neuronx_distributed/modules/moe/routing.py` | `RouterBase` line 12, `store_transposed_weights` / `weight_T` lines 74-75, `RouterTopK.forward()` line 192 |
| `neuronx_distributed/modules/moe/moe_fused_tkg.py` | `_router_topk()` line 267, `self.router.weight_T` usage line 306; mega-kernel line 446 |
| `neuronx_distributed/modules/moe/moe_fused_tkg_mx.py` | `_prepare_kernel_inputs()` line 294, `router_weights` line 341 |
| `neuronx_distributed/modules/moe/expert_mlps_v2.py` | `ExpertMLPsV2.forward_blockwise()` line 692, `get_expert_mask()` line 274, `get_expert_affinities_masked()` line 299 |
| `neuronx_distributed/modules/moe/blockwise.py` | `BlockwiseMatmulNKIFunc.forward()` line 956 |
| `qwen_with_attn_cte_nki.py` | `NeuronQwen3MoeDecoderLayerWithNKIAttn.__init__()` line 542, MoE call line 605, weight conversion line 248 |
| `kernels/router_topk/qwen3_router_topk_plan_a.py` | `qwen3_router_topk_cte()` line 84 |
