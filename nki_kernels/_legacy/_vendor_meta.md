# Vendored nkilib subkernels — audit metadata

The gpt-oss megakernel (`megakernels/gpt_oss/transformer_gpt_oss.py`)
imports per-layer subkernels from `nki_kernels/attention/` and
`nki_kernels/moe/`. Those are **vendored copies** of files in
`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/nkilib/`
with one mechanical edit applied: every direct
`nisa.dma_copy(name="<lit>")` / `nl.ndarray(name="<lit>")` /
`nl.zeros(name="<lit>")` / etc. has its `name=` argument prefixed with
`sbm.get_name_prefix()` so the same function can be traced multiple times
in one `nki.jit` (once per layer × 24 layers) without producing duplicate
op names.

## Pinned versions

| Component | Version |
|---|---|
| Neuron SDK compiler (`neuronx-cc`) | `2.24.5133.0+58f8de22` |
| `nkilib` | `0.1.0+gd4920de` (built `Mar 17 2026 04:03:17 UTC`) |
| `neuronx-distributed-inference` | `0.9.17334+ced6ae4e` |
| Audit date | 2026-05-08 |

## Vendored files

```
nki_kernels/attention/
  attention_block_tkg.py        ← nkilib/experimental/transformer/attention_block_tkg.py
  attention_tkg.py              ← nkilib/core/attention/attention_tkg.py
  qkv_tkg.py                    ← nkilib/core/qkv/qkv_tkg.py
  output_projection_tkg.py      ← nkilib/core/output_projection/output_projection_tkg.py
  gen_mask_tkg.py               ← nkilib/core/attention/gen_mask_tkg.py (only if reachable from our config)

nki_kernels/moe/
  rmsnorm_tkg.py                ← nkilib/core/subkernels/rmsnorm_tkg.py
  router_topk.py                ← nkilib/core/router_topk/router_topk.py
  moe_tkg.py                    ← nkilib/core/moe/moe_tkg/moe_tkg.py
  selective_expert_impl.py      ← nkilib/core/moe/moe_tkg/selective_expert_impl.py
  selective_expert_mx_impl.py   ← nkilib/core/moe/moe_tkg/selective_expert_mx_impl.py
  down_projection_mx_shard_H.py ← nkilib/core/mlp/mlp_tkg/down_projection_mx_shard_H.py
  moe_tkg_utils.py              ← nkilib/core/moe/moe_tkg/moe_tkg_utils.py
  moe_tkg_affinity_masking.py   ← nkilib/core/moe/moe_tkg/moe_tkg_affinity_masking.py
```

Files **not** vendored (clean — no `name=` literals in our call path):
- `nkilib/core/embeddings/rope.py` (still imported by vendored attention_tkg)
- `nkilib/core/qkv/qkv.py` (still imported by vendored qkv_tkg)
- `nkilib/core/utils/*` (allocator, common_types, kernel_helpers, tensor_view, kernel_assert)
- `nkilib/experimental/transformer/transformer_tkg.py` (only `_sb2sb_all_reduce_gather` is used; clean)

## Maintenance protocol

On every Neuron SDK upgrade:
1. `pip show neuronx-cc` → check version delta vs. table above.
2. For each vendored file, `diff -u nkilib/<path> kernels/<path>` and verify
   the diff is **only** `name=` argument rewrites + intra-package import
   path changes. Anything else is upstream drift and needs review.
3. Re-run the AST scan from
   `docs/vendor_nkilib_for_megakernel_plan.md` to catch newly-added
   `name=` literals that need the same prefix treatment.
4. Recompile the megakernel via `main.py --enable-nki --model gpt_oss
   --mode generate --compile-only` and confirm logits parity.

## Coexistence note

`nki_kernels/attention/attn_fused_nki.py` and
`nki_kernels/moe/moe_fused_nki.py` are pre-existing Qwen3-MoE custom
kernels, **unrelated to this vendor effort**. They share the directories but
not the namespace; the vendored gpt-oss files use upstream filenames so
the two paths never collide.

## Phase 1 indirect-DMA fusion divergence (NKI_MOE_INDIRECT_DMA_FUSION)

`nki_kernels/moe/selective_expert_impl.py` and
`nki_kernels/moe/mlp_tkg_gate_up_projection.py` now diverge from upstream
`nkilib/core/moe/moe_tkg/selective_expert_impl.py` and
`nkilib/core/mlp/mlp_tkg/mlp_tkg_gate_up_projection.py` to support a new
env-gated path: `NKI_MOE_INDIRECT_DMA_FUSION=1` combines the existing fused
gate+up TensorView (`select(dim=0, expert).flatten_dims(1,2)` → `[H, 2*I]`)
with the cross-expert 2-slot prefetch ring. The non-fused baseline path
issues two DMAs per expert against a `[H, I]` view whose contiguous-inner
caps at `I*bf16=384B` (the second `select(dim=1, GATE/UP)` slices the
`2` axis between gate and up in HBM). The fused view's contiguous-inner is
`2*I*bf16=768B`, halving descriptor count and doubling packet size at
Qwen3-30B-A3B shapes; profile of layer 25 showed 31,685 of the 384B packets
and ~10% MBU. Concretely: one `[H0, H1_shard, 2*I]` prefetch slot per ring
entry (vs. two `[H0, H1_shard, I]` slots), one `emit_hoisted_gate_up_dma`
call per expert (vs. two), and routed into the lhs_rhs_swap matmul via a
new `pre_loaded_fused_hoist` / `pre_loaded_hoisted_gate_up_fused` plumbing
through `process_gate_up_projection`. `emit_hoisted_gate_up_dma` forces
`dge_mode=unknown` when the source view has dynamic access (HWDGE/SWDGE
are not safe for indirect patterns). With the env flag off, behavior is
unchanged from the prior vendored baseline (a strict additive change).
