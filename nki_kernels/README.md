# DeltaNet NKI kernels

Custom NKI kernels for the Qwen3.6-27B DeltaNet (gated linear-attention) layers.

Consumers import entrypoints from the **package root** — `from ...nki_kernels import
deltanet_fused_tkg_fwd` — which re-exports the stable surface (see `__init__.py`).
This keeps callers decoupled from the internal layout, so kernels can move as the
megakernel grows. The `_upstream/` vendored copy has its own separate `nki_kernels/`
and is not affected by this tree.

## Layout

```
nki_kernels/
├── __init__.py            # stable public surface (re-exports entrypoints)
├── deltanet/
│   ├── components/        # reusable stage kernels — the megakernel building blocks
│   │   ├── in_proj.py     # input RMSNorm + fused 4-way input projection
│   │   ├── conv.py        # causal short conv (TKG entrypoints + composables)
│   │   ├── recurrence.py  # gated delta-rule recurrence (gated_delta_rule_tkg)
│   │   ├── norm_gate.py   # gated RMSNorm (norm_gate_row)
│   │   └── out_proj.py    # output projection (out_proj_compose)
│   ├── decode/            # token generation (TKG)
│   │   ├── fused_layer.py # assembled DeltaNet attention megakernel  ← current
│   │   └── recurrent.py   # per-token recurrent kernel (flag-gated fallback)
│   └── prefill/           # context encoding (CTE)
│       ├── chunked_step.py  # per-chunk-step, numerically stable  ← default
│       └── chunked_fused.py # single-kernel fused; faster but overflows this
│                            #   checkpoint (gated off — see model README)
├── lm_head/
│   └── components/
│       └── lm_head.py     # final RMSNorm + vocab projection + per-rank greedy argmax
└── specs/                 # design specs & implementation rationale (see below)
```

## Kernel map

| Module | Stage / role | Regime | Status |
|---|---|---|---|
| `components/in_proj.py` | in-proj + RMSNorm | shared | building block |
| `components/conv.py` | causal short conv | shared | building block |
| `components/recurrence.py` | gated delta-rule recurrence | decode | building block |
| `components/norm_gate.py` | gated RMSNorm | shared | building block |
| `components/out_proj.py` | output projection | shared | building block |
| `decode/fused_layer.py` | in_proj+conv+recurrence fused into one layer | decode | **current** |
| `decode/recurrent.py` | per-token recurrence | decode | fallback (gated) |
| `prefill/chunked_step.py` | per-chunk-step forward | prefill | **default** |
| `prefill/chunked_fused.py` | single-kernel fused forward | prefill | gated off (fp32 overflow) |
| `lm_head/components/lm_head.py` | final norm + vocab projection + per-rank argmax | shared | building block |

`decode/fused_layer.py` composes the `components/` kernels; that is the megakernel
under active development. New stage kernels go in `components/`; new model-block
kernels (e.g. MoE, GQA attention) get sibling packages under `deltanet/`'s parent.

## Design specs

Layout algebra, sharding decisions and perf rationale live in `specs/` rather than in the kernel
source, so the source stays a low-friction entry point. Per-subsystem:

| Spec | Covers |
|---|---|
| `specs/deltanet_tkg.md` | conv + recurrence: input contracts, per-token math, state double-buffering, the paired read, PSUM bank limits |
| `specs/deltanet_prefill.md` | chunked prefill: which path is live and why, the fp32 overflow, the Neumann-series framework |
| `specs/gqa_tkg.md` | head_dim=256 partition tiling, attention tensor layouts, o_proj sub-head ordering, KV cache write modes |
| `specs/moe_tkg.md` | the tp2013 H-permutation, work-split vs. layout, the gate/up slab load and DMA fragmentation |
| `specs/lm_head.md`, `specs/eh_proj.md`, `specs/norm_gate.md`, `specs/out_proj.md`, `specs/pre_attn_rmsnorm.md`, `specs/moe_layer.md` | per-kernel specs |
| `specs/draft_megakernel.md`, `specs/round_megakernel_perf.md` | megakernel assembly and round-level perf |

## Adding a kernel

1. Drop the module in the right folder (`components/` for a reusable stage, else
   the regime folder).
2. Use **relative** imports between kernels (`from ..components.conv import ...`).
3. Re-export its public entrypoint(s) from `nki_kernels/__init__.py` and add them
   to `__all__` so consumers import from the package root.
4. Add a row to the kernel map above.
