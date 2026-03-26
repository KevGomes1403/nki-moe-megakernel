# Router → Library Expert Kernel: CTE Pipeline Optimization Plan

Keeping the library `BlockwiseMatmulNKIFunc` expert kernel. Full analysis of every
optimization available between the custom router kernel output and the expert kernel input.

---

## Pipeline Overview (Current)

All sizes at T=640, E=128, K=8, N≤167, B=128, TP=4, E_kernel=E/TP=32.

```
hidden_states [T, H]  bf16
       |
       | .T.contiguous()  ← 2.5 MB HBM copy (T=640, H=2048)
       ↓
NKI Router kernel  [qwen3_router_topk_plan_a.py]
  → rl_out [T, E]   f32   320 KB  WRITE  ← UNUSED downstream
  → ea_out [T, E]   f32   320 KB  WRITE  ← scattered L1-normalized affinities
  → ei_out [T, K]   u32    20 KB  WRITE  ← used only by PyTorch mapping fallback
       |
       | maybe_get_expert_affinities_masked()        ← WASTED: rebuilds mask from ei_out
       |   get_expert_mask(expert_index)             ← O(T*E) XLA op
       |   get_expert_affinities_masked(ea, mask)    ← O(T*E) XLA op
       |
       | get_blockwise_expert_and_token_mapping_kernel()  or PyTorch fallback
       |   find_nonzero_indices(ea_out.to(f32))
       |     READ  ea_out   [T, E]         320 KB  ← reads what router just wrote
       |     WRITE indices  [E_kernel, T]   80 KB  ← intermediate, only consumed by indexed_flatten
       |   indexed_flatten(indices, row_offsets)
       |     READ  indices  [E_kernel, T]   80 KB
       |     WRITE token_position_to_id [N*B]  83 KB
       |
       ↓
Expert kernel
  READ  hidden_states        [T, H]    2.5 MB
  READ  ea_out               [T, E]    320 KB
  READ  token_position_to_id [N*B]      83 KB
  READ  block_to_expert      [N]         1 KB
  READ  gate_up_proj_weight  [E,H,2I]  dominant
  READ  down_proj_weight     [E,I,H]   dominant
```

---

## Checklist

### A. Router Kernel Output Changes

**A1. Drop `rl_out` entirely**
- Delete the `rl_out = nl.ndarray(...)` allocation and its `nisa.dma_copy` write.
- Delete `rl_out` from the return tuple.
- Update caller to unpack 2 tensors: `ea_out, ei_out = qwen3_router_topk_cte[2](...)`.
- Saves: 320 KB HBM write per forward pass, no correctness risk.

**A2. Conditionally drop `ei_out`**
- `find_nonzero_indices` reads `ea_out [T, E]` directly — it never looks at `ei_out`.
- When `use_index_calc_kernel=True` is guaranteed (see C1), `ei_out` is unused end-to-end.
- Gating: add a Python-level flag `write_expert_index: bool = False` to
  `NKIRouterTopK.__init__`; pass to the kernel as a compile-time constant.
- Saves: 20 KB HBM write; also eliminates the int32→int64 cast in the caller.
- Risk: if the path ever falls back to PyTorch mapping, `expert_index` is needed.
  Add an assertion (see D3) to hard-fail rather than silently produce wrong output.

**A3. Keep `ea_out` as float32 (do not change to bf16)**
- `find_nonzero_indices` calls `.to(torch.float32)` unconditionally — casting bf16 input
  would create an extra 320 KB HBM write instead of saving one.
- The expert blockwise kernel receives `expert_affinities_masked` after the bypass (see B1);
  it handles f32 natively and the bandwidth is small relative to weight loads.
- No change needed here.

**A4. Assert T divisibility before kernel launch**
- `can_use_find_index_kernel` returns `False` (falls back to slow PyTorch path) if
  `T % block_size != 0`. Add a hard assertion at the top of `NKIRouterTopK.forward()`:
  ```python
  T = hidden_states.shape[0]
  assert T % 128 == 0, f"T={T} must be divisible by block_size=128 for index calc kernel"
  assert T == _ROUTER_H  # ... existing shape checks
  ```
- To guarantee this holds: pad CTE input sequences to multiples of `TP × block_size = 512`
  tokens before the model forward.

**A5. Keep the `.T.contiguous()` input transpose**
- Plan A's burst DMA requires `x [H, T]` (H-major, T contiguous). For T-major input
  `[T, H]` the innermost DMA stride would be `H=2048` (gather), losing all burst benefit.
- The `.T.contiguous()` creates a 2.5 MB HBM copy (~5.6 µs at 900 GB/s), which is the
  unavoidable price of burst DMA in the router.
- No change. Accept this cost.

**A6. [Advanced] Inline `find_nonzero_indices` + `indexed_flatten` into the router kernel**

The `find_nonzero_indices` kernel (find_nonzero_indices.py:8) uses a hardware ISA
instruction `NonzeroWithCount` accessed via `neuronxcc.nki._private.private_api.inline_asm_bytes`
on the GPSIMD engine. This is not a software loop — it is a dedicated hardware stream-compaction
instruction that finds non-zero indices for 8 experts in parallel per round.

The router already has `mask_sb [T_TILE, E]` in SBUF — a binary matrix that encodes
exactly what `find_nonzero_indices` computes by reading `ea_out [T, E]` back from HBM.
The router could apply `NonzeroWithCount` to `mask_sb` directly, tile by tile, accumulating
per-expert token indices in SBUF without any HBM read. It could then write
`token_position_to_id [N*B]` via DMA copies (the same pattern `indexed_flatten` uses),
eliminating both intermediate kernels.

**What is eliminated:**

| Removed step | Bandwidth saved |
|---|---|
| `find_nonzero_indices` reads `ea_out [T, E]` | −320 KB read |
| `find_nonzero_indices` writes `indices [E_kernel, T]` | −80 KB write |
| `indexed_flatten` reads `indices [E_kernel, T]` | −80 KB read |
| Two kernel launch overheads | −2–4 µs |

`token_position_to_id [N*B]` (83 KB write) and `block_to_expert [N]` (1 KB) still need
to be written to HBM — they are required by the expert kernel. Net bandwidth saving: **480 KB**
(~0.5 µs at 900 GB/s) plus kernel launch overheads.

**What the router kernel would need to implement:**

1. Per T-tile: rearrange `mask_sb [T_TILE, E]` into GPSIMD-aligned format
   (`[128, 1, T_TILE]` with experts at partitions 0, 16, 32, ..., 112) — same layout
   that `find_nonzero_indices` constructs from `ea_out`.
2. Call `nki_asm_nonzero_with_count(input_local, index_offset=T_offset + t_tile*T_TILE)`
   — returns per-expert token indices and counts for this tile.
3. Accumulate `nonzero_counts [E]` and per-tile index lists across T-tiles in SBUF.
   SBUF budget: `num_t_tiles × E × K × 4B = 3 × 128 × 8 × 4 = 12 KB` — trivial.
4. After all tiles: compute prefix sums for `blocks_per_expert [E]` and
   `block_offsets [E]` (128 scalar ops).
5. LNC cross-core sync: each core writes its `nonzero_counts [E/2]` to shared HBM (512 B),
   `core_barrier`, reads the other core's counts to form `total_nonzero_counts [E]`, then
   adjusts each core's write offsets so Core 0's tokens precede Core 1's within each
   expert's block range.
6. DMA-copy token indices to `token_position_to_id [N*B]` per expert (same loop as
   `indexed_flatten`, using offsets from step 5).
7. Write `block_to_expert [N]` from `blocks_per_expert` (small loop over E).

**Gating concern:** `NonzeroWithCount` is accessed through
`neuronxcc.nki._private.private_api.inline_asm_bytes` — a private API. This is the same
function used by the library's own `find_nonzero_indices`. It is stable enough for the
library to ship, but carries no public API contract. Raise with the Neuron team as a use
case for a public `nisa.nonzero_with_count` primitive before implementing.

**Implementation order:** implement A1–A5 and B1–B3 first. Measure the residual
mapping step cost (kernel launch + bandwidth for `find_nonzero_indices` + `indexed_flatten`)
before investing in A6.

---

### B. Integration Changes (`NKIRouterTopK.forward()` + model call site)

**B1. Bypass `get_expert_mask` + `get_expert_affinities_masked` — highest-impact change**

`ExpertMLPsV2.maybe_get_expert_affinities_masked()` (expert_mlps_v2.py:676) has an
early-exit path:
```python
def maybe_get_expert_affinities_masked(self, expert_index, expert_affinities,
                                        expert_affinities_masked_full=None, ...):
    if expert_affinities_masked_full is not None:
        return expert_affinities_masked_full   # ← direct passthrough, no mask rebuild
    expert_mask = self.get_expert_mask(expert_index, ...)   # ← O(T*E) XLA op
    expert_affinities_masked = self.get_expert_affinities_masked(...)  # ← another O(T*E) op
    return expert_affinities_masked
```

Our router's `ea_out` IS already `expert_affinities_masked` (masked + L1-normalized).
Passing it as `expert_affinities_masked_full` skips both XLA ops entirely.

The fix is in `MoE._forward_compute_bound()` (model.py:154). Currently it sets
`expert_affinities_masked_full = None` for the non-SP case (line 172), even though our
router has already done the masking. Override this in the NKI model class:

```python
# In NeuronQwen3MoeDecoderLayerWithNKIAttn or the MoE subclass:
# After: router_logits, expert_affinities, expert_index = self.router(hidden_states)
# Add:
expert_affinities_masked_full = expert_affinities   # ea_out is already masked+normalized
# Then pass to: self.expert_mlps(..., expert_affinities_masked_full=expert_affinities_masked_full)
```

The cleanest way to do this without forking `model.py` is to subclass `MoE` and override
`_forward_compute_bound` to set `expert_affinities_masked_full = expert_affinities` before
the `self.expert_mlps(...)` call.

Eliminated ops:
- `torch.zeros([T, E])` allocation (640×128×8B = 640 KB)
- `expert_mask += (expert_index[:, e].unsqueeze(1) == expert_num_idx_arr)` × K iterations
- `expert_affinities.masked_fill(...)` over [T, E]

**B2. Verify the `expert_affinities_masked_full` propagation chain**

`MoE._forward_compute_bound()` passes `expert_affinities_masked_full` to
`ExpertMLPsV2.__call__()` → `ExpertMLPsV2.forward()` (line 1405) → `forward_blockwise()`
(line 1495). Trace this chain once with a print to confirm the value is not reset to None
anywhere along the path.

**B3. Ensure `ea_out` is returned as the second element from `NKIRouterTopK.forward()`**

The library's `_forward_compute_bound` receives the router outputs as
`(router_logits, expert_affinities, expert_index)`. For B1 to work, `expert_affinities`
(second return value) must be `ea_out` — the already-masked tensor. This is already true
in the current implementation. Confirm that no code between the router call and the
`expert_mlps(...)` call modifies or replaces `expert_affinities`.

---

### C. Config Flags

**C1. `use_index_calc_kernel=True` — required for NKI mapping path**

`use_index_calc_kernel` in `RoutedExpertsMlpConfig` defaults to `True` but the
`ExpertMLPsV2.use_index_calc_kernel()` guard (line 628) additionally requires:
- `self.is_prefill == True` — for CTE this is always True; verify the model sets this
- `enable_spmd_rank == True` — must be set in `RoutedExpertsMlpConfig`
- `logical_nc_config == 2` — set in `BlockwiseMatmulConfig`
- `T % block_size == 0` — enforced by A4
- `E_local % 2 == 0` — E=128 per EP rank, 128%2=0 ✓

If `enable_spmd_rank=False`, the function returns `False` and the PyTorch fallback runs.
This is a silent performance regression. Add a runtime check after model init:
```python
assert model.mlp.expert_mlps.use_index_calc_kernel(T=expected_T), \
    "Index calc kernel path not active — check enable_spmd_rank and is_prefill"
```

**C2. `BlockwiseMatmulConfig` settings**

```python
BlockwiseMatmulConfig.from_kwargs(
    block_size=128,                     # matches T_TILE=128 in router kernel
    logical_nc_config=2,                # selects shard_hidden kernel variant (LNC=2)
    normalize_top_k_affinities=True,    # ea_out is already L1-normalized — no-op renorm
    skip_dma_token=True,                # tokens fetched from hidden_states, not copied
    skip_dma_weight=True,               # weights are preloaded
    num_static_blocks=T * K // 128,     # exact for CTE fixed-T: T*8/128 = T//16
)
```

`num_static_blocks` matters for static compilation. If T is fixed (CTE with padded
sequences), set this explicitly to avoid worst-case over-allocation.

**C3. `normalize_top_k_affinities=True` in `RoutedExpertsMlpConfig`**

This flag controls the behavior of `get_expert_affinities_masked()`. With B1 in place,
this code path is bypassed. But if the bypass ever fails (e.g., during fallback), the
flag prevents double-normalization of the already-L1-normalized `ea_out`. Set it in both
`BlockwiseMatmulConfig` AND `RoutedExpertsMlpConfig` for defense in depth.

---

### D. Correctness Verification

**D1. Verify `find_nonzero_indices` non-zero threshold handles L1-normalized values**

Our `ea_out` has exactly K=8 non-zeros per token row with values ≈ 1/K = 0.125. The
remaining 120 values are exactly 0.0 (memset, not near-zero). `find_nonzero_indices`
checks for non-zero in f32 — a value of 0.125 is unambiguously above any threshold.
No risk here, but add a unit test that verifies K non-zeros per row in `ea_out`.

**D2. Verify `maybe_get_expert_affinities_masked` bypass semantics**

With `expert_affinities_masked_full = ea_out` (f32, [T, E]):
- `maybe_get_expert_affinities_masked` returns `ea_out` directly (line 677-679)
- `find_nonzero_indices` receives `ea_out.to(f32)` — no-op cast since it's already f32
- `BlockwiseMatmulNKIFunc` receives `ea_out` as `expert_affinities_masked`

Check that `expert_affinities_masked` arrives at `BlockwiseMatmulArgs` with:
- dtype: float32 (or bf16 if kernel supports it — verify)
- shape: [T, E] — correct
- values: L1-normalized top-K scattered, zeros at non-selected positions — correct

**D3. Hard assert that `use_index_calc_kernel` path is taken (gating for A2)**

If `ei_out` write is dropped (A2), the PyTorch mapping fallback (`get_blockwise_expert_and_token_mapping`) would receive `expert_index=None` and crash. This is the correct behavior — fail loudly rather than silently produce wrong output. Add:
```python
# In NKIRouterTopK.forward(), after the router call:
if not write_expert_index:
    expert_index = None   # intentionally poison — will crash if PyTorch fallback runs
```

**D4. Validate `core_barrier` placement with the bypass**

`core_barrier(ea_out, cores=[0, 1])` in the router kernel synchronizes both LNC cores
before the caller reads `ea_out`. This barrier is outside the T-tile loop and runs once
per kernel call. The bypass (B1) passes `ea_out` directly to `expert_affinities_masked_full`,
which is then read by `find_nonzero_indices`. The barrier ensures the read is safe.
No change needed; document this dependency explicitly.

**D5. Check `is_prefill` flag for the CTE model variant**

`ExpertMLPsV2.use_index_calc_kernel()` checks `self.is_prefill`. Verify that the
`NeuronQwen3MoeDecoderLayerWithNKIAttn` model initializes its `ExpertMLPsV2` with
`is_prefill=True` for the CTE (context encoding) model instance.

---

## Optimized Pipelines

### Phase 1: A1–A5 + B1–B3 + C1–C3 (near-term, no private API)

```
hidden_states [T, H]  bf16
       |
       | .T.contiguous()   2.5 MB copy — unavoidable for burst DMA
       ↓
NKI Router kernel
  WRITE ea_out [T, E]  f32   320 KB   ← only output (rl_out and ei_out dropped)
       |
       | expert_affinities_masked_full = ea_out  ← bypass (B1), no mask rebuild
       |
       | find_nonzero_indices(ea_out, row_start_id=..., n_rows=E_kernel)
       |   READ  ea_out              [T, E]        320 KB
       |   WRITE indices             [E_kernel, T]  80 KB
       |
       | indexed_flatten(indices, row_offsets)
       |   READ  indices             [E_kernel, T]  80 KB
       |   WRITE token_position_to_id [N*B]          83 KB
       |
       | block_to_expert ← tiny XLA op from nonzero_counts
       ↓
Expert kernel
```

Eliminated vs. baseline:
- `rl_out` WRITE: −320 KB
- `ei_out` WRITE: −20 KB
- `get_expert_mask` + `get_expert_affinities_masked` XLA ops: eliminated
- Remaining mapping step HBM: 320 (read) + 80 (write) + 80 (read) + 83 (write) = **563 KB**

### Phase 2: + A6 (inline mapping, requires private API)

```
hidden_states [T, H]  bf16
       |
       | .T.contiguous()   2.5 MB copy — unavoidable for burst DMA
       ↓
NKI Router kernel  (extended)
  WRITE ea_out               [T, E]    320 KB  ← still needed by expert kernel
  WRITE token_position_to_id [N*B]      83 KB  ← computed inline via NonzeroWithCount
  WRITE block_to_expert      [N]         1 KB  ← computed inline from prefix sums
  (indices [E_kernel, T] intermediate never touches HBM)
       |
       | expert_affinities_masked_full = ea_out  ← bypass (B1)
       | find_nonzero_indices: SKIPPED
       | indexed_flatten: SKIPPED
       ↓
Expert kernel
```

Additional eliminated vs. Phase 1:
- `find_nonzero_indices` READ of `ea_out`: −320 KB
- `indices [E_kernel, T]` WRITE: −80 KB
- `indexed_flatten` READ of `indices`: −80 KB
- Two kernel launch overheads: −2–4 µs
- Remaining mapping HBM: **84 KB** (token_position_to_id + block_to_expert writes only)

---

## HBM Bandwidth Summary

| Step | Baseline | Phase 1 | Phase 2 (A6) |
|------|----------|---------|--------------|
| `rl_out` write | 320 KB | — | — |
| `ei_out` write | 20 KB | — | — |
| `ea_out` write | 320 KB | 320 KB | 320 KB |
| `find_nonzero_indices` reads `ea_out` | 320 KB | 320 KB | — |
| `indices [E_kernel,T]` write | 80 KB | 80 KB | — |
| `indexed_flatten` reads `indices` | 80 KB | 80 KB | — |
| `token_position_to_id` write | 83 KB | 83 KB | 83 KB |
| **Total metadata** | **1223 KB** | **883 KB** | **403 KB** |

Weight loads (expert kernel, dominant cost) are identical across all phases.

---

## File References

| File | Location |
|------|----------|
| `kernels/router_topk/qwen3_router_topk_plan_a.py` | `rl_out` alloc line 128; `ei_out` alloc line 130; `mask_sb` (A6 source) line 330; `core_barrier` line 379 |
| `qwen_with_router_nki.py` | `NKIRouterTopK.forward()` — transpose line, return values |
| `neuronx_distributed/modules/moe/model.py` | `_forward_compute_bound()` line 154; `expert_affinities_masked_full` line 172; `expert_mlps()` call line 209 |
| `neuronx_distributed/modules/moe/expert_mlps_v2.py` | `maybe_get_expert_affinities_masked()` line 676; `use_index_calc_kernel()` line 628; `get_blockwise_expert_and_token_mapping_kernel()` line 1080 |
| `neuronx_distributed/modules/moe/moe_configs.py` | `RoutedExpertsMlpConfig.use_index_calc_kernel` line 181; `enable_spmd_rank` |
| `neuronx_distributed/kernels/find_nonzero_indices.py` | `nki_asm_nonzero_with_count()` line 8; `NonzeroWithCount` ISA via `inline_asm_bytes` line 41; GPSIMD alignment logic lines 138–153 |
| `neuronx_distributed/kernels/indexed_flatten.py` | per-expert DMA copy loop lines 82–91; `sendrecv` cross-core merge lines 110–117 |
