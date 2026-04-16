# Qwen3 MoE Multi-Layer TKG Megakernel — Implementation Plan

Target: **Trainium3 (trn3)**, LNC=2. Extends the current single-layer fused TKG kernel into a multi-layer megakernel that executes all decoder layers in a single NKI kernel invocation, eliminating Python/NxDI round-trips between layers.

## Repo context

- Single-layer kernel: `kernels/transformer/transformer_qwen.py:40` — `transformer_qwen3_moe_tkg_v2` (v13bc attn → AR → MoE → SB2SB AR-gather → HBM store).
- Attention primitive: `kernels/attn_tkg/agents/v13bc_sbm_tiled.py:23` — reads `K_cache`/`V_cache` (lines 334–335, 386, 437) but **does not write back**; returns `k_rope_out`/`v_out` to `shared_hbm` (lines 82–83).
- NxDI layer: `qwen_fused_transformer.py:298` — per-layer TKG call; KV-update TODO at ~line 362.
- Reference megakernel: `nkilib/experimental/transformer/transformer_tkg.py:91` — `transformer_tkg`. Python-unrolled loop `for layer_idx in range(num_layers)` at `:204`; SBUF-residual path at `:220-323`.
- In-kernel KV update primitive: `nkilib/experimental/transformer/attention_block_tkg.py:119-120` — `update_cache`, `kv_cache_update_idx`; in-kernel `update_kv_cache(...)` at `:305`.
- NxDI flags:
  - `neuronx_distributed_inference/models/config.py:469` — `attn_block_tkg_nki_kernel_cache_update`.
  - `modules/attention/attention_base.py:237, :1269, :1787-1793` — skips `kv_mgr.update_kv_by_layer_id` when flag is on.
  - `models/model_base.py:1299-1308` — `update_kv_per_layer` gating; `:1343` per-layer loop we bypass; `:1407-1416` trailing `kv_mgr.update_cache` to skip.
- KV manager: `modules/kvcache/kv_cache_manager.py:152` — flat `nn.ParameterList [k0,v0,k1,v1,...]`; `_fetch_cache` at `:238`.

---

## 1. Kernel side — `kernels/transformer/transformer_qwen.py`

### 1.1 New signature (conceptual)

- `X: [B,1,H]` HBM (raw; **`input_layernorm` moves inside the kernel** — see §1.4).
- Per-layer lists (`L = num_hidden_layers`):
  - `Wq_list, Wk_list, Wv_list, Wo_list`
  - `q_norm_list, k_norm_list`
  - **`gamma_pre_attn_list`** — new — `input_layernorm.weight` per layer (**pre-attention RMSNorm**)
  - `gamma_post_attn_list` — existing `gamma_moe` (post-attention / pre-MoE RMSNorm)
  - `router_w_list, gate_up_w_list, down_w_list`
- `K_caches: List[nl.ndarray]`, `V_caches: List[nl.ndarray]` — each `[B, Hkv_tp, S_max, d]`; mutated in place.
- Shared: `cos, sin` pre-indexed at `position_ids` to `[B,d]` once (keeps v13bc signature unchanged); `position_ids:[B,1]` int32 — also serves as `kv_cache_update_idx`; `num_layers:int` (compile-time); `replica_groups`.
- Return: `Y:[B,1,H]`. No KV return (in-place).

Note: these kernels are TKG-specialized — the cache is **always** updated in-kernel. No `update_cache` toggle; the scatter is unconditional.

### 1.2 Loop form — recommend Python-unroll (matches reference)

- **(A) Python `for layer_idx in range(num_layers)`** — reference approach (`transformer_tkg.py:204`). Best perf; best CC/compute overlap across layer boundaries; best BufferManager scope semantics. Cost: compile time and NEFF grow with `L`.
- **(B) `nl.sequential_range`** — smaller NEFF but loses cross-boundary scheduling; hard with SBUF-live residual and MoE dynamic routing. Reject.
- **(C) Chunked (e.g. 4 layers/call, 12 calls)** — fallback only if (A) blows compile time or NEFF size.

**Recommendation: (A)**; keep (C) as the escape hatch (§3).

### 1.3 In-place KV cache write

**What the cache actually is.** `K_caches[i]` and `V_caches[i]` are pre-allocated HBM tensors owned by NxDI's `KVCacheManager` (`modules/kvcache/kv_cache_manager.py:152`) — one pair per layer, shape `[B, S_max, d_head]`, sized to the max sequence length and held as `nn.Parameter`s for the whole session. At each TKG step the kernel writes exactly **one row per batch** into each cache: the freshly computed K/V for the new token lands at `K_cache[b, position_ids[b], :]` (and same for V). It's a single vector-indirect DMA (`nisa.dma_copy` with `vector_offset=position_ids, indirect_dim=0`) that scatters all batches in parallel; the DMA engine multiplies each index by the row stride automatically. Reference implementation at `attention_block_tkg.py:1284-1345`.

**v13bc modification required.** v13bc today only *reads* the caches and returns new K/V to `shared_hbm` for the caller to deal with. We need it to do what the reference does: scatter the computed `k_rope` and `v_new` rows into `K_cache_2d[position_ids]` / `V_cache_2d[position_ids]` via indirect DMA **before** the flash-decode loop so the read sees the newly written slot. Since these kernels are TKG-specialized, the scatter is unconditional — no toggle. The `k_rope_out`/`v_out` shared_hbm outputs can be deleted.

Confirm v13bc's `S_prior % PMAX == 0` assertion (line 71) still holds under the flag-on KV layout — `attention_base.py:1756-1766` pads S by 128 when `attn_block_tkg_nki_kernel_cache_update=True`.

### 1.4 SBUF-resident residual across boundaries — and why pre-attention RMSNorm must live in-kernel

Mirror `sbuf_residual_and_cc=True` (`transformer_tkg.py:220-323`). Per layer:

1. **RMSNorm(`residual_sb`, `gamma_pre_attn_list[i]`) → `attn_in_sb`** (pre-attention layernorm, in-kernel).
2. Call attention → SBUF `out_sb`.
3. SB2SB AR-gather; `residual_sb += attn_gathered`.
4. RMSNorm(`residual_sb`, `gamma_post_attn_list[i]`) → MoE in.
5. MoE SBUF→SBUF.
6. SB2SB AR-gather; `residual_sb += moe_gathered`.
7. After the last layer, apply final `model.norm` inside the kernel and store `Y` once to HBM (keeps lm_head input ready).

**Why pre-attention layernorm must be in-kernel.** Once the residual is kept in SBUF across layer boundaries, the post-AR residual of layer `i` is simultaneously:

- the *residual addend* for layer `i+1`, and
- the *input that must be RMSNormed* before layer `i+1`'s attention.

If pre-attention RMSNorm stays in Python (as it is today at the NxDI layer level), we must spill the residual to HBM between every pair of layers to run the norm, then reload — defeating the entire point of the fusion. So `gamma_pre_attn_list` is a new required kernel input, and the kernel runs RMSNorm at step (1) of every iteration. This matches the reference's `rmsnorm_X_enabled=True` semantics.

Existing single-layer kernel already runs this pattern per layer (`transformer_qwen.py:91-208`) with `residual_attn_sb`/`residual_moe_sb`. Multi-layer version just **elides the HBM store** between layers.

**SBUF budget (the big risk).** `SBM_SIZE_BYTES = 200 KiB` (`transformer_qwen.py:37`) is the BufferManager heap, not total SBUF. MoE peak allocs are materially heavier than the reference's dense MLP. Mitigations:

- Reuse `sbm.set_name_prefix(f"L{i}_")` + scope-close pattern at each boundary (lines 151, 185) so only `residual_sb` (~4 KiB) persists.
- Attention Wo SBUF hoist (~4 MiB) must release at layer end.
- Budget must be verified against trn3 per-LNC SBUF in a 2-layer profile before committing to full unroll.

### 1.5 Two AllReduces per layer with SBUF residual

Current kernel uses bare `nccl.all_reduce` post-attention (line 145) and `_sb2sb_all_reduce_gather` post-MoE (line 195); reference uses SB2SB for both. The bare-AR path was a workaround for an `ENC_ALG_MESH` issue on attention. **Re-test on trn3** — likely unneeded. If clean, switch both to `_sb2sb_all_reduce_gather` for symmetric tiled output.

### 1.6 trn3 notes

Target hardware is trn3 per CLAUDE.md; LNC=2 retained. Larger per-LNC SBUF on trn3 is what makes multi-layer MoE tractable; still verify. Confirm `nisa.sendrecv` semantics under `_sb2sb_all_reduce_gather` unchanged.

---

## 2. NxDI integration — `qwen_fused_transformer.py`

### 2.1 Bypass the per-layer loop

Subclass `NeuronQwen3MoeModelV2` (`qwen_fused_transformer.py:401`) and override `forward`, replacing `model_base.py:1343-1405` with a single megakernel call on the TKG branch. Keep pre-loop (embed, position_ids, mask, cache fetch) and post-loop (lm_head) logic. **Skip** the trailing `kv_mgr.update_cache` at `:1407-1416` (kernel already wrote).

### 2.2 Module structure — recommend (M-B) keep per-layer children

- **(M-A)** Flat `FusedLayersModule` ParameterList — big state-dict churn, loses CTE reuse. **Reject.**
- **(M-B)** Keep `nn.ModuleList` of `NeuronQwen3MoeDecoderLayerV2`; wrapping forward walks `self.layers[i]` and gathers **views** (Wq, Wk, Wv, Wo_nki, q_layernorm, k_layernorm, **input_layernorm**, post_attention_layernorm, router, gate_up_proj, down_proj) into Python lists, then calls `transformer_qwen3_moe_tkg_multilayer_jit[2]`. **Recommend.**

Router transpose: current code does `router_w.T.contiguous()` per step (line 343). At L=48 that's 48 tiny transposes per token — move into the state-dict converter once.

### 2.3 State-dict converter

`convert_qwen3_moe_hf_to_neuron_state_dict` (`:136-207`) is already per-layer — no renames needed with (M-B). Only change: pre-transpose `linear_router.weight` to `[H,E]` at conversion time.

### 2.4 KV plumbing

Default NxDI flow: `kv_mgr.get_cache` → per-layer attention returns new K/V → `attention_base` calls `kv_mgr.update_kv_by_layer_id` (Python-side scatter) → trailing `kv_mgr.update_cache` at `model_base.py:1407`. That's 48 Python round-trips per step — what we're replacing.

With in-kernel updates, set `neuron_config.attn_block_tkg_nki_kernel_cache_update = True`. This trips two gates:
- `attention_base.py:1787-1793` skips `update_kv_by_layer_id` (kernel already wrote).
- `model_base.py:1299-1308` sets `update_kv_per_layer=True`, routing fetches via `_fetch_cache` and skipping the trailing `update_cache`.

In the overridden forward: gather `K_caches = [pk[0] for pk in past_key_values]`, `V_caches = [pk[1] for pk in past_key_values]` and pass both lists plus `position_ids` to the kernel. Kernel mutates the tensors in place; NxDI trusts the mutation. Build `next_decoder_cache` for `model_base.py:642` from the same tensor handles.

### 2.5 CTE gating

Megakernel is **TKG-only**. Inside the overridden `forward`, branch on `is_for_context_encoding`: True → stock per-layer path (unchanged); False → megakernel. Also gate at compile time on `compile_tag == TOKEN_GENERATION_MODEL_TAG` (`qwen_fused_transformer.py:464`) so CTE builds don't pull the list-signature kernel.

### 2.6 Compile-time

- Static `L=48` unroll → NEFF specialized per `num_hidden_layers`; acceptable for a single production build.
- Expect compile 20–60 min at `-O3`. Use `-O2` during iteration; promote for final.
- `--cc-pipeline-tiling-factor=1` (current TKG) likely suboptimal at L=48 — try `=2`/`=4` to overlap layer `i+1` attention AR with layer `i` MoE. (Per CLAUDE.md, don't remove flags unilaterally — change values, ask before dropping.)
- Watch NEFF size; fall back to chunked (C) if > ~2 GB.

---

## 3. Risks & open questions

1. **SBUF pressure** across 48 layers with MoE-peak allocs + attention Wo hoist + residual. Profile 2-layer high-water first.
2. **v13bc in-kernel KV write correctness** — new path in a shared primitive; golden vs Python ref.
3. **Compile time / NEFF size** at L=48 — chunked fallback ready.
4. **DMA congestion**: 48 layers × 2 ARs + weight DMA; CC-pipeline-tiling tuning is load-bearing.
5. **Register/PSUM pressure** from full unroll; watch spill warnings.
6. **ENC_ALG_MESH workaround** on bare AR — retest on trn3.
7. **KV shape under flag-on path** — `attention_base.py:1756-1766` pads S by 128; verify vs v13bc `S_prior % PMAX == 0`.
8. **Sliding window / block KV layout** (`model_base.py:1317, :1754`) — confirm both remain false for Qwen3-MoE.

Measurements (neuron-profile on NTFF): SBUF high-water per boundary; AR-to-compute overlap; end-to-end TKG latency vs baseline; PSUM occupancy; DMA engine split.

---

## 4. Suggested implementation order

- **M1.** Add `update_cache`/`kv_cache_update_idx` to v13bc with indirect-DMA write pre-flash-decode. Validate single-layer kernel with golden KV state.
- **M2.** Resolve TKG KV TODO (`qwen_fused_transformer.py:362`): set the flag, return in-place tensors, end-to-end logit check T=16 vs HF. Validates NxDI plumbing in isolation.
- **M3.** 2-layer scaffold copy-pasted from `transformer_tkg.py` `sbuf_residual_and_cc=True`, using **reference** `attention_block_tkg` + **dense** MLP. Must compile on trn3. Pre-attention RMSNorm lives inside the kernel from this milestone onward.
- **M4.** Swap attention → v13bc (post-M1). Smoke-test vs two sequential single-layer calls.
- **M5.** Swap MLP → `_qwen3_moe_sbuf_in_sbuf_out`. Profile SBUF peak.
- **M6.** Scale unroll to `L=48`; promote `-O2 → -O3`; tune `--cc-pipeline-tiling-factor`.
- **M7.** NxDI integration via (M-B) `forward` override; router pre-transpose in converter; CTE gate.
- **M8.** End-to-end logit/perplexity match vs HF Qwen3-30B-A3B; compare latency to baseline.
- **M9.** Robustness: bucketing, B>1, speculative decoding interactions.

### Critical files

- `/home/ubuntu/nki-moe/kernels/transformer/transformer_qwen.py`
- `/home/ubuntu/nki-moe/kernels/attn_tkg/agents/v13bc_sbm_tiled.py`
- `/home/ubuntu/nki-moe/qwen_fused_transformer.py`
- `/opt/.../nkilib/experimental/transformer/transformer_tkg.py`
- `/opt/.../nkilib/experimental/transformer/attention_block_tkg.py`
- `/opt/.../neuronx_distributed_inference/models/model_base.py`
- `/opt/.../neuronx_distributed_inference/modules/attention/attention_base.py`
- `/opt/.../neuronx_distributed_inference/modules/kvcache/kv_cache_manager.py`
