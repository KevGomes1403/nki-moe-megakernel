"""Strict-tolerance MoE correctness test for the speculative megakernel.

Compares the NKI MoE block (as called by
``transformer_qwen3_moe_speculative._multilayer_body`` for layer 0) against
a PyTorch reference equivalent to HF's ``Qwen3MoeSparseMoeBlock.forward``.

Why a dedicated MoE-only kernel
-------------------------------
Per-component testing of the megakernel:
- ``test_attention_vs_hf.py`` covers the attention block.
- this file covers the MoE block (post_attention_layernorm + router_topk + moe_tkg).

The kernel under test (``_moe_only_kernel.py``) is a strict subset of the
production megakernel — it reuses the SAME helpers and constants. Any bug
in the MoE call site (X load layout, post-attn RMSNorm, router layout,
gate_up reshape, AR-gather) is reproduced here in isolation.

If this passes with strict tolerances and ``test_attention_vs_hf.py`` also
passes, the per-layer pipeline (attention + MoE + residuals) is correct,
and any remaining end-to-end issue is in multilayer orchestration
(KV-cache reuse across layers, residual carry, etc.).

TP=1 vs TP=4 weights
--------------------
The production kernel runs at TP=4 with ``I_PER_EXPERT = 768 / 4 = 192``.
For unit-testing on a single device, we use TP=4-shape weights
(``[E, H, 2*192]`` for gate_up, ``[E, 192, H]`` for down) and treat them as
the FULL MoE intermediate dim. The PyTorch reference uses the same shapes —
this is mathematically equivalent to a TP=1 model with a smaller
``intermediate_size = 192`` (no all-reduce involved at the test boundary
beyond the simulator's single-rank copy patch).

Tolerances
----------
Strict bf16 chain through softmax-router → topK norm → silu/up → down →
expert affinity scaling → sum over top-K experts. We aim for
``atol = rtol = 1e-2`` — the MoE chain has more rounding hops than
attention, so a slightly looser target than the attention test's 5e-3 is
realistic. If the kernel passes at 5e-3 we leave the constants tight; if
not, document and bump after investigation, NOT before.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import nki

# Reuse the simulator patch that lets us run kernels with collectives in a
# single-process unit test. The attention test installs this patch on
# import; we just need to make sure that module is imported before any
# call to ``nki.simulate``. Importing it here is enough.
import test_attention_vs_hf  # noqa: F401 — installs simulator all_reduce patch

from _moe_only_kernel import (
    moe_only_kernel,
)
from megakernels.qwen3_moe.transformer_qwen3_moe_speculative import (
    H, E, TOP_K, I_PER_EXPERT,
)


# --------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------

ATOL_STRICT = 1e-2
RTOL_STRICT = 1e-2


# --------------------------------------------------------------------------
# Config & fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return {
        "B":     1,
        "S_tkg": 1,
        "dtype": torch.bfloat16,
        "seed":  0xDECAFBAD,
    }


def _seeded(g, shape, dtype, scale=1.0):
    return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(dtype)


@pytest.fixture(scope="module")
def inputs(cfg):
    g = torch.Generator().manual_seed(cfg["seed"])
    B, S_tkg = cfg["B"], cfg["S_tkg"]
    dtype = cfg["dtype"]
    I = I_PER_EXPERT

    # Input: post-attention residual, small magnitude (typical scale).
    X = _seeded(g, (B, S_tkg, H), dtype, scale=0.05)

    # post_attention_layernorm gamma. Shape [1, H] (matches the kernel's
    # ``gpost_list[i]`` shape — see qwen_with_megakernel.py:413 which uses
    # ``.unsqueeze(0)`` to produce [1, H]).
    gpost = (torch.ones(1, H, dtype=dtype) + _seeded(g, (1, H), dtype, scale=0.01))

    # Router weight: NKI layout is [H, E] (kernel's ``router_list[i]`` is
    # the transpose of HF gate.weight which is [E, H]).
    router_w = _seeded(g, (H, E), dtype, scale=0.02)

    # Gate-up weight in NxDI ColumnParallelLinear stride=2 layout
    # [E, H, 2*I_per_rank]. The first I_per_rank along the last dim is the
    # gate slice, the next I_per_rank is the up slice (per stride=2 convention
    # used by the converter in qwen_with_megakernel.py:240-262).
    gate_up_w = _seeded(g, (E, H, 2 * I), dtype, scale=0.02)

    # Down weight [E, I_per_rank, H].
    down_w = _seeded(g, (E, I, H), dtype, scale=0.02)

    return {
        "X": X, "gpost": gpost,
        "router_w": router_w, "gate_up_w": gate_up_w, "down_w": down_w,
    }


# --------------------------------------------------------------------------
# PyTorch reference (matches HF Qwen3MoeSparseMoeBlock with
# norm_topk_prob=True, SiLU activation, no expert clamping or biases)
# --------------------------------------------------------------------------

def _rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    return (x32 * gamma.to(torch.float32)).to(in_dtype)


def _moe_reference(
    X:         torch.Tensor,  # [B, S_tkg, H]
    gpost:     torch.Tensor,  # [1, H]
    router_w:  torch.Tensor,  # [H, E]
    gate_up_w: torch.Tensor,  # [E, H, 2*I_per_rank]   (stride=2: first half gate, second half up)
    down_w:    torch.Tensor,  # [E, I_per_rank, H]
    eps: float = 1e-6,
):
    """Reference implementation of: post_attention_layernorm + router_topk +
    selective expert MoE (Qwen3 conventions: SiLU, norm_topk_prob=True,
    POST_SCALE affinities)."""
    B, S_tkg, _H = X.shape
    Hk = H
    I = I_PER_EXPERT
    K = TOP_K
    n_experts = E

    # post_attention_layernorm
    x_norm = _rmsnorm(X, gpost.squeeze(0), eps)   # [B, S_tkg, H]
    x_flat = x_norm.reshape(B * S_tkg, Hk)        # [T, H]

    # Router: softmax then topK then L1-normalize (Qwen3 + norm_topk_prob).
    # HF order: topk-on-softmax-probabilities, then renormalize. See
    # transformers/models/qwen3_moe/modeling_qwen3_moe.py:226-238.
    router_logits = x_flat.to(torch.float32) @ router_w.to(torch.float32)   # [T, E]
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_indices = torch.topk(routing_weights, K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(X.dtype)   # back to bf16 — matches kernel

    # Per-expert FFN: y_e = down_e( silu(x_norm @ gate_e^T) * (x_norm @ up_e^T) )
    # Then sum_t = sum_{k in topK_t} topk_weight_{t, k} * y_{expert_k}(x_t)
    # gate_up_w layout: last dim = 2*I, first I = gate (stride=0), next I = up
    # (per converter at qwen_with_megakernel.py:251-253).
    gate_proj = gate_up_w[..., :I]              # [E, H, I]  (gate)
    up_proj   = gate_up_w[..., I:]              # [E, H, I]  (up)

    T = B * S_tkg
    out = torch.zeros(T, Hk, dtype=torch.float32)

    for t in range(T):
        x_t = x_flat[t]                         # [H]
        for k in range(K):
            e_idx = topk_indices[t, k].item()
            w     = topk_weights[t, k].to(torch.float32)
            # Compute expert e_idx's MLP on x_t
            gate_e = gate_proj[e_idx]           # [H, I]
            up_e   = up_proj[e_idx]             # [H, I]
            down_e = down_w[e_idx]              # [I, H]
            gate_out = F.silu((x_t.to(torch.float32) @ gate_e.to(torch.float32)))
            up_out   = (x_t.to(torch.float32) @ up_e.to(torch.float32))
            inter    = gate_out * up_out        # [I]
            expert_out = inter @ down_e.to(torch.float32)   # [H]
            out[t] += w * expert_out

    return out.reshape(B, S_tkg, Hk).to(X.dtype)


# --------------------------------------------------------------------------
# Kernel launcher
# --------------------------------------------------------------------------

def _torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    import ml_dtypes
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return t.contiguous().numpy()


def _numpy_to_torch(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    import ml_dtypes
    if arr.dtype == ml_dtypes.bfloat16:
        u16 = arr.view(np.uint16)
        return torch.from_numpy(u16.copy()).view(torch.bfloat16)
    return torch.from_numpy(arr.copy()).to(dtype)


def _run_nki_kernel(inputs):
    np_inputs = {k: _torch_to_numpy(v) for k, v in inputs.items()}
    result = nki.simulate(moe_only_kernel[2])(
        np_inputs["X"], np_inputs["gpost"], np_inputs["router_w"],
        np_inputs["gate_up_w"], np_inputs["down_w"],
        replica_groups=([0],),
    )
    Y_np = result[0]
    Y = _numpy_to_torch(Y_np, inputs["X"].dtype)
    debug = {}
    debug_names = [
        ("dbg_moe_in",            inputs["X"].dtype),
        ("dbg_router_logits",     torch.float32),
        ("dbg_expert_index",      torch.uint8),  # special-cased below
        ("dbg_expert_affinities", torch.float32),
        ("dbg_moe_out_pershard",  inputs["X"].dtype),
    ]
    for i, (name, dt) in enumerate(debug_names):
        if 1 + i >= len(result):
            continue
        arr = result[1 + i]
        if name == "dbg_expert_index":
            # uint32 indices: pass through numpy then cast to torch.int64.
            import numpy as _np
            debug[name] = torch.from_numpy(arr.astype(_np.int64).copy())
        else:
            debug[name] = _numpy_to_torch(arr, dt)
    return Y, debug


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def _max_abs_err(a, b):
    return (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()


def _mean_abs_err(a, b):
    return (a.to(torch.float32) - b.to(torch.float32)).abs().mean().item()


def _reference_with_intermediates(X, gpost, router_w, gate_up_w, down_w,
                                   eps: float = 1e-6):
    """Same as ``_moe_reference`` but returns all intermediate tensors so
    the test can compare per-stage against the kernel dumps."""
    B, S_tkg, _H = X.shape
    Hk = H
    I = I_PER_EXPERT
    K = TOP_K
    n_experts = E

    x_norm = _rmsnorm(X, gpost.squeeze(0), eps)            # [B, S_tkg, H]
    x_flat = x_norm.reshape(B * S_tkg, Hk)                  # [T, H]

    router_logits = x_flat.to(torch.float32) @ router_w.to(torch.float32)  # [T, E]
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_indices = torch.topk(routing_weights, K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights_bf16 = topk_weights.to(X.dtype)

    gate_proj = gate_up_w[..., :I]
    up_proj   = gate_up_w[..., I:]

    T = B * S_tkg
    out = torch.zeros(T, Hk, dtype=torch.float32)
    for t in range(T):
        x_t = x_flat[t]
        for k in range(K):
            e_idx = topk_indices[t, k].item()
            w = topk_weights_bf16[t, k].to(torch.float32)
            gate_e = gate_proj[e_idx]
            up_e   = up_proj[e_idx]
            down_e = down_w[e_idx]
            gate_out = F.silu(x_t.to(torch.float32) @ gate_e.to(torch.float32))
            up_out   = x_t.to(torch.float32) @ up_e.to(torch.float32)
            inter    = gate_out * up_out
            expert_out = inter @ down_e.to(torch.float32)
            out[t] += w * expert_out
    Y = out.reshape(B, S_tkg, Hk).to(X.dtype)

    return {
        "x_norm":         x_norm,
        "x_flat":         x_flat,
        "router_logits":  router_logits,
        "routing_weights": routing_weights,
        "topk_weights":   topk_weights_bf16,
        "topk_indices":   topk_indices,
        "Y":              Y,
    }


def _reference_moe_pershard(x_norm: torch.Tensor, gate_up_w: torch.Tensor,
                             down_w: torch.Tensor, topk_indices: torch.Tensor,
                             topk_weights_bf16: torch.Tensor, shard: int):
    """Compute the per-shard MoE output that the kernel's shard `shard`
    produces in moe_out_sb[:, t, 0:H1_SHARD] (before AR-gather).

    Kernel layout per shard: input partition p at (t, h1=0..H1_SHARD-1) holds
    the post-RMSNorm value for hidden index ``shard*H0*H2 + p*H2 + h1``.
    The kernel processes the FULL H dimension (loads weight rows
    indexed by p*H2 + h1 with a per-shard H offset) and writes out the
    sum of expert contributions into the SAME H index pattern.

    Mathematically: when LNC=2 and shard_on_H is enabled, each shard
    computes the FULL gate/up matmul on its slice of H rows
    (H[shard*1024 : (shard+1)*1024]) and the FULL down matmul. The per-shard
    output is the contribution of THIS shard's H rows to the down result
    — i.e. the down-projection of the gate-up output restricted to this
    shard's intermediate slice.

    Wait — actually for MoE shard_on_H is on the *hidden* dim. The gate_up
    projection is `[H, I]` so sharding on H is sharding on the
    *contraction* axis — each shard computes a partial sum which is then
    summed via all-reduce. So per-shard output is the *partial down result*
    using only this shard's slice of H in the gate/up matmul **AND** this
    shard's slice of I in the down matmul.

    Look at the kernel: gate_up_projection uses `shard_dim_hidden` to slice
    the H rows of the gate/up weight (`hidden_input[H/2,I]` → matmul
    contracting on H/2). Down projection uses the same shard's I slice
    (since down_w is `[I, H]` and the intermediate from gate/up is sharded
    on I implicitly because each shard owns ITS H rows of gate/up which
    give the full I — wait, that's not right either).

    Actually: in MoE shard-on-H mode, the GATE/UP matmul on shard s is::

        gate_s[t,i] = sum_{h in shard_s_H_range} hidden[t,h] * gate[h,i]

    Sum across shards gives the FULL gate output. Then SiLU(gate_full)*up_full
    requires gate from BOTH shards (an all-reduce on I dim).

    Looking at the kernel structure — there is NO all-reduce between gate/up
    and down. So actually each shard must compute everything LOCALLY and the
    AR-gather at the very end stitches the per-shard outputs back together.

    Looking again at MLPTKGConstants and the projection: shard_dim_hidden
    is on H (the *contraction* dim for gate/up), but THE INPUT IS ALREADY
    SHARDED on H — each shard only HAS its slice of H. So the gate_up
    matmul on shard s gives a PARTIAL gate/up output (sum over only
    shard_s's H rows). Then SiLU is applied per-shard. Then down matmul
    on shard s uses its slice of I (from the partial gate/up) and the FULL
    H of down_w sliced on its I slice. The result is each shard's
    contribution to the H_out, but only on the shard's local H_out slice
    (H1_shard).

    OK — too complex to derive in head. Let me compute per-shard by running
    the kernel-equivalent math for shard s::
      1. take x_norm slices owned by shard s on H (H_s = x_norm[..., shard*1024:(shard+1)*1024])
      2. project: gate_s = x_norm @ gate_w[shard*1024:..., :]      (partial gate)
                 up_s   = x_norm @ up_w[shard*1024:..., :]
      3. inter_s = silu(gate_s) * up_s  -- WRONG: silu(gate) * up is non-linear,
                                            so silu(gate_s)*up_s != silu(gate)*up
      ...

    So the kernel CANNOT be shard-on-H without an all-reduce between gate/up
    and down. Either the kernel doesn't use shard-on-H (only shard on
    token), or there IS an all-reduce I missed, or shard_on_h_disabled is True.

    Conclusion: deriving the per-shard output requires reading the kernel
    impl carefully. For NOW: skip the per-shard reference computation and
    only validate (a) moe_in (post-RMSNorm), (b) router logits/topK,
    (c) final Y. If those tell the story, that's enough.
    """
    raise NotImplementedError("Per-shard reference deferred — kernel impl is non-trivial.")


def test_moe_matches_hf_reference_strict(inputs, cfg):
    """The NKI MoE block output must match the PyTorch reference within strict
    bf16 tolerances (atol=1e-2, rtol=1e-2)."""
    refs = _reference_with_intermediates(
        X=inputs["X"], gpost=inputs["gpost"],
        router_w=inputs["router_w"],
        gate_up_w=inputs["gate_up_w"], down_w=inputs["down_w"],
    )
    Y_ref = refs["Y"]
    Y_nki, debug = _run_nki_kernel(inputs)

    # ------------------------------------------------------------------
    # Stage-by-stage drift diagnostics
    # ------------------------------------------------------------------
    if "dbg_moe_in" in debug:
        moe_in_err = _max_abs_err(debug["dbg_moe_in"], refs["x_norm"])
        moe_in_mean = _mean_abs_err(debug["dbg_moe_in"], refs["x_norm"])
        print(f"[moe_in  vs x_norm]      max_abs_err = {moe_in_err:.6e}  "
              f"mean = {moe_in_mean:.6e}")

    if "dbg_router_logits" in debug:
        # router_logits stored as fp32 [T, E]
        rl_nki = debug["dbg_router_logits"]
        # The kernel's router_logits HBM is the post-softmax output (router_topk
        # writes softmax probabilities here when act_fn=SOFTMAX). The PyTorch
        # ref's `routing_weights` is the same quantity.
        rl_err = _max_abs_err(rl_nki, refs["routing_weights"])
        rl_mean = _mean_abs_err(rl_nki, refs["routing_weights"])
        # Also compare against raw pre-softmax logits (in case the kernel
        # stores pre-softmax instead of post-softmax — this would tell us).
        rl_pre_err = _max_abs_err(rl_nki, refs["router_logits"])
        print(f"[router_logits vs routing_weights] max_abs_err = {rl_err:.6e}  "
              f"mean = {rl_mean:.6e}")
        print(f"[router_logits vs raw router_logits] max_abs_err = {rl_pre_err:.6e}  "
              "(if this is ~0, kernel stores PRE-softmax logits)")

    if "dbg_expert_index" in debug:
        # Compare topK indices set-wise per token (order may differ within tied
        # probabilities, but the SET of chosen experts should match).
        idx_nki = debug["dbg_expert_index"].to(torch.int64)
        idx_ref = refs["topk_indices"].to(torch.int64)
        T_ = idx_nki.shape[0]
        K_ = idx_nki.shape[1]
        # Per-token set diff
        nki_sets = [set(idx_nki[t].tolist()) for t in range(T_)]
        ref_sets = [set(idx_ref[t].tolist()) for t in range(T_)]
        set_match = sum(1 for n, r in zip(nki_sets, ref_sets) if n == r)
        ordered_match = (idx_nki == idx_ref).all().item()
        print(f"[expert_index] set_match = {set_match}/{T_}  "
              f"ordered_match = {ordered_match}")
        if set_match < T_:
            for t in range(min(T_, 4)):
                print(f"  token {t}: nki={sorted(nki_sets[t])}  ref={sorted(ref_sets[t])}")

    if "dbg_expert_affinities" in debug:
        # expert_affinities_sb after router_topk is the sparse normalized
        # top-K probabilities scattered into [T, E] (zero at non-selected
        # experts). Compute reference equivalent and compare.
        T_ = refs["topk_indices"].shape[0]
        K_ = refs["topk_indices"].shape[1]
        ref_aff = torch.zeros(T_, E, dtype=torch.float32)
        for t in range(T_):
            for k in range(K_):
                e = refs["topk_indices"][t, k].item()
                ref_aff[t, e] = refs["topk_weights"][t, k].to(torch.float32)
        aff_err = _max_abs_err(debug["dbg_expert_affinities"], ref_aff)
        aff_mean = _mean_abs_err(debug["dbg_expert_affinities"], ref_aff)
        print(f"[expert_affinities (sparse normalized)] max_abs_err = "
              f"{aff_err:.6e}  mean = {aff_mean:.6e}")
        # Also print how many non-zero affinities there are vs expected K.
        nki_nnz_per_token = (debug["dbg_expert_affinities"] != 0).sum(dim=-1)
        print(f"  nki non-zero counts per token (first 4): "
              f"{nki_nnz_per_token[:4].tolist()}  (expected {K_})")

    if "dbg_moe_out_pershard" in debug:
        # Shape [N_PRGS, H0, BxS, H1_SHARD]. Sum across shards and remap to
        # canonical [B, S_tkg, H] to compare against Y_ref (the full MoE block
        # output should be the sum-of-shard outputs at single-rank
        # all-reduce + LNC gather).
        from megakernels.qwen3_moe.transformer_qwen3_moe_speculative import (
            H0 as _H0, H1, H1_SHARD as _H1_SHARD, N_PRGS as _N_PRGS, H2 as _H2,
        )
        pershard = debug["dbg_moe_out_pershard"].to(torch.float32)
        # Reconstruct canonical [B, S_tkg, H]:
        #   shard s writes H values into partition layout:
        #     output_sb[p, t, h1_idx] for h1_idx in [0..H1_SHARD) holds the
        #     value for hidden index ``s * H0 * H2 + p * H2 + h1_idx``
        #     (i.e., the shard's local H range).
        # After AR-gather, gathered_sb[p, t, s*H1_SHARD + h1_idx]
        #   = pershard[s, p, t, h1_idx]
        # Final canonical map: Y[b, s_tkg, h] = gathered_sb[p, t, h1] where
        #   h = s * H0 * H2 + p * H2 + h1_idx,  s = h // 1024,
        #   p = (h % 1024) // H2,  h1_idx = h % H2.
        B = inputs["X"].shape[0]
        S_tkg = inputs["X"].shape[1]
        BxS = B * S_tkg
        y_reconstructed = torch.zeros(B, S_tkg, H, dtype=torch.float32)
        for s in range(_N_PRGS):
            for p in range(_H0):
                for h2 in range(_H2):
                    h_idx = s * _H0 * _H2 + p * _H2 + h2
                    for t in range(BxS):
                        b = t // S_tkg
                        s_tkg = t % S_tkg
                        y_reconstructed[b, s_tkg, h_idx] = pershard[s, p, t, h2]
        # AR-gather is a no-op at single-rank, so reconstruction should
        # match Y_nki bit-for-bit modulo dtype.
        rec_vs_nki = _max_abs_err(y_reconstructed.to(Y_nki.dtype), Y_nki)
        print(f"[moe_out_pershard reconstructed vs Y_nki] max_abs_err = "
              f"{rec_vs_nki:.6e}  (should be ~0 — sanity check the layout)")
        rec_vs_ref = _max_abs_err(y_reconstructed.to(Y_ref.dtype), Y_ref)
        print(f"[moe_out_pershard reconstructed vs Y_ref] max_abs_err = "
              f"{rec_vs_ref:.6e}")

    # ------------------------------------------------------------------
    # Final Y comparison
    # ------------------------------------------------------------------
    max_err  = _max_abs_err(Y_ref, Y_nki)
    mean_err = _mean_abs_err(Y_ref, Y_nki)
    print(f"[Y]   max_abs_err = {max_err:.6e}   mean_abs_err = {mean_err:.6e}")

    torch.testing.assert_close(
        Y_nki.to(torch.float32), Y_ref.to(torch.float32),
        atol=ATOL_STRICT, rtol=RTOL_STRICT,
        msg=lambda m: (
            f"NKI MoE output does not match HF reference within strict "
            f"tolerances (atol={ATOL_STRICT}, rtol={RTOL_STRICT}). "
            f"max_abs_err={max_err:.6e}, mean_abs_err={mean_err:.6e}. "
            f"Original failure:\n{m}"
        ),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
