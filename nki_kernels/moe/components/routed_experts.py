# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routed-experts path of the Qwen3.6-A3B fused MoE layer (token generation), SBUF-resident.

Composes two validated nkilib sub-kernels with the post-attn RMSNorm, keeping the normed hidden
SBUF-resident with ZERO HBM round-trip (norm ONCE -> router + selective experts read the same tile):

    normed_sb [H0,T,H1]
       |-- router_topk            (fp32 softmax over E -> top-k -> L1-norm) -> index[T,K], eager[T,K] (SBUF)
       '-- routed_experts_selective (raw-NKI top-k loop: SiLU, POST_SCALE)  -> routed_local

This is the FIRST MoE slice (routed experts only). The shared expert, sigma-gate, gated sum and the
combined TP all-reduce are the NEXT slice; ``normed_sb`` is returned/reused so that slice consumes the
SAME tile.

Per-rank (TP=4) A3B config: H=2048, E=256, K=8, I=128 (moe_intermediate_size=512, sharded on I; EP=1,
all 256 experts replicated per rank). ``router_w`` is rank-replicated [H,E]. ``routed_local`` is the
per-rank partial over H (down-proj contracts the I-shard); the cross-rank all-reduce is deferred.
"""

import nki.language as nl

from nkilib.core.router_topk.router_topk import router_topk
from nkilib.core.utils.common_types import RouterActFnType
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from .post_attn_norm import post_attn_rmsnorm_compose
from .routed_experts_nki import routed_experts_selective

P_MAX = 128
HIDDEN = 2048

# router_topk SBUF x layout must match rmsnorm_tkg's emitted [H0,T,H1] H-permutation: num_H_shards=1 ->
# tp102 (H=h0*H1+j); num_H_shards=2 -> tp2013 (interleaved stride H/256). moe_tkg's H-shard mode (below)
# fixes num_H_shards, so the router never reloads/transposes.
X_SB_LAYOUT_TP102 = 0
X_SB_LAYOUT_TP2013 = 1

# moe_tkg selective disables token-sharding (-> shards on H) for this "big config" H*I threshold; below it,
# it shards on tokens for T>1. Mirrored here so the norm layout + router x_sb_layout track the consumer.
_MOE_BIG_CONFIG_HI = 3072 * 1536


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def moe_h_shard_mode(T, H, moe_intermediate, n_prgs):
    """Mirror moe_tkg selective's shard decision to keep the norm layout + router in lockstep.

    moe_tkg shards on TOKENS (num_H_shards=1, layout-0) unless T==1 or the big config, in which case it
    shards on H (num_H_shards=n_prgs, interleaved layout-1). Returns:
        single_core_forced: pass to the norm -> emit num_H_shards=1 (layout-0) when moe shards on tokens.
        x_sb_layout:        router layout matching the emitted num_H_shards.
    """
    moe_shard_on_T = (
        n_prgs > 1 and T > 1 and (H * moe_intermediate < _MOE_BIG_CONFIG_HI)
    )
    moe_num_h_shards = 1 if (moe_shard_on_T or n_prgs == 1) else n_prgs
    single_core_forced = moe_num_h_shards == 1
    x_sb_layout = X_SB_LAYOUT_TP2013 if moe_num_h_shards == 2 else X_SB_LAYOUT_TP102
    return single_core_forced, x_sb_layout


def routed_token_shard(T, H, moe_intermediate, n_prgs, shard_id):
    """Per-LNC-core token-shard geometry, mirroring the selective expert loop's shard decision.

    Returns (T_offset, T_per_shard) for this core. The routed loop token-shards (each core owns
    ``[T_offset:T_offset+T_per_shard]`` over the full H) only when there is more than one core, T>1 and
    it is not the big config; otherwise each core computes the full T. Matches ``moe_tkg_shard_decision``
    in shared_expert.py so ``routed_local`` keeps the same per-core layout as ``shared_local``.
    """
    single_core_forced, _ = moe_h_shard_mode(T, H, moe_intermediate, n_prgs)
    shard_on_T = single_core_forced and n_prgs > 1
    if shard_on_T:
        T_first = T // n_prgs
        T_per_shard = T_first if shard_id == 0 else T - T_first
        T_offset = 0 if shard_id == 0 else T_first
    else:
        T_per_shard, T_offset = T, 0
    return T_offset, T_per_shard


def routed_experts_compose(
    normed_sb,
    router_w,
    expert_gate_up_w,
    expert_down_w,
    k=8,
    output_in_sbuf=True,
):
    """Router + selective experts on an SBUF-resident normed tile (zero HBM round-trip).

    Args:
        normed_sb:        [H0=128, T, H1=H//128] SBUF post-attn-normed hidden (rmsnorm_tkg layout).
        router_w:         [H, E] HBM router weight, rank-replicated (load-transposed from stored [E,H]).
        expert_gate_up_w: [E, H, 2, I] HBM fused gate/up expert weights (kernel layout).
        expert_down_w:    [E, I, H] HBM down expert weights (kernel layout).
        k:                top-k experts per token (A3B: 8).
        output_in_sbuf:   True -> routed_local SBUF [H0,T,H1] (megakernel API); False -> HBM [T,H] natural.

    Returns:
        routed_local: SBUF [H0,T,H1] (output_in_sbuf=True) or HBM [T,H] (False) -- per-rank routed partial.
    """
    H0, T, H1 = normed_sb.shape
    H = H0 * H1
    E = router_w.shape[1]
    moe_intermediate = expert_gate_up_w.shape[3]
    kernel_assert(
        router_w.shape[0] == H, "router_w must be [H, E] matching normed_sb H"
    )
    kernel_assert(k <= 8, "router_topk supports k <= 8")

    # Router x layout must match the normed tile's num_H_shards (set by the token-shard mode). The caller
    # must have normed with the matching single_core_forced (moe_routed_compose does; the megakernel must).
    _, n_prgs, shard_id = get_verified_program_sharding_info("moe_routed", (0, 1), 2)
    _, x_sb_layout = moe_h_shard_mode(T, H, moe_intermediate, n_prgs)
    T_offset, T_per_shard = routed_token_shard(T, H, moe_intermediate, n_prgs, shard_id)

    # Router: fp32 softmax-over-E -> top-k -> L1-normalize (norm_topk_prob). SBUF outputs; eager returned.
    # NOTE: for SBUF expert_affinities the router REBINDS its output to an internal scattered tile, so the
    # passed `affinities` buffer is never written -- consume the RETURNED tensors (outs), not `affinities`.
    affinities = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    index = nl.ndarray((T, k), dtype=nl.uint32, buffer=nl.sbuf)
    outs = router_topk(
        x=normed_sb,
        w=router_w,
        w_bias=None,
        router_logits=None,
        expert_affinities=affinities,
        expert_index=index,
        act_fn=RouterActFnType.SOFTMAX,
        k=k,
        x_hbm_layout=0,
        x_sb_layout=x_sb_layout,
        router_pre_norm=True,  # ACT1: softmax BEFORE top-k
        norm_topk_prob=True,  # L1-normalize the top-k affinities
        return_eager_affi=True,
        skip_store_router_logits=True,  # router_logits unused downstream
    )
    # outs = [router_logits(None), expert_index[T,k], scattered_affinities[T,E], eager[T,k]] -- SBUF.
    # eager[t,k] == scattered[t, index[t,k]] (both the L1-normalized affinity of the k-th selected
    # expert); the selective loop scales by eager directly, avoiding a scatter/gather round-trip.
    _, index_out, _, eager = outs

    # Selective experts: top-k of E, SiLU, POST_SCALE by the (already-normalized) affinity multiplied
    # ONCE (no re-normalization). Raw-NKI loop with a fused [H0,H1,2I] gate/up slab load + prefetch ring.
    routed_local = routed_experts_selective(
        normed_sb,
        expert_gate_up_w,
        expert_down_w,
        index_out,
        eager,
        T_offset,
        T_per_shard,
        output_in_sbuf=output_in_sbuf,
    )
    return routed_local


def moe_routed_compose(
    hidden,
    gamma,
    router_w,
    expert_gate_up_w,
    expert_down_w,
    eps=1e-6,
    k=8,
    hidden_actual=None,
    output_in_sbuf=True,
):
    """Full routed-experts chain: post-attn RMSNorm -> router -> selective experts (all SBUF-resident).

    Args:
        hidden:   [B, S, H] HBM raw post-attn residual (B=1), left untouched.
        gamma:    [1, H] HBM post_attention_layernorm.weight (standard form).
        router_w / expert_gate_up_w / expert_down_w: see ``routed_experts_compose``.
        eps:      RMSNorm epsilon.  k: top-k experts.  hidden_actual: H for the mean if padded.
        output_in_sbuf: routed_local buffer (SBUF [H0,T,H1] default, else HBM [T,H]).

    Returns:
        (routed_local, normed_sb): the routed partial and the SBUF normed tile (shared by the next slice).
    """
    B, S, H = hidden.shape
    T = B * S
    moe_intermediate = expert_gate_up_w.shape[3]
    _, n_prgs, _ = get_verified_program_sharding_info("moe_routed", (0, 1), 2)
    # Emit the norm in the H layout moe_tkg's shard mode expects (the router matches it independently).
    single_core_forced, _ = moe_h_shard_mode(T, H, moe_intermediate, n_prgs)
    normed_sb = post_attn_rmsnorm_compose(
        hidden, gamma, eps, hidden_actual, single_core_forced=single_core_forced
    )
    routed_local = routed_experts_compose(
        normed_sb,
        router_w,
        expert_gate_up_w,
        expert_down_w,
        k=k,
        output_in_sbuf=output_in_sbuf,
    )
    return routed_local, normed_sb
