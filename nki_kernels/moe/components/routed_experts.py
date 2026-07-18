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

TWO INDEPENDENT DECISIONS, never welded together:
  * WORK SPLIT (``moe_token_shard``): the experts token-shard across the LNC cores whenever cores>1,
    T>1 and the config is not the big one. Each core then owns a token slice over the FULL H.
  * SBUF H-LAYOUT: always tp2013 -- ``n_s = n_prgs`` H-shards, ``H2 = H1 // n_s``, free index
    ``f = s*H2 + h2`` <-> H-column ``s*(H0*H2) + h0*H2 + h2`` -- so the hidden tile matches the
    attention kernels' SBUF residual. ``rmsnorm_tkg`` keys the emitted permutation off ``lnc``, so
    ``single_core_forced=False`` gives exactly that; at one core it degenerates to tp102.

Per-rank (TP=4) A3B config: H=2048, E=256, K=8, I=128 (moe_intermediate_size=512, sharded on I; EP=1,
all 256 experts replicated per rank). ``router_w`` is rank-replicated [H,E]. ``routed_local`` is the
per-rank partial over H (down-proj contracts the I-shard); the cross-rank all-reduce is deferred.
"""

import nki.language as nl

from ..vendored.router_topk import router_topk
from nkilib.core.utils.common_types import RouterActFnType
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from .post_attn_norm import post_attn_rmsnorm_compose
from .routed_experts_nki import routed_experts_selective

P_MAX = 128
HIDDEN = 2048

# rmsnorm_tkg emits num_H_shards = lnc when single_core_forced is False -- exactly the tp2013 hidden tile
# (tp102 at lnc=1). The H-layout is dictated by the core count alone, NEVER by the token-shard decision.
NORM_SINGLE_CORE_FORCED = False

# router_topk SBUF x layouts: 0 = tp102 (H = h0*H1 + f), 1 = tp2013 (2-way interleave, stride H/256).
X_SB_LAYOUT_TP102 = 0
X_SB_LAYOUT_TP2013 = 1

# Above this H*I the experts stop token-sharding (mirrors moe_tkg's "big config" threshold): each core
# then computes the full T.
MOE_BIG_CONFIG_HI = 3072 * 1536


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def router_x_sb_layout(n_prgs):
    """router_topk x layout matching the tp2013 hidden tile (tp102 when there is a single core)."""
    if n_prgs > 1:
        return X_SB_LAYOUT_TP2013
    return X_SB_LAYOUT_TP102


def moe_token_shard(T, H, moe_intermediate, n_prgs, shard_id):
    """Per-LNC-core token-shard geometry: (shard_on_T, T_offset, T_per_shard).

    Token-shard -- each core owns ``[T_offset:T_offset+T_per_shard]`` over the FULL H -- when there is
    more than one core, T > 1 and it is not the big config; otherwise every core computes the full T.
    Independent of the SBUF H-layout: the work split is the same for tp102 and tp2013.
    """
    shard_on_T = n_prgs > 1 and T > 1 and (H * moe_intermediate < MOE_BIG_CONFIG_HI)
    if shard_on_T:
        T_first = T // n_prgs
        T_per_shard = T_first if shard_id == 0 else T - T_first
        T_offset = 0 if shard_id == 0 else T_first
    else:
        T_per_shard, T_offset = T, 0
    return shard_on_T, T_offset, T_per_shard


def routed_experts_compose(
    normed_sb,
    router_w,
    expert_gate_up_w,
    expert_down_w,
    k=8,
    output_in_sbuf=True,
    name_prefix="",
):
    """Router + selective experts on an SBUF-resident normed tile (zero HBM round-trip).

    Args:
        normed_sb:        [H0=128, T, H1=H//128] SBUF post-attn-normed hidden (tp2013, n_s = n_prgs).
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

    # The caller must have normed with NORM_SINGLE_CORE_FORCED so normed_sb is tp2013 with n_s = n_prgs
    # (moe_routed_compose / moe_layer_compose do; the megakernel must). router_topk's layout-1 loader is
    # 2-way, which is why n_prgs is capped at 2 here.
    _, n_prgs, shard_id = get_verified_program_sharding_info("moe_routed", (0, 1), 2)
    kernel_assert(
        n_prgs in (1, 2), "routed experts support 1 or 2 LNC cores (tp2013 n_s)"
    )
    _, T_offset, T_per_shard = moe_token_shard(T, H, moe_intermediate, n_prgs, shard_id)

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
        x_sb_layout=router_x_sb_layout(n_prgs),
        router_pre_norm=True,  # ACT1: softmax BEFORE top-k
        norm_topk_prob=True,  # L1-normalize the top-k affinities
        return_eager_affi=True,
        skip_store_router_logits=True,  # router_logits unused downstream
        name_prefix=name_prefix,
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
        n_s=n_prgs,
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
    normed_sb = post_attn_rmsnorm_compose(
        hidden,
        gamma,
        eps,
        hidden_actual,
        single_core_forced=NORM_SINGLE_CORE_FORCED,
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
