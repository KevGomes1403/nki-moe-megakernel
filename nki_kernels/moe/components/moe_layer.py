# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fully fused MoE layer of the Qwen3.6-A3B decoder (token generation), SBUF-resident.

Stitches the three validated building blocks into ONE composable and adds the last two pieces
(sigma-gate + gated sum), all SBUF-resident with a single post-attn RMSNorm and ZERO HBM round-trip:

    normed_sb = post_attn_rmsnorm(hidden, gamma)               <- NORM ONCE, shared by all consumers
       |-- routed_local  = routed_experts_compose(normed_sb)   (router_topk + selective moe_tkg)
       |-- shared_local  = shared_expert_compose(normed_sb)    (mlp_tkg SwiGLU shared expert)
       '-- g             = sigmoid(normed_sb .h sigma_gate_w)  (tiny H->1 matmul, rank-replicated)
    combined_local = routed_local + broadcast(g) * shared_local

Reference (modeling_qwen36_a3b.py NeuronMoEBlock): all of router/routed/shared/sigma consume the SAME
post-attn-normed hidden, so norm ONCE. ``combined_local`` is the per-rank partial; the model applies the
SINGLE ``reduce_from_tensor_model_parallel_region`` -- valid because the sigma-gate is rank-replicated:
AR(routed) + g*AR(shared) == AR(routed + g*shared). No cross-rank all-reduce here (megakernel/model
boundary; a future megakernel could use ``nki.collectives.all_reduce`` on the SBUF tile).

Every SBUF tile here -- ``normed_sb`` in and ``combined_local`` out -- is in the tp2013 H-permutation the
attention kernels use for their residual (n_s = n_prgs H-shards, H2 = H1 // n_s, free index f = s*H2 + h2
<-> H-column s*(H0*H2) + h0*H2 + h2), so a megakernel can share ONE SBUF residual. At one core it
degenerates to tp102. The HBM contract ([B,S,H] in, [1,T,H] out) is layout-independent.

Per-rank (TP=4) A3B config: H=2048, E=256, K=8, routed I=128, shared I_s=128.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from .post_attn_norm import post_attn_rmsnorm_compose
from .routed_experts import NORM_SINGLE_CORE_FORCED, routed_experts_compose
from .routed_experts_nki import store_tile_to_hbm_th
from .shared_expert import moe_tkg_shard_decision, shared_expert_compose


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def sigma_gate_compose(normed_sb, sigma_gate_w):
    """Sigmoid shared-expert gate: g = sigmoid(normed .h sigma_gate_w) -> [1, T] (rank-replicated).

    An H->1 projection contracting the full H, which lives split as the H0 partition x H1 free axes of
    ``normed_sb``. The [H,1] weight is loaded through the same tp2013 AP -- w_sb[h0, s*H2 + h2] =
    w[s*H0*H2 + h0*H2 + h2] -- so each free-index matmul contracts H0 with matching operands. Output is a
    [1, T] row (M=1) ready for the partition-broadcast in the gated sum.

    Args:
        normed_sb:    [H0=128, T, H1=H//128] SBUF post-attn-normed hidden (tp2013, n_s = n_prgs).
        sigma_gate_w: [H, 1] HBM gate weight, rank-replicated (load-transposed from stored [1, H]).

    Returns:
        g: [1, T] SBUF fp32 -- the per-token sigmoid gate (same on every rank).
    """
    H0, T, H1 = normed_sb.shape
    H = H0 * H1
    kernel_assert(tuple(sigma_gate_w.shape) == (H, 1), "sigma_gate_w must be [H, 1]")

    _, n_s, _ = get_verified_program_sharding_info("moe_sigma_gate", (0, 1), 2)
    kernel_assert(H1 % n_s == 0, "tp2013 needs H1 divisible by the H-shard count n_s")
    H2 = H1 // n_s

    w_sb = nl.ndarray((H0, H1), dtype=normed_sb.dtype, buffer=nl.sbuf)
    for s in range(n_s):
        nisa.dma_copy(
            dst=w_sb[0:H0, s * H2 : (s + 1) * H2],
            src=sigma_gate_w.ap(pattern=[[H2, H0], [1, H2]], offset=s * H0 * H2),
        )

    logit_ps = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.psum)
    for f in range(H1):  # accumulate H0-contractions over H1 -> full-H logit [1, T]
        nisa.nc_matmul(
            dst=logit_ps, stationary=w_sb[:, f : f + 1], moving=normed_sb[:, :, f]
        )
    logit_sb = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=logit_sb, src=logit_ps)
    g = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=g, data=logit_sb, op=nl.sigmoid)
    return g


def gated_sum(routed_local, shared_local, g):
    """combined = routed_local + broadcast(g) * shared_local, in [H0, T, H1] SBUF.

    ``g`` is [1, T] but T is the MIDDLE axis of [H0,T,H1]; broadcast it across the H0 partitions (one
    ones-matmul: [H0,T] = ones[1,H0].T @ g[1,T]) then multiply each H1 slice (T on the free axis) and add
    routed. Operates on the full tile -- at cores=2/T>1 routed/shared hold only their token slice, so only
    that slice is valid, which the caller's per-core store selects.

    Args:
        routed_local / shared_local: [H0, T, H1] SBUF per-rank partials in identical per-core layout.
        g:                           [1, T] SBUF fp32 sigmoid gate.

    Returns:
        combined: [H0, T, H1] SBUF per-rank partial (same dtype/layout as the inputs).
    """
    H0, T, H1 = routed_local.shape
    dtype = routed_local.dtype

    ones_sb = nl.ndarray((1, H0), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_sb, value=1.0)
    g_bc_ps = nl.ndarray((H0, T), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=g_bc_ps, stationary=ones_sb, moving=g
    )  # [H0,T] = g broadcast on H0
    g_bc = nl.ndarray((H0, T), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g_bc, src=g_bc_ps)

    combined = nl.ndarray((H0, T, H1), dtype=dtype, buffer=nl.sbuf)
    for j in range(H1):  # T on free -> plain [H0,T] elementwise per H1 slice
        nisa.tensor_tensor(
            dst=combined[:, :, j],
            data1=shared_local[:, :, j],
            data2=g_bc,
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=combined[:, :, j],
            data1=combined[:, :, j],
            data2=routed_local[:, :, j],
            op=nl.add,
        )
    return combined


def _store_combined_hbm(combined, moe_intermediate, output_bsh=False):
    """Store the tp2013 combined [H0,T,H1] tile -> HBM [T,H] natural (or [1,T,H] if output_bsh).

    ``output_bsh`` emits the [B=1,S=T,H] shape the model's residual stream expects (free -- same memory
    layout as [T,H]), so the caller needs no reshape. Token-shard aware: each core writes only its slice."""
    H0, T, H1 = combined.shape
    H = H0 * H1
    _, n_s, _ = get_verified_program_sharding_info("moe_store", (0, 1), 2)
    H2 = H1 // n_s
    _, T_offset, T_per_shard = moe_tkg_shard_decision(T, H, moe_intermediate)
    out_hbm = nl.ndarray(
        (1, T, H) if output_bsh else (T, H), dtype=combined.dtype, buffer=nl.shared_hbm
    )
    dst2d = out_hbm.reshape((T, H)) if output_bsh else out_hbm
    store_tile_to_hbm_th(dst2d, combined, H0, H2, n_s, H, T_offset, T_per_shard)
    return out_hbm


def moe_layer_compose(
    hidden,
    gamma,
    router_w,
    expert_gate_up_w,
    expert_down_w,
    sigma_gate_w,
    shared_gate_w,
    shared_up_w,
    shared_down_w,
    eps=1e-6,
    k=8,
    hidden_actual=None,
    output_in_sbuf=True,
    output_bsh=False,
    name_prefix="",
):
    """Fully fused MoE layer: norm ONCE -> routed + shared + sigma-gate -> gated sum (per-rank partial).

    Args:
        hidden:           [B, S, H] HBM raw post-attn residual (B=1), left untouched.
        gamma:            [1, H] HBM post_attention_layernorm.weight (standard form).
        router_w:         [H, E] HBM router weight, rank-replicated (load-transposed from stored [E, H]).
        expert_gate_up_w: [E, H, 2, I] HBM fused routed gate/up weights (kernel layout).
        expert_down_w:    [E, I, H] HBM routed down weights (kernel layout).
        sigma_gate_w:     [H, 1] HBM sigma-gate weight, rank-replicated (load-transposed from [1, H]).
        shared_gate_w:    [H, I_s] HBM shared gate weight (contraction-first, from stored [I_s, H]).
        shared_up_w:      [H, I_s] HBM shared up weight (contraction-first, from stored [I_s, H]).
        shared_down_w:    [I_s, H] HBM shared down weight (from stored [H, I_s]).
        eps:              RMSNorm epsilon.  k: top-k experts (A3B: 8).  hidden_actual: H for the mean.
        output_in_sbuf:   True -> combined SBUF [H0,T,H1] (megakernel API); False -> HBM [T,H] (isolation).

    Returns:
        combined_local: SBUF [H0,T,H1] (output_in_sbuf=True) or HBM [T,H] (False) -- the SINGLE per-rank
                        MoE partial for ONE downstream reduce_from_tensor_model_parallel_region.
    """
    moe_intermediate = expert_gate_up_w.shape[3]

    # NORM ONCE, in the tp2013 layout every consumer (router, routed, shared, sigma-gate) reads.
    normed_sb = post_attn_rmsnorm_compose(
        hidden,
        gamma,
        eps,
        hidden_actual,
        single_core_forced=NORM_SINGLE_CORE_FORCED,
        name_prefix=name_prefix,
    )

    routed_local = routed_experts_compose(
        normed_sb,
        router_w,
        expert_gate_up_w,
        expert_down_w,
        k=k,
        output_in_sbuf=True,
        name_prefix=name_prefix,
    )
    shared_local = shared_expert_compose(
        normed_sb, shared_gate_w, shared_up_w, shared_down_w, output_in_sbuf=True
    )
    kernel_assert(
        tuple(shared_local.shape) == tuple(routed_local.shape),
        "shared_local/routed_local shape mismatch -- per-core layouts not aligned",
    )

    g = sigma_gate_compose(normed_sb, sigma_gate_w)
    combined = gated_sum(routed_local, shared_local, g)

    if output_in_sbuf:
        return combined
    return _store_combined_hbm(combined, moe_intermediate, output_bsh=output_bsh)


@nki.jit
def moe_layer_fwd(
    hidden,
    gamma,
    router_w,
    expert_gate_up_w,
    expert_down_w,
    sigma_gate_w,
    shared_gate_w,
    shared_up_w,
    shared_down_w,
    eps=1e-6,
    k=8,
):
    """Fully fused MoE layer entrypoint (LNC2 launch [2]): the per-rank partial as HBM [B,S,H]=[1,T,H].

    Model-facing @nki.jit wrapper over ``moe_layer_compose`` for the verify path. Weights are consumed in
    their kernel-native (contraction-first) layout with no runtime transpose; the [1,T,H] output matches
    the residual stream so the caller reshapes nothing -- it just applies the single TP all-reduce.
    """
    return moe_layer_compose(
        hidden,
        gamma,
        router_w,
        expert_gate_up_w,
        expert_down_w,
        sigma_gate_w,
        shared_gate_w,
        shared_up_w,
        shared_down_w,
        eps=eps,
        k=k,
        output_in_sbuf=False,
        output_bsh=True,
    )
