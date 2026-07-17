# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-expert path of the Qwen3.6-A3B fused MoE layer (token generation), SBUF-resident.

A plain SwiGLU FFN -- down(silu(gate(x)) * up(x)) -- on the SAME SBUF-resident ``normed_sb [H0,T,H1]``
the routed path consumes (zero HBM round-trip). Returns the per-rank partial over H (down_proj without
internal reduce); the sigma-gate, gated sum and combined TP all-reduce live in ``moe_layer``.

Written in raw NKI, structurally ONE routed expert without the expert index / affinity scale, reusing the
same AP idioms as ``routed_experts_nki``. nkilib's ``mlp_tkg`` primitives cannot be used: their gate/up
loader hardcodes ``weight.reshape_dim(dim=0, shape=(H0, H1_shard))`` -- the tp102 H-permutation -- and
tp2013 needs H viewed as (s, h0, h2) and then PERMUTED so h0 is the partition axis, which ``reshape_dim``
cannot express and no flag toggles.

Layout (tp2013, shared with the attention kernels' SBUF residual): with ``n_s = n_prgs`` H-shards and
``H2 = H1 // n_s``, free index ``f = s*H2 + h2`` maps to H-column ``s*(H0*H2) + h0*H2 + h2``; at one core
this degenerates to tp102. gate/up are SEPARATE ``[H, I_s]`` tensors, so each partition's H2 rows coalesce
into an ``H2 * I_s`` run (2 KB at bf16); down is the contiguous ``[I_s, H]`` row load, permutation-agnostic.

Work split: TOKEN-sharded, identical to the routed path (``moe_token_shard``), so ``shared_local`` keeps
the same per-core layout as ``routed_local`` and the gated sum is a plain per-core add. At cores>1 / T==1
there is no token split, so both cores compute the full tile -- redundant but correct on every core.
Weights are rank-replicated; ``shared_local`` stays the per-rank partial (cross-rank all-reduce deferred).

Per-rank (TP=4) A3B dims: H=2048 (H0=128, H1=16), I_s=128 (shared_expert_intermediate_size 512 / TP=4).
"""

import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from .post_attn_norm import post_attn_rmsnorm_compose
from .routed_experts import NORM_SINGLE_CORE_FORCED, moe_token_shard
from .routed_experts_nki import store_tile_to_hbm_th

P_MAX = 128


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def moe_tkg_shard_decision(T, H, intermediate):
    """This core's (shard_on_T, T_offset, T_per_shard) for the MoE expert paths (routed == shared)."""
    _, n_prgs, shard_id = get_verified_program_sharding_info("moe_shared", (0, 1), 2)
    return moe_token_shard(T, H, intermediate, n_prgs, shard_id)


def load_shared_weights(gate_w, up_w, down_w, H0, H1, H2, n_s, I_s, H, dtype):
    """Load the shared expert's [H,I_s] gate/up (tp2013) and [I_s,H] down (contiguous) into SBUF.

    gate/up slabs: slab[h0, s*H2 + h2, c] = w[s*H0*H2 + h0*H2 + h2, c]; dim1 step (I_s) equals the inner
    extent, so each partition's H2 rows coalesce into one H2*I_s run. Static (non-indirect) DMAs.
    """
    gate_sb = nl.ndarray((H0, H1, I_s), dtype=dtype, buffer=nl.sbuf)
    up_sb = nl.ndarray((H0, H1, I_s), dtype=dtype, buffer=nl.sbuf)
    for s in range(n_s):
        pattern = [[H2 * I_s, H0], [I_s, H2], [1, I_s]]
        offset = s * H0 * H2 * I_s
        nisa.dma_copy(
            dst=gate_sb[0:H0, s * H2 : (s + 1) * H2, 0:I_s],
            src=gate_w.ap(pattern=pattern, offset=offset),
        )
        nisa.dma_copy(
            dst=up_sb[0:H0, s * H2 : (s + 1) * H2, 0:I_s],
            src=up_w.ap(pattern=pattern, offset=offset),
        )
    down_sb = nl.ndarray((I_s, H), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=down_sb[0:I_s, 0:H], src=down_w)
    return gate_sb, up_sb, down_sb


def shared_expert_compose(normed_sb, gate_w, up_w, down_w, output_in_sbuf=True):
    """SwiGLU shared expert on an SBUF-resident normed tile (raw NKI, zero HBM round-trip).

    Args:
        normed_sb:      [H0=128, T, H1=H//128] SBUF post-attn-normed hidden (tp2013, n_s = n_prgs).
        gate_w:         [H, I_s] HBM gate weight (contraction-first; load-transposed from stored [I_s,H]).
        up_w:           [H, I_s] HBM up weight (contraction-first; load-transposed from stored [I_s,H]).
        down_w:         [I_s, H] HBM down weight (load-transposed from stored [H, I_s]).
        output_in_sbuf: True -> shared_local SBUF [H0,T,H1] (megakernel API, matches routed_local);
                        False -> HBM [T,H] natural (isolation authoritative gate).

    Returns:
        shared_local: SBUF [H0,T,H1] (output_in_sbuf=True) or HBM [T,H] (False) -- per-rank partial with
                      the SAME per-LNC-core token-shard layout as routed_local.
    """
    H0, T, H1 = normed_sb.shape
    H = H0 * H1
    I_s = up_w.shape[1]
    dtype = normed_sb.dtype
    kernel_assert(
        tuple(gate_w.shape) == (H, I_s), "gate_w must be [H, I_s] matching normed_sb H"
    )
    kernel_assert(
        tuple(up_w.shape) == (H, I_s), "up_w must be [H, I_s] matching normed_sb H"
    )
    kernel_assert(tuple(down_w.shape) == (I_s, H), "down_w must be [I_s, H]")
    kernel_assert(
        I_s <= P_MAX, "shared expert supports I_s <= 128 (single intermediate tile)"
    )

    _, n_prgs, shard_id = get_verified_program_sharding_info("moe_shared", (0, 1), 2)
    kernel_assert(
        H1 % n_prgs == 0, "tp2013 needs H1 divisible by the H-shard count n_s"
    )
    n_s = n_prgs
    H2 = H1 // n_s
    _, T_offset, T_per_shard = moe_token_shard(T, H, I_s, n_prgs, shard_id)

    gate_sb, up_sb, down_sb = load_shared_weights(
        gate_w, up_w, down_w, H0, H1, H2, n_s, I_s, H, dtype
    )

    # gate/up: contract H0 over the H1 free indices for this core's whole token slice at once.
    gate_psum = nl.ndarray((I_s, T_per_shard), dtype=nl.float32, buffer=nl.psum)
    up_psum = nl.ndarray((I_s, T_per_shard), dtype=nl.float32, buffer=nl.psum)
    for f in range(H1):
        moving = normed_sb[:, T_offset : T_offset + T_per_shard, f]
        nisa.nc_matmul(dst=gate_psum, stationary=gate_sb[:, f, 0:I_s], moving=moving)
        nisa.nc_matmul(dst=up_psum, stationary=up_sb[:, f, 0:I_s], moving=moving)

    gate_act = nl.ndarray((I_s, T_per_shard), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=gate_act, data=gate_psum, op=nl.silu)
    up_act = nl.ndarray((I_s, T_per_shard), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=up_act, src=up_psum)
    intermediate = nl.ndarray((I_s, T_per_shard), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=intermediate, data1=gate_act, data2=up_act, op=nl.multiply)

    # down: contract I_s -> [H0,1] per free index f, the strided stationary picking the tp2013 H-columns
    # {s*H0*H2 + h0*H2 + h2} (s, h2 = divmod(f, H2)). Each core writes ONLY its own token slice.
    shared_local = nl.ndarray((H0, T, H1), dtype=dtype, buffer=nl.sbuf)
    for local_token_idx in range(T_per_shard):
        down_psum = nl.ndarray((H0, H1), dtype=nl.float32, buffer=nl.psum)
        for f in range(H1):
            s, h2 = f // H2, f % H2
            nisa.nc_matmul(
                dst=down_psum[0:H0, f : f + 1],
                stationary=down_sb.ap(
                    pattern=[[H, I_s], [H2, H0]], offset=s * H0 * H2 + h2
                ),
                moving=intermediate[0:I_s, local_token_idx : local_token_idx + 1],
            )
        nisa.tensor_copy(
            dst=shared_local[:, T_offset + local_token_idx, :], src=down_psum
        )

    if output_in_sbuf:
        return shared_local

    out_hbm = nl.ndarray((T, H), dtype=dtype, buffer=nl.shared_hbm)
    store_tile_to_hbm_th(out_hbm, shared_local, H0, H2, n_s, H, T_offset, T_per_shard)
    return out_hbm


def moe_shared_compose(
    hidden,
    gamma,
    gate_w,
    up_w,
    down_w,
    eps=1e-6,
    hidden_actual=None,
    output_in_sbuf=True,
):
    """Full shared-expert chain: post-attn RMSNorm -> SwiGLU shared expert (all SBUF-resident).

    Args:
        hidden:   [B, S, H] HBM raw post-attn residual (B=1), left untouched.
        gamma:    [1, H] HBM post_attention_layernorm.weight (standard form).
        gate_w / up_w / down_w: see ``shared_expert_compose``.
        eps:      RMSNorm epsilon.  hidden_actual: H for the mean if padded.
        output_in_sbuf: shared_local buffer (SBUF [H0,T,H1] default, else HBM [T,H]).

    Returns:
        (shared_local, normed_sb): the shared partial and the SBUF normed tile (shared by the routed slice).
    """
    normed_sb = post_attn_rmsnorm_compose(
        hidden,
        gamma,
        eps,
        hidden_actual,
        single_core_forced=NORM_SINGLE_CORE_FORCED,
    )
    shared_local = shared_expert_compose(
        normed_sb, gate_w, up_w, down_w, output_in_sbuf=output_in_sbuf
    )
    return shared_local, normed_sb
