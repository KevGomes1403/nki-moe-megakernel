# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused DeltaNet causal conv + gated delta-rule recurrence for token generation.

One LNC=2 launch runs the depthwise conv and the recurrence back-to-back with q/k/v passed
conv->recurrence entirely in SBUF -- no HBM round-trip for q/k/v. The work is SPMD-sharded by
**value-head**: each NeuronCore runs a self-contained conv->recurrence pipeline over its own heads
with zero cross-core traffic, and writes a disjoint slice of every full-shape output. This collapses
the two separate model calls (conv_attn_nki + deltanet_attn_nki) into a single kernel.

Per core (program ``c`` of ``n=2``):
  1. ``conv_qkv_sbuf`` runs the head-sharded conv over this core's 3 owned channel segments (its
     q/k/v slices) and returns three separate partition-0-based SBUF tiles ``(q_sbuf, k_sbuf,
     v_sbuf)`` -- each silu'd conv output, partition = local_head*T+t, free = head_dim. It also
     scatters this core's conv-state slices to the full-shape HBM state output.
  2. ``gated_delta_rule_tkg`` consumes those SBUF tiles directly (SBUF-input variant): q/k feed
     ``_load_normed_qk`` (l2norm + transpose), v is gathered per token into ``v_row`` via an on-chip
     SBUF->SBUF DMA. The gating tables (a/b/A_log/dt_bias), init_state seed, attn_out columns, and
     state-head writes stay FULL HBM tensors sliced by the recurrence's ``vh_off``/``col_off``
     plumbing.

The q/k bridge is free: the conv's silu'd q/k sub-blocks already match the ``[Hk_loc*T, dim]`` layout
``_load_normed_qk`` expects after its HBM load. The v bridge is the only reshuffle and stays on-chip:
``v_sbuf`` (partition = local_head*T+t) is gathered to per-token ``v_row [1, W_loc]`` (free =
local_head*dim + j) with a strided-partition SBUF->SBUF DMA.

Entrypoints (head_dim d=128), launched ``[2]``:
  deltanet_fused_tkg_fwd        -> (attn_out [T,Hv*128], final_state [Hv,128,128],
                                    new_conv_state [conv_dim,K-1])          decode / commit (T=1)
  deltanet_fused_tkg_fwd_state  -> (attn_out, candidate_states [T,Hv,128,128],
                                    conv_cand [T,conv_dim,K-1])             speculative verify (T>=2)

Inputs (qkv f32, as today):
    qkv        (T, conv_dim)    raw in_proj_qkv output, token-major: cat(q,k,v) on free axis
    conv_state (conv_dim, K-1)  carried conv window
    conv_weight(conv_dim, K)    per-channel taps, no bias
    key_dim    python int       q/k segment width; Hk=key_dim//128, value_dim=conv_dim-2*key_dim
    a, b       (T, Hv)          raw in_proj_a / in_proj_b (token-major, head on free)
    A_log      (Hv,)            per-head decay param
    dt_bias    (Hv,)            per-head decay bias
    init_state (Hv, 128, 128)   carried recurrent state
Outputs are FULL-shape in shared_hbm; each core writes its disjoint head/column/channel slice.
"""

import nki
import nki.language as nl

from models.qwen3_6_moe.nki_kernels.nki_deltanet_conv_tkg import (
    P_MAX,
    conv_qkv_sbuf,
    kernel_assert,
)
from models.qwen3_6_moe.nki_kernels.nki_deltanet_tkg import gated_delta_rule_tkg


def fused_compose(qkv, conv_state, conv_weight, key_dim, a, b, A_log, dt_bias, init_state,
                  attn_out, state_hbm, conv_cand, write_candidates, cand_is_3d):
    """Compose the per-core conv -> recurrence pipeline (shared by both entrypoints).

    Derives the value-head shard once (the conv and recurrence both fold ``program_id``/``num_programs``
    to the same trace-time shard), runs ``conv_qkv_sbuf`` to produce this core's q/k/v SBUF tiles +
    scatter its conv-state slices, then drives the recurrence straight off those tiles. Hk_full /
    Hv_full are passed explicitly because the SBUF tiles carry only local heads (Hv_full from the state
    output's head axis, Hk_full from key_dim).
    """
    conv_dim = qkv.shape[1]
    Hk_full = key_dim // P_MAX
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    kernel_assert(state_hbm.shape[-3] == Hv_full, "state head count must equal Hv")

    q_sbuf, k_sbuf, v_sbuf = conv_qkv_sbuf(
        qkv, conv_state, conv_weight, key_dim, conv_cand, cand_is_3d
    )
    gated_delta_rule_tkg(
        None, None, None, a, b, A_log, dt_bias, init_state,
        attn_out, state_hbm, write_candidates,
        q_sbuf=q_sbuf, k_sbuf=k_sbuf, v_sbuf=v_sbuf, Hk_full=Hk_full, Hv_full=Hv_full,
    )


@nki.jit
def deltanet_fused_tkg_fwd(qkv, conv_state, conv_weight, key_dim,
                           a, b, A_log, dt_bias, init_state):
    """Decode / commit (T=1): fused conv + recurrence in one launch.

    Returns:
        attn_out       (T, Hv*128) f32 -- raw head-major recurrence output (caller RMSNorms/z-gates)
        final_state    (Hv, 128, 128) f32 -- recurrent state after the block token
        new_conv_state (conv_dim, K-1) -- committed new conv window (== conv_cand[0])
    Value-head sharded across cores; launch ``deltanet_fused_tkg_fwd[2](...)``.
    """
    T, conv_dim = qkv.shape
    state_w = conv_weight.shape[1] - 1
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    W_full = Hv_full * P_MAX
    attn_out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state = nl.ndarray((Hv_full, P_MAX, P_MAX), dtype=nl.float32, buffer=nl.shared_hbm)
    new_conv_state = nl.ndarray((conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm)
    fused_compose(
        qkv, conv_state, conv_weight, key_dim, a, b, A_log, dt_bias, init_state,
        attn_out, final_state, new_conv_state, write_candidates=False, cand_is_3d=False,
    )
    return attn_out, final_state, new_conv_state


@nki.jit
def deltanet_fused_tkg_fwd_state(qkv, conv_state, conv_weight, key_dim,
                                 a, b, A_log, dt_bias, init_state):
    """Speculative verify (T>=2): fused conv + recurrence in one launch.

    ``candidate_states[t]`` / ``conv_cand[t]`` are the recurrent / conv state after consuming block
    token t; on a reject the host selects ``[accept_count - 1]``. Returns:
        attn_out         (T, Hv*128) f32 -- raw head-major recurrence output
        candidate_states (T, Hv, 128, 128) f32 -- recurrent state after each token
        conv_cand        (T, conv_dim, K-1) -- conv window after each token
    Value-head sharded across cores; launch ``deltanet_fused_tkg_fwd_state[2](...)``.
    """
    T, conv_dim = qkv.shape
    state_w = conv_weight.shape[1] - 1
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    W_full = Hv_full * P_MAX
    attn_out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)
    candidate_states = nl.ndarray(
        (T, Hv_full, P_MAX, P_MAX), dtype=nl.float32, buffer=nl.shared_hbm
    )
    conv_cand = nl.ndarray((T, conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm)
    fused_compose(
        qkv, conv_state, conv_weight, key_dim, a, b, A_log, dt_bias, init_state,
        attn_out, candidate_states, conv_cand, write_candidates=True, cand_is_3d=True,
    )
    return attn_out, candidate_states, conv_cand
