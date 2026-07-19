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

The ``deltanet_attention_layer`` / ``deltanet_attention_layer_state`` entrypoints run the whole layer
(input RMSNorm + in_proj + conv + recurrence + gated norm + output projection) in one launch and return
the per-rank projected output ``o_out [T, hidden]`` plus the unchanged recurrent-state / conv-state
outputs (the projection only touches the attention output).

Inputs (qkv f32, as today):
    qkv        (T, conv_dim)    raw in_proj_qkv output, token-major: cat(q,k,v) on free axis
    conv_state (conv_dim, K-1)  carried conv window
    conv_weight(conv_dim, K)    per-channel taps, no bias
    key_dim    python int       q/k segment width; Hk=key_dim//128, value_dim=conv_dim-2*key_dim
    a, b       (T, Hv)          raw in_proj_a / in_proj_b (token-major, head on free)
    A_log      (Hv,)            per-head decay param
    dt_bias    (Hv,)            per-head decay bias
    init_state (Hv, 128, 128)   carried recurrent state
    z          (T, Hv*128)      optional in_proj_z gate (head-major); enables the gated per-head RMSNorm
    gamma      (128,)           optional per-head norm weight (replicated); enables the gated norm
    eps        python float     RMSNorm epsilon (default 1e-6)
With ``z``/``gamma`` provided, ``attn_out`` is the gated per-head RMSNorm'd output (norm over head_dim
* silu(z)); with them None it is the raw head-major recurrence output (caller applies the norm/gate).
Outputs are FULL-shape in shared_hbm; each core writes its disjoint head/column/channel slice.
"""

import nki
import nki.language as nl

from ..components.conv import (
    P_MAX,
    conv_qkv_sbuf,
    kernel_assert,
    qkv_to_channel_partition,
)
from ..components.in_proj import in_proj_compose
from ..components.out_proj import out_proj_compose
from ..components.recurrence import gated_delta_rule_tkg


def fused_compose(
    qkv,
    conv_state,
    conv_weight,
    key_dim,
    a,
    b,
    A_log,
    dt_bias,
    init_state,
    attn_out,
    state_hbm,
    conv_cand,
    write_candidates,
    cand_is_3d,
    z=None,
    gamma=None,
    eps=None,
):
    """Compose the per-core conv -> recurrence pipeline (shared by both entrypoints).

    Derives the value-head shard once (the conv and recurrence both fold ``program_id``/``num_programs``
    to the same trace-time shard), runs ``conv_qkv_sbuf`` to produce this core's q/k/v SBUF tiles +
    scatter its conv-state slices, then drives the recurrence straight off those tiles. Hk_full /
    Hv_full are passed explicitly because the SBUF tiles carry only local heads (Hv_full from the state
    output's head axis, Hk_full from key_dim).

    When ``z``/``gamma`` are provided the recurrence emits the gated per-head RMSNorm'd output (norm
    over head_dim * silu(z)); with them None it emits the raw head-major output exactly as before.
    """
    conv_dim = qkv.shape[1]
    Hk_full = key_dim // P_MAX
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    kernel_assert(state_hbm.shape[-3] == Hv_full, "state head count must equal Hv")

    q_sbuf, k_sbuf, v_sbuf = conv_qkv_sbuf(
        qkv, conv_state, conv_weight, key_dim, conv_cand, cand_is_3d
    )
    gated_delta_rule_tkg(
        None,
        None,
        None,
        a,
        b,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        state_hbm,
        write_candidates,
        q_sbuf=q_sbuf,
        k_sbuf=k_sbuf,
        v_sbuf=v_sbuf,
        Hk_full=Hk_full,
        Hv_full=Hv_full,
        z=z,
        gamma=gamma,
        eps=eps,
    )


@nki.jit
def deltanet_fused_tkg_fwd(
    qkv,
    conv_state,
    conv_weight,
    key_dim,
    a,
    b,
    A_log,
    dt_bias,
    init_state,
    z=None,
    gamma=None,
    eps=1e-6,
):
    """Decode / commit (T=1): fused conv + recurrence in one launch.

    Returns:
        attn_out       (T, Hv*128) f32 -- recurrence output; gated per-head RMSNorm'd when z/gamma are
                       given, else raw head-major (caller RMSNorms/z-gates)
        final_state    (Hv, 128, 128) f32 -- recurrent state after the block token
        new_conv_state (conv_dim, K-1) -- committed new conv window (== conv_cand[0])
    Value-head sharded across cores; launch ``deltanet_fused_tkg_fwd[2](...)``.
    """
    T, conv_dim = qkv.shape
    state_w = conv_weight.shape[1] - 1
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    W_full = Hv_full * P_MAX
    attn_out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state = nl.ndarray(
        (Hv_full, P_MAX, P_MAX), dtype=nl.float32, buffer=nl.shared_hbm
    )
    new_conv_state = nl.ndarray(
        (conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm
    )
    fused_compose(
        qkv,
        conv_state,
        conv_weight,
        key_dim,
        a,
        b,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        final_state,
        new_conv_state,
        write_candidates=False,
        cand_is_3d=False,
        z=z,
        gamma=gamma,
        eps=eps,
    )
    return attn_out, final_state, new_conv_state


@nki.jit
def deltanet_fused_tkg_fwd_state(
    qkv,
    conv_state,
    conv_weight,
    key_dim,
    a,
    b,
    A_log,
    dt_bias,
    init_state,
    z=None,
    gamma=None,
    eps=1e-6,
):
    """Speculative verify (T>=2): fused conv + recurrence in one launch.

    ``candidate_states[t]`` / ``conv_cand[t]`` are the recurrent / conv state after consuming block
    token t; on a reject the host selects ``[accept_count - 1]``. Returns:
        attn_out         (T, Hv*128) f32 -- recurrence output; gated per-head RMSNorm'd when z/gamma
                         are given, else raw head-major
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
    conv_cand = nl.ndarray(
        (T, conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm
    )
    fused_compose(
        qkv,
        conv_state,
        conv_weight,
        key_dim,
        a,
        b,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        candidate_states,
        conv_cand,
        write_candidates=True,
        cand_is_3d=True,
        z=z,
        gamma=gamma,
        eps=eps,
    )
    return attn_out, candidate_states, conv_cand


def in_proj_fused_compose(
    hidden,
    proj_w,
    gamma,
    eps,
    conv_state,
    conv_weight,
    key_dim,
    A_log,
    dt_bias,
    init_state,
    attn_out,
    state_hbm,
    conv_cand,
    write_candidates,
    cand_is_3d,
    z_gamma,
    z_eps,
):
    """Compose in_proj -> conv -> recurrence with qkv/z/a/b kept in SBUF (no HBM round-trip).

    Stage 0 leaves the full projection proj_sb [T, I] in SBUF (segments at free offsets 0/qkv, conv_dim/z,
    +value_dim/a, +Hv_full/b). Bridge 1: qkv_to_channel_partition transposes qkv to the conv's
    channel-on-partition layout, fed via conv_qkv_sbuf(qkv=None). Bridge 2: gated_delta_rule_tkg sources
    a/b/z from proj_sb (SBUF->SBUF gathers) instead of HBM. in_proj is contraction-sharded while
    conv+recurrence are value-head-sharded; the full-per-core projection bridges the two.
    """
    conv_dim = conv_weight.shape[0]
    value_dim = conv_dim - 2 * key_dim
    Hk_full = key_dim // P_MAX
    Hv_full = value_dim // P_MAX
    kernel_assert(state_hbm.shape[-3] == Hv_full, "state head count must equal Hv")
    # Free-axis offsets of the projection segments (qkv | z | a | b, in that order).
    z_off = conv_dim
    a_off = conv_dim + value_dim
    b_off = a_off + Hv_full

    # Stage 0: fused input RMSNorm + 4-projection, kept in SBUF [T, I].
    proj_sb = in_proj_compose(hidden, proj_w, gamma, eps, output_in_sbuf=True)
    T = proj_sb.shape[0]

    # Bridge 1: transpose the qkv sub-block to channel-on-partition, then run the conv off SBUF.
    qkv_cp = qkv_to_channel_partition(proj_sb, conv_dim, T)
    q_sbuf, k_sbuf, v_sbuf = conv_qkv_sbuf(
        None,
        conv_state,
        conv_weight,
        key_dim,
        conv_cand,
        cand_is_3d,
        qkv_cp_sbuf=qkv_cp,
    )

    # Bridge 2: drive the recurrence off the conv SBUF tiles + source a/b/z from proj_sb.
    gated_delta_rule_tkg(
        None,
        None,
        None,
        None,
        None,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        state_hbm,
        write_candidates,
        q_sbuf=q_sbuf,
        k_sbuf=k_sbuf,
        v_sbuf=v_sbuf,
        Hk_full=Hk_full,
        Hv_full=Hv_full,
        z=None,
        gamma=z_gamma,
        eps=z_eps,
        proj_sb=proj_sb,
        a_off=a_off,
        b_off=b_off,
        z_off=z_off,
    )


@nki.jit
def deltanet_in_proj_fused_tkg_fwd(
    hidden,
    proj_w,
    gamma,
    eps,
    conv_state,
    conv_weight,
    key_dim,
    A_log,
    dt_bias,
    init_state,
    z_gamma,
    z_eps=1e-6,
):
    """Decode/commit (T=1): in_proj + conv + recurrence + gated norm in one SBUF-resident launch.

    gamma/eps = input RMSNorm; z_gamma/z_eps = gated per-head RMSNorm. Returns (attn_out [T,Hv*128] gated,
    final_state [Hv,128,128], new_conv_state [conv_dim,K-1]). Value-head sharded; launch [2].
    """
    conv_dim = conv_weight.shape[0]
    state_w = conv_weight.shape[1] - 1
    T = hidden.shape[0] * hidden.shape[1]
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    W_full = Hv_full * P_MAX
    attn_out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state = nl.ndarray(
        (Hv_full, P_MAX, P_MAX), dtype=nl.float32, buffer=nl.shared_hbm
    )
    new_conv_state = nl.ndarray(
        (conv_dim, state_w), dtype=conv_weight.dtype, buffer=nl.shared_hbm
    )
    in_proj_fused_compose(
        hidden,
        proj_w,
        gamma,
        eps,
        conv_state,
        conv_weight,
        key_dim,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        final_state,
        new_conv_state,
        write_candidates=False,
        cand_is_3d=False,
        z_gamma=z_gamma,
        z_eps=z_eps,
    )
    return attn_out, final_state, new_conv_state


@nki.jit
def deltanet_in_proj_fused_tkg_fwd_state(
    hidden,
    proj_w,
    gamma,
    eps,
    conv_state,
    conv_weight,
    key_dim,
    A_log,
    dt_bias,
    init_state,
    z_gamma,
    z_eps=1e-6,
):
    """Speculative verify (T>=2): like the decode path but emits per-token candidate states.

    Returns (attn_out [T,Hv*128] gated, candidate_states [T,Hv,128,128] after each token,
    conv_cand [T,conv_dim,K-1]); on reject the host selects [accept_count-1]. Value-head sharded; launch [2].
    """
    conv_dim = conv_weight.shape[0]
    state_w = conv_weight.shape[1] - 1
    T = hidden.shape[0] * hidden.shape[1]
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    W_full = Hv_full * P_MAX
    attn_out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)
    candidate_states = nl.ndarray(
        (T, Hv_full, P_MAX, P_MAX), dtype=nl.float32, buffer=nl.shared_hbm
    )
    conv_cand = nl.ndarray(
        (T, conv_dim, state_w), dtype=conv_weight.dtype, buffer=nl.shared_hbm
    )
    in_proj_fused_compose(
        hidden,
        proj_w,
        gamma,
        eps,
        conv_state,
        conv_weight,
        key_dim,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        candidate_states,
        conv_cand,
        write_candidates=True,
        cand_is_3d=True,
        z_gamma=z_gamma,
        z_eps=z_eps,
    )
    return attn_out, candidate_states, conv_cand


def out_proj_from_recurrence(attn_sb, out_w, T, W_core, out_in_sb=False):
    """Project this core's SBUF gated output [T, W_core] through out_proj_compose; returns o_out [T, hidden]
    (HBM) or the per-core SBUF [H0, H1_shard*T] H-shard when out_in_sb=True (megakernel residual add)."""
    return out_proj_compose(attn_sb[0:T, 0:W_core], out_w, out_in_sb=out_in_sb)


def attention_layer_compose(
    hidden,
    proj_w,
    gamma,
    eps,
    conv_state,
    conv_weight,
    key_dim,
    A_log,
    dt_bias,
    init_state,
    out_w,
    state_hbm,
    conv_cand,
    write_candidates,
    cand_is_3d,
    z_gamma,
    z_eps,
    out_in_sb=False,
    name_prefix="",
):
    """Compose in_proj -> conv -> recurrence -> gated norm into SBUF, then project to o_out.

    ``hidden`` may be HBM [B, S, H] or the megakernel's SBUF [128, T, 16] residual (qkv_tkg sniffs the
    buffer). ``out_in_sb`` returns the per-core SBUF [H0, H1_shard*T] o_proj partial (transposed_out)
    instead of HBM [T, hidden].
    """
    conv_dim = conv_weight.shape[0]
    value_dim = conv_dim - 2 * key_dim
    Hk_full = key_dim // P_MAX
    Hv_full = value_dim // P_MAX
    kernel_assert(state_hbm.shape[-3] == Hv_full, "state head count must equal Hv")
    z_off = conv_dim
    a_off = conv_dim + value_dim
    b_off = a_off + Hv_full

    n = nl.num_programs(0)
    Hv_core = Hv_full // n
    W_core = Hv_core * P_MAX
    W_full = Hv_full * P_MAX

    proj_sb = in_proj_compose(
        hidden, proj_w, gamma, eps, output_in_sbuf=True, name_prefix=name_prefix
    )
    T = proj_sb.shape[0]

    attn_shape = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.sbuf)
    attn_sb = nl.ndarray((T, W_core), dtype=out_w.dtype, buffer=nl.sbuf)

    qkv_cp = qkv_to_channel_partition(proj_sb, conv_dim, T)
    q_sbuf, k_sbuf, v_sbuf = conv_qkv_sbuf(
        None,
        conv_state,
        conv_weight,
        key_dim,
        conv_cand,
        cand_is_3d,
        qkv_cp_sbuf=qkv_cp,
    )
    gated_delta_rule_tkg(
        None,
        None,
        None,
        None,
        None,
        A_log,
        dt_bias,
        init_state,
        attn_shape,
        state_hbm,
        write_candidates,
        q_sbuf=q_sbuf,
        k_sbuf=k_sbuf,
        v_sbuf=v_sbuf,
        Hk_full=Hk_full,
        Hv_full=Hv_full,
        z=None,
        gamma=z_gamma,
        eps=z_eps,
        proj_sb=proj_sb,
        a_off=a_off,
        b_off=b_off,
        z_off=z_off,
        attn_sb_out=attn_sb,
    )
    return out_proj_from_recurrence(attn_sb, out_w, T, W_core, out_in_sb=out_in_sb)


@nki.jit
def deltanet_attention_layer(
    hidden,
    proj_w,
    gamma,
    eps,
    conv_state,
    conv_weight,
    key_dim,
    A_log,
    dt_bias,
    init_state,
    z_gamma,
    out_w,
    z_eps=1e-6,
):
    """Decode/commit (T=1): in_proj + conv + recurrence + gated norm + output projection in one launch.

    gamma/eps = input RMSNorm; z_gamma/z_eps = gated per-head RMSNorm; out_w = [value_dim, hidden]
    o_proj weight transpose. Returns (o_out [T, hidden] per-rank partial, final_state [Hv,128,128],
    new_conv_state [conv_dim,K-1]). Value-head sharded; launch [2].
    """
    conv_dim = conv_weight.shape[0]
    state_w = conv_weight.shape[1] - 1
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    final_state = nl.ndarray(
        (Hv_full, P_MAX, P_MAX), dtype=nl.float32, buffer=nl.shared_hbm
    )
    new_conv_state = nl.ndarray(
        (conv_dim, state_w), dtype=conv_weight.dtype, buffer=nl.shared_hbm
    )
    o_out = attention_layer_compose(
        hidden,
        proj_w,
        gamma,
        eps,
        conv_state,
        conv_weight,
        key_dim,
        A_log,
        dt_bias,
        init_state,
        out_w,
        final_state,
        new_conv_state,
        write_candidates=False,
        cand_is_3d=False,
        z_gamma=z_gamma,
        z_eps=z_eps,
    )
    return o_out, final_state, new_conv_state


@nki.jit
def deltanet_attention_layer_state(
    hidden,
    proj_w,
    gamma,
    eps,
    conv_state,
    conv_weight,
    key_dim,
    A_log,
    dt_bias,
    init_state,
    z_gamma,
    out_w,
    z_eps=1e-6,
):
    """Speculative verify (T>=2): like the decode path but emits per-token candidate states.

    Returns (o_out [T, hidden] per-rank partial, candidate_states [T,Hv,128,128], conv_cand
    [T,conv_dim,K-1]); the state outputs are unchanged. Value-head sharded; launch [2].
    """
    conv_dim = conv_weight.shape[0]
    state_w = conv_weight.shape[1] - 1
    T = hidden.shape[0] * hidden.shape[1]
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    candidate_states = nl.ndarray(
        (T, Hv_full, P_MAX, P_MAX), dtype=nl.float32, buffer=nl.shared_hbm
    )
    conv_cand = nl.ndarray(
        (T, conv_dim, state_w), dtype=conv_weight.dtype, buffer=nl.shared_hbm
    )
    o_out = attention_layer_compose(
        hidden,
        proj_w,
        gamma,
        eps,
        conv_state,
        conv_weight,
        key_dim,
        A_log,
        dt_bias,
        init_state,
        out_w,
        candidate_states,
        conv_cand,
        write_candidates=True,
        cand_is_3d=True,
        z_gamma=z_gamma,
        z_eps=z_eps,
    )
    return o_out, candidate_states, conv_cand
