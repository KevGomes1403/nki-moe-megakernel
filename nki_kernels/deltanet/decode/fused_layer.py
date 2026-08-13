# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused DeltaNet causal conv + gated delta-rule recurrence for token generation.

One launch runs the conv and the recurrence back-to-back with q/k/v handed over entirely in SBUF.
Sharded by value-head: each core runs a self-contained conv->recurrence pipeline over its own heads
with zero cross-core traffic, writing a disjoint slice of every full-shape output.

Per core, conv_qkv_sbuf runs the head-sharded conv over this core's 3 owned channel segments and
returns three partition-0-based SBUF tiles plus the pending conv-state stores; gated_delta_rule_tkg
then consumes those tiles directly. The gating tables, init_state seed, attn_out columns and
state-head writes stay full HBM tensors, sliced per core.

Four kernel variants, all value-head sharded and launched [2]:
  deltanet_fused_tkg_fwd            -- conv + recurrence                       decode / commit (T=1)
  deltanet_fused_tkg_fwd_state      -- same, with per-token candidate states    verify (T>=2)
  deltanet_attention_layer          -- + in_proj, gated norm, output proj      decode / commit
  deltanet_attention_layer_state    -- same, with per-token candidate states    verify

With z/gamma provided, attn_out is the gated per-head RMSNorm'd output; without them it is the raw
head-major output and the caller applies the norm/gate.
Input contract and rationale: specs/deltanet_tkg.md.
"""

import os

import nki
import nki.language as nl

from nkilib.core.utils.kernel_helpers import div_ceil

from ..components.conv import (
    P_MAX,
    conv_preload_taps,
    conv_qkv_sbuf,
    conv_state_store_pending,
    kernel_assert,
    qkv_to_channel_partition,
    shard_segments,
)
from ..components.in_proj import in_proj_compose
from ..components.out_proj import out_proj_compose, out_proj_from_loc
from ..components.recurrence import gated_delta_rule_tkg
from ..vendored.qkv_tkg import qkv_tkg_i_shard_drain

# Shard in_proj on its output columns instead of on the hidden dim, dropping the cross-core reduce.
_I_COLUMN_SHARD = os.environ.get("NKI_DELTANET_IN_PROJ_I_SHARD", "0") == "1"
# Emit the z column run's matmuls between the conv segments instead of inside the in_proj block.
_Z_DEFER = os.environ.get("NKI_DELTANET_IN_PROJ_Z_DEFER", "0") == "1"
# Read the projection weight from a host-side repack whose rows are partition-contiguous.
_PROJW_PACKED = os.environ.get("NKI_DELTANET_PROJW_PACKED", "0") == "1"
# Hand the recurrence's gated rows to o_proj head_dim-on-partition, dropping o_proj's transposes.
_ATTN_LOC = os.environ.get("NKI_DELTANET_ATTN_LOC", "0") == "1"


def in_proj_column_shard(conv_dim, key_dim, Hv_full, z_off, a_off, n=None, c=None):
    """This core's (start, size) projection column runs, or None when I-sharding is off.

    Its own q|k|v conv segments and z block, plus the whole 2*Hv_full-wide a|b pair -- taking
    both halves of a|b keeps that DMA one run per weight row instead of two 4-column fragments.
    n/c default to the launch grid; pass them to get another core's runs.
    """
    if not _I_COLUMN_SHARD:
        return None
    segments, Hv_loc, _ = shard_segments(conv_dim, key_dim, n=n, c=c)
    if c == None:
        c = nl.program_id(0)
    W = Hv_loc * P_MAX
    runs = []
    for seg in range(len(segments)):
        t0, n_tiles = segments[seg]
        runs.append((t0 * P_MAX, n_tiles * P_MAX))
    runs.append((z_off + c * W, W))
    runs.append((a_off, 2 * Hv_full))
    return runs


def in_proj_z_defer(conv_dim, key_dim, H1):
    """(deferred run indices, per-conv-segment h1 drain ranges), or (None, None) when off.

    The z run follows this core's 3 conv segments in in_proj_column_shard, and its H1 matmuls are
    split evenly over those segments.
    """
    if not (_I_COLUMN_SHARD and _Z_DEFER):
        return None, None
    n_seg = len(shard_segments(conv_dim, key_dim)[0])
    ranges = []
    for seg in range(n_seg):
        ranges.append((div_ceil(H1 * seg, n_seg), div_ceil(H1 * (seg + 1), n_seg)))
    return (n_seg,), ranges


def in_proj_z_drain_seg(drain, seg):
    """conv_qkv_sbuf seg_hook: emit the seg-th h1 slice of the deferred in_proj column run."""
    pending, ranges = drain
    qkv_tkg_i_shard_drain(pending, ranges[seg][0], ranges[seg][1])


def in_proj_packed_offset(H, conv_dim, key_dim, Hv_full, z_off, a_off):
    """Element offset of this core's blocks in the packed weight."""
    if not (_I_COLUMN_SHARD and _PROJW_PACKED):
        return 0
    n = nl.num_programs(0)
    c = nl.program_id(0)
    off = 0
    for cc in range(c):
        runs = in_proj_column_shard(conv_dim, key_dim, Hv_full, z_off, a_off, n=n, c=cc)
        for run in range(len(runs)):
            off += H * runs[run][1]
    return off


def pack_proj_w(proj_w, conv_dim, key_dim, num_cores):
    """Repack [H, I] proj_w into the packed weight the kernel reads, or None when not packing.

    Block (core, run) is (P_MAX, H1, width) at
    proj_w[(h1 // H1_shard) * P_MAX * H1_shard + p * H1_shard + h1 % H1_shard, col], flattened
    partition-major and concatenated run-then-core, so one DMA per block is 128 contiguous runs.
    """
    if not (_I_COLUMN_SHARD and _PROJW_PACKED):
        return None
    import torch

    H = proj_w.shape[0]
    H1 = H // P_MAX
    H1_shard = H1 // num_cores
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    z_off, a_off = conv_dim, conv_dim + (conv_dim - 2 * key_dim)

    p = torch.arange(P_MAX).reshape(P_MAX, 1)
    h1 = torch.arange(H1).reshape(1, H1)
    rows = ((h1 // H1_shard) * (P_MAX * H1_shard) + p * H1_shard + h1 % H1_shard).reshape(-1)
    gathered = proj_w[rows]  # [P_MAX * H1, I], partition-major over the shard-major H1 order

    blocks = []
    for c in range(num_cores):
        for start, size in in_proj_column_shard(
            conv_dim, key_dim, Hv_full, z_off, a_off, n=num_cores, c=c
        ):
            blocks.append(gathered[:, start : start + size].reshape(-1))
    return torch.cat(blocks).contiguous()


def owned_qkv_tiles(conv_dim, key_dim):
    """Global 128-channel tile indices of this core's q|k|v conv segments, or None when unsharded."""
    if not _I_COLUMN_SHARD:
        return None
    segments, _, _ = shard_segments(conv_dim, key_dim)
    tiles = []
    for seg in range(len(segments)):
        t0, n_tiles = segments[seg]
        for i in range(n_tiles):
            tiles.append(t0 + i)
    return tiles


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
    """Compose the per-core conv -> recurrence pipeline, shared by both entrypoints.

    Hk_full/Hv_full are passed explicitly because the SBUF tiles carry only local heads.
    """
    conv_dim = qkv.shape[1]
    Hk_full = key_dim // P_MAX
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    kernel_assert(state_hbm.shape[-3] == Hv_full, "state head count must equal Hv")

    q_sbuf, k_sbuf, v_sbuf, pending_cand = conv_qkv_sbuf(
        qkv, conv_state, conv_weight, key_dim, conv_cand, cand_is_3d
    )
    conv_state_store_pending(pending_cand)
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
        attn_out:       (T, Hv*128) f32, gated when z/gamma are given, else raw head-major.
        final_state:    (Hv, 128, 128) f32, recurrent state after the block token.
        new_conv_state: (conv_dim, K-1), the committed new conv window.
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

    candidate_states[t] and conv_cand[t] are the recurrent and conv state after block token t;
    on reject the host selects [accept_count - 1].
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
    """Compose in_proj -> conv -> recurrence with qkv/z/a/b kept in SBUF.

    The projection stays in SBUF as proj_sb [T, I], with the qkv/z/a/b segments at successive
    free-axis offsets. The conv reads a transposed view of the qkv segment; the recurrence sources
    a/b/z from the rest.

    in_proj is contraction-sharded while conv+recurrence are value-head sharded; the
    full-per-core projection is what bridges the two.
    """
    conv_dim = conv_weight.shape[0]
    value_dim = conv_dim - 2 * key_dim
    Hk_full = key_dim // P_MAX
    Hv_full = value_dim // P_MAX
    kernel_assert(state_hbm.shape[-3] == Hv_full, "state head count must equal Hv")
    # Free-axis offsets of the projection segments: qkv | z | a | b.
    z_off = conv_dim
    a_off = conv_dim + value_dim
    b_off = a_off + Hv_full

    # Layer-static taps/state: issue the DMAs before in_proj so they stream under its matmuls.
    conv_taps = conv_preload_taps(conv_state, conv_weight, key_dim)

    # Fused input RMSNorm + 4-way projection, kept in SBUF.
    proj_sb = in_proj_compose(
        hidden,
        proj_w,
        gamma,
        eps,
        output_in_sbuf=True,
        i_column_shard=in_proj_column_shard(conv_dim, key_dim, Hv_full, z_off, a_off),
    )
    T = proj_sb.shape[0]

    # Transpose the qkv sub-block to channel-on-partition, then run the conv off SBUF.
    qkv_cp = qkv_to_channel_partition(
        proj_sb, conv_dim, T, tiles=owned_qkv_tiles(conv_dim, key_dim)
    )
    q_sbuf, k_sbuf, v_sbuf, pending_cand = conv_qkv_sbuf(
        None,
        conv_state,
        conv_weight,
        key_dim,
        conv_cand,
        cand_is_3d,
        qkv_cp_sbuf=qkv_cp,
        preloaded=conv_taps,
    )

    # Drive the recurrence off the conv SBUF tiles, sourcing a/b/z from proj_sb.
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
    # Drained here rather than at the conv, to keep the transposes off the hand-off.
    conv_state_store_pending(pending_cand)


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
    """Decode / commit (T=1): in_proj + conv + recurrence + gated norm, SBUF-resident.

    gamma/eps is the input RMSNorm; z_gamma/z_eps the gated per-head RMSNorm.
    Returns (attn_out gated, final_state, new_conv_state).
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
    """Speculative verify (T>=2): like the decode path, but emits per-token candidate states.

    On reject the host selects [accept_count - 1].
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
    """Project this core's gated output through out_proj_compose.

    Returns HBM [T, hidden], or the per-core SBUF H-shard when out_in_sb is set.
    """
    if _ATTN_LOC:
        return out_proj_from_loc(attn_sb, out_w, out_in_sb=out_in_sb)
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
    proj_w_packed=None,
):
    """Compose in_proj -> conv -> recurrence -> gated norm into SBUF, then project to o_out.

    hidden may be HBM [B, S, H] or the megakernel's SBUF residual; qkv_tkg sniffs the buffer.
    out_in_sb returns the per-core SBUF o_proj partial instead of HBM [T, hidden].
    proj_w_packed, when given, supplies the projection weight in the packed layout.
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

    # Layer-static taps/state: issue the DMAs before in_proj so they stream under its matmuls.
    conv_taps = conv_preload_taps(conv_state, conv_weight, key_dim)

    defer_runs, drain_ranges = in_proj_z_defer(conv_dim, key_dim, proj_w.shape[0] // P_MAX)
    proj_out = in_proj_compose(
        hidden,
        proj_w,
        gamma,
        eps,
        output_in_sbuf=True,
        name_prefix=name_prefix,
        i_column_shard=in_proj_column_shard(conv_dim, key_dim, Hv_full, z_off, a_off),
        i_column_shard_defer=defer_runs,
        qkv_w_packed=proj_w_packed,
        qkv_w_packed_offset=in_proj_packed_offset(
            proj_w.shape[0], conv_dim, key_dim, Hv_full, z_off, a_off
        ),
    )
    if defer_runs == None:
        proj_sb = proj_out
        seg_hook = None
    else:
        proj_sb = proj_out[0]
        seg_hook = (in_proj_z_drain_seg, (proj_out[1], drain_ranges))
    T = proj_sb.shape[0]

    attn_shape = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.sbuf)
    if _ATTN_LOC:
        attn_sb = nl.ndarray((P_MAX, Hv_core, T), dtype=out_w.dtype, buffer=nl.sbuf)
    else:
        attn_sb = nl.ndarray((T, W_core), dtype=out_w.dtype, buffer=nl.sbuf)

    qkv_cp = qkv_to_channel_partition(
        proj_sb, conv_dim, T, tiles=owned_qkv_tiles(conv_dim, key_dim)
    )
    q_sbuf, k_sbuf, v_sbuf, pending_cand = conv_qkv_sbuf(
        None,
        conv_state,
        conv_weight,
        key_dim,
        conv_cand,
        cand_is_3d,
        qkv_cp_sbuf=qkv_cp,
        preloaded=conv_taps,
        seg_hook=seg_hook,
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
    # Drained here rather than at the conv, to keep the transposes off the hand-off.
    conv_state_store_pending(pending_cand)
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
    proj_w_packed=None,
):
    """Decode / commit (T=1): the whole DeltaNet layer in one launch.

    in_proj + conv + recurrence + gated norm + output projection. gamma/eps is the input RMSNorm,
    z_gamma/z_eps the gated per-head RMSNorm, out_w the [value_dim, hidden] o_proj weight transpose.
    proj_w_packed, when given, replaces proj_w as the in_proj weight source.
    Returns (o_out per-rank partial, final_state, new_conv_state).
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
        proj_w_packed=proj_w_packed,
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
    proj_w_packed=None,
):
    """Speculative verify (T>=2): like the decode path, but emits per-token candidate states."""
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
        proj_w_packed=proj_w_packed,
    )
    return o_out, candidate_states, conv_cand
