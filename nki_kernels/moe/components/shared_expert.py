# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-expert path of the Qwen3.6-A3B fused MoE layer (token generation), SBUF-resident.

A plain SwiGLU FFN -- down(silu(gate(x)) * up(x)) -- on the SAME SBUF-resident ``normed_sb [H0,T,H1]``
the routed path consumes (zero HBM round-trip). Returns the per-rank partial over H (down_proj without
internal reduce); the sigma-gate, gated sum and combined TP all-reduce are the NEXT slice.

Reuses nkilib's mlp_tkg matmul machinery -- the exact ``process_gate_up_projection`` /
``process_down_projection`` primitives the routed experts (moe_tkg) use, so NO matmul is written from
scratch. The public ``mlp_tkg`` WRAPPER is not used: its store_output_in_sbuf + NO_NORM path is
unexercised in nkilib and broken for SBUF input (an unbalanced ``pop_heap`` with no matching alloc, plus
an unimported ``Logger`` on the sbm=None branch, and it forbids the auto-alloc SBM that SBUF input needs
to avoid colliding with the compiler-allocated normed tile). moe_tkg sidesteps the wrapper the same way
(selective_expert_impl drives the primitives directly with an auto-alloc SBM); we mirror that.

Per-LNC-core layout matches moe_tkg's ``routed_local`` so the future gated sum is a plain per-core add.
We mirror moe_tkg selective's TOKEN-shard decision (selective_expert_impl.py:117-142):
  * token-shard (cores>1, T>1, not big-config): each core runs the H-unsharded primitives on its token
    slice ``[T_offset:T_offset+T_per_shard]`` and writes ONLY that slice into the [H0,T,H1] output
    (selective_expert_impl.py:383-386 idiom) -- byte-for-byte routed_local.
  * single-core (cores==1): full [H0,T,H1] on the one core -- matches routed_local (num_shards=1).
``shard_on_h_disabled`` is kept True in ALL cases: the H-shard path moe_tkg takes at cores>1/T==1 needs a
genuinely per-core H-sharded norm input, but the token-sharded RMSNorm emits the SAME full tile on both
cores at T==1, so that path double-counts the gate/up sendrecv -- moe_tkg's own routed path fails there
identically. cores=2/T=1 therefore stays correct full-compute (both cores), which DIVERGES from
routed_local's (broken) H-shard layout -- the remaining reconciliation for the combine slice, which must
supply an H-sharded norm to fix routed AND shared together (see moe_layer.md).
Weights are rank-replicated; ``shared_local`` stays the per-rank partial (cross-rank all-reduce deferred).
"""

import nki.isa as nisa
import nki.language as nl

from nkilib.core.mlp.mlp_parameters import MLPParameters
from nkilib.core.mlp.mlp_tkg.mlp_tkg_constants import MLPTKGConstants
from nkilib.core.mlp.mlp_tkg.mlp_tkg_down_projection import process_down_projection
from nkilib.core.mlp.mlp_tkg.mlp_tkg_gate_up_projection import (
    process_gate_up_projection,
)
from nkilib.core.mlp.mlp_tkg.mlp_tkg_utils import (
    alloc_tensor_view,
    convert_params_to_views,
    transpose_store_sbuf_copy,
)
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.common_types import ActFnType, NormType
from nkilib.core.utils.kernel_helpers import (
    div_ceil,
    get_verified_program_sharding_info,
)
from nkilib.core.utils.logging import get_logger

from .post_attn_norm import post_attn_rmsnorm_compose
from .routed_experts import moe_h_shard_mode

_MLP_SBUF_BUDGET = 200 * 1024


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def moe_tkg_shard_decision(T, H, intermediate):
    """Per-LNC-core TOKEN-shard geometry mirroring moe_tkg selective (selective_expert_impl.py:117-142).

    Returns (shard_on_T, T_offset, T_per_shard). ``shard_on_T`` is True only when moe_tkg token-shards
    (cores>1, T>1, not big-config); each core then owns tokens [T_offset:T_offset+T_per_shard] over full
    H. Otherwise (cores==1, or cores>1/T==1) each core computes the full [H0,T,H1] tile -- shard_on_h stays
    disabled everywhere (module docstring: the H-shard path needs an H-sharded norm the token-sharded
    RMSNorm cannot supply at T==1).
    """
    _, n_prgs, shard_id = get_verified_program_sharding_info("moe_shared", (0, 1), 2)
    single_core_forced, _ = moe_h_shard_mode(T, H, intermediate, n_prgs)
    shard_on_T = (
        single_core_forced and n_prgs > 1
    )  # moe_tkg token-shards only when cores>1 and T>1
    if shard_on_T:
        T_first = T // n_prgs
        T_per_shard = T_first if shard_id == 0 else T - T_first
        T_offset = 0 if shard_id == 0 else T_first
    else:
        T_per_shard, T_offset = T, 0
    return shard_on_T, T_offset, T_per_shard


def _mlp_tkg_gate_up_down(params, sbm, dims, T_offset):
    """mlp_tkg's SBUF-in NO_NORM core over ``dims.T`` tokens starting at ``T_offset``.

    Reuses the nkilib primitives directly (the public wrapper's SBUF+NO_NORM path pops an unallocated
    heap frame). gate_up slices the full normed tile to ``[T_offset:T_offset+dims.T]``; with
    shard_on_h_disabled each core computes the full H per-rank partial for its token slice. Returns the
    down tile [H0, H1_shard(=H1), dims.T] SBUF.
    """
    sbm.open_scope()
    input_sb = (
        params.hidden_tensor
    )  # full SBUF normed tile [H0, T, H1]; gate_up slices T by T_offset

    gate_up_sb = alloc_tensor_view(
        sbm,
        (dims.I0, div_ceil(dims.I, dims.I0), dims.T),
        dtype=params.hidden_tensor.dtype,
        buffer=nl.sbuf,
        name="gate_up_sbuf",
    )
    sbm.open_scope()
    gate_tile_info = process_gate_up_projection(
        hidden=input_sb,
        output=gate_up_sb,
        params=params,
        dims=dims,
        sbm=sbm,
        T_offset=T_offset,
    )
    sbm.close_scope()

    down_sb = alloc_tensor_view(
        sbm,
        (dims.H0, dims.H1_shard, dims.T),
        dtype=params.hidden_tensor.dtype,
        buffer=nl.sbuf,
        name="down_sbuf",
    )
    sbm.open_scope()
    process_down_projection(
        hidden=gate_up_sb,
        output=down_sb,
        params=params,
        dims=dims,
        gate_tile_info=gate_tile_info,
        sbm=sbm,
    )
    sbm.close_scope()
    return down_sb


def shared_expert_compose(normed_sb, gate_w, up_w, down_w, output_in_sbuf=True):
    """SwiGLU shared expert on an SBUF-resident normed tile via nkilib mlp_tkg (zero HBM round-trip).

    Args:
        normed_sb:      [H0=128, T, H1=H//128] SBUF post-attn-normed hidden in the full layout-0
                        (single_core_forced) tile -- shard_on_h is disabled, so each core reads full H.
        gate_w:         [H, I_s] HBM gate weight (contraction-first; load-transposed from stored [I_s,H]).
        up_w:           [H, I_s] HBM up weight (contraction-first; load-transposed from stored [I_s,H]).
        down_w:         [I_s, H] HBM down weight (load-transposed from stored [H, I_s]).
        output_in_sbuf: True -> shared_local SBUF [H0,T,H1] (megakernel API, matches routed_local);
                        False -> HBM [T,H] natural (isolation authoritative gate).

    Returns:
        shared_local: SBUF [H0,T,H1] (output_in_sbuf=True) or HBM [T,H] (False) -- per-rank partial with
                      the SAME per-LNC-core shard layout as moe_tkg's routed_local.
    """
    H0, T, H1 = normed_sb.shape
    H = H0 * H1
    I_s = up_w.shape[1]
    kernel_assert(
        tuple(gate_w.shape) == (H, I_s), "gate_w must be [H, I_s] matching normed_sb H"
    )
    kernel_assert(
        tuple(up_w.shape) == (H, I_s), "up_w must be [H, I_s] matching normed_sb H"
    )
    kernel_assert(tuple(down_w.shape) == (I_s, H), "down_w must be [I_s, H]")

    shard_on_T, T_offset, T_per_shard = moe_tkg_shard_decision(T, H, I_s)

    params = MLPParameters(
        hidden_tensor=normed_sb,  # SBUF [H0,T,H1] -> input_in_sbuf auto-detected
        gate_proj_weights_tensor=gate_w,  # [H, I_s]
        up_proj_weights_tensor=up_w,  # [H, I_s]
        down_proj_weights_tensor=down_w,  # [I_s, H] -> derives H, I_s
        activation_fn=ActFnType.SiLU,
        normalization_type=NormType.NO_NORM,
        store_output_in_sbuf=output_in_sbuf,
        shard_on_h_disabled=True,  # each core computes full H; token-sharded over T when shard_on_T
    )

    sbm = SbufManager(
        0, _MLP_SBUF_BUDGET, get_logger("shared_expert"), use_auto_alloc=True
    )
    sbm.set_name_prefix("shared_mlp_")

    convert_params_to_views(params)
    dims = MLPTKGConstants.calculate_constants(params)
    if shard_on_T:
        dims.T = T_per_shard  # each core computes ONLY its token slice

    down_sb = _mlp_tkg_gate_up_down(
        params, sbm, dims, T_offset
    )  # [H0, H1_shard, dims.T]

    if not output_in_sbuf:
        out_hbm = nl.ndarray((T, H), dtype=normed_sb.dtype, buffer=nl.shared_hbm)
        transpose_store_sbuf_copy(
            down_sb.base_tensor, out_hbm, dims, normed_sb.dtype, sbm, T_offset
        )
        sbm.close_scope()
        return out_hbm

    # SBUF store mirrors selective_expert_impl.py:383-386 -- each core writes ONLY its own shard slice.
    shared_local = nl.ndarray((H0, T, H1), dtype=normed_sb.dtype, buffer=nl.sbuf)
    for h1_idx in range(dims.H1_shard):
        nisa.tensor_copy(
            dst=shared_local[:, T_offset : T_offset + dims.T, h1_idx],
            src=down_sb.base_tensor[:, h1_idx, :],
        )
    sbm.close_scope()
    return shared_local


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

    Isolation convenience: norms into the full layout-0 [H0,T,H1] tile (single_core_forced) that the
    shard_on_h-disabled shared expert consumes on every LNC core with zero HBM round-trip -- the token
    slicing happens inside ``shared_expert_compose`` from the same full tile.

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
        hidden, gamma, eps, hidden_actual, single_core_forced=True
    )
    shared_local = shared_expert_compose(
        normed_sb, gate_w, up_w, down_w, output_in_sbuf=output_in_sbuf
    )
    return shared_local, normed_sb
