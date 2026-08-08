# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GQA attention output projection (o_proj) for token generation (head_dim=256).

Thin composable over nkilib's output_projection_tkg. The GQA o_proj contracts the attention
output's value_dim (q_heads * head_dim) down to hidden. output_projection_tkg folds N sub-heads,
each at most 128 wide, and PSUM-accumulates them, so a head_dim=256 q-head is presented as two
128-wide sub-heads. With q_heads=4 that is 4*2 = 8 sub-heads of 128 -- structurally identical to
the DeltaNet o_proj (8 value-heads of 128); this wraps output_projection_tkg, it does not patch it.

Input layout matches the Phase 4 attention core's output `out_sb [128, D_TILES, Tq]`: head_dim is
already on the partition axis (split into D_TILES tiles of 128), so -- unlike the DeltaNet o_proj --
no per-head PE transpose is needed. Each (q-head, d-tile) pair is one 128-wide sub-head already sitting
on the partition axis; this composable only reorders the free axes into output_projection_tkg's
`attention [d=128, B=1, N, T]` sub-head layout, gathers the other LNC core's sub-heads via sendrecv,
then runs the H-sharded matmul. The TP all-reduce of the per-rank o_proj partial is deferred; this
returns the per-rank partial [T, hidden].

Sub-head ordering (q-head major, d-tile minor): global sub-head n = h_global * D_TILES + d_tile
maps to value_dim block [n*128, (n+1)*128). The o_proj weight `out_w [value_dim, hidden]` is row-indexed
by value_dim, so out_w[n*128:(n+1)*128] is exactly sub-head n's weight -- it already matches
output_projection_tkg's `weight [N*D, H]` indexing [n*D + d, h] and is passed through unreshaped.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.output_projection.output_projection_tkg import output_projection_tkg
from nkilib.core.utils.common_types import QuantizationType

# head_dim sub-head width; equals the partition-dim max (a head_dim=256 q-head -> 2 sub-heads of 128).
P_MAX = 128


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def out_proj_compose(attn_sb, out_w, T, gate_sb=None, sbm=None):
    """Per-rank GQA attention output projection.

    Args:
        attn_sb: [P_MAX, D_TILES, Tq] SBUF, this core's query heads in the Phase 4 attention core's
            output layout (head_dim on partition). Element [d_in, d_tile, h_local*T + t] =
            attn_out[t, h_local, d_tile*P_MAX + d_in]. Tq = qh_local * T (head-major); the head_dim of
            256 is split into D_TILES = 2 partition tiles of 128, so (h_local, d_tile) names a 128-wide
            sub-head already on the partition axis.
        out_w:  [value_dim, hidden] HBM, transpose of the o_proj nn.Linear weight (value_dim first).
            Row-indexed by value_dim; out_w[n*P_MAX:(n+1)*P_MAX] is global sub-head n's weight.
        T:      decode width (active tokens). qh_local = Tq // T query heads on this core.
        gate_sb: optional [P_MAX, D_TILES, Tq] SBUF, sigmoid output gate in the SAME head_dim-on-partition
            layout as attn_sb. When provided, the attention input is gated elementwise before the matmul:
            gated = attn_sb * sigmoid(gate_sb). Default None (no gate), keeping the core o_proj path clean.
        sbm:    optional BufferManager passed through to output_projection_tkg.

    Returns:
        o_out: [T, hidden] HBM -- per-rank o_proj PARTIAL. Each core writes its disjoint hidden/n shard;
            the full [T, hidden] is complete on return. The TP all-reduce across ranks is deferred.

    Steps: optional sigmoid gate -> reorder (d_tile, h_local) free axes into sub-head order [d, N_core, T]
    -> assemble the global [d, 1, N, T] (this core's sub-heads at [c*N_core, (c+1)*N_core), other core's
    gathered via sendrecv) -> output_projection_tkg(OUT_IN_SB=False, TRANSPOSE_OUT=False, NONE).
    """
    _, D_TILES, Tq = attn_sb.shape
    value_dim, hidden = out_w.shape

    kernel_assert(Tq % T == 0, "Tq must be a multiple of T (heads grouped, T per head)")
    qh_local = Tq // T
    N_core = qh_local * D_TILES

    n = nl.num_programs(0)
    c = nl.program_id(0)
    N = N_core * n
    kernel_assert(
        N * P_MAX == value_dim,
        "value_dim must equal N * P_MAX (all cores' sub-heads of 128)",
    )
    kernel_assert(T <= P_MAX, "B*S = T must not exceed P_MAX")
    kernel_assert(hidden % n == 0, "hidden must be divisible by the LNC core count")

    # Optional output gate: gated = attn_sb * sigmoid(gate_sb), elementwise in the head_dim-on-partition
    # layout (layout-agnostic). In a megakernel the gate projection emits this same layout.
    if gate_sb != None:
        kernel_assert(
            gate_sb.shape == attn_sb.shape, "gate_sb must match attn_sb shape/layout"
        )
        sig = nl.ndarray(attn_sb.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sig, op=nl.sigmoid, data=gate_sb)
        gated = nl.ndarray(attn_sb.shape, dtype=attn_sb.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gated, data1=attn_sb, data2=sig, op=nl.multiply)
        attn_src = gated
    else:
        attn_src = attn_sb

    # Reorder the free axes into sub-head order [d, N_core, T]: local sub-head n_local =
    # h_local*D_TILES + d_tile (q-head major, d-tile minor). head_dim is already on the partition
    # axis, so this is a pure SBUF reorder -- no transpose (the key simplification vs DeltaNet o_proj).
    attn_loc = nl.ndarray((P_MAX, N_core, T), dtype=attn_sb.dtype, buffer=nl.sbuf)
    for h_local in nl.affine_range(qh_local):
        for d_tile in nl.affine_range(D_TILES):
            n_local = h_local * D_TILES + d_tile
            nisa.tensor_copy(
                dst=attn_loc[0:P_MAX, n_local, 0:T],
                src=attn_src[0:P_MAX, d_tile, h_local * T : h_local * T + T],
            )

    # Assemble all N sub-heads at GLOBAL positions: core c's sub-heads at [c*N_core, (c+1)*N_core).
    # Because q-heads are sharded as contiguous blocks, global sub-head = c*N_core + n_local.
    attn_full = nl.ndarray((P_MAX, 1, N, T), dtype=attn_sb.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=attn_full[0:P_MAX, 0, c * N_core : (c + 1) * N_core, 0:T],
        src=attn_loc[0:P_MAX, 0:N_core, 0:T],
    )

    if n > 1:
        # Exchange local sub-heads with the other core; place them at the other core's positions.
        other = 1 - c
        nisa.sendrecv(
            src=attn_loc[0:P_MAX, 0:N_core, 0:T],
            dst=attn_full[0:P_MAX, 0, other * N_core : (other + 1) * N_core, 0:T],
            send_to_rank=other,
            recv_from_rank=other,
            pipe_id=0,
        )

    # output_projection_tkg infers N=N and D=P_MAX from attn_full.shape, checks
    # weight.shape[0] == N*P_MAX == value_dim, and H-shards the [T, hidden] output across cores.
    return output_projection_tkg(
        attention=attn_full,
        weight=out_w,
        bias=None,
        quantization_type=QuantizationType.NONE,
        TRANSPOSE_OUT=False,
        OUT_IN_SB=False,
        sbm=sbm,
    )
