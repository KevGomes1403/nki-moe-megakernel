# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeltaNet attention output projection (o_proj) for token generation.

Thin composable over nkilib's output_projection_tkg. The DeltaNet recurrence + gated norm are
value-head sharded across the LNC cores, so each core holds only its Hv_core heads of the gated
output, but the o_proj matmul contracts over ALL value heads. Design A bridges this with a tiny
cross-LNC sendrecv gather of the (small) attention tensor pre-matmul, then lets output_projection_tkg
H-shard the (large) output across cores -- no LNC reduce of the result (disjoint H-shards).

Input is (head-major [T, W_core], element [t, h_local*d+j]); the composable transposes each
head to head_dim-on-partition and assembles [d, 1, Hv, T] (output_projection_tkg's `attention`
layout) before the matmul. The TP all-reduce of the per-rank o_proj partial is deferred; this returns
the per-rank partial [T, hidden].
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.output_projection.output_projection_tkg import output_projection_tkg
from nkilib.core.utils.common_types import QuantizationType

# head_dim (value-head width); equals the partition-dim max.
P_MAX = 128


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def out_proj_compose(attn_sb, out_w, sbm=None):
    """Per-rank DeltaNet output projection

    Args:
        attn_sb: [T, W_core] SBUF, this core's value heads in Layout A (head-major,
            element [t, h_local*d + j] = local value-head h_local, head_dim j).
        out_w:   [value_dim, hidden] HBM, transpose of the o_proj nn.Linear weight.
        sbm:     optional BufferManager passed through to output_projection_tkg.

    Returns:
        o_out: [T, hidden] HBM -- per-rank o_proj PARTIAL. Each core writes its disjoint
            hidden/n shard; the full [T, hidden] is complete on return.

    Steps: cross-LNC sendrecv gather of all heads -> transpose Layout A to [d, 1, Hv, T]
    (head_dim on partition) -> output_projection_tkg(OUT_IN_SB=False, TRANSPOSE_OUT=False, NONE).
    """
    T, W_core = attn_sb.shape
    value_dim, hidden = out_w.shape
    d = P_MAX
    kernel_assert(W_core % d == 0, "W_core must be a multiple of head_dim")

    n = nl.num_programs(0)
    c = nl.program_id(0)
    Hv_core = W_core // d
    Hv = Hv_core * n
    kernel_assert(
        Hv * d == value_dim, "value_dim must equal Hv * head_dim (all cores' heads)"
    )
    kernel_assert(T <= P_MAX, "B*S = T must not exceed P_MAX")
    kernel_assert(hidden % n == 0, "hidden must be divisible by the LNC core count")

    # This core's local heads, transposed to head_dim-on-partition: [d, Hv_core, T] with element
    # [j, h_local, t]. nc_transpose maps a [T, d] slice (T on partition) to a [d, T] PSUM tile.
    attn_loc = nl.ndarray((d, Hv_core, T), dtype=attn_sb.dtype, buffer=nl.sbuf)
    for h_local in nl.affine_range(Hv_core):
        # nc_transpose uses matmul transpose mode; on gen3+ the PSUM dst dtype must match the input.
        head_t = nl.ndarray((d, T), dtype=attn_sb.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=head_t, data=attn_sb[0:T, h_local * d : (h_local + 1) * d]
        )
        nisa.tensor_copy(dst=attn_loc[0:d, h_local, 0:T], src=head_t[0:d, 0:T])

    # Assemble all Hv heads on this core at GLOBAL head positions: core c's heads at [c*Hv_core, ...).
    attn_full = nl.ndarray((d, 1, Hv, T), dtype=attn_sb.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=attn_full[0:d, 0, c * Hv_core : (c + 1) * Hv_core, 0:T],
        src=attn_loc[0:d, 0:Hv_core, 0:T],
    )

    if n > 1:
        # Exchange local heads with the other core; place them at the other core's head positions.
        other = 1 - c
        nisa.sendrecv(
            src=attn_loc[0:d, 0:Hv_core, 0:T],
            dst=attn_full[0:d, 0, other * Hv_core : (other + 1) * Hv_core, 0:T],
            send_to_rank=other,
            recv_from_rank=other,
            pipe_id=0,
        )

    # output_projection_tkg H-shards the output across cores by program_id and writes each core's
    # [T, hidden/n] slice into the full [T, hidden] shared_hbm output.
    return output_projection_tkg(
        attention=attn_full,
        weight=out_w,
        bias=None,
        quantization_type=QuantizationType.NONE,
        TRANSPOSE_OUT=False,
        OUT_IN_SB=False,
        sbm=sbm,
    )
