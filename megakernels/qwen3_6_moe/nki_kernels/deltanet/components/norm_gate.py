# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated per-head RMSNorm for the DeltaNet recurrence output (matches Qwen3_5MoeRMSNormGated).

Per (token, value-head): out = (x * rsqrt(mean(x^2) + eps)) * gamma * silu(z), normalized over the
128-wide head_dim. Layout A keeps head_dim on the free axis so the per-head reduce needs no transpose.
norm_gate_row is the composable SBUF helper used at the recurrence's per-token output seam;
deltanet_gated_rmsnorm is a thin HBM harness exposing the same math for unit testing.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# head_dim (value-head width); equals the partition-dim max.
P_MAX = 128


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def norm_gate_row(o_row, z_row, gamma_sb, eps, d):
    """One token's gated per-head RMSNorm (Layout A): o_row/z_row [1,W_core] SBUF, gamma_sb [1,d] -> gated [1,W_core] = RMSNorm(o_row)*silu(z)."""
    W = o_row.shape[1]
    kernel_assert(W % d == 0, "W_core must be a multiple of head_dim")
    kernel_assert(d == P_MAX, "head_dim must equal P_MAX")
    Hv = W // d

    # 3D head-major views of the [1, W] rows: [partition=1, head, j].
    o_3d = o_row.ap(pattern=[[W, 1], [d, Hv], [1, d]], offset=0)

    # Per-head sum-of-squares: free-axis reduce over the innermost d (activation_reduce can't do sub-block reduces).
    sq = nl.ndarray((1, Hv, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=o_3d, bias=None, scale=1.0)
    sumsq = nl.ndarray((1, Hv, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sumsq, op=nl.add, data=sq, axis=(2,))

    # rsqrt(sumsq * (1/d) + eps): fold mean-scale + eps into one Scalar-engine rsqrt.
    inv = nl.ndarray((1, Hv, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv, op=nl.rsqrt, data=sumsq, bias=eps, scale=1.0 / d)

    # Normalize: x_j * inv_h (broadcast inv over j), then * gamma_j (broadcast gamma over heads).
    out = nl.ndarray((1, Hv, d), dtype=nl.float32, buffer=nl.sbuf)
    inv_bc = inv.ap(pattern=[[Hv, 1], [1, Hv], [0, d]], offset=0)
    nisa.tensor_tensor(dst=out, data1=o_3d, data2=inv_bc, op=nl.multiply)
    gamma_bc = gamma_sb.ap(pattern=[[d, 1], [0, Hv], [1, d]], offset=0)
    nisa.tensor_tensor(dst=out, data1=out, data2=gamma_bc, op=nl.multiply)

    # Gate by silu(z): o_j *= z_j * sigmoid(z_j).
    sz = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sz, op=nl.silu, data=z_row, bias=None, scale=1.0)
    out_flat = out.ap(pattern=[[W, 1], [1, W]], offset=0)
    gated = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gated, data1=out_flat, data2=sz, op=nl.multiply)
    return gated


@nki.jit
def deltanet_gated_rmsnorm(attn_raw, gamma, z, eps):
    """Standalone HBM harness for testing: attn_raw/z (T,W) head-major, gamma (d,) -> (T,W) = RMSNorm(attn_raw)*silu(z). Launch [n]."""
    T, W_full = attn_raw.shape
    d = gamma.shape[0]
    Hv_full = W_full // d
    out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)

    # Value-head SPMD shard: this core owns 1/n of the heads (disjoint column slice of [T, W_full]).
    n = nl.num_programs(0)
    c = nl.program_id(0)
    kernel_assert(Hv_full % n == 0, "v-heads must divide across cores")
    Hv = Hv_full // n
    W = Hv * d
    col_off = c * W

    # gamma is replicated; load once into [1, d].
    gamma_sb = nl.ndarray((1, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_sb, src=gamma.ap(pattern=[[d, 1], [1, d]], offset=0))

    for t in nl.static_range(T):
        o_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=o_row[0:1, 0:W],
            src=attn_raw.ap(pattern=[[W_full, 1], [1, W]], offset=t * W_full + col_off),
        )
        z_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=z_row[0:1, 0:W],
            src=z.ap(pattern=[[W_full, 1], [1, W]], offset=t * W_full + col_off),
        )
        gated = norm_gate_row(o_row, z_row, gamma_sb, eps, d)
        nisa.dma_copy(
            dst=out.ap(pattern=[[W_full, 1], [1, W]], offset=t * W_full + col_off),
            src=gated[0:1, 0:W],
        )
    return out
