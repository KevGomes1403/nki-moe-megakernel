# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-RoPE q/k RMSNorm for the GQA attention block (head_dim=256, token generation).

Phase 2 of the GQA TKG pipeline (qkv_proj -> qk_norm -> rope -> attention -> out_proj). Consumes the
head-major NBSd QKV tile produced by Phase 1 and applies RMSNorm over head_dim to each Q head and each
K head; the V heads are passed through unchanged. RoPE is applied downstream.

Layout / why this is a FREE-AXIS reduce:
    The QKV tile is [T, N, D] in SBUF -- T = B*S tokens on the partition axis, N heads then head_dim D
    on the free axis (head-major, order [q0..q_{Q-1} | k0..k_{K-1} | v0..v_{K-1}]). Because head_dim D
    sits entirely on the FREE axis (D=256 <= the 512 free limit), RMSNorm over D is a single free-axis
    reduction per token -- no partition tiling, head_dim=256 needs no splitting. This is exactly why the
    norm is applied in this layout rather than a head_dim-on-partition RMSNorm (which would cap d at 128).

Math per normed head (standard-weight RMSNorm; gamma is the layernorm weight, NOT 1+weight):
    y[t, :] = x[t, :] * rsqrt(mean_D(x[t, :]^2) + eps) * gamma
    bf16 (or fp32) IO; the square, reduction, rsqrt, and scale run in fp32.

The composable is SBUF-in / SBUF-out (megakernel-ready): Phase 1's [B*S, I] SBUF result is the same
buffer reshaped to [T, N, D] (I = N*D head-major), so the megakernel passes it here with no copy.
"""

import nki.isa as nisa
import nki.language as nl

# Qwen3.6 GQA per-rank (TP=4) head config: head_dim 256, 4 Q heads, 1 K head; N = Q + 2*K heads.
HEAD_DIM = 256
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
NUM_HEADS = NUM_Q_HEADS + 2 * NUM_KV_HEADS  # 6 -> [q0|q1|q2|q3|k0|v0] on the N axis


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def rms_norm_over_free(x_head, gamma_head, eps_t, out_head):
    """Free-axis RMSNorm of one head tile [T, D] over head_dim D (D on the free axis).

    y[t, :] = x[t, :] * rsqrt(mean_D(x[t, :]^2) + eps) * gamma   (fp32 reduce/scale; out in IO dtype).

    Args:
        x_head:     [T, D] SBUF. One head's tokens (T on partition, head_dim D on free).
        gamma_head: [T, D] SBUF. Per-head_dim weight, partition-broadcast to the T token rows.
        eps_t:      [T, 1] SBUF fp32. RMSNorm epsilon (memset once, shared across heads).
        out_head:   [T, D] SBUF. Normalized output (written), same dtype as x_head.
    """
    T, D = x_head.shape

    # Sum of squares over the free axis, accumulated in fp32.
    sq = nl.ndarray((T, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=x_head)
    ss = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=[1], keepdims=True)

    # inv = rsqrt(ss * (1/D) + eps) = 1 / sqrt(mean_D(x^2) + eps).
    inv = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv, op=nl.rsqrt, data=ss, scale=1.0 / D, bias=eps_t)

    # y = (x * inv) * gamma in one Vector-engine pass: inv is the per-token scalar [T, 1]
    # (free-broadcast), gamma is the full [T, D] weight. fp32 math, cast to out_head.dtype.
    nisa.scalar_tensor_tensor(
        dst=out_head,
        data=x_head,
        op0=nl.multiply,
        operand0=inv,
        op1=nl.multiply,
        operand1=gamma_head,
    )


def qk_norm_compose(
    qkv_sb,
    gamma_q_sb,
    gamma_k_sb,
    num_q_heads=NUM_Q_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    eps=1e-6,
    out_sb=None,
):
    """Pre-RoPE q/k RMSNorm over a head-major NBSd QKV tile (free-axis reduce over head_dim).

    Norms heads [0, num_q_heads) with gamma_q and heads [num_q_heads, num_q_heads + num_kv_heads) with
    gamma_k; the V heads [num_q_heads + num_kv_heads, N) are copied through unchanged.

    Args:
        qkv_sb:     [T, N, D] SBUF. Head-major QKV (T tokens on partition; N heads then head_dim D on
                    free), N = num_q_heads + 2*num_kv_heads, order [q.. | k.. | v..]. Phase 1's
                    [B*S, I] result is this buffer reshaped to [T, N, D] (I = N*D head-major).
        gamma_q_sb: [T, D] SBUF. Q-layernorm weight, partition-broadcast to the T token rows.
        gamma_k_sb: [T, D] SBUF. K-layernorm weight, partition-broadcast to the T token rows.
        num_q_heads, num_kv_heads, head_dim: head config (defaults: Qwen3.6 per-rank 4 / 1 / 256).
        eps:        RMSNorm epsilon (config.rms_norm_eps).
        out_sb:     optional [T, N, D] SBUF output; allocated if None.

    Returns:
        out_sb: [T, N, D] SBUF (same dtype as qkv_sb) with q/k heads RMSNorm'd, v heads passed through.
    """
    T, N, D = qkv_sb.shape
    kernel_assert(D == head_dim, "qkv_sb last dim must equal head_dim")
    kernel_assert(
        N == num_q_heads + 2 * num_kv_heads,
        "N must equal num_q_heads + 2*num_kv_heads (head-major [q.. | k.. | v..])",
    )
    kernel_assert(gamma_q_sb.shape == (T, D), "gamma_q_sb must be [T, head_dim]")
    kernel_assert(gamma_k_sb.shape == (T, D), "gamma_k_sb must be [T, head_dim]")

    if out_sb is None:
        out_sb = nl.ndarray((T, N, D), dtype=qkv_sb.dtype, buffer=nl.sbuf)

    eps_t = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_t, value=float(eps))

    k_end = num_q_heads + num_kv_heads
    for n in range(num_q_heads):
        rms_norm_over_free(
            qkv_sb[0:T, n, 0:D], gamma_q_sb[0:T, 0:D], eps_t, out_sb[0:T, n, 0:D]
        )
    for n in range(num_q_heads, k_end):
        rms_norm_over_free(
            qkv_sb[0:T, n, 0:D], gamma_k_sb[0:T, 0:D], eps_t, out_sb[0:T, n, 0:D]
        )
    for n in range(k_end, N):
        nisa.tensor_copy(dst=out_sb[0:T, n, 0:D], src=qkv_sb[0:T, n, 0:D])

    return out_sb
