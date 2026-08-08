# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partial RoPE (rotate_half) for the GQA token-generation block (head_dim=256, Phase 3 of 5).

Applies rotary position embedding to the first ROPE_DIM=64 dimensions of each rope head's
HEAD_DIM=256 vector and passes the remaining 192 through unchanged. The Q heads and the K head are
rotated; the V head is not.

Layout (matches the upstream qk_norm output so the two chain with no transpose):
    x_sb [T, N, D] -- T = B*S tokens on the PARTITION axis; on the FREE axis, N heads (head-major
    [q.. | k.. | v..]) then head_dim D contiguous within each head. cos/sin are [T, rope_dim] (one
    row per token) and are broadcast across the N heads internally -- the same rotation applies to
    every rope head of a given token.

Rotation convention (rotate_half), with half = rope_dim // 2 (dim i pairs with i+half):
    out[:, :half]      = x[:, :half]      * cos[:, :half]      - x[:, half:rope] * sin[:, :half]
    out[:, half:rope]  = x[:, half:rope]  * cos[:, half:rope]  + x[:, :half]     * sin[:, half:rope]
    out[:, rope:D]     = x[:, rope:D]                                            (pass-through copy)

The two halves of cos/sin are applied generally (not assumed equal). Precision: fp32 internal, IO
dtype preserved (bf16 runtime / fp32 correctness gate); pass-through copied bit-exactly.
"""

import nki.isa as nisa
import nki.language as nl

# Qwen3.6 GQA per-rank (TP=4) head config; mirrors gqa/components/qk_norm.py.
HEAD_DIM = 256
ROPE_DIM = 64  # rotated width = head_dim * partial_rotary_factor (0.25)
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
NUM_ROPE_HEADS = (
    NUM_Q_HEADS + NUM_KV_HEADS
)  # 5: q heads + the k head get RoPE; v does not
NUM_HEADS = NUM_Q_HEADS + 2 * NUM_KV_HEADS  # 6 heads on the N axis [q0|q1|q2|q3|k0|v0]


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def rotate_half_head(x_head, cos_sb, sin_sb, out_head):
    """rotate_half RoPE on one head's rope slice [T, rope_dim] (fp32 internal, IO-dtype store).

    Writes only out_head[:, 0:rope_dim]; the [rope_dim:] tail is the caller's responsibility.

    Args:
        x_head:   [T, D] SBUF view of one head (T tokens on partition, head_dim on free).
        cos_sb:   [T, rope_dim] SBUF cosine (one row per token), shared across heads.
        sin_sb:   [T, rope_dim] SBUF sine, same shape as cos_sb.
        out_head: [T, D] SBUF view of the destination head; columns [0:rope_dim] are written.
    """
    T, rope_dim = cos_sb.shape
    half = rope_dim // 2

    # res = x * cos  (full rope_dim), accumulated in fp32.
    res = nl.ndarray((T, rope_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=res,
        data1=x_head[0:T, 0:rope_dim],
        data2=cos_sb[0:T, 0:rope_dim],
        op=nl.multiply,
    )

    # res[:, :half] -= x[:, half:rope] * sin[:, :half]   (= x1*cos1 - x2*sin1)
    t1 = nl.ndarray((T, half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=t1,
        data1=x_head[0:T, half:rope_dim],
        data2=sin_sb[0:T, 0:half],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=res[0:T, 0:half], data1=res[0:T, 0:half], data2=t1, op=nl.subtract
    )

    # res[:, half:rope] += x[:, :half] * sin[:, half:rope]   (= x2*cos2 + x1*sin2)
    t2 = nl.ndarray((T, half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=t2,
        data1=x_head[0:T, 0:half],
        data2=sin_sb[0:T, half:rope_dim],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=res[0:T, half:rope_dim],
        data1=res[0:T, half:rope_dim],
        data2=t2,
        op=nl.add,
    )

    # Cast the fp32 rope result back to the IO dtype into the rope slice.
    nisa.tensor_copy(dst=out_head[0:T, 0:rope_dim], src=res)


def rope_partial_compose(
    x_sb, cos_sb, sin_sb, num_rope_heads=NUM_ROPE_HEADS, out_sb=None
):
    """Partial RoPE (rotate_half) over a head-major [T, N, D] SBUF tile (head_dim on the free axis).

    Rotates columns [0:rope_dim] of heads [0, num_rope_heads) and passes their [rope_dim:D] tail
    through; heads [num_rope_heads, N) (the V head(s)) are copied through entirely. The same per-token
    cos/sin [T, rope_dim] are broadcast across all rope heads.

    Args:
        x_sb:   [T, N, D] SBUF. T = B*S tokens on partition; N heads (head-major [q.. | k.. | v..])
                then head_dim D contiguous on free. Same layout as the qk_norm output.
        cos_sb: [T, rope_dim] SBUF. Cosine, one row per token (partition-aligned with x_sb tokens).
        sin_sb: [T, rope_dim] SBUF. Sine, same shape/alignment as cos_sb.
        num_rope_heads: heads [0, num_rope_heads) are rotated (default NUM_ROPE_HEADS = q + k heads).
        out_sb: optional [T, N, D] SBUF output; allocated if None. Pass out_sb=x_sb for true in-place.

    Returns:
        out_sb: [T, N, D] SBUF (same dtype as x_sb). Rope heads have RoPE on columns [0:rope_dim] and
            pass columns [rope_dim:D] through; the remaining heads pass through entirely.

    Notes:
        Internal math is fp32; the rope slice is cast back to the IO dtype on store. Pass-through
        values are copied bit-exactly.
    """
    T, N, D = x_sb.shape
    Tc, rope_dim = cos_sb.shape
    kernel_assert(rope_dim % 2 == 0, "rope_dim must be even")
    kernel_assert(rope_dim <= D, "rope_dim must not exceed head_dim D")
    kernel_assert(Tc == T, "cos_sb partition dim must equal x_sb token count T")
    kernel_assert(
        tuple(cos_sb.shape) == tuple(sin_sb.shape),
        "cos and sin must have identical shapes",
    )
    kernel_assert(num_rope_heads <= N, "num_rope_heads must not exceed N")

    if out_sb is None:
        out_sb = nl.ndarray((T, N, D), dtype=x_sb.dtype, buffer=nl.sbuf)

    # Rope heads: rotate [0:rope_dim], pass the [rope_dim:D] tail through.
    for n in range(num_rope_heads):
        rotate_half_head(x_sb[0:T, n, 0:D], cos_sb, sin_sb, out_sb[0:T, n, 0:D])
        nisa.tensor_copy(dst=out_sb[0:T, n, rope_dim:D], src=x_sb[0:T, n, rope_dim:D])

    # Pass-through heads (the V head(s)): full head_dim copy.
    for n in range(num_rope_heads, N):
        nisa.tensor_copy(dst=out_sb[0:T, n, 0:D], src=x_sb[0:T, n, 0:D])

    return out_sb
