# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-attention RMSNorm for the GQA (full_attention) decoder layer (token generation).

The GQA layer's ``input_layernorm``, run so its output PERSISTS in SBUF as the [H0, T, H1] tile that
``qkv_tkg`` consumes with zero HBM round-trip (the megakernel omits the isolation-test's HBM store and
hands ``normed_sb`` straight to the two projection calls via NormType.NO_NORM).

100% of the math is ``nkilib.core.subkernels.rmsnorm_tkg.rmsnorm_tkg`` (square -> reduce over H1 ->
matmul reduce over H0 -> rsqrt -> gamma, fp32 internal, bf16/fp32 IO). This module is caller glue only.

Contract (A3B, TP=4, LNC=2, bs=1):
    hidden: [B, T, H] HBM raw pre-norm hidden (B=1, H=2048), bf16/fp32. Left untouched (residual).
    gamma:  [1, H] HBM = input_layernorm.weight in STANDARD form -- the (1+w) conversion is applied
            once at checkpoint load, so gamma is fed directly with no +1 in-kernel.
    Returns normed_sb: [H0=128, T, H1=16] SBUF (same dtype as hidden) -- the exact qkv_tkg SBUF-input
            layout, drop-in with zero reshape.

Sharding: default num_H_shards = lnc (2 at LNC=2, from the launch grid). This lays the H1=16 output
    columns out as [shard0_H2(8) | shard1_H2(8)] -- the byte-for-byte column order qkv_tkg's NO_NORM
    path slices per shard. At T = B*S <= SHARDING_THRESHOLD(18) both cores compute the full replicated
    norm (no BxS shard, no sendrecv), matching the unsharded runtime input_layernorm.
"""

import nki.language as nl

from nkilib.core.subkernels.rmsnorm_tkg import rmsnorm_tkg

# A3B GQA hidden config. input_layernorm is NOT TP-sharded -- gamma is full H, norm over full H,
# replicated on every rank/core.
H = 2048
H0 = 128  # partition dim (nl.tile_size.pmax)
H1 = H // H0  # 16 free H-tiles


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def pre_attn_rmsnorm_compose(
    hidden, gamma, eps=1e-6, hidden_actual=None, normed_sb=None
):
    """RMSNorm raw hidden into an SBUF-resident [H0, T, H1] tile for qkv_tkg (zero HBM round-trip).

    Args:
        hidden:        raw pre-norm hidden, either [B, T, H] HBM or [H0=128, T, H1] SBUF (the megakernel
                       residual); left untouched. rmsnorm_tkg consumes either directly.
        gamma:         [1, H] HBM input_layernorm.weight (standard form; no +1 applied here).
        eps:           RMSNorm epsilon (config.rms_norm_eps).
        hidden_actual: actual H used for the mean (defaults to H; set when input is padded).
        normed_sb:     optional [H0, T, H1] SBUF output; allocated if None.

    Returns:
        normed_sb: [H0=128, T, H1=H//128] SBUF (same dtype as hidden), qkv_tkg SBUF-input layout.
    """
    if hidden.buffer == nl.sbuf:
        h0_in, T, h1 = hidden.shape  # [H0, T, H1] megakernel residual
        kernel_assert(h0_in == H0, "SBUF hidden partition dim must be 128")
        hdim = h0_in * h1
    else:
        B, S, hdim = hidden.shape
        T = B * S
        kernel_assert(hdim % H0 == 0, "hidden H must be divisible by 128")
        h1 = hdim // H0
    kernel_assert(tuple(gamma.shape) == (1, hdim), "gamma must be [1, H]")

    if hidden_actual is None:
        hidden_actual = hdim

    if normed_sb is None:
        normed_sb = nl.ndarray((H0, T, h1), dtype=hidden.dtype, buffer=nl.sbuf)

    # rmsnorm_tkg default sharding (single_core_forced=False) -> num_H_shards = lnc (launch grid).
    rmsnorm_tkg(
        input=hidden,
        gamma=gamma,
        output=normed_sb,
        eps=eps,
        hidden_actual=hidden_actual,
    )
    return normed_sb
