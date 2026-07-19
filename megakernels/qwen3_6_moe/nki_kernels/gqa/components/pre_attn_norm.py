# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-attention RMSNorm for the GQA (full_attention) decoder layer (token generation).

The GQA layer's ``input_layernorm``, run so its output PERSISTS in SBUF as the [H0, T, H1] tile that
``qkv_tkg`` consumes with zero HBM round-trip (the megakernel omits the isolation-test's HBM store and
hands ``normed_sb`` straight to the two projection calls via NormType.NO_NORM).

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

from ...common import H0, rmsnorm_to_sbuf

# A3B GQA hidden config. input_layernorm is NOT TP-sharded -- gamma is full H, norm over full H,
# replicated on every rank/core.
H = 2048
H1 = H // H0  # 16 free H-tiles


def pre_attn_rmsnorm_compose(
    hidden, gamma, eps=1e-6, hidden_actual=None, normed_sb=None, name_prefix=""
):
    """RMSNorm raw hidden into an SBUF-resident [H0, T, H1] tile for qkv_tkg (zero HBM round-trip).

    ``gamma`` is the layer's input_layernorm.weight; see ``rmsnorm_to_sbuf`` for the full argument
    contract. Uses rmsnorm_tkg's default sharding (num_H_shards = lnc, from the launch grid).
    """
    return rmsnorm_to_sbuf(
        hidden,
        gamma,
        eps=eps,
        hidden_actual=hidden_actual,
        normed_sb=normed_sb,
        name_prefix=name_prefix + "prenorm_",
    )
