# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-attention RMSNorm for the Qwen3.6-A3B MoE decoder layer (token generation).

The MoE block's ``post_attention_layernorm``, run so its output PERSISTS in SBUF as the [H0, T, H1]
tile that the router and routed/shared experts consume with zero HBM round-trip. This is the MoE-layer
input norm -- ``normed_sb`` is shared by ALL downstream composables (router, routed experts, and the
next slice's shared expert).

``single_core_forced`` keys ``rmsnorm_tkg``'s num_H_shards, which selects the emitted [H0,T,H1]
H-permutation:
  * num_H_shards = n_prgs (single_core_forced=False, what the MoE consumers use): "tp2013" -- free index
    f = s*H2 + h2 <-> H-column s*(H0*H2) + h0*H2 + h2, H2 = H1 // n_prgs. This is the layout the attention
    kernels emit, so a megakernel can share ONE SBUF residual. At n_prgs=1 it degenerates to tp102.
  * num_H_shards = 1 (single_core_forced=True): "tp102" (H = h0*H1 + f) on every core.
Note num_H_shards is keyed off the LNC count only -- it is independent of whether rmsnorm itself shards
the BxS work (it does not below SHARDING_THRESHOLD, so at T<=2 both cores compute the full norm).
"""

from ...common import H0, rmsnorm_to_sbuf

H = 2048
H1 = H // H0  # 16 free H-tiles


def post_attn_rmsnorm_compose(
    hidden,
    gamma,
    eps=1e-6,
    hidden_actual=None,
    normed_sb=None,
    single_core_forced=False,
    name_prefix="",
):
    """RMSNorm raw post-attn hidden into an SBUF-resident [H0, T, H1] tile (zero HBM round-trip).

    ``gamma`` is the layer's post_attention_layernorm.weight; see ``rmsnorm_to_sbuf`` for the full
    argument contract, and the module docstring for what ``single_core_forced`` does to the layout.
    """
    return rmsnorm_to_sbuf(
        hidden,
        gamma,
        eps=eps,
        hidden_actual=hidden_actual,
        normed_sb=normed_sb,
        single_core_forced=single_core_forced,
        name_prefix=name_prefix + "postnorm_",
    )
