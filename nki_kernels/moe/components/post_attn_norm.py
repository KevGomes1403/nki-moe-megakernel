# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-attention RMSNorm for the Qwen3.6-A3B MoE decoder layer (token generation).

The MoE block's ``post_attention_layernorm``, run so its output PERSISTS in SBUF as the [H0, T, H1]
tile that the router and routed/shared experts consume with zero HBM round-trip. This is the MoE-layer
input norm -- ``normed_sb`` is shared by ALL downstream composables (router, routed experts, and the
next slice's shared expert).

100% of the math is nkilib ``rmsnorm_tkg`` -- identical to the GQA input RMSNorm
(``pre_attn_rmsnorm_compose``); only the gamma differs (post_attention_layernorm.weight) AND the emitted
[H0,T,H1] H-permutation is exposed via ``single_core_forced``, which keys ``rmsnorm_tkg``'s num_H_shards:
  * num_H_shards = n_prgs (single_core_forced=False, what the MoE consumers use): "tp2013" -- free index
    f = s*H2 + h2 <-> H-column s*(H0*H2) + h0*H2 + h2, H2 = H1 // n_prgs. This is the layout the attention
    kernels emit, so a megakernel can share ONE SBUF residual. At n_prgs=1 it degenerates to tp102.
  * num_H_shards = 1 (single_core_forced=True): "tp102" (H = h0*H1 + f) on every core.
Note num_H_shards is keyed off the LNC count only -- it is independent of whether rmsnorm itself shards
the BxS work (it does not below SHARDING_THRESHOLD, so at T<=2 both cores compute the full norm).
"""

import nki.language as nl

from nkilib.core.subkernels.rmsnorm_tkg import rmsnorm_tkg

H = 2048
H0 = 128  # partition dim (nl.tile_size.pmax)
H1 = H // H0  # 16 free H-tiles


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def post_attn_rmsnorm_compose(
    hidden,
    gamma,
    eps=1e-6,
    hidden_actual=None,
    normed_sb=None,
    single_core_forced=False,
):
    """RMSNorm raw post-attn hidden into an SBUF-resident [H0, T, H1] tile (zero HBM round-trip).

    Args:
        hidden:        [B, T, H] HBM raw post-attn residual (B*S = T tokens), bf16/fp32; left untouched.
        gamma:         [1, H] HBM post_attention_layernorm.weight (standard form; no +1 applied here).
        eps:           RMSNorm epsilon (config.rms_norm_eps).
        hidden_actual: actual H used for the mean (defaults to H; set when input is padded).
        normed_sb:     optional [H0, T, H1] SBUF output; allocated if None.
        single_core_forced: True forces num_H_shards=1 (tp102) on every core; False (what the MoE
                       consumers pass) emits num_H_shards=n_prgs -- the tp2013 megakernel residual layout.

    Returns:
        normed_sb: [H0=128, T, H1=H//128] SBUF (same dtype as hidden), the MoE-layer input norm tile.
    """
    B, S, hdim = hidden.shape
    T = B * S
    kernel_assert(hdim % H0 == 0, "hidden H must be divisible by 128")
    h1 = hdim // H0
    kernel_assert(tuple(gamma.shape) == (1, hdim), "gamma must be [1, H]")

    if hidden_actual is None:
        hidden_actual = hdim
    if normed_sb is None:
        normed_sb = nl.ndarray((H0, T, h1), dtype=hidden.dtype, buffer=nl.sbuf)

    rmsnorm_tkg(
        input=hidden,
        gamma=gamma,
        output=normed_sb,
        eps=eps,
        hidden_actual=hidden_actual,
        single_core_forced=single_core_forced,
    )
    return normed_sb
