# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused DeltaNet input RMSNorm + 4-way input projection for token generation.

Thin @nki.jit wrapper over nkilib's qkv_tkg: norm(hidden) @ proj_w in one call. proj_w concatenates
the in_proj_qkv|z|a|b weights on the output axis (I = conv_dim + value_dim + 2*num_v_heads); the
caller slices the BSD output back into qkv/z/a/b at offsets conv_dim, +value_dim, +num_v_heads.
proj_w is [H, I] -- the transpose of the nn.Linear [I, H] weights (qkv_tkg wants contraction H first).
Per rank (TP=4): hidden=2048, I=3088, T<=2. The SBUF-resident fusion into conv+recurrence lives in
deltanet/decode/fused_layer.py (deltanet_in_proj_fused_tkg_fwd).
"""

import nki
import nki.language as nl

from nkilib.core.qkv.qkv_tkg import qkv_tkg
from nkilib.core.utils.common_types import NormType, QKVOutputLayout, QuantizationType


def in_proj_compose(hidden, proj_w, gamma, eps, output_in_sbuf):
    """Fused input RMSNorm + 4-way projection via qkv_tkg; returns [B,S,I] HBM or [B*S,I] SBUF (caller slices qkv/z/a/b)."""
    return qkv_tkg(
        hidden=hidden,
        qkv_w=proj_w,
        norm_w=gamma,
        norm_type=NormType.RMS_NORM,
        quantization_type=QuantizationType.NONE,
        output_layout=QKVOutputLayout.BSD,
        eps=eps,
        d_head=None,
        num_q_heads=None,
        num_kv_heads=None,
        fused_add=False,
        output_in_sbuf=output_in_sbuf,
    )


@nki.jit
def deltanet_in_proj_fwd(hidden, proj_w, gamma, eps=1e-6):
    """Standalone HBM-output path: returns [B,S,I] = norm(hidden) @ proj_w; caller slices qkv/z/a/b. Launch [2] (or [1])."""
    return in_proj_compose(hidden, proj_w, gamma, eps, output_in_sbuf=False)
