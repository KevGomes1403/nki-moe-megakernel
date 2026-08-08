# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused input RMSNorm + GQA QKV projection for token generation (head_dim=256).

Thin wrapper over nkilib's qkv_tkg -- one fused call computes:

    hidden' = RMSNorm(hidden, norm_w, eps)   # input_layernorm over H
    qkv     = hidden' @ qkv_w                 # fused Q/K/V projection

Per rank (TP=4) the Qwen3.6 GQA block has q_heads=4, kv_heads=1, head_dim=256, so the fused output
dim is I = (q_heads + 2*kv_heads) * head_dim = (4 + 1 + 1) * 256 = 1536, head-major on the free axis
[q0|q1|q2|q3|k0|v0]. qkv_w is [H, I] -- the transpose of the nn.Linear [I, H] weight (qkv_tkg
contracts H first). norm_w is the [1, H] input_layernorm (RMSNorm) weight.

Output forms (both bf16 IO, fp32 matmul + RMSNorm accumulate, LNC=2 H-sharded contraction):
  - output_in_sbuf=True  (megakernel-ready): the result stays in SBUF as [B*S, I]. The free axis I
    carries the NBSd head ordering (q heads, then k, then v) with head_dim on the free axis -- so
    D=256 needs no partition tiling. The downstream phase slices per head: out_sb[:, n*D:(n+1)*D].
  - output_in_sbuf=False (standalone HBM): qkv_tkg arranges I into the NBSd layout [N, B, S, D] in
    shared_hbm (N = q_heads + 2*kv_heads = 6), the same head ordering.

Out of scope (handled in a later phase): the model's separate sigmoid output-gate projection
[H, q_heads*D]. This composable computes q/k/v only.
"""

import nki

from nkilib.core.qkv.qkv_tkg import qkv_tkg
from nkilib.core.utils.common_types import NormType, QKVOutputLayout, QuantizationType

# Qwen3.6 GQA per-rank (TP=4) head config: 16 Q / 4 (replicated 2->4) KV heads sharded over 4 ranks.
HEAD_DIM = 256
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
NUM_HEADS = (
    NUM_Q_HEADS + 2 * NUM_KV_HEADS
)  # 6 (q heads, then k, then v) on the I/N axis
I_DIM = NUM_HEADS * HEAD_DIM  # 1536


def qkv_proj_compose(hidden, qkv_w, norm_w, eps=1e-6, output_in_sbuf=True):
    """Fused input RMSNorm + GQA QKV projection via qkv_tkg (NBSd head-major, q -> k -> v).

    Args:
        hidden:  [B, S, H] HBM (or [128, B*S, H//128] SBUF) input hidden states, bf16.
        qkv_w:   [H, I] HBM fused QKV weight, I = (NUM_Q_HEADS + 2*NUM_KV_HEADS)*HEAD_DIM = 1536,
                 columns head-major [q0|q1|q2|q3|k0|v0]; transpose of the nn.Linear [I, H] weight.
        norm_w:  [1, H] HBM input_layernorm (RMSNorm) weight, bf16.
        eps:     RMSNorm epsilon (config.rms_norm_eps).
        output_in_sbuf: True -> SBUF [B*S, I] (megakernel-ready, I head-major = NBSd order);
                        False -> NBSd HBM [N, B, S, HEAD_DIM] with N = NUM_HEADS = 6.

    Returns:
        The fused QKV projection (q heads, then k, then v) in the requested buffer/layout.
    """
    return qkv_tkg(
        hidden=hidden,
        qkv_w=qkv_w,
        norm_w=norm_w,
        norm_type=NormType.RMS_NORM,
        quantization_type=QuantizationType.NONE,
        output_layout=QKVOutputLayout.NBSd,
        eps=eps,
        d_head=HEAD_DIM,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        fused_add=False,
        output_in_sbuf=output_in_sbuf,
    )


@nki.jit
def gqa_qkv_proj_fwd(hidden, qkv_w, norm_w, eps=1e-6):
    """Standalone HBM-output path: returns the NBSd [N, B, S, HEAD_DIM] fused QKV projection
    (N = NUM_HEADS = 6 heads, q heads then k then v). Launch [2] (or [1])."""
    return qkv_proj_compose(hidden, qkv_w, norm_w, eps, output_in_sbuf=False)
