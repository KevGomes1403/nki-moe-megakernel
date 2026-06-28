# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GQA token-generation attention for head_dim=256, wrapping nkilib's AWS-tuned
``attention_tkg`` (vendored + patched for head_dim on partition tiles).

This composable is a thin shim over the vendored ``attention_tkg`` kernel
(``..vendored.attention_tkg``). The decode QK^T / online-softmax / P.V compute
path is byte-for-byte AWS code; only the head_dim-on-partition layout sites are
tiled into D_TILES = ceil(d_head / 128) partition tiles so d_head=256 fits the PE
array (contraction-K <= 128, stationary-free-M <= 128, partition <= 128). See the
banner in ``vendored/attention_tkg.py`` for the exact patch sites.

Replaces the earlier from-scratch ``gqa_attention_core`` (preserved as
``attention_fresh_ref.py`` for cross-checking) so the perf-critical attention
math is AWS-authored rather than hand-written.

INPUT CONTRACT (single TP shard; GQA: ``q_head`` query heads share 1 kv head):

  * Q is PRE-SCALED by 1/sqrt(d_head) and already RoPE'd / qk-normed. The vendored
    kernel applies the 1/sqrt(d) scale internally ONLY when fuse_rope=True; here
    fuse_rope=False, so the caller (Phase 2 qk_norm + Phase 3 rope) owns scaling,
    RoPE and norm. K is RoPE'd / normed but NOT scaled. This composable does not
    touch Q/K/V values.
  * The caller supplies the full attention ``mask`` (1=keep, 0=mask). The kernel
    does not generate causality here (use_pos_id=False).
  * curr_sprior (== full KV length L, prior + active) MUST be a multiple of 128,
    and a multiple of 256 when s_prior is sharded across 2 cores (L >= 256 with
    LNC2). The active tokens occupy the LAST s_active slots of the L-length KV.

LAYOUT (head_dim tiled on partition as D_TILES = ceil(d_head/128)):

  q_sb       : [128, D_TILES, B*H*s_active] SBUF io_type. q_sb[d_in, dt, b*H*s_active
               + h*s_active + s] = Q[dt*128+d_in, b, h, s].
  k_active_sb: [128, D_TILES, B*s_active]   SBUF io_type. [d_in, dt, b*s_active + s].
  k_prior    : [B, 1, d_head, L]            HBM io_type (already transposed; flat-KV,
               tp_k_prior=False). Its last s_active s_prior slots are overwritten by
               k_active inside the kernel.
  v_prior    : [B, 1, L, d_head]            HBM io_type.
  v_active   : [B, 1, s_active, d_head]     HBM io_type.
  mask       : [L, B, H, s_active]          HBM uint8 (1=keep). s_prior-major (linear).
  out_sb     : [128, D_TILES, B*H*s_active] SBUF, written in place (out_in_sb=True).

Precision: io_type bf16, fp32 matmul/softmax accumulate (AWS default).
"""

import nki.language as nl

from ..vendored.attention_tkg import _D_HEAD_TILE, _d_tiles, attention_tkg
from ..vendored.attention_tkg_utils import AttnTKGConfig

try:
    from nkilib.core.utils.allocator import SbufManager
except ImportError:  # pragma: no cover - nkilib is expected to be installed
    SbufManager = None

# Auto-alloc budget. With use_auto_alloc the compiler places SBUF; this bound is
# only the manager's bookkeeping ceiling, so a generous value is safe.
SBM_BUDGET_BYTES = 24 * 1024 * 1024


def _kernel_assert(condition, msg):
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {msg}"
    )


def build_attention_tkg_config(
    bs,
    q_head,
    s_active,
    curr_sprior,
    head_dim,
    full_sprior=None,
):
    """Build the AttnTKGConfig for the head_dim=256 GQA decode path.

    Pins the flags our config takes: flat KV (no block cache), no fp8, no FA
    s_prior tiling, no fused RoPE, no in-kernel mask gen; q/k pre-loaded in SBUF
    and output kept in SBUF. Adaptive LNC2 sharding (s_prior or none) is decided
    inside the kernel from the SPMD grid size.
    """
    full_sprior = curr_sprior if full_sprior is None else full_sprior
    return AttnTKGConfig(
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        curr_sprior=curr_sprior,
        full_sprior=full_sprior,
        d_head=head_dim,
        block_len=0,
        tp_k_prior=False,
        strided_mm1=False,
        use_pos_id=False,
        fuse_rope=False,
        use_gpsimd_sb2sb=True,
        qk_in_sb=True,
        k_out_in_sb=False,
        out_in_sb=True,
        enable_fa_s_prior_tiling=False,
    )


def gqa_attention_d256(
    q_sb,
    k_active_sb,
    k_prior,
    v_prior,
    v_active,
    mask,
    out_sb,
    bs,
    q_head,
    s_active,
    curr_sprior,
    head_dim,
    full_sprior=None,
    sbm=None,
):
    """Head_dim=256 GQA decode attention via the vendored AWS ``attention_tkg``.

    SBUF-in (q_sb, k_active_sb) / SBUF-out (out_sb); KV cache streamed from HBM by
    the AWS kernel's tuned DMA path. See the module docstring for the full layout
    and input contract. Writes ``out_sb`` in place and returns it.

    Args:
        q_sb, k_active_sb: head_dim-tiled SBUF query / active key (see layout).
        k_prior, v_prior, v_active: HBM KV cache tensors (see layout).
        mask: HBM uint8 attention mask [L, B, H, s_active] (1=keep).
        out_sb: SBUF output [128, D_TILES, B*H*s_active], written in place.
        bs, q_head, s_active, curr_sprior, head_dim: decode dims (GQA: q_head
            query heads share 1 kv head; curr_sprior == full KV length L).
        full_sprior: KV buffer capacity (defaults to curr_sprior).
        sbm: optional SbufManager (a megakernel may pass its own). Allocated here
            in auto-alloc mode when None.

    Returns:
        out_sb (the same SBUF tensor passed in).
    """
    d_tiles = _d_tiles(head_dim)
    _kernel_assert(head_dim % _D_HEAD_TILE == 0, "head_dim must be a multiple of 128")
    _kernel_assert(
        q_sb.shape[0] == _D_HEAD_TILE and q_sb.shape[1] == d_tiles,
        f"q_sb must be [128, {d_tiles}, B*H*s_active], got {q_sb.shape}",
    )
    _kernel_assert(
        out_sb.shape[0] == _D_HEAD_TILE and out_sb.shape[1] == d_tiles,
        f"out_sb must be [128, {d_tiles}, B*H*s_active], got {out_sb.shape}",
    )
    _kernel_assert(
        curr_sprior % _D_HEAD_TILE == 0, "curr_sprior (L) must be a multiple of 128"
    )

    cfg = build_attention_tkg_config(
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        curr_sprior=curr_sprior,
        head_dim=head_dim,
        full_sprior=full_sprior,
    )

    own_sbm = sbm is None
    if own_sbm:
        _kernel_assert(SbufManager is not None, "nkilib SbufManager unavailable")
        sbm = SbufManager(0, SBM_BUDGET_BYTES, use_auto_alloc=True)

    # attention_tkg's first stack allocations (one_vec, position IDs) run before
    # its own per-tile open_scope, so a scope must be open when we own the sbm.
    if own_sbm:
        sbm.open_scope()
    attention_tkg(
        q=q_sb,
        k_active=k_active_sb,
        v_active=v_active,
        k_prior=k_prior,
        v_prior=v_prior,
        mask=mask,
        out=out_sb,
        cfg=cfg,
        sbm=sbm,
    )
    if own_sbm:
        sbm.close_scope()
    return out_sb
