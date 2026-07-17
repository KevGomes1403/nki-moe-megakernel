# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fully fused GQA token-generation attention layer (head_dim=256), one LNC2 launch.

Fuses the five individually-validated GQA composables into a single @nki.jit kernel with NO HBM
round-trip for intermediates -- qkv / gate / normed / roped / attention-out all stay in SBUF. The
validated kernels (qkv_tkg, qk_norm_compose, rope_partial_compose, vendored attention_tkg via
gqa_attention_d256, output_projection_tkg) are CALLED, not reimplemented; the bridge/glue between them
(the one on-chip head_dim transpose, the KV-cache persistence write, the sigmoid gate apply, and the
o_proj sub-head assembly) lives INLINE in this file for visibility.

Dataflow (one launch [2]; both cores hold the full result after each cross-core reduce):
  Stage 0  projections: qkv_tkg (NO norm; hidden is pre-normed) -> qkv_sb [T,I]; a second q-only
           qkv_tkg over gate_w -> gate_sb [T,G]. Contraction/H-sharded; sendrecv-reduced to full.
  Stage 1  qk_norm: free-axis RMSNorm over head_dim for q/k heads; v passed through (replicated).
  Stage 2  rope: partial rotate_half over the first rope_dim of q + k heads (replicated).
  Bridge A head_dim free->partition transpose (one nc_transpose per head per d-tile): q (pre-scaled by
           1/sqrt(D)), k_active, and gate -> [128, D_TILES, *] partition-major; v stays [T, D].
  Bridge B emit active K/V as outputs (BHDS k, BHSD v) for NxDI's scatter (design A, default).
  Stage 3  attention: gqa_attention_d256 (s_prior-sharded; out_in_sb leaves the full output on both
           cores). Active K/V are passed in SBUF; the caches supply prior context only.
  Bridge B' optional in-place KV-cache scatter (design B) when kv_write_idx is given: write active K/V
           into k_cache/v_cache at [idx : idx+T] via nkilib indirect DMA; runs AFTER the attention read.
  Bridge C gate apply (out * sigmoid(gate)) + reorder into output_projection_tkg's [d,1,N,T] sub-head
           layout (no sendrecv -- each core already holds all q-heads).
  Stage 4  output_projection_tkg: H-shards the [T, hidden] o_proj partial across cores (full on return).

Per-rank (TP=4) dims are module constants mirroring the components.
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.output_projection.output_projection_tkg import output_projection_tkg
from nkilib.core.qkv.qkv_tkg import qkv_tkg
from nkilib.core.utils.allocator import create_auto_alloc_manager
from nkilib.core.utils.common_types import NormType, QKVOutputLayout, QuantizationType

from ..components.attention import gqa_attention_d256
from ..components.pre_attn_norm import pre_attn_rmsnorm_compose
from ..components.qk_norm import qk_norm_compose
from ..components.rope import rope_partial_compose

# Per-rank (TP=4) Qwen3.6 GQA decode dims (mirror gqa/components/*).
P_MAX = 128
HIDDEN = 2048  # H
HEAD_DIM = 256  # D
D_TILES = HEAD_DIM // P_MAX  # 2 (head_dim on partition needs 2 tiles of 128)
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
NUM_HEADS = NUM_Q_HEADS + 2 * NUM_KV_HEADS  # 6: head-major [q0|q1|q2|q3|k0|v0]
NUM_ROPE_HEADS = NUM_Q_HEADS + NUM_KV_HEADS  # 5: q heads + the k head get RoPE
K_HEAD = NUM_Q_HEADS  # k head index in the N axis
V_HEAD = NUM_Q_HEADS + NUM_KV_HEADS  # v head index in the N axis
I_DIM = NUM_HEADS * HEAD_DIM  # 1536 fused qkv
GATE_DIM = NUM_Q_HEADS * HEAD_DIM  # 1024 (sigmoid output gate, head-major [g0..g3])
VALUE_DIM = NUM_Q_HEADS * HEAD_DIM  # 1024 attention value dim -> o_proj contraction
N_SUB = NUM_Q_HEADS * D_TILES  # 8 o_proj sub-heads of 128 (q-head major, d-tile minor)
SCALE = 1.0 / math.sqrt(
    HEAD_DIM
)  # attention 1/sqrt(d); Q is pre-scaled (fuse_rope=False)


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def heads_free_to_partition(src, tok_stride, head_base, n_heads, T, out, scale=None):
    """Bridge A: transpose head_dim from the free axis onto the partition axis (one nc_transpose per
    (head, d-tile)). For heads [head_base, head_base+n_heads) of a head-major [T, *, D] SBUF tile:
        out[d_in, dt, h*T + t] = src[t, head_base + h, dt*128 + d_in]   (h, t local; optional *scale).
    src token stride is tok_stride (N*D for the qkv tile, q_heads*D for the gate tile); out is
    [128, D_TILES, n_heads*T]. nc_transpose (transpose-mode matmul) requires the psum dtype to match
    src, so tp is allocated in out.dtype (== src.dtype, the IO type)."""
    tp = nl.ndarray((P_MAX, T), dtype=out.dtype, buffer=nl.psum)
    for h in range(n_heads):
        head = head_base + h
        for dt in range(D_TILES):
            nisa.nc_transpose(
                dst=tp[0:P_MAX, 0:T],
                data=src.ap(
                    pattern=[[tok_stride, T], [1, P_MAX]],
                    offset=head * HEAD_DIM + dt * P_MAX,
                ),
            )
            dst_view = out[0:P_MAX, dt, h * T : h * T + T]
            if scale == None:
                nisa.tensor_copy(dst=dst_view, src=tp[0:P_MAX, 0:T])
            else:
                nisa.activation(
                    dst=dst_view, op=nl.copy, data=tp[0:P_MAX, 0:T], scale=scale
                )


def scatter_kv_cache_inplace(k_cache, v_cache, k_active_sb, roped, kv_write_idx, T):
    """Bridge B' (design B): scatter active K/V into the KV caches in place at slots [idx : idx+T].

    Reuses the nkilib indirect-DMA primitive (``nisa.dma_copy`` with ``scalar_offset`` on the
    runtime write-start slot and ``indirect_dim`` = the cache's L axis) from
    attention_block_tkg._update_flat_cache. K is BHDS [B,1,D,L] (head_dim on partition, so the write
    is TILED over D_TILES with L-axis stride 1); V is BHSD [B,1,L,D] (token on partition, L-axis
    stride HEAD_DIM, single DMA). LNC2: V is written on prg 0 and K on prg 1 so the [2] launch writes
    each cache exactly once (both cores hold replicated active K/V; gating avoids the duplicate write).

    Args:
        k_cache: [B,1,D,L] BHDS key cache, mutated in place.
        v_cache: [B,1,L,D] BHSD value cache, mutated in place.
        k_active_sb: [128, D_TILES, T] head-dim-on-partition post-norm/RoPE active K (SBUF).
        roped: [T, N, D] post-norm/RoPE heads (SBUF); the V head supplies the active V.
        kv_write_idx: [B,1] int32 write-start slot (B==1).
        T: number of active tokens written (T consecutive slots).
    """
    L = k_cache.shape[3]
    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)
    # Write-start position: int32 HBM -> int32 SBUF (DMA) -> uint32 SBUF (compute cast for scalar_offset).
    pos_raw = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos_raw, src=kv_write_idx[0:1, 0:1])
    start_position = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=start_position, src=pos_raw)
    # V scatter (prg 0): BHSD, indirect on the L axis (axis 2, stride HEAD_DIM); token-on-partition src.
    if n_prgs == 1 or prg_id == 0:
        nisa.dma_copy(
            dst=v_cache.ap(
                pattern=[[HEAD_DIM, T], [1, HEAD_DIM]],
                offset=0,
                scalar_offset=start_position,
                indirect_dim=2,
            ),
            src=roped[0:T, V_HEAD, 0:HEAD_DIM],
        )
    # K scatter (prg 1): BHDS, indirect on the L axis (axis 3, stride 1), tiled over D_TILES.
    if n_prgs == 1 or prg_id == 1:
        for dt in range(D_TILES):
            nisa.dma_copy(
                dst=k_cache.ap(
                    pattern=[[L, P_MAX], [1, T]],
                    offset=dt * P_MAX * L,
                    scalar_offset=start_position,
                    indirect_dim=3,
                ),
                src=k_active_sb[0:P_MAX, dt, 0:T],
            )


def gqa_fused_compose(
    hidden,
    qkv_w,
    gate_w,
    gamma_q,
    gamma_k,
    cos,
    sin,
    k_cache,
    v_cache,
    mask,
    o_proj_w,
    eps,
    kv_write_idx=None,
    gamma_in=None,
    out_in_sb=False,
):
    """Compose the full fused GQA token-generation layer.

    ``hidden`` is PRE-NORMED when ``gamma_in`` is None; RAW when ``gamma_in`` is given (the input
    RMSNorm is applied in-kernel and stays SBUF-resident, feeding both qkv_tkg calls with zero HBM
    round-trip). ``gamma_in`` (input_layernorm.weight, [1, H], standard form) is the pre-attention
    hidden-state RMSNorm -- distinct from the per-head qk-norms ``gamma_q``/``gamma_k``.

    Returns ``(o_out [T, H], active_k [B,1,D,T] BHDS, active_v [B,1,T,D] BHSD)`` -- the per-rank
    o_proj partial plus the post-norm/RoPE active K/V (the tensors NxDI's update_kv_by_layer_id
    scatters into the caches when k_cache_transposed=True). When ``kv_write_idx`` is given the kernel
    ALSO scatters the active K/V into the caches in place (design B) and additionally returns the
    mutated ``(k_cache, v_cache)`` handles so callers can observe / alias the write; the first three
    returns are unchanged so design A keeps working.
    """
    if hidden.buffer == nl.sbuf:
        h0_in, T, h1 = hidden.shape  # [H0, T, H1] megakernel residual (B == 1)
        kernel_assert(h0_in == P_MAX, "SBUF hidden partition dim must be 128")
        B, H = 1, h0_in * h1
    else:
        B, S, H = hidden.shape
        kernel_assert(B == 1, "fused GQA kernel supports batch size B == 1")
        T = B * S
    kernel_assert(H == HIDDEN, "hidden dim must equal HIDDEN")
    L = k_cache.shape[3]  # full KV length (prior + active), multiple of 128
    kernel_assert(L % P_MAX == 0, "L (curr_sprior) must be a multiple of 128")
    io = hidden.dtype

    # Optional in-kernel pre-attention RMSNorm. When gamma_in is given, RMSNorm raw hidden once into
    # an SBUF-resident [128, T, 16] tile (the exact qkv_tkg SBUF-input layout) and feed BOTH projection
    # calls from it -- zero HBM round-trip. When gamma_in is None, hidden is already pre-normed in HBM.
    if gamma_in != None:
        normed_sb = nl.ndarray((P_MAX, T, HIDDEN // P_MAX), dtype=io, buffer=nl.sbuf)
        pre_attn_rmsnorm_compose(hidden, gamma_in, eps, HIDDEN, normed_sb)
        proj_input = normed_sb
    else:
        proj_input = hidden

    # Stage 0 -- projections (NO norm: proj_input is RMSNorm'd hidden). H-sharded; both cores end with full.
    # Distinct-prefix managers so the two qkv_tkg calls don't emit duplicate buffer op names.
    qkv_sbm = create_auto_alloc_manager()
    qkv_sbm.set_name_prefix("gqa_qkv_")
    gate_sbm = create_auto_alloc_manager()
    gate_sbm.set_name_prefix("gqa_gate_")
    qkv_sb = qkv_tkg(
        hidden=proj_input,
        qkv_w=qkv_w,
        norm_w=None,
        norm_type=NormType.NO_NORM,
        quantization_type=QuantizationType.NONE,
        output_layout=QKVOutputLayout.NBSd,
        eps=eps,
        d_head=HEAD_DIM,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        output_in_sbuf=True,
        sbm=qkv_sbm,
    )  # [T, I] head-major [q0|q1|q2|q3|k0|v0]
    gate_sb = qkv_tkg(
        hidden=proj_input,
        qkv_w=gate_w,
        norm_w=None,
        norm_type=NormType.NO_NORM,
        quantization_type=QuantizationType.NONE,
        output_layout=QKVOutputLayout.NBSd,
        eps=eps,
        d_head=HEAD_DIM,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=0,
        output_in_sbuf=True,
        sbm=gate_sbm,
    )  # [T, G] head-major [g0|g1|g2|g3]
    qkv_view = qkv_sb.reshape(
        (T, NUM_HEADS, HEAD_DIM)
    )  # free reshape (I = N*D head-major)

    # Stage 1 -- q/k RMSNorm over head_dim (free-axis reduce), v passed through; gammas broadcast to T.
    gamma_q_sb = nl.ndarray((T, HEAD_DIM), dtype=io, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=gamma_q_sb, src=gamma_q.ap(pattern=[[0, T], [1, HEAD_DIM]], offset=0)
    )
    gamma_k_sb = nl.ndarray((T, HEAD_DIM), dtype=io, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=gamma_k_sb, src=gamma_k.ap(pattern=[[0, T], [1, HEAD_DIM]], offset=0)
    )
    normed = qk_norm_compose(
        qkv_view, gamma_q_sb, gamma_k_sb, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, eps
    )  # [T, N, D]

    # Stage 2 -- partial RoPE (rotate_half over the first rope_dim) on q + k heads; v passed through.
    rope_dim = cos.shape[1]
    cos_sb = nl.ndarray((T, rope_dim), dtype=io, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos_sb, src=cos[0:T, 0:rope_dim])
    sin_sb = nl.ndarray((T, rope_dim), dtype=io, buffer=nl.sbuf)
    nisa.dma_copy(dst=sin_sb, src=sin[0:T, 0:rope_dim])
    roped = rope_partial_compose(normed, cos_sb, sin_sb, NUM_ROPE_HEADS)  # [T, N, D]

    # Bridge A -- head_dim free->partition transpose (the one on-chip transpose). Q is pre-scaled.
    q_sb = nl.ndarray((P_MAX, D_TILES, NUM_Q_HEADS * T), dtype=io, buffer=nl.sbuf)
    k_active_sb = nl.ndarray((P_MAX, D_TILES, T), dtype=io, buffer=nl.sbuf)
    gate_p = nl.ndarray((P_MAX, D_TILES, NUM_Q_HEADS * T), dtype=io, buffer=nl.sbuf)
    tok_stride = NUM_HEADS * HEAD_DIM
    heads_free_to_partition(roped, tok_stride, 0, NUM_Q_HEADS, T, q_sb, scale=SCALE)
    heads_free_to_partition(roped, tok_stride, K_HEAD, 1, T, k_active_sb)
    gate_view = gate_sb.reshape((T, NUM_Q_HEADS, HEAD_DIM))
    heads_free_to_partition(
        gate_view, NUM_Q_HEADS * HEAD_DIM, 0, NUM_Q_HEADS, T, gate_p
    )

    # Active K/V outputs (BHDS / BHSD). NxDI owns the cache scatter; the kernel only returns these.
    # k_active [B,1,D,T] from the head_dim-on-partition SBUF tile; v_active [B,1,T,D] = the V head.
    active_k = nl.ndarray((B, 1, HEAD_DIM, T), dtype=io, buffer=nl.shared_hbm)
    for dt in range(D_TILES):
        nisa.dma_copy(
            dst=active_k.ap(pattern=[[T, P_MAX], [1, T]], offset=(dt * P_MAX) * T),
            src=k_active_sb[0:P_MAX, dt, 0:T],
        )
    v_active_sb = nl.ndarray((T, HEAD_DIM), dtype=io, buffer=nl.sbuf)
    nisa.tensor_copy(dst=v_active_sb, src=roped[0:T, V_HEAD, 0:HEAD_DIM])
    active_v = nl.ndarray((B, 1, T, HEAD_DIM), dtype=io, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=active_v.ap(pattern=[[HEAD_DIM, T], [1, HEAD_DIM]], offset=0),
        src=v_active_sb,
    )

    # Stage 3 -- attention. Active K/V stay in SBUF (caches hold prior only); full out on both.
    out_sb = nl.ndarray((P_MAX, D_TILES, NUM_Q_HEADS * T), dtype=io, buffer=nl.sbuf)
    gqa_attention_d256(
        q_sb=q_sb,
        k_active_sb=k_active_sb,
        k_prior=k_cache,
        v_prior=v_cache,
        v_active=v_active_sb,
        mask=mask,
        out_sb=out_sb,
        bs=B,
        q_head=NUM_Q_HEADS,
        s_active=T,
        curr_sprior=L,
        head_dim=HEAD_DIM,
        v_in_sb=True,
    )

    # Bridge B' -- optional design-B in-place KV-cache scatter (AFTER attention's prior read).
    if kv_write_idx != None:
        scatter_kv_cache_inplace(k_cache, v_cache, k_active_sb, roped, kv_write_idx, T)

    # Bridge C -- sigmoid output gate + reorder to output_projection_tkg's [d, 1, N, T] sub-head layout
    # (sub-head n = h*D_TILES + dt; head_dim already on partition, so a pure SBUF reorder, no sendrecv).
    sig = nl.ndarray(
        (P_MAX, D_TILES, NUM_Q_HEADS * T), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.activation(dst=sig, op=nl.sigmoid, data=gate_p)
    gated = nl.ndarray((P_MAX, D_TILES, NUM_Q_HEADS * T), dtype=io, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gated, data1=out_sb, data2=sig, op=nl.multiply)
    attn_full = nl.ndarray((P_MAX, 1, N_SUB, T), dtype=io, buffer=nl.sbuf)
    for h in range(NUM_Q_HEADS):
        for dt in range(D_TILES):
            nisa.tensor_copy(
                dst=attn_full[0:P_MAX, 0, h * D_TILES + dt, 0:T],
                src=gated[0:P_MAX, dt, h * T : h * T + T],
            )

    # Stage 4 -- o_proj. H-sharded across cores; each core writes its disjoint hidden columns -> full.
    # out_in_sb=True returns the per-core H-shard as an SBUF [T, H/n_prgs] tile for the megakernel
    # residual add (P1 selects TRANSPOSE_OUT for the residual layout); default False keeps HBM [T, H].
    o_out = output_projection_tkg(
        attention=attn_full,
        weight=o_proj_w,
        bias=None,
        quantization_type=QuantizationType.NONE,
        TRANSPOSE_OUT=False,
        OUT_IN_SB=out_in_sb,
    )
    if kv_write_idx != None:
        return o_out, active_k, active_v, k_cache, v_cache
    return o_out, active_k, active_v


@nki.jit
def gqa_fused_tkg_fwd(
    hidden,
    qkv_w,
    gate_w,
    gamma_q,
    gamma_k,
    cos,
    sin,
    k_cache,
    v_cache,
    mask,
    o_proj_w,
    eps=1e-6,
    kv_write_idx=None,
    gamma_in=None,
):
    """Fully fused GQA decode (T=1) and speculative verify (T>=2) in one LNC2 launch [2].

    Args:
        hidden:   [B, S, H] hidden states. PRE-NORMED when gamma_in is None (the input RMSNorm is
                  already applied); RAW when gamma_in is given (the input RMSNorm runs in-kernel and
                  stays SBUF-resident, feeding both projections with zero HBM round-trip).
        qkv_w:    [H, I] fused QKV weight, head-major cols [q0|q1|q2|q3|k0|v0] (transpose of nn.Linear).
        gate_w:   [H, G] output-gate weight, head-major cols [g0|g1|g2|g3] (transpose of nn.Linear).
        gamma_q:  [D] q-RMSNorm gamma (standard weight).   gamma_k: [D] k-RMSNorm gamma.
        cos, sin: [T, rope_dim] per-token rotary tables (partial RoPE on the first rope_dim).
        k_cache:  [B, 1, D, L] BHDS key cache (k_cache_transposed); read prior context, written if
                  kv_write_idx is given.
        v_cache:  [B, 1, L, D] BHSD value cache; read prior context, written if kv_write_idx is given.
        mask:     [L, B, q_heads, s_active] uint8 attention mask (1=keep), s_prior-major.
        o_proj_w: [value_dim, H] o_proj weight (transpose of nn.Linear), row-indexed by value_dim.
        eps:      RMSNorm epsilon.
        kv_write_idx: optional [B,1] int32 write-start slot. None (default) = design A, no cache
                  write (returns 3 tensors, unchanged). When given = design B, the kernel scatters the
                  active K/V into the caches in place at [idx : idx+T] via nkilib indirect DMA and
                  ALSO returns the mutated cache handles (returns 5 tensors).
        gamma_in: optional [1, H] input_layernorm.weight (standard form, no +1). None (default) =
                  hidden is pre-normed in HBM (current behavior; the model's call is unaffected). When
                  given, the pre-attention RMSNorm runs in-kernel and its output stays SBUF-resident,
                  feeding both qkv_tkg calls directly. Distinct from the per-head qk-norms gamma_q/gamma_k.

    Returns:
        Design A (kv_write_idx is None): ``(o_out [T, H], active_k [B,1,D,T] BHDS,
        active_v [B,1,T,D] BHSD)`` -- the per-rank o_proj PARTIAL (TP all-reduce deferred) and the
        post-norm/RoPE active K/V for NxDI's cache scatter.
        Design B (kv_write_idx given): the same three plus ``(k_cache, v_cache)`` -- the in-place
        mutated cache handles.
    """
    return gqa_fused_compose(
        hidden,
        qkv_w,
        gate_w,
        gamma_q,
        gamma_k,
        cos,
        sin,
        k_cache,
        v_cache,
        mask,
        o_proj_w,
        eps,
        kv_write_idx,
        gamma_in,
    )
