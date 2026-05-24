"""Qwen3-30B-A3B multi-layer fused TKG megakernel for speculative decoding.

Runs all decoder layers in one NKI invocation on the vendored nkilib
subkernels (``attention_block_tkg``, ``rmsnorm_tkg``, ``router_topk``,
``moe_tkg``) shared with the gpt-oss megakernel. SBUF-resident residual
across layers; KV caches scattered in place at ``position_ids``.

Per-layer pipeline:
  1. ``attention_block_tkg`` — input RMSNorm, QKV proj, Q/K RMSNorm, RoPE,
     attention, output proj. K/V scattered in place at ``position_ids``.
  2. SB2SB all-reduce + LNC gather on the attention output.
  3. ``residual += attn_out``
  4. ``rmsnorm_tkg`` (post_attention_layernorm)
  5. ``router_topk`` (softmax, post-topk L1 norm)
  6. ``moe_tkg`` (SiLU, POST_SCALE, no clamping/bias)
  7. SB2SB all-reduce + LNC gather on the MoE output.
  8. ``residual += moe_out``
Final ``model.norm`` runs Python-side in NxDI; this kernel does not apply it.

Hardware: Trainium3, TP=4, LNC=2. Entry: ``get_multilayer_kernel_jit(L)[2]``.

Constraints
-----------
* LNC=2 shards prior-context K/V along s_prior, so the K-cache bucket length
  must be a multiple of 256 (= 128 * LNC). Buckets 256/512/768/1024 work;
  640/896/... trip an assert in ``attention_tkg``. Run smoke tests at
  multiple-of-256 seq lengths.
* ``NKI_MOE_LEGACY_DOWN_WEIGHT_LOAD=1``: down hoist reshape needs I%128==0;
  Qwen3 TP=4 gives I=192.
* ``NKI_MOE_LEGACY_GATE_UP_WEIGHT_LOAD=1``: prefetch ring's SWDGE-on-up
  contends with the 48L/LNC=2 scheduler — legacy is faster here.

IO contract (matches ``transformer_qwen.py``)
---------------------------------------------
    kernel_out = get_multilayer_kernel_jit(L)[2](
        hidden_states,                  # [B, S_tkg, H]  bf16
        *Wqkv_list, *Wo_list,
        *qn_list, *kn_list, *gpre_list, *gpost_list,
        *router_list, *gate_up_list, *down_list,
        *K_caches, *V_caches,           # mutated in place
        cos, sin,                       # [B, T, d] per-slot RoPE values
        position_ids,                   # [B, T] int32 (consecutive: [p, p+1, ..., p+T-1])
        replica_groups=...,
    )
    Y     = kernel_out[0]
    K_out = kernel_out[1     : 1 + L]
    V_out = kernel_out[1 + L : 1 + 2 * L]

Wq/Wk/Wv are fused into a single ``W_qkv`` HBM scratch per layer (the vendored
``attention_block_tkg`` takes a fused W_qkv; the integration ships them
separately). cos/sin are permuted from ``[B, T, d]`` to ``[d/2, B, S_tkg]`` for
``rope_contiguous_layout=True`` (with one position per query slot, supporting
both T=1 single-token and T>1 speculative decoding).

T>1 (speculative decoding)
--------------------------
With T>1 the target model verifies ``speculation_length`` consecutive tokens
in one TKG call. ``position_ids`` is ``[B, T]`` with consecutive values
``[p, p+1, ..., p+T-1]``:

* RoPE: each of the T query slots gets its own per-slot cos/sin.
* Attention mask: per-slot threshold — query ``t`` attends to cached slots
  ``s < position_ids[b, t]``.
* KV cache update: pass only the base position ``position_ids[:, :1]``
  (shape ``[B, 1]``) — the cache-update path writes ``S_tkg`` consecutive
  entries starting from the base.
"""

import os as _os

# Must be set before importing nki_kernels.moe. Down: correctness (I=192).
# Gate_up: perf (prefetch ring's SWDGE-on-up contends with 48L/LNC=2 sched).
# Force_shard_on_h: SBUF-resident MoE output at T>1 (spec verify).
_os.environ["NKI_MOE_LEGACY_DOWN_WEIGHT_LOAD"] = "1"
_os.environ["NKI_MOE_LEGACY_GATE_UP_WEIGHT_LOAD"] = "1"
_os.environ["NKI_MOE_FORCE_SHARD_ON_H"] = "1"
# _os.environ["NKI_MOE_INDIRECT_DMA_FUSION"] = "1"

import linecache

import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl

from nki_kernels.attention import attention_block_tkg
from nki_kernels.moe import (
    XHBMLayout_T_H__1,
    XSBLayout_tp102__0,
    XSBLayout_tp2013__1,
    moe_tkg,
    rmsnorm_tkg,
    router_topk,
)
from nkilib.core.utils.allocator import BufferManager, Logger
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    QuantizationType,
    RouterActFnType,
)
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.experimental.transformer.transformer_tkg import _sb2sb_all_reduce_gather


# ---------------------------------------------------------------------------
# Model constants (Qwen3-30B-A3B at TP=4, LNC=2)
# ---------------------------------------------------------------------------

H            = 2048
H0           = 128
H1           = H // H0          # 16
N_PRGS       = 2                # LNC degree
H1_SHARD     = H1 // N_PRGS     # 8 — H1 tiles per LNC shard
H2           = H1_SHARD         # per-shard slice of H1; subkernels view H as
                                # (num_shards, H0, H2) row-major at LNC=2
EPS          = 1e-6             # Qwen3 RMSNorm eps
PMAX         = H0
NUM_LAYERS   = 48

# Attention dims (post TP-shard)
D_HEAD          = 128
NUM_Q_HEADS_TP  = 8             # 32 q heads / TP=4
NUM_KV_HEADS_TP = 1             # 4 kv heads / TP=4
I_QKV           = (NUM_Q_HEADS_TP + 2 * NUM_KV_HEADS_TP) * D_HEAD   # 1280

# MoE dims
E            = 128              # num experts
TOP_K        = 8
TP_DEGREE    = 4                # must match NxDI's tp_degree
# Intermediate dim per TP rank. NxDI shards moe_intermediate_size=768 by TP=4;
# the kernel sees the sharded shapes [E, H, 2*I_PER_EXPERT] / [E, I_PER_EXPERT, H]
# and treats I_PER_EXPERT as the full intermediate dim.
I_PER_EXPERT = 768 // TP_DEGREE  # 192

SBM_SIZE_BYTES = 200 * 1024

# ---------------------------------------------------------------------------
# Full-attention mask builder (no SWA — every Qwen3 layer is full attention)
# ---------------------------------------------------------------------------

def _stream_shuffle_broadcast(src, dst):
    """Replicate src (1, F) across all partitions of dst (P, F)."""
    dst_npar = dst.shape[0]
    shuffle_mask = [0] * 32
    for i in range((dst_npar + 31) // 32):
        cur_npar = min(32, dst_npar - i * 32)
        nisa.nc_stream_shuffle(
            src=src[0:1, :],
            dst=dst[i * 32 : i * 32 + cur_npar, 0 : dst.shape[1]],
            shuffle_mask=shuffle_mask,
        )


def _build_full_mask_hbm(position_ids, S_ctx, B, num_heads, S_tkg):
    """Build full-attention mask [S_ctx, B, num_heads, S_tkg].

    Per query slot t (positions consecutive: [p, p+1, ..., p+S_tkg-1]):

        mask[s, b, h, t] = 1 iff
            (s < base_pos)                              # prior K_cache
            OR (active_base <= s <= active_base + t)    # causal active triangle

        base_pos    := position_ids[b, 0]
        active_base := S_ctx - S_tkg

    Why ``base_pos`` (not ``pos[b, t]``): attention_block_tkg's in-place KV
    scatter runs AFTER the attention block, so K_cache slots
    [base_pos, base_pos+S_tkg) hold stale data when attention reads them. The
    freshly computed K for these tokens lives in the SBUF k_active region,
    which the inner ``attention_tkg`` pastes onto the last ``S_tkg`` slots of
    its prior buffer.

    Assumes B=1: ``base_pos`` is loaded from batch 0 and broadcast; the HBM
    store does not iterate the batch dim.
    """
    P_MAX       = 128
    assert S_ctx % P_MAX == 0, "S_ctx must be a multiple of 128"
    n_tile      = S_ctx // P_MAX
    active_base = S_ctx - S_tkg

    out = nl.ndarray((S_ctx, B, num_heads, S_tkg), dtype=nl.bfloat16,
                     buffer=nl.shared_hbm, name="mask_full_hbm")

    iota_s = nl.ndarray((P_MAX, n_tile, num_heads, S_tkg), dtype=nl.float32,
                        buffer=nl.sbuf, name="iota_s")
    nisa.iota(iota_s,
              pattern=[[P_MAX, n_tile], [0, num_heads], [0, S_tkg]],
              offset=0, channel_multiplier=1)

    iota_t = nl.ndarray((P_MAX, n_tile, num_heads, S_tkg), dtype=nl.float32,
                        buffer=nl.sbuf, name="iota_t")
    nisa.iota(iota_t,
              pattern=[[0, n_tile], [0, num_heads], [1, S_tkg]],
              offset=0, channel_multiplier=0)

    base_pos_one = nl.ndarray((1, B), dtype=nl.float32, buffer=nl.sbuf,
                              name="base_pos_one")
    nisa.dma_copy(
        dst=base_pos_one,
        src=position_ids.reshape((B * S_tkg,)).ap(
            pattern=[[0, 1], [S_tkg, B]],
            offset=0,
        ),
    )
    base_pos = nl.ndarray((P_MAX, B), dtype=nl.float32, buffer=nl.sbuf,
                          name="base_pos")
    _stream_shuffle_broadcast(base_pos_one, base_pos)

    prior = nl.ndarray((P_MAX, n_tile, num_heads, S_tkg), dtype=nl.float32,
                       buffer=nl.sbuf, name="prior")
    nisa.tensor_scalar(prior, iota_s,
                       op0=nl.less, operand0=base_pos[:, 0:1])

    upper_bound = nl.ndarray((P_MAX, n_tile, num_heads, S_tkg), dtype=nl.float32,
                             buffer=nl.sbuf, name="active_upper_bound")
    nisa.tensor_scalar(upper_bound, iota_t,
                       op0=nl.add, operand0=active_base)

    upper = nl.ndarray((P_MAX, n_tile, num_heads, S_tkg), dtype=nl.float32,
                       buffer=nl.sbuf, name="active_upper")
    nisa.tensor_tensor(upper, iota_s, upper_bound, op=nl.less_equal)

    lower = nl.ndarray((P_MAX, n_tile, num_heads, S_tkg), dtype=nl.float32,
                       buffer=nl.sbuf, name="active_lower")
    nisa.tensor_scalar(lower, iota_s,
                       op0=nl.greater_equal, operand0=active_base)

    active = nl.ndarray((P_MAX, n_tile, num_heads, S_tkg), dtype=nl.float32,
                        buffer=nl.sbuf, name="active")
    nisa.tensor_tensor(active, lower, upper, op=nl.minimum)

    mask = nl.ndarray((P_MAX, n_tile, num_heads, S_tkg), dtype=nl.bfloat16,
                      buffer=nl.sbuf, name="mask")
    nisa.tensor_tensor(mask, prior, active, op=nl.maximum)

    out_flat = out.reshape((S_ctx * B * num_heads * S_tkg,))
    nisa.dma_copy(
        dst=out_flat.ap(
            pattern=[
                [num_heads * S_tkg,         P_MAX],
                [P_MAX * num_heads * S_tkg, n_tile],
                [S_tkg,                     num_heads],
                [1,                         S_tkg],
            ],
            offset=0,
        ),
        src=mask,
    )

    return out


# ---------------------------------------------------------------------------
# Build permuted RoPE cos/sin from pre-indexed [B, T, d] tensors
# ---------------------------------------------------------------------------

def _build_permuted_rope_hbm_from_pos(rope_at_pos, B, S_tkg, d_head, name):
    """Permute ``rope_at_pos`` [B, T, d_head] -> HBM [d_head//2, B, S_tkg].

    ``rope_at_pos`` is pre-indexed at each of the T query positions (HF
    contiguous-halves convention — the second half duplicates the first, so
    only the first half is used). The result feeds ``attention_block_tkg``
    with ``rope_contiguous_layout=True``. T must equal S_tkg — each query
    slot gets its own per-slot cos/sin (supports both T=1 and T>1 spec).

    Output mapping: ``out[p, b, t] = rope_at_pos[b, t, p]`` for
    ``p in [0, half_d)``, ``b in [0, B)``, ``t in [0, S_tkg)``.
    """
    half_d = d_head // 2

    out = nl.ndarray((half_d, B, S_tkg), dtype=rope_at_pos.dtype,
                     buffer=nl.shared_hbm, name=name)

    # rope_at_pos shape is [B, T, d_head] = [B, S_tkg, d_head]. Flatten to
    # [B*S_tkg*d_head] and address element [b, t, p] at flat-index
    # b*S_tkg*d_head + t*d_head + p.
    rope_flat = rope_at_pos.reshape((B * S_tkg * d_head,))

    # sb[p, b*S_tkg + t] = rope_at_pos[b, t, p] for p in [0, half_d).
    sb = nl.ndarray((half_d, B * S_tkg), dtype=rope_at_pos.dtype, buffer=nl.sbuf,
                    name=f"{name}_sb")
    nisa.dma_copy(
        dst=sb,
        src=rope_flat.ap(
            pattern=[[1, half_d],          # partition: d_head index (first half)
                     [S_tkg * d_head, B],  # free outer: batch stride
                     [d_head, S_tkg]],     # free inner: token stride (per-slot)
            offset=0,
        ),
    )

    nisa.dma_copy(dst=out.reshape((half_d, B * S_tkg)), src=sb)

    return out


# ---------------------------------------------------------------------------
# Multilayer body
# ---------------------------------------------------------------------------

def _store_shard_interleaved_sb_to_hbm(dst_hbm, src_sb, B, S_tkg, prg_id):
    """Inverse of the X load: write SBUF [H0, BxS*H1] (shard-interleaved) to
    HBM [B, S_tkg, H] canonical layout. Gated on prg_id==0.

    Layout: ``src_sb[p, bs*H1 + shard*H2 + h2] == dst.flat[bs*H + shard*(H0*H2) + p*H2 + h2]``.
    """
    BxS = B * S_tkg
    dst_flat = dst_hbm.reshape((BxS * H,))
    if prg_id == 0:
        nisa.dma_copy(
            dst=dst_flat.ap(
                pattern=[
                    [H2, H0],
                    [H0 * H2, N_PRGS],
                    [1, H2],
                    [H, BxS],
                ],
                offset=0,
            ),
            src=src_sb,
            dge_mode=nisa.dge_mode.hwdge,
        )


def _multilayer_body(
    X,             # [B, S_tkg, H]                    bf16  HBM
    Wqkv_list,     # tuple of L: [H, I_QKV] pre-fused  bf16  HBM
    Wo_list,       # tuple of L: [Hq_tp*d, H]          bf16  HBM
    qn_list,       # tuple of L: [d]                  bf16  HBM (Q pre-RoPE RMSNorm)
    kn_list,       # tuple of L: [d]                  bf16  HBM (K pre-RoPE RMSNorm)
    gpre_list,     # tuple of L: [H]                  bf16  HBM (input_layernorm)
    gpost_list,    # tuple of L: [1, H]               bf16  HBM (post_attention_layernorm)
    router_list,   # tuple of L: [H, E]               bf16  HBM
    gate_up_list,  # tuple of L: [E, H, 2*I_PER_EXPERT]   bf16  HBM (TP-sharded)
    down_list,     # tuple of L: [E, I_PER_EXPERT, H]     bf16  HBM (TP-sharded)
    K_caches,      # tuple of L: [B, 1, S_max, d]     bf16  HBM (mutated in place)
    V_caches,      # tuple of L: [B, 1, S_max, d]     bf16  HBM (mutated in place)
    cos,           # [B, T, d]                        bf16  HBM (per-slot pre-indexed)
    sin,           # [B, T, d]                        bf16  HBM (per-slot pre-indexed)
    position_ids,  # [B, T]                           int32 HBM (consecutive: [p, p+1, ..., p+T-1])
    num_layers,
    replica_groups=None,
):
    """Run ``num_layers`` fused decoder layers. See the module docstring for
    the per-layer pipeline."""
    B, S_tkg, _ = X.shape
    dtype = X.dtype
    BxS = B * S_tkg
    T = BxS

    # Capture post-update K/V refs so NKI preserves the in-place scatter
    # (otherwise NCC may DCE it as a dead store to a read-only input).
    K_post = list(K_caches)
    V_post = list(V_caches)

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "transformer_qwen3_moe_speculative", (0, 1), N_PRGS
    )

    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("transformer_qwen3_moe_speculative"))
    sbm.set_auto_alloc(True)

    S_FULL = K_caches[0].shape[-2]   # attention_block_tkg derives S_max_ctx from this

    # Layer-invariant HBM scratches (mask + permuted cos/sin).
    mask_full_hbm = _build_full_mask_hbm(
        position_ids, S_FULL, B, NUM_Q_HEADS_TP, S_tkg
    )
    cos_perm_hbm = _build_permuted_rope_hbm_from_pos(
        cos, B, S_tkg, D_HEAD, name="cos_perm_hbm"
    )
    sin_perm_hbm = _build_permuted_rope_hbm_from_pos(
        sin, B, S_tkg, D_HEAD, name="sin_perm_hbm"
    )

    # W_qkv is pre-fused at weight-conversion time (the converter in
    # qwen_with_megakernel.py packs q/k/v into one [H, I_QKV]-per-rank tensor).
    # No in-kernel QKV fusion, no per-step transpose DMA, no scratch buffer —
    # the old shared scratch was both a per-step redundant HBM round-trip and
    # a cross-layer write/read aliasing hazard.

    # Load X into the SBUF residual in LNC-aware shard-interleaved layout:
    #   residual_sb[p, b*H1 + shard*H2 + h2] = X[b, s, shard*(H0*H2) + p*H2 + h2]
    # The vendored subkernels slice the SBUF input on dim 2 at shard_id*H1_SHARD,
    # so h1 must equal shard*H2 + h2. A plain channel-interleaved load is only
    # equivalent at LNC=1.
    residual_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                              name="residual_sb")
    X_flat = X.reshape((BxS * H,))
    nisa.dma_copy(
        dst=residual_sb,
        src=X_flat.ap(
            pattern=[
                [H2, H0],            # partition p
                [H, BxS],            # bs outer
                [H0 * H2, N_PRGS],   # shard
                [1, H2],             # h2 inner
            ],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )

    for layer_idx in range(num_layers):
        # ===== Attention ==================================================
        sbm.set_name_prefix(f"L{layer_idx}_attn_")
        sbm.set_auto_alloc(True)

        # Copy residual — attention_block_tkg's fused RMSNorm overwrites its
        # input, so the residual buffer must be preserved separately.
        attn_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                 name=f"L{layer_idx}_attn_in_sb")
        nisa.tensor_copy(attn_in_sb, residual_sb)
        X_sb = attn_in_sb.reshape((H0, BxS, H1))

        # Reshape gamma/qn/kn to the [1, *] shapes the subkernels expect
        # (view reshapes — no data movement).
        gpre_w  = gpre_list[layer_idx].reshape((1, H))
        qnorm_w = qn_list[layer_idx].reshape((1, D_HEAD))
        knorm_w = kn_list[layer_idx].reshape((1, D_HEAD))

        attn_result = attention_block_tkg(
            X=X_sb,
            X_hidden_dim_actual=H,
            # input RMSNorm (input_layernorm)
            rmsnorm_X_enabled=True,
            rmsnorm_X_eps=EPS,
            rmsnorm_X_gamma=gpre_w,
            # QKV projection (Qwen3 has no QKV bias)
            W_qkv=Wqkv_list[layer_idx],
            bias_qkv=None,
            quantization_type_qkv=QuantizationType.NONE,
            weight_dequant_scale_qkv=None,
            input_dequant_scale_qkv=None,
            # pre-RoPE Q/K RMSNorm (Qwen3 q_norm / k_norm)
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=EPS,
            rmsnorm_QK_pre_rope_W_Q=qnorm_w,
            rmsnorm_QK_pre_rope_W_K=knorm_w,
            # RoPE (contiguous halves)
            cos=cos_perm_hbm,
            sin=sin_perm_hbm,
            rope_contiguous_layout=True,
            # post-RoPE QK RMSNorm: unused by Qwen3
            rmsnorm_QK_post_rope_enabled=False,
            rmsnorm_QK_post_rope_eps=EPS,
            rmsnorm_QK_post_rope_W_Q=None,
            rmsnorm_QK_post_rope_W_K=None,
            # attention (no sink)
            K_cache_transposed=False,
            active_blocks_table=None,
            K_cache=K_post[layer_idx],
            V_cache=V_post[layer_idx],
            attention_mask=mask_full_hbm,
            sink=None,
            # in-place KV cache update at position_ids. _update_flat_cache
            # writes S_tkg consecutive entries starting from the BASE position,
            # so for T>1 spec (consecutive positions [p, p+1, ..., p+T-1])
            # pass only the first column [B, 1] — the cache update path
            # asserts shape (B, 1).
            update_cache=True,
            kv_cache_update_idx=position_ids[:, :1],
            # output projection (Qwen3 has no O bias)
            W_out=Wo_list[layer_idx],
            bias_out=None,
            quantization_type_out=QuantizationType.NONE,
            weight_dequant_scale_out=None,
            input_dequant_scale_out=None,
            transposed_out=True,
            out_in_sb=True,
            sbm=sbm,
        )
        attn_kernel_out_sb = attn_result[0]
        K_post[layer_idx] = attn_result[1]
        V_post[layer_idx] = attn_result[2]

        # transposed_out=True, out_in_sb=True -> [H0, H1_SHARD*BxS], F-dim
        # outer H1_SHARD / inner BxS — the layout _sb2sb_all_reduce_gather wants.
        attn_sharded_sb = attn_kernel_out_sb.reshape((H0, H1_SHARD * BxS))

        attn_gathered_sb, _ = _sb2sb_all_reduce_gather(
            attn_sharded_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS
        )

        # Free attention's stack/heap before MoE.
        while sbm.heap:
            sbm.pop_heap()
        sbm.set_auto_alloc(True)

        # Residual add #1
        nisa.tensor_tensor(dst=residual_sb, data1=residual_sb,
                           data2=attn_gathered_sb, op=nl.add)

        # ===== MoE block ==================================================
        sbm.set_name_prefix(f"L{layer_idx}_moe_")
        sbm.set_auto_alloc(True)

        # Post-attention RMSNorm.
        moe_in_sb = nl.ndarray((H0, BxS, H1), dtype=dtype, buffer=nl.sbuf,
                                name=f"L{layer_idx}_moe_in_sb")
        rmsnorm_tkg(
            input=residual_sb.reshape((H0, BxS, H1)),
            gamma=gpost_list[layer_idx],
            output=moe_in_sb,
            eps=EPS,
            hidden_actual=H,
            sbm=sbm,
        )

        # ===== Router topK ================================================
        # router_logits gets a real HBM alloc to satisfy NCC_IGCA090 (every
        # mutable_tensor needs at least one store) even though it's discarded.
        router_logits_hbm = nl.ndarray((T, E), dtype=nl.float32,
                                       buffer=nl.shared_hbm,
                                       name=f"L{layer_idx}_router_logits_scratch")
        expert_index_sb = nl.ndarray((T, TOP_K), dtype=nl.uint32,
                                     buffer=nl.sbuf,
                                     name=f"L{layer_idx}_expert_index_sb")
        expert_affinities_sb = nl.ndarray((T, E), dtype=nl.float32,
                                          buffer=nl.sbuf,
                                          name=f"L{layer_idx}_expert_affinities_sb")
        # x_sb_layout=XSBLayout_tp2013__1 is the LNC=2 shard-interleaved layout
        # that rmsnorm_tkg / attention_block_tkg produce. tp102__0 only matches
        # at LNC=1 and would mis-index the hidden state here.
        router_outputs = router_topk(
            x=moe_in_sb,
            w=router_list[layer_idx],
            w_bias=None,
            router_logits=router_logits_hbm,
            expert_affinities=expert_affinities_sb,
            expert_index=expert_index_sb,
            act_fn=RouterActFnType.SOFTMAX,
            k=TOP_K,
            x_hbm_layout=XHBMLayout_T_H__1,
            x_sb_layout=XSBLayout_tp2013__1,
            router_pre_norm=False,                # topK then softmax
            norm_topk_prob=True,                  # Qwen3 normalizes top-K probs
            skip_store_router_logits=False,
            name_prefix=f"L{layer_idx}_moe_",
        )
        # The (topK, ACT2, Scatter) path rebinds expert_affinities.
        expert_affinities_sb = router_outputs[2]

        # ===== Selective expert MoE =======================================
        # gate_up_list[i] is [E, H, 2*I_PER_EXPERT] (NxDI stride=2 sharding);
        # reshape to the [E, H, 2, I_PER_EXPERT] view moe_tkg expects.
        gate_up_w = gate_up_list[layer_idx].reshape((E, H, 2, I_PER_EXPERT))

        # moe_tkg's selective-expert path (T=1) shards on H and expects
        # hidden_input as the PER-SHARD slice [H0, T, H1_SHARD]. Passing the
        # full [H0, T, H1] makes both cores read columns [0:H1_SHARD), pairing
        # core 1's weight rows with core 0's H values. Mirrors
        # nkilib/core/moe_block/moe_block_tkg.py:309-314. Guards:
        # tests/qwen3_moe/test_moe_vs_hf.py and the call-site shape guard in
        # test_speculative_megakernel.py.
        moe_in_pershard_sb = nl.ndarray((H0, BxS, H1_SHARD), dtype=dtype,
                                         buffer=nl.sbuf,
                                         name=f"L{layer_idx}_moe_in_pershard_sb")
        nisa.tensor_copy(
            dst=moe_in_pershard_sb,
            src=moe_in_sb[:, :, nl.ds(prg_id * H1_SHARD, H1_SHARD)],
        )

        moe_out_sb = moe_tkg(
            hidden_input=moe_in_pershard_sb,
            expert_gate_up_weights=gate_up_w,
            expert_down_weights=down_list[layer_idx],     # [E, I_PER_EXPERT, H]
            expert_affinities=expert_affinities_sb,
            expert_index=expert_index_sb,
            is_all_expert=False,
            expert_gate_up_bias=None,                     # Qwen3 has no expert biases
            expert_down_bias=None,
            activation_fn=ActFnType.SiLU,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            name_prefix=f"L{layer_idx}_moe_",
            gate_clamp_upper_limit=None,                  # Qwen3 has no expert clamping
            gate_clamp_lower_limit=None,
            up_clamp_upper_limit=None,
            up_clamp_lower_limit=None,
            output_in_sbuf=True,
            output_dtype=dtype,
        )
        # moe_out_sb is [H0, BxS, H1_SHARD] (F: BxS outer, H1_SHARD inner).
        # Copy into [H0, H1_SHARD*BxS] (H1_SHARD outer, BxS inner) — the
        # contiguous layout _sb2sb_all_reduce_gather expects. The per-column
        # copy avoids the illegal-partition-step / non-unit-stride asserts
        # that a strided view would trip.
        moe_sharded_sb = nl.ndarray((H0, H1_SHARD * BxS), dtype=dtype,
                                     buffer=nl.sbuf,
                                     name=f"L{layer_idx}_moe_sharded_sb")
        for h1s in range(H1_SHARD):
            nisa.tensor_copy(
                dst=moe_sharded_sb[:, h1s * BxS : (h1s + 1) * BxS],
                src=moe_out_sb[:, :, h1s],
            )

        moe_gathered_sb, _ = _sb2sb_all_reduce_gather(
            moe_sharded_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS
        )

        while sbm.heap:
            sbm.pop_heap()
        sbm.set_auto_alloc(True)

        # Residual add #2
        nisa.tensor_tensor(dst=residual_sb, data1=residual_sb,
                           data2=moe_gathered_sb, op=nl.add)

    # Single HBM store of the post-residual hidden state (inverse of X load).
    Y = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="Y")
    if prg_id == 0:
        Y_flat = Y.reshape((BxS * H,))
        nisa.dma_copy(
            dst=Y_flat.ap(
                pattern=[
                    [H2, H0],
                    [H, BxS],
                    [H0 * H2, N_PRGS],
                    [1, H2],
                ],
                offset=0,
            ),
            src=residual_sb,
            dge_mode=nisa.dge_mode.hwdge,
        )
    if N_PRGS > 1:
        nisa.core_barrier(data=Y, cores=(0, 1))

    # Return KV refs so NCC preserves the in-place scatters; NxDI's
    # model_wrapper aliases each back to its kv_mgr slot.
    return (Y,) + tuple(K_post) + tuple(V_post)


# ---------------------------------------------------------------------------
# Code-gen wrapper — explicit per-layer tensor args
# ---------------------------------------------------------------------------

def _build_multilayer_kernel(num_layers: int):
    """Code-gen a kernel function with explicit per-layer tensor args.

    NKI's frontend classifies tuple/list top-level args as scalars (no HBM
    binding), which breaks the input-output aliasing needed for in-place KV
    scatter. So each weight and KV cache must be its own positional arg.
    """
    def names(prefix):
        return [f"{prefix}_{i:02d}" for i in range(num_layers)]

    wqkv_names   = names("Wqkv")
    wo_names     = names("Wo")
    qn_names     = names("Qn")
    kn_names     = names("Kn")
    gpre_names   = names("Gpre")
    gpost_names  = names("Gpost")
    router_names = names("Router")
    gu_names     = names("GateUp")
    down_names   = names("Down")
    k_names      = names("K")
    v_names      = names("V")

    sig = ",\n    ".join(
        ["X"]
        + wqkv_names + wo_names
        + qn_names + kn_names + gpre_names + gpost_names
        + router_names + gu_names + down_names
        + k_names + v_names
        + ["cos", "sin", "position_ids"]
    )

    def tup(ns):
        return "(" + ", ".join(ns) + ",)"

    src = (
        f"def transformer_qwen3_moe_speculative(\n"
        f"    {sig},\n"
        f"    replica_groups=None,\n"
        f"):\n"
        f"    Wqkv_list    = {tup(wqkv_names)}\n"
        f"    Wo_list      = {tup(wo_names)}\n"
        f"    qn_list      = {tup(qn_names)}\n"
        f"    kn_list      = {tup(kn_names)}\n"
        f"    gpre_list    = {tup(gpre_names)}\n"
        f"    gpost_list   = {tup(gpost_names)}\n"
        f"    router_list  = {tup(router_names)}\n"
        f"    gate_up_list = {tup(gu_names)}\n"
        f"    down_list    = {tup(down_names)}\n"
        f"    K_caches     = {tup(k_names)}\n"
        f"    V_caches     = {tup(v_names)}\n"
        f"    return _multilayer_body(\n"
        f"        X, Wqkv_list, Wo_list,\n"
        f"        qn_list, kn_list, gpre_list, gpost_list,\n"
        f"        router_list, gate_up_list, down_list,\n"
        f"        K_caches, V_caches,\n"
        f"        cos, sin, position_ids,\n"
        f"        num_layers={num_layers},\n"
        f"        replica_groups=replica_groups,\n"
        f"    )\n"
    )

    fname = f"<generated:qwen3_moe_speculative_L{num_layers}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
    code = compile(src, fname, "exec")
    ns = {"_multilayer_body": _multilayer_body}
    exec(code, ns)
    return ns["transformer_qwen3_moe_speculative"]


transformer_qwen3_moe_speculative = _build_multilayer_kernel(NUM_LAYERS)
transformer_qwen3_moe_speculative_jit = nki.jit(transformer_qwen3_moe_speculative)

_kernel_cache: dict = {NUM_LAYERS: transformer_qwen3_moe_speculative_jit}


def get_multilayer_kernel_jit(num_layers: int):
    """Return the jit-compiled multilayer kernel, indexable by LNC degree
    (``[1]`` for LNC=1, ``[2]`` for LNC=2). The integration selects ``[2]``."""
    if num_layers not in _kernel_cache:
        _kernel_cache[num_layers] = nki.jit(_build_multilayer_kernel(num_layers))
    return _kernel_cache[num_layers]
