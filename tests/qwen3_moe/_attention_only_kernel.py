"""Single-layer attention-only NKI kernel that mirrors the speculative
megakernel's layer-0 attention path EXACTLY.

This file is only used by ``test_attention_vs_hf.py``. It exists so the
test can drive the attention portion of ``transformer_qwen3_moe_speculative``
in isolation against a PyTorch reference, without involving MoE / multilayer
state. The kernel must be a strict subset of the production megakernel so
any call-site bug present in the megakernel is also reproduced here.

What this kernel does, in order (matches lines 631-715 of the production
megakernel ``_multilayer_body`` for layer 0):

1. Build mask, cos/sin perm scratches (helpers re-used).
2. Fuse Wq/Wk/Wv into W_qkv (helper re-used).
3. Load X into SBUF in the 4-level LNC-aware shard-interleaved layout.
4. Call ``attention_block_tkg`` with the SAME kwargs as the production
   megakernel (input RMSNorm + QKV proj + pre-RoPE Q/K RMSNorm + RoPE +
   SDPA + W_out, in-place K/V scatter at ``position_ids``).
5. ``_sb2sb_all_reduce_gather`` across LNC cores (also across TP ranks in
   multi-rank execution; trivial sum-of-one with a single-rank replica
   group).
6. Inverse-of-X-load DMA to produce a canonical [B, S_tkg, H] output.

No residual add, no post-attention RMSNorm, no MoE. The output is the
*pure attention block output*, ready to be compared against
``hidden -> input_layernorm -> Qwen3MoeAttention.forward`` from the HF
reference.

The kernel returns ``(Y_attn, K_post, V_post)`` — Y_attn is the canonical
[B, S_tkg, H] attention output; K_post / V_post are the post-scatter KV
cache handles (NKI-aliasing requirement for in-place writes to be
preserved).

Constants (H, D_HEAD, NUM_Q_HEADS_TP, NUM_KV_HEADS_TP, I_QKV, H0, H1,
H2, H1_SHARD, N_PRGS, PMAX, EPS) are re-exported from the production
megakernel module to guarantee the test sees the exact same shapes.
"""

import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl

# Import the EXACT same helpers/constants used by the production megakernel
# so we exercise the same code paths.
from megakernels.qwen3_moe.transformer_qwen3_moe_speculative import (
    H,
    H0,
    H1,
    H2,
    H1_SHARD,
    N_PRGS,
    PMAX,
    EPS,
    D_HEAD,
    NUM_Q_HEADS_TP,
    NUM_KV_HEADS_TP,
    I_QKV,
    SBM_SIZE_BYTES,
    _fuse_qkv_weights,
    _build_full_mask_hbm,
    _build_permuted_rope_hbm_from_pos,
    _store_shard_interleaved_sb_to_hbm,
)

from nki_kernels.attention import attention_block_tkg
from nki_kernels.moe import rmsnorm_tkg  # not used; just to keep parity if needed
from nkilib.core.utils.allocator import BufferManager, Logger
from nkilib.core.utils.common_types import QuantizationType
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.experimental.transformer.transformer_tkg import _sb2sb_all_reduce_gather


def _attention_only_body(
    X,             # [B, S_tkg, H]                     bf16 HBM
    Wq,            # [NUM_Q_HEADS_TP * D_HEAD, H]      bf16 HBM
    Wk,            # [NUM_KV_HEADS_TP * D_HEAD, H]     bf16 HBM
    Wv,            # [NUM_KV_HEADS_TP * D_HEAD, H]     bf16 HBM
    Wo,            # [NUM_Q_HEADS_TP * D_HEAD, H]      bf16 HBM
    qn,            # [D_HEAD]                          bf16 HBM
    kn,            # [D_HEAD]                          bf16 HBM
    gpre,          # [H]                               bf16 HBM
    K_cache,       # [B, 1, S_max, D_HEAD]             bf16 HBM (mutated in place)
    V_cache,       # [B, 1, S_max, D_HEAD]             bf16 HBM (mutated in place)
    cos,           # [B, D_HEAD]                       bf16 HBM (pre-indexed at position)
    sin,           # [B, D_HEAD]                       bf16 HBM (pre-indexed at position)
    position_ids,  # [B, 1]                            int32 HBM
    replica_groups=None,
):
    B, S_tkg, _ = X.shape
    dtype = X.dtype
    BxS = B * S_tkg

    # Preserve in-place KV scatter from being DCE'd.
    K_post = K_cache
    V_post = V_cache

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "attention_only_kernel", (0, 1), N_PRGS
    )

    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("attention_only_kernel"))
    sbm.set_auto_alloc(True)

    S_FULL = K_cache.shape[-2]

    # ---- DEBUG dump HBM tensors. These get returned as extra outputs of the
    # kernel so the test can inspect the intermediate states.
    attn_in_dump_hbm   = nl.ndarray((B, S_tkg, H), dtype=dtype,
                                    buffer=nl.shared_hbm, name="dbg_attn_in")
    attn_gathered_dump_hbm = nl.ndarray((B, S_tkg, H), dtype=dtype,
                                        buffer=nl.shared_hbm, name="dbg_attn_gathered")
    # Per-shard pre-AR-gather output (each shard writes its slice).
    # Shape: [N_PRGS, H0, H1_SHARD, BxS] — shard prg_id writes into [prg_id, ...].
    attn_pershard_dump_hbm = nl.ndarray(
        (N_PRGS, H0, H1_SHARD, B * S_tkg), dtype=dtype,
        buffer=nl.shared_hbm, name="dbg_attn_pershard",
    )

    # Layer-invariant HBM scratches.
    mask_full_hbm = _build_full_mask_hbm(
        position_ids, S_FULL, B, NUM_Q_HEADS_TP, S_tkg
    )
    cos_perm_hbm = _build_permuted_rope_hbm_from_pos(
        cos, B, S_tkg, D_HEAD, name="cos_perm_hbm"
    )
    sin_perm_hbm = _build_permuted_rope_hbm_from_pos(
        sin, B, S_tkg, D_HEAD, name="sin_perm_hbm"
    )

    # Fused W_qkv scratch.
    W_qkv_scratch = nl.ndarray((H, I_QKV), dtype=dtype, buffer=nl.shared_hbm,
                               name="W_qkv_scratch_hbm")

    # ---- Load X into SBUF in shard-interleaved layout ----------------------
    residual_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                             name="residual_sb")
    X_flat = X.reshape((BxS * H,))
    nisa.dma_copy(
        dst=residual_sb,
        src=X_flat.ap(
            pattern=[
                [H2, H0],            # partition p
                [H0 * H2, N_PRGS],   # shard outer
                [1, H2],             # h2 inner
                [H, BxS],            # batch
            ],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )

    # ---- DEBUG: dump residual_sb (post X-load) via the same shard-interleaved
    # inverse store. Round-trips through HBM so the test can compare against
    # the input X (must match bit-for-bit modulo bf16 trunc).
    _store_shard_interleaved_sb_to_hbm(attn_in_dump_hbm, residual_sb, B, S_tkg, prg_id)

    # ===== Fuse Wq/Wk/Wv into W_qkv =====================================
    sbm.set_name_prefix("L0_qkvfuse_")
    sbm.set_auto_alloc(True)
    _fuse_qkv_weights(Wq, Wk, Wv, W_qkv_scratch)

    # ===== Attention block =============================================
    sbm.set_name_prefix("L0_attn_")
    sbm.set_auto_alloc(True)

    attn_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                            name="L0_attn_in_sb")
    nisa.tensor_copy(attn_in_sb, residual_sb)
    X_sb = attn_in_sb.reshape((H0, BxS, H1))

    gpre_w  = gpre.reshape((1, H))
    qnorm_w = qn.reshape((1, D_HEAD))
    knorm_w = kn.reshape((1, D_HEAD))

    attn_result = attention_block_tkg(
        X=X_sb,
        X_hidden_dim_actual=H,
        rmsnorm_X_enabled=True,
        rmsnorm_X_eps=EPS,
        rmsnorm_X_gamma=gpre_w,
        W_qkv=W_qkv_scratch,
        bias_qkv=None,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=True,
        rmsnorm_QK_pre_rope_eps=EPS,
        rmsnorm_QK_pre_rope_W_Q=qnorm_w,
        rmsnorm_QK_pre_rope_W_K=knorm_w,
        cos=cos_perm_hbm,
        sin=sin_perm_hbm,
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=EPS,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False,
        active_blocks_table=None,
        K_cache=K_post,
        V_cache=V_post,
        attention_mask=mask_full_hbm,
        sink=None,
        update_cache=True,
        kv_cache_update_idx=position_ids,
        W_out=Wo,
        bias_out=None,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=True,
        out_in_sb=True,
        sbm=sbm,
    )
    attn_kernel_out_sb = attn_result[0]
    K_post = attn_result[1]
    V_post = attn_result[2]

    attn_sharded_sb = attn_kernel_out_sb.reshape((H0, H1_SHARD * BxS))

    # ---- DEBUG: dump per-shard pre-AR-gather attention output. Each shard
    # writes its [H0, H1_SHARD, BxS] slice into the [prg_id, ...] slot.
    # The native shape of attn_kernel_out_sb (transposed_out=True, out_in_sb=True)
    # is [H0, H1_SHARD, BxS]; we copy it out so the test can compare against a
    # PyTorch-computed per-shard reference.
    nisa.dma_copy(
        dst=attn_pershard_dump_hbm[prg_id, :, :, :],
        src=attn_kernel_out_sb,
    )

    attn_gathered_sb, _ = _sb2sb_all_reduce_gather(
        attn_sharded_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS
    )

    # ---- DEBUG: dump attn_gathered (post AR-gather, pre store) ----
    _store_shard_interleaved_sb_to_hbm(attn_gathered_dump_hbm, attn_gathered_sb, B, S_tkg, prg_id)

    while sbm.heap:
        sbm.pop_heap()
    sbm.set_auto_alloc(True)

    # ===== Inverse-of-X-load store: SBUF shard-interleaved -> HBM [B,S,H] ===
    Y = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="Y_attn")
    _store_shard_interleaved_sb_to_hbm(Y, attn_gathered_sb, B, S_tkg, prg_id)
    if N_PRGS > 1:
        nisa.core_barrier(data=Y, cores=(0, 1))

    return (Y, K_post, V_post, attn_in_dump_hbm, attn_gathered_dump_hbm,
            attn_pershard_dump_hbm)


@nki.jit
def attention_only_kernel(
    X,
    Wq, Wk, Wv, Wo,
    qn, kn, gpre,
    K_cache, V_cache,
    cos, sin, position_ids,
    replica_groups=None,
):
    return _attention_only_body(
        X, Wq, Wk, Wv, Wo, qn, kn, gpre,
        K_cache, V_cache, cos, sin, position_ids,
        replica_groups=replica_groups,
    )


def get_attention_only_kernel_jit():
    """Return the jit-compiled attention-only kernel. Use ``[2]`` for LNC=2
    (matches the production megakernel)."""
    return attention_only_kernel


# ===========================================================================
# Debug variant: same as attention_only_kernel but with W_out=None and
# update_cache=False, returning the RAW pre-Wo attention output in canonical
# [B, q_heads, d_head, S_tkg] HBM layout.
#
# Why a separate kernel: attention_block_tkg's `W_out=None` path requires
# `transposed_out=False` and produces shape [B, q_heads, d_head, S_tkg] @ HBM,
# which is structurally incompatible with the production wrapper's
# AR-gather + shard-interleaved store. Splitting into two kernels lets us
# compare the raw attention math against the PyTorch reference's
# `torch.matmul(attn_weights, V_rep)` output independent of the W_out
# projection, AR-gather, and final store.
#
# K/V cache write semantics: this kernel runs with update_cache=False, so
# K_cache / V_cache are NOT mutated — the test can reuse the same fresh
# K/V cache tensors for both kernels via separate invocations.
# ===========================================================================


def _attention_only_pre_wo_body(
    X,             # [B, S_tkg, H]                     bf16 HBM
    Wq,            # [NUM_Q_HEADS_TP * D_HEAD, H]      bf16 HBM
    Wk,            # [NUM_KV_HEADS_TP * D_HEAD, H]     bf16 HBM
    Wv,            # [NUM_KV_HEADS_TP * D_HEAD, H]     bf16 HBM
    qn,            # [D_HEAD]                          bf16 HBM
    kn,            # [D_HEAD]                          bf16 HBM
    gpre,          # [H]                               bf16 HBM
    K_cache,       # [B, 1, S_max, D_HEAD]             bf16 HBM (NOT mutated)
    V_cache,       # [B, 1, S_max, D_HEAD]             bf16 HBM (NOT mutated)
    cos,           # [B, D_HEAD]                       bf16 HBM
    sin,           # [B, D_HEAD]                       bf16 HBM
    position_ids,  # [B, 1]                            int32 HBM
    replica_groups=None,
):
    """Returns raw pre-Wo attention output [B, q_heads, d_head, S_tkg] @ HBM."""
    B, S_tkg, _ = X.shape
    dtype = X.dtype
    BxS = B * S_tkg

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "attention_only_pre_wo_kernel", (0, 1), N_PRGS
    )

    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("attention_only_pre_wo_kernel"))
    sbm.set_auto_alloc(True)

    S_FULL = K_cache.shape[-2]

    mask_full_hbm = _build_full_mask_hbm(
        position_ids, S_FULL, B, NUM_Q_HEADS_TP, S_tkg
    )
    cos_perm_hbm = _build_permuted_rope_hbm_from_pos(
        cos, B, S_tkg, D_HEAD, name="cos_perm_hbm_prewo"
    )
    sin_perm_hbm = _build_permuted_rope_hbm_from_pos(
        sin, B, S_tkg, D_HEAD, name="sin_perm_hbm_prewo"
    )

    W_qkv_scratch = nl.ndarray((H, I_QKV), dtype=dtype, buffer=nl.shared_hbm,
                               name="W_qkv_scratch_hbm_prewo")

    # X load (same as production)
    residual_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                             name="residual_sb_prewo")
    X_flat = X.reshape((BxS * H,))
    nisa.dma_copy(
        dst=residual_sb,
        src=X_flat.ap(
            pattern=[
                [H2, H0],
                [H0 * H2, N_PRGS],
                [1, H2],
                [H, BxS],
            ],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )

    sbm.set_name_prefix("L0_qkvfuse_prewo_")
    sbm.set_auto_alloc(True)
    _fuse_qkv_weights(Wq, Wk, Wv, W_qkv_scratch)

    sbm.set_name_prefix("L0_attn_prewo_")
    sbm.set_auto_alloc(True)

    attn_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                            name="L0_attn_in_sb_prewo")
    nisa.tensor_copy(attn_in_sb, residual_sb)
    X_sb = attn_in_sb.reshape((H0, BxS, H1))

    gpre_w  = gpre.reshape((1, H))
    qnorm_w = qn.reshape((1, D_HEAD))
    knorm_w = kn.reshape((1, D_HEAD))

    # NOTE: W_out=None and update_cache=False.
    # With W_out=None, transposed_out must be False. With out_in_sb=False, the
    # returned attn_out is [B, q_heads, d_head, S_tkg] @ HBM. This is the
    # canonical layout we can compare directly with the PyTorch reference's
    # `torch.matmul(attn_weights, V_rep)` (after transposing).
    attn_result = attention_block_tkg(
        X=X_sb,
        X_hidden_dim_actual=H,
        rmsnorm_X_enabled=True,
        rmsnorm_X_eps=EPS,
        rmsnorm_X_gamma=gpre_w,
        W_qkv=W_qkv_scratch,
        bias_qkv=None,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=True,
        rmsnorm_QK_pre_rope_eps=EPS,
        rmsnorm_QK_pre_rope_W_Q=qnorm_w,
        rmsnorm_QK_pre_rope_W_K=knorm_w,
        cos=cos_perm_hbm,
        sin=sin_perm_hbm,
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=EPS,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False,
        active_blocks_table=None,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=mask_full_hbm,
        sink=None,
        update_cache=False,
        kv_cache_update_idx=None,
        W_out=None,                 # <-- KEY: no output projection
        bias_out=None,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        out_in_sb=False,
        sbm=sbm,
    )
    # attn_result[0]: [B, q_heads, d_head, S_tkg] @ HBM
    return attn_result[0]


@nki.jit
def attention_only_pre_wo_kernel(
    X,
    Wq, Wk, Wv,
    qn, kn, gpre,
    K_cache, V_cache,
    cos, sin, position_ids,
    replica_groups=None,
):
    return _attention_only_pre_wo_body(
        X, Wq, Wk, Wv, qn, kn, gpre,
        K_cache, V_cache, cos, sin, position_ids,
        replica_groups=replica_groups,
    )


# ===========================================================================
# Debug variant: dump Q post-RMSNorm-post-RoPE (i.e. immediately before the
# QK matmul). Uses attention_block_tkg with skip_attention=True, which
# returns Q_tkg_sb early instead of computing attention.
# Output shape: [d_head, B * q_heads * S_tkg] @ HBM, indexed
#   q[d, b * q_heads * S_tkg + n * S_tkg + s]
# ===========================================================================


def _attention_only_q_dump_body(
    X,             # [B, S_tkg, H]
    Wq, Wk, Wv,
    qn, kn, gpre,
    K_cache, V_cache,
    cos, sin, position_ids,
    replica_groups=None,
):
    B, S_tkg, _ = X.shape
    dtype = X.dtype
    BxS = B * S_tkg

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "attention_only_q_dump_kernel", (0, 1), N_PRGS
    )

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("attention_only_q_dump_kernel"))
    sbm.set_auto_alloc(True)

    S_FULL = K_cache.shape[-2]

    mask_full_hbm = _build_full_mask_hbm(
        position_ids, S_FULL, B, NUM_Q_HEADS_TP, S_tkg
    )
    cos_perm_hbm = _build_permuted_rope_hbm_from_pos(
        cos, B, S_tkg, D_HEAD, name="cos_perm_hbm_qdump"
    )
    sin_perm_hbm = _build_permuted_rope_hbm_from_pos(
        sin, B, S_tkg, D_HEAD, name="sin_perm_hbm_qdump"
    )

    W_qkv_scratch = nl.ndarray((H, I_QKV), dtype=dtype, buffer=nl.shared_hbm,
                               name="W_qkv_scratch_hbm_qdump")

    residual_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                             name="residual_sb_qdump")
    X_flat = X.reshape((BxS * H,))
    nisa.dma_copy(
        dst=residual_sb,
        src=X_flat.ap(
            pattern=[[H2, H0], [H0 * H2, N_PRGS], [1, H2], [H, BxS]],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )

    sbm.set_name_prefix("L0_qkvfuse_qdump_")
    sbm.set_auto_alloc(True)
    _fuse_qkv_weights(Wq, Wk, Wv, W_qkv_scratch)

    sbm.set_name_prefix("L0_attn_qdump_")
    sbm.set_auto_alloc(True)

    attn_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                            name="L0_attn_in_sb_qdump")
    nisa.tensor_copy(attn_in_sb, residual_sb)
    X_sb = attn_in_sb.reshape((H0, BxS, H1))

    gpre_w  = gpre.reshape((1, H))
    qnorm_w = qn.reshape((1, D_HEAD))
    knorm_w = kn.reshape((1, D_HEAD))

    # NOTE: skip_attention=True returns Q_tkg_sb directly (post RMSNorm,
    # post RoPE). update_cache=False, W_out=None.
    attn_result = attention_block_tkg(
        X=X_sb,
        X_hidden_dim_actual=H,
        rmsnorm_X_enabled=True,
        rmsnorm_X_eps=EPS,
        rmsnorm_X_gamma=gpre_w,
        W_qkv=W_qkv_scratch,
        bias_qkv=None,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=True,
        rmsnorm_QK_pre_rope_eps=EPS,
        rmsnorm_QK_pre_rope_W_Q=qnorm_w,
        rmsnorm_QK_pre_rope_W_K=knorm_w,
        cos=cos_perm_hbm,
        sin=sin_perm_hbm,
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=EPS,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False,
        active_blocks_table=None,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=mask_full_hbm,
        sink=None,
        update_cache=False,
        kv_cache_update_idx=None,
        W_out=None,
        bias_out=None,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        out_in_sb=False,
        sbm=sbm,
        skip_attention=True,    # <-- KEY: short-circuit, return Q
    )
    # attn_result[0]: Q post-RMSNorm post-RoPE,
    #   shape [d_head, B*q_heads*S_tkg] @ HBM
    #   indexing: [d, b*q_heads*S + n*S + s]
    return attn_result[0]


@nki.jit
def attention_only_q_dump_kernel(
    X,
    Wq, Wk, Wv,
    qn, kn, gpre,
    K_cache, V_cache,
    cos, sin, position_ids,
    replica_groups=None,
):
    return _attention_only_q_dump_body(
        X, Wq, Wk, Wv, qn, kn, gpre,
        K_cache, V_cache, cos, sin, position_ids,
        replica_groups=replica_groups,
    )
