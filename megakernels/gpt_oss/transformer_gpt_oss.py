"""GPT-OSS-20B multi-layer fused TKG megakernel.

Runs all `num_layers` decoder layers in a single NKI invocation using stock
nkilib subkernels (attention_block_tkg, rmsnorm_tkg, router_topk, moe_tkg).
SBUF-resident residual; in-place KV cache scatter at position_ids.
Even layers use sliding-window attention; odd layers use full attention.

Target: Trainium3, TP=8, LNC=1.
Entry point: transformer_gpt_oss_tkg_multilayer_jit[1](...)
"""

import linecache

import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl

from nki_kernels.attention import attention_block_tkg
from nki_kernels.moe import (
    XHBMLayout_T_H__1,
    XSBLayout_tp102__0,
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

NUM_LAYERS = 24
H          = 3072    # padded from 2880
H_ACTUAL   = 2880
H0         = 128
H1         = H // H0
N_PRGS     = 1
H1_SHARD   = H1 // N_PRGS
EPS        = 1e-5
PMAX       = H0
E          = 32
TOP_K      = 4
SLIDING_WIN = 128  # config.sliding_window for gpt-oss; matches SWA cache size

SBM_SIZE_BYTES = 200 * 1024


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
    """Build full-attention mask [S_ctx, B, num_heads, S_tkg] from position_ids.

    Rule per (t, b): mask = 1 if t < pos[b] OR t == S_ctx - 1 else 0.
    See tests/gpt_oss/test_phase4_mask_rope.py:_kernel_full_mask_impl.
    """
    P_MAX = 128
    # S_ctx must be a multiple of 128; gpt-oss buckets (128, 256, 384, 640) all are.
    n_tile = S_ctx // P_MAX

    out = nl.ndarray((S_ctx, B, num_heads, S_tkg), dtype=nl.bfloat16,
                     buffer=nl.shared_hbm, name="mask_full_hbm")

    # ----------------------- pos: load + broadcast ------------------------
    pos_one = nl.ndarray((1, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos_one, src=position_ids.reshape((1, B)))

    pos_bcast = nl.ndarray((P_MAX, B), dtype=nl.float32, buffer=nl.sbuf)
    _stream_shuffle_broadcast(pos_one, pos_bcast)

    # ----------------------- iota: t = 0..S_ctx-1 -------------------------
    iota_tile = nl.ndarray((P_MAX, n_tile, num_heads), dtype=nl.float32,
                           buffer=nl.sbuf)
    nisa.iota(
        iota_tile,
        pattern=[[P_MAX, n_tile], [0, num_heads]],
        offset=0,
        channel_multiplier=1,
    )

    # ----------------------- mask: (iota < pos) OR (iota == S_ctx - 1) ----
    # tensor_scalar requires fp32 arithmetic operands; bf16 cast at SBUF->HBM.
    mask_lt = nl.ndarray((P_MAX, n_tile, num_heads), dtype=nl.float32,
                         buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=mask_lt,
        data=iota_tile,
        op0=nl.less,
        operand0=pos_bcast[:, 0:1],
    )

    mask_eq = nl.ndarray((P_MAX, n_tile, num_heads), dtype=nl.float32,
                         buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=mask_eq,
        data=iota_tile,
        op0=nl.equal,
        operand0=S_ctx - 1,
    )

    mask_or = nl.ndarray((P_MAX, n_tile, num_heads), dtype=nl.bfloat16,
                         buffer=nl.sbuf)
    nisa.tensor_tensor(mask_or, mask_lt, mask_eq, op=nl.maximum)

    # SBUF [P_MAX, n_tile, num_heads] -> HBM [S_ctx, B, num_heads, S_tkg]
    # HBM flat offset(p, k, h) = (k*P_MAX + p) * B * num_heads * S_tkg + h * S_tkg
    out_flat = out.reshape((S_ctx * B * num_heads * S_tkg,))
    nisa.dma_copy(
        dst=out_flat.ap(
            pattern=[[num_heads * S_tkg, P_MAX],
                     [P_MAX * num_heads * S_tkg, n_tile],
                     [S_tkg, num_heads]],
            offset=0,
        ),
        src=mask_or,
    )

    return out


def _build_swa_mask_hbm(position_ids, S_ctx, B, num_heads, S_tkg, W):
    """Build SWA mask [S_ctx, B, num_heads, S_tkg] from position_ids.

    eff_pos = min(pos, W-1); mask = 1 if t < eff_pos OR t == S_ctx-1 else 0.
    See tests/gpt_oss/test_phase4_mask_rope.py:_kernel_swa_mask_impl.
    """
    P_MAX = 128
    # gpt-oss: S_ctx == W == 128, so single tile.
    assert S_ctx == P_MAX, "SWA S_ctx must equal 128 for gpt-oss"

    out = nl.ndarray((S_ctx, B, num_heads, S_tkg), dtype=nl.bfloat16,
                     buffer=nl.shared_hbm, name="mask_window_hbm")

    # ----------------------- pos: load + clamp + broadcast ----------------
    pos_one = nl.ndarray((1, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos_one, src=position_ids.reshape((1, B)))

    # eff_pos = min(pos, W - 1) — unifies short / long pos cases.
    eff_pos_one = nl.ndarray((1, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=eff_pos_one,
        data=pos_one,
        op0=nl.minimum,
        operand0=W - 1,
    )

    pos_bcast = nl.ndarray((P_MAX, B), dtype=nl.float32, buffer=nl.sbuf)
    _stream_shuffle_broadcast(eff_pos_one, pos_bcast)

    # ----------------------- iota: t = 0..S_ctx-1 -------------------------
    iota_tile = nl.ndarray((P_MAX, num_heads), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(
        iota_tile,
        pattern=[[0, num_heads]],
        offset=0,
        channel_multiplier=1,
    )

    # ----------------------- mask: (iota < eff_pos) OR (iota == S-1) ------
    mask_lt = nl.ndarray((P_MAX, num_heads), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=mask_lt,
        data=iota_tile,
        op0=nl.less,
        operand0=pos_bcast[:, 0:1],
    )

    mask_eq = nl.ndarray((P_MAX, num_heads), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=mask_eq,
        data=iota_tile,
        op0=nl.equal,
        operand0=S_ctx - 1,
    )

    mask_or = nl.ndarray((P_MAX, num_heads), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(mask_or, mask_lt, mask_eq, op=nl.maximum)

    out_flat = out.reshape((S_ctx * B * num_heads * S_tkg,))
    nisa.dma_copy(
        dst=out_flat.ap(
            pattern=[[num_heads * S_tkg, P_MAX],
                     [S_tkg, num_heads]],
            offset=0,
        ),
        src=mask_or,
    )

    return out


def _build_permuted_rope_hbm(rope_unperm, name):
    """Permute (B, S_tkg, half_d) -> (half_d, B, S_tkg) via strided DMA.

    See tests/gpt_oss/test_phase4_mask_rope.py:_kernel_permute_cos_impl.
    """
    Bd, Sd, D = rope_unperm.shape  # B, S_tkg, half_d
    BS = Bd * Sd

    out = nl.ndarray((D, Bd, Sd), dtype=rope_unperm.dtype, buffer=nl.shared_hbm,
                     name=name)

    # SBUF tile in the post-permute layout: [half_d partitions, B*S_tkg free]
    tile = nl.ndarray((D, BS), dtype=rope_unperm.dtype, buffer=nl.sbuf)

    # Strided HBM load: partition stride 1, free stride D in flat input.
    rope_flat = rope_unperm.reshape((BS * D,))
    nisa.dma_copy(
        dst=tile,
        src=rope_flat.ap(
            pattern=[[1, D],     # partition: D rows, stride 1 in flat input
                     [D, BS]],   # free dim:  BS cols, stride D in flat input
            offset=0,
        ),
    )

    # SBUF [D, BS] -> HBM [D, B, S_tkg] is a contiguous store.
    nisa.dma_copy(dst=out.reshape((D, BS)), src=tile)

    return out


def _multilayer_body(
    X,                   # [B, S_tkg, H]                        bf16  HBM
    Wqkv_list,           # tuple of L: [H, (q+2*kv)*d_head]     bf16  HBM (fused QKV)
    bqkv_list,           # tuple of L: [1, (q+2*kv)*d_head]     bf16  HBM
    Wo_list,             # tuple of L: [q_heads*d_head, H]      bf16  HBM
    bo_list,             # tuple of L: [1, H]                   bf16  HBM
    sink_list,           # tuple of L: [num_heads_per_rank, 1]  bf16  HBM
    gpre_list,           # tuple of L: [1, H]                   bf16  HBM (input_layernorm)
    gpost_list,          # tuple of L: [1, H]                   bf16  HBM (post_attn_layernorm)
    router_w_list,       # tuple of L: [H, E]                   bf16  HBM
    router_b_list,       # tuple of L: [1, E]                   bf16  HBM
    gate_up_list,        # tuple of L: [E, H, 2, I]             bf16  HBM (fused gate+up)
    gate_up_b_list,      # tuple of L: [E, 2, I]                bf16  HBM
    down_list,           # tuple of L: [E, I, H]                bf16  HBM
    down_b_list,         # tuple of L: [E, H]                   bf16  HBM
    K_caches,            # tuple of L: per-layer K cache        bf16  HBM (mutated in-place)
    V_caches,            # tuple of L: per-layer V cache        bf16  HBM (mutated in-place)
    cos_unperm,          # [B, S_tkg, d_head//2]                bf16  HBM (RoPE cos, un-permuted)
    sin_unperm,          # [B, S_tkg, d_head//2]                bf16  HBM (RoPE sin, un-permuted)
    position_ids,        # [B, 1]                               int32 HBM (full layers)
    position_ids_window, # [B, 1]                               int32 HBM (SWA layers, modulo'd)
    num_layers,
    num_heads_per_rank,  # q_heads after TP shard (e.g. 16 at TP=4, 8 at TP=8)
    replica_groups=None,
):
    B, S_tkg, _ = X.shape
    dtype = X.dtype
    BxS = B * S_tkg
    T = BxS

    # Capture post-update K/V refs so NKI preserves the in-place scatter.
    K_post = list(K_caches)
    V_post = list(V_caches)

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "transformer_gpt_oss_tkg_multilayer", (0, 1), N_PRGS
    )

    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("transformer_gpt_oss_tkg_multilayer"))
    sbm.set_auto_alloc(True)

    # Cache sizes by attention type. K_cache layout is
    # [B, kv_heads, S_max_ctx, d_head] (4-D, kv_heads=1 for GQA).
    S_WINDOW_CTX = K_caches[0].shape[-2]   # SWA cache size (128 for gpt-oss)
    S_FULL       = K_caches[1].shape[-2]   # full cache size (e.g. 640 for bk640)

    mask_full_hbm = _build_full_mask_hbm(
        position_ids, S_FULL, B, num_heads_per_rank, S_tkg
    )
    mask_window_hbm = _build_swa_mask_hbm(
        position_ids, S_WINDOW_CTX, B, num_heads_per_rank, S_tkg, SLIDING_WIN
    )
    cos_perm_hbm = _build_permuted_rope_hbm(cos_unperm, name="cos_perm_hbm")
    sin_perm_hbm = _build_permuted_rope_hbm(sin_unperm, name="sin_perm_hbm")

    # Load X into SBUF residual in XSBLayout_tp102__0 (P-stride H1 in H).
    residual_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                             name="residual_sb")
    X_flat = X.reshape((BxS * H,))
    nisa.dma_copy(
        dst=residual_sb,
        src=X_flat.ap(pattern=[[H1, H0], [1, H1], [H, BxS]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )

    for layer_idx in range(num_layers):
        is_sliding = (layer_idx % 2 == 0)
        # gpt-oss alternates SWA / full attention. Shared RoPE across both.
        attn_mask = mask_window_hbm if is_sliding else mask_full_hbm
        kv_idx = position_ids_window if is_sliding else position_ids

        # ---- Attention ----
        sbm.set_name_prefix(f"L{layer_idx}_attn_")
        sbm.set_auto_alloc(True)

        # Copy residual — attention_block_tkg's fused RMSNorm writes back
        # in-place and would corrupt the residual otherwise.
        attn_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                 name=f"L{layer_idx}_attn_in_sb")
        nisa.tensor_copy(attn_in_sb, residual_sb)
        X_sb = attn_in_sb.reshape((H0, BxS, H1))

        attn_result = attention_block_tkg(
            X=X_sb,
            X_hidden_dim_actual=H_ACTUAL,
            rmsnorm_X_enabled=True,
            rmsnorm_X_eps=EPS,
            rmsnorm_X_gamma=gpre_list[layer_idx],
            W_qkv=Wqkv_list[layer_idx],
            bias_qkv=bqkv_list[layer_idx],
            quantization_type_qkv=QuantizationType.NONE,
            weight_dequant_scale_qkv=None,
            input_dequant_scale_qkv=None,
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_pre_rope_eps=EPS,
            rmsnorm_QK_pre_rope_W_Q=None,
            rmsnorm_QK_pre_rope_W_K=None,
            cos=cos_perm_hbm,
            sin=sin_perm_hbm,
            rope_contiguous_layout=True,
            rmsnorm_QK_post_rope_enabled=False,
            rmsnorm_QK_post_rope_eps=EPS,
            rmsnorm_QK_post_rope_W_Q=None,
            rmsnorm_QK_post_rope_W_K=None,
            K_cache_transposed=False,
            active_blocks_table=None,
            K_cache=K_post[layer_idx],
            V_cache=V_post[layer_idx],
            attention_mask=attn_mask,
            sink=sink_list[layer_idx],
            update_cache=True,
            kv_cache_update_idx=kv_idx,
            W_out=Wo_list[layer_idx],
            bias_out=bo_list[layer_idx],
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
        attn_sharded_sb = attn_kernel_out_sb.reshape((H0, H1_SHARD * BxS))

        attn_gathered_sb, _ = _sb2sb_all_reduce_gather(
            attn_sharded_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS
        )

        while sbm.heap:
            sbm.pop_heap()
        sbm.set_auto_alloc(True)

        nisa.tensor_tensor(dst=residual_sb, data1=residual_sb,
                           data2=attn_gathered_sb, op=nl.add)

        # ---- MoE ----
        sbm.set_name_prefix(f"L{layer_idx}_moe_")
        sbm.set_auto_alloc(True)

        moe_in_sb = nl.ndarray((H0, BxS, H1), dtype=dtype, buffer=nl.sbuf,
                               name=f"L{layer_idx}_moe_in_sb")
        rmsnorm_tkg(
            input=residual_sb.reshape((H0, BxS, H1)),
            gamma=gpost_list[layer_idx],
            output=moe_in_sb,
            eps=EPS,
            hidden_actual=H_ACTUAL,
            sbm=sbm,
        )

        # NCC_IGCA090: every mutable_tensor needs at least one store.
        router_logits_hbm = nl.ndarray((T, E), dtype=nl.float32,
                                       buffer=nl.shared_hbm,
                                       name=f"L{layer_idx}_router_logits_scratch")
        expert_index_sb = nl.ndarray((T, TOP_K), dtype=nl.uint32,
                                     buffer=nl.sbuf,
                                     name=f"L{layer_idx}_expert_index_sb")
        expert_affinities_sb = nl.ndarray((T, E), dtype=nl.float32,
                                          buffer=nl.sbuf,
                                          name=f"L{layer_idx}_expert_affinities_sb")
        # HF routing: topK(logits) → softmax(topK) → scatter, no renorm.
        # The (topK, ACT2, Scatter) path rebinds expert_affinities, so we
        # capture the returned value.
        router_outputs = router_topk(
            x=moe_in_sb,
            w=router_w_list[layer_idx],
            w_bias=router_b_list[layer_idx],
            router_logits=router_logits_hbm,
            expert_affinities=expert_affinities_sb,
            expert_index=expert_index_sb,
            act_fn=RouterActFnType.SOFTMAX,
            k=TOP_K,
            x_hbm_layout=XHBMLayout_T_H__1,
            x_sb_layout=XSBLayout_tp102__0,
            router_pre_norm=False,
            norm_topk_prob=False,
            skip_store_router_logits=False,
            name_prefix=f"L{layer_idx}_moe_",
        )
        expert_affinities_sb = router_outputs[2]

        # up clamp +1: ExpertMLPsV2.preshard_hook folds +1.0 into the
        # up half of expert_gate_up_bias.
        moe_out_sb = moe_tkg(
            hidden_input=moe_in_sb,
            expert_gate_up_weights=gate_up_list[layer_idx],
            expert_down_weights=down_list[layer_idx],
            expert_affinities=expert_affinities_sb,
            expert_index=expert_index_sb,
            is_all_expert=False,
            expert_gate_up_bias=gate_up_b_list[layer_idx],
            expert_down_bias=down_b_list[layer_idx],
            activation_fn=ActFnType.Swish,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            name_prefix=f"L{layer_idx}_moe_",
            gate_clamp_upper_limit=7.0,
            gate_clamp_lower_limit=None,
            up_clamp_upper_limit=8.0,
            up_clamp_lower_limit=-6.0,
            output_in_sbuf=True,
            output_dtype=dtype,
        )

        # moe_tkg already produces full H1 in tp102_0; bare AR matches residual.
        moe_reduced_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                    name=f"L{layer_idx}_moe_reduced_sb")
        nccl.all_reduce(
            dsts=[moe_reduced_sb],
            srcs=[moe_out_sb.reshape((H0, BxS * H1))],
            op=nl.add,
            replica_group=rg,
        )

        while sbm.heap:
            sbm.pop_heap()
        sbm.set_auto_alloc(True)

        nisa.tensor_tensor(dst=residual_sb, data1=residual_sb,
                           data2=moe_reduced_sb, op=nl.add)

    # Store residual to HBM. Final model.norm runs Python-side.
    Y = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="Y")
    if prg_id == 0:
        Y_flat = Y.reshape((BxS * H,))
        nisa.dma_copy(
            dst=Y_flat.ap(pattern=[[H1, H0], [1, H1], [H, BxS]], offset=0),
            src=residual_sb,
            dge_mode=nisa.dge_mode.hwdge,
        )
    if N_PRGS > 1:
        nisa.core_barrier(data=Y, cores=(0, 1))

    # Return KV refs so NCC preserves the in-place scatters.
    return (Y,) + tuple(K_post) + tuple(V_post)


def _build_multilayer_kernel(num_layers: int):
    """Code-gen a kernel with explicit per-layer tensor args.

    NKI classifies tuple/list args as scalars (no HBM binding), so each weight
    and KV cache must be its own top-level arg.
    """
    def names(prefix):
        return [f"{prefix}_{i:02d}" for i in range(num_layers)]

    wqkv_names    = names("Wqkv")
    bqkv_names    = names("bqkv")
    wo_names      = names("Wo")
    bo_names      = names("bo")
    sink_names    = names("Sink")
    gpre_names    = names("Gpre")
    gpost_names   = names("Gpost")
    rw_names      = names("RouterW")
    rb_names      = names("RouterB")
    gu_names      = names("GateUp")
    gub_names     = names("GateUpB")
    dn_names      = names("Down")
    dnb_names     = names("DownB")
    k_names       = names("K")
    v_names       = names("V")

    sig = ",\n    ".join(
        ["X"]
        + wqkv_names + bqkv_names
        + wo_names + bo_names
        + sink_names
        + gpre_names + gpost_names
        + rw_names + rb_names
        + gu_names + gub_names
        + dn_names + dnb_names
        + k_names + v_names
        + [
            "cos_unperm", "sin_unperm",
            "position_ids",
            "position_ids_window",
        ]
    )

    def tup(ns):
        return "(" + ", ".join(ns) + ",)"

    src = (
        f"def transformer_gpt_oss_tkg_multilayer(\n"
        f"    {sig},\n"
        f"    num_heads_per_rank=16,\n"
        f"    replica_groups=None,\n"
        f"):\n"
        f"    Wqkv_list      = {tup(wqkv_names)}\n"
        f"    bqkv_list      = {tup(bqkv_names)}\n"
        f"    Wo_list        = {tup(wo_names)}\n"
        f"    bo_list        = {tup(bo_names)}\n"
        f"    sink_list      = {tup(sink_names)}\n"
        f"    gpre_list      = {tup(gpre_names)}\n"
        f"    gpost_list     = {tup(gpost_names)}\n"
        f"    router_w_list  = {tup(rw_names)}\n"
        f"    router_b_list  = {tup(rb_names)}\n"
        f"    gate_up_list   = {tup(gu_names)}\n"
        f"    gate_up_b_list = {tup(gub_names)}\n"
        f"    down_list      = {tup(dn_names)}\n"
        f"    down_b_list    = {tup(dnb_names)}\n"
        f"    K_caches       = {tup(k_names)}\n"
        f"    V_caches       = {tup(v_names)}\n"
        f"    return _multilayer_body(\n"
        f"        X,\n"
        f"        Wqkv_list, bqkv_list,\n"
        f"        Wo_list, bo_list,\n"
        f"        sink_list,\n"
        f"        gpre_list, gpost_list,\n"
        f"        router_w_list, router_b_list,\n"
        f"        gate_up_list, gate_up_b_list,\n"
        f"        down_list, down_b_list,\n"
        f"        K_caches, V_caches,\n"
        f"        cos_unperm, sin_unperm,\n"
        f"        position_ids,\n"
        f"        position_ids_window,\n"
        f"        num_layers={num_layers},\n"
        f"        num_heads_per_rank=num_heads_per_rank,\n"
        f"        replica_groups=replica_groups,\n"
        f"    )\n"
    )

    fname = f"<generated:gpt_oss_multilayer_L{num_layers}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
    code = compile(src, fname, "exec")
    ns = {"_multilayer_body": _multilayer_body}
    exec(code, ns)
    return ns["transformer_gpt_oss_tkg_multilayer"]


transformer_gpt_oss_tkg_multilayer = _build_multilayer_kernel(NUM_LAYERS)
transformer_gpt_oss_tkg_multilayer_jit = nki.jit(transformer_gpt_oss_tkg_multilayer)

_kernel_cache: dict = {NUM_LAYERS: transformer_gpt_oss_tkg_multilayer_jit}


def get_multilayer_kernel_jit(num_layers: int):
    if num_layers not in _kernel_cache:
        _kernel_cache[num_layers] = nki.jit(_build_multilayer_kernel(num_layers))
    return _kernel_cache[num_layers]
