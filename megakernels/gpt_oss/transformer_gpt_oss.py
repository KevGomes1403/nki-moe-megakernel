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

SBM_SIZE_BYTES = 200 * 1024


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
    cos_window,          # [d_head//2, B, S_tkg]                bf16  HBM (RoPE cos, SWA)
    sin_window,          # [d_head//2, B, S_tkg]                bf16  HBM (RoPE sin, SWA)
    cos_global,          # [d_head//2, B, S_tkg]                bf16  HBM (RoPE cos, full)
    sin_global,          # [d_head//2, B, S_tkg]                bf16  HBM (RoPE sin, full)
    mask_window,         # [S_ctx_window, B, q_heads_attn, S]   bf16  HBM (SWA mask)
    mask_full,           # [S_ctx_full,   B, q_heads_attn, S]   bf16  HBM (full mask)
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
        cos = cos_window if is_sliding else cos_global
        sin = sin_window if is_sliding else sin_global
        attn_mask = mask_window if is_sliding else mask_full
        kv_idx = position_ids_window if is_sliding else position_ids

        # ---- Attention ----
        sbm.set_name_prefix(f"L{layer_idx}_attn_")
        sbm.set_auto_alloc(True)

        # Copy residual: attention_block_tkg's fused RMSNorm writes back in-place
        # and would corrupt the residual stream otherwise.
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
            cos=cos,
            sin=sin,
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
        # router_pre_norm=False matches HF: topK(logits) → softmax(topK) →
        # scatter. The (topK, ACT2, Scatter) path rebinds expert_affinities
        # (router_topk.py:975), so we must capture the returned value.
        # norm_topk_prob=False — HF does not renormalize.
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

        # up clamp shifted by +1 because ExpertMLPsV2.preshard_hook folds +1.0
        # into expert_gate_up_bias's up half (expert_mlps_v2.py:220-225).
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
            "cos_window", "sin_window",
            "cos_global", "sin_global",
            "mask_window", "mask_full",
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
        f"        cos_window, sin_window,\n"
        f"        cos_global, sin_global,\n"
        f"        mask_window, mask_full,\n"
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
