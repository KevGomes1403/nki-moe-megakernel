"""
Debug variant of the multi-layer fused TKG megakernel (transformer_qwen_multilayer.py).

Identical compute to the production kernel, but additionally captures two layer-0
intermediates to shared_hbm and returns them so the caller can wire them into NxDI's
_register_tensor mechanism for on-chip tensor capture:

  - attn_out_l0:            layer-0 attention output (after TP AllReduce, before residual add)
  - final_hidden_states_l0: layer-0 residual after MoE (full decoder-layer output for layer 0)

Return layout vs. the non-debug kernel:
  kernel_out[0]          = Y                          (final hidden state, all layers)
  kernel_out[1]          = attn_out_l0
  kernel_out[2]          = final_hidden_states_l0
  kernel_out[3 : 3+L]    = K_caches  (per-layer K, in-place updated)
  kernel_out[3+L : 3+2L] = V_caches  (per-layer V, in-place updated)

kv_k / kv_v for layer 0 are kernel_out[3] / kernel_out[3+L] — already present in the
normal return path; the model-side caller passes them to _register_tensor directly.
"""

import linecache

import nki
import nki.isa as nisa
import nki.language as nl
import nki.collectives as nccl
from nkilib.core.utils.allocator import BufferManager, Logger
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.experimental.transformer.transformer_tkg import _sb2sb_all_reduce_gather
from kernels.attn_tkg.agents.v14d_kv_norm_hoisted_weights import (
    qwen3_attn_tkg_fused_oproj_v13bc_kv_norm_hoisted_weights,
)
from kernels.moe_fused_tkg.versions.kernel_v30c_hoisted import (
    _qwen3_moe_sbuf_in_sbuf_out_hoisted,
)
from kernels.transformer.transformer_qwen_multilayer import (
    H, H0, H1, N_PRGS, H1_SHARD, PMAX, NH, SBM_SIZE_BYTES, NUM_LAYERS,
)


def _store_sbuf_to_hbm(src_sbuf, dst_hbm, prg_id, BxS, dtype, prefix=""):
    """Transpose-store [H0, BxS*H1] SBUF tensor to [BxS, H] HBM (reshape of dst_hbm).

    Mirrors the Y-store in _multilayer_body_debug.  Only prg_id==0 writes;
    prg_id==1 is a no-op (both cores fence on the core_barrier at the end).
    """
    src_copy = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                           name=f"{prefix}src_copy")
    nisa.tensor_copy(src_copy, src_sbuf)
    if prg_id == 0:
        dst_flat = dst_hbm.reshape((BxS, H))
        for b in nl.static_range(BxS):
            for t in nl.static_range(H1):
                col_psum = nl.ndarray((1, H0), dtype=dtype, buffer=nl.psum)
                nisa.nc_transpose(
                    col_psum,
                    src_copy[0:H0, b * H1 + t : b * H1 + t + 1],
                )
                col_sb = nl.ndarray((1, H0), dtype=dtype, buffer=nl.sbuf,
                                    name=f"{prefix}col_b{b}_t{t}")
                nisa.tensor_copy(col_sb, col_psum)
                nisa.dma_copy(
                    dst=dst_flat[b : b + 1, t * H0 : (t + 1) * H0],
                    src=col_sb,
                    dge_mode=nisa.dge_mode.hwdge,
                )


def _multilayer_body_debug(
    X,
    Wq_list, Wk_list, Wv_list, Wo_list,
    qn_list, kn_list, gpre_list, gpost_list,
    router_list, gate_up_list, down_list,
    K_caches, V_caches,
    cos, sin, position_ids,
    num_layers,
    replica_groups=None,
):
    """Debug kernel body — same compute as _multilayer_body, plus layer-0 HBM stores."""
    B, S_tkg, _ = X.shape
    dtype = X.dtype
    BxS = B * S_tkg
    T = BxS

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "transformer_qwen3_moe_tkg_multilayer_debug", (0, 1), N_PRGS
    )

    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("transformer_qwen3_moe_tkg_multilayer_debug"))
    sbm.set_auto_alloc(True)

    # -----------------------------------------------------------------------
    # Load X into SBUF as the initial residual [H0, BxS*H1].
    # -----------------------------------------------------------------------
    residual_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                              name="residual_sb")
    X_flat = X.reshape((BxS * H,))
    residual_load_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                   name="residual_load_sb")
    nisa.dma_copy(
        dst=residual_load_sb,
        src=X_flat.ap(pattern=[[1, H0], [H0, H1], [H, BxS]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )
    nisa.tensor_copy(residual_sb, residual_load_sb)

    # -----------------------------------------------------------------------
    # DEBUG: allocate extra shared_hbm outputs for layer-0 intermediates.
    # -----------------------------------------------------------------------
    attn_out_l0 = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm,
                              name="attn_out_l0")
    final_hidden_states_l0 = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm,
                                         name="final_hidden_states_l0")

    # -----------------------------------------------------------------------
    # Hoist layer-invariant attention constants out of the per-layer loop.
    # -----------------------------------------------------------------------
    sbm.set_name_prefix("hoist_")
    sbm.open_scope(name="hoist")
    sbm.set_auto_alloc(True)

    cos_col = cos.reshape((PMAX, B))
    sin_col = sin.reshape((PMAX, B))

    cos_bf16_all = sbm.alloc_stack((PMAX, B), dtype, name="cos_bf16_all")
    sin_bf16_all = sbm.alloc_stack((PMAX, B), dtype, name="sin_bf16_all")
    nisa.dma_copy(dst=cos_bf16_all, src=cos_col, dge_mode=nisa.dge_mode.hwdge)
    nisa.dma_copy(dst=sin_bf16_all, src=sin_col, dge_mode=nisa.dge_mode.hwdge)
    cos_f32_all = sbm.alloc_stack((PMAX, B), nl.float32, name="cos_f32_all")
    sin_f32_all = sbm.alloc_stack((PMAX, B), nl.float32, name="sin_f32_all")
    nisa.tensor_copy(cos_f32_all, cos_bf16_all)
    nisa.tensor_copy(sin_f32_all, sin_bf16_all)

    qnw_bf16_all  = sbm.alloc_stack((PMAX, num_layers),      dtype, name="qnw_bf16_all")
    knw_bf16_all  = sbm.alloc_stack((PMAX, num_layers),      dtype, name="knw_bf16_all")
    gpan_bf16_all = sbm.alloc_stack((PMAX, num_layers * NH), dtype, name="gpan_bf16_all")

    for li in range(num_layers):
        nisa.dma_copy(
            dst=qnw_bf16_all[0:PMAX, li:li + 1],
            src=qn_list[li].reshape((PMAX, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.dma_copy(
            dst=knw_bf16_all[0:PMAX, li:li + 1],
            src=kn_list[li].reshape((PMAX, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.dma_copy(
            dst=gpan_bf16_all[0:PMAX, li * NH:(li + 1) * NH],
            src=gpre_list[li].reshape((H, 1)).ap(
                pattern=[[1, PMAX], [PMAX, NH]], offset=0,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )

    qnw_f32_all  = sbm.alloc_stack((PMAX, num_layers),      nl.float32, name="qnw_f32_all")
    knw_f32_all  = sbm.alloc_stack((PMAX, num_layers),      nl.float32, name="knw_f32_all")
    gpan_f32_all = sbm.alloc_stack((PMAX, num_layers * NH), nl.float32, name="gpan_f32_all")
    nisa.tensor_copy(qnw_f32_all,  qnw_bf16_all)
    nisa.tensor_copy(knw_f32_all,  knw_bf16_all)
    nisa.tensor_copy(gpan_f32_all, gpan_bf16_all)

    # -----------------------------------------------------------------------
    # Double-buffered cross-layer weight prefetch (interleave_degree=1).
    # -----------------------------------------------------------------------
    if prg_id == 0:
        OWNED = [0, 1, 2, 3]
    else:
        OWNED = [4, 5, 6, 7]
    NOH = 4
    HQ_TP = 8
    ROUTER_BATCH = 16
    E = 128

    sbm.set_auto_alloc(True)
    sbm.set_name_prefix("weights_")
    sbm.open_scope(interleave_degree=1, name="weight_db")

    for layer_idx in range(num_layers):
        Wk_cur = sbm.alloc_stack((PMAX, NH * PMAX), dtype, name=f"Wk_L{layer_idx}")
        Wv_cur = sbm.alloc_stack((PMAX, NH * PMAX), dtype, name=f"Wv_L{layer_idx}")
        Wq_cur = []
        Wo_cur = []
        for hi in range(NOH):
            Wq_cur.append(sbm.alloc_stack(
                (PMAX, NH * PMAX), dtype, name=f"Wq_L{layer_idx}_h{hi}"
            ))
            Wo_cur.append(sbm.alloc_stack(
                (PMAX, H), dtype, name=f"Wo_L{layer_idx}_h{hi}"
            ))
        Wq_cur = tuple(Wq_cur)
        Wo_cur = tuple(Wo_cur)
        Router_cur = sbm.alloc_stack(
            (PMAX, ROUTER_BATCH, E), nl.float32, name=f"Router_L{layer_idx}"
        )

        nisa.dma_copy(dst=Wk_cur, src=Wk_list[layer_idx], dge_mode=nisa.dge_mode.hwdge)
        nisa.dma_copy(dst=Wv_cur, src=Wv_list[layer_idx], dge_mode=nisa.dge_mode.hwdge)
        for hi in range(NOH):
            q_h = OWNED[hi]
            nisa.dma_copy(
                dst=Wq_cur[hi],
                src=Wq_list[layer_idx][q_h * PMAX:(q_h + 1) * PMAX, :],
                dge_mode=nisa.dge_mode.hwdge,
            )
            nisa.dma_copy(
                dst=Wo_cur[hi],
                src=Wo_list[layer_idx].reshape((HQ_TP, PMAX, H)).ap(
                    pattern=[[H, PMAX], [1, H]],
                    offset=q_h * PMAX * H,
                ),
                dge_mode=nisa.dge_mode.hwdge,
            )
        nisa.dma_copy(
            dst=Router_cur,
            src=router_list[layer_idx].ap(
                pattern=[[E, PMAX], [PMAX * E, ROUTER_BATCH], [1, E]],
                offset=0,
            ),
            dge_mode=3,
        )

        # ---- ATTENTION ----
        sbm.set_name_prefix(f"L{layer_idx}_attn_")
        sbm.set_auto_alloc(True)

        hidden_sb_bf16 = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                     name=f"L{layer_idx}_hidden_bf16")
        nisa.tensor_copy(hidden_sb_bf16, residual_sb)

        out_sb = qwen3_attn_tkg_fused_oproj_v13bc_kv_norm_hoisted_weights(
            hidden_sb=hidden_sb_bf16,
            Wq=Wq_list[layer_idx],
            Wk=Wk_list[layer_idx],
            Wv=Wv_list[layer_idx],
            Wo=Wo_list[layer_idx],
            q_norm_weight=qn_list[layer_idx],
            k_norm_weight=kn_list[layer_idx],
            gamma_pre_attn=gpre_list[layer_idx],
            K_cache=K_caches[layer_idx],
            V_cache=V_caches[layer_idx],
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            sbm=sbm,
            qnw_f32_sb  = qnw_f32_all[0:PMAX, layer_idx:layer_idx + 1],
            knw_f32_sb  = knw_f32_all[0:PMAX, layer_idx:layer_idx + 1],
            cos_f32_sb  = cos_f32_all,
            sin_f32_sb  = sin_f32_all,
            gpan_f32_sb = gpan_f32_all[0:PMAX, layer_idx * NH:(layer_idx + 1) * NH],
            wk_sb = Wk_cur,
            wv_sb = Wv_cur,
            wq_heads_sb = Wq_cur,
            wo_heads_sb = Wo_cur,
        )

        attn_reduced_sb = nl.zeros((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                    name=f"L{layer_idx}_attn_reduced_sb")
        nccl.all_reduce(
            dsts=[attn_reduced_sb], srcs=[out_sb],
            op=nl.add, replica_group=rg,
        )

        sbm.close_scope()
        sbm.set_auto_alloc(False)

        # Residual add #1
        nisa.tensor_tensor(dst=residual_sb, data1=residual_sb,
                           data2=attn_reduced_sb, op=nl.add)

        # DEBUG: store layer-0 attn output (after TP AR, before residual add #1).
        # attn_reduced_sb is nl.zeros-allocated (not sbm-managed) so it is
        # still valid here.
        if layer_idx == 0:
            _store_sbuf_to_hbm(attn_reduced_sb, attn_out_l0, prg_id, BxS, dtype,
                                prefix="dbg_attn_")

        # ---- MOE ----
        sbm.set_name_prefix(f"L{layer_idx}_moe_")
        sbm.set_auto_alloc(True)

        moe_inp_bf16 = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                   name=f"L{layer_idx}_moe_inp_bf16")
        nisa.tensor_copy(moe_inp_bf16, residual_sb)

        moe_out_sb = _qwen3_moe_sbuf_in_sbuf_out_hoisted(
            inp_sb=moe_inp_bf16,
            dtype=dtype,
            T=T,
            gamma=gpost_list[layer_idx],
            router_w=router_list[layer_idx],
            gate_up_w=gate_up_list[layer_idx],
            down_w=down_list[layer_idx],
            sbm=sbm,
            router_w_wide_sb=Router_cur,
        )

        moe_gathered_sb, _ = _sb2sb_all_reduce_gather(
            moe_out_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS
        )

        while sbm.heap:
            sbm.pop_heap()
        sbm.set_auto_alloc(False)

        # Residual add #2
        nisa.tensor_tensor(dst=residual_sb, data1=residual_sb,
                           data2=moe_gathered_sb, op=nl.add)

        # DEBUG: store layer-0 final hidden state (after MoE residual add).
        if layer_idx == 0:
            _store_sbuf_to_hbm(residual_sb, final_hidden_states_l0, prg_id, BxS, dtype,
                                prefix="dbg_fhs_")

        sbm.increment_section()

    # Close weight double-buffer + hoisted-constants scopes.
    sbm.close_scope()  # weight_db
    sbm.close_scope()  # hoist

    # -----------------------------------------------------------------------
    # After the last layer: single HBM store of the post-residual hidden state.
    # -----------------------------------------------------------------------
    Y = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="Y")
    residual_out_bf16_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                       name="residual_out_bf16_sb")
    nisa.tensor_copy(residual_out_bf16_sb, residual_sb)
    if prg_id == 0:
        Y_flat = Y.reshape((BxS, H))
        for b in nl.static_range(BxS):
            for t in nl.static_range(H1):
                col_psum = nl.ndarray((1, H0), dtype=dtype, buffer=nl.psum)
                nisa.nc_transpose(col_psum, residual_out_bf16_sb[0:H0, b*H1 + t : b*H1 + t + 1])
                col_sb = nl.ndarray((1, H0), dtype=dtype, buffer=nl.sbuf,
                                    name=f"out_col_b{b}_t{t}_sb")
                nisa.tensor_copy(col_sb, col_psum)
                nisa.dma_copy(
                    dst=Y_flat[b:b+1, t*H0:(t+1)*H0],
                    src=col_sb,
                    dge_mode=nisa.dge_mode.hwdge,
                )
    if n_prgs > 1:
        nisa.core_barrier(data=attn_out_l0, cores=(0, 1))
        nisa.core_barrier(data=final_hidden_states_l0, cores=(0, 1))
        nisa.core_barrier(data=Y, cores=(0, 1))

    return (Y, attn_out_l0, final_hidden_states_l0) + tuple(K_caches) + tuple(V_caches)


def _build_multilayer_kernel_debug(num_layers: int):
    """Code-gen the debug kernel with explicit per-layer KV tensor args.

    Identical signature to _build_multilayer_kernel; body calls
    _multilayer_body_debug so the extra debug outputs are returned.
    """
    def names(prefix):
        return [f"{prefix}_{i:02d}" for i in range(num_layers)]

    wq_names     = names("Wq")
    wk_names     = names("Wk")
    wv_names     = names("Wv")
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
        + wq_names + wk_names + wv_names + wo_names
        + qn_names + kn_names + gpre_names + gpost_names
        + router_names + gu_names + down_names
        + k_names + v_names
        + ["cos", "sin", "position_ids"]
    )

    def tup(ns):
        return "(" + ", ".join(ns) + ",)"

    src = (
        f"def transformer_qwen3_moe_tkg_multilayer_debug(\n"
        f"    {sig},\n"
        f"    replica_groups=None,\n"
        f"):\n"
        f"    Wq_list      = {tup(wq_names)}\n"
        f"    Wk_list      = {tup(wk_names)}\n"
        f"    Wv_list      = {tup(wv_names)}\n"
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
        f"    return _multilayer_body_debug(\n"
        f"        X, Wq_list, Wk_list, Wv_list, Wo_list,\n"
        f"        qn_list, kn_list, gpre_list, gpost_list,\n"
        f"        router_list, gate_up_list, down_list,\n"
        f"        K_caches, V_caches,\n"
        f"        cos, sin, position_ids,\n"
        f"        num_layers={num_layers},\n"
        f"        replica_groups=replica_groups,\n"
        f"    )\n"
    )

    fname = f"<generated:multilayer_debug_L{num_layers}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
    code = compile(src, fname, "exec")
    ns = {"_multilayer_body_debug": _multilayer_body_debug}
    exec(code, ns)
    return ns["transformer_qwen3_moe_tkg_multilayer_debug"]


_debug_kernel_cache: dict = {}


def get_multilayer_debug_kernel_jit(num_layers: int):
    if num_layers not in _debug_kernel_cache:
        _debug_kernel_cache[num_layers] = nki.jit(_build_multilayer_kernel_debug(num_layers))
    return _debug_kernel_cache[num_layers]
