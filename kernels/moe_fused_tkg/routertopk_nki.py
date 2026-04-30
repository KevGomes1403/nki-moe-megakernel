import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.allocator import SbufManager

from .rmsnorm_nki import _rmsnorm_sbuf_in_sbuf_out_hoisted

_PMAX = 128
_H    = 2048
_E    = 128
_K    = 8
_H_FREE = _H // _PMAX          # 16
_N_PRGS = 2
_H_FREE_SHARD = _H_FREE // _N_PRGS  # 8
_H_SHARD = _H_FREE_SHARD * _PMAX    # 1024
_ROUTER_BATCH = 16


def _routertopk_sbuf_in_sbuf_out(
    rmsnorm_normed_bf16,    # [PMAX, H_free*T] bf16 heap in SBUF
    dtype,
    T,
    router_w,               # [H, E=128] bf16 HBM
    sbm=None,
    router_w_wide_sb=None,  # [PMAX, ROUTER_BATCH, E] bf16 SBUF pre-loaded
    debug_router_logits=None,  # [T, E] HBM, optional
):
    """
    Router matmul + Softmax + TopK(8) sub-kernel.
    rmsnorm_normed_bf16 must already be a live heap tensor in SBUF (from _rmsnorm_sbuf_in_sbuf_out_hoisted).

    Heap lifecycle (caller's responsibility):
      alloc_heap top8_logits_bf16 [T, K] bf16
      alloc_heap top8_idx         [T, K] uint32
      alloc_heap top8_vals_bf16   [T, K] bf16

    Returns (top8_logits_bf16, top8_idx, top8_vals_bf16).
    Caller pops in reverse order: top8_vals_bf16, top8_idx, top8_logits_bf16.
    rmsnorm_normed_bf16 is NOT popped here (expert stage still needs it).
    """
    H      = _H
    E      = _E
    K      = _K
    H_free = _H_FREE

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul [T, H] @ [H, E] → logits [T, E]
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((_PMAX, E), dtype=nl.float32, buffer=nl.psum)

    sbm.open_scope(name="router_softmax")

    # --- router wide DMA (skipped when router_w_wide_sb is supplied) ---
    if router_w_wide_sb is None:
        _router_w_wide_sb = sbm.alloc_stack((_PMAX, _ROUTER_BATCH, E), dtype, buffer=nl.sbuf, name="router_w_wide_sb")
    else:
        _router_w_wide_sb = router_w_wide_sb  # caller-owned — do NOT free

    for h_chunk in nl.affine_range(H_free // _ROUTER_BATCH):
        if router_w_wide_sb is None:
            for h_sub_load in nl.static_range(_ROUTER_BATCH):
                h1_load = h_chunk * _ROUTER_BATCH + h_sub_load
                shard = h1_load // _H_FREE_SHARD
                h2 = h1_load - shard * _H_FREE_SHARD
                nisa.dma_copy(
                    dst=_router_w_wide_sb[0:_PMAX, h_sub_load, 0:E],
                    src=router_w.ap(
                        pattern=[[_H_FREE_SHARD * E, _PMAX], [1, E]],
                        offset=shard * _H_SHARD * E + h2 * E,
                    ),
                    dge_mode=3,
                )
        for h_sub in nl.static_range(_ROUTER_BATCH):
            h1 = h_chunk * _ROUTER_BATCH + h_sub
            column_tile = h1 % 4
            column_offset = column_tile * 32
            nisa.nc_matmul(
                dst=logits_psum[nl.ds(column_offset, T), 0:E],
                stationary=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                moving=_router_w_wide_sb[0:_PMAX, h_sub, 0:E],
                tile_position=(0, column_offset),
                tile_size=(_PMAX, 32),
            )

    logits_sb = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="logits_sb")
    nisa.tensor_copy(dst=logits_sb[0:T, 0:E], src=logits_psum[0:T, 0:E])
    for column_tile in nl.static_range(1, 4):
        column_offset = column_tile * 32
        nisa.tensor_tensor(
            dst=logits_sb[0:T, 0:E],
            data1=logits_sb[0:T, 0:E],
            data2=logits_psum[nl.ds(column_offset, T), 0:E],
            op=nl.add,
        )
    if debug_router_logits is not None:
        nisa.dma_copy(dst=debug_router_logits[0:T, 0:E], src=logits_sb[0:T, 0:E])

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8)
    # -----------------------------------------------------------------------
    max_logit = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="max_logit")
    nisa.tensor_reduce(
        max_logit[0:T, 0:1],
        nl.maximum,
        logits_sb[0:T, 0:E],
        axis=1,
        negate=True,
        keepdims=True,
    )

    exp_vals = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="exp_vals")
    sum_exp = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="sum_exp")
    nisa.activation(
        dst=exp_vals[0:T, 0:E],
        op=nl.exp,
        data=logits_sb[0:T, 0:E],
        bias=max_logit[0:T, 0:1],
        reduce_op=nl.add,
        reduce_res=sum_exp[0:T, 0:1],
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )

    inv_sum_exp = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="inv_sum_exp")
    nisa.reciprocal(dst=inv_sum_exp[0:T, 0:1], data=sum_exp[0:T, 0:1])

    softmax_probs = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="softmax_probs")
    nisa.tensor_scalar(
        softmax_probs[0:T, 0:E], data=exp_vals[0:T, 0:E],
        op0=nl.multiply, operand0=inv_sum_exp[0:T, 0:1],
    )

    top8_logits = sbm.alloc_heap((T, K), nl.float32, buffer=nl.sbuf, name="top8_logits")
    nisa.max8(dst=top8_logits[0:T, 0:K], src=softmax_probs[0:T, 0:E])

    top8_idx = sbm.alloc_heap((T, K), nl.uint32, buffer=nl.sbuf, name="top8_idx")
    nisa.nc_find_index8(dst=top8_idx[0:T, 0:K], data=softmax_probs[0:T, 0:E], vals=top8_logits[0:T, 0:K])

    sum_topk = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="sum_topk")
    nisa.tensor_reduce(sum_topk[0:T, 0:1], nl.add, top8_logits[0:T, 0:K], axis=1, keepdims=True)
    inv_sum_topk = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="inv_sum_topk")
    nisa.reciprocal(dst=inv_sum_topk[0:T, 0:1], data=sum_topk[0:T, 0:1])

    top8_vals = sbm.alloc_heap((T, K), nl.float32, buffer=nl.sbuf, name="top8_vals")
    nisa.tensor_scalar(
        dst=top8_vals[0:T, 0:K],
        data=top8_logits[0:T, 0:K],
        op0=nl.multiply,
        operand0=inv_sum_topk[0:T, 0:1],
    )

    sbm.close_scope()  # frees router/softmax stack tensors (not router_w_wide_sb — caller-owned)

    return top8_logits, top8_idx, top8_vals


@nki.jit
def rmsnorm_routertopk_hbm(inp, gamma, router_w):
    """
    Debug wrapper for comparing local RMSNorm+RouterTopK against the production
    initialize_moe_module(..., init_tkg_module=True) fused path.

    inp:      [T, H] pre-RMSNorm hidden states
    gamma:    [1, H] RMSNorm weight
    router_w: [H, E] transposed router weight
    Returns (top8_idx, normalized_top8_vals, router_logits).
    """
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    T = inp.shape[0]
    H_free = _H_FREE

    sbm.open_scope(name="inp_load")
    inp_2d = inp.reshape((T, _H))
    inp_sb = sbm.alloc_stack((_PMAX, H_free * T), inp.dtype, buffer=nl.sbuf, name="inp_sb")
    for t in nl.static_range(T):
        for h1 in nl.static_range(_H_FREE):
            shard = h1 // _H_FREE_SHARD
            h2 = h1 - shard * _H_FREE_SHARD
            nisa.dma_copy(
                dst=inp_sb[0:_PMAX, h1 * T + t:h1 * T + t + 1],
                src=inp_2d.ap(
                    pattern=[[_H_FREE_SHARD, _PMAX], [1, 1]],
                    offset=t * _H + shard * _H_SHARD + h2,
                ),
                dge_mode=3,
            )

    rmsnorm_normed_bf16 = _rmsnorm_sbuf_in_sbuf_out_hoisted(
        inp_sb, inp.dtype, T, gamma, sbm=sbm,
    )

    router_in = sbm.alloc_stack((_PMAX, H_free * T), router_w.dtype, buffer=nl.sbuf, name="router_in")
    nisa.tensor_copy(dst=router_in[0:_PMAX, 0:H_free * T], src=rmsnorm_normed_bf16[0:_PMAX, 0:H_free * T])

    router_logits = nl.ndarray((T, _E), dtype=inp.dtype, buffer=nl.shared_hbm)
    top8_logits, top8_idx, top8_vals = _routertopk_sbuf_in_sbuf_out(
        router_in, router_w.dtype, T, router_w, sbm=sbm, debug_router_logits=router_logits
    )

    out_idx = nl.ndarray((T, _K), dtype=nl.uint32, buffer=nl.shared_hbm)
    out_vals = nl.ndarray((T, _K), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out_idx[0:T, 0:_K], src=top8_idx[0:T, 0:_K])
    nisa.dma_copy(dst=out_vals[0:T, 0:_K], src=top8_vals[0:T, 0:_K])

    sbm.pop_heap()  # top8_vals
    sbm.pop_heap()  # top8_idx
    sbm.pop_heap()  # top8_logits
    sbm.pop_heap()  # rmsnorm_normed_bf16
    sbm.close_scope()
    return out_idx, out_vals, router_logits


@nki.jit
def routertopk_hbm(inp_normed, router_w, hoisted_router_w=False):
    """
    HBM wrapper for Router + TopK.

    inp_normed:  [T, H=2048] bf16 HBM — RMSNorm-normalized input
    router_w:    [H=2048, E=128] bf16 HBM
    Returns (top8_idx [T, K=8] uint32, top8_vals [T, K=8] bf16) in HBM.
    """
    sbm    = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    T      = inp_normed.shape[0]
    H_free = _H_FREE

    # --- Load inp_normed into SBUF as [PMAX, H_free*T] and cast to router_w dtype.
    sbm.open_scope(name="inp_load")
    inp_2d = inp_normed.reshape((T, _H))
    router_in = sbm.alloc_heap((_PMAX, H_free * T), router_w.dtype, buffer=nl.sbuf, name="router_in")
    for t in nl.static_range(T):
        for h1 in nl.static_range(_H_FREE):
            shard = h1 // _H_FREE_SHARD
            h2 = h1 - shard * _H_FREE_SHARD
            nisa.dma_copy(
                dst=router_in[0:_PMAX, h1 * T + t:h1 * T + t + 1],
                src=inp_2d.ap(
                    pattern=[[_H_FREE_SHARD, _PMAX], [1, 1]],
                    offset=t * _H + shard * _H_SHARD + h2,
                ),
                dge_mode=3,
            )

    # --- Optionally pre-load router_w into SBUF ---
    router_w_wide_sb = None
    if hoisted_router_w:
        sbm.open_scope(name="router_w_hoist")
        router_w_wide_sb = sbm.alloc_stack((_PMAX, _ROUTER_BATCH, _E), router_w.dtype, buffer=nl.sbuf, name="router_w_wide_sb_hoist")
        for h_sub_load in nl.static_range(_ROUTER_BATCH):
            shard = h_sub_load // _H_FREE_SHARD
            h2 = h_sub_load - shard * _H_FREE_SHARD
            nisa.dma_copy(
                dst=router_w_wide_sb[0:_PMAX, h_sub_load, 0:_E],
                src=router_w.ap(
                    pattern=[[_H_FREE_SHARD * _E, _PMAX], [1, _E]],
                    offset=shard * _H_SHARD * _E + h2 * _E,
                ),
                dge_mode=3,
            )

    top8_logits_bf16, top8_idx, top8_vals = _routertopk_sbuf_in_sbuf_out(
        router_in, router_w.dtype, T, router_w, sbm=sbm,
        router_w_wide_sb=router_w_wide_sb,
    )

    # --- Store outputs to HBM ---
    out_idx  = nl.ndarray((T, _K), dtype=nl.uint32, buffer=nl.shared_hbm)
    out_vals = nl.ndarray((T, _K), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out_idx[0:T, 0:_K],  src=top8_idx[0:T, 0:_K])
    nisa.dma_copy(dst=out_vals[0:T, 0:_K], src=top8_vals[0:T, 0:_K])

    sbm.pop_heap()  # top8_vals
    sbm.pop_heap()  # top8_idx
    sbm.pop_heap()  # top8_logits_bf16
    sbm.pop_heap()  # router_in
    if hoisted_router_w:
        sbm.close_scope()  # router_w_hoist
    sbm.close_scope()  # inp_load

    return out_idx, out_vals
