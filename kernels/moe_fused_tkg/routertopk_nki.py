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
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)

    sbm.open_scope(name="router_softmax")

    # --- router wide DMA (skipped when router_w_wide_sb is supplied) ---
    if router_w_wide_sb is None:
        _router_w_wide_sb = sbm.alloc_stack((_PMAX, _ROUTER_BATCH, E), dtype, buffer=nl.sbuf, name="router_w_wide_sb")
    else:
        _router_w_wide_sb = router_w_wide_sb  # caller-owned — do NOT free

    for h_chunk in nl.affine_range(H_free // _ROUTER_BATCH):
        if router_w_wide_sb is None:
            nisa.dma_copy(
                dst=_router_w_wide_sb,
                src=router_w.ap(
                    pattern=[[E, _PMAX], [_PMAX * E, _ROUTER_BATCH], [1, E]],
                    offset=h_chunk * _ROUTER_BATCH * _PMAX * E,
                ),
                dge_mode=3,
            )
        for h_sub in nl.static_range(_ROUTER_BATCH):
            h1 = h_chunk * _ROUTER_BATCH + h_sub
            nisa.nc_matmul(
                dst=logits_psum[0:T, 0:E],
                stationary=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                moving=_router_w_wide_sb[0:_PMAX, h_sub, 0:E],
            )

    # Reference path: round PSUM to bf16 per-logit, then upcast to fp32 for softmax chain.
    logits_bf16 = sbm.alloc_stack((T, E), dtype, buffer=nl.sbuf, name="logits_bf16")
    nisa.activation(logits_bf16[0:T, 0:E], op=nl.copy, data=logits_psum[0:T, 0:E])
    logits_sb = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="logits_sb")
    nisa.activation(logits_sb[0:T, 0:E], op=nl.copy, data=logits_bf16[0:T, 0:E])

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8)
    # -----------------------------------------------------------------------
    max_logit = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="max_logit")
    nisa.tensor_reduce(max_logit[0:T, 0:1], nl.maximum, logits_sb[0:T, 0:E], axis=1)

    centered = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="centered")
    nisa.tensor_scalar(
        centered[0:T, 0:E], data=logits_sb[0:T, 0:E],
        op0=nl.subtract, operand0=max_logit[0:T, 0:1],
    )

    exp_vals = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="exp_vals")
    nisa.activation(exp_vals[0:T, 0:E], op=nl.exp, data=centered[0:T, 0:E])

    sum_exp = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="sum_exp")
    nisa.tensor_reduce(sum_exp[0:T, 0:1], nl.add, exp_vals[0:T, 0:E], axis=1)

    inv_sum_exp = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="inv_sum_exp")
    nisa.activation(inv_sum_exp[0:T, 0:1], op=nl.reciprocal, data=sum_exp[0:T, 0:1])

    softmax_probs = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="softmax_probs")
    nisa.tensor_scalar(
        softmax_probs[0:T, 0:E], data=exp_vals[0:T, 0:E],
        op0=nl.multiply, operand0=inv_sum_exp[0:T, 0:1],
    )

    # NxDI rounds the full fp32 softmax result to bf16 before gathering the
    # selected affinities. Recomputing only selected exp/logit values is close
    # but not the same op sequence.
    softmax_probs_bf16 = sbm.alloc_stack((T, E), dtype, buffer=nl.sbuf, name="softmax_probs_bf16")
    nisa.activation(softmax_probs_bf16[0:T, 0:E], op=nl.copy, data=softmax_probs[0:T, 0:E])

    # NxDI takes top-K on bf16 router logits before the fp32 softmax chain.
    top8_logits_bf16 = sbm.alloc_heap((T, K), dtype, buffer=nl.sbuf, name="top8_logits_bf16")
    nisa.max8(dst=top8_logits_bf16[0:T, 0:K], src=logits_bf16[0:T, 0:E])

    top8_idx = sbm.alloc_heap((T, K), nl.uint32, buffer=nl.sbuf, name="top8_idx")
    nisa.nc_find_index8(dst=top8_idx[0:T, 0:K], data=logits_bf16[0:T, 0:E], vals=top8_logits_bf16[0:T, 0:K])

    top8_vals_bf16 = sbm.alloc_heap((T, K), dtype, buffer=nl.sbuf, name="top8_vals_bf16")
    nisa.nc_n_gather(
        dst=top8_vals_bf16[0:T, 0:K],
        data=softmax_probs_bf16[0:T, 0:E],
        indices=top8_idx[0:T, 0:K],
    )

    sbm.close_scope()  # frees router/softmax stack tensors (not router_w_wide_sb — caller-owned)

    return top8_logits_bf16, top8_idx, top8_vals_bf16


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

    # --- Load inp_normed into SBUF as [PMAX, H_free*T] col-major ---
    inp_2d          = inp_normed.reshape((T, _H))
    inp_2d_hbm_flat = inp_2d.reshape((H_free * T, _PMAX))
    sbm.open_scope(name="inp_load")
    inp_flat_sb = sbm.alloc_stack((H_free * T, _PMAX), inp_normed.dtype, buffer=nl.sbuf, name="inp_flat_sb")
    nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_flat, dge_mode=3)
    inp_trans_psum = nl.ndarray((_PMAX, H_free * T), dtype=inp_normed.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)
    rmsnorm_normed_bf16 = sbm.alloc_heap((_PMAX, H_free * T), inp_normed.dtype, buffer=nl.sbuf, name="rmsnorm_normed_bf16")
    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=inp_trans_psum[...])

    # --- Optionally pre-load router_w into SBUF ---
    router_w_wide_sb = None
    if hoisted_router_w:
        sbm.open_scope(name="router_w_hoist")
        router_w_wide_sb = sbm.alloc_stack((_PMAX, _ROUTER_BATCH, _E), router_w.dtype, buffer=nl.sbuf, name="router_w_wide_sb_hoist")
        nisa.dma_copy(
            dst=router_w_wide_sb,
            src=router_w.ap(
                pattern=[[_E, _PMAX], [_PMAX * _E, _ROUTER_BATCH], [1, _E]],
                offset=0,
            ),
            dge_mode=3,
        )

    top8_logits_bf16, top8_idx, top8_vals_bf16 = _routertopk_sbuf_in_sbuf_out(
        rmsnorm_normed_bf16, inp_normed.dtype, T, router_w, sbm=sbm,
        router_w_wide_sb=router_w_wide_sb,
    )

    # --- Store outputs to HBM ---
    out_idx  = nl.ndarray((T, _K), dtype=nl.uint32, buffer=nl.shared_hbm)
    out_vals = nl.ndarray((T, _K), dtype=inp_normed.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out_idx[0:T, 0:_K],  src=top8_idx[0:T, 0:_K])
    nisa.dma_copy(dst=out_vals[0:T, 0:_K], src=top8_vals_bf16[0:T, 0:_K])

    # Pop heaps in reverse allocation order
    sbm.pop_heap()  # top8_vals_bf16
    sbm.pop_heap()  # top8_idx
    sbm.pop_heap()  # top8_logits_bf16
    sbm.pop_heap()  # rmsnorm_normed_bf16 (loaded by this wrapper)

    if hoisted_router_w:
        sbm.close_scope()  # router_w_hoist
    sbm.close_scope()  # inp_load

    return out_idx, out_vals
