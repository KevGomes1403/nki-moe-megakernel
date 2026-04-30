import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.tensor_view import TensorView
from nkilib.core.utils.allocator import SbufManager

_PMAX = 128
_H    = 2048
_H_FREE = _H // _PMAX            # 16
_N_PRGS = 2
_H_FREE_SHARD = _H_FREE // _N_PRGS  # 8
_H_SHARD = _H_FREE_SHARD * _PMAX    # 1024
_EPS  = 1e-6


def _rmsnorm_sbuf_in_sbuf_out_hoisted(
    inp_sb,               # [PMAX, H_free*T] bf16 in SBUF
    dtype,
    T,
    gamma,                # [1, H] bf16 HBM
    sbm=None,
    gamma_sb_ready=None,  # [PMAX, H_free] bf16 SBUF pre-loaded
    debug_rmsnorm_out=None,
    hidden_actual=_H,
):
    """
    RMSNorm sub-kernel. inp_sb already in SBUF.

    Heap lifecycle (caller's responsibility):
      alloc_heap rmsnorm_normed_bf16 [PMAX, H_free*T] bf16  — returned; caller pops.

    Returns rmsnorm_normed_bf16 [PMAX, H_free*T] bf16 heap tensor.
    """
    H      = _H
    H_free = _H_FREE
    B      = T
    prg_id = nl.program_id(axis=0)

    # heap: bf16 normed — lives through router + expert stages; caller pops
    rmsnorm_normed_bf16 = sbm.alloc_heap((_PMAX, H_free * T), dtype, buffer=nl.sbuf, name="rmsnorm_normed_bf16")

    sbm.open_scope(name="rmsnorm")

    rmsnorm_out = inp_sb
    rmsnorm_out_view = TensorView(rmsnorm_out).reshape_dim(dim=1, shape=(H_free, T))

    # --- gamma load (skipped when gamma_sb_ready is supplied) ---
    if gamma_sb_ready is None:
        gamma_sb = sbm.alloc_stack((_PMAX, H_free), gamma.dtype, buffer=nl.sbuf, name="gamma_sb")
        gamma_1d = gamma.reshape((H,))
        for h1 in nl.static_range(H_free):
            shard = h1 // _H_FREE_SHARD
            h2 = h1 - shard * _H_FREE_SHARD
            nisa.dma_copy(
                dst=gamma_sb[0:_PMAX, h1:h1 + 1],
                src=gamma_1d.ap(
                    pattern=[[_H_FREE_SHARD, _PMAX], [1, 1]],
                    offset=shard * _H_SHARD + h2,
                ),
                dge_mode=3,
            )
    else:
        gamma_sb = gamma_sb_ready  # caller-owned — do NOT free

    rmsnorm_sq = sbm.alloc_stack((_PMAX, H_free * T), nl.float32, buffer=nl.sbuf, name="rmsnorm_sq")
    nisa.activation(rmsnorm_sq[...], op=nl.square, data=rmsnorm_out[...])

    # Within-partition reduce: [PMAX, H_free*T] → [PMAX, T]
    rmsnorm_reduced = sbm.alloc_stack((_PMAX, T), nl.float32, buffer=nl.sbuf, name="rmsnorm_reduced")
    for t in nl.static_range(T):
        rmsnorm_sq_t = TensorView(rmsnorm_sq).slice(dim=1, start=t, end=H_free * T, step=T)
        nisa.tensor_reduce(rmsnorm_reduced[0:_PMAX, t:t + 1], nl.add, rmsnorm_sq_t.get_view(), axis=1)

    # Cross-partition reduce via reduce-as-MATMUL with an all-ones [PMAX, PMAX] stationary
    # operand. Matches AwsNeuronRmsNorm's HLO lowering (nxdi_moe.md §10.8).
    mm_ones = sbm.alloc_stack((_PMAX, _PMAX), nl.float32, buffer=nl.sbuf, name="rmsnorm_mm_ones")
    nisa.memset(mm_ones, value=1.0)
    final_reduced_psum = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=final_reduced_psum[0:_PMAX, 0:T],
        stationary=mm_ones[0:_PMAX, 0:_PMAX],
        moving=rmsnorm_reduced[0:_PMAX, 0:T],
    )

    eps_sb = sbm.alloc_stack((_PMAX, 1), nl.float32, buffer=nl.sbuf, name="eps_sb")
    nisa.memset(eps_sb, value=_EPS)
    norm_factor_sb = sbm.alloc_stack((_PMAX, T), nl.float32, buffer=nl.sbuf, name="norm_factor_sb")
    nisa.activation(
        norm_factor_sb[0:_PMAX, 0:T],
        op=nl.rsqrt,
        data=final_reduced_psum[0:_PMAX, 0:T],
        scale=1.0 / hidden_actual,
        bias=eps_sb[0:_PMAX, :],
    )

    gamma_mult = sbm.alloc_stack((_PMAX, H_free * T), nl.float32, buffer=nl.sbuf, name="rmsnorm_gamma_mult")
    gamma_mult_view = TensorView(gamma_mult).reshape_dim(dim=1, shape=(H_free, T))
    gamma_bcast = TensorView(gamma_sb).expand_dim(dim=2).broadcast(dim=2, size=T)
    nisa.tensor_tensor(
        gamma_mult_view.get_view(),
        rmsnorm_out_view.get_view(),
        gamma_bcast.get_view(),
        nl.multiply,
    )

    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=1).broadcast(dim=1, size=H_free)
    rmsnorm_normed_bf16_view = TensorView(rmsnorm_normed_bf16).reshape_dim(dim=1, shape=(H_free, T))
    nisa.tensor_tensor(
        rmsnorm_normed_bf16_view.get_view(),
        gamma_mult_view.get_view(),
        norm_factor_bcast.get_view(),
        nl.multiply,
    )

    sbm.close_scope()  # frees rmsnorm stack tensors (not gamma_sb_ready — caller-owned)

    # Debug: DMA rmsnorm_normed_bf16 [PMAX, H_free*T] → HBM [T, H] (prg_id 0 only).
    if debug_rmsnorm_out is not None:
        if prg_id == 0:
            debug_flat = debug_rmsnorm_out.reshape((B, H))
            for _t in nl.static_range(B):
                for _h1 in nl.static_range(H_free):
                    _dbg_psum = nl.ndarray((1, _PMAX), dtype=dtype, buffer=nl.psum)
                    nisa.nc_transpose(
                        _dbg_psum,
                        rmsnorm_normed_bf16[0:_PMAX, _h1 * B + _t : _h1 * B + _t + 1],
                    )
                    _dbg_sb = nl.ndarray((1, _PMAX), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(_dbg_sb, _dbg_psum)
                    nisa.dma_copy(
                        dst=debug_flat[_t : _t + 1, _h1 * _PMAX : (_h1 + 1) * _PMAX],
                        src=_dbg_sb,
                        dge_mode=nisa.dge_mode.hwdge,
                    )

    return rmsnorm_normed_bf16  # [PMAX, H_free*T] bf16 heap; caller pops


@nki.jit
def rmsnorm_hbm(inp, gamma, hoisted_gamma=False, debug_rmsnorm_out=None, hidden_actual=_H):
    """
    HBM wrapper for RMSNorm.

    inp:   [T, H=2048] bf16 HBM
    gamma: [1, H=2048] bf16 HBM
    Returns output [T, H=2048] bf16 HBM (RMSNorm applied; both LNC cores
    produce identical results, prg_id==0 writes the output).
    """
    sbm    = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    T      = inp.shape[0]
    H_free = _H_FREE
    prg_id = nl.program_id(axis=0)

    # --- Load inp into SBUF ---
    sbm.open_scope(name="inp_load")
    inp_2d = inp.reshape((T, _H))
    inp_sb = sbm.alloc_stack((_PMAX, H_free * T), inp.dtype, buffer=nl.sbuf, name="inp_sb")
    for t in nl.static_range(T):
        for h1 in nl.static_range(H_free):
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

    # --- Optionally pre-load gamma into SBUF ---
    gamma_sb_ready = None
    if hoisted_gamma:
        sbm.open_scope(name="gamma_hoist")
        gamma_sb_ready     = sbm.alloc_stack((_PMAX, H_free), gamma.dtype, buffer=nl.sbuf, name="gamma_sb_hoist")
        gamma_1d           = gamma.reshape((_H,))
        for h1 in nl.static_range(H_free):
            shard = h1 // _H_FREE_SHARD
            h2 = h1 - shard * _H_FREE_SHARD
            nisa.dma_copy(
                dst=gamma_sb_ready[0:_PMAX, h1:h1 + 1],
                src=gamma_1d.ap(
                    pattern=[[_H_FREE_SHARD, _PMAX], [1, 1]],
                    offset=shard * _H_SHARD + h2,
                ),
                dge_mode=3,
            )

    rmsnorm_normed_bf16 = _rmsnorm_sbuf_in_sbuf_out_hoisted(
        inp_sb, inp.dtype, T, gamma, sbm=sbm,
        gamma_sb_ready=gamma_sb_ready,
        debug_rmsnorm_out=debug_rmsnorm_out,
        hidden_actual=hidden_actual,
    )

    # --- Store output to HBM: [PMAX, H_free*T] col-major → [T, H] row-major ---
    output = nl.ndarray((T, _H), dtype=inp.dtype, buffer=nl.shared_hbm)
    sbm.open_scope(name="store_hbm")
    out_row_sb = sbm.alloc_stack((T, _H), inp.dtype, buffer=nl.sbuf, name="out_row_sb")
    for h1 in nl.static_range(H_free):
        tp_psum = nl.ndarray((T, _PMAX), dtype=inp.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=tp_psum[0:T, 0:_PMAX],
            data=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
        )
        nisa.activation(
            dst=out_row_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)],
            op=nl.copy,
            data=tp_psum[0:T, 0:_PMAX],
        )
    if prg_id == 0:
        nisa.dma_copy(dst=output[0:T, 0:_H], src=out_row_sb[0:T, 0:_H])
    sbm.close_scope()  # store_hbm

    sbm.pop_heap()  # rmsnorm_normed_bf16

    if hoisted_gamma:
        sbm.close_scope()  # gamma_hoist
    sbm.close_scope()  # inp_load

    return output
