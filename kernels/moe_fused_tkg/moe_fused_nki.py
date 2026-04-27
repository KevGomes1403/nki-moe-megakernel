import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.allocator import SbufManager

from .rmsnorm_nki import _rmsnorm_sbuf_in_sbuf_out_hoisted
from .routertopk_nki import _routertopk_sbuf_in_sbuf_out
from .experts_nki import _experts_sbuf_in_sbuf_out

# Hardware constants
_PMAX = 128       # partition dimension max
_PSUM_FREE = 512  # PSUM free-dimension max on trn2

# Qwen3-30B-A3B at TP=4 fixed dims
_H = 2048    # hidden dim
_E = 128     # num experts
_K = 8       # top-K experts
_I = 192     # actual intermediate dim per TP rank
_I0 = 128    # first tile  (full 128 rows)
_I1 = 64     # second tile (partial: 64 valid rows, 64 zero-padded)
_I_TILES = 2 # two I-dimension tiles
_EPS = 1e-6

# Flat gate+up combined width
_GU_FLAT = 2 * _I   # = 384

# LNC=2 H-sharding constants
_N_PRGS = 2
_H_FREE = _H // _PMAX             # = 16
_H_FREE_SHARD = _H_FREE // _N_PRGS   # = 8
_H_SHARD = _H_FREE_SHARD * _PMAX     # = 1024

# Router DMA batching
_ROUTER_BATCH = 16

# 2-wave constants
_K_WAVE = 4  # experts per wave


def _qwen3_moe_sbuf_in_sbuf_out_hoisted(
    inp_sb,               # [PMAX=128, H_free*T] bf16 — already in SBUF (no HBM load needed)
    dtype,                # bf16 — explicit since inp.dtype is no longer available
    T,                    # int — number of tokens
    gamma,                # [1, H=2048] bf16 — HBM
    router_w,             # [H=2048, E=128] bf16 — HBM
    gate_up_w,            # [E=128, H=2048, 384] bf16 — HBM
    down_w,               # [E=128, 192, H=2048] bf16 — HBM
    sbm=None,             # required: SbufManager instance
    gamma_sb_ready=None,  # [PMAX, H_free=16] bf16 SBUF — pre-loaded & pre-transposed gamma
                          # When supplied, skip gamma_flat_sb alloc+DMA,
                          # gamma_trans_psum alloc+nc_transpose,
                          # gamma_sb alloc+activation copy.
    router_w_wide_sb=None,  # [PMAX, _ROUTER_BATCH=16, E=128] bf16 SBUF — replaces wide router DMA
                            # When supplied, skip the alloc + nisa.dma_copy in h_chunk loop.
    debug_rmsnorm_out=None,  # [T, H=2048] bf16 HBM — when provided, DMA rmsnorm_normed_bf16 here
):
    """
    Fused RMSNorm + Router + TopK(K=8) + Expert MLP for Qwen3 MoE TKG.
    Sub-kernel — no @nki.jit decorator. Called from inside a jitted function.
    inp_sb: [PMAX, H_free*T] bf16 already in SBUF.
    Returns: out_sb [PMAX=128, H_free*T=16*T] bf16 in SBUF — column-major, partition-first.
             caller must consume before any further sbm.alloc_stack

    New kwargs (all default None):
      gamma_sb_ready   : pre-loaded [PMAX, H_free] bf16 SBUF tensor — skips gamma DMA chain
      router_w_wide_sb : pre-loaded [PMAX, ROUTER_BATCH, E] bf16 SBUF — skips router wide DMA
    """
    # -----------------------------------------------------------------------
    # Stage 1: RMSNorm
    # -----------------------------------------------------------------------
    rmsnorm_normed_bf16 = _rmsnorm_sbuf_in_sbuf_out_hoisted(
        inp_sb, dtype, T, gamma, sbm=sbm,
        gamma_sb_ready=gamma_sb_ready,
        debug_rmsnorm_out=debug_rmsnorm_out,
    )
    # heap stack: [rmsnorm_normed_bf16]

    # -----------------------------------------------------------------------
    # Stage 2+3: Router matmul + Softmax + TopK(8)
    # -----------------------------------------------------------------------
    top8_logits_bf16, top8_idx, top8_vals_bf16 = _routertopk_sbuf_in_sbuf_out(
        rmsnorm_normed_bf16, dtype, T, router_w, sbm=sbm,
        router_w_wide_sb=router_w_wide_sb,
    )
    # heap stack: [rmsnorm_normed_bf16, top8_logits_bf16, top8_idx, top8_vals_bf16]

    # -----------------------------------------------------------------------
    # Stage 4+5: Expert MLP + output cast
    # -----------------------------------------------------------------------
    out_sb = _experts_sbuf_in_sbuf_out(
        rmsnorm_normed_bf16, top8_idx, top8_vals_bf16,
        dtype, T, gate_up_w, down_w, sbm=sbm,
    )

    # Free heap in reverse order of allocation
    sbm.pop_heap()  # top8_vals_bf16
    sbm.pop_heap()  # top8_idx
    sbm.pop_heap()  # top8_logits_bf16
    sbm.pop_heap()  # rmsnorm_normed_bf16

    return out_sb  # [PMAX=128, H_free*T=16*T] bf16 — column-major, partition-first
                   # caller must consume before any further sbm.alloc_stack


# ---------------------------------------------------------------------------
# Raw kernel body (identical to test_v30c_vs_nxdi_trn3.py)
# ---------------------------------------------------------------------------
def moe_fused_tkg(inp, gamma, router_w, gate_up_w, down_w):
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)

    T = inp.shape[0]

    inp_2d              = inp.reshape((T, _H))
    inp_2d_hbm_reshaped = inp_2d.reshape((_H_FREE * T, _PMAX))

    sbm.open_scope(name="inp_load")
    inp_flat_sb = sbm.alloc_stack(
        (_H_FREE * T, _PMAX), inp.dtype, buffer=nl.sbuf, name="inp_flat_sb"
    )
    nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_reshaped, dge_mode=3)
    inp_trans_psum = nl.ndarray((_PMAX, _H_FREE * T), dtype=inp.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)
    inp_sb = sbm.alloc_stack(
        (_PMAX, _H_FREE * T), inp.dtype, buffer=nl.sbuf, name="inp_sb"
    )
    nisa.activation(inp_sb[...], op=nl.copy, data=inp_trans_psum[...])

    router_w_wide_sb = sbm.alloc_stack(
        (_PMAX, _ROUTER_BATCH, _E), inp.dtype, buffer=nl.sbuf, name="router_w_wide_sb"
    )
    nisa.dma_copy(
        dst=router_w_wide_sb,
        src=router_w.ap(
            pattern=[[_E, _PMAX], [_PMAX * _E, _ROUTER_BATCH], [1, _E]],
            offset=0,
        ),
        dge_mode=3,
    )

    out_sb = _qwen3_moe_sbuf_in_sbuf_out_hoisted(
        inp_sb, inp.dtype, T,
        gamma, router_w, gate_up_w, down_w,
        sbm=sbm,
        gamma_sb_ready=None,
        router_w_wide_sb=router_w_wide_sb,
    )

    prg_id = nl.program_id(axis=0)
    output = nl.ndarray((T, _H), dtype=inp.dtype, buffer=nl.shared_hbm)

    sbm.open_scope(name="store_hbm")
    out_row_sb = sbm.alloc_stack(
        (T, _H_SHARD), inp.dtype, buffer=nl.sbuf, name="out_row_sb"
    )

    if prg_id == 0:
        for h1 in nl.static_range(_H_FREE_SHARD):
            tp_psum = nl.ndarray((T, _PMAX), dtype=inp.dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=tp_psum[0:T, 0:_PMAX],
                data=out_sb[0:_PMAX, nl.ds(h1 * T, T)],
            )
            nisa.activation(
                dst=out_row_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)],
                op=nl.copy,
                data=tp_psum[0:T, 0:_PMAX],
            )

        nisa.dma_copy(
            dst=output[0:T, 0:_H_SHARD],
            src=out_row_sb[0:T, 0:_H_SHARD],
        )
    else:
        for h1 in nl.static_range(_H_FREE_SHARD):
            tp_psum = nl.ndarray((T, _PMAX), dtype=inp.dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=tp_psum[0:T, 0:_PMAX],
                data=out_sb[0:_PMAX, nl.ds((h1 + _H_FREE_SHARD) * T, T)],
            )
            nisa.activation(
                dst=out_row_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)],
                op=nl.copy,
                data=tp_psum[0:T, 0:_PMAX],
            )

        nisa.dma_copy(
            dst=output[0:T, nl.ds(_H_SHARD, _H_SHARD)],
            src=out_row_sb[0:T, 0:_H_SHARD],
        )
    sbm.close_scope()  # store_hbm
    sbm.close_scope()  # inp_load

    return output
