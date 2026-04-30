import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.interleave_copy import interleave_copy

from .rmsnorm_nki import _rmsnorm_sbuf_in_sbuf_out_hoisted
from .routertopk_nki import _routertopk_sbuf_in_sbuf_out
from .experts_nki import _experts_sbuf_in_sbuf_out_hshard_exact

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
    router_w_wide_sb=None,
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
    router_in = sbm.alloc_stack(
        (_PMAX, _H_FREE * T), router_w.dtype, buffer=nl.sbuf, name="router_in"
    )
    nisa.tensor_copy(dst=router_in[0:_PMAX, 0:_H_FREE * T], src=rmsnorm_normed_bf16[0:_PMAX, 0:_H_FREE * T])
    top8_logits_bf16, top8_idx, top8_vals_bf16 = _routertopk_sbuf_in_sbuf_out(
        router_in, router_w.dtype, T, router_w, sbm=sbm,
        router_w_wide_sb=router_w_wide_sb,
    )
    # heap stack: [rmsnorm_normed_bf16, top8_logits_bf16, top8_idx, top8_vals_bf16]

    # -----------------------------------------------------------------------
    # Stage 4+5: Expert MLP + output cast
    # -----------------------------------------------------------------------
    out_sb = _experts_sbuf_in_sbuf_out_hshard_exact(
        rmsnorm_normed_bf16, top8_idx, top8_vals_bf16,
        dtype, T, gate_up_w, down_w, sbm=sbm,
    )

    # Free heap in reverse order of allocation
    sbm.pop_heap()  # top8_vals_bf16
    sbm.pop_heap()  # top8_idx
    sbm.pop_heap()  # top8_logits_bf16
    sbm.pop_heap()  # rmsnorm_normed_bf16

    return out_sb  # [PMAX=128, T, H_free_shard=8] bf16 — local H shard, partition-first
                   # caller must consume before any further sbm.alloc_stack


# ---------------------------------------------------------------------------
# Raw kernel body (identical to test_v30c_vs_nxdi_trn3.py)
# ---------------------------------------------------------------------------
def moe_fused_tkg(inp, gamma, router_w, gate_up_w, down_w):
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)

    T = inp.shape[0]

    sbm.open_scope(name="inp_load")
    inp_2d = inp.reshape((T, _H))
    inp_sb = sbm.alloc_stack(
        (_PMAX, _H_FREE * T), inp.dtype, buffer=nl.sbuf, name="inp_sb"
    )
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

    out_sb = _qwen3_moe_sbuf_in_sbuf_out_hoisted(
        inp_sb, inp.dtype, T,
        gamma, router_w, gate_up_w, down_w,
        sbm=sbm,
        gamma_sb_ready=None,
    )

    prg_id = nl.program_id(axis=0)
    output = nl.ndarray((T, _H), dtype=inp.dtype, buffer=nl.shared_hbm)

    sbm.open_scope(name="store_hbm")
    out_tile_sb = sbm.alloc_stack(
        (T, _H_SHARD), inp.dtype, buffer=nl.sbuf, name="out_tile_sb"
    )

    for t in nl.static_range(T):
        for h1 in nl.static_range(_H_FREE_SHARD):
            tp_psum = nl.ndarray((1, _PMAX), dtype=inp.dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=tp_psum[0:1, 0:_PMAX],
                data=out_sb[0:_PMAX, t, h1:h1 + 1],
            )
            interleave_copy(
                dst=out_tile_sb.ap(
                    pattern=[[_H_SHARD, 1], [_H_FREE_SHARD, _PMAX]],
                    offset=t * _H_SHARD + h1,
                ),
                src=tp_psum[0:1, 0:_PMAX],
                index=h1,
            )
        if prg_id == 0:
            nisa.dma_copy(
                dst=output[t:t + 1, 0:_H_SHARD],
                src=out_tile_sb[t:t + 1, 0:_H_SHARD],
            )
        else:
            nisa.dma_copy(
                dst=output[t:t + 1, _H_SHARD:_H],
                src=out_tile_sb[t:t + 1, 0:_H_SHARD],
            )
    sbm.close_scope()  # store_hbm
    sbm.close_scope()  # inp_load

    return output
