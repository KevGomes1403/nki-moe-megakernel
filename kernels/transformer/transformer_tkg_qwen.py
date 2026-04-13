"""
Qwen3 MoE transformer megakernel: single layer (attention + MoE), SBUF allreduce, LNC=2.

Hardware target: Trainium2 (trn2), TP=2 (LNC=2 sharding).
"""

import math
import numpy as np
from typing import List, Optional
import nki.isa as nisa
import nki.language as nl
import nki.collectives as nccl
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.core.utils.tensor_view import TensorView
from nkilib.core.utils.allocator import BufferManager, Logger

# ---------------------------------------------------------------------------
# Hardware / Qwen3 model dims (all hardcoded)
# ---------------------------------------------------------------------------
PMAX        = 128
F_MAX       = 512
H           = 2048
H0          = PMAX          # = 128
H1          = H // H0       # = 16
N_PRGS      = 2
H1_SHARD    = H1 // N_PRGS  # = 8
H_SHARD     = H1_SHARD * H0 # = 1024
D_HEAD      = 128
HQ_TP       = 8
GQA         = 8
HQ_OUT      = HQ_TP * D_HEAD   # = 1024
NUM_H_TILES = H // PMAX         # = 16
HQ_TP_CONST = 8
INV_SQRT_D  = float(1.0 / math.sqrt(128.0))
EPS         = 1e-6

# MoE dims
_PMAX        = 128
E            = 128
K_EXPERTS    = 8
I_DIM        = 192
I0           = 128
I1           = 64
I_TILES      = 2
GU_FLAT      = 2 * I_DIM       # = 384
H_FREE       = H // _PMAX      # = 16
H_FREE_SHARD = H_FREE // N_PRGS  # = 8
K_WAVE       = 4
ROUTER_BATCH = 16

SBM_SIZE_BYTES = 128 * 1024  # 128 KB: all SBUF routed through SBM; headroom for stack+heap

# ---------------------------------------------------------------------------
# Helper functions — verbatim copy from transformer_tkg.py
# ---------------------------------------------------------------------------

def _load_input_to_sbuf(dst_sb, src_hbm, BxS: int, H0: int, H1: int, H1_shard: int, n_prgs: int):
    """Load [B, S_tkg, H] HBM tensor to [H0, BxS*H1] SBUF layout."""
    src_view = TensorView(src_hbm.reshape((BxS, H0 * H1))).rearrange(
        ('bs', ('lnc', 'h0', 'h1')), ('h0', 'bs', 'lnc', 'h1'), {'lnc': n_prgs, 'h0': H0}
    )
    dst_reshaped = dst_sb.reshape((H0, BxS, n_prgs, H1_shard))
    for lnc_idx in nl.static_range(n_prgs):
        nisa.dma_copy(
            src=src_view.slice(dim=2, start=lnc_idx, end=lnc_idx + 1).get_view(),
            dst=dst_reshaped[:, :, lnc_idx : lnc_idx + 1, :],
        )


def _store_output_to_hbm(out_hbm, in_sb, BxS: int, H0: int, H1: int, H1_shard: int, n_prgs: int):
    """Store [H0, BxS*H1] SBUF tensor to [B, S_tkg, H] HBM layout."""
    src_reshaped = in_sb.reshape((H0, BxS, n_prgs, H1_shard))
    dst_view = TensorView(out_hbm.reshape((BxS, H0 * H1))).rearrange(
        ('bs', ('lnc', 'h0', 'h1')), ('h0', 'bs', 'lnc', 'h1'), {'lnc': n_prgs, 'h0': H0}
    )
    for lnc_idx in nl.static_range(n_prgs):
        nisa.dma_copy(
            src=src_reshaped[:, :, lnc_idx : lnc_idx + 1, :],
            dst=dst_view.slice(dim=2, start=lnc_idx, end=lnc_idx + 1).get_view(),
        )


def _gather_shards_via_barrier(
    sharded_sb, gather_hbm, output_sb,
    BxS: int, H0: int, H1: int, H1_shard: int, n_prgs: int, prg_id: int,
):
    """Gather column-sharded SBUF results into a full SBUF tensor via nisa.core_barrier.

    Each rank has computed its shard in sharded_sb: [H0, BxS*H1_shard].
    This function:
      1. Each rank writes its shard into the correct LNC slot of gather_hbm.
         prg_id is a compile-time integer, so the if/else selects one DMA per rank
         and the two writes go to disjoint non-overlapping HBM regions.
      2. nisa.core_barrier so both ranks have finished writing.
      3. Loads the full gather_hbm back to output_sb [H0, BxS*H1] SBUF.

    sharded_sb:  [H0, BxS*H1_shard] sbuf       — this rank's computed shard
    gather_hbm:  [BxS, H0*H1]       shared_hbm — full gather buffer (pre-allocated by caller,
                                                  same interleaved layout as input HBM)
    output_sb:   [H0, BxS*H1]       sbuf        — full gathered output (pre-allocated by caller)
    prg_id:      compile-time int 0 or 1
    """
    # Each rank writes to its own LNC slot — disjoint HBM regions, no write conflict.
    # prg_id is compile-time, so this is a static branch (one DMA per rank).
    src_reshaped = sharded_sb.reshape((H0, BxS, 1, H1_shard))
    dst_view = TensorView(gather_hbm.reshape((BxS, H0 * H1))).rearrange(
        ('bs', ('lnc', 'h0', 'h1')), ('h0', 'bs', 'lnc', 'h1'), {'lnc': n_prgs, 'h0': H0}
    )
    if prg_id == 0:
        nisa.dma_copy(
            src=src_reshaped[:, :, 0:1, :],
            dst=dst_view.slice(dim=2, start=0, end=1).get_view(),
            dge_mode=nisa.dge_mode.hwdge,
        )
    else:
        nisa.dma_copy(
            src=src_reshaped[:, :, 0:1, :],
            dst=dst_view.slice(dim=2, start=1, end=2).get_view(),
            dge_mode=nisa.dge_mode.hwdge,
        )

    if n_prgs > 1:
        nisa.core_barrier(data=gather_hbm, cores=(0, 1))

    # Load the full gathered HBM tensor back to SBUF
    _load_input_to_sbuf(output_sb, gather_hbm, BxS, H0, H1, H1_shard, n_prgs)


def _hbm_all_reduce_gather(
    sharded_sb, dtype, replica_group,
    prg_id: int, n_prgs: int, H0: int, H1: int, H1_shard: int, BxS: int,
    name_prefix: str = "",
):
    """HBM-based all-reduce + gather fallback for when SBUF all_reduce fails.

    Steps:
      1. DMA sharded_sb [H0, h1_shard] → shard_hbm (shared_hbm)
      2. HBM all_reduce: ar_hbm ← sum of both cores' shard_hbm
      3. Each core writes ar_hbm to its LNC slot in gather_hbm (same layout as X input)
      4. core_barrier to sync
      5. _load_input_to_sbuf to load full gather_hbm → output_sb

    Returns output_sb [H0, BxS*H1].
    """
    p = name_prefix

    # Step 1: DMA SBUF shard to HBM.
    # Use nl.static_range(n_prgs) so both LNC slots get written (unrolled at
    # trace time), matching the pattern NCC's localize_shared_memory uses to
    # decide tensors should be forked per-core.
    # Layout: shard_hbm_all[lnc_idx, H0, h1_shard*BxS]
    shard_hbm_all = nl.ndarray((n_prgs * H0, H1_shard * BxS), dtype=dtype,
                                buffer=nl.shared_hbm, name=f"{p}shard_hbm_all")
    for lnc_idx in nl.static_range(n_prgs):
        nisa.dma_copy(dst=shard_hbm_all[lnc_idx*H0:(lnc_idx+1)*H0, :],
                      src=sharded_sb)

    # Each core reads its own slot — accessed via prg_id
    shard_for_ar = shard_hbm_all[prg_id*H0:(prg_id+1)*H0, :]

    # Step 2: HBM all_reduce
    ar_hbm = nl.ndarray((H0, H1_shard * BxS), dtype=dtype,
                         buffer=nl.shared_hbm, name=f"{p}ar_hbm")
    if replica_group is not None:
        nccl.all_reduce(dsts=[ar_hbm], srcs=[shard_for_ar],
                        op=nl.add, replica_group=replica_group)
    else:
        nisa.dma_copy(dst=ar_hbm, src=shard_for_ar)

    # Step 3-5: Gather both cores' ar_hbm via shared_hbm + barrier + reload
    # gather_hbm uses the same interleaved layout as the X input: [BxS, H0*H1]
    gather_hbm = nl.ndarray((BxS, H0 * H1), dtype=dtype,
                              buffer=nl.shared_hbm, name=f"{p}gather_hbm")

    ar_hbm_reshaped = ar_hbm.reshape((H0, BxS, 1, H1_shard))
    dst_view = TensorView(gather_hbm.reshape((BxS, H0 * H1))).rearrange(
        ('bs', ('lnc', 'h0', 'h1')), ('h0', 'bs', 'lnc', 'h1'), {'lnc': n_prgs, 'h0': H0}
    )
    if prg_id == 0:
        nisa.dma_copy(
            src=ar_hbm_reshaped[:, :, 0:1, :],
            dst=dst_view.slice(dim=2, start=0, end=1).get_view(),
            dge_mode=nisa.dge_mode.hwdge,
        )
    else:
        nisa.dma_copy(
            src=ar_hbm_reshaped[:, :, 0:1, :],
            dst=dst_view.slice(dim=2, start=1, end=2).get_view(),
            dge_mode=nisa.dge_mode.hwdge,
        )

    if n_prgs > 1:
        nisa.core_barrier(data=gather_hbm, cores=(0, 1))

    # Load the full gathered HBM back to SBUF
    output_sb = nl.ndarray((H0, BxS * H1), dtype=dtype,
                            buffer=nl.sbuf, name=f"{p}ar_out_sb")
    _load_input_to_sbuf(output_sb, gather_hbm, BxS, H0, H1, H1_shard, n_prgs)

    return output_sb


def _sb2sb_gather(
    sharded_sb, dtype,
    prg_id: int, n_prgs: int, H0: int, H1: int, H1_shard: int, BxS: int,
    name_prefix: str = "",
):
    """Gather column-sharded SBUF tensors via sendrecv (no all_reduce).

    Each LNC holds its own H1_shard columns of the output.  Cross-LNC gather
    is done with nisa.sendrecv so both LNCs end up with the full [H0, BxS*H1]
    output.  No nccl.all_reduce is used, avoiding NCC_ILLC059.

    With n_prgs=1 (non-SPMD tracing) the sendrecv block is skipped and the
    function is a simple layout rearrangement.
    """
    p = name_prefix
    gathered_sb = nl.ndarray((H0, H1 * BxS), dtype=dtype, buffer=nl.sbuf,
                              name=f"{p}gathered_sb")

    # Copy own shard into the correct interleaved slot
    f_shard = nl.ds(start=prg_id * BxS * H1_shard, size=BxS * H1_shard)
    nisa.tensor_copy(dst=gathered_sb[:, f_shard], src=sharded_sb)

    if n_prgs > 1:
        other_lnc = 1 - prg_id
        f_other_shard = nl.ds(start=other_lnc * BxS * H1_shard, size=BxS * H1_shard)
        nisa.sendrecv(
            src=sharded_sb,
            dst=gathered_sb[:, f_other_shard],
            send_to_rank=other_lnc,
            recv_from_rank=other_lnc,
            pipe_id=0,
        )

    output_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                            name=f"{p}out_sb")
    src_view = TensorView(gathered_sb).rearrange(
        ('h0', ('h1', 'bs')), ('h0', 'bs', 'h1'), {'h1': H1}
    )
    nisa.tensor_copy(dst=output_sb.reshape((H0, BxS, H1)), src=src_view.get_view())

    return output_sb


def _sb2sb_all_reduce_gather(
    sharded_sb, dtype, replica_group,
    prg_id: int, n_prgs: int, H0: int, H1: int, H1_shard: int, BxS: int,
    name_prefix: str = "",
):
    """SB2SB all-reduce + local gather via SBUF only (no HBM roundtrip).
    Uses nccl.all_reduce on auto-alloc SBUF tensors — matches nkilib pattern.
    Returns (output_sb [H0, BxS*H1], sharded_AR_sb).
    """
    p = name_prefix
    sharded_AR_sb = nl.ndarray(sharded_sb.shape, dtype=dtype, buffer=nl.sbuf,
                                name=f"{p}ar_dst_sb")
    nccl.all_reduce(dsts=[sharded_AR_sb], srcs=[sharded_sb],
                    op=nl.add, replica_group=replica_group)

    gathered_sb = nl.ndarray((H0, H1 * BxS), dtype=dtype, buffer=nl.sbuf,
                              name=f"{p}gathered_sb")
    f_shard = nl.ds(start=prg_id * BxS * H1_shard, size=BxS * H1_shard)
    nisa.tensor_copy(dst=gathered_sb[:, f_shard], src=sharded_AR_sb)

    if n_prgs > 1:
        other_lnc = 1 - prg_id
        f_other_shard = nl.ds(start=other_lnc * BxS * H1_shard, size=BxS * H1_shard)
        nisa.sendrecv(
            src=sharded_AR_sb,
            dst=gathered_sb[:, f_other_shard],
            send_to_rank=other_lnc,
            recv_from_rank=other_lnc,
            pipe_id=0,
        )

    output_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                            name=f"{p}ar_out_sb")
    src_view = TensorView(gathered_sb).rearrange(
        ('h0', ('h1', 'bs')), ('h0', 'bs', 'h1'), {'h1': H1}
    )
    nisa.tensor_copy(dst=output_sb.reshape((H0, BxS, H1)), src=src_view.get_view())
    return output_sb, sharded_AR_sb


# ---------------------------------------------------------------------------
# _qwen3_attn_tkg_sbuf_out — modified attention (returns SBUF shard)
# ---------------------------------------------------------------------------

def _qwen3_attn_tkg_sbuf_out(
    hidden_states,    # [B=1, 1, H=2048] bf16
    Wq,               # [1024, 2048] bf16
    Wk,               # [128, 2048] bf16
    Wv,               # [128, 2048] bf16
    Wo,               # [1024, 2048] bf16 — FULL Wo, each core loads its H_SHARD half
    q_norm_weight,    # [128] bf16
    k_norm_weight,    # [128] bf16
    K_cache,          # [B, 1, S_prior, 128] bf16
    V_cache,          # [B, 1, S_prior, 128] bf16
    cos,              # [B, 128] bf16
    sin,              # [B, 128] bf16
    position_ids,     # [B, 1] int32
    prg_id,           # nl.program_id result — compile-time constant 0 or 1
    k_rope_out,       # [B, 128] nl.shared_hbm — pre-allocated, written here
    v_out,            # [B, 128] nl.shared_hbm — pre-allocated, written here
    h1_shard=H1_SHARD,  # dynamic: H1 // n_prgs (8 normally, 16 if n_prgs=1)
    out_sb=None,      # optional: if provided, write into it; else alloc_stack internally
    sbm=None,         # required: BufferManager for alloc_stack
):
    """Returns attn_sharded_sb: [H0, BxS*h1_shard] SBUF — this rank's output column shard."""

    # --- Dimensions ---
    B = hidden_states.shape[0]      # 1
    H_dim = hidden_states.shape[2]  # 2048
    Hq_out = Wq.shape[0]            # 1024  = Hq_tp * d
    d = PMAX                        # 128
    Hq_tp = Hq_out // d             # 8
    Hkv_tp = 1                      # per-rank KV heads (corrected)
    GQA_local = Hq_tp // Hkv_tp    # 8
    S_prior = K_cache.shape[2]
    num_h_tiles = H_dim // PMAX     # 16
    num_s_tiles = S_prior // PMAX

    half_d = d // 2                 # 64

    # Dynamic shard: h1_shard * H0 columns per core
    h_shard = h1_shard * H0        # 1024 when n_prgs=2, 2048 when n_prgs=1
    num_out_blocks = h_shard // F_MAX  # 2 when n_prgs=2, 4 when n_prgs=1

    # Output H: each core computes h_shard of Wo output
    H_wo = Wo.shape[1]              # 2048 (full, each core selects its h_shard half)

    # Static shape constants
    assert S_prior % PMAX == 0, f"S_prior={S_prior} must be a multiple of {PMAX}"
    NUM_S_TILES  = S_prior // PMAX
    NUM_H_TILES_LOCAL = 16
    HQ_TP_LOCAL  = 8
    assert H_dim == NUM_H_TILES_LOCAL * PMAX
    assert Hq_tp == HQ_TP_LOCAL

    # K/V outputs — written here (pre-allocated by caller)
    # hidden: [B, 1, H] -> [H, B] column layout
    hidden_col = hidden_states.reshape((H_dim, B))
    # cos/sin: [B, d] -> [PMAX, B]
    cos_col = cos.reshape((PMAX, B))
    sin_col = sin.reshape((PMAX, B))

    # =========================================================================
    # LOAD CONSTANTS
    # =========================================================================
    qnw_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="qnw_bf16")
    nisa.dma_copy(dst=qnw_bf16, src=q_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
    qnw_sb = sbm.alloc_stack((PMAX, 1), nl.float32, name="qnw_sb")
    nisa.tensor_copy(qnw_sb, qnw_bf16)

    knw_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="knw_bf16")
    nisa.dma_copy(dst=knw_bf16, src=k_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
    knw_sb = sbm.alloc_stack((PMAX, 1), nl.float32, name="knw_sb")
    nisa.tensor_copy(knw_sb, knw_bf16)

    cos_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="cos_bf16")
    sin_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="sin_bf16")
    nisa.dma_copy(dst=cos_bf16, src=cos_col, dge_mode=nisa.dge_mode.hwdge)
    nisa.dma_copy(dst=sin_bf16, src=sin_col, dge_mode=nisa.dge_mode.hwdge)
    cos_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name="cos_f32")
    sin_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name="sin_f32")
    nisa.tensor_copy(cos_f32, cos_bf16)
    nisa.tensor_copy(sin_f32, sin_bf16)

    rms_ones = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name="rms_ones")
    nisa.memset(rms_ones, value=1.0)

    # =========================================================================
    # HIDDEN TILE HOISTING
    # =========================================================================
    h_all = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="h_all")
    nisa.dma_copy(
        dst=h_all,
        src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )

    # =========================================================================
    # Pre-load all 8 Wq head tiles (per-h_t transposed tile, one buffer per head reused in loop)
    # Wq: [Hq_out=1024, H_dim=2048] — standard [out, in] layout.
    # For correct nc_matmul: stationary[p, m] = Wq[q_h*PMAX+m, h_t*PMAX+p]
    # achieved via AP pattern [[1, PMAX], [H_dim, PMAX]] with offset = q_h*PMAX*H_dim + h_t*PMAX.
    # =========================================================================
    wq_tiles = []
    for q_h in nl.affine_range(HQ_TP_CONST):
        wq_tile_qh = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name=f"wq_tile_{q_h}")
        wq_tiles.append(wq_tile_qh)

    # WO WEIGHT RESHAPE
    Wo_reshaped = Wo.reshape((Hq_tp, d, H_wo))  # logical [8, 128, 2048] view

    # =========================================================================
    # K PROJECTION
    # Wk: [D_HEAD=128, H_dim=2048] — standard [out, in] layout.
    # For correct nc_matmul: stationary[p, m] = Wk[m, h_t*PMAX+p]
    # achieved via AP pattern [[1, PMAX], [H_dim, PMAX]] with offset = h_t*PMAX.
    # =========================================================================
    wk_tile = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name="wk_tile")
    wv_tile = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name="wv_tile")

    k_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="k_psum")
    for h_t in nl.affine_range(NUM_H_TILES_LOCAL):
        nisa.dma_copy(dst=wk_tile, src=Wk.ap([[1, PMAX], [H_dim, PMAX]], offset=h_t * PMAX), dge_mode=nisa.dge_mode.hwdge)
        nisa.nc_matmul(k_psum, stationary=wk_tile, moving=h_all[0:PMAX, h_t:h_t+1])

    k_vec = sbm.alloc_stack((PMAX, B), nl.float32, name="k_vec")
    nisa.tensor_copy(k_vec, k_psum)

    # K RMSNorm
    k_sq = sbm.alloc_stack((PMAX, B), nl.float32, name="k_sq")
    nisa.tensor_tensor(k_sq, k_vec, k_vec, op=nl.multiply)
    k_sq_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_sq_bf16")
    nisa.tensor_copy(k_sq_bf16, k_sq)
    k_sum_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="k_sum_psum")
    nisa.nc_matmul(k_sum_psum, stationary=rms_ones, moving=k_sq_bf16)
    k_sum_sb = sbm.alloc_stack((PMAX, B), nl.float32, name="k_sum_sb")
    nisa.tensor_copy(k_sum_sb, k_sum_psum)
    k_mean_sq = sbm.alloc_stack((PMAX, B), nl.float32, name="k_mean_sq")
    nisa.tensor_scalar(k_mean_sq, k_sum_sb, op0=nl.multiply, operand0=1.0/d, op1=nl.add, operand1=EPS)
    k_rms_inv = sbm.alloc_stack((PMAX, B), nl.float32, name="k_rms_inv")
    nisa.activation(k_rms_inv, op=nl.rsqrt, data=k_mean_sq)
    k_normed = sbm.alloc_stack((PMAX, B), nl.float32, name="k_normed")
    nisa.tensor_tensor(k_normed, k_vec, k_rms_inv, op=nl.multiply)
    k_normed2 = sbm.alloc_stack((PMAX, B), nl.float32, name="k_normed2")
    nisa.tensor_tensor(k_normed2, k_normed, knw_sb, op=nl.multiply)

    # K RoPE
    rot_k = sbm.alloc_stack((PMAX, B), nl.float32, name="rot_k")
    neg_k_upper = sbm.alloc_stack((half_d, B), nl.float32, name="neg_k_upper")
    nisa.tensor_scalar(neg_k_upper, k_normed2[half_d:d, 0:B], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(rot_k[0:half_d, 0:B], neg_k_upper)
    nisa.tensor_copy(rot_k[half_d:d, 0:B], k_normed2[0:half_d, 0:B])
    k_cos = sbm.alloc_stack((PMAX, B), nl.float32, name="k_cos")
    k_sin_part = sbm.alloc_stack((PMAX, B), nl.float32, name="k_sin_part")
    nisa.tensor_tensor(k_cos, k_normed2, cos_f32, op=nl.multiply)
    nisa.tensor_tensor(k_sin_part, rot_k, sin_f32, op=nl.multiply)
    k_rope = sbm.alloc_stack((PMAX, B), nl.float32, name="k_rope")
    nisa.tensor_tensor(k_rope, k_cos, k_sin_part, op=nl.add)

    # Store k_rope to HBM for KV cache update
    k_rope_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="k_rope_bf16")
    nisa.tensor_copy(k_rope_bf16, k_rope)
    k_rope_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name="k_rope_T_psum")
    nisa.nc_transpose(k_rope_T_psum, k_rope_bf16)
    k_rope_T_sb = sbm.alloc_stack((B, PMAX), nl.bfloat16, name="k_rope_T_sb")
    nisa.tensor_copy(k_rope_T_sb, k_rope_T_psum)
    nisa.dma_copy(dst=k_rope_out, src=k_rope_T_sb, dge_mode=nisa.dge_mode.hwdge)

    # =========================================================================
    # V PROJECTION
    # Wv: [D_HEAD=128, H_dim=2048] — standard [out, in] layout.
    # For correct nc_matmul: stationary[p, m] = Wv[m, h_t*PMAX+p]
    # achieved via AP pattern [[1, PMAX], [H_dim, PMAX]] with offset = h_t*PMAX.
    # =========================================================================
    v_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name="v_psum")
    for h_t in nl.affine_range(NUM_H_TILES_LOCAL):
        nisa.dma_copy(dst=wv_tile, src=Wv.ap([[1, PMAX], [H_dim, PMAX]], offset=h_t * PMAX), dge_mode=nisa.dge_mode.hwdge)
        nisa.nc_matmul(v_psum, stationary=wv_tile, moving=h_all[0:PMAX, h_t:h_t+1])

    v_active = sbm.alloc_stack((PMAX, B), nl.float32, name="v_active")
    nisa.tensor_copy(v_active, v_psum)

    # Store v_active to HBM for KV cache update
    v_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="v_bf16")
    nisa.tensor_copy(v_bf16, v_active)
    v_T_psum = nl.ndarray((B, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name="v_T_psum")
    nisa.nc_transpose(v_T_psum, v_bf16)
    v_T_sb = sbm.alloc_stack((B, PMAX), nl.bfloat16, name="v_T_sb")
    nisa.tensor_copy(v_T_sb, v_T_psum)
    nisa.dma_copy(dst=v_out, src=v_T_sb, dge_mode=nisa.dge_mode.hwdge)

    # =========================================================================
    # Q PROJECTIONS
    # =========================================================================
    q_packed_f32 = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="q_packed_f32")
    q_head_tmp = sbm.alloc_stack((PMAX, 1), nl.float32, name="q_head_tmp")
    for q_h in nl.affine_range(HQ_TP_CONST):
        q_psum = nl.zeros((PMAX, B), dtype=nl.float32, buffer=nl.psum, name=f"q_psum_{q_h}")
        for h_t in nl.affine_range(NUM_H_TILES_LOCAL):
            # Load transposed Wq block: wq_tile[p, m] = Wq[q_h*PMAX+m, h_t*PMAX+p]
            # AP: partition_stride=1, partition_count=PMAX, free_stride=H_dim, free_count=PMAX
            # offset = q_h*PMAX*H_dim + h_t*PMAX
            nisa.dma_copy(
                dst=wq_tiles[q_h],
                src=Wq.ap([[1, PMAX], [H_dim, PMAX]], offset=q_h * PMAX * H_dim + h_t * PMAX),
                dge_mode=nisa.dge_mode.hwdge,
            )
            nisa.nc_matmul(
                q_psum,
                stationary=wq_tiles[q_h],
                moving=h_all[0:PMAX, h_t:h_t + 1],
            )
        # PSUM -> stack temp (legal: same shape, no stride issues)
        nisa.tensor_copy(q_head_tmp, q_psum)
        # stack temp -> sub-slice of q_packed_f32 (legal: SBUF->SBUF same-address-space copy)
        nisa.tensor_copy(q_packed_f32[0:PMAX, q_h:q_h + 1], q_head_tmp)

    # =========================================================================
    # PACKED Q RMSNORM on [PMAX, GQA=8]
    # =========================================================================
    q_sq = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="q_sq")
    nisa.tensor_tensor(q_sq, q_packed_f32, q_packed_f32, op=nl.multiply)
    q_sq_bf16 = sbm.alloc_stack((PMAX, GQA_local), nl.bfloat16, name="q_sq_bf16")
    nisa.tensor_copy(q_sq_bf16, q_sq)
    q_sum_psum = nl.zeros((PMAX, GQA_local), dtype=nl.float32, buffer=nl.psum, name="q_sum_psum")
    nisa.nc_matmul(q_sum_psum, stationary=rms_ones, moving=q_sq_bf16)
    q_sum_sb = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="q_sum_sb")
    nisa.tensor_copy(q_sum_sb, q_sum_psum)
    q_mean_sq = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="q_mean_sq")
    nisa.tensor_scalar(q_mean_sq, q_sum_sb, op0=nl.multiply, operand0=1.0/d, op1=nl.add, operand1=EPS)
    q_rms_inv = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="q_rms_inv")
    nisa.activation(q_rms_inv, op=nl.rsqrt, data=q_mean_sq)
    q_normed = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="q_normed")
    nisa.tensor_tensor(q_normed, q_packed_f32, q_rms_inv, op=nl.multiply)

    # Apply norm weight: qnw_sb [PMAX, 1] broadcast to [PMAX, GQA]
    qnw_gqa_psum_T = nl.ndarray((GQA_local, PMAX), dtype=nl.float32, buffer=nl.psum, name="qnw_gqa_psum_T")
    nisa.nc_transpose(qnw_gqa_psum_T, qnw_sb.ap([[1, PMAX], [0, GQA_local]], offset=0))
    qnw_gqa_sbuf_T = sbm.alloc_stack((GQA_local, PMAX), nl.float32, name="qnw_gqa_sbuf_T")
    nisa.tensor_copy(qnw_gqa_sbuf_T, qnw_gqa_psum_T)
    qnw_gqa_psum = nl.ndarray((PMAX, GQA_local), dtype=nl.float32, buffer=nl.psum, name="qnw_gqa_psum")
    nisa.nc_transpose(qnw_gqa_psum, qnw_gqa_sbuf_T)
    qnw_gqa = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="qnw_gqa")
    nisa.tensor_copy(qnw_gqa, qnw_gqa_psum)

    q_normed2 = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="q_normed2")
    nisa.tensor_tensor(q_normed2, q_normed, qnw_gqa, op=nl.multiply)

    # =========================================================================
    # PACKED Q ROPE on [PMAX, GQA=8]
    # =========================================================================
    cos_gqa_psum_T = nl.ndarray((GQA_local, PMAX), dtype=nl.float32, buffer=nl.psum, name="cos_gqa_psum_T")
    nisa.nc_transpose(cos_gqa_psum_T, cos_f32.ap([[1, PMAX], [0, GQA_local]], offset=0))
    cos_gqa_sbuf_T = sbm.alloc_stack((GQA_local, PMAX), nl.float32, name="cos_gqa_sbuf_T")
    nisa.tensor_copy(cos_gqa_sbuf_T, cos_gqa_psum_T)
    cos_gqa_psum = nl.ndarray((PMAX, GQA_local), dtype=nl.float32, buffer=nl.psum, name="cos_gqa_psum")
    nisa.nc_transpose(cos_gqa_psum, cos_gqa_sbuf_T)
    cos_gqa = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="cos_gqa")
    nisa.tensor_copy(cos_gqa, cos_gqa_psum)

    sin_gqa_psum_T = nl.ndarray((GQA_local, PMAX), dtype=nl.float32, buffer=nl.psum, name="sin_gqa_psum_T")
    nisa.nc_transpose(sin_gqa_psum_T, sin_f32.ap([[1, PMAX], [0, GQA_local]], offset=0))
    sin_gqa_sbuf_T = sbm.alloc_stack((GQA_local, PMAX), nl.float32, name="sin_gqa_sbuf_T")
    nisa.tensor_copy(sin_gqa_sbuf_T, sin_gqa_psum_T)
    sin_gqa_psum = nl.ndarray((PMAX, GQA_local), dtype=nl.float32, buffer=nl.psum, name="sin_gqa_psum")
    nisa.nc_transpose(sin_gqa_psum, sin_gqa_sbuf_T)
    sin_gqa = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="sin_gqa")
    nisa.tensor_copy(sin_gqa, sin_gqa_psum)

    rot_q = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="rot_q")
    neg_q_upper = sbm.alloc_stack((half_d, GQA_local), nl.float32, name="neg_q_upper")
    nisa.tensor_scalar(neg_q_upper, q_normed2[half_d:d, 0:GQA_local], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(rot_q[0:half_d, 0:GQA_local], neg_q_upper)
    nisa.tensor_copy(rot_q[half_d:d, 0:GQA_local], q_normed2[0:half_d, 0:GQA_local])

    q_cos = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="q_cos")
    q_sin_part = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="q_sin_part")
    nisa.tensor_tensor(q_cos, q_normed2, cos_gqa, op=nl.multiply)
    nisa.tensor_tensor(q_sin_part, rot_q, sin_gqa, op=nl.multiply)
    q_rope = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="q_rope")
    nisa.tensor_tensor(q_rope, q_cos, q_sin_part, op=nl.add)

    q_bf16 = sbm.alloc_stack((PMAX, GQA_local), nl.bfloat16, name="q_bf16")
    nisa.tensor_scalar(q_bf16, q_rope, op0=nl.multiply, operand0=INV_SQRT_D)

    # =========================================================================
    # WO WEIGHT HOISTING — load this core's H_SHARD columns
    # =========================================================================
    wo_sbuf = []
    for head in nl.affine_range(HQ_TP_CONST):
        wo_tile = sbm.alloc_stack((PMAX, h_shard), nl.bfloat16, name=f"wo_tile_h{head}")
        nisa.dma_copy(
            dst=wo_tile,
            src=Wo_reshaped.ap(
                pattern=[[H_wo, PMAX], [1, h_shard]],
                offset=head * PMAX * H_wo + prg_id * h_shard,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )
        wo_sbuf.append(wo_tile)

    # =========================================================================
    # TWO-PASS FLASH DECODE
    # =========================================================================
    K_cache_2d = K_cache.reshape((S_prior, d))   # [S_prior, 128]
    V_cache_2d = V_cache.reshape((S_prior, d))   # [S_prior, 128]

    # --- Active position score ---
    k_rope_packed_psum_T = nl.ndarray((GQA_local, PMAX), dtype=nl.float32, buffer=nl.psum, name="k_rope_packed_psum_T")
    nisa.nc_transpose(k_rope_packed_psum_T, k_rope.ap([[1, PMAX], [0, GQA_local]], offset=0))
    k_rope_packed_sbuf_T = sbm.alloc_stack((GQA_local, PMAX), nl.float32, name="k_rope_packed_sbuf_T")
    nisa.tensor_copy(k_rope_packed_sbuf_T, k_rope_packed_psum_T)
    k_rope_packed_psum = nl.ndarray((PMAX, GQA_local), dtype=nl.float32, buffer=nl.psum, name="k_rope_packed_psum")
    nisa.nc_transpose(k_rope_packed_psum, k_rope_packed_sbuf_T)
    k_rope_packed = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="k_rope_packed")
    nisa.tensor_copy(k_rope_packed, k_rope_packed_psum)

    kq_elem = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="kq_elem")
    nisa.tensor_tensor(kq_elem, k_rope_packed, q_bf16, op=nl.multiply)
    kq_elem_bf16 = sbm.alloc_stack((PMAX, GQA_local), nl.bfloat16, name="kq_elem_bf16")
    nisa.tensor_copy(kq_elem_bf16, kq_elem)
    score_active_psum = nl.zeros((PMAX, GQA_local), dtype=nl.float32, buffer=nl.psum, name="score_active_psum")
    nisa.nc_matmul(score_active_psum, stationary=rms_ones, moving=kq_elem_bf16)
    score_active = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="score_active")
    nisa.tensor_copy(score_active, score_active_psum)

    # Load position scalar for masking
    position_ids_2d = position_ids.reshape((B, 1))
    pos_sb = sbm.alloc_stack((1, 1), nl.int32, name="pos_sb")
    nisa.dma_copy(dst=pos_sb, src=position_ids_2d[0:1, 0:1], dge_mode=nisa.dge_mode.hwdge)
    pos_f32 = sbm.alloc_stack((1, 1), nl.float32, name="pos_f32")
    nisa.tensor_copy(pos_f32, pos_sb)

    # Build partition index [PMAX, 1]
    par_index_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name="par_index_f32")
    nisa.iota(par_index_f32, pattern=[[1, 1]], offset=0, channel_multiplier=1)

    # K-cache hoisting + masking
    k_cache_tiles = []
    mask_tiles = []
    mask_gqa_tiles = []
    for s_t in nl.affine_range(NUM_S_TILES):
        k_raw = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name=f"k_raw_{s_t}")
        nisa.dma_copy(dst=k_raw, src=K_cache_2d[s_t * PMAX:(s_t + 1) * PMAX, :], dge_mode=nisa.dge_mode.hwdge)

        k_ct_psum = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum, name=f"k_ct_psum_{s_t}")
        nisa.nc_transpose(k_ct_psum, k_raw)
        k_ct = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name=f"k_ct_{s_t}")
        nisa.tensor_copy(k_ct, k_ct_psum)
        k_cache_tiles.append(k_ct)

        tile_start = s_t * PMAX

        neg_threshold = sbm.alloc_stack((1, 1), nl.float32, name=f"neg_threshold_{s_t}")
        nisa.tensor_scalar(neg_threshold, pos_f32,
                           op0=nl.multiply, operand0=-1.0,
                           op1=nl.add, operand1=float(tile_start))

        neg_thresh_psum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum, name=f"neg_thresh_psum_{s_t}")
        nisa.nc_transpose(neg_thresh_psum, neg_threshold.ap([[1, 1], [0, PMAX]], offset=0))
        neg_thresh_sb = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"neg_thresh_sb_{s_t}")
        nisa.tensor_copy(neg_thresh_sb, neg_thresh_psum)

        delta = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"delta_{s_t}")
        nisa.tensor_tensor(delta, par_index_f32, neg_thresh_sb, op=nl.add)

        delta_eps = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"delta_eps_{s_t}")
        nisa.tensor_scalar(delta_eps, delta, op0=nl.add, operand0=1.0)
        relu_delta = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"relu_delta_{s_t}")
        nisa.activation(relu_delta, op=nl.relu, data=delta_eps)

        clamped = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"clamped_{s_t}")
        nisa.tensor_scalar(clamped, relu_delta, op0=nl.minimum, operand0=1.0)

        mask_tile_f32 = sbm.alloc_stack((PMAX, 1), nl.float32, name=f"mask_tile_f32_{s_t}")
        nisa.tensor_scalar(mask_tile_f32, clamped, op0=nl.multiply, operand0=-1e9)
        mask_tiles.append(mask_tile_f32)

        mask_gqa_pre_psum_T = nl.ndarray((GQA_local, PMAX), dtype=nl.float32, buffer=nl.psum, name=f"mask_gqa_pre_psum_T_{s_t}")
        nisa.nc_transpose(mask_gqa_pre_psum_T, mask_tile_f32.ap([[1, PMAX], [0, GQA_local]], offset=0))
        mask_gqa_pre_sbuf_T = sbm.alloc_stack((GQA_local, PMAX), nl.float32, name=f"mask_gqa_pre_sbuf_T_{s_t}")
        nisa.tensor_copy(mask_gqa_pre_sbuf_T, mask_gqa_pre_psum_T)
        mask_gqa_pre_psum = nl.ndarray((PMAX, GQA_local), dtype=nl.float32, buffer=nl.psum, name=f"mask_gqa_pre_psum_{s_t}")
        nisa.nc_transpose(mask_gqa_pre_psum, mask_gqa_pre_sbuf_T)
        mask_gqa_pre = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name=f"mask_gqa_pre_{s_t}")
        nisa.tensor_copy(mask_gqa_pre, mask_gqa_pre_psum)
        mask_gqa_tiles.append(mask_gqa_pre)

    # V-cache hoisting
    v_cache_tiles = []
    for s_t in nl.affine_range(NUM_S_TILES):
        v_ct = sbm.alloc_stack((PMAX, PMAX), nl.bfloat16, name=f"v_ct_{s_t}")
        nisa.dma_copy(
            dst=v_ct,
            src=V_cache_2d.ap(pattern=[[d, PMAX], [1, d]], offset=s_t * PMAX * d),
            dge_mode=nisa.dge_mode.hwdge,
        )
        v_cache_tiles.append(v_ct)

    # --- Pass 1: find global max scalar ---
    global_max_g1 = sbm.alloc_stack((GQA_local, 1), nl.float32, name="global_max_g1")
    nisa.memset(global_max_g1, value=-1e9)

    score_act_T_psum = nl.zeros((GQA_local, PMAX), dtype=nl.float32, buffer=nl.psum, name="score_act_T_psum")
    nisa.nc_transpose(score_act_T_psum, score_active)
    score_act_T_sb = sbm.alloc_stack((GQA_local, PMAX), nl.float32, name="score_act_T_sb")
    nisa.tensor_copy(score_act_T_sb, score_act_T_psum)
    score_active_g1 = sbm.alloc_stack((GQA_local, 1), nl.float32, name="score_active_g1")
    nisa.tensor_reduce(dst=score_active_g1, op=nl.maximum, data=score_act_T_sb, axis=1)
    nisa.tensor_tensor(global_max_g1, global_max_g1, score_active_g1, op=nl.maximum)

    saved_scores = []

    for s_t in nl.affine_range(NUM_S_TILES):
        score_psum = nl.zeros((PMAX, GQA_local), dtype=nl.float32, buffer=nl.psum, name=f"score_psum_{s_t}")
        nisa.nc_matmul(score_psum, stationary=k_cache_tiles[s_t], moving=q_bf16)
        score_sb = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name=f"score_sb_{s_t}")
        nisa.tensor_copy(score_sb, score_psum)

        mask_gqa = mask_gqa_tiles[s_t]

        score_sb_masked = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name=f"score_sb_masked_{s_t}")
        nisa.tensor_tensor(score_sb_masked, score_sb, mask_gqa, op=nl.add)
        saved_scores.append(score_sb_masked)

        score_T_psum = nl.ndarray((GQA_local, PMAX), dtype=nl.float32, buffer=nl.psum, name=f"score_T_psum_{s_t}")
        nisa.nc_transpose(score_T_psum, score_sb_masked)
        score_T_sb = sbm.alloc_stack((GQA_local, PMAX), nl.float32, name=f"score_T_sb_{s_t}")
        nisa.tensor_copy(score_T_sb, score_T_psum)

        tile_max_vec = sbm.alloc_stack((GQA_local, 1), nl.float32, name=f"tile_max_vec_{s_t}")
        nisa.tensor_reduce(dst=tile_max_vec, op=nl.maximum, data=score_T_sb, axis=1)
        nisa.tensor_tensor(global_max_g1, global_max_g1, tile_max_vec, op=nl.maximum)

    neg_max_g1 = sbm.alloc_stack((GQA_local, 1), nl.float32, name="neg_max_g1")
    nisa.tensor_scalar(neg_max_g1, global_max_g1, op0=nl.multiply, operand0=-1.0)

    neg_max_psum = nl.ndarray((PMAX, GQA_local), dtype=nl.float32, buffer=nl.psum, name="neg_max_psum")
    nisa.nc_transpose(
        neg_max_psum,
        neg_max_g1.ap([[1, GQA_local], [0, PMAX]], offset=0),
    )
    neg_max = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="neg_max")
    nisa.tensor_copy(neg_max, neg_max_psum)

    # --- Pass 2: exp(score - global_max), accumulate V ---
    v_acc = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="v_acc")
    nisa.memset(v_acc, value=0.0)
    sum_acc = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="sum_acc")
    nisa.memset(sum_acc, value=0.0)

    for s_t in nl.affine_range(NUM_S_TILES):
        score2_shifted = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name=f"score2_shifted_{s_t}")
        nisa.tensor_tensor(score2_shifted, saved_scores[s_t], neg_max, op=nl.add)

        score2_exp = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name=f"score2_exp_{s_t}")
        nisa.activation(score2_exp, op=nl.exp, data=score2_shifted)

        score2_exp_bf16 = sbm.alloc_stack((PMAX, GQA_local), nl.bfloat16, name=f"score2_exp_bf16_{s_t}")
        nisa.tensor_copy(score2_exp_bf16, score2_exp)

        tile_sum_psum = nl.zeros((PMAX, GQA_local), dtype=nl.float32, buffer=nl.psum, name=f"tile_sum_psum_{s_t}")
        nisa.nc_matmul(tile_sum_psum, stationary=rms_ones, moving=score2_exp_bf16)
        tile_sum = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name=f"tile_sum_{s_t}")
        nisa.tensor_copy(tile_sum, tile_sum_psum)
        nisa.tensor_tensor(sum_acc, sum_acc, tile_sum, op=nl.add)

        v_weighted_psum = nl.zeros((PMAX, GQA_local), dtype=nl.float32, buffer=nl.psum, name=f"v_weighted_psum_{s_t}")
        nisa.nc_matmul(v_weighted_psum, stationary=v_cache_tiles[s_t], moving=score2_exp_bf16)
        v_weighted = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name=f"v_weighted_{s_t}")
        nisa.tensor_copy(v_weighted, v_weighted_psum)
        nisa.tensor_tensor(v_acc, v_acc, v_weighted, op=nl.add)

    # --- Active position contribution ---
    score_act_shifted = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="score_act_shifted")
    nisa.tensor_tensor(score_act_shifted, score_active, neg_max, op=nl.add)
    score_act_exp = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="score_act_exp")
    nisa.activation(score_act_exp, op=nl.exp, data=score_act_shifted)
    nisa.tensor_tensor(sum_acc, sum_acc, score_act_exp, op=nl.add)

    v_act_packed_psum_T = nl.ndarray((GQA_local, PMAX), dtype=nl.float32, buffer=nl.psum, name="v_act_packed_psum_T")
    nisa.nc_transpose(v_act_packed_psum_T, v_active.ap([[1, PMAX], [0, GQA_local]], offset=0))
    v_act_packed_sbuf_T = sbm.alloc_stack((GQA_local, PMAX), nl.float32, name="v_act_packed_sbuf_T")
    nisa.tensor_copy(v_act_packed_sbuf_T, v_act_packed_psum_T)
    v_act_packed_psum = nl.ndarray((PMAX, GQA_local), dtype=nl.float32, buffer=nl.psum, name="v_act_packed_psum")
    nisa.nc_transpose(v_act_packed_psum, v_act_packed_sbuf_T)
    v_act_packed = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="v_act_packed")
    nisa.tensor_copy(v_act_packed, v_act_packed_psum)

    v_act_weighted = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="v_act_weighted")
    nisa.tensor_tensor(v_act_weighted, v_act_packed, score_act_exp, op=nl.multiply)
    nisa.tensor_tensor(v_acc, v_acc, v_act_weighted, op=nl.add)

    # --- Normalize ---
    sum_safe = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="sum_safe")
    nisa.tensor_scalar(sum_safe, sum_acc, op0=nl.add, operand0=1e-9)
    rsqrt_sum = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="rsqrt_sum")
    nisa.activation(rsqrt_sum, op=nl.rsqrt, data=sum_safe)
    inv_sum = sbm.alloc_stack((PMAX, GQA_local), nl.float32, name="inv_sum")
    nisa.tensor_tensor(inv_sum, rsqrt_sum, rsqrt_sum, op=nl.multiply)

    attn_out = sbm.alloc_stack((PMAX, GQA_local), nl.bfloat16, name="attn_out")
    nisa.tensor_tensor(attn_out, v_acc, inv_sum, op=nl.multiply)

    # =========================================================================
    # FUSED OUTPUT PROJECTION — num_out_blocks blocks of F_MAX each
    # Pre-allocate 4 PSUMs max (handles n_prgs=1 with 4 blocks, n_prgs=2 with 2 blocks)
    # =========================================================================
    res_psum_0 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_0")
    res_psum_1 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_1")
    res_psum_2 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_2")
    res_psum_3 = nl.zeros((1, F_MAX), dtype=nl.float32, buffer=nl.psum, name="res_psum_3")
    for head in nl.affine_range(HQ_TP_CONST):
        nisa.nc_matmul(res_psum_0, stationary=attn_out[0:PMAX, head:head+1],
                       moving=wo_sbuf[head][0:PMAX, 0*F_MAX:1*F_MAX])
        nisa.nc_matmul(res_psum_1, stationary=attn_out[0:PMAX, head:head+1],
                       moving=wo_sbuf[head][0:PMAX, 1*F_MAX:2*F_MAX])
        if num_out_blocks > 2:
            nisa.nc_matmul(res_psum_2, stationary=attn_out[0:PMAX, head:head+1],
                           moving=wo_sbuf[head][0:PMAX, 2*F_MAX:3*F_MAX])
            nisa.nc_matmul(res_psum_3, stationary=attn_out[0:PMAX, head:head+1],
                           moving=wo_sbuf[head][0:PMAX, 3*F_MAX:4*F_MAX])

    # Assemble output into out_sb [H0, h1_shard].
    # The PSUM blocks contain attn_out_ref in linear order:
    #   out_flat_full[0, d] = attn_out_ref[0, d]
    # The natural SBUF scatter out_sb[p, c] = out_flat_full[0, c*H0+p] gives
    # WRONG ordering (column-major), but we need row-major (p*h1_shard+c).
    #
    # Fix: use nc_transpose to transpose out_flat_full viewed as [h1_shard, H0] →
    # [H0, h1_shard] PSUM, then copy to out_sb. This produces:
    #   out_sb_psum[p, c] = out_flat_reshaped[c, p] = out_flat_full[0, c*H0+p]
    # WAIT — that's still wrong. We need out_sb[p, c] = out_flat[0, p*h1_shard+c].
    # nc_transpose([h1_shard, H0]) → PSUM [H0, h1_shard]:
    #   result[p, c] = input[c, p] = out_flat_full[0, c*H0+p]
    # Still same column-major. Need the OPPOSITE.
    #
    # Correct approach: build out_flat_full in PERMUTED column-major order
    # so that out_flat_full[0, c*H0+p] = attn_out_ref[0, p*h1_shard+c].
    # Equivalently: load Wo in a transposed-column order.
    #
    # Simplest working approach: view out_flat_full as [H0, h1_shard] and
    # nc_transpose to get [h1_shard, H0] PSUM, then use it column-by-column.
    # Actually: treat out_flat_full [1, h_shard] as the row-major AT [H0, h1_shard]
    # matrix. SBUF [1, h_shard] layout: element at [0, p*h1_shard+c] is A[p, c].
    # The nc_transpose scatter reads it as [0, c*H0+p] which transposes it.
    # We need to FIRST rearrange out_flat_full so that position c*H0+p holds A[p,c].
    # This is equivalent to transposing A: compute out_flat_full_T where
    # out_flat_full_T[0, c*H0+p] = out_flat_full[0, p*h1_shard+c].
    # Then the scatter gives out_sb[p, c] = out_flat_full_T[0, c*H0+p] = A[p, c]. ✓
    #
    # To transpose out_flat_full: reshape [1, h_shard] → [H0, h1_shard],
    # nc_transpose → PSUM [h1_shard, H0], reshape back [1, h_shard].
    # But reshape [1, h_shard] → [H0, h1_shard] has the illegal partition step.
    #
    # Alternative: just use a different scatter — for each row p in static_range(H0),
    # copy out_flat_full[0, p*h1_shard : (p+1)*h1_shard] to out_sb[p, 0:h1_shard].
    # This requires DMA with a stride on the source, which static_range creates
    # a diagonal AP. Use affine_range with DGE vector_dynamic_offsets instead.
    h_shard = H0 * h1_shard

    out_flat_full = sbm.alloc_stack((1, h_shard), nl.bfloat16, name="attn_out_flat_full")

    # PSUM blocks → flat stack sub-slices (free-dim writes, legal with [1, h_shard] native alloc)
    out_tmp_0 = sbm.alloc_stack((1, F_MAX), nl.bfloat16, name="attn_out_tmp_0")
    nisa.tensor_copy(out_tmp_0, res_psum_0)
    nisa.tensor_copy(out_flat_full[0:1, 0*F_MAX:1*F_MAX], out_tmp_0)

    out_tmp_1 = sbm.alloc_stack((1, F_MAX), nl.bfloat16, name="attn_out_tmp_1")
    nisa.tensor_copy(out_tmp_1, res_psum_1)
    nisa.tensor_copy(out_flat_full[0:1, 1*F_MAX:2*F_MAX], out_tmp_1)

    if num_out_blocks > 2:
        out_tmp_2 = sbm.alloc_stack((1, F_MAX), nl.bfloat16, name="attn_out_tmp_2")
        nisa.tensor_copy(out_tmp_2, res_psum_2)
        nisa.tensor_copy(out_flat_full[0:1, 2*F_MAX:3*F_MAX], out_tmp_2)

        out_tmp_3 = sbm.alloc_stack((1, F_MAX), nl.bfloat16, name="attn_out_tmp_3")
        nisa.tensor_copy(out_tmp_3, res_psum_3)
        nisa.tensor_copy(out_flat_full[0:1, 3*F_MAX:4*F_MAX], out_tmp_3)

    # Scatter out_flat_full [1, h_shard] into out_sb [H0, h1_shard] with correct ordering.
    # We want out_sb[p, c] = out_flat_full[0, p*h1_shard+c] (row-major/partition-major).
    # Strategy: DMA out_flat_full to HBM, then load back via _load_input_to_sbuf pattern.
    # _load_input_to_sbuf produces dst[p, c] = src_hbm[p*h1_shard+c]. Correct.
    attn_tmp_hbm = nl.ndarray((1, 1, h_shard), dtype=nl.bfloat16,
                               buffer=nl.shared_hbm, name="attn_tmp_hbm")
    nisa.dma_copy(dst=attn_tmp_hbm.reshape((1, h_shard)), src=out_flat_full,
                  dge_mode=nisa.dge_mode.hwdge)
    if out_sb is None:
        out_sb = sbm.alloc_stack((H0, h1_shard), nl.bfloat16, name="attn_sharded_sb")
    _load_input_to_sbuf(out_sb, attn_tmp_hbm, BxS=1, H0=H0, H1=h1_shard, H1_shard=h1_shard, n_prgs=1)
    return out_sb


# ---------------------------------------------------------------------------
# _qwen3_moe_tkg_sbuf_out — modified MoE (returns SBUF shard)
# ---------------------------------------------------------------------------

def _qwen3_moe_tkg_sbuf_out(
    inp,              # [B=1, 1, H=2048] bf16
    gamma,            # [1, H=2048] bf16
    router_w,         # [H=2048, E=128] bf16
    gate_up_w,        # [E=128, H=2048, 2*I=384] bf16
    down_w,           # [E=128, I=192, H=2048] bf16
    prg_id,           # compile-time int 0 or 1
    h1_shard=H1_SHARD,  # dynamic: H1 // n_prgs
    inp_sb=None,
    out_sb=None,      # required: caller-provided fixed-address SBUF tensor [H0, BxS*h1_shard]
    sbm=None,         # required: BufferManager for alloc_stack
):
    """Returns moe_sharded_sb: [H0, BxS*h1_shard] SBUF — this rank's output column shard."""
    if inp_sb is not None:
        B = inp_sb.shape[1] // (H_FREE)  # infer B from inp_sb shape (_PMAX, H_FREE*T)
        T = B
        inp_dtype = inp_sb.dtype
    else:
        B = inp.shape[0]
        T = B
        inp_dtype = inp.dtype

    H_dim = _PMAX * H_FREE
    K = K_EXPERTS
    I = I_DIM
    H_shard = h1_shard * _PMAX   # 1024 when n_prgs=2, 2048 when n_prgs=1
    H_free_shard = h1_shard      # 8 when n_prgs=2, 16 when n_prgs=1
    I_tiles = I_TILES

    # -----------------------------------------------------------------------
    # Stage 1: RMSNorm
    # -----------------------------------------------------------------------
    if inp_sb is not None:
        # inp_sb layout from _sb2sb_gather/all_reduce_gather:
        #   inp_sb[p, lnc*H1_shard + h1_local] = feature(lnc * H0 * H1_shard + p * H1_shard + h1_local)
        # MoE downstream code expects rmsnorm_out[p, h_free] = feature(h_free * _PMAX + p).
        # Convert via HBM round-trip: store inp_sb → HBM using _store_output_to_hbm layout,
        # then reload using the standard DMA+nc_transpose path.
        n_prgs_local = H_FREE // h1_shard  # 2 when h1_shard=8, 1 when h1_shard=16
        h1_shard_local = h1_shard
        inp_tmp_hbm = nl.ndarray((T, 1, H_dim), dtype=inp_sb.dtype,
                                  buffer=nl.shared_hbm, name="moe_inp_tmp_hbm")
        _store_output_to_hbm(inp_tmp_hbm, inp_sb, T, _PMAX, H_FREE, h1_shard_local, n_prgs_local)
        inp_2d = inp_tmp_hbm.reshape((T, H_dim))
        inp_2d_hbm_reshaped = inp_2d.reshape((H_FREE * T, _PMAX))
        inp_flat_sb = sbm.alloc_stack((H_FREE * T, _PMAX), inp_sb.dtype, name="moe_inp_flat_sb")
        nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_reshaped, dge_mode=3)
        inp_trans_psum = nl.ndarray((_PMAX, H_FREE * T), dtype=inp_sb.dtype, buffer=nl.psum, name="moe_inp_trans_psum")
        nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)
        rmsnorm_out = sbm.alloc_stack((_PMAX, H_FREE * T), inp_sb.dtype, name="moe_rmsnorm_out")
        nisa.activation(rmsnorm_out[...], op=nl.copy, data=inp_trans_psum[...])
    else:
        inp_2d = inp.reshape((T, H_dim))
        inp_2d_hbm_reshaped = inp_2d.reshape((H_FREE * T, _PMAX))
        inp_flat_sb = sbm.alloc_stack((H_FREE * T, _PMAX), inp.dtype, name="moe_inp_flat_sb")
        nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_reshaped, dge_mode=3)

        inp_trans_psum = nl.ndarray((_PMAX, H_FREE * T), dtype=inp.dtype, buffer=nl.psum, name="moe_inp_trans_psum")
        nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)

        rmsnorm_out = sbm.alloc_stack((_PMAX, H_FREE * T), inp.dtype, name="moe_rmsnorm_out")
        nisa.activation(rmsnorm_out[...], op=nl.copy, data=inp_trans_psum[...])

    gamma_1d = gamma.reshape((H_dim,))
    gamma_1d_hbm_reshaped = gamma_1d.reshape((H_FREE, _PMAX))
    gamma_flat_sb = sbm.alloc_stack((H_FREE, _PMAX), gamma.dtype, name="moe_gamma_flat_sb")
    nisa.dma_copy(dst=gamma_flat_sb, src=gamma_1d_hbm_reshaped, dge_mode=3)
    gamma_trans_psum = nl.ndarray((_PMAX, H_FREE), dtype=gamma.dtype, buffer=nl.psum, name="moe_gamma_trans_psum")
    nisa.nc_transpose(dst=gamma_trans_psum, data=gamma_flat_sb)
    gamma_sb = sbm.alloc_stack((_PMAX, H_FREE), gamma.dtype, name="moe_gamma_sb")
    nisa.activation(gamma_sb[...], op=nl.copy, data=gamma_trans_psum[...])

    rmsnorm_sq = sbm.alloc_stack((_PMAX, H_FREE * T), nl.float32, name="moe_rmsnorm_sq")
    nisa.activation(rmsnorm_sq[...], op=nl.square, data=rmsnorm_out[...])

    rmsnorm_reduced = sbm.alloc_stack((_PMAX, T), nl.float32, name="moe_rmsnorm_reduced")
    nisa.tensor_reduce(rmsnorm_reduced[0:_PMAX, 0:T], nl.add, rmsnorm_sq[0:_PMAX, 0:H_FREE * T], axis=1)

    gamma_mult = sbm.alloc_stack((_PMAX, H_FREE * T), nl.float32, name="moe_gamma_mult")
    nisa.tensor_tensor(gamma_mult[...], rmsnorm_out[...], gamma_sb[...], nl.multiply)

    sum_reduced_sb = sbm.alloc_stack((1, T), nl.float32, name="moe_sum_reduced_sb")
    nisa.tensor_partition_reduce(dst=sum_reduced_sb[0:1, 0:T], data=rmsnorm_reduced[0:_PMAX, 0:T], op=nl.add)

    norm_sum_sb = sbm.alloc_stack((_PMAX, T), nl.float32, name="moe_norm_sum_sb")
    nisa.tensor_copy(dst=norm_sum_sb[0:1, 0:T], src=sum_reduced_sb[0:1, 0:T])
    for g in nl.static_range(4):
        nisa.nc_stream_shuffle(
            dst=norm_sum_sb[nl.ds(g * 32, 32), 0:T],
            src=norm_sum_sb[0:1, 0:T],
            shuffle_mask=[0] * 32,
        )

    eps_sb = sbm.alloc_stack((_PMAX, 1), nl.float32, name="moe_eps_sb")
    nisa.memset(eps_sb, value=1e-6)
    norm_factor_sb = sbm.alloc_stack((_PMAX, T), nl.float32, name="moe_norm_factor_sb")
    nisa.activation(
        norm_factor_sb[0:_PMAX, 0:T],
        op=nl.rsqrt,
        data=norm_sum_sb[0:_PMAX, 0:T],
        scale=1.0 / H_dim,
        bias=eps_sb[0:_PMAX, :],
    )

    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=2).broadcast(dim=2, size=H_FREE)
    rmsnorm_normed = sbm.alloc_stack((_PMAX, H_FREE * T), nl.float32, name="moe_rmsnorm_normed")
    nisa.tensor_tensor(rmsnorm_normed[...], gamma_mult[...], norm_factor_bcast.get_view(), nl.multiply)

    rmsnorm_normed_bf16 = sbm.alloc_stack((_PMAX, H_FREE * T), inp_dtype, name="moe_rmsnorm_normed_bf16")
    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=rmsnorm_normed[...])

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum, name="moe_logits_psum")
    router_w_wide_sb = sbm.alloc_stack((_PMAX, ROUTER_BATCH, E), inp_dtype, name="moe_router_w_wide_sb")

    for h_chunk in nl.affine_range(H_FREE // ROUTER_BATCH):
        nisa.dma_copy(
            dst=router_w_wide_sb,
            src=router_w.ap(
                pattern=[[E, _PMAX], [_PMAX * E, ROUTER_BATCH], [1, E]],
                offset=h_chunk * ROUTER_BATCH * _PMAX * E,
            ),
            dge_mode=3,
        )
        for h_sub in nl.static_range(ROUTER_BATCH):
            h1 = h_chunk * ROUTER_BATCH + h_sub
            nisa.nc_matmul(
                dst=logits_psum[0:T, 0:E],
                stationary=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                moving=router_w_wide_sb[0:_PMAX, h_sub, 0:E],
            )

    logits_sb = sbm.alloc_stack((T, E), nl.float32, name="moe_logits_sb")
    nisa.activation(logits_sb[0:T, 0:E], op=nl.copy, data=logits_psum[0:T, 0:E])

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8)
    # -----------------------------------------------------------------------
    max_logit = sbm.alloc_stack((T, 1), nl.float32, name="moe_max_logit")
    nisa.tensor_reduce(max_logit[0:T, 0:1], nl.maximum, logits_sb[0:T, 0:E], axis=1)

    centered = sbm.alloc_stack((T, E), nl.float32, name="moe_centered")
    nisa.tensor_scalar(
        centered[0:T, 0:E], data=logits_sb[0:T, 0:E],
        op0=nl.subtract, operand0=max_logit[0:T, 0:1],
    )

    exp_vals = sbm.alloc_stack((T, E), nl.float32, name="moe_exp_vals")
    nisa.activation(exp_vals[0:T, 0:E], op=nl.exp, data=centered[0:T, 0:E])

    sum_exp = sbm.alloc_stack((T, 1), nl.float32, name="moe_sum_exp")
    nisa.tensor_reduce(sum_exp[0:T, 0:1], nl.add, exp_vals[0:T, 0:E], axis=1)

    inv_sum_exp = sbm.alloc_stack((T, 1), nl.float32, name="moe_inv_sum_exp")
    nisa.activation(inv_sum_exp[0:T, 0:1], op=nl.reciprocal, data=sum_exp[0:T, 0:1])

    probs = sbm.alloc_stack((T, E), nl.float32, name="moe_probs")
    nisa.tensor_scalar(
        probs[0:T, 0:E], data=exp_vals[0:T, 0:E],
        op0=nl.multiply, operand0=inv_sum_exp[0:T, 0:1],
    )

    top8_vals = sbm.alloc_stack((T, K), nl.float32, name="moe_top8_vals")
    nisa.max8(dst=top8_vals[0:T, 0:K], src=probs[0:T, 0:E])

    top8_idx = sbm.alloc_stack((T, K), nl.uint32, name="moe_top8_idx")
    nisa.nc_find_index8(dst=top8_idx[0:T, 0:K], data=probs[0:T, 0:E], vals=top8_vals[0:T, 0:K])

    # -----------------------------------------------------------------------
    # Stage 4: Selective-Expert MLP — 2-Wave Expert Processing
    # -----------------------------------------------------------------------
    output_temp = sbm.alloc_stack((_PMAX, H_free_shard, T), nl.float32, name="moe_output_temp")

    for t in nl.static_range(T):

        gate_up_buf0 = sbm.alloc_stack((_PMAX, H_FREE, GU_FLAT), gate_up_w.dtype, name=f"moe_gate_up_buf0_{t}")
        gate_up_buf1 = sbm.alloc_stack((_PMAX, H_FREE, GU_FLAT), gate_up_w.dtype, name=f"moe_gate_up_buf1_{t}")
        gate_up_buf2 = sbm.alloc_stack((_PMAX, H_FREE, GU_FLAT), gate_up_w.dtype, name=f"moe_gate_up_buf2_{t}")
        gate_up_buf3 = sbm.alloc_stack((_PMAX, H_FREE, GU_FLAT), gate_up_w.dtype, name=f"moe_gate_up_buf3_{t}")
        gate_up_bufs = [gate_up_buf0, gate_up_buf1, gate_up_buf2, gate_up_buf3]

        down_full0_buf0 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, name=f"moe_down_full0_buf0_{t}")
        down_full0_buf1 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, name=f"moe_down_full0_buf1_{t}")
        down_full0_buf2 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, name=f"moe_down_full0_buf2_{t}")
        down_full0_buf3 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, name=f"moe_down_full0_buf3_{t}")
        down_full0_bufs = [down_full0_buf0, down_full0_buf1, down_full0_buf2, down_full0_buf3]

        down_full1_buf0 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, name=f"moe_down_full1_buf0_{t}")
        down_full1_buf1 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, name=f"moe_down_full1_buf1_{t}")
        down_full1_buf2 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, name=f"moe_down_full1_buf2_{t}")
        down_full1_buf3 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, name=f"moe_down_full1_buf3_{t}")
        down_full1_bufs = [down_full1_buf0, down_full1_buf1, down_full1_buf2, down_full1_buf3]

        for k_pad in range(4):
            nisa.memset(down_full1_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        gate_t1_128 = sbm.alloc_stack((_PMAX, H_FREE, I0), gate_up_w.dtype, name=f"moe_gate_t1_128_{t}")
        up_t1_128   = sbm.alloc_stack((_PMAX, H_FREE, I0), gate_up_w.dtype, name=f"moe_up_t1_128_{t}")
        nisa.memset(gate_t1_128[0:_PMAX, 0:H_FREE, nl.ds(I1, I1)], value=0.0)
        nisa.memset(up_t1_128[0:_PMAX, 0:H_FREE, nl.ds(I1, I1)],   value=0.0)

        # WAVE 0: Experts 0-3
        for k in nl.static_range(K_WAVE):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            nisa.dma_copy(
                dst=gate_up_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[GU_FLAT, _PMAX], [_PMAX * GU_FLAT, H_FREE], [1, GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full0_bufs[k],
                src=down_w.ap(
                    pattern=[[H_dim, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full1_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H_dim, I1], [1, H_shard]],
                    offset=I0 * H_dim + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        sum_topk = sbm.alloc_stack((T, 1), nl.float32, name=f"moe_sum_topk_{t}")
        nisa.tensor_reduce(sum_topk[0:T, 0:1], nl.add, top8_vals[0:T, 0:K], axis=1)

        inv_sum_topk = sbm.alloc_stack((T, 1), nl.float32, name=f"moe_inv_sum_topk_{t}")
        nisa.activation(inv_sum_topk[0:T, 0:1], op=nl.reciprocal, data=sum_topk[0:T, 0:1])

        norm_weights = sbm.alloc_stack((T, K), nl.float32, name=f"moe_norm_weights_{t}")
        nisa.tensor_scalar(
            norm_weights[0:T, 0:K], data=top8_vals[0:T, 0:K],
            op0=nl.multiply, operand0=inv_sum_topk[0:T, 0:1],
        )

        aff_bcast = sbm.alloc_stack((_PMAX, K), nl.float32, name=f"moe_aff_bcast_{t}")
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K], src=norm_weights[t:t + 1, 0:K])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        gate_up_psum = nl.ndarray((_PMAX, K_WAVE * 2 * I_tiles), dtype=nl.float32, buffer=nl.psum, name=f"moe_gate_up_psum_{t}")
        down_psum    = nl.ndarray((_PMAX, K_WAVE * H_free_shard), dtype=nl.float32, buffer=nl.psum, name=f"moe_down_psum_{t}")
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        for k in nl.static_range(K_WAVE):
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_FREE, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_FREE, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_FREE, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_FREE, nl.ds(I + I0, I1)],
            )

            for h1 in nl.affine_range(H_FREE):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                    else:
                        g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]
                        u_stat = up_t1_128[0:_PMAX, h1, 0:I0]
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                        stationary=g_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )

            silu_res = sbm.alloc_stack((_PMAX, I_tiles), nl.float32, name=f"moe_w0_silu_res_{t}_{k}")
            nisa.activation(silu_res, op=nl.silu, data=gate_up_psum[0:_PMAX, gu_base:gu_base + I_tiles])

            up_sb = sbm.alloc_stack((_PMAX, I_tiles), nl.float32, name=f"moe_w0_up_sb_{t}_{k}")
            nisa.activation(up_sb, op=nl.copy, data=gate_up_psum[0:_PMAX, gu_base + I_tiles:gu_base + 2 * I_tiles])

            inter_f32 = sbm.alloc_stack((_PMAX, I_tiles), nl.float32, name=f"moe_w0_inter_f32_{t}_{k}")
            nisa.tensor_tensor(inter_f32, silu_res, up_sb, nl.multiply)

            inter_bf16 = sbm.alloc_stack((_PMAX, I_tiles), inp_dtype, name=f"moe_w0_inter_bf16_{t}_{k}")
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            for h1_out in nl.affine_range(H_free_shard):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full0_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            down_result_sb = sbm.alloc_stack((_PMAX, H_free_shard), nl.float32, name=f"moe_w0_down_result_sb_{t}_{k}")
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free_shard],
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + H_free_shard],
            )
            down_result_scaled = sbm.alloc_stack((_PMAX, H_free_shard), nl.float32, name=f"moe_w0_down_result_scaled_{t}_{k}")
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],
            )

            if k == 0:
                nisa.tensor_copy(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    src=down_result_scaled[0:_PMAX, 0:H_free_shard],
                )
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data1=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data2=down_result_scaled[0:_PMAX, 0:H_free_shard],
                    op=nl.add,
                )

        # WAVE 1: Experts 4-7
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        for k in nl.static_range(K_WAVE):
            kk = k + 4
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + kk)

            nisa.dma_copy(
                dst=gate_up_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[GU_FLAT, _PMAX], [_PMAX * GU_FLAT, H_FREE], [1, GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full0_bufs[k],
                src=down_w.ap(
                    pattern=[[H_dim, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full1_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H_dim, I1], [1, H_shard]],
                    offset=I0 * H_dim + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        for k in nl.static_range(K_WAVE):
            kk = k + 4
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_FREE, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_FREE, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_FREE, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_FREE, nl.ds(I + I0, I1)],
            )

            for h1 in nl.affine_range(H_FREE):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                    else:
                        g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]
                        u_stat = up_t1_128[0:_PMAX, h1, 0:I0]
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                        stationary=g_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )

            silu_res = sbm.alloc_stack((_PMAX, I_tiles), nl.float32, name=f"moe_w1_silu_res_{t}_{k}")
            nisa.activation(silu_res, op=nl.silu, data=gate_up_psum[0:_PMAX, gu_base:gu_base + I_tiles])

            up_sb = sbm.alloc_stack((_PMAX, I_tiles), nl.float32, name=f"moe_w1_up_sb_{t}_{k}")
            nisa.activation(up_sb, op=nl.copy, data=gate_up_psum[0:_PMAX, gu_base + I_tiles:gu_base + 2 * I_tiles])

            inter_f32 = sbm.alloc_stack((_PMAX, I_tiles), nl.float32, name=f"moe_w1_inter_f32_{t}_{k}")
            nisa.tensor_tensor(inter_f32, silu_res, up_sb, nl.multiply)

            inter_bf16 = sbm.alloc_stack((_PMAX, I_tiles), inp_dtype, name=f"moe_w1_inter_bf16_{t}_{k}")
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            for h1_out in nl.affine_range(H_free_shard):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full0_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            down_result_sb = sbm.alloc_stack((_PMAX, H_free_shard), nl.float32, name=f"moe_w1_down_result_sb_{t}_{k}")
            nisa.activation(
                down_result_sb[0:_PMAX, 0:H_free_shard],
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + H_free_shard],
            )
            down_result_scaled = sbm.alloc_stack((_PMAX, H_free_shard), nl.float32, name=f"moe_w1_down_result_scaled_{t}_{k}")
            nisa.tensor_scalar(
                down_result_scaled,
                data=down_result_sb,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, kk:kk + 1],
            )

            nisa.tensor_tensor(
                dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                data1=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                data2=down_result_scaled[0:_PMAX, 0:H_free_shard],
                op=nl.add,
            )

    # -----------------------------------------------------------------------
    # Stage 5: Transpose fp32->bf16, return SBUF shard in correct row-major layout.
    # -----------------------------------------------------------------------
    # Stage 5a: nc_transpose output_temp (_PMAX, H_free_shard, T) → out_flat_sb (T, H_shard)
    # After transpose: out_flat_sb[0, h1*_PMAX+p] = output_temp[p, h1, 0] = moe_out[h1*_PMAX+p]
    # So out_flat_sb[0, d] = moe_out_ref[d] (linear order).
    out_flat_sb = sbm.alloc_stack((T, H_shard), inp_dtype, name="moe_out_flat_sb")
    for h1 in nl.static_range(H_free_shard):
        tp_psum = nl.ndarray((T, _PMAX), dtype=nl.float32, buffer=nl.psum, name=f"moe_tp_{h1}")
        nisa.nc_transpose(dst=tp_psum[0:T, 0:_PMAX], data=output_temp[0:_PMAX, h1, 0:T])
        nisa.activation(dst=out_flat_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)], op=nl.copy, data=tp_psum[0:T, 0:_PMAX])

    # Stage 5b: scatter out_flat_sb [T=1, H_shard] → out_sb [_PMAX, h1_shard].
    # out_flat_sb[0, d] = moe_out_ref[d]. We want out_sb[p, c] = moe_out_ref[p*h1_shard+c].
    # Use HBM round-trip: DMA out_flat_sb to HBM, load back via _load_input_to_sbuf.
    h1_shard_local = H_free_shard * T  # = h1_shard
    H_shard_local = _PMAX * h1_shard_local
    moe_tmp_hbm = nl.ndarray((1, 1, H_shard_local), dtype=inp_dtype,
                              buffer=nl.shared_hbm, name="moe_tmp_hbm")
    nisa.dma_copy(dst=moe_tmp_hbm.reshape((1, H_shard_local)), src=out_flat_sb,
                  dge_mode=nisa.dge_mode.hwdge)
    if out_sb is None:
        out_sb = sbm.alloc_stack((_PMAX, h1_shard_local), inp_dtype, name="moe_sharded_sb")
    _load_input_to_sbuf(out_sb, moe_tmp_hbm, BxS=1, H0=_PMAX, H1=h1_shard_local, H1_shard=h1_shard_local, n_prgs=1)
    return out_sb


# ---------------------------------------------------------------------------
# transformer_tkg_qwen — Qwen3 MoE single-layer megakernel
# ---------------------------------------------------------------------------

def transformer_tkg_qwen(
    X,              # [B=1, 1, H=2048] bf16
    cos,            # [B=1, 128] bf16
    sin,            # [B=1, 128] bf16
    position_ids,   # [B=1, 1] int32
    Wq,             # [1024, 2048] bf16
    Wk,             # [128, 2048] bf16
    Wv,             # [128, 2048] bf16
    Wo,             # [1024, 2048] bf16 — full, each core loads its H_SHARD half
    q_norm_w,       # [128] bf16
    k_norm_w,       # [128] bf16
    K_cache,        # [B, 1, S_ctx, 128] bf16
    V_cache,        # [B, 1, S_ctx, 128] bf16
    gamma_mlp,      # [1, 2048] bf16
    router_w,       # [2048, 128] bf16
    gate_up_w,      # [128, 2048, 384] bf16
    down_w,         # [128, 192, 2048] bf16
    replica_groups=None,
    eps=1e-6,
):
    B, S_tkg, _ = X.shape
    BxS = B * S_tkg
    dtype = X.dtype

    _, n_prgs, prg_id = get_verified_program_sharding_info("transformer_tkg_qwen", (0, 1), 2)
    h1_shard = H1 // n_prgs   # dynamic: 8 when n_prgs=2, 16 when n_prgs=1

    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    attn_in_sb       = nl.ndarray((H0, BxS * H1),       dtype=dtype, buffer=nl.sbuf, name="attn_in_sb")
    residual_attn_sb = nl.ndarray((H0, BxS * H1),       dtype=dtype, buffer=nl.sbuf, name="residual_attn_sb")
    mlp_in_sb        = nl.ndarray((H0, BxS * H1),       dtype=dtype, buffer=nl.sbuf, name="mlp_in_sb")
    residual_mlp_sb  = nl.ndarray((H0, BxS * H1),       dtype=dtype, buffer=nl.sbuf, name="residual_mlp_sb")
    output_sb        = nl.ndarray((H0, BxS * H1),       dtype=dtype, buffer=nl.sbuf, name="output_sb")

    # SBM for compute temporaries inside helper functions (auto_alloc=True)
    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("transformer_tkg_qwen"))
    sbm.set_auto_alloc(True)
    sbm.open_scope(name="transformer_layer")

    _load_input_to_sbuf(attn_in_sb, X, BxS, H0, H1, h1_shard, n_prgs)
    nisa.tensor_copy(dst=residual_attn_sb, src=attn_in_sb)

    # Pre-allocate KV outputs (HBM)
    k_rope_out = nl.ndarray((B, D_HEAD), dtype=dtype, buffer=nl.shared_hbm, name="tkg_k_rope_out")
    v_out      = nl.ndarray((B, D_HEAD), dtype=dtype, buffer=nl.shared_hbm, name="tkg_v_out")

    # --- Attention phase — attn_sharded_sb allocated inside by sbm.alloc_stack ---
    sbm.open_scope(name="attn_compute")
    attn_sharded_sb = _qwen3_attn_tkg_sbuf_out(
        X, Wq, Wk, Wv, Wo, q_norm_w, k_norm_w,
        K_cache, V_cache, cos, sin, position_ids,
        prg_id, k_rope_out, v_out, h1_shard=h1_shard,
        sbm=sbm,
    )
    # Note: scope remains open so attn_sharded_sb (alloc_stack) stays live for all_reduce

    # --- Attention all-reduce + gather ---
    if rg is not None:
        attn_full_sb = _sb2sb_all_reduce_gather(
            attn_sharded_sb, dtype, rg, prg_id, n_prgs, H0, H1, h1_shard, BxS,
            name_prefix="attn_",
        )[0]
    else:
        # No inter-chip all_reduce needed, but still need intra-chip LNC gather
        # so that attn_full_sb is [H0, BxS*H1] (not just [H0, h1_shard]).
        attn_full_sb = _sb2sb_gather(
            attn_sharded_sb, dtype, prg_id, n_prgs, H0, H1, h1_shard, BxS,
            name_prefix="attn_",
        )
    sbm.close_scope()  # frees attn temporaries after gather is done

    # --- Attention residual add ---
    nisa.tensor_tensor(dst=mlp_in_sb, data1=residual_attn_sb, data2=attn_full_sb, op=nl.add)
    nisa.tensor_copy(dst=residual_mlp_sb, src=mlp_in_sb)

    # --- MoE phase — moe_sharded_sb allocated inside by sbm.alloc_stack ---
    sbm.open_scope(name="moe_compute")
    moe_sharded_sb = _qwen3_moe_tkg_sbuf_out(
        inp=None,  # not used when inp_sb provided
        gamma=gamma_mlp, router_w=router_w, gate_up_w=gate_up_w, down_w=down_w,
        prg_id=prg_id, h1_shard=h1_shard,
        inp_sb=mlp_in_sb,
        sbm=sbm,
    )
    # Note: scope remains open so moe_sharded_sb stays live for all_reduce

    # --- MoE all-reduce + gather ---
    if rg is not None:
        moe_full_sb = _sb2sb_all_reduce_gather(
            moe_sharded_sb, dtype, rg, prg_id, n_prgs, H0, H1, h1_shard, BxS,
            name_prefix="moe_",
        )[0]
    else:
        # No inter-chip all_reduce, but still need intra-chip LNC gather.
        moe_full_sb = _sb2sb_gather(
            moe_sharded_sb, dtype, prg_id, n_prgs, H0, H1, h1_shard, BxS,
            name_prefix="moe_",
        )
    sbm.close_scope()  # frees moe temporaries after gather is done

    # --- MoE residual add ---
    nisa.tensor_tensor(dst=output_sb, data1=residual_mlp_sb, data2=moe_full_sb, op=nl.add)

    # --- Store final output to HBM ---
    output = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="tkg_output")
    _store_output_to_hbm(output, output_sb, BxS, H0, H1, h1_shard, n_prgs)
    if n_prgs > 1:
        nisa.core_barrier(data=output, cores=(0, 1))

    sbm.close_scope()  # transformer_layer
    return output
