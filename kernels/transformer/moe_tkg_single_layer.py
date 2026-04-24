"""
Standalone single-layer Qwen3-MoE TKG MoE kernel (HBM in/out).

Wraps `_qwen3_moe_sbuf_in_sbuf_out_hoisted` (which is SBUF in/out and fuses
`post_attention_layernorm` internally) with:
  * HBM → SBUF load in plain-linear layout (matches `transformer_qwen_multilayer.py`)
  * SB2SB all-reduce + LNC gather
  * Per-column SBUF → HBM store

Intended for the NxDI-stock-attention + MoE-NKI-kernel A/B integration
(`qwen_nxd_attn_moe_kernel.py`). Layout deliberately matches the multilayer
megakernel so any numerical drift can be attributed to attention, not MoE.

Inputs are HBM bf16 tensors. The post-attention residual add is done in
PyTorch by the caller — the kernel returns the raw MoE output (pre-residual).
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.allocator import BufferManager, Logger
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.core.utils.tensor_view import TensorView

from kernels.moe_fused_tkg.versions.kernel_v30c_hoisted import (
    _qwen3_moe_sbuf_in_sbuf_out_hoisted,
)


H        = 2048
H0       = 128
H1       = H // H0     # 16
N_PRGS   = 2
H1_SHARD = H1 // N_PRGS  # 8
SBM_SIZE_BYTES = 200 * 1024


def qwen3_moe_tkg_singlelayer(
    X,           # [B, S_tkg, H]   bf16 HBM — post-attn-residual hidden state
    gpost,       # [1, H]          bf16 HBM — post_attention_layernorm gamma
    router_w,    # [H, E]          bf16 HBM — pre-transposed router weight
    gate_up_w,   # [E, H, 384]     bf16 HBM
    down_w,      # [E, 192, H]     bf16 HBM
    replica_groups=None,
):
    """Per-layer MoE for Qwen3 TKG.

    Body mirrors the MoE block of `transformer_qwen_multilayer.py` lines
    340-372: HBM → SBUF in plain-linear layout, MoE sub-kernel, within-LNC
    gather, HBM store. RMSNorm (gpost) is fused inside the sub-kernel.

    TP all-reduce is done at the XLA level by the caller
    (qwen_nxd_attn_moe_kernel.py); this kernel returns the per-rank pre-AR
    partial MoE output (LNC-gathered to full H).
    Returns moe_out [B, S_tkg, H] bf16 HBM.
    """
    B, S_tkg, _ = X.shape
    BxS = B * S_tkg
    T   = BxS
    dtype = X.dtype

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "qwen3_moe_tkg_singlelayer", (0, 1), N_PRGS,
    )
    # replica_groups kept in signature for API compatibility; AR now at XLA level.

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("qwen3_moe_tkg_singlelayer"))
    sbm.set_auto_alloc(True)

    # HBM → SBUF in plain-linear layout: inp_sb[p, b*H1+t] = X[b, 0, t*H0+p].
    # Matches multilayer kernel — do NOT use _load_input_to_sbuf
    # (channel-interleaved layout would scramble the in-kernel RMSNorm).
    inp_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf, name="moe_inp_sb")
    inp_load_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                              name="moe_inp_load_sb")
    X_flat = X.reshape((BxS * H,))
    nisa.dma_copy(
        dst=inp_load_sb,
        src=X_flat.ap(pattern=[[1, H0], [H0, H1], [H, BxS]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )
    nisa.tensor_copy(inp_sb, inp_load_sb)

    # MoE sub-kernel — gpost / router_w / gate_up_w / down_w loaded internally.
    moe_out_sb = _qwen3_moe_sbuf_in_sbuf_out_hoisted(
        inp_sb=inp_sb, dtype=dtype, T=T,
        gamma=gpost, router_w=router_w,
        gate_up_w=gate_up_w, down_w=down_w,
        sbm=sbm,
    )

    # TP AR is moved OUT to the XLA level (done in qwen_nxd_attn_moe_kernel.py).
    # Here we only do the within-LNC gather so each core gets the full H of the
    # per-rank partial MoE output (pre-AR).
    gathered_sb = nl.ndarray((H0, H1 * BxS), dtype=dtype, buffer=nl.sbuf,
                             name="moe_gathered_sb")
    f_shard = nl.ds(start=prg_id * BxS * H1_SHARD, size=BxS * H1_SHARD)
    nisa.tensor_copy(dst=gathered_sb[:, f_shard], src=moe_out_sb)
    if n_prgs > 1:
        other_lnc = 1 - prg_id
        f_other_shard = nl.ds(start=other_lnc * BxS * H1_SHARD, size=BxS * H1_SHARD)
        nisa.sendrecv(
            src=moe_out_sb,
            dst=gathered_sb[:, f_other_shard],
            send_to_rank=other_lnc,
            recv_from_rank=other_lnc,
            pipe_id=0,
        )
    moe_gathered_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                 name="moe_out_rearranged")
    src_view = TensorView(gathered_sb).rearrange(
        ('h0', ('h1', 'bs')), ('h0', 'bs', 'h1'), {'h1': H1},
    )
    nisa.tensor_copy(dst=moe_gathered_sb.reshape((H0, BxS, H1)), src=src_view.get_view())

    # SBUF → HBM in plain-linear layout (mirrors multilayer body lines 389-411):
    # Y[b, 0, t*H0+p] = moe_gathered_sb[p, b*H1+t]
    Y = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="Y")
    if prg_id == 0:
        Y_flat = Y.reshape((BxS, H))
        for b in nl.static_range(BxS):
            for t in nl.static_range(H1):
                col_psum = nl.ndarray((1, H0), dtype=dtype, buffer=nl.psum)
                nisa.nc_transpose(col_psum,
                                  moe_gathered_sb[0:H0, b*H1 + t : b*H1 + t + 1])
                col_sb = nl.ndarray((1, H0), dtype=dtype, buffer=nl.sbuf,
                                    name=f"out_col_b{b}_t{t}_sb")
                nisa.tensor_copy(col_sb, col_psum)
                nisa.dma_copy(
                    dst=Y_flat[b:b+1, t*H0:(t+1)*H0],
                    src=col_sb,
                    dge_mode=nisa.dge_mode.hwdge,
                )
    if n_prgs > 1:
        nisa.core_barrier(data=Y, cores=(0, 1))

    return Y


qwen3_moe_tkg_singlelayer_jit = nki.jit(qwen3_moe_tkg_singlelayer)


def qwen3_moe_tkg_singlelayer_debug(
    X,           # [B, S_tkg, H]   bf16 HBM — post-attn-residual hidden state
    gpost,       # [1, H]          bf16 HBM — post_attention_layernorm gamma
    router_w,    # [H, E]          bf16 HBM — pre-transposed router weight
    gate_up_w,   # [E, H, 384]     bf16 HBM
    down_w,      # [E, 192, H]     bf16 HBM
    replica_groups=None,
):
    """Debug variant of qwen3_moe_tkg_singlelayer.

    Returns (moe_out, rmsnorm_bf16) where rmsnorm_bf16 is the post_attention_layernorm
    output [B, S_tkg, H] bf16 stored in shared_hbm for torch.save capture.

    TP all-reduce is done at the XLA level by the caller
    (qwen_nxd_attn_moe_kernel.py); this kernel returns the per-rank pre-AR
    partial MoE output (LNC-gathered to full H).
    """
    B, S_tkg, _ = X.shape
    BxS = B * S_tkg
    T   = BxS
    dtype = X.dtype

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "qwen3_moe_tkg_singlelayer_debug", (0, 1), N_PRGS,
    )
    # replica_groups kept in signature for API compatibility; AR now at XLA level.

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("qwen3_moe_tkg_singlelayer_debug"))
    sbm.set_auto_alloc(True)

    inp_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf, name="moe_inp_sb")
    inp_load_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                              name="moe_inp_load_sb")
    X_flat = X.reshape((BxS * H,))
    nisa.dma_copy(
        dst=inp_load_sb,
        src=X_flat.ap(pattern=[[1, H0], [H0, H1], [H, BxS]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )
    nisa.tensor_copy(inp_sb, inp_load_sb)

    # Allocate rmsnorm debug output before sub-kernel call.
    rmsnorm_debug = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm,
                                name="rmsnorm_debug")

    moe_out_sb = _qwen3_moe_sbuf_in_sbuf_out_hoisted(
        inp_sb=inp_sb, dtype=dtype, T=T,
        gamma=gpost, router_w=router_w,
        gate_up_w=gate_up_w, down_w=down_w,
        sbm=sbm,
        debug_rmsnorm_out=rmsnorm_debug.reshape((T, H)),
    )

    # TP AR is moved OUT to the XLA level (done in qwen_nxd_attn_moe_kernel.py).
    # Here we only do the within-LNC gather so each core gets the full H of the
    # per-rank partial MoE output (pre-AR).
    gathered_sb = nl.ndarray((H0, H1 * BxS), dtype=dtype, buffer=nl.sbuf,
                             name="moe_gathered_sb")
    f_shard = nl.ds(start=prg_id * BxS * H1_SHARD, size=BxS * H1_SHARD)
    nisa.tensor_copy(dst=gathered_sb[:, f_shard], src=moe_out_sb)
    if n_prgs > 1:
        other_lnc = 1 - prg_id
        f_other_shard = nl.ds(start=other_lnc * BxS * H1_SHARD, size=BxS * H1_SHARD)
        nisa.sendrecv(
            src=moe_out_sb,
            dst=gathered_sb[:, f_other_shard],
            send_to_rank=other_lnc,
            recv_from_rank=other_lnc,
            pipe_id=0,
        )
    moe_gathered_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                 name="moe_out_rearranged")
    src_view = TensorView(gathered_sb).rearrange(
        ('h0', ('h1', 'bs')), ('h0', 'bs', 'h1'), {'h1': H1},
    )
    nisa.tensor_copy(dst=moe_gathered_sb.reshape((H0, BxS, H1)), src=src_view.get_view())

    Y = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="Y")
    if prg_id == 0:
        Y_flat = Y.reshape((BxS, H))
        for b in nl.static_range(BxS):
            for t in nl.static_range(H1):
                col_psum = nl.ndarray((1, H0), dtype=dtype, buffer=nl.psum)
                nisa.nc_transpose(col_psum,
                                  moe_gathered_sb[0:H0, b*H1 + t : b*H1 + t + 1])
                col_sb = nl.ndarray((1, H0), dtype=dtype, buffer=nl.sbuf,
                                    name=f"out_col_b{b}_t{t}_sb")
                nisa.tensor_copy(col_sb, col_psum)
                nisa.dma_copy(
                    dst=Y_flat[b:b+1, t*H0:(t+1)*H0],
                    src=col_sb,
                    dge_mode=nisa.dge_mode.hwdge,
                )
    if n_prgs > 1:
        nisa.core_barrier(data=Y, cores=(0, 1))

    return Y, rmsnorm_debug


qwen3_moe_tkg_singlelayer_debug_jit = nki.jit(qwen3_moe_tkg_singlelayer_debug)
