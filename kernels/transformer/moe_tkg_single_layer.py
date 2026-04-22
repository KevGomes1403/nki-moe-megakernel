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
import nki.collectives as nccl

from nkilib.core.utils.allocator import BufferManager, Logger
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.experimental.transformer.transformer_tkg import _sb2sb_all_reduce_gather

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
    router_w,    # [H, E]          fp32 HBM — pre-transposed router weight
    gate_up_w,   # [E, H, 384]     bf16 HBM
    down_w,      # [E, 192, H]     bf16 HBM
    replica_groups=None,
):
    """Per-layer MoE for Qwen3 TKG.

    Body mirrors the MoE block of `transformer_qwen_multilayer.py` lines
    340-372: HBM → SBUF in plain-linear layout, MoE sub-kernel, SB2SB
    AR-gather, HBM store. RMSNorm (gpost) is fused inside the sub-kernel.
    Returns moe_out [B, S_tkg, H] bf16 HBM.
    """
    B, S_tkg, _ = X.shape
    BxS = B * S_tkg
    T   = BxS
    dtype = X.dtype

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "qwen3_moe_tkg_singlelayer", (0, 1), N_PRGS,
    )
    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

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

    # SB2SB AR + LNC gather → full [H0, BxS*H1] on each core.
    moe_gathered_sb, _ = _sb2sb_all_reduce_gather(
        moe_out_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS,
    )

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
