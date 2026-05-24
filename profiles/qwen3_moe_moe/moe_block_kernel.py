"""Clean MoE-only NKI kernel mirroring the speculative megakernel's
per-layer MoE block, with NO debug HBM dumps.

This is a profiling-only file. It exactly mirrors the MoE block of
``transformer_qwen3_moe_speculative._multilayer_body`` (lines 444-557 in the
production file) for a single layer:

  1. Load X (the "residual") into SBUF in the LNC-aware shard-interleaved
     layout. (Boundary DMA — in the megakernel this is amortized over 48 layers.)
  2. ``rmsnorm_tkg`` (post_attention_layernorm).
  3. ``router_topk`` (softmax, top-K, L1-normalize).
  4. Per-shard ``tensor_copy`` to produce ``moe_in_pershard_sb``.
  5. ``moe_tkg`` (selective experts, SiLU, POST_SCALE).
  6. Per-shard ``tensor_copy`` to produce ``moe_sharded_sb`` (the layout
     the gather expects).
  7. Intra-LNC ``nisa.sendrecv`` gather across the 2 LNC sub-cores
     (the AR step of ``_sb2sb_all_reduce_gather`` is omitted — at single
     TP rank it is a no-op, and skipping it lets us run without setting
     up cross-rank collectives).
  8. Inverse-of-X-load store to canonical [B, S_tkg, H]. (Boundary DMA.)

Difference from the megakernel: the X-load (step 1) and the final HBM store
(step 8) are present here because device execution needs HBM inputs/outputs.
In the megakernel these are amortized over 48 layers (X loaded once, only the
final residual stored). Step 7 drops the cross-TP ``nccl.all_reduce`` (the only
caller of ``replica_groups``) because at single TP rank it is a mathematical
no-op; we keep the intra-LNC sendrecv so the gather still happens.

Returns ``(Y,)`` where Y is the canonical [B, S_tkg, H] MoE block output.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from megakernels.qwen3_moe.transformer_qwen3_moe_speculative import (
    H,
    H0,
    H1,
    H2,
    H1_SHARD,
    N_PRGS,
    EPS,
    E,
    TOP_K,
    I_PER_EXPERT,
    SBM_SIZE_BYTES,
    _store_shard_interleaved_sb_to_hbm,
)
from nki_kernels.moe import (
    XHBMLayout_T_H__1,
    XSBLayout_tp2013__1,
    moe_tkg,
    rmsnorm_tkg,
    router_topk,
)
from nkilib.core.utils.allocator import BufferManager, Logger
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.core.utils.tensor_view import TensorView


def _moe_block_body(
    X,             # [B, S_tkg, H]                bf16 HBM
    gpost,         # [1, H]                       bf16 HBM
    router_w,      # [H, E]                       bf16 HBM
    gate_up_w,     # [E, H, 2*I_PER_EXPERT]       bf16 HBM
    down_w,        # [E, I_PER_EXPERT, H]         bf16 HBM
):
    B, S_tkg, _ = X.shape
    dtype = X.dtype
    BxS = B * S_tkg
    T = BxS

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "moe_block_kernel", (0, 1), N_PRGS
    )

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("moe_block_kernel"))
    sbm.set_auto_alloc(True)

    # Load X into the SBUF "residual" in LNC-aware shard-interleaved layout
    # (same 4-level DMA pattern as transformer_qwen3_moe_speculative._multilayer_body).
    residual_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                             name="residual_sb")
    X_flat = X.reshape((BxS * H,))
    nisa.dma_copy(
        dst=residual_sb,
        src=X_flat.ap(
            pattern=[
                [H2, H0],
                [H0 * H2, N_PRGS],
                [1, H2],
                [H, BxS],
            ],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )

    # ===== Post-attention RMSNorm =====================================
    sbm.set_name_prefix("L0_moe_")
    sbm.set_auto_alloc(True)

    moe_in_sb = nl.ndarray((H0, BxS, H1), dtype=dtype, buffer=nl.sbuf,
                           name="L0_moe_in_sb")
    rmsnorm_tkg(
        input=residual_sb.reshape((H0, BxS, H1)),
        gamma=gpost,
        output=moe_in_sb,
        eps=EPS,
        hidden_actual=H,
        sbm=sbm,
    )

    # ===== Router top-K ===============================================
    router_logits_hbm = nl.ndarray((T, E), dtype=nl.float32,
                                   buffer=nl.shared_hbm,
                                   name="L0_router_logits_scratch")
    expert_index_sb = nl.ndarray((T, TOP_K), dtype=nl.uint32,
                                 buffer=nl.sbuf,
                                 name="L0_expert_index_sb")
    expert_affinities_sb = nl.ndarray((T, E), dtype=nl.float32,
                                      buffer=nl.sbuf,
                                      name="L0_expert_affinities_sb")
    router_outputs = router_topk(
        x=moe_in_sb,
        w=router_w,
        w_bias=None,
        router_logits=router_logits_hbm,
        expert_affinities=expert_affinities_sb,
        expert_index=expert_index_sb,
        act_fn=RouterActFnType.SOFTMAX,
        k=TOP_K,
        x_hbm_layout=XHBMLayout_T_H__1,
        x_sb_layout=XSBLayout_tp2013__1,
        router_pre_norm=False,
        norm_topk_prob=True,
        skip_store_router_logits=False,
        name_prefix="L0_moe_",
    )
    expert_affinities_sb = router_outputs[2]

    # ===== Selective expert MoE =======================================
    gate_up_w_view = gate_up_w.reshape((E, H, 2, I_PER_EXPERT))

    moe_in_pershard_sb = nl.ndarray((H0, BxS, H1_SHARD), dtype=dtype,
                                    buffer=nl.sbuf,
                                    name="L0_moe_in_pershard_sb")
    nisa.tensor_copy(
        dst=moe_in_pershard_sb,
        src=moe_in_sb[:, :, nl.ds(prg_id * H1_SHARD, H1_SHARD)],
    )

    moe_out_sb = moe_tkg(
        hidden_input=moe_in_pershard_sb,
        expert_gate_up_weights=gate_up_w_view,
        expert_down_weights=down_w,
        expert_affinities=expert_affinities_sb,
        expert_index=expert_index_sb,
        is_all_expert=False,
        expert_gate_up_bias=None,
        expert_down_bias=None,
        activation_fn=ActFnType.SiLU,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        name_prefix="L0_moe_",
        gate_clamp_upper_limit=None,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
        up_clamp_lower_limit=None,
        output_in_sbuf=True,
        output_dtype=dtype,
    )

    # ===== Per-shard copy → AR-gather ==================================
    moe_sharded_sb = nl.ndarray((H0, H1_SHARD * BxS), dtype=dtype,
                                buffer=nl.sbuf,
                                name="L0_moe_sharded_sb")
    for h1s in range(H1_SHARD):
        nisa.tensor_copy(
            dst=moe_sharded_sb[:, h1s * BxS : (h1s + 1) * BxS],
            src=moe_out_sb[:, :, h1s],
        )

    # Intra-LNC sendrecv gather (AR-free variant of _sb2sb_all_reduce_gather).
    # The cross-TP nccl.all_reduce that the megakernel uses is a no-op at
    # single TP rank, so we drop it; only the cross-core gather remains.
    gathered_sb = nl.ndarray((H0, H1 * BxS), dtype=dtype, buffer=nl.sbuf,
                              name="L0_moe_gathered_sb")
    f_shard = nl.ds(start=prg_id * BxS * H1_SHARD, size=BxS * H1_SHARD)
    nisa.tensor_copy(dst=gathered_sb[:, f_shard], src=moe_sharded_sb)
    if n_prgs > 1:
        other_lnc = 1 - prg_id
        f_other_shard = nl.ds(start=other_lnc * BxS * H1_SHARD, size=BxS * H1_SHARD)
        nisa.sendrecv(
            src=moe_sharded_sb,
            dst=gathered_sb[:, f_other_shard],
            send_to_rank=other_lnc,
            recv_from_rank=other_lnc,
            pipe_id=0,
        )

    # Reorder from (H1_outer, BxS_inner) → (BxS_outer, H1_inner) so the
    # inverse-of-X-load DMA pattern matches.
    moe_gathered_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                  name="L0_moe_gathered_reorder_sb")
    src_view = TensorView(gathered_sb).rearrange(
        ('h0', ('h1', 'bs')), ('h0', 'bs', 'h1'), {'h1': H1}
    )
    nisa.tensor_copy(dst=moe_gathered_sb.reshape((H0, BxS, H1)),
                     src=src_view.get_view())

    while sbm.heap:
        sbm.pop_heap()
    sbm.set_auto_alloc(True)

    # ===== Inverse-of-X-load store ====================================
    Y = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="Y_moe")
    _store_shard_interleaved_sb_to_hbm(Y, moe_gathered_sb, B, S_tkg, prg_id)
    if N_PRGS > 1:
        nisa.core_barrier(data=Y, cores=(0, 1))

    return (Y,)


@nki.jit
def moe_block_kernel(X, gpost, router_w, gate_up_w, down_w):
    return _moe_block_body(X, gpost, router_w, gate_up_w, down_w)
