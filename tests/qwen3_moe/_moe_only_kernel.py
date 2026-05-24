"""Single-layer MoE-only NKI kernel that mirrors the speculative megakernel's
layer-0 MoE path EXACTLY.

This file is only used by ``test_moe_vs_hf.py``. It exists so the test can
drive the MoE portion of ``transformer_qwen3_moe_speculative`` in isolation
against a PyTorch reference, without involving attention / multilayer state.
The kernel is a strict subset of the production megakernel.

What this kernel does, in order (matches lines 771-891 of the production
megakernel ``_multilayer_body`` for layer 0):

1. Load X (the post-attention residual) into SBUF in the 4-level LNC-aware
   shard-interleaved layout.
2. Post-attention RMSNorm (``rmsnorm_tkg``) with ``gpost`` → ``moe_in_sb``.
3. ``router_topk`` (softmax → topK → L1-normalize over top-K probabilities).
4. ``moe_tkg`` (SiLU, POST_SCALE expert affinities, no clamping, no biases).
5. Per-shard copy of the moe_out_sb slice → ``moe_sharded_sb``.
6. ``_sb2sb_all_reduce_gather`` across LNC cores (sum-of-one all-reduce at
   single-rank, gather across cores via sendrecv).
7. Inverse-of-X-load DMA to produce a canonical [B, S_tkg, H] output.

No residual add. The output is the *pure MoE block output*, ready to be
compared against `post_attention_layernorm(X) → Qwen3MoeSparseMoeBlock.forward`
from the HF reference.

Returns ``(Y_moe,)`` — Y_moe is the canonical [B, S_tkg, H] MoE block output.
"""

import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl

# Reuse helpers and constants from the production megakernel.
from megakernels.qwen3_moe.transformer_qwen3_moe_speculative import (
    H,
    H0,
    H1,
    H2,
    H1_SHARD,
    N_PRGS,
    PMAX,
    EPS,
    D_HEAD,
    NUM_Q_HEADS_TP,
    NUM_KV_HEADS_TP,
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
from nkilib.experimental.transformer.transformer_tkg import _sb2sb_all_reduce_gather


def _moe_only_body(
    X,             # [B, S_tkg, H]                          bf16 HBM (post-attention residual)
    gpost,         # [1, H]                                 bf16 HBM (post_attention_layernorm gamma)
    router_w,     # [H, E]                                  bf16 HBM
    gate_up_w,    # [E, H, 2*I_PER_EXPERT]                  bf16 HBM
    down_w,       # [E, I_PER_EXPERT, H]                    bf16 HBM
    replica_groups=None,
):
    B, S_tkg, _ = X.shape
    dtype = X.dtype
    BxS = B * S_tkg
    T = BxS

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "moe_only_kernel", (0, 1), N_PRGS
    )

    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("moe_only_kernel"))
    sbm.set_auto_alloc(True)

    # ---- DEBUG: layer-0 dump HBM tensors (canonical [B, S, H] layout where
    # applicable). The test inspects these to localize per-stage drift. ----
    dbg_moe_in_dump_hbm = nl.ndarray((B, S_tkg, H), dtype=dtype,
                                      buffer=nl.shared_hbm, name="dbg_moe_in")
    dbg_router_logits_hbm = nl.ndarray((T, E), dtype=nl.float32,
                                        buffer=nl.shared_hbm,
                                        name="dbg_router_logits")
    dbg_expert_index_hbm = nl.ndarray((T, TOP_K), dtype=nl.uint32,
                                       buffer=nl.shared_hbm,
                                       name="dbg_expert_index")
    dbg_expert_affinities_hbm = nl.ndarray((T, E), dtype=nl.float32,
                                            buffer=nl.shared_hbm,
                                            name="dbg_expert_affinities")
    # Per-shard MoE output (pre-AR-gather). Each shard writes its
    # [H0, BxS, H1_SHARD] slice. After kernel run, the test can compare
    # the [prg_id, :, t, h1s] slice against a per-shard PyTorch reference.
    dbg_moe_out_pershard_hbm = nl.ndarray(
        (N_PRGS, H0, BxS, H1_SHARD), dtype=dtype,
        buffer=nl.shared_hbm, name="dbg_moe_out_pershard",
    )

    # ---- Load X into SBUF in shard-interleaved layout (same 4-level LNC-aware
    # pattern as the production megakernel and the attention test) ----
    residual_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                             name="residual_sb")
    X_flat = X.reshape((BxS * H,))
    nisa.dma_copy(
        dst=residual_sb,
        src=X_flat.ap(
            pattern=[
                [H2, H0],            # partition p
                [H0 * H2, N_PRGS],   # shard outer
                [1, H2],             # h2 inner
                [H, BxS],            # batch
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

    # ---- DEBUG: dump moe_in_sb (post-RMSNorm) via the same shard-interleaved
    # inverse store. moe_in_sb is laid out [H0, BxS, H1] where, in the
    # tp2013 layout, partition p at (t, shard*H2 + h2) holds the
    # post-RMSNorm value for token t at hidden index (shard*H0*H2 + p*H2 + h2).
    _store_shard_interleaved_sb_to_hbm(
        dbg_moe_in_dump_hbm,
        moe_in_sb.reshape((H0, BxS * H1)),
        B, S_tkg, prg_id,
    )

    # ===== Router topK ================================================
    # NOTE: instead of allocating a private router_logits scratch, we point
    # the router_topk's router_logits output at the debug-dump HBM buffer.
    # That way we capture the raw post-softmax logits without a second copy
    # while still satisfying NCC_IGCA090's "every mutable tensor needs at
    # least one store" requirement.
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
        router_logits=dbg_router_logits_hbm,
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

    # ---- DEBUG: dump router topK indices + final expert affinities ----
    nisa.dma_copy(dst=dbg_expert_index_hbm, src=expert_index_sb)
    nisa.dma_copy(dst=dbg_expert_affinities_hbm, src=expert_affinities_sb)

    # ===== Selective expert MoE =======================================
    gate_up_w_view = gate_up_w.reshape((E, H, 2, I_PER_EXPERT))

    # CRITICAL: moe_tkg's selective-expert path (T=1) expects ``hidden_input``
    # to be the PER-SHARD slice of shape ``[H0, T, H1_SHARD]``, NOT the full
    # ``[H0, T, H1]`` post-RMSNorm output. The matmul column index inside
    # ``gate_up_projection`` iterates over [0, H1_shard) on BOTH cores; if we
    # passed the full ``moe_in_sb`` both cores would read the SAME H value
    # subset (columns [0..H1_SHARD)), pairing core 1's weight rows
    # [H/2..H) with core 0's H values [0..H/2). This produces garbage MoE
    # output (matches our observed [Y] max_abs_err = 1.85e-1 with the
    # router/RMSNorm verified correct).
    #
    # Mirrors the canonical pattern in
    # ``nkilib/core/moe_block/moe_block_tkg.py:309-314``:
    #     expert_mlp_in = nl.ndarray((_pmax, T, H_free_shard), ..., buffer=sbuf)
    #     nisa.tensor_copy(dst=expert_mlp_in, src=rmsnorm_out[:, :, prg_id*H_free_shard:(prg_id+1)*H_free_shard])
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

    # ---- DEBUG: dump per-shard moe_out (pre-AR-gather). Each shard writes
    # its slice into [prg_id, :, :, :]. Shape: [H0, BxS, H1_SHARD]. ----
    nisa.dma_copy(
        dst=dbg_moe_out_pershard_hbm[prg_id, :, :, :],
        src=moe_out_sb[:, :, 0:H1_SHARD],
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

    moe_gathered_sb, _ = _sb2sb_all_reduce_gather(
        moe_sharded_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS
    )

    while sbm.heap:
        sbm.pop_heap()
    sbm.set_auto_alloc(True)

    # ===== Inverse-of-X-load store ====================================
    Y = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="Y_moe")
    _store_shard_interleaved_sb_to_hbm(Y, moe_gathered_sb, B, S_tkg, prg_id)
    if N_PRGS > 1:
        nisa.core_barrier(data=Y, cores=(0, 1))

    return (
        Y,
        dbg_moe_in_dump_hbm,
        dbg_router_logits_hbm,
        dbg_expert_index_hbm,
        dbg_expert_affinities_hbm,
        dbg_moe_out_pershard_hbm,
    )


@nki.jit
def moe_only_kernel(
    X, gpost, router_w, gate_up_w, down_w,
    replica_groups=None,
):
    return _moe_only_body(
        X, gpost, router_w, gate_up_w, down_w,
        replica_groups=replica_groups,
    )
