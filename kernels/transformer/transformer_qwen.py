"""
Qwen3-MoE transformer layer — v2: custom attention (v13bc, no sendrecv) + bare AllReduce.

Replaces nkilib attention_block_tkg (which contains nisa.sendrecv internally) with
qwen3_attn_tkg_fused_oproj_v13bc — a sendrecv-free sub-function where both LNC cores
produce the full [1, H_wo=2048] output. This allows a bare nccl.all_reduce (no gather,
no sendrecv) which avoids the ENC_ALG_MESH runtime assert.

Target: Trainium2 (trn2), TP=4, LNC=2.
Entry point: transformer_qwen3_moe_tkg_v2_jit[2](...)
"""

import nki
import nki.isa as nisa
import nki.language as nl
import nki.collectives as nccl
from nkilib.core.utils.allocator import BufferManager, Logger
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.experimental.transformer.transformer_tkg import (
    _load_input_to_sbuf,
    _store_output_to_hbm,
    _sb2sb_all_reduce_gather,
)
from kernels.attn_tkg.agents.v13bc_sbm_tiled import qwen3_attn_tkg_fused_oproj_v13bc
from kernels.moe_fused_tkg.versions.kernel_v30a_sbuf_io import _qwen3_moe_sbuf_in_sbuf_out

# ---------------------------------------------------------------------------
# Hardware / model constants (Qwen3-30B-A3B, TP=4, LNC=2)
# ---------------------------------------------------------------------------
H        = 2048
H0       = 128          # partition dim (PMAX)
H1       = H // H0      # 16 free-dim tiles
N_PRGS   = 2            # LNC=2
H1_SHARD = H1 // N_PRGS # 8 free-dim tiles per core
EPS      = 1e-6

SBM_SIZE_BYTES = 200 * 1024


def transformer_qwen3_moe_tkg_v2(
    X,              # [B=1, S_tkg=1, H=2048]  bf16  HBM  (normed — input layernorm already applied)
    X_residual,     # [B=1, S_tkg=1, H=2048]  bf16  HBM  (original, pre-norm — for residual adds)
    # --- Attention weights (split, not combined W_qkv) ---
    Wq,             # [Hq_tp*d, H]   e.g. [1024, 2048]  bf16
    Wk,             # [Hkv_tp*d, H]  e.g. [128,  2048]  bf16
    Wv,             # [Hkv_tp*d, H]  e.g. [128,  2048]  bf16
    Wo,             # [Hq_tp*d, H]   e.g. [1024, 2048]  bf16
    q_norm_weight,  # [d=128]  bf16
    k_norm_weight,  # [d=128]  bf16
    K_cache,        # [B, 1, S_prior, d]  bf16
    V_cache,        # [B, 1, S_prior, d]  bf16
    cos,            # [B, d]  bf16  — RoPE cosine pre-indexed at position
    sin,            # [B, d]  bf16  — RoPE sine  pre-indexed at position
    position_ids,   # [B, 1]  int32
    # --- MoE weights ---
    gamma_moe,      # [1, H]         bf16
    router_w,       # [H, E=128]     bf16
    gate_up_w,      # [E, H, 384]    bf16
    down_w,         # [E, 192, H]    bf16
    # --- Collective config ---
    replica_groups=None,
):
    """
    Data flow:
      1. Load X → attn_in_sb                 [H0, BxS*H1]
      2. Save residual_attn_sb
      3. qwen3_attn_tkg_fused_oproj_v13bc → out_sb  [H0, H1=128,16]  (tiled, both LNCs)
      4. AllReduce #1 on out_sb (full tensor, bare nccl.all_reduce, no sendrecv)
      5. close attn_outer scope
      6. Residual add → moe_in_sb            [H0, BxS*H1]
      7. Save residual_moe_sb
      8. _qwen3_moe_sbuf_in_sbuf_out → moe_out_sb
      9. AllReduce #2 on moe_out_sb
     10. Residual add → output_sb            [H0, BxS*H1]
     11. Store → HBM Y                       [B, S_tkg, H]
    """
    B, S_tkg, _ = X.shape
    dtype = X.dtype
    BxS = B * S_tkg
    T   = BxS

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "transformer_qwen3_moe_tkg_v2", (0, 1), N_PRGS
    )

    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("transformer_qwen3_moe_tkg_v2"))
    sbm.set_auto_alloc(False)

    # -----------------------------------------------------------------------
    # Step 1: Load X → SBUF [H0, BxS*H1]
    # -----------------------------------------------------------------------
    attn_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                             name="attn_in_sb")
    _load_input_to_sbuf(attn_in_sb, X, BxS, H0, H1, H1_SHARD, n_prgs)

    # -----------------------------------------------------------------------
    # Step 2: Load X_residual (original un-normed X) into residual_attn_sb
    # X is already normed (input_layernorm applied before calling this kernel),
    # so we must load the original X separately for the attention skip connection.
    # -----------------------------------------------------------------------
    residual_attn_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                   name="residual_attn_sb")
    _load_input_to_sbuf(residual_attn_sb, X_residual, BxS, H0, H1, H1_SHARD, n_prgs)

    # -----------------------------------------------------------------------
    # Step 3: Attention (v13bc — no sendrecv, no LNC sharding)
    #
    # v13bc expects:
    #   hidden_states [B, 1, H]  — reshape X from [B, S_tkg, H]
    #   cos/sin [B, d]           — pre-indexed at position
    # Returns:
    #   out_sb      [1, H_wo=2048] in SBUF (attn_outer scope still open)
    #   k_rope_out  [B, d] in shared_hbm (updated KV, unused here)
    #   v_out       [B, d] in shared_hbm
    # -----------------------------------------------------------------------
    sbm.set_auto_alloc(True)

    out_sb, _k_rope_out, _v_out = qwen3_attn_tkg_fused_oproj_v13bc(
        hidden_states=X,
        Wq=Wq,
        Wk=Wk,
        Wv=Wv,
        Wo=Wo,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        K_cache=K_cache,
        V_cache=V_cache,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        sbm=sbm,
    )
    # out_sb: [H0, BxS*H1] = [128, 16] bf16 in SBUF — attn_outer scope still open
    # Layout: out_sb[p, j] = linear_out[j*128 + p]  (standard TKG partition layout)

    # -----------------------------------------------------------------------
    # Step 4: AllReduce #1 — sum across TP ranks
    # Both LNCs hold the full partial-sum [128, 16] output from Wo matmul.
    # Bare nccl.all_reduce on full tensor — no sendrecv, no gather.
    # -----------------------------------------------------------------------
    attn_reduced_sb = nl.zeros((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                  name="attn_reduced_sb")
    nccl.all_reduce(
        dsts=[attn_reduced_sb], srcs=[out_sb],
        op=nl.add, replica_group=rg,
    )

    # Close attn_outer scope — frees all sbm allocs from attention block
    sbm.close_scope()
    sbm.set_auto_alloc(False)

    # attn_reduced_sb is already in TKG layout [H0, BxS*H1] — no scatter needed.

    # -----------------------------------------------------------------------
    # Step 5+6: Residual add #1 + save MoE residual
    # -----------------------------------------------------------------------
    moe_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                            name="moe_in_sb")
    nisa.tensor_tensor(dst=moe_in_sb, data1=residual_attn_sb,
                       data2=attn_reduced_sb, op=nl.add)

    residual_moe_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                  name="residual_moe_sb")
    nisa.tensor_copy(dst=residual_moe_sb, src=moe_in_sb)

    # -----------------------------------------------------------------------
    # Step 7: MoE block
    # -----------------------------------------------------------------------
    sbm.set_name_prefix("moe_")
    sbm.set_auto_alloc(True)

    moe_out_sb = _qwen3_moe_sbuf_in_sbuf_out(
        inp_sb=moe_in_sb,
        dtype=dtype,
        T=T,
        gamma=gamma_moe,
        router_w=router_w,
        gate_up_w=gate_up_w,
        down_w=down_w,
        sbm=sbm,
    )

    while sbm.heap:
        sbm.pop_heap()
    sbm.set_auto_alloc(False)

    # -----------------------------------------------------------------------
    # Step 8: AllReduce #2 (TP) + LNC gather
    # _sb2sb_all_reduce_gather is the intended consumer of _qwen3_moe_sbuf_in_sbuf_out:
    # it AllReduces across TP ranks and gathers both LNC shards via sendrecv,
    # returning the full [H0, BxS*H1] tensor on both cores.
    # -----------------------------------------------------------------------
    moe_gathered_sb, _ = _sb2sb_all_reduce_gather(
        moe_out_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS
    )

    # -----------------------------------------------------------------------
    # Step 9: Residual add #2 + store
    # Both cores hold the same full moe_gathered_sb — no race on Y.
    # -----------------------------------------------------------------------
    output_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                            name="output_sb")
    nisa.tensor_tensor(dst=output_sb, data1=residual_moe_sb, data2=moe_gathered_sb, op=nl.add)

    Y = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="Y")
    _store_output_to_hbm(Y, output_sb, BxS, H0, H1, H1_SHARD, n_prgs)

    if n_prgs > 1:
        nisa.core_barrier(data=Y, cores=(0, 1))

    return Y


transformer_qwen3_moe_tkg_v2_jit = nki.jit(transformer_qwen3_moe_tkg_v2)
