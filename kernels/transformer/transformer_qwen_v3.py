"""
Qwen3-MoE transformer layer megakernel — SBUF residual path, LNC=2.

Single transformer layer: Attention (nkilib) + MoE (custom kernel_v29_sbm variant).
All activations stay in SBUF; both AllReduces operate on SBUF tensors.
No intermediate HBM roundtrips between attention output and final store.

Target: Trainium2 (trn2), TP=4, LNC=2 (n_prgs=2 per TP rank).
Entry point: transformer_qwen3_moe_tkg[2](...)
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.allocator import BufferManager, Logger
from nkilib.core.utils.common_types import QuantizationType
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg
from nkilib.experimental.transformer.transformer_tkg import (
    _load_input_to_sbuf,
    _store_output_to_hbm,
)
from nkilib.core.utils.tensor_view import TensorView
import nki.collectives as nccl
from kernels.moe_fused_tkg.versions.kernel_v30a_sbuf_io import _qwen3_moe_sbuf_in_sbuf_out

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


def _sb2sb_all_reduce_gather(
    sharded_sb, dtype, replica_group, prg_id: int, n_prgs: int, H0: int, H1: int, H1_shard: int, BxS: int
):
    """SB2SB all-reduce with local gather, returns (output_sb, sharded_AR_sb)."""
    sharded_AR_sb = nl.ndarray(sharded_sb.shape, dtype=dtype, buffer=nl.sbuf)
    nccl.all_reduce(dsts=[sharded_AR_sb], srcs=[sharded_sb], op=nl.add, replica_group=replica_group)

    gathered_sb = nl.ndarray((H0, H1 * BxS), dtype=dtype, buffer=nl.sbuf)
    f_shard = nl.ds(start=prg_id * BxS * H1_shard, size=BxS * H1_shard)
    nisa.tensor_copy(dst=gathered_sb[:, f_shard], src=sharded_AR_sb)

    # if n_prgs > 1:
    #     other_lnc = 1 - prg_id
    #     f_other_shard = nl.ds(start=other_lnc * BxS * H1_shard, size=BxS * H1_shard)
    #     nisa.sendrecv(
    #         src=sharded_AR_sb,
    #         dst=gathered_sb[:, f_other_shard],
    #         send_to_rank=other_lnc,
    #         recv_from_rank=other_lnc,
    #         pipe_id=0,
    #     )

    output_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
    src_view = TensorView(gathered_sb).rearrange(('h0', ('h1', 'bs')), ('h0', 'bs', 'h1'), {'h1': H1})
    nisa.tensor_copy(dst=output_sb.reshape((H0, BxS, H1)), src=src_view.get_view())

    return output_sb, sharded_AR_sb

# ---------------------------------------------------------------------------
# Hardware / model constants  (Qwen3-30B-A3B, TP=4, LNC=2)
# ---------------------------------------------------------------------------
H       = 2048          # hidden dim
H0      = 128           # partition dim (PMAX)
H1      = H // H0       # = 16   free-dim tiles
N_PRGS  = 2             # LNC=2
H1_SHARD = H1 // N_PRGS  # = 8   free-dim tiles per core
EPS     = 1e-6

SBM_SIZE_BYTES = 200 * 1024  # 200 KB — matches nkilib transformer_tkg


# ---------------------------------------------------------------------------
# MoE sub-kernel stub
# Implemented in kernels/moe_fused_tkg/versions/kernel_v29_sbm.py (in-progress).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------

def transformer_qwen3_moe_tkg(
    X,              # [B=1, S_tkg=1, H=2048]  bf16  HBM
    # --- Attention weights ---
    W_qkv,          # [(Hq_tp + 2*Hkv_tp)*d, H]  bf16  e.g. [2304, 2048]
    W_out,          # [Hq_tp*d, H]                bf16  e.g. [1024, 2048]
    gamma_attn,     # [1, H]  bf16 — pre-attention RMSNorm gamma
    q_norm_weight,  # [d=128] bf16 — per-head Q norm (Qwen3-specific)
    k_norm_weight,  # [d=128] bf16 — per-head K norm (Qwen3-specific)
    K_cache,        # [B, 1, S_prior, d]  bf16
    V_cache,        # [B, 1, S_prior, d]  bf16
    cos,            # [d//2, B, S_tkg]    bf16  RoPE cosine
    sin,            # [d//2, B, S_tkg]    bf16  RoPE sine
    mask_cache,     # attention mask for cached KV
    mask_active,    # attention mask for active tokens
    position_ids,   # [B, 1]  int32 — KV cache write position
    # --- MoE weights ---
    gamma_moe,      # [1, H]          bf16 — pre-MoE RMSNorm gamma
    router_w,       # [H, E=128]      bf16
    gate_up_w,      # [E, H, 384]     bf16
    down_w,         # [E, 192, H]     bf16
    # --- Collective config ---
    replica_groups=None,
):
    """
    Single Qwen3-MoE transformer layer for token generation (TKG).

    Data flow (all activations in SBUF after initial load):
      1. Load X from HBM → attn_in_sb                    [H0, BxS*H1]
      2. Copy residual   → residual_attn_sb               [H0, BxS*H1]
      3. attention_block_tkg (nkilib, Qwen3 settings)
         → attn_out_sb                                    [H0, H1_shard*BxS]
      4. AllReduce #1: _sb2sb_all_reduce_gather
         → attn_gathered_sb                               [H0, BxS*H1]
      5. Residual add    → moe_in_sb = attn_gathered_sb + residual_attn_sb
      6. Copy residual   → residual_moe_sb                [H0, BxS*H1]
      7. _qwen3_moe_sbuf_in_sbuf_out
         → moe_out_temp                                   [H0, H_free_shard, T]
      8. Reshape         → moe_sharded_sb                 [H0, H1_shard*BxS]
      9. AllReduce #2: _sb2sb_all_reduce_gather
         → moe_gathered_sb                                [H0, BxS*H1]
     10. Residual add    → output_sb = moe_gathered_sb + residual_moe_sb
     11. Store output_sb → HBM Y                          [B, S_tkg, H]
    """
    B, S_tkg, _ = X.shape
    dtype = X.dtype
    BxS = B * S_tkg      # = 1 for TKG
    T   = BxS            # alias used by MoE convention

    # LNC=2 setup — axis 0 is the LNC axis
    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "transformer_qwen3_moe_tkg", (0, 1), N_PRGS
    )

    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("transformer_qwen3_moe_tkg"))
    sbm.set_auto_alloc(False)

    # -----------------------------------------------------------------------
    # Step 1: Load input X from HBM to SBUF
    # Layout: [H0, BxS*H1]  where the H1 dimension is LNC-interleaved.
    # -----------------------------------------------------------------------
    attn_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                             name="attn_in_sb")
    _load_input_to_sbuf(attn_in_sb, X, BxS, H0, H1, H1_SHARD, n_prgs)

    # -----------------------------------------------------------------------
    # Step 2: Save residual before attention modifies the tensor
    # -----------------------------------------------------------------------
    residual_attn_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                   name="residual_attn_sb")
    nisa.tensor_copy(dst=residual_attn_sb, src=attn_in_sb)

    # -----------------------------------------------------------------------
    # Step 3: Attention block (nkilib)
    #
    # Qwen3-specific flags vs a vanilla transformer:
    #   rmsnorm_QK_pre_rope_enabled = True   — per-head RMSNorm on Q and K before RoPE
    #   rope_contiguous_layout      = True   — interleaved → contiguous RoPE layout
    #   K_cache_transposed          = True   — K cache stored transposed
    #   transposed_out              = True   — output in [H0, H1_shard, BxS] SBUF layout
    #   out_in_sb                   = True   — return in SBUF, skip HBM store
    #
    # Output shape: [H0=128, H1_shard=8, BxS=1]
    # -----------------------------------------------------------------------
    sbm.set_name_prefix("attn_")
    sbm.set_auto_alloc(True)

    X_sb = attn_in_sb.reshape((H0, BxS, H1))   # reshape for attention API

    attn_result = attention_block_tkg(
        X=X_sb,
        X_hidden_dim_actual=H,
        # Pre-attention RMSNorm
        rmsnorm_X_enabled=True,
        rmsnorm_X_eps=EPS,
        rmsnorm_X_gamma=gamma_attn,
        # QKV projection (no bias, no quantization)
        W_qkv=W_qkv,
        bias_qkv=None,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        # Qwen3 per-head Q/K RMSNorm before RoPE
        rmsnorm_QK_pre_rope_enabled=True,
        rmsnorm_QK_pre_rope_eps=EPS,
        rmsnorm_QK_pre_rope_W_Q=q_norm_weight,
        rmsnorm_QK_pre_rope_W_K=k_norm_weight,
        # RoPE
        cos=cos,
        sin=sin,
        rope_contiguous_layout=True,
        # No post-RoPE norm
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=EPS,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        # KV cache
        K_cache_transposed=True,
        active_blocks_table=None,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=mask_cache,
        sink=None,
        update_cache=position_ids is not None,
        kv_cache_update_idx=position_ids,
        # Output projection
        W_out=W_out,
        bias_out=None,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=True,   # output: [H0, H1_shard, BxS]
        out_in_sb=True,        # keep in SBUF
        sbm=sbm,
    )
    attn_out_sb = attn_result[0]   # [H0, H1_shard, BxS]

    # Free attention heap before AllReduce so SBUF space is reclaimed
    while sbm.heap:
        sbm.pop_heap()

    # -----------------------------------------------------------------------
    # Step 4: AllReduce #1 — sum attention outputs across TP ranks (SBUF→SBUF)
    # Input:  [H0, H1_shard*BxS]  (this core's shard)
    # Output: [H0, BxS*H1]        (full hidden dim, gathered)
    # -----------------------------------------------------------------------
    attn_sharded_sb = attn_out_sb.reshape((H0, H1_SHARD * BxS))
    attn_gathered_sb, _ = _sb2sb_all_reduce_gather(
        attn_sharded_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS
    )

    # -----------------------------------------------------------------------
    # Step 5: Residual add #1  — moe_in = attn_gathered + residual_attn
    # -----------------------------------------------------------------------
    moe_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                            name="moe_in_sb")
    nisa.tensor_tensor(dst=moe_in_sb, data1=residual_attn_sb, data2=attn_gathered_sb,
                       op=nl.add)

    # -----------------------------------------------------------------------
    # Step 6: Save residual before MoE
    # -----------------------------------------------------------------------
    residual_moe_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                  name="residual_moe_sb")
    nisa.tensor_copy(dst=residual_moe_sb, src=moe_in_sb)

    # -----------------------------------------------------------------------
    # Step 7: MoE block
    #
    # moe_in_sb is [H0=128, H1*BxS=16]  (T=1, so H_free*T = 16).
    # _qwen3_moe_sbuf_in_sbuf_out skips the input DMA load and feeds inp_sb
    # directly into the RMSNorm stage.
    #
    # Returns output_temp: [H0=128, H_free_shard=8, T=1]
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

    # Free MoE heap before AllReduce
    while sbm.heap:
        sbm.pop_heap()

    # -----------------------------------------------------------------------
    # Step 8 + 9: Reshape MoE output, AllReduce #2
    # moe_out_temp: [H0, H_free_shard, T] → reshape to [H0, H1_shard*BxS]
    # -----------------------------------------------------------------------
    moe_sharded_sb = moe_out_sb   # [H0=128, H1_SHARD*BxS=8] already — no reshape needed
    moe_gathered_sb, _ = _sb2sb_all_reduce_gather(
        moe_sharded_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS
    )

    # -----------------------------------------------------------------------
    # Step 10: Residual add #2  — output = moe_gathered + residual_moe
    # -----------------------------------------------------------------------
    output_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                            name="output_sb")
    nisa.tensor_tensor(dst=output_sb, data1=residual_moe_sb, data2=moe_gathered_sb,
                       op=nl.add)

    # -----------------------------------------------------------------------
    # Step 11: Store output_sb to HBM
    # Only LNC-0 drives the store; LNC-1 output is interleaved into the same
    # HBM buffer via _store_output_to_hbm's per-lnc DMA pattern.
    # -----------------------------------------------------------------------
    Y = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="Y")
    _store_output_to_hbm(Y, output_sb, BxS, H0, H1, H1_SHARD, n_prgs)

    if n_prgs > 1:
        nisa.core_barrier(data=Y, cores=(0, 1))

    return Y


# Wrap with @nki.jit at the module level so it can be called as:
#   transformer_qwen3_moe_tkg_jit[2](X, ...)
transformer_qwen3_moe_tkg_jit = nki.jit(transformer_qwen3_moe_tkg)
