"""Unit test for the gpt-oss attention kernel (attention_block_tkg).

Validates the kernel against a PyTorch reference modelled directly on
transformers/models/gpt_oss/modular_gpt_oss.py:eager_attention_forward.
Covers both full attention and sliding-window attention (SWA), at
multiple position_ids, and verifies attention output AND every slot of
the updated K/V caches.

Simulates rank 0 of TP=8 (gpt-oss-20b shard): q_heads_per_rank=8,
kv_heads_per_rank=1.
"""

import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
# SBUF-mode wrapper invokes nccl.all_reduce with replica_groups=([0],) (single
# rank) — this is a no-op AR but the runtime still requires every visible core
# to be either in a replica group or hidden from the runtime. Pin to core 0
# only. (HBM-mode test doesn't use collectives so it doesn't need this, but
# pinning is harmless for both.)
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ["NEURON_CC_FLAGS"] = (
    os.environ.get("NEURON_CC_FLAGS", "")
    + " --lnc=1 --auto-cast=none --model-type=transformer -O1 "
    + "--enable-saturate-infinity --enable-mixed-precision-accumulation "
    + "--internal-enable-dge-levels vector_dynamic_offsets"
)

import sys
sys.path.insert(0, "/home/ubuntu/nki-moe")

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl

from nki_kernels.attention.attention_block_tkg import attention_block_tkg
from nkilib.core.utils.common_types import QuantizationType
from nkilib.core.utils.tensor_view import TensorView
from nkilib.experimental.transformer.transformer_tkg import _sb2sb_all_reduce_gather


# ----------------------------------------------------------------------------
# Config — per-rank slice of gpt-oss-20b with TP=8
# ----------------------------------------------------------------------------
B               = 1
S_TKG           = 1
H_ACTUAL        = 2880               # hidden_size
H_PAD           = 3072               # padded to multiple of 128 (kernel req)
D_HEAD          = 64
Q_HEADS         = 8                  # 64 total / 8 ranks
KV_HEADS        = 1                  # 8 total / 8 ranks
GQA             = Q_HEADS // KV_HEADS
S_MAX_CTX_FULL  = 256                # cache size for full-attention layers
S_MAX_CTX_SWA   = 128                # cache size for SWA layers (== sliding_window)
SLIDING_WINDOW  = 128
EPS             = 1e-5

DTYPE = torch.bfloat16

COMPILER_ARGS = (
    "--enable-saturate-infinity "
    "--enable-mixed-precision-accumulation "
    "--model-type transformer "
    "-O1 "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-max-instruction-limit=15000000"
)


# ----------------------------------------------------------------------------
# PyTorch reference — mirrors transformers' GptOssAttention.forward +
# eager_attention_forward, with input_layernorm folded in (the kernel
# fuses RMSNorm internally via rmsnorm_X_enabled=True).
# ----------------------------------------------------------------------------
def gpt_oss_attn_reference(
    X,                # [B, S_tkg, H_actual]            fp
    Wq, bq,           # [Q*d, H_actual], [Q*d]
    Wk, bk,           # [KV*d, H_actual], [KV*d]
    Wv, bv,           # [KV*d, H_actual], [KV*d]
    Wo, bo,           # [Q*d, H_actual], [H_actual]
    gpre,             # [H_actual]                       input_layernorm gamma
    sinks,            # [Q]                              learned sinks
    K_cache, V_cache, # [B, KV, S_max, d]
    kv_idx,           # [B, 1] int32
    position_ids,     # [B, 1] int32
    cos_at_pos,       # [d/2, B, S_tkg]
    sin_at_pos,       # [d/2, B, S_tkg]
    sliding_window=None,
    use_sinks=True,
):
    """Returns (output [B, S_tkg, H_actual], K_new, V_new) all fp32."""
    X = X.float()
    Wq, bq = Wq.float(), bq.float()
    Wk, bk = Wk.float(), bk.float()
    Wv, bv = Wv.float(), bv.float()
    Wo, bo = Wo.float(), bo.float()
    gpre   = gpre.float()
    sinks  = sinks.float()
    K_cache, V_cache = K_cache.float(), V_cache.float()
    cos_at_pos, sin_at_pos = cos_at_pos.float(), sin_at_pos.float()
    sinks_arg = sinks  # already float()

    B_, S_tkg_, _ = X.shape
    S_max = K_cache.shape[2]

    # input_layernorm (RMSNorm)
    var = X.pow(2).mean(-1, keepdim=True)
    Xn = X * torch.rsqrt(var + EPS) * gpre

    # QKV projection (with bias — gpt-oss has attention_bias=True)
    Q = F.linear(Xn, Wq, bq).view(B_, S_tkg_, Q_HEADS,  D_HEAD).transpose(1, 2)
    K = F.linear(Xn, Wk, bk).view(B_, S_tkg_, KV_HEADS, D_HEAD).transpose(1, 2)
    V = F.linear(Xn, Wv, bv).view(B_, S_tkg_, KV_HEADS, D_HEAD).transpose(1, 2)

    # RoPE — contiguous-half layout (matches HF gpt-oss _apply_rotary_emb).
    # cos/sin: [d/2, B, S_tkg] → broadcast as [B, 1, S_tkg, d/2].
    cos_b = cos_at_pos.permute(1, 2, 0).unsqueeze(1)
    sin_b = sin_at_pos.permute(1, 2, 0).unsqueeze(1)
    def rope(x):
        x1, x2 = x[..., :D_HEAD//2], x[..., D_HEAD//2:]
        return torch.cat([x1 * cos_b - x2 * sin_b,
                          x2 * cos_b + x1 * sin_b], dim=-1)
    Q = rope(Q)
    K = rope(K)

    # KERNEL CONTRACT: attention_tkg attends over S_max slots laid out as
    #   [K_cache[0], K_cache[1], ..., K_cache[S_max-2], K_active]
    # i.e. the K/V cache is read as-is (un-updated) for slots 0..S_max-2, and the
    # newly-projected K_active is spliced in at the LAST slot (slot S_max-1).
    # The cache update happens AFTER attention compute, so the cache passed to
    # attention_tkg is "stale" at slot kv_idx during attention.
    #
    # To match the kernel exactly we (a) write the new K/V into K_new at the
    # last slot (mirroring k_active's position in the kernel's layout), and
    # (b) build the mask as [cache_mask | active=1] over S_max slots.
    K_new = K_cache.clone()
    V_new = V_cache.clone()
    for b in range(B_):
        # The kernel scatters new K/V into K_cache[kv_idx] only at the END of
        # the kernel call (after attention). So this `K_new` represents the
        # updated cache AS RETURNED BY THE KERNEL, not what attention saw.
        idx = int(kv_idx[b, 0])
        K_new[b, :, idx:idx+S_tkg_, :] = K[b]
        V_new[b, :, idx:idx+S_tkg_, :] = V[b]

    # K_attn / V_attn = the layout attention_tkg actually computes over.
    # NB: independent of kv_idx — the active slot is always S_max-1.
    K_attn = K_cache.clone()
    V_attn = V_cache.clone()
    K_attn[:, :, S_max - S_tkg_:, :] = K  # last s_active slots = K_active
    V_attn[:, :, S_max - S_tkg_:, :] = V

    # GQA: repeat KV heads to match Q heads
    K_full = K_attn.repeat_interleave(GQA, dim=1)
    V_full = V_attn.repeat_interleave(GQA, dim=1)

    # Build additive mask matching the kernel's layout:
    #   slots 0..S_max-2 → past cache positions (treat random pre-fill as
    #     past tokens at logical positions 0..S_max-2).
    #   slot S_max-1 → active token (= the current token; attends to itself).
    pos = int(position_ids[0, 0])
    add_mask = torch.full((B_, Q_HEADS, S_tkg_, S_max), float("-inf"))
    # Cache portion: slots 0..S_max-2 represent past tokens at positions 0..S_max-2
    for i in range(S_max - 1):
        valid = (i < pos)  # strict: past tokens only, current token is in active slot
        if sliding_window is not None:
            valid = valid and (pos - i < sliding_window)
        if valid:
            add_mask[:, :, :, i] = 0.0
    # Active slot (current token) — always valid
    add_mask[:, :, :, S_max - 1] = 0.0

    # Attention with sinks (matches eager_attention_forward exactly)
    scale = 1.0 / math.sqrt(D_HEAD)
    scores = torch.matmul(Q, K_full.transpose(-1, -2)) * scale + add_mask
    if use_sinks:
        sinks_b = sinks.reshape(1, Q_HEADS, 1, 1).expand(B_, -1, S_tkg_, -1)
        combined = torch.cat([scores, sinks_b], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined, dim=-1)[..., :-1]
    else:
        probs = F.softmax(scores - scores.max(dim=-1, keepdim=True).values, dim=-1)
    attn = torch.matmul(probs, V_full)                      # [B, q, S_tkg, d]
    attn = attn.transpose(1, 2).reshape(B_, S_tkg_, Q_HEADS * D_HEAD).contiguous()

    # O projection (with bias)
    out = F.linear(attn, Wo, bo)
    return out, K_new, V_new


# ----------------------------------------------------------------------------
# Kernel call — wraps attention_block_tkg with our gpt-oss config.
# ----------------------------------------------------------------------------
def run_kernel(
    X, Wqkv, bqkv, Wo, bo, gpre, sinks,
    K_cache, V_cache, mask, kv_idx, cos, sin,
    use_sinks: bool = True,
    skip_oproj: bool = False,
    use_bias: bool = True,
):
    """Call attention_block_tkg directly. Returns (Y, K_out, V_out) on CPU."""
    device = xm.xla_device()
    to_dev = lambda t: t.to(device).contiguous() if t is not None else None

    Y, K_out, V_out = attention_block_tkg[1](
        # input
        X=to_dev(X),
        X_hidden_dim_actual=H_ACTUAL,
        # rmsnorm X (input_layernorm)
        rmsnorm_X_enabled=True,
        rmsnorm_X_eps=EPS,
        rmsnorm_X_gamma=to_dev(gpre.reshape(1, H_PAD)),
        # QKV (fused weights, with bias)
        W_qkv=to_dev(Wqkv),
        bias_qkv=to_dev(bqkv) if use_bias else None,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        # No QK pre-norm in gpt-oss
        rmsnorm_QK_pre_rope_enabled=False,
        rmsnorm_QK_pre_rope_eps=EPS,
        rmsnorm_QK_pre_rope_W_Q=None,
        rmsnorm_QK_pre_rope_W_K=None,
        # RoPE (contiguous-half layout)
        cos=to_dev(cos),
        sin=to_dev(sin),
        rope_contiguous_layout=True,
        # No QK post-norm in gpt-oss
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=EPS,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        # Attention
        K_cache_transposed=False,
        active_blocks_table=None,
        K_cache=to_dev(K_cache),
        V_cache=to_dev(V_cache),
        attention_mask=to_dev(mask),
        sink=to_dev(sinks.reshape(Q_HEADS, 1)) if use_sinks else None,
        # KV cache update
        update_cache=True,
        kv_cache_update_idx=to_dev(kv_idx),
        # O projection
        W_out=None if skip_oproj else to_dev(Wo),
        bias_out=None if (skip_oproj or not use_bias)
                 else to_dev(bo.reshape(1, H_PAD)),
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        out_in_sb=False,
    )
    xm.mark_step()
    return Y.cpu().float(), K_out.cpu().float(), V_out.cpu().float()


# ----------------------------------------------------------------------------
# SBUF-mode kernel wrapper — mirrors the megakernel's calling convention:
#   * X loaded from HBM into residual_sb (H0, BxS*H1) via P-stride H1 pattern
#     [[H1, H0], [1, H1], [H, BxS]] (XSBLayout_tp102__0).
#   * X passed to attention_block_tkg as SBUF tensor with shape (H0, BxS, H1),
#     transposed_out=True, out_in_sb=True.
#   * Output (H0, H1*BxS) goes through _sb2sb_all_reduce_gather (replica_groups
#     = single-rank → AR is a no-op) → (H0, BxS*H1).
#   * Stored back to HBM via inverse pattern, yielding (B, S_tkg, H).
# This is the EXACT path the gpt-oss megakernel uses internally.
# ----------------------------------------------------------------------------
H0_KERNEL = 128
H1_KERNEL = H_PAD // H0_KERNEL    # 3072 / 128 = 24


def _build_sbuf_mode_kernel():
    """Build a kernel that wraps attention_block_tkg in SBUF+transposed_out mode."""

    def attention_block_tkg_sbuf_mode(
        X,                  # [B, S_tkg, H_PAD]              bf16 HBM
        W_qkv,              # [H_PAD, (Q+2KV)*d_head]        bf16 HBM
        bias_qkv,           # [1, (Q+2KV)*d_head]            bf16 HBM
        W_out,              # [Q*d_head, H_PAD]              bf16 HBM
        bias_out,           # [1, H_PAD]                     bf16 HBM
        rmsnorm_X_gamma,    # [1, H_PAD]                     bf16 HBM
        sink,               # [Q_HEADS, 1]                   bf16 HBM
        K_cache,            # [B, KV, S_max, d_head]         bf16 HBM (mutated)
        V_cache,            # [B, KV, S_max, d_head]         bf16 HBM (mutated)
        attention_mask,     # [S_max, B, Q_HEADS, S_tkg]     bf16 HBM
        kv_cache_update_idx, # [B, 1]                        int32 HBM
        cos,                # [d_head//2, B, S_tkg]          bf16 HBM
        sin,                # [d_head//2, B, S_tkg]          bf16 HBM
    ):
        """Single-layer attention block in the megakernel's SBUF-input path.

        Returns (Y_hbm[B, S_tkg, H_PAD], K_cache_post, V_cache_post).
        """
        B_, S_tkg_, H_ = X.shape
        dtype = X.dtype
        BxS = B_ * S_tkg_
        H0 = H0_KERNEL
        H1 = H1_KERNEL
        H1_SHARD = H1   # LNC=1, n_prgs=1 → no shard

        # Single-rank replica group → AR is a no-op (acts as identity).
        rg = nccl.ReplicaGroup(([0],))
        prg_id = 0
        n_prgs = 1

        # ---- Load X into residual_sb (H0, BxS*H1) using megakernel pattern ----
        residual_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                 name="residual_sb")
        residual_load_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                       name="residual_load_sb")
        X_flat = X.reshape((BxS * H_,))
        nisa.dma_copy(
            dst=residual_load_sb,
            src=X_flat.ap(pattern=[[H1, H0], [1, H1], [H_, BxS]], offset=0),
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.tensor_copy(residual_sb, residual_load_sb)

        # ---- Reshape and call attention_block_tkg in SBUF input + transposed_out mode ----
        X_sb = residual_sb.reshape((H0, BxS, H1))

        attn_result = attention_block_tkg(
            X=X_sb,
            X_hidden_dim_actual=H_ACTUAL,
            rmsnorm_X_enabled=True,
            rmsnorm_X_eps=EPS,
            rmsnorm_X_gamma=rmsnorm_X_gamma,
            W_qkv=W_qkv,
            bias_qkv=bias_qkv,
            quantization_type_qkv=QuantizationType.NONE,
            weight_dequant_scale_qkv=None,
            input_dequant_scale_qkv=None,
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_pre_rope_eps=EPS,
            rmsnorm_QK_pre_rope_W_Q=None,
            rmsnorm_QK_pre_rope_W_K=None,
            cos=cos,
            sin=sin,
            rope_contiguous_layout=True,
            rmsnorm_QK_post_rope_enabled=False,
            rmsnorm_QK_post_rope_eps=EPS,
            rmsnorm_QK_post_rope_W_Q=None,
            rmsnorm_QK_post_rope_W_K=None,
            K_cache_transposed=False,
            active_blocks_table=None,
            K_cache=K_cache,
            V_cache=V_cache,
            attention_mask=attention_mask,
            sink=sink,
            update_cache=True,
            kv_cache_update_idx=kv_cache_update_idx,
            W_out=W_out,
            bias_out=bias_out,
            quantization_type_out=QuantizationType.NONE,
            weight_dequant_scale_out=None,
            input_dequant_scale_out=None,
            transposed_out=True,
            out_in_sb=True,
        )
        # attn_result = (output_sb, K_cache_post, V_cache_post)
        # output_sb shape: [H0, H1_SHARD, BxS]   (per docstring line 266)
        attn_kernel_out_sb = attn_result[0]
        K_post = attn_result[1]
        V_post = attn_result[2]

        # Reshape sharded output to [H0, H1_SHARD * BxS] for the gather.
        attn_sharded_sb = attn_kernel_out_sb.reshape((H0, H1_SHARD * BxS))

        # ---- "AR-gather" at single-rank ----
        # _sb2sb_all_reduce_gather does (a) nccl.all_reduce, then (b) a layout
        # rearrange (h0, h1, bs) → (h0, bs, h1) so the result lands in
        # XSBLayout_tp102__0 (same layout as residual_sb) for the residual add.
        # At TP=1 the AllReduce is mathematically a no-op, but the Neuron
        # runtime's collective state machine does NOT support a single-rank
        # replica group (NRT asserts alg == ENC_ALG_MESH at runtime). So we
        # replicate ONLY step (b): the layout-transforming gather, which is
        # the load-bearing piece for verifying the SBUF+transposed_out path
        # produces a correctly laid-out residual contribution. The AllReduce
        # is exercised separately at multi-rank.
        attn_gathered_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                      name="attn_gathered_sb")
        # attn_sharded_sb is logically [H0, H1, BxS] (sharded), reinterpret as
        # [H0, H1*BxS] flat → rearrange to [H0, BxS, H1] (XSBLayout_tp102__0).
        src_view = TensorView(attn_sharded_sb).rearrange(
            ('h0', ('h1', 'bs')), ('h0', 'bs', 'h1'), {'h1': H1}
        )
        nisa.tensor_copy(
            dst=attn_gathered_sb.reshape((H0, BxS, H1)),
            src=src_view.get_view(),
        )
        # Suppress unused warning for `rg` / `_sb2sb_all_reduce_gather` import.
        _ = rg, prg_id, n_prgs

        # ---- Store back to HBM via inverse pattern ----
        Y = nl.ndarray((B_, S_tkg_, H_), dtype=dtype, buffer=nl.shared_hbm, name="Y")
        Y_flat = Y.reshape((BxS * H_,))
        nisa.dma_copy(
            dst=Y_flat.ap(pattern=[[H1, H0], [1, H1], [H_, BxS]], offset=0),
            src=attn_gathered_sb,
            dge_mode=nisa.dge_mode.hwdge,
        )

        return Y, K_post, V_post

    return attention_block_tkg_sbuf_mode


_sbuf_mode_kernel = nki.jit(_build_sbuf_mode_kernel())


def run_kernel_sbuf(
    X, Wqkv, bqkv, Wo, bo, gpre, sinks,
    K_cache, V_cache, mask, kv_idx, cos, sin,
    use_sinks: bool = True,
    skip_oproj: bool = False,
    use_bias: bool = True,
):
    """Call the SBUF-mode wrapper kernel. Returns (Y, K_out, V_out) on CPU."""
    if skip_oproj or not use_sinks or not use_bias:
        # The wrapper hard-wires Wo/sink/bias as required HBM tensors. Keep
        # the SBUF-mode test focused on the megakernel-equivalent path.
        raise ValueError(
            "SBUF-mode wrapper requires skip_oproj=False, use_sinks=True, use_bias=True"
        )
    device = xm.xla_device()
    to_dev = lambda t: t.to(device).contiguous() if t is not None else None

    Y, K_out, V_out = _sbuf_mode_kernel(
        X=to_dev(X),
        W_qkv=to_dev(Wqkv),
        bias_qkv=to_dev(bqkv),
        W_out=to_dev(Wo),
        bias_out=to_dev(bo.reshape(1, H_PAD)),
        rmsnorm_X_gamma=to_dev(gpre.reshape(1, H_PAD)),
        sink=to_dev(sinks.reshape(Q_HEADS, 1)),
        K_cache=to_dev(K_cache),
        V_cache=to_dev(V_cache),
        attention_mask=to_dev(mask),
        kv_cache_update_idx=to_dev(kv_idx),
        cos=to_dev(cos),
        sin=to_dev(sin),
    )
    xm.mark_step()
    return Y.cpu().float(), K_out.cpu().float(), V_out.cpu().float()


# ----------------------------------------------------------------------------
# Test harness
# ----------------------------------------------------------------------------
def make_inputs(seed, S_max_ctx, position):
    """Generate randomized weights, hidden state, KV cache, mask, RoPE tables."""
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: (torch.randn(*s, generator=g, dtype=torch.float32) * 0.02).to(DTYPE)
    rn_unit = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32).to(DTYPE)

    # Hidden state (small magnitude ~ post-residual scale)
    X_actual = rn_unit(B, S_TKG, H_ACTUAL) * 0.5
    # Pad to H_PAD (zero-padded tail; kernel uses X_hidden_dim_actual=H_ACTUAL)
    X = torch.zeros(B, S_TKG, H_PAD, dtype=DTYPE)
    X[..., :H_ACTUAL] = X_actual

    # Weights — F.linear convention is [out, in]. HF gpt-oss:
    #   q/k/v_proj: nn.Linear(H, *_d)   → weight [Q*d or KV*d, H_actual]
    #   o_proj:    nn.Linear(Q*d, H)    → weight [H_actual, Q*d]
    Wq = rn(Q_HEADS  * D_HEAD, H_ACTUAL); bq = rn(Q_HEADS  * D_HEAD)
    Wk = rn(KV_HEADS * D_HEAD, H_ACTUAL); bk = rn(KV_HEADS * D_HEAD)
    Wv = rn(KV_HEADS * D_HEAD, H_ACTUAL); bv = rn(KV_HEADS * D_HEAD)
    Wo = rn(H_ACTUAL, Q_HEADS * D_HEAD);  bo = rn(H_ACTUAL)

    # Fused QKV (HF concat-output dim): [(Q+2*KV)*d, H_actual]
    Wqkv_hf = torch.cat([Wq, Wk, Wv], dim=0)
    bqkv    = torch.cat([bq, bk, bv], dim=0)

    # ---- Kernel layouts ----
    # W_qkv kernel shape: [H, (Q+2*KV)*d] — transpose HF + pad H side.
    Wqkv_T = Wqkv_hf.t().contiguous()                        # [H_actual, I]
    Wqkv_kernel = torch.zeros(H_PAD, Wqkv_T.shape[1], dtype=DTYPE)
    Wqkv_kernel[:H_ACTUAL] = Wqkv_T

    # W_out kernel shape: [Q*d, H] — transpose HF + pad H side.
    Wo_T = Wo.t().contiguous()                               # [Q*d, H_actual]
    Wo_kernel = torch.zeros(Q_HEADS * D_HEAD, H_PAD, dtype=DTYPE)
    Wo_kernel[:, :H_ACTUAL] = Wo_T

    bo_kernel = torch.zeros(H_PAD, dtype=DTYPE); bo_kernel[:H_ACTUAL] = bo
    gpre_pad  = torch.zeros(H_PAD, dtype=DTYPE)
    gpre_pad[:H_ACTUAL] = (
        torch.randn(H_ACTUAL, generator=g, dtype=torch.float32) * 0.05 + 1.0
    ).to(DTYPE)

    bqkv_kernel = bqkv.reshape(1, -1).contiguous()           # [1, I]

    # Sinks
    sinks = rn_unit(Q_HEADS) * 0.1                    # [Q]

    # KV cache pre-fill (random) — verify both written slot AND unchanged slots
    K_cache = rn_unit(B, KV_HEADS, S_max_ctx, D_HEAD) * 0.5
    V_cache = rn_unit(B, KV_HEADS, S_max_ctx, D_HEAD) * 0.5

    # cos/sin: random unit-magnitude RoPE table at this position
    theta = torch.randn(D_HEAD // 2, B, S_TKG, generator=g, dtype=torch.float32)
    cos = torch.cos(theta).to(DTYPE)
    sin = torch.sin(theta).to(DTYPE)

    # position_ids and kv_cache_update_idx
    position_ids = torch.tensor([[position]], dtype=torch.int32)
    kv_idx       = torch.tensor([[position % S_max_ctx]], dtype=torch.int32)

    return dict(
        X=X, X_actual=X_actual,
        Wq=Wq, bq=bq, Wk=Wk, bk=bk, Wv=Wv, bv=bv, Wo=Wo, bo=bo,
        Wqkv_kernel=Wqkv_kernel, bqkv_kernel=bqkv_kernel,
        Wo_kernel=Wo_kernel, bo_kernel=bo_kernel,
        gpre=gpre_pad, gpre_actual=gpre_pad[:H_ACTUAL],
        sinks=sinks,
        K_cache=K_cache, V_cache=V_cache,
        cos=cos, sin=sin,
        position_ids=position_ids, kv_idx=kv_idx,
    )


def build_mask(position, S_max_ctx, sliding_window=None):
    """Multiplicative mask: 1 = valid, 0 = blocked. Shape [S_max_ctx, B, Q_HEADS, S_TKG].

    Matches the kernel contract: attention_tkg sees K layout
        [K_cache[0..S_max_ctx-2], K_active]
    where K_active is spliced at slot S_max_ctx-1 (NOT at kv_idx — the cache
    update happens after attention compute). So:
      slots 0..S_max_ctx-2  → past cache positions (treat as logical positions
                              0..S_max_ctx-2; valid iff < position and within SWA).
      slot S_max_ctx-1      → active token (current; always valid, modulo SWA).

    Multiplicative semantics: kernel uses 0/1 (uint8 in SBUF), applied via
    tensor_copy_predicated where mask=1 keeps qk and mask=0 leaves -inf.
    """
    mask = torch.zeros((S_max_ctx, B, Q_HEADS, S_TKG), dtype=torch.float32)
    # Cache portion: past tokens at slot i ↔ logical position i, i in [0, S_max_ctx-1).
    # Strict `i < position` (the current token lives in the active slot, not the cache).
    for i in range(S_max_ctx - 1):
        valid = (i < position)
        if sliding_window is not None:
            valid = valid and (position - i < sliding_window)
        if valid:
            mask[i] = 1.0
    # Active slot: current token (position - position = 0 < sw, always valid)
    mask[S_max_ctx - 1] = 1.0
    return mask.to(DTYPE)


def run_case(name, position, sliding_window, S_max_ctx, seed=42, atol=5e-2, rtol=1e-2,
             use_sinks: bool = True, mode: str = "hbm"):
    """Run one validation case.

    mode:
      "hbm"             — original path (X in HBM, transposed_out=False, out_in_sb=False).
      "sbuf_transposed" — megakernel path (X loaded into SBUF via XSBLayout_tp102__0,
                          transposed_out=True, out_in_sb=True, then sb2sb AR-gather +
                          inverse HBM store).
    """
    assert mode in ("hbm", "sbuf_transposed"), f"unknown mode: {mode}"
    print(f"\n=== {name} | pos={position} | sw={sliding_window} | S_max={S_max_ctx} | sinks={use_sinks} | mode={mode} ===")
    inp = make_inputs(seed=seed, S_max_ctx=S_max_ctx, position=position)
    mask = build_mask(position, S_max_ctx, sliding_window=sliding_window)

    # PyTorch reference
    ref_out, ref_K, ref_V = gpt_oss_attn_reference(
        X=inp["X_actual"],
        Wq=inp["Wq"], bq=inp["bq"],
        Wk=inp["Wk"], bk=inp["bk"],
        Wv=inp["Wv"], bv=inp["bv"],
        Wo=inp["Wo"], bo=inp["bo"],
        gpre=inp["gpre_actual"], sinks=inp["sinks"],
        K_cache=inp["K_cache"], V_cache=inp["V_cache"],
        kv_idx=inp["kv_idx"], position_ids=inp["position_ids"],
        cos_at_pos=inp["cos"], sin_at_pos=inp["sin"],
        sliding_window=sliding_window,
        use_sinks=use_sinks,
    )
    print(f"  ref out range : [{ref_out.min().item():.4f}, {ref_out.max().item():.4f}]")

    # Kernel
    if mode == "hbm":
        K_out, K_K, K_V = run_kernel(
            X=inp["X"],
            Wqkv=inp["Wqkv_kernel"], bqkv=inp["bqkv_kernel"],
            Wo=inp["Wo_kernel"], bo=inp["bo_kernel"],
            gpre=inp["gpre"], sinks=inp["sinks"],
            K_cache=inp["K_cache"], V_cache=inp["V_cache"],
            mask=mask, kv_idx=inp["kv_idx"],
            cos=inp["cos"], sin=inp["sin"],
            use_sinks=use_sinks,
        )
    else:  # sbuf_transposed
        K_out, K_K, K_V = run_kernel_sbuf(
            X=inp["X"],
            Wqkv=inp["Wqkv_kernel"], bqkv=inp["bqkv_kernel"],
            Wo=inp["Wo_kernel"], bo=inp["bo_kernel"],
            gpre=inp["gpre"], sinks=inp["sinks"],
            K_cache=inp["K_cache"], V_cache=inp["V_cache"],
            mask=mask, kv_idx=inp["kv_idx"],
            cos=inp["cos"], sin=inp["sin"],
            use_sinks=use_sinks,
        )
    # Output is [B*S_tkg, H_PAD]; trim padded tail and reshape to [B, S_tkg, H_actual]
    K_out_actual = K_out.reshape(B, S_TKG, H_PAD)[..., :H_ACTUAL]
    print(f"  ker out range: [{K_out_actual.min().item():.4f}, {K_out_actual.max().item():.4f}]")

    diff   = (K_out_actual - ref_out).abs()
    diff_k = (K_K - ref_K).abs()
    diff_v = (K_V - ref_V).abs()
    print(f"  output  max|d|={diff.max().item():.3e}  mean|d|={diff.mean().item():.3e}")
    print(f"  K_cache max|d|={diff_k.max().item():.3e}  mean|d|={diff_k.mean().item():.3e}")
    print(f"  V_cache max|d|={diff_v.max().item():.3e}  mean|d|={diff_v.mean().item():.3e}")

    # K written slot diff vs unchanged slots (tells us if scatter works)
    write_idx = int(inp["kv_idx"][0, 0])
    diff_k_wrote = (K_K[:, :, write_idx, :] - ref_K[:, :, write_idx, :]).abs().max().item()
    K_K_other = torch.cat([K_K[:, :, :write_idx, :], K_K[:, :, write_idx+1:, :]], dim=2)
    R_K_other = torch.cat([ref_K[:, :, :write_idx, :], ref_K[:, :, write_idx+1:, :]], dim=2)
    diff_k_other = (K_K_other - R_K_other).abs().max().item() if K_K_other.numel() else 0.0
    print(f"  K_written-slot max|d|={diff_k_wrote:.3e}  K_other-slots max|d|={diff_k_other:.3e}")

    failed = (diff.max().item() > atol or diff_k.max().item() > atol or diff_v.max().item() > atol)
    print("  FAIL" if failed else "  PASS")
    return not failed


if __name__ == "__main__":
    results_per_mode = {}
    for mode in ("hbm", "sbuf_transposed"):
        print(f"\n############################################################")
        print(f"#  MODE: {mode}")
        print(f"############################################################")
        results = []
        # Full attention — gpt-oss "full_attention" layers
        for pos in (0, 63, 200, 255):
            results.append(run_case(f"full | pos={pos}", position=pos,
                                    sliding_window=None, S_max_ctx=S_MAX_CTX_FULL,
                                    mode=mode))
        # SWA — gpt-oss "sliding_attention" layers (window=128, cache=128)
        for pos in (0, 63, 127, 200):
            results.append(run_case(f"SWA  | pos={pos}", position=pos,
                                    sliding_window=SLIDING_WINDOW, S_max_ctx=S_MAX_CTX_SWA,
                                    mode=mode))
        results_per_mode[mode] = results

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_pass = 0
    total = 0
    for mode, results in results_per_mode.items():
        n_pass = sum(results)
        total_pass += n_pass
        total += len(results)
        print(f"  {mode:20s}: {n_pass}/{len(results)} passed")
    print(f"  {'TOTAL':20s}: {total_pass}/{total} passed")
    sys.exit(0 if total_pass == total else 1)
