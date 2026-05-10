"""Unit test for the gpt-oss MoE block kernels (rmsnorm_tkg + router_topk + moe_tkg).

Validates the three subkernels invoked per-layer by the gpt-oss megakernel
(`megakernels/gpt_oss/transformer_gpt_oss.py`) against a PyTorch
reference modeled directly on `transformers/models/gpt_oss/modular_gpt_oss.py`
(`GptOssTopKRouter`, `GptOssExperts`, `GptOssMLP`) — but with NxDI's
gpt-oss-specific semantic overrides applied (see "NxDI semantic divergence"
note below), which is the actual contract the kernels implement.

================================================================================
KEY FINDING — megakernel layout bug surfaced by this test
================================================================================
The test runs three flavors of the MoE block:

  FLAVOR 1 ('megakernel'):    matches transformer_gpt_oss.py exactly:
    * X loaded with `pattern=[[1, H0], [H0, H1], [H, BxS]]`
      → SBUF P-dim stride 1 in H (XSBLayout_tp201__2 convention)
    * router_topk called with `x_sb_layout=XSBLayout_tp102__0`
      → router_topk_input_w_load loads w with P-dim stride H1 (incompatible)

  FLAVOR 2 ('matched'):       same X load, but router told `tp201_2` (matches data).

  FLAVOR 3 ('rmsnorm_native'): X loaded with P-stride H1
    (`pattern=[[H1, H0], [1, H1], [H, BxS]]`) AND router told `tp102_0`.
    All three (X, gamma loaded by rmsnorm_tkg, w loaded by router_topk)
    use the same convention → consistent.

Empirical results (5 randomised cases each):
  FLAVOR 1: 0/5 pass.  cos_sim ≈ 0   — kernel output uncorrelated with reference.
  FLAVOR 2: 0/5 pass.  cos_sim ≈ 0   — router matmul becomes correct (max|d|=1e-7
                                       vs F.linear of kernel's own moe_in), but
                                       moe_in itself is wrong (max|d| ≈ 3.0)
                                       because rmsnorm_tkg's gamma layout still
                                       doesn't match X's layout.
  FLAVOR 3: 5/5 pass.  cos_sim > 0.9999 — bf16-noise-only differences.

Root cause: rmsnorm_tkg's `load_gamma_to_sbuf` (norm_tkg_utils.py:341) uses
`hidden_dim_tp=False` by default, which loads gamma with P-stride H1
(layout 0). attention_block_tkg's internal RMSNorm (via qkv() with
fused_norm_type=RMS_NORM) uses a different code path that's compatible with
the megakernel's load layout — which is why the gpt-oss attention test passes
even with the same X load pattern. But the MoE block's rmsnorm_tkg does NOT
get told the input is in tp201_2 layout, so the elementwise `input * gamma`
multiply pairs each H-position of input with a DIFFERENT H-position's gamma.

Suggested fix (in transformer_gpt_oss.py): change the X load to
  pattern=[[H1, H0], [1, H1], [H, BxS]]
i.e. P-stride H1 (layout 0). Then the router flag is already correct
(XSBLayout_tp102__0) and rmsnorm_tkg sees the input in its expected layout.

Note: this change ALSO requires updating attention_block_tkg's call site —
that kernel currently relies on the existing P-stride-1 layout of residual_sb
(see comments at multilayer.py:202-205). So the fix needs to either (a) also
adapt attention_block_tkg's input expectation, or (b) reshape/permute
residual_sb between the attention and MoE blocks. The unit test deliberately
does NOT modify the kernels — it only surfaces the discrepancy.
================================================================================

Simulates rank 0 of TP=8 / EP=8 (so the test runs single-rank but exercises
the real per-rank shapes used in production):

  * H_actual = 2880, padded to H_padded = 3072 (multiple of 128)
  * I_per_rank = 3072 / 8 = 384
  * E = 32, TOP_K = 4
  * num_local_experts is unsharded (each rank holds all 32 expert shards
    under EP=1; EP=8 would shard the experts but our megakernel uses EP=1
    and shards the I dim across moe_tp_degree=8 — see gpt_oss_with_megakernel.py).

NxDI semantic divergence (from HF gpt-oss):
  * HF GptOssTopKRouter does `softmax(top_k(logits))` (softmax over only
    the K selected values, AFTER top-K).
  * NxDI overrides `normalize_top_k_affinities=False` and uses
    `router_pre_norm=True`, which means the kernel does
    `top_k(softmax(logits))` (softmax over ALL E experts, then top-K
    selection without further normalization).
  * gpt_oss_with_megakernel.py:511-style fold of `+1.0` into expert_gate_up_bias
    is done by NxDI's preshard_hook before the kernel sees the bias.
  * Up clamps are shifted (-6.0, +8.0) on the kernel side to compensate
    for that fold (so kernel sees `clamp(up + 1)` matching HF's
    `clamp(up) + 1`).

We mirror the kernel's semantics (softmax-over-E → top-K, no re-norm; bias
fold; clamp shift) in the PyTorch reference so the unit test validates what
the kernel was contracted to compute, not raw HF behavior.

The test is structured as one outer @nki.jit kernel that fuses the three
subkernels in the same order/composition as the megakernel's per-layer
body (residual is HBM in/out instead of SBUF, no AllReduce/residual
add — those are tested separately in the megakernel).
"""

import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
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
import nki.isa as nisa
import nki.language as nl

from nki_kernels.moe import (
    XHBMLayout_T_H__1,
    XSBLayout_tp102__0,
    moe_tkg,
    rmsnorm_tkg,
    router_topk,
)
# Not re-exported from the package __init__ — import directly from router_topk.
from nki_kernels.moe.router_topk import XSBLayout_tp201__2
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)


# ----------------------------------------------------------------------------
# Config — per-rank slice of gpt-oss-20b at TP=8 / moe_tp_degree=8 / EP=1.
# Matches transformer_gpt_oss.py constants for the MoE block.
# ----------------------------------------------------------------------------
B           = 1
S_TKG       = 1
T           = B * S_TKG          # 1 token (TKG)
H_ACTUAL    = 2880               # config.hidden_size
H           = 3072               # padded_hidden_size (multiple of 128)
H0          = 128                # PMAX
H1          = H // H0            # 24
I_PER_RANK  = 3072 // 8          # 384  (intermediate_size / moe_tp_degree)
E           = 32                 # num_local_experts
TOP_K       = 4                  # num_experts_per_tok
EPS         = 1e-5               # config.rms_norm_eps
ALPHA       = 1.702              # gpt-oss hidden_act_scaling_factor (Swish)
LIMIT       = 7.0                # gpt-oss config.limit (gate clamp)

DTYPE = torch.bfloat16


# ----------------------------------------------------------------------------
# Outer NKI kernel — wraps rmsnorm_tkg + router_topk + moe_tkg in the same
# composition the megakernel uses per layer. HBM in / HBM out (we don't keep
# anything SBUF-resident across calls; each test case runs the kernel once).
#
# We build TWO variants of the kernel, parameterised by the `x_sb_layout` flag
# passed to router_topk:
#   * MEGAKERNEL flavor (`XSBLayout_tp102__0`): exactly matches what
#     transformer_gpt_oss.py does today (production path).
#   * MATCHED  flavor (`XSBLayout_tp201__2`): the layout that actually matches
#     the in-SBUF X load pattern used by the megakernel; this is the layout
#     under which the matmul is mathematically correct vs the reference.
# Comparing the two pinpoints whether the megakernel's hard-coded layout flag
# is the source of a router-output discrepancy.
# ----------------------------------------------------------------------------
@nki.jit
def _moe_block_kernel_megakernel_layout(
    X, gpost, router_w, router_b,
    gate_up, gate_up_b, down, down_b,
):
    """As today's transformer_gpt_oss.py: X loaded with P-stride=1
    (i.e. layout tp201_2) AND router told x_sb_layout=tp102_0.

    rmsnorm_tkg gets input in layout-2; gamma is loaded internally in layout-0
    convention; the multiply scales each H position by a DIFFERENT gamma than
    the reference would — so post-RMSNorm output is wrong.
    Router_topk is told layout 0; matmul stationary is sliced the layout-0 way
    while X is laid out as layout 2; so contractions are mismatched.
    """
    return _moe_block_body(X, gpost, router_w, router_b,
                           gate_up, gate_up_b, down, down_b,
                           x_sb_layout=XSBLayout_tp102__0,
                           load_layout="tp201_2")


@nki.jit
def _moe_block_kernel_matched_layout(
    X, gpost, router_w, router_b,
    gate_up, gate_up_b, down, down_b,
):
    """X loaded with P-stride=1 (layout tp201_2) AND router told layout tp201_2.

    Router matmul is now correct (we proved router_logits matches F.linear of
    moe_in within 1e-7 fp32 noise). But rmsnorm_tkg STILL sees a layout-2
    input while loading gamma in layout-0 convention, so post-RMSNorm output
    is still wrong.
    """
    return _moe_block_body(X, gpost, router_w, router_b,
                           gate_up, gate_up_b, down, down_b,
                           x_sb_layout=XSBLayout_tp201__2,
                           load_layout="tp201_2")


@nki.jit
def _moe_block_kernel_rmsnorm_native_layout(
    X, gpost, router_w, router_b,
    gate_up, gate_up_b, down, down_b,
):
    """X loaded with P-stride=H1 (matches rmsnorm_tkg / router_topk_load
    layout-0 expectation), router told layout 0.

    This is the consistent-layout flavor: X, gamma, and W are all in layout-0
    convention, so RMSNorm is correct AND router matmul is correct.
    Hypothesis: this passes the unit test.
    """
    return _moe_block_body(X, gpost, router_w, router_b,
                           gate_up, gate_up_b, down, down_b,
                           x_sb_layout=XSBLayout_tp102__0,
                           load_layout="tp102_0")


@nki.jit
def _moe_block_kernel_option_b(
    X, gpost, router_w, router_b,
    gate_up, gate_up_b, down, down_b,
):
    """Option B: keep X loaded as XSBLayout_tp201__2 (P-stride 1) — same load
    pattern as FLAVOR 1/2 — but on the MoE side fix the consumers:
      * pass hidden_dim_tp=True to rmsnorm_tkg  (gamma loaded in tp201_2)
      * pass x_sb_layout=XSBLayout_tp201__2 to router_topk

    This validates: keep X load consistent with attention (which expects
    tp201_2), and fix the MoE side to read tp201_2 instead of tp102_0.
    """
    return _moe_block_body(X, gpost, router_w, router_b,
                           gate_up, gate_up_b, down, down_b,
                           x_sb_layout=XSBLayout_tp201__2,
                           load_layout="tp201_2",
                           hidden_dim_tp=True)


def _moe_block_body(
    X,                  # [B, S_tkg, H]                       bf16 HBM
    gpost,              # [1, H]                              bf16 HBM (post_attention_layernorm)
    router_w,           # [H, E]                              bf16 HBM
    router_b,           # [1, E]                              bf16 HBM
    gate_up,            # [E, H, 2, I_per_rank]               bf16 HBM (gate at [...,0,:], up at [...,1,:])
    gate_up_b,          # [E, 2, I_per_rank]                  bf16 HBM (already pre-folded +1 on up)
    down,               # [E, I_per_rank, H]                  bf16 HBM
    down_b,             # [E, H]                              bf16 HBM
    x_sb_layout: int = XSBLayout_tp102__0,
    load_layout: str = "tp201_2",
    hidden_dim_tp: bool = False,
):
    """Per-layer MoE block as a stand-alone kernel for testing.

    Mirrors `_multilayer_body`'s per-layer MoE block at
    transformer_gpt_oss.py:333-426, with the residual handling and
    AllReduce stripped out (we test the kernel-internal compute, not the
    cross-rank reduction).

    Returns:
      Y_hbm:            [B, S_tkg, H]      post-MoE hidden state (rank-0 partial)
      router_logits:    [T, E]             pre-softmax/post-bias router output
      expert_idx_hbm:   [T, TOP_K]         selected expert indices
      expert_aff_hbm:   [T, E]             scattered expert affinities (post-softmax,
                                            non-zero only at top-K positions)
    """
    BxS = T
    dtype = X.dtype

    # ----------------------------------------------------------------------
    # Load X into SBUF as [H0, BxS, H1] — same layout the megakernel feeds
    # to rmsnorm_tkg / router_topk / moe_tkg.
    # residual_sb[p, b*1, t] = X[b, 0, t*H0 + p]
    # ----------------------------------------------------------------------
    X_flat = X.reshape((BxS * H,))
    residual_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                             name="residual_sb")
    if load_layout == "tp201_2":
        # Megakernel-as-today pattern: residual_sb[p, b*H1+t] = X[b, t*H0+p].
        # P-dim stride in H is 1 (consecutive H elements per partition).
        nisa.dma_copy(
            dst=residual_sb,
            src=X_flat.ap(pattern=[[1, H0], [H0, H1], [H, BxS]], offset=0),
            dge_mode=nisa.dge_mode.hwdge,
        )
    else:  # load_layout == "tp102_0"
        # Layout that rmsnorm_tkg / router_topk_input_w_load (layout 0) expect:
        # residual_sb[p, b*H1+t] = X[b, p*H1 + t]. P-dim stride in H is H1.
        nisa.dma_copy(
            dst=residual_sb,
            src=X_flat.ap(pattern=[[H1, H0], [1, H1], [H, BxS]], offset=0),
            dge_mode=nisa.dge_mode.hwdge,
        )

    # ----------------------------------------------------------------------
    # Post-attention RMSNorm  (residual_sb -> moe_in_sb)
    # ----------------------------------------------------------------------
    moe_in_sb = nl.ndarray((H0, BxS, H1), dtype=dtype, buffer=nl.sbuf,
                           name="moe_in_sb")
    rmsnorm_tkg(
        input=residual_sb.reshape((H0, BxS, H1)),
        gamma=gpost,
        output=moe_in_sb,
        eps=EPS,
        hidden_actual=H_ACTUAL,
        hidden_dim_tp=hidden_dim_tp,
    )

    # ----------------------------------------------------------------------
    # Router top-K. Same wiring as the megakernel:
    #   * softmax over ALL E experts via router_pre_norm=True (ACT1 pipeline)
    #   * NO post-topk normalization (norm_topk_prob=False) — matches
    #     NxDI's `normalize_top_k_affinities=False`
    # router_logits gets a real HBM store (NCC_IGCA090: mutable_tensor must
    # have at least one store).
    # ----------------------------------------------------------------------
    router_logits_hbm = nl.ndarray((T, E), dtype=nl.float32,
                                   buffer=nl.shared_hbm,
                                   name="router_logits")
    expert_index_sb = nl.ndarray((T, TOP_K), dtype=nl.uint32,
                                 buffer=nl.sbuf,
                                 name="test_expert_index_sb")
    expert_affinities_sb = nl.ndarray((T, E), dtype=nl.float32,
                                      buffer=nl.sbuf,
                                      name="test_expert_affinities_sb")
    router_outputs = router_topk(
        x=moe_in_sb,
        w=router_w,
        w_bias=router_b,
        router_logits=router_logits_hbm,
        expert_affinities=expert_affinities_sb,
        expert_index=expert_index_sb,
        act_fn=RouterActFnType.SOFTMAX,
        k=TOP_K,
        x_hbm_layout=XHBMLayout_T_H__1,         # ignored: x is in SBUF
        x_sb_layout=x_sb_layout,
        router_pre_norm=True,
        norm_topk_prob=False,
        skip_store_router_logits=False,
    )
    # router_outputs = [router_logits, expert_index, expert_affinities]; the
    # affinities buffer is rebound to a new SBUF tensor inside router_topk,
    # so we must use the returned reference (not the input) downstream.
    expert_affinities_sb = router_outputs[2]

    # Mirror expert_index, affinities, and post-RMSNorm input to HBM for
    # diagnostic inspection.
    expert_idx_hbm = nl.ndarray((T, TOP_K), dtype=nl.uint32,
                                buffer=nl.shared_hbm, name="expert_idx_hbm")
    expert_aff_hbm = nl.ndarray((T, E), dtype=nl.float32,
                                buffer=nl.shared_hbm, name="expert_aff_hbm")
    nisa.dma_copy(dst=expert_idx_hbm, src=expert_index_sb)
    nisa.dma_copy(dst=expert_aff_hbm, src=expert_affinities_sb)

    # Store moe_in (post-RMSNorm) to HBM for inspection. We store in the
    # KERNEL-NATIVE layout: moe_in_hbm[b, t*H0 + p] = moe_in_sb[p, b, t].
    # The Python-side test reverses the permutation depending on load_layout
    # so that the comparison vs reference is always row-major-vs-row-major.
    moe_in_hbm = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.shared_hbm,
                            name="moe_in_hbm")
    moe_in_2d = moe_in_sb.reshape((H0, BxS * H1))
    for b in nl.static_range(BxS):
        for t in nl.static_range(H1):
            col_psum = nl.ndarray((1, H0), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(
                col_psum,
                moe_in_2d[0:H0, b * H1 + t : b * H1 + t + 1],
            )
            col_sb = nl.ndarray((1, H0), dtype=dtype, buffer=nl.sbuf,
                                name=f"moe_in_col_b{b}_t{t}_sb")
            nisa.tensor_copy(col_sb, col_psum)
            nisa.dma_copy(
                dst=moe_in_hbm[b:b + 1, t * H0:(t + 1) * H0],
                src=col_sb,
                dge_mode=nisa.dge_mode.hwdge,
            )

    # ----------------------------------------------------------------------
    # Selective MoE. Same args as the megakernel:
    #   * activation_fn=Swish: glu = gate * sigmoid(gate * 1.702)
    #   * gate_clamp_upper=7.0, gate_clamp_lower=None
    #   * up_clamp_upper=8.0, up_clamp_lower=-6.0  (shifted by +1 to match
    #     NxDI's preshard fold of +1 into expert_gate_up_bias on the up dim)
    #   * expert_affinities_scaling_mode=POST_SCALE
    # Output in SBUF (same layout as hidden_input: [H0, BxS, H1]).
    # ----------------------------------------------------------------------
    moe_out_sb = moe_tkg(
        hidden_input=moe_in_sb,
        expert_gate_up_weights=gate_up,
        expert_down_weights=down,
        expert_affinities=expert_affinities_sb,
        expert_index=expert_index_sb,
        is_all_expert=False,
        expert_gate_up_bias=gate_up_b,
        expert_down_bias=down_b,
        activation_fn=ActFnType.Swish,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        gate_clamp_upper_limit=7.0,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=8.0,
        up_clamp_lower_limit=-6.0,
        output_in_sbuf=True,
        output_dtype=dtype,
    )
    # moe_out_sb: [H0, BxS, H1] — same shape as moe_in_sb.

    # Store moe_out (rank-0 partial — no AllReduce in this unit test) to HBM.
    Y = nl.ndarray((B, S_TKG, H), dtype=dtype, buffer=nl.shared_hbm, name="Y")
    Y_flat = Y.reshape((BxS, H))
    # Transpose [H0, BxS, H1] -> [BxS, H1*H0] = [BxS, H] one column at a time
    # (mirrors the multilayer kernel's final-residual store at line 467+).
    moe_out_2d = moe_out_sb.reshape((H0, BxS * H1))
    for b in nl.static_range(BxS):
        for t in nl.static_range(H1):
            col_psum = nl.ndarray((1, H0), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(
                col_psum,
                moe_out_2d[0:H0, b * H1 + t : b * H1 + t + 1],
            )
            col_sb = nl.ndarray((1, H0), dtype=dtype, buffer=nl.sbuf,
                                name=f"out_col_b{b}_t{t}_sb")
            nisa.tensor_copy(col_sb, col_psum)
            nisa.dma_copy(
                dst=Y_flat[b:b + 1, t * H0:(t + 1) * H0],
                src=col_sb,
                dge_mode=nisa.dge_mode.hwdge,
            )

    return Y, router_logits_hbm, expert_idx_hbm, expert_aff_hbm, moe_in_hbm


# ----------------------------------------------------------------------------
# PyTorch reference — mirrors NxDI's gpt-oss MoE semantics (which is what the
# kernel implements), NOT raw HF GptOssMLP. See module docstring for divergence
# from HF.
# ----------------------------------------------------------------------------
def gpt_oss_moe_block_reference(
    X,           # [B, S_tkg, H]                     fp
    gpost,       # [1, H]                            fp  (post_attention_layernorm)
    router_w,    # [H, E]                            fp
    router_b,    # [1, E]                            fp
    gate_up,     # [E, H, 2, I_per_rank]             fp  (gate at [...,0,:], up at [...,1,:])
    gate_up_b,   # [E, 2, I_per_rank]                fp  (already +1-folded on up)
    down,        # [E, I_per_rank, H]                fp
    down_b,      # [E, H]                            fp
):
    """Returns (Y [B, S_tkg, H], router_logits [T, E], expert_index [T, K],
    expert_affinities [T, E]) all fp32."""
    X      = X.float()
    gpost  = gpost.float().squeeze(0)             # [H]
    rw     = router_w.float()
    rb     = router_b.float().squeeze(0)          # [E]
    gu     = gate_up.float()                      # [E, H, 2, I]
    gu_b   = gate_up_b.float()                    # [E, 2, I]
    dw     = down.float()                         # [E, I, H]
    db     = down_b.float()                       # [E, H]

    B_, S_tkg_, _ = X.shape
    T_ = B_ * S_tkg_
    H_, _ = gu.shape[1], gu.shape[3]
    I_ = gu.shape[3]

    # ---- 1. RMSNorm over ACTUAL hidden dim (H_ACTUAL=2880), zero-padded tail ----
    # Compute mean of X[..., :H_ACTUAL]^2 / H_ACTUAL; apply rsqrt; multiply by gpost.
    X_flat = X.reshape(T_, H_)
    var = X_flat[:, :H_ACTUAL].pow(2).mean(-1, keepdim=True)
    Xn = X_flat * torch.rsqrt(var + EPS) * gpost.unsqueeze(0)        # [T, H]

    # ---- 2. Router: linear → softmax(over E) → top-K (NO SCATTER) ----
    # NxDI semantics with `router_pre_norm=True` and `norm_topk_prob=False`:
    # the kernel stores the FULL [T, E] softmax (all E entries nonzero) as
    # `expert_affinities`, and the top-K *indices* separately. Downstream
    # `moe_tkg` selects `aff[expert_idx[k]]` per token. So the kernel's
    # `expert_affinities` is the dense softmax — NOT a top-K-only scatter.
    router_logits = F.linear(Xn, rw.t(), rb)                          # [T, E]
    router_softmax = F.softmax(router_logits, dim=-1)                 # [T, E]
    _, topk_idx = torch.topk(router_softmax, TOP_K, dim=-1)           # [T, K]
    # The full softmax IS the expert_affinities tensor (dense).
    expert_affinities = router_softmax

    # ---- 3. Selective experts (per-rank partial output) ----
    # Mirrors selective_expert_impl.py: for each token, for each k in 0..TOP_K-1:
    #   e   = topk_idx[t, k]
    #   gp  = gate_up[e, :, 0, :].T @ Xn[t]    (with bias, before clamp)
    #   up  = gate_up[e, :, 1, :].T @ Xn[t]    (with bias, before clamp; bias has +1 fold)
    #   gp  = clamp(gp, max=7.0)
    #   up  = clamp(up, lower=-6.0, upper=8.0)  # equiv to clamp(up, ±7) + 1
    #   glu = gp * sigmoid(gp * 1.702)
    #   gated = up * glu
    #   out_e = down[e].T @ gated + down_b[e]
    #   y[t] += out_e * affinities[t, e]
    Y = torch.zeros(T_, H_, dtype=torch.float32)
    for t_idx in range(T_):
        x_t = Xn[t_idx]                                                # [H]
        for k_idx in range(TOP_K):
            e = int(topk_idx[t_idx, k_idx].item())
            aff = expert_affinities[t_idx, e]                          # scalar
            # gate/up: x @ W_gate + b_gate, x @ W_up + b_up
            W_g = gu[e, :, 0, :]    # [H, I]
            W_u = gu[e, :, 1, :]    # [H, I]
            b_g = gu_b[e, 0, :]     # [I]
            b_u = gu_b[e, 1, :]     # [I] (already includes +1 fold)
            gate = x_t @ W_g + b_g                                      # [I]
            up   = x_t @ W_u + b_u                                      # [I]
            # Clamps: gate(None, 7.0); up(-6.0, 8.0)  (shifted +1 to match fold)
            gate = torch.clamp(gate, max=7.0)
            up   = torch.clamp(up, min=-6.0, max=8.0)
            # Swish glu: glu = gate * sigmoid(gate * 1.702); gated = up * glu
            glu = gate * torch.sigmoid(gate * ALPHA)
            gated = up * glu                                            # [I]
            # Down projection: gated @ W_down + b_down  (per-rank, no AllReduce)
            W_d = dw[e]    # [I, H]
            b_d = db[e]    # [H]
            out_e = gated @ W_d + b_d                                   # [H]
            # POST_SCALE: multiply by affinity AFTER the per-expert sum
            Y[t_idx] += out_e * aff

    Y = Y.reshape(B_, S_tkg_, H_)
    return Y, router_logits, topk_idx, expert_affinities


# ----------------------------------------------------------------------------
# Test harness
# ----------------------------------------------------------------------------
def make_inputs(seed: int, x_scale: float = 1.0, x_dist: str = "randn"):
    """Generate randomized weights, hidden state.

    `x_dist`:
      * "randn"            : X_actual ~ N(0, x_scale^2)   (default; classic test)
      * "post_attn_resid"  : X_actual = embed + attn_out, with embed ~ N(0, 1)
                             and attn_out ~ N(0, 1) (independent), summed and
                             optionally rescaled by x_scale. This better matches
                             the residual stream magnitude that the MoE block
                             actually sees in the megakernel (post-embedding +
                             one attention layer's output added back).
    """
    g = torch.Generator().manual_seed(seed)
    rn  = lambda *s: (torch.randn(*s, generator=g, dtype=torch.float32) * 0.02).to(DTYPE)
    rb  = lambda *s: (torch.randn(*s, generator=g, dtype=torch.float32) * 0.05).to(DTYPE)
    rn_unit = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32).to(DTYPE)

    # Hidden state: [B, S_tkg, H], padded tail zero (mimics what megakernel sees
    # post-residual / pre-norm). x_scale lets us probe different magnitudes.
    if x_dist == "randn":
        X_actual = rn_unit(B, S_TKG, H_ACTUAL) * x_scale
    elif x_dist == "post_attn_resid":
        # Residual stream after one attention layer: embed + attn_out. Both
        # contributions are roughly unit-Gaussian post-norm, and they sum
        # ≈ N(0, sqrt(2)) per element. This is the realistic distribution
        # MoE sees in the megakernel; bf16 noise here exercises a slightly
        # larger ulp budget than randn*1.0 alone.
        embed    = torch.randn(B, S_TKG, H_ACTUAL, generator=g, dtype=torch.float32)
        attn_out = torch.randn(B, S_TKG, H_ACTUAL, generator=g, dtype=torch.float32)
        X_actual = ((embed + attn_out) * x_scale).to(DTYPE)
    else:
        raise ValueError(f"unknown x_dist {x_dist!r}")
    X = torch.zeros(B, S_TKG, H, dtype=DTYPE)
    X[..., :H_ACTUAL] = X_actual

    # post_attention_layernorm γ — small offset around 1.0
    gpost = torch.zeros(1, H, dtype=DTYPE)
    gpost[0, :H_ACTUAL] = (
        torch.randn(H_ACTUAL, generator=g, dtype=torch.float32) * 0.05 + 1.0
    ).to(DTYPE)

    # Router weights (small)
    router_w = rn(H, E)
    router_b = rb(1, E)
    # Mask weight rows for padded H (>= H_ACTUAL) to 0 so the post-RMSNorm
    # zero-padded tail doesn't contribute to router logits.
    router_w[H_ACTUAL:, :] = 0

    # Expert weights
    gate_up   = rn(E, H, 2, I_PER_RANK)
    gate_up_b = rn(E, 2, I_PER_RANK)
    # Pre-fold +1 into the up bias (NxDI's preshard_hook does this on load,
    # so the kernel sees the already-folded bias).
    gate_up_b[:, 1, :] = (gate_up_b[:, 1, :].float() + 1.0).to(DTYPE)
    # Mask padded H rows (>= H_ACTUAL) on the gate/up weights so the padded
    # tail doesn't contribute to gate/up projections.
    gate_up[:, H_ACTUAL:, :, :] = 0

    down   = rn(E, I_PER_RANK, H)
    down_b = rn(E, H)
    # Mask down output for padded H so down only writes to the actual hidden dim.
    down[:, :, H_ACTUAL:] = 0
    down_b[:, H_ACTUAL:] = 0

    return dict(
        X=X, X_actual=X_actual,
        gpost=gpost,
        router_w=router_w, router_b=router_b,
        gate_up=gate_up, gate_up_b=gate_up_b,
        down=down, down_b=down_b,
    )


_FLAVORS = {
    "megakernel":      ("tp201_2", _moe_block_kernel_megakernel_layout),
    "matched":         ("tp201_2", _moe_block_kernel_matched_layout),
    "rmsnorm_native":  ("tp102_0", _moe_block_kernel_rmsnorm_native_layout),
    "option_b":        ("tp201_2", _moe_block_kernel_option_b),
}


def _inverse_permute(t_native: torch.Tensor, load_layout: str) -> torch.Tensor:
    """Convert the kernel's native-layout output to row-major.

    Kernel always stores out[b, j=t*H0+p] = sbuf[p, b, t]. The mapping from j
    to the row-major H index depends on the load_layout used:

      * tp201_2  : sbuf[p, b, t] = output[b, t*H0 + p] = output[b, j]
                   → t_native is already row-major; no permute.
      * tp102_0  : sbuf[p, b, t] = output[b, p*H1 + t]
                   → output[b, h] = t_native[b, (h % H1) * H0 + (h // H1)]
    """
    if load_layout == "tp201_2":
        return t_native
    elif load_layout == "tp102_0":
        # Build a row-major output tensor by gathering from t_native.
        BxS, H_padded = t_native.shape
        out = torch.empty_like(t_native)
        h = torch.arange(H_padded)
        j = (h % H1) * H0 + (h // H1)   # j as a function of h
        out[:, h] = t_native[:, j]
        return out
    else:
        raise ValueError(f"unknown load_layout {load_layout!r}")


def run_kernel(inp, flavor: str = "megakernel"):
    """Run the @nki.jit-wrapped MoE block kernel on XLA device."""
    if flavor not in _FLAVORS:
        raise ValueError(f"unknown flavor {flavor!r}")
    load_layout, kernel = _FLAVORS[flavor]
    device = xm.xla_device()
    to_dev = lambda t: t.to(device).contiguous()
    Y, rl, ei, ea, mi = kernel[1](
        X=to_dev(inp["X"]),
        gpost=to_dev(inp["gpost"]),
        router_w=to_dev(inp["router_w"]),
        router_b=to_dev(inp["router_b"]),
        gate_up=to_dev(inp["gate_up"]),
        gate_up_b=to_dev(inp["gate_up_b"]),
        down=to_dev(inp["down"]),
        down_b=to_dev(inp["down_b"]),
    )
    xm.mark_step()
    Y_cpu  = Y.cpu().float()
    mi_cpu = mi.cpu().float()
    # Reshape Y to [BxS, H] for permute-by-flavor, then back to [B, S_tkg, H].
    Y_2d_native = Y_cpu.reshape(B * S_TKG, H)
    Y_2d  = _inverse_permute(Y_2d_native, load_layout)
    mi_2d = _inverse_permute(mi_cpu,      load_layout)
    return (Y_2d.reshape(B, S_TKG, H),
            rl.cpu().float(),
            ei.cpu().to(torch.int64),
            ea.cpu().float(),
            mi_2d)


def run_case(name, seed, x_scale, flavor="megakernel", atol=5e-2, rtol=1e-2,
             x_dist="randn"):
    print(f"\n=== {name} | seed={seed} | x_scale={x_scale} | "
          f"x_dist={x_dist} | flavor={flavor} ===")
    inp = make_inputs(seed=seed, x_scale=x_scale, x_dist=x_dist)

    # PyTorch reference (CPU, fp32)
    ref_Y, ref_rl, ref_idx, ref_aff = gpt_oss_moe_block_reference(
        X=inp["X"],
        gpost=inp["gpost"],
        router_w=inp["router_w"],
        router_b=inp["router_b"],
        gate_up=inp["gate_up"],
        gate_up_b=inp["gate_up_b"],
        down=inp["down"],
        down_b=inp["down_b"],
    )
    print(f"  ref Y      range : [{ref_Y[..., :H_ACTUAL].min().item():.4f}, "
          f"{ref_Y[..., :H_ACTUAL].max().item():.4f}]")
    print(f"  ref logits range : [{ref_rl.min().item():.4f}, {ref_rl.max().item():.4f}]")
    print(f"  ref top-K idx[0] : {ref_idx[0].tolist()}")
    print(f"  ref top-K aff[0] : "
          f"{[f'{ref_aff[0, e].item():.4f}' for e in ref_idx[0].tolist()]}")

    # Kernel
    ker_Y, ker_rl, ker_idx, ker_aff, ker_moe_in = run_kernel(inp, flavor=flavor)

    # ----- Diagnostic: post-RMSNorm input -----
    # Reference RMSNorm
    X_ref = inp["X"].float().reshape(T, H)
    var = X_ref[:, :H_ACTUAL].pow(2).mean(-1, keepdim=True)
    Xn_ref = X_ref * torch.rsqrt(var + EPS) * inp["gpost"].float()
    moe_in_diff = (ker_moe_in - Xn_ref).abs()
    print(f"  moe_in (post-RMSNorm)  max|d|={moe_in_diff.max().item():.3e}  "
          f"mean|d|={moe_in_diff.mean().item():.3e}")

    # ----- Diagnostic: router logits computed from KER moe_in (so we isolate
    #       the router matmul from RMSNorm differences) -----
    rl_from_ker_moein = (
        ker_moe_in @ inp["router_w"].float() + inp["router_b"].float()
    )
    rl_isolation_diff = (ker_rl - rl_from_ker_moein).abs()
    print(f"  router_logits (vs F.linear(ker_moe_in)) max|d|="
          f"{rl_isolation_diff.max().item():.3e}  "
          f"mean|d|={rl_isolation_diff.mean().item():.3e}")

    # Trim padded tail of Y for comparison (kernel writes garbage to [H_ACTUAL:H]
    # because the padded-row weights are zero only for the gpt-oss-shaped
    # references; with our mask above, padded tail should be ~0 in both).
    ker_Y_actual = ker_Y[..., :H_ACTUAL]
    ref_Y_actual = ref_Y[..., :H_ACTUAL]
    print(f"  ker Y      range : [{ker_Y_actual.min().item():.4f}, "
          f"{ker_Y_actual.max().item():.4f}]")
    print(f"  ker logits range : [{ker_rl.min().item():.4f}, {ker_rl.max().item():.4f}]")
    print(f"  ker top-K idx[0] : {ker_idx[0].tolist()}")
    print(f"  ker top-K aff[0] : "
          f"{[f'{ker_aff[0, e].item():.4f}' for e in ker_idx[0].tolist()]}")

    # ----- Comparisons -----
    # 1. Router logits (pre-softmax)
    rl_diff = (ker_rl - ref_rl).abs()
    print(f"  router_logits  max|d|={rl_diff.max().item():.3e}  "
          f"mean|d|={rl_diff.mean().item():.3e}")

    # 2. Top-K indices: compare as a SET (top-K ordering can differ when two
    # softmax values are very close, but the SELECTED set should match).
    ref_set = set(ref_idx[0].tolist())
    ker_set = set(ker_idx[0].tolist())
    idx_match = (ref_set == ker_set)
    idx_intersection = len(ref_set & ker_set)
    print(f"  top-K idx set match : {idx_match}  "
          f"(intersection size {idx_intersection}/{TOP_K})")

    # 3. Expert affinities (full E vector, only top-K are nonzero in both).
    aff_diff = (ker_aff - ref_aff).abs()
    print(f"  expert_aff     max|d|={aff_diff.max().item():.3e}  "
          f"mean|d|={aff_diff.mean().item():.3e}")

    # 4. Final MoE output (rank-0 partial, on actual hidden dim only).
    y_diff = (ker_Y_actual - ref_Y_actual).abs()
    y_max  = y_diff.max().item()
    y_mean = y_diff.mean().item()
    print(f"  moe_output     max|d|={y_max:.3e}  mean|d|={y_mean:.3e}")

    # Relative — robust to varying output magnitudes across cases.
    ref_norm = ref_Y_actual.norm().item()
    diff_norm = y_diff.norm().item()
    rel = diff_norm / max(ref_norm, 1e-8)
    cos = F.cosine_similarity(ker_Y_actual.flatten(),
                              ref_Y_actual.flatten(), dim=0).item()
    print(f"  moe_output     ||d|| / ||ref|| = {rel:.3e}   cos_sim = {cos:.6f}")

    # ----- Pass criteria -----
    # bf16 ~2-3e-3 relative noise per multiply; ~10 stages x 32 MAC each per
    # token ≈ tolerance ~5e-2 abs on outputs of magnitude up to ~1.0.
    ok = True
    ok &= (rl_diff.max().item() < 5e-2)            # router logits
    ok &= idx_match                                # top-K set
    ok &= (aff_diff.max().item() < 1e-2)           # affinities
    ok &= (y_max < atol or rel < 5e-2 or cos > 0.99)  # MoE output

    print("  PASS" if ok else "  FAIL")
    return ok


if __name__ == "__main__":
    print("Compiling and running gpt-oss MoE block kernels test...")
    print(f"  H_actual={H_ACTUAL}, H_padded={H}, I_per_rank={I_PER_RANK}, "
          f"E={E}, TOP_K={TOP_K}, EPS={EPS}")
    print(f"  ActFn=Swish (alpha={ALPHA}), gate_clamp=(None, 7.0), "
          f"up_clamp=(-6.0, 8.0)")

    cases = [
        # name,                  seed, x_scale
        ("small post-residual",  42,   0.5),    # typical post-residual magnitude
        ("medium post-residual", 123,  1.0),    # mid-stack
        ("large post-residual",  2024, 2.0),    # deep-stack residual
        ("small alt-seed",       7,    0.5),    # guards against top-K ties
        ("clamp-stressing",      99,   1.5),    # exercises gate/up clamps
    ]

    # Extra FLAVOR-3-only scenario: realistic post-attention residual-stream
    # distribution (embed + attn_out). Magnitude is what MoE actually sees in
    # the megakernel (per-element ~ N(0, sqrt(2))), so this confirms MoE is
    # solid for any plausible residual distribution that the attention layer
    # would feed it in production.
    flavor3_extra_cases = [
        # name,                            seed, x_scale, x_dist
        ("post-attn residual (embed+attn)", 314,  1.0,    "post_attn_resid"),
    ]

    # ------------------------------------------------------------------
    # FLAVOR 1: megakernel-as-today
    #   X load pattern: residual_sb[p, b*H1+t] = X[b, t*H0 + p]
    #     (pattern=[[1, H0], [H0, H1], [H, BxS]]) — P-stride 1 in H
    #   Router told: x_sb_layout=XSBLayout_tp102__0  (P-stride H1 expected)
    #
    # This is exactly what transformer_gpt_oss.py does (lines
    # 213-218 for the load and 388 for the router flag). We expect it to fail
    # because (a) rmsnorm_tkg multiplies the input by a gamma that's loaded
    # with P-stride H1, mismatching the input's P-stride 1; and (b) the
    # router_topk is told layout 0 while the data is layout 2.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FLAVOR 1: 'megakernel' (as-today)")
    print("          load: P-stride 1 (tp201_2) | router: x_sb_layout=tp102_0")
    print("=" * 78)
    mega_results = [run_case(name, seed=seed, x_scale=xs, flavor="megakernel")
                    for (name, seed, xs) in cases]

    # ------------------------------------------------------------------
    # FLAVOR 2: 'matched' router layout — keep the X load as-is, but tell
    # router_topk x_sb_layout=tp201_2 (matches the data layout).
    # Expected: router matmul becomes correct, but rmsnorm_tkg still has the
    # X-vs-gamma layout mismatch, so post-RMSNorm input is still wrong.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FLAVOR 2: 'matched' (router layout fixed only)")
    print("          load: P-stride 1 (tp201_2) | router: x_sb_layout=tp201_2")
    print("=" * 78)
    matched_results = [run_case(name, seed=seed, x_scale=xs, flavor="matched")
                       for (name, seed, xs) in cases]

    # ------------------------------------------------------------------
    # FLAVOR 3: 'rmsnorm_native' — X loaded with P-stride H1 (matches what
    # rmsnorm_tkg's gamma loader and router_topk_input_w_load layout-0 BOTH
    # expect). Router told x_sb_layout=tp102_0 (matches).
    # Expected: BOTH RMSNorm and router are correct → kernel matches reference.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FLAVOR 3: 'rmsnorm_native' (consistent layout 0 throughout)")
    print("          load: P-stride H1 (tp102_0) | router: x_sb_layout=tp102_0")
    print("=" * 78)
    native_results = [run_case(name, seed=seed, x_scale=xs, flavor="rmsnorm_native")
                      for (name, seed, xs) in cases]
    # Append the realistic post-attention-residual-stream scenario.
    native_results += [
        run_case(name, seed=seed, x_scale=xs, flavor="rmsnorm_native", x_dist=xd)
        for (name, seed, xs, xd) in flavor3_extra_cases
    ]

    # ------------------------------------------------------------------
    # FLAVOR 4: 'option_b' — keep X loaded as P-stride 1 (tp201_2, the same
    # convention attention expects), but fix the MoE-side consumers:
    #   * rmsnorm_tkg:  hidden_dim_tp=True  (gamma loaded in tp201_2)
    #   * router_topk:  x_sb_layout=XSBLayout_tp201__2
    # If this passes, it validates Option B: the megakernel can keep its
    # current X load layout and only the MoE-block flag/kwarg pair needs
    # changing. If it fails, Option B is insufficient and a different fix
    # (e.g. Option A: change X load to tp102_0) is required.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FLAVOR 4: 'option_b' (keep X load tp201_2; fix MoE-side consumers)")
    print("          load: P-stride 1 (tp201_2) | rmsnorm hidden_dim_tp=True | "
          "router: x_sb_layout=tp201_2")
    print("=" * 78)
    option_b_results = [run_case(name, seed=seed, x_scale=xs, flavor="option_b")
                        for (name, seed, xs) in cases]

    n_mega         = sum(mega_results)
    n_matched      = sum(matched_results)
    n_native       = sum(native_results)
    n_option_b     = sum(option_b_results)
    n_cases        = len(cases)
    n_native_total = len(cases) + len(flavor3_extra_cases)
    print("\n" + "=" * 78)
    print(f"SUMMARY")
    print(f"  FLAVOR 1 (megakernel)     : {n_mega}/{n_cases} pass")
    print(f"  FLAVOR 2 (matched router) : {n_matched}/{n_cases} pass")
    print(f"  FLAVOR 3 (rmsnorm_native) : {n_native}/{n_native_total} pass")
    print(f"  FLAVOR 4 (option_b)       : {n_option_b}/{n_cases} pass")
    print("=" * 78)

    if n_native == n_native_total and n_mega < n_cases:
        print(
            "DIAGNOSIS — bug in transformer_gpt_oss.py:\n"
            "  The megakernel-as-today flavor (FLAVOR 1) fails on every case.\n"
            "  The consistent-layout flavor (FLAVOR 3) passes, isolating the\n"
            "  bug to a layout mismatch in the megakernel's MoE block.\n"
            "\n"
            "  Two independent issues:\n"
            "    (a) The X load pattern at multilayer.py:213-218\n"
            "          pattern=[[1, H0], [H0, H1], [H, BxS]]\n"
            "        produces an SBUF tensor with P-dim stride 1 in H\n"
            "        (XSBLayout_tp201__2 convention). But rmsnorm_tkg's\n"
            "        gamma loader (norm_tkg_utils.py:load_gamma_to_sbuf,\n"
            "        hidden_dim_tp=False) loads gamma with P-dim stride H1\n"
            "        (XSBLayout_tp102__0 convention). So the in-kernel\n"
            "        elementwise `gamma_mult = input * gamma` multiplies\n"
            "        each H position by the gamma of a DIFFERENT H position,\n"
            "        producing a wrong post-RMSNorm output.\n"
            "        FLAVOR 2 evidence: with X load unchanged, the post-\n"
            "        RMSNorm `moe_in` differs from the reference by\n"
            "        max|d| ≈ 3.0 even though the router matmul itself is\n"
            "        correct (max|d| ≈ 1e-7 vs F.linear of the kernel's\n"
            "        own moe_in).\n"
            "\n"
            "    (b) router_topk is called with x_sb_layout=XSBLayout_tp102__0\n"
            "        (multilayer.py:388) regardless of the data's actual\n"
            "        layout. Even if (a) is fixed, this flag must match the\n"
            "        data; otherwise router_topk_input_w_load loads `w` with\n"
            "        a stride pattern incompatible with X's, contracting\n"
            "        mismatched H positions in the matmul.\n"
            "\n"
            "  Fix: change multilayer.py to load X with P-stride H1 (i.e.\n"
            "    pattern=[[H1, H0], [1, H1], [H, BxS]]), and keep the router\n"
            "    flag as XSBLayout_tp102__0. This makes both rmsnorm_tkg and\n"
            "    router_topk consistent with the loaded X layout.\n"
            "    Alternatively, keep the current X load and (i) call\n"
            "    rmsnorm_tkg with hidden_dim_tp=True to get a P-stride-1\n"
            "    gamma; (ii) pass XSBLayout_tp201__2 to router_topk. Either\n"
            "    direction works as long as ALL three (X, gamma, router) use\n"
            "    the same convention.\n"
        )
    elif n_mega == n_cases:
        print("All three flavors pass — no megakernel layout bug detected.\n")
    else:
        print("Inconclusive — see per-case output above for details.\n")

    # Final exit code: PASS if FLAVOR 3 (the diagnostic baseline) passes for
    # all cases. FLAVOR 3 is the "kernels are correct under consistent layout"
    # check; that's what this unit test is meant to assert. FLAVOR 1 + 2 are
    # diagnostic and report the megakernel bug — they don't gate the exit
    # code (otherwise the test would always fail in the presence of the bug
    # the test is meant to surface).
    print(f"\nFINAL: FLAVOR 3 (kernel correctness) = {n_native}/{n_native_total} pass")
    print(f"FINAL: FLAVOR 4 (option_b)            = {n_option_b}/{n_cases} pass")
    sys.exit(0 if (n_native == n_native_total and n_option_b == n_cases) else 1)
