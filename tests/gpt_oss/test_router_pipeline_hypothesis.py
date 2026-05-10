"""Verify the hypothesis that the gpt-oss megakernel uses the wrong
`router_pre_norm` flag for the router pipeline.

HYPOTHESIS (to verify, NOT to assume true):
    `transformer_gpt_oss.py:404` calls
        router_topk(..., router_pre_norm=True, norm_topk_prob=False, ...)
    Per the router_topk pipeline:
        * router_pre_norm=True  -> ACT1 path: softmax over ALL E logits, then top-K
                                   selection on those probabilities. expert_affinities
                                   is the dense [T, E] softmax (length-E).
        * router_pre_norm=False -> ACT2+Scatter path: top-K(logits) first, then
                                   softmax over only those K. expert_affinities is
                                   a one-hot scattered length-E vector — zero
                                   except at the K positions where the values
                                   are softmax(top-K logits).
    HF gpt-oss reference does:
        top_idx = topk(logits, k); top_aff = softmax(logits.gather(top_idx));
        scatter top_aff into a length-E zero vector at top_idx positions.
    -> HF matches `router_pre_norm=False` (ACT2+Scatter), NOT `True`.

    Independent claim: with router_pre_norm=True AND SBUF input, the
    `pipeline_enable_scatter` flag in router_topk evaluates to False, so the
    rebinding `expert_affinities = expert_affinities_one_hot_scattered_sb.reshape(...)`
    at router_topk.py:975 never fires.

This test runs the megakernel's exact router_topk invocation against an HF
reference under both flag settings and reports max|d|, cos_sim, support-set
match, and row-sum.
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

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

import nki
import nki.isa as nisa
import nki.language as nl

from nki_kernels.moe.router_topk import (
    XHBMLayout_T_H__1,
    XSBLayout_tp102__0,
    router_topk,
)
from nkilib.core.utils.common_types import RouterActFnType


# ---- Megakernel-style config (single TKG token, gpt-oss MoE shapes) ---------
B = 1
S_TKG = 1
T = B * S_TKG          # 1
H = 3072               # padded hidden
H0 = 128               # PMAX
H1 = H // H0           # 24
E = 32                 # experts
TOP_K = 4              # k

DTYPE = torch.bfloat16


# ---- Kernel: ONLY the router_topk subkernel, with the megakernel's exact -----
#      SBUF input contract. Two variants by flag setting. -------------------
@nki.jit
def _router_only_pre_norm_true(X, router_w, router_b):
    """router_pre_norm=True (CURRENT megakernel setting at line 404)."""
    return _router_only_body(X, router_w, router_b, router_pre_norm_flag=True)


@nki.jit
def _router_only_pre_norm_false(X, router_w, router_b):
    """router_pre_norm=False (PROPOSED FIX)."""
    return _router_only_body(X, router_w, router_b, router_pre_norm_flag=False)


def _router_only_body(X, router_w, router_b, router_pre_norm_flag: bool):
    """Mirrors `transformer_gpt_oss.py` lines 209-410:
       * loads X to SBUF in XSBLayout_tp102__0 layout (P-stride H1)
       * calls router_topk with x_sb_layout=XSBLayout_tp102__0
       * skips the rmsnorm step (so we test the router in isolation against
         a reference computed on the SAME loaded X — no other source of
         numerical noise).
    """
    BxS = T
    dtype = X.dtype

    # Same X load as the megakernel (line 214-218): P-stride H1 (layout 0).
    X_flat = X.reshape((BxS * H,))
    moe_in_sb_2d = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf, name="moe_in_sb_2d")
    nisa.dma_copy(
        dst=moe_in_sb_2d,
        src=X_flat.ap(pattern=[[H1, H0], [1, H1], [H, BxS]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )
    # Reshape into router_topk's expected 3-D SBUF view [128, T, H/128].
    moe_in_sb = moe_in_sb_2d.reshape((H0, BxS, H1))

    # HBM scratch + SBUF outputs (same as megakernel).
    router_logits_hbm = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.shared_hbm,
                                   name="router_logits_hbm")
    expert_index_sb = nl.ndarray((T, TOP_K), dtype=nl.uint32, buffer=nl.sbuf,
                                 name="expert_index_sb")
    expert_affinities_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf,
                                      name="expert_affinities_sb")

    router_outputs = router_topk(
        x=moe_in_sb,
        w=router_w,
        w_bias=router_b,
        router_logits=router_logits_hbm,
        expert_affinities=expert_affinities_sb,
        expert_index=expert_index_sb,
        act_fn=RouterActFnType.SOFTMAX,
        k=TOP_K,
        x_hbm_layout=XHBMLayout_T_H__1,
        x_sb_layout=XSBLayout_tp102__0,
        router_pre_norm=router_pre_norm_flag,
        norm_topk_prob=False,
        skip_store_router_logits=False,
    )
    # Per the kernel's contract, the returned [2] is the rebound (or DMA-stored)
    # SBUF tensor that downstream callers (moe_tkg) actually consume.
    expert_aff_sb_used = router_outputs[2]
    expert_idx_sb_used = router_outputs[1]

    # Mirror SBUF outputs to HBM for inspection — we want to see EXACTLY what
    # moe_tkg would consume, not some earlier intermediate.
    expert_aff_hbm = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.shared_hbm,
                                name="expert_aff_hbm")
    expert_idx_hbm = nl.ndarray((T, TOP_K), dtype=nl.uint32, buffer=nl.shared_hbm,
                                name="expert_idx_hbm")
    nisa.dma_copy(dst=expert_aff_hbm, src=expert_aff_sb_used)
    nisa.dma_copy(dst=expert_idx_hbm, src=expert_idx_sb_used)

    return router_logits_hbm, expert_idx_hbm, expert_aff_hbm


# ---- HF-equivalent reference (CPU, fp32) -----------------------------------
def hf_router_reference(X, router_w, router_b):
    """Mirrors HF gpt-oss `GptOssTopKRouter.forward`:
        logits = linear(x, w) + b              # [T, E]
        top_logits, top_idx = topk(logits, k)  # [T, K]
        top_aff = softmax(top_logits, dim=-1)  # [T, K]
        scatter top_aff into zeros([T, E]) at top_idx -> length-E sparse vector
    Returns:
        ref_logits: [T, E]    fp32
        ref_indices: [T, K]   int64 (canonicalized: sorted ascending)
        ref_affinities: [T, E] fp32 (zero except at top_idx positions)
    """
    Xf = X.float().reshape(T, H)
    rw = router_w.float()                    # [H, E]
    rb = router_b.float().reshape(E)         # [E]
    logits = Xf @ rw + rb                    # [T, E]
    top_logits, top_idx = torch.topk(logits, TOP_K, dim=-1)   # [T, K]
    top_aff = F.softmax(top_logits, dim=-1)                   # [T, K]
    ref_aff = torch.zeros(T, E, dtype=torch.float32)
    ref_aff.scatter_(1, top_idx, top_aff)
    # Canonicalize index order (since topk returns descending order; HF doesn't
    # guarantee a canonical order but the SET is well-defined).
    ref_idx_sorted, _ = top_idx.sort(dim=-1)
    return logits, ref_idx_sorted, ref_aff, top_idx, top_aff


# ---- "ACT1-pipeline-as-spec'd" reference -----------------------------------
# This is what the kernel SHOULD output if router_pre_norm=True is exactly the
# spec — full-E softmax in expert_affinities, top-K of softmax probs in
# expert_index. Useful as a sanity check that the kernel under
# router_pre_norm=True is doing what its docs claim.
def act1_pipeline_reference(X, router_w, router_b):
    Xf = X.float().reshape(T, H)
    rw = router_w.float()
    rb = router_b.float().reshape(E)
    logits = Xf @ rw + rb                    # [T, E]
    full_softmax = F.softmax(logits, dim=-1) # [T, E]
    _, top_idx = torch.topk(full_softmax, TOP_K, dim=-1)  # [T, K]
    top_idx_sorted, _ = top_idx.sort(dim=-1)
    return logits, top_idx_sorted, full_softmax


def _to_dev(t, device):
    return t.to(device).contiguous()


def run_kernel(kernel_fn, X, router_w, router_b):
    device = xm.xla_device()
    rl, ei, ea = kernel_fn[1](
        X=_to_dev(X, device),
        router_w=_to_dev(router_w, device),
        router_b=_to_dev(router_b, device),
    )
    xm.mark_step()
    return rl.cpu().float(), ei.cpu().to(torch.int64), ea.cpu().float()


# ---- Single comparison row -------------------------------------------------
def compare(label, ker_logits, ker_idx, ker_aff,
            ref_logits, ref_idx_sorted, ref_aff,
            extra_label=None, extra_aff=None):
    """Compare kernel outputs to a reference. ref_idx_sorted is sorted ascending.
       ker_idx will be sorted before set comparison.
    """
    print(f"\n  --- {label} ---")
    # Logits (sanity check: the matmul should always agree to ~1e-2 in bf16)
    rl_diff = (ker_logits - ref_logits).abs()
    print(f"    router_logits     max|d|={rl_diff.max().item():.3e}  "
          f"mean|d|={rl_diff.mean().item():.3e}")

    # Top-K SET match
    ker_set = set(ker_idx[0].tolist())
    ref_set = set(ref_idx_sorted[0].tolist())
    inter = len(ker_set & ref_set)
    print(f"    top-K set match   : {ker_set == ref_set}   "
          f"(intersection {inter}/{TOP_K})")
    print(f"    ker top-K (sorted): {sorted(ker_set)}")
    print(f"    ref top-K (sorted): {sorted(ref_set)}")

    # Affinity full-vector compare
    aff_diff = (ker_aff - ref_aff).abs()
    aff_max = aff_diff.max().item()
    aff_mean = aff_diff.mean().item()
    cos = F.cosine_similarity(ker_aff.flatten(), ref_aff.flatten(), dim=0).item()
    ker_row_sum = ker_aff[0].sum().item()
    ref_row_sum = ref_aff[0].sum().item()
    print(f"    expert_affinities max|d|={aff_max:.3e}  mean|d|={aff_mean:.3e}  "
          f"cos_sim={cos:.6f}")
    print(f"    expert_aff row sum: ker={ker_row_sum:.4f}   ref={ref_row_sum:.4f}")
    print(f"    ker aff slice (first 8): "
          f"{[f'{v:.4f}' for v in ker_aff[0, :8].tolist()]}")
    print(f"    ref aff slice (first 8): "
          f"{[f'{v:.4f}' for v in ref_aff[0, :8].tolist()]}")

    # Optional extra reference (e.g. ACT1-spec pipeline)
    if extra_aff is not None:
        ed = (ker_aff - extra_aff).abs()
        ec = F.cosine_similarity(ker_aff.flatten(), extra_aff.flatten(), dim=0).item()
        print(f"    [{extra_label}] max|d|={ed.max().item():.3e}  cos_sim={ec:.6f}")

    return ker_set == ref_set, aff_max, cos


def make_inputs(seed):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: (torch.randn(*s, generator=g, dtype=torch.float32) * 0.02).to(DTYPE)
    rb = lambda *s: (torch.randn(*s, generator=g, dtype=torch.float32) * 0.05).to(DTYPE)
    # X: post-RMSNorm-magnitude — mimics the moe_in_sb the megakernel feeds.
    X_actual = (torch.randn(B, S_TKG, H, generator=g, dtype=torch.float32)).to(DTYPE)
    router_w = rn(H, E)
    router_b = rb(1, E)
    return X_actual, router_w, router_b


if __name__ == "__main__":
    print("=" * 78)
    print("VERIFICATION: router_pre_norm flag in transformer_gpt_oss.py:404")
    print(f"  H={H}  E={E}  TOP_K={TOP_K}  T={T}  dtype={DTYPE}")
    print("=" * 78)

    seed = int(os.environ.get("HYPOTHESIS_SEED", "42"))
    X, router_w, router_b = make_inputs(seed=seed)
    print(f"\nUsing seed={seed}")

    # Reference (HF top-K + softmax-over-K + scatter)
    ref_logits, ref_idx_sorted, ref_aff_hf, ref_top_idx_unsorted, ref_top_aff = \
        hf_router_reference(X, router_w, router_b)
    # Spec reference for router_pre_norm=True (ACT1 pipeline: full-E softmax)
    spec_logits, spec_idx_sorted, spec_full_softmax = \
        act1_pipeline_reference(X, router_w, router_b)

    print(f"\nReference (HF gpt-oss style):")
    print(f"  logits[0, :8]: {[f'{v:.4f}' for v in ref_logits[0, :8].tolist()]}")
    print(f"  top-K idx (sorted): {sorted(ref_idx_sorted[0].tolist())}")
    print(f"  top-K aff (at sorted idx): "
          f"{[f'{ref_aff_hf[0, e].item():.4f}' for e in sorted(ref_idx_sorted[0].tolist())]}")
    print(f"  row sum of HF affinities: {ref_aff_hf[0].sum().item():.4f}  "
          f"(expected ~1.0)")
    print(f"  row sum of full-E softmax: {spec_full_softmax[0].sum().item():.4f}  "
          f"(expected ~1.0)")

    # ----- Run kernel under BOTH flag settings -----
    print("\n" + "=" * 78)
    print("Compiling and running router_topk with router_pre_norm=True (current)")
    print("=" * 78)
    try:
        ker_logits_T, ker_idx_T, ker_aff_T = run_kernel(
            _router_only_pre_norm_true, X, router_w, router_b)
        ran_T = True
    except Exception as e:
        print(f"!!! COMPILE/RUN FAILED with router_pre_norm=True: {e}")
        ran_T = False

    print("\n" + "=" * 78)
    print("Compiling and running router_topk with router_pre_norm=False (proposed)")
    print("=" * 78)
    try:
        ker_logits_F, ker_idx_F, ker_aff_F = run_kernel(
            _router_only_pre_norm_false, X, router_w, router_b)
        ran_F = True
    except Exception as e:
        print(f"!!! COMPILE/RUN FAILED with router_pre_norm=False: {e}")
        ran_F = False

    # ----- Comparisons -----
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    if ran_T:
        T_idx_match, T_aff_max, T_cos = compare(
            "router_pre_norm=TRUE  vs HF reference",
            ker_logits_T, ker_idx_T, ker_aff_T,
            ref_logits, ref_idx_sorted, ref_aff_hf,
            extra_label="vs ACT1-spec (full-E softmax)",
            extra_aff=spec_full_softmax,
        )
    if ran_F:
        F_idx_match, F_aff_max, F_cos = compare(
            "router_pre_norm=FALSE vs HF reference",
            ker_logits_F, ker_idx_F, ker_aff_F,
            ref_logits, ref_idx_sorted, ref_aff_hf,
        )

    # ----- Verdict -----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if not (ran_T and ran_F):
        print("INCONCLUSIVE: at least one flag setting failed to compile/run.")
        sys.exit(2)

    # Define "matches HF" as: support set match AND aff max|d| < 1e-2 AND cos > 0.99
    T_matches_hf = T_idx_match and (T_aff_max < 1e-2) and (T_cos > 0.99)
    F_matches_hf = F_idx_match and (F_aff_max < 1e-2) and (F_cos > 0.99)

    print(f"  router_pre_norm=TRUE  matches HF: {T_matches_hf}   "
          f"(set={T_idx_match}, max|d|={T_aff_max:.3e}, cos={T_cos:.4f})")
    print(f"  router_pre_norm=FALSE matches HF: {F_matches_hf}   "
          f"(set={F_idx_match}, max|d|={F_aff_max:.3e}, cos={F_cos:.4f})")

    if F_matches_hf and not T_matches_hf:
        print("\n  HYPOTHESIS CONFIRMED.")
        print("  router_pre_norm=False matches HF; router_pre_norm=True does not.")
        print("  Fix: in transformer_gpt_oss.py:404, change")
        print("       router_pre_norm=True   ->   router_pre_norm=False")
        sys.exit(0)
    elif T_matches_hf and not F_matches_hf:
        print("\n  HYPOTHESIS REFUTED.")
        print("  router_pre_norm=True matches HF (despite the misleading name).")
        print("  router_pre_norm=False does not. Do NOT change the flag.")
        sys.exit(1)
    elif T_matches_hf and F_matches_hf:
        print("\n  HYPOTHESIS REFUTED (both settings match HF).")
        print("  Both pipelines produce HF-equivalent affinities for this input.")
        print("  This is unexpected — likely a numerical edge case. Re-test with more inputs.")
        sys.exit(1)
    else:
        print("\n  HYPOTHESIS REFUTED: neither flag setting matches HF.")
        print("  Something else is broken in the router_topk path. Investigate further.")
        sys.exit(1)
