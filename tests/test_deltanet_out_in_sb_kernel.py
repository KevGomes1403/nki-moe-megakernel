"""Diagnostic device test for the DeltaNet fused layer's SBUF-in + out_in_sb=True o_proj path.

DeltaNet analog of test_gqa_out_in_sb_kernel. Layer 0 of Qwen3.6-A3B is DeltaNet, and in the
verify-megakernel its input is the SBUF tp2013 residual (not HBM) AND its input RMSNorm is FUSED
inside qkv_tkg's SBUF-in path (via in_proj_compose) -- a DIFFERENT code path than GQA's separate
pre-attn norm, and one never previously run with SBUF input. This test drives exactly that seam.

Trusted references (do NOT re-derive the recurrence):
  (a) PyTorch ``golden(inp)`` from test_deltanet_in_proj_out_fused_kernel (RMSNorm@proj_w -> conv ->
      delta-rule recurrence -> gated norm -> o_proj), returning o_out [T, HIDDEN].
  (b) The validated Run-A path ``deltanet_attention_layer_state[cores]`` (HBM-in, out_in_sb=False) --
      the E2E-green kernel that ships in the model. The SBUF-in/out_in_sb path must match its o_out
      (the exact analog of GQA's out_in_sb=True vs =False cross-check, which came back bit-identical).

New wrapper ``deltanet_sbufin_out_in_sb_fwd`` (@nki.jit): loads the tp2013 residual with the
megakernel's own load_residual_to_sbuf, reshapes to [H0,T,H1], and feeds attention_layer_compose
SBUF-in + out_in_sb=True (name_prefix="dnsb_"), then reads each core's [H0, H1_shard*T] o_proj shard
back to HBM. Host reconstructs natural [T, HIDDEN] (generic in H1_shard, identical to GQA):
  out_natural[t, c*(H0*H1_shard) + h0*H1_shard + h2] = shard_c[h0, h2*T + t]
i.e. readback.reshape(n_prgs,128,H1_shard,T).permute(3,0,1,2).reshape(T, HIDDEN).

Gate: fp32 torch.allclose(atol=1e-5, rtol=1e-2). max_abs/max_rel printed for every comparison; NO
cosine (banned repo-wide). Cases: cores=2/T=2 (primary megakernel config), cores=1/T=2, cores=2/T=1.

If SBUF-in diverges from deltanet_attention_layer_state (while GQA matched), that is very likely THE
megakernel bug -- the test characterizes the divergence; it does NOT edit the kernel to make it pass.

Run (CORES 0,1):
    cd /home/ubuntu/trainium-model-translation && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_deltanet_out_in_sb_kernel
"""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

from nkilib.core.utils.kernel_helpers import (  # noqa: E402
    get_program_sharding_info,
)
from nkilib.core.utils.tensor_view import TensorView  # noqa: E402

from models.qwen3_6_moe.nki_kernels.deltanet.decode.fused_layer import (  # noqa: E402
    attention_layer_compose,
    deltanet_attention_layer_state,
)
from models.qwen3_6_moe.nki_kernels.megakernel.qwen36_verify_megakernel import (  # noqa: E402
    load_residual_to_sbuf,
)

# Reuse the trusted recurrence reference + input construction VERBATIM (do not re-derive).
from models.qwen3_6_moe.tests.test_deltanet_in_proj_out_fused_kernel import (  # noqa: E402
    ATOL,
    CONV_DIM,
    EPS,
    HEAD_DIM,
    HIDDEN,
    HV,
    KEY_DIM,
    RTOL,
    STATE_W,
    _set_dtype,
    golden,
    make_inputs,
    run_kernel,  # launches deltanet_attention_layer_state (HBM-in, out_in_sb=False)
)

P_MAX = 128


# ---------------------------------------------------------------------------
# New entrypoint: SBUF-in (tp2013 residual) + out_in_sb=True, mirroring the megakernel's
# attention_layer_compose(..., out_in_sb=True) call, with per-core readback to HBM.
# ---------------------------------------------------------------------------
@nki.jit
def deltanet_sbufin_out_in_sb_fwd(
    hidden,
    proj_w,
    gamma,
    eps,
    conv_state,
    conv_weight,
    key_dim,
    A_log,
    dt_bias,
    init_state,
    z_gamma,
    out_w,
    z_eps=1e-6,
):
    """Megakernel seam: RAW hidden [1,T,H] -> tp2013 SBUF residual [H0, T*H1] via the megakernel's
    load_residual_to_sbuf, reshaped to [H0,T,H1] and fed SBUF-in to attention_layer_compose
    (input norm FUSED in qkv_tkg's SBUF-in path, out_in_sb=True). Reads each core's [H0, H1_shard*T]
    o_proj shard back to HBM [n_prgs, H0, H1_shard*T]. Returns (out, candidate_states, conv_cand)."""
    B, S, H = hidden.shape
    T = B * S
    h0 = P_MAX
    h1 = H // h0
    _, n_prgs, prg_id = get_program_sharding_info()

    conv_dim = conv_weight.shape[0]
    state_w = conv_weight.shape[1] - 1
    hv_full = (conv_dim - 2 * key_dim) // h0

    residual = nl.ndarray((h0, T * h1), dtype=hidden.dtype, buffer=nl.sbuf)
    load_residual_to_sbuf(residual, hidden, T, h0, h1, n_prgs)
    x_sb = residual.reshape((h0, T, h1))

    cand_state = nl.ndarray(
        (T, hv_full, h0, h0), dtype=nl.float32, buffer=nl.shared_hbm
    )
    conv_cand = nl.ndarray(
        (T, conv_dim, state_w), dtype=conv_weight.dtype, buffer=nl.shared_hbm
    )

    o_sb = attention_layer_compose(
        x_sb,
        proj_w,
        gamma,
        eps,
        conv_state,
        conv_weight,
        key_dim,
        A_log,
        dt_bias,
        init_state,
        out_w,
        cand_state,
        conv_cand,
        write_candidates=True,
        cand_is_3d=True,
        z_gamma=z_gamma,
        z_eps=z_eps,
        out_in_sb=True,
        name_prefix="dnsb_",
    )
    out_h0, free = o_sb.shape  # [128, H1_shard*T]
    out = nl.ndarray((n_prgs, out_h0, free), dtype=o_sb.dtype, buffer=nl.shared_hbm)
    out_view = TensorView(out).select(dim=0, index=prg_id).get_view()
    nisa.dma_copy(dst=out_view, src=o_sb)
    return out, cand_state, conv_cand


# ---------------------------------------------------------------------------
# Host reconstruction (generic in H1_shard -- identical formula to GQA).
# ---------------------------------------------------------------------------
def _reconstruct_natural(readback, T):
    """[n_prgs, H0, H1_shard*T] -> natural [T, HIDDEN].

    Free axis is h2-major t-minor (f = h2*T + t); natural hidden h = c*(H0*H1_shard) + h0*H1_shard + h2."""
    n_prgs, h0, free = readback.shape
    h2 = free // T
    assert h0 == P_MAX and h2 * T == free
    assert n_prgs * h0 * h2 == HIDDEN, (
        f"shard decomposition {n_prgs}*{h0}*{h2} != H={HIDDEN}"
    )
    o = readback.reshape(n_prgs, h0, h2, T)  # [c, h0, h2, t]
    return o.permute(3, 0, 1, 2).reshape(T, HIDDEN)  # [t, (c,h0,h2)=h]


def run_kernel_sbufin(inp, cores):
    """Launch the SBUF-in megakernel-seam path (fp32) and reconstruct natural [T, HIDDEN]."""
    import torch_xla.core.xla_model as xm

    (
        hidden,
        proj_w,
        gamma,
        conv_state,
        conv_weight,
        A_log,
        dt_bias,
        init_state,
        z_gamma,
        out_w,
    ) = inp
    dev = xm.xla_device()
    f = torch.float32
    o_out, _cand, _conv = deltanet_sbufin_out_in_sb_fwd[cores](
        hidden.to(f).contiguous().to(dev),
        proj_w.to(f).contiguous().to(dev),
        gamma.reshape(1, HIDDEN).to(f).contiguous().to(dev),  # qkv_tkg wants [1, H]
        EPS,
        conv_state.to(f).contiguous().to(dev),
        conv_weight.to(f).contiguous().to(dev),
        KEY_DIM,
        A_log.to(f).contiguous().to(dev),
        dt_bias.to(f).contiguous().to(dev),
        init_state.to(f).contiguous().to(dev),
        z_gamma.to(f).contiguous().to(dev),
        out_w.to(f).contiguous().to(dev),
        EPS,
    )
    T = hidden.shape[1]
    readback = o_out.float().cpu()  # [n_prgs, 128, H1_shard*T]
    return _reconstruct_natural(readback, T)


# ---------------------------------------------------------------------------
# Metrics / checks
# ---------------------------------------------------------------------------
def _metrics(ker, ref):
    kd = ker.double().reshape(-1)
    rd = ref.double().reshape(-1)
    abs_err = (kd - rd).abs()
    denom = rd.abs().clamp_min(1e-4)
    return abs_err.max().item(), (abs_err / denom).max().item()


def _check(name, ker, ref):
    max_abs, max_rel = _metrics(ker.float(), ref.float())
    ok = torch.allclose(ker.float(), ref.float(), atol=ATOL, rtol=RTOL)
    status = "PASS" if ok else "FAIL"
    print(
        f"[{name}] {status}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
        f"gate=allclose(atol={ATOL} rtol={RTOL})"
    )
    return ok, max_abs, max_rel


def run_case(name, T, cores, seed):
    """Run the SBUF-in seam and compare reconstructed [T,HIDDEN] against (a) golden and
    (b) deltanet_attention_layer_state's o_out (Run-A HBM-in out_in_sb=False). All fp32."""
    _set_dtype(
        torch.float32
    )  # golden/run_kernel use the reference module's global DTYPE
    inp = make_inputs(T=T, seed=seed)

    ker_sbufin = run_kernel_sbufin(inp, cores=cores)  # [T, HIDDEN]
    ref_o, _rs, _rc = golden(inp)  # [T, HIDDEN] golden
    ker_runA, _ka, _kc = run_kernel(deltanet_attention_layer_state, inp, cores=cores)
    ker_runA = ker_runA.reshape(T, HIDDEN)

    print(f"--- {name} SBUF-IN (cores={cores}, T={T}) ---")
    ok_g, _, _ = _check(f"{name} SBUF-in vs golden", ker_sbufin, ref_o)
    ok_r, _, _ = _check(
        f"{name} SBUF-in vs deltanet_attention_layer_state", ker_sbufin, ker_runA
    )
    # Sanity: Run-A itself must match golden (proves inputs/reference are sound).
    _check(f"{name} (Run-A vs golden)", ker_runA, ref_o)
    return ok_g and ok_r, ker_sbufin, ref_o, ker_runA


def _diagnose_if_fail(tag, ok, ker, ref, krunA, T, cores):
    """If SBUF-in diverges, characterize the pattern vs both golden and the Run-A reference."""
    if ok:
        return
    print(f"  [DIAGNOSTIC {tag}] SBUF-in seam diverges. Characterizing pattern:")
    k = ker.double()
    r = ref.double()
    a = krunA.double()
    # Does Run-A itself still match golden? (True => fault is ONLY in the SBUF-in/out_in_sb path.)
    runA_ok = torch.allclose(a, r, atol=ATOL, rtol=RTOL)
    print(
        f"    Run-A (HBM-in) vs golden allclose: {runA_ok} "
        f"(True => divergence is ISOLATED to the SBUF-in tp2013 read / out_in_sb path)"
    )
    # Scale error?
    denom = a.abs().clamp_min(1e-6)
    ratio = k.abs() / denom
    print(
        f"    global |sbufin|/|runA| ratio: median={ratio.median().item():.4f} "
        f"min={ratio.min().item():.4f} max={ratio.max().item():.4f}"
    )
    # Permutation test vs Run-A (values present but reordered?).
    ks = k.flatten().sort().values
    as_ = a.flatten().sort().values
    perm_absdiff = (ks - as_).abs().max().item()
    print(
        f"    sorted-value max_abs vs Run-A (permutation test): {perm_absdiff:.3e} "
        f"(small => values present but SCRAMBLED/permuted; large => magnitudes wrong)"
    )
    # Per-core hidden-shard error vs Run-A (one core's shard swapped/wrong?).
    col_err = (k - a).abs().max(dim=0).values  # [HIDDEN]
    H = col_err.shape[0]
    per_core = col_err.reshape(cores, H // cores)
    for c in range(cores):
        print(
            f"    core {c} (h in [{c * (H // cores)},{(c + 1) * (H // cores)})): "
            f"max_col_err vs Run-A={per_core[c].max().item():.3e}"
        )
    # Token-permutation probe (residual t-axis mixup?).
    if T > 1:
        tt = torch.stack(
            [(k[x] - a[y]).abs().max() for x in range(T) for y in range(T)]
        ).reshape(T, T)
        best = tt.argmin(dim=1)
        print(
            f"    token match sbufin[t]->runA[argmin]: {best.tolist()} "
            f"(identity [0..T-1] => tokens aligned; permuted => residual token-axis mixup)"
        )


# ---------------------------------------------------------------------------
# Cases: cores=2/T=2 is the PRIMARY (megakernel) config. Plus cores=1/T=2 and cores=2/T=1.
# ---------------------------------------------------------------------------
def main():
    results = {}
    print(
        "=== DeltaNet SBUF-in + out_in_sb=True seam -- FP32 (gate: allclose atol=1e-5 rtol=1e-2) ===\n"
    )
    for tag, T, cores, seed in (
        ("cores=2/T=2 (PRIMARY)", 2, 2, 2),
        ("cores=1/T=2", 2, 1, 2),
        ("cores=2/T=1", 1, 2, 1),
    ):
        ok, ker, ref, krunA = run_case(
            f"dn_{tag.split()[0].replace('/', '_')}", T=T, cores=cores, seed=seed
        )
        results[tag] = ok
        _diagnose_if_fail(tag, ok, ker, ref, krunA, T=T, cores=cores)
        print()

    print("=== SUMMARY (DeltaNet SBUF-in + out_in_sb=True) ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    if all(results.values()):
        print(
            "\nALL CASES PASSED -- DeltaNet SBUF-in + out_in_sb=True matches both golden and Run-A."
        )
    else:
        print(
            "\nDIVERGENCE DETECTED -- DeltaNet SBUF-in/out_in_sb path diverges (LIKELY the megakernel "
            "bug; see diagnostics above)."
        )


# pytest entrypoint for the primary case.
def test_deltanet_sbufin_fp32_t2_c2():
    ok, *_ = run_case("dn_T2_c2", T=2, cores=2, seed=2)
    assert ok, (
        "DeltaNet SBUF-in + out_in_sb=True diverged for the megakernel config (cores=2, T=2)"
    )


if __name__ == "__main__":
    main()
