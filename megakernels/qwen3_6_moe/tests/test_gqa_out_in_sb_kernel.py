"""Diagnostic device test for the GQA fused layer's out_in_sb=True o_proj output path.

Purpose: verify whether ``gqa_fused_compose(out_in_sb=True)`` -- the transposed SBUF H-shard o_proj
output used ONLY by the verify-megakernel -- produces numerically correct output. The default
``gqa_fused_tkg_fwd`` entrypoint always uses out_in_sb=False (natural HBM [T, H]); there is no
public entrypoint exercising the SBUF-transposed path, so this test adds one.

The trusted PyTorch reference (``golden``, norm_in=True) and all input/weight/cache/mask construction
are IMPORTED VERBATIM from ``test_gqa_fused_layer_kernel`` -- a buggy re-derived reference would give a
false verdict. We compare the reconstructed out_in_sb=True output against BOTH the golden reference AND
the out_in_sb=False path from the existing entrypoint (same inputs); the (b) cross-check isolates
"does the transpose corrupt the data" from any golden mismatch.

Layout contract for out_in_sb=True (H=2048, H0=partition=128, n_prgs=LNC, H2=H//n_prgs//128 per shard):
  ``gqa_fused_compose`` returns SBUF ``o_out`` [H0=128, H2*T], free-axis h2-major t-minor
  (f = h2*T + t). Each LNC core c holds a DISJOINT hidden shard; core c's element o_out[h0, h2*T+t]
  is natural hidden index h = c*(H0*H2) + h0*H2 + h2, token t. Reconstruction is a CONCATENATE over
  cores (single-process/TP=1: no all-reduce): out_natural = readback.reshape(n_prgs,H0,H2,T)
  .permute(3,0,1,2).reshape(T,H).

Gate: fp32 torch.allclose(atol=1e-5, rtol=1e-2). max_abs / max_rel printed for every comparison; NO
cosine similarity (banned repo-wide).

Run (CORES 0,1):
    cd /home/ubuntu/trainium-model-translation && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_gqa_out_in_sb_kernel
"""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nki  # noqa: E402
import nki.collectives as nccl  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

from nkilib.core.utils.kernel_helpers import (  # noqa: E402
    get_program_sharding_info,
)
from nkilib.core.utils.tensor_view import TensorView  # noqa: E402

from models.qwen3_6_moe.nki_kernels.gqa.decode.fused_layer import (  # noqa: E402
    gqa_fused_compose,
    gqa_fused_tkg_fwd,
)
from models.qwen3_6_moe.nki_kernels.megakernel.qwen36_verify_megakernel import (  # noqa: E402
    all_reduce_gather_h,
    load_residual_to_sbuf,
)

# Reuse the trusted reference + input construction VERBATIM (do not re-derive).
from models.qwen3_6_moe.tests.test_gqa_fused_layer_kernel import (  # noqa: E402
    ATOL,
    EPS,
    HIDDEN,
    build_mask,
    golden,
    make_inputs,
    run_kernel,  # launches the out_in_sb=False path (gqa_fused_tkg_fwd)
    HEAD_DIM,
    RTOL,
)

P_MAX = 128


# ---------------------------------------------------------------------------
# New entrypoint: same signature as gqa_fused_tkg_fwd but out_in_sb=True, with a
# per-core readback of the SBUF H-shard into shared HBM [n_prgs, H0, H2*T].
# ---------------------------------------------------------------------------
@nki.jit
def gqa_fused_out_in_sb_fwd(
    hidden,
    qkv_w,
    gate_w,
    gamma_q,
    gamma_k,
    cos,
    sin,
    k_cache,
    v_cache,
    mask,
    o_proj_w,
    eps=1e-6,
    gamma_in=None,
):
    """Run the fused GQA layer with out_in_sb=True and read the per-core SBUF o_proj shard back to HBM.

    Returns (out [n_prgs, H0=128, H2*T], active_k, active_v). Each LNC core writes ONLY its own shard
    at index prg_id; the host reconstructs natural [T, H] by concatenating shards."""
    o_sb, active_k, active_v = gqa_fused_compose(
        hidden,
        qkv_w,
        gate_w,
        gamma_q,
        gamma_k,
        cos,
        sin,
        k_cache,
        v_cache,
        mask,
        o_proj_w,
        eps,
        kv_write_idx=None,
        gamma_in=gamma_in,
        out_in_sb=True,
    )
    h0, free = o_sb.shape  # [128, H2*T]
    _, n_prgs, prg_id = get_program_sharding_info()
    out = nl.ndarray((n_prgs, h0, free), dtype=o_sb.dtype, buffer=nl.shared_hbm)
    out_view = TensorView(out).select(dim=0, index=prg_id).get_view()
    nisa.dma_copy(dst=out_view, src=o_sb)
    return out, active_k, active_v


# ---------------------------------------------------------------------------
# SBUF-in megakernel seam: load the tp2013 residual with the megakernel's own
# load_residual_to_sbuf, then feed gqa_fused_compose SBUF-in + gamma_in + out_in_sb=True,
# EXACTLY as qwen36_verify_megakernel does. Same per-core readback + reconstruction.
# ---------------------------------------------------------------------------
@nki.jit
def gqa_fused_sbufin_fwd(
    hidden,
    qkv_w,
    gate_w,
    gamma_q,
    gamma_k,
    cos,
    sin,
    k_cache,
    v_cache,
    mask,
    o_proj_w,
    eps=1e-6,
    gamma_in=None,
):
    """Megakernel seam: RAW hidden [1,T,H] -> tp2013 SBUF residual [H0, T*H1] via the megakernel's
    load_residual_to_sbuf, reshaped to [H0,T,H1] and fed SBUF-in to gqa_fused_compose (gamma_in +
    out_in_sb=True). Reads each core's [H0, H1_shard*T] SBUF shard back to HBM [n_prgs, H0, H1_shard*T]."""
    B, S, H = hidden.shape
    T = B * S
    h0 = P_MAX
    h1 = H // h0
    _, n_prgs, prg_id = get_program_sharding_info()

    residual = nl.ndarray((h0, T * h1), dtype=hidden.dtype, buffer=nl.sbuf)
    load_residual_to_sbuf(residual, hidden, T, h0, h1, n_prgs)
    x_sb = residual.reshape((h0, T, h1))

    o_sb, active_k, active_v = gqa_fused_compose(
        x_sb,
        qkv_w,
        gate_w,
        gamma_q,
        gamma_k,
        cos,
        sin,
        k_cache,
        v_cache,
        mask,
        o_proj_w,
        eps,
        gamma_in=gamma_in,
        out_in_sb=True,
        name_prefix="sbufin_",
    )
    out_h0, free = o_sb.shape
    out = nl.ndarray((n_prgs, out_h0, free), dtype=o_sb.dtype, buffer=nl.shared_hbm)
    out_view = TensorView(out).select(dim=0, index=prg_id).get_view()
    nisa.dma_copy(dst=out_view, src=o_sb)
    return out, active_k, active_v


# ---------------------------------------------------------------------------
# On-device cross-core GATHER: run the REAL all_reduce_gather_h (LNC sendrecv +
# TensorView.rearrange) instead of host-reconstructing the two shards. This is the last
# untested megakernel code: every prior test rebuilt the full tile on the host.
#
# COLLECTIVE BLOCKER (verified): nccl.all_reduce cannot be loaded in this single-process TP=1 launch
# -- rg=nccl.ReplicaGroup(([0],)) COMPILES but fails at NEFF LOAD ("TDRV build_enc_ctx_replica_group
# failed to init collectives comms" -> "LoadCollectives: NEFF is invalid"). A minimal 1-op all_reduce
# repro fails identically: the Neuron CCOM collective needs an initialized multi-rank world (as NxDI
# sets up under a 4-rank torchrun), which a bare xm.xla_device() does not provide.
# gqa_gather_real_cc_fwd below reproduces this; it is NOT run by main() (it aborts the process).
# The all_reduce is an inert identity at TP=1 anyway (sum over 1 rank), which is why
# all_reduce_gather_h takes rg=None to skip it.
#
# So gqa_gather_fwd validates the DECISIVE untested logic -- the on-device LNC sendrecv +
# TensorView.rearrange (OUR code) -- by calling the REAL all_reduce_gather_h with rg=None, which
# skips only the inert TP=1 all_reduce and runs every post-collective line unchanged.
# ---------------------------------------------------------------------------
@nki.jit
def gqa_gather_fwd(
    hidden,
    qkv_w,
    gate_w,
    gamma_q,
    gamma_k,
    cos,
    sin,
    k_cache,
    v_cache,
    mask,
    o_proj_w,
    eps=1e-6,
    gamma_in=None,
):
    """gqa_fused_compose(out_in_sb=True) -> on-device LNC gather (sendrecv + rearrange) via all_reduce_gather_h.
    Returns the full [H0, T*H1] tp2013 tile in HBM (both cores hold identical data after the gather)."""
    o_sb, active_k, active_v = gqa_fused_compose(
        hidden,
        qkv_w,
        gate_w,
        gamma_q,
        gamma_k,
        cos,
        sin,
        k_cache,
        v_cache,
        mask,
        o_proj_w,
        eps,
        kv_write_idx=None,
        gamma_in=gamma_in,
        out_in_sb=True,
        name_prefix="gh_",
    )
    _, S, H = hidden.shape
    T = S  # B == 1
    h0 = P_MAX
    h1 = H // h0
    _, n_prgs, prg_id = get_program_sharding_info()
    h1_shard = h1 // n_prgs

    gathered = all_reduce_gather_h(
        o_sb.reshape((h0, h1_shard * T)), None, prg_id, n_prgs, T
    )  # full [H0, T*H1] tp2013, held on both cores

    out = nl.ndarray((h0, T * h1), dtype=hidden.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=gathered)  # both cores write identical full tile
    return out, active_k, active_v


@nki.jit
def gqa_gather_real_cc_fwd(
    hidden,
    qkv_w,
    gate_w,
    gamma_q,
    gamma_k,
    cos,
    sin,
    k_cache,
    v_cache,
    mask,
    o_proj_w,
    eps=1e-6,
    gamma_in=None,
):
    """Reproduces the COLLECTIVE BLOCKER: calls the REAL all_reduce_gather_h (nccl.all_reduce +
    sendrecv + rearrange) EXACTLY as qwen36_verify_megakernel.py ~line 247. Fails at NEFF LOAD in a
    single-process launch (collectives comms uninitialized). NOT called by main() -- it aborts the
    process. Provided so the blocker is reproducible on demand."""
    o_sb, active_k, active_v = gqa_fused_compose(
        hidden,
        qkv_w,
        gate_w,
        gamma_q,
        gamma_k,
        cos,
        sin,
        k_cache,
        v_cache,
        mask,
        o_proj_w,
        eps,
        kv_write_idx=None,
        gamma_in=gamma_in,
        out_in_sb=True,
        name_prefix="ghcc_",
    )
    _, S, H = hidden.shape
    T = S
    h0 = P_MAX
    h1 = H // h0
    _, n_prgs, prg_id = get_program_sharding_info()
    h1_shard = h1 // n_prgs
    rg = nccl.ReplicaGroup(
        ([0],)
    )  # 1-rank group: unloadable standalone (see COLLECTIVE BLOCKER)
    gathered = all_reduce_gather_h(
        o_sb.reshape((h0, h1_shard * T)), rg, prg_id, n_prgs, T
    )
    out = nl.ndarray((h0, T * h1), dtype=hidden.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=gathered)
    return out, active_k, active_v


def _reconstruct_from_gathered(gathered, T, n_prgs):
    """Full tp2013 [H0, T*H1] gather output -> natural [T, HIDDEN].

    tp2013 free = t*H1 + lnc*H1_shard + h2; natural h = lnc*(H0*H1_shard) + h0*H1_shard + h2. Reshape
    free (T*H1) to (T, n_prgs, H1_shard), permute [h0,t,lnc,h2] -> [t,lnc,h0,h2], flatten -> [T, H].
    This is the inverse of load_residual_to_sbuf / store_residual_to_hbm's rearrange."""
    h0, free = gathered.shape
    h1 = free // T
    h1_shard = h1 // n_prgs
    assert h0 == P_MAX and h1_shard * n_prgs == h1 and n_prgs * h0 * h1_shard == HIDDEN
    g = gathered.reshape(h0, T, n_prgs, h1_shard)  # [h0, t, lnc, h2]
    return g.permute(1, 2, 0, 3).reshape(T, HIDDEN)  # [t, lnc, h0, h2] -> [T, H]


def run_kernel_gather(inp, T, L, cores, dtype):
    """Launch the on-device-gather entrypoint and reconstruct natural [T, HIDDEN] from the full tile."""
    import torch_xla.core.xla_model as xm

    (
        hidden,
        qkv_w,
        gate_w,
        gamma_q,
        gamma_k,
        cos,
        sin,
        prior_k,
        prior_v,
        o_proj_w,
        gamma_in,
    ) = inp

    k_cache = torch.zeros(1, 1, HEAD_DIM, L)
    v_cache = torch.zeros(1, 1, L, HEAD_DIM)
    k_cache[0, 0, :, 0 : L - T] = prior_k.transpose(0, 1)
    v_cache[0, 0, 0 : L - T, :] = prior_v
    mask = build_mask(T, L)

    dev = xm.xla_device()
    gamma_in_dev = gamma_in.reshape(1, HIDDEN).to(dtype).contiguous().to(dev)
    out, _ak, _av = gqa_gather_fwd[cores](
        hidden.to(dtype).contiguous().to(dev),
        qkv_w.to(dtype).contiguous().to(dev),
        gate_w.to(dtype).contiguous().to(dev),
        gamma_q.to(dtype).contiguous().to(dev),
        gamma_k.to(dtype).contiguous().to(dev),
        cos.to(dtype).contiguous().to(dev),
        sin.to(dtype).contiguous().to(dev),
        k_cache.to(dtype).contiguous().to(dev),
        v_cache.to(dtype).contiguous().to(dev),
        mask.to(dev),
        o_proj_w.to(dtype).contiguous().to(dev),
        EPS,
        gamma_in=gamma_in_dev,
    )
    gathered = out.to(dtype).cpu().float()  # [H0, T*H1] tp2013
    return _reconstruct_from_gathered(gathered, T, cores)


# ---------------------------------------------------------------------------
# Device runner for the out_in_sb=True path (mirrors test_gqa_fused_layer_kernel.run_kernel).
# ---------------------------------------------------------------------------
def _run_entry(entry, inp, T, L, cores, dtype):
    """Build caches/mask (identical to run_kernel), launch ``entry[cores]`` (HBM-in or SBUF-in
    out_in_sb=True entrypoint), and reconstruct natural [T, H] from the per-core SBUF shards.
    norm_in is always True (gamma_in provided; hidden is RAW pre-norm)."""
    import torch_xla.core.xla_model as xm

    (
        hidden,
        qkv_w,
        gate_w,
        gamma_q,
        gamma_k,
        cos,
        sin,
        prior_k,
        prior_v,
        o_proj_w,
        gamma_in,
    ) = inp

    k_cache = torch.zeros(1, 1, HEAD_DIM, L)
    v_cache = torch.zeros(1, 1, L, HEAD_DIM)
    k_cache[0, 0, :, 0 : L - T] = prior_k.transpose(0, 1)  # BHDS prior
    v_cache[0, 0, 0 : L - T, :] = prior_v  # BHSD prior
    mask = build_mask(T, L)

    dev = xm.xla_device()
    gamma_in_dev = gamma_in.reshape(1, HIDDEN).to(dtype).contiguous().to(dev)
    out, _active_k, _active_v = entry[cores](
        hidden.to(dtype).contiguous().to(dev),
        qkv_w.to(dtype).contiguous().to(dev),
        gate_w.to(dtype).contiguous().to(dev),
        gamma_q.to(dtype).contiguous().to(dev),
        gamma_k.to(dtype).contiguous().to(dev),
        cos.to(dtype).contiguous().to(dev),
        sin.to(dtype).contiguous().to(dev),
        k_cache.to(dtype).contiguous().to(dev),
        v_cache.to(dtype).contiguous().to(dev),
        mask.to(dev),
        o_proj_w.to(dtype).contiguous().to(dev),
        EPS,
        gamma_in=gamma_in_dev,
    )
    readback = out.to(dtype).cpu().float()  # [n_prgs, H0, H2*T]
    return _reconstruct_natural(readback, T)


def run_kernel_out_in_sb(inp, T, L, cores, dtype):
    """HBM-in out_in_sb=True path (hidden fed directly as HBM [1,T,H])."""
    return _run_entry(gqa_fused_out_in_sb_fwd, inp, T, L, cores, dtype)


def run_kernel_sbufin(inp, T, L, cores, dtype):
    """SBUF-in megakernel-seam out_in_sb=True path (hidden -> tp2013 residual -> SBUF-in compose)."""
    return _run_entry(gqa_fused_sbufin_fwd, inp, T, L, cores, dtype)


def _reconstruct_natural(readback, T):
    """[n_prgs, H0, H2*T] -> natural [T, H].

    Free axis is h2-major t-minor (f = h2*T + t); natural hidden h = c*(H0*H2) + h0*H2 + h2.
    Reshape free to [H2, T], permute to [T, c, h0, h2], flatten (c, h0, h2) -> h."""
    n_prgs, h0, free = readback.shape
    h2 = free // T
    assert h0 == P_MAX and h2 * T == free
    assert n_prgs * h0 * h2 == HIDDEN, (
        f"shard decomposition {n_prgs}*{h0}*{h2} != H={HIDDEN}"
    )
    o = readback.reshape(n_prgs, h0, h2, T)  # [c, h0, h2, t]
    return o.permute(3, 0, 1, 2).reshape(T, HIDDEN)  # [t, (c,h0,h2)=h]


# ---------------------------------------------------------------------------
# Metrics / checks
# ---------------------------------------------------------------------------
def _metrics(ker, ref):
    kd = ker.double()
    rd = ref.double()
    abs_err = (kd - rd).abs()
    denom = rd.abs().clamp_min(1e-4)
    return abs_err.max().item(), (abs_err / denom).max().item()


def _check(name, ker, ref):
    max_abs, max_rel = _metrics(ker, ref)
    ok = torch.allclose(ker.double(), ref.double(), atol=ATOL, rtol=RTOL)
    status = "PASS" if ok else "FAIL"
    print(
        f"[{name}] {status}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
        f"gate=allclose(atol={ATOL} rtol={RTOL})"
    )
    return ok, max_abs, max_rel


def run_case(name, T, L, cores, seed):
    """Run the out_in_sb=True path and compare reconstructed [T,H] against (a) golden and
    (b) the out_in_sb=False path. All fp32, norm_in=True."""
    inp = make_inputs(T=T, L=L, seed=seed)
    ker_sb = run_kernel_out_in_sb(inp, T, L, cores=cores, dtype=torch.float32)  # [T,H]

    ref_o, _rk, _rv = golden(inp, T, L, torch.float32, norm_in=True)  # [T,H] golden
    ker_false, _fk, _fv = run_kernel(
        inp, T, L, cores=cores, dtype=torch.float32, norm_in=True
    )  # out_in_sb=False path
    ker_false = ker_false.reshape(T, HIDDEN)

    print(f"--- {name} (cores={cores}, T={T}, L={L}) ---")
    ok_g, _, _ = _check(f"{name} vs golden", ker_sb, ref_o)
    ok_f, _, _ = _check(f"{name} vs out_in_sb=False", ker_sb, ker_false)
    # Sanity: the out_in_sb=False path itself must match golden (proves inputs/reference are sound).
    _check(f"{name} (out_in_sb=False vs golden)", ker_false, ref_o)
    return ok_g and ok_f, ker_sb, ref_o, ker_false


def run_case_sbufin(name, T, L, cores, seed):
    """Megakernel seam: run the SBUF-in path (tp2013 residual via load_residual_to_sbuf) and compare
    reconstructed [T,H] against (a) golden and (b) the HBM-in out_in_sb=True path on the SAME inputs.
    (b) is decisive: HBM-in already passed, so an SBUF-in divergence is EXACTLY the residual-layout
    <-> composable SBUF-in read mismatch (the megakernel bug). All fp32, norm_in=True."""
    inp = make_inputs(T=T, L=L, seed=seed)
    ker_sbufin = run_kernel_sbufin(inp, T, L, cores=cores, dtype=torch.float32)  # [T,H]

    ref_o, _rk, _rv = golden(inp, T, L, torch.float32, norm_in=True)  # [T,H] golden
    ker_hbmin = run_kernel_out_in_sb(
        inp, T, L, cores=cores, dtype=torch.float32
    )  # HBM-in path

    print(f"--- {name} SBUF-IN (cores={cores}, T={T}, L={L}) ---")
    ok_g, _, _ = _check(f"{name} SBUF-in vs golden", ker_sbufin, ref_o)
    ok_h, _, _ = _check(f"{name} SBUF-in vs HBM-in", ker_sbufin, ker_hbmin)
    return ok_g and ok_h, ker_sbufin, ref_o, ker_hbmin


def run_case_gather(name, T, L, cores, seed):
    """On-device cross-core gather: run the REAL all_reduce_gather_h and compare the reconstructed
    [T,H] against (a) golden and (b) the HOST-reconstructed out_in_sb=True result (which already
    matched golden). (b) is decisive: if the on-device gather diverges from the host reconstruction
    of the SAME shards, the bug is in all_reduce_gather_h's on-device sendrecv/rearrange."""
    inp = make_inputs(T=T, L=L, seed=seed)
    ker_gather = run_kernel_gather(inp, T, L, cores=cores, dtype=torch.float32)  # [T,H]

    ref_o, _rk, _rv = golden(inp, T, L, torch.float32, norm_in=True)  # [T,H] golden
    ker_host = run_kernel_out_in_sb(
        inp, T, L, cores=cores, dtype=torch.float32
    )  # host reconstruction of the same shards

    print(f"--- {name} ON-DEVICE-GATHER (cores={cores}, T={T}, L={L}) ---")
    ok_g, _, _ = _check(f"{name} gather vs golden", ker_gather, ref_o)
    ok_h, _, _ = _check(f"{name} gather vs host-reconstruction", ker_gather, ker_host)
    return ok_g and ok_h, ker_gather, ref_o, ker_host


# ---------------------------------------------------------------------------
# Cases: cores=2/T=2/L=128 is the PRIMARY (megakernel) config. Plus cores=1/T=2 and cores=2/T=1.
# ---------------------------------------------------------------------------
def main():
    results = {}
    print(
        "=== out_in_sb=True diagnostic -- FP32 (gate: allclose atol=1e-5 rtol=1e-2) ===\n"
    )
    ok, ker, ref, kfalse = run_case("sb_T2_L128_c2", T=2, L=128, cores=2, seed=6)
    results["cores=2/T=2/L=128 (PRIMARY)"] = ok
    _diagnose_if_fail("cores=2/T=2/L=128", ok, ker, ref, kfalse, T=2, cores=2)
    print()

    ok, ker, ref, kfalse = run_case("sb_T2_L128_c1", T=2, L=128, cores=1, seed=2)
    results["cores=1/T=2/L=128"] = ok
    _diagnose_if_fail("cores=1/T=2/L=128", ok, ker, ref, kfalse, T=2, cores=1)
    print()

    ok, ker, ref, kfalse = run_case("sb_T1_L128_c2", T=1, L=128, cores=2, seed=5)
    results["cores=2/T=1/L=128"] = ok
    _diagnose_if_fail("cores=2/T=1/L=128", ok, ker, ref, kfalse, T=1, cores=2)
    print()

    print(
        "=== MEGAKERNEL SEAM: SBUF-in (tp2013 residual via load_residual_to_sbuf) ===\n"
    )
    sb_results = {}
    for tag, T, L, cores, seed in (
        ("cores=2/T=2/L=128 (PRIMARY)", 2, 128, 2, 6),
        ("cores=1/T=2/L=128", 2, 128, 1, 2),
        ("cores=2/T=1/L=128", 1, 128, 2, 5),
    ):
        ok, ker, ref, khbm = run_case_sbufin(
            f"seam_{tag.split()[0].replace('/', '_')}", T=T, L=L, cores=cores, seed=seed
        )
        sb_results[tag] = ok
        _diagnose_sbufin_if_fail(tag, ok, ker, ref, khbm, T=T, cores=cores)
        print()

    print(
        "=== MEGAKERNEL ON-DEVICE GATHER: LNC sendrecv + rearrange (all_reduce omitted -- see BLOCKER) ===\n"
    )
    gh_results = {}
    for tag, T, L, cores, seed in (
        ("cores=2/T=2/L=128 (PRIMARY)", 2, 128, 2, 6),
        ("cores=1/T=2/L=128", 2, 128, 1, 2),
        ("cores=2/T=1/L=128", 1, 128, 2, 5),
    ):
        ok, ker, ref, khost = run_case_gather(
            f"gather_{tag.split()[0].replace('/', '_')}",
            T=T,
            L=L,
            cores=cores,
            seed=seed,
        )
        gh_results[tag] = ok
        _diagnose_gather_if_fail(tag, ok, ker, ref, khost, T=T, cores=cores)
        print()

    print("=== SUMMARY ===")
    print("HBM-in out_in_sb=True (host reconstruction):")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print("SBUF-in megakernel seam (host reconstruction):")
    for k, v in sb_results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(
        "On-device gather (LNC sendrecv + rearrange; all_reduce inert/omitted at TP=1):"
    )
    for k, v in gh_results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_ok = (
        all(results.values()) and all(sb_results.values()) and all(gh_results.values())
    )
    if all_ok:
        print(
            "\nALL CASES PASSED -- out_in_sb=True, SBUF-in seam, AND on-device gather are all correct."
        )
    elif (
        all(results.values())
        and all(sb_results.values())
        and not all(gh_results.values())
    ):
        print(
            "\nON-DEVICE GATHER BUG -- shards are correct (host reconstruction passes) but "
            "all_reduce_gather_h's on-device sendrecv/rearrange diverges (see diagnostics above)."
        )
    elif all(results.values()) and not all(sb_results.values()):
        print(
            "\nMEGAKERNEL SEAM BUG -- HBM-in passes but SBUF-in diverges: the load_residual_to_sbuf "
            "tp2013 layout does NOT match the composable's SBUF-in read (see diagnostics above)."
        )
    else:
        print("\nDIVERGENCE DETECTED (see diagnostics above).")


def _diagnose_if_fail(tag, ok, ker, ref, kfalse, T, cores):
    """If a case fails, characterize the divergence pattern to aid diagnosis (do NOT edit the kernel)."""
    if ok:
        return
    print(f"  [DIAGNOSTIC {tag}] out_in_sb=True diverges. Characterizing pattern:")
    k = ker.double()
    r = ref.double()
    # Scale error? ratio of matching-magnitude elements.
    denom = r.abs().clamp_min(1e-6)
    ratio = k.abs() / denom
    print(
        f"    global |ker|/|ref| ratio: median={ratio.median().item():.4f} "
        f"min={ratio.min().item():.4f} max={ratio.max().item():.4f}"
    )
    # Is it a permutation? Check if the multiset of values matches (sorted abs).
    ks = k.flatten().sort().values
    rs = r.flatten().sort().values
    perm_absdiff = (ks - rs).abs().max().item()
    print(
        f"    sorted-value max_abs (permutation test): {perm_absdiff:.3e} "
        f"(small => values present but SCRAMBLED/permuted; large => magnitudes wrong)"
    )
    # Per-column (hidden) error to see if a specific shard/core is wrong.
    col_err = (k - r).abs().max(dim=0).values  # [H]
    H = col_err.shape[0]
    per_core = col_err.reshape(cores, H // cores)
    for c in range(cores):
        print(
            f"    core {c} (h in [{c * (H // cores)},{(c + 1) * (H // cores)})): "
            f"max_col_err={per_core[c].max().item():.3e}"
        )
    # Cross-check: does out_in_sb=False match golden (isolates whether golden/inputs are the issue)?
    ok_fg = torch.allclose(kfalse.double(), r, atol=ATOL, rtol=RTOL)
    print(
        f"    out_in_sb=False vs golden allclose: {ok_fg} "
        f"(True => bug is ISOLATED to the transpose/out_in_sb path)"
    )


def _diagnose_sbufin_if_fail(tag, ok, ker, ref, khbm, T, cores):
    """If the SBUF-in seam diverges, characterize the pattern vs both golden and the HBM-in path.
    The HBM-in comparison is decisive: it isolates the residual-layout <-> SBUF-in-read mismatch."""
    if ok:
        return
    print(f"  [DIAGNOSTIC {tag}] SBUF-in seam diverges. Characterizing pattern:")
    k = ker.double()
    r = ref.double()
    h = khbm.double()
    # vs HBM-in: does HBM-in itself still match golden? (True => the fault is ONLY in the SBUF-in read.)
    hbm_ok = torch.allclose(h, r, atol=ATOL, rtol=RTOL)
    print(
        f"    HBM-in vs golden allclose: {hbm_ok} "
        f"(True => divergence is ISOLATED to the tp2013 residual <-> SBUF-in read)"
    )
    # Scale error?
    denom = h.abs().clamp_min(1e-6)
    ratio = k.abs() / denom
    print(
        f"    global |sbufin|/|hbmin| ratio: median={ratio.median().item():.4f} "
        f"min={ratio.min().item():.4f} max={ratio.max().item():.4f}"
    )
    # Permutation test against the HBM-in reference (values present but reordered?).
    ks = k.flatten().sort().values
    hs = h.flatten().sort().values
    perm_absdiff = (ks - hs).abs().max().item()
    print(
        f"    sorted-value max_abs vs HBM-in (permutation test): {perm_absdiff:.3e} "
        f"(small => values present but SCRAMBLED/permuted; large => magnitudes wrong)"
    )
    # Per-core hidden-shard error vs HBM-in (is one core's shard swapped/wrong?).
    col_err = (k - h).abs().max(dim=0).values  # [H]
    H = col_err.shape[0]
    per_core = col_err.reshape(cores, H // cores)
    for c in range(cores):
        print(
            f"    core {c} (h in [{c * (H // cores)},{(c + 1) * (H // cores)})): "
            f"max_col_err vs HBM-in={per_core[c].max().item():.3e}"
        )
    # Token-permutation probe: does sbufin[t] match hbmin[t'] for some other t'? (residual t-axis mixup)
    if T > 1:
        tt = torch.stack(
            [(k[a] - h[b]).abs().max() for a in range(T) for b in range(T)]
        ).reshape(T, T)
        best = tt.argmin(dim=1)
        print(
            f"    token match sbufin[t]->hbmin[argmin]: {best.tolist()} "
            f"(identity [0..T-1] => tokens aligned; permuted => residual token-axis mixup)"
        )


# pytest entrypoint for the primary case.
def test_out_in_sb_fp32_t2_c2_l128():
    ok, *_ = run_case("sb_T2_L128_c2", T=2, L=128, cores=2, seed=6)
    assert ok, (
        "out_in_sb=True path diverged for the megakernel config (cores=2, T=2, L=128)"
    )


def test_sbufin_seam_fp32_t2_c2_l128():
    ok, *_ = run_case_sbufin("seam_T2_L128_c2", T=2, L=128, cores=2, seed=6)
    assert ok, (
        "SBUF-in megakernel seam diverged for the megakernel config (cores=2, T=2, L=128)"
    )


def _diagnose_gather_if_fail(tag, ok, ker, ref, khost, T, cores):
    """If the on-device gather diverges, characterize the pattern vs both golden and the host
    reconstruction. The host comparison is decisive: it isolates all_reduce_gather_h's on-device
    sendrecv/rearrange (the shards are identical; only the assembly differs)."""
    if ok:
        return
    print(f"  [DIAGNOSTIC {tag}] on-device gather diverges. Characterizing pattern:")
    k = ker.double()
    r = ref.double()
    h = khost.double()
    # Host reconstruction of the SAME shards vs golden (True => fault is ONLY in the on-device gather).
    host_ok = torch.allclose(h, r, atol=ATOL, rtol=RTOL)
    print(
        f"    host-reconstruction vs golden allclose: {host_ok} "
        f"(True => divergence is ISOLATED to all_reduce_gather_h's sendrecv/rearrange)"
    )
    # Scale error?
    denom = h.abs().clamp_min(1e-6)
    ratio = k.abs() / denom
    print(
        f"    global |gather|/|host| ratio: median={ratio.median().item():.4f} "
        f"min={ratio.min().item():.4f} max={ratio.max().item():.4f}"
    )
    # Permutation test vs host (values present but reordered?).
    ks = k.flatten().sort().values
    hs = h.flatten().sort().values
    perm_absdiff = (ks - hs).abs().max().item()
    print(
        f"    sorted-value max_abs vs host (permutation test): {perm_absdiff:.3e} "
        f"(small => values present but SCRAMBLED/permuted; large => magnitudes wrong)"
    )
    # Per-core hidden-shard error vs host (is the sendrecv swapping cores' shards?).
    col_err = (k - h).abs().max(dim=0).values  # [H]
    H = col_err.shape[0]
    per_core = col_err.reshape(cores, H // cores)
    for c in range(cores):
        print(
            f"    core {c} (h in [{c * (H // cores)},{(c + 1) * (H // cores)})): "
            f"max_col_err vs host={per_core[c].max().item():.3e}"
        )
    # Does gather match host with the two cores' H-shards SWAPPED? (isolates a sendrecv core swap.)
    if cores == 2:
        h_swapped = torch.cat([h[:, H // 2 :], h[:, : H // 2]], dim=1)
        swap_ok = torch.allclose(k, h_swapped, atol=ATOL, rtol=RTOL)
        print(
            f"    gather vs host with cores' H-shards SWAPPED: {swap_ok} "
            f"(True => the sendrecv put each core's shard in the WRONG half)"
        )
    # Token-permutation probe (rearrange mixing t vs h1?).
    if T > 1:
        tt = torch.stack(
            [(k[a] - h[b]).abs().max() for a in range(T) for b in range(T)]
        ).reshape(T, T)
        best = tt.argmin(dim=1)
        print(
            f"    token match gather[t]->host[argmin]: {best.tolist()} "
            f"(identity [0..T-1] => aligned; permuted => rearrange mixed t vs h1)"
        )


def test_gather_fp32_t2_c2_l128():
    ok, *_ = run_case_gather("gather_T2_L128_c2", T=2, L=128, cores=2, seed=6)
    assert ok, (
        "On-device all_reduce_gather_h diverged for the megakernel config (cores=2, T=2, L=128)"
    )


if __name__ == "__main__":
    main()
