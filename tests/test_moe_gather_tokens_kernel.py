"""Diagnostic device test for the MoE token-shard cross-core gather (all_reduce_gather_tokens).

This is a PURE gather-logic test -- NO MoE kernel is run. It validates the LNC token sendrecv +
offset (other_off / other_len) + copy logic in ``all_reduce_gather_tokens`` (qwen36_verify_megakernel
lines ~88-118), the last standalone-testable novel megakernel piece.

Like the attention gather, that function's nccl.all_reduce cannot be loaded in a single-process launch
(the Neuron CCOM collective needs an initialized multi-rank world -- verified in
test_gqa_out_in_sb_kernel). At single-process TP=1 the all_reduce is an IDENTITY no-op (sum over 1
rank), so the REAL all_reduce_gather_tokens is called with ``rg=None``, which skips only the
collective and runs every other line unchanged. This validates the token sendrecv + offset + copy --
the novel part -- against a deterministic host reference.

Design (synthetic, deterministic): H0=128, H1=16, T=2, cores in {1,2}. A host reference full tile
``full[H0,T,H1]`` holds values distinct per (h0,token,h1) so any token-swap/offset error is obvious.
Each core is fed a TOKEN-SHARDED input: core c holds ``full`` only in its own token block
[T_offset_c:T_offset_c+T_len_c]; ALL other token columns are SENTINEL (-999) to prove the gather never
reads them. After the gather BOTH cores must reconstruct the full tile exactly (max_abs=0).

Gate: fp32 torch.allclose(atol=1e-5, rtol=1e-2) (exact expected). Prints max_abs/max_rel; on failure,
reports WHICH token/column is wrong (swap? sentinel leak? wrong offset?). NO cosine.

Run (CORES 0,1):
    cd /home/ubuntu/trainium-model-translation && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_moe_gather_tokens_kernel
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

from models.qwen3_6_moe.nki_kernels.megakernel.qwen36_verify_megakernel import (  # noqa: E402
    all_reduce_gather_tokens,
)
from models.qwen3_6_moe.nki_kernels.moe.components.routed_experts import (  # noqa: E402
    MOE_BIG_CONFIG_HI,
    moe_token_shard,
)

P_MAX = 128
H0 = 128
H1 = 16
HIDDEN = H0 * H1  # 2048
# Any intermediate with H*intermediate < MOE_BIG_CONFIG_HI forces shard_on_T=True at n_prgs=2, T=2.
INTERMEDIATE = 768
assert HIDDEN * INTERMEDIATE < MOE_BIG_CONFIG_HI, "intermediate must force shard_on_T"
SENTINEL = -999.0
ATOL = 1e-5
RTOL = 1e-2


@nki.jit
def gather_tokens_fwd(inp):
    """Per-core token-sharded input ``inp[n_prgs, H0, T, H1]`` -> full [H0, T*H1] per core via the LNC
    token gather. Each core reads its own inp[prg_id] slice, computes its (T_offset, T_len) with the
    megakernel's moe_token_shard, gathers, and writes out[prg_id]. Returns out[n_prgs, H0, T*H1]."""
    n_prgs_in, h0, T, h1 = inp.shape
    _, n_prgs, prg_id = get_program_sharding_info()
    H = h0 * h1

    local_sb = nl.ndarray((h0, T, h1), dtype=inp.dtype, buffer=nl.sbuf)
    in_view = TensorView(inp).select(dim=0, index=prg_id).get_view()
    nisa.dma_copy(dst=local_sb, src=in_view)

    # EXACT megakernel shard decision (routed_experts.moe_token_shard), n_prgs/prg_id from SPMD.
    _shard_on_T, T_offset, T_len = moe_token_shard(T, H, INTERMEDIATE, n_prgs, prg_id)

    gathered = all_reduce_gather_tokens(local_sb, None, prg_id, n_prgs, T_offset, T_len)

    out = nl.ndarray((n_prgs, h0, T * h1), dtype=inp.dtype, buffer=nl.shared_hbm)
    out_view = TensorView(out).select(dim=0, index=prg_id).get_view()
    nisa.dma_copy(dst=out_view, src=gathered)
    return out


# ---------------------------------------------------------------------------
# Host reference + input construction
# ---------------------------------------------------------------------------
def _full_ref(T):
    """Deterministic full tile [H0, T, H1]: value distinct per (h0, token, h1)."""
    p = torch.arange(H0).view(H0, 1, 1).float()
    t = torch.arange(T).view(1, T, 1).float()
    h1 = torch.arange(H1).view(1, 1, H1).float()
    return t * 1000.0 + h1 + p * 0.001  # [H0, T, H1]


def _build_input(T, cores):
    """[cores, H0, T, H1]: core c holds `full` only in its own token block, SENTINEL elsewhere (proves
    the gather never reads the other core's token columns). cores=1 -> full everywhere (passthrough)."""
    full = _full_ref(T)
    inp = torch.empty(cores, H0, T, H1)
    for c in range(cores):
        shard_on_T, off, ln = moe_token_shard(T, HIDDEN, INTERMEDIATE, cores, c)
        tile = full.clone()
        if shard_on_T:
            keep = torch.zeros(T, dtype=torch.bool)
            keep[off : off + ln] = True
            tile[:, ~keep, :] = SENTINEL  # sentinel the tokens this core does NOT own
        inp[c] = tile
    return inp, full


# ---------------------------------------------------------------------------
# Runner + checks
# ---------------------------------------------------------------------------
def run_kernel(inp, cores):
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    out = gather_tokens_fwd[cores](inp.to(torch.float32).contiguous().to(dev))
    return out.float().cpu()  # [cores, H0, T*H1]


def _metrics(ker, ref):
    kd = ker.double().reshape(-1)
    rd = ref.double().reshape(-1)
    abs_err = (kd - rd).abs()
    denom = rd.abs().clamp_min(1e-4)
    return abs_err.max().item(), (abs_err / denom).max().item()


def run_case(name, T, cores, seed=0):
    """Run the token gather and assert BOTH cores reconstruct `full` exactly."""
    inp, full = _build_input(T, cores)
    out = run_kernel(inp, cores)  # [cores, H0, T*H1]

    print(f"--- {name} (cores={cores}, T={T}) ---")
    all_ok = True
    for c in range(cores):
        rec = out[c].reshape(H0, T, H1)
        max_abs, max_rel = _metrics(rec, full)
        ok = torch.allclose(rec.double(), full.double(), atol=ATOL, rtol=RTOL)
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{name} core{c}] {status}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
            f"gate=allclose(atol={ATOL} rtol={RTOL})"
        )
        if not ok:
            _diagnose(c, rec, full, T)
    return all_ok


def _diagnose(core, rec, full, T):
    """Characterize which token/column is wrong: swap? sentinel leak? wrong offset?"""
    r = rec.double()
    f = full.double()
    for t in range(T):
        te = (r[:, t, :] - f[:, t, :]).abs().max().item()
        marker = " <-- WRONG" if te > ATOL else ""
        # Does this token match a DIFFERENT source token (swap)? Or the sentinel?
        matches = [
            tt for tt in range(T) if torch.allclose(r[:, t, :], f[:, tt, :], atol=ATOL)
        ]
        is_sentinel = torch.allclose(
            r[:, t, :], torch.full_like(r[:, t, :], SENTINEL), atol=ATOL
        )
        note = ""
        if is_sentinel:
            note = " (SENTINEL LEAKED -- gather read an unowned column)"
        elif matches and matches != [t]:
            note = f" (matches source token{matches} -- TOKEN SWAP/wrong offset)"
        print(
            f"    core{core} out-token{t}: max_abs_vs_full[{t}]={te:.3e}{marker}{note}"
        )


def main():
    print(
        "=== MoE token-shard LNC gather (all_reduce omitted; identity at TP=1) -- FP32 ===\n"
    )
    results = {}
    results["cores=2/T=2 (PRIMARY)"] = run_case("moe_gather_T2_c2", T=2, cores=2)
    print()
    results["cores=1/T=2 (passthrough)"] = run_case("moe_gather_T2_c1", T=2, cores=1)
    print()

    print("=== SUMMARY (MoE token gather: sendrecv + offset + copy) ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    if all(results.values()):
        print(
            "\nALL CASES PASSED -- both cores reconstruct the full tile; the MoE token gather "
            "(sendrecv + offset + copy) is correct."
        )
    else:
        print(
            "\nDIVERGENCE DETECTED -- the MoE token gather mis-assembles tokens (LIKELY the megakernel "
            "bug; see per-token diagnostics above)."
        )


# pytest entrypoint for the primary case.
def test_moe_gather_tokens_fp32_t2_c2():
    ok = run_case("moe_gather_T2_c2", T=2, cores=2)
    assert ok, (
        "MoE token gather mis-assembled tokens for the megakernel config (cores=2, T=2)"
    )


if __name__ == "__main__":
    main()
