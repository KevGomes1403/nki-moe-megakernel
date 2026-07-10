"""On-device correctness test for the FULLY FUSED MoE layer kernel (token generation).

Exercises ``moe_layer_compose`` -- the combine slice that stitches the three validated building blocks
into ONE SBUF-resident composable (norm ONCE -> routed_experts + shared_expert + sigma-gate -> gated
sum), zero HBM round-trip, returning the SINGLE per-rank partial ``routed_local + g * shared_local``
for one downstream ``reduce_from_tensor_model_parallel_region``.

Readback (authoritative) = SBUF megakernel-API tile assembled token-shard-aware: each LNC core DMAs ONLY
its shard slice into shared HBM [H0,T,H1], host un-permutes rmsnorm's layout-0 back to [T,H]. An HBM
natural readback ([T,H] via one AP DMA in the kernel) cross-checks. A tiny sigma-gate readback checks g.

Oracle = full NeuronMoEBlock-equivalent per-rank partial (fp32): normed=RMSNorm(hidden,gamma); routed
top-8 (fp32 softmax over E -> L1-normalize) affinity-weighted expert sum; shared SwiGLU; combined =
routed + sigmoid(sigma_gate_w . normed) * shared. Per-rank I = I_s = 128.

GATES (cosine is BANNED repo-wide as a gate/metric):
  * FP32 IO: torch.allclose(atol=1e-5, rtol=1e-2) -- the HARD gate on combined_local.
  * BF16 IO: max_abs(kernel_bf16 - oracle_fp32) <= floor * BF16_HEADROOM, floor = the output-rounding
    step max_abs(oracle_fp32.to(bf16) - oracle_fp32). Headroom >= routed's 12 (the full layer adds the
    shared + gate stages on top of the routed path). max_abs / max_rel reported for every case.
  * Sub-check: sigma-gate g scalar allclose vs oracle sigmoid.

Cases: cores=1 {T=1, T=2}, cores=2 {T=2} (the verify priority). cores=2/T=1 (decode) is SKIPPED:
the routed path needs an H-sharded norm at T=1 while the shared path needs the full tile -- the norm
layouts diverge and decode is deprioritized (see moe_layer.md / shared_expert.py docstring).

Run (CORES 0,1):
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_moe_layer_kernel
"""

import math
import sys
from pathlib import Path

import nki
import nki.isa as nisa
import nki.language as nl
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.qwen3_6_moe.nki_kernels.moe.components.moe_layer import (  # noqa: E402
    moe_layer_compose,
    sigma_gate_compose,
)
from models.qwen3_6_moe.nki_kernels.moe.components.post_attn_norm import (  # noqa: E402
    post_attn_rmsnorm_compose,
)
from models.qwen3_6_moe.nki_kernels.moe.components.shared_expert import (  # noqa: E402
    moe_tkg_shard_decision,
)

# Per-rank (TP=4) Qwen3.6-A3B MoE decode dims.
HIDDEN = 2048  # H
H0 = 128
E_FULL = 256  # experts
K_FULL = 8  # top-k
I_DIM = 128  # moe_intermediate_size(512) sharded on I over TP=4
I_S = 128  # shared_expert_intermediate_size(512) sharded on I over TP=4
EPS = 1e-6

ATOL = 1e-5
RTOL = 1e-2
# bf16 gate (cosine BANNED): max_abs(kernel_bf16 - oracle_fp32) <= floor * BF16_HEADROOM, floor = the
# output-rounding step max_abs(oracle_fp32.to(bf16) - oracle_fp32). Headroom >= the routed path's 12:
# combined = routed (norm -> softmax -> 8x gate/up/silu/down) + g * shared (norm -> 3 matmuls + SiLU).
BF16_HEADROOM = 16.0


# ---------------------------------------------------------------------------
# Inputs (weights fan-in scaled so activations stay O(1))
# ---------------------------------------------------------------------------
def make_inputs(T, E, K, seed):
    """Random fp32 logical inputs (B=1). Routed weights in kernel layout; shared/sigma in STORED layout
    (repacked by the harness). Returns a dict of everything the kernel + oracle need."""
    torch.manual_seed(seed)
    hidden = torch.randn(1, T, HIDDEN)
    gamma = (
        torch.randn(1, HIDDEN) * 0.02 + 1.0
    )  # post_attention_layernorm.weight (~1.0)
    # Routed (kernel layout).
    router_w = torch.randn(HIDDEN, E) / math.sqrt(HIDDEN)  # [H,E]
    gate_up_w = torch.randn(E, HIDDEN, 2, I_DIM) / math.sqrt(HIDDEN)  # [E,H,2,I]
    down_w = torch.randn(E, I_DIM, HIDDEN) / math.sqrt(I_DIM)  # [E,I,H]
    # Shared (stored layout: gate/up ColumnParallel [I_s,H], down RowParallel [H,I_s]).
    s_gate = torch.randn(I_S, HIDDEN) / math.sqrt(HIDDEN)  # [I_s,H]
    s_up = torch.randn(I_S, HIDDEN) / math.sqrt(HIDDEN)  # [I_s,H]
    s_down = torch.randn(HIDDEN, I_S) / math.sqrt(I_S)  # [H,I_s]
    # Sigma-gate (stored layout: nn.Linear(H,1) weight [1,H]).
    sigma_stored = torch.randn(1, HIDDEN) / math.sqrt(HIDDEN)  # [1,H]
    return {
        "hidden": hidden,
        "gamma": gamma,
        "router_w": router_w,
        "gate_up_w": gate_up_w,
        "down_w": down_w,
        "s_gate": s_gate,
        "s_up": s_up,
        "s_down": s_down,
        "sigma_stored": sigma_stored,
    }


def repack(inp):
    """Load-time repack of shared/sigma weights into the contraction-first kernel layout."""
    return {
        "s_gate_w": inp["s_gate"].t().contiguous(),  # [I_s,H] -> [H,I_s]
        "s_up_w": inp["s_up"].t().contiguous(),  # [I_s,H] -> [H,I_s]
        "s_down_w": inp["s_down"].t().contiguous(),  # [H,I_s] -> [I_s,H]
        "sigma_w": inp["sigma_stored"].t().contiguous(),  # [1,H] -> [H,1]
    }


# ---------------------------------------------------------------------------
# Oracles (fp32 ground truth)
# ---------------------------------------------------------------------------
def _rmsnorm(hidden, gamma):
    """Post-attn RMSNorm over full H (fp32), matching in-kernel rmsnorm_tkg. Returns normed [T,H]."""
    T = hidden.shape[0] * hidden.shape[1]
    x = hidden.reshape(T, HIDDEN).float()
    inv = (x.square().mean(-1, keepdim=True) + EPS).rsqrt()
    return (x * inv) * gamma.reshape(1, HIDDEN).float()


def _oracle_routed(normed, router_w, gate_up_w, down_w, K):
    """HF-style routed partial: fp32 softmax over E -> top-K -> L1-normalize, per selected expert
    down(silu(gate(x))*up(x)) affinity-weighted sum. Returns routed [T,H]."""
    T = normed.shape[0]
    logits = normed @ router_w.float()  # [T,E]
    probs = torch.softmax(logits, dim=-1)
    topv, topi = probs.topk(K, dim=-1)  # [T,K]
    affn = topv / topv.sum(dim=-1, keepdim=True)  # L1 normalize
    out = torch.zeros(T, HIDDEN)
    for t in range(T):
        for j in range(K):
            e = int(topi[t, j].item())
            g = normed[t] @ gate_up_w[e, :, 0, :].float()  # [I]
            u = normed[t] @ gate_up_w[e, :, 1, :].float()  # [I]
            inter = torch.nn.functional.silu(g) * u
            out[t] += affn[t, j] * (inter @ down_w[e].float())  # [H]
    return out


def _oracle_shared(normed, s_gate, s_up, s_down):
    """SwiGLU shared expert: down(silu(gate(x)) * up(x)) (per-rank partial). Returns shared [T,H]."""
    gate = normed @ s_gate.t().float()  # [T,I_s]
    up = normed @ s_up.t().float()  # [T,I_s]
    inter = torch.nn.functional.silu(gate) * up
    return inter @ s_down.t().float()  # [T,H]


def oracle_layer(inp, K):
    """Full MoE-layer per-rank partial (fp32): combined = routed + sigmoid(sigma.normed) * shared.
    Returns (combined [T,H], g [T,1])."""
    normed = _rmsnorm(inp["hidden"], inp["gamma"])  # [T,H]
    routed = _oracle_routed(normed, inp["router_w"], inp["gate_up_w"], inp["down_w"], K)
    shared = _oracle_shared(normed, inp["s_gate"], inp["s_up"], inp["s_down"])
    sigma_w = inp["sigma_stored"].t().float()  # [H,1]
    g = torch.sigmoid(normed @ sigma_w)  # [T,1]
    return routed + g * shared, g


# ---------------------------------------------------------------------------
# Layout helpers (SBUF readback)
# ---------------------------------------------------------------------------
def _unpermute(sbuf_out, T):
    """Assembled layout-0 [H0,T,H1] -> natural [T,H] (H = h0*H1 + j; tp102)."""
    return sbuf_out.permute(1, 0, 2).reshape(T, HIDDEN)


def _assemble_store(src_sb, out_hbm, T, H, intermediate):
    """Each LNC core DMAs ONLY its own valid slice into shared HBM [H0,T,H1] (token slice when
    token-sharded, else the full tile -- both cores write identical data in the full-compute case)."""
    shard_on_T, T_offset, T_per_shard = moe_tkg_shard_decision(T, H, intermediate)
    if shard_on_T:
        nisa.dma_copy(
            dst=out_hbm[:, T_offset : T_offset + T_per_shard, :],
            src=src_sb[:, T_offset : T_offset + T_per_shard, :],
        )
    else:
        nisa.dma_copy(dst=out_hbm[:, :, :], src=src_sb[:, :, :])


# ---------------------------------------------------------------------------
# Device kernels
# ---------------------------------------------------------------------------
@nki.jit
def moe_layer_asm_fwd(
    hidden,
    gamma,
    router_w,
    gate_up_w,
    down_w,
    sigma_w,
    s_gate_w,
    s_up_w,
    s_down_w,
    eps=1e-6,
    k=8,
):
    """combined_local as SBUF [H0,T,H1] (megakernel API), assembled into HBM [H0,T,H1] per-core."""
    combined = moe_layer_compose(
        hidden,
        gamma,
        router_w,
        gate_up_w,
        down_w,
        sigma_w,
        s_gate_w,
        s_up_w,
        s_down_w,
        eps,
        k,
        output_in_sbuf=True,
    )
    H0_, T, H1 = combined.shape
    out = nl.ndarray((H0_, T, H1), dtype=combined.dtype, buffer=nl.shared_hbm)
    _assemble_store(combined, out, T, H0_ * H1, gate_up_w.shape[3])
    return out


@nki.jit
def moe_layer_hbm_fwd(
    hidden,
    gamma,
    router_w,
    gate_up_w,
    down_w,
    sigma_w,
    s_gate_w,
    s_up_w,
    s_down_w,
    eps=1e-6,
    k=8,
):
    """combined_local as HBM [T,H] natural (kernel AP DMA; assembled across LNC cores)."""
    return moe_layer_compose(
        hidden,
        gamma,
        router_w,
        gate_up_w,
        down_w,
        sigma_w,
        s_gate_w,
        s_up_w,
        s_down_w,
        eps,
        k,
        output_in_sbuf=False,
    )


@nki.jit
def sigma_gate_fwd(hidden, gamma, sigma_w, eps=1e-6):
    """Sigma-gate readback: norm once (layout-0) -> g = sigmoid(sigma . normed) -> HBM [1,T]."""
    normed_sb = post_attn_rmsnorm_compose(
        hidden, gamma, eps, None, single_core_forced=True
    )
    g = sigma_gate_compose(normed_sb, sigma_w)  # [1,T] fp32
    _, T, _ = normed_sb.shape
    out = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=g)
    return out


# ---------------------------------------------------------------------------
# Device runners
# ---------------------------------------------------------------------------
def _dev(x, dtype, dev):
    return x.to(dtype).contiguous().to(dev)


def _args(inp, packed, dtype, dev):
    return (
        _dev(inp["hidden"], dtype, dev),
        _dev(inp["gamma"], dtype, dev),
        _dev(inp["router_w"], dtype, dev),
        _dev(inp["gate_up_w"], dtype, dev),
        _dev(inp["down_w"], dtype, dev),
        _dev(packed["sigma_w"], dtype, dev),
        _dev(packed["s_gate_w"], dtype, dev),
        _dev(packed["s_up_w"], dtype, dev),
        _dev(packed["s_down_w"], dtype, dev),
    )


def run_layer(inp, T, cores, dtype, mode, K):
    """mode='asm' -> SBUF-assembled [H0,T,H1] un-permuted to [T,H]; mode='hbm' -> natural [T,H]."""
    import torch_xla.core.xla_model as xm

    packed = repack(inp)
    dev = xm.xla_device()
    args = _args(inp, packed, dtype, dev)
    if mode == "hbm":
        out = moe_layer_hbm_fwd[cores](*args, EPS, K)
        return out.to(torch.float32).cpu().reshape(T, HIDDEN)
    out = moe_layer_asm_fwd[cores](*args, EPS, K)
    return _unpermute(out.to(torch.float32).cpu(), T)


def run_sigma(inp, T, cores, dtype):
    """Sigma-gate g readback -> [T,1] fp32."""
    import torch_xla.core.xla_model as xm

    packed = repack(inp)
    dev = xm.xla_device()
    out = sigma_gate_fwd[cores](
        _dev(inp["hidden"], dtype, dev),
        _dev(inp["gamma"], dtype, dev),
        _dev(packed["sigma_w"], dtype, dev),
        EPS,
    )
    return out.to(torch.float32).cpu().reshape(T, 1)


# ---------------------------------------------------------------------------
# Metrics / checks
# ---------------------------------------------------------------------------
def _metrics(ker, ref):
    kd, rd = ker.double(), ref.double()
    abs_err = (kd - rd).abs()
    denom = rd.abs().clamp_min(1e-4)
    return abs_err.max().item(), (abs_err / denom).max().item()


def _bf16_floor(ref_fp32):
    r = ref_fp32.float()
    return (r.to(torch.bfloat16).float() - r).abs().max().item()


def _check(name, ker, ref_fp32, dtype):
    """Gate combined_local against the fp32 oracle. Prints max_abs / max_rel for every case."""
    max_abs, max_rel = _metrics(ker, ref_fp32)
    if dtype == torch.float32:
        ok = torch.allclose(ker.double(), ref_fp32.double(), atol=ATOL, rtol=RTOL)
        gate = f"allclose(atol={ATOL} rtol={RTOL})"
        extra = ""
    else:
        floor = _bf16_floor(ref_fp32)
        limit = floor * BF16_HEADROOM
        ratio = max_abs / max(floor, 1e-30)
        ok = max_abs <= limit
        gate = f"max_abs<=floor*{BF16_HEADROOM:g}"
        extra = f"  floor={floor:.3e}  limit={limit:.3e}  ratio={ratio:.2f}x"
    status = "PASS" if ok else "FAIL"
    print(
        f"[{name}] {status}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  gate={gate}{extra}"
    )
    assert ok, (
        f"{name}: gate {gate} failed (max_abs={max_abs:.3e} max_rel={max_rel:.3e})"
    )


def _check_sigma(name, g_ker, g_ref):
    """Sub-check: sigma-gate g scalar allclose vs oracle sigmoid."""
    ok = torch.allclose(g_ker.double(), g_ref.double(), atol=1e-5, rtol=1e-3)
    max_abs = (g_ker.double() - g_ref.double()).abs().max().item()
    print(f"[{name}.sigma] sigmoid_allclose={ok}  max_abs={max_abs:.3e}")
    assert ok, f"{name}: sigma-gate g mismatch (max_abs={max_abs:.3e})"


# ---------------------------------------------------------------------------
# Case driver
# ---------------------------------------------------------------------------
def run_case(name, T, cores, seed, dtype, E=E_FULL, K=K_FULL, mode="asm"):
    """Full layer vs the NeuronMoEBlock-equivalent oracle + sigma-gate sub-check."""
    inp = make_inputs(T=T, E=E, K=K, seed=seed)
    ref, g_ref = oracle_layer(inp, K)
    # Sigma-gate sub-check (fp32, layout-independent -- always single-core-forced norm).
    g_ker = run_sigma(inp, T, cores=1, dtype=torch.float32)
    _check_sigma(name, g_ker, g_ref)
    ker = run_layer(inp, T, cores=cores, dtype=dtype, mode=mode, K=K)
    _check(name, ker, ref, dtype)
    return ker


# ---------------------------------------------------------------------------
# pytest entrypoints (verify priority: cores=1 {T=1,T=2}, cores=2 {T=2})
# ---------------------------------------------------------------------------
def test_fp32_t1_c1():
    run_case("fp32_T1_c1", T=1, cores=1, seed=1, dtype=torch.float32)


def test_fp32_t2_c1():
    run_case("fp32_T2_c1", T=2, cores=1, seed=2, dtype=torch.float32)


def test_fp32_t2_c2():
    run_case("fp32_T2_c2", T=2, cores=2, seed=3, dtype=torch.float32)  # verify priority


def test_fp32_t2_c1_hbm():
    run_case("fp32_T2_c1_HBM", T=2, cores=1, seed=2, dtype=torch.float32, mode="hbm")


def test_fp32_t2_c2_hbm():
    run_case("fp32_T2_c2_HBM", T=2, cores=2, seed=3, dtype=torch.float32, mode="hbm")


def test_bf16_t1_c1():
    run_case("bf16_T1_c1", T=1, cores=1, seed=1, dtype=torch.bfloat16)


def test_bf16_t2_c2():
    run_case(
        "bf16_T2_c2", T=2, cores=2, seed=3, dtype=torch.bfloat16
    )  # verify priority


def test_fp32_t2_c2_e16_fast():
    run_case("fp32_T2_c2_E16", T=2, cores=2, seed=7, dtype=torch.float32, E=16, K=8)


def test_skip_t1_c2_decode():
    """cores=2/T=1 (decode) is out of scope: routed needs an H-sharded norm at T=1 while shared needs
    the full tile -- the norm layouts diverge. Decode is deprioritized; skipped by design."""
    import pytest

    pytest.skip(
        "cores=2/T=1 decode: routed H-shard vs shared full-tile norm layout divergence"
    )


def main():
    print(
        "=== FUSED MoE LAYER -- SBUF megakernel-API readback vs full-layer oracle ==="
    )
    run_case("fp32_T1_c1", T=1, cores=1, seed=1, dtype=torch.float32)
    run_case("fp32_T2_c1", T=2, cores=1, seed=2, dtype=torch.float32)
    run_case("fp32_T2_c2", T=2, cores=2, seed=3, dtype=torch.float32)  # verify priority

    print(
        "\n=== FUSED MoE LAYER -- HBM natural readback (layout-independent cross-check) ==="
    )
    run_case("fp32_T2_c1_HBM", T=2, cores=1, seed=2, dtype=torch.float32, mode="hbm")
    run_case("fp32_T2_c2_HBM", T=2, cores=2, seed=3, dtype=torch.float32, mode="hbm")

    print(
        f"\n=== FUSED MoE LAYER -- BF16 (gate: max_abs <= floor * {BF16_HEADROOM:g}) ==="
    )
    run_case("bf16_T1_c1", T=1, cores=1, seed=1, dtype=torch.bfloat16)
    run_case(
        "bf16_T2_c2", T=2, cores=2, seed=3, dtype=torch.bfloat16
    )  # verify priority

    print("\n=== REDUCED-E fast case (E=16) ===")
    run_case("fp32_T2_c2_E16", T=2, cores=2, seed=7, dtype=torch.float32, E=16, K=8)

    print(
        "\n=== SKIPPED: cores=2/T=1 (decode) -- routed H-shard vs shared full-tile norm divergence ==="
    )
    print("\nALL HARD-GATED CASES PASSED")


if __name__ == "__main__":
    main()
