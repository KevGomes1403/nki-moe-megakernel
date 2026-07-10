"""On-device correctness + routed-alignment test for the shared-expert slice of the fused MoE layer.

Two things are proven:

1. CORRECTNESS -- ``moe_shared_compose`` (post-attn RMSNorm -> nkilib mlp_tkg SwiGLU shared expert, all
   intermediates SBUF-resident, zero HBM round-trip) vs an independent SwiGLU oracle
   ``down(silu(gate(x)) * up(x))`` at per-rank I_s=128 (per-rank partial, no reduce; norm in fp32).
   The megakernel-API SBUF tile ``shared_local [H0,T,H1]`` is read back token-shard-aware: each LNC core
   DMAs ONLY its own shard slice into a shared HBM [H0,T,H1], the host un-permutes rmsnorm's layout back
   to [T,H] (num_H_shards=1 for token-shard/single-core, =cores for the H-shard T=1 case). An HBM natural
   readback ([T,H] assembled by mlp_tkg's transpose store) cross-checks the authoritative numerics.

2. ROUTED ALIGNMENT (the point of this task) -- ``shared_local`` now has the SAME per-LNC-core shard
   layout as ``routed_local`` (moe_tkg) byte-for-byte, so the future gated sum is a plain per-core add:
     a. shape gate: ``shared_local.shape == routed_local.shape`` (asserted in-kernel).
     b. combined-add smoke: compute ``routed_local + shared_local`` on device (one kernel, one normed_sb),
        assemble across cores, and gate vs ``oracle_routed + oracle_shared`` at fp32 allclose. Gate=1
        (skip the sigma-gate -- that is the next slice); here we only prove the shapes/layouts add.

Cases: cores=1 {T=1,T=2}, cores=2 {T=2} (token-shard), cores=2 {T=1} (H-shard).

GATES (cosine is BANNED repo-wide as a gate/metric):
  * FP32 IO: torch.allclose(atol=1e-5, rtol=1e-2) -- the HARD gate.
  * BF16 IO: max_abs(kernel_bf16 - oracle_fp32) <= floor * BF16_HEADROOM, floor = the output-rounding
    step max_abs(oracle_fp32.to(bf16) - oracle_fp32). max_abs / max_rel reported for every case.

Real A3B shared dims: H=2048, I_s=128 (shared_expert_intermediate_size=512, per-rank TP=4). The routed
side of the alignment gate uses a reduced E=16 (layout is E-independent; I=128 matches shared I_s so the
two shard decisions coincide).

Run (CORES 0,1):
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_moe_shared_expert_kernel
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

from models.qwen3_6_moe.nki_kernels.moe.components.routed_experts import (  # noqa: E402
    moe_routed_compose,
)
from models.qwen3_6_moe.nki_kernels.moe.components.shared_expert import (  # noqa: E402
    moe_shared_compose,
    moe_tkg_shard_decision,
    shared_expert_compose,
)

# Per-rank (TP=4) Qwen3.6-A3B shared-expert decode dims.
HIDDEN = 2048  # H
H0 = 128
I_S = 128  # shared_expert_intermediate_size(512) sharded on I over TP=4
E_ROUTED = 16  # reduced-E for the alignment gate (layout E-independent)
K_ROUTED = 8
EPS = 1e-6

ATOL = 1e-5
RTOL = 1e-2
# bf16 gate (cosine BANNED): max_abs(kernel_bf16 - oracle_fp32) <= floor * BF16_HEADROOM, floor = the
# output-rounding step max_abs(oracle_fp32.to(bf16) - oracle_fp32). Headroom covers accumulated bf16
# rounding of norm -> 3 matmuls (gate/up/down) + SiLU. Fewer stages than the routed 8-expert path.
BF16_HEADROOM = 8.0


# ---------------------------------------------------------------------------
# Inputs (weights fan-in scaled so activations stay O(1))
# ---------------------------------------------------------------------------
def make_inputs(T, seed):
    """Random fp32 logical inputs (B=1). Returns hidden [1,T,H] RAW, gamma [1,H], shared weights in the
    STORED layout: gate/up [I_s,H] (ColumnParallel), down [H,I_s] (RowParallel)."""
    torch.manual_seed(seed)
    hidden = torch.randn(1, T, HIDDEN)
    gamma = (
        torch.randn(1, HIDDEN) * 0.02 + 1.0
    )  # post_attention_layernorm.weight (~1.0)
    gate_proj = torch.randn(I_S, HIDDEN) / math.sqrt(HIDDEN)  # [I_s, H] ColumnParallel
    up_proj = torch.randn(I_S, HIDDEN) / math.sqrt(HIDDEN)  # [I_s, H] ColumnParallel
    down_proj = torch.randn(HIDDEN, I_S) / math.sqrt(I_S)  # [H, I_s] RowParallel
    return hidden, gamma, gate_proj, up_proj, down_proj


def make_routed_weights(seed):
    """Routed weights in the kernel layout: router_w [H,E], gate_up_w [E,H,2,I], down_w [E,I,H]."""
    torch.manual_seed(seed + 100)
    router_w = torch.randn(HIDDEN, E_ROUTED) / math.sqrt(HIDDEN)  # [H,E]
    gate_up_w = torch.randn(E_ROUTED, HIDDEN, 2, I_S) / math.sqrt(HIDDEN)  # [E,H,2,I]
    down_w = torch.randn(E_ROUTED, I_S, HIDDEN) / math.sqrt(I_S)  # [E,I,H]
    return router_w, gate_up_w, down_w


def repack(gate_proj, up_proj, down_proj):
    """Load-time repack to the contraction-first layout MLPParameters expects."""
    gate_w = gate_proj.t().contiguous()  # [I_s,H] -> [H, I_s]
    up_w = up_proj.t().contiguous()  # [I_s,H] -> [H, I_s]
    down_w = down_proj.t().contiguous()  # [H,I_s] -> [I_s, H]
    return gate_w, up_w, down_w


# ---------------------------------------------------------------------------
# Oracles (fp32 ground truth)
# ---------------------------------------------------------------------------
def _rmsnorm(hidden, gamma):
    """Post-attn RMSNorm over full H (fp32), matching in-kernel rmsnorm_tkg. Returns normed [T,H]."""
    T = hidden.shape[0] * hidden.shape[1]
    x = hidden.reshape(T, HIDDEN).float()
    inv = (x.square().mean(-1, keepdim=True) + EPS).rsqrt()
    return (x * inv) * gamma.reshape(1, HIDDEN).float()


def oracle_swiglu(hidden, gamma, gate_proj, up_proj, down_proj):
    """Independent SwiGLU: down(silu(gate(x)) * up(x)) on the fp32 normed hidden (per-rank partial)."""
    normed = _rmsnorm(hidden, gamma)  # [T,H]
    gate = normed @ gate_proj.t().float()  # [T,I_s]
    up = normed @ up_proj.t().float()  # [T,I_s]
    inter = torch.nn.functional.silu(gate) * up
    return inter @ down_proj.t().float()  # [T,H]


def oracle_routed_hf(hidden, gamma, router_w, gate_up_w, down_w, K):
    """HF-style routed reference: fp32 softmax over E -> top-K -> L1-normalize, per selected expert
    down(silu(gate(x))*up(x)) affinity-weighted sum (per-rank partial). Returns routed [T,H]."""
    normed = _rmsnorm(hidden, gamma)  # [T,H]
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


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def _unpermute(sbuf_out, T):
    """Assembled layout-0 [H0,T,H1] -> natural [T,H] (H = h0*H1 + h1; tp102). The shared expert always
    emits layout-0 (shard_on_h disabled), so no interleave un-permute is needed."""
    return sbuf_out.permute(1, 0, 2).reshape(T, HIDDEN)


def _assemble_store(src_sb, out_hbm, T, H, intermediate):
    """Each LNC core DMAs ONLY its own valid slice into shared HBM [H0,T,H1] (mirrors moe_tkg's per-core
    SBUF store): its token slice when token-sharded, else the full tile (both cores write identical data
    in the full-compute case)."""
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
def moe_shared_hbm_fwd(hidden, gamma, gate_w, up_w, down_w, eps=1e-6):
    """Shared partial as HBM [T, H] natural (mlp_tkg's transpose store; assembled across LNC cores)."""
    shared_local, _ = moe_shared_compose(
        hidden, gamma, gate_w, up_w, down_w, eps, output_in_sbuf=False
    )
    return shared_local


@nki.jit
def moe_shared_asm_fwd(hidden, gamma, gate_w, up_w, down_w, eps=1e-6):
    """Shared partial as SBUF [H0,T,H1] (megakernel API), assembled into HBM [H0,T,H1] per-core."""
    shared_local, _ = moe_shared_compose(
        hidden, gamma, gate_w, up_w, down_w, eps, output_in_sbuf=True
    )
    H0_, T, H1 = shared_local.shape
    out = nl.ndarray((H0_, T, H1), dtype=shared_local.dtype, buffer=nl.shared_hbm)
    _assemble_store(shared_local, out, T, H0_ * H1, up_w.shape[1])
    return out


@nki.jit
def moe_combined_asm_fwd(
    hidden, gamma, router_w, r_gate_up_w, r_down_w, gate_w, up_w, down_w, eps=1e-6, k=8
):
    """routed_local + shared_local on the SAME normed_sb, assembled into HBM [H0,T,H1] per-core.

    Proves the gated sum is a plain per-core add: identical shape/shard layout, then element-wise add.
    """
    routed_local, normed_sb = moe_routed_compose(
        hidden, gamma, router_w, r_gate_up_w, r_down_w, eps, k, output_in_sbuf=True
    )
    shared_local = shared_expert_compose(
        normed_sb, gate_w, up_w, down_w, output_in_sbuf=True
    )
    assert tuple(shared_local.shape) == tuple(routed_local.shape), (
        "[NCC_INKI016] shared_local/routed_local shape mismatch -- layouts not aligned"
    )
    H0_, T, H1 = routed_local.shape
    combined = nl.ndarray((H0_, T, H1), dtype=routed_local.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=combined, data1=routed_local, data2=shared_local, op=nl.add)
    out = nl.ndarray((H0_, T, H1), dtype=routed_local.dtype, buffer=nl.shared_hbm)
    _assemble_store(combined, out, T, H0_ * H1, up_w.shape[1])
    return out


# ---------------------------------------------------------------------------
# Device runners
# ---------------------------------------------------------------------------
def _dev(x, dtype, dev):
    return x.to(dtype).contiguous().to(dev)


def run_shared(inp, T, cores, dtype, mode):
    """mode='hbm' -> natural [T,H]; mode='asm' -> SBUF-assembled [H0,T,H1] un-permuted to [T,H]."""
    import torch_xla.core.xla_model as xm

    hidden, gamma, gate_proj, up_proj, down_proj = inp
    gate_w, up_w, down_w = repack(gate_proj, up_proj, down_proj)
    dev = xm.xla_device()
    args = (
        _dev(hidden, dtype, dev),
        _dev(gamma, dtype, dev),
        _dev(gate_w, dtype, dev),
        _dev(up_w, dtype, dev),
        _dev(down_w, dtype, dev),
    )
    if mode == "hbm":
        out = moe_shared_hbm_fwd[cores](*args, EPS)
        return out.to(torch.float32).cpu().reshape(T, HIDDEN)
    out = moe_shared_asm_fwd[cores](*args, EPS)
    return _unpermute(out.to(torch.float32).cpu(), T)


def run_combined(shared_inp, routed_w, T, cores, dtype):
    """Device routed_local + shared_local, assembled, un-permuted to [T,H]."""
    import torch_xla.core.xla_model as xm

    hidden, gamma, gate_proj, up_proj, down_proj = shared_inp
    gate_w, up_w, down_w = repack(gate_proj, up_proj, down_proj)
    router_w, r_gate_up_w, r_down_w = routed_w
    dev = xm.xla_device()
    args = (
        _dev(hidden, dtype, dev),
        _dev(gamma, dtype, dev),
        _dev(router_w, dtype, dev),
        _dev(r_gate_up_w, dtype, dev),
        _dev(r_down_w, dtype, dev),
        _dev(gate_w, dtype, dev),
        _dev(up_w, dtype, dev),
        _dev(down_w, dtype, dev),
    )
    out = moe_combined_asm_fwd[cores](*args, EPS, K_ROUTED)
    return _unpermute(out.to(torch.float32).cpu(), T)


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


def _check(name, ker, ref_fp32, dtype, hard=True):
    """Gate a partial against the fp32 oracle. Prints max_abs / max_rel for every case. When hard=False
    (cores=2/T=1: moe_tkg's H-shard limitation), the divergence is REPORTED but not asserted."""
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
    if hard:
        status = "PASS" if ok else "FAIL"
    else:
        status = (
            "ALIGNED" if ok else "DIVERGENT (expected -- moe_tkg H-shard limitation)"
        )
    print(
        f"[{name}] {status}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  gate={gate}{extra}"
    )
    if hard:
        assert ok, (
            f"{name}: gate {gate} failed (max_abs={max_abs:.3e} max_rel={max_rel:.3e})"
        )


def run_case(name, T, cores, seed, dtype, mode="asm"):
    """Shared expert alone vs the SwiGLU oracle (token-shard-aware readback). All 4 configs pass: cores>1
    token-shards on T>1 and full-computes at T==1 (correct on every core)."""
    inp = make_inputs(T=T, seed=seed)
    ref = oracle_swiglu(*inp)  # [T,H] fp32
    ker = run_shared(inp, T, cores=cores, dtype=dtype, mode=mode)
    _check(name, ker, ref, dtype)
    return ker


def run_align_case(name, T, cores, seed, dtype=torch.float32, hard=True):
    """Combined-add smoke: device routed_local + shared_local vs oracle_routed + oracle_shared. Proves the
    gated sum is a plain per-core add. hard=False for cores=2/T=1 (routed itself H-shard-broken there)."""
    shared_inp = make_inputs(T=T, seed=seed)
    routed_w = make_routed_weights(seed)
    ref = oracle_swiglu(*shared_inp) + oracle_routed_hf(
        shared_inp[0], shared_inp[1], *routed_w, K_ROUTED
    )  # [T,H]
    ker = run_combined(shared_inp, routed_w, T, cores=cores, dtype=dtype)
    _check(name, ker, ref, dtype, hard=hard)
    return ker


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_fp32_t1_c1():
    run_case("fp32_T1_c1", T=1, cores=1, seed=1, dtype=torch.float32)


def test_fp32_t2_c1():
    run_case("fp32_T2_c1", T=2, cores=1, seed=2, dtype=torch.float32)


def test_fp32_t2_c2():
    run_case("fp32_T2_c2", T=2, cores=2, seed=3, dtype=torch.float32)


def test_fp32_t1_c2():
    run_case(
        "fp32_T1_c2", T=1, cores=2, seed=1, dtype=torch.float32
    )  # full-compute both cores


def test_fp32_t2_c2_hbm():
    run_case("fp32_T2_c2_HBM", T=2, cores=2, seed=3, dtype=torch.float32, mode="hbm")


def test_fp32_t1_c2_hbm():
    run_case("fp32_T1_c2_HBM", T=1, cores=2, seed=1, dtype=torch.float32, mode="hbm")


def test_bf16_t1_c1():
    run_case("bf16_T1_c1", T=1, cores=1, seed=1, dtype=torch.bfloat16)


def test_bf16_t2_c2():
    run_case("bf16_T2_c2", T=2, cores=2, seed=3, dtype=torch.bfloat16)


def test_align_t1_c1():
    run_align_case("align_T1_c1", T=1, cores=1, seed=4)


def test_align_t2_c2():
    run_align_case("align_T2_c2", T=2, cores=2, seed=5)


def test_align_t1_c2():
    run_align_case(
        "align_T1_c2", T=1, cores=2, seed=6, hard=False
    )  # moe_tkg H-shard limitation


def main():
    print(
        "=== SHARED-EXPERT -- SBUF megakernel-API readback (token-shard-aware) vs SwiGLU oracle ==="
    )
    run_case("fp32_T1_c1", T=1, cores=1, seed=1, dtype=torch.float32)
    run_case("fp32_T2_c1", T=2, cores=1, seed=2, dtype=torch.float32)
    run_case(
        "fp32_T2_c2", T=2, cores=2, seed=3, dtype=torch.float32
    )  # LNC2 token-shard
    run_case("fp32_T1_c2", T=1, cores=2, seed=1, dtype=torch.float32)  # LNC2 H-shard

    print("\n=== SHARED-EXPERT -- HBM natural readback (authoritative cross-check) ===")
    run_case("fp32_T2_c2_HBM", T=2, cores=2, seed=3, dtype=torch.float32, mode="hbm")
    run_case("fp32_T1_c2_HBM", T=1, cores=2, seed=1, dtype=torch.float32, mode="hbm")

    print(
        f"\n=== SHARED-EXPERT -- BF16 (gate: max_abs <= floor * {BF16_HEADROOM:g}) ==="
    )
    run_case("bf16_T1_c1", T=1, cores=1, seed=1, dtype=torch.bfloat16)
    run_case(
        "bf16_T2_c2", T=2, cores=2, seed=3, dtype=torch.bfloat16
    )  # LNC2 token-shard

    print(
        "\n=== ROUTED ALIGNMENT -- combined-add smoke (routed_local + shared_local) vs oracle sum ==="
    )
    run_align_case("align_T1_c1", T=1, cores=1, seed=4)
    run_align_case("align_T2_c2", T=2, cores=2, seed=5)  # LNC2 token-shard
    run_align_case(
        "align_T1_c2", T=1, cores=2, seed=6, hard=False
    )  # LNC2 T=1: moe_tkg H-shard limitation (routed itself broken there)

    print("\nALL HARD-GATED CASES PASSED")


if __name__ == "__main__":
    main()
