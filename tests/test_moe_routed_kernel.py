"""On-device correctness test for the routed-experts slice of the fused MoE layer kernel.

Exercises ``moe_routed_compose`` (post-attn RMSNorm -> router_topk -> selective moe_tkg, all
intermediates SBUF-resident with zero HBM round-trip) against a PyTorch oracle composed from the
nkilib torch references (router_topk_torch_ref + moe_tkg_torch_ref), cross-checked by a plain
HF-style routed reference.

Two readback paths share the norm+router+experts chain:
  * HBM  (output_in_sbuf=False): the kernel returns natural [T, H] and assembles across LNC cores in HBM.
    This is the AUTHORITATIVE correctness gate -- no layout assumption on the output. cores in {1, 2},
    covering both LNC work splits: token-shard (T>1) and H-shard (T==1).
  * SBUF (output_in_sbuf=True):  moe_tkg returns the megakernel-API SBUF [H0, T, H1]; the harness
    dma_copies it to HBM and the host un-permutes rmsnorm's [H0,T,H1] layout back to [T, H]. cores=1
    (at cores=2 each core holds only its H-shard; the cross-core gather belongs to the AR slice).

GATES (cosine is BANNED repo-wide as a gate/metric):
  * FP32 IO: torch.allclose(atol=1e-5, rtol=1e-2) -- the HARD gate. Wrong x_sb_layout / wrong H
    permutation is an O(1) mismatch here, not 1e-6.
  * BF16 IO: max_abs(kernel_bf16 - oracle_fp32) <= floor * BF16_HEADROOM, floor = the output-rounding
    step max_abs(oracle_fp32.to(bf16) - oracle_fp32). max_abs / max_rel reported for every case.
  * Discrete routing: router top-k expert INDEX exact-match (a wrong index is a hard fail) + top-k
    affinity allclose.

Real A3B routed dims: H=2048, E=256, K=8, I=128 (per-rank, TP=4). A reduced-E (E=16) fast case is
also run for quick iteration; at least one real-dims case is always exercised.

Run (CORES 0,1):
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_moe_routed_kernel
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

from nkilib.core.moe.moe_tkg.moe_tkg_torch import moe_tkg_torch_ref  # noqa: E402
from nkilib.core.router_topk.router_topk_torch import (  # noqa: E402
    router_topk_torch_ref,
)
from nkilib.core.utils.common_types import (  # noqa: E402
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)

from models.qwen3_6_moe.nki_kernels.moe.components.routed_experts import (  # noqa: E402
    moe_routed_compose,
)

# Per-rank (TP=4) Qwen3.6-A3B MoE decode dims.
P_MAX = 128
HIDDEN = 2048  # H
E_FULL = 256  # experts
K_FULL = 8  # top-k
I_DIM = 128  # moe_intermediate_size(512) sharded on I over TP=4
EPS = 1e-6

ATOL = 1e-5
RTOL = 1e-2
# bf16 gate (cosine BANNED): max_abs(kernel_bf16 - oracle_fp32) <= floor * BF16_HEADROOM, floor =
# the end-to-end output-rounding step max_abs(oracle_fp32.to(bf16) - oracle_fp32). Headroom covers the
# accumulated bf16 rounding of norm -> router(fp32 softmax) -> 8x (gate/up/silu/down) matmuls per token
# on top of that single output-rounding floor.
BF16_HEADROOM = 12.0


# ---------------------------------------------------------------------------
# Inputs (weights fan-in scaled so activations stay O(1))
# ---------------------------------------------------------------------------
def make_inputs(T, E, K, seed):
    """Random fp32 logical inputs (B=1). Returns hidden [1,T,H] RAW, gamma [1,H], router_w [H,E],
    expert_gate_up_w [E,H,2,I], expert_down_w [E,I,H] -- all in the kernel's layout (harness repack)."""
    torch.manual_seed(seed)
    hidden = torch.randn(1, T, HIDDEN)
    gamma = (
        torch.randn(1, HIDDEN) * 0.02 + 1.0
    )  # post_attention_layernorm.weight, standard form (~1.0)
    router_w = torch.randn(HIDDEN, E) / math.sqrt(
        HIDDEN
    )  # [H,E] (transpose of stored [E,H])
    gate_up_w = torch.randn(E, HIDDEN, 2, I_DIM) / math.sqrt(HIDDEN)  # [E,H,2,I]
    down_w = torch.randn(E, I_DIM, HIDDEN) / math.sqrt(I_DIM)  # [E,I,H]
    return hidden, gamma, router_w, gate_up_w, down_w


# ---------------------------------------------------------------------------
# Oracles (fp32 ground truth)
# ---------------------------------------------------------------------------
def _rmsnorm(hidden, gamma):
    """Post-attn RMSNorm over full H (fp32), matching in-kernel rmsnorm_tkg. Returns normed [T,H]."""
    T = hidden.shape[0] * hidden.shape[1]
    x = hidden.reshape(T, HIDDEN).float()
    inv = (x.square().mean(-1, keepdim=True) + EPS).rsqrt()
    return (x * inv) * gamma.reshape(1, HIDDEN).float()


def oracle_nkilib(hidden, gamma, router_w, gate_up_w, down_w, K):
    """Compose the nkilib torch refs: norm(fp32) -> router_topk_torch -> moe_tkg_torch (selective,
    POST_SCALE, SiLU). Returns (routed [T,H], index [T,K], affinities [T,E]) all fp32."""
    normed = _rmsnorm(hidden, gamma)  # [T,H]
    r = router_topk_torch_ref(
        x=normed,
        w=router_w.float(),
        w_bias=None,
        router_logits=None,
        expert_affinities=None,
        expert_index=None,
        act_fn=RouterActFnType.SOFTMAX,
        k=K,
        x_hbm_layout=1,  # x is [T,H]
        x_sb_layout=1,
        router_pre_norm=True,
        norm_topk_prob=True,
    )
    index = r["expert_index"]  # [T,K]
    affinities = r["expert_affinities"]  # [T,E] L1-normalized scattered
    m = moe_tkg_torch_ref(
        hidden_input=normed,
        expert_gate_up_weights=gate_up_w.float(),
        expert_down_weights=down_w.float(),
        expert_affinities=affinities,
        expert_index=index,
        is_all_expert=False,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        activation_fn=ActFnType.SiLU,
    )
    return m["out"].float(), index, affinities


def oracle_hf(hidden, gamma, router_w, gate_up_w, down_w, K):
    """Independent HF-style routed reference: fp32 softmax over E -> top-K -> L1-normalize the K, per
    selected expert down(silu(gate(x))*up(x)) affinity-weighted sum. Returns (routed [T,H], idx, affn)."""
    normed = _rmsnorm(hidden, gamma)  # [T,H]
    T = normed.shape[0]
    E = router_w.shape[1]
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
    return out, topi, affn


# ---------------------------------------------------------------------------
# Device kernels (routed-experts slice; readback for isolation only)
# ---------------------------------------------------------------------------
@nki.jit
def moe_routed_hbm_fwd(hidden, gamma, router_w, gate_up_w, down_w, eps=1e-6, k=8):
    """Return the routed partial as HBM [T, H] natural (moe_tkg assembles across LNC cores)."""
    routed_local, _ = moe_routed_compose(
        hidden, gamma, router_w, gate_up_w, down_w, eps, k, output_in_sbuf=False
    )
    return routed_local


@nki.jit
def moe_routed_sbuf_fwd(hidden, gamma, router_w, gate_up_w, down_w, eps=1e-6, k=8):
    """Return the megakernel-API SBUF [H0, T, H1] routed tile, dma_copied to HBM for readback."""
    routed_local, _ = moe_routed_compose(
        hidden, gamma, router_w, gate_up_w, down_w, eps, k, output_in_sbuf=True
    )
    H0, T, H1 = routed_local.shape
    out = nl.ndarray((H0, T, H1), dtype=routed_local.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=routed_local)
    return out


def _unpermute_sbuf_c1(sbuf_out, T):
    """cores=1 rmsnorm layout (num_H_shards=1, tp102): H = h0*H1 + j. [H0,T,H1] -> natural [T,H]."""
    return sbuf_out.permute(1, 0, 2).reshape(T, HIDDEN)


# ---------------------------------------------------------------------------
# Device runner
# ---------------------------------------------------------------------------
def run_kernel(inp, T, cores, dtype, sbuf_out, K):
    """Launch on `cores` cores; return the routed partial as fp32 [T, H]."""
    import torch_xla.core.xla_model as xm

    hidden, gamma, router_w, gate_up_w, down_w = inp
    dev = xm.xla_device()
    args = (
        hidden.to(dtype).contiguous().to(dev),
        gamma.to(dtype).contiguous().to(dev),
        router_w.to(dtype).contiguous().to(dev),
        gate_up_w.to(dtype).contiguous().to(dev),
        down_w.to(dtype).contiguous().to(dev),
    )
    if sbuf_out:
        out = moe_routed_sbuf_fwd[cores](*args, EPS, K)
        return _unpermute_sbuf_c1(out.to(torch.float32).cpu(), T)
    out = moe_routed_hbm_fwd[cores](*args, EPS, K)
    return out.to(torch.float32).cpu()  # [T, H]


# ---------------------------------------------------------------------------
# Metrics / checks
# ---------------------------------------------------------------------------
def _metrics(ker, ref):
    kd = ker.double()
    rd = ref.double()
    abs_err = (kd - rd).abs()
    denom = rd.abs().clamp_min(1e-4)
    return abs_err.max().item(), (abs_err / denom).max().item()


def _bf16_floor(ref_fp32):
    r = ref_fp32.float()
    return (r.to(torch.bfloat16).float() - r).abs().max().item()


def _check(name, ker, ref_fp32, dtype):
    """Gate the routed partial against the fp32 oracle. Prints max_abs / max_rel for every case."""
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


def _check_routing(name, idx_ref, affn_ref, idx_hf, affn_hf):
    """Discrete routing sub-check: top-K index exact-match (as a set) + top-K affinity allclose."""
    ref_sorted = torch.sort(idx_ref.long(), dim=-1).values
    hf_sorted = torch.sort(idx_hf.long(), dim=-1).values
    idx_ok = torch.equal(ref_sorted, hf_sorted)
    affn_ok = torch.allclose(
        torch.sort(affn_ref.float(), dim=-1).values,
        torch.sort(affn_hf.float(), dim=-1).values,
        atol=1e-5,
        rtol=1e-3,
    )
    print(f"[{name}.routing] index_match={idx_ok}  affinity_allclose={affn_ok}")
    assert idx_ok, f"{name}: router top-K index set mismatch (nkilib vs HF oracle)"
    assert affn_ok, f"{name}: router top-K affinity mismatch (nkilib vs HF oracle)"


def run_case(name, T, cores, seed, dtype, E=E_FULL, K=K_FULL, sbuf_out=False):
    """Run one case: kernel (routed partial) vs the composed nkilib oracle, cross-checked by HF."""
    inp = make_inputs(T=T, E=E, K=K, seed=seed)
    ref_routed, ref_idx, ref_affn = oracle_nkilib(*inp, K)
    hf_routed, hf_idx, hf_affn = oracle_hf(*inp, K)
    # Cross-check the two oracles agree (proves the semantic ground truth) before gating the kernel.
    assert torch.allclose(ref_routed, hf_routed, atol=1e-4, rtol=1e-2), (
        f"{name}: nkilib vs HF oracle disagree "
        f"(max_abs={(ref_routed - hf_routed).abs().max().item():.3e})"
    )
    _check_routing(name, ref_idx, ref_affn.gather(1, ref_idx.long()), hf_idx, hf_affn)

    ker = run_kernel(inp, T, cores=cores, dtype=dtype, sbuf_out=sbuf_out, K=K)
    _check(name, ker, ref_routed, dtype)
    return ker


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_fp32_t1_c1_e256():
    run_case("fp32_T1_c1_E256", T=1, cores=1, seed=1, dtype=torch.float32)


def test_fp32_t2_c1_e256():
    run_case("fp32_T2_c1_E256", T=2, cores=1, seed=2, dtype=torch.float32)


def test_fp32_t2_c2_e256():
    run_case("fp32_T2_c2_E256", T=2, cores=2, seed=3, dtype=torch.float32)


def test_fp32_t1_c2_e256():
    run_case("fp32_T1_c2_E256", T=1, cores=2, seed=1, dtype=torch.float32)  # H-shard


def test_fp32_t1_c1_sbuf_e256():
    run_case(
        "fp32_T1_c1_SBUF_E256", T=1, cores=1, seed=1, dtype=torch.float32, sbuf_out=True
    )


def test_fp32_t2_c1_sbuf_e256():
    run_case(
        "fp32_T2_c1_SBUF_E256", T=2, cores=1, seed=2, dtype=torch.float32, sbuf_out=True
    )


def test_bf16_t1_c1_e256():
    run_case("bf16_T1_c1_E256", T=1, cores=1, seed=1, dtype=torch.bfloat16)


def test_bf16_t2_c2_e256():
    run_case("bf16_T2_c2_E256", T=2, cores=2, seed=3, dtype=torch.bfloat16)


def test_bf16_t1_c2_e256():
    run_case("bf16_T1_c2_E256", T=1, cores=2, seed=1, dtype=torch.bfloat16)  # H-shard


def test_fp32_t2_c2_e16_fast():
    run_case("fp32_T2_c2_E16", T=2, cores=2, seed=7, dtype=torch.float32, E=16, K=8)


def main():
    print(
        "=== ROUTED-EXPERTS -- HBM readback (natural [T,H]); FP32 HARD gate allclose ==="
    )
    run_case("fp32_T1_c1_E256", T=1, cores=1, seed=1, dtype=torch.float32)
    run_case("fp32_T2_c1_E256", T=2, cores=1, seed=2, dtype=torch.float32)
    run_case(
        "fp32_T2_c2_E256", T=2, cores=2, seed=3, dtype=torch.float32
    )  # LNC2 token-shard
    run_case(
        "fp32_T1_c2_E256", T=1, cores=2, seed=1, dtype=torch.float32
    )  # LNC2 H-shard

    print(
        "\n=== ROUTED-EXPERTS -- SBUF readback (megakernel API, un-permuted); FP32 HARD gate ==="
    )
    run_case(
        "fp32_T1_c1_SBUF_E256", T=1, cores=1, seed=1, dtype=torch.float32, sbuf_out=True
    )
    run_case(
        "fp32_T2_c1_SBUF_E256", T=2, cores=1, seed=2, dtype=torch.float32, sbuf_out=True
    )

    print(
        f"\n=== ROUTED-EXPERTS -- BF16 (gate: max_abs <= floor * {BF16_HEADROOM:g}) ==="
    )
    run_case("bf16_T1_c1_E256", T=1, cores=1, seed=1, dtype=torch.bfloat16)
    run_case(
        "bf16_T2_c2_E256", T=2, cores=2, seed=3, dtype=torch.bfloat16
    )  # LNC2 token-shard
    run_case(
        "bf16_T1_c2_E256", T=1, cores=2, seed=1, dtype=torch.bfloat16
    )  # LNC2 H-shard

    print("\n=== REDUCED-E fast case (E=16) ===")
    run_case("fp32_T2_c2_E16", T=2, cores=2, seed=7, dtype=torch.float32, E=16, K=8)

    print("\nALL CASES PASSED")


if __name__ == "__main__":
    main()
