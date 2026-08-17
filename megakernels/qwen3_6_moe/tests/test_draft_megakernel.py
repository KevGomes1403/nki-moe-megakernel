"""On-device correctness test for the Qwen3.6-A3B MTP draft megakernel (one launch per draft step).

Exercises ``build_draft_megakernel`` end to end: eh_proj front end -> GQA decoder layer with the
in-kernel input norm and the in-place KV scatter -> MoE FFN -> pre-final-norm carry -> vocab head +
greedy argmax. TP=1 (replica_groups=None, every TP collective the identity); the LNC launch is the
thing under test, so every case runs at cores=2 and the cheap ones repeat at cores=1.

Cases:
  (1) draft (T=1, with_lm_head): hidden allclose, tokens EXACT vs torch argmax, cache written slots
      allclose vs the oracle active K/V, every other slot bit-identical to the pre-launch cache.
  (3) replay (T=2, no lm_head): the same cache + hidden gates on the headless build.
  chaining: (1) then (3) over the same cache buffers must land the same cache as (3) alone -- the
      empirical form of the benign cross-launch write-after-write at k=1.

Oracle = ``Qwen36MTPDraft.forward``'s math in fp32 on CPU, assembled from the per-stage oracles the
component tests already gate against (eh_proj, GQA fused layer with a partial-cache mask, MoE layer,
LM head). Gate: torch.allclose(atol=1e-5, rtol=1e-2).

Run (CORES 0,1):
    cd /home/ubuntu/nki-moe && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    NEURON_CC_FLAGS="--target trn2 --lnc 2" \
    python -m megakernels.qwen3_6_moe.tests.test_draft_megakernel
"""

import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nki_kernels.megakernel.qwen36_draft_megakernel import (  # noqa: E402
    build_draft_megakernel,
    flatten_draft_args,
)

from megakernels.qwen3_6_moe.tests.test_eh_proj_kernel import (  # noqa: E402
    oracle as eh_proj_oracle,
)
from megakernels.qwen3_6_moe.tests.test_gqa_fused_layer_kernel import (  # noqa: E402
    ATOL,
    EPS,
    GATE_DIM,
    HEAD_DIM,
    HIDDEN,
    I_DIM,
    RTOL,
    VALUE_DIM,
    build_cos_sin,
)
from megakernels.qwen3_6_moe.tests.test_gqa_fused_layer_t1_kernel import (  # noqa: E402
    build_keep,
    build_mask,
    golden as gqa_golden,
)
from megakernels.qwen3_6_moe.tests.test_lm_head_kernel import (  # noqa: E402
    golden as lm_head_golden,
)
from megakernels.qwen3_6_moe.tests.test_moe_layer_kernel import (  # noqa: E402
    E_FULL,
    I_DIM as MOE_I,
    I_S,
    K_FULL,
    oracle_layer,
    repack,
)

H = HIDDEN
V = 8192  # fast-iteration vocab (embed rows and lm_head columns); tp_degree=1, so V_rank == V
L_CTX = 256  # KV cache length (n_positions)
COMMITTED_LEN = 150  # committed context; [COMMITTED_LEN, L-T) is garbage the mask excludes


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def make_inputs(T, L, committed_len, seed):
    """Random fp32 draft-step inputs at the per-rank (TP=4) A3B shapes, B=1.

    Weights are fan-in scaled so activations stay O(1). The KV cache is fully random so the
    untouched-slot check has something to catch; only slots [0, committed_len) are unmasked."""
    torch.manual_seed(seed)
    return {
        "ids": torch.randint(0, V, (1, T), dtype=torch.int32),
        "prev_hidden": torch.randn(1, T, H) * 0.5,
        "kv_write_idx": torch.tensor([[committed_len]], dtype=torch.int32),
        "cos_sin": build_cos_sin(torch.arange(committed_len, committed_len + T)),
        "embed_w": torch.randn(V, H) * 0.02,
        "gamma_e": torch.randn(1, H) * 0.02 + 1.0,
        "gamma_h": torch.randn(1, H) * 0.02 + 1.0,
        "eh_w": torch.randn(2 * H, H) * 0.02,
        "gamma_in": torch.randn(1, H) * 0.02 + 1.0,
        "qkv_w": torch.randn(H, I_DIM) / math.sqrt(H),
        "gate_w": torch.randn(H, GATE_DIM) / math.sqrt(H),
        "gamma_q": torch.randn(HEAD_DIM),
        "gamma_k": torch.randn(HEAD_DIM),
        "o_proj_w": torch.randn(VALUE_DIM, H) / math.sqrt(VALUE_DIM),
        "k_cache": torch.randn(1, 1, HEAD_DIM, L),
        "v_cache": torch.randn(1, 1, L, HEAD_DIM),
        "moe_gamma": torch.randn(1, H) * 0.02 + 1.0,
        "moe_router_w": torch.randn(H, E_FULL) / math.sqrt(H),
        "moe_gate_up_w": torch.randn(E_FULL, H, 2, MOE_I) / math.sqrt(H),
        "moe_down_w": torch.randn(E_FULL, MOE_I, H) / math.sqrt(MOE_I),
        "moe_sigma_stored": torch.randn(1, H) / math.sqrt(H),
        "moe_s_gate": torch.randn(I_S, H) / math.sqrt(H),
        "moe_s_up": torch.randn(I_S, H) / math.sqrt(H),
        "moe_s_down": torch.randn(H, I_S) / math.sqrt(I_S),
        "final_gamma": torch.randn(1, H) * 0.02 + 1.0,
        "lm_head_w": torch.randn(H, V) * (H**-0.5),
    }


def _moe_stored(inp):
    """The MoE oracle's input dict (stored weight layouts), less the hidden it is applied to."""
    return {
        "gamma": inp["moe_gamma"],
        "router_w": inp["moe_router_w"],
        "gate_up_w": inp["moe_gate_up_w"],
        "down_w": inp["moe_down_w"],
        "sigma_stored": inp["moe_sigma_stored"],
        "s_gate": inp["moe_s_gate"],
        "s_up": inp["moe_s_up"],
        "s_down": inp["moe_s_down"],
    }


# ---------------------------------------------------------------------------
# Oracle (fp32 on CPU; Qwen36MTPDraft.forward stage by stage)
# ---------------------------------------------------------------------------
def oracle(inp, T, L, keep, with_lm_head):
    """(hidden [T,H], tokens [T] or None, active_k [1,1,D,T], active_v [1,1,T,D]) in fp32.

    ``hidden`` is the PRE-final-norm residual, matching the kernel's carry contract."""
    fp32 = torch.float32
    cos, sin = inp["cos_sin"]

    seed = eh_proj_oracle(
        inp["ids"],
        inp["embed_w"],
        inp["prev_hidden"],
        inp["gamma_e"],
        inp["gamma_h"],
        inp["eh_w"],
    ).reshape(T, H)

    gqa_inp = (
        seed.reshape(1, T, H),
        inp["qkv_w"],
        inp["gate_w"],
        inp["gamma_q"],
        inp["gamma_k"],
        cos,
        sin,
        inp["k_cache"][0, 0, :, 0 : L - T].transpose(0, 1),  # prior K [L-T, D]
        inp["v_cache"][0, 0, 0 : L - T, :],  # prior V [L-T, D]
        inp["o_proj_w"],
        inp["gamma_in"],
    )
    attn, active_k, active_v = gqa_golden(gqa_inp, T, L, keep, fp32)
    residual = seed + attn

    moe_inp = dict(_moe_stored(inp), hidden=residual.reshape(1, T, H))
    combined, _ = oracle_layer(moe_inp, K_FULL)
    hidden = residual + combined

    tokens = None
    if with_lm_head:
        logits = lm_head_golden(
            hidden.reshape(1, T, H), inp["final_gamma"].reshape(H), inp["lm_head_w"], fp32
        )
        tokens = logits.argmax(dim=-1).to(torch.int32)
    return hidden, tokens, active_k, active_v


# ---------------------------------------------------------------------------
# Device runner
# ---------------------------------------------------------------------------
def run_kernel(inp, T, keep, cores, with_lm_head, k_cache=None, v_cache=None):
    """Launch the draft megakernel on `cores` cores; returns CPU (tokens|None, hidden, k, v).

    ``k_cache``/``v_cache`` override the input caches so a chained launch can consume the previous
    launch's mutated handles."""
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    cos, sin = inp["cos_sin"]

    def d(x):
        return x.to(torch.float32).contiguous().to(dev)

    gqa = {
        "in_gamma": d(inp["gamma_in"]),
        "qkv_w": d(inp["qkv_w"]),
        "gate_w": d(inp["gate_w"]),
        "gamma_q": d(inp["gamma_q"]),
        "gamma_k": d(inp["gamma_k"]),
        "o_proj_w": d(inp["o_proj_w"]),
        "k_cache": d(inp["k_cache"]) if k_cache is None else k_cache,
        "v_cache": d(inp["v_cache"]) if v_cache is None else v_cache,
    }
    packed = repack(_moe_stored(inp))
    moe = {
        "gamma": d(inp["moe_gamma"]),
        "router_w": d(inp["moe_router_w"]),
        "gate_up_w": d(inp["moe_gate_up_w"]),
        "down_w": d(inp["moe_down_w"]),
        "sigma_gate_w": d(packed["sigma_w"]),
        "shared_gate_w": d(packed["s_gate_w"]),
        "shared_up_w": d(packed["s_up_w"]),
        "shared_down_w": d(packed["s_down_w"]),
    }
    flat = flatten_draft_args(
        inp["ids"].to(dev),
        d(inp["prev_hidden"]),
        inp["kv_write_idx"].to(dev),
        d(cos),
        d(sin),
        build_mask(keep).to(dev),
        d(inp["embed_w"]),
        d(inp["gamma_e"]),
        d(inp["gamma_h"]),
        d(inp["eh_w"]),
        gqa,
        moe,
        EPS,
        None,
        final_gamma=d(inp["final_gamma"]) if with_lm_head else None,
        lm_head_w=d(inp["lm_head_w"]) if with_lm_head else None,
    )

    rets = build_draft_megakernel(with_lm_head)[cores](*flat)
    if with_lm_head:
        tokens, hidden, ret_k, ret_v, _, _ = rets
        return tokens.cpu().reshape(T), hidden.float().cpu().reshape(T, H), ret_k, ret_v
    hidden, ret_k, ret_v, _, _ = rets
    return None, hidden.float().cpu().reshape(T, H), ret_k, ret_v


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def _check(name, ker, ref):
    kd, rd = ker.double(), ref.double()
    abs_err = (kd - rd).abs()
    max_abs = abs_err.max().item()
    max_rel = (abs_err / rd.abs().clamp_min(1e-4)).max().item()
    ok = torch.allclose(kd, rd, atol=ATOL, rtol=RTOL)
    print(
        f"[{name}] {'PASS' if ok else 'FAIL'}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
    )
    assert ok, f"{name}: allclose(atol={ATOL} rtol={RTOL}) failed (max_abs={max_abs:.3e})"


def _bitexact(name, a, b):
    ok = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    print(f"[{name}] {'PASS' if ok else 'FAIL'}  bit-exact  shape={tuple(a.shape)}")
    assert ok, f"{name}: not bit-exact"


def _check_cache(name, ker_k, ker_v, inp, ref_k, ref_v, T, idx):
    """Every slot but [idx, idx+T) is untouched; the written ones match the oracle active K/V.

    Structure before values: the untouched-slot check is independent of numerics, so it runs first
    and a tolerance miss on the window cannot mask a stray write."""
    rk, ik = ker_k.clone(), inp["k_cache"].clone()
    rk[0, 0, :, idx : idx + T] = 0
    ik[0, 0, :, idx : idx + T] = 0
    _bitexact(f"{name}.kcache_untouched", rk, ik)
    rv, iv = ker_v.clone(), inp["v_cache"].clone()
    rv[0, 0, idx : idx + T, :] = 0
    iv[0, 0, idx : idx + T, :] = 0
    _bitexact(f"{name}.vcache_untouched", rv, iv)
    _check(
        f"{name}.kcache", ker_k[0, 0, :, idx : idx + T].reshape(1, 1, HEAD_DIM, T), ref_k
    )
    _check(
        f"{name}.vcache", ker_v[0, 0, idx : idx + T, :].reshape(1, 1, T, HEAD_DIM), ref_v
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def run_case(name, T, cores, seed, with_lm_head, L=L_CTX, committed_len=COMMITTED_LEN):
    inp = make_inputs(T=T, L=L, committed_len=committed_len, seed=seed)
    keep = build_keep(T, L, committed_len)
    ker_tokens, ker_hidden, ret_k, ret_v = run_kernel(inp, T, keep, cores, with_lm_head)
    ref_hidden, ref_tokens, ref_k, ref_v = oracle(inp, T, L, keep, with_lm_head)

    _check(f"{name}.hidden", ker_hidden, ref_hidden)
    if with_lm_head:
        ok = torch.equal(ker_tokens, ref_tokens)
        print(
            f"[{name}.tokens] {'PASS' if ok else 'FAIL'}  kernel={ker_tokens.tolist()}  "
            f"oracle={ref_tokens.tolist()}"
        )
        assert ok, f"{name}: token mismatch"
    _check_cache(
        name, ret_k.float().cpu(), ret_v.float().cpu(), inp, ref_k, ref_v, T, committed_len
    )


def run_chaining(name, cores, seed, L=L_CTX, committed_len=COMMITTED_LEN):
    """(1) then (3) over the same cache buffers must equal (3) alone -- the benign-WAW check."""
    draft = make_inputs(T=1, L=L, committed_len=committed_len, seed=seed)
    replay = make_inputs(T=2, L=L, committed_len=committed_len, seed=seed + 1)
    replay["k_cache"] = draft["k_cache"]
    replay["v_cache"] = draft["v_cache"]

    keep_1 = build_keep(1, L, committed_len)
    keep_2 = build_keep(2, L, committed_len)
    _, _, k_only, v_only = run_kernel(replay, 2, keep_2, cores, False)
    _, _, k_1, v_1 = run_kernel(draft, 1, keep_1, cores, True)
    _, _, k_chain, v_chain = run_kernel(
        replay, 2, keep_2, cores, False, k_cache=k_1, v_cache=v_1
    )
    _bitexact(f"{name}.kcache", k_chain.float().cpu(), k_only.float().cpu())
    _bitexact(f"{name}.vcache", v_chain.float().cpu(), v_only.float().cpu())


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_draft_t1_c2():
    run_case("draft_T1_c2", T=1, cores=2, seed=31, with_lm_head=True)


def test_replay_t2_c2():
    run_case("replay_T2_c2", T=2, cores=2, seed=32, with_lm_head=False)


def test_chaining_c2():
    run_chaining("chain_c2", cores=2, seed=31)


def test_draft_t1_c1():
    run_case("draft_T1_c1", T=1, cores=1, seed=31, with_lm_head=True)


def test_replay_t2_c1():
    run_case("replay_T2_c1", T=2, cores=1, seed=32, with_lm_head=False)


def main():
    print("=== Draft megakernel -- fp32, TP=1, L=256, committed_len=150 ===")
    test_draft_t1_c2()
    test_replay_t2_c2()
    test_chaining_c2()
    test_draft_t1_c1()
    test_replay_t2_c1()
    print("\nALL DRAFT MEGAKERNEL CASES PASSED")


if __name__ == "__main__":
    main()
