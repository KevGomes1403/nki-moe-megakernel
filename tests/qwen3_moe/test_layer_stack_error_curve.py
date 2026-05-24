"""Per-layer hidden-state error-curve diagnostic for the qwen3_moe speculative
megakernel.

WHY THIS TEST EXISTS
--------------------
``test_kv_cache_roundtrip.py`` ruled out the KV cache as the cause of the
end-to-end degeneration (coherent token 1, then collapse into repetition). The
remaining hypothesis is per-step compute *amplification*: a small biased error
introduced by each decoder layer, compounding across the 48-layer stack until
the logit distribution derails. Token 1 being coherent only proves token 1's
argmax survived — not that the 48-layer compute is accurate.

WHAT THIS TEST DOES
-------------------
A full decoder layer in ``transformer_qwen3_moe_speculative._multilayer_body``
is:  attention block -> residual add -> MoE block -> residual add. This test
composes that layer from the two faithful single-block kernels already in the
suite (``attention_only_kernel`` + ``moe_only_kernel``, each a strict subset of
the megakernel) with bf16 residual adds, then CHAINS ``N_LAYERS`` of them.

Two hidden states are propagated in lockstep from the same X0:
  * ``nki_h`` — advanced by the NKI blocks;
  * ``ref_h`` — advanced by the golden HF reference blocks.

Per layer it reports:
  * ``in_rel``    — relative L2 error entering the layer (compounded so far);
  * ``attn_add``  — relative error the NKI attention block adds, measured
                    against HF run on the *same* (drifted) input;
  * ``moe_add``   — same, for the NKI MoE block;
  * ``out_rel``   — relative L2 error leaving the layer (compounded);
  * ``moe_bias``  — *signed* mean error of the MoE block / RMS(reference). A
                    consistent sign across layers is a systematic bias (the
                    thing that amplifies autoregressively); zero-mean is benign
                    bf16 drift.

INTERPRETING THE CURVE
----------------------
* ``out_rel`` flat / sub-linear  -> the stack does NOT amplify; per-step compute
  is fine and the degeneration is elsewhere (e.g. an on-device-only effect).
* ``out_rel`` growing geometrically -> a layer amplifies. The block with the
  larger, biased ``*_add`` is the culprit.
* ``moe_bias`` with a consistent sign -> systematic MoE error (matches the F2
  "expert-chain accumulation" lead); ~zero-mean -> benign rounding.

Faithfulness: the composed layer runs the exact NKI ops of the megakernel; the
canonical<->shard-interleaved round trip each block does is a lossless bf16
permutation, and a bf16 residual add is layout-independent — so a per-block
error shown here is a per-block error in the megakernel.

LIMITATIONS: runs single-rank (``replica_groups=([0],)``), so the TP>1
cross-rank ``nccl.all_reduce`` inside ``_sb2sb_all_reduce_gather`` executes as
identity — the multi-rank reduction is NOT exercised here. The compounded
``out_rel`` also uses random per-layer weights, so it is contaminated by
expansive layers / residual-stream cancellation and is only bounded by a loose
LINEAR envelope; the per-block columns and the signed bias are the trustworthy
gates.

Runs on ``nki.simulate`` (CPU, LNC=2): 2 simulations per layer. Set the env var
``QWEN3_LAYER_CURVE_N`` to override the layer count (default 48).
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
import torch

# Reference + kernel runners, single-sourced from the single-block tests.
from test_attention_vs_hf import (  # noqa: E402
    _attention_reference,
    _build_cos_sin_at_pos,
    _seeded,
)
from test_attention_vs_hf import _run_nki_kernel as _run_attn_nki  # noqa: E402
from test_moe_vs_hf import _moe_reference  # noqa: E402
from test_moe_vs_hf import _run_nki_kernel as _run_moe_nki  # noqa: E402
from _attention_only_kernel import (  # noqa: E402
    D_HEAD,
    H,
    NUM_KV_HEADS_TP,
    NUM_Q_HEADS_TP,
)
from megakernels.qwen3_moe.transformer_qwen3_moe_speculative import (  # noqa: E402
    E,
    I_PER_EXPERT,
)

# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

B         = 1
S_TKG     = 1
S_MAX     = 256
POS       = 10        # small decode position — keeps RoPE rounding clean so
                      # the signal is structural amplification, not ULP noise
DTYPE     = torch.bfloat16
SEED      = 0x5709
ROPE_BASE = 1_000_000.0

N_LAYERS = int(os.environ.get("QWEN3_LAYER_CURVE_N", "48"))

# A correct bf16 transformer block adds ~5e-3 relative L2 error vs the fp32
# reference. The trustworthy signals are the per-block columns and the bias;
# the compounded out_rel with random weights is contaminated by expansive
# layers / residual-stream cancellation, so it is bounded only by a loose
# LINEAR envelope (a geometric amplifier breaks a linear bound).
BLOCK_ADD_CEIL = 5e-2     # per-layer single-block relative L2 error ceiling
BIAS_CEIL      = 1e-3     # |mean signed MoE error| — guards a systematic bias
LINEAR_RATE    = 1.5e-2   # out_rel must stay under (layer+1) * LINEAR_RATE


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _rel(nki: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative L2 error — scale-invariant, so robust to residual-stream growth."""
    n = nki.to(torch.float32)
    r = ref.to(torch.float32)
    return (torch.linalg.norm(n - r) / (torch.linalg.norm(r) + 1e-12)).item()


def _bias(nki: torch.Tensor, ref: torch.Tensor) -> float:
    """Signed mean error / RMS(reference). Consistent sign => systematic bias."""
    n = nki.to(torch.float32)
    r = ref.to(torch.float32)
    rms = r.pow(2).mean().sqrt() + 1e-12
    return ((n - r).mean() / rms).item()


def _badd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 residual add: fp32-accumulate one add, round to bf16 — exactly what
    the kernel's ``nisa.tensor_tensor(op=add)`` into a bf16 dst does."""
    return (a.to(torch.float32) + b.to(torch.float32)).to(DTYPE)


# --------------------------------------------------------------------------
# Per-layer weights (distinct random weights per layer, deterministic)
# --------------------------------------------------------------------------

def _make_layer_weights(layer_idx: int) -> dict:
    g = torch.Generator().manual_seed(SEED ^ (0xA5A5 + layer_idx * 7919))

    K = torch.zeros(B, NUM_KV_HEADS_TP, S_MAX, D_HEAD, dtype=DTYPE)
    V = torch.zeros(B, NUM_KV_HEADS_TP, S_MAX, D_HEAD, dtype=DTYPE)
    K[:, :, :POS, :] = _seeded(g, (B, NUM_KV_HEADS_TP, POS, D_HEAD), DTYPE, scale=0.1)
    V[:, :, :POS, :] = _seeded(g, (B, NUM_KV_HEADS_TP, POS, D_HEAD), DTYPE, scale=0.1)
    cos, sin = _build_cos_sin_at_pos(POS, B, D_HEAD, DTYPE, base=ROPE_BASE)

    return {
        "Wq": _seeded(g, (NUM_Q_HEADS_TP  * D_HEAD, H), DTYPE, scale=0.02),
        "Wk": _seeded(g, (NUM_KV_HEADS_TP * D_HEAD, H), DTYPE, scale=0.02),
        "Wv": _seeded(g, (NUM_KV_HEADS_TP * D_HEAD, H), DTYPE, scale=0.02),
        "Wo": _seeded(g, (NUM_Q_HEADS_TP  * D_HEAD, H), DTYPE, scale=0.02),
        "qn":   torch.ones(D_HEAD, dtype=DTYPE) + _seeded(g, (D_HEAD,), DTYPE, scale=0.01),
        "kn":   torch.ones(D_HEAD, dtype=DTYPE) + _seeded(g, (D_HEAD,), DTYPE, scale=0.01),
        "gpre": torch.ones(H,      dtype=DTYPE) + _seeded(g, (H,),      DTYPE, scale=0.01),
        "gpost": torch.ones(1, H,  dtype=DTYPE) + _seeded(g, (1, H),    DTYPE, scale=0.01),
        "router_w":  _seeded(g, (H, E), DTYPE, scale=0.02),
        "gate_up_w": _seeded(g, (E, H, 2 * I_PER_EXPERT), DTYPE, scale=0.02),
        "down_w":    _seeded(g, (E, I_PER_EXPERT, H), DTYPE, scale=0.02),
        "K": K, "V": V, "cos": cos, "sin": sin,
        "pos_ids": torch.full((B, 1), POS, dtype=torch.int32),
    }


# --------------------------------------------------------------------------
# Single-block runners — NKI (via nki.simulate) and golden HF reference
# --------------------------------------------------------------------------

def _attn_block_nki(h: torch.Tensor, lw: dict) -> torch.Tensor:
    Y, _, _, _ = _run_attn_nki({
        "X": h,
        "Wq": lw["Wq"], "Wk": lw["Wk"], "Wv": lw["Wv"], "Wo": lw["Wo"],
        "qn": lw["qn"], "kn": lw["kn"], "gpre": lw["gpre"],
        "K_cache": lw["K"], "V_cache": lw["V"],
        "cos_at_pos": lw["cos"], "sin_at_pos": lw["sin"],
        "position_ids": lw["pos_ids"],
    })
    return Y


def _attn_block_ref(h: torch.Tensor, lw: dict) -> torch.Tensor:
    Y, _, _ = _attention_reference(
        X=h,
        Wq=lw["Wq"], Wk=lw["Wk"], Wv=lw["Wv"], Wo=lw["Wo"],
        qn=lw["qn"], kn=lw["kn"], gpre=lw["gpre"],
        K_cache=lw["K"], V_cache=lw["V"],
        cos_at_pos=lw["cos"], sin_at_pos=lw["sin"], pos=POS,
    )
    return Y


def _moe_block_nki(h: torch.Tensor, lw: dict) -> torch.Tensor:
    # _run_moe_nki converts EVERY dict value — pass tensors only.
    Y, _ = _run_moe_nki({
        "X": h, "gpost": lw["gpost"], "router_w": lw["router_w"],
        "gate_up_w": lw["gate_up_w"], "down_w": lw["down_w"],
    })
    return Y


def _moe_block_ref(h: torch.Tensor, lw: dict) -> torch.Tensor:
    return _moe_reference(
        X=h, gpost=lw["gpost"], router_w=lw["router_w"],
        gate_up_w=lw["gate_up_w"], down_w=lw["down_w"],
    )


# --------------------------------------------------------------------------
# The diagnostic
# --------------------------------------------------------------------------

def test_layer_stack_error_curve():
    layers = [_make_layer_weights(k) for k in range(N_LAYERS)]

    g0 = torch.Generator().manual_seed(SEED)
    X0 = _seeded(g0, (B, S_TKG, H), DTYPE, scale=0.02)

    nki_h = X0.clone()
    ref_h = X0.clone()

    in_rels: list[float] = []
    attn_adds: list[float] = []
    moe_adds: list[float] = []
    out_rels: list[float] = []
    moe_biases: list[float] = []

    print(f"\n[layer-stack error curve]  N_LAYERS={N_LAYERS}  pos={POS}")
    print(f"{'layer':>5} {'in_rel':>11} {'attn_add':>11} {'moe_add':>11} "
          f"{'out_rel':>11} {'moe_bias':>11}")
    for k in range(N_LAYERS):
        lw = layers[k]
        in_rel = _rel(nki_h, ref_h)

        # --- NKI layer, advancing its own (drifted) hidden state ---
        nki_Yattn = _attn_block_nki(nki_h, lw)
        nki_r1    = _badd(nki_h, nki_Yattn)
        nki_Ymoe  = _moe_block_nki(nki_r1, lw)
        nki_out   = _badd(nki_r1, nki_Ymoe)

        # --- HF blocks on the SAME nki inputs: isolates each block's own NKI
        #     error from the error inherited via the input. ---
        ref_Yattn_on_nki = _attn_block_ref(nki_h, lw)
        ref_Ymoe_on_nki  = _moe_block_ref(nki_r1, lw)
        attn_add = _rel(nki_Yattn, ref_Yattn_on_nki)
        moe_add  = _rel(nki_Ymoe,  ref_Ymoe_on_nki)
        moe_bias = _bias(nki_Ymoe, ref_Ymoe_on_nki)

        # --- Golden reference layer, advancing its own hidden state ---
        ref_Yattn = _attn_block_ref(ref_h, lw)
        ref_r1    = _badd(ref_h, ref_Yattn)
        ref_Ymoe  = _moe_block_ref(ref_r1, lw)
        ref_out   = _badd(ref_r1, ref_Ymoe)

        out_rel = _rel(nki_out, ref_out)

        print(f"{k:5d} {in_rel:11.3e} {attn_add:11.3e} {moe_add:11.3e} "
              f"{out_rel:11.3e} {moe_bias:11.3e}")

        in_rels.append(in_rel)
        attn_adds.append(attn_add)
        moe_adds.append(moe_add)
        out_rels.append(out_rel)
        moe_biases.append(moe_bias)

        nki_h, ref_h = nki_out, ref_out

    # ---- Verdict ---------------------------------------------------------
    final          = out_rels[-1]
    attn_add_med   = statistics.median(attn_adds)
    moe_add_med    = statistics.median(moe_adds)
    mean_moe_bias  = statistics.fmean(moe_biases)
    max_attn       = max(attn_adds)
    max_moe        = max(moe_adds)

    # A non-amplifying stack accumulates per-layer rounding roughly linearly.
    # A layer that amplifies error makes out_rel grow geometrically, which
    # breaks a linear envelope; random-weight expansive layers / residual
    # cancellation only cause bounded jumps, which the envelope absorbs.
    envelope_viol = [(k, out_rels[k]) for k in range(N_LAYERS)
                     if out_rels[k] > (k + 1) * LINEAR_RATE]

    print("-" * 68)
    print(f"final out_rel          = {final:.3e}")
    print(f"median attn_add        = {attn_add_med:.3e}   (max {max_attn:.3e})")
    print(f"median moe_add         = {moe_add_med:.3e}   (max {max_moe:.3e})")
    print(f"mean moe_bias (signed) = {mean_moe_bias:.3e}  "
          f"(|mean| << {BIAS_CEIL:.0e} => no systematic MoE bias)")
    print(f"linear-envelope breaches = {len(envelope_viol)}")
    print("-" * 68)

    # (1) Per-block local accuracy: each NKI block must match HF on an
    #     identical input. A blown bound localizes a structural bug to that
    #     block.
    assert max_attn <= BLOCK_ADD_CEIL, (
        f"attention block per-layer error exceeds {BLOCK_ADD_CEIL:.0e} "
        f"(max={max_attn:.3e} at layer {attn_adds.index(max_attn)})."
    )
    assert max_moe <= BLOCK_ADD_CEIL, (
        f"MoE block per-layer error exceeds {BLOCK_ADD_CEIL:.0e} "
        f"(max={max_moe:.3e} at layer {moe_adds.index(max_moe)})."
    )

    # (2) No systematic bias: the MoE block's signed mean error must be
    #     ~zero. A consistent sign is the error shape that compounds
    #     autoregressively (the F2 "expert-chain accumulation" hypothesis).
    assert abs(mean_moe_bias) <= BIAS_CEIL, (
        f"MoE block has a systematic signed bias: mean moe_bias="
        f"{mean_moe_bias:.3e} (> {BIAS_CEIL:.0e}). This is the error shape "
        f"that compounds autoregressively."
    )

    # (3) No geometric amplification: out_rel must stay under a linear
    #     envelope. A layer that amplifies incoming error breaks this; the
    #     per-block columns localize which block.
    assert not envelope_viol, (
        f"compounded error grows faster than linear — a layer amplifies "
        f"error. First breach: layer {envelope_viol[0][0]} "
        f"out_rel={envelope_viol[0][1]:.3e} > "
        f"{(envelope_viol[0][0] + 1) * LINEAR_RATE:.3e}. "
        f"median attn_add={attn_add_med:.3e}, moe_add={moe_add_med:.3e}."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
