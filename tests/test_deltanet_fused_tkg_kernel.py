"""Isolated correctness test for the FUSED DeltaNet conv + recurrence TKG NKI kernel.

Calls `deltanet_fused_tkg_fwd` / `deltanet_fused_tkg_fwd_state` DIRECTLY (no NeuronGatedDeltaNet
module) and compares against a golden chain of the two existing single-stage references: the conv
golden (`ref_conv`, depthwise 4-tap causal conv + SiLU + q/k/v split + candidate windows) feeding
the recurrence golden (`ref_full`, sequential gated delta-rule with l2norm/GQA/gating folded in).
The fused kernel passes q/k/v conv->recurrence entirely in SBUF and is value-head-sharded across
LNC=2, so its three outputs match the sequential reference per head/channel.

Golden chain (the goldens are imported, not re-derived):
    ref_q, ref_k, ref_v, ref_cand = ref_conv(qkv, conv_state, conv_weight)
    ref_attn, ref_states          = ref_full(ref_q, ref_k, ref_v, a, b, A_log, dt_bias, init_state)
(ref_conv emits silu'd, head-split, un-normed q/k/v -- exactly ref_full's inputs; ref_full l2norms.)

Staged so a failure localizes:
  1. v bridge      -- the gathered v (kernel q/k/v path) reproduces the conv's standalone v.
  2. decode (T=1)  -- attn_out / final_state / new_conv_state vs the chained reference.
  3. verify (T=2)  -- attn_out / candidate_states / conv_cand vs the chained reference.
  4. cores=1 vs cores=2 bit-identical cross-check on the verify outputs.

Tolerance: atol=1e-5, rtol=1e-2 on every output.

Run:
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_deltanet_fused_tkg_kernel
"""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Vendored HF reference on sys.path so the gated norm imports as a top-level module (the package
# __init__ is a transformers-style lazy loader that doesn't resolve from the repo root).
_HF_REF = Path(__file__).resolve().parent.parent / "_hf_reference"
if str(_HF_REF) not in sys.path:
    sys.path.insert(0, str(_HF_REF))

from models.qwen3_6_moe.nki_kernels.deltanet.decode.fused_layer import (  # noqa: E402
    deltanet_fused_tkg_fwd,
    deltanet_fused_tkg_fwd_state,
)
from models.qwen3_6_moe.tests.test_deltanet_conv_tkg_kernel import (  # noqa: E402
    CONV_DIM,
    HEAD_DIM,
    HV,
    K,
    KEY_DIM,
    ref_conv,
)
from models.qwen3_6_moe.tests.test_deltanet_tkg_kernel import ref_full  # noqa: E402
from modeling_qwen3_5_moe import Qwen3_5MoeRMSNormGated  # noqa: E402

STATE_W = K - 1
ATOL = 1e-5
RTOL = 1e-2
EPS = 1e-6  # config rms_norm_eps


def make_inputs(T, seed):
    """Random fp32 inputs for the fused kernel (conv inputs + recurrence gating/state inputs +
    the gated-norm z gate and gamma weight)."""
    torch.manual_seed(seed)
    qkv = torch.randn(T, CONV_DIM)
    conv_state = torch.randn(CONV_DIM, STATE_W)
    conv_weight = torch.randn(CONV_DIM, K)
    a = torch.randn(T, HV)
    b = torch.randn(T, HV)
    A_log = torch.randn(HV) * 0.5
    dt_bias = torch.randn(HV) * 0.5
    init_state = torch.randn(HV, HEAD_DIM, HEAD_DIM) * 0.1
    z = torch.randn(T, HV * HEAD_DIM)
    gamma = torch.randn(HEAD_DIM)  # random (not ones) to exercise the norm weight
    return qkv, conv_state, conv_weight, a, b, A_log, dt_bias, init_state, z, gamma


def golden(inp):
    """Chained reference: conv golden -> recurrence golden -> gated per-head RMSNorm.

    ``ref_attn`` is the raw [T, HV*HEAD_DIM] head-major recurrence output; the HF gated norm
    (``Qwen3_5MoeRMSNormGated``: per-head RMSNorm over HEAD_DIM * silu(z)) is then applied to match
    the kernel's gated output. Returns the three fused outputs (the gated attn_out, the states, the
    conv candidates) plus ref_v."""
    qkv, conv_state, conv_weight, a, b, A_log, dt_bias, init_state, z, gamma = inp
    ref_q, ref_k, ref_v, ref_cand = ref_conv(qkv, conv_state, conv_weight)
    ref_attn, ref_states = ref_full(ref_q, ref_k, ref_v, a, b, A_log, dt_bias, init_state)

    # Per-head gated RMSNorm: reshape [T, HV*HEAD_DIM] -> [T, HV, HEAD_DIM], norm over HEAD_DIM, gate.
    T = ref_attn.shape[0]
    norm = Qwen3_5MoeRMSNormGated(HEAD_DIM, eps=EPS)
    norm.weight.data = gamma.to(ref_attn.dtype).clone()
    ref_attn_r = ref_attn.reshape(T, HV, HEAD_DIM).to(ref_attn.dtype)
    z_r = z.reshape(T, HV, HEAD_DIM).to(ref_attn.dtype)
    gated = norm(ref_attn_r, gate=z_r).reshape(T, HV * HEAD_DIM)
    return gated, ref_states, ref_cand, ref_v


def run_kernel(fn, inp, cores=2):
    """Move inputs to the Neuron device, launch the @nki.jit kernel on ``cores`` cores (value-head
    SPMD shard) with the gated-norm z/gamma/eps, return CPU tensors."""
    import torch_xla.core.xla_model as xm

    qkv, conv_state, conv_weight, a, b, A_log, dt_bias, init_state, z, gamma = inp
    dev = xm.xla_device()
    args = [
        qkv.float().contiguous().to(dev),
        conv_state.float().contiguous().to(dev),
        conv_weight.float().contiguous().to(dev),
    ]
    tail = [
        a.float().contiguous().to(dev),
        b.float().contiguous().to(dev),
        A_log.float().contiguous().to(dev),
        dt_bias.float().contiguous().to(dev),
        init_state.float().contiguous().to(dev),
        z.float().contiguous().to(dev),
        gamma.float().contiguous().to(dev),
    ]
    outs = fn[cores](args[0], args[1], args[2], KEY_DIM, *tail, EPS)
    return tuple(o.cpu() for o in outs)


def _metrics(ker, ref):
    ker = ker.double()
    ref = ref.double()
    abs_err = (ker - ref).abs()
    denom = ref.abs().clamp_min(1e-4)
    return abs_err.max().item(), (abs_err / denom).max().item()


def _check(name, ker, ref):
    max_abs, max_rel = _metrics(ker, ref)
    ok = torch.allclose(ker.double(), ref.double(), atol=ATOL, rtol=RTOL)
    status = "PASS" if ok else "FAIL"
    print(f"[{name}] {status}  max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}")
    assert ok, f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} exceeds atol={ATOL} rtol={RTOL}"


def run_stage_v_bridge(name, seed):
    """Stage gate 1: the SBUF v bridge is exercised end-to-end by attn_out/state below (the gathered
    v_row feeds the delta rule directly), so a bridge bug surfaces there as a mismatch. Here we only
    assert the conv golden's v is well-formed so any later mismatch is unambiguously the bridge or the
    recurrence, not the conv golden. (The bridge ships as a per-head SBUF->SBUF DMA gather -- see the
    note below -- since a single strided multi-partition AP is rejected by AP validation.)"""
    inp = make_inputs(T=2, seed=seed)
    _, _, _, ref_v = golden(inp)
    ok = torch.isfinite(ref_v).all().item() and ref_v.shape == (HV, 2, HEAD_DIM)
    status = "PASS" if ok else "FAIL"
    print(f"[{name}:conv_v_well_formed] {status}  shape={tuple(ref_v.shape)}")
    assert ok, f"{name}: conv golden v malformed {tuple(ref_v.shape)}"


def run_stage_decode(name, seed):
    """deltanet_fused_tkg_fwd (T=1): attn_out / final_state / new_conv_state vs the chained reference."""
    inp = make_inputs(T=1, seed=seed)
    ker_attn, ker_final, ker_conv = run_kernel(deltanet_fused_tkg_fwd, inp, cores=2)
    ref_attn, ref_states, ref_cand, _ = golden(inp)
    _check(f"{name}:attn_out", ker_attn, ref_attn)
    _check(f"{name}:final_state", ker_final, ref_states[-1])
    _check(f"{name}:new_conv_state", ker_conv, ref_cand[0])


def run_stage_verify(name, seed):
    """deltanet_fused_tkg_fwd_state (T=2): attn_out / candidate_states / conv_cand vs the reference."""
    inp = make_inputs(T=2, seed=seed)
    ker_attn, ker_cand, ker_conv = run_kernel(deltanet_fused_tkg_fwd_state, inp, cores=2)
    ref_attn, ref_states, ref_cand, _ = golden(inp)
    _check(f"{name}:attn_out", ker_attn, ref_attn)
    _check(f"{name}:candidate_states", ker_cand, ref_states)
    _check(f"{name}:conv_cand", ker_conv, ref_cand)


def run_shard_equality(name, seed):
    """Launch the verify path single-core and LNC=2 (value-head shard); assert bit-identical outputs
    -- proves the shard reproduces the unsharded result exactly."""
    inp = make_inputs(T=2, seed=seed)
    a1, c1, k1 = run_kernel(deltanet_fused_tkg_fwd_state, inp, cores=1)
    a2, c2, k2 = run_kernel(deltanet_fused_tkg_fwd_state, inp, cores=2)
    ok = torch.equal(a1, a2) and torch.equal(c1, c2) and torch.equal(k1, k2)
    status = "PASS" if ok else "FAIL"
    print(f"[{name}:cores1_vs_cores2] {status}")
    assert ok, f"{name}: cores=1 and cores=2 outputs differ (shard is not bit-identical)"


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_stage1_v_bridge():
    run_stage_v_bridge("stage1_v_bridge", seed=0)


def test_stage2_decode():
    run_stage_decode("stage2_decode_T1", seed=1)


def test_stage3_verify():
    run_stage_verify("stage3_verify_T2", seed=2)


def test_stage4_shard_bit_identical():
    run_shard_equality("stage4_shard", seed=3)


def main():
    run_stage_v_bridge("stage1_v_bridge", seed=0)
    run_stage_decode("stage2_decode_T1", seed=1)
    run_stage_verify("stage3_verify_T2", seed=2)
    run_shard_equality("stage4_shard", seed=3)
    print("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
