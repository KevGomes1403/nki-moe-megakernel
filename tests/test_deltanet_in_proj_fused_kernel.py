"""Isolated correctness test for the FUSED DeltaNet in_proj + conv + recurrence TKG NKI kernel.

Calls ``deltanet_in_proj_fused_tkg_fwd`` / ``deltanet_in_proj_fused_tkg_fwd_state`` DIRECTLY and
compares the full pipeline against a chained **fp32** CPU golden:

    normed       = RMSNorm(hidden) @ proj_w                         (standard (x*rsqrt(mean(x^2)+eps))*g)
    qkv|z|a|b    = slice normed at offsets conv_dim, +value_dim, +Hv, +Hv
    q,k,v,cand   = ref_conv(qkv, conv_state, conv_weight)           (the conv golden, fp32)
    attn,states  = ref_full_fp32(q, k, v, a, b, A_log, dt_bias, init_state)   (delta-rule, fp32)
    gated        = gated_norm_fp32(attn, gate=z)                    (per-head RMSNorm over head_dim, fp32)

DTYPE: every device input is quantized to bf16 (the dtype the model invokes the kernel in), then the
reference carries those bf16-rounded values through the whole chain in fp32. fp32 matches the kernel's
internal accumulation precision, so it is the ideal oracle: it shares the kernel's input quantization
but isolates real compute/layout bugs from bf16 accumulation noise. (A full-bf16 reference is NOT a
valid bit-correctness oracle here -- its own rounding through the deep contraction + recurrent chain
diverges from the true math by far more than any real kernel bug would.)

The fused kernel keeps qkv/z/a/b SBUF-resident from the projection through conv and recurrence (no HBM
round-trip) and is value-head-sharded across LNC=2.

Hard gate (bit-correctness intent): plain ``torch.allclose(atol=1e-5, rtol=1e-2)`` per output -- no
magnitude-scaled atol, no cosine similarity. max_abs_err / max_rel_err are printed for diagnosis only.

Staged so a failure localizes:
  1. decode (T=1) -- attn_out / final_state / new_conv_state vs the fp32 reference.
  2. verify (T=2) -- attn_out / candidate_states / conv_cand vs the fp32 reference.
  3. cores=1 vs cores=2 on the verify path -- same hard gate.

Run:
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_deltanet_in_proj_fused_kernel
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.qwen3_6_moe.nki_kernels.deltanet.decode.fused_layer import (  # noqa: E402
    deltanet_in_proj_fused_tkg_fwd,
    deltanet_in_proj_fused_tkg_fwd_state,
)
from models.qwen3_6_moe.tests.test_deltanet_conv_tkg_kernel import (  # noqa: E402
    CONV_DIM,
    HEAD_DIM,
    HV,
    K,
    KEY_DIM,
    ref_conv,
)

HIDDEN = 2048  # per-rank hidden (TP=4)
VALUE_DIM = CONV_DIM - 2 * KEY_DIM  # 1024
NUM_V_HEADS = HV  # 8
I_DIM = CONV_DIM + VALUE_DIM + 2 * NUM_V_HEADS  # 3088

# Slice offsets into the fused projection (qkv | z | a | b, in that order).
OFF_QKV = 0
OFF_Z = CONV_DIM
OFF_A = OFF_Z + VALUE_DIM
OFF_B = OFF_A + NUM_V_HEADS
OFF_END = OFF_B + NUM_V_HEADS

STATE_W = K - 1
# Hard bit-correctness gate against the fp32 reference: plain allclose, no scaling, no cosine.
ATOL = 1e-5
RTOL = 1e-2
DTYPE = torch.bfloat16  # device input dtype (the model's dtype)
REF_DTYPE = (
    torch.float32
)  # reference compute precision (matches the kernel's internal accumulation)
EPS = 1e-6  # input RMSNorm + gated RMSNorm epsilon (config rms_norm_eps)


def make_inputs(T, seed):
    """Random fp32 inputs: hidden + the concatenated [H, I] projection weight + input/gated norm
    weights + the conv and recurrence state/gating inputs.

    Weights use realistic 1/sqrt(fan_in) init (matching nn.Linear / the depthwise conv) so the deep
    contractions produce O(1) outputs. This matters for the hard bf16 gate: with unscaled randn
    weights the 2048-deep projection reaches magnitude ~165, where one bf16 ULP is ~1.0 and atol=1e-5
    is physically unmeetable -- the kernel is exact there, only bf16 quantization of huge values shows.
    """
    torch.manual_seed(seed)
    hidden = torch.randn(1, T, HIDDEN)
    # nn.Linear weights are [out, in] = [I, H]; the kernel wants [H, I] (the transpose). Scale by
    # 1/sqrt(H) (Linear's fan_in) so normed @ proj_w stays O(1) rather than O(sqrt(H)).
    w_scale = 1.0 / math.sqrt(HIDDEN)
    w_qkv = torch.randn(CONV_DIM, HIDDEN) * w_scale
    w_z = torch.randn(VALUE_DIM, HIDDEN) * w_scale
    w_a = torch.randn(NUM_V_HEADS, HIDDEN) * w_scale
    w_b = torch.randn(NUM_V_HEADS, HIDDEN) * w_scale
    proj_w = torch.cat([w_qkv, w_z, w_a, w_b], dim=0).t().contiguous()  # [H, I]
    gamma = torch.randn(HIDDEN)  # input RMSNorm weight (random, not ones)
    conv_state = torch.randn(CONV_DIM, STATE_W)
    conv_weight = torch.randn(CONV_DIM, K) / math.sqrt(
        K
    )  # K-tap depthwise: keep conv output O(1)
    A_log = torch.randn(HV) * 0.5
    dt_bias = torch.randn(HV) * 0.5
    init_state = torch.randn(HV, HEAD_DIM, HEAD_DIM) * 0.1
    z_gamma = torch.randn(HEAD_DIM)  # gated per-head RMSNorm weight
    return (
        hidden,
        proj_w,
        gamma,
        conv_state,
        conv_weight,
        A_log,
        dt_bias,
        init_state,
        z_gamma,
    )


def quant(x):
    """Quantize to bf16 (what the kernel sees), then carry the value in fp32 for the reference compute."""
    return x.to(DTYPE).to(REF_DTYPE)


def rms_norm_ref(x, gamma, eps):
    """Standard RMSNorm in fp32: (x * rsqrt(mean(x^2) + eps)) * gamma -- matches qkv_tkg."""
    x = x.to(REF_DTYPE)
    rms = (x.square().mean(dim=-1, keepdim=True) + eps).sqrt()
    return (x * rms.reciprocal()) * gamma.to(REF_DTYPE)


def ref_full_fp32(q, k, v, a, b, A_log, dt_bias, init):
    """Delta-rule recurrence in fp32 (mirrors test_deltanet_tkg's ref_full). Inputs are the
    bf16-quantized values carried in fp32; the recurrence accumulates in fp32 like the kernel.

    Shapes: q,k (Hk,T,d); v (Hv,T,d); a,b (T,Hv); A_log,dt_bias (Hv,); init (Hv,d,d).
    Returns attn_out (T, Hv*d) head-major and states (T, Hv, d, d). No norm/z-gate here.
    """
    Hk, T, d = q.shape
    Hv = v.shape[0]
    rep = Hv // Hk

    qn = F.normalize(q, p=2, dim=-1) / math.sqrt(d)
    kn = F.normalize(k, p=2, dim=-1)
    beta = torch.sigmoid(b)  # (T, Hv)
    g = -torch.exp(A_log) * F.softplus(a + dt_bias)  # (T, Hv)

    states = torch.zeros(Hv, T, d, d, dtype=REF_DTYPE)
    out_raw = torch.zeros(Hv, T, d, dtype=REF_DTYPE)
    for h in range(Hv):
        kh = h // rep
        state = init[h].clone()
        for t in range(T):
            state = state * torch.exp(g[t, h])
            kv = kn[kh, t] @ state
            delta = (v[h, t] - kv) * beta[t, h]
            state = state + torch.outer(kn[kh, t], delta)
            out_raw[h, t] = qn[kh, t] @ state
            states[h, t] = state

    attn_out = out_raw.permute(1, 0, 2).contiguous().reshape(T, Hv * d)  # (T, Hv*d)
    states = states.permute(1, 0, 2, 3).contiguous()  # (T, Hv, d, d)
    return attn_out, states


def gated_norm_fp32(attn, gate, weight, eps):
    """Per-head gated RMSNorm in fp32 (mirrors Qwen3_5MoeRMSNormGated). attn/gate: (T, Hv, d)."""
    x = attn.to(REF_DTYPE)
    var = x.square().mean(dim=-1, keepdim=True)
    x = x * (var + eps).rsqrt()
    x = weight.to(REF_DTYPE) * x
    return x * F.silu(gate.to(REF_DTYPE))


def golden(inp):
    """Chained fp32 CPU reference on bf16-quantized inputs: RMSNorm @ proj_w -> slice -> conv ->
    recurrence -> gated per-head RMSNorm. Returns (gated_attn, ref_states, ref_cand)."""
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
    ) = (quant(t) for t in inp)

    T = hidden.shape[1]
    normed = rms_norm_ref(hidden, gamma, EPS).reshape(T, HIDDEN)
    proj = normed @ proj_w  # [T, I], fp32

    qkv = proj[:, OFF_QKV:OFF_Z].contiguous()
    z = proj[:, OFF_Z:OFF_A].contiguous()
    a = proj[:, OFF_A:OFF_B].contiguous()
    b = proj[:, OFF_B:OFF_END].contiguous()

    ref_q, ref_k, ref_v, ref_cand = ref_conv(qkv, conv_state, conv_weight)
    ref_attn, ref_states = ref_full_fp32(
        ref_q, ref_k, ref_v, a, b, A_log, dt_bias, init_state
    )

    gated = gated_norm_fp32(
        ref_attn.reshape(T, HV, HEAD_DIM), z.reshape(T, HV, HEAD_DIM), z_gamma, EPS
    ).reshape(T, HV * HEAD_DIM)
    return gated, ref_states, ref_cand


def run_kernel(fn, inp, cores=2):
    """Move bf16 inputs to the Neuron device, launch the @nki.jit kernel on ``cores`` cores (value-head
    SPMD shard), return CPU tensors."""
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
    ) = inp
    dev = xm.xla_device()
    h = hidden.to(DTYPE).contiguous().to(dev)
    w = proj_w.to(DTYPE).contiguous().to(dev)
    g = gamma.reshape(1, HIDDEN).to(DTYPE).contiguous().to(dev)  # qkv_tkg wants [1, H]
    tail = [
        conv_state.to(DTYPE).contiguous().to(dev),
        conv_weight.to(DTYPE).contiguous().to(dev),
    ]
    rec = [
        A_log.to(DTYPE).contiguous().to(dev),
        dt_bias.to(DTYPE).contiguous().to(dev),
        init_state.to(DTYPE).contiguous().to(dev),
        z_gamma.to(DTYPE).contiguous().to(dev),
    ]
    outs = fn[cores](h, w, g, EPS, tail[0], tail[1], KEY_DIM, *rec, EPS)
    return tuple(o.float().cpu() for o in outs)


def _metrics(ker, ref):
    ker = ker.float().reshape(-1)
    ref = ref.float().reshape(-1)
    abs_err = (ker - ref).abs()
    denom = ref.abs().clamp_min(1e-4)
    return abs_err.max().item(), (abs_err / denom).max().item()


def _check(name, ker, ref):
    max_abs, max_rel = _metrics(ker, ref)
    ok = torch.allclose(ker.float(), ref.float(), atol=ATOL, rtol=RTOL)
    status = "PASS" if ok else "FAIL"
    print(
        f"[{name}] {status}  max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}  "
        f"(atol={ATOL} rtol={RTOL})"
    )
    assert ok, (
        f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
        f"exceeds atol={ATOL} rtol={RTOL}"
    )


def run_stage_decode(name, seed):
    """deltanet_in_proj_fused_tkg_fwd (T=1): attn_out / final_state / new_conv_state vs reference."""
    inp = make_inputs(T=1, seed=seed)
    ker_attn, ker_final, ker_conv = run_kernel(
        deltanet_in_proj_fused_tkg_fwd, inp, cores=2
    )
    ref_attn, ref_states, ref_cand = golden(inp)
    _check(f"{name}:attn_out", ker_attn, ref_attn)
    _check(f"{name}:final_state", ker_final, ref_states[-1])
    _check(f"{name}:new_conv_state", ker_conv, ref_cand[0])


def run_stage_verify(name, seed):
    """deltanet_in_proj_fused_tkg_fwd_state (T=2): attn_out / candidate_states / conv_cand vs ref."""
    inp = make_inputs(T=2, seed=seed)
    ker_attn, ker_cand, ker_conv = run_kernel(
        deltanet_in_proj_fused_tkg_fwd_state, inp, cores=2
    )
    ref_attn, ref_states, ref_cand = golden(inp)
    _check(f"{name}:attn_out", ker_attn, ref_attn)
    _check(f"{name}:candidate_states", ker_cand, ref_states)
    _check(f"{name}:conv_cand", ker_conv, ref_cand)


def run_shard_equality(name, seed):
    """Single-core [1] vs LNC=2 [2] on the verify path under the same hard gate (in_proj's contraction
    shard sendrecv-sums, so any difference is float reduction order, not a logic error)."""
    inp = make_inputs(T=2, seed=seed)
    a1, c1, k1 = run_kernel(deltanet_in_proj_fused_tkg_fwd_state, inp, cores=1)
    a2, c2, k2 = run_kernel(deltanet_in_proj_fused_tkg_fwd_state, inp, cores=2)
    ok = True
    for nm, x1, x2 in (("attn", a1, a2), ("cand", c1, c2), ("conv", k1, k2)):
        max_abs, max_rel = _metrics(x1, x2)
        sub_ok = torch.allclose(x1.float(), x2.float(), atol=ATOL, rtol=RTOL)
        print(
            f"[{name}:cores1_vs_cores2:{nm}] {'PASS' if sub_ok else 'FAIL'}  "
            f"max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}  (atol={ATOL} rtol={RTOL})"
        )
        ok = ok and sub_ok
    assert ok, f"{name}: cores=1 and cores=2 differ beyond the hard gate"


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_stage1_decode():
    run_stage_decode("stage1_decode_T1", seed=1)


def test_stage2_verify():
    run_stage_verify("stage2_verify_T2", seed=2)


def test_stage3_shard():
    run_shard_equality("stage3_shard", seed=3)


def main():
    run_stage_decode("stage1_decode_T1", seed=1)
    run_stage_verify("stage2_verify_T2", seed=2)
    run_shard_equality("stage3_shard", seed=3)
    print("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
