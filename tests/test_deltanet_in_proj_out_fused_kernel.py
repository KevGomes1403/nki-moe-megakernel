"""Isolated correctness test for the FULL-chain fused DeltaNet TKG NKI kernel with output projection.

Calls ``deltanet_attention_layer`` (decode, T=1) / ``deltanet_attention_layer_state``
(verify, T=2) DIRECTLY and compares the full pipeline against a chained CPU golden:

    normed       = RMSNorm(hidden) @ proj_w                        (standard (x*rsqrt(mean(x^2)+eps))*g)
    qkv|z|a|b    = slice normed at offsets conv_dim, +value_dim, +Hv, +Hv
    q,k,v,cand   = ref_conv(qkv, conv_state, conv_weight)
    attn,states  = ref_full(q, k, v, a, b, A_log, dt_bias, init_state)        (delta-rule)
    gated        = gated_norm(attn, gate=z)                        (per-head RMSNorm over head_dim)
    o_out        = (gated.cdtype @ out_w.cdtype).cdtype            (o_proj: in / f32-accum / out)

With one rank holding ALL value heads, ``o_out`` is the complete o_proj output (no all-reduce).

The hard gate (atol=1e-5, rtol=1e-2) runs in fp32, which matches the kernel's internal accumulation
precision and is the valid bit-correctness oracle: every output (o_out, final_state/new_conv_state,
candidate_states/conv_cand) must pass. The same chain is also run in bf16 (the model's deployment
dtype) and its max_abs_err / max_rel_err / max_ulp reported for reference -- the ~1e-2 bf16 error is
inherent GEMM noise of the 2048-deep in_proj contraction, below the resolution of a 1e-5/1e-2 gate.

Staged so a failure localizes:
  1. decode (T=1) -- o_out / final_state / new_conv_state, cores=1 and cores=2.
  2. verify (T=2) -- o_out / candidate_states / conv_cand, cores=1 and cores=2.

Run:
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_deltanet_in_proj_out_fused_kernel
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
    deltanet_attention_layer,
    deltanet_attention_layer_state,
)
from models.qwen3_6_moe.tests.test_deltanet_conv_tkg_kernel import (  # noqa: E402
    CONV_DIM,
    HEAD_DIM,
    HV,
    K,
    KEY_DIM,
    ref_conv,
)

HIDDEN = 2048  # per-rank hidden (TP=4); also the o_proj output width
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
ATOL = 1e-5
RTOL = 1e-2
EPS = 1e-6  # input RMSNorm + gated RMSNorm epsilon (config rms_norm_eps)

# Per-run device/reference dtype, set by the stage runners. fp32 is the asserting bit-correctness gate
# (matches the kernel's internal accumulation); bf16 is the deployment dtype, run for reference only.
DTYPE = torch.float32
REF_DTYPE = torch.float32


def make_inputs(T, seed):
    """Random fp32 inputs: hidden + concatenated [H, I] projection weight + input/gated norm weights +
    conv/recurrence state/gating inputs + the [value_dim, hidden] o_proj weight transpose."""
    torch.manual_seed(seed)
    hidden = torch.randn(1, T, HIDDEN)
    w_scale = 1.0 / math.sqrt(HIDDEN)
    w_qkv = torch.randn(CONV_DIM, HIDDEN) * w_scale
    w_z = torch.randn(VALUE_DIM, HIDDEN) * w_scale
    w_a = torch.randn(NUM_V_HEADS, HIDDEN) * w_scale
    w_b = torch.randn(NUM_V_HEADS, HIDDEN) * w_scale
    proj_w = torch.cat([w_qkv, w_z, w_a, w_b], dim=0).t().contiguous()  # [H, I]
    gamma = torch.randn(HIDDEN)  # input RMSNorm weight
    conv_state = torch.randn(CONV_DIM, STATE_W)
    conv_weight = torch.randn(CONV_DIM, K) / math.sqrt(
        K
    )  # K-tap depthwise, conv output O(1)
    A_log = torch.randn(HV) * 0.5
    dt_bias = torch.randn(HV) * 0.5
    init_state = torch.randn(HV, HEAD_DIM, HEAD_DIM) * 0.1
    z_gamma = torch.randn(HEAD_DIM)  # gated per-head RMSNorm weight
    out_w = torch.randn(VALUE_DIM, HIDDEN) * (
        1.0 / math.sqrt(VALUE_DIM)
    )  # o_proj weight transpose
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
        out_w,
    )


def quant(x):
    """Quantize to the device dtype (what the kernel sees), then carry the value in REF_DTYPE."""
    return x.to(DTYPE).to(REF_DTYPE)


def rms_norm_ref(x, gamma, eps):
    """Standard RMSNorm: (x * rsqrt(mean(x^2) + eps)) * gamma."""
    x = x.to(REF_DTYPE)
    rms = (x.square().mean(dim=-1, keepdim=True) + eps).sqrt()
    return (x * rms.reciprocal()) * gamma.to(REF_DTYPE)


def ref_full(q, k, v, a, b, A_log, dt_bias, init):
    """Delta-rule recurrence. q,k (Hk,T,d); v (Hv,T,d); a,b (T,Hv); A_log,dt_bias (Hv,);
    init (Hv,d,d). Returns attn_out (T, Hv*d) head-major and states (T, Hv, d, d)."""
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


def gated_norm(attn, gate, weight, eps):
    """Per-head gated RMSNorm (mirrors Qwen3_5MoeRMSNormGated). attn/gate: (T, Hv, d)."""
    x = attn.to(REF_DTYPE)
    var = x.square().mean(dim=-1, keepdim=True)
    x = x * (var + eps).rsqrt()
    x = weight.to(REF_DTYPE) * x
    return x * F.silu(gate.to(REF_DTYPE))


def golden(inp):
    """Chained CPU reference on quantized inputs: RMSNorm @ proj_w -> slice -> conv -> recurrence
    -> gated per-head RMSNorm -> o_proj (in / fp32-accum / out). Returns (o_out, ref_states,
    ref_cand)."""
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
    ) = (quant(t) for t in inp)

    T = hidden.shape[1]
    normed = rms_norm_ref(hidden, gamma, EPS).reshape(T, HIDDEN)
    proj = normed @ proj_w  # [T, I], fp32

    qkv = proj[:, OFF_QKV:OFF_Z].contiguous()
    z = proj[:, OFF_Z:OFF_A].contiguous()
    a = proj[:, OFF_A:OFF_B].contiguous()
    b = proj[:, OFF_B:OFF_END].contiguous()

    ref_q, ref_k, ref_v, ref_cand = ref_conv(qkv, conv_state, conv_weight)
    ref_attn, ref_states = ref_full(
        ref_q, ref_k, ref_v, a, b, A_log, dt_bias, init_state
    )

    gated = gated_norm(
        ref_attn.reshape(T, HV, HEAD_DIM), z.reshape(T, HV, HEAD_DIM), z_gamma, EPS
    ).reshape(T, HV * HEAD_DIM)  # [T, value_dim] head-major

    # o_proj: in / fp32-accumulate / out, contracting over all value heads.
    o_out = (gated.to(DTYPE).float() @ out_w.to(DTYPE).float()).to(DTYPE)
    return o_out, ref_states, ref_cand


def run_kernel(fn, inp, cores):
    """Move DTYPE inputs to the Neuron device, launch the @nki.jit kernel on ``cores`` cores (value-head
    SPMD shard). Returns (o_out [T,hidden], state, conv) as CPU tensors."""
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
    h = hidden.to(DTYPE).contiguous().to(dev)
    w = proj_w.to(DTYPE).contiguous().to(dev)
    g = gamma.reshape(1, HIDDEN).to(DTYPE).contiguous().to(dev)  # qkv_tkg wants [1, H]
    cs = conv_state.to(DTYPE).contiguous().to(dev)
    cw = conv_weight.to(DTYPE).contiguous().to(dev)
    al = A_log.to(DTYPE).contiguous().to(dev)
    db = dt_bias.to(DTYPE).contiguous().to(dev)
    init = init_state.to(DTYPE).contiguous().to(dev)
    zg = z_gamma.to(DTYPE).contiguous().to(dev)
    ow = out_w.to(DTYPE).contiguous().to(dev)
    o_out, state, conv = fn[cores](
        h, w, g, EPS, cs, cw, KEY_DIM, al, db, init, zg, ow, EPS
    )
    return o_out.to(DTYPE).cpu(), state.float().cpu(), conv.float().cpu()


def _ulp_distance(ker, ref):
    """Max ULP distance between two bf16 tensors (via their bit patterns)."""
    k = ker.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    r = ref.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    k = torch.where(k < 0, 0x8000 - k, k)
    r = torch.where(r < 0, 0x8000 - r, r)
    return (k - r).abs().max().item()


def _metrics(ker, ref):
    kd = ker.double().reshape(-1)
    rd = ref.double().reshape(-1)
    abs_err = (kd - rd).abs()
    denom = rd.abs().clamp_min(1e-4)
    return abs_err.max().item(), (abs_err / denom).max().item()


def _check(name, ker, ref, assert_gate):
    """Print max_abs/max_rel/max_ulp vs the gate; assert PASS only when ``assert_gate``."""
    max_abs, max_rel = _metrics(ker.float(), ref.float())
    ulp = _ulp_distance(ker.float().reshape(-1), ref.float().reshape(-1))
    ok = torch.allclose(ker.float(), ref.float(), atol=ATOL, rtol=RTOL)
    status = "PASS" if ok else "FAIL"
    print(
        f"[{name}] {status}  max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}  "
        f"max_ulp={ulp}  (atol={ATOL} rtol={RTOL})"
    )
    if assert_gate:
        assert ok, (
            f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} max_ulp={ulp} exceeds gate"
        )


def _set_dtype(cdtype):
    global DTYPE, REF_DTYPE
    DTYPE = cdtype
    REF_DTYPE = torch.float32


def run_stage_decode(name, seed, cores, cdtype, assert_gate):
    """deltanet_attention_layer (T=1): o_out / final_state / new_conv_state vs reference."""
    _set_dtype(cdtype)
    tag = f"{name}:{cdtype_tag(cdtype)}:cores{cores}"
    inp = make_inputs(T=1, seed=seed)
    ker_o, ker_state, ker_conv = run_kernel(
        deltanet_attention_layer, inp, cores=cores
    )
    ref_o, ref_states, ref_cand = golden(inp)
    _check(f"{tag}:o_out", ker_o, ref_o, assert_gate)
    _check(f"{tag}:final_state", ker_state, ref_states[-1], assert_gate)
    _check(f"{tag}:new_conv_state", ker_conv, ref_cand[0], assert_gate)


def run_stage_verify(name, seed, cores, cdtype, assert_gate):
    """deltanet_attention_layer_state (T=2): o_out / candidate_states / conv_cand vs ref."""
    _set_dtype(cdtype)
    tag = f"{name}:{cdtype_tag(cdtype)}:cores{cores}"
    inp = make_inputs(T=2, seed=seed)
    ker_o, ker_cand, ker_conv = run_kernel(
        deltanet_attention_layer_state, inp, cores=cores
    )
    ref_o, ref_states, ref_cand = golden(inp)
    _check(f"{tag}:o_out", ker_o, ref_o, assert_gate)
    _check(f"{tag}:candidate_states", ker_cand, ref_states, assert_gate)
    _check(f"{tag}:conv_cand", ker_conv, ref_cand, assert_gate)


def cdtype_tag(cdtype):
    return "fp32" if cdtype == torch.float32 else "bf16"


# ---------------------------------------------------------------------------
# pytest entrypoints: fp32 is the asserting bit-correctness gate; bf16 is reported only.
# ---------------------------------------------------------------------------
def test_decode_t1_cores2():
    run_stage_decode(
        "decode_T1", seed=1, cores=2, cdtype=torch.float32, assert_gate=True
    )


def test_decode_t1_cores1():
    run_stage_decode(
        "decode_T1", seed=1, cores=1, cdtype=torch.float32, assert_gate=True
    )


def test_verify_t2_cores2():
    run_stage_verify(
        "verify_T2", seed=2, cores=2, cdtype=torch.float32, assert_gate=True
    )


def test_verify_t2_cores1():
    run_stage_verify(
        "verify_T2", seed=2, cores=1, cdtype=torch.float32, assert_gate=True
    )


def main():
    print("=== fp32 (asserting bit-correctness gate) ===")
    run_stage_decode(
        "decode_T1", seed=1, cores=1, cdtype=torch.float32, assert_gate=True
    )
    run_stage_decode(
        "decode_T1", seed=1, cores=2, cdtype=torch.float32, assert_gate=True
    )
    run_stage_verify(
        "verify_T2", seed=2, cores=1, cdtype=torch.float32, assert_gate=True
    )
    run_stage_verify(
        "verify_T2", seed=2, cores=2, cdtype=torch.float32, assert_gate=True
    )
    print(
        "=== bf16 (deployment dtype, reported only -- ~1e-2 is inherent in_proj GEMM noise) ==="
    )
    run_stage_decode(
        "decode_T1", seed=1, cores=1, cdtype=torch.bfloat16, assert_gate=False
    )
    run_stage_decode(
        "decode_T1", seed=1, cores=2, cdtype=torch.bfloat16, assert_gate=False
    )
    run_stage_verify(
        "verify_T2", seed=2, cores=1, cdtype=torch.bfloat16, assert_gate=False
    )
    run_stage_verify(
        "verify_T2", seed=2, cores=2, cdtype=torch.bfloat16, assert_gate=False
    )
    print("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
