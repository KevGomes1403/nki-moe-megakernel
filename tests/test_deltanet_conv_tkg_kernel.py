"""Isolated correctness test for the DeltaNet causal conv1d TKG NKI kernel.

Calls `deltanet_conv_tkg_fwd` / `deltanet_conv_tkg_fwd_cand` / `deltanet_conv_tkg_fwd_sbuf`
DIRECTLY (no NeuronGatedDeltaNet module) and compares against a self-contained fp32 PyTorch
golden reference that replicates the model's depthwise 4-tap causal conv + SiLU, the per-head
q/k/v split (head_dim innermost), and the per-position candidate windows.

Staged from simplest to hardest so a failure localizes:
  1. decode T=1   -> deltanet_conv_tkg_fwd: q/k/v and new_state (== conv_cand[0]).
  2. verify T=2   -> deltanet_conv_tkg_fwd_cand: q/k/v and conv_cand[0:2]; exact window slices.
  3. decode == verify[T=1] cross-check.
  4. SBUF-output variant matches the HBM variant.

Tolerance: atol=1e-5, rtol=1e-2 on every output (q/k/v AND every candidate window).

Run:
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre \
    NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
    python -m models.qwen3_6_moe.tests.test_deltanet_conv_tkg_kernel
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.qwen3_6_moe.nki_kernels.nki_deltanet_conv_tkg import (  # noqa: E402
    deltanet_conv_tkg_fwd,
    deltanet_conv_tkg_fwd_cand,
    deltanet_conv_tkg_fwd_sbuf,
)

CONV_DIM = 2048  # per-rank conv_dim = 2*key_dim + value_dim (TP=4)
KEY_DIM = 512  # per-rank key_dim = q-dim = k-dim; value_dim = CONV_DIM - 2*KEY_DIM
K = 4  # linear_conv_kernel_dim
STATE_W = K - 1  # carried window width
HEAD_DIM = 128  # linear_{key,value}_head_dim
HK = KEY_DIM // HEAD_DIM  # q/k heads per rank
HV = (CONV_DIM - 2 * KEY_DIM) // HEAD_DIM  # v heads per rank
ATOL = 1e-5
RTOL = 1e-2


# ---------------------------------------------------------------------------
# fp32 golden reference: depthwise K-tap causal conv + SiLU, q/k/v split, candidate windows.
# Mirrors modeling_qwen36_a3b.py decode (:1082-1091) and verify_block / conv_window_candidates.
# ---------------------------------------------------------------------------
def ref_conv(qkv, conv_state, w):
    """qkv: (T, conv_dim); conv_state: (conv_dim, K-1); w: (conv_dim, K). All fp32.

    Returns (q (Hk,T,d), k (Hk,T,d), v (Hv,T,d), cand (T, conv_dim, K-1)) where cand[t] is the
    K-1 window ending at token t -- i.e. conv_input[:, t+1 : t+1+(K-1)].
    """
    T, conv_dim = qkv.shape
    x = qkv.transpose(0, 1)  # (conv_dim, T) channels-first
    conv_input = torch.cat([conv_state, x], dim=-1)  # (conv_dim, K-1+T)
    conv_out = torch.zeros_like(x)
    for k in range(K):
        conv_out = conv_out + w[:, k : k + 1] * conv_input[:, k : k + T]
    y = F.silu(conv_out)  # (conv_dim, T)

    # Split into q/k/v segments, then per-head with head_dim innermost: [h, t, i].
    def split(seg):  # seg: (n_heads*d, T) -> (n_heads, T, d)
        n_heads = seg.shape[0] // HEAD_DIM
        return seg.reshape(n_heads, HEAD_DIM, T).permute(0, 2, 1).contiguous()

    q = split(y[0:KEY_DIM])
    k = split(y[KEY_DIM : 2 * KEY_DIM])
    v = split(y[2 * KEY_DIM : conv_dim])
    cand = torch.stack(
        [conv_input[:, t + 1 : t + 1 + STATE_W] for t in range(T)], dim=0
    )
    return q, k, v, cand


def make_inputs(T, seed):
    """Random fp32 qkv (T, conv_dim), conv_state (conv_dim, K-1), conv_weight (conv_dim, K)."""
    torch.manual_seed(seed)
    qkv = torch.randn(T, CONV_DIM)
    conv_state = torch.randn(CONV_DIM, STATE_W)
    conv_weight = torch.randn(CONV_DIM, K)
    return qkv, conv_state, conv_weight


def run_kernel(fn, qkv, conv_state, conv_weight, cores=2):
    """Move inputs to the Neuron device, call the @nki.jit kernel, return CPU tensors.

    ``cores`` is the SPMD launch grid (cores=2 shards across both physical cores).
    """
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    args = [
        qkv.float().contiguous().to(dev),
        conv_state.float().contiguous().to(dev),
        conv_weight.float().contiguous().to(dev),
    ]
    # Kernel returns a unified qkv_out [NT,T,128]; slice into q/k/v (NKI can't return partial views).
    qkv_out, last = fn[cores](*args, KEY_DIM)
    q, k, v = qkv_out[0:HK], qkv_out[HK : 2 * HK], qkv_out[2 * HK :]
    return q.cpu(), k.cpu(), v.cpu(), last.cpu()


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


def run_stage_decode(name, seed):
    """deltanet_conv_tkg_fwd (T=1): q/k/v and new_state vs reference."""
    qkv, conv_state, conv_weight = make_inputs(T=1, seed=seed)
    ker_q, ker_k, ker_v, ker_state = run_kernel(deltanet_conv_tkg_fwd, qkv, conv_state, conv_weight)
    ref_q, ref_k, ref_v, ref_cand = ref_conv(qkv, conv_state, conv_weight)
    _check(f"{name}:q", ker_q, ref_q)
    _check(f"{name}:k", ker_k, ref_k)
    _check(f"{name}:v", ker_v, ref_v)
    _check(f"{name}:new_state", ker_state, ref_cand[0])


def run_stage_verify(name, seed):
    """deltanet_conv_tkg_fwd_cand (T=2): q/k/v and every candidate window vs reference."""
    qkv, conv_state, conv_weight = make_inputs(T=2, seed=seed)
    ker_q, ker_k, ker_v, ker_cand = run_kernel(
        deltanet_conv_tkg_fwd_cand, qkv, conv_state, conv_weight
    )
    ref_q, ref_k, ref_v, ref_cand = ref_conv(qkv, conv_state, conv_weight)
    _check(f"{name}:q", ker_q, ref_q)
    _check(f"{name}:k", ker_k, ref_k)
    _check(f"{name}:v", ker_v, ref_v)
    _check(f"{name}:conv_cand", ker_cand, ref_cand)

    # Exact window-slide structure: cand[0]==[s1,s2,x0], cand[1]==[s2,x0,x1].
    s = conv_state
    x = qkv.transpose(0, 1)  # (conv_dim, T)
    win0 = torch.stack([s[:, 1], s[:, 2], x[:, 0]], dim=-1)
    win1 = torch.stack([s[:, 2], x[:, 0], x[:, 1]], dim=-1)
    _check(f"{name}:cand0_slice", ker_cand[0], win0)
    _check(f"{name}:cand1_slice", ker_cand[1], win1)


def run_stage_cross_check(name, seed):
    """decode (T=1) output equals verify(T=1)."""
    qkv, conv_state, conv_weight = make_inputs(T=1, seed=seed)
    dec_q, dec_k, dec_v, dec_state = run_kernel(
        deltanet_conv_tkg_fwd, qkv, conv_state, conv_weight
    )
    ver_q, ver_k, ver_v, ver_cand = run_kernel(
        deltanet_conv_tkg_fwd_cand, qkv, conv_state, conv_weight
    )
    _check(f"{name}:q", dec_q, ver_q)
    _check(f"{name}:k", dec_k, ver_k)
    _check(f"{name}:v", dec_v, ver_v)
    _check(f"{name}:state", dec_state, ver_cand[0])


def run_stage_sbuf(name, seed):
    """deltanet_conv_tkg_fwd_sbuf (T=2): SBUF-output path matches the reference."""
    qkv, conv_state, conv_weight = make_inputs(T=2, seed=seed)
    ker_q, ker_k, ker_v, ker_cand = run_kernel(
        deltanet_conv_tkg_fwd_sbuf, qkv, conv_state, conv_weight, cores=1
    )
    ref_q, ref_k, ref_v, ref_cand = ref_conv(qkv, conv_state, conv_weight)
    _check(f"{name}:q", ker_q, ref_q)
    _check(f"{name}:k", ker_k, ref_k)
    _check(f"{name}:v", ker_v, ref_v)
    _check(f"{name}:conv_cand", ker_cand, ref_cand)


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_stage1_decode():
    run_stage_decode("stage1_decode_T1", seed=0)


def test_stage2_verify():
    run_stage_verify("stage2_verify_T2", seed=1)


def test_stage3_cross_check():
    run_stage_cross_check("stage3_decode_eq_verify", seed=2)


def test_stage4_sbuf():
    run_stage_sbuf("stage4_sbuf_T2", seed=3)


def main():
    run_stage_decode("stage1_decode_T1", seed=0)
    run_stage_verify("stage2_verify_T2", seed=1)
    run_stage_cross_check("stage3_decode_eq_verify", seed=2)
    run_stage_sbuf("stage4_sbuf_T2", seed=3)
    print("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
