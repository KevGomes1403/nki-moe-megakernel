"""
Test for transformer_qwen3_moe_tkg_v2 — v13bc attention (no sendrecv) + bare AllReduce.

Validates:
  1. DCE tripwire: output != input X (catches silent empty-kernel)
  2. torch.allclose vs float32 reference
"""

import os
import sys
import tempfile

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch.nn as nn
import torch.nn.functional as F
import nki
from neuronx_distributed.trace import parallel_model_trace

from kernels.transformer.transformer_qwen_v3_v2 import transformer_qwen3_moe_tkg_v2_jit

# ---------------------------------------------------------------------------
# Dimensions — match kernel constants
# ---------------------------------------------------------------------------
B       = 1
S_tkg   = 1
H       = 2048
Hq_tp   = 8       # Q heads per TP rank
Hkv_tp  = 1       # KV heads per TP rank
d_head  = 128
S_prior = 256
E       = 128
I       = 192
TP      = 2
EPS     = 1e-6

COMPILER_ARGS = (
    "--enable-saturate-infinity "
    "--enable-mixed-precision-accumulation "
    "--model-type transformer "
    "-O3 "
    "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-hlo2tensorizer-options='--verify-hlo=true' "
    "--internal-max-instruction-limit=15000000"
)


# ---------------------------------------------------------------------------
# PyTorch reference (float32)
# ---------------------------------------------------------------------------

def rms_norm(x, weight, eps=EPS):
    x = x.float(); w = weight.float().reshape(-1)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

def qwen3_layer_ref(X, Wq, Wk, Wv, Wo, gamma_attn, q_norm_w, k_norm_w,
                    K_cache_t, V_cache, cos_bd, sin_bd,
                    gamma_moe, router_w, gate_up_w, down_w):
    """
    X:         [B, S, H]   float32
    Wq:        [Hq*d, H]   float32
    Wk:        [Hkv*d, H]  float32
    Wv:        [Hkv*d, H]  float32
    Wo:        [Hq*d, H]   float32
    K_cache_t: [B, d, S_prior] float32  (transposed — kernel convention)
    V_cache:   [B, S_prior, d] float32
    cos_bd:    [B, d]      float32  (pre-indexed, v13bc convention)
    sin_bd:    [B, d]      float32
    """
    # --- Attention ---
    x_norm = rms_norm(X, gamma_attn)                      # [B, S, H]
    Q = (x_norm @ Wq.T).reshape(B, S_tkg, Hq_tp, d_head) # [B, S, Hq, d]
    K = (x_norm @ Wk.T).reshape(B, S_tkg, Hkv_tp, d_head)
    V = (x_norm @ Wv.T).reshape(B, S_tkg, Hkv_tp, d_head)

    Q = rms_norm(Q, q_norm_w)
    K = rms_norm(K, k_norm_w)

    # RoPE: cos_bd [B, d] → [B, 1, 1, d]
    cos = cos_bd.float().unsqueeze(1).unsqueeze(2)
    sin = sin_bd.float().unsqueeze(1).unsqueeze(2)
    Q = Q * cos + rotate_half(Q) * sin
    K = K * cos + rotate_half(K) * sin

    # Flash-decode style: attend over KV cache + new K,V
    # K_cache_t: [B, d, S_prior], V_cache: [B, S_prior, d]
    K_ctx = torch.cat([K_cache_t.permute(0,2,1), K.squeeze(1)], dim=1)  # [B, S+1, d]  (Hkv=1)
    V_ctx = torch.cat([V_cache, V.squeeze(1)], dim=1)                    # [B, S+1, d]

    Q_h = Q.permute(0, 2, 1, 3)          # [B, Hq, S, d]
    K_h = K_ctx.unsqueeze(1).expand(-1, Hq_tp, -1, -1)  # [B, Hq, S_ctx, d]
    V_h = V_ctx.unsqueeze(1).expand(-1, Hq_tp, -1, -1)

    scale = d_head ** -0.5
    scores = (Q_h @ K_h.transpose(-2, -1)) * scale       # [B, Hq, S, S_ctx]
    attn_w = torch.softmax(scores, dim=-1)
    ctx = (attn_w @ V_h)                                 # [B, Hq, S, d]
    ctx = ctx.permute(0, 2, 1, 3).reshape(B, S_tkg, Hq_tp * d_head)

    # Output projection — TP allreduce simulated by *2 (both ranks same weights)
    attn_out = ctx @ Wo.T                                # [B, S, H]
    attn_out = attn_out * 2.0                            # simulate allreduce across TP=2

    # Residual
    X2 = X.float() + attn_out

    # --- MoE ---
    x_moe = rms_norm(X2, gamma_moe)                     # [B, S, H]
    router_logits = x_moe.reshape(B * S_tkg, H) @ router_w.float()  # [T, E]
    router_scores, router_ids = torch.topk(torch.softmax(router_logits, dim=-1), k=8, dim=-1)
    norm_scores = router_scores / router_scores.sum(-1, keepdim=True)

    x_flat = x_moe.reshape(B * S_tkg, H)
    moe_out = torch.zeros_like(x_flat)
    for t in range(B * S_tkg):
        for ki in range(8):
            eid = router_ids[t, ki].item()
            w   = norm_scores[t, ki].item()
            g   = gate_up_w[eid, :, :I].float()
            u   = gate_up_w[eid, :, I:].float()
            inter = F.silu(x_flat[t] @ g) * (x_flat[t] @ u)
            moe_out[t] += w * (inter @ down_w[eid].float())

    moe_out = moe_out.reshape(B, S_tkg, H)
    # Simulate allreduce *2
    moe_out = moe_out * 2.0

    return (X2 + moe_out).float()


# ---------------------------------------------------------------------------
# NKI Module wrapper
# ---------------------------------------------------------------------------

class Qwen3MoEV2Module(nn.Module):
    def __init__(self, weights):
        super().__init__()
        for k, v in weights.items():
            self.register_buffer(k, v)

    def forward(self, X, cos, sin, position_ids):
        return transformer_qwen3_moe_tkg_v2_jit[2](
            X,
            self.Wq, self.Wk, self.Wv, self.Wo,
            self.q_norm_weight, self.k_norm_weight,
            self.K_cache, self.V_cache,
            cos, sin, position_ids,
            self.gamma_moe, self.router_w, self.gate_up_w, self.down_w,
            replica_groups=([0, 1],),
        )


class _WeightsFactory:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self):
        m = Qwen3MoEV2Module(self.weights)
        m.eval()
        return m, {}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def main():
    global _WEIGHTS
    torch.manual_seed(42)
    dtype = torch.bfloat16

    _WEIGHTS = dict(
        Wq           = torch.randn(Hq_tp * d_head, H, dtype=dtype) * 0.02,
        Wk           = torch.randn(Hkv_tp * d_head, H, dtype=dtype) * 0.02,
        Wv           = torch.randn(Hkv_tp * d_head, H, dtype=dtype) * 0.02,
        Wo           = torch.randn(Hq_tp * d_head, H, dtype=dtype) * 0.02,
        q_norm_weight= torch.ones(d_head, dtype=dtype),
        k_norm_weight= torch.ones(d_head, dtype=dtype),
        K_cache      = torch.zeros(B, 1, S_prior, d_head, dtype=dtype),
        V_cache      = torch.zeros(B, 1, S_prior, d_head, dtype=dtype),
        gamma_attn   = torch.ones(1, H, dtype=dtype),
        gamma_moe    = torch.ones(1, H, dtype=dtype),
        router_w     = torch.randn(H, E, dtype=dtype) * 0.02,
        gate_up_w    = torch.randn(E, H, 2 * I, dtype=dtype) * 0.02,
        down_w       = torch.randn(E, I, H, dtype=dtype) * 0.02,
    )

    X            = torch.randn(B, S_tkg, H, dtype=dtype) * 0.1
    cos          = torch.ones(B, d_head, dtype=dtype)
    sin          = torch.zeros(B, d_head, dtype=dtype)
    position_ids = torch.zeros(B, 1, dtype=torch.int32)

    example_args = (X, cos, sin, position_ids)

    print("=" * 60)
    print("transformer_qwen3_moe_tkg_v2 — v13bc attn, no sendrecv")
    print(f"  tp_degree={TP}, LNC=2, H={H}, S_prior={S_prior}")
    print("=" * 60)

    workdir = "/tmp/qwen3_moe_v2_test_workdir"
    os.makedirs(workdir, exist_ok=True)

    print("\nCompiling...")
    factory = _WeightsFactory(_WEIGHTS)
    try:
        traced = parallel_model_trace(
            factory, example_args,
            tp_degree=TP,
            compiler_workdir=workdir,
            compiler_args=COMPILER_ARGS,
            inline_weights_to_neff=True,
        )
    except Exception as e:
        print(f"COMPILE FAIL: {e}")
        import traceback; traceback.print_exc()
        return 1

    print("Compilation SUCCEEDED. Running inference...")
    try:
        result = traced(*example_args)
    except Exception as e:
        print(f"INFERENCE FAIL: {e}")
        import traceback; traceback.print_exc()
        return 1

    r = result.float()
    print(f"Output shape: {r.shape}")
    print(f"Output stats: min={r.min():.4f}  max={r.max():.4f}  "
          f"mean={r.mean():.4f}  std={r.std():.4f}")

    # DCE tripwire
    x_in = X.float()
    if torch.allclose(r, x_in, atol=1e-3, rtol=1e-3):
        print("\nFAIL: DCE tripwire — output == input X (kernel body eliminated)")
        return 1
    print("DCE tripwire: PASS")

    # Torch reference
    print("\nComputing torch reference...")
    K_cache_t = _WEIGHTS["K_cache"].squeeze(1).permute(0, 2, 1).float()  # [B, d, S_prior]
    V_cache_r = _WEIGHTS["V_cache"].squeeze(1).float()                    # [B, S_prior, d]

    ref = qwen3_layer_ref(
        X.float(),
        _WEIGHTS["Wq"].float(), _WEIGHTS["Wk"].float(),
        _WEIGHTS["Wv"].float(), _WEIGHTS["Wo"].float(),
        _WEIGHTS["gamma_attn"].float(),
        _WEIGHTS["q_norm_weight"].float(), _WEIGHTS["k_norm_weight"].float(),
        K_cache_t, V_cache_r,
        cos.float(), sin.float(),
        _WEIGHTS["gamma_moe"].float(), _WEIGHTS["router_w"].float(),
        _WEIGHTS["gate_up_w"].float(), _WEIGHTS["down_w"].float(),
    )

    print(f"Reference stats: min={ref.min():.4f}  max={ref.max():.4f}  "
          f"mean={ref.mean():.4f}  std={ref.std():.4f}")
    abs_err = (r - ref).abs()
    print(f"Abs err: max={abs_err.max():.4e}  mean={abs_err.mean():.4e}")

    atol, rtol = 0.1, 0.1
    if torch.allclose(r, ref, atol=atol, rtol=rtol):
        print(f"\nRESULT: PASS (allclose atol={atol} rtol={rtol})")
        return 0
    else:
        print(f"\nRESULT: FAIL (allclose atol={atol} rtol={rtol})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
