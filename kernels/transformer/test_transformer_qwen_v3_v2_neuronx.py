"""
Test for transformer_qwen3_moe_tkg_v2 using torch_neuronx.trace.

Validates:
  1. DCE tripwire: output != input X (catches silent empty-kernel)
  2. torch.allclose vs float32 reference
"""

import os
import sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch.nn as nn
import torch.nn.functional as F
import nki
import torch_neuronx

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

COMPILER_ARGS = [
    "--target=trn2",
    "--enable-saturate-infinity",
    "--enable-mixed-precision-accumulation",
    "--model-type", "transformer",
    "-O3",
    "--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2",
    "--auto-cast=none",
    "--internal-enable-dge-levels", "vector_dynamic_offsets",
    "--internal-hlo2tensorizer-options=--verify-hlo=true",
    "--internal-max-instruction-limit=15000000",
]


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
    # --- Attention ---
    x_norm = rms_norm(X, gamma_attn)
    Q = (x_norm @ Wq.T).reshape(B, S_tkg, Hq_tp, d_head)
    K = (x_norm @ Wk.T).reshape(B, S_tkg, Hkv_tp, d_head)
    V = (x_norm @ Wv.T).reshape(B, S_tkg, Hkv_tp, d_head)

    Q = rms_norm(Q, q_norm_w)
    K = rms_norm(K, k_norm_w)

    cos = cos_bd.float().unsqueeze(1).unsqueeze(2)
    sin = sin_bd.float().unsqueeze(1).unsqueeze(2)
    Q = Q * cos + rotate_half(Q) * sin
    K = K * cos + rotate_half(K) * sin

    K_ctx = torch.cat([K_cache_t.permute(0,2,1), K.squeeze(1)], dim=1)
    V_ctx = torch.cat([V_cache, V.squeeze(1)], dim=1)

    Q_h = Q.permute(0, 2, 1, 3)
    K_h = K_ctx.unsqueeze(1).expand(-1, Hq_tp, -1, -1)
    V_h = V_ctx.unsqueeze(1).expand(-1, Hq_tp, -1, -1)

    scale = d_head ** -0.5
    scores = (Q_h @ K_h.transpose(-2, -1)) * scale
    attn_w = torch.softmax(scores, dim=-1)
    ctx = (attn_w @ V_h)
    ctx = ctx.permute(0, 2, 1, 3).reshape(B, S_tkg, Hq_tp * d_head)

    attn_out = ctx @ Wo.T
    attn_out = attn_out * 2.0  # simulate allreduce across TP=2

    X2 = X.float() + attn_out

    # --- MoE ---
    x_moe = rms_norm(X2, gamma_moe)
    router_logits = x_moe.reshape(B * S_tkg, H) @ router_w.float()
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
    moe_out = moe_out * 2.0  # simulate allreduce

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


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    dtype = torch.bfloat16

    weights = dict(
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

    example_inputs = (X, cos, sin, position_ids)

    print("=" * 60)
    print("transformer_qwen3_moe_tkg_v2 — torch_neuronx.trace")
    print(f"  H={H}, S_prior={S_prior}, TP={TP}, LNC=2")
    print("=" * 60)

    model = Qwen3MoEV2Module(weights)
    model.eval()

    print("\nCompiling with torch_neuronx.trace...")
    try:
        traced = torch_neuronx.trace(
            model,
            example_inputs,
            compiler_args=COMPILER_ARGS,
            compiler_workdir="/tmp/qwen3_moe_v2_neuronx_workdir",
        )
    except Exception as e:
        print(f"COMPILE FAIL: {e}")
        import traceback; traceback.print_exc()
        return 1

    print("Compilation SUCCEEDED. Running inference...")
    try:
        result = traced(*example_inputs)
    except Exception as e:
        print(f"INFERENCE FAIL: {e}")
        import traceback; traceback.print_exc()
        return 1

    r = result.float()
    print(f"Output shape: {r.shape}")
    print(f"Output stats: min={r.min():.4f}  max={r.max():.4f}  "
          f"mean={r.mean():.4f}  std={r.std():.4f}")

    # DCE tripwire
    if torch.allclose(r, X.float(), atol=1e-3, rtol=1e-3):
        print("\nFAIL: DCE tripwire — output == input X (kernel body eliminated)")
        return 1
    print("DCE tripwire: PASS")

    # Torch reference
    print("\nComputing torch reference...")
    K_cache_t = weights["K_cache"].squeeze(1).permute(0, 2, 1).float()  # [B, d, S_prior]
    V_cache_r = weights["V_cache"].squeeze(1).float()                    # [B, S_prior, d]

    ref = qwen3_layer_ref(
        X.float(),
        weights["Wq"].float(), weights["Wk"].float(),
        weights["Wv"].float(), weights["Wo"].float(),
        weights["gamma_attn"].float(),
        weights["q_norm_weight"].float(), weights["k_norm_weight"].float(),
        K_cache_t, V_cache_r,
        cos.float(), sin.float(),
        weights["gamma_moe"].float(), weights["router_w"].float(),
        weights["gate_up_w"].float(), weights["down_w"].float(),
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
