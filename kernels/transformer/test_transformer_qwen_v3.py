"""
Test for transformer_qwen3_moe_tkg (Qwen3-MoE transformer layer megakernel).
Validates correctness against a pure PyTorch float32 reference implementation.

Uses parallel_model_trace (required for NKI collectives).
"""

import os
import sys

# MUST be set before any neuron/torch_xla imports
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

sys.path.insert(0, "/home/ubuntu/nki-moe")

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nki
from neuronx_distributed.trace import parallel_model_trace

from kernels.transformer.transformer_qwen_v3 import transformer_qwen3_moe_tkg_jit

# ---------------------------------------------------------------------------
# Model dimensions (match kernel constants)
# ---------------------------------------------------------------------------
B       = 1
S_tkg   = 1
H       = 2048
H0      = 128
H1      = 16
N_PRGS  = 2
H1_SHARD = 8

Hq_tp   = 8       # Q heads per TP rank
Hkv_tp  = 1       # KV heads per TP rank
d_head  = 128

# S_prior must be divisible by p_max * N_PRGS = 128*2 = 256
S_prior = 256

E = 128
K = 8
I = 192
EPS = 1e-6
TP  = 2

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
# Helper functions
# ---------------------------------------------------------------------------

def rms_norm(x, weight, eps=1e-6):
    """x: [..., H], weight: [H]"""
    x = x.float()
    weight = weight.float().reshape(-1)
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(rms + eps) * weight


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    """
    x: [B, heads, S, d_head]
    cos, sin: [d//2, B, S] -> expand to [B, 1, S, d_head]
    Contiguous layout: cos/sin for full d (two halves identical).
    """
    cos_bs = cos.permute(1, 2, 0)                                    # [B, S, d//2]
    sin_bs = sin.permute(1, 2, 0)                                    # [B, S, d//2]
    cos_full = torch.cat([cos_bs, cos_bs], dim=-1).unsqueeze(1)      # [B, 1, S, d]
    sin_full = torch.cat([sin_bs, sin_bs], dim=-1).unsqueeze(1)      # [B, 1, S, d]
    return x * cos_full + rotate_half(x) * sin_full


# ---------------------------------------------------------------------------
# Pure PyTorch reference (float32)
# ---------------------------------------------------------------------------

def qwen3_layer_ref(
    X,           # [B, S, H] float32
    W_qkv,       # [H, (Hq+2Hkv)*d] float32  — nkilib convention: x @ W_qkv
    W_out,       # [Hq*d, H] float32  — nkilib convention: x @ W_out (no T)
    gamma_attn,  # [1, H] float32
    q_norm_w,    # [d] float32
    k_norm_w,    # [d] float32
    K_cache,     # [B, d_head, S_prior] float32  (transposed, kernel convention)
    V_cache,     # [B, S_prior, d] float32
    cos,         # [d//2, B, S] float32
    sin,         # [d//2, B, S] float32
    position_ids,  # [B, 1] int64
    gamma_moe,   # [1, H] float32
    router_w,    # [H, E] float32
    gate_up_w,   # [E, H, 2*I] float32
    down_w,      # [E, I, H] float32
):
    BxS = B * S_tkg

    # ---- Attention ----
    X_normed = rms_norm(X, gamma_attn, eps=EPS)
    X_flat = X_normed.reshape(BxS, H).float()

    # QKV: W_qkv [H, (Hq+2Hkv)*d] nkilib convention: x @ W_qkv
    QKV = X_flat @ W_qkv.float()                                        # [BxS, (Hq+2Hkv)*d]

    q_dim = Hq_tp * d_head    # 1024
    k_dim = Hkv_tp * d_head   # 128

    Q     = QKV[:, :q_dim].reshape(B, S_tkg, Hq_tp, d_head).transpose(1, 2)
    K_new = QKV[:, q_dim:q_dim+k_dim].reshape(B, S_tkg, Hkv_tp, d_head).transpose(1, 2)
    V_new = QKV[:, q_dim+k_dim:].reshape(B, S_tkg, Hkv_tp, d_head).transpose(1, 2)

    # Per-head Q/K RMSNorm
    Q_norm = torch.stack([rms_norm(Q[:, h], q_norm_w, eps=EPS) for h in range(Hq_tp)], dim=1)
    K_norm = torch.stack([rms_norm(K_new[:, h], k_norm_w, eps=EPS) for h in range(Hkv_tp)], dim=1)

    # RoPE
    Q_rope = apply_rope(Q_norm, cos, sin)
    K_rope = apply_rope(K_norm, cos, sin)

    # KV cache update
    pos = position_ids[0, 0].item()
    # K_cache [B, d_head, S_prior] -> [B, 1, S_prior, d_head]
    K_cache_nt = K_cache.transpose(-1, -2).unsqueeze(1).float().clone()
    # V_cache [B, S_prior, d] -> [B, 1, S_prior, d]
    V_cache_copy = V_cache.unsqueeze(1).float().clone()
    K_cache_nt[:, :, pos:pos+1, :] = K_rope
    V_cache_copy[:, :, pos:pos+1, :] = V_new.float()

    # GQA
    K_full = K_cache_nt[:, :, :pos+1, :]
    V_full = V_cache_copy[:, :, :pos+1, :]
    K_exp = K_full.expand(B, Hq_tp, pos+1, d_head)
    V_exp = V_full.expand(B, Hq_tp, pos+1, d_head)

    scale = d_head ** -0.5
    scores = torch.einsum("bhsd,bhcd->bhsc", Q_rope, K_exp) * scale
    attn_w = torch.softmax(scores, dim=-1)
    attn_out = torch.einsum("bhsc,bhcd->bhsd", attn_w, V_exp)          # [B, Hq, S, d]

    # Output projection: W_out [Hq*d, H], attn_flat [BxS, Hq*d] @ W_out -> [BxS, H]
    attn_flat = attn_out.transpose(1, 2).reshape(BxS, q_dim).float()
    attn_proj = (attn_flat @ W_out.float()).reshape(B, S_tkg, H)

    # Residual #1
    out_attn = X.float() + attn_proj

    # ---- MoE ----
    X_moe_normed = rms_norm(out_attn, gamma_moe, eps=EPS)
    X_moe_flat = X_moe_normed.reshape(BxS, H).float()

    logits = X_moe_flat @ router_w.float()
    probs = torch.softmax(logits, dim=-1)
    top_vals, top_idx = torch.topk(probs, K, dim=-1)
    norm_weights = top_vals / top_vals.sum(dim=-1, keepdim=True)

    moe_out = torch.zeros(BxS, H, dtype=torch.float32)
    for t in range(BxS):
        x_t = X_moe_flat[t]
        for ki in range(K):
            eid = top_idx[t, ki].item()
            w_expert = norm_weights[t, ki].item()
            gate_w  = gate_up_w[eid, :, :I].float()
            up_w    = gate_up_w[eid, :, I:].float()
            down_w_e = down_w[eid].float()
            gate_val = x_t @ gate_w
            up_val   = x_t @ up_w
            inter = F.silu(gate_val) * up_val
            out_e = inter @ down_w_e
            moe_out[t] += w_expert * out_e

    moe_out = moe_out.reshape(B, S_tkg, H)
    return (out_attn + moe_out).float()


# ---------------------------------------------------------------------------
# NKI Module wrapper
# ---------------------------------------------------------------------------

class Qwen3MoEModule(nn.Module):
    def __init__(self, weights):
        super().__init__()
        for k, v in weights.items():
            self.register_buffer(k, v)

    def forward(self, X, cos, sin, mask_cache, mask_active, position_ids):
        return transformer_qwen3_moe_tkg_jit[2](
            X,
            self.W_qkv,
            self.W_out,
            self.gamma_attn,
            self.q_norm_w,
            self.k_norm_w,
            self.K_cache,
            self.V_cache,
            cos,
            sin,
            mask_cache,
            mask_active,
            position_ids,
            self.gamma_moe,
            self.router_w,
            self.gate_up_w,
            self.down_w,
            replica_groups=([0, 1],)
        )


class _WeightsFactory:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self):
        m = Qwen3MoEModule(self.weights)
        m.eval()
        return m, {}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_transformer_qwen3_moe():
    print("=" * 60)
    print("transformer_qwen3_moe_tkg correctness test")
    print("=" * 60)

    torch.manual_seed(42)
    dtype = torch.bfloat16

    weights = dict(
        W_qkv      = torch.randn(H, (Hq_tp + 2 * Hkv_tp) * d_head, dtype=dtype) * 0.02,
        W_out      = torch.randn(Hq_tp * d_head, H, dtype=dtype) * 0.02,
        gamma_attn = torch.ones(1, H, dtype=dtype),
        q_norm_w   = torch.ones(1, d_head, dtype=dtype),
        k_norm_w   = torch.ones(1, d_head, dtype=dtype),
        K_cache    = torch.zeros(B, d_head, S_prior, dtype=dtype),
        V_cache    = torch.zeros(B, S_prior, d_head, dtype=dtype),
        gamma_moe  = torch.ones(1, H, dtype=dtype),
        router_w   = torch.randn(H, E, dtype=dtype) * 0.02,
        gate_up_w  = torch.randn(E, H, 2 * I, dtype=dtype) * 0.02,
        down_w     = torch.randn(E, I, H, dtype=dtype) * 0.02,
    )

    X         = torch.randn(B, S_tkg, H, dtype=dtype) * 0.1
    # cos/sin [d//2, B, S_tkg]
    cos = torch.ones(d_head // 2, B, S_tkg, dtype=dtype)
    sin = torch.zeros(d_head // 2, B, S_tkg, dtype=dtype)
    # Attention masks [S_prior, B, q_heads, S_tkg] and [S_tkg, B, q_heads, S_tkg]
    mask_cache  = torch.zeros(S_prior, B, Hq_tp, S_tkg, dtype=dtype)
    mask_active = torch.zeros(S_tkg,   B, Hq_tp, S_tkg, dtype=dtype)
    position_ids = torch.zeros(B, 1, dtype=torch.int32)

    # --- Reference (float32) ---
    print("\nComputing PyTorch reference...")
    ref = qwen3_layer_ref(
        X.float(), weights["W_qkv"], weights["W_out"],
        weights["gamma_attn"], weights["q_norm_w"].reshape(d_head), weights["k_norm_w"].reshape(d_head),
        weights["K_cache"], weights["V_cache"],
        cos, sin,
        position_ids.long(),
        weights["gamma_moe"], weights["router_w"], weights["gate_up_w"], weights["down_w"],
    )
    ref_np = ref.numpy()
    print(f"Reference shape: {ref.shape}, range: [{ref_np.min():.4f}, {ref_np.max():.4f}]")

    # --- NKI kernel via parallel_model_trace ---
    example_args = (X, cos, sin, mask_cache, mask_active, position_ids)
    factory = _WeightsFactory(weights)

    workdir = "/tmp/qwen3_moe_test_workdir"
    os.makedirs(workdir, exist_ok=True)
    print(f"\nCompiling transformer_qwen3_moe_tkg kernel (tp_degree={TP}), workdir={workdir}...")
    try:
        traced = parallel_model_trace(
            factory, example_args, tp_degree=TP,
            compiler_workdir=workdir,
            compiler_args=COMPILER_ARGS,
            inline_weights_to_neff=True,
        )
        print("Compilation SUCCEEDED.")
    except Exception as e:
        print(f"Compilation FAILED: {e}")
        raise

    print("Running kernel inference...")
    try:
        result = traced(*example_args)
        result_np = result.cpu().float().numpy()
        print(f"Kernel output shape: {result.shape}, range: [{result_np.min():.4f}, {result_np.max():.4f}]")
    except Exception as e:
        print(f"Inference FAILED: {e}")
        raise

    # Comparison
    print("\n--- Numerical comparison ---")
    abs_diff = np.abs(result_np - ref_np)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    print(f"max_diff  = {max_diff:.6f}")
    print(f"mean_diff = {mean_diff:.6f}")

    rtol, atol = 2e-2, 2e-2
    print(f"\nTolerance: rtol={rtol}, atol={atol}")
    np.testing.assert_allclose(result_np, ref_np, rtol=rtol, atol=atol)
    print(f"PASS  max_diff={max_diff:.6f}")
    return True, max_diff


if __name__ == "__main__":
    passed, max_diff = test_transformer_qwen3_moe()
    print("\n" + "=" * 60)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"max_diff: {max_diff:.6f}")
    print("=" * 60)
    sys.exit(0 if passed else 1)
