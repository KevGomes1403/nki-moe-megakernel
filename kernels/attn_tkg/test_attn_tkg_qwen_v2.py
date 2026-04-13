"""
Correctness + benchmark test for attn_tkg_qwen_v2 NKI kernel.

Environment variables must be set before any neuron/torch_xla imports.
"""

import os
import sys

# MUST be set before any neuron/torch_xla imports
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

sys.path.insert(0, "/home/ubuntu/nki-moe")

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nki
from neuronx_distributed.trace import parallel_model_trace

from kernels.attn_tkg.attn_tkg_qwen_v2 import attn_tkg_qwen_v2

# ---------------------------------------------------------------------------
# Shapes (match transformer test)
# ---------------------------------------------------------------------------
B, S_TKG, H = 1, 1, 2048
D_HEAD  = 128
Q_HEADS = 8        # per TP rank
I_MLP   = 192
S_CTX   = 512      # KV cache entries
TP      = 2

COMPILER_ARGS = (
    "--enable-saturate-infinity "
    "--enable-mixed-precision-accumulation "
    "--model-type transformer "
    "-O1 "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-max-instruction-limit=15000000"
)


# ---------------------------------------------------------------------------
# PyTorch reference — full Qwen3 attention with GQA, RMSNorm, RoPE, Wo proj
# ---------------------------------------------------------------------------

def rotate_half(x):
    """Rotates half the hidden dims: [-x2, x1] where x = [x1, x2]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rmsnorm_per_head(x, weight, eps=1e-6):
    """Apply RMSNorm per head. x: [..., d], weight: [d]"""
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(variance + eps)
    return (x_normed * weight.float()).to(x.dtype)


def pytorch_reference(
    hidden_states,   # [B, 1, H] bf16
    Wq,              # [Hq_tp*d, H] bf16
    Wk,              # [Hkv_tp*d, H] bf16
    Wv,              # [Hkv_tp*d, H] bf16
    Wo,              # [Hq_tp*d, H] bf16
    q_norm_weight,   # [d] bf16
    k_norm_weight,   # [d] bf16
    K_cache,         # [B, Hkv_tp, S_prior, d] bf16
    V_cache,         # [B, Hkv_tp, S_prior, d] bf16
    cos_at_pos,      # [B, d] bf16  (pre-indexed by position_ids)
    sin_at_pos,      # [B, d] bf16
    d=128,
):
    """Pure PyTorch reference implementation. Returns [B, 1, H] bf16."""
    B, _, H_full = hidden_states.shape
    Hq_out = Wq.shape[0]
    Hkv_out = Wk.shape[0]
    Hq_tp = Hq_out // d
    Hkv_tp = Hkv_out // d
    gqa = Hq_tp // Hkv_tp
    S_prior = K_cache.shape[2]

    # QKV projection: [B, 1, H] @ W.T -> [B, 1, Hq_out/Hkv_out]
    hs = hidden_states.float()
    Q = hs @ Wq.float().T  # [B, 1, Hq_out]
    K = hs @ Wk.float().T  # [B, 1, Hkv_out]
    V = hs @ Wv.float().T  # [B, 1, Hkv_out]

    # Reshape to [B, heads, 1, d]
    Q = Q.reshape(B, Hq_tp, 1, d)   # [B, 8, 1, 128]
    K = K.reshape(B, Hkv_tp, 1, d)  # [B, 1, 1, 128]
    V = V.reshape(B, Hkv_tp, 1, d)  # [B, 1, 1, 128]

    # Per-head RMSNorm on Q and K
    Q_norm = torch.zeros_like(Q)
    K_norm = torch.zeros_like(K)
    for h in range(Hq_tp):
        Q_norm[:, h, :, :] = apply_rmsnorm_per_head(Q[:, h, :, :], q_norm_weight)
    for h in range(Hkv_tp):
        K_norm[:, h, :, :] = apply_rmsnorm_per_head(K[:, h, :, :], k_norm_weight)

    # RoPE — apply same embedding to all heads (shared position)
    # cos_at_pos: [B, d], expand to [B, 1, 1, d]
    cos = cos_at_pos.float().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, d]
    sin = sin_at_pos.float().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, d]

    Q_rope = Q_norm.float() * cos + rotate_half(Q_norm.float()) * sin  # [B, Hq, 1, d]
    K_rope = K_norm.float() * cos + rotate_half(K_norm.float()) * sin  # [B, Hkv, 1, d]

    scale = 1.0 / math.sqrt(d)

    # Flash attention with KV cache (GQA)
    output = torch.zeros(B, Hq_tp, d, dtype=torch.float32)

    for b in range(B):
        for q_h in range(Hq_tp):
            kv_h = q_h // gqa
            q_vec = Q_rope[b, q_h, 0, :]  # [d]

            # Concatenate cached + active K,V
            K_full = torch.cat([
                K_cache[b, kv_h].float(),       # [S_prior, d]
                K_rope[b, kv_h, 0:1, :].float(), # [1, d]
            ], dim=0)  # [S_prior+1, d]

            V_full = torch.cat([
                V_cache[b, kv_h].float(),        # [S_prior, d]
                V[b, kv_h, 0:1, :].float(),      # [1, d] (no RoPE on V)
            ], dim=0)  # [S_prior+1, d]

            # Attention scores [S_prior+1]
            scores = (K_full @ q_vec) * scale  # [S_prior+1]

            # Softmax
            attn_weights = F.softmax(scores, dim=0)  # [S_prior+1]

            # Weighted sum of V [d]
            out_vec = (attn_weights.unsqueeze(-1) * V_full).sum(dim=0)  # [d]
            output[b, q_h, :] = out_vec

    # Reshape output [B, Hq_tp, d] -> [B, 1, Hq_tp*d]
    attn_flat = output.reshape(B, 1, Hq_tp * d)  # [B, 1, 1024]

    # Output projection: [B, 1, Hq_tp*d] @ Wo -> [B, 1, H]
    # Wo shape: [Hq_tp*d, H] = [1024, 2048], attn_flat: [B, 1, 1024]
    attn_proj = attn_flat @ Wo.float()  # [B, 1, H]

    return attn_proj.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# NKI Module wrapper
# ---------------------------------------------------------------------------

class AttnModule(nn.Module):
    def __init__(self, weights):
        super().__init__()
        for k, v in weights.items():
            self.register_buffer(k, v)

    def forward(self, X, cos, sin, position_ids):
        return attn_tkg_qwen_v2(
            X, cos, sin, position_ids,
            self.Wq, self.Wk, self.Wv, self.Wo,
            self.q_norm_w, self.k_norm_w,
            self.K_cache, self.V_cache,
            replica_groups=([0, 1],),
        )


class _WeightsFactory:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self):
        m = AttnModule(self.weights)
        m.eval()
        return m, {}


# ---------------------------------------------------------------------------
# Weights and inputs
# ---------------------------------------------------------------------------

def make_weights_and_inputs():
    torch.manual_seed(42)
    scale = 0.02
    weights = dict(
        Wq       = torch.randn(Q_HEADS * D_HEAD, H, dtype=torch.bfloat16) * scale,
        Wk       = torch.randn(D_HEAD, H, dtype=torch.bfloat16) * scale,
        Wv       = torch.randn(D_HEAD, H, dtype=torch.bfloat16) * scale,
        Wo       = torch.randn(Q_HEADS * D_HEAD, H, dtype=torch.bfloat16) * scale,
        q_norm_w = torch.ones(D_HEAD, dtype=torch.bfloat16),
        k_norm_w = torch.ones(D_HEAD, dtype=torch.bfloat16),
        K_cache  = torch.zeros(B, 1, S_CTX, D_HEAD, dtype=torch.bfloat16),
        V_cache  = torch.zeros(B, 1, S_CTX, D_HEAD, dtype=torch.bfloat16),
    )
    inputs = dict(
        X            = torch.randn(B, S_TKG, H, dtype=torch.bfloat16) * 0.1,
        cos          = torch.ones(B, D_HEAD, dtype=torch.bfloat16),
        sin          = torch.zeros(B, D_HEAD, dtype=torch.bfloat16),
        position_ids = torch.zeros(B, 1, dtype=torch.int32),
    )
    return weights, inputs


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def bench(fn, *args, warmup=5, iters=20):
    import torch_xla.core.xla_model as xm
    for _ in range(warmup):
        fn(*args)
    xm.mark_step()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    xm.mark_step()
    elapsed = (time.perf_counter() - t0) / iters * 1e3
    print(f"attn_tkg_qwen_v2: {elapsed:.3f} ms/iter")
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Correctness ===")

    weights, inputs = make_weights_and_inputs()

    # PyTorch reference
    print("\nComputing PyTorch reference...")
    ref_out = pytorch_reference(
        inputs["X"], weights["Wq"], weights["Wk"], weights["Wv"], weights["Wo"],
        weights["q_norm_w"], weights["k_norm_w"],
        weights["K_cache"], weights["V_cache"],
        inputs["cos"], inputs["sin"],
        d=D_HEAD,
    )
    ref_np = ref_out.float().numpy()
    print(f"  ref_out shape: {ref_out.shape}")
    print(f"  ref_out range: [{ref_np.min():.4f}, {ref_np.max():.4f}]")

    # Compile NKI kernel
    example_args = (
        inputs["X"], inputs["cos"], inputs["sin"], inputs["position_ids"]
    )
    factory = _WeightsFactory(weights)

    workdir = "/tmp/attn_tkg_qwen_v2_workdir"
    os.makedirs(workdir, exist_ok=True)
    print(f"\nCompiling attn_tkg_qwen_v2 (tp_degree=2), workdir={workdir}...")
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
    result = traced(*example_args)
    result_np = result.cpu().float().numpy()
    print(f"  kernel output shape: {result.shape}")
    print(f"  kernel output range: [{result_np.min():.4f}, {result_np.max():.4f}]")

    # Compare
    abs_diff = np.abs(result_np - ref_np)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    print(f"\n  max  |diff|: {max_diff:.6e}")
    print(f"  mean |diff|: {mean_diff:.6e}")

    try:
        import torch
        torch.testing.assert_close(
            torch.from_numpy(result_np),
            torch.from_numpy(ref_np),
            rtol=1e-2,
            atol=1e-2,
        )
        print("\nPASS: assert_close passed!")
    except AssertionError as e:
        print(f"\nFAIL: assert_close failed!\n{e}")
        sys.exit(1)

    print("\n=== Benchmark ===")
    elapsed = bench(traced, *example_args)
    print(f"Result: {elapsed:.3f} ms/iter")
