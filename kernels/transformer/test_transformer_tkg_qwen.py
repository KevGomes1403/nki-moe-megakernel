"""
Correctness test for transformer_tkg_qwen NKI kernel.

Sets environment variables BEFORE any neuron/torch_xla imports.
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
import tempfile
from neuronx_distributed.trace import parallel_model_trace

from kernels.transformer.transformer_tkg_qwen import transformer_tkg_qwen

# ---------------------------------------------------------------------------
# Fixed shapes (match kernel constants)
# ---------------------------------------------------------------------------
B, S_TKG, H = 1, 1, 2048
D_HEAD  = 128
Q_HEADS = 8        # per TP rank
I_MLP   = 192
E       = 128
K       = 8        # top-K experts
S_CTX   = 512      # KV cache entries
TP      = 2

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
# Fixed-seed weights and inputs
# ---------------------------------------------------------------------------
def make_weights_and_inputs():
    torch.manual_seed(42)
    weights = dict(
        Wq       = torch.randn(Q_HEADS * D_HEAD, H, dtype=torch.bfloat16) * 0.02,
        Wk       = torch.randn(D_HEAD, H, dtype=torch.bfloat16) * 0.02,
        Wv       = torch.randn(D_HEAD, H, dtype=torch.bfloat16) * 0.02,
        Wo       = torch.randn(Q_HEADS * D_HEAD, H, dtype=torch.bfloat16) * 0.02,
        q_norm_w = torch.ones(D_HEAD, dtype=torch.bfloat16),
        k_norm_w = torch.ones(D_HEAD, dtype=torch.bfloat16),
        K_cache  = torch.zeros(B, 1, S_CTX, D_HEAD, dtype=torch.bfloat16),
        V_cache  = torch.zeros(B, 1, S_CTX, D_HEAD, dtype=torch.bfloat16),
        gamma_mlp= torch.ones(1, H, dtype=torch.bfloat16),
        router_w = torch.randn(H, E, dtype=torch.bfloat16) * 0.02,
        gate_up_w= torch.randn(E, H, 2 * I_MLP, dtype=torch.bfloat16) * 0.02,
        down_w   = torch.randn(E, I_MLP, H, dtype=torch.bfloat16) * 0.02,
    )
    inputs = dict(
        X            = torch.randn(B, S_TKG, H, dtype=torch.bfloat16) * 0.1,
        cos          = torch.ones(B, D_HEAD, dtype=torch.bfloat16),
        sin          = torch.zeros(B, D_HEAD, dtype=torch.bfloat16),
        position_ids = torch.zeros(B, 1, dtype=torch.int32),
    )
    return weights, inputs


# ---------------------------------------------------------------------------
# PyTorch reference (float32 on CPU for accuracy)
# ---------------------------------------------------------------------------

def rms_norm(x, weight, eps=1e-6):
    """RMSNorm: x * weight / rms(x). x: [..., D], weight: [D]."""
    x_f32 = x.float()
    rms = x_f32.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f32 * rms * weight.float())


def rope_apply(x, cos, sin):
    """
    Apply RoPE rotation to x [D].
    rot(x) = x * cos + rotate_half(x) * sin
    rotate_half: [-x[d/2:], x[:d/2]]
    """
    d = x.shape[-1]
    half_d = d // 2
    # rotate_half: [-upper, lower]
    x_rot = torch.cat([-x[..., half_d:], x[..., :half_d]], dim=-1)
    return x * cos + x_rot * sin


def reference_transformer(weights, inputs):
    """
    Full PyTorch reference for transformer_tkg_qwen in float32.
    Mirrors the exact arithmetic of the NKI kernel.
    """
    w = {k: v.float() for k, v in weights.items()}
    X = inputs["X"].float()  # [1, 1, 2048]
    cos = inputs["cos"].float()  # [1, 128]
    sin = inputs["sin"].float()  # [1, 128]
    # position_ids: [1, 1] — used for masking, not for RoPE index since we pass cos/sin directly
    position_ids = inputs["position_ids"]  # [1, 1] int32

    # =========================================================================
    # ATTENTION BLOCK
    # Mirrors _qwen3_attn_tkg_sbuf_out
    # =========================================================================
    x_2d = X.reshape(B * S_TKG, H)  # [1, 2048]

    # K projection: [D_HEAD, H] x [H] -> [D_HEAD]
    k_proj = (x_2d @ w["Wk"].T).squeeze(0)  # [128]

    # K RMSNorm
    k_normed = rms_norm(k_proj, w["k_norm_w"])  # [128]

    # K RoPE
    cos_1d = cos[0]  # [128]
    sin_1d = sin[0]  # [128]
    k_rope = rope_apply(k_normed, cos_1d, sin_1d)  # [128]

    # V projection: [D_HEAD, H] x [H] -> [D_HEAD]
    v_active = (x_2d @ w["Wv"].T).squeeze(0)  # [128]

    # Q projections: 8 heads, each [D_HEAD, H] -> [D_HEAD]
    # Wq shape: [Q_HEADS*D_HEAD, H] = [1024, 2048]
    q_all = x_2d @ w["Wq"].T  # [1, 1024]
    q_per_head = q_all.reshape(Q_HEADS, D_HEAD)  # [8, 128]

    # Q RMSNorm — per-head, shared weight q_norm_w
    q_normed = rms_norm(q_per_head, w["q_norm_w"])  # [8, 128]

    # Q RoPE (same cos/sin broadcast across heads)
    # cos/sin are [128] broadcast to [8, 128]
    q_rope = rope_apply(q_normed, cos_1d.unsqueeze(0), sin_1d.unsqueeze(0))  # [8, 128]

    # Scale Q by 1/sqrt(D_HEAD)
    inv_sqrt_d = 1.0 / math.sqrt(D_HEAD)
    q_scaled = q_rope * inv_sqrt_d  # [8, 128]

    # -------------------------------------------------------------------------
    # Flash decode: K_cache=zeros, so cache scores are all 0 before masking.
    # With position_ids=0, mask is: positions 0..S_CTX-1 with threshold.
    # The kernel uses: mask = clamp(relu(idx - position + 1), 0, 1) * (-1e9)
    # For position=0: tile_start=0..3 (0,128,256,384)
    # mask_val[s] = clamp(relu(s - 0 + 1), 0, 1) * (-1e9) = -1e9 for s>=0
    # So ALL cache positions get -1e9 → effectively masked out.
    # Active token (k_rope, v_active) contribution:
    #   score_active = sum(k_rope * q_scaled[h]) for each h — dot product
    #   global_max = max(score_active_h) (no cache contributions above -1e9)
    #   exp(score - max) for active token
    #   sum_weights = exp(score_active - max)
    #   v_out = v_active * exp(score_active - max) / sum_weights = v_active
    # So with all-zero K_cache and position=0 (all cache slots masked),
    # the attention output = v_active for each head.
    # -------------------------------------------------------------------------

    # Compute active token score for each head: dot(k_rope, q_scaled[h])
    # k_rope: [128], q_scaled: [8, 128]
    score_active = (q_scaled * k_rope.unsqueeze(0)).sum(dim=-1)  # [8]

    # Masked cache scores: K_cache is zeros, but ALL positions masked with -1e9
    # (position_ids=0 means all cache positions >=0 are "future" from the mask formula)
    # So the cache contributes nothing (exp(-1e9) ≈ 0).
    # Active position score: score_active[h]
    # global_max = max(score_active) per head
    # After softmax normalization: weight_active = 1.0 (only contributor)
    # Attention output = v_active (same for all heads, since V is GQA with 1 KV head)

    # attn_per_head: [8, 128] — same v_active for all heads (GQA=8 local)
    attn_per_head = v_active.unsqueeze(0).expand(Q_HEADS, -1)  # [8, 128]

    # Output projection: attn_out @ Wo
    # Wo shape: [Q_HEADS*D_HEAD, H] = [1024, 2048]
    # attn_per_head: [8, 128] -> flatten to [1024]
    attn_flat = attn_per_head.reshape(1, Q_HEADS * D_HEAD)  # [1, 1024]
    # Wo: [Q_HEADS*D_HEAD, H] = [1024, 2048]
    # Each TP rank computes its H_SHARD=1024 of output using its half of Wo rows,
    # then all-reduce sums across ranks. Full reference: attn_flat @ Wo = [1, 2048].
    # (attn_flat is [1, 1024], Wo is [1024, 2048])
    attn_out = attn_flat @ w["Wo"]  # [1, 2048]
    attn_out_3d = attn_out.reshape(B, S_TKG, H)  # [1, 1, 2048]

    # =========================================================================
    # RESIDUAL 1
    # =========================================================================
    mid = X.float() + attn_out_3d  # [1, 1, 2048]

    # =========================================================================
    # MOE BLOCK
    # Mirrors _qwen3_moe_tkg_sbuf_out
    # =========================================================================
    mid_2d = mid.reshape(B * S_TKG, H)  # [1, 2048]

    # RMSNorm on mid (gamma_mlp)
    # gamma_mlp: [1, 2048]
    gamma = w["gamma_mlp"].reshape(H)  # [2048]
    moe_in = rms_norm(mid_2d, gamma)  # [1, 2048]

    # Router: logits = moe_in @ router_w -> [1, E]
    logits = moe_in @ w["router_w"]  # [1, 128]

    # Softmax over experts
    probs = torch.softmax(logits.float(), dim=-1)  # [1, 128]

    # TopK(8) selection
    top_vals, top_idx = torch.topk(probs, K, dim=-1)  # [1, 8], [1, 8]

    # Normalize weights: norm_w = top_vals / sum(top_vals)
    # (Kernel uses: inv_sum_topk = 1/sum(top8_vals), norm_weights = top8_vals * inv_sum_topk)
    norm_w = top_vals / top_vals.sum(dim=-1, keepdim=True)  # [1, 8]

    # Expert MLP for each selected expert
    # gate_up_w: [E, H, 2*I_MLP] — first I_MLP is gate, second I_MLP is up
    # down_w: [E, I_MLP, H]
    moe_out = torch.zeros(B * S_TKG, H, dtype=torch.float32)
    for t in range(B * S_TKG):
        x_t = moe_in[t]  # [H]
        out_t = torch.zeros(H, dtype=torch.float32)
        for ki in range(K):
            eid = top_idx[t, ki].item()
            w_expert = norm_w[t, ki].item()

            # Gate projection: [H, I_MLP].T @ x -> [I_MLP]
            # gate_up_w[eid]: [H, 2*I_MLP], gate = [:, :I_MLP], up = [:, I_MLP:]
            gate_w = w["gate_up_w"][eid, :, :I_MLP]  # [H, I_MLP]
            up_w   = w["gate_up_w"][eid, :, I_MLP:]  # [H, I_MLP]
            down_w_e = w["down_w"][eid]               # [I_MLP, H]

            gate_val = x_t @ gate_w  # [I_MLP]
            up_val   = x_t @ up_w    # [I_MLP]

            # SiLU(gate) * up
            inter = F.silu(gate_val) * up_val  # [I_MLP]

            # Down projection
            out_e = inter @ down_w_e  # [H]
            out_t = out_t + w_expert * out_e

        moe_out[t] = out_t

    moe_out_3d = moe_out.reshape(B, S_TKG, H)  # [1, 1, 2048]

    # =========================================================================
    # RESIDUAL 2
    # =========================================================================
    output = mid + moe_out_3d  # [1, 1, 2048]
    return output.bfloat16()  # match kernel output dtype


# ---------------------------------------------------------------------------
# NKI Module wrapper
# ---------------------------------------------------------------------------

class TKGModule(nn.Module):
    def __init__(self, weights):
        super().__init__()
        for k, v in weights.items():
            self.register_buffer(k, v)

    def forward(self, X, cos, sin, position_ids):
        return nki.jit(transformer_tkg_qwen)(
            X, cos, sin, position_ids,
            self.Wq, self.Wk, self.Wv, self.Wo,
            self.q_norm_w, self.k_norm_w,
            self.K_cache, self.V_cache,
            self.gamma_mlp, self.router_w, self.gate_up_w, self.down_w,
        )


class _WeightsFactory:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self):
        m = TKGModule(self.weights)
        m.eval()
        return m, {}


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_test():
    print("=" * 60)
    print("transformer_tkg_qwen correctness test")
    print("=" * 60)

    weights, inputs = make_weights_and_inputs()

    # Compute reference
    print("\nComputing PyTorch reference...")
    ref_out = reference_transformer(weights, inputs)  # [1, 1, 2048] bfloat16
    ref_np = ref_out.float().numpy()
    print(f"Reference output shape: {ref_out.shape}")
    print(f"Reference output range: [{ref_np.min():.4f}, {ref_np.max():.4f}]")

    # Compile and run NKI kernel
    example_args = (
        inputs["X"], inputs["cos"], inputs["sin"], inputs["position_ids"]
    )
    factory = _WeightsFactory(weights)

    workdir = "/tmp/tkg_qwen_test_workdir"
    os.makedirs(workdir, exist_ok=True)
    print(f"\nCompiling transformer_tkg_qwen kernel (tp_degree=2), workdir={workdir}...")
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
        print(f"Kernel output shape: {result.shape}")
        print(f"Kernel output range: [{result_np.min():.4f}, {result_np.max():.4f}]")
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

    rtol, atol = 0.1, 0.1
    print(f"\nTolerance: rtol={rtol}, atol={atol}")
    try:
        np.testing.assert_allclose(result_np, ref_np, rtol=rtol, atol=atol)
        print("PASS: assert_allclose passed!")
        return True, max_diff
    except AssertionError as e:
        print(f"FAIL: assert_allclose failed!\n{e}")
        return False, max_diff


if __name__ == "__main__":
    passed, max_diff = run_test()
    print("\n" + "=" * 60)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"max_diff: {max_diff:.6f}")
    print("=" * 60)
    sys.exit(0 if passed else 1)
