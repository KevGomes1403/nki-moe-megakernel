"""
Test harness for qwen3_attn_tkg_fused NKI kernel.

Implements PyTorch reference and compares against NKI kernel output.
"""

import sys
import math
import torch
import torch.nn.functional as F

# Activate device
import torch_xla.core.xla_model as xm

import os
os.environ["NEURON_CC_FLAGS"] = " "
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"


def rotate_half(x):
    """Rotates half the hidden dims: [-x2, x1] where x = [x1, x2]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rmsnorm_per_head(x, weight, eps=1e-6):
    """Apply RMSNorm per head. x: [..., d], weight: [d]"""
    # x shape: [..., d]
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(variance + eps)
    return (x_normed * weight.float()).to(x.dtype)


def pytorch_reference(
    hidden_states,   # [B, 1, H] bf16
    Wq,              # [Hq_tp*d, H] bf16
    Wk,              # [Hkv_tp*d, H] bf16
    Wv,              # [Hkv_tp*d, H] bf16
    q_norm_weight,   # [d] bf16
    k_norm_weight,   # [d] bf16
    K_cache,         # [B, Hkv_tp, S_prior, d] bf16
    V_cache,         # [B, Hkv_tp, S_prior, d] bf16
    cos_at_pos,      # [B, d] bf16  (pre-indexed by position_ids)
    sin_at_pos,      # [B, d] bf16
    d=128,
):
    """Pure PyTorch reference implementation."""
    B, _, H = hidden_states.shape
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
    K = K.reshape(B, Hkv_tp, 1, d)  # [B, 4, 1, 128]
    V = V.reshape(B, Hkv_tp, 1, d)  # [B, 4, 1, 128]

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
    # For each Q head, use KV head = q_head // gqa
    # Concatenate active K,V with cache
    # K_cache: [B, Hkv_tp, S_prior, d]
    # Active K: K_rope [B, Hkv_tp, 1, d]

    output = torch.zeros(B, Hq_tp, d, dtype=torch.float32)

    for b in range(B):
        for q_h in range(Hq_tp):
            kv_h = q_h // gqa
            q_vec = Q_rope[b, q_h, 0, :]  # [d]

            # Concatenate cached + active K,V
            K_full = torch.cat([
                K_cache[b, kv_h].float(),  # [S_prior, d]
                K_rope[b, kv_h, 0:1, :].float(),  # [1, d]
            ], dim=0)  # [S_prior+1, d]

            V_full = torch.cat([
                V_cache[b, kv_h].float(),  # [S_prior, d]
                V[b, kv_h, 0:1, :].float(),  # [1, d] (no RoPE on V)
            ], dim=0)  # [S_prior+1, d]

            # Attention scores [S_prior+1]
            scores = (K_full @ q_vec) * scale  # [S_prior+1]

            # Softmax
            attn_weights = F.softmax(scores, dim=0)  # [S_prior+1]

            # Weighted sum of V [d]
            out_vec = (attn_weights.unsqueeze(-1) * V_full).sum(dim=0)  # [d]
            output[b, q_h, :] = out_vec

    # Reshape output [B, Hq_tp, d] -> [B, 1, Hq_tp*d]
    output = output.reshape(B, 1, Hq_tp * d)
    return output.to(torch.bfloat16)


def run_test(B=1, S_prior=128, H=2048, d=128, Hq_tp=8, Hkv_tp=4):
    print(f"\n{'='*70}")
    print(f"qwen3_attn_tkg_fused Test")
    print(f"B={B}, S_prior={S_prior}, H={H}, d={d}, Hq_tp={Hq_tp}, Hkv_tp={Hkv_tp}")
    print(f"{'='*70}")

    device = xm.xla_device()
    dtype = torch.bfloat16

    Hq_out = Hq_tp * d    # 1024
    Hkv_out = Hkv_tp * d  # 512

    torch.manual_seed(42)

    # Create test inputs (small values for numerical stability)
    scale = 0.05
    hidden_states = (torch.randn(B, 1, H) * scale).to(dtype)
    Wq = (torch.randn(Hq_out, H) * scale).to(dtype)
    Wk = (torch.randn(Hkv_out, H) * scale).to(dtype)
    Wv = (torch.randn(Hkv_out, H) * scale).to(dtype)
    q_norm_weight = torch.ones(d, dtype=dtype)  # identity norm weight
    k_norm_weight = torch.ones(d, dtype=dtype)
    K_cache = (torch.randn(B, Hkv_tp, S_prior, d) * scale).to(dtype)
    V_cache = (torch.randn(B, Hkv_tp, S_prior, d) * scale).to(dtype)

    # RoPE embeddings — create a simple set
    # position_ids = [B, 1] with some position value
    pos = S_prior  # current position is after all cached positions
    position_ids = torch.full((B, 1), pos, dtype=torch.int32)

    # Pre-compute cos/sin at the position for the test
    # Simple cos/sin for testing: use fixed frequencies
    # In practice these come from RotaryEmbedding
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
    t = torch.tensor([float(pos)])
    freqs = torch.outer(t, inv_freq)  # [1, d/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [1, d]
    cos_at_pos = emb.cos().to(dtype).expand(B, d)  # [B, d]
    sin_at_pos = emb.sin().to(dtype).expand(B, d)  # [B, d]

    print("\n[1/4] Computing PyTorch reference...")
    ref_out = pytorch_reference(
        hidden_states, Wq, Wk, Wv,
        q_norm_weight, k_norm_weight,
        K_cache, V_cache,
        cos_at_pos, sin_at_pos,
        d=d,
    )
    print(f"  ref_out shape: {ref_out.shape}")
    print(f"  ref_out stats: min={ref_out.float().min():.4f}, max={ref_out.float().max():.4f}")

    print("\n[2/4] Preparing NKI kernel inputs...")
    # cos/sin pre-indexed as [B, d]
    cos_nki = cos_at_pos.reshape(B, d)
    sin_nki = sin_at_pos.reshape(B, d)

    # Move to device
    hidden_dev = hidden_states.to(device)
    Wq_dev = Wq.to(device)
    Wk_dev = Wk.to(device)
    Wv_dev = Wv.to(device)
    q_norm_dev = q_norm_weight.to(device)
    k_norm_dev = k_norm_weight.to(device)
    K_cache_dev = K_cache.to(device)
    V_cache_dev = V_cache.to(device)
    cos_dev = cos_nki.to(device)
    sin_dev = sin_nki.to(device)
    pos_dev = position_ids.to(device)

    print("\n[3/4] Running NKI kernel...")
    from attn_tkg_fused import qwen3_attn_tkg_fused
    nki_out_dev = qwen3_attn_tkg_fused(
        hidden_dev, Wq_dev, Wk_dev, Wv_dev,
        q_norm_dev, k_norm_dev,
        K_cache_dev, V_cache_dev,
        cos_dev, sin_dev, pos_dev,
    )
    xm.mark_step()
    nki_out = nki_out_dev.cpu()
    print(f"  nki_out shape: {nki_out.shape}")
    print(f"  nki_out stats: min={nki_out.float().min():.4f}, max={nki_out.float().max():.4f}")

    print("\n[4/4] Comparing results...")
    ref_f = ref_out.float()
    nki_f = nki_out.float()
    diff = (ref_f - nki_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max  |diff| : {max_diff:.6e}")
    print(f"  Mean |diff| : {mean_diff:.6e}")

    threshold = 0.1
    if max_diff < threshold:
        print(f"\n{'='*70}")
        print(f"PASS: max_diff={max_diff:.4e} < threshold={threshold}")
        print(f"{'='*70}")
        return True
    else:
        print(f"\n{'='*70}")
        print(f"FAIL: max_diff={max_diff:.4e} >= threshold={threshold}")
        print(f"{'='*70}")
        # Print some sample values for debugging
        print(f"\nRef sample:  {ref_f[0, 0, :8].tolist()}")
        print(f"NKI sample:  {nki_f[0, 0, :8].tolist()}")
        return False


if __name__ == "__main__":
    import os
    import sys

    # Add kernel directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Start with a small test case
    # S_prior=128 (one full tile), simple shapes
    ok = run_test(B=1, S_prior=128, H=2048, d=128, Hq_tp=8, Hkv_tp=4)

    if ok:
        # Test with larger S_prior
        print("\n\nRunning second test with S_prior=640...")
        ok2 = run_test(B=1, S_prior=640, H=2048, d=128, Hq_tp=8, Hkv_tp=4)
        if ok2:
            print("\nAll tests PASSED.")
        else:
            print("\nSecond test FAILED.")
            sys.exit(1)
    else:
        sys.exit(1)
