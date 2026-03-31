"""
Correctness + benchmark harness for v10c.
Tests: (a) partial-fill case (valid_seq_len < S_prior), (b) full cache (valid_seq_len == S_prior).
"""
import os, sys, math
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench_out")
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import numpy as np

# Copy benchmark.py from skill root
import shutil
skill_bench = "/home/ubuntu/nki-moe/.claude/skills/nki-kernel-optimizer/scripts/benchmark.py"
shutil.copy(skill_bench, os.path.dirname(os.path.abspath(__file__)))
from benchmark import wrap_benchmark

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v10c import qwen3_attn_tkg_fused_oproj_v10c

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rmsnorm_per_head(x, weight, eps=1e-6):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float())

def pytorch_reference(hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
                       K_cache, V_cache, cos_at_pos, sin_at_pos, valid_seq_len, d=128):
    """
    Reference: attends to exactly valid_seq_len cache positions + 1 active position.
    K_cache: [B, 1, S_prior, d] — only K_cache[:, :, :valid_seq_len, :] are valid.
    """
    B, _, H = hidden_states.shape
    Hq_out = Wq.shape[0]; Hkv_out = Wk.shape[0]
    Hq_tp = Hq_out // d; Hkv_tp = Hkv_out // d
    gqa = Hq_tp // Hkv_tp
    H_out = Wo.shape[0]

    hs = hidden_states.float()
    Q = (hs @ Wq.float().T).reshape(B, Hq_tp, 1, d)
    K = (hs @ Wk.float().T).reshape(B, Hkv_tp, 1, d)
    V = (hs @ Wv.float().T).reshape(B, Hkv_tp, 1, d)

    Q_norm = torch.stack([apply_rmsnorm_per_head(Q[:, h], q_norm_weight) for h in range(Hq_tp)], dim=1)
    K_norm = torch.stack([apply_rmsnorm_per_head(K[:, h], k_norm_weight) for h in range(Hkv_tp)], dim=1)

    cos = cos_at_pos.float().unsqueeze(1).unsqueeze(2)
    sin = sin_at_pos.float().unsqueeze(1).unsqueeze(2)
    Q_rope = Q_norm * cos + rotate_half(Q_norm) * sin
    K_rope = K_norm * cos + rotate_half(K_norm) * sin

    scale = 1.0 / math.sqrt(d)
    attn_out_heads = torch.zeros(B, Hq_tp, d, dtype=torch.float32)

    for b in range(B):
        for q_h in range(Hq_tp):
            kv_h = q_h // gqa
            q_vec = Q_rope[b, q_h, 0, :]
            # Only use VALID cache positions
            K_full = torch.cat([
                K_cache[b, kv_h, :valid_seq_len, :].float(),
                K_rope[b, kv_h, 0:1, :].float(),
            ], dim=0)
            V_full = torch.cat([
                V_cache[b, kv_h, :valid_seq_len, :].float(),
                V[b, kv_h, 0:1, :].float(),
            ], dim=0)
            scores = (K_full @ q_vec) * scale
            attn_weights = F.softmax(scores, dim=0)
            attn_out_heads[b, q_h, :] = (attn_weights.unsqueeze(-1) * V_full).sum(0)

    attn_output = attn_out_heads.reshape(B, 1, Hq_tp * d).to(torch.bfloat16)
    output = (attn_output.float() @ Wo.float().T).to(torch.bfloat16)

    # K_rope_new and V_new (same NKI-equivalent computation as v10b reference)
    num_h_tiles = H // d
    hs_1d = hidden_states[0, 0, :].float()
    hidden_r = hs_1d.reshape(1, num_h_tiles * d)
    Wk_perm = Wk.float().reshape(d, num_h_tiles, d).permute(1, 0, 2).reshape(num_h_tiles * d, d)
    k_nki_vec = (hidden_r @ Wk_perm).squeeze(0)
    var_k = k_nki_vec.pow(2).mean() + 1e-6
    k_nki_normed = k_nki_vec * torch.rsqrt(var_k) * k_norm_weight.float()
    half_d = d // 2
    cos_1d = cos_at_pos[0].float(); sin_1d = sin_at_pos[0].float()
    rot_k_nki = torch.cat([-k_nki_normed[half_d:], k_nki_normed[:half_d]])
    k_rope_nki_vec = k_nki_normed * cos_1d + rot_k_nki * sin_1d
    Wv_perm = Wv.float().reshape(d, num_h_tiles, d).permute(1, 0, 2).reshape(num_h_tiles * d, d)
    v_nki_vec = (hidden_r @ Wv_perm).squeeze(0)

    K_rope_new = k_rope_nki_vec.to(torch.bfloat16).reshape(1, d)
    V_new = v_nki_vec.to(torch.bfloat16).reshape(1, d)
    return output, K_rope_new, V_new


def run_test(valid_seq_len, S_prior=640, B=1, H=2048, d=128, Hq_tp=8, Hkv_tp=1):
    print(f"\n{'='*70}")
    print(f"v10c Test  valid_seq_len={valid_seq_len}  S_prior={S_prior}")
    print(f"{'='*70}")
    device = xm.xla_device()
    dtype = torch.bfloat16
    Hq_out = Hq_tp * d; Hkv_out = Hkv_tp * d; H_wo = H
    scale = 0.05

    torch.manual_seed(42)
    hidden_states = (torch.randn(B, 1, H) * scale).to(dtype)
    Wq = (torch.randn(Hq_out, H) * scale).to(dtype)
    Wk = (torch.randn(Hkv_out, H) * scale).to(dtype)
    Wv = (torch.randn(Hkv_out, H) * scale).to(dtype)
    Wo_orig = (torch.randn(H_wo, Hq_out) * scale).to(dtype)  # [2048, 1024]
    q_norm_weight = torch.ones(d, dtype=dtype)
    k_norm_weight = torch.ones(d, dtype=dtype)

    # K_cache / V_cache: valid positions random, padded positions zero
    K_cache = torch.zeros(B, Hkv_tp, S_prior, d, dtype=dtype)
    V_cache = torch.zeros(B, Hkv_tp, S_prior, d, dtype=dtype)
    K_cache[:, :, :valid_seq_len, :] = (torch.randn(B, Hkv_tp, valid_seq_len, d) * scale).to(dtype)
    V_cache[:, :, :valid_seq_len, :] = (torch.randn(B, Hkv_tp, valid_seq_len, d) * scale).to(dtype)

    pos = S_prior
    position_ids = torch.full((B, 1), pos, dtype=torch.int32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
    freqs = torch.outer(torch.tensor([float(pos)]), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_at_pos = emb.cos().to(dtype).expand(B, d)
    sin_at_pos = emb.sin().to(dtype).expand(B, d)

    # Build additive attention mask: 0 for valid, -1e9 for padded
    attn_mask = torch.zeros(S_prior, 1, dtype=dtype)
    attn_mask[valid_seq_len:, :] = -1e9

    # PyTorch reference (slices to valid_seq_len)
    ref_out, ref_k_rope, ref_v = pytorch_reference(
        hidden_states, Wq, Wk, Wv, Wo_orig,
        q_norm_weight, k_norm_weight,
        K_cache, V_cache, cos_at_pos, sin_at_pos,
        valid_seq_len=valid_seq_len, d=d,
    )

    # Kernel inputs (Wo transposed to [Hq_out, H_wo])
    Wo_kernel = Wo_orig.T.contiguous()

    nki_out_dev, nki_k_rope_dev, nki_v_dev = qwen3_attn_tkg_fused_oproj_v10c[2](
        hidden_states.to(device), Wq.to(device), Wk.to(device), Wv.to(device),
        Wo_kernel.to(device), q_norm_weight.to(device), k_norm_weight.to(device),
        K_cache.to(device), V_cache.to(device),
        cos_at_pos.to(device), sin_at_pos.to(device),
        position_ids.to(device),
        attn_mask.to(device),
    )
    xm.mark_step()
    nki_out = nki_out_dev.cpu()
    nki_k_rope = nki_k_rope_dev.cpu()
    nki_v = nki_v_dev.cpu()

    diff_out = (ref_out.float() - nki_out.float()).abs()
    diff_k   = (ref_k_rope.float() - nki_k_rope.float()).abs()
    diff_v   = (ref_v.float() - nki_v.float()).abs()
    print(f"  Max |diff| output : {diff_out.max():.6e}")
    print(f"  Max |diff| k_rope : {diff_k.max():.6e}")
    print(f"  Max |diff| v      : {diff_v.max():.6e}")

    np.testing.assert_allclose(nki_out.float().numpy(), ref_out.float().numpy(), rtol=3e-2, atol=3e-2)
    np.testing.assert_allclose(nki_k_rope.float().numpy(), ref_k_rope.float().numpy(), rtol=3e-2, atol=3e-2)
    np.testing.assert_allclose(nki_v.float().numpy(), ref_v.float().numpy(), rtol=3e-2, atol=3e-2)
    max_diff = max(diff_out.max().item(), diff_k.max().item(), diff_v.max().item())
    print(f"PASS  max_diff={max_diff:.4e}")
    return max_diff


if __name__ == "__main__":
    # Test 1: partial fill (valid_seq_len = 100, S_prior = 640) — primary bug case
    run_test(valid_seq_len=100)
    # Test 2: full cache (valid_seq_len = S_prior) — no-op mask, should match v10b behavior
    run_test(valid_seq_len=640)

    # Benchmark
    print("\n" + "="*70)
    print("Benchmarking v10c...")
    print("="*70)

    device = xm.xla_device()
    dtype = torch.bfloat16
    B, H, d, S_prior, Hq_tp, Hkv_tp = 1, 2048, 128, 640, 8, 1
    Hq_out, Hkv_out = Hq_tp * d, Hkv_tp * d
    scale = 0.05

    torch.manual_seed(42)
    hidden_states = (torch.randn(B, 1, H) * scale).to(dtype).to(device)
    Wq = (torch.randn(Hq_out, H) * scale).to(dtype).to(device)
    Wk = (torch.randn(Hkv_out, H) * scale).to(dtype).to(device)
    Wv = (torch.randn(Hkv_out, H) * scale).to(dtype).to(device)
    Wo = ((torch.randn(H, Hq_out) * scale).to(dtype).T.contiguous()).to(device)  # [1024, 2048]
    q_norm = torch.ones(d, dtype=dtype).to(device)
    k_norm = torch.ones(d, dtype=dtype).to(device)
    K_cache = (torch.randn(B, Hkv_tp, S_prior, d) * scale).to(dtype).to(device)
    V_cache = (torch.randn(B, Hkv_tp, S_prior, d) * scale).to(dtype).to(device)
    pos = S_prior
    position_ids = torch.full((B, 1), pos, dtype=torch.int32).to(device)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
    freqs = torch.outer(torch.tensor([float(pos)]), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype).expand(B, d).to(device)
    sin_t = emb.sin().to(dtype).expand(B, d).to(device)
    attn_mask = torch.zeros(S_prior, 1, dtype=dtype).to(device)  # all valid for benchmark

    kernel_bench = wrap_benchmark(qwen3_attn_tkg_fused_oproj_v10c[2], warmup=5, iters=50)
    kernel_bench(hidden_states, Wq, Wk, Wv, Wo, q_norm, k_norm,
                 K_cache, V_cache, cos, sin_t, position_ids, attn_mask)

    r = kernel_bench.last_result
    if r:
        print(f"device_time_us       = {r.device_time_us:.2f}")
        print(f"tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
        print(f"dma_active_pct       = {r.dma_active_pct:.1f}%")
        print(f"spill_bytes          = {r.spill_bytes}")
        print(f"mfu_estimated        = {r.prof.get('mfu_estimated_percent', 0):.2f}%")
        print(f"hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
        print(f"hbm_write_KiB        = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")
    else:
        print("WARNING: last_result is None — benchmark artifact not captured")
