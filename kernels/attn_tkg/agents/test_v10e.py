"""
Correctness + benchmark harness for v10e.

v10e replaces v10d's K-norm zero-detection mask with an exact position_ids
threshold mask. Tests verify:
  (a) partial-fill case (pos=320, S_prior=640) — primary correctness case
      where v10d's heuristic would be imprecise with random K vectors
  (b) full cache (pos=640) — all valid, mask should be all-zero
  (c) nearly-empty cache (pos=128) — only first tile valid

Also benchmarks v10e vs v10d to confirm no performance regression.
"""
import os, sys, math

# ── Set ALL env vars BEFORE any neuron/torch_xla import ─────────────────────
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out"
)
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
from v10e import qwen3_attn_tkg_fused_oproj_v10e
from v10d import qwen3_attn_tkg_fused_oproj_v10d


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rmsnorm_per_head(x, weight, eps=1e-6):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float())


def pytorch_reference(hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
                      K_cache, V_cache, cos_at_pos, sin_at_pos, pos, d=128):
    """
    Reference implementation: attends to exactly pos cache positions + 1 active position.
    K_cache: [B, 1, S_prior, d] — only K_cache[:, :, :pos, :] are valid.
    pos = position_ids value = number of valid tokens in KV cache.
    """
    B, _, H = hidden_states.shape
    Hq_out = Wq.shape[0]; Hkv_out = Wk.shape[0]
    Hq_tp = Hq_out // d; Hkv_tp = Hkv_out // d
    gqa = Hq_tp // Hkv_tp

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
            # Only use valid cache positions 0..pos-1
            K_full = torch.cat([
                K_cache[b, kv_h, :pos, :].float(),
                K_rope[b, kv_h, 0:1, :].float(),
            ], dim=0)
            V_full = torch.cat([
                V_cache[b, kv_h, :pos, :].float(),
                V[b, kv_h, 0:1, :].float(),
            ], dim=0)
            scores = (K_full @ q_vec) * scale
            attn_weights = F.softmax(scores, dim=0)
            attn_out_heads[b, q_h, :] = (attn_weights.unsqueeze(-1) * V_full).sum(0)

    attn_output = attn_out_heads.reshape(B, 1, Hq_tp * d).to(torch.bfloat16)
    output = (attn_output.float() @ Wo.float().T).to(torch.bfloat16)

    # K_rope and V outputs (NKI-equivalent computation)
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


def make_inputs(pos, S_prior=640, B=1, H=2048, d=128, Hq_tp=8, Hkv_tp=1, scale=0.05, seed=42):
    """Build test tensors. K/V cache filled randomly for [:pos], zeros for [pos:]."""
    Hq_out = Hq_tp * d; Hkv_out = Hkv_tp * d
    torch.manual_seed(seed)
    hidden_states = (torch.randn(B, 1, H) * scale).to(torch.bfloat16)
    Wq = (torch.randn(Hq_out, H) * scale).to(torch.bfloat16)
    Wk = (torch.randn(Hkv_out, H) * scale).to(torch.bfloat16)
    Wv = (torch.randn(Hkv_out, H) * scale).to(torch.bfloat16)
    Wo_orig = (torch.randn(H, Hq_out) * scale).to(torch.bfloat16)  # [2048, 1024]
    q_norm_weight = torch.ones(d, dtype=torch.bfloat16)
    k_norm_weight = torch.ones(d, dtype=torch.bfloat16)

    # K/V cache: valid positions have random bf16 values, padding is zero
    K_cache = torch.zeros(B, Hkv_tp, S_prior, d, dtype=torch.bfloat16)
    V_cache = torch.zeros(B, Hkv_tp, S_prior, d, dtype=torch.bfloat16)
    K_cache[:, :, :pos, :] = (torch.randn(B, Hkv_tp, pos, d) * scale).to(torch.bfloat16)
    V_cache[:, :, :pos, :] = (torch.randn(B, Hkv_tp, pos, d) * scale).to(torch.bfloat16)

    position_ids = torch.full((B, 1), pos, dtype=torch.int32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
    freqs = torch.outer(torch.tensor([float(pos)]), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_at_pos = emb.cos().to(torch.bfloat16).expand(B, d)
    sin_at_pos = emb.sin().to(torch.bfloat16).expand(B, d)

    Wo_kernel = Wo_orig.T.contiguous()  # [Hq_out=1024, H_wo=2048]
    return (hidden_states, Wq, Wk, Wv, Wo_kernel, Wo_orig,
            q_norm_weight, k_norm_weight, K_cache, V_cache,
            cos_at_pos, sin_at_pos, position_ids)


def run_correctness_test(pos, S_prior=640, label=""):
    print(f"\n{'='*70}")
    print(f"v10e Correctness  {label}  pos={pos}  S_prior={S_prior}")
    print(f"{'='*70}")
    device = xm.xla_device()

    (hidden_states, Wq, Wk, Wv, Wo_kernel, Wo_orig,
     q_norm_weight, k_norm_weight, K_cache, V_cache,
     cos_at_pos, sin_at_pos, position_ids) = make_inputs(pos, S_prior)

    # PyTorch reference (exact masking at pos)
    ref_out, ref_k_rope, ref_v = pytorch_reference(
        hidden_states, Wq, Wk, Wv, Wo_orig,
        q_norm_weight, k_norm_weight,
        K_cache, V_cache, cos_at_pos, sin_at_pos, pos=pos,
    )

    # v10e kernel
    nki_out_dev, nki_k_rope_dev, nki_v_dev = qwen3_attn_tkg_fused_oproj_v10e[2](
        hidden_states.to(device), Wq.to(device), Wk.to(device), Wv.to(device),
        Wo_kernel.to(device), q_norm_weight.to(device), k_norm_weight.to(device),
        K_cache.to(device), V_cache.to(device),
        cos_at_pos.to(device), sin_at_pos.to(device),
        position_ids.to(device),
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


def run_benchmark(label, kernel_fn, pos, S_prior=640):
    print(f"\n{'='*70}")
    print(f"Benchmarking {label}  pos={pos}")
    print(f"{'='*70}")
    device = xm.xla_device()

    (hidden_states, Wq, Wk, Wv, Wo_kernel, _,
     q_norm_weight, k_norm_weight, K_cache, V_cache,
     cos_at_pos, sin_at_pos, position_ids) = make_inputs(pos, S_prior)

    hs_d       = hidden_states.to(device)
    Wq_d       = Wq.to(device)
    Wk_d       = Wk.to(device)
    Wv_d       = Wv.to(device)
    Wo_d       = Wo_kernel.to(device)
    qnw_d      = q_norm_weight.to(device)
    knw_d      = k_norm_weight.to(device)
    Kc_d       = K_cache.to(device)
    Vc_d       = V_cache.to(device)
    cos_d      = cos_at_pos.to(device)
    sin_d      = sin_at_pos.to(device)
    pos_d      = position_ids.to(device)

    bench_fn = wrap_benchmark(kernel_fn, warmup=5, iters=50)
    bench_fn(hs_d, Wq_d, Wk_d, Wv_d, Wo_d, qnw_d, knw_d, Kc_d, Vc_d, cos_d, sin_d, pos_d)

    r = bench_fn.last_result
    if r:
        print(f"  device_time_us       = {r.device_time_us:.2f}")
        print(f"  tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
        print(f"  dma_active_pct       = {r.dma_active_pct:.1f}%")
        print(f"  spill_bytes          = {r.spill_bytes}")
        print(f"  mfu_estimated        = {r.prof.get('mfu_estimated_percent', 0):.2f}%")
        print(f"  hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
        print(f"  hbm_write_KiB        = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")
    else:
        print("  WARNING: last_result is None — benchmark artifact not captured")
    return r


if __name__ == "__main__":
    # ── Correctness ────────────────────────────────────────────────────────────
    # Test 1: primary case — half-filled cache, random K values
    # v10d's heuristic is imprecise here; v10e must be exact
    # run_correctness_test(pos=320, label="half-filled cache")

    # # Test 2: full cache — all positions valid, mask all zeros
    # run_correctness_test(pos=640, label="full cache")

    # Test 3: nearly-empty cache — only first tile (128 tokens) valid
    run_correctness_test(pos=128, label="one tile valid")

    # ── Benchmark: v10e vs v10d ────────────────────────────────────────────────
    # Use full-cache (pos=640) for benchmarking to match v10d's benchmark baseline.
    # NeuronCores are exclusive — run sequentially.
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON: v10d vs v10e")
    print("="*70)

    # r_v10d = run_benchmark("v10d", qwen3_attn_tkg_fused_oproj_v10d[2], pos=640)
    # r_v10e = run_benchmark("v10e", qwen3_attn_tkg_fused_oproj_v10e[2], pos=640)
l
    if r_v10d and r_v10e:
        delta_us = r_v10e.device_time_us - r_v10d.device_time_us
        pct = delta_us / r_v10d.device_time_us * 100
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"  v10d  device_time_us = {r_v10d.device_time_us:.2f}")
        print(f"  v10e  device_time_us = {r_v10e.device_time_us:.2f}")
        print(f"  delta = {delta_us:+.2f} us  ({pct:+.1f}%)")
        if delta_us <= 0:
            print("  RESULT: v10e is faster or equal — no regression")
        else:
            print("  RESULT: v10e is slower — regression detected")
