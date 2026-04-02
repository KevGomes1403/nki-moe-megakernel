"""
Correctness + benchmark harness for v11a (Single-Pass Online Flash Decode).

Compares v11a output vs v10e output (v10e as reference, not PyTorch reference,
to isolate the flash decode change).

Tests:
  (a) pos=320, S_prior=640 — primary correctness case (partial fill)
  (b) pos=640, S_prior=640 — full cache
  (c) pos=128, S_prior=640 — nearly empty (one tile valid)

Benchmarks v11a with pos=320, S_prior=640.
"""
import os, sys, math

# ── Set ALL env vars BEFORE any neuron/torch_xla import ─────────────────────
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v11a"
)
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import torch
import torch_xla.core.xla_model as xm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import wrap_benchmark
from v10e import qwen3_attn_tkg_fused_oproj_v10e
from v11a import qwen3_attn_tkg_fused_oproj_v11a


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
    return (hidden_states, Wq, Wk, Wv, Wo_kernel,
            q_norm_weight, k_norm_weight, K_cache, V_cache,
            cos_at_pos, sin_at_pos, position_ids)


def run_correctness_test(pos, S_prior=640, label=""):
    print(f"\n{'='*70}")
    print(f"v11a vs v10e Correctness  {label}  pos={pos}  S_prior={S_prior}")
    print(f"{'='*70}")
    device = xm.xla_device()

    (hidden_states, Wq, Wk, Wv, Wo_kernel,
     q_norm_weight, k_norm_weight, K_cache, V_cache,
     cos_at_pos, sin_at_pos, position_ids) = make_inputs(pos, S_prior)

    def to_dev(t): return t.to(device)

    args = (
        to_dev(hidden_states), to_dev(Wq), to_dev(Wk), to_dev(Wv), to_dev(Wo_kernel),
        to_dev(q_norm_weight), to_dev(k_norm_weight),
        to_dev(K_cache), to_dev(V_cache),
        to_dev(cos_at_pos), to_dev(sin_at_pos), to_dev(position_ids),
    )

    # v10e reference
    ref_out_dev, ref_k_dev, ref_v_dev = qwen3_attn_tkg_fused_oproj_v10e[2](*args)
    xm.mark_step()
    ref_out = ref_out_dev.cpu()
    ref_k   = ref_k_dev.cpu()
    ref_v   = ref_v_dev.cpu()

    # v11a under test
    nki_out_dev, nki_k_dev, nki_v_dev = qwen3_attn_tkg_fused_oproj_v11a[2](*args)
    xm.mark_step()
    nki_out = nki_out_dev.cpu()
    nki_k   = nki_k_dev.cpu()
    nki_v   = nki_v_dev.cpu()

    diff_out = (ref_out.float() - nki_out.float()).abs()
    diff_k   = (ref_k.float()   - nki_k.float()).abs()
    diff_v   = (ref_v.float()   - nki_v.float()).abs()
    print(f"  Max |diff| output : {diff_out.max():.6e}")
    print(f"  Max |diff| k_rope : {diff_k.max():.6e}")
    print(f"  Max |diff| v      : {diff_v.max():.6e}")

    # Tolerances: rtol=1e-2, atol=1e-2 for output; rtol=1e-3, atol=1e-3 for k_rope/v
    np.testing.assert_allclose(
        nki_out.float().numpy(), ref_out.float().numpy(), rtol=1e-2, atol=1e-2,
        err_msg=f"output mismatch at pos={pos}"
    )
    np.testing.assert_allclose(
        nki_k.float().numpy(), ref_k.float().numpy(), rtol=1e-3, atol=1e-3,
        err_msg=f"k_rope mismatch at pos={pos}"
    )
    np.testing.assert_allclose(
        nki_v.float().numpy(), ref_v.float().numpy(), rtol=1e-3, atol=1e-3,
        err_msg=f"v mismatch at pos={pos}"
    )
    max_diff = max(diff_out.max().item(), diff_k.max().item(), diff_v.max().item())
    print(f"PASS  max_diff={max_diff:.4e}")
    return max_diff


def run_benchmark(pos, S_prior=640):
    print(f"\n{'='*70}")
    print(f"Benchmarking v11a  pos={pos}  S_prior={S_prior}")
    print(f"{'='*70}")
    device = xm.xla_device()

    (hidden_states, Wq, Wk, Wv, Wo_kernel,
     q_norm_weight, k_norm_weight, K_cache, V_cache,
     cos_at_pos, sin_at_pos, position_ids) = make_inputs(pos, S_prior)

    hs_d   = hidden_states.to(device)
    Wq_d   = Wq.to(device)
    Wk_d   = Wk.to(device)
    Wv_d   = Wv.to(device)
    Wo_d   = Wo_kernel.to(device)
    qnw_d  = q_norm_weight.to(device)
    knw_d  = k_norm_weight.to(device)
    Kc_d   = K_cache.to(device)
    Vc_d   = V_cache.to(device)
    cos_d  = cos_at_pos.to(device)
    sin_d  = sin_at_pos.to(device)
    pos_d  = position_ids.to(device)

    bench_fn = wrap_benchmark(qwen3_attn_tkg_fused_oproj_v11a[2], warmup=5, iters=50)
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
    md_320 = run_correctness_test(pos=320, label="partial fill (primary)")
    md_640 = run_correctness_test(pos=640, label="full cache")
    md_128 = run_correctness_test(pos=128, label="nearly empty (one tile)")

    print(f"\n{'='*70}")
    print(f"CORRECTNESS SUMMARY")
    print(f"{'='*70}")
    print(f"  pos=320 PASS  max_diff={md_320:.4e}")
    print(f"  pos=640 PASS  max_diff={md_640:.4e}")
    print(f"  pos=128 PASS  max_diff={md_128:.4e}")

    # ── Benchmark ──────────────────────────────────────────────────────────────
    r = run_benchmark(pos=320, S_prior=640)

    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY (v11a, pos=320, S_prior=640)")
    print(f"{'='*70}")
    if r:
        print(f"  device_time_us       = {r.device_time_us:.2f}")
        print(f"  tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
        print(f"  dma_active_pct       = {r.dma_active_pct:.1f}%")
        print(f"  spill_bytes          = {r.spill_bytes}")
        print(f"  mfu_estimated_pct    = {r.prof.get('mfu_estimated_percent', 0):.2f}%")
        print(f"  hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
        print(f"  hbm_write_KiB        = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")
