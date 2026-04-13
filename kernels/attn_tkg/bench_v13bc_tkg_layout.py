"""
Correctness + benchmark harness for v13bc_tkg_layout.

out_sb is [PMAX=128, H_wo//PMAX=16] (TKG layout) instead of flat [1, H_wo=2048].
Correctness checked against the same PyTorch reference as bench_v13bc_sbm.py.
"""

import os
import sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench_out_tkg_layout")
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

import math
import numpy as np
import torch
import nki
import nki.language as nl
import nki.isa as nisa
from nkilib.core.utils.allocator import SbufManager

from v13bc_tkg_layout import qwen3_attn_tkg_fused_oproj_v13bc
from benchmark import wrap_benchmark

# ---------------------------------------------------------------------------
# Shape constants (identical to bench_v13bc_sbm.py)
# ---------------------------------------------------------------------------
B = 1
H = 2048
d = 128
Hq_tp = 8
GQA = 8
S_prior = 640
H_wo = 2048
PMAX = 128
H1_tkg = H_wo // PMAX  # = 16


# ---------------------------------------------------------------------------
# Thin @nki.jit wrapper — stores TKG-layout out_sb to HBM
# ---------------------------------------------------------------------------
@nki.jit
def attn_kernel_wrapper(
    hidden_states, Wq, Wk, Wv, Wo,
    q_norm_weight, k_norm_weight,
    K_cache, V_cache, cos, sin, position_ids,
):
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("wrapper_outer")

    out_sb, k_rope_out, v_out = qwen3_attn_tkg_fused_oproj_v13bc(
        hidden_states, Wq, Wk, Wv, Wo,
        q_norm_weight, k_norm_weight,
        K_cache, V_cache, cos, sin, position_ids,
        out_sb=None,
        sbm=sbm,
    )

    # out_sb is presented as [PMAX=128, H1_tkg=16] TKG layout (via reshape of flat (1, H_wo)).
    # For DMA to HBM we reshape back to flat (1, H_wo) — hwdge DMA requires contiguous SBUF.
    output_hbm = nl.ndarray((B, 1, H_wo), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=output_hbm.reshape((1, H_wo)), src=out_sb.reshape((1, H_wo)), dge_mode=nisa.dge_mode.hwdge)

    sbm.close_scope()  # wrapper_outer
    sbm.close_scope()  # attn_outer (opened inside sub-function)

    return output_hbm, k_rope_out, v_out


# ---------------------------------------------------------------------------
# Reference (PyTorch float32) — identical to bench_v13bc_sbm.py
# ---------------------------------------------------------------------------
def reference_attn(hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
                   K_cache, V_cache, cos, sin, position_ids):
    B_r, _, H_r = hidden_states.shape
    d_r = 128; Hq_tp_r = 8; GQA_r = 8; S_prior_r = K_cache.shape[2]
    h = hidden_states.reshape(H_r, B_r).float()

    q = Wq.float() @ h
    k = Wk.float() @ h
    v = Wv.float() @ h

    # K RMSNorm
    k_ms = (k**2).mean(dim=0, keepdim=True) + 1e-6
    k = k * k_ms.rsqrt() * k_norm_weight.reshape(d_r, 1).float()

    # K RoPE
    half_d = d_r // 2
    cos_v = cos.reshape(d_r, B_r).float()
    sin_v = sin.reshape(d_r, B_r).float()
    rot_k = torch.cat([-k[half_d:], k[:half_d]], dim=0)
    k_rope = k * cos_v + rot_k * sin_v

    # Q RMSNorm (per head)
    q_heads = q.reshape(Hq_tp_r, d_r, B_r)
    q_ms = (q_heads**2).mean(dim=1, keepdim=True) + 1e-6
    q_heads = q_heads * q_ms.rsqrt() * q_norm_weight.reshape(1, d_r, 1).float()

    # Q RoPE (broadcast)
    cos_h = cos_v.unsqueeze(0).expand(Hq_tp_r, -1, -1)
    sin_h = sin_v.unsqueeze(0).expand(Hq_tp_r, -1, -1)
    rot_q = torch.cat([-q_heads[:, half_d:], q_heads[:, :half_d]], dim=1)
    q_rope = q_heads * cos_h + rot_q * sin_h
    q_scaled = q_rope / math.sqrt(d_r)

    # Flash-decode attention
    pos = int(position_ids[0, 0])
    K_all = torch.cat([K_cache.reshape(S_prior_r, d_r).float().T, k_rope], dim=1)
    V_all = torch.cat([V_cache.reshape(S_prior_r, d_r).float().T, v], dim=1)

    attn_out = []
    for h_idx in range(Hq_tp_r):
        q_h = q_scaled[h_idx]
        scores = (K_all.T @ q_h)
        mask = torch.zeros(S_prior_r + 1, 1)
        mask[pos:S_prior_r] = -1e9
        scores = scores + mask
        weights = torch.softmax(scores, dim=0)
        out_h = V_all @ weights
        attn_out.append(out_h)

    attn_packed = torch.stack(attn_out, dim=1).reshape(d_r, Hq_tp_r)
    attn_flat = attn_packed.T.reshape(1, Hq_tp_r * d_r)
    out = attn_flat @ Wo.float()
    return out.reshape(B_r, 1, H_r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "xla"
    torch.manual_seed(42)

    # Create inputs (same seed as bench_v13bc_sbm.py)
    hidden_states_cpu = torch.randn(B, 1, H, dtype=torch.bfloat16)
    Wq_cpu = torch.randn(Hq_tp * d, H, dtype=torch.bfloat16) * 0.02
    Wk_cpu = torch.randn(d, H, dtype=torch.bfloat16) * 0.02
    Wv_cpu = torch.randn(d, H, dtype=torch.bfloat16) * 0.02
    Wo_cpu = torch.randn(Hq_tp * d, H_wo, dtype=torch.bfloat16) * 0.02
    q_norm_weight_cpu = torch.ones(d, dtype=torch.bfloat16)
    k_norm_weight_cpu = torch.ones(d, dtype=torch.bfloat16)
    K_cache_cpu = torch.randn(B, 1, S_prior, d, dtype=torch.bfloat16) * 0.1
    V_cache_cpu = torch.randn(B, 1, S_prior, d, dtype=torch.bfloat16) * 0.1
    cos_cpu = torch.cos(torch.linspace(0, 1, d)).reshape(B, d).to(torch.bfloat16)
    sin_cpu = torch.sin(torch.linspace(0, 1, d)).reshape(B, d).to(torch.bfloat16)
    position_ids_cpu = torch.tensor([[S_prior // 2]], dtype=torch.int32)  # pos = 320

    # Move to XLA device
    hidden_states = hidden_states_cpu.to(device)
    Wq = Wq_cpu.to(device)
    Wk = Wk_cpu.to(device)
    Wv = Wv_cpu.to(device)
    Wo = Wo_cpu.to(device)
    q_norm_weight = q_norm_weight_cpu.to(device)
    k_norm_weight = k_norm_weight_cpu.to(device)
    K_cache = K_cache_cpu.to(device)
    V_cache = V_cache_cpu.to(device)
    cos = cos_cpu.to(device)
    sin = sin_cpu.to(device)
    position_ids = position_ids_cpu.to(device)

    # Run kernel
    print("Running v13bc_tkg_layout kernel...")
    output_hbm, k_rope_out, v_out = attn_kernel_wrapper(
        hidden_states, Wq, Wk, Wv, Wo,
        q_norm_weight, k_norm_weight,
        K_cache, V_cache, cos, sin, position_ids,
    )

    result = output_hbm.cpu().float().numpy()

    # Reference
    print("Computing reference...")
    ref = reference_attn(
        hidden_states_cpu, Wq_cpu, Wk_cpu, Wv_cpu, Wo_cpu,
        q_norm_weight_cpu, k_norm_weight_cpu,
        K_cache_cpu, V_cache_cpu, cos_cpu, sin_cpu, position_ids_cpu,
    ).float().numpy()

    # Correctness check
    print(f"Result shape: {result.shape}, ref shape: {ref.shape}")
    print(f"Result range: [{result.min():.4f}, {result.max():.4f}]")
    print(f"Ref range:    [{ref.min():.4f}, {ref.max():.4f}]")
    max_abs_err = np.abs(result - ref).max()
    mean_abs_err = np.abs(result - ref).mean()
    print(f"Max abs error: {max_abs_err:.6f}")
    print(f"Mean abs error: {mean_abs_err:.6f}")

    np.testing.assert_allclose(result, ref, rtol=1e-2, atol=2e-2)
    print("CORRECTNESS PASSED")

    # Benchmark
    print("\nRunning benchmark (v13bc_tkg_layout)...")
    bench_kernel = wrap_benchmark(attn_kernel_wrapper, warmup=5, iters=50)
    bench_kernel(
        hidden_states, Wq, Wk, Wv, Wo,
        q_norm_weight, k_norm_weight,
        K_cache, V_cache, cos, sin, position_ids,
    )
    r = bench_kernel.last_result
    if r:
        print(f"\n--- v13bc_tkg_layout benchmark ---")
        print(f"device_time_us       = {r.device_time_us:.2f}")
        print(f"tensor_engine_pct    = {r.tensor_engine_pct:.1f}%")
        print(f"dma_active_pct       = {r.dma_active_pct:.1f}%")
        print(f"spill_bytes          = {r.spill_bytes}")
        print(f"mfu_estimated        = {r.prof.get('mfu_estimated_percent', 0):.2f}%")
        print(f"hbm_read_KiB         = {r.prof.get('hbm_read_bytes', 0)/1024:.1f}")
        print(f"hbm_write_KiB        = {r.prof.get('hbm_write_bytes', 0)/1024:.1f}")


if __name__ == "__main__":
    main()
