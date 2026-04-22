"""
Trn3 benchmark for current attention TKG kernel — v13bc_sbm_tiled.

Mirror of bench_v14a_trn3.py for attention:
  - NEURON_PLATFORM_TARGET_OVERRIDE=trn3
  - LNC=2
  - trn3 skill's benchmark.py for wrap_benchmark
  - full BenchmarkResult fields reported

No correctness check — just the benchmark. Correctness was verified
against the trn2 baseline in bench_v13bc_sbm_tiled.py.
"""

import os
import sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v13bc_trn3"
)

import shutil
_HERE = os.path.dirname(os.path.abspath(__file__))
shutil.copy(
    "/home/ubuntu/nki-moe/.claude/skills/nki-kernel-optimizer-trn3/scripts/benchmark.py",
    os.path.join(_HERE, "benchmark.py"),
)

sys.path.insert(0, "/home/ubuntu/nki-moe")
sys.path.insert(0, _HERE)

import torch
import nki
import nki.language as nl
import nki.isa as nisa
from nkilib.core.utils.allocator import SbufManager

from v13bc_sbm_tiled import qwen3_attn_tkg_fused_oproj_v13bc
from benchmark import wrap_benchmark

# ---------------------------------------------------------------------------
# Shape constants (identical to bench_v13bc_sbm_tiled.py)
# ---------------------------------------------------------------------------
B = 1
H = 2048
d = 128
Hq_tp = 8
GQA = 8
S_prior = 640
H_wo = 2048
PMAX = 128


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

    output_hbm = nl.ndarray((128, 16), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=output_hbm, src=out_sb, dge_mode=nisa.dge_mode.hwdge)

    sbm.close_scope()
    sbm.close_scope()

    return output_hbm, k_rope_out, v_out


def make_inputs(seed=42):
    torch.manual_seed(seed)
    device = "xla"

    hidden_states = torch.randn(B, 1, H, dtype=torch.bfloat16).to(device)
    Wq = (torch.randn(Hq_tp * d, H, dtype=torch.bfloat16) * 0.02).to(device)
    Wk = (torch.randn(d, H, dtype=torch.bfloat16) * 0.02).to(device)
    Wv = (torch.randn(d, H, dtype=torch.bfloat16) * 0.02).to(device)
    Wo = (torch.randn(Hq_tp * d, H_wo, dtype=torch.bfloat16) * 0.02).to(device)
    q_norm_weight = torch.ones(d, dtype=torch.bfloat16).to(device)
    k_norm_weight = torch.ones(d, dtype=torch.bfloat16).to(device)
    K_cache = (torch.randn(B, 1, S_prior, d, dtype=torch.bfloat16) * 0.1).to(device)
    V_cache = (torch.randn(B, 1, S_prior, d, dtype=torch.bfloat16) * 0.1).to(device)
    cos = torch.cos(torch.linspace(0, 1, d)).reshape(B, d).to(torch.bfloat16).to(device)
    sin = torch.sin(torch.linspace(0, 1, d)).reshape(B, d).to(torch.bfloat16).to(device)
    position_ids = torch.tensor([[S_prior // 2]], dtype=torch.int32).to(device)

    return (hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
            K_cache, V_cache, cos, sin, position_ids)


bench = wrap_benchmark(attn_kernel_wrapper, warmup=5, iters=50)
args = make_inputs()

print("Running v13bc attention bench (trn3)...")
bench(*args)

r = bench.last_result
if r and r.prof:
    prof = r.prof
    def pct(key): return prof.get(key, 0) * 100
    print("\n=== v13bc attention Benchmark (trn3) ===")
    print(f"device_time_us       = {r.device_time_us:.2f}")
    print(f"tensor_engine_pct    = {pct('tensor_engine_active_time_percent'):.2f}%")
    print(f"vector_engine_pct    = {pct('vector_engine_active_time_percent'):.2f}%")
    print(f"scalar_engine_pct    = {pct('scalar_engine_active_time_percent'):.2f}%")
    print(f"gpsimd_engine_pct    = {pct('gpsimd_engine_active_time_percent'):.2f}%")
    print(f"dma_active_pct       = {pct('dma_active_time_percent'):.2f}%")
    print(f"spill_bytes          = {r.spill_bytes}")
    print(f"mfu_estimated_pct    = {prof.get('mfu_estimated_percent', 0):.4f}%")
    print(f"mbu_estimated_pct    = {prof.get('mbu_estimated_percent', 0):.4f}%")
    print(f"mm_arithmetic_intensity = {prof.get('mm_arithmetic_intensity', 0):.3f}")
    print(f"hbm_read_KiB         = {prof.get('hbm_read_bytes', 0)/1024:.1f}")
    print(f"hbm_write_KiB        = {prof.get('hbm_write_bytes', 0)/1024:.1f}")
    print(f"cc_op_count          = {prof.get('cc_op_count', 0)}")
    print(f"cc_op_active_time_us = {prof.get('cc_op_active_time', 0)*1e6:.2f}")
else:
    print("No profile data")
