"""
Test harness for kernel_v31b.py
Compares v31b (Plan C3-on-v4: exact tile sizing, no padding waste) against a
pure PyTorch fp32-promoted reference.
Tolerance: atol=1e-3, rtol=1e-2 (fp32-promoted reference per CLAUDE.md).
Also benchmarks v31b.
"""
import sys
import os

# Must be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v31b"
)

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_v31b import run as run_v31b
from kernel_v31b import qwen3_moe_fused_tkg_sbuf_io

# Fixed dims matching kernel contracts
_H = 2048
_E = 128
_I = 192
_K = 8

rng = np.random.default_rng(42)
B = 1

device = xm.xla_device()

def make_tensor(shape, dtype=torch.bfloat16, scale=0.1):
    arr = rng.standard_normal(shape).astype(np.float32) * scale
    return torch.tensor(arr, dtype=dtype).to(device)

# Generate inputs matching run() contract
inp       = make_tensor((B, 1, _H))           # [B, 1, H] bf16
gamma     = make_tensor((1, _H))              # [1, H] bf16
router_w  = make_tensor((_H, _E))             # [H, E] bf16
gate_up_w = make_tensor((_E, _H, 2 * _I))     # [E, H, 2*I] bf16
down_w    = make_tensor((_E, _I, _H))         # [E, I, H] bf16

# -----------------------------------------------------------------------
# PyTorch reference (fp32-promoted)
# -----------------------------------------------------------------------

def pytorch_reference(inp_flat, gamma_flat, router_w, gate_up_w, down_w):
    """
    inp_flat:  [T, H=2048]     bf16
    gamma_flat:[H=2048]        bf16
    router_w:  [H=2048, E=128] bf16
    gate_up_w: [E=128, H=2048, 2*I=384] bf16  (gate cols 0:I, up cols I:2I)
    down_w:    [E=128, I=192, H=2048]   bf16
    Returns:   [T, H=2048]     bf16
    """
    H, E, I_dim, K = 2048, 128, 192, 8
    x = inp_flat.float()
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    normed = x * rms * gamma_flat.float()   # [T, H]

    logits = normed @ router_w.float()      # [T, E]
    probs  = torch.softmax(logits, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, K, dim=-1)  # [T, K]
    norm_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

    T = inp_flat.shape[0]
    output = torch.zeros(T, H, dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            e = topk_idx[t, k].item()
            w = norm_weights[t, k].item()
            x_t = normed[t]
            gu  = gate_up_w[e].float()          # [H, 2I]
            gate = x_t @ gu[:, :I_dim]          # [I]
            up   = x_t @ gu[:, I_dim:]          # [I]
            inter = F.silu(gate) * up
            out_e = inter @ down_w[e].float()   # [H]
            output[t] += w * out_e
    return output.to(torch.bfloat16)

# Compute reference on CPU (detach from XLA device)
print("Computing PyTorch reference...")
inp_cpu       = inp.cpu()          # [B, 1, H]
gamma_cpu     = gamma.cpu()        # [1, H]
router_w_cpu  = router_w.cpu()     # [H, E]
gate_up_w_cpu = gate_up_w.cpu()    # [E, H, 2*I]
down_w_cpu    = down_w.cpu()       # [E, I, H]

# Flatten for reference: inp [B, 1, H] -> [T, H], gamma [1, H] -> [H]
inp_flat_cpu   = inp_cpu.reshape(B, _H)   # [T=B, H]
gamma_flat_cpu = gamma_cpu.squeeze(0)     # [H]

ref_out = pytorch_reference(inp_flat_cpu, gamma_flat_cpu, router_w_cpu, gate_up_w_cpu, down_w_cpu)
# ref_out: [T, H] bf16 on CPU

print("Running v31b kernel...")
out_v31b = run_v31b(inp, gamma, router_w, gate_up_w, down_w)
xm.mark_step()

# Transfer to CPU for comparison
out_v31b_cpu = out_v31b.cpu().float().numpy()
ref_out_np   = ref_out.float().numpy()

max_diff  = np.max(np.abs(ref_out_np - out_v31b_cpu))
mean_diff = np.mean(np.abs(ref_out_np - out_v31b_cpu))

print(f"\n--- Comparison Results ---")
print(f"ref   output shape : {ref_out_np.shape}")
print(f"v31b  output shape : {out_v31b_cpu.shape}")
print(f"max_diff           : {max_diff:.6f}")
print(f"mean_diff          : {mean_diff:.6f}")

try:
    np.testing.assert_allclose(ref_out_np, out_v31b_cpu, rtol=1e-2, atol=1e-3)
    print("\nPASS — v31b matches PyTorch reference within atol=1e-3, rtol=1e-2")
except AssertionError as e:
    print(f"\nFAIL — {e}")
    # Extra diagnostics
    diff = np.abs(ref_out_np - out_v31b_cpu)
    failing = np.argwhere(diff > 1e-3 + 1e-2 * np.abs(ref_out_np))
    print(f"Number of failing elements: {len(failing)}")
    if len(failing) > 0:
        idx = failing[0]
        print(f"First failing element idx={tuple(idx)}: ref={ref_out_np[tuple(idx)]:.6f}, v31b={out_v31b_cpu[tuple(idx)]:.6f}")
        # Show worst element
        worst_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"Worst element idx={worst_idx}: ref={ref_out_np[worst_idx]:.6f}, v31b={out_v31b_cpu[worst_idx]:.6f}, diff={diff[worst_idx]:.6f}")
        # Show distribution of failures across H dimension
        if diff.ndim >= 2:
            h_diffs = diff.max(axis=0)
            top_h = np.argsort(h_diffs)[-10:]
            print(f"Top 10 H indices with worst diff: {top_h}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Benchmark section
# -----------------------------------------------------------------------
print("\n--- Benchmarking v31b ---")

import glob
import time
from benchmark import wrap_benchmark, _parse_ntff, _print_profile_results, BenchmarkResult, _pending_bench, _COMPILE_WORKDIR

# Prepare benchmark inputs in the shape the kernel expects:
# qwen3_moe_fused_tkg_sbuf_io expects inp [B, 1, H], gamma [1, H], etc.
rng2 = np.random.default_rng(123)
def make2(shape, dtype=torch.bfloat16, scale=0.1):
    arr = rng2.standard_normal(shape).astype(np.float32) * scale
    return torch.tensor(arr, dtype=dtype).to(device)

inp_b       = make2((B, 1, _H))
gamma_b     = make2((1, _H))
router_w_b  = make2((_H, _E))
gate_up_w_b = make2((_E, _H, 2 * _I))
down_w_b    = make2((_E, _I, _H))

# The kernel is already compiled. The inspect dir has NTFF/NEFF from the correctness run.
# We run one more timed iteration and look for the latest NTFF written.
bench_inspect_dir = os.environ.get("NEURON_RT_INSPECT_OUTPUT_DIR", "")

# Warm up
print("Warming up...")
for _ in range(5):
    qwen3_moe_fused_tkg_sbuf_io[2](inp_b, gamma_b, router_w_b, gate_up_w_b, down_w_b)
    xm.mark_step()

# Snapshot ntff files before timed run
def _snap_ntffs(d):
    fs = glob.glob(f"{d}/**/*.ntff", recursive=True)
    return {p: os.path.getmtime(p) for p in fs}

before_ntffs = _snap_ntffs(bench_inspect_dir) if bench_inspect_dir else {}

print("Running timed iteration...")
qwen3_moe_fused_tkg_sbuf_io[2](inp_b, gamma_b, router_w_b, gate_up_w_b, down_w_b)
xm.mark_step()

# Wait briefly for NTFF to be written
time.sleep(5)

# Find new/updated NTFF
ntff_path = None
neff_path = None
if bench_inspect_dir:
    after_ntffs = _snap_ntffs(bench_inspect_dir)
    new_ntffs = [p for p, mt in after_ntffs.items() if before_ntffs.get(p) != mt]
    if new_ntffs:
        ntff_path = max(new_ntffs, key=os.path.getmtime)
        # Find paired NEFF in same directory
        ntff_dir = os.path.dirname(ntff_path)
        neff_candidates = glob.glob(f"{ntff_dir}/*.neff")
        if neff_candidates:
            neff_path = max(neff_candidates, key=os.path.getmtime)
    else:
        # Fall back to the most recently written NTFF
        all_ntffs = list(after_ntffs.keys())
        if all_ntffs:
            ntff_path = max(all_ntffs, key=os.path.getmtime)
            ntff_dir = os.path.dirname(ntff_path)
            neff_candidates = glob.glob(f"{ntff_dir}/*.neff")
            if neff_candidates:
                neff_path = max(neff_candidates, key=os.path.getmtime)
            print(f"[benchmark] Using most recent NTFF (kernel cached): {os.path.basename(ntff_path)}")

prof = {}
if ntff_path and neff_path:
    print(f"[benchmark] Parsing NTFF: {os.path.basename(ntff_path)}")
    print(f"[benchmark] Parsing NEFF: {os.path.basename(neff_path)}")
    prof = _parse_ntff(neff_path, ntff_path)
    _print_profile_results(prof, "qwen3_moe_fused_tkg_sbuf_io (v31b)")
else:
    print("[benchmark] No NTFF found — profiling data unavailable.")

# Also find NEFF from compile workdir for neuron-bench
bench_neffs = glob.glob(f"{_COMPILE_WORKDIR}/**/*.neff", recursive=True)
bench_neff = max(bench_neffs, key=os.path.getmtime) if bench_neffs else None
if bench_neff:
    _pending_bench.append({
        "neff": bench_neff, "warmup": 5, "iters": 50, "name": "qwen3_moe_fused_tkg_sbuf_io_v31b"
    })

r = BenchmarkResult(prof, bench_neff or neff_path or "", ntff_path)
print(f"\n--- v31b Benchmark Metrics ---")
print(f"device_time_us        : {r.device_time_us:.2f}")
print(f"tensor_engine_pct     : {r.tensor_engine_pct:.2f}")
print(f"dma_active_pct        : {r.dma_active_pct:.2f}")
print(f"spill_bytes           : {r.spill_bytes}")
print(f"mfu_estimated_percent : {prof.get('mfu_estimated_percent', 'N/A')}")
print(f"hbm_read_KiB          : {prof.get('hbm_read_bytes', 0)/1024:.1f}")
print(f"hbm_write_KiB         : {prof.get('hbm_write_bytes', 0)/1024:.1f}")
