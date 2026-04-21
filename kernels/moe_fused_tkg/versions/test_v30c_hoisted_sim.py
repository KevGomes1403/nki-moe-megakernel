"""
Test B — hoisted mode: invoke v30c with both new kwargs supplied (pre-loaded SBUF tensors).
Output must match the PyTorch fp32-promoted reference within atol=1e-3, rtol=1e-2.

The hoisted wrapper (qwen3_moe_fused_tkg_sbuf_io_hoisted) pre-loads gamma and router_w
into SBUF before calling the sub-kernel with gamma_sb_ready and router_w_wide_sb set.

Run with trn2 simulator:
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python test_v30c_hoisted_sim.py
"""
import sys
import os

# Must be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v30c_hoisted_mode"
)

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_v30c_hoisted import qwen3_moe_fused_tkg_sbuf_io_hoisted

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

# Compute reference on CPU
print("Computing PyTorch reference...")
inp_cpu       = inp.cpu()
gamma_cpu     = gamma.cpu()
router_w_cpu  = router_w.cpu()
gate_up_w_cpu = gate_up_w.cpu()
down_w_cpu    = down_w.cpu()

inp_flat_cpu   = inp_cpu.reshape(B, _H)
gamma_flat_cpu = gamma_cpu.squeeze(0)

ref_out = pytorch_reference(inp_flat_cpu, gamma_flat_cpu, router_w_cpu, gate_up_w_cpu, down_w_cpu)

# -----------------------------------------------------------------------
# Test B — hoisted mode: both gamma and router_w pre-loaded
# -----------------------------------------------------------------------
print("Running v30c hoisted kernel (both kwargs supplied via hoisted_gamma=True, hoisted_router_w=True)...")
out_hoisted = qwen3_moe_fused_tkg_sbuf_io_hoisted[2](
    inp, gamma, router_w, gate_up_w, down_w,
    hoisted_gamma=True,
    hoisted_router_w=True,
)
xm.mark_step()

out_hoisted_cpu = out_hoisted.cpu().float().numpy()
ref_out_np      = ref_out.float().numpy()

max_diff  = np.max(np.abs(ref_out_np - out_hoisted_cpu))
mean_diff = np.mean(np.abs(ref_out_np - out_hoisted_cpu))

print(f"\n--- Comparison Results (hoisted mode vs PyTorch reference) ---")
print(f"ref     output shape : {ref_out_np.shape}")
print(f"hoisted output shape : {out_hoisted_cpu.shape}")
print(f"max_diff             : {max_diff:.6e}")
print(f"mean_diff            : {mean_diff:.6e}")

try:
    np.testing.assert_allclose(ref_out_np, out_hoisted_cpu, rtol=1e-2, atol=1e-3)
    print(f"\nPASS (hoisted mode) — v30c hoisted matches PyTorch reference within atol=1e-3, rtol=1e-2")
    print(f"  max_diff={max_diff:.3e}")
except AssertionError as e:
    print(f"\nFAIL (hoisted mode) — {e}")
    diff = np.abs(ref_out_np - out_hoisted_cpu)
    failing = np.argwhere(diff > 1e-3 + 1e-2 * np.abs(ref_out_np))
    print(f"Number of failing elements: {len(failing)}")
    if len(failing) > 0:
        idx = failing[0]
        print(f"First failing element idx={tuple(idx)}: ref={ref_out_np[tuple(idx)]:.6f}, hoisted={out_hoisted_cpu[tuple(idx)]:.6f}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Also test gamma-only hoisted mode
# -----------------------------------------------------------------------
print("\nRunning v30c hoisted kernel (gamma-only hoisted: hoisted_gamma=True, hoisted_router_w=False)...")
out_gamma_only = qwen3_moe_fused_tkg_sbuf_io_hoisted[2](
    inp, gamma, router_w, gate_up_w, down_w,
    hoisted_gamma=True,
    hoisted_router_w=False,
)
xm.mark_step()

out_gamma_only_cpu = out_gamma_only.cpu().float().numpy()
max_diff_gamma = np.max(np.abs(ref_out_np - out_gamma_only_cpu))

try:
    np.testing.assert_allclose(ref_out_np, out_gamma_only_cpu, rtol=1e-2, atol=1e-3)
    print(f"PASS (gamma-only hoisted) — max_diff={max_diff_gamma:.3e}")
except AssertionError as e:
    print(f"FAIL (gamma-only hoisted) — {e}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Also test router_w-only hoisted mode
# -----------------------------------------------------------------------
print("\nRunning v30c hoisted kernel (router_w-only: hoisted_gamma=False, hoisted_router_w=True)...")
out_router_only = qwen3_moe_fused_tkg_sbuf_io_hoisted[2](
    inp, gamma, router_w, gate_up_w, down_w,
    hoisted_gamma=False,
    hoisted_router_w=True,
)
xm.mark_step()

out_router_only_cpu = out_router_only.cpu().float().numpy()
max_diff_router = np.max(np.abs(ref_out_np - out_router_only_cpu))

try:
    np.testing.assert_allclose(ref_out_np, out_router_only_cpu, rtol=1e-2, atol=1e-3)
    print(f"PASS (router_w-only hoisted) — max_diff={max_diff_router:.3e}")
except AssertionError as e:
    print(f"FAIL (router_w-only hoisted) — {e}")
    sys.exit(1)

print("\n=== All hoisted-mode tests PASSED ===")
