"""
Correctness harness: compare qwen3_router_topk_cte (optimized, [H, T] layout)
against a numpy reference implementation.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
    python kernels/router_topk/test_optimized.py
"""

import numpy as np
import torch
import torch_xla.core.xla_model as xm

# Optimized kernel (x: [H, T]) — Plans A+B+C applied
from qwen3_router_topk_cte import qwen3_router_topk_cte

# ---- Constants (must match kernel hardcodes) ----
H = 2048
E = 128
K = 8
T = 128   # full batch

rng = np.random.default_rng(42)

# ---- Generate random inputs in float32 for reference ----
x_np = rng.standard_normal((T, H)).astype(np.float32)   # [T, H] for numpy reference
w_np = rng.standard_normal((H, E)).astype(np.float32)   # [H, E]

# ---- Numpy reference ----
# logits[t, e] = sum_h(x[t,h] * w[h,e])
ref_logits = x_np @ w_np   # [T, E]

# softmax
shifted = ref_logits - ref_logits.max(axis=1, keepdims=True)
exp_vals = np.exp(shifted)
ref_affinities = exp_vals / exp_vals.sum(axis=1, keepdims=True)

# top-K indices (argsort descending)
ref_topk_idx = np.argsort(-ref_affinities, axis=1)[:, :K]   # [T, K]

# L1 normalization on top-K affinities
ref_topk_vals = ref_affinities[np.arange(T)[:, None], ref_topk_idx]   # [T, K]
sum_topk = ref_topk_vals.sum(axis=1, keepdims=True)
ref_topk_norm = ref_topk_vals / sum_topk

# Scattered expert affinities: [T, E]
ref_scattered = np.zeros((T, E), dtype=np.float32)
for t in range(T):
    for ki in range(K):
        e_idx = ref_topk_idx[t, ki]
        ref_scattered[t, e_idx] = ref_topk_norm[t, ki]

# ---- Run the optimized NKI kernel ----
device = xm.xla_device()

router_logits_out   = torch.zeros((T, E), dtype=torch.float32, device=device)
expert_affi_out     = torch.zeros((T, E), dtype=torch.float32, device=device)
expert_index_out    = torch.zeros((T, K), dtype=torch.int32,   device=device)

# Kernel expects x in [H, T] layout — transpose and make contiguous
x_np_bf16 = x_np.astype(np.float16)   # kernel uses bfloat16
w_np_bf16 = w_np.astype(np.float16)

x_transposed = torch.tensor(x_np_bf16.T.copy(), dtype=torch.bfloat16, device=device)  # [H, T]
w_t          = torch.tensor(w_np_bf16,           dtype=torch.bfloat16, device=device)  # [H, E]

print("Running optimized kernel (x: [H, T], Plans A+B+C) ...")
qwen3_router_topk_cte[2](
    x_transposed, w_t,
    router_logits_out, expert_affi_out, expert_index_out,
)
xm.mark_step()

opt_rl  = router_logits_out.cpu().numpy()
opt_ea  = expert_affi_out.cpu().numpy()
opt_ei  = expert_index_out.cpu().numpy()

# ---- Compare against numpy reference ----
# Note: bf16 accumulation means we expect small but non-zero differences vs fp32 reference
print()
outputs = [
    ("router_logits",      ref_logits,    opt_rl),
    ("expert_affinities",  ref_scattered, opt_ea),
    ("expert_index (sorted)", np.sort(ref_topk_idx, axis=1).astype(np.float32),
                              np.sort(opt_ei,        axis=1).astype(np.float32)),
]

all_pass = True
for name, ref, opt in outputs:
    max_diff = np.abs(ref - opt).max()
    try:
        np.testing.assert_allclose(opt, ref, rtol=1e-3, atol=1e-3)
        print(f"  {name}: max_diff={max_diff:.2e}  PASS")
    except AssertionError as e:
        print(f"  {name}: max_diff={max_diff:.2e}  FAIL")
        print(f"    {e}")
        all_pass = False

print()
if all_pass:
    print("All outputs match — optimized kernel is correct.")
else:
    print("FAILURES detected — see above.")
    raise SystemExit(1)
