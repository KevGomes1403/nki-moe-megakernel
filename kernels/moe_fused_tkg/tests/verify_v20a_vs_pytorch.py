"""
verify_v20a_vs_pytorch.py

Exhaustive correctness verification of kernel_v20a against the PyTorch CPU
reference baseline (reference.py).

Tests:
  1 - Basic correctness (seed=42, B=1, scale=0.1)
  2 - Multiple seeds sweep (seeds 0-9, B=1)
  3 - Router expert index comparison: fp32 vs bf16 router logits (seeds 0-19)
  4 - Float32 vs bf16 router computation comparison (seeds 0-19)
  5 - Output sensitivity to expert index mismatches
  6 - Multi-token (B=4)
"""

import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import importlib.util

# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------
def _load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_BASE = "/home/ubuntu/nki-moe/kernels/moe_fused_tkg"
kernel_mod = _load_mod(f"{_BASE}/kernel_v20a.py", "_kv20a")
ref_mod    = _load_mod(f"{_BASE}/reference.py",    "_ref")

run_kernel    = kernel_mod.run
run_reference = ref_mod.qwen3_moe_fused_tkg_reference

device = xm.xla_device()

# Fixed model dims
H, E, K, I = 2048, 128, 8, 192

# ---------------------------------------------------------------------------
# Helper: create random inputs on CPU (bf16), then push to XLA for kernel
# ---------------------------------------------------------------------------
def make_inputs(seed, B=1, scale=0.1):
    torch.manual_seed(seed)
    inp      = (torch.randn(B, 1, H) * scale).to(torch.bfloat16)
    gamma    = (torch.randn(1, H) * scale + 1.0).to(torch.bfloat16)
    router_w = (torch.randn(H, E) * scale).to(torch.bfloat16)
    gate_up_w = torch.zeros(E, H, 2 * I, dtype=torch.bfloat16)
    gate_up_w[:, :, 0:I]   = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
    gate_up_w[:, :, I:2*I] = (torch.randn(E, H, I) * scale).to(torch.bfloat16)
    down_w   = (torch.randn(E, I, H) * scale).to(torch.bfloat16)
    return inp, gamma, router_w, gate_up_w, down_w


def run_kernel_safe(inp, gamma, router_w, gate_up_w, down_w):
    """Move to XLA, run kernel, return CPU tensor."""
    xi        = inp.to(device)
    xg        = gamma.to(device)
    xr        = router_w.to(device)
    xgu       = gate_up_w.to(device)
    xd        = down_w.to(device)
    xm.mark_step()
    out_xla = run_kernel(xi, xg, xr, xgu, xd)
    xm.mark_step()
    return out_xla.cpu()


# ---------------------------------------------------------------------------
# Reference router logic (inlined so we can compare fp32 vs bf16 precision)
# ---------------------------------------------------------------------------
def ref_router_fp32(inp, gamma, router_w, eps=1e-6):
    """Returns (affinities, topk_weights, topk_idx) using fp32 router matmul."""
    x = inp.reshape(-1, inp.shape[-1]).float()
    T, _H = x.shape
    rms = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(rms + eps)
    x_norm = x_norm * gamma.float().reshape(1, _H)
    x_norm_bf16 = x_norm.to(torch.bfloat16)

    logits = x_norm_bf16.float() @ router_w.float()  # fp32 matmul
    affinities = torch.softmax(logits, dim=-1)
    topk_weights, topk_idx = torch.topk(affinities, k=K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return affinities, topk_weights, topk_idx


def ref_router_bf16(inp, gamma, router_w, eps=1e-6):
    """Returns (affinities, topk_weights, topk_idx) using bf16 router matmul (simulates kernel)."""
    x = inp.reshape(-1, inp.shape[-1]).float()
    T, _H = x.shape
    rms = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(rms + eps)
    x_norm = x_norm * gamma.float().reshape(1, _H)
    x_norm_bf16 = x_norm.to(torch.bfloat16)

    # Kernel does: bf16 @ bf16 → fp32 psum
    logits = (x_norm_bf16 @ router_w.to(torch.bfloat16)).float()  # bf16 matmul upcast
    affinities = torch.softmax(logits, dim=-1)
    topk_weights, topk_idx = torch.topk(affinities, k=K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return affinities, topk_weights, topk_idx


def ref_forward_with_topk(inp, gamma, router_w, gate_up_w, down_w, topk_idx, topk_weights, eps=1e-6):
    """Run MLP forward pass given pre-computed routing indices & weights."""
    x = inp.reshape(-1, inp.shape[-1]).float()
    T, _H = x.shape
    rms = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(rms + eps)
    x_norm = x_norm * gamma.float().reshape(1, _H)
    x_norm_bf16 = x_norm.to(torch.bfloat16)

    output = torch.zeros(T, _H, dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            e = topk_idx[t, k].item()
            w = topk_weights[t, k].item()
            gate_w = gate_up_w[e, :, 0, :]
            up_w_e = gate_up_w[e, :, 1, :]
            d_w    = down_w[e, :, :]
            xt = x_norm_bf16[t].unsqueeze(0)
            gate = (xt @ gate_w).float()
            up   = (xt @ up_w_e).float()
            inter = F.silu(gate) * up
            inter_bf16 = inter.to(torch.bfloat16)
            out_e = (inter_bf16 @ d_w).float()
            output[t] += w * out_e.squeeze(0)
    return output.to(torch.bfloat16)


# ===========================================================================
# TEST 1 — Basic correctness (seed=42, B=1, scale=0.1)
# ===========================================================================
print("=" * 70)
print("TEST 1 — Basic correctness (seed=42, B=1, scale=0.1)")
print("=" * 70)

inp, gamma, router_w, gate_up_w, down_w = make_inputs(42, B=1)

# Reference uses [E, H, 2, I] shape
gate_up_w_4d = gate_up_w.reshape(E, H, 2, I)

ref_out = run_reference(inp, gamma, router_w, gate_up_w_4d, down_w)
kern_out = run_kernel_safe(inp, gamma, router_w, gate_up_w, down_w)

ref_out_flat  = ref_out.reshape(-1, H)
kern_out_flat = kern_out.reshape(-1, H)

diff = (ref_out_flat.float() - kern_out_flat.float()).abs()
max_diff  = diff.max().item()
mean_diff = diff.mean().item()
close = torch.allclose(ref_out_flat.float(), kern_out_flat.float(), rtol=0.05, atol=0.05)

print(f"  max_diff  = {max_diff:.4e}")
print(f"  mean_diff = {mean_diff:.4e}")
print(f"  allclose(rtol=0.05, atol=0.05) = {close}")

# NaN / Inf check
print(f"  ref  has NaN: {ref_out.isnan().any().item()},  Inf: {ref_out.isinf().any().item()}")
print(f"  kern has NaN: {kern_out.isnan().any().item()},  Inf: {kern_out.isinf().any().item()}")
print()


# ===========================================================================
# TEST 2 — Multiple seeds sweep (seeds 0–9, B=1)
# ===========================================================================
print("=" * 70)
print("TEST 2 — Multiple seeds sweep (seeds 0-9, B=1)")
print("=" * 70)
print(f"  {'seed':>4}  {'max_diff':>12}  {'allclose':>10}")
print(f"  {'-'*4}  {'-'*12}  {'-'*10}")

test2_results = []
for seed in range(10):
    inp, gamma, router_w, gate_up_w, down_w = make_inputs(seed, B=1)
    gate_up_w_4d = gate_up_w.reshape(E, H, 2, I)
    ref_out  = run_reference(inp, gamma, router_w, gate_up_w_4d, down_w)
    kern_out = run_kernel_safe(inp, gamma, router_w, gate_up_w, down_w)
    d = (ref_out.float() - kern_out.reshape(-1, H).float()).abs()
    md = d.max().item()
    cl = torch.allclose(ref_out.float(), kern_out.reshape(-1, H).float(), rtol=0.05, atol=0.05)
    test2_results.append((seed, md, cl))
    print(f"  {seed:>4}  {md:>12.4e}  {'PASS' if cl else 'FAIL':>10}")

n_pass = sum(1 for _, _, cl in test2_results if cl)
print(f"\n  Summary: {n_pass}/10 seeds PASS")
print()


# ===========================================================================
# TEST 3 — Router expert index comparison: fp32 vs bf16 (seeds 0–19)
# ===========================================================================
print("=" * 70)
print("TEST 3 — Router expert index comparison fp32 vs bf16 (seeds 0-19)")
print("=" * 70)
print(f"  {'seed':>4}  {'mismatches/8':>14}  {'mismatch positions'}")
print(f"  {'-'*4}  {'-'*14}  {'-'*30}")

test3_total_mismatches = 0
test3_total_tokens = 0

for seed in range(20):
    inp, gamma, router_w, gate_up_w, down_w = make_inputs(seed, B=1)
    _, _, idx_fp32 = ref_router_fp32(inp, gamma, router_w)
    _, _, idx_bf16 = ref_router_bf16(inp, gamma, router_w)

    # idx_fp32 and idx_bf16 are [T, K] — compare as sets per token
    T_local = idx_fp32.shape[0]
    for t in range(T_local):
        fp32_set = set(idx_fp32[t].tolist())
        bf16_set = set(idx_bf16[t].tolist())
        pos_diffs = (idx_fp32[t] != idx_bf16[t]).sum().item()
        test3_total_mismatches += len(fp32_set.symmetric_difference(bf16_set))
        test3_total_tokens += 1
        mismatch_positions = [i for i in range(K) if idx_fp32[t, i] != idx_bf16[t, i]]
        print(f"  {seed:>4}  {pos_diffs:>14}  {mismatch_positions}")

print(f"\n  Total positional mismatches over 20 seeds: {test3_total_mismatches}")
print(f"  Mismatch rate (set sym-diff): {test3_total_mismatches / (test3_total_tokens * 8):.3f} per expert slot")
print()


# ===========================================================================
# TEST 4 — Float32 vs bf16 router computation comparison (seeds 0–19)
# ===========================================================================
print("=" * 70)
print("TEST 4 — FP32 router vs BF16 router logit comparison (seeds 0-19)")
print("=" * 70)
print("  (This is the pure dtype comparison, same as Test 3 but reported differently)")
print(f"  {'seed':>4}  {'idx_match':>10}  {'logit_max_diff':>16}  {'any_diff':>10}")
print(f"  {'-'*4}  {'-'*10}  {'-'*16}  {'-'*10}")

test4_mismatch_seeds = []
for seed in range(20):
    inp, gamma, router_w, gate_up_w, down_w = make_inputs(seed, B=1)

    x = inp.reshape(-1, inp.shape[-1]).float()
    T_local, _H = x.shape
    rms = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(rms + 1e-6)
    x_norm = x_norm * gamma.float().reshape(1, _H)
    x_norm_bf16 = x_norm.to(torch.bfloat16)

    logits_fp32 = x_norm_bf16.float() @ router_w.float()
    logits_bf16 = (x_norm_bf16 @ router_w.to(torch.bfloat16)).float()

    logit_diff = (logits_fp32 - logits_bf16).abs().max().item()

    aff_fp32 = torch.softmax(logits_fp32, dim=-1)
    aff_bf16 = torch.softmax(logits_bf16, dim=-1)
    _, idx_fp32 = torch.topk(aff_fp32, k=K, dim=-1)
    _, idx_bf16 = torch.topk(aff_bf16, k=K, dim=-1)

    n_mismatches = (idx_fp32 != idx_bf16).sum().item()
    print(f"  {seed:>4}  {K - n_mismatches}/{K} match  {logit_diff:>16.4e}  {'YES' if n_mismatches > 0 else 'no':>10}")
    if n_mismatches > 0:
        test4_mismatch_seeds.append(seed)

print(f"\n  Seeds with any index mismatch: {test4_mismatch_seeds}")
print(f"  ({len(test4_mismatch_seeds)}/20 seeds affected)")
print()


# ===========================================================================
# TEST 5 — Output sensitivity to expert index mismatches
# ===========================================================================
print("=" * 70)
print("TEST 5 — Output sensitivity to expert index mismatches")
print("=" * 70)

# Find a seed with an index mismatch (from test3/4)
mismatch_seed = None
mismatch_position = None
mismatch_fp32_idx = None
mismatch_bf16_idx = None

for seed in range(20):
    inp_s, gamma_s, router_w_s, gate_up_w_s, down_w_s = make_inputs(seed, B=1)
    gate_up_w_4d_s = gate_up_w_s.reshape(E, H, 2, I)
    _, tw_fp32, idx_fp32 = ref_router_fp32(inp_s, gamma_s, router_w_s)
    _, tw_bf16, idx_bf16 = ref_router_bf16(inp_s, gamma_s, router_w_s)
    mismatched = [(k, idx_fp32[0, k].item(), idx_bf16[0, k].item())
                  for k in range(K) if idx_fp32[0, k] != idx_bf16[0, k]]
    if mismatched:
        mismatch_seed = seed
        mismatch_position = mismatched[0][0]
        mismatch_fp32_idx = mismatched[0][1]
        mismatch_bf16_idx = mismatched[0][2]
        break

if mismatch_seed is not None:
    print(f"  Using seed={mismatch_seed}, token=0, position k={mismatch_position}")
    print(f"  fp32 picks expert {mismatch_fp32_idx}, bf16 picks expert {mismatch_bf16_idx}")

    inp_s, gamma_s, router_w_s, gate_up_w_s, down_w_s = make_inputs(mismatch_seed, B=1)
    gate_up_w_4d_s = gate_up_w_s.reshape(E, H, 2, I)

    _, tw_fp32, idx_fp32 = ref_router_fp32(inp_s, gamma_s, router_w_s)

    # Baseline: reference output with fp32 routing
    out_baseline = ref_forward_with_topk(
        inp_s, gamma_s, router_w_s, gate_up_w_4d_s, down_w_s, idx_fp32, tw_fp32
    )

    # Perturbed: swap one expert at the mismatch position
    idx_perturbed = idx_fp32.clone()
    idx_perturbed[0, mismatch_position] = mismatch_bf16_idx
    # Recompute weights with perturbed index (use the bf16 affinities at the swapped position)
    aff_fp32, _, _ = ref_router_fp32(inp_s, gamma_s, router_w_s)
    tw_perturbed = tw_fp32.clone()
    # Replace the weight at the swapped position with the bf16 expert's affinity
    tw_perturbed[0, mismatch_position] = aff_fp32[0, mismatch_bf16_idx]
    tw_perturbed = tw_perturbed / tw_perturbed.sum(dim=-1, keepdim=True)

    out_perturbed = ref_forward_with_topk(
        inp_s, gamma_s, router_w_s, gate_up_w_4d_s, down_w_s, idx_perturbed, tw_perturbed
    )

    swap_diff = (out_baseline.float() - out_perturbed.float()).abs()
    print(f"  Output diff when swapping expert {mismatch_fp32_idx}→{mismatch_bf16_idx} at position k={mismatch_position}:")
    print(f"    max_diff  = {swap_diff.max().item():.4e}")
    print(f"    mean_diff = {swap_diff.mean().item():.4e}")
    print(f"    This {'IS' if swap_diff.max().item() > 0.05 else 'is NOT'} catastrophic (threshold=0.05)")
else:
    print("  No index mismatches found in seeds 0-19 — cannot measure sensitivity.")
print()


# ===========================================================================
# TEST 6 — Multi-token (B=4)
# ===========================================================================
print("=" * 70)
print("TEST 6 — Multi-token (B=4, seed=42, scale=0.1)")
print("=" * 70)

inp, gamma, router_w, gate_up_w, down_w = make_inputs(42, B=4)
gate_up_w_4d = gate_up_w.reshape(E, H, 2, I)

ref_out_b4 = run_reference(inp, gamma, router_w, gate_up_w_4d, down_w)
kern_out_b4 = None
b4_error = None
try:
    kern_out_b4 = run_kernel_safe(inp, gamma, router_w, gate_up_w, down_w)
except Exception as e:
    b4_error = str(e)[:300]

if kern_out_b4 is not None:
    ref_flat  = ref_out_b4.reshape(-1, H)
    kern_flat = kern_out_b4.reshape(-1, H)
    diff_b4 = (ref_flat.float() - kern_flat.float()).abs()
    max_diff_b4  = diff_b4.max().item()
    mean_diff_b4 = diff_b4.mean().item()
    close_b4 = torch.allclose(ref_flat.float(), kern_flat.float(), rtol=0.05, atol=0.05)
    print(f"  max_diff  = {max_diff_b4:.4e}")
    print(f"  mean_diff = {mean_diff_b4:.4e}")
    print(f"  allclose(rtol=0.05, atol=0.05) = {close_b4}")
    print(f"  ref  has NaN: {ref_out_b4.isnan().any().item()},  Inf: {ref_out_b4.isinf().any().item()}")
    print(f"  kern has NaN: {kern_out_b4.isnan().any().item()},  Inf: {kern_out_b4.isinf().any().item()}")
else:
    max_diff_b4 = None
    mean_diff_b4 = None
    close_b4 = None
    print(f"  KERNEL ERROR (B=4 not supported): {b4_error}")
print()


# ===========================================================================
# SUMMARY
# ===========================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nTest 1 (seed=42, B=1):")
print(f"  max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}, allclose={close}")

print(f"\nTest 2 (seeds 0-9, B=1):")
print(f"  {n_pass}/10 seeds pass allclose(rtol=0.05, atol=0.05)")
max_diffs_t2 = [md for _, md, _ in test2_results]
print(f"  max_diff range: [{min(max_diffs_t2):.4e}, {max(max_diffs_t2):.4e}]")

print(f"\nTest 3 (fp32 vs bf16 router indices, seeds 0-19):")
print(f"  Total positional mismatches: {test3_total_mismatches} / {test3_total_tokens * 8} slots")
print(f"  Mismatch rate: {test3_total_mismatches / (test3_total_tokens * 8):.3f}")
if test3_total_mismatches > 0:
    print(f"  => BUG 2 CONFIRMED: bf16 router matmul causes different expert selection vs fp32")
else:
    print(f"  => BUG 2 NOT OBSERVED: bf16 and fp32 router matmuls select same experts in test data")

print(f"\nTest 4 (fp32 vs bf16 logit comparison, seeds 0-19):")
print(f"  Seeds with index mismatches: {test4_mismatch_seeds} ({len(test4_mismatch_seeds)}/20)")
if test4_mismatch_seeds:
    print(f"  => BUG 1/2 confirmed: dtype of router matmul changes expert selection")
else:
    print(f"  => No dtype-induced routing changes in test data")

print(f"\nTest 5 (sensitivity to 1 expert swap):")
if mismatch_seed is not None:
    print(f"  1-expert swap causes max_diff={swap_diff.max().item():.4e}")
    if swap_diff.max().item() > 0.05:
        print(f"  => Catastrophic: single routing error causes large output change")
    else:
        print(f"  => Benign: single routing error causes small output change")
else:
    print(f"  No mismatches to test.")

print(f"\nTest 6 (B=4, seed=42):")
if max_diff_b4 is not None:
    print(f"  max_diff={max_diff_b4:.4e}, mean_diff={mean_diff_b4:.4e}, allclose={close_b4}")
else:
    print(f"  KERNEL COMPILATION ERROR for B=4 (kernel only supports B=1 / T=1)")

print(f"\nConclusion on Bugs:")
print(f"  Bug 1 (fp32 weights → bf16 matmul cast): dtype mismatch confirmed if")
print(f"         test4_mismatch_seeds is non-empty: {test4_mismatch_seeds}")
print(f"  Bug 2 (bf16 matmul precision vs fp32):    confirmed if test3_total_mismatches>0:")
print(f"         total mismatches = {test3_total_mismatches}")
