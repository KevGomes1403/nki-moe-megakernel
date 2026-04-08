"""
Diagnostic script: identify NaN source in v5 with the inputs from test_v6a_e4m3.py.
Runs PyTorch reference step-by-step to find where NaN first appears.
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import numpy as np
import torch

rng = np.random.default_rng(42)
torch.manual_seed(42)

# ── Exact inputs from test_v6a_e4m3.py ──────────────────────────────────────
T, H, E, I, GU = 1, 2048, 128, 192, 384
H_SHARD_TP = 512
K = 8
EPS = 1e-6

inp           = torch.tensor(rng.random((T, 1, H)).astype(np.float16) * 0.1,       dtype=torch.bfloat16)
gamma         = torch.tensor(rng.random((1, H)).astype(np.float16) + 0.5,           dtype=torch.bfloat16)
router_w      = torch.tensor(rng.random((H, E)).astype(np.float16) * 0.1,           dtype=torch.bfloat16)
gate_up_w_i8  = torch.tensor((rng.integers(-127, 127, (E, H, GU))).astype(np.int8))
gate_up_scales = torch.tensor(rng.random((E, GU)).astype(np.float32) * 0.01 + 0.001)
down_w_i8     = torch.tensor((rng.integers(-127, 127, (E, I, H))).astype(np.int8))
down_scales   = torch.tensor(rng.random((E, H_SHARD_TP)).astype(np.float32) * 0.01 + 0.001)

print("=" * 60)
print("STEP 1: RMSNorm")
x = inp.reshape(T, H).to(torch.float32)
rms_val = x.pow(2).mean(dim=-1, keepdim=True)
print(f"  x stats: min={x.min():.4f}  max={x.max():.4f}  mean={x.mean():.6f}")
print(f"  rms (mean-sq): {rms_val.item():.6f}")
rms = (rms_val + EPS).rsqrt()
x_norm = (x * rms * gamma.to(torch.float32)).to(torch.bfloat16)
print(f"  x_norm stats: min={x_norm.float().min():.4f}  max={x_norm.float().max():.4f}")
print(f"  x_norm has NaN: {x_norm.isnan().any().item()}")

print("\nSTEP 2: Router softmax")
logits = x_norm.to(torch.float32) @ router_w.to(torch.float32)
print(f"  logits stats: min={logits.min():.4f}  max={logits.max():.4f}")
print(f"  logits has NaN: {logits.isnan().any().item()}")
probs = torch.softmax(logits, dim=-1)
print(f"  probs stats: min={probs.min():.6f}  max={probs.max():.6f}")
print(f"  probs has NaN: {probs.isnan().any().item()}")
top_vals, top_idx = torch.topk(probs, K, dim=-1)
norm_weights = top_vals / top_vals.sum(dim=-1, keepdim=True)
print(f"  norm_weights: {norm_weights[0].tolist()}")
print(f"  top_idx: {top_idx[0].tolist()}")
print(f"  norm_weights has NaN: {norm_weights.isnan().any().item()}")

print("\nSTEP 3: FP8 weight reinterpretation")
# gate_up_w is stored as int8 but reinterpreted as fp8_e4m3fn
gate_up_w_fp8 = gate_up_w_i8.view(torch.float8_e4m3fn)
gate_up_w_bf16 = gate_up_w_fp8.to(torch.bfloat16)
down_w_fp8 = down_w_i8.view(torch.float8_e4m3fn)
down_w_bf16 = down_w_fp8.to(torch.bfloat16)

print(f"  gate_up_w bf16 stats: min={gate_up_w_bf16.float().min():.4f}  max={gate_up_w_bf16.float().max():.4f}")
print(f"  gate_up_w bf16 has NaN: {gate_up_w_bf16.isnan().any().item()}")
print(f"  down_w bf16 stats: min={down_w_bf16.float().min():.4f}  max={down_w_bf16.float().max():.4f}")
print(f"  down_w bf16 has NaN: {down_w_bf16.isnan().any().item()}")

# Check if any int8 values map to NaN in fp8_e4m3fn
# fp8_e4m3fn: NaN is represented by 0x7F or 0xFF (all-ones mantissa and exponent)
nan_mask_gate = (gate_up_w_i8 == 127) | (gate_up_w_i8 == -1)  # 0x7F or 0xFF
nan_count_gate = nan_mask_gate.sum().item()
print(f"\n  int8 values that map to fp8 NaN (0x7F=127 or 0xFF=-1 as int8): {nan_count_gate}")

# fp8_e4m3fn: 0x7F = +NaN, 0x80 = -0, 0xFF = -NaN
# In signed int8: 0x7F = 127, 0xFF = -1
nan_mask_all = gate_up_w_bf16.isnan()
print(f"  NaN positions in gate_up_w_bf16: {nan_mask_all.sum().item()}")
nan_mask_down = down_w_bf16.isnan()
print(f"  NaN positions in down_w_bf16: {nan_mask_down.sum().item()}")

print("\nSTEP 4: Check dequant scale ranges")
print(f"  gate_up_scales: min={gate_up_scales.min():.6f}  max={gate_up_scales.max():.6f}")
# After matmul, gate_up_out = x_norm @ gate_up_w_bf16[e]  (shape [GU])
# then gate_out = gate_up_out[:I] * gate_up_scales[e, :I]
# Let's check if the matmul output * scale can overflow BF16
e0 = top_idx[0, 0].item()
gate_up_out_e0 = (x_norm[0:1].to(torch.float32) @ gate_up_w_bf16[e0].to(torch.float32)).squeeze(0)
print(f"\n  Expert {e0} gate_up matmul output: min={gate_up_out_e0.min():.4f}  max={gate_up_out_e0.max():.4f}")
print(f"  gate_up_out has NaN: {gate_up_out_e0.isnan().any().item()}")

gate_out_e0 = gate_up_out_e0[:I] * gate_up_scales[e0, :I]
up_out_e0   = gate_up_out_e0[I:] * gate_up_scales[e0, I:]
print(f"  gate_out (scaled): min={gate_out_e0.min():.6f}  max={gate_out_e0.max():.6f}")
print(f"  gate_out has NaN: {gate_out_e0.isnan().any().item()}")

inter_e0 = torch.nn.functional.silu(gate_out_e0) * up_out_e0
print(f"  inter (silu*up): min={inter_e0.min():.6f}  max={inter_e0.max():.6f}")
print(f"  inter has NaN: {inter_e0.isnan().any().item()}")

down_out_e0 = (inter_e0.unsqueeze(0) @ down_w_bf16[e0, :, 0:H_SHARD].to(torch.float32)).squeeze(0)
print(f"  down_out: min={down_out_e0.min():.6f}  max={down_out_e0.max():.6f}")
print(f"  down_out has NaN: {down_out_e0.isnan().any().item()}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
# Now run full reference to see output
ref_output = torch.zeros(T, H_SHARD_TP, dtype=torch.float32)
any_nan = False
for t in range(T):
    for ki in range(K):
        e = top_idx[t, ki].item()
        w = norm_weights[t, ki].item()
        gu_out = (x_norm[t:t+1].to(torch.float32) @ gate_up_w_bf16[e].to(torch.float32)).squeeze(0)
        if gu_out.isnan().any():
            print(f"  NaN in gate_up matmul at t={t} ki={ki} e={e}")
            any_nan = True
        gate_out = gu_out[:I] * gate_up_scales[e, :I]
        up_out   = gu_out[I:] * gate_up_scales[e, I:]
        inter = torch.nn.functional.silu(gate_out) * up_out
        if inter.isnan().any():
            print(f"  NaN in inter at t={t} ki={ki} e={e}")
            any_nan = True
        d_out = (inter.unsqueeze(0) @ down_w_bf16[e, :, 0:H_SHARD].to(torch.float32)).squeeze(0)
        if d_out.isnan().any():
            print(f"  NaN in down_out at t={t} ki={ki} e={e}")
            any_nan = True
        d_scaled = d_out * down_scales[e]
        ref_output[t] += w * d_scaled

print(f"\nRef output has NaN: {ref_output.isnan().any().item()}")
print(f"Ref output stats: min={ref_output.min():.6f}  max={ref_output.max():.6f}")
if not any_nan:
    print("\nNo NaN found in reference computation — NaN may be hardware-specific (kernel bug)")
