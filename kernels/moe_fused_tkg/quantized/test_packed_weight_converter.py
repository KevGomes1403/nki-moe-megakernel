"""
Phase 3 STEP 1b — packed-weight converter smoke test (pure Python, no Neuron compile).

Verifies the state-dict converter in qwen_complete.py produces byte-identical
output to bench_v15c_trn3.py's make_inputs recipe.

Triple-check pattern (see docs/integration_findings.md):
  1. zero/non-finite check on quantized bytes
  2. shape check
  3. byte-for-byte equality vs the in-bench reference path
"""
import sys
sys.path.insert(0, "/home/ubuntu/nki-moe")
import numpy as np
import torch

from kernels.moe_fused_tkg.quantized._qwen_integration import quantize_and_pack_gate_up, quantize_down
from kernels.moe_fused_tkg.quantized.v15c import pack_gate_up

E, H, I, GU = 128, 2048, 192, 384
PMAX, H_FREE = 128, 16
GU_PACKED_PLANES = H_FREE + 1  # 17

# --- Reference: bench_v15c_trn3.py's exact recipe ---
torch.manual_seed(42)
gate_up_bf16 = (torch.randn(E, H, GU) * 0.1).to(torch.bfloat16)
down_bf16 = (torch.randn(E, I, H) * 0.1).to(torch.bfloat16)

# Bench-path quantization (matches bench_v15c_trn3.py lines 49-57)
gu_fp32 = gate_up_bf16.to(torch.float32)
gu_scales_ref = (gu_fp32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
gu_i8_ref = (gu_fp32 / gu_scales_ref.unsqueeze(1)).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8)
gu_packed_ref = pack_gate_up(gu_i8_ref, gu_scales_ref)

dn_fp32 = down_bf16.to(torch.float32)
dn_scales_ref = (dn_fp32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
dn_i8_ref = (dn_fp32 / dn_scales_ref.unsqueeze(1)).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8)

# --- Converter path ---
gu_packed_got = quantize_and_pack_gate_up(gate_up_bf16)
dn_i8_got, dn_scales_got = quantize_down(down_bf16)

# ===========================================================================
# 1. shape checks
# ===========================================================================
assert gu_packed_got.shape == (E, GU_PACKED_PLANES, PMAX, GU), \
    f"gate_up_packed shape {gu_packed_got.shape} != expected {(E, GU_PACKED_PLANES, PMAX, GU)}"
assert gu_packed_got.dtype == torch.int8, f"gate_up_packed dtype {gu_packed_got.dtype} != int8"
assert dn_i8_got.shape == (E, I, H) and dn_i8_got.dtype == torch.int8
assert dn_scales_got.shape == (E, H) and dn_scales_got.dtype == torch.float32
print("[SHAPE CHECK] PASS")

# ===========================================================================
# 2. zero/non-finite check on quantized bytes
# ===========================================================================
# MXFP bytes CAN be zero in normal operation, but if the converter is broken,
# we'd see uniformly zero or all-NaN output. Check that the distribution is sensible.
gu_bytes_abs = gu_packed_got.abs().float()
assert gu_bytes_abs.mean() > 1.0, f"gate_up_packed mean(abs) {gu_bytes_abs.mean():.4f} is suspiciously low"
assert torch.isfinite(dn_scales_got).all(), "down_scales contains NaN/Inf"
assert (dn_scales_got > 0).all(), "down_scales contains non-positive values"
print(f"[DISTRIBUTION CHECK] PASS  gu_packed mean(abs)={gu_bytes_abs.mean():.2f}, "
      f"dn_scales range=[{dn_scales_got.min():.6f}, {dn_scales_got.max():.6f}]")

# ===========================================================================
# 3. byte-for-byte equality vs bench reference
# ===========================================================================
gu_max_diff = (gu_packed_got.to(torch.int32) - gu_packed_ref.to(torch.int32)).abs().max().item()
dn_w_max_diff = (dn_i8_got.to(torch.int32) - dn_i8_ref.to(torch.int32)).abs().max().item()
dn_scales_max_diff = (dn_scales_got - dn_scales_ref).abs().max().item()

print(f"gate_up_packed   vs bench ref: max_diff = {gu_max_diff}")
print(f"down_w (int8)    vs bench ref: max_diff = {dn_w_max_diff}")
print(f"down_scales (f32) vs bench ref: max_diff = {dn_scales_max_diff:.6e}")

assert gu_max_diff == 0, "gate_up_packed bytes differ from bench reference"
assert dn_w_max_diff == 0, "down_w int8 bytes differ from bench reference"
assert dn_scales_max_diff < 1e-7, "down_scales fp32 differ from bench reference"
print("[BYTE EQUALITY CHECK] PASS — converter produces bench-identical output")

print("\n[STEP 1b] PASS — packed-weight converter smoke test cleared")
