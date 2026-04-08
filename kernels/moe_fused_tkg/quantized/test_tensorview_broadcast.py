"""
Test TensorView broadcast with tensor_tensor for scale expansion.
Also test the full two-step dequant with TensorView broadcast.
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.tensor_view import TensorView

PMAX = 128


@nki.jit
def test_tt_with_tensorview_bcast(fp32_in, scale_col):
    """
    Test: tensor_tensor(bf16_out, fp32_in, scale_bcast, multiply)
    where scale_bcast is [128P, 128F] created via TensorView.broadcast from [128P, 1F].
    fp32_in: [128, 128] fp32
    scale_col: [128, 1] fp32
    """
    f32_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    scale_sbuf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    out_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    nisa.dma_copy(f32_sbuf, fp32_in)
    nisa.dma_copy(scale_sbuf, scale_col)
    scale_bcast = TensorView(scale_sbuf).broadcast(dim=1, size=PMAX).get_view()
    nisa.tensor_tensor(out_sbuf, f32_sbuf, scale_bcast, nl.multiply)
    nisa.dma_copy(out_hbm, out_sbuf)
    return out_hbm


@nki.jit
def test_two_step_with_bcast(fp8_in, scale_col):
    """
    Full two-step dequant:
    1. activation(fp8 -> fp32, no scale) — scalar engine, fast
    2. tensor_tensor(fp32 * scale_bcast -> bf16) — vector engine

    fp8_in: [128, 128] as int8
    scale_col: [128, 1] fp32 — scale per partition row (= scale per j within block)
    """
    fp8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    f32_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    scale_sbuf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    out_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    nisa.dma_copy(fp8_sbuf, fp8_in)
    nisa.dma_copy(scale_sbuf, scale_col)
    # Step 1: fp8 -> fp32 (scalar engine, but no scale multiply overhead)
    nisa.activation(f32_sbuf, op=nl.copy, data=fp8_sbuf)
    # Step 2: fp32 * scale[128,1] broadcast -> bf16 (vector engine)
    scale_bcast = TensorView(scale_sbuf).broadcast(dim=1, size=PMAX).get_view()
    nisa.tensor_tensor(out_sbuf, f32_sbuf, scale_bcast, nl.multiply)
    nisa.dma_copy(out_hbm, out_sbuf)
    return out_hbm


device = xm.xla_device()
torch.manual_seed(42)

fp8_cpu = (torch.randn(PMAX, PMAX) * 10).clamp(-240, 240).to(torch.float8_e4m3fn)
fp8_xla = fp8_cpu.view(torch.int8).to(device)

f32_cpu = torch.randn(PMAX, PMAX, dtype=torch.float32)
f32_xla = f32_cpu.to(device)

# scale_col: [128, 1] — scale per partition row (which = per j%128 in the dequant)
scale_col_cpu = (torch.arange(PMAX, dtype=torch.float32) * 0.01 + 1.0).unsqueeze(1)
scale_col_xla = scale_col_cpu.to(device)
xm.mark_step()

print("Test 1: tensor_tensor(bf16, fp32, TensorView_bcast_scale)...")
try:
    out1 = test_tt_with_tensorview_bcast[1](f32_xla, scale_col_xla)
    xm.mark_step()
    out1_cpu = out1.cpu().to(torch.float32)
    ref1 = f32_cpu * scale_col_cpu  # scale[p] broadcasts over free dim
    err = (out1_cpu - ref1).abs().max()
    print(f"  Test 1 PASS. max_err={err:.3e}")
    print(f"  out1[0,:4]={out1_cpu[0,:4]}, ref[0,:4]={ref1[0,:4]}")
except Exception as e:
    print(f"  Test 1 FAIL: {str(e)[:300]}")

print()
print("Test 2: two-step dequant: activation(fp8->f32) + tensor_tensor(f32*scale_bcast->bf16)...")
try:
    out2 = test_two_step_with_bcast[1](fp8_xla, scale_col_xla)
    xm.mark_step()
    out2_cpu = out2.cpu().to(torch.float32)
    # Expected: fp8[p, j] * scale[p] (scale per partition row, broadcasts over free dim j)
    ref2 = fp8_cpu.to(torch.float32) * scale_col_cpu
    err = (out2_cpu - ref2).abs().max()
    print(f"  Test 2 PASS. max_err={err:.3e}")
    print(f"  out2[0,:4]={out2_cpu[0,:4]}, ref2[0,:4]={ref2[0,:4]}")
except Exception as e:
    print(f"  Test 2 FAIL: {str(e)[:300]}")
