"""
Minimal test: can nisa.tensor_tensor take fp8_e4m3fn as one operand?
Also tests: can nisa.activation with scale= be replaced with tensor_tensor?
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

PMAX = 128

# Attempt 1: tensor_tensor(bf16_dst, fp8_src, fp32_scale, multiply)
@nki.jit
def test_tt_fp8_attempt1(fp8_in, scale_in):
    """Try direct tensor_tensor with fp8 src and fp32 scale."""
    fp8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    scale_sbuf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    out_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    nisa.dma_copy(fp8_sbuf, fp8_in)
    nisa.dma_copy(scale_sbuf, scale_in)
    # Attempt: fp8 * scale -> bf16 via tensor_tensor
    nisa.tensor_tensor(out_sbuf, fp8_sbuf, scale_sbuf, nl.multiply)
    nisa.dma_copy(out_hbm, out_sbuf)
    return out_hbm


# Attempt 2: tensor_tensor(fp32_dst, fp8_src, fp32_scale, multiply)
@nki.jit
def test_tt_fp8_attempt2(fp8_in, scale_in):
    """Try tensor_tensor with fp8 src, fp32 scale, fp32 dst."""
    fp8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    scale_sbuf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    out_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.shared_hbm)

    nisa.dma_copy(fp8_sbuf, fp8_in)
    nisa.dma_copy(scale_sbuf, scale_in)
    nisa.tensor_tensor(out_sbuf, fp8_sbuf, scale_sbuf, nl.multiply)
    nisa.dma_copy(out_hbm, out_sbuf)
    return out_hbm


# Attempt 3: cast fp8 to fp32 via activation(copy), then tensor_tensor
@nki.jit
def test_cast_then_tt(fp8_in, scale_in):
    """Cast fp8->fp32 with activation(copy), then tensor_tensor for scale multiply."""
    fp8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    scale_sbuf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    f32_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    out_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    nisa.dma_copy(fp8_sbuf, fp8_in)
    nisa.dma_copy(scale_sbuf, scale_in)
    # Step 1: fp8 -> fp32 cast (activation copy - scalar engine but no scale param)
    nisa.activation(f32_sbuf, op=nl.copy, data=fp8_sbuf)
    # Step 2: fp32 * scale[128,1] -> bf16 (VectorE tensor_tensor)
    nisa.tensor_tensor(out_sbuf, f32_sbuf, scale_sbuf, nl.multiply)
    nisa.dma_copy(out_hbm, out_sbuf)
    return out_hbm


device = xm.xla_device()
# fp8 stored as int8 on XLA device (matching how v0.py receives weights)
fp8_data_cpu = torch.ones(PMAX, PMAX, dtype=torch.float8_e4m3fn).view(torch.int8)
fp8_data = fp8_data_cpu.to(device)
scale_data = (torch.arange(PMAX, dtype=torch.float32).unsqueeze(1) * 0.01 + 1.0).to(device)
xm.mark_step()

print("Testing Attempt 1: tensor_tensor(bf16_dst, fp8_src, fp32_scale)...")
try:
    out1 = test_tt_fp8_attempt1[1](fp8_data, scale_data)
    xm.mark_step()
    out1_cpu = out1.cpu()
    print(f"  Attempt 1 COMPILED+RAN OK. out shape={out1_cpu.shape}, dtype={out1_cpu.dtype}")
    print(f"  out1 sample: {out1_cpu[0,:4]}")
except Exception as e:
    print(f"  Attempt 1 FAILED: {e}")

print("Testing Attempt 2: tensor_tensor(fp32_dst, fp8_src, fp32_scale)...")
try:
    out2 = test_tt_fp8_attempt2[1](fp8_data, scale_data)
    xm.mark_step()
    out2_cpu = out2.cpu()
    print(f"  Attempt 2 COMPILED+RAN OK. out shape={out2_cpu.shape}, dtype={out2_cpu.dtype}")
    print(f"  out2 sample: {out2_cpu[0,:4]}")
except Exception as e:
    print(f"  Attempt 2 FAILED: {e}")

print("Testing Attempt 3: cast fp8->fp32 then tensor_tensor...")
try:
    out3 = test_cast_then_tt[1](fp8_data, scale_data)
    xm.mark_step()
    out3_cpu = out3.cpu()
    print(f"  Attempt 3 COMPILED+RAN OK. out shape={out3_cpu.shape}, dtype={out3_cpu.dtype}")
    print(f"  out3 sample: {out3_cpu[0,:4]}")
except Exception as e:
    print(f"  Attempt 3 FAILED: {e}")
