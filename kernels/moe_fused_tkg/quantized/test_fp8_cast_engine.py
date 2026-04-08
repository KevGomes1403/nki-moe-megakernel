"""
Test whether fp8->fp32 cast (no scale) is scalar or vector engine.
Also test the 2-step approach: cast then scale_multiply.
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "/tmp/test_fp8_engine_profile"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch_xla.core.xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

PMAX = 128
N_REPS = 16  # repeat to get measurable times


@nki.jit
def test_fp8_cast_no_scale(fp8_in):
    """
    fp8 -> fp32 cast using activation(copy) WITHOUT scale.
    This is what we hypothesize may be VectorE (pure dtype conversion).
    """
    # fp8_in comes as int8 (dtype reinterpret in ap())
    out = nl.ndarray((PMAX, N_REPS * PMAX), dtype=nl.float32, buffer=nl.shared_hbm)
    for rep in nl.static_range(N_REPS):
        fp8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        f32_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            fp8_sbuf,
            fp8_in.ap(pattern=[[PMAX, PMAX], [1, PMAX]], offset=0),
        )
        nisa.activation(f32_sbuf, op=nl.copy, data=fp8_sbuf)
        nisa.dma_copy(out[0:PMAX, nl.ds(rep * PMAX, PMAX)], f32_sbuf)
    return out


@nki.jit
def test_fp8_cast_with_scale(fp8_in, scale_in):
    """
    fp8 -> bf16 cast using activation(copy, scale=) — the ORIGINAL approach.
    This is the scalar engine bottleneck we're trying to replace.
    """
    out = nl.ndarray((PMAX, N_REPS * PMAX), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for rep in nl.static_range(N_REPS):
        fp8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        scale_sbuf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        bf16_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            fp8_sbuf,
            fp8_in.ap(pattern=[[PMAX, PMAX], [1, PMAX]], offset=0),
        )
        nisa.dma_copy(scale_sbuf, scale_in)
        nisa.activation(bf16_sbuf, op=nl.copy, data=fp8_sbuf, scale=scale_sbuf)
        nisa.dma_copy(out[0:PMAX, nl.ds(rep * PMAX, PMAX)], bf16_sbuf)
    return out


@nki.jit
def test_two_step_cast_then_scale(fp8_in, scale_in):
    """
    Two-step approach:
    Step 1: fp8 -> fp32 cast (activation copy, no scale)
    Step 2: fp32 * scale[128P, 1F] -> bf16 (tensor_tensor, VectorE)
    """
    out = nl.ndarray((PMAX, N_REPS * PMAX), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for rep in nl.static_range(N_REPS):
        fp8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        scale_sbuf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        f32_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
        bf16_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            fp8_sbuf,
            fp8_in.ap(pattern=[[PMAX, PMAX], [1, PMAX]], offset=0),
        )
        nisa.dma_copy(scale_sbuf, scale_in)
        # Step 1: fp8 -> fp32 (no scale)
        nisa.activation(f32_sbuf, op=nl.copy, data=fp8_sbuf)
        # Step 2: fp32 * scale -> bf16 via tensor_tensor (VectorE?)
        nisa.tensor_tensor(bf16_sbuf, f32_sbuf, scale_sbuf, nl.multiply)
        nisa.dma_copy(out[0:PMAX, nl.ds(rep * PMAX, PMAX)], bf16_sbuf)
    return out


device = xm.xla_device()
torch.manual_seed(42)

fp8_cpu = (torch.randn(PMAX, PMAX) * 10).clamp(-240, 240).to(torch.float8_e4m3fn)
fp8_xla = fp8_cpu.view(torch.int8).to(device)
scale_xla = (torch.arange(PMAX, dtype=torch.float32).unsqueeze(1) * 0.01 + 1.0).to(device)
xm.mark_step()

print("=" * 60)
print("Test 1: fp8 cast WITHOUT scale (pure dtype conversion)")
print("=" * 60)
try:
    out1 = test_fp8_cast_no_scale[1](fp8_xla)
    xm.mark_step()
    out1_cpu = out1.cpu()
    ref = fp8_cpu.to(torch.float32)
    err = (out1_cpu[0:PMAX, 0:PMAX] - ref).abs().max()
    print(f"  COMPILED+RAN OK. max_err={err:.3e}")
except Exception as e:
    print(f"  FAILED: {str(e)[:300]}")

print()
print("=" * 60)
print("Test 2: fp8 cast WITH scale (original activation approach)")
print("=" * 60)
try:
    out2 = test_fp8_cast_with_scale[1](fp8_xla, scale_xla)
    xm.mark_step()
    out2_cpu = out2.cpu().to(torch.float32)
    scale_cpu = scale_xla.cpu()
    ref2 = fp8_cpu.to(torch.float32) * scale_cpu
    err = (out2_cpu[0:PMAX, 0:PMAX] - ref2).abs().max()
    print(f"  COMPILED+RAN OK. max_err={err:.3e}")
except Exception as e:
    print(f"  FAILED: {str(e)[:300]}")

print()
print("=" * 60)
print("Test 3: two-step fp8->f32 then f32*scale->bf16")
print("=" * 60)
try:
    out3 = test_two_step_cast_then_scale[1](fp8_xla, scale_xla)
    xm.mark_step()
    out3_cpu = out3.cpu().to(torch.float32)
    scale_cpu = scale_xla.cpu()
    ref3 = fp8_cpu.to(torch.float32) * scale_cpu
    err = (out3_cpu[0:PMAX, 0:PMAX] - ref3).abs().max()
    print(f"  COMPILED+RAN OK. max_err={err:.3e}")
except Exception as e:
    print(f"  FAILED: {str(e)[:300]}")
