"""
Test if nc_matmul can accept fp8_e4m3fn inputs directly (no dequant needed).
Also test if tensor_tensor with full [128,128] scale works on fp32.
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


# Test A: nc_matmul with fp8 stationary weight
@nki.jit
def test_fp8_matmul(fp8_weights, bf16_input):
    """
    Try nc_matmul with fp8 stationary and bf16 moving.
    fp8_weights: [128P, 128F] fp8 (passed as int8)
    bf16_input:  [128P, 1F] bf16
    """
    fp8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    inp_sbuf = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum)
    out_sbuf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    nisa.dma_copy(fp8_sbuf, fp8_weights)
    nisa.dma_copy(inp_sbuf, bf16_input)
    nisa.nc_matmul(dst=psum, stationary=fp8_sbuf, moving=inp_sbuf)
    nisa.activation(out_sbuf, op=nl.copy, data=psum)
    nisa.dma_copy(out_hbm, out_sbuf)
    return out_hbm


# Test B: tensor_tensor(f32 * f32 full [128,128]) - confirm VectorE works
@nki.jit
def test_tt_f32_full(f32_in, f32_scale_full):
    """
    tensor_tensor(bf16, fp32[128,128], fp32[128,128]) - full shapes, no broadcast
    """
    f32_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    scale_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    bf16_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    nisa.dma_copy(f32_sbuf, f32_in)
    nisa.dma_copy(scale_sbuf, f32_scale_full)
    nisa.tensor_tensor(bf16_sbuf, f32_sbuf, scale_sbuf, nl.multiply)
    nisa.dma_copy(out_hbm, bf16_sbuf)
    return out_hbm


# Test C: Can we do fp8 matmul with scale_factor per-column applied before matmul?
# What if we use nc_matmul with fp8 weights directly and apply scale to output?
# The matmul: [128P, 128F_j] @ [128F_j, 1] -> [128P, 1]
# If we want: sum_j fp8[p,j] * x[j] * scale[j]
# vs proper dequant: (fp8[p,j] * scale[j]) * x[j]  (same by associativity)
# We can post-multiply scale to the output only if scale is scalar (same for all j).
# But scale[j] varies per j — can't post-multiply.

# Test D: two-step with full scale broadcast
@nki.jit
def test_two_step_full_scale(fp8_in, scale_full):
    """
    Step 1: fp8 -> fp32 (activation, scalar engine, no scale param)
    Step 2: fp32 * fp32[128,128] -> bf16 (tensor_tensor, vector engine)
    scale_full: [128, 128] fp32, fully broadcast
    """
    fp8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    scale_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    f32_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    scaled_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    nisa.dma_copy(fp8_sbuf, fp8_in)
    nisa.dma_copy(scale_sbuf, scale_full)
    # Step 1: fp8 -> fp32 (NO scale param)
    nisa.activation(f32_sbuf, op=nl.copy, data=fp8_sbuf)
    # Step 2: full shape multiply (VectorE?)
    nisa.tensor_tensor(scaled_sbuf, f32_sbuf, scale_sbuf, nl.multiply)
    nisa.dma_copy(out_hbm, scaled_sbuf)
    return out_hbm


# Test A2: nc_matmul with nl.float8_e4m3 (no fn suffix) as SBUF buffer dtype
@nki.jit
def test_fp8_matmul_e4m3(fp8_weights, bf16_input):
    """
    Same as test_fp8_matmul but with nl.float8_e4m3 (no fn suffix).
    """
    fp8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    inp_sbuf = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum)
    out_sbuf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    nisa.dma_copy(fp8_sbuf, fp8_weights)
    nisa.dma_copy(inp_sbuf, bf16_input)
    nisa.nc_matmul(dst=psum, stationary=fp8_sbuf, moving=inp_sbuf)
    nisa.activation(out_sbuf, op=nl.copy, data=psum)
    nisa.dma_copy(out_hbm, out_sbuf)
    return out_hbm


# Test A3: int8 SBUF with ap() reinterpret as fp8e4m3fn in matmul
@nki.jit
def test_fp8_matmul_int8_ap(fp8_weights, bf16_input):
    """
    Declare SBUF as int8, reinterpret as float8_e4m3fn via .ap() in matmul.
    """
    int8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.int8, buffer=nl.sbuf)
    inp_sbuf = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum)
    out_sbuf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    nisa.dma_copy(int8_sbuf, fp8_weights)
    nisa.dma_copy(inp_sbuf, bf16_input)
    nisa.nc_matmul(
        dst=psum,
        stationary=int8_sbuf.ap([[1, PMAX], [PMAX, PMAX]], dtype=nl.float8_e4m3fn),
        moving=inp_sbuf,
    )
    nisa.activation(out_sbuf, op=nl.copy, data=psum)
    nisa.dma_copy(out_hbm, out_sbuf)
    return out_hbm


# Test A4: int8 SBUF with ap() reinterpret as float8_e4m3 (no fn) in matmul
@nki.jit
def test_fp8_matmul_int8_ap_e4m3(fp8_weights, bf16_input):
    """
    Declare SBUF as int8, reinterpret as float8_e4m3 via .ap() in matmul.
    """
    int8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.int8, buffer=nl.sbuf)
    inp_sbuf = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum)
    out_sbuf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    nisa.dma_copy(int8_sbuf, fp8_weights)
    nisa.dma_copy(inp_sbuf, bf16_input)
    nisa.nc_matmul(
        dst=psum,
        stationary=int8_sbuf.ap([[1, PMAX], [PMAX, PMAX]], dtype=nl.float8_e4m3),
        moving=inp_sbuf,
    )
    nisa.activation(out_sbuf, op=nl.copy, data=psum)
    nisa.dma_copy(out_hbm, out_sbuf)
    return out_hbm


# Test E: DMA fp8→bf16 conversion via dma_copy src dtype reinterpret
@nki.jit
def test_dma_fp8_to_bf16(fp8_weights_int8):
    """Test if DMA can convert fp8 to bf16 during copy."""
    bf16_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    nisa.dma_copy(
        dst=bf16_sbuf,
        src=fp8_weights_int8.ap(
            pattern=[[1, PMAX], [PMAX, PMAX]],
            dtype=nl.float8_e4m3fn,
            offset=0,
        ),
    )
    nisa.dma_copy(out_hbm, bf16_sbuf)
    return out_hbm


# Test F: nl.load fp8 with dtype=bfloat16 conversion
@nki.jit
def test_load_fp8_to_bf16(fp8_weights_int8):
    """Test nl.load with fp8 src reinterpreted and loaded as bf16."""
    out_hbm = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    tile = nl.load(
        fp8_weights_int8.ap(pattern=[[1, PMAX], [PMAX, PMAX]], dtype=nl.float8_e4m3fn),
        dtype=nl.bfloat16,
    )
    nl.store(out_hbm, tile)
    return out_hbm


device = xm.xla_device()
torch.manual_seed(42)

fp8_cpu = (torch.randn(PMAX, PMAX) * 10).clamp(-240, 240).to(torch.float8_e4m3fn)
fp8_xla = fp8_cpu.view(torch.int8).to(device)

f32_in = torch.randn(PMAX, PMAX, dtype=torch.float32).to(device)
bf16_in = torch.randn(PMAX, 1, dtype=torch.bfloat16).to(device)
scale_col = (torch.arange(PMAX, dtype=torch.float32) * 0.01 + 1.0).unsqueeze(1).to(device)  # [128, 1]
scale_full = scale_col.expand(PMAX, PMAX).contiguous().to(device)  # [128, 128]
xm.mark_step()

print("Test A: nc_matmul with fp8_e4m3fn stationary weight (original)...")
try:
    out_a = test_fp8_matmul[1](fp8_xla, bf16_in)
    xm.mark_step()
    print(f"  Test A PASS: fp8 matmul works! out[0]={out_a.cpu()[0,0]:.4f}")
except Exception as e:
    print(f"  Test A FAIL: {str(e)[:300]}")

print("Test A2: nc_matmul with fp8_e4m3 (no fn) stationary weight...")
try:
    out_a2 = test_fp8_matmul_e4m3[1](fp8_xla, bf16_in)
    xm.mark_step()
    print(f"  Test A2 PASS: fp8_e4m3 matmul works! out[0]={out_a2.cpu()[0,0]:.4f}")
except Exception as e:
    print(f"  Test A2 FAIL: {str(e)[:300]}")

print("Test A3: nc_matmul with int8 SBUF + .ap(float8_e4m3fn)...")
try:
    out_a3 = test_fp8_matmul_int8_ap[1](fp8_xla, bf16_in)
    xm.mark_step()
    print(f"  Test A3 PASS: int8 SBUF + ap(fp8e4m3fn) works! out[0]={out_a3.cpu()[0,0]:.4f}")
except Exception as e:
    print(f"  Test A3 FAIL: {str(e)[:300]}")

print("Test A4: nc_matmul with int8 SBUF + .ap(float8_e4m3 no fn)...")
try:
    out_a4 = test_fp8_matmul_int8_ap_e4m3[1](fp8_xla, bf16_in)
    xm.mark_step()
    print(f"  Test A4 PASS: int8 SBUF + ap(fp8e4m3) works! out[0]={out_a4.cpu()[0,0]:.4f}")
except Exception as e:
    print(f"  Test A4 FAIL: {str(e)[:300]}")

print("Test B: tensor_tensor(f32[128,128] * f32[128,128] -> bf16)...")
try:
    out_b = test_tt_f32_full[1](f32_in, scale_full)
    xm.mark_step()
    out_b_cpu = out_b.cpu().to(torch.float32)
    f32_in_cpu = f32_in.cpu()
    scale_full_cpu = scale_full.cpu()
    ref = f32_in_cpu * scale_full_cpu
    err = (out_b_cpu - ref).abs().max()
    print(f"  Test B PASS. max_err={err:.3e} (expected small due to bf16 rounding)")
except Exception as e:
    print(f"  Test B FAIL: {str(e)[:200]}")

print("Test D: two-step fp8->f32 then full scale multiply...")
try:
    out_d = test_two_step_full_scale[1](fp8_xla, scale_full)
    xm.mark_step()
    out_d_cpu = out_d.cpu().to(torch.float32)
    scale_1d_cpu = scale_col.cpu().squeeze()
    ref_d = fp8_cpu.to(torch.float32) * scale_1d_cpu.unsqueeze(1)
    err = (out_d_cpu - ref_d).abs().max()
    print(f"  Test D PASS. max_err={err:.3e}")
    print(f"  out_d[0,:4]={out_d_cpu[0,:4]}, ref[0,:4]={ref_d[0,:4]}")
except Exception as e:
    print(f"  Test D FAIL: {str(e)[:200]}")

print("Test E: DMA fp8→bf16 conversion via dma_copy with reinterpret...")
try:
    out_e = test_dma_fp8_to_bf16[1](fp8_xla)
    xm.mark_step()
    out_e_cpu = out_e.cpu().to(torch.float32)
    ref_e = fp8_cpu.to(torch.float32)
    err = (out_e_cpu - ref_e).abs().max()
    print(f"  Test E PASS. max_err={err:.3e}")
except Exception as e:
    print(f"  Test E FAIL: {str(e)[:300]}")

print("Test F: nl.load fp8→bf16 conversion...")
try:
    out_f = test_load_fp8_to_bf16[1](fp8_xla)
    xm.mark_step()
    out_f_cpu = out_f.cpu().to(torch.float32)
    ref_f = fp8_cpu.to(torch.float32)
    err = (out_f_cpu - ref_f).abs().max()
    print(f"  Test F PASS. max_err={err:.3e}")
except Exception as e:
    print(f"  Test F FAIL: {str(e)[:300]}")
