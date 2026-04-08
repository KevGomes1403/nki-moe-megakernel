"""
Refined tests for fp8 dequant alternatives.
Tests what actually works on VectorE for fp8 dequant.
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

# Test A: tensor_tensor with same-shape fp8 src (no broadcast) — check if fp8 src is accepted
@nki.jit
def test_tt_fp8_same_shape(fp8_in, scale_in):
    """tensor_tensor(bf16_dst[128,128], fp8_src[128,128], fp32_scale[128,128])"""
    fp8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    scale_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    out_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    nisa.dma_copy(fp8_sbuf, fp8_in)
    nisa.dma_copy(scale_sbuf, scale_in)
    nisa.tensor_tensor(out_sbuf, fp8_sbuf, scale_sbuf, nl.multiply)
    nisa.dma_copy(out_hbm, out_sbuf)
    return out_hbm


# Test B: activation(fp8 -> fp32) without scale param - is this VectorE?
@nki.jit
def test_activation_fp8_to_f32(fp8_in):
    """activation(op=copy, fp8->fp32) - is this scalar or vector engine?"""
    fp8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    f32_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.shared_hbm)

    nisa.dma_copy(fp8_sbuf, fp8_in)
    # Pure dtype cast with no scale - which engine?
    nisa.activation(f32_sbuf, op=nl.copy, data=fp8_sbuf)
    nisa.dma_copy(out_hbm, f32_sbuf)
    return out_hbm


# Test C: Can we use nc_transpose to swap fp8 block, then use activation with scale from partition?
# The key insight: if we transpose [128P_h, 128F_j] -> [128P_j, 128F_h],
# then the j dimension becomes partition, and scale[j%128] = scale[p_after_transpose]
# is naturally aligned with partition dim.
@nki.jit
def test_transpose_then_dequant(fp8_in, scale_in):
    """
    Transpose fp8 block so j is in partition, then use activation with scale per partition.
    fp8_in: [128, 128] (h_rows x j_cols)
    scale_in: [128, 1] (scale per j_col, constant over h)
    """
    fp8_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    scale_sbuf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)

    # transpose to [128P_j, 128F_h]
    tp_psum = nl.ndarray((PMAX, PMAX), dtype=nl.float8_e4m3fn, buffer=nl.psum)
    # After transpose: partition=j, scale[p] = scale[j] - matches!
    bf16_tp = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Transpose back
    out_psum = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
    out_sbuf = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    out_hbm = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    nisa.dma_copy(fp8_sbuf, fp8_in)
    nisa.dma_copy(scale_sbuf, scale_in)
    # Transpose: [128P_h, 128F_j] -> psum [128P_j, 128F_h]
    nisa.nc_transpose(tp_psum, fp8_sbuf)
    # Now partition dim = j, scale[p=j] is correct
    # activation(fp8 * scale_per_partition -> bf16)
    nisa.activation(bf16_tp, op=nl.copy, data=tp_psum, scale=scale_sbuf)
    # Transpose back [128P_j, 128F_h] -> psum [128P_h, 128F_j]
    nisa.nc_transpose(out_psum, bf16_tp)
    # SBUF copy
    nisa.activation(out_sbuf, op=nl.copy, data=out_psum)
    nisa.dma_copy(out_hbm, out_sbuf)
    return out_hbm


device = xm.xla_device()
torch.manual_seed(42)

# Create test data
fp8_cpu = (torch.randn(PMAX, PMAX) * 10).clamp(-240, 240).to(torch.float8_e4m3fn)
fp8_xla = fp8_cpu.view(torch.int8).to(device)
scale_1d_cpu = torch.arange(PMAX, dtype=torch.float32) * 0.01 + 1.0
scale_2d_xla = scale_1d_cpu.unsqueeze(1).to(device)  # [128, 1]
scale_2d_full_xla = scale_1d_cpu.unsqueeze(0).expand(PMAX, PMAX).contiguous().to(device)  # [128, 128]
xm.mark_step()

print("Test A: tensor_tensor(bf16_dst, fp8_src[128,128], fp32_scale[128,128]) - same shape...")
try:
    out_a = test_tt_fp8_same_shape[1](fp8_xla, scale_2d_full_xla)
    xm.mark_step()
    out_a_cpu = out_a.cpu()
    print(f"  Test A PASS. out[0,:4]={out_a_cpu[0,:4]}")
except Exception as e:
    print(f"  Test A FAIL: {str(e)[:200]}")

print("Test B: activation(fp8 -> fp32) without scale (pure cast)...")
try:
    out_b = test_activation_fp8_to_f32[1](fp8_xla)
    xm.mark_step()
    out_b_cpu = out_b.cpu()
    ref_b = fp8_cpu.to(torch.float32)
    err = (out_b_cpu - ref_b).abs().max()
    print(f"  Test B PASS. max_err={err:.3e}, out[0,:4]={out_b_cpu[0,:4]}")
except Exception as e:
    print(f"  Test B FAIL: {str(e)[:200]}")

print("Test C: transpose fp8, dequant via activation(scale=per_partition), transpose back...")
try:
    out_c = test_transpose_then_dequant[1](fp8_xla, scale_2d_xla)
    xm.mark_step()
    out_c_cpu = out_c.cpu().to(torch.float32)
    # Expected: fp8[h, j] * scale[j] for each h, j
    expected = fp8_cpu.to(torch.float32) * scale_1d_cpu.unsqueeze(0)
    err = (out_c_cpu - expected).abs().max()
    print(f"  Test C PASS. max_err={err:.3e}")
    print(f"  out[0,:4]={out_c_cpu[0,:4]}, expected[0,:4]={expected[0,:4]}")
except Exception as e:
    print(f"  Test C FAIL: {str(e)[:200]}")
