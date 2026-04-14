import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
import nki, nki.language as nl, nki.isa as nisa
import torch, torch_xla.core.xla_model as xm, numpy as np

@nki.jit
def test_mxfp8(dummy):
    """Test nc_matmul_mx with MXFP8 (float8_e4m3fn_x4).

    nc_matmul convention:
      stationary [P, STAT_F] dtype float8_e4m3fn_x4 → P is partition, STAT_F*4 = output neurons
      moving     [P, MOV_F]  dtype float8_e4m3fn_x4 → P is partition, MOV_F*4 = tokens
      stationary_scale [P, STAT_SCALE_F] uint8
      moving_scale     [P, MOV_SCALE_F]  uint8
      dst [STAT_F, MOV_F] float32 — PSUM output
    For MXFP8 (float8_e4m3fn_x4), x4 packs 4 fp8 values per element.
    Block size for MXFP8 is 32 fp8 values = 8 x4-packed elements.
    """
    P = 128
    STAT_F = 2   # 2*4 = 8 output neurons
    MOV_F = 1    # 1*4 = 4 tokens

    # 1 scale per 32-element block = 1 scale per 8 x4 elements
    # STAT_SCALE_F = ceil(STAT_F * 4 / 32) = ceil(8/32) = 1
    # Actually scale is per-block of 32 *partition* elements
    # Based on MXFP spec: block scale covers 32 elements in the contraction dim
    # With P=128 partitions each holding 1 element of the x4-packed vector,
    # scale is per 32-partition block: ceil(128/32) = 4 blocks
    STAT_SCALE_P = 4   # P/32 = 128/32 = 4 scale values per output neuron group
    MOV_SCALE_P  = 4   # same

    # Try float8_e4m3fn_x4 (MXFP8 e4m3)
    stat_sb = nl.ndarray((P, STAT_F), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    mov_sb  = nl.ndarray((P, MOV_F),  dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    # Scale: [P/32, STAT_F] or [STAT_F, P/32]? Try [STAT_SCALE_P, STAT_F]:
    stat_scale = nl.ndarray((STAT_SCALE_P, STAT_F), dtype=nl.uint8, buffer=nl.sbuf)
    mov_scale  = nl.ndarray((MOV_SCALE_P,  MOV_F),  dtype=nl.uint8, buffer=nl.sbuf)

    nisa.memset(stat_sb, value=0)
    nisa.memset(mov_sb, value=0)
    nisa.memset(stat_scale, value=127)  # 2^(127-127) = 1.0 scale
    nisa.memset(mov_scale, value=127)

    dst = nl.ndarray((STAT_F, MOV_F), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul_mx(
        dst=dst, stationary=stat_sb, moving=mov_sb,
        stationary_scale=stat_scale, moving_scale=mov_scale,
    )

    res_sb = nl.ndarray((STAT_F, MOV_F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(res_sb, op=nl.copy, data=dst)
    out = nl.ndarray((STAT_F, MOV_F), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=res_sb)
    return out


@nki.jit
def test_mxfp8_v2(dummy):
    """Alternative: scale shape matches stationary shape (P, STAT_F) same as data."""
    P = 128
    STAT_F = 2
    MOV_F = 1

    stat_sb    = nl.ndarray((P, STAT_F), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    mov_sb     = nl.ndarray((P, MOV_F),  dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    stat_scale = nl.ndarray((P, STAT_F), dtype=nl.uint8, buffer=nl.sbuf)
    mov_scale  = nl.ndarray((P, MOV_F),  dtype=nl.uint8, buffer=nl.sbuf)

    nisa.memset(stat_sb, value=0)
    nisa.memset(mov_sb, value=0)
    nisa.memset(stat_scale, value=127)
    nisa.memset(mov_scale, value=127)

    dst = nl.ndarray((STAT_F, MOV_F), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul_mx(
        dst=dst, stationary=stat_sb, moving=mov_sb,
        stationary_scale=stat_scale, moving_scale=mov_scale,
    )

    res_sb = nl.ndarray((STAT_F, MOV_F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(res_sb, op=nl.copy, data=dst)
    out = nl.ndarray((STAT_F, MOV_F), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=res_sb)
    return out


device = xm.xla_device()
dummy = torch.zeros(1, 1, dtype=torch.float32).to(device)

print("=== Probe 1: scale shape (P/32, STAT_F) ===")
try:
    out = test_mxfp8(dummy)
    xm.mark_step()
    if isinstance(out, (list, tuple)):
        out = out[0]
    print(f"PASS! output shape={out.shape}, values=\n{out.cpu()}")
except Exception as e:
    print(f"FAIL: {e}")

print("\n=== Probe 2: scale shape (P, STAT_F) same as data ===")
try:
    out2 = test_mxfp8_v2(dummy)
    xm.mark_step()
    if isinstance(out2, (list, tuple)):
        out2 = out2[0]
    print(f"PASS! output shape={out2.shape}, values=\n{out2.cpu()}")
except Exception as e:
    print(f"FAIL: {e}")
