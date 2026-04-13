"""
Phase 2: Minimal MXFP4 matmul test using nc_matmul_mx with float4_e2m1fn_x4.
"""
import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import nki
import nki.language as nl
import nki.isa as nisa
import numpy as np
import torch
import torch_xla.core.xla_model as xm


@nki.jit
def test_fp4_matmul_v1(dummy_in):
    """
    Minimal test: 32P contraction, 2 x4 output neurons, 1 x4 tokens.
    dummy_in: [1, 1] just to satisfy 'must have at least 1 argument'
    """
    P = 32
    STAT_F = 2   # x4 packed → 8 actual output neurons
    MOV_F = 1    # x4 packed → 4 actual tokens

    stat_sb = nl.ndarray((P, STAT_F), dtype=nl.float4_e2m1fn_x4, buffer=nl.sbuf)
    mov_sb  = nl.ndarray((P, MOV_F), dtype=nl.float4_e2m1fn_x4, buffer=nl.sbuf)
    stat_scale_sb = nl.ndarray((P, STAT_F), dtype=nl.uint8, buffer=nl.sbuf)
    mov_scale_sb  = nl.ndarray((P, MOV_F),  dtype=nl.uint8, buffer=nl.sbuf)

    nisa.memset(stat_sb, value=0)
    nisa.memset(mov_sb, value=0)
    nisa.memset(stat_scale_sb, value=127)
    nisa.memset(mov_scale_sb, value=127)

    dst_psum = nl.ndarray((STAT_F, MOV_F), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul_mx(
        dst=dst_psum,
        stationary=stat_sb,
        moving=mov_sb,
        stationary_scale=stat_scale_sb,
        moving_scale=mov_scale_sb,
    )

    result_sb = nl.ndarray((STAT_F, MOV_F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(result_sb, op=nl.copy, data=dst_psum)

    out_hbm = nl.ndarray((STAT_F, MOV_F), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out_hbm, src=result_sb)
    return out_hbm


def run_test():
    device = xm.xla_device()
    dummy = torch.zeros(1, 1, dtype=torch.float32).to(device)

    print("Test v1: stationary [32P, 2F_x4], moving [32P, 1F_x4], dst [2, 1] float32")
    try:
        result = test_fp4_matmul_v1(dummy)
        xm.mark_step()
        if isinstance(result, (list, tuple)):
            result = result[0]
        print(f"  Shape: {result.shape}, values:\n{result.cpu()}")
        print("  PASS: FP4 matmul (v1) compiled and ran!")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


if __name__ == "__main__":
    run_test()
