import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
import nki, nki.language as nl, nki.isa as nisa
import torch, torch_xla.core.xla_model as xm, numpy as np

@nki.jit
def test_mxfp8_real(stat_hbm, mov_hbm, stat_scale_hbm, mov_scale_hbm):
    """
    Test nc_matmul_mx with real data.
    stat_hbm:       [P, STAT_F] uint8 (fp8_e4m3fn_x4 reinterpreted)
    mov_hbm:        [P, MOV_F]  uint8 (fp8_e4m3fn_x4 reinterpreted)
    stat_scale_hbm: [STAT_SCALE_P, STAT_F] uint8
    mov_scale_hbm:  [MOV_SCALE_P,  MOV_F]  uint8
    """
    P, STAT_F = stat_hbm.shape
    _, MOV_F  = mov_hbm.shape

    stat_sb    = nl.ndarray((P, STAT_F), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    mov_sb     = nl.ndarray((P, MOV_F),  dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)

    nisa.dma_copy(dst=stat_sb, src=stat_hbm)
    nisa.dma_copy(dst=mov_sb,  src=mov_hbm)

    # Scale shape probe 1: (P/32, STAT_F)
    SP = P // 32
    stat_scale = nl.ndarray((SP, STAT_F), dtype=nl.uint8, buffer=nl.sbuf)
    mov_scale  = nl.ndarray((SP, MOV_F),  dtype=nl.uint8, buffer=nl.sbuf)
    nisa.dma_copy(dst=stat_scale, src=stat_scale_hbm)
    nisa.dma_copy(dst=mov_scale,  src=mov_scale_hbm)

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
def test_quantize_mx(inp_hbm):
    """Test nisa.quantize_mx on trn3.
    inp_hbm: [P, F] bf16
    """
    P, F = inp_hbm.shape

    inp_sb = nl.ndarray((P, F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=inp_sb, src=inp_hbm)

    # quantize_mx output: data same shape as input but fp8_e4m3fn_x4
    # scale: [P, ceil(F/32)] uint8
    SCALE_F = (F + 31) // 32
    out_data  = nl.ndarray((P, F), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    out_scale = nl.ndarray((P, SCALE_F), dtype=nl.uint8, buffer=nl.sbuf)

    nisa.quantize_mx(dst_data=out_data, dst_scale=out_scale, src=inp_sb)

    # Copy out as uint8 for inspection
    out_data_hbm  = nl.ndarray((P, F), dtype=nl.float8_e4m3fn_x4, buffer=nl.shared_hbm)
    out_scale_hbm = nl.ndarray((P, SCALE_F), dtype=nl.uint8, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out_data_hbm, src=out_data)
    nisa.dma_copy(dst=out_scale_hbm, src=out_scale)
    return out_data_hbm, out_scale_hbm


device = xm.xla_device()

P = 128
STAT_F = 2
MOV_F = 1
SP = P // 32  # 4

# Build fp8 data: stat[p, f] = 1.0 in fp8_e4m3fn = 0x3C
# For x4 packing: 4 fp8 values in one uint8[4]-sized element?
# Actually float8_e4m3fn_x4 packs 4 fp8 bytes. Since our tensor is [P, STAT_F],
# each "element" in the x4 type represents 4 fp8 values along the free dimension?
# Or along the partition dimension? Let's test with all-ones encoding.

# fp8_e4m3fn(1.0) = 0x3C
fp8_one = 0x3C

stat_np = np.full((P, STAT_F), fp8_one, dtype=np.uint8)
mov_np  = np.full((P, MOV_F),  fp8_one, dtype=np.uint8)

# Scale: 127 means 2^(127-127) = 1.0
stat_scale_np = np.full((SP, STAT_F), 127, dtype=np.uint8)
mov_scale_np  = np.full((SP, MOV_F),  127, dtype=np.uint8)

stat_t = torch.from_numpy(stat_np).to(device)
mov_t  = torch.from_numpy(mov_np).to(device)
stat_sc = torch.from_numpy(stat_scale_np).to(device)
mov_sc  = torch.from_numpy(mov_scale_np).to(device)
xm.mark_step()

print("=== Test nc_matmul_mx with all-ones fp8 data, scale=127 ===")
print(f"stat shape: {stat_t.shape}, mov shape: {mov_t.shape}")
print(f"scale shapes: {stat_sc.shape}, {mov_sc.shape}")
try:
    result = test_mxfp8_real(stat_t, mov_t, stat_sc, mov_sc)
    xm.mark_step()
    if isinstance(result, (list, tuple)):
        result = result[0]
    r_cpu = result.cpu().float().numpy()
    print(f"PASS! output shape={r_cpu.shape}")
    print(f"values:\n{r_cpu}")
    # Expected: P partitions each contribute 1.0 * 1.0 = 1.0, summed = 128
    # But nc_matmul in NKI: stationary [P, STAT_F], moving [P, MOV_F]
    # The contraction is over P (partition dim).
    # With x4 packing: each element = 4 fp8 values.
    # So effective: 4 * P = 512 elements contracted
    # With value 1.0 each: sum = 512?
    print(f"Expected ~128 or ~512 if x4 packing multiplies count")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n=== Test quantize_mx on trn3 ===")
# inp: [128P, 16F] bf16 with value 2.0
inp_np = np.full((P, 16), 2.0, dtype=np.float32)
inp_t = torch.from_numpy(inp_np).to(torch.bfloat16).to(device)
xm.mark_step()

try:
    qdata, qscale = test_quantize_mx(inp_t)
    xm.mark_step()
    if isinstance(qdata, (list, tuple)):
        qdata, qscale = qdata[0], qdata[1] if len(qdata) > 1 else qscale
    qd_cpu = qdata.cpu()
    qs_cpu = qscale.cpu()
    print(f"quantize_mx PASS!")
    print(f"data shape={qd_cpu.shape}, scale shape={qs_cpu.shape}")
    print(f"scale values (first row): {qs_cpu[0].numpy()}")
    # Scale of 2.0 in fp8_e4m3: scale = log2(2.0) + 127 = 1 + 127 = 128
    # So expect scale ~ 128
except Exception as e:
    print(f"FAIL: {e}")
    import traceback; traceback.print_exc()
