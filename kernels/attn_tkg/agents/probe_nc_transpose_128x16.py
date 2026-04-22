"""
Phase 3 STEP 1a.1 — nc_transpose shape probe.

Question: does trn3's nc_transpose support [128, 16] bf16 SBUF → [16, 128] bf16 PSUM?
(Required by wrapper fix option β.) Run with known fixed-seed values; verify
transpose is correct numerically.

Triple-check pattern mandated by user:
  1. zero-check: output absmean > 1e-4
  2. range-check: output max ≈ src max
  3. allclose(rtol=1e-3, atol=1e-3) vs numpy reference
"""
import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import nki
import nki.language as nl
import nki.isa as nisa

P, F = 128, 16


@nki.jit
def probe(src_hbm):
    src_sb = nl.ndarray((P, F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=src_sb, src=src_hbm, dge_mode=3)
    dst_psum = nl.ndarray((F, P), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=dst_psum, data=src_sb)
    dst_sb = nl.ndarray((F, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=dst_sb, src=dst_psum)
    out = nl.ndarray((F, P), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nl.store(out, dst_sb)
    return out


if __name__ == "__main__":
    d = xm.xla_device()
    torch.manual_seed(42)
    src = torch.randn(P, F, dtype=torch.bfloat16) * 0.1   # small BF16 values

    try:
        out_t = probe(src.to(d))
        got = out_t.cpu().float().numpy()
    except Exception as e:
        print(f"[nc_transpose PROBE] FAIL (compile/runtime): {type(e).__name__}")
        print(str(e)[:600])
        raise SystemExit(1)

    ref = src.float().numpy().T  # [F, P]

    print(f"src shape={tuple(src.shape)} range=[{src.float().min():.4f}, {src.float().max():.4f}]")
    print(f"got shape={got.shape} absmean={np.abs(got).mean():.4e}")
    print(f"ref shape={ref.shape} absmean={np.abs(ref).mean():.4e}")

    # 1. zero-check
    if np.abs(got).mean() < 1e-4:
        raise SystemExit("[nc_transpose PROBE] HARD FAIL — output is zero")
    # 2. range-check
    got_max, ref_max = float(np.abs(got).max()), float(np.abs(ref).max())
    if got_max < 0.5 * ref_max:
        raise SystemExit(f"[nc_transpose PROBE] HARD FAIL — got max {got_max:.4f} << ref max {ref_max:.4f}")
    # 3. allclose
    max_err = np.abs(got - ref).max()
    print(f"max_abs_err = {max_err:.6e}")
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)
    print("[nc_transpose PROBE] PASS — option β is viable")
