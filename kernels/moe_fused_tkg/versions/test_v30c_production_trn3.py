"""
Correctness test for _qwen3_moe_sbuf_in_sbuf_out_hoisted — production calling mode on trn3.

Tests the kernel exactly as called in transformer_qwen_multilayer.py:
  - Input pre-loaded into SBUF column-major [PMAX, H_free*T] (matching residual_sb layout)
  - gamma_sb_ready=None  (gamma loaded per-layer from HBM, as in the multilayer)
  - router_w_wide_sb pre-loaded as [PMAX, ROUTER_BATCH, E] SBUF tensor (matching Router_cur)
  - LNC=2 via kernel[2] on real trn3 hardware (no simulator)
  - Accuracy checked against fp32-promoted PyTorch reference: atol=1e-3, rtol=1e-2

Run on trn3:
    python kernels/moe_fused_tkg/versions/test_v30c_production_trn3.py
"""
import os
import sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.allocator import SbufManager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_v30c_hoisted import _qwen3_moe_sbuf_in_sbuf_out_hoisted

# Kernel-level constants (must match kernel_v30c_hoisted.py)
_PMAX         = 128
_H            = 2048
_E            = 128
_I            = 192
_K            = 8
_H_FREE       = _H // _PMAX          # 16
_N_PRGS       = 2
_H_FREE_SHARD = _H_FREE // _N_PRGS   # 8
_H_SHARD      = _H_FREE_SHARD * _PMAX  # 1024
_ROUTER_BATCH = 16

rng    = np.random.default_rng(42)
B      = 1  # single decode token — same scenario as the multilayer (S_tkg=1)
T      = B
device = xm.xla_device()

def make_tensor(shape, scale=0.4):
    arr = rng.standard_normal(shape).astype(np.float32) * scale
    return torch.tensor(arr, dtype=torch.bfloat16).to(device)


# ---------------------------------------------------------------------------
# PyTorch reference (fp32-promoted)
# ---------------------------------------------------------------------------
def pytorch_reference(inp_flat, gamma_flat, router_w, gate_up_w, down_w):
    x     = inp_flat.float()
    rms   = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    normed = x * rms * gamma_flat.float()

    logits    = normed @ router_w.float()
    probs     = torch.softmax(logits, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, _K, dim=-1)
    norm_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

    output = torch.zeros(inp_flat.shape[0], _H, dtype=torch.float32)
    for t in range(inp_flat.shape[0]):
        for k in range(_K):
            e = topk_idx[t, k].item()
            w = norm_weights[t, k].item()
            gu    = gate_up_w[e].float()
            gate  = normed[t] @ gu[:, :_I]
            up    = normed[t] @ gu[:, _I:]
            inter = F.silu(gate) * up
            output[t] += w * (inter @ down_w[e].float())
    return output.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# NKI wrapper — exact production calling convention
#
# Mirrors what transformer_qwen_multilayer._multilayer_body does:
#   1. residual_sb already in SBUF column-major [PMAX, H_free*T]
#      (here we load from HBM into that layout to replicate the state
#       at the MoE call site after the attention residual-add)
#   2. Router_cur pre-loaded into [PMAX, ROUTER_BATCH, E] SBUF (Plan B ring slot)
#   3. gamma_sb_ready=None  — gamma is per-layer HBM weight, not hoisted
# ---------------------------------------------------------------------------
@nki.jit
def v30c_production_wrapper(inp, gamma, router_w, gate_up_w, down_w):
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope(name="wrapper")

    T_local = inp.shape[0]   # B

    # ------------------------------------------------------------------
    # Load inp → SBUF column-major [PMAX, H_free*T]
    # Matches transformer_qwen_multilayer lines ~119-124:
    #   X_flat = X.reshape((BxS * H,))
    #   nisa.dma_copy(dst=residual_sb,
    #                 src=X_flat.ap(pattern=[[1, H0], [H0, H1], [H, BxS]], offset=0))
    # Result: inp_sb[p, h] = inp_flat[h*PMAX + p]
    # ------------------------------------------------------------------
    inp_flat = inp.reshape((T_local * _H,))
    inp_sb = sbm.alloc_stack((_PMAX, _H_FREE * T_local), inp.dtype, buffer=nl.sbuf, name="inp_sb")
    nisa.dma_copy(
        dst=inp_sb,
        src=inp_flat.ap(pattern=[[1, _PMAX], [_PMAX, _H_FREE], [_H, T_local]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )

    # ------------------------------------------------------------------
    # Pre-load router_w → router_w_wide_sb [PMAX, ROUTER_BATCH=16, E=128]
    # Matches transformer_qwen_multilayer lines ~259-266 (Router_cur DMA):
    #   nisa.dma_copy(dst=Router_cur,
    #                 src=router_list[i].ap(pattern=[[E, PMAX],[PMAX*E, ROUTER_BATCH],[1,E]]))
    # Result: router_w_wide_sb[p, rb, e] = router_w[rb*PMAX + p, e]
    # ------------------------------------------------------------------
    router_w_wide_sb = sbm.alloc_stack(
        (_PMAX, _ROUTER_BATCH, _E), inp.dtype, buffer=nl.sbuf, name="router_w_wide_sb"
    )
    nisa.dma_copy(
        dst=router_w_wide_sb,
        src=router_w.ap(
            pattern=[[_E, _PMAX], [_PMAX * _E, _ROUTER_BATCH], [1, _E]],
            offset=0,
        ),
        dge_mode=3,
    )

    # ------------------------------------------------------------------
    # Call sub-function in production mode:
    #   gamma_sb_ready=None      — gamma loaded from HBM inside the kernel
    #   router_w_wide_sb=loaded  — pre-loaded ring slot (Plan B)
    # ------------------------------------------------------------------
    out_sb = _qwen3_moe_sbuf_in_sbuf_out_hoisted(
        inp_sb=inp_sb,
        dtype=inp.dtype,
        T=T_local,
        gamma=gamma,
        router_w=router_w,
        gate_up_w=gate_up_w,
        down_w=down_w,
        sbm=sbm,
        gamma_sb_ready=None,
        router_w_wide_sb=router_w_wide_sb,
    )
    # out_sb: [PMAX, H_free_shard*T] bf16 column-major, one H_shard per LNC core

    # ------------------------------------------------------------------
    # Store out_sb → HBM output [T, H]
    # Each core writes its owned H_shard; both slices are non-overlapping.
    # Mirrors qwen3_moe_fused_tkg_sbuf_io_hoisted store_hbm block.
    # ------------------------------------------------------------------
    prg_id = nl.program_id(axis=0)
    output = nl.ndarray((T_local, _H), dtype=inp.dtype, buffer=nl.shared_hbm)

    sbm.open_scope(name="store_hbm")
    out_row_sb = sbm.alloc_stack((T_local, _H_SHARD), inp.dtype, buffer=nl.sbuf, name="out_row_sb")
    for h1 in nl.static_range(_H_FREE_SHARD):
        tp_psum = nl.ndarray((T_local, _PMAX), dtype=inp.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=tp_psum[0:T_local, 0:_PMAX],
            data=out_sb[0:_PMAX, nl.ds(h1 * T_local, T_local)],
        )
        nisa.activation(
            dst=out_row_sb[0:T_local, nl.ds(h1 * _PMAX, _PMAX)],
            op=nl.copy,
            data=tp_psum[0:T_local, 0:_PMAX],
        )
    nisa.dma_copy(
        dst=output[0:T_local, nl.ds(prg_id * _H_SHARD, _H_SHARD)],
        src=out_row_sb[0:T_local, 0:_H_SHARD],
    )
    sbm.close_scope()  # store_hbm
    sbm.close_scope()  # wrapper

    return output


# ---------------------------------------------------------------------------
# Run and check
# ---------------------------------------------------------------------------

for s in [0.1, 0.2, 0.3, 0.4]:
    inp       = make_tensor((B, 1, _H), scale=s)       # [B, 1, H] bf16
    gamma     = make_tensor((1, _H), scale=s)          # [1, H] bf16
    router_w  = make_tensor((_H, _E), scale=s)         # [H, E] bf16
    gate_up_w = make_tensor((_E, _H, 2*_I), scale=s)   # [E, H, 2*I=384] bf16
    down_w    = make_tensor((_E, _I, _H), scale=s)     # [E, I=192, H] bf16

    print("Computing PyTorch reference...")
    ref_out = pytorch_reference(
        inp.cpu().reshape(B, _H),
        gamma.cpu().squeeze(0),
        router_w.cpu(),
        gate_up_w.cpu(),
        down_w.cpu(),
    )

    print(f"Running v30c production mode on trn3 (LNC=2) with scale {s}")
    out_kernel = v30c_production_wrapper[2](inp, gamma, router_w, gate_up_w, down_w)
    xm.mark_step()

    out_kernel_cpu = out_kernel.cpu().float().numpy()
    ref_out_np     = ref_out.float().numpy()

    max_diff  = np.max(np.abs(ref_out_np - out_kernel_cpu))
    mean_diff = np.mean(np.abs(ref_out_np - out_kernel_cpu))

    print(f"\n--- v30c production mode vs PyTorch fp32 reference ---")
    print(f"kernel output shape : {out_kernel_cpu.shape}")
    print(f"ref    output shape : {ref_out_np.shape}")
    print(f"max_diff            : {max_diff:.6e}")
    print(f"mean_diff           : {mean_diff:.6e}")

    try:
        np.testing.assert_allclose(ref_out_np, out_kernel_cpu, rtol=1e-2, atol=1e-3)
        print(f"\nPASS (scale {s}) — v30c production mode matches PyTorch reference within atol=1e-3, rtol=1e-2")
        print(f"  max_diff={max_diff:.3e}")
    except AssertionError as e:
        print(f"\nFAIL with scale {s} — {e}")
        diff    = np.abs(ref_out_np - out_kernel_cpu)
        failing = np.argwhere(diff > 1e-3 + 1e-2 * np.abs(ref_out_np))
        print(f"Number of failing elements: {len(failing)}")
        if len(failing) > 0:
            idx = failing[0]
            print(f"First failing element idx={tuple(idx)}: "
                f"ref={ref_out_np[tuple(idx)]:.6f}, kernel={out_kernel_cpu[tuple(idx)]:.6f}")
        # sys.exit(1)

