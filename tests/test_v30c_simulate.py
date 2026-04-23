"""
test_v30c_simulate.py

CPU-simulator correctness test for the v30c MoE kernel.

Reference : RefMoEModule compiled with build_module (Trainium hardware).
Kernel    : _v30c_moe_raw executed via nki.simulate (CPU, no hardware needed for
            the kernel path).  The simulator produces expected_outputs which are
            then validated against the hardware reference via validate_accuracy.

Runs the same 512 inputs (4 scales x 2 distributions, seeded at 200+i) as
test_v30c_vs_nxdi_trn3.py, but for a single weight scale and using nki.simulate
instead of a compiled KernelMoEModule.

Precision: bf16 (NKI_PRECISE_FP=1 forces ml_dtypes.bfloat16 storage in simulator).
Platform : trn3 (NEURON_PLATFORM_TARGET_OVERRIDE=trn3), LNC=2.

Run:
    NKI_PRECISE_FP=1 python tests/test_v30c_simulate.py
    NKI_PRECISE_FP=1 python tests/test_v30c_simulate.py --weight-scale 2.0 --n-samples 16
"""

import os
import sys

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NKI_PRECISE_FP"] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOE_VERSIONS = os.path.join(_HERE, "..", "kernels", "moe_fused_tkg", "versions")
sys.path.insert(0, "/home/ubuntu/nki-moe")
sys.path.insert(0, _MOE_VERSIONS)

import argparse
import tempfile

import ml_dtypes
import numpy as np
import torch
import torch.nn as nn
import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.allocator import SbufManager

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

from qwen import Qwen3MoeInferenceConfig

# Kernel
import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.tensor_view import TensorView
from nkilib.core.utils.allocator import SbufManager

# Hardware constants
_PMAX = 128       # partition dimension max
_PSUM_FREE = 512  # PSUM free-dimension max on trn2

# Qwen3-30B-A3B at TP=4 fixed dims
_H = 2048    # hidden dim
_E = 128     # num experts
_K = 8       # top-K experts
_I = 192     # actual intermediate dim per TP rank
_I0 = 128    # first tile  (full 128 rows)
_I1 = 64     # second tile (partial: 64 valid rows, 64 zero-padded)
_I_TILES = 2 # two I-dimension tiles
_EPS = 1e-6

# Flat gate+up combined width
_GU_FLAT = 2 * _I   # = 384

# LNC=2 H-sharding constants
_N_PRGS = 2
_H_FREE = _H // _PMAX             # = 16
_H_FREE_SHARD = _H_FREE // _N_PRGS   # = 8
_H_SHARD = _H_FREE_SHARD * _PMAX     # = 1024

# Router DMA batching
_ROUTER_BATCH = 16

# 2-wave constants
_K_WAVE = 4  # experts per wave

# Debug trace flag (module-level so NKI tracer resolves it at trace time, not runtime).
# Enable by setting `KERN_TRACE=1` before importing this module. Tracer evaluates the
# boolean once and constant-folds — `os.environ.get` inside the jit body breaks compile
# because the tracer tries to lower `os` as an NKI class.
_KERN_TRACE = os.environ.get("KERN_TRACE", "0") == "1"


def _qwen3_moe_sbuf_in_sbuf_out_hoisted(
    inp_sb,               # [PMAX=128, H_free*T] bf16 — already in SBUF (no HBM load needed)
    dtype,                # bf16 — explicit since inp.dtype is no longer available
    T,                    # int — number of tokens
    gamma,                # [1, H=2048] bf16 — HBM
    router_w,             # [H=2048, E=128] bf16 — HBM
    gate_up_w,            # [E=128, H=2048, 384] bf16 — HBM
    down_w,               # [E=128, 192, H=2048] bf16 — HBM
    sbm=None,             # required: SbufManager instance
    gamma_sb_ready=None,  # [PMAX, H_free=16] bf16 SBUF — pre-loaded & pre-transposed gamma
                          # When supplied, skip gamma_flat_sb alloc+DMA,
                          # gamma_trans_psum alloc+nc_transpose,
                          # gamma_sb alloc+activation copy.
    router_w_wide_sb=None,  # [PMAX, _ROUTER_BATCH=16, E=128] bf16 SBUF — replaces wide router DMA
                            # When supplied, skip the alloc + nisa.dma_copy in h_chunk loop.
    debug_rmsnorm_out=None,  # [T, H=2048] bf16 HBM — when provided, DMA rmsnorm_normed_bf16 here
):
    """
    Fused RMSNorm + Router + TopK(K=8) + Expert MLP for Qwen3 MoE TKG.
    Sub-kernel — no @nki.jit decorator. Called from inside a jitted function.
    inp_sb: [PMAX, H_free*T] bf16 already in SBUF.
    Returns: out_sb [PMAX=128, H_free_shard*T=8] bf16 in SBUF — column-major, partition-first.
             caller must consume before any further sbm.alloc_stack

    New kwargs (all default None):
      gamma_sb_ready   : pre-loaded [PMAX, H_free] bf16 SBUF tensor — skips gamma DMA chain
      router_w_wide_sb : pre-loaded [PMAX, ROUTER_BATCH, E] bf16 SBUF — skips router wide DMA
    """
    B = T

    H = _H
    E = _E
    K = _K
    I = _I
    I0 = _I0
    I1 = _I1
    H_free = _H_FREE
    H_free_shard = _H_FREE_SHARD
    H_shard = _H_SHARD
    I_tiles = _I_TILES

    prg_id = nl.program_id(axis=0)

    # -----------------------------------------------------------------------
    # Stage 1: RMSNorm
    # -----------------------------------------------------------------------
    # heap: long-lived, freed after stage 4
    rmsnorm_normed_bf16 = sbm.alloc_heap((_PMAX, H_free * T), dtype, buffer=nl.sbuf, name="rmsnorm_normed_bf16")
    # heap: fp32 normed output — must survive close_scope(rmsnorm) for gamma multiply
    rmsnorm_normed = sbm.alloc_heap((_PMAX, H_free * T), nl.float32, buffer=nl.sbuf, name="rmsnorm_normed")

    sbm.open_scope(name="rmsnorm")

    # inp_sb is already [PMAX, H_free*T] bf16 in SBUF — no load needed
    rmsnorm_out = inp_sb  # already [PMAX, H_free*T] bf16 in SBUF

    # --- gamma load (skipped when gamma_sb_ready is supplied) ---
    if gamma_sb_ready is None:
        gamma_1d = gamma.reshape((H,))
        gamma_1d_hbm_reshaped = gamma_1d.reshape((H_free, _PMAX))
        gamma_flat_sb = sbm.alloc_stack((H_free, _PMAX), gamma.dtype, buffer=nl.sbuf, name="gamma_flat_sb")
        nisa.dma_copy(dst=gamma_flat_sb, src=gamma_1d_hbm_reshaped, dge_mode=3)
        gamma_trans_psum = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=gamma_trans_psum, data=gamma_flat_sb)
        gamma_sb = sbm.alloc_stack((_PMAX, H_free), gamma.dtype, buffer=nl.sbuf, name="gamma_sb")
        nisa.activation(gamma_sb[...], op=nl.copy, data=gamma_trans_psum[...])
    else:
        gamma_sb = gamma_sb_ready  # caller-owned — do NOT free

    rmsnorm_sq = sbm.alloc_stack((_PMAX, H_free * T), nl.float32, buffer=nl.sbuf, name="rmsnorm_sq")
    nisa.activation(rmsnorm_sq[...], op=nl.square, data=rmsnorm_out[...])

    rmsnorm_reduced = sbm.alloc_stack((_PMAX, T), nl.float32, buffer=nl.sbuf, name="rmsnorm_reduced")
    nisa.tensor_reduce(rmsnorm_reduced[0:_PMAX, 0:T], nl.add, rmsnorm_sq[0:_PMAX, 0:H_free * T], axis=1)

    sum_reduced_sb = sbm.alloc_stack((1, T), nl.float32, buffer=nl.sbuf, name="sum_reduced_sb")
    nisa.tensor_partition_reduce(dst=sum_reduced_sb[0:1, 0:T], data=rmsnorm_reduced[0:_PMAX, 0:T], op=nl.add)

    norm_sum_sb = sbm.alloc_stack((_PMAX, T), nl.float32, buffer=nl.sbuf, name="norm_sum_sb")
    nisa.tensor_copy(dst=norm_sum_sb[0:1, 0:T], src=sum_reduced_sb[0:1, 0:T])
    for g in nl.static_range(4):
        nisa.nc_stream_shuffle(
            dst=norm_sum_sb[nl.ds(g * 32, 32), 0:T],
            src=norm_sum_sb[0:1, 0:T],
            shuffle_mask=[0] * 32,
        )

    eps_sb = sbm.alloc_stack((_PMAX, 1), nl.float32, buffer=nl.sbuf, name="eps_sb")
    nisa.memset(eps_sb, value=_EPS)
    norm_factor_sb = sbm.alloc_stack((_PMAX, T), nl.float32, buffer=nl.sbuf, name="norm_factor_sb")
    nisa.activation(
        norm_factor_sb[0:_PMAX, 0:T],
        op=nl.rsqrt,
        data=norm_sum_sb[0:_PMAX, 0:T],
        scale=1.0 / H,
        bias=eps_sb[0:_PMAX, :],
    )

    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=2).broadcast(dim=2, size=H_free)
    x_scaled = sbm.alloc_stack((_PMAX, H_free * T), nl.float32, buffer=nl.sbuf, name="x_scaled")
    nisa.tensor_tensor(x_scaled[...], rmsnorm_out[...], norm_factor_bcast.get_view(), nl.multiply)
    # rmsnorm_normed is a heap tensor (allocated before this scope) — just write into it
    nisa.tensor_tensor(rmsnorm_normed[...], x_scaled[...], gamma_sb[...], nl.multiply)

    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=rmsnorm_normed[...])

    if _KERN_TRACE and prg_id == 0:
        # rmsnorm_normed_bf16 is [PMAX, H_free*T] col-major. First 10 values of H correspond
        # to partitions 0..9, index h1=0 t=0 (for T=1 which is our case).
        nl.device_print("[KERN] rmsnorm_out[:10] bf16 (first 10 partitions, h1=0)", rmsnorm_normed_bf16[0:10, 0:1])

    sbm.close_scope()  # frees all rmsnorm transient sbuf tensors (not gamma_sb_ready — caller-owned)

    # Debug: DMA rmsnorm_normed_bf16 [PMAX, H_free*T] → HBM [T, H] (prg_id 0 only).
    # Both LNC cores carry the full H at this stage (sharding happens later), so
    # writing from prg_id==0 alone is correct.
    if debug_rmsnorm_out is not None:
        if prg_id == 0:
            debug_flat = debug_rmsnorm_out.reshape((B, H))
            for _t in nl.static_range(B):
                for _h1 in nl.static_range(H_free):
                    _dbg_psum = nl.ndarray((1, _PMAX), dtype=dtype, buffer=nl.psum)
                    nisa.nc_transpose(
                        _dbg_psum,
                        rmsnorm_normed_bf16[0:_PMAX, _h1 * B + _t : _h1 * B + _t + 1],
                    )
                    _dbg_sb = nl.ndarray((1, _PMAX), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(_dbg_sb, _dbg_psum)
                    nisa.dma_copy(
                        dst=debug_flat[_t : _t + 1, _h1 * _PMAX : (_h1 + 1) * _PMAX],
                        src=_dbg_sb,
                        dge_mode=nisa.dge_mode.hwdge,
                    )

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul [T, H] @ [H, E=128] → logits [T, E]
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)

    sbm.open_scope(name="router_softmax")

    # --- router wide DMA (skipped when router_w_wide_sb is supplied) ---
    if router_w_wide_sb is None:
        _router_w_wide_sb = sbm.alloc_stack((_PMAX, _ROUTER_BATCH, E), dtype, buffer=nl.sbuf, name="router_w_wide_sb")
    else:
        _router_w_wide_sb = router_w_wide_sb  # caller-owned — do NOT free

    for h_chunk in nl.affine_range(H_free // _ROUTER_BATCH):
        if router_w_wide_sb is None:
            nisa.dma_copy(
                dst=_router_w_wide_sb,
                src=router_w.ap(
                    pattern=[[E, _PMAX], [_PMAX * E, _ROUTER_BATCH], [1, E]],
                    offset=h_chunk * _ROUTER_BATCH * _PMAX * E,
                ),
                dge_mode=3,
            )
        for h_sub in nl.static_range(_ROUTER_BATCH):
            h1 = h_chunk * _ROUTER_BATCH + h_sub
            nisa.nc_matmul(
                dst=logits_psum[0:T, 0:E],
                stationary=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                moving=_router_w_wide_sb[0:_PMAX, h_sub, 0:E],
            )

    # Reference path (HLO dot.339 bf16 → convert.380 fp32): round PSUM to bf16 per-logit,
    # then upcast to fp32 for the softmax chain. bf16-rounding here is the bit-exact match
    # vs reference, since the HF-dtype router weight in qwen.py is bf16.
    logits_bf16 = sbm.alloc_stack((T, E), dtype, buffer=nl.sbuf, name="logits_bf16")
    nisa.activation(logits_bf16[0:T, 0:E], op=nl.copy, data=logits_psum[0:T, 0:E])
    logits_sb = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="logits_sb")
    nisa.activation(logits_sb[0:T, 0:E], op=nl.copy, data=logits_bf16[0:T, 0:E])
    if _KERN_TRACE and prg_id == 0:
        nl.device_print("[KERN] router_logits_bf16[0, :10]", logits_bf16[0:1, 0:10])
    sbm.pop_heap()  # rmsnorm_normed (fp32) — no longer needed after router matmul
    # rmsnorm_normed_bf16 stays alive through Stage 4 (used as gate/up matmul moving);
    # it is popped at the end of Stage 4 alongside the other heap tensors.

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8)
    # -----------------------------------------------------------------------
    max_logit = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="max_logit")
    nisa.tensor_reduce(max_logit[0:T, 0:1], nl.maximum, logits_sb[0:T, 0:E], axis=1)

    centered = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="centered")
    nisa.tensor_scalar(
        centered[0:T, 0:E], data=logits_sb[0:T, 0:E],
        op0=nl.subtract, operand0=max_logit[0:T, 0:1],
    )

    exp_vals = sbm.alloc_stack((T, E), nl.float32, buffer=nl.sbuf, name="exp_vals")
    nisa.activation(exp_vals[0:T, 0:E], op=nl.exp, data=centered[0:T, 0:E])

    sum_exp = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="sum_exp")
    nisa.tensor_reduce(sum_exp[0:T, 0:1], nl.add, exp_vals[0:T, 0:E], axis=1)

    inv_sum_exp = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name="inv_sum_exp")
    nisa.activation(inv_sum_exp[0:T, 0:1], op=nl.reciprocal, data=sum_exp[0:T, 0:1])

    # NxDI selects top-K on fp32 router_logits (routing.py:204
    # `_, expert_index = torch.topk(router_logits, self.top_k)`), NOT on softmax
    # probs. max8+nc_find_index8 over probs aliases under fp32 softmax ties /
    # tail underflows (nc_find_index8 returns "first occurrence"), dropping
    # experts on weight seeds where two logits round to equal probs.
    top8_logits = sbm.alloc_heap((T, K), nl.float32, buffer=nl.sbuf, name="top8_logits")
    nisa.max8(dst=top8_logits[0:T, 0:K], src=logits_sb[0:T, 0:E])

    top8_idx = sbm.alloc_heap((T, K), nl.uint32, buffer=nl.sbuf, name="top8_idx")
    nisa.nc_find_index8(dst=top8_idx[0:T, 0:K], data=logits_sb[0:T, 0:E], vals=top8_logits[0:T, 0:K])

    # Recover softmax values at top-K indices via softmax identity:
    #   softmax(l)[i] = exp(l_i - max_l) / sum_j exp(l_j - max_l)
    # Reuses max_logit and inv_sum_exp already computed above. Equivalent to
    # gather(softmax_probs, top8_idx) but avoids materializing the full
    # [T, E] probs tensor.
    top8_centered = sbm.alloc_stack((T, K), nl.float32, buffer=nl.sbuf, name="top8_centered")
    nisa.tensor_scalar(
        top8_centered[0:T, 0:K], data=top8_logits[0:T, 0:K],
        op0=nl.subtract, operand0=max_logit[0:T, 0:1],
    )
    top8_exp = sbm.alloc_stack((T, K), nl.float32, buffer=nl.sbuf, name="top8_exp")
    nisa.activation(top8_exp[0:T, 0:K], op=nl.exp, data=top8_centered[0:T, 0:K])
    top8_vals = sbm.alloc_stack((T, K), nl.float32, buffer=nl.sbuf, name="top8_vals")
    nisa.tensor_scalar(
        top8_vals[0:T, 0:K], data=top8_exp[0:T, 0:K],
        op0=nl.multiply, operand0=inv_sum_exp[0:T, 0:1],
    )

    # Cast top8_vals fp32 → bf16 for L1-normalize in bf16 (matches NxDI F.normalize bf16)
    top8_vals_bf16 = sbm.alloc_heap((T, K), dtype, buffer=nl.sbuf, name="top8_vals_bf16")
    nisa.activation(top8_vals_bf16[0:T, 0:K], op=nl.copy, data=top8_vals[0:T, 0:K])

    if _KERN_TRACE and prg_id == 0:
        nl.device_print("[KERN] top8_idx", top8_idx[0:T, 0:K])
        nl.device_print("[KERN] top8_vals_bf16 (softmax-identity, pre-norm)", top8_vals_bf16[0:T, 0:K])

    sbm.close_scope()  # frees router/softmax transients (not router_w_wide_sb — caller-owned)

    # -----------------------------------------------------------------------
    # Stage 4: Selective-Expert MLP — 2-Wave Expert Processing
    # -----------------------------------------------------------------------
    sbm.open_scope(name="expert_loop_outer")

    # output_temp: fp32 accumulator. HLO shows reference's torch.sum(bf16_tensor, dim=0) lowers
    # to a bf16 reduce.503 that the Neuron compiler implements with fp32-internal accumulation
    # (flags: --enable-mixed-precision-accumulation, --accumulate-on-alu-dtype) and a single
    # bf16 round at the end. Matching that = fp32 accumulator + one final bf16 cast at Stage 5.
    output_temp = sbm.alloc_stack((_PMAX, H_free_shard, T), nl.float32, buffer=nl.sbuf, name="output_temp")

    for t in nl.static_range(T):

        sbm.open_scope(name=f"token_{t}")

        # ------------------------------------------------------------------
        # Allocate 4 named SBUF buffers (token-scoped — live for both waves)
        # v30b: gate_up_bufs now hold only H_free_shard=8 H-tiles (not H_free=16)
        # ------------------------------------------------------------------
        gate_up_buf0 = sbm.alloc_stack((_PMAX, H_free_shard, _GU_FLAT), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf0_t{t}")
        gate_up_buf1 = sbm.alloc_stack((_PMAX, H_free_shard, _GU_FLAT), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf1_t{t}")
        gate_up_buf2 = sbm.alloc_stack((_PMAX, H_free_shard, _GU_FLAT), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf2_t{t}")
        gate_up_buf3 = sbm.alloc_stack((_PMAX, H_free_shard, _GU_FLAT), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_up_buf3_t{t}")
        gate_up_bufs = [gate_up_buf0, gate_up_buf1, gate_up_buf2, gate_up_buf3]

        down_full0_buf0 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf0_t{t}")
        down_full0_buf1 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf1_t{t}")
        down_full0_buf2 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf2_t{t}")
        down_full0_buf3 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full0_buf3_t{t}")
        down_full0_bufs = [down_full0_buf0, down_full0_buf1, down_full0_buf2, down_full0_buf3]

        down_full1_buf0 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf0_t{t}")
        down_full1_buf1 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf1_t{t}")
        down_full1_buf2 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf2_t{t}")
        down_full1_buf3 = sbm.alloc_stack((_PMAX, H_shard), down_w.dtype, buffer=nl.sbuf, name=f"down_full1_buf3_t{t}")
        down_full1_bufs = [down_full1_buf0, down_full1_buf1, down_full1_buf2, down_full1_buf3]

        # ------------------------------------------------------------------
        # Zero pad region (rows I1:I0 = 64:128) for 4 down_full1 buffers
        # ------------------------------------------------------------------
        for k_pad in range(4):
            nisa.memset(down_full1_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        # gate_t1_128/up_t1_128: single pair of reused buffers (token-scoped)
        # v30b: shrink from H_free=16 to H_free_shard=8
        gate_t1_128 = sbm.alloc_stack((_PMAX, H_free_shard, I0), gate_up_w.dtype, buffer=nl.sbuf, name=f"gate_t1_128_t{t}")
        up_t1_128   = sbm.alloc_stack((_PMAX, H_free_shard, I0), gate_up_w.dtype, buffer=nl.sbuf, name=f"up_t1_128_t{t}")
        nisa.memset(gate_t1_128[0:_PMAX, 0:H_free_shard, nl.ds(I1, I1)], value=0.0)
        nisa.memset(up_t1_128[0:_PMAX, 0:H_free_shard, nl.ds(I1, I1)],   value=0.0)

        # ==================================================================
        # WAVE 0: Experts 0-3
        # ==================================================================

        # Phase 1a: Load experts 0-3 (12 DMAs) — outside per-expert compute scope
        for k in nl.static_range(_K_WAVE):
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k)

            # v30b: Load only H_free_shard=8 H-tiles, offset by prg_id * H_free_shard tiles
            # gate_up_w layout: [E, H=2048, GU_FLAT=384]
            # Viewed as [E, H_free=16, PMAX=128, GU_FLAT=384]
            # Core prg_id loads rows [prg_id*H_shard : (prg_id+1)*H_shard]
            # = tiles [prg_id*H_free_shard : (prg_id+1)*H_free_shard]
            # offset in elements: prg_id * H_free_shard * PMAX * GU_FLAT
            nisa.dma_copy(
                dst=gate_up_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free_shard], [1, _GU_FLAT]],
                    offset=prg_id * H_free_shard * _PMAX * _GU_FLAT,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full0_bufs[k],
                src=down_w.ap(
                    pattern=[[H, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full1_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H_shard]],
                    offset=I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # Match reference F.normalize (qwen.py path → expert_mlps_v2.py:606 → HLO divide.415 bf16).
        # Reference: bf16 norm (reduce.407) then bf16 divide — Neuron lowers bf16 divide to fp32
        # internal division with a single bf16 round at the output.
        sum_topk = sbm.alloc_stack((T, 1), dtype, buffer=nl.sbuf, name=f"sum_topk_t{t}")
        nisa.tensor_reduce(sum_topk[0:T, 0:1], nl.add, top8_vals_bf16[0:T, 0:K], axis=1)

        inv_sum_topk = sbm.alloc_stack((T, 1), nl.float32, buffer=nl.sbuf, name=f"inv_sum_topk_t{t}")
        nisa.activation(inv_sum_topk[0:T, 0:1], op=nl.reciprocal, data=sum_topk[0:T, 0:1])

        # bf16 data × fp32 scalar → bf16 (single bf16 round at dst, matches reference's bf16 divide).
        norm_weights = sbm.alloc_stack((T, K), dtype, buffer=nl.sbuf, name=f"norm_weights_t{t}")
        nisa.tensor_scalar(
            norm_weights[0:T, 0:K], data=top8_vals_bf16[0:T, 0:K],
            op0=nl.multiply, operand0=inv_sum_topk[0:T, 0:1],
        )

        if _KERN_TRACE and prg_id == 0:
            nl.device_print(f"[KERN] sum_topk_bf16 t{t}", sum_topk[0:T, 0:1])
            nl.device_print(f"[KERN] norm_weights (post L1-norm) t{t}", norm_weights[0:T, 0:K])

        # Cast to fp32 for broadcast (tensor_scalar requires fp32 operand0).
        norm_weights_f32 = sbm.alloc_stack((T, K), nl.float32, buffer=nl.sbuf, name=f"norm_weights_f32_t{t}")
        nisa.activation(norm_weights_f32[0:T, 0:K], op=nl.copy, data=norm_weights[0:T, 0:K])
        aff_bcast = sbm.alloc_stack((_PMAX, K), nl.float32, buffer=nl.sbuf, name=f"aff_bcast_t{t}")
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K], src=norm_weights_f32[t:t + 1, 0:K])
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        # PSUM allocation for wave 0 (4-expert capacity)
        gate_up_psum = nl.ndarray((_PMAX, _K_WAVE * 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
        down_psum    = nl.ndarray((_PMAX, _K_WAVE * H_free_shard), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # Phase 2a: Compute experts 0-3
        for k in nl.static_range(_K_WAVE):
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            # Tile-1 tensor_copy
            # v30b: only H_free_shard slices (index dim 1 is H_free_shard not H_free)
            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_free_shard, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free_shard, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_free_shard, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free_shard, nl.ds(I + I0, I1)],
            )

            # Gate/Up matmul — v30b: loop over H_free_shard instead of H_free
            # Moving index: pick correct H-shard column of rmsnorm_normed_bf16
            # Global h index = prg_id * H_free_shard + h1
            for h1 in nl.affine_range(H_free_shard):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                    else:
                        g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]
                        u_stat = up_t1_128[0:_PMAX, h1, 0:I0]
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                        stationary=g_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((prg_id * H_free_shard + h1) * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((prg_id * H_free_shard + h1) * T, T)],
                    )

        # ------------------------------------------------------------------
        # All-reduce gate_up_psum across LNC cores (Wave 0)
        # ------------------------------------------------------------------
        # Plan C: gu_full_bf16_w0 allocated outside allreduce scope so it survives
        # into the per-expert loop below
        gu_full_bf16_w0 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), dtype, buffer=nl.sbuf, name=f"gu_full_bf16_w0_t{t}")

        sbm.open_scope(name=f"w0_allreduce_t{t}")

        gu_send_w0 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), nl.float32, buffer=nl.sbuf, name=f"gu_send_w0_t{t}")
        gu_recv_w0 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), nl.float32, buffer=nl.sbuf, name=f"gu_recv_w0_t{t}")
        gu_full_w0 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), nl.float32, buffer=nl.sbuf, name=f"gu_full_w0_t{t}")

        # Copy PSUM → SBUF (compiler requires sendrecv src in SBUF)
        nisa.activation(gu_send_w0, op=nl.copy, data=gate_up_psum[0:_PMAX, 0:_K_WAVE * 2 * I_tiles])

        nisa.sendrecv(
            send_to_rank=1 - prg_id,
            recv_from_rank=1 - prg_id,
            src=gu_send_w0,
            dst=gu_recv_w0,
            pipe_id=0,
        )

        # Reduce: local + received = full gate_up activation
        nisa.tensor_tensor(gu_full_w0, gu_send_w0, gu_recv_w0, nl.add)

        # Plan C: cast gu_full_w0 fp32 → bf16 before SiLU/gate×up (matches NxDI bf16)
        nisa.activation(gu_full_bf16_w0, op=nl.copy, data=gu_full_w0[0:_PMAX, 0:_K_WAVE * 2 * I_tiles])

        sbm.close_scope()  # w0_allreduce

        # Phase 2a continued: apply SiLU, multiply, down projection per expert
        for k in nl.static_range(_K_WAVE):
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            sbm.open_scope(name=f"w0_expert_{k}_t{t}")

            # Plan C: SiLU and gate×up in bf16
            silu_res = sbm.alloc_stack((_PMAX, I_tiles), dtype, buffer=nl.sbuf, name=f"silu_res_w0k{k}_t{t}")
            nisa.activation(silu_res, op=nl.silu, data=gu_full_bf16_w0[0:_PMAX, gu_base:gu_base + I_tiles])

            up_sb = sbm.alloc_stack((_PMAX, I_tiles), dtype, buffer=nl.sbuf, name=f"up_sb_w0k{k}_t{t}")
            nisa.activation(up_sb, op=nl.copy, data=gu_full_bf16_w0[0:_PMAX, gu_base + I_tiles:gu_base + 2 * I_tiles])

            # Plan C: bf16 gate × up → bf16 (eliminates inter_f32 + cast)
            inter_bf16 = sbm.alloc_stack((_PMAX, I_tiles), dtype, buffer=nl.sbuf, name=f"inter_bf16_w0k{k}_t{t}")
            nisa.tensor_tensor(inter_bf16, silu_res, up_sb, nl.multiply)

            # Down matmul
            for h1_out in nl.affine_range(H_free_shard):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full0_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            # F2: match NxDI HLO expert chain (dot.169 → convert.173 → multiply.176 →
            # convert.177 → reduce.184): down_proj stores bf16, upcast to fp32, fp32
            # multiply by fp32 aff, bf16-round the scaled product, then fp32-internal
            # reduce across 8 experts to bf16. Our kernel's fp32 PSUM has to materialize
            # both bf16 rounds — F2a (down_psum → bf16 → fp32) and F2b (scaled fp32 →
            # bf16 → fp32 for accumulate).
            down_result_bf16 = sbm.alloc_stack((_PMAX, H_free_shard), dtype, buffer=nl.sbuf, name=f"down_result_bf16_w0k{k}_t{t}")
            nisa.activation(
                down_result_bf16,
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + H_free_shard],
            )

            # tensor_scalar with bf16 data input + fp32 operand0 + bf16 dst does
            # fp32(data) * fp32(operand0) = fp32 internal, then one bf16 round at dst —
            # exact match for NxDI multiply.176 + convert.177 (fp32 multiply, bf16 round).
            down_result_scaled_bf16 = sbm.alloc_stack((_PMAX, H_free_shard), dtype, buffer=nl.sbuf, name=f"down_result_scaled_bf16_w0k{k}_t{t}")
            nisa.tensor_scalar(
                down_result_scaled_bf16,
                data=down_result_bf16,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, k:k + 1],  # wave 0: global k = k (0-3)
            )

            # Upcast bf16 → fp32 for the fp32-internal reduce (reduce.184).
            down_result_scaled = sbm.alloc_stack((_PMAX, H_free_shard), nl.float32, buffer=nl.sbuf, name=f"down_result_scaled_w0k{k}_t{t}")
            nisa.activation(down_result_scaled, op=nl.copy, data=down_result_scaled_bf16)

            if _KERN_TRACE and prg_id == 0 and k == 0:
                nl.device_print(
                    f"[KERN] wave0 expert0 down_psum[:8, :8] fp32",
                    down_psum[0:8, d_base:d_base + 8],
                )
                nl.device_print(
                    f"[KERN] wave0 expert0 down_result_bf16[:8, :8] (bf16 round of matmul)",
                    down_result_bf16[0:8, 0:8],
                )
                nl.device_print(
                    f"[KERN] wave0 expert0 down_result_scaled_bf16[:8, :8] (after x aff, bf16)",
                    down_result_scaled_bf16[0:8, 0:8],
                )
                nl.device_print(
                    f"[KERN] wave0 expert0 down_result_scaled[:8, :8] (upcast to fp32 for reduce)",
                    down_result_scaled[0:8, 0:8],
                )

            if k == 0:
                nisa.tensor_copy(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    src=down_result_scaled[0:_PMAX, 0:H_free_shard],
                )
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data1=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                    data2=down_result_scaled[0:_PMAX, 0:H_free_shard],
                    op=nl.add,
                )

            sbm.close_scope()  # w0_expert_k

        # ==================================================================
        # WAVE 1: Experts 4-7
        # ==================================================================

        # NOTE: No down_full1 re-memset needed
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # Phase 1b: Load experts 4-7 (reusing buffers 0-3) — outside per-expert compute scope
        for k in nl.static_range(_K_WAVE):
            kk = k + 4  # global expert index
            expert_id = top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + kk)

            # v30b: load only H_free_shard H-tiles for gate/up
            nisa.dma_copy(
                dst=gate_up_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free_shard], [1, _GU_FLAT]],
                    offset=prg_id * H_free_shard * _PMAX * _GU_FLAT,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full0_bufs[k],
                src=down_w.ap(
                    pattern=[[H, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full1_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H_shard]],
                    offset=I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # Phase 2b: Compute experts 4-7 (gate/up matmul)
        for k in nl.static_range(_K_WAVE):
            kk = k + 4  # global expert index for affinity lookup
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            # Tile-1 tensor_copy (H_free_shard slices)
            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_free_shard, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free_shard, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_free_shard, 0:I1],
                src=gate_up_bufs[k][0:_PMAX, 0:H_free_shard, nl.ds(I + I0, I1)],
            )

            # Gate/Up matmul — v30b: loop over H_free_shard
            for h1 in nl.affine_range(H_free_shard):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gate_up_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                    else:
                        g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]
                        u_stat = up_t1_128[0:_PMAX, h1, 0:I0]
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                        stationary=g_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((prg_id * H_free_shard + h1) * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds((prg_id * H_free_shard + h1) * T, T)],
                    )

        # ------------------------------------------------------------------
        # All-reduce gate_up_psum across LNC cores (Wave 1)
        # ------------------------------------------------------------------
        # Plan C: gu_full_bf16_w1 allocated outside allreduce scope so it survives
        # into the per-expert loop below
        gu_full_bf16_w1 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), dtype, buffer=nl.sbuf, name=f"gu_full_bf16_w1_t{t}")

        sbm.open_scope(name=f"w1_allreduce_t{t}")

        gu_send_w1 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), nl.float32, buffer=nl.sbuf, name=f"gu_send_w1_t{t}")
        gu_recv_w1 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), nl.float32, buffer=nl.sbuf, name=f"gu_recv_w1_t{t}")
        gu_full_w1 = sbm.alloc_stack((_PMAX, _K_WAVE * 2 * I_tiles), nl.float32, buffer=nl.sbuf, name=f"gu_full_w1_t{t}")

        # Copy PSUM → SBUF
        nisa.activation(gu_send_w1, op=nl.copy, data=gate_up_psum[0:_PMAX, 0:_K_WAVE * 2 * I_tiles])

        nisa.sendrecv(
            send_to_rank=1 - prg_id,
            recv_from_rank=1 - prg_id,
            src=gu_send_w1,
            dst=gu_recv_w1,
            pipe_id=1,
        )

        # Reduce: local + received = full gate_up activation
        nisa.tensor_tensor(gu_full_w1, gu_send_w1, gu_recv_w1, nl.add)

        # Plan C: cast gu_full_w1 fp32 → bf16 before SiLU/gate×up (matches NxDI bf16)
        nisa.activation(gu_full_bf16_w1, op=nl.copy, data=gu_full_w1[0:_PMAX, 0:_K_WAVE * 2 * I_tiles])

        sbm.close_scope()  # w1_allreduce

        # Phase 2b continued: apply SiLU, multiply, down projection per expert
        for k in nl.static_range(_K_WAVE):
            kk = k + 4  # global expert index for affinity lookup
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            sbm.open_scope(name=f"w1_expert_{k}_t{t}")

            # Plan C: SiLU and gate×up in bf16
            silu_res = sbm.alloc_stack((_PMAX, I_tiles), dtype, buffer=nl.sbuf, name=f"silu_res_w1k{k}_t{t}")
            nisa.activation(silu_res, op=nl.silu, data=gu_full_bf16_w1[0:_PMAX, gu_base:gu_base + I_tiles])

            up_sb = sbm.alloc_stack((_PMAX, I_tiles), dtype, buffer=nl.sbuf, name=f"up_sb_w1k{k}_t{t}")
            nisa.activation(up_sb, op=nl.copy, data=gu_full_bf16_w1[0:_PMAX, gu_base + I_tiles:gu_base + 2 * I_tiles])

            # Plan C: bf16 gate × up → bf16 (eliminates inter_f32 + cast)
            inter_bf16 = sbm.alloc_stack((_PMAX, I_tiles), dtype, buffer=nl.sbuf, name=f"inter_bf16_w1k{k}_t{t}")
            nisa.tensor_tensor(inter_bf16, silu_res, up_sb, nl.multiply)

            # Down matmul
            for h1_out in nl.affine_range(H_free_shard):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full0_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            # F2: bf16 round after down_proj + bf16 round after scale multiply,
            # then upcast to fp32 for the reduce (see wave 0).
            down_result_bf16 = sbm.alloc_stack((_PMAX, H_free_shard), dtype, buffer=nl.sbuf, name=f"down_result_bf16_w1k{k}_t{t}")
            nisa.activation(
                down_result_bf16,
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + H_free_shard],
            )
            down_result_scaled_bf16 = sbm.alloc_stack((_PMAX, H_free_shard), dtype, buffer=nl.sbuf, name=f"down_result_scaled_bf16_w1k{k}_t{t}")
            nisa.tensor_scalar(
                down_result_scaled_bf16,
                data=down_result_bf16,
                op0=nl.multiply,
                operand0=aff_bcast[0:_PMAX, kk:kk + 1],  # wave 1: global k = kk (4-7)
            )
            down_result_scaled = sbm.alloc_stack((_PMAX, H_free_shard), nl.float32, buffer=nl.sbuf, name=f"down_result_scaled_w1k{k}_t{t}")
            nisa.activation(down_result_scaled, op=nl.copy, data=down_result_scaled_bf16)

            # Always accumulate (output_temp already initialized by wave 0)
            nisa.tensor_tensor(
                dst=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                data1=output_temp[0:_PMAX, 0:H_free_shard, t:t + 1],
                data2=down_result_scaled[0:_PMAX, 0:H_free_shard],
                op=nl.add,
            )

            sbm.close_scope()  # w1_expert_k

        sbm.close_scope()  # token_t

    sbm.close_scope()  # expert_loop_outer

    # Free heap in reverse order of allocation
    sbm.pop_heap()  # top8_vals_bf16
    sbm.pop_heap()  # top8_idx
    sbm.pop_heap()  # top8_logits
    sbm.pop_heap()  # rmsnorm_normed_bf16

    # -----------------------------------------------------------------------
    # Stage 5: Cast fp32→bf16, store to SBUF in column-major layout (no transpose)
    # -----------------------------------------------------------------------
    # output_temp [PMAX, H_free_shard, T] fp32 is already column-major (partition-first).
    # Reshape to [PMAX, H_free_shard*T] and cast fp32→bf16.
    sbm.open_scope(name="store")
    out_sb = sbm.alloc_stack((_PMAX, H_free_shard * T), dtype, buffer=nl.sbuf, name="out_sb")
    nisa.activation(
        out_sb[0:_PMAX, 0:H_free_shard * T],
        op=nl.copy,
        data=output_temp.reshape((_PMAX, H_free_shard * T))[0:_PMAX, 0:H_free_shard * T],
    )
    if _KERN_TRACE and prg_id == 0:
        # output_temp [PMAX=128, H_free_shard=8, T=1] fp32. On prg_id=0, H indices are
        # prg_id*H_shard..(prg_id+1)*H_shard = 0..1024. So col-major (p, h1, t=0)
        # maps to H index h1*128 + p. Print [p=0..7, h1=0] = H indices 0..7.
        nl.device_print("[KERN] output_temp[:8, h1=0] fp32 (accum before bf16)", output_temp[0:8, 0:1, 0:1])
        nl.device_print("[KERN] out_sb[:8, h1=0] bf16 (final)", out_sb[0:8, 0:1])
    sbm.close_scope()  # store

    return out_sb  # [PMAX=128, H_free_shard*T=8] bf16 — column-major, partition-first
                   # caller must consume before any further sbm.alloc_stack


@nki.jit
def qwen3_moe_fused_tkg_sbuf_io_hoisted(
    inp, gamma, router_w, gate_up_w, down_w,
    hoisted_gamma=False,
    hoisted_router_w=False,
):
    """
    Thin wrapper: loads inp into SBUF, optionally pre-loads gamma / router_w into SBUF
    (hoisted mode), then delegates to _qwen3_moe_sbuf_in_sbuf_out_hoisted,
    then transposes out_sb back to row-major and DMAs to HBM.

    Parameters:
      hoisted_gamma   : bool — if True, pre-load gamma into SBUF before calling sub-kernel
                               (demonstrates/tests the gamma_sb_ready kwarg path)
      hoisted_router_w: bool — if True, pre-load router_w into SBUF before calling sub-kernel
                               (demonstrates/tests the router_w_wide_sb kwarg path)

    In production (multi-layer megakernel), the caller would allocate these tensors
    outside the layer loop and pass them directly. Here we allocate inside the JIT
    function to show both modes in a single test.

    NOTE: The 8 transposes below are test-only overhead. In production,
    _sb2sb_all_reduce_gather consumes out_sb directly without any transpose.
    """
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)

    B = inp.shape[0]
    T = B
    H_free = _H // _PMAX

    # --- Load inp into SBUF ---
    inp_2d = inp.reshape((T, _H))
    inp_2d_hbm_reshaped = inp_2d.reshape((H_free * T, _PMAX))

    sbm.open_scope(name="inp_load")
    inp_flat_sb = sbm.alloc_stack((H_free * T, _PMAX), inp.dtype, buffer=nl.sbuf, name="inp_flat_sb_wrap")
    nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_reshaped, dge_mode=3)
    inp_trans_psum = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)
    inp_sb = sbm.alloc_stack((_PMAX, H_free * T), inp.dtype, buffer=nl.sbuf, name="inp_sb_wrap")
    nisa.activation(inp_sb[...], op=nl.copy, data=inp_trans_psum[...])
    # NOTE: do NOT close inp_load scope here — inp_sb must stay alive for the sub-kernel

    # --- Optionally pre-load gamma into SBUF (hoisted mode) ---
    gamma_sb_ready = None
    if hoisted_gamma:
        sbm.open_scope(name="gamma_hoist")
        gamma_1d = gamma.reshape((_H,))
        gamma_1d_hbm_reshaped = gamma_1d.reshape((H_free, _PMAX))
        gamma_flat_sb_h = sbm.alloc_stack((H_free, _PMAX), gamma.dtype, buffer=nl.sbuf, name="gamma_flat_sb_hoist")
        nisa.dma_copy(dst=gamma_flat_sb_h, src=gamma_1d_hbm_reshaped, dge_mode=3)
        gamma_trans_psum_h = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=gamma_trans_psum_h, data=gamma_flat_sb_h)
        gamma_sb_ready = sbm.alloc_stack((_PMAX, H_free), gamma.dtype, buffer=nl.sbuf, name="gamma_sb_hoist")
        nisa.activation(gamma_sb_ready[...], op=nl.copy, data=gamma_trans_psum_h[...])
        # NOTE: do NOT close gamma_hoist scope — gamma_sb_ready must stay alive for sub-kernel

    # --- Optionally pre-load router_w into SBUF (hoisted mode) ---
    router_w_wide_sb = None
    if hoisted_router_w:
        sbm.open_scope(name="router_w_hoist")
        router_w_wide_sb = sbm.alloc_stack((_PMAX, _ROUTER_BATCH, _E), router_w.dtype, buffer=nl.sbuf, name="router_w_wide_sb_hoist")
        # H_free=16, _ROUTER_BATCH=16 → h_chunk loop runs once (h_chunk=0)
        nisa.dma_copy(
            dst=router_w_wide_sb,
            src=router_w.ap(
                pattern=[[_E, _PMAX], [_PMAX * _E, _ROUTER_BATCH], [1, _E]],
                offset=0,
            ),
            dge_mode=3,
        )
        # NOTE: do NOT close router_w_hoist scope — router_w_wide_sb must stay alive for sub-kernel

    out_sb = _qwen3_moe_sbuf_in_sbuf_out_hoisted(
        inp_sb, inp.dtype, T, gamma, router_w, gate_up_w, down_w,
        sbm=sbm,
        gamma_sb_ready=gamma_sb_ready,
        router_w_wide_sb=router_w_wide_sb,
    )

    # Convert column-major [PMAX, H_free_shard*T] back to row-major [T, H_shard] for HBM
    prg_id = nl.program_id(axis=0)
    output = nl.ndarray((T, _H), dtype=inp.dtype, buffer=nl.shared_hbm)

    sbm.open_scope(name="store_hbm")
    out_row_sb = sbm.alloc_stack((T, _H_SHARD), inp.dtype, buffer=nl.sbuf, name="out_row_sb")

    for h1 in nl.static_range(_H_FREE_SHARD):
        # out_sb[:, h1*T:(h1+1)*T] is [PMAX, T] bf16 — transpose to [T, PMAX] bf16
        tp_psum = nl.ndarray((T, _PMAX), dtype=inp.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=tp_psum[0:T, 0:_PMAX],
            data=out_sb[0:_PMAX, nl.ds(h1 * T, T)],
        )
        nisa.activation(
            dst=out_row_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)],
            op=nl.copy,
            data=tp_psum[0:T, 0:_PMAX],
        )

    nisa.dma_copy(
        dst=output[0:T, nl.ds(prg_id * _H_SHARD, _H_SHARD)],
        src=out_row_sb[0:T, 0:_H_SHARD],
    )
    sbm.close_scope()  # store_hbm

    # Close hoisted scopes in reverse order
    if hoisted_router_w:
        sbm.close_scope()  # router_w_hoist
    if hoisted_gamma:
        sbm.close_scope()  # gamma_hoist

    sbm.close_scope()  # inp_load

    return output


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/ubuntu/models/Qwen3-30B-A3B"
B   = 1
S   = 640
LNC = 2

COMPILER_ARGS = (
    "--enable-saturate-infinity --enable-mixed-precision-accumulation "
    "--model-type transformer -O1 "
    "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def make_config() -> Qwen3MoeInferenceConfig:
    neuron_config = MoENeuronConfig(
        tp_degree=1,
        moe_tp_degree=1,
        batch_size=1,
        seq_len=S,
        flash_decoding_enabled=False,
        logical_nc_config=LNC,
        moe_fused_nki_kernel_enabled=False,
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
        mlp_kernel_enabled=False,
        attn_tkg_nki_kernel_enabled=False,
        attn_block_tkg_nki_kernel_enabled=False,
    )
    cfg = Qwen3MoeInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )
    cfg.intermediate_size = cfg.moe_intermediate_size // 4  # 768 // 4 = 192
    return cfg


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------
def _to_bf16_np(t: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to an ml_dtypes.bfloat16 numpy array."""
    return t.bfloat16().contiguous().view(torch.int16).numpy().view(ml_dtypes.bfloat16)


def _sim_out_to_torch(arr: np.ndarray, shape) -> torch.Tensor:
    """Convert a simulator output (ml_dtypes.bfloat16 numpy) back to a bf16 torch tensor."""
    return (
        torch.from_numpy(arr.view(np.int16))
        .view(torch.int16)
        .view(torch.bfloat16)
        .reshape(shape)
    )


def _bf16_round(t: torch.Tensor) -> torch.Tensor:
    """Force one bf16 rounding step, then keep working in fp32."""
    return t.to(torch.bfloat16).to(torch.float32)


def _rmsnorm_reference_like(hidden_2d: torch.Tensor, gamma_1d: torch.Tensor) -> torch.Tensor:
    """Mirror Qwen3 RMSNorm: fp32 variance, bf16 output."""
    x = hidden_2d.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(var + _EPS)
    normed = normed * gamma_1d.to(torch.float32)
    return normed.to(torch.bfloat16).to(torch.float32)


def _topk_softmax_weights(router_logits_f32: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(router_logits_f32, dim=-1)
    vals = torch.gather(probs, dim=-1, index=topk_idx)
    return vals / vals.sum(dim=-1, keepdim=True)


def _trace_reference_like(
    hidden: torch.Tensor,
    gamma: torch.Tensor,
    router_w: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
):
    """
    Reference-side stage trace.

    Uses the intended high-level NxDI semantics:
    bf16 RMSNorm output, fp32 router logits/softmax, top-k on fp32 logits,
    fp32 normalized top-k weights.
    """
    hidden_2d = hidden.reshape(-1, _H)
    gamma_1d = gamma.reshape(_H)

    rmsnorm_out = _rmsnorm_reference_like(hidden_2d, gamma_1d)
    router_logits = rmsnorm_out @ router_w.to(torch.float32)
    topk_vals, topk_idx = torch.topk(router_logits, k=_K, dim=-1)
    topk_weights = _topk_softmax_weights(router_logits, topk_idx)

    expert_details = []
    expert_contribs = []
    for k in range(_K):
        expert_id = int(topk_idx[0, k].item())
        gu_psum = rmsnorm_out @ gate_up[expert_id].to(torch.float32)
        gu = gu_psum.to(torch.bfloat16).to(torch.float32)
        gate = gu[:, :_I]
        up = gu[:, _I:]
        silu_gate = torch.nn.functional.silu(gate).to(torch.bfloat16).to(torch.float32)
        inter = (silu_gate.to(torch.bfloat16) * up.to(torch.bfloat16)).to(torch.float32)
        down_psum = inter @ down[expert_id].to(torch.float32)
        down_out = down_psum.to(torch.bfloat16).to(torch.float32)
        scaled = down_out * topk_weights[:, k : k + 1]
        expert_contribs.append(scaled)
        expert_details.append(
            {
                "expert_id": expert_id,
                "gu_psum": gu_psum,
                "gu_bf16": gu,
                "gate": gate,
                "up": up,
                "silu_gate": silu_gate,
                "inter": inter,
                "down_psum": down_psum,
                "down_bf16": down_out,
                "scaled": scaled,
            }
        )

    out = torch.stack(expert_contribs, dim=0).sum(dim=0).to(torch.bfloat16).reshape_as(hidden)
    return {
        "rmsnorm_out": rmsnorm_out.reshape_as(hidden).to(torch.bfloat16),
        "router_logits": router_logits,
        "topk_logits": topk_vals,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
        "expert_details": expert_details,
        "out": out,
    }


def _trace_kernel_like(
    hidden: torch.Tensor,
    gamma: torch.Tensor,
    router_w: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
):
    """
    Kernel-side stage trace.

    Mirrors the current v30c math contract closely enough to localize routing bugs:
    bf16 RMSNorm output, router logits rounded fp32->bf16->fp32 before top-k,
    top-k weights reconstructed from rounded logits, bf16 top-k renorm.
    """
    hidden_2d = hidden.reshape(-1, _H)
    gamma_1d = gamma.reshape(_H)

    rmsnorm_out = _rmsnorm_reference_like(hidden_2d, gamma_1d)
    router_logits_psum = rmsnorm_out @ router_w.to(torch.float32)
    router_logits_bf16 = router_logits_psum.to(torch.bfloat16)
    router_logits = router_logits_bf16.to(torch.float32)

    topk_logits, topk_idx = torch.topk(router_logits, k=_K, dim=-1)
    max_logit = router_logits.max(dim=-1, keepdim=True).values
    inv_sum_exp = 1.0 / torch.exp(router_logits - max_logit).sum(dim=-1, keepdim=True)
    topk_vals = torch.exp(topk_logits - max_logit) * inv_sum_exp
    topk_vals_bf16 = topk_vals.to(torch.bfloat16)
    sum_topk_bf16 = topk_vals_bf16.sum(dim=-1, keepdim=True).to(torch.bfloat16)
    norm_weights = (topk_vals_bf16.to(torch.float32) * sum_topk_bf16.to(torch.float32).reciprocal()).to(torch.bfloat16)
    norm_weights_f32 = norm_weights.to(torch.float32)

    expert_details = []
    expert_contribs = []
    for k in range(_K):
        expert_id = int(topk_idx[0, k].item())
        gu_psum = rmsnorm_out @ gate_up[expert_id].to(torch.float32)
        gu_bf16 = gu_psum.to(torch.bfloat16).to(torch.float32)
        gate_part = gu_bf16[:, :_I]
        up_part = gu_bf16[:, _I:]
        silu_res = torch.nn.functional.silu(gate_part.to(torch.bfloat16)).to(torch.float32)
        inter_bf16 = (silu_res.to(torch.bfloat16) * up_part.to(torch.bfloat16)).to(torch.float32)
        down_psum = inter_bf16 @ down[expert_id].to(torch.float32)
        down_result_bf16 = down_psum.to(torch.bfloat16).to(torch.float32)
        scaled_bf16 = (down_result_bf16 * norm_weights_f32[:, k : k + 1]).to(torch.bfloat16).to(torch.float32)
        expert_contribs.append(scaled_bf16)
        expert_details.append(
            {
                "expert_id": expert_id,
                "gu_psum": gu_psum,
                "gu_bf16": gu_bf16,
                "gate": gate_part,
                "up": up_part,
                "silu_gate": silu_res,
                "inter": inter_bf16,
                "down_psum": down_psum,
                "down_bf16": down_result_bf16,
                "scaled": scaled_bf16,
            }
        )

    out = torch.stack(expert_contribs, dim=0).sum(dim=0).to(torch.bfloat16).reshape_as(hidden)
    return {
        "rmsnorm_out": rmsnorm_out.reshape_as(hidden).to(torch.bfloat16),
        "router_logits_psum": router_logits_psum,
        "router_logits_bf16": router_logits_bf16,
        "router_logits": router_logits,
        "topk_logits": topk_logits,
        "topk_idx": topk_idx,
        "topk_vals_pre_norm": topk_vals_bf16,
        "topk_weights": norm_weights_f32,
        "expert_details": expert_details,
        "out": out,
    }


def _summarize_topk_overlap(ref_idx: torch.Tensor, kern_idx: torch.Tensor) -> str:
    ref_list = [int(x) for x in ref_idx.flatten().tolist()]
    kern_list = [int(x) for x in kern_idx.flatten().tolist()]
    same_order = ref_list == kern_list
    overlap = len(set(ref_list) & set(kern_list))
    return (
        f"ref={ref_list}  kern={kern_list}  "
        f"overlap={overlap}/{_K}  same_order={same_order}"
    )


def _print_tensor_delta(name: str, a: torch.Tensor, b: torch.Tensor, k: int = 8) -> None:
    a_f = a.to(torch.float32).flatten()
    b_f = b.to(torch.float32).flatten()
    diff = (a_f - b_f).abs()
    print(
        f"{name}: max_abs={diff.max().item():.6g}  "
        f"mean_abs={diff.mean().item():.6g}",
        flush=True,
    )
    if a_f.numel() <= k:
        print(f"  A={a_f.tolist()}", flush=True)
        print(f"  B={b_f.tolist()}", flush=True)
    else:
        top = torch.topk(diff, k=min(k, diff.numel()))
        idx = top.indices.tolist()
        print(
            f"  top_diff_idx={idx}  "
            f"A={[float(a_f[i]) for i in idx]}  "
            f"B={[float(b_f[i]) for i in idx]}",
            flush=True,
        )


def _print_expert_stage_deltas(ref_trace: dict, kern_trace: dict, max_experts: int = 8) -> None:
    stages = ["gu_psum", "gu_bf16", "silu_gate", "inter", "down_psum", "down_bf16", "scaled"]
    ref_details = ref_trace["expert_details"]
    kern_details = kern_trace["expert_details"]
    n = min(max_experts, len(ref_details), len(kern_details))
    for k in range(n):
        ref_d = ref_details[k]
        kern_d = kern_details[k]
        print(
            f"  expert_slot={k}  ref_id={ref_d['expert_id']}  kern_id={kern_d['expert_id']}",
            flush=True,
        )
        for stage in stages:
            a = ref_d[stage].to(torch.float32).flatten()
            b = kern_d[stage].to(torch.float32).flatten()
            diff = (a - b).abs()
            print(
                f"    {stage:<10} max_abs={diff.max().item():.6g}  mean_abs={diff.mean().item():.6g}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Reference module (compiled with build_module)
# Accepts an optional _weight_store dict; if provided, populates it with the
# kernel's numpy weight arrays while inside build_module's distributed context
# (the only place where initialize_moe_module can run).
# ---------------------------------------------------------------------------
class RefMoEModule(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeInferenceConfig,
        seed: int = 42,
        weight_scale: float = 1.0,
        _weight_store: dict = None,
    ):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(seed)
        self.post_attention_layernorm = CustomRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        ).to(dtype)
        self.mlp = initialize_moe_module(config, init_tkg_module=False).to(dtype)
        if weight_scale != 1.0:
            with torch.no_grad():
                for p in self.parameters():
                    if p.is_floating_point():
                        p.mul_(weight_scale)
        # ExpertMLPs.forward dispatches on self.training: training=True -> forward_all_experts,
        # training=False -> forward_selective_loading (the real TKG path the kernel models).
        # build_module traces modules as-is, and nn.Module defaults to training=True, so without
        # this call the reference would compute all 128 experts (zero-masked, re-normalized),
        # which rounds differently than computing only the top-8. Force eval mode to match
        # the kernel's selective-loading semantics.
        self.eval()
        # Only capture on the _save_checkpoint instantiation (first CPU call, dict is still empty).
        # Later calls happen during AOT tracing where weight init is skipped (get_aot_mode()==True),
        # leaving weights as torch.empty zeros — those must not overwrite the real values.
        if (
            _weight_store is not None
            and not _weight_store  # first-write-wins: skip AOT-mode and meta re-instantiations
            and self.post_attention_layernorm.weight.device.type != "meta"
        ):
            with torch.no_grad():
                _weight_store["gamma"]    = _to_bf16_np(self.post_attention_layernorm.weight.unsqueeze(0))
                _weight_store["router_w"] = _to_bf16_np(self.mlp.router.linear_router.weight.T.contiguous())
                _weight_store["gate_up"]  = _to_bf16_np(self.mlp.expert_mlps.mlp_op.gate_up_proj.weight)
                _weight_store["down"]     = _to_bf16_np(self.mlp.expert_mlps.mlp_op.down_proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.post_attention_layernorm(hidden_states)
        return self.mlp(normed, None)[0]


# ---------------------------------------------------------------------------
# Raw kernel body (identical to test_v30c_vs_nxdi_trn3.py)
# ---------------------------------------------------------------------------
def _v30c_moe_raw(inp, gamma, router_w, gate_up_w, down_w):
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)

    T = inp.shape[0]

    inp_2d              = inp.reshape((T, _H))
    inp_2d_hbm_reshaped = inp_2d.reshape((_H_FREE * T, _PMAX))

    sbm.open_scope(name="inp_load")
    inp_flat_sb = sbm.alloc_stack(
        (_H_FREE * T, _PMAX), inp.dtype, buffer=nl.sbuf, name="inp_flat_sb"
    )
    nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_reshaped, dge_mode=3)
    inp_trans_psum = nl.ndarray((_PMAX, _H_FREE * T), dtype=inp.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)
    inp_sb = sbm.alloc_stack(
        (_PMAX, _H_FREE * T), inp.dtype, buffer=nl.sbuf, name="inp_sb"
    )
    nisa.activation(inp_sb[...], op=nl.copy, data=inp_trans_psum[...])

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

    out_sb = _qwen3_moe_sbuf_in_sbuf_out_hoisted(
        inp_sb, inp.dtype, T,
        gamma, router_w, gate_up_w, down_w,
        sbm=sbm,
        gamma_sb_ready=None,
        router_w_wide_sb=router_w_wide_sb,
    )

    prg_id = nl.program_id(axis=0)
    output = nl.ndarray((T, _H), dtype=inp.dtype, buffer=nl.shared_hbm)

    sbm.open_scope(name="store_hbm")
    out_row_sb = sbm.alloc_stack(
        (T, _H_SHARD), inp.dtype, buffer=nl.sbuf, name="out_row_sb"
    )
    for h1 in nl.static_range(_H_FREE_SHARD):
        tp_psum = nl.ndarray((T, _PMAX), dtype=inp.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=tp_psum[0:T, 0:_PMAX],
            data=out_sb[0:_PMAX, nl.ds(h1 * T, T)],
        )
        nisa.activation(
            dst=out_row_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)],
            op=nl.copy,
            data=tp_psum[0:T, 0:_PMAX],
        )
    nisa.dma_copy(
        dst=output[0:T, nl.ds(prg_id * _H_SHARD, _H_SHARD)],
        src=out_row_sb[0:T, 0:_H_SHARD],
    )
    sbm.close_scope()  # store_hbm
    sbm.close_scope()  # inp_load
    return output


# ---------------------------------------------------------------------------
# Simulate wrapper — plain Python callable, not compiled.
# Receives pre-extracted numpy weight arrays from RefMoEModule._weight_store
# so it never needs to call initialize_moe_module (which requires distributed
# process group state only available inside build_module).
# ---------------------------------------------------------------------------
class KernelSimModule:
    def __init__(self, weight_store: dict):
        self._gamma_np    = weight_store["gamma"]
        self._router_w_np = weight_store["router_w"]
        self._gate_up_np  = weight_store["gate_up"]
        self._down_np     = weight_store["down"]
        # nki.simulate expects the already-grid-sized kernel: nki.jit(fn)[grid]
        # NOT nki.simulate(fn)[grid] (subscripting the simulate result is incorrect).
        self._sim = nki.simulate(nki.jit(_v30c_moe_raw)[LNC])

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        inp_np = _to_bf16_np(hidden_states.reshape(B, _H))
        out_np = self._sim(
            inp_np, self._gamma_np, self._router_w_np,
            self._gate_up_np, self._down_np,
        )  # (T, H) ml_dtypes.bfloat16
        out_f32 = np.array(out_np, dtype=np.float32)
        if not np.any(out_f32 != 0):
            raise RuntimeError(
                "nki.simulate produced all-zero output — check SbufManager "
                "compatibility with the simulator or grid/LNC configuration."
            )
        return _sim_out_to_torch(out_np, hidden_states.shape)


# ---------------------------------------------------------------------------
# Input sampling — identical to test_v30c_vs_nxdi_trn3.py
# ---------------------------------------------------------------------------
def _sample_hidden(rng: np.random.Generator, scale: float, heavy_tail: bool) -> torch.Tensor:
    if heavy_tail:
        df = 5.0
        x = rng.standard_t(df, size=(B, 1, _H))
        x = x * np.sqrt((df - 2.0) / df)
    else:
        x = rng.standard_normal((B, 1, _H))
    return torch.from_numpy((x * scale).astype(np.float32)).bfloat16()


def make_inputs_and_expected(kern_sim: "KernelSimModule", n_samples):
    scales = [0.1, 0.5, 3.0, 5.0]
    combos = [(s, ht) for s in scales for ht in (False, True)]

    inputs_list   = []
    expected_list = []
    for i in range(n_samples):
        scale, heavy_tail = combos[i % len(combos)]
        rng = np.random.default_rng(200 + i)
        hidden = _sample_hidden(rng, scale, heavy_tail)

        out = kern_sim(hidden)

        inputs_list.append((hidden,))
        expected_list.append(out)
        tag = "t5" if heavy_tail else "N"
        print(f"  simulated sample {i:3d}  scale={scale:<4}  dist={tag}", flush=True)

    return inputs_list, expected_list


# ---------------------------------------------------------------------------
# Dump-load helper
# ---------------------------------------------------------------------------
def _run_from_dump(dump_path: str) -> None:
    """Load a failure dump written by test_v30c_vs_nxdi_trn3.py and run the
    simulator on the captured input+weights, comparing against both the kernel
    output and the hardware reference recorded in the dump."""
    print(f"Loading failure dump: {dump_path}", flush=True)
    dump = torch.load(dump_path, map_location="cpu", weights_only=True)
    hidden   = dump["input"]     # [B, 1, H] bf16
    kern_out = dump["kern_out"]  # [B, 1, H] bf16 — what the hardware kernel produced
    ref_out  = dump["ref_out"]   # [B, 1, H] bf16 — hardware reference output
    print(
        f"  weight_scale={dump.get('weight_scale', '?')}  "
        f"sample_idx={dump.get('sample_idx', '?')}",
        flush=True,
    )
    has_ref_weights = all(k in dump for k in ("ref_gamma", "ref_router_w", "ref_gate_up", "ref_down"))
    print(
        f"  separate_ref_weights={'yes' if has_ref_weights else 'no (using kernel weights as fallback)'}",
        flush=True,
    )

    weight_store = {
        "gamma":    _to_bf16_np(dump["gamma"]),
        "router_w": _to_bf16_np(dump["router_w"]),
        "gate_up":  _to_bf16_np(dump["gate_up"]),
        "down":     _to_bf16_np(dump["down"]),
    }
    kern_sim = KernelSimModule(weight_store)

    print("Running nki.simulate on failing input...", flush=True)
    sim_out = kern_sim(hidden)

    ref_trace = _trace_reference_like(
        hidden=hidden.cpu(),
        gamma=dump.get("ref_gamma", dump["gamma"]).cpu(),
        router_w=dump.get("ref_router_w", dump["router_w"]).cpu(),
        gate_up=dump.get("ref_gate_up", dump["gate_up"]).cpu(),
        down=dump.get("ref_down", dump["down"]).cpu(),
    )
    kern_trace = _trace_kernel_like(
        hidden=hidden.cpu(),
        gamma=dump["gamma"].cpu(),
        router_w=dump["router_w"].cpu(),
        gate_up=dump["gate_up"].cpu(),
        down=dump["down"].cpu(),
    )

    print("\n--- simulator vs kernel output (expected by hardware kernel) ---", flush=True)
    try:
        torch.testing.assert_close(sim_out, kern_out, atol=1e-5, rtol=0, check_device=False)
        print("MATCH", flush=True)
    except AssertionError as e:
        print(f"MISMATCH: {e}", flush=True)

    print("\n--- simulator vs hardware reference (per-element torch.assert_close) ---", flush=True)
    try:
        torch.testing.assert_close(sim_out, ref_out, atol=1e-5, rtol=0, check_device=False)
        print("MATCH", flush=True)
    except AssertionError as e:
        print(f"MISMATCH: {e}", flush=True)

    print("\n--- simulator vs hardware reference (hardware test semantics: torch_neuronx.assert_close) ---", flush=True)
    import torch_neuronx.testing as _tnx_testing
    try:
        _tnx_testing.assert_close(sim_out, ref_out, atol=1e-5, rtol=0, check_device=False)
        print("MATCH (passes hardware test)", flush=True)
    except AssertionError as e:
        print(f"MISMATCH: {e}", flush=True)

    print("\n--- traced-stage comparison: reference-like vs kernel-like routing ---", flush=True)
    _print_tensor_delta("rmsnorm_out", ref_trace["rmsnorm_out"], kern_trace["rmsnorm_out"])
    _print_tensor_delta("router_logits", ref_trace["router_logits"], kern_trace["router_logits"])
    _print_tensor_delta("topk_logits", ref_trace["topk_logits"], kern_trace["topk_logits"])
    print(
        "topk_idx: "
        + _summarize_topk_overlap(ref_trace["topk_idx"], kern_trace["topk_idx"]),
        flush=True,
    )
    _print_tensor_delta("topk_weights", ref_trace["topk_weights"], kern_trace["topk_weights"])

    print("\n--- traced outputs vs recorded outputs ---", flush=True)
    _print_tensor_delta("reference-like out vs ref_out", ref_trace["out"], ref_out)
    _print_tensor_delta("kernel-like out vs kern_out", kern_trace["out"], kern_out)
    _print_tensor_delta("kernel-like out vs ref_out", kern_trace["out"], ref_out)

    print("\n--- routing values ---", flush=True)
    print(
        f"  ref topk_idx     = {[int(x) for x in ref_trace['topk_idx'].flatten().tolist()]}",
        flush=True,
    )
    print(
        f"  kern topk_idx    = {[int(x) for x in kern_trace['topk_idx'].flatten().tolist()]}",
        flush=True,
    )
    print(
        f"  ref topk_weights = {[float(x) for x in ref_trace['topk_weights'].flatten().tolist()]}",
        flush=True,
    )
    print(
        f"  kern topk_weights= {[float(x) for x in kern_trace['topk_weights'].flatten().tolist()]}",
        flush=True,
    )
    print(
        f"  ref topk_logits  = {[float(x) for x in ref_trace['topk_logits'].flatten().tolist()]}",
        flush=True,
    )
    print(
        f"  kern topk_logits = {[float(x) for x in kern_trace['topk_logits'].flatten().tolist()]}",
        flush=True,
    )

    print("\n--- expert-stage deltas: reference-like vs kernel-like ---", flush=True)
    _print_expert_stage_deltas(ref_trace, kern_trace)

    # Print values at the known failing index
    print("\n--- values at failing index (0,0,594) ---", flush=True)
    idx = 594
    print(f"  sim_out  [0,0,{idx}] = {sim_out.flatten()[idx].item():.6f}", flush=True)
    print(f"  kern_out [0,0,{idx}] = {kern_out.flatten()[idx].item():.6f}", flush=True)
    print(f"  ref_out  [0,0,{idx}] = {ref_out.flatten()[idx].item():.6f}", flush=True)
    # Also print neighbourhood for context
    print(f"  sim_out  [0,0,590:600] = {sim_out.flatten()[590:600].tolist()}", flush=True)
    print(f"  ref_out  [0,0,590:600] = {ref_out.flatten()[590:600].tolist()}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-scale", type=float, default=1.0,
                        help="Weight scale factor (default: 1.0)")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument(
        "--load-dump",
        default=None,
        metavar="PATH",
        help="Load a .pt failure dump from test_v30c_vs_nxdi_trn3.py and run "
             "the simulator on the captured input/weights (skips compilation).",
    )
    args = parser.parse_args()

    if args.load_dump:
        _run_from_dump(args.load_dump)
        return

    cfg = make_config()
    print(
        f"Config: H={cfg.hidden_size} E={cfg.num_experts} I={cfg.moe_intermediate_size} "
        f"K={cfg.num_experts_per_tok} B={B} S={S} TP=1 LNC={LNC}",
        flush=True,
    )
    print(f"weight_scale={args.weight_scale}  n_samples: {args.n_samples}  |  NKI_PRECISE_FP=1 (bf16)", flush=True)

    example_inputs = [(torch.zeros(B, 1, _H, dtype=torch.bfloat16),)]

    with tempfile.TemporaryDirectory(prefix="v30c_simulate_") as workdir:
        print("\nCompiling RefMoEModule...", flush=True)
        weight_store: dict = {}
        ref_traced = build_module(
            module_cls=RefMoEModule,
            example_inputs=example_inputs,
            module_init_kwargs={
                "config": cfg,
                "seed": args.seed,
                "weight_scale": args.weight_scale,
                "_weight_store": weight_store,
            },
            tp_degree=1,
            logical_nc_config=LNC,
            compiler_args=COMPILER_ARGS,
            compiler_workdir=os.path.join(workdir, "ref_workdir"),
        )
        print("RefMoEModule compile PASS\n", flush=True)

        print("Building KernelSimModule...", flush=True)
        kern_sim = KernelSimModule(weight_store)
        print("KernelSimModule ready\n", flush=True)

        print("Collecting simulator outputs...", flush=True)
        inputs_list, expected_list = make_inputs_and_expected(kern_sim, n_samples=args.n_samples)

        print("\nRunning validate_accuracy (ref hardware vs simulator)...", flush=True)
        validate_accuracy(
            ref_traced,
            inputs_list,
            expected_outputs=expected_list,
            assert_close_kwargs={"atol": 1e-5, "rtol": 0, "check_device": False},
        )

    print("\nOVERALL: PASS")


if __name__ == "__main__":
    main()
