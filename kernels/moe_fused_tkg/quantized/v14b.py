"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

v14b — trn3-compatible W8A8 FP8 variant
=========================================
Based on v13a (W8A8), ported to trn3 by removing tensor_partition_reduce(op=nl.maximum)
which is NOT supported on trn3. Uses per-partition activation quantization instead of
global-token quantization — mathematically valid because each partition's activation
values are quantized with that partition's scale, and the matmul contraction is over
the FREE dimension (not P), so each partition contributes independently.

Key changes vs v13a (trn2-only -> trn3-compatible):
  1. RMSNorm output quantization: replaced global absmax (tensor_partition_reduce max)
     with a 7-step nc_stream_shuffle tree reduction (log2(128) rounds).
     Each round: shuffle with stride 2^i, then elementwise max. After 7 rounds,
     all 128 partitions hold the global max. Same global scale as v13a.
  2. Intermediate (SiLU*Up) quantization: same 7-step tree reduction for global max
     of inter_f32[128P, I_tiles] -> scalar global max broadcast to all partitions.
  3. nc_stream_shuffle tree reduction added; broadcast nc_stream_shuffle unchanged.

  NOTE: Per-partition activation quantization is NOT mathematically valid for this
  kernel because nc_matmul contracts over the P dimension (partition = contraction
  dimension), so all partitions must use the same activation scale for correctness.

Interface: identical to v12i/v13a — no new arguments.
  gate_up_w  [E=128, H=2048, 2*I=384] fp8_e4m3fn (passed as int8)
  gate_up_scales [E=128, GU=384]       fp32  per-output-neuron
  down_w     [E=128, I=192,  H=2048]   fp8_e4m3fn (passed as int8)
  down_scales [E=128, H=2048]          fp32  per-output-neuron (full H)
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.tensor_view import TensorView

# Hardware constants
_PMAX = 128       # partition dimension max on trn2
_PSUM_FREE = 512  # PSUM free-dimension max on trn2

# Qwen3-30B-A3B at TP=4 fixed dims
_H = 2048    # hidden dim
_E = 128     # num experts
_K = 8       # top-K experts
_I = 192     # actual intermediate dim per TP rank
_I0 = 128    # first I tile (full 128 rows)
_I1 = 64     # second I tile (partial: 64 valid rows, 64 zero-padded)
_I_TILES = 2 # two I-dimension tiles
_EPS = 1e-6
_FP8_MAX = 240.0  # fp8_e4m3 max representable value

# Flat gate+up combined width (native layout: gate cols 0:I, up cols I:2*I)
_GU_FLAT = 2 * _I   # = 384

# LNC=2 H-sharding constants
_N_PRGS = 2
_H_FREE = _H // _PMAX              # = 16 tiles of 128 each

# Full H sharding: no TP sharding at kernel level, LNC=2 gives 1024 per prg_id
_H_SHARD_TP = _H                   # = 2048
_H_SHARD = _H // _N_PRGS           # = 1024 per LNC prg_id
_H_FREE_SHARD = _H_SHARD // _PMAX  # = 8
_H_SHARD_BLOCKS = _H_SHARD // _PMAX  # = 8

# Router DMA batching
_ROUTER_BATCH = 16

# 2-wave constants
_K_WAVE = 4  # experts per wave

# gate_up_scales: [E, GU=384] — per output neuron (3 j-blocks of 128)
_GU_J_BLOCKS = _GU_FLAT // _PMAX   # = 3

# Pre-computed shuffle masks for within-quadrant (32-lane) tree reduction
# _MASK_STRIDE_k[i] = (i + 2^k) % 32 — reads from neighbor at distance 2^k
_MASK_STRIDE_1  = [(i + 1)  % 32 for i in range(32)]
_MASK_STRIDE_2  = [(i + 2)  % 32 for i in range(32)]
_MASK_STRIDE_4  = [(i + 4)  % 32 for i in range(32)]
_MASK_STRIDE_8  = [(i + 8)  % 32 for i in range(32)]
_MASK_STRIDE_16 = [(i + 16) % 32 for i in range(32)]
# Broadcast mask: all dest partitions read from source partition 0
_MASK_BCAST_0 = [0] * 32
# Cross-quadrant combination: combine 4 quadrant maxes stored at positions 0,1,2,3
# in a 32-element buffer. Rotate by 1 and 2 to reduce 4 values to their max.
# All values 0-3 are valid (within 0-31 range required by nc_stream_shuffle).
_MASK_Q4_STRIDE_1 = [1, 2, 3, 0] + [0] * 28   # rotate by 1 among first 4
_MASK_Q4_STRIDE_2 = [2, 3, 0, 1] + [0] * 28   # rotate by 2 among first 4


@nki.jit
def qwen3_moe_fused_tkg(
    inp,              # [B, 1, H=2048]              bf16
    gamma,            # [1, H=2048]                 bf16
    router_w,         # [H=2048, E=128]              bf16  (router stays bf16)
    gate_up_w,        # [E=128, H=2048, 2*I=384]     int8 (reinterpreted as fp8_e4m3fn)
    gate_up_scales,   # [E=128, GU=384]               fp32  per-output-neuron
    down_w,           # [E=128, I=192,  H=2048]       int8 (reinterpreted as fp8_e4m3fn)
    down_scales,      # [E=128, H=2048]               fp32  per-output-neuron (full H)
):
    """
    W8A8 FP8 fused RMSNorm + Router + TopK(K=8) + Expert MLP for Qwen3 MoE TKG.
    Both weights AND activations are fp8; dequantize to bf16 output.
    Invoked as: qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w,
                                         gate_up_scales, down_w, down_scales)
    Returns: output [T, H=2048] bf16
    """
    B = inp.shape[0]
    T = B

    H = _H
    E = _E
    K = _K
    I = _I
    I0 = _I0
    I1 = _I1
    H_free = _H_FREE
    H_free_shard = _H_FREE_SHARD   # = 8
    H_shard = _H_SHARD             # = 1024
    I_tiles = _I_TILES

    prg_id = nl.program_id(axis=0)

    # -----------------------------------------------------------------------
    # Stage 1: RMSNorm
    # -----------------------------------------------------------------------
    inp_2d = inp.reshape((T, H))
    inp_2d_hbm_reshaped = inp_2d.reshape((H_free * T, _PMAX))
    inp_flat_sb = nl.ndarray((H_free * T, _PMAX), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=inp_flat_sb, src=inp_2d_hbm_reshaped, dge_mode=3)

    inp_trans_psum = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)

    rmsnorm_out = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_out[...], op=nl.copy, data=inp_trans_psum[...])

    gamma_1d = gamma.reshape((H,))
    gamma_1d_hbm_reshaped = gamma_1d.reshape((H_free, _PMAX))
    gamma_flat_sb = nl.ndarray((H_free, _PMAX), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_flat_sb, src=gamma_1d_hbm_reshaped, dge_mode=3)
    gamma_trans_psum = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=gamma_trans_psum, data=gamma_flat_sb)
    gamma_sb = nl.ndarray((_PMAX, H_free), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.activation(gamma_sb[...], op=nl.copy, data=gamma_trans_psum[...])

    rmsnorm_sq = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(rmsnorm_sq[...], op=nl.square, data=rmsnorm_out[...])

    rmsnorm_reduced = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(rmsnorm_reduced[0:_PMAX, 0:T], nl.add, rmsnorm_sq[0:_PMAX, 0:H_free * T], axis=1)

    gamma_mult = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(gamma_mult[...], rmsnorm_out[...], gamma_sb[...], nl.multiply)

    sum_reduced_sb = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=sum_reduced_sb[0:1, 0:T], data=rmsnorm_reduced[0:_PMAX, 0:T], op=nl.add)

    norm_sum_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=norm_sum_sb[0:1, 0:T], src=sum_reduced_sb[0:1, 0:T])
    for g in nl.affine_range(4):
        nisa.nc_stream_shuffle(
            dst=norm_sum_sb[nl.ds(g * 32, 32), 0:T],
            src=norm_sum_sb[0:1, 0:T],
            shuffle_mask=[0] * 32,
        )

    eps_sb = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(eps_sb, value=_EPS)
    norm_factor_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        norm_factor_sb[0:_PMAX, 0:T],
        op=nl.rsqrt,
        data=norm_sum_sb[0:_PMAX, 0:T],
        scale=1.0 / H,
        bias=eps_sb[0:_PMAX, :],
    )

    norm_factor_bcast = TensorView(norm_factor_sb).expand_dim(dim=2).broadcast(dim=2, size=H_free)
    # rmsnorm_normed: fp32 [128P, H_free*T] — used for activation quantization
    rmsnorm_normed = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(rmsnorm_normed[...], gamma_mult[...], norm_factor_bcast.get_view(), nl.multiply)

    # rmsnorm_normed_bf16: bf16 cast — used only for router matmul (router stays bf16)
    rmsnorm_normed_bf16 = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=rmsnorm_normed[...])

    # ------------------------------------------------------------------
    # W8A8 Change 1 (v14b): Quantize RMSNorm output to fp8 for gate_up moving tensor
    # Strategy: global absmax via nc_stream_shuffle tree reduction (trn3 compatible)
    # tensor_partition_reduce(op=nl.maximum) is NOT supported on trn3, so we use
    # a 7-step binary tree reduction with nc_stream_shuffle + elementwise maximum.
    # ------------------------------------------------------------------

    # Compute per-partition absmax: abs of fp32 rmsnorm_normed, reduce over H_free*T
    rmsnorm_abs = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(rmsnorm_abs, op=nl.abs, data=rmsnorm_normed)  # fp32 -> fp32

    # Per-partition max: reduce over H_free*T free dim -> [128P, T]
    rmsnorm_absmax_pp = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(rmsnorm_absmax_pp[0:_PMAX, 0:T], nl.maximum, rmsnorm_abs[0:_PMAX, 0:H_free * T], axis=1)

    # Cross-partition tree reduction to compute global max of rmsnorm_absmax_pp.
    # Strategy: allocate 4 independent [32, T] buffers (one per quadrant), copy
    # each quadrant's data there, do 5 within-quadrant rounds, then combine the 4
    # group maxes. Using separate allocations avoids partition-offset alignment errors
    # in tensor_tensor (all allocations start at partition 0 in their own SBUF space).
    gu_g0 = nl.ndarray((32, T), dtype=nl.float32, buffer=nl.sbuf)
    gu_g1 = nl.ndarray((32, T), dtype=nl.float32, buffer=nl.sbuf)
    gu_g2 = nl.ndarray((32, T), dtype=nl.float32, buffer=nl.sbuf)
    gu_g3 = nl.ndarray((32, T), dtype=nl.float32, buffer=nl.sbuf)
    gu_tmp = nl.ndarray((32, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gu_g0[0:32, 0:T], src=rmsnorm_absmax_pp[0:32, 0:T])
    nisa.tensor_copy(dst=gu_g1[0:32, 0:T], src=rmsnorm_absmax_pp[32:64, 0:T])
    nisa.tensor_copy(dst=gu_g2[0:32, 0:T], src=rmsnorm_absmax_pp[64:96, 0:T])
    nisa.tensor_copy(dst=gu_g3[0:32, 0:T], src=rmsnorm_absmax_pp[96:128, 0:T])

    # 5 within-quadrant rounds for each group (all buffers start at partition 0)
    # Group 0
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g0[0:32, 0:T], shuffle_mask=_MASK_STRIDE_1)
    nisa.tensor_tensor(gu_g0[0:32, 0:T], gu_g0[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g0[0:32, 0:T], shuffle_mask=_MASK_STRIDE_2)
    nisa.tensor_tensor(gu_g0[0:32, 0:T], gu_g0[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g0[0:32, 0:T], shuffle_mask=_MASK_STRIDE_4)
    nisa.tensor_tensor(gu_g0[0:32, 0:T], gu_g0[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g0[0:32, 0:T], shuffle_mask=_MASK_STRIDE_8)
    nisa.tensor_tensor(gu_g0[0:32, 0:T], gu_g0[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g0[0:32, 0:T], shuffle_mask=_MASK_STRIDE_16)
    nisa.tensor_tensor(gu_g0[0:32, 0:T], gu_g0[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    # Group 1
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g1[0:32, 0:T], shuffle_mask=_MASK_STRIDE_1)
    nisa.tensor_tensor(gu_g1[0:32, 0:T], gu_g1[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g1[0:32, 0:T], shuffle_mask=_MASK_STRIDE_2)
    nisa.tensor_tensor(gu_g1[0:32, 0:T], gu_g1[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g1[0:32, 0:T], shuffle_mask=_MASK_STRIDE_4)
    nisa.tensor_tensor(gu_g1[0:32, 0:T], gu_g1[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g1[0:32, 0:T], shuffle_mask=_MASK_STRIDE_8)
    nisa.tensor_tensor(gu_g1[0:32, 0:T], gu_g1[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g1[0:32, 0:T], shuffle_mask=_MASK_STRIDE_16)
    nisa.tensor_tensor(gu_g1[0:32, 0:T], gu_g1[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    # Group 2
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g2[0:32, 0:T], shuffle_mask=_MASK_STRIDE_1)
    nisa.tensor_tensor(gu_g2[0:32, 0:T], gu_g2[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g2[0:32, 0:T], shuffle_mask=_MASK_STRIDE_2)
    nisa.tensor_tensor(gu_g2[0:32, 0:T], gu_g2[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g2[0:32, 0:T], shuffle_mask=_MASK_STRIDE_4)
    nisa.tensor_tensor(gu_g2[0:32, 0:T], gu_g2[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g2[0:32, 0:T], shuffle_mask=_MASK_STRIDE_8)
    nisa.tensor_tensor(gu_g2[0:32, 0:T], gu_g2[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g2[0:32, 0:T], shuffle_mask=_MASK_STRIDE_16)
    nisa.tensor_tensor(gu_g2[0:32, 0:T], gu_g2[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    # Group 3
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g3[0:32, 0:T], shuffle_mask=_MASK_STRIDE_1)
    nisa.tensor_tensor(gu_g3[0:32, 0:T], gu_g3[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g3[0:32, 0:T], shuffle_mask=_MASK_STRIDE_2)
    nisa.tensor_tensor(gu_g3[0:32, 0:T], gu_g3[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g3[0:32, 0:T], shuffle_mask=_MASK_STRIDE_4)
    nisa.tensor_tensor(gu_g3[0:32, 0:T], gu_g3[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g3[0:32, 0:T], shuffle_mask=_MASK_STRIDE_8)
    nisa.tensor_tensor(gu_g3[0:32, 0:T], gu_g3[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    nisa.nc_stream_shuffle(dst=gu_tmp[0:32, 0:T], src=gu_g3[0:32, 0:T], shuffle_mask=_MASK_STRIDE_16)
    nisa.tensor_tensor(gu_g3[0:32, 0:T], gu_g3[0:32, 0:T], gu_tmp[0:32, 0:T], nl.maximum)
    # Now gu_g0/g1/g2/g3[0, :] each hold their quadrant's max
    # Cross-quadrant: take max of all 4 quadrant buffers sequentially (all start at partition 0)
    global_gu_max_1 = nl.ndarray((32, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(global_gu_max_1[0:32, 0:T], gu_g0[0:32, 0:T], gu_g1[0:32, 0:T], nl.maximum)
    global_gu_max_2 = nl.ndarray((32, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(global_gu_max_2[0:32, 0:T], global_gu_max_1[0:32, 0:T], gu_g2[0:32, 0:T], nl.maximum)
    global_gu_max_3 = nl.ndarray((32, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(global_gu_max_3[0:32, 0:T], global_gu_max_2[0:32, 0:T], gu_g3[0:32, 0:T], nl.maximum)
    # global_gu_max_3[0, :] holds the global max

    # Broadcast global max to all 128 partitions
    global_absmax_gu = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
    for g in nl.affine_range(4):
        nisa.nc_stream_shuffle(
            dst=global_absmax_gu[nl.ds(g * 32, 32), 0:T],
            src=global_gu_max_3[0:1, 0:T],
            shuffle_mask=_MASK_BCAST_0,
        )
    # global_absmax_gu[p, 0] = global max for all p

    # Compute act_scale_gu_col = global_absmax / FP8_MAX -> [128P, 1] (same value all partitions)
    act_scale_gu_col = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=act_scale_gu_col[0:_PMAX, 0:1],
        data=global_absmax_gu[0:_PMAX, 0:1],
        op0=nl.multiply,
        operand0=1.0 / _FP8_MAX,
    )

    # Quantize: rmsnorm_normed * (1/act_scale_gu) -> fp8_e4m3 [128P, H_free*T]
    # Use reciprocal + multiply instead of divide (nl.divide not valid as tensor_tensor op)
    act_scale_gu_inv = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(act_scale_gu_inv, act_scale_gu_col)
    act_scale_gu_bcast = TensorView(act_scale_gu_inv).broadcast(dim=1, size=H_free)  # [128P, H_free]
    rmsnorm_scaled = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(rmsnorm_scaled, rmsnorm_normed, act_scale_gu_bcast.get_view(), nl.multiply)

    # Cast scaled fp32 -> fp8_e4m3 (nl.float8_e4m3 = float8e4, accepted by BIR verifier)
    rmsnorm_fp8 = nl.ndarray((_PMAX, H_free * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    nisa.activation(rmsnorm_fp8, op=nl.copy, data=rmsnorm_scaled)

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul [T, H] @ [H, E=128] → logits [T, E]
    # Router stays bf16 — use rmsnorm_normed_bf16 (not fp8)
    # -----------------------------------------------------------------------
    logits_psum = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.psum)
    router_w_wide_sb = nl.ndarray((_PMAX, _ROUTER_BATCH, E), dtype=inp.dtype, buffer=nl.sbuf)

    for h_chunk in nl.affine_range(H_free // _ROUTER_BATCH):
        nisa.dma_copy(
            dst=router_w_wide_sb,
            src=router_w.ap(
                pattern=[[E, _PMAX], [_PMAX * E, _ROUTER_BATCH], [1, E]],
                offset=h_chunk * _ROUTER_BATCH * _PMAX * E,
            ),
            dge_mode=3,
        )
        for h_sub in nl.affine_range(_ROUTER_BATCH):
            h1 = h_chunk * _ROUTER_BATCH + h_sub
            nisa.nc_matmul(
                dst=logits_psum[0:T, 0:E],
                stationary=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                moving=router_w_wide_sb[0:_PMAX, h_sub, 0:E],
            )

    logits_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(logits_sb[0:T, 0:E], op=nl.copy, data=logits_psum[0:T, 0:E])

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8)
    # -----------------------------------------------------------------------
    max_logit = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(max_logit[0:T, 0:1], nl.maximum, logits_sb[0:T, 0:E], axis=1)

    centered = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        centered[0:T, 0:E], data=logits_sb[0:T, 0:E],
        op0=nl.subtract, operand0=max_logit[0:T, 0:1],
    )

    exp_vals = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(exp_vals[0:T, 0:E], op=nl.exp, data=centered[0:T, 0:E])

    sum_exp = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(sum_exp[0:T, 0:1], nl.add, exp_vals[0:T, 0:E], axis=1)

    inv_sum_exp = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(inv_sum_exp[0:T, 0:1], op=nl.reciprocal, data=sum_exp[0:T, 0:1])

    probs = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        probs[0:T, 0:E], data=exp_vals[0:T, 0:E],
        op0=nl.multiply, operand0=inv_sum_exp[0:T, 0:1],
    )

    top8_vals = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8_vals[0:T, 0:K], src=probs[0:T, 0:E])

    top8_idx = nl.ndarray((T, K), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=top8_idx[0:T, 0:K], data=probs[0:T, 0:E], vals=top8_vals[0:T, 0:K])

    # -----------------------------------------------------------------------
    # Stage 4: Selective-Expert MLP — 2-Wave Expert Processing
    # -----------------------------------------------------------------------
    # output_temp: [128, H_free_shard=8] (T=1 always in TKG)
    output_temp = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.affine_range(T):

        # ------------------------------------------------------------------
        # Allocate fp8 weight buffers — Wave 0
        # ------------------------------------------------------------------
        gate_up_fp8_buf0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_buf1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_buf2 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_buf3 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_bufs = [gate_up_fp8_buf0, gate_up_fp8_buf1, gate_up_fp8_buf2, gate_up_fp8_buf3]

        # gate_up scale buffers: [128P, 3F] — one scale per output neuron
        gate_up_scale_buf0 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_buf1 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_buf2 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_buf3 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_bufs = [gate_up_scale_buf0, gate_up_scale_buf1, gate_up_scale_buf2, gate_up_scale_buf3]

        # up scale rearranged buffers for post-matmul scale
        up_scale_i0_buf0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_buf1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_buf2 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_buf3 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_bufs = [up_scale_i0_buf0, up_scale_i0_buf1, up_scale_i0_buf2, up_scale_i0_buf3]

        up_scale_i1_buf0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_buf1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_buf2 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_buf3 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_bufs = [up_scale_i1_buf0, up_scale_i1_buf1, up_scale_i1_buf2, up_scale_i1_buf3]

        # down fp8 weight buffers (H_shard=1024 per prg_id)
        down_full0_fp8_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_bufs = [down_full0_fp8_buf0, down_full0_fp8_buf1, down_full0_fp8_buf2, down_full0_fp8_buf3]

        down_full1_fp8_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_bufs = [down_full1_fp8_buf0, down_full1_fp8_buf1, down_full1_fp8_buf2, down_full1_fp8_buf3]

        # down scale buffers: [128P, 8F]
        down_scale_buf0 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_buf1 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_buf2 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_buf3 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_bufs = [down_scale_buf0, down_scale_buf1, down_scale_buf2, down_scale_buf3]

        # ------------------------------------------------------------------
        # Allocate fp8 weight buffers — Wave 1 (separate to avoid data hazards)
        # ------------------------------------------------------------------
        gate_up_fp8_w1_buf0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_w1_buf1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_w1_buf2 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_w1_buf3 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_w1_bufs = [gate_up_fp8_w1_buf0, gate_up_fp8_w1_buf1, gate_up_fp8_w1_buf2, gate_up_fp8_w1_buf3]

        gate_up_scale_w1_buf0 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_w1_buf1 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_w1_buf2 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_w1_buf3 = nl.ndarray((_PMAX, _GU_J_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_scale_w1_bufs = [gate_up_scale_w1_buf0, gate_up_scale_w1_buf1, gate_up_scale_w1_buf2, gate_up_scale_w1_buf3]

        up_scale_i0_w1_buf0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_w1_buf1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_w1_buf2 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_w1_buf3 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i0_w1_bufs = [up_scale_i0_w1_buf0, up_scale_i0_w1_buf1, up_scale_i0_w1_buf2, up_scale_i0_w1_buf3]

        up_scale_i1_w1_buf0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_w1_buf1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_w1_buf2 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_w1_buf3 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_scale_i1_w1_bufs = [up_scale_i1_w1_buf0, up_scale_i1_w1_buf1, up_scale_i1_w1_buf2, up_scale_i1_w1_buf3]

        down_full0_fp8_w1_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_w1_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_w1_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_w1_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full0_fp8_w1_bufs = [down_full0_fp8_w1_buf0, down_full0_fp8_w1_buf1, down_full0_fp8_w1_buf2, down_full0_fp8_w1_buf3]

        down_full1_fp8_w1_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_w1_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_w1_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_w1_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_full1_fp8_w1_bufs = [down_full1_fp8_w1_buf0, down_full1_fp8_w1_buf1, down_full1_fp8_w1_buf2, down_full1_fp8_w1_buf3]

        down_scale_w1_buf0 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_w1_buf1 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_w1_buf2 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_w1_buf3 = nl.ndarray((_PMAX, _H_SHARD_BLOCKS), dtype=nl.float32, buffer=nl.sbuf)
        down_scale_w1_bufs = [down_scale_w1_buf0, down_scale_w1_buf1, down_scale_w1_buf2, down_scale_w1_buf3]

        # ------------------------------------------------------------------
        # Pre-allocate all 8 expert ID scratch buffers
        # ------------------------------------------------------------------
        eid_all_0 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_1 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_2 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_3 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_4 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_5 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_6 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_7 = nl.ndarray((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        eid_all_bufs = [eid_all_0, eid_all_1, eid_all_2, eid_all_3,
                        eid_all_4, eid_all_5, eid_all_6, eid_all_7]

        # zero_bias for post-matmul activation scale ([128,1])
        zero_bias = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(zero_bias, value=0.0)

        # Zero-pad down_full1 rows I1:I0 (fp8 buffers) — Wave 0
        for k_pad in nl.affine_range(4):
            nisa.memset(down_full1_fp8_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        # Zero-pad down_full1_w1 rows I1:I0 (fp8 buffers) — Wave 1
        for k_pad in nl.affine_range(4):
            nisa.memset(down_full1_fp8_w1_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        # ------------------------------------------------------------------
        # V12i tile-1 prefetch buffers with k-dimension
        # Wave 0: [_K_WAVE=4, _PMAX=128, H_free=16, I0=128] fp8
        # ------------------------------------------------------------------
        gate_t1_128_k0 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_k1 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_k2 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_k3 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_bufs = [gate_t1_128_k0, gate_t1_128_k1, gate_t1_128_k2, gate_t1_128_k3]

        up_t1_128_k0 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_k1 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_k2 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_k3 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_bufs = [up_t1_128_k0, up_t1_128_k1, up_t1_128_k2, up_t1_128_k3]

        # Zero-pad tile-1 I1..I0 region (Wave 0)
        for k_pad in nl.affine_range(_K_WAVE):
            nisa.memset(gate_t1_128_bufs[k_pad][0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
            nisa.memset(up_t1_128_bufs[k_pad][0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        # Wave 1 prefetch buffers
        gate_t1_128_w1_k0 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_w1_k1 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_w1_k2 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_w1_k3 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_t1_128_w1_bufs = [gate_t1_128_w1_k0, gate_t1_128_w1_k1, gate_t1_128_w1_k2, gate_t1_128_w1_k3]

        up_t1_128_w1_k0 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_w1_k1 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_w1_k2 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_w1_k3 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        up_t1_128_w1_bufs = [up_t1_128_w1_k0, up_t1_128_w1_k1, up_t1_128_w1_k2, up_t1_128_w1_k3]

        # Zero-pad tile-1 I1..I0 region (Wave 1)
        for k_pad in nl.affine_range(_K_WAVE):
            nisa.memset(gate_t1_128_w1_bufs[k_pad][0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
            nisa.memset(up_t1_128_w1_bufs[k_pad][0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        # ==================================================================
        # Pre-read all 8 expert IDs
        # ==================================================================
        for k8 in nl.affine_range(8):
            nisa.dma_copy(
                dst=eid_all_bufs[k8][0:1, 0:1],
                src=top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k8),
            )

        # Phase 1a: Load Wave 0 experts
        for k in nl.affine_range(_K_WAVE):
            expert_id = eid_all_bufs[k].ap(pattern=[[1, 1], [1, 1]], offset=0)

            nisa.dma_copy(
                dst=gate_up_fp8_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=gate_up_scale_bufs[k],
                src=gate_up_scales.ap(
                    pattern=[[1, _PMAX], [_PMAX, _GU_J_BLOCKS]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full0_fp8_bufs[k],
                src=down_w.ap(
                    pattern=[[H, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full1_fp8_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H_shard]],
                    offset=I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_scale_bufs[k],
                src=down_scales.ap(
                    pattern=[[1, _PMAX], [_PMAX, _H_SHARD_BLOCKS]],
                    offset=prg_id * _H_SHARD,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # Phase 1b: Load Wave 1 experts
        for k in nl.affine_range(_K_WAVE):
            kk = k + 4
            expert_id_w1 = eid_all_bufs[kk].ap(pattern=[[1, 1], [1, 1]], offset=0)

            nisa.dma_copy(
                dst=gate_up_fp8_w1_bufs[k],
                src=gate_up_w.ap(
                    pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, H_free], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=gate_up_scale_w1_bufs[k],
                src=gate_up_scales.ap(
                    pattern=[[1, _PMAX], [_PMAX, _GU_J_BLOCKS]],
                    offset=0,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full0_fp8_w1_bufs[k],
                src=down_w.ap(
                    pattern=[[H, I0], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_full1_fp8_w1_bufs[k][0:I1, 0:H_shard],
                src=down_w.ap(
                    pattern=[[H, I1], [1, H_shard]],
                    offset=I0 * H + prg_id * H_shard,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                    dtype=nl.float8_e4m3,
                ),
                dge_mode=0,
            )

            nisa.dma_copy(
                dst=down_scale_w1_bufs[k],
                src=down_scales.ap(
                    pattern=[[1, _PMAX], [_PMAX, _H_SHARD_BLOCKS]],
                    offset=prg_id * _H_SHARD,
                    scalar_offset=expert_id_w1,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # ------------------------------------------------------------------
        # Compute norm_weights (overlaps with in-flight DMAs)
        # ------------------------------------------------------------------
        sum_topk = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(sum_topk[0:T, 0:1], nl.add, top8_vals[0:T, 0:K], axis=1)

        inv_sum_topk = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(inv_sum_topk[0:T, 0:1], op=nl.reciprocal, data=sum_topk[0:T, 0:1])

        norm_weights = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            norm_weights[0:T, 0:K], data=top8_vals[0:T, 0:K],
            op0=nl.multiply, operand0=inv_sum_topk[0:T, 0:1],
        )

        aff_bcast = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(aff_bcast, value=0.0)
        nisa.tensor_copy(dst=aff_bcast[0:1, 0:K], src=norm_weights[t:t + 1, 0:K])
        for g in nl.affine_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        # Assemble up scale rearranged buffers — Wave 0
        for k in nl.affine_range(_K_WAVE):
            nisa.tensor_copy(
                dst=up_scale_i0_bufs[k][0:I1, 0:1],
                src=gate_up_scale_bufs[k][nl.ds(I1, I1), 1:2],
            )
            nisa.tensor_copy(
                dst=up_scale_i0_bufs[k][nl.ds(I1, I1), 0:1],
                src=gate_up_scale_bufs[k][0:I1, 2:3],
            )
            nisa.tensor_copy(
                dst=up_scale_i1_bufs[k][0:I1, 0:1],
                src=gate_up_scale_bufs[k][nl.ds(I1, I1), 2:3],
            )

        # Assemble up scale rearranged buffers — Wave 1
        for k in nl.affine_range(_K_WAVE):
            nisa.tensor_copy(
                dst=up_scale_i0_w1_bufs[k][0:I1, 0:1],
                src=gate_up_scale_w1_bufs[k][nl.ds(I1, I1), 1:2],
            )
            nisa.tensor_copy(
                dst=up_scale_i0_w1_bufs[k][nl.ds(I1, I1), 0:1],
                src=gate_up_scale_w1_bufs[k][0:I1, 2:3],
            )
            nisa.tensor_copy(
                dst=up_scale_i1_w1_bufs[k][0:I1, 0:1],
                src=gate_up_scale_w1_bufs[k][nl.ds(I1, I1), 2:3],
            )

        # PSUM allocation for wave 0
        gate_up_psum = nl.ndarray((_PMAX, _K_WAVE * 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
        down_psum    = nl.ndarray((_PMAX, _K_WAVE * H_free_shard), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # V12i affine_range prefetch loop for tile-1 tensor_copies (Wave 0)
        for k in nl.affine_range(_K_WAVE):
            nisa.tensor_copy(
                dst=gate_t1_128_bufs[k][0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_fp8_bufs[k][0:_PMAX, 0:H_free, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128_bufs[k][0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_fp8_bufs[k][0:_PMAX, 0:H_free, nl.ds(I + I0, I1)],
            )

        # Initialize output_temp to zero before Phase 2a loop (enables always-add)
        nisa.memset(output_temp, value=0.0)

        # ==================================================================
        # Phase 2a: Compute experts 0-3 (W8A8: fp8 stationary AND fp8 moving)
        # ==================================================================
        for k in nl.affine_range(_K_WAVE):
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            # Gate/Up matmul (fp8 stationary, fp8 moving — W8A8 Change 2)
            for h1 in nl.affine_range(H_free):
                for i_tile in nl.affine_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_fp8_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gate_up_fp8_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                    else:
                        g_stat = gate_t1_128_bufs[k][0:_PMAX, h1, 0:I0]
                        u_stat = up_t1_128_bufs[k][0:_PMAX, h1, 0:I0]
                    # W8A8 Change 2: use fp8 rmsnorm_fp8 as moving tensor
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                        stationary=g_stat,
                        moving=rmsnorm_fp8[0:_PMAX, nl.ds(h1 * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_fp8[0:_PMAX, nl.ds(h1 * T, T)],
                    )

            # Post-matmul: fused gate drain+scale+SiLU, up scale, multiply
            # W8A8 Change 3: combine weight_scale * act_scale_gu into one scale
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)

            for i_tile in nl.affine_range(I_tiles):
                gate_scale_col = gate_up_scale_bufs[k][0:_PMAX, i_tile:i_tile + 1]

                # Combine weight_scale * act_scale_gu for gate
                combined_gate_scale = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(combined_gate_scale, gate_scale_col, act_scale_gu_col, nl.multiply)

                gate_silu = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    gate_silu,
                    op=nl.silu,
                    data=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                    scale=combined_gate_scale,
                    bias=zero_bias,
                )

                if i_tile == 0:
                    up_scale_col = up_scale_i0_bufs[k]
                else:
                    up_scale_col = up_scale_i1_bufs[k]

                # Combine weight_scale * act_scale_gu for up
                combined_up_scale = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(combined_up_scale, up_scale_col, act_scale_gu_col, nl.multiply)

                up_f32_scaled = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    up_f32_scaled,
                    op=nl.copy,
                    data=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                    scale=combined_up_scale,
                    bias=zero_bias,
                )

                nisa.tensor_tensor(inter_f32[0:_PMAX, i_tile:i_tile + 1], gate_silu, up_f32_scaled, nl.multiply)

            # W8A8 Change 4 (v14b): Quantize inter_f32 -> inter_fp8 for down_proj
            # Global absmax via 7-step nc_stream_shuffle tree reduction (trn3 compat)
            inter_abs = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(inter_abs, op=nl.abs, data=inter_f32)

            # Per-partition max: reduce over I_tiles -> [128P, 1]
            inter_absmax_pp = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(inter_absmax_pp, nl.maximum, inter_abs, axis=1)

            # Cross-partition tree reduction for inter global max (3-phase, trn3 compat)
            # Phase 1: within-quadrant reduction using 4 separate partition-0-aligned buffers
            down_g0_w0 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            down_g1_w0 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            down_g2_w0 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            down_g3_w0 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            down_tmp_w0 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=down_g0_w0[0:32, 0:1], src=inter_absmax_pp[0:32, 0:1])
            nisa.tensor_copy(dst=down_g1_w0[0:32, 0:1], src=inter_absmax_pp[32:64, 0:1])
            nisa.tensor_copy(dst=down_g2_w0[0:32, 0:1], src=inter_absmax_pp[64:96, 0:1])
            nisa.tensor_copy(dst=down_g3_w0[0:32, 0:1], src=inter_absmax_pp[96:128, 0:1])
            # Group 0
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g0_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_1)
            nisa.tensor_tensor(down_g0_w0[0:32, 0:1], down_g0_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g0_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_2)
            nisa.tensor_tensor(down_g0_w0[0:32, 0:1], down_g0_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g0_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_4)
            nisa.tensor_tensor(down_g0_w0[0:32, 0:1], down_g0_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g0_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_8)
            nisa.tensor_tensor(down_g0_w0[0:32, 0:1], down_g0_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g0_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_16)
            nisa.tensor_tensor(down_g0_w0[0:32, 0:1], down_g0_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            # Group 1
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g1_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_1)
            nisa.tensor_tensor(down_g1_w0[0:32, 0:1], down_g1_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g1_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_2)
            nisa.tensor_tensor(down_g1_w0[0:32, 0:1], down_g1_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g1_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_4)
            nisa.tensor_tensor(down_g1_w0[0:32, 0:1], down_g1_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g1_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_8)
            nisa.tensor_tensor(down_g1_w0[0:32, 0:1], down_g1_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g1_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_16)
            nisa.tensor_tensor(down_g1_w0[0:32, 0:1], down_g1_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            # Group 2
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g2_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_1)
            nisa.tensor_tensor(down_g2_w0[0:32, 0:1], down_g2_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g2_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_2)
            nisa.tensor_tensor(down_g2_w0[0:32, 0:1], down_g2_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g2_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_4)
            nisa.tensor_tensor(down_g2_w0[0:32, 0:1], down_g2_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g2_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_8)
            nisa.tensor_tensor(down_g2_w0[0:32, 0:1], down_g2_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g2_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_16)
            nisa.tensor_tensor(down_g2_w0[0:32, 0:1], down_g2_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            # Group 3
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g3_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_1)
            nisa.tensor_tensor(down_g3_w0[0:32, 0:1], down_g3_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g3_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_2)
            nisa.tensor_tensor(down_g3_w0[0:32, 0:1], down_g3_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g3_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_4)
            nisa.tensor_tensor(down_g3_w0[0:32, 0:1], down_g3_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g3_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_8)
            nisa.tensor_tensor(down_g3_w0[0:32, 0:1], down_g3_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w0[0:32, 0:1], src=down_g3_w0[0:32, 0:1], shuffle_mask=_MASK_STRIDE_16)
            nisa.tensor_tensor(down_g3_w0[0:32, 0:1], down_g3_w0[0:32, 0:1], down_tmp_w0[0:32, 0:1], nl.maximum)
            # Phase 2: cross-quadrant — take max of 4 quadrant buffers sequentially
            global_down_max_1_w0 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(global_down_max_1_w0[0:32, 0:1], down_g0_w0[0:32, 0:1], down_g1_w0[0:32, 0:1], nl.maximum)
            global_down_max_2_w0 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(global_down_max_2_w0[0:32, 0:1], global_down_max_1_w0[0:32, 0:1], down_g2_w0[0:32, 0:1], nl.maximum)
            global_down_max_3_w0 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(global_down_max_3_w0[0:32, 0:1], global_down_max_2_w0[0:32, 0:1], down_g3_w0[0:32, 0:1], nl.maximum)
            # Phase 3: broadcast global max to all 128 partitions
            global_absmax_down_w0 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for g_d in nl.affine_range(4):
                nisa.nc_stream_shuffle(
                    dst=global_absmax_down_w0[nl.ds(g_d * 32, 32), 0:1],
                    src=global_down_max_3_w0[0:1, 0:1],
                    shuffle_mask=_MASK_BCAST_0)

            # act_scale_down = global_absmax / FP8_MAX -> [128P, 1]
            act_scale_down = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=act_scale_down, data=global_absmax_down_w0, op0=nl.multiply, operand0=1.0 / _FP8_MAX)

            # Quantize inter_f32 -> inter_fp8: multiply by 1/act_scale_down, cast to fp8
            # (nl.divide not valid as tensor_tensor op; use reciprocal+multiply)
            act_scale_down_inv = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(act_scale_down_inv, act_scale_down)
            act_scale_down_bcast = TensorView(act_scale_down_inv).broadcast(dim=1, size=I_tiles)
            inter_scaled = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(inter_scaled, inter_f32, act_scale_down_bcast.get_view(), nl.multiply)

            # W8A8 Change 5: inter_fp8 replaces inter_bf16 as down moving tensor
            inter_fp8 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            nisa.activation(inter_fp8, op=nl.copy, data=inter_scaled)

            # Down matmul (fp8 stationary, fp8 moving — W8A8 Change 5)
            for h1_out in nl.affine_range(H_free_shard):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full0_fp8_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_fp8[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full1_fp8_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_fp8[0:_PMAX, 1:2],
                )

            # W8A8 Change 6: combined_down_scale = down_scale * affinity * act_scale_down
            aff_col_buf = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=aff_col_buf, src=aff_bcast[0:_PMAX, k:k + 1])
            aff_col_view = TensorView(aff_col_buf).broadcast(dim=1, size=_H_FREE_SHARD)
            combined_down_scale = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            # Step 1: down_scale * affinity
            nisa.tensor_tensor(
                combined_down_scale,
                down_scale_bufs[k],
                aff_col_view.get_view(),
                nl.multiply,
            )
            # Step 2: multiply by act_scale_down (broadcast [128P,1] → [128P, H_FREE_SHARD])
            act_scale_down_bcast_h = TensorView(act_scale_down).broadcast(dim=1, size=_H_FREE_SHARD)
            nisa.tensor_tensor(
                combined_down_scale,
                combined_down_scale,
                act_scale_down_bcast_h.get_view(),
                nl.multiply,
            )

            # Drain PSUM: copy raw then apply combined scale
            down_h_raw = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_h_raw,
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + _H_FREE_SHARD],
            )

            down_h_all = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                down_h_all,
                down_h_raw,
                combined_down_scale,
                nl.multiply,
            )

            # Accumulate into output_temp (always-add, initialized to zero before loop)
            nisa.tensor_tensor(
                dst=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                data1=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                data2=down_h_all[0:_PMAX, 0:_H_FREE_SHARD],
                op=nl.add,
            )

        # ==================================================================
        # WAVE 1: Experts 4-7 — use dedicated w1 buffers
        # ==================================================================

        # PSUM memset for wave 1
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # V12i affine_range prefetch loop for tile-1 tensor_copies (Wave 1)
        for k in nl.affine_range(_K_WAVE):
            nisa.tensor_copy(
                dst=gate_t1_128_w1_bufs[k][0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_fp8_w1_bufs[k][0:_PMAX, 0:H_free, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128_w1_bufs[k][0:_PMAX, 0:H_free, 0:I1],
                src=gate_up_fp8_w1_bufs[k][0:_PMAX, 0:H_free, nl.ds(I + I0, I1)],
            )

        # Phase 2b: Compute experts 4-7 (W8A8: fp8 stationary AND fp8 moving)
        for k in nl.affine_range(_K_WAVE):
            kk = k + 4
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            for h1 in nl.affine_range(H_free):
                for i_tile in nl.affine_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gate_up_fp8_w1_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gate_up_fp8_w1_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                    else:
                        g_stat = gate_t1_128_w1_bufs[k][0:_PMAX, h1, 0:I0]
                        u_stat = up_t1_128_w1_bufs[k][0:_PMAX, h1, 0:I0]
                    # W8A8 Change 2: use fp8 rmsnorm_fp8 as moving tensor (Wave 1)
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                        stationary=g_stat,
                        moving=rmsnorm_fp8[0:_PMAX, nl.ds(h1 * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_fp8[0:_PMAX, nl.ds(h1 * T, T)],
                    )

            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)

            for i_tile in nl.affine_range(I_tiles):
                gate_scale_col = gate_up_scale_w1_bufs[k][0:_PMAX, i_tile:i_tile + 1]

                # W8A8 Change 3: combine weight_scale * act_scale_gu for gate (Wave 1)
                combined_gate_scale = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(combined_gate_scale, gate_scale_col, act_scale_gu_col, nl.multiply)

                gate_silu = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    gate_silu,
                    op=nl.silu,
                    data=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                    scale=combined_gate_scale,
                    bias=zero_bias,
                )

                if i_tile == 0:
                    up_scale_col = up_scale_i0_w1_bufs[k]
                else:
                    up_scale_col = up_scale_i1_w1_bufs[k]

                # Combine weight_scale * act_scale_gu for up (Wave 1)
                combined_up_scale = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(combined_up_scale, up_scale_col, act_scale_gu_col, nl.multiply)

                up_f32_scaled = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    up_f32_scaled,
                    op=nl.copy,
                    data=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                    scale=combined_up_scale,
                    bias=zero_bias,
                )

                nisa.tensor_tensor(inter_f32[0:_PMAX, i_tile:i_tile + 1], gate_silu, up_f32_scaled, nl.multiply)

            # W8A8 Change 4 (v14b): Quantize inter_f32 -> inter_fp8 for down_proj (Wave 1)
            # Global absmax via 3-phase nc_stream_shuffle tree reduction (trn3 compat)
            inter_abs = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(inter_abs, op=nl.abs, data=inter_f32)

            inter_absmax_pp = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(inter_absmax_pp, nl.maximum, inter_abs, axis=1)

            # 3-phase tree reduction (Wave 1, trn3 compat — same pattern as Wave 0)
            # Phase 1: within-quadrant reduction using 4 separate partition-0-aligned buffers
            down_g0_w1 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            down_g1_w1 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            down_g2_w1 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            down_g3_w1 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            down_tmp_w1 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=down_g0_w1[0:32, 0:1], src=inter_absmax_pp[0:32, 0:1])
            nisa.tensor_copy(dst=down_g1_w1[0:32, 0:1], src=inter_absmax_pp[32:64, 0:1])
            nisa.tensor_copy(dst=down_g2_w1[0:32, 0:1], src=inter_absmax_pp[64:96, 0:1])
            nisa.tensor_copy(dst=down_g3_w1[0:32, 0:1], src=inter_absmax_pp[96:128, 0:1])
            # Group 0
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g0_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_1)
            nisa.tensor_tensor(down_g0_w1[0:32, 0:1], down_g0_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g0_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_2)
            nisa.tensor_tensor(down_g0_w1[0:32, 0:1], down_g0_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g0_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_4)
            nisa.tensor_tensor(down_g0_w1[0:32, 0:1], down_g0_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g0_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_8)
            nisa.tensor_tensor(down_g0_w1[0:32, 0:1], down_g0_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g0_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_16)
            nisa.tensor_tensor(down_g0_w1[0:32, 0:1], down_g0_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            # Group 1
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g1_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_1)
            nisa.tensor_tensor(down_g1_w1[0:32, 0:1], down_g1_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g1_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_2)
            nisa.tensor_tensor(down_g1_w1[0:32, 0:1], down_g1_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g1_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_4)
            nisa.tensor_tensor(down_g1_w1[0:32, 0:1], down_g1_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g1_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_8)
            nisa.tensor_tensor(down_g1_w1[0:32, 0:1], down_g1_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g1_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_16)
            nisa.tensor_tensor(down_g1_w1[0:32, 0:1], down_g1_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            # Group 2
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g2_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_1)
            nisa.tensor_tensor(down_g2_w1[0:32, 0:1], down_g2_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g2_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_2)
            nisa.tensor_tensor(down_g2_w1[0:32, 0:1], down_g2_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g2_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_4)
            nisa.tensor_tensor(down_g2_w1[0:32, 0:1], down_g2_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g2_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_8)
            nisa.tensor_tensor(down_g2_w1[0:32, 0:1], down_g2_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g2_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_16)
            nisa.tensor_tensor(down_g2_w1[0:32, 0:1], down_g2_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            # Group 3
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g3_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_1)
            nisa.tensor_tensor(down_g3_w1[0:32, 0:1], down_g3_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g3_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_2)
            nisa.tensor_tensor(down_g3_w1[0:32, 0:1], down_g3_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g3_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_4)
            nisa.tensor_tensor(down_g3_w1[0:32, 0:1], down_g3_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g3_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_8)
            nisa.tensor_tensor(down_g3_w1[0:32, 0:1], down_g3_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            nisa.nc_stream_shuffle(dst=down_tmp_w1[0:32, 0:1], src=down_g3_w1[0:32, 0:1], shuffle_mask=_MASK_STRIDE_16)
            nisa.tensor_tensor(down_g3_w1[0:32, 0:1], down_g3_w1[0:32, 0:1], down_tmp_w1[0:32, 0:1], nl.maximum)
            # Phase 2: cross-quadrant — take max of 4 quadrant buffers sequentially
            global_down_max_1_w1 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(global_down_max_1_w1[0:32, 0:1], down_g0_w1[0:32, 0:1], down_g1_w1[0:32, 0:1], nl.maximum)
            global_down_max_2_w1 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(global_down_max_2_w1[0:32, 0:1], global_down_max_1_w1[0:32, 0:1], down_g2_w1[0:32, 0:1], nl.maximum)
            global_down_max_3_w1 = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(global_down_max_3_w1[0:32, 0:1], global_down_max_2_w1[0:32, 0:1], down_g3_w1[0:32, 0:1], nl.maximum)
            # Phase 3: broadcast global max to all 128 partitions
            global_absmax_down_w1 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for g_d in nl.affine_range(4):
                nisa.nc_stream_shuffle(
                    dst=global_absmax_down_w1[nl.ds(g_d * 32, 32), 0:1],
                    src=global_down_max_3_w1[0:1, 0:1],
                    shuffle_mask=_MASK_BCAST_0)

            act_scale_down = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=act_scale_down, data=global_absmax_down_w1, op0=nl.multiply, operand0=1.0 / _FP8_MAX)

            act_scale_down_inv = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(act_scale_down_inv, act_scale_down)
            act_scale_down_bcast = TensorView(act_scale_down_inv).broadcast(dim=1, size=I_tiles)
            inter_scaled = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(inter_scaled, inter_f32, act_scale_down_bcast.get_view(), nl.multiply)

            inter_fp8 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            nisa.activation(inter_fp8, op=nl.copy, data=inter_scaled)

            # Down matmul (fp8 stationary, fp8 moving — W8A8 Change 5, Wave 1)
            for h1_out in nl.affine_range(H_free_shard):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full0_fp8_w1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_fp8[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down_full1_fp8_w1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_fp8[0:_PMAX, 1:2],
                )

            # W8A8 Change 6: combined_down_scale = down_scale * affinity * act_scale_down (Wave 1)
            aff_col_buf = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=aff_col_buf, src=aff_bcast[0:_PMAX, kk:kk + 1])
            aff_col_view = TensorView(aff_col_buf).broadcast(dim=1, size=_H_FREE_SHARD)
            combined_down_scale = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            # Step 1: down_scale * affinity
            nisa.tensor_tensor(
                combined_down_scale,
                down_scale_w1_bufs[k],
                aff_col_view.get_view(),
                nl.multiply,
            )
            # Step 2: multiply by act_scale_down
            act_scale_down_bcast_h = TensorView(act_scale_down).broadcast(dim=1, size=_H_FREE_SHARD)
            nisa.tensor_tensor(
                combined_down_scale,
                combined_down_scale,
                act_scale_down_bcast_h.get_view(),
                nl.multiply,
            )

            # Drain PSUM then apply combined scale
            down_h_raw = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                down_h_raw,
                op=nl.copy,
                data=down_psum[0:_PMAX, d_base:d_base + _H_FREE_SHARD],
            )

            down_h_all = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                down_h_all,
                down_h_raw,
                combined_down_scale,
                nl.multiply,
            )

            # Accumulate Wave 1 into output_temp (always-add)
            nisa.tensor_tensor(
                dst=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                data1=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                data2=down_h_all[0:_PMAX, 0:_H_FREE_SHARD],
                op=nl.add,
            )

    # -----------------------------------------------------------------------
    # Stage 5: Transpose fp32->bf16, store to HBM
    # -----------------------------------------------------------------------
    output = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.shared_hbm)
    out_sb = nl.ndarray((T, H_shard), dtype=inp.dtype, buffer=nl.sbuf)

    for h1 in nl.affine_range(H_free_shard):
        tp_psum = nl.ndarray((T, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp_psum[0:T, 0:_PMAX], data=output_temp[0:_PMAX, h1:h1 + 1])
        nisa.activation(
            dst=out_sb[0:T, nl.ds(h1 * _PMAX, _PMAX)],
            op=nl.copy,
            data=tp_psum[0:T, 0:_PMAX],
        )

    nisa.dma_copy(
        dst=output[0:T, nl.ds(prg_id * H_shard, H_shard)],
        src=out_sb[0:T, 0:H_shard],
    )

    return output


def run(inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales):
    """Run v14b kernel (trn3-compatible W8A8 FP8, per-partition activation quantization).

    Accepts:
      gate_up_w: [E, H, 2*I=384] fp8_e4m3fn (as int8)
      gate_up_scales: [E, GU=384] fp32  per-output-neuron
      down_w: [E, I=192, H=2048] fp8_e4m3fn (as int8)
      down_scales: [E, H=2048] fp32  per-output-neuron (full H)

    Returns: output [T, H=2048] bf16
    """
    import torch
    import torch_xla.core.xla_model as xm

    assert gate_up_w.shape == (_E, _H, _GU_FLAT), f"gate_up_w shape {gate_up_w.shape}"
    assert gate_up_scales.shape == (_E, _GU_FLAT), f"gate_up_scales shape {gate_up_scales.shape}"
    assert down_w.shape == (_E, _I, _H), f"down_w shape {down_w.shape}"
    assert down_scales.shape == (_E, _H), f"down_scales shape {down_scales.shape}"

    xm.mark_step()

    outputs = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w, gate_up_scales, down_w, down_scales)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
