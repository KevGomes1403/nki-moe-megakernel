"""
Custom fused MoE TKG kernel for Qwen3-30B-A3B (TP=4, LNC=2).
Implements from scratch: RMSNorm + Router + TopK(8) + Selective-Expert MLPs

v12a — Plan A: Block FP8 quant + fat-replicated scales + bulk dequant
=====================================================================
Based on v11c (affine_range expert loads + output_temp batching).

Quantization scheme change vs v11c:
  v11c: per-output-neuron fp32 scale, applied POST-matmul on PSUM drain
        via `nisa.activation(scale=gate_scale)` etc.
  v12a: ONE fp32 scale per [128 partitions × 128 free-dim] block, applied
        BEFORE matmul via bulk fp8→bf16 dequant. Block boundaries match
        the v11c matmul tile access pattern:
          gate_up_w blocks per (expert, h1∈[0,16)):
            - gate_t0 = cols [0:128]    (gate first 128, full)
            - gate_t1 = cols [128:192]  (gate last  64,  pad-0 to 128 in t1_128 stationary)
            - up_t0   = cols [192:320]  (up   first 128, full)
            - up_t1   = cols [320:384]  (up   last  64,  pad-0 to 128 in t1_128 stationary)
          → 4 scales per (e, h1), 64 scales per expert.
          down_w  blocks per expert:
            - 2 I-tiles × 16 H-blocks = 32 scales per expert; each LNC program
              consumes 8 of those H-blocks for its H_shard.

Fat scale layout (the key Plan A trick):
  gate_up_scales_fat[E, H_free=16, GU=384] fp32 — pre-replicated so the per-(e,h1)
    free-dim row holds:
      [s_gate_t0]×128, [s_gate_t1]×64, [s_up_t0]×128, [s_up_t1]×64 = 384.
    This matches the gate_up weight free-dim exactly, so the dequant collapses
    to a single bulk multiply with a partition-broadcast load.
  down_scales_fat[E, I_tiles=2, H=2048] fp32 — per (e, i_tile), each 128-block
    of H holds the same scalar repeated 128 times.

Both fat scale tensors are loaded into SBUF with the **load-1-partition + nc_stream_shuffle**
broadcast pattern (the more obvious "stride-0 partition broadcast" path COMPILES and is
bit-correct, but in practice the DMA engine still streams 128P × free_size bytes through
HBM, costing ~32 MB/iter of extra reads — verified empirically: it pushed v12a from
104 μs to 426 μs).  Loading just the 1-partition row and broadcasting via stream_shuffle
keeps HBM scale traffic at ~256 KB/iter (negligible) at the cost of 64 cheap VectorE
shuffle ops.

Dequant strategy:
  After all DMAs are issued (Phase 1a/1b), a SEPARATE affine_range loop dequants
  each k-buffer in one bulk VectorE op:
    nisa.tensor_tensor(bf_buf[k], fp8_buf[k], scale_fat_sb[k], nl.multiply)
  This replaces v11c's per-output-neuron post-matmul `activation(scale=...)` chain
  with a single VectorE pass per buffer.

  We keep load and dequant in SEPARATE affine_ranges per OPTIMIZATION_LOG.md
  Plan C lessons (merged loops regress).

Post-matmul changes:
  - gate silu / up copy activations no longer take a `scale=` argument
    (the stationary is already dequantized to bf16).
  - down PSUM drain no longer multiplies by per-output-neuron `down_scale`;
    it only multiplies by `affinity` (broadcast).

Interface:
  gate_up_w            [E=128, H=2048, 2*I=384]      int8 (reinterpreted as fp8_e4m3)
  gate_up_scales_fat   [E=128, H_free=16, GU=384]    fp32 (FAT replicated, see above)
  down_w               [E=128, I=192, H=2048]        int8 (reinterpreted as fp8_e4m3)
  down_scales_fat      [E=128, I_tiles=2, H=2048]    fp32 (FAT replicated, see above)
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.tensor_view import TensorView

# Hardware constants
_PMAX = 128       # partition dimension max
_PSUM_FREE = 512  # PSUM free-dimension max on trn2

# Qwen3-30B-A3B at TP=4 fixed dims
_H = 2048    # hidden dim
_E = 128     # num experts
_K = 8       # top-K experts
_I = 192     # actual intermediate dim per TP rank (no padding anywhere)
_I0 = 128    # first tile  (full 128 rows)
_I1 = 64     # second tile (partial: 64 valid rows, 64 zero-padded)
_I_TILES = 2 # two I-dimension tiles
_EPS = 1e-6

# Flat gate+up combined width (native layout: gate cols 0:I, up cols I:2*I)
_GU_FLAT = 2 * _I   # = 384

# LNC=2 H-sharding constants (always launched with [2])
_N_PRGS = 2
_H_FREE = _H // _PMAX             # = 16 tiles of 128 each

# Full H sharding: no TP sharding at kernel level, LNC=2 gives 1024 per prg_id
_H_SHARD_TP = _H                  # = 2048 (full H, matches BF16 baseline)
_H_SHARD = _H // _N_PRGS          # = 1024 per LNC prg_id
_H_FREE_SHARD = _H_SHARD // _PMAX  # = 8
_H_SHARD_BLOCKS = _H_SHARD // _PMAX  # = 8 (for scale access)

# Router DMA batching: 16 tiles per DMA
_ROUTER_BATCH = 16

# 2-wave constants
_K_WAVE = 4  # experts per wave


@nki.jit
def qwen3_moe_fused_tkg(
    inp,                  # [B, 1, H=2048]                bf16
    gamma,                # [1, H=2048]                   bf16
    router_w,             # [H=2048, E=128]               bf16  (router stays bf16)
    gate_up_w,            # [E=128, H=2048, 2*I=384]      int8 (reinterpreted as fp8_e4m3)
    gate_up_scales_fat,   # [E=128, H_free=16, GU=384]    fp32  FAT layout (per-block, replicated)
    down_w,               # [E=128, I=192, H=2048]        int8 (reinterpreted as fp8_e4m3)
    down_scales_fat,      # [E=128, I_tiles=2, H=2048]    fp32  FAT layout (per-block, replicated)
):
    """v12a — Plan A: block FP8 quant + fat scales + bulk pre-matmul dequant."""
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
    # Stage 1: RMSNorm   (unchanged from v11c)
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
    for g in nl.static_range(4):
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
    rmsnorm_normed = nl.ndarray((_PMAX, H_free * T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(rmsnorm_normed[...], gamma_mult[...], norm_factor_bcast.get_view(), nl.multiply)

    rmsnorm_normed_bf16 = nl.ndarray((_PMAX, H_free * T), dtype=inp.dtype, buffer=nl.sbuf)
    nisa.activation(rmsnorm_normed_bf16[...], op=nl.copy, data=rmsnorm_normed[...])

    # -----------------------------------------------------------------------
    # Stage 2: Router matmul   (unchanged from v11c)
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
        for h_sub in nl.static_range(_ROUTER_BATCH):
            h1 = h_chunk * _ROUTER_BATCH + h_sub
            nisa.nc_matmul(
                dst=logits_psum[0:T, 0:E],
                stationary=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                moving=router_w_wide_sb[0:_PMAX, h_sub, 0:E],
            )

    logits_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(logits_sb[0:T, 0:E], op=nl.copy, data=logits_psum[0:T, 0:E])

    # -----------------------------------------------------------------------
    # Stage 3: Softmax + TopK(8)   (unchanged from v11c)
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
    # Stage 4: Selective-Expert MLP — 2-Wave with bulk pre-matmul dequant
    # -----------------------------------------------------------------------
    output_temp = nl.ndarray((_PMAX, _H_FREE_SHARD), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.static_range(T):

        # ------------------------------------------------------------------
        # Allocate Wave 0 fp8 weight buffers + fat scale buffers + bf16 mirrors
        # ------------------------------------------------------------------
        gate_up_fp8_buf0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_buf1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_buf2 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_buf3 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_bufs = [gate_up_fp8_buf0, gate_up_fp8_buf1, gate_up_fp8_buf2, gate_up_fp8_buf3]

        # Per-k scale buffers (compiler handles live-range reuse).
        gu_scale_buf0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float32, buffer=nl.sbuf)
        gu_scale_buf1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float32, buffer=nl.sbuf)
        gu_scale_buf2 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float32, buffer=nl.sbuf)
        gu_scale_buf3 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float32, buffer=nl.sbuf)
        gu_scale_bufs = [gu_scale_buf0, gu_scale_buf1, gu_scale_buf2, gu_scale_buf3]

        gu_bf16_buf0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.bfloat16, buffer=nl.sbuf)
        gu_bf16_buf1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.bfloat16, buffer=nl.sbuf)
        gu_bf16_buf2 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.bfloat16, buffer=nl.sbuf)
        gu_bf16_buf3 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.bfloat16, buffer=nl.sbuf)
        gu_bf16_bufs = [gu_bf16_buf0, gu_bf16_buf1, gu_bf16_buf2, gu_bf16_buf3]

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

        down0_scale_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.float32, buffer=nl.sbuf)
        down0_scale_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.float32, buffer=nl.sbuf)
        down0_scale_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.float32, buffer=nl.sbuf)
        down0_scale_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.float32, buffer=nl.sbuf)
        down0_scale_bufs = [down0_scale_buf0, down0_scale_buf1, down0_scale_buf2, down0_scale_buf3]

        down1_scale_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.float32, buffer=nl.sbuf)
        down1_scale_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.float32, buffer=nl.sbuf)
        down1_scale_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.float32, buffer=nl.sbuf)
        down1_scale_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.float32, buffer=nl.sbuf)
        down1_scale_bufs = [down1_scale_buf0, down1_scale_buf1, down1_scale_buf2, down1_scale_buf3]

        down0_bf16_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down0_bf16_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down0_bf16_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down0_bf16_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down0_bf16_bufs = [down0_bf16_buf0, down0_bf16_buf1, down0_bf16_buf2, down0_bf16_buf3]

        down1_bf16_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down1_bf16_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down1_bf16_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down1_bf16_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down1_bf16_bufs = [down1_bf16_buf0, down1_bf16_buf1, down1_bf16_buf2, down1_bf16_buf3]

        # ------------------------------------------------------------------
        # Wave 1 buffers (separate to keep DMAs/dequants pipelined across waves)
        # ------------------------------------------------------------------
        gate_up_fp8_w1_buf0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_w1_buf1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_w1_buf2 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_w1_buf3 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        gate_up_fp8_w1_bufs = [gate_up_fp8_w1_buf0, gate_up_fp8_w1_buf1, gate_up_fp8_w1_buf2, gate_up_fp8_w1_buf3]

        # Wave 1 reuses Wave 0 scale buffers to halve SBUF fp32 scale pressure.
        # Correct because wave1 scale DMAs happen AFTER wave0 dequants finish.
        gu_scale_w1_bufs = gu_scale_bufs

        gu_bf16_w1_buf0 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.bfloat16, buffer=nl.sbuf)
        gu_bf16_w1_buf1 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.bfloat16, buffer=nl.sbuf)
        gu_bf16_w1_buf2 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.bfloat16, buffer=nl.sbuf)
        gu_bf16_w1_buf3 = nl.ndarray((_PMAX, H_free, _GU_FLAT), dtype=nl.bfloat16, buffer=nl.sbuf)
        gu_bf16_w1_bufs = [gu_bf16_w1_buf0, gu_bf16_w1_buf1, gu_bf16_w1_buf2, gu_bf16_w1_buf3]

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

        # Share wave 0 scale bufs
        down0_scale_w1_bufs = down0_scale_bufs
        down1_scale_w1_bufs = down1_scale_bufs

        down0_bf16_w1_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down0_bf16_w1_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down0_bf16_w1_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down0_bf16_w1_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down0_bf16_w1_bufs = [down0_bf16_w1_buf0, down0_bf16_w1_buf1, down0_bf16_w1_buf2, down0_bf16_w1_buf3]

        down1_bf16_w1_buf0 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down1_bf16_w1_buf1 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down1_bf16_w1_buf2 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down1_bf16_w1_buf3 = nl.ndarray((_PMAX, H_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down1_bf16_w1_bufs = [down1_bf16_w1_buf0, down1_bf16_w1_buf1, down1_bf16_w1_buf2, down1_bf16_w1_buf3]

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

        # Zero-pad down_full1 fp8 rows I1:I0 (Wave 0 + Wave 1)
        for k_pad in nl.static_range(4):
            nisa.memset(down_full1_fp8_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)
        for k_pad in nl.static_range(4):
            nisa.memset(down_full1_fp8_w1_bufs[k_pad][nl.ds(I1, I1), 0:H_shard], value=0.0)

        # gate_t1_128/up_t1_128: now bf16 (since stationaries are bf16 after dequant).
        # Pre-zero rows I1:I0 to keep the t1 stationary's last 64 columns at 0.
        gate_t1_128 = nl.ndarray((_PMAX, H_free, I0), dtype=nl.bfloat16, buffer=nl.sbuf)
        up_t1_128   = nl.ndarray((_PMAX, H_free, I0), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(gate_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)], value=0.0)
        nisa.memset(up_t1_128[0:_PMAX, 0:H_free, nl.ds(I1, I1)],   value=0.0)

        # Pre-read all 8 expert IDs (8 tiny 4-byte DMAs)
        for k8 in nl.static_range(8):
            nisa.dma_copy(
                dst=eid_all_bufs[k8][0:1, 0:1],
                src=top8_idx.ap(pattern=[[K, 1], [1, 1]], offset=t * K + k8),
            )

        # ==================================================================
        # Phase 1a: Load Wave 0 expert fp8 weights + fat scales (affine_range)
        # ==================================================================
        for k in nl.affine_range(_K_WAVE):
            expert_id = eid_all_bufs[k].ap(pattern=[[1, 1], [1, 1]], offset=0)

            # fp8 gate_up weights
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

            # gate_up FAT scales — stride-0 partition broadcast (full 128P filled by DMA).
            # Costs ~384KB HBM read per expert, but avoids VectorE stream_shuffle overhead.
            nisa.dma_copy(
                dst=gu_scale_bufs[k],
                src=gate_up_scales_fat.ap(
                    pattern=[[0, _PMAX], [_GU_FLAT, H_free], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # fp8 down tile 0 (rows 0:I0=128)
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

            # fp8 down tile 1 (rows I0:I0+I1=64)
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

            # down tile 0 FAT scales — stride-0 partition broadcast
            nisa.dma_copy(
                dst=down0_scale_bufs[k],
                src=down_scales_fat.ap(
                    pattern=[[0, _PMAX], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

            # down tile 1 FAT scales — stride-0 partition broadcast
            nisa.dma_copy(
                dst=down1_scale_bufs[k],
                src=down_scales_fat.ap(
                    pattern=[[0, _PMAX], [1, H_shard]],
                    offset=H + prg_id * H_shard,
                    scalar_offset=expert_id,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # ==================================================================
        # Phase 1b: Load Wave 1 expert fp8 weights + fat scales (affine_range)
        # ==================================================================
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

        # ==================================================================
        # Phase 1c: Bulk dequant Wave 0 — single affine_range, one VectorE
        # op per (k, buffer), fused fp8→bf16 cast and scale multiply.
        # Scale buffers are filled across all 128 partitions by stride-0
        # DMA broadcast, so no stream_shuffle bcast is needed.
        # ==================================================================
        for k in nl.affine_range(_K_WAVE):
            nisa.tensor_tensor(
                dst=gu_bf16_bufs[k],
                data1=gate_up_fp8_bufs[k],
                data2=gu_scale_bufs[k],
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=down0_bf16_bufs[k],
                data1=down_full0_fp8_bufs[k],
                data2=down0_scale_bufs[k],
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=down1_bf16_bufs[k],
                data1=down_full1_fp8_bufs[k],
                data2=down1_scale_bufs[k],
                op=nl.multiply,
            )

        # ==================================================================
        # Phase 1d: Wave 1 scale loads (stride-0 broadcast, shared scale bufs)
        # ==================================================================
        for k in nl.affine_range(_K_WAVE):
            kk = k + 4
            expert_id_w1s = eid_all_bufs[kk].ap(pattern=[[1, 1], [1, 1]], offset=0)
            nisa.dma_copy(
                dst=gu_scale_w1_bufs[k],
                src=gate_up_scales_fat.ap(
                    pattern=[[0, _PMAX], [_GU_FLAT, H_free], [1, _GU_FLAT]],
                    offset=0,
                    scalar_offset=expert_id_w1s,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )
            nisa.dma_copy(
                dst=down0_scale_w1_bufs[k],
                src=down_scales_fat.ap(
                    pattern=[[0, _PMAX], [1, H_shard]],
                    offset=prg_id * H_shard,
                    scalar_offset=expert_id_w1s,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )
            nisa.dma_copy(
                dst=down1_scale_w1_bufs[k],
                src=down_scales_fat.ap(
                    pattern=[[0, _PMAX], [1, H_shard]],
                    offset=H + prg_id * H_shard,
                    scalar_offset=expert_id_w1s,
                    indirect_dim=0,
                ),
                dge_mode=0,
            )

        # ==================================================================
        # Phase 1e: Bulk dequant Wave 1
        # ==================================================================
        for k in nl.affine_range(_K_WAVE):
            nisa.tensor_tensor(
                dst=gu_bf16_w1_bufs[k],
                data1=gate_up_fp8_w1_bufs[k],
                data2=gu_scale_w1_bufs[k],
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=down0_bf16_w1_bufs[k],
                data1=down_full0_fp8_w1_bufs[k],
                data2=down0_scale_w1_bufs[k],
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=down1_bf16_w1_bufs[k],
                data1=down_full1_fp8_w1_bufs[k],
                data2=down1_scale_w1_bufs[k],
                op=nl.multiply,
            )

        # ------------------------------------------------------------------
        # Compute norm_weights / aff_bcast (overlaps with in-flight dequants)
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
        for g in nl.static_range(4):
            nisa.nc_stream_shuffle(
                dst=aff_bcast[nl.ds(g * 32, 32), 0:K],
                src=aff_bcast[0:1, 0:K],
                shuffle_mask=[0] * 32,
            )

        # PSUM allocation for wave 0
        gate_up_psum = nl.ndarray((_PMAX, _K_WAVE * 2 * I_tiles), dtype=nl.float32, buffer=nl.psum)
        down_psum    = nl.ndarray((_PMAX, _K_WAVE * H_free_shard), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        # ==================================================================
        # Phase 2a: Compute Wave 0 (bf16 stationaries)
        # ==================================================================
        for k in nl.static_range(_K_WAVE):
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            # Tile-1 tensor_copy (bf16): copy the 64 valid columns into rows [0:64]
            # of the I0=128 tile; rows [64:128] stay zero from the pre-memset above.
            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gu_bf16_bufs[k][0:_PMAX, 0:H_free, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gu_bf16_bufs[k][0:_PMAX, 0:H_free, nl.ds(I + I0, I1)],
            )

            # Gate/Up matmul (bf16 stationary, dequantized in-place above)
            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gu_bf16_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gu_bf16_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                    else:
                        g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]
                        u_stat = up_t1_128[0:_PMAX, h1, 0:I0]
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                        stationary=g_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )

            # Post-matmul: SiLU(gate) × up — NO scale arg (already applied pre-matmul)
            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            for i_tile in nl.static_range(I_tiles):
                gate_silu = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    gate_silu,
                    op=nl.silu,
                    data=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                )
                up_f32 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    up_f32,
                    op=nl.copy,
                    data=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                )
                nisa.tensor_tensor(inter_f32[0:_PMAX, i_tile:i_tile + 1], gate_silu, up_f32, nl.multiply)

            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            # Down matmul (bf16 stationary)
            for h1_out in nl.affine_range(H_free_shard):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down0_bf16_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down1_bf16_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            # PSUM drain × affinity broadcast (no per-output dequant scale anymore)
            aff_col_buf = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=aff_col_buf, src=aff_bcast[0:_PMAX, k:k + 1])
            aff_col_view = TensorView(aff_col_buf).broadcast(dim=1, size=_H_FREE_SHARD)

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
                aff_col_view.get_view(),
                nl.multiply,
            )

            if k == 0:
                nisa.tensor_copy(
                    dst=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                    src=down_h_all[0:_PMAX, 0:_H_FREE_SHARD],
                )
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                    data1=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                    data2=down_h_all[0:_PMAX, 0:_H_FREE_SHARD],
                    op=nl.add,
                )

        # ==================================================================
        # Phase 2b: Compute Wave 1 (bf16 stationaries from w1 buffers)
        # ==================================================================
        nisa.memset(gate_up_psum, value=0.0)
        nisa.memset(down_psum, value=0.0)

        for k in nl.static_range(_K_WAVE):
            kk = k + 4
            gu_base = k * 2 * I_tiles
            d_base  = k * H_free_shard

            nisa.tensor_copy(
                dst=gate_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gu_bf16_w1_bufs[k][0:_PMAX, 0:H_free, nl.ds(I0, I1)],
            )
            nisa.tensor_copy(
                dst=up_t1_128[0:_PMAX, 0:H_free, 0:I1],
                src=gu_bf16_w1_bufs[k][0:_PMAX, 0:H_free, nl.ds(I + I0, I1)],
            )

            for h1 in nl.affine_range(H_free):
                for i_tile in nl.static_range(I_tiles):
                    if i_tile == 0:
                        g_stat = gu_bf16_w1_bufs[k][0:_PMAX, h1, nl.ds(0, I0)]
                        u_stat = gu_bf16_w1_bufs[k][0:_PMAX, h1, nl.ds(I, I0)]
                    else:
                        g_stat = gate_t1_128[0:_PMAX, h1, 0:I0]
                        u_stat = up_t1_128[0:_PMAX, h1, 0:I0]
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                        stationary=g_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )
                    nisa.nc_matmul(
                        dst=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                        stationary=u_stat,
                        moving=rmsnorm_normed_bf16[0:_PMAX, nl.ds(h1 * T, T)],
                    )

            inter_f32 = nl.ndarray((_PMAX, I_tiles), dtype=nl.float32, buffer=nl.sbuf)
            for i_tile in nl.static_range(I_tiles):
                gate_silu = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    gate_silu,
                    op=nl.silu,
                    data=gate_up_psum[0:_PMAX, gu_base + i_tile:gu_base + i_tile + 1],
                )
                up_f32 = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    up_f32,
                    op=nl.copy,
                    data=gate_up_psum[0:_PMAX, gu_base + I_tiles + i_tile:gu_base + I_tiles + i_tile + 1],
                )
                nisa.tensor_tensor(inter_f32[0:_PMAX, i_tile:i_tile + 1], gate_silu, up_f32, nl.multiply)

            inter_bf16 = nl.ndarray((_PMAX, I_tiles), dtype=inp.dtype, buffer=nl.sbuf)
            nisa.activation(inter_bf16, op=nl.copy, data=inter_f32)

            for h1_out in nl.affine_range(H_free_shard):
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down0_bf16_w1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 0:1],
                )
                nisa.nc_matmul(
                    dst=down_psum[0:_PMAX, d_base + h1_out:d_base + h1_out + 1],
                    stationary=down1_bf16_w1_bufs[k][0:_PMAX, nl.ds(h1_out * _PMAX, _PMAX)],
                    moving=inter_bf16[0:_PMAX, 1:2],
                )

            aff_col_buf = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=aff_col_buf, src=aff_bcast[0:_PMAX, kk:kk + 1])
            aff_col_view = TensorView(aff_col_buf).broadcast(dim=1, size=_H_FREE_SHARD)

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
                aff_col_view.get_view(),
                nl.multiply,
            )

            nisa.tensor_tensor(
                dst=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                data1=output_temp[0:_PMAX, 0:_H_FREE_SHARD],
                data2=down_h_all[0:_PMAX, 0:_H_FREE_SHARD],
                op=nl.add,
            )

    # -----------------------------------------------------------------------
    # Stage 5: Transpose fp32 → bf16, store to HBM   (unchanged from v11c)
    # -----------------------------------------------------------------------
    output = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.shared_hbm)
    out_sb = nl.ndarray((T, H_shard), dtype=inp.dtype, buffer=nl.sbuf)

    for h1 in nl.static_range(H_free_shard):
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


def run(inp, gamma, router_w, gate_up_w, gate_up_scales_fat, down_w, down_scales_fat):
    """Run v12a kernel (Plan A: block FP8 quant + fat scales + bulk pre-matmul dequant).

    Accepts:
      gate_up_w: [E, H, 2*I=384] fp8_e4m3 (as int8)
      gate_up_scales_fat: [E, H_free=16, GU=384] fp32  (FAT replicated)
      down_w: [E, I=192, H=2048] fp8_e4m3 (as int8)
      down_scales_fat: [E, I_tiles=2, H=2048] fp32  (FAT replicated)

    Returns: output [T, H=2048] bf16
    """
    import torch_xla.core.xla_model as xm

    assert gate_up_w.shape == (_E, _H, _GU_FLAT), f"gate_up_w shape {gate_up_w.shape}"
    assert gate_up_scales_fat.shape == (_E, _H_FREE, _GU_FLAT), \
        f"gate_up_scales_fat shape {gate_up_scales_fat.shape}"
    assert down_w.shape == (_E, _I, _H), f"down_w shape {down_w.shape}"
    assert down_scales_fat.shape == (_E, _I_TILES, _H), \
        f"down_scales_fat shape {down_scales_fat.shape}"

    xm.mark_step()

    outputs = qwen3_moe_fused_tkg[2](inp, gamma, router_w, gate_up_w,
                                       gate_up_scales_fat, down_w, down_scales_fat)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs
