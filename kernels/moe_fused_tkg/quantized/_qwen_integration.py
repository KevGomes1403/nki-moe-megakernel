"""Integration utilities for wiring v15c MoE TKG and v17_fast_exp attention TKG
into qwen_complete.py.

Contains:
  - ``quantize_and_pack_gate_up``: offline FP8 quantization + v15c packing for
    the gate_up expert weight tensor.
  - ``quantize_down``: offline FP8 quantization for the down expert weight tensor
    (per v14a / v15c's interface — int8 bytes + fp32 scales; no packing).
  - ``qwen3_attn_tkg_v17_wrapper``: @nki.jit wrapper around v17_fast_exp that
    produces output in the same ``[B, 1, H_wo]`` layout as v10e, so qwen_complete's
    attention call site is unchanged.

Note: v17_fast_exp.qwen3_attn_tkg_fused_oproj_v13bc is a sub-function (NOT a JIT
kernel) that writes its output to an SBUF tensor ``out_sb [PMAX, H_wo//PMAX]``
with column-major layout (``out_sb[p, j] = linear_out[j*PMAX + p]``). The wrapper
below opens an SbufManager, calls v17_fast_exp, and performs a per-column
``nisa.dma_copy`` so the final HBM output is ``output [B, 1, H_wo]`` in the same
linear order v10e produces — making the change a drop-in at the Python call site.
"""

import torch

import nki
import nki.language as nl
import nki.isa as nisa
from nkilib.core.utils.allocator import SbufManager

# NOTE: v17_fast_exp (Round 1 Plan B) was discovered to produce all-zero output in
# Phase 3 STEP 1a — the nisa.exponential swap silently corrupts the kernel.
# The original bench PASS was a false positive from rtol=1e-2 tolerance that
# masked zero output given the small (~0.015) attention output magnitude.
# See docs/integration_findings.md. Reverted import to v13bc_sbm_tiled (verified
# non-zero by probe_v13bc_passthrough.py). The 0.7% v17 "gain" is fictional.
from kernels.attn_tkg.agents.v13bc_sbm_tiled import qwen3_attn_tkg_fused_oproj_v13bc
from kernels.moe_fused_tkg.quantized.v15c import pack_gate_up

_PMAX = 128


# ---------------------------------------------------------------------------
# Offline FP8 quantization + v15c packing
# ---------------------------------------------------------------------------

def quantize_and_pack_gate_up(gate_up_bf16: torch.Tensor) -> torch.Tensor:
    """Offline FP8 quantize + v15c-pack the gate_up expert weight tensor.

    Input:
      gate_up_bf16: [E, H, GU=2*I] bf16   (native MoE layout: gate cols 0:I, up cols I:2I)

    Output:
      gate_up_packed_i8: [E, H_FREE+1=17, PMAX=128, GU=384] int8
        (see v15c.pack_gate_up for the exact packed layout)

    Quantization recipe (identical to bench_v14a_trn3.py / bench_v15c_trn3.py):
      - Per-expert, per-output-neuron absmax over the H axis.
      - scale = absmax / 240.0 (clamped to >=1e-6).
      - int8 bytes = bitcast of fp8_e4m3fn(x / scale).
    """
    assert gate_up_bf16.dim() == 3, f"expected [E, H, GU], got {gate_up_bf16.shape}"
    w_f32 = gate_up_bf16.detach().to(torch.float32).contiguous()
    # Per-expert, per-output-neuron absmax over H axis → scales shape [E, GU]
    gate_up_scales = (w_f32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)
    gate_up_w_i8 = (
        w_f32 / gate_up_scales.unsqueeze(1)
    ).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8)
    # Pack weight bytes + fp32 scales into the single int8 blob v15c expects.
    packed = pack_gate_up(gate_up_w_i8, gate_up_scales)
    return packed.contiguous()


def quantize_down(down_bf16: torch.Tensor) -> tuple:
    """Offline FP8 quantize the down expert weight tensor (v15c's down layout).

    Input:
      down_bf16: [E, I, H] bf16

    Output:
      down_w_i8:      [E, I, H] int8 (fp8_e4m3fn bits)
      down_scales_f32: [E, H]   fp32

    NOTE: v15c accepts down_w / down_scales unchanged from v14a (separate int8
    weight and fp32 scales). Round 2 Plan B [F3b] is in flight and may change
    this layout; a follow-up converter change will be needed when [F3b] lands.
    """
    assert down_bf16.dim() == 3, f"expected [E, I, H], got {down_bf16.shape}"
    w_f32 = down_bf16.detach().to(torch.float32).contiguous()
    down_scales = (w_f32.abs().amax(dim=1) / 240.0).clamp(min=1e-6)   # [E, H]
    down_w_i8 = (
        w_f32 / down_scales.unsqueeze(1)
    ).clamp(-240, 240).to(torch.float8_e4m3fn).view(torch.int8)
    return down_w_i8.contiguous(), down_scales.contiguous()


# ---------------------------------------------------------------------------
# v17_fast_exp attention TKG wrapper — preserves v10e's output shape
# ---------------------------------------------------------------------------

@nki.jit
def qwen3_attn_tkg_v17_wrapper(
    hidden_states,   # [B, 1, H]          bf16  (B=1)
    Wq,              # [Hq_tp*d, H]       bf16  [1024, 2048]
    Wk,              # [Hkv_tp*d, H]      bf16  [128,  2048]
    Wv,              # [Hkv_tp*d, H]      bf16  [128,  2048]
    Wo,              # [Hq_tp*d, H_wo]    bf16  [1024, 2048]  transposed o_proj weight
    q_norm_weight,   # [d]                bf16  [128]
    k_norm_weight,   # [d]                bf16  [128]
    K_cache,         # [B, 1, S_prior, d] bf16
    V_cache,         # [B, 1, S_prior, d] bf16
    cos,             # [B, d]             bf16
    sin,             # [B, d]             bf16
    position_ids,    # [B, 1]             int32
):
    """JIT wrapper around v17_fast_exp sub-function.

    Produces output in the same ``[B, 1, H_wo]`` HBM layout as v10e, so qwen_complete's
    attention call site can substitute this for v10e without editing the Python side.

    v17_fast_exp writes into an SBUF tile ``out_sb[p, j] = linear_out[j*PMAX + p]``.
    For each of the NUM_OUT_COLS=16 columns we nc_transpose the [PMAX, 1] slice into
    a [1, PMAX] PSUM/SBUF tensor, then DMA it into ``output[0, 0, j*PMAX:(j+1)*PMAX]``
    to restore the linear element order.
    """
    B = hidden_states.shape[0]      # 1
    H_wo = Wo.shape[1]              # 2048
    PMAX = _PMAX
    NUM_OUT_COLS = H_wo // PMAX     # 16

    # Allocate output directly in [NUM_OUT_COLS, PMAX] shape so the DMA dst
    # is a clean shared_hbm tensor (no .reshape on the destination, which was
    # the root cause of the earlier all-zero-output bug — reshape-on-dst appears
    # not to alias correctly with nisa.dma_copy). We then reshape the returned
    # HBM tensor to [B, 1, H_wo] so the call site contract is unchanged.
    output_flat = nl.ndarray((NUM_OUT_COLS, PMAX), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("wrapper_outer")

    out_sb, k_rope_out, v_out = qwen3_attn_tkg_fused_oproj_v13bc(
        hidden_states, Wq, Wk, Wv, Wo,
        q_norm_weight, k_norm_weight,
        K_cache, V_cache, cos, sin, position_ids,
        out_sb=None,
        sbm=sbm,
    )

    # Option β — Phase 3 STEP 1a.2:
    #   out_sb[p, j] = linear_out[j*PMAX + p]  (v17 emits column-major layout)
    #   transposed[j, p] = out_sb[p, j] = linear_out[j*PMAX + p]
    # One whole-tile [PMAX, NUM_OUT_COLS] → [NUM_OUT_COLS, PMAX] transpose + one
    # nl.store, matching the known-good pattern in probe_nc_transpose_128x16.py.
    # Then we reshape output_flat → [B, 1, H_wo] on return, so linear_out[j*PMAX+p]
    # lands at output[0, 0, j*PMAX+p] in the correct order.
    transposed_psum = nl.ndarray((NUM_OUT_COLS, PMAX), dtype=nl.bfloat16,
                                 buffer=nl.psum, name="transposed_psum")
    nisa.nc_transpose(dst=transposed_psum, data=out_sb[0:PMAX, 0:NUM_OUT_COLS])
    transposed_sb = nl.ndarray((NUM_OUT_COLS, PMAX), dtype=nl.bfloat16,
                               buffer=nl.sbuf, name="transposed_sb")
    nisa.tensor_copy(dst=transposed_sb, src=transposed_psum)
    nl.store(output_flat, transposed_sb)

    sbm.close_scope()   # wrapper_outer
    sbm.close_scope()   # attn_outer (opened inside v17_fast_exp sub-function)

    # Return in v10e-compatible [B, 1, H_wo] shape.
    return output_flat.reshape((B, 1, H_wo)), k_rope_out, v_out
