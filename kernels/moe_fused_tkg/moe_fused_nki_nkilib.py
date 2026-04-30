import nki.isa as nisa
import nki.language as nl
from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg as _moe_tkg
from nkilib.core.router_topk.router_topk import (
    XSBLayout_tp2013__1,
    router_topk as _router_topk,
)
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode, RouterActFnType
from nkilib.core.utils.interleave_copy import interleave_copy

from .rmsnorm_nki import _rmsnorm_sbuf_in_sbuf_out_hoisted

# Hardware constants
_PMAX = 128

# Qwen3-30B-A3B at TP=4 fixed dims
_H = 2048
_E = 128
_K = 8
_I = 192
_H_FREE = _H // _PMAX

# LNC=2 H-sharding constants
_N_PRGS = 2
_H_FREE_SHARD = _H_FREE // _N_PRGS
_H_SHARD = _H_FREE_SHARD * _PMAX


def _flattened_custom_rmsnorm_to_nkilib_layout(rmsnorm_flat, dtype, T, sbm):
    """Convert custom RMSNorm [P, H_free*T] layout to nkilib [P, T, H_free]."""
    rmsnorm_out = sbm.alloc_stack(
        (_PMAX, T, _H_FREE), dtype, buffer=nl.sbuf, name="rmsnorm_out_nkilib_layout"
    )
    for h1 in nl.static_range(_H_FREE):
        nisa.tensor_copy(
            dst=rmsnorm_out[0:_PMAX, 0:T, h1],
            src=rmsnorm_flat[0:_PMAX, h1 * T : h1 * T + T],
        )
    return rmsnorm_out


def _qwen3_moe_sbuf_in_sbuf_out_custom_rms_nkilib(
    inp_sb,               # [PMAX, H_free*T] bf16 in SBUF
    dtype,
    T,
    gamma,                # [1, H] HBM
    router_w,             # [H, E] HBM
    gate_up_w,            # [E, H, 2*I] HBM
    down_w,               # [E, I, H] HBM
    sbm=None,
    gamma_sb_ready=None,
    debug_rmsnorm_out=None,
):
    """
    Custom RMSNorm followed by the same nkilib router_topk and moe_tkg kernels
    used by NxDI's moe_fused_nki_kernel_enabled Qwen TKG path.

    Intermediates remain in SBUF:
      - router logits are not stored to HBM
      - expert_index and expert_affinities are SBUF tensors
      - moe_tkg is called with output_in_sbuf=True
    """
    prg_id = nl.program_id(axis=0)

    rmsnorm_flat = _rmsnorm_sbuf_in_sbuf_out_hoisted(
        inp_sb,
        dtype,
        T,
        gamma,
        sbm=sbm,
        gamma_sb_ready=gamma_sb_ready,
        debug_rmsnorm_out=debug_rmsnorm_out,
    )
    rmsnorm_out = _flattened_custom_rmsnorm_to_nkilib_layout(rmsnorm_flat, dtype, T, sbm)
    sbm.pop_heap()  # rmsnorm_flat

    router_in = rmsnorm_out
    if router_w.dtype != rmsnorm_out.dtype:
        router_in = sbm.alloc_stack(
            (_PMAX, T, _H_FREE), router_w.dtype, buffer=nl.sbuf, name="router_in"
        )
        nisa.tensor_copy(dst=router_in[0:_PMAX, 0:T, 0:_H_FREE], src=rmsnorm_out[0:_PMAX, 0:T, 0:_H_FREE])

    expert_index = nl.ndarray((T, _K), dtype=nl.uint32, buffer=nl.sbuf, name="expert_index")
    expert_affinities = nl.ndarray((T, _E), dtype=nl.float32, buffer=nl.sbuf, name="expert_affinities")
    # nkilib currently requires router_logits to receive a store even when the
    # caller does not need logits. Keep that required scratch write in SBUF.
    router_logits_scratch = nl.ndarray((T, _E), dtype=dtype, buffer=nl.sbuf, name="router_logits_scratch")

    router_outputs = _router_topk(
        x=router_in,
        w=router_w,
        w_bias=None,
        router_logits=router_logits_scratch,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        act_fn=RouterActFnType.SOFTMAX,
        k=_K,
        x_hbm_layout=0,
        x_sb_layout=XSBLayout_tp2013__1,
        router_pre_norm=True,
        norm_topk_prob=True,
        use_column_tiling=True,
        use_indirect_dma_scatter=False,
        return_eager_affi=False,
        use_PE_broadcast_w_bias=False,
        shard_on_tokens=T > 1,
        skip_store_expert_index=False,
        skip_store_router_logits=False,
    )
    expert_index = router_outputs[1]
    expert_affinities = router_outputs[2]

    if T > 1:
        expert_mlp_in = rmsnorm_out
    else:
        expert_mlp_in = nl.ndarray(
            (_PMAX, T, _H_FREE_SHARD), dtype=dtype, buffer=nl.sbuf, name="expert_mlp_in"
        )
        nisa.tensor_copy(
            dst=expert_mlp_in[0:_PMAX, 0:T, 0:_H_FREE_SHARD],
            src=rmsnorm_out[0:_PMAX, 0:T, nl.ds(prg_id * _H_FREE_SHARD, _H_FREE_SHARD)],
        )

    return _moe_tkg(
        hidden_input=expert_mlp_in,
        expert_gate_up_weights=gate_up_w.reshape((_E, _H, 2, _I)),
        expert_down_weights=down_w,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        is_all_expert=False,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        activation_fn=ActFnType.SiLU,
        output_dtype=dtype,
    )


def _store_h_sharded_sbuf_output(out_sb, output, dtype, T, sbm):
    prg_id = nl.program_id(axis=0)
    out_tile_sb = sbm.alloc_stack((T, _H_SHARD), dtype, buffer=nl.sbuf, name="out_tile_sb")

    for t in nl.static_range(T):
        for h1 in nl.static_range(_H_FREE_SHARD):
            tp_psum = nl.ndarray((1, _PMAX), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=tp_psum[0:1, 0:_PMAX],
                data=out_sb[0:_PMAX, t, h1 : h1 + 1],
            )
            interleave_copy(
                dst=out_tile_sb.ap(
                    pattern=[[_H_SHARD, 1], [_H_FREE_SHARD, _PMAX]],
                    offset=t * _H_SHARD + h1,
                ),
                src=tp_psum[0:1, 0:_PMAX],
                index=h1,
            )
        if prg_id == 0:
            nisa.dma_copy(dst=output[t : t + 1, 0:_H_SHARD], src=out_tile_sb[t : t + 1, 0:_H_SHARD])
        else:
            nisa.dma_copy(dst=output[t : t + 1, _H_SHARD:_H], src=out_tile_sb[t : t + 1, 0:_H_SHARD])


def _store_t_sharded_sbuf_output(out_sb, output, dtype, T, sbm):
    prg_id = nl.program_id(axis=0)
    t_first_shard = T // _N_PRGS
    t_second_shard = T - t_first_shard
    t_local = t_first_shard if prg_id == 0 else t_second_shard
    t_offset = 0 if prg_id == 0 else t_first_shard

    out_tile_sb = sbm.alloc_stack((t_local, _H), dtype, buffer=nl.sbuf, name="out_tile_sb")
    for local_t in nl.static_range(t_local):
        global_t = t_offset + local_t
        for h1 in nl.static_range(_H_FREE):
            tp_psum = nl.ndarray((1, _PMAX), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=tp_psum[0:1, 0:_PMAX],
                data=out_sb[0:_PMAX, global_t, h1 : h1 + 1],
            )
            interleave_copy(
                dst=out_tile_sb.ap(
                    pattern=[[_H, 1], [_H_FREE, _PMAX]],
                    offset=local_t * _H + h1,
                ),
                src=tp_psum[0:1, 0:_PMAX],
                index=h1,
            )
        nisa.dma_copy(dst=output[global_t : global_t + 1, 0:_H], src=out_tile_sb[local_t : local_t + 1, 0:_H])


def moe_fused_tkg(inp, gamma, router_w, gate_up_w, down_w):
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)

    T = inp.shape[0]

    sbm.open_scope(name="inp_load")
    inp_2d = inp.reshape((T, _H))
    inp_sb = sbm.alloc_stack(
        (_PMAX, _H_FREE * T), inp.dtype, buffer=nl.sbuf, name="inp_sb"
    )
    for t in nl.static_range(T):
        for h1 in nl.static_range(_H_FREE):
            shard = h1 // _H_FREE_SHARD
            h2 = h1 - shard * _H_FREE_SHARD
            nisa.dma_copy(
                dst=inp_sb[0:_PMAX, h1 * T + t : h1 * T + t + 1],
                src=inp_2d.ap(
                    pattern=[[_H_FREE_SHARD, _PMAX], [1, 1]],
                    offset=t * _H + shard * _H_SHARD + h2,
                ),
                dge_mode=3,
            )

    output = _qwen3_moe_sbuf_in_sbuf_out_custom_rms_nkilib(
        inp_sb,
        inp.dtype,
        T,
        gamma,
        router_w,
        gate_up_w,
        down_w,
        sbm=sbm,
        gamma_sb_ready=None,
    )

    sbm.close_scope()

    return output
