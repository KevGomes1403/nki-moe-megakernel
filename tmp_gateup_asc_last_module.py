import nki
import nki.language as nl
import nki.isa as nisa
from nkilib.core.utils.allocator import SbufManager

from nki.backends.mlir_tracer.context import get_current_context
from nki.backends.mlir_tracer.isa_emit import emit_matmul
from nki.compiler._internal.dialects.nisa import PsumAccumulateFlags as _PsumAccumulateFlags
from nki.language.tensor import tensor_ap_view


def _nc_matmul_last(dst, stationary, moving):
    ctx = get_current_context()
    loc = ctx.setup_loc(None)
    with loc, ctx.insertion_point:
        emit_matmul(
            dst=tensor_ap_view(dst),
            stationary=tensor_ap_view(stationary),
            moving=tensor_ap_view(moving),
            psum_accumulate_flags=_PsumAccumulateFlags.LastMatmul,
            ip=ctx.insertion_point,
            loc=loc,
            context=ctx.context,
        )


_PMAX = 128
_H = 2048
_I = 192
_GU_FLAT = 384
_H_FREE = 16
_GU_P = 96
_GU_LNC_FLAT = 192


@nki.jit
def gateup_asc_last_outputs(inp_normed, expert_id_in, gate_up_w):
    i_lnc = nl.program_id(0)
    dtype = inp_normed.dtype
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope(name="gateup_asc_last_outputs")

    inp_2d = inp_normed.reshape((1, _H))
    inp_flat = inp_2d.reshape((_H_FREE, _PMAX))
    inp_flat_sb = sbm.alloc_stack((_H_FREE, _PMAX), dtype, buffer=nl.sbuf, name="inp_flat_sb")
    nisa.dma_copy(dst=inp_flat_sb, src=inp_flat, dge_mode=3)
    inp_trans_psum = nl.ndarray((_PMAX, _H_FREE), dtype=dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=inp_trans_psum, data=inp_flat_sb)
    inp_sb = sbm.alloc_stack((_PMAX, _H_FREE), dtype, buffer=nl.sbuf, name="inp_sb")
    nisa.activation(inp_sb, op=nl.copy, data=inp_trans_psum)

    expert_id = sbm.alloc_stack((1, 1), nl.uint32, buffer=nl.sbuf, name="expert_id")
    nisa.dma_copy(dst=expert_id[0:1, 0:1], src=expert_id_in.reshape((1, 1))[0:1, 0:1])
    eid = expert_id.ap(pattern=[[1, 1], [1, 1]], offset=0)

    gu_buf = sbm.alloc_stack((_PMAX, _H_FREE, _GU_LNC_FLAT), gate_up_w.dtype, buffer=nl.sbuf, name="gu_buf")
    nisa.dma_copy(
        dst=gu_buf[0:_PMAX, 0:_H_FREE, 0:_GU_P],
        src=gate_up_w.ap(
            pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, _H_FREE], [1, _GU_P]],
            offset=i_lnc * _GU_P,
            scalar_offset=eid,
            indirect_dim=0,
        ),
        dge_mode=0,
    )
    nisa.dma_copy(
        dst=gu_buf[0:_PMAX, 0:_H_FREE, _GU_P:_GU_LNC_FLAT],
        src=gate_up_w.ap(
            pattern=[[_GU_FLAT, _PMAX], [_PMAX * _GU_FLAT, _H_FREE], [1, _GU_P]],
            offset=_I + i_lnc * _GU_P,
            scalar_offset=eid,
            indirect_dim=0,
        ),
        dge_mode=0,
    )

    ps = nl.ndarray((_GU_P, 2), dtype=nl.float32, buffer=nl.psum)
    for col in nl.static_range(2):
        base = col * _GU_P
        nisa.nc_matmul(
            dst=ps[0:_GU_P, col:col + 1],
            stationary=gu_buf[0:_PMAX, 0, nl.ds(base, _GU_P)],
            moving=inp_sb[0:_PMAX, 0:1],
            accumulate=False,
        )
        for h1 in nl.affine_range(1, _H_FREE - 1):
            nisa.nc_matmul(
                dst=ps[0:_GU_P, col:col + 1],
                stationary=gu_buf[0:_PMAX, h1, nl.ds(base, _GU_P)],
                moving=inp_sb[0:_PMAX, h1:h1 + 1],
                accumulate=True,
            )
        _nc_matmul_last(
            dst=ps[0:_GU_P, col:col + 1],
            stationary=gu_buf[0:_PMAX, _H_FREE - 1, nl.ds(base, _GU_P)],
            moving=inp_sb[0:_PMAX, _H_FREE - 1:_H_FREE],
        )

    local = sbm.alloc_stack((_GU_P, 4), dtype, buffer=nl.sbuf, name="local")
    nisa.activation(local[0:_GU_P, 0:1], op=nl.copy, data=ps[0:_GU_P, 0:1])
    nisa.activation(local[0:_GU_P, 1:2], op=nl.copy, data=ps[0:_GU_P, 1:2])
    nisa.activation(local[0:_GU_P, 2:3], op=nl.silu, data=ps[0:_GU_P, 0:1])
    nisa.tensor_tensor(local[0:_GU_P, 3:4], local[0:_GU_P, 2:3], ps[0:_GU_P, 1:2], nl.multiply)

    out = nl.ndarray((_I, 4), dtype=dtype, buffer=nl.shared_hbm)
    if i_lnc == 0:
        nisa.dma_copy(dst=out[0:_GU_P, 0:4], src=local[0:_GU_P, 0:4])
    else:
        nisa.dma_copy(dst=out[_GU_P:_I, 0:4], src=local[0:_GU_P, 0:4])
    sbm.close_scope()
    return out
