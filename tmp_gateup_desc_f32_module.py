import nki
import nki.language as nl
import nki.isa as nisa
from nkilib.core.utils.allocator import SbufManager


_PMAX = 128
_H = 2048
_I = 192
_GU_FLAT = 384
_H_FREE = 16
_GU_P = 96
_GU_LNC_FLAT = 192


@nki.jit
def gateup_desc_f32_debug(inp_normed, expert_id_in, gate_up_w):
    i_lnc = nl.program_id(0)
    dtype = inp_normed.dtype
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope(name="gateup_desc_f32_debug")

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

    gu_psum = nl.ndarray((_GU_P, 2), dtype=nl.float32, buffer=nl.psum)
    for col in nl.static_range(2):
        base = col * _GU_P
        nisa.nc_matmul(
            dst=gu_psum[0:_GU_P, col:col + 1],
            stationary=gu_buf[0:_PMAX, _H_FREE - 1, nl.ds(base, _GU_P)],
            moving=inp_sb[0:_PMAX, _H_FREE - 1:_H_FREE],
            accumulate=False,
        )
        for h_rev in nl.affine_range(1, _H_FREE):
            h1 = _H_FREE - 1 - h_rev
            nisa.nc_matmul(
                dst=gu_psum[0:_GU_P, col:col + 1],
                stationary=gu_buf[0:_PMAX, h1, nl.ds(base, _GU_P)],
                moving=inp_sb[0:_PMAX, h1:h1 + 1],
                accumulate=True,
            )

    local = sbm.alloc_stack((_GU_P, 2), nl.float32, buffer=nl.sbuf, name="local_out")
    nisa.activation(local[0:_GU_P, 0:2], op=nl.copy, data=gu_psum[0:_GU_P, 0:2])

    out = nl.ndarray((_I, 2), dtype=nl.float32, buffer=nl.shared_hbm)
    if i_lnc == 0:
        nisa.dma_copy(dst=out[0:_GU_P, 0:2], src=local[0:_GU_P, 0:2])
    else:
        nisa.dma_copy(dst=out[_GU_P:_I, 0:2], src=local[0:_GU_P, 0:2])

    sbm.close_scope()
    return out
