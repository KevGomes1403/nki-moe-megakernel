# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeltaNet causal depthwise conv1d for token generation (decode + speculative verify).

The conv runs before the DeltaNet recurrence in every layer:
    in_proj_qkv -> qkv -> CONV (this kernel) -> silu -> split q/k/v -> l2norm -> recurrence

Depthwise (per-channel, groups=conv_dim) K-tap causal conv seeded by a carried K-1 wide
state window, followed by SiLU, then split per-head into q/k/v with head_dim innermost.
Input ``qkv`` is token-major ``[T, conv_dim]`` (raw in_proj_qkv output, no caller transpose).
The conv_dim channels are laid out ``[ q:0..key_dim | k:key_dim..2*key_dim | v:2*key_dim.. ]``.
Channels lie on the partition axis: conv_dim=2048 splits into NT=16 independent 128-partition
tiles, K/T on the free axis. No matmul on the conv path, no cross-partition reduction -- pure
VectorE.

Per tile the window is ``win = concat_free(conv_state[K-1], qkv[T])`` and:
    y[:, t]      = silu( sum_j w[:, j] * win[:, t+j] )      K-tap MAC, weight col per-partition
    conv_cand[t] = win[:, t+1 : t+1+(K-1)]                  K-1 cols ending at token t (slice)
Decode (T=1) is verify with one position, where ``conv_cand[0]`` is the committed new state.

The silu'd result is transposed (head_dim onto the free axis) to head/token-on-partition
``[NT*T, 128]`` and stored as one contiguous bulk DMA into a unified ``qkv_out`` [NT,T,d]; the
caller slices q/k/v (tiles 0..Hk-1, Hk..2Hk-1, 2Hk..). Under an LNC=2 launch each core owns a
contiguous half of the NT channel-tiles (independent -- depthwise) and runs its own load+store.

The per-channel taps, carried state window, and candidate windows move through HBM as bulk
contiguous DMAs onto NT partitions and are transposed into the channel-on-partition compute
layout with TensorE; ``qkv`` uses a strided DMA. The depthwise MAC is one windowed
``tensor_tensor(multiply)`` (sliding-window image AP, filter broadcast over the T outputs)
feeding a ``tensor_reduce(add)`` over the K-tap axis -- all T output columns at once. The MAC
accumulates in fp32 and SiLU runs in fp32; outputs are cast to the I/O dtype. The candidate
columns are exact copies of the raw projection output, so the state window is bit-exact.

Entrypoints (head_dim d=128) all return a unified qkv_out [NT,T,128] (head_dim innermost;
caller slices q/k/v = tiles 0..Hk-1, Hk..2Hk-1, 2Hk.. via key_dim) plus a state tensor:
  deltanet_conv_tkg_fwd       -> (qkv_out, new_state [conv_dim,K-1])      decode / commit (T=1)
  deltanet_conv_tkg_fwd_cand  -> (qkv_out, conv_cand [T,conv_dim,K-1])    speculative verify (T>=2)
  deltanet_conv_tkg_fwd_sbuf  -> (qkv_out, conv_cand [T,conv_dim,K-1])    SBUF-output (megakernel) demo

Input contract:
    qkv        (T, conv_dim)    raw in_proj_qkv output, token-major: cat(q,k,v) on free axis
    conv_state (conv_dim, K-1)  carried window (from conv_state_buffer)
    conv_weight(conv_dim, K)    per-channel taps, no bias
    key_dim    python int       q/k segment width; Hk=key_dim//128, value_dim=conv_dim-2*key_dim
conv_dim % 128 == 0; K-1 == conv_state width; output dtype follows qkv.dtype.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Partition dimension max (NeuronCore SBUF tile width) == head_dim.
P_MAX = 128


def conv_load_compute(qkv, conv_state, conv_weight, win, acc, prod, out, NT, K, T, state_w, ch0):
    """Batched load + depthwise MAC + SiLU over this core's NT channel-tiles (from channel ch0).

    ``win`` [128, NT*(state_w+T)], fp32 ``acc``/``prod`` [128, NT*T]/[128, T*K], and ``out``
    [NT*T, 128] in the I/O dtype are caller-owned. Bulk-loads the per-channel taps and the carried
    state window (NT on partitions, large contiguous free axis) and transposes them into the
    channel-on-partition compute layout, loads ``qkv`` (token-major) with a strided DMA, computes
    ``acc[:, nt, t] = sum_j w*win``, then transposes head_dim onto the free axis and SiLUs into
    ``out[nt*T+t, i]`` (head/token on partition, head_dim innermost) so the q/k/v store is
    contiguous. Leaves ``win`` packed as ``[conv_state | qkv]`` per tile for candidate slicing.
    All source reads start at channel ``ch0`` so each core touches only its tile range.
    """
    conv_dim = qkv.shape[1]
    img_w = state_w + T  # per-tile window width on the free axis
    prod_w = T * K       # per-tile (W_out * W_f) product width

    # Per-channel taps: w_p[:, nt, j] = conv_weight[nt*128 + c, j], channel c on partition.
    # conv_weight is row-major [(nt*128 + c), j] = [nt*(128*K) + c*K + j], so the whole tensor
    # is one contiguous run per NT block. Bulk-load it onto NT partitions with a 128*K-wide
    # contiguous free axis (large, descriptor-cheap), then transpose each tap into the
    # channel-on-partition layout with TensorE.
    w_p = nl.ndarray((P_MAX, NT * K), dtype=nl.float32, buffer=nl.sbuf)
    w_blk = nl.ndarray((NT, P_MAX * K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=w_blk.ap(pattern=[[P_MAX * K, NT], [1, P_MAX * K]], offset=0),
        src=conv_weight.ap(pattern=[[P_MAX * K, NT], [1, P_MAX * K]], offset=ch0 * K),
    )
    w_tap = nl.ndarray((P_MAX, NT), dtype=nl.float32, buffer=nl.psum)
    for j in range(K):
        # w_blk[:, j::K] is [NT(part), 128(c)] for tap j; transpose -> [128(c), NT].
        nisa.nc_transpose(
            dst=w_tap[0:P_MAX, 0:NT],
            data=w_blk.ap(pattern=[[P_MAX * K, NT], [K, P_MAX]], offset=j),
        )
        nisa.tensor_copy(
            dst=w_p.ap(pattern=[[NT * K, P_MAX], [K, NT]], offset=j),
            src=w_tap[0:P_MAX, 0:NT],
        )

    # win = concat_free(conv_state, qkv). The carried state window (channel c on partition) is the
    # first state_w columns of each tile's window. conv_state is row-major
    # [(nt*128 + c), w] = [nt*(128*state_w) + c*state_w + w], one contiguous run per NT block, so
    # bulk-load it onto NT partitions (state_w*128-wide contiguous free axis) and transpose each
    # window column into the channel-on-partition state slots with TensorE.
    cs_blk = nl.ndarray((NT, P_MAX * state_w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=cs_blk.ap(pattern=[[P_MAX * state_w, NT], [1, P_MAX * state_w]], offset=0),
        src=conv_state.ap(pattern=[[P_MAX * state_w, NT], [1, P_MAX * state_w]], offset=ch0 * state_w),
    )
    cs_tap = nl.ndarray((P_MAX, NT), dtype=nl.float32, buffer=nl.psum)
    for w in range(state_w):
        # cs_blk[:, w::state_w] is [NT(part), 128(c)] for state column w; transpose -> [128(c), NT]
        # and write into win[:, nt*img_w + w] (the w-th state column of every tile).
        nisa.nc_transpose(
            dst=cs_tap[0:P_MAX, 0:NT],
            data=cs_blk.ap(pattern=[[P_MAX * state_w, NT], [state_w, P_MAX]], offset=w),
        )
        nisa.tensor_copy(
            dst=win.ap(pattern=[[NT * img_w, P_MAX], [img_w, NT]], offset=w),
            src=cs_tap[0:P_MAX, 0:NT],
        )

    # qkv is token-major [T, conv_dim]: qkv[t, nt*128 + c] at flat offset t*conv_dim + nt*128 + c.
    # Strided DMA into the channel-on-partition window slots after the state columns.
    nisa.dma_copy(
        dst=win.ap(pattern=[[NT * img_w, P_MAX], [img_w, NT], [1, T]], offset=state_w),
        src=qkv.ap(pattern=[[1, P_MAX], [P_MAX, NT], [conv_dim, T]], offset=ch0),
    )

    # Depthwise MAC: prod[:, t, j] = win[:, t+j] * w[:, j] (sliding window, filter broadcast over
    # the T outputs), then reduce the K-tap axis -> acc[:, nt, t] = sum_j w[:,j]*win[:,t+j].
    for nt in nl.affine_range(NT):
        nisa.tensor_tensor(
            dst=prod.ap(pattern=[[prod_w, P_MAX], [1, prod_w]], offset=0),
            data1=win.ap(pattern=[[NT * img_w, P_MAX], [1, T], [1, K]], offset=nt * img_w),
            data2=w_p.ap(pattern=[[NT * K, P_MAX], [0, T], [1, K]], offset=nt * K),
            op=nl.multiply,
        )
        nisa.tensor_reduce(
            dst=acc.ap(pattern=[[NT * T, P_MAX], [1, T]], offset=nt * T),
            data=prod.ap(pattern=[[prod_w, P_MAX], [K, T], [1, K]], offset=0),
            op=nl.add,
            axis=2,
        )

    # SiLU over all tiles/positions, fused with a head_dim transpose: nc_transpose moves head_dim
    # (i) off the partition axis onto the contiguous free axis (head/token = nt*T+t on partition),
    # and the SiLU activation doubles as the PSUM->SBUF cast to the I/O dtype. This lets store_qkv
    # write q/k/v as a contiguous bulk DMA instead of a per-element transposed store.
    acc_t = nl.ndarray((NT * T, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=acc_t[0 : NT * T, 0:P_MAX], data=acc[0:P_MAX, 0 : NT * T])
    nisa.activation(dst=out[0 : NT * T, 0:P_MAX], op=nl.silu, data=acc_t[0 : NT * T, 0:P_MAX])


def deltanet_conv(qkv, conv_state, conv_weight, qkv_out, conv_cand, cand_is_3d):
    """Depthwise causal conv + SiLU into a unified ``qkv_out`` [NT,T,d], plus candidate windows.

    SPMD-sharded under LNC: each core owns a contiguous ``NT // num_programs`` slice of the
    conv_dim channel-tiles and runs all loads/compute/store for that slice (channels are
    independent -- depthwise). ``qkv_out`` is [NT,T,d], head_dim innermost; the caller slices it
    into q/k/v. ``conv_cand`` is [T,conv_dim,K-1] (cand_is_3d) or the T=1 [conv_dim,K-1] state.
    """
    T, conv_dim = qkv.shape
    K = conv_weight.shape[1]
    state_w = K - 1
    img_w = state_w + T
    assert conv_dim % P_MAX == 0, "conv_dim must be a multiple of 128"
    assert conv_state.shape[1] == state_w, "conv_state width must be K-1"
    NT_full = conv_dim // P_MAX
    n_cores = nl.num_programs(0)
    assert NT_full % n_cores == 0, "channel-tiles must divide across cores"
    NT = NT_full // n_cores                  # tiles owned by this core
    assert NT * T <= P_MAX, "NT*T must fit the transpose partition axis (<=128)"
    t0 = nl.program_id(0) * NT               # this core's first tile
    ch0 = t0 * P_MAX                         # ... and its channel offset

    win = nl.ndarray((P_MAX, NT * img_w), dtype=nl.float32, buffer=nl.sbuf)
    acc = nl.ndarray((P_MAX, NT * T), dtype=nl.float32, buffer=nl.sbuf)
    prod = nl.ndarray((P_MAX, T * K), dtype=nl.float32, buffer=nl.sbuf)
    out = nl.ndarray((NT * T, P_MAX), dtype=qkv_out.dtype, buffer=nl.sbuf)

    conv_load_compute(qkv, conv_state, conv_weight, win, acc, prod, out, NT, K, T, state_w, ch0)

    # Contiguous bulk store of this core's tiles into qkv_out[NT,T,d] (head_dim already innermost).
    nisa.dma_copy(
        dst=qkv_out.ap(pattern=[[P_MAX, NT * T], [1, P_MAX]], offset=t0 * T * P_MAX),
        src=out[0 : NT * T, 0:P_MAX],
    )

    # Candidate window after token t: win[:, t+1 : t+1+state_w] (bit-exact slice). Transpose each
    # column into cand_blk[nt, c*state_w + w] (TensorE), then scatter this core's tiles in one DMA.
    cand_blk = nl.ndarray((NT, P_MAX * state_w), dtype=nl.float32, buffer=nl.sbuf)
    cand_tap = nl.ndarray((NT, P_MAX), dtype=nl.float32, buffer=nl.psum)
    for t in range(T):
        cand_offset = (t * conv_dim * state_w if cand_is_3d else 0) + ch0 * state_w
        for w in range(state_w):
            nisa.nc_transpose(
                dst=cand_tap[0:NT, 0:P_MAX],
                data=win.ap(pattern=[[NT * img_w, P_MAX], [img_w, NT]], offset=t + 1 + w),
            )
            nisa.tensor_copy(
                dst=cand_blk.ap(pattern=[[P_MAX * state_w, NT], [state_w, P_MAX]], offset=w),
                src=cand_tap[0:NT, 0:P_MAX],
            )
        nisa.dma_copy(
            dst=conv_cand.ap(
                pattern=[[P_MAX * state_w, NT], [1, P_MAX * state_w]], offset=cand_offset
            ),
            src=cand_blk.ap(pattern=[[P_MAX * state_w, NT], [1, P_MAX * state_w]], offset=0),
        )


@nki.jit
def deltanet_conv_tkg_fwd(qkv, conv_state, conv_weight, key_dim):
    """Decode / commit (T=1): silu'd conv output and the committed new conv state.

    Returns qkv_out [NT,1,128] (head_dim innermost, NOT l2-normed; caller slices q/k/v = tiles
    0..Hk-1, Hk..2Hk-1, 2Hk.. via key_dim) and new_state [conv_dim, K-1] (== conv_cand[0]).
    Shards across cores under an LNC=2 launch.
    """
    T, conv_dim = qkv.shape
    state_w = conv_weight.shape[1] - 1
    NT = conv_dim // P_MAX
    qkv_out = nl.ndarray((NT, T, P_MAX), dtype=qkv.dtype, buffer=nl.shared_hbm)
    new_state = nl.ndarray((conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm)
    deltanet_conv(qkv, conv_state, conv_weight, qkv_out, new_state, cand_is_3d=False)
    return qkv_out, new_state


@nki.jit
def deltanet_conv_tkg_fwd_cand(qkv, conv_state, conv_weight, key_dim):
    """Speculative verify (T>=2): silu'd conv output and per-position candidate conv states.

    ``conv_cand[t]`` is the carried window after block token t; the host commits
    ``conv_cand[accept_count - 1]`` on a reject. Returns qkv_out [NT,T,128] (caller slices q/k/v
    via key_dim) and conv_cand [T,conv_dim,K-1]. Shards across cores under an LNC=2 launch.
    """
    T, conv_dim = qkv.shape
    state_w = conv_weight.shape[1] - 1
    NT = conv_dim // P_MAX
    qkv_out = nl.ndarray((NT, T, P_MAX), dtype=qkv.dtype, buffer=nl.shared_hbm)
    conv_cand = nl.ndarray((T, conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm)
    deltanet_conv(qkv, conv_state, conv_weight, qkv_out, conv_cand, cand_is_3d=True)
    return qkv_out, conv_cand


@nki.jit
def deltanet_conv_tkg_fwd_sbuf(qkv, conv_state, conv_weight, key_dim):
    """SBUF-output variant (megakernel demo): q/k/v and conv_cand stay resident in SBUF.

    A real megakernel feeds the SBUF tiles straight into the l2norm + recurrence; here they are
    copied to HBM only so callers/tests can read them.

    Returns qkv_out [NT,T,128] (caller slices q/k/v via key_dim) and conv_cand [T,conv_dim,K-1].
    Single-core (the megakernel demo path).
    """
    T, conv_dim = qkv.shape
    K = conv_weight.shape[1]
    state_w = K - 1
    img_w = state_w + T
    NT = conv_dim // P_MAX
    assert NT * T <= P_MAX, "NT*T must fit the transpose partition axis (<=128)"

    # out_sbuf is head_dim-on-free [NT*T, 128] (contiguous readout store); cand_sbuf stays
    # channel-on-partition. A real megakernel consumes these SBUF tiles in place.
    win = nl.ndarray((P_MAX, NT * img_w), dtype=nl.float32, buffer=nl.sbuf)
    acc = nl.ndarray((P_MAX, NT * T), dtype=nl.float32, buffer=nl.sbuf)
    prod = nl.ndarray((P_MAX, T * K), dtype=nl.float32, buffer=nl.sbuf)
    out_sbuf = nl.ndarray((NT * T, P_MAX), dtype=qkv.dtype, buffer=nl.sbuf)
    cand_sbuf = nl.ndarray((P_MAX, NT * T * state_w), dtype=qkv.dtype, buffer=nl.sbuf)

    conv_load_compute(qkv, conv_state, conv_weight, win, acc, prod, out_sbuf, NT, K, T, state_w, 0)

    # Candidate windows packed into cand_sbuf[:, (nt*T + t)*state_w ...] (pure slices).
    for nt in nl.affine_range(NT):
        for t in range(T):
            c0 = (nt * T + t) * state_w
            nisa.tensor_copy(
                dst=cand_sbuf[0:P_MAX, c0 : c0 + state_w],
                src=win[0:P_MAX, nt * img_w + t + 1 : nt * img_w + t + 1 + state_w],
            )

    qkv_out = nl.ndarray((NT, T, P_MAX), dtype=qkv.dtype, buffer=nl.shared_hbm)
    conv_cand = nl.ndarray((T, conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=qkv_out.ap(pattern=[[P_MAX, NT * T], [1, P_MAX]], offset=0),
        src=out_sbuf[0 : NT * T, 0:P_MAX],
    )
    for t in range(T):
        nisa.dma_copy(
            dst=conv_cand.ap(
                pattern=[[state_w, P_MAX], [P_MAX * state_w, NT], [1, state_w]],
                offset=t * conv_dim * state_w,
            ),
            src=cand_sbuf.ap(
                pattern=[[NT * T * state_w, P_MAX], [T * state_w, NT], [1, state_w]],
                offset=t * state_w,
            ),
        )
    return qkv_out, conv_cand
