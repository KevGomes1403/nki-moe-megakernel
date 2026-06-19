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

Under an LNC=n launch the work is sharded by **value-head**: each core owns a contiguous slice
of the q segment, the k segment, and the v segment (3 channel sub-ranges), so the q/k/v tiles a
core's downstream recurrence heads consume are all produced locally. The two cores' writes are
disjoint and tile the full output: ``qkv_out``/``conv_cand`` are bit-identical to a single-core
run, only the producing core differs.

Per tile the window is ``win = concat_free(conv_state[K-1], qkv[T])`` and:
    y[:, t]      = silu( sum_j w[:, j] * win[:, t+j] )      K-tap MAC, weight col per-partition
    conv_cand[t] = win[:, t+1 : t+1+(K-1)]                  K-1 cols ending at token t (slice)
Decode (T=1) is verify with one position, where ``conv_cand[0]`` is the committed new state.

The silu'd result is transposed (head_dim onto the free axis) to head/token-on-partition
``[NT*T, 128]`` and stored as one contiguous bulk DMA into a unified ``qkv_out`` [NT,T,d]; the
caller slices q/k/v (tiles 0..Hk-1, Hk..2Hk-1, 2Hk..). Each core writes its owned tiles into
their global positions; channels are independent (depthwise), so cores never interact.

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


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def conv_load_compute(qkv, conv_state, conv_weight, win, acc, prod, out, NT, K, T, state_w, ch0):
    """Batched load + depthwise MAC + SiLU over one segment's NT channel-tiles (from channel ch0).

    ``win`` [128, NT*(state_w+T)], fp32 ``acc``/``prod`` [128, NT*T]/[128, T*K], and ``out``
    [NT*T, 128] in the I/O dtype are caller-owned. Bulk-loads the per-channel taps and the carried
    state window (NT on partitions, large contiguous free axis) and transposes them into the
    channel-on-partition compute layout, loads ``qkv`` (token-major) with a strided DMA, computes
    ``acc[:, nt, t] = sum_j w*win``, then transposes head_dim onto the free axis and SiLUs into
    ``out[nt*T+t, i]`` (head/token on partition, head_dim innermost) so the store is contiguous.
    Leaves ``win`` packed as ``[conv_state | qkv]`` per tile for candidate slicing. All source reads
    start at channel ``ch0`` so each segment touches only its tile range. ``out`` is written from
    partition 0 (the Activation engine output cannot target a partition offset); the caller places
    the result into any combined buffer.
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
    ) # 4 B DMA packets
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
    ) # 132 B DMA packets

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


def conv_state_store(win, cand_blk, cand_tap, conv_cand, NT, T, state_w, ch0, cand_is_3d):
    """Scatter one segment's per-token candidate windows to its global HBM channels.

    Identical math to ``deltanet_conv``'s candidate store: the window after token t is
    ``win[:, t+1 : t+1+state_w]`` (bit-exact slice). Each column is transposed into
    ``cand_blk[nt, c*state_w + w]`` (TensorE), then scattered to the segment's global channels in
    one DMA per token. ``cand_is_3d`` selects the [T,conv_dim,K-1] candidate stack vs the T=1
    [conv_dim,K-1] state. Caller owns ``cand_blk``/``cand_tap`` (sized at the segment tile count).
    """
    conv_dim = conv_cand.shape[-2]
    img_w = state_w + T
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
        ) # 4 B DMA packets


def conv_qkv_sbuf(qkv, conv_state, conv_weight, key_dim, conv_cand, cand_is_3d):
    """Head-sharded conv -> three partition-0-based SBUF tiles (q/k/v) + scatter conv state to HBM.

    Runs the depthwise MAC + SiLU for this core's 3 owned channel segments (its q/k/v slices) and
    returns SEPARATE partition-0-based tiles ``(q_sbuf, k_sbuf, v_sbuf)`` -- each silu'd conv output
    with partition = ``local_head*T + t`` and free = head_dim j. Keeping them separate (rather than a
    combined out_sbuf) keeps every tile partition-0-based, which is exactly the layout the recurrence's
    ``_load_normed_qk`` consumes (q/k) and the SBUF->SBUF v bridge gathers (v). This core's conv-state
    slices are scattered to ``conv_cand`` (``new_conv_state`` decode / ``conv_cand`` verify) here.

    Returns ``(q_sbuf [Hk_loc*T,128], k_sbuf [Hk_loc*T,128], v_sbuf [Hv_loc*T,128])``.
    """
    T, conv_dim = qkv.shape
    K = conv_weight.shape[1]
    state_w = K - 1
    img_w = state_w + T
    segments, Hv_loc, _ = shard_segments(conv_dim, key_dim)
    Hk_loc = segments[0][1]
    kernel_assert(Hv_loc * T <= P_MAX, "Hv_loc*T must fit the transpose partition axis (<=128)")

    q_sbuf = nl.ndarray((Hk_loc * T, P_MAX), dtype=qkv.dtype, buffer=nl.sbuf)
    k_sbuf = nl.ndarray((Hk_loc * T, P_MAX), dtype=qkv.dtype, buffer=nl.sbuf)
    v_sbuf = nl.ndarray((Hv_loc * T, P_MAX), dtype=qkv.dtype, buffer=nl.sbuf)
    seg_out = [q_sbuf, k_sbuf, v_sbuf]

    for seg in range(len(segments)):
        t0, n_tiles = segments[seg]          # global start tile, tile count for this segment
        ch0 = t0 * P_MAX                     # this segment's global channel offset
        win = nl.ndarray((P_MAX, n_tiles * img_w), dtype=nl.float32, buffer=nl.sbuf)
        acc = nl.ndarray((P_MAX, n_tiles * T), dtype=nl.float32, buffer=nl.sbuf)
        prod = nl.ndarray((P_MAX, T * K), dtype=nl.float32, buffer=nl.sbuf)
        cand_blk = nl.ndarray((n_tiles, P_MAX * state_w), dtype=nl.float32, buffer=nl.sbuf)
        cand_tap = nl.ndarray((n_tiles, P_MAX), dtype=nl.float32, buffer=nl.psum)
        # SiLU writes from partition 0 directly into this segment's own tile (partition-0-based;
        # no combined-buffer placement DMA needed because q/k/v are kept separate).
        conv_load_compute(
            qkv, conv_state, conv_weight, win, acc, prod, seg_out[seg], n_tiles, K, T, state_w, ch0
        )
        conv_state_store(win, cand_blk, cand_tap, conv_cand, n_tiles, T, state_w, ch0, cand_is_3d)

    return q_sbuf, k_sbuf, v_sbuf


def shard_segments(conv_dim, key_dim):
    """The 3 channel segments (q,k,v) this core owns under value-head sharding.

    Returns ``(segments, Hv_loc, NT_loc)`` where each segment is ``(global_start_tile, n_tiles)``
    (channel offset = global_start_tile*128). ``n=1`` yields the full contiguous q|k|v block.
    """
    Hk = key_dim // P_MAX                      # q-heads = k-heads
    Hv = (conv_dim - 2 * key_dim) // P_MAX     # v-heads
    n = nl.num_programs(0)
    c = nl.program_id(0)
    kernel_assert(conv_dim % P_MAX == 0, "conv_dim must be a multiple of 128")
    kernel_assert(key_dim % P_MAX == 0, "key_dim must be a multiple of 128")
    kernel_assert(Hk % n == 0, "q/k-heads must divide across cores")
    kernel_assert(Hv % n == 0, "v-heads must divide across cores")
    Hk_loc = Hk // n
    Hv_loc = Hv // n
    NT_loc = 2 * Hk_loc + Hv_loc
    segments = [
        (c * Hk_loc, Hk_loc),              # q
        (Hk + c * Hk_loc, Hk_loc),         # k
        (2 * Hk + c * Hv_loc, Hv_loc),     # v
    ]
    return segments, Hv_loc, NT_loc


def deltanet_conv(qkv, conv_state, conv_weight, key_dim, qkv_out, conv_cand, cand_is_3d):
    """Depthwise causal conv + SiLU into a unified ``qkv_out`` [NT,T,d], plus candidate windows.

    Value-head sharded under LNC: this core owns 3 contiguous channel sub-ranges (its slice of the
    q, k, and v segments). It runs the per-tile load/compute/store for each segment and writes the
    owned tiles into their global ``qkv_out``/``conv_cand`` positions; the cores' writes are
    disjoint and tile the full output. ``qkv_out`` is [NT,T,d], head_dim innermost (caller slices
    q/k/v). ``conv_cand`` is [T,conv_dim,K-1] (cand_is_3d) or the T=1 [conv_dim,K-1] state.
    """
    T, conv_dim = qkv.shape
    K = conv_weight.shape[1]
    state_w = K - 1
    img_w = state_w + T
    kernel_assert(conv_state.shape[1] == state_w, "conv_state width must be K-1")
    kernel_assert(qkv_out.dtype == qkv.dtype, "qkv_out dtype must match qkv")
    segments, Hv_loc, _ = shard_segments(conv_dim, key_dim)
    kernel_assert(Hv_loc * T <= P_MAX, "Hv_loc*T must fit the transpose partition axis (<=128)")

    # Buffers are allocated per segment at its exact tile count NT (nc_transpose requires the data
    # AP partition stride NT*img_w to equal the tensor free dim, so they can't be over-sized).
    for seg in range(len(segments)):
        t0, NT = segments[seg]               # global start tile, tile count for this segment
        ch0 = t0 * P_MAX                     # this segment's global channel offset
        win = nl.ndarray((P_MAX, NT * img_w), dtype=nl.float32, buffer=nl.sbuf)
        acc = nl.ndarray((P_MAX, NT * T), dtype=nl.float32, buffer=nl.sbuf)
        prod = nl.ndarray((P_MAX, T * K), dtype=nl.float32, buffer=nl.sbuf)
        out = nl.ndarray((NT * T, P_MAX), dtype=qkv_out.dtype, buffer=nl.sbuf)
        cand_blk = nl.ndarray((NT, P_MAX * state_w), dtype=nl.float32, buffer=nl.sbuf)
        cand_tap = nl.ndarray((NT, P_MAX), dtype=nl.float32, buffer=nl.psum)
        conv_load_compute(
            qkv, conv_state, conv_weight, win, acc, prod, out, NT, K, T, state_w, ch0
        )

        # Contiguous bulk store of this segment's tiles into qkv_out[NT,T,d] at its global rows.
        nisa.dma_copy(
            dst=qkv_out.ap(pattern=[[P_MAX, NT * T], [1, P_MAX]], offset=t0 * T * P_MAX),
            src=out[0 : NT * T, 0:P_MAX],
        )

        # Scatter this segment's per-token candidate windows to its global HBM channels.
        conv_state_store(win, cand_blk, cand_tap, conv_cand, NT, T, state_w, ch0, cand_is_3d)


@nki.jit
def deltanet_conv_tkg_fwd(qkv, conv_state, conv_weight, key_dim):
    """Decode / commit (T=1): silu'd conv output and the committed new conv state.

    Returns qkv_out [NT,1,128] (head_dim innermost, NOT l2-normed; caller slices q/k/v = tiles
    0..Hk-1, Hk..2Hk-1, 2Hk.. via key_dim) and new_state [conv_dim, K-1] (== conv_cand[0]).
    Value-head sharded across cores under an LNC=2 launch.
    """
    T, conv_dim = qkv.shape
    state_w = conv_weight.shape[1] - 1
    NT = conv_dim // P_MAX
    qkv_out = nl.ndarray((NT, T, P_MAX), dtype=qkv.dtype, buffer=nl.shared_hbm)
    new_state = nl.ndarray((conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm)
    deltanet_conv(qkv, conv_state, conv_weight, key_dim, qkv_out, new_state, cand_is_3d=False)
    return qkv_out, new_state


@nki.jit
def deltanet_conv_tkg_fwd_cand(qkv, conv_state, conv_weight, key_dim):
    """Speculative verify (T>=2): silu'd conv output and per-position candidate conv states.

    ``conv_cand[t]`` is the carried window after block token t; the host commits
    ``conv_cand[accept_count - 1]`` on a reject. Returns qkv_out [NT,T,128] (caller slices q/k/v
    via key_dim) and conv_cand [T,conv_dim,K-1]. Value-head sharded under an LNC=2 launch.
    """
    T, conv_dim = qkv.shape
    state_w = conv_weight.shape[1] - 1
    NT = conv_dim // P_MAX
    qkv_out = nl.ndarray((NT, T, P_MAX), dtype=qkv.dtype, buffer=nl.shared_hbm)
    conv_cand = nl.ndarray((T, conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm)
    deltanet_conv(qkv, conv_state, conv_weight, key_dim, qkv_out, conv_cand, cand_is_3d=True)
    return qkv_out, conv_cand


@nki.jit
def deltanet_conv_tkg_fwd_sbuf(qkv, conv_state, conv_weight, key_dim):
    """SBUF-output variant (megakernel demo): q/k/v and conv_cand stay resident in SBUF.

    A real megakernel feeds the SBUF tiles straight into the l2norm + recurrence; here they are
    copied to HBM only so callers/tests can read them.

    Returns qkv_out [NT,T,128] (caller slices q/k/v via key_dim) and conv_cand [T,conv_dim,K-1].
    Builds the per-core ``out_sbuf`` [NT_loc*T,128] in [q_loc | k_loc | v_loc] partition order so a
    future recurrence can slice q/k/v from it; the test launches this single-core (n=1 -> the full
    contiguous q|k|v block).
    """
    T, conv_dim = qkv.shape
    K = conv_weight.shape[1]
    state_w = K - 1
    img_w = state_w + T
    NT = conv_dim // P_MAX
    segments, _, NT_loc = shard_segments(conv_dim, key_dim)
    kernel_assert(NT_loc * T <= P_MAX, "NT_loc*T must fit the transpose partition axis (<=128)")

    # out_sbuf is head_dim-on-free [NT_loc*T, 128] in [q_loc | k_loc | v_loc] order (contiguous
    # readout); cand_sbuf stays channel-on-partition, packed by global tile. Both persist across
    # segments; a real megakernel consumes them in place. win/acc/prod are per-segment scratch
    # (sized at the segment's exact tile count, as nc_transpose requires inside conv_load_compute).
    out_sbuf = nl.ndarray((NT_loc * T, P_MAX), dtype=qkv.dtype, buffer=nl.sbuf)
    cand_sbuf = nl.ndarray((P_MAX, NT * T * state_w), dtype=qkv.dtype, buffer=nl.sbuf)

    row0 = 0
    for seg in range(len(segments)):
        t0, n_tiles = segments[seg]          # global start tile, tile count for this segment
        win = nl.ndarray((P_MAX, n_tiles * img_w), dtype=nl.float32, buffer=nl.sbuf)
        acc = nl.ndarray((P_MAX, n_tiles * T), dtype=nl.float32, buffer=nl.sbuf)
        prod = nl.ndarray((P_MAX, T * K), dtype=nl.float32, buffer=nl.sbuf)
        out_seg = nl.ndarray((n_tiles * T, P_MAX), dtype=qkv.dtype, buffer=nl.sbuf)
        conv_load_compute(
            qkv, conv_state, conv_weight, win, acc, prod, out_seg, n_tiles, K, T, state_w,
            t0 * P_MAX,
        )
        # Place this segment's rows into the combined out_sbuf (SBUF->SBUF DMA can target a
        # partition offset; the Activation engine inside conv_load_compute cannot).
        nisa.dma_copy(
            dst=out_sbuf[row0 : row0 + n_tiles * T, 0:P_MAX], src=out_seg[0 : n_tiles * T, 0:P_MAX]
        )
        # Candidate windows packed into cand_sbuf[:, (global_tile*T + t)*state_w ...] (pure slices).
        for nt in range(n_tiles):
            for t in range(T):
                c0 = ((t0 + nt) * T + t) * state_w
                nisa.tensor_copy(
                    dst=cand_sbuf[0:P_MAX, c0 : c0 + state_w],
                    src=win[0:P_MAX, nt * img_w + t + 1 : nt * img_w + t + 1 + state_w],
                )
        row0 += n_tiles * T

    qkv_out = nl.ndarray((NT, T, P_MAX), dtype=qkv.dtype, buffer=nl.shared_hbm)
    conv_cand = nl.ndarray((T, conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=qkv_out.ap(pattern=[[P_MAX, NT_loc * T], [1, P_MAX]], offset=0),
        src=out_sbuf[0 : NT_loc * T, 0:P_MAX],
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
