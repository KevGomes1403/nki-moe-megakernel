# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeltaNet gated delta-rule recurrence for token generation (decode + speculative verify).

One launch processes all (batch, head) pairs. Each head's 128x128 recurrent state is packed
into a wide SBUF tile ``Sp[128, W]`` (W = BH*128) as ``Sp[i, h*128+j] = state_h[i, j]`` --
key index ``i`` on the 128 partitions, ``(head, value index j)`` on the free axis -- and the
gated delta rule is iterated over the S-token block with the state resident in SBUF (no
per-token HBM round-trip), one wide op per step.

Per token (math identical to NeuronGatedDeltaNet._recurrent_step):
    decay   Sp = src * exp(g)              per-head scalar, free-broadcast across j
    read    kv = sum_i Sp[i,:] * k_h[i]    partition-reduce
    delta   d  = (v - kv) * beta
    update  Sp[i,:] += k_h[i] * d
    output  o  = sum_i Sp[i,:] * q_h[i]    partition-reduce

Implementation notes:
  * State is double-buffered (S0/S1 ping-pong): token t's output reads its working tile while
    token t+1's out-of-place decay writes the other, breaking the inter-token write-after-read
    on the state so the scheduler can overlap them.
  * Per-head scalars (decay, beta) are partition-broadcast at width BH (one small matmul) and
    free-broadcast across j. beta is folded into the update's k-view (``k*beta``) off the serial
    chain, so the delta step is a single subtract.
  * exp(g) for every (head, token) is computed once before the loop (one activation-table load).
  * Partition broadcast/reduce results stay in PSUM and are consumed directly by the next Vector
    op; only the final output needs a PSUM->SBUF copy (its HBM/SBUF sink cannot source PSUM).
  * The read/output reduces are pipelined over 512-wide tiles so each reduce matmul overlaps the
    next tile's multiply.
  * Each state write is one 3D store DMA.

Entrypoints:
  deltanet_tkg_fwd        -> (output, final_state)       decode / commit
  deltanet_tkg_fwd_state  -> (output, candidate_states)  speculative verify
  deltanet_tkg_fwd_sbuf   -> (output_sbuf, final_state)  SBUF-output (megakernel) demo

``candidate_states[bh, t]`` is the state after consuming block token t; on a speculative reject
the host selects ``candidate_states[:, accept_count - 1]``. ``final_state`` equals the last
candidate.

Input contract (all f32):
    query   (BH, S, 128)   l2-normed and scaled by 1/sqrt(k_dim)
    key     (BH, S, 128)   l2-normed
    value   (BH, S, 128)
    g_in    (BH, S, 1)     raw per-token log-decay (exp'd in-kernel)
    beta_in (BH, S, 1)     per-token write-gate
    init_state (BH, 128, 128)
k_dim = v_dim = 128 = P_MAX.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Partition dimension max (NeuronCore SBUF tile width) = k_dim = v_dim.
P_MAX = 128

# Per-matmul moving free width = one PSUM bank (512 f32). Broadcast/reduce matmuls tile at this
# width into a wide PSUM tile that the next Vector op reads whole, so tiling adds no copies.
_PSUM_FMAX = 512


def div_ceil(n, d):
    """Ceil division for tile-count computation."""
    return (n + d - 1) // d


def partition_broadcast_psum(row_1W, width, ones_row, psum):
    """Partition-broadcast a (1, width) row to a (128, width) PSUM tile (all partitions equal).

    ones-row nc_matmul on the Tensor engine (result[m, n] = row[0, n]); the result is left in
    PSUM for direct consumption. Tiled at _PSUM_FMAX per matmul; accumulate=False overwrites
    (PSUM tiles are reused across steps).
    """
    for c in nl.static_range(div_ceil(width, _PSUM_FMAX)):
        c0 = c * _PSUM_FMAX
        tile_w = min(_PSUM_FMAX, width - c0)
        nisa.nc_matmul(
            dst=psum[0:P_MAX, c0 : c0 + tile_w],
            stationary=ones_row[0:1, 0:P_MAX],
            moving=row_1W[0:1, c0 : c0 + tile_w],
            accumulate=False,
        )


def mul_then_reduce_tiled(Sp, mat, BH, dim, W, mul_buf, ones_col, red_p):
    """Partition-reduce of ``Sp * free_broadcast(mat)``, pipelined over the 512-wide free-tiles.

    Computes ``red_p[0, h*dim+j] = sum_i Sp[i, h*dim+j] * mat[i, h]`` (read uses mat = k_mat,
    output uses mat = q_mat). ``mat`` is the per-head ``[128, BH]`` load free-broadcast across j.
    Per tile c (512 wide = 4 heads): the [128,512] multiply, then that tile's reduce matmul.
    Issuing mul_0 -> reduce_0 -> mul_1 -> reduce_1 lets mul_1 (Vector) overlap reduce_0 (Tensor):
    the tiles touch disjoint columns of ``mul_buf`` and disjoint PSUM banks of ``red_p``, hiding
    the reduce matmul behind the next multiply. ``red_p`` is consumed PSUM-direct downstream.
    """
    for c in nl.static_range(div_ceil(W, _PSUM_FMAX)):
        c0 = c * _PSUM_FMAX
        tile_w = min(_PSUM_FMAX, W - c0)
        h0 = c0 // dim  # first head in this tile
        nh = tile_w // dim  # heads in this tile
        # mat_view_c[i, (h-h0)*dim + j] = mat[i, h0+h]; free-broadcast across j over this tile's
        # heads. mat is [128, BH] (partition stride BH); offset h0 shifts to head h0.
        mat_view_c = mat.ap(pattern=[[BH, P_MAX], [1, nh], [0, dim]], offset=h0)
        nisa.tensor_tensor(
            dst=mul_buf[0:P_MAX, c0 : c0 + tile_w],
            data1=Sp[0:P_MAX, c0 : c0 + tile_w],
            data2=mat_view_c,
            op=nl.multiply,
        )
        nisa.nc_matmul(
            dst=red_p[0:1, c0 : c0 + tile_w],
            stationary=ones_col[0:P_MAX, 0:1],
            moving=mul_buf[0:P_MAX, c0 : c0 + tile_w],
            accumulate=False,
        )


def _write_state(state_hbm, Sp, BH, dim, W, base_off, head_stride):
    """Write the packed state to HBM as one 3D store DMA: ``state[..., h, i, j] <- Sp[i, h*dim+j]``.

    Walks (partition i, head h, value index j). ``head_stride`` is the destination element
    distance between heads: dim*dim for the final state, seq_len*dim*dim for the candidate stack.
    """
    nisa.dma_copy(
        dst=state_hbm.ap(
            pattern=[[dim, P_MAX], [head_stride, BH], [1, dim]], offset=base_off
        ),
        src=Sp.ap(pattern=[[W, P_MAX], [dim, BH], [1, dim]], offset=0),
    )


def gated_delta_rule_tkg(
    query,
    key,
    value,
    g_in,
    beta_in,
    init_state,
    out,
    state_hbm,
    out_sbuf_tile,
    write_candidates,
    out_in_sbuf,
):
    """Batched-over-heads gated delta-rule recurrence (all heads at once).

    Packs every head's 128x128 state into a wide ``[128, W]`` tile (W = BH*128), double-buffered
    (S0/S1 ping-pong), and iterates the gated delta rule over ``S`` tokens. Per-token output goes
    to ``out`` (HBM, BH,S,dim) or ``out_sbuf_tile`` (SBUF, S,W) per ``out_in_sbuf``. When
    ``write_candidates`` the state after every token is written to ``state_hbm`` (BH,S,dim,dim);
    otherwise only the final state is written (BH,dim,dim).
    """
    BH, seq_len, dim = query.shape
    W = BH * dim

    # ones operands for the broadcast/reduce matmuls (alloc once).
    ones_col = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_col, value=1.0)
    ones_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_row, value=1.0)

    # Reusable per-token SBUF tiles (alloc once). SK/SQ hold the read/output per-tile products.
    SK = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.sbuf)
    SQ = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.sbuf)
    outer = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.sbuf)
    delta_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
    O_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
    kbeta_mat = nl.ndarray((P_MAX, BH), dtype=nl.float32, buffer=nl.sbuf)

    # Hoisted decay: exp(g) for ALL (head, token) in one activation, so the exp table loads once.
    # g_all[0, t*BH + h] = g_in[h, t, 0]  (flat h*S + t).
    g_all = nl.ndarray((1, seq_len * BH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=g_all.ap(pattern=[[seq_len * BH, 1], [BH, seq_len], [1, BH]], offset=0),
        src=g_in.ap(pattern=[[seq_len * BH, 1], [1, seq_len], [seq_len, BH]], offset=0),
    )
    exp_g_all = nl.ndarray((1, seq_len * BH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_g_all, op=nl.exp, data=g_all, bias=None, scale=1.0)

    # Reusable per-token PSUM tiles (matmul outputs; alloc once). Steps are sequential so a tile
    # frees before the next reuses its banks. red_p's two 512-tiles land in separate banks, so
    # the interleaved read/output reduces overlap.
    eg_p = nl.ndarray((P_MAX, BH), dtype=nl.float32, buffer=nl.psum)
    beta_p = nl.ndarray((P_MAX, BH), dtype=nl.float32, buffer=nl.psum)
    bcast_p = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.psum)
    red_p = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.psum)

    # Two ping-pong state tiles. Each token reads the previous buffer and writes the other; the
    # out-of-place decay breaks the inter-token output-read vs. next-decay-write hazard.
    S0 = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.sbuf)
    S1 = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.sbuf)
    bufs = [S0, S1]

    # Seed the state into S0: S0[i, h*dim+j] <- init_state[h, i, j]. Token 0 reads S0, writes S1.
    nisa.dma_copy(
        dst=S0.ap(pattern=[[W, P_MAX], [dim, BH], [1, dim]], offset=0),
        src=init_state.ap(
            pattern=[[dim, P_MAX], [dim * dim, BH], [1, dim]], offset=0
        ),
    )

    for t in nl.static_range(seq_len):
        # Ping-pong (t % 2 is compile-time): src = previous final state (read-only this token),
        # Sp = this token's working state (holds the state after token t).
        src = bufs[t % 2]
        Sp = bufs[(t + 1) % 2]

        # Per-token loads, all heads at once.
        # k_mat / q_mat [128, BH]: [i, h] <- key/query[h, t, i].
        k_mat = nl.ndarray((P_MAX, BH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k_mat,
            src=key.ap(pattern=[[1, P_MAX], [seq_len * dim, BH]], offset=t * dim),
        )
        q_mat = nl.ndarray((P_MAX, BH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_mat,
            src=query.ap(pattern=[[1, P_MAX], [seq_len * dim, BH]], offset=t * dim),
        )
        # v_row [1, W]: [0, h*dim+j] <- value[h, t, j]  (v on the free axis: kv/delta index j).
        v_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=v_row.ap(pattern=[[W, 1], [dim, BH], [1, dim]], offset=0),
            src=value.ap(
                pattern=[[W, 1], [seq_len * dim, BH], [1, dim]], offset=t * dim
            ),
        )
        # beta_vec [1, BH]: [0, h] <- beta_in[h, t, 0].
        beta_vec = nl.ndarray((1, BH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=beta_vec,
            src=beta_in.ap(pattern=[[seq_len, 1], [seq_len, BH]], offset=t),
        )

        # Per-token decay slice of the precomputed table: exp_g_vec[0, h] = exp(g_{h,t}).
        exp_g_vec = exp_g_all[0:1, t * BH : (t + 1) * BH]

        # Fold beta into k off the serial chain: kbeta_mat[i, h] = k[i, h] * beta_h. beta is
        # per-head (constant over i), so partition-broadcast it (Tensor) then one narrow multiply.
        # Depends only on loads, so it overlaps the read/reduce. Kbeta_view free-broadcasts over j.
        partition_broadcast_psum(beta_vec, BH, ones_row, beta_p)
        nisa.tensor_tensor(
            dst=kbeta_mat, data1=k_mat, data2=beta_p[0:P_MAX, 0:BH], op=nl.multiply
        )
        Kbeta_view = kbeta_mat.ap(pattern=[[BH, P_MAX], [1, BH], [0, dim]], offset=0)

        # Step 1: decay  Sp = src * exp(g)  (out-of-place: src stays readable by token t-1's
        # output, breaking the WAR). Per-head scalar broadcast at width BH into eg_p (PSUM), read
        # via a free-broadcast view over j -- no wide decay tile, no PSUM->SBUF copy.
        partition_broadcast_psum(exp_g_vec, BH, ones_row, eg_p)
        eg_view = eg_p.ap(pattern=[[BH, P_MAX], [1, BH], [0, dim]], offset=0)
        nisa.tensor_tensor(dst=Sp, data1=src, data2=eg_view, op=nl.multiply)

        # Step 2: read  kv[0, h*dim+j] = sum_i Sp[i, :] * k_h[i]  (into red_p, PSUM; pipelined).
        mul_then_reduce_tiled(Sp, k_mat, BH, dim, W, SK, ones_col, red_p)

        # Step 3: delta = v - kv  (single Vector op; *beta is folded into the update). kv read
        # PSUM-direct from red_p.
        nisa.tensor_tensor(
            dst=delta_row, data1=v_row, data2=red_p[0:1, 0:W], op=nl.subtract
        )

        # Step 4: update  Sp[i, :] += (k_h[i]*beta_h) * delta_h[j]. Broadcast delta into bcast_p
        # (PSUM), multiply by Kbeta_view PSUM-direct, accumulate into Sp.
        partition_broadcast_psum(delta_row, W, ones_row, bcast_p)
        nisa.tensor_tensor(
            dst=outer, data1=Kbeta_view, data2=bcast_p[0:P_MAX, 0:W], op=nl.multiply
        )
        nisa.tensor_tensor(dst=Sp, data1=Sp, data2=outer, op=nl.add)

        # Step 5: output  O_row[0, h*dim+j] = sum_i Sp[i, :] * q_h[i]  (pipelined, like the read).
        # One PSUM->SBUF copy (the output sink cannot source PSUM); run on the Scalar engine,
        # which is idle in-loop, to keep it off the Vector chain.
        mul_then_reduce_tiled(Sp, q_mat, BH, dim, W, SQ, ones_col, red_p)
        nisa.tensor_copy(
            dst=O_row[0:1, 0:W], src=red_p[0:1, 0:W], engine=nisa.scalar_engine
        )

        # Write per-token output.
        if out_in_sbuf:
            # out_sbuf_tile [1, S*W]: token t in columns t*W : (t+1)*W (all on partition 0).
            nisa.tensor_copy(
                dst=out_sbuf_tile[0:1, t * W : (t + 1) * W], src=O_row[0:1, 0:W]
            )
        else:
            # output[h, t, j] <- O_row[0, h*dim+j].
            nisa.dma_copy(
                dst=out.ap(
                    pattern=[[W, 1], [seq_len * dim, BH], [1, dim]], offset=t * dim
                ),
                src=O_row.ap(pattern=[[W, 1], [dim, BH], [1, dim]], offset=0),
            )

        # Candidate state after token t (speculation): candidate_states[h, t, i, j] <- Sp.
        if write_candidates:
            _write_state(
                state_hbm, Sp, BH, dim, W,
                base_off=t * dim * dim, head_stride=seq_len * dim * dim,
            )

    if not write_candidates:
        # final_state <- the last token's working tile. Iteration t=seq_len-1 wrote
        # bufs[seq_len % 2], so that tile holds the final state for both even and odd seq_len.
        final_buf = bufs[seq_len % 2]
        _write_state(
            state_hbm, final_buf, BH, dim, W, base_off=0, head_stride=dim * dim,
        )


@nki.jit
def deltanet_tkg_fwd(query, key, value, g_in, beta_in, init_state):
    """Decode / commit: per-token output and the final post-block state.

    Returns:
        output:      (BH, S, 128) float32 -- per-token recurrence output
        final_state: (BH, 128, 128) float32 -- state after the last block token
    """
    BH, seq_len, dim = query.shape
    output = nl.ndarray((BH, seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    final_state = nl.ndarray((BH, dim, dim), dtype=nl.float32, buffer=nl.shared_hbm)
    gated_delta_rule_tkg(
        query, key, value, g_in, beta_in, init_state, output, final_state, None,
        write_candidates=False, out_in_sbuf=False,
    )
    return output, final_state


@nki.jit
def deltanet_tkg_fwd_state(query, key, value, g_in, beta_in, init_state):
    """Speculative verify: per-token output and the per-position candidate states.

    ``candidate_states[bh, t]`` is the state after consuming block token t, so the host selects
    ``[:, accept_count - 1]`` on a reject.

    Returns:
        output:           (BH, S, 128) float32 -- per-token recurrence output
        candidate_states: (BH, S, 128, 128) float32 -- state after each token
    """
    BH, seq_len, dim = query.shape
    output = nl.ndarray((BH, seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    candidate_states = nl.ndarray(
        (BH, seq_len, dim, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    gated_delta_rule_tkg(
        query, key, value, g_in, beta_in, init_state, output, candidate_states, None,
        write_candidates=True, out_in_sbuf=False,
    )
    return output, candidate_states


@nki.jit
def deltanet_tkg_fwd_sbuf(query, key, value, g_in, beta_in, init_state):
    """SBUF-output variant (megakernel demo): per-token output stays resident in SBUF.

    The recurrence writes output into an (S, BH*128) SBUF tile (no per-token HBM write); a real
    megakernel would feed that tile straight into the downstream fused op. Here it is copied to
    HBM only so callers/tests can read it. Output layout: ``output[t, h*128+j] = output_(h,t)[j]``.

    Returns:
        output:      (S, BH*128) float32 -- packed per-token output
        final_state: (BH, 128, 128) float32
    """
    BH, seq_len, dim = query.shape
    W = BH * dim
    # Packed output accumulates on partition 0: token t in columns t*W : (t+1)*W.
    out_sbuf_tile = nl.ndarray((1, seq_len * W), dtype=query.dtype, buffer=nl.sbuf)
    final_state = nl.ndarray((BH, dim, dim), dtype=nl.float32, buffer=nl.shared_hbm)
    gated_delta_rule_tkg(
        query, key, value, g_in, beta_in, init_state, None, final_state, out_sbuf_tile,
        write_candidates=False, out_in_sbuf=True,
    )
    # Test readout: reshape the packed [1, S*W] tile to HBM (S, W).
    output = nl.ndarray((seq_len, W), dtype=query.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=output.ap(pattern=[[seq_len * W, 1], [W, seq_len], [1, W]], offset=0),
        src=out_sbuf_tile.ap(pattern=[[seq_len * W, 1], [W, seq_len], [1, W]], offset=0),
    )
    return output, final_state
