# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeltaNet gated delta-rule recurrence for token generation (decode + speculative verify).

One launch processes all value-heads. Each head's 128x128 recurrent state is packed into a wide
SBUF tile ``Sp[128, W]`` (W = Hv*128) as ``Sp[i, h*128+j] = state_h[i, j]`` -- key index ``i`` on
the 128 partitions, ``(value-head, value index j)`` on the free axis -- and the gated delta rule is
iterated over the T-token block with the state resident in SBUF (no per-token HBM round-trip).

This kernel folds the input-side glue in so the caller does no transposes/reshapes/replication:
  * l2norm of q/k over d, then q *= 1/sqrt(d)
  * beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)  (then exp(g))
  * GQA head replication via an access-pattern broadcast (no data copy): value-head h reads
    k-head h//rep
The output RMSNorm / z-gate stay with the caller; the raw recurrence output is emitted head-major.

Per token (math identical to NeuronGatedDeltaNet._recurrent_step):
    decay   Sp = src * exp(g)              per-head scalar, free-broadcast across j
    read    kv = sum_i Sp[i,:] * k_h[i]    partition-reduce
    delta   d  = (v - kv) * beta
    update  Sp[i,:] += k_h[i] * d
    output  o  = sum_i Sp[i,:] * q_h[i]    partition-reduce

Implementation notes:
  * State is double-buffered (S0/S1 ping-pong): token t's output reads its working tile while
    token t+1's out-of-place decay writes the other, breaking the inter-token write-after-read.
  * Per-head scalars (decay, beta) are partition-broadcast at width Hv (one small matmul) and
    free-broadcast across j. beta is folded into the update's k-view (``k*beta``) off the serial
    chain, so the delta step is a single subtract.
  * exp(g) for every (head, token) is computed once before the loop (one activation-table load).
  * key/query (Hk heads) are loaded once up front via a free-axis contiguous bulk DMA, l2-normed
    over d, then a single on-chip nc_transpose each, landing the dim index on the 128 partitions
    the reduce contraction needs. The GQA expansion is an access-pattern broadcast on this Hk-head
    transposed tile -- value-head h sources k-head h//rep -- so only Hk copies live in SBUF.
    Requires Hk*T <= 128 (nc_transpose output partitions).
  * Partition broadcast/reduce results stay in PSUM and are consumed directly downstream.
  * The read/output reduces are pipelined over 512-wide tiles so each reduce matmul overlaps the
    next tile's multiply.

Entrypoints:
  deltanet_tkg_fwd        -> (attn_out, final_state)       decode / commit
  deltanet_tkg_fwd_state  -> (attn_out, candidate_states)  speculative verify

``candidate_states[t]`` is the state after consuming block token t; on a speculative reject the
host selects ``candidate_states[accept_count - 1]``. ``final_state`` equals the last candidate.

Input contract (all f32):
    q   (Hk, T, 128)   raw silu(conv) -- l2-normed over d then scaled by 1/sqrt(d) in-kernel
    k   (Hk, T, 128)   raw silu(conv) -- l2-normed over d in-kernel
    v   (Hv, T, 128)   raw silu(conv)
    a   (T, Hv)        raw in_proj_a   (token-major, head on free)
    b   (T, Hv)        raw in_proj_b
    A_log     (Hv,)    per-head decay param
    dt_bias   (Hv,)    per-head decay bias
    init_state  (Hv, 128, 128)
d = 128 = P_MAX.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Partition dimension max (NeuronCore SBUF tile width) = d.
P_MAX = 128

# Per-matmul moving free width = one PSUM bank (512 f32). Broadcast/reduce matmuls tile at this
# width into a wide PSUM tile that the next Vector op reads whole, so tiling adds no copies.
_PSUM_FMAX = 512


def div_ceil(n, d):
    """Ceil division for tile-count computation."""
    return (n + d - 1) // d


def partition_broadcast_psum(row_1W, width, ones_row, psum):
    """Partition-broadcast a (1, width) row to a (128, width) PSUM tile (all partitions equal).

    ones-row nc_matmul on the Tensor engine (result[m, n] = row[0, n]); left in PSUM for direct
    consumption. Tiled at _PSUM_FMAX per matmul; accumulate=False overwrites (PSUM tiles reused).
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


def mul_then_reduce_tiled(Sp, mat_t, t, T, Hk, Hv, rep, dim, W, mul_buf, ones_col, red_p):
    """Partition-reduce of ``Sp * free_broadcast(GQA-mapped mat)``, pipelined over 512-wide tiles.

    Computes ``red_p[0, h*dim+j] = sum_i Sp[i, h*dim+j] * mat_t[i, (h//rep)*T + t]`` (read uses
    mat_t = k_t, output uses mat_t = q_t). ``mat_t`` is the transposed ``[128, Hk*T]`` Hk-head tile
    (``mat_t[i, kh*T+t] = key/query[kh, t, i]``). The GQA broadcast: value-head h sources k-head
    h//rep, free-broadcast across j -- expressed as a free axis [group, rep, dim] with strides
    [T, 0, 0] so each k-head is reused rep times. Per 512-tile (4 value-heads), interleaving
    mul_c -> reduce_c lets the next multiply (Vector) overlap this reduce matmul (Tensor).
    """
    for c in nl.static_range(div_ceil(W, _PSUM_FMAX)):
        c0 = c * _PSUM_FMAX
        tile_w = min(_PSUM_FMAX, W - c0)
        vh0 = c0 // dim  # first value-head in this tile
        nvh = tile_w // dim  # value-heads in this tile (multiple of rep when rep>1 and tile>=rep*dim)
        kh0 = vh0 // rep  # first k-head in this tile
        ngrp = nvh // rep  # k-head groups in this tile
        # mat_view[i, ((g*rep + r)*dim) + j] = mat_t[i, (kh0+g)*T + t]: groups stride T, the rep
        # reuse and the dim free-broadcast both stride 0.
        mat_view_c = mat_t.ap(
            pattern=[[Hk * T, P_MAX], [T, ngrp], [0, rep], [0, dim]],
            offset=t + kh0 * T,
        )
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


def _write_state(state_hbm, Sp, Hv, dim, W, base_off, head_stride):
    """Write the packed state to HBM as one 3D store DMA: ``state[..., h, i, j] <- Sp[i, h*dim+j]``.

    Walks (partition i, head h, value index j). ``head_stride`` is the destination element distance
    between heads: dim*dim for the final state, T*dim*dim for the candidate stack.
    """
    nisa.dma_copy(
        dst=state_hbm.ap(
            pattern=[[dim, P_MAX], [head_stride, Hv], [1, dim]], offset=base_off
        ),
        src=Sp.ap(pattern=[[W, P_MAX], [dim, Hv], [1, dim]], offset=0),
    )


def _load_normed_qk(src, heads, T, dim, scale):
    """Load q or k (heads, T, dim), l2-norm over d, optionally scale, return transposed [dim, PS].

    PS = heads*T. Loads dim-contiguous (partition p=head*T+t strides HBM by dim), reduces sum-of-
    squares over the free dim, rsqrt-normalizes (and scales), then nc_transpose so dim lands on the
    128 partitions the reduce contracts over. Off the per-token critical path.
    """
    PS = heads * T
    x_f = nl.ndarray((PS, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_f, src=src.ap(pattern=[[dim, PS], [1, dim]], offset=0))
    # l2norm over d (free axis): sum of squares -> rsqrt -> scale rows.
    sq = nl.ndarray((PS, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=x_f, data2=x_f, op=nl.multiply)
    ss = nl.ndarray((PS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ss, data=sq, op=nl.add, axis=(1,))
    inv = nl.ndarray((PS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv, op=nl.rsqrt, data=ss, bias=None, scale=1.0)
    if scale != 1.0:
        nisa.tensor_scalar(dst=inv, data=inv, op0=nl.multiply, operand0=scale)
    nisa.tensor_scalar(dst=x_f, data=x_f, op0=nl.multiply, operand0=inv)
    x_t = nl.ndarray((dim, PS), dtype=nl.float32, buffer=nl.sbuf)
    x_t_p = nl.ndarray((dim, PS), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=x_t_p, data=x_f)
    nisa.tensor_copy(dst=x_t, src=x_t_p)
    return x_t


def gated_delta_rule_tkg(
    q, k, v, a, b, A_log, dt_bias, init_state,
    attn_out, state_hbm, write_candidates,
):
    """Batched-over-value-heads gated delta-rule recurrence with the input glue folded in.

    Packs every value-head's 128x128 state into a wide ``[128, W]`` tile (W = Hv*128), double-
    buffered (S0/S1 ping-pong), and iterates the gated delta rule over ``T`` tokens. The raw
    per-token recurrence output is written head-major to ``attn_out`` (HBM, T, Hv*dim); the caller
    applies the output RMSNorm / z-gate. When ``write_candidates`` the state after every token is
    written to ``state_hbm`` (T,Hv,dim,dim); otherwise only the final state is written (Hv,dim,dim).
    """
    Hk, T, dim = q.shape
    Hv = v.shape[0]
    rep = Hv // Hk
    W = Hv * dim
    inv_sqrt_d = 1.0 / (dim ** 0.5)
    # The key/query transpose lands dim on partitions, so its output has Hk*T partitions:
    # require Hk*T <= 128 (tile the token axis for larger blocks).
    PS = Hk * T
    assert PS <= P_MAX, (
        f"[NCC_INKI016] Kernel validation exception: Hk*T={PS} exceeds 128 (nc_transpose output "
        "partitions); the transposed-load path needs Hk*T<=128 -- tile the token axis"
    )

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
    kbeta_mat = nl.ndarray((P_MAX, Hv), dtype=nl.float32, buffer=nl.sbuf)

    # --- Gating math, hoisted (off the per-token critical path) ---
    # g_{t,h} = -exp(A_log_h) * softplus(a_{t,h} + dt_bias_h); then exp(g). All gating tables live
    # on partition 0 as flat (1, T*Hv) rows (col t*Hv + h), so the per-token slice [0:1, t*Hv:..]
    # stays on partition 0 for the partition-broadcast matmuls. dt_bias/A_log (per head, constant
    # over t) are a free-axis stride-0 broadcast over t.
    TH = T * Hv
    a_sb = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_sb, src=a.ap(pattern=[[TH, 1], [1, TH]], offset=0))
    b_sb = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_sb, src=b.ap(pattern=[[TH, 1], [1, TH]], offset=0))
    # dt_bias / exp(A_log) as (1, Hv) rows, free-broadcast across t (stride 0 over t, 1 over h).
    dtb = nl.ndarray((1, Hv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dtb, src=dt_bias.ap(pattern=[[Hv, 1], [1, Hv]], offset=0))
    Alog = nl.ndarray((1, Hv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Alog, src=A_log.ap(pattern=[[Hv, 1], [1, Hv]], offset=0))
    expA = nl.ndarray((1, Hv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=expA, op=nl.exp, data=Alog, bias=None, scale=1.0)
    dtb_bc = dtb.ap(pattern=[[Hv, 1], [0, T], [1, Hv]], offset=0)
    expA_bc = expA.ap(pattern=[[Hv, 1], [0, T], [1, Hv]], offset=0)
    # softplus(a + dt_bias).
    sp = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=sp.ap(pattern=[[TH, 1], [Hv, T], [1, Hv]], offset=0),
        data1=a_sb.ap(pattern=[[TH, 1], [Hv, T], [1, Hv]], offset=0),
        data2=dtb_bc,
        op=nl.add,
    )
    nisa.activation(dst=sp, op=nl.softplus, data=sp, bias=None, scale=1.0)
    # g = -exp(A_log) * softplus(...), then exp(g). exp_g_all (1, T*Hv): col t*Hv+h = exp(g_{t,h}).
    exp_g_all = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=exp_g_all.ap(pattern=[[TH, 1], [Hv, T], [1, Hv]], offset=0),
        data1=sp.ap(pattern=[[TH, 1], [Hv, T], [1, Hv]], offset=0),
        data2=expA_bc,
        op=nl.multiply,
    )
    nisa.activation(dst=exp_g_all, op=nl.exp, data=exp_g_all, bias=None, scale=-1.0)
    # beta = sigmoid(b); (1, T*Hv), col t*Hv+h = per-head write-gate.
    beta_all = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=beta_all, op=nl.sigmoid, data=b_sb, bias=None, scale=1.0)

    # --- Hoisted q/k: load (Hk heads), l2norm over d (+ scale q), transpose so dim on partitions.
    # GQA expansion is an AP broadcast in mul_then_reduce_tiled (no data replication).
    k_t = _load_normed_qk(k, Hk, T, dim, 1.0)
    q_t = _load_normed_qk(q, Hk, T, dim, inv_sqrt_d)

    # Reusable per-token PSUM tiles (matmul outputs; alloc once).
    eg_p = nl.ndarray((P_MAX, Hv), dtype=nl.float32, buffer=nl.psum)
    beta_p = nl.ndarray((P_MAX, Hv), dtype=nl.float32, buffer=nl.psum)
    bcast_p = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.psum)
    red_p = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.psum)

    # Two ping-pong state tiles. Each token reads the previous buffer and writes the other.
    S0 = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.sbuf)
    S1 = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.sbuf)
    bufs = [S0, S1]

    # Seed the state into S0: S0[i, h*dim+j] <- init_state[h, i, j]. Token 0 reads S0, writes S1.
    nisa.dma_copy(
        dst=S0.ap(pattern=[[W, P_MAX], [dim, Hv], [1, dim]], offset=0),
        src=init_state.ap(
            pattern=[[dim, P_MAX], [dim * dim, Hv], [1, dim]], offset=0
        ),
    )

    for t in nl.static_range(T):
        # Ping-pong (t % 2 is compile-time): src = previous final state, Sp = this token's working
        # state (holds the state after token t).
        src = bufs[t % 2]
        Sp = bufs[(t + 1) % 2]

        # Per-token GQA key view from the transposed Hk-head tile, value-head-expanded:
        # k_mat[i, h] = k_t[i, (h//rep)*T + t]  (h in [0,Hv)). For the beta-fold only.
        k_mat = k_t.ap(
            pattern=[[Hk * T, P_MAX], [T, Hk], [0, rep]], offset=t
        )
        # v_row [1, W]: [0, h*dim+j] <- value[h, t, j]  (v on the free axis: kv/delta index j).
        v_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=v_row.ap(pattern=[[W, 1], [dim, Hv], [1, dim]], offset=0),
            src=v.ap(
                pattern=[[W, 1], [T * dim, Hv], [1, dim]], offset=t * dim
            ),
        )
        # Per-token slices of the precomputed gating tables (partition 0, free cols t*Hv:(t+1)*Hv).
        beta_vec = beta_all[0:1, t * Hv : (t + 1) * Hv]
        exp_g_vec = exp_g_all[0:1, t * Hv : (t + 1) * Hv]

        # Fold beta into k off the serial chain: kbeta_mat[i, h] = k[i, h] * beta_h (per value-head,
        # GQA-mapped). beta per-head -> partition-broadcast (Tensor) then one narrow multiply.
        partition_broadcast_psum(beta_vec, Hv, ones_row, beta_p)
        nisa.tensor_tensor(
            dst=kbeta_mat, data1=k_mat, data2=beta_p[0:P_MAX, 0:Hv], op=nl.multiply
        )
        Kbeta_view = kbeta_mat.ap(pattern=[[Hv, P_MAX], [1, Hv], [0, dim]], offset=0)

        # Step 1: decay  Sp = src * exp(g)  (out-of-place: src stays readable). Per-head scalar
        # broadcast at width Hv into eg_p (PSUM), read via a free-broadcast view over j.
        partition_broadcast_psum(exp_g_vec, Hv, ones_row, eg_p)
        eg_view = eg_p.ap(pattern=[[Hv, P_MAX], [1, Hv], [0, dim]], offset=0)
        nisa.tensor_tensor(dst=Sp, data1=src, data2=eg_view, op=nl.multiply)

        # Step 2: read  kv[0, h*dim+j] = sum_i Sp[i, :] * k_h[i]  (into red_p, PSUM; pipelined).
        mul_then_reduce_tiled(Sp, k_t, t, T, Hk, Hv, rep, dim, W, SK, ones_col, red_p)

        # Step 3: delta = v - kv  (single Vector op; *beta folded into the update).
        nisa.tensor_tensor(
            dst=delta_row, data1=v_row, data2=red_p[0:1, 0:W], op=nl.subtract
        )

        # Step 4: update  Sp[i, :] += (k_h[i]*beta_h) * delta_h[j]. Broadcast delta, multiply by
        # Kbeta_view PSUM-direct, accumulate into Sp.
        partition_broadcast_psum(delta_row, W, ones_row, bcast_p)
        nisa.tensor_tensor(
            dst=outer, data1=Kbeta_view, data2=bcast_p[0:P_MAX, 0:W], op=nl.multiply
        )
        nisa.tensor_tensor(dst=Sp, data1=Sp, data2=outer, op=nl.add)

        # Step 5: output  O_row[0, h*dim+j] = sum_i Sp[i, :] * q_h[i]  (pipelined, like the read).
        mul_then_reduce_tiled(Sp, q_t, t, T, Hk, Hv, rep, dim, W, SQ, ones_col, red_p)
        nisa.tensor_copy(
            dst=O_row[0:1, 0:W], src=red_p[0:1, 0:W], engine=nisa.scalar_engine
        )

        # Write per-token raw recurrence output head-major: attn_out[t, h*dim+j] = O_row[0, h*dim+j].
        nisa.dma_copy(dst=attn_out[t : t + 1, 0:W], src=O_row[0:1, 0:W])

        # Candidate state after token t (speculation): candidate_states[t, h, i, j] <- Sp.
        if write_candidates:
            _write_state(
                state_hbm, Sp, Hv, dim, W,
                base_off=t * Hv * dim * dim, head_stride=dim * dim,
            )

    if not write_candidates:
        # final_state <- the last token's working tile. Iteration t=T-1 wrote bufs[T % 2].
        final_buf = bufs[T % 2]
        _write_state(
            state_hbm, final_buf, Hv, dim, W, base_off=0, head_stride=dim * dim,
        )


@nki.jit
def deltanet_tkg_fwd(q, k, v, a, b, A_log, dt_bias, init_state):
    """Decode / commit: raw recurrence output and the final post-block state.

    Returns:
        attn_out:    (T, Hv*128) float32 -- raw head-major recurrence output (caller RMSNorms/z-gates)
        final_state: (Hv, 128, 128) float32 -- state after the last block token
    """
    Hk, T, dim = q.shape
    Hv = v.shape[0]
    W = Hv * dim
    attn_out = nl.ndarray((T, W), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state = nl.ndarray((Hv, dim, dim), dtype=nl.float32, buffer=nl.shared_hbm)
    gated_delta_rule_tkg(
        q, k, v, a, b, A_log, dt_bias, init_state,
        attn_out, final_state, write_candidates=False,
    )
    return attn_out, final_state


@nki.jit
def deltanet_tkg_fwd_state(q, k, v, a, b, A_log, dt_bias, init_state):
    """Speculative verify: raw recurrence output and the per-position candidate states.

    ``candidate_states[t]`` is the state after consuming block token t, so the host selects
    ``[accept_count - 1]`` on a reject (axis-0 = position/accept axis).

    Returns:
        attn_out:         (T, Hv*128) float32 -- raw head-major recurrence output (caller RMSNorms/z-gates)
        candidate_states: (T, Hv, 128, 128) float32 -- state after each token
    """
    Hk, T, dim = q.shape
    Hv = v.shape[0]
    W = Hv * dim
    attn_out = nl.ndarray((T, W), dtype=nl.float32, buffer=nl.shared_hbm)
    candidate_states = nl.ndarray(
        (T, Hv, dim, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    gated_delta_rule_tkg(
        q, k, v, a, b, A_log, dt_bias, init_state,
        attn_out, candidate_states, write_candidates=True,
    )
    return attn_out, candidate_states
