# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeltaNet gated delta-rule recurrence for token generation (decode + speculative verify).

Each head's 128x128 recurrent state is packed into a wide SBUF tile ``Sp[128, W]`` (W = Hv*128) as
``Sp[i, h*128+j] = state_h[i, j]`` -- key index ``i`` on the 128 partitions, ``(value-head, value
index j)`` on the free axis -- and the gated delta rule is iterated over the T-token block with the
state resident in SBUF (no per-token HBM round-trip).

The work is SPMD-sharded by **value-head** across ``n = nl.num_programs(0)`` cores: SPMD broadcasts
the full HBM tensors to every core, and core ``c`` computes only its ``Hv = Hv_full//n`` value-heads
(reading the matching ``Hk = Hk_full//n`` k/q-heads) and writes a disjoint slice of every full-shape
output. The per-token math body is fully parametric in the LOCAL head counts; the only per-core
differences are the AP offsets threaded into each HBM load/store. GQA stays local: local value-head
``h`` reads local k-head ``h//rep`` (with ``rep = Hv//Hk``), which equals the global mapping because
``Hv = rep*Hk`` holds per core. ``n=1`` reduces to a single core owning all heads (offsets 0).

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
  * The decay scalar is partition-broadcast once for the whole block (width T*Hv) and free-broadcast
    across j; the per-token operand is an AP at offset t*Hv. beta is folded into the update's key
    rows (``k*beta``) for all tokens up front, so the delta step is a single subtract.
  * The update is a rank-1 matmul per value-head (key row stationary, delta segment moving), so the
    outer product needs no partition broadcast of delta and no Vector product tile.
  * exp(g) for every (head, token) is computed once before the loop (one activation-table load).
  * key/query (Hk heads) are loaded once up front via a free-axis contiguous bulk DMA, l2-normed
    over d, then a single on-chip nc_transpose each, landing the dim index on the 128 partitions
    the reduce contraction needs. The GQA expansion is an access-pattern broadcast on this Hk-head
    transposed tile -- value-head h sources k-head h//rep -- so only Hk copies live in SBUF.
    Requires Hk*T <= 128 (nc_transpose output partitions).
  * Partition broadcast/reduce results stay in PSUM and are consumed directly downstream.
  * The read is one matmul per GQA group: the group's key/query column pair is the stationary operand
    and its state columns are the moving tile, so the elementwise multiply happens inside the PE array
    (no [128, W] Vector product) and both reads come out of one pass over the state, landing on PSUM
    partitions 0 (key) and 1 (query). A stream shuffle moves the query read to partition 0 on the
    output branch (compute accesses must start on a quadrant boundary), off the state chain.
  * The output takes its read on the PRE-update state and adds the update's rank-1 term in closed
    form, ``(q.kbeta)*delta``, so it needs no second pass over the state.

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

from .norm_gate import fold_gamma_silu, norm_gate_row

# Partition dimension max (NeuronCore SBUF tile width) = d.
P_MAX = 128

# Per-matmul moving free width = one PSUM bank (512 f32). Broadcast/reduce matmuls tile at this
# width into a wide PSUM tile that the next Vector op reads whole, so tiling adds no copies.
_PSUM_FMAX = 512


def div_ceil(n, d):
    """Ceil division for tile-count computation."""
    return (n + d - 1) // d


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


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


def reduce_pair_by_head_group(Sp, kq_t, t, T, Hk, rep, dim, pair_p):
    """Partition-reduce of ``Sp`` against the GQA-mapped query AND key columns in one state pass.

    Computes ``pair_p[r, h*dim+j] = sum_i Sp[i, h*dim+j] * kq_t[i, (h//rep)*T + t, r]`` for r = 0
    (key) and r = 1 (query). ``kq_t`` is the transposed ``[128, Hk*T, 2]`` Hk-head tile
    (``kq_t[i, kh*T+t, 0/1] = key/query[kh, t, i]``). A GQA group's ``rep`` value-heads share one
    key/query column pair and occupy ``rep*dim`` contiguous state columns, so the pair is the
    stationary tile and the group's state slice is the moving tile: the PE does the elementwise
    multiply and the partition sum in one pass. Results land on PSUM partitions 0 (key) / 1 (query).
    Requires ``rep*dim <= _PSUM_FMAX`` (one matmul per group).
    """
    grp_w = rep * dim
    for g in nl.static_range(Hk):
        nisa.nc_matmul(
            dst=pair_p[0:2, g * grp_w : (g + 1) * grp_w],
            stationary=kq_t[0:P_MAX, g * T + t, 0:2],
            moving=Sp[0:P_MAX, g * grp_w : (g + 1) * grp_w],
            accumulate=False,
        )


def _write_state(state_hbm, Sp, Hv, dim, W, base_off, head_stride):
    """Write this core's packed state to its slice of the full-shape HBM tensor as one 3D store DMA:
    ``state[..., vh_off+h, i, j] <- Sp[i, h*dim+j]`` for h in [0, Hv) (LOCAL head count).

    Walks (partition i, local head h, value index j). ``head_stride = dim*dim`` (heads contiguous in
    both the final and candidate layouts); the caller's ``base_off`` carries the value-head offset
    ``vh_off*dim*dim`` and, for the candidate stack, the per-token block ``t*Hv_full*dim*dim``.
    """
    nisa.dma_copy(
        dst=state_hbm.ap(
            pattern=[[dim, P_MAX], [head_stride, Hv], [1, dim]], offset=base_off
        ),
        src=Sp.ap(pattern=[[W, P_MAX], [dim, Hv], [1, dim]], offset=0),
    )


def _load_normed_qk(src, heads, T, dim, scale, src_off, x_f_in=None):
    """Load q or k (heads, T, dim), l2-norm over d, optionally scale; returns ``(x_t, x_f)``.

    PS = heads*T (LOCAL head count). Loads dim-contiguous from this core's head slice (``src_off``
    skips earlier cores' heads in the full HBM tensor; partition p=head*T+t strides HBM by dim),
    reduces sum-of-squares over the free dim, rsqrt-normalizes (and scales), then nc_transpose so
    dim lands on the 128 partitions the reduce contracts over. Off the per-token critical path.
    Returns the transposed ``x_t`` [dim, PS] (sized at the local PS, the nc_transpose data-AP
    partition stride) and the pre-transpose normed rows ``x_f`` [PS, dim], which the update's rank-1
    matmul takes as its stationary operand.

    SBUF-input variant: when ``x_f_in`` is given it is the pre-loaded ``[PS, dim]`` conv q/k tile
    (partition = head*T+t, free = dim -- exactly the conv's silu'd q/k sub-block); it is upcast-copied
    into the fp32 working buffer in place of the HBM DMA, so ``src``/``src_off`` are ignored. The
    fp32 buffer is required because gen3 nc_transpose needs dst dtype == input dtype and the recurrence
    matmuls consume fp32 q/k (the conv tile is bf16); the conv tile itself is left unmutated.
    """
    PS = heads * T
    x_f = nl.ndarray((PS, dim), dtype=nl.float32, buffer=nl.sbuf)
    if x_f_in != None:
        nisa.tensor_copy(
            dst=x_f, src=x_f_in
        )  # upcast the bf16 conv tile into the fp32 buffer
    else:
        nisa.dma_copy(
            dst=x_f, src=src.ap(pattern=[[dim, PS], [1, dim]], offset=src_off)
        )
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
    return x_t, x_f


def gated_delta_rule_tkg(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    init_state,
    attn_out,
    state_hbm,
    write_candidates,
    q_sbuf=None,
    k_sbuf=None,
    v_sbuf=None,
    Hk_full=None,
    Hv_full=None,
    z=None,
    gamma=None,
    eps=None,
    proj_sb=None,
    a_off=None,
    b_off=None,
    z_off=None,
    attn_sb_out=None,
):
    """Value-head-sharded gated delta-rule recurrence with the input glue folded in.

    Receives the FULL HBM tensors; this core (program ``c`` of ``n``) computes only its
    ``Hv = Hv_full//n`` value-heads. Packs those heads' 128x128 states into a wide ``[128, W]`` tile
    (W = Hv*128, LOCAL), double-buffered (S0/S1 ping-pong), and iterates the gated delta rule over
    ``T`` tokens. The raw per-token recurrence output is written head-major into this core's columns
    of the full-shape ``attn_out`` (HBM, T, Hv_full*dim); the caller applies the output RMSNorm /
    z-gate. When ``write_candidates`` the state after every token is written to the core's head slice
    of ``state_hbm`` (T,Hv_full,dim,dim); otherwise only the final state is written (Hv_full,dim,dim).
    Every HBM access is offset by the per-core shard; the per-token math body is unchanged and fully
    parametric in the LOCAL counts.

    SBUF-input (fused) variant: when ``q_sbuf``/``k_sbuf``/``v_sbuf`` are given they are this core's
    silu'd conv tiles (already LOCAL heads -- partition = local_head*T+t, free = head_dim), so q/k feed
    ``_load_normed_qk`` directly (no q/k HBM load, no head offset) and v is gathered per token from
    ``v_sbuf`` via an SBUF->SBUF DMA (no v HBM load, no head offset). The gating tables / init_state seed
    / attn_out columns / state heads stay FULL HBM tensors sliced by ``vh_off``/``col_off``/``W_full``.
    With these None the behavior is byte-identical to the HBM path. ``Hk_full``/``Hv_full`` must be
    passed in this variant (the SBUF tiles carry only local heads); ``T``/``dim`` come from ``attn_out``.

    Gated-norm (optional): when ``z``/``gamma`` are provided, the per-token output is gated per-head
    RMSNorm'd (``norm_gate_row``: per-head RMSNorm over dim * silu(z)) before the ``attn_out`` write;
    ``z`` is the FULL [T, W_full] head-major gate sliced by ``col_off`` (matching the output columns)
    and ``gamma`` is the replicated [dim] norm weight (loaded once per core). With these None the raw
    recurrence output is written exactly as before -- the norm seam is opt-in and leaves the state /
    candidate outputs untouched.

    in_proj-SBUF gating (Bridge 2, no HBM round-trip): when ``proj_sb`` (the full in_proj output
    [T, I] in SBUF, token on partition, channel on free) is given, a/b/z are sourced from it instead of
    from the HBM ``a``/``b``/``z`` tensors. ``a_off``/``b_off``/``z_off`` are the free-axis offsets of
    the a/b/z segments in the projection (a/b are [T, Hv_full] head-on-free, z is [T, W_full]
    head-major). a/b are gathered to the LOCAL (1, T*Hv) gating rows and z per token to (1, W) with
    cross-partition SBUF->SBUF DMAs (proj token t lives on partition t). With ``proj_sb`` None the
    gating loads from HBM exactly as before.

    SBUF output collection (o_proj fusion): when ``attn_sb_out`` (a [T, W] SBUF tile) is given, each
    token's gated output row is copied into ``attn_sb_out[t]`` and the ``attn_out`` HBM write is
    skipped, leaving the gated tensor SBUF-resident for the output projection.
    """
    from_sbuf = q_sbuf != None
    gate_from_proj = proj_sb != None
    collect_sbuf = attn_sb_out != None
    if from_sbuf:
        T = attn_out.shape[0]
        dim = attn_out.shape[1] // Hv_full
    else:
        Hk_full, T, dim = q.shape
        Hv_full = v.shape[0]
    inv_sqrt_d = 1.0 / (dim**0.5)

    # Value-head SPMD shard: this core owns 1/n of the heads. nl.num_programs/program_id fold to
    # trace-time ints, so all of the shard math (LOCAL counts, FULL strides, per-core offsets) runs
    # at trace time and drives the AP offsets directly. n=1 -> full ownership at offset 0.
    n = nl.num_programs(0)
    c = nl.program_id(0)
    kernel_assert(Hv_full % n == 0, "v-heads must divide across cores")
    kernel_assert(Hk_full % n == 0, "q/k-heads must divide across cores")
    # LOCAL counts: the per-token math body is written entirely in these (textually unchanged).
    Hk = Hk_full // n
    Hv = Hv_full // n
    rep = Hv // Hk
    W = Hv * dim
    kernel_assert(
        Hv % rep == 0, "whole GQA groups per core (Hv_loc must be a multiple of rep)"
    )
    # FULL strides (for indexing the full input/output HBM tensors).
    W_full = Hv_full * dim
    # Per-core offsets: q/k head offset, v head offset, attn_out column offset.
    kv_off = c * Hk
    vh_off = c * Hv
    col_off = c * W

    # The key/query transpose lands dim on partitions, so its output has Hk*T partitions (LOCAL):
    # require Hk*T <= 128 (tile the token axis for larger blocks).
    PS = Hk * T
    kernel_assert(
        PS <= P_MAX,
        f"Hk_loc*T={PS} exceeds 128 (nc_transpose output partitions) -- tile the token axis",
    )
    # The paired read is one matmul per GQA group, so a group's state columns must fit one PSUM bank.
    kernel_assert(
        rep * dim <= _PSUM_FMAX,
        f"rep*dim={rep * dim} exceeds one PSUM bank ({_PSUM_FMAX} f32)",
    )

    # ones operand for the broadcast matmuls (alloc once).
    ones_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_row, value=1.0)

    # Reusable per-token SBUF tiles (alloc once). qs_row is the query-read brought back to partition 0.
    qs_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
    delta_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
    O_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)

    # --- Gating math, hoisted (off the per-token critical path) ---
    # g_{t,h} = -exp(A_log_h) * softplus(a_{t,h} + dt_bias_h); then exp(g). All gating tables live
    # on partition 0 as flat (1, T*Hv) rows (col t*Hv + h), so the per-token slice [0:1, t*Hv:..]
    # stays on partition 0 for the partition-broadcast matmuls. dt_bias/A_log (per head, constant
    # over t) are a free-axis stride-0 broadcast over t.
    TH = T * Hv
    # Pack this core's per-head a/b slice into the LOCAL (1, T*Hv) row: from the HBM token-major tables,
    # or (in_proj fusion) gathered per-token from proj_sb onto partition 0 (same SBUF->SBUF gather as the v-bridge).
    a_sb = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    b_sb = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    if gate_from_proj:
        for t in range(T):
            nisa.dma_copy(
                dst=a_sb[0:1, t * Hv : (t + 1) * Hv],
                src=proj_sb[t : t + 1, a_off + vh_off : a_off + vh_off + Hv],
            )
            nisa.dma_copy(
                dst=b_sb[0:1, t * Hv : (t + 1) * Hv],
                src=proj_sb[t : t + 1, b_off + vh_off : b_off + vh_off + Hv],
            )
    else:
        nisa.dma_copy(
            dst=a_sb.ap(pattern=[[TH, 1], [Hv, T], [1, Hv]], offset=0),
            src=a.ap(pattern=[[1, 1], [Hv_full, T], [1, Hv]], offset=vh_off),
        )
        nisa.dma_copy(
            dst=b_sb.ap(pattern=[[TH, 1], [Hv, T], [1, Hv]], offset=0),
            src=b.ap(pattern=[[1, 1], [Hv_full, T], [1, Hv]], offset=vh_off),
        )
    # dt_bias / exp(A_log) are full [Hv_full]; take this core's contiguous head slice (offset vh_off,
    # count Hv) as (1, Hv) rows, free-broadcast across t (stride 0 over t, 1 over h).
    dtb = nl.ndarray((1, Hv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dtb, src=dt_bias.ap(pattern=[[Hv, 1], [1, Hv]], offset=vh_off))
    Alog = nl.ndarray((1, Hv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Alog, src=A_log.ap(pattern=[[Hv, 1], [1, Hv]], offset=vh_off))
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

    # --- Hoisted q/k: l2norm over d (+ scale q), transpose so dim on partitions. GQA expansion is an
    # AP broadcast in reduce_pair_by_head_group (no data replication). HBM path loads this core's Hk heads
    # (from the full tensor, offset kv_off*T*dim); SBUF path consumes the conv tiles directly (already
    # this core's local heads -- no offset, no DMA). The conv tiles ARE the [Hk*T, dim] x_f layout.
    if from_sbuf:
        k_t, k_f = _load_normed_qk(None, Hk, T, dim, 1.0, 0, x_f_in=k_sbuf)
        q_t, _ = _load_normed_qk(None, Hk, T, dim, inv_sqrt_d, 0, x_f_in=q_sbuf)
    else:
        qk_off = kv_off * T * dim
        k_t, k_f = _load_normed_qk(k, Hk, T, dim, 1.0, qk_off)
        q_t, _ = _load_normed_qk(q, Hk, T, dim, inv_sqrt_d, qk_off)

    # Key/query columns interleaved into one [dim, Hk*T, 2] stationary tile so a single pass over the
    # state yields both per-token reads. Key takes PSUM partition 0 so the delta -- which gates the
    # state update -- reads it directly; the query read is shuffled off the chain.
    kq_t = nl.ndarray((dim, PS, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kq_t[0:dim, 0:PS, 0], src=k_t[0:dim, 0:PS])
    nisa.tensor_copy(dst=kq_t[0:dim, 0:PS, 1], src=q_t[0:dim, 0:PS])

    # a_{t,h} = q_t . kbeta_{t,h} = beta_{t,h} * (q_t . k_t): the weight of the token's own rank-1
    # update in its output, which is what lets the output read the PRE-update state (see step 5).
    # (q_t . k_t) is per GQA group -- one 1x1 matmul per (group, token) -- then expanded over rep.
    qk_p = nl.ndarray((1, PS), dtype=nl.float32, buffer=nl.psum)
    for g in nl.static_range(Hk):
        for t in nl.static_range(T):
            nisa.nc_matmul(
                dst=qk_p[0:1, g * T + t : g * T + t + 1],
                stationary=q_t[0:P_MAX, g * T + t : g * T + t + 1],
                moving=k_t[0:P_MAX, g * T + t : g * T + t + 1],
                accumulate=False,
            )
    a_all = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=a_all.ap(pattern=[[TH, 1], [Hv, T], [rep, Hk], [1, rep]], offset=0),
        data1=beta_all.ap(pattern=[[TH, 1], [Hv, T], [rep, Hk], [1, rep]], offset=0),
        data2=qk_p.ap(pattern=[[PS, 1], [1, T], [T, Hk], [0, rep]], offset=0),
        op=nl.multiply,
    )

    # Optional gated per-head RMSNorm at the output seam. The gate depends only on the projection, so
    # the whole block's gamma*silu(z) is built here, leaving one multiply per token at the seam. The
    # proj_sb gather stays per token (token t lives on partition t); the HBM z path is one DMA.
    apply_norm = gamma != None
    if apply_norm:
        gsz_all = nl.ndarray((1, T * W), dtype=nl.float32, buffer=nl.sbuf)
        if gate_from_proj:
            for t in range(T):
                nisa.dma_copy(
                    dst=gsz_all[0:1, t * W : (t + 1) * W],
                    src=proj_sb[t : t + 1, z_off + col_off : z_off + col_off + W],
                )
        else:
            nisa.dma_copy(
                dst=gsz_all.ap(pattern=[[T * W, 1], [W, T], [1, W]], offset=0),
                src=z.ap(pattern=[[1, 1], [W_full, T], [1, W]], offset=col_off),
            )
        fold_gamma_silu(gsz_all, gamma, T, W, dim)

    # Reusable per-token PSUM tiles (matmul outputs; alloc once).
    upd_p = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.psum)
    pair_p = nl.ndarray((2, W), dtype=nl.float32, buffer=nl.psum)

    # Decay scalars, partition-broadcast once for the whole block: the per-token operand is an AP at
    # offset t*Hv into eg_p, free-broadcast across j.
    eg_p = nl.ndarray((P_MAX, TH), dtype=nl.float32, buffer=nl.psum)
    partition_broadcast_psum(exp_g_all, TH, ones_row, eg_p)

    # kbeta rows, the update's rank-1 stationary operand, built once for the whole block:
    # kb_rows[0, (t*Hv+h)*dim + i] = k_normed[(h//rep)*T+t, i] * beta_{t,h}. The GQA expansion is a
    # stride-0 read of each group's key row, then one free-broadcast multiply by beta. Rows stay on
    # partition 0, which is where nc_matmul takes a stationary tile.
    kb_rows = nl.ndarray((1, T * W), dtype=nl.float32, buffer=nl.sbuf)
    for t in range(T):
        for g in range(Hk):
            nisa.dma_copy(
                dst=kb_rows[
                    0:1, (t * Hv + g * rep) * dim : (t * Hv + (g + 1) * rep) * dim
                ],
                src=k_f.ap(
                    pattern=[[dim, 1], [0, rep], [1, dim]], offset=(g * T + t) * dim
                ),
            )
    nisa.tensor_tensor(
        dst=kb_rows.ap(pattern=[[T * W, 1], [dim, TH], [1, dim]], offset=0),
        data1=kb_rows.ap(pattern=[[T * W, 1], [dim, TH], [1, dim]], offset=0),
        data2=beta_all.ap(pattern=[[TH, 1], [1, TH], [0, dim]], offset=0),
        op=nl.multiply,
    )

    # Two ping-pong state tiles. Each token reads the previous buffer and writes the other.
    S0 = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.sbuf)
    S1 = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.sbuf)
    bufs = [S0, S1]

    # Seed the state into S0 from this core's head slice of the full init_state [Hv_full,dim,dim]:
    # S0[i, h*dim+j] <- init_state[vh_off+h, i, j] for h in [0, Hv). Token 0 reads S0, writes S1.
    nisa.dma_copy(
        dst=S0.ap(pattern=[[W, P_MAX], [dim, Hv], [1, dim]], offset=0),
        src=init_state.ap(
            pattern=[[dim, P_MAX], [dim * dim, Hv], [1, dim]], offset=vh_off * dim * dim
        ),
    )  # 512 B DMA

    for t in nl.static_range(T):
        # Ping-pong (t % 2 is compile-time): src = previous final state, Sp = this token's working
        # state (holds the state after token t).
        src = bufs[t % 2]
        Sp = bufs[(t + 1) % 2]

        # v_row [1, W]: [0, h*dim+j] <- value[h, t, j]  (this core's Hv heads, v on the free axis:
        # kv/delta index j). HBM path loads from the full [Hv_full,T,dim] tensor (offset by vh_off).
        # SBUF path gathers from the conv v tile (already this core's local heads, partition =
        # local_head*T+t): one on-chip SBUF->SBUF DMA per head copies partition h*T+t -> v_row's
        # head-h columns. Per-head (not one strided DMA) because a multi-dim SBUF AP can only stride
        # whole partitions at free-dim granularity; the head partitions are interleaved by token
        # (stride T), so the gather isn't expressible as a single multi-partition AP. No vh_off (the
        # conv tile is already this core's slice).
        v_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
        if from_sbuf:
            for h in range(Hv):
                nisa.dma_copy(
                    dst=v_row[0:1, h * dim : (h + 1) * dim],
                    src=v_sbuf[h * T + t : h * T + t + 1, 0:dim],
                )  # 4 B DMA packets
        else:
            nisa.dma_copy(
                dst=v_row.ap(pattern=[[W, 1], [dim, Hv], [1, dim]], offset=0),
                src=v.ap(
                    pattern=[[W, 1], [T * dim, Hv], [1, dim]],
                    offset=vh_off * T * dim + t * dim,
                ),
            )
        # Step 1: decay  Sp = src * exp(g)  (out-of-place: src stays readable). Per-token slice of
        # the block-wide broadcast, free-broadcast across j.
        eg_view = eg_p.ap(pattern=[[TH, P_MAX], [1, Hv], [0, dim]], offset=t * Hv)
        nisa.tensor_tensor(dst=Sp, data1=src, data2=eg_view, op=nl.multiply)

        # Step 2: one pass over the decayed state yields both reads -- kv = k^T Sp on PSUM partition 0
        # and q^T Sp on partition 1.
        reduce_pair_by_head_group(Sp, kq_t, t, T, Hk, rep, dim, pair_p)

        # Step 3: delta = v - kv  (single Vector op; *beta folded into the update).
        nisa.tensor_tensor(
            dst=delta_row, data1=v_row, data2=pair_p[0:1, 0:W], op=nl.subtract
        )

        # Step 4: update  Sp[i, h*dim+j] += (k_h[i]*beta_h) * delta[h*dim+j] -- one rank-1 matmul per
        # value-head (stationary = the head's kbeta row, moving = its delta segment), straight to PSUM.
        for h in nl.static_range(Hv):
            nisa.nc_matmul(
                dst=upd_p[0:P_MAX, h * dim : (h + 1) * dim],
                stationary=kb_rows[0:1, (t * Hv + h) * dim : (t * Hv + h + 1) * dim],
                moving=delta_row[0:1, h * dim : (h + 1) * dim],
                accumulate=False,
            )
        nisa.tensor_tensor(dst=Sp, data1=Sp, data2=upd_p[0:P_MAX, 0:W], op=nl.add)

        # Step 5: output  O = q^T Sp_post = q^T Sp + (q . kbeta) * delta -- step 2's pre-update query
        # read plus the update's rank-1 term, which collapses to the per-head scalar a_{t,h}.
        nisa.nc_stream_shuffle(
            dst=qs_row[0:1, 0:W], src=pair_p[0:2, 0:W], shuffle_mask=[1] * 32
        )
        a_view = a_all.ap(pattern=[[TH, 1], [1, Hv], [0, dim]], offset=t * Hv)
        nisa.tensor_tensor(dst=O_row, data1=delta_row, data2=a_view, op=nl.multiply)
        nisa.tensor_tensor(dst=O_row, data1=O_row, data2=qs_row, op=nl.add)

        # Optional gated per-head RMSNorm of the raw row (norm over dim * this token's gate slice).
        if apply_norm:
            out_row = norm_gate_row(O_row, gsz_all[0:1, t * W : (t + 1) * W], eps, dim)
        else:
            out_row = O_row

        if collect_sbuf:
            # Keep the gated row SBUF-resident for the output projection (DMA places it on partition t).
            nisa.dma_copy(dst=attn_sb_out[t : t + 1, 0:W], src=out_row[0:1, 0:W])
        else:
            # Write this core's head-major columns of the full-shape attn_out [T, W_full].
            nisa.dma_copy(
                dst=attn_out.ap(
                    pattern=[[W_full, 1], [1, W]], offset=t * W_full + col_off
                ),
                src=out_row[0:1, 0:W],
            )

        # Candidate state after token t (speculation): candidate_states[t, vh_off+h, i, j] <- Sp.
        # Full per-token block stride is Hv_full*dim*dim; this core's heads start at vh_off.
        if write_candidates:
            _write_state(
                state_hbm,
                Sp,
                Hv,
                dim,
                W,
                base_off=t * Hv_full * dim * dim + vh_off * dim * dim,
                head_stride=dim * dim,
            )

    if not write_candidates:
        # final_state <- the last token's working tile (this core's heads at vh_off).
        # Iteration t=T-1 wrote bufs[T % 2].
        final_buf = bufs[T % 2]
        _write_state(
            state_hbm,
            final_buf,
            Hv,
            dim,
            W,
            base_off=vh_off * dim * dim,
            head_stride=dim * dim,
        )


@nki.jit
def deltanet_tkg_fwd(q, k, v, a, b, A_log, dt_bias, init_state):
    """Decode / commit: raw recurrence output and the final post-block state.

    Allocates FULL-shape outputs (all value-heads); under an LNC=n launch each core fills only its
    disjoint head/column slice. Launched ``deltanet_tkg_fwd[n](...)``.

    Returns:
        attn_out:    (T, Hv*128) float32 -- raw head-major recurrence output (caller RMSNorms/z-gates)
        final_state: (Hv, 128, 128) float32 -- state after the last block token
    """
    T, dim = q.shape[1], q.shape[2]
    Hv_full = v.shape[0]
    W_full = Hv_full * dim
    attn_out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state = nl.ndarray(
        (Hv_full, dim, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    gated_delta_rule_tkg(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        final_state,
        write_candidates=False,
    )
    return attn_out, final_state


@nki.jit
def deltanet_tkg_fwd_state(q, k, v, a, b, A_log, dt_bias, init_state):
    """Speculative verify: raw recurrence output and the per-position candidate states.

    ``candidate_states[t]`` is the state after consuming block token t, so the host selects
    ``[accept_count - 1]`` on a reject (axis-0 = position/accept axis). Allocates FULL-shape outputs;
    under an LNC=n launch each core fills only its disjoint head/column slice. Launched
    ``deltanet_tkg_fwd_state[n](...)``.

    Returns:
        attn_out:         (T, Hv*128) float32 -- raw head-major recurrence output (caller RMSNorms/z-gates)
        candidate_states: (T, Hv, 128, 128) float32 -- state after each token
    """
    T, dim = q.shape[1], q.shape[2]
    Hv_full = v.shape[0]
    W_full = Hv_full * dim
    attn_out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)
    candidate_states = nl.ndarray(
        (T, Hv_full, dim, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    gated_delta_rule_tkg(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        candidate_states,
        write_candidates=True,
    )
    return attn_out, candidate_states
