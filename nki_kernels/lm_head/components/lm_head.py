# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LM head for Qwen3.6-A3B token generation: final RMSNorm -> vocab matmul -> per-rank greedy argmax.

One SBUF-resident composable shared by both heads -- the verify pass (T=2) and the MTP draft step
(T=1) differ only in T. Everything between the norm and the argmax stays in SBUF; the only HBM
traffic is the streamed lm_head weight.

    normed_sb  = rmsnorm(hidden, gamma)                      [H0, T, H1] tp2013
    attn_sb    = head_major_from_tp2013(normed_sb)           [128, 1, H1, T]
    logits_sb  = output_projection_tkg(attn_sb, lm_head_w)   [T, V_core]  (vocab LNC-sharded)
    (max, idx) = staged_argmax(logits_sb)                    per core, then combined across cores

``output_projection_tkg`` LNC-shards its OUTPUT dimension, so V_core = V_rank / n_prgs with no
collective. The composable stops at the per-rank (max, index); the cross-rank argmax belongs in the
megakernel, next to its all-reduce helpers.

Precision is bf16 end to end, matching the model's bare bf16 ``torch.matmul`` lm_head. Weights are
consumed VERBATIM: TransposedColumnParallelLinear already stores [H, V_rank], which is
``output_projection_tkg``'s [in, out] layout.

A3B per-rank config (TP=4, LNC=2): H=2048, H0=128, H1=16, V_rank=62080, V_core=31040, T in {1,2}.
Spec: nki_kernels/specs/lm_head.md.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.output_projection.output_projection_tkg import output_projection_tkg
from nkilib.core.utils.allocator import BufferManager
from nkilib.core.utils.common_types import QuantizationType
from nkilib.core.utils.kernel_helpers import (
    div_ceil,
    get_verified_program_sharding_info,
)
from nkilib.core.utils.tensor_view import TensorView

from ...common import H0, kernel_assert, rmsnorm_to_sbuf

# Per-instruction element cap for a DVE reduce; the V_core-wide search is staged under it.
MAX_REDUCE_WIDTH = 1 << 14
# nc_find_index8 always reports 8 candidate positions; only the first (lowest) one is used.
FIND_INDEX_WIDTH = 8
# Manually-managed SBUF region, per partition; sized for the vocab-scale logits tile. Claiming ALL
# of SBUF fails outright -- the backend needs room for its own scratch.
SBM_SIZE_BYTES = 224 * 1024


def make_manual_sbuf_manager():
    """A MANUAL BufferManager over the SBM_SIZE_BYTES region.

    ``output_projection_tkg`` only streams weight blocks under manual allocation; the auto-alloc path
    preloads every block, which does not fit at vocab scale. Because this manager places tensors
    itself, EVERY SBUF tile in the composable must come from it.
    """
    return BufferManager(0, SBM_SIZE_BYTES, use_auto_alloc=False)


def head_major_from_tp2013(normed_sb, n_prgs, sbm):
    """Permute a tp2013 [H0, T, H1] tile into output_projection_tkg's [D=128, B=1, N=H1, S=T] layout.

    The two layouts factor the hidden index differently and a free-axis reorder cannot bridge them:
    tp2013 puts the partition axis at hidden-stride H2, output_projection_tkg at hidden-stride 1, and
    H2 in {8, 16} never equals D = 128. Writing h0 = a*G + b with G = H0/H2 gives the target

        attn[b*H2 + h2, 0, s*H2 + a, t] = normed_sb[a*G + b, t, s*H2 + h2]

    which takes two moves, because neither engine can do it alone (spec §3):
      1. one nc_transpose per (s, t) of the contiguous [H0, H2] block -> st[h2, h0];
      2. one DMA per (b, s, t) of st[0:H2, b::G] into partitions [b*H2, (b+1)*H2) -- a free-axis
         stride plus a partition BASE offset, which the DMA engine handles.
    G*n_prgs*T <= 64 small DMAs, negligible against a ~127 MB/core weight stream.
    """
    _, T, H1 = normed_sb.shape
    kernel_assert(H1 % n_prgs == 0, "tp2013 needs H1 divisible by the H-shard count")
    H2 = H1 // n_prgs
    G = H0 // H2
    kernel_assert(G * H2 == H0, "tp2013 H-tile width must divide the partition dim")

    attn_sb = sbm.alloc_stack((H0, 1, H1, T), dtype=normed_sb.dtype, buffer=nl.sbuf)
    st_sb = sbm.alloc_stack((H2, H0), dtype=normed_sb.dtype, buffer=nl.sbuf)
    for s in range(n_prgs):
        for t in range(T):
            # nc_transpose runs in matmul transpose mode: on gen3+ the PSUM dtype must match the input.
            st_psum = nl.ndarray((H2, H0), dtype=normed_sb.dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=st_psum, data=normed_sb[0:H0, t, s * H2 : (s + 1) * H2]
            )
            nisa.tensor_copy(dst=st_sb, src=st_psum)
            for b in range(G):
                src = TensorView(st_sb).slice(dim=1, start=b, end=H0, step=G)
                nisa.dma_copy(
                    dst=attn_sb[b * H2 : (b + 1) * H2, 0, s * H2 : (s + 1) * H2, t],
                    src=src.get_view(),
                )
    return attn_sb


class Epilogue(nl.NKIObject):
    """Every post-matmul scratch tile, allocated BEFORE output_projection_tkg runs.

    ``output_projection_tkg`` returns its OUT_IN_SB tile from inside a scope it then closes, so that
    address range is nominally free again on return. Allocating the epilogue up front means nothing
    is ever placed over the logits while they are still being read, without paying for a second
    [T, V_core] copy.
    """

    def __init__(self, T, n_chunks, logits_dtype, sbm):
        f32 = nl.float32
        # Per-core search (staged_argmax).
        self.chunk_max = sbm.alloc_stack(
            (T, n_chunks), dtype=logits_dtype, buffer=nl.sbuf
        )
        self.core_max = sbm.alloc_stack((T, 1), dtype=logits_dtype, buffer=nl.sbuf)
        self.ind = sbm.alloc_stack(
            (T, FIND_INDEX_WIDTH), dtype=nl.uint32, buffer=nl.sbuf
        )
        self.hit = sbm.alloc_stack((T, 1), dtype=f32, buffer=nl.sbuf)
        self.cand = sbm.alloc_stack((T, 1), dtype=f32, buffer=nl.sbuf)
        self.best_r = sbm.alloc_stack((T, 1), dtype=f32, buffer=nl.sbuf)
        self.core_idx = sbm.alloc_stack((T, 1), dtype=f32, buffer=nl.sbuf)
        # LNC combine (combine_across_cores).
        self.send_buf = sbm.alloc_stack((T, 2), dtype=f32, buffer=nl.sbuf)
        self.recv_buf = sbm.alloc_stack((T, 2), dtype=f32, buffer=nl.sbuf)
        self.peak = sbm.alloc_stack((T, 1), dtype=f32, buffer=nl.sbuf)
        self.owns_peak = sbm.alloc_stack((T, 1), dtype=f32, buffer=nl.sbuf)
        self.r_self = sbm.alloc_stack((T, 1), dtype=f32, buffer=nl.sbuf)
        self.r_peer = sbm.alloc_stack((T, 1), dtype=f32, buffer=nl.sbuf)
        # Results.
        self.rank_max = sbm.alloc_stack((T, 1), dtype=logits_dtype, buffer=nl.sbuf)
        self.rank_idx_f = sbm.alloc_stack((T, 1), dtype=f32, buffer=nl.sbuf)
        self.rank_idx = sbm.alloc_stack((T, 1), dtype=nl.int32, buffer=nl.sbuf)


def staged_argmax(logits_sb, shard_base, ep):
    """Greedy argmax over this core's vocab shard into ``ep.core_max`` / ``ep.core_idx``.

    A DVE instruction is capped at MAX_REDUCE_WIDTH elements, so the V_core-wide search is staged
    over n_chunks chunks of C. ``nc_find_index8`` reports the FIRST positions matching the value it
    is given, so within a chunk the lowest index comes for free; across chunks the reversed index
    r = V_core - g does the rest, since maximising r minimises g and "no match in this chunk" falls
    out as r = 0 -- torch's LOWEST-index tie-break, which the gate at test_lm_head_kernel.py checks.

    ``shard_base`` is this core's first vocab column within V_rank (prg_id * V_core); ``core_idx`` is
    returned already offset by it.
    """
    T, V_core = logits_sb.shape
    n_chunks = div_ceil(V_core, MAX_REDUCE_WIDTH)
    C = div_ceil(V_core, n_chunks)

    for c in range(n_chunks):
        size = min(C, V_core - c * C)
        nisa.tensor_reduce(
            dst=ep.chunk_max[0:T, c : c + 1],
            data=logits_sb[0:T, nl.ds(c * C, size)],
            op=nl.maximum,
            axis=1,
        )
    nisa.tensor_reduce(dst=ep.core_max, data=ep.chunk_max, op=nl.maximum, axis=1)

    nisa.memset(dst=ep.best_r, value=0.0)
    for c in range(n_chunks):
        size = min(C, V_core - c * C)
        # Broadcast core_max across the 8 candidate slots. The comparison is exact -- core_max was
        # reduced from these very elements.
        nisa.nc_find_index8(
            data=logits_sb[0:T, nl.ds(c * C, size)],
            vals=ep.core_max.ap([[1, T], [0, FIND_INDEX_WIDTH]]),
            dst=ep.ind,
        )
        # ind[:, 0] is meaningless unless this chunk actually holds the core-wide max.
        nisa.tensor_tensor(
            dst=ep.hit,
            data1=ep.chunk_max[0:T, c : c + 1],
            data2=ep.core_max,
            op=nl.equal,
        )
        # Widen the uint32 index before any float arithmetic: arithmetic on the raw integer tile is
        # done in the integer dtype and wraps.
        nisa.tensor_copy(dst=ep.cand, src=ep.ind[0:T, 0:1])
        # r = V_core - (c*C + ind0), gated to 0 when this chunk holds no match.
        nisa.tensor_scalar(
            dst=ep.cand,
            data=ep.cand,
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.add,
            operand1=float(V_core - c * C),
        )
        nisa.tensor_tensor(dst=ep.cand, data1=ep.cand, data2=ep.hit, op=nl.multiply)
        nisa.tensor_tensor(dst=ep.best_r, data1=ep.best_r, data2=ep.cand, op=nl.maximum)

    # g = V_core - best_r, shifted into V_rank by this core's vocab-shard base.
    nisa.tensor_scalar(
        dst=ep.core_idx,
        data=ep.best_r,
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.add,
        operand1=float(shard_base + V_core),
    )


def combine_across_cores(ep, V_rank, n_prgs, prg_id):
    """Reduce the two cores' vocab-shard winners into ``ep.rank_max`` / ``ep.rank_idx``, on both cores.

    Each core owns a disjoint vocab shard, so a core-local winner is not yet the rank winner. The
    (max, index) pair is exchanged as one contiguous fp32 [T, 2] tile -- the repo idiom, since a
    strided sendrecv destination silently delivers nothing.

    The winner is then chosen symmetrically, with no dependence on which core is which: reuse the
    reversed index r = V_rank - idx, gated on holding the peak, and take the max. Maximising r
    minimises idx, so a tie between the two shards resolves to the lower vocab index exactly as
    torch.argmax does. Packing the max as fp32 keeps the equality test exact.
    """
    T, _ = ep.core_max.shape
    if n_prgs == 1:
        nisa.tensor_copy(dst=ep.rank_max, src=ep.core_max)
        nisa.tensor_copy(dst=ep.rank_idx, src=ep.core_idx)
        return

    nisa.tensor_copy(dst=ep.send_buf[0:T, 0:1], src=ep.core_max)
    nisa.tensor_copy(dst=ep.send_buf[0:T, 1:2], src=ep.core_idx)
    other = 1 - prg_id
    nisa.sendrecv(
        src=ep.send_buf,
        dst=ep.recv_buf,
        send_to_rank=other,
        recv_from_rank=other,
        pipe_id=0,
    )

    nisa.tensor_tensor(
        dst=ep.peak,
        data1=ep.send_buf[0:T, 0:1],
        data2=ep.recv_buf[0:T, 0:1],
        op=nl.maximum,
    )
    nisa.tensor_copy(dst=ep.rank_max, src=ep.peak)

    # Same three steps for each side; not loop-folded because the NKI tracer rejects
    # tuple-unpacking loop targets.
    nisa.tensor_tensor(
        dst=ep.owns_peak, data1=ep.send_buf[0:T, 0:1], data2=ep.peak, op=nl.equal
    )
    nisa.tensor_scalar(
        dst=ep.r_self,
        data=ep.send_buf[0:T, 1:2],
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.add,
        operand1=float(V_rank),
    )
    nisa.tensor_tensor(
        dst=ep.r_self, data1=ep.r_self, data2=ep.owns_peak, op=nl.multiply
    )

    nisa.tensor_tensor(
        dst=ep.owns_peak, data1=ep.recv_buf[0:T, 0:1], data2=ep.peak, op=nl.equal
    )
    nisa.tensor_scalar(
        dst=ep.r_peer,
        data=ep.recv_buf[0:T, 1:2],
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.add,
        operand1=float(V_rank),
    )
    nisa.tensor_tensor(
        dst=ep.r_peer, data1=ep.r_peer, data2=ep.owns_peak, op=nl.multiply
    )

    nisa.tensor_tensor(dst=ep.r_self, data1=ep.r_self, data2=ep.r_peer, op=nl.maximum)
    nisa.tensor_scalar(
        dst=ep.rank_idx_f,
        data=ep.r_self,
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.add,
        operand1=float(V_rank),
    )
    nisa.tensor_copy(dst=ep.rank_idx, src=ep.rank_idx_f)


def lm_head_compose(hidden, gamma, lm_head_w, eps=1e-6, sbm=None, name_prefix=""):
    """Final RMSNorm -> vocab-parallel matmul -> per-rank greedy argmax, SBUF-resident.

    A3B pads the vocab by 0, so padded-logit masking is not implemented.

    Args:
        hidden:    [H0=128, T, H1] SBUF (the megakernel tp2013 residual) or [B, S, H] HBM, detected
                   from the buffer. Left untouched (it is the residual).
        gamma:     [1, H] HBM final-norm weight in STANDARD form (no +1 applied here).
        lm_head_w: [H, V_rank] HBM bf16, consumed verbatim.
        eps:       RMSNorm epsilon (config.rms_norm_eps).
        sbm:       optional MANUAL BufferManager. Default None creates one over SBM_SIZE_BYTES; a
                   megakernel that runs its own manual manager must pass it, since two managers
                   would each believe they own the same region.

    Returns:
        (rank_max [T, 1], rank_idx [T, 1] int32, logits_sb [T, V_core]) -- all SBUF. The index is
        within V_rank; the caller adds rank_id * V_rank. ``logits_sb`` is this core's vocab shard.
    """
    hdim, V_rank = lm_head_w.shape
    if hidden.buffer == nl.sbuf:
        h0_in, T, h1 = hidden.shape
        kernel_assert(h0_in * h1 == hdim, "SBUF hidden H must match lm_head_w rows")
    else:
        B, S, h_in = hidden.shape
        T = B * S
        kernel_assert(h_in == hdim, "hidden H must match lm_head_w rows")

    _, n_prgs, prg_id = get_verified_program_sharding_info("lm_head", (0, 1), 2)
    kernel_assert(
        V_rank % n_prgs == 0, "V_rank must be divisible by the LNC core count"
    )
    V_core = V_rank // n_prgs

    # One manual manager owns every SBUF tile below, including rmsnorm's and the epilogue's
    # (see make_manual_sbuf_manager and Epilogue for why both are required).
    if sbm is None:
        sbm = make_manual_sbuf_manager()
    sbm.set_name_prefix(name_prefix + "lmhead_")
    # The scope stays open: the returned SBUF tiles must outlive this call, and closing it would
    # hand their address range back to the allocator.
    sbm.open_scope(name=name_prefix + "lm_head")
    ep = Epilogue(T, div_ceil(V_core, MAX_REDUCE_WIDTH), hidden.dtype, sbm)

    normed_sb = sbm.alloc_stack((H0, T, hdim // H0), dtype=hidden.dtype, buffer=nl.sbuf)
    rmsnorm_to_sbuf(
        hidden,
        gamma,
        eps=eps,
        normed_sb=normed_sb,
        single_core_forced=False,
        name_prefix=name_prefix + "lmhead_",
        sbm=sbm,
    )
    attn_sb = head_major_from_tp2013(normed_sb, n_prgs, sbm)

    logits_sb = output_projection_tkg(
        attention=attn_sb,
        weight=lm_head_w,
        bias=None,
        quantization_type=QuantizationType.NONE,
        TRANSPOSE_OUT=False,
        OUT_IN_SB=True,
        sbm=sbm,
    )

    staged_argmax(logits_sb, prg_id * V_core, ep)
    combine_across_cores(ep, V_rank, n_prgs, prg_id)
    return ep.rank_max, ep.rank_idx, logits_sb


@nki.jit
def lm_head_fwd(hidden, gamma, lm_head_w, eps=1e-6):
    """LM head entrypoint (LNC launch [n_prgs]): logits plus the per-rank greedy token, in HBM.

    Isolation-mode twin of ``lm_head_compose``: the megakernel keeps everything in SBUF, this stores
    results so a host test can read them. ``rank_max``/``rank_idx`` are emitted per core so the rows
    must agree, which checks the LNC combine.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("lm_head_fwd", (0, 1), 2)

    rank_max, rank_idx, logits_sb = lm_head_compose(hidden, gamma, lm_head_w, eps=eps)
    T, V_core = logits_sb.shape

    logits = nl.ndarray(
        (T, V_core * n_prgs), dtype=logits_sb.dtype, buffer=nl.shared_hbm
    )
    nisa.dma_copy(
        dst=logits[0:T, prg_id * V_core : (prg_id + 1) * V_core], src=logits_sb
    )
    # The results are [T, 1] SBUF columns -- one element per partition -- so the HBM row is written
    # through a [T, 1]-shaped view of it. A flat T-wide access pattern would instead read one
    # partition and run off the end of the tile into whatever is allocated next to it.
    max_out = nl.ndarray((n_prgs, T), dtype=rank_max.dtype, buffer=nl.shared_hbm)
    idx_out = nl.ndarray((n_prgs, T), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=TensorView(max_out)
        .select(dim=0, index=prg_id)
        .expand_dim(dim=1)
        .get_view(),
        src=rank_max,
    )
    nisa.dma_copy(
        dst=TensorView(idx_out)
        .select(dim=0, index=prg_id)
        .expand_dim(dim=1)
        .get_view(),
        src=rank_idx,
    )
    return logits, max_out, idx_out
