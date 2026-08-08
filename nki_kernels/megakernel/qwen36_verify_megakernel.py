"""Qwen3.6-A3B verify-trunk megakernel: all decoder layers in one LNC2 launch.

The residual stays SBUF-resident across every layer (tp2013 shard-interleaved [H0, T*H1]);
only the entry load and final store touch HBM. Each layer runs attention (DeltaNet or GQA)
then the MoE FFN, each followed by an in-kernel TP all-reduce + LNC gather of its per-rank
partial back into the residual. Structure mirrors nkilib's transformer_tkg.

Attention H-shards its o_proj output across the two LNC cores -> H-gather. MoE token-shards
its output -> token-gather. Both reduce across the TP replica group first.

The finished residual then runs the final norm + vocab head + greedy argmax, so the kernel returns
token ids directly; the residual itself is still returned pre-final-norm because the caller needs
it as the draft's rolling-buffer seed.

Not decorated: wrap with nki.jit() at the call site (avoids a double-jit stack overflow).
"""

import linecache

import nki.isa as nisa
import nki.language as nl
import nki.collectives as nccl
from nki import jit as nki_jit

from nkilib.core.utils.tensor_view import TensorView
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from ..deltanet.decode.fused_layer import attention_layer_compose
from ..gqa.decode.fused_layer import gqa_fused_compose
from ..lm_head.components.lm_head import lm_head_compose
from ..moe.components.moe_layer import moe_layer_compose
from ..moe.components.shared_expert import moe_tkg_shard_decision


def load_residual_to_sbuf(dst_sb, src_hbm, T, H0, H1, n_prgs):
    """[B, S, H] HBM -> [H0, T*H1] SBUF, tp2013 shard-interleaved (free = t*H1 + shard*H1_shard + h2)."""
    src_view = TensorView(src_hbm.reshape((T, H0 * H1))).rearrange(
        ("bs", ("lnc", "h0", "h1")),
        ("h0", "bs", "lnc", "h1"),
        {"lnc": n_prgs, "h0": H0},
    )
    dst = dst_sb.reshape((H0, T, n_prgs, H1 // n_prgs))
    for lnc in nl.static_range(n_prgs):
        nisa.dma_copy(
            src=src_view.slice(dim=2, start=lnc, end=lnc + 1).get_view(),
            dst=dst[:, :, lnc : lnc + 1, :],
        )


def store_residual_to_hbm(dst_hbm, src_sb, T, H0, H1, n_prgs):
    """Inverse of load_residual_to_sbuf: [H0, T*H1] SBUF -> [B, S, H] HBM."""
    src = src_sb.reshape((H0, T, n_prgs, H1 // n_prgs))
    dst_view = TensorView(dst_hbm.reshape((T, H0 * H1))).rearrange(
        ("bs", ("lnc", "h0", "h1")),
        ("h0", "bs", "lnc", "h1"),
        {"lnc": n_prgs, "h0": H0},
    )
    for lnc in nl.static_range(n_prgs):
        nisa.dma_copy(
            src=src[:, :, lnc : lnc + 1, :],
            dst=dst_view.slice(dim=2, start=lnc, end=lnc + 1).get_view(),
        )


def all_reduce_gather_h(sharded_sb, rg, prg_id, n_prgs, T):
    """Attention path: TP all-reduce the [H0, H1_shard*T] H-shard, LNC-gather to full [H0, T*H1].

    ``rg=None`` skips the collective (identity at TP=1) so the LNC gather can run in a
    single-process launch, where collectives comms are uninitialized and fail at NEFF load.
    """
    dtype = sharded_sb.dtype
    H0 = sharded_sb.shape[0]
    H1_shard = sharded_sb.shape[1] // T
    H1 = H1_shard * n_prgs
    if rg is None:
        reduced = sharded_sb
    else:
        reduced = nl.ndarray(sharded_sb.shape, dtype=dtype, buffer=nl.sbuf)
        nccl.all_reduce(dsts=[reduced], srcs=[sharded_sb], op=nl.add, replica_group=rg)

    gathered = nl.ndarray((H0, H1 * T), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=gathered[:, nl.ds(start=prg_id * T * H1_shard, size=T * H1_shard)],
        src=reduced,
    )
    if n_prgs > 1:
        other = 1 - prg_id
        nisa.sendrecv(
            src=reduced,
            dst=gathered[:, nl.ds(start=other * T * H1_shard, size=T * H1_shard)],
            send_to_rank=other,
            recv_from_rank=other,
            pipe_id=0,
        )

    out = nl.ndarray((H0, T * H1), dtype=dtype, buffer=nl.sbuf)
    src_view = TensorView(gathered).rearrange(
        ("h0", ("h1", "bs")), ("h0", "bs", "h1"), {"h1": H1}
    )
    nisa.tensor_copy(dst=out.reshape((H0, T, H1)), src=src_view.get_view())
    return out


def all_reduce_gather_tokens(local_sb, rg, prg_id, n_prgs, T_offset, T_len):
    """MoE path: TP all-reduce this core's token block, LNC-gather the other core's block -> full [H0, T*H1].

    Token-sharded MoE lays tokens on the free axis (f = t*H1 + h1), so a token block is a contiguous
    free-slice; nccl SBUF collectives require 2D, so the reduce runs on the flat [H0, T*H1] view.

    ``rg=None`` skips the collective (identity at TP=1) so the LNC gather can run in a
    single-process launch, where collectives comms are uninitialized and fail at NEFF load.
    """
    dtype = local_sb.dtype
    H0, T, H1 = local_sb.shape
    flat = local_sb.reshape((H0, T * H1))
    # This core's token block is a free-slice of flat, so it inherits flat's partition stride (T*H1)
    # while spanning only T_len*H1 -- a partition-strided view. nccl SBUF all_reduce needs a densely
    # packed operand, so copy into a fresh [H0, T_len*H1] tile first.
    block = flat[:, T_offset * H1 : (T_offset + T_len) * H1]
    block_in = nl.ndarray((H0, T_len * H1), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=block_in, src=block)
    if rg is None:
        reduced = block_in
    else:
        reduced = nl.ndarray((H0, T_len * H1), dtype=dtype, buffer=nl.sbuf)
        nccl.all_reduce(dsts=[reduced], srcs=[block_in], op=nl.add, replica_group=rg)

    out = nl.ndarray((H0, T * H1), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out[:, T_offset * H1 : (T_offset + T_len) * H1], src=reduced)
    if n_prgs > 1:
        other = 1 - prg_id
        other_off = 0 if prg_id == 1 else T_offset + T_len
        other_len = T - T_len
        nisa.sendrecv(
            src=reduced,
            dst=out[:, other_off * H1 : (other_off + other_len) * H1],
            send_to_rank=other,
            recv_from_rank=other,
            pipe_id=0,
        )
    return out


def all_gather_argmax(rank_max, rank_idx, rg, tp_degree, V_rank):
    """LM-head path: fold the per-rank (max, index) winners into one global greedy token id.

    The vocab is TP-sharded, so a rank-local winner is not yet the global one. Rather than gather
    [T, V_global] logits, gather only each rank's [T, 1] winner pair -- the reduction NxD's
    ``nxd_argmax`` does on the host. Gathering rather than reducing lands each rank's entry at a
    known column, so the vocab offset is a compile-time constant and no dynamic rank id is needed.

    Ties resolve to the lowest global id, matching torch.argmax: columns that do not hold the peak
    are lifted past every real id, then the row is min-reduced. The rank owning the peak always
    survives, so the reduce is never over an all-loser row.

    ``rg=None`` skips the collective (identity at TP=1) so the head can still run in a
    single-process launch, where collectives comms are uninitialized and fail at NEFF load.
    """
    if rg is None:
        return rank_idx

    T = rank_max.shape[0]
    # Above any real global vocab id and exact in fp32, which holds ids up to 2**24.
    loser = float(1 << 30)

    val = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    idx = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=val, src=rank_max)
    nisa.tensor_copy(dst=idx, src=rank_idx)

    all_val = nl.ndarray((T, tp_degree), dtype=nl.float32, buffer=nl.sbuf)
    all_idx = nl.ndarray((T, tp_degree), dtype=nl.float32, buffer=nl.sbuf)
    nccl.all_gather(srcs=[val], dsts=[all_val], replica_group=rg, collective_dim=1)
    nccl.all_gather(srcs=[idx], dsts=[all_idx], replica_group=rg, collective_dim=1)

    peak = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=peak, op=nl.maximum, data=all_val, axis=1)

    cand = nl.ndarray((T, tp_degree), dtype=nl.float32, buffer=nl.sbuf)
    penalty = nl.ndarray((T, tp_degree), dtype=nl.float32, buffer=nl.sbuf)
    for r in nl.static_range(tp_degree):
        nisa.tensor_scalar(
            dst=cand[:, r : r + 1],
            data=all_idx[:, r : r + 1],
            op0=nl.add,
            operand0=float(r * V_rank),
        )
    nisa.tensor_tensor(
        dst=penalty,
        data1=all_val,
        data2=peak.ap([[1, T], [0, tp_degree]]),
        op=nl.equal,
    )
    # (1 - owns_peak) * loser, fused into one instruction.
    nisa.tensor_scalar(
        dst=penalty,
        data=penalty,
        op0=nl.multiply,
        operand0=-loser,
        op1=nl.add,
        operand1=loser,
    )
    nisa.tensor_tensor(dst=cand, data1=cand, data2=penalty, op=nl.add)

    best = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    tokens = nl.ndarray((T, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=best, op=nl.minimum, data=cand, axis=1)
    nisa.tensor_copy(dst=tokens, src=best)
    return tokens


def qwen36_verify_megakernel(
    X,
    layer_is_gqa,
    key_dim,
    eps,
    replica_groups,
    # DeltaNet layers (indexed by DeltaNet position)
    dn_proj_w,
    dn_in_gamma,
    dn_conv_state,
    dn_conv_weight,
    dn_A_log,
    dn_dt_bias,
    dn_init_state,
    dn_out_w,
    dn_z_gamma,
    # GQA layers (indexed by GQA position; cos/sin/mask shared)
    gqa_qkv_w,
    gqa_gate_w,
    gqa_gamma_q,
    gqa_gamma_k,
    gqa_in_gamma,
    gqa_o_proj_w,
    gqa_k_cache,
    gqa_v_cache,
    cos,
    sin,
    gqa_mask,
    # MoE (every layer)
    moe_gamma,
    moe_router_w,
    moe_gate_up_w,
    moe_down_w,
    moe_sigma_gate_w,
    moe_shared_gate_w,
    moe_shared_up_w,
    moe_shared_down_w,
    # LM head
    final_gamma,
    lm_head_w,
):
    """Run all decoder layers, SBUF-resident residual, per-rank partials reduced in-kernel.

    ``layer_is_gqa[i]`` selects the attention type for layer i; DeltaNet/GQA weight lists are
    indexed by each type's running position. Returns
    ``(tokens [B,S] int32 HBM, hidden [B,S,H] HBM, gqa_active_kv, dn_cand)``: the greedy token ids,
    the trunk hidden, per-GQA-layer (active_k, active_v) for the caller's KV scatter, and
    per-DeltaNet-layer (candidate_states, conv_cand) for the accept/reject commit.

    ``hidden`` is PRE-final-norm: the head applies its own norm, and the caller needs the un-normed
    hidden as the draft's rolling-buffer seed.
    """
    B, S, H = X.shape
    dtype = X.dtype
    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "qwen36_verify_megakernel", (0, 1), 2
    )
    H0 = nl.tile_size.pmax
    H1 = H // H0
    H1_shard = H1 // n_prgs
    T = B * S
    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None
    tp_degree = len(replica_groups[0]) if replica_groups is not None else 1

    residual = nl.ndarray((H0, T * H1), dtype=dtype, buffer=nl.sbuf)
    load_residual_to_sbuf(residual, X, T, H0, H1, n_prgs)

    gqa_out = []
    dn_out = []
    dn = 0
    gqa = 0
    for i in range(len(layer_is_gqa)):
        pfx = f"L{i}_"
        x_sb = residual.reshape((H0, T, H1))

        if layer_is_gqa[i]:
            attn_partial, active_k, active_v = gqa_fused_compose(
                x_sb,
                gqa_qkv_w[gqa],
                gqa_gate_w[gqa],
                gqa_gamma_q[gqa],
                gqa_gamma_k[gqa],
                cos,
                sin,
                gqa_k_cache[gqa],
                gqa_v_cache[gqa],
                gqa_mask,
                gqa_o_proj_w[gqa],
                eps,
                gamma_in=gqa_in_gamma[gqa],
                out_in_sb=True,
                name_prefix=pfx,
            )
            gqa_out.append(active_k)
            gqa_out.append(active_v)
            gqa += 1
        else:
            conv_dim = dn_conv_weight[dn].shape[0]
            state_w = dn_conv_weight[dn].shape[1] - 1
            Hv = (conv_dim - 2 * key_dim) // H0
            cand_state = nl.ndarray(
                (T, Hv, H0, H0), dtype=nl.float32, buffer=nl.shared_hbm
            )
            conv_cand = nl.ndarray(
                (T, conv_dim, state_w),
                dtype=dn_conv_weight[dn].dtype,
                buffer=nl.shared_hbm,
            )
            attn_partial = attention_layer_compose(
                x_sb,
                dn_proj_w[dn],
                dn_in_gamma[dn],
                eps,
                dn_conv_state[dn],
                dn_conv_weight[dn],
                key_dim,
                dn_A_log[dn],
                dn_dt_bias[dn],
                dn_init_state[dn],
                dn_out_w[dn],
                cand_state,
                conv_cand,
                write_candidates=True,
                cand_is_3d=True,
                z_gamma=dn_z_gamma[dn],
                z_eps=eps,
                out_in_sb=True,
                name_prefix=pfx,
            )
            dn_out.append(cand_state)
            dn_out.append(conv_cand)
            dn += 1

        # attn_partial is the H-sharded o_proj output [H0, H1_shard*T] (out_in_sb + transposed_out).
        attn_out = all_reduce_gather_h(
            attn_partial.reshape((H0, H1_shard * T)), rg, prg_id, n_prgs, T
        )
        nisa.tensor_tensor(dst=residual, data1=residual, data2=attn_out, op=nl.add)

        moe_partial = moe_layer_compose(
            residual.reshape((H0, T, H1)),
            moe_gamma[i],
            moe_router_w[i],
            moe_gate_up_w[i],
            moe_down_w[i],
            moe_sigma_gate_w[i],
            moe_shared_gate_w[i],
            moe_shared_up_w[i],
            moe_shared_down_w[i],
            eps=eps,
            output_in_sbuf=True,
            name_prefix=pfx,
        )
        _, T_offset, T_len = moe_tkg_shard_decision(T, H, moe_gate_up_w[i].shape[3])
        moe_out = all_reduce_gather_tokens(
            moe_partial, rg, prg_id, n_prgs, T_offset, T_len
        )
        nisa.tensor_tensor(dst=residual, data1=residual, data2=moe_out, op=nl.add)

    output = nl.ndarray((B, S, H), dtype=dtype, buffer=nl.shared_hbm)
    if prg_id == 0:
        store_residual_to_hbm(output, residual, T, H0, H1, n_prgs)
    if n_prgs > 1:
        nisa.core_barrier(data=output, cores=(0, 1))

    # residual is still the pre-final-norm hidden stored above; the head norms its own copy.
    rank_max, rank_idx, _ = lm_head_compose(
        residual.reshape((H0, T, H1)),
        final_gamma,
        lm_head_w,
        eps=eps,
        name_prefix="lm_",
    )
    token_idx = all_gather_argmax(rank_max, rank_idx, rg, tp_degree, lm_head_w.shape[1])
    # [T, 1] SBUF is one element per partition, so the HBM row is written through a [T, 1] view
    # of it; a flat T-wide access pattern would read one partition and run off the end.
    tokens = nl.ndarray((B, S), dtype=nl.int32, buffer=nl.shared_hbm)
    if prg_id == 0:
        nisa.dma_copy(dst=tokens.reshape((T, 1)), src=token_idx)
    if n_prgs > 1:
        nisa.core_barrier(data=tokens, cores=(0, 1))
    return tuple([tokens, output] + gqa_out + dn_out)


DN_FIELDS = (
    "proj_w",
    "in_gamma",
    "conv_state",
    "conv_weight",
    "A_log",
    "dt_bias",
    "init_state",
    "out_w",
    "z_gamma",
)
GQA_FIELDS = (
    "qkv_w",
    "gate_w",
    "gamma_q",
    "gamma_k",
    "in_gamma",
    "o_proj_w",
    "k_cache",
    "v_cache",
)
MOE_FIELDS = (
    "gamma",
    "router_w",
    "gate_up_w",
    "down_w",
    "sigma_gate_w",
    "shared_gate_w",
    "shared_up_w",
    "shared_down_w",
)


def flatten_megakernel_args(
    X,
    dn,
    gqa,
    moe,
    cos,
    sin,
    gqa_mask,
    key_dim,
    eps,
    replica_groups,
    final_gamma,
    lm_head_w,
):
    """The one definition of the flat positional argument order.

    ``dn``/``gqa``/``moe`` map each field name to its per-layer list. Used both by
    ``build_verify_megakernel`` (over parameter NAMES, to generate the wrapper signature) and by
    the caller (over tensor VALUES), so the two cannot drift.
    """
    flat = [X]
    for f in DN_FIELDS:
        flat += list(dn[f])
    for f in GQA_FIELDS:
        flat += list(gqa[f])
    flat += [cos, sin, gqa_mask]
    for f in MOE_FIELDS:
        flat += list(moe[f])
    flat += [final_gamma, lm_head_w]
    flat += [key_dim, eps, replica_groups]
    return flat


def split_megakernel_returns(rets, n_gqa, n_dn):
    """Un-flatten the return tuple into (tokens, hidden, gqa_active_kv, dn_candidates)."""
    gqa_flat = rets[2 : 2 + 2 * n_gqa]
    dn_flat = rets[2 + 2 * n_gqa :]
    return (
        rets[0],
        rets[1],
        [(gqa_flat[2 * i], gqa_flat[2 * i + 1]) for i in range(n_gqa)],
        [(dn_flat[2 * i], dn_flat[2 * i + 1]) for i in range(n_dn)],
    )


_MEGAKERNEL_CACHE = {}


def build_verify_megakernel(layer_is_gqa):
    key = tuple(bool(x) for x in layer_is_gqa)
    if key in _MEGAKERNEL_CACHE:
        return _MEGAKERNEL_CACHE[key]

    n = len(key)
    n_dn = sum(1 for x in key if not x)
    n_gqa = sum(1 for x in key if x)

    def col(prefix, fields, count):
        return {f: [f"{prefix}_{f}_{j}" for j in range(count)] for f in fields}

    dn = col("dn", DN_FIELDS, n_dn)
    gqa = col("gqa", GQA_FIELDS, n_gqa)
    moe = col("moe", MOE_FIELDS, n)

    flat = flatten_megakernel_args(
        "X",
        dn,
        gqa,
        moe,
        "cos",
        "sin",
        "gqa_mask",
        "key_dim",
        "eps",
        "replica_groups",
        "final_gamma",
        "lm_head_w",
    )

    def tup(ns):
        return "(" + ", ".join(ns) + ("," if ns else "") + ")"

    lines = ["def _verify_wrapper(\n    " + ",\n    ".join(flat) + ",\n):"]
    for f in DN_FIELDS:
        lines.append(f"    dn_{f} = {tup(dn[f])}")
    for f in GQA_FIELDS:
        lines.append(f"    gqa_{f} = {tup(gqa[f])}")
    for f in MOE_FIELDS:
        lines.append(f"    moe_{f} = {tup(moe[f])}")
    body_args = ["X", repr(key), "key_dim", "eps", "replica_groups"]
    body_args += [f"dn_{f}" for f in DN_FIELDS]
    body_args += [f"gqa_{f}" for f in GQA_FIELDS]
    body_args += ["cos", "sin", "gqa_mask"]
    body_args += [f"moe_{f}" for f in MOE_FIELDS]
    body_args += ["final_gamma", "lm_head_w"]
    lines.append(
        "    return _body(\n        " + ",\n        ".join(body_args) + ",\n    )"
    )
    src = "\n".join(lines) + "\n"

    fname = f"<verify_megakernel_L{n}_g{n_gqa}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
    code = compile(src, fname, "exec")
    ns = {"_body": qwen36_verify_megakernel}
    exec(code, ns)
    jitted = nki_jit(ns["_verify_wrapper"])
    _MEGAKERNEL_CACHE[key] = jitted
    return jitted
