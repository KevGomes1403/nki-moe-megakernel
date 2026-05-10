"""
Qwen3-MoE multi-layer fused TKG megakernel.

Runs all `num_layers` decoder layers inside a single NKI kernel invocation, with
the residual kept SBUF-resident across layer boundaries and KV caches updated
in-place by v14a attention. Bypasses NxDI's Python-side per-layer loop.

Per-layer pipeline (SBUF-resident residual):
  1. attention (v14a_kv_norm): pre-attention RMSNorm(residual_sb, gamma_pre_attn[i])
     is applied inside the attention sub-function; scatter-DMA writes new K/V
     rows of K_caches[i] / V_caches[i] at position_ids.
  2. AllReduce #1
  3. residual_sb += attn_reduced
  4. MoE (_qwen3_moe_sbuf_in_sbuf_out): post-attention RMSNorm with
     gamma_post_attn[i] applied inside the MoE sub-function.
  5. SB2SB AR-gather over TP × LNC → full tensor on both cores.
  6. residual_sb += moe_gathered

After the last layer: single HBM store of the post-residual hidden state.
Final `model.norm` is applied by NxDI's `get_model_output` on the Python side.

Target: Trainium3, TP=4, LNC=2.
Entry point: transformer_qwen3_moe_tkg_multilayer_jit[2](...)

Argument layout (see plan §2.3):
  * Per-layer weights are STACKED along a leading layer dim at the call site
    (torch.stack). The kernel slices them via `Wq_all[i]` etc. — NkiTensor
    integer indexing on HBM tensors reduces the layer dim to a plain
    [Hq_out, H] view with no runtime cost.
  * Per-layer KV caches are passed as INDIVIDUAL tensor args (K_00..K_{L-1},
    V_00..V_{L-1}). They can't be stacked because each one is a separately
    aliased NxDI Parameter (in-place scatter DMA updates them).

The exposed kernel function is built by `_build_multilayer_kernel(num_layers)`
which code-gens the explicit KV signature via `exec`. This keeps the per-layer
loop body clean while satisfying NKI's "top-level args must be individual
tensors" constraint.
"""

import linecache

import nki
import nki.isa as nisa
import nki.language as nl
import nki.collectives as nccl
from nkilib.core.utils.allocator import BufferManager, Logger
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.experimental.transformer.transformer_tkg import (
    _sb2sb_all_reduce_gather,
)
from megakernels.qwen3_moe.attn_fused_nki import attn_fused_qwen
from megakernels.qwen3_moe.moe_fused_nki import _qwen3_moe_sbuf_in_sbuf_out_hoisted

H        = 2048
H0       = 128
H1       = H // H0
N_PRGS   = 2
H1_SHARD = H1 // N_PRGS
EPS      = 1e-6
PMAX     = 128
NH       = H // PMAX   # num_h_tiles = 16

SBM_SIZE_BYTES = 200 * 1024

NUM_LAYERS = 48  # Qwen3-30B-A3B


def _multilayer_body(
    X,             # [B, S_tkg, H]         bf16  HBM  — raw hidden state (pre-norm)
    Wq_list,       # tuple of L tensors, each [Hq_tp*d, H]      bf16  HBM  — per-layer Q projection
    Wk_list,       # tuple of L tensors, each [Hkv_tp*d, H]     bf16  HBM  — per-layer K projection
    Wv_list,       # tuple of L tensors, each [Hkv_tp*d, H]     bf16  HBM  — per-layer V projection
    Wo_list,       # tuple of L tensors, each [Hq_tp*d, H]      bf16  HBM  — per-layer O projection
    qn_list,       # tuple of L tensors, each [d]               bf16  HBM  — per-layer Q RMSNorm
    kn_list,       # tuple of L tensors, each [d]               bf16  HBM  — per-layer K RMSNorm
    gpre_list,     # tuple of L tensors, each [H]               bf16  HBM  — per-layer pre-attn layernorm
    gpost_list,    # tuple of L tensors, each [1, H]            bf16  HBM  — per-layer post-attn layernorm
    router_list,   # tuple of L tensors, each [H, E]            bf16  HBM  — per-layer router weights
    gate_up_list,  # tuple of L tensors, each [E, H, 384]       bf16  HBM  — per-layer gate+up projection weights
    down_list,     # tuple of L tensors, each [E, 192, H]       bf16  HBM  — per-layer down projection weights
    K_caches,      # tuple of L tensors, each [B, 1, S_prior, d]  bf16  HBM  — per-layer K caches (mutated in-place)
    V_caches,      # tuple of L tensors, each [B, 1, S_prior, d]  bf16  HBM  — per-layer V caches (mutated in-place)
    cos,           # [B, d]                bf16  HBM  — RoPE cosine, pre-indexed at position_ids
    sin,           # [B, d]                bf16  HBM  — RoPE sine,   pre-indexed at position_ids
    position_ids,  # [B, 1]                int32 HBM  — token position in the KV cache
    num_layers,    # int scalar — number of decoder layers (== L above)
    replica_groups=None,
):
    """Kernel body — runs `num_layers` fused decoder layers.

    Weight tensors are stacked along a leading layer dim; this function slices
    them via integer indexing per iteration. KV caches arrive as a Python tuple
    of already-registered NkiTensors (built by the code-gen wrapper).
    """
    B, S_tkg, _ = X.shape
    dtype = X.dtype
    BxS = B * S_tkg
    T   = BxS

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "transformer_qwen3_moe_tkg_multilayer", (0, 1), N_PRGS
    )

    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("transformer_qwen3_moe_tkg_multilayer"))
    sbm.set_auto_alloc(True)

    # -----------------------------------------------------------------------
    # Load X into SBUF as the initial residual [H0, BxS*H1].
    # This tensor is updated in-place at each layer boundary
    # (attn residual add, then MoE residual add).
    # -----------------------------------------------------------------------
    # Plain-linear layout: residual_sb[p, b*H1 + t] = X[b, 0, t*H0 + p].
    # Matches what v14a (gamma_pre_attn load) and v30a (gamma load) expect;
    # do NOT use _load_input_to_sbuf (channel-interleaved — scrambles RMSNorm).
    residual_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                              name="residual_sb")
    X_flat = X.reshape((BxS * H,))
    residual_load_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                   name="residual_load_sb")
    nisa.dma_copy(
        dst=residual_load_sb,
        src=X_flat.ap(pattern=[[1, H0], [H0, H1], [H, BxS]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )
    nisa.tensor_copy(residual_sb, residual_load_sb)

    # -----------------------------------------------------------------------
    # Hoist layer-invariant attention constants out of the per-layer loop
    # -----------------------------------------------------------------------
    sbm.set_name_prefix("hoist_")
    sbm.open_scope(name="hoist")
    sbm.set_auto_alloc(True)

    cos_col = cos.reshape((PMAX, B))
    sin_col = sin.reshape((PMAX, B))

    cos_bf16_all = sbm.alloc_stack((PMAX, B), dtype, name="cos_bf16_all")
    sin_bf16_all = sbm.alloc_stack((PMAX, B), dtype, name="sin_bf16_all")
    nisa.dma_copy(dst=cos_bf16_all, src=cos_col, dge_mode=nisa.dge_mode.hwdge)
    nisa.dma_copy(dst=sin_bf16_all, src=sin_col, dge_mode=nisa.dge_mode.hwdge)
    cos_f32_all = sbm.alloc_stack((PMAX, B), nl.float32, name="cos_f32_all")
    sin_f32_all = sbm.alloc_stack((PMAX, B), nl.float32, name="sin_f32_all")
    nisa.tensor_copy(cos_f32_all, cos_bf16_all)
    nisa.tensor_copy(sin_f32_all, sin_bf16_all)

    # Stacked per-layer constants. Shapes match one slice per layer matching
    # what v14c expects internally: qnw/knw = [PMAX, 1], gpan = [PMAX, NH].
    qnw_bf16_all  = sbm.alloc_stack((PMAX, num_layers),      dtype, name="qnw_bf16_all")
    knw_bf16_all  = sbm.alloc_stack((PMAX, num_layers),      dtype, name="knw_bf16_all")
    gpan_bf16_all = sbm.alloc_stack((PMAX, num_layers * NH), dtype, name="gpan_bf16_all")

    # Each DMA is independent across layers — compiler can parallelize them.
    for li in range(num_layers):
        nisa.dma_copy(
            dst=qnw_bf16_all[0:PMAX, li:li + 1],
            src=qn_list[li].reshape((PMAX, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.dma_copy(
            dst=knw_bf16_all[0:PMAX, li:li + 1],
            src=kn_list[li].reshape((PMAX, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        # gpan layout: gpan_bf16[p, h_t] = gpre[h_t*PMAX + p]
        nisa.dma_copy(
            dst=gpan_bf16_all[0:PMAX, li * NH:(li + 1) * NH],
            src=gpre_list[li].reshape((H, 1)).ap(
                pattern=[[1, PMAX], [PMAX, NH]], offset=0,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )

    # Bulk bf16→f32 cast (Vector engine, three instructions total).
    qnw_f32_all  = sbm.alloc_stack((PMAX, num_layers),      nl.float32, name="qnw_f32_all")
    knw_f32_all  = sbm.alloc_stack((PMAX, num_layers),      nl.float32, name="knw_f32_all")
    gpan_f32_all = sbm.alloc_stack((PMAX, num_layers * NH), nl.float32, name="gpan_f32_all")
    nisa.tensor_copy(qnw_f32_all,  qnw_bf16_all)
    nisa.tensor_copy(knw_f32_all,  knw_bf16_all)
    nisa.tensor_copy(gpan_f32_all, gpan_bf16_all)

    # -----------------------------------------------------------------------
    # Plan B steps 4 & 5: double-buffered cross-layer weight prefetch.
    # Uses an interleave_degree=2 sub-scope: each layer allocates a fresh slot
    # via sbm.alloc_stack and increment_section rotates the physical SBUF
    # backing between two sections, so current-layer consumption and
    # next-layer prefetch don't alias.
    #
    # Step 4: Wk, Wv     (1× tensor per slot)
    # Step 5: Wq, Wo     (NOH=4 owned-head tensors per slot — LNC sharded)
    #         Router_w   (wide [PMAX, ROUTER_BATCH=16, E=128] form per slot)
    # -----------------------------------------------------------------------

    # Owned head indices mirror v14d's internal LNC sharding (4 heads/core at N_PRGS=2).
    # Used both at prefetch time (which Wq/Wo head slice to DMA) and to align
    # the per-head SBUF list passed back into v14d.
    if prg_id == 0:
        OWNED = [0, 1, 2, 3]
    else:
        OWNED = [4, 5, 6, 7]
    NOH = 4
    HQ_TP = 8           # total Q heads per TP rank — used in Wo reshape
    ROUTER_BATCH = 16   # mirrors _ROUTER_BATCH in v30c
    E = 128             # num experts (Qwen3-30B-A3B)

    # Switch to explicit-address mode so interleave_degree=2 actually cycles
    # the SBUF backing. With auto_alloc=True the allocator ignores the
    # section cursor (see _get_safe_batch_interleave_degree in attention_tkg).
    sbm.set_auto_alloc(True)
    sbm.set_name_prefix("weights_")
    # interleave_degree=2 gives the compiler a second physical slot per
    # per-iteration alloc_stack, so it can schedule iter (i+1)'s weight DMAs
    # in parallel with iter i's attention/MoE compute without any explicit
    # prefetch loop on our part.
    sbm.open_scope(interleave_degree=2, name="weight_db")

    for layer_idx in range(num_layers):
        # Fresh per-iter alloc in the current ring section; increment_section
        # at the end of the iter rotates the next alloc into the other slot.
        Wk_cur = sbm.alloc_stack((PMAX, NH * PMAX), dtype, name=f"Wk_L{layer_idx}")
        Wv_cur = sbm.alloc_stack((PMAX, NH * PMAX), dtype, name=f"Wv_L{layer_idx}")
        Wq_cur = []
        Wo_cur = []
        for hi in range(NOH):
            Wq_cur.append(sbm.alloc_stack(
                (PMAX, NH * PMAX), dtype, name=f"Wq_L{layer_idx}_h{hi}"
            ))
            Wo_cur.append(sbm.alloc_stack(
                (PMAX, H), dtype, name=f"Wo_L{layer_idx}_h{hi}"
            ))
        Wq_cur = tuple(Wq_cur)
        Wo_cur = tuple(Wo_cur)
        Router_cur = sbm.alloc_stack(
            (PMAX, ROUTER_BATCH, E), nl.float32, name=f"Router_L{layer_idx}"
        )

        # HBM → SBUF loads for this layer's weights. No explicit "prefetch":
        # interleave_degree=2 lets the compiler overlap these with the
        # previous iteration's compute automatically.
        nisa.dma_copy(dst=Wk_cur, src=Wk_list[layer_idx], dge_mode=nisa.dge_mode.hwdge)
        nisa.dma_copy(dst=Wv_cur, src=Wv_list[layer_idx], dge_mode=nisa.dge_mode.hwdge)
        for hi in range(NOH):
            q_h = OWNED[hi]
            nisa.dma_copy(
                dst=Wq_cur[hi],
                src=Wq_list[layer_idx][q_h * PMAX:(q_h + 1) * PMAX, :],
                dge_mode=nisa.dge_mode.hwdge,
            )
            # Wo AP pattern matches v14d's internal Wo loader (lines 512-521):
            # Wo.reshape((Hq_tp, d, H_wo)) then per-head [PMAX, H_wo] tile via stride pattern.
            nisa.dma_copy(
                dst=Wo_cur[hi],
                src=Wo_list[layer_idx].reshape((HQ_TP, PMAX, H)).ap(
                    pattern=[[H, PMAX], [1, H]],
                    offset=q_h * PMAX * H,
                ),
                dge_mode=nisa.dge_mode.hwdge,
            )
        # Router wide DMA — single h_chunk since H_free=16 == ROUTER_BATCH=16.
        # Pattern mirrors v30c lines 192-199.
        nisa.dma_copy(
            dst=Router_cur,
            src=router_list[layer_idx].ap(
                pattern=[[E, PMAX], [PMAX * E, ROUTER_BATCH], [1, E]],
                offset=0,
            ),
            dge_mode=3,
        )

        # ------------------------------------------------------------------
        # ATTENTION (v14a_kv_norm)
        # Slicing Wq_all[layer_idx] etc. reduces the leading [L] dim via
        # NkiTensor integer indexing — no runtime copy or transpose.
        # Pre-attn RMSNorm (input_layernorm) is fused inside the sub-function.
        # K/V are scattered in-place into K_caches[layer_idx] / V_caches[layer_idx].
        # ------------------------------------------------------------------
        sbm.set_name_prefix(f"L{layer_idx}_attn_")
        sbm.set_auto_alloc(True)

        hidden_sb_bf16 = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                     name=f"L{layer_idx}_hidden_bf16")
        nisa.tensor_copy(hidden_sb_bf16, residual_sb)

        out_sb = attn_fused_qwen(
            hidden_sb=hidden_sb_bf16,
            Wq=Wq_list[layer_idx],          # [Hq_tp*d, H]  = [1024, 2048]
            Wk=Wk_list[layer_idx],          # HBM fallback — unused when wk_sb is set
            Wv=Wv_list[layer_idx],          # HBM fallback — unused when wv_sb is set
            Wo=Wo_list[layer_idx],          # [Hq_tp*d, H]  = [1024, 2048]
            q_norm_weight=qn_list[layer_idx],    # HBM fallback — unused when qnw_f32_sb is set
            k_norm_weight=kn_list[layer_idx],    # HBM fallback — unused when knw_f32_sb is set
            gamma_pre_attn=gpre_list[layer_idx], # HBM fallback — unused when gpan_f32_sb is set
            K_cache=K_caches[layer_idx],        # [B, 1, S_prior, d]
            V_cache=V_caches[layer_idx],        # [B, 1, S_prior, d]
            cos=cos,                             # HBM fallback — unused when cos_f32_sb is set
            sin=sin,                             # HBM fallback — unused when sin_f32_sb is set
            position_ids=position_ids,
            sbm=sbm,
            # Plan A: pre-loaded layer-invariant constants (hoisted outside the loop)
            qnw_f32_sb  = qnw_f32_all[0:PMAX, layer_idx:layer_idx + 1],
            knw_f32_sb  = knw_f32_all[0:PMAX, layer_idx:layer_idx + 1],
            cos_f32_sb  = cos_f32_all,
            sin_f32_sb  = sin_f32_all,
            gpan_f32_sb = gpan_f32_all[0:PMAX, layer_idx * NH:(layer_idx + 1) * NH],
            # Plan B step 4: pre-loaded Wk / Wv from the current ring slot.
            wk_sb = Wk_cur,
            wv_sb = Wv_cur,
            # Plan B step 5: pre-loaded per-owned-head Wq / Wo from current slot.
            # Tuple of NOH=4 tensors; v14d aligns these with its internal owned_heads.
            wq_heads_sb = Wq_cur,
            wo_heads_sb = Wo_cur,
        )
        # out_sb: [H0, BxS*H1] = [128, 16] bf16 in SBUF — partial-sum (TP shard)

        # ------------------------------------------------------------------
        # AllReduce #1 — sum partial-sum output across TP ranks.
        # Bare nccl.all_reduce (no sendrecv/gather) since v14a produces the
        # full [128, 16] output on both LNC cores already.
        # ------------------------------------------------------------------
        attn_reduced_sb = nl.zeros((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                      name=f"L{layer_idx}_attn_reduced_sb")
        nccl.all_reduce(
            dsts=[attn_reduced_sb], srcs=[out_sb],
            op=nl.add, replica_group=rg,
        )

        sbm.close_scope()
        sbm.set_auto_alloc(True)

        # Residual add #1: bf16 add into bf16 residual
        nisa.tensor_tensor(dst=residual_sb, data1=residual_sb,
                           data2=attn_reduced_sb, op=nl.add)

        # ------------------------------------------------------------------
        # MOE (_qwen3_moe_sbuf_in_sbuf_out)
        # Post-attn RMSNorm (post_attention_layernorm) is fused inside.
        # Returns moe_out_sb as an H1_SHARD-wide shard on each LNC core.
        # ------------------------------------------------------------------
        sbm.set_name_prefix(f"L{layer_idx}_moe_")
        sbm.set_auto_alloc(True)

        moe_inp_bf16 = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                   name=f"L{layer_idx}_moe_inp_bf16")
        nisa.tensor_copy(moe_inp_bf16, residual_sb)

        moe_out_sb = _qwen3_moe_sbuf_in_sbuf_out_hoisted(
            inp_sb=moe_inp_bf16,
            dtype=dtype,
            T=T,
            gamma=gpost_list[layer_idx],        # HBM fallback — gpost not yet hoisted
            router_w=router_list[layer_idx],    # HBM fallback — unused when router_w_wide_sb is set
            gate_up_w=gate_up_list[layer_idx],  # [E, H, 384]
            down_w=down_list[layer_idx],        # [E, 192, H]
            sbm=sbm,
            # Plan B step 5: pre-loaded wide router_w from the current ring slot.
            router_w_wide_sb = Router_cur,
        )

        # AllReduce #2 + LNC gather: both TP shards → full [H0, BxS*H1] on each core
        moe_gathered_sb, _ = _sb2sb_all_reduce_gather(
            moe_out_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS
        )

        # Free all sbm allocs from the MoE block
        while sbm.heap:
            sbm.pop_heap()
        sbm.set_auto_alloc(True)

        # Residual add #2: bf16 add into bf16 residual
        nisa.tensor_tensor(dst=residual_sb, data1=residual_sb,
                           data2=moe_gathered_sb, op=nl.add)

        # Rotate to the other SBUF section for the next layer's weight alloc.
        # The compiler uses this to double-buffer: next iter's DMAs land in
        # the freed section and can issue while this iter's compute is still
        # reading the current section.
        sbm.increment_section()

    # Close the weight double-buffer + hoisted-constants scopes.
    sbm.close_scope()  # weight_db
    sbm.close_scope()  # hoist

    # -----------------------------------------------------------------------
    # After the last layer: single HBM store of the post-residual hidden state.
    # Final model.norm (RMSNorm) is applied by NxDI's get_model_output — we
    # deliberately do NOT apply it here.
    # -----------------------------------------------------------------------
    Y = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="Y")
    # Cast f32 residual to bf16 before storing to HBM (matches baseline bf16 output dtype).
    residual_out_bf16_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                       name="residual_out_bf16_sb")
    nisa.tensor_copy(residual_out_bf16_sb, residual_sb)
    # Plain-linear store: Y[b, 0, t*H0 + p] = residual_out_bf16_sb[p, b*H1 + t].
    # Gate on prg_id==0 to avoid duplicate writes across LNC cores.
    if prg_id == 0:
        Y_flat = Y.reshape((BxS, H))
        for b in nl.static_range(BxS):
            for t in nl.static_range(H1):
                col_psum = nl.ndarray((1, H0), dtype=dtype, buffer=nl.psum)
                nisa.nc_transpose(col_psum, residual_out_bf16_sb[0:H0, b*H1 + t : b*H1 + t + 1])
                col_sb = nl.ndarray((1, H0), dtype=dtype, buffer=nl.sbuf,
                                    name=f"out_col_b{b}_t{t}_sb")
                nisa.tensor_copy(col_sb, col_psum)
                nisa.dma_copy(
                    dst=Y_flat[b:b+1, t*H0:(t+1)*H0],
                    src=col_sb,
                    dge_mode=nisa.dge_mode.hwdge,
                )
    if n_prgs > 1:
        nisa.core_barrier(data=Y, cores=(0, 1))

    # Return KV caches as pass-through outputs so the NKI compiler preserves
    # v14a's in-place scatter DMAs (without this, writes are DCE'd as dead
    # stores to read-only inputs). NxDI's model_wrapper aliases each returned
    # KV tensor back to its kv_mgr.past_key_values[i] slot.
    return (Y,) + tuple(K_caches) + tuple(V_caches)


def _build_multilayer_kernel(num_layers: int):
    """Code-gen a kernel function with 48 explicit K and V tensor args.

    NKI's frontend classifies top-level args as either individual tensors or
    scalars; tuples/lists of tensors are scalar-classified and never get HBM
    bindings, which breaks both tracing and input-output aliasing for
    in-place KV scatter. So each KV cache must appear as its own top-level arg.

    Stacked weights (Wq_all etc.) don't need aliasing — a single tensor per
    weight type is fine.
    """
    def names(prefix):
        return [f"{prefix}_{i:02d}" for i in range(num_layers)]

    wq_names     = names("Wq")
    wk_names     = names("Wk")
    wv_names     = names("Wv")
    wo_names     = names("Wo")
    qn_names     = names("Qn")
    kn_names     = names("Kn")
    gpre_names   = names("Gpre")
    gpost_names  = names("Gpost")
    router_names = names("Router")
    gu_names     = names("GateUp")
    down_names   = names("Down")
    k_names      = names("K")
    v_names      = names("V")

    sig = ",\n    ".join(
        ["X"]
        + wq_names + wk_names + wv_names + wo_names
        + qn_names + kn_names + gpre_names + gpost_names
        + router_names + gu_names + down_names
        + k_names + v_names
        + ["cos", "sin", "position_ids"]
    )

    def tup(ns):
        return "(" + ", ".join(ns) + ",)"

    src = (
        f"def transformer_qwen3_moe_tkg_multilayer(\n"
        f"    {sig},\n"
        f"    replica_groups=None,\n"
        f"):\n"
        f"    Wq_list      = {tup(wq_names)}\n"
        f"    Wk_list      = {tup(wk_names)}\n"
        f"    Wv_list      = {tup(wv_names)}\n"
        f"    Wo_list      = {tup(wo_names)}\n"
        f"    qn_list      = {tup(qn_names)}\n"
        f"    kn_list      = {tup(kn_names)}\n"
        f"    gpre_list    = {tup(gpre_names)}\n"
        f"    gpost_list   = {tup(gpost_names)}\n"
        f"    router_list  = {tup(router_names)}\n"
        f"    gate_up_list = {tup(gu_names)}\n"
        f"    down_list    = {tup(down_names)}\n"
        f"    K_caches     = {tup(k_names)}\n"
        f"    V_caches     = {tup(v_names)}\n"
        f"    return _multilayer_body(\n"
        f"        X, Wq_list, Wk_list, Wv_list, Wo_list,\n"
        f"        qn_list, kn_list, gpre_list, gpost_list,\n"
        f"        router_list, gate_up_list, down_list,\n"
        f"        K_caches, V_caches,\n"
        f"        cos, sin, position_ids,\n"
        f"        num_layers={num_layers},\n"
        f"        replica_groups=replica_groups,\n"
        f"    )\n"
    )

    fname = f"<generated:multilayer_L{num_layers}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
    code = compile(src, fname, "exec")
    ns = {"_multilayer_body": _multilayer_body}
    exec(code, ns)
    return ns["transformer_qwen3_moe_tkg_multilayer"]


transformer_qwen3_moe_tkg_multilayer = _build_multilayer_kernel(NUM_LAYERS)
transformer_qwen3_moe_tkg_multilayer_jit = nki.jit(transformer_qwen3_moe_tkg_multilayer)

_kernel_cache: dict = {NUM_LAYERS: transformer_qwen3_moe_tkg_multilayer_jit}

def get_multilayer_kernel_jit(num_layers: int):
    if num_layers not in _kernel_cache:
        _kernel_cache[num_layers] = nki.jit(_build_multilayer_kernel(num_layers))
    return _kernel_cache[num_layers]
