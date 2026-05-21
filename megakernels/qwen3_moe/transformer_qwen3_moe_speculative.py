"""Qwen3-30B-A3B multi-layer fused TKG megakernel for speculative decoding.

Runs all `num_layers` decoder layers in a single NKI invocation using the
vendored nkilib subkernels (``attention_block_tkg``, ``rmsnorm_tkg``,
``router_topk``, ``moe_tkg``) shared with the gpt-oss megakernel. SBUF-resident
residual across layer boundaries; KV caches scattered in place at
``position_ids``.

Differences from ``transformer_qwen.py``
----------------------------------------
The existing Qwen3 megakernel calls hand-rolled ``attn_fused_nki`` /
``moe_fused_nki`` kernels that bake ``n_active_tokens=1`` into shape math.
Under fused speculation the target processes ``S_tkg = speculation_length``
positions per step, which trips an NKI shape-validation error in the MoE
gamma broadcast. The vendored ``nki_kernels`` subkernels (already used by
the gpt-oss megakernel) handle generic ``S_tkg`` cleanly, so this module
rebuilds the per-layer pipeline on top of them.

Per-layer pipeline (mirrors gpt-oss, no SWA, with pre-RoPE Q/K RMSNorm)
----------------------------------------------------------------------
1. ``attention_block_tkg`` with::

      rmsnorm_X_enabled=True                 (input_layernorm  = gpre_list[i])
      rmsnorm_QK_pre_rope_enabled=True       (q_norm / k_norm  = qn/kn_list[i])
      rmsnorm_QK_post_rope_enabled=False
      sink=None                              (no attention sink)
      bias_qkv=None / bias_out=None          (no biases in Qwen3)

   K/V are scattered in place into ``K_caches[i]`` / ``V_caches[i]`` at
   ``position_ids``.
2. SB2SB all-reduce + LNC gather on the attention output (LNC=2 needs a
   gather across cores; gpt-oss uses bare ``nccl.all_reduce`` because it
   runs LNC=1).
3. ``residual_sb += attn_gathered_sb``.
4. ``rmsnorm_tkg`` with ``gpost_list[i]`` (post_attention_layernorm).
5. ``router_topk`` (SOFTMAX, post-topk normalization).
6. ``moe_tkg`` (SiLU, POST_SCALE, no expert clamping, no expert bias).
7. SB2SB all-reduce + LNC gather on the MoE output.
8. ``residual_sb += moe_gathered_sb``.

After the last layer: single HBM store of the post-residual hidden state.
Final ``model.norm`` is applied by NxDI's ``get_model_output`` on the
Python side; this kernel does NOT apply it.

External IO contract
--------------------
The integration layer (``qwen_with_megakernel.py``) calls this kernel with
the SAME argument layout and return tuple as ``transformer_qwen.py``::

    kernel_out = get_multilayer_kernel_jit(L)[2](
        hidden_states,                         # [B, S_tkg, H]   bf16  HBM
        *Wq_list, *Wk_list, *Wv_list, *Wo_list,
        *qn_list, *kn_list, *gpre_list, *gpost_list,
        *router_list, *gate_up_list, *down_list,
        *K_caches, *V_caches,                  # mutated in-place
        cos_at_pos, sin_at_pos,                # [B, d] pre-indexed
        position_ids.to(torch.int32),          # [B, 1]
        replica_groups=...,
    )
    Y      = kernel_out[0]
    K_out  = kernel_out[1     : 1 + L]
    V_out  = kernel_out[1 + L : 1 + 2 * L]

Wq / Wk / Wv handling
---------------------
``attention_block_tkg`` takes a single fused ``W_qkv`` of shape
``[H, (q_heads + 2*kv_heads) * d_head]``. The integration provides separate
Wq/Wk/Wv tensors of shape ``[head_count * d, H]``. We chose option (a) from
the design plan: **fuse Wq/Wk/Wv into a single HBM scratch tensor at the
start of each layer**, then call ``attention_block_tkg`` on the scratch.

Rationale: option (b) (bypass ``attention_block_tkg`` and compose
``qkv`` / ``attention_tkg`` / ``output_projection_tkg`` directly) would
duplicate ~300 lines of orchestration code that the vendored kernel already
handles. Option (a) costs three HBM-to-HBM DMA copies per layer (one each
for Wq, Wk, Wv) with strided writes to handle the implicit transpose from
``[head_count*d, H]`` to ``[H, head_count*d]``. The scratch is allocated
once and reused across all layers in the call.

Approximate cost: Wq fusion ≈ 4 MB, Wk/Wv ≈ 0.5 MB each → ~5 MB / layer ×
48 layers ≈ 240 MB of HBM traffic per kernel invocation. At ~400 GB/s this
is roughly 600 µs, small relative to the rest of the multilayer kernel.

cos / sin handling
------------------
The integration provides ``cos_at_pos`` / ``sin_at_pos`` of shape ``[B, d]``
already indexed at the current position (same layout as
``transformer_qwen.py``). ``attention_block_tkg`` with
``rope_contiguous_layout=True`` expects ``[d/2, B, S_tkg]``. We take the
first half of d (HF's contiguous-halves RoPE convention duplicates the
second half) and permute the layout via a small SBUF round-trip into an
HBM scratch tensor that is reused for all layers.

Note: for ``S_tkg > 1`` (e.g. fused-spec verification), the integration
still passes a single ``[B, d]`` cos/sin tensor. This kernel broadcasts
that single position across all ``S_tkg`` slots. Semantically correct
multi-position cos/sin is the integration's responsibility to provide once
the speculative path is wired up; this kernel only reshuffles whatever
shape the integration ships.

S_tkg correctness scope
-----------------------
The primary correctness target is ``S_tkg = 1`` (single-token TKG decode),
which must match ``transformer_qwen.py`` bit-for-bit up to bf16 ULP drift.
The kernel ALSO compiles and runs for ``S_tkg > 1`` (fused-spec
verification shape), but numerical correctness at ``S_tkg > 1`` requires
the integration to supply per-position cos/sin instead of a single
``[B, d]`` tensor.

Hardware target
---------------
Trainium 3 (NeuronCore-v4), TP=4, LNC=2.

LNC=2 sequence-length constraint
--------------------------------
``attention_block_tkg`` shards the prior-context K/V across LNC cores
along the s_prior dimension when ``curr_sprior >= 2*p_max=256`` and batch
isn't sharded (see ``nkilib.core.attention.attention_tkg_utils.is_s_prior_sharded``).
At LNC=2 this asserts ``(curr_sprior // 2) % 128 == 0``, i.e. the K-cache
bucket length (== ``K_caches[i].shape[-2]``) MUST be a multiple of
``128 * LNC = 256``. NxDI's TKG bucket selection picks the smallest bucket
``>= position`` from the configured bucket list — buckets at 256, 512, 768,
1024, ... work; buckets at 640, 896, 1152, ... do NOT and trip
``kernel_assert(atp.s_prior % TC.p_max == 0)`` in
``attention_tkg.py:832`` with message::

    Sharded s_prior must be divisible by p_max. Got sharded s_prior=320, p_max=128.

There is no kwarg on ``attention_block_tkg`` to bypass this assertion.
**Run smoke tests at multiple-of-256 seq_lens (256, 512, 768, 1024)**; the
README's default ``seq_len=640`` reaches a bucket of 640 which is
incompatible with LNC=2 sharded s_prior. Workaround: either run at the
allowed bucket boundaries, or pad NxDI's TKG bucket list to multiples of
256 in the model config (e.g. ``buckets=[256, 512, 768, 1024, 2048]``).

MoE TP/LNC weight-layout constraint
-----------------------------------
NxDI auto-shards the per-expert MLP weights when TP > 1:
  - ``gate_up_proj`` is ``ExpertFusedColumnParallelLinear`` (stride=2)
    → output dim sharded by TP: weight shape ``[E, H, 2*I/TP]``.
  - ``down_proj`` is ``ExpertFusedRowParallelLinear``
    → input dim sharded by TP: weight shape ``[E, I/TP, H]``.

The vendored ``moe_tkg`` kernel treats whatever ``I``-axis it sees as the
*full* intermediate dim and does its OWN H-axis split across LNC cores
(``H_per_shard = H / LNC``). That means the kernel's view of ``I`` must
match the shape that NxDI has already produced — i.e. ``I_per_rank``, NOT
the unsharded ``moe_intermediate_size``.

For Qwen3-30B-A3B at TP=4: ``I_per_rank = 768 / 4 = 192``. The kernel sees
``dims.I = 192``, ``dims.num_total_128_tiles_per_I = ceil(192/128) = 2``
(with a 64-wide remainder tile). The per-HTile weight ring buffer + matmul
loops use ``TiledRange(I, I0)`` which handles the remainder cleanly.

However, the vendored ``mlp_tkg_down_projection.py`` adds a hoisted
single-DMA weight-load path (``use_hoisted_down_load``, set when
``not _MOE_LEGACY_WEIGHT_LOAD``) that reshapes the down-weight view via
``reshape_dim(dim=0, shape=(num_total_128_tiles_per_I, I0))``. That reshape
requires ``I == num_total_128_tiles_per_I * I0``, i.e. ``I`` divisible by
128. For Qwen3 ``I_per_rank=192`` this fails with::

    Size mismatch: 192 != 768
      at: nki_kernels/moe/mlp_tkg_down_projection.py:306 (weight.slice(...))

We disable the hoist by setting ``NKI_MOE_LEGACY_WEIGHT_LOAD=1`` at module
import (BEFORE importing ``nki_kernels.moe``). The legacy per-HTile DMA
path handles non-128-multiple ``I`` via ``TiledRange``. The same env var
also disables the cross-expert gate/up prefetch ring in
``selective_expert_impl.py`` (which would otherwise allocate
``(H0, H1_shard, I)`` SBUF tiles per expert — sized correctly at I=192,
but the prefetch logic doesn't gain anything once the down hoist is off).

If a future ``moe_intermediate_size`` gives ``I/TP`` divisible by 128
(e.g. via padding ``moe_intermediate_pad_size``), the env var can be
removed for performance.

Entry point
-----------
``get_multilayer_kernel_jit(NUM_LAYERS)[2]`` returns the LNC=2 jit callable
(``[1]`` for LNC=1, ``[2]`` for LNC=2, matching the existing kernel's index
convention).
"""

import os as _os

# MUST be set BEFORE the import of ``nki_kernels.moe`` below: the vendored
# down/gate-up sub-kernels read this env var at module-load time
# (``_MOE_LEGACY_WEIGHT_LOAD`` constant). See the "MoE TP/LNC weight-layout
# constraint" section of the module docstring for why this is required at
# TP=4 with I_per_rank=192 (not a multiple of 128).
_os.environ["NKI_MOE_LEGACY_WEIGHT_LOAD"] = "1"

import linecache

import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl

from nki_kernels.attention import attention_block_tkg
from nki_kernels.moe import (
    XHBMLayout_T_H__1,
    XSBLayout_tp102__0,
    XSBLayout_tp2013__1,
    moe_tkg,
    rmsnorm_tkg,
    router_topk,
)
from nkilib.core.utils.allocator import BufferManager, Logger
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    QuantizationType,
    RouterActFnType,
)
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.experimental.transformer.transformer_tkg import _sb2sb_all_reduce_gather


# ---------------------------------------------------------------------------
# Model constants (Qwen3-30B-A3B at TP=4, LNC=2)
# ---------------------------------------------------------------------------

H            = 2048
H0           = 128
H1           = H // H0          # 16
N_PRGS       = 2
H1_SHARD     = H1 // N_PRGS     # 8
# H2 = per-partition slice of H1 owned by each LNC shard. attention_block_tkg
# / rmsnorm_tkg view H as (num_shards, H0, H2) row-major and slice dim 2 of
# the SBUF input on shard_id * H1_SHARD; we MUST load X in that layout, not
# the simpler channel-interleaved (H0, H1) view that works only at LNC=1.
H2           = H1_SHARD          # H1 // num_shards = 16 // 2 = 8 at LNC=2
EPS          = 1e-6             # Qwen3 RMSNorm eps
PMAX         = H0
NUM_LAYERS   = 48

# Attention dims (post TP-shard)
D_HEAD          = 128
NUM_Q_HEADS_TP  = 8             # 32 q heads / TP=4
NUM_KV_HEADS_TP = 1             # 4 kv heads / TP=4
I_QKV           = (NUM_Q_HEADS_TP + 2 * NUM_KV_HEADS_TP) * D_HEAD   # 1280

# MoE dims
E              = 128            # num experts
TOP_K          = 8
TP_DEGREE      = 4              # tensor-parallel degree (must match NxDI's tp_degree)
# Intermediate dim *per TP rank* — NxDI shards moe_intermediate_size=768 by
# TP=4 on both gate_up (ColumnParallel stride=2) and down (RowParallel). The
# kernel sees the sharded shapes:
#   gate_up_list[i].shape = [E, H, 2 * I_PER_EXPERT] = [128, 2048, 384]
#   down_list[i].shape    = [E, I_PER_EXPERT, H]     = [128, 192,  2048]
# and treats I_PER_EXPERT as the "full" intermediate dim. See the docstring's
# "MoE TP/LNC weight-layout constraint" section.
I_PER_EXPERT   = 768 // TP_DEGREE   # 192

SBM_SIZE_BYTES = 200 * 1024


# ---------------------------------------------------------------------------
# Per-layer Wq/Wk/Wv → W_qkv fusion (HBM scratch)
# ---------------------------------------------------------------------------

def _fuse_qkv_weights(Wq, Wk, Wv, W_qkv_scratch):
    """Fuse separate Wq/Wk/Wv (each [head_count*d, H]) into W_qkv_scratch
    of shape [H, (q_heads + 2*kv_heads) * d_head].

    Each Wx is laid out in HBM as ``Wx[head_count*d_head, H]``. The fused
    W_qkv has layout ``W_qkv[H, head_count*d_head]`` so the matmul
    ``hidden @ W_qkv`` gives the projected tokens. The transpose is
    performed by reading each row of the source contiguously and scattering
    into a column of the destination (strided write into the fused layout).

    For each Wq element at (i, h) where i in [0, Hq_d), h in [0, H):
        src flat offset = i*H + h    (row-major over [Hq_d, H])
        dst flat offset = h*I + i    (row-major over [H, I]) — Q slot
    The iteration is outer-i (stride H, count Hq_d), inner-h (stride 1,
    count H) so each i picks up H contiguous source bytes and writes them
    column-strided into the fused destination.
    """
    Hq_d  = Wq.shape[0]   # NUM_Q_HEADS_TP  * d_head = 1024
    Hkv_d = Wk.shape[0]   # NUM_KV_HEADS_TP * d_head = 128

    I_total = Hq_d + 2 * Hkv_d
    # Flat view of the scratch: [H * I] row-major where row=H, col=I.
    W_qkv_flat = W_qkv_scratch.reshape((H * I_total,))

    # Wq → W_qkv[:, 0:Hq_d]
    Wq_flat = Wq.reshape((Hq_d * H,))
    nisa.dma_copy(
        dst=W_qkv_flat.ap(
            pattern=[[1, Hq_d],         # outer i: stride 1, count Hq_d
                     [I_total, H]],     # inner h: stride I_total, count H
            offset=0,
        ),
        src=Wq_flat.ap(
            pattern=[[H, Hq_d],         # outer i: stride H, count Hq_d
                     [1, H]],           # inner h: stride 1, count H (contiguous)
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )

    # Wk → W_qkv[:, Hq_d:Hq_d+Hkv_d]
    Wk_flat = Wk.reshape((Hkv_d * H,))
    nisa.dma_copy(
        dst=W_qkv_flat.ap(
            pattern=[[1, Hkv_d],
                     [I_total, H]],
            offset=Hq_d,                # column offset within each row of the fused layout
        ),
        src=Wk_flat.ap(
            pattern=[[H, Hkv_d],
                     [1, H]],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )

    # Wv → W_qkv[:, Hq_d+Hkv_d : Hq_d+2*Hkv_d]
    Wv_flat = Wv.reshape((Hkv_d * H,))
    nisa.dma_copy(
        dst=W_qkv_flat.ap(
            pattern=[[1, Hkv_d],
                     [I_total, H]],
            offset=Hq_d + Hkv_d,
        ),
        src=Wv_flat.ap(
            pattern=[[H, Hkv_d],
                     [1, H]],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )


# ---------------------------------------------------------------------------
# Full-attention mask builder (no SWA — every Qwen3 layer is full attention)
# ---------------------------------------------------------------------------

def _stream_shuffle_broadcast(src, dst):
    """Replicate src (1, F) across all partitions of dst (P, F)."""
    dst_npar = dst.shape[0]
    shuffle_mask = [0] * 32
    for i in range((dst_npar + 31) // 32):
        cur_npar = min(32, dst_npar - i * 32)
        nisa.nc_stream_shuffle(
            src=src[0:1, :],
            dst=dst[i * 32 : i * 32 + cur_npar, 0 : dst.shape[1]],
            shuffle_mask=shuffle_mask,
        )


def _build_full_mask_hbm(position_ids, S_ctx, B, num_heads, S_tkg):
    """Build full-attention mask [S_ctx, B, num_heads, S_tkg] from position_ids.

    Rule per (t, b): ``mask = 1 if t < pos[b] OR t == S_ctx - 1 else 0``.

    The mask is consumed by ``attention_block_tkg`` -> ``attention_tkg``,
    where it covers the s_prior (cached K) axis. The kernel ALSO places
    the K_active row (the freshly computed K for the new token) at the
    LAST slot of each LNC shard's local k_sb buffer
    (``k_sb[:, fa_tile_s_prior - 1]`` — see
    nki_kernels/attention/attention_tkg.py:1891). Globally, the last shard
    owns prior-axis positions ``[S_ctx - s_prior_per_shard, S_ctx)`` and
    its last column lands at global ``S_ctx - 1``. The mask MUST therefore
    set position ``S_ctx - 1`` to 1, otherwise the kernel masks out
    K_active itself and never attends to the new token's K/V — making
    attention output identical to ``softmax(Q · K_prior[0..pos-1]) @ V_prior[..]``
    (i.e. the new token contributes nothing to itself).

    This OR-with-(S_ctx-1) is NOT related to attention sink. Sink handling
    is a separate ``sink=`` parameter to ``attention_block_tkg`` and is
    applied via ``_apply_sink_to_max`` post-softmax. The same OR term is
    used by gpt-oss for the same structural reason (enable K_active
    matmul), even though gpt-oss happens to store its sink token at the
    same slot.

    K_cache[S_ctx - 1] being uninitialized is irrelevant: the kernel
    overwrites that local k_sb slot with K_active before the QK matmul.

    History: an earlier "fix" to this file removed the OR-with-(S_ctx-1)
    term under the incorrect assumption that it was a sink-related hack.
    That fix introduced the bug where the new token's K_active was
    silently masked out, producing systematically wrong attention
    output. Guard: tests/qwen3_moe/test_attention_vs_hf.py.
    """
    P_MAX = 128
    # S_ctx must be a multiple of 128. NxDI's TKG buckets satisfy this.
    n_tile = S_ctx // P_MAX

    out = nl.ndarray((S_ctx, B, num_heads, S_tkg), dtype=nl.bfloat16,
                     buffer=nl.shared_hbm, name="mask_full_hbm")

    # ----------------------- pos: load + broadcast ------------------------
    pos_one = nl.ndarray((1, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos_one, src=position_ids.reshape((1, B)))

    pos_bcast = nl.ndarray((P_MAX, B), dtype=nl.float32, buffer=nl.sbuf)
    _stream_shuffle_broadcast(pos_one, pos_bcast)

    # ----------------------- iota: t = 0..S_ctx-1 -------------------------
    iota_tile = nl.ndarray((P_MAX, n_tile, num_heads), dtype=nl.float32,
                           buffer=nl.sbuf)
    nisa.iota(
        iota_tile,
        pattern=[[P_MAX, n_tile], [0, num_heads]],
        offset=0,
        channel_multiplier=1,
    )

    # ----------------------- mask: (iota < pos) OR (iota == S_ctx - 1) ----
    # The OR-with-(S_ctx-1) term is REQUIRED by attention_block_tkg's
    # internal layout: the kernel places K_active (the freshly computed K
    # for the new token) at slot ``S_ctx - 1`` of the local k_sb buffer
    # (see nki_kernels/attention/attention_tkg.py:1891-1898 — last column
    # of each LNC shard's k_sb is overwritten with K_active). If mask
    # position S_ctx-1 is 0, K_active is masked to -inf and never attends,
    # leaving attention as `softmax(Q · K_prior[0..pos-1]) @ V_prior[0..pos-1]`
    # — the new token's own K/V are completely ignored.
    #
    # Note: this is NOT the gpt-oss attention-sink hack. The gpt-oss kernel
    # uses the SAME OR-with-(S_ctx-1) term for the SAME structural reason
    # (enable K_active matmul). The earlier comment in this file that
    # claimed this term was sink-related was incorrect — sink handling is
    # a separate `sink=` parameter in attention_block_tkg, not a mask slot.
    # K_cache[S_ctx-1] being uninitialized is irrelevant: the kernel
    # overwrites that slot with K_active before the matmul.
    #
    # Guard: tests/qwen3_moe/test_attention_vs_hf.py and the
    # test_full_mask_hits_s_ctx_minus_1 unit test in test_speculative_megakernel.py.
    mask_lt = nl.ndarray((P_MAX, n_tile, num_heads), dtype=nl.float32,
                         buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=mask_lt,
        data=iota_tile,
        op0=nl.less,
        operand0=pos_bcast[:, 0:1],
    )

    mask_eq = nl.ndarray((P_MAX, n_tile, num_heads), dtype=nl.float32,
                         buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=mask_eq,
        data=iota_tile,
        op0=nl.equal,
        operand0=S_ctx - 1,
    )

    mask_bf16 = nl.ndarray((P_MAX, n_tile, num_heads), dtype=nl.bfloat16,
                           buffer=nl.sbuf)
    nisa.tensor_tensor(mask_bf16, mask_lt, mask_eq, op=nl.maximum)

    # SBUF [P_MAX, n_tile, num_heads] -> HBM [S_ctx, B, num_heads, S_tkg].
    # HBM flat offset(p, k, h) = (k*P_MAX + p) * B * num_heads * S_tkg + h * S_tkg.
    out_flat = out.reshape((S_ctx * B * num_heads * S_tkg,))
    nisa.dma_copy(
        dst=out_flat.ap(
            pattern=[[num_heads * S_tkg, P_MAX],
                     [P_MAX * num_heads * S_tkg, n_tile],
                     [S_tkg, num_heads]],
            offset=0,
        ),
        src=mask_bf16,
    )

    return out


# ---------------------------------------------------------------------------
# Build permuted RoPE cos/sin from pre-indexed [B, d] tensors
# ---------------------------------------------------------------------------

def _build_permuted_rope_hbm_from_pos(rope_at_pos, B, S_tkg, d_head, name):
    """Take ``rope_at_pos`` of shape ``[B, d_head]`` (pre-indexed at the
    current position, with HF contiguous-halves convention where the second
    half duplicates the first) and produce an HBM tensor of shape
    ``[d_head//2, B, S_tkg]`` ready for ``attention_block_tkg`` with
    ``rope_contiguous_layout=True``.

    For ``S_tkg > 1``, the same single-position cos/sin is broadcast across
    all ``S_tkg`` slots. The integration is responsible for supplying
    genuinely per-position cos/sin if multi-position semantics are
    required; this kernel only reshuffles whatever shape it receives.
    """
    half_d = d_head // 2

    out = nl.ndarray((half_d, B, S_tkg), dtype=rope_at_pos.dtype,
                     buffer=nl.shared_hbm, name=name)

    # rope_at_pos flat layout: index = b * d_head + i for i in [0, d_head).
    # Build SBUF tile sb[p, b * S_tkg + t] = rope_at_pos[b, p] for p in
    # [0, half_d), broadcast over t (stride 0 along the t axis).
    rope_flat = rope_at_pos.reshape((B * d_head,))

    sb = nl.ndarray((half_d, B * S_tkg), dtype=rope_at_pos.dtype, buffer=nl.sbuf,
                    name=f"{name}_sb")
    nisa.dma_copy(
        dst=sb,
        src=rope_flat.ap(
            pattern=[[1, half_d],     # partition: half_d rows, stride 1 (within batch's first half)
                     [d_head, B],     # free outer: B batches, stride d_head
                     [0, S_tkg]],     # free inner: S_tkg broadcast (stride 0)
            offset=0,
        ),
    )

    # SBUF [half_d, B*S_tkg] -> HBM [half_d, B, S_tkg] is a contiguous store.
    nisa.dma_copy(dst=out.reshape((half_d, B * S_tkg)), src=sb)

    return out


# ---------------------------------------------------------------------------
# Multilayer body
# ---------------------------------------------------------------------------

def _store_shard_interleaved_sb_to_hbm(dst_hbm, src_sb, B, S_tkg, prg_id):
    """Inverse of the X load: write SBUF [H0, BxS*H1] (shard-interleaved
    layout) to HBM [B, S_tkg, H] canonical layout. Gated on prg_id==0.

    Layout (shard-interleaved):
        src_sb[p, bs*H1 + shard*H2 + h2]
            == dst_hbm.flat[bs*H + shard*(H0*H2) + p*H2 + h2]
    """
    BxS = B * S_tkg
    dst_flat = dst_hbm.reshape((BxS * H,))
    if prg_id == 0:
        nisa.dma_copy(
            dst=dst_flat.ap(
                pattern=[
                    [H2, H0],
                    [H0 * H2, N_PRGS],
                    [1, H2],
                    [H, BxS],
                ],
                offset=0,
            ),
            src=src_sb,
            dge_mode=nisa.dge_mode.hwdge,
        )


def _multilayer_body(
    X,             # [B, S_tkg, H]                    bf16  HBM
    Wq_list,       # tuple of L: [Hq_tp*d, H]         bf16  HBM
    Wk_list,       # tuple of L: [Hkv_tp*d, H]        bf16  HBM
    Wv_list,       # tuple of L: [Hkv_tp*d, H]        bf16  HBM
    Wo_list,       # tuple of L: [Hq_tp*d, H]         bf16  HBM
    qn_list,       # tuple of L: [d]                  bf16  HBM (Q pre-RoPE RMSNorm)
    kn_list,       # tuple of L: [d]                  bf16  HBM (K pre-RoPE RMSNorm)
    gpre_list,     # tuple of L: [H]                  bf16  HBM (input_layernorm)
    gpost_list,    # tuple of L: [1, H]               bf16  HBM (post_attention_layernorm)
    router_list,   # tuple of L: [H, E]               bf16  HBM
    gate_up_list,  # tuple of L: [E, H, 2*I_PER_EXPERT]   bf16  HBM (TP-sharded)
    down_list,     # tuple of L: [E, I_PER_EXPERT, H]     bf16  HBM (TP-sharded)
    K_caches,      # tuple of L: [B, 1, S_max, d]     bf16  HBM (mutated in place)
    V_caches,      # tuple of L: [B, 1, S_max, d]     bf16  HBM (mutated in place)
    cos,           # [B, d]                           bf16  HBM (pre-indexed at position)
    sin,           # [B, d]                           bf16  HBM (pre-indexed at position)
    position_ids,  # [B, 1]                           int32 HBM
    num_layers,
    replica_groups=None,
):
    """Kernel body — runs ``num_layers`` fused decoder layers.

    Mirrors ``megakernels/gpt_oss/transformer_gpt_oss.py:_multilayer_body``
    with Qwen3-specific adaptations:
      * separate Wq/Wk/Wv → HBM-fused W_qkv per layer (no Wo fusion: Wo
        layout matches attention_block_tkg's W_out expectation directly);
      * pre-RoPE Q/K RMSNorm enabled (gammas qn/kn);
      * no attention sink, no SWA, no biases anywhere, no expert clamping;
      * softmax router with norm_topk_prob=True, POST_SCALE expert affinities;
      * LNC=2 SB2SB AR-gather for both attention and MoE outputs (gpt-oss
        uses bare nccl.all_reduce because it runs LNC=1).
    """
    B, S_tkg, _ = X.shape
    dtype = X.dtype
    BxS = B * S_tkg
    T = BxS

    # Capture post-update K/V refs so NKI preserves the in-place scatter
    # (mirrors gpt-oss). Without this, NCC may DCE the scatter as a dead
    # store to a read-only input.
    K_post = list(K_caches)
    V_post = list(V_caches)

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "transformer_qwen3_moe_speculative", (0, 1), N_PRGS
    )

    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None

    sbm = BufferManager(0, SBM_SIZE_BYTES, Logger("transformer_qwen3_moe_speculative"))
    sbm.set_auto_alloc(True)

    # K_cache shape: [B, 1, S_max, d]. attention_block_tkg derives S_max_ctx
    # from K_cache.shape[-2].
    S_FULL = K_caches[0].shape[-2]

    # ---- Build per-call HBM scratches (mask, cos/sin perm, fused W_qkv) ----
    # The mask and cos/sin scratches are layer-invariant and reused unchanged
    # across all layers. The W_qkv scratch is overwritten per layer.
    mask_full_hbm = _build_full_mask_hbm(
        position_ids, S_FULL, B, NUM_Q_HEADS_TP, S_tkg
    )
    cos_perm_hbm = _build_permuted_rope_hbm_from_pos(
        cos, B, S_tkg, D_HEAD, name="cos_perm_hbm"
    )
    sin_perm_hbm = _build_permuted_rope_hbm_from_pos(
        sin, B, S_tkg, D_HEAD, name="sin_perm_hbm"
    )

    # Fused W_qkv scratch: [H, I_QKV] = [2048, 1280]. Overwritten per layer.
    W_qkv_scratch = nl.ndarray((H, I_QKV), dtype=dtype, buffer=nl.shared_hbm,
                                name="W_qkv_scratch_hbm")

    # ---- DEBUG: layer-0 dump HBM tensors (canonical [B, S, H] layout) ----
    # These are dumped from layer 0 ONLY. Used to numerically compare against
    # the OLD kernel (transformer_qwen.py) for bug localization. See
    # `transformer_qwen3_moe_speculative.py` instrumentation notes.
    attn_in_dump_hbm = nl.ndarray((B, S_tkg, H), dtype=dtype,
                                   buffer=nl.shared_hbm, name="attn_in_dump")
    attn_out_dump_hbm = nl.ndarray((B, S_tkg, H), dtype=dtype,
                                    buffer=nl.shared_hbm, name="attn_out_dump")
    moe_out_dump_hbm = nl.ndarray((B, S_tkg, H), dtype=dtype,
                                   buffer=nl.shared_hbm, name="moe_out_dump")

    # ---- Load X into SBUF residual in LNC-aware shard-interleaved layout ----
    # Layout: residual_sb[p, b*H1 + (shard*H2 + h2)] = X[b, s, shard*(H0*H2) + p*H2 + h2].
    #
    # Why this 4-level pattern instead of the simpler channel-interleaved
    # [[H1, H0], [1, H1], [H, BxS]]? attention_block_tkg's internal
    # _input_load (norm_tkg_utils.load_input_to_sbuf, hidden_dim_tp=False)
    # views HBM X as (BxS, num_shards, H0, H2) row-major where
    # H2 = H1 // num_shards. The SBUF input is sliced on dim 2 at
    # shard_id * H1_SHARD; for that slice to actually correspond to a single
    # shard's H values, h1 (SBUF dim 2 index) must equal shard*H2 + h2.
    #
    # The old 3-level pattern only happens to be equivalent when
    # num_shards = 1 (gpt-oss runs LNC=1, so H1 = H2 and the [num_shards]
    # level collapses). At LNC=2 the old pattern interleaves shards along
    # the SBUF partition axis and the subkernel reads garbage. This was the
    # root cause of garbage TKG logits in the speculative megakernel.
    residual_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                              name="residual_sb")
    X_flat = X.reshape((BxS * H,))
    nisa.dma_copy(
        dst=residual_sb,
        src=X_flat.ap(
            pattern=[
                [H2, H0],            # partition p   : stride=H2,    count=H0=128
                [H0 * H2, N_PRGS],   # shard outer   : stride=H0*H2, count=num_shards
                [1, H2],             # h2 inner      : stride=1,     count=H2
                [H, BxS],            # batch         : stride=H,     count=BxS
            ],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )

    # ---- DEBUG: dump attn_in (residual right after X load) ----
    _store_shard_interleaved_sb_to_hbm(attn_in_dump_hbm, residual_sb, B, S_tkg, prg_id)

    # ---- Per-layer loop -------------------------------------------------
    for layer_idx in range(num_layers):
        # ===== Fuse Wq/Wk/Wv into W_qkv_scratch for this layer ============
        sbm.set_name_prefix(f"L{layer_idx}_qkvfuse_")
        sbm.set_auto_alloc(True)
        _fuse_qkv_weights(
            Wq_list[layer_idx], Wk_list[layer_idx], Wv_list[layer_idx],
            W_qkv_scratch,
        )

        # ===== Attention ===================================================
        sbm.set_name_prefix(f"L{layer_idx}_attn_")
        sbm.set_auto_alloc(True)

        # Copy residual — attention_block_tkg's fused RMSNorm reads X and
        # writes the normalized output, so we don't want to overwrite the
        # residual buffer. (gpt-oss does the same.)
        attn_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf,
                                 name=f"L{layer_idx}_attn_in_sb")
        nisa.tensor_copy(attn_in_sb, residual_sb)
        X_sb = attn_in_sb.reshape((H0, BxS, H1))

        # Reshape per-layer gamma/qn/kn to the shapes the vendored kernels
        # expect: [H] -> [1, H], [d] -> [1, d]. These are NkiTensor view
        # reshapes — no runtime data movement.
        gpre_w  = gpre_list[layer_idx].reshape((1, H))
        qnorm_w = qn_list[layer_idx].reshape((1, D_HEAD))
        knorm_w = kn_list[layer_idx].reshape((1, D_HEAD))

        attn_result = attention_block_tkg(
            X=X_sb,
            X_hidden_dim_actual=H,
            # ---- input RMSNorm (input_layernorm) ----
            rmsnorm_X_enabled=True,
            rmsnorm_X_eps=EPS,
            rmsnorm_X_gamma=gpre_w,
            # ---- QKV projection ----
            W_qkv=W_qkv_scratch,
            bias_qkv=None,                       # Qwen3 has no QKV bias
            quantization_type_qkv=QuantizationType.NONE,
            weight_dequant_scale_qkv=None,
            input_dequant_scale_qkv=None,
            # ---- Pre-RoPE Q/K RMSNorm (Qwen3-specific: q_norm / k_norm) ----
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=EPS,
            rmsnorm_QK_pre_rope_W_Q=qnorm_w,
            rmsnorm_QK_pre_rope_W_K=knorm_w,
            # ---- RoPE (contiguous halves) ----
            cos=cos_perm_hbm,
            sin=sin_perm_hbm,
            rope_contiguous_layout=True,
            # ---- Post-RoPE QK RMSNorm: not used by Qwen3 ----
            rmsnorm_QK_post_rope_enabled=False,
            rmsnorm_QK_post_rope_eps=EPS,
            rmsnorm_QK_post_rope_W_Q=None,
            rmsnorm_QK_post_rope_W_K=None,
            # ---- attention ----
            K_cache_transposed=False,
            active_blocks_table=None,
            K_cache=K_post[layer_idx],
            V_cache=V_post[layer_idx],
            attention_mask=mask_full_hbm,
            sink=None,                           # Qwen3 has no attention sink
            # ---- KV cache update (in place at position_ids) ----
            update_cache=True,
            kv_cache_update_idx=position_ids,
            # ---- output projection ----
            W_out=Wo_list[layer_idx],
            bias_out=None,                       # Qwen3 has no O bias
            quantization_type_out=QuantizationType.NONE,
            weight_dequant_scale_out=None,
            input_dequant_scale_out=None,
            transposed_out=True,
            out_in_sb=True,
            sbm=sbm,
        )
        attn_kernel_out_sb = attn_result[0]
        K_post[layer_idx] = attn_result[1]
        V_post[layer_idx] = attn_result[2]

        # attn_kernel_out_sb shape with transposed_out=True, out_in_sb=True:
        # native shape is [h_1=H0, h_2 * bxs] = [H0, H1_SHARD * BxS] — F-dim
        # outer is H1_SHARD, inner is BxS, which is exactly the layout
        # _sb2sb_all_reduce_gather expects for `sharded_sb`.
        attn_sharded_sb = attn_kernel_out_sb.reshape((H0, H1_SHARD * BxS))

        attn_gathered_sb, _ = _sb2sb_all_reduce_gather(
            attn_sharded_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS
        )

        # ---- DEBUG: dump attn_gathered for layer 0 only ----
        if layer_idx == 0:
            _store_shard_interleaved_sb_to_hbm(
                attn_out_dump_hbm, attn_gathered_sb, B, S_tkg, prg_id
            )

        # Free attention sub-function's stack/heap allocations before moving
        # to MoE.
        while sbm.heap:
            sbm.pop_heap()
        sbm.set_auto_alloc(True)

        # Residual add #1
        nisa.tensor_tensor(dst=residual_sb, data1=residual_sb,
                           data2=attn_gathered_sb, op=nl.add)

        # ===== MoE block ===================================================
        sbm.set_name_prefix(f"L{layer_idx}_moe_")
        sbm.set_auto_alloc(True)

        # Post-attention RMSNorm (post_attention_layernorm).
        moe_in_sb = nl.ndarray((H0, BxS, H1), dtype=dtype, buffer=nl.sbuf,
                                name=f"L{layer_idx}_moe_in_sb")
        rmsnorm_tkg(
            input=residual_sb.reshape((H0, BxS, H1)),
            gamma=gpost_list[layer_idx],
            output=moe_in_sb,
            eps=EPS,
            hidden_actual=H,
            sbm=sbm,
        )

        # ===== Router topK (softmax + L1-normalize) ========================
        # NCC_IGCA090: every mutable_tensor needs at least one store, so
        # router_logits gets a real HBM allocation even though it's discarded.
        router_logits_hbm = nl.ndarray((T, E), dtype=nl.float32,
                                       buffer=nl.shared_hbm,
                                       name=f"L{layer_idx}_router_logits_scratch")
        expert_index_sb = nl.ndarray((T, TOP_K), dtype=nl.uint32,
                                     buffer=nl.sbuf,
                                     name=f"L{layer_idx}_expert_index_sb")
        expert_affinities_sb = nl.ndarray((T, E), dtype=nl.float32,
                                          buffer=nl.sbuf,
                                          name=f"L{layer_idx}_expert_affinities_sb")
        # Qwen3 routing: softmax(topK(logits)) with L1 norm over the top-K
        # probabilities (norm_topk_prob=True).
        #
        # x_sb_layout=XSBLayout_tp2013__1 is the LNC=2 shard-interleaved
        # layout (P-stride = H/256 = 8) — this MATCHES the layout that
        # rmsnorm_tkg / attention_block_tkg produce at LNC=2. Using
        # XSBLayout_tp102__0 (P-stride = H/128 = 16) would silently
        # mis-index the hidden state and produce garbage logits, since
        # the two layouts only coincide at LNC=1 (where gpt-oss runs).
        # See router_topk.py:1389 for the tp2013 diagram and
        # nkilib/core/mlp/mlp_tkg_utils.py:input_norm_load for proof
        # that the shard-interleaved layout is what rmsnorm_tkg produces.
        router_outputs = router_topk(
            x=moe_in_sb,
            w=router_list[layer_idx],
            w_bias=None,
            router_logits=router_logits_hbm,
            expert_affinities=expert_affinities_sb,
            expert_index=expert_index_sb,
            act_fn=RouterActFnType.SOFTMAX,
            k=TOP_K,
            x_hbm_layout=XHBMLayout_T_H__1,
            x_sb_layout=XSBLayout_tp2013__1,
            router_pre_norm=False,                # ACT2 path: topK then softmax
            norm_topk_prob=True,                  # Qwen3 normalizes top-K probs
            skip_store_router_logits=False,
            name_prefix=f"L{layer_idx}_moe_",
        )
        # router_topk's (topK, ACT2, Scatter) path rebinds expert_affinities;
        # capture the returned tensor (mirrors gpt-oss).
        expert_affinities_sb = router_outputs[2]

        # ===== Selective expert MoE ========================================
        # moe_tkg expects gate_up_w of shape [E, H, 2, I_per_rank]. NxDI's
        # ExpertFusedColumnParallelLinear(stride=2) has already TP-sharded
        # the last dim, so gate_up_list[i].shape = [E, H, 2*I_PER_EXPERT].
        # The reshape is a NkiTensor view — no data movement. Importantly,
        # I_PER_EXPERT MUST be the post-TP-shard value (192 at TP=4) so the
        # element count actually matches the storage; the kernel does its
        # own H-axis split across LNC cores and treats this I as the "full"
        # intermediate dim. See the docstring's "MoE TP/LNC weight-layout
        # constraint" section.
        gate_up_w = gate_up_list[layer_idx].reshape((E, H, 2, I_PER_EXPERT))

        # CRITICAL (LNC=2 per-shard slice): moe_tkg's selective-expert (T=1)
        # path expects ``hidden_input`` to be the PER-SHARD slice of shape
        # ``[H0, T, H1_SHARD]`` — NOT the full ``[H0, T, H1]`` post-RMSNorm
        # output. Inside ``gate_up_projection`` the matmul column index runs
        # over [0, H1_shard) on BOTH cores. If we passed the full ``moe_in_sb``
        # both cores would read the SAME H value subset (columns
        # [0..H1_SHARD)), pairing core 1's weight rows [H/2..H) with core 0's
        # H values [0..H/2). That produces garbage MoE output. Mirrors the
        # canonical pattern in nkilib/core/moe_block/moe_block_tkg.py:309-314.
        #
        # See tests/qwen3_moe/test_moe_vs_hf.py for the strict-tolerance
        # regression test and tests/qwen3_moe/test_speculative_megakernel.py
        # for the call-site shape guard.
        moe_in_pershard_sb = nl.ndarray((H0, BxS, H1_SHARD), dtype=dtype,
                                         buffer=nl.sbuf,
                                         name=f"L{layer_idx}_moe_in_pershard_sb")
        nisa.tensor_copy(
            dst=moe_in_pershard_sb,
            src=moe_in_sb[:, :, nl.ds(prg_id * H1_SHARD, H1_SHARD)],
        )

        moe_out_sb = moe_tkg(
            hidden_input=moe_in_pershard_sb,
            expert_gate_up_weights=gate_up_w,
            expert_down_weights=down_list[layer_idx],     # [E, I_PER_EXPERT, H]
            expert_affinities=expert_affinities_sb,
            expert_index=expert_index_sb,
            is_all_expert=False,
            expert_gate_up_bias=None,                     # Qwen3 has no expert biases
            expert_down_bias=None,
            activation_fn=ActFnType.SiLU,                 # Qwen3 uses SiLU
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            name_prefix=f"L{layer_idx}_moe_",
            gate_clamp_upper_limit=None,                  # Qwen3 has no expert clamping
            gate_clamp_lower_limit=None,
            up_clamp_upper_limit=None,
            up_clamp_lower_limit=None,
            output_in_sbuf=True,
            output_dtype=dtype,
        )
        # moe_out_sb shape: now matches the per-shard input — [H0, BxS, H1_SHARD].
        # Under LNC=2 at TKG (T==1, shard_on_T disabled), moe_tkg shards on H
        # and writes the local shard's H values into output[:, :, 0:H1_SHARD]
        # (F layout: BxS outer, H1_SHARD inner). _sb2sb_all_reduce_gather
        # expects sharded_sb laid out as [H0, H1_SHARD outer, BxS inner], in
        # a fresh contiguous SBUF buffer. We copy the valid slice into a
        # fresh contiguous tile via H1-major column copies; this avoids both
        # the "illegal partition step" (from a strided slice fed to a
        # collective) and the "non-unit stride in the free dim" assertion
        # (from feeding a rearrange view through ``.ap()``).
        moe_sharded_sb = nl.ndarray((H0, H1_SHARD * BxS), dtype=dtype,
                                     buffer=nl.sbuf,
                                     name=f"L{layer_idx}_moe_sharded_sb")
        for h1s in range(H1_SHARD):
            # moe_out_sb[:, :, h1s] view: [H0, BxS] — partition-contiguous,
            # free dim contiguous after the slice on the last axis.
            nisa.tensor_copy(
                dst=moe_sharded_sb[:, h1s * BxS : (h1s + 1) * BxS],
                src=moe_out_sb[:, :, h1s],
            )

        moe_gathered_sb, _ = _sb2sb_all_reduce_gather(
            moe_sharded_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_SHARD, BxS
        )

        # ---- DEBUG: dump moe_gathered for layer 0 only ----
        if layer_idx == 0:
            _store_shard_interleaved_sb_to_hbm(
                moe_out_dump_hbm, moe_gathered_sb, B, S_tkg, prg_id
            )

        while sbm.heap:
            sbm.pop_heap()
        sbm.set_auto_alloc(True)

        # Residual add #2
        nisa.tensor_tensor(dst=residual_sb, data1=residual_sb,
                           data2=moe_gathered_sb, op=nl.add)

    # ---- Single HBM store of the post-residual hidden state -------------
    # Final model.norm runs Python-side (NxDI's get_model_output).
    Y = nl.ndarray((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="Y")
    if prg_id == 0:
        # Inverse of the LNC-aware load: same 4-level pattern, dst side now.
        Y_flat = Y.reshape((BxS * H,))
        nisa.dma_copy(
            dst=Y_flat.ap(
                pattern=[
                    [H2, H0],
                    [H0 * H2, N_PRGS],
                    [1, H2],
                    [H, BxS],
                ],
                offset=0,
            ),
            src=residual_sb,
            dge_mode=nisa.dge_mode.hwdge,
        )
    if N_PRGS > 1:
        nisa.core_barrier(data=Y, cores=(0, 1))

    # Return KV refs so NCC preserves the in-place scatters (mirrors gpt-oss
    # / transformer_qwen). NxDI's model_wrapper aliases each returned KV
    # tensor back to its kv_mgr.past_key_values[i] slot.
    # DEBUG: also return the 3 layer-0 dumps for numerical comparison.
    return (
        (Y,)
        + tuple(K_post)
        + tuple(V_post)
        + (attn_in_dump_hbm, attn_out_dump_hbm, moe_out_dump_hbm)
    )


# ---------------------------------------------------------------------------
# Code-gen wrapper — explicit per-layer tensor args
# ---------------------------------------------------------------------------

def _build_multilayer_kernel(num_layers: int):
    """Code-gen a kernel function with explicit per-layer tensor args.

    NKI's frontend classifies tuple/list top-level args as scalars (no HBM
    binding), which breaks tracing and the input-output aliasing required
    for in-place KV scatter. So each weight and KV cache tensor must appear
    as its own top-level positional arg.
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
        f"def transformer_qwen3_moe_speculative(\n"
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

    fname = f"<generated:qwen3_moe_speculative_L{num_layers}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
    code = compile(src, fname, "exec")
    ns = {"_multilayer_body": _multilayer_body}
    exec(code, ns)
    return ns["transformer_qwen3_moe_speculative"]


transformer_qwen3_moe_speculative = _build_multilayer_kernel(NUM_LAYERS)
transformer_qwen3_moe_speculative_jit = nki.jit(transformer_qwen3_moe_speculative)

_kernel_cache: dict = {NUM_LAYERS: transformer_qwen3_moe_speculative_jit}


def get_multilayer_kernel_jit(num_layers: int):
    """Return the jit-compiled multilayer kernel for ``num_layers`` decoder
    layers. The returned object is a sequence indexed by LNC degree —
    ``[1]`` for LNC=1, ``[2]`` for LNC=2, matching the existing kernel's
    indexing convention. The integration in ``qwen_with_megakernel.py``
    selects ``[2]`` because Qwen3 runs at LNC=2.
    """
    if num_layers not in _kernel_cache:
        _kernel_cache[num_layers] = nki.jit(_build_multilayer_kernel(num_layers))
    return _kernel_cache[num_layers]
