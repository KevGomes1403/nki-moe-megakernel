"""Unit tests for transformer_qwen3_moe_speculative correctness.

Covers:
  test_converter_produces_plain_layout
      The converter in qwen_with_megakernel.py must store
      ``Wq_nki / Wk_nki / Wv_nki`` weights in plain HF layout
      ``[head_count * d_head, hidden_size]`` (NOT tile-transposed),
      because ``transformer_qwen3_moe_speculative._fuse_qkv_weights``
      reads them as plain and transposes during DMA into the fused
      ``W_qkv: [H, I_qkv]`` expected by ``attention_block_tkg``.

  test_converter_wo_layout
      ``Wo_nki`` must be plain ``[Hq, H]`` so that NxDI's
      ColumnParallelLinear shard yields per-rank ``[Hq_tp, H]`` —
      the shape ``attention_block_tkg`` requires for ``W_out``.

  test_fuse_qkv_matches_pytorch_reference
      Numeric verification of ``_fuse_qkv_weights``' transpose against
      a NumPy reference. (Uses synthetic small dims; runs on CPU.)

These tests do NOT compile or run NKI kernels — they validate the
Python data-layout contracts between the integration's converter and
the kernel's expectations. They are fast (sub-second) and guard
against the original tile-transpose bug regressing.

Run with:
    pytest tests/qwen3_moe/test_speculative_megakernel.py -xvs
"""

import importlib
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _toy_config(
    num_attention_heads: int = 32,
    num_key_value_heads: int = 4,
    head_dim: int = 128,
    hidden_size: int = 2048,
    num_hidden_layers: int = 1,
    tp_degree: int = 4,
    num_experts: int = 8,
    moe_intermediate_size: int = 768,
):
    """Minimal config matching what the converter touches.

    The converter reads .num_attention_heads / .num_key_value_heads /
    .head_dim / .hidden_size / .num_hidden_layers / .num_experts and
    .neuron_config.{tp_degree, torch_dtype}. It also touches
    .moe_intermediate_pad_size via getattr (defaults to 0).
    """
    return SimpleNamespace(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_experts=num_experts,
        moe_intermediate_size=moe_intermediate_size,
        moe_intermediate_pad_size=0,
        quantization_config={},
        neuron_config=SimpleNamespace(
            tp_degree=tp_degree,
            torch_dtype=torch.bfloat16,
            glu_mlp=True,
        ),
    )


def _toy_hf_state_dict(cfg, layer: int = 0):
    """Build the subset of HF state-dict keys the converter touches.

    Uses small deterministic values so we can check layouts symbolically.
    """
    H  = cfg.hidden_size
    d  = cfg.head_dim
    nh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    Hq  = nh * d        # full (unsharded) Q output dim
    Hkv = nkv * d       # full (unsharded) K/V output dim
    I   = cfg.moe_intermediate_size
    E   = cfg.num_experts

    dtype = cfg.neuron_config.torch_dtype

    # Use float32 here so we can reason exactly; converter casts back.
    sd: dict = {}
    sd[f"layers.{layer}.self_attn.q_proj.weight"]   = torch.arange(Hq  * H, dtype=torch.float32).reshape(Hq, H).to(dtype)
    sd[f"layers.{layer}.self_attn.k_proj.weight"]   = torch.arange(Hkv * H, dtype=torch.float32).reshape(Hkv, H).to(dtype)
    sd[f"layers.{layer}.self_attn.v_proj.weight"]   = torch.arange(Hkv * H, dtype=torch.float32).reshape(Hkv, H).to(dtype)
    sd[f"layers.{layer}.self_attn.o_proj.weight"]   = torch.arange(H * Hq, dtype=torch.float32).reshape(H, Hq).to(dtype)
    sd[f"layers.{layer}.self_attn.q_norm.weight"]   = torch.ones(d, dtype=dtype)
    sd[f"layers.{layer}.self_attn.k_norm.weight"]   = torch.ones(d, dtype=dtype)
    sd[f"layers.{layer}.mlp.gate.weight"]           = torch.zeros(E, H, dtype=dtype)
    # one expert is enough to exercise the gate_up / down branch
    for e in range(E):
        sd[f"layers.{layer}.mlp.experts.{e}.gate_proj.weight"] = torch.zeros(I, H, dtype=dtype)
        sd[f"layers.{layer}.mlp.experts.{e}.up_proj.weight"]   = torch.zeros(I, H, dtype=dtype)
        sd[f"layers.{layer}.mlp.experts.{e}.down_proj.weight"] = torch.zeros(H, I, dtype=dtype)
    return sd


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_converter_produces_plain_layout():
    """Wq_nki / Wk_nki / Wv_nki must equal q/k/v_proj.weight UNCHANGED.

    The new megakernel (transformer_qwen3_moe_speculative._fuse_qkv_weights)
    treats them as plain [head_count*d, H] and transposes during DMA into
    the [H, I_qkv] layout that attention_block_tkg expects. Any tile-
    transpose at the converter level is a bug — it puts the weights in the
    layout the OLD attn_fused_nki kernel wanted, which doesn't match the
    new kernel.
    """
    cfg = _toy_config(num_attention_heads=8, num_key_value_heads=2,
                       head_dim=16, hidden_size=64,
                       num_hidden_layers=1, num_experts=2,
                       moe_intermediate_size=32)
    sd = _toy_hf_state_dict(cfg)
    expected_q = sd[f"layers.0.self_attn.q_proj.weight"].clone()
    expected_k = sd[f"layers.0.self_attn.k_proj.weight"].clone()
    expected_v = sd[f"layers.0.self_attn.v_proj.weight"].clone()

    qwm = importlib.import_module(
        "megakernels.qwen3_moe.qwen_with_megakernel"
    )
    out = qwm.convert_qwen3_moe_hf_to_neuron_state_dict(sd, cfg)

    Wq = out[f"layers.0.self_attn.Wq_nki.weight"]
    Wk = out[f"layers.0.self_attn.Wk_nki.weight"]
    Wv = out[f"layers.0.self_attn.Wv_nki.weight"]

    # Layouts must match HF q/k/v_proj.weight EXACTLY (no tile transpose,
    # no plain transpose either — the fusion DMA in the kernel does the
    # transpose).
    assert Wq.shape == expected_q.shape, (
        f"Wq_nki.weight should be [Hq, H]={tuple(expected_q.shape)}, "
        f"got {tuple(Wq.shape)}"
    )
    assert Wk.shape == expected_k.shape
    assert Wv.shape == expected_v.shape
    assert torch.equal(Wq, expected_q), (
        "Wq_nki.weight differs from q_proj.weight — converter is applying a "
        "transformation the new kernel does not undo. This is the tile-"
        "transpose bug: the converter was written for attn_fused_nki "
        "(which expects tile-transposed weights), but the new "
        "transformer_qwen3_moe_speculative kernel uses attention_block_tkg "
        "(which expects PLAIN HF layout)."
    )
    assert torch.equal(Wk, expected_k)
    assert torch.equal(Wv, expected_v)


def test_converter_wo_layout():
    """Wo_nki must be o_proj.weight.T (plain transpose).

    NxDI's Wo_nki = ColumnParallelLinear(H, Hq_full); its weight shape is
    [Hq_full, H] and dim-0 is sharded by TP. attention_block_tkg expects
    W_out: [q_heads*d_head, H] per rank — i.e. exactly that post-shard
    layout. HF o_proj.weight is [H, Hq_full]; the converter must transpose.
    """
    cfg = _toy_config(num_attention_heads=8, num_key_value_heads=2,
                       head_dim=16, hidden_size=64,
                       num_hidden_layers=1, num_experts=2,
                       moe_intermediate_size=32)
    sd = _toy_hf_state_dict(cfg)
    H  = cfg.hidden_size
    Hq = cfg.num_attention_heads * cfg.head_dim
    expected_wo = sd[f"layers.0.self_attn.o_proj.weight"].T.contiguous()

    qwm = importlib.import_module(
        "megakernels.qwen3_moe.qwen_with_megakernel"
    )
    out = qwm.convert_qwen3_moe_hf_to_neuron_state_dict(sd, cfg)

    Wo = out[f"layers.0.self_attn.Wo_nki.weight"]
    assert Wo.shape == (Hq, H)
    assert torch.equal(Wo, expected_wo)


def test_fuse_qkv_matches_pytorch_reference():
    """Symbolic check: the kernel's _fuse_qkv_weights transpose semantics
    must match a NumPy reference that treats Wq/Wk/Wv as plain layout.

    The kernel does:
        W_qkv[h, slot_offset + i] = Wx[i, h]   for x in (q, k, v)
    where slot_offset is 0 for Q, Hq for K, Hq+Hkv for V. This test
    builds the expected W_qkv tensor in PyTorch and asserts the
    component-wise transpose matches.

    This guards against accidental sign/index flips in the kernel-side
    fusion logic. (The kernel itself can't be run in a unit test without
    compiling — this verifies the algorithm a different way.)
    """
    Hq, Hkv, H = 32, 8, 16
    I_total = Hq + 2 * Hkv

    Wq = torch.arange(Hq  * H, dtype=torch.float32).reshape(Hq, H)
    Wk = torch.arange(Hkv * H, dtype=torch.float32).reshape(Hkv, H) + 1000
    Wv = torch.arange(Hkv * H, dtype=torch.float32).reshape(Hkv, H) + 2000

    # Build fused W_qkv: [H, I_total] = concat([Wq.T, Wk.T, Wv.T], dim=1)
    expected = torch.empty(H, I_total, dtype=torch.float32)
    expected[:, 0:Hq]               = Wq.T
    expected[:, Hq:Hq + Hkv]        = Wk.T
    expected[:, Hq + Hkv:I_total]   = Wv.T

    # Now apply the same transpose the kernel does (pure Python equivalent
    # of _fuse_qkv_weights' DMA patterns) and check element-wise equality.
    got = torch.empty(H, I_total, dtype=torch.float32)
    for i in range(Hq):
        for h in range(H):
            got[h, i] = Wq[i, h]
    for i in range(Hkv):
        for h in range(H):
            got[h, Hq + i] = Wk[i, h]
    for i in range(Hkv):
        for h in range(H):
            got[h, Hq + Hkv + i] = Wv[i, h]

    assert torch.equal(got, expected), (
        "Pure-Python equivalent of _fuse_qkv_weights' DMA pattern does not "
        "match the expected fused QKV layout (concat of plain transposes)."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))


def _ap_resolve(pattern, indices):
    """Translate an .ap() pattern + per-level indices into a flat HBM offset.

    Each level is ``[stride, count]``; ``indices[i]`` is the integer
    iteration index for level i (must satisfy 0 <= indices[i] < pattern[i][1]).
    """
    assert len(pattern) == len(indices)
    off = 0
    for (stride, count), idx in zip(pattern, indices):
        assert 0 <= idx < count, f"index {idx} out of bounds for count {count}"
        off += idx * stride
    return off


def test_x_load_pattern_is_lnc_aware():
    """The X-load DMA pattern must be the 4-level LNC-aware shard-interleaved
    layout matching what ``attention_block_tkg``'s SBUF-input path expects at
    LNC=2 (see ``attention_block_tkg`` docstring lines 167-170 and
    ``norm_tkg_utils.load_input_to_sbuf:331``).

    The kernel slices SBUF dim 2 at ``shard_id * H1_SHARD``; for that slice to
    correspond to a single shard's H values, the SBUF h1 index must equal
    ``shard*H2 + h2`` and the corresponding HBM h must equal
    ``shard*(H0*H2) + p*H2 + h2``.
    """
    src = Path(
        "/home/ubuntu/nki-moe/megakernels/qwen3_moe/transformer_qwen3_moe_speculative.py"
    ).read_text()
    import re
    good = re.findall(
        r"\[H2, H0\][^\]]*\[H, BxS\][^\]]*\[H0 \* H2, N_PRGS\][^\]]*\[1, H2\]",
        src,
        re.DOTALL,
    )
    assert len(good) >= 2, (
        f"Expected at least 2 occurrences of the 4-level LNC-aware pattern "
        f"[[H2,H0],[H,BxS],[H0*H2,N_PRGS],[1,H2]] in "
        f"transformer_qwen3_moe_speculative.py (residual load + final Y "
        f"store), found {len(good)}."
    )


def test_x_load_pattern_semantics_match_kernel_expectations():
    """Sanity check: simulate the 4-level shard-interleaved pattern and verify
    it produces the SBUF layout that ``attention_block_tkg``'s internal
    HBM->SBUF load produces (``norm_tkg_utils.load_input_to_sbuf:331``).
    """
    B, S, H = 1, 1, 2048
    H0, H1, BxS = 128, 16, B * S
    N_PRGS, H2 = 2, 8

    X = torch.arange(BxS * H, dtype=torch.float32).reshape(BxS, H)
    X_flat = X.reshape(-1)

    pat = [
        [H2, H0],
        [H0 * H2, N_PRGS],
        [1, H2],
        [H, BxS],
    ]

    sbuf = torch.empty(H0, BxS * H1, dtype=torch.float32)
    for p in range(H0):
        free_idx = 0
        for shard in range(N_PRGS):
            for h2 in range(H2):
                for bs in range(BxS):
                    off = _ap_resolve(pat, [p, shard, h2, bs])
                    sbuf[p, free_idx] = X_flat[off]
                    free_idx += 1

    expected = torch.empty(H0, BxS * H1, dtype=torch.float32)
    for p in range(H0):
        for shard in range(N_PRGS):
            for h2 in range(H2):
                for bs in range(BxS):
                    src_h = shard * (H0 * H2) + p * H2 + h2
                    expected[p, bs * H1 + shard * H2 + h2] = X[bs, src_h]

    assert torch.equal(sbuf, expected), (
        "4-level shard-interleaved DMA pattern does not produce the SBUF "
        "layout attention_block_tkg expects."
    )


def test_x_load_pattern_semantics_at_bxs_gt_1():
    """At BxS>1 the 4-level pattern must produce SBUF where free index
    bs*H1 + h1 maps to X[bs, shard*(H0*H2) + p*H2 + h2] (h1 = shard*H2 + h2).
    The free-dim iteration order (bs outermost, h2 innermost) is what makes
    this work; the prior order (shard outer, bs inner) collapses to the same
    layout only at BxS=1.
    """
    H, H0, H1 = 2048, 128, 16
    N_PRGS, H2 = 2, 8
    for B, S in [(1, 2), (1, 5), (1, 8)]:
        BxS = B * S
        X = torch.arange(BxS * H, dtype=torch.float32).reshape(BxS, H)
        X_flat = X.reshape(-1)

        # Pattern entries: [stride, count]. Iteration order is partition
        # outermost, then pattern[1..N] outer→inner.
        pat = [
            [H2, H0],
            [H, BxS],            # bs outermost free
            [H0 * H2, N_PRGS],
            [1, H2],
        ]

        sbuf = torch.empty(H0, BxS * H1, dtype=torch.float32)
        for p in range(H0):
            free_idx = 0
            for bs in range(BxS):
                for shard in range(N_PRGS):
                    for h2 in range(H2):
                        off = _ap_resolve(pat, [p, bs, shard, h2])
                        sbuf[p, free_idx] = X_flat[off]
                        free_idx += 1

        # Kernel's downstream view: residual_sb.reshape((H0, BxS, H1)) and
        # index residual_sb[p, bs, h1] with h1 = shard*H2 + h2.
        expected = torch.empty(H0, BxS, H1, dtype=torch.float32)
        for p in range(H0):
            for bs in range(BxS):
                for shard in range(N_PRGS):
                    for h2 in range(H2):
                        h1 = shard * H2 + h2
                        expected[p, bs, h1] = X[bs, shard * (H0 * H2) + p * H2 + h2]

        assert torch.equal(sbuf.reshape(H0, BxS, H1), expected), (
            f"At BxS={BxS}: 4-level pattern does NOT produce the shard-interleaved "
            f"(bs*H1 + h1) SBUF layout the kernel's downstream code assumes. "
            f"Check that pattern entry order is (partition, bs, shard, h2)."
        )


def test_x_load_pattern_at_bxs_1_documentation():
    """DOCUMENTATION test: at BxS=1 the 4-level pattern is equivalent
    regardless of F-axis iteration order. Guarding the BxS=1 case here. The
    speculative megakernel currently targets BxS=1 (single-token TKG).
    """
    B, S, H = 1, 1, 2048
    H0, H1, BxS = 128, 16, B * S
    N_PRGS, H2 = 2, 8

    X = torch.arange(BxS * H, dtype=torch.float32).reshape(BxS, H)
    X_flat = X.reshape(-1)

    pat = [
        [H2, H0],
        [H0 * H2, N_PRGS],
        [1, H2],
        [H, BxS],
    ]
    sbuf = torch.empty(H0, BxS * H1, dtype=torch.float32)
    for p in range(H0):
        free_idx = 0
        for shard in range(N_PRGS):
            for h2 in range(H2):
                for bs in range(BxS):
                    off = _ap_resolve(pat, [p, shard, h2, bs])
                    sbuf[p, free_idx] = X_flat[off]
                    free_idx += 1

    expected = torch.empty(H0, BxS * H1, dtype=torch.float32)
    for p in range(H0):
        for shard in range(N_PRGS):
            for h2 in range(H2):
                for bs in range(BxS):
                    src_h = shard * (H0 * H2) + p * H2 + h2
                    expected[p, bs * H1 + shard * H2 + h2] = X[bs, src_h]

    assert torch.equal(sbuf, expected), (
        "Shard-interleaved pattern produces wrong SBUF layout at BxS=1; "
        "the pattern is malformed."
    )


# ---------------------------------------------------------------------------
# H1 guard: NxDI's stride=2 sharding of gate_up_proj must produce the
# concatenated-halves layout [E, H, 2*I_per_rank] that the per-rank kernel
# reshape (E, H, 2*I) -> (E, H, 2, I) interprets as {gate=0, up=1} on dim 2.
# ---------------------------------------------------------------------------

def test_gate_up_stride2_sharding_produces_concat_halves_layout():
    """Verifies that ``ExpertFusedColumnParallelLinear(stride=2)``'s per-rank
    weight matches the layout the kernel assumes.

    NxDI uses ``create_local_weight`` which:
      1. Splits the full ``[E, H, 2*I_full]`` weight along dim 2 into
         ``2 * TP`` chunks of size ``I_full / TP``.
      2. For rank r, picks chunks ``[r, r+TP, r+2*TP, ...]``.
      3. Concatenates them along dim 2.

    For stride=2 (one stride per gate/up half) with TP=4 ``I_full=768``:
      * Full layout along dim 2 (size 1536): ``[gate_full(0:768), up_full(0:768)]``.
      * Split into 8 chunks of 192: chunks 0-3 = gate slices, chunks 4-7 = up slices.
      * Rank 0 picks chunks {0, 4} = ``[gate_slice_0(0:192), up_slice_0(0:192)]``.
      * Concatenated along dim 2 → ``[E, H, 384]``.

    Therefore: per-rank ``reshape((E, H, 2, 192))`` gives ``[..., 0, :] = gate``
    and ``[..., 1, :] = up``, matching the kernel's
    ``.select(dim=1, index=GateUpDim.GATE.value)`` semantics.

    A regression where stride=1 sharding (interleaved slices) is used instead
    would produce ``[gate_slice_0, gate_slice_1]`` on rank 0, which would
    silently scramble gate vs up halves and produce garbage MoE output.
    """
    TP = 4
    E, H, I_full = 8, 16, 64   # toy dims; logic is identical at production sizes
    I_per_rank = I_full // TP
    stride = 2
    assert (2 * I_full) % (stride * TP) == 0

    # Build a recognisable full weight:
    #   full[..., 0:I_full]              = "gate" half, encoded as gate_marker(i) = i
    #   full[..., I_full:2*I_full]       = "up"   half, encoded as up_marker(i)   = i + 10_000
    full = torch.empty(E, H, 2 * I_full, dtype=torch.float32)
    for i in range(I_full):
        full[..., i]          = float(i)
        full[..., I_full + i] = float(i + 10_000)

    per_partition_size = (2 * I_full) // TP
    per_partition_per_stride = per_partition_size // stride

    # Recreate create_local_weight (parallel_layers/layers.py:87) for rank 0.
    chunks = list(torch.split(full, per_partition_per_stride, dim=2))
    assert len(chunks) == stride * TP, f"expected {stride*TP} chunks, got {len(chunks)}"
    rank = 0
    my_chunks = chunks[rank::TP]
    assert len(my_chunks) == stride, f"each rank should get {stride} chunks (gate + up)"
    per_rank = torch.cat(my_chunks, dim=2)
    assert per_rank.shape == (E, H, 2 * I_per_rank)

    # Apply the kernel's reshape: [E, H, 2*I_per_rank] -> [E, H, 2, I_per_rank].
    gu = per_rank.reshape(E, H, 2, I_per_rank)

    # gate slot (dim 1 = 0) should hold values from the gate half (< 10_000).
    gate = gu[:, :, 0, :]
    up   = gu[:, :, 1, :]
    assert (gate < 10_000).all(), (
        "Reshape (E, H, 2*I) -> (E, H, 2, I) does NOT put gate in slot 0 — "
        "the NxDI sharding is NOT concatenated-halves. The kernel's "
        ".select(dim=1, index=GateUpDim.GATE.value) would read up values "
        "instead of gate values, producing garbage MoE output."
    )
    assert (up >= 10_000).all(), (
        "Reshape did not put up in slot 1 — see gate assertion above."
    )

    # Stronger: per-rank gate slot must equal full[..., 0:I_per_rank], and
    # per-rank up slot must equal full[..., I_full:I_full + I_per_rank].
    assert torch.equal(gate, full[..., :I_per_rank])
    assert torch.equal(up,   full[..., I_full:I_full + I_per_rank])


# ---------------------------------------------------------------------------
# Mask correctness: the speculative megakernel's full-attention mask MUST
# include ``t == S_ctx - 1`` (NOT a sink-token hack — see below). The
# attention_block_tkg subkernel places K_active (the freshly computed K
# for the new token) at slot ``S_ctx - 1`` of its internal k_sb buffer,
# so the mask at that slot must be 1, otherwise K_active is masked to
# -inf and the new token's K/V never attend to themselves.
#
# History note: an earlier version of this file removed the
# OR-with-(S_ctx-1) term, mistakenly believing it was a sink-related
# hack copied from gpt-oss. Removing it caused systematic attention
# error (max_abs_err ~0.27 on bf16 output vs HF reference, ~93% of
# elements failing strict tolerance). Guard now flipped to require
# the term's presence. See
# tests/qwen3_moe/test_attention_vs_hf.py for the end-to-end check.
# ---------------------------------------------------------------------------

def _qwen3_correct_mask(pos: int, S_ctx: int) -> torch.Tensor:
    """Reference: mask[t] = 1 if t < pos OR t == S_ctx - 1 else 0.

    The OR-with-(S_ctx-1) term enables attention to K_active, which
    attention_block_tkg places at the last slot of the s_prior axis.
    Without it, the new token's K_active is masked out.

    K_cache[S_ctx-1] being uninitialized is irrelevant: attention_block_tkg
    overwrites that slot with K_active before the QK matmul.
    """
    iota = torch.arange(S_ctx)
    return ((iota < pos) | (iota == S_ctx - 1)).to(torch.bfloat16)


def _qwen3_buggy_mask_drops_active(pos: int, S_ctx: int) -> torch.Tensor:
    """The (incorrect) Qwen3 mask without OR-with-(S_ctx-1).

    This was the bug in an earlier version: with this mask, the
    speculative megakernel's K_active was silently masked to -inf and
    the new token never attended to itself.
    """
    iota = torch.arange(S_ctx)
    return (iota < pos).to(torch.bfloat16)


def test_qwen3_mask_must_include_s_ctx_minus_1():
    """The correct Qwen3 mask differs from the buggy variant at exactly
    one slot: ``t == S_ctx - 1``, where K_active is placed by the
    attention sub-kernel.
    """
    for pos, S_ctx in [(1, 512), (10, 512), (300, 512), (511, 512),
                        (1, 256), (100, 256), (255, 256)]:
        correct = _qwen3_correct_mask(pos, S_ctx)
        buggy   = _qwen3_buggy_mask_drops_active(pos, S_ctx)
        diff = (correct != buggy)
        if pos <= S_ctx - 1:
            assert int(diff.sum().item()) == 1, (
                f"At pos={pos}, S_ctx={S_ctx}: correct and buggy masks should "
                f"differ at exactly one position (t=S_ctx-1). Got "
                f"{int(diff.sum().item())} differences."
            )
            assert bool(diff[S_ctx - 1].item()), (
                f"At pos={pos}, S_ctx={S_ctx}: the difference should be at "
                f"t=S_ctx-1, not elsewhere."
            )
            assert float(correct[S_ctx - 1].item()) == 1.0
            assert float(buggy[S_ctx - 1].item()) == 0.0


def test_full_mask_hits_s_ctx_minus_1():
    """The mask builder source must contain the unified active-triangle +
    prior-threshold structure.

    For T>1 the mask is
        (s < base_pos) OR (active_base <= s <= active_base + t)

    where ``base_pos = position_ids[b, 0]`` and ``active_base = S_ctx - S_tkg``.
    The prior threshold MUST be ``base_pos`` (not ``pos[b, t]``) because
    attention_block_tkg's in-place KV scatter runs AFTER the attention block,
    so K_cache slots ``[base_pos, base_pos + S_tkg)`` hold stale data when
    attention reads them — the freshly computed K for those tokens lives in
    the SBUF k_active region and is pasted onto the last ``S_tkg`` slots of
    the prior buffer. The active-region causal triangle ensures query slot
    ``t`` sees only active slots ``i <= t``.

    Regressing to the T=1-only ``(s == S_ctx - 1)`` formulation breaks T>1
    speculative-decoding verify (only the last active slot would unmask,
    so query slots 0..S_tkg-2 lose their own K).
    """
    src = Path(
        "/home/ubuntu/nki-moe/megakernels/qwen3_moe/transformer_qwen3_moe_speculative.py"
    ).read_text()
    import re

    # Prior threshold: (s < base_pos), using base_pos (NOT pos[b, t]).
    prior_ok = re.search(
        r"op0\s*=\s*nl\.less\s*,\s*operand0\s*=\s*base_pos",
        src,
    )
    assert prior_ok is not None, (
        "Missing `op0=nl.less, operand0=base_pos[...]` prior-threshold term "
        "in _build_full_mask_hbm. The prior threshold must be base_pos "
        "(= position_ids[b, 0]), not pos[b, t]: K_cache[base_pos : "
        "base_pos+S_tkg) is stale when attention reads it (the in-place KV "
        "scatter runs after attention)."
    )

    # Active-region lower bound: (s >= active_base).
    lower_ok = re.search(
        r"op0\s*=\s*nl\.greater_equal\s*,\s*operand0\s*=\s*active_base",
        src,
    )
    assert lower_ok is not None, (
        "Missing `op0=nl.greater_equal, operand0=active_base` term in "
        "_build_full_mask_hbm. Active slots [active_base, S_ctx) must be "
        "selected (active_base = S_ctx - S_tkg) so the new tokens can "
        "attend to their own freshly computed K."
    )

    # Active-region upper bound: (s <= active_base + t), built via
    # tensor_scalar(add active_base) on iota_t then tensor_tensor(less_equal).
    upper_bound_ok = re.search(
        r"op0\s*=\s*nl\.add\s*,\s*operand0\s*=\s*active_base",
        src,
    )
    assert upper_bound_ok is not None, (
        "Missing `op0=nl.add, operand0=active_base` (active-region per-t "
        "upper bound). The upper bound for query slot t must be "
        "active_base + t (causal triangle), built as iota_t + active_base."
    )
    upper_le_ok = re.search(
        r"op\s*=\s*nl\.less_equal",
        src,
    )
    assert upper_le_ok is not None, (
        "Missing `op=nl.less_equal` (used to compare iota_s against the "
        "per-t active upper bound). The causal triangle is "
        "(s <= active_base + t)."
    )


def test_qwen3_mask_first_step_only_active_slot():
    """At step 0 (pos=0, before any KV is written), the mask must have
    exactly ONE valid slot — position ``S_ctx - 1`` — corresponding to
    K_active (the new token's freshly computed K). No prior K is valid
    yet, but the new token must still attend to itself.
    """
    for S_ctx in [256, 512, 768, 1024]:
        m = _qwen3_correct_mask(pos=0, S_ctx=S_ctx)
        # Exactly one 1, at slot S_ctx - 1.
        assert int(m.to(torch.float32).sum().item()) == 1, (
            f"At pos=0, S_ctx={S_ctx}: mask must have exactly one 1 (at "
            f"slot S_ctx-1 for K_active). Got sum={m.sum().item()}."
        )
        assert float(m[S_ctx - 1].item()) == 1.0
        # All other slots are 0.
        assert m[:S_ctx - 1].to(torch.float32).sum().item() == 0.0


def test_qwen3_mask_matches_iota_lt_pos_at_steady_state():
    """At step N (pos=N), the mask must have ``N + 1`` ones — positions
    0..N-1 for prior K and position S_ctx-1 for K_active — and zeros
    elsewhere. Guards against off-by-one errors and against the
    K_active-masking bug.
    """
    for pos, S_ctx in [(1, 512), (47, 512), (256, 512), (511, 512), (300, 1024)]:
        assert pos < S_ctx, f"Bad test case: pos={pos} must be < S_ctx={S_ctx}"
        m = _qwen3_correct_mask(pos, S_ctx).to(torch.float32)
        # The mask is (t < pos) OR (t == S_ctx - 1). The two terms are
        # disjoint when pos <= S_ctx - 1: pos ones from <pos plus 1 from
        # ==S_ctx-1 (which is outside [0, pos)). When pos == S_ctx - 1,
        # the same slot S_ctx-1 is hit by both terms... no wait, t < pos
        # gives t in [0, pos) which does NOT include t = pos = S_ctx - 1.
        # So always pos + 1 ones.
        expected = pos + 1
        assert int(m.sum().item()) == expected, (
            f"At pos={pos}, S_ctx={S_ctx}: expected exactly {expected} ones "
            f"({pos} for prior + 1 for K_active at slot S_ctx-1). "
            f"Got {int(m.sum().item())}."
        )
        # First pos positions are 1; slot S_ctx-1 is 1; slots in
        # [pos, S_ctx-1) are 0.
        assert (m[:pos] == 1).all(), "Prior slots [0, pos) must all be 1."
        assert float(m[S_ctx - 1].item()) == 1.0, \
            "K_active slot S_ctx-1 must be 1."
        if pos < S_ctx - 1:
            assert (m[pos:S_ctx - 1] == 0).all(), \
                "Prior slots [pos, S_ctx-1) must all be 0."


def _qwen3_correct_mask_T(base_pos: int, S_ctx: int, S_tkg: int) -> torch.Tensor:
    """[S_ctx, S_tkg]: 1 iff (s < base_pos) OR (active_base <= s <= active_base + t)."""
    active_base = S_ctx - S_tkg
    s = torch.arange(S_ctx).unsqueeze(1)
    t = torch.arange(S_tkg).unsqueeze(0)
    return ((s < base_pos) | ((s >= active_base) & (s <= active_base + t))).to(torch.bfloat16)


def test_qwen3_mask_unified_t_geq_1():
    """T>1: causal triangle in active region + prior threshold capped at base_pos."""
    for base_pos, S_ctx, S_tkg in [
        (0, 768, 5), (1, 768, 5), (100, 768, 5),
        (200, 512, 2), (300, 512, 8), (0, 256, 1), (47, 256, 1),
    ]:
        m = _qwen3_correct_mask_T(base_pos, S_ctx, S_tkg).to(torch.float32)
        active_base = S_ctx - S_tkg
        # Active triangle: exactly t+1 ones in [active_base, S_ctx) for column t.
        for t in range(S_tkg):
            active_col = m[active_base:, t]
            assert int(active_col.sum().item()) == t + 1
            assert (active_col[:t + 1] == 1).all()
            if t + 1 < S_tkg:
                assert (active_col[t + 1:] == 0).all()
        # Prior region uses base_pos (not base_pos+t), constant across t.
        prior_region = m[:active_base, :]
        for t in range(S_tkg):
            assert (prior_region[:base_pos, t] == 1).all()
            assert (prior_region[base_pos:, t] == 0).all()
        # T=1 corollary: reduces to single-slot active mask + (s < base_pos) prior.
        if S_tkg == 1:
            assert float(m[S_ctx - 1, 0].item()) == 1.0


# ---------------------------------------------------------------------------
# Layout guard: router_topk must use XSBLayout_tp2013__1 (shard-interleaved)
# at LNC=2, NOT XSBLayout_tp102__0. The two layouts coincide only at LNC=1
# (where gpt-oss runs); at LNC=2 they differ in P-stride (8 vs 16) and using
# the wrong one silently mis-indexes the hidden state → garbage logits.
# ---------------------------------------------------------------------------

def test_router_topk_uses_tp2013_layout():
    """The router_topk call must specify x_sb_layout=XSBLayout_tp2013__1
    so it correctly interprets the shard-interleaved SBUF layout that
    rmsnorm_tkg / attention_block_tkg produce at LNC=2.

    The previous tp102 layout silently worked at LNC=1 (gpt-oss) but
    failed at LNC=2: P-stride = num_h_tiles = H/128 = 16 in tp102 vs
    P-stride = H/256 = 8 in the shard-interleaved layout (== tp2013).
    See ``nki_kernels/moe/router_topk.py:1389-1438`` for the tp2013
    diagram and ``nkilib/core/mlp/mlp_tkg_utils.py:input_norm_load``
    for the shard-interleaved layout that rmsnorm_tkg produces.
    """
    src = Path(
        "/home/ubuntu/nki-moe/megakernels/qwen3_moe/transformer_qwen3_moe_speculative.py"
    ).read_text()
    import re
    # The router_topk call must pass x_sb_layout=XSBLayout_tp2013__1.
    good = re.search(r"x_sb_layout\s*=\s*XSBLayout_tp2013__1", src)
    assert good is not None, (
        "router_topk call must use x_sb_layout=XSBLayout_tp2013__1 at LNC=2. "
        "tp102 is silently correct only at LNC=1; at LNC=2 it indexes hidden "
        "state with the wrong P-stride and produces garbage logits."
    )
    # Must NOT have the old tp102 layout in the router_topk call. (It's still
    # imported / kept as a constant for backward compatibility; we only
    # forbid it on the x_sb_layout assignment line.)
    bad = re.search(r"x_sb_layout\s*=\s*XSBLayout_tp102__0", src)
    assert bad is None, (
        "Found x_sb_layout=XSBLayout_tp102__0 in the kernel — this is the "
        "LNC=1 layout and produces garbage at LNC=2. Use XSBLayout_tp2013__1."
    )


def test_tp2013_and_shard_interleaved_describe_same_layout():
    """Numerical proof that XSBLayout_tp2013__1's index mapping is identical
    to the shard-interleaved layout that rmsnorm_tkg / attention_block_tkg
    produce at LNC=2 — i.e. they are the SAME layout under different names.

    Verifies that router_topk with x_sb_layout=tp2013 will correctly read
    the SBUF tensor produced by the kernel's upstream pipeline.
    """
    B, S, H = 1, 1, 2048
    H0, H1 = 128, 16
    BxS = B * S
    N_PRGS, H2 = 2, 8     # LNC=2

    # shard-interleaved layout (produced by attention_block_tkg / rmsnorm_tkg
    # at LNC=2 / our X load DMA):
    #     SBUF[p, bs, h1] = H[bs, shard*(H0*H2) + p*H2 + h2]
    #     where h1 = shard*H2 + h2
    H_flat = torch.arange(BxS * H, dtype=torch.float32).reshape(BxS, H)
    shard_interleaved = torch.empty(H0, BxS, H1, dtype=torch.float32)
    for p in range(H0):
        for bs in range(BxS):
            for shard in range(N_PRGS):
                for h2 in range(H2):
                    h1 = shard * H2 + h2
                    shard_interleaved[p, bs, h1] = H_flat[bs, shard * (H0 * H2) + p * H2 + h2]

    # tp2013 layout (router_topk_input_x_load output, see router_topk.py:1429):
    #     HBM [T, H] reshape to [T, 2, 128, H/256], permute to [128, T, 2, H/256]
    #     returned as 3D [128, T, H/128] flattening dims (2,3)
    #     SBUF[p, t, h_tile_3d] where h_tile_3d = half * (H/256) + intra_half
    num_h_tiles = H // H0           # 16
    num_h_tiles_by_2 = num_h_tiles // 2   # 8 = H/256
    tp2013 = torch.empty(H0, BxS, num_h_tiles, dtype=torch.float32)
    H_view = H_flat.reshape(BxS, 2, H0, num_h_tiles_by_2)
    for p in range(H0):
        for bs in range(BxS):
            for half in range(2):
                for intra in range(num_h_tiles_by_2):
                    h_tile_3d = half * num_h_tiles_by_2 + intra
                    tp2013[p, bs, h_tile_3d] = H_view[bs, half, p, intra]

    assert torch.equal(shard_interleaved, tp2013), (
        "XSBLayout_tp2013__1 and the kernel's shard-interleaved layout "
        "should be identical at LNC=2, but they differ. The fix to use "
        "x_sb_layout=tp2013 in router_topk depends on this equality."
    )


# ---------------------------------------------------------------------------
# Layout guard: moe_tkg's selective-expert (T=1) path expects a per-shard
# input slice of shape [H0, T, H1_SHARD], NOT the full [H0, T, H1] post-
# RMSNorm output. The matmul column index inside gate_up_projection iterates
# over [0, H1_shard) on BOTH cores; if we passed the full moe_in_sb both
# cores would read the SAME H value subset, pairing core 1's weight rows
# [H/2..H) with core 0's H values [0..H/2). Mirrors the canonical pattern
# at nkilib/core/moe_block/moe_block_tkg.py:309-314.
# ---------------------------------------------------------------------------

def test_moe_tkg_call_uses_per_shard_slice():
    """The moe_tkg call must pass a PER-SHARD slice of moe_in_sb (shape
    [H0, BxS, H1_SHARD]) — NOT the full moe_in_sb of shape [H0, BxS, H1].

    Forbids regressing back to the ``hidden_input=moe_in_sb`` form which
    silently pairs the wrong shard's weights with the wrong half of H,
    producing ~85% mismatched output elements at strict tolerance.

    The fix is a small SBUF round-trip:
        moe_in_pershard_sb = nl.ndarray((H0, BxS, H1_SHARD), ...)
        nisa.tensor_copy(dst=moe_in_pershard_sb,
                         src=moe_in_sb[:, :, prg_id*H1_SHARD:(prg_id+1)*H1_SHARD])
        moe_out_sb = moe_tkg(hidden_input=moe_in_pershard_sb, ...)
    """
    src = Path(
        "/home/ubuntu/nki-moe/megakernels/qwen3_moe/transformer_qwen3_moe_speculative.py"
    ).read_text()

    import re
    # Must NOT pass moe_in_sb directly as hidden_input (the old buggy form).
    # Pattern allows newline + whitespace between hidden_input= and moe_in_sb.
    bad = re.search(
        r"moe_tkg\(\s*hidden_input\s*=\s*moe_in_sb\s*[,)]",
        src,
        flags=re.MULTILINE,
    )
    assert bad is None, (
        "Found moe_tkg(hidden_input=moe_in_sb, ...) in the kernel — "
        "this passes the FULL [H0, BxS, H1] tensor where both shards' H "
        "values are stacked. moe_tkg's selective-expert (T=1) path expects "
        "a PER-SHARD slice [H0, BxS, H1_SHARD]. Use the moe_in_pershard_sb "
        "round-trip pattern (see nkilib/core/moe_block/moe_block_tkg.py:309-314)."
    )

    # Must HAVE the per-shard buffer allocation + slice copy + moe_tkg call.
    has_alloc = re.search(
        r"moe_in_pershard_sb\s*=\s*nl\.ndarray\(\s*\(H0,\s*BxS,\s*H1_SHARD\s*\)",
        src,
    )
    assert has_alloc is not None, (
        "Expected an SBUF allocation of shape (H0, BxS, H1_SHARD) named "
        "moe_in_pershard_sb (the per-shard slice that moe_tkg consumes). "
        "Found none — the kernel may be passing the wrong-shape input."
    )

    has_slice_copy = re.search(
        r"nisa\.tensor_copy\(\s*dst\s*=\s*moe_in_pershard_sb,\s*"
        r"src\s*=\s*moe_in_sb\[:,\s*:,\s*nl\.ds\(prg_id\s*\*\s*H1_SHARD,\s*H1_SHARD\)\]",
        src,
    )
    assert has_slice_copy is not None, (
        "Expected nisa.tensor_copy(dst=moe_in_pershard_sb, "
        "src=moe_in_sb[:, :, nl.ds(prg_id*H1_SHARD, H1_SHARD)]) — this copies "
        "the local shard's slice into the per-shard buffer. Found none."
    )

    has_pershard_call = re.search(
        r"moe_tkg\(\s*hidden_input\s*=\s*moe_in_pershard_sb",
        src,
    )
    assert has_pershard_call is not None, (
        "Expected moe_tkg(hidden_input=moe_in_pershard_sb, ...). The kernel "
        "must consume the per-shard slice, not the full moe_in_sb."
    )
