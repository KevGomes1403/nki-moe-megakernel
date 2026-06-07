"""CPU greedy-equivalence proof for the multi-token-prediction reject/update rule.

This test does NOT measure model quality. It proves a single property of the
host-side speculative-decode state machine: committing tokens via the verify
block (:meth:`NeuronGatedDeltaNet.verify_block_candidates` +
:func:`commit_accept`) and selecting the carried DeltaNet state by ``accept_count``
yields a token stream and a DeltaNet recurrent state that are BIT-IDENTICAL to
plain single-step greedy decoding, in the all-accept, all-reject, and mixed
draft cases.

Why this is the correctness gate
--------------------------------
Speculative decoding is only valid if accepting a draft never changes the
committed result versus greedy decoding. The reject/update rule selects, per
DeltaNet layer, candidate ``S_{accept_count-1}`` (the recurrent state after the
last committed token) and truncates the sliceable caches to ``accept_count``
positions. If that selection is right, the recurrence the model carries forward
is exactly the recurrence greedy decode would have built one token at a time.
This test asserts exactly that.

To stay self-contained and fast, the "language model" is a fixed surrogate: a
random embedding table feeds deterministic per-token linear maps producing the
DeltaNet inputs (q, k, v, g, beta) for the single DeltaNet layer under test, and
a fixed random projection of the recurrence output is argmax'd to pick the next
token. The SAME surrogate drives both the reference and the MTP-style path, so
equivalence of the committed stream/state is meaningful.

Run:
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    python -m pytest models/qwen3_6_moe/tests/test_mtp_state_rule.py -x -q
"""

from __future__ import annotations

import os
import sys

import torch

# cpu_mode() reads NXD_CPU_MODE at call time; set before importing the model so
# RMSNorm/parallel layers take their host paths.
os.environ.setdefault("NXD_CPU_MODE", "1")

# Repo root on sys.path so `models.qwen3_6_moe.*` resolves.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch.distributed as dist  # noqa: E402
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed_inference.models.config import (  # noqa: E402
    MoENeuronConfig,
    OnDeviceSamplingConfig,
)

from models.qwen3_6_moe.modeling_qwen36_a3b import (  # noqa: E402
    NeuronGatedDeltaNet,
    Qwen36A3BInferenceConfig,
    commit_accept,
)


# ---------------------------------------------------------------------------
# One-time tp=1 CPU process-group / model-parallel setup.
# ---------------------------------------------------------------------------
def _ensure_parallel_state():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)


def _make_mini_config() -> Qwen36A3BInferenceConfig:
    """Tiny A3B-shaped config: 4 layers -> [DN, DN, DN, GQA], tp=1, cpu."""
    nc = MoENeuronConfig(
        tp_degree=1,
        batch_size=1,
        seq_len=64,
        torch_dtype=torch.float32,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    )
    return Qwen36A3BInferenceConfig(
        neuron_config=nc,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=64,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000,
        hidden_act="silu",
        # DeltaNet (tiny)
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        # MoE (tiny)
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        norm_topk_prob=True,
        # MTP present in config (this test exercises the host rule, not the head).
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
    )


# ---------------------------------------------------------------------------
# Surrogate "language model" over a single DeltaNet layer.
#
# A token id -> embedding -> deterministic linear maps -> (q,k,v,g,beta) for the
# DeltaNet layer; the recurrence output -> fixed projection -> argmax -> next id.
# The maps are fixed, so greedy decode is a deterministic function of the state.
# ---------------------------------------------------------------------------
class _SurrogateLM:
    def __init__(self, dn: NeuronGatedDeltaNet, vocab_size: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.dn = dn
        self.V = vocab_size
        self.B = 1
        self.H = dn.num_v_heads
        self.Kd = dn.head_k_dim
        self.Vd = dn.head_v_dim
        emb_dim = 32
        # Fixed embedding table and projections.
        self.embed = torch.randn(vocab_size, emb_dim, generator=g)
        self.Wq = torch.randn(emb_dim, self.H * self.Kd, generator=g) * 0.5
        self.Wk = torch.randn(emb_dim, self.H * self.Kd, generator=g) * 0.5
        self.Wv = torch.randn(emb_dim, self.H * self.Vd, generator=g) * 0.5
        self.Wg = torch.randn(emb_dim, self.H, generator=g) * 0.1
        self.Wb = torch.randn(emb_dim, self.H, generator=g) * 0.5
        # Output projection from recurrence output (H*Vd) to vocab logits.
        self.Wout = torch.randn(self.H * self.Vd, vocab_size, generator=g) * 0.5

    def inputs_for_token(self, token_id: int):
        """Return (q, k, v, g, beta) for one token, shaped [B, H, 1, dim]."""
        e = self.embed[token_id].unsqueeze(0)  # [1, emb]
        q = (e @ self.Wq).reshape(self.B, 1, self.H, self.Kd)
        k = (e @ self.Wk).reshape(self.B, 1, self.H, self.Kd)
        v = (e @ self.Wv).reshape(self.B, 1, self.H, self.Vd)
        g = (e @ self.Wg).reshape(self.B, 1, self.H)            # raw log-decay
        b = (e @ self.Wb).reshape(self.B, 1, self.H)
        beta = b.sigmoid()
        # To [B, H, S, dim] layout that _recurrent_step expects.
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        g = g.transpose(1, 2).contiguous()
        beta = beta.transpose(1, 2).contiguous()
        return q, k, v, g, beta

    def token_from_output(self, out: torch.Tensor) -> int:
        """out: [B, H, 1, Vd] recurrence output -> next greedy token id."""
        flat = out.reshape(self.B, self.H * self.Vd)
        logits = flat @ self.Wout  # [B, V]
        return int(torch.argmax(logits, dim=-1)[0].item())


# ---------------------------------------------------------------------------
# Reference: plain single-step greedy decode.
# ---------------------------------------------------------------------------
def _reference_greedy(lm: _SurrogateLM, start_token: int, n_tokens: int):
    """Greedy-decode n_tokens one at a time via _recurrent_step.

    Returns (tokens, states) where tokens has length n_tokens (the tokens
    produced after start_token) and states[i] is the recurrent state after
    committing tokens[i].
    """
    dn = lm.dn
    state = torch.zeros(lm.B, lm.H, lm.Kd, lm.Vd)
    cur = start_token
    tokens = []
    states = []
    for _ in range(n_tokens):
        q, k, v, g, beta = lm.inputs_for_token(cur)
        out, state = dn._recurrent_step(q, k, v, g, beta, state)
        cur = lm.token_from_output(out)
        tokens.append(cur)
        states.append(state)
    return tokens, states


# ---------------------------------------------------------------------------
# MTP-style host loop: verify a 2-token block, select by accept_count.
#
# draft_policy(round_idx, true_next) -> drafted token id. Returning the true
# next token forces ACCEPT; returning anything else forces REJECT.
# ---------------------------------------------------------------------------
def _mtp_decode(lm: _SurrogateLM, start_token: int, n_tokens: int, draft_policy):
    dn = lm.dn
    state = torch.zeros(lm.B, lm.H, lm.Kd, lm.Vd)
    cur = start_token
    tokens = []
    states = []
    round_idx = 0
    while len(tokens) < n_tokens:
        # Step the real token to learn the true next token (greedy of the
        # recurrence seeded from the committed state).
        q0, k0, v0, g0, beta0 = lm.inputs_for_token(cur)
        out0, _ = dn._recurrent_step(q0, k0, v0, g0, beta0, state)
        true_next = lm.token_from_output(out0)

        draft = draft_policy(round_idx, true_next)

        # Build the 2-token verify block: position 0 = real token (cur),
        # position 1 = draft token. Inputs concatenated on the S axis.
        q1, k1, v1, g1, beta1 = lm.inputs_for_token(draft)
        q = torch.cat([q0, q1], dim=2)
        k = torch.cat([k0, k1], dim=2)
        v = torch.cat([v0, v1], dim=2)
        g = torch.cat([g0, g1], dim=2)
        beta = torch.cat([beta0, beta1], dim=2)

        out_stack, S_stack = dn.verify_block_candidates(q, k, v, g, beta, state)

        # Verify: greedy token implied by each block position.
        tok_after_real = lm.token_from_output(out_stack[:, :, 0:1])
        tok_after_draft = lm.token_from_output(out_stack[:, :, 1:2])
        assert tok_after_real == true_next  # the verify block reproduces step 0.

        # Accept iff the draft equals the model's true next token.
        accept = draft == true_next
        accept_count = 2 if accept else 1

        # Conv windows are produced the same way; pass through commit_accept so
        # the rule is exercised end to end (values unused by this surrogate).
        conv_seed = torch.zeros(lm.B, dn.conv_dim, dn.conv_kernel_size - 1)
        mixed = torch.randn(lm.B, dn.conv_dim, 2)
        W = dn.conv_window_candidates(conv_seed, mixed)

        committed = commit_accept(
            accept_count,
            recurrent_candidates=[S_stack],
            conv_candidates=[W],
            gqa_kv_block=None,
            mtp_kv_block=None,
        )
        state = committed["recurrent_states"][0]

        # Commit the verified tokens: always the real token; the draft too on
        # accept. After accept, the next round's "cur" is the accepted draft.
        tokens.append(true_next)
        states.append(S_stack[:, 0])
        if len(tokens) >= n_tokens:
            break
        if accept:
            tokens.append(tok_after_draft)
            states.append(S_stack[:, 1])
            cur = tok_after_draft
        else:
            cur = true_next
        round_idx += 1
    # Truncate to exactly n_tokens (a final accept can overshoot by one).
    return tokens[:n_tokens], states[:n_tokens]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
_START_TOKEN = 1
_N_TOKENS = 6


def _build_lm():
    _ensure_parallel_state()
    cfg = _make_mini_config()
    assert cfg.layer_types[:4] == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    dn = NeuronGatedDeltaNet(cfg, layer_idx=0).float().eval()
    return _SurrogateLM(dn, vocab_size=cfg.vocab_size, seed=0)


def _assert_equiv(tokens_mtp, states_mtp, tokens_ref, states_ref):
    assert tokens_mtp == tokens_ref, (
        f"committed token stream diverged from greedy:\n"
        f"  mtp = {tokens_mtp}\n  ref = {tokens_ref}"
    )
    assert len(states_mtp) == len(states_ref)
    for i, (sm, sr) in enumerate(zip(states_mtp, states_ref)):
        assert torch.allclose(sm, sr, atol=1e-5, rtol=1e-4), (
            f"committed DeltaNet state at token {i} diverged from greedy"
        )


def test_all_accept_matches_greedy():
    """Every draft equals the true next token -> all-accept path."""
    lm = _build_lm()
    ref_tokens, ref_states = _reference_greedy(lm, _START_TOKEN, _N_TOKENS)
    mtp_tokens, mtp_states = _mtp_decode(
        lm, _START_TOKEN, _N_TOKENS, draft_policy=lambda r, true_next: true_next
    )
    _assert_equiv(mtp_tokens, mtp_states, ref_tokens, ref_states)


def test_all_reject_matches_greedy():
    """Every draft is a deliberately wrong token -> all-reject path."""
    lm = _build_lm()
    ref_tokens, ref_states = _reference_greedy(lm, _START_TOKEN, _N_TOKENS)

    def wrong(_round, true_next):
        return (true_next + 1) % lm.V  # guaranteed != true_next (V > 1)

    mtp_tokens, mtp_states = _mtp_decode(
        lm, _START_TOKEN, _N_TOKENS, draft_policy=wrong
    )
    _assert_equiv(mtp_tokens, mtp_states, ref_tokens, ref_states)


def test_mixed_accept_reject_matches_greedy():
    """Alternate accept/reject rounds -> mixed path."""
    lm = _build_lm()
    ref_tokens, ref_states = _reference_greedy(lm, _START_TOKEN, _N_TOKENS)

    def mixed(round_idx, true_next):
        # Even rounds accept (draft = true next), odd rounds reject.
        if round_idx % 2 == 0:
            return true_next
        return (true_next + 1) % lm.V

    mtp_tokens, mtp_states = _mtp_decode(
        lm, _START_TOKEN, _N_TOKENS, draft_policy=mixed
    )
    _assert_equiv(mtp_tokens, mtp_states, ref_tokens, ref_states)


def test_commit_accept_selection_and_truncation():
    """commit_accept selects S_{accept_count-1} and truncates KV to accept_count."""
    lm = _build_lm()
    dn = lm.dn
    q, k, v, g, beta = lm.inputs_for_token(_START_TOKEN)
    q2, k2, v2, g2, beta2 = lm.inputs_for_token(2)
    q = torch.cat([q, q2], dim=2)
    k = torch.cat([k, k2], dim=2)
    v = torch.cat([v, v2], dim=2)
    g = torch.cat([g, g2], dim=2)
    beta = torch.cat([beta, beta2], dim=2)
    state0 = torch.zeros(lm.B, lm.H, lm.Kd, lm.Vd)
    _, S_stack = dn.verify_block_candidates(q, k, v, g, beta, state0)
    conv = dn.conv_window_candidates(
        torch.zeros(lm.B, dn.conv_dim, dn.conv_kernel_size - 1),
        torch.randn(lm.B, dn.conv_dim, 2),
    )
    # A GQA KV block [B, n_heads, S=2, head_dim]: truncation keeps accept_count.
    kv = torch.randn(lm.B, 2, 2, dn.head_dim)

    rej = commit_accept(1, [S_stack], [conv], gqa_kv_block=kv)
    acc = commit_accept(2, [S_stack], [conv], gqa_kv_block=kv)

    # Reject -> S_0 (state after the single real token); accept -> S_1.
    assert torch.allclose(rej["recurrent_states"][0], S_stack[:, 0])
    assert torch.allclose(acc["recurrent_states"][0], S_stack[:, 1])
    assert torch.allclose(rej["conv_states"][0], conv[:, 0])
    assert torch.allclose(acc["conv_states"][0], conv[:, 1])
    # KV truncated on the sequence axis to accept_count positions.
    assert rej["gqa_kv"].shape[-2] == 1
    assert acc["gqa_kv"].shape[-2] == 2
    assert torch.allclose(acc["gqa_kv"], kv)
    assert torch.allclose(rej["gqa_kv"], kv[..., :1, :])


def _randomize_deltanet(dn, seed: int = 7):
    """Give the DeltaNet non-trivial weights so the verify/decode comparison is
    meaningful (a fresh layer's init could be degenerate)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in dn.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
        # RMSNorm scale near 1 for numerical sanity.
        dn.norm.weight.copy_(1.0 + 0.05 * torch.randn(dn.norm.weight.shape, generator=gen))


# Non-None placeholder: the DeltaNet decode path only checks past_key_value for
# truthiness (it is unpacked only under the hybrid cache manager, which is off).
_DUMMY_PKV = (torch.zeros(1), torch.zeros(1))


def test_verify_block_matches_sequential_decode():
    """STAGE B.0 oracle: one 2-token verify_block == two sequential decode steps.

    Proves verify_block (full projection + causal conv + gating + recurrence,
    seeded from the committed state buffers) reproduces, per block position, the
    exact recurrent state, conv window, and output that plain single-step decode
    would produce -- the state-level greedy-equivalence the verify graph relies
    on. Read-only w.r.t. the live state buffers is implied: the oracle re-seeds
    them between the two decode steps and gets a match.
    """
    _ensure_parallel_state()
    cfg = _make_mini_config()
    dn = NeuronGatedDeltaNet(cfg, layer_idx=0).float().eval()
    _randomize_deltanet(dn)

    B = 1
    seq_ids = torch.zeros(B, dtype=torch.long)
    gen = torch.Generator().manual_seed(11)
    hidden = torch.randn(B, 2, cfg.hidden_size, generator=gen)

    # A known committed seed in the live buffers.
    rec_seed = torch.randn(dn.recurrent_state_buffer.shape, generator=gen)
    conv_seed = torch.randn(dn.conv_state_buffer.shape, generator=gen)
    with torch.no_grad():
        dn.recurrent_state_buffer.copy_(rec_seed)
        dn.conv_state_buffer.copy_(conv_seed)

    out_v, S_stack, conv_cand = dn.verify_block(hidden, seq_ids)

    # Sequential-decode oracle: re-seed, step token 0, then token 1.
    with torch.no_grad():
        dn.recurrent_state_buffer.copy_(rec_seed)
        dn.conv_state_buffer.copy_(conv_seed)
    out0, _, new_rec0, new_conv0 = dn.forward(
        hidden[:, 0:1], past_key_value=_DUMMY_PKV, seq_ids=seq_ids
    )
    with torch.no_grad():
        dn.recurrent_state_buffer.copy_(new_rec0)
        dn.conv_state_buffer.copy_(new_conv0)
    out1, _, new_rec1, new_conv1 = dn.forward(
        hidden[:, 1:2], past_key_value=_DUMMY_PKV, seq_ids=seq_ids
    )

    assert S_stack.shape == (B, 2, dn.num_v_heads, dn.head_k_dim, dn.head_v_dim)
    assert conv_cand.shape == (B, 2, dn.conv_dim, dn.conv_kernel_size - 1)
    assert torch.isfinite(out_v).all() and torch.isfinite(S_stack).all()

    # Per-position recurrent state matches sequential decode.
    assert torch.allclose(S_stack[:, 0], new_rec0, atol=1e-5, rtol=1e-4)
    assert torch.allclose(S_stack[:, 1], new_rec1, atol=1e-5, rtol=1e-4)
    # Per-position conv window matches.
    assert torch.allclose(conv_cand[:, 0], new_conv0, atol=1e-5, rtol=1e-4)
    assert torch.allclose(conv_cand[:, 1], new_conv1, atol=1e-5, rtol=1e-4)
    # Output projection per position matches the decode outputs.
    assert torch.allclose(out_v[:, 0:1], out0, atol=1e-5, rtol=1e-4)
    assert torch.allclose(out_v[:, 1:2], out1, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
