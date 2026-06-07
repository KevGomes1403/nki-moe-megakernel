"""CPU unit test for the Stage C spec-decode commit-selection logic.

Proves the rank-LOCAL candidate -> seed-buffer copy in
``NeuronQwen36A3BForCausalLM.commit_specdec`` selects the correct candidate
(``cand_idx = accept_count - 1``: S1 on reject, S2 on accept) and writes it
into the live DeltaNet seed buffers via a same-shard device-to-device copy --
NEVER gathering across ranks (each rank's seed shard is filled from that same
rank's candidate shard).

This is the host state-commit step the greedy-equivalence gate depends on:
after a verify round, the committed recurrent/conv state carried into the next
forward must be exactly the candidate state after the last committed token.

No device required: we exercise the method against a synthetic
``nxd_model.state`` (a list-of-per-rank-dicts mirroring NxD's runtime layout),
binding only the small set of attributes ``commit_specdec`` touches.

Run:
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    python -m pytest models/qwen3_6_moe/tests/test_mtp_commit_selection.py -q
"""

from __future__ import annotations

import os
import sys

import torch

os.environ.setdefault("NXD_CPU_MODE", "1")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.qwen3_6_moe.modeling_qwen36_a3b import (  # noqa: E402
    NeuronQwen36A3BForCausalLM,
)


# Production-ish small dims for the test (per-rank shard shapes).
B = 1
SEQ_BLK = 2  # k+1 verify block
HD = 2       # v-heads per rank
KD = 4       # head_k_dim
VD = 4       # head_v_dim
CONV_DIM = 6
CONV_KM1 = 2
NUM_DN = 3   # DeltaNet layers
NUM_RANKS = 4  # TP=4


class _FakeNxdModel:
    def __init__(self, state):
        self.state = state


class _FakeWrapperModel:
    def __init__(self, state):
        self.nxd_model = _FakeNxdModel(state)


class _FakeWrapper:
    def __init__(self, state):
        self.model = _FakeWrapperModel(state)


class _CommitHarness:
    """Minimal object exposing only what commit_specdec / _dn_seed_keys need.

    Borrows the REAL methods off NeuronQwen36A3BForCausalLM so the logic under
    test is the production code, not a reimplementation.
    """

    _loop_state_dicts = NeuronQwen36A3BForCausalLM._loop_state_dicts
    _dn_seed_keys = NeuronQwen36A3BForCausalLM._dn_seed_keys
    commit_specdec = NeuronQwen36A3BForCausalLM.commit_specdec

    def __init__(self, state):
        # _loop_state_dicts resolves via getattr(self, "verify_model", ...).
        self.verify_model = _FakeWrapper(state)
        self.mtp_head_model = None
        self.token_generation_model = None
        self.context_encoding_model = None


def _build_state():
    """Build a synthetic per-rank state dict with distinct, known candidate /
    seed buffers per DeltaNet layer.

    Candidate S1 (axis-1 idx 0) and S2 (idx 1) hold distinct sentinel values so
    selection is observable; the seed buffers start as zeros and must end equal
    to the selected candidate slot. Each rank gets DIFFERENT values so a wrong
    cross-rank gather would be caught.
    """
    # absolute layer indices (mix of full/linear): linear at 0, 2, 4.
    dn_abs = [0, 2, 4]
    state = []
    for r in range(NUM_RANKS):
        rs = {}
        for i, abs_idx in enumerate(dn_abs):
            rec = torch.zeros(B, SEQ_BLK, HD, KD, VD)
            conv = torch.zeros(B, SEQ_BLK, CONV_DIM, CONV_KM1)
            # S1 and S2 sentinels: encode (rank, layer, slot) so any mis-select
            # or cross-rank leak is detectable.
            rec[:, 0] = 100 + 10 * r + i          # S1
            rec[:, 1] = 200 + 10 * r + i          # S2
            conv[:, 0] = 300 + 10 * r + i
            conv[:, 1] = 400 + 10 * r + i
            rs[f"recurrent_cand_buffers.{i}"] = rec
            rs[f"conv_cand_buffers.{i}"] = conv
            rs[f"layers.{abs_idx}.linear_attn.recurrent_state_buffer"] = torch.zeros(
                B, HD, KD, VD
            )
            rs[f"layers.{abs_idx}.linear_attn.conv_state_buffer"] = torch.zeros(
                B, CONV_DIM, CONV_KM1
            )
        # an unrelated full-attention KV key (should be untouched by commit).
        rs["kv_mgr.past_key_values.1"] = torch.full((B, 3, 8, KD), float(r))
        state.append(rs)
    return state, dn_abs


def _run_commit(cand_idx):
    state, dn_abs = _build_state()
    h = _CommitHarness(state)
    n = h.commit_specdec(cand_idx, NUM_DN)
    return state, dn_abs, n


def test_commit_accept_selects_s2_rank_local():
    """accept (cand_idx=1) -> every rank's seed == that rank's S2 candidate."""
    cand_idx = 1  # accept_count=2 -> S2
    state, dn_abs, n = _run_commit(cand_idx)
    assert n == NUM_RANKS * NUM_DN, f"expected {NUM_RANKS*NUM_DN} copies, got {n}"
    for r in range(NUM_RANKS):
        rs = state[r]
        for i, abs_idx in enumerate(dn_abs):
            rec_seed = rs[f"layers.{abs_idx}.linear_attn.recurrent_state_buffer"]
            conv_seed = rs[f"layers.{abs_idx}.linear_attn.conv_state_buffer"]
            # S2 sentinel for THIS rank/layer.
            assert torch.allclose(rec_seed, torch.full_like(rec_seed, 200 + 10 * r + i))
            assert torch.allclose(conv_seed, torch.full_like(conv_seed, 400 + 10 * r + i))


def test_commit_reject_selects_s1_rank_local():
    """reject (cand_idx=0) -> every rank's seed == that rank's S1 candidate."""
    cand_idx = 0  # accept_count=1 -> S1
    state, dn_abs, n = _run_commit(cand_idx)
    assert n == NUM_RANKS * NUM_DN
    for r in range(NUM_RANKS):
        rs = state[r]
        for i, abs_idx in enumerate(dn_abs):
            rec_seed = rs[f"layers.{abs_idx}.linear_attn.recurrent_state_buffer"]
            conv_seed = rs[f"layers.{abs_idx}.linear_attn.conv_state_buffer"]
            assert torch.allclose(rec_seed, torch.full_like(rec_seed, 100 + 10 * r + i))
            assert torch.allclose(conv_seed, torch.full_like(conv_seed, 300 + 10 * r + i))


def test_commit_is_rank_local_no_cross_rank_leak():
    """Each rank's seed must come from its OWN shard, not another rank's.

    The per-rank sentinels differ by rank (the 10*r term); after a correct
    rank-local commit, rank r's seed must NOT equal rank r'!=r's candidate.
    """
    state, dn_abs, _ = _run_commit(cand_idx=1)  # accept -> S2
    for r in range(NUM_RANKS):
        rec_seed = state[r][f"layers.{dn_abs[0]}.linear_attn.recurrent_state_buffer"]
        for other in range(NUM_RANKS):
            other_s2 = 200 + 10 * other + 0
            same = torch.allclose(rec_seed, torch.full_like(rec_seed, other_s2))
            if other == r:
                assert same, "rank-local select failed (own S2 not written)"
            else:
                assert not same, f"cross-rank leak: rank {r} got rank {other}'s S2"


def test_commit_leaves_unrelated_kv_untouched():
    """commit_specdec must not touch GQA KV buffers (verify is the GQA writer)."""
    state, _, _ = _run_commit(cand_idx=1)
    for r in range(NUM_RANKS):
        kv = state[r]["kv_mgr.past_key_values.1"]
        assert torch.allclose(kv, torch.full_like(kv, float(r))), (
            "commit_specdec modified an unrelated GQA KV buffer"
        )
