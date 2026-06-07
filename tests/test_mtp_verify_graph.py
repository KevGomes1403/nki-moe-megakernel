"""CPU smoke for the on-device verify backbone graph (NeuronVerifyModel).

Stage B.1 wires a full 40-layer backbone at n_active=2 as an additional NxDI
graph (beyond context-encoding, token-generation, and the MTP draft head). This
test does NOT compile on device. It proves the new graph module runs end to end
on the host: a tiny A3B-shaped config builds NeuronVerifyModel, and a single
2-token verify block over dummy inputs yields FINITE per-position logits of
shape [B, 2, vocab] plus the graph's private full-stack KV cache.

It also asserts the Stage B.1 non-destructiveness invariant on the host: the
live DeltaNet recurrent/conv state buffers are UNCHANGED after a verify pass
(verify_block seeds them read-only), so a rejected draft cannot corrupt decode.

Run:
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    python -m pytest models/qwen3_6_moe/tests/test_mtp_verify_graph.py -x -q
"""

from __future__ import annotations

import os
import sys

import torch

os.environ.setdefault("NXD_CPU_MODE", "1")

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
    NeuronVerifyModel,
    Qwen36A3BInferenceConfig,
)


def _ensure_parallel_state():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29582")
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
        enable_bucketing=False,
    )
    return Qwen36A3BInferenceConfig(
        neuron_config=nc,
        pad_token_id=0,
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
        rope_parameters={
            "rope_theta": 10000,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        norm_topk_prob=True,
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        enable_verify_backbone=True,
    )


def _build_verify_model(cfg: Qwen36A3BInferenceConfig) -> NeuronVerifyModel:
    cfg.neuron_config.n_positions = cfg.neuron_config.seq_len
    model = NeuronVerifyModel(cfg).float().eval()
    model.n_positions = cfg.neuron_config.seq_len
    return model


def _args24(block_token_ids, position_ids, seq_ids, prev_hidden, batch_size):
    empty_i32 = lambda: torch.zeros((0,), dtype=torch.int32)
    empty_f = lambda: torch.zeros((0,), dtype=torch.float32)
    sampling_params = torch.ones((batch_size, 3), dtype=torch.float32)
    args24 = [
        block_token_ids,  # 0  input_ids [B,2]
        empty_i32(),      # 1  attention_mask
        position_ids,     # 2  position_ids [B,2]
        seq_ids,          # 3  seq_ids
        sampling_params,  # 4  sampling_params
        prev_hidden,      # 5  prev_hidden [B,2,H]
    ]
    args24 += [empty_i32() for _ in range(15)]  # 6..20
    args24.append(empty_i32())  # 21 rotary_position_id
    args24.append(empty_f())    # 22 vision_embeddings
    args24.append(empty_i32())  # 23 vision_mask
    assert len(args24) == 24, len(args24)
    return args24


def test_verify_model_runs_and_yields_finite_logits():
    """One 2-token verify block yields finite per-position logits [B, 2, vocab]."""
    _ensure_parallel_state()
    cfg = _make_mini_config()
    model = _build_verify_model(cfg)

    torch.manual_seed(0)
    batch_size = 1
    block_token_ids = torch.tensor([[3, 7]], dtype=torch.int32)
    position_ids = torch.tensor([[5, 6]], dtype=torch.int32)
    seq_ids = torch.arange(batch_size, dtype=torch.int32)
    prev_hidden = torch.randn(batch_size, 2, cfg.hidden_size)

    args24 = _args24(block_token_ids, position_ids, seq_ids, prev_hidden, batch_size)

    with torch.no_grad():
        outputs = model(*args24)

    # Verify graph returns [logits, *updated_kv, *seed_passthrough, *candidates]
    # (Stage B.2 adds the S/conv candidate + trunk-hidden candidate block).
    logits = outputs[0]
    assert logits.shape == (batch_size, 2, cfg.vocab_size), logits.shape
    assert torch.isfinite(logits).all(), "verify logits contain non-finite values"

    # Private full-stack KV cache: 2 entries (K,V) per layer; plus the DeltaNet
    # SEED state buffers (recurrent + conv [+ trunk_hidden]) emitted as
    # passthrough state outputs; plus the candidate block.
    num_kv = len(model.kv_mgr.past_key_values)
    num_seed = len(model._deltanet_state_params)
    num_cand = len(model._verify_candidate_params)
    assert len(outputs) == 1 + num_kv + num_seed + num_cand, (
        len(outputs),
        num_kv,
        num_seed,
        num_cand,
    )
    for extra in outputs[1:]:
        assert torch.isfinite(extra).all(), "verify trailing output contains non-finite values"

    # Candidate block layout: per DeltaNet layer (recurrent [B,2,Vh,Kd,Vd],
    # conv [B,2,conv_dim,K-1]) interleaved, then 1 trunk hidden [B,2,H].
    num_dn = len(model.recurrent_cand_buffers)
    cand_start = 1 + num_kv + num_seed
    candidates = outputs[cand_start:]
    assert len(candidates) == 2 * num_dn + 1, (len(candidates), num_dn)
    for i in range(num_dn):
        rec = candidates[2 * i]
        conv = candidates[2 * i + 1]
        dn = [l for l in model.layers if hasattr(l, "linear_attn")][i].linear_attn
        assert rec.shape == (
            batch_size,
            2,
            dn.num_v_heads,
            dn.head_k_dim,
            dn.head_v_dim,
        ), rec.shape
        assert conv.shape == (
            batch_size,
            2,
            dn.conv_dim,
            dn.conv_kernel_size - 1,
        ), conv.shape
    trunk = candidates[-1]
    assert trunk.shape == (batch_size, 2, cfg.hidden_size), trunk.shape


def test_verify_is_read_only_on_live_deltanet_state():
    """Non-destructiveness: a verify pass leaves the live DeltaNet buffers intact."""
    _ensure_parallel_state()
    cfg = _make_mini_config()
    model = _build_verify_model(cfg)

    # Seed the live DeltaNet state buffers with known values.
    torch.manual_seed(1)
    snapshots = []
    for layer in model.layers:
        if hasattr(layer, "linear_attn"):
            dn = layer.linear_attn
            with torch.no_grad():
                dn.recurrent_state_buffer.copy_(
                    torch.randn_like(dn.recurrent_state_buffer)
                )
                dn.conv_state_buffer.copy_(torch.randn_like(dn.conv_state_buffer))
            snapshots.append(
                (
                    dn,
                    dn.recurrent_state_buffer.clone(),
                    dn.conv_state_buffer.clone(),
                )
            )

    batch_size = 1
    block_token_ids = torch.tensor([[3, 7]], dtype=torch.int32)
    position_ids = torch.tensor([[5, 6]], dtype=torch.int32)
    seq_ids = torch.arange(batch_size, dtype=torch.int32)
    prev_hidden = torch.randn(batch_size, 2, cfg.hidden_size)
    args24 = _args24(block_token_ids, position_ids, seq_ids, prev_hidden, batch_size)

    with torch.no_grad():
        model(*args24)

    # The live recurrent/conv buffers must be byte-identical to the seed.
    for dn, rec0, conv0 in snapshots:
        assert torch.equal(dn.recurrent_state_buffer, rec0), (
            "verify pass mutated a live recurrent_state_buffer"
        )
        assert torch.equal(dn.conv_state_buffer, conv0), (
            "verify pass mutated a live conv_state_buffer"
        )


def test_candidate_buffers_are_distinct_from_seed_buffers():
    """Risk #1: the S/conv candidate scratch buffers + the per-position
    trunk-hidden buffer must be DISTINCT objects from the live/seed
    recurrent/conv state buffers, and the verify alias map must exclude every
    seed-buffer id from the candidate alias entries.

    An input_output-aliased output is written in place to its aliased buffer, so
    if a candidate aliased a live/seed buffer the verify pass would silently
    commit that candidate and corrupt decode. This asserts the disjointness the
    surfacing relies on, and replays VerifyModelInstance.get()'s alias-map
    construction (seed entries then candidate entries) to confirm no seed buffer
    id leaks into the candidate alias entries.
    """
    _ensure_parallel_state()
    cfg = _make_mini_config()
    model = _build_verify_model(cfg)

    seed_params = list(model._deltanet_state_params)
    cand_params = list(model._verify_candidate_params)
    seed_ids = {id(p) for p in seed_params}
    cand_ids = {id(p) for p in cand_params}

    # 1) Object-level distinctness: no shared object between seed and candidate.
    assert seed_ids.isdisjoint(cand_ids), (
        "a candidate scratch buffer is the SAME object as a live/seed buffer"
    )
    # The live recurrent/conv buffers themselves must never be candidate buffers.
    for layer in model.layers:
        if hasattr(layer, "linear_attn"):
            dn = layer.linear_attn
            assert id(dn.recurrent_state_buffer) not in cand_ids
            assert id(dn.conv_state_buffer) not in cand_ids

    # 2) Replay the alias-map construction (mirrors VerifyModelInstance.get()):
    #    KV first, then seed passthrough, then candidates. Build it and assert no
    #    candidate alias entry points at a seed buffer.
    num_output_from_trace = 1
    num_kv = len(model.kv_mgr.past_key_values)
    state_start_idx = num_output_from_trace + num_kv

    alias_map = {}
    for i, param in enumerate(seed_params):
        alias_map[param] = state_start_idx + i
    cand_start_idx = state_start_idx + len(seed_params)
    for i, param in enumerate(cand_params):
        assert id(param) not in seed_ids, "candidate aliases a seed buffer (Risk #1)"
        alias_map[param] = cand_start_idx + i

    # No seed buffer's alias index collides with a candidate's, and the candidate
    # entries occupy the contiguous block right after the seed entries.
    seed_indices = {alias_map[p] for p in seed_params}
    cand_indices = {alias_map[p] for p in cand_params}
    assert seed_indices.isdisjoint(cand_indices)
    assert min(cand_indices) == cand_start_idx
    assert len(alias_map) == len(seed_params) + len(cand_params)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
