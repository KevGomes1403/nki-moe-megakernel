"""CPU smoke for the on-device MTP draft graph (NeuronMTPDraftModel).

Stage A wires the MTP draft head as a third compiled NxDI graph (beyond
context-encoding and token-generation). This test does NOT compile on device.
It proves the new graph module runs end to end on the host: a tiny A3B-shaped
config with mtp_num_hidden_layers=1 builds NeuronMTPDraftModel, and a single
draft step over dummy inputs yields FINITE draft logits of shape [B, 1, vocab],
plus the draft layer's one-layer KV. This exercises the key risk -- whether the
MoE/attention CPU path runs inside the MTP decoder layer with a private one-layer
KV cache and the tied embeddings.

Run:
    cd /home/ubuntu/trainium-model-translation && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    python -m pytest models/qwen3_6_moe/tests/test_mtp_draft_graph.py -x -q
"""

from __future__ import annotations

import os
import sys

import torch

# cpu_mode() reads NXD_CPU_MODE at call time; set before importing the model so
# RMSNorm/parallel layers and the layer-boundary markers take their host paths.
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
    NeuronMTPDraftModel,
    Qwen36A3BInferenceConfig,
)


def _ensure_parallel_state():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29581")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)


def _make_mini_config() -> Qwen36A3BInferenceConfig:
    """Tiny A3B-shaped config: 4 layers -> [DN, DN, DN, GQA], tp=1, cpu, MTP on."""
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
        # head_dim=16 -> rope_dim=4 -> rope_dim//2=2, so mrope_section must sum to 2.
        rope_parameters={
            "rope_theta": 10000,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
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
        # MTP head present -> NeuronMTPDraftModel instantiates mtp_head.
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
    )


def _build_draft_model(cfg: Qwen36A3BInferenceConfig) -> NeuronMTPDraftModel:
    # n_positions feeds the one-layer KV cache shapes; mirror a single-bucket
    # decode-style graph at seq_len.
    cfg.neuron_config.n_positions = cfg.neuron_config.seq_len
    model = NeuronMTPDraftModel(cfg).float().eval()
    model.n_positions = cfg.neuron_config.seq_len
    return model


def test_draft_model_runs_and_yields_finite_logits():
    """One draft step yields finite logits [B, 1, vocab] + a finite one-layer KV."""
    _ensure_parallel_state()
    cfg = _make_mini_config()
    model = _build_draft_model(cfg)

    torch.manual_seed(0)
    batch_size = 1
    prev_hidden = torch.randn(batch_size, 1, cfg.hidden_size)
    next_token_ids = torch.tensor([[3]], dtype=torch.int32)
    position_ids = torch.tensor([[5]], dtype=torch.int32)
    seq_ids = torch.arange(batch_size, dtype=torch.int32)

    # The draft forward now takes the SAME 24 positional args as
    # NeuronQwen36A3BModel.forward (uniform arity for the shared executor).
    # Only idx0 (input_ids = committed next token), idx2 (position_ids),
    # idx3 (seq_ids) and idx5 (prev_hidden = trunk hidden) carry real data;
    # every other slot is an empty placeholder.
    empty_i32 = torch.zeros((0,), dtype=torch.int32)
    empty_f = torch.zeros((0,), dtype=torch.float32)
    sampling_params = torch.ones((batch_size, 3), dtype=torch.float32)
    args24 = [
        next_token_ids,   # 0  input_ids
        empty_i32,        # 1  attention_mask
        position_ids,     # 2  position_ids
        seq_ids,          # 3  seq_ids
        sampling_params,  # 4  sampling_params
        prev_hidden,      # 5  prev_hidden (trunk hidden)
    ]
    args24 += [empty_i32 for _ in range(15)]  # 6..20  unused placeholders
    args24.append(empty_i32)  # 21 rotary_position_id
    args24.append(empty_f)    # 22 vision_embeddings
    args24.append(empty_i32)  # 23 vision_mask
    assert len(args24) == 24, len(args24)

    with torch.no_grad():
        outputs = model(*args24)

    # Draft graph returns [draft_logits, draft_k, draft_v].
    assert len(outputs) == 3, f"expected [logits, k, v], got {len(outputs)} outputs"

    draft_logits = outputs[0]
    assert draft_logits.shape == (batch_size, 1, cfg.vocab_size), draft_logits.shape
    assert torch.isfinite(draft_logits).all(), "draft logits contain non-finite values"

    # The trailing one-layer KV (aliased back into kv_mgr on device) is finite.
    for kv in outputs[1:]:
        assert torch.isfinite(kv).all(), "draft KV contains non-finite values"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
