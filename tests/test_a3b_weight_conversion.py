"""CPU unit tests for the A3B HF-to-NxDI weight converter.

Validates:
  * router rename            mlp.gate -> mlp.moe.router.linear_router
  * routed expert fusion     256 x {gate,up,down}_proj  ->
                             mlp.moe.expert_mlps.mlp_op.{gate_up_proj,down_proj}
  * shared expert pass-through (singular name, owned by NeuronMoEBlock)
  * shared_expert_gate pass-through (sigmoid gate over shared output)
  * MTP key remap            mtp.layers.0.* -> mtp_head.*
  * outer wrapper preserves mtp.* keys
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

# Ensure the parent project root is on sys.path so 'models.qwen3_6_moe.*' resolves.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig  # noqa: E402

from models.qwen3_6_moe.modeling_qwen36_a3b import (  # noqa: E402
    NeuronQwen36A3BForCausalLM,
    Qwen36A3BInferenceConfig,
    convert_qwen36_a3b_hf_to_neuron_state_dict,
)


# ---------------------------------------------------------------------------
# Fixture builders -- a deliberately tiny A3B-shaped model.
# ---------------------------------------------------------------------------


def _make_mini_config(num_layers: int = 4, num_experts: int = 4) -> Qwen36A3BInferenceConfig:
    nc_cls = Qwen36A3BInferenceConfig.get_neuron_config_cls()
    nc = nc_cls(
        tp_degree=2,
        batch_size=1,
        seq_len=128,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    )
    cfg = Qwen36A3BInferenceConfig(
        neuron_config=nc,
        hidden_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_theta=10000,
        hidden_act="silu",
        # DeltaNet (tiny)
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        # MoE (tiny)
        num_experts=num_experts,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        norm_topk_prob=True,
        # MTP
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
    )
    return cfg


def _make_mini_state_dict(cfg: Qwen36A3BInferenceConfig) -> dict:
    sd: dict = {}
    H = cfg.hidden_size
    V = cfg.vocab_size
    I = cfg.moe_intermediate_size
    SI = cfg.shared_expert_intermediate_size
    num_heads = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim

    bf = torch.bfloat16
    sd["embed_tokens.weight"] = torch.randn(V, H, dtype=bf) * 0.02
    sd["lm_head.weight"] = torch.randn(V, H, dtype=bf) * 0.02
    sd["norm.weight"] = torch.zeros(H, dtype=bf)  # +1 convention

    for l in range(cfg.num_hidden_layers):
        sd[f"layers.{l}.input_layernorm.weight"] = torch.zeros(H, dtype=bf)
        sd[f"layers.{l}.post_attention_layernorm.weight"] = torch.zeros(H, dtype=bf)

        # --- MoE replaces dense MLP. HF ships routed experts already fused
        #     into 3D tensors per layer (gate_up_proj: (E, 2I, H);
        #     down_proj: (E, H, I)). Our converter renames + transposes.
        sd[f"layers.{l}.mlp.gate.weight"] = torch.randn(cfg.num_experts, H, dtype=bf) * 0.02
        sd[f"layers.{l}.mlp.experts.gate_up_proj"] = (
            torch.randn(cfg.num_experts, 2 * I, H, dtype=bf) * 0.02
        )
        sd[f"layers.{l}.mlp.experts.down_proj"] = (
            torch.randn(cfg.num_experts, H, I, dtype=bf) * 0.02
        )
        sd[f"layers.{l}.mlp.shared_expert.gate_proj.weight"] = torch.randn(SI, H, dtype=bf) * 0.02
        sd[f"layers.{l}.mlp.shared_expert.up_proj.weight"] = torch.randn(SI, H, dtype=bf) * 0.02
        sd[f"layers.{l}.mlp.shared_expert.down_proj.weight"] = torch.randn(H, SI, dtype=bf) * 0.02
        sd[f"layers.{l}.mlp.shared_expert_gate.weight"] = torch.randn(1, H, dtype=bf) * 0.02

        if cfg.layer_types[l] == "full_attention":
            # GQA: q_proj is doubled (query + output gate, interleaved per head).
            sd[f"layers.{l}.self_attn.q_proj.weight"] = (
                torch.randn(num_heads * head_dim * 2, H, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.self_attn.k_proj.weight"] = (
                torch.randn(num_kv * head_dim, H, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.self_attn.v_proj.weight"] = (
                torch.randn(num_kv * head_dim, H, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.self_attn.o_proj.weight"] = (
                torch.randn(H, num_heads * head_dim, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.self_attn.q_norm.weight"] = torch.zeros(head_dim, dtype=bf)
            sd[f"layers.{l}.self_attn.k_norm.weight"] = torch.zeros(head_dim, dtype=bf)
        else:
            key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
            value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
            conv_dim = key_dim * 2 + value_dim
            sd[f"layers.{l}.linear_attn.in_proj_qkv.weight"] = torch.randn(conv_dim, H, dtype=bf) * 0.02
            sd[f"layers.{l}.linear_attn.in_proj_z.weight"] = torch.randn(value_dim, H, dtype=bf) * 0.02
            sd[f"layers.{l}.linear_attn.in_proj_a.weight"] = torch.randn(cfg.linear_num_value_heads, H, dtype=bf) * 0.02
            sd[f"layers.{l}.linear_attn.in_proj_b.weight"] = torch.randn(cfg.linear_num_value_heads, H, dtype=bf) * 0.02
            sd[f"layers.{l}.linear_attn.conv1d.weight"] = (
                torch.randn(conv_dim, 1, cfg.linear_conv_kernel_dim, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.linear_attn.A_log"] = torch.randn(cfg.linear_num_value_heads, dtype=bf)
            sd[f"layers.{l}.linear_attn.dt_bias"] = torch.randn(cfg.linear_num_value_heads, dtype=bf)
            sd[f"layers.{l}.linear_attn.norm.weight"] = torch.randn(value_dim, dtype=bf) * 0.5
            sd[f"layers.{l}.linear_attn.out_proj.weight"] = torch.randn(H, value_dim, dtype=bf) * 0.02

    # --- MTP weights (HF-style "mtp.layers.0.*") ---
    sd["mtp.layers.0.embed_layernorm.weight"] = torch.zeros(H, dtype=bf)
    sd["mtp.layers.0.hidden_layernorm.weight"] = torch.zeros(H, dtype=bf)
    sd["mtp.layers.0.input_proj.weight"] = torch.randn(H, 2 * H, dtype=bf) * 0.02
    sd["mtp.layers.0.final_layernorm.weight"] = torch.zeros(H, dtype=bf)
    sd["mtp.layers.0.lm_head.weight"] = torch.randn(V, H, dtype=bf) * 0.02
    # Inner decoder-layer weights for MTP get routed under mtp_head.decoder_layer.*
    sd["mtp.layers.0.input_layernorm.weight"] = torch.zeros(H, dtype=bf)
    sd["mtp.layers.0.post_attention_layernorm.weight"] = torch.zeros(H, dtype=bf)

    return sd


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRouterRename(unittest.TestCase):
    def test_gate_to_linear_router(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        for l in range(cfg.num_hidden_layers):
            self.assertNotIn(f"layers.{l}.mlp.gate.weight", out)
            self.assertIn(f"layers.{l}.mlp.moe.router.linear_router.weight", out)
            self.assertEqual(
                out[f"layers.{l}.mlp.moe.router.linear_router.weight"].shape,
                (cfg.num_experts, cfg.hidden_size),
            )


class TestRoutedExpertFusion(unittest.TestCase):
    def test_fused_shapes(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        for l in range(cfg.num_hidden_layers):
            gu = out[f"layers.{l}.mlp.moe.expert_mlps.mlp_op.gate_up_proj.weight"]
            dp = out[f"layers.{l}.mlp.moe.expert_mlps.mlp_op.down_proj.weight"]
            self.assertEqual(gu.shape, (cfg.num_experts, cfg.hidden_size, 2 * cfg.moe_intermediate_size))
            self.assertEqual(dp.shape, (cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size))

    def test_fused_hf_keys_removed(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        for l in range(cfg.num_hidden_layers):
            self.assertNotIn(f"layers.{l}.mlp.experts.gate_up_proj", out)
            self.assertNotIn(f"layers.{l}.mlp.experts.down_proj", out)

    def test_transpose_values_preserved(self):
        """NxDI gate_up_proj should be HF gate_up_proj transposed on dims (1, 2)."""
        cfg = _make_mini_config(num_layers=4, num_experts=2)
        sd = _make_mini_state_dict(cfg)
        orig_gu = sd["layers.0.mlp.experts.gate_up_proj"].clone()
        orig_dp = sd["layers.0.mlp.experts.down_proj"].clone()
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        gu = out["layers.0.mlp.moe.expert_mlps.mlp_op.gate_up_proj.weight"]
        dp = out["layers.0.mlp.moe.expert_mlps.mlp_op.down_proj.weight"]
        torch.testing.assert_close(gu, orig_gu.transpose(1, 2).contiguous())
        torch.testing.assert_close(dp, orig_dp.transpose(1, 2).contiguous())


class TestSharedExpert(unittest.TestCase):
    def test_pass_through(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        for l in range(cfg.num_hidden_layers):
            for suf in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
                self.assertIn(f"layers.{l}.mlp.shared_expert.{suf}", out)

    def test_shared_expert_gate_pass_through(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        for l in range(cfg.num_hidden_layers):
            self.assertIn(f"layers.{l}.mlp.shared_expert_gate.weight", out)


class TestMTPRemap(unittest.TestCase):
    def test_outer_wrapper_preserves_mtp(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        # Outer wrapper would also strip prefixes; verify mtp.* survives.
        prefixed = {f"model.{k}" if not k.startswith("lm_head.") and not k.startswith("mtp.") else k: v for k, v in sd.items()}
        out = NeuronQwen36A3BForCausalLM.convert_hf_to_neuron_state_dict(prefixed, cfg)
        # After conversion, MTP keys live under mtp_head.*
        self.assertIn("mtp_head.embed_norm.weight", out)
        self.assertIn("mtp_head.hidden_norm.weight", out)
        self.assertIn("mtp_head.eh_proj.weight", out)
        self.assertIn("mtp_head.final_norm.weight", out)
        self.assertIn("mtp_head.mtp_lm_head.weight", out)
        # Decoder-layer-internal MTP weights end up under mtp_head.decoder_layer.*
        self.assertIn("mtp_head.decoder_layer.input_layernorm.weight", out)
        self.assertIn("mtp_head.decoder_layer.post_attention_layernorm.weight", out)

    def test_inner_remap(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        self.assertNotIn("mtp.layers.0.embed_layernorm.weight", out)
        self.assertIn("mtp_head.embed_norm.weight", out)
        self.assertEqual(
            out["mtp_head.eh_proj.weight"].shape,
            (cfg.hidden_size, 2 * cfg.hidden_size),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
