"""
L=1 integration test: multi-layer fused TKG megakernel vs qwen.py reference.

Goal
----
Exercise `transformer_qwen3_moe_tkg_multilayer` at num_layers=1 and compare its
outputs against a `NeuronQwen3MoeDecoderLayer` (imported from qwen.py). The two
models are compiled separately via `build_module`, fed the same inputs and the
same initial weights, then compared on hidden state and post-scatter KV caches.

Design
------
* `RefModule`    — wraps `NeuronQwen3MoeDecoderLayer`; standard NxDI TKG path.
* `KernelModule` — holds the same decoder layer only to carry weights; extracts
  and tile-transposes Wq/Wk/Wv, then calls `_build_multilayer_kernel(1)`.

Both modules receive the same `decoder_state_dict` at construction so they start
from identical weights despite being compiled into separate NEFF files.

Tolerance
---------
The reference uses NxDI's standard TKG attention + standard MoE kernels; the
megakernel uses v14d attention + v30c MoE. Both are bf16 with different fusion
strategies, so element-wise agreement is bf16-noise limited: atol=1e-2,
rtol=1e-2.

Run
---
    python tests/test_megakernel_L1_vs_qwen.py
"""

import os
import sys
import tempfile

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch.nn as nn

import nki

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.testing import build_module

from qwen import NeuronQwen3MoeDecoderLayer, Qwen3MoeInferenceConfig
from kernels.transformer.transformer_qwen_multilayer import _build_multilayer_kernel, get_multilayer_kernel_jit


# ---------------------------------------------------------------------------
# Test config — Qwen3-30B-A3B shapes, TP=4 / LNC=2
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/ubuntu/models/Qwen3-30B-A3B/"
NUM_LAYERS = 1
TP = 4
LNC = 2
B, S_TKG = 1, 1
S_CTX = 640  # KV cache length, multiple of PMAX=128

# Sweep positions that exercise KV-cache tile edges
POSITIONS = [0, 1, 127, 128, 129, 255, 320]

COMPILER_ARGS = (
    "--enable-saturate-infinity "
    "--enable-mixed-precision-accumulation "
    "--model-type transformer "
    f"--lnc={LNC} "
    "-O1 "
    "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-hlo2tensorizer-options='--verify-hlo=true' "
    "--internal-max-instruction-limit=15000000 "
    "--internal-backend-options=--enable-verifier=false"
)


def tile_transpose(W: torch.Tensor, n_heads: int, d: int, n_tiles: int) -> torch.Tensor:
    """Mirror of qwen_with_nki.py's Wq/Wk/Wv pretranspose (v14d access pattern)."""
    return (
        W.reshape(n_heads, d, n_tiles, d)
        .permute(0, 3, 2, 1)
        .reshape(n_heads * d, n_tiles * d)
        .contiguous()
    )


def make_inference_config(num_layers: int = NUM_LAYERS) -> Qwen3MoeInferenceConfig:
    """Construct a Qwen3MoeInferenceConfig truncated to `num_layers` decoder layers."""
    neuron_config = MoENeuronConfig(
        tp_degree=TP,
        batch_size=1,
        max_context_length=640,
        seq_len=S_CTX,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=LNC,
        torch_dtype=torch.bfloat16,
        fused_qkv=False,
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
        mlp_kernel_enabled=False,
        attn_tkg_nki_kernel_enabled=False,
        attn_block_tkg_nki_kernel_enabled=False,
    )
    cfg = Qwen3MoeInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )
    cfg.num_hidden_layers = num_layers
    return cfg


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------

class RefModule(nn.Module):
    """Standard NxDI TKG path via NeuronQwen3MoeDecoderLayer."""

    def __init__(self, config: Qwen3MoeInferenceConfig):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        torch.manual_seed(0)
        self.decoder_layer = NeuronQwen3MoeDecoderLayer(config, layer_idx=0).to(dtype)
        self.register_buffer("K_cache", torch.zeros(B, 1, S_CTX, config.head_dim, dtype=dtype))
        self.register_buffer("V_cache", torch.zeros(B, 1, S_CTX, config.head_dim, dtype=dtype))

    def forward(self, hidden_in, position_ids, attention_mask):
        out, kv, _, _, _ = self.decoder_layer(
            hidden_states=hidden_in,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=(self.K_cache, self.V_cache),
        )
        return out, kv[0], kv[1]


# ---------------------------------------------------------------------------
# Megakernel model
# ---------------------------------------------------------------------------

class KernelModule(nn.Module):
    """L=1 megakernel path; same seed as RefModule ensures identical weights."""

    def __init__(self, config: Qwen3MoeInferenceConfig):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        self._d = config.head_dim
        self._H = config.hidden_size
        torch.manual_seed(0)
        self.decoder_layer = NeuronQwen3MoeDecoderLayer(config, layer_idx=0).to(dtype)
        self.register_buffer("K_cache", torch.zeros(B, 1, S_CTX, config.head_dim, dtype=dtype))
        self.register_buffer("V_cache", torch.zeros(B, 1, S_CTX, config.head_dim, dtype=dtype))
        self._replica_groups = (list(range(TP)),)
        self._kernel_jit = get_multilayer_kernel_jit(NUM_LAYERS)

    def forward(self, hidden_in, position_ids, attention_mask):
        d, H = self._d, self._H
        nh_tiles = H // d
        dlay = self.decoder_layer

        cos_cache, sin_cache = dlay.self_attn.rotary_emb(hidden_in, position_ids)
        cos_at_pos = cos_cache.squeeze(1)
        sin_at_pos = sin_cache.squeeze(1)

        Wq_full = dlay.self_attn.qkv_proj.q_proj.weight   # [Hq_tp*d, H]
        Wk_full = dlay.self_attn.qkv_proj.k_proj.weight   # [Hkv_tp*d, H]
        Wv_full = dlay.self_attn.qkv_proj.v_proj.weight
        Wo_full = dlay.self_attn.o_proj.o_proj.weight      # [H, Hq_tp*d]

        Hq_tp  = Wq_full.shape[0] // d
        Hkv_tp = Wk_full.shape[0] // d
        Wq_tt = tile_transpose(Wq_full, Hq_tp,  d, nh_tiles)
        Wk_tt = tile_transpose(Wk_full, Hkv_tp, d, nh_tiles)
        Wv_tt = tile_transpose(Wv_full, Hkv_tp, d, nh_tiles)
        Wo_k  = Wo_full.T.contiguous()                     # [Hq_tp*d, H]

        qn     = dlay.self_attn.q_layernorm.weight
        kn     = dlay.self_attn.k_layernorm.weight
        gpre   = dlay.input_layernorm.weight
        gpost  = dlay.post_attention_layernorm.weight.unsqueeze(0)   # [1, H]
        router = dlay.mlp.router.linear_router.weight.T.contiguous() # [H, E]
        gate_up = dlay.mlp.expert_mlps.mlp_op.gate_up_proj.weight    # [E, H, 2*I_tp]
        down    = dlay.mlp.expert_mlps.mlp_op.down_proj.weight        # [E, I_tp, H]

        out = self._kernel_jit[LNC](
            hidden_in,
            Wq_tt, Wk_tt, Wv_tt, Wo_k,
            qn, kn, gpre, gpost,
            router, gate_up, down,
            self.K_cache, self.V_cache,
            cos_at_pos, sin_at_pos, position_ids.to(torch.int32),
            replica_groups=self._replica_groups,
        )
        Y_kern, K_out, V_out = out[0], out[1], out[2]
        return Y_kern, K_out, V_out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_attention_mask(pos: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Causal mask over S_CTX: positions > pos are -inf (not attended)."""
    m = torch.zeros(B, 1, 1, S_CTX, dtype=dtype)
    m[:, :, :, pos + 1:] = float("-inf")
    return m


def run_one(ref_traced, kern_traced, pos: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    hidden = (torch.randn(B, S_TKG, 2048, generator=g, dtype=torch.float32) * 0.1).bfloat16()
    pos_t = torch.tensor([[pos]], dtype=torch.int32)
    mask = make_attention_mask(pos)

    ref_out, K_ref, V_ref = ref_traced(hidden, pos_t, mask)
    Y_kern, K_kern, V_kern = kern_traced(hidden, pos_t, mask)

    r = ref_out.float()
    k = Y_kern.float()
    hidden_max  = (r - k).abs().max().item()
    hidden_mean = (r - k).abs().mean().item()

    # New K/V row at position `pos` — isolates attention scatter correctness
    kn_ref = K_ref[:, 0, pos, :].float()
    vn_ref = V_ref[:, 0, pos, :].float()
    kn_k   = K_kern[:, 0, pos, :].float()
    vn_k   = V_kern[:, 0, pos, :].float()
    kn_max = (kn_ref - kn_k).abs().max().item()
    vn_max = (vn_ref - vn_k).abs().max().item()

    # All positions except `pos` should still be zero
    K_mask = torch.ones(S_CTX, dtype=torch.bool); K_mask[pos] = False
    V_mask = torch.ones(S_CTX, dtype=torch.bool); V_mask[pos] = False
    k_other_ref = K_ref[0, 0, K_mask].float().abs().max().item()
    v_other_ref = V_ref[0, 0, V_mask].float().abs().max().item()
    k_other_k   = K_kern[0, 0, K_mask].float().abs().max().item()
    v_other_k   = V_kern[0, 0, V_mask].float().abs().max().item()

    atol, rtol = 1e-2, 1e-2
    pass_hidden    = torch.allclose(r, k, atol=atol, rtol=rtol)
    pass_k         = torch.allclose(kn_ref, kn_k, atol=atol, rtol=rtol)
    pass_v         = torch.allclose(vn_ref, vn_k, atol=atol, rtol=rtol)
    pass_untouched = all(x < 1e-6 for x in (k_other_ref, v_other_ref, k_other_k, v_other_k))

    return dict(
        pos=pos,
        hidden_max=hidden_max, hidden_mean=hidden_mean,
        kn_max=kn_max, vn_max=vn_max,
        k_other_ref=k_other_ref, v_other_ref=v_other_ref,
        k_other_k=k_other_k,     v_other_k=v_other_k,
        pass_hidden=pass_hidden, pass_k=pass_k, pass_v=pass_v,
        pass_untouched=pass_untouched,
    )


def _fix_rank_util(path):
    """Patch rank_util.rank from zeros([1]) to arange([TP]) so shard_children can split it."""
    ckpt = torch.load(path)
    ckpt["decoder_layer.self_attn.rank_util.rank"] = torch.arange(0, TP, dtype=torch.int32)
    return ckpt


def main():
    print("Building Qwen3MoeInferenceConfig (num_hidden_layers=1)...", flush=True)
    cfg = make_inference_config(NUM_LAYERS)
    H = cfg.hidden_size
    d = cfg.head_dim
    print(f"  H={H} d={d} E={cfg.num_experts} tp={TP} lnc={LNC}")

    example_hidden = torch.zeros(B, S_TKG, H, dtype=torch.bfloat16)
    example_pos    = torch.zeros(B, 1, dtype=torch.int32)
    example_mask   = make_attention_mask(0)
    example_inputs = [(example_hidden, example_pos, example_mask)]

    results = []
    with tempfile.TemporaryDirectory(prefix="megakernel_l1_test_") as workdir:
        print(f"\nCompiling RefModule @ TP={TP} LNC={LNC}...", flush=True)
        ref_traced = build_module(
            module_cls=RefModule,
            example_inputs=example_inputs,
            module_init_kwargs={"config": cfg},
            tp_degree=TP,
            logical_nc_config=LNC,
            compiler_args=COMPILER_ARGS,
            compiler_workdir=os.path.join(workdir, "ref_compiler_workdir"),
            checkpoint_path=os.path.join(workdir, "ref_checkpoint.pt"),
            checkpoint_loader_fn=_fix_rank_util,
        )
        print("RefModule compile PASS\n", flush=True)

        print(f"Compiling KernelModule @ TP={TP} LNC={LNC}...", flush=True)
        kern_traced = build_module(
            module_cls=KernelModule,
            example_inputs=example_inputs,
            module_init_kwargs={"config": cfg},
            tp_degree=TP,
            logical_nc_config=LNC,
            compiler_args=COMPILER_ARGS,
            compiler_workdir=os.path.join(workdir, "kern_compiler_workdir"),
            checkpoint_path=os.path.join(workdir, "kern_checkpoint.pt"),
            checkpoint_loader_fn=_fix_rank_util,
        )
        print("KernelModule compile PASS\n", flush=True)

        print("Running position sweep...\n", flush=True)
        for i, pos in enumerate(POSITIONS):
            print(f"=== position_ids={pos} ===", flush=True)
            r = run_one(ref_traced, kern_traced, pos, seed=100 + i)
            results.append(r)
            ok = r["pass_hidden"] and r["pass_k"] and r["pass_v"] and r["pass_untouched"]
            verdict = "PASS" if ok else "FAIL"
            print(
                f"  hidden_max={r['hidden_max']:.3e}  k_new_max={r['kn_max']:.3e}  "
                f"v_new_max={r['vn_max']:.3e}  untouched_ok={r['pass_untouched']}  {verdict}",
                flush=True,
            )

    print("\n" + "=" * 72)
    print("  SUMMARY — megakernel(L=1) vs NeuronQwen3MoeDecoderLayer")
    print("=" * 72)
    print(f"{'pos':>5} {'hidden_max':>12} {'kn_max':>10} {'vn_max':>10}  verdict")
    fail = 0
    for r in results:
        ok = r["pass_hidden"] and r["pass_k"] and r["pass_v"] and r["pass_untouched"]
        if not ok:
            fail += 1
        print(
            f"{r['pos']:>5} {r['hidden_max']:>12.3e} {r['kn_max']:>10.3e} "
            f"{r['vn_max']:>10.3e}  {'PASS' if ok else 'FAIL'}"
        )
    n = len(results)
    print(f"\n{n - fail}/{n} positions PASS")
    print(f"OVERALL: {'PASS' if fail == 0 else 'FAIL'}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
