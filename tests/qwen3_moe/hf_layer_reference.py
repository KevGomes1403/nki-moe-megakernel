#!/usr/bin/env python
"""Golden per-layer hidden-state reference for Qwen3-30B-A3B.

Produces a [48, seq, H] tensor of the hidden state LEAVING each decoder layer,
to be diffed against on-device NKI megakernel dumps of `layer_resid_all`
(shape [48, B, S, H] = [48, 1, 1, 2048] per decode step).

----------------------------------------------------------------------------
HF source verification
----------------------------------------------------------------------------
Checked: transformers/models/qwen3_moe/modeling_qwen3_moe.py  (transformers 4.57.6)

  - Decoder layer class: `Qwen3MoeDecoderLayer` (modeling_qwen3_moe.py:288).
  - `Qwen3MoeDecoderLayer.forward` (line 306) is annotated `-> torch.FloatTensor`
    and ends with:

        residual = hidden_states                       # post-attention residual
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):            # MoE block returns
            hidden_states, _ = hidden_states            # (hidden, router_logits)
        hidden_states = residual + hidden_states        # MoE residual add
        return hidden_states                            # line 365

    So in transformers 4.57.6 the layer returns a BARE TENSOR (not a tuple),
    and that tensor is exactly the residual stream AFTER layer k's MoE
    residual add and BEFORE the model-level `model.norm`. That is precisely
    what the megakernel dumps as `layer_resid_all[k]`.

  - Older transformers releases returned a tuple `(hidden_states, ...)` whose
    element 0 is the same post-residual tensor. The hook below handles both:
    if the layer output is a tuple/list it takes element 0, otherwise it uses
    the tensor directly. This keeps the script correct across versions.

  - `Qwen3MoeModel.forward` (line 443) loops over `self.layers` feeding each
    layer's output straight into the next, then applies `self.norm` once at
    the end -- confirming the per-layer output IS the inter-layer residual
    stream.

  - Post-attention residual: in `Qwen3MoeDecoderLayer.forward` the sequence is

        residual = hidden_states                        # line 340
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(...)
        hidden_states = residual + hidden_states         # line 354
        residual = hidden_states                         # line 357  <-- this
        hidden_states = self.post_attention_layernorm(hidden_states)  # line 358
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states          # line 363
        return hidden_states                              # line 365

    The post-attention residual (line 354 / 357) is EXACTLY the input tensor
    to `post_attention_layernorm`. We capture it with a forward PRE-hook on
    each `model.model.layers[k].post_attention_layernorm`: a pre-hook fires
    with `(module, args)` and `args[0]` is that input tensor, shape
    `[1, seq, H]`. Verified against transformers 4.57.6 source above.
----------------------------------------------------------------------------
"""

import argparse
import json
import sys

import torch


def parse_tokens(path: str) -> list[int]:
    """Load a token-id sequence from a JSON list or whitespace-separated ints."""
    with open(path, "r") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"token file {path!r} is empty")
    # Try JSON first (handles "[1, 2, 3]" and nested "[[...]]").
    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
            data = data[0]  # unwrap a batch dimension
        tokens = [int(x) for x in data]
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fall back to whitespace/comma-separated ints.
        tokens = [int(x) for x in text.replace(",", " ").split()]
    if not tokens:
        raise ValueError(f"no tokens parsed from {path!r}")
    return tokens


def capture_layer_hidden_states(model, input_ids: torch.Tensor):
    """Run one teacher-forced forward pass, return per-layer reference tensors.

    A single forward over the full sequence is equivalent to autoregressive
    decoding for the captured hidden states: causal attention means position
    t only attends to positions <= t, so the layer output at position t is
    identical to what a decode step at position t would produce.

    Returns a tuple `(layer_hidden, layer_attn_resid)`, each
    `[num_layers, seq, H]` fp32:
      - layer_hidden     : post-MoE decoder-layer OUTPUT (the inter-layer
                           residual stream).
      - layer_attn_resid : post-ATTENTION residual, i.e. the input tensor to
                           `post_attention_layernorm` (line 357/358 of HF
                           `Qwen3MoeDecoderLayer.forward`).
    """
    layers = model.model.layers
    num_layers = len(layers)
    captured: list[torch.Tensor | None] = [None] * num_layers
    captured_resid: list[torch.Tensor | None] = [None] * num_layers

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            # Qwen3MoeDecoderLayer returns a bare tensor in transformers 4.57.6;
            # older versions return a tuple whose element 0 is the same tensor.
            hs = output[0] if isinstance(output, (tuple, list)) else output
            # Capture in native dtype, detach, cast to fp32 for the reference.
            captured[idx] = hs.detach().to(torch.float32).cpu()
        return hook

    def make_resid_pre_hook(idx: int):
        # A forward pre-hook fires as `hook(module, args)`; `args[0]` is the
        # input tensor to post_attention_layernorm == the post-attention
        # residual.
        def pre_hook(_module, args):
            resid = args[0]
            captured_resid[idx] = resid.detach().to(torch.float32).cpu()
        return pre_hook

    handles = [layers[k].register_forward_hook(make_hook(k)) for k in range(num_layers)]
    handles += [
        layers[k].post_attention_layernorm.register_forward_pre_hook(
            make_resid_pre_hook(k)
        )
        for k in range(num_layers)
    ]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    missing = [k for k, v in enumerate(captured) if v is None]
    if missing:
        raise RuntimeError(f"layer-output hooks did not fire for layers: {missing}")
    missing_resid = [k for k, v in enumerate(captured_resid) if v is None]
    if missing_resid:
        raise RuntimeError(
            f"post-attention residual pre-hooks did not fire for layers: {missing_resid}"
        )

    def _stack(tensors, name):
        # Each captured tensor is [1, seq, H]; squeeze the batch dim then stack.
        per_layer = []
        for k, hs in enumerate(tensors):
            if hs.dim() != 3 or hs.shape[0] != 1:
                raise RuntimeError(
                    f"layer {k} {name} has unexpected shape {tuple(hs.shape)}, "
                    f"expected [1, seq, H]"
                )
            per_layer.append(hs[0])  # [seq, H]
        return torch.stack(per_layer, dim=0)  # [num_layers, seq, H]

    layer_hidden = _stack(captured, "hidden state")
    layer_attn_resid = _stack(captured_resid, "post-attention residual")
    return layer_hidden, layer_attn_resid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tokens",
        required=True,
        help="path to token-id file (JSON list or whitespace/comma-separated ints): "
        "prompt tokens followed by generated tokens",
    )
    ap.add_argument(
        "--out",
        default="/tmp/hf_layer_ref.pt",
        help="output .pt path (default: /tmp/hf_layer_ref.pt)",
    )
    ap.add_argument(
        "--model-path",
        default="/home/ubuntu/Qwen3-30B-A3B/hf_model",
        help="path to HF model weights",
    )
    args = ap.parse_args()

    from transformers import Qwen3MoeForCausalLM

    tokens = parse_tokens(args.tokens)
    print(f"loaded {len(tokens)} token ids from {args.tokens}")

    print(f"loading model from {args.model_path} (bfloat16, CPU) ...")
    model = Qwen3MoeForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    model.eval()

    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    print(f"model: num_layers={num_layers}, hidden_size={hidden_size}")

    input_ids = torch.tensor([tokens], dtype=torch.long)  # [1, seq]
    print(f"running teacher-forced forward over [1, {input_ids.shape[1]}] ...")
    layer_hidden, layer_attn_resid = capture_layer_hidden_states(model, input_ids)
    print(f"captured layer_hidden:     {tuple(layer_hidden.shape)} dtype={layer_hidden.dtype}")
    print(f"captured layer_attn_resid: {tuple(layer_attn_resid.shape)} dtype={layer_attn_resid.dtype}")

    ref = {
        "layer_hidden": layer_hidden,  # [num_layers, seq, H] fp32 (post-MoE output)
        "layer_attn_resid": layer_attn_resid,  # [num_layers, seq, H] fp32 (post-attn residual)
        "tokens": tokens,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
    }
    torch.save(ref, args.out)
    print(f"saved reference to {args.out}")

    # Sanity print: norm of each layer's hidden state at the LAST position.
    print("\nper-layer hidden-state norm at last position:")
    for k in range(num_layers):
        n = layer_hidden[k, -1, :].norm().item()
        print(f"  layer {k:2d}: ||h[-1]|| = {n:.6f}")

    return 0


def _smoke_test() -> int:
    """Tiny correctness check of the hook logic on a random small model.

    Does NOT touch the 30B weights. Builds a 2-layer Qwen3MoeConfig with
    hidden_size 64 and confirms hooks capture [1, seq, 64] per layer and the
    stacked result is [2, seq, 64].
    """
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).eval()

    seq = 5
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq), dtype=torch.long)

    layer_hidden, layer_attn_resid = capture_layer_hidden_states(model, input_ids)

    assert layer_hidden.shape == (2, seq, 64), (
        f"expected stack [2, {seq}, 64], got {tuple(layer_hidden.shape)}"
    )
    assert layer_hidden.dtype == torch.float32, (
        f"expected fp32 reference, got {layer_hidden.dtype}"
    )
    assert layer_attn_resid.shape == (2, seq, 64), (
        f"expected attn-resid stack [2, {seq}, 64], got {tuple(layer_attn_resid.shape)}"
    )
    assert layer_attn_resid.dtype == torch.float32, (
        f"expected fp32 attn-resid reference, got {layer_attn_resid.dtype}"
    )

    # Re-run with probe hooks to confirm per-layer shape and return kind for
    # both the layer-output hook and the post-attention residual pre-hook.
    probe = {}

    def probe_hook(_m, _i, output):
        probe["is_tuple"] = isinstance(output, (tuple, list))
        hs = output[0] if probe["is_tuple"] else output
        probe["shape"] = tuple(hs.shape)

    def probe_pre_hook(_m, args):
        probe["resid_shape"] = tuple(args[0].shape)

    h = model.model.layers[0].register_forward_hook(probe_hook)
    hp = model.model.layers[0].post_attention_layernorm.register_forward_pre_hook(
        probe_pre_hook
    )
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    h.remove()
    hp.remove()

    assert probe["shape"] == (1, seq, 64), (
        f"expected per-layer hidden [1, {seq}, 64], got {probe['shape']}"
    )
    # Post-attention capture must be [1, seq, hidden] per layer.
    assert probe["resid_shape"] == (1, seq, 64), (
        f"expected post-attention residual [1, {seq}, 64], got {probe['resid_shape']}"
    )

    print("smoke test PASSED")
    print(f"  per-layer hidden shape    : {probe['shape']}")
    print(f"  post-attn residual shape  : {probe['resid_shape']}")
    print(f"  layer returns tuple       : {probe['is_tuple']}")
    print(f"  stacked layer_hidden      : {tuple(layer_hidden.shape)} ({layer_hidden.dtype})")
    print(f"  stacked layer_attn_resid  : {tuple(layer_attn_resid.shape)} ({layer_attn_resid.dtype})")
    return 0


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        sys.exit(_smoke_test())
    sys.exit(main())
