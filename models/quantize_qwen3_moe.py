"""
quantize_qwen3_moe.py

FP8 (e4m3) MoE expert quantization for Qwen3 MoE.

MoE experts (llmcompressor oneshot):
  Weights:     per-output-channel FP8, calibrated via llmcompressor.
               Stored as <name>.weight (float8_e4m3fn) and <name>.weight_scale (float32).
  Activations: dynamic per-token FP8 at inference time (no offline calibration needed).
  Scale clamp:  ±448 → ±240 (Neuron hardware saturates outside ±240 for fp8 e4m3).

Attention weights are NOT quantized (no fp8 attention kernel in nxdi).

Delta checkpoint written to output_path (fp8 expert weights + float32 scales only).

Usage:
    python quantize_qwen3_moe.py \\
        --model-path  /path/to/Qwen3-MoE \\
        --output-path /path/to/quantized
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROMPTS_FILE = Path(__file__).parent / "prompts.txt"


def _load_default_prompts() -> List[str]:
    """Load calibration prompts from prompts.txt, split on blank lines."""
    if not _PROMPTS_FILE.exists():
        return []
    text = _PROMPTS_FILE.read_text()
    return [p.strip() for p in text.split("\n\n") if p.strip()]

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

BLOCK_H: int = 128   # attention weight block height (output-channel axis)
BLOCK_W: int = 128   # attention weight block width  (input-channel axis)

# Neuron hardware clips fp8 e4m3 values outside ±240 to NaN.
# llmcompressor defaults to ±448 — monkey-patch it down to FP8_MAX.
FP8_MAX: float = 240.0

ATTN_PROJ_NAMES: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


# ── Phase 1 helpers ───────────────────────────────────────────────────────────

def _block_quantize_weight(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block-wise symmetric FP8 quantization for a 2-D weight matrix.

    Each [BLOCK_H × BLOCK_W] tile gets its own scale so that
        weight ≈ fp8_weight * scale_inv
    where scale_inv[i, j] is the per-block float32 scale.

    Returns
    -------
    fp8_weight   : torch.float8_e4m3fn, shape (out_features, in_features)
    scale_inv    : torch.float32,       shape (n_blocks_h, n_blocks_w)

    Caller must ensure out_features % BLOCK_H == 0 and
    in_features % BLOCK_W == 0 (true for all standard Qwen3-MoE shapes).
    """
    out_features, in_features = weight.shape
    assert out_features % BLOCK_H == 0, (
        f"out_features={out_features} not divisible by BLOCK_H={BLOCK_H}"
    )
    assert in_features % BLOCK_W == 0, (
        f"in_features={in_features} not divisible by BLOCK_W={BLOCK_W}"
    )

    n_bh = out_features // BLOCK_H
    n_bw = in_features  // BLOCK_W

    # Reshape into (n_bh, BLOCK_H, n_bw, BLOCK_W) to get per-block max-abs.
    w = weight.float()
    blocks = w.reshape(n_bh, BLOCK_H, n_bw, BLOCK_W)

    # Per-block max absolute value, clamped away from zero for numerical safety.
    max_abs = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)  # (n_bh, n_bw)

    # scale_inv such that  fp8_val * scale_inv == original_val  (to dequant).
    # We divide by FP8_MAX (240) instead of the fp8 type max (448) so that
    # the stored fp8 values never exceed what Neuron hardware can represent.
    scale_inv = max_abs / FP8_MAX  # (n_bh, n_bw)

    # Expand scale for element-wise division: (n_bh, BLOCK_H, n_bw, BLOCK_W)
    scale_exp = scale_inv.unsqueeze(1).unsqueeze(3).expand_as(blocks)

    quantized = (blocks / scale_exp).clamp(-FP8_MAX, FP8_MAX)
    fp8_weight = quantized.reshape(out_features, in_features).to(torch.float8_e4m3fn)

    return fp8_weight, scale_inv.to(torch.float32)


def quantize_attention_weights(
    model: torch.nn.Module,
    num_hidden_layers: int,
) -> Dict[str, torch.Tensor]:
    """
    Block-wise FP8 quantization for all attention projection weights.

    Returns a dict of new/replacement tensors keyed by their HF state-dict name
    (i.e., with the 'model.' prefix).  Each projection contributes:
      model.layers.{l}.self_attn.{proj}.weight           → fp8
      model.layers.{l}.self_attn.{proj}.weight_scale_inv → float32 (n_bh, n_bw)
    """
    out: Dict[str, torch.Tensor] = {}
    named_modules = dict(model.named_modules())

    for l in range(num_hidden_layers):
        for proj in ATTN_PROJ_NAMES:
            key = f"model.layers.{l}.self_attn.{proj}"
            w = named_modules[key].weight.detach().float()
            fp8_w, scale_inv = _block_quantize_weight(w)

            out[f"{key}.weight"] = fp8_w
            # _scale_inv suffix is the convention read by maybe_dequantize_layer.
            out[f"{key}.weight_scale_inv"] = scale_inv

        gc.collect()

    return out


# ── Activation calibration ────────────────────────────────────────────────────

class _MaxAbsHook:
    """Accumulates max-abs activation value across calibration batches."""

    def __init__(self) -> None:
        self.max_abs: float = 0.0
        self._handle = None

    def attach(self, module: torch.nn.Module) -> None:
        def _hook(mod, inp, out):
            x = inp[0].detach().float()
            self.max_abs = max(self.max_abs, x.abs().max().item())

        self._handle = module.register_forward_hook(_hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def as_scale(self) -> torch.Tensor:
        """Return input_scale = max_abs / FP8_MAX as a scalar float32 tensor."""
        return torch.tensor(max(self.max_abs, 1e-12) / FP8_MAX, dtype=torch.float32)


def calibrate_attention_activations(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    num_hidden_layers: int,
) -> Dict[str, torch.Tensor]:
    """
    Run calibration prompts through the model and compute a static per-module
    input scale for every attention projection linear layer.

    Returns a dict:
      model.layers.{l}.self_attn.{proj}.input_scale → float32 scalar tensor

    These are NOT consumed by maybe_dequantize_layer; they are stored for the
    NxD quantized attention kernel (quantized / quantized_mlp_kernel_enabled).
    """
    named_modules = dict(model.named_modules())
    hooks: Dict[str, _MaxAbsHook] = {}

    for l in range(num_hidden_layers):
        for proj in ATTN_PROJ_NAMES:
            name = f"model.layers.{l}.self_attn.{proj}"
            hook = _MaxAbsHook()
            hook.attach(named_modules[name])
            hooks[name] = hook

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            model(**inputs)
            gc.collect()

    out: Dict[str, torch.Tensor] = {}
    for name, hook in hooks.items():
        hook.remove()
        out[f"{name}.input_scale"] = hook.as_scale()

    return out


# ── Phase 2: llmcompressor MoE quantization ───────────────────────────────────

def _build_llmcompressor_ignore(num_hidden_layers: int) -> List[str]:
    """
    Modules to exclude from llmcompressor quantization.
    Attention projections are handled in Phase 1; router/norm/lm_head stay bf16.
    """
    ignore = ["lm_head", "model.embed_tokens"]
    for l in range(num_hidden_layers):
        for proj in ATTN_PROJ_NAMES:
            ignore.append(f"model.layers.{l}.self_attn.{proj}")
        # Router must stay high precision — NxD forces router dtype=float32.
        ignore.append(f"model.layers.{l}.mlp.gate")
        # Norm layers are not Linear so llmcompressor won't touch them anyway,
        # but list them explicitly to be safe.
        ignore.append(f"model.layers.{l}.input_layernorm")
        ignore.append(f"model.layers.{l}.post_attention_layernorm")
    return ignore


def quantize_moe_with_llmcompressor(
    model_path: str,
    num_hidden_layers: int,
    output_path: str,
    calibration_prompts: Optional[List[str]] = None,
) -> None:
    """
    Apply FP8 e4m3 quantization to MoE expert linear layers using llmcompressor
    oneshot, then write a delta checkpoint containing only the fp8 expert weights
    and float32 scales.

    Uses llmcompressor's native save_compressed=True to produce genuine fp8_e4m3fn
    weight tensors — no manual quantization.  Calibration prompts are forwarded to
    oneshot for per-expert scale calibration (moe_calibrate_all_experts=True).

    Expert layers targeted:
        model.layers.{l}.mlp.experts.{e}.{gate_proj, up_proj, down_proj}

    Scale range is monkey-patched from llmcompressor's default ±448 to ±240 for
    Neuron hardware compatibility.
    """
    import shutil
    import tempfile
    import compressed_tensors.quantization.utils.helpers as _helpers
    import compressed_tensors.offload.dispatch as _dispatch
    import psutil
    from compressed_tensors.quantization.quant_args import QuantizationType
    from llmcompressor import oneshot
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM

    ignore = _build_llmcompressor_ignore(num_hidden_layers)

    recipe = f"""
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: {ignore}
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: channel
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: token
                        dynamic: true
                        symmetric: true
                    targets: ["Linear"]
"""

    # Monkey-patch 1: clamp fp8 scale range from ±448 → ±240 for Neuron hardware.
    _orig_range = _helpers.calculate_range

    def _patched_calculate_range(*args, **kwargs):
        q_min, q_max = _orig_range(*args, **kwargs)
        if (
            hasattr(args[0], "type")
            and args[0].type == QuantizationType.FLOAT
            and args[0].num_bits == 8
        ):
            device = args[1] if len(args) > 1 else "cpu"
            return torch.tensor(-FP8_MAX, device=device), torch.tensor(FP8_MAX, device=device)
        return q_min, q_max

    # Monkey-patch 2: no CUDA on Trainium — fall back to CPU for dispatch_model.
    _orig_device_memory = _dispatch.get_device_memory

    def _patched_get_device_memory():
        result = _orig_device_memory()
        if not result:
            result = {torch.device("cpu"): psutil.virtual_memory().available}
        return result

    _helpers.calculate_range = _patched_calculate_range
    _dispatch.get_device_memory = _patched_get_device_memory

    tmp_compressed = tempfile.mkdtemp(prefix="qwen3_fp8_compressed_")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")

        # oneshot expects a HuggingFace Dataset, not a plain list.
        from datasets import Dataset as HFDataset
        hf_dataset = HFDataset.from_dict({"text": calibration_prompts or []})

        # oneshot with save_compressed=True writes fp8 weights natively via
        # compressed_tensors — no manual quantization required.
        oneshot(
            model=model,
            recipe=recipe,
            output_dir=tmp_compressed,
            save_compressed=True,
            dataset=hf_dataset,
            moe_calibrate_all_experts=True,
        )

        # Load the compressed safetensors and extract only the expert delta.
        # llmcompressor may save to a single file or multiple shards.
        compressed_files = sorted(Path(tmp_compressed).glob("*.safetensors"))
        if not compressed_files:
            raise RuntimeError(f"No safetensors found in {tmp_compressed}")

        full_sd: Dict[str, torch.Tensor] = {}
        for f in compressed_files:
            full_sd.update(load_file(str(f)))

        # Filter: keep only expert fp8 weights and float32 weight_scale tensors.
        EXPERT_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")
        delta: Dict[str, torch.Tensor] = {}
        for key, tensor in full_sd.items():
            # Must belong to an expert projection (not router, attention, etc.)
            if not any(f".{p}." in key or key.endswith(f".{p}") for p in
                       [f"{n}.weight" for n in EXPERT_PROJ_NAMES] +
                       [f"{n}.weight_scale" for n in EXPERT_PROJ_NAMES]):
                continue
            if "mlp.experts." not in key:
                continue
            # Ensure scales are float32.
            if key.endswith("weight_scale"):
                tensor = tensor.to(torch.float32)
            delta[key] = tensor.contiguous()

        print(f"  Saving delta checkpoint ({len(delta)} tensors) → {output_path}/model.safetensors")
        output_path_p = Path(output_path)
        output_path_p.mkdir(parents=True, exist_ok=True)
        save_file(delta, str(output_path_p / "model.safetensors"))

    finally:
        _helpers.calculate_range = _orig_range
        _dispatch.get_device_memory = _orig_device_memory
        shutil.rmtree(tmp_compressed, ignore_errors=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def quantize(
    model_path: str,
    output_path: str,
    calibration_prompts: Optional[List[str]] = None,
) -> None:
    if calibration_prompts is None:
        calibration_prompts = _load_default_prompts()
    print("=" * 60)
    print("Qwen3-MoE FP8 (e4m3) Quantization")
    print("=" * 60)

    base_config = AutoConfig.from_pretrained(model_path)
    num_hidden_layers: int = base_config.num_hidden_layers
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ── MoE experts ───────────────────────────────────────────────────────────
    print("\n[MoE] llmcompressor FP8 (per-channel weights, dynamic activations)")
    quantize_moe_with_llmcompressor(
        model_path=model_path,
        num_hidden_layers=num_hidden_layers,
        output_path=output_path,
        calibration_prompts=calibration_prompts,
    )
    gc.collect()

    # Write minimal config.json with quantization metadata.
    config = AutoConfig.from_pretrained(model_path)
    config_dict = config.to_dict()
    config_dict["quantization_config"] = {
        "quant_type": "fp8_e4m3",
        "fp8_max": FP8_MAX,
        "moe_experts": {
            "scheme": "llmcompressor_fp8",
            "weight_strategy": "per_output_channel",
            "activation_strategy": "dynamic_per_token",
        },
    }
    config_path = Path(output_path) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Saved config.json → {config_path}")

    tokenizer.save_pretrained(output_path)
    print(f"\nDone.  Delta checkpoint saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FP8 e4m3 MoE quantization for Qwen3 MoE (llmcompressor per-channel)"
    )
    parser.add_argument("--model-path", required=True, help="HuggingFace Qwen3-MoE model path")
    parser.add_argument("--output-path", required=True, help="Output path for quantized checkpoint")
    parser.add_argument(
        "--calibration-prompts",
        nargs="*",
        default=None,
        help="Optional calibration prompts for per-expert scale estimation",
    )
    args = parser.parse_args()

    quantize(
        model_path=args.model_path,
        output_path=args.output_path,
        calibration_prompts=args.calibration_prompts,
    )
