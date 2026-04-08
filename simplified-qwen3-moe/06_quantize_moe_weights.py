"""
06_quantize_moe_weights.py

Offline weight-only blockwise FP8 quantization for MoE expert weights
in Qwen-style (HuggingFace) checkpoints.

Only MoE expert Linear weights are quantized:
  - experts.<id>.gate_proj.weight
  - experts.<id>.up_proj.weight
  - experts.<id>.down_proj.weight

All other tensors (attention, router, embeddings, lm_head, norms) are
left unchanged.

Quantization format: FP8 E4M3FN (or E5M2, selectable).
Blockwise scheme: block size 128 over the in_features (input) dimension.

NOTE on FP8 encoding:
  This script stores quantized values as torch.float8_e4m3fn or
  torch.float8_e5m2 tensors, which are natively supported since
  PyTorch 2.1.  The encoding is therefore EXACT — PyTorch handles
  the round-to-nearest-even and NaN/Inf clamping per the OFP8 spec.
  No bit-level emulation is needed.

Usage:
  python 06_quantize_moe_weights.py \
      --input  ~/models/Qwen/Qwen3-30B-A3B \
      --output ~/models/Qwen/Qwen3-30B-A3B-fp8 \
      [--block-size 128] \
      [--fp8-format e4m3fn] \
      [--verify] \
      [--dry-run] \
      [--test]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# FP8 format constants
# ---------------------------------------------------------------------------
# Max finite magnitude for each supported FP8 format.
# E4M3FN: 448.0   (per OFP8 / IEEE draft spec)
# E5M2:   57344.0
_FP8_QMAX: Dict[str, float] = {
    "e4m3fn": 448.0,
    "e5m2": 57344.0,
}

_FP8_DTYPE: Dict[str, torch.dtype] = {
    "e4m3fn": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class QuantConfig:
    """All knobs for the offline quantization script."""

    input_path: Path
    output_path: Path
    block_size: int = 128
    fp8_format: str = "e4m3fn"  # "e4m3fn" | "e5m2"

    # Regex patterns: a tensor name must match *all* of these to be quantized.
    # Two separate lists allow AND-ing: must match any in name_patterns AND
    # any in proj_patterns.
    name_patterns: List[str] = field(
        default_factory=lambda: [
            r"experts\.\d+\.",          # identifies expert sub-module
        ]
    )
    proj_patterns: List[str] = field(
        default_factory=lambda: [
            r"gate_proj\.weight$",
            r"up_proj\.weight$",
            r"down_proj\.weight$",
        ]
    )

    verify: bool = False   # dequantize and compare after quantization
    dry_run: bool = False  # list matching tensors only, do not write output

    @property
    def fp8_dtype(self) -> torch.dtype:
        if self.fp8_format not in _FP8_DTYPE:
            raise ValueError(
                f"Unsupported FP8 format: {self.fp8_format!r}. "
                f"Choose from {list(_FP8_DTYPE.keys())}"
            )
        return _FP8_DTYPE[self.fp8_format]

    @property
    def qmax(self) -> float:
        return _FP8_QMAX[self.fp8_format]


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    """
    Load a HuggingFace-style checkpoint from *path*.

    Accepts:
      - A single .pt / .bin / .safetensors file.
      - A directory containing one or more such shards.

    For safetensors files the ``safetensors`` package is required at runtime;
    we fall back to torch.load for everything else.
    """
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        # Collect all shard files, sorted for determinism.
        shards = sorted(
            list(path.glob("*.safetensors"))
            + list(path.glob("*.bin"))
            + list(path.glob("*.pt"))
        )
        if not shards:
            raise FileNotFoundError(
                f"No checkpoint shards found in {path}"
            )
        state_dict: Dict[str, torch.Tensor] = {}
        for shard in shards:
            print(f"  Loading shard: {shard.name}")
            state_dict.update(_load_single_file(shard))
        return state_dict
    else:
        return _load_single_file(path)


def _load_single_file(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            return load_file(str(path))
        except ImportError:
            raise ImportError(
                "safetensors package required for .safetensors files. "
                "Install with: pip install safetensors"
            )
    else:
        return torch.load(str(path), map_location="cpu", weights_only=True)


def save_quantized_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    quant_meta: Dict[str, dict],
    output_path: Path,
) -> None:
    """
    Save the (partly-quantized) state_dict and per-tensor metadata.

    Writes two files:
      <output_path>/quantized_weights.pt   – state_dict with FP8 tensors
      <output_path>/quant_metadata.pt      – dict of per-tensor metadata dicts
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path / "quantized_weights.pt")
    torch.save(quant_meta, output_path / "quant_metadata.pt")
    print(f"\nSaved quantized checkpoint → {output_path}")


# ---------------------------------------------------------------------------
# Tensor name matching
# ---------------------------------------------------------------------------

def is_moe_expert_weight(
    name: str,
    tensor: torch.Tensor,
    cfg: QuantConfig,
) -> bool:
    """
    Return True iff *name* identifies an MoE expert projection weight.

    A tensor qualifies when:
      1. Its name matches at least one pattern in cfg.name_patterns, AND
      2. Its name matches at least one pattern in cfg.proj_patterns, AND
      3. The tensor is 2-D (a weight matrix, not a bias or norm vector).
    """
    if tensor.ndim != 2:
        return False
    if not any(re.search(p, name) for p in cfg.name_patterns):
        return False
    if not any(re.search(p, name) for p in cfg.proj_patterns):
        return False
    return True


# ---------------------------------------------------------------------------
# Blockwise helpers
# ---------------------------------------------------------------------------

def blockify_tensor(
    tensor: torch.Tensor,
    block_size: int,
    blocked_dim: int = 1,
) -> Tuple[torch.Tensor, int]:
    """
    Reshape *tensor* so that the *blocked_dim* axis is tiled into blocks of
    *block_size*, padding with zeros if necessary.

    For a weight of shape [out, in] and blocked_dim=1 (in_features):
      padded shape : [out, ceil(in/block_size) * block_size]
      blocked shape: [out, n_blocks, block_size]

    Returns
    -------
    blocked : torch.Tensor
        Shape [..., n_blocks, block_size]
    pad_amount : int
        Number of zeros appended on the blocked dimension.
    """
    # Make contiguous; work in float32 for numerical stability.
    t = tensor.contiguous().float()
    original_size = t.shape[blocked_dim]
    n_blocks = math.ceil(original_size / block_size)
    padded_size = n_blocks * block_size
    pad_amount = padded_size - original_size

    if pad_amount > 0:
        # Build pad tuple for torch.nn.functional.pad (rightmost dim first).
        # We only pad the blocked_dim; all others stay the same.
        ndim = t.ndim
        pad_spec = [0, 0] * ndim
        # torch.nn.functional.pad counts from the last dim backwards.
        dim_from_end = ndim - 1 - blocked_dim
        pad_spec[dim_from_end * 2 + 1] = pad_amount  # pad on the right
        import torch.nn.functional as F
        t = F.pad(t, pad_spec)

    # Move blocked_dim to second-to-last, then split into blocks.
    # For 2-D [out, in_padded] → [out, n_blocks, block_size]
    shape = list(t.shape)
    shape[blocked_dim : blocked_dim + 1] = [n_blocks, block_size]
    blocked = t.reshape(shape)
    return blocked, pad_amount


def compute_block_scales(
    blocked: torch.Tensor,
    qmax: float,
) -> torch.Tensor:
    """
    Compute per-block FP32 scales.

    For each block of 128 values:
      amax  = max(|values|)
      scale = amax / qmax   (or 1.0 if amax == 0)

    *blocked* has shape [..., n_blocks, block_size].
    Returned *scales* has shape [..., n_blocks].
    """
    # amax over the last (block_size) dimension.
    amax = blocked.abs().amax(dim=-1)       # [..., n_blocks]
    # Avoid division by zero for zero blocks.
    safe_amax = amax.clamp(min=1e-30)
    scales = safe_amax / qmax
    # Where amax was exactly 0, use scale = 1.0 (blocks remain zero anyway).
    scales = torch.where(amax == 0, torch.ones_like(scales), scales)
    return scales.float()


# ---------------------------------------------------------------------------
# Quantize / Dequantize
# ---------------------------------------------------------------------------

def quantize_block_to_fp8_codes(
    block: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
    qmax: float,
) -> torch.Tensor:
    """
    Quantize a single block (1-D float32 tensor of length block_size) to FP8.

    Encoding is EXACT: PyTorch's cast to float8_e4m3fn / float8_e5m2
    applies round-to-nearest-even and clamps to the representable range per
    the OFP8 spec.  No manual bit manipulation is required.

    Steps:
      1. Normalize: normalized = block / scale
      2. Clamp to [-qmax, qmax]   (redundant after step 1 but explicit)
      3. Cast to FP8 dtype        (round-to-nearest-even, exact encoding)
    """
    normalized = block / scale.unsqueeze(-1)          # broadcast scale
    normalized = normalized.clamp(-qmax, qmax)
    # Cast: exact per-spec rounding handled by PyTorch.
    return normalized.to(fp8_dtype)


def dequantize_fp8_codes(
    fp8_tensor: torch.Tensor,
    scales: torch.Tensor,
    original_shape: Tuple[int, ...],
    block_size: int,
    blocked_dim: int,
    pad_amount: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Reconstruct an approximate float tensor from FP8-encoded blocks + scales.

    fp8_tensor : shaped [..., n_blocks, block_size]
    scales     : shaped [..., n_blocks]
    """
    # Upcast to float32 for arithmetic.
    f32 = fp8_tensor.float()
    # Multiply each block by its scale.
    dequantized = f32 * scales.unsqueeze(-1)          # [..., n_blocks, block_size]

    # Flatten the n_blocks and block_size dimensions back into the blocked_dim.
    shape = list(dequantized.shape)
    # shape[blocked_dim] == n_blocks, shape[blocked_dim+1] == block_size
    merged_size = shape[blocked_dim] * shape[blocked_dim + 1]
    shape[blocked_dim : blocked_dim + 2] = [merged_size]
    flat = dequantized.reshape(shape)

    # Strip padding.
    if pad_amount > 0:
        slices = [slice(None)] * flat.ndim
        slices[blocked_dim] = slice(0, flat.shape[blocked_dim] - pad_amount)
        flat = flat[tuple(slices)]

    assert tuple(flat.shape) == tuple(original_shape), (
        f"Shape mismatch after dequantization: {flat.shape} vs {original_shape}"
    )
    return flat.to(target_dtype)


# ---------------------------------------------------------------------------
# Main quantization entry point for one tensor
# ---------------------------------------------------------------------------

def quantize_weight_blockwise_fp8(
    name: str,
    tensor: torch.Tensor,
    cfg: QuantConfig,
    blocked_dim: int = 1,
) -> Tuple[torch.Tensor, dict]:
    """
    Blockwise FP8 quantization of a single 2-D weight tensor.

    For shape [out_features, in_features]:
      - Block over the in_features dimension (blocked_dim=1 by default).
      - Produces n_blocks = ceil(in_features / block_size) blocks per row.
      - Total scales tensor shape: [out_features, n_blocks].

    Returns
    -------
    fp8_weight : torch.Tensor
        FP8-encoded weight, shaped [out_features, n_blocks, block_size].
        (The last two dims are the tiled in_features.)
    meta : dict
        Metadata required for dequantization and shape reconstruction.
    """
    original_shape = tuple(tensor.shape)
    original_dtype = tensor.dtype

    blocked, pad_amount = blockify_tensor(tensor, cfg.block_size, blocked_dim)
    # blocked: [out_features, n_blocks, block_size]

    scales = compute_block_scales(blocked, cfg.qmax)
    # scales: [out_features, n_blocks]

    # Quantize all blocks at once using broadcasting.
    normalized = blocked / scales.unsqueeze(-1)
    normalized = normalized.clamp(-cfg.qmax, cfg.qmax)
    fp8_weight = normalized.to(cfg.fp8_dtype)

    padded_shape = tuple(blocked.shape[:-1]) + (blocked.shape[-2] * blocked.shape[-1],)
    # e.g. for [out, n_blocks, block_size] → [out, n_blocks * block_size]

    meta = {
        "tensor_name": name,
        "original_shape": original_shape,
        "padded_shape": padded_shape,
        "logical_shape": original_shape,   # same as original (no logical reshape)
        "block_size": cfg.block_size,
        "blocked_dim": blocked_dim,
        "fp8_format": cfg.fp8_format,
        "original_dtype": str(original_dtype),
        "pad_amount": pad_amount,
        "scales": scales,
    }
    return fp8_weight, meta


def dequantize_weight_blockwise_fp8(
    fp8_weight: torch.Tensor,
    meta: dict,
) -> torch.Tensor:
    """
    Reconstruct a float32 tensor from the FP8 weight + metadata dict produced
    by quantize_weight_blockwise_fp8.
    """
    original_dtype = getattr(torch, meta["original_dtype"].split(".")[-1])
    return dequantize_fp8_codes(
        fp8_tensor=fp8_weight,
        scales=meta["scales"],
        original_shape=meta["original_shape"],
        block_size=meta["block_size"],
        blocked_dim=meta["blocked_dim"],
        pad_amount=meta["pad_amount"],
        target_dtype=original_dtype,
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize_sizes(
    state_dict: Dict[str, torch.Tensor],
    quant_meta: Dict[str, dict],
) -> None:
    """Print a concise quantization summary."""
    n_expert = len(quant_meta)
    total_expert_params = sum(
        math.prod(m["original_shape"]) for m in quant_meta.values()
    )
    total_other_params = sum(
        t.numel()
        for name, t in state_dict.items()
        if name not in quant_meta
    )

    # Bytes before: all expert params in their original dtype (assume BF16 = 2B).
    bytes_before = 0
    bytes_after = 0
    for name, meta in quant_meta.items():
        n = math.prod(meta["original_shape"])
        orig_bytes = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4}.get(
            meta["original_dtype"], 2
        )
        bytes_before += n * orig_bytes
        # FP8 weight: 1 byte per element.
        bytes_after += n * 1
        # Scales: float32, shape [out, n_blocks].
        bytes_after += meta["scales"].numel() * 4

    print("\n" + "=" * 60)
    print("Quantization Summary")
    print("=" * 60)
    print(f"  Expert tensors quantized : {n_expert}")
    print(f"  Expert params quantized  : {total_expert_params:,}")
    print(f"  Other params (unchanged) : {total_other_params:,}")
    print(f"  Expert weight bytes (before): {bytes_before / 1e9:.3f} GB")
    print(f"  Expert weight bytes (after) : {bytes_after / 1e9:.3f} GB")
    print(f"  Compression ratio           : {bytes_before / max(bytes_after,1):.2f}x")
    print("=" * 60)


def verify_quantization(
    original: torch.Tensor,
    fp8_weight: torch.Tensor,
    meta: dict,
) -> None:
    """Dequantize and report per-tensor error statistics."""
    recon = dequantize_weight_blockwise_fp8(fp8_weight, meta).float()
    orig_f32 = original.float()
    diff = (recon - orig_f32).abs()
    rel_l2 = diff.norm() / (orig_f32.norm() + 1e-12)
    print(
        f"    max_abs={diff.max().item():.4e}  "
        f"mean_abs={diff.mean().item():.4e}  "
        f"rel_L2={rel_l2.item():.4e}"
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(cfg: QuantConfig) -> None:
    print(f"Input  : {cfg.input_path}")
    print(f"Output : {cfg.output_path}")
    print(f"FP8    : {cfg.fp8_format}  (qmax={cfg.qmax})")
    print(f"Block  : {cfg.block_size}")

    print("\nLoading checkpoint …")
    state_dict = load_checkpoint(cfg.input_path)
    print(f"  Loaded {len(state_dict)} tensors.")

    # Identify expert tensors.
    expert_names = [
        name
        for name, tensor in state_dict.items()
        if is_moe_expert_weight(name, tensor, cfg)
    ]

    if cfg.dry_run:
        print(f"\n[DRY RUN] Would quantize {len(expert_names)} tensors:")
        for name in expert_names:
            t = state_dict[name]
            print(f"  {name}  {tuple(t.shape)}  {t.dtype}")
        return

    print(f"\nQuantizing {len(expert_names)} expert tensors …")
    quant_meta: Dict[str, dict] = {}

    for name in expert_names:
        tensor = state_dict[name]
        print(f"  {name}  {tuple(tensor.shape)}  {tensor.dtype}", end="")
        fp8_weight, meta = quantize_weight_blockwise_fp8(name, tensor, cfg)

        if cfg.verify:
            verify_quantization(tensor, fp8_weight, meta)
        else:
            print()

        # Replace the BF16 weight with the FP8 tensor.
        state_dict[name] = fp8_weight
        quant_meta[name] = meta

    summarize_sizes(state_dict, quant_meta)

    print("\nSaving …")
    save_quantized_checkpoint(state_dict, quant_meta, cfg.output_path)


# ---------------------------------------------------------------------------
# Built-in self-test / demo
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    """
    Minimal self-test:
      - Creates a fake state_dict with one dense layer and three expert layers.
      - Runs quantization.
      - Asserts only expert tensors were touched.
      - Verifies dequantized shapes match originals.
      - Prints per-tensor error stats.
    """
    print("\n" + "=" * 60)
    print("Running built-in self-test …")
    print("=" * 60)

    torch.manual_seed(42)
    fake_state: Dict[str, torch.Tensor] = {
        # Non-expert: should NOT be quantized.
        "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 512, dtype=torch.bfloat16),
        # Expert projections: SHOULD be quantized.
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(256, 512, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.up_proj.weight":   torch.randn(256, 512, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(512, 256, dtype=torch.bfloat16),
    }

    cfg = QuantConfig(
        input_path=Path("/dev/null"),   # not used in self-test
        output_path=Path("/tmp/fp8_selftest_out"),
        block_size=128,
        fp8_format="e4m3fn",
        verify=True,
        dry_run=False,
    )

    expert_names = [
        n for n, t in fake_state.items()
        if is_moe_expert_weight(n, t, cfg)
    ]

    print(f"\nDetected expert tensors: {len(expert_names)}")
    assert len(expert_names) == 3, f"Expected 3, got {len(expert_names)}"
    assert "model.layers.0.self_attn.q_proj.weight" not in expert_names, \
        "Dense attention weight incorrectly flagged as expert!"

    quant_meta: Dict[str, dict] = {}
    for name in expert_names:
        tensor = fake_state[name]
        fp8_weight, meta = quantize_weight_blockwise_fp8(name, tensor, cfg)
        print(f"\n  {name}  {tuple(tensor.shape)} → fp8 {tuple(fp8_weight.shape)}")
        verify_quantization(tensor, fp8_weight, meta)

        # Check that dequantized shape matches original.
        recon = dequantize_weight_blockwise_fp8(fp8_weight, meta)
        assert recon.shape == tensor.shape, (
            f"Shape mismatch: {recon.shape} vs {tensor.shape}"
        )
        quant_meta[name] = meta
        fake_state[name] = fp8_weight

    # Ensure the dense layer was not touched.
    dense = fake_state["model.layers.0.self_attn.q_proj.weight"]
    assert dense.dtype == torch.bfloat16, "Dense layer dtype was altered!"

    print("\nAll assertions passed.  Self-test OK.")

    # Test non-128-aligned in_features (padding path).
    print("\nTesting padding path (in_features=300, not 128-aligned) …")
    odd_tensor = torch.randn(64, 300, dtype=torch.float32)
    cfg2 = QuantConfig(
        input_path=Path("/dev/null"),
        output_path=Path("/tmp/fp8_selftest_out"),
        block_size=128,
        fp8_format="e4m3fn",
    )
    fp8_odd, meta_odd = quantize_weight_blockwise_fp8("test.weight", odd_tensor, cfg2)
    assert meta_odd["pad_amount"] == 300 % 128 and meta_odd["pad_amount"] != 0 or \
           meta_odd["pad_amount"] == (128 - 300 % 128) % 128, \
           f"Unexpected pad_amount: {meta_odd['pad_amount']}"
    recon_odd = dequantize_weight_blockwise_fp8(fp8_odd, meta_odd)
    assert recon_odd.shape == (64, 300), f"Bad shape after deq: {recon_odd.shape}"
    print(f"  pad_amount={meta_odd['pad_amount']}  recon shape={recon_odd.shape}  OK")

    print("\nSelf-test complete.\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline blockwise FP8 quantization of MoE expert weights."
    )
    p.add_argument("--input",  type=Path, default=None, help="Input checkpoint path")
    p.add_argument("--output", type=Path, default=None, help="Output directory")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument(
        "--fp8-format", choices=["e4m3fn", "e5m2"], default="e4m3fn"
    )
    p.add_argument(
        "--verify", action="store_true",
        help="Dequantize after quantization and report error stats."
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="List matching tensor names and shapes; do not write output."
    )
    p.add_argument(
        "--test", action="store_true",
        help="Run built-in self-test and exit."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.test:
        run_self_test()
        sys.exit(0)

    if args.input is None:
        print("Error: --input is required (unless --test is specified).")
        sys.exit(1)
    if args.output is None and not args.dry_run:
        print("Error: --output is required (unless --dry-run is specified).")
        sys.exit(1)

    cfg = QuantConfig(
        input_path=args.input.expanduser(),
        output_path=(args.output or Path("/tmp/fp8_quant_out")).expanduser(),
        block_size=args.block_size,
        fp8_format=args.fp8_format,
        verify=args.verify,
        dry_run=args.dry_run,
    )
    main(cfg)
