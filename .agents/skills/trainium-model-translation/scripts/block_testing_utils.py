"""
Minimal parameterizable testing utility for validating NxDI blocks against PyTorch references.

This module provides a single function to test any NeuronX Distributed Inference (NxDI) block
against its PyTorch reference implementation by synchronizing weights and comparing outputs.
"""

import sys
from pathlib import Path
from typing import Type, Dict, Optional, Any

import torch

# Add project root for imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Add NKI project root
NKI_ROOT = ROOT_DIR / "NKI"
if str(NKI_ROOT) not in sys.path:
    sys.path.append(str(NKI_ROOT))

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig, MoENeuronConfig  # type: ignore
from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy  # type: ignore

# Artifacts directory
_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _create_default_config(
    batch_size: int = 2,
    seq_len: int = 128,
    hidden_size: int = 64,
    head_dim: int = 16,
    num_attention_heads: int = 8,
    num_key_value_heads: int = 2,
    tp_degree: int = 1,
    torch_dtype: torch.dtype = torch.bfloat16,
    use_moe: bool = False,
) -> InferenceConfig:
    """
    Create a default InferenceConfig for testing.

    These defaults match the example_usage_block_testing.py configuration
    and work well for small-scale correctness testing.

    Args:
        batch_size: Batch size for testing (default: 2)
        seq_len: Sequence length for testing (default: 128)
        hidden_size: Hidden dimension size (default: 64)
        head_dim: Dimension per attention head (default: 16)
        num_attention_heads: Number of attention heads (default: 8)
        num_key_value_heads: Number of KV heads for GQA (default: 2)
        tp_degree: Tensor parallel degree (default: 1)
        torch_dtype: Data type for tensors (default: torch.bfloat16)
        use_moe: Use MoENeuronConfig for MoE models (default: False)

    Returns:
        InferenceConfig with sensible defaults for testing
    """
    # Use MoENeuronConfig for MoE models, otherwise use base NeuronConfig
    config_class = MoENeuronConfig if use_moe else NeuronConfig

    neuron_config = config_class(
        batch_size=batch_size,
        seq_len=seq_len,
        tp_degree=tp_degree,
        torch_dtype=torch_dtype,
        on_cpu=True,  # Prevent TKG module initialization
        fused_qkv=True,  # Use fused QKV projection
    )
    
    # Base config parameters
    config_params = {
        'neuron_config': neuron_config,
        'hidden_size': hidden_size,
        'head_dim': head_dim,
        'num_attention_heads': num_attention_heads,
        'num_key_value_heads': num_key_value_heads,
        'sliding_window': 32,
        'initial_context_length': 4096,
        'rope_theta': 10000.0,
        'rope_scaling_factor': 1.0,
        'rope_ntk_alpha': 1.0,
        'rope_ntk_beta': 32.0,
        'max_position_embeddings': 4096,
        'attention_bias': True,
    }

    # Add MoE-specific parameters if requested
    if use_moe:
        config_params.update({
            'num_experts': 8,
            'num_local_experts': 8,
            'num_experts_per_tok': 2,
            'n_shared_experts': 0,
            'intermediate_size': 256,
            'swiglu_limit': None,
            "hybrid_sharding_config": None,
        })

    config = InferenceConfig(**config_params)

    # Required for some implementations
    config.num_cores_per_group = 1

    return config


class _BlockWrapper(torch.nn.Module):
    """
    Generic wrapper for XLA tracing compatibility.
    
    Extracts tensor outputs from dataclasses/tuples that blocks may return.
    This is required because XLA tracing needs pure tensor outputs.
    """
    def __init__(self, block_class: Type[torch.nn.Module], **init_kwargs):
        super().__init__()
        self.block = block_class(**init_kwargs)
    
    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        
        # Extract tensor from common output formats
        if hasattr(output, 'hidden_states'):
            return output.hidden_states
        elif isinstance(output, (tuple, list)):
            return output[0]
        return output


def _sync_weights(
    reference_block: torch.nn.Module,
    checkpoint_path: Path,
    weight_mapping: Dict[str, str],
    verbose: bool = True,
) -> int:
    """
    Synchronize weights from PyTorch reference to Neuron checkpoint.
    
    Args:
        reference_block: PyTorch reference module
        checkpoint_path: Path to Neuron checkpoint file
        weight_mapping: Dict mapping PyTorch keys to Neuron checkpoint keys
                       (will auto-prepend "block." prefix from wrapper)
        verbose: Whether to print detailed logging
    
    Returns:
        Number of weights successfully synced
    """
    neuron_state_dict = torch.load(checkpoint_path)
    ref_state_dict = reference_block.state_dict()

    if verbose:
        print("\n" + "=" * 80)
        print("WEIGHT SYNCHRONIZATION")
        print("=" * 80)
        print("\nNeuron checkpoint keys:")
        for key in sorted(neuron_state_dict.keys()):
            print(f"  {key}: {neuron_state_dict[key].shape}")
        print("\nPyTorch reference keys:")
        for key in sorted(ref_state_dict.keys()):
            print(f"  {key}: {ref_state_dict[key].shape}")
        print("=" * 80 + "\n")

    updated_count = 0
    for ref_key, neuron_key in weight_mapping.items():
        # Add "block." prefix from wrapper if not already present
        if not neuron_key.startswith("block."):
            neuron_key = f"block.{neuron_key}"
        if ref_key not in ref_state_dict:
            if verbose:
                print(f"  Warning: {ref_key} not found in PyTorch reference")
            continue
        if neuron_key not in neuron_state_dict:
            if verbose:
                print(f"  Warning: {neuron_key} not found in Neuron checkpoint")
            continue

        ref_tensor = ref_state_dict[ref_key]

        # Convert dtype if necessary
        if ref_tensor.dtype != neuron_state_dict[neuron_key].dtype:
            ref_tensor = ref_tensor.to(dtype=neuron_state_dict[neuron_key].dtype)

        # Clone to ensure clean memory layout
        ref_tensor = ref_tensor.contiguous().clone()

        # Skip if shapes are incompatible
        if ref_tensor.shape != neuron_state_dict[neuron_key].shape:
            if verbose:
                print(f"  Warning: Shape mismatch for {ref_key} -> {neuron_key}")
                print(f"    PyTorch: {ref_tensor.shape}, Neuron: {neuron_state_dict[neuron_key].shape}")
            continue

        neuron_state_dict[neuron_key] = ref_tensor
        updated_count += 1
        if verbose:
            print(f"  ✓ Synced {ref_key} -> {neuron_key} (shape: {ref_tensor.shape})")

    torch.save(neuron_state_dict, checkpoint_path)
    if verbose:
        print(f"\nTotal weights synced: {updated_count}\n")
    return updated_count


def test_block_correctness(
    neuron_block_class: Type[torch.nn.Module],
    pytorch_block_class: Type[torch.nn.Module],
    weight_mapping: Dict[str, str],
    config: Optional[InferenceConfig] = None,
    neuron_init_kwargs: Optional[Dict[str, Any]] = None,
    pytorch_init_kwargs: Optional[Dict[str, Any]] = None,
    example_inputs: Optional[list] = None,
    test_inputs: Optional[list] = None,
    reference_inputs: Optional[list] = None,
    checkpoint_name: str = "test_block.pt",
    seed: int = 42,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    hidden_size: Optional[int] = None,
    use_moe: bool = False,
    verbose: bool = True,
    sync_weights_fn: Optional[callable] = None,
):
    """
    Test that a Neuron block matches its PyTorch reference implementation.

    This function:
    1. Compiles the Neuron block to generate an initial checkpoint
    2. Creates a PyTorch reference block with random weights (using seed)
    3. Synchronizes weights from PyTorch to Neuron checkpoint
    4. Recompiles the Neuron block with synced weights
    5. Runs test inputs through both blocks and validates accuracy

    Args:
        neuron_block_class: NxDI block class (e.g., NeuronAttentionBlock)
        pytorch_block_class: PyTorch reference class (e.g., AttentionBlock)
        weight_mapping: Dict mapping PyTorch state_dict keys to Neuron checkpoint keys
        config: InferenceConfig with model and neuron configuration.
                If None, uses default config (batch=2, seq=128, hidden=64, heads=8, etc.).
                If provided, merges with default config - only overrides matching keys,
                ensuring required parameters are never accidentally omitted.
        neuron_init_kwargs: Optional kwargs for neuron block __init__
        pytorch_init_kwargs: Optional kwargs for pytorch block __init__
        example_inputs: List of input tuples for Neuron block XLA compilation (can use zeros)
                       Example: [(hidden_states, mask, position_ids)]
        test_inputs: List of input tuples for Neuron block testing (should use random data)
                    Example: [(sample, mask, position_ids)]
        reference_inputs: List of input tuples for PyTorch reference execution
                         Example: [(sample,)] - must match reference block's signature
        checkpoint_name: Name for checkpoint file in artifacts directory
        seed: Random seed for reproducible weight initialization
        batch_size: Override batch size for testing (overrides config value if provided)
        seq_len: Override sequence length for testing (overrides config value if provided)
        hidden_size: Override hidden size for testing (overrides config value if provided)
        use_moe: Use MoENeuronConfig for MoE models (default: False)
        verbose: Whether to print detailed logging
        sync_weights_fn: Optional custom function for syncing weights. Must have signature:
                        (reference_block, checkpoint_path, weight_mapping, verbose) -> int
                        If None, uses default _sync_weights implementation.

    Raises:
        AssertionError: If outputs don't match within tolerance
        ValueError: If example_inputs, test_inputs, or reference_inputs are not provided

    Examples:
        # Attention block usage
        bs, sl, hs = 2, 128, 64
        position_ids = torch.arange(sl).unsqueeze(0).expand(bs, -1)
        mask = torch.ones(bs, 1, sl, sl)
        sample = torch.rand(bs, sl, hs, dtype=torch.bfloat16)

        # Wrap PyTorch reference to handle batching if needed
        class BatchedAttentionBlock(torch.nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block
            def forward(self, hidden_states):
                outputs = [self.block(hidden_states[b]) for b in range(hidden_states.shape[0])]
                return torch.stack(outputs)

        test_block_correctness(
            neuron_block_class=NeuronAttentionBlock,
            pytorch_block_class=BatchedAttentionBlock,
            weight_mapping={"qkv.weight": "qkv_proj.Wqkv.weight", ...},
            example_inputs=[(torch.zeros(bs, sl, hs, dtype=torch.bfloat16), mask, position_ids)],
            test_inputs=[(sample, mask, position_ids)],
            reference_inputs=[(sample,)],  # Batched reference expects 3D input
            neuron_init_kwargs={"layer_idx": 1},
            pytorch_init_kwargs={"block": original_block},
        )

        # MoE block usage (simpler - same signature for both)
        test_block_correctness(
            neuron_block_class=NeuronMLPBlock,
            pytorch_block_class=MLPBlock,
            weight_mapping={...},
            example_inputs=[(torch.zeros(bs, sl, hs, dtype=torch.bfloat16),)],
            test_inputs=[(sample,)],
            reference_inputs=[(sample,)],
            use_moe=True,
        )

        # Custom sync_weights function for complex weight transformations
        def custom_sync_with_reshape(reference_block, checkpoint_path, weight_mapping, verbose):
            neuron_state_dict = torch.load(checkpoint_path)
            ref_state_dict = reference_block.state_dict()
            count = 0
            for ref_key, neuron_key in weight_mapping.items():
                if not neuron_key.startswith("block."):
                    neuron_key = f"block.{neuron_key}"
                ref_tensor = ref_state_dict[ref_key].to(neuron_state_dict[neuron_key].dtype)
                ref_converted = ref_tensor.contiguous()

                # Apply custom transformations based on weight name
                if 'mlp1_weight' in ref_key:
                    # mlp1_weight: (E, I*2, H) -> gate_up_proj: (E, H, I*2)
                    ref_converted = ref_converted.transpose(1, 2).contiguous()

                if ref_converted.shape != neuron_state_dict[neuron_key].shape:
                    if verbose:
                        print(f"  Shape mismatch: {ref_key} -> {neuron_key}")
                    continue

                neuron_state_dict[neuron_key] = ref_converted
                count += 1
            torch.save(neuron_state_dict, checkpoint_path)
            return count

        test_block_correctness(
            ...,
            sync_weights_fn=custom_sync_with_reshape,
        )

    Notes:
        - A default config is always created first with batch_size=2, seq_len=128, hidden_size=64
        - If a custom config is provided, its non-None values are merged into the default config
        - This ensures required parameters are never accidentally omitted from the config
        - Pass config to neuron blocks via neuron_init_kwargs if they require it
        - Users must provide inputs that match each block's signature
        - If PyTorch reference has different input signature than Neuron block, wrap it accordingly
    """
    # Validate inputs are provided
    if example_inputs is None or test_inputs is None or reference_inputs is None:
        raise ValueError("example_inputs, test_inputs, and reference_inputs must all be provided")

    # Always start with default config, then override with custom config values
    if verbose:
        print("Creating default config...")
    default_config = _create_default_config(
        batch_size=batch_size or 2,
        seq_len=seq_len or 128,
        hidden_size=hidden_size or 64,
        use_moe=use_moe,
    )

    if config is not None:
        if verbose:
            print("Merging custom config into defaults...")
        # Override default config values with user-provided config values
        # Handle NeuronConfig attributes
        if hasattr(config, 'neuron_config') and config.neuron_config is not None:
            user_neuron_config = config.neuron_config
            default_neuron_config = default_config.neuron_config
            for attr in dir(user_neuron_config):
                if not attr.startswith('_') and hasattr(default_neuron_config, attr):
                    user_val = getattr(user_neuron_config, attr)
                    # Only override if user explicitly set a non-None value
                    if user_val is not None and not callable(user_val):
                        try:
                            setattr(default_neuron_config, attr, user_val)
                        except AttributeError:
                            pass  # Skip read-only attributes

        # Handle top-level InferenceConfig attributes
        for attr in dir(config):
            if not attr.startswith('_') and attr != 'neuron_config' and hasattr(default_config, attr):
                user_val = getattr(config, attr)
                # Only override if user explicitly set a non-None value
                if user_val is not None and not callable(user_val):
                    try:
                        setattr(default_config, attr, user_val)
                    except AttributeError:
                        pass  # Skip read-only attributes

        config = default_config
    else:
        config = default_config
    
    # Auto-inject config into neuron_init_kwargs if 'config' key not present
    neuron_kwargs = neuron_init_kwargs or {}
    if 'config' not in neuron_kwargs:
        neuron_kwargs = {'config': config, **neuron_kwargs}
    
    # Setup paths
    checkpoint_path = _ARTIFACTS_DIR / checkpoint_name
    temp_checkpoint_path = _ARTIFACTS_DIR / f"temp_{checkpoint_name}"
    
    # Clean up any old artifacts
    for path in (checkpoint_path, temp_checkpoint_path):
        if path.exists():
            path.unlink()
    
    # Get dimensions (with parameter overrides)
    bs = batch_size if batch_size is not None else config.neuron_config.batch_size
    sl = seq_len if seq_len is not None else config.neuron_config.seq_len
    hs = hidden_size if hidden_size is not None else config.hidden_size
    dtype = config.neuron_config.torch_dtype
    
    # Use user-provided example_inputs directly
    
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"TESTING BLOCK CORRECTNESS: {neuron_block_class.__name__} vs {pytorch_block_class.__name__}")
        print(f"{'=' * 80}")
        print(f"Dimensions: batch_size={bs}, seq_len={sl}, hidden_size={hs}")
        print(f"Example inputs: {len(example_inputs)} tuple(s)")
        print(f"Test inputs: {len(test_inputs)} tuple(s)")
        print(f"Reference inputs: {len(reference_inputs)} tuple(s)")
        print(f"Checkpoint: {checkpoint_name}")
        print(f"{'=' * 80}\n")
    
    # Step 1: Initial compilation to generate checkpoint
    if verbose:
        print(f"Step 1: Initial Neuron compilation...")
    
    # Use wrapper for XLA tracing compatibility
    temp_neuron = build_module(
        _BlockWrapper,
        example_inputs,
        tp_degree=config.neuron_config.tp_degree,
        module_init_kwargs={
            "block_class": neuron_block_class,
            **neuron_kwargs,
        },
        checkpoint_path=str(temp_checkpoint_path),
    )
    del temp_neuron  # Free memory before recompiling
    
    # Step 2: Create PyTorch reference block with random weights
    if verbose:
        print(f"Step 2: Creating PyTorch reference with seed={seed}...")
    torch.manual_seed(seed)
    pytorch_kwargs = pytorch_init_kwargs or {}
    reference_block = pytorch_block_class(**pytorch_kwargs)

    # Apply sensible weight initialization
    torch.manual_seed(seed)  # Reset seed for consistent initialization
    for name, param in reference_block.named_parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.02)

    reference_block = reference_block.to(dtype=torch.bfloat16)
    
    # Step 3: Sync weights from PyTorch to Neuron checkpoint
    if verbose:
        print(f"Step 3: Syncing weights...")

    # Use custom sync function if provided, otherwise use default
    sync_fn = sync_weights_fn if sync_weights_fn is not None else _sync_weights
    synced_count = sync_fn(
        reference_block=reference_block,
        checkpoint_path=temp_checkpoint_path,
        weight_mapping=weight_mapping,
        verbose=verbose,
    )
    
    if synced_count == 0:
        print("WARNING: No weights were synced! Check weight_mapping.")
    
    # Step 4: Save final checkpoint
    if verbose:
        print(f"Step 4: Saving final checkpoint...")
    final_checkpoint = torch.load(temp_checkpoint_path)
    torch.save(final_checkpoint, checkpoint_path)
    
    # Step 5: Recompile Neuron module with synced weights
    if verbose:
        print(f"Step 5: Recompiling Neuron module with synced weights...")
    
    # Use wrapper for XLA tracing compatibility
    neuron_block = build_module(
        _BlockWrapper,
        example_inputs,
        tp_degree=config.neuron_config.tp_degree,
        module_init_kwargs={
            "block_class": neuron_block_class,
            **neuron_kwargs,
        },
        checkpoint_path=str(checkpoint_path),
    )
    
    # Clean up temp checkpoint
    if temp_checkpoint_path.exists():
        temp_checkpoint_path.unlink()
    
    # Step 6: Run inference and validate accuracy
    if verbose:
        print(f"\nStep 6: Running inference and validating accuracy...")
    
    # Run PyTorch reference
    reference_block.eval()
    with torch.no_grad():
        expected_output = reference_block(reference_inputs[0][0])
    
    if verbose:
        print(f"  Expected output shape: {expected_output.shape}")
        print(f"  Expected output mean: {expected_output.float().mean().item():.6f}")
    
    # Step 7: Validate accuracy
    validate_accuracy(neuron_block, test_inputs, expected_outputs=[expected_output])
    
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"✓ Test PASSED: {neuron_block_class.__name__} matches {pytorch_block_class.__name__}")
        print(f"{'=' * 80}\n")


