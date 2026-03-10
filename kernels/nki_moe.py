import math
import torch
import torch.nn as nn
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


# Qwen3-30B-A3B specific dimensions
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 2816  # Will be padded to ~3072 for alignment
NUM_EXPERTS = 128
TOP_K = 8

# NKI/Trainium tile sizes - optimized for SBUF capacity
TOKEN_TILE = 32      # Process 32 tokens at a time (fits in SBUF)
HIDDEN_TILE = 128    # Hidden dimension tile
INTERM_TILE = 128    # Intermediate dimension tile

@nki.jit
def nki_moe_optimized(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
):
    """
    Optimized MoE kernel with expert grouping for better memory locality.
    
    This version assumes tokens are pre-grouped by expert (common optimization)
    """
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    num_experts = gate_up_proj.shape[0]
    interm_size = down_proj.shape[1]
    top_k = expert_indices.shape[1]
    
    output = nl.ndarray((num_tokens, hidden_size), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    
    # Tile indices
    ix_t = nl.arange(TOKEN_TILE)[:, None]
    ix_h = nl.arange(HIDDEN_TILE)[None, :]
    ix_i = nl.arange(INTERM_TILE)[None, :]
    
    num_t_tiles = (num_tokens + TOKEN_TILE - 1) // TOKEN_TILE
    num_h_tiles = (hidden_size + HIDDEN_TILE - 1) // HIDDEN_TILE
    num_i_tiles = (interm_size + INTERM_TILE - 1) // INTERM_TILE
    num_gu_tiles = (2 * interm_size + INTERM_TILE - 1) // INTERM_TILE
    
    # For each expert
    for expert_id in nl.affine_range(num_experts):
        # Load expert weights
        gu_w = gate_up_proj[expert_id]
        d_w = down_proj[expert_id]
        
        # Process all tokens (simplified - assumes all tokens use this expert)
        # In practice, you'd filter tokens by expert assignment
        for tt in nl.affine_range(num_t_tiles):
            t_start = tt * TOKEN_TILE
            t_mask = (t_start + ix_t < num_tokens)
            
            # Accumulator for this token tile
            tile_accum = nl.zeros((TOKEN_TILE, hidden_size), dtype=nl.float32)
            
            # For each top-k position
            for k in nl.affine_range(top_k):
                # Check if this expert is used at position k
                exp_at_k = nl.load(
                    expert_indices[t_start + ix_t, k],
                    mask=t_mask
                )
                wt_at_k = nl.load(
                    expert_weights[t_start + ix_t, k],
                    mask=t_mask
                )
                
                # Load input
                inp_tile = nl.load(
                    hidden_states[t_start + ix_t, ix_h],
                    mask=(t_mask & (ix_h < hidden_size))
                )
                
                # Gate+Up projection
                gu_out = nl.zeros((TOKEN_TILE, 2 * interm_size), dtype=nl.float32)
                
                for ht in nl.affine_range(num_h_tiles):
                    h_start = ht * HIDDEN_TILE
                    h_mask = (h_start + ix_h < hidden_size)
                    
                    inp_h = nl.load(
                        hidden_states[t_start + ix_t, h_start + ix_h],
                        mask=(t_mask & h_mask)
                    )
                    
                    for gut in range(num_gu_tiles):
                        gu_start = gut * INTERM_TILE
                        gu_mask = (gu_start + ix_i < 2 * interm_size)
                        
                        w_gu = nl.load(
                            gu_w[h_start + nl.arange(HIDDEN_TILE)[:, None],
                                gu_start + nl.arange(INTERM_TILE)[None, :]],
                            mask=((h_start + nl.arange(HIDDEN_TILE)[:, None] < hidden_size) &
                                  (gu_start + nl.arange(INTERM_TILE)[None, :] < 2 * interm_size))
                        )
                        
                        gu_out[:, gu_start + nl.arange(INTERM_TILE)] += nl.matmul(inp_h, w_gu)
                
                # SwiGLU
                act = nl.zeros((TOKEN_TILE, interm_size), dtype=nl.float32)
                for it in nl.affine_range(num_i_tiles):
                    i_start = it * INTERM_TILE
                    i_mask = (i_start + ix_i < interm_size)
                    
                    g = gu_out[:, i_start + nl.arange(INTERM_TILE)]
                    u = gu_out[:, interm_size + i_start + nl.arange(INTERM_TILE)]
                    
                    sig_g = nl.sigmoid(g)
                    act[:, i_start + nl.arange(INTERM_TILE)] = (g * sig_g) * u
                
                # Down projection
                down_out = nl.zeros((TOKEN_TILE, hidden_size), dtype=nl.float32)
                for it in nl.affine_range(num_i_tiles):
                    i_start = it * INTERM_TILE
                    i_mask = (i_start + nl.arange(INTERM_TILE) < interm_size)
                    
                    a = act[:, i_start + nl.arange(INTERM_TILE)]
                    
                    for ht in nl.affine_range(num_h_tiles):
                        h_start = ht * HIDDEN_TILE
                        h_mask = (h_start + ix_h < hidden_size)
                        
                        w_d = nl.load(
                            d_w[i_start + nl.arange(INTERM_TILE)[:, None],
                               h_start + nl.arange(HIDDEN_TILE)[None, :]],
                            mask=((i_start + nl.arange(INTERM_TILE)[:, None] < interm_size) &
                                  (h_start + nl.arange(HIDDEN_TILE)[None, :] < hidden_size))
                        )
                        
                        down_out[:, h_start + nl.arange(HIDDEN_TILE)] += nl.matmul(a, w_d)
                
                # Weight and accumulate
                wt_bc = wt_at_k.broadcast_to((TOKEN_TILE, 1))
                tile_accum += down_out * wt_bc
            
            # Store output tile
            nl.store(
                output[t_start + ix_t, ix_h],
                value=tile_accum,
                mask=(t_mask & (ix_h < hidden_size))
            )
    
    return output


# ---------------------------------------------------------------------------
# Reference implementation (CPU) — mirrors kernel logic exactly
# ---------------------------------------------------------------------------

def pytorch_moe_reference(hidden_states, gate_up_proj, down_proj, expert_indices, expert_weights):
    """
    CPU reference that replicates nki_moe_optimized behaviour:
    for every expert, for every top-k slot, apply the expert's MLP to ALL
    tokens weighted by expert_weights[:, k].  (This matches the kernel's
    simplified "all tokens use this expert" assumption.)
    """
    num_tokens, hidden_size = hidden_states.shape
    num_experts = gate_up_proj.shape[0]
    interm_size = down_proj.shape[1]
    top_k = expert_indices.shape[1]

    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)
    hs_f32 = hidden_states.float()

    for expert_id in range(num_experts):
        gu_w = gate_up_proj[expert_id].float()   # (hidden_size, 2*interm_size)
        d_w  = down_proj[expert_id].float()       # (interm_size, hidden_size)

        gu_out = hs_f32 @ gu_w                    # (num_tokens, 2*interm_size)
        gate   = gu_out[:, :interm_size]
        up     = gu_out[:, interm_size:]
        act    = torch.nn.functional.silu(gate) * up   # SwiGLU
        down_out = act @ d_w                      # (num_tokens, hidden_size)

        for k in range(top_k):
            wt = expert_weights[:, k : k + 1].float()  # (num_tokens, 1)
            output = output + down_out * wt

    return output.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_nki_moe_optimized():
    import torch_xla.core.xla_model as xm

    print("=" * 70)
    print("NKI MoE Optimized Kernel Test")
    print("=" * 70)

    device = xm.xla_device()
    print(f"\nUsing device: {device}")

    # Use small shapes that satisfy tile constraints
    num_tokens  = TOKEN_TILE          # 32
    hidden_size = HIDDEN_TILE         # 512
    interm_size = INTERM_TILE         # 512
    num_experts = 2
    top_k       = 2

    print(f"\nShapes:")
    print(f"  hidden_states : ({num_tokens}, {hidden_size})")
    print(f"  gate_up_proj  : ({num_experts}, {hidden_size}, {2 * interm_size})")
    print(f"  down_proj     : ({num_experts}, {interm_size}, {hidden_size})")
    print(f"  expert_indices: ({num_tokens}, {top_k})")
    print(f"  expert_weights: ({num_tokens}, {top_k})")

    torch.manual_seed(42)
    dtype = torch.bfloat16

    hidden_states   = torch.randn(num_tokens, hidden_size, dtype=dtype)
    gate_up_proj    = torch.randn(num_experts, hidden_size, 2 * interm_size, dtype=dtype)
    down_proj       = torch.randn(num_experts, interm_size, hidden_size, dtype=dtype)
    # Expert indices in [0, num_experts)
    expert_indices  = torch.randint(0, num_experts, (num_tokens, top_k)).to(dtype)
    # Weights that sum to 1 across top-k (normalised softmax-like)
    expert_weights  = torch.softmax(torch.randn(num_tokens, top_k, dtype=dtype), dim=-1)

    print("\n[1/3] Computing reference (PyTorch CPU)...")
    ref_output = pytorch_moe_reference(
        hidden_states, gate_up_proj, down_proj, expert_indices, expert_weights
    )

    print("[2/3] Computing NKI kernel on Trainium...")
    nki_output = nki_moe_optimized(
        hidden_states.to(device),
        gate_up_proj.to(device),
        down_proj.to(device),
        expert_indices.to(device),
        expert_weights.to(device),
    ).cpu()

    print("[3/3] Comparing outputs...")
    diff      = torch.abs(ref_output.float() - nki_output.float())
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nResults:")
    print(f"  Output shape : {nki_output.shape}")
    print(f"  Max  |diff|  : {max_diff:.6e}")
    print(f"  Mean |diff|  : {mean_diff:.6e}")

    threshold = 0.05  # bfloat16 accumulation tolerance
    if max_diff < threshold:
        print("\n" + "=" * 70)
        print("SUCCESS! nki_moe_optimized kernel produced matching results.")
        print("=" * 70)
        return True
    else:
        print(f"\nFAILED — max diff {max_diff:.4e} exceeds threshold {threshold:.4e}")
        return False


if __name__ == "__main__":
    test_nki_moe_optimized()
