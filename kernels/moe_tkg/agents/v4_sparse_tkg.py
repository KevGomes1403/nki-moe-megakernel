"""
Optimization Plan 4: Sparse K-Expert Execution (TKG-style) — Optimized

Core insight: Rather than iterating over all E experts (with most having zero
routing weight), only process the K active experts per token.  This eliminates
the (E - K) / E fraction of wasted compute and HBM reads.

Performance optimizations over baseline v4:
  1. Double-buffered gate+up weight DMA (ping-pong across gu_t tiles within h_t)
  2. PSUM tile reuse — pre-allocate outside K loop, memset per iteration
  3. Double-buffered down weight DMA (ping-pong across i_t tiles)
  4. Aggressive nl.affine_range for all independent iterations
  5. SBUF buffer reuse across K iterations (gu_tiles, act_bf16_tiles, rw_scalar,
     down_sb, scaled, out_cast buffers allocated once, reused per K)

Constraints (inherited):
    T <= 128
    H % 128 == 0
    I % 128 == 0
    K must fit in SBUF partition dimension (K <= 128)
"""

import os
import time
import torch
import nki
import nki.language as nl
import nki.isa as nisa

os.environ["NEURON_CC_FLAGS"] = " "
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"


# ---------------------------------------------------------------------------
# Optimised sparse kernel
# ---------------------------------------------------------------------------

@nki.jit(platform_target="trn2")
def nki_moe_v4(
    hidden_states,       # [T, H]    bf16  (T must be 1 for this kernel)
    gate_up_weights,     # [E, H, 2*I] bf16
    down_weights,        # [E, I, H]   bf16
    expert_indices,      # [T, K]    int32  — which K experts are active
    routing_weights_k,   # [T, K]    float32 — non-zero affinities only
):
    """
    Sparse MoE TKG: process only K active experts per token.
    Specialized for T=1 (TKG decode case) to avoid SBUF partition issues.

    Performance optimizations:
      - Double-buffered weight DMA for gate+up and down projections
      - PSUM tile reuse across K iterations (memset instead of re-allocate)
      - Aggressive nl.affine_range for independent iterations
      - SBUF buffer reuse across K iterations

    Tile layout:
      Gate+Up intermediate : [P=128, 1]  — partition=gu_dim
      Down output / accum  : [1,   P=128] — partition=token_dim (always partition 0)
    """
    T      = hidden_states.shape[0]            # must be 1
    H      = hidden_states.shape[1]
    E      = gate_up_weights.shape[0]          # noqa: F841
    two_I  = gate_up_weights.shape[2]
    I_size = two_I // 2
    K      = expert_indices.shape[1]
    P      = 128

    num_h_tiles  = H      // P
    num_i_tiles  = I_size // P
    num_gu_tiles = two_I  // P

    # Output tensor in HBM
    output = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # =========================================================================
    # Hoist hidden states: load [1, H] from HBM once, store as per-tile [P, 1]
    # =========================================================================
    h_tiles_sb = []
    for h_t in nl.affine_range(num_h_tiles):
        h_off = h_t * P
        h_sb  = nl.ndarray((P, 1), dtype=hidden_states.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=h_sb,
            src=hidden_states.ap(
                pattern=[[1, P], [H, 1]],
                offset=h_off,
            ),
        )
        h_tiles_sb.append(h_sb)

    # =========================================================================
    # Load expert_indices into SBUF: [P_tk, K] with T=1 rows meaningful
    # =========================================================================
    P_tk  = 128
    expert_idx_sb = nl.ndarray((P_tk, K), dtype=expert_indices.dtype, buffer=nl.sbuf)
    nisa.memset(dst=expert_idx_sb, value=0)
    nisa.dma_copy(
        dst=expert_idx_sb[0:T, 0:K],
        src=expert_indices[0:T, 0:K],
    )

    # =========================================================================
    # Prefetch routing_weights_k: [P_rw, K] SBUF with T=1 rows meaningful
    # =========================================================================
    P_rw = 128
    rw_k_sb = nl.ndarray((P_rw, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=rw_k_sb, value=0.0)
    nisa.dma_copy(
        dst=rw_k_sb[0:T, 0:K],
        src=routing_weights_k[0:T, 0:K],
    )

    # =========================================================================
    # Output accumulators: [1, P] f32, one per H tile
    # All at partition 0 since T=1
    # =========================================================================
    out_accum = []
    for _ in nl.affine_range(num_h_tiles):
        tmp = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=tmp, value=0)
        out_accum.append(tmp)

    # PSUM tiles allocated per-iteration inside the K loop (compiler manages lifecycle)

    # =========================================================================
    # OPT 5: Pre-allocate reusable SBUF buffers OUTSIDE the K loop
    # =========================================================================
    # Gate+Up SBUF tiles for PSUM->SBUF copy (one per gu_t)
    gu_tiles = []
    for gu_t in range(num_gu_tiles):
        gu_tiles.append(nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf))

    # Activation bf16 tiles (one per i_t)
    act_bf16_tiles = []
    for i_t in range(num_i_tiles):
        act_bf16_tiles.append(nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf))

    # Routing weight scalar
    rw_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)

    # Down projection temporary buffers (one per h_t for parallel processing)
    down_sb_tiles = []
    for h_t in range(num_h_tiles):
        down_sb_tiles.append(nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf))

    scaled_tiles = []
    for h_t in range(num_h_tiles):
        scaled_tiles.append(nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf))

    # =========================================================================
    # OPT 1: Double-buffer gate+up weight tiles (ping-pong)
    # Two [P, P] buffers to overlap DMA with compute
    # =========================================================================
    gu_w_buf = [
        nl.ndarray((P, P), dtype=gate_up_weights.dtype, buffer=nl.sbuf),
        nl.ndarray((P, P), dtype=gate_up_weights.dtype, buffer=nl.sbuf),
    ]

    # =========================================================================
    # OPT 3: Double-buffer down weight tiles (ping-pong)
    # =========================================================================
    dw_buf = [
        nl.ndarray((P, P), dtype=down_weights.dtype, buffer=nl.sbuf),
        nl.ndarray((P, P), dtype=down_weights.dtype, buffer=nl.sbuf),
    ]

    # =========================================================================
    # K-expert loop: iterate K active experts, accumulate into out_accum
    # Sequential because each expert accumulates into the same output
    # =========================================================================
    for k_idx in range(K):

        # Dynamic expert ID scalar for token 0, expert k_idx
        eid_scalar = expert_idx_sb.ap(
            pattern=[[K, 1], [1, 1]],
            offset=k_idx,
        )

        # Extract routing weight (reuse pre-allocated rw_scalar)
        nisa.tensor_copy(
            dst=rw_scalar,
            src=rw_k_sb[0:1, k_idx:k_idx + 1],
        )

        # =================================================================
        # Stage 1: Gate+Up Projection with double-buffered DMA
        #   hidden[1, H] @ gate_up_w[H, 2*I] -> [1, 2*I]
        #   nc_matmul: dst[P_gu, 1] += w_tile[P_h, P_gu].T @ h_tile[P_h, 1]
        # =================================================================

        # Allocate PSUM tiles fresh each K iteration
        gu_psum_tiles = []
        for gu_t in nl.affine_range(num_gu_tiles):
            gu_psum_tiles.append(
                nl.zeros((P, 1), dtype=nl.float32, buffer=nl.psum,
                         name=f"gu_psum_k{k_idx}_gu{gu_t}")
            )

        # OPT 1: Double-buffered gate+up DMA
        # h_t loop: accumulates into gu_psum_tiles (sequential for PSUM dep)
        for h_t in range(num_h_tiles):
            h_off = h_t * P

            # Prefetch first gu_t tile into buf[0]
            nisa.dma_copy(
                dst=gu_w_buf[0],
                src=gate_up_weights.ap(
                    pattern=[[two_I, P], [1, P]],
                    offset=h_off * two_I + 0,
                    scalar_offset=eid_scalar,
                    indirect_dim=0,
                ),
            )

            for gu_t in range(num_gu_tiles):
                cur_buf = gu_t % 2

                # Prefetch NEXT gu_t tile into the OTHER buffer (if not last)
                if gu_t + 1 < num_gu_tiles:
                    next_buf = (gu_t + 1) % 2
                    next_gu_off = (gu_t + 1) * P
                    nisa.dma_copy(
                        dst=gu_w_buf[next_buf],
                        src=gate_up_weights.ap(
                            pattern=[[two_I, P], [1, P]],
                            offset=h_off * two_I + next_gu_off,
                            scalar_offset=eid_scalar,
                            indirect_dim=0,
                        ),
                    )

                nisa.nc_matmul(
                    dst=gu_psum_tiles[gu_t],
                    stationary=gu_w_buf[cur_buf],
                    moving=h_tiles_sb[h_t],
                )

        # PSUM -> SBUF for gate+up tiles (independent — use affine_range)
        for gu_t in nl.affine_range(num_gu_tiles):
            nisa.tensor_copy(dst=gu_tiles[gu_t], src=gu_psum_tiles[gu_t])

        # =================================================================
        # Stage 2: SiLU(gate) * up  (independent per i_t — use affine_range)
        # =================================================================
        for i_t in nl.affine_range(num_i_tiles):
            gate_act = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=gate_act, op=nl.silu,
                data=gu_tiles[i_t], scale=1.0,
            )

            act_f32 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=act_f32,
                data1=gate_act,
                data2=gu_tiles[num_i_tiles + i_t],
                op=nl.multiply,
            )

            nisa.tensor_copy(dst=act_bf16_tiles[i_t], src=act_f32)

        # =================================================================
        # Stage 3: Down Projection with double-buffered DMA
        #   act[1, I] @ down_w[I, H] -> [1, H]
        #   nc_matmul: dst[1, P_h] += act_tile[P_i, 1].T @ dw_tile[P_i, P_h]
        # =================================================================

        # h_t sequential because dw_buf is shared across iterations
        for h_t in range(num_h_tiles):
            h_off = h_t * P

            # Allocate PSUM tile fresh per h_t
            down_psum = nl.zeros((1, P), dtype=nl.float32, buffer=nl.psum,
                                 name=f"down_psum_k{k_idx}_h{h_t}")

            # OPT 3: Double-buffered down weight DMA
            # Prefetch first i_t tile into buf[0]
            nisa.dma_copy(
                dst=dw_buf[0],
                src=down_weights.ap(
                    pattern=[[H, P], [1, P]],
                    offset=0 * H + h_off,
                    scalar_offset=eid_scalar,
                    indirect_dim=0,
                ),
            )

            # i_t loop: accumulates into down_psum (sequential for PSUM dep)
            for i_t in range(num_i_tiles):
                cur_buf = i_t % 2

                # Prefetch NEXT i_t tile into the OTHER buffer (if not last)
                if i_t + 1 < num_i_tiles:
                    next_buf = (i_t + 1) % 2
                    next_i_off = (i_t + 1) * P
                    nisa.dma_copy(
                        dst=dw_buf[next_buf],
                        src=down_weights.ap(
                            pattern=[[H, P], [1, P]],
                            offset=next_i_off * H + h_off,
                            scalar_offset=eid_scalar,
                            indirect_dim=0,
                        ),
                    )

                nisa.nc_matmul(
                    dst=down_psum,
                    stationary=act_bf16_tiles[i_t],
                    moving=dw_buf[cur_buf],
                )

            # PSUM -> SBUF, POST_SCALE, accumulate (same h_t iteration)
            nisa.tensor_copy(dst=down_sb_tiles[h_t], src=down_psum)

            nisa.tensor_scalar(
                dst=scaled_tiles[h_t], data=down_sb_tiles[h_t],
                op0=nl.multiply, operand0=rw_scalar,
            )

            nisa.tensor_tensor(
                dst=out_accum[h_t],
                data1=out_accum[h_t],
                data2=scaled_tiles[h_t],
                op=nl.add,
            )

    # =========================================================================
    # Store accumulated output to HBM (after all K experts)
    # =========================================================================
    for h_t in nl.affine_range(num_h_tiles):
        h_off    = h_t * P
        out_cast = nl.ndarray((1, P), dtype=hidden_states.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_cast, src=out_accum[h_t])
        nisa.dma_copy(
            dst=output.ap(
                pattern=[[H, 1], [1, P]],
                offset=h_off,
            ),
            src=out_cast,
        )

    return output


# ---------------------------------------------------------------------------
# PyTorch reference (sparse inputs)
# ---------------------------------------------------------------------------

def pytorch_moe_reference_sparse(hidden_states, gate_up_weights, down_weights,
                                  expert_indices, routing_weights_k):
    """
    Reference MoE for sparse inputs.
    gate_up_weights: [E, H, 2*I]
    down_weights:    [E, I, H]
    expert_indices:  [T, K]
    routing_weights_k: [T, K]
    """
    T, H     = hidden_states.shape
    K        = expert_indices.shape[1]
    two_I    = gate_up_weights.shape[2]
    I_size   = two_I // 2

    output = torch.zeros(T, H, dtype=torch.float32)
    hs_f32 = hidden_states.float()

    for t in range(T):
        for ki in range(K):
            e   = expert_indices[t, ki].item()
            rw  = routing_weights_k[t, ki].item()
            gu_w = gate_up_weights[e].float()
            d_w  = down_weights[e].float()

            gu_out = hs_f32[t:t + 1] @ gu_w          # [1, 2*I]
            gate   = gu_out[:, :I_size]
            up     = gu_out[:, I_size:]
            act    = torch.nn.functional.silu(gate) * up  # [1, I]
            down   = act @ d_w                         # [1, H]
            output[t:t + 1] += down * rw

    return output.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Helper: build sparse routing from dense random weights
# ---------------------------------------------------------------------------

def _make_sparse_routing(T, E, K, seed=42):
    """
    Generate top-K routing data.
    Returns:
        expert_indices  [T, K]  int32
        routing_weights_k [T, K]  float32 (normalized, sum-to-1 per token)
        routing_weights_dense [T, E] float32 (for v1/v3 reference compatibility)
    """
    torch.manual_seed(seed)
    logits = torch.randn(T, E)
    # Top-K selection
    topk_vals, topk_idx = torch.topk(logits, K, dim=-1)
    topk_probs = torch.softmax(topk_vals, dim=-1).float()

    expert_indices = topk_idx.int()
    routing_weights_k = topk_probs

    # Build dense routing weight matrix for reference comparison
    routing_weights_dense = torch.zeros(T, E, dtype=torch.float32)
    for t in range(T):
        for ki in range(K):
            routing_weights_dense[t, expert_indices[t, ki].item()] += routing_weights_k[t, ki].item()

    return expert_indices, routing_weights_k, routing_weights_dense


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_nki_moe_v4_small():
    """Small correctness test: T=1, H=256, I=256, E=8, K=2."""
    import torch_xla.core.xla_model as xm

    print("=" * 70)
    print("nki_moe_v4 (Sparse K-Expert) — Small Correctness Test (T=1)")
    print("=" * 70)

    device = xm.xla_device()
    print(f"\nUsing device: {device}")

    T, H, I_size, E, K = 1, 256, 256, 8, 2
    two_I = 2 * I_size

    print(f"Shapes: T={T}, H={H}, I={I_size}, E={E}, K={K}")

    dtype = torch.bfloat16
    torch.manual_seed(42)

    hidden_states   = torch.randn(T, H, dtype=dtype)
    gate_up_weights = torch.randn(E, H, two_I, dtype=dtype) * 0.02
    down_weights    = torch.randn(E, I_size, H, dtype=dtype) * 0.02

    expert_indices, routing_weights_k, _ = _make_sparse_routing(T, E, K, seed=42)

    print(f"  expert_indices row 0: {expert_indices[0].tolist()}")
    print(f"  routing_weights_k row 0: {routing_weights_k[0].tolist()}")

    print("\n[1/3] Computing reference (PyTorch CPU)...")
    ref = pytorch_moe_reference_sparse(
        hidden_states, gate_up_weights, down_weights,
        expert_indices, routing_weights_k,
    )

    print("[2/3] Computing nki_moe_v4 on Trainium...")
    nki_out = nki_moe_v4(
        hidden_states.to(device),
        gate_up_weights.to(device),
        down_weights.to(device),
        expert_indices.to(device),
        routing_weights_k.to(device),
    ).cpu()

    print("[3/3] Comparing outputs...")
    diff      = torch.abs(ref.float() - nki_out.float())
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nResults:")
    print(f"  Output shape : {nki_out.shape}")
    print(f"  Max  |diff|  : {max_diff:.6e}")
    print(f"  Mean |diff|  : {mean_diff:.6e}")

    threshold = 0.05
    if max_diff < threshold:
        print("\n" + "=" * 70)
        print("SUCCESS! nki_moe_v4 small-shape test PASSED.")
        print("=" * 70)
        return True
    else:
        print(f"\nFAILED — max diff {max_diff:.4e} exceeds threshold {threshold:.4e}")
        return False


def test_nki_moe_v4_qwen3():
    """Qwen3 TKG shapes: T=1, H=2048, I=256, E=128, K=8."""
    import torch_xla.core.xla_model as xm

    print("\n" + "=" * 70)
    print("nki_moe_v4 — Qwen3 TKG Shape Test (T=1, H=2048, I=256, E=128, K=8)")
    print("=" * 70)

    device = xm.xla_device()

    T, H, I_size, E, K = 1, 2048, 256, 128, 8
    two_I = 2 * I_size

    print(f"Shapes: T={T}, H={H}, I={I_size}, E={E}, K={K}")
    print(f"gate_up_weights: [{E}, {H}, {two_I}] bf16")
    print(f"down_weights:    [{E}, {I_size}, {H}] bf16")

    dtype = torch.bfloat16
    torch.manual_seed(123)

    hidden_states   = torch.randn(T, H, dtype=dtype)
    gate_up_weights = torch.randn(E, H, two_I, dtype=dtype) * 0.02
    down_weights    = torch.randn(E, I_size, H, dtype=dtype) * 0.02

    expert_indices, routing_weights_k, _ = _make_sparse_routing(T, E, K, seed=99)

    print(f"  expert_indices: {expert_indices[0].tolist()}")

    print("\n[1/3] Reference (PyTorch CPU)...")
    ref = pytorch_moe_reference_sparse(
        hidden_states, gate_up_weights, down_weights,
        expert_indices, routing_weights_k,
    )

    print("[2/3] nki_moe_v4 on Trainium...")
    nki_out = nki_moe_v4(
        hidden_states.to(device),
        gate_up_weights.to(device),
        down_weights.to(device),
        expert_indices.to(device),
        routing_weights_k.to(device),
    ).cpu()

    print("[3/3] Comparing...")
    diff      = torch.abs(ref.float() - nki_out.float())
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nResults:")
    print(f"  Output shape : {nki_out.shape}")
    print(f"  Max  |diff|  : {max_diff:.6e}")
    print(f"  Mean |diff|  : {mean_diff:.6e}")

    threshold = 0.05
    if max_diff < threshold:
        print("\n" + "=" * 70)
        print("SUCCESS! nki_moe_v4 Qwen3 shape test PASSED.")
        print("=" * 70)
        return True
    else:
        print(f"\nFAILED — max diff {max_diff:.4e} exceeds threshold {threshold:.4e}")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ok_small = test_nki_moe_v4_small()

    if ok_small:
        print("\nSmall-shape test passed — proceeding to Qwen3 shape test...")
        ok_qwen3 = test_nki_moe_v4_qwen3()
        if ok_qwen3:
            print("\nAll tests PASSED.")
        else:
            print("\nQwen3 shape test FAILED.")
    else:
        print("\nSmall-shape test FAILED — skipping Qwen3 test.")
