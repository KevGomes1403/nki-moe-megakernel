"""
Optimized MoE kernel — v6: Double-Buffered DMA Pipeline.

Builds on v3 (all optimizations: hoisted hidden-state loads, prefetched routing
weights, hoisted act cast, PSUM reuse, affine_range on inner loops) and adds:

  Double-buffering for gate+up weight tiles:
    Two pre-allocated SBUF ping-pong buffers (w_buf[0], w_buf[1]) alternate
    between DMA load (for the next tile) and compute (for the current tile).
    This allows DMA and tensor-engine to overlap, hiding HBM load latency.

  Double-buffering for down weight tiles:
    Same ping-pong pattern for dw_tile loads vs matmul compute.

  All other v3 optimizations retained:
    - Hidden states loaded once, all num_h_tiles [P, T] tiles pre-loaded.
    - Routing weights prefetched as [T, E] SBUF array.
    - PSUM tiles allocated once and reused (memset at expert start).
    - Separate down_psum_tiles[h_t] for nl.affine_range(h_t) independence.
    - Act bf16 cast hoisted outside h_t loop.
    - nl.affine_range on gu_t (independent) and h_t in down projection.

Double-buffer algorithm (gate+up example):
  total_tiles = num_h_tiles * num_gu_tiles
  w_buf[0], w_buf[1] = two pre-allocated SBUF [P, P] buffers

  # Prefetch first tile into buf 0 before the loop
  dma_copy(dst=w_buf[0], src=weight_tile(expert, h_t=0, gu_t=0))

  flat counter across (h_t, gu_t):
    for h_t in range(num_h_tiles):          # sequential: PSUM accumulates across h_t
        for gu_t in nl.affine_range(num_gu_tiles):  # independent per gu_t
            idx = h_t * num_gu_tiles + gu_t
            # Prefetch NEXT tile into other buffer (while compute uses cur)
            next_idx = idx + 1
            if next_idx < total_tiles:
                next_h_t  = next_idx // num_gu_tiles
                next_gu_t = next_idx %  num_gu_tiles
                dma_copy(dst=w_buf[(idx+1) % 2], src=weight_tile(expert, next_h_t, next_gu_t))
            # Compute with current buffer
            nc_matmul(dst=gu_psum_tiles[gu_t], stationary=w_buf[idx % 2], moving=h_tiles_sb[h_t])

Constraints (same as v1/v3):
    T <= 128
    H % 128 == 0
    I % 128 == 0
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
# v6 kernel — double-buffered weight DMA + all v3 optimizations
# ---------------------------------------------------------------------------

@nki.jit(platform_target="trn2")
def nki_moe_v6(
    hidden_states,      # [T, H]       bf16
    gate_up_weights,    # [E, H, 2*I]  bf16
    down_weights,       # [E, I, H]    bf16
    routing_weights,    # [T, E]       float32
):
    """
    Fused MoE with double-buffered weight DMA pipeline.

    Tile layouts (same as v1/v3):
      Gate+Up intermediate: [P=128, T]  (partition=gu_dim,    free=token_dim)
      Down output / accum:  [T,   P=128](partition=token_dim, free=hidden_dim)

    Key optimization — double-buffered weight loads:
      Two SBUF ping-pong buffers per projection (gate+up, down) alternate
      between DMA (loading tile n+1) and compute (using tile n), so that
      DMA and tensor-engine execution overlap and HBM latency is hidden.
    """
    T      = hidden_states.shape[0]
    H      = hidden_states.shape[1]
    E      = gate_up_weights.shape[0]
    two_I  = gate_up_weights.shape[2]
    I_size = two_I // 2
    P      = 128

    num_h_tiles  = H      // P
    num_i_tiles  = I_size // P
    num_gu_tiles = two_I  // P

    # Flatten 3-D weight tensors → 2-D for .ap() indexing
    gate_up_flat = gate_up_weights.reshape((E * H, two_I))
    down_flat    = down_weights.reshape((E * I_size, H))

    # HBM output buffer
    output = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # =========================================================================
    # PRE-LOOP: load all loop-invariant data into SBUF
    # =========================================================================

    # 1. Load routing weights [T, E] f32 once
    rw_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rw_sb, src=routing_weights[0:T, 0:E])

    # 2. Load hidden states: num_h_tiles [P, T] bf16 tiles (transposed view)
    h_tiles_sb = []
    for h_t in nl.affine_range(num_h_tiles):
        h_off = h_t * P
        h_sb  = nl.ndarray((P, T), dtype=hidden_states.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=h_sb,
            src=hidden_states.ap(
                pattern=[[1, P], [H, T]],
                offset=h_off,
            ),
        )
        h_tiles_sb.append(h_sb)

    # 3. Allocate gate+up PSUM tiles once; reuse per expert (memset each time)
    gu_psum_tiles = []
    for gu_t in nl.affine_range(num_gu_tiles):
        gu_psum_tiles.append(
            nl.ndarray((P, T), dtype=nl.float32, buffer=nl.psum)
        )

    # 4. Allocate down PSUM tiles once per h_t; reuse per expert (memset each time)
    down_psum_tiles = []
    for h_t in nl.affine_range(num_h_tiles):
        down_psum_tiles.append(
            nl.ndarray((T, P), dtype=nl.float32, buffer=nl.psum)
        )

    # 5. Output accumulator: [T, P] f32 SBUF, zeroed once
    out_accum = []
    for _ in nl.affine_range(num_h_tiles):
        tmp = nl.ndarray((T, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=tmp, value=0)
        out_accum.append(tmp)

    # =========================================================================
    # Double-buffer ping-pong buffers for weight tiles
    #
    # We allocate exactly 2 SBUF tiles for gate+up weights and 2 for down
    # weights.  On every iteration we load tile[n+1] into the "other" buffer
    # while computing with tile[n] from the "current" buffer.
    # =========================================================================

    # Gate+up double-buffer: 2 × [P, P] bf16
    gu_w_buf = [
        nl.ndarray((P, P), dtype=gate_up_weights.dtype, buffer=nl.sbuf),
        nl.ndarray((P, P), dtype=gate_up_weights.dtype, buffer=nl.sbuf),
    ]

    # Down double-buffer: 2 × [P, P] bf16
    dw_buf = [
        nl.ndarray((P, P), dtype=down_weights.dtype, buffer=nl.sbuf),
        nl.ndarray((P, P), dtype=down_weights.dtype, buffer=nl.sbuf),
    ]

    # =========================================================================
    # EXPERT LOOP
    # =========================================================================
    for expert_id in range(E):

        # --- Step a: zero gate+up PSUM tiles for this expert ---
        for gu_t in nl.affine_range(num_gu_tiles):
            nisa.memset(dst=gu_psum_tiles[gu_t], value=0)

        # --- Step b: Gate+Up matmul with double-buffered weight loads ---
        #
        # Linear tile ordering: idx = h_t * num_gu_tiles + gu_t
        # w_buf[idx % 2]       <- tile being computed THIS iteration
        # w_buf[(idx+1) % 2]   <- tile being loaded FOR NEXT iteration
        #
        # Before the loop starts we prefetch tile 0 into w_buf[0].
        # Inside the loop:
        #   - Issue DMA for tile (idx+1) into w_buf[(idx+1)%2]
        #   - Issue matmul using w_buf[idx%2]
        #
        # NKI schedules DMA and matmul concurrently → HBM latency hidden.

        # Prefetch the very first gate+up weight tile (expert_id, h_t=0, gu_t=0)
        nisa.dma_copy(
            dst=gu_w_buf[0],
            src=gate_up_flat.ap(
                pattern=[[two_I, P], [1, P]],
                offset=expert_id * H * two_I + 0 * two_I + 0,
            ),
        )

        total_gu_tiles = num_h_tiles * num_gu_tiles

        # Use Python range() for both loops so that idx, idx%2, (idx+1)%2 are
        # all Python integers at trace time — required for buffer selection.
        # The double-buffer pattern (dma_copy followed by nc_matmul on different
        # SBUF addresses) still allows the compiler to overlap DMA with compute.
        for h_t in range(num_h_tiles):
            h_tile = h_tiles_sb[h_t]

            for gu_t in range(num_gu_tiles):
                gu_off = gu_t * P
                idx    = h_t * num_gu_tiles + gu_t

                # Prefetch next tile (if it exists) into the other buffer
                next_idx = idx + 1
                if next_idx < total_gu_tiles:
                    next_h_t  = next_idx // num_gu_tiles
                    next_gu_t = next_idx %  num_gu_tiles
                    next_h_off  = next_h_t  * P
                    next_gu_off = next_gu_t * P
                    nisa.dma_copy(
                        dst=gu_w_buf[(idx + 1) % 2],
                        src=gate_up_flat.ap(
                            pattern=[[two_I, P], [1, P]],
                            offset=expert_id * H * two_I + next_h_off * two_I + next_gu_off,
                        ),
                    )

                # Compute gate+up matmul using current buffer
                nisa.nc_matmul(
                    dst=gu_psum_tiles[gu_t],
                    stationary=gu_w_buf[idx % 2],
                    moving=h_tile,
                )

        # --- Step c: PSUM -> SBUF for gate+up tiles ---
        gu_tiles = []
        for gu_t in nl.affine_range(num_gu_tiles):
            gu_sb = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gu_sb, src=gu_psum_tiles[gu_t])
            gu_tiles.append(gu_sb)

        # --- Step d+e: SiLU(gate)*up + cast to bf16 (hoisted outside h_t loop) ---
        act_bf16_tiles = []
        for i_t in nl.affine_range(num_i_tiles):
            gate_act = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=gate_act, op=nl.silu,
                data=gu_tiles[i_t], scale=1.0,
            )

            act_f32 = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=act_f32,
                data1=gate_act,
                data2=gu_tiles[num_i_tiles + i_t],
                op=nl.multiply,
            )

            # Cast f32 -> bf16 once per i_t (hoisted outside h_t loop)
            act_bf16 = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=act_bf16, src=act_f32)
            act_bf16_tiles.append(act_bf16)

        # --- Step f: zero down PSUM tiles for this expert ---
        for h_t in nl.affine_range(num_h_tiles):
            nisa.memset(dst=down_psum_tiles[h_t], value=0)

        # --- Step g: Down matmul with double-buffered weight loads ---
        #
        # Double-buffer: linear tile order idx = h_t * num_i_tiles + i_t.
        # dw_buf[idx%2]     <- tile being computed this iteration
        # dw_buf[(idx+1)%2] <- tile being loaded for the next iteration
        #
        # Both loops use Python range() so that idx, idx%2, (idx+1)%2 are
        # Python integers at trace time — required for list indexing and
        # the if-branch guard.  Different h_t tiles write to distinct
        # down_psum_tiles[h_t], so they are logically independent; the
        # compiler can still overlap DMA with nc_matmul.

        # Prefetch the first down weight tile (expert_id, h_t=0, i_t=0)
        nisa.dma_copy(
            dst=dw_buf[0],
            src=down_flat.ap(
                pattern=[[H, P], [1, P]],
                offset=expert_id * I_size * H + 0 * H + 0,
            ),
        )

        total_down_tiles = num_h_tiles * num_i_tiles

        for h_t in range(num_h_tiles):
            h_off = h_t * P

            for i_t in range(num_i_tiles):
                i_off = i_t * P
                idx   = h_t * num_i_tiles + i_t

                # Prefetch next down weight tile
                next_idx = idx + 1
                if next_idx < total_down_tiles:
                    next_h_t   = next_idx // num_i_tiles
                    next_i_t   = next_idx %  num_i_tiles
                    next_h_off = next_h_t * P
                    next_i_off = next_i_t * P
                    nisa.dma_copy(
                        dst=dw_buf[(idx + 1) % 2],
                        src=down_flat.ap(
                            pattern=[[H, P], [1, P]],
                            offset=expert_id * I_size * H + next_i_off * H + next_h_off,
                        ),
                    )

                # Compute down matmul using current buffer
                nisa.nc_matmul(
                    dst=down_psum_tiles[h_t],
                    stationary=act_bf16_tiles[i_t],
                    moving=dw_buf[idx % 2],
                )

        # --- Step h: routing weight extract, scale, accumulate ---
        rw_col = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=rw_col,
            src=rw_sb[0:T, expert_id:expert_id + 1],
        )

        for h_t in nl.affine_range(num_h_tiles):
            # PSUM -> SBUF
            down_sb = nl.ndarray((T, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=down_sb, src=down_psum_tiles[h_t])

            # Scale by routing weight (broadcasts [T,1] -> [T,P])
            scaled = nl.ndarray((T, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=scaled, data=down_sb,
                op0=nl.multiply, operand0=rw_col,
            )

            # Accumulate into output
            nisa.tensor_tensor(
                dst=out_accum[h_t],
                data1=out_accum[h_t],
                data2=scaled,
                op=nl.add,
            )

    # =========================================================================
    # STORE to HBM (cast f32 -> bf16)
    # =========================================================================
    for h_t in nl.affine_range(num_h_tiles):
        h_off    = h_t * P
        out_cast = nl.ndarray((T, P), dtype=hidden_states.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_cast, src=out_accum[h_t])
        nisa.dma_copy(
            dst=output.ap(
                pattern=[[H, T], [1, P]],
                offset=h_off,
            ),
            src=out_cast,
        )

    return output


# ---------------------------------------------------------------------------
# PyTorch reference (exact match to v1/v3)
# ---------------------------------------------------------------------------

def pytorch_moe_reference(hidden_states, gate_up_weights, down_weights,
                           routing_weights):
    T, H   = hidden_states.shape
    E      = gate_up_weights.shape[0]
    two_I  = gate_up_weights.shape[2]
    I_size = two_I // 2

    output = torch.zeros(T, H, dtype=torch.float32)
    hs_f32 = hidden_states.float()

    for e in range(E):
        gu_w = gate_up_weights[e].float()
        d_w  = down_weights[e].float()

        gu_out = hs_f32 @ gu_w
        gate   = gu_out[:, :I_size]
        up     = gu_out[:, I_size:]
        act    = torch.nn.functional.silu(gate) * up
        down   = act @ d_w

        rw     = routing_weights[:, e:e + 1].float()
        output = output + down * rw

    return output.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Helper: build sparse routing weights from top-K indices/values
# ---------------------------------------------------------------------------

def make_routing_weights(T, E, K, seed=42):
    """Return a [T, E] float32 routing weight tensor with K non-zeros per row."""
    torch.manual_seed(seed)
    expert_indices = torch.randint(0, E, (T, K))
    expert_weights = torch.softmax(torch.randn(T, K), dim=-1)
    rw = torch.zeros(T, E, dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            rw[t, expert_indices[t, k].item()] += expert_weights[t, k].item()
    return rw


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_nki_moe_v6_small():
    import torch_xla.core.xla_model as xm

    print("=" * 70)
    print("nki_moe_v6 — Small correctness test (T=4, H=256, I=256, E=4, K=2)")
    print("=" * 70)

    device = xm.xla_device()
    print(f"Using device: {device}")

    T, H, I_size, E, K = 4, 256, 256, 4, 2
    two_I = 2 * I_size

    torch.manual_seed(42)
    dtype = torch.bfloat16

    hidden_states   = torch.randn(T, H, dtype=dtype)
    gate_up_weights = torch.randn(E, H, two_I, dtype=dtype) * 0.02
    down_weights    = torch.randn(E, I_size, H, dtype=dtype) * 0.02
    routing_weights = make_routing_weights(T, E, K, seed=42)

    print(f"routing_weights row 0: {routing_weights[0].tolist()}")

    print("\n[1/3] PyTorch reference...")
    ref = pytorch_moe_reference(hidden_states, gate_up_weights, down_weights,
                                routing_weights)

    print("[2/3] NKI v6 kernel...")
    nki_out = nki_moe_v6(
        hidden_states.to(device),
        gate_up_weights.to(device),
        down_weights.to(device),
        routing_weights.to(device),
    ).cpu()

    print("[3/3] Comparing...")
    diff      = torch.abs(ref.float() - nki_out.float())
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nOutput shape : {nki_out.shape}")
    print(f"Max  |diff|  : {max_diff:.6e}")
    print(f"Mean |diff|  : {mean_diff:.6e}")

    ok = max_diff < 0.05
    print("\nSMALL TEST: " + ("PASS" if ok else f"FAIL (max_diff={max_diff:.4e})"))
    return ok


def test_nki_moe_v6_qwen3():
    """Qwen3 TKG shapes: T=1, H=2048, I=256 (padded from 192), E=128, K=8."""
    import torch_xla.core.xla_model as xm

    print("\n" + "=" * 70)
    print("nki_moe_v6 — Qwen3 TKG shapes (T=1, H=2048, I=256, E=128, K=8)")
    print("=" * 70)

    device = xm.xla_device()

    T, H, I_size, E, K = 1, 2048, 256, 128, 8
    two_I = 2 * I_size

    torch.manual_seed(123)
    dtype = torch.bfloat16

    hidden_states   = torch.randn(T, H, dtype=dtype)
    gate_up_weights = torch.randn(E, H, two_I, dtype=dtype) * 0.02
    down_weights    = torch.randn(E, I_size, H, dtype=dtype) * 0.02
    routing_weights = make_routing_weights(T, E, K, seed=123)

    print(f"Routing weight row 0 non-zeros: {(routing_weights[0] != 0).sum().item()}")

    ref = pytorch_moe_reference(hidden_states, gate_up_weights, down_weights,
                                routing_weights)
    nki_out = nki_moe_v6(
        hidden_states.to(device),
        gate_up_weights.to(device),
        down_weights.to(device),
        routing_weights.to(device),
    ).cpu()

    diff      = torch.abs(ref.float() - nki_out.float())
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max  |diff|  : {max_diff:.6e}")
    print(f"Mean |diff|  : {mean_diff:.6e}")

    ok = max_diff < 0.05
    print("QWEN3 TEST: " + ("PASS" if ok else f"FAIL (max_diff={max_diff:.4e})"))
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ok1 = test_nki_moe_v6_small()
    ok2 = test_nki_moe_v6_qwen3()

    print("\n" + "=" * 70)
    if ok1 and ok2:
        print("All tests PASSED.")
    else:
        results = []
        if not ok1:
            results.append("small")
        if not ok2:
            results.append("qwen3")
        print(f"FAILED tests: {', '.join(results)}")
    print("=" * 70)
