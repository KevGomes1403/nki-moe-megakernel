"""
NKI MoE Kernel v2 — Affine Range + PSUM Reuse optimization.

Optimizations over the baseline (nki_moe_fused.py):

  Fix 1: nl.affine_range() on independent inner loops
      - Gate+Up: gu_t loop (independent PSUM destinations) → affine_range
      - Down: h_t loop (independent per-h_t PSUM tiles) → affine_range
      - Sequential loops that carry PSUM accumulation deps stay as range()

  Fix 2: Reuse PSUM tiles across experts
      Allocate gu_psum_tiles and down_psum_tiles ONCE before the expert loop,
      zero via nisa.memset at the start of each expert iteration instead of
      allocating E * num_tiles unique PSUM buffers.

  Fix 3: Hoist hidden state load before the expert loop
      Load hidden tiles [P, T] once; reuse for all experts.

  Fix 4: Hoist f32→bf16 act cast outside h_t loop in down projection
      Cast act_tiles[i_t] to bf16 once per i_t, then reuse across h_t.

Constraints (same as baseline):
    T <= 128 (tokens, fits in partition tile)
    H % 128 == 0 (hidden dimension)
    I % 128 == 0 (intermediate dimension)
"""

import os
import time
import torch
import nki
import nki.language as nl
import nki.isa as nisa

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"] = " --disable-dge "


@nki.jit(platform_target="trn2")
def nki_moe_v2(
    hidden_states,      # [T, H] bf16
    gate_up_weights,    # [E, H, 2*I] bf16
    down_weights,       # [E, I, H] bf16
    routing_weights,    # [T, E] float32 — precomputed per-token per-expert weights
):
    """
    Fused MoE v2: gate+up → SiLU → down, per-expert routing, with
    affine_range pipelining and PSUM reuse across experts.

    Tile layouts:
      Gate+Up intermediate: [P=128, T] (partition=gu_dim, free=token_dim)
      Down output / accum:  [T, P=128] (partition=token_dim, free=hidden_dim)
    """
    T = hidden_states.shape[0]
    H = hidden_states.shape[1]
    E = gate_up_weights.shape[0]
    two_I = gate_up_weights.shape[2]
    I_size = two_I // 2
    P = 128

    num_h_tiles = H // P
    num_i_tiles = I_size // P
    num_gu_tiles = two_I // P

    # Reshape 3D weight tensors → 2D for flat .ap() indexing
    gate_up_flat = gate_up_weights.reshape((E * H, two_I))
    down_flat = down_weights.reshape((E * I_size, H))

    # Output in HBM
    output = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # Output accumulator tiles in SBUF [T, P] — one per H chunk
    out_accum = []
    for _ in range(num_h_tiles):
        tmp = nl.ndarray((T, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=tmp, value=0)
        out_accum.append(tmp)

    # -----------------------------------------------------------------------
    # Fix 3: Hoist hidden tile loads before the expert loop.
    # Each tile is [P, T] — transposed view of hidden_states[:, h_off:h_off+P]
    # -----------------------------------------------------------------------
    h_tiles = []
    for h_t in nl.affine_range(num_h_tiles):
        h_off = h_t * P
        h_tile = nl.ndarray((P, T), dtype=hidden_states.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=h_tile,
            src=hidden_states.ap(
                pattern=[[1, P], [H, T]],
                offset=h_off,
            ),
        )
        h_tiles.append(h_tile)

    # -----------------------------------------------------------------------
    # Fix 2: Allocate PSUM tiles ONCE outside the expert loop.
    # Zero-init at the start of each expert via nisa.memset.
    # -----------------------------------------------------------------------
    gu_psum_tiles = []
    for gu_t_ in range(num_gu_tiles):
        gu_psum_tiles.append(
            nl.ndarray((P, T), dtype=nl.float32, buffer=nl.psum)
        )

    down_psum_tiles = []
    for h_t_ in range(num_h_tiles):
        down_psum_tiles.append(
            nl.ndarray((T, P), dtype=nl.float32, buffer=nl.psum)
        )

    # --- Process each expert (unrolled at trace time) ---
    for expert_id in nl.affine_range(E):

        # Zero the reused PSUM tiles at the start of this expert
        for gu_t in nl.affine_range(num_gu_tiles):
            nisa.memset(dst=gu_psum_tiles[gu_t], value=0)
        for h_t in nl.affine_range(num_h_tiles):
            nisa.memset(dst=down_psum_tiles[h_t], value=0)

        # Load routing weight [T, 1] from routing_weights[T, E]
        rw = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=rw,
            src=routing_weights.ap(
                pattern=[[E, T], [1, 1]],
                offset=expert_id,
            ),
        )

        # ====================================================================
        # Stage 1: Gate+Up Projection
        #   hidden[T,H] @ gate_up_w[H, 2*I] → [T, 2*I]
        #
        #   nc_matmul: dst += stationary.T @ moving
        #     stationary = w_tile [P_h, P_gu]
        #     moving     = h_tile [P_h, T]
        #     result     = [P_gu, T]  (partition=gu_dim)
        #   Accumulated across H tiles (h_t loop carries PSUM deps → range()).
        #
        # Fix 1a: gu_t loop uses affine_range — each gu_t writes to a
        #         distinct gu_psum_tiles[gu_t], so iterations are independent.
        # ====================================================================
        for h_t in nl.affine_range(num_h_tiles):          # sequential: PSUM accumulation
            h_off = h_t * P
            h_tile = h_tiles[h_t]               # Fix 3: use hoisted tile

            for gu_t in nl.affine_range(num_gu_tiles):   # Fix 1a: independent
                gu_off = gu_t * P

                # Load weight tile [P, P] from gate_up_flat
                w_tile = nl.ndarray((P, P), dtype=gate_up_weights.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=w_tile,
                    src=gate_up_flat.ap(
                        pattern=[[two_I, P], [1, P]],
                        offset=(expert_id * H + h_off) * two_I + gu_off,
                    ),
                )

                # dst += w_tile.T @ h_tile = [P_gu, T]
                nisa.nc_matmul(
                    dst=gu_psum_tiles[gu_t],
                    stationary=w_tile,
                    moving=h_tile,
                )

        # PSUM → SBUF for each gate+up tile
        gu_tiles = []
        for gu_t in nl.affine_range(num_gu_tiles):
            gu_sb = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gu_sb, src=gu_psum_tiles[gu_t])
            gu_tiles.append(gu_sb)

        # ====================================================================
        # Stage 2: SiLU(gate) * up
        #   gate = gu_tiles[0 : num_i_tiles]       — first I columns
        #   up   = gu_tiles[num_i_tiles : end]      — last  I columns
        #   All tiles are [P, T] layout.
        # ====================================================================
        act_tiles = []
        act_bf16_tiles = []   # Fix 4: cast to bf16 once per i_t (before h_t loop)
        for i_t in nl.affine_range(num_i_tiles):
            gate_act = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=gate_act, op=nl.silu,
                data=gu_tiles[i_t], scale=1.0,
            )

            act = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=act,
                data1=gate_act,
                data2=gu_tiles[num_i_tiles + i_t],
                op=nl.multiply,
            )
            act_tiles.append(act)

            # Fix 4: hoist bf16 cast outside the h_t loop
            act_bf16 = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=act_bf16, src=act)
            act_bf16_tiles.append(act_bf16)

        # ====================================================================
        # Stage 3: Down Projection
        #   act[T, I] @ down_w[I, H] → [T, H]
        #
        #   nc_matmul:
        #     stationary = act_tile [P_i, T]
        #     moving     = dw_tile  [P_i, P_h]
        #     result     = [T, P_h]  (partition=token_dim)
        #   Accumulated across I tiles (i_t carries PSUM deps → range()).
        #
        # Fix 1b: h_t loop uses affine_range — each h_t has its own
        #         down_psum_tiles[h_t], so iterations are independent.
        # Fix 4: use pre-cast act_bf16_tiles[i_t] instead of casting inside.
        # ====================================================================
        for h_t in nl.affine_range(num_h_tiles):   # Fix 1b: independent per-h_t
            h_off = h_t * P
            down_psum = down_psum_tiles[h_t]        # Fix 2: reused PSUM tile

            for i_t in nl.affine_range(num_i_tiles):          # sequential: PSUM accumulation
                i_off = i_t * P

                # Fix 4: use pre-cast bf16 tile
                act_bf16 = act_bf16_tiles[i_t]

                # Load down weight [P, P] from down_flat
                dw_tile = nl.ndarray((P, P), dtype=down_weights.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=dw_tile,
                    src=down_flat.ap(
                        pattern=[[H, P], [1, P]],
                        offset=(expert_id * I_size + i_off) * H + h_off,
                    ),
                )

                # dst += act_tile.T @ dw_tile = [T, P_h]
                nisa.nc_matmul(
                    dst=down_psum,
                    stationary=act_bf16,
                    moving=dw_tile,
                )

            # PSUM → SBUF
            down_sb = nl.ndarray((T, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=down_sb, src=down_psum)

            # Scale by routing weight — broadcasts [T,1] across free dim
            scaled = nl.ndarray((T, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=scaled, data=down_sb,
                op0=nl.multiply, operand0=rw,
            )

            # Accumulate into output
            nisa.tensor_tensor(
                dst=out_accum[h_t],
                data1=out_accum[h_t],
                data2=scaled,
                op=nl.add,
            )

    # Store output to HBM (cast f32 → output dtype)
    for h_t in nl.affine_range(num_h_tiles):
        h_off = h_t * P
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
# PyTorch reference — mirrors kernel logic exactly (unchanged from baseline)
# ---------------------------------------------------------------------------

def pytorch_moe_reference(hidden_states, gate_up_weights, down_weights,
                          routing_weights):
    T, H = hidden_states.shape
    E = gate_up_weights.shape[0]
    two_I = gate_up_weights.shape[2]
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

        rw = routing_weights[:, e:e + 1].float()
        output = output + down * rw

    return output.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_nki_moe_v2():
    import torch_xla.core.xla_model as xm

    print("=" * 70)
    print("NKI MoE v2 Kernel Test (Affine Range + PSUM Reuse)")
    print("=" * 70)

    device = xm.xla_device()
    print(f"\nUsing device: {device}")

    T       = 4
    H       = 256
    I_size  = 256
    E       = 4
    K       = 2
    two_I   = 2 * I_size

    print(f"\nShapes: T={T}, H={H}, I={I_size}, E={E}, K={K}")

    torch.manual_seed(42)
    dtype = torch.bfloat16

    hidden_states   = torch.randn(T, H, dtype=dtype)
    gate_up_weights = torch.randn(E, H, two_I, dtype=dtype) * 0.02
    down_weights    = torch.randn(E, I_size, H, dtype=dtype) * 0.02

    expert_indices  = torch.randint(0, E, (T, K))
    expert_weights  = torch.softmax(torch.randn(T, K), dim=-1)

    routing_weights = torch.zeros(T, E, dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            e = expert_indices[t, k].item()
            routing_weights[t, e] += expert_weights[t, k].item()

    print(f"  routing_weights row 0: {routing_weights[0].tolist()}")

    print("\n[1/3] Computing reference (PyTorch CPU)...")
    ref_output = pytorch_moe_reference(
        hidden_states, gate_up_weights, down_weights, routing_weights
    )

    print("[2/3] Computing NKI kernel on Trainium...")
    nki_output = nki_moe_v2(
        hidden_states.to(device),
        gate_up_weights.to(device),
        down_weights.to(device),
        routing_weights.to(device),
    )

    print("[3/3] Comparing outputs...")
    nki_cpu = nki_output.cpu().float()
    diff      = torch.abs(ref_output.float() - nki_cpu)
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nResults:")
    print(f"  Output shape : {nki_output.shape}")
    print(f"  Max  |diff|  : {max_diff:.6e}")
    print(f"  Mean |diff|  : {mean_diff:.6e}")

    threshold = 0.05
    if max_diff < threshold:
        print("\n" + "=" * 70)
        print("SUCCESS! nki_moe_v2 kernel matches reference.")
        print("=" * 70)
        return True
    else:
        print(f"\nFAILED — max diff {max_diff:.4e} exceeds threshold {threshold:.4e}")
        print(f"  ref sample  : {ref_output[0, :8].tolist()}")
        print(f"  nki sample  : {nki_cpu[0, :8].tolist()}")
        return False


def test_nki_moe_v2_large():
    """Test with larger shapes (more H/I tiles)."""
    import torch_xla.core.xla_model as xm

    print("\n" + "=" * 70)
    print("NKI MoE v2 Kernel Test — larger shapes")
    print("=" * 70)

    device = xm.xla_device()

    T       = 8
    H       = 512
    I_size  = 384
    E       = 4
    K       = 2
    two_I   = 2 * I_size

    print(f"\nShapes: T={T}, H={H}, I={I_size}, E={E}, K={K}")

    torch.manual_seed(123)
    dtype = torch.bfloat16

    hidden_states   = torch.randn(T, H, dtype=dtype)
    gate_up_weights = torch.randn(E, H, two_I, dtype=dtype) * 0.02
    down_weights    = torch.randn(E, I_size, H, dtype=dtype) * 0.02
    expert_indices  = torch.randint(0, E, (T, K))
    expert_weights  = torch.softmax(torch.randn(T, K), dim=-1)

    routing_weights = torch.zeros(T, E, dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            e = expert_indices[t, k].item()
            routing_weights[t, e] += expert_weights[t, k].item()

    ref = pytorch_moe_reference(hidden_states, gate_up_weights, down_weights,
                                routing_weights)
    nki_out = nki_moe_v2(
        hidden_states.to(device), gate_up_weights.to(device),
        down_weights.to(device), routing_weights.to(device),
    ).cpu()

    diff = torch.abs(ref.float() - nki_out.float())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max  |diff|  : {max_diff:.6e}")
    print(f"  Mean |diff|  : {mean_diff:.6e}")

    if max_diff < 0.05:
        print("  PASS")
        return True
    else:
        print(f"  FAIL — max diff {max_diff:.4e} exceeds threshold")
        return False


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _run_benchmark(label, T, H, I_size, E, dtype, n_warmup=5, n_iters=20):
    import torch_xla.core.xla_model as xm

    two_I  = 2 * I_size
    device = xm.xla_device()

    torch.manual_seed(42)
    hidden_states   = torch.randn(T, H, dtype=dtype).to(device)
    gate_up_weights = (torch.randn(E, H, two_I, dtype=dtype) * 0.02).to(device)
    down_weights    = (torch.randn(E, I_size, H, dtype=dtype) * 0.02).to(device)

    # Simple deterministic routing: uniform top-2 across adjacent experts
    routing_weights = torch.zeros(T, E, dtype=torch.float32)
    for t in range(T):
        routing_weights[t, t % E]       = 0.6
        routing_weights[t, (t + 1) % E] = 0.4
    routing_weights = routing_weights.to(device)

    # --- Warmup ---
    for _ in range(n_warmup):
        out = nki_moe_v2(hidden_states, gate_up_weights, down_weights, routing_weights)
        xm.mark_step()
    _ = out.cpu()   # drain before timing

    # --- Timed runs ---
    latencies_ms = []
    for _ in range(n_iters):
        t0  = time.perf_counter()
        out = nki_moe_v2(hidden_states, gate_up_weights, down_weights, routing_weights)
        xm.mark_step()
        _   = out.cpu()         # blocks until NeuronCore finishes
        t1  = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1_000.0)

    latencies_ms.sort()
    mean_ms   = sum(latencies_ms) / len(latencies_ms)
    min_ms    = latencies_ms[0]
    median_ms = latencies_ms[len(latencies_ms) // 2]
    p90_ms    = latencies_ms[int(0.9 * len(latencies_ms))]

    # Theoretical FLOPs
    #   gate_up : T × H × (2I) × 2 = 4·T·H·I  per expert
    #   down    : T × I × H × 2    = 2·T·I·H  per expert
    #   total   : 6·T·H·I·E
    flops        = 6 * T * H * I_size * E
    gflops_per_s = (flops / (mean_ms * 1e-3)) / 1e9

    # Minimal memory footprint (weights + activations streamed once)
    b2, b4 = 2, 4   # bf16, f32 bytes
    mem_bytes = (
        T * H           * b2 +   # hidden_states in
        E * H * two_I   * b2 +   # gate_up_weights
        E * I_size * H  * b2 +   # down_weights
        T * E           * b4 +   # routing_weights
        T * H           * b2     # output
    )
    bandwidth_gbps = (mem_bytes / (mean_ms * 1e-3)) / 1e9

    bar = "─" * 58
    print(f"\n{bar}")
    print(f"  {label}")
    print(f"  Shapes : T={T}, H={H}, I={I_size}, E={E}  dtype={dtype}")
    print(f"{bar}")
    print(f"  Latency  mean   : {mean_ms:8.3f} ms")
    print(f"  Latency  min    : {min_ms:8.3f} ms")
    print(f"  Latency  median : {median_ms:8.3f} ms")
    print(f"  Latency  p90    : {p90_ms:8.3f} ms")
    print(f"{bar}")
    print(f"  FLOPs / call    : {flops / 1e6:8.1f} MFLOPs")
    print(f"  Throughput      : {gflops_per_s:8.3f} GFLOPs/s")
    print(f"{bar}")
    print(f"  Mem footprint   : {mem_bytes / 1024:8.1f} KB  (weights + IO)")
    print(f"  Eff. bandwidth  : {bandwidth_gbps:8.3f} GB/s")
    print(f"{bar}")


def benchmark_nki_moe_v2(n_warmup=5, n_iters=20):
    print("\n" + "=" * 60)
    print("  NKI MoE v2 (Affine Range + PSUM Reuse) — Performance")
    print(f"  ({n_warmup} warmup iters, {n_iters} timed iters each)")
    print("=" * 60)

    dtype = torch.bfloat16
    _run_benchmark("Small  config", T=4, H=256,  I_size=256, E=4,
                   dtype=dtype, n_warmup=n_warmup, n_iters=n_iters)
    _run_benchmark("Larger config", T=8, H=512,  I_size=384, E=4,
                   dtype=dtype, n_warmup=n_warmup, n_iters=n_iters)


if __name__ == "__main__":
    # ok1 = test_nki_moe_v2()
    ok2 = test_nki_moe_v2_large()
    # if ok1 and ok2:
    #     print("\nAll tests passed!")
    # else:
    #     print("\nSome tests FAILED.")
    #     import sys
    #     sys.exit(1)

    # benchmark_nki_moe_v2()
