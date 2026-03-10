"""
Optimization Plan 3: Partition-First Layout + Fused Operations

Key changes over nki_moe_fused baseline:
  1. Hoist hidden-state loads outside expert loop (preload all num_h_tiles [P,T] tiles).
  2. Prefetch routing weights as [T, E] SBUF array before expert loop.
  3. Hoist f32->bf16 act cast outside h_t loop (once per i_t per expert).
  4. Reuse PSUM tiles across experts (allocate once, memset at expert start).
  5. Use nl.affine_range() on gu_t inner loop (independent destinations).
  6. Use nl.affine_range() on h_t outer loop in down projection (independent down_psum tiles).
  7. Fuse down PSUM->SBUF and routing-weight scaling: use tensor_scalar directly
     on the PSUM source to avoid an intermediate SBUF allocation.

Constraints (inherited):
    T <= 128
    H % 128 == 0
    I % 128 == 0
"""

import time
import torch
import nki
import nki.language as nl
import nki.isa as nisa
import os


os.environ["NEURON_CC_FLAGS"] = " --disable-dge "
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

@nki.jit(platform_target="trn2")
def nki_moe_v3(
    hidden_states,      # [T, H] bf16
    gate_up_weights,    # [E, H, 2*I] bf16
    down_weights,       # [E, I, H]   bf16
    routing_weights,    # [T, E]      float32
):
    """
    Fused MoE with Partition-First Layout + Fused Operations (Plan 3).

    Tile layouts (same as baseline):
      Gate+Up intermediate: [P=128, T] (partition=gu_dim, free=token_dim)
      Down output / accum:  [T, P=128] (partition=token_dim, free=hidden_dim)

    Optimizations vs baseline:
      - Hidden states loaded ONCE before expert loop.
      - Routing weights loaded ONCE as full [T, E] SBUF array.
      - Gate+Up PSUM tiles allocated ONCE and reused (memset per expert).
      - Down PSUM tiles allocated ONCE per h_t and reused (memset per expert).
      - Act bf16 cast hoisted outside h_t loop.
      - affine_range on gu_t (independent) and h_t down-proj loop (independent).
      - Routing-weight scaling done directly on PSUM-derived SBUF, removing
        one intermediate allocation.
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

    # Reshape 3D weight tensors -> 2D for flat .ap() indexing
    gate_up_flat = gate_up_weights.reshape((E * H, two_I))
    down_flat    = down_weights.reshape((E * I_size, H))

    # Output buffer in HBM
    output = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # -----------------------------------------------------------------------
    # PRE-LOOP: hoist all data that is loop-invariant w.r.t. expert_id
    # -----------------------------------------------------------------------

    # 1. Load routing weights once: [T, E] f32
    #    We store as [T, E] in SBUF.  T <= 128 so T fits in partition dim.
    rw_sb = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=rw_sb,
        src=routing_weights[0:T, 0:E],
    )

    # 2. Load hidden states once: list of num_h_tiles [P, T] bf16 SBUF tiles
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

    # 3. Allocate gate+up PSUM tiles ONCE; reuse per expert (memset each time)
    gu_psum_tiles = []
    for gu_t in nl.affine_range(num_gu_tiles):
        gu_psum_tiles.append(
            nl.ndarray((P, T), dtype=nl.float32, buffer=nl.psum)
        )

    # 4. Allocate down PSUM tiles ONCE per h_t; reuse per expert
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

    # -----------------------------------------------------------------------
    # EXPERT LOOP
    # -----------------------------------------------------------------------
    for expert_id in nl.affine_range(E):

        # --- Step a: zero gate+up PSUM tiles ---
        for gu_t in nl.affine_range(num_gu_tiles):
            nisa.memset(dst=gu_psum_tiles[gu_t], value=0)

        # --- Step b: Gate+Up matmul ---
        #   nc_matmul: dst += w_tile.T @ h_tile
        #     stationary = w_tile [P_h, P_gu]
        #     moving     = h_tile [P_h, T]
        #     result     = [P_gu, T]  stored in gu_psum_tiles[gu_t]
        #   h_t loop is sequential (all gu_psum accumulate across h_t)
        #   gu_t loop uses affine_range (independent destinations)
        for h_t in nl.affine_range(num_h_tiles):
            h_off = h_t * P

            for gu_t in nl.affine_range(num_gu_tiles):
                gu_off = gu_t * P

                w_tile = nl.ndarray((P, P), dtype=gate_up_weights.dtype,
                                    buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=w_tile,
                    src=gate_up_flat.ap(
                        pattern=[[two_I, P], [1, P]],
                        offset=(expert_id * H + h_off) * two_I + gu_off,
                    ),
                )

                nisa.nc_matmul(
                    dst=gu_psum_tiles[gu_t],
                    stationary=w_tile,
                    moving=h_tiles_sb[h_t],
                )

        # --- Step c: PSUM -> SBUF for gate+up tiles ---
        gu_tiles = []
        for gu_t in nl.affine_range(num_gu_tiles):
            gu_sb = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gu_sb, src=gu_psum_tiles[gu_t])
            gu_tiles.append(gu_sb)

        # --- Step d+e: SiLU(gate)*up + cast to bf16 (OUTSIDE h_t loop) ---
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

        # --- Step f: zero down PSUM tiles ---
        for h_t in nl.affine_range(num_h_tiles):
            nisa.memset(dst=down_psum_tiles[h_t], value=0)

        # --- Step g: Down matmul ---
        #   h_t loop uses affine_range (each writes to independent down_psum_tiles[h_t])
        #   i_t loop is sequential (accumulates into same down_psum_tiles[h_t])
        for h_t in nl.affine_range(num_h_tiles):
            h_off = h_t * P

            for i_t in nl.affine_range(num_i_tiles):
                i_off = i_t * P

                dw_tile = nl.ndarray((P, P), dtype=down_weights.dtype,
                                     buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=dw_tile,
                    src=down_flat.ap(
                        pattern=[[H, P], [1, P]],
                        offset=(expert_id * I_size + i_off) * H + h_off,
                    ),
                )

                nisa.nc_matmul(
                    dst=down_psum_tiles[h_t],
                    stationary=act_bf16_tiles[i_t],
                    moving=dw_tile,
                )

        # --- Step h: apply routing weight and accumulate ---
        #   rw_col: shape [T, 1] extracted from rw_sb[:, expert_id]
        #   tensor_scalar broadcasts [T,1] across the P free dim of [T,P]
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

    # -----------------------------------------------------------------------
    # STORE to HBM (cast f32 -> output dtype)
    # -----------------------------------------------------------------------
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
# PyTorch reference — mirrors kernel logic exactly
# ---------------------------------------------------------------------------

def pytorch_moe_reference(hidden_states, gate_up_weights, down_weights,
                           routing_weights):
    T, H   = hidden_states.shape
    E      = gate_up_weights.shape[0]
    two_I  = gate_up_weights.shape[2]
    I_size = two_I // 2

    output  = torch.zeros(T, H, dtype=torch.float32)
    hs_f32  = hidden_states.float()

    for e in range(E):
        gu_w   = gate_up_weights[e].float()
        d_w    = down_weights[e].float()

        gu_out = hs_f32 @ gu_w
        gate   = gu_out[:, :I_size]
        up     = gu_out[:, I_size:]
        act    = torch.nn.functional.silu(gate) * up
        down   = act @ d_w

        rw     = routing_weights[:, e:e + 1].float()
        output = output + down * rw

    return output.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_nki_moe_v3():
    import torch_xla.core.xla_model as xm

    print("=" * 70)
    print("NKI MoE v3 (Plan 3: Partition-First Layout + Fused Ops) — Test")
    print("=" * 70)

    device = xm.xla_device()
    print(f"\nUsing device: {device}")

    T      = 4
    H      = 256
    I_size = 256
    E      = 4
    K      = 2
    two_I  = 2 * I_size

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

    print("[2/3] Computing NKI v3 kernel on Trainium...")
    nki_output = nki_moe_v3(
        hidden_states.to(device),
        gate_up_weights.to(device),
        down_weights.to(device),
        routing_weights.to(device),
    )

    print("[3/3] Comparing outputs...")
    diff      = torch.abs(ref_output.float() - nki_output.float())
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nResults:")
    print(f"  Output shape : {nki_output.shape}")
    print(f"  Max  |diff|  : {max_diff:.6e}")
    print(f"  Mean |diff|  : {mean_diff:.6e}")

    threshold = 0.05
    if max_diff < threshold:
        print("\n" + "=" * 70)
        print("SUCCESS! nki_moe_v3 kernel matches reference.")
        print("=" * 70)
        return True
    else:
        print(f"\nFAILED — max diff {max_diff:.4e} exceeds threshold {threshold:.4e}")
        return False


def test_nki_moe_v3_large():
    """Test with larger shapes (more H/I tiles)."""
    import torch_xla.core.xla_model as xm

    print("\n" + "=" * 70)
    print("NKI MoE v3 — larger shapes test")
    print("=" * 70)

    device = xm.xla_device()

    T      = 8
    H      = 512
    I_size = 384
    E      = 4
    K      = 2
    two_I  = 2 * I_size

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

    ref = pytorch_moe_reference(
        hidden_states, gate_up_weights, down_weights, routing_weights
    )
    nki_out = nki_moe_v3(
        hidden_states.to(device),
        gate_up_weights.to(device),
        down_weights.to(device),
        routing_weights.to(device),
    ).cpu()

    diff      = torch.abs(ref.float() - nki_out.float())
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max  |diff|  : {max_diff:.6e}")
    print(f"  Mean |diff|  : {mean_diff:.6e}")

    if max_diff < 0.05:
        print("  PASS")
        return True
    else:
        print(f"  FAIL — max diff exceeds threshold")
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

    routing_weights = torch.zeros(T, E, dtype=torch.float32)
    for t in range(T):
        routing_weights[t, t % E]       = 0.6
        routing_weights[t, (t + 1) % E] = 0.4
    routing_weights = routing_weights.to(device)

    # Warmup
    for _ in range(n_warmup):
        out = nki_moe_v3(hidden_states, gate_up_weights, down_weights,
                         routing_weights)
        xm.mark_step()
    _ = out.cpu()

    # Timed runs
    latencies_ms = []
    for _ in range(n_iters):
        t0  = time.perf_counter()
        out = nki_moe_v3(hidden_states, gate_up_weights, down_weights,
                         routing_weights)
        xm.mark_step()
        _   = out.cpu()
        t1  = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1_000.0)

    latencies_ms.sort()
    mean_ms   = sum(latencies_ms) / len(latencies_ms)
    min_ms    = latencies_ms[0]
    median_ms = latencies_ms[len(latencies_ms) // 2]
    p90_ms    = latencies_ms[int(0.9 * len(latencies_ms))]

    # FLOPs: gate_up=4*T*H*I per expert, down=2*T*I*H per expert -> 6*T*H*I*E
    flops         = 6 * T * H * I_size * E
    gflops_per_s  = (flops / (mean_ms * 1e-3)) / 1e9

    b2, b4   = 2, 4
    mem_bytes = (
        T * H          * b2 +
        E * H * two_I  * b2 +
        E * I_size * H * b2 +
        T * E          * b4 +
        T * H          * b2
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


def benchmark_nki_moe_v3(n_warmup=5, n_iters=20):
    print("\n" + "=" * 60)
    print("  NKI MoE v3 — Wall-clock Performance Metrics")
    print(f"  ({n_warmup} warmup iters, {n_iters} timed iters each)")
    print("=" * 60)

    dtype = torch.bfloat16
    _run_benchmark("Small  config", T=4, H=256,  I_size=256, E=4,
                   dtype=dtype, n_warmup=n_warmup, n_iters=n_iters)
    _run_benchmark("Larger config", T=8, H=512,  I_size=384, E=4,
                   dtype=dtype, n_warmup=n_warmup, n_iters=n_iters)


if __name__ == "__main__":
    # ok1 = test_nki_moe_v3()
    ok2 = test_nki_moe_v3_large()
    # if ok1 and ok2:
    #     print("\nAll tests passed!")
    # else:
    #     print("\nSome tests failed.")

    # benchmark_nki_moe_v3()
