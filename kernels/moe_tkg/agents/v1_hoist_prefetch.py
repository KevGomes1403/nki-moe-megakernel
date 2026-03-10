"""
Optimized MoE kernel — Plan 1: Hoist Hidden State + Routing Weight Prefetch.

Three optimizations over the baseline nki_moe_fused:

  Fix 1: Hoist hidden_states load outside the expert loop (eliminate E×
          redundant HBM reads — the baseline re-reads [T,H] bf16 for every
          expert).

  Fix 2: Prefetch all routing weights into SBUF before the expert loop
          (baseline issues E separate tiny HBM fetches, one per expert).

  Fix 3: Hoist f32→bf16 activation cast outside the h_t loop (baseline
          re-casts act_tiles[i_t] for every h_t even though the source
          doesn't change).

Constraints (same as baseline):
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

os.environ["NEURON_CC_FLAGS"] = " --disable-dge "
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"


# ---------------------------------------------------------------------------
# Optimised kernel
# ---------------------------------------------------------------------------

@nki.jit(platform_target="trn2")
def nki_moe_v1(
    hidden_states,      # [T, H] bf16
    gate_up_weights,    # [E, H, 2*I] bf16
    down_weights,       # [E, I, H] bf16
    routing_weights,    # [T, E] float32
):
    """
    Fused MoE with hoisted hidden-state loads, prefetched routing weights,
    and hoisted activation cast.

    Tile layouts match baseline:
      Gate+Up intermediate : [P=128, T]   (partition=intermediate_dim)
      Down output / accum  : [T,   P=128] (partition=token_dim)
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

    # Reshape 3D weight tensors → 2D for flat .ap() indexing
    gate_up_flat = gate_up_weights.reshape((E * H, two_I))
    down_flat    = down_weights.reshape((E * I_size, H))

    # Output tensor in HBM
    output = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # Output accumulator tiles in SBUF [T, P] — one per H chunk, zeroed once
    out_accum = []
    for _ in range(num_h_tiles):
        tmp = nl.ndarray((T, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=tmp, value=0)
        out_accum.append(tmp)

    # =========================================================================
    # FIX 1 — Hoist hidden_states load: read [T, H] from HBM exactly once.
    #
    # We store num_h_tiles separate [P, T] SBUF buffers (transposed view).
    # Inside the expert loop we just reference h_tiles_sb[h_t].
    # =========================================================================
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

    # =========================================================================
    # FIX 2 — Prefetch all routing weights into SBUF before expert loop.
    #
    # SBUF partition dim must be 128; T <= 128, so we use P_rw=128 and
    # only fill rows [0:T].  We zero the full tile first for safety.
    # =========================================================================
    P_rw  = 128   # >= T by constraint
    rw_sb = nl.ndarray((P_rw, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=rw_sb, value=0.0)
    nisa.dma_copy(
        dst=rw_sb[0:T, 0:E],
        src=routing_weights[0:T, 0:E],
    )

    # =========================================================================
    # Expert loop
    # =========================================================================
    for expert_id in nl.affine_range(E):

        # Routing weight for this expert: slice [T, 1] from SBUF prefetch.
        # rw_sb has shape [P_rw=128, E]; we need a [T, 1] view for
        # tensor_scalar broadcasting.  We copy the relevant column into a
        # dedicated [T, 1] SBUF tile so the shapes are unambiguous.
        rw = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=rw,
            src=rw_sb[0:T, expert_id:expert_id + 1],
        )

        # =====================================================================
        # Stage 1: Gate+Up Projection
        #   hidden[T,H] @ gate_up_w[H, 2*I] → [T, 2*I]
        #   nc_matmul: dst += stationary.T @ moving
        #     stationary = w_tile [P_h, P_gu]  moving = h_tile [P_h, T]
        #     result     = [P_gu, T]
        # =====================================================================
        gu_psum_tiles = []
        for gu_t_ in nl.affine_range(num_gu_tiles):
            gu_psum_tiles.append(
                nl.zeros((P, T), dtype=nl.float32, buffer=nl.psum,
                         name=f"gu_psum_e{expert_id}_t{gu_t_}")
            )

        for h_t in nl.affine_range(num_h_tiles):
            h_off  = h_t * P
            h_tile = h_tiles_sb[h_t]   # FIX 1: reuse pre-loaded tile

            for gu_t in nl.affine_range(num_gu_tiles):
                gu_off = gu_t * P

                w_tile = nl.ndarray((P, P), dtype=gate_up_weights.dtype, buffer=nl.sbuf)
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
                    moving=h_tile,
                )

        # PSUM → SBUF for each gate+up tile
        gu_tiles = []
        for gu_t in nl.affine_range(num_gu_tiles):
            gu_sb = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gu_sb, src=gu_psum_tiles[gu_t])
            gu_tiles.append(gu_sb)

        # =====================================================================
        # Stage 2: SiLU(gate) * up
        #   gate = gu_tiles[0 : num_i_tiles]       — first I columns
        #   up   = gu_tiles[num_i_tiles : end]      — last  I columns
        # =====================================================================
        act_tiles = []
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

        # =====================================================================
        # FIX 3 — Hoist f32→bf16 activation cast outside the h_t loop.
        #
        # act_tiles[i_t] doesn't change across h_t; casting once avoids
        # num_h_tiles redundant tensor_copy ops per expert per i_t.
        # =====================================================================
        act_bf16_tiles = []
        for i_t in nl.affine_range(num_i_tiles):
            act_bf16 = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=act_bf16, src=act_tiles[i_t])
            act_bf16_tiles.append(act_bf16)

        # =====================================================================
        # Stage 3: Down Projection
        #   act[T, I] @ down_w[I, H] → [T, H]
        #   nc_matmul: dst += stationary.T @ moving
        #     stationary = act_tile [P_i, T]   moving = dw_tile [P_i, P_h]
        #     result     = [T, P_h]
        # =====================================================================
        for h_t in nl.affine_range(num_h_tiles):
            h_off     = h_t * P
            down_psum = nl.zeros((T, P), dtype=nl.float32, buffer=nl.psum,
                                 name=f"down_psum_e{expert_id}_h{h_t}")

            for i_t in nl.affine_range(num_i_tiles):
                i_off = i_t * P

                # FIX 3: use pre-cast bf16 tile (no redundant cast here)
                act_bf16 = act_bf16_tiles[i_t]

                dw_tile = nl.ndarray((P, P), dtype=down_weights.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=dw_tile,
                    src=down_flat.ap(
                        pattern=[[H, P], [1, P]],
                        offset=(expert_id * I_size + i_off) * H + h_off,
                    ),
                )

                nisa.nc_matmul(
                    dst=down_psum,
                    stationary=act_bf16,
                    moving=dw_tile,
                )

            # PSUM → SBUF
            down_sb = nl.ndarray((T, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=down_sb, src=down_psum)

            # Scale by routing weight — [T, 1] broadcasts across free dim
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

    # Store output to HBM (cast f32 → bf16)
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
# PyTorch reference — unchanged from baseline
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
# Tests
# ---------------------------------------------------------------------------

def test_nki_moe_v1():
    import torch_xla.core.xla_model as xm

    print("=" * 70)
    print("nki_moe_v1 — Correctness Test (T=4, H=256, I=256, E=4)")
    print("=" * 70)

    device = xm.xla_device()
    print(f"\nUsing device: {device}")

    T, H, I_size, E, K = 4, 256, 256, 4, 2
    two_I = 2 * I_size

    print(f"Shapes: T={T}, H={H}, I={I_size}, E={E}, K={K}")

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
            routing_weights[t, expert_indices[t, k].item()] += expert_weights[t, k].item()

    print(f"  routing_weights row 0: {routing_weights[0].tolist()}")

    print("\n[1/3] Computing reference (PyTorch CPU)...")
    ref = pytorch_moe_reference(hidden_states, gate_up_weights, down_weights,
                                routing_weights)

    print("[2/3] Computing nki_moe_v1 on Trainium...")
    nki_out = nki_moe_v1(
        hidden_states.to(device),
        gate_up_weights.to(device),
        down_weights.to(device),
        routing_weights.to(device),
    ).cpu()

    print("[3/3] Comparing outputs...")
    diff     = torch.abs(ref.float() - nki_out.float())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nResults:")
    print(f"  Output shape : {nki_out.shape}")
    print(f"  Max  |diff|  : {max_diff:.6e}")
    print(f"  Mean |diff|  : {mean_diff:.6e}")

    threshold = 0.05
    if max_diff < threshold:
        print("\n" + "=" * 70)
        print("SUCCESS! nki_moe_v1 matches reference.")
        print("=" * 70)
        return True
    else:
        print(f"\nFAILED — max diff {max_diff:.4e} exceeds threshold {threshold:.4e}")
        return False


def test_nki_moe_v1_large():
    import torch_xla.core.xla_model as xm

    print("\n" + "=" * 70)
    print("nki_moe_v1 — Correctness Test — large shapes (T=8, H=512, I=384, E=4)")
    print("=" * 70)

    device = xm.xla_device()

    T, H, I_size, E, K = 1, 2048, 384, 128, 8
    two_I = 2 * I_size

    print(f"Shapes: T={T}, H={H}, I={I_size}, E={E}, K={K}")

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
            routing_weights[t, expert_indices[t, k].item()] += expert_weights[t, k].item()

    ref = pytorch_moe_reference(hidden_states, gate_up_weights, down_weights,
                                routing_weights)
    nki_out = nki_moe_v1(
        hidden_states.to(device),
        gate_up_weights.to(device),
        down_weights.to(device),
        routing_weights.to(device),
    ).cpu()

    diff     = torch.abs(ref.float() - nki_out.float())
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

    routing_weights = torch.zeros(T, E, dtype=torch.float32)
    for t in range(T):
        routing_weights[t, t % E]       = 0.6
        routing_weights[t, (t + 1) % E] = 0.4
    routing_weights = routing_weights.to(device)

    # Warmup
    for _ in range(n_warmup):
        out = nki_moe_v1(hidden_states, gate_up_weights, down_weights, routing_weights)
        xm.mark_step()
    _ = out.cpu()

    # Timed runs
    latencies_ms = []
    for _ in range(n_iters):
        t0  = time.perf_counter()
        out = nki_moe_v1(hidden_states, gate_up_weights, down_weights, routing_weights)
        xm.mark_step()
        _   = out.cpu()
        t1  = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1_000.0)

    latencies_ms.sort()
    mean_ms   = sum(latencies_ms) / len(latencies_ms)
    min_ms    = latencies_ms[0]
    median_ms = latencies_ms[len(latencies_ms) // 2]
    p90_ms    = latencies_ms[int(0.9 * len(latencies_ms))]

    # Theoretical FLOPs: gate_up 4·T·H·I + down 2·T·H·I per expert
    flops        = 6 * T * H * I_size * E
    gflops_per_s = (flops / (mean_ms * 1e-3)) / 1e9

    # Memory footprint (weights + IO read once each)
    b2, b4 = 2, 4
    mem_bytes = (
        T * H          * b2 +   # hidden_states in
        E * H * two_I  * b2 +   # gate_up_weights
        E * I_size * H * b2 +   # down_weights
        T * E          * b4 +   # routing_weights
        T * H          * b2     # output
    )
    bandwidth_gbps = (mem_bytes / (mean_ms * 1e-3)) / 1e9

    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {label}  [nki_moe_v1]")
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
    print(f"  Note: v1 eliminates E={E}× redundant HBM hidden-state reads,")
    print(f"        prefetches routing weights in one SBUF load, and hoists")
    print(f"        bf16 cast outside inner h_t loop — expected lower latency")
    print(f"        vs baseline, especially as E and H grow.")
    print(f"{bar}")


def benchmark_nki_moe_v1(n_warmup=5, n_iters=20):
    print("\n" + "=" * 62)
    print("  nki_moe_v1 — Wall-clock Performance Metrics")
    print(f"  ({n_warmup} warmup iters, {n_iters} timed iters each)")
    print("=" * 62)

    dtype = torch.bfloat16
    _run_benchmark("Small  config", T=4, H=256,  I_size=256, E=4,
                   dtype=dtype, n_warmup=n_warmup, n_iters=n_iters)
    _run_benchmark("Larger config", T=8, H=512,  I_size=384, E=4,
                   dtype=dtype, n_warmup=n_warmup, n_iters=n_iters)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ok1 = test_nki_moe_v1()
    ok2 = test_nki_moe_v1_large()

    # if ok1 and ok2:
    #     print("\nAll correctness tests passed!")
    # else:
    #     print("\nSome tests FAILED.")

    # benchmark_nki_moe_v1()
