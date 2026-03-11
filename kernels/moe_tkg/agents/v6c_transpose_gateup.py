"""
Optimized MoE kernel — v6c: Transpose Gate+Up Matmul Layout (Full 512-wide Moving).

Built on v5a (Coalesce Gate+Up Weight DMAs).

Key optimization over v5a:
  - Swap stationary and moving in gate+up nc_matmul:
      OLD: stationary=w_tile[P,P], moving=h_tile[P,1]  -> dst[P,1]  (4 matmuls/h_t)
      NEW: stationary=h_tile[P,1], moving=w_row[P,512]  -> dst[1,512] (1 matmul/h_t)
  - This uses the full 512-wide moving dimension (was 1/512 = 0.2% utilization).
  - Reduces gate+up matmuls from num_h_tiles * num_gu_tiles to num_h_tiles per expert
    (e.g. 64 -> 16 for H=2048, I=256).
  - Post-matmul: result is [1, two_I] instead of num_gu_tiles x [P, 1], so we
    split gate/up in the free dimension, apply SiLU+mul, then use tiled nc_transpose
    (32-element chunks) to convert [1, P] slices into [P, 1] tiles for the
    (unchanged) down projection.

Implementation note on nc_transpose:
  The Scalar Engine limits nc_transpose to (32, 32) operands. To transpose
  [1, 128] -> [128, 1] we tile the operation into four [1, 32] -> [32, 1]
  sub-transposes, writing each into a sub-view of the [P, 1] destination.

Everything else (down projection, token/expert loops, output accum) is identical to v5a.

Constraints:
    T <= 128
    H % 128 == 0
    I % 128 == 0
    2*I <= 512  (moving dimension limit for nc_matmul)
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
# Kernel
# ---------------------------------------------------------------------------

@nki.jit(platform_target="trn2")
def nki_moe_v6c(
    hidden_states,        # [T, H]    bf16
    gate_up_weights,      # [E, H, 2*I] bf16
    down_weights,         # [E, I, H]   bf16
    expert_indices,       # [T, K]    int32   — top-K expert indices per token
    routing_weights_k,    # [T, K]    float32 — routing weights for the K experts
):
    """
    Sparse Token-Parallel MoE kernel (v6c) with transposed gate+up matmul.

    The gate+up matmul uses h_tile as stationary and the full [P, two_I] weight
    row as moving, producing a [1, two_I] result per h_tile in a single matmul
    (4x fewer matmul instructions than v5a for I=256).
    """
    T      = hidden_states.shape[0]
    H      = hidden_states.shape[1]
    E      = gate_up_weights.shape[0]   # noqa: F841 — kept for documentation
    two_I  = gate_up_weights.shape[2]
    I_size = two_I // 2
    K      = expert_indices.shape[1]
    P      = 128

    num_h_tiles  = H      // P   # number of P-wide hidden-dim tiles
    num_i_tiles  = I_size // P   # number of P-wide intermediate tiles

    # HBM output buffer
    output = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # Tile size for nc_transpose: hardware Scalar Engine limits to (32, 32).
    # We transpose [1, P] -> [P, 1] in TILE-sized chunks: [1, TILE] -> [TILE, 1].
    TILE = 32
    num_transpose_chunks = P // TILE  # 128 / 32 = 4 chunks per [1, P] block

    # =========================================================================
    # Hoist expert_indices into SBUF (read once).
    #   expert_idx_sb  : [pmax, K]  int32  (pmax=128, T rows meaningful)
    #   Pad to pmax so that the .ap() scalar_offset math works on a full
    #   128-partition tensor (required by DGE indirect addressing).
    # =========================================================================
    pmax = 128
    expert_idx_sb = nl.ndarray((pmax, K), dtype=expert_indices.dtype, buffer=nl.sbuf)
    nisa.memset(dst=expert_idx_sb, value=0)
    nisa.dma_copy(dst=expert_idx_sb[0:T, 0:K], src=expert_indices[0:T, 0:K])

    # =========================================================================
    # Pre-allocate PSUM tiles (reused across T and K loops).
    #
    # v6c change: gate+up uses a SINGLE [1, two_I] wide PSUM accumulator
    # instead of num_gu_tiles separate [P, 1] tiles.  The "1" partition dim
    # means we accumulate h_tile^T @ w_row across all h_tiles into one row.
    # =========================================================================

    # PSUM tile for gate+up: one wide [1, two_I] accumulator (e.g. [1, 512])
    gu_psum_wide = nl.ndarray((1, two_I), dtype=nl.float32, buffer=nl.psum)

    # PSUM tile for down: one [1, P] tile per h column block (unchanged from v5a)
    down_psum_tiles = []
    for _h_t in nl.affine_range(num_h_tiles):
        down_psum_tiles.append(
            nl.ndarray((1, P), dtype=nl.float32, buffer=nl.psum)
        )

    # =========================================================================
    # TOKEN LOOP — sequential: each token reuses the same SBUF allocations
    # =========================================================================
    for t in range(T):

        # -------------------------------------------------------------------
        # Load token t's hidden state: num_h_tiles x [P, 1] SBUF tiles.
        # hidden_states[t, h_off:h_off+P] -> h_sb[P, 1]
        # Access pattern: stride along H (inner), one row (token t).
        # -------------------------------------------------------------------
        h_tiles_sb = []
        for h_t in nl.affine_range(num_h_tiles):
            h_off = h_t * P
            h_sb  = nl.ndarray((P, 1), dtype=hidden_states.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=h_sb,
                src=hidden_states.ap(
                    pattern=[[1, P], [H, 1]],
                    offset=t * H + h_off,
                ),
            )
            h_tiles_sb.append(h_sb)

        # Output accumulator for this token: num_h_tiles x [1, P] f32 SBUF
        # Zero once; K expert contributions are accumulated here.
        out_accum = []
        for _h_t in nl.affine_range(num_h_tiles):
            tmp = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=tmp, value=0)
            out_accum.append(tmp)

        # -------------------------------------------------------------------
        # K-EXPERT LOOP — sequential (accumulates into out_accum)
        # -------------------------------------------------------------------
        for k in range(K):

            # Get expert id as scalar offset (production pattern)
            # expert_idx_sb has shape [T, K]; flat offset = t * K + k
            eid_offset = expert_idx_sb.ap(
                pattern=[[K, 1], [1, 1]],
                offset=t * K + k,
            )

            # Load routing weight for (t, k) from HBM -> partition-0 SBUF scalar.
            # Cannot use SBUF .ap() view as operand0 in tensor_scalar when
            # accessing non-zero partitions (t > 0), so we DMA-load per element.
            rw_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=rw_scalar,
                src=routing_weights_k.ap(
                    pattern=[[K, 1], [1, 1]],
                    offset=t * K + k,
                ),
            )

            # ---------------------------------------------------------------
            # Stage 1: Gate+Up projection  (v6c transposed layout)
            #   hidden[1, H] @ gate_up_w[H, 2*I] -> [1, 2*I]
            #
            #   nc_matmul computes:  dst += stationary.T @ moving
            #   v6c layout:
            #     stationary = h_tile   [P, 1]     (token hidden tile)
            #     moving     = w_row    [P, two_I]  (full gate+up weight row)
            #     result     = [1, two_I]           (h_tile^T @ w_row)
            #
            #   Accumulating across num_h_tiles h_t iterations gives the full
            #   [1, two_I] gate+up output — same math as v5a, fewer matmuls.
            # ---------------------------------------------------------------

            # Zero the wide PSUM accumulator at start of each k iteration
            nisa.memset(dst=gu_psum_wide, value=0)

            # Accumulate h_t contributions (sequential — PSUM accumulation dep)
            for h_t in range(num_h_tiles):
                h_off  = h_t * P
                h_tile = h_tiles_sb[h_t]

                # Load the entire [P, two_I] weight row for this h_t in ONE DMA
                # (same coalesced DMA as v5a — unchanged)
                w_row = nl.ndarray((P, two_I), dtype=gate_up_weights.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=w_row,
                    src=gate_up_weights.ap(
                        # Pattern: each of P partitions reads two_I contiguous elements
                        # stride = two_I between partitions (one row of H), free dim = two_I
                        pattern=[[two_I, P], [1, two_I]],
                        offset=h_off * two_I,          # start at row h_t in the [H, 2I] slice
                        scalar_offset=eid_offset,      # dynamic expert selection
                        indirect_dim=0,                 # expert dim is dim-0 of the 3D tensor
                    ),
                )

                # v6c: ONE matmul per h_t instead of num_gu_tiles matmuls.
                # stationary=h_tile[P,1], moving=w_row[P,two_I] -> dst[1,two_I]
                # This fills the full 512-wide moving dimension (was only 1 in v5a).
                nisa.nc_matmul(
                    dst=gu_psum_wide,
                    stationary=h_tile,       # [P, 1]
                    moving=w_row,            # [P, two_I]  (up to 512 wide)
                )

            # ---------------------------------------------------------------
            # Post-matmul: PSUM -> SBUF, split gate/up, SiLU+mul, transpose
            #
            # gu_psum_wide is [1, two_I] = [gate[1, I_size] | up[1, I_size]].
            # v5a had num_gu_tiles separate [P, 1] tiles; v6c has one wide tile.
            # ---------------------------------------------------------------

            # PSUM -> SBUF: copy the single [1, two_I] wide result (1 copy vs 4 in v5a)
            gu_sb_wide = nl.ndarray((1, two_I), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gu_sb_wide, src=gu_psum_wide)

            # Split gate and up halves from the wide [1, two_I] tensor.
            # Copy to separate SBUF tensors (slice views cause issues with nisa ops).
            gate_sb = nl.ndarray((1, I_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gate_sb, src=gu_sb_wide[0:1, 0:I_size])

            up_sb = nl.ndarray((1, I_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=up_sb, src=gu_sb_wide[0:1, I_size:two_I])

            # SiLU(gate): element-wise on [1, I_size] — processes all I_size elements at once
            gate_act = nl.ndarray((1, I_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=gate_act, op=nl.silu, data=gate_sb, scale=1.0)

            # SiLU(gate) * up: element-wise on [1, I_size]
            act_f32 = nl.ndarray((1, I_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=act_f32,
                data1=gate_act,
                data2=up_sb,
                op=nl.multiply,
            )

            # Cast f32 -> bf16 once for the whole [1, I_size] activation
            act_bf16 = nl.ndarray((1, I_size), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=act_bf16, src=act_f32)

            # Transpose [1, P] chunks -> [P, 1] tiles for down projection.
            # The Scalar Engine limits nc_transpose to (32, 32) operands, so we tile
            # each [1, P=128] chunk into 4 sub-chunks of [1, TILE=32] -> [TILE, 1],
            # writing each into a sub-view of the [P, 1] destination tensor.
            act_bf16_tiles = []
            for i_t in nl.affine_range(num_i_tiles):
                act_tile = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
                for c in nl.affine_range(num_transpose_chunks):
                    # Source: [1, TILE] slice from the [1, I_size] activation
                    src_slice = act_bf16[0:1, i_t * P + c * TILE : i_t * P + (c + 1) * TILE]
                    # Destination: [TILE, 1] sub-view in the [P, 1] output tile
                    dst_slice = act_tile[c * TILE:(c + 1) * TILE, 0:1]
                    nisa.nc_transpose(dst=dst_slice, data=src_slice)
                act_bf16_tiles.append(act_tile)

            # ---------------------------------------------------------------
            # Stage 2: Down projection (UNCHANGED from v5a)
            #   act[1, I] @ down_w[I, H] -> [1, H]
            #   nc_matmul layout:  dst += stationary.T @ moving
            #     stationary = act_tile  [P_i, 1]  (intermediate tile)
            #     moving     = dw_tile   [P_i, P]  (down weight tile)
            #     result     = [1, P]               stored in down_psum_tiles[h_t]
            # ---------------------------------------------------------------

            # h_t is independent -> affine_range; i_t sequential (PSUM acc dep)
            for h_t in nl.affine_range(num_h_tiles):
                h_off = h_t * P

                # Zero this h_t's PSUM before accumulating i_t contributions
                nisa.memset(dst=down_psum_tiles[h_t], value=0)

                for i_t in range(num_i_tiles):
                    i_off = i_t * P

                    dw_tile = nl.ndarray((P, P), dtype=down_weights.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=dw_tile,
                        src=down_weights.ap(
                            pattern=[[H, P], [1, P]],
                            offset=i_off * H + h_off,
                            scalar_offset=eid_offset,
                            indirect_dim=0,
                        ),
                    )

                    nisa.nc_matmul(
                        dst=down_psum_tiles[h_t],
                        stationary=act_bf16_tiles[i_t],
                        moving=dw_tile,
                    )

            # Scale by routing weight and accumulate into out_accum
            for h_t in nl.affine_range(num_h_tiles):
                down_sb = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=down_sb, src=down_psum_tiles[h_t])

                # Scale by rw_tk: [1, 1] broadcasts across free dim P
                scaled = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=scaled,
                    data=down_sb,
                    op0=nl.multiply,
                    operand0=rw_scalar,
                )

                # Accumulate expert k's contribution
                nisa.tensor_tensor(
                    dst=out_accum[h_t],
                    data1=out_accum[h_t],
                    data2=scaled,
                    op=nl.add,
                )

        # -------------------------------------------------------------------
        # Store token t's output: out_accum[h_t] -> output[t, h_off:h_off+P]
        # -------------------------------------------------------------------
        for h_t in nl.affine_range(num_h_tiles):
            h_off    = h_t * P
            out_cast = nl.ndarray((1, P), dtype=hidden_states.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_cast, src=out_accum[h_t])
            nisa.dma_copy(
                dst=output.ap(
                    pattern=[[H, 1], [1, P]],
                    offset=t * H + h_off,
                ),
                src=out_cast,
            )

    return output


# ---------------------------------------------------------------------------
# PyTorch sparse reference (matches kernel's sparse K-expert dispatch)
# ---------------------------------------------------------------------------

def pytorch_moe_sparse_reference(
    hidden_states,      # [T, H] bf16
    gate_up_weights,    # [E, H, 2*I] bf16
    down_weights,       # [E, I, H]   bf16
    expert_indices,     # [T, K] int64
    routing_weights_k,  # [T, K] float32
):
    T, H   = hidden_states.shape
    I_size = gate_up_weights.shape[2] // 2
    K      = expert_indices.shape[1]

    output = torch.zeros(T, H, dtype=torch.float32)
    hs_f32 = hidden_states.float()

    for t in range(T):
        for k in range(K):
            eid = expert_indices[t, k].item()
            rw  = routing_weights_k[t, k].item()

            gu_w   = gate_up_weights[eid].float()   # [H, 2*I]
            d_w    = down_weights[eid].float()       # [I, H]

            gu_out = hs_f32[t:t+1] @ gu_w           # [1, 2*I]
            gate   = gu_out[:, :I_size]
            up     = gu_out[:, I_size:]
            act    = torch.nn.functional.silu(gate) * up   # [1, I]
            down   = act @ d_w                       # [1, H]

            output[t] += rw * down[0]

    return output.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_sparse_inputs(T, H, I_size, E, K, dtype=torch.bfloat16, seed=42):
    """Generate random sparse MoE inputs (no expert collision per token)."""
    torch.manual_seed(seed)
    hidden_states   = torch.randn(T, H, dtype=dtype)
    gate_up_weights = torch.randn(E, H, 2 * I_size, dtype=dtype) * 0.02
    down_weights    = torch.randn(E, I_size, H, dtype=dtype) * 0.02

    # Sample K distinct experts per token without replacement
    expert_indices = torch.zeros(T, K, dtype=torch.int32)
    for t in range(T):
        perm = torch.randperm(E)[:K]
        expert_indices[t] = perm

    expert_weights  = torch.softmax(torch.randn(T, K, dtype=torch.float32), dim=-1)

    return hidden_states, gate_up_weights, down_weights, expert_indices, expert_weights


def test_nki_moe_v6c_qwen3_t1():
    """Qwen3 shapes, T=1: H=2048, I=256, E=128, K=8."""
    import torch_xla.core.xla_model as xm

    print("\n" + "=" * 70)
    print("nki_moe_v6c — Qwen3 T=1 (H=2048, I=256, E=128, K=8)")
    print("=" * 70)

    device = xm.xla_device()
    T, H, I_size, E, K = 1, 2048, 256, 128, 8

    hidden, gu_w, d_w, eidx, rw_k = _make_sparse_inputs(T, H, I_size, E, K, seed=123)

    print(f"  Shapes: T={T}, H={H}, I={I_size}, E={E}, K={K}")
    print(f"  num_h_tiles={H//128}, num_i_tiles={I_size//128}, two_I={2*I_size}")

    print("\n[1/3] Reference (PyTorch CPU)...")
    ref = pytorch_moe_sparse_reference(hidden, gu_w, d_w, eidx, rw_k)

    print("[2/3] NKI v6c kernel on Trainium...")
    nki_out = nki_moe_v6c(
        hidden.to(device),
        gu_w.to(device),
        d_w.to(device),
        eidx.to(device),
        rw_k.to(device),
    ).cpu()

    print("[3/3] Comparing...")
    diff     = torch.abs(ref.float() - nki_out.float())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\n  Output shape : {nki_out.shape}")
    print(f"  Max  |diff|  : {max_diff:.6e}")
    print(f"  Mean |diff|  : {mean_diff:.6e}")

    threshold = 0.05
    if max_diff < threshold:
        print(f"\n  max_diff={max_diff:.2e}  PASS")
        print("=" * 70)
        return True
    else:
        print(f"\n  FAIL — max_diff {max_diff:.4e} > {threshold:.4e}")
        return False


def test_nki_moe_v6c():
    """Run all v6c correctness tests."""
    results = []
    results.append(("qwen3 T=1 (H=2048,I=256,E=128,K=8)", test_nki_moe_v6c_qwen3_t1()))

    print("\n" + "=" * 70)
    print("Summary:")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        all_ok = all_ok and ok
    print("=" * 70)
    if all_ok:
        print("All v6c tests PASSED.")
    else:
        print("Some v6c tests FAILED.")
    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.perf_counter()
    ok = test_nki_moe_v6c()
    t1 = time.perf_counter()
    print(f"\nTotal time: {t1 - t0:.1f}s")
    import sys
    sys.exit(0 if ok else 1)
