"""
Optimized MoE kernel — v7b: Pre-load All Gate+Up Weight Rows Before Matmul Loop.

Built on v6b (Coalesce Down Weight DMAs).

Key optimization over v6b:
  - Gate+up weight pre-loading: instead of loading the [P, two_I] weight row
    inside the sequential h_t matmul loop (which serializes DMA with compute),
    we pre-load ALL num_h_tiles gate+up weight rows via affine_range BEFORE
    the h_t matmul loop.  This lets the compiler schedule all 16 DMAs as a
    batch, overlapping them with prior computation.
  - The matmul loop itself becomes DMA-free — all data is already in SBUF.

Gate+up DMA strategy:
  - Old (v6b): inside sequential h_t loop, loads [P, two_I] per h_t (1 DMA
    per h_t iteration, serialized with PSUM accumulation)
  - New (v7b): pre-loads ALL num_h_tiles [P, two_I] rows via affine_range
    BEFORE the h_t loop, then indexes gu_rows[h_t] for each matmul.

Everything else is identical to v6b — down weight coalescing, activation,
expert/routing weight loading, output store.

SBUF budget (Qwen3 shapes H=2048, I=256):
  - Gate+up rows: 16 × [128, 512] bf16 = 16 × 128K = 2MB
  - Down rows:    2 × [128, 2048] bf16 = 2 × 512K  = 1MB
  - Hidden tiles: 16 × [128, 1] bf16   ≈ 4KB
  - Misc (act, accum, scalars)          ≈ 0.1MB
  - Total ≈ 3.1MB, well within 24MB SBUF limit.

Constraints:
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
# Kernel
# ---------------------------------------------------------------------------

@nki.jit(platform_target="trn2")
def nki_moe_v7b(
    hidden_states,        # [T, H]    bf16
    gate_up_weights,      # [E, H, 2*I] bf16
    down_weights,         # [E, I, H]   bf16
    expert_indices,       # [T, K]    int32   — top-K expert indices per token
    routing_weights_k,    # [T, K]    float32 — routing weights for the K experts
):
    """
    Sparse Token-Parallel MoE kernel (v7b) with pre-loaded gate+up rows
    and coalesced down DMAs.

    Gate+up: pre-loads ALL [P, two_I] rows for all h_tiles via affine_range
             BEFORE the sequential matmul loop (new in v7b).
    Down:    loads full [P, H] row per i_t in a single DMA (from v6b),
             then slices [P, P] per h_t for each matmul.
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
    num_gu_tiles = two_I  // P   # num_i_tiles * 2 (gate + up)

    # Keep 3D weight tensors — use .ap() with scalar_offset/indirect_dim
    # for dynamic expert selection (hardware computes eid * stride automatically).

    # HBM output buffer
    output = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

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
    #   gu_psum_tiles[gu_t]  : [P, 1] float32 psum  (T=1 per token slice)
    #   down_psum            : [1, P] float32 psum   (T=1 per token slice)
    #
    # We process one token at a time so the partition = P (token in free dim).
    # For a single token: gate+up result is [P_gu, 1], down result is [1, P_h].
    # =========================================================================
    # PSUM tile for gate+up: one [P, 1] tile per gu column block
    gu_psum_tiles = []
    for _gu_t in nl.affine_range(num_gu_tiles):
        gu_psum_tiles.append(
            nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
        )

    # PSUM tile for down: one [1, P] tile per h column block
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
        # Load token t's hidden state: num_h_tiles × [P, 1] SBUF tiles.
        # hidden_states[t, h_off:h_off+P] → h_sb[P, 1]
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

        # Output accumulator for this token: num_h_tiles × [1, P] f32 SBUF
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

            # Load routing weight for (t, k) from HBM → partition-0 SBUF scalar.
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
            # Stage 1: Gate+Up projection  (v7b — pre-loaded weight rows)
            #   hidden[1, H] @ gate_up_w[H, 2*I] → [1, 2*I]
            #   nc_matmul layout:  dst += stationary.T @ moving
            #     stationary = w_tile   [P_h, P_gu]  (weight row tile)
            #     moving     = h_tile   [P_h, 1]     (token hidden tile)
            #     result     = [P_gu, 1]              stored in gu_psum_tiles[gu_t]
            #
            # v7b change: pre-load ALL num_h_tiles [P, two_I] gate+up rows
            # via affine_range BEFORE the sequential h_t matmul loop.
            # This allows the compiler to batch all 16 DMAs together,
            # enabling overlap with prior computation (e.g., previous expert's
            # down projection / scaling).  The matmul loop becomes DMA-free.
            # ---------------------------------------------------------------

            # Zero all gu PSUM tiles at start of each k iteration
            for gu_t in nl.affine_range(num_gu_tiles):
                nisa.memset(dst=gu_psum_tiles[gu_t], value=0)

            # ==============================================================
            # v7b: Pre-load ALL gate+up weight rows for this expert.
            # One [P, two_I] DMA per h_tile, issued via affine_range so the
            # compiler sees them as independent and can batch/overlap them.
            # gate_up_weights shape: [E, H, 2*I]
            # For expert eid, h_tile h_t: rows [h_off:h_off+P], all 2*I cols.
            # ==============================================================
            gu_rows = []
            for h_t in nl.affine_range(num_h_tiles):
                h_off = h_t * P
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
                gu_rows.append(w_row)

            # Matmul loop — no DMA inside, all gate+up data already in SBUF.
            # h_t is sequential because each h_t accumulates into the same
            # gu_psum_tiles[gu_t] (PSUM accumulation dependency).
            for h_t in range(num_h_tiles):
                h_tile = h_tiles_sb[h_t]

                # Slice the pre-loaded [P, two_I] row for each gu_t's [P, P] tile
                for gu_t in nl.affine_range(num_gu_tiles):
                    gu_off = gu_t * P
                    nisa.nc_matmul(
                        dst=gu_psum_tiles[gu_t],
                        stationary=gu_rows[h_t][0:P, gu_off:gu_off+P],  # slice from pre-loaded row
                        moving=h_tile,
                    )

            # PSUM → SBUF for gate+up, then SiLU + multiply + cast to bf16
            act_bf16_tiles = []
            for i_t in nl.affine_range(num_i_tiles):
                # Gate half: gu_psum_tiles[i_t]      → [P, 1]
                # Up   half: gu_psum_tiles[i_t + num_i_tiles] → [P, 1]
                gate_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=gate_sb, src=gu_psum_tiles[i_t])

                up_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=up_sb, src=gu_psum_tiles[num_i_tiles + i_t])

                # SiLU(gate)
                gate_act = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=gate_act, op=nl.silu, data=gate_sb, scale=1.0)

                # SiLU(gate) * up
                act_f32 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=act_f32,
                    data1=gate_act,
                    data2=up_sb,
                    op=nl.multiply,
                )

                # Cast f32 → bf16 once (hoisted outside down loop)
                act_bf16 = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=act_bf16, src=act_f32)
                act_bf16_tiles.append(act_bf16)

            # ---------------------------------------------------------------
            # Stage 2: Down projection  (v6b — coalesced DMA, unchanged)
            #   act[1, I] @ down_w[I, H] → [1, H]
            #   nc_matmul layout:  dst += stationary.T @ moving
            #     stationary = act_tile  [P_i, 1]  (intermediate tile)
            #     moving     = dw_tile   [P_i, P]  (down weight tile)
            #     result     = [1, P]               stored in down_psum_tiles[h_t]
            #
            # Pre-load all num_i_tiles [P, H] down weight rows BEFORE the
            # h_t/i_t loops (from v6b).  Reduces down DMAs from
            # num_h_tiles*num_i_tiles to num_i_tiles per expert.
            # ---------------------------------------------------------------

            # ============================================================
            # v6b: Pre-load ALL down weight rows for this expert.
            # One [P, H] DMA per i_tile, issued BEFORE the h_t loop.
            # down_weights shape: [E, I, H]
            # For expert eid, i_tile i_t: rows [i_off:i_off+P], all H cols.
            # ============================================================
            dw_rows = []
            for i_t in nl.affine_range(num_i_tiles):
                i_off = i_t * P
                dw_row = nl.ndarray((P, H), dtype=down_weights.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=dw_row,
                    src=down_weights.ap(
                        pattern=[[H, P], [1, H]],     # partition: P rows spaced H apart; free: H contiguous
                        offset=i_off * H,              # start at row i_t*P in the [I, H] slice
                        scalar_offset=eid_offset,      # dynamic expert selection
                        indirect_dim=0,                 # expert dim is dim-0 of the 3D tensor
                    ),
                )
                dw_rows.append(dw_row)

            # h_t is independent → affine_range; i_t sequential (PSUM acc dep)
            # Same loop structure as v6b, matmul slices from pre-loaded rows.
            for h_t in nl.affine_range(num_h_tiles):
                h_off = h_t * P

                # Zero this h_t's PSUM before accumulating i_t contributions
                nisa.memset(dst=down_psum_tiles[h_t], value=0)

                for i_t in range(num_i_tiles):
                    # Slice [P, P] from pre-loaded [P, H] row (no DMA needed)
                    nisa.nc_matmul(
                        dst=down_psum_tiles[h_t],
                        stationary=act_bf16_tiles[i_t],
                        moving=dw_rows[i_t][0:P, h_off:h_off+P],  # [P, P] slice from coalesced row
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
        # Store token t's output: out_accum[h_t] → output[t, h_off:h_off+P]
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


def test_nki_moe_v7b_qwen3_t1():
    """Qwen3 shapes, T=1: H=2048, I=256, E=128, K=8."""
    import torch_xla.core.xla_model as xm

    print("\n" + "=" * 70)
    print("nki_moe_v7b — Qwen3 T=1 (H=2048, I=256, E=128, K=8)")
    print("=" * 70)

    device = xm.xla_device()
    T, H, I_size, E, K = 1, 2048, 256, 128, 8

    hidden, gu_w, d_w, eidx, rw_k = _make_sparse_inputs(T, H, I_size, E, K, seed=123)

    print(f"  Shapes: T={T}, H={H}, I={I_size}, E={E}, K={K}")
    print(f"  num_h_tiles={H//128}, num_i_tiles={I_size//128}, num_gu_tiles={2*I_size//128}")

    print("\n[1/3] Reference (PyTorch CPU)...")
    ref = pytorch_moe_sparse_reference(hidden, gu_w, d_w, eidx, rw_k)

    print("[2/3] NKI v7b kernel on Trainium...")
    nki_out = nki_moe_v7b(
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


def test_nki_moe_v7b():
    """Run all v7b correctness tests."""
    results = []
    results.append(("qwen3 T=1 (H=2048,I=256,E=128,K=8)", test_nki_moe_v7b_qwen3_t1()))

    print("\n" + "=" * 70)
    print("Summary:")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        all_ok = all_ok and ok
    print("=" * 70)
    if all_ok:
        print("All v7b tests PASSED.")
    else:
        print("Some v7b tests FAILED.")
    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.perf_counter()
    ok = test_nki_moe_v7b()
    t1 = time.perf_counter()
    print(f"\nTotal time: {t1 - t0:.1f}s")
    import sys
    sys.exit(0 if ok else 1)
