"""
Optimized MoE kernel — v6b: Coalesce Down Weight DMAs.

Built on v5a (Coalesce Gate+Up Weight DMAs).

Key optimization over v5a:
  - Down weight DMA coalescing: instead of loading one [P, P] tile per
    (h_t, i_t) combination (16*2 = 32 DMAs per expert for H=2048, I=256),
    we load the entire [P, H] weight row in ONE DMA per i_tile.  Then we
    slice dw_row[:, h_t*P:(h_t+1)*P] for each h_t's nc_matmul.
  - This reduces down DMAs from num_h_tiles * num_i_tiles per expert
    to just num_i_tiles per expert (e.g. 32 → 2 for H=2048, I=256).

Down projection DMA strategy:
  - Old (v5a): inside h_t outer / i_t inner loop, loads [P,P] per (h_t,i_t)
  - New (v6b): pre-loads ALL num_i_tiles [P,H] rows via affine_range BEFORE
    the h_t/i_t loop, then slices [P,P] from the pre-loaded rows for each matmul.
  - Loop structure (h_t outer affine, i_t inner sequential) is preserved from v5a.

Everything else is identical to v5a.

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
_I_SIZE = 256
_TWO_I  = 2 * _I_SIZE   # 512
_P      = 128

@nki.jit(platform_target="trn2")
def nki_moe_v6b(
    hidden_states,        # [T, H]    bf16
    gate_up_weights,      # [E, H, 2*I] bf16
    down_weights,         # [E, I, H]   bf16
    expert_indices,       # [T, K]    int32   — top-K expert indices per token
    routing_weights_k,    # [T, K]    float32 — routing weights for the K experts
):
    """
    Sparse Token-Parallel MoE kernel (v6b) with coalesced gate+up AND down DMAs.
    Hardcoded for Qwen3-30B-A3B (I=256, bf16 weights).
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

    # HBM output buffer
    output = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # ── Pre-allocate PSUM tiles ───────────────────────────────────────────
    gu_psum_tiles = []
    for _gu_t in nl.affine_range(num_gu_tiles):
        gu_psum_tiles.append(
            nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
        )

    down_psum_tiles = []
    for _h_t in nl.affine_range(num_h_tiles):
        down_psum_tiles.append(
            nl.ndarray((1, P), dtype=nl.float32, buffer=nl.psum)
        )

    # ── TOKEN LOOP ────────────────────────────────────────────────────────
    for t in range(T):

        # Load token hidden state tiles
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

        # Output accumulator
        out_accum = []
        for _h_t in nl.affine_range(num_h_tiles):
            tmp = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=tmp, value=0)
            out_accum.append(tmp)

        # ── K-EXPERT LOOP ─────────────────────────────────────────────────
        for k in range(K):

            # Load this (t, k) expert index from HBM into a dedicated
            # P-partition SBUF scalar at partition-0 / element-0 (the base).
            #
            # The DGE scalar_offset MUST be an .ap() descriptor on a
            # full-P-partition SBUF tensor so the compiler can derive
            # IndirectDimMaxIndex = P-1 = 127 = E-1 (NCC_IBIR030 otherwise).
            #
            # Critically we always write to offset-0 and always read back at
            # offset-0.  .ap(offset=0) resolves to the tensor's own base
            # address regardless of where the E2E compiler allocates the
            # tensor in SBUF, avoiding the out-of-bound DGE fault seen when
            # a non-zero varying offset (t*K+k) was used and the standalone-
            # compiled absolute SBUF address became stale after E2E relocation.
            eid_pmax = nl.ndarray((P, 1), dtype=expert_indices.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=eid_pmax[0:1, 0:1],   # always partition-0, element-0 (base)
                src=expert_indices.ap(
                    pattern=[[K, 1], [1, 1]],
                    offset=t * K + k,
                ),
            )
            # offset=0 → always the base of eid_pmax (partition-0, element-0)
            # pattern=[[1,1],[1,1]] → 1×1 scalar read
            eid_offset = eid_pmax.ap(pattern=[[1, 1], [1, 1]], offset=0)

            rw_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=rw_scalar,
                src=routing_weights_k.ap(
                    pattern=[[K, 1], [1, 1]],
                    offset=t * K + k,
                ),
            )

            # ── Stage 1: Gate+Up projection ───────────────────────────────

            for gu_t in nl.affine_range(num_gu_tiles):
                nisa.memset(dst=gu_psum_tiles[gu_t], value=0)

            for h_t in range(num_h_tiles):
                h_off  = h_t * P
                h_tile = h_tiles_sb[h_t]

                # Coalesced gate+up weight DMA: one [P, two_I] row per h_t
                w_row = nl.ndarray((P, two_I), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=w_row,
                    src=gate_up_weights.ap(
                        pattern=[[two_I, P], [1, two_I]],
                        offset=h_off * two_I,
                        scalar_offset=eid_offset,
                        indirect_dim=0,
                    ),
                )

                for gu_t in nl.affine_range(num_gu_tiles):
                    gu_off = gu_t * P
                    nisa.nc_matmul(
                        dst=gu_psum_tiles[gu_t],
                        stationary=w_row[0:P, gu_off:gu_off+P],
                        moving=h_tile,
                    )

            # SiLU + multiply + cast to bf16
            act_bf16_tiles = []
            for i_t in nl.affine_range(num_i_tiles):
                gate_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=gate_sb, src=gu_psum_tiles[i_t])

                up_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=up_sb, src=gu_psum_tiles[num_i_tiles + i_t])

                gate_act = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=gate_act, op=nl.silu, data=gate_sb, scale=1.0)

                act_f32 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=act_f32, data1=gate_act, data2=up_sb, op=nl.multiply,
                )

                act_bf16 = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=act_bf16, src=act_f32)
                act_bf16_tiles.append(act_bf16)

            # ── Stage 2: Down projection (coalesced DMA) ─────────────────

            dw_rows = []
            for i_t in nl.affine_range(num_i_tiles):
                i_off = i_t * P
                dw_row = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=dw_row,
                    src=down_weights.ap(
                        pattern=[[H, P], [1, H]],
                        offset=i_off * H,
                        scalar_offset=eid_offset,
                        indirect_dim=0,
                    ),
                )
                dw_rows.append(dw_row)

            for h_t in nl.affine_range(num_h_tiles):
                h_off = h_t * P
                nisa.memset(dst=down_psum_tiles[h_t], value=0)

                for i_t in range(num_i_tiles):
                    nisa.nc_matmul(
                        dst=down_psum_tiles[h_t],
                        stationary=act_bf16_tiles[i_t],
                        moving=dw_rows[i_t][0:P, h_off:h_off+P],
                    )

            # Scale + accumulate
            for h_t in nl.affine_range(num_h_tiles):
                down_sb = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=down_sb, src=down_psum_tiles[h_t])

                scaled = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=scaled, data=down_sb,
                    op0=nl.multiply, operand0=rw_scalar,
                )

                nisa.tensor_tensor(
                    dst=out_accum[h_t], data1=out_accum[h_t],
                    data2=scaled, op=nl.add,
                )

        # ── Store output ──────────────────────────────────────────────────
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


def test_nki_moe_v6b_qwen3_t1():
    """Qwen3 shapes, T=1: H=2048, I=256, E=128, K=8."""
    import torch_xla.core.xla_model as xm

    print("\n" + "=" * 70)
    print("nki_moe_v6b — Qwen3 T=1 (H=2048, I=256, E=128, K=8)")
    print("=" * 70)

    device = xm.xla_device()
    T, H, I_size, E, K = 1, 2048, 256, 128, 8

    hidden, gu_w, d_w, eidx, rw_k = _make_sparse_inputs(T, H, I_size, E, K, seed=123)

    print(f"  Shapes: T={T}, H={H}, I={I_size}, E={E}, K={K}")
    print(f"  num_h_tiles={H//128}, num_i_tiles={I_size//128}, num_gu_tiles={2*I_size//128}")

    print("\n[1/3] Reference (PyTorch CPU)...")
    ref = pytorch_moe_sparse_reference(hidden, gu_w, d_w, eidx, rw_k)

    print("[2/3] NKI v6b kernel on Trainium...")
    nki_out = nki_moe_v6b(
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


def test_nki_moe_v6b():
    """Run all v6b correctness tests."""
    results = []
    results.append(("qwen3 T=1 (H=2048,I=256,E=128,K=8)", test_nki_moe_v6b_qwen3_t1()))

    print("\n" + "=" * 70)
    print("Summary:")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        all_ok = all_ok and ok
    print("=" * 70)
    if all_ok:
        print("All v6b tests PASSED.")
    else:
        print("Some v6b tests FAILED.")
    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.perf_counter()
    ok = test_nki_moe_v6b()
    t1 = time.perf_counter()
    print(f"\nTotal time: {t1 - t0:.1f}s")
    import sys
    sys.exit(0 if ok else 1)
