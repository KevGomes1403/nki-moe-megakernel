# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw-NKI selective routed-experts loop for the Qwen3.6-A3B MoE decode kernel (SBUF-resident).

Replaces the nkilib ``moe_tkg`` selective path with a self-contained expert loop written in
``nki`` / ``nki.isa`` / ``nki.language`` only. The reason to own the loop is a DMA-fragmentation
fix: ``moe_tkg`` loads each selected expert's gate and up weights as TWO separate strided ``[H, I]``
views (``[E,H,2,I].select(dim=0,e).select(dim=1, GATE/UP)``), so the innermost contiguous DMA run is
only ``I`` elements (256 B at bf16). Interleaved gate/up on the middle "2" axis fragments the load
into tens of thousands of tiny packets, leaving the sync engine bound and the tensor engine starved.

The fix loads each selected expert's gate+up as ONE contiguous ``[H0, H1, 2I]`` slab (inner run ``2I``,
512 B at bf16 -- half the descriptors), then slices ``gate = slab[:, :, 0:I]`` / ``up = slab[:, :, I:2I]``
in SBUF (free). The slab load is split into two H1-halves so the gate/up matmul over the first half can
start before the second half lands, and a 2-slot cross-expert prefetch ring issues expert k+1's weight
DMAs while expert k computes. This path is always on -- there is no legacy/env-gated fallback.

Reproduces the ``moe_tkg`` selective contract exactly: top-K experts per token, SiLU activation,
POST_SCALE by the (already L1-normalized) affinity with NO re-normalization, summed over the K experts.
The stored ``[E,H,2,I]`` gate/up and ``[E,I,H]`` down weight layouts are unchanged.

Layout: consumes the ``rmsnorm_tkg`` [H0,T,H1] tile (H = h0*H1 + j, "tp102") and emits ``routed_local``
in the same convention (routed_local[h0, t, h1] = down output at H-column h0*H1 + h1), so both the SBUF
megakernel readback and the natural [T,H] HBM store match what ``moe_tkg`` produced. Token-shard aware:
each LNC core computes over the full H for its ``[T_offset:T_offset+T_per_shard]`` token slice (no
H-sharding, no cross-core sendrecv) -- the caller supplies the shard geometry.

Per-rank (TP=4) A3B routed dims: H=2048 (H0=128, H1=16), E=256, K=8, I=128 (single I tile).
"""

import nki.isa as nisa
import nki.language as nl

P_MAX = 128
PREFETCH_SLOTS = 2  # double-buffer: expert k+1 loads while expert k computes


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def expert_scalar_offset(expert_index, global_token_idx, k, K):
    """Dynamic scalar offset selecting expert_index[global_token_idx, k] for indirect DMA."""
    return expert_index.ap(pattern=[[K, 1], [1, 1]], offset=global_token_idx * K + k)


def load_expert_weight_slab(
    gate_up_flat, down_w, slab_slot, down_slot, expert_offset, H0, H1, two_I, I0, H
):
    """Load one selected expert into a ring slot: fused gate+up slab (2 H1-halves) + down weight.

    gate/up: ONE contiguous ``[H0, H1, 2I]`` slab per expert (inner run 2I, vs moe_tkg's I). Split into
    two H1-halves (two DMAs) so the matmul over the first half can begin before the second half lands.
    down: ``[I0, H]`` contiguous (inner run H, already efficient). All three are indirect DMAs (dynamic
    expert via ``expert_offset``), so dge_mode is unknown (HW/SW DGE are unsafe for scalar_offset).

    Slab convention: slab[h0, h1, c] = gate_up_flat[e][h0*H1 + h1, c] (c<I gate, c>=I up), matching the
    normed tile's H = h0*H1 + j so each H1 matmul contracts H0 with aligned operands.
    """
    half_h1 = H1 // 2
    nisa.dma_copy(
        dst=slab_slot[0:H0, 0:half_h1, 0:two_I],
        src=gate_up_flat.ap(
            pattern=[[H1 * two_I, H0], [two_I, half_h1], [1, two_I]],
            offset=0,
            scalar_offset=expert_offset,
            indirect_dim=0,
        ),
        dge_mode=nisa.dge_mode.unknown,
    )
    nisa.dma_copy(
        dst=slab_slot[0:H0, half_h1:H1, 0:two_I],
        src=gate_up_flat.ap(
            pattern=[[H1 * two_I, H0], [two_I, H1 - half_h1], [1, two_I]],
            offset=half_h1 * two_I,
            scalar_offset=expert_offset,
            indirect_dim=0,
        ),
        dge_mode=nisa.dge_mode.unknown,
    )
    nisa.dma_copy(
        dst=down_slot[0:I0, 0:H],
        src=down_w.ap(
            pattern=[[H, I0], [1, H]],
            offset=0,
            scalar_offset=expert_offset,
            indirect_dim=0,
        ),
        dge_mode=nisa.dge_mode.unknown,
    )


def accumulate_expert_output(
    normed_sb,
    slab_slot,
    down_slot,
    affinity_col,
    output_temp,
    local_token_idx,
    global_token_idx,
    H0,
    H1,
    I,
    I0,
    two_I,
    H,
    io_dtype,
    is_first,
):
    """Compute one selected expert for one token and POST_SCALE-accumulate into output_temp.

    gate_up: contract H0 over H1 slabs -> gate/up [I,1] fp32 PSUM; SiLU(gate)*up -> intermediate [I,1].
    down: contract I0 -> [H0,1] per H1 slice via a strided stationary (columns h0*H1+h1) so
    output_temp[h0, h1] lands at H-column h0*H1+h1. Scale by the L1-normalized affinity (POST_SCALE,
    no re-normalization) and add to the token's running sum (fp32 accumulator).
    """
    gate_psum = nl.ndarray((I, 1), dtype=nl.float32, buffer=nl.psum)
    up_psum = nl.ndarray((I, 1), dtype=nl.float32, buffer=nl.psum)
    for j in range(H1):  # accumulate H0-contractions over H1 -> full-H gate/up logits
        moving = normed_sb[:, global_token_idx : global_token_idx + 1, j]
        nisa.nc_matmul(dst=gate_psum, stationary=slab_slot[:, j, 0:I], moving=moving)
        nisa.nc_matmul(dst=up_psum, stationary=slab_slot[:, j, I:two_I], moving=moving)

    gate_sb = nl.ndarray((I, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=gate_sb, data=gate_psum, op=nl.silu)
    up_sb = nl.ndarray((I, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=up_sb, src=up_psum)
    intermediate = nl.ndarray((I, 1), dtype=io_dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=intermediate, data1=gate_sb, data2=up_sb, op=nl.multiply)

    down_psum = nl.ndarray((H0, H1), dtype=nl.float32, buffer=nl.psum)
    for h1 in range(H1):  # H1 slices; strided stationary picks columns {h0*H1 + h1}
        stationary = down_slot.ap(pattern=[[H, I0], [H1, H0]], offset=h1)
        nisa.nc_matmul(
            dst=down_psum[0:H0, h1 : h1 + 1],
            stationary=stationary,
            moving=intermediate[0:I0, 0:1],
        )

    down_sb = nl.ndarray((H0, H1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=down_sb, src=down_psum)
    if is_first:
        nisa.tensor_scalar(
            dst=output_temp[:, :, local_token_idx],
            data=down_sb,
            op0=nl.multiply,
            operand0=affinity_col,
        )
    else:
        nisa.tensor_scalar(
            dst=down_sb, data=down_sb, op0=nl.multiply, operand0=affinity_col
        )
        nisa.tensor_tensor(
            dst=output_temp[:, :, local_token_idx],
            data1=output_temp[:, :, local_token_idx],
            data2=down_sb,
            op=nl.add,
        )


def routed_experts_selective(
    normed_sb,
    expert_gate_up_w,
    expert_down_w,
    expert_index,
    expert_affinities_eager,
    T_offset,
    T_per_shard,
    output_in_sbuf=True,
):
    """Selective top-K routed experts on an SBUF-resident normed tile (fused-slab load + prefetch ring).

    Args:
        normed_sb:                [H0=128, T, H1=H//128] SBUF post-attn-normed hidden (tp102 layout).
        expert_gate_up_w:         [E, H, 2, I] HBM fused gate/up expert weights (stored layout, unchanged).
        expert_down_w:            [E, I, H] HBM down expert weights (stored layout, unchanged).
        expert_index:             [T, K] uint32 SBUF top-K expert indices (from router_topk).
        expert_affinities_eager:  [T, K] fp32 SBUF L1-normalized top-K affinities in index order.
        T_offset / T_per_shard:   this core's token slice (token-shard geometry from the caller).
        output_in_sbuf:           True -> routed_local SBUF [H0,T,H1]; False -> HBM [T,H] natural.

    Returns:
        routed_local: SBUF [H0,T,H1] (output_in_sbuf=True) or HBM [T,H] (False) -- per-rank routed partial.
    """
    H0, T, H1 = normed_sb.shape
    H = H0 * H1
    E = expert_gate_up_w.shape[0]
    I = expert_gate_up_w.shape[3]
    K = expert_index.shape[1]
    io_dtype = normed_sb.dtype
    two_I = 2 * I
    I0 = I  # single I tile (I <= 128 for A3B routed)

    kernel_assert(
        I <= P_MAX, "routed_experts_nki supports I <= 128 (single intermediate tile)"
    )
    kernel_assert(expert_down_w.shape == (E, I, H), "expert_down_w must be [E, I, H]")

    # Flatten the stored [E,H,2,I] gate/up to [E,H,2I]: gate = cols 0:I, up = cols I:2I are contiguous.
    gate_up_flat = expert_gate_up_w.reshape((E, H, two_I))

    # fp32 accumulator over the K experts, per token: [H0, H1, T_per_shard].
    output_temp = nl.ndarray((H0, H1, T_per_shard), dtype=nl.float32, buffer=nl.sbuf)

    # Cross-expert prefetch ring: 2 slots for the fused gate/up slab + down weight.
    slab_slot_0 = nl.ndarray((H0, H1, two_I), dtype=io_dtype, buffer=nl.sbuf)
    slab_slot_1 = nl.ndarray((H0, H1, two_I), dtype=io_dtype, buffer=nl.sbuf)
    slab_slots = [slab_slot_0, slab_slot_1]
    down_slot_0 = nl.ndarray((I0, H), dtype=io_dtype, buffer=nl.sbuf)
    down_slot_1 = nl.ndarray((I0, H), dtype=io_dtype, buffer=nl.sbuf)
    down_slots = [down_slot_0, down_slot_1]

    # Per-partition token index (token_iota[t, h0] = t), reused to build each token's one-hot selector.
    token_iota = nl.ndarray((T, H0), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=token_iota, pattern=[[0, H0]], offset=0, channel_multiplier=1)

    for local_token_idx in range(T_per_shard):
        global_token_idx = local_token_idx + T_offset

        # Broadcast eager[global_token_idx, :] to all H0 partitions via a one-hot selector matmul:
        # selector[t, h0] = (t == global_token_idx); aff_bc[h0, k] = sum_t selector[t,h0]*eager[t,k].
        selector = nl.ndarray((T, H0), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=selector,
            data=token_iota,
            op0=nl.equal,
            operand0=float(global_token_idx),
        )
        aff_psum = nl.ndarray((H0, K), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=aff_psum, stationary=selector, moving=expert_affinities_eager
        )
        aff_bc = nl.ndarray((H0, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=aff_bc, src=aff_psum)

        # Prime slot 0 with expert k=0 before the K-loop.
        load_expert_weight_slab(
            gate_up_flat,
            expert_down_w,
            slab_slots[0],
            down_slots[0],
            expert_scalar_offset(expert_index, global_token_idx, 0, K),
            H0,
            H1,
            two_I,
            I0,
            H,
        )
        for k in range(K):
            cur = k % PREFETCH_SLOTS
            # Issue expert k+1's weight DMAs before computing k -- the DMA queue works on k+1
            # while the tensor engine consumes k.
            if k + 1 < K:
                nxt = (k + 1) % PREFETCH_SLOTS
                load_expert_weight_slab(
                    gate_up_flat,
                    expert_down_w,
                    slab_slots[nxt],
                    down_slots[nxt],
                    expert_scalar_offset(expert_index, global_token_idx, k + 1, K),
                    H0,
                    H1,
                    two_I,
                    I0,
                    H,
                )
            accumulate_expert_output(
                normed_sb,
                slab_slots[cur],
                down_slots[cur],
                aff_bc[:, k : k + 1],
                output_temp,
                local_token_idx,
                global_token_idx,
                H0,
                H1,
                I,
                I0,
                two_I,
                H,
                io_dtype,
                is_first=(k == 0),
            )

    # Store: output_temp [H0, H1, T] -> routed_local [H0, T, H1] (H = h0*H1 + j). Each core writes its slice.
    routed_local = nl.ndarray((H0, T, H1), dtype=io_dtype, buffer=nl.sbuf)
    for h1 in range(H1):
        nisa.tensor_copy(
            dst=routed_local[:, T_offset : T_offset + T_per_shard, h1],
            src=output_temp[:, h1, :],
        )
    if output_in_sbuf:
        return routed_local

    # HBM natural [T,H]: flat = global_token * H + (h0*H1 + h1); each core writes its token slice.
    out_hbm = nl.ndarray((T, H), dtype=io_dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=out_hbm.ap(
            pattern=[[H1, H0], [H, T_per_shard], [1, H1]], offset=T_offset * H
        ),
        src=routed_local[:, T_offset : T_offset + T_per_shard, :],
    )
    return out_hbm
