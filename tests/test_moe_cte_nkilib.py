"""
Test moe_cte NKI kernel with Qwen3-30B-A3B shapes.

Validates the kernel for context-encoding scenarios using the exact model dimensions
from ~/models/Qwen/config.json:
  H = 2048  (hidden_size)
  I = 768   (moe_intermediate_size)  — full, single-device (no TP sharding)
  E = 128   (num_experts)
  K = 8     (num_experts_per_tok)
  norm_topk_prob = True  (Qwen3 normalizes selected expert probs to sum to 1)

moe_cte API shapes:
  hidden_states:            [T+1, H]       (padded with zero row at index T)
  expert_affinities_masked: [(T+1)*E, 1]   (flattened, float32)
  gate_up_proj_weight:      [E, H, 2, I]   (bfloat16)
  down_proj_weight:         [E, I, H]      (bfloat16)
  token_position_to_id:     [N*B]          (int32, padding positions = T)
  block_to_expert:          [N, 1]         (int32)

Run:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  python tests/test_moe_cte_nkilib.py
"""

import os
import sys
import math

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

from nkilib.core.moe.moe_cte.moe_cte import moe_cte, MoECTESpec, MoECTEImplementation
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

# ── Qwen3-30B-A3B shapes ────────────────────────────────────────────────────
H = 2048   # hidden_size
I = 768    # moe_intermediate_size (full; no TP in this standalone test)
E = 128    # num_experts
K = 8      # num_experts_per_tok

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")


# ── Blockwise routing (mirrors NxD ExpertMLPs logic) ────────────────────────

def _build_blockwise_routing(
    expert_mask: torch.Tensor,   # [T, E] int64, top-K-hot
    expert_index: torch.Tensor,  # [T, K] int64
    num_experts: int,
    block_size: int,
):
    """Build token_position_to_id and block_to_expert from router output.

    Returns:
        block_to_expert:      [N] int64  — expert id for each block
        token_position_to_id: [N*B] int64 — token index for each position (-1 for padding)
        num_blocks:           int
    """
    T = expert_mask.shape[0]
    top_k = expert_index.shape[1]

    tokens_per_expert = expert_mask.sum(dim=0)  # [E]
    blocks_per_expert = ((tokens_per_expert + block_size - 1) // block_size).long()  # [E]

    # Total blocks (worst-case formula from NxD)
    num_blocks = math.ceil((T * top_k - (num_experts - 1)) / block_size) + num_experts - 1
    num_blocks = min(num_blocks, T * top_k)

    # block_to_expert: [N]
    block_ids = torch.arange(num_blocks, dtype=torch.long)
    cumulative_blocks = torch.cumsum(blocks_per_expert, dim=0)  # [E] inclusive
    block_to_expert = (block_ids.unsqueeze(1) >= cumulative_blocks[:-1]).sum(dim=1).long()

    # Token positions within expert blocks (1-indexed; index 0 is sentinel)
    token_pos = torch.cumsum(expert_mask.long(), dim=0)  # [T, E]
    expert_offsets = cumulative_blocks * block_size  # [E]
    token_pos[:, 1:] += expert_offsets[:-1]
    token_pos = token_pos.masked_fill(expert_mask == 0, 0).long()

    # Invert: token_position_to_id
    tpti = -torch.ones(num_blocks * block_size + 1, dtype=torch.long)
    tokens_idx = torch.arange(T, dtype=torch.long)
    topk_pos = torch.gather(token_pos, dim=1, index=expert_index.long())  # [T, K]
    tpti[topk_pos] = tokens_idx.unsqueeze(1)
    tpti = tpti[1:]  # drop sentinel at index 0

    return block_to_expert, tpti, num_blocks


# ── PyTorch reference (vectorized per-expert) ──────────────────────────────

def moe_ref_vectorized(
    hidden: torch.Tensor,        # [T, H]  float32
    gate_up_w: torch.Tensor,     # [E, H, 2, I]  float32
    down_w: torch.Tensor,        # [E, I, H]  float32
    affinities: torch.Tensor,    # [T, E]  float32
    expert_index: torch.Tensor,  # [T, K]  int64
    num_experts: int,
) -> torch.Tensor:
    """Reference MoE: per-expert batched matmul with POST_SCALE affinity weighting."""
    T, H_dim = hidden.shape
    output = torch.zeros(T, H_dim, dtype=torch.float32)

    for e in range(num_experts):
        # Find tokens assigned to this expert
        mask = (expert_index == e).any(dim=1)  # [T]
        if not mask.any():
            continue
        token_ids = mask.nonzero(as_tuple=True)[0]
        h = hidden[token_ids].float()                          # [n, H]
        aff = affinities[token_ids, e].float().unsqueeze(1)    # [n, 1]

        gate_out = F.silu(h @ gate_up_w[e, :, 0, :].float())  # [n, I]
        up_out = h @ gate_up_w[e, :, 1, :].float()             # [n, I]
        expert_out = (gate_out * up_out) @ down_w[e].float()   # [n, H]

        output.index_add_(0, token_ids, aff * expert_out)

    return output


# ── Test helper ──────────────────────────────────────────────────────────────

def run_test(T: int, block_size: int = 512, tol: float = 5e-2) -> bool:
    """Run one test with T tokens and validate against the PyTorch reference."""
    print(f"\n{'─' * 60}")
    print(f"  T={T}, H={H}, I={I}, E={E}, K={K}, block_size={block_size}")
    print(f"{'─' * 60}")

    device = xm.xla_device()
    torch.manual_seed(42)
    scale = 0.05  # small scale for BF16 numerical stability

    # Random BF16 inputs
    hidden_cpu = torch.randn(T, H, dtype=torch.bfloat16) * scale
    gate_up_cpu = torch.randn(E, H, 2, I, dtype=torch.bfloat16) * scale
    down_cpu = torch.randn(E, I, H, dtype=torch.bfloat16) * scale

    # Simulate Qwen3 router: softmax → top-K → re-normalize (norm_topk_prob=True)
    router_logits = torch.randn(T, E)
    softmax_probs = torch.softmax(router_logits.float(), dim=-1)    # [T, E]
    topk_vals, expert_idx = torch.topk(softmax_probs, K, dim=-1)   # [T, K]
    topk_norm = topk_vals / topk_vals.sum(dim=-1, keepdim=True)    # [T, K]: sum=1
    affinities_sparse = torch.zeros(T, E)
    affinities_sparse.scatter_(1, expert_idx, topk_norm)           # [T, E] sparse

    # ── Reference (CPU, float32) ──
    print("  [1/3] Computing PyTorch reference...")
    ref = moe_ref_vectorized(
        hidden=hidden_cpu.float(),
        gate_up_w=gate_up_cpu.float(),
        down_w=down_cpu.float(),
        affinities=affinities_sparse,
        expert_index=expert_idx,
        num_experts=E,
    )

    # ── Build blockwise routing tensors ──
    expert_mask = torch.zeros(T, E, dtype=torch.long)
    expert_mask.scatter_(1, expert_idx, 1)

    block_to_expert, tpti, num_blocks = _build_blockwise_routing(
        expert_mask=expert_mask,
        expert_index=expert_idx,
        num_experts=E,
        block_size=block_size,
    )

    # ── Pad to T+1 (API requires padding row at index T) ──
    hidden_padded = torch.cat([
        hidden_cpu,
        torch.zeros(1, H, dtype=torch.bfloat16),
    ])  # [T+1, H]

    affinities_padded = torch.cat([
        affinities_sparse,
        torch.zeros(1, E),
    ])  # [T+1, E]

    # Replace -1 padding positions with T (points to the zero-padded row)
    tpti = tpti.masked_fill(tpti == -1, T).to(torch.int32)

    # Flatten affinities: [(T+1)*E, 1]
    affinities_flat = affinities_padded.to(torch.float32).reshape(-1, 1)

    # block_to_expert: [N, 1] int32
    bte = block_to_expert.to(torch.int32).unsqueeze(1)

    print(f"  num_blocks={num_blocks}, tpti.shape={tpti.shape}, bte.shape={bte.shape}")
    print(f"  hidden_padded.shape={hidden_padded.shape}, affinities_flat.shape={affinities_flat.shape}")

    # ── Run moe_cte NKI kernel ──
    print("  [2/3] Running moe_cte NKI kernel...")

    spec = MoECTESpec(implementation=MoECTEImplementation.shard_on_i)

    output = moe_cte(
        hidden_states=hidden_padded.to(device),
        expert_affinities_masked=affinities_flat.to(device),
        gate_up_proj_weight=gate_up_cpu.to(device),
        down_proj_weight=down_cpu.to(device),
        token_position_to_id=tpti.to(device),
        block_to_expert=bte.to(device),
        block_size=block_size,
        spec=spec,
        activation_function=ActFnType.SiLU,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        is_tensor_update_accumulating=True,
    )

    # output is [T+1, H]; take first T rows
    output_cpu = output.cpu()[:T, :].float()

    # ── Compare ──
    print("  [3/3] Comparing...")
    max_diff = (ref - output_cpu).abs().max().item()
    mean_diff = (ref - output_cpu).abs().mean().item()

    passed = max_diff < tol
    status = "PASS" if passed else "FAIL"
    print(f"  max_diff={max_diff:.3e}  mean_diff={mean_diff:.3e}  tol={tol:.1e}  -> {status}")
    return passed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("moe_cte Qwen3 shape validation (trn2)")
    print(f"  Model dims: H={H}, I={I}, E={E}, K={K}")
    print("=" * 60)

    results = {}
    for T in [1024]:
        results[T] = run_test(T)

    print("\n" + "=" * 60)
    all_passed = all(results.values())
    for T, ok in results.items():
        mark = "PASS" if ok else "FAIL"
        print(f"  T={T:4d}  ->  {mark}")
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED - moe_cte works with Qwen3 shapes on trn2")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
