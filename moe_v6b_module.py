"""
Custom MoE module using RouterTopK + v6b NKI kernel for expert computation.

Designed as a drop-in replacement for NxDI's MoE module in the TKG (token generation) path.
Uses RouterTopK from neuronx_distributed for routing, then the v6b coalesced-DMA NKI kernel
for expert FFN computation.

State dict keys match NxDI convention:
    router.linear_router.weight                  [E, H]
    expert_mlps.mlp_op.gate_up_proj.weight        [E, H, 2*I]
    expert_mlps.mlp_op.down_proj.weight            [E, I, H]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.modules.moe.routing import RouterTopK
from kernels.v6b_qwen3 import nki_moe_v6b_qwen3

import logging
logger = logging.getLogger("Neuron")

class _WeightContainer(nn.Module):
    """Holds a single weight parameter."""
    def __init__(self, shape, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*shape, dtype=dtype))


class _MLPOp(nn.Module):
    """Holds gate_up_proj and down_proj weights."""
    def __init__(self, E, H, I, dtype):
        super().__init__()
        self.gate_up_proj = _WeightContainer((E, H, 2 * I), dtype)
        self.down_proj = _WeightContainer((E, I, H), dtype)


class _ExpertMLPs(nn.Module):
    """Wrapper matching NxDI key path: expert_mlps.mlp_op.{gate_up,down}_proj.weight"""
    def __init__(self, E, H, I, dtype):
        super().__init__()
        self.mlp_op = _MLPOp(E, H, I, dtype)


class MoEV6bModule(nn.Module):
    """
    MoE module: RouterTopK (routing) + v6b NKI kernel (expert FFN).

    For token generation (seq_len=1): routes tokens via RouterTopK, then
    computes expert FFN using the v6b coalesced-DMA kernel.

    Args:
        num_experts: Number of experts (E)
        top_k: Top-K experts per token
        hidden_size: Hidden dimension (H), must be divisible by 128
        intermediate_size: Expert intermediate dimension (I), must be divisible by 128
        normalize_top_k: Whether to L1-normalize top-k routing weights
        router_dtype: dtype for router linear layer (default float32)
        router_act_fn: Activation for routing scores (default "softmax")
        weight_dtype: dtype for expert weights (default bfloat16)
        rmsnorm: Optional RMSNorm module to apply before routing
    """

    def __init__(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        normalize_top_k=False,
        router_dtype=torch.float32,
        router_act_fn="softmax",
        weight_dtype=torch.bfloat16,
        rmsnorm=None,
    ):
        super().__init__()
        self.rmsnorm = rmsnorm
        self.normalize_top_k = normalize_top_k
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.router = RouterTopK(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            dtype=router_dtype,
            act_fn=router_act_fn,
        )

        self.expert_mlps = _ExpertMLPs(num_experts, hidden_size, intermediate_size, weight_dtype)

    def forward(self, hidden_states, padding_mask=None):
        if self.rmsnorm is not None:
            hidden_states = self.rmsnorm(hidden_states)

        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_flat = hidden_states.reshape(-1, self.hidden_size)
        else:
            hidden_flat = hidden_states

        # Route
        router_logits, expert_affinities, expert_index = self.router(hidden_flat)

        # Extract top-k affinities [T, K]
        routing_weights_k = torch.gather(expert_affinities, 1, expert_index)

        if self.normalize_top_k:
            routing_weights_k = F.normalize(routing_weights_k, p=1.0, dim=1)

        # v6b kernel: expert FFN computation
        logger.debug("Using custom moe_tkg NKI kernel for token generation")
        output = nki_moe_v6b_qwen3(
            hidden_flat.contiguous(),
            self.expert_mlps.mlp_op.gate_up_proj.weight.data,
            self.expert_mlps.mlp_op.down_proj.weight.data,
            expert_index.to(torch.int32).contiguous(),
            routing_weights_k.to(torch.float32).contiguous(),
        )

        if len(original_shape) == 3:
            output = output.reshape(original_shape)

        return output, router_logits
