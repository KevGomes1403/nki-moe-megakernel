# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.6-A3B fused MoE layer NKI kernels (token generation).

The post-attention FFN of every A3B decoder layer, built SBUF-resident so the input RMSNorm output
persists (no HBM round-trip) and is shared by all consumers. Full pipeline:

    normed_sb = post_attn_norm(hidden, gamma)          <- NORM ONCE, shared by all consumers
       |-- routed_experts (router_topk + moe_tkg)      -> routed_local
       |-- shared_expert  (mlp_tkg SwiGLU)             -> shared_local
       '-- sigma_gate     (sigmoid H->1 matmul)        -> g [T,1]
    combined_local = routed_local + g * shared_local   <- ONE per-rank partial, single downstream AR

- post_attn_norm  : nkilib rmsnorm_tkg glue (reuses the GQA input-norm composable) -> normed_sb SBUF
- routed_experts  : nkilib router_topk (softmax/top-8/L1) + moe_tkg selective (SiLU, POST_SCALE)
- shared_expert   : nkilib mlp_tkg SwiGLU primitives (per-rank partial)
- moe_layer       : the combine slice -- sigma-gate + broadcast gated sum over all three consumers

``moe_layer_compose`` returns the SINGLE per-rank partial for one ``reduce_from_tensor_model_parallel_region``
(rank-replicated sigma-gate: AR(routed) + g*AR(shared) == AR(routed + g*shared)). Spec:
``nki_kernels/specs/moe_layer.md``. Isolation tests: ``tests/test_moe_{routed,shared_expert,layer}_kernel.py``.
"""

from .components.moe_layer import (
    gated_sum,
    moe_layer_compose,
    moe_layer_fwd,
    sigma_gate_compose,
)
from .components.post_attn_norm import post_attn_rmsnorm_compose
from .components.routed_experts import moe_routed_compose, routed_experts_compose
from .components.shared_expert import moe_shared_compose, shared_expert_compose

__all__ = [
    "post_attn_rmsnorm_compose",
    "routed_experts_compose",
    "moe_routed_compose",
    "shared_expert_compose",
    "moe_shared_compose",
    "moe_layer_compose",
    "moe_layer_fwd",
    "sigma_gate_compose",
    "gated_sum",
]
