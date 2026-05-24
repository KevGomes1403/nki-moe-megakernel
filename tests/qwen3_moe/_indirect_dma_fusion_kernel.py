"""Standalone selective-expert MoE kernel for the indirect-DMA fusion parity test.

This is a minimal, self-contained NKI kernel that drives the selective-expert
MoE path (``nki_kernels.moe.moe_tkg`` with ``is_all_expert=False``) directly
from HBM inputs. It does NOT include rmsnorm/router scaffolding — the test
provides pre-computed ``expert_index`` and ``expert_affinities`` and a
pre-RMSNorm'd hidden tensor, so the kernel exercises only the MoE expert
loop where ``NKI_MOE_INDIRECT_DMA_FUSION`` actually changes behavior.

This file does NOT import the broken ``_attention_only_kernel.py``; it's
self-contained for the fusion parity test.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

from nki_kernels.moe import moe_tkg
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
)


def _moe_fusion_kernel_body(
    hidden_input,          # [T, H]                      bf16 HBM (already RMSNorm'd)
    expert_gate_up_weights,  # [E, H, 2, I]              bf16 HBM
    expert_down_weights,     # [E, I, H]                 bf16 HBM
    expert_affinities,       # [T, E]                    fp32 HBM (sparse top-K normalized)
    expert_index,            # [T, K]                    uint32 HBM
    replica_groups=None,
):
    """Call moe_tkg with selective-expert mode and POST_SCALE affinities.

    This is exactly the moe_tkg invocation that ``_moe_only_kernel.py`` uses,
    minus the rmsnorm/router/AR-gather scaffolding that depends on the broken
    test_attention_vs_hf import.
    """
    output = moe_tkg(
        hidden_input=hidden_input,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        is_all_expert=False,
        expert_gate_up_bias=None,
        expert_down_bias=None,
        activation_fn=ActFnType.SiLU,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        gate_clamp_upper_limit=None,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
        up_clamp_lower_limit=None,
        output_in_sbuf=False,
        name_prefix="parity_",
    )
    return (output,)


@nki.jit
def moe_fusion_kernel(
    hidden_input,
    expert_gate_up_weights,
    expert_down_weights,
    expert_affinities,
    expert_index,
    replica_groups=None,
):
    return _moe_fusion_kernel_body(
        hidden_input=hidden_input,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        replica_groups=replica_groups,
    )
