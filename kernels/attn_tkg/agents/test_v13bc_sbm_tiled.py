"""
Correctness test for v13bc_sbm_tiled: compares NKI kernel attention output
against a PyTorch float32 reference (Qwen3-MoE attention spec).

Checks:
  - attention output [1, 1, 2048] (via column-major unflatten of [128, 16])

Note: k_rope_out and v_out are intentionally not checked here.  In the
transformer kernel (transformer_qwen_v3_v2.py) those outputs are ignored
(_k_rope_out, _v_out).  When they are used, they go through NxDI's KV
manager which imposes its own layout conventions — testing them against a
plain PyTorch reference would be misleading.

Tolerances: rtol=1e-2, atol=2e-2  (bf16 kernel vs fp32 ref)
"""

import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

import math
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import nki
import nki.language as nl
import nki.isa as nisa
from nkilib.core.utils.allocator import SbufManager

from v13bc_sbm_tiled import qwen3_attn_tkg_fused_oproj_v13bc

# ---------------------------------------------------------------------------
# Shape constants
# ---------------------------------------------------------------------------
B       = 1
H       = 2048
d       = 128
Hq_tp   = 8
S_prior = 640
H_wo    = 2048
PMAX    = 128


# ---------------------------------------------------------------------------
# Thin @nki.jit wrapper
# ---------------------------------------------------------------------------
@nki.jit
def attn_kernel_wrapper(
    hidden_states, Wq, Wk, Wv, Wo,
    q_norm_weight, k_norm_weight,
    K_cache, V_cache, cos, sin, position_ids,
):
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("wrapper_outer")

    out_sb, _k_rope_out, _v_out = qwen3_attn_tkg_fused_oproj_v13bc(
        hidden_states, Wq, Wk, Wv, Wo,
        q_norm_weight, k_norm_weight,
        K_cache, V_cache, cos, sin, position_ids,
        out_sb=None,
        sbm=sbm,
    )

    # out_sb is [PMAX, H_wo//PMAX] = [128, 16] in SBUF — DMA to HBM
    output_hbm = nl.ndarray((128, 16), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=output_hbm, src=out_sb, dge_mode=nisa.dge_mode.hwdge)

    sbm.close_scope()  # wrapper_outer
    sbm.close_scope()  # attn_outer (opened inside sub-function)

    return output_hbm


# ---------------------------------------------------------------------------
# PyTorch float32 reference
# ---------------------------------------------------------------------------
def reference_attn(hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
                   K_cache, V_cache, cos, sin, position_ids):
    _, _, H_r = hidden_states.shape
    d_r = 128; Hq_tp_r = 8; S_prior_r = K_cache.shape[2]
    half_d = d_r // 2

    h = hidden_states.reshape(H_r, 1).float()

    q = Wq.float() @ h               # [1024, 1]
    k = Wk.float() @ h               # [128,  1]
    v = Wv.float() @ h               # [128,  1]

    # K RMSNorm
    k_ms = (k ** 2).mean(dim=0, keepdim=True) + 1e-6
    k = k * k_ms.rsqrt() * k_norm_weight.reshape(d_r, 1).float()

    # K RoPE
    cos_v = cos.reshape(d_r, 1).float()
    sin_v = sin.reshape(d_r, 1).float()
    rot_k = torch.cat([-k[half_d:], k[:half_d]], dim=0)
    k_rope = k * cos_v + rot_k * sin_v

    # Q RMSNorm (per head)
    q_heads = q.reshape(Hq_tp_r, d_r, 1)
    q_ms = (q_heads ** 2).mean(dim=1, keepdim=True) + 1e-6
    q_heads = q_heads * q_ms.rsqrt() * q_norm_weight.reshape(1, d_r, 1).float()

    # Q RoPE (broadcast across heads)
    cos_h = cos_v.unsqueeze(0).expand(Hq_tp_r, -1, -1)
    sin_h = sin_v.unsqueeze(0).expand(Hq_tp_r, -1, -1)
    rot_q = torch.cat([-q_heads[:, half_d:], q_heads[:, :half_d]], dim=1)
    q_rope = q_heads * cos_h + rot_q * sin_h
    q_scaled = q_rope / math.sqrt(d_r)

    # Flash-decode attention
    pos = int(position_ids[0, 0])
    K_all = torch.cat([K_cache.reshape(S_prior_r, d_r).float().T, k_rope], dim=1)
    V_all = torch.cat([V_cache.reshape(S_prior_r, d_r).float().T, v], dim=1)

    attn_out = []
    for h_idx in range(Hq_tp_r):
        q_h = q_scaled[h_idx]                          # [128, 1]
        scores = (K_all.T @ q_h)                       # [S+1, 1]
        mask = torch.zeros(S_prior_r + 1, 1)
        mask[pos:S_prior_r] = -1e9
        scores = scores + mask
        weights = torch.softmax(scores, dim=0)
        out_h = V_all @ weights                        # [128, 1]
        attn_out.append(out_h)

    attn_packed = torch.stack(attn_out, dim=1).reshape(d_r, Hq_tp_r)   # [128, 8]
    attn_flat = attn_packed.T.reshape(1, Hq_tp_r * d_r)                 # [1, 1024]
    out = attn_flat @ Wo.float()                                         # [1, 2048]

    return out.reshape(1, 1, H_r)   # [1, 1, 2048]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(42)
    device = "xla"

    # Build inputs on CPU
    hidden_cpu   = torch.randn(B, 1, H,          dtype=torch.bfloat16)
    Wq_cpu       = torch.randn(Hq_tp * d, H,     dtype=torch.bfloat16) * 0.02
    Wk_cpu       = torch.randn(d, H,             dtype=torch.bfloat16) * 0.02
    Wv_cpu       = torch.randn(d, H,             dtype=torch.bfloat16) * 0.02
    Wo_cpu       = torch.randn(Hq_tp * d, H_wo,  dtype=torch.bfloat16) * 0.02
    qnw_cpu      = torch.ones(d,                 dtype=torch.bfloat16)
    knw_cpu      = torch.ones(d,                 dtype=torch.bfloat16)
    K_cache_cpu  = torch.randn(B, 1, S_prior, d, dtype=torch.bfloat16) * 0.1
    V_cache_cpu  = torch.randn(B, 1, S_prior, d, dtype=torch.bfloat16) * 0.1
    cos_cpu      = torch.cos(torch.linspace(0, 1, d)).reshape(B, d).to(torch.bfloat16)
    sin_cpu      = torch.sin(torch.linspace(0, 1, d)).reshape(B, d).to(torch.bfloat16)
    pos_cpu      = torch.tensor([[S_prior // 2]], dtype=torch.int32)   # pos = 320

    # Move to XLA
    inputs_xla = [t.to(device) for t in [
        hidden_cpu, Wq_cpu, Wk_cpu, Wv_cpu, Wo_cpu,
        qnw_cpu, knw_cpu, K_cache_cpu, V_cache_cpu,
        cos_cpu, sin_cpu, pos_cpu,
    ]]

    # Run kernel
    print("Running NKI kernel...")
    out_hbm = attn_kernel_wrapper(*inputs_xla)

    # [128, 16] column-major -> flat [2048]
    out_np = out_hbm.cpu().float().numpy()

    # PyTorch reference
    print("Computing PyTorch reference...")
    ref_out = reference_attn(
        hidden_cpu, Wq_cpu, Wk_cpu, Wv_cpu, Wo_cpu,
        qnw_cpu, knw_cpu, K_cache_cpu, V_cache_cpu,
        cos_cpu, sin_cpu, pos_cpu,
    )
    ref_out_np = ref_out.float().numpy()   # [1, 1, 2048]

    # Unflatten column-major [128, 16] -> flat [2048]
    # out_hbm[p, j] = linear_out[j*128 + p]  (column-major)
    out_flat     = out_np.flatten(order='F')
    ref_out_flat = ref_out_np.reshape(-1)

    max_diff  = np.abs(out_flat - ref_out_flat).max()
    mean_diff = np.abs(out_flat - ref_out_flat).mean()
    print(f"\n  attention output  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")

    np.testing.assert_allclose(out_flat, ref_out_flat, rtol=1e-2, atol=2e-2)
    print("CORRECTNESS: PASS")


if __name__ == "__main__":
    main()
