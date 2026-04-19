"""
Benchmark harness for v13bc_kv_norm (fused pre-attn RMSNorm + QKV + RoPE +
in-kernel KV scatter + flash-decode + o_proj).

Inputs match the correctness test (test_v13bc_kv_norm.py): SC=0.02 weight
scaling, random seed 42, pos = S/2.
"""

import os
import sys

# Set BEFORE any neuron/torch_xla import.
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_kv_norm"
)
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

import nki, nki.language as nl, nki.isa as nisa
import torch, numpy as np
from nkilib.core.utils.allocator import SbufManager
from v14a_kv_norm import qwen3_attn_tkg_fused_oproj_v13bc_kv_norm
from benchmark import wrap_benchmark

PMAX = 128


@nki.jit
def v13bc_kv_norm_wrapper(
    hidden_states, Wq, Wk, Wv, Wo,
    q_norm_weight, k_norm_weight, gamma_pre_attn,
    K_cache, V_cache, cos, sin, position_ids,
    output,
):
    B    = cos.shape[0]
    H    = Wq.shape[1]
    H_wo = Wo.shape[1]
    num_h_tiles = H // PMAX
    num_out_cols = H_wo // PMAX

    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("wrapper")

    hidden_col = hidden_states.reshape((H, B))
    hidden_sb  = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="hidden_sb")
    nisa.dma_copy(
        dst=hidden_sb,
        src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )

    out_sb = qwen3_attn_tkg_fused_oproj_v13bc_kv_norm(
        hidden_sb=hidden_sb,
        Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        gamma_pre_attn=gamma_pre_attn,
        K_cache=K_cache, V_cache=V_cache,
        cos=cos, sin=sin, position_ids=position_ids,
        sbm=sbm,
    )

    output_flat = output.reshape((B, H_wo))
    for j in range(num_out_cols):
        col_psum = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(col_psum, out_sb[0:PMAX, j:j+1])
        col_sb = sbm.alloc_stack((1, PMAX), nl.bfloat16, name=f"col_sb_{j}")
        nisa.tensor_copy(col_sb, col_psum)
        nisa.dma_copy(
            dst=output_flat[0:B, j*PMAX:(j+1)*PMAX],
            src=col_sb,
            dge_mode=nisa.dge_mode.hwdge,
        )

    sbm.close_scope()  # attn_outer (left open by sub-function)
    sbm.close_scope()  # wrapper
    return output, K_cache, V_cache


def main():
    rng = np.random.default_rng(42)
    B, H, d, Hq_tp, S = 1, 2048, 128, 8, 640
    H_wo = H

    def r(*shape):
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).bfloat16()

    SC = 0.02
    hidden = r(B, H)
    Wq = r(Hq_tp*d, H) * SC; Wk = r(d, H) * SC
    Wv = r(d, H) * SC;       Wo = r(Hq_tp*d, H_wo) * SC
    qnw = r(d); knw = r(d); gpan = r(H)
    K_cache = r(B, 1, S, d) * 0.1
    V_cache = r(B, 1, S, d) * 0.1
    pos = torch.tensor([[S // 2]], dtype=torch.int32)
    cos_full = r(S, d); sin_full = r(S, d)
    cos = cos_full[pos[:, 0]]; sin = sin_full[pos[:, 0]]

    device = "xla"
    output = torch.zeros(B, 1, H_wo, dtype=torch.bfloat16).to(device)
    inputs = [
        hidden.reshape(B, 1, H).to(device),
        Wq.to(device), Wk.to(device), Wv.to(device), Wo.to(device),
        qnw.to(device), knw.to(device), gpan.to(device),
        K_cache.clone().to(device), V_cache.clone().to(device),
        cos.to(device), sin.to(device), pos.to(device),
        output,
    ]

    print("Warming up + benchmarking v13bc_kv_norm...")
    bench = wrap_benchmark(v13bc_kv_norm_wrapper[2], warmup=5, iters=50)
    bench(*inputs)

    r_ = bench.last_result
    print("\n=== v13bc_kv_norm benchmark ===")
    print(f"device_time_us:     {r_.device_time_us:.2f}")
    print(f"tensor_engine_pct:  {r_.tensor_engine_pct:.1f}%")
    print(f"dma_active_pct:     {r_.dma_active_pct:.1f}%")
    print(f"spill_bytes:        {r_.spill_bytes}")
    mfu = r_.prof.get("mfu_estimated_percent")
    if mfu is not None:
        print(f"mfu_estimated_pct:  {mfu:.2f}%")
    hr = r_.prof.get("hbm_read_bytes")
    hw = r_.prof.get("hbm_write_bytes")
    if hr is not None:
        print(f"hbm_read_KiB:       {hr/1024:.1f}")
    if hw is not None:
        print(f"hbm_write_KiB:      {hw/1024:.1f}")


if __name__ == "__main__":
    main()
