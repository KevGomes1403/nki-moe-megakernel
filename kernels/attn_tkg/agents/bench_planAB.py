import os, sys

# MUST be before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_planAB"
)

from benchmark import wrap_benchmark
import nki, nki.language as nl, nki.isa as nisa
import torch, numpy as np
import torch_xla.core.xla_model as xm
from nkilib.core.utils.allocator import SbufManager
from v14b_kv_norm_planAB import qwen3_attn_tkg_fused_oproj_v13bc_kv_norm

PMAX = 128

@nki.jit
def v14b_wrapper(
    hidden_states, Wq, Wk, Wv, Wo,
    q_norm_weight, k_norm_weight, gamma_pre_attn,
    K_cache, V_cache, cos, sin, position_ids, output,
):
    B = cos.shape[0]; H = Wq.shape[1]; H_wo = Wo.shape[1]
    num_h_tiles = H // PMAX; num_out_cols = H_wo // PMAX
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("wrapper")
    hidden_col = hidden_states.reshape((H, B))
    hidden_sb = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="hidden_sb")
    nisa.dma_copy(
        dst=hidden_sb,
        src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )
    out_sb = qwen3_attn_tkg_fused_oproj_v13bc_kv_norm(
        hidden_sb=hidden_sb, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
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
        nisa.dma_copy(dst=output_flat[0:B, j*PMAX:(j+1)*PMAX], src=col_sb, dge_mode=nisa.dge_mode.hwdge)
    sbm.close_scope()
    sbm.close_scope()
    return output, K_cache, V_cache

v14b_wrapper_lnc2 = wrap_benchmark(v14b_wrapper[2], warmup=5, iters=50)

# Create inputs
rng = np.random.default_rng(42)
B, H, d, Hq_tp, S, H_wo = 1, 2048, 128, 8, 640, 2048
SC = 0.02
def r(*shape): return torch.from_numpy((rng.standard_normal(shape)*SC).astype(np.float32)).bfloat16()

device = xm.xla_device()
hidden = r(B, 1, H).to(device)
Wq = r(Hq_tp*d, H).to(device); Wk = r(d, H).to(device)
Wv = r(d, H).to(device); Wo = r(Hq_tp*d, H_wo).to(device)
qnw = r(d).to(device); knw = r(d).to(device); gpan = r(H).to(device)
K_cache = r(B, 1, S, d).to(device); V_cache = r(B, 1, S, d).to(device)
rng2 = np.random.default_rng(1234)
cos_full = torch.from_numpy(rng2.standard_normal((S, d)).astype(np.float32)).bfloat16()
sin_full = torch.from_numpy(rng2.standard_normal((S, d)).astype(np.float32)).bfloat16()
pos_val = 320
cos = cos_full[pos_val:pos_val+1].to(device)
sin = sin_full[pos_val:pos_val+1].to(device)
pos = torch.tensor([[pos_val]], dtype=torch.int32).to(device)
output = torch.zeros(B, 1, H_wo, dtype=torch.bfloat16).to(device)

v14b_wrapper_lnc2(hidden, Wq, Wk, Wv, Wo, qnw, knw, gpan, K_cache, V_cache, cos, sin, pos, output)
xm.mark_step()

r_result = v14b_wrapper_lnc2.last_result
if r_result:
    print(f"device_time_us    = {r_result.device_time_us:.2f}")
    print(f"tensor_engine_pct = {r_result.tensor_engine_pct:.1f}%")
    print(f"dma_active_pct    = {r_result.dma_active_pct:.1f}%")
    print(f"spill_bytes       = {r_result.spill_bytes}")
    mfu = r_result.prof.get('mfu_estimated_percent', 0)
    mbu = r_result.prof.get('mbu_estimated_percent', 0)
    hbm_r = r_result.prof.get('hbm_read_bytes', 0)
    hbm_w = r_result.prof.get('hbm_write_bytes', 0)
    ve_pct = r_result.prof.get('vector_engine_active_time_percent', 0)
    print(f"vector_engine_pct = {ve_pct:.1f}%")
    print(f"mfu_estimated     = {mfu:.3f}%")
    print(f"mbu_estimated     = {mbu:.3f}%")
    print(f"hbm_read_KiB      = {hbm_r/1024:.1f}")
    print(f"hbm_write_KiB     = {hbm_w/1024:.1f}")
else:
    print("Benchmark result not available")
