import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v16a"
)
from benchmark import wrap_benchmark
import nki, nki.language as nl, nki.isa as nisa
import torch, numpy as np
import torch_xla.core.xla_model as xm
from nkilib.core.utils.allocator import SbufManager
from v16a_seq_parallel import qwen3_attn_tkg_fused_oproj_v16a_seq_parallel

PMAX = 128

# LNC to benchmark — set via LNC_OVERRIDE env var (default=2)
_lnc = int(os.environ.get("LNC_OVERRIDE", "2"))

@nki.jit
def v16a_wrapper(hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
                 gamma_pre_attn, K_cache, V_cache, cos, sin, position_ids, output):
    B = cos.shape[0]; H = Wq.shape[1]; H_wo = Wo.shape[1]
    num_h_tiles = H // PMAX; num_out_cols = H_wo // PMAX
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("wrapper")
    hidden_col = hidden_states.reshape((H, B))
    hidden_sb = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="hidden_sb")
    nisa.dma_copy(dst=hidden_sb, src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0), dge_mode=nisa.dge_mode.hwdge)
    out_sb = qwen3_attn_tkg_fused_oproj_v16a_seq_parallel(
        hidden_sb=hidden_sb, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        gamma_pre_attn=gamma_pre_attn, K_cache=K_cache, V_cache=V_cache,
        cos=cos, sin=sin, position_ids=position_ids, sbm=sbm)
    output_flat = output.reshape((B, H_wo))
    for j in range(num_out_cols):
        col_psum = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(col_psum, out_sb[0:PMAX, j:j+1])
        col_sb = sbm.alloc_stack((1, PMAX), nl.bfloat16, name=f"col_sb_{j}")
        nisa.tensor_copy(col_sb, col_psum)
        nisa.dma_copy(dst=output_flat[0:B, j*PMAX:(j+1)*PMAX], src=col_sb, dge_mode=nisa.dge_mode.hwdge)
    sbm.close_scope(); sbm.close_scope()
    return output, K_cache, V_cache

v16a_wrapper = wrap_benchmark(v16a_wrapper[_lnc], warmup=5, iters=50)

def tile_transpose(W, n_heads, d, n_tiles):
    return (W.reshape(n_heads, d, n_tiles, d)
              .permute(0, 3, 2, 1)
              .reshape(n_heads * d, n_tiles * d)
              .contiguous())

rng = np.random.default_rng(42)
B, H, d, Hq_tp, S, H_wo = 1, 2048, 128, 8, 640, 2048
SC = 0.02
def r(*shape): return torch.from_numpy((rng.standard_normal(shape)*SC).astype(np.float32)).bfloat16()
device = xm.xla_device()

n_tiles = H // d  # 16
Wq_raw = r(Hq_tp*d, H); Wk_raw = r(d, H); Wv_raw = r(d, H); Wo = r(Hq_tp*d, H_wo)
Wq = tile_transpose(Wq_raw, Hq_tp, d, n_tiles).to(device)
Wk = tile_transpose(Wk_raw, 1,     d, n_tiles).to(device)
Wv = tile_transpose(Wv_raw, 1,     d, n_tiles).to(device)
Wo = Wo.to(device)

hidden = r(B,1,H).to(device)
qnw = r(d).to(device); knw = r(d).to(device); gpan = r(H).to(device)
K_cache = r(B,1,S,d).to(device); V_cache = r(B,1,S,d).to(device)
rng2 = np.random.default_rng(1234)
cos_f = torch.from_numpy(rng2.standard_normal((S,d)).astype(np.float32)).bfloat16()
sin_f = torch.from_numpy(rng2.standard_normal((S,d)).astype(np.float32)).bfloat16()
pos_val = 320
cos = cos_f[pos_val:pos_val+1].to(device)
sin = sin_f[pos_val:pos_val+1].to(device)
pos = torch.tensor([[pos_val]], dtype=torch.int32).to(device)
output = torch.zeros(B,1,H_wo,dtype=torch.bfloat16).to(device)

print(f"\n=== bench_v16a  LNC={_lnc} ===")
v16a_wrapper(hidden,Wq,Wk,Wv,Wo,qnw,knw,gpan,K_cache,V_cache,cos,sin,pos,output)
xm.mark_step()

r_res = v16a_wrapper.last_result
if r_res:
    ve = r_res.prof.get('vector_engine_active_time_percent', 0)
    mbu = r_res.prof.get('mbu_estimated_percent', 0)
    print(f"device_time_us    = {r_res.device_time_us:.2f}")
    print(f"tensor_engine_pct = {r_res.tensor_engine_pct:.1f}%")
    print(f"vector_engine_pct = {ve:.1f}%")
    print(f"dma_active_pct    = {r_res.dma_active_pct:.1f}%")
    print(f"spill_bytes       = {r_res.spill_bytes}")
    print(f"mbu_estimated     = {mbu:.3f}%")
    print(f"hbm_read_KiB      = {r_res.prof.get('hbm_read_bytes',0)/1024:.1f}")
    print(f"hbm_write_KiB     = {r_res.prof.get('hbm_write_bytes',0)/1024:.1f}")
else:
    print("WARNING: last_result is None — NTFF was not captured")
