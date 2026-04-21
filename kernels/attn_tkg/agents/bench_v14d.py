"""
bench_v14d.py — Benchmark v14d_kv_norm_hoisted_weights in two modes:
  Mode 1: back-compat  (all new weight kwargs = None)
  Mode 2: hoisted      (Wk, Wv, Wq heads, Wo heads pre-loaded in harness)
"""
import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_bench_out_v14d"
)
from benchmark import wrap_benchmark
import nki, nki.language as nl, nki.isa as nisa
import torch, numpy as np
import torch_xla.core.xla_model as xm
from nkilib.core.utils.allocator import SbufManager
from v14d_kv_norm_hoisted_weights import qwen3_attn_tkg_fused_oproj_v13bc_kv_norm_hoisted_weights as v14d_kernel

PMAX = 128


def tile_transpose(W, n_heads, d, n_tiles):
    return (W.reshape(n_heads, d, n_tiles, d)
              .permute(0, 3, 2, 1)
              .reshape(n_heads * d, n_tiles * d)
              .contiguous())


# ---------------------------------------------------------------------------
# Mode 1: back-compat (all new weight kwargs = None)
# ---------------------------------------------------------------------------
@nki.jit
def v14d_backcompat_wrapper(hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
                             gamma_pre_attn, K_cache, V_cache, cos, sin, position_ids, output):
    B = cos.shape[0]; H = Wq.shape[1]; H_wo = Wo.shape[1]
    num_h_tiles = H // PMAX; num_out_cols = H_wo // PMAX
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("wrapper")

    hidden_col = hidden_states.reshape((H, B))
    hidden_sb = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="hidden_sb")
    nisa.dma_copy(dst=hidden_sb,
                  src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge)

    out_sb = v14d_kernel(
        hidden_sb=hidden_sb, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        gamma_pre_attn=gamma_pre_attn, K_cache=K_cache, V_cache=V_cache,
        cos=cos, sin=sin, position_ids=position_ids, sbm=sbm,
        wk_sb=None, wv_sb=None, wq_heads_sb=None, wo_heads_sb=None,
    )
    output_flat = output.reshape((B, H_wo))
    for j in range(num_out_cols):
        col_psum = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(col_psum, out_sb[0:PMAX, j:j + 1])
        col_sb = sbm.alloc_stack((1, PMAX), nl.bfloat16, name=f"col_sb_{j}")
        nisa.tensor_copy(col_sb, col_psum)
        nisa.dma_copy(dst=output_flat[0:B, j * PMAX:(j + 1) * PMAX],
                      src=col_sb, dge_mode=nisa.dge_mode.hwdge)
    sbm.close_scope(); sbm.close_scope()
    return output, K_cache, V_cache


# ---------------------------------------------------------------------------
# Mode 2: hoisted weights
# ---------------------------------------------------------------------------
@nki.jit
def v14d_hoisted_wrapper(hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
                          gamma_pre_attn, K_cache, V_cache, cos, sin, position_ids, output):
    B = cos.shape[0]; H = Wq.shape[1]; H_wo = Wo.shape[1]
    num_h_tiles = H // PMAX; num_out_cols = H_wo // PMAX
    Hq_tp = Wq.shape[0] // PMAX

    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)
    if n_prgs == 1:
        owned_heads = [0, 1, 2, 3, 4, 5, 6, 7]
    elif prg_id == 0:
        owned_heads = [0, 1, 2, 3]
    else:
        owned_heads = [4, 5, 6, 7]

    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("wrapper")

    hidden_col = hidden_states.reshape((H, B))
    hidden_sb = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="hidden_sb")
    nisa.dma_copy(dst=hidden_sb,
                  src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge)

    # Pre-load weight matrices
    wk_loaded = sbm.alloc_stack((PMAX, num_h_tiles * PMAX), nl.bfloat16, name="wk_loaded")
    nisa.dma_copy(dst=wk_loaded, src=Wk, dge_mode=nisa.dge_mode.hwdge)

    wv_loaded = sbm.alloc_stack((PMAX, num_h_tiles * PMAX), nl.bfloat16, name="wv_loaded")
    nisa.dma_copy(dst=wv_loaded, src=Wv, dge_mode=nisa.dge_mode.hwdge)

    wq_loaded_list = []
    for q_h in owned_heads:
        w = sbm.alloc_stack((PMAX, num_h_tiles * PMAX), nl.bfloat16, name=f"wq_h_{q_h}")
        nisa.dma_copy(dst=w, src=Wq[q_h * PMAX:(q_h + 1) * PMAX, :], dge_mode=nisa.dge_mode.hwdge)
        wq_loaded_list.append(w)

    Wo_reshaped = Wo.reshape((Hq_tp, PMAX, H_wo))
    wo_loaded_list = []
    for head in owned_heads:
        wo_tile = sbm.alloc_stack((PMAX, H_wo), nl.bfloat16, name=f"wo_h_{head}")
        nisa.dma_copy(
            dst=wo_tile,
            src=Wo_reshaped.ap(pattern=[[H_wo, PMAX], [1, H_wo]], offset=head * PMAX * H_wo),
            dge_mode=nisa.dge_mode.hwdge,
        )
        wo_loaded_list.append(wo_tile)

    # Pre-load layer-invariant scalar constants
    cos_col = cos.reshape((PMAX, B))
    sin_col = sin.reshape((PMAX, B))

    qnw_bf16_h = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="qnw_bf16_h")
    nisa.dma_copy(dst=qnw_bf16_h, src=q_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
    qnw_f32_h = sbm.alloc_stack((PMAX, 1), nl.float32, name="qnw_f32_h")
    nisa.tensor_copy(qnw_f32_h, qnw_bf16_h)

    knw_bf16_h = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="knw_bf16_h")
    nisa.dma_copy(dst=knw_bf16_h, src=k_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
    knw_f32_h = sbm.alloc_stack((PMAX, 1), nl.float32, name="knw_f32_h")
    nisa.tensor_copy(knw_f32_h, knw_bf16_h)

    cos_bf16_h = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="cos_bf16_h")
    nisa.dma_copy(dst=cos_bf16_h, src=cos_col, dge_mode=nisa.dge_mode.hwdge)
    cos_f32_h = sbm.alloc_stack((PMAX, 1), nl.float32, name="cos_f32_h")
    nisa.tensor_copy(cos_f32_h, cos_bf16_h)

    sin_bf16_h = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="sin_bf16_h")
    nisa.dma_copy(dst=sin_bf16_h, src=sin_col, dge_mode=nisa.dge_mode.hwdge)
    sin_f32_h = sbm.alloc_stack((PMAX, 1), nl.float32, name="sin_f32_h")
    nisa.tensor_copy(sin_f32_h, sin_bf16_h)

    gpan_bf16_h = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="gpan_bf16_h")
    nisa.dma_copy(dst=gpan_bf16_h,
                  src=gamma_pre_attn.reshape((H, 1)).ap(
                      pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge)
    gpan_f32_h = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="gpan_f32_h")
    nisa.tensor_copy(gpan_f32_h, gpan_bf16_h)

    out_sb = v14d_kernel(
        hidden_sb=hidden_sb, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        gamma_pre_attn=gamma_pre_attn, K_cache=K_cache, V_cache=V_cache,
        cos=cos, sin=sin, position_ids=position_ids, sbm=sbm,
        qnw_f32_sb=qnw_f32_h, knw_f32_sb=knw_f32_h,
        cos_f32_sb=cos_f32_h, sin_f32_sb=sin_f32_h, gpan_f32_sb=gpan_f32_h,
        wk_sb=wk_loaded, wv_sb=wv_loaded,
        wq_heads_sb=wq_loaded_list, wo_heads_sb=wo_loaded_list,
    )
    output_flat = output.reshape((B, H_wo))
    for j in range(num_out_cols):
        col_psum = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(col_psum, out_sb[0:PMAX, j:j + 1])
        col_sb = sbm.alloc_stack((1, PMAX), nl.bfloat16, name=f"col_sb_{j}")
        nisa.tensor_copy(col_sb, col_psum)
        nisa.dma_copy(dst=output_flat[0:B, j * PMAX:(j + 1) * PMAX],
                      src=col_sb, dge_mode=nisa.dge_mode.hwdge)
    sbm.close_scope(); sbm.close_scope()
    return output, K_cache, V_cache


v14d_backcompat_wrapper = wrap_benchmark(v14d_backcompat_wrapper[2], warmup=5, iters=50)
v14d_hoisted_wrapper    = wrap_benchmark(v14d_hoisted_wrapper[2],    warmup=5, iters=50)

rng = np.random.default_rng(42)
B, H, d, Hq_tp, S, H_wo = 1, 2048, 128, 8, 640, 2048
SC = 0.02
def r(*shape): return torch.from_numpy((rng.standard_normal(shape)*SC).astype(np.float32)).bfloat16()
device = xm.xla_device()
n_tiles = H // d

Wq_raw = r(Hq_tp*d, H); Wk_raw = r(d, H); Wv_raw = r(d, H); Wo_raw = r(Hq_tp*d, H_wo)
Wq = tile_transpose(Wq_raw, Hq_tp, d, n_tiles).to(device)
Wk = tile_transpose(Wk_raw, 1,     d, n_tiles).to(device)
Wv = tile_transpose(Wv_raw, 1,     d, n_tiles).to(device)
Wo = Wo_raw.to(device)

hidden  = r(B, 1, H).to(device)
qnw     = r(d).to(device); knw = r(d).to(device); gpan = r(H).to(device)
K_cache = r(B, 1, S, d).to(device); V_cache = r(B, 1, S, d).to(device)
rng2 = np.random.default_rng(1234)
cos_f = torch.from_numpy(rng2.standard_normal((S,d)).astype(np.float32)).bfloat16()
sin_f = torch.from_numpy(rng2.standard_normal((S,d)).astype(np.float32)).bfloat16()
pos_val = 320
cos = cos_f[pos_val:pos_val+1].to(device); sin = sin_f[pos_val:pos_val+1].to(device)
pos = torch.tensor([[pos_val]], dtype=torch.int32).to(device)
output = torch.zeros(B, 1, H_wo, dtype=torch.bfloat16).to(device)


def print_result(name, res):
    if res:
        mfu = res.prof.get('mfu_estimated_percent', 0)
        ve  = res.prof.get('vector_engine_active_time_percent', 0)
        mbu = res.prof.get('mbu_estimated_percent', 0)
        print(f"\n[{name}]")
        print(f"device_time_us    = {res.device_time_us:.2f}")
        print(f"tensor_engine_pct = {res.tensor_engine_pct:.1f}%")
        print(f"vector_engine_pct = {ve:.1f}%")
        print(f"dma_active_pct    = {res.dma_active_pct:.1f}%")
        print(f"spill_bytes       = {res.spill_bytes}")
        print(f"mfu_estimated     = {mfu:.2f}%")
        print(f"mbu_estimated     = {mbu:.3f}%")
        print(f"hbm_read_KiB      = {res.prof.get('hbm_read_bytes',  0)/1024:.1f}")
        print(f"hbm_write_KiB     = {res.prof.get('hbm_write_bytes', 0)/1024:.1f}")


# --- Mode 1: back-compat ---
v14d_backcompat_wrapper(hidden, Wq, Wk, Wv, Wo, qnw, knw, gpan,
                        K_cache, V_cache, cos, sin, pos, output)
xm.mark_step()
print_result("v14d back-compat", v14d_backcompat_wrapper.last_result)

# Reset caches for mode 2
K_cache2 = r(B, 1, S, d).to(device); V_cache2 = r(B, 1, S, d).to(device)
output2 = torch.zeros(B, 1, H_wo, dtype=torch.bfloat16).to(device)

# --- Mode 2: hoisted weights ---
v14d_hoisted_wrapper(hidden, Wq, Wk, Wv, Wo, qnw, knw, gpan,
                     K_cache2, V_cache2, cos, sin, pos, output2)
xm.mark_step()
print_result("v14d hoisted weights", v14d_hoisted_wrapper.last_result)
