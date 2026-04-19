"""
bench_v14c.py — Benchmark v14c_kv_norm_hoisted in two modes:
  Mode 1 (back-compat):  all *_f32_sb = None  — HBM loads happen inside kernel
  Mode 2 (hoisted):      all 5 constants pre-loaded before the kernel call

Also records v14b baseline for comparison.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
    python bench_v14c.py
"""

import os, sys

# MUST be set before any neuron/torch_xla import
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"

# We run two kernels sequentially in this script, using different output dirs
# to avoid NTFF collision.
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))

from benchmark import wrap_benchmark
import nki, nki.language as nl, nki.isa as nisa
import torch, numpy as np
import torch_xla.core.xla_model as xm
from nkilib.core.utils.allocator import SbufManager

# Import both kernels
from v14b_kv_norm_pretransposed import qwen3_attn_tkg_fused_oproj_v13bc_kv_norm_pretransposed as v14b_kernel
from v14c_kv_norm_hoisted import qwen3_attn_tkg_fused_oproj_v13bc_kv_norm_hoisted as v14c_kernel

PMAX = 128


# ---------------------------------------------------------------------------
# Helper: tile-transpose weights
# ---------------------------------------------------------------------------
def tile_transpose(W, n_heads, d, n_tiles):
    return (W.reshape(n_heads, d, n_tiles, d)
              .permute(0, 3, 2, 1)
              .reshape(n_heads * d, n_tiles * d)
              .contiguous())


# ---------------------------------------------------------------------------
# Common wrapper body (output projection transpose loop)
# ---------------------------------------------------------------------------
def _emit_output_transpose(sbm, out_sb, output, B, H_wo, num_out_cols):
    output_flat = output.reshape((B, H_wo))
    for j in range(num_out_cols):
        col_psum = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(col_psum, out_sb[0:PMAX, j:j + 1])
        col_sb = sbm.alloc_stack((1, PMAX), nl.bfloat16, name=f"col_sb_{j}")
        nisa.tensor_copy(col_sb, col_psum)
        nisa.dma_copy(
            dst=output_flat[0:B, j * PMAX:(j + 1) * PMAX],
            src=col_sb,
            dge_mode=nisa.dge_mode.hwdge,
        )


# ---------------------------------------------------------------------------
# v14b baseline wrapper
# ---------------------------------------------------------------------------
@nki.jit
def v14b_wrapper(hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
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
    out_sb = v14b_kernel(
        hidden_sb=hidden_sb, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        gamma_pre_attn=gamma_pre_attn, K_cache=K_cache, V_cache=V_cache,
        cos=cos, sin=sin, position_ids=position_ids, sbm=sbm)
    _emit_output_transpose(sbm, out_sb, output, B, H_wo, num_out_cols)
    sbm.close_scope(); sbm.close_scope()
    return output, K_cache, V_cache


# ---------------------------------------------------------------------------
# v14c back-compat wrapper (all *_f32_sb = None)
# ---------------------------------------------------------------------------
@nki.jit
def v14c_backcompat_wrapper(hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
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
    out_sb = v14c_kernel(
        hidden_sb=hidden_sb, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        gamma_pre_attn=gamma_pre_attn, K_cache=K_cache, V_cache=V_cache,
        cos=cos, sin=sin, position_ids=position_ids, sbm=sbm,
        # All pre-loaded tensors = None → HBM loads happen inside kernel
        qnw_f32_sb=None, knw_f32_sb=None,
        cos_f32_sb=None, sin_f32_sb=None, gpan_f32_sb=None,
    )
    _emit_output_transpose(sbm, out_sb, output, B, H_wo, num_out_cols)
    sbm.close_scope(); sbm.close_scope()
    return output, K_cache, V_cache


# ---------------------------------------------------------------------------
# v14c hoisted wrapper (all 5 constants pre-loaded)
# ---------------------------------------------------------------------------
@nki.jit
def v14c_hoisted_wrapper(hidden_states, Wq, Wk, Wv, Wo, q_norm_weight, k_norm_weight,
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

    # Pre-load all 5 layer-invariant constants (same code as was inside v14b)
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

    out_sb = v14c_kernel(
        hidden_sb=hidden_sb, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        gamma_pre_attn=gamma_pre_attn, K_cache=K_cache, V_cache=V_cache,
        cos=cos, sin=sin, position_ids=position_ids, sbm=sbm,
        # All 5 pre-loaded f32 tensors provided
        qnw_f32_sb=qnw_f32_h, knw_f32_sb=knw_f32_h,
        cos_f32_sb=cos_f32_h, sin_f32_sb=sin_f32_h, gpan_f32_sb=gpan_f32_h,
    )
    _emit_output_transpose(sbm, out_sb, output, B, H_wo, num_out_cols)
    sbm.close_scope(); sbm.close_scope()
    return output, K_cache, V_cache


# ---------------------------------------------------------------------------
# Build inputs
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
B, H, d, Hq_tp, S, H_wo = 1, 2048, 128, 8, 640, 2048
SC = 0.02

def r(*shape):
    return torch.from_numpy((rng.standard_normal(shape) * SC).astype(np.float32)).bfloat16()

device = xm.xla_device()
n_tiles = H // d  # 16

Wq_raw = r(Hq_tp * d, H); Wk_raw = r(d, H); Wv_raw = r(d, H); Wo_raw = r(Hq_tp * d, H_wo)
# Pre-transpose Wq, Wk, Wv (tile-transposed layout)
Wq = tile_transpose(Wq_raw, Hq_tp, d, n_tiles).to(device)
Wk = tile_transpose(Wk_raw, 1,     d, n_tiles).to(device)
Wv = tile_transpose(Wv_raw, 1,     d, n_tiles).to(device)
Wo = Wo_raw.to(device)

hidden = r(B, 1, H).to(device)
qnw    = r(d).to(device)
knw    = r(d).to(device)
gpan   = r(H).to(device)
K_cache = r(B, 1, S, d).to(device)
V_cache = r(B, 1, S, d).to(device)

rng2 = np.random.default_rng(1234)
cos_full = torch.from_numpy(rng2.standard_normal((S, d)).astype(np.float32)).bfloat16()
sin_full = torch.from_numpy(rng2.standard_normal((S, d)).astype(np.float32)).bfloat16()
pos_val  = 320
cos = cos_full[pos_val:pos_val + 1].to(device)
sin = sin_full[pos_val:pos_val + 1].to(device)
pos = torch.tensor([[pos_val]], dtype=torch.int32).to(device)
output = torch.zeros(B, 1, H_wo, dtype=torch.bfloat16).to(device)


def print_result(label, r_res):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    if r_res is None:
        print("  [no profile data — NTFF not captured]")
        return
    mfu = r_res.prof.get('mfu_estimated_percent', 0)
    ve  = r_res.prof.get('vector_engine_active_time_percent', 0)
    mbu = r_res.prof.get('mbu_estimated_percent', 0)
    print(f"  device_time_us    = {r_res.device_time_us:.2f}")
    print(f"  tensor_engine_pct = {r_res.tensor_engine_pct:.1f}%")
    print(f"  vector_engine_pct = {ve:.1f}%")
    print(f"  dma_active_pct    = {r_res.dma_active_pct:.1f}%")
    print(f"  spill_bytes       = {r_res.spill_bytes}")
    print(f"  mfu_estimated     = {mfu:.2f}%")
    print(f"  mbu_estimated     = {mbu:.3f}%")
    print(f"  hbm_read_KiB      = {r_res.prof.get('hbm_read_bytes',  0)/1024:.1f}")
    print(f"  hbm_write_KiB     = {r_res.prof.get('hbm_write_bytes', 0)/1024:.1f}")


# ---------------------------------------------------------------------------
# Run kernels sequentially (NeuronCores exclusive — one at a time)
# ---------------------------------------------------------------------------

# ---- v14b baseline ----
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(_BENCH_DIR, "_bench_out_v14b_baseline")
v14b_benchmarked = wrap_benchmark(v14b_wrapper[2], warmup=5, iters=50)
v14b_benchmarked(hidden, Wq, Wk, Wv, Wo, qnw, knw, gpan,
                 K_cache.clone(), V_cache.clone(), cos, sin, pos, output.clone())
xm.mark_step()
r_v14b = v14b_benchmarked.last_result
print_result("v14b baseline", r_v14b)

# ---- v14c back-compat ----
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(_BENCH_DIR, "_bench_out_v14c_backcompat")
v14c_bc_benchmarked = wrap_benchmark(v14c_backcompat_wrapper[2], warmup=5, iters=50)
v14c_bc_benchmarked(hidden, Wq, Wk, Wv, Wo, qnw, knw, gpan,
                    K_cache.clone(), V_cache.clone(), cos, sin, pos, output.clone())
xm.mark_step()
r_v14c_bc = v14c_bc_benchmarked.last_result
print_result("v14c back-compat (all *_f32_sb=None)", r_v14c_bc)

# ---- v14c hoisted ----
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = os.path.join(_BENCH_DIR, "_bench_out_v14c_hoisted")
v14c_h_benchmarked = wrap_benchmark(v14c_hoisted_wrapper[2], warmup=5, iters=50)
v14c_h_benchmarked(hidden, Wq, Wk, Wv, Wo, qnw, knw, gpan,
                   K_cache.clone(), V_cache.clone(), cos, sin, pos, output.clone())
xm.mark_step()
r_v14c_h = v14c_h_benchmarked.last_result
print_result("v14c hoisted (all *_f32_sb provided)", r_v14c_h)

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
print("\n\n" + "="*70)
print("  COMPARISON TABLE")
print("="*70)
print(f"  {'Kernel':<38} {'time_us':>8} {'te%':>6} {'dma%':>6} {'spill':>8} {'mfu%':>6}")
print(f"  {'-'*38} {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")

for label, r_res in [
    ("v14b baseline",           r_v14b),
    ("v14c back-compat",        r_v14c_bc),
    ("v14c hoisted",            r_v14c_h),
]:
    if r_res is None:
        print(f"  {label:<38}  [no data]")
    else:
        mfu = r_res.prof.get('mfu_estimated_percent', 0)
        print(f"  {label:<38} {r_res.device_time_us:>8.2f} "
              f"{r_res.tensor_engine_pct:>6.1f} {r_res.dma_active_pct:>6.1f} "
              f"{r_res.spill_bytes:>8} {mfu:>6.2f}")
print("="*70)
