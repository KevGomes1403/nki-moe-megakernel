"""
Correctness test for v14d_kv_norm_hoisted_weights — full hoisted mode on trn3.

Tests the kernel exactly as called in transformer_qwen_multilayer.py:
  - All scalar constants pre-hoisted into f32 SBUF (qnw_f32_sb, knw_f32_sb,
    cos_f32_sb, sin_f32_sb, gpan_f32_sb all supplied)
  - All weight matrices pre-hoisted into bf16 SBUF (wk_sb, wv_sb,
    wq_heads_sb, wo_heads_sb all supplied)
  - LNC=2 via kernel[2] on real trn3 hardware (no simulator)
  - Position sweep: [0, 1, 63, 64, 127, 128, 129, 255, 320]
  - Accuracy checked against fp32-promoted PyTorch reference: atol=1e-2, rtol=1e-2

Run:
    python test_v14d_full_hoisted_trn3.py
"""

import os, sys
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import nki
import nki.language as nl
import nki.isa as nisa
from nkilib.core.utils.allocator import SbufManager
from v14d_kv_norm_hoisted_weights import qwen3_attn_tkg_fused_oproj_v13bc_kv_norm_hoisted_weights as v14d_kernel

PMAX = 128
EPS  = 1e-6


def tile_transpose(W, n_heads, d, n_tiles):
    return (W.reshape(n_heads, d, n_tiles, d)
              .permute(0, 3, 2, 1)
              .reshape(n_heads * d, n_tiles * d)
              .contiguous())


# ---------------------------------------------------------------------------
# NKI wrapper — all hoisted kwargs supplied, matches multilayer call site
# ---------------------------------------------------------------------------
@nki.jit
def v14d_full_hoisted_wrapper(
    hidden_states,   # [B, 1, H]        bf16 HBM
    Wq,              # [Hq_tp*d, H]     bf16 HBM tile-transposed
    Wk,              # [PMAX, NH*PMAX]  bf16 HBM tile-transposed
    Wv,              # [PMAX, NH*PMAX]  bf16 HBM tile-transposed
    Wo,              # [Hq_tp*d, H]     bf16 HBM plain
    q_norm_weight,   # [d]              bf16 HBM
    k_norm_weight,   # [d]              bf16 HBM
    gamma_pre_attn,  # [H]              bf16 HBM
    K_cache, V_cache,
    cos, sin,        # [B, d]           bf16 HBM
    position_ids,    # [B, 1]           int32 HBM
    output,          # [B, 1, H]        bf16 HBM
):
    B          = cos.shape[0]
    H          = Wq.shape[1]
    H_wo       = Wo.shape[1]
    num_h_tiles = H // PMAX
    num_out_cols = H_wo // PMAX
    Hq_tp      = Wq.shape[0] // PMAX

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

    # Load hidden into SBUF — same column layout as multilayer's residual_sb
    hidden_col = hidden_states.reshape((H, B))
    hidden_sb  = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="hidden_sb")
    nisa.dma_copy(
        dst=hidden_sb,
        src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )

    # -------------------------------------------------------------------------
    # Hoist scalar constants — Plan A path in multilayer
    # Shapes match multilayer: qnw/knw [PMAX,1], cos/sin [PMAX,B], gpan [PMAX,NH]
    # -------------------------------------------------------------------------
    cos_col = cos.reshape((PMAX, B))
    sin_col = sin.reshape((PMAX, B))

    qnw_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="qnw_bf16")
    nisa.dma_copy(dst=qnw_bf16, src=q_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
    qnw_f32  = sbm.alloc_stack((PMAX, 1), nl.float32,  name="qnw_f32")
    nisa.tensor_copy(qnw_f32, qnw_bf16)

    knw_bf16 = sbm.alloc_stack((PMAX, 1), nl.bfloat16, name="knw_bf16")
    nisa.dma_copy(dst=knw_bf16, src=k_norm_weight.reshape((PMAX, 1)), dge_mode=nisa.dge_mode.hwdge)
    knw_f32  = sbm.alloc_stack((PMAX, 1), nl.float32,  name="knw_f32")
    nisa.tensor_copy(knw_f32, knw_bf16)

    cos_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="cos_bf16")
    nisa.dma_copy(dst=cos_bf16, src=cos_col, dge_mode=nisa.dge_mode.hwdge)
    cos_f32  = sbm.alloc_stack((PMAX, B), nl.float32,  name="cos_f32")
    nisa.tensor_copy(cos_f32, cos_bf16)

    sin_bf16 = sbm.alloc_stack((PMAX, B), nl.bfloat16, name="sin_bf16")
    nisa.dma_copy(dst=sin_bf16, src=sin_col, dge_mode=nisa.dge_mode.hwdge)
    sin_f32  = sbm.alloc_stack((PMAX, B), nl.float32,  name="sin_f32")
    nisa.tensor_copy(sin_f32, sin_bf16)

    # gpan AP pattern matches multilayer lines 163-170 and v14d internal load
    gpan_bf16 = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="gpan_bf16")
    nisa.dma_copy(
        dst=gpan_bf16,
        src=gamma_pre_attn.reshape((H, 1)).ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )
    gpan_f32  = sbm.alloc_stack((PMAX, num_h_tiles), nl.float32, name="gpan_f32")
    nisa.tensor_copy(gpan_f32, gpan_bf16)

    # -------------------------------------------------------------------------
    # Hoist weight matrices — Plan B path in multilayer
    # -------------------------------------------------------------------------
    wk_sb = sbm.alloc_stack((PMAX, num_h_tiles * PMAX), nl.bfloat16, name="wk_sb")
    nisa.dma_copy(dst=wk_sb, src=Wk, dge_mode=nisa.dge_mode.hwdge)

    wv_sb = sbm.alloc_stack((PMAX, num_h_tiles * PMAX), nl.bfloat16, name="wv_sb")
    nisa.dma_copy(dst=wv_sb, src=Wv, dge_mode=nisa.dge_mode.hwdge)

    wq_heads = []
    for q_h in owned_heads:
        w = sbm.alloc_stack((PMAX, num_h_tiles * PMAX), nl.bfloat16, name=f"wq_h_{q_h}")
        nisa.dma_copy(dst=w, src=Wq[q_h * PMAX:(q_h + 1) * PMAX, :], dge_mode=nisa.dge_mode.hwdge)
        wq_heads.append(w)

    # Wo AP pattern matches multilayer lines 249-256
    wo_heads = []
    for q_h in owned_heads:
        wo_tile = sbm.alloc_stack((PMAX, H_wo), nl.bfloat16, name=f"wo_h_{q_h}")
        nisa.dma_copy(
            dst=wo_tile,
            src=Wo.reshape((Hq_tp, PMAX, H_wo)).ap(
                pattern=[[H_wo, PMAX], [1, H_wo]],
                offset=q_h * PMAX * H_wo,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )
        wo_heads.append(wo_tile)

    # -------------------------------------------------------------------------
    # Call v14d with all hoisted kwargs — no HBM loads happen inside v14d
    # -------------------------------------------------------------------------
    out_sb = v14d_kernel(
        hidden_sb=hidden_sb,
        Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        gamma_pre_attn=gamma_pre_attn,
        K_cache=K_cache, V_cache=V_cache,
        cos=cos, sin=sin, position_ids=position_ids,
        sbm=sbm,
        qnw_f32_sb=qnw_f32,
        knw_f32_sb=knw_f32,
        cos_f32_sb=cos_f32,
        sin_f32_sb=sin_f32,
        gpan_f32_sb=gpan_f32,
        wk_sb=wk_sb,
        wv_sb=wv_sb,
        wq_heads_sb=wq_heads,
        wo_heads_sb=wo_heads,
    )

    # out_sb is [PMAX, num_out_cols=16] bf16; v14d already did sendrecv so both
    # cores hold the same full sum — write from both (consistent with bench_v14d)
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

    sbm.close_scope()
    sbm.close_scope()
    return output, K_cache, V_cache


# ---------------------------------------------------------------------------
# PyTorch fp32-promoted reference (identical to existing sim tests)
# ---------------------------------------------------------------------------
def pytorch_attn_ref(
    hidden_states, Wq, Wk, Wv, Wo,
    q_norm_weight, k_norm_weight, gamma_pre_attn,
    K_cache, V_cache, cos, sin, position_ids,
):
    B    = hidden_states.shape[0]
    d    = q_norm_weight.shape[0]
    Hq_tp = Wq.shape[0] // d

    if hidden_states.dim() == 3:
        hidden_states = hidden_states.squeeze(1)

    x = hidden_states.float()
    x = x * (x.pow(2).mean(-1, keepdim=True) + EPS).rsqrt() * gamma_pre_attn.float()

    cos_f = cos.float()
    sin_f = sin.float()

    k = x @ Wk.float().t()
    k = k * (k.pow(2).mean(-1, keepdim=True) + EPS).rsqrt() * k_norm_weight.float()
    k_rot = torch.cat([-k[:, d // 2:], k[:, :d // 2]], dim=-1)
    k = k * cos_f + k_rot * sin_f
    new_k = k.bfloat16()

    v     = x @ Wv.float().t()
    new_v = v.bfloat16()

    K_cache_out = K_cache.clone()
    V_cache_out = V_cache.clone()
    for b in range(B):
        pos = int(position_ids[b, 0])
        K_cache_out[b, 0, pos, :] = new_k[b]
        V_cache_out[b, 0, pos, :] = new_v[b]

    q = (x @ Wq.float().t()).reshape(B, Hq_tp, d)
    q = q * (q.pow(2).mean(-1, keepdim=True) + EPS).rsqrt() * q_norm_weight.float()
    cos_q = cos_f.unsqueeze(1).expand(-1, Hq_tp, -1)
    sin_q = sin_f.unsqueeze(1).expand(-1, Hq_tp, -1)
    q_rot = torch.cat([-q[..., d // 2:], q[..., :d // 2]], dim=-1)
    q_scaled = (q * cos_q + q_rot * sin_q) / (d ** 0.5)

    K_ctx = K_cache_out[:, 0, :, :].float()
    V_ctx = V_cache_out[:, 0, :, :].float()
    S     = K_ctx.shape[1]
    mask  = torch.zeros(B, S)
    for b in range(B):
        mask[b, int(position_ids[b, 0]) + 1:] = -1e9

    heads = []
    for h in range(Hq_tp):
        sc = (q_scaled[:, h:h + 1, :] @ K_ctx.transpose(-2, -1)).squeeze(1) + mask
        w  = F.softmax(sc, dim=-1)
        heads.append((w.unsqueeze(1) @ V_ctx).squeeze(1))

    attn_out = (torch.stack(heads, dim=1).reshape(B, Hq_tp * d) @ Wo.float()).bfloat16()
    return attn_out, new_k, new_v, K_cache_out, V_cache_out


# ---------------------------------------------------------------------------
# Per-position test runner
# ---------------------------------------------------------------------------
def run_one(pos_val, weights_cpu, weights_device, hidden_rng_seed, device):
    B, H, d, Hq_tp, S = 1, 2048, 128, 8, 640
    H_wo   = H
    n_tiles = H // d

    Wq_np, Wk_np, Wv_np, Wo_np, qnw_np, knw_np, gpan_np = weights_cpu

    rng_h = np.random.default_rng(hidden_rng_seed)
    hidden_np = rng_h.standard_normal((B, 1, H)).astype(np.float32) * 0.1

    rng_kv = np.random.default_rng(hidden_rng_seed + 1000)
    K_cache_np = rng_kv.standard_normal((B, 1, S, d)).astype(np.float32) * 0.02
    V_cache_np = rng_kv.standard_normal((B, 1, S, d)).astype(np.float32) * 0.02

    cos_full_rng = np.random.default_rng(1234)
    cos_full_np  = cos_full_rng.standard_normal((S, d)).astype(np.float32) * 0.5
    sin_full_np  = cos_full_rng.standard_normal((S, d)).astype(np.float32) * 0.5
    cos_np = cos_full_np[pos_val:pos_val + 1]
    sin_np = sin_full_np[pos_val:pos_val + 1]
    pos_np = np.array([[pos_val]], dtype=np.int32)

    def t(a): return torch.from_numpy(a.astype(np.float32)).bfloat16()

    # PyTorch reference (CPU)
    ref_out, ref_nk, ref_nv, ref_K_full, ref_V_full = pytorch_attn_ref(
        t(hidden_np), t(Wq_np), t(Wk_np), t(Wv_np), t(Wo_np),
        t(qnw_np), t(knw_np), t(gpan_np),
        t(K_cache_np).clone(), t(V_cache_np).clone(),
        t(cos_np), t(sin_np),
        torch.tensor([[pos_val]], dtype=torch.int32),
    )

    # NKI kernel on trn3
    Wq_pt, Wk_pt, Wv_pt, Wo_dev, qnw_dev, knw_dev, gpan_dev = weights_device

    hidden_dev  = t(hidden_np).to(device)
    K_cache_dev = t(K_cache_np).to(device)
    V_cache_dev = t(V_cache_np).to(device)
    cos_dev     = t(cos_np).to(device)
    sin_dev     = t(sin_np).to(device)
    pos_dev     = torch.tensor([[pos_val]], dtype=torch.int32).to(device)
    output_dev  = torch.zeros(B, 1, H_wo, dtype=torch.bfloat16).to(device)

    result_dev, K_out_dev, V_out_dev = v14d_full_hoisted_wrapper[2](
        hidden_dev, Wq_pt, Wk_pt, Wv_pt, Wo_dev,
        qnw_dev, knw_dev, gpan_dev,
        K_cache_dev, V_cache_dev,
        cos_dev, sin_dev, pos_dev, output_dev,
    )
    xm.mark_step()

    attn_nki  = result_dev.cpu().float().reshape(B, -1)
    K_full_nki = K_out_dev.cpu().float()
    V_full_nki = V_out_dev.cpu().float()

    nk_nki = K_full_nki[:, 0, pos_val, :]
    nv_nki = V_full_nki[:, 0, pos_val, :]

    K_mask = torch.ones(S, dtype=torch.bool); K_mask[pos_val] = False
    V_mask = torch.ones(S, dtype=torch.bool); V_mask[pos_val] = False
    k_other_diff = (K_full_nki[0, 0, K_mask] - t(K_cache_np)[0, 0, K_mask].float()).abs().max().item()
    v_other_diff = (V_full_nki[0, 0, V_mask] - t(V_cache_np)[0, 0, V_mask].float()).abs().max().item()

    attn_max  = (attn_nki        - ref_out.float()).abs().max().item()
    nk_max    = (nk_nki          - ref_nk.float()).abs().max().item()
    nv_max    = (nv_nki          - ref_nv.float()).abs().max().item()
    Kfull_max = (K_full_nki      - ref_K_full.float()).abs().max().item()
    Vfull_max = (V_full_nki      - ref_V_full.float()).abs().max().item()

    def close(a, b, atol=1e-3, rtol=1e-2):
        return bool(torch.all(torch.abs(a - b) <= atol + rtol * torch.abs(b)).item())

    return {
        "pos": pos_val,
        "attn_max": attn_max, "nk_max": nk_max, "nv_max": nv_max,
        "k_other_untouched_max": k_other_diff,
        "v_other_untouched_max": v_other_diff,
        "K_full_max_vs_ref": Kfull_max,
        "V_full_max_vs_ref": Vfull_max,
        "pass_attn":  close(attn_nki.reshape(B, -1), ref_out.float()),
        "pass_nk":    close(nk_nki,    ref_nk.float()),
        "pass_nv":    close(nv_nki,    ref_nv.float()),
        "pass_Kfull": close(K_full_nki, ref_K_full.float()),
        "pass_Vfull": close(V_full_nki, ref_V_full.float()),
    }


def main():
    device = xm.xla_device()

    rng = np.random.default_rng(42)
    B, H, d, Hq_tp = 1, 2048, 128, 8
    H_wo   = H
    n_tiles = H // d
    SC = 0.02

    def r(*shape):
        return (rng.standard_normal(shape).astype(np.float32) * SC)

    Wq_np  = r(Hq_tp * d, H)
    Wk_np  = r(d, H)
    Wv_np  = r(d, H)
    Wo_np  = r(Hq_tp * d, H_wo)
    qnw_np = r(d)
    knw_np = r(d)
    gpan_np = r(H)

    def t(a): return torch.from_numpy(a.astype(np.float32)).bfloat16()

    # Tile-transpose weights once; send to device
    Wq_pt = tile_transpose(t(Wq_np), Hq_tp, d, n_tiles).to(device)
    Wk_pt = tile_transpose(t(Wk_np), 1,     d, n_tiles).to(device)
    Wv_pt = tile_transpose(t(Wv_np), 1,     d, n_tiles).to(device)
    Wo_dev   = t(Wo_np).to(device)
    qnw_dev  = t(qnw_np).to(device)
    knw_dev  = t(knw_np).to(device)
    gpan_dev = t(gpan_np).to(device)

    weights_cpu    = (Wq_np, Wk_np, Wv_np, Wo_np, qnw_np, knw_np, gpan_np)
    weights_device = (Wq_pt, Wk_pt, Wv_pt, Wo_dev, qnw_dev, knw_dev, gpan_dev)

    positions = [0, 1, 63, 64, 127, 128, 129, 255, 320]

    print("\n" + "=" * 65)
    print("  v14d full hoisted mode — trn3, LNC=2, all 9 kwargs supplied")
    print("=" * 65)

    results = []
    for i, p in enumerate(positions):
        print(f"\n=== position_ids={p} ===", flush=True)
        res = run_one(
            pos_val=p,
            weights_cpu=weights_cpu,
            weights_device=weights_device,
            hidden_rng_seed=100 + i,
            device=device,
        )
        results.append(res)
        ok_parts = [
            res["pass_attn"], res["pass_nk"], res["pass_nv"],
            res["pass_Kfull"], res["pass_Vfull"],
            res["k_other_untouched_max"] < 1e-6,
            res["v_other_untouched_max"] < 1e-6,
        ]
        verdict = "PASS" if all(ok_parts) else "FAIL"
        print(f"  attn_max={res['attn_max']:.2e}  nk_max={res['nk_max']:.2e}  "
              f"nv_max={res['nv_max']:.2e}  k_other={res['k_other_untouched_max']:.2e}  "
              f"v_other={res['v_other_untouched_max']:.2e}  {verdict}", flush=True)

    print(f"\n\n{'='*65}")
    print("  SUMMARY [full hoisted, trn3, LNC=2]")
    print("=" * 65)
    print(f"{'pos':>5} {'attn':>10} {'nk':>10} {'nv':>10} {'k_other':>10} {'v_other':>10}  verdict")

    fail = 0
    for r_ in results:
        k_ok = r_["k_other_untouched_max"] < 1e-6
        v_ok = r_["v_other_untouched_max"] < 1e-6
        ok   = (r_["pass_attn"] and r_["pass_nk"] and r_["pass_nv"]
                and r_["pass_Kfull"] and r_["pass_Vfull"] and k_ok and v_ok)
        if not ok:
            fail += 1
        verdict = "PASS" if ok else "FAIL"
        print(f"{r_['pos']:>5} {r_['attn_max']:>10.3e} {r_['nk_max']:>10.3e} "
              f"{r_['nv_max']:>10.3e} {r_['k_other_untouched_max']:>10.3e} "
              f"{r_['v_other_untouched_max']:>10.3e}  {verdict}")

    n = len(results)
    print(f"\n{n - fail}/{n} positions PASS")
    print(f"OVERALL: {'PASS' if fail == 0 else 'FAIL'}")

    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
