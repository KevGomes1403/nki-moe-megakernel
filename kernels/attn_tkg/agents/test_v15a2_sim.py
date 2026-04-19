"""
Correctness test for v15a2_h_shard_qkv.

Wq, Wk, Wv are tile-transposed before being passed to the kernel (same as v14b).
Tolerance: rtol=1e-2, atol=1e-2.

Run:
    NKI_PRECISE_FP=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3 \\
        python test_v15a2_sim.py
"""

import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NKI_PRECISE_FP"] = "1"

import math
import sys
import nki
import nki.language as nl
import nki.isa as nisa
import numpy as np
import ml_dtypes
import torch
import torch.nn.functional as F
from nkilib.core.utils.allocator import SbufManager
from v15a2_h_shard_qkv import qwen3_attn_tkg_fused_oproj_v15a2

PMAX = 128
EPS = 1e-6
INV_SQRT_D = float(1.0 / math.sqrt(128.0))


def tile_transpose_weight(W_np, n_heads, d, n_tiles):
    W_f32 = np.array(W_np, dtype=np.float32)
    W_pt_f32 = (W_f32.reshape(n_heads, d, n_tiles, d)
                      .transpose(0, 3, 2, 1)
                      .reshape(n_heads * d, n_tiles * d)
                      .copy())
    return W_pt_f32.astype(ml_dtypes.bfloat16)


@nki.jit
def v15a2_wrapper(
    hidden_states, Wq, Wk, Wv, Wo,
    q_norm_weight, k_norm_weight, gamma_pre_attn,
    K_cache, V_cache, cos, sin, position_ids, output,
):
    B = cos.shape[0]
    H = Wq.shape[1]
    H_wo = Wo.shape[1]
    num_h_tiles = H // PMAX
    num_out_cols = H_wo // PMAX

    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("wrapper")

    hidden_col = hidden_states.reshape((H, B))
    hidden_sb = sbm.alloc_stack((PMAX, num_h_tiles), nl.bfloat16, name="hidden_sb")
    nisa.dma_copy(
        dst=hidden_sb,
        src=hidden_col.ap(pattern=[[1, PMAX], [PMAX, num_h_tiles]], offset=0),
        dge_mode=nisa.dge_mode.hwdge,
    )

    out_sb = qwen3_attn_tkg_fused_oproj_v15a2(
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


def pytorch_attn_ref(
    hidden_states, Wq, Wk, Wv, Wo,
    q_norm_weight, k_norm_weight, gamma_pre_attn,
    K_cache, V_cache, cos, sin, position_ids,
):
    B = hidden_states.shape[0]
    d = q_norm_weight.shape[0]
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

    v = x @ Wv.float().t()
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
    S = K_ctx.shape[1]
    mask = torch.zeros(B, S)
    for b in range(B):
        mask[b, int(position_ids[b, 0]) + 1:] = -1e9

    heads = []
    for h in range(Hq_tp):
        sc = (q_scaled[:, h:h + 1, :] @ K_ctx.transpose(-2, -1)).squeeze(1) + mask
        w = F.softmax(sc, dim=-1)
        heads.append((w.unsqueeze(1) @ V_ctx).squeeze(1))

    attn_out = (torch.stack(heads, dim=1).reshape(B, Hq_tp * d) @ Wo.float()).bfloat16()
    return attn_out, new_k, new_v, K_cache_out, V_cache_out


def run_one(pos_val, weights, hidden_rng_seed):
    B, H, d, Hq_tp, S = 1, 2048, 128, 8, 640
    H_wo = H
    n_tiles = H // d

    Wq_np, Wk_np, Wv_np, Wo_np, qnw_np, knw_np, gpan_np = weights

    Wq_pt = tile_transpose_weight(Wq_np, n_heads=Hq_tp, d=d, n_tiles=n_tiles)
    Wk_pt = tile_transpose_weight(Wk_np, n_heads=1,     d=d, n_tiles=n_tiles)
    Wv_pt = tile_transpose_weight(Wv_np, n_heads=1,     d=d, n_tiles=n_tiles)

    rng_h = np.random.default_rng(hidden_rng_seed)
    def r_bf16(*shape):
        return rng_h.standard_normal(shape).astype(np.float32).astype(ml_dtypes.bfloat16)

    hidden_np = r_bf16(B, 1, H)
    K_cache_np = r_bf16(B, 1, S, d)
    V_cache_np = r_bf16(B, 1, S, d)
    pos_np = np.array([[pos_val]], dtype=np.int32)

    cos_full_rng = np.random.default_rng(1234)
    cos_full_np = cos_full_rng.standard_normal((S, d)).astype(np.float32).astype(ml_dtypes.bfloat16)
    sin_full_np = cos_full_rng.standard_normal((S, d)).astype(np.float32).astype(ml_dtypes.bfloat16)
    cos_np = cos_full_np[pos_np[:, 0]]
    sin_np = sin_full_np[pos_np[:, 0]]

    def to_torch(arr):
        return torch.from_numpy(np.array(arr, dtype=np.float32)).bfloat16()

    hidden_t = to_torch(hidden_np)
    Wq_t = to_torch(Wq_np); Wk_t = to_torch(Wk_np)
    Wv_t = to_torch(Wv_np); Wo_t = to_torch(Wo_np)
    qnw_t = to_torch(qnw_np); knw_t = to_torch(knw_np); gpan_t = to_torch(gpan_np)
    K_cache_t = to_torch(K_cache_np); V_cache_t = to_torch(V_cache_np)
    cos_t = to_torch(cos_np); sin_t = to_torch(sin_np)
    pos_t = torch.tensor([[pos_val]], dtype=torch.int32)

    ref_out, ref_nk, ref_nv, ref_K_full, ref_V_full = pytorch_attn_ref(
        hidden_t, Wq_t, Wk_t, Wv_t, Wo_t, qnw_t, knw_t, gpan_t,
        K_cache_t.clone(), V_cache_t.clone(), cos_t, sin_t, pos_t,
    )

    K_cache_sim = K_cache_np.copy()
    V_cache_sim = V_cache_np.copy()
    output_np = np.zeros((B, 1, H_wo), dtype=ml_dtypes.bfloat16)

    result_np, K_out_np, V_out_np = nki.simulate(v15a2_wrapper)(
        hidden_np, Wq_pt, Wk_pt, Wv_pt, Wo_np,
        qnw_np, knw_np, gpan_np,
        K_cache_sim, V_cache_sim, cos_np, sin_np, pos_np, output_np,
    )

    attn_nki = torch.from_numpy(np.array(result_np, dtype=np.float32)).reshape(B, -1)
    K_full_nki = torch.from_numpy(np.array(K_out_np, dtype=np.float32))
    V_full_nki = torch.from_numpy(np.array(V_out_np, dtype=np.float32))

    nk_nki = K_full_nki[:, 0, pos_val, :]
    nv_nki = V_full_nki[:, 0, pos_val, :]

    K_mask = torch.ones(S, dtype=torch.bool); K_mask[pos_val] = False
    V_mask = torch.ones(S, dtype=torch.bool); V_mask[pos_val] = False
    k_other_diff = (K_full_nki[0, 0, K_mask].float() - K_cache_t[0, 0, K_mask].float()).abs().max().item()
    v_other_diff = (V_full_nki[0, 0, V_mask].float() - V_cache_t[0, 0, V_mask].float()).abs().max().item()

    attn_max = (attn_nki.float() - ref_out.float()).abs().max().item()
    nk_max = (nk_nki.float() - ref_nk.float()).abs().max().item()
    nv_max = (nv_nki.float() - ref_nv.float()).abs().max().item()
    Kfull_max = (K_full_nki.float() - ref_K_full.float()).abs().max().item()
    Vfull_max = (V_full_nki.float() - ref_V_full.float()).abs().max().item()

    def close(a, b, rtol=1e-2, atol=1e-2):
        return bool(torch.all(torch.abs(a.float() - b.float()) <= atol + rtol * torch.abs(b.float())).item())

    pass_attn  = close(attn_nki.reshape(B, -1), ref_out)
    pass_nk    = close(nk_nki, ref_nk)
    pass_nv    = close(nv_nki, ref_nv)
    pass_Kfull = close(K_full_nki, ref_K_full)
    pass_Vfull = close(V_full_nki, ref_V_full)

    return {
        "pos": pos_val,
        "attn_max": attn_max, "nk_max": nk_max, "nv_max": nv_max,
        "k_other_untouched_max": k_other_diff,
        "v_other_untouched_max": v_other_diff,
        "K_full_max_vs_ref": Kfull_max,
        "V_full_max_vs_ref": Vfull_max,
        "pass_attn": pass_attn, "pass_nk": pass_nk, "pass_nv": pass_nv,
        "pass_Kfull": pass_Kfull, "pass_Vfull": pass_Vfull,
    }


def main():
    import logging
    logging.getLogger("SBM").setLevel(logging.WARNING)

    rng = np.random.default_rng(42)
    B, H, d, Hq_tp = 1, 2048, 128, 8
    H_wo = H

    def r_bf16(*shape):
        return (rng.standard_normal(shape).astype(np.float32) * 0.02).astype(ml_dtypes.bfloat16)

    Wq = r_bf16(Hq_tp * d, H)
    Wk = r_bf16(d, H)
    Wv = r_bf16(d, H)
    Wo = r_bf16(Hq_tp * d, H_wo)
    qnw = r_bf16(d)
    knw = r_bf16(d)
    gpan = r_bf16(H)
    weights = (Wq, Wk, Wv, Wo, qnw, knw, gpan)

    positions = [0, 1, 63, 64, 127, 128, 129, 255, 320]

    results = []
    for i, p in enumerate(positions):
        print(f"\n=== position_ids={p} ===", flush=True)
        res = run_one(pos_val=p, weights=weights, hidden_rng_seed=100 + i)
        results.append(res)
        ok_parts = [res["pass_attn"], res["pass_nk"], res["pass_nv"],
                    res["pass_Kfull"], res["pass_Vfull"],
                    res["k_other_untouched_max"] < 1e-6,
                    res["v_other_untouched_max"] < 1e-6]
        verdict = "PASS" if all(ok_parts) else "FAIL"
        print(f"  attn_max={res['attn_max']:.2e}  nk_max={res['nk_max']:.2e}  "
              f"nv_max={res['nv_max']:.2e}  k_other={res['k_other_untouched_max']:.2e}  "
              f"v_other={res['v_other_untouched_max']:.2e}  {verdict}", flush=True)

    print("\n\n======== SUMMARY ========")
    hdr = (f"{'pos':>5} {'attn':>10} {'nk':>10} {'nv':>10} "
           f"{'k_other':>10} {'v_other':>10} {'K_full':>10} {'V_full':>10}  verdict")
    print(hdr)
    fail = 0
    atol = 1e-2
    for r_ in results:
        k_other_ok = r_["k_other_untouched_max"] < 1e-6
        v_other_ok = r_["v_other_untouched_max"] < 1e-6
        ok = (r_["pass_attn"] and r_["pass_nk"] and r_["pass_nv"]
              and r_["pass_Kfull"] and r_["pass_Vfull"]
              and k_other_ok and v_other_ok)
        verdict = "PASS" if ok else "FAIL"
        if not ok:
            fail += 1
        print(f"{r_['pos']:>5} {r_['attn_max']:>10.3e} {r_['nk_max']:>10.3e} {r_['nv_max']:>10.3e} "
              f"{r_['k_other_untouched_max']:>10.3e} {r_['v_other_untouched_max']:>10.3e} "
              f"{r_['K_full_max_vs_ref']:>10.3e} {r_['V_full_max_vs_ref']:>10.3e}  {verdict}")

    print(f"\n{len(results) - fail}/{len(results)} positions PASS  (atol={atol})")
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
