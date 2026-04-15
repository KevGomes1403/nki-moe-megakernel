import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"

import nki, nki.language as nl, nki.isa as nisa
import torch, numpy as np
import torch.nn.functional as F
from nkilib.core.utils.allocator import SbufManager
from v13bc_kv_norm import qwen3_attn_tkg_fused_oproj_v13bc_kv_norm

PMAX = 128

@nki.jit
def v13bc_kv_norm_wrapper(
    hidden_states,   # [B, 1, H]      bf16 HBM
    Wq, Wk, Wv, Wo,
    q_norm_weight, k_norm_weight, gamma_pre_attn,
    K_cache, V_cache,
    cos, sin, position_ids,
    output,          # [B, 1, H_wo]   bf16 HBM pre-allocated
):
    B      = cos.shape[0]
    H      = Wq.shape[1]
    H_wo   = Wo.shape[1]
    num_h_tiles  = H    // PMAX
    num_out_cols = H_wo // PMAX

    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope("wrapper")

    # Load hidden HBM → SBUF column-major [PMAX, num_h_tiles]
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

    # Store out_sb [PMAX, num_out_cols] column-major → output [B, 1, H_wo]
    # out_sb[p, j] = output[0, 0, j*PMAX + p]
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


def pytorch_attn_ref_bf16(
    hidden_states, Wq, Wk, Wv, Wo,
    q_norm_weight, k_norm_weight, gamma_pre_attn,
    K_cache, V_cache, cos, sin, position_ids,
):
    EPS = 1e-6
    B = hidden_states.shape[0]
    d = q_norm_weight.shape[0]
    H = hidden_states.shape[1]
    Hq_tp = Wq.shape[0] // d

    x = hidden_states.float()
    cos_f, sin_f = cos.float(), sin.float()

    # input_layernorm
    x = x * (x.pow(2).mean(-1, keepdim=True) + EPS).rsqrt() * gamma_pre_attn.float()

    # K proj + RMSNorm + RoPE
    k = x @ Wk.float().t()
    k = k * (k.pow(2).mean(-1, keepdim=True) + EPS).rsqrt() * k_norm_weight.float()
    k_rot = torch.cat([-k[:, d//2:], k[:, :d//2]], dim=-1)
    k = k * cos_f + k_rot * sin_f
    new_k = k.bfloat16()

    # V proj
    v = x @ Wv.float().t()
    new_v = v.bfloat16()

    # cache scatter
    K_cache_out = K_cache.clone()
    V_cache_out = V_cache.clone()
    for b in range(B):
        pos = int(position_ids[b, 0])
        K_cache_out[b, 0, pos, :] = new_k[b]
        V_cache_out[b, 0, pos, :] = new_v[b]

    # Q proj + RMSNorm + RoPE
    q = (x @ Wq.float().t()).reshape(B, Hq_tp, d)
    q = q * (q.pow(2).mean(-1, keepdim=True) + EPS).rsqrt() * q_norm_weight.float()
    cos_q = cos_f.unsqueeze(1).expand(-1, Hq_tp, -1)
    sin_q = sin_f.unsqueeze(1).expand(-1, Hq_tp, -1)
    q_rot = torch.cat([-q[..., d//2:], q[..., :d//2]], dim=-1)
    q = (q * cos_q + q_rot * sin_q) / (d ** 0.5)

    # GQA flash-decode
    K_ctx = K_cache_out[:, 0, :, :].float()
    V_ctx = V_cache_out[:, 0, :, :].float()
    S = K_ctx.shape[1]
    mask = torch.zeros(B, S)
    for b in range(B):
        mask[b, int(position_ids[b, 0]) + 1:] = -1e9

    heads = []
    for h in range(Hq_tp):
        sc = (q[:, h:h+1, :] @ K_ctx.transpose(-2, -1)).squeeze(1) + mask
        w  = F.softmax(sc, dim=-1)
        heads.append((w.unsqueeze(1) @ V_ctx).squeeze(1))

    attn_out = (torch.stack(heads, dim=1).reshape(B, Hq_tp*d) @ Wo.float()).bfloat16()
    return attn_out, new_k, new_v, K_cache_out, V_cache_out


def run_correctness():
    import torch_xla.core.xla_model as xm

    rng = np.random.default_rng(42)
    B, H, d, Hq_tp, S = 1, 2048, 128, 8, 640
    H_wo = H

    def r(*shape):
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).bfloat16()

    hidden  = r(B, H)
    SC = 0.02
    Wq = r(Hq_tp*d, H) * SC; Wk = r(d, H) * SC; Wv = r(d, H) * SC; Wo = r(Hq_tp*d, H_wo) * SC
    qnw = r(d); knw = r(d); gpan = r(H)
    K_cache = r(B, 1, S, d); V_cache = r(B, 1, S, d)
    pos = torch.randint(0, S-1, (B, 1), dtype=torch.int32)
    cos_full = r(S, d); sin_full = r(S, d)
    cos = cos_full[pos[:, 0]]; sin = sin_full[pos[:, 0]]

    ref_out, ref_nk, ref_nv, *_ = pytorch_attn_ref_bf16(
        hidden, Wq, Wk, Wv, Wo, qnw, knw, gpan,
        K_cache.clone(), V_cache.clone(), cos, sin, pos,
    )

    output = torch.zeros(B, 1, H_wo, dtype=torch.bfloat16).to("xla")
    K_kv = K_cache.clone().to("xla")
    V_kv = V_cache.clone().to("xla")

    result, K_kv_out, V_kv_out = v13bc_kv_norm_wrapper[2](
        hidden.reshape(B, 1, H).to("xla"),
        Wq.to("xla"), Wk.to("xla"), Wv.to("xla"), Wo.to("xla"),
        qnw.to("xla"), knw.to("xla"), gpan.to("xla"),
        K_kv, V_kv,
        cos.to("xla"), sin.to("xla"), pos.to("xla"),
        output,
    )
    xm.mark_step()

    attn_nki = result.cpu()
    nk_nki   = K_kv_out.cpu()[:, 0, pos[0, 0], :]
    nv_nki   = V_kv_out.cpu()[:, 0, pos[0, 0], :]

    print(f"DIAG attn_max={abs(attn_nki.reshape(B,-1).float()-ref_out.float()).max():.3e} "
          f"nk_max={abs(nk_nki.float()-ref_nk.float()).max():.3e} "
          f"nv_max={abs(nv_nki.float()-ref_nv.float()).max():.3e}")
    np.testing.assert_allclose(nk_nki.float().numpy(),  ref_nk.float().numpy(),
                               rtol=1e-2, atol=1e-2, err_msg="new_k")
    np.testing.assert_allclose(nv_nki.float().numpy(),  ref_nv.float().numpy(),
                               rtol=1e-2, atol=1e-2, err_msg="new_v")
    np.testing.assert_allclose(attn_nki.reshape(B, -1).float().numpy(), ref_out.float().numpy(),
                               rtol=1e-2, atol=1e-2, err_msg="attn_out")
    print(f"PASS  attn={abs(attn_nki-ref_out).max():.2e}  "
          f"new_k={abs(nk_nki-ref_nk).max():.2e}  "
          f"new_v={abs(nv_nki-ref_nv).max():.2e}")


if __name__ == "__main__":
    run_correctness()
