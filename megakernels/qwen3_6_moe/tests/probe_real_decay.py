"""CPU probe: does the REAL layer-0 DeltaNet decay overflow the kernel's exp(-gc)?

The DeltaNet-only model NaNs even for a 1-token prompt. The fused kernel forms
exp(-cumsum(g)); this overflows fp32 when cumsum(g) < -88.7. A single token can
trigger it if its gating g[0] is strongly negative (g = -exp(A_log)*softplus(a+dt_bias),
large exp(A_log) => huge per-token decay). This computes layer-0 g/gc from the REAL
checkpoint on CPU and reports the cliff crossing per prompt -- no device needed.
"""

import glob
import json
import math
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

CKPT = "/home/ubuntu/models/Qwen3.6-35B-A3B"
PROMPTS = ["Hi", "The capital of France is"]
FP32_EXP_CLIFF = 88.72  # exp(x) overflows float32 for x > ~88.72


def load_keys(keys):
    idx = json.load(open(glob.glob(os.path.join(CKPT, "model*.index.json"))[0]))["weight_map"]
    out = {}
    by_shard = {}
    for k in keys:
        by_shard.setdefault(idx[k], []).append(k)
    for shard, ks in by_shard.items():
        with safe_open(os.path.join(CKPT, shard), framework="pt", device="cpu") as f:
            for k in ks:
                out[k] = f.get_tensor(k).float()
    return out


def main():
    cfg = json.load(open(os.path.join(CKPT, "config.json")))
    tc = cfg.get("text_config", cfg)
    eps = tc.get("rms_norm_eps", 1e-6)

    P = "model.language_model."
    w = load_keys([
        P + "embed_tokens.weight",
        P + "layers.0.input_layernorm.weight",
        P + "layers.0.linear_attn.in_proj_a.weight",
        P + "layers.0.linear_attn.A_log",
        P + "layers.0.linear_attn.dt_bias",
    ])
    embed = w[P + "embed_tokens.weight"]
    ln_w = w[P + "layers.0.input_layernorm.weight"]
    in_proj_a = w[P + "layers.0.linear_attn.in_proj_a.weight"]   # (num_v_heads, hidden)
    A_log = w[P + "layers.0.linear_attn.A_log"]                  # (num_v_heads,)
    dt_bias = w[P + "layers.0.linear_attn.dt_bias"]              # (num_v_heads,)
    print(f"shapes: in_proj_a={tuple(in_proj_a.shape)} A_log={tuple(A_log.shape)} "
          f"dt_bias={tuple(dt_bias.shape)}")
    print(f"A_log: min={A_log.min():.3f} max={A_log.max():.3f}  "
          f"exp(A_log): min={A_log.exp().min():.3f} max={A_log.exp().max():.3f}")

    tok = AutoTokenizer.from_pretrained(CKPT)
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids[0]
        x = embed[ids]                                           # (S, H)
        # HF Qwen3_5MoeRMSNorm: x/rms(x) * (1 + weight)
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        normed = normed * (1.0 + ln_w)
        a = normed @ in_proj_a.T                                 # (S, num_v_heads)
        g = -A_log.exp() * F.softplus(a + dt_bias)               # (S, num_v_heads)
        gc = torch.cumsum(g, dim=0)                              # (S, num_v_heads)
        neg_gc_max = (-gc).max().item()
        print(f"\nprompt={prompt!r}  tokens={ids.tolist()}")
        print(f"  per-token g:  min={g.min().item():.2f} max={g.max().item():.2f}")
        print(f"  cumsum gc:    min={gc.min().item():.2f} (most negative)")
        print(f"  max(-gc) = {neg_gc_max:.2f}  -> exp overflow fp32 (cliff {FP32_EXP_CLIFF}): "
              f"{'YES *** KERNEL exp(-gc) -> inf -> NaN ***' if neg_gc_max > FP32_EXP_CLIFF else 'no'}")
        # how many (token,head) entries cross the cliff
        crossed = int(((-gc) > FP32_EXP_CLIFF).sum())
        print(f"  (token,head) entries over cliff: {crossed} / {gc.numel()}")


if __name__ == "__main__":
    main()
