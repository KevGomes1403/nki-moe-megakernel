"""Scan presharded checkpoint shards for non-finite (NaN/Inf) weights.

The full model NaNs structurally (even a 1-token prompt), and DeltaNet is
ruled out. A single non-finite weight on disk would explain a NaN at any
length (incl. via 0*inf in the all-experts MoE path). This scans the loaded
shards tensor-by-tensor (lazily) and reports any non-finite key.

Run:  python -m models.qwen3_6_moe.tests.scan_weights_finite [shard.safetensors ...]
"""

import sys
import torch
from safetensors import safe_open

DEFAULT_SHARDS = [
    "/home/ubuntu/models/qwen36_a3b_traced/weights/tp0_sharded_checkpoint.safetensors",
    "/home/ubuntu/models/qwen36_a3b_traced/weights/tp1_sharded_checkpoint.safetensors",
]


def scan(path):
    print(f"\n=== {path} ===", flush=True)
    n_keys = 0
    n_bad = 0
    bad = []
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            n_keys += 1
            tf = t.float()
            n_nan = int(torch.isnan(tf).sum())
            n_inf = int(torch.isinf(tf).sum())
            if n_nan or n_inf:
                n_bad += 1
                amax = tf[torch.isfinite(tf)].abs().max().item() if torch.isfinite(tf).any() else float("nan")
                bad.append((key, tuple(t.shape), str(t.dtype), n_nan, n_inf, amax))
    print(f"scanned {n_keys} tensors; {n_bad} non-finite", flush=True)
    for key, shape, dt, n_nan, n_inf, amax in bad:
        print(f"  BAD {key}  shape={shape} dtype={dt} nan={n_nan} inf={n_inf} finite_absmax={amax:.3e}", flush=True)
    return n_bad


def main():
    shards = sys.argv[1:] or DEFAULT_SHARDS
    total_bad = 0
    for p in shards:
        total_bad += scan(p)
    print(f"\nTOTAL non-finite tensors across {len(shards)} shard(s): {total_bad}", flush=True)


if __name__ == "__main__":
    main()
