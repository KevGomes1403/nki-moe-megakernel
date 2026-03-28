# Attention TKG Kernels

Fused NKI kernels for the Qwen3 GQA decode step (TP=4 on Trainium 2).

Each kernel implements: **QKV projection → QK RMSNorm → RoPE → GQA decode attention**
(and optionally the output projection) in a single NKI kernel.

## Kernel Variants

Kernels are versioned in `agents/`. The harness in `test_correctness.py` auto-detects
which variant a kernel implements based on its entry-point signature:

| Variant | Entry point | Wo | KV cache shape |
|---------|------------|-----|---------------|
| `no_wo` | `qwen3_attn_tkg_fused` | No | `[B, 4, S, d]` |
| `wo_normal` | `qwen3_attn_tkg_fused_oproj` | `[H, Hq_tp*d]` | `[B, 1, S, d]` |
| `wo_transposed` | `qwen3_attn_tkg_fused_oproj` | `[Hq_tp*d, H]` | `[B, 1, S, d]` |

## Shape Assumptions (TP=4)

- Hidden dim: `H = 2048`
- Head dim: `d = 128`
- Query heads per TP rank: `Hq_tp = 8`
- KV heads per TP rank: `Hkv_tp = 4` (no_wo) or `1` (with Wo)
- GQA ratio: `8` query heads per KV head

## Verification and Profiling

`test_correctness.py` runs numerical correctness checks against a pure PyTorch CPU
reference and can also profile kernels on-device.

```bash
# Correctness check, single S_prior
python test_correctness.py agents/v9_optimized.py

# Correctness check across multiple KV cache lengths
python test_correctness.py agents/v9_optimized.py --s-prior 128 640 2048

# Benchmark with default settings (S_prior=2048)
python test_correctness.py agents/v9_optimized.py --benchmark

# Benchmark with custom warmup/iters
python test_correctness.py agents/v9_optimized.py --benchmark --warmup 5 --iters 50

# Custom batch size
python test_correctness.py agents/v9_optimized.py --batch 4
```

Profiling output (device profile JSON) is written to `output/`. The benchmark prints
MFU, MBU, HBM bytes, and engine active times.

Correctness tolerance: `rtol=0.02, atol=0.1` against float32 reference.
