# GQA head_dim=256 TKG kernels

Token-generation NKI kernels for Qwen3.6's GQA attention block, which runs at
**head_dim=256** — above nkilib's 128-partition cap. The cap is a *tiling gap*, not
a hardware limit: head_dim sits on the SBUF/PSUM partition axis (nc_matmul K≤128,
stationary-free M≤128, partition≤128), so we tile it into `D_TILES = ceil(256/128) = 2`
partition tiles — PSUM-accumulate where head_dim is a contraction, two tiles where it
is an output partition. Norm and RoPE are applied in the `[token, head_dim]` layout
(head_dim on the **free** axis), so they need no tiling at all.

Per single TP shard (TP=4): hidden=2048, q_heads=4, kv_heads=1 (GQA 4:1), head_dim=256,
rope_dim=64, mRoPE sections [11,11,10], decode width T∈{1,2}. bf16 IO, fp32 accumulate, LNC=2.

## Phases (data-flow order)

| Phase | Entrypoint (`components/`) | nkilib | Notes |
|-------|---------------------------|--------|-------|
| qkv_proj  | `qkv_proj_compose`     | wrap `core/qkv/qkv_tkg`                 | NBSd output, fused input RMSNorm |
| qk_norm   | `qk_norm_compose`      | fresh (free-axis)                       | RMSNorm over head_dim=256 on q/k; standard-weight gamma |
| rope      | `rope_partial_compose` | fresh (free-axis)                       | partial 64/256 + mRoPE, rotate_half |
| attention | `gqa_attention_d256`   | **vendor+patch** `core/attention/attention_tkg` | `D_TILES` at the two matmuls + store; softmax byte-for-byte AWS |
| out_proj  | `out_proj_compose`     | wrap `core/output_projection/output_projection_tkg` | 256 as 2×128 sub-heads; deferred TP all-reduce |

`vendored/` holds the patched copy of nkilib `attention_tkg.py` (+ unmodified
`attention_tkg_utils.py`); all `# --- D256 PATCH` edits are head_dim layout only.
`components/attention_fresh_ref.py` is a from-scratch reference core kept for cross-checking.

## Contracts (megakernel handoff)

- qkv_proj → qk_norm: NBSd `[T, N=6, D=256]` SBUF, heads `[q0..q3, k0, v0]` (zero-copy reshape of `[T, I]`).
- attention core: expects **Q pre-scaled by 1/sqrt(d)**, q/k pre-normed + pre-RoPE'd (`fuse_rope=False`),
  caller-supplied causal mask; layout `[128, D_TILES, …]`. `curr_sprior` (prior+active KV length)
  must be a multiple of 128 (AWS requirement).
- attention → out_proj: `[128, D_TILES, Tq]` (head_dim on partition) feeds `output_projection_tkg` as 8 sub-heads of 128.

## Tests

Single-TP-shard PyTorch-reference tests, fp32 hard gate `atol=1e-5, rtol=1e-2`
(bf16 reported with cosine/floor — deep-contraction bf16-vs-fp32 noise is expected):

```
cd /home/ubuntu/trainium-model-translation && source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn3pre NEURON_CC_FLAGS="--target trn3pre --lnc 2" \
python -m models.qwen3_6_moe.tests.test_gqa_<phase>_kernel
```

Pin one logical core per job (`NEURON_RT_VISIBLE_CORES=0`/`1`/`2`/`3`) to run phases in parallel.
Not yet integrated into the model.
