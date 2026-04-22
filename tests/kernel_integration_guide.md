# Kernel Integration Guide

A practical reference for adding a new NKI kernel to this repo, wiring it into
a Qwen model, registering it in `main.py`, and debugging hidden-state outputs.

---

## 1. The Two Qwen Models

### `qwen.py` — Stock NxDI baseline

This is the reference model. It runs every decoder layer independently using
NxDI's built-in attention and MoE modules. Use it as:

- The **accuracy target** every custom kernel must match.
- A stand-alone sanity-check when your kernel output looks wrong.
- The **reference module** inside unit tests (see Section 4).

Key class: `NeuronQwen3MoeForCausalLM` (re-exported from NxDI). The alias
`NeuronQwen3MoeForCausalLM` in `qwen.py` is what `main.py` imports.

Key config: `Qwen3MoeInferenceConfig` — wraps `MoENeuronConfig` and adds
model-shape attributes. Most tests construct this directly (see Section 4).

### `qwen_fused_transformer_multilayer.py` — 48-layer megakernel model

This is the custom model. On the **token-generation (TKG) path**, layer 0
invokes a single NKI kernel (`transformer_qwen3_moe_tkg_multilayer_jit`) that
runs all 48 decoder layers in one shot. The residual stays SBUF-resident across
layer boundaries; K/V caches are updated in-place.

Key class: `NeuronQwen3MoeForCausalLMV3` (also aliased to
`NeuronQwen3MoeForCausalLM` at the bottom of the file so `main.py` works
identically for both models).

On the **context-encoding (CTE) path**, this model falls through to the same
per-layer PyTorch forward as `qwen.py`.

---

## 2. Where Kernels Live

```
kernels/
  transformer/
    transformer_qwen_multilayer.py   ← the multi-layer fused TKG kernel
    transformer_qwen.py              ← single-layer version (earlier iteration)
    ...
```

Each kernel file exposes a JIT-compiled entry point. For the multilayer kernel:

```python
# kernels/transformer/transformer_qwen_multilayer.py

def get_multilayer_kernel_jit(num_layers: int):
    """Returns (jit_fn_lnc1, jit_fn_lnc2, jit_fn_lnc2_again).
    Index with [LNC] to get the right variant, e.g. jit_fn[2] for LNC=2."""
    ...

def _build_multilayer_kernel(num_layers: int):
    """Lower-level builder used by tests; same JIT fn but easier to call
    without the production model wrapper overhead."""
    ...
```

---

## 3. Integrating a Kernel into a Model

The cleanest pattern: create a new `qwen_<name>.py` file that copies the model
skeleton from `qwen.py` and overrides only what the kernel touches.

**Step-by-step:**

### 3a. Write the kernel function

Put the kernel in `kernels/transformer/your_kernel.py`. Decorate with
`@nki.jit` or export via a builder function like `get_multilayer_kernel_jit`.

### 3b. Create `qwen_<name>.py`

Copy the imports and class structure from `qwen_fused_transformer_multilayer.py`.
Override `NeuronQwen3MoeDecoderLayerV3.forward()` — the TKG branch (guarded by
`if is_tkg`) is where layer 0 fires the kernel.

The critical call site in `qwen_fused_transformer_multilayer.py:462`:

```python
_kernel_fn = get_multilayer_kernel_jit(L)[2]   # [2] = LNC=2
kernel_out = _kernel_fn(
    hidden_states,
    *Wq_list, *Wk_list, *Wv_list, *Wo_list,
    *qn_list, *kn_list, *gpre_list, *gpost_list,
    *router_list, *gate_up_list, *down_list,
    *K_caches, *V_caches,
    cos_at_pos, sin_at_pos, position_ids.to(torch.int32),
    replica_groups=self._replica_groups,
)
```

The return convention (production kernel):
- `kernel_out[0]`       → `Y` (the output hidden states)
- `kernel_out[1:1+L]`  → `K_out[0..L-1]` (in-place-updated K caches)
- `kernel_out[1+L:]`   → `V_out[0..L-1]` (in-place-updated V caches)

Pass-through layers (idx 1..L-1) just forward the K/V handles that layer 0
stashed on `self._parent_model._tkg_kv_out` (see `:502`).

### 3c. Wire the NeuronConfig flags

`qwen_fused_transformer_multilayer.py` sets these in its `NeuronConfig.__init__`:

```python
kwargs["attn_tkg_nki_kernel_enabled"]           = True
kwargs["attn_block_tkg_nki_kernel_cache_update"] = True
```

- `attn_tkg_nki_kernel_enabled` — tells NxDI the attention kernel handles K/V.
- `attn_block_tkg_nki_kernel_cache_update` — tells NxDI to **skip** its own
  Python K/V scatter and trailing `kv_mgr.update_cache`, because the kernel
  writes the caches in-place.

If your kernel does not update K/V in-place, leave these flags at their defaults.

---

## 4. Registering the Model in `main.py`

### 4a. Add an alias to `resolve_qwen_module_name()`

Find the `alias_map` dict in `main.py` (~line 741) and add your module:

```python
alias_map = {
    ...
    "qwen_my_kernel":         "qwen_my_kernel",
    "qwen_mk":                "qwen_my_kernel",   # short alias
    ...
}
```

The value is the Python module name (file name without `.py`) that `importlib`
will import. The file must live at the repo root.

### 4b. Add the `--qwen` help text

Find the `add_argument("--qwen", ...)` call near line 93 and add a line to the
`help=` string. This keeps the `--help` output accurate.

### 4c. Run it

```bash
# Compile and generate (first run)
python main.py \
  --mode generate \
  --qwen qwen_my_kernel \
  --model-path ~/models/Qwen3-30B-A3B/ \
  --compiled-model-path /tmp/my_kernel_compiled \
  --seq-len 640 \
  --tp-degree 4

# Skip recompile on subsequent runs
python main.py \
  --mode generate \
  --qwen qwen_my_kernel \
  --skip-compile true \
  --compiled-model-path /tmp/my_kernel_compiled \
  ...
```

---

## 5. Unit Tests — Comparing Kernel vs Reference

The standard pattern (used by `tests/test_megakernel_L1_vs_qwen.py`):

```
build_module(RefModule, ...)    ← stock NxDI decoder layer
build_module(KernelModule, ...) ← your kernel wrapped in an nn.Module
run both with identical inputs and identical weight seeds → compare outputs
```

### 5a. Constructing the config

```python
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from qwen import Qwen3MoeInferenceConfig

MODEL_PATH = "/home/ubuntu/models/Qwen3-30B-A3B/"

neuron_config = MoENeuronConfig(
    tp_degree=4,
    batch_size=1,
    max_context_length=640,
    seq_len=640,
    enable_bucketing=False,
    flash_decoding_enabled=False,
    logical_nc_config=2,
    torch_dtype=torch.bfloat16,
    fused_qkv=False,
    # Disable NxDI's built-in TKG kernels so the reference is pure PyTorch.
    qkv_kernel_enabled=False,
    attn_kernel_enabled=False,
    mlp_kernel_enabled=False,
    attn_tkg_nki_kernel_enabled=False,
    attn_block_tkg_nki_kernel_enabled=False,
)
cfg = Qwen3MoeInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(MODEL_PATH),
)
cfg.num_hidden_layers = 1   # test with one layer to keep compile fast
```

### 5b. Writing a RefModule

```python
class RefModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        torch.manual_seed(0)   # same seed as KernelModule → identical weights
        self.decoder_layer = NeuronQwen3MoeDecoderLayer(config, layer_idx=0)
        self.register_buffer("K_cache", torch.zeros(1, 1, S_CTX, config.head_dim, dtype=torch.bfloat16))
        self.register_buffer("V_cache", torch.zeros(1, 1, S_CTX, config.head_dim, dtype=torch.bfloat16))

    def forward(self, hidden_in, position_ids, attention_mask):
        out, kv, _, _, _ = self.decoder_layer(
            hidden_states=hidden_in,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=(self.K_cache, self.V_cache),
        )
        return out, kv[0], kv[1]
```

### 5c. Compiling and running

```python
import tempfile
from neuronx_distributed_inference.utils.testing import build_module

COMPILER_ARGS = (
    "--enable-saturate-infinity "
    "--enable-mixed-precision-accumulation "
    "--model-type transformer "
    "--lnc=2 "
    "-O1 "
    "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-hlo2tensorizer-options='--verify-hlo=true' "
    "--internal-backend-options=--enable-verifier=false"
)

example_inputs = [(
    torch.zeros(1, 1, 2048, dtype=torch.bfloat16),   # hidden_states
    torch.zeros(1, 1, dtype=torch.int32),             # position_ids
    torch.zeros(1, 1, 1, 640, dtype=torch.bfloat16), # attention_mask
)]

with tempfile.TemporaryDirectory() as workdir:
    ref_traced = build_module(
        module_cls=RefModule,
        example_inputs=example_inputs,
        module_init_kwargs={"config": cfg},
        tp_degree=4,
        logical_nc_config=2,
        compiler_args=COMPILER_ARGS,
        compiler_workdir=f"{workdir}/ref",
        checkpoint_path=f"{workdir}/ref_ckpt.pt",
    )
    kern_traced = build_module(
        module_cls=KernelModule,
        example_inputs=example_inputs,
        module_init_kwargs={"config": cfg},
        tp_degree=4,
        logical_nc_config=2,
        compiler_args=COMPILER_ARGS,
        compiler_workdir=f"{workdir}/kern",
        checkpoint_path=f"{workdir}/kern_ckpt.pt",
    )
    # Run with the same input
    hidden = (torch.randn(1, 1, 2048) * 0.1).bfloat16()
    pos    = torch.tensor([[42]], dtype=torch.int32)
    mask   = torch.zeros(1, 1, 1, 640, dtype=torch.bfloat16)
    mask[:, :, :, 43:] = float("-inf")

    ref_out, K_ref, V_ref   = ref_traced(hidden, pos, mask)
    kern_out, K_kern, V_kern = kern_traced(hidden, pos, mask)

    print(torch.allclose(ref_out.float(), kern_out.float(), atol=1e-2, rtol=1e-2))
```

### 5d. Accuracy tolerances

| Comparison | atol | rtol | Notes |
|---|---|---|---|
| Kernel vs fp32-promoted ref | 1e-3 | 1e-2 | bf16 quantization noise |
| Kernel vs bf16 ref | 1e-5 | 1e-2 | element-wise near-exact |
| E2E logit validation | 1e-5 | 0.05 | top-None rank; looser for top-5/50/1000 |

---

## 6. Debug Dumps — Reading `.pt` Files

### 6a. What gets captured

When you compile with `--debug-dump`, four tensors from layer 0 are written to
a `.pt` file after each of the first two forward passes:

| Name | Shape | When |
|---|---|---|
| `attn_out` | `[B, 1, H]` | output of self-attention (pre-residual-add) |
| `kv_k` | `[B, nkv_heads, S_max, head_dim]` | full K cache after the step |
| `kv_v` | `[B, nkv_heads, S_max, head_dim]` | full V cache after the step |
| `final_hidden_states` | `[B, 1, H]` | post-MoE hidden state (output of the layer) |

Call 1 → `debug_tensors.pt` (CTE pass)
Call 2 → `debug_dump_tkg.pt` (first TKG step)

### 6b. How to compile with debug dump

```bash
python main.py \
  --mode generate \
  --qwen qwen_fused_transformer_multilayer \
  --debug-dump \
  --debug-dump-path my_debug.pt \
  --skip-compile false \
  --compiled-model-path /tmp/debug_compiled \
  --model-path ~/models/Qwen3-30B-A3B/ \
  --seq-len 640
```

> **Important:** `--debug-dump` requires a fresh compile. If you pass
> `--skip-compile true` with a model that was NOT compiled with debug-dump,
> the hook silently receives no tensors.

### 6c. Reading the dump

```python
import torch

cte = torch.load("debug_tensors.pt")
tkg = torch.load("debug_dump_tkg.pt")

# Print shapes and statistics
for name, t in cte.items():
    print(f"CTE {name}: shape={tuple(t.shape)} "
          f"max={t.float().abs().max():.4f} "
          f"mean={t.float().abs().mean():.4f}")

for name, t in tkg.items():
    print(f"TKG {name}: shape={tuple(t.shape)} "
          f"max={t.float().abs().max():.4f}")
```

### 6d. Comparing baseline vs kernel

Compile both models with `--debug-dump` and compare the dumps:

```python
base = torch.load("debug_baseline.pt")
kern = torch.load("debug_kernel.pt")

for name in base:
    b = base[name].float()
    k = kern[name].float()
    diff = (b - k).abs()
    print(f"{name}: max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}  "
          f"allclose={torch.allclose(b, k, atol=1e-2, rtol=1e-2)}")
```

A useful intermediate check when debugging the TKG path:

```python
# Does attn_out match? If yes, the attention kernel is correct.
# Does kv_k match? If yes, K/V cache scatter is correct.
# Does final_hidden_states match? If yes, MoE is correct.
```

Work backwards from `final_hidden_states` → `attn_out` → `kv_k/kv_v` to
isolate which sub-block diverged.

---

## 7. End-to-End Validation

Once `--debug-dump` output looks reasonable, run full logit validation against
the compiled baseline:

```bash
python main.py \
  --mode validate \
  --qwen qwen_fused_transformer_multilayer \
  --skip-compile true \
  --compiled-model-path /tmp/my_kernel_compiled \
  --baseline-compiled-model-path ~/models/Qwen-baseline-compiled/ \
  --model-path ~/models/Qwen3-30B-A3B/ \
  --seq-len 640 \
  --divergence-difference-tol 0.001
```

`validate` mode runs `logit_validation` which compares the top-k logit
rankings token-by-token. It prints `PASS` or `FAIL` with the diverging token
position if it fails.

---

## 8. Quick Reference — Common Shapes (Qwen3-30B-A3B, TP=4)

| Symbol | Value | Meaning |
|---|---|---|
| `H` | 2048 | hidden size |
| `d` | 128 | head dim |
| `nh` | 32 | num attention heads (full) |
| `nkv` | 4 | num KV heads (full) |
| `nh_tp` | 8 | attention heads per TP rank |
| `nkv_tp` | 1 | KV heads per TP rank |
| `E` | 128 | num experts |
| `K` | 8 | top-K experts per token |
| `I` | 768 | MoE intermediate size (full) |
| `I_tp` | 192 | MoE intermediate size per TP rank |
| `LNC` | 2 | logical NC config (trn3) |
| `PMAX` | 128 | NKI partition tile size |

These constants appear repeatedly in kernel code. When you see a shape like
`[nkv_tp, d, S_max, d]` for a KV cache, that is `[1, 128, seq_len, 128]`.
