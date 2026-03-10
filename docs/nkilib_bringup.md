# NKI Library Kernel Bring-Up Guide (Trainium)

This document captures a practical workflow for making `nkilib` kernels invocable from Python, with examples and debugging notes validated on `trn2`.

## 1. Environment and Runtime Setup

Use the Neuron virtualenv and set target explicitly when platform detection is unreliable:

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2   # or trn2 as needed
```

If omitted, you may hit:
- `ValueError: Could not identify a target platform...`

## 2. Bring-Up Checklist (Any nkilib Kernel)

1. Read kernel signature and shape contracts in `<kernel>.py`.
2. Find torch reference in `<kernel>_torch.py` and use it as golden.
3. Determine invocation style:
- Normal NKI kernel (`@nki.jit`) returning output tensor.
- Trace-style library kernel (`nki.jit(func, mode="trace")`) with in-place outputs.
4. Build smallest legal test shape first.
5. Compare vs torch reference with FP32 diff metrics.
6. Only then scale shape/features.

## 3. Two Invocation Patterns You Must Distinguish

## Pattern A: Standard JIT kernel call
Example: `attention_cte`

```python
out = attention_cte(q, k, v, causal_mask=True, tp_q=True, tp_k=False, tp_out=False)
```

## Pattern B: Trace-style function kernel
Example: `attention_tkg`

```python
attention_tkg_jit = nki.jit(attention_tkg, mode="trace")
status = attention_tkg_jit(..., out=out_tensor, ...)
# output is written into out_tensor in-place; status is often int (e.g., 0)
```

Do not assume trace-style kernels return `(out, ...)`.

## 4. attention_cte Bring-Up Notes

File:
- `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/nkilib/core/attention/attention_cte.py`

Golden:
- `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/nkilib/core/attention/attention_cte_torch.py`

Known-good minimal shapes:
- `q`: `(bs, seqlen_q, d)` when `tp_q=True`
- `k`: `(bs, d, seqlen_k)` when `tp_k=False`
- `v`: `(bs, seqlen_k, d)`

Test example in repo:
- `/home/ubuntu/nki-moe/nki_attention_cte_example.py`

### trn1-specific fixes applied

On `trn1` (CoreV2), stock kernel hit unsupported compiler paths. Two fixes were needed:

1. **Non-FP32 PSUM transpose issue**
- Symptom: `Cannot create non-FP32 PSUM tensor ... in CoreV2`
- Fix: force transpose PSUM dtype to `nl.float32` on `gen2` in:
  - `attention_cte.py` around `_load_q_tile` and `_load_k_tile`.

2. **SBUF->SBUF dma_transpose issue**
- Symptom: `transpose only supported for HBM->SB`
- Fix: in `_exp_impl`, replace `dma_transpose` path on `gen2` with tile-wise `nc_transpose + tensor_copy` fallback.

Patched locations:
- `/opt/.../attention_cte.py` around lines ~1595, ~1760, ~2151.



## 6. Debugging Playbook (Symptom -> Action)

- `Could not identify a target platform`  
  -> Set `NEURON_PLATFORM_TARGET_OVERRIDE`.

- `entry function ... not found` while wrapping library kernel in another jitted function  
  -> JIT the library kernel directly: `nki.jit(attention_tkg, mode="trace")`.

- `Cannot allocate in stack without an open scope`  
  -> `sbm.open_scope()` before invoking kernel.

- `Insufficient memory ... only have -X bytes`  
  -> Use explicit positive SBUF range for `SbufManager`.

- RoPE/head-dim assertion failures  
  -> Ensure `d_head` satisfies kernel constraints (e.g., divisible by 64 for this TKG path).

- Large mismatch vs torch  
  -> Lower random input scale first (e.g., `0.1`) and compare in FP32.

## 7. Validation Pattern

Always compute:

```python
diff = (nki_out.float() - ref_out.float()).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()
```

Recommended for bring-up:
- Start with permissive BF16 tolerance (e.g., `1e-2` to `5e-2` depending on kernel complexity).
- Tighten after stable shape/path is confirmed.

## 8. Known Working Commands

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python /home/ubuntu/nki-moe/nki_attention_cte_example.py
python /home/ubuntu/nki-moe/nki_attention_tkg_example.py
```

## 9. Relevant Local Files

- `/home/ubuntu/nki-moe/nki_tensor_add_example.py`
- `/home/ubuntu/nki-moe/nki_attention_cte_example.py`
- `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/nkilib/core/attention/attention_cte.py`
- `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/nkilib/core/attention/attention_cte_torch.py`
- `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/nkilib/core/attention/attention_tkg.py`
- `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/nkilib/core/attention/attention_tkg_torch.py`
