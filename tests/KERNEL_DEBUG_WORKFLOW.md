# v30c Kernel Debug Workflow

Two test files form a hardware-validate → CPU-debug loop for the v30c MoE NKI kernel.

---

## File Roles

### `test_v30c_vs_nxdi_trn3.py` — hardware accuracy gate

Compiles two modules on real Trainium hardware:
- **`RefMoEModule`** — pure-PyTorch NxDI MoE (the ground truth, `moe_fused_nki_kernel_enabled=False`)
- **`KernelMoEModule`** — v30c NKI kernel wrapped for hardware execution

Runs 512 inputs per weight scale (4 activation scales × 2 distributions × 64 repeats, seeds `200+i`).
Validates **per-sample** so the first failure is captured immediately and a `.pt` dump is written.

### `test_v30c_simulate.py` — CPU simulator for development and debugging

Runs the same kernel body (`_v30c_moe_raw`) through `nki.simulate` — no hardware needed.
Two modes:

| Mode | When to use |
|------|-------------|
| Default (512-sample run) | Full correctness sweep on CPU; much faster to iterate than hardware |
| `--load-dump PATH` | Reproduce a specific hardware failure locally for debugging |

---

## The Kernel Is In the Simulator Test

When improving or debugging the kernel:

1. **Edit `_v30c_moe_raw` only in `test_v30c_simulate.py`** — the simulator is the fast iteration target.
2. **It is imported in `test_v30c_vs_nxdi_trn3.py`** instead of maintaining a separate copy.

This ensures hardware and simulator tests always run the same kernel body.

---

## Failure Workflow

```
python tests/test_v30c_vs_nxdi_trn3.py
  ↓ (first failing sample)
  First failure: sample 47  →  dump written to: ./failing_ws1_s47.pt
  Debug with simulator:
    NKI_PRECISE_FP=1 python tests/test_v30c_simulate.py --load-dump ./failing_ws1_s47.pt

NKI_PRECISE_FP=1 python tests/test_v30c_simulate.py --load-dump ./failing_ws1_s47.pt
  ↓
  --- simulator vs kernel output (expected by hardware kernel) ---
  MATCH          ← simulator reproduces hardware kernel faithfully; debug here

  --- simulator vs hardware reference ---
  MISMATCH       ← this is the actual bug
```

**Interpreting the two comparisons:**

- **`simulator vs kernel output: MATCH`** — the CPU simulator reproduces what the hardware kernel did. The simulator is trustworthy for this failure; any fix you validate in the simulator will translate to hardware.
- **`simulator vs hardware reference: MISMATCH`** — the numerical discrepancy between the kernel and the NxDI reference. This is what needs to be fixed.
- **`simulator vs kernel output: MISMATCH`** — rare; means the simulator and hardware diverge for this kernel (e.g. due to a simulator gap). Treat simulator results with caution in this case.

### Dump file contents

The `.pt` file (loaded with `torch.load`) contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `input` | `[1, 1, 2048]` bf16 | The failing input token |
| `gamma` | `[1, 2048]` bf16 | RMSNorm weight |
| `router_w` | `[2048, 128]` bf16 | Router weight |
| `gate_up` | `[128, 2048, 384]` bf16 | Expert gate+up projection |
| `down` | `[128, 192, 2048]` bf16 | Expert down projection |
| `kern_out` | `[1, 1, 2048]` bf16 | Hardware kernel output (buggy) |
| `ref_out` | `[1, 1, 2048]` bf16 | Hardware reference output (correct) |
| `weight_scale` | float | Weight scale factor used |
| `seed` | int | Module init seed (42) |
| `sample_idx` | int | Which of the 512 samples failed |

---

## NKI Simulator Debugging Tools

The simulator executes kernel code as regular Python, giving full access to Python's debugging ecosystem. Always run with `NKI_PRECISE_FP=1` to get real bf16 storage (via `ml_dtypes`) instead of float32 upcast.

### `nl.device_print` — inspect tensor values mid-kernel

```python
nl.device_print("router logits", router_logits_sb)
```

Prints the tensor to stdout during simulation. Works anywhere in the kernel body.

### `print()` — inspect any Python value

Since the simulator runs as regular Python, `print()` works on any intermediate value, shape, or scalar.

### `breakpoint()` / PDB — step through execution

```python
@nki.jit
def _v30c_moe_raw(inp, gamma, router_w, gate_up_w, down_w):
    ...
    breakpoint()   # execution stops here; inspect tensors interactively
    ...
```

Run `NKI_PRECISE_FP=1 python tests/test_v30c_simulate.py --load-dump dump.pt` and PDB will stop at the breakpoint. All NKI tensor objects are inspectable as NumPy-backed `NkiTensor` instances.

### IDE debuggers (VSCode / PyCharm)

Set breakpoints directly in the kernel source. The simulator runs in-process, so IDE debuggers attach with no special configuration.

### NaN sentinel detection

The simulator fills all newly allocated tensors with NaN (floats) or 4 (ints). Any read from uninitialized memory will propagate NaN to the output. The `KernelSimModule` already checks for all-zero output; extend it to check for NaN if needed:

```python
assert not np.any(np.isnan(np.array(out_np, dtype=np.float32))), "uninitialized read"
```

---

## Running the Tests

```bash
# Full hardware sweep (compiles twice, runs 512 samples × 3 weight scales on trn3)
python tests/test_v30c_vs_nxdi_trn3.py

# Subset for faster iteration
python tests/test_v30c_vs_nxdi_trn3.py --weight-scales 1.0 --n-samples 32

# Full simulator sweep (no hardware; use for rapid kernel iteration)
NKI_PRECISE_FP=1 python tests/test_v30c_simulate.py

# Subset
NKI_PRECISE_FP=1 python tests/test_v30c_simulate.py --n-samples 32

# Debug a specific hardware failure
NKI_PRECISE_FP=1 python tests/test_v30c_simulate.py --load-dump ./failing_ws1_s47.pt
```

---

## Simulator Limitations (What It Cannot Catch)

| Gap | Impact |
|-----|--------|
| No compilation | Simulator passes ≠ hardware compiles. Meta-programming restrictions (no arbitrary closures, restricted Python subset) only surface at compile time. |
| SBUF/PSUM capacity | Simulator allocates tensors independently; it does not validate total SBUF usage against hardware limits. Kernels that overflow SBUF compile and run on simulator but fail or corrupt on hardware. |
| Numerical precision | Even with `NKI_PRECISE_FP=1`, some accumulation sequences differ from hardware (e.g. matmul internal precision). Use hardware tests to confirm bf16 tolerance. |
| No engine parallelism | Simulator executes instructions sequentially. DMA/compute overlap and pipeline effects are not modeled. |
