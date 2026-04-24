# Harness

## Goal

Build a test harness that compares:

- a compiled Neuron reference path, usually PyTorch/NxDI via `build_module`
- a compiled NKI kernel wrapper using the same weights and inputs

The harness should answer:

1. Do the outputs differ?
2. Which inputs fail, and how often?
3. Which stage is responsible?

## `build_module`

Use `build_module` to compile both the reference and the kernel wrapper.

Pattern:

```python
ref_traced = build_module(
    model_cls=RefModule,
    model_kwargs={...},
    example_inputs=(hidden_states,),
    compiler_args=COMPILER_ARGS,
)
```

Important points:

- Mirror the real production path exactly.
- Force the same eval/training mode as production.
- Capture weights during the non-meta instantiation if you need simulator replay or failure dumps.
- Persist compiler workdirs when debugging.

## Kernel Wrapper

Wrap the NKI kernel in a small `nn.Module` that:

- reshapes inputs into kernel layout
- extracts weights in the kernel’s expected layout and dtype
- calls `nki.jit(kernel)[grid](...)`
- reshapes outputs back to the model boundary

Keep this wrapper minimal.

## Comparison and Tolerances

Use both strict and practical tolerances.

- strict: `atol=1e-5, rtol=0`
- loose bf16-native: `atol=1e-5, rtol=1e-2`
- bf16-noise-floor: `atol=max_abs(out) * 2^-7`

For hardware-gated comparisons, also use:

```python
import torch_neuronx.testing as tnx_testing
tnx_testing.assert_close(kern_out, ref_out, atol=1e-5, rtol=1e-2, check_device=False)
```

Do not treat universal strict failure as a bug by itself. bf16 kernels usually fail strict everywhere.

## Sample Sweeps

When debugging routing-sensitive kernels, sweep across:

- multiple activation scales
- multiple distributions
- multiple weight scales if needed

Useful distributions:

- normal
- student-t / heavy-tail
- laplace

## Failure Dumps

Save per-sample dumps for:

- the first strict failure
- the worst loose failure per configuration
- the top-N worst loose failures globally

Each dump should include:

- input tensor
- kernel output
- reference output
- kernel weights
- reference weights if available
- metadata: scale, distribution, sample index, max error, relative error

## Failure Classification

For MoE-style kernels, classify failures into:

- routing-set mismatch
- routing-order mismatch
- routing-weight mismatch with matched indices
- expert-path mismatch after matched routing

This stops you from debugging the wrong subsystem.

## Stage Tracers

Use Python-side stage tracers as diagnostics, not as ground truth.

Good traced tensors:

- RMSNorm output
- router logits before and after rounding
- top-k indices
- top-k weights
- per-expert activation intermediates
- down-projection output
- weighted expert contribution

If a Python “reference-like” trace disagrees with compiled Neuron, stop trusting it and inspect the real HLO.
