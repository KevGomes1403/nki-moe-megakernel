# `nki.simulate`

## What It Is

`nki.simulate` runs an NKI kernel on CPU using Python and NumPy. It is ideal for:

- fast local iteration
- correctness checks
- inspecting intermediate tensors
- catching uninitialized-memory bugs

It is not a compiler-faithful or performance-faithful model of Neuron hardware.

## Basic Usage

```python
sim_result = nki.simulate(my_kernel)(a_np, b_np)
```

For a grid-decorated kernel:

```python
sim = nki.simulate(nki.jit(my_kernel)[grid])
out = sim(a_np, b_np)
```

Inputs should be NumPy arrays.

## Precise Floating Point

Use:

```bash
NKI_PRECISE_FP=1 python my_script.py
```

Without this, low-precision types such as bf16 are stored as float32 for speed.

With `NKI_PRECISE_FP=1`, the simulator uses `ml_dtypes` low-precision storage and is much closer to hardware numerically.

## Platform Target

Set:

```bash
NEURON_PLATFORM_TARGET_OVERRIDE=trn1
NEURON_PLATFORM_TARGET_OVERRIDE=trn2
NEURON_PLATFORM_TARGET_OVERRIDE=trn3
```

Use the same target as the hardware path you are comparing against.

## Debugging

Because the kernel runs as Python:

- `breakpoint()` works
- IDE breakpoints work
- `print()` works
- `nl.device_print()` works

This is useful for:

- checking shapes and dtypes
- printing partial tiles
- validating branch conditions
- understanding control flow

## Uninitialized Memory Detection

The simulator fills newly allocated memory with sentinel values:

- NaN for floating-point
- `4` for integer types

A useful check is:

```python
assert not np.any(np.isnan(result))
```

If you see NaNs, suspect:

- partial writes
- off-by-one loop bounds
- conditional writes that skip elements

## LNC2 / Multi-Core Behavior

For kernels run as `kernel[2]`, the simulator uses two Python threads and supports:

- `program_id`
- `shared_hbm`
- `sendrecv`
- `core_barrier`

This is good for functional debugging, not performance conclusions.

## Limitations

- it does not model instruction scheduling or engine parallelism
- some hardware checks are missing
- some ISA ops have correctness gaps
- memory-capacity conflicts may not appear in simulation
- arbitrary Python meta-programming may simulate successfully and still fail to compile

## Best Practice

Use the simulator to narrow bugs and inspect state.

Use hardware HLO/runtime artifacts to settle semantic disputes.
