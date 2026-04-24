# HLO, NEFF, and Runtime Inspect

## What Each Artifact Tells You

- HLO:
  op order, dtype conversions, custom calls, reduction structure

- NEFF:
  compiled binary artifact used by Neuron runtime

- NTFF:
  runtime trace for an actual execution; needed for `neuron-profile`

- `neuron-profile` JSON:
  instruction timing, source attribution, engine/opcode breakdown

## Dumping HLO and NEFF

The easiest path is a persistent `build_module` workdir.

Typical contents:

- `graph.neff`
- `model.<hash>.neff`
- `model.<hash>.hlo_module.pb`
- `command.txt`
- compile logs

If you only have `.hlo_module.pb`, convert it to a readable textproto with a helper script in the repo.

## Runtime Inspect Environment

Set these on a real compile + run when you want runtime artifacts and source attribution:

```bash
NEURON_FRAMEWORK_DEBUG=1
XLA_IR_DEBUG=1
XLA_HLO_DEBUG=1
NEURON_RT_INSPECT_ENABLE=1
NEURON_RT_INSPECT_DEVICE_PROFILE=1
NEURON_RT_INSPECT_OUTPUT_DIR=./output
```

`NEURON_FRAMEWORK_DEBUG=1` matters for source-code attribution.

## Identifying the Right NTFF

Runtime inspect may emit multiple `.ntff` files. Use:

```bash
neuron-profile show-session -n <neff> -s <ntff>
```

Pick the NTFF whose reported model name matches the NEFF you care about.

## Exporting a Full Profile

```bash
neuron-profile view -n <neff> -s <ntff> --output-format json --output-file profile.json
```

Use the JSON for:

- layer/source aggregation
- engine/opcode aggregation
- reference vs kernel diffs

## What to Look For in HLO

For each stage, identify:

- operand dtype
- output dtype
- explicit `convert` nodes
- reduction dtype
- custom calls

For example:

- Is router matmul bf16 or f32?
- Is top-k on logits or softmax probabilities?
- Is selected-affinity renorm bf16 or f32?
- Is final expert weighting bf16 or f32?
- Is the final sum bf16 or f32?

## What to Look For in `neuron-profile`

Start with:

- hottest layers
- hottest source lines
- engine totals
- opcode totals

Examples:

- reference dominated by `aten__index_gather`:
  selective-loading overhead

- kernel dominated by `GpSimd MOVE`, `EVENT_SEMAPHORE`, or opaque custom-call work:
  control or data-motion overhead inside the fused kernel

## Custom NKI Kernels

At HLO level, an NKI kernel often appears as:

- `AwsNeuronCustomNativeKernel`

When that happens:

1. inspect the kernel compile-cache JSON if available
2. use runtime inspect for instruction-level shape
3. map source lines back to the kernel file

The NKI compile-cache JSON often exposes:

- allocation dtypes
- instruction list
- engine/opcode
- source filename and line number
