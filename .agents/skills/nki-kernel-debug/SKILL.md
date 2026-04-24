---
name: nki-kernel-debug
description: Debug Neuron NKI kernels against a compiled Neuron or NxDI reference implementation. Use when Codex needs to compare an NKI kernel to a PyTorch or NxDI reference, build a hardware harness with `build_module`, inspect HLO/NEFF/NTFF artifacts, map compiled ops back to model semantics, use `torch_neuronx.testing.assert_close`, or use `nki.simulate` for local debugging.
---

# NKI Kernel Debug

Use this skill to debug correctness gaps between an NKI kernel and a compiled Neuron reference path.

## Workflow

1. Start with the harness in [references/harness.md](references/harness.md).
2. Run the reference and kernel on the same inputs and collect failure dumps.
3. If the gap is still unclear, inspect compiled artifacts with [references/hlo-neff-runtime-inspect.md](references/hlo-neff-runtime-inspect.md).
4. Map those artifacts back to model stages with [references/semantic-mapping.md](references/semantic-mapping.md).
5. Use [references/nki-simulate.md](references/nki-simulate.md) to reproduce and narrow issues locally.

## Decision Tree

- If there is no trustworthy reference path yet:
  Build one with `build_module`.

- If hardware and a Python reconstruction disagree:
  Inspect the real HLO before trusting the reconstruction.

- If the question is “what precision/order does NxDI really execute?”:
  Use HLO first, then runtime inspect and `neuron-profile`.

- If the question is “what model op does this compiled block correspond to?”:
  Use the semantic-mapping reference.

- If the question is “why does the simulator differ from hardware?”:
  Read the simulator limitations before editing the kernel.

## Working Rules

- Prefer compiled Neuron artifacts over CPU lookalikes when they disagree.
- Treat routing, expert math, and final accumulation as separate failure domains.
- Check exact dtype transitions and rounding points, not just broad “bf16 vs fp32” labels.
- Keep the debug harness aligned with the production kernel source.

## References

- [references/harness.md](references/harness.md)
  Build the reference/kernel harness, choose tolerances, capture dumps, and classify failures.

- [references/hlo-neff-runtime-inspect.md](references/hlo-neff-runtime-inspect.md)
  Dump HLO and NEFF, generate NTFFs with runtime inspect, and use `neuron-profile` for source attribution.

- [references/semantic-mapping.md](references/semantic-mapping.md)
  Turn compiled artifacts into an execution model the kernel can match.

- [references/nki-simulate.md](references/nki-simulate.md)
  Use `nki.simulate`, `NKI_PRECISE_FP=1`, and Python-native debugging effectively.
