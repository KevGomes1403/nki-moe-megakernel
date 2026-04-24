# Semantic Mapping

## Goal

Turn compiled artifacts into an execution contract the kernel can match.

The useful questions are:

- in what order do stages execute?
- in what dtype?
- where are the rounding points?
- which model stage does each compiled block correspond to?

## Start From Model Stages

Write down the intended high-level stages first.

For a fused MoE example:

1. RMSNorm
2. router matmul
3. top-k
4. softmax
5. chosen-affinity renorm
6. gate/up matmul
7. activation
8. down matmul
9. expert weighting
10. final reduce

Then map HLO/custom calls/profile regions back to those stages.

## Trust Lowering Over Mental Models

A common mistake is stopping at “fp32 router” or “bf16 expert”.

Compiled HLO often reveals a more specific contract, for example:

- bf16 router dot
- top-k on bf16 logits
- fp32 full softmax
- bf16 selected-weight renorm
- bf16 SiLU input
- fp32 final weighting
- bf16 final reduce

That level of detail is what the kernel must match.

## For Each Stage, Record

- source op or source line
- HLO op or custom call
- input dtype
- output dtype
- whether a cast happens before or after the stage

## MoE-Specific Failure Domains

Separate:

### Routing

- router matmul dtype
- whether top-k runs on full logits or post-softmax values
- whether top-k sees bf16-rounded logits
- whether top-k set and order match
- renorm dtype for selected affinities

### Expert Path

- gate/up matmul dtype
- all-reduce dtype if applicable
- activation input/output dtype
- `silu(gate) * up` dtype
- down matmul dtype
- expert-weight multiply dtype

### Final Combine

- accumulation dtype across experts
- final output cast point

## Source Attribution

If `NEURON_FRAMEWORK_DEBUG=1` was enabled, `neuron-profile` can often attribute runtime cost back to framework source lines.

This is especially useful for selective-loading references:

- large time in gather/index paths usually means dynamic expert slicing dominates

## Practical Rule

If the compiled reference and your Python reconstruction disagree, fix the reconstruction or stop using it.

If the compiled reference and the kernel disagree, use HLO/profile evidence to decide which stage to edit first.
