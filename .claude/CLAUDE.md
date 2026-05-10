## NKI Kernel Development

When fixing NKI kernels, never remove existing optimizations (e.g., SBUF hoisting, DMA optimizations) unless explicitly asked. Fixes must preserve performance characteristics.

## General Rules
Do not modify files the user hasn't asked you to modify. If integration work requires changes to adjacent files, ask first before editing.

Do not read files in this repo for context unless the user references them. Some of them may be deprecated. 
You can reliably read the library files for context:
1. NxDI: /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/
2. NKI Library: /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/nkilib/

## Neuron Compiler

When exploring compiler flags or CLI options, do not assume a flag is invalid just because it doesn't appear in --help output. Ask the user before removing flags. Avoid long rabbit holes of bash exploration for internal binaries.

## NKI Kernel Development
For MoE kernels on trn2: bucket sizes must be 128-aligned, expert/tensor dimensions are intentionally hardcoded (pad inputs in the forward pass, not in the kernel), and always verify TP=4 shape assumptions before editing.

## Hardware Constraints
- Target hardware is Trainium3 (trn3). 
- Kernels must work with LNC=2 sharding and respect `shared_hbm` requirements.

## Multi-Layer Fused TKG Megakernel
Ongoing effort: extend the single-layer Qwen3 MoE TKG kernel (`megakernels/qwen3_moe/transformer_qwen.py`) into a multi-layer megakernel that runs all decoder layers in one NKI invocation, with SBUF-resident residual across layer boundaries and in-place KV cache updates. Requires moving pre-attention RMSNorm (`input_layernorm`) inside the kernel, modifying v13bc attention to write KV in-place, and overriding `NeuronQwen3MoeModelV2.forward` to bypass NxDI's per-layer loop. Reference template: `nkilib/experimental/transformer/transformer_tkg.py`.

## Repo Layout
- `megakernels/<model>/` — per-model megakernel + NxDI integration. Each model folder contains `<model>.py` (XLA baseline subclass), `<model>_with_megakernel.py` (NKI-enabled subclass, used when `--enable-nki`), `transformer_<model>.py` (the multilayer megakernel itself), and any model-specific NKI subkernels.
- `nki_kernels/{attention,moe,norm}/` — shared NKI primitives reused across models (vendored from nkilib for op-name uniqueness — see `nki_kernels/_vendor_meta.md`).
- `tests/<model>/` — per-model unit tests for kernels.
- `main.py` — CLI / benchmark entrypoint. Model registry in `_MODEL_REGISTRY` maps `--model` to the per-model module triple.