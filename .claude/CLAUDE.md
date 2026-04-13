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
- Target hardware is Trainium2 (trn2). DO NOT suggest trn3-only instructions like `nc_matmul_mx` or MXFP4 primitives unless explicitly asked.
- Kernels must work with LNC=2 sharding and respect `shared_hbm` requirements.