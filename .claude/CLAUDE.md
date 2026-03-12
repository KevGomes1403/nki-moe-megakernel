## NKI Kernel Development

When fixing NKI kernels, never remove existing optimizations (e.g., SBUF hoisting, DMA optimizations) unless explicitly asked. Fixes must preserve performance characteristics.

## General Rules
Do not modify files the user hasn't asked you to modify. If integration work requires changes to adjacent files, ask first before editing.

## Neuron Compiler

When exploring compiler flags or CLI options, do not assume a flag is invalid just because it doesn't appear in --help output. Ask the user before removing flags. Avoid long rabbit holes of bash exploration for internal binaries.

## NKI Kernel Development
For MoE kernels on trn2: bucket sizes must be 128-aligned, expert/tensor dimensions are intentionally hardcoded (pad inputs in the forward pass, not in the kernel), and always verify TP=4 shape assumptions before editing.