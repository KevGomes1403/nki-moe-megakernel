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

## Code Style
Applies to all code written in this repo. Default Claude verbosity is wrong here — write less, write tighter.

- **Comments are concise and focused.** One short line. Describe what's non-obvious about the code, not what the code already says through its identifiers.
- **No long rationales in code.** Don't leave paragraphs explaining "I tried X but it regressed by Yus because the scheduler...". Past A/B tests, profiling deltas, and investigation history belong in PR descriptions, commit messages, or design docs — not in the source file. Future readers need to know what the code does, not how it was arrived at.
- **No task-specific or plan-specific references in code.** Don't write "Phase 1", "the live path", "added for the qwen3 spec branch", "ROOT CAUSED in <commit>", "this fixes the bug from issue #123", "TODO from yesterday's review". They make no sense out of context and rot fast.
- **Docstrings are brief and structured.** Match the surrounding file's existing docstring sections (e.g. `Computes:`, `Tiled computation:`, `Returns:`, `Note:` in nkilib-derived files). For optional kwargs that select alternative code paths, list them as a short precedence block, not multi-paragraph "Optional X:" sub-sections.
- **No commentary about the change itself.** Don't write "NEW: this kwarg was added for...", "MODIFIED to handle...", "this used to be Y but is now Z." The diff shows what changed; the code shows the current state.
- **Module-level constants** (env flags, dimension defaults, dtype maps) live together at the top of the module, not wedged between import groups or buried near first use.
- **Don't make drive-by edits.** When fixing or adding something, change only what the task requires. Dropped `align=` kwargs, swapped dim references in unrelated branches, reformatted unrelated functions — all show up as suspect in review and waste reviewer time.
- **Additive when extending vendored code.** Files under `nki_kernels/` are vendored from upstream `nkilib/` (see `nki_kernels/_vendor_meta.md`). When extending them with new functionality, the upstream behavior must remain reachable (typically env-gated off by default), and the diff against upstream should read as "imports + additive code + tighter comments."

When asked to "clean up", these are the rules to apply.

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