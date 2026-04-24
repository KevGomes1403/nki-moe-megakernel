# VLM Translation Plan: HF -> AWS Neuron (NxDI)

This guide is the execution plan for translating **vision-language (multimodal)** HuggingFace models to Neuron with NxDI.  
Use this when the target model consumes image inputs (`pixel_values`, image chunks, aspect ratios, image masks) in addition to text.

If the target is text-only, use the main workflow in [SKILL.md](../SKILL.md).

## 1. Neuron/NxDI Mental Model

1. Trainium execution is graph-compiled and shape-sensitive. Variable input lengths must be handled through **bucketing**.
2. NxDI is split into:
   - `models/`: model/application abstractions (`NeuronBaseModel`, `NeuronBaseForCausalLM`, `NeuronBaseForImageToText`).
   - `modules/`: reusable primitives (`NeuronAttentionBase`, KV cache, autobucketing, padding, custom calls).
3. Multimodal Image-to-Text models usually compile **two subgraphs**:
   - Text graph (context encoding + token generation).
   - Vision graph (image encoder + projector).
4. The base class [`image_to_text_model_base.py`](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/image_to_text_model_base.py) handles separate compile/load/shard for text and vision builders.
5. Tracing currently does not support kwargs for this path. [`ImageToTextModelWrapper`](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/image_to_text_model_wrapper.py) enforces a fixed 24-argument ordered signature (vision args at positions 22 and 23).

## 2. Pick the Correct Reference Pattern

Use the closest model family before writing any code:

1. **Pixtral-style (vision embeddings scattered into text sequence)**  
   Files:
   - [`models/pixtral/modeling_pixtral.py`](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/pixtral/modeling_pixtral.py)
   - [`models/pixtral/modeling_pixtral_vision.py`](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/pixtral/modeling_pixtral_vision.py)
   Use for LLaVA-like and Qwen2-VL-like integration (vision tower + projector + scatter into token positions).

2. **Llama4-style (chunked images + vision adapter + scatter)**
   Files:
   - [`models/llama4/modeling_llama4.py`](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/llama4/modeling_llama4.py)
   - [`models/llama4/modeling_llama4_vision.py`](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/llama4/modeling_llama4_vision.py)
   Use when the model already uses chunked/tiling image flow.

3. **MLlama-style (cross-attention in text decoder, not simple scatter)**
   Files:
   - [`models/mllama/modeling_mllama.py`](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/mllama/modeling_mllama.py)
   - [`models/mllama/modeling_mllama_vision.py`](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/mllama/modeling_mllama_vision.py)
   Use when text layers include explicit cross-attention blocks and dedicated multimodal KV handling.

4. **Diffusion/vision generation models (Flux)**  
   Files:
   - [`models/diffusers/flux/application.py`](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/diffusers/flux/application.py)
   Use a separate pipeline/application architecture; do not force Image-to-Text base classes.

## 3. Canonical Runtime Flow (Image-to-Text Path)

1. `NeuronBaseForImageToText.__init__` builds text context/token models first, then `enable_vision_encoder()`.
2. `compile()` traces and saves **text model** and **vision model** separately.
3. At inference prefill:
   - Vision wrapper preprocesses image input.
   - Vision model returns projected vision embeddings.
   - Vision mask is converted to integer positions and padded to selected text bucket.
   - Text model receives `vision_embeddings` and `vision_mask`.
4. At token generation (or text-only requests):
   - Use dummy vision tensors from `ImageToTextModelWrapper.get_dummy_vision_inputs`.

## 4. Translation Steps (Execution Plan)

### Step 0: Preflight

1. Activate environment:
   - `source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`
2. Confirm model family (Pixtral/Llama4/MLlama/Flux-like).
3. Confirm compile constraints:
   - bucket strategy,
   - tp/dp degree,
   - unsupported features for the target reference (e.g., medusa/prefix caching for many VLM paths).

Deliverable: short architecture classification note and selected NxDI reference file paths.

### Step 1: Inventory Source HF Model

1. Enumerate blocks in source:
   - patch embedding,
   - vision attention/MLP/norm,
   - projector,
   - text decoder integration method (scatter vs cross-attn),
   - positional encoding (2D RoPE/MRoPE/absolute/etc.).
2. Extract HF config attributes needed at runtime.
3. Decide whether text and vision need separate neuron configs.

Deliverable: block map + config attribute list.

### Step 2: Define Config Classes

1. For image-to-text pattern, create `ImageToTextInferenceConfig` subclass with:
   - nested `text_config`,
   - nested `vision_config`,
   - `get_required_attributes()` including both trees.
2. Use `NeuronConfig` or a custom subclass if extra Neuron-only fields are required.
3. For MLlama-like integration, use multimodal config class if cross-attention/multimodal KV requires custom config behavior.

Deliverable: config classes with validated required attributes.

### Step 3: Translate Vision Encoder

1. Replace unsupported/high-cost ops with NxDI-compatible primitives:
   - `Conv2d` patch embed -> unfold/patchify + `ColumnParallelLinear` (Pixtral/MLlama style) or approved parallel conv wrapper.
   - attention -> `NeuronAttentionBase` subclass.
   - dense layers -> `ColumnParallelLinear` / `RowParallelLinear`.
   - norm -> `CustomRMSNorm` on Neuron and CPU-safe fallback for tests.
2. Implement vision positional encoding compatible with the source model:
   - 2D RoPE pattern (Pixtral),
   - chunk/packing-specific frequency handling (Llama4),
   - any source-specific variant.
3. Include multimodal projector to map vision hidden size -> text hidden size.

Deliverable: Neuron vision model module returning text-dimension vision embeddings.

### Step 4: Implement Vision Wrapper

1. Subclass `ModelWrapper` for vision encoder.
2. Implement `input_generator()` per bucket.
3. Implement preprocessing `forward(...)`:
   - patchify/chunk,
   - build attention mask/position ids,
   - bucket routing,
   - pad to target bucket,
   - call compiled model,
   - unpad if needed.
4. Use `EncoderModelInstance` in `get_model_instance()`.

Deliverable: wrapper that converts raw image inputs to valid traced args across buckets.

### Step 5: Integrate Text Model with Vision

1. If scatter-based:
   - override text model `encode_vision_to_input(...)` and scatter embeddings using positional mask.
2. If cross-attention-based:
   - implement multimodal cross-attention path and cache behavior (MLlama pattern).
3. Ensure text forward accepts/propagates vision args in prefill and uses dummy inputs in token generation.

Deliverable: text model that correctly consumes vision outputs in both prefill and generation paths.

### Step 6: Build Application Head

1. Subclass `NeuronBaseForImageToText` (or alternative base for non-image-to-text pipeline).
2. Set:
   - `text_model_cls`,
   - `vision_model_cls`,
   - `text_model_wrapper`,
   - `vision_model_wrapper`.
3. Implement `enable_vision_encoder()`.
4. Implement `get_required_kwargs()` for generation adapter (for example: `pixel_values`, `vision_mask`, `image_sizes`).
5. Implement forward branch logic:
   - prefill with images -> run vision encoder,
   - token generation/text-only -> dummy vision inputs.

Deliverable: complete multimodal application class.

### Step 7: Implement Weight Mapping

1. Start from text mapping (reuse reference model conversion where possible).
2. Add vision mapping:
   - rename prefixes (`vision_model`, `vision_tower`, etc.),
   - convert/fuse QKV layout if needed,
   - reshape patch conv weights when moving between conv and unfold+linear forms.
3. Inject rank metadata tensors required by NxDI parallel layers where applicable.
4. Validate converted keys against Neuron model `state_dict()` patterns.

Deliverable: `convert_hf_to_neuron_state_dict` with no missing required keys.

### Step 8: Validate Before Compile

1. Unit test each translated block against HF equivalents on CPU.
2. Run end-to-end CPU forward parity (prefill with image + one decode step).
3. Verify expected shapes and dtypes:
   - `vision_embeddings`: `[B, seq_or_patch_tokens, hidden_text]` (or model-specific chunk form before flattening),
   - `vision_mask`: integer positions for traced path.

Deliverable: passing block and E2E parity checks.

### Step 9: Compile, Load, and Smoke Test

1. Compile text and vision subgraphs.
2. Load weights for both subgraphs.
3. Run warmup/smoke generation:
   - image+text prefill,
   - text-only prefill,
   - token generation continuation.
4. Check bucket routing logs and ensure no overflow beyond largest bucket.

Deliverable: compiled and runnable multimodal Neuron model.

### Step 10: Performance Tuning

1. Set vision buckets to realistic patch/chunk ranges.
2. Minimize unnecessary casts and int64 usage in traced inputs.
3. Tune compiler args and cc overlap based on reference model defaults.
4. Use per-subgraph compiler work directories and snapshot hooks for debugging if needed.

Deliverable: stable baseline latency/throughput profile with documented compiler settings.

## 5. Primitive Replacement Map (Default)

1. `nn.Linear` (Q/K/V/gate/up) -> `ColumnParallelLinear`
2. `nn.Linear` (O/down) -> `RowParallelLinear`
3. `nn.Embedding` -> `ParallelEmbedding`
4. Attention -> `NeuronAttentionBase` subclass
5. `nn.RMSNorm` -> `CustomRMSNorm` (CPU fallback in tests)
6. Vision conv patch embedding -> unfold/patchify + parallel linear if direct conv path is unsupported for target flow

## 6. Critical Gotchas

1. Image-to-text tracing uses ordered positional args; missing placeholder args can break runtime.
2. Vision mask is often converted from boolean mask to **positions tensor** for performance/tracing compatibility.
3. Bucket mismatch between text and vision paths is a common failure mode.
4. Always use vision-specific neuron config inside vision wrappers/models to avoid cross-config contamination.
5. CPU tests require fallback norms/operators where Neuron custom calls are device-specific.

## 7. Definition of Done

1. Config classes validate against HF config.
2. Vision and text models compile/load successfully.
3. Weight conversion loads without missing required keys.
4. Prefill with image input runs and injects vision context correctly.
5. Token generation path runs with dummy or cached vision inputs.
6. Documented bucket plan and compiler flags are checked into the model translation artifact.
