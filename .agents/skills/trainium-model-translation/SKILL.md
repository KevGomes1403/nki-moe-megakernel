---
name: translate-model
description: Port models defined in PyTorch to run on AWS Trainium accelerators.
---

# AWS Trainium
AWS Neuron hardware consists of custom-designed machine learning accelerators optimized for deep learning workloads. 

At the heart of the Trn1 instance are 16 x Trainium chips (each Trainium include 2 x NeuronCore-v2). Trainium is the second generation purpose-built Machine Learning accelerator from AWS.

| **Category**        | **Specification**                                                                                                                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Compute**         | Two NeuronCore-v2 delivering:<br>• 380 INT8 TOPS<br>• 190 FP16 / BF16 / cFP8 / TF32 TFLOPS<br>• 47.5 FP32 TFLOPS                                                                                          |
| **Device Memory**   | • 32 GiB device memory (model state storage)<br>• 820 GiB/sec memory bandwidth                                                                                                                            |
| **NeuronLink**      | NeuronLink-v2 chip-to-chip interconnect enabling:<br>• Efficient scale-out training<br>• Memory pooling across Trainium chips                                                                             |
| **Programmability** | • Dynamic shapes and control flow via NeuronCore-v2 ISA extensions<br>• User-programmable rounding mode (Round Nearest Even Stochastic Rounding)<br>• Custom operators via deeply embedded GPSIMD engines |


# NeuronX Distributed Inference

NxD Inference (where NxD stands for NeuronX Distributed) is an open-source PyTorch-based inference library that simplifies deep learning model deployment on AWS Inferentia and Trainium instances. It offers advanced inference capabilities, including features such as continuous batching and speculative decoding for high performance inference. Neuronx Distributed Inference includes a model hub and modules that users can reference to implement their own models on Neuron.

NxD Inference(NxDI) library offers the following benefits:
- Production ready models: NxD Inference provides production ready models like Llama-3.1, DBRX, and Mixtral that you can quickly deploy for high performance inference.
- LLM Inference Features: NxD Inference provides support for various LLM inference features like KV Cache, Multi-Head Attention (MHA), Grouped Query Attention (GQA), Flash Attention, Quantization, MoE , Continuous Batching and Speculative Decoding enabling high performance inference.
- Modular Design: Inference features in NxDI like KV Caching are implemented with a modular design, allowing developers to easily incorporate them into new models or customize and extend them.
- Distributed Strategies: NxD Inference enables distributing inference workload of large models across multiple NeuronCores in a single instance using Tensor parallelism and Sequence Parallelism. Pipeline parallelism and multi-node inference will be supported in future Neuron releases.
- Support for NKI Kernels: NxD Inference provides support for integrating custom NKI kernels on Trainium and Inferentia instances.

## Defining Models in NxDI
This guide demonstrates how to adapt an existing PyTorch model to run on Neuron with the NeuronX Distributed (NxD) Inference library. 

### 1. Define a NeuronConfig Class
Define a Neuron configuration class, which extends NeuronConfig. NeuronConfig includes Neuron-specific configuration parameters. In the config class for your model, you can define any additional Neuron-specific configuration parameters that your model requires.

```python
from neuronx_distributed_inference.models.config import NeuronConfig

class NeuronLlamaConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set any args/defaults
```
### 2. Define an InferenceConfig class

Define an inference configuration class, which extends InferenceConfig. InferenceConfig includes model parameters, such as those from a HuggingFace PretrainedConfig (like LlamaConfig). When users initialize your config, they can provide required attributes directly, or they can populate the config from a HuggingFace PretrainedConfig. You can also override get_required_attributes to enforce that certain attributes are present.

```python
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig

class LlamaInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronLlamaConfig
```

### 3. Define a Neuron model

This class is a subclass of NeuronBaseModel, which is a PyTorch module.

1. In this class, you provide implementations for setup_attr_for_model(self, config) and init_model(self, config).

    1. In setup_attr_for_model, set values for the following attributes. You can set these attributes from values in config and config.neuron_config.

        - self.on_device_sampling
        - self.tp_degree
        - self.hidden_size
        - self.num_attention_heads
        - self.num_key_value_heads
        - self.max_batch_size
        - self.buckets

    2. In init_model, initialize the modules that make up the model.
        -  For attention modules, extend NeuronAttentionBase, which provides a group query attention (GQA) implementation adapted to Neuron.
        - Replace linear layers (such as in attention and MLP) with Neuron parallel layers (RowParallelLinear and ColumnParallelLinear).
        - Replace embeddings with Neuron parallel embeddings (ParallelEmbedding)
        - Replace any other modules that require Neuron-specific implementations.

### 4. Define an application/task head

Define an application/task head. Applications includes causal LM, classification, and so on. This class extends a task-specific Neuron application head class (such as NeuronBaseForCausalLM), or the general NeuronApplicationHead class.

In this class, you provide an value for _model_cls which is the Neuron model class you defined.

You can also override any other functions as needed for your model, such as get_compiler_args(self) or convert_hf_to_neuron_state_dict(model_state_dict, neuron_config).

Note: This example demonstrates a simplified version of NeuronLlamaForCausalLM from the NxD Inference model hub.

```python
class NeuronLlamaForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronLlamaModel

    @classmethod
    def get_config_cls(cls):
        return LlamaInferenceConfig
```

# High Level Workflow

Translating a model will involve the following phases. Some phases are more involved and contain details in the linked documentation - load resources as needed during development.

**Note**: Activate the environment before running any code `source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`

**Vision/multimodal routing:** If the target model accepts image inputs (for example `pixel_values`, aspect ratios, image chunks, vision masks), load and follow [reference/vlm_translation.md](reference/vlm_translation.md) first, then apply the phases below.

## Phase 1: Model Understanding and Planning

The orchestrator must **not** read source files or plan directly. It dispatches a single `plan` subagent that performs source exploration and returns the full Phase 2 execution plan.

### Step 1 — Dispatch a plan agent for exploration + work plan

Launch a `plan` subagent (thoroughness: "very thorough") with a prompt that instructs it to return one self-contained plan covering:

1. **Source model architecture inventory.** Read the model's PyTorch source and HuggingFace config. Identify every major block type present: attention (MHA/GQA/MQA), MLP, MoE routing and expert layers, embedding tables, normalization layers, positional encodings (RoPE, ALiBi, etc.), and any custom ops. Include file paths and class names for each block.

2. **Reference NxDI model.** Identify which reference model best matches the target architecture and return its full file path:
   - **Text MoE**: [Qwen3 MoE](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/qwen3_moe/modeling_qwen3_moe.py)
   - **VLM (scatter integration)**: [Pixtral](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/pixtral/modeling_pixtral.py)
   - **VLM (cross-attention integration)**: [MLlama](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/mllama/modeling_mllama.py)
   - **VLM (chunked vision flow)**: [Llama4](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/llama4/modeling_llama4.py)
   - **Other (dense text)**: [Llama](/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/llama/modeling_llama.py)

3. **Neuron substitution map.** For each block type found, map it to the corresponding NxDI primitive (`NeuronAttentionBase`, `RowParallelLinear`/`ColumnParallelLinear`, `ParallelEmbedding`, etc.). Flag any blocks with no obvious NxDI equivalent.

4. **HuggingFace config attribute inventory.** List all fields from the model's `PretrainedConfig` that must be surfaced in `InferenceConfig.get_required_attributes()`. For each attribute, note whether it exists verbatim in `config.json` or is computed/renamed at Python object construction time (the latter must be handled in `add_derived_config`).

5. **Block partition.** Divide the model into independent translation units (one per Phase 2 subagent). Each unit must be self-contained (no shared mutable state). Typical partition:
   - Attention subagent (attention block + KV cache integration)
   - MLP/FFN subagent (dense feed-forward layers)
   - MoE subagent (router + expert dispatch + expert layers), if applicable
   - Embedding & normalization subagent
   - Positional encoding subagent, if non-standard

6. **Per-subagent instructions.** For each translation unit, specify:
   - The source PyTorch class(es) to translate (file path + class name)
   - The NxDI base class and primitives to use
   - Any flagged deviations or unsupported ops to watch for

7. **Integration contracts.** For each block, specify the exact input/output tensor shapes and dtypes it must satisfy so blocks compose correctly in Phase 3.

The orchestrator consumes this plan output directly to drive Phase 2.

---

## Phase 2: Block Translation and Unit Testing
*The orchestrator agent dispatches nxdi-block-translator agents to translate all block partitions identified in Phase 1 in parallel. Each agent operates independently on its assigned block.*

Before launching subagents, copy `scripts/block_testing_utils.py` into a `tests/` directory.

Each subagent prompt must be constructed directly from the plan created in Phase 1. For each translation unit, the orchestrator extracts the relevant section of the plan and passes it as the subagent's full context. Each subagent receives: the source PyTorch implementation of its block, the integration contract, and a reference to the relevant NxDI primitives.

**Important:** Make the execution workflow as clear as possible for each nxdi-block-translator subagent. Each subagent should spend minimal time planning, spending most of its time implementing and debugging. The orchestrator must give it access to all necessary information needed to perform translation.

**Each subagent must:**

1. **Implement the Neuron block class.** Subclass the appropriate NxDI base (`NeuronAttentionBase`, `NeuronBaseModel`, etc.). Replace all standard PyTorch layers with their Neuron parallel equivalents per the substitution map from Phase 1.

2. **Preserve the forward pass contract.** Ensure the translated block accepts and returns tensors matching the shapes and dtypes specified in the integration contract. Do not change semantics — only change the layer implementations.

3. **Write a unit test.** Instantiate both the original PyTorch block and the translated Neuron block with identical weights. Run a forward pass with identical inputs and assert numerical equivalence within an acceptable tolerance (typically `atol=1e-3` for BF16). **The unit test must use `test_block_correctness` from `scripts/block_testing_utils.py`.**

4. **Document deviations.** If the source block could not be translated exactly (e.g., an unsupported op), document the deviation, the workaround applied, and any expected numerical differences. Flag cases that may require a custom NKI kernel.

**Subagent deliverables:** a translated block class file and a passing unit test. Deviations go in inline comments only. Do NOT produce README, summary, status, or documentation files.

### Auditing Subagent Test Files (Anti-Cheat Check)

After each subagent returns (or while it is operating, if you can observe its workspace), **read the generated `test_block.py` and verify it is not cheating**. A subagent cheats when it defines or imports the PyTorch reference class from a file it wrote itself, rather than from the original source. This produces a circular test that always passes regardless of correctness.

**Check for these red flags:**

1. **Local `pytorch_block.py` exists** — If a `pytorch_block.py` file is present in the workspace, the subagent almost certainly wrote the reference class itself. Read it and confirm it is not a copy or paraphrase of any class in `nxdi_block.py`.
2. **Import from a local file** — The test imports `PyTorchBlock` (or any reference class) from a file inside the workspace directory (e.g. `from pytorch_block import ...`). The import must point to the original source path outside the workspace.
3. **Reference class shares code with NxDI block** — The reference class re-uses helpers, constants, or logic defined in `nxdi_block.py`.

**If any red flag is detected, relaunch the subagent** with an explicit instruction prepended to its prompt:

> "IMPORTANT: Your previous test was rejected because it imported the PyTorch reference class from a file you wrote yourself. You MUST import `[ClassName]` directly from its original source at `[original_source_path]`. Do NOT create a `pytorch_block.py` file. Do NOT copy or rewrite the reference class."

Re-run the audit after the subagent returns again. Only accept a result once the test file imports the reference class from the unmodified original source.

---

## Phase 3: Scaffolding and Integration
*See [scaffolding & integration guide](reference/scaffolding_integration.md) for detailed API usage.*

The orchestrating agent collects all subagent deliverables and assembles the complete model. This phase is sequential.

1. **Define `NeuronConfig` and `InferenceConfig`.** Implement both config classes using the attribute inventory from Phase 1. Wire `get_neuron_config_cls` to return the `NeuronConfig` subclass.

2. **Assemble `NeuronBaseModel`.** Implement `setup_attr_for_model` and `init_model` using the translated block classes from Phase 2. Ensure all required attributes (`tp_degree`, `hidden_size`, `buckets`, etc.) are set correctly from config.

3. **Define the application head.** Subclass the appropriate task head (e.g., `NeuronBaseForCausalLM`). Set `_model_cls` and wire `get_config_cls`. Leave `convert_hf_to_neuron_state_dict` as a pass-through placeholder — it will be implemented in Phase 4:

    ```python
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        return state_dict  # placeholder — implemented in Phase 4
    ```

4. **Resolve any deviations flagged in Phase 2.** If subagents reported blocks requiring NKI kernels or workarounds, address them now and re-run affected unit tests.

---

## Phase 4: Weight Mapping
*This phase requires the assembled Neuron model from Phase 3. See [weight mapping guide](reference/weight_mapping.md) for detailed instructions.*

NxDI models load weights through `convert_hf_to_neuron_state_dict`. Now that the Neuron model is fully assembled, its `state_dict()` can be inspected directly to drive the key mapping. Dispatch a general-purpose agent to implement the weight mapping.

1. **Diff the state dict keys.** Instantiate the Neuron model on CPU (no compilation). Load the HF checkpoint. Print keys present in one but not the other to find every rename, fusion, and missing metadata tensor that the conversion function must produce.

2. **Implement `convert_hf_to_neuron_state_dict`.** Replace the placeholder with the real implementation. For each discrepancy found in the diff: rename keys, fuse weights (e.g. Q/K/V → Wqkv), apply any required transformations (transpose, scale fusion), and inject rank metadata tensors. Do not manually shard weights — the framework handles sharding at load time.

3. **Validate conversion.** Assert that the converted state dict contains no missing keys relative to the Neuron model and that all tensor shapes match. Then load the weights into the Neuron model and verify forward-pass numerical equivalence against the original HF model (pre-compilation, on CPU) within tolerance.

**Subagent deliverables:** the implemented conversion function replacing the placeholder, and a passing validation script.
