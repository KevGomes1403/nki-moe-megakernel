# Phase 3: Scaffolding & Integration

Assemble the translated block classes from Phase 2 into a single model file. The output is a complete, importable Python module — minus the weight conversion function, which is a placeholder until Phase 4.

---

## Inputs

- Translated block classes and their unit tests from Phase 2 subagents (e.g., `NeuronMyModelAttention`, `NeuronMyModelMLP`, `NeuronMyModelDecoderLayer`)
- Integration contracts from Phase 1 (tensor shapes, dtypes, layer counts)
- The source model's HuggingFace `PretrainedConfig` attribute list

---

## 1. Imports

```python
import torch
import torch.nn as nn
from typing import List, Type

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel, NeuronBaseForCausalLM
```

---

## 2. NeuronConfig Subclass

Only create a subclass if the model needs custom Neuron-specific parameters (e.g., a non-default attention class, MoE routing config). Otherwise use `NeuronConfig` directly and skip this class.

```python
class NeuronMyModelConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add model-specific Neuron parameters here, e.g.:
        # self.attn_cls = NeuronMyModelAttention
```

---

## 3. InferenceConfig Subclass

Surfaces the HuggingFace model attributes needed at runtime and wires the NeuronConfig class.

```python
class MyModelInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        # All PretrainedConfig fields the model reads during forward pass
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            # ... add model-specific fields
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronMyModelConfig  # or NeuronConfig if no subclass was needed
```

Override `add_derived_config(self)` only if the model needs values computed from other config fields (e.g., `num_cores_per_group` for flash decoding).

---

## 4. NeuronBaseModel Subclass

Wires the translated block classes into a complete model module.

### `setup_attr_for_model`

Must set all of the following before `init_model` runs:

```python
def setup_attr_for_model(self, config: InferenceConfig):
    self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
    self.tp_degree = config.neuron_config.tp_degree
    self.hidden_size = config.hidden_size
    self.num_attention_heads = config.num_attention_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.max_batch_size = config.neuron_config.max_batch_size
    self.buckets = config.neuron_config.buckets
```

### `init_model`

Branch on whether tensor-parallel state is active. Use Neuron parallel layers in the parallel branch; fall back to standard PyTorch layers for CPU testing.

```python
def init_model(self, config: InferenceConfig):
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    if parallel_state.model_parallel_is_initialized():
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
            bias=False,
        )
    else:
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    self.layers = nn.ModuleList(
        [NeuronMyModelDecoderLayer(config) for _ in range(config.num_hidden_layers)]
    )
    self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

Replace `NeuronMyModelDecoderLayer` and `nn.RMSNorm` with the actual translated classes from Phase 2.

---

## 5. Application Head

```python
class NeuronMyModelForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronMyModelModel

    @classmethod
    def get_config_cls(cls):
        return MyModelInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        return state_dict  # placeholder — implemented in Phase 4
```

---

## 6. File Layout

Place the assembled file at:

```
<output_dir>/modeling_<modelname>_neuron.py
```

It should contain, in order:
1. Imports
2. `NeuronMyModelConfig` (if needed)
3. `MyModelInferenceConfig`
4. Translated block classes from Phase 2 (or import them if kept in separate files)
5. `NeuronMyModelModel`
6. `NeuronMyModelForCausalLM`
