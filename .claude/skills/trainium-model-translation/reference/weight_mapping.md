# Weight Mapping Guide

## Goal

Implement `convert_hf_to_neuron_state_dict` in the application head class. This function
receives the HuggingFace state dict and must return a state dict whose keys and tensor
shapes match the Neuron model's `state_dict()`.

## Step 1: Diff the keys

You need the key set from each side without loading full model weights.

**HF keys — read the safetensors index instead of loading the model:**

For large models, `AutoModelForCausalLM.from_pretrained` loads all weights into CPU RAM
(tens of GB). Read the index file instead — it maps every key to its shard without
loading any tensors:

```python
import json

with open(f"{model_path}/model.safetensors.index.json") as f:
    hf_keys = set(json.load(f)["weight_map"].keys())
```

If the model is small enough to load, `AutoModelForCausalLM.from_pretrained` works too.

**Neuron model keys — instantiate with a minimal 1-layer config:**

Config construction requires both a `NeuronConfig` subclass and an `InferenceConfig`
subclass. Populate `InferenceConfig` from the HF `config.json` and override
`num_hidden_layers=1` — you only need the key structure, which is identical across
layers:

```python
from transformers import AutoConfig
from my_model import MyNeuronConfig, MyInferenceConfig, MyNeuronModelForCausalLM

hf_cfg = AutoConfig.from_pretrained(model_path)

neuron_cfg = MyNeuronConfig(tp_degree=1, torch_dtype="bfloat16")
inference_cfg = MyInferenceConfig(hf_cfg, neuron_config=neuron_cfg)
inference_cfg.num_hidden_layers = 1  # key structure is layer-invariant

neuron_model = MyNeuronModelForCausalLM(inference_cfg)
neuron_keys = set(neuron_model.state_dict().keys())
```

**Print the diff:**

```python
# Strip layer indices to find the structural pattern (keys repeat per layer)
import re
def strip_layer(k): return re.sub(r"layers\.\d+\.", "layers.N.", k)

hf_patterns = {strip_layer(k) for k in hf_keys}
neuron_patterns = {strip_layer(k) for k in neuron_keys}

print("In HF, not Neuron:", sorted(hf_patterns - neuron_patterns))
print("In Neuron, not HF:", sorted(neuron_patterns - hf_patterns))
```

Every pattern in `neuron_patterns - hf_patterns` must be produced by the conversion
function. Every pattern in `hf_patterns - neuron_patterns` must either be renamed or
intentionally dropped.

## Step 2: Categorise each discrepancy

| Category | What to do |
|---|---|
| **Key rename** | `state_dict[new_key] = state_dict.pop(old_key)` |
| **Weight fusion** | Concatenate multiple source tensors into one target (e.g. Q/K/V → Wqkv) |
| **Weight transformation** | Apply a mathematical op (transpose, scale fusion) before assigning |
| **Rank metadata** | Inject `torch.arange(0, tp_degree, dtype=torch.int32)` tensors required by parallel layers |
| **Unneeded HF key** | Leave it — the framework ignores extra keys during load |

## Step 3: Implement the function

```python
@staticmethod
def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
    neuron_config = config.neuron_config
    tp_degree = neuron_config.tp_degree
    num_layers = config.num_hidden_layers

    for i in range(num_layers):
        # --- rank metadata (always required) ---
        state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
            0, tp_degree, dtype=torch.int32
        )

        # --- key renames (example) ---
        state_dict[f"layers.{i}.self_attn.q_layernorm.weight"] = state_dict.pop(
            f"layers.{i}.self_attn.q_norm.weight"
        )

        # --- weight fusion (example: fuse Q/K/V) ---
        state_dict[f"layers.{i}.self_attn.Wqkv.weight"] = torch.cat([
            state_dict.pop(f"layers.{i}.self_attn.q_proj.weight"),
            state_dict.pop(f"layers.{i}.self_attn.k_proj.weight"),
            state_dict.pop(f"layers.{i}.self_attn.v_proj.weight"),
        ], dim=0)

    # rank metadata for base model and embedding
    state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
    if neuron_config.vocab_parallel:
        state_dict["embed_tokens.rank_util.rank"] = torch.arange(
            0, neuron_config.local_ranks_size, dtype=torch.int32
        )

    return state_dict
```

Include only the transformations your diff identified. Remove the examples that do not
apply to the model being translated.

## Step 4: Validate

```python
converted = MyNeuronModelForCausalLM.convert_hf_to_neuron_state_dict(
    hf_model.state_dict(), config
)

missing = neuron_keys - set(converted.keys())
assert not missing, f"Missing keys: {missing}"

# Spot-check shapes
for key in neuron_keys:
    if key in converted:
        expected = neuron_model.state_dict()[key].shape
        actual = converted[key].shape
        assert expected == actual, f"{key}: expected {expected}, got {actual}"
```

## Notes

- **Sharding is handled by the framework.** `ColumnParallelLinear` and `RowParallelLinear`
  slice weights at load time. Do not manually shard tensors in the conversion function.
- **Preserve dtype and device.** Use `.detach().clone()` when creating derived tensors
  from existing ones to avoid aliasing.
- **MoE models** require concatenating per-expert weight tensors into a single stacked
  tensor and may also require transposing (`.T`) to match the expected layout.
- **Tied weights** (e.g. input and output embeddings) are handled by
  `update_state_dict_for_tied_weights()` which runs after this function. Do not
  duplicate tied weights here.
