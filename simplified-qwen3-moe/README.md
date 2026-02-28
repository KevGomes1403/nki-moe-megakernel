This repository contains materials and scripts for running experiments with the Qwen3-30B-A3B MoE model, including model download, preprocessing, and simplification.

## Setup

1. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

2. **Download the Model**

    Download the [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) model from Hugging Face and place it in the correct directory:
    ```sh
    mkdir -p Qwen/Qwen3-30B-A3B
    hf download Qwen/Qwen3-30B-A3B --local-dir Qwen/Qwen3-30B-A3B
    ```

3. **Run Model**

    Run the initial model script:
    ```sh
    python 01_run.py
    ```

    Check whether the output is generated and the results are reasonable.

2. **Carve Out Model Components**

    This step extracts and saves model submodules:
    ```sh
    python 02_carve.py
    ```

    If successful, you should see files named like:
    ```
    model.layers.16.self_attn.pt
    model.layers.19.mlp.experts.act_fn.pt
    model.layers.2.pt
    model.layers.25.mlp.experts.pt
    model.layers.30.mlp.gate.pt
    model.layers.31.mlp.pt
    model.layers.32.input_layernorm.pt
    model.pt
    model.rotary_emb.pt
    ```
## Model Analysis

- The [03_live_model_analysis.ipynb](03_live_model_analysis.ipynb) notebook guides you through interactive, manual model analysis.  
- Insights from this analysis were used in the development of [04_qwen3_moe_functional.py](04_qwen3_moe_functional.py).
