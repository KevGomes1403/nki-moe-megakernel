# AWS Trainium2/3 MoE Kernel Challenge

**MLSys 2026 Competition Track**

Participants will write custom kernels with the Neuron Kernel Interface (NKI) for the Qwen3-30B-A3B Mixture of Experts model and optimize inference performance on AWS Trainium2/3 hardware.

For full details on the competition, read [the competition guidelines](https://github.com/aws-neuron/nki-moe/blob/main/CONTEST.md). 

To register your team, [enter your information here](https://docs.google.com/forms/d/e/1FAIpQLSeWuJ9h9F0-aC5OwhKcIKgzUB8Sc3DFdBNEgzxzHfM4QsajcA/viewform?usp=sharing&ouid=108119140038382966223&resourcekey=0-VVlo6GUSizIcln6HhBFvKQ) (just one entry per team).

## Getting Started

To learn NKI, follow [the official NKI guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html) and various example NKI kernels from the [nki-samples repository](https://github.com/aws-neuron/nki-samples). Another tool to help with optimizing NKI kernels is [NKI autotune](https://github.com/awslabs/nki-autotune).

## Setup Steps

1. Create a Trainium2 instance with AWS Neuron SDK v2.27 using EC2 based on the [setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/neuron-setup/multiframework/multi-framework-ubuntu24-neuron-dlami.html#setup-ubuntu24-multi-framework-dlami).
2. Activate the Neuron virtual environment to run inference by running the appropriate activation command for your SDK version:
   ```bash
   source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
   ```
3. Clone this repository and run `cd [PATH]/nki-moe` where `[PATH]` is the directory where you have performed the clone.
4. Download the Qwen3-30B-A3B model to a `~/qwen-30b-a3b/hf_model` folder in your root directory. We recommend doing so using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli). You can install this by running `pip3 install huggingface_hub[cli]`.
5. To run inference, navigate to `[PATH]/nki-moe` and run:
   ```bash
   python3 main.py --mode generate --model-path ~/qwen-30b-a3b/hf_model --compiled-model-path ~/qwen-30b-a3b/traced_model --prompt "What is the capital of France?"
   ```

## NKI Kernel Development

This repository contains the standard model implementation in `qwen.py`.

Your task is to identify parts of the model (operators, fused operators, layers, or even the whole model) that can be implemented as NKI kernels and add them to create optimized versions of the model.

### Sample NKI Kernels

This repository includes two NKI kernel examples to help you get started:

#### 1. Tensor Add Example (`nki_tensor_add_example.py`)

A simple NKI kernel demonstrating basic tensor operations. This serves as a minimal reference implementation showing:
- Basic NKI kernel structure with `@nki.jit` decorator
- Tensor indexing and loading from HBM to SBUF
- Element-wise operations
- Storing results back to HBM

This example is not integrated into the model but provides a foundation for understanding NKI kernel development.

#### 2. RMSNorm Kernel (`nki_custom_rmsnorm.py`)

A production-ready NKI RMSNorm implementation integrated into the Qwen model. This kernel follows the pattern from the [official AWS NKI RMSNorm tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/tutorials/rmsnorm.html).


We also have `qwen_with_nki.py` which has model implementation with custom NKI kernels integrated. To test the different implementations:

```bash
# Standard inference (uses qwen.py)
python3 main.py --mode generate --model-path ~/qwen-30b-a3b/hf_model --compiled-model-path ~/qwen-30b-a3b/traced_model --prompt "What is the capital of France?"

# With NKI RMSNorm kernel (uses qwen_with_nki.py)
python3 main.py --mode generate --enable-nki --model-path ~/qwen-30b-a3b/hf_model --compiled-model-path ~/qwen-30b-a3b/traced_model --prompt "What is the capital of France?"
```

**Important:** When switching between NKI and standard modes, remove the traced model directory and compile cache to ensure proper recompilation:
```bash
rm -rf ~/qwen-30b-a3b/traced_model
rm -rf /var/tmp/neuron-compile-cache/*
```

The `--enable-nki` flag in `main.py` controls which model file is loaded:
- Without flag: loads `qwen.py` (standard implementation)
- With flag: loads `qwen_with_nki.py` (NKI-accelerated implementation)

Key areas to focus on:
* MoE routing and expert selection logic
* Expert computation (gate_proj, up_proj, down_proj)
* Attention mechanisms with MoE-specific optimizations
* Memory-efficient tensor operations for sparse expert execution

## Additional Tools

1. **Profiling:** If you would like to profile your implementation in order to get a better understanding of performance bottlenecks and opportunities for optimization, you can use the [Neuron Explorer](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/index.html).
2. **Benchmarking:** You can also leverage the [NKI benchmarking API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.benchmark.html) to retrieve execution latency statistics.

## Contact

**Email**: [nki-mlsys-2026@amazon.com](mailto:nki-mlsys-2026@amazon.com)
