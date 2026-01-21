# AWS Trainium2/3 MoE Kernel Challenge

**MLSys 2026 Competition Track**

Participants will write custom kernels with the Neuron Kernel Interface (NKI) for the Qwen3-30B-A3B Mixture of Experts model and optimize inference performance on AWS Trainium2/3 hardware.

For full details on the competition, read [the competition guidelines](https://github.com/aws-neuron/nki-moe/blob/main/CONTEST.md). Register for the competition [here](https://docs.google.com/forms/d/e/1FAIpQLSeWuJ9h9F0-aC5OwhKcIKgzUB8Sc3DFdBNEgzxzHfM4QsajcA/viewform?resourcekey=0-VVlo6GUSizIcln6HhBFvKQ).

## Getting Started

To learn NKI, follow [the official NKI guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html) and various example NKI kernels from the [nki-samples repository](https://github.com/aws-neuron/nki-samples). Another tool to help with optimizing NKI kernels is [NKI autotune](https://github.com/awslabs/nki-autotune).

## Setup Steps

1. Create a Trainium2 instance with AWS Neuron SDK v2.27 using EC2 based on the [setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/neuron-setup/multiframework/multi-framework-ubuntu24-neuron-dlami.html#setup-ubuntu24-multi-framework-dlami).
2. Activate the Neuron virtual environment to run inference by running the appropriate activation command for your SDK version. `source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`
3. Clone this repository and run `cd [PATH]/nki-moe` where `[PATH]` is the directory where you have performed the clone.
4. Download the Qwen3-30B-A3B model to a `~/qwen-30b-a3b/hf_model` folder in your root directory. We recommend doing so using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli). You can install this by running `pip3 install huggingface_hub[cli]`.
5. To run inference, navigate to `[PATH]/nki-moe` and run `python3 main.py --mode generate`.

## NKI Kernel Development

The `qwen.py` file contains the model implementation where you can add your custom NKI kernels. Your task is to identify parts of the model (operators, fused operators, layers, or even the whole model) that can be implemented as NKI kernels and replace them in the original model to achieve better performance.

Key areas to focus on:
* MoE routing and expert selection logic
* Expert computation (gate_proj, up_proj, down_proj)
* Attention mechanisms with MoE-specific optimizations
* Memory-efficient tensor operations for sparse expert execution

## Additional Tools

1. **Profiling:** If you would like to profile your implementation in order to get a better understanding of performance bottlenecks and opportunities for optimization, you can use the [Neuron Profiler](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html).
2. **Benchmarking:** You can also leverage the [NKI benchmarking API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.benchmark.html) to retrieve execution latency statistics.

## Contact

**Email**: [nki-mlsys-2026@amazon.com](mailto:nki-mlsys-2026@amazon.com)
