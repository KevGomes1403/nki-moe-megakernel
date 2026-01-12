# NKI MoE Challenge

## Challenge Overview

This repository provides a package containing the PyTorch model of Qwen3-30B-A3B, a state-of-the-art Mixture of Experts (MoE) architecture. This model **can be compiled with AWS Neuron SDK and run on** a **Trainium3 instance.** The main file in this package is `qwen.py` which contains the model implementation in PyTorch.

Participants will develop custom kernels using the Neuron Kernel Interface (NKI) to optimize inference for the Qwen3-30B-A3B model, which features 128 total experts with 8 active experts per token.

## Core Task

Write NKI kernels that improve upon the baseline implementation by optimizing areas among the following:

### MoE-Specific Kernels
* **Expert routing kernel:** Token-to-expert assignment with efficient gating computation
* **Sparse expert execution:** Optimized matrix multiplications that exploit 8/128 sparsity
* **Expert aggregation kernel:** Combining outputs from active experts with learned weights
* **Dynamic batching kernel:** Grouping tokens by expert assignment for efficient execution

### Memory Management Kernels
* **Expert weight loading:** Efficient on-chip caching strategies for 128 expert parameters
* **Activation management:** Optimized tensor layouts for token routing and expert computation
* **KV-cache optimization:** Memory-efficient attention mechanisms for MoE decoder layers

### Attention and FFN Kernels
* **Flash attention variants:** Optimized attention for MoE architectures
* **Fused operations:** Combining multiple operations to reduce memory traffic
* **Layer normalization:** Efficient normalization kernels for MoE layers

## Technical Environment

### Hardware
* **Platform:** AWS Trainium3 (trn3.3xlarge instances)
* **Compute:** 4 Logical NeuronCores per instance
* **Memory:** 144GB HBM memory capacity
* **Architecture:** NeuronCore architecture with tensor, vector, and scalar engines

### Model
* **Architecture:** Qwen3-30B-A3B
* **Parameters:** 30B total parameters
* **Experts:** 128 experts with 8 active per token
* **Type:** Mixture of Experts architecture

### Software
* **SDK:** AWS Neuron SDK 2.27 (released in December 2025), with extra points to upgrade to the latest SDK
* **Programming Interface:** Neuron Kernel Interface (NKI) with Python
  * Direct access to NeuronCore hardware primitives
  * Support for custom memory layouts and data movement
* **Framework:** Software support for the specified MoE model with a TP=4 baseline provided

### Baseline
Reference implementation with standard PyTorch operators compiled through Neuron compiler.

**Key requirement:** All performance-critical operations must be implemented as custom NKI kernels. Participants cannot rely solely on compiler optimizations—they must write low-level NKI code to achieve competitive performance.

## Getting Started

To learn NKI, follow [the official NKI guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html) and various example NKI kernels from the [nki-samples repository](https://github.com/aws-neuron/nki-samples). Another tool to help with optimizing NKI kernels is [NKI autotune](https://github.com/awslabs/nki-autotune).

## Setup Steps

1. Create a Trainium3 instance with AWS Neuron SDK v2.28 using EC2 based on the [setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/neuron-setup/multiframework/multi-framework-ubuntu24-neuron-dlami.html#setup-ubuntu24-multi-framework-dlami).
2. Activate the Neuron virtual environment to run inference by running the appropriate activation command for your SDK version. `source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`
3. Clone this repository and run `cd [PATH]/nki-moe` where `[PATH]` is the directory where you have performed the clone.
4. Download the Qwen3-30B-A3B model to a `~/qwen-30b-a3b/hf_model` folder in your root directory. We recommend doing so using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli). You can install this by running `pip3 install huggingface_hub[cli]`. You will also need to create an [access token](https://huggingface.co/docs/hub/en/security-tokens).
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

## Submission

Your submission should be a single Python file called `qwen.py`. This file should contain implementations of NKI kernels and also modifications to the original model to invoke these NKI kernels. This file should work as a plug-in replacement for the original `qwen.py` of the reference PyTorch implementation provided in this repository.

Submission link will be provided closer to the contest date.

## Benchmarks

Submissions will be tested using 25 benchmarks (prompts) with varying context lengths and batch sizes. We have provided prompts in `prompts.txt` with their corresponding metadata (prompt ID, prompt length, recommended sequence length, and baseline latency/throughput) in `prompt_data.txt`. There are 2 methods of testing these prompts:

1. To avoid recompilation per prompt, you can use a global sequence length (we suggest 1024) for all prompts. Run `python main.py --enable-nki --mode evaluate_all --seq-len 1024`.
2. Alternatively, you can use a unique sequence length for each prompt (suggested sequence lengths are the third entry in each row of `prompt_data.txt`) at the cost of recompiling the model for each prompt. Run `python test.py` to evaluate these prompts in this fashion.

The remaining 20 prompts will be withheld for evaluation. All benchmarks will become publicly available after the contest is complete.

## Evaluation and Scoring

The contest organizers will execute each team's submission across the twenty withheld benchmarks on a dedicated Trainium3 instance. The submissions will be evaluated on:

1) **Accuracy** of generated output vs. our reference implementation. Accuracy evaluation will be a binary assessor: Any benchmark that fails an accuracy threshold will result in a score of 0.
2) **Latency** (Time to first token (TTFT))
3) **Throughput** measured as output tokens / second
4) **Amount of model written in NKI** (measured as NKI FLOPS / total model FLOPS) (will be applied as a scaling factor for (b) and (c)). Note: NKI FLOPs measures the number of multiply-accumulate (MAC) operations.

Rankings will be established by calculating the total normalized number of points per team, where points are normalized against the baseline.

We define **points** as **Accuracy** (binary) **\* Reduced Latency \* Increased Throughput \* (1 + Normalized NKI FLOPS)**, where:

* **Accuracy** = 1 if accuracy matches or exceeds a predetermined threshold, 0 otherwise
* **Reduced Latency** = Reference implementation TTFT divided by submission TTFT
* **Increased Throughput** = Submission tokens/sec divided by reference implementation tokens/sec
* **Normalized NKI FLOPS** = Submission NKI FLOPS divided by total model FLOPS

For example, a submission that is sufficiently accurate, with 10x reduced latency, 2x increased throughput, and 0.85 normalized NKI FLOPS would obtain 1 \* 10 \* 2 \* 1.85 \= 37 points. For reference, the baseline submission would receive a score of 1.

## Presentations

Teams who successfully submit an entry will be invited to present an informal overview of their approach (roughly 10 to 15 minutes) at a special session held during the Workshop & Tutorial days. Winners will be announced later in the week, with full results being released soon after the conference.

## Contest Eligibility

All are welcome to participate in the contest (including teams from academia, industry, and elsewhere) with the exception of the Contest Organizers and employees of the Contest Sponsor. Individuals are prohibited from participating in multiple teams. In order to be eligible for prizes, teams must commit to releasing an open-source version of their implementation prior to the next conference.

## Frequently Asked Questions

To raise a question, please create an issue in this repository, or feel free to reach out to the contest organizers directly.

## Related Work

* TBD

## Contest Organizers

* Emery Berger (Amazon Web Services), [emerydb@amazon.com](mailto:emerydb@amazon.com)
* Aninda Manocha (Amazon Web Services)
* Wei Tang (Amazon Web Services)
* Emily Webber (Amazon Web Services)
* Ziyang Xu (Amazon Web Services)
