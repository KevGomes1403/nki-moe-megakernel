# AWS Trainium2/3 MoE Kernel Challenge

**MLSys 2026 Competition Track**

Participants will write custom kernels with the Neuron Kernel Interface (NKI) for the Qwen3-30B-A3B Mixture of Experts model and optimize inference performance on AWS Trainium2/3 hardware.

[![Conference](https://img.shields.io/badge/MLSys-2026-blue)](https://mlsys.org/)
[![Location](https://img.shields.io/badge/Location-Bellevue%2C%20WA-green)]()
[![Dates](https://img.shields.io/badge/Dates-May%2017--22%2C%202026-orange)]()

---

## Table of Contents

- [Overview](#overview)
- [Challenge Description](#challenge-description)
  - [Core Task](#core-task)
  - [MoE-Specific Kernels](#moe-specific-kernels)
  - [Memory Management Kernels](#memory-management-kernels)
  - [Attention and FFN Kernels](#attention-and-ffn-kernels)
- [Technical Environment](#technical-environment)
- [Submission Requirements](#submission-requirements)
  - [Team Eligibility](#team-eligibility)
  - [Submission Format](#submission-format)
  - [Verification Criteria](#verification-criteria)
- [Prize Structure](#prize-structure)
  - [Awards](#awards)
  - [Evaluation Criteria](#evaluation-criteria)
- [Winner Deliverables](#winner-deliverables)
- [Competition Timeline](#competition-timeline)
- [Presentation at MLSys 2026](#presentation-at-mlsys-2026)
- [Open-Source Policy](#open-source-policy)
- [Support and Resources](#support-and-resources)
- [Judging Committee](#judging-committee)
- [Expected Impact](#expected-impact)
- [Organizers](#organizers)
- [Contact](#contact)

---

## Overview

We present an AWS Trainium competition track for MLSys 2026 (May 17-22, 2026, Bellevue, WA). Participants will develop custom kernels using the Neuron Kernel Interface (NKI) for the **Qwen3-30B-A3B Mixture of Experts** model and optimize inference performance on AWS Trainium3 hardware.

This competition builds on the successful ASPLOS 2025/EuroSys 2025 contest while focusing on the unique challenges of sparse MoE architectures.

---

## Challenge Description

### Core Task

Write NKI kernels that improve upon the baseline implementation by optimizing the following areas.

### MoE-Specific Kernels

| Kernel | Description |
|--------|-------------|
| **Expert Routing** | Token-to-expert assignment with efficient gating computation |
| **Sparse Expert Execution** | Optimized matrix multiplications that exploit 8/128 sparsity |
| **Expert Aggregation** | Combining outputs from active experts with learned weights |
| **Dynamic Batching** | Grouping tokens by expert assignment for efficient execution |

### Memory Management Kernels

| Kernel | Description |
|--------|-------------|
| **Expert Weight Loading** | Efficient on-chip caching strategies for 128 expert parameters |
| **Activation Management** | Optimized tensor layouts for token routing and expert computation |
| **KV-Cache Optimization** | Memory-efficient attention mechanisms for MoE decoder layers |

### Attention and FFN Kernels

| Kernel | Description |
|--------|-------------|
| **Flash Attention Variants** | Optimized attention for MoE architecture |
| **Fused Operations** | Combining multiple operations to reduce memory traffic |
| **Layer Normalization** | Efficient normalization kernels for MoE layers |

---

## Technical Environment

### Hardware: AWS Trainium2/3

### Model: Qwen3-30B-A3B

| Specification | Value |
|---------------|-------|
| Total Parameters | 30B |
| Total Experts | 128 |
| Active Experts per Token | 8 |
| Architecture | Mixture of Experts |

### Software Stack

- **SDK**: AWS Neuron SDK 2.28
- **Programming Interface**: Neuron Kernel Interface (NKI) with Python
  - Direct access to NeuronCore hardware primitives
  - Support for custom memory layouts and data movement
- **Framework**: TP=4 baseline provided
- **Baseline**: Reference implementation with standard PyTorch operators compiled through Neuron compiler

> **Key Requirement**: All performance-critical operations must be implemented as custom NKI kernels. Participants cannot rely solely on compiler optimizations—they must write low-level NKI code to achieve competitive performance.

---

## Submission Requirements

### Team Eligibility

- **Team Size**: 1-4 members per team
- **Maximum Teams**: 50 (first 50 to register)
- **Affiliation Restrictions**: Amazon employees, interns, and scholars in either 2025 or 2026 and their immediate family members are **ineligible**
- **Academic Integrity**: Submissions must comply with participants' institutional academic integrity policies

### Submission Format

1. **Python Implementation**
   - Uploaded Python file with your implementation of the model and kernels
   - NKI kernel implementations as Python modules containing custom NKI kernels for MoE operations
   - Integration code showing how NKI kernels replace baseline operators

2. **Technical Documentation** (4-6 pages, LaTeX, double-column format)
   - Description of NKI kernel designs and optimization strategies
   - Analysis of performance bottlenecks and how kernels address them
   - Tile size selection, memory layout decisions, and parallelization strategies
   - Performance benchmarking results with profiling data
   - Correctness validation against reference implementation

### Verification Criteria

All submissions must:

- [ ] Run successfully on provided AWS Trainium3 infrastructure
- [ ] Use NKI for at least 3 major parts of the model (e.g., routing, expert execution, aggregation)
- [ ] Demonstrate performance improvement over baseline
- [ ] Teams with the highest measurable performance impact will carry the most points

---

## Prize Structure

### Awards

| Place | Prize | Additional Benefits |
|-------|-------|---------------------|
| **First Place** | $25,000 USD | Guaranteed internship interviews |
| **Second Place** | $10,000 USD | Guaranteed internship interviews |
| **Third Place** | $5,000 USD | Guaranteed internship interviews |

> Members from the top three teams have **guaranteed interviews** for internships in the AWS Trainium team. Other participants will be considered based on strength of submission.

Prizes will be distributed equally among team members.

### Evaluation Criteria

#### Performance Metrics (85%)

- **Throughput**: Tokens/second across batch sizes
- **Latency**: Time-to-first-token, per-token latency
- **Memory Efficiency**: Bandwidth utilization
- **Hardware Utilization**: NeuronCore utilization and expert load balance
- **Token Accuracy**: Logits must exactly match CPU reference

#### NKI Kernel Innovation (15%)

- Novel kernel designs and optimization techniques
- Effective use of NKI programming model (tiling, memory management, parallelization)
- Creative solutions to MoE-specific challenges
- *Qualitative estimation by the judging team*

#### Tie Breakers

1. Use of most recent Neuron SDK versions
2. Clarity and documentation of NKI kernels
3. Reproducibility and ease of integration
4. Profiling analysis and performance insights

---

## Winner Deliverables

All prize winners must provide:

| Deliverable | Requirements |
|-------------|--------------|
| **Open-Source Code** | Complete NKI kernel implementations under Apache 2.0 license |
| **Technical Report** | 4-6 pages covering kernel designs, performance analysis, and lessons learned |
| **Presentation** | 15-minute talk at MLSys 2026 focusing on NKI kernel development |
| **Documentation** | Well-documented code, integration examples, usage instructions |
| **Reproducibility** | Scripts and profiling tools |
| **Benchmark Results** | Comprehensive performance analysis across configurations |

> **Deadline**: All deliverables must be completed by **May 3, 2026** (two weeks before conference)

---

## Competition Timeline

| Date | Milestone |
|------|-----------|
| December 30, 2025 | Competition announced |
| January 23, 2026 | Competition starter kit published |
| February 1, 2026 | Registration opens |
| March 1, 2026 | Registration closes |
| **April 7, 2026** | **Submission Deadline** (11:59 PM Pacific Time) |
| April 24, 2026 | Winner notification |
| May 3, 2026 | Final deliverables due |
| May 17-22, 2026 | Winner presentations at MLSys 2026 |

### Internal Milestones

| Date | Milestone |
|------|-----------|
| January 12, 2026 | Competition tracks ready to submit to MLSys PC |
| January 16, 2026 | Final day to make changes to competition tracks |
| January 19, 2026 | All company portals go live |
| February 16, 2026 | Competition kicks off |
| March 20, 2026 | Competition concludes |
| March 31, 2026 | Winners notified (for travel/visa planning) |

---

## Presentation at MLSys 2026

Three (3) winning teams will present their solutions:

- Each team receives a **15-minute presentation slot** during a dedicated competition track session
- All team members are invited to attend
- Presentations will be recorded and made publicly available

---

## Open-Source Policy

### Public Communication

- All competition materials, rules, and updates published on official MLSys 2026 website
- Competition repository hosted on GitHub ([aws-neuron](https://github.com/aws-neuron) organization)
- Regular updates via MLSys mailing list and social media channels
- Unified branding across all multi-company competition tracks
- Coordinated press releases with other competition track organizers

### Open-Source Requirements

| Requirement | Details |
|-------------|---------|
| **Winners** | Full open-source release of NKI kernel implementations (Apache 2.0) prior to conference |
| **Baseline Code** | All provided baseline implementations and evaluation frameworks are open-source |
| **Documentation** | Comprehensive NKI tutorials and MoE optimization guides in competition repository |
| **Community Engagement** | Winners encouraged to engage through blog posts, tutorials, and workshops |

---

## Support and Resources

### Documentation

- Comprehensive NKI programming tutorials and API reference
- MoE optimization guides with examples
- Qwen3-30B-A3B model architecture documentation
- NeuronCore hardware architecture guide

### Compute Credits

> **Academic teams** will be provided **$250 in AWS credits** after registration for development and testing on `trn3.3xlarge` instances.

### Starter Kit

- Reference Qwen3-30B-A3B implementation with Neuron software support (TP=4)
- Baseline NKI kernel examples (attention, matrix multiplication)
- Profiling tools and performance analysis scripts
- Sample MoE kernel implementations with optimization opportunities

---

## Judging Committee

| Role | Count |
|------|-------|
| AWS Neuron team members (engineers, kernel developers, scientists, solution architects) | 5 |
| External ML systems researcher with MoE and kernel optimization expertise | 1 |
| MLSys 2026 organizing committee member | 1 |

---

## Expected Impact

This competition will:

- **Advance state-of-the-art** in NKI kernel programming for MoE architectures
- **Demonstrate practical applications** of low-level kernel optimization on specialized hardware
- **Build community expertise** in AWS Trainium development and NKI programming
- **Produce open-source NKI kernels** benefiting the broader ML systems community
- **Foster collaboration** between academia and industry on MoE systems challenges
- **Contribute to knowledge** on efficient inference for sparse models
- **Showcase hardware-aware kernel design** for modern LLM architectures

---

## Organizers

**Organization**: AWS Trainium

**Organizer Team**:
- Emery Berger
- Emily Webber
- Ziyang Xu
- Aninda Manocha
- Wei Tang
- Armin Agha-Ebrahim

**Previous Experience**: Successfully organized ASPLOS 2025/EuroSys 2025 Contest on NKI-based Llama 3.2 1B optimization ([aws-neuron/nki-llama](https://github.com/aws-neuron/nki-llama) on GitHub)

---

## Contact

**Email**: [nki-mlsys-2026@amazon.com](mailto:nki-mlsys-2026@amazon.com)

---

## Conclusion

The AWS Trainium3 MoE Kernel Optimization Challenge offers students a unique opportunity to write custom NKI kernels for a cutting-edge sparse model architecture on state-of-the-art ML acceleration hardware.

The **Qwen3-30B-A3B** model, with its 128-expert architecture and 8-active-per-token sparsity pattern, presents compelling optimization challenges that require deep understanding of both the NKI programming model and MoE workload characteristics.

By requiring participants to implement custom kernels rather than relying on compiler optimizations, this competition emphasizes hands-on systems programming skills that are increasingly valuable in the ML systems community.

**We look forward to engaging students in this exciting challenge and seeing innovative NKI kernel designs that push the boundaries of MoE inference performance.**

---

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

---

<p align="center">
  <i>MLSys 2026 Competition Track | May 17-22, 2026 | Bellevue, WA</i>
</p>
