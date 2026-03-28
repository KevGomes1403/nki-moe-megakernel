# MoE Fused TKG Kernels

Fused NKI kernels for the Qwen3-30B-A3B MoE block (TP=4, LNC=2 on Trainium 2).

Each kernel implements: **RMSNorm → Router → TopK(8) → Selective Expert MLPs** in a single
NKI kernel, replacing the multi-kernel dispatch path.

## Kernel Versions

| Version | Strategy | E2E compatible? |
|---------|----------|-----------------|
| v1a–v8a | Various H-shard + sendrecv between cores | No — see compiler bug below |
| v9a | I-dimension sharding + shared-HBM all-reduce via `core_barrier` | No — see compiler bug below |
| v9b | Full gate/up replication per core, H-shard down, no inter-core communication | **Yes** |
| v10a | — | — |

### Compiler Bug: v9a and Earlier Cannot Be Used in an E2E Model

The Neuron compiler has a bug where **`core_barrier` and `sendrecv` cannot appear more than
once in the same XLA/HLO graph**. Since an E2E model has many MoE layers, any kernel that
uses either primitive (all versions up to and including v9a) will fail to compile when
integrated into the full model.

- v1a–v8a use `sendrecv` for H-shard partial results across the two LNC cores.
- v9a replaces `sendrecv` with a shared-HBM all-reduce gated by `core_barrier` — but
  `core_barrier` triggers the same compiler limitation.

### v9b: Avoiding the Bug via Gate/Up Replication

v9b eliminates all inter-core synchronization by having each core independently load the
**full H dimension** of the gate and up weights (instead of an H-shard). Each core then
produces its own H-shard of the output without any communication.

Trade-off: gate/up DMA and matmul double relative to v8a (full H instead of H/2), but down
projection DMA and matmul are unchanged. No `sendrecv`, no `core_barrier` — zero
in-kernel synchronization.

## Shape Assumptions (TP=4, LNC=2)

- Input hidden dim: `H = 2048`
- Experts: `E = 128`, top-K: `K = 8`
- Intermediate dim (per TP rank): `I_tp = 192`, padded to `I_padded = 256` (128-aligned)
- LNC cores: 2, each owning `H_shard = 1024` output columns
- Bucket sizes must be 128-aligned; pad inputs in the forward pass, not in the kernel.

## Verification and Profiling

`test_correctness.py` runs numerical correctness checks against a pure PyTorch CPU
reference and can also profile kernels on-device.

```bash
# Correctness check against default kernel.py
python test_correctness.py

# Correctness check for a specific kernel
python test_correctness.py kernel_v9b.py

# Benchmark with default warmup/iters
python test_correctness.py kernel_v9b.py --benchmark

# Benchmark with custom settings
python test_correctness.py kernel_v9b.py --benchmark --warmup 5 --iters 50
```

Profiling output (device profile JSON) is written to `output/`. The benchmark prints
MFU, MBU, HBM bytes read/written, and engine active times.

The kernel under test must expose a `run(inp, gamma, router_w, gate_up_w, down_w)`
function. Fixed inputs: `B=1, S=1, H=2048, E=128, K=8, I_padded=256`.
