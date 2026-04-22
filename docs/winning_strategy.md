# Winning Plan: AWS Trainium3 MoE Kernel Competition

Updated 2026-04-17.

---

## Where you stand today

| Metric | Current | Round 1 | Gap |
|---|---|---|---|
| TKG/token | 8.76 ms | 8.47 ms | **Regression** (-3.3%) |
| E2E | 5933 ms | 5745 ms | — |

**The multilayer kernel is slightly worse than round-1's split-kernel approach.** That's the first fire to put out.

Scoring (from `main.py` lines 637-652): `Score = Accuracy × Speedup² × NKI_FLOP_Ratio`. **Speedup is quadratic**, so +15% speed → +32% score. That's where the game is.

| Speedup | Score (NKI=0.85) | vs Competitor (1.9x) |
|---|---|---|
| 1.37x (current) | 1.60 | 0.52x their score |
| 1.9x (competitor) | 3.07 | 1.0x |
| 2.2x | 4.11 | **1.34x** |
| 2.5x | 5.31 | **1.73x** |
| 3.0x | 7.65 | **2.49x** |

---

## Mathematical floor (bf16, TP=4, trn3.3xlarge)

Conservative HBM: 550 GB/s effective per chip × 85% utilization ≈ **467 GB/s**. Qwen3-30B-A3B per-token working set:

| Component | Bytes/token | HBM-bound latency |
|---|---|---|
| Attention weights (48L × 12 MB) | 777 MB | **1.67 ms** |
| MoE weights (48L × 16.8 MB, K=8 experts) | 806 MB | **1.73 ms** |
| KV cache (S=2048, 48L) | 202 MB | 0.43 ms |
| Collectives (96 all-reduces × 4 KB) | — | ~0.5 ms |
| **Total serial** | **1.6 GB** | **~3.4 ms** |
| With compute-DMA overlap | | **~3.0 ms** |

**Current effective HBM BW is ~205 GB/s (37% utilization).** The ~2.6× headroom is purely kernel efficiency — not speed-of-light. Realistic target in bf16 with one week: **4.0–4.5 ms/token (2.0–2.2× speedup over baseline)**. Stretch with perfect overlap: 3.5 ms.

Per-component breakdown of the 8.76 ms:
- Attn kernel: 2.64 ms (floor 1.67 — **1.6× headroom**)
- MoE kernel: 4.94 ms (floor 1.73 — **2.9× headroom** — the prize)
- Collectives: 1.15 ms (floor 0.5 — overlap-able)
- LM head + post: 0.55 ms
- Framework overhead: 1.26 ms

---

## Current Per-Step Breakdown (48 layers)

| Component | Time | Source |
|---|---|---|
| Attn kernel x 48 | 2,640 us | 55 us x 48 |
| Attn AllReduce x 48 | ~576 us | ~12 us x 48 |
| MoE kernel x 48 | 4,944 us | 103 us x 48 |
| MoE AllReduce x 48 | ~576 us | ~12 us x 48 |
| Post-transformer (LM head + AllGather + 250 us dead) | 550 us | Profiler |
| **Accounted in NEFF** | **9,286 us** | |
| **NEFF total** | **9,340 us** | Profiler |
| **In-NEFF gap** | **~54 us** | Embedding + residuals |
| **Framework/harness** | **1,260 us** | 10,600 - 9,340 |
| **Benchmark total** | **10,600 us** | benchmark_report_best.json |

---

## Ranked actions (7 days)

### P0 — Fix the regression (Day 1)
The multilayer kernel lost ~0.3 ms vs round-1. Two suspects flagged in `transformer_qwen_multilayer.py`:
- **Line 13**: "ENC_ALG_MESH workaround... retest on trn3." Test bare `nccl.all_reduce` vs `_sb2sb_all_reduce_gather` — both patterns currently run; pick one.
- **Lines 73–78**: SBUF budget risk on 48 layers. Confirm via profile that no spill/reload beyond the 25.2 MB already logged.

**If multilayer can't beat round-1 by EOD Day 1, revert and optimize the split-kernel path instead.** Don't sink days into a worse baseline.

### P1 — Port to nkilib's fused MoE block (Days 2–3)
`nkilib/core/moe_block/moe_block_tkg.py` fuses RMSNorm → Router → Shared+Expert MLPs in one kernel — handwritten by Neuron engineers with LNC-aware sendrecv reduction. v30a is 4.94 ms (103 µs/layer). nkilib's moe_block with proper sendrecv pipelining should hit 60–75 µs/layer in bf16 → **saves ~1.3–2.0 ms**.

Risk: integration plumbing. Reward: largest single win.

### P1 — Collective-compute overlap (Day 3–4)
96 all-reduces at 10 µs each = 0.96 ms. Two approaches:
1. **Pipeline sendrecv with arithmetic** inside `_sb2sb_all_reduce_gather` — start reducing halves while still receiving (see `nkilib/experimental/collectives/fgcc.py` for the pattern).
2. **Overlap next-layer QKV weight DMA with current-layer's MoE all-reduce.** The all-reduce engine and DMA engine are independent.

Expected: 0.5 ms recovered.

### P1 — LM head in NKI (Day 4)
77.8M MACs = 2048×37984×2 bytes = 155 MB weight read → HBM floor 0.33 ms. Currently 0.55 ms in XLA. Also **lifts `NKI_FLOP_Ratio` from ~0.80 to ~0.90** — **+12.5% score independent of speedup**.

Write a dedicated NKI kernel: final RMSNorm + LM head matmul fused, reading residual directly from the megakernel's output HBM. Reuse nkilib row-parallel matmul patterns.

### P2 — Framework overhead (Day 5)
1.26 ms (14% of total) is Python/NxDI gap between steps. Enable:
- **On-device sampling** (`on_device_sampling_config`) — eliminates CPU logits roundtrip
- **`gather_output=False`** in LM head to keep logits local

Worth ~0.5–0.8 ms. Pure config, low-risk.

### P2 — Attention tightening (Day 5–6)
v14a at 55 µs/layer (1.67 ms floor → 34 µs speed-of-light). Two changes:
- KV cache load and QKV-projection DMA in parallel (two DMA channels)
- RoPE tables preloaded to SBUF once across all 48 layers (currently reloaded per-layer)

Expected: 40–45 µs/layer → **0.5 ms saved**.

### P3 — Skip quantization (confirmed)
MXFP4 would give ~2 ms in theory but (a) weight pre-layout is nontrivial, (b) calibration + logit-exact validation is a multi-day project, (c) scoring rubric requires tight logit match — one week is too short. `docs/fp8_expert_quantization_plan.md` confirms the DMA-scale overhead problem. **Locked out of scope.**

### P3 — Speculative decoding (Day 6–7, only if ahead of schedule)
EAGLE gives 2–2.5× effective throughput but needs a draft head. Contest rubric measures per-token throughput — verify the scoring script accepts speculative results before investing. If yes, single biggest win available.

---

## Realistic outcome

| Path | Per-token | Speedup | Score multiplier |
|---|---|---|---|
| Fix regression only | 8.5 ms | 1.37× | 1.88 (baseline) |
| + nkilib moe_block + CC overlap | 6.0 ms | 1.95× | 3.80 (**+102%**) |
| + LM head NKI + on-device sampling + attn tighten | 4.5 ms | 2.60× | 6.76 (**+260%**) |
| + Speculative decoding (if rubric allows) | ~2.0 ms effective | 5.8× | 33.6 |

**Concrete target: 4.5 ms/token (2.6× speedup).** Achievable in bf16 without quantization. Puts the submission firmly in prize contention. Anything below 4.0 ms requires either quantization or speculative decoding.

---

## Model Architecture Reference

| Parameter | Value |
|---|---|
| hidden_size (H) | 2048 |
| num_attention_heads (Hq) | 32 |
| num_key_value_heads (Hkv) | 4 (GQA, replicated across TP) |
| head_dim (d) | 128 |
| moe_intermediate_size (I) | 768 (192 per TP rank, padded to 256) |
| num_experts (E) | 128 |
| num_experts_per_tok (K) | 8 |
| num_hidden_layers (L) | 48 |
| vocab_size (V) | 151936 |

---

## Scoring Details

### Accuracy Validation (from main.py lines 418-500)

- NOT bit-exact — uses floating-point tolerance maps
- `tol_map = {None: (1e-5, 0.05), 1000: (1e-5, 0.03), 50: (1e-5, 0.03), 5: (1e-5, 0.03)}`
- `divergence_difference_tol = 0.001`
- Greedy: argmax of logits must match or be within 0.001 of the expected top token
- If accuracy = 0, **entire submission scores 0** (hard failure)

### NKI FLOP Ratio (from main.py lines 552-634)

- Parses HLO modules from context encoding + token generation
- NKI MACs: from `AwsNeuronCustomNativeKernel` custom-calls' `mac_count` backend config
- Total MACs: all NKI custom-calls + all dot operations
- Ratio = NKI_MACs / Total_MACs (0 to 1)

### Test Configurations

- Batch size: always 1
- 5 test prompts with varying lengths (64, 128, 256, 640 tokens)
- Uses P99 latency (tail latency matters)
- Aggregate score across all prompts

---

## First action right now

Run the multilayer kernel through Neuron profiler and confirm which of the two AllReduce paths fires, and whether SBUF is spilling. That one measurement decides whether Day 1 is "tune multilayer" or "revert to split-kernel".
