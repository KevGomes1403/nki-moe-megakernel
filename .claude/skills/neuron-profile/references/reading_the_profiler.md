# Reading the Neuron Profiler — Terminology & Timeline Guide

A field guide to what you see in Perfetto / `summary-text` / raw trace, based on hands-on exploration of trn3 token-generation profiles. Answers the "what even is `PSEUDO_DMA_DIRECT2D`?" questions.

---

## File types

| File | What it is |
|---|---|
| **NEFF** (`.neff`) | Compiled model binary — HLO graph + BIR instructions + weight layout. Needed by `neuron-profile view` to resolve instruction IDs to source lines and op names. |
| **NTFF** (`.ntff`) | Trace file — timestamped events captured from the device during execution. Standalone data, no compiled-program info. `show-session` can read it without a NEFF. |
| **NTFF v6** | Current format. `neuron-profile` auto-converts to v0 internally ("matched notifications back to N graphs"). |
| Bundle (`.tar`) | Multiple NTFFs for multi-worker profiles (a directory of `*_rank_*.ntff`). Pass with `-d`. |

A single `neuron-profile view` run needs the NEFF that produced the NTFF — they're a matched pair identified by module hash.

---

## Hardware model (what the tracks in Perfetto represent)

A NeuronCore (trn3 = NeuronCore-v4) has several independent engines that run in parallel. Each one becomes its own row in the timeline.

| Engine | What it does | Typical opcodes |
|---|---|---|
| **TensorMatrix** (a.k.a. PE) | 128×128 systolic matmul array. The heaviest compute unit. | `MATMUL`, `MATMUL_MX` (MXFP), `LDWEIGHTS` (stationary load), transposes via `nc_transpose` |
| **Tensor** | Issues matmul "launch" instructions / load-weights sequencing. Often shown alongside TensorMatrix. | `LDWEIGHTS`, matmul issue |
| **Vector** | Elementwise + reduction along free dim. Most "tensor_tensor" / "tensor_scalar" ops. | `TENSOR_TENSOR`, `TENSOR_SCALAR`, `ACTIVATION` (some) |
| **Scalar** | Scalar arithmetic on partition dim, activations. | `ACTIVATION` (exp/relu/etc.), `TENSOR_SCALAR` |
| **GpSimd** | General-purpose SIMD on partition dim. Scatter/gather, some elementwise, control logic. | `ALU_OP`, indirect DMA descriptor prep |
| **Sync** | Semaphore & DMA-trigger control. Doesn't compute — coordinates. | `NOTIFY`, `SEMAPHORE_WAIT`, `PSEUDO_DMA_TRIGGER`, `DMA_ADVANCE` |
| **DMA queues** (many) | Physical memory movers (HBM↔SBUF, SBUF↔SBUF, scatter/gather). | `DMA_DIRECT2D`, `DMA_HBM_*`, hardware-descriptor variants |
| **CC-cores** | Collective communication (AllReduce, sendrecv across ranks). One per TP direction; numbered (e.g. 8, 9, 10, 11 for TP=4). | `TPB_TRIGGER`, collective events |

### Memory spaces

| Space | Size (trn3 order) | Visible in profile as |
|---|---|---|
| **HBM** | 96 GB off-chip | `hbm_read_bytes`, `hbm_write_bytes`; source of weight streaming |
| **SBUF** | ~24 MB on-chip scratchpad, 128 partitions × free-dim | `sbuf_read_bytes`, `sbuf_write_bytes` |
| **PSUM** | ~2 MB accumulator banks, close to the PE | `psum_read_bytes`, `psum_write_bytes` |

SBUF is partitioned across 128 rows (the "partition dim"); the "free dim" is contiguous along each row. Most op layouts in code read as `[P, F]` = `[partition, free]`.

---

## Opcode taxonomy (what's actually running)

### Compute opcodes

- **`MATMUL` / `MATMUL_MX`** — systolic matmul on the TensorMatrix. MX variant uses microscaling FP8/FP4 on trn3.
- **`LDWEIGHTS`** — load the "stationary" operand into the PE array. Precedes a matmul; shows up on the Tensor engine.
- **`ACTIVATION`** — non-linearity (exp, relu, sigmoid, …). Runs on Scalar (usually) or Vector.
- **`TENSOR_TENSOR`** — elementwise binary op between two tensors (add, multiply, etc.) on Vector.
- **`TENSOR_SCALAR`** — op between a tensor and a scalar operand.
- **`ALU_OP`** — generic GpSimd compute (often scatter indices or post-processing).
- **`nc_transpose`** — PE-based transpose (partition↔free swap). Emitted on Tensor/TensorMatrix tracks.

### DMA opcodes (this is where the "pseudo" confusion comes from)

A DMA on Neuron has two phases:
1. **Descriptor generation** — figure out the (src addr, dst addr, stride, shape) tuples.
2. **Execution** — the DMA queue hardware actually moves the bytes.

The compiler distinguishes these in the opcode naming:

| Opcode | Meaning |
|---|---|
| **`DMA_DIRECT2D`** | A real DMA with a 2D descriptor (partition × free). The hardware executes it. |
| **`PSEUDO_DMA_DIRECT2D`** | A DMA whose descriptor is generated at runtime and issued via the **Descriptor Generation Engine (DGE)** — shows up on Sync/GpSimd tracks. The "pseudo" prefix means "not a static compile-time DMA queue entry" — it's software-triggered. Still moves real bytes. |
| **`PSEUDO_DMA_TRIGGER`** | Fires a pre-staged DMA descriptor via a Sync-engine trigger. Appears on Sync tracks with near-zero duration but non-zero `evt_wait_time` (the actual transfer happens on a DMA queue track). |
| **`DMA_HBM_TO_SBUF`** / **`DMA_SBUF_TO_HBM`** | Explicit direction-tagged bulk transfers. |
| **`DMA_ADVANCE`** | Advance a DMA queue's head pointer — control, not transfer. |
| **HWDGE vs SWDGE** | "Hardware Descriptor Generation Engine" vs "Software" — `dge_mode=nisa.dge_mode.hwdge` in NKI code means the hardware DGE generates descriptors (cheaper). SWDGE uses GpSimd to generate them (more flexible, slower). |

**Rule of thumb when reading the timeline:**
- See `PSEUDO_DMA_*` on the **Sync** row? That's a trigger/descriptor op, look for the matching DMA-queue slice (same timestamp) for the actual byte transfer.
- See `DMA_DIRECT2D` on a **DMA queue** row? That's the real transfer — its duration × bytes gives effective BW.
- `LDWEIGHTS` on Tensor often pairs with a PE matmul a few nanoseconds later.

### Control / synchronization opcodes

- **`NOTIFY`** — posts a semaphore increment (producer signals "done").
- **`SEMAPHORE_WAIT`** — consumer blocks until a semaphore value is reached.
- **`TPB_TRIGGER`** — collective-communication trigger (AllReduce start). Appears on CC-core tracks.
- **`CC_*`** — collective variants (AllReduce, ReduceScatter, etc.).

`evt_wait_time` on any instruction = time spent at `SEMAPHORE_WAIT` before it could run. **This is the key stall metric** in the JSON export.

---

## Key fields in `summary-text` output

When you run `neuron-profile view --output-format summary-text`, you get one block per subgraph. Field groupings:

### Time
- `total_time` — wall-clock, seconds.
- `total_active_time` — union of all engine-active time. `total_active_time / total_time` is an "any engine busy" measure.
- `<engine>_active_time` / `_active_time_percent` — per-engine busy time.

### Compute quality
- `mfu_estimated_percent` — matmul FLOPs achieved vs peak. Low MFU (<10%) ⇒ memory- or collective-bound.
- `mm_arithmetic_intensity` — FLOPs per byte moved. <10 ⇒ memory-bound, >100 ⇒ compute-bound (roughly).
- `hardware_flops` / `adjusted_hardware_flops` — FLOPs executed (raw vs counting only useful work).

### Memory / DMA
- `hbm_read_bytes`, `hbm_write_bytes` — HBM traffic.
- `sbuf_*`, `psum_*` — on-chip traffic.
- `spill_reload_bytes` — bytes that had to round-trip via HBM because SBUF didn't fit. >0 is a warning sign; high values mean refactor for reuse.
- `dma_packet_count`, `static_dma_size` / `hardware_dynamic_dma_size` — DMA work split into statically-scheduled vs runtime-generated.
- `mbu_estimated_percent` — memory bandwidth utilization estimate.

### Collectives
- `cc_op_count`, `cc_op_active_time` — number & total duration of AllReduce / sendrecv ops.

### Throttling
- `throttle_*_nc[0|1]_percent` — how often the core ran at reduced power/clock. <1.0 means compiler-reported throttle; usually fine.

---

## Reading the Perfetto timeline

Open `.pb` files at https://ui.perfetto.dev.

**Typical layout (top to bottom):**
1. CC-core tracks — tall gaps between `TPB_TRIGGER` slices are collective latency.
2. Tensor / TensorMatrix — matmul density. Gaps = PE idle.
3. Vector, Scalar, GpSimd — elementwise work.
4. Sync — short tick marks at semaphore/DMA trigger events.
5. DMA queues (many rows) — byte-movement slices.

**What to look for:**
- **Wide PE gaps with busy DMA rows** = memory-bound, weight streaming is the bottleneck.
- **Wide DMA gaps with busy PE** = compute-bound.
- **All engines quiet simultaneously** = global stall, usually at a CC boundary.
- **`TPB_TRIGGER`s very far apart** on one CC-core vs another = rank imbalance.
- **Long single DMA slice** vs lots of short ones = thick vs skinny transfer. Skinny is bad (issue overhead dominates).

**Annotations:** slices in Perfetto are labeled with the PyTorch `layer` path (e.g. `...ParallelEmbedding/aten.embedding`) and the source-file:line if you pass `--nki-source-root` / `--framework-source-root` to `view`. This is how you correlate a hotspot on the timeline to a kernel source line.

---

## Terminology cheat sheet

| Term | Meaning |
|---|---|
| **BIR** | Binary Intermediate Representation — the compiler's pre-assembly instruction stream. `bir_id` / `bir_debug_info_source_location` live here. |
| **HLO** | High-Level Operations — XLA-level ops, pre-compilation. `hlo_name`, `hlo_attrs` trace back to the PyTorch graph. |
| **Subgraph** (`sg00`, `sg01`) | A compiler-created execution unit. In TP-sharded models, different ranks/workers often map to different subgraphs in the same NEFF. |
| **Penguin ID** (`penguin_id`) | Internal compiler instruction ID; not user-facing, but useful for cross-referencing. |
| **NC** (NeuronCore) | The compute core. `ND 0 NC 4` = NeuronDevice 0, NeuronCore 4. |
| **LNC** | "Logical NeuronCore" — software-level view when LNC=2 (two physical cores treated as one logical unit for sharding). |
| **DGE** | Descriptor Generation Engine — hardware (HWDGE) or software (SWDGE) unit that builds DMA descriptors. |
| **CC-core** | Collective Communication core. Handles cross-device collective ops. |
| **Stationary / moving** | `nc_matmul(stationary=A, moving=B)` — stationary stays in the PE array (uses `LDWEIGHTS` once), moving streams through. |
| **`evt_wait_time`** | Picoseconds spent waiting on a semaphore before this instruction could issue. Stall metric. |
| **`duration`** | Picoseconds the instruction itself took to run. Engine-local. |
| **`timestamp`** | Absolute start time in picoseconds. Max timestamp ≈ total_time × 1e12. |
| **`layer`** | PyTorch module path hierarchy producing this instruction. |
| **CycleCount / profile clock** | ~1 GHz on trn2/trn3 — so 1 cycle ≈ 1 ns. Verify against `total_time`. |

---

## Mental model for common patterns

- **"Why is there a gap on the PE before a matmul?"** — LDWEIGHTS wasn't ready (weights still DMA'ing from HBM). Check DMA rows around that timestamp.
- **"Why does `PSEUDO_DMA_TRIGGER` have a 17 µs stall but 0 duration?"** — The trigger itself is instant, but the Sync engine was waiting on a semaphore before it could fire (usually: the producer hasn't finished yet, or the DGE is backlogged). The real bytes move on a DMA queue track separately.
- **"Why is `total_active_time` < sum of per-engine active time?"** — engines run in parallel; their active windows overlap. That's good; you want overlap.
- **"Why is MFU 0.5%?"** — token-generation is batch-1, memory-bound. MFU is a matmul utilization metric; in decode you care about MBU (memory BW utilization) instead.
- **"Why two matching AR triggers per layer?"** — one after attention output projection, one after MoE/FFN output projection. That's how `references/layer_latency.md` derives per-layer timing.
