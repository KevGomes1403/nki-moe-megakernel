---
name: neuron-profile
description: General-purpose analysis of AWS Neuron profiler NTFF output. Given NTFF (and optionally NEFF) file paths, extracts and analyzes any aspect of device execution — instruction mix, DMA bandwidth, collective timing, engine utilization, per-op timing, or custom queries against the raw trace.
argument-hint: <ntff-file-or-dir> [neff-file]
allowed-tools: Bash(neuron-profile *) Bash(ls *) Bash(find *) Bash(awk *) Bash(grep *) Bash(python3 *) Bash(head *) Bash(tail *) Bash(wc *)
---

Analyze Neuron profiler output. Arguments: **$ARGUMENTS**

Parse as: first token ending in `.ntff` or a directory → NTFF; first token ending in `.neff` → NEFF. If a directory is given, find all `.ntff` files inside it.

---

## Tool reference

### `neuron-profile show-session` — raw trace data (no NEFF needed)

```bash
# Session summary table (execution time, instruction counts per engine)
neuron-profile show-session -s <file.ntff>

# Same as JSON
neuron-profile show-session -s <file.ntff> -j

# Raw instruction trace (Tensor/Vector/Scalar/GpSimd/Sync engines, with timestamps)
neuron-profile show-session -s <file.ntff> --show-trace

# Raw DMA trace (M2S/S2M transfers with timestamps and sizes)
neuron-profile show-session -s <file.ntff> --show-dma

# Collective communication trace (AllReduce/sendrecv triggers per CC-core)
# Embedded in --show-trace output under "Collectives trace for CC-core N"

# Show runtime errors
neuron-profile show-session -s <file.ntff> -e

# Dump per-instruction details to files
neuron-profile show-session -s <file.ntff> -i
```

### `neuron-profile view` — aggregated metrics (requires NEFF)

```bash
# Human-readable summary: engine utilization, DMA bandwidth, MFU, arithmetic intensity
neuron-profile view -n <model.neff> -s <file.ntff> --output-format summary-text

# Same metrics as JSON (pipe to python3 for filtering)
neuron-profile view -n <model.neff> -s <file.ntff> --output-format summary-json

# Perfetto binary timeline (open in ui.perfetto.dev)
neuron-profile view -n <model.neff> -s <file.ntff> --output-format perfetto --output-file out.pb

# Full per-instruction JSON with source-line attribution (kernel-level analysis)
# Each instruction has: timestamp, duration, evt_wait_time (stall!), engine label,
# opcode, hbm/sbuf bytes, layer (PyTorch module path), bir_debug_info_source_location,
# nki_source_location, hlo_attrs. Write to /tmp — output is ~1 GB per 250k instructions.
neuron-profile view -n <model.neff> -s <file.ntff> --output-format json --output-file /tmp/prof.json

# NOTE: --output-format parquet is NOT supported (tool says "use neuron-explorer"). Use json.
```

---

## Data available in each source

| Source | What you get |
|--------|-------------|
| `show-session` (default) | Execution time (ns), per-engine instruction counts (Tensor/Vector/Scalar/GpSimd/Sync/DMA), error count |
| `show-session -j` | Same as JSON; `NeffNodes[].NodeInfo.Graphs[].CycleCount` for total cycles |
| `show-session --show-trace` | Per-instruction timestamps, engine type (Tensor/Vector/Scalar/GpSimd/Sync), PC, start/end; also collective events (TPB_TRIGGER, SEMAPHORE_WAIT, DMA_ADVANCE) per CC-core |
| `show-session --show-dma` | Per-DMA-queue events (M2S_SOP, S2M_EOP, S2M_COMPLETION) with absolute timestamps and CRC |
| `view summary-text/json` | Aggregated metrics: `total_time`, `tensor_engine_active_time`, `vector_engine_active_time`, `dma_active_time`, `hbm_read_bytes`, `hbm_write_bytes`, `sbuf_read_bytes`, `sbuf_write_bytes`, `mfu_estimated_percent`, `mm_arithmetic_intensity`, `cc_op_count`, `cc_op_active_time`, `spill_reload_bytes`, `psum_*` |
| `view --output-format json` | **Per-instruction table with source-line attribution.** Every instruction tagged with `bir_debug_info_source_location`, `nki_source_location`, PyTorch `layer` path, plus `timestamp`, `duration`, `evt_wait_time` (stall), engine `label`, `opcode`, and per-instr HBM/SBUF bytes. Essential for kernel-level analysis (skinny DMAs, stall attribution, idle gaps, bytes-per-line). See [`references/kernel_source_attribution.md`](.claude/skills/neuron-profile/references/kernel_source_attribution.md). |

---

## Procedure

### 1. Orient — identify what's in the NTFF

```bash
neuron-profile show-session -s <file.ntff>
```

Note: Model Name, total execution time, per-engine instruction counts and any errors. If a directory was given with multiple NTFFs, run this on each unique prefix to identify model types (context_encoding_model, token_generation_model, layout_opt, etc.).

### 2. Understand the analysis goal

Based on what the user wants to know, pick the right data source:

- **Overall performance / utilization / bandwidth** → `view summary-text` (needs NEFF)
- **Engine balance / idle time / bottleneck** → `view summary-text` or `show-session` instruction counts
- **Timeline / overlap / stalls** → `show-session --show-trace`, parse timestamps
- **Collective/AllReduce timing** → `show-session --show-trace`, parse CC-core section
- **DMA bandwidth / transfer sizes** → `show-session --show-dma`
- **Spill/reload pressure** → `view summary-text`, `spill_reload_bytes`
- **MFU / arithmetic intensity** → `view summary-text`, `mfu_estimated_percent` + `mm_arithmetic_intensity`
- **Kernel-level attribution (stalls per source line, skinny DMAs, idle gaps, bytes-per-line)** → `view --output-format json` + python3, see [`references/kernel_source_attribution.md`](.claude/skills/neuron-profile/references/kernel_source_attribution.md)
- **Custom op-level breakdown** → `view --output-format json` + python3 analysis

### 3. Extract the data

Run the appropriate command(s) from the tool reference above. For trace-based analysis, pipe through `awk`/`grep`/`python3` to isolate the signal. Keep commands focused — large traces have hundreds of thousands of lines.

Useful trace extraction patterns:

```bash
# All instruction timestamps for one engine type
neuron-profile show-session -s <f.ntff> --show-trace 2>&1 \
  | awk '/Instruction trace for ND 0 NC 4/{found=1;next} found && /Tensor.*START/{print $5}'

# CC-core collective events (AllReduce triggers)
neuron-profile show-session -s <f.ntff> --show-trace 2>&1 \
  | grep -A99999 "Collectives trace for CC-core 8" \
  | grep "TPB_TRIGGER" | grep "START" | awk '{print $2}'

# DMA transfer sizes and timing
neuron-profile show-session -s <f.ntff> --show-dma 2>&1 \
  | awk '/S2M_EOP/{print $4, $5}'   # CRC (proxy for size bucket) and timestamp

# Engine active time breakdown from summary
neuron-profile view -n <model.neff> -s <f.ntff> --output-format summary-json 2>/dev/null \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
sg=list(d.values())[0]
keys=['tensor_engine_active_time','vector_engine_active_time','dma_active_time',
      'cc_op_active_time','total_time','mfu_estimated_percent','hbm_read_bytes']
for k in keys:
    if k in sg: print(f'{k}: {sg[k]}')
"
```

### 4. Analyze and interpret

Compute derived quantities as needed (ratios, gaps, rates). Convert timestamps to time using the cycle-to-time ratio from `CycleCount` and `total_time`:

```bash
# Get CycleCount and total_time for unit conversion
neuron-profile show-session -s <f.ntff> -j 2>/dev/null \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
g=d['NeffNodes'][0]['NodeInfo']['Graphs'][0]
print('CycleCount:', g['CycleCount'])
"
neuron-profile view -n <neff> -s <f.ntff> --output-format summary-json 2>/dev/null \
  | python3 -c "import json,sys; d=json.load(sys.stdin); sg=list(d.values())[0]; print('total_time_s:', sg['total_time'])"
# ns_per_cycle = total_time_s * 1e9 / CycleCount
```

In practice on trn2/trn3 with ~1 GHz profile clock, **1 cycle ≈ 1 ns** is a good approximation. Verify against total_time.

### 5. Present results

Report findings clearly: tables for multi-value results, concrete numbers with units, and interpretation of what the numbers mean for performance. Reference the relevant analysis pattern from `references/` if applicable.

---

## References

- [`references/reading_the_profiler.md`](.claude/skills/neuron-profile/references/reading_the_profiler.md) — field guide to profiler terminology (engines, memory spaces, opcodes including `PSEUDO_DMA_*`, DGE, CC-cores) and how to read the Perfetto timeline
- [`references/layer_latency.md`](.claude/skills/neuron-profile/references/layer_latency.md) — per-decoder-layer timing via CC AllReduce gap analysis
- [`references/kernel_source_attribution.md`](.claude/skills/neuron-profile/references/kernel_source_attribution.md) — per-instruction source-line attribution via JSON export; recipes for stall attribution, skinny DMAs, idle gaps, bytes-per-line
