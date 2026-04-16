# Kernel Source Attribution — Per-Instruction Code References

The `neuron-profile view --output-format json` export tags every device instruction with the source line that produced it. This lets you answer "which line of my NKI kernel is causing this stall / DMA / idle gap" directly from the profile — the key tool for e2e kernel-level analysis.

## Generating the JSON

```bash
neuron-profile view -n <model.neff> -s <file.ntff> \
  --output-format json --output-file /tmp/prof.json
```

**Warning:** output is large (~1 GB per ~250k instructions, ~5 GB RAM to `json.load`). Write to `/tmp`, not the repo. For repeated analysis, load once in a Python REPL and keep the object in memory.

`parquet` is NOT supported by `neuron-profile` (`"please use neuron-explorer"`). JSON is the path.

## Schema — top level

```python
d = json.load(open('/tmp/prof.json'))
d.keys()  # ['instruction_usage', 'stack_frame_function_name', 'metadata', 'cc_stream', 'instruction']
```

- `d['instruction']` — list of every device instruction (the main table)
- `d['stack_frame_function_name']` — id → function name, resolves `stack_frame_ids`
- `d['cc_stream']` — collective-communication events
- `d['instruction_usage']` — per-opcode aggregate stats
- `d['metadata']` — version info

## Per-instruction fields

Every entry in `d['instruction']` has these fields (relevant ones for kernel attribution):

| Field | Meaning |
|---|---|
| `timestamp` | Start time in **picoseconds** (÷1e6 → µs, ÷1e9 → ms). Verify: max timestamp ≈ `total_time * 1e12`. |
| `duration` | Instruction length in ps. Engine-local — parallel engines overlap. |
| `evt_wait_time` | **Stall time** — ps spent waiting on a semaphore before the instruction could issue. This is the key stall metric. |
| `label` / `subgroup` | Engine: `Tensor`, `Vector`, `Scalar`, `GpSimd`, `Sync`, `DMA` variants |
| `opcode` | e.g. `DMA_DIRECT2D`, `ACTIVATION`, `MATMUL`, `TENSOR_TENSOR`, `NOTIFY` |
| `hbm_read_bytes`, `hbm_write_bytes`, `sbuf_read_bytes`, `sbuf_write_bytes` | Bytes moved by this one instruction |
| `bir_debug_info_source_location` | **`<path>:<line>`** — most densely populated source link (compiler BIR level). Use this for aggregation. |
| `nki_source_location` | `<path>:<line>` — sparser, NKI-specific. Present on kernels written in NKI. |
| `hlo_attrs` | JSON string with `op_type`, `source_file`, `source_line` at the PyTorch/HLO level |
| `layer` | PyTorch module hierarchy, e.g. `.../ParallelEmbedding[.2][3]/_forward_shard_across_embed/aten.embedding.default` |
| `stack_frame_ids` | Comma-joined ids → resolve against `d['stack_frame_function_name']` |
| `subgraph` | `sg00`, `sg01`, … — which compiler subgraph (roughly which TP rank / worker) |
| `compiler_operands`, `operands` | Full instruction operands (pattern shapes, DMA descriptors, semaphore ids) |

## Common analyses

### 1. Hotspot by source line (time-weighted)

```python
import json, collections
d = json.load(open('/tmp/prof.json'))

by_src = collections.defaultdict(lambda: {'dur':0,'cnt':0,'stall':0,
                                           'hbm_r':0,'hbm_w':0,'sbuf_w':0})
for i in d['instruction']:
    src = i.get('bir_debug_info_source_location') or i.get('nki_source_location')
    if not src: continue
    s = by_src[src]
    s['dur']   += i.get('duration', 0)
    s['stall'] += i.get('evt_wait_time', 0)
    s['cnt']   += 1
    s['hbm_r'] += i.get('hbm_read_bytes', 0)
    s['hbm_w'] += i.get('hbm_write_bytes', 0)
    s['sbuf_w']+= i.get('sbuf_write_bytes', 0)

# Top lines by total stall time
top = sorted(by_src.items(), key=lambda kv: -kv[1]['stall'])[:20]
for src,s in top:
    print(f"{s['stall']/1e6:8.1f}us stall  {s['cnt']:6d}x  {src}")
```

### 2. Skinny DMAs (small transfers, lots of overhead)

DMAs pay a fixed issue cost; many small transfers stall the DMA engine. Find source lines issuing sub-KB DMAs at high frequency:

```python
dma_ops = {'DMA_DIRECT2D','DMA_HBM_TO_SBUF','DMA_SBUF_TO_HBM','PSEUDO_DMA_DIRECT2D'}
skinny = collections.defaultdict(lambda: {'cnt':0,'bytes':0,'dur':0})
for i in d['instruction']:
    op = i.get('compiler_opcode') or i.get('opcode','')
    if 'DMA' not in op: continue
    b = i.get('hbm_read_bytes',0) + i.get('hbm_write_bytes',0) + i.get('sbuf_write_bytes',0)
    if b > 0 and b < 512:  # skinny threshold
        src = i.get('bir_debug_info_source_location','<none>')
        s = skinny[src]
        s['cnt'] += 1; s['bytes'] += b; s['dur'] += i.get('duration',0)

for src,s in sorted(skinny.items(), key=lambda kv:-kv[1]['cnt'])[:15]:
    avg = s['bytes']/s['cnt']
    print(f"{s['cnt']:6d}x  avg={avg:6.0f}B  total_dur={s['dur']/1e6:.1f}us  {src}")
```

### 3. Stall attribution — who's waiting on whom

`evt_wait_time` is the gap between when a semaphore makes an instruction eligible and when the instruction actually starts. High `evt_wait_time` on a consumer engine usually means the producer was late.

```python
# Per-engine stall totals, grouped by source line
eng_stalls = collections.defaultdict(lambda: collections.Counter())
for i in d['instruction']:
    w = i.get('evt_wait_time', 0)
    if w <= 0: continue
    eng_stalls[i.get('label','?')][i.get('bir_debug_info_source_location','<none>')] += w

for eng, ctr in eng_stalls.items():
    total = sum(ctr.values())
    print(f'\n=== {eng}: {total/1e6:.1f}us total stall ===')
    for src,w in ctr.most_common(5):
        print(f'  {w/1e6:7.1f}us  {src}')
```

### 4. Engine idle gaps — where the timeline has holes

Sort instructions for one engine by timestamp; any gap larger than the surrounding duration is idle time.

```python
eng = 'Tensor'
instrs = sorted([i for i in d['instruction'] if i.get('label')==eng],
                key=lambda x: x['timestamp'])
gaps = []
for prev,cur in zip(instrs, instrs[1:]):
    end_prev = prev['timestamp'] + prev.get('duration',0)
    gap = cur['timestamp'] - end_prev
    if gap > 1000:  # >1ns ~ threshold in ps; tune
        gaps.append((gap, prev.get('bir_debug_info_source_location'),
                          cur.get('bir_debug_info_source_location')))
gaps.sort(reverse=True)
for g,a,b in gaps[:20]:
    print(f'{g/1e6:8.1f}us  after {a}  before {b}')
```

### 5. Bytes-per-line (DMA volume attribution)

```python
by_line_bytes = collections.Counter()
for i in d['instruction']:
    src = i.get('bir_debug_info_source_location')
    if not src: continue
    by_line_bytes[src] += i.get('hbm_read_bytes',0)

for src,b in by_line_bytes.most_common(15):
    print(f'{b/1e6:8.1f} MB  {src}')
```

### 6. Per-layer source attribution

The `layer` field gives the PyTorch module path. Combined with source lines, you can ask "which kernel line dominates inside `self_attn.o_proj` specifically":

```python
target = 'self_attn/o_proj'  # substring match
hits = collections.Counter()
for i in d['instruction']:
    if target in i.get('layer',''):
        src = i.get('bir_debug_info_source_location','<none>')
        hits[src] += i.get('duration',0) + i.get('evt_wait_time',0)
for src,t in hits.most_common(10):
    print(f'{t/1e6:7.1f}us  {src}')
```

## Practical tips

- **`duration` is engine-local.** Summing all instruction durations gives *far less* than total_time because engines run in parallel. For wall-clock attribution, use `evt_wait_time` (stalls), timeline gaps, or per-engine active time from `view summary-json`.
- **`bir_debug_info_source_location` is the densest link.** `nki_source_location` is often empty for non-NKI code; prefer bir unless you specifically want NKI-only.
- **Filter by subgraph** (`sg00` vs `sg01`) to isolate one TP rank when the NTFF covers multiple ranks/workers.
- **Opcodes to know:** `MATMUL`/`MATMUL_MX` (Tensor), `ACTIVATION`/`TENSOR_TENSOR` (Vector/Scalar), `DMA_*` and `PSEUDO_DMA_*` (DMA queues), `NOTIFY`/`SEM_*` (GpSimd/Sync control), `CC_*` (collectives). Filter on `opcode` or `compiler_opcode` to isolate a category.
- **Resolving `--nki-source-root` / `--framework-source-root`** on `neuron-profile view` changes how paths embed in Perfetto output; for JSON analysis you already have the absolute path.
- For interactive browsing instead of scripting, also emit `--output-format perfetto --output-file out.pb` and load in https://ui.perfetto.dev — slices are annotated with the same `layer` and source-location strings.
