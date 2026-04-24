# Per-Decoder-Layer Latency via CC AllReduce Gap Analysis

## Technique

Each decoder layer executes exactly two AllReduce collectives — one after attention output projection, one after MoE/FFN output projection. On the CC-core trace, each AllReduce appears as a `TPB_TRIGGER` event. Measuring the gaps between consecutive `TPB_TRIGGER START` timestamps gives per-AllReduce durations, which map directly to per-layer compute time.

This works even without a NEFF because the pattern is structural: every transformer layer produces the same two-trigger rhythm.

## Extraction Command

```bash
neuron-profile show-session -s <file.ntff> --show-trace 2>&1 \
  | grep -A9999999 "Collectives trace for CC-core 8" \
  | grep "TPB_TRIGGER" \
  | grep "START" \
  | awk '{print $2}' \
  | awk 'NR>1 { gap = $1 - prev; printf "gap %d: %d ns (%.1f us)\n", NR-1, gap, gap/1000 } { prev = $1 }'
```

**Why CC-core 8?** The CC-core index depends on the TP group. For TP=4 on trn2, core 8 is reliably in the first TP rank. Adjust if your topology differs — check `show-session` output for the highest CC-core index that shows `TPB_TRIGGER` events.

**Why `$2`?** In the `--show-trace` output, CC-core events are printed as:
```
<event_type>  <timestamp_ns>  <START|END>  ...
```
Column 2 is the absolute timestamp in nanoseconds.

## Gap-to-Layer Mapping

| Gap index | Meaning |
|-----------|---------|
| 1 | Prologue (model setup / first DMA load) |
| 2 | Layer 0, AllReduce #1 (attention AR) |
| 3 | Layer 0, AllReduce #2 (MoE/FFN AR) |
| 4 | Layer 1, AllReduce #1 |
| 5 | Layer 1, AllReduce #2 |
| … | … |
| 2N | Layer N-1, AllReduce #1 |
| 2N+1 | Layer N-1, AllReduce #2 |

So **layer K** spans gaps **2K+2** and **2K+3** (0-indexed layers).

To sum per-layer total time:

```bash
neuron-profile show-session -s <file.ntff> --show-trace 2>&1 \
  | grep -A9999999 "Collectives trace for CC-core 8" \
  | grep "TPB_TRIGGER" \
  | grep "START" \
  | awk '{print $2}' \
  | awk 'NR>1 { gaps[NR-1] = $1 - prev } { prev = $1 }
         END {
           for (i=2; i<=length(gaps); i+=2) {
             layer = (i-2)/2
             total = gaps[i] + gaps[i+1]
             printf "layer %d: %d ns (%.1f us)\n", layer, total, total/1000
           }
         }'
```

## Example Output

From a Qwen3-MoE TKG profile (TP=4, 28 decoder layers, trn2):

```
gap 1:  ~10,000 ns   (10.0 us)   prologue
gap 2:  ~511,000 ns  (511 us)    layer 0, AR #1   ← slow: pipeline startup
gap 3:  ~511,000 ns  (511 us)    layer 0, AR #2
gap 4:  ~75,000 ns   (75 us)     layer 1, AR #1   ← steady-state
gap 5:  ~75,000 ns   (75 us)     layer 1, AR #2
...
gap 56: ~75,000 ns   (75 us)     layer 27, AR #1
gap 57: ~75,000 ns   (75 us)     layer 27, AR #2
```

Layer 0 ≈ 1.0 ms total. Layers 1–27 ≈ 150 µs each.

## Why Layer 0 Is Slow

With `--enable-ccop-compute-overlap`, the compiler pipelines DMA prefetches for layer N's weights during layer N-1's computation. Layer 0 has no preceding computation, so its weight loads run serially before the compute begins — paying the full DMA latency (~350 µs extra) that all subsequent layers hide behind pipelining.

This is expected behavior. There is no code fix; it is an inherent pipeline startup cost.

## Verification

Cross-check the total time:

```python
# Sum all gaps and compare to show-session execution time
total_gaps_ns = sum(all_gaps)
# Should equal (or be close to) the "Total execution time" from:
# neuron-profile show-session -s <file.ntff>
```

Small discrepancies (~5%) are normal due to prologue/epilogue overhead outside the CC-core trace window.
