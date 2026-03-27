"""
Parse neuron-profile parquet output and print key metrics to terminal.

Replaces manual GUI inspection. Works on any parquet directory produced by
neuron-profile or by the nki_benchmark decorator.

Usage:
    python parquet_reader.py /path/to/parquet_dir
    python parquet_reader.py parquet_files/profiles/global/attn-tkg-v6@latest/
"""

from __future__ import annotations

import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(parquet_dir: str, table: str) -> pd.DataFrame | None:
    path = os.path.join(parquet_dir, f"{table}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def _fmt_us(seconds: float) -> str:
    return f"{seconds * 1e6:.2f} μs"


def _fmt_pct(v: float) -> str:
    v = float(v)
    if 0 <= abs(v) <= 1:
        v *= 100
    return f"{v:.1f}%"


def _fmt_bytes(value: float) -> str:
    value = float(value)
    units = ("B", "KiB", "MiB", "GiB")
    unit_idx = 0
    while value >= 1024 and unit_idx < len(units) - 1:
        value /= 1024
        unit_idx += 1
    return f"{value:.1f} {units[unit_idx]}"


def _fmt_ns(value_ns: float) -> str:
    value_ns = float(value_ns)
    if value_ns >= 1e6:
        return f"{value_ns / 1e6:.2f} ms"
    if value_ns >= 1e3:
        return f"{value_ns / 1e3:.2f} μs"
    return f"{value_ns:.0f} ns"


def _fmt_rate(value_bps: float) -> str:
    value_bps = float(value_bps)
    if value_bps >= 1024 ** 3:
        return f"{value_bps / (1024 ** 3):.2f} GiB/s"
    if value_bps >= 1024 ** 2:
        return f"{value_bps / (1024 ** 2):.2f} MiB/s"
    if value_bps >= 1024:
        return f"{value_bps / 1024:.2f} KiB/s"
    return f"{value_bps:.1f} B/s"


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _print_distribution(label: str, values: pd.Series, formatter) -> None:
    if values.empty:
        return
    print(f"  {label:<20} avg={formatter(values.mean())}"
          f"  p50={formatter(values.quantile(0.50))}"
          f"  p95={formatter(values.quantile(0.95))}"
          f"  max={formatter(values.max())}")


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _pct_of_total(value_s: float, total_s: float) -> str:
    """Percentage of value relative to total_time (not the raw trace window)."""
    if total_s <= 0:
        return "0.0%"
    return f"{value_s / total_s * 100:.1f}%"


def _print_summary(parquet_dir: str) -> None:
    df = _load(parquet_dir, "Summary")
    if df is None or df.empty:
        print("  [Summary.parquet not found]")
        return

    r = df.iloc[0]
    total_s = float(r.get("total_time", 0))

    print(f"\n  ── Execution Time ──────────────────────────────")
    print(f"  total_time          = {_fmt_us(total_s)}")
    active_s = float(r.get("total_active_time", 0))
    print(f"  total_active_time   = {_fmt_us(active_s)}"
          f"  ({_pct_of_total(active_s, total_s)} of total)")

    print(f"\n  ── Engine Utilization (% of total_time) ────────")
    for label, key in [
        ("tensor_engine", "tensor_engine_active_time"),
        ("vector_engine", "vector_engine_active_time"),
        ("scalar_engine", "scalar_engine_active_time"),
        ("dma_active",    "dma_active_time"),
        ("gpsimd_engine", "gpsimd_engine_active_time"),
    ]:
        v = float(r.get(key, 0))
        print(f"  {label:<20} = {_pct_of_total(v, total_s)}"
              f"  ({_fmt_us(v)})")

    print(f"\n  ── Compute Efficiency ──────────────────────────")
    print(f"  mfu_estimated       = {_fmt_pct(r.get('mfu_estimated_percent', 0))}")
    print(f"  mfu_max_achievable  = {_fmt_pct(r.get('mfu_max_achievable_estimated_percent', 0))}")
    print(f"  mbu_estimated       = {_fmt_pct(r.get('mbu_estimated_percent', 0))}")
    print(f"  mm_arith_intensity  = {r.get('mm_arithmetic_intensity', 0):.3f}")

    print(f"\n  ── Memory Traffic ──────────────────────────────")
    print(f"  hbm_read            = {r.get('hbm_read_bytes', 0) / 1024:.1f} KiB")
    print(f"  hbm_write           = {r.get('hbm_write_bytes', 0) / 1024:.1f} KiB")
    print(f"  sbuf_read           = {r.get('sbuf_read_bytes', 0) / 1024:.1f} KiB")
    print(f"  sbuf_write          = {r.get('sbuf_write_bytes', 0) / 1024:.1f} KiB")
    print(f"  spill_save          = {r.get('spill_save_bytes', 0)} bytes")
    print(f"  spill_reload        = {r.get('spill_reload_bytes', 0)} bytes")
    print(f"  dma_transfer_total  = {r.get('dma_transfer_total_bytes', 0) / 1024:.1f} KiB")

    print(f"\n  ── DMA Breakdown ───────────────────────────────")
    print(f"  static_dma          = {_fmt_pct(r.get('static_dma_active_time_percent', 0))}"
          f"  ({_fmt_us(r.get('static_dma_active_time', 0))})")
    print(f"  dynamic_dma         = {_fmt_pct(r.get('dynamic_dma_active_time_percent', 0))}")
    print(f"  sw_dynamic_dma      = {_fmt_pct(r.get('software_dynamic_dma_active_time_percent', 0))}"
          f"  ({_fmt_us(r.get('software_dynamic_dma_active_time', 0))})")
    print(f"  dma_packet_count    = {int(r.get('dma_transfer_count', 0))}")
    print(f"  avg_packet_size     = {_fmt_bytes(r.get('dma_transfer_average_bytes', 0))}")
    print(f"  dma_packet_time     = {_fmt_us(r.get('dma_packet_time', 0))}")
    print(f"  dma_transfer_time   = {_fmt_us(r.get('dma_transfer_time', 0))}")
    print(f"  dma_queue_count     = {int(r.get('dma_queue_count', 0))}")
    print(f"  static_packet_count = {int(r.get('static_dma_packet_count', 0))}")
    print(f"  sw_dynamic_packets  = {int(r.get('software_dynamic_dma_packet_count', 0))}")
    print(f"  hw_dynamic_packets  = {int(r.get('hardware_dynamic_dma_packet_count', 0))}")


def _print_dma_packets(parquet_dir: str) -> None:
    packets = _load(parquet_dir, "DmaPacket")
    aggregated = _load(parquet_dir, "DmaPacketAggregated")
    usage = _load(parquet_dir, "DmaUsage")
    pending = _load(parquet_dir, "PendingDma")
    queues = _load(parquet_dir, "DmaQueuesInfo")

    if packets is None and aggregated is None and usage is None and pending is None:
        return

    print(f"\n  ── DMA Packet Detail ───────────────────────────")

    if packets is not None and not packets.empty:
        packet_sizes = _numeric_series(packets, "transfer_bytes")
        packet_durations = _numeric_series(packets, "duration_ns")
        packet_throughput = _numeric_series(packets, "throughput")

        print(f"  packet_rows         = {len(packets)}")
        _print_distribution("packet_size", packet_sizes, _fmt_bytes)
        _print_distribution("packet_duration", packet_durations, _fmt_ns)
        _print_distribution("packet_throughput", packet_throughput, _fmt_rate)

        if {"queue_type", "transfer_bytes"}.issubset(packets.columns):
            by_queue = (
                packets.assign(
                    transfer_bytes_num=pd.to_numeric(
                        packets["transfer_bytes"], errors="coerce"
                    ).fillna(0)
                )
                .groupby("queue_type")
                .agg(
                    packet_count=("transfer_bytes_num", "size"),
                    total_bytes=("transfer_bytes_num", "sum"),
                    avg_bytes=("transfer_bytes_num", "mean"),
                )
                .sort_values("total_bytes", ascending=False)
            )
            if not by_queue.empty:
                print(f"\n  queue_type breakdown:")
                for queue_type, row in by_queue.iterrows():
                    print(f"  {str(queue_type):<18} packets={int(row['packet_count']):>6}"
                          f"  bytes={_fmt_bytes(row['total_bytes']):>10}"
                          f"  avg={_fmt_bytes(row['avg_bytes'])}")

    if aggregated is not None and not aggregated.empty:
        agg_sizes = _numeric_series(aggregated, "transfer_bytes")
        agg_durations = _numeric_series(aggregated, "duration_ns")
        agg_throughput = _numeric_series(aggregated, "throughput")

        print(f"\n  aggregated transfers = {len(aggregated)}")
        _print_distribution("agg_size", agg_sizes, _fmt_bytes)
        _print_distribution("agg_duration", agg_durations, _fmt_ns)
        _print_distribution("agg_throughput", agg_throughput, _fmt_rate)

        columns = [c for c in [
            "queue_name", "queue_type", "op", "variable", "transfer_bytes",
            "duration_ns", "throughput", "source", "dest"
        ] if c in aggregated.columns]
        top_dma = aggregated[columns].copy()
        if "transfer_bytes" in top_dma.columns:
            top_dma["transfer_bytes"] = pd.to_numeric(
                top_dma["transfer_bytes"], errors="coerce"
            ).fillna(0)
            top_dma = top_dma.sort_values("transfer_bytes", ascending=False).head(5)
            if not top_dma.empty:
                print(f"\n  top aggregated transfers:")
                for _, row in top_dma.iterrows():
                    queue_name = row.get("queue_name", "?")
                    queue_type = row.get("queue_type", "?")
                    op = row.get("op", "?")
                    variable = row.get("variable", "?")
                    size = _fmt_bytes(row.get("transfer_bytes", 0))
                    duration = _fmt_ns(row.get("duration_ns", 0))
                    throughput = _fmt_rate(row.get("throughput", 0))
                    source = str(row.get("source", "?"))
                    dest = str(row.get("dest", "?"))
                    print(f"  {queue_name:<12} {queue_type:<8} {op:<8} {variable:<18}"
                          f" size={size:<10} dur={duration:<10} thr={throughput}")
                    print(f"  source={source}  dest={dest}")

    if usage is not None and not usage.empty:
        throughput = _numeric_series(usage, "current_throughput")
        transfer_bytes = _numeric_series(usage, "total_transfer_bytes")
        if not throughput.empty or not transfer_bytes.empty:
            print(f"\n  dma pressure:")
        if not throughput.empty:
            print(f"  sampled_throughput  avg={_fmt_rate(throughput.mean())}"
                  f"  p95={_fmt_rate(throughput.quantile(0.95))}"
                  f"  peak={_fmt_rate(throughput.max())}")
        if not transfer_bytes.empty:
            print(f"  sampled_bytes       avg={_fmt_bytes(transfer_bytes.mean())}"
                  f"  p95={_fmt_bytes(transfer_bytes.quantile(0.95))}"
                  f"  max={_fmt_bytes(transfer_bytes.max())}")

    if pending is not None and not pending.empty:
        depth = _numeric_series(pending, "value")
        if not depth.empty:
            print(f"  pending_dma_depth   avg={depth.mean():.1f}"
                  f"  p95={depth.quantile(0.95):.1f}"
                  f"  max={depth.max():.0f}")

    if queues is not None and not queues.empty:
        type_col = "type" if "type" in queues.columns else None
        if type_col is not None:
            queue_counts = queues.groupby(type_col).size().sort_values(ascending=False)
            print(f"\n  dma queues:")
            for queue_type, count in queue_counts.items():
                subset = queues[queues[type_col] == queue_type]
                channels = sorted(
                    int(v) for v in pd.to_numeric(
                        subset.get("dram_channel"), errors="coerce"
                    ).dropna().unique()
                )
                engines = sorted(
                    int(v) for v in pd.to_numeric(
                        subset.get("engine_idx"), errors="coerce"
                    ).dropna().unique()
                )
                print(f"  {str(queue_type):<18} queues={count:<3}"
                      f"  dram_channels={channels}"
                      f"  engines={engines[:6]}{'...' if len(engines) > 6 else ''}")


def _print_active_time(parquet_dir: str) -> None:
    df = _load(parquet_dir, "ActiveTime")
    if df is None or df.empty:
        return

    # Aggregate by engine type; ActiveTime has one row per instruction event
    engine_col = next((c for c in ("engine", "name") if c in df.columns), None)
    time_col = next(
        (c for c in ("active_time_ns", "active_time", "duration_ns", "duration")
         if c in df.columns), None
    )
    if engine_col is None or time_col is None:
        return

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0)
    by_engine = df.groupby(engine_col)[time_col].sum().sort_values(ascending=False)
    if by_engine.empty:
        return

    # Convert to μs if values look like nanoseconds (> 1000 for any entry)
    scale, unit = (1e-3, "μs") if by_engine.max() > 1000 else (1.0, "ns")
    print(f"\n  ── ActiveTime by Engine (aggregated) ───────────")
    for eng, val in by_engine.items():
        print(f"  {str(eng):<20} {val * scale:.2f} {unit}")


def _print_hbm_usage(parquet_dir: str) -> None:
    df = _load(parquet_dir, "HbmUsageSummaryByType")
    if df is None or df.empty:
        return

    # Keep only rows from neuroncore_idx 0 or the first NC, skip Total/Profiler rows
    if "neuroncore_idx" in df.columns:
        first_nc = df["neuroncore_idx"].min()
        df = df[df["neuroncore_idx"] == first_nc]

    skip_types = {"Total", "Profiler Buffers", "Shared Scratchpad", "Scratchpad",
                  "XT CC (unused)", "Collectives", "IO", "DRAM Spill",
                  "DMA Rings Collectives", "GpSimd STDIO"}
    if "usage_type" in df.columns:
        df = df[~df["usage_type"].isin(skip_types)]
        df = df[df.get("usage_bytes", df.iloc[:, -1]) > 0]

    if df.empty:
        return

    print(f"\n  ── HBM Usage by Type (NC{df.get('neuroncore_idx', [0]).iloc[0] if 'neuroncore_idx' in df.columns else 0}) ─────────────────")
    type_col = "usage_type" if "usage_type" in df.columns else df.columns[0]
    bytes_col = "usage_bytes" if "usage_bytes" in df.columns else df.columns[-1]
    for _, row in df.sort_values(bytes_col, ascending=False).iterrows():
        kib = row[bytes_col] / 1024
        print(f"  {str(row[type_col]):<28} {kib:>8.1f} KiB")


def _print_warnings(parquet_dir: str) -> None:
    df = _load(parquet_dir, "Warning")
    if df is None or df.empty:
        return

    print(f"\n  ── Compiler/Profiler Warnings ──────────────────")
    for _, row in df.iterrows():
        msg = row.get("message") or row.get("warning") or str(row.iloc[0])
        print(f"  ⚠  {msg}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_profile(parquet_dir: str) -> None:
    """Print all key profiling metrics from a neuron-profile parquet directory."""
    sep = "=" * 60
    label = os.path.basename(parquet_dir.rstrip("/"))
    print(f"\n{sep}")
    print(f"  Profile: {label}")
    print(f"  Path   : {parquet_dir}")
    print(sep)

    _print_summary(parquet_dir)
    _print_dma_packets(parquet_dir)
    _print_active_time(parquet_dir)
    _print_hbm_usage(parquet_dir)
    _print_warnings(parquet_dir)

    print(sep + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: print all profiles in parquet_files/
        base = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../parquet_files/profiles/global"
        )
        if not os.path.isdir(base):
            print(f"Usage: python parquet_reader.py <parquet_dir>")
            sys.exit(1)
        dirs = sorted(
            d for d in (os.path.join(base, n) for n in os.listdir(base))
            if os.path.isdir(d)
        )
        for d in dirs:
            print_profile(d)
    else:
        print_profile(sys.argv[1])
