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
    return f"{v:.1f}%"


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
