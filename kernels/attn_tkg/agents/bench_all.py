"""
Master benchmark runner for all attn_tkg agent-generated kernels.

Runs each bench script as a subprocess, parses output, and saves
results to results/benchmark_results.json.

Usage:
    cd kernels/attn_tkg/agents
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python bench_all.py [--skip-existing] [--only v6_ultimate,v7_combined]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(AGENTS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ordered list of (version_label, bench_script, round_num, description)
# round_num groups variants from the same optimization round for the chart
KERNELS = [
    # Round 6 — first oproj integration
    ("v6_ultimate",    "bench_v6_ultimate.py",    6,  "Fused o_proj baseline"),
    # Round 7 — DMA transpose fix
    ("v7_combined",    "bench_v7_combined.py",    7,  "Corrected DMA transpose"),
    # Round 8 — contiguous Wo DMA
    ("v8_wo_transposed","bench_v8_wo_transposed.py", 8, "Contiguous Wo DMA (caller transpose)"),
    # Round 9 — broadcast & cached scores
    ("v9_broadcasts",  "bench_v9_broadcasts.py",  9,  "Broadcast GQA + cached scores"),
    ("v9_optimized",   "bench_v9_optimized.py",   9,  "tp_broadcast + saved scores"),
    ("v9_tp_broadcast","bench_v9_tp_broadcast.py",9,  "tp_broadcast all GQA loops"),
    # Round 10 — affine_range DMA pipelines
    ("v10a",           "bench_v10a.py",           10, "affine_range DMA + contiguous Wo"),
    ("v10b",           "bench_v10b.py",           10, "DMA pipeline v10b"),
    ("v10c",           "bench_v10c.py",           10, "dma_transpose K-cache"),
    ("v10d",           "bench_v10d.py",           10, "DMA pipeline v10d"),
    ("v10e",           "bench_v10e.py",           10, "Position threshold masking"),
    # Round 11 — SBUF hoisting
    ("v11a",           "bench_v11a.py",           11, "SBUF hoisting"),
    ("v11b",           "bench_v11b.py",           11, "Deferred Wo DMA pipeline"),
    # Round 12 — dma_transpose K-cache (LNC=2)
    ("v12a",           "bench_v12a.py",           12, "dma_transpose K-cache"),
    ("v12b",           "bench_v12b.py",           12, "K-cache layout opt"),
    ("v12c",           "bench_v12c.py",           12, "V-cache dma_transpose"),
    ("v12d",           "bench_v12d.py",           12, "KV-cache dma_transpose"),
    ("v12e",           "bench_v12e.py",           12, "Fused KV transpose"),
    # Round 13 — LNC-2 sharding plans
    ("v13_planA",      "bench_v13_planA.py",      13, "Plan A: LNC-2 KV sharding"),
    ("v13_planB",      "bench_v13_planB.py",      13, "Plan B: LNC-2 Q sharding"),
    ("v13_planC",      "bench_v13_planC.py",      13, "Plan C: LNC-2 pipeline"),
    ("v13_BC",         "bench_v13_BC.py",         13, "Plan BC: combined B+C"),
    # Round 14 — output projection sharding
    ("v14_planD",      "bench_v14_planD.py",      14, "Plan D: LNC-2 Wo sharding"),
    ("v14_planE",      "bench_v14_planE.py",      14, "Plan E: Wo sharding variant"),
    ("v14_planG",      "bench_v14_planG.py",      14, "Plan G: pipeline overlap"),
    ("v14_planH",      "bench_v14_planH.py",      14, "Plan H: fused pipeline"),
    ("v14_planI",      "bench_v14_planI.py",      14, "Plan I: combined pipeline"),
]

METRIC_RE = {
    "device_time_us":    re.compile(r"device_time_us\s*=\s*([\d.]+)"),
    "tensor_engine_pct": re.compile(r"tensor_engine_pct\s*=\s*([\d.]+)%"),
    "dma_active_pct":    re.compile(r"dma_active_pct\s*=\s*([\d.]+)%"),
    "spill_bytes":       re.compile(r"spill_bytes\s*=\s*(\d+)"),
    "mfu_estimated":     re.compile(r"mfu_estimated\s*=\s*([\d.]+)%"),
    "hbm_read_KiB":      re.compile(r"hbm_read_KiB\s*=\s*([\d.]+)"),
    "hbm_write_KiB":     re.compile(r"hbm_write_KiB\s*=\s*([\d.]+)"),
}


def parse_output(stdout: str) -> dict:
    metrics = {}
    for key, pattern in METRIC_RE.items():
        m = pattern.search(stdout)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def run_bench(label: str, script: str, timeout: int = 600) -> dict:
    script_path = os.path.join(AGENTS_DIR, script)
    if not os.path.exists(script_path):
        return {"status": "missing", "label": label, "script": script}

    print(f"\n{'='*60}")
    print(f"  Benchmarking: {label}  ({script})")
    print(f"{'='*60}")
    start = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=AGENTS_DIR,
        )
        elapsed = time.time() - start
        stdout = result.stdout
        stderr = result.stderr

        print(stdout)
        if result.returncode != 0:
            print(f"[STDERR] {stderr[-2000:]}")
            return {
                "status": "failed",
                "label": label,
                "returncode": result.returncode,
                "stderr_tail": stderr[-500:],
                "elapsed_s": elapsed,
            }

        metrics = parse_output(stdout)
        if not metrics.get("device_time_us"):
            return {
                "status": "no_metrics",
                "label": label,
                "stdout_tail": stdout[-300:],
                "elapsed_s": elapsed,
            }

        return {
            "status": "ok",
            "label": label,
            "elapsed_s": elapsed,
            **metrics,
        }

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "label": label, "timeout_s": timeout}
    except Exception as e:
        return {"status": "error", "label": label, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip kernels already in results JSON")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated list of labels to run")
    args = parser.parse_args()

    results_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    existing = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        existing = {r["label"]: r for r in data.get("results", [])}

    only_set = set(args.only.split(",")) if args.only else set()

    to_run = []
    for label, script, round_num, desc in KERNELS:
        if only_set and label not in only_set:
            continue
        if args.skip_existing and label in existing and existing[label].get("status") == "ok":
            print(f"Skipping {label} (already have results)")
            continue
        to_run.append((label, script, round_num, desc))

    print(f"\nRunning {len(to_run)} benchmarks...")
    print("Estimated time: ~{:.0f} min (assuming ~2 min/kernel)\n".format(len(to_run) * 2))

    all_results = dict(existing)  # preserve existing
    for label, script, round_num, desc in to_run:
        rec = run_bench(label, script)
        rec["round"] = round_num
        rec["description"] = desc
        all_results[label] = rec

        # Save after each run so partial results are preserved
        ordered = [all_results[k] for k in [lbl for lbl, _, _, _ in KERNELS] if k in all_results]
        with open(results_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": ordered,
            }, f, indent=2)
        print(f"  -> Saved to {results_path}")

    # Summary
    ok = [r for r in all_results.values() if r.get("status") == "ok"]
    failed = [r for r in all_results.values() if r.get("status") not in ("ok",)]
    print(f"\n{'='*60}")
    print(f"Done. {len(ok)} succeeded, {len(failed)} failed/skipped.")
    if ok:
        times = [(r["label"], r["device_time_us"]) for r in ok]
        times.sort(key=lambda x: x[1])
        print(f"Fastest: {times[0][0]} = {times[0][1]:.2f} us")
        print(f"Slowest: {times[-1][0]} = {times[-1][1]:.2f} us")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
