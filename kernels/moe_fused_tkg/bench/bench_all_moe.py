"""
Master benchmark runner for all non-quantized MoE fused TKG kernels.

Runs bench_runner.py for each version as a subprocess, parses output,
saves to results/benchmark_results.json.

Usage:
    cd kernels/moe_fused_tkg/bench
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python bench_all_moe.py [--skip-existing] [--only v1a,v2]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ordered list: (version_label, round_num, description)
# Excludes quantized (in quantized/) and FP8 (v29_fp8)
KERNELS = [
    # Rounds 1-2: baseline optimizations
    ("v1a",          1,  "Persistent PSUM accumulation"),
    ("v1b",          1,  "Persistent PSUM variant B"),
    ("v2",           2,  "Round 2"),
    # Rounds 3-6: layout & routing optimizations
    ("v3a",          3,  "Round 3 Plan A"),
    ("v3b",          3,  "Round 3 Plan B"),
    ("v4a",          4,  "Round 4 Plan A"),
    ("v4b",          4,  "Round 4 Plan B"),
    ("v5a",          5,  "Round 5 Plan A"),
    ("v5b",          5,  "Round 5 Plan B"),
    ("v6a",          6,  "Round 6 Plan A"),
    ("v6b",          6,  "Round 6 Plan B"),
    ("v7a",          7,  "Round 7 Plan A"),
    # Rounds 8-9: DMA / weight layout
    ("v8a",          8,  "Round 8 Plan A"),
    ("v8b_original", 8,  "Round 8 Plan B (original)"),
    ("v9a",          9,  "Round 9 Plan A"),
    ("v9b",          9,  "Round 9 Plan B"),
    # Rounds 10-13: H-sharding, LNC-2 kernel
    ("v10a",         10, "Round 10 Plan A"),
    ("v10b",         10, "Round 10 Plan B"),
    ("v11c",         11, "Round 11 Plan C"),
    ("v11d",         11, "Round 11 Plan D"),
    ("v12e",         12, "Round 12 Plan E"),
    ("v12f",         12, "Round 12 Plan F"),
    ("v13",          13, "Round 13"),
    # Rounds 14-16: K-expert split
    ("v14a",         14, "K-expert split Plan A"),
    ("v14b",         14, "K-expert split I=192 native"),
    ("v14c",         14, "K-expert split Plan C"),
    ("v15",          15, "Round 15"),
    ("v16a",         16, "Round 16 Plan A"),
    ("v16b",         16, "Round 16 Plan B"),
    ("v16c",         16, "Round 16 Plan C"),
    # Rounds 17-21: pipeline / SBUF optimizations
    ("v17a",         17, "Round 17 Plan A"),
    ("v17b",         17, "Round 17 Plan B"),
    ("v18a",         18, "Round 18 Plan A"),
    ("v18b",         18, "Round 18 Plan B"),
    ("v19a",         19, "Round 19 Plan A"),
    ("v19b",         19, "Round 19 Plan B"),
    ("v20a",         20, "Round 20 Plan A"),
    ("v20b",         20, "Round 20 Plan B"),
    ("v21a",         21, "Round 21 Plan A"),
    ("v21b",         21, "Round 21 Plan B"),
    # Rounds 22-24: matmul restructuring
    ("v22a",         22, "K=64 matmul, no tensor_copy"),
    ("v22b",         22, "Round 22 Plan B"),
    ("v23a",         23, "Round 23 Plan A"),
    ("v23b",         23, "Round 23 Plan B"),
    ("v24a",         24, "Round 24 Plan A"),
    ("v24b",         24, "Round 24 Plan B"),
    # Round 25: multi-plan
    ("v25a",         25, "Round 25 Plan A"),
    ("v25b",         25, "Round 25 Plan B"),
    ("v25c",         25, "Round 25 Plan C"),
    ("v25d",         25, "Round 25 Plan D"),
    ("v25e",         25, "Round 25 Plan E"),
    # Round 26
    ("v26a",         26, "Round 26 Plan A"),
    ("v26b",         26, "Round 26 Plan B"),
    ("v26c",         26, "Round 26 Plan C"),
    ("v26d",         26, "Round 26 Plan D"),
    ("v26e",         26, "Round 26 Plan E"),
    ("v26f",         26, "Round 26 Plan F"),
    ("v26g",         26, "Round 26 Plan G"),
    # Round 27
    ("v27a",         27, "Round 27 Plan A"),
    ("v27b",         27, "Round 27 Plan B"),
    ("v27c",         27, "Round 27 Plan C"),
    ("v27d",         27, "Round 27 Plan D"),
    ("v27d_fixed",   27, "Round 27 Plan D (fixed)"),
    ("v27e",         27, "Round 27 Plan E"),
    ("v27f",         27, "Round 27 Plan F"),
    ("v27g",         27, "Round 27 Plan G"),
    ("v27h",         27, "Round 27 Plan H"),
    ("v27i",         27, "Round 27 Plan I"),
    # Round 28
    ("v28a",         28, "Round 28 Plan A: K=64 matmul"),
    ("v28b",         28, "Round 28 Plan B"),
    ("v28c",         28, "Round 28 Plan C"),
    ("v28d",         28, "Round 28 Plan D"),
    ("v28e",         28, "Round 28 Plan E"),
    ("v28f",         28, "Round 28 Plan F"),
    ("v28g",         28, "Round 28 Plan G"),
    ("v28h",         28, "Round 28 Plan H"),
    ("v28i",         28, "Round 28 Plan I"),
    # Round 29 (non-FP8)
    ("v29j",         29, "Round 29 Plan J"),
    ("v29k",         29, "Round 29 Plan K"),
    ("v29m",         29, "Round 29 Plan M"),
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

RUNNER = os.path.join(BENCH_DIR, "bench_runner.py")


def parse_output(stdout: str) -> dict:
    metrics = {}
    for key, pattern in METRIC_RE.items():
        m = pattern.search(stdout)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def run_bench(version: str, timeout: int = 600) -> dict:
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {version}")
    print(f"{'='*60}")
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, RUNNER, version],
            capture_output=True, text=True,
            timeout=timeout, cwd=BENCH_DIR,
        )
        elapsed = time.time() - start
        stdout = result.stdout
        print(stdout)
        if result.returncode != 0:
            print(f"[STDERR] {result.stderr[-2000:]}")
            return {"status": "failed", "label": version,
                    "returncode": result.returncode,
                    "stderr_tail": result.stderr[-500:], "elapsed_s": elapsed}
        metrics = parse_output(stdout)
        if not metrics.get("device_time_us"):
            return {"status": "no_metrics", "label": version,
                    "stdout_tail": stdout[-300:], "elapsed_s": elapsed}
        return {"status": "ok", "label": version, "elapsed_s": elapsed, **metrics}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "label": version, "timeout_s": timeout}
    except Exception as e:
        return {"status": "error", "label": version, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only", type=str, default="")
    args = parser.parse_args()

    results_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    existing = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        existing = {r["label"]: r for r in data.get("results", [])}

    only_set = set(args.only.split(",")) if args.only else set()

    to_run = []
    for version, round_num, desc in KERNELS:
        if only_set and version not in only_set:
            continue
        if args.skip_existing and version in existing and existing[version].get("status") == "ok":
            print(f"Skipping {version} (already have results)")
            continue
        to_run.append((version, round_num, desc))

    print(f"\nRunning {len(to_run)} benchmarks...")

    all_results = dict(existing)
    for version, round_num, desc in to_run:
        rec = run_bench(version)
        rec["round"] = round_num
        rec["description"] = desc
        all_results[version] = rec

        ordered = [all_results[v] for v, _, _ in KERNELS if v in all_results]
        with open(results_path, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "results": ordered}, f, indent=2)
        print(f"  -> Saved to {results_path}")

    ok = [r for r in all_results.values() if r.get("status") == "ok"]
    failed = [r for r in all_results.values() if r.get("status") != "ok"]
    print(f"\nDone. {len(ok)} succeeded, {len(failed)} failed/skipped.")
    if ok:
        times = sorted([(r["label"], r["device_time_us"]) for r in ok], key=lambda x: x[1])
        print(f"Fastest: {times[0][0]} = {times[0][1]:.2f} us")
        print(f"Slowest: {times[-1][0]} = {times[-1][1]:.2f} us")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
