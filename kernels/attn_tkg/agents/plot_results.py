"""
Generate benchmark chart from results/benchmark_results.json.

Creates:
  results/plots/runtime_by_version.png  — device_time_us for all kernels
  results/plots/best_per_round.png      — best (min) device_time_us per round
  results/plots/metrics_heatmap.png     — tensor_engine% / dma% / spill by version

Usage:
    python plot_results.py [--results results/benchmark_results.json]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(AGENTS_DIR, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

ROUND_COLORS = {
    6:  "#d62728",
    7:  "#ff7f0e",
    8:  "#bcbd22",
    9:  "#2ca02c",
    10: "#17becf",
    11: "#1f77b4",
    12: "#9467bd",
    13: "#8c564b",
    14: "#e377c2",
}


def load_results(path: str):
    with open(path) as f:
        data = json.load(f)
    return [r for r in data["results"] if r.get("status") == "ok"]


def plot_all_runtimes(results, out_path: str):
    labels = [r["label"] for r in results]
    times = [r["device_time_us"] for r in results]
    rounds = [r.get("round", 0) for r in results]
    colors = [ROUND_COLORS.get(rnd, "#888888") for rnd in rounds]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(labels))

    bars = ax.bar(x, times, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yscale("log")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f"{t:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Device Time (µs) — log scale", fontsize=11)
    ax.set_title("Attention TKG Kernel — Device Runtime by Version\n(Qwen3-30B-A3B, B=1, S=640, trn2 LNC=2)", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0, which="both")
    ax.set_axisbelow(True)

    seen_rounds = sorted(set(rounds))
    patches = [mpatches.Patch(color=ROUND_COLORS.get(r, "#888888"), label=f"Round {r}")
               for r in seen_rounds]
    ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_best_per_round(results, out_path: str):
    from collections import defaultdict
    round_best: dict[int, dict] = {}
    for r in results:
        rnd = r.get("round", 0)
        if rnd not in round_best or r["device_time_us"] < round_best[rnd]["device_time_us"]:
            round_best[rnd] = r

    sorted_rounds = sorted(round_best.keys())
    labels = [round_best[rnd]["label"] for rnd in sorted_rounds]
    times = [round_best[rnd]["device_time_us"] for rnd in sorted_rounds]
    colors = [ROUND_COLORS.get(rnd, "#888888") for rnd in sorted_rounds]

    # Compute speedup vs round 6
    baseline = times[0]
    speedups = [baseline / t for t in times]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    x = np.arange(len(labels))

    # Top: device time
    bars = ax1.bar(x, times, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax1.plot(x, times, "o--", color="#333333", linewidth=1.2, markersize=5, zorder=4)
    for bar, t, lbl in zip(bars, times, labels):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{t:.0f}µs", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Device Time (µs)", fontsize=11)
    ax1.set_title("Best Kernel per Optimization Round — Attention TKG\n(Qwen3-30B-A3B, B=1, S=640, trn2 LNC=2)", fontsize=12)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax1.set_axisbelow(True)

    # Bottom: cumulative speedup
    ax2.bar(x, speedups, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax2.plot(x, speedups, "o--", color="#333333", linewidth=1.2, markersize=5, zorder=4)
    for i, (sp, lbl) in enumerate(zip(speedups, labels)):
        ax2.text(i, sp + 0.01, f"{sp:.2f}×", ha="center", va="bottom", fontsize=8)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("Speedup vs Round 6", fontsize=10)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax2.set_axisbelow(True)

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"Round {rnd}\n({lbl})" for rnd, lbl in zip(sorted_rounds, labels)],
        fontsize=8
    )

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_metrics_heatmap(results, out_path: str):
    labels = [r["label"] for r in results]
    metrics = {
        "tensor_engine_pct": [r.get("tensor_engine_pct", 0) for r in results],
        "dma_active_pct":    [r.get("dma_active_pct", 0) for r in results],
        "mfu_estimated":     [r.get("mfu_estimated", 0) for r in results],
    }

    data = np.array([metrics[k] for k in metrics])
    metric_labels = list(metrics.keys())

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(["Tensor Engine %", "DMA Active %", "MFU %"], fontsize=10)
    ax.set_title("Hardware Utilization Heatmap by Kernel Version", fontsize=12)

    for i in range(len(metric_labels)):
        for j in range(len(labels)):
            val = data[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=6.5, color="black" if val < 60 else "white")

    plt.colorbar(im, ax=ax, label="% utilization", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(AGENTS_DIR, "results", "benchmark_results.json"))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        print("Run bench_all.py first.")
        return

    results = load_results(args.results)
    print(f"Loaded {len(results)} successful benchmark results.")

    if not results:
        print("No successful results to plot.")
        return

    plot_all_runtimes(results, os.path.join(PLOTS_DIR, "runtime_by_version.png"))
    plot_best_per_round(results, os.path.join(PLOTS_DIR, "best_per_round.png"))
    plot_metrics_heatmap(results, os.path.join(PLOTS_DIR, "metrics_heatmap.png"))

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
