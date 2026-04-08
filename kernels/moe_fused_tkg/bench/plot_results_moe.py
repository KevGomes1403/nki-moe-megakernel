"""
Generate benchmark charts from results/benchmark_results.json.

Creates:
  results/plots/runtime_by_version.png  — all kernels, log scale
  results/plots/best_per_round.png      — best device_time_us per round + speedup

Usage:
    python plot_results_moe.py [--results results/benchmark_results.json]
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BENCH_DIR, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# One color per round, cycling a colormap for the many rounds
def round_color(rnd, total_rounds=30):
    cmap = plt.cm.get_cmap("tab20", total_rounds)
    return cmap(rnd % total_rounds)


def load_results(path: str):
    with open(path) as f:
        data = json.load(f)
    return [r for r in data["results"] if r.get("status") == "ok"]


def plot_all_runtimes(results, out_path: str):
    labels = [r["label"] for r in results]
    times = [r["device_time_us"] for r in results]
    rounds = [r.get("round", 0) for r in results]
    colors = [round_color(rnd) for rnd in rounds]

    fig, ax = plt.subplots(figsize=(22, 7))
    x = np.arange(len(labels))

    bars = ax.bar(x, times, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_yscale("log")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f"{t:.0f}", ha="center", va="bottom", fontsize=5.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("Device Time (µs) — log scale", fontsize=11)
    ax.set_title(
        "MoE Fused TKG Kernel — Device Runtime by Version\n"
        "(Qwen3-30B-A3B, B=1, E=128, K=8, H=2048, I=192, trn2 LNC=2)",
        fontsize=12
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0, which="both")
    ax.set_axisbelow(True)

    seen_rounds = sorted(set(rounds))
    patches = [mpatches.Patch(color=round_color(r), label=f"Round {r}") for r in seen_rounds]
    ncol = max(1, len(patches) // 4)
    ax.legend(handles=patches, loc="upper right", fontsize=6, ncol=ncol)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _best_per_round_data(results):
    round_best: dict[int, dict] = {}
    for r in results:
        rnd = r.get("round", 0)
        if rnd not in round_best or r["device_time_us"] < round_best[rnd]["device_time_us"]:
            round_best[rnd] = r
    sorted_rounds = sorted(round_best.keys())
    labels  = [round_best[rnd]["label"] for rnd in sorted_rounds]
    times   = [round_best[rnd]["device_time_us"] for rnd in sorted_rounds]
    colors  = [round_color(rnd) for rnd in sorted_rounds]
    return sorted_rounds, labels, times, colors


def plot_best_per_round_runtime(results, out_path: str):
    sorted_rounds, labels, times, colors = _best_per_round_data(results)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(x, times, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    ax.plot(x, times, "o--", color="#333333", linewidth=1.2, markersize=4, zorder=4)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{rnd}\n({lbl})" for rnd, lbl in zip(sorted_rounds, labels)], fontsize=7)
    ax.set_ylabel("Device Time (µs)", fontsize=11)
    ax.set_title(
        "Best Kernel per Optimization Round — MoE Fused TKG\n"
        "(Qwen3-30B-A3B, B=1, E=128, K=8, H=2048, I=192, trn2 LNC=2)",
        fontsize=12
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_best_per_round_speedup(results, out_path: str):
    sorted_rounds, labels, times, colors = _best_per_round_data(results)
    x = np.arange(len(labels))
    baseline = times[0]
    speedups = [baseline / t for t in times]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, speedups, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    ax.plot(x, speedups, "o--", color="#333333", linewidth=1.2, markersize=4, zorder=4)
    for i, sp in enumerate(speedups):
        ax.text(i, sp + 0.02, f"{sp:.1f}×", ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{rnd}\n({lbl})" for rnd, lbl in zip(sorted_rounds, labels)], fontsize=7)
    ax.set_ylabel(f"Speedup vs Round {sorted_rounds[0]}", fontsize=11)
    ax.set_title(
        "Cumulative Speedup per Optimization Round — MoE Fused TKG\n"
        "(Qwen3-30B-A3B, B=1, E=128, K=8, H=2048, I=192, trn2 LNC=2)",
        fontsize=12
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",
                        default=os.path.join(BENCH_DIR, "results", "benchmark_results.json"))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        return

    results = load_results(args.results)
    print(f"Loaded {len(results)} successful benchmark results.")
    if not results:
        return

    plot_all_runtimes(results, os.path.join(PLOTS_DIR, "runtime_by_version.png"))
    plot_best_per_round_runtime(results, os.path.join(PLOTS_DIR, "best_per_round_runtime.png"))
    plot_best_per_round_speedup(results, os.path.join(PLOTS_DIR, "best_per_round_speedup.png"))
    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
