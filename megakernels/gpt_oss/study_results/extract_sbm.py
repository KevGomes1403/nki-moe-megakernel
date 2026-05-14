"""Extract SBM (state-buffer memory) statistics from compile logs.

For each cell log, summarize peak SBM usage per kernel name. Writes CSV.
"""
import argparse
import glob
import os
import re
from collections import defaultdict


def parse_log(path):
    """Return {kernel_name: peak_pct} from compile log."""
    peak = defaultdict(float)
    peak_b = defaultdict(int)
    alloc_b = defaultdict(int)
    init_re = re.compile(r"\[INFO\] \[(\w+)\] SBM initialized: range=\[0, (\d+)\)")
    stats_re = re.compile(
        r"\[INFO\] \[(\w+)\] \[SBM\] SB memory statistics: max_usage=(\d+) B \((\d+)%\)"
    )
    with open(path) as f:
        for line in f:
            m = init_re.search(line)
            if m:
                k, size = m.group(1), int(m.group(2))
                alloc_b[k] = max(alloc_b[k], size)
            m = stats_re.search(line)
            if m:
                k = m.group(1)
                b, p = int(m.group(2)), int(m.group(3))
                if p > peak[k]:
                    peak[k] = p
                    peak_b[k] = b
    return peak, peak_b, alloc_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="/home/ubuntu/models/study/logs")
    ap.add_argument("--out", default="/home/ubuntu/nki-moe/megakernels/gpt_oss/study_results/sbm_summary.csv")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.log_dir, "*.log"))):
        name = os.path.basename(path).replace(".log", "")
        if name.endswith("_bench"):
            continue
        peak, peak_b, alloc = parse_log(path)
        for k in sorted(peak):
            rows.append((name, k, peak_b[k], alloc[k], peak[k]))

    with open(args.out, "w") as f:
        f.write("cell,kernel,peak_usage_B,alloc_B,peak_pct\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out}")

    # Also print to stdout, focus on key kernels
    key_kernels = (
        "transformer_gpt_oss_tkg_multilayer",
        "selective_expert_moe_tkg",
        "attention_block_tkg",
        "rmsnorm_tkg",
    )
    print()
    print(f"{'cell':<28} {'kernel':<40} {'peak_B':>8} {'alloc_B':>8} {'pct':>4}")
    for (cell, k, peak_b, alloc, pct) in rows:
        if any(kk in k for kk in key_kernels):
            print(f"{cell:<28} {k:<40} {peak_b:>8} {alloc:>8} {int(pct):>3}%")


if __name__ == "__main__":
    main()
