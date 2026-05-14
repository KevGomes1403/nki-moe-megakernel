"""Extract compile time from each cell log."""
import glob
import os
import re

LOG_DIR = "/home/ubuntu/models/study/logs"

rows = []
for path in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
    name = os.path.basename(path).replace(".log", "")
    if name.endswith("_bench"):
        continue
    compile_s = None
    priority_s = None
    hlo_s = None
    with open(path) as f:
        for line in f:
            m = re.search(r"Compile took ([\d.]+)s", line)
            if m:
                compile_s = float(m.group(1))
            m = re.search(r"Done compilation for the priority HLO in ([\d.]+)", line)
            if m:
                priority_s = float(m.group(1))
            m = re.search(r"Generated all HLOs in ([\d.]+)", line)
            if m:
                hlo_s = float(m.group(1))
    rows.append((name, compile_s, hlo_s, priority_s))

print(f"{'cell':<25} {'total_s':>9} {'hlo_s':>7} {'priority_s':>10}")
for r in rows:
    name, total, hlo, pri = r
    print(f"{name:<25} {str(total):>9} {str(hlo):>7} {str(pri):>10}")
