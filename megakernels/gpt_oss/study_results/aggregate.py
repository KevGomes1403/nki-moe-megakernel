"""Aggregate per-cell JSON reports into a Markdown matrix.

Usage:  python aggregate.py [--out /path/to/study.md]
Scans study_results/b*_s*_{mega,xla}.json and prints a summary table to stdout.
With --out, replaces the auto-generated table block in megakernel_study.md.
"""
import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def load_cells():
    cells = []
    for path in sorted(glob.glob(os.path.join(HERE, "b*_s*_*.json"))):
        name = os.path.basename(path).replace(".json", "")
        m = re.match(r"b(\d+)_s(\d+)_(mega|xla)$", name)
        if not m:
            continue
        bs, seq, mode = int(m.group(1)), int(m.group(2)), m.group(3)
        try:
            with open(path) as f:
                rep = json.load(f)
        except Exception as e:
            cells.append({"bs": bs, "seq": seq, "mode": mode, "err": str(e)})
            continue
        tkg = rep.get("token_generation_model", {})
        cte = rep.get("context_encoding_model", {})
        e2e = rep.get("e2e_model", {})
        cells.append({
            "bs": bs, "seq": seq, "mode": mode,
            "tkg_p50": tkg.get("latency_ms_p50"),
            "tkg_p99": tkg.get("latency_ms_p99"),
            "cte_p50": cte.get("latency_ms_p50"),
            "e2e_p50": e2e.get("latency_ms_p50"),
            "e2e_p99": e2e.get("latency_ms_p99"),
            "e2e_thru": e2e.get("throughput"),
            "tkg_thru": tkg.get("throughput"),
        })
    return cells


def fmt(v, n=2):
    if v is None:
        return "—"
    return f"{v:.{n}f}"


def pair(cells, predicate):
    """Return list of (xla, mega) tuples matching predicate."""
    by_key = {}
    for c in cells:
        if predicate(c):
            by_key.setdefault((c["bs"], c["seq"]), {})[c["mode"]] = c
    out = []
    for key, modes in sorted(by_key.items()):
        out.append((key, modes.get("xla"), modes.get("mega")))
    return out


def render_table(rows, key_label):
    lines = []
    lines.append(f"| {key_label} | xla TKG p50 (ms) | mega TKG p50 (ms) | Δ p50 | "
                 f"mega TKG p99 (ms) | xla tok/s | mega tok/s | Δ throughput |")
    lines.append(f"|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (key, xla, mega) in rows:
        bs, seq = key
        klabel = f"b{bs}, s{seq}" if key_label == "config" else (
            f"{seq}" if "seq" in key_label else f"{bs}")
        xla_p50 = xla["tkg_p50"] if xla else None
        mega_p50 = mega["tkg_p50"] if mega else None
        mega_p99 = mega["tkg_p99"] if mega else None
        xla_thru = xla["tkg_thru"] if xla else None
        mega_thru = mega["tkg_thru"] if mega else None
        dpct = None
        if xla_p50 and mega_p50:
            dpct = 100 * (mega_p50 - xla_p50) / xla_p50
        dthru = None
        if xla_thru and mega_thru:
            dthru = 100 * (mega_thru - xla_thru) / xla_thru
        lines.append(
            f"| {klabel} | {fmt(xla_p50,3)} | {fmt(mega_p50,3)} | "
            f"{fmt(dpct,1)+'%' if dpct is not None else '—'} | "
            f"{fmt(mega_p99,3)} | "
            f"{fmt(xla_thru,1)} | {fmt(mega_thru,1)} | "
            f"{fmt(dthru,1)+'%' if dthru is not None else '—'} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="If set, overwrite the matrix block inside megakernel_study.md")
    args = ap.parse_args()

    cells = load_cells()
    print(f"# Loaded {len(cells)} cells")
    seq_sweep = pair(cells, lambda c: c["bs"] == 1)
    batch_sweep = pair(cells, lambda c: c["seq"] == 640)

    out_blocks = []
    out_blocks.append("### seq_len sweep at bs=1\n" +
                      render_table(seq_sweep, "seq_len"))
    out_blocks.append("### batch sweep at seq_len=640\n" +
                      render_table(batch_sweep, "batch"))
    table_md = "\n\n".join(out_blocks)
    print(table_md)

    if args.out:
        with open(args.out, "r") as f:
            doc = f.read()
        # Replace from "## Results matrix" to "## Findings"
        new = re.sub(
            r"## Results matrix.*?(?=^## Findings)",
            "## Results matrix (updated live)\n\n" + table_md + "\n\n",
            doc, flags=re.DOTALL | re.MULTILINE,
        )
        with open(args.out, "w") as f:
            f.write(new)
        print(f"\nUpdated {args.out}")


if __name__ == "__main__":
    main()
