#!/usr/bin/env python3

"""
nki_kernel_lint.py

Best-effort static checks for common NKI-kernel issues.
This is intentionally lightweight and does NOT require Neuron hardware.

Usage:
  python scripts/nki_kernel_lint.py path/to/kernel_file.py

Checks:
- presence of SKILL-like kernel patterns: nl.ndarray + nl.store + return
- warns about affine_range usage (manual review needed)
- warns if output tensors allocated in sbuf without store to hbm

This script is heuristic: it reduces obvious mistakes; it cannot prove correctness.
"""

import ast
import sys
from pathlib import Path

def find_calls(tree, func_names):
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # handle nl.store(...) and store(...)
            if isinstance(node.func, ast.Attribute):
                name = f"{getattr(node.func.value, 'id', '')}.{node.func.attr}"
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            else:
                continue
            if name in func_names:
                calls.append((name, node.lineno))
    return calls

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/nki_kernel_lint.py path/to/kernel_file.py")
        sys.exit(2)

    path = Path(sys.argv[1])
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    ndarray_calls = find_calls(tree, {"nl.ndarray", "ndarray"})
    store_calls = find_calls(tree, {"nl.store", "store"})
    affine_calls = find_calls(tree, {"nl.affine_range", "affine_range"})
    bench_calls = find_calls(tree, {"benchmark", "nki.benchmark"})

    ok = True

    if not ndarray_calls:
        print("WARN: no nl.ndarray(...) allocations found (may be fine if using ptr/view).")
    if not store_calls:
        print("WARN: no nl.store(...) found. Most kernels must store outputs explicitly.")
        ok = False

    if affine_calls:
        lines = ", ".join(str(l) for _, l in affine_calls)
        print(f"NOTE: nl.affine_range used at lines: {lines}. Verify loop iterations are independent.")

    if bench_calls:
        print("NOTE: benchmark decorator detected. Remember: benchmark output is not correctness-validated.")

    if ok:
        print("Lint finished: no blocking issues detected.")
    else:
        print("Lint finished: blocking warnings present.")
        sys.exit(1)

if __name__ == "__main__":
    main()
