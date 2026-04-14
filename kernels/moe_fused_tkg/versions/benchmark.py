"""
General-purpose NKI benchmark harness for nki beta 2.

All kernel code uses nki.* namespace (not neuronxcc). Benchmarking data
comes from neuron-profile (device metrics from NTFF captured at runtime)
and neuron-bench (end-to-end latency, run at script exit after XLA releases
the NeuronCores).

REQUIRED: Set these env vars BEFORE any neuron/torch_xla imports:

    import os
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
    os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
    os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "/abs/path/to/output"

Two usage styles:

  Style A — wrap after @nki.jit:

      @nki.jit(platform_target="trn2")
      def my_kernel(a, b): ...

      my_kernel = wrap_benchmark(my_kernel, warmup=5, iters=50)
      my_kernel(a, b)   # prints device metrics immediately;
                        # neuron-bench latency printed at script exit

  Style B — one-shot:

      result = nki_benchmark(my_kernel, a, b, warmup=5, iters=50)
"""

from __future__ import annotations

import atexit
import glob
import json
import os
import subprocess
import tempfile
import time
from functools import wraps

import torch_xla.core.xla_model as xm


# ---------------------------------------------------------------------------
# Global state: queue of pending neuron-bench runs (executed at atexit)
# ---------------------------------------------------------------------------

_pending_bench: list[dict] = []   # list of {neff, warmup, iters, name}


def _run_all_pending_bench():
    """
    atexit handler: print neuron-bench commands for manual execution.
    We don't run neuron-bench here because XLA still holds the NeuronCores
    during atexit. Run the printed commands after this script finishes.
    """
    if not _pending_bench:
        return
    print("\n" + "=" * 60)
    print("  neuron-bench commands (run after this script exits):")
    print("=" * 60)
    for job in _pending_bench:
        print(f"\n  [{job['name']}]")
        print(f"    neuron-bench exec -w {job['warmup']} -n {job['iters']}"
              f" -o /tmp/bench_out '{job['neff']}'")
    print()


atexit.register(_run_all_pending_bench)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_COMPILE_WORKDIR = "/tmp/ubuntu/neuroncc_compile_workdir"


def _snapshot_neffs(directory: str) -> dict[str, float]:
    neffs = glob.glob(f"{directory}/**/*.neff", recursive=True)
    return {p: os.path.getmtime(p) for p in neffs}


def _find_new_neffs(directory: str, before: dict[str, float]) -> list[str]:
    neffs = glob.glob(f"{directory}/**/*.neff", recursive=True)
    return [p for p in neffs if before.get(p) != os.path.getmtime(p)]


def _find_kernel_artifacts(
    before_inspect: dict, before_compile: dict, ntff_wait_s: float = 5.0
) -> tuple[str | None, str | None, str | None]:
    """
    Return (bench_neff, profile_neff, ntff_path) for the kernel that just ran.

    bench_neff:   NEFF from compile workdir — works with neuron-bench exec.
    profile_neff: NEFF from NEURON_RT_INSPECT_OUTPUT_DIR — paired with NTFF.
    ntff:         NTFF captured at runtime — parsed for device metrics.

    The NRT writes the NTFF asynchronously; polls up to ntff_wait_s seconds.
    """
    inspect_dir = os.environ.get("NEURON_RT_INSPECT_OUTPUT_DIR", "")
    profile_neff, ntff = None, None

    if inspect_dir:
        new_neffs = _find_new_neffs(inspect_dir, before_inspect)
        if new_neffs:
            profile_neff = max(new_neffs, key=os.path.getmtime)
            # neff basename looks like "neff_449630869622063_vnc_3.neff"
            # Strip prefix/suffix to get the numeric hash, then drop the _vnc_N part.
            neff_stem = (
                os.path.basename(profile_neff)
                .replace("neff_", "")
                .replace(".neff", "")
            )
            neff_hash = neff_stem.split("_vnc_")[0]
            ntff_pattern = f"{os.path.dirname(profile_neff)}/{neff_hash}_vnc_*.ntff"
            deadline = time.monotonic() + ntff_wait_s
            while time.monotonic() < deadline:
                candidates = glob.glob(ntff_pattern)
                if candidates:
                    ntff = candidates[0]
                    break
                time.sleep(0.2)

    # Compile workdir NEFF: this is the PyTorch/XLA-compiled NEFF that works
    # with neuron-bench exec (the inspect-dir NEFF is runtime-captured and
    # cannot be used with neuron-bench).
    new_compile = _find_new_neffs(_COMPILE_WORKDIR, before_compile)
    bench_neff = max(new_compile, key=os.path.getmtime) if new_compile else None

    return bench_neff, profile_neff, ntff


def _parse_ntff(neff: str, ntff: str) -> dict:
    """
    Run neuron-profile view --output-format summary-json on a captured NTFF.
    Returns the summary dict. No hardware needed — parses the trace file.
    """
    ret = subprocess.run(
        [
            "neuron-profile", "view",
            "--neff-path", neff,
            "--session-file", ntff,
            "--output-format", "summary-json",
        ],
        capture_output=True, text=True,
    )
    if ret.returncode != 0 or not ret.stdout.strip():
        return {}
    raw = json.loads(ret.stdout)
    return next(iter(raw.values()), {})




def _pct_of_total(value_s: float, total_s: float) -> float:
    """Percentage of value relative to total_time (not the raw trace window)."""
    return (value_s / total_s * 100) if total_s > 0 else 0.0


def _print_profile_results(prof: dict, name: str) -> None:
    """Print device metrics extracted from NTFF via neuron-profile view."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Kernel: {name}  (device metrics from NTFF)")
    print(sep)

    if not prof:
        print("  [no profile data available]")
        print(sep)
        return

    total_s  = prof.get("total_time", 0)
    total_us = total_s * 1e6
    active_s = prof.get("total_active_time", 0)
    active_us = active_s * 1e6

    def _us(key: str) -> float:
        return prof.get(key, 0) * 1e6

    def _pct(key_s: str) -> str:
        return f"{_pct_of_total(prof.get(key_s, 0), total_s):.1f}%"

    print(f"\n  Execution time:")
    print(f"    total_time     = {total_us:.2f} μs  (hardware trace)")
    print(f"    active_time    = {active_us:.2f} μs"
          f"  ({_pct_of_total(active_s, total_s):.1f}% of total)")

    print(f"\n  Engine utilization (% of total_time):")
    print(f"    tensor_engine  = {_pct('tensor_engine_active_time')}"
          f"  ({_us('tensor_engine_active_time'):.2f} μs)")
    print(f"    vector_engine  = {_pct('vector_engine_active_time')}"
          f"  ({_us('vector_engine_active_time'):.2f} μs)")
    print(f"    scalar_engine  = {_pct('scalar_engine_active_time')}"
          f"  ({_us('scalar_engine_active_time'):.2f} μs)")
    print(f"    dma_active     = {_pct('dma_active_time')}"
          f"  ({_us('dma_active_time'):.2f} μs)")

    print(f"\n  Compute efficiency:")
    print(f"    mfu_estimated  = {prof.get('mfu_estimated_percent', 0):.2f}%")
    print(f"    mbu_estimated  = {prof.get('mbu_estimated_percent', 0):.2f}%")
    print(f"    mm_intensity   = {prof.get('mm_arithmetic_intensity', 0):.3f}")

    print(f"\n  Memory traffic:")
    print(f"    hbm_read       = {prof.get('hbm_read_bytes', 0)/1024:.1f} KiB")
    print(f"    hbm_write      = {prof.get('hbm_write_bytes', 0)/1024:.1f} KiB")
    spill = prof.get("spill_save_bytes", 0) + prof.get("spill_reload_bytes", 0)
    print(f"    spill bytes    = {spill}")

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BenchmarkResult:
    """Holds device metrics from NTFF. neuron-bench results printed at atexit."""

    def __init__(self, prof: dict, neff: str, ntff: str | None):
        self.prof = prof
        self.neff_path = neff
        self.ntff_path = ntff

    @property
    def device_time_us(self) -> float:
        """Actual device execution time in μs from hardware trace."""
        return self.prof.get("total_time", 0) * 1e6

    @property
    def tensor_engine_pct(self) -> float:
        return self.prof.get("tensor_engine_active_time_percent", 0)

    @property
    def dma_active_pct(self) -> float:
        return self.prof.get("dma_active_time_percent", 0)

    @property
    def spill_bytes(self) -> int:
        return (self.prof.get("spill_save_bytes", 0)
                + self.prof.get("spill_reload_bytes", 0))


def wrap_benchmark(
    jit_kernel,
    *,
    warmup: int = 10,
    iters: int = 100,
) -> callable:
    """
    Wrap an already-@nki.jit-compiled kernel with benchmarking.

    On first call:
      - Runs the kernel (compiles on first call, cached thereafter)
      - Parses the NTFF captured by NEURON_RT_INSPECT_DEVICE_PROFILE and
        prints device metrics immediately
      - Queues neuron-bench exec (for end-to-end latency) to run at atexit,
        after XLA releases the NeuronCores

    IMPORTANT: Apply AFTER @nki.jit to avoid confusing the kernel rewriter:

        @nki.jit(platform_target="trn2")
        def my_kernel(a, b): ...

        my_kernel = wrap_benchmark(my_kernel, warmup=5, iters=50)
        my_kernel(a, b)

    Access last_result for device metrics:
        my_kernel.last_result.device_time_us
    """
    kernel_name = getattr(jit_kernel, "__name__", str(jit_kernel))

    @wraps(jit_kernel)
    def wrapper(*args, **kwargs):
        inspect_dir = os.environ.get("NEURON_RT_INSPECT_OUTPUT_DIR", "")
        before_inspect = _snapshot_neffs(inspect_dir) if inspect_dir else {}
        before_compile = _snapshot_neffs(_COMPILE_WORKDIR)

        result = jit_kernel(*args, **kwargs)
        xm.mark_step()

        bench_neff, profile_neff, ntff = _find_kernel_artifacts(
            before_inspect, before_compile
        )
        if bench_neff is None and profile_neff is None:
            print("[benchmark] Could not find NEFF. "
                  "Ensure NEURON_RT_INSPECT_OUTPUT_DIR is set before any neuron imports.")
            return result

        if bench_neff:
            print(f"[benchmark] NEFF (bench): {os.path.basename(bench_neff)}")
        if profile_neff:
            print(f"[benchmark] NEFF (profile): {os.path.basename(profile_neff)}")

        # Parse device metrics from the NTFF captured at runtime.
        # neuron-profile view works without hardware — it just parses the trace.
        prof = {}
        if ntff and profile_neff:
            print(f"[benchmark] NTFF captured: {os.path.basename(ntff)}")
            prof = _parse_ntff(profile_neff, ntff)
        elif profile_neff is None:
            print("[benchmark] No NTFF found. "
                  "Set NEURON_RT_INSPECT_DEVICE_PROFILE=1 before neuron imports.")

        _print_profile_results(prof, kernel_name)

        # Queue the compile-workdir NEFF for neuron-bench at atexit.
        # XLA releases NeuronCores after full process exit, so neuron-bench
        # must run after this script finishes — we print the command.
        if bench_neff:
            _pending_bench.append({
                "neff": bench_neff, "warmup": warmup, "iters": iters, "name": kernel_name
            })
            print(f"[benchmark] neuron-bench command will be printed at script exit.")

        wrapper.last_result = BenchmarkResult(prof, bench_neff or profile_neff, ntff)
        return result

    wrapper.last_result = None
    return wrapper


def nki_benchmark(
    jit_kernel,
    *args,
    warmup: int = 10,
    iters: int = 100,
    **kwargs,
) -> BenchmarkResult | None:
    """
    One-shot benchmark: run a @nki.jit kernel and print device metrics.
    neuron-bench latency is printed at script exit.

    Usage:
        @nki.jit(platform_target="trn2")
        def my_kernel(a, b): ...

        result = nki_benchmark(my_kernel, a, b, warmup=5, iters=50)
        print(result.device_time_us)
    """
    wrapped = wrap_benchmark(jit_kernel, warmup=warmup, iters=iters)
    wrapped(*args, **kwargs)
    return wrapped.last_result
