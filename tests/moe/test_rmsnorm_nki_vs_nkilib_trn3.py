"""
test_rmsnorm_nki_vs_nkilib_trn3.py

Verify the standalone RMSNorm NKI kernel against the nkilib RMSNorm TKG
subkernel used by qwen_with_nki.py.

Inputs:
  hidden_states: [T, H=2048] bf16
  gamma:         [1, H=2048] bf16

Reference:
  nkilib.core.subkernels.rmsnorm_tkg.rmsnorm_tkg with the same call shape used
  by qwen_with_nki.py.

Kernel:
  kernels.moe_fused_tkg.rmsnorm_nki.rmsnorm_hbm(hidden_states, gamma)

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python tests/moe/test_rmsnorm_nki_vs_nkilib_trn3.py --tokens 1 --n-samples 256
"""

import argparse
import contextlib
import os
import sys
import tempfile

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch.nn as nn

from nkilib.core.subkernels.rmsnorm_tkg import rmsnorm_tkg as _rmsnorm_tkg
from nkilib.core.utils.allocator import SbufManager

from kernels.moe_fused_tkg.rmsnorm_nki import rmsnorm_hbm


LNC = 2
PMAX = 128
H = 2048
H_FREE = H // PMAX
EPS = 1.0e-6

COMPILER_ARGS = (
    "--enable-saturate-infinity --enable-mixed-precision-accumulation "
    "--model-type transformer -O1 "
    "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
)

HIDDEN_SCALES = [0.0, 1.0e-3, 0.1, 1.0, 3.0, 10.0]
HIDDEN_DISTS = ["normal", "laplace", "student-t5", "zeros", "spiky"]
GAMMA_MODES = ["ones", "qwen-like", "wide", "tiny", "ramp"]


@nki.jit
def nkilib_rmsnorm_hbm(inp, gamma):
    """
    HBM wrapper around nkilib RMSNorm TKG.

    Mirrors the RMSNorm call in qwen_with_nki.py and stores nkilib's
    [PMAX, T, H_FREE] SBUF output back to row-major [T, H] HBM.
    """
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    T = inp.shape[0]
    prg_id = nl.program_id(axis=0)

    sbm.open_scope(name="nkilib_rmsnorm_hbm")
    rmsnorm_out = sbm.alloc_stack(
        (PMAX, T, H_FREE), inp.dtype, buffer=nl.sbuf, name="rmsnorm_out"
    )
    _rmsnorm_tkg(
        input=inp.reshape((1, T, H)),
        gamma=gamma,
        output=rmsnorm_out,
        eps=EPS,
        hidden_dim_tp=False,
        single_core_forced=(T > 1),
        sbm=sbm,
    )

    output = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.shared_hbm)
    out_row_sb = sbm.alloc_stack((T, H), inp.dtype, buffer=nl.sbuf, name="out_row_sb")
    for h1 in nl.static_range(H_FREE):
        tp_psum = nl.ndarray((T, PMAX), dtype=inp.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=tp_psum[0:T, 0:PMAX],
            data=rmsnorm_out[0:PMAX, 0:T, h1],
        )
        nisa.activation(
            dst=out_row_sb[0:T, nl.ds(h1 * PMAX, PMAX)],
            op=nl.copy,
            data=tp_psum[0:T, 0:PMAX],
        )
    if prg_id == 0:
        nisa.dma_copy(dst=output[0:T, 0:H], src=out_row_sb[0:T, 0:H])

    sbm.close_scope()
    return output


class RefNkilibRMSNormModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._rmsnorm_jit = nki.jit(nkilib_rmsnorm_hbm)
        self.eval()

    def forward(self, hidden_states: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return self._rmsnorm_jit[LNC](hidden_states, gamma)


class KernelRMSNormModule(nn.Module):
    def __init__(self, hoisted_gamma: bool = False):
        super().__init__()
        self.hoisted_gamma = hoisted_gamma
        self._rmsnorm_jit = nki.jit(rmsnorm_hbm)
        self.eval()

    def forward(self, hidden_states: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return self._rmsnorm_jit[LNC](hidden_states, gamma, self.hoisted_gamma)


def _sample_hidden(rng: np.random.Generator, tokens: int, scale: float, dist: str) -> torch.Tensor:
    if dist == "normal":
        x = rng.standard_normal((tokens, H))
    elif dist == "laplace":
        x = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=(tokens, H))
    elif dist == "student-t5":
        df = 5.0
        x = rng.standard_t(df, size=(tokens, H))
        x = x * np.sqrt((df - 2.0) / df)
    elif dist == "zeros":
        x = np.zeros((tokens, H), dtype=np.float32)
    elif dist == "spiky":
        x = rng.standard_normal((tokens, H)).astype(np.float32) * 0.05
        n_spikes = max(1, tokens * H // 128)
        flat_idx = rng.choice(tokens * H, size=n_spikes, replace=False)
        x.reshape(-1)[flat_idx] = rng.choice([-1.0, 1.0], size=n_spikes) * rng.uniform(
            8.0, 32.0, size=n_spikes
        )
    else:
        raise ValueError(f"unknown hidden distribution: {dist}")
    return torch.from_numpy((x * scale).astype(np.float32)).bfloat16()


def _sample_gamma(rng: np.random.Generator, mode: str) -> torch.Tensor:
    if mode == "ones":
        gamma = np.ones((1, H), dtype=np.float32)
    elif mode == "qwen-like":
        gamma = 1.0 + rng.normal(0.0, 0.05, size=(1, H)).astype(np.float32)
    elif mode == "wide":
        gamma = rng.uniform(-3.0, 3.0, size=(1, H)).astype(np.float32)
    elif mode == "tiny":
        gamma = rng.uniform(-1.0e-3, 1.0e-3, size=(1, H)).astype(np.float32)
    elif mode == "ramp":
        gamma = np.linspace(-2.0, 2.0, H, dtype=np.float32).reshape(1, H)
    else:
        raise ValueError(f"unknown gamma mode: {mode}")
    return torch.from_numpy(gamma).bfloat16()


def make_inputs(tokens: int, n_samples: int):
    combos = [
        (hidden_scale, hidden_dist, gamma_mode)
        for hidden_scale in HIDDEN_SCALES
        for hidden_dist in HIDDEN_DISTS
        for gamma_mode in GAMMA_MODES
    ]
    inputs_list = []
    combos_list = []
    for i in range(n_samples):
        hidden_scale, hidden_dist, gamma_mode = combos[i % len(combos)]
        rng = np.random.default_rng(200 + i)
        hidden = _sample_hidden(rng, tokens, hidden_scale, hidden_dist)
        gamma = _sample_gamma(rng, gamma_mode)
        inputs_list.append((hidden, gamma))
        combos_list.append((hidden_scale, hidden_dist, gamma_mode))
    return inputs_list, combos_list


def _bf16_bits(t: torch.Tensor) -> str:
    return f"0x{int(t.detach().cpu().view(torch.int16).item()) & 0xffff:04x}"


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().contiguous().view(torch.int16)


def _mismatch_stats(ref_out: torch.Tensor, kernel_out: torch.Tensor) -> dict:
    ref_cpu = ref_out.detach().cpu()
    kernel_cpu = kernel_out.detach().cpu()
    bit_mismatch = _bits(ref_cpu) != _bits(kernel_cpu)
    abs_diff = (ref_cpu.float() - kernel_cpu.float()).abs()
    return {
        "num_mismatch": int(bit_mismatch.sum().item()),
        "max_abs_err": float(abs_diff.max().item()),
        "mean_abs_err": float(abs_diff.mean().item()),
    }


def _print_failure_details(
    sample_idx: int,
    combo: tuple,
    hidden: torch.Tensor,
    gamma: torch.Tensor,
    ref_out: torch.Tensor,
    kernel_out: torch.Tensor,
    max_coords: int = 8,
) -> None:
    ref_cpu = ref_out.detach().cpu()
    kernel_cpu = kernel_out.detach().cpu()
    hidden_cpu = hidden.detach().cpu()
    gamma_cpu = gamma.detach().cpu()
    abs_diff = (ref_cpu.float() - kernel_cpu.float()).abs()
    bit_mismatch = _bits(ref_cpu) != _bits(kernel_cpu)
    failing = torch.nonzero(bit_mismatch, as_tuple=False)
    flat_order = torch.argsort(abs_diff.flatten(), descending=True)

    print(
        f"\n  BIT FAILURE sample={sample_idx} combo={combo} "
        f"num_coords={failing.shape[0]} max_abs={abs_diff.max().item():.6g}",
        flush=True,
    )

    shown = 0
    for flat_idx in flat_order.tolist():
        if shown >= max_coords:
            break
        t = flat_idx // ref_cpu.shape[1]
        h = flat_idx % ref_cpu.shape[1]
        if not bool(bit_mismatch[t, h].item()):
            continue
        k_val = kernel_cpu[t, h]
        r_val = ref_cpu[t, h]
        print(
            "    "
            f"coord=(t={t}, h={h}) "
            f"custom={float(k_val): .8g} {_bf16_bits(k_val)} "
            f"nkilib={float(r_val): .8g} {_bf16_bits(r_val)} "
            f"diff={float((k_val.float() - r_val.float()).item()):+.8g} "
            f"hidden={float(hidden_cpu[t, h]): .8g} {_bf16_bits(hidden_cpu[t, h])} "
            f"gamma={float(gamma_cpu[0, h]): .8g} {_bf16_bits(gamma_cpu[0, h])}",
            flush=True,
        )
        shown += 1


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def _fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def _build_module(*args, **kwargs):
    from neuronx_distributed_inference.utils.testing import build_module

    return build_module(*args, **kwargs)


def _compile_ref(args, example_inputs, workdir):
    print("\nCompiling RefNkilibRMSNormModule...", flush=True)
    ref_traced = _build_module(
        module_cls=RefNkilibRMSNormModule,
        example_inputs=example_inputs,
        tp_degree=1,
        logical_nc_config=LNC,
        compiler_args=args.ref_compiler_args or COMPILER_ARGS,
        compiler_workdir=os.path.join(workdir, "ref_workdir"),
    )
    print("RefNkilibRMSNormModule compile PASS", flush=True)
    return ref_traced


def _compile_kernel(args, example_inputs, workdir, hoisted_gamma: bool):
    mode = f"hoisted_{_fmt_bool(hoisted_gamma)}"
    print(f"\nCompiling KernelRMSNormModule ({mode})...", flush=True)
    kern_traced = _build_module(
        module_cls=KernelRMSNormModule,
        example_inputs=example_inputs,
        module_init_kwargs={"hoisted_gamma": hoisted_gamma},
        tp_degree=1,
        logical_nc_config=LNC,
        compiler_args=args.kernel_compiler_args or COMPILER_ARGS,
        compiler_workdir=os.path.join(workdir, f"kern_workdir_{mode}"),
    )
    print(f"KernelRMSNormModule compile PASS ({mode})", flush=True)
    return kern_traced


def _validate_mode(
    ref_traced,
    kern_traced,
    inputs_list,
    combos_list,
    hoisted_gamma: bool,
    dump_failures: int,
) -> dict:
    print(f"\nValidating hoisted_gamma={_fmt_bool(hoisted_gamma)}...", flush=True)
    mode_stats = []
    dumped_failures = 0

    for i, ((hidden, gamma), combo) in enumerate(zip(inputs_list, combos_list)):
        ref_out = ref_traced(hidden, gamma)
        kernel_out = kern_traced(hidden, gamma)
        stats = _mismatch_stats(ref_out, kernel_out)
        stats["combo"] = combo
        stats["hoisted_gamma"] = hoisted_gamma
        mode_stats.append(stats)

        if stats["num_mismatch"] and dumped_failures < dump_failures:
            _print_failure_details(i, combo, hidden, gamma, ref_out, kernel_out)
            dumped_failures += 1

        if (i + 1) % 128 == 0 or i == len(inputs_list) - 1:
            print(f"    validated {i + 1:4d} / {len(inputs_list)} samples", flush=True)

    mismatch_samples = sum(1 for s in mode_stats if s["num_mismatch"])
    mismatch_coords = sum(s["num_mismatch"] for s in mode_stats)
    max_abs_errs = [s["max_abs_err"] for s in mode_stats]
    mean_abs_errs = [s["mean_abs_err"] for s in mode_stats]

    print(f"\n--- hoisted_gamma={_fmt_bool(hoisted_gamma)} summary ---", flush=True)
    print(f"  mismatch samples: {mismatch_samples}/{len(mode_stats)}", flush=True)
    print(f"  mismatch coords:  {mismatch_coords}", flush=True)
    print(
        f"  Max |err|:  p50={_percentile(max_abs_errs, 50):.6g}  "
        f"p95={_percentile(max_abs_errs, 95):.6g}  max={max(max_abs_errs):.6g}",
        flush=True,
    )
    print(
        f"  Mean |err|: p50={_percentile(mean_abs_errs, 50):.6g}  "
        f"p95={_percentile(mean_abs_errs, 95):.6g}  max={max(mean_abs_errs):.6g}",
        flush=True,
    )
    return {
        "hoisted_gamma": hoisted_gamma,
        "mismatch_samples": mismatch_samples,
        "mismatch_coords": mismatch_coords,
        "total_samples": len(mode_stats),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--dump-failures", type=int, default=2)
    parser.add_argument("--skip-hoisted", action="store_true")
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--ref-compiler-args", type=str, default=None)
    parser.add_argument("--kernel-compiler-args", type=str, default=None)
    args = parser.parse_args()

    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")

    print(
        f"Config: H={H} H_FREE={H_FREE} T={args.tokens} LNC={LNC} EPS={EPS}",
        flush=True,
    )
    example_inputs = [
        (
            torch.zeros(args.tokens, H, dtype=torch.bfloat16),
            torch.ones(1, H, dtype=torch.bfloat16),
        )
    ]
    inputs_list, combos_list = make_inputs(args.tokens, args.n_samples)

    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
        workdir_cm = contextlib.nullcontext(args.workdir)
    elif args.keep_workdir:
        workdir_cm = contextlib.nullcontext(tempfile.mkdtemp(prefix="rmsnorm_vs_nkilib_"))
    else:
        workdir_cm = tempfile.TemporaryDirectory(prefix="rmsnorm_vs_nkilib_")

    with workdir_cm as workdir:
        print(f"Compiler workdir: {workdir}", flush=True)
        ref_traced = _compile_ref(args, example_inputs, workdir)

        summaries = []
        for hoisted_gamma in (False, True):
            if hoisted_gamma and args.skip_hoisted:
                continue
            kern_traced = _compile_kernel(args, example_inputs, workdir, hoisted_gamma)
            summaries.append(
                _validate_mode(
                    ref_traced,
                    kern_traced,
                    inputs_list,
                    combos_list,
                    hoisted_gamma,
                    args.dump_failures,
                )
            )

    total_mismatch_samples = sum(s["mismatch_samples"] for s in summaries)
    total_mismatch_coords = sum(s["mismatch_coords"] for s in summaries)
    total_samples = sum(s["total_samples"] for s in summaries)
    print("\n--- overall bit-accuracy summary ---", flush=True)
    print(f"  mismatch samples: {total_mismatch_samples}/{total_samples}", flush=True)
    print(f"  mismatch coords:  {total_mismatch_coords}", flush=True)
    if total_mismatch_coords:
        raise AssertionError(
            f"RMSNorm bit mismatch: {total_mismatch_coords} bf16 coordinates "
            f"across {total_mismatch_samples} samples"
        )
    print("  OVERALL: PASS (exact bf16 bit match)", flush=True)


if __name__ == "__main__":
    main()
