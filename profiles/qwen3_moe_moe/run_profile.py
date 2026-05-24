"""Driver: compile and execute the clean MoE-block kernel on trn3 + LNC=2,
producing a NEFF that the profiling skill's later steps can capture.

Inputs match the megakernel's per-rank shapes (TP=4 sharding pre-applied):
  X         [1, 1, 2048]                       bf16
  gpost     [1, 2048]                          bf16
  router_w  [2048, 128]                        bf16
  gate_up_w [128, 2048, 384]   (= 2 * 192)     bf16
  down_w    [128, 192, 2048]                   bf16

Two executions are issued so neuron-explorer can profile the 2nd (warm) run
via --profile-nth-exec=2.
"""

import os
import sys

# --- Profiling env vars MUST be set before any neuronx import -------------
os.environ.setdefault("NEURON_RT_INSPECT_ENABLE", "1")
os.environ.setdefault("NEURON_RT_INSPECT_DEVICE_PROFILE", "1")
os.environ.setdefault("NEURON_RT_INSPECT_OUTPUT_DIR",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "output")))
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3")

# Make the repo importable (megakernels.*, nki_kernels.*).
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch_xla.core.xla_model as xm

from moe_block_kernel import moe_block_kernel
from megakernels.qwen3_moe.transformer_qwen3_moe_speculative import (
    H, E, TOP_K, I_PER_EXPERT,
)


def _seeded(g, shape, dtype, scale=1.0):
    return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(dtype)


def main():
    B = 1
    S_tkg = 1
    dtype = torch.bfloat16

    g = torch.Generator().manual_seed(0xDECAFBAD)
    X         = _seeded(g, (B, S_tkg, H), dtype, scale=0.05)
    gpost     = (torch.ones(1, H, dtype=dtype)
                 + _seeded(g, (1, H), dtype, scale=0.01))
    router_w  = _seeded(g, (H, E), dtype, scale=0.02)
    gate_up_w = _seeded(g, (E, H, 2 * I_PER_EXPERT), dtype, scale=0.02)
    down_w    = _seeded(g, (E, I_PER_EXPERT, H), dtype, scale=0.02)

    device = xm.xla_device()
    X         = X.to(device=device)
    gpost     = gpost.to(device=device)
    router_w  = router_w.to(device=device)
    gate_up_w = gate_up_w.to(device=device)
    down_w    = down_w.to(device=device)

    print(f"Output dir: {os.environ['NEURON_RT_INSPECT_OUTPUT_DIR']}")
    print(f"Target:     {os.environ.get('NEURON_PLATFORM_TARGET_OVERRIDE')}")
    print(f"Visible:    {os.environ.get('NEURON_RT_VISIBLE_CORES')}")
    print("Compiling + running MoE-block kernel (LNC=2)…")

    # Execute twice so neuron-explorer can profile the 2nd (warm) run.
    for it in range(2):
        xm.mark_step()
        out = moe_block_kernel[2](X, gpost, router_w, gate_up_w, down_w)
        if isinstance(out, (tuple, list)):
            Y = out[0]
        else:
            Y = out
        # Force materialization.
        Y_cpu = Y.cpu()
        print(f"  iter {it}: Y.shape={tuple(Y_cpu.shape)} dtype={Y_cpu.dtype} "
              f"nan={torch.isnan(Y_cpu).any().item()} "
              f"sample={Y_cpu.flatten()[:4].tolist()}")

    print("Done. NEFF written under:", os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"])


if __name__ == "__main__":
    main()
