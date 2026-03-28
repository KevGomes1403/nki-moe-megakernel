"""
Minimal reproducer for NCC_IXLV002 compiler bug — core_barrier variant.

A simple LNC2 NKI kernel that:
  - Each core loads its own row from a [2, H] input
  - Stores partial to shared HBM
  - core_barrier to sync
  - Loads the other core's partial and adds
  - Writes result

When this kernel[2] is called 2+ times in a graph followed by a matmul
with output dim <= 128, neuronx-cc may crash with NCC_IXLV002 (core barrier
name mismatch in lnc_verifier).

Environment:
  - neuronx-cc 2.23.6484.0
  - trn2 (LNC2 default)
"""

import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import torch
import torch.nn as nn
import nki
import nki.isa as nisa
from nki.isa import core_barrier
import nki.language as nl
import torch_neuronx

HIDDEN = 2048
N_PRGS = 2
H_SHARD = HIDDEN // N_PRGS  # = 1024


@nki.jit
def lnc2_allreduce(x):
    """
    Minimal LNC2 kernel: each core loads its row, stores to shared HBM,
    core_barrier, then loads the other core's row and reduces.

    Input:  x[2, H]   — 2 rows, one per core
    Output: (out[2, H_SHARD], workspace[2, H])
      out[core_id, :] = sum of both rows for this core's H_SHARD columns
      workspace is the shared-HBM partial buffer (must be returned)
    """
    H = x.shape[1]
    core_id = nl.program_id(axis=0)
    other = 1 - core_id

    # Each core loads its own row
    my_row = nl.ndarray((1, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=my_row, src=x[core_id:core_id + 1, :])

    # Shared-HBM workspace: [2, H] — each core writes its own slot
    workspace = nl.ndarray((N_PRGS, H), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=workspace[core_id:core_id + 1, :], src=my_row)

    # Sync: wait for both cores to finish writing their rows
    core_barrier(workspace, cores=[0, 1])

    # Each core loads its own H_SHARD from both partials, adds, and stores
    out = nl.ndarray((N_PRGS, H_SHARD), dtype=x.dtype, buffer=nl.shared_hbm)

    my_partial   = nl.ndarray((1, H_SHARD), dtype=x.dtype, buffer=nl.sbuf)
    other_partial = nl.ndarray((1, H_SHARD), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=my_partial,
        src=workspace[core_id:core_id + 1,
                      nl.ds(core_id * H_SHARD, H_SHARD)],
        dge_mode=3,
    )
    nisa.dma_copy(
        dst=other_partial,
        src=workspace[other:other + 1,
                      nl.ds(core_id * H_SHARD, H_SHARD)],
        dge_mode=3,
    )
    nisa.tensor_tensor(my_partial, my_partial, other_partial, op=nl.add)
    nisa.dma_copy(dst=out[core_id:core_id + 1, :], src=my_partial)

    return out, workspace


class AllReduceLayer(nn.Module):
    """Wraps the LNC2 core_barrier kernel + a projection matmul."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16))

    def forward(self, x):
        # x: [1, 1, HIDDEN]
        x_2d = x.view(1, HIDDEN).expand(2, HIDDEN).contiguous()
        out, _ws = lnc2_allreduce[2](x_2d)
        # out: [2, H_SHARD] — reassemble and project
        full = out.view(1, HIDDEN)
        result = torch.matmul(full, self.proj.data)
        return result.unsqueeze(0)  # -> [1, 1, HIDDEN]


class Model(nn.Module):
    def __init__(self, n_layers, out_dim):
        super().__init__()
        self.layers = nn.ModuleList([AllReduceLayer() for _ in range(n_layers)])
        self.head = nn.Linear(HIDDEN, out_dim, bias=False).to(torch.bfloat16)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def trace_test(n_layers, out_dim):
    x = torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16)
    model = Model(n_layers, out_dim)
    model.eval()
    try:
        torch_neuronx.trace(
            model, x,
            compiler_args=["--target=trn2", "--model-type", "transformer", "-O1"],
        )
        return "PASS"
    except Exception as e:
        err = str(e)
        if "NCC_IXLV002" in err or "neuronx-cc failed with 70" in err:
            return "NCC_IXLV002"
        return f"other: {type(e).__name__}: {err[:200]}"


def main():
    print(f"{'Config':<45} Result")
    print("-" * 60)

    configs = [
        (1, 128,  "1 layer,  out=128"),
        (1, 256,  "1 layer,  out=256"),
        (2, 256,  "2 layers, out=256"),
        (2, 128,  "2 layers, out=128"),
        (2, 2048, "2 layers, out=2048"),
        (3, 128,  "3 layers, out=128"),
    ]

    for n_layers, out_dim, label in configs:
        r = trace_test(n_layers, out_dim)
        print(f"  {label:<43} {r}")

    print("-" * 60)


if __name__ == "__main__":
    main()
