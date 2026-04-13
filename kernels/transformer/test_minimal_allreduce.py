"""
Minimal NKI nccl.all_reduce reproducer.

Two independent variants:
  A) SBUF allreduce: HBM -> SBUF -> all_reduce -> SBUF -> HBM
  B) HBM allreduce: all_reduce directly on shared_hbm tensors

Goal: isolate whether the ENC_ALG_MESH runtime assert is caused by
SBUF-path allreduce specifically, or nccl.all_reduce in general.
"""

import os
import sys
import traceback
import tempfile

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

sys.path.insert(0, "/home/ubuntu/nki-moe")

import torch
import torch.nn as nn

import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as nccl
from nki.collectives import ReplicaGroup

from neuronx_distributed.trace import parallel_model_trace
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

# ── config ─────────────────────────────────────────────────────────────────
B = 1
S = 1
H = 256
TP = 2

COMPILER_ARGS = (
    "--enable-saturate-infinity "
    "--enable-mixed-precision-accumulation "
    "--model-type transformer "
    "-O3 "
    "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-hlo2tensorizer-options='--verify-hlo=true' "
    "--internal-max-instruction-limit=15000000"
)


# ── Variant A: SBUF allreduce ──────────────────────────────────────────────
def kernel_sbuf_allreduce(X):
    """X: [B, S, H] HBM. Returns [B, S, H] HBM = allreduce(X) across TP=2."""
    BxS = B * S
    dtype = X.dtype
    rg = ReplicaGroup([[0, 1]])

    out = nl.ndarray((B, S, H), dtype=dtype, buffer=nl.shared_hbm)

    # HBM -> SBUF
    src_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=src_sb, src=X.reshape((BxS, H)))

    # SBUF allreduce
    dst_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
    nccl.all_reduce(dsts=[dst_sb], srcs=[src_sb], op=nl.add, replica_group=rg)

    # SBUF -> HBM
    nisa.dma_copy(dst=out.reshape((BxS, H)), src=dst_sb)

    return out


# ── Variant B: HBM allreduce ───────────────────────────────────────────────
def kernel_hbm_allreduce(X):
    """X: [B, S, H] HBM. Returns allreduce on HBM tensors directly."""
    dtype = X.dtype
    rg = ReplicaGroup([[0, 1]])

    out = nl.ndarray((B, S, H), dtype=dtype, buffer=nl.shared_hbm)
    nccl.all_reduce(dsts=[out], srcs=[X], op=nl.add, replica_group=rg)
    return out


# ── Variant C: SBUF allreduce + sendrecv gather ────────────────────────────
# Mirrors the full _sb2sb_all_reduce_gather pattern from
# nkilib/experimental/transformer/transformer_tkg.py:64-87
def kernel_sbuf_ar_gather(X):
    """Reproduce the SBUF allreduce + sendrecv gather pattern.

    Output Y: [1, 1, H_C] bf16 filled with 2.0 (1.0 + 1.0 from allreduce).
    """
    H_C = 2048
    H0 = 128
    H1 = 16
    H1_SHARD = 8
    BxS_C = 1

    _, n_prgs, prg_id = get_verified_program_sharding_info("test_kernel", (0, 1), 2)
    rg = ReplicaGroup([[0, 1]])

    # Sharded SBUF input: fill with 1.0 so allreduce across 2 ranks -> 2.0
    sharded_sb = nl.ndarray((H0, H1_SHARD * BxS_C), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=sharded_sb, value=1.0)

    # AllReduce
    sharded_AR_sb = nl.ndarray(sharded_sb.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    nccl.all_reduce(dsts=[sharded_AR_sb], srcs=[sharded_sb], op=nl.add, replica_group=rg)

    # Local gather
    gathered_sb = nl.ndarray((H0, H1 * BxS_C), dtype=nl.bfloat16, buffer=nl.sbuf)
    f_shard = nl.ds(start=prg_id * BxS_C * H1_SHARD, size=BxS_C * H1_SHARD)
    nisa.tensor_copy(dst=gathered_sb[:, f_shard], src=sharded_AR_sb)

    # SendRecv: exchange own shard with the other LNC
    if n_prgs > 1:
        other_lnc = 1 - prg_id
        f_other = nl.ds(start=other_lnc * BxS_C * H1_SHARD, size=BxS_C * H1_SHARD)
        nisa.sendrecv(
            src=sharded_AR_sb,
            dst=gathered_sb[:, f_other],
            send_to_rank=other_lnc,
            recv_from_rank=other_lnc,
            pipe_id=0,
        )

    # Store gathered_sb to HBM: observable output so it isn't DCE'd.
    Y = nl.ndarray((1, 1, H_C), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="Y")
    nisa.dma_copy(dst=Y.reshape((H0, H1)), src=gathered_sb)
    return Y


# ── Torch module wrappers ──────────────────────────────────────────────────
class SBUFAllReduceMod(nn.Module):
    def forward(self, X):
        return nki.jit(kernel_sbuf_allreduce)[2](X)


class HBMAllReduceMod(nn.Module):
    def forward(self, X):
        return nki.jit(kernel_hbm_allreduce)[2](X)


class SBUFARGatherMod(nn.Module):
    def forward(self, X):
        return nki.jit(kernel_sbuf_ar_gather)[2](X)


def _sbuf_factory():
    m = SBUFAllReduceMod()
    m.eval()
    return m, {}


def _hbm_factory():
    m = HBMAllReduceMod()
    m.eval()
    return m, {}


def _sbuf_ar_gather_factory():
    m = SBUFARGatherMod()
    m.eval()
    return m, {}


def run_variant(name, factory, example_args, expected):
    import gc, time
    print("=" * 70)
    print(f"Variant {name}")
    print("=" * 70)
    traced = None
    try:
        with tempfile.TemporaryDirectory(prefix=f"minimal_ar_{name}_") as workdir:
            print(f"Workdir: {workdir}")
            print("Compiling...")
            traced = parallel_model_trace(
                factory,
                example_args,
                tp_degree=TP,
                compiler_workdir=workdir,
                compiler_args=COMPILER_ARGS,
            )
            print("Compile PASS. Running...")
            result = traced(*example_args)
            r = result.float().cpu()
            print(f"  shape={tuple(r.shape)}  min={r.min():.4f}  max={r.max():.4f}  mean={r.mean():.4f}")
            ok = torch.allclose(r, expected, atol=1e-3, rtol=1e-3)
            max_err = (r - expected).abs().max().item()
            print(f"  [{'PASS' if ok else 'FAIL'}] allclose(result, 2*X)  max_abs_err={max_err:.6f}")
    except Exception as e:
        print(f"Variant {name} CRASHED:")
        traceback.print_exc()
    finally:
        try:
            del traced
        except Exception:
            pass
        gc.collect()
        time.sleep(5)


def _child(which):
    X = torch.ones(B, S, H, dtype=torch.bfloat16)
    expected = (2.0 * X).float()
    example_args = (X,)
    if which == "A":
        run_variant("A_SBUF", _sbuf_factory, example_args, expected)
    elif which == "B":
        run_variant("B_HBM", _hbm_factory, example_args, expected)
    else:
        # Variant C: output shape is [1,1,2048] filled with 2.0
        H_C = 2048
        X_c = torch.zeros(1, 1, H_C, dtype=torch.bfloat16)
        expected_c = 2.0 * torch.ones(1, 1, H_C, dtype=torch.float32)
        run_variant("C_SBUF_AR_GATHER", _sbuf_ar_gather_factory, (X_c,), expected_c)


def main():
    # Run each variant in its own subprocess so NeuronCores are fully released.
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    for which in ("A", "B", "C"):
        p = ctx.Process(target=_child, args=(which,))
        p.start()
        p.join()
        print(f"[parent] Variant {which} subprocess exited with code {p.exitcode}")
        print()


if __name__ == "__main__":
    main()
