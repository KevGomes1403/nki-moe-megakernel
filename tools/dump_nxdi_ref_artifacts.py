#!/usr/bin/env python3
"""
Compile the NxDI reference MoE path to a persistent workdir and dump readable HLO.

Example:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  python tools/dump_nxdi_ref_artifacts.py --weight-scale 3.0 \
      --outdir /home/ubuntu/nki-moe/artifacts/nxdi_ref_ws3_compile
"""

import argparse
import os
import shutil

from google.protobuf import text_format
import torch

from neuronx_distributed_inference.utils.testing import build_module
from tests.test_v30c_vs_nxdi_trn3 import (
    B,
    LNC,
    COMPILER_ARGS,
    RefMoEModule,
    _H,
    make_config,
)
from neuronx_distributed.trace import hlo_utils


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-scale", type=float, default=3.0)
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Persistent output directory for compiler artifacts",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    workdir = os.path.join(args.outdir, f"ref_workdir_ws{args.weight_scale:g}")
    if os.path.exists(workdir) and not args.keep_existing:
        shutil.rmtree(workdir)
    os.makedirs(args.outdir, exist_ok=True)

    cfg = make_config()
    example_inputs = [(torch.zeros(B, 1, _H, dtype=torch.bfloat16),)]

    build_module(
        module_cls=RefMoEModule,
        example_inputs=example_inputs,
        module_init_kwargs={
            "config": cfg,
            "seed": args.seed,
            "weight_scale": args.weight_scale,
        },
        tp_degree=1,
        logical_nc_config=LNC,
        compiler_args=COMPILER_ARGS,
        compiler_workdir=workdir,
    )

    bk0 = os.path.join(workdir, "RefMoEModule", "_tp0_bk0")
    hlo_pb = None
    for name in os.listdir(bk0):
        if name.endswith(".hlo_module.pb"):
            hlo_pb = os.path.join(bk0, name)
            break
    if hlo_pb is None:
        raise FileNotFoundError(f"Could not find HLO protobuf in {bk0}")

    hlo_text = os.path.join(bk0, "hlo_module.textproto")
    hm = hlo_utils.read_hlo(hlo_pb)
    with open(hlo_text, "w") as f:
        f.write(text_format.MessageToString(hm))

    print(f"workdir: {workdir}")
    print(f"hlo_pb: {hlo_pb}")
    print(f"hlo_text: {hlo_text}")
    print(f"neff: {os.path.join(bk0, 'graph.neff')}")
    print(f"metaneff: {os.path.join(bk0, 'metaneff.pb')}")
    print(f"command: {os.path.join(bk0, 'command.txt')}")


if __name__ == "__main__":
    main()
