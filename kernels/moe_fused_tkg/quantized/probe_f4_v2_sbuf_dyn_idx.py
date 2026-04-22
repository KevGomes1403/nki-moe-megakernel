"""
[F4] Round 2 side probe — 20-line variant.
SBUF→SBUF tensor_copy with runtime scalar_offset: does trn3 accept it?

This probe uses the pattern that trn2 rejected with NCC_INLA001
(OPTIMIZATION_LOG.md Round 7 Plan M). If trn3 accepts: add [F4] to
Round 3. If NCC_INLA001 still fires: log and skip.
"""
import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"

import nki
import nki.language as nl
import nki.isa as nisa
import torch
import torch_xla.core.xla_model as xm

@nki.jit
def probe(scales_hbm, eid_hbm):
    sbuf_all = nl.ndarray((128, 4), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf_all, src=scales_hbm, dge_mode=3)
    eid = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eid, src=eid_hbm)
    picked = nl.ndarray((1, 4), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_copy(dst=picked,
                     src=sbuf_all.ap(pattern=[[4, 1], [1, 4]],
                                     offset=0,
                                     scalar_offset=eid,
                                     indirect_dim=0))
    out = nl.ndarray((1, 4), dtype=nl.uint8, buffer=nl.shared_hbm)
    nl.store(out, picked)
    return out


if __name__ == "__main__":
    d = xm.xla_device()
    try:
        r = probe(torch.zeros((128, 4), dtype=torch.uint8).to(d),
                  torch.tensor([3], dtype=torch.int32).to(d))
        _ = r.cpu()
        print("[F4 PROBE v2] PASS — trn3 accepts. Add to Round 3 candidates.")
    except Exception as e:
        s = str(e)
        code = "NCC_INLA001" if "NCC_INLA001" in s else ("NCC_IBIR010" if "NCC_IBIR010" in s else "other")
        print(f"[F4 PROBE v2] FAIL — compiler error code: {code}")
        print(f"  first 300 chars: {s[:300]}")
        if code == "NCC_INLA001":
            print("  action: same as trn2. Skip [F4] per user instruction.")
        elif code == "NCC_IBIR010":
            print("  action: syntactically deeper rejection than trn2. Ambiguous — defer to Round 3 investigation.")
        else:
            print("  action: unexpected error, investigate.")
