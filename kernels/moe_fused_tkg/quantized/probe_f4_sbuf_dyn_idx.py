"""
[F4] side probe — does trn3 compiler accept SBUF→SBUF tensor_copy with
runtime scalar_offset (SBUF-resident expert_id)?

Trn2 raised NCC_INLA001 on exactly this pattern (OPTIMIZATION_LOG.md
Round 7, Plan M / v12m). If trn3 accepts: promote [F4] scale-preloading
to Round 2.

5-minute gate before Plan C dispatch.
"""
import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"

import nki
import nki.language as nl
import nki.isa as nisa
import torch
import torch_xla.core.xla_model as xm

_PMAX = 128    # trn3 max partition dim
_E = 128       # num experts (matches full partition)
_F = 192       # bytes per expert's scale row (down_scales per-expert width)


@nki.jit
def probe(all_scales_hbm, expert_id_hbm):
    # 1. Preload all 128 experts' scales to SBUF once (the [F4] prologue).
    all_sb = nl.ndarray((_E, _F), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.dma_copy(dst=all_sb, src=all_scales_hbm, dge_mode=3)

    # 2. Load runtime expert_id into SBUF.
    eid_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eid_sb[0:1, 0:1], src=expert_id_hbm[0:1])

    # 3. SBUF→SBUF tensor_copy with SBUF-resident scalar_offset.
    #    This is the exact NCC_INLA001 pattern from trn2.
    picked = nl.ndarray((1, _F), dtype=nl.uint8, buffer=nl.sbuf)
    eid_ap = eid_sb.ap(pattern=[[1, 1], [1, 1]], offset=0)
    nisa.tensor_copy(
        dst=picked,
        src=all_sb.ap(
            pattern=[[_F, 1], [1, _F]],
            offset=0,
            scalar_offset=eid_ap,
            indirect_dim=0,
        ),
    )

    # 4. Store to HBM so the kernel output is not dead.
    out = nl.ndarray((1, _F), dtype=nl.uint8, buffer=nl.shared_hbm)
    nl.store(out, picked)
    return out


if __name__ == "__main__":
    device = xm.xla_device()
    scales = torch.randint(0, 255, (_E, _F), dtype=torch.uint8).to(device)
    eid = torch.tensor([7], dtype=torch.int32).to(device)
    try:
        r = probe(scales, eid)
        _ = r.cpu()
        print("[F4 PROBE] RESULT: PASS — trn3 accepts SBUF-resident runtime scalar_offset on tensor_copy")
        print("[F4 PROBE] action: promote [F4] scale-preloading to Round 2")
    except Exception as e:
        err = str(e)
        print(f"[F4 PROBE] RESULT: FAIL — {type(e).__name__}")
        print(f"[F4 PROBE] first 500 chars:")
        print(err[:500])
        if "NCC_INLA001" in err:
            print("[F4 PROBE] action: same constraint as trn2. Kill [F4]. Proceed without it.")
        elif "par_dim" in err or "resolve name" in err:
            print("[F4 PROBE] action: syntax error in probe — fix and re-run")
        else:
            print("[F4 PROBE] action: different error — investigate before Round 2 decision")
