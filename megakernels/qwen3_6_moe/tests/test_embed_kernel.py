"""Isolated correctness test for the token-embedding building block.

Exercises ``embed_fwd`` and ``embed_compose`` (token ids -> one indirect-DMA row gather -> the tp2013
SBUF residual tile) on real hardware via torch_xla. The component replaces the XLA
``ParallelEmbedding`` + ``load_residual_to_sbuf`` pair at the front of the verify megakernel and is
the mirror of lm_head at the other end of it. It serves two call sites that differ only in T: the
verify pass (T=2, one committed + one speculative token) and the MTP draft step (T=1).

Every gate runs at BOTH n_prgs=1 and n_prgs=2 (production launches at LNC=2). The distinction is not
cosmetic: at n_prgs=1, H2 == H1, the shard loop in ``natural_to_tp2013`` runs once and the
interleave term ``s*(H0*H2)`` is identically zero, so tp2013 collapses to tp102 and the
highest-risk arithmetic in the component is never evaluated. n_prgs=2 is the config that runs it.

tp_degree is 1 throughout, so rg=None and ``all_gather_embed_h`` is the identity; the TP all-gather
branch is NOT covered (see Note).

A gather is a copy, so every gate is bit-exact -- ``torch.equal``, never ``allclose``. A tolerance
would hide a misaddressed row. The oracle is ``F.embedding`` on CPU (never on the XLA device, which
would emit a second NEFF).

Gates (each at n_prgs in {1, 2}):
  G1  exactness, T=2 (verify shape)   embed_fwd == F.embedding                          torch.equal
  G2  exactness, T=1 (MTP draft)      same assertion, same code path                    torch.equal
  G3a layout, CPU closed form         embed_compose tile == the tp2013 index map        torch.equal
  G3b layout, vs megakernel           embed_compose tile == load_residual_to_sbuf tile  torch.equal
  G3c layout, literal round trip      embed_fwd == store_residual_to_hbm(load_residual_to_sbuf(o))
  G3d replication                     core 1's tile == core 0's tile
  G4  index edges                     id=0, id=V-1, id=PAD_IDX, repeated ids, distinct ids
  G5  dtype                           bf16 (production) and fp32
  G6  production shape                V=248320, H_rank=512, bf16, T=2

G3a is the load-bearing layout gate: it compares against an independent CPU implementation of
``residual[h0, t*H1 + s*H2 + h2] = emb[t, s*(H0*H2) + h0*H2 + h2]``, so it fails even if the
kernel's index derivation is wrong in a way ``load_residual_to_sbuf`` shares. G3b pins consistency
with the path being replaced; G3c is the end-to-end round trip. G1/G3c alone would pass if
``natural_to_tp2013`` and embed_fwd's inverse store were wrong in mutually cancelling ways.

At n_prgs=2 the layout gate carries two negative controls, asserted inline so it cannot go vacuous:
the expected tile must differ from the same data laid out with n_prgs=1 semantics, and from itself
with the two shard blocks swapped. If either control stops firing, G3a proves nothing.

G4's pad case is the one with a real failure mode behind it: ``F.embedding``'s ``padding_idx``
affects gradients only, so the runtime value of the pad row is w[PAD_IDX], never zeros. The gate
asserts the returned row is w[PAD_IDX] AND that w[PAD_IDX] is not itself all zeros, so a
zero-returning kernel cannot pass vacuously.

Note:
    ``all_gather_embed_h`` with a real replica group is untestable in a single-process launch --
    collectives comms are uninitialized and fail at NEFF load -- so only its rg=None identity branch
    is covered here.

Run (LNC=2; covers both n_prgs=1 and n_prgs=2):
    cd /home/ubuntu/nki-moe && \
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && \
    NEURON_RT_VISIBLE_CORES=0,1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    NEURON_CC_FLAGS="--target trn2 --lnc 2" \
    python -m megakernels.qwen3_6_moe.tests.test_embed_kernel
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402
from nkilib.core.utils.kernel_helpers import (  # noqa: E402
    get_verified_program_sharding_info,
)

from nki_kernels.common import H0  # noqa: E402
from nki_kernels.embed import (  # noqa: E402
    embed_compose,
    embed_fwd,
    load_token_ids_to_sbuf,
)
from nki_kernels.megakernel.qwen36_verify_megakernel import (  # noqa: E402
    load_residual_to_sbuf,
    store_residual_to_hbm,
)

V = 248320  # vocab_size (padded)
H_RANK = 512  # hidden_size(2048) / TP(4); tp_degree=1 here, so H == H_RANK
PAD_IDX = 248044  # config.pad_token_id
SMALL_V = 4096  # fast-iteration vocab for G1-G5 (H stays at the production H_RANK)
N_PRGS_CASES = (1, 2)  # 2 is production (LNC=2); 1 degenerates tp2013 to tp102


@nki.jit
def embed_tp2013_harness(input_ids, embed_w):
    """embed_compose's residual tile stored FLAT to HBM, per core, so the host reads the tp2013 tile
    itself rather than embed_fwd's inverse of it -- and reads every core's copy, not just core 0's."""
    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "embed_tp2013_harness", (0, 1), 2
    )
    B, S = input_ids.shape
    T = B * S
    H1 = embed_w.shape[1] // H0

    ids_sb = load_token_ids_to_sbuf(input_ids, T)
    residual = embed_compose(ids_sb, embed_w, n_prgs=n_prgs)

    out = nl.ndarray((n_prgs, H0, T * H1), dtype=embed_w.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out[prg_id], src=residual)
    if n_prgs > 1:
        nisa.core_barrier(data=out, cores=(0, 1))
    return out


@nki.jit
def natural_tp2013_harness(hidden):
    """The megakernel's own entry path -- ``load_residual_to_sbuf`` on natural [B, S, H] HBM, tile
    stored flat. The reference tp2013 tile for G3b."""
    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "natural_tp2013_harness", (0, 1), 2
    )
    B, S, hdim = hidden.shape
    T = B * S
    H1 = hdim // H0

    residual = nl.ndarray((H0, T * H1), dtype=hidden.dtype, buffer=nl.sbuf)
    load_residual_to_sbuf(residual, hidden, T, H0, H1, n_prgs)

    out = nl.ndarray((H0, T * H1), dtype=hidden.dtype, buffer=nl.shared_hbm)
    if prg_id == 0:
        nisa.dma_copy(dst=out, src=residual)
    if n_prgs > 1:
        nisa.core_barrier(data=out, cores=(0, 1))
    return out


@nki.jit
def residual_roundtrip_harness(hidden):
    """store_residual_to_hbm(load_residual_to_sbuf(x)): the megakernel entry/exit pair, which is the
    already-validated definition of the layout embed_fwd must reproduce."""
    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "residual_roundtrip_harness", (0, 1), 2
    )
    B, S, hdim = hidden.shape
    T = B * S
    H1 = hdim // H0

    residual = nl.ndarray((H0, T * H1), dtype=hidden.dtype, buffer=nl.sbuf)
    load_residual_to_sbuf(residual, hidden, T, H0, H1, n_prgs)

    out = nl.ndarray((B, S, hdim), dtype=hidden.dtype, buffer=nl.shared_hbm)
    if prg_id == 0:
        store_residual_to_hbm(out, residual, T, H0, H1, n_prgs)
    if n_prgs > 1:
        nisa.core_barrier(data=out, cores=(0, 1))
    return out


def make_weight(vocab, hdim, dtype, seed):
    """[V, H_rank] table in ``ParallelEmbedding(shard_across_embedding=True).weight`` layout, consumed
    verbatim. Rows are i.i.d. normal, so any misaddressed row shows up as a bit mismatch."""
    torch.manual_seed(seed)
    return (torch.randn(vocab, hdim) * 0.02).to(dtype).contiguous()


def oracle(ids, w):
    """CPU oracle: the gather itself. No math, so no dtype promotion -- w's rows are copied verbatim,
    which is exactly the kernel's contract."""
    return F.embedding(ids.long(), w).reshape(1, ids.numel(), w.shape[1])


def tp2013_cpu(emb_th, n_prgs):
    """[T, H] natural -> [H0, T*H1] tp2013, straight from the component docstring:
    residual[h0, t*H1 + s*H2 + h2] = emb[t, s*(H0*H2) + h0*H2 + h2]."""
    t, hdim = emb_th.shape
    h1 = hdim // H0
    h2 = h1 // n_prgs
    return emb_th.reshape(t, n_prgs, H0, h2).permute(2, 0, 1, 3).reshape(H0, t * h1)


def swap_shard_blocks(tile, t, n_prgs):
    """Reverse the shard-interleave blocks inside each token's H1 span -- a wrong-``s`` layout."""
    h1 = tile.shape[1] // t
    return tile.reshape(H0, t, n_prgs, h1 // n_prgs).flip(2).reshape(H0, t * h1)


def _dev():
    import torch_xla.core.xla_model as xm

    return xm.xla_device()


def run_embed(ids, w, n_prgs):
    return embed_fwd[n_prgs](ids.to(_dev()), w.to(_dev()), 1).cpu()


def run_embed_tiles(ids, w, n_prgs):
    """[n_prgs, H0, T*H1] -- one tp2013 tile per core."""
    return embed_tp2013_harness[n_prgs](ids.to(_dev()), w.to(_dev())).cpu()


def run_natural_tile(hidden, n_prgs):
    return natural_tp2013_harness[n_prgs](hidden.contiguous().to(_dev())).cpu()


def run_roundtrip(hidden, n_prgs):
    return residual_roundtrip_harness[n_prgs](hidden.contiguous().to(_dev())).cpu()


def _check_exact(name, got, want, extra=""):
    """Bit-exact gate. A gather has no numerical error, so any diff at all is a real bug."""
    ok = got.shape == want.shape and torch.equal(got, want)
    if ok:
        n_bad, first = 0, ""
    else:
        diff = got.float() != want.float()
        n_bad = int(diff.sum())
        idx = diff.nonzero()
        first = f"  first_mismatch={idx[0].tolist()}" if len(idx) else ""
    print(
        f"[{name}] {'PASS' if ok else 'FAIL'}  shape={tuple(got.shape)} "
        f"dtype={got.dtype}  n_mismatch={n_bad}/{got.numel()}{first}  {extra}"
    )
    assert ok, f"{name}: {n_bad}/{got.numel()} elements differ{first}"


def run_exactness(name, ids, vocab, hdim, dtype, seed, n_prgs):
    """G1/G2/G4/G5/G6 body: embed_fwd vs the CPU gather, bit-exact."""
    w = make_weight(vocab, hdim, dtype, seed)
    _check_exact(
        name, run_embed(ids, w, n_prgs), oracle(ids, w), f"ids={ids.flatten().tolist()}"
    )


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------
def test_g1_exact_t2():
    """G1: the verify-trunk shape, T = B*S = 2."""
    ids = torch.tensor([[17, 3300]], dtype=torch.int32)
    for n in N_PRGS_CASES:
        run_exactness(f"G1/bf16/T2/n{n}", ids, SMALL_V, H_RANK, torch.bfloat16, 1, n)


def test_g2_exact_t1():
    """G2: the MTP draft shape, T = 1, through the SAME code path as T=2."""
    ids = torch.tensor([[2049]], dtype=torch.int32)
    for n in N_PRGS_CASES:
        run_exactness(f"G2/bf16/T1/n{n}", ids, SMALL_V, H_RANK, torch.bfloat16, 1, n)


def test_g3_layout():
    """G3: pin the tp2013 layout against the closed-form index map, the megakernel entry path and the
    end-to-end round trip -- at both n_prgs, since only n_prgs=2 evaluates the shard term."""
    w = make_weight(SMALL_V, H_RANK, torch.bfloat16, seed=1)
    for t, ids in [
        (2, torch.tensor([[17, 3300]], dtype=torch.int32)),
        (1, torch.tensor([[2049]], dtype=torch.int32)),
    ]:
        natural = oracle(ids, w)  # [1, T, H] CPU
        flat = natural.reshape(t, H_RANK)
        for n in N_PRGS_CASES:
            tag = f"T{t}/n{n}"
            tiles = run_embed_tiles(ids, w, n)
            want = tp2013_cpu(flat, n)

            if n > 1:
                assert not torch.equal(want, tp2013_cpu(flat, 1)), (
                    f"{tag}: tp2013 at n_prgs={n} is indistinguishable from n_prgs=1, so "
                    "G3a cannot detect a wrong shard term"
                )
                assert not torch.equal(want, swap_shard_blocks(want, t, n)), (
                    f"{tag}: swapping the shard blocks is undetectable; G3a is vacuous"
                )

            _check_exact(f"G3a/cpu-closed-form/{tag}", tiles[0], want)
            _check_exact(
                f"G3b/vs-megakernel/{tag}", tiles[0], run_natural_tile(natural, n)
            )
            _check_exact(
                f"G3c/roundtrip/{tag}", run_embed(ids, w, n), run_roundtrip(natural, n)
            )
            for c in range(1, n):
                _check_exact(f"G3d/core{c}-vs-core0/{tag}", tiles[c], tiles[0])


def test_g4_index_edges():
    """G4: first row, last row, repeats and distinct ids in one batch."""
    w = make_weight(SMALL_V, H_RANK, torch.bfloat16, seed=2)
    for label, ids in [
        ("zero+last", torch.tensor([[0, SMALL_V - 1]], dtype=torch.int32)),
        ("repeated", torch.tensor([[1234, 1234]], dtype=torch.int32)),
        ("distinct", torch.tensor([[7, 4001]], dtype=torch.int32)),
        ("zero/T1", torch.tensor([[0]], dtype=torch.int32)),
        ("last/T1", torch.tensor([[SMALL_V - 1]], dtype=torch.int32)),
    ]:
        for n in N_PRGS_CASES:
            _check_exact(
                f"G4/{label}/n{n}",
                run_embed(ids, w, n),
                oracle(ids, w),
                f"ids={ids.flatten().tolist()}",
            )


def test_g5_dtype_fp32():
    """G5: fp32 alongside the production bf16 covered by G1/G2."""
    for label, ids in [
        ("T2", torch.tensor([[17, 3300]], dtype=torch.int32)),
        ("T1", torch.tensor([[2049]], dtype=torch.int32)),
    ]:
        for n in N_PRGS_CASES:
            run_exactness(
                f"G5/fp32/{label}/n{n}", ids, SMALL_V, H_RANK, torch.float32, 3, n
            )


def test_g6_production_shape():
    """G6: the real table (V=248320, H_rank=512, bf16, T=2), plus the padding_idx row.

    The pad row must come back as w[PAD_IDX]; ``F.embedding``'s ``padding_idx`` is a gradient-only
    concept, so a zero row here would be a real bug. The gate checks w[PAD_IDX] is itself nonzero so
    a zero-returning kernel cannot pass by coincidence."""
    w = make_weight(V, H_RANK, torch.bfloat16, seed=4)
    assert w[PAD_IDX].abs().sum().item() > 0, "pad row of the test table is zero; gate is vacuous"

    ids = torch.tensor([[0, V - 1]], dtype=torch.int32)
    pad_ids = torch.tensor([[PAD_IDX, 12345]], dtype=torch.int32)
    for n in N_PRGS_CASES:
        _check_exact(
            f"G6/edges/n{n}", run_embed(ids, w, n), oracle(ids, w), f"V={V} H={H_RANK}"
        )
        got = run_embed(pad_ids, w, n)
        _check_exact(f"G6/pad-row/n{n}", got, oracle(pad_ids, w), f"pad_idx={PAD_IDX}")
        assert torch.equal(got[0, 0], w[PAD_IDX]), "pad row was not returned verbatim"
        assert got[0, 0].abs().sum().item() > 0, "pad row came back as zeros"
        print(
            f"[G6/pad-row-nonzero/n{n}] PASS  |w[{PAD_IDX}]|_1="
            f"{got[0, 0].abs().sum().item():.4f}"
        )


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ALL CASES PASSED")


if __name__ == "__main__":
    main()
