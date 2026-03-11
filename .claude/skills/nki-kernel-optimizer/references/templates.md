# NKI Kernel Templates

Copy-paste scaffolding for common kernel patterns.

---

## Template 1: Elementwise / Reduction Kernel

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit
def elementwise_kernel(input_hbm, output_hbm):
    """
    Template: elementwise or reduction op on [T, H] input.
    Tiles along T (partition dim) and processes H in full.
    """
    T, H = input_hbm.shape

    # Tile over T dimension (partition dim = 128 on trn2)
    TILE_T = nl.par_dim(128)
    n_tiles = T // TILE_T

    for t in nl.affine_range(n_tiles):
        # Load tile: [TILE_T, H]
        inp = nl.load(input_hbm[t*TILE_T:(t+1)*TILE_T, :])

        # --- Your compute here ---
        out = nl.exp(inp)           # example: elementwise exp
        # out = nl.sum(inp, axis=1) # example: reduction

        # Store tile
        nl.store(output_hbm[t*TILE_T:(t+1)*TILE_T, :], out)
```

---

## Template 2: Matrix Multiply (GEMM)

```python
@nki.jit
def matmul_kernel(lhs_hbm, rhs_hbm, output_hbm):
    """
    Template: [M, K] x [K, N] → [M, N]
    Tiles M (partition dim), streams K blocks.
    """
    M, K = lhs_hbm.shape
    _, N = rhs_hbm.shape
    TILE_M = nl.par_dim(128)
    TILE_K = 128
    n_m_tiles = M // TILE_M

    for m in nl.affine_range(n_m_tiles):
        # Accumulate in PSUM
        psum = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.psum)
        psum[:, :] = nl.zeros((TILE_M, N), dtype=nl.float32)

        for k in range(K // TILE_K):  # sequential: carries accumulation dep
            lhs_tile = nl.load(lhs_hbm[m*TILE_M:(m+1)*TILE_M, k*TILE_K:(k+1)*TILE_K])
            rhs_tile = nl.load(rhs_hbm[k*TILE_K:(k+1)*TILE_K, :])
            nisa.nc_matmul(dst=psum, stationary=lhs_tile, moving=rhs_tile)

        # PSUM → SBUF → HBM
        sbuf_tmp = nl.ndarray((TILE_M, N), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sbuf_tmp, src=psum)
        nl.store(output_hbm[m*TILE_M:(m+1)*TILE_M, :], sbuf_tmp)
```

---

## Template 3: Fused Two-Op Kernel (elementwise chain)

```python
@nki.jit
def fused_op_kernel(input_hbm, output_hbm, scale: float, bias: float):
    """
    Template: fused scale + exp + store (avoids SBUF spilling intermediate).
    """
    T, H = input_hbm.shape
    TILE_T = nl.par_dim(128)
    n_tiles = T // TILE_T

    for t in nl.affine_range(n_tiles):
        inp = nl.load(input_hbm[t*TILE_T:(t+1)*TILE_T, :])

        # op0: scale
        scaled = nl.multiply(inp, scale)

        # op1: exp (stays in SBUF, no spill)
        result = nl.exp(scaled)

        nl.store(output_hbm[t*TILE_T:(t+1)*TILE_T, :], result)
```

---

## Template 4: Correctness Test Harness

```python
import numpy as np
import torch
import torch_xla.core.xla_model as xm

def test_kernel(kernel_fn, reference_fn, shape, dtype=np.float32,
                rtol=1e-3, atol=1e-3, seed=42):
    """Verify NKI kernel matches NumPy reference."""
    rng = np.random.default_rng(seed)
    x_np = rng.random(shape).astype(dtype)
    x_torch = torch.from_numpy(x_np).to(xm.xla_device())

    # NKI result
    result = kernel_fn(x_torch).cpu().numpy()

    # Reference
    ref = reference_fn(x_np)

    max_diff = np.abs(result - ref).max()
    try:
        np.testing.assert_allclose(result, ref, rtol=rtol, atol=atol)
        status = "PASS"
    except AssertionError:
        status = "FAIL"

    print(f"[{status}] shape={shape} max_diff={max_diff:.2e} rtol={rtol} atol={atol}")
    return max_diff, status == "PASS"

# Usage:
# test_kernel(my_nki_kernel, np_reference, shape=(128, 512))
```

---

## Template 5: Wall-Clock Benchmark

```python
import time
import torch
import torch_xla.core.xla_model as xm

def bench(fn, *args, warmup=5, iters=20, label=""):
    device_args = [a.to(xm.xla_device()) if isinstance(a, torch.Tensor) else a
                   for a in args]
    for _ in range(warmup):
        fn(*device_args)
    xm.mark_step(); xm.wait_device_ops()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*device_args)
    xm.mark_step(); xm.wait_device_ops()

    ms = (time.perf_counter() - t0) / iters * 1e3
    tag = f"[{label}] " if label else ""
    print(f"{tag}{ms:.3f} ms/iter")
    return ms
```

---

## Template 6: Multi-Expert / MoE Pattern

```python
@nki.jit
def moe_kernel(hidden_hbm, weights_hbm, affinities_hbm, output_hbm,
               n_experts: int, top_k: int):
    """
    Template: MoE dispatch with top-k routing.
    hidden:     [T, H]
    weights:    [E, H, I] (down projection) or [E, I, H] (up projection)
    affinities: [T, E]  — must be float32 for POST_SCALE
    output:     [T, H]
    """
    T, H = hidden_hbm.shape
    E = n_experts

    for t in range(T):  # sequential: each token accumulates independently
        hidden_tile = nl.load(hidden_hbm[t:t+1, :])

        acc = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
        acc[:, :] = nl.zeros((1, H), dtype=nl.float32)

        for e in range(E):   # sequential: accumulates into acc
            affinity = nl.load(affinities_hbm[t:t+1, e:e+1])  # float32
            w = nl.load(weights_hbm[e, :, :])
            expert_out = nl.matmul(hidden_tile, w)
            acc = nl.add(acc, nl.multiply(expert_out, affinity))

        nl.store(output_hbm[t:t+1, :], acc)
```

---

## Template 7: Environment Setup Script

```bash
#!/bin/bash
# setup_env.sh — run before any NKI work

source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2

echo "Neuron environment ready."
python -c "import torch_xla; print('torch_xla:', torch_xla.__version__)"
```
