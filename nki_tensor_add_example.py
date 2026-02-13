"""
Simple Working NKI Kernel - Tensor Addition
"""

import torch
import torch_xla.core.xla_model as xm
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit(platform_target="trn2")
def nki_tensor_add_kernel(a_input, b_input):
    """
    NKI kernel to compute element-wise addition of two input tensors.
    """
    # Check both input tensor shapes are the same for element-wise operation
    assert a_input.shape == b_input.shape

    # Check the first dimension does not exceed on-chip memory tile size
    assert a_input.shape[0] <= nl.tile_size.pmax

    # Allocate SBUF tiles and copy input data from HBM via DMA
    a_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_tile, src=a_input)

    b_tile = nl.ndarray(b_input.shape, dtype=b_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=b_input)

    # Compute element-wise addition using tensor_tensor ISA
    c_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

    # Create output in HBM and copy result back via DMA
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=c_output, src=c_tile)

    # Return the output tensor
    return c_output


def test_simple_add():
    """Test the simple addition kernel"""
    print("=" * 70)
    print("Simple NKI Tensor Addition Test")
    print("=" * 70)
    
    # Get XLA device
    device = xm.xla_device()
    print(f"\nUsing device: {device}")
    
    # Create test tensors (must fit in a single tile: shape[0] <= pmax)
    shape = (4, 3)
    print(f"Tensor shape: {shape}")

    a = torch.ones(shape, dtype=torch.float16, device=device)
    b = torch.ones(shape, dtype=torch.float16, device=device)

    print("\n[1/3] Computing reference (PyTorch)...")
    ref_output = a + b

    print("[2/3] Computing NKI kernel...")
    nki_output = nki_tensor_add_kernel(a, b)
    
    print("[3/3] Comparing...")
    max_diff = torch.abs(ref_output - nki_output).max().item()
    mean_diff = torch.abs(ref_output - nki_output).mean().item()
    
    print(f"\nResults:")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")
    
    if max_diff < 1e-5:
        print("\n" + "=" * 70)
        print("SUCCESS! NKI KERNEL WORKS PERFECTLY!")
        print("=" * 70)
        print("\nThis proves:")
        print("  * NKI kernels compile")
        print("  * NKI kernels execute correctly")
        print("  * Results match PyTorch exactly")
        print("\nYou can now build more complex kernels!")
        return True
    else:
        print("\nFAILED - Numerical mismatch")
        return False


if __name__ == "__main__":
    test_simple_add()
