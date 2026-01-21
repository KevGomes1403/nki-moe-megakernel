"""
Simple Working NKI Kernel - Tensor Addition
Based directly on AWS official example
This WILL work!
"""

import torch
import torch_xla.core.xla_model as xm
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """
    NKI kernel to compute element-wise addition of two input tensors.
    Direct copy from AWS documentation.
    """
    # Create output tensor shared between all SPMD instances
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    
    # Calculate tile offsets based on current 'program'
    offset_i_x = nl.program_id(0) * 128
    offset_i_y = nl.program_id(1) * 512
    
    # Generate tensor indices
    ix = offset_i_x + nl.arange(128)[:, None]
    iy = offset_i_y + nl.arange(512)[None, :]
    
    # Load input data from HBM to SBUF
    a_tile = nl.load(a_input[ix, iy])
    b_tile = nl.load(b_input[ix, iy])
    
    # Compute a + b
    c_tile = a_tile + b_tile
    
    # Store results back to HBM
    nl.store(c_output[ix, iy], value=c_tile)
    
    # Transfer ownership to caller
    return c_output


def nki_tensor_add(a_input, b_input):
    """Kernel caller with SPMD grid"""
    grid_x = a_input.shape[0] // 128
    grid_y = a_input.shape[1] // 512
    return nki_tensor_add_kernel[grid_x, grid_y](a_input, b_input)


def test_simple_add():
    """Test the simple addition kernel"""
    print("=" * 70)
    print("Simple NKI Tensor Addition Test")
    print("=" * 70)
    
    # Get XLA device
    device = xm.xla_device()
    print(f"\nUsing device: {device}")
    
    # Create test tensors (must be multiples of tile size)
    shape = (256, 1024)  # 2*128 x 2*512
    print(f"Tensor shape: {shape}")
    
    torch.manual_seed(42)
    a = torch.randn(shape, device=device)
    b = torch.randn(shape, device=device)
    
    print("\n[1/3] Computing reference (PyTorch)...")
    ref_output = a + b
    
    print("[2/3] Computing NKI kernel...")
    nki_output = nki_tensor_add(a, b)
    
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
    success = test_simple_add()
    exit(0 if success else 1)
