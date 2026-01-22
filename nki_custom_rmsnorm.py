import math
import torch
import torch.nn as nn
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_rmsnorm_kernel(input_tensor, weight, eps):
    """
    RMSNorm NKI kernel - based on AWS official tutorial pattern.
    
    Args:
        input_tensor: Input tensor [batch*seq_len, hidden_size]
        weight: RMSNorm weight parameter [hidden_size]
        eps: Small epsilon for numerical stability
    
    Returns:
        output: Normalized tensor with same shape as input
    """
    # Create output tensor in shared HBM
    output = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)
    
    # Make sure shapes match
    assert input_tensor.shape[1] == weight.shape[0]
    
    # Generate tensor indices
    ix = nl.arange(128)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(input_tensor.shape[1])[None, :]
    
    num_rows = input_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    
    # Load RMSNorm weight once, reused by all rows
    g_tile = nl.load(weight.reshape((1, weight.shape[0]))[iw, iy])
    
    # Process 128 rows at a time due to 128-partition tile size limitation
    for i in nl.affine_range(math.ceil(input_tensor.shape[0]/128)):
        # Load input data from HBM to SBUF
        a_tile = nl.load(input_tensor[i * 128 + ix, iy],
                        mask=(i * 128 + ix < num_rows))
        
        # Compute element-wise square
        in_square = nl.square(a_tile)
        
        # Calculate sum of squared elements along last dimension
        square_sum = nl.sum(in_square, axis=[1])
        
        # Compute mean
        mean = square_sum / hidden_size
        
        # Take reciprocal of sqrt with eps
        rms_reciprocal = nl.rsqrt(mean + eps)
        
        # Normalize: multiply input by reciprocal of RMS
        out_tile = nl.multiply(a_tile, rms_reciprocal)
        
        # Broadcast weight along first axis to match tensor shape
        g_bcast = g_tile.broadcast_to((128, hidden_size))
        
        # Multiply with the RMSNorm weight
        out_tile[...] = nl.multiply(out_tile, g_bcast,
                               mask=(i * 128 + ix < num_rows))
        
        # Store results back to HBM
        nl.store(output[i * 128 + ix, iy], value=out_tile,
                mask=(i * 128 + ix < num_rows))
    
    return output


class NKIRMSNorm(nn.Module):
    """
    NKI-accelerated RMSNorm layer compatible with NxDI.
    """
    
    def __init__(self, hidden_size, eps=1e-6):
        """
        Initialize NKI RMSNorm layer.
        
        Args:
            hidden_size: Size of the hidden dimension
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
        # Enable NKI kernel for all hidden sizes
        self.use_nki = True
        
        print(f"Info: Using NKI RMSNorm kernel for hidden_size={hidden_size}")
        
    def forward(self, x):
        """
        Forward pass using NKI kernel or fallback.
        
        Args:
            x: Input tensor of various shapes
            
        Returns:
            Normalized tensor with same shape as input
        """
        if not self.use_nki:
            return self._fallback_forward(x)
        
        original_shape = x.shape
        
        # Handle various input shapes by flattening to 2D
        if x.dim() >= 2:
            x = x.view(-1, x.shape[-1])
        else:
            raise ValueError(f"Expected input with at least 2 dimensions, got {x.dim()}D")
        
        # Call NKI kernel directly
        output = nki_rmsnorm_kernel(x, self.weight, self.eps)
        
        # Reshape back to original shape
        output = output.view(original_shape)
        
        return output
    
    def _fallback_forward(self, x):
        """Fallback RMSNorm implementation."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x / torch.sqrt(variance + self.eps)
        return x_normalized * self.weight
    
    def extra_repr(self):
        """String representation for debugging."""
        return f'hidden_size={self.hidden_size}, eps={self.eps}, use_nki={self.use_nki}'


def get_nki_rmsnorm_cls():
    """
    Factory function to return NKI RMSNorm class.
    """
    return NKIRMSNorm
