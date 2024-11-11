import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load_inline
import seaborn as sns
import numpy as np
from enum import Enum
from dataclasses import dataclass

# Custom CUDA kernel for sparse attention
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sparse_attention_forward_kernel(
    const float* query,
    const float* key,
    const float* value,
    const bool* mask,
    float* output,
    const int B,
    const int H,
    const int L,
    const int D) {
    
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int i = threadIdx.x;
    
    if (i >= L) return;
    
    extern __shared__ float shared_mem[];
    float* key_shared = shared_mem;
    float* value_shared = key_shared + D * L;
    
    for (int j = 0; j < L; j++) {
        float qk = 0.0f;
        for (int d = 0; d < D; d++) {
            qk += query[b * H * L * D + h * L * D + i * D + d] *
                  key[b * H * L * D + h * L * D + j * D + d];
        }
        qk /= sqrt(float(D));
        
        if (!mask[b * H * L * L + h * L * L + i * L + j]) {
            qk = -1e9f;
        }
        
        key_shared[i * L + j] = qk;
    }
    __syncthreads();
    
    // Compute softmax
    float max_val = -1e9f;
    for (int j = 0; j < L; j++) {
        max_val = max(max_val, key_shared[i * L + j]);
    }
    
    float sum = 0.0f;
    for (int j = 0; j < L; j++) {
        key_shared[i * L + j] = exp(key_shared[i * L + j] - max_val);
        sum += key_shared[i * L + j];
    }
    
    for (int j = 0; j < L; j++) {
        key_shared[i * L + j] /= sum;
    }
    __syncthreads();
    
    for (int d = 0; d < D; d++) {
        float acc = 0.0f;
        for (int j = 0; j < L; j++) {
            acc += key_shared[i * L + j] *
                   value[b * H * L * D + h * L * D + j * D + d];
        }
        output[b * H * L * D + h * L * D + i * D + d] = acc;
    }
}

torch::Tensor sparse_attention_cuda_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask) {
    
    const int B = query.size(0);
    const int H = query.size(1);
    const int L = query.size(2);
    const int D = query.size(3);
    
    auto output = torch::empty_like(query);
    
    const dim3 blocks(B, H);
    const int threads = L;
    const size_t shared_mem_size = sizeof(float) * L * (D + L);
    
    sparse_attention_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        mask.data_ptr<bool>(),
        output.data_ptr<float>(),
        B, H, L, D);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sparse_attention_cuda_forward, "Sparse Attention forward (CUDA)");
}
"""

# Load CUDA kernel
sparse_attention_cuda = load_inline(
    name="sparse_attention_cuda",
    cpp_sources="",
    cuda_sources=cuda_source,
    functions=["forward"],
    with_cuda=True
)

class SparsityPattern(Enum):
    RANDOM = "random"
    STRIDED = "strided"
    BLOCK = "block"
    AXIAL = "axial"
    LONGFORMER = "longformer"

@dataclass
class AttentionStats:
    mean: float
    std: float
    sparsity: float
    pattern_type: SparsityPattern
    head_importance: torch.Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class EnhancedSparseAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        attention_ratio: float = 0.5,
        dropout: float = 0.1,
        use_bias: bool = True,
        skip_connection: bool = True,
        sparsity_pattern: SparsityPattern = SparsityPattern.RANDOM,
        use_relative_pos: bool = True,
        max_relative_positions: int = 32,
        use_learnable_pos: bool = True,
        kernel_size: int = 7  # for axial attention
    ):
        """
        Initialize Enhanced Sparse Attention module.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_heads (int): Number of attention heads
            attention_ratio (float): Ratio of attention connections to keep
            dropout (float): Dropout probability
            use_bias (bool): Whether to use bias in linear projections
            skip_connection (bool): Whether to use skip connection
            sparsity_pattern (SparsityPattern): Type of sparsity pattern to use
            use_relative_pos (bool): Whether to use relative positional encoding
            max_relative_positions (int): Maximum relative position for positional encoding
            use_learnable_pos (bool): Whether to use learnable positional encoding
            kernel_size (int): Kernel size for local attention in certain sparsity patterns
        """
        super().__init__()
        
        # [Previous initialization code remains the same]
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the sparse attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            mask (Optional[torch.Tensor]): Optional attention mask
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        L = H * W
        residual = x
        
        # Add positional encoding if enabled
        if self.learnable_pos is not None:
            x = x + self.learnable_pos
        
        # Compute query, key, value projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape and transpose for attention
        def reshape_for_attention(t: torch.Tensor) -> torch.Tensor:
            t = t.view(B, self.num_heads, self.head_dim, L)
            return t.transpose(2, 3)
        
        q = reshape_for_attention(q)  # B, num_heads, L, head_dim
        k = reshape_for_attention(k)  # B, num_heads, L, head_dim
        v = reshape_for_attention(v)  # B, num_heads, L, head_dim
        
        # Compute relative positions if enabled
        relative_positions = None
        if self.rel_pos_enc is not None:
            relative_positions = self._get_relative_positions(L)
        
        # Compute attention with custom CUDA kernel if available
        try:
            # Attempt to use custom CUDA kernel
            out = sparse_attention_cuda.forward(
                q, k, v,
                self._generate_sparsity_mask(
                    (B, self.num_heads, L, L),
                    x.device
                )
            )
        except:
            # Fallback to regular implementation
            out = self._compute_attention(q, k, v, mask, relative_positions)
        
        # Reshape output and project
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.out_proj(out)
        
        # Apply skip connection and normalization
        if self.skip_connection:
            # Permute for LayerNorm and back
            out = self.norm(
                (residual + out).permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)
        else:
            out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return out

    def get_attention_stats(self) -> AttentionStats:
        """Return current attention statistics."""
        return AttentionStats(
            mean=self.attention_stats[0].item(),
            std=self.attention_stats[1].item(),
            sparsity=self.attention_stats[2].item(),
            pattern_type=self.sparsity_pattern,
            head_importance=self.head_importance.detach()
        )
    
    def reset_stats(self):
        """Reset attention statistics."""
        self.attention_stats.zero_()
        self.attention_maps.clear()
    
    def prune_heads(self, heads_to_prune: List[int]):
        """
        Prune specified attention heads.
        
        Args:
            heads_to_prune (List[int]): List of head indices to prune
        """
        mask = torch.ones_like(self.head_importance)
        mask[heads_to_prune] = 0
        self.head_importance.data *= mask
    
    @torch.no_grad()
    def analyze_head_importance(self) -> Dict[int, float]:
        """
        Analyze and return the importance of each attention head.
        
        Returns:
            Dict[int, float]: Dictionary mapping head index to importance score
        """
        importance_scores = self.head_importance.cpu().tolist()
        return {i: score for i, score in enumerate(importance_scores)}
    
    def visualize_attention(self, save_path: Optional[str] = None):
        """
        Visualize attention patterns and save to file if path provided.
        
        Args:
            save_path (Optional[str]): Path to save visualization
        """
        if not self.attention_maps:
            print("No attention maps available. Run forward pass first.")
            return
        
        attention_map = next(iter(self.attention_maps.values()))
        B, H, L, _ = attention_map.shape
        
        fig, axes = plt.subplots(2, H//2, figsize=(15, 8))
        axes = axes.flatten()
        
        for h in range(H):
            ax = axes[h]
            attn = attention_map[0, h].cpu()
            sns.heatmap(attn, ax=ax, cmap='viridis')
            ax.set_title(f'Head {h}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

# Additional utility functions for the module
def create_attention_mask(
    height: int,
    width: int,
    window_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create a local attention mask for a given window size.
    
    Args:
        height (int): Height of feature map
        width (int): Width of feature map
        window_size (int): Size of local attention window
        device (torch.device): Device to create mask on
        
    Returns:
        torch.Tensor: Boolean attention mask
    """
    mask = torch.zeros(height * width, height * width, device=device, dtype=torch.bool)
    
    for i in range(height):
        for j in range(width):
            cur_idx = i * width + j
            for di in range(-window_size//2, window_size//2 + 1):
                for dj in range(-window_size//2, window_size//2 + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_idx = ni * width + nj
                        mask[cur_idx, neighbor_idx] = True
    
    return mask

def get_relative_positions_2d(
    height: int,
    width: int,
    max_relative_position: int
) -> torch.Tensor:
    """
    Compute 2D relative positions for image data.
    
    Args:
        height (int): Height of feature map
        width (int): Width of feature map
        max_relative_position (int): Maximum relative position
        
    Returns:
        torch.Tensor: Tensor of relative positions
    """
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(height),
            torch.arange(width)
        )
    ).flatten(1)
    
    relative_coords = coords[:, :, None] - coords[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0)
    
    relative_coords = torch.clamp(
        relative_coords,
        -max_relative_position,
        max_relative_position
    )
    
    return relative_coords + max_relative_position