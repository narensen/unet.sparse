# Enhanced Sparse Attention
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, customizable implementation of Sparse Attention for computer vision tasks. This implementation includes various sparsity patterns, CUDA optimization, and visualization tools.

## Features

- 🚀 Multiple sparsity patterns (Random, Block, Strided, Axial, Longformer)
- ⚡ Custom CUDA kernel for optimized performance
- 📊 Built-in visualization and monitoring tools
- 🔧 Relative and learnable positional encodings
- 📈 Attention statistics tracking
- 🎯 Head importance analysis for pruning
- 🔄 Skip connections and residual learning

## Installation

```bash
# Clone the repository
git clone https://github.com/narensen/enhanced-sparse-attention.git
cd enhanced-sparse-attention

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from sparse_attention import EnhancedSparseAttention

# Initialize the module
attention = EnhancedSparseAttention(
    in_channels=256,
    out_channels=256,
    num_heads=8,
    attention_ratio=0.5
)

# Process input
x = torch.randn(4, 256, 32, 32)  # [batch_size, channels, height, width]
output = attention(x)
```

## Usage Examples

### 1. Basic CNN with Sparse Attention

```python
import torch.nn as nn
from sparse_attention import EnhancedSparseAttention, SparsityPattern

class CNNWithSparseAttention(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.attention = EnhancedSparseAttention(
            in_channels=64,
            out_channels=64,
            num_heads=8,
            attention_ratio=0.5,
            sparsity_pattern=SparsityPattern.AXIAL
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        return self.classifier(x)

# Create model and process batch
model = CNNWithSparseAttention()
x = torch.randn(4, 3, 32, 32)
output = model(x)
```

### 2. Different Sparsity Patterns

```python
# Block-sparse attention
block_attention = EnhancedSparseAttention(
    in_channels=256,
    out_channels=256,
    sparsity_pattern=SparsityPattern.BLOCK,
    attention_ratio=0.5
)

# Axial attention
axial_attention = EnhancedSparseAttention(
    in_channels=256,
    out_channels=256,
    sparsity_pattern=SparsityPattern.AXIAL,
    attention_ratio=0.5
)

# Longformer-style attention
longformer_attention = EnhancedSparseAttention(
    in_channels=256,
    out_channels=256,
    sparsity_pattern=SparsityPattern.LONGFORMER,
    attention_ratio=0.5,
    kernel_size=7  # local attention window size
)
```

### 3. Monitoring and Visualization

```python
# Monitor attention statistics
def train_step(model, x):
    output = model(x)
    stats = model.attention.get_attention_stats()
    print(f"Mean attention: {stats.mean:.3f}")
    print(f"Sparsity: {stats.sparsity:.3f}")
    return output

# Visualize attention patterns
from sparse_attention.utils import visualize_attention_patterns
fig = visualize_attention_patterns(model.attention, input_tensor)
plt.show()
```

## Advanced Features

### Custom CUDA Kernel

The implementation includes a custom CUDA kernel for optimized sparse attention computation. It automatically falls back to PyTorch implementation if CUDA is not available.

```python
# Enable/disable CUDA kernel
attention = EnhancedSparseAttention(
    in_channels=256,
    out_channels=256,
    use_cuda_kernel=True  # Set to False to use PyTorch implementation
)
```

### Positional Encodings

```python
# Use relative positional encoding
attention = EnhancedSparseAttention(
    in_channels=256,
    out_channels=256,
    use_relative_pos=True,
    max_relative_positions=32
)

# Use learnable positional encoding
attention = EnhancedSparseAttention(
    in_channels=256,
    out_channels=256,
    use_learnable_pos=True
)
```

### Head Importance Analysis

```python
# Get head importance scores
importance = attention.head_importance.detach()
print(f"Head importance: {importance}")

# Prune least important heads
threshold = importance.mean()
attention.head_importance.data *= (importance > threshold)
```

## Benchmarks

| Sparsity Pattern | Memory Usage | Forward Time | Backward Time |
|-----------------|--------------|--------------|---------------|
| Dense           | 1.00x        | 1.00x        | 1.00x        |
| Random (50%)    | 0.52x        | 0.48x        | 0.51x        |
| Block           | 0.31x        | 0.29x        | 0.33x        |
| Axial           | 0.25x        | 0.22x        | 0.24x        |
| Longformer      | 0.28x        | 0.26x        | 0.29x        |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{enhanced-sparse-attention,
  author = {Naren Sengodan},
  title = {Enhanced Sparse Attention},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/narensen/enhanced-sparse-attention}}
}
```
