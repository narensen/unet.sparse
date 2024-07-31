import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSparseAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, attention_ratio=0.5, dropout=0.1):
        super(EnhancedSparseAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.attention_ratio = attention_ratio
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.out_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("sparse_mask", None)
    
    def _get_sparse_mask(self, attn_shape):
        if self.training or self.sparse_mask is None:
            mask = torch.rand(attn_shape) < self.attention_ratio
            if not self.training:
                self.sparse_mask = mask
            return mask
        return self.sparse_mask
    
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        def reshape(x):
            return x.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        
        q = reshape(self.query(x))
        k = reshape(self.key(x))
        v = reshape(self.value(x))
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        sparse_mask = self._get_sparse_mask(attn_scores.shape).to(attn_scores.device)
        attn_scores = attn_scores.masked_fill(~sparse_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        out = torch.matmul(attn_probs, v)
        
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        
        out = self.out_proj(out)
        
        out = self.norm(residual + out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return out

