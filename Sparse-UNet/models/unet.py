import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SA import EnhancedSparseAttention


class UNetWithSparseAttention(nn.Module):
    def __init__(self):
        super(UNetWithSparseAttention, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            EnhancedSparseAttention(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            EnhancedSparseAttention(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            EnhancedSparseAttention(256, 256)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            EnhancedSparseAttention(128, 128)
        )
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            EnhancedSparseAttention(64, 64)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        
        self.final_layer = nn.Conv2d(64 + 64, 1, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        
        bottleneck = self.bottleneck(enc2)
        
        dec2 = self.decoder2(bottleneck)
        dec2_up = self.up2(dec2)
        
        dec1 = self.decoder1(torch.cat([dec2_up, enc2], dim=1))
        dec1_up = self.up1(dec1)
        
        out = self.final_layer(torch.cat([dec1_up, enc1], dim=1))
        
        return torch.sigmoid(out)

