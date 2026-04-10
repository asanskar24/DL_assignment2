from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    
    Architecture:
        Block 1: Conv(64)           + BN + ReLU → MaxPool
        Block 2: Conv(128)          + BN + ReLU → MaxPool
        Block 3: Conv(256) x2       + BN + ReLU → MaxPool
        Block 4: Conv(512) x2       + BN + ReLU → MaxPool
        Block 5: Conv(512) x2       + BN + ReLU → MaxPool
    
    BatchNorm is placed after each Conv and before ReLU for training stability.
    CustomDropout is placed after Block 4 and Block 5 to regularize deeper,
    more task-specific features while preserving generic low-level features.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model.
        
        Args:
            in_channels: Number of input channels (3 for RGB).
        """
        super().__init__()

        # Block 1: (B, 3, 224, 224) → (B, 64, 112, 112)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: (B, 64, 112, 112) → (B, 128, 56, 56)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: (B, 128, 56, 56) → (B, 256, 28, 28)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 4: (B, 256, 28, 28) → (B, 512, 14, 14)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CustomDropout(p=0.2)   # Light dropout after block 4
        )

        # Block 5: (B, 512, 14, 14) → (B, 512, 7, 7)
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CustomDropout(p=0.3)   # Slightly stronger dropout at bottleneck
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        f1 = self.block1(x)   # (B, 64,  H/2,  W/2)
        f2 = self.block2(f1)  # (B, 128, H/4,  W/4)
        f3 = self.block3(f2)  # (B, 256, H/8,  W/8)
        f4 = self.block4(f3)  # (B, 512, H/16, W/16)
        f5 = self.block5(f4)  # (B, 512, H/32, W/32)  ← bottleneck

        if return_features:
            features = {
                'block1': f1,
                'block2': f2,
                'block3': f3,
                'block4': f4,
                'block5': f5,
            }
            return f5, features

        return f5
