"""Segmentation model
"""
import torch
import torch.nn as nn

from .vgg11 import VGG11
from .layers import CustomDropout


class DecoderBlock(nn.Module):
    """Single decoder block: TransposedConv upsample + skip connection + Conv.
    
    Uses Transposed Convolution for learnable upsampling as required.
    Concatenates skip connection from encoder before applying convolutions.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout_p: float = 0.0):
        """
        Args:
            in_channels:   channels coming from the layer below (upsampled)
            skip_channels: channels coming from the encoder skip connection
            out_channels:  output channels after convolutions
            dropout_p:     dropout probability (0 = no dropout)
        """
        super().__init__()

        # Learnable upsampling via transposed convolution (2x spatial size)
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )

        # After concatenation: (in_channels//2 + skip_channels) input channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = CustomDropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    feature map from previous decoder layer  [B, in_channels, H, W]
            skip: skip connection from encoder             [B, skip_channels, 2H, 2W]
        Returns:
            Output feature map [B, out_channels, 2H, 2W]
        """
        x = self.upsample(x)                  # (B, in_channels//2, 2H, 2W)
        x = torch.cat([x, skip], dim=1)       # concatenate along channel dim
        x = self.conv(x)
        x = self.dropout(x)
        return x


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.

    Encoder: VGG11 backbone (blocks 1-5) with skip connections.
    Decoder: Symmetric expansive path using TransposedConv upsampling.
    
    Skip connections fuse encoder feature maps with decoder at each scale,
    allowing fine-grained spatial detail to be recovered.

    Loss: CrossEntropyLoss is used since the Oxford Pet trimap has 3 classes
    (foreground, background, boundary). CrossEntropy handles multi-class 
    pixel classification naturally and works well with class imbalance when
    combined with class weights.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.
        Args:
            num_classes: Number of output segmentation classes (3 for Pet trimap).
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()

        # Encoder: VGG11 backbone — returns skip features at each block
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Decoder: symmetric expansive path
        # block5 output: (B, 512, 7, 7)   → upsample to (B, 256, 14, 14)  + skip from block4 (512)
        self.decoder4 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256, dropout_p=dropout_p * 0.5)

        # → upsample to (B, 128, 28, 28)  + skip from block3 (256)
        self.decoder3 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128, dropout_p=dropout_p * 0.5)

        # → upsample to (B, 64, 56, 56)   + skip from block2 (128)
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=128, out_channels=64, dropout_p=0.0)

        # → upsample to (B, 32, 112, 112) + skip from block1 (64)
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32, dropout_p=0.0)

        # Final upsample to original resolution (B, 32, 224, 224) → (B, num_classes, 224, 224)
        self.final_upsample = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        # 1x1 conv to map to num_classes — no softmax, CrossEntropyLoss expects logits
        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Encoder forward with skip connections
        bottleneck, features = self.encoder(x, return_features=True)

        # Decoder: upsample + fuse skip connections at each scale
        d4 = self.decoder4(bottleneck,       features['block4'])  # (B, 256, 14, 14)
        d3 = self.decoder3(d4,               features['block3'])  # (B, 128, 28, 28)
        d2 = self.decoder2(d3,               features['block2'])  # (B, 64,  56, 56)
        d1 = self.decoder1(d2,               features['block1'])  # (B, 32,  112, 112)

        # Final upsample to full resolution
        out = self.final_upsample(d1)        # (B, 32, 224, 224)
        logits = self.output_conv(out)       # (B, num_classes, 224, 224)

        return logits
