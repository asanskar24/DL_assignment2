"""Classification components
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead.
    
    The classification head follows the original VGG design:
    AdaptiveAvgPool → Flatten → FC(4096) → BN → ReLU → Dropout
                               → FC(4096) → BN → ReLU → Dropout
                               → FC(num_classes)
    
    BatchNorm is added after each FC layer for training stability.
    CustomDropout (p=0.5) is placed after each FC+BN+ReLU block to
    prevent co-adaptation of neurons in the dense layers.
    """

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()

        # Shared convolutional backbone
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Pool bottleneck to fixed 7x7 regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Classification head: 512 * 7 * 7 = 25088 input features
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, num_classes)
            # No softmax here — CrossEntropyLoss expects raw logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        # Extract features from encoder
        features = self.encoder(x)

        # Pool to fixed spatial size
        pooled = self.adaptive_pool(features)

        # Classify
        logits = self.classifier(pooled)

        return logits
