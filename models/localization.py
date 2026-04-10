"""Localization modules
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer.
    
    Uses VGG11Encoder as the feature extractor backbone, followed by a 
    regression head that predicts bounding box coordinates in pixel space.
    
    Output format: [x_center, y_center, width, height] in original image 
    pixel coordinates (not normalized).
    
    The regression head uses Sigmoid on the output and scales by image size
    to produce pixel-space coordinates. This constrains predictions to valid
    image regions and avoids unbounded regression outputs.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5, image_size: int = 224):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
            image_size: Input image size (assumed square). Used to scale
                        normalized sigmoid outputs to pixel coordinates.
        """
        super().__init__()

        self.image_size = image_size

        # Shared convolutional backbone (pretrained weights can be loaded here)
        self.encoder = VGG11(in_channels=in_channels)

        # Pool bottleneck to fixed spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Regression head: predicts 4 bbox coordinates
        self.regressor = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(1024, 4),
            nn.Sigmoid()   # Outputs values in [0, 1], scaled to pixel space below
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height)
            format in original image pixel space (not normalized values).
        """
        # Extract features
        features = self.encoder(x)

        # Pool to fixed spatial size
        pooled = self.adaptive_pool(features)

        # Predict normalized bbox coordinates in [0, 1]
        bbox_normalized = self.regressor(pooled)

        # Scale from [0, 1] to pixel coordinates
        bbox_pixels = bbox_normalized * self.image_size

        return bbox_pixels
