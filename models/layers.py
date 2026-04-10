"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer implementing inverted dropout.
    
    During training, randomly zeroes elements with probability p and scales
    remaining elements by 1/(1-p) to maintain expected values (inverted dropout).
    During evaluation, passes input through unchanged.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability. Must be between 0 and 1.
        """
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor of any shape.

        Returns:
            Output tensor with same shape as input.
        """
        # During evaluation, dropout is disabled — return input unchanged
        if not self.training:
            return x

        # Generate binary mask: 1 with probability (1-p), 0 with probability p
        mask = (torch.rand_like(x) > self.p).float()

        # Inverted dropout: scale by 1/(1-p) to keep expected value the same
        return x * mask / (1 - self.p)
