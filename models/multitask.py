"""Unified multi-task model
"""
import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet, DecoderBlock


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.
    
    Architecture:
        - Single shared VGG11 encoder backbone
        - Three task-specific heads branching from the shared backbone:
            1. Classification head: 37-class breed prediction
            2. Localization head: bounding box regression [x_center, y_center, w, h]
            3. Segmentation head: U-Net decoder for pixel-wise mask prediction
    
    Weights are loaded from individually trained task models, then the encoder
    is shared and fine-tuned jointly during multi-task training.
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        image_size: int = 224,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
        dropout_p: float = 0.5,
    ):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            image_size: Input image size (assumed square).
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
            dropout_p: Dropout probability for task heads.
        """
        super().__init__()

        import gdown
        gdown.download(id="<1jxL0-hvjiondA2OXDguubfh3_Lr2xXqZ>", output=classifier_path, quiet=False)
        gdown.download(id="<1Q1AJzjX8b030qOslm8icQxJ3O17Y-wd8>", output=localizer_path, quiet=False)
        gdown.download(id="1SqOCrf3vLrpULxU7x2VbZKBFrdumGnJx", output=unet_path, quiet=False)

        self.image_size = image_size

        # ── Shared backbone ──────────────────────────────────────────────────
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ── Classification head ──────────────────────────────────────────────
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_breeds),
        )

        # ── Localization head ────────────────────────────────────────────────
        self.localization_head = nn.Sequential(
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
            nn.Sigmoid(),
        )

        # ── Segmentation head (U-Net decoder) ────────────────────────────────
        self.decoder4 = DecoderBlock(512, 512, 256, dropout_p=dropout_p * 0.5)
        self.decoder3 = DecoderBlock(256, 256, 128, dropout_p=dropout_p * 0.5)
        self.decoder2 = DecoderBlock(128, 128, 64,  dropout_p=0.0)
        self.decoder1 = DecoderBlock(64,  64,  32,  dropout_p=0.0)
        self.final_upsample = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.output_conv = nn.Conv2d(32, seg_classes, kernel_size=1)

        # ── Load pretrained weights from individual task models ──────────────
        self._load_pretrained_weights(classifier_path, localizer_path, unet_path)

    def _load_pretrained_weights(
        self,
        classifier_path: str,
        localizer_path: str,
        unet_path: str,
    ):
        """Load weights from individually trained task models.
        
        Encoder weights are taken from the classifier (best trained backbone).
        Each task head loads its own pretrained weights.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load classifier — take encoder + classification head weights
        clf_state = torch.load(classifier_path, map_location=device)
        encoder_state = {
            k.replace('encoder.', ''): v
            for k, v in clf_state.items() if k.startswith('encoder.')
        }
        self.encoder.load_state_dict(encoder_state)

        clf_head_state = {
            k.replace('classifier.', ''): v
            for k, v in clf_state.items() if k.startswith('classifier.')
        }
        self.classification_head.load_state_dict(clf_head_state)

        # Load localization head weights
        loc_state = torch.load(localizer_path, map_location=device)
        loc_head_state = {
            k.replace('regressor.', ''): v
            for k, v in loc_state.items() if k.startswith('regressor.')
        }
        self.localization_head.load_state_dict(loc_head_state)

        # Load segmentation decoder weights
        unet_state = torch.load(unet_path, map_location=device)
        seg_keys = ['decoder4', 'decoder3', 'decoder2', 'decoder1',
                    'final_upsample', 'output_conv']
        seg_state = {k: v for k, v in unet_state.items()
                     if any(k.startswith(sk) for sk in seg_keys)}
        self.load_state_dict(seg_state, strict=False)

        print("Pretrained weights loaded successfully.")

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization':   [B, 4] bounding box tensor (pixel coords).
            - 'segmentation':   [B, seg_classes, H, W] segmentation logits tensor.
        """
        # ── Single shared forward pass through encoder ────────────────────
        bottleneck, features = self.encoder(x, return_features=True)

        # ── Classification branch ─────────────────────────────────────────
        pooled = self.adaptive_pool(bottleneck)          # (B, 512, 7, 7)
        cls_logits = self.classification_head(pooled)    # (B, num_breeds)

        # ── Localization branch ───────────────────────────────────────────
        bbox = self.localization_head(pooled)            # (B, 4) in [0,1]
        bbox = bbox * self.image_size                    # scale to pixel space

        # ── Segmentation branch ───────────────────────────────────────────
        d4  = self.decoder4(bottleneck,    features['block4'])  # (B, 256, 14, 14)
        d3  = self.decoder3(d4,            features['block3'])  # (B, 128, 28, 28)
        d2  = self.decoder2(d3,            features['block2'])  # (B, 64,  56, 56)
        d1  = self.decoder1(d2,            features['block1'])  # (B, 32,  112,112)
        out = self.final_upsample(d1)                           # (B, 32,  224,224)
        seg_logits = self.output_conv(out)                      # (B, seg_classes, 224,224)

        return {
            'classification': cls_logits,
            'localization':   bbox,
            'segmentation':   seg_logits,
        }
