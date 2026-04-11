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
            1. classifier       : 37-class breed prediction
            2. regressor        : bounding box regression [cx, cy, w, h]
            3. Segmentation decoder: U-Net decoder for pixel-wise mask prediction

    Attribute names match checkpoint key prefixes exactly:
        classifier.pth  →  encoder.*  +  classifier.*
        localizer.pth   →  regressor.*
        unet.pth        →  decoder4.* decoder3.* decoder2.* decoder1.*
                           final_upsample.*  output_conv.*
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
        super().__init__()

        import gdown
        gdown.download(id="1jxL0-hvjiondA2OXDguubfh3_Lr2xXqZ", output=classifier_path, quiet=False)
        gdown.download(id="1Q1AJzjX8b030qOslm8icQxJ3O17Y-wd8", output=localizer_path,  quiet=False)
        gdown.download(id="1SqOCrf3vLrpULxU7x2VbZKBFrdumGnJx", output=unet_path,       quiet=False)

        self.image_size = image_size

        # ── Shared backbone ──────────────────────────────────────────────────
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ── Classification head ──────────────────────────────────────────────
        # NOTE: attribute is named 'classifier' to match keys in classifier.pth
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
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
            nn.Linear(4096, num_breeds),
        )

        # ── Localization head ────────────────────────────────────────────────
        # NOTE: attribute is named 'regressor' to match keys in localizer.pth
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
            nn.Sigmoid(),
        )

        # ── Segmentation head (U-Net decoder) ────────────────────────────────
        self.decoder4       = DecoderBlock(512, 512, 256, dropout_p=dropout_p * 0.5)
        self.decoder3       = DecoderBlock(256, 256, 128, dropout_p=dropout_p * 0.5)
        self.decoder2       = DecoderBlock(128, 128, 64,  dropout_p=0.0)
        self.decoder1       = DecoderBlock(64,  64,  32,  dropout_p=0.0)
        self.final_upsample = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.output_conv    = nn.Conv2d(32, seg_classes, kernel_size=1)

        # ── Load pretrained weights ──────────────────────────────────────────
        self._load_pretrained_weights(classifier_path, localizer_path, unet_path)

    def _load_pretrained_weights(
        self,
        classifier_path: str,
        localizer_path: str,
        unet_path: str,
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def _load(path):
            obj = torch.load(path, map_location=device)
            return obj.state_dict() if not isinstance(obj, dict) else obj

        # ── 1. Classifier: encoder + classifier head ─────────────────────────
        clf = _load(classifier_path)

        enc_state = {k.replace('encoder.', ''): v
                     for k, v in clf.items() if k.startswith('encoder.')}
        m, u = self.encoder.load_state_dict(enc_state, strict=True)
        print(f"[Encoder from classifier] missing={m}  unexpected={u}")

        clf_head = {k.replace('classifier.', ''): v
                    for k, v in clf.items() if k.startswith('classifier.')}
        m, u = self.classifier.load_state_dict(clf_head, strict=True)
        print(f"[classifier head]         missing={m}  unexpected={u}")

        # ── 2. Localizer: regressor head only ────────────────────────────────
        loc = _load(localizer_path)

        reg_state = {k.replace('regressor.', ''): v
                     for k, v in loc.items() if k.startswith('regressor.')}
        m, u = self.regressor.load_state_dict(reg_state, strict=True)
        print(f"[regressor head]          missing={m}  unexpected={u}")

        # ── 3. UNet: decoder ONLY — do NOT override encoder ──────────────────
        # The classifier head and regressor were trained against the classifier
        # encoder. Overriding encoder with the UNet encoder causes F1 -> 0.
        unet = _load(unet_path)

        for prefix, module in [
            ('decoder4',       self.decoder4),
            ('decoder3',       self.decoder3),
            ('decoder2',       self.decoder2),
            ('decoder1',       self.decoder1),
            ('final_upsample', self.final_upsample),
            ('output_conv',    self.output_conv),
        ]:
            sub = {k[len(prefix)+1:]: v
                   for k, v in unet.items() if k.startswith(prefix + '.')}
            if sub:
                m, u = module.load_state_dict(sub, strict=True)
                print(f"[{prefix}]  missing={m}  unexpected={u}")
            else:
                print(f"[{prefix}]  WARNING — no weights found in unet.pth!")

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
        bottleneck, features = self.encoder(x, return_features=True)

        # Classification + Localization share pooled bottleneck features
        pooled     = self.adaptive_pool(bottleneck)
        cls_logits = self.classifier(pooled)
        bbox       = self.regressor(pooled) * self.image_size

        # Segmentation uses full spatial hierarchy via skip connections
        d4  = self.decoder4(bottleneck,  features['block4'])
        d3  = self.decoder3(d4,          features['block3'])
        d2  = self.decoder2(d3,          features['block2'])
        d1  = self.decoder1(d2,          features['block1'])
        seg_logits = self.output_conv(self.final_upsample(d1))

        return {
            'classification': cls_logits,
            'localization':   bbox,
            'segmentation':   seg_logits,
        }