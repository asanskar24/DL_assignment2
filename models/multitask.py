"""Unified multi-task perception model (root-level, imported by autograder).

    from multitask import MultiTaskPerceptionModel
"""
import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout
from models.segmentation import DecoderBlock


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    Architecture:
        - Single shared VGG11 encoder backbone
        - Three task-specific heads:
            1. classification_head  → 37-class breed logits
            2. localization_head    → [cx, cy, w, h] in pixel space
            3. Segmentation decoder → (B, 3, H, W) logits

    Weights are downloaded from Google Drive on first instantiation,
    then loaded from the three saved task checkpoints.
    """

    # VGG11 paper fixes input at 224×224
    IMAGE_SIZE: int = 224

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
        dropout_p: float = 0.5,
    ):
        super().__init__()

        # ── Download checkpoints from Drive ──────────────────────────────
        import gdown
        gdown.download(id="1eRaeNVw7F-0yu8VRtQLdmg2jPy81234J", output=classifier_path, quiet=False)
        gdown.download(id="1H9r67PsumwlAaRTV-D1wFQ650vnEYPPN", output=localizer_path,  quiet=False)
        gdown.download(id="1IoaUdLttfXDuzd5tGx3WdxblUo9yYrTy", output=unet_path,       quiet=False)

        # ── Shared backbone ───────────────────────────────────────────────
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ── Classification head ───────────────────────────────────────────
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

        # ── Localization head ─────────────────────────────────────────────
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

        # ── Segmentation decoder ──────────────────────────────────────────
        self.decoder4 = DecoderBlock(512, 512, 256, dropout_p=dropout_p * 0.5)
        self.decoder3 = DecoderBlock(256, 256, 128, dropout_p=dropout_p * 0.5)
        self.decoder2 = DecoderBlock(128, 128, 64,  dropout_p=0.0)
        self.decoder1 = DecoderBlock(64,  64,  32,  dropout_p=0.0)
        self.final_upsample = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.output_conv    = nn.Conv2d(32, seg_classes, kernel_size=1)

        # ── Load pretrained weights ───────────────────────────────────────
        self._load_pretrained_weights(classifier_path, localizer_path, unet_path)

    # ─────────────────────────────────────────────────────────────────────
    def _load_pretrained_weights(
        self,
        classifier_path: str,
        localizer_path: str,
        unet_path: str,
    ) -> None:
        """Load weights from individually trained task checkpoints.

        Key mapping:
          classifier.pth  → encoder.*  +  classifier.*
          localizer.pth   → regressor.*
          unet.pth        → encoder.*  (overrides classifier encoder)
                          + decoder{1-4}.*, final_upsample.*, output_conv.*
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _load(path):
            obj = torch.load(path, map_location=device)
            # Support both torch.save(model, ...) and torch.save(model.state_dict(), ...)
            return obj.state_dict() if not isinstance(obj, dict) else obj

        # ── 1. Classifier: encoder + classification head ──────────────────
        clf = _load(classifier_path)

        encoder_state = {k.replace("encoder.", ""): v
                         for k, v in clf.items() if k.startswith("encoder.")}
        miss, unexp = self.encoder.load_state_dict(encoder_state, strict=True)
        print(f"[Encoder from classifier] missing={miss}  unexpected={unexp}")

        clf_head_state = {k.replace("classifier.", ""): v
                          for k, v in clf.items() if k.startswith("classifier.")}
        miss, unexp = self.classifier.load_state_dict(clf_head_state, strict=True)
        print(f"[Classification head]    missing={miss}  unexpected={unexp}")

        # ── 2. Localizer: regression head only ───────────────────────────
        loc = _load(localizer_path)

        reg_state = {k.replace("regressor.", ""): v
                     for k, v in loc.items() if k.startswith("regressor.")}
        miss, unexp = self.regressor.load_state_dict(reg_state, strict=True)
        print(f"[Localization head]      missing={miss}  unexpected={unexp}")

        # ── 3. UNet: override encoder + load decoder ──────────────────────
        unet = _load(unet_path)

        # Override encoder with UNet's encoder (decoder was trained with it)
        unet_encoder = {k.replace("encoder.", ""): v
                        for k, v in unet.items() if k.startswith("encoder.")}
        if unet_encoder:
            miss, unexp = self.encoder.load_state_dict(unet_encoder, strict=True)
            print(f"[Encoder from UNet]      missing={miss}  unexpected={unexp}")

        # Load each decoder module individually for clear error reporting
        dec_modules = {
            "decoder4":      self.decoder4,
            "decoder3":      self.decoder3,
            "decoder2":      self.decoder2,
            "decoder1":      self.decoder1,
            "final_upsample": self.final_upsample,
            "output_conv":   self.output_conv,
        }
        for prefix, module in dec_modules.items():
            sub = {k[len(prefix) + 1:]: v
                   for k, v in unet.items() if k.startswith(prefix + ".")}
            if sub:
                miss, unexp = module.load_state_dict(sub, strict=True)
                print(f"[{prefix}]  missing={miss}  unexpected={unexp}")
            else:
                print(f"[{prefix}]  WARNING — no weights found in unet.pth!")

        print("Pretrained weights loaded successfully.")

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> dict:
        """Single forward pass → all three task outputs.

        Args:
            x: [B, 3, 224, 224] normalised input images.

        Returns:
            dict with keys:
              'classification' : [B, 37]       logits
              'localization'   : [B, 4]         [cx,cy,w,h] in pixels
              'segmentation'   : [B, 3, 224, 224] logits
        """
        # Shared encoder with skip connections for segmentation
        bottleneck, features = self.encoder(x, return_features=True)

        # Classification branch
        pooled   = self.adaptive_pool(bottleneck)   # (B, 512, 7, 7)
        cls_out  = self.classifier(pooled)           # (B, 37)

        # Localization branch — sigmoid output scaled to pixel space
        bbox_out = self.regressor(pooled) * self.IMAGE_SIZE  # (B, 4)

        # Segmentation branch — U-Net decoder
        d4  = self.decoder4(bottleneck,  features["block4"])  # (B,256,14,14)
        d3  = self.decoder3(d4,          features["block3"])  # (B,128,28,28)
        d2  = self.decoder2(d3,          features["block2"])  # (B, 64,56,56)
        d1  = self.decoder1(d2,          features["block1"])  # (B, 32,112,112)
        seg_out = self.output_conv(self.final_upsample(d1))   # (B,3,224,224)

        return {
            "classification": cls_out,
            "localization":   bbox_out,
            "segmentation":   seg_out,
        }