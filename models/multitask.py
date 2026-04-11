import torch
import torch.nn as nn
import gdown

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3,
                 classifier_path: str = "classifier.pth",
                 localizer_path: str = "localizer.pth",
                 unet_path: str = "unet.pth"):

        super(MultiTaskPerceptionModel, self).__init__()

        # ── Download pretrained weights (REQUIRED) ─────────────────────
        gdown.download(id="<classifier.pth drive id>", output=classifier_path, quiet=False)
        gdown.download(id="<localizer.pth drive id>", output=localizer_path, quiet=False)
        gdown.download(id="<unet.pth drive id>", output=unet_path, quiet=False)

        self.loc_scale = 224.0

        # ✅ Shared encoder
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ── Classification head ────────────────────────────────────────
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            CustomDropout(p=0.6),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            CustomDropout(p=0.6),
            nn.Linear(4096, num_breeds),
        )

        # ── Localization head ─────────────────────────────────────────
        self.localizer_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            CustomDropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            CustomDropout(p=0.3),
            nn.Linear(512, 4),
            nn.Sigmoid(),
        )

        # ── Segmentation decoder ──────────────────────────────────────
        self.bottleneck_drop = CustomDropout(p=0.6)

        self.up1 = UpBlock(512, 512, 512)
        self.up2 = UpBlock(512, 512, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up4 = UpBlock(128, 128, 64)
        self.up5 = UpBlock(64, 64, 64)

        self.final = nn.Conv2d(64, seg_classes, kernel_size=1)

        # ✅ Load pretrained weights
        self._load_pretrained_weights(classifier_path, localizer_path, unet_path)

    # ────────────────────────────────────────────────────────────────
    def _load_pretrained_weights(self, classifier_path, localizer_path, unet_path):

        # Load weights
        classifier_state = torch.load(classifier_path, map_location="cpu")
        localizer_state = torch.load(localizer_path, map_location="cpu")
        unet_state = torch.load(unet_path, map_location="cpu")

        # ✅ Load encoder (strict match required)
        encoder_state = classifier_state["encoder"]
        self.encoder.load_state_dict(encoder_state, strict=True)

        # Load heads
        self.classifier_head.load_state_dict(classifier_state["classifier_head"], strict=False)
        self.localizer_head.load_state_dict(localizer_state["localizer_head"], strict=False)

        # Load segmentation decoder
        self.load_state_dict(unet_state, strict=False)

    # ────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):

        # Encoder
        features, skips = self.encoder(x)

        # ── Classification ───────────────────────────────────────────
        cls_out = self.classifier_head(features)

        # ── Localization ─────────────────────────────────────────────
        loc_out = self.localizer_head(features) * self.loc_scale

        # ── Segmentation ────────────────────────────────────────────
        x = self.bottleneck_drop(features)

        x = self.up1(x, skips[4])
        x = self.up2(x, skips[3])
        x = self.up3(x, skips[2])
        x = self.up4(x, skips[1])
        x = self.up5(x, skips[0])

        seg_out = self.final(x)

        # ✅ IMPORTANT: return dict (as per skeleton)
        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out
        }