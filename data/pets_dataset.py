"""Dataset skeleton for Oxford-IIIT Pet.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Returns per sample:
        image:      FloatTensor [3, H, W]
        label:      LongTensor  scalar (breed index 0-36)
        bbox:       FloatTensor [4]  (x_center, y_center, width, height) in pixels
        mask:       LongTensor  [H, W] with values {0=background,1=foreground,2=boundary}
    """

    # 37 breed names in alphabetical order (matches annotation files)
    BREEDS = [
        'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
        'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
        'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier',
        'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
        'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
        'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher',
        'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed',
        'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
        'wheaten_terrier', 'yorkshire_terrier'
    ]
    BREED2IDX = {b: i for i, b in enumerate(BREEDS)}

    def __init__(
        self,
        root: str = 'data',
        split: str = 'trainval',
        image_size: int = 224,
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        """
        Args:
            root:       Root directory where dataset is stored / will be downloaded.
            split:      'trainval' or 'test'.
            image_size: Resize images to this square size.
            transform:  Optional additional transforms applied to the image tensor.
            download:   If True, download the dataset if not already present.
        """
        self.root       = Path(root)
        self.split      = split
        self.image_size = image_size
        self.transform  = transform

        if download:
            self._download()

        self.images_dir  = self.root / 'oxford-iiit-pet' / 'images'
        self.annots_dir  = self.root / 'oxford-iiit-pet' / 'annotations'
        self.masks_dir   = self.annots_dir / 'trimaps'
        self.xmls_dir    = self.annots_dir / 'xmls'

        # Read split file
        split_file = self.annots_dir / f'{split}.txt'
        self.samples = []
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    name = parts[0]           # e.g. 'Abyssinian_1'
                    self.samples.append(name)

        # Base image transforms (always applied)
        self.img_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.mask_resize = T.Resize(
            (image_size, image_size),
            interpolation=T.InterpolationMode.NEAREST
        )

    def _download(self):
        """Download the Oxford-IIIT Pet dataset using torchvision."""
        try:
            from torchvision.datasets import OxfordIIITPet
            OxfordIIITPet(root=str(self.root), split=self.split,
                          target_types=['category', 'segmentation'],
                          download=True)
            print("Dataset downloaded successfully.")
        except Exception as e:
            print(f"Download failed: {e}. Please download manually.")

    def _parse_bbox(self, name: str, orig_w: int, orig_h: int) -> torch.Tensor:
        """Parse bounding box from XML annotation.

        Returns [x_center, y_center, width, height] scaled to image_size.
        """
        xml_path = self.xmls_dir / f'{name}.xml'
        if not xml_path.exists():
            # Return full image box if annotation missing
            s = float(self.image_size)
            return torch.tensor([s/2, s/2, s, s], dtype=torch.float32)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('.//bndbox')

        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # Scale to resized image coordinates
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        xmin *= scale_x;  xmax *= scale_x
        ymin *= scale_y;  ymax *= scale_y

        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width    = xmax - xmin
        height   = ymax - ymin

        return torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

    def _get_label(self, name: str) -> int:
        """Extract breed label from filename."""
        # Filename format: BreedName_N.jpg — breed is everything before last underscore+number
        breed = '_'.join(name.split('_')[:-1])
        return self.BREED2IDX.get(breed, 0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name = self.samples[idx]

        # ── Load image ────────────────────────────────────────────────────
        img_path = self.images_dir / f'{name}.jpg'
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # ── Load mask (trimap) ────────────────────────────────────────────
        mask_path = self.masks_dir / f'{name}.png'
        mask = Image.open(mask_path)
        mask = self.mask_resize(mask)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long) - 1  # 1,2,3 → 0,1,2

        # ── Parse bbox ────────────────────────────────────────────────────
        bbox = self._parse_bbox(name, orig_w, orig_h)

        # ── Transform image ───────────────────────────────────────────────
        image = self.img_transform(image)
        if self.transform:
            image = self.transform(image)

        # ── Label ─────────────────────────────────────────────────────────
        label = self._get_label(name)

        return {
            'image': image,                                    # [3, H, W]
            'label': torch.tensor(label, dtype=torch.long),   # scalar
            'bbox':  bbox,                                     # [4]
            'mask':  mask,                                     # [H, W]
            'name':  name,
        }
