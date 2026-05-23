# utils/dataset.py
# Handles WHU Building Dataset and SpaceNet loading.
# Also provides a PseudoLabelDataset for Phase 4 mixing.

import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 512):
    """Strong augmentations for training. Remote sensing images benefit from
    heavy geometric augmentation since buildings appear at any orientation."""
    return A.Compose([
        A.RandomResizedCrop(image_size, image_size, scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        # Photometric augmentations for varying lighting / sensors
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.GaussNoise(p=0.2),
        # Normalize to ImageNet mean/std (used by most pretrained encoders)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 512):
    """Minimal transforms for validation/test: just resize and normalize."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# WHU Building Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WHUBuildingDataset(Dataset):
    """
    WHU Building Dataset loader.

    Directory layout expected:
        root/
          train/image/*.tif (or .png)
          train/label/*.tif
          val/image/...
          val/label/...
          test/image/...
          test/label/...

    Masks are binary: 255 = building, 0 = background.
    We convert them to {0, 1} float tensors.
    """

    def __init__(self, img_dir: str, mask_dir: str, transforms=None):
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms

        # Collect all image files (supports .tif, .tiff, .png, .jpg)
        self.img_paths = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        ])

        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        # Mask filename matches image filename (same stem, possibly .png)
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            mask_path = self.mask_dir / (img_path.stem + '.png')

        # Load as RGB numpy arrays
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask  = np.array(Image.open(mask_path).convert("L"),  dtype=np.float32)

        # Binarize: WHU labels use 255 for building pixels
        mask = (mask > 127).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]          # (C, H, W) float tensor
            mask  = augmented["mask"].unsqueeze(0)  # (1, H, W)

        return {"image": image, "mask": mask, "path": str(img_path), "pseudo": False}


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-label Dataset (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

class PseudoLabelDataset(Dataset):
    """
    Loads images together with SAM-generated pseudo-label masks.

    Expected layout (created by phase4/sam_pipeline.py):
        pseudo_label_dir/
          images/
            frame_0001.png
            ...
          masks/
            frame_0001.png   ← binary mask, same stem
    """

    def __init__(self, pseudo_label_dir: str, transforms=None):
        self.img_dir  = Path(pseudo_label_dir) / "images"
        self.mask_dir = Path(pseudo_label_dir) / "masks"
        self.transforms = transforms

        self.img_paths = sorted(self.img_dir.glob("*.png"))
        if len(self.img_paths) == 0:
            raise FileNotFoundError(
                f"No pseudo-labeled images found in {self.img_dir}. "
                "Run phase4/sam_pipeline.py first."
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        mask_path = self.mask_dir / img_path.name

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask  = np.array(Image.open(mask_path).convert("L"),  dtype=np.float32)
        mask  = (mask > 127).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"].unsqueeze(0)

        # Flag these samples so loss weighting or logging can distinguish them
        return {"image": image, "mask": mask, "path": str(img_path), "pseudo": True}


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict, pseudo_label_dir: str = None):
    """
    Build train/val/test DataLoaders from config dict.

    Args:
        cfg:               Full config dictionary (from config.yaml).
        pseudo_label_dir:  If provided, mixes pseudo-labeled data into training
                           set at the ratio specified in cfg['phase4'].
    Returns:
        dict with keys 'train', 'val', 'test'
    """
    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]
    img_size  = data_cfg["image_size"]

    train_ds = WHUBuildingDataset(
        img_dir  = os.path.join(data_cfg["root"], data_cfg["train_img_dir"]),
        mask_dir = os.path.join(data_cfg["root"], data_cfg["train_mask_dir"]),
        transforms = get_train_transforms(img_size),
    )

    val_ds = WHUBuildingDataset(
        img_dir  = os.path.join(data_cfg["root"], data_cfg["val_img_dir"]),
        mask_dir = os.path.join(data_cfg["root"], data_cfg["val_mask_dir"]),
        transforms = get_val_transforms(img_size),
    )

    test_ds = WHUBuildingDataset(
        img_dir  = os.path.join(data_cfg["root"], data_cfg["test_img_dir"]),
        mask_dir = os.path.join(data_cfg["root"], data_cfg["test_mask_dir"]),
        transforms = get_val_transforms(img_size),
    )

    # ── Phase 4: mix pseudo-labels into the training pool ────────────────────
    if pseudo_label_dir is not None:
        pseudo_ds = PseudoLabelDataset(
            pseudo_label_dir = pseudo_label_dir,
            transforms       = get_train_transforms(img_size),
        )
        ratio = cfg["phase4"]["pseudo_label_ratio"]
        n_pseudo = int(len(train_ds) * ratio / (1 - ratio))
        # Subsample pseudo dataset to hit the target ratio
        if n_pseudo < len(pseudo_ds):
            indices  = torch.randperm(len(pseudo_ds))[:n_pseudo].tolist()
            pseudo_ds = torch.utils.data.Subset(pseudo_ds, indices)

        train_ds = ConcatDataset([train_ds, pseudo_ds])
        print(f"[Dataset] Mixed {n_pseudo} pseudo-labeled samples into training "
              f"({len(train_ds)} total).")

    loader_kwargs = dict(
        batch_size  = train_cfg["batch_size"],
        num_workers = data_cfg["num_workers"],
        pin_memory  = True,
    )

    return {
        "train": DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        "val":   DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        "test":  DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    }
