# phase4_sam_pseudolabel/sam_pipeline.py
# SAM (Segment Anything Model) zero-shot pseudo-labeling pipeline.
#
# Strategy:
#   1. Run SAM in automatic mask generation mode on unlabeled satellite frames.
#   2. Score each mask by SAM's own predicted IoU (internal quality estimate).
#   3. Keep masks above a confidence threshold → pseudo-labels.
#   4. Save image crops + binary masks to disk for Phase 4 dataset mixing.
#
# This is the core of the solo expansion strategy: we treat SAM as a
# free annotation oracle and use its output to artificially grow our dataset.
# The key question is: at what threshold does pseudo-label noise hurt vs help?

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Optional

# SAM from the official segment-anything package
# pip install git+https://github.com/facebookresearch/segment-anything.git
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("[WARNING] segment-anything not installed. "
          "Run: pip install git+https://github.com/facebookresearch/segment-anything.git")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SAMPipelineConfig:
    # Model
    model_type: str   = "vit_h"         # vit_h, vit_l, vit_b
    checkpoint:  str  = "./weights/sam_vit_h.pth"
    device:      str  = "cuda"

    # Quality filters
    pred_iou_threshold: float  = 0.88   # SAM's internal IoU score (0–1)
    stability_threshold: float = 0.95   # Mask stability across thresholds
    min_mask_area: int         = 500    # Minimum pixel area to keep (noise filter)
    max_mask_area_ratio: float = 0.4    # Max fraction of image to be one mask

    # Automatic mask generator settings
    points_per_side:    int   = 32      # Dense grid of prompt points
    points_per_batch:   int   = 64      # Points processed per GPU batch
    crop_n_layers:      int   = 1       # Multi-scale cropping (0 = single scale)
    crop_overlap_ratio: float = 0.5

    # I/O
    unlabeled_dir:    str = "./data/unlabeled"
    output_dir:       str = "./data/pseudo_labels"
    image_extensions: tuple = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-label generator
# ─────────────────────────────────────────────────────────────────────────────

class SAMPseudoLabelGenerator:
    """
    Generates pseudo-labels for unlabeled satellite imagery using SAM.

    Workflow per image:
        1. Load and optionally tile the image (SAM works best ≤1024px).
        2. Run SamAutomaticMaskGenerator → list of mask dicts.
        3. Filter by pred_iou_score, stability_score, and size.
        4. Merge surviving masks into a single binary map.
        5. Save (image, binary_mask) pair to the output directory.
    """

    def __init__(self, config: SAMPipelineConfig):
        self.cfg = config

        if not SAM_AVAILABLE:
            raise RuntimeError("segment-anything package is not installed.")

        print(f"[SAM] Loading {config.model_type} from {config.checkpoint}")
        sam = sam_model_registry[config.model_type](checkpoint=config.checkpoint)
        sam.to(config.device)
        sam.eval()

        self.mask_generator = SamAutomaticMaskGenerator(
            model               = sam,
            points_per_side     = config.points_per_side,
            points_per_batch    = config.points_per_batch,
            pred_iou_thresh     = config.pred_iou_threshold,
            stability_score_thresh = config.stability_threshold,
            crop_n_layers       = config.crop_n_layers,
            crop_overlap_ratio  = config.crop_overlap_ratio,
            # Return compressed masks to save memory on large images
            output_mode         = "binary_mask",
        )

    def _filter_masks(self, masks: list, img_h: int, img_w: int) -> list:
        """
        Apply post-hoc quality filters on top of SAM's own thresholds.

        Each mask dict from SAM contains:
            segmentation:       (H, W) bool array
            area:               pixel count
            pred_iou_score:     SAM's estimated mask quality [0, 1]
            stability_score:    consistency score [0, 1]
            bbox:               [x, y, w, h]
        """
        img_area = img_h * img_w
        filtered = []

        for m in masks:
            area = m["area"]

            # Reject masks that are too small (likely noise / texture artifacts)
            if area < self.cfg.min_mask_area:
                continue

            # Reject masks that cover most of the image (likely background)
            if area / img_area > self.cfg.max_mask_area_ratio:
                continue

            filtered.append(m)

        return filtered

    def _merge_masks_to_binary(self, masks: list, img_h: int, img_w: int) -> np.ndarray:
        """
        Combine all accepted masks into one binary map.
        Pixels in any mask → 1 (foreground), rest → 0 (background).
        """
        combined = np.zeros((img_h, img_w), dtype=np.uint8)
        for m in masks:
            combined[m["segmentation"]] = 1
        return combined

    def _tile_image(self, image: np.ndarray, tile_size: int = 1024, overlap: int = 64):
        """
        Split large images into overlapping tiles for SAM.
        SAM's attention map is fixed at 1024×1024 internally.

        Returns: list of (tile, y_start, x_start) tuples
        """
        H, W = image.shape[:2]
        tiles = []
        y = 0
        while y < H:
            x = 0
            while x < W:
                y2 = min(y + tile_size, H)
                x2 = min(x + tile_size, W)
                tiles.append((image[y:y2, x:x2], y, x))
                x += tile_size - overlap
                if x2 == W:
                    break
            y += tile_size - overlap
            if y + overlap >= H:
                break
        return tiles

    def generate_pseudo_label(self, image_path: str) -> Optional[np.ndarray]:
        """
        Generate a pseudo-label binary mask for one image.

        Returns the binary mask array (H, W) with values {0, 1},
        or None if no valid masks were found.
        """
        image = np.array(Image.open(image_path).convert("RGB"))
        H, W  = image.shape[:2]

        # For large images, tile and re-compose
        if max(H, W) > 1024:
            full_mask = np.zeros((H, W), dtype=np.uint8)
            tiles     = self._tile_image(image)

            for tile, y0, x0 in tiles:
                raw_masks = self.mask_generator.generate(tile)
                h_t, w_t  = tile.shape[:2]
                filtered  = self._filter_masks(raw_masks, h_t, w_t)
                tile_mask = self._merge_masks_to_binary(filtered, h_t, w_t)
                full_mask[y0:y0 + h_t, x0:x0 + w_t] |= tile_mask

            binary_mask = full_mask

        else:
            raw_masks   = self.mask_generator.generate(image)
            filtered    = self._filter_masks(raw_masks, H, W)

            if not filtered:
                return None

            binary_mask = self._merge_masks_to_binary(filtered, H, W)

        # Optional: morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1
        ).astype(np.uint8)

        return binary_mask

    def process_dataset(self, unlabeled_dir: str = None, output_dir: str = None):
        """
        Process all unlabeled images and save pseudo-label pairs.

        Output structure:
            output_dir/
              images/   ← copies of original images
              masks/    ← binary pseudo-label masks (255 = foreground)
        """
        unlabeled_dir = Path(unlabeled_dir or self.cfg.unlabeled_dir)
        output_dir    = Path(output_dir    or self.cfg.output_dir)

        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "masks").mkdir(parents=True, exist_ok=True)

        image_paths = [
            p for p in sorted(unlabeled_dir.iterdir())
            if p.suffix.lower() in self.cfg.image_extensions
        ]

        if not image_paths:
            raise FileNotFoundError(f"No images found in {unlabeled_dir}")

        print(f"[SAM] Processing {len(image_paths)} unlabeled images...")

        saved = 0
        skipped = 0

        for img_path in tqdm(image_paths, desc="Pseudo-labeling"):
            try:
                binary_mask = self.generate_pseudo_label(str(img_path))

                if binary_mask is None:
                    skipped += 1
                    continue

                stem = img_path.stem
                # Save original image
                img_out = output_dir / "images" / f"{stem}.png"
                Image.open(img_path).convert("RGB").save(img_out)

                # Save mask (0/255 for compatibility with standard loaders)
                mask_out = output_dir / "masks" / f"{stem}.png"
                Image.fromarray(binary_mask * 255).save(mask_out)
                saved += 1

            except Exception as e:
                print(f"[SAM] Error on {img_path.name}: {e}")
                skipped += 1

        print(f"\n[SAM] Done. Saved: {saved} | Skipped: {skipped}")
        print(f"[SAM] Pseudo-labels written to: {output_dir}")
        return saved


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def threshold_sensitivity_report(
    model,
    val_loader,
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
    device: str = "cuda",
) -> dict:
    """
    Sweep segmentation decision thresholds on the validation set to find
    the optimal operating point.

    Returns dict mapping threshold → {iou, f1, precision, recall}.
    """
    from utils.metrics import SegmentationMetrics

    results = {}
    model.eval()
    model.to(device)

    for thresh in thresholds:
        metrics = SegmentationMetrics(threshold=thresh)
        with torch.no_grad():
            for batch in val_loader:
                images  = batch["image"].to(device)
                masks   = batch["mask"].to(device)
                logits  = model(images)
                metrics.update(logits, masks)
        results[thresh] = metrics.compute()
        print(f"  thresh={thresh:.2f} → {results[thresh]}")

    best_thresh = max(results, key=lambda t: results[t]["iou"])
    print(f"\nBest threshold: {best_thresh} (IoU={results[best_thresh]['iou']:.4f})")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-label quality diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pseudo_label_quality(
    pseudo_label_dir: str,
    gt_mask_dir: str,
    max_samples: int = 200,
) -> dict:
    """
    If a small labeled holdout exists, measure how accurate SAM's pseudo-labels
    are against ground truth. Useful for calibrating your confidence threshold.

    Args:
        pseudo_label_dir: Directory with masks/ subfolder from SAM pipeline.
        gt_mask_dir:      Ground truth masks for the same image set.
        max_samples:      Limit evaluation to this many images.

    Returns:
        dict with mean IoU, mean Dice, coverage (% images with any mask).
    """
    pseudo_dir = Path(pseudo_label_dir) / "masks"
    gt_dir     = Path(gt_mask_dir)

    iou_scores  = []
    dice_scores = []
    covered     = 0

    mask_paths = sorted(pseudo_dir.glob("*.png"))[:max_samples]

    for mask_path in mask_paths:
        gt_path = gt_dir / mask_path.name
        if not gt_path.exists():
            continue

        pred = (np.array(Image.open(mask_path).convert("L")) > 127).astype(float)
        gt   = (np.array(Image.open(gt_path).convert("L"))   > 127).astype(float)

        if pred.sum() > 0:
            covered += 1

        # IoU
        intersection = (pred * gt).sum()
        union        = pred.sum() + gt.sum() - intersection
        iou  = (intersection + 1e-6) / (union + 1e-6)
        dice = (2 * intersection + 1e-6) / (pred.sum() + gt.sum() + 1e-6)

        iou_scores.append(iou)
        dice_scores.append(dice)

    results = {
        "mean_iou":  float(np.mean(iou_scores)),
        "mean_dice": float(np.mean(dice_scores)),
        "coverage":  covered / len(mask_paths) if mask_paths else 0.0,
        "n_samples": len(iou_scores),
    }
    print(f"[Pseudo-label QA] Mean IoU: {results['mean_iou']:.3f} | "
          f"Mean Dice: {results['mean_dice']:.3f} | "
          f"Coverage: {results['coverage']:.1%}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM pseudo-label generation")
    parser.add_argument("--unlabeled_dir",  default="./data/unlabeled")
    parser.add_argument("--output_dir",     default="./data/pseudo_labels")
    parser.add_argument("--sam_checkpoint", default="./weights/sam_vit_h.pth")
    parser.add_argument("--model_type",     default="vit_h")
    parser.add_argument("--iou_thresh",     type=float, default=0.88)
    parser.add_argument("--stability",      type=float, default=0.95)
    parser.add_argument("--device",         default="cuda")
    args = parser.parse_args()

    cfg = SAMPipelineConfig(
        model_type          = args.model_type,
        checkpoint          = args.sam_checkpoint,
        device              = args.device,
        pred_iou_threshold  = args.iou_thresh,
        stability_threshold = args.stability,
        unlabeled_dir       = args.unlabeled_dir,
        output_dir          = args.output_dir,
    )

    generator = SAMPseudoLabelGenerator(cfg)
    generator.process_dataset()
