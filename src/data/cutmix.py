import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# Colors that match your palette.py                 (B, G, R)
PALETTE = {
    0: (0,   0,   0),      # bg  – black
    1: (0,   0, 255),      # kp  – blue
    2: (199, 33,  28),     # red – red
    3: (255, 247, 0),      # yellow
    4: (0, 255,   0),      # green
    5: (225, 0, 255),      # mixed – magenta
    6: (255,102,  0),      # mixed_red – orange
    7: (209,196, 21),      # mixed_yellow – olive
}

# Your dataset normalisation stats (train_transform)
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255.0
ADE_STD  = np.array([58.395,  57.120,  57.375]) / 255.0


class SegmentationCutMix:
    """
    CutMix that copes with variable-length mask stacks (Mask2Former)
    and saves ONE visual example (original vs CutMix) for confirmation.
    """
    def __init__(self,
                 num_classes: int,
                 alpha: float = 1.0,
                 prob: float = 0.5,
                 save_samples: bool = False,
                 out_dir: str = "Output/cutmix"):
        self.num_classes   = num_classes
        self.alpha         = alpha
        self.prob          = prob
        self.save_samples  = save_samples
        self.out_dir       = Path(out_dir)
        self._already_saved = False

        if self.save_samples:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    # Public entry point
    def __call__(self, imgs, masks):
        """
        imgs  : Tensor (B,3,H,W)   normalised
        masks : Tensor (B, H, W) or list of (B, H, W) (should be class indices)
        """
        B = imgs.size(0)
        if B < 2 or np.random.rand() > self.prob:
            return imgs, masks                           # nothing to do

        # Convert one-hot or multi-channel masks to class index maps if needed
        if isinstance(masks, list):
            # If each mask is (N, H, W) and N > 1, assume one-hot and convert
            if all(m.dim() == 3 for m in masks):
                masks = torch.stack([torch.argmax(m, dim=0) if m.shape[0] > 1 else m.squeeze(0) for m in masks], dim=0)
            else:
                masks = torch.stack([m for m in masks], dim=0)
        elif isinstance(masks, torch.Tensor) and masks.dim() == 4:
            # (B, N, H, W) -> (B, H, W) via argmax
            masks = torch.argmax(masks, dim=1)



        # Save ONE batch for visual sanity-check
        if self.save_samples and not self._already_saved:
            self._save_batch(imgs, masks, apply_cutmix=True)
            self._already_saved = True

        return self._cutmix(imgs, masks)

    # Core CutMix
    def _cutmix(self, imgs, masks):
        B, _, H, W = imgs.shape
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.
        rw, rh = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1, y1 = np.clip(cx - rw // 2, 0, W), np.clip(cy - rh // 2, 0, H)
        x2, y2 = np.clip(cx + rw // 2, 0, W), np.clip(cy + rh // 2, 0, H)

        mix_imgs  = imgs.clone()
        # ------------------------------------------------------------------
        # Image part
        # ------------------------------------------------------------------
        for i in range(B):
            j = (i + 1) % B
            mix_imgs[i,:, y1:y2, x1:x2] = imgs[j,:, y1:y2, x1:x2]

        # ------------------------------------------------------------------
        # Mask part – always (B, H, W) with class indices
        # ------------------------------------------------------------------
        mix_masks = masks.clone()
        for i in range(B):
            j = (i + 1) % B
            mix_masks[i, y1:y2, x1:x2] = masks[j, y1:y2, x1:x2]



        return mix_imgs, mix_masks

    # Visualisation helpers – save *one* batch before/after CutMix
    def _save_batch(self, imgs, masks, apply_cutmix=True, max_vis: int = 4):
        """
        Saves <sampleN>_original_*   and   <sampleN>_cutmix_*  triplets.
        """
        if apply_cutmix:
            mix_imgs, mix_masks = self._cutmix(imgs, masks)
        else:
            mix_imgs, mix_masks = imgs, masks

        for idx in range(min(imgs.size(0), max_vis)):
            name = f"sample{idx}"
            self._save_one(imgs[idx], masks[idx] if isinstance(masks, list) else masks[idx],
                           f"{name}_original")
            self._save_one(mix_imgs[idx], mix_masks[idx] if isinstance(mix_masks, list) else mix_masks[idx],
                           f"{name}_cutmix")
        print(f"[CutMix] saved one-batch visualisation → {self.out_dir}")


    def _save_one(self, img_t, mask_t, stem):
        """Save RGB, grayscale mask, colorized mask, and overlay using the same logic as save_problematic_samples.py"""
        # Denormalize image
        img = img_t.cpu().clone()
        mean = torch.tensor(ADE_MEAN).view(3,1,1)
        std  = torch.tensor(ADE_STD ).view(3,1,1)
        img  = torch.clamp(img * std + mean, 0, 1)
        img_np = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # Get single-channel mask
        from src.utils.palette import remap_labels, label2id, palette
        if mask_t.dim() == 3:
            mask_np = mask_t[0].cpu().numpy()
        else:
            mask_np = mask_t.cpu().numpy()
        mask_remapped = remap_labels(mask_np, label2id).numpy().astype(np.uint8)
        # If remapped mask is all zeros but original mask had nonzero values, restore those for visualization
        if np.all(mask_remapped == 0) and np.any(mask_np != 0):
            mask_remapped = mask_np.astype(np.uint8)

        # Save grayscale mask
        Image.fromarray(mask_remapped).save(self.out_dir / f"{stem}_mask.png")

        # Colorize mask (background as white)
        palette_arr = np.array(palette)
        def colorize_mask(mask, palette):
            color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for class_idx, color in enumerate(palette):
                color_mask[mask == class_idx] = color
            color_mask[mask == 0] = (255, 255, 255)
            return color_mask
        color_mask = colorize_mask(mask_remapped, palette_arr)
        Image.fromarray(color_mask).save(self.out_dir / f"{stem}_mask_color.png")

        # Overlay: blend original image and color mask (alpha=0.5)
        if img_np.shape[2] == 3:
            orig_img_float = img_np.astype(np.float32) / 255.0
            color_mask_float = color_mask.astype(np.float32) / 255.0
            overlay = (0.5 * orig_img_float + 0.5 * color_mask_float)
            overlay = (overlay * 255).astype(np.uint8)
            overlay_pil = Image.fromarray(overlay)
            overlay_pil.save(self.out_dir / f"{stem}_overlay.png")
        # Save original image
        img_pil.save(self.out_dir / f"{stem}_image.png")


# --------------------------------------------------------------------------
# Factory function used in train.py
# --------------------------------------------------------------------------
def get_cutmix_transform(num_classes, alpha=1.0, prob=0.5, save_samples=False):
    return SegmentationCutMix(num_classes, alpha, prob, save_samples)
