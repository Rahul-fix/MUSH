# run it with command
# PYTHONPATH=. python scripts/save_problematic_samples.py
import os
import numpy as np
from PIL import Image
import torch
from src.data.coco_dataset import COCODataset, ImageSegmentationDataset
from src.utils.palette import label2id

# Indices to save (from error analysis)
problem_indices = [108, 66, 31]

# Paths (update as needed)
coco_file = "CKA_sweet_pepper_2020_summer.json"
dataset_root = "/scratch/s7rakuma/datasets/"
output_dir = "Output/problematic_samples"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
base_ds = COCODataset(coco_file, root_dir=dataset_root)
remap_ds = ImageSegmentationDataset(base_ds, label2id=label2id)

from src.utils.palette import palette


# For visualization, set background (class 0) to white for better contrast
def colorize_mask(mask, palette):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(palette):
        color_mask[mask == class_idx] = color
    # Set background (class 0) to white for visibility
    color_mask[mask == 0] = (255, 255, 255)
    return color_mask

for idx in problem_indices:
    img_tensor, mask_tensor, orig_img_np, orig_mask_np = remap_ds[idx]
    # Print unique values and counts for original and remapped masks
    uniq_orig, counts_orig = np.unique(orig_mask_np, return_counts=True)
    uniq_remap, counts_remap = np.unique(mask_tensor.numpy(), return_counts=True)
    print(f"idx={idx} [ORIG MASK] unique: {dict(zip(uniq_orig, counts_orig))}")
    print(f"idx={idx} [REMAP MASK] unique: {dict(zip(uniq_remap, counts_remap))}")
    # Save original image
    img_pil = Image.fromarray(orig_img_np)
    img_pil.save(os.path.join(output_dir, f"img_{idx}.png"))
    # Save original mask (before remap, grayscale)
    mask_orig_pil = Image.fromarray(orig_mask_np.astype(np.uint8))
    mask_orig_pil.save(os.path.join(output_dir, f"mask_orig_{idx}.png"))
    # Save remapped mask (grayscale)
    mask_remap_np = mask_tensor.numpy().astype(np.uint8)
    mask_remap_pil = Image.fromarray(mask_remap_np)
    mask_remap_pil.save(os.path.join(output_dir, f"mask_remap_{idx}.png"))
    # Save colorized remapped mask (background as white)
    color_mask = colorize_mask(mask_remap_np, palette)
    color_mask_pil = Image.fromarray(color_mask)
    color_mask_pil.save(os.path.join(output_dir, f"mask_remap_color_{idx}.png"))
    # Overlay: blend original image and color mask (alpha=0.5)
    if orig_img_np.shape[2] == 3:
        orig_img_float = orig_img_np.astype(np.float32) / 255.0
        color_mask_float = color_mask.astype(np.float32) / 255.0
        overlay = (0.5 * orig_img_float + 0.5 * color_mask_float)
        overlay = (overlay * 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay)
        overlay_pil.save(os.path.join(output_dir, f"overlay_{idx}.png"))
    print(f"Saved image and masks for idx={idx}")
