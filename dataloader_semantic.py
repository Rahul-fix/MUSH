# ============================================================
# Cleaned and Commented Code for COCO-style Semantic Segmentation
# ============================================================

# ---------------------------------------
# 1. Consolidated Imports
# ---------------------------------------
import os
import random
import json
import pickle

from PIL import Image
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from scipy.special import softmax
from tqdm.auto import tqdm

from torchvision import transforms

import evaluate

from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    pipeline
)

# ---------------------------------------
# 2. Set Random Seeds for Reproducibility
# ---------------------------------------
seed = 78
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ---------------------------------------
# 3. COCO Dataset Class Definition
# ---------------------------------------
class COCODataset(Dataset):
    """
    A custom Dataset class for COCO-format JSON annotations and images.
    Each item returns:
      - 'image': a PIL Image (converted to RGB)
      - 'semantic_map': a 2D uint8 tensor where each pixel's value is the category ID
      - 'image_id': the original COCO image ID
      - 'width', 'height': dimensions of the image
    """
    def __init__(self, coco_file: str, root_dir: str, split: str = None, transform=None):
        """
        Args:
            coco_file (str): Path to the COCO-style JSON file.
            root_dir (str): Root directory where images are stored.
            split (str, optional): One of 'train', 'valid', 'test'. Determines which images to keep.
            transform (callable, optional): A torchvision-style transform to apply to the image.
        """
        # Load the COCO JSON data
        with open(coco_file, 'r') as f:
            self.coco_data = json.load(f)

        # Manually defined split indices (image IDs) for train/valid/test
        # Adjust these lists if your dataset uses a different splitting scheme.
        self.split_image_ids = {
            'train': list(range(283, 314)) + list(range(314, 345)) + list(range(408, 471)),
            'valid': list(range(345, 377)) + list(range(533, 564)),
            'test':  list(range(377, 408)) + list(range(471, 533))
        }

        # Load the 'images' array from the JSON and filter by split if provided
        all_images = self.coco_data['images']
        if split in self.split_image_ids:
            valid_ids = set(self.split_image_ids[split])
            self.images = [img for img in all_images if img['id'] in valid_ids]
        else:
            self.images = all_images

        # Load annotations and categories from JSON
        self.annotations = self.coco_data['annotations']
        # Build a dict mapping category_id -> {name, color, supercategory}
        self.categories = {
            cat['id']: {
                'name': cat['name'],
                'color': cat.get('color', "#000000"),
                'supercategory': cat['supercategory']
            }
            for cat in self.coco_data['categories']
        }

        # Print category IDs and names for reference/debugging
        print("Category IDs and their names:")
        for cat_id, cat_info in self.categories.items():
            print(f"  ID {cat_id}: {cat_info['name']}")

        self.root_dir = root_dir
        self.transform = transform

        # Build a lookup from image_id -> list of annotation dicts
        self.image_id_to_annotations = {}
        for anno in self.annotations:
            image_id = anno['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(anno)

    def __len__(self):
        """Return the number of images in this split."""
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): Index of the image in self.images.

        Returns:
            dict with keys:
              - 'image': transformed PIL Image or tensor
              - 'semantic_map': Tensor of shape (H, W) with category IDs
              - 'image_id': original COCO image ID
              - 'width', 'height': image dimensions
        """
        # Fetch image metadata
        image_info = self.images[idx]
        image_id = image_info['id']
        width, height = image_info['width'], image_info['height']

        # Construct the path to the image file
        # The JSON file stores paths like "/datasets/...", so remove the prefix
        relative_path = image_info['path'].lstrip('/datasets/')
        image_path = os.path.join(self.root_dir, relative_path)

        # Load the image and convert to RGB
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Retrieve all annotations for this image_id
        annotations = self.image_id_to_annotations.get(image_id, [])
        segmentations = [anno.get('segmentation', []) for anno in annotations]
        category_ids = [anno['category_id'] for anno in annotations]

        # Create an empty semantic map (H x W), initialized to 0 (background).
        # Each pixel will be overwritten with its category ID.
        semantic_map = np.zeros((height, width), dtype=np.uint8)

        # For each annotation, draw the polygons onto the semantic_map
        for seg, cat_id in zip(segmentations, category_ids):
            for poly in seg:
                # Convert the flat list [x1, y1, x2, y2, ...] to Nx2 array
                coords = np.array(poly).reshape(-1, 2)
                rr, cc = skimage.draw.polygon(coords[:, 1], coords[:, 0], semantic_map.shape)
                semantic_map[rr, cc] = cat_id

        semantic_map_tensor = torch.tensor(semantic_map, dtype=torch.uint8)

        return {
            'image': image,
            'semantic_map': semantic_map_tensor,
            'image_id': image_id,
            'width': width,
            'height': height
        }

# ---------------------------------------
# 4. Custom Collate Function for DataLoader
# ---------------------------------------
def custom_collate_fn(batch):
    """
    A collate function that stacks tensors where possible and collects
    other elements (like IDs, original images) into lists.

    Input `batch` is a list of dicts (each dict from __getitem__ of COCODataset).
    This function returns a single dict where:
      - 'images': stacked tensor (batch_size, C, H, W)
      - 'semantic_map': stacked tensor (batch_size, H, W)
      - other keys (e.g. 'image_id', 'width', 'height'): lists of values
    """
    collated = {}
    # Inspect keys of the first sample to know what to expect
    for key in batch[0]:
        if key == 'image':
            # Stack all image tensors into a single (B, C, H, W) tensor
            collated['images'] = torch.stack([item['image'] for item in batch])
        elif key == 'semantic_map':
            # Stack all semantic maps into (B, H, W)
            collated['semantic_map'] = torch.stack([item['semantic_map'] for item in batch])
        else:
            # For everything else (image_id, width, height), store as a list
            collated[key] = [item[key] for item in batch]
    return collated

# ---------------------------------------
# 5. ID-to-Label and Color Palette Setup
# ---------------------------------------
# Original mapping from COCO category IDs to human-readable labels
# Here, we only keep the pepper-related IDs and remap them to contiguous IDs
id2label = {
    0: "bg",             # background
    11: "pepper_kp",
    12: "pepper_red",
    13: "pepper_yellow",
    14: "pepper_green",
    15: "pepper_mixed",
    17: "pepper_mixed_red",
    18: "pepper_mixed_yellow"
}

# Build a label2id mapping from old (COCO) IDs to new contiguous IDs (0..7)
# sorted(id2label.keys()) = [0, 11, 12, 13, 14, 15, 17, 18]
label2id = {old_id: new_id for new_id, old_id in enumerate(sorted(id2label.keys()))}

# Build the reverse mapping: new ID -> label name
id2label_remapped = {new_id: id2label[old_id] for old_id, new_id in label2id.items()}

print("Remapped ID-to-label:", id2label_remapped)

# Define a color palette for visualization (BGR hex values)
id2color = {
    0: "#000000",  # black for background
    1: "#0000ff",  # blue for pepper_kp
    2: "#c7211c",  # red for pepper_red
    3: "#fff700",  # yellow for pepper_yellow
    4: "#00ff00",  # green for pepper_green
    5: "#e100ff",  # purple for pepper_mixed
    6: "#ff6600",  # orange for pepper_mixed_red
    7: "#d1c415",  # gold for pepper_mixed_yellow
}

# Convert hex colors to RGB tuples and store in a NumPy array of shape (num_classes, 3)
palette = []
for class_id in range(len(id2label_remapped)):
    hex_color = id2color.get(class_id, "#000000")
    rgb = tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    palette.append(rgb)
palette = np.array(palette, dtype=np.uint8)
print("Color palette (RGB):\n", palette)

# ---------------------------------------
# 6. Utility Function to Remap Mask Labels
# ---------------------------------------
def remap_labels(mask: np.ndarray, label2id_map: dict) -> torch.Tensor:
    """
    Given a 2D ndarray of original COCO category IDs, convert it to a 2D tensor
    of contiguous IDs according to `label2id_map`.

    Args:
        mask (np.ndarray): 2D array of shape (H, W) with values in the set of original IDs.
        label2id_map (dict): Mapping from original_id -> new_id (0..num_classes-1).

    Returns:
        remapped_mask (torch.Tensor): 2D (H, W) tensor with new IDs.
    """
    # Convert numpy array to torch tensor if not already
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.int64)

    # Initialize an output mask of the same shape, filled with zeros
    remapped_mask = torch.zeros_like(mask)

    # For each (old_id, new_id) pair, set those pixels
    for old_id, new_id in label2id_map.items():
        remapped_mask[mask == old_id] = new_id

    return remapped_mask

# ---------------------------------------
# 7. Wrapper Dataset: ImageSegmentationDataset
# ---------------------------------------
class ImageSegmentationDataset(Dataset):
    """
    A wrapper around a base dataset (e.g. COCODataset) to:
      - Remap original class IDs in the mask to contiguous IDs
      - Apply image transforms (normalization, augmentation) to the input image
      - Optionally apply target transforms to the segmentation mask
    Returns a tuple: (image_tensor, remapped_mask_tensor, original_image_numpy, original_mask_numpy)
    """
    def __init__(self, base_dataset: Dataset, transform=None, target_transform=None):
        """
        Args:
            base_dataset (Dataset): An instance of COCODataset.
            transform (callable, optional): Transform applied to PIL image.
            target_transform (callable, optional): Transform applied to segmentation map.
        """
        self.dataset = base_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # Fetch raw sample dictionary from base_dataset
        sample = self.dataset[idx]
        orig_pil_image = sample['image']  # PIL Image
        orig_mask_np = np.array(sample['semantic_map'])  # NumPy array (H, W)

        # Remap original COCO IDs to contiguous IDs
        remapped_mask = remap_labels(orig_mask_np, label2id)

        # Apply transforms to the image: Convert to tensor, Normalize, etc.
        if self.transform:
            image_tensor = self.transform(orig_pil_image)
        else:
            # If no transform is provided, convert to tensor (C, H, W)
            image_tensor = torch.tensor(np.array(orig_pil_image), dtype=torch.float32).permute(2, 0, 1)

        # Apply transforms to the mask if provided (e.g. random flip)
        if self.target_transform:
            mask_transformed = self.target_transform(Image.fromarray(remapped_mask.numpy()))
            # Convert back to a tensor of dtype int64 (required for segmentation loss)
            mask_tensor = torch.tensor(np.array(mask_transformed), dtype=torch.int64)
        else:
            mask_tensor = remapped_mask

        return image_tensor, mask_tensor, np.array(orig_pil_image), orig_mask_np

# ---------------------------------------
# 8. Define Image and Target Transforms
# ---------------------------------------
# ADE dataset normalization constants (divided by 255 to get [0,1] range)
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255.0
ADE_STD  = np.array([58.395,  57.120,  57.375]) / 255.0

# Training-time image transforms: convert to tensor and normalize
train_transform = transforms.Compose([
    # (Optional) Resize, random crop, or random flip can be added here
    transforms.ToTensor(),
    transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

# For this example, no geometric augmentation is done on the mask (but you could add RandomFlip)
target_transform = transforms.Compose([
    # e.g. transforms.RandomHorizontalFlip(p=1.0)
])

# Testing-time transforms: only ToTensor + Normalize
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

# ---------------------------------------
# 9. Instantiate Base and Wrapped Datasets
# ---------------------------------------
# Paths to the COCO JSON file and root directory of images
coco_file_path   = "/data2/datasets/aid4crops_sweet/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json"
dataset_root_dir = "/data2/datasets/aid4crops_sweet"

# Create COCODataset instances for each split
base_train_ds = COCODataset(coco_file=coco_file_path, root_dir=dataset_root_dir, split='train', transform=None)
base_val_ds   = COCODataset(coco_file=coco_file_path, root_dir=dataset_root_dir, split='valid', transform=None)
base_test_ds  = COCODataset(coco_file=coco_file_path, root_dir=dataset_root_dir, split='test', transform=None)

# Wrap them in ImageSegmentationDataset to apply transforms and remapping
train_dataset = ImageSegmentationDataset(base_train_ds, transform=train_transform, target_transform=None)
valid_dataset = ImageSegmentationDataset(base_val_ds,   transform=train_transform, target_transform=None)
test_dataset  = ImageSegmentationDataset(base_test_ds,  transform=test_transform,  target_transform=None)

# Quick sanity check: print shapes for first sample
image_tensor, mask_tensor, orig_img_np, orig_mask_np = train_dataset[0]
print("Sample shapes (train_dataset[0]):")
print("  image tensor shape =", image_tensor.shape)     # (3, H, W)
print("  remapped mask shape =", mask_tensor.shape)      # (H, W)
print("  original image shape =", orig_img_np.shape)      # (H, W, 3)
print("  original mask shape =", orig_mask_np.shape)      # (H, W)

# ---------------------------------------
# 10. Prepare Mask2Former Processor and DataLoaders
# ---------------------------------------
# Initialize a Mask2FormerImageProcessor to handle batching and padding
preprocessor = Mask2FormerImageProcessor(
    ignore_index=255,     # Pixel value to ignore during loss
    reduce_labels=False,  # Keep full label range (no label reduction)
    do_resize=False,      # We assume all images are same size or are handled externally
    do_rescale=False,     # We manually normalize with our transforms
    do_normalize=False,   # We manually normalize with our transforms
    num_labels=len(id2label_remapped)  # Number of output classes
)

def segmentation_collate_fn(batch):
    """
    Collate function for DataLoader that uses the Mask2Former processor.
    Input `batch` is a list of tuples: (image_tensor, mask_tensor, orig_img_np, orig_mask_np).
    This function returns a dict with:
      - pixel_values: stacked image tensors ready for the model
      - mask_labels: padded segmentation masks
      - class_labels: empty list (since we do semantic segmentation, not instance)
      - original_images: list of original PIL/NumPy images (for post-processing)
      - original_segmentation_maps: list of original mask arrays (for metric computation)
    """
    images, masks, orig_images, orig_masks = zip(*batch)

    # The preprocessor expects PIL Images and raw segmentation maps if do_resize/do_rescale/do_normalize=False
    # But here we already converted images to normalized tensors, so call preprocessor with those tensors.
    processed = preprocessor(
        list(images),
        segmentation_maps=list(masks),
        return_tensors="pt"
    )

    # Attach original images and masks for later use (e.g. metric computation)
    processed["original_images"] = orig_images
    processed["original_segmentation_maps"] = orig_masks

    return processed

# Create DataLoaders with appropriate batch sizes and the custom collate function
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=segmentation_collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=segmentation_collate_fn)
test_dataloader  = DataLoader(test_dataset,  batch_size=2, shuffle=False, collate_fn=segmentation_collate_fn)

# Print number of batches for each split
print(f"Number of train batches: {len(train_dataloader)}")
print(f"Number of valid batches: {len(valid_dataloader)}")
print(f"Number of test batches:  {len(test_dataloader)}")

# Inspect one batch's keys and tensor shapes
batch_example = next(iter(train_dataloader))
print("Keys in one batch_example:")
for key, value in batch_example.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape}")
    else:
        # original_images and original_segmentation_maps are lists of NumPy arrays
        print(f"  {key}: (list of length {len(value)})")

# Print a single pixel_values tensor shape and one mask_labels shape
pixel_values = batch_example["pixel_values"][0].numpy()
print("  pixel_values[0] shape (C, H, W):", pixel_values.shape)
print("  mask_labels[0] shape (H, W):", batch_example["mask_labels"][0].shape)

# ---------------------------------------
# 11. Define id2label for Mask2Former Model
# ---------------------------------------
categories_data = [
    {"id": 11, "name": "pepper_kp",       "supercategory": "",        "color": "#0000ff"},
    {"id": 12, "name": "red",             "supercategory": "pepper",  "color": "#c7211c"},
    {"id": 13, "name": "yellow",          "supercategory": "pepper",  "color": "#fff700"},
    {"id": 14, "name": "green",           "supercategory": "pepper",  "color": "#00ff00"},
    {"id": 15, "name": "mixed",           "supercategory": "pepper",  "color": "#e100ff"},
    {"id": 17, "name": "mixed_red",       "supercategory": "pepper",  "color": "#ff6600"},
    {"id": 18, "name": "mixed_yellow",    "supercategory": "pepper",  "color": "#d1c415"},
]

# Build id2label mapping for the model: new_contiguous_id -> label string
id2label_for_model = {new_id: id2label_remapped[new_id] for new_id in range(len(id2label_remapped))}
print("id2label mapping passed to the model:", id2label_for_model)

# ---------------------------------------
# 12. Load Pretrained Mask2Former Model
# ---------------------------------------
# We load a Mask2Former variant pretrained on ADE20K semantics (Swin-base backbone),
# but override the final head to predict our 8-class pepper segmentation.
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-base-ade-semantic",
    id2label=id2label_for_model,
    ignore_mismatched_sizes=True  # In case the head’s number of classes doesn't match
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------------------
# 13. Forward Pass Example (Debugging)
# ---------------------------------------
# Take one batch from the train DataLoader and run a forward pass to ensure shapes align
batch = next(iter(train_dataloader))

# Move input tensors to device
batch_pixel_values = batch["pixel_values"].to(device)
batch_mask_labels = [lbl.to(device) for lbl in batch["mask_labels"]]  # List of (H, W) masks
batch_class_labels = [torch.zeros_like(lbl, dtype=torch.int64).to(device) for lbl in batch["mask_labels"]]
# Note: For semantic segmentation, Mask2Former expects 'mask_labels' and 'class_labels'.
# Here, class_labels can be dummy or derived from mask_labels. We set class_labels to zeros for simplicity.

outputs = model(
    pixel_values=batch_pixel_values,
    mask_labels=batch_mask_labels,
    class_labels=batch_class_labels
)

# Print out the loss and the shape of the predicted segmentation logits
print("Forward pass outputs:")
print("  loss (float):", outputs.loss.item())
print("  logits shape:", outputs.logits.shape)   # (batch_size, num_classes, H_out, W_out)
print("  predicted semantic maps:", len(outputs.predicted_semantic_maps))  # List length == batch_size

# ============================================================
# End of Cleaned and Commented Script
# ============================================================
