import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from src.data.coco_dataset import COCODataset, ImageSegmentationDataset
from src.data.transforms import train_transform, test_transform, target_transform
from src.models.mask2former import get_mask2former_model, get_preprocessor
from src.training.loop import train
from src.utils.palette import id2label_remapped, label2id
from collections import Counter
import numpy as np

# GPU AVAILABILITY
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.current_device())
print(torch.cuda.get_device_name())

# TIME CHECK 
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("Start Time:", current_time)

# adding a print statement to check if the script is running
print("Training script is running...")
print("*********************************")

# # Paths to dataset (macos)
coco_file_path = os.path.expanduser("/scratch/s7rakuma/datasets/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
dataset_root_dir = os.path.expanduser("/scratch/s7rakuma/datasets/")
# Paths to dataset (linux)
# coco_file_path = os.path.expanduser("/home/rkumar/Downloads/Thesis/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
# dataset_root_dir = os.path.expanduser("/home/rkumar/Downloads/Thesis")

# Instantiate base datasets
base_train_ds = COCODataset(coco_file=coco_file_path, root_dir=dataset_root_dir, split='train', transform=None)
base_val_ds   = COCODataset(coco_file=coco_file_path, root_dir=dataset_root_dir, split='valid', transform=None)

# Wrapped datasets
train_dataset = ImageSegmentationDataset(base_train_ds, transform=train_transform, target_transform=target_transform, label2id=label2id)
valid_dataset = ImageSegmentationDataset(base_val_ds,   transform=train_transform, target_transform=target_transform, label2id=label2id)

# Print class label distribution at startup
def print_class_distribution(dataset, split_name):
    all_labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        # Use the correct mask key for this dataset
        if isinstance(sample, dict):
            if 'semantic_map' in sample:
                mask = sample['semantic_map']
            else:
                raise KeyError(f"Sample dict does not contain 'semantic_map' key. Keys: {list(sample.keys())}")
        else:
            mask = sample[1]
        mask_np = np.array(mask)
        valid_pixels = mask_np[mask_np != 255].flatten()
        all_labels.extend(valid_pixels.tolist())
    label_counts = Counter(all_labels)
    print(f"[INFO] Class label distribution for {split_name}:")
    for k in sorted(label_counts.keys()):
        print(f"  Label {k}: {label_counts[k]}")

print_class_distribution(base_train_ds, "train")
print_class_distribution(base_val_ds, "valid")

# DataLoaders
def segmentation_collate_fn(batch):
    images, masks, orig_images, orig_masks = zip(*batch)
    # Debug: Check for invalid masks
    filtered = []
    for idx, (img, mask, oimg, omask) in enumerate(zip(images, masks, orig_images, orig_masks)):
        mask_np = np.array(mask)
        if np.isnan(mask_np).any() or np.isinf(mask_np).any():
            print(f"[ERROR] NaN or Inf in mask at batch idx {idx}")
        if mask_np.max() == mask_np.min():
            print(f"[WARNING] Mask at batch idx {idx} is constant: {mask_np.max()}")
        if mask_np.sum() == 0:
            print(f"[WARNING] Mask at batch idx {idx} is all zeros")
        # Filter: skip if all mask pixels are 0 or 255
        valid_pixels = mask_np[(mask_np != 0) & (mask_np != 255)]
        if valid_pixels.size > 0:
            filtered.append((img, mask, oimg, omask))
        else:
            print(f"[WARNING] Skipping sample at batch idx {idx} (all bg/ignore)")
    if len(filtered) == 0:
        print("[WARNING] All samples in batch are empty after filtering!")
        # Return a dummy batch to avoid crash (will be skipped in training loop)
        return None
    images, masks, orig_images, orig_masks = zip(*filtered)
    preprocessor = get_preprocessor(len(id2label_remapped))
    processed = preprocessor(
        list(images),
        segmentation_maps=list(masks),
        return_tensors="pt"
    )
    processed["original_images"] = orig_images
    processed["original_segmentation_maps"] = orig_masks
    # Move all tensors/lists of tensors in the dict to the correct device
    for k, v in processed.items():
        if isinstance(v, torch.Tensor):
            processed[k] = v.to(device)
        elif isinstance(v, list) and all(isinstance(i, torch.Tensor) for i in v):
            processed[k] = [i.to(device) for i in v]
    return processed

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=segmentation_collate_fn, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=segmentation_collate_fn, num_workers=0)

# Model and preprocessor
model = get_mask2former_model(num_labels=len(id2label_remapped), device=device)

# Train
train(
    model,
    train_dataloader,
    valid_dataloader,
    id2label_remapped,
    device,
    epochs=10
)

# TIME CHECK ###
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("End Time:", current_time)
