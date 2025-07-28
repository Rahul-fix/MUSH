import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from src.data.coco_dataset import COCODataset, ImageSegmentationDataset
from src.data.transforms import train_transform, test_transform, target_transform
from src.models.mask2former import get_mask2former_model, get_preprocessor
from src.training.loop import train
from src.utils.palette import id2label_remapped, label2id
from accelerate import Accelerator

# torch.cuda.empty_cache()

# Initialize Accelerator for distributed training
accelerator = Accelerator()

# GPU AVAILABILITY
# Use accelerator.device instead of torch.device
print(f"Using device: {accelerator.device}")

# TIME CHECK 
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
accelerator.print("Start Time:", current_time)

# adding a print statement to check if the script is running
accelerator.print("Training script is running...")
accelerator.print("*********************************")

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

# DataLoaders
# Only return tensors in the collate_fn for Accelerate compatibility
def segmentation_collate_fn(batch):
    images, masks, orig_images, orig_masks = zip(*batch)
    preprocessor = get_preprocessor(len(id2label_remapped))
    processed = preprocessor(
        list(images),
        segmentation_maps=list(masks),
        return_tensors="pt"
    )
    return {
        "pixel_values": processed["pixel_values"],
        "mask_labels": list(processed["mask_labels"]),
        "class_labels": list(processed["class_labels"]),
        "original_segmentation_maps": list(orig_masks)  # <-- add this for evaluation/visualization
    }

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=segmentation_collate_fn, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=segmentation_collate_fn, num_workers=0)

# Model and preprocessor
model = get_mask2former_model(num_labels=len(id2label_remapped), device=accelerator.device)

# Prepare for distributed training
model, train_dataloader, valid_dataloader = accelerator.prepare(model, train_dataloader, valid_dataloader)

# Train
train(
    model,
    train_dataloader,
    valid_dataloader,
    id2label_remapped,
    accelerator.device,
    epochs=3
)

# TIME CHECK ###
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
accelerator.print("End Time:", current_time)
