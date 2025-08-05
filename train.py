import os
import torch
# import random
# import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from src.data.coco_dataset import COCODataset, ImageSegmentationDataset
from src.data.transforms import train_transform, test_transform, target_transform
from src.data.cutmix import get_cutmix_transform
from src.models.mask2former import get_mask2former_model, get_preprocessor
from src.training.loop import train
from src.utils.palette import id2label_remapped, label2id
from accelerate import Accelerator

# Set Random Seeds for Reproducibility
seed = 78
torch.manual_seed(seed)
# random.seed(seed) 
# np.random.seed(seed) # cutmix.py should not use seed for randomness

# Initialize Accelerator
accelerator = Accelerator()

# GPU and Time Setup
print(f"Using device: {accelerator.device}")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
accelerator.print("Start Time:", current_time)
accelerator.print("Training script is running...")

# Dataset Paths
coco_file_path = os.path.expanduser("/scratch/s7rakuma/datasets/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
dataset_root_dir = os.path.expanduser("/scratch/s7rakuma/datasets/")

# Create Datasets
base_train_ds = COCODataset(coco_file_path, root_dir=dataset_root_dir, split='train', transform=None)
base_val_ds = COCODataset(coco_file_path, root_dir=dataset_root_dir, split='valid', transform=None)

train_dataset = ImageSegmentationDataset(base_train_ds, transform=train_transform, target_transform=target_transform, label2id=label2id)
valid_dataset = ImageSegmentationDataset(base_val_ds, transform=train_transform, target_transform=target_transform, label2id=label2id)

# Batch Size Configuration
TRAIN_BATCH_SIZE = 3
VALID_BATCH_SIZE = 3
accelerator.print(f"Using train batch size: {TRAIN_BATCH_SIZE} (CutMix compatible)")

# CutMix Setup (ONLY ONE INITIALIZATION)
os.makedirs("Output/cutmix", exist_ok=True)
CUTMIX = get_cutmix_transform(
    num_classes=len(id2label_remapped),
    alpha=1.0,
    prob=0.5,
    save_samples=True
)
accelerator.print("CutMix enabled with visualization")

# Collate Functions
def train_collate_fn(batch):
    images, masks, _, orig_masks = zip(*batch)
    processor = get_preprocessor(len(id2label_remapped))
    proc = processor(list(images), segmentation_maps=list(masks), return_tensors="pt")
    
    # Apply CutMix only during training
    proc["pixel_values"], proc["mask_labels"] = CUTMIX(proc["pixel_values"], proc["mask_labels"])
    
    return {
        "pixel_values": proc["pixel_values"],
        "mask_labels": list(proc["mask_labels"]),
        "class_labels": list(proc["class_labels"]),
        "original_segmentation_maps": list(orig_masks)
    }

def validation_collate_fn(batch):
    images, masks, orig_images, orig_masks = zip(*batch)
    preprocessor = get_preprocessor(len(id2label_remapped))
    processed = preprocessor(list(images), segmentation_maps=list(masks), return_tensors="pt")
    
    # NO CutMix for validation
    return {
        "pixel_values": processed["pixel_values"],
        "mask_labels": list(processed["mask_labels"]),
        "class_labels": list(processed["class_labels"]),
        "original_segmentation_maps": list(orig_masks)
    }

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=train_collate_fn, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, collate_fn=validation_collate_fn, num_workers=0)

# Model, Optimizer, Scheduler
model = get_mask2former_model(num_labels=len(id2label_remapped), device=accelerator.device)

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

optimizer = optim.SGD(model.parameters(), lr=2e-4)  
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-6)

# Prepare for distributed training
model, optimizer, scheduler, train_dataloader, valid_dataloader = accelerator.prepare(
    model, optimizer, scheduler, train_dataloader, valid_dataloader
)

# Train
train(model, optimizer, train_dataloader, valid_dataloader, id2label_remapped, 
      accelerator.device, accelerator, epochs=100, scheduler=scheduler)

# End time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
accelerator.print("End Time:", current_time)
