import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from src.data.coco_dataset import COCODataset, ImageSegmentationDataset
from src.data.transforms import train_transform, test_transform
from src.models.mask2former import get_mask2former_model, get_preprocessor
from src.training.loop import train
from src.utils.palette import id2label_remapped, label2id

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
# coco_file_path = os.path.expanduser("~//Thesis/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
# dataset_root_dir = os.path.expanduser("~/Downloads/Thesis")
# Paths to dataset (linux)
coco_file_path = os.path.expanduser("/scratch/s7rakuma/datasets/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
dataset_root_dir = os.path.expanduser("/scratch/s7rakuma/datasets")

# Instantiate base datasets
base_train_ds = COCODataset(coco_file=coco_file_path, root_dir=dataset_root_dir, split='train', transform=None)
base_val_ds   = COCODataset(coco_file=coco_file_path, root_dir=dataset_root_dir, split='valid', transform=None)

# Wrapped datasets
train_dataset = ImageSegmentationDataset(base_train_ds, transform=train_transform, target_transform=None, label2id=label2id)
valid_dataset = ImageSegmentationDataset(base_val_ds,   transform=train_transform, target_transform=None, label2id=label2id)

# DataLoaders
def segmentation_collate_fn(batch):
    images, masks, orig_images, orig_masks = zip(*batch)
    preprocessor = get_preprocessor(len(id2label_remapped))
    processed = preprocessor(
        list(images),
        segmentation_maps=list(masks),
        return_tensors="pt"
    )
    processed["original_images"] = orig_images
    processed["original_segmentation_maps"] = orig_masks
    return processed

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=segmentation_collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=segmentation_collate_fn)

# Model and preprocessor
model = get_mask2former_model(num_labels=len(id2label_remapped), device=device)

# Train
train(
    model,
    train_dataloader,
    valid_dataloader,
    id2label_remapped,
    device,
    epochs=100
)

# TIME CHECK ###
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("End Time:", current_time)
