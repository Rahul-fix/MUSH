import os
import torch
import argparse
import time
from datetime import datetime
from torch.utils.data import DataLoader
from src.data.coco_dataset import COCODataset, ImageSegmentationDataset
from src.data.transforms import train_transform, test_transform, target_transform
from src.data.cutmix import get_cutmix_transform
from src.models.mask2former import get_mask2former_model, get_preprocessor
from src.training.loop import train
from src.utils.palette import id2label_remapped, label2id
from accelerate import Accelerator
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--cutmix_prob", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 288], help="Image size as [width, height] for resizing input images")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--project_name", type=str, default="pepper-segmentation")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_freq", type=int, default=10, help="Log every N steps")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="Wandb API key")
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    
    # Set Random Seeds for Reproducibility
    seed = 78
    torch.manual_seed(seed)
    
    # Set wandb API key
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    elif "WANDB_API_KEY" not in os.environ:
        api_key_file = os.path.expanduser("~/.wandb_api_key")
        if os.path.exists(api_key_file):
            with open(api_key_file, 'r') as f:
                os.environ["WANDB_API_KEY"] = f.read().strip()
    
    # Initialize Accelerator with wandb tracking
    accelerator = Accelerator(log_with="wandb")

    # Initialize tracking ONLY on main process - SIMPLIFIED
    if accelerator.is_main_process:
        run_name = args.run_name or f"resize_{args.image_size}_lr{args.learning_rate}_bs{args.batch_size}"
        accelerator.init_trackers(
            project_name=args.project_name,
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "name": run_name,
                    "tags": ["resize", f"image_size_{args.image_size}", "mask2former", "pepper-segmentation"],
                    "notes": f"Grid search run with Resize augmentation, image_size={args.image_size}"
                }
            }
        )
        # Log system info
        accelerator.log({
            "system/gpu_count": torch.cuda.device_count(),
            "system/device": str(accelerator.device),
            "system/num_processes": accelerator.num_processes,
            "system/node_name": os.environ.get("HOSTNAME", "unknown")
        })

    # Dataset Paths
    coco_file_path = os.path.expanduser("/scratch/s7rakuma/datasets/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    dataset_root_dir = os.path.expanduser("/scratch/s7rakuma/datasets/")

    # Create Datasets
    base_train_ds = COCODataset(coco_file_path, root_dir=dataset_root_dir, split='train', transform=None)
    base_val_ds = COCODataset(coco_file_path, root_dir=dataset_root_dir, split='valid', transform=None)

    # Set image size for training and validation from argument
    IMAGE_SIZE = tuple(args.image_size)  # (width, height)
    accelerator.print(f"[AUG] Using Resize augmentation with image size: {IMAGE_SIZE}")
    from torchvision import transforms
    import numpy as np
    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255.0
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255.0
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    target_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST)
    ])
    train_dataset = ImageSegmentationDataset(base_train_ds, transform=train_transform, target_transform=target_transform, label2id=label2id)
    valid_dataset = ImageSegmentationDataset(base_val_ds, transform=test_transform, target_transform=target_transform, label2id=label2id)

    # Use args directly - simplified config handling
    TRAIN_BATCH_SIZE = args.batch_size
    VALID_BATCH_SIZE = args.batch_size
    accelerator.print(f"Using train batch size: {TRAIN_BATCH_SIZE}")

    # CutMix removed for this branch
    # No CutMix augmentation. Only Resize augmentation is applied.

    # Collate Functions
    def train_collate_fn(batch):
        images, masks, _, orig_masks = zip(*batch)
        processor = get_preprocessor(len(id2label_remapped))
        proc = processor(list(images), segmentation_maps=list(masks), return_tensors="pt")
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
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=train_collate_fn, num_workers=2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, collate_fn=validation_collate_fn, num_workers=2)

    # Model, Optimizer, Scheduler with args parameters
    model = get_mask2former_model(num_labels=len(id2label_remapped), device=accelerator.device)
    optimizer = optim.SGD(model.parameters(), lr=float(args.learning_rate))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-6)

    # Log dataset and model info
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        accelerator.log({
            "dataset/num_classes": len(id2label_remapped),
            "dataset/class_names": list(id2label_remapped.values()),
            "dataset/train_size": len(train_dataset),
            "dataset/valid_size": len(valid_dataset),
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/backbone": "mask2former-swin-base-ade-semantic",
            "config/image_size": args.image_size,
            "config/learning_rate": args.learning_rate,
            "config/batch_size": args.batch_size
        })

    # Prepare for distributed training
    model, optimizer, scheduler, train_dataloader, valid_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, valid_dataloader
    )

    # Train with args parameters
    train(model, optimizer, train_dataloader, valid_dataloader, id2label_remapped,
          accelerator.device, accelerator, epochs=args.epochs, scheduler=scheduler, 
          log_freq=args.log_freq)

    # End time
    end_time = time.time()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    accelerator.print("End Time:", current_time)

    # Final logging
    if accelerator.is_main_process:
        accelerator.log({
            "training/end_time": current_time,
            "training/total_training_time_hours": (end_time - start_time) / 3600
        })
    
    # End tracking
    accelerator.end_training()

if __name__ == "__main__":
    main()
