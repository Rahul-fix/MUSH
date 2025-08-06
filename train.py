import os
import torch
import argparse
import wandb
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
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--project_name", type=str, default="pepper-segmentation-sweep")
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
    
    # Initialize tracking only on main process
    if accelerator.is_main_process:
        # Check if we're in a sweep (wandb.run exists) or manual run
        if wandb.run is None:
            # Manual run - use args to initialize wandb
            run_name = args.run_name or f"cutmix_a{args.cutmix_alpha}_p{args.cutmix_prob}_lr{args.learning_rate}_bs{args.batch_size}"
            
            accelerator.init_trackers(
                project_name=args.project_name,
                config=vars(args),
                init_kwargs={
                    "wandb": {
                        "name": run_name,
                        "tags": ["cutmix", "mask2former", "pepper-segmentation"],
                        "notes": f"Manual run with CutMix alpha={args.cutmix_alpha}, prob={args.cutmix_prob}"
                    }
                }
            )
        else:
            # Sweep run - wandb is already initialized, just init accelerate tracking
            accelerator.init_trackers(
                project_name=wandb.run.project,
                config=dict(wandb.config),
                init_kwargs={
                    "wandb": {
                        "name": wandb.run.name,
                        "tags": ["cutmix", "mask2former", "pepper-segmentation", "sweep"],
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

    # Use wandb.config if available (sweep), otherwise use args (manual run)
    if wandb.run is not None:
        config = wandb.config
        accelerator.print("Using wandb.config from sweep")
    else:
        # Create a config-like object from args for consistency
        class ConfigFromArgs:
            def __init__(self, args):
                for key, value in vars(args).items():
                    setattr(self, key, value)
        config = ConfigFromArgs(args)
        accelerator.print("Using args config for manual run")

    # Dataset Paths
    coco_file_path = os.path.expanduser("/scratch/s7rakuma/datasets/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    dataset_root_dir = os.path.expanduser("/scratch/s7rakuma/datasets/")

    # Create Datasets
    base_train_ds = COCODataset(coco_file_path, root_dir=dataset_root_dir, split='train', transform=None)
    base_val_ds = COCODataset(coco_file_path, root_dir=dataset_root_dir, split='valid', transform=None)

    # Create ImageSegmentationDataset with transforms(normalization, resizing)
    train_dataset = ImageSegmentationDataset(base_train_ds, transform=train_transform, target_transform=target_transform, label2id=label2id)
    valid_dataset = ImageSegmentationDataset(base_val_ds, transform=train_transform, target_transform=target_transform, label2id=label2id)

    # Use config (from wandb.config or args)
    TRAIN_BATCH_SIZE = config.batch_size
    VALID_BATCH_SIZE = config.batch_size
    accelerator.print(f"Using train batch size: {TRAIN_BATCH_SIZE}")

    # CutMix Setup with config parameters
    os.makedirs("Output/cutmix", exist_ok=True)
    CUTMIX = get_cutmix_transform(
        num_classes=len(id2label_remapped),
        alpha=config.cutmix_alpha,
        prob=config.cutmix_prob,
        save_samples=True
    )
    accelerator.print(f"CutMix enabled: alpha={config.cutmix_alpha}, prob={config.cutmix_prob}")

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
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=train_collate_fn, num_workers=2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, collate_fn=validation_collate_fn, num_workers=2)

    # Model, Optimizer, Scheduler with config parameters
    model = get_mask2former_model(num_labels=len(id2label_remapped), device=accelerator.device)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
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
            "config/cutmix_alpha": config.cutmix_alpha,
            "config/cutmix_prob": config.cutmix_prob,
            "config/learning_rate": config.learning_rate,
            "config/batch_size": config.batch_size
        })

    # Prepare for distributed training
    model, optimizer, scheduler, train_dataloader, valid_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, valid_dataloader
    )

    # Train with config parameters
    train(model, optimizer, train_dataloader, valid_dataloader, id2label_remapped,
          accelerator.device, accelerator, epochs=config.epochs, scheduler=scheduler, 
          log_freq=getattr(config, 'log_freq', 10))

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
