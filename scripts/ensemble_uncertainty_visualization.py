"""
Enhanced Uncertainty Estimation with Visualization for Ensemble Semantic Segmentation
Memory-efficient version: processes one model at a time
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from PIL import Image
from torchmetrics.classification import MulticlassCalibrationError

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import your existing modules
from src.models.mask2former import get_mask2former_model, get_preprocessor
from src.data.coco_dataset import COCODataset, ImageSegmentationDataset
from src.data.transforms import test_transform, target_transform
from src.utils.palette import label2id, palette, id2label_remapped, remap_labels


def run_ensemble_inference_sequential(checkpoint_files, ensemble_models_dir, test_dataloader, device, num_labels):
    """
    Run inference sequentially - one model at a time to minimize GPU memory usage.
    """
    print("Running ensemble inference sequentially (one model at a time)...")
    
    all_model_predictions = []
    gt_maps = None
    original_images_subset = []
    
    total_start_time = time.time()
    
    for model_idx, checkpoint_file in enumerate(checkpoint_files):
        print(f"\n🤖 Processing model {model_idx + 1}/{len(checkpoint_files)}: {checkpoint_file}")
        
        # Load model
        model = get_mask2former_model(num_labels=num_labels, device=device)
        checkpoint_path = os.path.join(ensemble_models_dir, checkpoint_file)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model.to(device).eval()
        
        # Process this model on entire dataset
        model_predictions = []
        
        start_time = time.time()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc=f"Model {model_idx+1}")):
                inputs = batch["pixel_values"].to(device)
                outputs = model(inputs)
                
                # Generate model outputs
                class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
                masks_probs = outputs.masks_queries_logits.sigmoid()
                
                # Validate tensor shapes before einsum
                assert class_probs.shape[1] == masks_probs.shape[1], f"Query dimension mismatch: {class_probs.shape[1]} vs {masks_probs.shape[1]}"
                segmentation = torch.einsum("bqc, bqhw -> bchw", class_probs, masks_probs)
                
                # Resize segmentation
                segmentation_resized = F.interpolate(
                    segmentation.float(),
                    size=(inputs.shape[2], inputs.shape[3]),
                    mode="bilinear",
                    align_corners=True
                )
                
                # Move to CPU immediately to free GPU memory
                model_predictions.append(segmentation_resized.cpu())
                
                # Collect ground truth only for first model
                if model_idx == 0:
                    if gt_maps is None:
                        gt_maps = []
                    ground_truth_segmentation_maps = batch["original_segmentation_maps"]
                    gt_maps.append(torch.stack([torch.from_numpy(np.array(gt)) for gt in ground_truth_segmentation_maps]))
                    
                    # Collect original images only for first few batches
                    if batch_idx < 5:
                        original_images_subset.extend(batch.get("original_images", []))
                
                # Clear GPU cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        print(f"Model {model_idx + 1} inference time: {time.time() - start_time:.4f} sec")
        
        # Concatenate predictions for this model and store on CPU
        model_predictions_tensor = torch.cat(model_predictions, dim=0)
        all_model_predictions.append(model_predictions_tensor)
        
        # Delete model to free GPU memory
        del model
        torch.cuda.empty_cache()
        
        print(f"✅ Model {model_idx + 1} completed. GPU memory cleared.")
    
    print(f"\nTotal inference time: {time.time() - total_start_time:.4f} sec")
    
    # Stack all model predictions: [models, samples, classes, height, width]
    print("Stacking all model predictions...")
    segmentation_testing_tensor = torch.stack(all_model_predictions, dim=0)
    
    # Concatenate ground truth
    gt_maps_tensor = torch.cat(gt_maps, dim=0)
    
    # Normalize predictions
    segmentation_testing_tensor = segmentation_testing_tensor / (
        segmentation_testing_tensor.sum(dim=2, keepdim=True) + 1e-8
    )
    
    print(f"✅ Processed {gt_maps_tensor.shape[0]} samples with {len(checkpoint_files)} models")
    print(f"Final tensor shape: {segmentation_testing_tensor.shape}")
    
    return segmentation_testing_tensor, gt_maps_tensor, original_images_subset


def calculate_uncertainty_full_dataset(pred_dist_model, device):
    """
    Calculate uncertainties on entire dataset with proper tensor broadcasting.
    """
    print("Calculating uncertainties on entire dataset...")
    start_time = time.time()
    
    num_models, num_samples, num_classes, height, width = pred_dist_model.shape
    
    # Check what device pred_dist_model is actually on
    pred_device = pred_dist_model.device
    print(f"Predictions are on device: {pred_device}")
    
    # Use the same device as predictions for consistency
    calculation_device = pred_device
    
    # Equal model weights
    posterior_proxy_weighted = torch.full((num_models,), 1.0/num_models, device=calculation_device)
    
    # Calculate aleatoric uncertainty using vectorized operations
    print("Calculating aleatoric uncertainty...")
    log_probs = torch.log2(pred_dist_model + 1e-12)
    entropy_per_model = -(pred_dist_model * log_probs).sum(dim=2)  # Sum over classes: [models, samples, height, width]
    
    # CRITICAL FIX: Reshape weights for proper broadcasting
    # From (6,) to (6, 1, 1, 1) to broadcast with (6, 93, 640, 360)
    weights_entropy = posterior_proxy_weighted.view(num_models, 1, 1, 1)
    aleatoric_uncertainty = (weights_entropy * entropy_per_model).sum(dim=0)  # Sum over models: [samples, height, width]
    
    # Calculate ensemble mean predictions
    print("Calculating ensemble predictions...")
    # CRITICAL FIX: Reshape weights for proper broadcasting  
    # From (6,) to (6, 1, 1, 1, 1) to broadcast with (6, 93, 8, 640, 360)
    weights_pred = posterior_proxy_weighted.view(num_models, 1, 1, 1, 1)
    pred_posterior_model = weights_pred * pred_dist_model
    ensemble_mean = pred_posterior_model.sum(dim=0)  # [samples, classes, height, width]
    
    # Calculate total uncertainty
    print("Calculating total uncertainty...")
    ensemble_log_probs = torch.log2(ensemble_mean + 1e-12)
    total_uncertainty = -(ensemble_mean * ensemble_log_probs).sum(dim=1)  # Sum over classes: [samples, height, width]
    
    # Epistemic uncertainty
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
    
    print(f"Uncertainty calculation time: {time.time() - start_time:.4f} sec")
    print(f"Calculations performed on: {calculation_device}")
    
    # Print tensor shapes for debugging
    print(f"Shape verification:")
    print(f"  - Aleatoric uncertainty: {aleatoric_uncertainty.shape}")
    print(f"  - Epistemic uncertainty: {epistemic_uncertainty.shape}")
    print(f"  - Total uncertainty: {total_uncertainty.shape}")
    print(f"  - Ensemble predictions: {ensemble_mean.shape}")
    
    return {
        'aleatoric_uncertainty': aleatoric_uncertainty,
        'epistemic_uncertainty': epistemic_uncertainty, 
        'total_uncertainty': total_uncertainty,
        'ensemble_predictions': ensemble_mean,
        'model_predictions': pred_dist_model
    }

def save_uncertainty_visualizations_subset(original_images, predictions_mean, aleatoric_unc, 
                                         epistemic_unc, total_unc, gt_masks, save_dir, 
                                         max_visualize=10):
    """
    Visualize only a subset of samples with proper device handling.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'individual'), exist_ok=True)
    
    # Create custom colormap for segmentation
    custom_cmap = mcolors.ListedColormap(np.array(palette) / 255.0)
    
    # Enhanced uncertainty colormap
    uncertainty_colors = ['#000033', '#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#FFFFFF']
    uncertainty_cmap = mcolors.LinearSegmentedColormap.from_list('uncertainty', uncertainty_colors, N=256)
    
    # Only visualize first max_visualize samples
    num_to_visualize = min(len(original_images), max_visualize, predictions_mean.shape[0])
    
    for i in range(num_to_visualize):
        # Prepare original image
        if isinstance(original_images[i], Image.Image):
            img_np = np.array(original_images[i])
        else:
            img_np = original_images[i]
        
        # Resize predictions and uncertainties to original image size
        orig_height, orig_width = img_np.shape[0], img_np.shape[1]
        
        # Ensure all tensors are on CPU for F.interpolate operations
        pred_cpu = predictions_mean[i:i+1].cpu() if predictions_mean[i:i+1].device.type == 'cuda' else predictions_mean[i:i+1]
        alea_cpu = aleatoric_unc[i:i+1].unsqueeze(0).cpu() if aleatoric_unc[i:i+1].device.type == 'cuda' else aleatoric_unc[i:i+1].unsqueeze(0)
        epis_cpu = epistemic_unc[i:i+1].unsqueeze(0).cpu() if epistemic_unc[i:i+1].device.type == 'cuda' else epistemic_unc[i:i+1].unsqueeze(0)
        total_cpu = total_unc[i:i+1].unsqueeze(0).cpu() if total_unc[i:i+1].device.type == 'cuda' else total_unc[i:i+1].unsqueeze(0)
        
        # Resize ensemble prediction
        pred_resized = F.interpolate(
            pred_cpu,
            size=(orig_height, orig_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        pred_mask = torch.argmax(pred_resized, dim=0).cpu().numpy()
        
        # Resize ground truth if needed
        gt_mask_raw = gt_masks[i].cpu().numpy() if torch.is_tensor(gt_masks[i]) else gt_masks[i]
        if gt_mask_raw.shape != (orig_height, orig_width):
            gt_tensor = torch.from_numpy(gt_mask_raw).unsqueeze(0).unsqueeze(0).float()
            gt_resized = F.interpolate(gt_tensor, size=(orig_height, orig_width), mode='nearest').squeeze().numpy().astype(np.int64)
        else:
            gt_resized = gt_mask_raw
        gt_mask_remapped = remap_labels(torch.from_numpy(gt_resized), label2id).numpy()
        
        # Resize uncertainty maps
        alea_resized = F.interpolate(
            alea_cpu, size=(orig_height, orig_width), mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()
        
        epis_resized = F.interpolate(
            epis_cpu, size=(orig_height, orig_width), mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()
        
        total_resized = F.interpolate(
            total_cpu, size=(orig_height, orig_width), mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Top row: Segmentation results
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_np)
        axes[0, 1].imshow(gt_mask_remapped, cmap=custom_cmap, alpha=0.6, vmin=0, vmax=len(palette)-1)
        axes[0, 1].set_title('Ground Truth Segmentation', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(img_np)
        axes[0, 2].imshow(pred_mask, cmap=custom_cmap, alpha=0.6, vmin=0, vmax=len(palette)-1)
        axes[0, 2].set_title('Ensemble Prediction', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Bottom row: Enhanced uncertainty maps
        im1 = axes[1, 0].imshow(alea_resized, cmap=uncertainty_cmap, interpolation='bilinear', vmin=0)
        axes[1, 0].set_title('Aleatoric Uncertainty\n(Data Noise)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        im2 = axes[1, 1].imshow(epis_resized, cmap=uncertainty_cmap, interpolation='bilinear', vmin=0)
        axes[1, 1].set_title('Epistemic Uncertainty\n(Model Disagreement)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        im3 = axes[1, 2].imshow(total_resized, cmap=uncertainty_cmap, interpolation='bilinear', vmin=0)
        axes[1, 2].set_title('Total Uncertainty', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Sample {i}: Uncertainty Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save visualization
        save_path = os.path.join(save_dir, f'uncertainty_analysis_sample_{i:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Saved {num_to_visualize} uncertainty visualizations in {save_dir}")


def main():
    """
    Main function with sequential model processing for memory efficiency.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    coco_json_path = os.path.expanduser("/scratch/s7rakuma/datasets/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    dataset_root_dir = os.path.expanduser("/scratch/s7rakuma/datasets/")
    ensemble_models_dir = "Output/"
    
    checkpoint_files = [
        'best_model_551ouxf3.pt',
        'model_epoch100_551ouxf3.pt', 
        'model_epoch20_551ouxf3.pt',
        'model_epoch40_551ouxf3.pt',
        'model_epoch60_551ouxf3.pt',
        'model_epoch80_551ouxf3.pt'
    ]
    
    output_dir = "Output/uncertainty_visualizations"
    max_visualize = 10
    
    print("🚀 Starting Sequential Ensemble Uncertainty Estimation")
    
    # Validate paths
    if not os.path.exists(coco_json_path):
        raise FileNotFoundError(f"COCO dataset not found: {coco_json_path}")
    
    # Step 1: Create test dataset
    print("\n📁 Loading test dataset...")
    base_test_ds = COCODataset(coco_json_path, dataset_root_dir, split='test')
    test_dataset = ImageSegmentationDataset(
        base_test_ds, transform=test_transform, target_transform=target_transform, label2id=label2id
    )
    
    def inference_collate_fn(batch):
        images, masks, orig_images, orig_masks = zip(*batch)
        preprocessor = get_preprocessor(len(id2label_remapped))
        processed = preprocessor(list(images), segmentation_maps=list(masks), return_tensors="pt")
        return {
            "pixel_values": processed["pixel_values"],
            "original_images": list(orig_images),
            "original_segmentation_maps": list(orig_masks)
        }
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=3, shuffle=False, collate_fn=inference_collate_fn, num_workers=2
    )
    
    # Step 2: Run sequential inference (one model at a time)
    print("\n🔄 Processing models sequentially...")
    pred_dist_model, gt_maps_tensor, original_images_subset = run_ensemble_inference_sequential(
        checkpoint_files, ensemble_models_dir, test_dataloader, device, len(id2label_remapped)
    )
    
    # Step 3: Calculate uncertainties on ALL samples
    print("\n📊 Calculating uncertainties on all samples...")
    uncertainty_results = calculate_uncertainty_full_dataset(pred_dist_model, device)
    
    # Step 4: Calculate calibration error on full dataset
    print("\n📏 Calculating calibration error on full dataset...")

    def prepare_calibration_tensors(pred_dist_model, gt_maps_tensor, num_classes):
        """
        Prepare tensors for MulticlassCalibrationError metric.
        """
        # Average predictions over models: (6, 93, 8, 640, 360) -> (93, 8, 640, 360)
        mean_segmentation = pred_dist_model.mean(dim=0)
        
        N, C, H, W = mean_segmentation.shape
        
        # Reshape predictions: (93, 8, 640, 360) -> (93, 8, 230400)
        preds = mean_segmentation.reshape(N, C, H * W).cpu()
        
        # Ensure ground truth has same spatial dimensions as predictions
        if gt_maps_tensor.shape[-2:] != (H, W):
            print(f"Resizing ground truth from {gt_maps_tensor.shape} to match predictions...")
            gt_resized = F.interpolate(
                gt_maps_tensor.unsqueeze(1).float(), 
                size=(H, W), 
                mode='nearest'
            ).squeeze(1).long()
        else:
            gt_resized = gt_maps_tensor.long()
        
        # Reshape target: (93, 640, 360) -> (93, 230400)  
        target = gt_resized.reshape(N, H * W).cpu()
        
        # Validate class ranges
        unique_classes = torch.unique(target)
        print(f"Ground truth classes found: {unique_classes.tolist()}")
        print(f"Expected classes: 0 to {num_classes-1}")
        
        # Ensure target values are in valid range [0, num_classes-1]
        target = torch.clamp(target, 0, num_classes-1)
        
        return preds, target

    # Prepare tensors for calibration metric
    try:
        metric = MulticlassCalibrationError(num_classes=len(id2label_remapped), n_bins=10, norm='l1')
        
        preds, target = prepare_calibration_tensors(pred_dist_model, gt_maps_tensor, len(id2label_remapped))
        
        # Validate tensor shapes before calling metric
        print(f"Calibration tensor shapes:")
        print(f"  - preds: {preds.shape} (should be [N, C, *])")
        print(f"  - target: {target.shape} (should be [N, *])")
        print(f"  - preds dtype: {preds.dtype}")
        print(f"  - target dtype: {target.dtype}")
        
        # Validate shape requirements
        assert len(preds.shape) == len(target.shape) + 1, f"preds should have one more dimension than target"
        assert preds.shape[0] == target.shape[0], f"Batch sizes must match: {preds.shape[0]} vs {target.shape[0]}"
        assert preds.shape[1] == len(id2label_remapped), f"Number of classes must match: {preds.shape[1]} vs {len(id2label_remapped)}"
        assert preds.shape[2:] == target.shape[1:], f"Spatial dimensions must match: {preds.shape[2:]} vs {target.shape[1:]}"
        
        calibration_error = metric(preds, target)
        print(f"📈 Calibration Error (full dataset): {calibration_error:.4f}")
        
    except Exception as e:
        print(f"⚠️  Calibration error calculation failed: {str(e)}")
        print("Skipping calibration metric and continuing with visualization...")
        calibration_error = float('nan')

    
    # Step 5: Visualize only subset
    print(f"\n🎨 Generating visualizations for {max_visualize} samples...")
    save_uncertainty_visualizations_subset(
        original_images_subset,
        uncertainty_results['ensemble_predictions'],
        uncertainty_results['aleatoric_uncertainty'],
        uncertainty_results['epistemic_uncertainty'], 
        uncertainty_results['total_uncertainty'],
        gt_maps_tensor,
        output_dir,
        max_visualize=max_visualize
    )
    
    # Step 6: Print comprehensive statistics
    print("\n📋 Full Dataset Statistics:")
    print(f"  • Total samples processed: {gt_maps_tensor.shape[0]}")
    print(f"  • Number of ensemble models: {len(checkpoint_files)}")
    print(f"  • Number of classes: {len(id2label_remapped)}")
    print(f"  • Calibration error: {calibration_error:.4f}")
    print(f"  • Samples visualized: {max_visualize}")
    
    # Print uncertainty statistics for entire dataset
    for unc_type, unc_tensor in [
        ("Aleatoric", uncertainty_results['aleatoric_uncertainty']),
        ("Epistemic", uncertainty_results['epistemic_uncertainty']),
        ("Total", uncertainty_results['total_uncertainty'])
    ]:
        mean_unc = unc_tensor.mean().item()
        std_unc = unc_tensor.std().item()
        max_unc = unc_tensor.max().item()
        print(f"  • {unc_type} uncertainty - Mean: {mean_unc:.4f}, Std: {std_unc:.4f}, Max: {max_unc:.4f}")
    
    print(f"\n✅ Processing completed! Visualizations saved to: {output_dir}")
    return uncertainty_results


if __name__ == "__main__":
    results = main()
