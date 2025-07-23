"""
visualize_inference.py
=====================

This script visualizes the inference results of a trained Mask2Former model on a test dataset.
It allows you to interactively browse test images, view their ground truth and predicted segmentation masks, and inspect class/color mappings.

Features:
- Loads a trained Mask2Former model checkpoint.
- Loads a COCO-format annotation JSON and dataset images.
- Lists all test images in the dataset.
- Prompts the user to select an image index to visualize.
- Displays the input image, ground truth mask, and predicted mask side by side.
- Prints class mappings and unique classes present in each image.

Usage:
------
Run from the command line:

    python scripts/visualize_inference.py \
        --checkpoint <path_to_checkpoint> \
        --coco_json <path_to_coco_json> \
        --dataset_root <dataset_root_dir>

All arguments are optional if you have the corresponding environment variables set:
    CHECKPOINT_PATH, COCO_JSON, DATASET_ROOT
or if you want to use the defaults.

Example:
    python scripts/visualize_inference.py --checkpoint best_model_epoch_13.pt --coco_json CKA_sweet_pepper_2020_summer.json --dataset_root .

Interactive mode:
    - The script will list all test images.
    - Enter the index of an image to visualize its input, ground truth, and predicted segmentation masks.
    - Enter 'q' to quit.

Dependencies:
    - torch
    - matplotlib
    - numpy
    - transformers (for Mask2FormerImageProcessor)
    - Project-specific modules in src/

"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import Mask2FormerImageProcessor
import matplotlib.colors as mcolors

# Adjust paths so that src/ is importable when running from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import get_mask2former_model
from src.data.coco_dataset import COCODataset, ImageSegmentationDataset
from src.data.transforms import test_transform, target_transform
from src.utils.palette import label2id, palette, id2label_remapped, remap_labels

import argparse

# Parse command-line arguments or use environment variables for paths
parser = argparse.ArgumentParser(description="Visualize Mask2Former inference results.")
parser.add_argument('--checkpoint', type=str, default=os.environ.get('CHECKPOINT_PATH', 'best_model_epoch_13.pt'),
                    help='Path to the model checkpoint file.')
parser.add_argument('--coco_json', type=str, default=os.environ.get('COCO_JSON', 'CKA_sweet_pepper_2020_summer.json'),
                    help='Path to the COCO annotation JSON file.')
parser.add_argument('--dataset_root', type=str, default=os.environ.get('DATASET_ROOT', '~/Downloads/Thesis/'),
                    help='Root directory of the dataset.')
args, unknown = parser.parse_known_args()

CHECKPOINT_PATH = args.checkpoint
COCO_JSON = os.path.expanduser(args.coco_json)
dataset_root_dir = os.path.expanduser(args.dataset_root)

NUM_SAMPLES = 6  # Number of test samples to visualize

# Print the number of classes and the mapping from original to contiguous IDs
print(f"Number of classes: {len(label2id)}")
print("Class mapping (original_id: contiguous_id):")
for orig_id, contig_id in label2id.items():
    print(f"  {orig_id}: {contig_id}")

def load_model(checkpoint_path, num_labels, device):
    """
    Load the trained Mask2Former model from a checkpoint file.
    Args:
        checkpoint_path (str): Path to the model checkpoint.
        num_labels (int): Number of output classes.
        device (torch.device): Device to load the model on.
    Returns:
        model (torch.nn.Module): Loaded model in eval mode.
    """
    model_tuple = get_mask2former_model(num_labels=num_labels, device=device)
    if isinstance(model_tuple, tuple):
        model = model_tuple[0]
    else:
        model = model_tuple
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    return model

def plot_sample(img_np, mask_pred, mask_gt, idx):
    """
    Plot the input image, ground truth mask, and predicted mask side by side.
    Args:
        img_np (np.ndarray): Input image as numpy array.
        mask_pred (np.ndarray): Predicted mask.
        mask_gt (np.ndarray): Ground truth mask.
        idx (int): Sample index (for display purposes).
    """
    # Create a custom color map for the masks
    custom_cmap = mcolors.ListedColormap(np.array(palette) / 255.0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Show input image
    axs[0].imshow(img_np)
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    # Overlay ground truth mask
    axs[1].imshow(img_np)
    axs[1].imshow(mask_gt, cmap=custom_cmap, vmin=0, vmax=len(palette)-1, alpha=0.6)
    axs[1].set_title('Ground Truth')
    axs[1].axis('off')
    # Overlay predicted mask
    axs[2].imshow(img_np)
    axs[2].imshow(mask_pred, cmap=custom_cmap, vmin=0, vmax=len(palette)-1, alpha=0.6)
    axs[2].set_title('Predicted Mask')
    axs[2].axis('off')
    plt.tight_layout()
    plt.show()

def run_inference_on_test():
    """
    Run inference on the test set and visualize a few samples.
    """
    # Select device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create base COCO dataset and segmentation dataset with transforms
    base_ds = COCODataset(COCO_JSON, dataset_root_dir, split='test')
    test_ds = ImageSegmentationDataset(base_ds, transform=test_transform, target_transform=target_transform, label2id=label2id)
    num_labels = len(label2id)
    # Load the trained model
    model = load_model(CHECKPOINT_PATH, num_labels, device)
    # Create the processor for post-processing model outputs
    processor = Mask2FormerImageProcessor(ignore_index=255, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False, num_labels=num_labels)
    for i in range(NUM_SAMPLES):
        # Get image and mask tensors, and their numpy versions
        image_tensor, mask_tensor, img_np, mask_gt = test_ds[i]
        input_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        # Use processor to get per-pixel class predictions
        target_size = (img_np.shape[0], img_np.shape[1])
        semantic_seg = processor.post_process_semantic_segmentation(output, target_sizes=[target_size])[0]
        mask_pred_resized = semantic_seg.cpu().numpy()
        # Remap ground truth mask to contiguous IDs
        mask_gt_remapped = remap_labels(mask_gt, label2id).numpy()
        # Print unique classes in ground truth and prediction
        gt_classes = np.unique(mask_gt_remapped)
        pred_classes = np.unique(mask_pred_resized)
        print(f"Image {i}: Ground truth classes: {gt_classes} (count: {len(gt_classes)})")
        for class_id in gt_classes:
            name = id2label_remapped.get(class_id, 'unknown')
            color = palette[class_id] if class_id < len(palette) else 'N/A'
            print(f"  GT class {class_id}: {name}, Color: {color}")
        print(f"Image {i}: Predicted classes: {pred_classes} (count: {len(pred_classes)})")
        for class_id in pred_classes:
            name = id2label_remapped.get(class_id, 'unknown')
            color = palette[class_id] if class_id < len(palette) else 'N/A'
            print(f"  Pred class {class_id}: {name}, Color: {color}")
        # Visualize the sample
        plot_sample(img_np, mask_pred_resized, mask_gt_remapped, i)

def list_test_images():
    """
    List the file names or IDs of all test images in the dataset.
    """
    base_ds = COCODataset(COCO_JSON, dataset_root_dir, split='test')
    # Try to get file names or IDs from the dataset
    if hasattr(base_ds, 'images'):
        # COCO-style: images is a list of dicts with 'file_name' or 'id'
        for idx, img_info in enumerate(base_ds.images):
            file_name = img_info.get('file_name', None)
            img_id = img_info.get('id', None)
            print(f"[{idx}] file_name: {file_name}, id: {img_id}")
    elif hasattr(base_ds, 'img_files'):
        # Custom dataset: list of file paths
        for idx, file_path in enumerate(base_ds.img_files):
            print(f"[{idx}] {file_path}")
    else:
        print("Could not find image file names or IDs in the dataset.")


def visualize_image_by_index(index):
    """
    Visualize a specific test image by its index in the dataset.
    Args:
        index (int): Index of the image to visualize.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_ds = COCODataset(COCO_JSON, dataset_root_dir, split='test')
    test_ds = ImageSegmentationDataset(base_ds, transform=test_transform, target_transform=target_transform, label2id=label2id)
    num_labels = len(label2id)
    model = load_model(CHECKPOINT_PATH, num_labels, device)
    processor = Mask2FormerImageProcessor(ignore_index=255, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False, num_labels=num_labels)
    image_tensor, mask_tensor, img_np, mask_gt = test_ds[index]
    input_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    target_size = (img_np.shape[0], img_np.shape[1])
    semantic_seg = processor.post_process_semantic_segmentation(output, target_sizes=[target_size])[0]
    mask_pred_resized = semantic_seg.cpu().numpy()
    mask_gt_remapped = remap_labels(mask_gt, label2id).numpy()
    gt_classes = np.unique(mask_gt_remapped)
    pred_classes = np.unique(mask_pred_resized)
    print(f"Image {index}: Ground truth classes: {gt_classes} (count: {len(gt_classes)})")
    for class_id in gt_classes:
        name = id2label_remapped.get(class_id, 'unknown')
        color = palette[class_id] if class_id < len(palette) else 'N/A'
        print(f"  GT class {class_id}: {name}, Color: {color}")
    print(f"Image {index}: Predicted classes: {pred_classes} (count: {len(pred_classes)})")
    for class_id in pred_classes:
        name = id2label_remapped.get(class_id, 'unknown')
        color = palette[class_id] if class_id < len(palette) else 'N/A'
        print(f"  Pred class {class_id}: {name}, Color: {color}")
    plot_sample(img_np, mask_pred_resized, mask_gt_remapped, index)

if __name__ == '__main__':
    # Interactive mode: list images and prompt user for index
    print("Listing all test images:")
    list_test_images()
    while True:
        try:
            idx = input("\nEnter the index of the image to visualize (or 'q' to quit): ")
            if idx.lower() == 'q':
                print("Exiting.")
                break
            idx = int(idx)
            visualize_image_by_index(idx)
        except ValueError:
            print("Please enter a valid integer index or 'q' to quit.")
        except IndexError:
            print("Index out of range. Please enter a valid index from the list.")
        except Exception as e:
            print(f"Error: {e}")
