import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import skimage.draw

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
        with open(coco_file, 'r') as f:
            self.coco_data = json.load(f)
        self.split_image_ids = {
            'train': list(range(283, 314)) + list(range(314, 345)) + list(range(408, 471)),
            'valid': list(range(345, 377)) + list(range(533, 564)),
            'test':  list(range(377, 408)) + list(range(471, 533))
        }
        all_images = self.coco_data['images']
        if split in self.split_image_ids:
            valid_ids = set(self.split_image_ids[split])
            self.images = [img for img in all_images if img['id'] in valid_ids]
        else:
            self.images = all_images
        self.annotations = self.coco_data['annotations']
        self.categories = {
            cat['id']: {
                'name': cat['name'],
                'color': cat.get('color', "#000000"),
                'supercategory': cat['supercategory']
            }
            for cat in self.coco_data['categories']
        }
        self.root_dir = root_dir
        self.transform = transform
        self.image_id_to_annotations = {}
        for anno in self.annotations:
            image_id = anno['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(anno)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx: int):
        image_info = self.images[idx]
        image_id = image_info['id']
        width, height = image_info['width'], image_info['height']
        relative_path = image_info['path'].lstrip('/datasets/')
        image_path = os.path.join(self.root_dir, relative_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        annotations = self.image_id_to_annotations.get(image_id, [])
        segmentations = [anno.get('segmentation', []) for anno in annotations]
        category_ids = [anno['category_id'] for anno in annotations]
        semantic_map = np.zeros((height, width), dtype=np.uint8)
        for seg, cat_id in zip(segmentations, category_ids):
            for poly in seg:
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

class ImageSegmentationDataset(Dataset):
    """
    A wrapper around a base dataset (e.g. COCODataset) to:
      - Remap original class IDs in the mask to contiguous IDs
      - Apply image transforms (normalization, augmentation) to the input image
      - Optionally apply target transforms to the segmentation mask
    Returns a tuple: (image_tensor, remapped_mask_tensor, original_image_numpy, original_mask_numpy)
    """
    def __init__(self, base_dataset: Dataset, transform=None, target_transform=None, label2id=None):
        self.dataset = base_dataset
        self.transform = transform
        self.target_transform = target_transform
        self.label2id = label2id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        orig_pil_image = sample['image']
        orig_mask_np = np.array(sample['semantic_map'])
        remapped_mask = remap_labels(orig_mask_np, self.label2id)
        if self.transform:
            image_tensor = self.transform(orig_pil_image)
        else:
            image_tensor = torch.tensor(np.array(orig_pil_image), dtype=torch.float32).permute(2, 0, 1)
        if self.target_transform:
            mask_transformed = self.target_transform(Image.fromarray(remapped_mask.numpy()))
            mask_tensor = torch.tensor(np.array(mask_transformed), dtype=torch.int64)
        else:
            mask_tensor = remapped_mask
        return image_tensor, mask_tensor, np.array(orig_pil_image), orig_mask_np

# Utility function for remapping labels
from src.utils.palette import remap_labels
