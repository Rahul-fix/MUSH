import torch
import numpy as np
from src.models.mask2former import get_mask2former_model, get_preprocessor
from src.utils.palette import id2label_remapped

def main():
    # Settings
    num_classes = len(id2label_remapped)  # Should match your training config
    height, width = 512, 512  # Match your resize transform
    batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy input (simulate a batch of images and masks)
    dummy_image = [np.random.rand(height, width, 3).astype(np.float32) for _ in range(batch_size)]  # float32, [0,1]
    dummy_mask = [np.random.randint(0, num_classes, (height, width), dtype=np.uint8) for _ in range(batch_size)]  # uint8

    # Get preprocessor and model
    preprocessor = get_preprocessor(num_classes)
    model = get_mask2former_model(num_labels=num_classes, device=device)

    # Preprocess (simulate what your collate_fn does)
    inputs = preprocessor(
        list(dummy_image),
        segmentation_maps=list(dummy_mask),
        return_tensors="pt"
    )
    # Move all tensors/lists of tensors in the dict to the correct device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
        elif isinstance(v, list) and all(isinstance(i, torch.Tensor) for i in v):
            inputs[k] = [i.to(device) for i in v]

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        print("Forward pass successful. Output keys:", outputs.keys())
        if "loss" in outputs:
            print("Loss value:", outputs["loss"].item())

if __name__ == "__main__":
    main()