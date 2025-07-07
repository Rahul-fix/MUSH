# Import all necessary components from dataloader_semantic.py
from dataloader_semantic import (
    train_dataloader, valid_dataloader, test_dataloader,
    model, preprocessor, id2label_remapped
)

import evaluate
import torch
from tqdm.auto import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

metric = evaluate.load("mean_iou")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=2e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-6)

best_val_loss = float('inf')
best_epoch = 0
running_loss = 0.0
num_samples = 0

# Remove 'class_labels' from dataloader_semantic.py or set it in train.py
# We'll set class_labels as dummy zeros here for semantic segmentation

def get_class_labels(mask_labels):
    return [torch.zeros_like(lbl, dtype=torch.int64) for lbl in mask_labels]

for epoch in range(100):
    print("Epoch:", epoch)
    model.train()
    for idx, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        class_labels = get_class_labels(batch["mask_labels"])
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in class_labels],
        )
        loss = outputs.loss
        loss.backward()
        batch_size = batch["pixel_values"].size(0)
        running_loss += loss.item()
        num_samples += batch_size
        if idx % 100 == 0:
            print("Loss:", running_loss/num_samples)
        optimizer.step()
        # scheduler.step(epoch)

    model.eval()
    val_loss = 0.0
    for idx, batch in enumerate(tqdm(valid_dataloader)):
        with torch.no_grad():
            class_labels = get_class_labels(batch["mask_labels"])
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in class_labels],
            )
            valid_loss = outputs.loss
        val_loss += valid_loss.item()
        original_images = batch["original_images"]
        target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
        predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                      target_sizes=target_sizes)
        ground_truth_segmentation_maps = batch["original_segmentation_maps"]
        metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)
    print("Mean IoU:", metric.compute(num_labels = len(id2label_remapped ), ignore_index=0)['mean_iou'])
    avg_val_loss = val_loss / len(valid_dataloader)
    print("Validation Loss:", avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        model_save_path = f"~/best_model_epoch_{best_epoch}.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {best_epoch} with validation loss: {best_val_loss}")
