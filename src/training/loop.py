import torch
from tqdm.auto import tqdm
import evaluate
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.models.mask2former import get_preprocessor
import numpy as np
from src.utils.palette import remap_labels, label2id

def train(model, train_dataloader, valid_dataloader, id2label_remapped, device, epochs=2):
    optimizer = optim.SGD(model.parameters(), lr=2e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-6)
    best_val_loss = float('inf')
    best_epoch = 0
    running_loss = 0.0
    num_samples = 0

    # Print unique values of remapped masks in the training set (first 5 batches)
    # print("[INFO] Checking unique values in remapped training masks...")
    # for idx, batch in enumerate(train_dataloader):
    #     if idx >= 5:
    #         break
    #     mask_batch = batch["mask_labels"]
    #     for i, mask in enumerate(mask_batch):
    #         mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
    #         print(f"[TRAIN DEBUG] Batch {idx} mask {i} unique: {np.unique(mask_np)}")

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[m.to(device) for m in batch["mask_labels"]],
                class_labels=[c.to(device) for c in batch["class_labels"]],
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
        metric = evaluate.load("mean_iou")
        for idx, batch in enumerate(tqdm(valid_dataloader)):
            with torch.no_grad():
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    mask_labels=[m.to(device) for m in batch["mask_labels"]],
                    class_labels=[c.to(device) for c in batch["class_labels"]],
                )
                valid_loss = outputs.loss
                # Post-process predictions to get semantic maps
                pred_semantic_maps = batch["original_segmentation_maps"]  # ground truth
                preprocessor = get_preprocessor(len(id2label_remapped))
                preds = preprocessor.post_process_semantic_segmentation(
                    outputs, target_sizes=[m.shape for m in batch["original_segmentation_maps"]]
                )
                # Remap validation ground truth masks to contiguous IDs before metric
                refs_np = [remap_labels(np.array(m), label2id).cpu().numpy() for m in pred_semantic_maps]
                preds_np = [p.cpu().numpy() if hasattr(p, 'cpu') else np.array(p) for p in preds]
                # print(f"[DEBUG] Batch {idx} preds shape: {[p.shape for p in preds_np]}, refs shape: {[r.shape for r in refs_np]}")
                # print(f"[DEBUG] Batch {idx} preds unique: {[np.unique(p) for p in preds_np]}")
                # print(f"[DEBUG] Batch {idx} refs unique: {[np.unique(r) for r in refs_np]}")
                metric.add_batch(
                    references=refs_np,
                    predictions=preds_np
                )
            val_loss += valid_loss.item()
        metric_result = metric.compute(num_labels=len(id2label_remapped), ignore_index=0)
        print("Mean IoU:", metric_result["mean_iou"] if metric_result else None)
        avg_val_loss = val_loss / len(valid_dataloader)
        print("Validation Loss:", avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            model_save_path = f"Output/best_model_epoch_{best_epoch}.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {best_epoch} with validation loss: {best_val_loss}")
