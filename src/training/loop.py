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
            # --- Device and NaN/Inf checks ---
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                    if torch.isnan(batch[k]).any() or torch.isinf(batch[k]).any():
                        raise ValueError(f"[ERROR] NaN or Inf in tensor '{k}' at batch {idx}")
                elif isinstance(v, list) and all(isinstance(i, torch.Tensor) for i in v):
                    batch[k] = [i.to(device) for i in v]
                    for i, t in enumerate(batch[k]):
                        if torch.isnan(t).any() or torch.isinf(t).any():
                            raise ValueError(f"[ERROR] NaN or Inf in list tensor '{k}[{i}]' at batch {idx}")
            # --- Debug prints ---
            for i, m in enumerate(batch["mask_labels"]):
                print(f"[DEBUG] Batch {idx} mask_labels[{i}] unique: {torch.unique(m)}, shape: {m.shape}, min: {m.min()}, max: {m.max()}")
            for i, c in enumerate(batch["class_labels"]):
                print(f"[DEBUG] Batch {idx} class_labels[{i}]: {c}")
            print(f"[DEBUG] Batch {idx} pixel_values shape: {batch['pixel_values'].shape}, min: {batch['pixel_values'].min()}, max: {batch['pixel_values'].max()}")
            # --- End debug ---
            # Remove empty/ignore-only masks and filter class_labels to match mask content
            filtered_mask_labels = []
            filtered_class_labels = []
            for m, c in zip(batch["mask_labels"], batch["class_labels"]):
                present_classes = torch.unique(m)
                present_classes = present_classes[(present_classes != 0) & (present_classes != 255)]
                valid_class_labels = c[torch.isin(c, present_classes)]
                if len(present_classes) > 0 and len(valid_class_labels) > 0:
                    filtered_mask_labels.append(m)
                    filtered_class_labels.append(valid_class_labels)
            # Skip batch if any sample has empty mask or class_labels after filtering
            if len(filtered_mask_labels) == 0 or len(filtered_class_labels) == 0 or any(len(c) == 0 for c in filtered_class_labels):
                print(f"[WARNING] Skipping batch {idx} as some masks or class labels are empty after filtering.")
                continue
            batch["mask_labels"] = filtered_mask_labels
            batch["class_labels"] = filtered_class_labels

            try:
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    mask_labels=[m for m in batch["mask_labels"]],
                    class_labels=[c for c in batch["class_labels"]],
                )
            except Exception as e:
                print(f"[ERROR] Exception in model forward at batch {idx}: {e}")
                print(f"[ERROR] Batch {idx} mask_labels: {[torch.unique(m) for m in batch['mask_labels']]}")
                print(f"[ERROR] Batch {idx} class_labels: {batch['class_labels']}")
                print(f"[ERROR] Batch {idx} pixel_values shape: {batch['pixel_values'].shape}")
                raise
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
                # --- Device and NaN/Inf checks for validation ---
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                        if torch.isnan(batch[k]).any() or torch.isinf(batch[k]).any():
                            raise ValueError(f"[ERROR] NaN or Inf in tensor '{k}' at batch {idx} (val)")
                    elif isinstance(v, list) and all(isinstance(i, torch.Tensor) for i in v):
                        batch[k] = [i.to(device) for i in v]
                        for i, t in enumerate(batch[k]):
                            if torch.isnan(t).any() or torch.isinf(t).any():
                                raise ValueError(f"[ERROR] NaN or Inf in list tensor '{k}[{i}]' at batch {idx} (val)")
                # --- Debug prints ---
                for i, m in enumerate(batch["mask_labels"]):
                    print(f"[DEBUG] [VAL] Batch {idx} mask_labels[{i}] unique: {torch.unique(m)}, shape: {m.shape}, min: {m.min()}, max: {m.max()}")
                for i, c in enumerate(batch["class_labels"]):
                    print(f"[DEBUG] [VAL] Batch {idx} class_labels[{i}]: {c}")
                print(f"[DEBUG] [VAL] Batch {idx} pixel_values shape: {batch['pixel_values'].shape}, min: {batch['pixel_values'].min()}, max: {batch['pixel_values'].max()}")
                # --- End debug ---
                # Remove empty/ignore-only masks and their class labels
                filtered_mask_labels = []
                filtered_class_labels = []
                for m, c in zip(batch["mask_labels"], batch["class_labels"]):
                    # Exclude ignore index (255) and background (0)
                    valid_pixels = (m != 0) & (m != 255)
                    if valid_pixels.sum() > 0:
                        filtered_mask_labels.append(m)
                        filtered_class_labels.append(c)
                if len(filtered_mask_labels) == 0:
                    print(f"[WARNING] Skipping validation batch {idx} as all masks are empty or ignore.")
                    continue
                batch["mask_labels"] = filtered_mask_labels
                batch["class_labels"] = filtered_class_labels

                try:
                    outputs = model(
                        pixel_values=batch["pixel_values"],
                        mask_labels=[m for m in batch["mask_labels"]],
                        class_labels=[c for c in batch["class_labels"]],
                    )
                except Exception as e:
                    print(f"[ERROR] Exception in model forward at batch {idx} (val): {e}")
                    print(f"[ERROR] [VAL] Batch {idx} mask_labels: {[torch.unique(m) for m in batch['mask_labels']]}")
                    print(f"[ERROR] [VAL] Batch {idx} class_labels: {batch['class_labels']}")
                    print(f"[ERROR] [VAL] Batch {idx} pixel_values shape: {batch['pixel_values'].shape}")
                    raise
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
