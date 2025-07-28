import torch
from tqdm.auto import tqdm
import evaluate
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.models.mask2former import get_preprocessor
import numpy as np
from src.utils.palette import remap_labels, label2id
from src.utils.training_summary import log_training_summary
from PIL import Image

# log summary of training parameters
try:
    import psutil
except ImportError:
    psutil = None

def train(model, optimizer, train_dataloader, valid_dataloader, id2label_remapped, device, accelerator, epochs=2, scheduler=None):
    log_training_summary(model, optimizer, train_dataloader, id2label_remapped, device, accelerator, epochs, scheduler)

    best_val_loss = float('inf')
    best_epoch = 0
    running_loss = 0.0
    num_samples = 0

    for epoch in range(epochs):
        accelerator.print(f"Epoch: {epoch}")
        # Log learning rate at start of epoch
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            accelerator.print(f"Current learning rate: {current_lr:.6f}")
        # Training phase
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[m.to(device) for m in batch["mask_labels"]],
                class_labels=[c.to(device) for c in batch["class_labels"]],
            )
            loss = outputs.loss
            accelerator.backward(loss)
            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()
            num_samples += batch_size
            if idx % 100 == 0:
                accelerator.print("Loss:", running_loss/num_samples)
                if scheduler is not None:
                    current_lr = optimizer.param_groups[0]['lr']
                    accelerator.print(f"LR: {current_lr:.6f}")
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        # CORRECTED VALIDATION PHASE
        model.eval()
        val_losses = []
        all_predictions = []
        all_references = []
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valid_dataloader)):
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    mask_labels=[m.to(device) for m in batch["mask_labels"]],
                    class_labels=[c.to(device) for c in batch["class_labels"]],
                )
                # Collect validation loss
                val_losses.append(outputs.loss.detach())
                preprocessor = get_preprocessor(len(id2label_remapped))
                target_sizes = [(m.shape[0], m.shape[1]) for m in batch["original_segmentation_maps"]]
                processed_predictions = preprocessor.post_process_semantic_segmentation(
                    outputs, target_sizes=target_sizes
                )
                for pred, orig_mask in zip(processed_predictions, batch["original_segmentation_maps"]):
                    gt_remapped = remap_labels(np.array(orig_mask), label2id)
                    refs_np = gt_remapped.cpu().numpy().astype(np.int64)
                    pred_np = pred.cpu().numpy().astype(np.int64) if hasattr(pred, 'cpu') else np.array(pred).astype(np.int64)
                    if refs_np.shape != pred_np.shape:
                        pred_pil = Image.fromarray(pred_np.astype(np.uint8))
                        pred_pil = pred_pil.resize((refs_np.shape[1], refs_np.shape[0]), Image.NEAREST)
                        pred_np = np.array(pred_pil).astype(np.int64)
                    all_predictions.append(pred_np)
                    all_references.append(refs_np)
        # Gather validation losses (all processes get the same result)
        val_losses_tensor = torch.stack(val_losses)
        gathered_val_losses = accelerator.gather(val_losses_tensor)
        avg_val_loss = gathered_val_losses.mean().item()
        # Prepare tensors for gathering (ALL processes must do this)
        if all_predictions:
            pred_tensors = [torch.from_numpy(p) for p in all_predictions]
            ref_tensors = [torch.from_numpy(r) for r in all_references]
            max_h = max(t.shape[0] for t in pred_tensors + ref_tensors) if pred_tensors else 1
            max_w = max(t.shape[1] for t in pred_tensors + ref_tensors) if pred_tensors else 1
            def pad_tensor(t, target_h, target_w):
                pad_h = target_h - t.shape[0]
                pad_w = target_w - t.shape[1]
                return torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), value=0)
            pred_tensors_padded = [pad_tensor(t, max_h, max_w) for t in pred_tensors]
            ref_tensors_padded = [pad_tensor(t, max_h, max_w) for t in ref_tensors]
            pred_stack = torch.stack(pred_tensors_padded).to(accelerator.device)
            ref_stack = torch.stack(ref_tensors_padded).to(accelerator.device)
        else:
            pred_stack = torch.empty(0, 1, 1).to(accelerator.device)
            ref_stack = torch.empty(0, 1, 1).to(accelerator.device)
        # ALL processes must call gather (not just main process)
        gathered_preds = accelerator.gather(pred_stack)
        gathered_refs = accelerator.gather(ref_stack)
        # Only main process computes and prints metrics
        if accelerator.is_main_process:
            if gathered_preds.numel() > 0:
                gathered_preds_list = [p.cpu().numpy() for p in gathered_preds]
                gathered_refs_list = [r.cpu().numpy() for r in gathered_refs]
                try:
                    metric = evaluate.load("mean_iou")
                    metric.add_batch(
                        references=gathered_refs_list,
                        predictions=gathered_preds_list
                    )
                    metric_result = metric.compute(
                        num_labels=len(id2label_remapped), 
                        ignore_index=0,
                        reduce_labels=False
                    )
                    mean_iou = metric_result.get("mean_iou", 0.0)
                    accelerator.print(f"Mean IoU: {mean_iou:.6f}")
                except Exception as e:
                    accelerator.print(f"Metric computation error: {e}")
                    accelerator.print("Mean IoU: Could not compute")
            else:
                accelerator.print("Mean IoU: No predictions to evaluate")
        # Memory management AFTER all validation processing is complete
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        # REMOVED BROADCAST - since avg_val_loss is already the same across all processes
        accelerator.print(f"Validation Loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            if accelerator.is_main_process:
                model_save_path = f"Output/best_model_epoch_{best_epoch}.pt"
                torch.save(accelerator.unwrap_model(model).state_dict(), model_save_path)
                accelerator.print(f"Model saved at epoch {best_epoch} with validation loss: {best_val_loss:.6f}")
